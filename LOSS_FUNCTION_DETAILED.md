# 损失函数详细计算说明

本文档详细说明 `AdditionACTLossHead` 中各个损失的计算方式。

## 损失函数结构

```python
total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss) + copy_loss_weight * copy_loss
```

## 1. 语言模型损失 (lm_loss)

**目的**：让模型预测的token与标签一致

**计算方式**：
```python
# 第176行
lm_loss = (
    self.loss_fn(
        outputs["logits"],           # B × seq_len × vocab_size
        labels,                      # B × seq_len
        ignore_index=IGNORE_LABEL_ID,  # -100
        valid_mask=mask              # B × seq_len (True表示有效位置)
    ) / loss_divisor                 # 每个样本的loss除以该样本的有效token数
).sum()                              # 对所有样本求和
```

**详细步骤**：
1. `mask = (labels != IGNORE_LABEL_ID)` - 标记有效位置（忽略-100）
2. `loss_counts = mask.sum(-1)` - 每个样本的有效token数
3. `loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)` - 归一化因子
4. 使用 `self.loss_fn`（通常是 `stablemax_cross_entropy`）计算每个token的loss
5. 除以 `loss_divisor` 归一化（每个样本的平均loss）
6. 对所有样本求和

**数学公式**：
```
lm_loss = Σ_sample Σ_token [loss_fn(logits[sample, token], labels[sample, token])] / loss_counts[sample]
```

## 2. Q Halt 损失 (q_halt_loss)

**目的**：让模型学会在序列完全正确时停止（halt）

**计算方式**：
```python
# 第177行
q_halt_loss = F.binary_cross_entropy_with_logits(
    outputs["q_halt_logits"],        # B × 1 (每个样本一个logit)
    seq_is_correct.to(outputs["q_halt_logits"].dtype),  # B × 1 (True/False转换为float)
    reduction="sum"                  # 对所有样本求和
)
```

**详细步骤**：
1. `is_correct = mask & (preds == labels)` - 每个token是否正确
2. `seq_is_correct = (is_correct.sum(-1) == loss_counts)` - 整个序列是否完全正确
3. 使用二元交叉熵，让 `q_halt_logits >= 0` 当且仅当 `seq_is_correct == True`

**数学公式**：
```
q_halt_loss = Σ_sample BCE(q_halt_logits[sample], seq_is_correct[sample])
```

## 3. Copy 损失 (copy_loss)

**目的**：约束前两行（加数）保持不变

**计算方式**：
```python
# 第225-233行
# 1. 提取前两行
seq_len = inputs.shape[-1]           # 例如：44
num_cols = seq_len // 4              # 例如：11
first_two_rows_len = 2 * num_cols    # 例如：22

input_first_two_rows = inputs[:, :first_two_rows_len]  # B × 22
pred_logits_first_two_rows = outputs["logits"][:, :first_two_rows_len, :]  # B × 22 × vocab_size

# 2. 创建标签和mask
copy_labels = input_first_two_rows.long()  # B × 22
copy_mask = (copy_labels > 0)              # B × 22 (True表示非pad位置)

# 3. 计算loss
copy_loss_per_token = self.loss_fn(
    pred_logits_first_two_rows,      # B × 22 × vocab_size
    copy_labels,                     # B × 22
    ignore_index=0,                  # 忽略pad（值为0）
    valid_mask=copy_mask             # B × 22
)

# 4. 只对非pad位置求平均
copy_loss = (
    (copy_loss_per_token * copy_mask.float()).sum() / 
    copy_mask.float().sum().clamp_min(1)
)
```

**详细步骤**：
1. 提取前两行（前 `seq_len//2` 个位置）
2. 创建mask，标记非pad位置（值>0）
3. 使用与 `lm_loss` 相同的 `loss_fn` 计算每个位置的loss
4. 只对非pad位置求平均

**数学公式**：
```
copy_loss = Σ_sample Σ_token [loss_fn(logits[sample, token], inputs[sample, token]) * mask[sample, token]] / Σ_sample Σ_token mask[sample, token]
```

## 4. Q Continue 损失 (q_continue_loss)

**目的**：Q-learning的bootstrapping目标（可选，通常不使用）

**计算方式**：
```python
# 第252-254行
if "target_q_continue" in outputs:
    q_continue_loss = F.binary_cross_entropy_with_logits(
        outputs["q_continue_logits"],
        outputs["target_q_continue"],
        reduction="sum"
    )
else:
    q_continue_loss = 0
```

**注意**：在当前的配置中（`no_ACT_continue: True`），这个loss通常为0。

## 5. 总损失 (total_loss)

**计算方式**：
```python
# 第257行
total_loss = (
    lm_loss + 
    0.5 * (q_halt_loss + q_continue_loss) + 
    self.copy_loss_weight * copy_loss
)
```

**权重说明**：
- `lm_loss`: 权重 = 1.0（主要损失）
- `q_halt_loss`: 权重 = 0.5
- `q_continue_loss`: 权重 = 0.5（如果存在）
- `copy_loss`: 权重 = `copy_loss_weight`（默认1.0，可通过参数调整）

## 完整代码示例

```python
def forward(self, return_keys, **model_kwargs):
    batch = model_kwargs.get("batch", {})
    new_carry, outputs = self.model(**model_kwargs)
    labels = new_carry.current_data["labels"]
    inputs = batch.get("inputs", new_carry.current_data["inputs"])
    
    # 1. 计算mask和正确性
    mask = (labels != IGNORE_LABEL_ID)
    loss_counts = mask.sum(-1)
    loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)
    
    is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
    seq_is_correct = is_correct.sum(-1) == loss_counts
    
    # 2. 计算 lm_loss
    lm_loss = (
        self.loss_fn(
            outputs["logits"], 
            labels, 
            ignore_index=IGNORE_LABEL_ID, 
            valid_mask=mask
        ) / loss_divisor
    ).sum()
    
    # 3. 计算 q_halt_loss
    q_halt_loss = F.binary_cross_entropy_with_logits(
        outputs["q_halt_logits"],
        seq_is_correct.to(outputs["q_halt_logits"].dtype),
        reduction="sum"
    )
    
    # 4. 计算 copy_loss
    seq_len = inputs.shape[-1]
    if seq_len % 4 == 0:
        num_cols = seq_len // 4
        first_two_rows_len = 2 * num_cols
        
        input_first_two_rows = inputs[:, :first_two_rows_len]
        pred_logits_first_two_rows = outputs["logits"][:, :first_two_rows_len, :]
        
        copy_labels = input_first_two_rows.long()
        copy_mask = (copy_labels > 0)
        
        if copy_mask.any():
            copy_loss_per_token = self.loss_fn(
                pred_logits_first_two_rows,
                copy_labels,
                ignore_index=0,
                valid_mask=copy_mask
            )
            copy_loss = (
                (copy_loss_per_token * copy_mask.float()).sum() / 
                copy_mask.float().sum().clamp_min(1)
            )
        else:
            copy_loss = 0.0
    else:
        copy_loss = 0.0
    
    # 5. 计算 q_continue_loss（通常为0）
    q_continue_loss = 0
    if "target_q_continue" in outputs:
        q_continue_loss = F.binary_cross_entropy_with_logits(
            outputs["q_continue_logits"],
            outputs["target_q_continue"],
            reduction="sum"
        )
    
    # 6. 组合总损失
    total_loss = (
        lm_loss + 
        0.5 * (q_halt_loss + q_continue_loss) + 
        self.copy_loss_weight * copy_loss
    )
    
    return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()
```

## 损失函数类型

### stablemax_cross_entropy

**特点**：数值稳定的交叉熵损失

**实现**：
```python
def stablemax_cross_entropy(logits, labels, ignore_index=-100, valid_mask=None):
    # 1. 计算稳定的log概率
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)
    
    # 2. 创建valid_mask
    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    
    # 3. 获取预测位置的log概率
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(
        logprobs, 
        index=transformed_labels.to(torch.long).unsqueeze(-1), 
        dim=-1
    ).squeeze(-1)
    
    # 4. 返回负log概率（只在有效位置）
    return -torch.where(valid_mask, prediction_logprobs, 0)
```

### softmax_cross_entropy

**特点**：标准的交叉熵损失（使用PyTorch的F.cross_entropy）

**实现**：
```python
def softmax_cross_entropy(logits, labels, ignore_index=-100):
    return F.cross_entropy(
        logits.to(torch.float32).view(-1, logits.shape[-1]),
        labels.to(torch.long).view(-1),
        ignore_index=ignore_index,
        reduction="none"
    ).view(labels.shape)
```

## 归一化说明

### 在损失函数中

- `lm_loss`: 除以 `loss_divisor`（每个样本的有效token数）
- `copy_loss`: 除以非pad位置的数量
- `q_halt_loss`: 直接求和，不归一化（因为每个样本只有一个值）

### 在 train_batch 中（第460行）

所有loss在记录到wandb之前，会再次除以 `global_batch_size`：

```python
reduced_metrics = {
    f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) 
    for k, v in reduced_metrics.items()
}
```

这样确保：
- Loss值在不同batch size下可比较
- 其他指标（accuracy等）除以有效样本数

## 总结

1. **lm_loss**: 主要损失，让模型预测正确
2. **q_halt_loss**: 让模型学会在正确时停止
3. **copy_loss**: 约束前两行保持不变（加法特有）
4. **q_continue_loss**: 通常不使用

总损失 = lm_loss + 0.5 * (q_halt_loss + q_continue_loss) + copy_loss_weight * copy_loss

