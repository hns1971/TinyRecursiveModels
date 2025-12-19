# 推理指南

本指南介绍如何使用训练好的checkpoint进行推理。

## 快速开始

### 基本用法

```bash
python inference.py \
  --checkpoint checkpoints/Arc1concept-aug-1000-ACT-torch/my_training_run/step_606 \
  --data_paths data/arc1concept-aug-1000
```

### 完整参数示例

```bash
python inference.py \
  --checkpoint checkpoints/Arc1concept-aug-1000-ACT-torch/my_training_run/step_303191 \
  --data_paths data/arc1concept-aug-1000 \
  --max_batches 1 \
  --save_outputs preds inputs puzzle_identifiers
```

## 参数说明

### 必需参数

- `--checkpoint`: Checkpoint文件路径
  - 示例: `checkpoints/Arc1concept-aug-1000-ACT-torch/my_training_run/step_606`
  - 注意: 这是checkpoint文件路径，不是目录

### 可选参数

- `--data_paths`: 数据集路径列表（默认: `["data/arc1concept-aug-1000"]`）
  - 可以指定多个数据集路径
  - 示例: `--data_paths data/arc1concept-aug-1000 data/arc2concept-aug-1000`

- `--max_batches`: 最大评估batch数量（默认: 全部）
  - 用于快速测试，限制处理的batch数量
  - 示例: `--max_batches 10` 只处理前10个batch

- `--save_outputs`: 要保存的输出键（默认: `["preds", "inputs"]`）
  - 可选项: `preds`, `inputs`, `puzzle_identifiers`, `q_halt_logits` 等
  - 保存的预测结果会在checkpoint目录下: `step_{step}_all_preds.0`

- `--config_path`: 配置文件目录（默认: `config`）

- `--config_name`: 配置文件名（默认: `cfg_pretrain`）

## 推理流程

推理脚本会执行以下步骤：

1. **加载配置**: 从配置文件加载模型和评估器配置
2. **加载checkpoint**: 加载指定checkpoint的模型权重
3. **创建数据加载器**: 使用test模式加载测试数据
4. **运行推理**: 
   - 对每个batch进行前向传播
   - 使用ACT机制进行多步推理（直到所有序列halt）
   - 收集预测结果和指标
5. **评估**: 如果配置了评估器（如ARC），会计算评估指标
6. **保存结果**: 保存预测结果和评估指标

## 输出说明

### 控制台输出

推理过程会显示：
- 推理配置信息
- 每个batch的处理进度
- 每个batch的推理步数
- 最终的评估指标（如果配置了评估器）

### 保存的文件

1. **预测结果**: `{checkpoint_dir}/step_{step}_all_preds.0`
   - 包含模型的所有预测输出
   - 格式: PyTorch tensor字典

2. **评估结果**: `{checkpoint_dir}/evaluator_ARC_step_{step}/submission.json`
   - ARC评估器的提交文件
   - 格式: JSON，包含每个puzzle的预测

## 示例输出

```
============================================================
推理配置
============================================================
Checkpoint: checkpoints/Arc1concept-aug-1000-ACT-torch/my_training_run/step_606
数据集: ['data/arc1concept-aug-1000']
最大batch数: 全部
保存输出: ['preds', 'inputs']
============================================================

开始推理...
Processing batch 1: all
  Completed inference in 16 steps
Processing batch 2: all
  Completed inference in 16 steps
...

Running 1 evaluator(s)...
Running evaluator 1/1: ARC

============================================================
推理结果
============================================================

all:
  accuracy: 0.779153
  exact_accuracy: 0.188750
  q_halt_accuracy: 0.728594
  steps: 16.000000
  lm_loss: 1.043360
  q_halt_loss: 1.014530

ARC/pass@1: 0.005000
ARC/pass@2: 0.005000
ARC/pass@5: 0.005000
ARC/pass@10: 0.005000
ARC/pass@100: 0.005000
ARC/pass@1000: 0.007500
============================================================
```

## 代码位置

推理的核心代码在 `pretrain.py` 的 `evaluate` 函数中：

```python
# 推理循环
while True:
    carry, loss, metrics, preds, all_finish = train_state.model(
        carry=carry, batch=batch, return_keys=return_keys
    )
    inference_steps += 1
    
    if all_finish:  # 所有序列都halt了
        break
```

## 注意事项

1. **GPU内存**: 确保有足够的GPU内存加载模型和进行推理
2. **评估模式**: 推理时模型会自动设置为 `eval()` 模式
3. **ACT机制**: 评估时强制使用最大步数（`halt_max_steps`），与训练时的动态halt不同
4. **分布式**: 当前脚本只支持单GPU推理，如需多GPU请修改代码

## 故障排除

### 问题1: Checkpoint加载失败

```
错误: 找不到checkpoint文件
解决: 检查checkpoint路径是否正确，确保文件存在
```

### 问题2: 数据集路径错误

```
错误: 找不到数据集
解决: 检查data_paths参数，确保数据集目录存在
```

### 问题3: CUDA内存不足

```
错误: CUDA out of memory
解决: 
- 减小batch size（修改配置文件）
- 使用--max_batches限制处理的batch数量
- 使用更小的模型配置
```

## 与训练时的评估对比

训练时也会进行评估（在 `pretrain.py` 中），但推理脚本的优势：

1. **独立运行**: 不需要重新训练，直接使用已有checkpoint
2. **灵活配置**: 可以指定不同的数据集和评估参数
3. **结果保存**: 更方便地保存和查看预测结果

