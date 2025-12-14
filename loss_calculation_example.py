#!/usr/bin/env python3
"""
损失函数计算示例代码

展示 AdditionACTLossHead 中各个损失的具体计算过程
"""

import torch
import torch.nn.functional as F

# 模拟数据
batch_size = 2
seq_len = 44  # 4行 × 11列
vocab_size = 11  # 0-10 (0是pad, 1-10对应数字0-9)

# 模拟模型输出
logits = torch.randn(batch_size, seq_len, vocab_size)  # B × seq_len × vocab_size
q_halt_logits = torch.randn(batch_size, 1)  # B × 1

# 模拟标签（值+1格式，0是pad）
labels = torch.randint(0, 11, (batch_size, seq_len))  # B × seq_len
labels[:, -10:] = 0  # 后面一些位置是pad

# 模拟输入（前两行是加数，应该保持不变）
inputs = labels.clone()  # 简化：假设输入和标签形状相同

# 模拟mask
IGNORE_LABEL_ID = -100
mask = (labels != 0)  # 简化：假设0是pad（实际应该是-100）
loss_counts = mask.sum(-1)  # [有效token数1, 有效token数2]
loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # B × 1

print("=" * 60)
print("损失函数计算示例")
print("=" * 60)

# ========== 1. 计算 lm_loss ==========
print("\n1. 语言模型损失 (lm_loss):")
print("-" * 60)

# 使用标准交叉熵（简化示例）
lm_loss_per_token = F.cross_entropy(
    logits.view(-1, vocab_size),
    labels.view(-1),
    ignore_index=0,
    reduction="none"
).view(batch_size, seq_len)

# 只对有效位置计算，然后归一化
lm_loss_per_sample = (lm_loss_per_token * mask.float()).sum(-1) / loss_counts.float()
lm_loss = lm_loss_per_sample.sum()

print(f"  logits形状: {logits.shape}")
print(f"  labels形状: {labels.shape}")
print(f"  mask形状: {mask.shape}")
print(f"  每个样本的有效token数: {loss_counts.tolist()}")
print(f"  每个样本的平均loss: {lm_loss_per_sample.tolist()}")
print(f"  lm_loss (总和): {lm_loss.item():.6f}")

# ========== 2. 计算 q_halt_loss ==========
print("\n2. Q Halt 损失 (q_halt_loss):")
print("-" * 60)

# 计算每个样本是否完全正确
preds = torch.argmax(logits, dim=-1)  # B × seq_len
is_correct = mask & (preds == labels)  # B × seq_len
seq_is_correct = (is_correct.sum(-1) == loss_counts)  # B (True/False)

# 转换为float用于loss计算
seq_is_correct_float = seq_is_correct.float()  # B

# 计算二元交叉熵
q_halt_loss = F.binary_cross_entropy_with_logits(
    q_halt_logits.squeeze(-1),  # B
    seq_is_correct_float,       # B
    reduction="sum"
)

print(f"  q_halt_logits形状: {q_halt_logits.shape}")
print(f"  每个样本是否完全正确: {seq_is_correct.tolist()}")
print(f"  q_halt_loss (总和): {q_halt_loss.item():.6f}")

# ========== 3. 计算 copy_loss ==========
print("\n3. Copy 损失 (copy_loss):")
print("-" * 60)

seq_len = inputs.shape[-1]
num_cols = seq_len // 4  # 11
first_two_rows_len = 2 * num_cols  # 22

# 提取前两行
input_first_two_rows = inputs[:, :first_two_rows_len]  # B × 22
pred_logits_first_two_rows = logits[:, :first_two_rows_len, :]  # B × 22 × vocab_size

# 创建copy标签和mask
copy_labels = input_first_two_rows.long()  # B × 22
copy_mask = (copy_labels > 0)  # B × 22 (非pad位置)

# 计算copy loss
copy_loss_per_token = F.cross_entropy(
    pred_logits_first_two_rows.reshape(-1, vocab_size),
    copy_labels.reshape(-1),
    ignore_index=0,
    reduction="none"
).reshape(batch_size, first_two_rows_len)

# 只对非pad位置求平均
copy_loss = (
    (copy_loss_per_token * copy_mask.float()).sum() / 
    copy_mask.float().sum().clamp_min(1)
)

print(f"  序列长度: {seq_len}")
print(f"  网格列数: {num_cols}")
print(f"  前两行长度: {first_two_rows_len}")
print(f"  input_first_two_rows形状: {input_first_two_rows.shape}")
print(f"  pred_logits_first_two_rows形状: {pred_logits_first_two_rows.shape}")
print(f"  copy_mask中非pad位置数: {copy_mask.sum().item()}")
print(f"  copy_loss: {copy_loss.item():.6f}")

# ========== 4. 计算总损失 ==========
print("\n4. 总损失 (total_loss):")
print("-" * 60)

copy_loss_weight = 1.0  # 默认权重
q_continue_loss = 0.0  # 通常不使用

total_loss = (
    lm_loss + 
    0.5 * (q_halt_loss + q_continue_loss) + 
    copy_loss_weight * copy_loss
)

print(f"  lm_loss: {lm_loss.item():.6f}")
print(f"  q_halt_loss: {q_halt_loss.item():.6f}")
print(f"  q_continue_loss: {q_continue_loss:.6f}")
print(f"  copy_loss: {copy_loss.item():.6f}")
print(f"  copy_loss_weight: {copy_loss_weight}")
print(f"\n  总损失公式:")
print(f"    total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss) + copy_loss_weight * copy_loss")
print(f"    total_loss = {lm_loss.item():.6f} + 0.5 * ({q_halt_loss.item():.6f} + {q_continue_loss:.6f}) + {copy_loss_weight} * {copy_loss.item():.6f}")
print(f"    total_loss = {total_loss.item():.6f}")

# ========== 5. 归一化到wandb ==========
print("\n5. 归一化到wandb (在train_batch中):")
print("-" * 60)

global_batch_size = 64  # 全局batch size
count = 2  # 有效样本数（简化）

# 在train_batch中，所有loss会除以global_batch_size
wandb_metrics = {
    "train/lm_loss": lm_loss.item() / global_batch_size,
    "train/q_halt_loss": q_halt_loss.item() / global_batch_size,
    "train/copy_loss": copy_loss.item() / global_batch_size,
    "train/total_loss": total_loss.item() / global_batch_size,
}

print(f"  归一化前:")
print(f"    lm_loss: {lm_loss.item():.6f}")
print(f"    q_halt_loss: {q_halt_loss.item():.6f}")
print(f"    copy_loss: {copy_loss.item():.6f}")
print(f"    total_loss: {total_loss.item():.6f}")
print(f"\n  归一化后 (除以 global_batch_size={global_batch_size}):")
for k, v in wandb_metrics.items():
    print(f"    {k}: {v:.6f}")

print("\n" + "=" * 60)
print("总结:")
print("  - lm_loss: 主要损失，让模型预测正确")
print("  - q_halt_loss: 让模型学会在正确时停止")
print("  - copy_loss: 约束前两行保持不变（加法特有）")
print("  - 总损失 = lm_loss + 0.5 * (q_halt_loss + q_continue_loss) + copy_loss_weight * copy_loss")
print("=" * 60)

