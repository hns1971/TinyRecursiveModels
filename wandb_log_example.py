#!/usr/bin/env python3
"""
展示 wandb.log 实际记录的内容示例

这个脚本展示了训练过程中 wandb.log 实际记录的 metrics 字典结构
"""

# 示例：AdditionACTLossHead 返回的 metrics（在 forward 方法中）
metrics_from_loss_head = {
    "count": 64,  # 有效样本数（halted & loss_counts > 0）
    "accuracy": 0.85,  # 平均准确率
    "exact_accuracy": 45,  # 完全匹配的样本数
    "q_halt_accuracy": 50,  # Q halt 预测准确的样本数
    "steps": 320,  # 总步数（所有样本的步数之和）
    "lm_loss": 0.5,  # 语言模型损失（未归一化）
    "q_halt_loss": 0.3,  # Q halt 损失（未归一化）
    "copy_loss": 0.1,  # Copy 损失（未归一化）
    "copy_accuracy": 0.95,  # Copy 准确率
    "total_loss": 0.9,  # 总损失（未归一化）
    # "q_continue_loss": 0.2,  # Q continue 损失（如果有）
}

# 在 train_batch 函数中处理后的 metrics（第460行）
global_batch_size = 64
count = metrics_from_loss_head["count"]

# 处理逻辑：
# 1. 所有 loss 类型的指标除以 global_batch_size
# 2. 其他指标除以 count
# 3. 添加 "train/" 前缀
# 4. 添加 "train/lr"

processed_metrics = {}
for k, v in metrics_from_loss_head.items():
    if k.endswith("loss"):
        # Loss 类型：除以 global_batch_size
        processed_metrics[f"train/{k}"] = v / global_batch_size
    else:
        # 其他指标：除以 count
        processed_metrics[f"train/{k}"] = v / count

# 添加学习率
processed_metrics["train/lr"] = 5e-5

print("=" * 60)
print("wandb.log() 实际记录的内容示例")
print("=" * 60)
print("\n1. 从损失函数返回的原始 metrics:")
print("-" * 60)
for k, v in metrics_from_loss_head.items():
    print(f"  {k:20s}: {v}")

print("\n2. 在 train_batch 中处理后的 metrics（实际记录到 wandb）:")
print("-" * 60)
for k in sorted(processed_metrics.keys()):
    v = processed_metrics[k]
    print(f"  {k:25s}: {v:.6f}")

print("\n3. wandb.log() 调用:")
print("-" * 60)
print(f"  wandb.log({processed_metrics}, step=train_state.step)")
print(f"  # step 是当前的训练步数，例如: step=1000")

print("\n4. 在 WandB 界面中显示的指标:")
print("-" * 60)
print("  所有指标都会以 'train/' 为前缀显示在 WandB 中")
print("  例如：")
for k in sorted(processed_metrics.keys())[:5]:  # 只显示前5个
    print(f"    - {k}")

print("\n5. 指标说明:")
print("-" * 60)
print("  train/total_loss      : 总损失（归一化后）")
print("  train/lm_loss        : 语言模型损失（归一化后）")
print("  train/q_halt_loss     : Q halt 损失（归一化后）")
print("  train/copy_loss       : Copy 损失（归一化后）")
print("  train/copy_accuracy   : Copy 准确率（前两行保持不变的比例）")
print("  train/accuracy        : 平均准确率")
print("  train/exact_accuracy  : 完全匹配准确率（所有token都正确）")
print("  train/q_halt_accuracy : Q halt 预测准确率")
print("  train/steps           : 平均推理步数")
print("  train/count           : 有效样本数（用于归一化）")
print("  train/lr              : 当前学习率")

print("\n" + "=" * 60)
print("注意：")
print("  - 所有 loss 类型的指标都会除以 global_batch_size")
print("  - 其他指标会除以 count（有效样本数）")
print("  - 这样确保指标在不同 batch size 下可比较")
print("=" * 60)

