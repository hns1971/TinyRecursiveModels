#!/usr/bin/env python3
"""
单独调用一次forward的调试程序（使用 evaluate_single_batch）

使用方法:
    # 从命令行参数输入4行数据
    python debug_single_forward.py \
        --checkpoint checkpoints/... \
        --row1 "10 1 2 3" \
        --row2 "10 4 5 6" \
        --row3 "10 0 0 0" \
        --row4 "10 0 0 0" \
        --max-len 11
    
    # 从文件读取4行数据
    python debug_single_forward.py \
        --checkpoint checkpoints/... \
        --input-file input_grid.txt \
        --max-len 11
    
    # 交互式输入
    python debug_single_forward.py \
        --checkpoint checkpoints/... \
        --max-len 11 \
        --interactive
"""

import os
import json
import numpy as np
import argparse
import sys
import torch
import torch.distributed as dist
from typing import Any, Tuple, Dict

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pretrain import evaluate_single_batch
from inference import create_model_from_checkpoint, create_train_state_from_checkpoint

# 前导词：使用10表示前导位置和未计算位置
LEADING_VALUE = 10
# PAD值：使用11表示PAD，用于batch填充
PAD_VALUE = 11


def parse_grid_from_strings(row1_str, row2_str, row3_str, row4_str, max_len):
    """从字符串解析4行数据"""
    def parse_row(row_str):
        # 支持空格或逗号分隔
        row_str = row_str.replace(',', ' ')
        values = [int(x.strip()) for x in row_str.split() if x.strip()]
        return values
    
    row1 = parse_row(row1_str)
    row2 = parse_row(row2_str)
    row3 = parse_row(row3_str)
    row4 = parse_row(row4_str)
    
    # 确保每行长度一致，不足的用LEADING_VALUE补齐
    max_row_len = max(len(row1), len(row2), len(row3), len(row4))
    if max_row_len > max_len:
        print(f"警告：行长度 {max_row_len} 超过 max_len {max_len}，将截断")
        max_row_len = max_len
    
    # 补齐到max_len
    def pad_row(row, target_len):
        if len(row) < target_len:
            return row + [LEADING_VALUE] * (target_len - len(row))
        elif len(row) > target_len:
            return row[:target_len]
        return row
    
    row1 = pad_row(row1, max_len)
    row2 = pad_row(row2, max_len)
    row3 = pad_row(row3, max_len)
    row4 = pad_row(row4, max_len)
    
    grid = np.array([row1, row2, row3, row4], dtype=np.uint8)
    return grid


def parse_grid_from_file(file_path, max_len):
    """从文件读取4行数据"""
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    if len(lines) < 4:
        raise ValueError(f"文件至少需要4行数据，但只有 {len(lines)} 行")
    
    return parse_grid_from_strings(lines[0], lines[1], lines[2], lines[3], max_len)


def interactive_input(max_len):
    """交互式输入4行数据"""
    print("请输入4行数据（每行用空格或逗号分隔数字）：")
    print(f"注意：值范围 0-9 为数字，10 为前导值，11 为PAD值")
    print()
    
    row1_str = input("第1行（第一个加数）: ").strip()
    row2_str = input("第2行（第二个加数）: ").strip()
    row3_str = input("第3行（进位数）: ").strip()
    row4_str = input("第4行（结果）: ").strip()
    
    return parse_grid_from_strings(row1_str, row2_str, row3_str, row4_str, max_len)


def visualize_grid(grid, title="网格"):
    """可视化4行网格"""
    print(f"\n{title}:")
    print("=" * 60)
    for i, row in enumerate(grid):
        row_str = " ".join([f"{val:2d}" for val in row])
        row_name = ["第1行（加数1）", "第2行（加数2）", "第3行（进位）", "第4行（结果）"][i]
        print(f"{row_name}: {row_str}")
    print("=" * 60)


def print_tensor_info(name, tensor, max_elements=50):
    """打印tensor的详细信息"""
    print(f"\n{name}:")
    print(f"  shape: {tensor.shape}")
    print(f"  dtype: {tensor.dtype}")
    print(f"  device: {tensor.device}")
    
    # 处理不支持的类型（如bfloat16），转换为float32
    if tensor.dtype == torch.bfloat16:
        tensor_for_numpy = tensor.cpu().float()
    else:
        tensor_for_numpy = tensor.cpu()
    
    if tensor.numel() <= max_elements:
        print(f"  values: {tensor_for_numpy.numpy()}")
    else:
        flat = tensor_for_numpy.flatten().numpy()
        print(f"  first 20 values: {flat[:20]}")
        print(f"  last 20 values: {flat[-20:]}")


def print_carry_info(carry):
    """打印carry的详细信息"""
    print("\n" + "=" * 80)
    print("Carry 状态:")
    print("=" * 80)
    
    if hasattr(carry, 'halted'):
        halted_np = carry.halted.cpu()
        if halted_np.dtype == torch.bfloat16:
            halted_np = halted_np.float()
        print(f"halted: {halted_np.numpy()}")
    
    if hasattr(carry, 'steps'):
        steps_np = carry.steps.cpu()
        if steps_np.dtype == torch.bfloat16:
            steps_np = steps_np.float()
        print(f"steps: {steps_np.numpy()}")
    
    if hasattr(carry, 'current_data') and carry.current_data:
        print("\ncurrent_data:")
        for k, v in carry.current_data.items():
            if isinstance(v, torch.Tensor):
                print_tensor_info(f"  {k}", v)
            else:
                print(f"  {k}: {type(v)}")
    
    if hasattr(carry, 'inner_carry'):
        print("\ninner_carry: (已设置，详细信息略)")


def forward_once(
    model,
    grid: np.ndarray,
    max_len: int,
    model_seq_len: int,
    puzzle_id_value: int = 0,
    carry: Any = None,
    training_step: int = 0,
    warmup_steps: int = 1000,
    verbose: bool = True,
):
    """
    根据给定的4行网格输入，执行一次forward推理（使用 evaluate_single_batch）
    
    Args:
        model: 模型实例
        grid: 4行n列的输入网格（numpy数组，dtype=uint8）
        max_len: 网格列数
        model_seq_len: 模型期望的序列长度
        puzzle_id_value: Puzzle identifier值
        carry: 可选的carry状态（如果为None，则初始化新的carry）
        training_step: 训练步数（用于动态loss权重）
        warmup_steps: warmup步数（用于动态loss权重）
        verbose: 是否打印详细信息
    
    Returns:
        tuple: (new_carry, metrics, extracted_metrics, batch, pred_grid)
            - new_carry: 新的carry状态
            - metrics: 模型返回的完整metrics字典
            - extracted_metrics: 提取的指标字典
            - batch: 使用的batch
            - pred_grid: 预测结果的网格（4行n列）
    """
    # 准备输入序列
    input_grid = grid.copy()
    input_seq = (input_grid.flatten() + 1).astype(np.int32)  # 值+1（0-9变成1-10，10变成11，11变成12）
    
    # 填充到模型期望的seq_len
    if len(input_seq) < model_seq_len:
        input_seq = np.pad(input_seq, (0, model_seq_len - len(input_seq)), constant_values=PAD_VALUE + 1)
    elif len(input_seq) > model_seq_len:
        input_seq = input_seq[:model_seq_len]
    
    if verbose:
        print(f"\n输入序列长度: {len(input_seq)} (网格: {4 * max_len}, 模型期望: {model_seq_len})")
    
    # 创建batch（推理阶段，labels为空）
    batch = {
        "inputs": torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).cuda(),
        # 推理阶段：labels 为空，表示不需要计算loss和metrics
        "labels": None,
        "puzzle_identifiers": torch.tensor([puzzle_id_value], dtype=torch.long).cuda(),
    }
    
    if verbose:
        print("\n" + "=" * 80)
        print("Batch 信息")
        print("=" * 80)
        for k, v in batch.items():
            if v is not None:
                print_tensor_info(k, v)
            else:
                print(f"{k}: None (推理阶段，不计算loss和metrics)")
    
    # 使用 evaluate_single_batch 进行单步推理
    if verbose:
        print("\n" + "=" * 80)
        print("调用 evaluate_single_batch 进行单步推理")
        print("=" * 80)
    
    new_carry, metrics, extracted_metrics = evaluate_single_batch(
        model=model,
        batch=batch,
        carry=carry,
        training_step=training_step,
        warmup_steps=warmup_steps,
        enable_debug=verbose,
        rank=0,
    )
    
    # 处理输出，提取预测网格
    pred_grid = None
    if "preds" in metrics:
        preds = metrics["preds"]
        # 转换为网格显示
        pred_seq_np = preds[0].cpu()
        if pred_seq_np.dtype == torch.bfloat16:
            pred_seq_np = pred_seq_np.float()
        pred_seq = pred_seq_np.numpy() - 1  # 减去1恢复原始值
        four_rows_size = 4 * max_len
        if len(pred_seq) >= four_rows_size:
            pred_grid = pred_seq[:four_rows_size].reshape(4, max_len)
        else:
            padded = np.pad(pred_seq, (0, four_rows_size - len(pred_seq)), constant_values=LEADING_VALUE)
            pred_grid = padded.reshape(4, max_len)
    elif "logits" in metrics:
        # 如果没有preds，从logits计算
        logits = metrics["logits"]
        preds = logits.argmax(dim=-1)
        pred_seq_np = preds[0].cpu()
        if pred_seq_np.dtype == torch.bfloat16:
            pred_seq_np = pred_seq_np.float()
        pred_seq = pred_seq_np.numpy() - 1
        four_rows_size = 4 * max_len
        if len(pred_seq) >= four_rows_size:
            pred_grid = pred_seq[:four_rows_size].reshape(4, max_len)
        else:
            padded = np.pad(pred_seq, (0, four_rows_size - len(pred_seq)), constant_values=LEADING_VALUE)
            pred_grid = padded.reshape(4, max_len)
    
    if verbose:
        # 显示输出
        print("\n" + "=" * 80)
        print("推理输出")
        print("=" * 80)
        
        # Metrics
        print("\nMetrics:")
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    print(f"  {k}: {v.item()}")
                else:
                    print_tensor_info(f"  {k}", v)
            else:
                print(f"  {k}: {v}")
        
        # Extracted metrics
        if extracted_metrics:
            print("\nExtracted Metrics:")
            for k, v in extracted_metrics.items():
                print(f"  {k}: {v}")
        
        # 预测网格
        if pred_grid is not None:
            visualize_grid(pred_grid, "输出网格（预测结果）")
        
        # 显示新的carry状态
        print_carry_info(new_carry)
    
    return new_carry, metrics, extracted_metrics, batch, pred_grid


def main():
    parser = argparse.ArgumentParser(
        description="单独调用一次forward的调试程序（使用 evaluate_single_batch）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从命令行参数输入
  python debug_single_forward.py \\
      --checkpoint checkpoints/... \\
      --row1 "10 1 2 3" \\
      --row2 "10 4 5 6" \\
      --row3 "10 0 0 0" \\
      --row4 "10 0 0 0" \\
      --max-len 11
  
  # 从文件读取
  python debug_single_forward.py \\
      --checkpoint checkpoints/... \\
      --input-file input_grid.txt \\
      --max-len 11
  
  # 交互式输入
  python debug_single_forward.py \\
      --checkpoint checkpoints/... \\
      --max-len 11 \\
      --interactive
        """
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint文件路径"
    )
    parser.add_argument(
        "--max-len",
        type=int,
        required=True,
        help="网格列数（数据宽度）"
    )
    parser.add_argument(
        "--row1",
        type=str,
        default=None,
        help="第1行数据（空格或逗号分隔）"
    )
    parser.add_argument(
        "--row2",
        type=str,
        default=None,
        help="第2行数据（空格或逗号分隔）"
    )
    parser.add_argument(
        "--row3",
        type=str,
        default=None,
        help="第3行数据（空格或逗号分隔）"
    )
    parser.add_argument(
        "--row4",
        type=str,
        default=None,
        help="第4行数据（空格或逗号分隔）"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="从文件读取4行数据（每行一个）"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="交互式输入4行数据"
    )
    parser.add_argument(
        "--puzzle-id",
        type=int,
        default=None,
        help="可选：指定 puzzle identifier；不指定则禁用 puzzle_emb（puzzle_emb_len=0）"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="config",
        help="配置文件目录（默认：config）"
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="cfg_finetune_addition",
        help="配置文件名（默认：cfg_finetune_addition）"
    )
    parser.add_argument(
        "--data-paths",
        type=str,
        nargs="+",
        default=["data/addition"],
        help="数据集路径列表（默认：['data/addition']）"
    )
    parser.add_argument(
        "--training-step",
        type=int,
        default=0,
        help="训练步数（用于动态loss权重，默认：0）"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
        help="Warmup步数（用于动态loss权重，默认：1000）"
    )
    
    args = parser.parse_args()
    
    # 确定输入方式
    if args.interactive:
        grid = interactive_input(args.max_len)
    elif args.input_file:
        grid = parse_grid_from_file(args.input_file, args.max_len)
    elif args.row1 and args.row2 and args.row3 and args.row4:
        grid = parse_grid_from_strings(args.row1, args.row2, args.row3, args.row4, args.max_len)
    else:
        print("错误：必须提供输入数据（--row1/2/3/4, --input-file, 或 --interactive）")
        return
    
    # 显示输入的网格
    visualize_grid(grid, "输入网格")
    
    # 使用 inference 中的函数创建模型
    print("\n" + "=" * 80)
    print("加载模型和配置")
    print("=" * 80)
    model, config, metadata = create_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        data_paths=args.data_paths,
        config_path=args.config_path,
        config_name=args.config_name,
        rank=0,
        world_size=1,
        auto_detect_task=True,
    )
    
    # 获取模型序列长度
    model_seq_len = metadata.seq_len
    
    # 创建 train_state（使用 inference 中的函数）
    train_state = create_train_state_from_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        step=None,  # 从checkpoint文件名中自动提取
    )
    
    print("\n" + "=" * 80)
    print("模型信息")
    print("=" * 80)
    print(f"模型序列长度: {model_seq_len}")
    print(f"网格列数: {args.max_len}")
    print(f"Puzzle ID: {args.puzzle_id or 0}")
    print(f"训练步数: {train_state.step}")
    print("=" * 80)
    
    # 执行一次forward（使用 evaluate_single_batch）
    new_carry, metrics, extracted_metrics, batch, pred_grid = forward_once(
        model=model,
        grid=grid,
        max_len=args.max_len,
        model_seq_len=model_seq_len,
        puzzle_id_value=args.puzzle_id or 0,
        carry=None,  # 使用新的carry
        training_step=args.training_step,
        warmup_steps=args.warmup_steps,
        verbose=True,
    )
    
    print("\n" + "=" * 80)
    print("调试完成")
    print("=" * 80)
    
    # 清理
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
