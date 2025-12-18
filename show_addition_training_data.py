#!/usr/bin/env python3
"""
显示加法训练数据的一条样本

使用方法:
    python show_addition_training_data.py [--index INDEX]
"""

import os
import json
import numpy as np
import argparse


def visualize_grid(grid: np.ndarray, title: str = "", pad_value: int = 10, leading_value: int = 10):
    """可视化4行n列的网格
    
    Args:
        grid: 4行n列的网格，PAD值用pad_value（10）表示
        title: 标题
        pad_value: PAD值（默认10）
    """
    print(f"\n{title}:")
    print("-" * 60)
    
    # 将前导词（10）显示为"F"，PAD值（11）显示为"P"，数字转换为int
    def format_row(row):
        result = []
        for val in row:
            if val == leading_value:
                result.append("F")  # LEADING_VALUE显示为F
            elif val == pad_value:
                result.append("P")  # PAD_VALUE显示为P
            else:
                # 转换为Python int，避免显示np.uint8(3)这样的格式
                result.append(int(val))
        return result
    
    # 格式化输出，去掉单引号
    def format_row_display(row):
        formatted = format_row(row)
        return "[" + ", ".join(str(x) for x in formatted) + "]"
    
    print(f"第1行（加数1）: {format_row_display(grid[0])}")
    print(f"第2行（加数2）: {format_row_display(grid[1])}")
    print(f"第3行（进位）: {format_row_display(grid[2])}")
    print(f"第4行（结果）: {format_row_display(grid[3])}")
    print("-" * 60)


def extract_number_from_row(row: np.ndarray, pad_value: int = 10) -> int:
    """从一行中提取数字（去掉前导PAD和前导零）
    
    Args:
        row: 一行数据，PAD值用pad_value（10）表示
        pad_value: PAD值（默认10）
    """
    start_idx = 0
    # 跳过前导PAD和前导零
    while start_idx < len(row) and (row[start_idx] == pad_value or row[start_idx] == 0):
        start_idx += 1
    if start_idx < len(row):
        # 只提取非PAD的数字
        digits = [str(int(val)) for val in row[start_idx:] if val != pad_value]
        if digits:
            return int(''.join(digits))
    return 0


def main():
    parser = argparse.ArgumentParser(description="显示加法训练数据")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/addition/train",
        help="数据集目录（默认：data/addition/train）"
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="要显示的样本索引（默认：0）"
    )
    parser.add_argument(
        "--grid-width",
        type=int,
        default=None,
        help="网格宽度（列数），如果未指定，从metadata的seq_len计算（seq_len // 4）"
    )
    
    parser.add_argument(
        "--show-puzzle",
        action="store_true",
        help="显示同一个puzzle的所有状态转移对（如果指定了puzzle_id）"
    )
    
    parser.add_argument(
        "--puzzle-id",
        type=int,
        default=None,
        help="要显示的puzzle ID（如果指定，会显示该puzzle的所有状态转移对）"
    )
    
    args = parser.parse_args()
    
    # 加载元数据
    metadata_path = os.path.join(args.data_dir, "dataset.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"找不到metadata文件: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print("=" * 60)
    print("数据集元数据")
    print("=" * 60)
    print(f"序列长度（单个网格大小）: {metadata['seq_len']}")
    print(f"词汇表大小: {metadata['vocab_size']}")
    print(f"总组数: {metadata['total_groups']}")
    print(f"总puzzle数: {metadata['total_puzzles']}")
    print(f"平均每个puzzle的状态转移数: {metadata['mean_puzzle_examples']:.2f}")
    print()
    
    # 从metadata计算grid_width（如果未指定）
    if args.grid_width is None:
        args.grid_width = metadata['seq_len'] // 4
        print(f"从metadata计算网格宽度: {args.grid_width}列")
    print()
    
    # 加载数据
    inputs_path = os.path.join(args.data_dir, "all__inputs.npy")
    labels_path = os.path.join(args.data_dir, "all__labels.npy")
    puzzle_ids_path = os.path.join(args.data_dir, "all__puzzle_identifiers.npy")
    
    if not os.path.exists(inputs_path):
        raise FileNotFoundError(f"找不到输入文件: {inputs_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"找不到标签文件: {labels_path}")
    
    inputs = np.load(inputs_path)
    labels = np.load(labels_path)
    puzzle_identifiers = np.load(puzzle_ids_path) if os.path.exists(puzzle_ids_path) else None
    
    print(f"加载的数据形状:")
    print(f"  inputs: {inputs.shape}")
    print(f"  labels: {labels.shape}")
    if puzzle_identifiers is not None:
        print(f"  puzzle_identifiers: {puzzle_identifiers.shape}")
    print()
    
    # 前导词（LEADING_VALUE）是10（值+1后是11，减去1后是10）
    # PAD值是11（值+1后是12，减去1后是11）
    LEADING_VALUE = 10
    PAD_VALUE = 11
    
    # 使用传入的网格宽度参数
    grid_width = args.grid_width
    grid_size = 4 * grid_width  # 单个网格的大小（4行×grid_width列）
    
    # 如果指定了puzzle_id，显示该puzzle的所有状态转移对
    if args.puzzle_id is not None:
        if puzzle_identifiers is None:
            print(f"错误：数据中没有puzzle_identifiers信息")
            return
        
        # 找到所有属于该puzzle的样本
        puzzle_indices = np.where(puzzle_identifiers == args.puzzle_id)[0]
        if len(puzzle_indices) == 0:
            print(f"错误：找不到puzzle_id={args.puzzle_id}的样本")
            return
        
        print("=" * 60)
        print(f"Puzzle ID: {args.puzzle_id} 的所有状态转移对（共 {len(puzzle_indices)} 个）")
        print("=" * 60)
        
        # 显示每个状态转移对
        for idx, sample_idx in enumerate(puzzle_indices):
            input_seq = inputs[sample_idx]
            label_seq = labels[sample_idx]
            
            # 转换为原始值（减去1）
            input_seq = input_seq - 1
            label_seq = label_seq - 1
            
            # 重塑为网格
            input_grid = input_seq[:grid_size].reshape(4, grid_width)
            label_grid = label_seq[:grid_size].reshape(4, grid_width)
            
            print(f"\n{'=' * 60}")
            print(f"状态转移对 #{idx + 1} (样本索引: {sample_idx})")
            print(f"{'=' * 60}")
            visualize_grid(input_grid, "当前状态 (s_i)", pad_value=PAD_VALUE, leading_value=LEADING_VALUE)
            visualize_grid(label_grid, "下一个状态 (s_{i+1})", pad_value=PAD_VALUE, leading_value=LEADING_VALUE)
        
        return
    
    # 检查索引范围
    if args.index >= len(inputs):
        print(f"错误：索引 {args.index} 超出范围（最大索引：{len(inputs) - 1}）")
        return
    
    # 获取样本
    input_seq = inputs[args.index]
    label_seq = labels[args.index]
    puzzle_id = puzzle_identifiers[args.index] if puzzle_identifiers is not None else None
    
    # 转换为原始值（减去1，因为存储时加了1）
    # PAD值：11（值+1后）-> 10（减去1后）
    # 数字0-9：1-10（值+1后）-> 0-9（减去1后）
    input_seq = input_seq - 1
    label_seq = label_seq - 1
    
    # 重塑为网格（新格式：输入和标签都是单个网格）
    if len(input_seq) < grid_size:
        print(f"错误：输入序列长度 {len(input_seq)} 小于预期的网格大小 {grid_size}")
        return
    
    input_grid = input_seq[:grid_size].reshape(4, grid_width)
    label_grid = label_seq[:grid_size].reshape(4, grid_width)
    
    # 不要clip！PAD值（10）应该保持为10，数字0-9保持为0-9
    # 在显示时，我们会将PAD值（10）显示为特殊标记
    
    # 显示信息
    print("=" * 60)
    print(f"样本 #{args.index} (状态转移对)")
    print("=" * 60)
    if puzzle_id is not None:
        print(f"Puzzle ID: {puzzle_id}")
    print(f"网格大小: 4行 × {grid_width}列")
    print()
    
    # 显示当前状态（输入）
    visualize_grid(input_grid, "当前状态 (s_i)", pad_value=PAD_VALUE, leading_value=LEADING_VALUE)
    
    # 显示下一个状态（标签）
    visualize_grid(label_grid, "下一个状态 (s_{i+1})", pad_value=PAD_VALUE, leading_value=LEADING_VALUE)
    
    # 如果指定了显示同一个puzzle的所有状态转移对
    if args.show_puzzle and puzzle_id is not None:
        if puzzle_identifiers is None:
            print("\n警告：数据中没有puzzle_identifiers信息，无法显示同一puzzle的其他状态转移对")
        else:
            # 找到所有属于该puzzle的样本
            puzzle_indices = np.where(puzzle_identifiers == puzzle_id)[0]
            print(f"\n{'=' * 60}")
            print(f"Puzzle ID: {puzzle_id} 的所有状态转移对（共 {len(puzzle_indices)} 个）")
            print(f"{'=' * 60}")
            
            # 显示每个状态转移对
            for idx, sample_idx in enumerate(puzzle_indices):
                if sample_idx == args.index:
                    print(f"\n状态转移对 #{idx + 1} (样本索引: {sample_idx}) [当前样本]")
                else:
                    other_input_seq = inputs[sample_idx]
                    other_label_seq = labels[sample_idx]
                    
                    # 转换为原始值
                    other_input_seq = other_input_seq - 1
                    other_label_seq = other_label_seq - 1
                    
                    # 重塑为网格
                    other_input_grid = other_input_seq[:grid_size].reshape(4, grid_width)
                    other_label_grid = other_label_seq[:grid_size].reshape(4, grid_width)
                    
                    print(f"\n状态转移对 #{idx + 1} (样本索引: {sample_idx})")
                    visualize_grid(other_input_grid, "当前状态 (s_i)", pad_value=PAD_VALUE, leading_value=LEADING_VALUE)
                    visualize_grid(other_label_grid, "下一个状态 (s_{i+1})", pad_value=PAD_VALUE, leading_value=LEADING_VALUE)
    
    # 显示原始序列信息
    print(f"\n原始序列信息:")
    print(f"  输入序列长度: {len(input_seq)} (网格大小: {grid_size})")
    print(f"  标签序列长度: {len(label_seq)} (网格大小: {grid_size})")
    print(f"  输入序列（前{min(20, grid_size)}个值）: {input_seq[:min(20, grid_size)].tolist()}")
    print(f"  标签序列（前{min(20, grid_size)}个值）: {label_seq[:min(20, grid_size)].tolist()}")


if __name__ == "__main__":
    main()

