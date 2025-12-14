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
        required=True,
        help="网格宽度（列数），应该等于生成数据时的max_digits"
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
    print(f"序列长度: {metadata['seq_len']}")
    print(f"词汇表大小: {metadata['vocab_size']}")
    print(f"总组数: {metadata['total_groups']}")
    print(f"总puzzle数: {metadata['total_puzzles']}")
    print(f"平均每个puzzle的样本数: {metadata['mean_puzzle_examples']:.2f}")
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
    
    # 前导词（LEADING_VALUE）是10（值+1后是11，减去1后是10）
    # PAD值是11（值+1后是12，减去1后是11）
    LEADING_VALUE = 10
    PAD_VALUE = 11
    
    # 找到有效长度（去掉padding）
    # 注意：数据生成时使用0填充，而不是PAD值（10）
    # 所以我们需要通过其他方式确定有效长度
    
    # 对于输入：输入是单个网格，长度应该是4的倍数
    # 对于标签：标签是所有中间步骤的拼接，长度应该是 step_count × (4 × grid_width)
    
    # 使用传入的网格宽度参数
    grid_width = args.grid_width
    input_grid_size = 4 * grid_width  # 输入网格的大小（4行×grid_width列）
    
    # 重塑输入网格（4行×grid_width列）
    if len(input_seq) < input_grid_size:
        print(f"错误：输入序列长度 {len(input_seq)} 小于预期的网格大小 {input_grid_size}")
        return
    
    input_grid = input_seq[:input_grid_size].reshape(4, grid_width)
    
    # 标签序列是所有中间步骤的拼接
    # 每个步骤也是一个4行×grid_width列的网格
    # 所以标签序列的长度应该是 step_count × (4 × grid_width)
    step_size = 4 * grid_width  # 每个步骤的大小
    
    # 找到标签序列的有效长度
    # 由于标签可能用0填充，我们需要找到最后一个非零步骤
    label_seq_len = len(label_seq)
    
    # 计算可能的步骤数（向下取整）
    num_steps = label_seq_len // step_size if step_size > 0 else 0
    
    # 解析标签序列为多个步骤的网格
    step_grids = []
    for step_idx in range(num_steps):
        start_idx = step_idx * step_size
        end_idx = start_idx + step_size
        if end_idx > label_seq_len:
            break
        step_seq = label_seq[start_idx:end_idx]
        step_grid = step_seq.reshape(4, grid_width)
        
        # 检查这个步骤是否全为零（可能是填充）
        # 如果步骤中有非零值，或者这是第一个步骤，就添加它
        if step_idx == 0 or np.any(step_grid != 0):
            step_grids.append(step_grid)
        else:
            # 如果遇到全零的步骤，可能是填充，停止解析
            break
    
    # 不要clip！PAD值（10）应该保持为10，数字0-9保持为0-9
    # 在显示时，我们会将PAD值（10）显示为特殊标记
    
    # 显示信息
    print("=" * 60)
    print(f"样本 #{args.index}")
    print("=" * 60)
    if puzzle_id is not None:
        print(f"Puzzle ID: {puzzle_id}")
    print(f"网格大小: 4行 × {grid_width}列")
    print()
    
    # 显示输入网格
    visualize_grid(input_grid, "输入网格（初始状态 s₀）", pad_value=PAD_VALUE, leading_value=LEADING_VALUE)
    
    # 提取输入中的两个加数
    num1 = extract_number_from_row(input_grid[0], pad_value=LEADING_VALUE)
    num2 = extract_number_from_row(input_grid[1], pad_value=LEADING_VALUE)
    print(f"\n解析的加数:")
    print(f"  加数1: {num1}")
    print(f"  加数2: {num2}")
    print(f"  期望结果: {num1 + num2}")
    
    # 显示所有中间步骤
    print(f"\n{'=' * 60}")
    print(f"标签：所有中间步骤（共 {len(step_grids)} 步）")
    print(f"{'=' * 60}")
    
    if len(step_grids) == 0:
        print("警告：标签中没有找到任何步骤")
    else:
        # 显示每个中间步骤
        for step_idx, step_grid in enumerate(step_grids):
            visualize_grid(step_grid, f"步骤 {step_idx + 1} (s_{step_idx + 1})", pad_value=PAD_VALUE, leading_value=LEADING_VALUE)
            
            # 提取当前步骤的结果
            result_row = step_grid[3]
            carry_row = step_grid[2]
            
            # 检查是否有非前导词和非零值
            valid_values = result_row[(result_row != LEADING_VALUE) & (result_row != PAD_VALUE) & (result_row >= 0) & (result_row <= 9)]
            if len(valid_values) > 0:
                # 从左边第一个非前导词、非零值开始
                start_idx = 0
                while start_idx < len(result_row) and (result_row[start_idx] == LEADING_VALUE or result_row[start_idx] == PAD_VALUE or result_row[start_idx] == 0):
                    start_idx += 1
                if start_idx < len(result_row):
                    # 提取从start_idx到末尾的所有非前导词数字
                    digits = [str(int(val)) for val in result_row[start_idx:] if val != LEADING_VALUE and val != PAD_VALUE and 0 <= val <= 9]
                    if digits:
                        result = int(''.join(digits))
                    else:
                        result = 0
                else:
                    result = 0
            else:
                result = 0
            
            # 显示进位信息（将前导词显示为"F"，PAD值显示为"P"）
            def format_row_for_display(row, leading_value=LEADING_VALUE, pad_value=PAD_VALUE):
                result = []
                for val in row:
                    if val == leading_value:
                        result.append("F")  # LEADING_VALUE显示为F
                    elif val == pad_value:
                        result.append("P")  # PAD_VALUE显示为P
                    else:
                        # 转换为Python int，避免显示np.uint8(3)这样的格式
                        result.append(int(val))
                return "[" + ", ".join(str(x) for x in result) + "]"
            
            print(f"  步骤 {step_idx + 1} 的结果: {result}")
            if step_idx == len(step_grids) - 1:
                # 最后一步
                print(f"  期望最终结果: {num1 + num2}")
                print(f"  是否正确: {'✓' if result == num1 + num2 else '✗'}")
    
    # 显示步骤统计信息
    print(f"\n步骤统计:")
    print(f"  总步骤数: {len(step_grids)}")
    if puzzle_id is not None:
        print(f"  Puzzle ID: {puzzle_id}")
    
    # 显示原始序列信息
    print(f"\n原始序列信息:")
    print(f"  输入序列长度: {len(input_seq)} (网格大小: {input_grid_size})")
    print(f"  标签序列长度: {len(label_seq)} (步骤数: {len(step_grids)})")
    print(f"  输入序列（前{min(20, input_grid_size)}个值）: {input_seq[:min(20, input_grid_size)].tolist()}")
    print(f"  标签序列（前{min(40, len(label_seq))}个值）: {label_seq[:min(40, len(label_seq))].tolist()}")


if __name__ == "__main__":
    main()

