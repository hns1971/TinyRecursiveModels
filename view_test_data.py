#!/usr/bin/env python3
"""
查看测试数据的前N条记录
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ARC网格最大大小
ARCMaxGridSize = 30

# 定义颜色映射（ARC标准颜色）
COLORS = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"
]
cmap = mcolors.ListedColormap(COLORS)
norm = mcolors.BoundaryNorm(np.arange(-0.5, 10.5, 1), cmap.N)

def decode_sequence_to_grid(seq: np.ndarray, max_size: int = ARCMaxGridSize):
    """
    将序列解码回网格格式
    
    Args:
        seq: 一维序列，值范围 [0, 11]
        max_size: 最大网格大小
    
    Returns:
        grid: 2D网格，值范围 [0, 9]（减去偏移2）
        valid_shape: 有效网格大小 (rows, cols)
    """
    # 重塑为网格
    grid = seq.reshape(max_size, max_size)
    
    # 找到EOS token（值为1）的位置
    valid_rows = max_size
    valid_cols = max_size
    
    # 查找行EOS（第一列中的EOS）
    for i in range(max_size):
        if grid[i, 0] == 1:  # EOS token
            valid_rows = i
            break
    
    # 查找列EOS（第一行中的EOS）
    for j in range(max_size):
        if grid[0, j] == 1:  # EOS token
            valid_cols = j
            break
    
    # 裁剪到有效区域
    if valid_rows < max_size or valid_cols < max_size:
        grid = grid[:valid_rows, :valid_cols]
    else:
        # 如果没有找到EOS，尝试找到最后一个非填充值
        # 填充值是0（pad_id）
        non_zero_mask = grid != 0
        if non_zero_mask.any():
            rows, cols = np.where(non_zero_mask)
            if len(rows) > 0:
                valid_rows = rows.max() + 1
                valid_cols = cols.max() + 1
                grid = grid[:valid_rows, :valid_cols]
    
    # 减去偏移（输入序列中，值+2，所以需要减2）
    grid = grid - 2
    grid = np.clip(grid, 0, 9)  # 确保值在0-9范围内
    
    return grid, (valid_rows, valid_cols)

def visualize_grid(grid: np.ndarray, title: str = "", save_path: str = None):
    """可视化ARC网格"""
    if grid.size == 0:
        print(f"  {title}: (空网格)")
        return
    
    fig, ax = plt.subplots(figsize=(max(grid.shape[1], 3), max(grid.shape[0], 3)))
    ax.imshow(grid, cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    图像已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()

def print_grid_text(grid: np.ndarray, title: str = ""):
    """以文本形式打印网格"""
    if grid.size == 0:
        print(f"  {title}: (空网格)")
        return
    
    print(f"  {title}:")
    print("  " + "-" * (grid.shape[1] * 4 + 1))
    for row in grid:
        print("  [" + " ".join(f"{int(cell):2d}" for cell in row) + "]")
    print("  " + "-" * (grid.shape[1] * 4 + 1))

def print_grid_as_array(grid: np.ndarray, title: str = ""):
    """以Python二维数组格式打印网格"""
    if grid.size == 0:
        print(f"  {title}: (空网格)")
        return
    
    print(f"  {title}:")
    print("  [")
    for i, row in enumerate(grid):
        row_str = "    [" + ", ".join(f"{int(cell)}" for cell in row) + "]"
        if i < len(grid) - 1:
            row_str += ","
        print(row_str)
    print("  ]")

def view_test_data(data_dir: str = "data/arc1concept-aug-1000/test", num_samples: int = 5, save_images: bool = True, output_dir: str = "test_data_visualizations"):
    """查看测试数据的前N条记录"""
    
    # 创建输出目录
    if save_images:
        os.makedirs(output_dir, exist_ok=True)
        print(f"图像将保存到: {output_dir}/")
    
    # 加载元数据
    metadata_path = os.path.join(data_dir, "dataset.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print("=" * 60)
    print("测试数据集信息")
    print("=" * 60)
    print(f"数据集路径: {data_dir}")
    print(f"序列长度: {metadata['seq_len']}")
    print(f"词汇表大小: {metadata['vocab_size']}")
    print(f"总样本数: {metadata.get('total_puzzles', 'N/A')}")
    print(f"总组数: {metadata.get('total_groups', 'N/A')}")
    print("=" * 60)
    
    # 加载数据文件
    print("\n加载数据文件...")
    inputs = np.load(os.path.join(data_dir, "all__inputs.npy"), mmap_mode='r')
    labels = np.load(os.path.join(data_dir, "all__labels.npy"), mmap_mode='r')
    puzzle_identifiers = np.load(os.path.join(data_dir, "all__puzzle_identifiers.npy"), mmap_mode='r')
    puzzle_indices = np.load(os.path.join(data_dir, "all__puzzle_indices.npy"), mmap_mode='r')
    group_indices = np.load(os.path.join(data_dir, "all__group_indices.npy"), mmap_mode='r')
    
    print(f"输入数据形状: {inputs.shape}")
    print(f"标签数据形状: {labels.shape}")
    print(f"Puzzle标识符形状: {puzzle_identifiers.shape}")
    
    # 显示前N条数据
    print("\n" + "=" * 60)
    print(f"前 {num_samples} 条测试数据")
    print("=" * 60)
    
    for i in range(min(num_samples, len(inputs))):
        print(f"\n样本 {i+1}:")
        print(f"  Puzzle ID: {puzzle_identifiers[i]}")
        print(f"  Puzzle Index: {puzzle_indices[i]}")
        print(f"  Group Index: {group_indices[i]}")
        
        # 解码输入序列
        input_seq = inputs[i]
        input_grid, input_shape = decode_sequence_to_grid(input_seq)
        print(f"  输入网格大小: {input_shape}")
        print("\n  文本格式:")
        print_grid_text(input_grid, "输入网格")
        print("\n  Python二维数组格式:")
        print_grid_as_array(input_grid, "输入网格")
        
        # 解码标签序列（如果有有效标签）
        label_seq = labels[i]
        # 检查是否有有效标签（忽略-100）
        valid_labels = label_seq[label_seq != -100]
        if len(valid_labels) > 0:
            # 找到第一个有效标签的位置
            first_valid_idx = np.where(label_seq != -100)[0][0]
            # 从第一个有效标签开始解码
            label_seq_valid = label_seq[first_valid_idx:]
            # 如果长度不够，可能需要填充
            if len(label_seq_valid) < ARCMaxGridSize * ARCMaxGridSize:
                # 填充到完整长度
                padded = np.zeros(ARCMaxGridSize * ARCMaxGridSize, dtype=label_seq.dtype)
                padded[:len(label_seq_valid)] = label_seq_valid
                label_seq_valid = padded
            else:
                label_seq_valid = label_seq_valid[:ARCMaxGridSize * ARCMaxGridSize]
            
            label_grid, label_shape = decode_sequence_to_grid(label_seq_valid)
            print(f"\n  标签网格大小: {label_shape}")
            print("\n  文本格式:")
            print_grid_text(label_grid, "标签网格（目标输出）")
            print("\n  Python二维数组格式:")
            print_grid_as_array(label_grid, "标签网格（目标输出）")
        else:
            print("  标签: 无有效标签（全为-100）")
        
        # 可视化
        print("\n  可视化:")
        if save_images:
            input_img_path = os.path.join(output_dir, f"sample_{i+1:03d}_input.png")
            visualize_grid(input_grid, f"样本 {i+1} - 输入", save_path=input_img_path)
            if len(valid_labels) > 0:
                label_img_path = os.path.join(output_dir, f"sample_{i+1:03d}_label.png")
                visualize_grid(label_grid, f"样本 {i+1} - 目标输出", save_path=label_img_path)
        else:
            visualize_grid(input_grid, f"样本 {i+1} - 输入")
            if len(valid_labels) > 0:
                visualize_grid(label_grid, f"样本 {i+1} - 目标输出")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="查看测试数据")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/arc1concept-aug-1000/test",
        help="测试数据目录路径"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="要查看的样本数量"
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        default=True,
        help="将图像保存到文件（默认：True）"
    )
    parser.add_argument(
        "--no_save_images",
        action="store_false",
        dest="save_images",
        help="不保存图像，尝试显示（需要GUI环境）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_data_visualizations",
        help="图像保存目录（默认：test_data_visualizations）"
    )
    
    args = parser.parse_args()
    
    view_test_data(args.data_dir, args.num_samples, args.save_images, args.output_dir)

