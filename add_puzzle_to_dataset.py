#!/usr/bin/env python3
"""
向现有加法数据集中添加指定题目

使用方法:
    python add_puzzle_to_dataset.py \
        --data-dir data/addition/train \
        --num1 123 \
        --num2 456
"""

import os
import json
import numpy as np
import argparse
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 直接导入原函数（保证一致性）
try:
    from dataset.build_addition_dataset import generate_addition_puzzle, PAD_VALUE
    from dataset.common import PuzzleDatasetMetadata
except ImportError as e:
    print(f"错误：无法导入必要的模块")
    print(f"原因：{e}")
    print("\n提示：")
    print("请确保已安装所有必要的依赖，包括：")
    print("  - argdantic")
    print("  - pydantic")
    print("  - tqdm")
    print("\n可以使用以下命令安装：")
    print("  pip install argdantic pydantic tqdm")
    sys.exit(1)


def load_dataset(data_dir: str):
    """加载现有数据集"""
    metadata_path = os.path.join(data_dir, "dataset.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"找不到数据集元数据文件: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # 加载数据文件
    inputs = np.load(os.path.join(data_dir, "all__inputs.npy"))
    labels = np.load(os.path.join(data_dir, "all__labels.npy"))
    puzzle_identifiers = np.load(os.path.join(data_dir, "all__puzzle_identifiers.npy"))
    puzzle_indices = np.load(os.path.join(data_dir, "all__puzzle_indices.npy"))
    group_indices = np.load(os.path.join(data_dir, "all__group_indices.npy"))
    
    return metadata, inputs, labels, puzzle_identifiers, puzzle_indices, group_indices


def add_puzzle_to_dataset(
    data_dir: str,
    num1: int,
    num2: int,
    puzzle_id: int = None,
    backup: bool = True
):
    """
    向数据集中添加一道题目
    
    Args:
        data_dir: 数据集目录（train或test）
        num1: 第一个加数
        num2: 第二个加数
        puzzle_id: 可选的puzzle ID（如果为None，则使用当前最大ID+1）
        backup: 是否备份原文件
    """
    print("=" * 60)
    print("向数据集添加题目")
    print("=" * 60)
    print(f"数据集目录: {data_dir}")
    print(f"题目: {num1} + {num2} = {num1 + num2}")
    print("=" * 60)
    
    # 加载现有数据集
    print("\n加载现有数据集...")
    metadata, inputs, labels, puzzle_identifiers, puzzle_indices, group_indices = load_dataset(data_dir)
    
    # 获取网格宽度
    grid_size = metadata['seq_len']
    grid_width = grid_size // 4
    
    print(f"当前数据集信息:")
    print(f"  - 总样本数: {len(inputs)}")
    print(f"  - 总puzzle数: {metadata['total_puzzles']}")
    print(f"  - 网格宽度: {grid_width}")
    print(f"  - 序列长度: {grid_size}")
    
    # 备份原文件（如果需要）
    if backup:
        print("\n备份原文件...")
        backup_dir = os.path.join(data_dir, "backup")
        os.makedirs(backup_dir, exist_ok=True)
        import shutil
        for filename in ["all__inputs.npy", "all__labels.npy", "all__puzzle_identifiers.npy", 
                        "all__puzzle_indices.npy", "all__group_indices.npy", "dataset.json"]:
            src = os.path.join(data_dir, filename)
            if os.path.exists(src):
                dst = os.path.join(backup_dir, filename)
                shutil.copy2(src, dst)
        print(f"  备份完成: {backup_dir}")
    
    # 确定新的puzzle_id
    if puzzle_id is None:
        puzzle_id = metadata['total_puzzles']
        print(f"\n使用新的puzzle_id: {puzzle_id}")
    else:
        print(f"\n使用指定的puzzle_id: {puzzle_id}")
        if puzzle_id < metadata['num_puzzle_identifiers']:
            print(f"  警告: puzzle_id {puzzle_id} 已存在，将覆盖相关数据")
    
    # 生成新题目的数据（使用与原数据集生成相同的函数）
    print(f"\n生成题目数据...")
    input_grid, step_grids = generate_addition_puzzle(num1, num2, grid_width)
    
    if len(step_grids) == 0:
        print("  警告: 没有生成任何步骤，跳过添加")
        return
    
    print(f"  生成了 {len(step_grids)} 个步骤")
    
    # 按照 convert_subset 的逻辑生成数据
    # 构建完整轨迹：s₀（初始状态）, s₁, s₂, ..., sₜ（最终状态）
    current_state_grid = input_grid.copy()
    
    new_inputs = []
    new_labels = []
    new_puzzle_identifiers = []
    new_puzzle_indices = []
    
    # 为每个状态转移生成一条数据
    for next_state_grid in step_grids:
        # 输入：当前状态，前两行保持为加数（不变），后两行是当前状态的后两行
        input_state_grid = current_state_grid.copy()
        input_state_grid[0, :] = input_grid[0, :]  # 第1行保持为第一个加数
        input_state_grid[1, :] = input_grid[1, :]  # 第2行保持为第二个加数
        current_state = input_state_grid.flatten()
        
        # 标签：下一个状态，前两行保持为加数（与input一样）
        label_grid = next_state_grid.copy()
        label_grid[0, :] = input_grid[0, :]  # 第1行保持为第一个加数
        label_grid[1, :] = input_grid[1, :]  # 第2行保持为第二个加数
        next_state = label_grid.flatten()
        
        # 添加状态转移对 (current_state, next_state)
        new_inputs.append(current_state)
        new_labels.append(next_state)
        new_puzzle_identifiers.append(puzzle_id)
        
        # 更新当前状态为下一个状态，用于下一个转移对
        current_state_grid = next_state_grid.copy()
        current_state_grid[0, :] = input_grid[0, :]  # 第1行保持为第一个加数
        current_state_grid[1, :] = input_grid[1, :]  # 第2行保持为第二个加数
    
    # 在最后一步之后，添加两条 (s(n), s(n)) 的数据
    final_state_grid = current_state_grid.copy()
    final_state = final_state_grid.flatten()
    
    for _ in range(2):
        new_inputs.append(final_state)
        new_labels.append(final_state)
        new_puzzle_identifiers.append(puzzle_id)
    
    num_transitions = len(step_grids) + 2  # 包括最后添加的2条终止数据
    print(f"  生成了 {num_transitions} 条数据（包括2条终止数据）")
    
    # 填充序列到相同长度
    def _pad_sequences(seq_list, pad_value=PAD_VALUE):
        padded = []
        for seq in seq_list:
            if len(seq) < grid_size:
                pad_len = grid_size - len(seq)
                pad = np.full(pad_len, pad_value, dtype=np.uint8)
                padded_seq = np.concatenate([seq, pad])
            else:
                padded_seq = seq[:grid_size]
            padded.append(padded_seq)
        return np.array(padded, dtype=np.uint8)
    
    # 转换为numpy数组（值+1）
    new_inputs_array = _pad_sequences(new_inputs, pad_value=PAD_VALUE) + 1
    new_labels_array = _pad_sequences(new_labels, pad_value=PAD_VALUE) + 1
    new_puzzle_identifiers_array = np.array(new_puzzle_identifiers, dtype=np.int32)
    
    # 计算新的puzzle_indices
    # puzzle_indices记录每个样本的索引（从0开始，每次添加样本时递增）
    # 新数据的起始索引是当前inputs的长度
    start_index = len(inputs)
    new_puzzle_indices = []
    for i in range(len(new_inputs_array)):
        new_puzzle_indices.append(start_index + i)
    new_puzzle_indices_array = np.array(new_puzzle_indices, dtype=np.int32)
    
    # 合并到现有数据集
    print(f"\n合并数据...")
    inputs = np.concatenate([inputs, new_inputs_array], axis=0)
    labels = np.concatenate([labels, new_labels_array], axis=0)
    puzzle_identifiers = np.concatenate([puzzle_identifiers, new_puzzle_identifiers_array], axis=0)
    puzzle_indices = np.concatenate([puzzle_indices, new_puzzle_indices_array], axis=0)
    
    # 更新group_indices
    # group_indices记录每个group的结束位置
    # 新puzzle的所有步骤组成一个新的group
    new_group_end = len(inputs)
    group_indices = np.concatenate([group_indices, [new_group_end]], axis=0)
    
    # 更新元数据
    print(f"\n更新元数据...")
    max_puzzle_identifier = max(puzzle_identifiers) if len(puzzle_identifiers) > 0 else 0
    num_puzzle_identifiers = max_puzzle_identifier + 1
    
    # 计算新的平均步骤数
    # 需要从puzzle_indices计算每个puzzle的步骤数
    puzzle_step_counts = []
    for i in range(len(group_indices) - 1):
        start_idx = group_indices[i]
        end_idx = group_indices[i + 1]
        puzzle_step_counts.append(end_idx - start_idx)
    
    mean_steps = np.mean(puzzle_step_counts) if len(puzzle_step_counts) > 0 else 0
    
    new_metadata = PuzzleDatasetMetadata(
        seq_len=metadata['seq_len'],
        vocab_size=metadata['vocab_size'],
        pad_id=metadata['pad_id'],
        ignore_label_id=metadata.get('ignore_label_id', metadata['pad_id']),
        blank_identifier_id=metadata['blank_identifier_id'],
        num_puzzle_identifiers=num_puzzle_identifiers,
        total_groups=len(group_indices) - 1,
        mean_puzzle_examples=mean_steps,
        total_puzzles=metadata['total_puzzles'] + 1,
        sets=metadata['sets']
    )
    
    # 保存更新后的数据
    print(f"\n保存数据...")
    np.save(os.path.join(data_dir, "all__inputs.npy"), inputs)
    np.save(os.path.join(data_dir, "all__labels.npy"), labels)
    np.save(os.path.join(data_dir, "all__puzzle_identifiers.npy"), puzzle_identifiers)
    np.save(os.path.join(data_dir, "all__puzzle_indices.npy"), puzzle_indices)
    np.save(os.path.join(data_dir, "all__group_indices.npy"), group_indices)
    
    with open(os.path.join(data_dir, "dataset.json"), "w") as f:
        json.dump(new_metadata.model_dump(), f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ 题目添加完成！")
    print("=" * 60)
    print(f"更新后的数据集信息:")
    print(f"  - 总样本数: {len(inputs)}")
    print(f"  - 总puzzle数: {new_metadata.total_puzzles}")
    print(f"  - 新添加的puzzle_id: {puzzle_id}")
    print(f"  - 新puzzle的步骤数: {num_transitions}")
    print(f"  - 平均每个puzzle的状态转移数: {mean_steps:.2f}")
    if backup:
        print(f"\n原文件已备份到: {os.path.join(data_dir, 'backup')}")


def main():
    parser = argparse.ArgumentParser(
        description="向现有加法数据集中添加指定题目",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 向训练集添加题目 123 + 456
  python add_puzzle_to_dataset.py \\
      --data-dir data/addition/train \\
      --num1 123 \\
      --num2 456

  # 向测试集添加题目，并指定puzzle_id
  python add_puzzle_to_dataset.py \\
      --data-dir data/addition/test \\
      --num1 999 \\
      --num2 1 \\
      --puzzle-id 1000

  # 不备份原文件
  python add_puzzle_to_dataset.py \\
      --data-dir data/addition/train \\
      --num1 123 \\
      --num2 456 \\
      --no-backup
        """
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="数据集目录（train或test）"
    )
    parser.add_argument(
        "--num1",
        type=int,
        required=True,
        help="第一个加数"
    )
    parser.add_argument(
        "--num2",
        type=int,
        required=True,
        help="第二个加数"
    )
    parser.add_argument(
        "--puzzle-id",
        type=int,
        default=None,
        help="可选的puzzle ID（如果未指定，则使用当前最大ID+1）"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="不备份原文件"
    )
    
    args = parser.parse_args()
    
    add_puzzle_to_dataset(
        data_dir=args.data_dir,
        num1=args.num1,
        num2=args.num2,
        puzzle_id=args.puzzle_id,
        backup=not args.no_backup
    )


if __name__ == "__main__":
    main()
