#!/usr/bin/env python3
"""
调试预测比较逻辑，验证提取工具是否正确识别出错的样本
"""
import os
import json
import torch
import numpy as np
import argparse
from extract_failed_inputs import (
    extract_numbers_from_input,
    extract_numbers_from_label,
    check_prediction_correct
)


def main():
    parser = argparse.ArgumentParser(description="调试预测比较逻辑")
    parser.add_argument(
        "--preds_file",
        type=str,
        required=True,
        help="预测结果文件路径"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="数据集路径"
    )
    parser.add_argument(
        "--sample_indices",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="要检查的样本索引（默认：前5个）"
    )
    
    args = parser.parse_args()
    
    # 加载metadata
    metadata_path = os.path.join(args.data_path, "test", "dataset.json")
    if not os.path.exists(metadata_path):
        metadata_path = os.path.join(args.data_path, "train", "dataset.json")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    seq_len = metadata['seq_len']
    
    # 加载预测结果
    data = torch.load(args.preds_file, map_location='cpu')
    inputs = data['inputs'].numpy()
    preds = data['preds'].numpy()
    has_labels = 'labels' in data
    labels = data['labels'].numpy() if has_labels else None
    
    print("=" * 80)
    print("调试预测比较逻辑")
    print("=" * 80)
    print(f"序列长度: {seq_len}")
    print(f"样本总数: {len(inputs)}")
    print(f"是否有labels: {has_labels}")
    print()
    
    if not has_labels:
        print("警告: 预测结果文件中没有labels，无法验证预测是否正确")
        print("请重新运行评估并添加 --save_outputs labels")
        return
    
    # 检查指定样本
    for idx in args.sample_indices:
        if idx >= len(inputs):
            print(f"\n样本 {idx}: 索引超出范围（最大索引：{len(inputs) - 1}）")
            continue
        
        print(f"\n{'=' * 80}")
        print(f"样本 {idx}")
        print(f"{'=' * 80}")
        
        input_seq = inputs[idx]
        pred_seq = preds[idx]
        label_seq = labels[idx]
        
        # 提取输入
        try:
            num1, num2 = extract_numbers_from_input(input_seq, seq_len)
            expected_result = num1 + num2
            print(f"输入: {num1} + {num2} = {expected_result}")
        except Exception as e:
            print(f"提取输入失败: {e}")
            continue
        
        # 提取预测结果和标签结果
        pred_result = extract_numbers_from_label(pred_seq, seq_len)
        label_result = extract_numbers_from_label(label_seq, seq_len)
        
        print(f"预测结果: {pred_result}")
        print(f"标签结果: {label_result}")
        print(f"期望结果: {expected_result}")
        
        # 检查是否正确
        is_correct_by_comparison = check_prediction_correct(pred_seq, label_seq, seq_len)
        is_correct_by_value = (pred_result == label_result)
        is_correct_by_expected = (pred_result == expected_result)
        
        print(f"\n比较结果:")
        print(f"  预测 vs 标签（函数）: {'✓ 正确' if is_correct_by_comparison else '✗ 错误'}")
        print(f"  预测 vs 标签（值）: {'✓ 正确' if is_correct_by_value else '✗ 错误'}")
        print(f"  预测 vs 期望: {'✓ 正确' if is_correct_by_expected else '✗ 错误'}")
        print(f"  标签 vs 期望: {'✓ 正确' if label_result == expected_result else '✗ 错误'}")
        
        # 显示网格对比
        print(f"\n网格对比:")
        grid_width = seq_len // 4
        
        pred_grid = (pred_seq - 1)[:seq_len].reshape(4, grid_width)
        label_grid = (label_seq - 1)[:seq_len].reshape(4, grid_width)
        
        print(f"预测网格第4行（结果）: {pred_grid[3].tolist()}")
        print(f"标签网格第4行（结果）: {label_grid[3].tolist()}")
        
        # 检查是否有差异
        diff_mask = pred_grid[3] != label_grid[3]
        if diff_mask.any():
            print(f"差异位置: {np.where(diff_mask)[0].tolist()}")
            print(f"差异值: 预测={pred_grid[3][diff_mask].tolist()}, 标签={label_grid[3][diff_mask].tolist()}")
        else:
            print("第4行完全相同")
        
        # 检查前两行是否相同（应该相同）
        row1_diff = (pred_grid[0] != label_grid[0]).any()
        row2_diff = (pred_grid[1] != label_grid[1]).any()
        if row1_diff or row2_diff:
            print(f"警告: 前两行有差异！")
            if row1_diff:
                print(f"  第1行差异: 预测={pred_grid[0].tolist()}, 标签={label_grid[0].tolist()}")
            if row2_diff:
                print(f"  第2行差异: 预测={pred_grid[1].tolist()}, 标签={label_grid[1].tolist()}")


if __name__ == "__main__":
    main()

