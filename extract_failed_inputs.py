#!/usr/bin/env python3
"""
从评估结果中提取出错的输入值，用于单独测试

用法:
    python extract_failed_inputs.py \
        --preds_file checkpoints/Addition-ACT-torch/finetune_addition_step2/step_25000_all_preds.0 \
        --data_path data/addition \
        --output_file failed_inputs.txt
"""
import os
import json
import torch
import numpy as np
import argparse
from typing import List, Tuple


def load_metadata(data_path: str) -> dict:
    """加载数据集metadata"""
    # 优先查找test目录，因为评估通常使用测试集
    # 注意：metadata文件实际叫 dataset.json，不是 metadata.json
    metadata_paths = [
        os.path.join(data_path, "test", "dataset.json"),
        os.path.join(data_path, "train", "dataset.json"),
        os.path.join(data_path, "dataset.json"),  # 也可能在根目录
        # 兼容旧格式（如果存在metadata.json）
        os.path.join(data_path, "test", "metadata.json"),
        os.path.join(data_path, "train", "metadata.json"),
        os.path.join(data_path, "metadata.json"),
    ]
    
    for metadata_path in metadata_paths:
        if os.path.exists(metadata_path):
            print(f"找到metadata文件: {metadata_path}")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
    
    # 如果都找不到，列出可用的路径
    available_dirs = []
    available_files = []
    if os.path.exists(data_path):
        available_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        available_files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and f.endswith('.json')]
    
    error_msg = f"找不到metadata文件。已尝试:\n" + "\n".join(f"  - {p}" for p in metadata_paths)
    if available_dirs:
        error_msg += f"\n数据目录 '{data_path}' 下可用的子目录: {available_dirs}"
    if available_files:
        error_msg += f"\n数据目录 '{data_path}' 下可用的JSON文件: {available_files}"
    
    raise FileNotFoundError(error_msg)


def extract_numbers_from_input(input_seq: np.ndarray, seq_len: int) -> Tuple[int, int]:
    """
    从输入序列中提取两个加数
    
    Args:
        input_seq: 输入序列（值+1后的格式，1-10表示0-9，11表示PAD）
        seq_len: 序列长度
    
    Returns:
        (num1, num2): 两个加数
    """
    # 转换为原始值（减去1）
    input_seq = input_seq - 1
    
    # 重塑为网格（4行n列）
    grid_width = seq_len // 4
    if grid_width == 0:
        raise ValueError(f"无法确定网格宽度: seq_len={seq_len}")
    
    # 重塑为网格
    grid = input_seq[:seq_len].reshape(4, grid_width)
    
    # 第1行和第2行是加数
    row1 = grid[0]  # 第一个加数
    row2 = grid[1]  # 第二个加数
    
    # 提取有效数字（去掉前导0和padding）
    def extract_number_from_row(row: np.ndarray) -> int:
        # 找到第一个非0数字的位置（跳过前导0）
        start_idx = 0
        while start_idx < len(row) and row[start_idx] == 0:
            start_idx += 1
        
        # 如果全是0，返回0
        if start_idx >= len(row):
            return 0
        
        # 提取有效数字
        digits = []
        for i in range(start_idx, len(row)):
            val = row[i]
            # 如果遇到padding（10）或超出范围，停止
            if val >= 10:
                break
            digits.append(int(val))
        
        # 转换为数字
        if len(digits) == 0:
            return 0
        
        num = 0
        for digit in digits:
            num = num * 10 + digit
        return num
    
    num1 = extract_number_from_row(row1)
    num2 = extract_number_from_row(row2)
    
    return num1, num2


def extract_numbers_from_label(label_seq: np.ndarray, seq_len: int) -> int:
    """
    从标签序列中提取结果（第4行）
    
    Args:
        label_seq: 标签序列（值+1后的格式）
        seq_len: 序列长度
    
    Returns:
        result: 结果数字
    """
    # 转换为原始值（减去1）
    label_seq = label_seq - 1
    
    # 重塑为网格
    grid_width = seq_len // 4
    grid = label_seq[:seq_len].reshape(4, grid_width)
    
    # 第4行是结果
    row4 = grid[3]
    
    # 提取有效数字
    start_idx = 0
    while start_idx < len(row4) and row4[start_idx] == 0:
        start_idx += 1
    
    if start_idx >= len(row4):
        return 0
    
    digits = []
    for i in range(start_idx, len(row4)):
        val = row4[i]
        if val >= 10:
            break
        digits.append(int(val))
    
    if len(digits) == 0:
        return 0
    
    num = 0
    for digit in digits:
        num = num * 10 + digit
    return num


def check_prediction_correct(pred_seq: np.ndarray, label_seq: np.ndarray, seq_len: int) -> bool:
    """
    检查预测是否正确
    
    Args:
        pred_seq: 预测序列（值+1后的格式）
        label_seq: 标签序列（值+1后的格式）
        seq_len: 序列长度
    
    Returns:
        is_correct: 是否完全正确
    """
    # 转换为原始值
    pred_seq = pred_seq - 1
    label_seq = label_seq - 1
    
    # 重塑为网格
    grid_width = seq_len // 4
    pred_grid = pred_seq[:seq_len].reshape(4, grid_width)
    label_grid = label_seq[:seq_len].reshape(4, grid_width)
    
    # 比较第4行（结果行）
    pred_result = extract_numbers_from_label(pred_seq + 1, seq_len)  # 需要+1因为函数内部会-1
    label_result = extract_numbers_from_label(label_seq + 1, seq_len)
    
    result = pred_result == label_result
    #print(f"pred_result: {pred_result}, label_result: {label_result}, result: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="从评估结果中提取出错的输入值",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 提取所有出错的输入
    python extract_failed_inputs.py \\
        --preds_file checkpoints/Addition-ACT-torch/finetune_addition_step2/step_25000_all_preds.0 \\
        --data_path data/addition \\
        --output_file failed_inputs.txt
    
    # 只提取前10个出错的输入
    python extract_failed_inputs.py \\
        --preds_file checkpoints/Addition-ACT-torch/finetune_addition_step2/step_25000_all_preds.0 \\
        --data_path data/addition \\
        --output_file failed_inputs.txt \\
        --max_failures 10
        """
    )
    
    parser.add_argument(
        "--preds_file",
        type=str,
        required=True,
        help="预测结果文件路径（例如：checkpoints/.../step_25000_all_preds.0）"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="数据集路径（例如：data/addition）"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="failed_inputs.txt",
        help="输出文件路径（默认：failed_inputs.txt）"
    )
    parser.add_argument(
        "--max_failures",
        type=int,
        default=None,
        help="最大提取的失败数量（默认：全部）"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint路径（用于生成测试命令，可选）"
    )
    
    args = parser.parse_args()
    
    # 加载metadata
    print("加载metadata...")
    metadata = load_metadata(args.data_path)
    seq_len = metadata['seq_len']
    print(f"序列长度: {seq_len}")
    
    # 加载预测结果
    print(f"加载预测结果: {args.preds_file}")
    if not os.path.exists(args.preds_file):
        raise FileNotFoundError(f"找不到预测结果文件: {args.preds_file}")
    
    data = torch.load(args.preds_file, map_location='cpu')
    
    # 检查必要的键
    if 'inputs' not in data:
        raise ValueError("预测结果文件中没有'inputs'键")
    
    # 检查是否有每一步的预测（新格式）
    step_keys = [k for k in data.keys() if k.startswith('preds_step_')]
    has_step_data = len(step_keys) > 0
    
    if has_step_data:
        print(f"检测到每一步的预测数据（共 {len(step_keys)} 步）")
        # 提取步数
        step_indices = sorted([int(k.split('_')[-1]) for k in step_keys])
        print(f"步骤索引: {step_indices}")
    else:
        # 旧格式：只有最后一步的预测
        if 'preds' not in data:
            raise ValueError("预测结果文件中没有'preds'键（也没有每一步的预测数据）")
        print("使用旧格式数据（只有最后一步的预测）")
    
    # 注意：评估过程只使用每道题目的第一步数据（s₀, s₁），所以inputs中已经是第一步数据
    # 我们只需要从inputs中提取两个加数，然后与最后一步的预测结果比较
    inputs = data['inputs'].numpy()
    print(f"样本数量: {len(inputs)}")
    print("注意: 评估过程只使用每道题目的第一步数据（s₀, s₁），进行多步推理")
    print("      这里提取的inputs已经是每道题目的第一步数据")
    
    # 提取出错的输入
    failed_inputs = []
    
    print("\n检查预测结果...")
    for i in range(len(inputs)):
    #for i in range(1):
        input_seq = inputs[i]
        
        # 提取两个加数
        try:
            num1, num2 = extract_numbers_from_input(input_seq, seq_len)
        except Exception as e:
            print(f"警告: 样本 {i} 提取数字失败: {e}")
            continue
        
        expected_result = num1 + num2
        
        if has_step_data:
            # 新格式：检查最后一步的预测（halted时的预测）
            # 注意：应该与期望结果（num1 + num2）比较，而不是与标签比较
            # 因为评估代码中的exact_accuracy也是与期望结果比较的
            has_error = False
            pred_result = None
            step_idx = None
            
            # 找到最后一步的预测（最大的step_idx）
            if step_indices:
                last_step_idx = max(step_indices)
                pred_key = f'preds_step_{last_step_idx}'
                
                if pred_key in data:
                    pred_seq = data[pred_key][i].numpy() if isinstance(data[pred_key], torch.Tensor) else data[pred_key][i]
                    pred_result = extract_numbers_from_label(pred_seq, seq_len)
                    
                    # 与期望结果比较（而不是与标签比较）
                    if pred_result != expected_result:
                        has_error = True
                        step_idx = last_step_idx + 1  # step_idx是从0开始的，显示时+1
            
            # 如果有错误，记录
            if has_error and pred_result is not None:
                failed_inputs.append((i, num1, num2, expected_result, pred_result, expected_result, step_idx, []))
        else:
            # 旧格式：只检查最后一步的预测
            # 注意：应该与期望结果（num1 + num2）比较，而不是与标签比较
            # 因为评估代码中的exact_accuracy也是与期望结果比较的
            pred_seq = data['preds'][i].numpy() if isinstance(data['preds'], torch.Tensor) else data['preds'][i]
            pred_result = extract_numbers_from_label(pred_seq, seq_len)
            
            # 与期望结果比较（而不是与标签比较）
            if pred_result != expected_result:
                failed_inputs.append((i, num1, num2, expected_result, pred_result, expected_result, None, []))
    
    print(f"\n找到 {len(failed_inputs)} 个出错的样本")
    print("（预测结果与期望结果（num1 + num2）不匹配）")
    
    # 限制数量
    if args.max_failures is not None and len(failed_inputs) > args.max_failures:
        failed_inputs = failed_inputs[:args.max_failures]
        print(f"限制为前 {args.max_failures} 个")
    
    # 保存结果
    print(f"\n保存结果到: {args.output_file}")
    with open(args.output_file, 'w') as f:
        if has_step_data:
            f.write("# 出错的加法题目（最后一步的预测与期望结果不匹配）\n")
            f.write("# 格式: 样本索引 | num1 | num2 | 期望最终结果 | 预测结果 | 期望结果 | 出错步骤 | 测试命令\n")
            f.write("# 注意: 比较的是最后一步（halted时）的预测结果与期望结果（num1 + num2）\n")
            f.write("# 评估过程只使用每道题目的第一步数据（s₀, s₁），进行多步推理，使用最后一步的预测结果\n")
        else:
            f.write("# 出错的加法题目（预测与期望结果不匹配）\n")
            f.write("# 格式: 样本索引 | num1 | num2 | 期望最终结果 | 预测结果 | 期望结果 | 测试命令\n")
            f.write("# 注意: 比较的是预测结果与期望结果（num1 + num2）\n")
            f.write("# 评估过程只使用每道题目的第一步数据（s₀, s₁），进行多步推理，使用最后一步的预测结果\n")
        f.write("# " + "=" * 70 + "\n\n")
        
        for item in failed_inputs:
            if has_step_data:
                idx, num1, num2, expected_result, pred_result, label_result, step_idx, error_steps = item
            else:
                idx, num1, num2, expected_result, pred_result, label_result, _, _ = item
                step_idx = None
                error_steps = []
            
            if args.checkpoint:
                cmd = f"python test_addition_puzzle.py --checkpoint {args.checkpoint} --num1 {num1} --num2 {num2}"
            else:
                cmd = f"# python test_addition_puzzle.py --checkpoint <checkpoint_path> --num1 {num1} --num2 {num2}"
            
            if step_idx is not None:
                # step_idx已经是+1后的值（实际步骤数），直接显示
                f.write(f"样本 {idx}: {num1} + {num2} = {expected_result} (预测: {pred_result}, 期望: {expected_result}, 出错步骤: {step_idx})\n")
                if len(error_steps) > 1:
                    # error_steps中的step_idx是从0开始的，显示时需要+1
                    other_steps = [s[0] + 1 for s in error_steps[1:]]
                    f.write(f"  其他出错的步骤: {other_steps}\n")
            else:
                f.write(f"样本 {idx}: {num1} + {num2} = {expected_result} (预测: {pred_result}, 期望: {expected_result})\n")
            f.write(f"  命令: {cmd}\n\n")
    
    # 打印前几个示例
    print("\n前5个出错的样本:")
    for item in failed_inputs[:5]:
        if has_step_data:
            idx, num1, num2, expected_result, pred_result, label_result, step_idx, error_steps = item
            if step_idx is not None:
                # step_idx已经是+1后的值（实际步骤数），直接显示
                print(f"  样本 {idx}: {num1} + {num2} = {expected_result} (预测: {pred_result}, 期望: {expected_result}, 出错步骤: {step_idx})")
            else:
                print(f"  样本 {idx}: {num1} + {num2} = {expected_result} (预测: {pred_result}, 期望: {expected_result})")
        else:
            idx, num1, num2, expected_result, pred_result, label_result, _, _ = item
            print(f"  样本 {idx}: {num1} + {num2} = {expected_result} (预测: {pred_result}, 期望: {expected_result})")
    
    print(f"\n完成！结果已保存到: {args.output_file}")


if __name__ == "__main__":
    main()

