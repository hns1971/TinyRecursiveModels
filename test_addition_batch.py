#!/usr/bin/env python3
"""
批量测试加法题目的脚本

使用方法:
    python test_addition_batch.py \
        --checkpoint checkpoints/Addition-ACT-torch/finetune_addition_step1/step_30000 \
        --num-tests 10000 \
        --max-len 11 \
        --max-steps 16
"""

import os
import json
import numpy as np
import argparse
import sys
import torch
import torch.distributed as dist
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.build_addition_dataset import generate_random_addition
from debug_single_forward import initialize_model
from test_addition_puzzle import test_single_addition_puzzle





def main():
    parser = argparse.ArgumentParser(
        description="批量测试加法题目",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 测试10000个题目，网格宽度11
  python test_addition_batch.py \\
      --checkpoint checkpoints/Addition-ACT-torch/finetune_addition_step1/step_30000 \\
      --num-tests 10000 \\
      --max-len 11 \\
      --max-steps 16
        """
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint文件路径"
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=10000,
        help="测试题目数量（默认：10000）"
    )
    parser.add_argument(
        "--max-len",
        type=int,
        required=True,
        help="网格列数（数据宽度）"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=16,
        help="最大推理步数（默认：16）"
    )
    parser.add_argument(
        "--min-digits",
        type=int,
        default=1,
        help="最小数字位数（默认：1）"
    )
    parser.add_argument(
        "--max-digits",
        type=int,
        default=None,
        help="最大数字位数（默认：max-len - 1）"
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
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认：42）"
    )
    parser.add_argument(
        "--error-file",
        type=str,
        default=None,
        help="错误题目输出文件路径（默认：不保存）"
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 如果没有指定max-digits，使用max-len - 1
    if args.max_digits is None:
        args.max_digits = args.max_len - 1
    
    # 使用 debug_single_forward 的初始化函数
    base_model, model_seq_len, puzzle_id_value, config, eval_metadata = initialize_model(
        checkpoint=args.checkpoint,
        config_path=args.config_path,
        config_name=args.config_name,
        puzzle_id=args.puzzle_id,
    )
    
    print("=" * 80)
    print("批量测试加法题目")
    print("=" * 80)
    print(f"测试题目数量: {args.num_tests}")
    print(f"网格列数（数据宽度）: {args.max_len}")
    print(f"最小数字位数: {args.min_digits}")
    print(f"最大数字位数: {args.max_digits}")
    print(f"最大推理步数: {args.max_steps}")
    print(f"模型序列长度: {model_seq_len}")
    print("=" * 80)
    print()
    
    # 统计信息
    correct_count = 0
    total_count = 0
    step_distribution = {}  # 记录每个步数下正确的数量
    error_cases = []  # 记录错误题目
    
    # 批量测试（使用已初始化的模型，避免重复初始化）
    for i in tqdm(range(args.num_tests), desc="测试进度"):
        # 生成随机加法题目
        num1, num2, _ = generate_random_addition(args.min_digits, args.max_digits)
        
        # 使用 test_addition_puzzle 的单题计算函数（传入已初始化的模型参数）
        result = test_single_addition_puzzle(
            num1=num1,
            num2=num2,
            checkpoint=None,  # 不重新初始化模型
            max_len=args.max_len,
            max_steps=args.max_steps,
            puzzle_id=args.puzzle_id,
            config_path=args.config_path,
            config_name=args.config_name,
            verbose=False,  # 批量测试时不打印详细信息
            base_model=base_model,  # 使用已初始化的模型
            model_seq_len=model_seq_len,
            puzzle_id_value=puzzle_id_value,
        )
        
        total_count += 1
        is_correct = result["is_correct"]
        predicted_result = result["predicted_result"]
        expected_result = result["expected_result"]
        total_steps = result["total_steps"]
        
        if is_correct:
            correct_count += 1
            if total_steps not in step_distribution:
                step_distribution[total_steps] = 0
            step_distribution[total_steps] += 1
        else:
            # 记录错误题目
            error_cases.append({
                "num1": int(num1),
                "num2": int(num2),
                "max_len": int(args.max_len),
                "expected": int(expected_result),
                "predicted": int(predicted_result) if predicted_result is not None else None,
                "steps": int(total_steps),
                "test_id": int(i)
            })
    
    # 计算准确度
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    # 输出结果
    print("\n" + "=" * 80)
    print("测试结果")
    print("=" * 80)
    print(f"总题目数: {total_count}")
    print(f"正确答案数: {correct_count}")
    print(f"准确度: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print()
    
    if step_distribution:
        print("正确题目的步数分布:")
        sorted_steps = sorted(step_distribution.keys())
        for steps in sorted_steps:
            count = step_distribution[steps]
            percentage = count / correct_count * 100 if correct_count > 0 else 0
            print(f"  步数 {steps}: {count} 题 ({percentage:.2f}%)")
        print()
    
    # 保存错误题目到文件
    if args.error_file and error_cases:
        error_file_path = args.error_file
        # 确保目录存在
        error_dir = os.path.dirname(error_file_path)
        if error_dir and not os.path.exists(error_dir):
            os.makedirs(error_dir, exist_ok=True)
        
        # 保存为JSON格式
        error_data = {
            "total_errors": len(error_cases),
            "total_tests": total_count,
            "accuracy": accuracy,
            "test_config": {
                "max_len": args.max_len,
                "min_digits": args.min_digits,
                "max_digits": args.max_digits,
                "max_steps": args.max_steps,
                "checkpoint": args.checkpoint
            },
            "error_cases": error_cases
        }
        
        with open(error_file_path, 'w', encoding='utf-8') as f:
            json.dump(error_data, f, indent=2, ensure_ascii=False)
        
        print(f"错误题目已保存到: {error_file_path}")
        print(f"错误题目数量: {len(error_cases)}")
        print()
    
    print("=" * 80)
    
    # 清理
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
