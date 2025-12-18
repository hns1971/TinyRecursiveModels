#!/usr/bin/env python3
"""
显示错误题目文件的程序

使用方法:
    python view_error_cases.py \
        --error-file errors/error_cases.json \
        --num-cases 10
"""

import json
import argparse
import os


def format_error_case(case, index):
    """格式化单个错误题目"""
    num1 = case.get("num1", "N/A")
    num2 = case.get("num2", "N/A")
    expected = case.get("expected", "N/A")
    predicted = case.get("predicted", "N/A")
    max_len = case.get("max_len", "N/A")
    steps = case.get("steps", "N/A")
    test_id = case.get("test_id", "N/A")
    
    # 计算实际表达式
    expression = f"{num1} + {num2} = {expected}"
    
    # 判断预测结果
    if predicted is None:
        pred_status = "无法提取结果"
    elif predicted == expected:
        pred_status = "✓ 正确"
    else:
        pred_status = f"✗ 错误 (预测: {predicted})"
    
    lines = [
        f"错题 #{index + 1} (测试ID: {test_id})",
        f"  题目: {expression}",
        f"  预测: {pred_status}",
        f"  网格宽度: {max_len}",
        f"  推理步数: {steps}",
    ]
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="显示错误题目文件中的错题",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 显示前10道错题
  python view_error_cases.py \\
      --error-file errors/error_cases.json \\
      --num-cases 10
  
  # 显示所有错题
  python view_error_cases.py \\
      --error-file errors/error_cases.json \\
      --num-cases -1
        """
    )
    
    parser.add_argument(
        "--error-file",
        type=str,
        required=True,
        help="错误题目文件路径（JSON格式）"
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=10,
        help="显示前N道错题（默认：10，-1表示显示全部）"
    )
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="显示统计信息"
    )
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.error_file):
        print(f"错误：文件不存在: {args.error_file}")
        return
    
    # 读取错误文件
    try:
        with open(args.error_file, 'r', encoding='utf-8') as f:
            error_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"错误：无法解析JSON文件: {e}")
        return
    except Exception as e:
        print(f"错误：读取文件失败: {e}")
        return
    
    # 提取信息
    total_errors = error_data.get("total_errors", 0)
    total_tests = error_data.get("total_tests", 0)
    accuracy = error_data.get("accuracy", 0.0)
    test_config = error_data.get("test_config", {})
    error_cases = error_data.get("error_cases", [])
    
    # 显示统计信息
    if args.show_stats or args.num_cases == -1:
        print("=" * 80)
        print("统计信息")
        print("=" * 80)
        print(f"总测试题目数: {total_tests}")
        print(f"错误题目数: {total_errors}")
        print(f"准确度: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print()
        
        if test_config:
            print("测试配置:")
            for key, value in test_config.items():
                print(f"  {key}: {value}")
            print()
    
    # 确定要显示的数量
    if args.num_cases == -1:
        num_to_show = len(error_cases)
    else:
        num_to_show = min(args.num_cases, len(error_cases))
    
    if num_to_show == 0:
        print("没有错误题目。")
        return
    
    # 显示错题
    print("=" * 80)
    print(f"前 {num_to_show} 道错题 (共 {len(error_cases)} 道)")
    print("=" * 80)
    print()
    
    for i in range(num_to_show):
        case = error_cases[i]
        formatted = format_error_case(case, i)
        print(formatted)
        print()
    
    if num_to_show < len(error_cases):
        print(f"... (还有 {len(error_cases) - num_to_show} 道错题未显示)")
        print()
    
    print("=" * 80)
    
    # 显示单测命令提示
    if num_to_show > 0 and len(error_cases) > 0:
        first_case = error_cases[0]
        print("\n提示：使用以下命令测试第一道错题：")
        print(f"python test_addition_puzzle.py \\")
        print(f"    --checkpoint {test_config.get('checkpoint', 'CHECKPOINT_PATH')} \\")
        print(f"    --num1 {first_case.get('num1')} \\")
        print(f"    --num2 {first_case.get('num2')} \\")
        print(f"    --max-len {first_case.get('max_len')} \\")
        print(f"    --max-steps {test_config.get('max_steps', 16)}")
        print()


if __name__ == "__main__":
    main()
