#!/usr/bin/env python3
"""
测试单个加法题目的脚本

使用方法:
    python test_addition_puzzle.py \
        --checkpoint checkpoints/Addition-ACT-torch/finetune_addition_step1/step_30000 \
        --num1 123 \
        --num2 456 \
        --max-steps 16
"""

import os
import json
import numpy as np
import argparse
import sys
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pretrain import (
    PretrainConfig,
    load_synced_config,
    create_model,
)
from dataset.common import PuzzleDatasetMetadata
from dataset.build_addition_dataset import generate_addition_puzzle
from debug_single_forward import forward_once
from inference import create_model_from_checkpoint

# 前导词：使用10表示前导位置和未计算位置，以区分数字0和前导/未计算位置
LEADING_VALUE = 10
# PAD值：使用11表示PAD，用于batch填充
PAD_VALUE = 11


def count_digits(n: int) -> int:
    """计算数字的位数（不考虑前导0）"""
    if n == 0:
        return 1
    count = 0
    while n > 0:
        count += 1
        n //= 10
    return count


def number_to_digits(num: int, max_len: int, actual_digits: int = None, pad_to_len: int = None, use_leading_pad: bool = True) -> np.ndarray:
    """将数字转换为数字数组，右对齐（个位在最右边）
    
    Args:
        num: 要转换的数字
        max_len: 最大长度（网格宽度）
        actual_digits: 数字的实际位数（不包括前导0），如果为None则自动计算
        pad_to_len: 先用0补齐到的长度（如果为None，则直接用前导词补齐到max_len）
        use_leading_pad: 如果True，前导位置用LEADING_VALUE补齐；如果False，用PAD_VALUE补齐
    
    Returns:
        数字数组，右对齐
    """
    if actual_digits is None:
        actual_digits = count_digits(num)
    
    digits = []
    if num == 0:
        digits = [0]  # 数字0本身用0表示
    else:
        while num > 0:
            digits.append(num % 10)
            num //= 10
    # 反转数组，使得个位在最后（最右边）
    digits = digits[::-1]
    
    # 如果指定了pad_to_len，先用0补齐到pad_to_len
    if pad_to_len is not None and len(digits) < pad_to_len:
        digits = [0] * (pad_to_len - len(digits)) + digits
    
    # 然后补齐到max_len，前导位置用LEADING_VALUE或PAD_VALUE
    if use_leading_pad:
        # 用LEADING_VALUE补齐（前导位置）
        padded_digits = [LEADING_VALUE] * (max_len - len(digits)) + digits
    else:
        # 用PAD_VALUE补齐（用于batch填充）
        padded_digits = [PAD_VALUE] * (max_len - len(digits)) + digits
    
    return np.array(padded_digits[:max_len], dtype=np.uint8)


def create_addition_grid(num1: int, num2: int, max_len: int = None) -> np.ndarray:
    """
    创建加法题目的4行n列网格（与训练数据生成逻辑一致）
    
    Args:
        num1: 第一个加数
        num2: 第二个加数
        max_len: 最大位数（如果为None，从metadata读取）
    
    Returns:
        4行n列的网格
        - 第1行：第一个加数（右对齐，前导位置用LEADING_VALUE补齐）
        - 第2行：第二个加数（右对齐，前导位置用LEADING_VALUE补齐）
        - 第3行：进位数（全LEADING_VALUE，初始状态，表示未计算）
        - 第4行：结果（全LEADING_VALUE，初始状态，表示未计算）
    
    补齐规则（与训练数据生成逻辑一致）：
        - 如果两个数位数不同，位数少的先用0补齐到与多的位数一样
        - 为避免矛盾，两个数前面多补一个0，用于计算
        - 前导位置用LEADING_VALUE补齐，未计算位置也用LEADING_VALUE表示
    """
    # 如果max_len为None，使用默认值11（与当前训练数据一致）
    # 注意：如果重新生成了12位的数据集，这里应该改为12
    if max_len is None:
        max_len = 11  # 当前训练数据是4行×11列=44
    
    # 计算两个数的实际位数
    digits1_count = count_digits(num1)
    digits2_count = count_digits(num2)
    actual_max_digits = max(digits1_count, digits2_count)
    
    # 确定补齐规则（与训练数据生成逻辑一致）：
    # 1. 如果两个数位数不同，位数少的先用0补齐到与多的位数一样
    # 2. 为避免矛盾，两个数前面多补一个0，用于计算
    # 所以 pad_to_len = actual_max_digits + 1
    pad_to_len = actual_max_digits + 1
    
    # 转换为数字数组（右对齐）
    # 先用0补齐到pad_to_len，然后用LEADING_VALUE补齐到max_len
    digits1 = number_to_digits(num1, max_len, digits1_count, pad_to_len=pad_to_len, use_leading_pad=True)
    digits2 = number_to_digits(num2, max_len, digits2_count, pad_to_len=pad_to_len, use_leading_pad=True)
    
    # 创建4行网格，初始化为LEADING_VALUE（与训练数据生成逻辑一致）
    grid = np.full((4, max_len), LEADING_VALUE, dtype=np.uint8)
    grid[0, :] = digits1  # 第1行：第一个加数（前导位置已经是LEADING_VALUE）
    grid[1, :] = digits2  # 第2行：第二个加数（前导位置已经是LEADING_VALUE）
    # 第3行和第4行保持为LEADING_VALUE（初始状态，表示未计算）
    # 注意：在中间步骤的预测结果中，进位行（第3行）的个位位置（最右边）应该是PAD_VALUE，
    # 因为个位没有来自右边的进位输入。但在初始状态时，第3行全为LEADING_VALUE是正确的。
    
    return grid


def visualize_addition_grid(grid: np.ndarray, title: str = "", pad_value: int = PAD_VALUE, leading_value: int = LEADING_VALUE):
    """可视化加法网格，LEADING_VALUE显示为 F，PAD值显示为 P"""
    print(f"\n{title}:")
    print("-" * 60)
    
    def format_row_for_display(row):
        """格式化行显示，将LEADING_VALUE（10）显示为 F，PAD值（11）显示为 P，其他值转换为int"""
        result = []
        for val in row:
            if val == leading_value:
                result.append("F")  # LEADING_VALUE显示为F
            elif val == pad_value:
                result.append("P")  # PAD_VALUE显示为P
            else:
                result.append(int(val))
        return result
    
    def row_to_str(row_vals):
        return " ".join(str(x) for x in format_row_for_display(row_vals))
    
    print(f"第1行（加数1）: {row_to_str(grid[0])}")
    print(f"第2行（加数2）: {row_to_str(grid[1])}")
    print(f"第3行（进位）: {row_to_str(grid[2])}")
    print(f"第4行（结果）: {row_to_str(grid[3])}")
    print("-" * 60)


def extract_result_from_grid(pred_grid, row_index=3):
    """从预测网格中提取结果数字（默认从第4行提取）"""
    result_digits = pred_grid[row_index]
    # 去掉前导LEADING_VALUE、PAD_VALUE和前导0
    start_idx = 0
    while start_idx < len(result_digits) and (result_digits[start_idx] == LEADING_VALUE or result_digits[start_idx] == PAD_VALUE or result_digits[start_idx] == 0):
        start_idx += 1
    if start_idx < len(result_digits):
        # 保留有效数字（跳过LEADING_VALUE和PAD_VALUE）
        valid_digits = [str(int(val)) for val in result_digits[start_idx:] if val != LEADING_VALUE and val != PAD_VALUE and 0 <= val <= 9]
        if len(valid_digits) > 0:
            try:
                return int(''.join(valid_digits))
            except:
                pass
    # 如果所有位置都被跳过，检查是否全为0（0是有效结果）
    if start_idx >= len(result_digits):
        # 检查是否全为0：所有值都是0或LEADING_VALUE（没有PAD_VALUE），且至少有一个0
        has_pad = any(val == PAD_VALUE for val in result_digits)
        if not has_pad:
            # 没有PAD_VALUE，检查是否有0
            has_zero = any(val == 0 for val in result_digits)
            if has_zero:
                return 0
    return None


def test_single_addition_puzzle(
    num1: int,
    num2: int,
    checkpoint: str = None,
    max_len: int = None,
    max_steps: int = 16,
    puzzle_id: int = None,
    config_path: str = "config",
    config_name: str = "cfg_finetune_addition",
    confidence_threshold: float = 0.9,
    verbose: bool = True,
    stop_on_halt: bool = False,
    # 可选：传入已初始化的模型参数（避免重复初始化）
    base_model=None,
    model_seq_len=None,
    puzzle_id_value=None,
):
    """
    测试单个加法题目
    
    Args:
        num1: 第一个加数
        num2: 第二个加数
        checkpoint: Checkpoint文件路径（如果base_model已提供，则不需要）
        max_len: 网格列数（如果为None，从metadata读取）
        max_steps: 最大推理步数
        puzzle_id: Puzzle identifier（None则禁用puzzle_emb）
        config_path: 配置文件目录
        config_name: 配置文件名
        confidence_threshold: 置信度阈值
        verbose: 是否打印详细信息
        stop_on_halt: 是否在模型预测停止时提前退出（True：按模型预测停止，False：跑满max_steps）
        base_model: 可选的已初始化模型（如果提供，则不会重新初始化）
        model_seq_len: 可选的模型序列长度（如果base_model已提供，需要提供此参数）
        puzzle_id_value: 可选的puzzle_id值（如果base_model已提供，需要提供此参数）
    
    Returns:
        dict: 包含以下键的字典
            - is_correct: 是否正确
            - predicted_result: 预测结果（整数）
            - expected_result: 期望结果（整数）
            - total_steps: 总步数
            - all_predictions: 所有步骤的预测网格列表
            - all_q_halt_logits: 所有步骤的q_halt_logits列表
            - all_confidence_scores: 所有步骤的置信度列表
            - final_pred_grid: 最终预测网格
    """
    # 初始化模型（如果未提供）
    if base_model is None:
        if checkpoint is None:
            raise ValueError("必须提供checkpoint或base_model")
        # 使用新的模型初始化方式
        base_model, config, eval_metadata = create_model_from_checkpoint(
            checkpoint_path=checkpoint,
            data_paths=["data/addition"],  # 默认数据路径
            config_path=config_path,
            config_name=config_name,
            rank=0,
            world_size=1,
            auto_detect_task=True,
        )
        model_seq_len = eval_metadata.seq_len
        puzzle_id_value = puzzle_id if puzzle_id is not None else 0
    else:
        if model_seq_len is None or puzzle_id_value is None:
            raise ValueError("如果提供base_model，必须同时提供model_seq_len和puzzle_id_value")
    
    # 根据metadata确定网格列数
    if max_len is None:
        max_len = model_seq_len // 4  # 从metadata读取
        if verbose:
            print(f"从metadata读取: seq_len={model_seq_len}, 网格列数={max_len}")
    elif verbose:
        print(f"使用用户指定的网格列数: {max_len}")
    
    # 创建加法网格
    grid = create_addition_grid(num1, num2, max_len)
    expected_result = num1 + num2
    
    if verbose:
        print("=" * 60)
        print("加法题目测试")
        print("=" * 60)
        print(f"题目: {num1} + {num2} = {expected_result}")
        print(f"网格大小: {grid.shape[0]}行 × {grid.shape[1]}列")
        visualize_addition_grid(grid, "输入网格（初始状态）", pad_value=PAD_VALUE, leading_value=LEADING_VALUE)
        print("=" * 60)
        print()
        print("=" * 60)
        print("开始推理...")
        print("=" * 60)
    
    # 推理循环
    all_predictions = []
    all_q_halt_logits = []
    all_confidence_scores = []
    
    # 当前输入网格（开始时使用初始网格，之后使用预测结果）
    current_grid = grid.copy()
    
    with torch.no_grad():
        for step in range(max_steps):
            if verbose:
                print("\n" + "="*80)
                print(f"[单测] 步骤 {step + 1} - 输入网格:")
                print("="*80)
                visualize_addition_grid(current_grid, f"步骤 {step + 1} 输入", pad_value=PAD_VALUE, leading_value=LEADING_VALUE)
            
            # 调用 forward_once
            carry, metrics, extracted_metrics, batch, pred_grid = forward_once(
                model=base_model,
                grid=current_grid,
                max_len=max_len,
                model_seq_len=model_seq_len,
                puzzle_id_value=puzzle_id_value,
                carry=None,  # 每次都重新初始化carry，只使用预测结果（网格）作为输入
                verbose=False,  # 不打印详细信息，我们自己打印
            )
            
            # 获取 q_halt_logits（从metrics中获取）
            if "q_halt_logits" in metrics:
                q_halt_logits = metrics["q_halt_logits"]
                model_halted = carry.halted[0].item() if hasattr(carry, 'halted') else False
                model_halt_pred = (q_halt_logits[0] >= 0).item()
                confidence = torch.sigmoid(q_halt_logits[0]).item()
            else:
                # 如果没有q_halt_logits，使用默认值
                q_halt_logits = None
                model_halted = carry.halted[0].item() if hasattr(carry, 'halted') else False
                model_halt_pred = False
                confidence = 0.0
                if verbose:
                    print(f"警告：步骤 {step + 1} 的 metrics 中没有 q_halt_logits")
            
            all_predictions.append(pred_grid.copy())
            if q_halt_logits is not None:
                all_q_halt_logits.append(q_halt_logits[0].item())
            else:
                all_q_halt_logits.append(0.0)  # 默认值
            all_confidence_scores.append(confidence)
            
            if verbose:
                print(f"\n步骤 {step + 1} 输出:")
                if q_halt_logits is not None:
                    print(f"  Q_halt logit: {q_halt_logits[0].item():.4f}")
                else:
                    print(f"  Q_halt logit: N/A")
                print(f"  置信度 (sigmoid): {confidence:.4f}")
                print(f"  模型 halted 标志: {model_halted}  (来自 carry.halted)")
                print(f"  模型 halt 预测: {model_halt_pred}  (q_halt_logits >= 0)")
                visualize_addition_grid(pred_grid, f"步骤 {step + 1} 预测", pad_value=PAD_VALUE, leading_value=LEADING_VALUE)
            
            # 将预测结果作为下一次的输入
            current_grid = pred_grid.copy()
            
            # 提取结果并检查
            predicted_result = extract_result_from_grid(pred_grid)
            if predicted_result is not None and verbose:
                status = "✓" if predicted_result == expected_result else "✗"
                print(f"  提取的结果: {predicted_result} (期望: {expected_result}) {status}")
            
            # 如果启用了按模型预测停止，检查是否应该提前退出
            if stop_on_halt:
                # 检查模型是否预测停止（使用置信度阈值或halt预测）
                should_stop = False
                if q_halt_logits is not None:
                    # 如果置信度超过阈值，或者模型明确预测停止
                    if confidence >= confidence_threshold or model_halt_pred:
                        should_stop = True
                        if verbose:
                            print(f"\n[停止] 模型预测停止（置信度: {confidence:.4f}, halt预测: {model_halt_pred}）")
                elif model_halted:
                    # 如果carry中的halted标志为True，也停止
                    should_stop = True
                    if verbose:
                        print(f"\n[停止] 模型halted标志为True")
                
                if should_stop:
                    if verbose:
                        print(f"提前停止推理（步数: {step + 1}/{max_steps}）")
                    break
    
    # 提取最终结果
    final_pred_grid = all_predictions[-1] if all_predictions else None
    predicted_result = extract_result_from_grid(final_pred_grid) if final_pred_grid is not None else None
    is_correct = (predicted_result == expected_result) if predicted_result is not None else False
    
    if verbose:
        # 显示最终结果
        print("\n" + "=" * 60)
        print("最终结果")
        print("=" * 60)
        
        if final_pred_grid is not None:
            visualize_addition_grid(final_pred_grid, "最终预测网格", pad_value=PAD_VALUE, leading_value=LEADING_VALUE)
            
            if predicted_result is not None:
                print(f"\n最终结果: {predicted_result}")
                print(f"期望结果: {expected_result}")
                print(f"是否正确: {'✓ 正确' if is_correct else '✗ 错误'}")
            else:
                print(f"\n无法解析结果")
            
            print(f"\n推理统计:")
            print(f"  总步数: {len(all_predictions)}")
            print(f"  最终Q_halt logit: {all_q_halt_logits[-1]:.4f}")
            print(f"  最终置信度: {all_confidence_scores[-1]:.4f}")
            
            # 显示所有步骤的Q值和置信度变化
            print(f"\nQ值和置信度变化:")
            for i, (q_halt, conf) in enumerate(zip(all_q_halt_logits, all_confidence_scores)):
                threshold_mark = "✓" if conf >= confidence_threshold else ""
                print(f"  步骤 {i+1}: Q_halt={q_halt:7.4f}, 置信度={conf:.4f} {threshold_mark}")
    
    return {
        "is_correct": is_correct,
        "predicted_result": predicted_result,
        "expected_result": expected_result,
        "total_steps": len(all_predictions),
        "all_predictions": all_predictions,
        "all_q_halt_logits": all_q_halt_logits,
        "all_confidence_scores": all_confidence_scores,
        "final_pred_grid": final_pred_grid,
    }


def main():
    parser = argparse.ArgumentParser(
        description="测试单个加法题目",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 测试 123 + 456
  python test_addition_puzzle.py \\
      --checkpoint checkpoints/Addition-ACT-torch/finetune_addition_step1/step_30000 \\
      --num1 123 \\
      --num2 456

  # 测试 999 + 1（有进位）
  python test_addition_puzzle.py \\
      --checkpoint checkpoints/Addition-ACT-torch/finetune_addition_step1/step_30000 \\
      --num1 999 \\
      --num2 1 \\
      --max-steps 20
        """
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint文件路径"
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
        "--max-len",
        type=int,
        default=None,
        help="网格列数（默认：None，从训练数据metadata自动读取）"
    )
    parser.add_argument(
        "--puzzle-id",
        type=int,
        default=None,
        help="可选：指定 puzzle identifier；不指定则禁用 puzzle_emb（puzzle_emb_len=0）"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=16,
        help="最大推理步数（默认：16）"
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="调试用：显示每一步用于监督的标签网格 (s_{i+1})，方便核对是否与题目匹配"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.9,
        help="置信度阈值（sigmoid(q_halt_logits)），超过此值则停止推理（默认：0.9）"
    )
    parser.add_argument(
        "--stop-on-halt",
        action="store_true",
        help="是否在模型预测停止时提前退出（默认：False，跑满max_steps）"
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
    
    args = parser.parse_args()
    
    # 调用测试函数
    result = test_single_addition_puzzle(
        num1=args.num1,
        num2=args.num2,
        checkpoint=args.checkpoint,
        max_len=args.max_len,
        max_steps=args.max_steps,
        puzzle_id=args.puzzle_id,
        config_path=args.config_path,
        config_name=args.config_name,
        confidence_threshold=args.confidence_threshold,
        verbose=True,
        stop_on_halt=args.stop_on_halt,
    )
    
    # 清理
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
