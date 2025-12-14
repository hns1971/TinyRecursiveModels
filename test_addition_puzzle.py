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

# PAD值：使用10表示PAD，以区分数字0和PAD
PAD_VALUE = 10


def count_digits(n: int) -> int:
    """计算数字的位数（不考虑前导0）"""
    if n == 0:
        return 1
    count = 0
    while n > 0:
        count += 1
        n //= 10
    return count


def number_to_digits(num: int, max_len: int, actual_digits: int = None, pad_to_len: int = None, use_zero_pad: bool = False) -> np.ndarray:
    """将数字转换为数字数组，右对齐（个位在最右边）
    
    Args:
        num: 要转换的数字
        max_len: 最大长度（网格宽度）
        actual_digits: 数字的实际位数（不包括前导0），如果为None则自动计算
        pad_to_len: 先用0补齐到的长度（如果为None，则直接用PAD补齐到max_len）
        use_zero_pad: 如果True，前导位置用0补齐；如果False，用PAD_VALUE补齐
    
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
    
    # 然后补齐到max_len
    if use_zero_pad:
        # 用0补齐
        padded_digits = [0] * (max_len - len(digits)) + digits
    else:
        # 用PAD_VALUE补齐
        padded_digits = [PAD_VALUE] * (max_len - len(digits)) + digits
    
    return np.array(padded_digits[:max_len], dtype=np.uint8)


def create_addition_grid(num1: int, num2: int, max_len: int = None) -> np.ndarray:
    """
    创建加法题目的4行n列网格
    
    Args:
        num1: 第一个加数
        num2: 第二个加数
        max_len: 最大位数（如果为None，从metadata读取）
    
    Returns:
        4行n列的网格，值范围0-9
        - 第1行：第一个加数（右对齐，前导位置用0补齐）
        - 第2行：第二个加数（右对齐，前导位置用0补齐）
        - 第3行：进位数（全0，初始状态）
        - 第4行：结果（全0，初始状态）
    
    补齐规则（与训练数据生成逻辑一致）：
        - 如果两个数位数不同，位数少的补0到与多的位数一样
        - 如果两个数位数相同，两个都补一个0
        - 然后用0补齐到max_len（不再使用PAD）
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
    # 1. 如果两个数位数不同，位数少的补0到与多的位数一样
    # 2. 如果两个数位数相同，两个都补一个0
    if digits1_count != digits2_count:
        # 位数不同：位数少的补0到与多的位数一样
        pad_to_len = actual_max_digits
    else:
        # 位数相同：两个都补一个0
        pad_to_len = actual_max_digits + 1
    
    # 转换为数字数组（右对齐）
    # 先用0补齐到pad_to_len，然后用0补齐到max_len（不再使用PAD）
    digits1 = number_to_digits(num1, max_len, digits1_count, pad_to_len=pad_to_len, use_zero_pad=True)
    digits2 = number_to_digits(num2, max_len, digits2_count, pad_to_len=pad_to_len, use_zero_pad=True)
    
    # 创建4行网格，初始化为0
    grid = np.zeros((4, max_len), dtype=np.uint8)
    grid[0, :] = digits1  # 第1行：第一个加数（前导位置已经是0）
    grid[1, :] = digits2  # 第2行：第二个加数（前导位置已经是0）
    # 第3行和第4行保持为0（初始状态，表示未计算）
    
    return grid


def visualize_addition_grid(grid: np.ndarray, title: str = "", pad_value: int = PAD_VALUE):
    """可视化加法网格，PAD值显示为'P'"""
    print(f"\n{title}:")
    print("-" * 60)
    
    def format_row_for_display(row):
        """格式化行显示，将PAD值（10）显示为'P'，其他值转换为int"""
        return [int(val) if val != pad_value else "P" for val in row]
    
    print(f"第1行（加数1）: {format_row_for_display(grid[0])}")
    print(f"第2行（加数2）: {format_row_for_display(grid[1])}")
    print(f"第3行（进位）: {format_row_for_display(grid[2])}")
    print(f"第4行（结果）: {format_row_for_display(grid[3])}")
    print("-" * 60)


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
        type=str,
        default=None,
        help="Puzzle标识符（默认：自动生成）"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=16,
        help="最大推理步数（默认：16）"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.9,
        help="置信度阈值（sigmoid(q_halt_logits)），超过此值则停止推理（默认：0.9）"
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
    
    # 生成puzzle_id
    if args.puzzle_id is None:
        args.puzzle_id = f"addition_{args.num1}_{args.num2}"
    
    # 初始化分布式（单GPU）
    RANK = 0
    WORLD_SIZE = 1
    
    if not dist.is_initialized():
        import tempfile
        tmp_file = tempfile.mktemp()
        try:
            if torch.cuda.is_available():
                dist.init_process_group(
                    backend="nccl",
                    init_method=f"file://{tmp_file}",
                    rank=RANK,
                    world_size=WORLD_SIZE,
                )
            else:
                dist.init_process_group(
                    backend="gloo",
                    init_method=f"file://{tmp_file}",
                    rank=RANK,
                    world_size=WORLD_SIZE,
                )
        finally:
            try:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
            except:
                pass
    
    # 加载配置
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    with initialize_config_dir(config_dir=os.path.abspath(args.config_path), version_base=None):
        hydra_config = compose(config_name=args.config_name)
    
    # 设置配置 - 使用addition数据集
    OmegaConf.set_struct(hydra_config, False)
    hydra_config.load_checkpoint = args.checkpoint
    hydra_config.data_paths = ["data/addition"]  # 使用addition数据集
    OmegaConf.set_struct(hydra_config, True)
    
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)
    
    # 加载metadata（先加载metadata以确定正确的网格大小）
    metadata_path = os.path.join(config.data_paths[0], "train", "dataset.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"找不到metadata文件: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata_dict = json.load(f)
    eval_metadata = PuzzleDatasetMetadata(**metadata_dict)
    
    # 根据metadata确定网格列数
    if args.max_len is None:
        args.max_len = eval_metadata.seq_len // 4  # 从metadata读取
        print(f"从metadata读取: seq_len={eval_metadata.seq_len}, 网格列数={args.max_len}")
    
    # 创建加法网格（使用正确的列数）
    grid = create_addition_grid(args.num1, args.num2, args.max_len)
    
    # 显示输入信息
    print("=" * 60)
    print("加法题目测试")
    print("=" * 60)
    print(f"题目: {args.num1} + {args.num2} = {args.num1 + args.num2}")
    print(f"网格大小: {grid.shape[0]}行 × {grid.shape[1]}列")
    visualize_addition_grid(grid, "输入网格（初始状态）")
    print("=" * 60)
    print()
    
    # 创建模型
    model, optimizers, optimizer_lrs = create_model(config, eval_metadata, rank=RANK, world_size=WORLD_SIZE)
    model.eval()
    
    # 获取底层模型
    base_model = model.model if hasattr(model, 'model') else model
    
    # 处理输入：将4行n列的网格转换为序列格式
    # 加法数据集格式：值+1（数字0-9变成1-10），然后展平
    input_grid = grid.copy()
    
    # 展平为一维序列（值+1：数字0-9变成1-10）
    input_seq = (input_grid.flatten() + 1).astype(np.int32)
    
    # 填充到seq_len（如果长度不足，右填充0+1=1）
    seq_len = eval_metadata.seq_len
    if len(input_seq) < seq_len:
        # 右填充0+1=1（因为现在不再使用PAD_VALUE）
        input_seq = np.pad(input_seq, (0, seq_len - len(input_seq)), constant_values=1)
    elif len(input_seq) > seq_len:
        input_seq = input_seq[:seq_len]
    
    # 创建batch
    batch = {
        "inputs": torch.tensor(input_seq, dtype=torch.int32).unsqueeze(0).cuda(),
        "labels": torch.full((1, len(input_seq)), -100, dtype=torch.long).cuda(),
        "puzzle_identifiers": torch.zeros(1, dtype=torch.int32).cuda(),
    }
    
    print("=" * 60)
    print("开始推理...")
    print("=" * 60)
    print(f"注意: 训练数据seq_len={eval_metadata.seq_len}, 测试输入seq_len={len(input_seq)}")
    if len(input_seq) != eval_metadata.seq_len:
        print(f"⚠️  警告: 序列长度不匹配！这可能导致推理错误。")
    print("=" * 60)
    
    # 初始化carry
    with torch.device("cuda"):
        carry = base_model.initial_carry(batch)
        # 重要：初始化时halted=True，需要先设置为False才能开始推理
        carry.halted = torch.zeros_like(carry.halted)
        # 关键：初始化carry.current_data为batch的值（而不是empty_like的未初始化值）
        # 这样第一次迭代时，如果halted=False，会使用正确的初始输入
        carry.current_data = {k: v.clone() for k, v in batch.items()}
        
        # 关键修复：在第一步时，即使halted=False，也要重置inner_carry，避免使用未初始化的值
        # 因为empty_carry创建的是未初始化的tensor，可能包含NaN
        # 在第一步时，所有序列都应该使用初始化的carry值
        if hasattr(base_model, 'inner') and hasattr(base_model.inner, 'reset_carry'):
            # 在第一步时，强制重置所有序列的carry
            reset_all = torch.ones_like(carry.halted, dtype=torch.bool)
            carry.inner_carry = base_model.inner.reset_carry(reset_all, carry.inner_carry)
    
    # 推理循环
    all_predictions = []
    all_q_halt_logits = []
    all_confidence_scores = []
    
    puzzle_emb_len = base_model.inner.puzzle_emb_len if hasattr(base_model.inner, 'puzzle_emb_len') else 16
    
    import torch.nn.functional as F
    
    with torch.no_grad():
        for step in range(args.max_steps):
            carry, outputs = base_model(carry=carry, batch=batch)
            
            # 在评估模式下，模型不会自动根据halt信号停止，需要手动检查
            # 检查halt信号：如果q_halt_logits > 0（对于no_ACT_continue）或满足其他halt条件
            if not base_model.training:
                # 从config获取halt配置
                no_ACT_continue = getattr(config.arch, 'no_ACT_continue', True)
                halt_max_steps = getattr(config.arch, 'halt_max_steps', 16)
                
                # 检查是否达到最大步数
                is_last_step = carry.steps >= halt_max_steps
                
                # 根据halt信号决定是否停止
                q_halt_logits = outputs["q_halt_logits"]
                
                # 检查q_halt_logits是否有NaN
                has_nan = torch.isnan(q_halt_logits).any()
                if has_nan:
                    # 如果有NaN，将NaN位置设置为False（不halt），避免传播
                    q_halt_logits = torch.where(torch.isnan(q_halt_logits), torch.zeros_like(q_halt_logits), q_halt_logits)
                
                if no_ACT_continue:
                    # 如果q_halt_logits > 0，则halt
                    halt_signal = q_halt_logits > 0
                else:
                    # 如果q_halt_logits > q_continue_logits，则halt
                    q_continue_logits = outputs.get("q_continue_logits", torch.zeros_like(q_halt_logits))
                    # 检查q_continue_logits是否有NaN
                    if torch.isnan(q_continue_logits).any():
                        q_continue_logits = torch.where(torch.isnan(q_continue_logits), torch.zeros_like(q_continue_logits), q_continue_logits)
                    halt_signal = q_halt_logits > q_continue_logits
                
                # 确保halt_signal的形状与carry.halted匹配
                if halt_signal.ndim > carry.halted.ndim:
                    halt_signal = halt_signal.squeeze(-1)
                elif halt_signal.ndim < carry.halted.ndim:
                    halt_signal = halt_signal.unsqueeze(-1)
                
                # 确保halt_signal和is_last_step的形状匹配
                if halt_signal.shape != carry.halted.shape:
                    if halt_signal.numel() == 1:
                        halt_signal = halt_signal.expand_as(carry.halted)
                    elif carry.halted.numel() == 1:
                        carry.halted = carry.halted.expand_as(halt_signal)
                
                # 更新halted状态：达到最大步数或halt信号为True
                carry.halted = is_last_step | halt_signal
            
            # 获取q_halt_logits用于后续显示（在评估模式已在上面获取，但这里统一获取以确保可用）
            q_halt_logits = outputs["q_halt_logits"]
            
            logits = outputs["logits"]
            preds = logits.argmax(dim=-1)
            
            # 计算置信度：sigmoid(q_halt_logits)
            confidence = torch.sigmoid(q_halt_logits[0]).item()
            
            pred_seq = preds[0].cpu().numpy()
            
            # 转换为网格（减去1，因为训练时值+1）
            # 数字1-10变成0-9，PAD(11)变成10
            pred_seq = pred_seq - 1
            # 不要clip，保留PAD值（10）
            # pred_seq = np.clip(pred_seq, 0, 9)  # 移除clip，保留PAD值（10）
            
            # 重塑为4行n列（使用训练数据的实际列数）
            num_cols = eval_metadata.seq_len // 4
            expected_size = 4 * num_cols
            if len(pred_seq) >= expected_size:
                pred_grid = pred_seq[:expected_size].reshape(4, num_cols)
            else:
                # 如果长度不足，填充0
                padded = np.pad(pred_seq, (0, expected_size - len(pred_seq)), constant_values=0)
                pred_grid = padded.reshape(4, num_cols)
            
            all_predictions.append(pred_grid.copy())
            all_q_halt_logits.append(q_halt_logits[0].item())
            all_confidence_scores.append(confidence)
            
            print(f"\n步骤 {step + 1}:")
            print(f"  Q_halt logit: {q_halt_logits[0].item():.4f}")
            print(f"  置信度 (sigmoid): {confidence:.4f}")
            print(f"  Halted: {carry.halted[0].item()}")
            visualize_addition_grid(pred_grid, f"步骤 {step + 1} 预测")
            
            # 提取结果（第4行）
            result_digits = pred_grid[3]
            # 去掉前导0
            start_idx = 0
            while start_idx < len(result_digits) and result_digits[start_idx] == 0:
                start_idx += 1
            if start_idx < len(result_digits):
                # 保留有效数字（现在没有PAD值了，都是0-9）
                valid_digits = result_digits[start_idx:]
                if len(valid_digits) > 0:
                    result_str = ''.join(map(str, valid_digits))
                    try:
                        result = int(result_str)
                        expected = args.num1 + args.num2
                        status = "✓" if result == expected else "✗"
                        print(f"  提取的结果: {result} (期望: {expected}) {status}")
                    except:
                        pass
            
            # 检查是否应该停止：置信度足够高或模型halted
            if carry.halted.all():
                print(f"\n模型在第 {step + 1} 步halt")
                break
            
            # 关键修复：更新carry.current_data为当前预测结果，用于下一步的递归推理
            # 在递归推理中，每一步的输入应该是上一步的输出
            # 模型在第256行使用：new_current_data = {k: torch.where(..., batch[k], v) for k, v in carry.current_data.items()}
            # 如果序列halted，使用batch[k]；如果未halted，使用carry.current_data中的值v
            # 所以我们需要将预测结果更新到carry.current_data中，这样下一步才能使用上一步的输出
            if step < args.max_steps - 1:
                # 将预测结果转换回模型输入格式（值+1：数字0-9变成1-10）
                next_input_seq = (pred_grid.flatten() + 1).astype(np.int32)
                if len(next_input_seq) < eval_metadata.seq_len:
                    # 右填充0+1=1（因为现在不再使用PAD_VALUE）
                    next_input_seq = np.pad(next_input_seq, (0, eval_metadata.seq_len - len(next_input_seq)), constant_values=1)
                elif len(next_input_seq) > eval_metadata.seq_len:
                    next_input_seq = next_input_seq[:eval_metadata.seq_len]
                
                # 更新carry.current_data（这是关键！）
                # 模型在下一步会使用carry.current_data作为输入（如果序列未halted）
                next_input_tensor = torch.tensor(next_input_seq, dtype=torch.int32).unsqueeze(0).cuda()
                carry.current_data["inputs"] = next_input_tensor.clone()  # 使用clone确保是新的tensor
                # 同时更新batch，确保一致性（虽然模型优先使用carry.current_data）
                batch["inputs"] = next_input_tensor.clone()
    
    # 显示最终结果
    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)
    
    if all_predictions:
        final_pred = all_predictions[-1]
        visualize_addition_grid(final_pred, "最终预测网格")
        
        # 提取最终结果
        result_digits = final_pred[3]
        # 去掉前导0
        start_idx = 0
        while start_idx < len(result_digits) and result_digits[start_idx] == 0:
            start_idx += 1
        if start_idx < len(result_digits):
            # 保留有效数字（现在没有PAD值了，都是0-9）
            valid_digits = result_digits[start_idx:]
            if len(valid_digits) > 0:
                result_str = ''.join(map(str, valid_digits))
                try:
                    result = int(result_str)
                    expected = args.num1 + args.num2
                    print(f"\n最终结果: {result}")
                    print(f"期望结果: {expected}")
                    print(f"是否正确: {'✓ 正确' if result == expected else '✗ 错误'}")
                except:
                    print(f"\n无法解析结果: {result_digits}")
            else:
                print(f"\n无法解析结果: 没有有效数字")
        else:
            print(f"\n无法解析结果: 结果全为0")
        
        print(f"\n推理统计:")
        print(f"  总步数: {len(all_predictions)}")
        print(f"  最终Q_halt logit: {all_q_halt_logits[-1]:.4f}")
        print(f"  最终置信度: {all_confidence_scores[-1]:.4f}")
        
        # 显示所有步骤的Q值和置信度变化
        print(f"\nQ值和置信度变化:")
        for i, (q_halt, confidence) in enumerate(zip(all_q_halt_logits, all_confidence_scores)):
            threshold_mark = "✓" if confidence >= args.confidence_threshold else ""
            print(f"  步骤 {i+1}: Q_halt={q_halt:7.4f}, 置信度={confidence:.4f} {threshold_mark}")
    
    # 清理
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
