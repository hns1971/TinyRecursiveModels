from typing import Optional, List, Tuple
import os
import json
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

from dataset.common import PuzzleDatasetMetadata

# 前导词：使用10表示前导位置和未计算位置，以区分数字0和前导/未计算位置
LEADING_VALUE = 10
# PAD值：使用11表示PAD，用于batch填充
PAD_VALUE = 11


cli = ArgParser()


class DataProcessConfig(BaseModel):
    output_dir: str = "data/addition"
    
    # 训练集和测试集大小
    train_size: int = 10000
    test_size: int = 1000
    
    # 数字位数范围
    min_digits: int = 2
    max_digits: int = 10
    
    # 数据增强（交换两个加数的顺序）
    num_aug: int = 1


def number_to_digits(num: int, max_len: int, actual_digits: int, pad_to_len: Optional[int] = None, use_leading_pad: bool = True) -> np.ndarray:
    """将数字转换为数字数组，右对齐（个位在最右边）
    
    Args:
        num: 要转换的数字
        max_len: 最大长度（网格宽度）
        actual_digits: 数字的实际位数（不包括前导0）
        pad_to_len: 先用0补齐到的长度（如果为None，则直接用前导词补齐到max_len）
        use_leading_pad: 如果True，前导位置用LEADING_VALUE补齐；如果False，用PAD_VALUE补齐
    
    Returns:
        数字数组，右对齐
    """
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


def generate_addition_puzzle(num1: int, num2: int, max_len: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    生成一个加法谜题及其所有中间步骤
    
    Args:
        num1: 第一个加数
        num2: 第二个加数
        max_len: 最大位数（网格的列数）
    
    Returns:
        input_grid: 4行max_len列的输入网格
        step_grids: 每一步的中间状态网格列表
    """
    # 计算结果
    result = num1 + num2
    
    # 计算两个数的实际位数（不考虑前导0）
    def count_digits(n: int) -> int:
        if n == 0:
            return 1
        count = 0
        while n > 0:
            count += 1
            n //= 10
        return count
    
    digits1_count = count_digits(num1)
    digits2_count = count_digits(num2)
    actual_max_digits = max(digits1_count, digits2_count)
    
    # 确定补齐规则：
    # 1. 如果两个数位数不同，位数少的先用0补齐到与多的位数一样
    # 2. 为避免矛盾，两个数前面多补一个0，用于计算
    # 所以 pad_to_len = actual_max_digits + 1
    pad_to_len = actual_max_digits + 1
    
    # 步数：补齐后的位数（pad_to_len）
    # 如果最高位计算完后还有进位，会在循环后额外添加一步
    
    # 转换为数字数组（右对齐，个位在最后）
    # 如果两个数位数不同，位数小的先用0补齐到与多的位数一样，然后再用前导词补齐到max_len
    # 两个数前面多补一个0（用于计算），这个0应该用前导词填充
    digits1 = number_to_digits(num1, max_len, digits1_count, pad_to_len=pad_to_len, use_leading_pad=True)
    digits2 = number_to_digits(num2, max_len, digits2_count, pad_to_len=pad_to_len, use_leading_pad=True)
    result_digits = number_to_digits(result, max_len + 1, count_digits(result), use_leading_pad=True)  # 结果可能多一位
    
    # 初始化网格：4行 x max_len列（序列长度固定为48=4×12）
    # 第1行：第一个加数
    # 第2行：第二个加数
    # 第3行：进位数
    # 第4行：结果
    grid_width = max_len  # 固定为max_len，而不是max_len+1
    
    # 输入网格：前两行有数字，后两行用前导词填充（初始状态）
    # 前导位置用LEADING_VALUE补齐，未计算位置也用LEADING_VALUE表示
    input_grid = np.full((4, grid_width), LEADING_VALUE, dtype=np.uint8)
    input_grid[0, :max_len] = digits1  # 前导位置已经是LEADING_VALUE
    input_grid[1, :max_len] = digits2  # 前导位置已经是LEADING_VALUE
    # 第3行和第4行保持为LEADING_VALUE（初始状态，表示未计算）
    
    # 生成所有中间步骤
    step_grids = []
    carry_input = 0  # 当前位的进位输入（从右边一位来的）
    # 初始化进位行和结果行为全LEADING_VALUE，表示未计算
    carry_row = np.full(grid_width, LEADING_VALUE, dtype=np.uint8)  # LEADING_VALUE表示未计算
    result_row = np.full(grid_width, LEADING_VALUE, dtype=np.uint8)  # LEADING_VALUE表示未计算
    
    # 从右到左（个位到最高位）逐步计算
    # digits1和digits2是右对齐的：索引0是最高位，索引max_len-1是个位
    # 所以从max_len-1开始倒序计算
    # 计算所有有效位（pad_to_len个位置）
    for i in range(pad_to_len):
        pos = max_len - 1 - i  # 从最右边（个位）开始，pos递减
        
        # 计算当前位置的和
        # 检查是否为前导位置（前导0）
        # 前导位置：pos < (max_len - pad_to_len)
        is_leading_zero = pos < (max_len - pad_to_len)
        
        if is_leading_zero:
            # 前导位置保持为0，不参与计算
            # 结果行和进位行都保持为0
            # 前导位置不生成步骤，跳过
            continue
        else:
            # 正常计算位置，参与计算
            # 前导位置是0，正常位置是数字0-9
            d1 = digits1[pos]
            d2 = digits2[pos]
            s = d1 + d2 + carry_input
            
            # 当前位的结果和新的进位输出
            result_digit = s % 10
            carry_output = s // 10  # 这个进位会传递给左边一位
            
            # 更新进位行和结果行
            # 第3行显示传递给下一位的进位输入（当前位产生的进位，显示在左边一位的位置）
            # 注意：在递归推理中，需要保留之前位的进位信息，所以不清除进位行
            # 只更新当前位产生的进位（显示在左边一位的位置）
            # 如果当前位不是最高位（pos > 0），将进位显示在左边一位的位置
            # 如果当前位是最高位（pos == 0），不再处理进位（因为已经超出网格范围，前导位置不应该有进位）
            if pos > 0:  # 如果还有左边一位，将进位显示在左边一位的位置
                left_pos = pos - 1
                # 检查左边一位是否是前导位置
                is_left_leading = left_pos < (max_len - pad_to_len)
                if not is_left_leading:
                    # 左边一位不是前导位置，可以显示进位
                    carry_row[left_pos] = carry_output  # 进位显示在左边一位的位置（传递给下一位的进位输入）
                # 如果左边一位是前导位置，不显示进位（保持为LEADING_VALUE）
            # 如果当前位是最高位（pos == 0），即使有进位也不显示（因为前导位置不应该有进位）
            result_row[pos] = result_digit
            
            # 更新进位输入，用于左边一位的计算
            carry_input = carry_output
            
            # 创建当前步骤的网格（只在有实际计算时创建）
            # 前导位置保持为LEADING_VALUE，未计算位置也保持为LEADING_VALUE
            step_grid = np.full((4, grid_width), LEADING_VALUE, dtype=np.uint8)
            step_grid[0, :max_len] = digits1  # 第一个加数（前导位置已经是LEADING_VALUE）
            step_grid[1, :max_len] = digits2  # 第二个加数（前导位置已经是LEADING_VALUE）
            
            # 确保进位行的前导位置保持为LEADING_VALUE
            # 进位行的最右边（个位位置）用PAD_VALUE填充，因为个位没有来自右边的进位输入
            leading_start = max_len - pad_to_len
            final_carry_row = carry_row.copy()
            final_carry_row[:leading_start] = LEADING_VALUE  # 前导位置保持为LEADING_VALUE
            final_carry_row[max_len - 1] = PAD_VALUE  # 最右边（个位位置）用PAD填充，因为个位没有来自右边的进位
            
            step_grid[2, :] = final_carry_row  # 进位行（前导位置和个位位置是LEADING_VALUE/PAD，未计算位置是LEADING_VALUE）
            step_grid[3, :] = result_row.copy()  # 结果行（未计算位置是LEADING_VALUE）
            
            step_grids.append(step_grid)
    
    # 注意：最高位（前导0所在位）计算完后，不再处理进位位
    # 如果最高位计算完后还有进位，直接忽略（因为已经超出了网格范围）
    
    
    return input_grid, step_grids


def generate_random_addition(min_digits: int, max_digits: int) -> Tuple[int, int, int]:
    """生成随机加法题目
    
    Args:
        min_digits: 最小位数
        max_digits: 最大位数（网格宽度），但实际生成的最大位数应该是 max_digits - 1
                    以防止两个 max_digits-1 位数相加结果超过 max_digits 位
    
    Returns:
        num1: 第一个加数
        num2: 第二个加数
        max_len: 需要的最大位数（用于确定网格宽度，等于 max_digits）
    """
    # 实际生成的最大位数应该是 max_digits - 1，以防止结果超出网格
    # 例如：如果 max_digits=4，网格是4列，那么两个3位数相加最多是4位数，不会超出
    actual_max_digits = max_digits - 1
    
    # 确保 actual_max_digits 不小于 min_digits
    if actual_max_digits < min_digits:
        actual_max_digits = min_digits
    
    # 使用循环而不是递归，避免无限递归
    max_attempts = 100
    for attempt in range(max_attempts):
        # 生成随机位数（在 min_digits 和 actual_max_digits 之间）
        num_digits = np.random.randint(min_digits, actual_max_digits + 1)
        
        # 生成随机数字
        min_val = 10 ** (num_digits - 1) if num_digits > 1 else 0
        max_val = 10 ** num_digits - 1
        
        num1 = np.random.randint(min_val, max_val + 1)
        num2 = np.random.randint(min_val, max_val + 1)
        
        # 验证结果不会超出 max_digits 位
        result = num1 + num2
        result_digits = len(str(result))
        
        # 如果结果在范围内，返回
        if result_digits <= max_digits:
            # max_len 始终等于 max_digits（网格宽度）
            max_len = max_digits
            return num1, num2, max_len
    
    # 如果尝试多次都失败（理论上不应该发生），使用更保守的策略
    # 生成更小的数字
    num_digits = min_digits
    min_val = 10 ** (num_digits - 1) if num_digits > 1 else 0
    max_val = 10 ** num_digits - 1
    
    num1 = np.random.randint(min_val, max_val + 1)
    num2 = np.random.randint(min_val, max_val + 1)
    max_len = max_digits
    
    return num1, num2, max_len


def convert_subset(set_name: str, config: DataProcessConfig):
    """生成训练集或测试集"""
    size = config.train_size if set_name == "train" else config.test_size
    num_augments = config.num_aug if set_name == "train" else 0
    
    results = {
        "inputs": [],
        "labels": [],
        "puzzle_identifiers": [],
        "puzzle_indices": [],
        "group_indices": [],
        "step_counts": []  # 记录每道题目的实际步骤数
    }
    
    puzzle_id = 0
    example_id = 0
    
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)
    
    # 统一使用最大位数作为网格宽度
    fixed_max_len = config.max_digits
    
    # 生成数据
    for _ in tqdm(range(size), desc=f"生成{set_name}集"):
        # 生成随机加法题目
        num1, num2, _ = generate_random_addition(config.min_digits, config.max_digits)
        
        # 生成所有可能的增强（交换顺序）
        pairs = [(num1, num2)]
        if num_augments > 0:
            pairs.append((num2, num1))  # 交换顺序
        
        for aug_idx, (n1, n2) in enumerate(pairs[:1 + num_augments]):
            # 生成谜题和所有步骤（使用固定的max_len）
            input_grid, step_grids = generate_addition_puzzle(n1, n2, fixed_max_len)
            
            # 构建完整轨迹：s₀（初始状态）, s₁, s₂, ..., sₜ（最终状态）
            # input_grid 是 s₀（初始状态）
            # step_grids 是 [s₁, s₂, ..., sₜ]（每一步的状态）
            
            # 修改：将一道题目的所有中间步骤合并到一条数据中
            # 输入：初始状态 s₀
            # 标签：所有中间步骤的序列拼接 [s₁, s₂, ..., sₜ]
            if len(step_grids) > 0:
                # 将所有中间步骤的状态按顺序拼接成一个长序列
                all_steps_flat = [step_grid.flatten() for step_grid in step_grids]
                combined_label = np.concatenate(all_steps_flat)
                
                results["inputs"].append(input_grid.flatten())
                results["labels"].append(combined_label)
                results["step_counts"].append(len(step_grids))  # 记录实际步骤数
                
                example_id += 1
                results["puzzle_indices"].append(example_id)
                results["puzzle_identifiers"].append(puzzle_id)
            
            puzzle_id += 1
        
        # 每个puzzle的所有步骤组成一个group
        results["group_indices"].append(puzzle_id)
    
    # 转换为numpy数组
    # 找到最大序列长度
    max_seq_len = max(
        max(len(inp) for inp in results["inputs"]),
        max(len(lab) for lab in results["labels"])
    )
    
    # 计算每个步骤的大小（4行×grid_width列）
    step_size = 4 * fixed_max_len
    
    def _pad_sequences(seq_list, step_counts_list, pad_value=PAD_VALUE):
        """填充序列到相同长度
        对于标签序列：如果题目已完成，后续步骤应该复制最终状态，而不是用PAD填充
        这样可以避免在批量训练时，已完成题目的损失计算错误
        """
        padded = []
        for seq, step_count in zip(seq_list, step_counts_list):
            if len(seq) < max_seq_len:
                # 计算需要填充的长度
                pad_len = max_seq_len - len(seq)
                
                # 对于标签序列：如果题目已完成，后续步骤应该复制最终状态
                # 最终状态是最后一个步骤（step_count - 1）
                if step_count > 0:
                    # 获取最终状态（最后一个步骤）
                    final_step_start = (step_count - 1) * step_size
                    final_step_end = step_count * step_size
                    final_step = seq[final_step_start:final_step_end]
                    
                    # 计算需要复制多少个完整步骤
                    num_full_steps = pad_len // step_size
                    remaining = pad_len % step_size
                    
                    # 复制最终状态
                    pad_parts = []
                    for _ in range(num_full_steps):
                        pad_parts.append(final_step)
                    if remaining > 0:
                        pad_parts.append(final_step[:remaining])
                    
                    if pad_parts:
                        padded_seq = np.concatenate([seq] + pad_parts)
                    else:
                        padded_seq = seq
                else:
                    # 如果没有步骤，用PAD填充
                    pad = np.full(pad_len, pad_value, dtype=np.uint8)
                    padded_seq = np.concatenate([seq, pad])
            else:
                padded_seq = seq[:max_seq_len]
            padded.append(padded_seq)
        return np.array(padded, dtype=np.uint8)
    
    # 转换为numpy数组（值+1：数字0-9变成1-10，LEADING_VALUE(10)变成11，PAD_VALUE(11)变成12）
    results_numpy = {
        "inputs": _pad_sequences(results["inputs"], [0] * len(results["inputs"]), pad_value=PAD_VALUE) + 1,
        "labels": _pad_sequences(results["labels"], results["step_counts"], pad_value=PAD_VALUE) + 1,
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }
    
    # 元数据
    # num_puzzle_identifiers应该是puzzle_identifiers的最大值+1
    # 因为puzzle_identifiers的范围是[0, puzzle_id-1]
    max_puzzle_identifier = max(results_numpy["puzzle_identifiers"]) if len(results_numpy["puzzle_identifiers"]) > 0 else 0
    num_puzzle_identifiers = max_puzzle_identifier + 1
    
    # 计算平均步骤数（实际步骤数，不是样本数）
    mean_steps = np.mean(results["step_counts"]) if len(results["step_counts"]) > 0 else 0
    
    metadata = PuzzleDatasetMetadata(
        seq_len=max_seq_len,
        vocab_size=13,  # 数字1-10（值+1后的0-9） + LEADING(11，值+1后的10) + PAD(12，值+1后的11)
        pad_id=12,  # PAD_VALUE(11)值+1后变成12
        ignore_label_id=12,  # PAD应该被忽略
        blank_identifier_id=0,
        num_puzzle_identifiers=num_puzzle_identifiers,  # 使用实际的puzzle identifier数量
        total_groups=len(results_numpy["group_indices"]) - 1,
        mean_puzzle_examples=mean_steps,  # 使用实际平均步骤数
        total_puzzles=puzzle_id,
        sets=["all"]
    )
    
    # 保存数据
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
    
    for k, v in results_numpy.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
    
    # 保存标识符映射
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)
    
    print(f"✅ {set_name}集生成完成:")
    print(f"   - 总样本数: {len(results_numpy['inputs'])}")
    print(f"   - 总puzzle数: {puzzle_id}")
    print(f"   - 序列长度: {max_seq_len}")
    print(f"   - 平均每个puzzle的步骤数: {mean_steps:.2f}")
    if len(results["step_counts"]) > 0:
        print(f"   - 步骤数范围: {min(results['step_counts'])} - {max(results['step_counts'])}")


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    """生成加法数据集"""
    print("=" * 60)
    print("生成任意数加法训练数据集")
    print("=" * 60)
    print(f"输出目录: {config.output_dir}")
    print(f"训练集大小: {config.train_size}")
    print(f"测试集大小: {config.test_size}")
    print(f"数字位数范围: {config.min_digits}-{config.max_digits}")
    print(f"数据增强倍数: {config.num_aug + 1}")
    print("=" * 60)
    
    convert_subset("train", config)
    convert_subset("test", config)
    
    print("\n🎉 数据集生成完成！")


if __name__ == "__main__":
    cli()

