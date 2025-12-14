#!/usr/bin/env python3
"""独立测试加法数据生成逻辑（不依赖外部模块）"""

import numpy as np
from typing import Tuple, List


def number_to_digits(num: int, max_len: int) -> np.ndarray:
    """将数字转换为数字数组，右对齐（个位在最右边）"""
    digits = []
    if num == 0:
        digits = [0]
    else:
        while num > 0:
            digits.append(num % 10)
            num //= 10
    # 右对齐：左填充0（个位在数组末尾，即网格最右边）
    while len(digits) < max_len:
        digits.append(0)
    # 反转数组，使得个位在最后（最右边）
    digits = digits[::-1]
    return np.array(digits[:max_len], dtype=np.uint8)


def generate_addition_puzzle(num1: int, num2: int, max_len: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    """生成一个加法谜题及其所有中间步骤"""
    result = num1 + num2
    digits1 = number_to_digits(num1, max_len)
    digits2 = number_to_digits(num2, max_len)
    grid_width = max_len + 1
    
    input_grid = np.zeros((4, grid_width), dtype=np.uint8)
    input_grid[0, :max_len] = digits1
    input_grid[1, :max_len] = digits2
    
    step_grids = []
    carry_input = 0
    carry_row = np.zeros(grid_width, dtype=np.uint8)
    result_row = np.zeros(grid_width, dtype=np.uint8)
    
    for i in range(max_len):
        pos = max_len - 1 - i
        d1 = digits1[pos]
        d2 = digits2[pos]
        s = d1 + d2 + carry_input
        
        result_digit = s % 10
        carry_output = s // 10
        
        carry_row[pos] = carry_input
        result_row[pos] = result_digit
        
        step_grid = np.zeros((4, grid_width), dtype=np.uint8)
        step_grid[0, :max_len] = digits1
        step_grid[1, :max_len] = digits2
        step_grid[2, :] = carry_row.copy()
        step_grid[3, :] = result_row.copy()
        
        step_grids.append(step_grid)
        carry_input = carry_output
    
    # 处理最高位的进位（如果有）
    # 注意：如果最后还有进位，说明结果需要多一位
    # 此时需要将之前的结果向右移动，新进位放在最左边
    if carry_input > 0:
        new_carry_row = np.zeros(grid_width, dtype=np.uint8)
        new_result_row = np.zeros(grid_width, dtype=np.uint8)
        new_carry_row[0] = 0  # 最高位的进位输入是0（这是新的一位）
        new_result_row[0] = carry_input  # 最高位的结果就是进位值
        # 将之前的结果向右移动一位
        # result_row的有效部分是[:max_len]，需要移动到[1:max_len+1]
        # 但要注意：result_row的长度是grid_width，所以result_row[:max_len]是正确的
        new_result_row[1:max_len+1] = result_row[:max_len]
        
        step_grid = np.zeros((4, grid_width), dtype=np.uint8)
        step_grid[0, :max_len] = digits1
        step_grid[1, :max_len] = digits2
        step_grid[2, :] = new_carry_row.copy()
        step_grid[3, :] = new_result_row.copy()
        step_grids.append(step_grid)
    
    return input_grid, step_grids


def test_simple():
    print("测试 123 + 456:")
    num1, num2 = 123, 456
    max_len = max(len(str(num1)), len(str(num2)), len(str(num1 + num2)))
    input_grid, step_grids = generate_addition_puzzle(num1, num2, max_len)
    
    print(f"  结果应该是: {num1 + num2} = 579")
    print(f"  生成了 {len(step_grids)} 个步骤")
    
    final_step = step_grids[-1]
    result_digits = final_step[3][:max_len]
    result = int(''.join(map(str, result_digits)))
    print(f"  提取的结果: {result}")
    assert result == 579, f"应该是579，但得到{result}"
    print("  ✅ 通过\n")


def test_carry():
    print("测试 999 + 1:")
    num1, num2 = 999, 1
    max_len = max(len(str(num1)), len(str(num2)), len(str(num1 + num2)))
    input_grid, step_grids = generate_addition_puzzle(num1, num2, max_len)
    
    print(f"  结果应该是: {num1 + num2} = 1000")
    print(f"  生成了 {len(step_grids)} 个步骤")
    print(f"  max_len = {max_len}, grid_width = {max_len + 1}")
    
    final_step = step_grids[-1]
    print(f"  最后一步的结果行: {final_step[3]}")
    print(f"  最后一步的进位行: {final_step[2]}")
    
    # 提取有效结果：只取前max_len位（因为max_len已经考虑了结果可能的位数）
    # 如果结果真的需要多一位，max_len应该已经包含了这一位
    result_digits = final_step[3][:max_len]
    
    # 去掉前导零
    start_idx = 0
    while start_idx < len(result_digits) and result_digits[start_idx] == 0:
        start_idx += 1
    if start_idx < len(result_digits):
        result_str = ''.join(map(str, result_digits[start_idx:]))
        result = int(result_str)
        print(f"  提取的结果: {result} (从索引{start_idx}开始)")
        assert result == 1000, f"应该是1000，但得到{result}"
        print("  ✅ 通过\n")
    else:
        print("  ❌ 未找到有效结果\n")


if __name__ == "__main__":
    print("=" * 60)
    test_simple()
    test_carry()
    print("=" * 60)
    print("所有测试通过!")

