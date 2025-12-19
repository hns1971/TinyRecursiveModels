#!/usr/bin/env python3
"""
extract_result_from_grid 函数的单元测试

使用方法:
    pytest test_extract_result_from_grid.py -v
    或
    python test_extract_result_from_grid.py
"""

import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 优先尝试导入原函数和常量（测试应该测试实际的实现）
try:
    from test_addition_puzzle import extract_result_from_grid, LEADING_VALUE, PAD_VALUE
    USING_ORIGINAL_FUNCTION = True
except ImportError as e:
    # 如果导入失败（例如缺少依赖），使用本地实现作为回退
    # 注意：这确保了测试可以独立运行，但理想情况下应该安装依赖以测试实际实现
    import warnings
    warnings.warn(
        f"无法导入原函数（原因：{e}），使用本地实现。"
        f"建议安装依赖（omegaconf, hydra等）以测试实际实现。",
        ImportWarning
    )
    USING_ORIGINAL_FUNCTION = False
    
    # 定义常量（与 test_addition_puzzle.py 中一致）
    LEADING_VALUE = 10
    PAD_VALUE = 11
    
    # 本地实现（与原函数完全一致）
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


def test_extract_result_normal_case():
    """测试正常情况：有有效数字"""
    # 创建测试网格：4行11列
    grid = np.full((4, 11), LEADING_VALUE, dtype=np.uint8)
    # 第4行（索引3）：前导LEADING_VALUE + 结果 579
    grid[3, 8:11] = [5, 7, 9]  # 右对齐，579在最后3列
    
    result = extract_result_from_grid(grid)
    assert result == 579, f"期望 579，实际得到 {result}"


def test_extract_result_with_leading_zeros():
    """测试前导0的情况"""
    grid = np.full((4, 11), LEADING_VALUE, dtype=np.uint8)
    # 第4行：前导LEADING_VALUE + 前导0 + 结果 123
    grid[3, 6:11] = [0, 0, 1, 2, 3]  # 前导0应该被跳过
    
    result = extract_result_from_grid(grid)
    assert result == 123, f"期望 123，实际得到 {result}"


def test_extract_result_with_leading_pad_values():
    """测试前导PAD_VALUE的情况"""
    grid = np.full((4, 11), PAD_VALUE, dtype=np.uint8)
    # 第4行：前导PAD_VALUE + 结果 456
    grid[3, 8:11] = [4, 5, 6]
    
    result = extract_result_from_grid(grid)
    assert result == 456, f"期望 456，实际得到 {result}"


def test_extract_result_mixed_leading_values():
    """测试混合前导值（LEADING_VALUE、PAD_VALUE和0）"""
    grid = np.full((4, 11), LEADING_VALUE, dtype=np.uint8)
    # 第4行：LEADING_VALUE + PAD_VALUE + 0 + 0 + 结果 789
    grid[3, 0:3] = [LEADING_VALUE, PAD_VALUE, 0]
    grid[3, 3:7] = [0, 7, 8, 9]
    
    result = extract_result_from_grid(grid)
    assert result == 789, f"期望 789，实际得到 {result}"


def test_extract_result_with_middle_pad_values():
    """测试中间有PAD_VALUE的情况（应该被跳过）"""
    grid = np.full((4, 11), LEADING_VALUE, dtype=np.uint8)
    # 第4行：前导LEADING_VALUE + 结果 12 + PAD_VALUE + 34
    grid[3, 6:11] = [1, 2, PAD_VALUE, 3, 4]
    
    result = extract_result_from_grid(grid)
    assert result == 1234, f"期望 1234，实际得到 {result}"


def test_extract_result_single_digit():
    """测试单个数字"""
    grid = np.full((4, 11), LEADING_VALUE, dtype=np.uint8)
    # 第4行：只有最后一个位置是有效数字
    grid[3, 10] = 5
    
    result = extract_result_from_grid(grid)
    assert result == 5, f"期望 5，实际得到 {result}"


def test_extract_result_all_leading_values():
    """测试全为LEADING_VALUE的情况（应该返回None）"""
    grid = np.full((4, 11), LEADING_VALUE, dtype=np.uint8)
    
    result = extract_result_from_grid(grid)
    assert result is None, f"期望 None，实际得到 {result}"


def test_extract_result_all_pad_values():
    """测试全为PAD_VALUE的情况（应该返回None）"""
    grid = np.full((4, 11), PAD_VALUE, dtype=np.uint8)
    
    result = extract_result_from_grid(grid)
    assert result is None, f"期望 None，实际得到 {result}"


def test_extract_result_all_zeros():
    """测试全为0的情况（应该返回0，因为0是有效的计算结果）"""
    grid = np.zeros((4, 11), dtype=np.uint8)
    
    result = extract_result_from_grid(grid)
    assert result == 0, f"期望 0，实际得到 {result}"


def test_extract_result_with_zero_in_middle():
    """测试中间有0的情况（0应该被保留）"""
    grid = np.full((4, 11), LEADING_VALUE, dtype=np.uint8)
    # 第4行：结果 10203（中间有0）
    grid[3, 6:11] = [1, 0, 2, 0, 3]
    
    result = extract_result_from_grid(grid)
    assert result == 10203, f"期望 10203，实际得到 {result}"


def test_extract_result_different_row_index():
    """测试从不同行提取结果"""
    grid = np.full((4, 11), LEADING_VALUE, dtype=np.uint8)
    # 第3行（索引2）：结果 999
    grid[2, 8:11] = [9, 9, 9]
    # 第4行（索引3）：结果 111
    grid[3, 8:11] = [1, 1, 1]
    
    # 从第3行提取
    result = extract_result_from_grid(grid, row_index=2)
    assert result == 999, f"期望 999，实际得到 {result}"
    
    # 从第4行提取
    result = extract_result_from_grid(grid, row_index=3)
    assert result == 111, f"期望 111，实际得到 {result}"


def test_extract_result_large_number():
    """测试大数字"""
    grid = np.full((4, 11), LEADING_VALUE, dtype=np.uint8)
    # 第4行：结果 123456789（9位数字）
    grid[3, 2:11] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    result = extract_result_from_grid(grid)
    assert result == 123456789, f"期望 123456789，实际得到 {result}"


def test_extract_result_with_invalid_values():
    """测试包含无效值（>9）的情况（应该被跳过）"""
    grid = np.full((4, 11), LEADING_VALUE, dtype=np.uint8)
    # 第4行：包含无效值12和13（应该被跳过）
    grid[3, 6:11] = [1, 2, 12, 3, 13]  # 12和13应该被跳过
    
    result = extract_result_from_grid(grid)
    assert result == 123, f"期望 123，实际得到 {result}"


def test_extract_result_empty_valid_digits():
    """测试没有有效数字的情况（只有LEADING_VALUE、PAD_VALUE和0）"""
    grid = np.full((4, 11), LEADING_VALUE, dtype=np.uint8)
    # 第4行：只有前导0和PAD_VALUE
    grid[3, 8:11] = [0, PAD_VALUE, 0]
    
    result = extract_result_from_grid(grid)
    assert result is None, f"期望 None，实际得到 {result}"


def test_extract_result_complex_case():
    """测试复杂情况：混合各种值"""
    grid = np.full((4, 11), LEADING_VALUE, dtype=np.uint8)
    # 第4行：LEADING_VALUE + PAD_VALUE + 0 + 0 + 1 + 2 + PAD_VALUE + 3 + 4 + LEADING_VALUE + 5
    grid[3, 0:2] = [LEADING_VALUE, PAD_VALUE]
    grid[3, 2:4] = [0, 0]
    grid[3, 4:7] = [1, 2, PAD_VALUE]
    grid[3, 7:9] = [3, 4]
    grid[3, 9] = LEADING_VALUE
    grid[3, 10] = 5
    
    result = extract_result_from_grid(grid)
    assert result == 12345, f"期望 12345，实际得到 {result}"


def test_extract_result_with_trailing_pad():
    """测试末尾有PAD_VALUE的情况（应该被跳过）"""
    grid = np.full((4, 11), LEADING_VALUE, dtype=np.uint8)
    # 第4行：结果 123 + 末尾PAD_VALUE
    grid[3, 7:10] = [1, 2, 3]
    grid[3, 10] = PAD_VALUE
    
    result = extract_result_from_grid(grid)
    assert result == 123, f"期望 123，实际得到 {result}"


def test_extract_result_with_trailing_leading():
    """测试末尾有LEADING_VALUE的情况（应该被跳过）"""
    grid = np.full((4, 11), LEADING_VALUE, dtype=np.uint8)
    # 第4行：结果 456 + 末尾LEADING_VALUE
    grid[3, 7:10] = [4, 5, 6]
    # 第10列已经是LEADING_VALUE（默认值）
    
    result = extract_result_from_grid(grid)
    assert result == 456, f"期望 456，实际得到 {result}"


def run_tests():
    """运行所有测试"""
    # 显示使用的函数版本
    if USING_ORIGINAL_FUNCTION:
        print("=" * 60)
        print("✓ 使用原函数（test_addition_puzzle.extract_result_from_grid）")
        print("=" * 60)
    else:
        print("=" * 60)
        print("⚠ 使用本地实现（回退方案）")
        print("  建议安装依赖以测试实际实现")
        print("=" * 60)
    print()
    
    test_functions = [
        test_extract_result_normal_case,
        test_extract_result_with_leading_zeros,
        test_extract_result_with_leading_pad_values,
        test_extract_result_mixed_leading_values,
        test_extract_result_with_middle_pad_values,
        test_extract_result_single_digit,
        test_extract_result_all_leading_values,
        test_extract_result_all_pad_values,
        test_extract_result_all_zeros,
        test_extract_result_with_zero_in_middle,
        test_extract_result_different_row_index,
        test_extract_result_large_number,
        test_extract_result_with_invalid_values,
        test_extract_result_empty_valid_digits,
        test_extract_result_complex_case,
        test_extract_result_with_trailing_pad,
        test_extract_result_with_trailing_leading,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__}: 异常 - {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
