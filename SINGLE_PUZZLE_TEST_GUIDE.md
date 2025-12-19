# 单题测试指南

本指南介绍如何使用 `test_single_puzzle.py` 脚本单独测试某道题，查看完整的解题过程。

## 快速开始

### 基本用法

```bash
python3 test_single_puzzle.py \
  --checkpoint checkpoints/Arc1concept-aug-1000-ACT-torch/my_training_run/step_303191 \
  --input "[[6,8,6,8,6,3],[8,6,8,6,8,3],[6,8,6,8,6,3],[8,6,8,6,8,3],[6,8,6,8,6,3],[3,3,3,3,3,3]]" \
  --puzzle_id test_puzzle \
  --output_dir my_visualizations
```

### 从文件加载题目

创建一个JSON文件 `test_input.json`:
```json
[[0, 1, 2], [3, 4, 5], [6, 7, 8]]
```

然后运行：
```bash
python test_single_puzzle.py \
  --checkpoint checkpoints/Arc1concept-aug-1000-ACT-torch/my_training_run/step_303191 \
  --input test_input.json
```

## 参数说明

### 必需参数

- `--checkpoint`: Checkpoint文件路径
  - 示例: `checkpoints/Arc1concept-aug-1000-ACT-torch/my_training_run/step_303191`

### 可选参数

- `--input`: 输入网格
  - 可以是JSON文件路径或JSON字符串
  - 格式: 2D数组，值范围0-9
  - 示例: `"[[0,1,2],[3,4,5]]"` 或 `test_input.json`
  - 如果不提供，会使用默认示例

- `--puzzle_id`: Puzzle标识符（默认: `test_puzzle`）

- `--max_steps`: 最大推理步数（默认: 16）

- `--config_path`: 配置文件目录（默认: `config`）

- `--config_name`: 配置文件名（默认: `cfg_pretrain`）

## 输出说明

脚本会显示：

1. **题目信息**:
   - Puzzle ID
   - 输入网格大小
   - 输入网格可视化

2. **推理过程**（每一步）:
   - Q_halt logit: halt的Q值
   - Q_continue logit: continue的Q值
   - Halted: 是否halt
   - Steps: 当前步数
   - 前3步的预测网格可视化

3. **最终结果**:
   - 最终预测网格
   - 推理统计（总步数、Q值等）
   - 所有步骤的Q值变化

## 示例输出

```
============================================================
单题测试
============================================================
Puzzle ID: test_puzzle
输入网格大小: (3, 3)

输入网格:
----------------------------------------
 0  1  2
 3  4  5
 6  7  8
----------------------------------------

============================================================
开始推理...
============================================================

步骤 1:
  Q_halt logit: -2.3456
  Q_continue logit: 1.2345
  Halted: False
  Steps: 1

步骤 1 预测:
----------------------------------------
 0  1  2
 3  4  5
 6  7  8
----------------------------------------

步骤 2:
  Q_halt logit: -1.1234
  Q_continue logit: 0.9876
  Halted: False
  Steps: 2

...

所有序列在第 16 步halt

============================================================
最终结果
============================================================

最终预测网格大小: (3, 3)

最终预测:
----------------------------------------
 0  1  2
 3  4  5
 6  7  8
----------------------------------------

推理统计:
  总步数: 16
  最终Q_halt logit: 0.1234
  最终Q_continue logit: -0.5678

Q值变化:
  步骤 1: halt=-2.3456, continue= 1.2345
  步骤 2: halt=-1.1234, continue= 0.9876
  ...
```

## 使用示例

### 示例1: 测试简单网格

```bash
python test_single_puzzle.py \
  --checkpoint checkpoints/Arc1concept-aug-1000-ACT-torch/my_training_run/step_303191 \
  --input "[[0,0,0],[0,1,0],[0,0,0]]" \
  --puzzle_id "simple_test"
```

### 示例2: 从文件加载

```bash
# 创建输入文件
cat > test_puzzle.json << EOF
[[0, 1, 2, 3],
 [4, 5, 6, 7],
 [8, 9, 0, 1]]
EOF

# 运行测试
python test_single_puzzle.py \
  --checkpoint checkpoints/Arc1concept-aug-1000-ACT-torch/my_training_run/step_303191 \
  --input test_puzzle.json \
  --max_steps 20
```

### 示例3: 测试ARC题目

如果你有ARC格式的题目文件，可以提取input部分：

```python
import json

# 加载ARC题目
with open('arc_puzzle.json', 'r') as f:
    puzzle = json.load(f)

# 提取第一个test example的input
input_grid = puzzle['test'][0]['input']

# 保存为简单格式
with open('test_input.json', 'w') as f:
    json.dump(input_grid, f)
```

然后运行：
```bash
python test_single_puzzle.py \
  --checkpoint checkpoints/Arc1concept-aug-1000-ACT-torch/my_training_run/step_303191 \
  --input test_input.json
```

## 注意事项

1. **输入格式**: 输入必须是2D数组，值范围0-9
2. **网格大小**: 最大支持30x30的网格
3. **GPU内存**: 确保有足够的GPU内存
4. **模型结构**: 脚本会自动处理模型包装（loss head）

## 代码位置

- 脚本: `test_single_puzzle.py`
- 核心推理逻辑: `models/recursive_reasoning/trm.py` 的 `forward` 方法

