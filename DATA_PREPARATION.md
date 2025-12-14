# 数据集准备指南

## 关于数据下载

**重要：`build_arc_dataset.py` 不会自动从网上下载数据集！**

这个脚本需要你**手动准备**原始的 ARC-AGI 数据集 JSON 文件。脚本会读取本地已存在的 JSON 文件并进行处理和转换。

## 数据集来源

### ARC-AGI 数据集

ARC-AGI 数据集需要从 Kaggle 下载：
- 访问 [ARC-AGI 竞赛页面](https://www.kaggle.com/competitions/arc-agi)
- 下载数据集文件
- 将文件放置在 `kaggle/combined/` 目录下

需要的文件格式：
- `arc-agi_training_challenges.json`
- `arc-agi_training_solutions.json`
- `arc-agi_evaluation_challenges.json`
- `arc-agi_evaluation_solutions.json`
- `arc-agi_evaluation2_challenges.json`
- `arc-agi_evaluation2_solutions.json`
- `arc-agi_training2_challenges.json`
- `arc-agi_training2_solutions.json`
- `arc-agi_concept_challenges.json`
- `arc-agi_concept_solutions.json`

### 其他数据集

- **Maze-Hard**: `build_maze_dataset.py` 会从 HuggingFace Hub 自动下载（使用 `hf_hub_download`）
- **Sudoku-Extreme**: `build_sudoku_dataset.py` 会生成合成数据

## 运行 build_arc_dataset.py

### 基本用法

```bash
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation
```

### 参数说明

- `--input-file-prefix`: 输入文件的前缀路径（例如 `kaggle/combined/arc-agi`）
  - 脚本会自动查找 `{prefix}_{subset}_challenges.json` 和 `{prefix}_{subset}_solutions.json`
  
- `--output-dir`: 输出目录（处理后的数据集将保存在这里）

- `--subsets`: 要处理的子集列表（例如 `training evaluation concept`）

- `--test-set-name`: 测试集名称（例如 `evaluation`）

- `--num-aug`: 数据增强数量（默认 1000）

- `--seed`: 随机种子（默认 42）

### ARC-AGI-1 示例

```bash
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation \
  --num-aug 1000 \
  --seed 42
```

### ARC-AGI-2 示例

```bash
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2 \
  --num-aug 1000 \
  --seed 42
```

## 脚本功能

`build_arc_dataset.py` 主要完成以下工作：

1. **读取原始 JSON 文件**：从 `kaggle/combined/` 目录读取 ARC-AGI 挑战和解决方案

2. **数据转换**：
   - 将网格数据转换为 NumPy 数组
   - 处理网格大小（最大 30x30）
   - 添加填充和 EOS 标记

3. **数据增强**：
   - 使用 8 种二面体变换（旋转、翻转）
   - 随机颜色排列（保持黑色不变）
   - 平移增强（仅训练集）

4. **数据集划分**：
   - 将数据分为训练集和测试集
   - 生成 puzzle identifiers 映射

5. **保存处理后的数据**：
   - 保存为 `.npy` 格式（inputs, labels, puzzle_identifiers 等）
   - 保存元数据 JSON 文件
   - 保存测试集 puzzles

## 输出结构

处理后的数据集目录结构：

```
data/arc1concept-aug-1000/
├── train/
│   ├── all__inputs.npy
│   ├── all__labels.npy
│   ├── all__puzzle_identifiers.npy
│   ├── all__puzzle_indices.npy
│   ├── all__group_indices.npy
│   └── dataset.json
├── test/
│   ├── all__inputs.npy
│   ├── all__labels.npy
│   ├── all__puzzle_identifiers.npy
│   ├── all__puzzle_indices.npy
│   ├── all__group_indices.npy
│   └── dataset.json
├── identifiers.json
└── test_puzzles.json
```

## 常见问题

### Q: 如果找不到 solutions 文件怎么办？
A: 脚本会自动用 dummy 数据填充（输出全零网格），但会打印警告信息。

### Q: 数据增强数量不够怎么办？
A: 脚本会尝试 `ARCAugmentRetriesFactor * num_aug` 次来生成足够的增强数据。如果仍然不够，会打印警告但继续处理。

### Q: 如何检查数据集是否正确生成？
A: 检查输出目录中的 `dataset.json` 文件，查看统计信息（总 puzzles 数、平均 examples 数等）。

