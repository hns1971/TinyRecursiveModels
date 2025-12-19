# TRM 训练脚本使用指南

## 训练流程概览

1. **准备数据集** → 2. **配置训练参数** → 3. **运行训练脚本** → 4. **监控训练过程**

## 第一步：准备数据集

首先需要运行数据准备脚本（见 `DATA_PREPARATION.md`）：

```bash
# ARC-AGI-1
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation
```

## 第二步：理解训练脚本结构

训练脚本 `pretrain.py` 使用 Hydra 进行配置管理，支持：

- **命令行参数覆盖**：可以直接在命令行修改配置
- **配置文件继承**：使用 YAML 配置文件
- **分布式训练**：支持多 GPU 训练（使用 `torchrun`）

## 第三步：运行训练脚本

### 单 GPU 训练（示例）

```bash
python pretrain.py \
  arch=trm \
  arch.puzzle_emb_ndim=512 \
  data_paths="[data/arc1concept-aug-1000]" \
  arch.L_layers=2 \
  arch.H_cycles=3 \
  arch.L_cycles=4 \
  +run_name=my_training_run \
  ema=True \
  global_batch_size=64 > logs/pretrain.log 2>&1 &
```

### 多 GPU 分布式训练（推荐）

```bash
# 使用 4 个 GPU
torchrun --nproc-per-node 4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  --nnodes=1 \
  pretrain.py \
  arch=trm \
  data_paths="[data/arc1concept-aug-1000]" \
  arch.L_layers=2 \
  arch.H_cycles=3 \
  arch.L_cycles=4 \
  +run_name=my_training_run \
  ema=True
```

## 常用训练命令示例

### ARC-AGI-1 训练（4 H-100 GPUs）

```bash
run_name="pretrain_att_arc1concept_4"
torchrun --nproc-per-node 4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  --nnodes=1 \
  pretrain.py \
  arch=trm \
  data_paths="[data/arc1concept-aug-1000]" \
  arch.L_layers=2 \
  arch.H_cycles=3 \
  arch.L_cycles=4 \
  +run_name=${run_name} \
  ema=True
```

**预计运行时间**：约 3 天

### ARC-AGI-2 训练（4 H-100 GPUs）

```bash
run_name="pretrain_att_arc2concept_4"
torchrun --nproc-per-node 4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  --nnodes=1 \
  pretrain.py \
  arch=trm \
  data_paths="[data/arc2concept-aug-1000]" \
  arch.L_layers=2 \
  arch.H_cycles=3 \
  arch.L_cycles=4 \
  +run_name=${run_name} \
  ema=True
```

**预计运行时间**：约 3 天

### Sudoku-Extreme 训练（1 L40S GPU）

```bash
# MLP 版本
run_name="pretrain_mlp_t_sudoku"
python pretrain.py \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  evaluators="[]" \
  epochs=50000 \
  eval_interval=5000 \
  lr=1e-4 \
  puzzle_emb_lr=1e-4 \
  weight_decay=1.0 \
  puzzle_emb_weight_decay=1.0 \
  arch.mlp_t=True \
  arch.pos_encodings=none \
  arch.L_layers=2 \
  arch.H_cycles=3 \
  arch.L_cycles=6 \
  +run_name=${run_name} \
  ema=True

# Attention 版本
run_name="pretrain_att_sudoku"
python pretrain.py \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  evaluators="[]" \
  epochs=50000 \
  eval_interval=5000 \
  lr=1e-4 \
  puzzle_emb_lr=1e-4 \
  weight_decay=1.0 \
  puzzle_emb_weight_decay=1.0 \
  arch.L_layers=2 \
  arch.H_cycles=3 \
  arch.L_cycles=6 \
  +run_name=${run_name} \
  ema=True
```

**预计运行时间**：< 36 小时

### Maze-Hard 训练（4 L40S GPUs）

```bash
run_name="pretrain_att_maze30x30"
torchrun --nproc-per-node 4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  --nnodes=1 \
  pretrain.py \
  arch=trm \
  data_paths="[data/maze-30x30-hard-1k]" \
  evaluators="[]" \
  epochs=50000 \
  eval_interval=5000 \
  lr=1e-4 \
  puzzle_emb_lr=1e-4 \
  weight_decay=1.0 \
  puzzle_emb_weight_decay=1.0 \
  arch.L_layers=2 \
  arch.H_cycles=3 \
  arch.L_cycles=4 \
  +run_name=${run_name} \
  ema=True
```

**预计运行时间**：< 24 小时

## 重要参数说明

### 架构参数（arch.*）

- `arch.L_layers`: L 层数（通常为 2）
- `arch.H_cycles`: H 循环次数（通常为 3）
- `arch.L_cycles`: L 循环次数（通常为 4-6）
- `arch.hidden_size`: 隐藏层大小（默认 512）
- `arch.num_heads`: 注意力头数（默认 8）
- `arch.mlp_t`: 是否在 L 层使用 MLP 而非 Transformer（默认 False）
- `arch.pos_encodings`: 位置编码类型（`rope` 或 `none`）

### 训练参数

- `data_paths`: 数据集路径列表（使用 `"[path]"` 格式）
- `global_batch_size`: 全局批次大小（默认 768）
- `epochs`: 训练轮数（默认 100000）
- `eval_interval`: 评估间隔（默认 10000）
- `lr`: 学习率（默认 1e-4）
- `puzzle_emb_lr`: Puzzle embedding 学习率（默认 1e-2）
- `weight_decay`: 权重衰减（默认 0.1）
- `ema`: 是否使用指数移动平均（默认 False）
- `ema_rate`: EMA 率（默认 0.999）

### 评估器参数

- `evaluators`: 评估器列表
  - ARC: `evaluators="[{name: arc@ARC}]"`
  - 无评估器: `evaluators="[]"`

## 从检查点恢复训练

```bash
python pretrain.py \
  arch=trm \
  data_paths="[data/arc1concept-aug-1000]" \
  load_checkpoint=checkpoints/YourProject/your_run_name/step_10000 \
  +run_name=resume_training \
  ema=True
```

## 监控训练

训练过程会：

1. **自动保存检查点**：保存在 `checkpoints/{project_name}/{run_name}/`
2. **记录到 WandB**：如果已登录 WandB，会自动记录训练指标
3. **定期评估**：根据 `eval_interval` 设置进行评估

### 检查点结构

```
checkpoints/
└── YourProject/
    └── your_run_name/
        ├── step_10000
        ├── step_20000
        ├── all_config.yaml
        └── ...
```

## 常见问题

### Q: 如何修改批次大小？
A: 使用 `global_batch_size=512`（根据你的 GPU 内存调整）

### Q: 如何禁用评估？
A: 设置 `evaluators="[]"` 和 `eval_interval=1000000`（一个很大的值）

### Q: 如何只训练 puzzle embeddings？
A: 设置 `freeze_weights=True`

### Q: 分布式训练失败怎么办？
A: 检查：
- CUDA 版本是否匹配
- 所有 GPU 是否可见（`nvidia-smi`）
- 端口是否被占用（可以尝试不同的 `--rdzv_endpoint`）

### Q: 内存不足怎么办？
A: 
- 减小 `global_batch_size`
- 减小 `arch.hidden_size`
- 使用梯度累积（需要修改代码）

## 环境变量

- `DISABLE_COMPILE`: 设置为任意值可禁用 `torch.compile`（用于调试）
- `LOCAL_RANK`: 分布式训练时自动设置（表示当前 GPU 的本地 rank）
- `CUDA_VISIBLE_DEVICES`: 指定可见的 GPU（例如 `CUDA_VISIBLE_DEVICES=0,1,2,3`）

