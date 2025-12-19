# 象棋微调环境

这个目录包含了在象棋数据集上微调基础模型的所有代码和配置。

## 文件结构

```
chess/
├── finetune_chess.py      # 微调脚本（主入口）
└── README.md              # 本文件
```

## 使用方法

### 基本用法

```bash
cd /root/hns/TinyRecursiveModels
python chess/finetune_chess.py \
    --base-checkpoint checkpoints/YourProject/base_model/step_30000 \
    --data-path data/chess \
    --run-name finetune_chess
```

### 多GPU微调

```bash
torchrun --nproc-per-node 4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --nnodes=1 \
    chess/finetune_chess.py \
    --base-checkpoint checkpoints/YourProject/base_model/step_30000 \
    --data-path data/chess \
    --run-name finetune_chess \
    --epochs 10000 \
    --lr 5e-5
```

### 自定义参数示例

```bash
python chess/finetune_chess.py \
    --base-checkpoint checkpoints/Project/base_model/step_30000 \
    --data-path data/chess \
    --run-name finetune_chess \
    --lr 1e-4 \
    --global-batch-size 128 \
    --total-steps 150000 \
    --eval-interval-steps 2000
```

## 参数说明

### 必需参数

- `--base-checkpoint`: 基础模型checkpoint路径
- `--data-path`: 象棋数据集路径（例如：`data/chess`）
- `--run-name`: 运行名称（用于保存checkpoint和日志）

### 训练参数

- `--epochs`: 训练轮数（默认：None，使用total_steps）
- `--total-steps`: 总训练步数（默认：100000）
- `--eval-interval`: 评估间隔（单位：epochs）
- `--eval-interval-steps`: 评估间隔（单位：steps，优先级更高）
- `--lr`: 学习率（默认：5e-5）
- `--puzzle-emb-lr`: Puzzle embedding学习率（默认：1e-3）
- `--global-batch-size`: 全局批次大小（默认：使用配置文件中的值）

### 架构参数

- `--arch`: 架构名称（默认：trm）
- `--L-layers`: L层数
- `--H-cycles`: H循环次数
- `--L-cycles`: L循环次数

### 其他参数

- `--ema`: 使用指数移动平均（默认：True）
- `--no-ema`: 禁用指数移动平均
- `--ema-rate`: EMA率（默认：0.999）
- `--freeze-weights`: 冻结权重，只训练puzzle embeddings
- `--loss-type`: 基础损失函数类型（默认：stablemax_cross_entropy）
- `--lm-loss-weight`: lm_loss的权重
- `--q-halt-loss-weight`: q_halt_loss的权重

## 配置文件

微调配置位于 `config/cfg_finetune_chess.yaml`，包含：

- 训练超参数（学习率、批次大小等）
- 损失权重配置
- EMA设置
- 评估设置

## 数据格式要求

象棋数据集应该遵循与加法数据集相同的格式：

```
data/chess/
├── train/
│   ├── all__inputs.npy
│   ├── all__labels.npy
│   ├── all__puzzle_identifiers.npy
│   ├── all__puzzle_indices.npy
│   ├── all__group_indices.npy
│   └── dataset.json
└── test/
    └── ... (同上)
```

## 注意事项

1. **数据格式**：当前实现假设使用新数据格式，每条数据是一个状态转移对 `(s_i, s_{i+1})`，因此 `halt_max_steps` 默认设置为 1。

2. **损失函数**：当前使用标准的 `ACTLossHead`。如果需要针对象棋任务的特殊损失函数（例如保持棋盘某些部分不变），可以创建 `ChessACTLossHead` 并在脚本中使用。

3. **评估器**：当前配置中 `evaluators=[]`，可以根据需要添加象棋特定的评估器。

4. **推理步数**：如果希望模型能够进行多步推理（从一个状态推理到最终状态），可以手动设置 `halt_max_steps` 为一个更大的值。

## 与加法微调的差异

1. **损失函数**：象棋使用 `ACTLossHead`，而加法使用 `AdditionACTLossHead`（包含 copy loss）
2. **数据格式**：虽然格式相同，但象棋的网格表示和语义不同
3. **评估器**：可以根据象棋任务的特点添加专门的评估器

## 后续扩展

可以考虑添加：

1. **ChessACTLossHead**：如果需要在训练时保持棋盘某些部分不变（类似加法的 copy loss）
2. **象棋评估器**：用于评估模型在象棋任务上的表现
3. **测试脚本**：用于测试单个象棋puzzle的推理过程
