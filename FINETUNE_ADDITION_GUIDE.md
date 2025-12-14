# 加法数据集微调指南

本指南介绍如何在加法数据集上微调基础模型（3万步）。

## 前提条件

1. **基础模型checkpoint**：已完成3万步训练的基础模型
2. **加法数据集**：已生成加法数据集（见 `dataset/ADDITION_DATASET_README.md`）

## 快速开始

### 方法1：使用微调脚本（推荐）

```bash
cd /root/hns/TinyRecursiveModels
conda activate trm

# 单GPU微调
python finetune_addition.py \
    --base-checkpoint checkpoints/YourProject/base_model/step_30000 \
    --data-path data/addition \
    --run-name finetune_addition \
    --epochs 10000 \
    --lr 5e-5 \
    --global-batch-size 64
```

### 方法2：使用pretrain.py直接微调

```bash
cd /root/hns/TinyRecursiveModels
conda activate trm

python pretrain.py \
    arch=trm \
    data_paths="[data/addition]" \
    load_checkpoint=checkpoints/YourProject/base_model/step_30000 \
    +run_name=finetune_addition \
    epochs=10000 \
    eval_interval=1000 \
    lr=5e-5 \
    puzzle_emb_lr=1e-3 \
    global_batch_size=64 \
    ema=True
```

### 多GPU微调

```bash
torchrun --nproc-per-node 4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --nnodes=1 \
    finetune_addition.py \
    --base-checkpoint checkpoints/YourProject/base_model/step_30000 \
    --data-path data/addition \
    --run-name finetune_addition_4gpu \
    --epochs 10000 \
    --lr 5e-5 \
    --global-batch-size 256
```

## 参数说明

### 必需参数

- `--base-checkpoint`: 基础模型checkpoint路径
  - 示例：`checkpoints/YourProject/base_model/step_30000`
- `--data-path`: 加法数据集路径
  - 示例：`data/addition`
- `--run-name`: 运行名称（用于保存checkpoint和日志）

### 训练参数

- `--total-steps`: 总训练步数（推荐：50000-200000，默认：100000）
  - **注意**：使用total_steps可以精确控制训练步数，避免epochs导致的步数过多
  - 计算公式：`steps = epochs × total_groups × mean_puzzle_examples / batch_size`
  - 对于addition数据集：`steps ≈ epochs × 10000 × 6.55 / batch_size`
  - 例如：2000 epochs × 64 batch_size ≈ 200万steps（可能过多）
- `--epochs`: 训练轮数（默认：None，使用total_steps）
  - **注意**：如果使用epochs，建议值：500-1000（对应约50万-100万steps）
- `--eval-interval`: 评估间隔（默认：自动设置）
- `--lr`: 学习率（默认：5e-5，微调时建议较小）
- `--puzzle-emb-lr`: Puzzle embedding学习率（默认：1e-3）
- `--global-batch-size`: 全局批次大小（默认：64）

### 架构参数

- `--L-layers`: L层数（可选，默认使用配置文件中的值）
- `--H-cycles`: H循环次数（可选）
- `--L-cycles`: L循环次数（可选）

### 其他参数

- `--ema`: 启用指数移动平均（默认：True）
- `--no-ema`: 禁用EMA
- `--freeze-weights`: 冻结权重，只训练puzzle embeddings

## 微调策略

### 1. 全参数微调（推荐）

使用较小的学习率微调所有参数：

```bash
# 使用total_steps（推荐，约15个epochs）
python finetune_addition.py \
    --base-checkpoint checkpoints/YourProject/base_model/step_30000 \
    --data-path data/addition \
    --run-name finetune_addition_full \
    --total-steps 100000 \
    --lr 5e-5 \
    --puzzle-emb-lr 1e-3 \
    --ema

# 或使用epochs（约50万steps）
python finetune_addition.py \
    --base-checkpoint checkpoints/YourProject/base_model/step_30000 \
    --data-path data/addition \
    --run-name finetune_addition_full \
    --epochs 500 \
    --global-batch-size 128 \
    --lr 5e-5 \
    --puzzle-emb-lr 1e-3 \
    --ema
```

### 2. 只训练Puzzle Embeddings

如果只想让模型学习新的puzzle类型，可以冻结其他权重：

```bash
python finetune_addition.py \
    --base-checkpoint checkpoints/YourProject/base_model/step_30000 \
    --data-path data/addition \
    --run-name finetune_addition_emb_only \
    --freeze-weights \
    --puzzle-emb-lr 1e-2
```

### 3. 渐进式微调

先冻结权重训练embeddings，再全参数微调：

```bash
# 第一步：只训练embeddings
python finetune_addition.py \
    --base-checkpoint checkpoints/Arc1concept-aug-1000-ACT-torch/my_training_run/step_303191 \
    --data-path data/addition_fixed \
    --run-name finetune_addition_step1 \
    --freeze-weights \
    --total-steps 3000 \
    --eval-interval 1500 \
    --puzzle-emb-lr 1e-2 \
    --copy-loss-weight 0 \
    --global-batch-size 64

# 第二步：全参数微调
python finetune_addition.py \
    --base-checkpoint checkpoints/Addition_fixed-ACT-torch/finetune_addition_step1/step_3000 \
    --data-path data/addition_fixed \
    --run-name finetune_addition_step2 \
    --lr 5e-5 \
    --total-steps 25000 \
    --eval-interval 5000 \
    --ema
```

## 学习率建议

微调时的学习率通常比预训练小：

- **全参数微调**：`lr=5e-5`（预训练的1/2到1/10）
- **Puzzle embedding**：`puzzle_emb_lr=1e-3`（预训练的1/10）
- **冻结权重时**：`puzzle_emb_lr=1e-2`（可以稍大）

## 监控训练

训练过程会：

1. **保存checkpoint**：保存在 `checkpoints/AdditionFinetune/{run_name}/`
2. **记录到WandB**：如果已登录WandB，会自动记录训练指标
3. **定期评估**：根据 `eval_interval` 设置进行评估

### 查看训练日志

```bash
# 实时查看日志
tail -f logs/finetune_addition.log

# 查看checkpoint
ls -lh checkpoints/AdditionFinetune/finetune_addition/
```

## 常见问题

### Q: 如何选择合适的学习率？

A: 
- 从较小的学习率开始（如 `5e-5`）
- 如果损失下降很慢，可以适当增大（如 `1e-4`）
- 如果损失不稳定，减小学习率（如 `1e-5`）

### Q: 需要训练多少轮/步数？

A: 
- **推荐使用total_steps而不是epochs**，可以精确控制训练步数
- 对于微调，建议：
  - **total_steps**: 50,000 - 200,000（约7-30个epochs）
  - **epochs**: 500 - 1000（对应约50万-100万steps）
- **注意**：2000 epochs ≈ 200万steps，对于微调来说可能过多
- 观察验证损失，如果已经收敛可以提前停止
- 计算公式：`steps = epochs × 10000 × 6.55 / batch_size`

### Q: 批次大小如何选择？

A: 
- 单GPU：32-64
- 4 GPU：128-256
- 根据GPU内存调整

### Q: 如何从微调checkpoint继续训练？

A: 使用 `load_checkpoint` 参数：

```bash
python finetune_addition.py \
    --base-checkpoint checkpoints/AdditionFinetune/finetune_addition/step_5000 \
    --data-path data/addition \
    --run-name finetune_addition_continue \
    --epochs 10000
```

### Q: 微调和预训练的区别？

A: 
- **预训练**：从头开始训练，学习通用推理能力
- **微调**：在预训练模型基础上，针对特定任务（加法）进行优化
- 微调通常使用更小的学习率和更少的训练轮数

## 评估微调后的模型

训练完成后，可以使用 `inference.py` 评估模型：

```bash
python inference.py \
    --checkpoint checkpoints/AdditionFinetune/finetune_addition/step_10000 \
    --data-paths data/addition \
    --max-eval-batches 100
```

## 示例：完整微调流程

```bash
# 1. 生成数据集（如果还没有）
python -m dataset.build_addition_dataset \
    --output-dir data/addition \
    --train-size 10000 \
    --test-size 1000

# 2. 微调模型
python finetune_addition.py \
    --base-checkpoint checkpoints/YourProject/base_model/step_30000 \
    --data-path data/addition \
    --run-name finetune_addition_v1 \
    --epochs 10000 \
    --lr 5e-5 \
    --global-batch-size 64 \
    --ema

# 3. 评估模型
python inference.py \
    --checkpoint checkpoints/AdditionFinetune/finetune_addition_v1/step_10000 \
    --data-paths data/addition
```

## 注意事项

1. **确保checkpoint路径正确**：基础模型的checkpoint必须存在
2. **数据集路径**：确保 `data/addition` 目录存在且包含训练和测试数据
3. **GPU内存**：如果内存不足，减小 `global_batch_size`
4. **学习率**：微调时使用较小的学习率，避免破坏预训练权重
5. **EMA**：建议启用EMA，可以获得更稳定的模型
6. **词汇表大小不匹配**：如果基础模型和微调数据集的词汇表大小不同，代码会自动调整embedding层
7. **CUDA断言警告**：训练开始时可能出现一些CUDA断言警告（来自torch.compile），通常可以忽略，不影响训练

## 常见错误处理

### 词汇表大小不匹配

如果遇到词汇表大小不匹配的错误，代码会自动处理：
- 会自动调整embedding和输出层的尺寸
- 会打印警告信息，但训练会继续进行

### CUDA断言错误

如果看到大量CUDA断言错误（`Assertion 'index out of bounds'`），可以尝试：

```bash
# 禁用torch.compile（如果问题持续）
DISABLE_COMPILE=1 python finetune_addition.py ...
```

### 内存不足

如果遇到GPU内存不足：

```bash
# 减小批次大小
python finetune_addition.py ... --global-batch-size 32

# 或者使用梯度累积（需要修改代码）
```

