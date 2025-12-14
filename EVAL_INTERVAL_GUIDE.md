# 评估间隔（eval_interval）说明

## 基本概念

**重要**：`eval_interval` 的单位是 **epochs（轮数）**，不是 steps（步数）。

## 评估间隔的计算

### 1. 自动设置（推荐）

如果不指定 `--eval-interval`，`finetune_addition.py` 会自动设置：

#### 使用 `--total-steps` 时：
```python
eval_interval = max(1000, total_steps // 10)  # 每10%评估一次，最少1000 epochs
```

**示例**：
- `total_steps=3000` → `eval_interval = max(1000, 3000//10) = max(1000, 300) = 1000`
- `total_steps=50000` → `eval_interval = max(1000, 50000//10) = max(1000, 5000) = 5000`

#### 使用 `--epochs` 时：
```python
if epochs >= 10000:
    eval_interval = 1000
elif epochs >= 1000:
    eval_interval = epochs // 10  # 每10%评估一次
else:
    eval_interval = max(1, epochs // 10)
```

**示例**：
- `epochs=10000` → `eval_interval = 1000`
- `epochs=5000` → `eval_interval = 5000 // 10 = 500`
- `epochs=100` → `eval_interval = max(1, 100//10) = 10`

**注意**：代码会自动调整 `eval_interval`，确保它是 `epochs` 的除数。

### 2. 手动设置

可以通过以下方式手动设置：

#### 方式1: 命令行参数
```bash
python finetune_addition.py \
    --eval-interval 1000 \
    ...
```

#### 方式2: 配置文件
编辑 `config/cfg_finetune_addition.yaml`：
```yaml
eval_interval: 1000  # 每1000个epochs评估一次
```

## 评估间隔与Steps的关系

### 计算公式

```
steps_per_epoch = total_examples / global_batch_size
steps_per_eval = eval_interval * steps_per_epoch
```

### 实际示例

假设：
- 训练集：10000个puzzles，平均每个puzzle 7个样本
- 总样本数：70000
- `global_batch_size = 64`
- `eval_interval = 1000`（epochs）

计算：
```
steps_per_epoch = 70000 / 64 ≈ 1094 steps
steps_per_eval = 1000 * 1094 ≈ 1,094,000 steps
```

**所以**：`eval_interval=1000` 意味着每 **1,094,000 steps** 评估一次。

## 训练流程

训练会分成多个迭代（iter），每个迭代的训练和评估流程：

```
总epochs = 10000
eval_interval = 1000
total_iters = 10000 / 1000 = 10

训练流程：
- Iter 0: 训练 1000 epochs → 评估
- Iter 1: 训练 1000 epochs → 评估
- Iter 2: 训练 1000 epochs → 评估
- ...
- Iter 9: 训练 1000 epochs → 评估
```

## 设置建议

### 1. 快速测试
```bash
--eval-interval 100  # 每100个epochs评估一次（更频繁）
```

### 2. 正常训练
```bash
--eval-interval 1000  # 每1000个epochs评估一次（推荐）
```

### 3. 长时间训练
```bash
--eval-interval 5000  # 每5000个epochs评估一次（减少评估频率）
```

### 4. 使用自动设置（推荐）
```bash
# 不指定 --eval-interval，让程序自动设置
python finetune_addition.py --total-steps 30000 ...
```

## 注意事项

1. **eval_interval 必须是 epochs 的除数**
   - 如果 `epochs=10000`，`eval_interval` 必须是 10000 的除数（如 1000, 2000, 5000）
   - 如果不是，代码会自动调整

2. **min_eval_interval**
   - 从第几个迭代开始评估
   - 默认：0（从一开始就评估）

3. **评估会暂停训练**
   - 评估需要时间，会暂停训练
   - 如果评估太频繁，会影响训练速度

4. **checkpoint_every_eval**
   - 如果设置为 `True`，每次评估后都会保存checkpoint
   - 默认：`True`（在 `cfg_finetune_addition.yaml` 中）

## 配置文件示例

```yaml
# config/cfg_finetune_addition.yaml
epochs: 10000
eval_interval: 1000  # 每1000个epochs评估一次
min_eval_interval: 0  # 从第0个迭代开始评估
checkpoint_every_eval: True  # 每次评估后保存checkpoint
```

## 查看实际评估间隔

训练开始时会打印：
```
评估间隔: 1000
```

训练过程中会显示：
```
Iter 0: 训练 1000 epochs → 评估
Iter 1: 训练 1000 epochs → 评估
...
```

## 常见问题

### Q: 我想每1000 steps评估一次，怎么设置？

A: `eval_interval` 的单位是 epochs，不是 steps。需要先计算：
```
steps_per_epoch = total_examples / batch_size
eval_interval = 1000 / steps_per_epoch
```

例如：如果 `steps_per_epoch = 1094`，要每1000 steps评估一次：
```
eval_interval = 1000 / 1094 ≈ 0.91
```
但 `eval_interval` 必须是整数（epochs），所以无法精确设置为每1000 steps评估一次。

### Q: 为什么评估间隔是epochs而不是steps？

A: 因为训练数据是按epoch组织的，每个epoch遍历一次完整数据集。使用epochs作为单位更直观，也更容易与数据集大小无关。

### Q: 如何知道实际多少steps评估一次？

A: 查看训练日志，会显示：
```
ℹ️  Using total_steps=30000, calculated epochs=273
   (total_groups=1000, mean_puzzle_examples=7.01, batch_size=64)
评估间隔: 1000
```

然后计算：
```
steps_per_epoch = 30000 / 273 ≈ 110
steps_per_eval = 1000 * 110 = 110,000 steps
```

