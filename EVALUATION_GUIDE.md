# Addition模型评估指南

本指南介绍如何评估训练好的addition模型。

## 评估代码说明

项目中有两个评估相关的脚本：

1. **`inference.py`** - 评估整个测试数据集，计算整体指标
2. **`test_addition_puzzle.py`** - 测试单个加法题目，查看详细推理过程

## 方法1: 评估整个数据集（推荐）

使用 `inference.py` 评估模型在整个测试集上的表现。

### 基本用法

```bash
cd /root/hns/TinyRecursiveModels
conda activate trm

python inference.py \
    --checkpoint checkpoints/Addition-ACT-torch/finetune_addition_step2/step_25000 \
    --data_paths data/addition \
    --config_name cfg_finetune_addition
```

### 完整参数示例

```bash
python inference.py \
    --checkpoint checkpoints/Addition-ACT-torch/finetune_addition_step2/step_25000 \
    --data_paths data/addition \
    --config_name cfg_finetune_addition \
    --save_outputs preds inputs labels
```

### 参数说明

- `--checkpoint`: **必需**，checkpoint文件路径
  - 示例: `checkpoints/Addition-ACT-torch/finetune_addition_step2/step_25000`
  - 注意: 这是checkpoint文件路径，不是目录

- `--data_paths`: 数据集路径列表（默认: `["data/arc1concept-aug-1000"]`）
  - 对于addition任务，使用: `data/addition`
  - 示例: `--data_paths data/addition`

- `--config_name`: 配置文件名（默认: `cfg_pretrain`）
  - 对于addition微调，使用: `cfg_finetune_addition`
  - 示例: `--config_name cfg_finetune_addition`

- `--max_batches`: 最大评估batch数量（可选）
  - 用于快速测试，限制处理的batch数量
  - 示例: `--max_batches 10` 只处理前10个batch
  - **全量评估**: 不指定此参数，但需要注意配置文件中的 `max_eval_batches` 设置
  - 如果配置文件中有 `max_eval_batches: 100`，需要将其改为 `null` 或删除该行才能进行全量评估

- `--save_outputs`: 要保存的输出键（可选）
  - 可选项: `preds`, `inputs`, `puzzle_identifiers`, `q_halt_logits` 等
  - 保存的预测结果会在checkpoint目录下: `step_{step}_all_preds.0`
  - 默认: `["preds", "inputs"]`

- `--config_path`: 配置文件目录（默认: `config`）

### 评估指标说明

评估会计算以下指标：

1. **accuracy**: 位置级别的准确率
   - 计算每个位置预测正确的比例
   - 范围: 0.0 - 1.0

2. **exact_accuracy**: 序列级别的准确率
   - 整个序列（整个加法题目）完全正确的比例
   - 范围: 0.0 - 1.0
   - **这是最重要的指标**，表示模型能正确完成多少个加法题目

3. **q_halt_accuracy**: Q-halt预测的准确率
   - 模型预测是否应该停止的准确率
   - 范围: 0.0 - 1.0

4. **steps**: 平均推理步数
   - 模型完成推理所需的平均步数
   - 越小越好（表示模型能更快地完成推理）

5. **lm_loss**: 语言模型损失
   - 预测序列的交叉熵损失
   - 越小越好

6. **q_halt_loss**: Q-halt损失
   - 停止预测的损失
   - 越小越好

### 示例输出

```
============================================================
推理配置
============================================================
Checkpoint: checkpoints/Addition-ACT-torch/finetune_addition_step2/step_25000
数据集: ['data/addition']
最大batch数: 全部
保存输出: ['preds', 'inputs']
============================================================

开始推理...
Processing batch 1: all
  Completed inference in 4 steps
Processing batch 2: all
  Completed inference in 5 steps
...

============================================================
推理结果
============================================================

all:
  accuracy: 0.985234
  exact_accuracy: 0.923456
  q_halt_accuracy: 0.876543
  steps: 4.234567
  lm_loss: 0.123456
  q_halt_loss: 0.234567
============================================================
```

## 方法2: 测试单个题目

使用 `test_addition_puzzle.py` 测试单个加法题目，查看详细的推理过程。

### 基本用法

```bash
python test_addition_puzzle.py \
    --checkpoint checkpoints/Addition-ACT-torch/finetune_addition_step2/step_25000 \
    --num1 123 \
    --num2 456 \
    --confidence-threshold 0.95 \
    --max-steps 12
```

### 参数说明

- `--checkpoint`: checkpoint文件路径
- `--num1`: 第一个加数
- `--num2`: 第二个加数
- `--confidence-threshold`: 置信度阈值，达到此阈值时停止推理（默认: 0.95）
- `--max-steps`: 最大推理步数（默认: 16）

### 输出说明

脚本会显示：
1. 每一步的预测网格
2. Q_halt logit和置信度
3. 是否达到置信度阈值
4. 最终结果和是否正确

详细说明请参考 `TEST_ADDITION_PUZZLE_GUIDE.md`。

## 快速测试示例

### 1. 评估整个测试集（快速测试）

```bash
# 只评估前10个batch，快速查看结果
python inference.py \
    --checkpoint checkpoints/Addition-ACT-torch/finetune_addition_step2/step_25000 \
    --data_paths data/addition \
    --config_name cfg_finetune_addition \
    --max_batches 10
```

### 2. 测试单个题目

```bash
# 测试一个简单的加法
python test_addition_puzzle.py \
    --checkpoint checkpoints/Addition-ACT-torch/finetune_addition_step2/step_25000 \
    --num1 22 \
    --num2 81

# 测试一个有进位的加法
python test_addition_puzzle.py \
    --checkpoint checkpoints/Addition-ACT-torch/finetune_addition_step2/step_25000 \
    --num1 999 \
    --num2 1
```

### 3. 评估完整测试集（全量评估）

**重要**: 配置文件 `cfg_finetune_addition.yaml` 中默认设置了 `max_eval_batches: 100`，这会限制评估的batch数量。

要进行全量评估，有两种方式：

**方式1: 修改配置文件（推荐）**

编辑 `config/cfg_finetune_addition.yaml`，将：
```yaml
max_eval_batches: 100
```
改为：
```yaml
max_eval_batches: null
```
或者直接删除这一行。

然后运行：
```bash
# 评估整个测试集（全量评估）
python inference.py \
    --checkpoint checkpoints/Addition-ACT-torch/finetune_addition_step2/step_25000 \
    --data_paths data/addition \
    --config_name cfg_finetune_addition
```

**方式2: 使用命令行参数覆盖（临时）**

如果不想修改配置文件，可以通过命令行传递一个很大的值：
```bash
# 评估整个测试集（假设测试集不超过10000个batch）
python inference.py \
    --checkpoint checkpoints/Addition-ACT-torch/finetune_addition_step2/step_25000 \
    --data_paths data/addition \
    --config_name cfg_finetune_addition \
    --max_batches 10000
```

**注意**: 方式2需要知道测试集的大致batch数量，不够灵活。推荐使用方式1。

## 评估代码可用性检查

评估代码是**可用的**，主要功能：

1. ✅ **加载checkpoint**: 可以正确加载训练好的模型
2. ✅ **数据加载**: 可以加载测试数据集
3. ✅ **推理**: 可以进行多步递归推理
4. ✅ **指标计算**: 可以计算accuracy、exact_accuracy等指标
5. ✅ **结果保存**: 可以保存预测结果

## 全量评估说明

**重要**: 配置文件 `config/cfg_finetune_addition.yaml` 中默认设置了 `max_eval_batches: 100`，这会限制评估的batch数量。

要进行全量评估（评估整个测试集），需要修改配置文件：

1. 编辑 `config/cfg_finetune_addition.yaml`
2. 将 `max_eval_batches: 100` 改为 `max_eval_batches: null`，或直接删除这一行
3. 然后运行评估命令（不指定 `--max_batches` 参数）

```bash
# 全量评估（评估整个测试集）
python inference.py \
    --checkpoint checkpoints/Addition-ACT-torch/finetune_addition_step2/step_25000 \
    --data_paths data/addition \
    --config_name cfg_finetune_addition
```

**注意**: 
- 如果不修改配置文件，即使不指定 `--max_batches`，也会被限制为100个batch
- 全量评估可能需要较长时间，建议先用 `--max_batches 10` 快速测试

## 注意事项

1. **配置文件**: 确保使用正确的配置文件（`cfg_finetune_addition`）
2. **数据集路径**: 确保测试数据集存在（`data/addition/test/`）
3. **GPU内存**: 确保有足够的GPU内存
4. **Checkpoint路径**: checkpoint路径必须是文件路径，不是目录
5. **全量评估**: 如果配置文件中有 `max_eval_batches` 限制，需要将其设置为 `null` 才能进行全量评估

## 故障排除

### 问题1: 找不到checkpoint

```
错误: FileNotFoundError: checkpoint文件不存在
解决: 检查checkpoint路径是否正确
```

### 问题2: 数据集路径错误

```
错误: 找不到数据集
解决: 确保 data/addition/test/ 目录存在
```

### 问题3: CUDA内存不足

```
错误: CUDA out of memory
解决: 
- 使用 --max_batches 限制batch数量
- 减小配置文件中的 global_batch_size
```

### 问题4: 配置文件不匹配

```
错误: 配置错误
解决: 确保使用 --config_name cfg_finetune_addition
```

## 评估流程建议

1. **快速测试**: 先用 `--max_batches 10` 快速测试
2. **单个题目**: 用 `test_addition_puzzle.py` 测试几个题目，查看推理过程
3. **完整评估**: 最后运行完整评估，获取准确指标

## 提取出错的输入值

如果评估时出现错误，可以使用 `extract_failed_inputs.py` 工具从保存的预测结果中提取出错的输入值，用于单独测试。

### 步骤1: 重新运行评估并保存labels

首先，需要重新运行评估，并保存 `labels` 以便验证预测是否正确：

```bash
python inference.py \
    --checkpoint checkpoints/Addition_fixed-ACT-torch/finetune_addition_step2/step_25000 \
    --data_paths data/addition_fixed \
    --config_name cfg_finetune_addition \
    --save_outputs preds inputs labels
```

**注意**: 添加 `labels` 到 `--save_outputs` 参数中，这样才能验证预测是否正确。

### 步骤2: 提取出错的输入值

运行提取工具：

```bash
python extract_failed_inputs.py \
    --preds_file checkpoints/Addition_fixed-ACT-torch/finetune_addition_step2/step_25000_all_preds.0 \
    --data_path data/addition_fixed \
    --output_file failed_inputs.txt \
    --checkpoint checkpoints/Addition_fixed-ACT-torch/finetune_addition_step2/step_25000
```

### 步骤3: 单独测试

提取工具会生成一个文本文件，包含所有出错的输入值和测试命令。可以直接使用这些命令进行单独测试：

```bash
# 从 failed_inputs.txt 中复制命令，例如：
python test_addition_puzzle.py \
    --checkpoint checkpoints/Addition_fixed-ACT-torch/finetune_addition_step2/step_25000 \
    --num1 123 \
    --num2 456
```

### 参数说明

- `--preds_file`: 预测结果文件路径（`step_{step}_all_preds.0`）
- `--data_path`: 数据集路径（用于加载metadata）
- `--output_file`: 输出文件路径（默认：`failed_inputs.txt`）
- `--max_failures`: 最大提取的失败数量（可选）
- `--checkpoint`: Checkpoint路径（用于生成测试命令，可选）

### 如果没有保存labels

如果评估时没有保存 `labels`，工具仍然可以提取输入值，但无法验证预测是否正确。此时会提取所有样本的输入值。

## 相关文件

- `inference.py`: 评估脚本
- `test_addition_puzzle.py`: 单个题目测试脚本
- `extract_failed_inputs.py`: 提取出错输入值的工具脚本
- `pretrain.py`: 评估函数实现（`evaluate`函数）
- `models/losses.py`: 损失函数和指标计算（`AdditionACTLossHead`）

