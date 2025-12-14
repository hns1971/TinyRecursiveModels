# pretrain.py 详细解析

这是 TRM 项目的核心训练脚本，使用 Hydra 进行配置管理，支持分布式训练、EMA、评估等功能。

## 📋 目录结构

1. [配置类定义](#配置类定义)
2. [核心数据结构](#核心数据结构)
3. [数据加载](#数据加载)
4. [模型创建](#模型创建)
5. [训练流程](#训练流程)
6. [评估流程](#评估流程)
7. [主函数](#主函数)

---

## 配置类定义

### 1. LossConfig / ArchConfig / EvaluatorConfig

```python
class LossConfig(pydantic.BaseModel):
    name: str  # 损失函数类名，如 "losses@ACTLossHead"
    
class ArchConfig(pydantic.BaseModel):
    name: str  # 模型架构类名，如 "recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1"
    loss: LossConfig
    
class EvaluatorConfig(pydantic.BaseModel):
    name: str  # 评估器类名，如 "arc@ARC"
```

**作用**：使用 Pydantic 进行类型验证和配置管理，支持 `extra='allow'` 以允许额外参数。

### 2. PretrainConfig

```44:84:pretrain.py
class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_paths: List[str]
    data_paths_test: List[str] = []
    # Evaluators
    evaluators: List[EvaluatorConfig] = []

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0 # when to start eval
    eval_save_outputs: List[str] = []

    ema: bool = False # use Exponential-Moving-Average
    ema_rate: float = 0.999 # EMA-rate
    freeze_weights: bool = False # If True, freeze weights and only learn the embeddings
```

**关键参数说明**：
- `data_paths`: 训练数据集路径列表
- `data_paths_test`: 测试数据集路径（可选）
- `global_batch_size`: 全局批次大小（在多 GPU 时会自动分配到各 GPU）
- `eval_interval`: 每 N 个 epoch 评估一次
- `ema`: 是否使用指数移动平均（通常能提升模型性能）

---

## 核心数据结构

### TrainState

```86:94:pretrain.py
@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int
```

**作用**：保存训练状态
- `carry`: 模型的状态（用于递归推理，在训练过程中保持）
- `step`: 当前训练步数
- `total_steps`: 总训练步数（根据 epochs 和数据集大小计算）

---

## 数据加载

### create_dataloader

```97:113:pretrain.py
def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if len(config.data_paths_test)>0 and split=="test" else config.data_paths,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata
```

**功能**：
1. 创建 `PuzzleDataset`（支持分布式数据分片）
2. 配置 DataLoader（`batch_size=None` 因为数据集自己处理批次）
3. 返回数据加载器和元数据

**分布式支持**：
- `rank`: 当前进程的 rank（0 到 world_size-1）
- `num_replicas`: 总进程数
- 每个进程只加载分配给它的数据分片

---

## 模型创建

### create_model

```116:192:pretrain.py
def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False  # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)  # type: ignore

        # Load checkpoint
        if rank == 0:
            load_checkpoint(model, config)

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Optimizers and lr
    if config.arch.puzzle_emb_ndim == 0:
        optimizers = [
            AdamATan2(
                model.parameters(),
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [
            config.lr
        ]
    elif config.freeze_weights:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr
        ]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            ),
            AdamATan2(
                model.parameters(),
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr,
            config.lr
        ]

    return model, optimizers, optimizer_lrs
```

**关键步骤**：

1. **构建模型配置**：
   - 从 `config.arch` 提取所有额外参数
   - 设置批次大小（全局批次大小除以 GPU 数量）
   - 设置词汇表大小、序列长度等

2. **动态加载模型类**：
   - `load_model_class()` 根据字符串名称加载模型类（如 `"recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1"`）
   - 先创建模型，再包装损失函数头

3. **torch.compile**：
   - 如果未设置 `DISABLE_COMPILE` 环境变量，会编译模型以加速

4. **分布式同步**：
   - Rank 0 加载检查点
   - 然后广播参数到所有进程

5. **优化器配置**：
   - **情况 1**：无 puzzle embedding → 只用 AdamATan2
   - **情况 2**：冻结权重 → 只用 SignSGD 训练 puzzle embedding
   - **情况 3**：正常训练 → 两个优化器（puzzle embedding + 模型权重）

---

## 训练流程

### train_batch

```289:343:pretrain.py
def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    train_state.step += 1
    if train_state.step > train_state.total_steps:  # At most train_total_steps
        return

    # To device
    batch = {k: v.cuda() for k, v in batch.items()}

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward
    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    ((1 / global_batch_size) * loss).backward()

    # Allreduce
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
            
    # Apply optimizer
    lr_this_step = None    
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
            
        optim.step()
        optim.zero_grad()

    # Reduce metrics
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
        # Reduce and reconstruct
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            
            # Postprocess
            count = max(reduced_metrics["count"], 1)  # Avoid NaNs
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics
```

**训练步骤详解**：

1. **初始化 carry**：
   - `carry` 是模型的状态（用于递归推理）
   - 第一次调用时初始化，之后在训练过程中保持

2. **前向传播**：
   - 模型返回：`carry, loss, metrics, _, _`
   - `carry` 会被更新并保存到 `train_state.carry`

3. **反向传播**：
   - 损失除以 `global_batch_size`（梯度累积）
   - 然后调用 `backward()`

4. **梯度同步（分布式）**：
   - 如果多 GPU，使用 `all_reduce` 同步梯度
   - 所有进程的梯度会被求和

5. **优化器更新**：
   - 计算当前步的学习率（带 warmup 的余弦退火）
   - 更新所有优化器
   - 清零梯度

6. **指标聚合**：
   - 收集所有指标
   - 在多 GPU 时，reduce 到 rank 0
   - 只在 rank 0 返回指标（用于日志记录）

### 学习率调度

```207:214:pretrain.py
def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))
```

**学习率策略**：
- **Warmup 阶段**：线性增长到 `base_lr`
- **余弦退火**：从 `base_lr` 衰减到 `base_lr * min_ratio`
- `num_cycles=0.5` 表示半个余弦周期

---

## 评估流程

### evaluate

```345:486:pretrain.py
def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
):
    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        # Run evaluation
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}
        save_preds = {}
        metric_keys = []
        metric_values = None
        carry = None
        processed_batches = 0
        
        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")
            
            # To device
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1

                if all_finish:
                    break

            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            del carry, loss, preds, batch, all_finish

            # Aggregate metrics
            set_id = set_ids[set_name]
            if metric_values is None:
                metric_keys = list(sorted(metrics.keys()))
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda"
                )
            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])
            del metrics

        # ... 保存预测结果、聚合指标、运行评估器 ...
```

**评估关键点**：

1. **推理模式**：
   - 使用 `torch.inference_mode()` 禁用梯度计算

2. **递归推理循环**：
   ```python
   while True:
       carry, loss, metrics, preds, all_finish = model(...)
       if all_finish:
           break
   ```
   - 模型会递归推理直到 `all_finish=True`
   - 这是 TRM 的核心：模型会多次迭代改进答案

3. **评估器**：
   - 每个 batch 后调用 `evaluator.update_batch()`
   - 最后调用 `evaluator.result()` 计算最终指标（如 ARC 准确率）

---

## 主函数

### launch

```535:654:pretrain.py
@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
        # CPU GLOO process group
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")
        assert (
            dist.get_rank(CPU_PROCESS_GROUP) == RANK and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
        )

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    try:
        eval_loader,  eval_metadata  = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    except:
        print("NO EVAL DATA FOUND")
        eval_loader = eval_metadata = None

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except:
        print("No evaluator found")
        evaluators = []

    # Train state
    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)

    # Progress bar and logger
    progress_bar = None
    ema_helper = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))  # type: ignore
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config)
    if config.ema:
        print('Setup EMA')
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    # Training Loop
    for _iter_id in range(total_iters):
        print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        ############ Train Iter
        if RANK == 0:
            print("TRAIN")
        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
            if config.ema:
                ema_helper.update(train_state.model)

        if _iter_id >= config.min_eval_interval:
            ############ Evaluation
            if RANK == 0:
                print("EVALUATE")
            if config.ema:
                print("SWITCH TO EMA")
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state
            train_state_eval.model.eval()
            metrics = evaluate(config, 
                train_state_eval, 
                eval_loader, 
                eval_metadata, 
                evaluators,
                rank=RANK, 
                world_size=WORLD_SIZE,
                cpu_group=CPU_PROCESS_GROUP)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                
            ############ Checkpointing
            if RANK == 0:
                print("SAVE CHECKPOINT")
            if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
                save_train_state(config, train_state_eval)

            if config.ema:
                del train_state_eval

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()
```

**主流程**：

1. **分布式初始化**：
   - 检查 `LOCAL_RANK` 环境变量（由 `torchrun` 设置）
   - 初始化 NCCL（GPU 通信）和 GLOO（CPU 通信）进程组

2. **配置加载**：
   - Rank 0 加载配置并广播到其他进程

3. **数据加载**：
   - 创建训练和测试数据加载器
   - 如果测试数据不存在，继续训练但不评估

4. **初始化训练状态**：
   - 创建模型、优化器
   - 初始化 WandB（只在 rank 0）
   - 如果启用 EMA，创建 EMA helper

5. **训练循环**：
   - 每个 iteration 训练 `train_epochs_per_iter` 个 epoch
   - 训练后如果达到评估间隔，进行评估
   - 如果启用 EMA，评估时使用 EMA 版本的模型

6. **检查点保存**：
   - 根据配置保存模型权重

---

## 关键设计特点

### 1. 递归推理的 carry 机制

`carry` 是模型的状态，在训练过程中保持：
- 训练时：每个 batch 更新 `carry`
- 评估时：每个 batch 重新初始化 `carry`，然后递归推理直到完成

### 2. 分布式训练支持

- **数据并行**：每个进程处理不同的数据分片
- **梯度同步**：使用 `all_reduce` 同步梯度
- **参数同步**：初始化时从 rank 0 广播参数

### 3. EMA（指数移动平均）

- 训练时持续更新 EMA 权重
- 评估时使用 EMA 版本的模型（通常性能更好）

### 4. 灵活的优化器配置

- 支持只训练 puzzle embedding
- 支持冻结权重只训练 embedding
- 支持同时训练 embedding 和模型权重

### 5. Hydra 配置管理

- 使用 Hydra 进行配置管理
- 支持命令行覆盖配置
- 自动保存配置到检查点目录

---

## 使用示例

```bash
# 单 GPU 训练
python pretrain.py \
  arch=trm \
  data_paths="[data/arc1concept-aug-1000]" \
  +run_name=my_run \
  ema=True

# 多 GPU 训练
torchrun --nproc-per-node 4 pretrain.py \
  arch=trm \
  data_paths="[data/arc1concept-aug-1000]" \
  +run_name=my_run \
  ema=True
```

---

## 常见问题

### Q: 为什么 `batch_size=None`？
A: `PuzzleDataset` 自己处理批次构建，返回的每个 item 已经是一个 batch。

### Q: carry 是什么？
A: 模型的状态，用于递归推理。训练时保持，评估时每个 batch 重新初始化。

### Q: 如何从检查点恢复训练？
A: 使用 `load_checkpoint=path/to/checkpoint` 参数。注意：当前代码只保存模型权重，不保存优化器状态。

### Q: EMA 如何工作？
A: 训练时持续更新 EMA 权重（`ema_helper.update()`），评估时创建 EMA 版本的模型副本。

