from typing import Optional, Any, Sequence, List, Dict, Tuple
from dataclasses import dataclass
import os
import math
import yaml
import shutil
import copy
import numpy as np

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
from adam_atan2 import AdamATan2

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.ema import EMAHelper
from models.losses import IGNORE_LABEL_ID


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


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
    eval_interval: Optional[int] = None  # Evaluation interval in epochs
    eval_interval_steps: Optional[int] = None  # Evaluation interval in steps (takes precedence over eval_interval if set)
    min_eval_interval: Optional[int] = 0 # when to start eval (in iterations if using eval_interval, or steps if using eval_interval_steps)
    eval_save_outputs: List[str] = []
    total_steps: Optional[int] = None  # If set, override calculated total_steps
    max_eval_batches: Optional[int] = None  # Limit number of evaluation batches (for faster eval)
    early_stopping_patience: Optional[int] = None  # Early stopping patience: stop training if metric doesn't improve for N evaluations
    early_stopping_metric: str = "exact_accuracy"  # Metric name to monitor for early stopping

    ema: bool = False # use Exponential-Moving-Average
    ema_rate: float = 0.999 # EMA-rate
    freeze_weights: bool = False # If True, freeze weights and only learn the embeddings
    train_only_q_head: bool = False # If True, freeze main model and only train q_head (halt head)

@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any
    step: int
    total_steps: int
    scaler: Optional[GradScaler] = None


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
    return dataloader, dataset.metadata, dataset  # 返回dataset以便访问原始数据


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
            # 对于AdditionACTLossHead，不编译整个模型（因为使用了numpy转换，与torch.compile不兼容）
            # 只编译底层模型
            if hasattr(model, 'model') and config.arch.loss.name == 'losses@AdditionACTLossHead':
                # 只编译底层模型，不编译损失头
                model.model = torch.compile(model.model)  # type: ignore
            else:
                # 其他情况，编译整个模型
                model = torch.compile(model)  # type: ignore

        # Load checkpoint
        if rank == 0:
            load_checkpoint(model, config)

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)
        
        # Freeze main model and only train q_head if requested
        if config.train_only_q_head:
            # Freeze all parameters first
            for param in model.parameters():
                param.requires_grad = False
            
            # Unfreeze q_head parameters
            # Handle both compiled and uncompiled models
            base_model = model.model if hasattr(model, 'model') else model
            q_head = None
            
            # Try to find q_head (handle torch.compile wrapping)
            if hasattr(base_model, 'inner'):
                q_head = base_model.inner.q_head
            elif hasattr(base_model, '_orig_mod') and hasattr(base_model._orig_mod, 'inner'):
                q_head = base_model._orig_mod.inner.q_head
            
            if q_head is not None:
                for param in q_head.parameters():
                    param.requires_grad = True
                if rank == 0:
                    num_params = sum(p.numel() for p in q_head.parameters())
                    print(f"✓ Frozen main model, only training q_head (halt head) with {num_params} parameters")
            else:
                if rank == 0:
                    print("⚠️  Warning: train_only_q_head=True but q_head not found, training all parameters")

    # Optimizers and lr
    if config.train_only_q_head:
        # Only optimize q_head parameters
        base_model = model.model if hasattr(model, 'model') else model
        q_head = None
        
        # Try to find q_head (handle torch.compile wrapping)
        if hasattr(base_model, 'inner') and hasattr(base_model.inner, 'q_head'):
            q_head = base_model.inner.q_head
        elif hasattr(base_model, '_orig_mod') and hasattr(base_model._orig_mod, 'inner') and hasattr(base_model._orig_mod.inner, 'q_head'):
            q_head = base_model._orig_mod.inner.q_head
        
        if q_head is not None:
            q_head_params = [p for p in q_head.parameters() if p.requires_grad]
            if len(q_head_params) > 0:
                optimizers = [
                    AdamATan2(
                        q_head_params,
                        lr=0,  # Needs to be set by scheduler
                        weight_decay=config.weight_decay,
                        betas=(config.beta1, config.beta2)
                    )
                ]
                optimizer_lrs = [config.lr]
                if rank == 0:
                    num_params = sum(p.numel() for p in q_head_params)
                    print(f"✓ Created optimizer for q_head only ({num_params} trainable parameters)")
            else:
                # Fallback: no trainable parameters found
                if rank == 0:
                    print("⚠️  Warning: q_head found but no trainable parameters, training all parameters")
                optimizers = [
                    AdamATan2(
                        model.parameters(),
                        lr=0,
                        weight_decay=config.weight_decay,
                        betas=(config.beta1, config.beta2)
                    )
                ]
                optimizer_lrs = [config.lr]
        else:
            # Fallback: optimize all parameters if q_head not found
            if rank == 0:
                print("⚠️  Warning: q_head not found, training all parameters")
            optimizers = [
                AdamATan2(
                    model.parameters(),
                    lr=0,
                    weight_decay=config.weight_decay,
                    betas=(config.beta1, config.beta2)
                )
            ]
            optimizer_lrs = [config.lr]
    elif config.arch.puzzle_emb_ndim == 0:
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

def mix_weights_direct(device, alpha, net, nets):
    sd = []
    for i in range(len(nets)):
        sd += [nets[i].state_dict()]
    sd_alpha = {}
    for k in sd[0].keys():
        comb_net = alpha[0]*sd[0][k].to(device)
        for i in range(1,len(nets)):
            comb_net += alpha[i]*sd[i][k].to(device)
        sd_alpha[k] =  comb_net
    net.load_state_dict(sd_alpha)
    return net

def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    # Total training steps: use config.total_steps if set, otherwise calculate from epochs
    if config.total_steps is not None:
        total_steps = config.total_steps
    else:
        total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    # Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, rank=rank, world_size=world_size)

    # Initialize GradScaler for AMP (use bfloat16 for better stability)
    scaler = GradScaler(enabled=True) if torch.cuda.is_available() else None

    return TrainState(
        step=0,
        total_steps=total_steps,

        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None,
        scaler=scaler
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    # FIXME: Only saved model.
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}"))


def _resize_puzzle_embedding_if_needed(model: nn.Module, state_dict: dict):
    """Helper function to resize puzzle embedding in state_dict for any key format."""
    # Get expected shape from model
    try:
        expected_shape: torch.Size = model.model.puzzle_emb.weights.shape  # type: ignore
    except:
        # If model doesn't have puzzle_emb, skip
        return
    
    # Check all possible key formats for puzzle embedding
    possible_keys = [
        "model.inner.puzzle_emb.weights",
        "model._orig_mod.inner.puzzle_emb.weights",
        "_orig_mod.model.inner.puzzle_emb.weights",
    ]
    
    for puzzle_emb_name in possible_keys:
        if puzzle_emb_name in state_dict:
            puzzle_emb = state_dict[puzzle_emb_name]
            if puzzle_emb.shape != expected_shape:
                print(f"Resetting puzzle embedding as shape is different. Found {puzzle_emb.shape}, Expected {expected_shape} (key: {puzzle_emb_name})")
                # If expected shape is larger (more puzzle identifiers), expand using mean
                if expected_shape[0] > puzzle_emb.shape[0]:
                    # Expand: use mean of existing embeddings and replicate
                    mean_emb = torch.mean(puzzle_emb, dim=0, keepdim=True)
                    state_dict[puzzle_emb_name] = mean_emb.expand(expected_shape).contiguous()
                    print(f"   ✓ Expanded puzzle_emb from {puzzle_emb.shape} to {expected_shape}")
                elif expected_shape[0] < puzzle_emb.shape[0]:
                    # Shrink: take first expected_shape[0] embeddings
                    state_dict[puzzle_emb_name] = puzzle_emb[:expected_shape[0]].contiguous()
                    print(f"   ✓ Shrunk puzzle_emb from {puzzle_emb.shape} to {expected_shape}")
                else:
                    # Same size but different shape (shouldn't happen, but handle it)
                    state_dict[puzzle_emb_name] = puzzle_emb.reshape(expected_shape).contiguous()
                    print(f"   ✓ Reshaped puzzle_emb from {puzzle_emb.shape} to {expected_shape}")
            break  # Only resize once


def load_checkpoint(model: nn.Module, config: PretrainConfig):
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint {config.load_checkpoint}")

        # Load state dict
        state_dict = torch.load(config.load_checkpoint, map_location="cuda")
        
        # 修复键名不匹配问题：checkpoint中的键名可能是 `_orig_mod.model.inner...`
        # 需要根据模型的实际结构转换键名
        # 1. 如果模型被编译：期望 `model._orig_mod.inner...`
        # 2. 如果模型未编译：期望 `model.inner...`
        # 检查模型是否被编译：检查model.model的类型或是否有_orig_mod属性
        is_compiled = False
        if hasattr(model, 'model'):
            # 检查model.model是否被编译
            model_type_str = str(type(model.model))
            # 检查是否有_orig_mod属性（编译后的模型会有这个属性）
            # 或者类型字符串包含_dynamo或OptimizedModule
            if hasattr(model.model, '_orig_mod'):
                is_compiled = True
            elif '_dynamo' in model_type_str or 'OptimizedModule' in model_type_str:
                is_compiled = True
            # 调试信息
            print(f"Debug checkpoint loading: model.model type = {model_type_str}")
            print(f"  hasattr(model.model, '_orig_mod') = {hasattr(model.model, '_orig_mod')}")
            print(f"  is_compiled = {is_compiled}")
        
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            # 如果键名是 `_orig_mod.model.inner...`，需要转换
            if key.startswith("_orig_mod.model."):
                if is_compiled:
                    # 模型被编译，转换为 `model._orig_mod.inner...`
                    new_key = key.replace("_orig_mod.model.", "model._orig_mod.", 1)
                else:
                    # 模型未编译，转换为 `model.inner...`
                    new_key = key.replace("_orig_mod.model.", "model.", 1)
            # 如果键名是 `model._orig_mod.inner...`，但模型未编译，需要转换为 `model.inner...`
            elif key.startswith("model._orig_mod.inner.") and not is_compiled:
                # checkpoint是在编译时保存的，但现在模型未编译，需要移除 `_orig_mod.`
                new_key = key.replace("model._orig_mod.inner.", "model.inner.", 1)
            new_state_dict[new_key] = value
        state_dict = new_state_dict

        # Resize puzzle embedding if needed (before first load attempt)
        _resize_puzzle_embedding_if_needed(model, state_dict)
        
        # Handle vocab size mismatch for embedding and output layers
        # Get current model's vocab size
        model_vocab_size = None
        checkpoint_vocab_size = None
        
        # Check embed_tokens
        if is_compiled:
            embed_key = "model._orig_mod.inner.embed_tokens.embedding_weight"
        else:
            embed_key = "model.inner.embed_tokens.embedding_weight"
        if embed_key in state_dict:
            checkpoint_vocab_size = state_dict[embed_key].shape[0]
            # Get model's vocab size
            if hasattr(model, 'model') and hasattr(model.model, 'inner') and hasattr(model.model.inner, 'embed_tokens'):
                model_vocab_size = model.model.inner.embed_tokens.embedding_weight.shape[0]  # type: ignore
            elif hasattr(model, '_orig_mod') and hasattr(model._orig_mod, 'model'):
                model_vocab_size = model._orig_mod.model.inner.embed_tokens.embedding_weight.shape[0]  # type: ignore
        
        if model_vocab_size is not None and checkpoint_vocab_size is not None and model_vocab_size != checkpoint_vocab_size:
            print(f"⚠️  Vocab size mismatch: checkpoint={checkpoint_vocab_size}, model={model_vocab_size}")
            print(f"   Resizing embedding and output layers to match model vocab size...")
            
            # Resize embed_tokens: take first model_vocab_size tokens from checkpoint
            if embed_key in state_dict:
                checkpoint_embed = state_dict[embed_key]
                if checkpoint_vocab_size >= model_vocab_size:
                    # Take first model_vocab_size tokens (indices 0 to model_vocab_size-1)
                    state_dict[embed_key] = checkpoint_embed[:model_vocab_size].contiguous()
                    print(f"   ✓ Resized embed_tokens: {checkpoint_embed.shape} -> {state_dict[embed_key].shape}")
                else:
                    # Pad with zeros if checkpoint vocab is smaller
                    padding = torch.zeros(model_vocab_size - checkpoint_vocab_size, checkpoint_embed.shape[1], 
                                        dtype=checkpoint_embed.dtype, device=checkpoint_embed.device)
                    state_dict[embed_key] = torch.cat([checkpoint_embed, padding], dim=0).contiguous()
                    print(f"   ✓ Padded embed_tokens: {checkpoint_embed.shape} -> {state_dict[embed_key].shape}")
            
            # Resize lm_head similarly
            if is_compiled:
                lm_head_key = "model._orig_mod.inner.lm_head.weight"
            else:
                lm_head_key = "model.inner.lm_head.weight"
            if lm_head_key in state_dict:
                checkpoint_lm_head = state_dict[lm_head_key]
                if checkpoint_vocab_size >= model_vocab_size:
                    state_dict[lm_head_key] = checkpoint_lm_head[:model_vocab_size].contiguous()
                    print(f"   ✓ Resized lm_head: {checkpoint_lm_head.shape} -> {state_dict[lm_head_key].shape}")
                else:
                    padding = torch.zeros(model_vocab_size - checkpoint_vocab_size, checkpoint_lm_head.shape[1],
                                        dtype=checkpoint_lm_head.dtype, device=checkpoint_lm_head.device)
                    state_dict[lm_head_key] = torch.cat([checkpoint_lm_head, padding], dim=0).contiguous()
                    print(f"   ✓ Padded lm_head: {checkpoint_lm_head.shape} -> {state_dict[lm_head_key].shape}")
            
            # Also check for q_head if it exists (some models have this)
            if is_compiled:
                q_head_key = "model._orig_mod.inner.q_head.weight"
            else:
                q_head_key = "model.inner.q_head.weight"
            if q_head_key in state_dict:
                checkpoint_q_head = state_dict[q_head_key]
                if checkpoint_q_head.shape[0] == checkpoint_vocab_size:
                    if checkpoint_vocab_size >= model_vocab_size:
                        state_dict[q_head_key] = checkpoint_q_head[:model_vocab_size].contiguous()
                        print(f"   ✓ Resized q_head: {checkpoint_q_head.shape} -> {state_dict[q_head_key].shape}")
                    else:
                        padding = torch.zeros(model_vocab_size - checkpoint_vocab_size, checkpoint_q_head.shape[1],
                                            dtype=checkpoint_q_head.dtype, device=checkpoint_q_head.device)
                        state_dict[q_head_key] = torch.cat([checkpoint_q_head, padding], dim=0).contiguous()
                        print(f"   ✓ Padded q_head: {checkpoint_q_head.shape} -> {state_dict[q_head_key].shape}")
        
        # 如果第一次加载失败（因为键名不匹配），尝试另一种键名格式
        # 尝试加载，如果missing_keys太多，说明键名格式不对，需要重新转换
        _resize_puzzle_embedding_if_needed(model, state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, assign=True, strict=False)
        first_missing_count = len(missing_keys)
        first_unexpected_count = len(unexpected_keys)
        
        # 如果missing_keys太多或unexpected_keys太多，可能是键名格式不对
        # 检查missing_keys中的键名格式，了解模型期望的键名格式
        if first_missing_count > 5 or first_unexpected_count > 5:
            # 检查missing_keys中的键名格式
            missing_key_sample = list(missing_keys)[0] if missing_keys else ""
            unexpected_key_sample = list(unexpected_keys)[0] if unexpected_keys else ""
            print(f"  missing_keys示例: {missing_key_sample}")
            print(f"  unexpected_keys示例: {unexpected_key_sample}")
            
            # 如果missing_keys是 _orig_mod.model.inner...，说明模型期望这个格式
            # 但checkpoint中的键名已经被转换了，需要恢复原始格式
            if missing_key_sample.startswith("_orig_mod.model."):
                # 模型期望 _orig_mod.model.inner...，但state_dict中可能没有这个格式
                # 需要从原始state_dict重新转换，但这次不转换（保持原始格式）
                print("模型期望 _orig_mod.model.inner... 格式，但state_dict中没有，尝试从原始键名恢复")
                # 重新从原始state_dict转换，但这次不转换（保持原始格式）
                original_state_dict = torch.load(config.load_checkpoint, map_location="cuda")
                # 如果原始键名是 _orig_mod.model.inner...，直接使用
                if any(key.startswith("_orig_mod.model.") for key in original_state_dict.keys()):
                    print("使用原始 _orig_mod.model.inner... 格式（不转换）")
                    _resize_puzzle_embedding_if_needed(model, original_state_dict)
                    missing_keys_orig, unexpected_keys_orig = model.load_state_dict(original_state_dict, assign=True, strict=False)
                    if len(missing_keys_orig) < first_missing_count and len(unexpected_keys_orig) < first_unexpected_count:
                        print(f"✓ 使用原始 _orig_mod.model.inner... 格式，missing_keys: {first_missing_count} -> {len(missing_keys_orig)}, unexpected_keys: {first_unexpected_count} -> {len(unexpected_keys_orig)}")
                        missing_keys, unexpected_keys = missing_keys_orig, unexpected_keys_orig
                        state_dict = original_state_dict
                    elif len(missing_keys_orig) + len(unexpected_keys_orig) < first_missing_count + first_unexpected_count:
                        print(f"✓ 使用原始 _orig_mod.model.inner... 格式（总数减少），missing_keys: {first_missing_count} -> {len(missing_keys_orig)}, unexpected_keys: {first_unexpected_count} -> {len(unexpected_keys_orig)}")
                        missing_keys, unexpected_keys = missing_keys_orig, unexpected_keys_orig
                        state_dict = original_state_dict
                # 如果原始键名是 model._orig_mod.inner...，需要转换为 _orig_mod.model.inner...
                elif any(key.startswith("model._orig_mod.inner.") for key in original_state_dict.keys()):
                    print("尝试将 model._orig_mod.inner... 转换为 _orig_mod.model.inner...")
                    converted_state_dict = {}
                    for key, value in original_state_dict.items():
                        if key.startswith("model._orig_mod.inner."):
                            new_key = key.replace("model._orig_mod.inner.", "_orig_mod.model.inner.", 1)
                            converted_state_dict[new_key] = value
                        else:
                            converted_state_dict[key] = value
                    _resize_puzzle_embedding_if_needed(model, converted_state_dict)
                    missing_keys_conv, unexpected_keys_conv = model.load_state_dict(converted_state_dict, assign=True, strict=False)
                    if len(missing_keys_conv) < first_missing_count and len(unexpected_keys_conv) < first_unexpected_count:
                        print(f"✓ 使用 _orig_mod.model.inner... 格式，missing_keys: {first_missing_count} -> {len(missing_keys_conv)}, unexpected_keys: {first_unexpected_count} -> {len(unexpected_keys_conv)}")
                        missing_keys, unexpected_keys = missing_keys_conv, unexpected_keys_conv
                        state_dict = converted_state_dict
                    elif len(missing_keys_conv) + len(unexpected_keys_conv) < first_missing_count + first_unexpected_count:
                        print(f"✓ 使用 _orig_mod.model.inner... 格式（总数减少），missing_keys: {first_missing_count} -> {len(missing_keys_conv)}, unexpected_keys: {first_unexpected_count} -> {len(unexpected_keys_conv)}")
                        missing_keys, unexpected_keys = missing_keys_conv, unexpected_keys_conv
                        state_dict = converted_state_dict
            # 如果当前是 model.inner...，尝试转换为 model._orig_mod.inner...
            elif any(key.startswith("model.inner.") for key in state_dict.keys()) and missing_key_sample.startswith("model._orig_mod."):
                print("尝试将 model.inner... 转换为 model._orig_mod.inner...")
                new_state_dict_v2 = {}
                for key, value in state_dict.items():
                    if key.startswith("model.inner."):
                        new_key = key.replace("model.inner.", "model._orig_mod.inner.", 1)
                        new_state_dict_v2[new_key] = value
                    else:
                        new_state_dict_v2[key] = value
                # 重新加载
                _resize_puzzle_embedding_if_needed(model, new_state_dict_v2)
                missing_keys_v2, unexpected_keys_v2 = model.load_state_dict(new_state_dict_v2, assign=True, strict=False)
                # 如果missing_keys和unexpected_keys都减少了，说明这个格式是对的
                if len(missing_keys_v2) < first_missing_count and len(unexpected_keys_v2) < first_unexpected_count:
                    print(f"✓ 使用 model._orig_mod.inner... 格式，missing_keys: {first_missing_count} -> {len(missing_keys_v2)}, unexpected_keys: {first_unexpected_count} -> {len(unexpected_keys_v2)}")
                    missing_keys, unexpected_keys = missing_keys_v2, unexpected_keys_v2
                    state_dict = new_state_dict_v2
                elif len(missing_keys_v2) + len(unexpected_keys_v2) < first_missing_count + first_unexpected_count:
                    print(f"✓ 使用 model._orig_mod.inner... 格式（总数减少），missing_keys: {first_missing_count} -> {len(missing_keys_v2)}, unexpected_keys: {first_unexpected_count} -> {len(unexpected_keys_v2)}")
                    missing_keys, unexpected_keys = missing_keys_v2, unexpected_keys_v2
                    state_dict = new_state_dict_v2
            # 如果当前是 model._orig_mod.inner...，尝试转换为 model.inner...
            elif any(key.startswith("model._orig_mod.inner.") for key in state_dict.keys()) and missing_key_sample.startswith("model.inner."):
                print("尝试将 model._orig_mod.inner... 转换为 model.inner...")
                new_state_dict_v2 = {}
                for key, value in state_dict.items():
                    if key.startswith("model._orig_mod.inner."):
                        new_key = key.replace("model._orig_mod.inner.", "model.inner.", 1)
                        new_state_dict_v2[new_key] = value
                    else:
                        new_state_dict_v2[key] = value
                # 重新加载
                _resize_puzzle_embedding_if_needed(model, new_state_dict_v2)
                missing_keys_v2, unexpected_keys_v2 = model.load_state_dict(new_state_dict_v2, assign=True, strict=False)
                # 如果missing_keys和unexpected_keys都减少了，说明这个格式是对的
                if len(missing_keys_v2) < first_missing_count and len(unexpected_keys_v2) < first_unexpected_count:
                    print(f"✓ 使用 model.inner... 格式，missing_keys: {first_missing_count} -> {len(missing_keys_v2)}, unexpected_keys: {first_unexpected_count} -> {len(unexpected_keys_v2)}")
                    missing_keys, unexpected_keys = missing_keys_v2, unexpected_keys_v2
                    state_dict = new_state_dict_v2
                elif len(missing_keys_v2) + len(unexpected_keys_v2) < first_missing_count + first_unexpected_count:
                    print(f"✓ 使用 model.inner... 格式（总数减少），missing_keys: {first_missing_count} -> {len(missing_keys_v2)}, unexpected_keys: {first_unexpected_count} -> {len(unexpected_keys_v2)}")
                    missing_keys, unexpected_keys = missing_keys_v2, unexpected_keys_v2
                    state_dict = new_state_dict_v2
        if missing_keys:
            print(f"⚠️  Missing keys (not loaded): {len(missing_keys)} keys")
            if len(missing_keys) <= 10:
                for key in missing_keys:
                    print(f"   - {key}")
            else:
                for key in missing_keys[:10]:
                    print(f"   - {key}")
                print(f"   ... and {len(missing_keys) - 10} more")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys (ignored): {len(unexpected_keys)} keys")
            if len(unexpected_keys) <= 10:
                for key in unexpected_keys:
                    print(f"   - {key}")
            else:
                for key in unexpected_keys[:10]:
                    print(f"   - {key}")
                print(f"   ... and {len(unexpected_keys) - 10} more")


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )



def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    data_paths =config.data_paths_test if len(config.data_paths_test)>0 else config.data_paths
    # Initialize evaluators
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "evaluators.")(
                data_path=data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__
            )  # type: ignore
            evaluators.append(cls)

    return evaluators

def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    # Check before incrementing step
    if train_state.step >= train_state.total_steps:  # At most train_total_steps
        return None  # Return None to indicate training should stop
    train_state.step += 1

    # To device
    batch = {k: v.cuda() for k, v in batch.items()}

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward with AMP
    scaler = train_state.scaler
    with autocast('cuda', dtype=torch.bfloat16, enabled=(scaler is not None)):
        # 传递训练步数和warmup步数给模型（用于动态loss权重）
        # 通过batch传递，这样loss head可以访问
        batch_with_step = batch.copy()
        batch_with_step['_training_step'] = train_state.step
        batch_with_step['_warmup_steps'] = config.lr_warmup_steps if hasattr(config, 'lr_warmup_steps') else 1000
        train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch_with_step, return_keys=[])

    # Backward with AMP
    if scaler is not None:
        scaler.scale((1 / global_batch_size) * loss).backward()
    else:
        ((1 / global_batch_size) * loss).backward()

    # Allreduce
    if world_size > 1:
        # Unscale gradients before allreduce if using AMP
        if scaler is not None:
            for optim in train_state.optimizers:
                scaler.unscale_(optim)
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
            
    # Apply optimizer with AMP
    lr_this_step = None    
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
        
        if scaler is not None:
            scaler.step(optim)
        else:
            optim.step()
        optim.zero_grad()
    
    # Update scaler once after all optimizers
    if scaler is not None:
        scaler.update()

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

def evaluate_single_batch(
    model: nn.Module,
    batch: Dict[str, Any],
    carry: Optional[Any] = None,
    training_step: int = 0,
    warmup_steps: int = 1000,
    enable_debug: bool = False,
    rank: int = 0,
) -> Tuple[Any, Dict[str, torch.Tensor], Dict[str, float]]:
    """
    评估单个 batch 的处理过程（从 evaluate 函数中抽取）
    
    Args:
        model: 模型实例
        batch: 输入 batch（字典，包含 inputs, labels 等）
        carry: 可选的初始 carry 状态（如果为 None，会从 batch 初始化）
        training_step: 当前训练步数（用于动态 loss 权重）
        warmup_steps: warmup 步数（用于动态 loss 权重）
        enable_debug: 是否启用调试输出
        rank: 当前进程的 rank（用于控制调试输出）
    
    Returns:
        tuple: (updated_carry, metrics, extracted_metrics)
            - updated_carry: 更新后的 carry 状态（可用于下一个 batch）
            - metrics: 模型返回的完整 metrics 字典
            - extracted_metrics: 提取的指标字典，包含：
                - exact_accuracy: 正确样本数（累计值）
                - q_halt_accuracy: q_halt 正确样本数（累计值）
                - count: 样本数
    """
    # To device (only for tensor values)
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # Init carry if it is None
    if carry is None:
        with torch.device("cuda"):
            carry = model.initial_carry(batch)  # type: ignore
    
    # Forward (same as training)
    # 传递训练步数和warmup步数给模型（用于动态loss权重）
    batch_with_step = batch.copy()
    batch_with_step['_training_step'] = training_step
    batch_with_step['_warmup_steps'] = warmup_steps
    
    # 启用调试输出（如果启用）
    if enable_debug and rank == 0:
        # 设置调试标志
        if hasattr(model, '_debug_eval'):
            model._debug_eval = True
            print(f"[评估] 已启用调试输出，将在调用底层模型前打印详细信息")
    
    with torch.inference_mode():
        carry, loss, metrics, detached_outputs, _ = model(carry=carry, batch=batch_with_step, return_keys=[])
    
    # 关闭调试输出
    if enable_debug and hasattr(model, '_debug_eval'):
        model._debug_eval = False
    
    # Extract metrics
    extracted_metrics = {}
    if 'exact_accuracy' in metrics:
        extracted_metrics['exact_accuracy'] = metrics['exact_accuracy'].item()
    if 'q_halt_accuracy' in metrics:
        extracted_metrics['q_halt_accuracy'] = metrics['q_halt_accuracy'].item()
    if 'count' in metrics:
        extracted_metrics['count'] = metrics.get('count', torch.tensor(1.0)).item()
    
    # 合并 metrics 和 detached_outputs，以便访问 logits, preds 等
    # 注意：detached_outputs 可能包含 return_keys 中指定的键
    # 但我们需要从模型内部获取 logits 和 preds
    # 如果 metrics 中没有，尝试从 carry 或模型状态中获取
    full_metrics = metrics.copy()
    if detached_outputs:
        full_metrics.update(detached_outputs)
    
    # 如果 metrics 中有 preds，添加到 full_metrics
    if 'preds' in metrics:
        full_metrics['preds'] = metrics['preds']
    
    return carry, full_metrics, extracted_metrics


def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
    eval_dataset=None,  # 添加dataset参数，用于访问原始数据
):
    """
    评估函数：使用训练时的 exact_accuracy 和 q_halt_accuracy 指标
    与训练时使用相同的逻辑
    """
    total_exact_accuracy = 0.0
    total_q_halt_accuracy = 0.0
    total_count = 0
    processed_batches = 0
    max_batches = config.max_eval_batches
    
    with torch.inference_mode():
        carry = None
        
        # 启用调试输出：打印传递给底层模型的内容（用于与单测对比）
        # 只在第一个batch时打印，避免输出过多
        enable_debug = True
        debug_printed = False
        
        for set_name, batch, global_batch_size in eval_loader:
            # Limit number of batches if max_eval_batches is set
            if max_batches is not None and processed_batches >= max_batches:
                if rank == 0:
                    print(f"Reached max_eval_batches limit ({max_batches}), stopping evaluation")
                break
                
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")
            
            # 使用抽取的单个 batch 处理函数
            enable_debug_this_batch = enable_debug and not debug_printed
            carry, metrics, extracted_metrics = evaluate_single_batch(
                model=train_state.model,
                batch=batch,
                carry=carry,
                training_step=train_state.step,
                warmup_steps=config.lr_warmup_steps if hasattr(config, 'lr_warmup_steps') else 1000,
                enable_debug=enable_debug_this_batch,
                rank=rank,
            )
            
            if enable_debug_this_batch:
                debug_printed = True
            
            # 累计指标
            if 'exact_accuracy' in extracted_metrics:
                total_exact_accuracy += extracted_metrics['exact_accuracy']
            if 'q_halt_accuracy' in extracted_metrics:
                total_q_halt_accuracy += extracted_metrics['q_halt_accuracy']
            if 'count' in extracted_metrics:
                total_count += extracted_metrics['count']
            
            # Clean up
            del metrics, batch
            # Note: don't delete carry here, as it's reused in the next iteration
            torch.cuda.empty_cache()
        
        # Clean up carry after evaluation loop
        del carry
        torch.cuda.empty_cache()
        
        # Reduce metrics across processes
        if world_size > 1:
            total_exact_accuracy_tensor = torch.tensor(total_exact_accuracy, device='cuda')
            total_q_halt_accuracy_tensor = torch.tensor(total_q_halt_accuracy, device='cuda')
            total_count_tensor = torch.tensor(total_count, device='cuda')
            dist.all_reduce(total_exact_accuracy_tensor)
            dist.all_reduce(total_q_halt_accuracy_tensor)
            dist.all_reduce(total_count_tensor)
            total_exact_accuracy = total_exact_accuracy_tensor.item()
            total_q_halt_accuracy = total_q_halt_accuracy_tensor.item()
            total_count = total_count_tensor.item()
            # Clean up tensors after all_reduce
            del total_exact_accuracy_tensor, total_q_halt_accuracy_tensor, total_count_tensor
            torch.cuda.empty_cache()
        
        # Calculate final metrics
        if rank == 0:
            # exact_accuracy 和 q_halt_accuracy 是累计值（正确样本数），需要除以 total_count 得到比例
            if total_count > 0:
                final_exact_accuracy = total_exact_accuracy / total_count
                final_q_halt_accuracy = total_q_halt_accuracy / total_count
            else:
                final_exact_accuracy = 0.0
                final_q_halt_accuracy = 0.0
            
            reduced_metrics = {
                'all': {
                    'exact_accuracy': final_exact_accuracy,
                    'q_halt_accuracy': final_q_halt_accuracy,
                    'count': total_count
                }
            }
            print(f"\nEvaluation Results:")
            print(f"  exact_accuracy: {final_exact_accuracy:.4f} (正确样本数: {total_exact_accuracy:.0f}, 总样本数: {total_count:.0f})")
            print(f"  q_halt_accuracy: {final_q_halt_accuracy:.4f} (正确样本数: {total_q_halt_accuracy:.0f}, 总样本数: {total_count:.0f})")
            print(f"  count: {total_count}")
            return reduced_metrics
        else:
            return None

def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


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
    # Load metadata first to calculate epochs if using total_steps
    # We need to create a minimal dataset just to get metadata
    from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
    temp_dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths,
        rank=RANK,
        num_replicas=WORLD_SIZE,
        global_batch_size=config.global_batch_size,
        test_set_mode=False,
        epochs_per_iter=1
    ), split="train")
    train_metadata = temp_dataset.metadata
    
    # If using total_steps, calculate epochs from total_steps
    if config.total_steps is not None:
        # Calculate epochs from total_steps
        # total_steps = epochs * total_groups * mean_puzzle_examples / global_batch_size
        # So: epochs = total_steps * global_batch_size / (total_groups * mean_puzzle_examples)
        calculated_epochs = int(config.total_steps * config.global_batch_size / (train_metadata.total_groups * train_metadata.mean_puzzle_examples))
        # Round up to ensure we have enough steps
        calculated_epochs = max(calculated_epochs, 1)
        if RANK == 0:
            print(f"ℹ️  Using total_steps={config.total_steps}, calculated epochs={calculated_epochs}")
            print(f"   (total_groups={train_metadata.total_groups}, mean_puzzle_examples={train_metadata.mean_puzzle_examples:.2f}, batch_size={config.global_batch_size})")
        # Override epochs with calculated value
        config.epochs = calculated_epochs
    
    # 如果设置了eval_interval_steps，不需要按epochs分割训练
    # 如果只设置了eval_interval，按原逻辑分割训练
    if config.eval_interval_steps is not None:
        # 按步数评估，不需要按epochs分割，整个训练作为一个iter
        train_epochs_per_iter = config.epochs
        total_iters = 1
        if RANK == 0:
            print(f"ℹ️  使用按步数评估模式: eval_interval_steps={config.eval_interval_steps}")
    else:
        train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
        total_iters = config.epochs // train_epochs_per_iter

    # Adjust eval_interval if needed to make it a divisor of epochs (only if using eval_interval)
    if config.eval_interval_steps is None and config.epochs % train_epochs_per_iter != 0:
        if RANK == 0:
            print(f"⚠️  Warning: eval_interval ({train_epochs_per_iter}) is not a divisor of epochs ({config.epochs})")
            # Find the largest divisor of epochs that is <= eval_interval
            for divisor in range(train_epochs_per_iter, 0, -1):
                if config.epochs % divisor == 0:
                    train_epochs_per_iter = divisor
                    total_iters = config.epochs // train_epochs_per_iter
                    if RANK == 0:
                        print(f"   Adjusted eval_interval to {train_epochs_per_iter}, total_iters={total_iters}")
                    break
    
    assert config.epochs % train_epochs_per_iter == 0, f"Eval interval ({train_epochs_per_iter}) must be a divisor of total epochs ({config.epochs})."

    # Create actual training dataloader (we already have metadata from temp_dataset)
    train_loader, _, _ = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    try:
        eval_loader, eval_metadata, eval_dataset = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    except:
        print("NO EVAL DATA FOUND")
        eval_loader = eval_metadata = eval_dataset = None

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except:
        print("No evaluator found")
        evaluators = []

    # Train state
    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)
    
    # Track last evaluation step for eval_interval_steps
    last_eval_step = 0
    
    # Early stopping tracking
    best_metric_value = None
    patience_counter = 0
    early_stopping_enabled = config.early_stopping_patience is not None and config.early_stopping_patience > 0

    # Progress bar and logger
    progress_bar = None
    ema_helper = None
    tb_writer = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True, init_timeout=120))  # type: ignore
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        # 初始化TensorBoard writer
        tb_log_dir = os.path.join("runs", config.run_name)
        os.makedirs(tb_log_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        tb_writer.add_scalar("info/num_params", sum(x.numel() for x in train_state.model.parameters()), 0)
        save_code_and_config(config)
    if config.ema:
        print('Setup EMA')
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    # Training Loop
    # When using total_steps, we need to continue training until we reach total_steps
    # even if we've completed all iterations
    _iter_id = 0
    while _iter_id < total_iters or (config.total_steps is not None and train_state.step < train_state.total_steps):
        if _iter_id < total_iters:
            print(f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        ############ Train Iter
        if RANK == 0:
            print("TRAIN")
        train_state.model.train()
        
        # Track if we processed any batches in this iteration
        batches_processed = False
        # Flag to indicate if we need to evaluate after processing batches
        need_eval_after_batches = False
        for set_name, batch, global_batch_size in train_loader:
            # Check if we've reached total_steps
            if config.total_steps is not None and train_state.step >= train_state.total_steps:
                if RANK == 0:
                    print(f"Reached total_steps ({config.total_steps}), stopping training")
                break
                
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)
            
            # If train_batch returns None, we've reached total_steps
            if metrics is None:
                if RANK == 0:
                    print(f"Reached total_steps ({train_state.total_steps}), stopping training")
                break
            
            batches_processed = True

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                # 同时记录到TensorBoard
                if tb_writer is not None:
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            tb_writer.add_scalar(f"train/{key}", value, train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
            if config.ema:
                ema_helper.update(train_state.model)
            
            # Check again after processing batch
            if config.total_steps is not None and train_state.step >= train_state.total_steps:
                if RANK == 0:
                    print(f"Reached total_steps ({config.total_steps}), stopping training")
                break
            
            # 如果使用 eval_interval_steps，在每个 batch 后检查是否应该评估
            # 注意：如果设置了 eval_interval_steps，应该立即触发评估，而不是等到 iter 结束
            # 这样可以确保按步数准确触发评估
            if config.eval_interval_steps is not None:
                min_eval_steps = config.min_eval_interval if config.min_eval_interval is not None else 0
                if train_state.step >= min_eval_steps:
                    steps_since_last_eval = train_state.step - last_eval_step
                    if steps_since_last_eval >= config.eval_interval_steps:
                        # 需要评估，立即跳出 batch 循环，进入评估逻辑
                        need_eval_after_batches = True
                        break
        
        # If using total_steps and we haven't reached it yet, but no more batches,
        # we need to recreate the dataloader to continue
        if config.total_steps is not None and train_state.step < train_state.total_steps and not batches_processed:
            if RANK == 0:
                print(f"⚠️  Warning: Reached end of dataset but haven't reached total_steps ({train_state.step}/{train_state.total_steps})")
                print(f"   Recreating dataloader to continue training...")
            # Recreate dataloader to continue training
            train_loader, _ = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
            continue

        # 检查是否训练完成（达到total_steps或完成所有iterations）
        is_training_complete = False
        if config.total_steps is not None:
            is_training_complete = train_state.step >= train_state.total_steps
        else:
            is_training_complete = (_iter_id >= total_iters)
        
        # 评估逻辑：
        # 1. 如果设置了eval_interval_steps，按步数触发评估（优先级最高）
        #    注意：如果使用eval_interval_steps，评估检查已经在batch循环内进行，这里检查标志或训练完成
        # 2. 如果设置了eval_interval，按epochs/iterations触发评估
        # 3. 否则只在训练完成后评估
        should_eval = False
        if config.eval_interval_steps is not None:
            # 按步数触发评估
            # 检查标志（在batch循环内设置的）或训练完成
            if need_eval_after_batches:
                should_eval = True
            # 训练完成时也要评估
            if is_training_complete:
                should_eval = True
        elif config.eval_interval is not None:
            # 按epochs/iterations触发评估（原逻辑）
            # 但也要确保在训练完成时评估
            should_eval = (_iter_id >= config.min_eval_interval) or is_training_complete
        else:
            # 如果没有设置eval_interval或eval_interval_steps，只在训练完成后评估
            should_eval = is_training_complete
        
        if should_eval:
            ############ Evaluation
            if RANK == 0:
                if is_training_complete:
                    print("EVALUATE (Training Complete)")
                else:
                    if config.eval_interval_steps is not None:
                        print(f"EVALUATE (Step {train_state.step}, interval: {config.eval_interval_steps} steps)")
                    else:
                        print("EVALUATE")
            # Update last evaluation step
            last_eval_step = train_state.step
            # Reset the flag after evaluation
            need_eval_after_batches = False
            if config.ema:
                print("SWITCH TO EMA")
                # 清理 GPU 缓存以释放内存
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                # 创建新的 TrainState，只复制必要的字段，避免 deepcopy 整个对象
                # 这样可以减少内存占用
                train_state_eval = TrainState(
                    model=ema_helper.ema_copy(train_state.model),
                    optimizers=train_state.optimizers,  # 评估时不需要 optimizers，但保持引用避免被回收
                    optimizer_lrs=train_state.optimizer_lrs,  # 评估时不需要
                    step=train_state.step,
                    total_steps=train_state.total_steps,
                    carry=None,  # 评估时会重新初始化
                    scaler=train_state.scaler,  # 评估时不需要 scaler
                )
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
                cpu_group=CPU_PROCESS_GROUP,
                eval_dataset=eval_dataset)  # 传递dataset以便访问原始数据

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                # 同时记录到TensorBoard
                if tb_writer is not None:
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            tb_writer.add_scalar(f"eval/{key}", value, train_state.step)
                
                # Early stopping check
                if early_stopping_enabled:
                    # Extract metric value from metrics dict
                    # metrics structure: {'all': {'exact_accuracy': value, ...}}
                    metric_value = None
                    if 'all' in metrics and config.early_stopping_metric in metrics['all']:
                        metric_value = metrics['all'][config.early_stopping_metric]
                    elif config.early_stopping_metric in metrics:
                        metric_value = metrics[config.early_stopping_metric]
                    
                    if metric_value is not None:
                        if best_metric_value is None or metric_value > best_metric_value:
                            # Metric improved
                            best_metric_value = metric_value
                            patience_counter = 0
                            print(f"✅ Early stopping: {config.early_stopping_metric} improved to {metric_value:.4f} (best: {best_metric_value:.4f}, patience: {patience_counter}/{config.early_stopping_patience})")
                        else:
                            # Metric didn't improve
                            patience_counter += 1
                            print(f"⚠️  Early stopping: {config.early_stopping_metric} = {metric_value:.4f} (best: {best_metric_value:.4f}, patience: {patience_counter}/{config.early_stopping_patience})")
                            
                            if patience_counter >= config.early_stopping_patience:
                                print(f"🛑 Early stopping triggered: {config.early_stopping_metric} hasn't improved for {config.early_stopping_patience} evaluations")
                                print(f"   Best {config.early_stopping_metric}: {best_metric_value:.4f}")
                                print(f"   Current {config.early_stopping_metric}: {metric_value:.4f}")
                                is_training_complete = True  # Trigger training completion to save checkpoint and exit
                
            ############ Checkpointing
            if RANK == 0:
                print("SAVE CHECKPOINT")
            if RANK == 0 and (config.checkpoint_every_eval or is_training_complete):
                save_train_state(config, train_state_eval)

            if config.ema:
                del train_state_eval
                # 清理GPU缓存以释放EMA模型占用的内存
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            
            # 重置训练状态的carry，确保下次训练时重新初始化
            # 这样可以避免评估时的carry状态影响训练，并释放内存
            train_state.carry = None
            # 强制清理GPU缓存
            torch.cuda.empty_cache()
            
            # 如果训练完成，退出循环
            if is_training_complete:
                break
            
            # 如果使用 eval_interval_steps 并且刚刚评估过，继续训练（不增加 iter_id）
            # 这样可以继续处理剩余的 batches，而不是重新开始 iter
            if config.eval_interval_steps is not None:
                # 评估完成，继续当前 iter 的训练（重新进入 batch 循环）
                continue
        
        # Increment iteration counter (only if not using eval_interval_steps or if iter is complete)
        if config.eval_interval_steps is None:
            _iter_id += 1
        # 如果使用 eval_interval_steps，只有在真正完成 iter 时才增加
        # 如果是因为评估而跳出，不增加 iter_id（已经在上面 continue 了）
        
        # Check if we've reached total_steps (for total_steps-based training)
        if config.total_steps is not None and train_state.step >= train_state.total_steps:
            if RANK == 0:
                print(f"✅ Reached total_steps ({config.total_steps}), training complete")
            break

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    if RANK == 0:
        wandb.finish()
        if tb_writer is not None:
            tb_writer.close()


if __name__ == "__main__":
    launch()
