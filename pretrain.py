from typing import Optional, Any, Sequence, List
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
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0 # when to start eval
    eval_save_outputs: List[str] = []
    total_steps: Optional[int] = None  # If set, override calculated total_steps
    max_eval_batches: Optional[int] = None  # Limit number of evaluation batches (for faster eval)

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

        # Resize and reset puzzle emb if needed
        # 根据模型是否被编译，选择正确的键名
        if is_compiled:
            puzzle_emb_name = "model._orig_mod.inner.puzzle_emb.weights"
        else:
            puzzle_emb_name = "model.inner.puzzle_emb.weights"
        expected_shape: torch.Size = model.model.puzzle_emb.weights.shape  # type: ignore
        if puzzle_emb_name in state_dict:
            puzzle_emb = state_dict[puzzle_emb_name]
            if puzzle_emb.shape != expected_shape:
                print(f"Resetting puzzle embedding as shape is different. Found {puzzle_emb.shape}, Expected {expected_shape}")
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
        train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

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
        max_batches = config.max_eval_batches
        
        for set_name, batch, global_batch_size in eval_loader:
            # Limit number of batches if max_eval_batches is set
            if max_batches is not None and processed_batches >= max_batches:
                if rank == 0:
                    print(f"Reached max_eval_batches limit ({max_batches}), stopping evaluation")
                break
                
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")
            
            # To device
            # 保存CPU副本用于后续访问（只对tensor类型的值调用clone）
            # 同时保存非tensor字段（如_dataset_start_idx, _dataset_set_name）
            batch_cpu = {}
            batch_for_model = {}  # 只包含tensor字段，用于传递给模型
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_cpu[k] = v.clone()
                    batch_for_model[k] = v.cuda()
                else:
                    # 非tensor类型（如_dataset_start_idx, _dataset_set_name）保存到batch_cpu，但不传递给模型
                    batch_cpu[k] = v
            
            # 打印batch数据信息（用于调试）
            if rank == 0:
                print(f"  Batch信息:")
                batch_size = batch_cpu.get('inputs', torch.tensor([])).shape[0] if 'inputs' in batch_cpu else 0
                print(f"    batch_size: {batch_size}")
                if 'inputs' in batch_cpu:
                    inputs_shape = batch_cpu['inputs'].shape
                    print(f"    inputs shape: {inputs_shape}")
                    # 打印前3个样本的inputs信息
                    num_samples_to_show = min(3, inputs_shape[0])
                    for sample_idx in range(num_samples_to_show):
                        sample_input = batch_cpu['inputs'][sample_idx]
                        if len(sample_input.shape) == 1:
                            print(f"    sample {sample_idx} inputs (first 20): {sample_input[:20].tolist()}")
                            # 尝试从inputs中提取两个加数（用于验证是否是第一步数据）
                            try:
                                # 转换为numpy并提取数字
                                input_seq_np = sample_input.cpu().numpy()
                                seq_len = len(input_seq_np)
                                # 转换为原始值（减去1）
                                input_seq_np = input_seq_np - 1
                                # 重塑为网格（4行n列）
                                grid_width = seq_len // 4
                                if grid_width > 0:
                                    grid = input_seq_np[:seq_len].reshape(4, grid_width)
                                    row1 = grid[0]  # 第一个加数
                                    row2 = grid[1]  # 第二个加数
                                    
                                    # 提取有效数字
                                    def extract_num_from_row(row):
                                        start_idx = 0
                                        while start_idx < len(row) and row[start_idx] == 0:
                                            start_idx += 1
                                        if start_idx >= len(row):
                                            return 0
                                        digits = []
                                        for i in range(start_idx, len(row)):
                                            val = int(row[i])
                                            if val >= 10:
                                                break
                                            digits.append(val)
                                        if len(digits) == 0:
                                            return 0
                                        num = 0
                                        for digit in digits:
                                            num = num * 10 + digit
                                        return num
                                    
                                    num1 = extract_num_from_row(row1)
                                    num2 = extract_num_from_row(row2)
                                    print(f"    sample {sample_idx} 提取的数字: num1={num1}, num2={num2}, 期望结果={num1 + num2}")
                            except Exception as e:
                                print(f"    sample {sample_idx} 无法从inputs提取数字: {e}")
                        else:
                            print(f"    sample {sample_idx} inputs shape: {sample_input.shape}")
                if 'labels' in batch_cpu:
                    labels_shape = batch_cpu['labels'].shape
                    print(f"    labels shape: {labels_shape}")
                    # 打印前3个样本的labels信息
                    num_samples_to_show = min(3, labels_shape[0])
                    for sample_idx in range(num_samples_to_show):
                        sample_label = batch_cpu['labels'][sample_idx]
                        if len(sample_label.shape) == 1:
                            print(f"    sample {sample_idx} labels (first 20): {sample_label[:20].tolist()}")
                        else:
                            print(f"    sample {sample_idx} labels shape: {sample_label.shape}")
                if 'puzzle_identifiers' in batch_cpu:
                    puzzle_ids = batch_cpu['puzzle_identifiers']
                    if isinstance(puzzle_ids, torch.Tensor):
                        puzzle_ids_list = puzzle_ids.tolist()
                        # 只显示前3个
                        num_to_show = min(3, len(puzzle_ids_list))
                        print(f"    puzzle_identifiers (first {num_to_show}): {puzzle_ids_list[:num_to_show]}")
                        if len(puzzle_ids_list) > num_to_show:
                            print(f"    puzzle_identifiers (total {len(puzzle_ids_list)}): {puzzle_ids_list}")
                    else:
                        print(f"    puzzle_identifiers: {puzzle_ids}")
                if '_dataset_start_idx' in batch_cpu:
                    start_idx = batch_cpu['_dataset_start_idx']
                    if isinstance(start_idx, torch.Tensor):
                        start_idx = start_idx.item()
                    print(f"    _dataset_start_idx: {start_idx}")
                if '_dataset_set_name' in batch_cpu:
                    print(f"    _dataset_set_name: {batch_cpu['_dataset_set_name']}")
            
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch_for_model)  # type: ignore
                # 关键修复：初始化carry，确保与test_addition_puzzle.py一致
                # initial_carry返回的carry默认halted=True，需要设置为False才能开始推理
                carry.halted = torch.zeros_like(carry.halted)
                # initial_carry返回的current_data是empty_like的，需要设置为batch的值
                # 这样第一次迭代时，如果halted=False，会使用正确的初始输入
                carry.current_data = {k: v.clone() for k, v in batch_for_model.items()}
                
                # 关键修复：在第一步时，即使halted=False，也要重置inner_carry，避免使用未初始化的值
                # 因为empty_carry创建的是未初始化的tensor，可能包含NaN
                # 在第一步时，所有序列都应该使用初始化的carry值
                base_model = train_state.model.model if hasattr(train_state.model, 'model') else train_state.model
                if hasattr(base_model, 'inner') and hasattr(base_model.inner, 'reset_carry'):
                    # 在第一步时，强制重置所有序列的carry
                    reset_all = torch.ones_like(carry.halted, dtype=torch.bool)
                    carry.inner_carry = base_model.inner.reset_carry(reset_all, carry.inner_carry)

            # Forward
            inference_steps = 0
            # 保存每一步的预测，用于与label进行比较
            all_step_preds = []  # 存储每一步的preds
            all_step_metrics = []  # 存储每一步的metrics
            all_step_labels = []  # 存储每一步的labels（每一步的目标状态）
            
            # 获取底层模型（用于推理，避免损失头可能的问题）
            # 使用底层模型进行推理，与test_addition_puzzle.py一致，确保推理结果正确
            base_model = train_state.model.model if hasattr(train_state.model, 'model') else train_state.model
            
            # 在评估模式下，始终运行到最大步数，不根据halt标记停止
            # 这样可以确保所有样本都运行相同的步数，便于比较
            # 注意：数据加载时已经过滤，只使用每道题目的第一步数据（s₀, s₁）
            halt_max_steps = getattr(config.arch, 'halt_max_steps', 16)
            
            # 保存初始输入（用于计算exact_accuracy，因为后续步骤会更新inputs为预测结果）
            initial_inputs = batch_for_model.get("inputs", None)
            if initial_inputs is not None:
                initial_inputs = initial_inputs.clone()
            
            while True:
                # 使用底层模型进行推理（与test_addition_puzzle.py一致）
                carry, outputs = base_model(carry=carry, batch=batch_for_model)
                inference_steps += 1
                
                # 在评估模式下，仍然计算halt信号用于metrics，但不用于停止循环
                if not base_model.training:
                    # 从config获取halt配置
                    no_ACT_continue = getattr(config.arch, 'no_ACT_continue', True)
                    
                    # 检查是否达到最大步数
                    is_last_step = carry.steps >= halt_max_steps
                    
                    # 计算halt信号（用于metrics，但不用于停止）
                    q_halt_logits = outputs["q_halt_logits"]
                    
                    # 检查q_halt_logits是否有NaN
                    if torch.isnan(q_halt_logits).any():
                        # 如果有NaN，将NaN位置设置为False（不halt），避免传播
                        q_halt_logits = torch.where(torch.isnan(q_halt_logits), torch.zeros_like(q_halt_logits), q_halt_logits)
                    
                    if no_ACT_continue:
                        # 如果q_halt_logits > 0，则halt
                        halt_signal = q_halt_logits > 0
                    else:
                        # 如果q_halt_logits > q_continue_logits，则halt
                        q_continue_logits = outputs.get("q_continue_logits", torch.zeros_like(q_halt_logits))
                        # 检查q_continue_logits是否有NaN
                        if torch.isnan(q_continue_logits).any():
                            q_continue_logits = torch.where(torch.isnan(q_continue_logits), torch.zeros_like(q_continue_logits), q_continue_logits)
                        halt_signal = q_halt_logits > q_continue_logits
                    
                    # 确保halt_signal的形状与carry.halted匹配
                    # q_halt_logits可能是 [batch_size] 或 [batch_size, 1]
                    if halt_signal.ndim > carry.halted.ndim:
                        halt_signal = halt_signal.squeeze(-1)
                    elif halt_signal.ndim < carry.halted.ndim:
                        halt_signal = halt_signal.unsqueeze(-1)
                    
                    # 确保halt_signal和is_last_step的形状匹配
                    if halt_signal.shape != carry.halted.shape:
                        # 如果形状不匹配，尝试broadcast
                        if halt_signal.numel() == 1:
                            halt_signal = halt_signal.expand_as(carry.halted)
                        elif carry.halted.numel() == 1:
                            carry.halted = carry.halted.expand_as(halt_signal)
                    
                    # 更新halted状态：达到最大步数或halt信号为True（用于metrics，但不用于停止）
                    carry.halted = is_last_step | halt_signal
                
                # 从outputs中提取preds
                preds = {"preds": torch.argmax(outputs["logits"], dim=-1)}
                
                # 手动计算基本的metrics（不通过损失头，避免NaN问题）
                with torch.no_grad():
                    labels = carry.current_data.get("labels", batch_for_model.get("labels", None))
                    
                    # 检查是否是加法任务（通过检查损失头类型或数据路径）
                    is_addition_task = (
                        hasattr(train_state.model, '__class__') and 
                        'AdditionACTLossHead' in train_state.model.__class__.__name__
                    ) or any('addition' in str(path).lower() for path in config.data_paths)
                    
                    if labels is not None:
                        mask = (labels != IGNORE_LABEL_ID)
                        loss_counts = mask.sum(-1)
                        loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)
                        is_correct = mask & (preds["preds"] == labels)
                        
                        # 对于加法任务，exact_accuracy应该通过比较预测结果与期望结果（num1 + num2）来计算
                        if is_addition_task:
                            # 使用AdditionACTLossHead的逻辑：从初始输入中提取两个加数，从预测中提取结果，然后比较
                            # 注意：必须使用初始输入（initial_inputs），因为后续步骤会更新inputs为预测结果
                            import numpy as np
                            # 使用初始输入，而不是batch_for_model["inputs"]（可能已被更新为预测结果）
                            inputs = initial_inputs if initial_inputs is not None else batch_for_model.get("inputs", carry.current_data.get("inputs", None))
                            if inputs is not None:
                                seq_len = int(inputs.shape[-1])
                                batch_size = int(inputs.shape[0])
                                grid_width = seq_len // 4
                                
                                # 转换为原始值（值+1格式，需要减1）
                                input_values = (inputs - 1).detach().cpu().numpy()  # B × seq_len
                                pred_values = (preds["preds"] - 1).detach().cpu().numpy()  # B × seq_len
                                
                                # 重塑为网格
                                input_grid = input_values[:, :seq_len].reshape(batch_size, 4, grid_width)
                                pred_grid = pred_values[:, :seq_len].reshape(batch_size, 4, grid_width)
                                
                                # 提取两个加数和预测结果
                                row1 = input_grid[:, 0, :]  # B × grid_width
                                row2 = input_grid[:, 1, :]  # B × grid_width
                                pred_row4 = pred_grid[:, 3, :]  # B × grid_width
                                
                                # 提取数字并比较
                                def extract_number_from_row_np(row, valid_mask):
                                    """从numpy数组行中提取数字"""
                                    for i in range(len(row)):
                                        if valid_mask[i] and row[i] != 0:
                                            start_idx = i
                                            break
                                    else:
                                        return 0
                                    num = 0
                                    for i in range(start_idx, len(row)):
                                        val = int(row[i])
                                        if val >= 10 or not valid_mask[i]:
                                            break
                                        num = num * 10 + val
                                    return num
                                
                                seq_is_correct_list = []
                                for b in range(batch_size):
                                    valid_mask_row1 = (row1[b] >= 0) & (row1[b] < 10)
                                    valid_mask_row2 = (row2[b] >= 0) & (row2[b] < 10)
                                    valid_mask_pred = (pred_row4[b] >= 0) & (pred_row4[b] < 10)
                                    
                                    num1 = extract_number_from_row_np(row1[b], valid_mask_row1)
                                    num2 = extract_number_from_row_np(row2[b], valid_mask_row2)
                                    pred_result = extract_number_from_row_np(pred_row4[b], valid_mask_pred)
                                    expected_result = num1 + num2
                                    
                                    seq_is_correct_list.append(pred_result == expected_result)
                                
                                seq_is_correct = torch.tensor(seq_is_correct_list, dtype=torch.bool, device=carry.halted.device)
                            else:
                                # 如果无法获取inputs，回退到标准逻辑
                                seq_is_correct = is_correct.sum(-1) == loss_counts
                        else:
                            # 非加法任务，使用标准逻辑
                            seq_is_correct = is_correct.sum(-1) == loss_counts
                        
                        valid_metrics = carry.halted & (loss_counts > 0)
                        
                        # 计算q_halt_accuracy：检查在halt时，q_halt_logits的预测是否正确
                        # q_halt_logits >= 0 表示模型预测应该halt（即预测结果正确）
                        # seq_is_correct 表示实际预测结果是否正确
                        # 所以 q_halt_accuracy 检查：模型预测halt时，实际结果是否真的正确
                        q_halt_logits = outputs["q_halt_logits"]
                        # 确保q_halt_logits的形状与seq_is_correct匹配
                        if q_halt_logits.ndim > seq_is_correct.ndim:
                            q_halt_logits = q_halt_logits.squeeze(-1)
                        elif q_halt_logits.ndim < seq_is_correct.ndim:
                            q_halt_logits = q_halt_logits.unsqueeze(-1)
                        
                        # 处理NaN：如果有NaN，将其替换为0（表示不halt）
                        if torch.isnan(q_halt_logits).any():
                            q_halt_logits = torch.where(torch.isnan(q_halt_logits), torch.zeros_like(q_halt_logits), q_halt_logits)
                        
                        # q_halt_logits >= 0 表示模型预测应该halt（预测结果正确）
                        q_halt_prediction = q_halt_logits >= 0
                        # 在halt时，检查q_halt预测是否正确
                        q_halt_correct = (q_halt_prediction == seq_is_correct)
                        
                        metrics = {
                            "count": valid_metrics.sum(),
                            "accuracy": torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                            "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                            "q_halt_accuracy": (valid_metrics & q_halt_correct).sum(),
                            "steps": torch.where(valid_metrics, carry.steps, 0).sum(),
                        }
                        # 计算损失（如果需要）
                        if hasattr(train_state.model, 'loss_fn'):
                            # 简化版损失计算
                            loss = torch.tensor(0.0, device=carry.halted.device)  # 评估时不需要真实损失
                        else:
                            loss = torch.tensor(0.0, device=carry.halted.device)
                    else:
                        metrics = {"count": carry.halted.sum(), "steps": carry.steps.sum()}
                        loss = torch.tensor(0.0, device=carry.halted.device)
                
                # 在评估模式下，只根据最大步数停止，不根据halt标记
                # 检查是否所有序列都达到了最大步数
                # 注意：使用inference_steps而不是carry.steps，因为halted序列的steps会被重置
                # 确保all_finish在每次迭代中都被定义
                all_finish = (inference_steps >= halt_max_steps) if carry.steps.numel() > 0 else True
                
                # 如果inference_steps达到halt_max_steps，强制退出
                if inference_steps >= halt_max_steps:
                    all_finish = True

                # 保存每一步的预测（移到CPU以节省GPU内存）
                all_step_preds.append({k: v.detach().cpu().clone() for k, v in preds.items()})
                # 保存每一步的metrics（移到CPU以节省GPU内存）
                all_step_metrics.append({k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()})

                # 在评估模式下，只使用每道题目的第一步数据（s₀, s₁）
                # 所以每一步都使用第一步的label（s₁），不需要从数据集中加载后续步骤的label
                # 注意：虽然进行多步推理，但评估时只关心最后一步的预测结果与期望结果（num1 + num2）的比较
                # 不需要每一步的label，因为中间步骤的数据不作为评估数据
                if 'labels' in batch_cpu:
                    step_label = {'labels': batch_cpu['labels'].clone()}
                else:
                    step_label = {}
                
                all_step_labels.append(step_label)

                # 关键修复：更新carry.current_data和batch为当前预测结果，用于下一步的递归推理
                # 与test_addition_puzzle.py保持一致
                if not all_finish and 'preds' in preds:
                    # 获取预测结果（token id格式）
                    preds_tensor = preds['preds']  # B × seq_len
                    
                    # 更新carry.current_data["inputs"]为预测结果
                    # 注意：preds_tensor是token id，应该可以直接用作inputs（因为模型输入就是token id）
                    if 'inputs' in carry.current_data:
                        carry.current_data["inputs"] = preds_tensor.clone()
                    # 更新batch["inputs"]为预测结果，确保同步
                    if 'inputs' in batch_for_model:
                        batch_for_model["inputs"] = preds_tensor.clone()

                if all_finish:
                    break

            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")
                # 清理GPU缓存
                torch.cuda.empty_cache()

            # 保存每一步的预测结果
            # 每一步的预测都要与对应的label进行比较
            # 注意：在递归推理中，每一步的预测应该与下一步的label比较
            # 所以我们需要保存每一步的预测和对应的label
            if len(all_step_preds) > 0:
                # 保存每一步的预测和对应的label
                for step_idx, (step_preds, step_labels) in enumerate(zip(all_step_preds, all_step_labels)):                    
                    # 保存每一步的预测
                    for k, v in step_preds.items():
                        if k in config.eval_save_outputs:
                            save_preds.setdefault(f"{k}_step_{step_idx}", [])
                            save_preds[f"{k}_step_{step_idx}"].append(v.cpu())
                    # 保存每一步的label（如果存在）
                    for k, v in step_labels.items():
                        if k in config.eval_save_outputs:
                            save_preds.setdefault(f"{k}_step_{step_idx}", [])
                            save_preds[f"{k}_step_{step_idx}"].append(v.cpu())
                
                # 使用最后一步的预测作为主要预测（用于保存，为了兼容性）
                final_preds = all_step_preds[-1]
                # 使用第一步的label作为主要label（用于保存）
                if len(all_step_labels) > 0 and 'labels' in all_step_labels[0]:
                    final_labels = {'labels': all_step_labels[0]['labels']}
                else:
                    final_labels = {}
            else:
                final_preds = preds
                final_labels = {'labels': batch_cpu['labels']} if 'labels' in batch_cpu else {}
            
            # 保存batch和最后一步的预测（为了兼容性）
            # 注意：使用batch_cpu而不是batch_for_model，因为batch_cpu包含所有字段（包括非tensor字段）
            # 对于inputs，确保保存初始输入（而不是可能被更新为预测结果的batch_for_model["inputs"]）
            for collection in (batch_cpu, final_preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs and isinstance(v, torch.Tensor):
                        # 对于inputs，如果存在initial_inputs，使用initial_inputs（确保是初始输入）
                        if k == "inputs" and initial_inputs is not None:
                            save_preds.setdefault(k, [])
                            save_preds[k].append(initial_inputs.cpu() if initial_inputs.is_cuda else initial_inputs)
                        else:
                            save_preds.setdefault(k, [])
                            save_preds[k].append(v.cpu() if v.is_cuda else v)  # Move to CPU for saving GPU memory
            
            # 保存第一步的label（为了兼容性）
            if 'labels' in final_labels and 'labels' in config.eval_save_outputs:
                save_preds.setdefault('labels', [])
                save_preds['labels'].append(final_labels['labels'].cpu())

            # 对于评估器，需要传递每一步的预测，以便评估器能够收集所有步骤的预测和对应的q_halt_logits进行投票
            # 评估器会使用q_halt_logits作为置信度分数，对所有步骤的预测进行排序和投票
            # 注意：使用batch_for_model而不是batch，因为评估器期望tensor类型的数据
            for evaluator in evaluators:
                # 对每一步的预测都调用update_batch，让评估器收集所有步骤的预测
                if len(all_step_preds) > 0:
                    for step_preds in all_step_preds:
                        evaluator.update_batch(batch_for_model, step_preds)
                else:
                    evaluator.update_batch(batch_for_model, final_preds)

            # Aggregate metrics
            # 使用所有步骤的metrics，而不仅仅是最后一步
            # 这样评估器可以评估每一步的中间结果
            set_id = set_ids[set_name]

            if metric_values is None:
                # 使用最后一步的metrics来确定keys（所有步骤的keys应该相同）
                # 注意：如果all_step_metrics为空，使用最后一步的metrics（在循环中计算的）
                final_metrics = all_step_metrics[-1] if len(all_step_metrics) > 0 else (metrics if 'metrics' in locals() else {})
                metric_keys = list(
                    sorted(final_metrics.keys())
                )  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros(
                    (len(set_ids), len(final_metrics.values())), dtype=torch.float32, device="cuda"
                )

            # 对于exact_accuracy等指标，应该只使用最后一步（halted状态）的结果
            # 因为只有halted=True时，valid_metrics才为True，这些指标才有意义
            # 注意：每个batch只累加一次（使用最后一步的metrics），多个batch之间会累加
            if len(all_step_metrics) > 0:
                # 只使用最后一步的metrics（因为只有最后一步halted=True）
                final_step_metrics = all_step_metrics[-1]
                # 累加到metric_values（多个batch之间会累加）
                # 注意：all_step_metrics在CPU上，需要移到CUDA
                batch_metrics = torch.stack([final_step_metrics[k] for k in metric_keys]).cuda()
                metric_values[set_id] += batch_metrics
            else:
                # 如果没有收集到步骤metrics，使用最后一步的metrics（向后兼容）
                # 注意：metrics在循环中已经计算，但可能已经被删除，所以需要从all_step_metrics获取
                if len(all_step_metrics) > 0:
                    # all_step_metrics在CPU上，需要移到CUDA
                    batch_metrics = torch.stack([all_step_metrics[-1][k] for k in metric_keys]).cuda()
                else:
                    # 如果all_step_metrics也为空，创建一个零metrics（不应该发生）
                    batch_metrics = torch.zeros(len(metric_keys), dtype=torch.float32, device="cuda")
                metric_values[set_id] += batch_metrics

            # 清理内存
            del carry, loss, preds, batch_for_model, all_finish, all_step_preds, all_step_labels
            if 'metrics' in locals():
                del metrics
            del all_step_metrics
            # all_step_preds和all_step_labels在后面还会用到，在处理完后再删除
            # 但我们已经将它们移到CPU了，所以GPU内存应该已经释放
            torch.cuda.empty_cache()

        # concatenate save preds
        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

        # Save preds
        if config.checkpoint_path is not None and len(save_preds):
            # Each rank save predictions independently
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(
                save_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}")
            )

        del save_preds

        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                # Postprocess
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count if count > 0 else 0.0 for k, v in m.items()}

        # Run evaluators
        if rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")
            
        for i, evaluator in enumerate(evaluators):
            if rank == 0:
                print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")
                
            # Path for saving
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            # Run and log
            metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group)
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}

                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")
                
        if rank == 0:
            print("All evaluators completed!")

    return reduced_metrics

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
    
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    # Adjust eval_interval if needed to make it a divisor of epochs
    if config.epochs % train_epochs_per_iter != 0:
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

    # Progress bar and logger
    progress_bar = None
    ema_helper = None
    tb_writer = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))  # type: ignore
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
        
        # 评估逻辑：如果设置了eval_interval，按原逻辑；否则只在训练完成后评估
        # 如果用户想要只在训练完成后评估，可以设置eval_interval=None
        should_eval = False
        if config.eval_interval is None:
            # 如果没有设置eval_interval，只在训练完成后评估
            should_eval = is_training_complete
        else:
            # 如果设置了eval_interval，按原逻辑（中间评估 + 最后评估）
            # 但也要确保在训练完成时评估
            should_eval = (_iter_id >= config.min_eval_interval) or is_training_complete
        
        if should_eval:
            ############ Evaluation
            if RANK == 0:
                if is_training_complete:
                    print("EVALUATE (Training Complete)")
                else:
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
                cpu_group=CPU_PROCESS_GROUP,
                eval_dataset=eval_dataset)  # 传递dataset以便访问原始数据

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                # 同时记录到TensorBoard
                if tb_writer is not None:
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            tb_writer.add_scalar(f"eval/{key}", value, train_state.step)
                
            ############ Checkpointing
            if RANK == 0:
                print("SAVE CHECKPOINT")
            if RANK == 0 and (config.checkpoint_every_eval or is_training_complete):
                save_train_state(config, train_state_eval)

            if config.ema:
                del train_state_eval
            
            # 如果训练完成，退出循环
            if is_training_complete:
                break
        
        # Increment iteration counter
        _iter_id += 1
        
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
