#!/usr/bin/env python3
"""
计算TRM模型的参数量
"""
import torch
import torch.nn as nn
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from dataset.common import PuzzleDatasetMetadata
import json
import os

def calculate_params(config_dict, vocab_size=12, seq_len=900, num_puzzle_identifiers=1000):
    """
    计算模型参数量
    
    Args:
        config_dict: 模型配置字典
        vocab_size: 词汇表大小（默认12：0-9 + PAD + EOS）
        seq_len: 序列长度（默认900：30x30）
        num_puzzle_identifiers: puzzle标识符数量
    """
    # 创建模型配置
    model_cfg = dict(
        **config_dict,
        batch_size=64,  # 不影响参数量
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_puzzle_identifiers=num_puzzle_identifiers,
        causal=False
    )
    
    # 创建模型
    model = TinyRecursiveReasoningModel_ACTV1(model_cfg)
    
    # 计算总参数量（只计算model.parameters()，与训练代码一致）
    # puzzle_emb使用单独的优化器，不包含在model.parameters()中
    total_params = sum(p.numel() for p in model.parameters())
    
    # 计算puzzle_emb的参数（单独统计，使用Buffer存储）
    puzzle_emb_params = 0
    if hasattr(model.inner, 'puzzle_emb'):
        puzzle_emb_params = model.inner.puzzle_emb.weights.numel()
    
    # 详细统计各部分参数量
    param_details = {}
    
    # 遍历所有模块
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子节点
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                param_details[name] = params
    
    return total_params, param_details, puzzle_emb_params

def print_param_breakdown(param_details):
    """打印参数详细分解"""
    print("\n" + "="*60)
    print("参数详细分解")
    print("="*60)
    
    # 按模块分组
    modules = {}
    for name, params in param_details.items():
        parts = name.split('.')
        if len(parts) >= 2:
            module_name = '.'.join(parts[:2])
            if module_name not in modules:
                modules[module_name] = []
            modules[module_name].append((name, params))
        else:
            if 'root' not in modules:
                modules['root'] = []
            modules['root'].append((name, params))
    
    total = 0
    for module_name in sorted(modules.keys()):
        module_params = modules[module_name]
        module_total = sum(p for _, p in module_params)
        total += module_total
        print(f"\n{module_name}: {module_total:,}")
        for name, params in sorted(module_params):
            print(f"  {name}: {params:,}")
    
    print(f"\n总计: {total:,}")

if __name__ == "__main__":
    # 默认配置（来自trm.yaml）
    config = {
        "puzzle_emb_ndim": 512,
        "H_cycles": 3,
        "L_cycles": 6,
        "H_layers": 0,
        "L_layers": 2,
        "hidden_size": 512,
        "expansion": 4,
        "num_heads": 8,
        "pos_encodings": "rope",
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "halt_max_steps": 16,
        "halt_exploration_prob": 0.1,
        "forward_dtype": "bfloat16",
        "mlp_t": False,
        "puzzle_emb_len": 16,
        "no_ACT_continue": True
    }
    
    # 尝试从实际数据集获取配置
    dataset_path = "data/arc1concept-aug-1000"
    if os.path.exists(dataset_path):
        metadata_path = os.path.join(dataset_path, "train", "dataset.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            vocab_size = metadata['vocab_size']
            seq_len = metadata['seq_len']
            num_puzzle_identifiers = metadata['num_puzzle_identifiers']
            print(f"从数据集加载配置:")
            print(f"  vocab_size: {vocab_size}")
            print(f"  seq_len: {seq_len}")
            print(f"  num_puzzle_identifiers: {num_puzzle_identifiers}")
        else:
            vocab_size = 12
            seq_len = 900
            num_puzzle_identifiers = 1000
            print("使用默认配置（数据集元数据不存在）")
    else:
        vocab_size = 12
        seq_len = 900
        num_puzzle_identifiers = 1000
        print("使用默认配置（数据集不存在）")
    
    print("\n" + "="*60)
    print("TRM 模型参数量计算")
    print("="*60)
    print(f"\n模型配置:")
    print(f"  hidden_size: {config['hidden_size']}")
    print(f"  num_heads: {config['num_heads']}")
    print(f"  L_layers: {config['L_layers']}")
    print(f"  expansion: {config['expansion']}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  puzzle_emb_ndim: {config['puzzle_emb_ndim']}")
    print(f"  num_puzzle_identifiers: {num_puzzle_identifiers}")
    
    # 计算参数量
    total_params, param_details, puzzle_emb_params = calculate_params(
        config, 
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_puzzle_identifiers=num_puzzle_identifiers
    )
    
    print("\n" + "="*60)
    print(f"模型参数量 (model.parameters()): {total_params:,}")
    print(f"模型参数量 (M): {total_params / 1e6:.2f}M")
    if puzzle_emb_params > 0:
        print(f"\nPuzzle Embeddings参数量: {puzzle_emb_params:,}")
        print(f"Puzzle Embeddings参数量 (M): {puzzle_emb_params / 1e6:.2f}M")
        print(f"\n总参数量 (包含puzzle_emb): {total_params + puzzle_emb_params:,}")
        print(f"总参数量 (M): {(total_params + puzzle_emb_params) / 1e6:.2f}M")
    print("="*60)
    
    # 打印详细分解
    print_param_breakdown(param_details)

