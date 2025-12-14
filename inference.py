#!/usr/bin/env python3
"""
使用checkpoint进行推理的脚本
"""
import os
import torch
import torch.nn as nn
from omegaconf import DictConfig
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from pretrain import (
    PretrainConfig, 
    load_synced_config,
    create_model,
    create_dataloader,
    create_evaluators,
    load_checkpoint,
    evaluate
)
from utils.functions import load_model_class
import torch.distributed as dist


def run_inference(
    checkpoint_path: str,
    data_paths: list,
    config_path: str = "config",
    config_name: str = "cfg_pretrain",
    max_eval_batches: int = None,
    eval_save_outputs: list = None,
):
    """
    使用checkpoint进行推理
    
    Args:
        checkpoint_path: checkpoint文件路径（例如：checkpoints/.../step_606）
        data_paths: 数据集路径列表
        config_path: 配置文件目录
        config_name: 配置文件名
        max_eval_batches: 最大评估batch数量（None表示全部）
        eval_save_outputs: 要保存的输出键列表
    """
    # 初始化分布式（单GPU推理）
    RANK = 0
    WORLD_SIZE = 1
    
    # 单GPU推理时，使用file://初始化方式（不需要MASTER_ADDR等环境变量）
    if not dist.is_initialized():
        import tempfile
        tmp_file = tempfile.mktemp()
        try:
            if torch.cuda.is_available():
                dist.init_process_group(
                    backend="nccl",
                    init_method=f"file://{tmp_file}",
                    rank=RANK,
                    world_size=WORLD_SIZE,
                )
            else:
                dist.init_process_group(
                    backend="gloo",
                    init_method=f"file://{tmp_file}",
                    rank=RANK,
                    world_size=WORLD_SIZE,
                )
        finally:
            # 清理临时文件
            try:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
            except:
                pass
    
    # 加载配置
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    with initialize_config_dir(config_dir=os.path.abspath(config_path), version_base=None):
        hydra_config = compose(config_name=config_name)
    
    # 覆盖数据路径和checkpoint路径（使用OmegaConf的set方法）
    from omegaconf import OmegaConf
    import yaml
    OmegaConf.set_struct(hydra_config, False)  # 临时禁用struct模式以允许添加新键
    hydra_config.data_paths = data_paths
    hydra_config.data_paths_test = []
    hydra_config.load_checkpoint = checkpoint_path
    if max_eval_batches is not None:
        hydra_config.max_eval_batches = max_eval_batches
    if eval_save_outputs is not None:
        hydra_config.eval_save_outputs = eval_save_outputs
    
    # 尝试从checkpoint目录加载训练时的完整配置（确保与训练时一致）
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_config_path = os.path.join(checkpoint_dir, "all_config.yaml")
    loaded_config_from_checkpoint = False
    if os.path.exists(checkpoint_config_path):
        try:
            with open(checkpoint_config_path, 'r') as f:
                checkpoint_config = yaml.safe_load(f)
            # 如果checkpoint配置中有arch设置，使用它（确保模型架构与训练时一致）
            if checkpoint_config and 'arch' in checkpoint_config:
                print("从checkpoint配置加载完整的arch配置，确保与训练时一致")
                # 加载完整的arch配置
                for key, value in checkpoint_config['arch'].items():
                    if key == 'loss':
                        # 对于loss配置，需要特殊处理
                        if isinstance(value, dict):
                            for loss_key, loss_value in value.items():
                                setattr(hydra_config.arch.loss, loss_key, loss_value)
                        print(f"  损失头: {checkpoint_config['arch']['loss'].get('name', '未知')}")
                    else:
                        # 对于其他arch参数，直接覆盖
                        setattr(hydra_config.arch, key, value)
                loaded_config_from_checkpoint = True
        except Exception as e:
            print(f"警告: 无法加载checkpoint配置 ({checkpoint_config_path}): {e}")
    
    # 如果没有从checkpoint加载到配置，且数据路径包含"addition"，自动使用AdditionACTLossHead
    if not loaded_config_from_checkpoint:
        # 检查数据路径是否包含"addition"
        is_addition_task = any('addition' in str(path).lower() for path in data_paths)
        if is_addition_task and hydra_config.arch.loss.name == 'losses@ACTLossHead':
            hydra_config.arch.loss.name = 'losses@AdditionACTLossHead'
            print(f"检测到加法任务，自动使用损失头: {hydra_config.arch.loss.name}")
            # 如果没有设置copy_loss_weight，设置默认值0.0（只监控，不参与训练）
            if not hasattr(hydra_config.arch.loss, 'copy_loss_weight'):
                setattr(hydra_config.arch.loss, 'copy_loss_weight', 0.0)
    
    OmegaConf.set_struct(hydra_config, True)  # 重新启用struct模式
    
    # 转换为PretrainConfig
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)
    
    # 设置checkpoint路径用于保存结果（使用checkpoint所在的目录）
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config.checkpoint_path = checkpoint_dir
    # 确保目录存在
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("=" * 60)
    print("推理配置")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"数据集: {data_paths}")
    print(f"最大batch数: {max_eval_batches or '全部'}")
    print(f"保存输出: {config.eval_save_outputs}")
    print(f"使用的配置文件: {config_name}")
    print(f"损失头: {config.arch.loss.name}")
    if hasattr(config.arch.loss, 'copy_loss_weight'):
        print(f"copy_loss_weight: {config.arch.loss.copy_loss_weight}")
    print(f"halt_max_steps: {config.arch.halt_max_steps}")
    print(f"H_cycles: {config.arch.H_cycles}")
    print(f"L_cycles: {config.arch.L_cycles}")
    print("=" * 60)
    
    # 创建数据加载器（使用test模式）
    eval_loader, eval_metadata, eval_dataset = create_dataloader(
        config, 
        split="test", 
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=RANK, 
        world_size=WORLD_SIZE
    )
    
    # 创建模型（create_model返回model, optimizers, optimizer_lrs）
    model, optimizers, optimizer_lrs = create_model(config, eval_metadata, rank=RANK, world_size=WORLD_SIZE)
    model.eval()
    
    # checkpoint已经在create_model中加载了，这里不需要再次加载
    
    # 创建评估器
    try:
        evaluators = create_evaluators(config, eval_metadata)
    except Exception as e:
        print(f"创建评估器失败: {e}")
        evaluators = []
    
    # 创建TrainState（简化版，只用于推理）
    class SimpleTrainState:
        def __init__(self, model, step):
            self.model = model
            self.step = step
    
    # 从checkpoint文件名提取step
    step = int(os.path.basename(checkpoint_path).split("_")[-1]) if "_" in os.path.basename(checkpoint_path) else 0
    train_state = SimpleTrainState(model, step)
    
    # 运行评估/推理
    print("\n开始推理...")
    metrics = evaluate(
        config=config,
        train_state=train_state,
        eval_loader=eval_loader,
        eval_metadata=eval_metadata,
        evaluators=evaluators,
        rank=RANK,
        world_size=WORLD_SIZE,
        cpu_group=None,
        eval_dataset=eval_dataset,  # 传递dataset以便访问原始数据
    )
    
    # 打印结果
    if RANK == 0 and metrics is not None:
        print("\n" + "=" * 60)
        print("推理结果")
        print("=" * 60)
        for key, value in metrics.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
            else:
                print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
        print("=" * 60)
    
    # 清理
    if dist.is_initialized():
        dist.destroy_process_group()
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用checkpoint进行推理")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint文件路径（例如：checkpoints/Arc1concept-aug-1000-ACT-torch/my_training_run/step_606）"
    )
    parser.add_argument(
        "--data_paths",
        type=str,
        nargs="+",
        default=["data/arc1concept-aug-1000"],
        help="数据集路径列表"
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="最大评估batch数量（默认：全部）"
    )
    parser.add_argument(
        "--save_outputs",
        type=str,
        nargs="+",
        default=["preds", "inputs"],
        help="要保存的输出键（默认：preds, inputs）"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config",
        help="配置文件目录"
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="cfg_pretrain",
        help="配置文件名"
    )
    
    args = parser.parse_args()
    
    # 运行推理
    run_inference(
        checkpoint_path=args.checkpoint,
        data_paths=args.data_paths,
        config_path=args.config_path,
        config_name=args.config_name,
        max_eval_batches=args.max_batches,
        eval_save_outputs=args.save_outputs,
    )

