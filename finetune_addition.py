#!/usr/bin/env python3
"""
在加法数据集上微调基础模型的便捷脚本

这个脚本会调用pretrain.py，并通过命令行参数覆盖配置。

使用方法:
    # 单GPU微调
    python finetune_addition.py \
        --base-checkpoint checkpoints/YourProject/base_model/step_30000 \
        --data-path data/addition \
        --run-name finetune_addition \
        --epochs 10000 \
        --lr 5e-5
    
    # 多GPU微调（使用torchrun）
    torchrun --nproc-per-node 4 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:0 \
        --nnodes=1 \
        finetune_addition.py \
        --base-checkpoint checkpoints/YourProject/base_model/step_30000 \
        --data-path data/addition \
        --run-name finetune_addition \
        --epochs 10000 \
        --lr 5e-5
"""

import os
import sys
import subprocess
import argparse
import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(
        description="在加法数据集上微调基础模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本微调
  python finetune_addition.py \\
      --base-checkpoint checkpoints/Project/base_model/step_30000 \\
      --data-path data/addition \\
      --run-name finetune_addition

  # 自定义学习率和批次大小
  python finetune_addition.py \\
      --base-checkpoint checkpoints/Project/base_model/step_30000 \\
      --data-path data/addition \\
      --run-name finetune_addition \\
      --lr 1e-4 \\
      --global-batch-size 128
        """
    )
    
    # 必需参数
    parser.add_argument("--base-checkpoint", type=str, required=True,
                       help="基础模型checkpoint路径")
    parser.add_argument("--data-path", type=str, required=True,
                       help="数据集路径（例如：data/addition）")
    parser.add_argument("--run-name", type=str, required=True,
                       help="运行名称（用于保存checkpoint和日志）")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=None,
                       help="训练轮数（默认：None，使用total_steps）")
    parser.add_argument("--total-steps", type=int, default=None,
                       help="总训练步数（默认：None，使用epochs计算。推荐：50000-200000）")
    parser.add_argument("--eval-interval", type=int, default=None,
                       help="评估间隔（单位：epochs，默认：自动设置）。可以与--total-steps同时使用")
    parser.add_argument("--lr", type=float, default=5e-5,
                       help="学习率（默认：5e-5）")
    parser.add_argument("--puzzle-emb-lr", type=float, default=1e-3,
                       help="Puzzle embedding学习率（默认：1e-3）")
    parser.add_argument("--global-batch-size", type=int, default=None,
                       help="全局批次大小（默认：使用配置文件中的值）。如果不指定，将使用cfg_finetune_addition.yaml中的值")
    
    # 配置/架构参数
    parser.add_argument("--config-name", type=str, default="cfg_finetune_addition",
                       help="Hydra 配置名（默认：cfg_finetune_addition）")
    parser.add_argument("--arch", type=str, default="trm",
                       help="架构名称（默认：trm）")
    parser.add_argument("--L-layers", type=int, default=None,
                       help="L层数")
    parser.add_argument("--H-cycles", type=int, default=None,
                       help="H循环次数")
    parser.add_argument("--L-cycles", type=int, default=None,
                       help="L循环次数")
    
    # 其他参数
    parser.add_argument("--ema", action="store_true", default=True,
                       help="使用指数移动平均（默认：True）")
    parser.add_argument("--no-ema", dest="ema", action="store_false",
                       help="禁用指数移动平均")
    parser.add_argument("--ema-rate", type=float, default=0.999,
                       help="EMA率（默认：0.999）")
    parser.add_argument("--freeze-weights", action="store_true",
                       help="冻结权重，只训练puzzle embeddings")
    
    # Copy loss参数（用于加法微调）
    parser.add_argument("--copy-loss-weight", type=float, default=1.0,
                       help="Copy loss权重（默认：1.0）。Copy loss约束前两行（加数）保持不变")
    parser.add_argument("--loss-type", type=str, default="stablemax_cross_entropy",
                       help="基础损失函数类型（默认：stablemax_cross_entropy）")
    
    # 损失权重参数（用于分阶段训练）
    # 注意：如果不指定这些参数，将使用配置文件中的值
    parser.add_argument("--lm-loss-weight", type=float, default=None,
                       help="lm_loss的权重（默认：使用配置文件中的值）。如果不指定，将使用cfg_finetune_addition.yaml中的值")
    parser.add_argument("--q-halt-loss-weight", type=float, default=None,
                       help="q_halt_loss的权重（默认：使用配置文件中的值）。如果不指定，将使用cfg_finetune_addition.yaml中的值")
    
    # 解析参数
    args = parser.parse_args()
    
    # 验证checkpoint路径
    if not os.path.exists(args.base_checkpoint):
        print(f"❌ 错误：checkpoint路径不存在: {args.base_checkpoint}")
        sys.exit(1)
    
    # 验证数据集路径
    if not os.path.exists(args.data_path):
        print(f"❌ 错误：数据集路径不存在: {args.data_path}")
        sys.exit(1)
    
    # 从数据源获取信息
    import json
    from dataset.common import PuzzleDatasetMetadata
    
    # 读取配置文件以获取默认值
    config_file_path = os.path.join("config", f"{args.config_name}.yaml")
    config_defaults = {}
    if os.path.exists(config_file_path):
        with open(config_file_path, 'r') as f:
            config_data = yaml.safe_load(f)
            if 'arch' in config_data and 'loss' in config_data.get('arch', {}):
                loss_config = config_data['arch']['loss']
                if 'lm_loss_weight' in loss_config:
                    config_defaults['lm_loss_weight'] = loss_config['lm_loss_weight']
                if 'q_halt_loss_weight' in loss_config:
                    config_defaults['q_halt_loss_weight'] = loss_config['q_halt_loss_weight']
            # 读取global_batch_size
            if 'global_batch_size' in config_data:
                config_defaults['global_batch_size'] = config_data['global_batch_size']
    
    # 加载metadata以获取推理步数和数据形状
    metadata_path = os.path.join(args.data_path, "train", "dataset.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        metadata = PuzzleDatasetMetadata(**metadata_dict)
        
        # 从metadata获取推理步数（向上取整）
        halt_max_steps = int(np.ceil(metadata.mean_puzzle_examples))
        print(f"ℹ️  从数据源获取信息:")
        print(f"   平均步骤数: {metadata.mean_puzzle_examples:.2f}")
        print(f"   推理步数 (halt_max_steps): {halt_max_steps}")
        print(f"   序列长度: {metadata.seq_len}")
        print(f"   网格宽度 (grid_width): {metadata.seq_len // 4}")
    else:
        # 如果metadata不存在，使用默认值
        halt_max_steps = 16
        print(f"⚠️  警告：找不到metadata文件，使用默认halt_max_steps={halt_max_steps}")
    
    # 如果既没有设置epochs也没有设置total_steps，使用默认值
    if args.epochs is None and args.total_steps is None:
        # 默认使用total_steps=100000（约15个epochs，对于微调来说比较合理）
        args.total_steps = 100000
        print(f"ℹ️  未指定epochs或total_steps，使用默认total_steps={args.total_steps}")
    
    # 自动设置eval_interval
    # 如果用户想要只在训练完成后评估，可以设置--eval-interval 0或--no-eval-during-training
    if args.eval_interval is None:
        # 默认行为：只在训练完成后评估（不进行中间评估）
        args.eval_interval = None
        print(f"ℹ️  评估模式: 只在训练完成后评估（不进行中间评估）")
        print(f"   提示: 如果想进行中间评估，可以使用 --eval-interval <epochs>")
    elif args.eval_interval == 0:
        # eval_interval=0 表示只在训练完成后评估
        args.eval_interval = None
        print(f"ℹ️  评估模式: 只在训练完成后评估（不进行中间评估）")
    else:
        # 用户明确指定了eval_interval，使用用户指定的值
        print(f"ℹ️  评估间隔: {args.eval_interval} epochs（将进行中间评估）")
    
    # 如果使用epochs，验证eval_interval是epochs的除数（如果eval_interval不为None）
    if args.epochs is not None and args.eval_interval is not None and args.epochs % args.eval_interval != 0:
        print(f"⚠️  警告：eval_interval ({args.eval_interval}) 不是 epochs ({args.epochs}) 的除数")
        print(f"   自动调整为: {args.epochs // (args.epochs // args.eval_interval)}")
        args.eval_interval = args.epochs // (args.epochs // args.eval_interval)
    
    # 构建pretrain.py命令
    # 注意：Hydra中列表格式应该是 data_paths=[path1,path2] 而不是 data_paths="[path1]"
    cmd = [
        sys.executable, "pretrain.py",
        "--config-name", args.config_name,  # 覆盖Hydra默认配置
        f"arch={args.arch}",
        f"+data_paths=[{args.data_path}]",  # 使用+添加新键，Hydra列表格式
        f"+load_checkpoint={args.base_checkpoint}",  # 使用+前缀添加新配置项
        f"+run_name={args.run_name}",
        f"lr={args.lr}",
        f"puzzle_emb_lr={args.puzzle_emb_lr}",
        f"ema={str(args.ema).lower()}",
        f"ema_rate={args.ema_rate}",
        "evaluators=[]",  # 加法任务不需要特殊评估器
        # 使用AdditionACTLossHead，添加copy loss
        f"arch.loss.name=losses@AdditionACTLossHead",
        f"arch.loss.loss_type={args.loss_type}",
        f"+arch.loss.copy_loss_weight={args.copy_loss_weight}",  # copy_loss_weight 可能不存在，使用+添加
    ]
    
    # 只在命令行参数明确指定时才覆盖配置文件中的值
    if args.lm_loss_weight is not None:
        cmd.append(f"arch.loss.lm_loss_weight={args.lm_loss_weight}")
    if args.q_halt_loss_weight is not None:
        cmd.append(f"arch.loss.q_halt_loss_weight={args.q_halt_loss_weight}")
    if args.global_batch_size is not None:
        cmd.append(f"global_batch_size={args.global_batch_size}")
    
    # 添加eval_interval（如果指定了）
    if args.eval_interval is not None:
        cmd.append(f"eval_interval={args.eval_interval}")
    else:
        # eval_interval=None表示只在训练完成后评估
        cmd.append("eval_interval=null")
    
    # 添加epochs或total_steps
    if args.total_steps is not None:
        cmd.append(f"+total_steps={args.total_steps}")
    if args.epochs is not None:
        cmd.append(f"epochs={args.epochs}")
    
    # 添加架构参数
    if args.L_layers is not None:
        cmd.append(f"arch.L_layers={args.L_layers}")
    if args.H_cycles is not None:
        cmd.append(f"arch.H_cycles={args.H_cycles}")
    if args.L_cycles is not None:
        cmd.append(f"arch.L_cycles={args.L_cycles}")
    if args.freeze_weights:
        cmd.append("freeze_weights=True")
    
    # 添加从数据源获取的halt_max_steps
    if 'halt_max_steps' in locals():
        cmd.append(f"arch.halt_max_steps={halt_max_steps}")
    
    # 打印配置信息
    print("=" * 60)
    print("微调配置")
    print("=" * 60)
    print(f"配置文件: {args.config_name}")
    print(f"基础checkpoint: {args.base_checkpoint}")
    print(f"数据集路径: {args.data_path}")
    print(f"运行名称: {args.run_name}")
    if args.total_steps is not None:
        print(f"总训练步数: {args.total_steps:,}")
    if args.epochs is not None:
        print(f"训练轮数: {args.epochs}")
    print(f"评估间隔: {args.eval_interval}")
    print(f"学习率: {args.lr}")
    print(f"Puzzle embedding学习率: {args.puzzle_emb_lr}")
    if args.global_batch_size is not None:
        print(f"全局批次大小: {args.global_batch_size} (命令行指定)")
    else:
        config_value = config_defaults.get('global_batch_size', '未设置')
        print(f"全局批次大小: {config_value} (配置文件: {args.config_name}.yaml)")
    print(f"EMA: {args.ema}")
    if args.freeze_weights:
        print("⚠️  权重已冻结，只训练puzzle embeddings")
    print(f"损失函数: AdditionACTLossHead")
    print(f"  - Copy loss: 已禁用（不参与训练）")
    print(f"基础损失类型: {args.loss_type}")
    if args.lm_loss_weight is not None:
        print(f"lm_loss权重: {args.lm_loss_weight} (命令行指定)")
    else:
        config_value = config_defaults.get('lm_loss_weight', '未设置')
        print(f"lm_loss权重: {config_value} (配置文件: {args.config_name}.yaml)")
    if args.q_halt_loss_weight is not None:
        print(f"q_halt_loss权重: {args.q_halt_loss_weight} (命令行指定)")
    else:
        config_value = config_defaults.get('q_halt_loss_weight', '未设置')
        print(f"q_halt_loss权重: {config_value} (配置文件: {args.config_name}.yaml)")
    if 'halt_max_steps' in locals():
        print(f"推理步数 (halt_max_steps): {halt_max_steps}")
    print("=" * 60)
    print()
    print("执行命令:")
    print(" ".join(cmd))
    print("=" * 60)
    print()
    
    # 运行训练
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败，退出码: {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        sys.exit(130)


if __name__ == "__main__":
    main()

