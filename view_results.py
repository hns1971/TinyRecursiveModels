#!/usr/bin/env python3
"""
查看训练成果的脚本
"""
import json
import os
import glob
import torch

def view_checkpoints(checkpoint_dir):
    """查看checkpoint信息"""
    print("=" * 60)
    print("Checkpoint 信息")
    print("=" * 60)
    
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "step_*")))
    if not checkpoint_files:
        print("未找到checkpoint文件")
        return
    
    for ckpt_file in checkpoint_files:
        step = os.path.basename(ckpt_file)
        file_size = os.path.getsize(ckpt_file) / (1024**3)  # GB
        print(f"\n{step}:")
        print(f"  文件大小: {file_size:.2f} GB")
        
        # 尝试加载并查看参数数量
        try:
            state_dict = torch.load(ckpt_file, map_location="cpu")
            num_params = sum(p.numel() for p in state_dict.values())
            print(f"  参数量: {num_params:,}")
            print(f"  参数键数量: {len(state_dict)}")
        except Exception as e:
            print(f"  无法加载: {e}")

def view_evaluation_results(checkpoint_dir):
    """查看评估结果"""
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    
    eval_dirs = sorted(glob.glob(os.path.join(checkpoint_dir, "evaluator_ARC_step_*")))
    if not eval_dirs:
        print("未找到评估结果")
        return
    
    for eval_dir in eval_dirs:
        step = eval_dir.split("_")[-1]
        submission_file = os.path.join(eval_dir, "submission.json")
        
        if os.path.exists(submission_file):
            with open(submission_file, 'r') as f:
                data = json.load(f)
            
            total_puzzles = len(data)
            puzzles_with_preds = sum(1 for v in data.values() if v and len(v) > 0)
            puzzles_with_nonempty = sum(
                1 for v in data.values() 
                if v and len(v) > 0 and any(
                    len(attempt.get('attempt_1', [])) > 0 
                    for attempt in v 
                    if isinstance(attempt, dict)
                )
            )
            
            print(f"\nStep {step}:")
            print(f"  总puzzles: {total_puzzles}")
            print(f"  有预测的puzzles: {puzzles_with_preds} ({puzzles_with_preds/total_puzzles*100:.1f}%)")
            print(f"  有非空预测的puzzles: {puzzles_with_nonempty} ({puzzles_with_nonempty/total_puzzles*100:.1f}%)")

def view_wandb_summary():
    """查看wandb摘要"""
    print("\n" + "=" * 60)
    print("WandB 训练指标")
    print("=" * 60)
    
    wandb_dirs = glob.glob("wandb/offline-run-*")
    if not wandb_dirs:
        print("未找到wandb日志")
        return
    
    latest_dir = max(wandb_dirs, key=os.path.getmtime)
    print(f"\n最新运行: {os.path.basename(latest_dir)}")
    
    summary_file = os.path.join(latest_dir, "files", "wandb-summary.json")
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        print("\n关键指标:")
        for key in sorted(summary.keys()):
            value = summary[key]
            if isinstance(value, (int, float)):
                if 'loss' in key.lower():
                    print(f"  {key}: {value:.6f}")
                elif 'accuracy' in key.lower() or 'acc' in key.lower():
                    print(f"  {key}: {value:.4f}")
                elif 'lr' in key.lower():
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value}")
    else:
        print("未找到summary文件")
        print("\n提示: 可以使用以下命令查看wandb日志:")
        print("  wandb sync wandb/offline-run-*")
        print("  或访问 https://wandb.ai 查看")

if __name__ == "__main__":
    checkpoint_dir = "checkpoints/Arc1concept-aug-1000-ACT-torch/my_training_run"
    
    if os.path.exists(checkpoint_dir):
        view_checkpoints(checkpoint_dir)
        view_evaluation_results(checkpoint_dir)
    else:
        print(f"Checkpoint目录不存在: {checkpoint_dir}")
    
    view_wandb_summary()
    
    print("\n" + "=" * 60)
    print("如何使用checkpoint继续训练或推理")
    print("=" * 60)
    print("\n1. 继续训练:")
    print("   python pretrain.py \\")
    print("     arch=trm \\")
    print("     data_paths=\"[data/arc1concept-aug-1000]\" \\")
    print("     load_checkpoint=checkpoints/Arc1concept-aug-1000-ACT-torch/my_training_run/step_606")
    print("\n2. 查看完整评估结果:")
    print("   评估结果保存在: checkpoints/Arc1concept-aug-1000-ACT-torch/my_training_run/evaluator_ARC_step_*/submission.json")
    print("\n3. 查看训练曲线:")
    print("   tensorboard --logdir wandb/")

