#!/usr/bin/env python3
"""
查看训练指标的脚本 - 从wandb离线日志中提取信息
"""
import json
import os
import glob
import sys

def view_wandb_offline_data():
    """查看wandb离线数据"""
    wandb_dirs = glob.glob("wandb/offline-run-*")
    if not wandb_dirs:
        print("未找到wandb日志目录")
        return None
    
    latest_dir = max(wandb_dirs, key=os.path.getmtime)
    print(f"找到wandb运行: {os.path.basename(latest_dir)}\n")
    
    # 尝试使用wandb API读取离线数据
    try:
        import wandb
        api = wandb.Api()
        
        # 对于离线运行，我们需要手动读取
        print("注意: wandb离线日志需要使用wandb sync同步到云端后才能通过API查看")
        print("或者使用以下方法:\n")
        print("1. 同步到wandb云端:")
        print(f"   wandb sync {latest_dir}")
        print("\n2. 或者查看训练时的控制台输出")
        
    except ImportError:
        print("wandb未安装，无法直接读取.wandb文件")
    
    # 检查是否有日志文件
    log_files = glob.glob(os.path.join(latest_dir, "logs", "*.log"))
    if log_files:
        print("\n找到日志文件:")
        for log_file in log_files:
            print(f"  {log_file}")
            # 尝试从日志中提取关键信息
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    # 查找包含loss、accuracy等关键词的行
                    relevant_lines = [l for l in lines if any(keyword in l.lower() 
                                                             for keyword in ['loss', 'accuracy', 'step', 'train', 'eval'])]
                    if relevant_lines:
                        print(f"    相关日志行数: {len(relevant_lines)}")
                        print("    示例:")
                        for line in relevant_lines[-5:]:  # 显示最后5行
                            print(f"      {line.strip()}")
            except Exception as e:
                print(f"    无法读取: {e}")
    
    return latest_dir

def view_from_console_output():
    """提示用户如何从控制台输出查看"""
    print("\n" + "=" * 60)
    print("查看训练指标的其他方法")
    print("=" * 60)
    print("\n1. 如果训练还在运行，查看控制台输出")
    print("2. 如果训练已完成，检查是否有保存的训练日志文件")
    print("3. 使用wandb sync同步到云端:")
    print("   wandb sync wandb/offline-run-*")
    print("   然后访问 https://wandb.ai 查看")
    print("\n4. 查看checkpoint目录中的评估结果:")
    print("   checkpoints/Arc1concept-aug-1000-ACT-torch/my_training_run/evaluator_ARC_step_*/")

def create_simple_metrics_viewer():
    """创建一个简单的指标查看器，从checkpoint信息推断"""
    print("\n" + "=" * 60)
    print("从Checkpoint推断训练进度")
    print("=" * 60)
    
    checkpoint_dir = "checkpoints/Arc1concept-aug-1000-ACT-torch/my_training_run"
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint目录不存在: {checkpoint_dir}")
        return
    
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "step_*")))
    if checkpoints:
        print(f"\n找到 {len(checkpoints)} 个checkpoint:")
        for ckpt in checkpoints:
            step = os.path.basename(ckpt)
            step_num = int(step.replace("step_", ""))
            print(f"  {step}: Step {step_num}")
        
        # 查看评估结果
        eval_dirs = sorted(glob.glob(os.path.join(checkpoint_dir, "evaluator_ARC_step_*")))
        if eval_dirs:
            print(f"\n评估结果:")
            for eval_dir in eval_dirs:
                step = eval_dir.split("_")[-1]
                submission_file = os.path.join(eval_dir, "submission.json")
                if os.path.exists(submission_file):
                    with open(submission_file, 'r') as f:
                        data = json.load(f)
                    puzzles_with_preds = sum(1 for v in data.values() if v and len(v) > 0)
                    puzzles_with_nonempty = sum(
                        1 for v in data.values() 
                        if v and len(v) > 0 and any(
                            len(attempt.get('attempt_1', [])) > 0 
                            for attempt in v 
                            if isinstance(attempt, dict)
                        )
                    )
                    print(f"  Step {step}:")
                    print(f"    有预测: {puzzles_with_preds}/400 ({puzzles_with_preds/400*100:.1f}%)")
                    print(f"    非空预测: {puzzles_with_nonempty}/400 ({puzzles_with_nonempty/400*100:.1f}%)")

if __name__ == "__main__":
    print("=" * 60)
    print("训练指标查看器")
    print("=" * 60)
    
    # 方法1: 尝试从wandb读取
    wandb_dir = view_wandb_offline_data()
    
    # 方法2: 从checkpoint推断
    create_simple_metrics_viewer()
    
    # 方法3: 提示其他方法
    view_from_console_output()
    
    print("\n" + "=" * 60)
    print("推荐方案")
    print("=" * 60)
    print("\n由于wandb离线日志格式特殊，推荐:")
    print("1. 使用 wandb sync 同步到云端查看（需要wandb账号）")
    print("2. 查看评估结果文件了解模型性能")
    print("3. 如果训练还在运行，直接查看控制台输出")
    print("\n或者，我可以帮你修改代码，同时记录TensorBoard格式的日志")

