#!/usr/bin/env python3
"""
单独测试某道题的程序，显示完整的解题过程
"""
import os
import json
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import torch.distributed as dist
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from pretrain import (
    PretrainConfig,
    load_synced_config,
    create_model,
    load_checkpoint,
)
from dataset.build_arc_dataset import arc_grid_to_np, np_grid_to_seq_translational_augment
from dataset.common import PuzzleDatasetMetadata
from utils.functions import load_model_class

# ARC颜色映射
COLORS = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"
]
cmap = mcolors.ListedColormap(COLORS)
norm = mcolors.BoundaryNorm(np.arange(-0.5, 10.5, 1), cmap.N)


def visualize_grid(grid, title="Grid"):
    """可视化网格（文本格式）"""
    print(f"\n{title}:")
    print("-" * 40)
    for row in grid:
        print(" ".join(f"{cell:2d}" for cell in row))
    print("-" * 40)


def visualize_grid_image(grid: np.ndarray, title: str = "", save_path: str = None, ax=None):
    """可视化ARC网格（图像格式）"""
    if grid.size == 0:
        print(f"  {title}: (空网格)")
        return None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(grid.shape[1], 3), max(grid.shape[0], 3)))
        standalone = True
    else:
        standalone = False
    
    ax.imshow(grid, cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks([])
    ax.set_yticks([])
    
    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  图像已保存: {save_path}")
        else:
            plt.show()
        plt.close()
        return None
    else:
        return ax


def process_single_puzzle(
    checkpoint_path: str,
    puzzle_input: list,
    puzzle_id: str = "test_puzzle",
    config_path: str = "config",
    config_name: str = "cfg_pretrain",
    max_steps: int = 16,
    verbose: bool = True,
    save_images: bool = True,
    output_dir: str = "puzzle_solving_visualizations",
):
    """
    处理单个puzzle，显示完整的解题过程
    
    Args:
        checkpoint_path: checkpoint文件路径
        puzzle_input: 输入网格（2D列表，值0-9）
        puzzle_id: puzzle标识符
        config_path: 配置文件目录
        config_name: 配置文件名
        max_steps: 最大推理步数
        verbose: 是否显示详细信息
    """
    # 初始化分布式（单GPU）
    RANK = 0
    WORLD_SIZE = 1
    
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
    
    # 设置配置
    OmegaConf.set_struct(hydra_config, False)
    hydra_config.load_checkpoint = checkpoint_path
    hydra_config.data_paths = ["data/arc1concept-aug-1000"]  # 用于获取metadata
    OmegaConf.set_struct(hydra_config, True)
    
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)
    
    # 加载metadata（用于获取vocab_size等）
    metadata_path = os.path.join(config.data_paths[0], "train", "dataset.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        eval_metadata = PuzzleDatasetMetadata(**metadata_dict)
    else:
        raise FileNotFoundError(f"找不到metadata文件: {metadata_path}")
    
    # 创建模型
    model, optimizers, optimizer_lrs = create_model(config, eval_metadata, rank=RANK, world_size=WORLD_SIZE)
    model.eval()
    
    # 获取底层模型（去掉loss head包装）
    base_model = model.model if hasattr(model, 'model') else model
    
    # 处理输入
    input_grid = np.array(puzzle_input, dtype=np.uint8)
    if input_grid.ndim != 2:
        raise ValueError(f"输入必须是2D数组，得到的是{input_grid.ndim}D")
    
    # 转换为序列格式（与训练时一致）
    # 使用与训练时相同的数据处理方式
    ARCMaxGridSize = 30
    pad_r = pad_c = 0  # 不进行平移增强
    input_padded = np.pad(input_grid + 2, ((pad_r, ARCMaxGridSize - pad_r - input_grid.shape[0]), 
                                             (pad_c, ARCMaxGridSize - pad_c - input_grid.shape[1])), 
                          constant_values=0)
    
    # 添加EOS token
    eos_row, eos_col = pad_r + input_grid.shape[0], pad_c + input_grid.shape[1]
    if eos_row < ARCMaxGridSize:
        input_padded[eos_row, pad_c:eos_col] = 1
    if eos_col < ARCMaxGridSize:
        input_padded[pad_r:eos_row, eos_col] = 1
    
    input_seq = input_padded.flatten()
    
    # 确保input_seq长度正确（应该是seq_len + puzzle_emb_len）
    # 但实际只需要seq_len，puzzle_emb会在模型内部处理
    # 创建batch
    batch = {
        "inputs": torch.tensor(input_seq, dtype=torch.int32).unsqueeze(0).cuda(),  # [1, seq_len]
        "labels": torch.full((1, len(input_seq)), -100, dtype=torch.long).cuda(),  # 推理时不需要labels
        "puzzle_identifiers": torch.zeros(1, dtype=torch.int32).cuda(),  # 使用0作为puzzle_id
    }
    
    # 创建输出目录
    if save_images:
        os.makedirs(output_dir, exist_ok=True)
        print(f"图像将保存到: {output_dir}/")
    
    print("=" * 60)
    print("单题测试")
    print("=" * 60)
    print(f"Puzzle ID: {puzzle_id}")
    print(f"输入网格大小: {input_grid.shape}")
    visualize_grid(input_grid, "输入网格")
    
    # 保存输入图像
    if save_images:
        input_img_path = os.path.join(output_dir, f"{puzzle_id}_input.png")
        visualize_grid_image(input_grid, f"输入网格 - {puzzle_id}", save_path=input_img_path)
    
    # 初始化carry
    with torch.device("cuda"):
        carry = base_model.initial_carry(batch)
    
    print("\n" + "=" * 60)
    print("开始推理...")
    print("=" * 60)
    
    # 推理循环
    all_predictions = []
    all_q_halt_logits = []
    all_q_continue_logits = []
    
    # 获取puzzle_emb_len
    puzzle_emb_len = base_model.inner.puzzle_emb_len if hasattr(base_model.inner, 'puzzle_emb_len') else 16
    
    with torch.no_grad():
        for step in range(max_steps):
            # 前向传播（模型返回carry和outputs）
            carry, outputs = base_model(carry=carry, batch=batch)
            
            # 获取预测
            logits = outputs["logits"]  # [batch_size, seq_len, vocab_size]
            preds = logits.argmax(dim=-1)  # [batch_size, seq_len]
            q_halt_logits = outputs["q_halt_logits"]  # [batch_size]
            q_continue_logits = outputs.get("q_continue_logits", torch.zeros_like(q_halt_logits))
            
            # 提取预测的网格部分
            # 注意：从代码看，lm_head的输出已经是[:, puzzle_emb_len:]，所以logits的长度就是seq_len
            # 不需要再减去puzzle_emb_len
            pred_seq = preds[0].cpu().numpy()  # 直接使用全部预测
            
            # 调试信息（第一步）
            if step == 0:
                print(f"  Debug: logits shape={logits.shape}, preds shape={preds.shape}")
                print(f"  Debug: puzzle_emb_len={puzzle_emb_len}, eval_metadata.seq_len={eval_metadata.seq_len}")
                print(f"  Debug: pred_seq length={len(pred_seq)}")
            
            # 转换为网格（使用实际的seq_len）
            if len(pred_seq) == eval_metadata.seq_len:
                pred_grid = pred_seq.reshape(ARCMaxGridSize, ARCMaxGridSize)
            elif len(pred_seq) < eval_metadata.seq_len:
                # 如果长度不足，填充到正确长度
                pred_seq = np.pad(pred_seq, (0, eval_metadata.seq_len - len(pred_seq)), constant_values=0)
                pred_grid = pred_seq.reshape(ARCMaxGridSize, ARCMaxGridSize)
            else:
                # 如果长度超出，裁剪到正确长度
                pred_seq = pred_seq[:eval_metadata.seq_len]
                pred_grid = pred_seq.reshape(ARCMaxGridSize, ARCMaxGridSize)
            
            # 裁剪到有效区域（找到EOS token）
            valid_rows = ARCMaxGridSize
            valid_cols = ARCMaxGridSize
            for i in range(ARCMaxGridSize):
                if i < ARCMaxGridSize and pred_grid[i, 0] == 1:  # EOS token
                    valid_rows = i
                    break
            for j in range(ARCMaxGridSize):
                if j < ARCMaxGridSize and pred_grid[0, j] == 1:  # EOS token
                    valid_cols = j
                    break
            
            # 如果没找到EOS，使用整个网格
            if valid_rows == ARCMaxGridSize:
                valid_rows = input_grid.shape[0]
            if valid_cols == ARCMaxGridSize:
                valid_cols = input_grid.shape[1]
            
            pred_grid_cropped = pred_grid[:valid_rows, :valid_cols] - 2  # 减去偏移
            pred_grid_cropped = np.clip(pred_grid_cropped, 0, 9)  # 确保值在0-9范围内
            
            all_predictions.append(pred_grid_cropped.copy())
            all_q_halt_logits.append(q_halt_logits[0].item())
            all_q_continue_logits.append(q_continue_logits[0].item())
            
            # 显示当前步骤
            print(f"\n步骤 {step + 1}:")
            print(f"  Q_halt logit: {q_halt_logits[0].item():.4f}")
            print(f"  Q_continue logit: {q_continue_logits[0].item():.4f}")
            print(f"  Halted: {carry.halted[0].item()}")
            print(f"  Steps: {carry.steps[0].item()}")
            
            if verbose:
                visualize_grid(pred_grid_cropped, f"步骤 {step + 1} 预测")
            
            # 保存每一步的预测图像
            if save_images:
                step_img_path = os.path.join(output_dir, f"{puzzle_id}_step_{step+1:02d}.png")
                visualize_grid_image(
                    pred_grid_cropped, 
                    f"步骤 {step + 1} - Q_halt={q_halt_logits[0].item():.4f}, Halted={carry.halted[0].item()}", 
                    save_path=step_img_path
                )
            
            # 检查是否所有序列都halt了
            if carry.halted.all():
                print(f"\n所有序列在第 {step + 1} 步halt")
                break
    
    # 显示最终结果
    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)
    
    if all_predictions:
        final_pred = all_predictions[-1]
        print(f"\n最终预测网格大小: {final_pred.shape}")
        visualize_grid(final_pred, "最终预测")
        
        # 保存最终预测图像
        if save_images:
            final_img_path = os.path.join(output_dir, f"{puzzle_id}_final.png")
            visualize_grid_image(final_pred, f"最终预测 - {puzzle_id}", save_path=final_img_path)
        
        print(f"\n推理统计:")
        print(f"  总步数: {len(all_predictions)}")
        print(f"  最终Q_halt logit: {all_q_halt_logits[-1]:.4f}")
        print(f"  最终Q_continue logit: {all_q_continue_logits[-1]:.4f}")
        
        # 显示所有步骤的Q值变化
        print(f"\nQ值变化:")
        for i, (q_halt, q_cont) in enumerate(zip(all_q_halt_logits, all_q_continue_logits)):
            print(f"  步骤 {i+1}: halt={q_halt:7.4f}, continue={q_cont:7.4f}")
        
        # 创建对比图：显示输入和所有步骤的预测
        if save_images and len(all_predictions) > 0:
            create_comparison_figure(
                input_grid, 
                all_predictions, 
                all_q_halt_logits,
                puzzle_id,
                output_dir
            )
    
    # 清理
    if dist.is_initialized():
        dist.destroy_process_group()
    
    return all_predictions, all_q_halt_logits, all_q_continue_logits


def create_comparison_figure(input_grid: np.ndarray, predictions: list, q_halt_logits: list, 
                             puzzle_id: str, output_dir: str):
    """创建对比图，显示输入和所有步骤的预测"""
    num_steps = len(predictions)
    if num_steps == 0:
        return
    
    # 计算网格大小（取输入和所有预测的最大尺寸）
    max_rows = max(input_grid.shape[0], max(p.shape[0] for p in predictions))
    max_cols = max(input_grid.shape[1], max(p.shape[1] for p in predictions))
    
    # 创建子图：输入 + 每一步预测
    num_cols = min(4, num_steps + 1)  # 每行最多4个图
    num_rows = (num_steps + 1 + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # 显示输入
    ax = axes[0, 0] if num_rows > 1 else axes[0]
    visualize_grid_image(input_grid, "输入", ax=ax)
    
    # 显示每一步的预测
    for step_idx in range(num_steps):
        row = (step_idx + 1) // num_cols
        col = (step_idx + 1) % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]
        
        title = f"步骤 {step_idx + 1}\nQ_halt={q_halt_logits[step_idx]:.3f}"
        visualize_grid_image(predictions[step_idx], title, ax=ax)
    
    # 隐藏多余的子图
    total_plots = num_steps + 1
    for idx in range(total_plots, num_rows * num_cols):
        row = idx // num_cols
        col = idx % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]
        ax.axis('off')
    
    plt.suptitle(f"Puzzle {puzzle_id} - 推理过程对比", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    comparison_path = os.path.join(output_dir, f"{puzzle_id}_comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"\n对比图已保存: {comparison_path}")
    plt.close()


def load_puzzle_from_file(file_path: str):
    """从JSON文件加载puzzle"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "input" in data:
        return data["input"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"无法解析puzzle文件: {file_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="单独测试某道题")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint文件路径"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="输入网格（JSON文件路径或直接输入，例如：'[[0,1,2],[3,4,5]]'）"
    )
    parser.add_argument(
        "--puzzle_id",
        type=str,
        default="test_puzzle",
        help="Puzzle标识符"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=16,
        help="最大推理步数"
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
    parser.add_argument(
        "--save_images",
        action="store_true",
        default=True,
        help="保存图像到文件（默认：True）"
    )
    parser.add_argument(
        "--no_save_images",
        action="store_false",
        dest="save_images",
        help="不保存图像"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="puzzle_solving_visualizations",
        help="图像保存目录（默认：puzzle_solving_visualizations）"
    )
    
    args = parser.parse_args()
    
    # 加载输入
    if args.input:
        if os.path.exists(args.input):
            # 从文件加载
            puzzle_input = load_puzzle_from_file(args.input)
        else:
            # 尝试解析为JSON
            try:
                puzzle_input = json.loads(args.input)
            except:
                raise ValueError(f"无法解析输入: {args.input}")
    else:
        # 使用示例输入
        print("未提供输入，使用示例输入...")
        puzzle_input = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
        ]
    
    # 运行测试
    process_single_puzzle(
        checkpoint_path=args.checkpoint,
        puzzle_input=puzzle_input,
        puzzle_id=args.puzzle_id,
        config_path=args.config_path,
        config_name=args.config_name,
        max_steps=args.max_steps,
        verbose=True,
        save_images=args.save_images,
        output_dir=args.output_dir,
    )

