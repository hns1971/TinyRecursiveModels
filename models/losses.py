from typing import Any, Tuple, Dict, Sequence, Optional
import json
import os

import torch
import torch.nn.functional as F
from torch import nn
import math
import numpy as np


IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str, lm_loss_weight: float = 1.0, q_halt_loss_weight: float = 0.5):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.lm_loss_weight = lm_loss_weight
        self.q_halt_loss_weight = q_halt_loss_weight
        self._debug_eval = False  # 调试标志
        self._debug_printed = False  # 是否已打印过
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        # 底层模型只需要carry和batch，不需要return_keys
        model_kwargs_for_base = {k: v for k, v in model_kwargs.items() if k != "return_keys"}
        
        # 打印传递给底层模型的内容（用于与单测对比）
        # 检查调试标志（支持通过属性或环境变量启用）
        debug_enabled = (
            (hasattr(self, '_debug_eval') and self._debug_eval) or 
            os.environ.get('DEBUG_EVAL', '').lower() == 'true'
        )
        if debug_enabled and not getattr(self, '_debug_printed', False):
            carry_for_base = model_kwargs_for_base.get("carry")
            batch_for_base = model_kwargs_for_base.get("batch", {})
            print("\n" + "="*80)
            print("[评估-ACTLossHead] 调用底层模型前，传递给底层模型的内容:")
            print("="*80)
            if carry_for_base is not None:
                print(f"carry.halted: {carry_for_base.halted if hasattr(carry_for_base, 'halted') else 'N/A'}")
                print(f"carry.steps: {carry_for_base.steps if hasattr(carry_for_base, 'steps') else 'N/A'}")
                if hasattr(carry_for_base, 'current_data') and carry_for_base.current_data:
                    print("carry.current_data:")
                    for k, v in carry_for_base.current_data.items():
                        if isinstance(v, torch.Tensor):
                            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
                            if v.numel() <= 20:  # 只打印小tensor的值
                                print(f"    values: {v.cpu().numpy()}")
                            elif k in ["inputs", "labels"]:
                                # 对于inputs和labels，打印第一个样本的第四行（结果行）
                                if v.shape[0] > 0:
                                    first_sample = v[0]  # [seq_len]
                                    seq_len = first_sample.shape[0]
                                    row_len = seq_len // 4  # 每行的长度
                                    if row_len > 0:
                                        row4 = first_sample[row_len * 3:row_len * 4].cpu().numpy()
                                        print(f"    row4 (第一个样本的第四行，结果行): {row4}")
                                    else:
                                        print(f"    values: {first_sample.cpu().numpy()}")
                        else:
                            print(f"  {k}: {type(v)}")
            print("batch:")
            for k, v in batch_for_base.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
                    if v.numel() <= 20:  # 只打印小tensor的值
                        print(f"    values: {v.cpu().numpy()}")
                else:
                    print(f"  {k}: {type(v)}")
            print("="*80 + "\n")
            # 标记已打印，避免重复打印
            self._debug_printed = True
        
        new_carry, outputs = self.model(**model_kwargs_for_base)
        
        # 打印预测结果（用于检查预测是否正确）
        if getattr(self, '_debug_printed', False):
            print("\n" + "="*80)
            print("[评估-ACTLossHead] 模型预测结果:")
            print("="*80)
            if "logits" in outputs:
                logits = outputs["logits"]
                preds = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]
                print(f"logits shape: {logits.shape}")
                print(f"predictions shape: {preds.shape}")
                # 打印第一个样本的第四行（结果行）
                if preds.shape[0] > 0:
                    first_pred = preds[0]  # [seq_len]
                    seq_len = first_pred.shape[0]
                    row_len = seq_len // 4  # 每行的长度
                    if row_len > 0:
                        row4 = first_pred[row_len * 3:row_len * 4].cpu().numpy()
                        print(f"predictions[0] row4 (第一个样本的第四行，结果行): {row4}")
                    else:
                        print(f"predictions[0]: {first_pred.cpu().numpy()}")
                # 打印用于比较的label的第四行（第一个数据）
                # 使用原始传入的batch中的labels，而不是new_carry.current_data中的labels
                # 因为new_carry.current_data["labels"]可能已经被模型更新为预测结果
                original_labels = model_kwargs.get("batch", {}).get("labels")
                if original_labels is None:
                    # 如果原始batch中没有labels，尝试从batch_for_base获取
                    original_labels = model_kwargs_for_base.get("batch", {}).get("labels")
                if original_labels is not None and original_labels.shape[0] > 0:
                    first_label = original_labels[0]  # [seq_len]
                    seq_len = first_label.shape[0]
                    row_len = seq_len // 4  # 每行的长度
                    if row_len > 0:
                        label_row4 = first_label[row_len * 3:row_len * 4].cpu().numpy()
                        print(f"original_batch.labels[0] row4 (第一个样本的第四行，用于比较): {label_row4}")
                        # 比较预测和label是否一致
                        if preds.shape[0] > 0:
                            pred_row4 = preds[0][row_len * 3:row_len * 4].cpu().numpy()
                            match = (pred_row4 == label_row4).all()
                            print(f"预测与label是否一致: {'✓ 一致' if match else '✗ 不一致'}")
            if "q_halt_logits" in outputs:
                q_halt_logits = outputs["q_halt_logits"]
                print(f"q_halt_logits: {q_halt_logits.cpu().numpy()}")
                print(f"q_halt_confidence (sigmoid): {torch.sigmoid(q_halt_logits).cpu().numpy()}")
            if hasattr(new_carry, 'current_data') and new_carry.current_data:
                print("new_carry.current_data (模型更新后的):")
                for k, v in new_carry.current_data.items():
                    if isinstance(v, torch.Tensor) and k in ["inputs", "labels"]:
                        # 打印第一个样本的第四行（结果行）
                        if v.shape[0] > 0:
                            first_sample = v[0]  # [seq_len]
                            seq_len = first_sample.shape[0]
                            row_len = seq_len // 4  # 每行的长度
                            if row_len > 0:
                                row4 = first_sample[row_len * 3:row_len * 4].cpu().numpy()
                                print(f"  {k}[0] row4 (第一个样本的第四行，结果行): {row4}")
                            else:
                                print(f"  {k}[0]: {first_sample.cpu().numpy()}")
            print("="*80 + "\n")
        
        # 关键修复：使用原始传入的batch中的labels，而不是new_carry.current_data["labels"]
        # 因为new_carry.current_data["labels"]可能已经被模型更新为预测结果
        # 对于exact_accuracy计算，应该使用原始batch中的labels（数据集中真正的标签）
        original_labels = model_kwargs.get("batch", {}).get("labels")
        if original_labels is None:
            # 如果原始batch中没有labels，尝试从batch_for_base获取
            original_labels = model_kwargs_for_base.get("batch", {}).get("labels")
        if original_labels is not None:
            labels = original_labels  # 使用原始batch中的labels
        else:
            # 如果原始batch中没有labels，fallback到new_carry.current_data["labels"]
            labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics：不再仅在halted时计数，所有有效位置都计入
            valid_metrics = (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses
        # 只在第一步计算lm_loss（steps == 1），后续步骤不计算lm_loss
        # 因为训练数据是(s_t, s_{t+1})，只有第一步有正确的标签
        # 后续步骤如果继续推理，没有对应的标签，所以不计算lm_loss
        is_first_step = (new_carry.steps == 1)  # steps == 1 表示刚完成第一步
        if is_first_step.any():
            # 计算所有样本的损失，但只对第一步的样本求和
            step_losses = self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor  # [B, seq_len]
            step_losses_per_sample = step_losses.sum(-1)  # [B]
            # 只对第一步的样本求和
            lm_loss = (step_losses_per_sample * is_first_step.to(step_losses_per_sample.dtype)).sum()
        else:
            # 如果没有第一步，lm_loss为0
            lm_loss = torch.tensor(0.0, device=outputs["logits"].device, dtype=outputs["logits"].dtype)
        
        # q_halt_loss: 每一步都计算，让模型在所有步骤学习 halt 判断
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })
        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()
        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        # Total loss with configurable weights
        total_loss = self.lm_loss_weight * lm_loss + self.q_halt_loss_weight * (q_halt_loss + q_continue_loss)
        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()


class AdditionACTLossHead(ACTLossHead):
    """
    用于加法微调的损失函数，添加了copy loss来约束前两行（加数）保持不变。
    
    加法问题的网格格式：
    - 4行 × N列
    - 第1行：第一个加数
    - 第2行：第二个加数
    - 第3行：进位
    - 第4行：结果
    
    在递归推理中，前两行（加数）应该保持不变。
    
    注意：在微调中，PAD值不进行掩码处理，所有位置都参与损失计算。
    """
    def __init__(self, model: nn.Module, loss_type: str, copy_loss_weight: float = 1.0, copy_loss_coeff: float = 0.1, lm_loss_weight: float = 1.0, q_halt_loss_weight: float = 0.5):
        super().__init__(model, loss_type, lm_loss_weight=lm_loss_weight, q_halt_loss_weight=q_halt_loss_weight)
        self.copy_loss_weight = copy_loss_weight
        self.copy_loss_coeff = copy_loss_coeff  # copy_loss的系数，因为copy_loss很容易学习，所以使用较低的值
        # 加法数据集的pad_id是11（值+1后的PAD值）
        self.pad_id = 11
        # 动态系数相关参数
        self.current_step = 0  # 当前训练步数（需要从外部更新）
        self.warmup_steps = 1000  # warmup阶段步数（可以从config获取）
        
    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # 调用父类的forward，但需要添加copy loss
        # 注意：batch包含原始输入，carry.current_data可能包含上一步的输出（递归推理）
        batch = model_kwargs.get("batch", {})
        # 从batch中提取训练步数和warmup步数（如果存在）
        training_step = batch.get("_training_step", None)
        warmup_steps = batch.get("_warmup_steps", None)
        # 创建不包含这些额外字段的batch给底层模型
        batch_for_model = {k: v for k, v in batch.items() if not k.startswith("_")}
        # 修改：labels应该使用inputs的值（因为这是一步推理，inputs包含输入及初始状态）
        # 数据里的labels是答案（目标状态），不是当前状态，所以传递给模型时应该用inputs的值
        if "inputs" in batch_for_model:
            batch_for_model["labels"] = batch_for_model["inputs"].clone()
        model_kwargs_for_base = {k: v for k, v in model_kwargs.items() if k != "return_keys"}
        model_kwargs_for_base["batch"] = batch_for_model
        
        # 打印传递给底层模型的内容（用于与单测对比）
        # 检查调试标志（支持通过属性或环境变量启用）
        debug_enabled = (
            (hasattr(self, '_debug_eval') and self._debug_eval) or 
            os.environ.get('DEBUG_EVAL', '').lower() == 'true'
        )
        if debug_enabled and not getattr(self, '_debug_printed', False):
            carry_for_base = model_kwargs_for_base.get("carry")
            print("\n" + "="*80)
            print("[评估-AdditionACTLossHead] 调用底层模型前，传递给底层模型的内容:")
            print("="*80)
            print(f"原始batch中的keys: {list(batch.keys())}")
            print(f"处理后的batch_for_model中的keys: {list(batch_for_model.keys())}")
            if carry_for_base is not None:
                print(f"carry.halted: {carry_for_base.halted if hasattr(carry_for_base, 'halted') else 'N/A'}")
                print(f"carry.steps: {carry_for_base.steps if hasattr(carry_for_base, 'steps') else 'N/A'}")
                if hasattr(carry_for_base, 'current_data') and carry_for_base.current_data:
                    print("carry.current_data:")
                    for k, v in carry_for_base.current_data.items():
                        if isinstance(v, torch.Tensor):
                            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
                            if v.numel() <= 20:  # 只打印小tensor的值
                                print(f"    values: {v.cpu().numpy()}")
                            elif k in ["inputs", "labels"]:
                                # 对于inputs和labels，打印第一个样本的第四行（结果行）
                                if v.shape[0] > 0:
                                    first_sample = v[0]  # [seq_len]
                                    seq_len = first_sample.shape[0]
                                    row_len = seq_len // 4  # 每行的长度
                                    if row_len > 0:
                                        row4 = first_sample[row_len * 3:row_len * 4].cpu().numpy()
                                        print(f"    row4 (第一个样本的第四行，结果行): {row4}")
                                    else:
                                        print(f"    values: {first_sample.cpu().numpy()}")
                        else:
                            print(f"  {k}: {type(v)}")
            print("batch_for_model (传递给底层模型):")
            for k, v in batch_for_model.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
                    if v.numel() <= 20:  # 只打印小tensor的值
                        print(f"    values: {v.cpu().numpy()}")
                    elif k in ["inputs", "labels"]:
                        # 对于inputs和labels，打印第一个样本的第四行（结果行）
                        if v.shape[0] > 0:
                            first_sample = v[0]  # [seq_len]
                            seq_len = first_sample.shape[0]
                            row_len = seq_len // 4  # 每行的长度
                            if row_len > 0:
                                row4 = first_sample[row_len * 3:row_len * 4].cpu().numpy()
                                print(f"    row4 (第一个样本的第四行，结果行): {row4}")
                            else:
                                print(f"    values: {first_sample.cpu().numpy()}")
                else:
                    print(f"  {k}: {type(v)}")
            print("="*80 + "\n")
            # 标记已打印，避免重复打印
            self._debug_printed = True
        
        new_carry, outputs = self.model(**model_kwargs_for_base)
        
        # 打印预测结果（用于检查预测是否正确）
        if getattr(self, '_debug_printed', False):
            print("\n" + "="*80)
            print("[评估-AdditionACTLossHead] 模型预测结果:")
            print("="*80)
            if "logits" in outputs:
                logits = outputs["logits"]
                preds = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]
                print(f"logits shape: {logits.shape}")
                print(f"predictions shape: {preds.shape}")
                # 打印第一个样本的第四行（结果行）
                if preds.shape[0] > 0:
                    first_pred = preds[0]  # [seq_len]
                    seq_len = first_pred.shape[0]
                    row_len = seq_len // 4  # 每行的长度
                    if row_len > 0:
                        row4 = first_pred[row_len * 3:row_len * 4].cpu().numpy()
                        print(f"predictions[0] row4 (第一个样本的第四行，结果行): {row4}")
                    else:
                        print(f"predictions[0]: {first_pred.cpu().numpy()}")
                # 打印用于比较的label的第四行（第一个数据）
                # 使用原始传入的batch中的labels，而不是new_carry.current_data中的labels
                # 注意：在AdditionACTLossHead中，batch_for_model["labels"]被设置为inputs的clone
                # 所以需要使用原始batch中的labels（即数据集中真正的标签）
                original_labels = batch.get("labels")
                if original_labels is not None and original_labels.shape[0] > 0:
                    first_label = original_labels[0]  # [seq_len]
                    seq_len = first_label.shape[0]
                    row_len = seq_len // 4  # 每行的长度
                    if row_len > 0:
                        label_row4 = first_label[row_len * 3:row_len * 4].cpu().numpy()
                        print(f"original_batch.labels[0] row4 (第一个样本的第四行，用于比较): {label_row4}")
                        # 比较预测和label是否一致
                        if preds.shape[0] > 0:
                            pred_row4 = preds[0][row_len * 3:row_len * 4].cpu().numpy()
                            match = (pred_row4 == label_row4).all()
                            print(f"预测与label是否一致: {'✓ 一致' if match else '✗ 不一致'}")
            if "q_halt_logits" in outputs:
                q_halt_logits = outputs["q_halt_logits"]
                print(f"q_halt_logits: {q_halt_logits.cpu().numpy()}")
                print(f"q_halt_confidence (sigmoid): {torch.sigmoid(q_halt_logits).cpu().numpy()}")
            if hasattr(new_carry, 'current_data') and new_carry.current_data:
                print("new_carry.current_data (模型更新后的):")
                for k, v in new_carry.current_data.items():
                    if isinstance(v, torch.Tensor) and k in ["inputs", "labels"]:
                        # 打印第一个样本的第四行（结果行）
                        if v.shape[0] > 0:
                            first_sample = v[0]  # [seq_len]
                            seq_len = first_sample.shape[0]
                            row_len = seq_len // 4  # 每行的长度
                            if row_len > 0:
                                row4 = first_sample[row_len * 3:row_len * 4].cpu().numpy()
                                print(f"  {k}[0] row4 (第一个样本的第四行，结果行): {row4}")
                            else:
                                print(f"  {k}[0]: {first_sample.cpu().numpy()}")
            print("="*80 + "\n")
        
        # 关键修复：使用原始batch中的labels，而不是new_carry.current_data["labels"]
        # 因为new_carry.current_data["labels"]可能已经被模型更新为预测结果
        # 对于exact_accuracy计算，应该使用原始batch中的labels（数据集中真正的标签）
        if "labels" in batch and batch["labels"] is not None:
            labels = batch["labels"]  # 使用原始batch中的labels
        else:
            # 如果原始batch中没有labels，fallback到new_carry.current_data["labels"]
            labels = new_carry.current_data["labels"]
        
        # 使用batch中的原始输入，而不是carry.current_data（可能是上一步的输出）
        # 添加安全检查，确保inputs是有效的tensor
        # 优先级：batch["inputs"] > new_carry.current_data["inputs"] > 从carry获取
        if "inputs" in batch and batch["inputs"] is not None:
            inputs = batch["inputs"]
        elif "inputs" in new_carry.current_data and new_carry.current_data["inputs"] is not None:
            # 检查new_carry.current_data["inputs"]是否有效（不是empty或全0）
            candidate_inputs = new_carry.current_data["inputs"]
            if candidate_inputs.numel() > 0 and not (candidate_inputs == 0).all():
                inputs = candidate_inputs
            else:
                # 如果new_carry.current_data["inputs"]无效，尝试从原始carry获取
                original_carry = model_kwargs.get("carry", None)
                if original_carry is not None and "inputs" in original_carry.current_data:
                    inputs = original_carry.current_data["inputs"]
                else:
                    # 最后fallback：使用batch中的inputs（如果batch中没有，说明有问题）
                    raise RuntimeError(f"Cannot find valid inputs: batch has inputs={('inputs' in batch)}, new_carry.current_data has inputs={('inputs' in new_carry.current_data)}")
        else:
            # 如果new_carry.current_data中没有inputs，尝试从原始carry获取
            original_carry = model_kwargs.get("carry", None)
            if original_carry is not None and "inputs" in original_carry.current_data:
                inputs = original_carry.current_data["inputs"]
            else:
                # 最后fallback：不应该使用labels，因为labels可能是-100
                raise RuntimeError(f"Cannot find valid inputs: batch has inputs={('inputs' in batch)}, new_carry.current_data has inputs={('inputs' in new_carry.current_data)}")
        
        # 获取current_data的labels（上一步的计算结果，只有后两行）
        original_carry = model_kwargs.get("carry", None)
        if original_carry is not None and "labels" in original_carry.current_data:
            current_data_labels = original_carry.current_data["labels"]
        else:
            # 如果没有，使用labels作为fallback（但这不是正确的，因为labels是目标）
            current_data_labels = labels
        
        # 确保inputs是2D tensor (batch_size, seq_len)
        if inputs.dim() != 2:
            raise RuntimeError(f"Invalid inputs shape: {inputs.shape}, expected 2D tensor (batch_size, seq_len)")

        # 在微调中，PAD值不进行掩码处理
        # 将labels中被转换为-100的PAD值恢复为原始的pad_id（11）
        # 注意：这里使用的是原始batch中的labels，确保exact_accuracy计算正确
        labels_for_loss = labels.clone()
        labels_for_loss[labels == IGNORE_LABEL_ID] = self.pad_id
        
        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # 比较预测的结果行与标签的结果行（排除PAD位）
            # 从预测的第4行（结果行）提取数字，从标签的第4行（结果行）提取数字，然后比较
            # 注意：这部分计算需要在torch.no_grad()中，并且使用detach()避免梯度追踪
            # 为了兼容torch.compile，我们使用torch.jit.ignore装饰器或者将这部分移到外部
            # 但更简单的方法是：在no_grad中，将tensor移到CPU并转换为Python对象进行计算
            seq_len = int(inputs.shape[-1])
            batch_size = int(inputs.shape[0])
            # 恢复为4行格式：inputs、labels、模型输出都是完整的4行
            # 将inputs、preds和labels转换为原始值（值+1格式，需要减1）
            # labels_for_loss: B × seq_len（完整的4行），值范围是 [1, 11]（1-10对应数字0-9，11是PAD）
            # current_data_labels: B × seq_len（完整的4行，上一步的计算结果），值范围是 [1, 11]
            input_values = (inputs - 1).detach()  # B × seq_len（完整的4行），值范围是 [0, 10]
            pred_values = (outputs["preds"] - 1).detach()  # B × seq_len（完整的4行），值范围是 [0, 10]
            label_values = (labels_for_loss - 1).detach()  # B × seq_len（完整的4行），值范围是 [0, 10]
            current_data_values = (current_data_labels - 1).detach()  # B × seq_len（完整的4行），值范围是 [0, 10]
            
            # 计算4行的序列长度
            four_rows_len = inputs.shape[1]  # inputs是完整的4行
            grid_width = four_rows_len // 4  # 每行的列数
            
            # 重塑为网格：都是完整的4行
            input_grid = input_values[:, :four_rows_len].view(batch_size, 4, grid_width)  # B × 4 × grid_width（完整的4行）
            pred_grid = pred_values[:, :four_rows_len].view(batch_size, 4, grid_width)  # B × 4 × grid_width（完整的4行）
            label_grid = label_values[:, :four_rows_len].view(batch_size, 4, grid_width)  # B × 4 × grid_width（完整的4行）
            current_data_grid = current_data_values[:, :four_rows_len].view(batch_size, 4, grid_width)  # B × 4 × grid_width（完整的4行）
            
            # 提取：都是完整的4行
            row1 = input_grid[:, 0, :]  # B × grid_width（第1行：加数1）
            row2 = input_grid[:, 1, :]  # B × grid_width（第2行：加数2）
            pred_row3 = pred_grid[:, 2, :]  # B × grid_width（第3行：进位）
            pred_row4 = pred_grid[:, 3, :]  # B × grid_width（第4行：结果）
            label_row3 = label_grid[:, 2, :]  # B × grid_width（第3行：进位）
            label_row4 = label_grid[:, 3, :]  # B × grid_width（第4行：结果）
            current_data_row3 = current_data_grid[:, 2, :]  # B × grid_width（上一步的第3行：进位）
            current_data_row4 = current_data_grid[:, 3, :]  # B × grid_width（上一步的第4行：结果）
            
            # 创建mask：有效数字（0-9，不包括PAD=11和LEADING_VALUE=10）
            # 注意：LEADING_VALUE=10是前导F，不是PAD，但在提取数字时需要排除
            # PAD_VALUE=11是真正的PAD，应该排除
            valid_mask_row1 = (row1 >= 0) & (row1 < 10)  # B × grid_width（0-9是有效数字）
            valid_mask_row2 = (row2 >= 0) & (row2 < 10)  # B × grid_width（0-9是有效数字）
            valid_mask_pred = (pred_row4 >= 0) & (pred_row4 < 10)  # B × grid_width（0-9是有效数字）
            valid_mask_label = (label_row4 >= 0) & (label_row4 < 10)  # B × grid_width（0-9是有效数字）
            
            # 将tensor移到CPU并转换为numpy进行计算（在no_grad中，这是安全的）
            # 注意：这部分代码在torch.no_grad()中，并且使用detach()，不会影响梯度
            # 为了兼容torch.compile，我们需要确保numpy转换不会被编译
            # 使用torch._dynamo.mark_dynamic标记这些tensor，或者直接转换（在no_grad中应该安全）
            # 最简单的方法：直接转换，因为已经在no_grad()中，并且使用了detach()
            row1_np = row1.cpu().numpy()
            row2_np = row2.cpu().numpy()
            pred_row4_np = pred_row4.cpu().numpy()
            label_row4_np = label_row4.cpu().numpy()
            valid_mask_row1_np = valid_mask_row1.cpu().numpy()
            valid_mask_row2_np = valid_mask_row2.cpu().numpy()
            valid_mask_pred_np = valid_mask_pred.cpu().numpy()
            valid_mask_label_np = valid_mask_label.cpu().numpy()
            
            def extract_number_from_row_np(row, valid_mask, batch_idx=None):
                """从numpy数组行中提取数字
                
                如果数字后面出现 F（LEADING_VALUE=10），返回 -1 表示无效
                因为数字后面出现 F 是不合理的，表示计算未完成
                """
                LEADING_VALUE = 10  # 前导F值
                PAD_VALUE = 11  # PAD值
                
                # 找到第一个非0且有效的位置
                start_idx = None
                for i in range(len(row)):
                    if valid_mask[i] and row[i] != 0:
                        start_idx = i
                        break
                
                if start_idx is None:
                    return 0
                
                # 提取有效数字，记录最后一个数字的位置
                num = 0
                last_digit_idx = None
                digits = []
                for i in range(start_idx, len(row)):
                    val = int(row[i])
                    if val >= 10 or not valid_mask[i]:
                        break
                    num = num * 10 + val
                    digits.append(val)
                    last_digit_idx = i
                
                # 如果提取到了数字，检查数字后面是否有 F（LEADING_VALUE）
                if last_digit_idx is not None:
                    # 检查从最后一个数字位置之后到行尾，是否还有非 PAD 的有效位置，且这些位置的值是 LEADING_VALUE
                    for i in range(last_digit_idx + 1, len(row)):
                        if valid_mask[i]:  # 有效位置（非 PAD）
                            if row[i] == LEADING_VALUE:
                                # 数字后面出现了 F，这是不合理的，返回 -1 表示无效
                                return -1
                            elif row[i] != PAD_VALUE:
                                # 数字后面出现了其他非 PAD 的有效值（0-9），这也是不合理的
                                # 因为数字序列应该是连续的，不应该中间有 F 或其他值
                                return -1
                
                return num
            
            # 为了兼容性，仍然计算位置级别的accuracy（与label比较）
            # mask：排除PAD_VALUE=11，但包括LEADING_VALUE=10
            # labels_for_loss 中的值：0-10（包括LEADING_VALUE=10），11（PAD_VALUE）
            # 需要排除 PAD_VALUE=11，但包括 LEADING_VALUE=10
            PAD_VALUE = 11  # PAD值
            mask = (labels_for_loss != PAD_VALUE)  # 排除PAD_VALUE=11，包括LEADING_VALUE=10
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)
            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels_for_loss)
            
            # 恢复为4行格式：pred_row4是第4行（结果行），label_row4也是第4行（结果行）
            # 从预测的第4行（结果行）提取数字
            pred_result_list = [extract_number_from_row_np(pred_row4_np[b], valid_mask_pred_np[b], batch_idx=b) for b in range(batch_size)]
            
            # 从标签的第4行（结果行）提取数字（排除PAD）
            label_result_list = [extract_number_from_row_np(label_row4_np[b], valid_mask_label_np[b], batch_idx=b) for b in range(batch_size)]
            # 终止判定：当输入与标签完全相等时，应该halt（与预测无关）
            inputs_equal_labels = (inputs == labels).all(dim=1)  # B
            inputs_equal_labels_np = inputs_equal_labels.detach().cpu().numpy()
            
            # 两个停止逻辑：
            # 1. 模型认为该停止了（模型输出的halted信号）- 用于halt loss, steps监控, halt_accuracy
            # 2. 实际应该停止了（按计算应该停止了）- 用于exact_accuracy
            
            # valid_metrics_for_halt: 基于模型的halted信号
            # 只要模型首次出现halted信号，检查当前步骤是否与实际要停止的步数（final_step）一致
            # 不一致，就认为是错的；一致，就认为是对的
            # 用于halt loss, steps监控, halt_accuracy
            original_carry = model_kwargs.get("carry", None)
            current_steps = new_carry.steps.cpu().numpy() if new_carry.steps.is_cuda else new_carry.steps.numpy()
            halted_np = new_carry.halted.cpu().numpy() if new_carry.halted.is_cuda else new_carry.halted.numpy()
            
            # 判断哪些样本是首次halted（之前没有halted，现在halted了）
            first_halted_list = []
            if original_carry is not None:
                prev_halted_np = original_carry.halted.cpu().numpy() if original_carry.halted.is_cuda else original_carry.halted.numpy()
                for b in range(batch_size):
                    # 首次halted：之前没有halted，现在halted了
                    first_halted_list.append(not prev_halted_np[b] and halted_np[b])
            else:
                # 如果没有之前的carry，假设所有halted都是首次halted
                for b in range(batch_size):
                    first_halted_list.append(halted_np[b])
            
            # 计算每个样本是否应该计算halt loss
            # 条件：首次halted
            valid_metrics_for_halt_list = []
            # 计算halt_accuracy：halted时，当前步骤是否等于final_step
            halt_correct_list = []
            for b in range(batch_size):
                is_first_halted = first_halted_list[b]
                # 是否应该计算halt loss：首次halted
                valid_metrics_for_halt_list.append(is_first_halted)
                # halt_accuracy：halted时，输入与标签是否完全一致（应该halt）
                if halted_np[b]:
                    halt_correct_list.append(bool(inputs_equal_labels_np[b]))
                else:
                    halt_correct_list.append(False)  # 未halted，不参与halt_accuracy计算
            
            valid_metrics_for_halt = torch.tensor(
                valid_metrics_for_halt_list,
                dtype=torch.bool,
                device=inputs.device
            )  # B
            
            halt_correct = torch.tensor(
                halt_correct_list,
                dtype=torch.bool,
                device=inputs.device
            )  # B
            
            # 计算seq_is_correct：对所有样本计算，用于后续的halt loss和exact_accuracy
            # 比较预测的结果行与标签的结果行（排除PAD位）
            # 注意：如果提取结果是 -1，表示数字后面有 F，这是不合理的，应该判定为错误
            seq_is_correct_list = []
            for b in range(batch_size):
                # 对所有样本计算正确性（不管是否halted或是否在最后一步）
                # 比较预测的结果行与标签的结果行
                pred_result = pred_result_list[b]
                label_result = label_result_list[b]
                
                # 如果预测或标签的提取结果是 -1（表示数字后面有 F），判定为错误
                if pred_result == -1 or label_result == -1:
                    seq_is_correct_list.append(False)
                else:
                    seq_is_correct_list.append(pred_result == label_result)
            
            seq_is_correct = torch.tensor(
                seq_is_correct_list,
                dtype=torch.bool,
                device=inputs.device
            )  # B
            
        # Standard losses
        # 新数据格式：每条数据是一个状态转移对 (s_i, s_{i+1})
        # 输入是当前状态 s_i，标签是下一个状态 s_{i+1}
        # labels 本身就是单个状态（单个网格），不需要从多个步骤中提取
        # 直接使用 labels 作为目标即可
        
        # 新格式：inputs只有前两行（题目），labels只有后两行（目标），模型输出只有后两行
        # 获取单个网格的大小（从模型配置或labels的长度）
        labels_seq_len = int(labels_for_loss.shape[1])
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'seq_len'):
            # 从模型配置获取seq_len（单个网格的大小）
            step_size = self.model.config.seq_len
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'config') and hasattr(self.model.model.config, 'seq_len'):
            # 如果模型被包装，从底层模型获取
            step_size = self.model.model.config.seq_len
        else:
            # Fallback: 从labels的长度获取（新格式下，labels就是后两行）
            step_size = labels_seq_len
        
        # 恢复为4行格式：计算4行的序列长度和网格宽度
        four_rows_len = labels_seq_len  # labels是完整的4行
        grid_width = four_rows_len // 4  # 每行的列数
        batch_size = int(inputs.shape[0])
        
        # 恢复为4行格式，labels本身就是完整的4行（目标），直接使用即可
        step_labels = labels_for_loss  # B × seq_len（完整的4行）
        
        # 将preds和labels转换为原始值（值+1格式，需要减1）
        pred_values = (outputs["preds"] - 1)  # B × seq_len（完整的4行），值范围是 [0, 10]
        label_values = (labels_for_loss - 1)  # B × seq_len（完整的4行），值范围是 [0, 10]
        current_data_values = (current_data_labels - 1)  # B × seq_len（完整的4行），值范围是 [0, 10]
        # 将inputs转换为原始值（值+1格式，需要减1）
        input_values = (inputs - 1)  # B × seq_len（完整的4行），值范围是 [0, 10]
        
        # 重塑为网格：完整的4行
        input_grid = input_values[:, :four_rows_len].view(batch_size, 4, grid_width)  # B × 4 × grid_width（完整的4行，当前输入）
        pred_grid = pred_values[:, :four_rows_len].view(batch_size, 4, grid_width)  # B × 4 × grid_width（完整的4行）
        label_grid = label_values[:, :four_rows_len].view(batch_size, 4, grid_width)  # B × 4 × grid_width（完整的4行）
        current_data_grid = current_data_values[:, :four_rows_len].view(batch_size, 4, grid_width)  # B × 4 × grid_width（完整的4行，上一步的计算结果）
        
        # 提取前两行和后两行用于损失计算
        pred_row12 = pred_grid[:, :2, :]  # B × 2 × grid_width（前两行：第1行和第2行）
        input_row12 = input_grid[:, :2, :]  # B × 2 × grid_width（前两行：第1行和第2行）
        pred_row34 = pred_grid[:, 2:, :]  # B × 2 × grid_width（后两行：第3行和第4行）
        label_row34 = label_grid[:, 2:, :]  # B × 2 × grid_width（后两行：第3行和第4行）
        # 关键修复：current_data_row34应该是batch的input的后两行（当前输入状态），而不是current_data_grid
        # 因为要判断哪些位置应该变化，需要比较当前输入的后两行和目标的后两行
        input_row34 = input_grid[:, 2:, :]  # B × 2 × grid_width（后两行：当前输入的后两行）
        current_data_row34 = input_row34  # 使用batch的input的后两行
        
        # 1. 前两行的copy loss：前两行（加数）应该保持不变
        # 前两行的预测应该与输入一致
        pred_row12_valid = (pred_row12 >= 0) & (pred_row12 != PAD_VALUE)  # B × 2 × grid_width
        input_row12_valid = (input_row12 >= 0) & (input_row12 != PAD_VALUE)  # B × 2 × grid_width
        row12_should_keep = pred_row12_valid & input_row12_valid  # B × 2 × grid_width（有效位置）
        row12_changed = row12_should_keep & (pred_row12 != input_row12)  # B × 2 × grid_width（不应该变化但变了的位置）
        
        # 将前两行的位置映射回原始序列位置（完整的4行，前两行是第1行和第2行）
        seq_len = step_labels.shape[1]
        row12_loss_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=step_labels.device)
        for b in range(batch_size):
            for row in range(2):
                for col in range(grid_width):
                    if row12_changed[b, row, col]:
                        pos = row * grid_width + col  # 前两行的位置：第1行(row=0)和第2行(row=1)
                        if pos < seq_len:
                            row12_loss_mask[b, pos] = True
        
        # 计算前两行的copy loss
        # 前两行的目标标签应该是inputs（值+1后的格式，范围[1, 12]）
        # 但loss函数需要的是完整的序列，所以使用inputs作为标签，只在row12_loss_mask为True的位置计算loss
        vocab_size = outputs["logits"].shape[-1]
        input_labels = inputs  # B × seq_len（值+1后的格式，范围[1, 12]）
        input_labels_clamped = torch.clamp(input_labels, min=0, max=vocab_size - 1)
        row12_losses = self.loss_fn(outputs["logits"], input_labels_clamped, ignore_index=12, valid_mask=row12_loss_mask) / loss_divisor  # [B, seq_len]
        row12_loss = row12_losses.sum()
        
        # 2. 后两行的loss：将lm_loss拆解为2个惩罚项
        # 1. 后两行中不应该变化的位置如果变了，单独惩罚（不变部分）
        # 2. 应该变的位置没有变对，单独惩罚（变换部分）
        
        # 创建mask：排除PAD位置
        pred_row34_valid = (pred_row34 >= 0) & (pred_row34 != PAD_VALUE)  # B × 2 × grid_width
        label_row34_valid = (label_row34 >= 0) & (label_row34 != PAD_VALUE)  # B × 2 × grid_width
        # current_data_row34现在是batch的input的后两行（当前输入状态）
        current_data_row34_valid = (current_data_row34 >= 0) & (current_data_row34 != PAD_VALUE)  # B × 2 × grid_width
        
        # 1. 后两行不变位置的惩罚项：后两行中不应该变化的位置如果变了，单独惩罚
        # 识别应该变化的位置：比较当前输入的后两行（batch["inputs"]的后两行）和label（目标的后两行），找出变化的位置
        row34_should_change = current_data_row34_valid & label_row34_valid & (current_data_row34 != label_row34)  # B × 2 × grid_width（应该变化的位置）
        row34_should_keep = current_data_row34_valid & label_row34_valid & (current_data_row34 == label_row34)  # B × 2 × grid_width（应该保持不变的位置）
        # 预测中不应该变化的位置如果变了，则惩罚
        row34_keep_changed = row34_should_keep & (pred_row34 != current_data_row34)  # B × 2 × grid_width（不应该变化但变了的位置）
        # 计算后两行不变位置的loss
        row34_keep_loss_mask = row34_keep_changed  # B × 2 × grid_width
        # 将后两行的位置映射回原始序列位置（完整的4行，后两行是第3行和第4行）
        # 确保mask的长度与step_labels和outputs["logits"]的长度匹配
        seq_len = step_labels.shape[1]
        row34_keep_full_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=step_labels.device)
        for b in range(batch_size):
            for row in range(2):
                for col in range(grid_width):
                    if row34_keep_loss_mask[b, row, col]:
                        pos = (row + 2) * grid_width + col  # 后两行的位置：第3行(row=0对应第3行)和第4行(row=1对应第4行)
                        if pos < seq_len:  # 确保不超出序列长度
                            row34_keep_full_mask[b, pos] = True
        
        # 确保 step_labels 的值在有效范围内 [0, vocab_size-1]
        # step_labels 的值范围是 [1, 12]（值+1后的格式），需要转换为 [0, 11] 作为索引
        # 但 PAD (12) 应该被忽略，所以使用 ignore_index=12
        # 同时确保所有值都在 [0, vocab_size-1] 范围内
        vocab_size = outputs["logits"].shape[-1]
        step_labels_clamped = torch.clamp(step_labels, min=0, max=vocab_size - 1)
        row34_keep_losses = self.loss_fn(outputs["logits"], step_labels_clamped, ignore_index=12, valid_mask=row34_keep_full_mask) / loss_divisor  # [B, seq_len]
        row34_keep_loss = row34_keep_losses.sum()
        
        # 2. 应该变的位置没有变对的惩罚项：应该变化的位置如果预测不对，单独惩罚
        row34_change_wrong = row34_should_change & (pred_row34 != label_row34)  # B × 2 × grid_width（应该变化但预测不对的位置）
        # 计算后两行应该变化位置的loss
        row34_change_loss_mask = row34_change_wrong  # B × 2 × grid_width
        # 将后两行的位置映射回原始序列位置（完整的4行，后两行是第3行和第4行）
        # 确保mask的长度与step_labels和outputs["logits"]的长度匹配
        seq_len = step_labels.shape[1]
        row34_change_full_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=step_labels.device)
        for b in range(batch_size):
            for row in range(2):
                for col in range(grid_width):
                    if row34_change_loss_mask[b, row, col]:
                        pos = (row + 2) * grid_width + col  # 后两行的位置：第3行(row=0对应第3行)和第4行(row=1对应第4行)
                        if pos < seq_len:  # 确保不超出序列长度
                            row34_change_full_mask[b, pos] = True
        
        # 确保 step_labels 的值在有效范围内 [0, vocab_size-1]
        vocab_size = outputs["logits"].shape[-1]
        step_labels_clamped = torch.clamp(step_labels, min=0, max=vocab_size - 1)
        row34_change_losses = self.loss_fn(outputs["logits"], step_labels_clamped, ignore_index=12, valid_mask=row34_change_full_mask) / loss_divisor  # [B, seq_len]
        row34_change_loss = row34_change_losses.sum()
        
        # 计算动态系数
        # 1. 后两行冻结行（不变部分）：从1.5降到0.5，再缓慢降到0.3，最后保持
        # 2. 要改动的行（变换部分）：在warm阶段逐渐上升到1.0并保持
        
        # 获取当前步数（从batch或使用默认值）
        if training_step is not None:
            if isinstance(training_step, torch.Tensor):
                current_step = training_step.item() if training_step.numel() == 1 else int(training_step.cpu().numpy()[0])
            else:
                current_step = int(training_step)
        else:
            current_step = self.current_step
        
        # 获取warmup步数（从batch或使用默认值）
        if warmup_steps is not None:
            if isinstance(warmup_steps, torch.Tensor):
                warmup_steps = warmup_steps.item() if warmup_steps.numel() == 1 else int(warmup_steps.cpu().numpy()[0])
            else:
                warmup_steps = int(warmup_steps)
        else:
            warmup_steps = self.warmup_steps
        
        device = outputs["logits"].device
        dtype = outputs["logits"].dtype
        
        # 1. 后两行冻结行系数（不变部分）：从1.5降到0.5，再缓慢降到0.3，最后保持
        # 第一阶段：warmup的前50%，从1.5降到0.5
        # 第二阶段：warmup的后50%，从0.5缓慢降到0.3
        # 第三阶段：warmup之后，保持0.3
        row34_keep_phase1 = warmup_steps * 0.5  # 第一阶段：前50%
        row34_keep_phase2 = warmup_steps  # 第二阶段：后50%
        if current_step < row34_keep_phase1:
            # 第一阶段：从1.5降到0.5
            row34_keep_weight = 1.5 - (1.5 - 0.5) * (current_step / row34_keep_phase1)
        elif current_step < row34_keep_phase2:
            # 第二阶段：从0.5缓慢降到0.3
            progress = (current_step - row34_keep_phase1) / (row34_keep_phase2 - row34_keep_phase1)
            row34_keep_weight = 0.5 - (0.5 - 0.3) * progress
        else:
            # 第三阶段：保持0.3
            row34_keep_weight = 0.3
        row34_keep_weight = torch.tensor(row34_keep_weight, device=device, dtype=dtype)
        
        # 2. 要改动的行系数（变换部分）：在warm阶段逐渐上升到1.0并保持
        if current_step < warmup_steps:
            # warmup阶段：从0逐渐上升到1.0
            row34_change_weight = current_step / warmup_steps
        else:
            # warmup之后：保持1.0
            row34_change_weight = 1.0
        row34_change_weight = torch.tensor(row34_change_weight, device=device, dtype=dtype)
        
        # lm_loss 由三项组成：
        # 1. copy_loss（前两行的loss，带系数，因为copy_loss很容易学习）
        # 2. 后两行不变的loss（row34_keep_loss，带动态权重）
        # 3. 后两行改变的loss（row34_change_loss，带动态权重）
        copy_loss = row12_loss
        copy_loss_coeff = torch.tensor(self.copy_loss_coeff, device=device, dtype=dtype)
        lm_loss = copy_loss_coeff * copy_loss + row34_keep_weight * row34_keep_loss + row34_change_weight * row34_change_loss
        
        # q_halt_loss: 每一步都计算，目标为“输入与标签是否完全一致”
        # 与模型是否输出 halted 信号无关，覆盖全部样本
        halt_targets_bool = inputs_equal_labels  # B，是否应当停止
        halt_targets = halt_targets_bool.to(outputs["q_halt_logits"].dtype)
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            halt_targets,
            reduction="sum"
        )
        
        # halt_accuracy: 每一步比较预测的halt信号与halt_targets是否一致
        # 若没有q_halt_logits，回退为0
        if "q_halt_logits" in outputs:
            halt_pred = outputs["q_halt_logits"] >= 0  # logits>=0 表示预测halt
            halt_accuracy_value = (halt_pred == halt_targets_bool).sum()
        else:
            halt_accuracy_value = torch.tensor(0, device=outputs["logits"].device, dtype=torch.long)
        
        # Copy loss: 前两行的copy loss（已恢复）
        
        # Q continue loss
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")
            metrics["q_continue_loss"] = q_continue_loss.detach()
        
        # 总损失 = lm_loss + q_loss
        # lm_loss 已经包含了 copy_loss、后两行不变的loss和后两行改变的loss
        # 使用可配置的权重
        total_loss = self.lm_loss_weight * lm_loss + self.q_halt_loss_weight * (q_halt_loss + q_continue_loss)
        
        # 汇总metrics（包括copy_loss）
        # 注意：count 需要包含 padding 样本，因为 exact_accuracy 和 q_halt_accuracy 的计算包含了所有样本（包括 padding）
        # seq_is_correct 和 halt_accuracy_value 都是对 batch_size 个样本计算的，所以 count 也应该是 batch_size
        metrics = {
            # count: 本批所有样本数（包含padding），以匹配 exact_accuracy 和 q_halt_accuracy 的计算
            "count": torch.tensor(batch_size, dtype=torch.long, device=inputs.device),
            
            # 位置级别的accuracy（与label比较，用于兼容性）
            # 使用首次halted信号，因为这是模型认为应该停止的时候
            "accuracy":       torch.where(valid_metrics_for_halt, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
            
            # exact_accuracy: 不管halt状态，统计所有完全正确的样本数
            # 只要预测结果与期望结果完全一致，就计入exact_accuracy
            # 不依赖步骤或halt状态
            "exact_accuracy": seq_is_correct.sum(),
            
            # q_halt_accuracy: 每一步比较halt预测与halt_target（输入与标签是否一致）
            "q_halt_accuracy": halt_accuracy_value,
            
            # steps: 基于模型的halted信号，记录halted时的步数（不管是否一致）
            # 所有halted样本的步数
            "steps": torch.where(new_carry.halted, new_carry.steps, 0).sum(),
        }
        
        # 更新metrics
        metrics.update({
            "lm_loss": lm_loss.detach(),  # 总lm_loss（包含copy_loss、后两行不变的loss、后两行改变的loss）
            "copy_loss": copy_loss.detach(),  # 前两行的copy loss（未加权）
            "copy_loss_coeff": copy_loss_coeff.detach(),  # copy_loss的系数
            "lm_loss_row34_keep": row34_keep_loss.detach(),  # 后两行不应该变化但变了的惩罚
            "lm_loss_row34_change": row34_change_loss.detach(),  # 后两行应该变化但预测不对的惩罚
            "weight_row34_keep": row34_keep_weight.detach(),  # 后两行冻结行动态系数
            "weight_row34_change": row34_change_weight.detach(),  # 后两行改动行动态系数
            "q_halt_loss": q_halt_loss.detach(),
            "total_loss": total_loss.detach(),
        })
        
        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()

