from typing import Any, Tuple, Dict, Sequence, Optional

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
        new_carry, outputs = self.model(**model_kwargs_for_base)
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
    def __init__(self, model: nn.Module, loss_type: str, copy_loss_weight: float = 1.0, lm_loss_weight: float = 1.0, q_halt_loss_weight: float = 0.5):
        super().__init__(model, loss_type, lm_loss_weight=lm_loss_weight, q_halt_loss_weight=q_halt_loss_weight)
        self.copy_loss_weight = copy_loss_weight
        # 加法数据集的pad_id是11（值+1后的PAD值）
        self.pad_id = 11
        
    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # 调用父类的forward，但需要添加copy loss
        # 注意：batch包含原始输入，carry.current_data可能包含上一步的输出（递归推理）
        batch = model_kwargs.get("batch", {})
        # 底层模型只需要carry和batch，不需要return_keys
        model_kwargs_for_base = {k: v for k, v in model_kwargs.items() if k != "return_keys"}
        new_carry, outputs = self.model(**model_kwargs_for_base)
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
        
        # 确保inputs是2D tensor (batch_size, seq_len)
        if inputs.dim() != 2:
            raise RuntimeError(f"Invalid inputs shape: {inputs.shape}, expected 2D tensor (batch_size, seq_len)")

        # 在微调中，PAD值不进行掩码处理
        # 将labels中被转换为-100的PAD值恢复为原始的pad_id（11）
        labels_for_loss = labels.clone()
        labels_for_loss[labels == IGNORE_LABEL_ID] = self.pad_id
        
        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # 比较预测与期望结果（num1 + num2），而不是与label比较
            # 从inputs中提取两个加数，从预测中提取结果，然后比较
            # 注意：这部分计算需要在torch.no_grad()中，并且使用detach()避免梯度追踪
            # 为了兼容torch.compile，我们使用torch.jit.ignore装饰器或者将这部分移到外部
            # 但更简单的方法是：在no_grad中，将tensor移到CPU并转换为Python对象进行计算
            seq_len = int(inputs.shape[-1])
            batch_size = int(inputs.shape[0])
            grid_width = seq_len // 4
            
            # 将inputs和preds转换为原始值（值+1格式，需要减1）
            # inputs: B × seq_len，值范围是 [1, 11]（1-10对应数字0-9，11是PAD）
            # outputs["preds"]: B × seq_len，token id范围是 [0, vocab_size-1]
            # 在加法任务中，token id 1对应数字0（值+1），token id 2对应数字1，...，token id 10对应数字9，token id 11对应PAD
            # 所以pred_values = outputs["preds"] - 1，值范围是 [0, 10]（0-9对应数字0-9，10是PAD）
            input_values = (inputs - 1).detach()  # B × seq_len，值范围是 [0, 10]（0-9对应数字0-9，10是PAD）
            pred_values = (outputs["preds"] - 1).detach()  # B × seq_len，值范围是 [0, 10]
            
            # 重塑为网格：B × 4 × grid_width
            input_grid = input_values[:, :seq_len].view(batch_size, 4, grid_width)  # B × 4 × grid_width
            pred_grid = pred_values[:, :seq_len].view(batch_size, 4, grid_width)  # B × 4 × grid_width
            
            # 提取两个加数（第1行和第2行）和预测结果（第4行）
            row1 = input_grid[:, 0, :]  # B × grid_width
            row2 = input_grid[:, 1, :]  # B × grid_width
            pred_row4 = pred_grid[:, 3, :]  # B × grid_width
            
            # 创建mask：有效数字（0-9，不包括PAD=10）
            valid_mask_row1 = (row1 >= 0) & (row1 < 10)  # B × grid_width
            valid_mask_row2 = (row2 >= 0) & (row2 < 10)  # B × grid_width
            valid_mask_pred = (pred_row4 >= 0) & (pred_row4 < 10)  # B × grid_width
            
            # 将tensor移到CPU并转换为numpy进行计算（在no_grad中，这是安全的）
            # 注意：这部分代码在torch.no_grad()中，并且使用detach()，不会影响梯度
            # 为了兼容torch.compile，我们需要确保numpy转换不会被编译
            # 使用torch._dynamo.mark_dynamic标记这些tensor，或者直接转换（在no_grad中应该安全）
            # 最简单的方法：直接转换，因为已经在no_grad()中，并且使用了detach()
            row1_np = row1.cpu().numpy()
            row2_np = row2.cpu().numpy()
            pred_row4_np = pred_row4.cpu().numpy()
            valid_mask_row1_np = valid_mask_row1.cpu().numpy()
            valid_mask_row2_np = valid_mask_row2.cpu().numpy()
            valid_mask_pred_np = valid_mask_pred.cpu().numpy()
            
            def extract_number_from_row_np(row, valid_mask):
                """从numpy数组行中提取数字"""
                # 找到第一个非0且有效的位置
                for i in range(len(row)):
                    if valid_mask[i] and row[i] != 0:
                        start_idx = i
                        break
                else:
                    return 0
                
                # 提取有效数字
                num = 0
                for i in range(start_idx, len(row)):
                    val = int(row[i])
                    if val >= 10 or not valid_mask[i]:
                        break
                    num = num * 10 + val
                return num
            
            # 为了兼容性，仍然计算位置级别的accuracy（与label比较）
            mask = torch.ones(labels.shape, dtype=torch.bool, device=labels.device)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)
            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels_for_loss)
            
            # 对每个batch提取数字（这部分在no_grad中，不会影响梯度）
            num1_list = [extract_number_from_row_np(row1_np[b], valid_mask_row1_np[b]) for b in range(batch_size)]
            num2_list = [extract_number_from_row_np(row2_np[b], valid_mask_row2_np[b]) for b in range(batch_size)]
            
            pred_result_list = [extract_number_from_row_np(pred_row4_np[b], valid_mask_pred_np[b]) for b in range(batch_size)]
            
            # 计算期望结果
            expected_results = [n1 + n2 for n1, n2 in zip(num1_list, num2_list)]
            
            # 计算每个样本的最大位数，确定最后一步
            # 根据数据生成逻辑：pad_to_len = max(digits1_count, digits2_count) + 1
            # 最后一步应该是 pad_to_len 步
            def count_digits_np(num: int) -> int:
                """计算数字的位数"""
                if num == 0:
                    return 1
                count = 0
                n = num
                while n > 0:
                    count += 1
                    n //= 10
                return count
            
            # 计算每个样本的最大位数和最后一步
            final_steps_list = []
            for b in range(batch_size):
                digits1_count = count_digits_np(num1_list[b])
                digits2_count = count_digits_np(num2_list[b])
                actual_max_digits = max(digits1_count, digits2_count)
                # pad_to_len = actual_max_digits + 1，这是最后一步
                final_step = actual_max_digits + 1
                final_steps_list.append(final_step)
            
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
            # 注意：初始状态halted=True是默认值，需要特殊处理
            first_halted_list = []
            if original_carry is not None:
                prev_halted_np = original_carry.halted.cpu().numpy() if original_carry.halted.is_cuda else original_carry.halted.numpy()
                prev_steps_np = original_carry.steps.cpu().numpy() if original_carry.steps.is_cuda else original_carry.steps.numpy()
                for b in range(batch_size):
                    # 首次halted的判断：
                    # 1. 之前没有halted（prev_halted=False），现在halted了（正常情况）
                    # 2. 或者之前halted但steps=0（初始状态），现在halted了且steps>0（初始状态的特殊情况）
                    # 注意：如果之前halted但steps>0，说明之前已经halt过了，不算首次halted
                    is_initial_state = prev_halted_np[b] and prev_steps_np[b] == 0
                    # 如果是初始状态，需要确保现在halted了且已经执行了步骤（steps>0）
                    if is_initial_state:
                        is_first_halted = halted_np[b] and current_steps[b] > 0
                    else:
                        # 正常情况：之前没有halted，现在halted了
                        is_first_halted = not prev_halted_np[b] and halted_np[b]
                    first_halted_list.append(is_first_halted)
            else:
                # 如果没有之前的carry，说明这是第一次调用
                # 如果halted=True且steps>0，这是首次halted（已经执行了步骤后halt）
                # 如果halted=True且steps=0，这是初始状态，不算首次halted
                for b in range(batch_size):
                    if halted_np[b] and current_steps[b] > 0:
                        # halted且已经执行了步骤，算首次halted
                        first_halted_list.append(True)
                    else:
                        # 没有halted，或者halted但steps=0（初始状态），不算首次halted
                        first_halted_list.append(False)
            
            # 检测哪些样本是因为达到最大步数而强制halted的（模型没有输出halt信号）
            # 需要检查模型的halt信号：q_halt_logits > 0 (no_ACT_continue) 或 q_halt_logits > q_continue_logits
            # 获取模型的halt信号
            q_halt_logits = outputs["q_halt_logits"]  # [B] 或 [B, 1]
            if q_halt_logits.ndim > 1:
                q_halt_logits = q_halt_logits.squeeze(-1)
            
            # 检查是否有q_continue_logits
            has_q_continue = "q_continue_logits" in outputs
            if has_q_continue:
                q_continue_logits = outputs["q_continue_logits"]
                if q_continue_logits.ndim > 1:
                    q_continue_logits = q_continue_logits.squeeze(-1)
                # 模型halt信号：q_halt_logits > q_continue_logits
                model_halt_signal = (q_halt_logits > q_continue_logits).cpu().numpy()
            else:
                # 如果没有q_continue_logits，使用no_ACT_continue逻辑：q_halt_logits > 0
                model_halt_signal = (q_halt_logits > 0).cpu().numpy()
            
            # 获取halt_max_steps（从模型配置中）
            halt_max_steps = getattr(self.model.config, 'halt_max_steps', 16) if hasattr(self.model, 'config') else 16
            
            # 计算每个样本是否应该计算halt loss
            # 条件：首次halted（包括模型halt和强制halt）
            valid_metrics_for_halt_list = []
            # 计算halt_accuracy：halted时，当前步骤是否等于final_step
            halt_correct_list = []
            # 计算halt_targets：应该halt的目标值
            halt_targets_list = []
            for b in range(batch_size):
                is_first_halted = first_halted_list[b]
                current_step = current_steps[b]
                final_step = final_steps_list[b]
                is_max_step = current_step >= halt_max_steps
                
                # 判断是否是因为达到最大步数而强制halted的（模型没有输出halt信号）
                is_forced_halt = is_first_halted and is_max_step and not model_halt_signal[b]
                
                # 是否应该计算halt loss：首次halted（包括模型halt和强制halt）
                valid_metrics_for_halt_list.append(is_first_halted)
                
                # halt_accuracy：halted时，当前步骤是否等于final_step
                if halted_np[b]:
                    halt_correct_list.append(current_step == final_step)
                else:
                    halt_correct_list.append(False)  # 未halted，不参与halt_accuracy计算
                
                # halt_targets：应该halt的目标值
                # 如果是因为达到最大步数而强制halted的，目标值应该是1（应该halt）
                # 如果是模型halt的，目标值取决于当前步骤是否等于final_step
                if is_forced_halt:
                    # 达到最大步数但模型没有halt，应该halt，目标值为1
                    halt_targets_list.append(1.0)
                elif is_first_halted:
                    # 模型halt的，目标值取决于当前步骤是否等于final_step
                    halt_targets_list.append(1.0 if (current_step == final_step) else 0.0)
                else:
                    # 不应该计算halt loss，目标值设为0（不会被使用）
                    halt_targets_list.append(0.0)
            
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
            
            # valid_metrics_for_exact: 基于实际应该停止的步骤（最后一步）
            # 用于exact_accuracy，不依赖模型的halted信号
            # 注意：current_steps已经在上面计算过了
            valid_metrics_for_exact_list = []
            for b in range(batch_size):
                # 只有在最后一步时，才认为是有效样本（用于exact_accuracy）
                is_final_step = (current_steps[b] == final_steps_list[b])
                # 当前步骤不应该超过最后一步（如果超过，说明已经完成过，不应该再次统计）
                is_not_over = (current_steps[b] <= final_steps_list[b])
                # 同时满足两个条件：是最后一步、没有超过最后一步
                valid_metrics_for_exact_list.append(is_final_step and is_not_over)
            
            valid_metrics_for_exact = torch.tensor(
                valid_metrics_for_exact_list,
                dtype=torch.bool,
                device=inputs.device
            )  # B
            
            # 计算seq_is_correct：对所有样本计算，用于后续的halt loss和exact_accuracy
            seq_is_correct_list = []
            for b in range(batch_size):
                # 对所有样本计算正确性（不管是否halted或是否在最后一步）
                seq_is_correct_list.append(pred_result_list[b] == expected_results[b])
            
            seq_is_correct = torch.tensor(
                seq_is_correct_list,
                dtype=torch.bool,
                device=inputs.device
            )  # B
            
            metrics = {
                # count: 首次halted的样本数（用于halt相关指标）
                "count": valid_metrics_for_halt.sum(),
                
                # 位置级别的accuracy（与label比较，用于兼容性）
                # 使用首次halted信号，因为这是模型认为应该停止的时候
                "accuracy":       torch.where(valid_metrics_for_halt, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                
                # exact_accuracy: 基于实际应该停止的步骤（最后一步）
                # 不依赖模型的halted信号，因为如果模型停早了，继续推理可能会正确
                "exact_accuracy": (valid_metrics_for_exact & seq_is_correct).sum(),
                
                # q_halt_accuracy: 基于模型的halted信号，检查halt预测是否正确
                # 如果halted且当前步骤等于final_step，则halt预测正确；否则halt预测错误
                "q_halt_accuracy": (valid_metrics_for_halt & halt_correct).sum(),
                
                # steps: 基于模型的halted信号，记录首次halted时的步数（不管是否一致）
                # 只统计首次halted的样本的步数，与count保持一致
                "steps": torch.where(valid_metrics_for_halt, new_carry.steps, 0).sum(),
            }

        # Standard losses
        # 修改：在所有步骤都计算lm_loss，使用对应步骤的标签
        # labels包含所有步骤的拼接：[s1(step_size), s2(step_size), s3(step_size), ...]
        # 当前步骤是steps，应该使用labels中的第(steps-1)个步骤
        # step_size是单个步骤的大小（4行×grid_width列）
        # 注意：inputs的seq_len是填充后的长度（64），但实际网格大小是16（4行×4列）
        # labels的seq_len是所有步骤拼接后的总长度（64），每个步骤是16
        # 我们需要从inputs的实际网格大小计算step_size
        # inputs的前grid_size个值是单个网格，grid_size = 4 * grid_width
        # 由于inputs可能被填充，我们需要从labels的长度和步骤数推断step_size
        # 或者直接从inputs的实际网格大小计算（假设inputs的前16个值是网格）
        # 更简单的方法：从labels的长度推断step_size
        # labels的seq_len是所有步骤的总长度，我们需要知道有多少个步骤
        # 但我们可以从inputs的seq_len计算：如果inputs的seq_len是64，那么实际网格大小可能是16
        # 实际上，我们可以从outputs["logits"]的形状推断：logits的seq_len应该等于单个网格的大小
        # 但更直接的方法：从labels的长度和已知的步骤数计算
        # 为了简化，我们假设step_size = outputs["logits"].shape[1]，因为logits的seq_len应该等于单个网格的大小
        logits_seq_len = outputs["logits"].shape[1]  # 这应该是单个网格的大小（16）
        step_size = logits_seq_len  # 每个步骤的大小等于单个网格的大小
        grid_width = step_size // 4  # 网格宽度（列数）
        batch_size = int(inputs.shape[0])
        
        # 从labels中提取对应步骤的标签
        # 对于每个样本，根据其当前步骤提取对应的标签
        step_labels = torch.zeros_like(labels_for_loss)  # B × seq_len
        for b in range(batch_size):
            current_step = int(new_carry.steps[b].item())
            if current_step > 0:
                # 提取第(current_step-1)个步骤的标签（因为步骤从1开始，索引从0开始）
                step_idx = current_step - 1
                start_idx = step_idx * step_size
                end_idx = start_idx + step_size
                if end_idx <= labels_for_loss.shape[1]:
                    step_labels[b, :step_size] = labels_for_loss[b, start_idx:end_idx]
                else:
                    # 如果超出范围，使用最后一个步骤的标签（填充时复制的最终状态）
                    # 找到最后一个完整的步骤
                    last_step_start = (labels_for_loss.shape[1] // step_size - 1) * step_size
                    if last_step_start >= 0:
                        step_labels[b, :step_size] = labels_for_loss[b, last_step_start:last_step_start + step_size]
        
        # 计算所有步骤的lm_loss
        step_losses = self.loss_fn(outputs["logits"], step_labels, ignore_index=-999, valid_mask=mask) / loss_divisor  # [B, seq_len]
        step_losses_per_sample = step_losses.sum(-1)  # [B]
        lm_loss = step_losses_per_sample.sum()
        
        # q_halt_loss: 在模型首次输出halted信号时，看当前步骤是否与实际要停止的步数（final_step）一致
        # 如果达到最大步数但模型没有halt，也应该计算loss，目标值为1（应该halt）
        # 只做一次（第一次halted时），用于训练halt预测
        # 逻辑：
        # - 如果当前步骤 == final_step：q_halt_logits应该>=0（正确halt，应该halt）
        # - 如果当前步骤 != final_step：q_halt_logits应该<0（错误halt，不应该halt，应该继续）
        # - 如果达到最大步数但模型没有halt：q_halt_logits应该>=0（应该halt，目标值为1）
        q_halt_loss = torch.tensor(0.0, device=outputs["logits"].device, dtype=outputs["logits"].dtype)
        if valid_metrics_for_halt.any():
            # 对所有首次halted的样本计算q_halt_loss
            # 目标值：从halt_targets_list中获取（已经考虑了强制halt的情况）
            halt_targets_tensor = torch.tensor(
                halt_targets_list,
                dtype=outputs["q_halt_logits"].dtype,
                device=outputs["q_halt_logits"].device
            )
            halt_targets = halt_targets_tensor[valid_metrics_for_halt]
            q_halt_loss = F.binary_cross_entropy_with_logits(
                outputs["q_halt_logits"][valid_metrics_for_halt],
                halt_targets,
                reduction="sum"
            )
        
        # Copy loss: 已禁用，不再计算
        # 不再记录到metrics中
        
        # Q continue loss
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")
            metrics["q_continue_loss"] = q_continue_loss.detach()
        
        # 总损失 = lm_loss + q_loss（不包含copy_loss）
        # copy_loss已禁用，不再参与训练
        # 使用可配置的权重
        total_loss = self.lm_loss_weight * lm_loss + self.q_halt_loss_weight * (q_halt_loss + q_continue_loss)
        
        # 更新metrics（不包括copy_loss, copy_accuracy, halt_accuracy, lm_loss_weight, q_halt_loss_weight）
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
            "total_loss": total_loss.detach(),
        })
        
        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()

