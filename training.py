"""单设备、单轮训练；多轮调度、保存与评估由训练入口组织。"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from lr_scheduler import WarmupStepLR


@dataclass
class TrainEpochResult:
    """loss 是按实际裁剪样本数加权的平均 SmoothL1 损失。"""

    loss: float
    batches: int
    samples: int
    optimizer_steps: int
    skipped_steps: int


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: Optimizer,
    *,
    scaler: torch.amp.GradScaler | None = None,
    clip_grad: float = 5.0,
    scheduler: WarmupStepLR | None = None,
    progress_callback: Callable[[TrainEpochResult], None] | None = None,
) -> TrainEpochResult:
    """完成一轮训练，每个 batch 更新一次，不做梯度累积。

    默认 FP32；CUDA 混合精度需传入启用的 GradScaler，同一个 scaler 应跨轮复用。
    传入 scheduler 时在每个 batch 后推进调度；同一 scheduler 必须跨轮复用。
    不传 scheduler 时保持优化器当前学习率。输入标签必须为 [B, 1]。
    训练样本可以打乱，不能按连续 patch_num 个样本计算图像级相关系数。
    """
    if not math.isfinite(clip_grad) or clip_grad <= 0:
        raise ValueError("clip_grad 必须为有限正数")
    if len(data_loader) == 0:
        raise ValueError("训练 DataLoader 为空，请检查数据量、batch_size 和 drop_last")
    if scheduler is not None:
        if scheduler.optimizer is not optimizer:
            raise ValueError("scheduler 与训练必须使用同一个 optimizer")
        if scheduler.steps_per_epoch != len(data_loader):
            raise ValueError("scheduler.steps_per_epoch 必须等于训练 DataLoader 的 batch 数")
    device = next(model.parameters()).device
    use_amp = scaler is not None and scaler.is_enabled()
    if use_amp and device.type != "cuda":
        raise ValueError("本模块的混合精度训练仅支持 CUDA")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    criterion = nn.SmoothL1Loss(beta=1.0, reduction="mean")
    total_loss = 0.0
    sample_count = batch_count = optimizer_steps = skipped_steps = 0
    model.train()
    try:
        for images, targets in data_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if targets.shape != (images.shape[0], 1):
                raise ValueError("训练标签形状必须为 [B, 1]，避免损失计算发生广播")
            if not torch.isfinite(targets).all():
                raise ValueError("训练标签包含 NaN 或无穷值")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(images)
            if outputs.shape != targets.shape:
                raise ValueError("模型输出与训练标签均应为 [B, 1]")
            # 在 FP32 中计算回归损失，避免半精度下的大 MOS 残差降低数值稳定性。
            loss = criterion(outputs.float(), targets.float())
            if not torch.isfinite(loss):
                raise FloatingPointError("训练损失非有限，已停止更新")

            if use_amp:
                scaler.scale(loss).backward()
                # 必须先还原真实梯度，再裁剪；缩放后的梯度不能直接裁剪。
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(parameters, clip_grad)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # GradScaler 溢出时跳过更新并减小 scale，不能记成有效优化步。
                skipped = scaler.get_scale() < previous_scale
                skipped_steps += int(skipped)
                optimizer_steps += int(not skipped)
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(parameters, clip_grad, error_if_nonfinite=True)
                optimizer.step()
                optimizer_steps += 1

            if scheduler is not None:
                # 对齐作者：即使 AMP 本批跳过参数更新，batch 时间轴也继续推进。
                scheduler.step()
            total_loss += loss.detach().item() * images.shape[0]
            sample_count += images.shape[0]
            batch_count += 1
            if progress_callback is not None:
                progress_callback(TrainEpochResult(
                    total_loss / sample_count, batch_count, sample_count,
                    optimizer_steps, skipped_steps,
                ))
    finally:
        # 异常退出或一轮结束后都不残留旧梯度，方便下一轮或评估。
        optimizer.zero_grad(set_to_none=True)
    return TrainEpochResult(
        total_loss / sample_count, batch_count, sample_count, optimizer_steps, skipped_steps
    )
