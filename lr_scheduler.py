"""按 timm 0.6.13 StepLRScheduler 规则实现预热与阶梯衰减。"""

import math
from collections.abc import Mapping

from torch.optim import Optimizer


class WarmupStepLR:
    """按 batch 调度，适用于每个 batch 更新一次的单卡训练。

    作者未固定 timm 版本；这里明确采用 v0.6.13 的规则：预热计入衰减周期，
    不额外减去 warmup_steps，也不应用 MIN_LR（作者的 step 分支未使用它）。
    参考：https://github.com/huggingface/pytorch-image-models/blob/v0.6.13/timm/scheduler/step_lr.py
    """

    def __init__(
        self,
        optimizer: Optimizer,
        steps_per_epoch: int,
        warmup_epochs: int = 3,
        decay_epochs: int = 3,
        warmup_lr: float = 2e-7,
        decay_rate: float = 0.1,
    ) -> None:
        for name, value, minimum in (
            ("steps_per_epoch", steps_per_epoch, 1),
            ("warmup_epochs", warmup_epochs, 0),
            ("decay_epochs", decay_epochs, 1),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} 必须为大于等于 {minimum} 的整数")
        if not math.isfinite(warmup_lr) or warmup_lr < 0:
            raise ValueError("warmup_lr 必须为有限非负数")
        if not math.isfinite(decay_rate) or not 0 < decay_rate <= 1:
            raise ValueError("decay_rate 必须在 (0, 1] 范围内")
        base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        if not base_lrs or any(not math.isfinite(lr) or lr <= 0 for lr in base_lrs):
            raise ValueError("优化器各参数组的基础学习率必须为有限正数")
        if warmup_epochs and any(warmup_lr > lr for lr in base_lrs):
            raise ValueError("warmup_lr 不能高于基础学习率")
        self.optimizer = optimizer
        self.steps_per_epoch = steps_per_epoch
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.warmup_lr = warmup_lr
        self.decay_rate = decay_rate
        self.base_lrs = base_lrs
        self.completed_batches = 0
        self._set_lrs(self._get_lrs(0))

    def _get_lrs(self, step_index: int) -> list[float]:
        warmup_steps = self.warmup_epochs * self.steps_per_epoch
        if step_index < warmup_steps:
            return [
                self.warmup_lr + step_index * (base_lr - self.warmup_lr) / warmup_steps
                for base_lr in self.base_lrs
            ]
        decay_steps = self.decay_epochs * self.steps_per_epoch
        return [
            base_lr * self.decay_rate ** (step_index // decay_steps)
            for base_lr in self.base_lrs
        ]

    def _set_lrs(self, learning_rates: list[float]) -> None:
        for group, learning_rate in zip(self.optimizer.param_groups, learning_rates):
            group["lr"] = learning_rate

    def step(self) -> None:
        """每个 batch 处理完成后调用一次，包括 AMP 溢出跳过更新的 batch。"""
        # 作者在更新后传 epoch * num_steps + idx（从 0 开始），因此首个
        # batch 后仍设置 t=0 的学习率。这里保留这个偏移，不擅自改成 t=1。
        self._set_lrs(self._get_lrs(self.completed_batches))
        self.completed_batches += 1

    def state_dict(self) -> dict[str, object]:
        """保存基础学习率、调度配置与累计 batch 数，不保存优化器对象。"""
        return {
            "steps_per_epoch": self.steps_per_epoch,
            "warmup_epochs": self.warmup_epochs,
            "decay_epochs": self.decay_epochs,
            "warmup_lr": self.warmup_lr,
            "decay_rate": self.decay_rate,
            "base_lrs": self.base_lrs.copy(),
            "completed_batches": self.completed_batches,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """恢复进度并同步优化器学习率；恢复时必须使用相同调度配置。"""
        expected = self.state_dict()
        if not isinstance(state, Mapping) or state.keys() != expected.keys():
            raise ValueError("调度器状态字段不完整或包含未知字段")
        for name, value in expected.items():
            if name != "completed_batches" and state[name] != value:
                raise ValueError(f"调度器配置不一致：{name}；恢复时不能改变每轮步数或学习率配置")
        completed = state["completed_batches"]
        if type(completed) is not int or completed < 0:
            raise ValueError("completed_batches 必须为非负整数")
        # 全部校验通过后再修改状态，避免失败恢复留下部分更新。
        self.completed_batches = completed
        self._set_lrs(self._get_lrs(max(completed - 1, 0)))
