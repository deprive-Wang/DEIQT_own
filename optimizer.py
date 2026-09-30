"""DEIQT 的 AdamW 优化器及权重衰减分组。"""

import math

from torch.optim import AdamW

from model import DEIQT


def build_optimizer(
    model: DEIQT, learning_rate: float = 2e-4, weight_decay: float = 0.05
) -> AdamW:
    """使用作者 LIVE-C 配置；learning_rate 是基础值，预热由调度模块控制。"""
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate 必须为有限正数")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay 必须为有限非负数")
    decay, no_decay = [], []
    skip = model.no_weight_decay()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        # 一维参数包括 LayerNorm 和 LayerScale；panel embedding 仍参与衰减。
        if parameter.ndim == 1 or name.endswith(".bias") or name in skip:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    if not decay and not no_decay:
        raise ValueError("模型没有可训练参数")
    return AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
