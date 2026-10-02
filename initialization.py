"""模型组件共用的参数初始化规则。"""

# 线性层 - mean=0, std=0.02
# 归一化层 - weight=1, bias=0

from torch import nn


def init_weights(module: nn.Module) -> None:
    """统一 Encoder 和 Decoder 中线性层、归一化层的初始化方式。"""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
