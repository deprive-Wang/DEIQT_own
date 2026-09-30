"""组合 Encoder 和 Decoder，定义完整 DEIQT 的前向流程。"""

import torch
from torch import nn

from decoder import AttentionPanelDecoder
from encoder import ViTEncoder


class DEIQT(nn.Module):
    """完整的图像质量回归模型：图像 [B, 3, 224, 224] -> 分数 [B, 1]。"""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = ViTEncoder()
        self.head = AttentionPanelDecoder()

    def no_weight_decay(self) -> set[str]:
        """保留作者对 CLS token 和位置编码免除权重衰减的设置。"""
        return {"encoder.cls_token", "encoder.pos_embed"}

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        cls_features, patch_tokens = self.encoder(images)
        return self.head(patch_tokens, cls_features)
