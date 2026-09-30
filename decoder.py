"""Attention-Panel Decoder 与共享质量评分层。"""

import torch
from torch import nn

from initialization import init_weights


class AttentionPanelDecoder(nn.Module):
    """用六个 panel 成员读取图像特征，并汇总为一个质量分数。"""

    def __init__(self) -> None:
        super().__init__()
        # panel 成员数和注意力头数含义不同，作者默认恰好都取 6。
        self.num_panel_members = 6
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=384,
            nhead=6,
            dim_feedforward=1536,
            dropout=0.0,
            activation=nn.functional.gelu,
            batch_first=True,
            norm_first=True,
            # 作者 Decoder 使用默认的 1e-5，与 Encoder 的 1e-6 不同。
            layer_norm_eps=1e-5,
        )
        self.bunch_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.bunch_embedding = nn.Parameter(
            torch.empty(1, self.num_panel_members, 384)
        )
        # 六个成员共用一个无偏置的评分层，并非各自拥有独立回归头。
        self.heads = nn.Linear(384, 1, bias=False)
        nn.init.trunc_normal_(self.bunch_embedding, std=0.02)
        # 与作者一致：只重置 Linear/LayerNorm，保留注意力 in_proj 的默认初始化。
        self.apply(init_weights)

    def forward(
        self, patch_tokens: torch.Tensor, cls_features: torch.Tensor
    ) -> torch.Tensor:
        # 广播后得到 [B, 6, 384]；不同图像的查询同时包含图像语义与成员差异。
        queries = self.bunch_embedding + cls_features.unsqueeze(1)
        # 查询先做自注意力，再以 patch 特征为 K/V 做交叉注意力；不使用因果掩码。
        panel_features = self.bunch_decoder(
            tgt=queries, memory=patch_tokens
        )  # [B, 6, 384]
        member_scores = self.heads(panel_features)  # [B, 6, 1]
        return member_scores.mean(dim=1)  # [B, 1]，仅对 panel 成员维求平均
