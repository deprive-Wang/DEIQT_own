"""ViT-S 图像编码器，包括图像分块和 Transformer 编码层。"""

import torch
from torch import nn

from initialization import init_weights


class PatchEmbed(nn.Module):
    """将 224×224 RGB 图像转换为 196 个 384 维 patch token。"""

    def __init__(self) -> None:
        super().__init__()
        # 卷积核和步长均为 16，对应互不重叠的 16×16 图像块。
        self.proj = nn.Conv2d(3, 384, kernel_size=16, stride=16)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 224, 224):
            raise ValueError("输入图像形状必须为 [B, 3, 224, 224]")

        patch_features = self.proj(images)  # [B, 384, 14, 14]
        # 将空间网格展平，并把 token 维放在通道维之前。
        return patch_features.flatten(2).transpose(1, 2)  # [B, 196, 384]


class MultiHeadSelfAttention(nn.Module):
    """对全部 token 计算六头自注意力。"""

    def __init__(self) -> None:
        super().__init__()
        self.num_heads = 6
        self.scale = (384 // self.num_heads) ** -0.5
        self.qkv = nn.Linear(384, 384 * 3, bias=True)
        self.proj = nn.Linear(384, 384)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, embed_dim = tokens.shape
        # [B, N, 3, 6, 64] -> [3, B, 6, N, 64]
        qkv = self.qkv(tokens).reshape(
            batch_size, token_count, 3, self.num_heads, embed_dim // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        # 缩放后沿最后一维归一化，即每个 query 对所有 key 的注意力权重。
        attention = ((query * self.scale) @ key.transpose(-2, -1)).softmax(dim=-1)
        attended = (attention @ value).transpose(1, 2).reshape(
            batch_size, token_count, embed_dim
        )
        return self.proj(attended)


class FeedForward(nn.Module):
    """ViT Encoder 中的两层前馈网络。"""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(384, 1536)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(1536, 384)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(tokens)))


class EncoderBlock(nn.Module):
    """带 LayerScale 的单层 ViT Encoder Block。"""

    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(384, eps=1e-6)
        self.attn = MultiHeadSelfAttention()
        self.norm2 = nn.LayerNorm(384, eps=1e-6)
        self.mlp = FeedForward()
        # 作者实验的 dropout 和 drop path 均为 0；保留其 LayerScale 残差形式。
        self.gamma_1 = nn.Parameter(torch.full((384,), 1e-4))
        self.gamma_2 = nn.Parameter(torch.full((384,), 1e-4))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.gamma_1 * self.attn(self.norm1(tokens))
        return tokens + self.gamma_2 * self.mlp(self.norm2(tokens))


class ViTEncoder(nn.Module):
    """从图像提取 CLS 特征和全部 patch 特征的 ViT-S Encoder。"""

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.cls_token = nn.Parameter(torch.randn(1, 1, 384))
        self.pos_embed = nn.Parameter(torch.zeros(1, 196, 384))
        self.blocks = nn.ModuleList(EncoderBlock() for _ in range(12))
        self.norm = nn.LayerNorm(384, eps=1e-6)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(init_weights)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patch_tokens = self.patch_embed(images)  # [B, 196, 384]
        # DeiT III 的位置编码只对应图像块，CLS token 在相加之后拼接。
        patch_tokens = patch_tokens + self.pos_embed
        cls_token = self.cls_token.expand(images.shape[0], -1, -1)
        tokens = torch.cat((cls_token, patch_tokens), dim=1)  # [B, 197, 384]

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        return tokens[:, 0], tokens[:, 1:]  # [B, 384], [B, 196, 384]
