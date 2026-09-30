"""DeiT III 预训练文件的读取、校验和 Encoder 权重加载。"""

from collections.abc import Mapping
from pathlib import Path

import torch

from encoder import ViTEncoder


def load_pretrained_encoder(
    encoder: ViTEncoder, checkpoint_path: str | Path
) -> None:
    """加载 DeiT III checkpoint['model']，仅初始化 Encoder。

    相对路径以运行时的工作目录为基准。只忽略原 ImageNet 分类头
    head.weight/head.bias；Encoder 参数必须完整匹配，加载后仍可训练。
    建议在模型迁移到 GPU、创建优化器之前调用。
    """
    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"预训练权重文件不存在：{checkpoint_path}")

    # 先在 CPU 读取，避免占用 GPU 显存；不允许反序列化任意 Python 对象。
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if not isinstance(checkpoint, Mapping) or not isinstance(
        checkpoint.get("model"), Mapping
    ):
        raise ValueError("预训练文件必须包含字典形式的 checkpoint['model']")
    state_dict = checkpoint["model"]
    if not all(isinstance(key, str) for key in state_dict):
        raise ValueError("预训练参数名称必须为字符串")

    encoder_weights = {
        key: value for key, value in state_dict.items()
        if key not in {"head.weight", "head.bias"}
    }
    expected = encoder.state_dict()
    missing = sorted(expected.keys() - encoder_weights.keys())
    unexpected = sorted(encoder_weights.keys() - expected.keys())
    if missing or unexpected:
        raise ValueError(
            f"Encoder 参数名称不匹配；缺少：{missing}；多余：{unexpected}"
        )

    # load_state_dict 遇到错误时可能已复制部分参数，因此在写入前检查全部形状。
    for name, value in encoder_weights.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"预训练参数 {name} 必须为 Tensor")
        if value.shape != expected[name].shape:
            raise ValueError(
                f"预训练参数 {name} 形状不匹配："
                f"收到 {tuple(value.shape)}，预期 {tuple(expected[name].shape)}"
            )

    encoder.load_state_dict(encoder_weights, strict=True)
