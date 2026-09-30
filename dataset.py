"""LIVE Challenge 数据读取、按图像划分与重复随机裁剪采样。"""

import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset

from preprocessing import build_transform


def split_livec_indices(seed: int) -> tuple[list[int], list[int]]:
    """按作者的 80%/20% 比例划分 1,162 张正式图像，返回 930/232 个索引。

    索引对应排除前七张练习图后的 MAT 顺序。先划分图像，再重复裁剪，
    避免同一张图像的裁剪进入不同集合。显式种子使本地划分可复现；
    与作者作数值对比时，双方仍须使用同一份索引。
    """
    if type(seed) is not int:
        raise ValueError("seed 必须为整数")
    indices = list(range(1162))
    random.Random(seed).shuffle(indices)
    boundary = round(0.8 * len(indices))
    return indices[:boundary], indices[boundary:]


class LiveChallengeDataset(Dataset):
    """每次返回一块图像 [3, 224, 224] 和原始 MOS 标签 [1]。

    每张图像在逻辑上连续出现 patch_num 次，每次读取重新随机裁剪。
    顺序评估时，同一图像的预测可按 patch_num 分组；训练可打乱样本。
    """

    def __init__(
        self,
        root: str | Path,
        indices: Sequence[int],
        is_train: bool,
        patch_num: int = 10,
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"LIVE-C 根目录不存在：{self.root}")
        if type(patch_num) is not int or patch_num < 1:
            raise ValueError("patch_num 必须为正整数")
        self.indices = tuple(indices)
        if not self.indices or any(
            type(index) is not int or not 0 <= index < 1162
            for index in self.indices
        ):
            raise ValueError("indices 必须是非空整数序列，索引范围为 0 到 1161")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("indices 不得重复；重复裁剪由 patch_num 控制")
        self.patch_num = patch_num
        self.is_train = is_train
        self.transform = build_transform("livec", is_train)

        names_data = loadmat(self.root / "Data/AllImages_release.mat")
        scores_data = loadmat(self.root / "Data/AllMOS_release.mat")
        if "AllImages_release" not in names_data or "AllMOS_release" not in scores_data:
            raise ValueError("MAT 文件缺少 AllImages_release 或 AllMOS_release 字段")
        names = names_data["AllImages_release"]
        scores = scores_data["AllMOS_release"]
        if names.shape != (1169, 1) or scores.shape != (1, 1169):
            raise ValueError("LIVE-C 原始标签应含 1169 项：图像名 [1169, 1]、MOS [1, 1169]")

        # 前七项用于人类受试者熟悉评分，不是机器学习的训练集合。
        image_names = []
        for cell in names[7:, 0]:
            if (
                not isinstance(cell, np.ndarray)
                or cell.size != 1
                or not isinstance(cell.flat[0], (str, np.str_))
            ):
                raise ValueError("AllImages_release 中包含无效图像名称")
            name = str(cell.flat[0])
            if not name or Path(name).name != name or "/" in name or "\\" in name:
                raise ValueError(f"图像名称应为单个文件名：{name!r}")
            image_names.append(name)
        if len(set(image_names)) != 1162:
            raise ValueError("正式图像名称存在重复")
        scores = scores[0, 7:].astype(np.float32)
        if not np.isfinite(scores).all():
            raise ValueError("MOS 标签包含 NaN 或无穷值")

        self.image_paths = tuple(
            self.root / "Images" / image_names[index] for index in self.indices
        )
        for path in self.image_paths:
            if not path.is_file():
                raise FileNotFoundError(f"LIVE-C 图像不存在：{path}")
        # 标签保留原始 MOS 尺度；[1] 经 DataLoader 拼成 [B, 1]，与模型输出一致。
        self.targets = torch.tensor(scores[list(self.indices)]).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.indices) * self.patch_num

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(f"裁剪样本索引越界：{index}")
        image_index = index // self.patch_num
        path = self.image_paths[image_index]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
        except OSError as error:
            # 作者在读取失败时替换成随机图；这里明确报错，避免训练混入假样本。
            raise OSError(f"无法解码 LIVE-C 图像：{path}") from error
        return self.transform(image), self.targets[image_index].clone()
