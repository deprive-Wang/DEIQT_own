"""单设备 IQA 评估：先汇总裁剪，再计算图像级 SRCC 和 PLCC。"""

from dataclasses import dataclass

import torch
from scipy.stats import pearsonr, spearmanr
from torch import nn
from torch.utils.data import DataLoader

from dataset import LiveChallengeDataset


@dataclass
class EvaluationResult:
    """图像级分数保存在 CPU，顺序对应数据集的 indices。"""

    srcc: float
    plcc: float
    predictions: torch.Tensor
    targets: torch.Tensor


def summarize_predictions(
    predictions: torch.Tensor, targets: torch.Tensor, patch_num: int
) -> EvaluationResult:
    """输入顺序排列的裁剪分数 [N] 或 [N, 1]，输出图像级结果。

    调用方必须确保每张图像的 patch_num 个样本连续排列；不能通过 MOS
    是否相等判断图像身份，因为不同图像也可能具有相同的 MOS。
    """
    if type(patch_num) is not int or patch_num < 1:
        raise ValueError("patch_num 必须为正整数")
    for name, values in (("预测", predictions), ("标签", targets)):
        if values.ndim != 1 and not (values.ndim == 2 and values.shape[1] == 1):
            raise ValueError(f"{name}形状必须为 [N] 或 [N, 1]")
        if values.is_complex() or not torch.isfinite(values).all():
            raise ValueError(f"{name}必须为有限实数，不能含 NaN 或无穷值")
    if predictions.numel() != targets.numel():
        raise ValueError("预测与标签数量不一致")
    if predictions.numel() % patch_num != 0:
        raise ValueError("裁剪数量不能被 patch_num 整除，可能遗漏了最后一个 batch")
    if predictions.numel() // patch_num < 2:
        raise ValueError("计算相关系数至少需要两张图像")

    # 与作者一致，在所有 batch 拼接完成后分组，允许同一图像跨越 batch 边界。
    grouped_predictions = predictions.detach().cpu().float().reshape(-1, patch_num)
    grouped_targets = targets.detach().cpu().float().reshape(-1, patch_num)
    if not torch.all(grouped_targets == grouped_targets[:, :1]):
        raise ValueError("同一裁剪组的标签不一致，请检查样本顺序与 patch_num")
    image_predictions = grouped_predictions.mean(dim=1)
    image_targets = grouped_targets.mean(dim=1)
    if not torch.isfinite(image_predictions).all() or not torch.isfinite(image_targets).all():
        raise ValueError("裁剪均值出现非有限值")
    if torch.all(image_predictions == image_predictions[0]) or torch.all(
        image_targets == image_targets[0]
    ):
        # 常量序列的相关系数未定义，不能把它当成正常的 0 分。
        raise ValueError("图像级预测或标签为常量，相关系数未定义")

    # 使用已有的 SciPy；双精度计算相关系数，SRCC 的并列值按平均秩处理。
    predicted = image_predictions.double().numpy()
    expected = image_targets.double().numpy()
    srcc = float(spearmanr(predicted, expected).statistic)
    plcc = float(pearsonr(predicted, expected).statistic)
    # 保留相关系数正负号；作者直接计算 PLCC，没有额外的非线性拟合。
    return EvaluationResult(srcc, plcc, image_predictions, image_targets)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: LiveChallengeDataset,
    batch_size: int = 16,
    num_workers: int = 0,
) -> EvaluationResult:
    """评估模型所在设备上的 LIVE-C 数据，并恢复调用前的 train/eval 模式。

    内部固定顺序采样且保留末尾 batch，保证裁剪分组正确。裁剪仍有随机性，
    重复对比时由调用方固定随机种子。此处使用 FP32 推理，不启用 AMP。
    Windows 使用多 worker 时，调用入口需放在 if __name__ == '__main__' 下。
    """
    if dataset.is_train:
        raise ValueError("评估必须使用 is_train=False 的数据集")
    if len(dataset.indices) < 2:
        raise ValueError("评估至少需要两张图像")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size 必须为正整数")
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers 必须为非负整数")
    device = next(model.parameters()).device
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    predictions, targets = [], []
    was_training = model.training
    model.eval()
    try:
        for images, batch_targets in loader:
            outputs = model(images.to(device, non_blocking=True))
            if outputs.shape != batch_targets.shape:
                raise ValueError("模型输出与标签均应为 [B, 1]")
            predictions.append(outputs.detach().cpu())
            targets.append(batch_targets)
        return summarize_predictions(
            torch.cat(predictions), torch.cat(targets), dataset.patch_num
        )
    finally:
        model.train(was_training)
