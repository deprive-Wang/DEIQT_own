"""评估逻辑的合成数据测试；固定分数仅用于验证算法，不是模型实验结果。"""

import math
import unittest

import torch
from torch import nn
from torch.utils.data import Dataset

from evaluation import evaluate, summarize_predictions


class SyntheticCropDataset(Dataset):
    """三张合成图像各有两个裁剪，像素值直接编码预设预测。"""

    def __init__(self) -> None:
        self.indices = (0, 1, 2)
        self.patch_num = 2
        self.is_train = False
        self.scores = (2.0, 0.0, 4.0, 2.0, 6.0, 4.0)

    def __len__(self) -> int:
        return len(self.scores)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.full((3, 2, 2), self.scores[index]), torch.tensor([index // 2 + 1.0])


class PixelMeanModel(nn.Module):
    """合成测试模型，记录评估是否关闭梯度并切换到 eval 模式。"""

    def __init__(self, wrong_shape: bool = False) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.wrong_shape = wrong_shape
        self.observations: list[tuple[bool, bool, int]] = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.observations.append((self.training, torch.is_grad_enabled(), len(images)))
        scores = images.mean(dim=(1, 2, 3)) * self.scale
        return scores if self.wrong_shape else scores.unsqueeze(1)


class EvaluationTests(unittest.TestCase):
    def test_averages_crops_before_computing_metrics(self) -> None:
        result = summarize_predictions(
            torch.tensor([2., 0., 4., 2., 6., 4.]),
            torch.tensor([1., 1., 2., 2., 3., 3.]),
            patch_num=2,
        )
        torch.testing.assert_close(result.predictions, torch.tensor([1., 3., 5.]))
        torch.testing.assert_close(result.targets, torch.tensor([1., 2., 3.]))
        self.assertAlmostEqual(result.srcc, 1.0)
        self.assertAlmostEqual(result.plcc, 1.0)

    def test_negative_correlation_and_tied_ranks(self) -> None:
        result = summarize_predictions(torch.tensor([3., 2., 1.]), torch.arange(1., 4.), 1)
        self.assertAlmostEqual(result.srcc, -1.0)
        self.assertAlmostEqual(result.plcc, -1.0)
        tied = summarize_predictions(torch.tensor([1., 1., 2., 3.]), torch.arange(1., 5.), 1)
        self.assertAlmostEqual(tied.srcc, 3 / math.sqrt(10))

    def test_plcc_uses_raw_scores_without_curve_fitting(self) -> None:
        result = summarize_predictions(torch.tensor([1., 4., 9.]), torch.arange(1., 4.), 1)
        self.assertAlmostEqual(result.srcc, 1.0)
        self.assertAlmostEqual(result.plcc, 4 * math.sqrt(3) / 7)

    def test_invalid_scores_and_groups_are_rejected(self) -> None:
        cases = [
            ([1., 2.], [1., 2.], 0),
            ([], [], 1),
            ([1.], [2.], 1),
            ([1., 2.], [1.], 1),
            ([1., 2., 3.], [1., 1., 2.], 2),
            ([1., 2., 3., 4.], [1., 2., 3., 3.], 2),
            ([1., 1.], [1., 2.], 1),
            ([1., 2.], [3., 3.], 1),
            ([float("nan"), 2.], [1., 2.], 1),
            ([1., 2.], [1., float("inf")], 1),
        ]
        for predictions, targets, patch_num in cases:
            with self.subTest(predictions=predictions, targets=targets, patch_num=patch_num):
                with self.assertRaises(ValueError):
                    summarize_predictions(torch.tensor(predictions), torch.tensor(targets), patch_num)
        with self.assertRaisesRegex(ValueError, "形状"):
            summarize_predictions(torch.ones(2, 2), torch.arange(4.), 1)

    def test_evaluate_preserves_order_tail_batch_and_mode(self) -> None:
        dataset = SyntheticCropDataset()
        model = PixelMeanModel().train()
        # batch_size=5 会切开第三张图像的裁剪组，且最后一批仅一个样本。
        result = evaluate(model, dataset, batch_size=5)
        torch.testing.assert_close(result.predictions, torch.tensor([1., 3., 5.]))
        self.assertEqual(model.observations, [(False, False, 5), (False, False, 1)])
        self.assertTrue(model.training)
        self.assertIsNone(model.scale.grad)
        model.eval()
        evaluate(model, dataset, batch_size=4)
        self.assertFalse(model.training)

    def test_evaluate_rejects_training_data_and_restores_mode_on_error(self) -> None:
        dataset = SyntheticCropDataset()
        dataset.is_train = True
        with self.assertRaisesRegex(ValueError, "is_train=False"):
            evaluate(PixelMeanModel(), dataset)
        dataset.is_train = False
        model = PixelMeanModel(wrong_shape=True).train()
        with self.assertRaisesRegex(ValueError, "模型输出"):
            evaluate(model, dataset)
        self.assertTrue(model.training)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
