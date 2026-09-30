"""模型结构的短时验证；随机输入仅用于检查计算，不代表真实 IQA 实验。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from decoder import AttentionPanelDecoder
from model import DEIQT
from pretrained import load_pretrained_encoder


class ModelTests(unittest.TestCase):
    def test_decoder_keeps_images_independent(self) -> None:
        """合并 batch 不应让不同图像的查询或注意力相互混合。"""
        torch.manual_seed(0)
        decoder = AttentionPanelDecoder().eval()
        patches = torch.randn(2, 196, 384)
        cls_features = torch.randn(2, 384)
        with torch.no_grad():
            batched = decoder(patches, cls_features)
            separate = torch.cat([
                decoder(patches[i:i + 1], cls_features[i:i + 1])
                for i in range(2)
            ])
        self.assertEqual(tuple(batched.shape), (2, 1))
        torch.testing.assert_close(batched, separate, rtol=1e-4, atol=1e-6)

    def test_complete_model_backpropagates(self) -> None:
        """评分损失应能传回 panel、Decoder 和 Encoder 的全部参数。"""
        torch.manual_seed(0)
        model = DEIQT().train()
        scores = model(torch.randn(1, 3, 224, 224))
        self.assertEqual(tuple(scores.shape), (1, 1))
        self.assertTrue(torch.isfinite(scores).all().item())
        scores.sum().backward()

        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
        # 梯度非空之外，确认质量分数确实依赖可学习查询和输入编码。
        self.assertGreater(model.head.bunch_embedding.grad.abs().sum().item(), 0)
        self.assertGreater(
            model.encoder.patch_embed.proj.weight.grad.abs().sum().item(), 0
        )

    def test_rejects_wrong_image_shape(self) -> None:
        model = DEIQT()
        with self.assertRaisesRegex(ValueError, "输入图像形状"):
            model(torch.randn(1, 3, 256, 256))


class PretrainedWeightsTests(unittest.TestCase):
    """使用本地真实权重验证加载，并用临时文件检查失败路径。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.checkpoint_path = (
            Path(__file__).resolve().parent / "weights/deit_3_small_224_1k.pth"
        )
        cls.weights = torch.load(
            cls.checkpoint_path, map_location="cpu", weights_only=True
        )["model"]

    def test_loads_encoder_without_changing_decoder(self) -> None:
        model = DEIQT()
        head_before = {
            name: value.clone() for name, value in model.head.state_dict().items()
        }
        load_pretrained_encoder(model.encoder, self.checkpoint_path)
        for name, value in model.encoder.state_dict().items():
            torch.testing.assert_close(value, self.weights[name], rtol=0, atol=0)
        for name, value in model.head.state_dict().items():
            torch.testing.assert_close(value, head_before[name], rtol=0, atol=0)
        self.assertTrue(all(p.requires_grad for p in model.encoder.parameters()))
        with torch.no_grad():
            scores = model.eval()(torch.randn(1, 3, 224, 224))
        self.assertEqual(tuple(scores.shape), (1, 1))
        self.assertTrue(torch.isfinite(scores).all().item())

    def test_missing_file(self) -> None:
        model = DEIQT()
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "权重文件不存在"):
                load_pretrained_encoder(model.encoder, Path(directory) / "missing.pth")

    def test_invalid_checkpoint_structure(self) -> None:
        model = DEIQT()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.pth"
            for checkpoint in ({}, {"model": []}):
                with self.subTest(checkpoint=checkpoint):
                    torch.save(checkpoint, path)
                    with self.assertRaisesRegex(ValueError, "checkpoint"):
                        load_pretrained_encoder(model.encoder, path)

    def test_invalid_parameters_leave_encoder_unchanged(self) -> None:
        model = DEIQT()
        before = {
            name: value.clone() for name, value in model.encoder.state_dict().items()
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.pth"
            for problem in ("missing", "unexpected", "shape", "type"):
                with self.subTest(problem=problem):
                    weights = dict(self.weights)
                    if problem == "missing":
                        del weights["norm.bias"]
                    elif problem == "unexpected":
                        weights["head.extra"] = torch.zeros(1)
                    elif problem == "shape":
                        weights["norm.bias"] = torch.zeros(385)
                    else:
                        weights["norm.bias"] = "不是 Tensor"
                    torch.save({"model": weights}, path)
                    with self.assertRaises(ValueError):
                        load_pretrained_encoder(model.encoder, path)
                    for name, value in model.encoder.state_dict().items():
                        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
