"""以合成图像验证预处理；这些图像不属于真实 IQA 数据集。"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torchvision import transforms

from preprocessing import build_transform


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # 只提取作者的纯预处理函数，避免导入其分布式训练和数据集依赖。
        author_path = (
            Path(__file__).resolve().parents[1] / "DEIQT_author/IQA/build.py"
        )
        tree = ast.parse(author_path.read_text(encoding="utf-8"))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "build_transform"
        )
        namespace = {"transforms": transforms}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(author_path), "exec"), namespace)
        cls.author_build_transform = staticmethod(namespace["build_transform"])
        generator = torch.Generator().manual_seed(123)
        pixels = torch.randint(0, 256, (320, 480, 3), dtype=torch.uint8, generator=generator)
        cls.image = Image.fromarray(pixels.numpy())

    def test_matches_author_for_each_dataset_and_mode(self) -> None:
        datasets = ("koniq", "livec", "live", "tid2013", "csiq", "kadid", "spaq", "livefb")
        for dataset in datasets:
            for is_train in (True, False):
                with self.subTest(dataset=dataset, is_train=is_train):
                    config = SimpleNamespace(
                        DATA=SimpleNamespace(DATASET=dataset, CROP_SIZE=(224, 224))
                    )
                    reference = self.author_build_transform(is_train, config)
                    own = build_transform(dataset, is_train)
                    torch.manual_seed(19)
                    expected = reference(self.image)
                    torch.manual_seed(19)
                    actual = own(self.image)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_normalization_and_output_shape(self) -> None:
        actual = build_transform("livec", False)(Image.new("RGB", (300, 300), "white"))
        self.assertEqual(tuple(actual.shape), (3, 224, 224))
        self.assertEqual(actual.dtype, torch.float32)
        expected = (1 - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
        torch.testing.assert_close(actual, expected[:, None, None].expand_as(actual))

    def test_evaluation_crops_are_random_and_seed_reproducible(self) -> None:
        transform = build_transform("livec", False)
        torch.manual_seed(3)
        first = transform(self.image)
        second = transform(self.image)
        self.assertFalse(torch.equal(first, second))
        torch.manual_seed(3)
        torch.testing.assert_close(first, transform(self.image), rtol=0, atol=0)

    def test_rejects_small_images_and_unknown_datasets(self) -> None:
        with self.assertRaises(ValueError):
            build_transform("livec", False)(Image.new("RGB", (100, 100)))
        with self.assertRaisesRegex(ValueError, "不支持的数据集"):
            build_transform("unknown", True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
