"""使用真实 LIVE-C 验证读取；临时损坏文件仅用于测试失败路径。"""

import ast
import os
import random
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image
from scipy import io
from torch.utils import data

from dataset import LiveChallengeDataset, split_livec_indices
from preprocessing import build_transform


ROOT = Path(__file__).resolve().parent / "data/ChallengeDB_release"


class DatasetTests(unittest.TestCase):
    def test_split_is_reproducible_disjoint_and_complete(self) -> None:
        random.seed(123)
        before = random.getstate()
        train, test = split_livec_indices(42)
        self.assertEqual((len(train), len(test)), (930, 232))
        self.assertFalse(set(train) & set(test))
        self.assertEqual(set(train) | set(test), set(range(1162)))
        self.assertEqual((train, test), split_livec_indices(42))
        self.assertNotEqual((train, test), split_livec_indices(43))
        self.assertEqual(before, random.getstate())
        with self.assertRaises(ValueError):
            split_livec_indices(None)

    def test_matches_author_samples_and_crops(self) -> None:
        author_path = Path(__file__).resolve().parents[1] / "DEIQT_author/IQA/iqa_dataset.py"
        tree = ast.parse(author_path.read_text(encoding="utf-8"))
        definition = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "LIVECDATASET"
        )
        namespace = {"data": data, "io": io, "np": np, "os": os, "Image": Image}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(author_path), "exec"), namespace)
        for is_train in (True, False):
            with self.subTest(is_train=is_train):
                indices = [1161, 0, 57]
                own = LiveChallengeDataset(ROOT, indices, is_train)
                author = namespace["LIVECDATASET"](
                    str(ROOT), indices, 10, build_transform("livec", is_train)
                )
                self.assertEqual(len(own), len(author))
                for index in (0, 9, 10, 29):
                    torch.manual_seed(31)
                    actual_image, actual_score = own[index]
                    torch.manual_seed(31)
                    expected_image, expected_score = author[index]
                    torch.testing.assert_close(actual_image, expected_image, rtol=0, atol=0)
                    self.assertEqual(actual_score.shape, (1,))
                    self.assertEqual(actual_score.item(), float(expected_score))

    def test_dataloader_batch_and_windows_worker(self) -> None:
        dataset = LiveChallengeDataset(ROOT, [0, 1], False)
        # 单个 worker 实测 Windows spawn 能导入并序列化 Dataset 和预处理。
        loader = data.DataLoader(dataset, batch_size=20, num_workers=1, shuffle=False)
        batches = list(loader)
        self.assertEqual(len(batches), 1)
        images, scores = batches[0]
        self.assertEqual(images.shape, (20, 3, 224, 224))
        self.assertEqual(scores.shape, (20, 1))
        self.assertTrue(torch.isfinite(images).all())
        self.assertEqual(scores.dtype, torch.float32)
        torch.testing.assert_close(scores[:10], dataset.targets[0].expand(10, 1))
        torch.testing.assert_close(scores[10:], dataset.targets[1].expand(10, 1))

    def test_invalid_arguments(self) -> None:
        for indices, patch_num in (([], 10), ([0, 0], 10), ([-1], 10), ([1162], 10), ([0], 0)):
            with self.subTest(indices=indices, patch_num=patch_num):
                with self.assertRaises(ValueError):
                    LiveChallengeDataset(ROOT, indices, False, patch_num)
        dataset = LiveChallengeDataset(ROOT, [0], False)
        for index in (-1, len(dataset)):
            with self.assertRaises(IndexError):
                dataset[index]

    def test_missing_corrupt_images_and_invalid_labels(self) -> None:
        reference = LiveChallengeDataset(ROOT, [0], False)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Data").mkdir()
            (root / "Images").mkdir()
            for name in ("AllImages_release.mat", "AllMOS_release.mat"):
                shutil.copyfile(ROOT / "Data" / name, root / "Data" / name)
            with self.assertRaisesRegex(FileNotFoundError, "图像不存在"):
                LiveChallengeDataset(root, [0], False)
            (root / "Images" / reference.image_paths[0].name).write_bytes(b"corrupt image")
            dataset = LiveChallengeDataset(root, [0], False)
            with self.assertRaisesRegex(OSError, "无法解码"):
                dataset[0]
            labels = io.loadmat(root / "Data/AllMOS_release.mat")["AllMOS_release"]
            labels[0, 7] = np.nan
            io.savemat(root / "Data/AllMOS_release.mat", {"AllMOS_release": labels})
            with self.assertRaisesRegex(ValueError, "MOS 标签"):
                LiveChallengeDataset(root, [0], False)
            io.savemat(root / "Data/AllMOS_release.mat", {"wrong_key": labels})
            with self.assertRaisesRegex(ValueError, "缺少"):
                LiveChallengeDataset(root, [0], False)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
