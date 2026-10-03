"""优化器与单轮训练验证；合成样本用于检验更新规则，不是 IQA 实验。"""

import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import DEIQT
from optimizer import build_optimizer
from lr_scheduler import WarmupStepLR
from train import train_one_epoch


class OptimizerTests(unittest.TestCase):
    def test_weight_decay_matches_author_and_excludes_frozen_parameters(self) -> None:
        model = DEIQT()
        model.head.heads.weight.requires_grad_(False)
        optimizer = build_optimizer(model)
        author_path = Path(__file__).resolve().parents[1] / "DEIQT_author/optimizer.py"
        namespace = {}
        exec(compile(author_path.read_text(encoding="utf-8"), str(author_path), "exec"), namespace)
        expected = namespace["set_weight_decay"](model, model.no_weight_decay())
        for actual_group, expected_group in zip(optimizer.param_groups, expected):
            self.assertEqual(
                {id(parameter) for parameter in actual_group["params"]},
                {id(parameter) for parameter in expected_group["params"]},
            )
        grouped = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertEqual(set(grouped), {id(p) for p in model.parameters() if p.requires_grad})
        no_decay = {id(parameter) for parameter in optimizer.param_groups[1]["params"]}
        self.assertIn(id(model.encoder.cls_token), no_decay)
        self.assertIn(id(model.encoder.pos_embed), no_decay)
        self.assertIn(id(model.encoder.blocks[0].gamma_1), no_decay)
        self.assertNotIn(id(model.head.bunch_embedding), no_decay)
        self.assertEqual([group["weight_decay"] for group in optimizer.param_groups], [0.05, 0.0])
        self.assertEqual(optimizer.defaults["lr"], 2e-4)
        self.assertEqual(optimizer.defaults["betas"], (0.9, 0.999))
        self.assertEqual(optimizer.defaults["eps"], 1e-8)


class TrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(self.model.weight)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

    def test_updates_parameters_and_weights_loss_by_sample_count(self) -> None:
        loader = DataLoader(TensorDataset(torch.ones(3, 1), torch.full((3, 1), 2.0)), batch_size=2)
        self.model.eval()
        result = train_one_epoch(self.model, loader, self.optimizer)
        # 两次梯度均为 -1；两批损失分别为 1.5 和 1.4，末批仅一个样本。
        self.assertAlmostEqual(self.model.weight.item(), 0.2, places=6)
        self.assertAlmostEqual(result.loss, (1.5 * 2 + 1.4) / 3, places=6)
        self.assertEqual((result.samples, result.batches, result.optimizer_steps, result.skipped_steps), (3, 2, 2, 0))
        self.assertTrue(self.model.training)
        self.assertIsNone(self.model.weight.grad)

    def test_clips_gradient_before_update(self) -> None:
        loader = DataLoader(TensorDataset(torch.ones(1, 1), torch.full((1, 1), 2.0)))
        train_one_epoch(self.model, loader, self.optimizer, clip_grad=0.25)
        self.assertAlmostEqual(self.model.weight.item(), 0.025, places=6)

    def test_rejects_invalid_targets_without_updating(self) -> None:
        for targets in (torch.ones(2), torch.full((2, 1), float("nan"))):
            with self.subTest(shape=targets.shape):
                loader = DataLoader(TensorDataset(torch.ones(2, 1), targets), batch_size=2)
                with self.assertRaises(ValueError):
                    train_one_epoch(self.model, loader, self.optimizer)
                self.assertEqual(self.model.weight.item(), 0)
                self.assertIsNone(self.model.weight.grad)

    def test_rejects_empty_loader_and_nonfinite_loss(self) -> None:
        empty = DataLoader(TensorDataset(torch.ones(1, 1), torch.ones(1, 1)), batch_size=2, drop_last=True)
        with self.assertRaisesRegex(ValueError, "为空"):
            train_one_epoch(self.model, empty, self.optimizer)
        invalid = DataLoader(TensorDataset(torch.full((1, 1), float("inf")), torch.ones(1, 1)))
        with self.assertRaises(FloatingPointError):
            train_one_epoch(self.model, invalid, self.optimizer)
        self.assertEqual(self.model.weight.item(), 0)

    def test_nonfinite_gradient_does_not_update_fp32_parameters(self) -> None:
        loader = DataLoader(TensorDataset(torch.ones(1, 1), torch.ones(1, 1)))
        handle = self.model.weight.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
        try:
            with self.assertRaises(RuntimeError):
                train_one_epoch(self.model, loader, self.optimizer)
        finally:
            handle.remove()
        self.assertEqual(self.model.weight.item(), 0)
        self.assertIsNone(self.model.weight.grad)

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA 验证混合精度")
    def test_amp_updates_and_counts_overflow_skips(self) -> None:
        model = self.model.cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loader = DataLoader(TensorDataset(torch.ones(1, 1), torch.full((1, 1), 2.0)))
        scaler = torch.amp.GradScaler("cuda", init_scale=128)
        scheduler = WarmupStepLR(optimizer, 1, warmup_epochs=0, decay_epochs=1)
        result = train_one_epoch(model, loader, optimizer, scaler=scaler, scheduler=scheduler)
        self.assertEqual((result.optimizer_steps, result.skipped_steps), (1, 0))
        self.assertAlmostEqual(model.weight.item(), 0.1, places=5)
        before = model.weight.detach().clone()
        previous_scale = scaler.get_scale()
        handle = model.weight.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
        try:
            result = train_one_epoch(model, loader, optimizer, scaler=scaler, scheduler=scheduler)
        finally:
            handle.remove()
        self.assertEqual((result.optimizer_steps, result.skipped_steps), (0, 1))
        torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
        self.assertLess(scaler.get_scale(), previous_scale)
        self.assertEqual(scheduler.completed_batches, 2)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.01)
        self.assertIsNone(model.weight.grad)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
