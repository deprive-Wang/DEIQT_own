"""用固定边界和小型训练验证学习率时间轴，不运行正式 IQA 训练。"""

import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lr_scheduler import WarmupStepLR
from train import train_one_epoch


class RecordingSGD(torch.optim.SGD):
    """记录优化器实际更新时使用的学习率，而非仅检查调度器返回值。"""

    def __init__(self, model: nn.Module) -> None:
        super().__init__(model.parameters(), lr=0.1)
        self.used_lrs: list[float] = []

    def step(self, closure=None):
        self.used_lrs.append(self.param_groups[0]["lr"])
        return super().step(closure)


class SchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = nn.Linear(1, 1, bias=False)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=2e-4)

    def test_livec_warmup_and_decay_boundaries(self) -> None:
        scheduler = WarmupStepLR(self.optimizer, steps_per_epoch=2)
        self.assertAlmostEqual(self.optimizer.param_groups[0]["lr"], 2e-7)
        expected_after_batch = {
            0: 2e-7,
            3: 0.0001001,
            5: 0.0001667,
            6: 2e-5,
            12: 2e-6,
            18: 2e-7,
            24: 2e-8,
        }
        for index in range(25):
            scheduler.step()
            if index in expected_after_batch:
                with self.subTest(index=index):
                    self.assertAlmostEqual(self.optimizer.param_groups[0]["lr"], expected_after_batch[index], places=12)
        self.assertEqual(scheduler.completed_batches, 25)

    def test_actual_training_lrs_continue_across_epochs(self) -> None:
        nn.init.zeros_(self.model.weight)
        optimizer = RecordingSGD(self.model)
        scheduler = WarmupStepLR(optimizer, 2, warmup_epochs=1, decay_epochs=2, warmup_lr=0.01)
        loader = DataLoader(TensorDataset(torch.ones(2, 1), torch.full((2, 1), 100.0)), batch_size=1)
        for _ in range(3):
            result = train_one_epoch(self.model, loader, optimizer, scheduler=scheduler)
            self.assertEqual(result.optimizer_steps, 2)
        for actual, expected in zip(optimizer.used_lrs, [0.01, 0.01, 0.055, 0.1, 0.1, 0.01]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(len(optimizer.used_lrs), 6)
        self.assertAlmostEqual(self.model.weight.item(), 0.285, places=6)
        self.assertEqual(scheduler.completed_batches, 6)

    def test_resume_matches_uninterrupted_schedule(self) -> None:
        scheduler = WarmupStepLR(self.optimizer, 2)
        for _ in range(5):
            scheduler.step()
        state = scheduler.state_dict()
        restored_optimizer = torch.optim.SGD(nn.Linear(1, 1).parameters(), lr=2e-4)
        restored = WarmupStepLR(restored_optimizer, 2)
        restored.load_state_dict(state)
        self.assertEqual(self.optimizer.param_groups[0]["lr"], restored_optimizer.param_groups[0]["lr"])
        for _ in range(15):
            scheduler.step()
            restored.step()
            self.assertEqual(scheduler.state_dict(), restored.state_dict())
            self.assertEqual(self.optimizer.param_groups[0]["lr"], restored_optimizer.param_groups[0]["lr"])

    def test_invalid_restore_does_not_change_progress_or_lr(self) -> None:
        scheduler = WarmupStepLR(self.optimizer, 2)
        scheduler.step()
        before = scheduler.state_dict()
        previous_lr = self.optimizer.param_groups[0]["lr"]
        for name, value in (("steps_per_epoch", 3), ("completed_batches", -1), ("base_lrs", [0.1])):
            with self.subTest(name=name):
                invalid = dict(before)
                invalid[name] = value
                with self.assertRaises(ValueError):
                    scheduler.load_state_dict(invalid)
                self.assertEqual(scheduler.state_dict(), before)
                self.assertEqual(self.optimizer.param_groups[0]["lr"], previous_lr)

    def test_no_warmup_and_multiple_parameter_groups(self) -> None:
        optimizer = torch.optim.SGD([
            {"params": [nn.Parameter(torch.ones(1))], "lr": 0.1},
            {"params": [nn.Parameter(torch.ones(1))], "lr": 0.2},
        ])
        scheduler = WarmupStepLR(optimizer, 2, warmup_epochs=0, decay_epochs=1)
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [0.1, 0.2])
        for _ in range(3):
            scheduler.step()
        for actual, expected in zip(optimizer.param_groups, [0.01, 0.02]):
            self.assertAlmostEqual(actual["lr"], expected)

    def test_rejects_invalid_configuration_and_loader(self) -> None:
        for arguments in ({"steps_per_epoch": 0}, {"warmup_epochs": -1}, {"decay_epochs": 0}, {"decay_rate": 0}, {"warmup_lr": float("nan")}):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    WarmupStepLR(self.optimizer, **({"steps_per_epoch": 2} | arguments))
        scheduler = WarmupStepLR(self.optimizer, 2)
        loader = DataLoader(TensorDataset(torch.ones(1, 1), torch.ones(1, 1)))
        with self.assertRaisesRegex(ValueError, "batch 数"):
            train_one_epoch(self.model, loader, self.optimizer, scheduler=scheduler)
        other_optimizer = torch.optim.SGD(self.model.parameters(), lr=2e-4)
        with self.assertRaisesRegex(ValueError, "同一个 optimizer"):
            train_one_epoch(self.model, loader, other_optimizer, scheduler=scheduler)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
