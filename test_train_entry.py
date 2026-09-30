"""训练入口集成测试：真实 LIVE-C 图像配小型测试模型，不代表 DEIQT 指标。"""

import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from torch import nn
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from checkpoint import save_checkpoint, save_epoch_checkpoints
from evaluation import EvaluationResult
from train import PROJECT_DIR, parse_args, run_training
from visualization import TensorBoardLogger


class SmallTestModel(nn.Module):
    """减少入口测试计算量；完整 DEIQT 另做一次真实 GPU 流程检查。"""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.head = nn.Linear(4, 1)

    def no_weight_decay(self) -> set[str]:
        return set()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(images.mean(dim=(2, 3))))


class EntryTests(unittest.TestCase):
    def test_pipeline_outputs_are_consistent_and_readable(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            args = parse_args(["--smoke-test", "--device", "cpu", "--output", str(output),
                               "--log-dir", str(Path(directory) / "tf-logs")])
            with patch("train.DEIQT", SmallTestModel), patch("train.load_pretrained_encoder"):
                run_training(args)
            expected_files = {"config.json", "split.json", "train.log", "metrics.csv",
                              "summary.json", "last.pth", "best.pth"}
            self.assertEqual({path.name for path in output.iterdir()}, expected_files)
            last = torch.load(output / "last.pth", map_location="cpu", weights_only=True)
            best = torch.load(output / "best.pth", map_location="cpu", weights_only=True)
            self.assertEqual(last["epoch"], 1)
            self.assertEqual(last["evaluation_status"], "completed")
            self.assertEqual(last["scheduler"]["completed_batches"], 4)
            self.assertEqual(last["history"][0]["samples"], 8)
            self.assertEqual(last["history"][0]["optimizer_steps"], 4)
            self.assertEqual(last["predictions"].shape, (3,))
            self.assertEqual(last["best"]["plcc"], best["metrics"]["plcc"])
            self.assertEqual(last["config"]["steps_per_epoch"], 4)
            self.assertTrue(last["config"]["smoke_test"])
            model = SmallTestModel()
            model.load_state_dict(best["model"], strict=True)
            split = json.loads((output / "split.json").read_text(encoding="utf-8"))
            self.assertEqual(len(split["train_indices"]), 4)
            self.assertEqual(len(split["test_indices"]), 3)
            self.assertFalse(set(split["train_indices"]) & set(split["test_indices"]))
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["completed_epochs"], 1)
            self.assertEqual(summary["best"], last["best"])
            self.assertEqual(len((output / "metrics.csv").read_text().splitlines()), 2)
            events = EventAccumulator(str(args.log_dir / output.name)).Reload()
            self.assertEqual([point.step for point in events.Scalars("train/running_loss")], [1, 4])
            self.assertAlmostEqual(events.Scalars("train/epoch_loss")[0].value, last["history"][0]["loss"], places=5)
            self.assertAlmostEqual(events.Scalars("test/plcc")[0].value, summary["best"]["plcc"], places=6)
            self.assertEqual(events.Scalars("best/epoch")[0].value, summary["best"]["epoch"])
            self.assertIn("experiment/config/text_summary", events.Tags()["tensors"])
            self.assertEqual(last["config"]["tensorboard_log_dir"], str(args.log_dir / output.name))
            original_config = (output / "config.json").read_bytes()
            with self.assertRaises(FileExistsError):
                run_training(args)
            self.assertEqual((output / "config.json").read_bytes(), original_config)

    def test_parser_rejects_invalid_configuration(self) -> None:
        for options in (["--batch-size", "0"], ["--epochs", "0"], ["--learning-rate", "nan"],
                        ["--seed", "-1"], ["--warmup-lr", "1"]):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(options)

    def test_log_directory_default_and_cli_override(self) -> None:
        self.assertEqual(parse_args([]).log_dir, PROJECT_DIR / "tf-logs")
        with TemporaryDirectory() as directory:
            self.assertEqual(parse_args(["--log-dir", directory]).log_dir, Path(directory).resolve())

    def test_existing_tensorboard_run_is_not_merged(self) -> None:
        with TemporaryDirectory() as directory:
            log_dir = Path(directory) / "events"
            visualizer = TensorBoardLogger(log_dir)
            visualizer.log_config({"name": "first"})
            visualizer.close()
            original = {path.name: path.read_bytes() for path in log_dir.iterdir()}
            with self.assertRaises(FileExistsError):
                TensorBoardLogger(log_dir)
            self.assertEqual({path.name: path.read_bytes() for path in log_dir.iterdir()}, original)

    def test_checkpoint_write_failure_preserves_previous_file(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "last.pth"
            save_checkpoint(path, {"epoch": 1, "value": torch.ones(2)})
            original = path.read_bytes()

            def fail_save(state: dict, file) -> None:
                file.write(b"incomplete")
                raise OSError("simulated disk error")

            with patch("checkpoint.torch.save", side_effect=fail_save):
                with self.assertRaises(OSError):
                    save_checkpoint(path, {"epoch": 2})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_evaluation_failure_preserves_trained_parameters(self) -> None:
        class ConstantTestModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = nn.Identity()
                self.score = nn.Parameter(torch.zeros(1))

            def no_weight_decay(self) -> set[str]:
                return set()

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return self.score.expand(images.shape[0], 1)

        with TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            args = parse_args(["--smoke-test", "--device", "cpu", "--output", str(output),
                               "--log-dir", str(Path(directory) / "tf-logs")])
            with patch("train.DEIQT", ConstantTestModel), patch("train.load_pretrained_encoder"):
                with self.assertRaisesRegex(ValueError, "常量"):
                    run_training(args)
            last = torch.load(output / "last.pth", weights_only=True)
            self.assertEqual(last["epoch"], 1)
            self.assertEqual(last["evaluation_status"], "pending")
            self.assertEqual(last["train_result"]["optimizer_steps"], 4)
            self.assertEqual(last["scheduler"]["completed_batches"], 4)
            self.assertGreater(last["model"]["score"].item(), 0)
            self.assertIsNone(last["metrics"])
            self.assertIsNone(last["best"])
            self.assertEqual(last["history"], [])
            self.assertNotIn("predictions", last)
            self.assertFalse((output / "best.pth").exists())
            events = EventAccumulator(str(args.log_dir / output.name)).Reload()
            self.assertEqual(len(events.Scalars("train/epoch_loss")), 1)
            self.assertNotIn("test/srcc", events.Tags()["scalars"])

    def test_best_write_failure_keeps_previous_best_and_latest_training(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            args = parse_args(["--smoke-test", "--device", "cpu", "--output", str(output),
                               "--log-dir", str(Path(directory) / "tf-logs")])
            args.epochs = 2
            # 仅为制造“第二轮改善”的保存场景，这些固定指标不代表真实模型效果。
            results = [EvaluationResult(score, score, torch.arange(3), torch.arange(3))
                       for score in (0.2, 0.4)]
            original_save = torch.save

            def fail_second_best(state: dict, file) -> None:
                if state["epoch"] == 2 and ".best.pth." in str(file.name):
                    file.write(b"incomplete")
                    raise OSError("simulated best write error")
                original_save(state, file)

            with patch("train.DEIQT", SmallTestModel), patch("train.load_pretrained_encoder"), \
                    patch("train.evaluate", side_effect=results), \
                    patch("checkpoint.torch.save", side_effect=fail_second_best):
                with self.assertRaisesRegex(OSError, "simulated best write error"):
                    run_training(args)
            last = torch.load(output / "last.pth", weights_only=True)
            best = torch.load(output / "best.pth", weights_only=True)
            self.assertEqual(last["epoch"], 2)
            self.assertEqual(last["evaluation_status"], "pending")
            self.assertEqual(last["scheduler"]["completed_batches"], 8)
            self.assertEqual(last["best"]["epoch"], best["epoch"])
            self.assertEqual(best["epoch"], 1)
            self.assertEqual(last["best"]["plcc"], best["metrics"]["plcc"])
            self.assertEqual(len(last["history"]), 1)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["best"], last["best"])
            self.assertFalse(list(output.glob(".checkpoint-*")))
            events = EventAccumulator(str(args.log_dir / output.name)).Reload()
            self.assertEqual([point.step for point in events.Scalars("train/running_loss")], [1, 4, 5, 8])
            self.assertEqual(len(events.Scalars("train/epoch_loss")), 2)
            self.assertEqual(len(events.Scalars("test/plcc")), 1)

    def test_group_replace_failure_rolls_back_all_published_files(self) -> None:
        for has_previous in (False, True):
            with self.subTest(has_previous=has_previous), TemporaryDirectory() as directory:
                output = Path(directory)
                previous = {}
                if has_previous:
                    for name in ("last.pth", "best.pth"):
                        save_checkpoint(output / name, {"epoch": 1})
                        previous[name] = (output / name).read_bytes()
                original_replace = Path.replace

                def fail_best_replace(source: Path, target: Path) -> Path:
                    if source.name == "best.pth" and source.parent.name.startswith(".checkpoint-"):
                        raise OSError("simulated replace error")
                    return original_replace(source, target)

                with patch.object(Path, "replace", fail_best_replace):
                    with self.assertRaisesRegex(OSError, "simulated replace error"):
                        save_epoch_checkpoints(output, {"epoch": 2}, {"epoch": 2})
                self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, previous)

    def test_epoch_without_improvement_preserves_best(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory)
            save_epoch_checkpoints(output, {"epoch": 1, "best": {"epoch": 1}}, {"epoch": 1})
            previous_best = (output / "best.pth").read_bytes()
            save_epoch_checkpoints(output, {"epoch": 2, "best": {"epoch": 1}}, None)
            self.assertEqual((output / "best.pth").read_bytes(), previous_best)
            self.assertEqual(torch.load(output / "last.pth", weights_only=True)["epoch"], 2)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
