"""重复实验的真实训练入口与汇总边界测试。"""

import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from repeat import parse_repeat_args, run_repeats
from test_train_entry import SmallTestModel


class RepeatTests(unittest.TestCase):
    def test_two_independent_smoke_runs_and_summary(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "repeat"
            args, runs = parse_repeat_args([
                "--runs", "2", "--smoke-test", "--device", "cpu",
                "--seed", "42", "--output", str(output),
                "--log-dir", str(Path(directory) / "tf-logs"),
            ])
            with patch("train.DEIQT", SmallTestModel), patch("train.load_pretrained_encoder"):
                run_repeats(args, runs)
            summary = json.loads((output / "repeat_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["completed_runs"], 2)
            self.assertEqual(summary["seeds"], [42, 43])
            self.assertEqual([row["best_epoch"] for row in summary["runs"]], [1, 1])
            self.assertEqual({row["seed"] for row in summary["runs"]}, {42, 43})
            for metric in ("srcc", "plcc"):
                values = [row[metric] for row in summary["runs"]]
                self.assertAlmostEqual(summary["median"][metric], sum(values) / 2)
            split_one = json.loads((output / "run_01/split.json").read_text(encoding="utf-8"))
            split_two = json.loads((output / "run_02/split.json").read_text(encoding="utf-8"))
            self.assertNotEqual(split_one["train_indices"], split_two["train_indices"])
            self.assertNotEqual(split_one["test_indices"], split_two["test_indices"])
            experiment_logs = args.log_dir / output.name
            for run in ("run_01", "run_02"):
                events = EventAccumulator(str(experiment_logs / run)).Reload()
                self.assertEqual(len(events.Scalars("test/plcc")), 1)
            aggregate = EventAccumulator(str(experiment_logs / "summary")).Reload()
            self.assertEqual([point.step for point in aggregate.Scalars("repeat/best_plcc")], [1, 2])
            self.assertAlmostEqual(aggregate.Scalars("repeat/median_srcc")[0].value,
                                   summary["median"]["srcc"], places=6)
            self.assertEqual(summary["tensorboard_log_dir"], str(experiment_logs))
            with self.assertRaises(FileExistsError):
                run_repeats(args, runs)

    def test_failed_run_is_not_counted_in_median(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "repeat"
            args, runs = parse_repeat_args([
                "--runs", "3", "--smoke-test", "--device", "cpu", "--output", str(output),
                "--log-dir", str(Path(directory) / "tf-logs"),
            ])

            def train_or_fail(run_args) -> None:
                if run_args.seed == args.seed + 1:
                    raise OSError("simulated second run failure")
                run_output = run_args.output
                run_output.mkdir()
                (run_output / "best.pth").write_bytes(b"test-only")
                (run_output / "split.json").write_text(json.dumps({"seed": run_args.seed}))
                (run_output / "summary.json").write_text(json.dumps({
                    "completed_epochs": run_args.epochs,
                    "best": {"epoch": 1, "srcc": 0.5, "plcc": 0.6},
                }))

            with patch("repeat.run_training", side_effect=train_or_fail):
                with self.assertRaisesRegex(OSError, "second run failure"):
                    run_repeats(args, runs)
            summary = json.loads((output / "repeat_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "running")
            self.assertEqual(summary["completed_runs"], 1)
            self.assertIsNone(summary["median"])
            events = EventAccumulator(str(args.log_dir / output.name / "summary")).Reload()
            self.assertEqual(len(events.Scalars("repeat/best_srcc")), 1)
            self.assertNotIn("repeat/median_srcc", events.Tags()["scalars"])

    def test_invalid_repeat_configuration(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            for options in (["--runs", "1"], ["--seed", str(2**32 - 1)]):
                with self.subTest(options=options), self.assertRaises(SystemExit):
                    parse_repeat_args(options)

    def test_equals_output_argument_is_preserved(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "chosen"
            args, runs = parse_repeat_args([f"--output={output}", "--runs=2"])
            self.assertEqual(args.output, output)
            self.assertEqual(runs, 2)


if __name__ == "__main__":
    unittest.main()
