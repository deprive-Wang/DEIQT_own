"""顺序运行多次 LIVE-C 划分实验，并汇总各次最佳轮的指标。正式训练请手动启动。"""

import argparse
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

from dataset import split_livec_indices
from train import PROJECT_DIR, parse_args, run_training
from visualization import TensorBoardLogger


def parse_repeat_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, int]:
    """复用单次训练参数；--seed 表示首个实验的种子。"""
    parser = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False,
        epilog="其它训练参数沿用 train.py，包括 --log-dir 日志根目录（默认 tf-logs）；详见 train.py --help。",
    )
    parser.add_argument("--runs", type=int, default=10, help="独立划分和从头训练的次数，默认 10")
    repeat_args, training_argv = parser.parse_known_args(argv)
    if repeat_args.runs < 2:
        parser.error("runs 必须大于等于 2")
    training_args = parse_args(training_argv)
    if training_args.seed + repeat_args.runs > 2**32:
        parser.error("seed 到最后一次运行的种子必须小于 2**32")
    if not any(option == "--output" or option.startswith("--output=")
               for option in training_argv):
        training_args.output = PROJECT_DIR / "outputs" / f"repeat_{datetime.now():%Y%m%d_%H%M%S_%f}"
    return training_args, repeat_args.runs


def _write_summary(path: Path, content: dict) -> None:
    """用同目录临时文件替换汇总，避免进程中断留下截断的 JSON。"""
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_run_result(output: Path, seed: int, epochs: int) -> dict:
    """只接受完整轮数的真实训练输出，避免把中断实验计入中位数。"""
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    split = json.loads((output / "split.json").read_text(encoding="utf-8"))
    if summary["completed_epochs"] != epochs or split["seed"] != seed:
        raise ValueError(f"实验输出的轮数或种子不匹配：{output}")
    if not (output / "best.pth").is_file():
        raise FileNotFoundError(f"实验缺少最佳权重：{output / 'best.pth'}")
    best = summary["best"]
    if type(best["epoch"]) is not int or not 1 <= best["epoch"] <= epochs:
        raise ValueError(f"最佳轮数无效：{output}")
    for metric in ("srcc", "plcc"):
        value = best[metric]
        if type(value) not in (int, float) or not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError(f"{metric} 必须是 [-1, 1] 内的有限数值：{output}")
    return {"seed": seed, "best_epoch": best["epoch"],
            "srcc": best["srcc"], "plcc": best["plcc"], "output": str(output)}


def run_repeats(training_args: argparse.Namespace, runs: int = 10) -> Path:
    """逐次重建模型与数据划分；全部完成后分别取 SRCC、PLCC 的中位数。"""
    if type(runs) is not int or runs < 2:
        raise ValueError("runs 必须大于等于 2")
    if training_args.seed + runs > 2**32:
        raise ValueError("seed 到最后一次运行的种子必须小于 2**32")
    output = Path(training_args.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"重复实验输出目录非空，请另选新目录：{output}")

    seeds = list(range(training_args.seed, training_args.seed + runs))
    splits = [tuple(split_livec_indices(seed)[0]) for seed in seeds]
    if len(set(splits)) != runs:
        raise ValueError("重复实验产生了相同的训练图像划分，请更换起始种子")

    output.mkdir(parents=True, exist_ok=True)
    # 一次重复实验占用一个日志组，各独立训练再使用 run_01 等子目录。
    experiment_log_dir = training_args.log_dir / output.name
    summary_path = output / "repeat_summary.json"
    summary = {"status": "running", "planned_runs": runs, "completed_runs": 0,
               "epochs_per_run": training_args.epochs, "seeds": seeds,
               "smoke_test": training_args.smoke_test, "runs": [], "median": None,
               "tensorboard_log_dir": str(experiment_log_dir)}
    _write_summary(summary_path, summary)
    visualizer = TensorBoardLogger(experiment_log_dir / "summary")
    try:
        visualizer.log_config(summary)
        for index, seed in enumerate(seeds, start=1):
            run_output = output / f"run_{index:02d}"
            run_args = argparse.Namespace(**{**vars(training_args), "seed": seed, "output": run_output,
                                             "log_dir": experiment_log_dir})
            run_training(run_args)
            result = _read_run_result(run_output, seed, training_args.epochs)
            result["run"] = index
            summary["runs"].append(result)
            summary["completed_runs"] = index
            _write_summary(summary_path, summary)
            visualizer.log_repeat_result(index, result["srcc"], result["plcc"])

        summary["median"] = {
            "srcc": statistics.median(result["srcc"] for result in summary["runs"]),
            "plcc": statistics.median(result["plcc"] for result in summary["runs"]),
        }
        summary["status"] = "completed"
        _write_summary(summary_path, summary)
        visualizer.log_repeat_median(runs, summary["median"]["srcc"], summary["median"]["plcc"])
    finally:
        visualizer.close()
    return output


if __name__ == "__main__":
    args, repeat_count = parse_repeat_args()
    run_repeats(args, repeat_count)
