"""LIVE-C 单卡训练：单轮参数更新与多轮入口。--smoke-test 只检查小规模流程。"""

import argparse
import csv
import json
import logging
import math
import platform
import random
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy
import torch
import torchvision
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from checkpoint import capture_random_state, save_checkpoint, save_epoch_checkpoints
from dataset import LiveChallengeDataset, split_livec_indices
from evaluation import evaluate
from lr_scheduler import WarmupStepLR
from model import DEIQT
from optimizer import build_optimizer
from pretrained import load_pretrained_encoder
from visualization import TensorBoardLogger


PROJECT_DIR = Path(__file__).resolve().parent


@dataclass
class TrainEpochResult:
    """loss 是按实际裁剪样本数加权的平均 SmoothL1 损失。"""

    loss: float
    batches: int
    samples: int
    optimizer_steps: int
    skipped_steps: int


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: Optimizer,
    *,
    scaler: torch.amp.GradScaler | None = None,
    clip_grad: float = 5.0,
    scheduler: WarmupStepLR | None = None,
    progress_callback: Callable[[TrainEpochResult], None] | None = None,
) -> TrainEpochResult:
    """完成一轮训练，每个 batch 更新一次，不做梯度累积。

    默认 FP32；CUDA 混合精度需传入启用的 GradScaler，同一个 scaler 应跨轮复用。
    传入 scheduler 时在每个 batch 后推进调度；同一 scheduler 必须跨轮复用。
    不传 scheduler 时保持优化器当前学习率。输入标签必须为 [B, 1]。
    训练样本可以打乱，不能按连续 patch_num 个样本计算图像级相关系数。
    """
    if not math.isfinite(clip_grad) or clip_grad <= 0:
        raise ValueError("clip_grad 必须为有限正数")
    if len(data_loader) == 0:
        raise ValueError("训练 DataLoader 为空，请检查数据量、batch_size 和 drop_last")
    if scheduler is not None:
        if scheduler.optimizer is not optimizer:
            raise ValueError("scheduler 与训练必须使用同一个 optimizer")
        if scheduler.steps_per_epoch != len(data_loader):
            raise ValueError("scheduler.steps_per_epoch 必须等于训练 DataLoader 的 batch 数")
    device = next(model.parameters()).device
    use_amp = scaler is not None and scaler.is_enabled()
    if use_amp and device.type != "cuda":
        raise ValueError("本模块的混合精度训练仅支持 CUDA")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    criterion = nn.SmoothL1Loss(beta=1.0, reduction="mean")
    total_loss = 0.0
    sample_count = batch_count = optimizer_steps = skipped_steps = 0
    model.train()
    try:
        for images, targets in data_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if targets.shape != (images.shape[0], 1):
                raise ValueError("训练标签形状必须为 [B, 1]，避免损失计算发生广播")
            if not torch.isfinite(targets).all():
                raise ValueError("训练标签包含 NaN 或无穷值")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(images)
            if outputs.shape != targets.shape:
                raise ValueError("模型输出与训练标签均应为 [B, 1]")
            # 在 FP32 中计算回归损失，避免半精度下的大 MOS 残差降低数值稳定性。
            loss = criterion(outputs.float(), targets.float())
            if not torch.isfinite(loss):
                raise FloatingPointError("训练损失非有限，已停止更新")

            if use_amp:
                scaler.scale(loss).backward()
                # 必须先还原真实梯度，再裁剪；缩放后的梯度不能直接裁剪。
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(parameters, clip_grad)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # GradScaler 溢出时跳过更新并减小 scale，不能记成有效优化步。
                skipped = scaler.get_scale() < previous_scale
                skipped_steps += int(skipped)
                optimizer_steps += int(not skipped)
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(parameters, clip_grad, error_if_nonfinite=True)
                optimizer.step()
                optimizer_steps += 1

            if scheduler is not None:
                # 对齐作者：即使 AMP 本批跳过参数更新，batch 时间轴也继续推进。
                scheduler.step()
            total_loss += loss.detach().item() * images.shape[0]
            sample_count += images.shape[0]
            batch_count += 1
            if progress_callback is not None:
                progress_callback(TrainEpochResult(
                    total_loss / sample_count, batch_count, sample_count,
                    optimizer_steps, skipped_steps,
                ))
    finally:
        # 异常退出或一轮结束后都不残留旧梯度，方便下一轮或评估。
        optimizer.zero_grad(set_to_none=True)
    return TrainEpochResult(
        total_loss / sample_count, batch_count, sample_count, optimizer_steps, skipped_steps
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """默认路径以本文件所在目录为基准，显式传入的相对路径以工作目录为基准。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_DIR / "data/ChallengeDB_release")
    parser.add_argument("--pretrained", type=Path, default=PROJECT_DIR / "weights/deit_3_small_224_1k.pth")
    parser.add_argument("--output", type=Path, help="使用新的或空的输出目录；默认自动生成时间戳目录")
    parser.add_argument("--log-dir", type=Path, default=PROJECT_DIR / "tf-logs",
                        help="TensorBoard 日志根目录，默认项目根目录下的 tf-logs")
    parser.add_argument("--epochs", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=4, help="单卡训练 batch size，默认 4")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0, help="Windows 首次运行建议 0")
    parser.add_argument("--patch-num", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--warmup-lr", type=float, default=2e-7)
    parser.add_argument("--decay-epochs", type=int, default=3)
    parser.add_argument("--decay-rate", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="仅使用 4 张训练图和 3 张测试图，各裁剪 2 次，batch size 2，运行 1 轮",
    )
    args = parser.parse_args(argv)
    for name in ("epochs", "batch_size", "eval_batch_size", "patch_num", "decay_epochs", "log_interval"):
        if getattr(args, name) < 1:
            parser.error(f"{name} 必须大于 0")
    if args.num_workers < 0 or args.warmup_epochs < 0 or not 0 <= args.seed < 2**32:
        parser.error("num_workers/warmup_epochs 必须非负，seed 必须在 [0, 2**32) 范围内")
    for name in ("learning_rate", "weight_decay", "clip_grad", "warmup_lr", "decay_rate"):
        if not math.isfinite(getattr(args, name)):
            parser.error(f"{name} 必须为有限数值")
    if args.learning_rate <= 0 or args.clip_grad <= 0 or args.weight_decay < 0:
        parser.error("学习率和梯度裁剪值必须为正，权重衰减必须非负")
    if not 0 <= args.warmup_lr <= args.learning_rate or not 0 < args.decay_rate <= 1:
        parser.error("warmup_lr 必须在 [0, learning_rate] 内，decay_rate 必须在 (0, 1] 内")
    if args.smoke_test:
        args.epochs = 1
        args.batch_size = args.eval_batch_size = args.patch_num = 2
    if args.device == "cpu":
        args.amp = False
    if args.output is None:
        mode = "smoke" if args.smoke_test else "livec"
        args.output = PROJECT_DIR / "outputs" / f"{mode}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    for name in ("data_root", "pretrained", "output", "log_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def seed_worker(worker_id: int) -> None:
    """作为顶层函数供 Windows spawn 调用；每个 worker 使用其独立种子。"""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _write_json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _create_logger(output: Path) -> logging.Logger:
    logger = logging.getLogger(f"deiqt.{output}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(output / "train.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def run_training(args: argparse.Namespace) -> Path:
    """运行指定配置并返回输出目录；每次运行拥有独立目录，防止覆盖已有实验。"""
    if not args.pretrained.is_file():
        raise FileNotFoundError(f"预训练权重不存在：{args.pretrained}")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise FileExistsError(f"输出目录非空，请另选新目录：{args.output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；请检查 deiqt 环境，或明确选择 --device cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    # 每轮重新创建非持久 worker；随机状态与实际索引一并记录，便于复核实验。
    generator = torch.Generator().manual_seed(args.seed)
    train_indices, test_indices = split_livec_indices(args.seed)
    if args.smoke_test:
        train_indices, test_indices = train_indices[:4], test_indices[:3]
    train_dataset = LiveChallengeDataset(args.data_root, train_indices, True, args.patch_num)
    test_dataset = LiveChallengeDataset(args.data_root, test_indices, False, args.patch_num)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        generator=generator, worker_init_fn=seed_worker,
    )
    if len(train_loader) == 0:
        raise ValueError("训练 batch_size 超过裁剪样本数量，drop_last=True 会产生空数据流")
    model = DEIQT()
    load_pretrained_encoder(model.encoder, args.pretrained)
    model.to(device)
    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
    scheduler = WarmupStepLR(
        optimizer, len(train_loader), args.warmup_epochs,
        args.decay_epochs, args.warmup_lr, args.decay_rate,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    tensorboard_dir = args.log_dir / args.output.name
    config = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    config.update({
        "scheduler_rule": "timm_0.6.13_step_after_batch_zero_based",
        "train_drop_last": True, "steps_per_epoch": len(train_loader),
        "model_selection": "max_test_plcc_over_epochs",
        "evaluation_precision": "float32", "gradient_accumulation": 1,
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "torchvision": str(torchvision.__version__), "scipy": scipy.__version__},
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "tensorboard_log_dir": str(tensorboard_dir),
    })
    split = {"seed": args.seed, "train_indices": train_indices, "test_indices": test_indices,
             "index_basis": "AllImages_release[7:1169]", "smoke_test": args.smoke_test}
    args.output.mkdir(parents=True, exist_ok=True)
    _write_json(args.output / "config.json", config)
    _write_json(args.output / "split.json", split)
    logger = _create_logger(args.output)
    history = []
    best = None
    visualizer = None
    try:
        visualizer = TensorBoardLogger(tensorboard_dir)
        visualizer.log_config(config)
        logger.info("模式=%s；设备=%s；训练/测试图像=%d/%d；每图裁剪=%d；batch=%d；每轮 batch=%d",
                    "流程检查，指标不用于论文对比" if args.smoke_test else "完整单次划分",
                    config["device_name"], len(train_indices), len(test_indices), args.patch_num,
                    args.batch_size, len(train_loader))
        logger.info("输出目录：%s", args.output)
        logger.info("TensorBoard 日志：%s", tensorboard_dir)
        logger.info("best.pth 按测试集 PLCC 选择，与作者口径一致；这不是独立验证集选模。")
        if args.batch_size != 16:
            logger.info("当前为单卡 batch=%d；作者 LIVE-C 配置为每进程 batch=16，多卡总 batch 还需乘进程数。", args.batch_size)
        for epoch in range(1, args.epochs + 1):
            start = time.perf_counter()
            lr_start = optimizer.param_groups[0]["lr"]

            def report_progress(result: TrainEpochResult) -> None:
                if result.batches == 1 or result.batches % args.log_interval == 0 or result.batches == len(train_loader):
                    logger.info("轮 %d/%d，batch %d/%d，平均损失=%.6f，下一步 lr=%.3g，已更新/跳过=%d/%d",
                                epoch, args.epochs, result.batches, len(train_loader), result.loss,
                                optimizer.param_groups[0]["lr"], result.optimizer_steps, result.skipped_steps)
                    # 仅在日志采样点写入，减少长训练中事件文件的写入量。
                    step = (epoch - 1) * len(train_loader) + result.batches
                    visualizer.log_batch(result, step, optimizer.param_groups[0]["lr"], scaler.get_scale())

            train_result = train_one_epoch(
                model, train_loader, optimizer, scaler=scaler, clip_grad=args.clip_grad,
                scheduler=scheduler, progress_callback=report_progress,
            )
            if train_result.optimizer_steps == 0:
                logger.warning("本轮没有有效参数更新，请检查 AMP scale、梯度或数据；不应据此判断训练效果。")
            # 先保留已完成的参数更新；评估报错时不能把本轮训练结果一起丢掉。
            training_state = {
                "format_version": 1, "model": model.state_dict(), "epoch": epoch,
                "config": config, "split": split, "train_result": asdict(train_result),
                "evaluation_status": "pending", "metrics": None,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "random_state": capture_random_state(generator, device),
                "history": history, "best": best,
            }
            save_checkpoint(args.output / "last.pth", training_state)
            visualizer.log_train_epoch(train_result, epoch)
            logger.info("轮 %d：开始评估 %d 张图像（%d 个裁剪）", epoch, len(test_indices), len(test_dataset))
            evaluation = evaluate(model, test_dataset, args.eval_batch_size, args.num_workers)
            row = {"epoch": epoch, **asdict(train_result), "lr_start": lr_start,
                   "lr_next": optimizer.param_groups[0]["lr"], "amp_scale": scaler.get_scale(),
                   "srcc": evaluation.srcc, "plcc": evaluation.plcc,
                   "seconds": time.perf_counter() - start}
            next_history = [*history, row]
            # 与作者一样按 PLCC 选模；初始值不设为 0，避免全负相关时没有 best 文件。
            improved = best is None or evaluation.plcc >= best["plcc"]
            next_best = {"epoch": epoch, "srcc": evaluation.srcc, "plcc": evaluation.plcc} if improved else best
            inference_state = {
                "format_version": 1, "model": model.state_dict(), "epoch": epoch,
                "metrics": {"srcc": evaluation.srcc, "plcc": evaluation.plcc},
                "evaluation_status": "completed",
                "config": config, "split": split,
                "predictions": evaluation.predictions, "targets": evaluation.targets,
            }
            # last 保存完整训练状态；best 仅保存模型及其对应评估信息，节省磁盘空间。
            save_epoch_checkpoints(args.output, {
                **training_state, **inference_state,
                "random_state": capture_random_state(generator, device), "history": next_history,
                "best": next_best,
            }, inference_state if improved else None)
            history, best = next_history, next_best



            with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(history)
            _write_json(args.output / "summary.json", {
                "completed_epochs": epoch, "planned_epochs": args.epochs,
                "smoke_test": args.smoke_test, "best": best,
            })
            visualizer.log_evaluation(epoch, evaluation.srcc, evaluation.plcc, best, row["seconds"])
            logger.info("轮 %d 完成：损失=%.6f，SRCC=%.6f，PLCC=%.6f，耗时=%.1f 秒",
                        epoch, train_result.loss, evaluation.srcc, evaluation.plcc, row["seconds"])
        logger.info("运行完成；最佳 PLCC 对应第 %d 轮，结果见 summary.json。", best["epoch"])
    except BaseException:
        logger.exception("运行中断；last.pth 的 evaluation_status 区分待评估/已评估状态；"
                         "summary.json 只记录此前已完成输出的轮次。")
        raise


    
    finally:
        try:
            if visualizer is not None:
                visualizer.close()
        finally:
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
    return args.output


if __name__ == "__main__":
    run_training(parse_args())
