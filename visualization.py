"""TensorBoard 日志：记录训练曲线、测试指标和多次实验的汇总结果。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from train import TrainEpochResult


class TensorBoardLogger:
    """一个目录对应一组曲线，避免不同训练的同名事件混在一起。"""

    def __init__(self, log_dir: Path) -> None:
        # 延迟导入，让缺少依赖时仍可查看命令行帮助。
        from torch.utils.tensorboard import SummaryWriter

        if log_dir.exists() and (not log_dir.is_dir() or any(log_dir.iterdir())):
            raise FileExistsError(f"TensorBoard 日志目录非空，请使用新实验名或新 --log-dir：{log_dir}")
        self.writer: SummaryWriter = SummaryWriter(log_dir=str(log_dir), flush_secs=20)

    def log_config(self, config: dict) -> None:
        """将实际运行配置放在 Text 面板，方便核对曲线对应的设置。"""
        content = json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False)
        self.writer.add_text("experiment/config", f"```json\n{content}\n```", 0)

    def log_batch(
        self, result: TrainEpochResult, step: int, next_lr: float, amp_scale: float
    ) -> None:
        """step 是跨轮累计的 batch 数；loss 是本轮截至当前的样本加权均值。"""
        self.writer.add_scalar("train/running_loss", result.loss, step)
        # 回调发生在调度之后，因此这是下一批将使用的学习率。
        self.writer.add_scalar("optimizer/next_learning_rate", next_lr, step)
        self.writer.add_scalar("amp/scale", amp_scale, step)

    def log_train_epoch(self, result: TrainEpochResult, epoch: int) -> None:
        """训练完成就记录，即使后续评估失败也保留训练曲线。"""
        self.writer.add_scalar("train/epoch_loss", result.loss, epoch)
        self.writer.add_scalar("train/optimizer_steps", result.optimizer_steps, epoch)
        self.writer.add_scalar("amp/skipped_steps", result.skipped_steps, epoch)
        self.writer.flush()

    def log_evaluation(
        self, epoch: int, srcc: float, plcc: float, best: dict, seconds: float
    ) -> None:
        """只记录已保存成功的评估；best 是按 PLCC 选定的同一轮指标。"""
        self.writer.add_scalar("test/srcc", srcc, epoch)
        self.writer.add_scalar("test/plcc", plcc, epoch)
        self.writer.add_scalar("best/srcc", best["srcc"], epoch)
        self.writer.add_scalar("best/plcc", best["plcc"], epoch)
        self.writer.add_scalar("best/epoch", best["epoch"], epoch)
        self.writer.add_scalar("time/epoch_seconds", seconds, epoch)
        self.writer.flush()

    def log_repeat_result(self, run: int, srcc: float, plcc: float) -> None:
        """横轴为独立实验编号，而非训练轮数。"""
        self.writer.add_scalar("repeat/best_srcc", srcc, run)
        self.writer.add_scalar("repeat/best_plcc", plcc, run)
        self.writer.flush()

    def log_repeat_median(self, runs: int, srcc: float, plcc: float) -> None:
        """所有计划实验完成后才记录中位数。"""
        self.writer.add_scalar("repeat/median_srcc", srcc, runs)
        self.writer.add_scalar("repeat/median_plcc", plcc, runs)
        self.writer.flush()

    def close(self) -> None:
        """正常结束和异常退出均刷新事件队列并关闭写入线程。"""
        self.writer.close()
