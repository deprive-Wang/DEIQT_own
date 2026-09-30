"""训练状态的原子保存，避免保存中断破坏已有 checkpoint。"""

import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch


def capture_random_state(generator: torch.Generator, device: torch.device) -> dict:
    """记录主进程、CUDA 和 DataLoader 的随机状态，保留后续恢复所需信息。"""
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        # 把 NumPy 数组转换为普通列表，保持 torch.load(weights_only=True) 可读取。
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        "loader": generator.get_state(),
    }


def save_checkpoint(path: str | Path, state: dict) -> None:
    """先写入同目录临时文件，再替换目标；写入失败时保留旧 checkpoint。"""
    path = Path(path)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            torch.save(state, temporary_file)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def save_epoch_checkpoints(output: Path, last: dict, best: dict | None) -> None:
    """成组保存本轮权重；写入异常时回滚已替换的文件，不提供断电事务保证。"""
    if best is None:
        save_checkpoint(output / "last.pth", last)
        return

    # 临时文件和备份与目标位于同一文件系统；全部准备成功后才开始替换。
    staging = Path(tempfile.mkdtemp(dir=output, prefix=".checkpoint-"))
    published = []
    backups = {}
    keep_backups = False
    try:
        for name, state in (("last.pth", last), ("best.pth", best)):
            save_checkpoint(staging / name, state)
            target = output / name
            backup = staging / f"previous_{name}"
            if target.exists():
                shutil.copyfile(target, backup)
                backups[name] = backup
            else:
                backups[name] = None
        try:
            for name in ("last.pth", "best.pth"):
                (staging / name).replace(output / name)
                published.append(name)
        except BaseException:
            rollback_errors = []
            for name in reversed(published):
                try:
                    backup = backups[name]
                    if backup is None:
                        (output / name).unlink()
                    else:
                        backup.replace(output / name)
                except OSError as error:
                    rollback_errors.append(error)
            if rollback_errors:
                # 回滚也被磁盘/权限错误阻止时，必须留下可供人工恢复的副本。
                keep_backups = True
                raise RuntimeError(f"checkpoint 回滚失败，保留恢复文件：{staging}") from rollback_errors[0]
            raise
    finally:
        if not keep_backups:
            shutil.rmtree(staging)
