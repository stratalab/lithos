"""Training: optimizer, scheduler, loop, checkpointing, logging (Phases 3-4)."""

from lithos.train.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from lithos.train.config import (
    DataConfig,
    OptimConfig,
    ScheduleConfig,
    TrainConfig,
)
from lithos.train.entry import train_from_config
from lithos.train.logging import JsonlWriter, RunDir, create_run_dir
from lithos.train.loop import evaluate, train
from lithos.train.optim import build_optimizer
from lithos.train.scheduler import cosine_lr, set_lr

__all__ = [
    "DataConfig",
    "JsonlWriter",
    "OptimConfig",
    "RunDir",
    "ScheduleConfig",
    "TrainConfig",
    "build_optimizer",
    "cosine_lr",
    "create_run_dir",
    "evaluate",
    "find_latest_checkpoint",
    "load_checkpoint",
    "save_checkpoint",
    "set_lr",
    "train",
    "train_from_config",
]
