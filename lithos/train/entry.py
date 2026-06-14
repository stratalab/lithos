"""Config-driven training entrypoint (PRD §16).

Named ``entry`` (not ``train``) to avoid shadowing the re-exported ``train``
function at the ``lithos.train`` package level.
"""

from __future__ import annotations

from lithos.train.config import TrainConfig
from lithos.train.logging import RunDir
from lithos.train.loop import train
from lithos.utils.config import load_and_validate


def train_from_config(
    config_path: str, overrides: list[str] | None = None, resume_from: str | None = None
) -> RunDir:
    cfg = load_and_validate(config_path, TrainConfig, overrides)
    return train(cfg, resume_from=resume_from)
