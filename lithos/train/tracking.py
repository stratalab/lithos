"""Optional experiment tracking (PRD §15).

The canonical run record is always the local ``metrics.jsonl`` -- you own it and
it survives any outage. This module adds an OPTIONAL Weights & Biases mirror on
top, initialised on rank 0 only and lazily importing ``wandb`` so it is never a
hard dependency. Disabled by default; enable per-run via the ``wandb`` config.
"""

from __future__ import annotations

from typing import Any

from lithos.train.config import TrainConfig
from lithos.utils.env import load_env


class Reporter:
    """Thin wrapper around a wandb run; a no-op when tracking is disabled."""

    def __init__(self, run: Any = None) -> None:
        self._run = run

    @property
    def enabled(self) -> bool:
        return self._run is not None

    def log(self, record: dict[str, Any], step: int) -> None:
        """Forward a metrics record to wandb at ``step`` (no-op if disabled)."""
        if self._run is None:
            return
        # ``step``/``timestamp`` are bookkeeping; everything else is a metric.
        data = {k: v for k, v in record.items() if k not in ("step", "timestamp")}
        if data:
            self._run.log(data, step=step)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None


def init_reporter(cfg: TrainConfig, *, run_id: str, run_dir: str, is_main: bool) -> Reporter:
    """Build a Reporter; a no-op unless tracking is enabled on the main rank."""
    wb = cfg.wandb
    if not (wb.enabled and is_main):
        return Reporter(None)
    load_env()  # pick up WANDB_API_KEY from a local .env, like the storage layer
    try:
        import wandb
    except ImportError as e:
        raise ImportError(
            "wandb.enabled=true but wandb is not installed. Install it "
            "(`uv sync --extra tracking` / `pip install 'lithos[tracking]'`) "
            "or set wandb.enabled=false."
        ) from e
    run = wandb.init(
        project=wb.project,
        entity=wb.entity,
        name=run_id,
        group=wb.group or cfg.run_name,
        tags=list(wb.tags),
        notes=wb.notes,
        mode=wb.mode,
        dir=run_dir,
        config=cfg.model_dump(mode="json"),
    )
    return Reporter(run)
