"""Run directories and JSONL metrics logging (PRD §9.7).

A run directory is created once per training run and never clobbered (PRD §20.7)::

    runs/<YYYY-MM-DD_HHMMSS>_<name>/
      resolved_config.yaml
      metrics.jsonl
      run_manifest.json
      samples/
      checkpoints/
      evals/

The full training loop (Phase 4) writes through these helpers.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from lithos.utils.io import ensure_dir, ensure_new_dir, write_json

__all__ = ["JsonlWriter", "RunDir", "create_run_dir", "git_commit", "write_run_manifest"]

RUN_SUBDIRS = ("samples", "checkpoints", "evals")


@dataclass(frozen=True)
class RunDir:
    """Resolved paths inside a single run directory."""

    root: Path

    @property
    def resolved_config(self) -> Path:
        return self.root / "resolved_config.yaml"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics.jsonl"

    @property
    def manifest(self) -> Path:
        return self.root / "run_manifest.json"

    @property
    def samples(self) -> Path:
        return self.root / "samples"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def evals(self) -> Path:
        return self.root / "evals"


def _timestamp(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%Y-%m-%d_%H%M%S")


def create_run_dir(
    name: str,
    base: str | os.PathLike[str] = "runs",
    *,
    now: datetime | None = None,
    allow_existing: bool = False,
) -> RunDir:
    """Create ``runs/<timestamp>_<name>/`` with standard subdirs (no-clobber)."""
    root = Path(base) / f"{_timestamp(now)}_{name}"
    ensure_new_dir(root, allow_existing=allow_existing)
    for sub in RUN_SUBDIRS:
        ensure_dir(root / sub)
    return RunDir(root=root)


def git_commit() -> str | None:
    """Current HEAD commit SHA, or ``None`` outside a git repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    return out.stdout.strip() or None


def write_run_manifest(
    run: RunDir,
    *,
    stage: str,
    num_parameters: int,
    device: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write ``run_manifest.json`` for a run.

    The pretrain/SFT loop writes its own richer manifest; the DPO and GRPO trainers
    call this so every run directory carries the same provenance record (run id,
    commit, resolved config, parameter count) instead of silently omitting it.
    """
    manifest: dict[str, Any] = {
        "run_id": run.root.name,
        "stage": stage,
        "git_commit": git_commit(),
        "resolved_config": str(run.resolved_config),
        "num_parameters": num_parameters,
        "device": device,
    }
    if extra:
        manifest.update(extra)
    write_json(run.manifest, manifest)


class JsonlWriter:
    """Append-only JSON-Lines writer for ``metrics.jsonl`` and similar logs."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        """Append one record as a JSON line and flush."""
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
