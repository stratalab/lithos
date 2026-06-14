"""Load a local ``.env`` file into the environment (best-effort, no extra deps).

Credentials (R2/S3 keys, etc.) live in a git-ignored ``.env``; this loads them so
any process that touches storage picks them up without a manual ``source``.
Existing environment variables always win unless ``override=True``.
"""

from __future__ import annotations

import os
from pathlib import Path

_LOADED = False


def _find(filename: str) -> Path | None:
    for directory in [Path.cwd(), *Path.cwd().parents]:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def load_env(filename: str = ".env", *, override: bool = False) -> None:
    """Load ``KEY=VALUE`` lines from ``.env`` (searched from cwd upward). Idempotent."""
    global _LOADED
    if _LOADED and not override:
        return
    _LOADED = True
    path = _find(filename)
    if path is None:
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        line = line.removeprefix("export ").strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (override or key not in os.environ):
            os.environ[key] = value
