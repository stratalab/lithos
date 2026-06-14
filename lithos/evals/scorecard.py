"""Append-only benchmark scorecard (PRD §11.3) — compare models/recipes on a frozen battery.

The whole point of the data-ablation loop is comparison: recipe A vs B, 100M vs 1B. The
scorecard is a JSONL ledger keyed by (model, size, data_recipe, battery_version) so any two
runs can be diffed cleanly — but **only within a battery_version** (scores across versions
are not comparable, so ``diff`` refuses to mix them).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def append_entry(scorecard_path: str | Path, entry: dict[str, Any]) -> Path:
    """Append one result row as a JSON line; creates the file/parents if needed."""
    path = Path(scorecard_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path


def read_entries(scorecard_path: str | Path) -> list[dict[str, Any]]:
    """Read all result rows (empty list if the scorecard does not exist yet)."""
    path = Path(scorecard_path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _task_values(entry: dict[str, Any]) -> dict[str, float]:
    """Flatten an entry's benchmark block to ``{task: primary_value}``."""
    tasks = entry.get("benchmarks", {}).get("tasks", {})
    return {t: v["value"] for t, v in tasks.items() if v.get("value") is not None}


def diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Per-task and mean deltas (b - a). Refuses to compare across battery versions."""
    va = a.get("benchmarks", {}).get("battery_version")
    vb = b.get("benchmarks", {}).get("battery_version")
    if va != vb:
        raise ValueError(f"cannot diff across battery versions: {va!r} vs {vb!r}")
    ta, tb = _task_values(a), _task_values(b)
    per_task = {task: tb[task] - ta[task] for task in sorted(ta.keys() & tb.keys())}
    ma, mb = a.get("benchmarks", {}).get("mean"), b.get("benchmarks", {}).get("mean")
    return {
        "battery_version": va,
        "a": a.get("label"),
        "b": b.get("label"),
        "per_task": per_task,
        "mean_delta": (mb - ma) if (ma is not None and mb is not None) else None,
    }
