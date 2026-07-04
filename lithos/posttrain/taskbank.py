"""Task bank + verify dispatch for RLVR/eval (epics E1c/E1d/E1f, Phase 12).

A ``Task`` is a problem plus its checker spec — the pool GRPO samples from and the
eval battery scores against. Loaded from ``kind=problems`` acquisitions (JSONL).
``verify`` routes a response to the right ``check_*`` primitive by ``Task.kind``;
``verify_batch`` runs many concurrently (verification is subprocess-bound, so
threads overlap — the throughput hinge for RL rollouts); ``split_by_year`` +
``assert_disjoint`` enforce the RLVR-pool / eval-set separation (`docs/eval-plan.md`
principle 5: train on pre-cutoff, eval on post-cutoff — contamination-resistant by
construction).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lithos.posttrain.verifier import (
    CheckResult,
    check_code,
    check_numeric,
    check_symbolic,
    check_units,
)

VALID_KINDS = frozenset({"numeric", "symbolic", "code", "units"})


@dataclass(frozen=True)
class Task:
    """A verifiable problem. ``kind`` selects the checker; the fields it needs
    depend on the kind (``answer`` for numeric/symbolic/units, ``tests`` for code,
    ``units`` for units). ``level``/``year`` drive the difficulty ladder + split."""

    id: str
    prompt: str
    kind: str = "numeric"
    answer: str = ""
    tests: str | None = None  # code: unit-test harness (asserts)
    units: str | None = None  # units: expected unit string, e.g. "kPa"
    tol: float = 1e-6  # numeric/units relative tolerance
    level: str | None = None  # difficulty ladder rung
    year: int | None = None  # for the train/eval year split
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in VALID_KINDS:
            raise ValueError(f"unknown task kind {self.kind!r}; expected one of {sorted(VALID_KINDS)}")
        if self.kind == "code" and not self.tests:
            raise ValueError(f"code task {self.id!r} needs a `tests` harness")
        if self.kind == "units" and not self.units:
            raise ValueError(f"units task {self.id!r} needs a `units` field")


def task_from_record(rec: dict[str, Any]) -> Task:
    """Coerce a JSONL record into a ``Task`` (defaults fill missing optional keys)."""
    return Task(
        id=str(rec["id"]),
        prompt=rec["prompt"],
        kind=rec.get("kind", "numeric"),
        answer=str(rec.get("answer", "")),
        tests=rec.get("tests"),
        units=rec.get("units"),
        tol=float(rec.get("tol", 1e-6)),
        level=rec.get("level"),
        year=rec.get("year"),
        metadata=rec.get("metadata", {}),
    )


def load_tasks(path: str | Path) -> list[Task]:
    """Read a JSONL problem bank into ``Task`` objects."""
    tasks: list[Task] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(task_from_record(json.loads(line)))
    return tasks


def verify(response: str, task: Task, *, timeout_s: float = 5.0) -> CheckResult:
    """Route ``response`` to the checker for ``task.kind`` and return its verdict."""
    if task.kind == "numeric":
        return check_numeric(response, task.answer, rel_tol=task.tol)
    if task.kind == "symbolic":
        return check_symbolic(response, task.answer)
    if task.kind == "code":
        result, _ = check_code(response, task.tests or "", timeout_s=timeout_s)
        return result
    if task.kind == "units":
        return check_units(response, task.answer, units=task.units or "", rel_tol=task.tol)
    raise ValueError(f"unhandled task kind {task.kind!r}")  # pragma: no cover — Task validates


def verify_batch(
    items: list[tuple[str, Task]], *, max_workers: int = 8, timeout_s: float = 5.0
) -> list[CheckResult]:
    """Verify many ``(response, task)`` pairs concurrently, order preserved.

    Verification blocks on subprocesses (code) or is CPU-light (numeric/symbolic),
    so threads overlap the subprocess waits — this keeps the GPU from idling while
    a batch of RL rollouts is checked serially. Order matches ``items``.
    """
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(lambda it: verify(it[0], it[1], timeout_s=timeout_s), items))


# --------------------------------------------------------------------------- #
# Year split + level filter (E1f): contamination-resistant train/eval pools.   #
# --------------------------------------------------------------------------- #


def filter_by_level(tasks: Iterable[Task], levels: Iterable[str]) -> list[Task]:
    """Keep tasks whose ``level`` is in ``levels`` (the difficulty ladder rung)."""
    wanted = set(levels)
    return [t for t in tasks if t.level in wanted]


def split_by_year(tasks: Iterable[Task], cutoff_year: int) -> tuple[list[Task], list[Task]]:
    """Split into (train ≤ cutoff, eval > cutoff). Tasks without a ``year`` go to
    train — only *dated* problems are eligible for the contamination-proof eval
    pool (`eval-plan.md` principle 5)."""
    train: list[Task] = []
    hold: list[Task] = []
    for t in tasks:
        (hold if (t.year is not None and t.year > cutoff_year) else train).append(t)
    return train, hold


def assert_disjoint(train: Iterable[Task], evalset: Iterable[Task]) -> None:
    """Structural guarantee that no problem leaks from eval into the RLVR pool.
    Raises with the offending ids if the id sets intersect. Call it wherever an
    RLVR pool and an eval set are built from the same acquisition."""
    train_ids = {t.id for t in train}
    overlap = train_ids & {t.id for t in evalset}
    if overlap:
        sample = sorted(overlap)[:5]
        raise ValueError(
            f"RLVR/eval pools share {len(overlap)} task id(s), e.g. {sample} — "
            "eval problems must never enter the training pool (eval-plan §5)"
        )
