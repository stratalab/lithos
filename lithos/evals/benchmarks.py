"""External benchmark evaluation via EleutherAI lm-evaluation-harness (PRD §11.2, §26.8).

A Lithos checkpoint exports to a HF/Qwen3-loadable directory, so the harness runs
through its ``hf`` backend with no bespoke adapter. ``lm-eval`` is heavy and optional,
so it is imported lazily — the core install never needs it.

The **battery is frozen and versioned**: bumping the task list or its version is a
new ``battery_version``, recorded with every result, so scores are only ever compared
within a version. The same battery runs identically at 100M and 3B (scale-invariant).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# The frozen small-model battery (v1). These are the standard zero-shot tasks where a
# sub-1B model shows signal; bumping this set or its pins => bump BATTERY_VERSION.
BATTERY_VERSION = "v1"
DEFAULT_TASKS = [
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "piqa",
    "winogrande",
    "lambada_openai",
    "sciq",
    "openbookqa",
]

# Metric preference order — lm-eval reports several per task (acc, acc_norm, ...),
# sometimes with a ",none" filter suffix. We pick one primary per task, consistently.
_PRIMARY_PREFERENCE = ("acc_norm,none", "acc_norm", "acc,none", "acc")


def _primary_metric(metrics: dict[str, Any]) -> tuple[str, float | None]:
    """Choose one primary (name, value) for a task from lm-eval's metric dict."""
    for key in _PRIMARY_PREFERENCE:
        val = metrics.get(key)
        if isinstance(val, (int, float)):
            return key, float(val)
    for name, val in metrics.items():
        if isinstance(val, (int, float)) and "stderr" not in name:
            return name, float(val)
    return "none", None


def normalize_results(results: dict[str, Any]) -> dict[str, Any]:
    """Flatten lm-eval output into ``{tasks: {task: {metric, value}}, mean, num_tasks}``.

    Pure function (no lm-eval dependency) so it is unit-tested directly.
    """
    raw = results.get("results", {})
    tasks: dict[str, Any] = {}
    for task, metrics in raw.items():
        name, value = _primary_metric(metrics)
        tasks[task] = {"metric": name, "value": value}
    values = [t["value"] for t in tasks.values() if t["value"] is not None]
    return {
        "tasks": tasks,
        "mean": (sum(values) / len(values)) if values else None,
        "num_tasks": len(tasks),
    }


def run_benchmarks(
    export_dir: str | Path,
    tasks: list[str],
    *,
    battery_version: str = BATTERY_VERSION,
    num_fewshot: int = 0,
    limit: int | None = None,
    batch_size: str | int = "auto",
    dtype: str = "bfloat16",
    device: str = "cuda",
) -> dict[str, Any]:
    """Run lm-eval-harness on an exported HF model dir; return normalized scores.

    ``limit`` caps examples/task (use a small value for smoke; ``None`` = full).
    """
    try:
        import lm_eval
    except ImportError as e:
        raise ImportError(
            "lm-eval is required for benchmark evaluation. Install it "
            "(`uv sync --extra eval` / `pip install 'lithos[eval]'`) or run perplexity-only "
            "with benchmarks.enabled=false."
        ) from e

    raw = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={Path(export_dir)},dtype={dtype}",
        tasks=list(tasks),
        num_fewshot=num_fewshot,
        limit=limit,
        batch_size=batch_size,
        device=device,
    )
    out = normalize_results(raw or {})
    out["battery_version"] = battery_version
    out["num_fewshot"] = num_fewshot
    if limit is not None:
        out["limit"] = limit
    return out
