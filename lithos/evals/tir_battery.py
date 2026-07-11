"""The TIR tool-uplift battery (docs/eval-tir-battery-plan.md, Part A).

Runs each problem through ``tir_rollout`` **twice** — tools off (``max_tool_calls=0``,
pure chain-of-thought) and tools on — grades both with ``verify_tir``, and reports the
verified solve-rate difference **per difficulty tier**. That difference is tool-uplift:
the product thesis ("STEM computation is better executed than recalled"), quantified.

This is the eval consumer of the one verifier (`eval-plan.md` §0.5): the same sandbox
that scores RLVR reward and filters synthetic data now scores the battery. The two arms
share greedy decoding, so they differ *only* in whether the sandbox was available — the
clean control for the uplift claim.

Heavy checkpoint/tokenizer imports are deferred into ``run_tir_battery_eval`` so the
runner + stats can be imported (and unit-tested) without pulling the training stack.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from lithos.evals.tir_stats import UpliftStat, paired_uplift
from lithos.posttrain.chat_template import render_prompt, special_ids, tir_token_ids
from lithos.posttrain.taskbank import Task, verify_tir
from lithos.posttrain.tir_rollout import RolloutResult, tir_rollout

_UNSPECIFIED = "unspecified"  # tier label for tasks with no `level`


@dataclass(frozen=True)
class ArmOutcome:
    """One arm's result on one problem."""

    correct: bool
    num_tool_calls: int
    num_malformed_calls: int
    truncated: bool


@dataclass(frozen=True)
class TaskOutcome:
    """Both arms on one problem, plus the on-arm transcript for spot-checking."""

    task_id: str
    level: str | None
    family_id: str | None
    off: ArmOutcome
    on: ArmOutcome
    # Captured so every reported delta is checkable against a real rollout, not
    # believed blind (eval-plan §0.8) — and the same records feed Petra's channel view.
    on_tool_calls: list[tuple[str, str]] = field(default_factory=list)
    on_tool_outputs: list[str] = field(default_factory=list)
    on_completion: str = ""


def _arm(roll: RolloutResult, task: Task, *, timeout_s: float) -> ArmOutcome:
    result = verify_tir(roll.completion_text, roll.tool_calls, task, timeout_s=timeout_s)
    return ArmOutcome(
        correct=result.correct,
        num_tool_calls=roll.num_tool_calls,
        num_malformed_calls=roll.num_malformed_calls,
        truncated=roll.truncated,
    )


def run_two_arm(
    model: Any,
    task: Task,
    tok: Any,
    tir_ids: dict[str, int],
    sids: dict[str, int],
    *,
    device: str = "cpu",
    max_new: int = 512,
    max_tool_calls: int = 4,
    timeout_s: float = 5.0,
    temperature: float = 0.0,  # greedy: the honest pass@1 default, deterministic, so
    top_p: float | None = 0.95,  # the two arms share decoding exactly
    result_token_cap: int = 256,
    use_cache: bool = True,
) -> TaskOutcome:
    """Score one problem in both arms with identical decoding.

    Off-arm ``max_tool_calls=0`` makes the rollout loop a single segment, so the model
    answers from reasoning alone (off = *no sandbox*, not *no reasoning*). Assumes a
    stateless ``model`` (real ``LithosForCausalLM`` builds fresh KV caches per call).
    """
    prompt_ids = render_prompt([{"role": "user", "content": task.prompt}], tok)
    off_roll = tir_rollout(
        model, prompt_ids, tok, tir_ids, sids, device=device, max_new=max_new,
        max_tool_calls=0, temperature=temperature, top_p=top_p, timeout_s=timeout_s,
        result_token_cap=result_token_cap, use_cache=use_cache,
    )
    on_roll = tir_rollout(
        model, prompt_ids, tok, tir_ids, sids, device=device, max_new=max_new,
        max_tool_calls=max_tool_calls, temperature=temperature, top_p=top_p,
        timeout_s=timeout_s, result_token_cap=result_token_cap, use_cache=use_cache,
    )
    return TaskOutcome(
        task_id=task.id,
        level=task.level,
        family_id=task.family_id,
        off=_arm(off_roll, task, timeout_s=timeout_s),
        on=_arm(on_roll, task, timeout_s=timeout_s),
        on_tool_calls=on_roll.tool_calls,
        on_tool_outputs=on_roll.tool_outputs,
        on_completion=on_roll.completion_text,
    )


def run_battery(
    model: Any,
    tasks: list[Task],
    tok: Any,
    tir_ids: dict[str, int],
    sids: dict[str, int],
    **kw: Any,
) -> list[TaskOutcome]:
    """Two-arm every task in order. ``kw`` is forwarded to ``run_two_arm``."""
    return [run_two_arm(model, task, tok, tir_ids, sids, **kw) for task in tasks]


def _tier_of(outcome: TaskOutcome) -> str:
    return outcome.level or _UNSPECIFIED


def _stat(outcomes: list[TaskOutcome]) -> UpliftStat:
    return paired_uplift(
        [o.off.correct for o in outcomes],
        [o.on.correct for o in outcomes],
        [o.family_id for o in outcomes],
    )


def summarize(outcomes: list[TaskOutcome]) -> dict[str, Any]:
    """Per-tier + overall tool-uplift with paired CIs, plus on-arm rollout health."""
    tiers = sorted({_tier_of(o) for o in outcomes})
    per_tier = {t: asdict(_stat([o for o in outcomes if _tier_of(o) == t])) for t in tiers}
    n = len(outcomes)
    denom = n or 1
    solves_on = sum(1 for o in outcomes if o.on.correct)
    health = {
        "tool_call_rate": round(sum(1 for o in outcomes if o.on.num_tool_calls) / denom, 4),
        "malformed_call_rate": round(sum(o.on.num_malformed_calls for o in outcomes) / denom, 4),
        "truncation_rate_on": round(sum(1 for o in outcomes if o.on.truncated) / denom, 4),
        "tool_calls_per_solve": round(
            sum(o.on.num_tool_calls for o in outcomes) / max(solves_on, 1), 3
        ),
    }
    return {
        "battery": "tir-uplift",
        "n": n,
        "overall": asdict(_stat(outcomes)),
        "per_tier": per_tier,
        "health": health,
    }


def sample_transcripts(outcomes: list[TaskOutcome], k: int) -> list[dict[str, Any]]:
    """A deterministic transcript sample for spot-checking (eval-plan §0.8), the
    *uplift* cases first (on-right, off-wrong — those are the claim), then the rest;
    ordered by ``task_id`` for reproducibility."""
    if k <= 0:
        return []
    is_uplift = lambda o: o.on.correct and not o.off.correct  # noqa: E731
    chosen = sorted(
        outcomes, key=lambda o: (not is_uplift(o), o.task_id)
    )[:k]
    return [
        {
            "task_id": o.task_id,
            "level": o.level,
            "off_correct": o.off.correct,
            "on_correct": o.on.correct,
            "tool_calls": [list(tc) for tc in o.on_tool_calls],
            "tool_outputs": o.on_tool_outputs,
            "completion": o.on_completion,
        }
        for o in chosen
    ]


def run_tir_battery_eval(cfg: Any, checkpoint_path: str) -> Path:
    """Orchestrate a full battery run against a checkpoint: build the post-cutoff eval
    pool, two-arm every task, write the report + scorecard row. ``cfg`` is an
    ``EvalConfig`` (its ``.tir`` block configures the battery).
    """
    import datetime as dt

    from lithos.evals.report import write_eval_report
    from lithos.evals.run import load_model_from_checkpoint
    from lithos.evals.scorecard import append_entry
    from lithos.posttrain.taskbank import (
        assert_disjoint,
        filter_by_level,
        load_tasks,
        split_by_year,
    )
    from lithos.tokenizer import load_tokenizer
    from lithos.utils.device import resolve_device

    tb = cfg.tir
    if not tb.task_bank:
        raise ValueError("tir.task_bank is required (a kind=problems JSONL)")

    model, _train_cfg = load_model_from_checkpoint(checkpoint_path)
    device = resolve_device("auto")
    model.to(device)
    tok = load_tokenizer(cfg.tokenizer_path)
    sids = special_ids(tok)
    tir_ids = tir_token_ids(tok)  # requires the STEM tokenizer's TIR vocab

    tasks = load_tasks(tb.task_bank)
    if tb.levels:
        tasks = filter_by_level(tasks, tb.levels)
    if tb.cutoff_year is not None:
        train_tasks, eval_tasks = split_by_year(tasks, tb.cutoff_year)
        assert_disjoint(train_tasks, eval_tasks)  # eval-plan §5 guard
        tasks = eval_tasks
    if tb.limit is not None:
        tasks = tasks[: tb.limit]
    if not tasks:
        raise ValueError(
            "no eval tasks after level/year filtering — check tir.cutoff_year / tir.levels"
        )

    outcomes = run_battery(
        model, tasks, tok, tir_ids, sids, device=device, max_new=tb.max_new_tokens,
        max_tool_calls=tb.max_tool_calls, timeout_s=tb.tool_timeout_s,
        temperature=tb.temperature, result_token_cap=tb.result_token_cap,
    )
    summary = summarize(outcomes)
    summary["battery_version"] = tb.battery_version
    num_params = model.num_parameters()

    results: dict[str, Any] = {
        "tir": summary,
        "tir_transcripts": sample_transcripts(outcomes, tb.transcript_sample),
    }
    reference = {
        "checkpoint": str(checkpoint_path),
        "tokenizer": cfg.tokenizer_path,
        "num_parameters": num_params,
        "task_bank": tb.task_bank,
        "cutoff_year": tb.cutoff_year,
        "data_recipe": cfg.data_recipe,
    }
    out = write_eval_report(
        Path(cfg.output_dir) / cfg.name, name=cfg.name, results=results,
        model_reference=reference, config=cfg.model_dump(),
    )
    if cfg.scorecard_path:
        append_entry(
            cfg.scorecard_path,
            {
                "label": cfg.name,
                "timestamp": dt.datetime.now(dt.UTC).isoformat(),
                "checkpoint": str(checkpoint_path),
                "num_parameters": num_params,
                "data_recipe": cfg.data_recipe,
                "battery_version": tb.battery_version,
                "tir": summary,
            },
        )
    return out
