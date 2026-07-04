"""Verifier-labeled DPO preferences (epic E8, Phase 12).

Generates on-policy preference pairs labeled by **correctness** instead of the
token-F1-to-a-reference judge (`scripts/prepare_dpo_prefs.py`): sample several
completions per verifiable task, verify each with the E1 verifier, and pair a
correct one (chosen) against an incorrect one (rejected). Both responses are the
model's own samples, so they are in-distribution — the sovereign pref path that
avoids the OOD Goodharting seen with far-off-policy `chosen` (banked DPO lesson).

Generator only: the output is the unchanged pref format
``{"prompt": [...], "chosen": str, "rejected": str}`` that ``PreferenceDataset``
and ``train_dpo`` already consume. ``sample``/``is_correct`` are injected so the
orchestration is pure and mock-testable.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


def make_pairs(
    prompt_messages: list[dict[str, str]],
    labeled: list[tuple[str, bool]],
    *,
    max_pairs: int = 1,
) -> list[dict[str, Any]]:
    """Form preference pairs from one prompt's labeled completions.

    ``labeled`` is ``[(text, is_correct)]``. Completions are de-duplicated by text,
    split into correct/incorrect, and zipped (correct → chosen, incorrect →
    rejected) up to ``max_pairs``. Returns ``[]`` when either side is empty (an
    all-correct or all-incorrect task carries no preference signal).
    """
    seen: set[str] = set()
    correct: list[str] = []
    incorrect: list[str] = []
    for text, ok in labeled:
        text = text.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        (correct if ok else incorrect).append(text)

    return [
        {"prompt": prompt_messages, "chosen": c, "rejected": r}
        for c, r in zip(correct, incorrect, strict=False)
    ][:max_pairs]


def build_verifier_prefs(
    tasks: Iterable[Any],
    sample: Callable[[Any], list[str]],
    is_correct: Callable[[str, Any], bool],
    *,
    samples_per_task: int = 4,
    max_pairs_per_task: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build verifier-labeled preference pairs across ``tasks``.

    ``sample(task)`` returns up to ``samples_per_task`` completion texts;
    ``is_correct(text, task)`` is the E1 verifier verdict. Returns ``(prefs,
    stats)`` where ``stats`` reports how many tasks yielded a usable correct∧
    incorrect mix vs were skipped.
    """
    prefs: list[dict[str, Any]] = []
    n_tasks = skipped_no_mix = 0
    for task in tasks:
        n_tasks += 1
        completions = sample(task)[:samples_per_task]
        labeled = [(text, is_correct(text, task)) for text in completions]
        pairs = make_pairs(_prompt_of(task), labeled, max_pairs=max_pairs_per_task)
        if not pairs:
            skipped_no_mix += 1
            continue
        prefs.extend(pairs)

    stats = {
        "tasks": n_tasks,
        "pairs": len(prefs),
        "skipped_no_mix": skipped_no_mix,  # all-correct or all-incorrect → no signal
        "samples_per_task": samples_per_task,
    }
    return prefs, stats


def _prompt_of(task: Any) -> list[dict[str, str]]:
    """The DPO prompt for a task: a single user turn with the problem statement."""
    prompt = task.prompt if hasattr(task, "prompt") else task["prompt"]
    return [{"role": "user", "content": prompt}]
