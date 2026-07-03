"""LLM quality-labeling: prompts, parsing, agreement (docs/quality-classifiers.md §3).

The pure layer of the labeling pipeline — prompt assembly from versioned
rubrics, strict response parsing, and pilot agreement statistics. Network I/O
(the OpenAI-compatible endpoint client) lives in `scripts/label_quality.py`;
everything here is testable offline.

Labels are only comparable within a rubric version; every record carries it.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

DOC_CHAR_BUDGET = 8000  # ~2k tokens of document is enough to judge quality


def build_prompt(
    rubric: str, response_format: str, doc_text: str, *, char_budget: int = DOC_CHAR_BUDGET
) -> list[dict[str, str]]:
    """Chat messages for one labeling call (truncated doc, rubric, format)."""
    body = doc_text[:char_budget]
    truncated = " [TRUNCATED]" if len(doc_text) > char_budget else ""
    return [
        {
            "role": "system",
            "content": (
                "You are a strict data-quality rater for LLM pretraining corpora. "
                "Judge only the text provided. " + response_format.strip()
            ),
        },
        {
            "role": "user",
            "content": f"{rubric.strip()}\n\n--- DOCUMENT{truncated} ---\n{body}",
        },
    ]


_SCORE_RE = re.compile(r"SCORE:\s*([0-9]+)", re.IGNORECASE)
_WHY_RE = re.compile(r"WHY:\s*(.+)", re.IGNORECASE)


def parse_label(response: str, *, max_score: int = 5) -> tuple[int, str] | None:
    """(score, justification) from a model response, or None if malformed.

    Strict on the score (must match the SCORE: line, in range); lenient on the
    justification (missing WHY becomes empty string).
    """
    m = _SCORE_RE.search(response)
    if m is None:
        return None
    score = int(m.group(1))
    if not 0 <= score <= max_score:
        return None
    why = _WHY_RE.search(response)
    return score, (why.group(1).strip() if why else "")


@dataclass
class LabelRecord:
    doc_id: str  # stable id (source id or content hash)
    domain: str
    rubric_version: int
    score: int
    justification: str
    labeler: str  # model name
    source: str  # where the doc came from (hf id / file)
    # carried source fields (e.g. FineMath's own score, for rubric-vs-classifier
    # correlation checks during the pilot)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def agreement(first: list[int], second: list[int]) -> dict[str, float]:
    """Pilot double-label stability: exact, within-1, mean |Δ| (docs §3)."""
    if len(first) != len(second) or not first:
        raise ValueError("agreement needs two equal-length, non-empty label lists")
    n = len(first)
    diffs = [abs(a - b) for a, b in zip(first, second, strict=True)]
    return {
        "n": float(n),
        "exact": sum(d == 0 for d in diffs) / n,
        "within_1": sum(d <= 1 for d in diffs) / n,
        "mean_abs_diff": sum(diffs) / n,
    }


def score_histogram(scores: list[int], *, max_score: int = 5) -> dict[int, int]:
    return {s: scores.count(s) for s in range(max_score + 1)}
