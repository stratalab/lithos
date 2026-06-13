"""Document quality filters with drop accounting (PRD §8.7).

Filters never silently delete: every drop is counted by reason so the corpus
manifest can record exactly what filtering did.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, ConfigDict


class FilterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_chars: int = 1
    max_chars: int = 500_000
    allowed_languages: list[str] | None = None  # None -> allow all
    max_repeated_char_run: int = 100
    max_duplicate_line_fraction: float = 0.5
    max_symbol_fraction: float = 0.5


def _longest_char_run(text: str) -> int:
    longest = run = 0
    prev = ""
    for ch in text:
        run = run + 1 if ch == prev else 1
        prev = ch
        longest = max(longest, run)
    return longest


def _duplicate_line_fraction(text: str) -> float:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    return 1.0 - len(set(lines)) / len(lines)


def _symbol_fraction(text: str) -> float:
    if not text:
        return 0.0
    n_symbol = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    return n_symbol / len(text)


def check_document(text: str, cfg: FilterConfig) -> str | None:
    """Return a drop-reason for ``text``, or None if it passes (PRD §8.7)."""
    if not text or not text.strip():
        return "empty"
    n = len(text)
    if n < cfg.min_chars:
        return "too_short"
    if n > cfg.max_chars:
        return "too_long"
    if _longest_char_run(text) > cfg.max_repeated_char_run:
        return "repeated_chars"
    if _duplicate_line_fraction(text) > cfg.max_duplicate_line_fraction:
        return "duplicate_lines"
    if _symbol_fraction(text) > cfg.max_symbol_fraction:
        return "symbol_density"
    return None


def check_language(doc: dict[str, Any], cfg: FilterConfig) -> str | None:
    if cfg.allowed_languages is None:
        return None
    return None if doc.get("language") in cfg.allowed_languages else "language"


class DocumentFilter:
    """Apply all filters to a document stream, tallying drops by reason."""

    def __init__(self, cfg: FilterConfig) -> None:
        self.cfg = cfg
        self.kept = 0
        self.dropped: Counter[str] = Counter()

    def keep(self, doc: dict[str, Any]) -> bool:
        reason = check_language(doc, self.cfg) or check_document(doc["text"], self.cfg)
        if reason:
            self.dropped[reason] += 1
            return False
        self.kept += 1
        return True

    def stats(self) -> dict[str, Any]:
        return {"kept": self.kept, "dropped": dict(self.dropped)}
