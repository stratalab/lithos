"""Decontamination gate for post-training data (epic F2, Phase 12).

Post-training data is the *most* contamination-dense input we handle:
OpenMathReasoning ← AoPS ← MATH/AIME, so an eval problem can arrive verbatim in
an SFT trace. This wraps the corpus pipeline's 13-gram ``DecontaminationFilter``
so every converter (SFT ingest, DPO prefs, distillation) screens its output
against the frozen eval battery before writing — the cheap insurance that
protects every parity claim (`docs/eval-plan.md` principle 3).

Reuses `lithos/data/decontam.py` (same n-gram machinery as the corpus build), so
training and post-training share one contamination definition.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from lithos.data.decontam import DecontaminationFilter, read_probes


def _turn_text(msg: dict[str, Any]) -> str:
    """All screenable text of one turn — flat ``content`` or TIR ``segments``
    (think/text/tool/tool_result), so a leak in an assistant trace can't slip past.

    Best-effort: this runs *before* the renderer validates, so a malformed segment
    must not crash the build — non-dict segments are stringified, not rejected here.
    """
    if "segments" in msg:
        parts = [
            str(seg.get("text") or seg.get("code") or seg.get("output") or "")
            if isinstance(seg, dict)
            else str(seg)
            for seg in msg["segments"]
        ]
        return "\n".join(parts)
    return str(msg.get("content", ""))


def messages_text(record: dict[str, Any]) -> str:
    """Screenable text of an SFT record ``{"messages": [...]}`` — all turn text
    joined (a leaked problem shows up in the user prompt *or* an assistant trace)."""
    return "\n".join(_turn_text(m) for m in record.get("messages", []))


def prefs_text(record: dict[str, Any]) -> str:
    """Screenable text of a DPO record ``{"prompt", "chosen", "rejected"}``."""
    prompt = "\n".join(m.get("content", "") for m in record.get("prompt", []))
    return f"{prompt}\n{record.get('chosen', '')}\n{record.get('rejected', '')}"


class PostTrainDecontaminator:
    """Screens post-training records against benchmark probes; drops any that carry
    a 13-gram overlap with the eval battery, with a report of what was cut."""

    def __init__(self, probes: Iterable[str], *, n: int = 13) -> None:
        self._filter = DecontaminationFilter(probes, n=n)
        self.read = 0
        self.dropped = 0

    @classmethod
    def from_probe_file(cls, path: str | Path, *, n: int = 13) -> PostTrainDecontaminator:
        """Build from a persisted probe JSONL (written by ``decontam.write_probes``);
        no network — the recommended path for reproducible converter runs."""
        return cls(read_probes(path), n=n)

    def is_clean(self, text: str) -> bool:
        return not self._filter.is_contaminated(text)

    def screen(
        self, records: Iterable[dict[str, Any]], text_of: Callable[[dict[str, Any]], str]
    ) -> list[dict[str, Any]]:
        """Return only the clean records; ``text_of`` extracts the text to check.
        Updates ``read``/``dropped`` for the report."""
        kept: list[dict[str, Any]] = []
        for rec in records:
            self.read += 1
            if self.is_clean(text_of(rec)):
                kept.append(rec)
            else:
                self.dropped += 1
        return kept

    def report(self) -> dict[str, Any]:
        return {
            "read": self.read,
            "dropped": self.dropped,
            "kept": self.read - self.dropped,
            "drop_rate": round(self.dropped / self.read, 4) if self.read else 0.0,
            **self._filter.stats(),
        }
