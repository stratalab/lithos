"""Quality filtering via an existing classifier score (Phase 10).

"Use the existing scorer to begin with": FineWeb-Edu already carries the edu-quality
classifier's per-document ``score`` (0-5), so quality filtering is just reading that
score and keeping documents above a threshold — *zero inference*. A ``DocumentSource``
copies the raw score field into ``doc["quality_score"]`` (see documents.py); this filter
thresholds it. Running a classifier *ourselves* over unscored / synthetic data — labeling
a sample and training a cheap classifier — is the deferred next step (keeps the moving
pieces down for now).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class QualityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    threshold: float = 3.0  # keep documents with quality_score >= threshold
    missing_score: float | None = None  # assumed score when a doc carries none (None -> drop)


class QualityFilter:
    """Keep documents whose carried ``quality_score`` clears a threshold."""

    def __init__(self, cfg: QualityConfig) -> None:
        self.cfg = cfg
        self.kept = 0
        self.dropped = 0
        self.missing = 0

    def keep(self, doc: dict[str, Any]) -> bool:
        score = doc.get("quality_score")
        if score is None:
            self.missing += 1
            if self.cfg.missing_score is None:
                self.dropped += 1
                return False
            score = self.cfg.missing_score
        if float(score) >= self.cfg.threshold:
            self.kept += 1
            return True
        self.dropped += 1
        return False

    def stats(self) -> dict[str, Any]:
        return {
            "threshold": self.cfg.threshold,
            "kept": self.kept,
            "dropped": self.dropped,
            "missing_score": self.missing,
        }
