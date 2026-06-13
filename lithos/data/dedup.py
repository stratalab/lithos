"""Deduplication (PRD §8.8): exact document and line dedup; MinHash deferred.

Near-dedup with MinHash (§8.8.3 / §25.2) is deferred for v0, but the pipeline is
structured so a ``MinHashDeduper`` exposing the same ``is_duplicate()`` interface
can drop in later without changing callers.
"""

from __future__ import annotations

import hashlib
from typing import Any


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ExactDocumentDeduper:
    """Drop documents whose full text has been seen before (PRD §8.8.1)."""

    def __init__(self) -> None:
        self._seen: set[str] = set()
        self.duplicates = 0

    def is_duplicate(self, text: str) -> bool:
        h = _hash(text)
        if h in self._seen:
            self.duplicates += 1
            return True
        self._seen.add(h)
        return False

    def stats(self) -> dict[str, Any]:
        return {"unique": len(self._seen), "duplicates": self.duplicates}


class ExactLineDeduper:
    """Optional global line-level dedup (PRD §8.8.2)."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def filter_lines(self, text: str) -> str:
        out: list[str] = []
        for line in text.splitlines():
            key = line.strip()
            if key and key in self._seen:
                continue
            if key:
                self._seen.add(key)
            out.append(line)
        return "\n".join(out)
