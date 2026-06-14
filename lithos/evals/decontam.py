"""N-gram decontamination (PRD §8.9): flag benchmark leakage into a text corpus.

Honest benchmark numbers require that the eval tasks' text never appeared in training.
The standard check is n-gram overlap (13-grams, à la GPT-3/The Pile): if any n-gram of
a benchmark example also occurs in the corpus, that example is "contaminated."

Implementation is streaming-friendly: build the (small) set of benchmark n-grams once,
then scan the (large) corpus once, checking membership — O(corpus) time, O(probe) memory.
The same primitive guards both the held-out perplexity set and the pretraining corpus
(wired into the data pipeline in Phase 10).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

DEFAULT_N = 13

_WORD = re.compile(r"\w+")


def _tokens(text: str) -> list[str]:
    """Lowercased word tokens — normalization that ignores whitespace/punctuation noise."""
    return _WORD.findall(text.lower())


def ngrams(text: str, n: int = DEFAULT_N) -> set[tuple[str, ...]]:
    """All word n-grams of ``text`` (empty if it has fewer than n tokens)."""
    toks = _tokens(text)
    if len(toks) < n:
        # Short example: fall back to the single full-text gram so it's still checkable.
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def build_probe_index(examples: Iterable[str], n: int = DEFAULT_N) -> dict[tuple[str, ...], set[int]]:
    """Map each benchmark n-gram -> the set of example indices it came from."""
    index: dict[tuple[str, ...], set[int]] = {}
    for i, ex in enumerate(examples):
        for gram in ngrams(ex, n):
            index.setdefault(gram, set()).add(i)
    return index


def scan_corpus(
    corpus_texts: Iterable[str],
    probe_index: dict[tuple[str, ...], set[int]],
    n: int = DEFAULT_N,
) -> set[int]:
    """Stream the corpus; return the set of probe-example indices it contaminates."""
    hit: set[int] = set()
    for doc in corpus_texts:
        for gram in ngrams(doc, n):
            owners = probe_index.get(gram)
            if owners:
                hit |= owners
    return hit


def scan_contamination(
    examples: list[str],
    corpus_texts: Iterable[str],
    *,
    n: int = DEFAULT_N,
) -> dict[str, Any]:
    """Report how many benchmark ``examples`` are contaminated by ``corpus_texts``."""
    probe_index = build_probe_index(examples, n)
    contaminated = scan_corpus(corpus_texts, probe_index, n)
    total = len(examples)
    return {
        "n": n,
        "num_examples": total,
        "num_contaminated": len(contaminated),
        "rate": (len(contaminated) / total) if total else 0.0,
        "contaminated_indices": sorted(contaminated),
    }
