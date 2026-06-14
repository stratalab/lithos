"""N-gram decontamination (PRD §8.9): flag/drop benchmark leakage from a text corpus.

Honest benchmark numbers require that the eval tasks' text never appeared in training.
The standard check is n-gram overlap (13-grams, à la GPT-3/The Pile): if any n-gram of
a benchmark example also occurs in a document, that document is "contaminated."

- ``scan_contamination`` (batch): how many benchmark examples a corpus contaminates.
- ``DecontaminationFilter`` (streaming): per-document ``is_contaminated(text)`` for the
  corpus build — drops training docs that overlap the eval battery.
- ``load_benchmark_probes``: best-effort extraction of the battery's test-set text.

Lives in ``data`` (it analyzes corpus text); ``evals`` imports it, not vice-versa.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
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


class DecontaminationFilter:
    """Streaming per-document benchmark-contamination check (drops into the corpus build).

    Builds the (small) set of benchmark n-grams once, then for each document checks
    membership and short-circuits on the first hit — O(doc) per document.
    """

    def __init__(self, benchmark_texts: Iterable[str], *, n: int = DEFAULT_N) -> None:
        self.n = n
        self._probe: set[tuple[str, ...]] = set()
        self.num_probes = 0
        for ex in benchmark_texts:
            self._probe |= ngrams(ex, n)
            self.num_probes += 1
        self.contaminated = 0

    def is_contaminated(self, text: str) -> bool:
        for gram in ngrams(text, self.n):
            if gram in self._probe:
                self.contaminated += 1
                return True
        return False

    def stats(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "probe_examples": self.num_probes,
            "probe_ngrams": len(self._probe),
            "contaminated_docs": self.contaminated,
        }


# Per-task probe extractors for the frozen v1 battery: (hf_path, config, split, text_field).
_BATTERY_PROBE_SPECS: dict[str, tuple[str, str | None, str, str]] = {
    "hellaswag": ("Rowan/hellaswag", None, "validation", "ctx"),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test", "question"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test", "question"),
    # NOTE: ybisk/piqa is script-based (unsupported by datasets>=3), so it is
    # best-effort-skipped until a parquet mirror is wired; 7/8 tasks still decontaminate.
    "piqa": ("ybisk/piqa", None, "validation", "goal"),
    "winogrande": ("allenai/winogrande", "winogrande_xl", "validation", "sentence"),
    "lambada_openai": ("EleutherAI/lambada_openai", None, "test", "text"),
    "sciq": ("allenai/sciq", None, "test", "question"),
    "openbookqa": ("allenai/openbookqa", "main", "test", "question_stem"),
}


def load_benchmark_probes(tasks: list[str], *, limit: int | None = None) -> list[str]:
    """Best-effort: pull benchmark test-example texts for the battery (for decontamination).

    Tasks whose dataset can't be loaded are warned and skipped — never crash a build.
    Generate once and persist with ``write_probes`` so the corpus build just reads the file.
    """
    from datasets import load_dataset  # lazy, heavy

    texts: list[str] = []
    for task in tasks:
        spec = _BATTERY_PROBE_SPECS.get(task)
        if spec is None:
            print(f"[decontam] no probe spec for task {task!r}; skipping")
            continue
        path, config, split, field = spec
        try:
            ds = load_dataset(path, config, split=split)
            n_before = len(texts)
            for i, row in enumerate(ds):
                if limit is not None and i >= limit:
                    break
                value = row.get(field)
                if isinstance(value, str) and value:
                    texts.append(value)
            print(f"[decontam] {task}: +{len(texts) - n_before} probe texts")
        except Exception as e:  # pragma: no cover - dataset/network variability
            print(f"[decontam] WARNING: could not load probes for {task}: {e}")
    return texts


def write_probes(path: str | Path, texts: Iterable[str]) -> Path:
    """Persist probe texts as JSONL (one {"text": ...} per line)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    return p


def read_probes(path: str | Path) -> list[str]:
    """Read probe texts written by ``write_probes``."""
    p = Path(path)
    return [json.loads(line)["text"] for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
