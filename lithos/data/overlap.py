"""Cross-corpus overlap estimation from streamed samples (doc §1.8 math-slice warning).

Multiple corpora mined from the same source (four math corpora all mine Common
Crawl) overlap unknown amounts; naively concatenating them silently multi-epochs
the intersection, corrupting both mix weights and epoch-cap accounting. This
module estimates the pairwise overlap matrix from *samples*, so the question is
answered before any bulk download.

The statistics (the part that is easy to get wrong): with samples from BOTH
sides, a duplicated document's counterpart appears in the other sample only
with probability n_B/N_B. Raw cross-sample match counts therefore understate
true overlap by that factor, and the estimator inverts it:

    overlap(A→B) ≈ matched_A / (n_A · n_B / N_B)

where matched_A = number of sample-A docs with ≥1 match in sample B. Assumes
roughly uniform samples (use a shuffled stream) and ~1 counterpart per dup
(clusters inflate the estimate — treat results as an upper-ish bound). With
n=200k each and true overlap ≥10%, expected matches are in the hundreds —
measurable; single-digit match counts mean wide error bars, say so in reports.

Three matchers, strongest-signal first: exact URL (where both corpora carry
one), exact normalized-text hash, MinHash-LSH near-dup (reuses data/minhash.py).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from lithos.data.minhash import MinHasher, estimate_jaccard


class CorpusSpec(BaseModel):
    """One corpus to sample. `total_docs` drives the estimator inversion."""

    model_config = ConfigDict(extra="forbid")

    name: str
    hf_id: str
    config: str | None = None  # None = single-config dataset
    config_prefer: str | None = None  # substring to pick among configs (e.g. "web")
    data_dir: str | None = None  # subdirectory datasets (e.g. MegaMath's megamath-web)
    split: str = "train"
    text_field: str | None = None  # None = auto-detect from TEXT_FIELD_CANDIDATES
    url_field: str | None = None
    total_docs: int | None = None  # None = fetch from datasets-server at runtime
    total_docs_approx: bool = False  # True = N is an estimate; flag in the report


TEXT_FIELD_CANDIDATES = ("text", "content", "markdown", "body", "code")
URL_FIELD_CANDIDATES = ("url", "URL", "uri", "source_url", "metadata.url")


def get_field(doc: dict, dotted: str):
    """Fetch a possibly-nested field ("metadata.url")."""
    cur: Any = doc
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur

_WS_RE = re.compile(r"\s+")


def normalize_text_hash(text: str) -> int:
    """Exact-dup hash, whitespace/case-insensitive (64-bit)."""
    canon = _WS_RE.sub(" ", text.strip().lower())
    return int.from_bytes(hashlib.sha1(canon.encode()).digest()[:8], "big")


def normalize_url(url: str) -> int | None:
    """URL identity hash: scheme/www/trailing-slash/fragment-insensitive."""
    u = url.strip().lower()
    if not u:
        return None
    u = re.sub(r"^https?://", "", u)
    u = re.sub(r"^www\.", "", u)
    u = u.split("#", 1)[0].rstrip("/")
    if not u:
        return None
    return int.from_bytes(hashlib.sha1(u.encode()).digest()[:8], "big")


@dataclass
class SampleSigs:
    """MinHash signatures + identity hashes for one corpus sample."""

    name: str
    total_docs: int
    sigs: np.ndarray  # (n, num_perm) uint64
    text_hashes: np.ndarray  # (n,) uint64
    url_hashes: np.ndarray | None = None  # (n,) uint64, or None if no url field

    @property
    def n(self) -> int:
        return int(self.sigs.shape[0])


def build_sample(
    name: str,
    docs: Iterable[dict],
    *,
    total_docs: int,
    sample_size: int,
    text_field: str,
    url_field: str | None,
    hasher: MinHasher | None = None,
    min_chars: int = 200,
) -> SampleSigs:
    """Consume up to `sample_size` docs from an (ideally shuffled) stream."""
    hasher = hasher or MinHasher()
    sigs: list[np.ndarray] = []
    t_hashes: list[int] = []
    u_hashes: list[int] = []
    for doc in docs:
        text = get_field(doc, text_field) or ""
        if len(text) < min_chars:  # tiny docs make degenerate signatures
            continue
        sigs.append(hasher.signature(text))
        t_hashes.append(normalize_text_hash(text))
        if url_field is not None:
            u_hashes.append(normalize_url(str(get_field(doc, url_field) or "")) or 0)
        if len(sigs) >= sample_size:
            break
    if not sigs:
        raise ValueError(f"{name}: no usable documents sampled")
    return SampleSigs(
        name=name,
        total_docs=total_docs,
        sigs=np.stack(sigs),
        text_hashes=np.asarray(t_hashes, dtype=np.uint64),
        url_hashes=np.asarray(u_hashes, dtype=np.uint64) if url_field is not None else None,
    )


# ---------------------------------------------------------------------------
# Pairwise matching
# ---------------------------------------------------------------------------


@dataclass
class PairResult:
    a: str
    b: str
    n_a: int
    n_b: int
    total_a: int
    total_b: int
    url_matched_a: int | None  # docs of sample A with an exact-URL match in sample B
    url_matched_b: int | None
    text_matched_a: int = 0
    text_matched_b: int = 0
    near_matched_a: int = 0  # includes text matches (near-dup subsumes exact)
    near_matched_b: int = 0
    notes: list[str] = field(default_factory=list)

    def estimate(self, matched_a: int, matched_b: int) -> tuple[float, float]:
        """(overlap A→B, overlap B→A) via the counterpart-inclusion inversion."""
        exp_a = self.n_a * self.n_b / self.total_b  # expected matches if overlap were 100%
        exp_b = self.n_b * self.n_a / self.total_a
        return min(matched_a / exp_a, 1.0) if exp_a else 0.0, (
            min(matched_b / exp_b, 1.0) if exp_b else 0.0
        )


def _match_hashes(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    """(#a-values present in b, #b-values present in a), ignoring 0 sentinels."""
    common = np.intersect1d(a[a != 0], b[b != 0])
    if common.size == 0:
        return 0, 0
    return int(np.isin(a, common).sum()), int(np.isin(b, common).sum())


def _lsh_cross_match(
    a: SampleSigs, b: SampleSigs, *, bands: int, threshold: float
) -> tuple[int, int]:
    """(#a-docs with a near-dup in b, #b-docs with a near-dup in a)."""
    num_perm = a.sigs.shape[1]
    rows = num_perm // bands
    buckets: dict[bytes, list[int]] = {}
    for j in range(b.n):
        sig = b.sigs[j]
        for band in range(bands):
            buckets.setdefault(sig[band * rows : (band + 1) * rows].tobytes(), []).append(j)
    matched_a: set[int] = set()
    matched_b: set[int] = set()
    for i in range(a.n):
        sig = a.sigs[i]
        cands: set[int] = set()
        for band in range(bands):
            cands.update(buckets.get(sig[band * rows : (band + 1) * rows].tobytes(), ()))
        for j in cands:
            if estimate_jaccard(sig, b.sigs[j]) >= threshold:
                matched_a.add(i)
                matched_b.add(j)
    return len(matched_a), len(matched_b)


def pair_overlap(
    a: SampleSigs, b: SampleSigs, *, bands: int = 16, threshold: float = 0.8
) -> PairResult:
    res = PairResult(
        a=a.name, b=b.name, n_a=a.n, n_b=b.n, total_a=a.total_docs, total_b=b.total_docs,
        url_matched_a=None, url_matched_b=None,
    )
    if a.url_hashes is not None and b.url_hashes is not None:
        res.url_matched_a, res.url_matched_b = _match_hashes(a.url_hashes, b.url_hashes)
    else:
        res.notes.append("no shared url field — url overlap skipped")
    res.text_matched_a, res.text_matched_b = _match_hashes(
        a.text_hashes.astype(np.uint64), b.text_hashes.astype(np.uint64)
    )
    res.near_matched_a, res.near_matched_b = _lsh_cross_match(
        a, b, bands=bands, threshold=threshold
    )
    low = max(res.near_matched_a, res.url_matched_a or 0)
    if 0 < low < 30:
        res.notes.append(f"only {low} matches — wide error bars, treat as noisy")
    return res


def format_report(results: list[PairResult]) -> str:
    """Markdown report: one row per direction, all three matchers."""
    lines = [
        "# Math-corpus overlap matrix (sample-based estimates)",
        "",
        "`overlap(A→B)` = est. fraction of A's documents with a duplicate in B.",
        "Estimator inverts counterpart-inclusion probability (see lithos/data/overlap.py);",
        "assumes shuffled samples and ~1 counterpart per dup — read as upper-ish bounds.",
        "",
        "| A → B | n_A | N_B | url | exact-text | near-dup | raw near matches | notes |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def pct(v: float | None) -> str:
        return "—" if v is None else f"{v:.1%}"

    for r in results:
        url_ab, url_ba = (None, None)
        if r.url_matched_a is not None:
            url_ab, url_ba = r.estimate(r.url_matched_a, r.url_matched_b or 0)
        text_ab, text_ba = r.estimate(r.text_matched_a, r.text_matched_b)
        near_ab, near_ba = r.estimate(r.near_matched_a, r.near_matched_b)
        notes = "; ".join(r.notes)
        lines.append(
            f"| {r.a} → {r.b} | {r.n_a} | {r.total_b:,} | {pct(url_ab)} | "
            f"{pct(text_ab)} | {pct(near_ab)} | {r.near_matched_a} | {notes} |"
        )
        lines.append(
            f"| {r.b} → {r.a} | {r.n_b} | {r.total_a:,} | {pct(url_ba)} | "
            f"{pct(text_ba)} | {pct(near_ba)} | {r.near_matched_b} | {notes} |"
        )
    return "\n".join(lines) + "\n"


def iter_pairs(samples: list[SampleSigs]) -> Iterator[tuple[SampleSigs, SampleSigs]]:
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            yield samples[i], samples[j]
