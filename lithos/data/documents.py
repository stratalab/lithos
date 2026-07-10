"""Document reading and the canonical record schema (PRD §8.3-8.4).

Canonical record: ``{id, text, source, subset, language, license, tier, metadata}``.
Readers stream raw dicts from JSONL(.zst), Parquet, or an HF dataset; ``normalize``
fills defaults and drops records without usable text.

``license`` is what the rightsholder granted; ``tier`` is how the bytes reached us and
whether they may enter the weights (``lithos.data.tiers``). They are independent, and
the second is the one with teeth.
"""

from __future__ import annotations

import contextlib
import glob
import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import zstandard
from pydantic import BaseModel, ConfigDict, Field

from lithos.data.tiers import TIER_UNKNOWN, Tier


class DocumentSource(BaseModel):
    """Where a slice of documents comes from, plus its provenance defaults."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["jsonl", "parquet", "hf"] = "jsonl"
    paths: list[str] = Field(default_factory=list)
    dataset: str | None = None
    config_name: str | None = None
    split: str = "train"
    text_field: str = "text"
    quality_field: str | None = None  # raw field carrying a precomputed quality score (e.g. "score")
    source_name: str = "unknown"
    subset: str | None = None
    language: str = "en"
    license: str = "unknown"
    # Acquisition tier. Undeclared -> "unknown" -> barred from the weights (fail-closed).
    # Pydantic's Literal rejects a typo'd tier at config-load time, not mid-build.
    tier: Tier = TIER_UNKNOWN
    limit: int | None = None


def normalize(
    record: dict[str, Any],
    *,
    source: str,
    subset: str | None,
    language: str,
    license: str,
    tier: str = TIER_UNKNOWN,
    text_field: str = "text",
    quality_field: str | None = None,
) -> dict[str, Any] | None:
    """Coerce a raw record into the canonical schema; return None if no text."""
    text = record.get(text_field)
    if not isinstance(text, str) or not text:
        return None
    doc = {
        "id": str(record.get("id", "")),
        "text": text,
        "source": record.get("source", source),
        "subset": record.get("subset", subset),
        "language": record.get("language", language),
        "license": record.get("license", license),
        "tier": record.get("tier", tier),
        "metadata": record.get("metadata", {}),
    }
    if quality_field is not None:
        doc["quality_score"] = record.get(quality_field)
    return doc


@contextlib.contextmanager
def _open_text(path: str | Path) -> Iterator[io.TextIOBase]:
    p = Path(path)
    if p.suffix == ".zst":
        with open(p, "rb") as fh:
            reader = zstandard.ZstdDecompressor().stream_reader(fh)
            yield io.TextIOWrapper(reader, encoding="utf-8")
    else:
        with open(p, encoding="utf-8") as f:
            yield f


def _expand_paths(paths: list[str]) -> list[str]:
    """Expand glob patterns (``*?[``); literal paths pass through. Sorted for
    deterministic shard order. A pattern that matches nothing is an error — a
    silently-empty source would corrupt a corpus build without warning."""
    out: list[str] = []
    for p in paths:
        if any(c in p for c in "*?["):
            matched = sorted(glob.glob(p, recursive=True))
            if not matched:
                raise FileNotFoundError(f"no files match pattern: {p}")
            out.extend(matched)
        else:
            out.append(p)
    return out


def read_jsonl(paths: list[str]) -> Iterator[dict[str, Any]]:
    for path in _expand_paths(paths):
        with _open_text(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def read_parquet(paths: list[str]) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    for path in _expand_paths(paths):
        table = pq.read_table(path)
        for batch in table.to_batches():
            yield from batch.to_pylist()


def iter_documents(source: DocumentSource) -> Iterator[dict[str, Any]]:
    """Stream normalized documents from a source, honoring ``limit``."""
    if source.kind == "jsonl":
        raw: Iterator[dict[str, Any]] = read_jsonl(source.paths)
    elif source.kind == "parquet":
        raw = read_parquet(source.paths)
    else:  # hf
        from datasets import load_dataset

        if not source.dataset:
            raise ValueError("source.kind='hf' requires source.dataset.")
        raw = iter(
            load_dataset(
                source.dataset, name=source.config_name, split=source.split, streaming=True
            )
        )

    n = 0
    for record in raw:
        doc = normalize(
            record,
            source=source.source_name,
            subset=source.subset,
            language=source.language,
            license=source.license,
            tier=source.tier,
            text_field=source.text_field,
            quality_field=source.quality_field,
        )
        if doc is None:
            continue
        yield doc
        n += 1
        if source.limit is not None and n >= source.limit:
            return
