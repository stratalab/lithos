"""Corpus manifest builder (PRD §8.6).

Shard manifest entries (PRD §8.5) are produced by ``ShardWriter``; this assembles
the top-level corpus manifest that ties shards, sources, filters, dedup, and
license/provenance notes together for reproducibility.
"""

from __future__ import annotations

import datetime as dt
from typing import Any


def corpus_manifest(
    *,
    name: str,
    version: str,
    tokenizer: str,
    num_documents: int,
    num_tokens: int,
    sources: list[str],
    mixture: dict[str, int],
    filters: dict[str, Any],
    dedup: dict[str, Any],
    shards: list[dict[str, Any]],
    license_notes: list[str],
    decontamination: dict[str, Any] | None = None,
    tiers: dict[str, Any] | None = None,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    created = (now or dt.datetime.now(dt.UTC)).strftime("%Y-%m-%d")
    return {
        "corpus_name": name,
        "version": version,
        "created_at": created,
        "tokenizer": tokenizer,
        "num_documents": num_documents,
        "num_tokens": num_tokens,
        "sources": sources,
        "mixture": mixture,
        "filters": filters,
        "dedup": dedup,
        "decontamination": decontamination or {},
        # Acquisition-tier attestation (lithos.data.tiers): what entered the weights.
        "tiers": tiers or {},
        "license_notes": license_notes,
        "shards": shards,
    }
