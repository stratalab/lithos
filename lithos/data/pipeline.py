"""Corpus build pipeline: documents -> filter -> dedup -> tokenize -> shards.

Ties together the §8.2 data stages and writes a reproducible corpus manifest
(PRD §8.6). Driven by ``scripts/tokenize_corpus.py``.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lithos.data.dedup import ExactDocumentDeduper
from lithos.data.documents import DocumentSource, iter_documents
from lithos.data.filters import DocumentFilter, FilterConfig
from lithos.data.manifest import corpus_manifest
from lithos.data.minhash import MinHashConfig, MinHashDeduper
from lithos.data.shard import ShardWriter, dtype_for_vocab
from lithos.data.tokenize import DocumentTokenizer
from lithos.tokenizer.inspect_tokenizer import load_tokenizer
from lithos.utils.io import ensure_dir, write_json


class CorpusBuildConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "lithos-corpus"
    version: str = "v0.1"
    tokenizer_path: str
    sources: list[DocumentSource]
    output_dir: str
    seq_len: int = 1024
    tokens_per_shard: int = 1_000_000
    add_bos: bool = True
    add_eos: bool = True
    exact_dedup: bool = True
    near_dedup: bool = False  # MinHash/LSH near-dedup (Phase 10); off by default
    minhash: MinHashConfig = Field(default_factory=MinHashConfig)
    filters: FilterConfig = Field(default_factory=FilterConfig)
    license_notes: list[str] = Field(default_factory=list)


def build_corpus(cfg: CorpusBuildConfig, *, now: Any = None) -> dict[str, Any]:
    """Run the full pipeline and write shards + corpus manifest; return the manifest."""
    out = ensure_dir(cfg.output_dir)
    tokenizer = load_tokenizer(cfg.tokenizer_path)
    tokenizer_name = Path(cfg.tokenizer_path).parent.name
    doctok = DocumentTokenizer.from_tokenizer(tokenizer, add_bos=cfg.add_bos, add_eos=cfg.add_eos)

    filt = DocumentFilter(cfg.filters)
    dedup = ExactDocumentDeduper() if cfg.exact_dedup else None
    near = MinHashDeduper(cfg.minhash) if cfg.near_dedup else None
    writer = ShardWriter(
        out / "tokenized",
        tokens_per_shard=cfg.tokens_per_shard,
        dtype=dtype_for_vocab(tokenizer.get_vocab_size()),
        tokenizer_name=tokenizer_name,
        rel_base=out,  # store shard paths relative to the corpus dir (portable)
    )

    mixture: Counter[str] = Counter()
    n_docs = 0
    for source in cfg.sources:
        for doc in iter_documents(source):
            if not filt.keep(doc):
                continue
            if dedup is not None and dedup.is_duplicate(doc["text"]):
                continue
            if near is not None and near.is_duplicate(doc["text"]):
                continue
            writer.add(doctok.encode(doc["text"]))
            mixture[doc["source"]] += 1
            n_docs += 1
    shards = writer.close()

    manifest = corpus_manifest(
        name=cfg.name,
        version=cfg.version,
        tokenizer=tokenizer_name,
        num_documents=n_docs,
        num_tokens=writer.total_tokens,
        sources=[s.source_name for s in cfg.sources],
        mixture=dict(mixture),
        filters={"config": cfg.filters.model_dump(), "stats": filt.stats()},
        dedup={
            "exact": dedup.stats() if dedup is not None else {},
            "near": near.stats() if near is not None else {},
        },
        shards=shards,
        license_notes=cfg.license_notes,
        now=now,
    )
    write_json(out / "corpus_manifest.json", manifest)
    return manifest
