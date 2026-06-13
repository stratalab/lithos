"""Train a byte-level BPE tokenizer and write reproducible artifacts (PRD §7.2).

Outputs (written to the tokenizer output directory):
- ``tokenizer.json``           — the HF tokenizers model
- ``tokenizer_config.json``    — the Lithos TokenizerConfig
- ``tokenizer_manifest.json``  — provenance/training manifest (PRD §7.2)
- ``sample_report.json``       — example tokenizations
"""

from __future__ import annotations

import datetime as dt
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer, pre_tokenizers, trainers

from lithos.tokenizer.tokenizer_config import TokenizerConfig, build_tokenizer
from lithos.utils.io import ensure_dir, write_json


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    return out.stdout.strip() or None


def train_tokenizer(cfg: TokenizerConfig, texts: Iterable[str]) -> tuple[Tokenizer, dict[str, int]]:
    """Train a tokenizer from an iterable of text; returns (tokenizer, corpus stats)."""
    stats = {"num_documents": 0, "approx_chars": 0}

    def counted() -> Iterable[str]:
        for text in texts:
            stats["num_documents"] += 1
            stats["approx_chars"] += len(text)
            yield text

    tok = build_tokenizer(cfg)
    trainer = trainers.BpeTrainer(
        vocab_size=cfg.vocab_size,
        min_frequency=cfg.min_frequency,
        special_tokens=cfg.special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(counted(), trainer=trainer)
    return tok, stats


def build_manifest(
    cfg: TokenizerConfig,
    stats: dict[str, int],
    sources: list[str],
    *,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """Assemble the tokenizer training manifest (PRD §7.2)."""
    created = now or dt.datetime.now(dt.UTC)
    return {
        "name": cfg.full_name,
        "tokenizer_type": "byte-level BPE",
        "sources": sources,
        "num_documents": stats["num_documents"],
        "approx_chars": stats["approx_chars"],
        "vocab_size": cfg.vocab_size,
        "special_tokens": cfg.special_tokens,
        "normalization": "none (byte-level)",
        "pre_tokenization": {
            "individual_digits": cfg.individual_digits,
            "byte_level": True,
            "add_prefix_space": cfg.add_prefix_space,
            "use_regex": True,
        },
        "created_at": created.strftime("%Y-%m-%d"),
        "git_commit": _git_commit(),
    }


def sample_report(tok: Tokenizer, samples: list[str]) -> dict[str, Any]:
    """Tokenize example strings for a human-readable sanity report."""
    rows = []
    for text in samples:
        enc = tok.encode(text)
        rows.append({"text": text, "n_tokens": len(enc.ids), "tokens": enc.tokens[:64]})
    return {"samples": rows}


def save_tokenizer(
    tok: Tokenizer,
    cfg: TokenizerConfig,
    output_dir: str | Path,
    manifest: dict[str, Any],
    report: dict[str, Any],
) -> Path:
    """Write the tokenizer and its config/manifest/report to ``output_dir``."""
    out = ensure_dir(output_dir)
    tok.save(str(out / "tokenizer.json"))
    write_json(out / "tokenizer_config.json", cfg.model_dump())
    write_json(out / "tokenizer_manifest.json", manifest)
    write_json(out / "sample_report.json", report)
    return out
