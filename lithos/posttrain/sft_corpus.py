"""Offline SFT-corpus build: render → decontam → mix → packed dual-stream shards (E2).

Mirrors the pretraining build (`lithos/data/pipeline.py::build_corpus`) but for
instruction data: each conversation is rendered with the chat template into
``(input_ids, loss_mask)``, screened against the eval battery (F2), blended across
sources at controlled ratios, and packed into fixed-size shards that carry BOTH a
token stream and a per-token loss-mask stream. The packed loader
(`PackedSFTDataset`) memory-maps these instead of holding a dense padded array in
RAM, and packing removes the per-conversation padding that dominated the old path.

Packing uses cross-document attention **bleed** — the established pretraining
convention (`lithos/data/packing.py`): conversations are concatenated, the loss
mask isolates training targets, and the per-conversation ``<bos>`` signals the
boundary. No model/attention changes (block-diagonal masking is the PRD §27
deferred flag for pretrain and SFT alike).
"""

from __future__ import annotations

import datetime as dt
import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from lithos.data.shard import dtype_for_vocab
from lithos.data.tiers import TIER_UNKNOWN, Tier, assert_prompt_source, assert_trainable
from lithos.posttrain.chat_template import render_conversation
from lithos.posttrain.decontam_gate import PostTrainDecontaminator, messages_text
from lithos.tokenizer.inspect_tokenizer import load_tokenizer
from lithos.utils.io import ensure_dir, sha256_file, write_json

# (tokens_path, mask_path, num_tokens, dtype) — parallel to data.shard.ShardSpec.
SFTShardSpec = tuple[str, str, int, str]


class SFTSourceSpec(BaseModel):
    """One instruction source in the blend. ``max_examples`` caps a giant set;
    ``repeats`` upsamples a small excellent one (the LIMA-in-reverse guard)."""

    model_config = ConfigDict(extra="forbid")

    path: str  # messages-JSONL ({"messages": [{"role","content"}, ...]})
    name: str
    # Acquisition tier of the **assistant targets** — the only spans that receive a
    # gradient (`lithos.data.tiers`). Fail-closed.
    tier: Tier = TIER_UNKNOWN
    # Acquisition tier of the **prompts**, which the loss mask zeroes. May be
    # `restricted`: a textbook problem statement is a stimulus, never a target, so the
    # model is never trained to reproduce it. Defaults to `tier`.
    prompt_tier: Tier | None = None
    # Required when tier='synthetic-verified': the source_ids this was derived from.
    grounded_on: list[str] = Field(default_factory=list)
    max_examples: int | None = Field(default=None, gt=0)  # None = no cap
    repeats: int = Field(default=1, ge=1)  # >=1; 0 would silently drop the source

    def target_tier(self) -> str:
        return self.tier

    def masked_prompt_tier(self) -> str:
        return self.prompt_tier or self.tier


class SFTCorpusBuildConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "lithos-sft"
    version: str = "v0.1"
    tokenizer_path: str
    sources: list[SFTSourceSpec] = Field(min_length=1)
    output_dir: str
    seq_len: int = Field(default=2048, gt=0)
    tokens_per_shard: int = Field(default=1_000_000, gt=0)
    add_bos: bool = True
    seed: int = 0
    # divert this fraction of each source to a disjoint val set; <1 so train is non-empty
    val_fraction: float = Field(default=0.0, ge=0.0, lt=1.0)
    decontam_probes: str | None = None  # probe JSONL (decontam.write_probes) for F2 screening
    enforce_tiers: bool = True  # acquisition gate; disabling is deliberate and auditable
    license_notes: list[str] = Field(default_factory=list)


class SFTShardWriter:
    """Accumulate rendered conversations and flush fixed-size dual-stream shards:
    ``shard_NNNNNN.tokens.bin`` (token ids) + ``shard_NNNNNN.mask.bin`` (uint8, 1 =
    loss target). The two files are chopped at identical boundaries, so a window
    carved from one aligns with the other. Reuses shard.py's buffer/flush/sha256
    mechanics, extended to two channels.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        tokens_per_shard: int,
        dtype: str,
        tokenizer_name: str,
        rel_base: str | Path | None = None,
    ) -> None:
        self.dir = ensure_dir(output_dir)
        self.tokens_per_shard = tokens_per_shard
        self.dtype = np.dtype(dtype)
        self.tokenizer_name = tokenizer_name
        self.rel_base = Path(rel_base) if rel_base is not None else None
        self._tok_buf: list[int] = []
        self._mask_buf: list[int] = []
        self._shard_idx = 0
        self.total_tokens = 0
        self.total_loss_tokens = 0
        self.shards: list[dict[str, Any]] = []

    def add(self, token_ids: list[int], loss_mask: list[bool]) -> None:
        if len(token_ids) != len(loss_mask):
            raise ValueError(f"token/mask length mismatch: {len(token_ids)} != {len(loss_mask)}")
        self._tok_buf.extend(token_ids)
        self._mask_buf.extend(1 if m else 0 for m in loss_mask)
        while len(self._tok_buf) >= self.tokens_per_shard:
            self._write(
                self._tok_buf[: self.tokens_per_shard], self._mask_buf[: self.tokens_per_shard]
            )
            del self._tok_buf[: self.tokens_per_shard]
            del self._mask_buf[: self.tokens_per_shard]

    def _rel(self, path: Path) -> str:
        return str(path.relative_to(self.rel_base)) if self.rel_base is not None else str(path)

    def _write(self, ids: list[int], mask: list[int]) -> None:
        self._shard_idx += 1
        shard_id = f"shard_{self._shard_idx:06d}"
        tok_path = self.dir / f"{shard_id}.tokens.bin"
        mask_path = self.dir / f"{shard_id}.mask.bin"
        tok_arr = np.asarray(ids, dtype=self.dtype)
        mask_arr = np.asarray(mask, dtype=np.uint8)
        tok_arr.tofile(tok_path)
        mask_arr.tofile(mask_path)
        self.total_tokens += int(tok_arr.size)
        self.total_loss_tokens += int(mask_arr.sum())
        self.shards.append(
            {
                "shard_id": shard_id,
                "tokens_path": self._rel(tok_path),
                "mask_path": self._rel(mask_path),
                "num_tokens": int(tok_arr.size),
                "dtype": self.dtype.name,
                "tokenizer": self.tokenizer_name,
                "tokens_sha256": sha256_file(tok_path),
                "mask_sha256": sha256_file(mask_path),
            }
        )

    def close(self, *, flush_remainder: bool = True) -> list[dict[str, Any]]:
        if flush_remainder and self._tok_buf:
            self._write(self._tok_buf, self._mask_buf)
            self._tok_buf, self._mask_buf = [], []
        return self.shards


def _read_message_records(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sft_corpus_manifest(
    *,
    name: str,
    version: str,
    tokenizer: str,
    seq_len: int,
    num_examples: int,
    num_tokens: int,
    num_loss_tokens: int,
    mixture: dict[str, Any],
    shards: list[dict[str, Any]],
    decontam: dict[str, Any],
    license_notes: list[str],
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    created = (now or dt.datetime.now(dt.UTC)).strftime("%Y-%m-%d")
    return {
        "corpus_name": name,
        "version": version,
        "kind": "sft_packed",
        "created_at": created,
        "tokenizer": tokenizer,
        "seq_len": seq_len,
        "num_examples": num_examples,
        "num_tokens": num_tokens,
        "num_loss_tokens": num_loss_tokens,
        # fraction of packed positions carrying loss — no padding in the denominator,
        # so this reads far higher than the dense path's examples*seq_len basis.
        "loss_token_fraction": round(num_loss_tokens / max(num_tokens, 1), 4),
        "mixture": mixture,
        "decontam": decontam,
        "license_notes": license_notes,
        "shards": shards,
    }


def _render_source(
    src: SFTSourceSpec,
    tokenizer: Any,
    seq_len: int,
    add_bos: bool,
    decon: PostTrainDecontaminator | None,
    rng: random.Random,
) -> tuple[list[tuple[list[int], list[bool]]], dict[str, Any]]:
    """Render one source to (input_ids, loss_mask) pairs, applying decontam, the
    overlong drop, the max_examples cap, and repeats. Returns (examples, accounting)."""
    records = list(_read_message_records(src.path))
    read = len(records)
    decontam_dropped = 0
    if decon is not None:
        before = decon.dropped
        records = decon.screen(records, messages_text)
        decontam_dropped = decon.dropped - before
    if src.max_examples is not None and len(records) > src.max_examples:
        rng.shuffle(records)  # seeded random subset, not just the file head
        records = records[: src.max_examples]

    unique: list[tuple[list[int], list[bool]]] = []
    dropped_overlong = 0
    dropped_no_target = 0
    for rec in records:
        rendered = render_conversation(rec["messages"], tokenizer, add_bos=add_bos)
        ids, mask = rendered.input_ids, rendered.loss_mask
        if len(ids) > seq_len:  # can't fit one context window; kept short until E10 raises seq_len
            dropped_overlong += 1
            continue
        if not any(mask):  # no assistant tokens to learn
            dropped_no_target += 1
            continue
        unique.append((ids, mask))

    examples = unique * src.repeats  # upsample the whole (deduped) set
    tokens = sum(len(i) for i, _ in examples)
    loss_tokens = sum(sum(m) for _, m in examples)
    accounting = {
        "read": read,
        "kept_unique": len(unique),
        "examples": len(examples),
        "repeats": src.repeats,
        "tokens": tokens,
        "loss_tokens": loss_tokens,
        "decontam_dropped": decontam_dropped,
        "dropped_overlong": dropped_overlong,
        "dropped_no_target": dropped_no_target,
    }
    return examples, accounting


def build_sft_corpus(
    cfg: SFTCorpusBuildConfig, *, now: dt.datetime | None = None
) -> dict[str, Any]:
    """Render + blend + pack the sources into dual-stream shards; write the train
    (and optional val) SFT manifests. Returns the train manifest.

    Sources are rendered and written **one at a time**, so peak memory is bounded
    by the largest single (capped) source rather than the sum of all sources — the
    caps are the memory control. Shards are source-blocked; the train-time window
    shuffle (`PackedDataLoader`) mixes sources across batches, so no build-time
    global shuffle is needed. Val is split per-source (stratified, exact).
    Reservoir-sampling an *uncapped* giant source is the next scale-up step.
    """
    # Acquisition gate, before any work. It applies to the **assistant targets** — the
    # only gradient-bearing spans — not to the prompts, which the loss mask zeroes. So a
    # `restricted` textbook problem may be a prompt; it may not be a target. All sources
    # are checked up front so a bad blend fails before a single token is rendered.
    if cfg.enforce_tiers:
        for src in cfg.sources:
            assert_trainable(
                {
                    "id": src.name,
                    "source": src.path,
                    "tier": src.target_tier(),
                    "metadata": {"grounded_on": src.grounded_on},
                }
            )
            assert_prompt_source(src.masked_prompt_tier(), where=f"prompt of {src.name!r}")

    out = ensure_dir(cfg.output_dir)
    tokenizer = load_tokenizer(cfg.tokenizer_path)
    tokenizer_name = Path(cfg.tokenizer_path).parent.name
    dtype = dtype_for_vocab(tokenizer.get_vocab_size())
    rng = random.Random(cfg.seed)
    decon = (
        PostTrainDecontaminator.from_probe_file(cfg.decontam_probes)
        if cfg.decontam_probes
        else None
    )

    train_writer = SFTShardWriter(
        out / "tokenized",
        tokens_per_shard=cfg.tokens_per_shard,
        dtype=dtype,
        tokenizer_name=tokenizer_name,
        rel_base=out,
    )
    val_out = ensure_dir(out / "val") if cfg.val_fraction > 0 else None
    val_writer = (
        SFTShardWriter(
            val_out / "tokenized",
            tokens_per_shard=cfg.tokens_per_shard,
            dtype=dtype,
            tokenizer_name=tokenizer_name,
            rel_base=val_out,
        )
        if val_out is not None
        else None
    )

    mixture: dict[str, Any] = {}
    n_train = n_val = 0
    for src in cfg.sources:
        examples, accounting = _render_source(src, tokenizer, cfg.seq_len, cfg.add_bos, decon, rng)
        rng.shuffle(examples)  # shuffle within the source, so the val slice is unbiased
        split = int(len(examples) * cfg.val_fraction) if val_writer is not None else 0
        for ids, mask in examples[:split]:
            val_writer.add(ids, mask)  # type: ignore[union-attr]  # split==0 when val_writer is None
        for ids, mask in examples[split:]:
            train_writer.add(ids, mask)
        mixture[src.name] = accounting
        n_val += split
        n_train += len(examples) - split
        del examples  # release this source before rendering the next

    if n_train + n_val == 0:
        raise ValueError("no usable SFT examples across all sources")

    train_shards = train_writer.close()
    decontam_report = decon.report() if decon is not None else {}
    manifest = sft_corpus_manifest(
        name=cfg.name,
        version=cfg.version,
        tokenizer=tokenizer_name,
        seq_len=cfg.seq_len,
        num_examples=n_train,
        num_tokens=train_writer.total_tokens,
        num_loss_tokens=train_writer.total_loss_tokens,
        mixture=mixture,
        shards=train_shards,
        decontam=decontam_report,
        license_notes=cfg.license_notes,
        now=now,
    )
    write_json(out / "sft_manifest.json", manifest)

    if val_writer is not None and val_out is not None:
        val_shards = val_writer.close()
        write_json(
            val_out / "sft_manifest.json",
            sft_corpus_manifest(
                name=f"{cfg.name}-val",
                version=cfg.version,
                tokenizer=tokenizer_name,
                seq_len=cfg.seq_len,
                num_examples=n_val,
                num_tokens=val_writer.total_tokens,
                num_loss_tokens=val_writer.total_loss_tokens,
                mixture=mixture,
                shards=val_shards,
                decontam=decontam_report,
                license_notes=cfg.license_notes,
                now=now,
            ),
        )
    return manifest
