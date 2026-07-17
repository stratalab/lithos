"""SFT dataset: messages-JSONL -> (x, y) windows for the existing training loop.

Each input line is ``{"messages": [{"role","content"}, ...]}``. Every conversation
is rendered with the chat template, shifted into a next-token ``(input, label)``
pair, and padded/truncated to ``seq_len`` with ``-100`` on padding and on every
non-assistant token. The class implements the ``PackedDataset`` interface
(``__len__`` + ``__getitem__ -> (x, y)``), so it drops straight into
``PackedDataLoader`` and ``train()`` with no loop changes (Phase 11).
"""

from __future__ import annotations

import bisect
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lithos.data.packing import get_sequence, num_sequences
from lithos.data.shard import load_shard
from lithos.posttrain.chat_template import TokenizerLike, render_conversation, special_ids
from lithos.posttrain.record import IGNORE_INDEX, TrainingRecord
from lithos.utils.io import read_json

__all__ = [
    "IGNORE_INDEX",  # canonical home is record.py; re-exported for existing importers
    "PackedSFTDataset",
    "SFTDataset",
    "SFTShardSpec",
    "build_xy",
    "load_sft_shard_specs",
]

# (tokens_path, weights_path, num_tokens, dtype) — for the packed dual-stream loader.
SFTShardSpec = tuple[str, str, int, str]


def build_xy(
    messages: list[dict[str, str]],
    tokenizer: TokenizerLike,
    seq_len: int,
    pad_id: int,
    *,
    add_bos: bool = True,
) -> tuple[list[int], list[int]] | None:
    """Render a conversation to a padded ``(x, y)`` training pair.

    Returns ``None`` if the example doesn't fit (overlong — dropped, never
    right-truncated, since that loses the reply) or has no assistant tokens to
    learn. Shared by SFT and DPO (chosen/rejected) so masking is identical.
    """
    rec = TrainingRecord.from_rendered(render_conversation(messages, tokenizer, add_bos=add_bos))
    if len(rec.tokens) < 2 or not rec.has_targets():
        return None
    x = rec.tokens[:-1]
    y = rec.labels()
    if len(x) > seq_len:
        return None
    if (pad := seq_len - len(x)) > 0:  # right-pad; causal attn + masked loss keep it safe
        x = x + [pad_id] * pad
        y = y + [IGNORE_INDEX] * pad
    return x, y


class SFTDataset:
    """Indexable view of tokenized SFT examples (one conversation per sequence)."""

    def __init__(
        self,
        path: str | Path,
        tokenizer: TokenizerLike,
        seq_len: int,
        *,
        add_bos: bool = True,
    ) -> None:
        self.seq_len = seq_len
        self.pad_id = special_ids(tokenizer)["<pad>"]

        xs: list[list[int]] = []
        ys: list[list[int]] = []
        read = dropped = loss_tokens = 0
        for messages in _read_messages(path):
            read += 1
            pair = build_xy(messages, tokenizer, seq_len, self.pad_id, add_bos=add_bos)
            if pair is None:
                dropped += 1
                continue
            x, y = pair
            xs.append(x)
            ys.append(y)
            loss_tokens += sum(1 for t in y if t != IGNORE_INDEX)

        if not xs:
            raise ValueError(f"no usable SFT examples in {path}")
        self._x = np.asarray(xs, dtype=np.int64)
        self._y = np.asarray(ys, dtype=np.int64)
        self._stats = {
            "examples": len(xs),
            "read": read,
            "dropped": dropped,
            "seq_len": seq_len,
            "loss_token_fraction": round(loss_tokens / (len(xs) * seq_len), 4),
        }

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return torch.from_numpy(self._x[index]), torch.from_numpy(self._y[index])

    def stats(self) -> dict[str, Any]:
        return dict(self._stats)


def _read_messages(path: str | Path) -> Iterator[list[dict[str, str]]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)["messages"]


def load_sft_shard_specs(manifest_path: str | Path) -> list[SFTShardSpec]:
    """Read an SFT manifest, resolving both channel paths relative to its directory
    (portable across moves), mirroring ``data.shard.read_shard_specs``. Accepts both
    the current ``weights_path`` (float32) and the legacy ``mask_path`` (uint8) key;
    the loader tells the stream dtypes apart by file suffix."""
    manifest_path = Path(manifest_path)
    base = manifest_path.parent
    specs: list[SFTShardSpec] = []
    for s in read_json(manifest_path)["shards"]:
        tp = Path(s["tokens_path"])
        wp = Path(s["weights_path"] if "weights_path" in s else s["mask_path"])
        tokens = str(tp if tp.is_absolute() else base / tp)
        weights = str(wp if wp.is_absolute() else base / wp)
        specs.append((tokens, weights, int(s["num_tokens"]), str(s["dtype"])))
    return specs


class PackedSFTDataset:
    """Indexable view over packed dual-stream SFT shards, mirroring ``PackedDataset``.

    Memory-maps the token + loss-weight streams of each shard and carves ``seq_len``
    windows at load time: ``x`` is the token window, ``y`` is the shifted token
    window with ``IGNORE_INDEX`` wherever the (shifted) weight is 0 — exactly the
    ``(x, y)`` int64 contract the training loop already consumes, so it drops into
    ``PackedDataLoader`` and ``train()`` unchanged. Windows cross conversation
    boundaries (bleed packing); the weights keep loss on assistant tokens only.

    The ``(x, y)`` contract is a **binary projection**: the model's loss is
    ``F.cross_entropy(ignore_index=IGNORE_INDEX)``, which can drop a position but
    not scale it. Until the loop grows a weighted-CE path, a shard with fractional
    weights would be *silently trained at weight 1.0* — so this loader refuses it
    loudly instead (fail-closed, same posture as the tier gate).
    """

    def __init__(self, shards: list[SFTShardSpec], seq_len: int) -> None:
        self.seq_len = seq_len
        self._tok: list[np.memmap] = []
        self._weights: list[np.memmap] = []
        self._cumulative: list[int] = [0]
        for tokens_path, weights_path, num_tokens, dtype in shards:
            tok = load_shard(tokens_path, dtype)
            # Legacy shards store a uint8 ``.mask.bin``; current ones float32
            # ``.weights.bin``. Same semantics (> 0 = loss target), different dtype.
            weights_dtype = "uint8" if weights_path.endswith(".mask.bin") else "float32"
            weights = load_shard(weights_path, weights_dtype)
            if len(tok) != len(weights):  # streams must be lockstep; catch corruption early
                raise ValueError(
                    f"token/weights stream length mismatch in {tokens_path}: "
                    f"{len(tok)} != {len(weights)}"
                )
            if weights_dtype == "float32" and not bool(
                ((weights == 0) | (weights == 1)).all()
            ):
                raise NotImplementedError(
                    f"{weights_path} carries fractional loss weights; the (x, y) "
                    "ignore-index contract can only drop tokens, not scale them — "
                    "weighted cross-entropy in the train loop is the pending seam "
                    "(docs/tinker-learnings.md T1)"
                )
            self._tok.append(tok)
            self._weights.append(weights)
            self._cumulative.append(self._cumulative[-1] + num_sequences(num_tokens, seq_len))
        self.total = self._cumulative[-1]

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self.total:
            raise IndexError(index)
        shard = bisect.bisect_right(self._cumulative, index) - 1
        local = index - self._cumulative[shard]
        tok_x, tok_y = get_sequence(self._tok[shard], local, self.seq_len)
        _, weights_y = get_sequence(self._weights[shard], local, self.seq_len)
        x = torch.from_numpy(tok_x.astype(np.int64))
        y = torch.from_numpy(tok_y.astype(np.int64))
        y[torch.from_numpy(np.asarray(weights_y) <= 0)] = IGNORE_INDEX
        return x, y
