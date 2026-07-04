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
from lithos.utils.io import read_json

IGNORE_INDEX = -100  # matches F.cross_entropy(ignore_index=...) in the model

# (tokens_path, mask_path, num_tokens, dtype) — for the packed dual-stream loader.
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
    r = render_conversation(messages, tokenizer, add_bos=add_bos)
    ids, m = r.input_ids, r.loss_mask
    if len(ids) < 2:
        return None
    x = ids[:-1]
    y = [ids[i + 1] if m[i + 1] else IGNORE_INDEX for i in range(len(ids) - 1)]
    if len(x) > seq_len:
        return None
    if all(t == IGNORE_INDEX for t in y):
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
    (portable across moves), mirroring ``data.shard.read_shard_specs``."""
    manifest_path = Path(manifest_path)
    base = manifest_path.parent
    specs: list[SFTShardSpec] = []
    for s in read_json(manifest_path)["shards"]:
        tp, mp = Path(s["tokens_path"]), Path(s["mask_path"])
        tokens = str(tp if tp.is_absolute() else base / tp)
        mask = str(mp if mp.is_absolute() else base / mp)
        specs.append((tokens, mask, int(s["num_tokens"]), str(s["dtype"])))
    return specs


class PackedSFTDataset:
    """Indexable view over packed dual-stream SFT shards, mirroring ``PackedDataset``.

    Memory-maps the token + mask streams of each shard and carves ``seq_len``
    windows at load time: ``x`` is the token window, ``y`` is the shifted token
    window with ``IGNORE_INDEX`` wherever the (shifted) mask is 0 — exactly the
    ``(x, y)`` int64 contract the training loop already consumes, so it drops into
    ``PackedDataLoader`` and ``train()`` unchanged. Windows cross conversation
    boundaries (bleed packing); the mask keeps loss on assistant tokens only.
    """

    def __init__(self, shards: list[SFTShardSpec], seq_len: int) -> None:
        self.seq_len = seq_len
        self._tok: list[np.memmap] = []
        self._mask: list[np.memmap] = []
        self._cumulative: list[int] = [0]
        for tokens_path, mask_path, num_tokens, dtype in shards:
            tok = load_shard(tokens_path, dtype)
            mask = load_shard(mask_path, "uint8")
            if len(tok) != len(mask):  # streams must be lockstep; catch corruption early
                raise ValueError(
                    f"token/mask stream length mismatch in {tokens_path}: {len(tok)} != {len(mask)}"
                )
            self._tok.append(tok)
            self._mask.append(mask)
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
        _, mask_y = get_sequence(self._mask[shard], local, self.seq_len)
        x = torch.from_numpy(tok_x.astype(np.int64))
        y = torch.from_numpy(tok_y.astype(np.int64))
        y[torch.from_numpy(np.asarray(mask_y) == 0)] = IGNORE_INDEX
        return x, y
