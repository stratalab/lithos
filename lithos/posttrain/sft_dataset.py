"""SFT dataset: messages-JSONL -> (x, y) windows for the existing training loop.

Each input line is ``{"messages": [{"role","content"}, ...]}``. Every conversation
is rendered with the chat template, shifted into a next-token ``(input, label)``
pair, and padded/truncated to ``seq_len`` with ``-100`` on padding and on every
non-assistant token. The class implements the ``PackedDataset`` interface
(``__len__`` + ``__getitem__ -> (x, y)``), so it drops straight into
``PackedDataLoader`` and ``train()`` with no loop changes (Phase 11).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from lithos.posttrain.chat_template import TokenizerLike, render_conversation, special_ids

IGNORE_INDEX = -100  # matches F.cross_entropy(ignore_index=...) in the model


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
