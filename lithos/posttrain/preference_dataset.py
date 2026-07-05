"""Preference dataset for DPO (Phase 11).

Each input line is ``{"prompt": [{"role","content"}...], "chosen": str, "rejected": str}``.
The prompt + each response is rendered with the *same* chat template / masking as
SFT (via ``build_xy``), giving two ``(x, y)`` pairs per example that share the
(masked) prompt and differ only in the response. ``__getitem__`` returns the
4-tuple ``(chosen_x, chosen_y, rejected_x, rejected_y)``; the DPO trainer batches
and scores them under the policy + frozen reference.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lithos.posttrain.chat_template import TokenizerLike, special_ids
from lithos.posttrain.sft_dataset import build_xy


class PreferenceDataset:
    def __init__(
        self,
        path: str | Path,
        tokenizer: TokenizerLike,
        seq_len: int,
        *,
        add_bos: bool = True,
    ) -> None:
        self.seq_len = seq_len
        pad_id = special_ids(tokenizer)["<pad>"]

        cx: list[list[int]] = []
        cy: list[list[int]] = []
        rx: list[list[int]] = []
        ry: list[list[int]] = []
        read = dropped = 0
        for rec in _read_prefs(path):
            read += 1
            prompt = rec["prompt"]
            chosen = build_xy(
                [*prompt, {"role": "assistant", "content": rec["chosen"]}],
                tokenizer, seq_len, pad_id, add_bos=add_bos,
            )
            rejected = build_xy(
                [*prompt, {"role": "assistant", "content": rec["rejected"]}],
                tokenizer, seq_len, pad_id, add_bos=add_bos,
            )
            if chosen is None or rejected is None:  # drop the pair if either won't fit
                dropped += 1
                continue
            cx.append(chosen[0])
            cy.append(chosen[1])
            rx.append(rejected[0])
            ry.append(rejected[1])

        if not cx:
            raise ValueError(f"no usable preference pairs in {path}")
        self._cx = np.asarray(cx, dtype=np.int64)
        self._cy = np.asarray(cy, dtype=np.int64)
        self._rx = np.asarray(rx, dtype=np.int64)
        self._ry = np.asarray(ry, dtype=np.int64)
        self._stats = {"pairs": len(cx), "read": read, "dropped": dropped, "seq_len": seq_len}

    def __len__(self) -> int:
        return int(self._cx.shape[0])

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return (
            torch.from_numpy(self._cx[index]),
            torch.from_numpy(self._cy[index]),
            torch.from_numpy(self._rx[index]),
            torch.from_numpy(self._ry[index]),
        )

    def stats(self) -> dict[str, Any]:
        return dict(self._stats)


def _read_prefs(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
