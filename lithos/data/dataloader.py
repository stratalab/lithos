"""Resumable packed dataloader over tokenized shards (PRD §9.9, §27).

``PackedDataset`` maps a global sequence index to an (x, y) window across shards.
``PackedDataLoader`` is an infinite, shuffled, **resumable** batch iterator: its
``state_dict``/``load_state_dict`` capture (epoch, position, seed) so a resumed run
continues at the exact same data position instead of silently re-seeing data.
"""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lithos.data.packing import get_sequence, num_sequences
from lithos.data.shard import load_shard

ShardSpec = tuple[str | Path, int, str]  # (path, num_tokens, dtype)


class PackedDataset:
    """Indexable view of packed (x, y) sequences over a list of shards."""

    def __init__(self, shards: list[ShardSpec], seq_len: int) -> None:
        self.seq_len = seq_len
        self._mm: list[np.memmap] = []
        self._cumulative: list[int] = [0]
        for path, num_tokens, dtype in shards:
            self._mm.append(load_shard(path, dtype))
            self._cumulative.append(self._cumulative[-1] + num_sequences(num_tokens, seq_len))
        self.total = self._cumulative[-1]

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self.total:
            raise IndexError(index)
        shard = bisect.bisect_right(self._cumulative, index) - 1
        local = index - self._cumulative[shard]
        x, y = get_sequence(self._mm[shard], local, self.seq_len)
        return (
            torch.from_numpy(x.astype(np.int64)),
            torch.from_numpy(y.astype(np.int64)),
        )


class PackedDataLoader:
    """Infinite, resumable, shuffled batch iterator over a ``PackedDataset``."""

    def __init__(
        self, dataset: PackedDataset, batch_size: int, *, seed: int = 0, shuffle: bool = True
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.position = 0
        self._perm = self._make_perm(self.epoch)

    def _make_perm(self, epoch: int) -> np.ndarray:
        n = len(self.dataset)
        if not self.shuffle:
            return np.arange(n)
        return np.random.RandomState(self.seed + epoch).permutation(n)

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self.epoch, "position": self.position, "seed": self.seed}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.seed = state["seed"]
        self.epoch = state["epoch"]
        self.position = state["position"]
        self._perm = self._make_perm(self.epoch)

    def __iter__(self) -> PackedDataLoader:
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(self.dataset)
        if n < self.batch_size:
            raise RuntimeError(f"dataset has {n} sequences < batch_size {self.batch_size}")
        if self.position + self.batch_size > n:
            self.epoch += 1
            self.position = 0
            self._perm = self._make_perm(self.epoch)
        idx = self._perm[self.position : self.position + self.batch_size]
        self.position += self.batch_size
        batch = [self.dataset[int(i)] for i in idx]
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        return xs, ys
