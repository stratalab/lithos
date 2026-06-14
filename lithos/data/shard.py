"""Binary tokenized shards: flat uint16/uint32 token arrays (PRD §8.5).

A shard is a contiguous little-endian array of token ids written to ``.bin``.
Sequence windows are carved at load time by the dataloader, so shards stay simple
and memory-mappable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from lithos.utils.io import ensure_dir, read_json, sha256_file

ShardSpec = tuple[str, int, str]  # (resolved path, num_tokens, dtype)


def dtype_for_vocab(vocab_size: int) -> str:
    """uint16 covers vocabularies up to 65536 tokens; otherwise uint32."""
    return "uint16" if vocab_size <= 2**16 else "uint32"


class ShardWriter:
    """Accumulate token ids and flush fixed-size binary shards with manifests."""

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
        # Manifest shard paths are stored relative to ``rel_base`` (e.g. the corpus
        # dir) so a corpus stays loadable after being moved or pulled from storage.
        self.rel_base = Path(rel_base) if rel_base is not None else None
        self._buf: list[int] = []
        self._shard_idx = 0
        self.total_tokens = 0
        self.shards: list[dict[str, Any]] = []

    def add(self, token_ids: list[int]) -> None:
        self._buf.extend(token_ids)
        while len(self._buf) >= self.tokens_per_shard:
            chunk = self._buf[: self.tokens_per_shard]
            del self._buf[: self.tokens_per_shard]
            self._write(chunk)

    def _write(self, ids: list[int]) -> None:
        self._shard_idx += 1
        shard_id = f"shard_{self._shard_idx:06d}"
        path = self.dir / f"{shard_id}.bin"
        arr = np.asarray(ids, dtype=self.dtype)
        arr.tofile(path)
        self.total_tokens += int(arr.size)
        stored = str(path.relative_to(self.rel_base)) if self.rel_base is not None else str(path)
        self.shards.append(
            {
                "shard_id": shard_id,
                "path": stored,
                "num_tokens": int(arr.size),
                "dtype": self.dtype.name,
                "tokenizer": self.tokenizer_name,
                "sha256": sha256_file(path),
            }
        )

    def close(self, *, flush_remainder: bool = True) -> list[dict[str, Any]]:
        if flush_remainder and self._buf:
            self._write(self._buf)
            self._buf = []
        return self.shards


def read_shard_specs(manifest_path: str | Path) -> list[ShardSpec]:
    """Read a corpus manifest, resolving shard paths relative to its directory.

    Shard paths are stored relative to the manifest (portable across moves and
    storage round-trips); absolute paths from older manifests are passed through.
    """
    manifest_path = Path(manifest_path)
    base = manifest_path.parent
    specs: list[ShardSpec] = []
    for s in read_json(manifest_path)["shards"]:
        p = Path(s["path"])
        resolved = str(p if p.is_absolute() else base / p)
        specs.append((resolved, int(s["num_tokens"]), str(s["dtype"])))
    return specs


def load_shard(path: str | Path, dtype: str) -> np.memmap:
    """Memory-map a shard for reading."""
    return np.memmap(path, dtype=np.dtype(dtype), mode="r")
