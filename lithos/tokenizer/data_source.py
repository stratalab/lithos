"""Text sources for tokenizer training: local JSONL(.zst) or an HF dataset.

These are deliberately minimal — the full corpus pipeline lands in Phase 3. Here
we only need to stream raw ``text`` strings to the BPE trainer.
"""

from __future__ import annotations

import contextlib
import io
import json
from collections.abc import Iterator
from pathlib import Path

import zstandard

from lithos.tokenizer.tokenizer_config import DataSourceSpec


@contextlib.contextmanager
def _open_text(path: str | Path) -> Iterator[io.TextIOBase]:
    p = Path(path)
    if p.suffix == ".zst":
        with open(p, "rb") as fh:
            reader = zstandard.ZstdDecompressor().stream_reader(fh)
            yield io.TextIOWrapper(reader, encoding="utf-8")
    else:
        with open(p, encoding="utf-8") as f:
            yield f


def iter_jsonl_texts(
    paths: list[str], text_field: str = "text", limit: int | None = None
) -> Iterator[str]:
    """Yield the ``text_field`` of each JSON record across ``paths``."""
    n = 0
    for path in paths:
        with _open_text(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                text = json.loads(line).get(text_field)
                if text:
                    yield text
                    n += 1
                    if limit is not None and n >= limit:
                        return


def iter_hf_texts(
    dataset: str,
    split: str = "train",
    config_name: str | None = None,
    text_field: str = "text",
    limit: int | None = None,
) -> Iterator[str]:
    """Stream the ``text_field`` from an HF dataset (e.g. nvidia/Nemotron-CC-v2)."""
    from datasets import load_dataset

    ds = load_dataset(dataset, name=config_name, split=split, streaming=True)
    n = 0
    for row in ds:
        text = row.get(text_field)
        if text:
            yield text
            n += 1
            if limit is not None and n >= limit:
                return


def resolve_texts(data: DataSourceSpec) -> tuple[list[str], Iterator[str]]:
    """Return a (sources_description, text_iterator) pair for a data spec."""
    if data.kind == "hf":
        if not data.dataset:
            raise ValueError("data.kind='hf' requires data.dataset to be set.")
        sources = [f"hf:{data.dataset}:{data.config_name or '-'}:{data.split}"]
        texts = iter_hf_texts(
            data.dataset, data.split, data.config_name, data.text_field, data.max_documents
        )
        return sources, texts
    if not data.paths:
        raise ValueError("data.kind='jsonl' requires data.paths to be set.")
    return list(data.paths), iter_jsonl_texts(data.paths, data.text_field, data.max_documents)
