"""Tokenize documents into token-id streams (PRD §8.2).

Each document is wrapped with ``<bos>``/``<eos>`` before concatenation, giving
the model document-boundary signal under standard (bleed) packing.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from tokenizers import Tokenizer


class DocumentTokenizer:
    """Encode document text to token ids, optionally wrapping with bos/eos."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        bos_id: int | None,
        eos_id: int | None,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.add_bos = add_bos
        self.add_eos = add_eos

    def encode(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(text).ids
        if self.add_bos and self.bos_id is not None:
            ids = [self.bos_id, *ids]
        if self.add_eos and self.eos_id is not None:
            ids = [*ids, self.eos_id]
        return ids

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: Tokenizer,
        *,
        bos: str = "<bos>",
        eos: str = "<eos>",
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> DocumentTokenizer:
        return cls(
            tokenizer,
            bos_id=tokenizer.token_to_id(bos),
            eos_id=tokenizer.token_to_id(eos),
            add_bos=add_bos,
            add_eos=add_eos,
        )


def tokenize_documents(
    doctok: DocumentTokenizer, docs: Iterable[dict[str, Any]]
) -> Iterator[list[int]]:
    for doc in docs:
        yield doctok.encode(doc["text"])
