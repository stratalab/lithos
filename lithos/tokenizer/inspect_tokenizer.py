"""Inspect a trained tokenizer: vocab size, special-token IDs, fertility (PRD §7.2)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def special_token_ids(tok: Tokenizer, special_tokens: list[str]) -> dict[str, int | None]:
    return {t: tok.token_to_id(t) for t in special_tokens}


def fertility(tok: Tokenizer, texts: Iterable[str]) -> dict[str, float]:
    """Compute tokens-per-word and chars-per-token over a sample."""
    n_tokens = n_words = n_chars = 0
    for text in texts:
        n_tokens += len(tok.encode(text).ids)
        n_words += len(text.split())
        n_chars += len(text)
    return {
        "tokens": n_tokens,
        "words": n_words,
        "chars": n_chars,
        "tokens_per_word": n_tokens / max(n_words, 1),
        "chars_per_token": n_chars / max(n_tokens, 1),
    }


def inspect(tok: Tokenizer, special_tokens: list[str], texts: Iterable[str]) -> dict[str, Any]:
    return {
        "vocab_size": tok.get_vocab_size(),
        "special_token_ids": special_token_ids(tok, special_tokens),
        "fertility": fertility(tok, texts),
    }
