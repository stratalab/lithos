"""Tests for the byte-level BPE tokenizer (PRD §7.3).

Byte-level encoding is lossless regardless of vocabulary, so roundtrip holds even
for a tiny tokenizer trained on a small synthetic corpus.
"""

import json

import pytest
from lithos.tokenizer import (
    DEFAULT_SPECIAL_TOKENS,
    TokenizerConfig,
    build_manifest,
    load_tokenizer,
    sample_report,
    save_tokenizer,
    special_token_ids,
    train_tokenizer,
)

CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Lithos trains small language models from scratch.",
    "def add(a, b):\n    return a + b",
    "for i in range(10):\n    print(i * i)",
    "import numpy as np  # numerical computing",
    "The sum from 1 to n equals n times (n + 1) over 2.",
    "Energy equals mass times the speed of light squared.",
    "x = 3.14159 * radius ** 2",
    "Tokenizers map text to integer ids and back again.",
    "Byte-level encoding handles any unicode input losslessly.",
    "Hello, world! Greetings from the Strata ecosystem.",
    "A B C D E F G 0 1 2 3 4 5 6 7 8 9",
]


@pytest.fixture(scope="module")
def tok():
    cfg = TokenizerConfig(vocab_size=800)
    tokenizer, _ = train_tokenizer(cfg, CORPUS * 25)
    return tokenizer


def _roundtrip(tokenizer, text: str) -> str:
    return tokenizer.decode(tokenizer.encode(text).ids)


def test_roundtrip_ordinary_text(tok):
    s = "The quick brown fox jumps over 13 lazy dogs."
    assert _roundtrip(tok, s) == s


def test_roundtrip_code(tok):
    s = "def f(x):\n    return x ** 2 + 1  # a comment\n\twith\ttabs"
    assert _roundtrip(tok, s) == s


def test_roundtrip_math_symbols(tok):
    s = "∑_{i=1}^{n} i = n(n+1)/2,  α+β ≤ γ,  π ≈ 3.14159,  ∫ f dx"  # noqa: RUF001
    assert _roundtrip(tok, s) == s


def test_special_token_ids_are_stable(tok):
    ids = special_token_ids(tok, DEFAULT_SPECIAL_TOKENS)
    assert ids == {t: i for i, t in enumerate(DEFAULT_SPECIAL_TOKENS)}


def test_unusual_unicode_does_not_crash(tok):
    s = "emoji 🦙🔥, CJK 漢字, RTL مرحبا, accents café, control\x07 chars"
    assert _roundtrip(tok, s) == s


def test_empty_string_does_not_crash(tok):
    assert tok.encode("").ids == []
    assert _roundtrip(tok, "") == ""


def test_digits_are_split_individually(tok):
    # individual_digits=True -> each digit is its own (single-byte) token.
    assert tok.encode("2024").tokens == ["2", "0", "2", "4"]


def test_train_save_reload_artifacts(tmp_path):
    cfg = TokenizerConfig(vocab_size=600)
    tokenizer, stats = train_tokenizer(cfg, CORPUS * 25)
    manifest = build_manifest(cfg, stats, sources=["synthetic-test-corpus"])
    report = sample_report(tokenizer, ["hello 7 world", "x = 1 + 2"])
    out = save_tokenizer(tokenizer, cfg, tmp_path / "tok", manifest, report)

    for fname in (
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer_manifest.json",
        "sample_report.json",
    ):
        assert (out / fname).is_file()

    # Manifest carries the required §7.2 provenance fields.
    m = json.loads((out / "tokenizer_manifest.json").read_text())
    for key in (
        "sources",
        "num_documents",
        "approx_chars",
        "vocab_size",
        "special_tokens",
        "normalization",
        "pre_tokenization",
        "created_at",
    ):
        assert key in m
    assert m["num_documents"] == len(CORPUS) * 25

    # Reloaded tokenizer still roundtrips.
    reloaded = load_tokenizer(out / "tokenizer.json")
    assert reloaded.decode(reloaded.encode("hello 7 world").ids) == "hello 7 world"
