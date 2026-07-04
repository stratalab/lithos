"""Tests for tokenizer quality evaluation (lithos/tokenizer/evaluate.py).

Metrics are exercised on a tiny in-test tokenizer (like the roundtrip tests);
the probe fixtures under corpus/probes/ are validated for shape so the eval
script can always rely on them.
"""

import json
from pathlib import Path

import pytest
from lithos.tokenizer import (
    TokenizerConfig,
    compare_tokenizers,
    compression_stats,
    evaluate_tokenizer,
    roundtrip_failures,
    segmentation_rows,
    special_token_check,
    train_tokenizer,
    vocab_usage,
)

PROBES_DIR = Path(__file__).parent.parent / "corpus" / "probes"

CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "def add(a, b):\n    return a + b",
    "Energy equals mass times the speed of light squared.",
    "x = 3.14159 * radius ** 2",
    "Byte-level encoding handles any unicode input losslessly.",
    "for i in range(10):\n    print(i * i)",
]


@pytest.fixture(scope="module")
def tok():
    tokenizer, _ = train_tokenizer(TokenizerConfig(vocab_size=600), CORPUS * 25)
    return tokenizer


def read_probe(name: str) -> list[dict]:
    path = PROBES_DIR / name
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_compression_stats_counts(tok):
    stats = compression_stats(tok, CORPUS)
    assert stats["docs"] == len(CORPUS)
    assert stats["tokens"] > 0
    assert stats["bytes"] == sum(len(t.encode("utf-8")) for t in CORPUS)
    # byte-level BPE never produces more tokens than bytes
    assert stats["tokens"] <= stats["bytes"]
    assert stats["bytes_per_token"] == pytest.approx(stats["bytes"] / stats["tokens"])


def test_compression_stats_empty(tok):
    stats = compression_stats(tok, [])
    assert stats["docs"] == 0
    assert stats["tokens_per_word"] == 0


def test_vocab_usage_bounds(tok):
    usage = vocab_usage(tok, CORPUS)
    assert 0 < usage["used"] <= usage["vocab_size"]
    assert 0 < usage["used_fraction"] <= 1
    assert usage["rare_used"] <= usage["used"]


def test_roundtrip_clean_on_training_corpus(tok):
    assert roundtrip_failures(tok, CORPUS) == []


def test_roundtrip_reports_failures_shape(tok):
    # sanity that the failure record shape is stable when a mismatch is forced
    class Broken:
        def encode(self, text, add_special_tokens=True):
            return tok.encode(text, add_special_tokens=add_special_tokens)

        def decode(self, ids):
            return "corrupted"

    failures = roundtrip_failures(Broken(), ["hello"])
    assert failures and failures[0]["decoded"] == "corrupted"


def test_special_tokens_stable_low_ids(tok):
    check = special_token_check(tok)
    assert check["stable_low_ids"] is True
    assert check["ids"]["<pad>"] == 0
    assert check["ids"]["<bos>"] == 1


def test_segmentation_rows(tok):
    probes = [{"text": "12345", "note": "digits"}, {"text": "def f():"}]
    rows = segmentation_rows(tok, probes)
    assert len(rows) == 2
    assert rows[0]["note"] == "digits"
    # individual_digits pre-tokenization: one token per digit, at least
    assert rows[0]["n_tokens"] >= 5
    assert rows[1]["note"] == ""
    assert all(len(r["tokens"]) == r["n_tokens"] for r in rows)


def test_compare_tokenizers_identical(tok):
    domains = {"a": CORPUS[:3], "b": CORPUS[3:]}
    out = compare_tokenizers(domains, {"x": tok, "y": tok})
    for domain in domains:
        assert out[domain]["x"] == out[domain]["y"]
        assert out[domain]["x"]["tokens"] > 0


def test_evaluate_tokenizer_report_shape(tok):
    report = evaluate_tokenizer(
        tok,
        domains={"general": CORPUS},
        segmentation_probes=[{"text": "3.14", "note": "pi"}],
        adversarial_texts=["tab\there", "emoji 🎉"],
    )
    assert set(report) == {"vocab", "special_tokens", "domains", "roundtrip", "segmentation"}
    assert report["roundtrip"]["checked"] == len(CORPUS) + 2
    assert report["roundtrip"]["failures"] == []
    assert report["domains"]["general"]["docs"] == len(CORPUS)


@pytest.mark.parametrize(
    "name",
    [
        "general.jsonl",
        "math.jsonl",
        "code.jsonl",
        "physics.jsonl",
        "engineering.jsonl",
        "adversarial.jsonl",
    ],
)
def test_probe_files_well_formed(name):
    records = read_probe(name)
    assert len(records) >= 8
    for r in records:
        assert isinstance(r["text"], str) and r["text"]


def test_segmentation_probe_file_has_notes():
    records = read_probe("segmentation.jsonl")
    assert len(records) >= 15
    for r in records:
        assert r["text"] and r["note"]


def test_probes_roundtrip_lossless(tok):
    """Adversarial probes must decode byte-exact through any byte-level tokenizer."""
    texts = [r["text"] for r in read_probe("adversarial.jsonl")]
    assert roundtrip_failures(tok, texts) == []
