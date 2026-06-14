"""Tests for lithos.evals.generate_samples — repetition score and sampling."""

from lithos.evals.generate_samples import generate_samples, repetition_score
from lithos.tokenizer import TokenizerConfig, train_tokenizer

from tests.helpers import make_model


def test_repetition_score():
    assert repetition_score([1, 2, 3, 4, 5], n=2) == 0.0  # all distinct bigrams
    assert repetition_score([7, 7, 7, 7], n=2) > 0.6  # highly repetitive
    assert repetition_score([1], n=3) == 0.0  # too short -> 0


def test_generate_samples_returns_completions():
    corpus = ["lithos trains small language models from scratch "] * 40
    tok, _ = train_tokenizer(TokenizerConfig(vocab_size=400), corpus)
    model = make_model(vocab_size=tok.get_vocab_size(), seq_len=64)

    out = generate_samples(model, tok, ["lithos", "models"], max_new_tokens=8, greedy=True)
    assert len(out) == 2
    assert out[0]["prompt"] == "lithos"
    assert out[0]["n_new_tokens"] == 8
    assert 0.0 <= out[0]["repetition"] <= 1.0
    assert isinstance(out[0]["completion"], str)
