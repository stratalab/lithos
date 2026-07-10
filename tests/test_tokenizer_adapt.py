"""Give a Qwen base tokenizer the Lithos chat + TIR specials (`lithos/serve/tokenizer_adapt.py`).

v1 post-trains a Qwen3 base; its tokenizer lacks our named specials, which the chat template
inserts by id. These tests cover the augmentation (add the missing, reuse Qwen's `<think>`,
keep everything atomic) and the load-bearing property that **growing the vocab does not break
import parity** — the whole v1-on-Qwen decision rests on it.
"""

from __future__ import annotations

import pytest
import torch
from lithos.posttrain.chat_template import (
    REQUIRED_SPECIAL_TOKENS,
    special_ids,
    tir_token_ids,
)
from lithos.serve.tokenizer_adapt import augment_tokenizer, import_vocab_size


def _base_tokenizer(vocab_size: int = 400, prehave: list[str] | None = None):
    """A tiny byte-level BPE standing in for Qwen's backend tokenizer."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>"])
    tok.train_from_iterator(
        ["the quick brown fox", "a b c d e f g", "solve for x given y and z"], trainer
    )
    if prehave:  # simulate a base that already carries some of our tokens (Qwen has <think>)
        tok.add_special_tokens(prehave)
    return tok


# ── augmentation ──────────────────────────────────────────────────────────────


def test_all_required_specials_resolve_after_augmentation():
    tok = _base_tokenizer()
    res = augment_tokenizer(tok)
    for name in REQUIRED_SPECIAL_TOKENS:
        assert res.ids[name] == tok.token_to_id(name)
        assert tok.token_to_id(name) is not None


def test_specials_encode_atomically():
    """A special that split into pieces would break insert-by-id and the loss mask."""
    tok = _base_tokenizer()
    augment_tokenizer(tok)
    for name in REQUIRED_SPECIAL_TOKENS:
        assert tok.encode(name).ids == [tok.token_to_id(name)]


def test_ids_are_distinct():
    res = augment_tokenizer(_base_tokenizer())
    assert len(set(res.ids.values())) == len(res.ids)


def test_vocab_grows_by_exactly_the_number_added():
    tok = _base_tokenizer()
    before = tok.get_vocab_size()
    res = augment_tokenizer(tok)
    assert res.base_vocab_size == before
    assert res.vocab_size == before + len(res.added)
    assert set(res.added) | set(res.reused) == set(REQUIRED_SPECIAL_TOKENS)


def test_qwens_existing_think_token_is_reused_not_duplicated():
    """Qwen3 already tokenizes <think>/</think>. Keep its id; the model knows that embedding."""
    tok = _base_tokenizer(prehave=["<think>", "</think>"])
    think_id = tok.token_to_id("<think>")
    res = augment_tokenizer(tok)
    assert "<think>" in res.reused and "</think>" in res.reused
    assert res.ids["<think>"] == think_id  # id preserved
    assert "<|end|>" in res.added  # ...while genuinely-missing ones are still added


def test_augmentation_is_idempotent():
    tok = _base_tokenizer()
    first = augment_tokenizer(tok)
    second = augment_tokenizer(tok)  # nothing left to add
    assert second.added == ()
    assert second.ids == first.ids
    assert second.vocab_size == first.vocab_size


def test_the_augmented_tokenizer_satisfies_special_ids_and_tir_token_ids():
    """The downstream resolvers (chat template, tir_rollout) must both succeed on it."""
    tok = _base_tokenizer()
    augment_tokenizer(tok)
    sids = special_ids(tok)  # raises if any core special is missing
    tir = tir_token_ids(tok)  # raises if any TIR token is missing
    assert sids["<|end|>"] is not None
    assert tir["<|python|>"] is not None


# ── the load-bearing property: import parity survives vocab growth ────────────

transformers = pytest.importorskip("transformers")


def _tiny_qwen3(*, tie: bool, vocab_size: int = 64):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        tie_word_embeddings=tie,
    )
    torch.manual_seed(0)
    m = Qwen3ForCausalLM(cfg)
    m.eval()
    return m


@pytest.mark.parametrize("tie", [True, False])
def test_growing_the_vocab_preserves_import_parity_on_the_original_slice(tie):
    """THE decision rests here: adding specials must not move Qwen's logits."""
    from lithos.serve.hf_import import load_qwen3

    hf = _tiny_qwen3(tie=tie, vocab_size=64)
    grown = 64 + 13  # pretend augmentation added 13 specials past the base vocab

    baseline = load_qwen3(hf)  # imported at the native vocab
    adapted = load_qwen3(hf, vocab_size=grown)  # imported with room for the specials

    ids = torch.randint(0, 64, (1, 10))
    with torch.no_grad():
        theirs = hf(ids).logits
        base_ours, _ = baseline(ids)
        grown_ours, _ = adapted(ids)

    # parity with the source, and with the un-grown import, on the ORIGINAL vocab
    torch.testing.assert_close(base_ours[:, :, :64], theirs, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(grown_ours[:, :, :64], theirs, atol=2e-3, rtol=2e-3)


def test_added_specials_are_valid_tokens_not_masked_padding():
    """An added special must be *emittable*: its logit column is real, not -inf."""
    from lithos.serve.hf_import import load_qwen3

    hf = _tiny_qwen3(tie=True, vocab_size=64)
    adapted = load_qwen3(hf, vocab_size=64 + 4)
    ids = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        logits, _ = adapted(ids)
    neg_inf = torch.finfo(logits.dtype).min
    # columns 64..67 are our specials -> finite; 68.. up to the pad boundary -> masked
    assert torch.isfinite(logits[0, -1, 64:68]).all()
    assert (logits[0, -1, adapted.cfg.vocab_size :] == neg_inf).all()


def test_import_vocab_size_uses_spare_rows_when_specials_fit_under_config_vocab():
    """Qwen ships embedding rows below config.vocab_size; if the specials fit there, no growth."""
    from types import SimpleNamespace

    from lithos.serve.tokenizer_adapt import AugmentResult

    # config vocab 100, specials landed at 60..72 -> already covered, no growth needed
    res = AugmentResult(tokenizer=None, ids={f"t{i}": 60 + i for i in range(13)})
    assert import_vocab_size(SimpleNamespace(vocab_size=100), res) == 100
    # specials past config vocab -> grow to cover the highest id + 1
    res2 = AugmentResult(tokenizer=None, ids={f"t{i}": 100 + i for i in range(13)})
    assert import_vocab_size(SimpleNamespace(vocab_size=100), res2) == 113
