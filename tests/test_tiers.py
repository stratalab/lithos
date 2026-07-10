"""The acquisition-tier gate: what may enter the weights (`lithos.data.tiers`).

The axis is how the bytes reached us, not what the license says. `restricted` (paywalled
or shadow-library acquisition) is barred from the training corpus and belongs in the
retrieval datastore, where the model cites what it consults. Fail-closed: an undeclared
source cannot be trained on.

Chisel mirrors this module rather than reimplementing it, so "trainable" cannot drift
between the repos. See `docs/chisel-tier-gate.md`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from lithos.data.pipeline import TierPolicy
from lithos.data.tiers import (
    DATASTORE_ALLOWED_TIERS,
    TIER_LAWFUL,
    TIER_OPEN,
    TIER_RESTRICTED,
    TIER_SYNTHETIC_VERIFIED,
    TIER_UNKNOWN,
    WEIGHTS_ALLOWED_TIERS,
    TierViolation,
    assert_trainable,
    is_trainable,
    tier_of,
)


def _doc(tier: str | None = None, **kw):
    d = {"id": "d1", "text": "x", "source": "s", "metadata": {}}
    if tier is not None:
        d["tier"] = tier
    d.update(kw)
    return d


# ── the vocabulary ────────────────────────────────────────────────────────────


def test_restricted_is_barred_from_weights_but_allowed_in_the_datastore():
    assert TIER_RESTRICTED not in WEIGHTS_ALLOWED_TIERS
    assert TIER_RESTRICTED in DATASTORE_ALLOWED_TIERS


def test_unknown_is_barred_everywhere():
    assert TIER_UNKNOWN not in WEIGHTS_ALLOWED_TIERS
    assert TIER_UNKNOWN not in DATASTORE_ALLOWED_TIERS


def test_open_and_lawful_are_trainable():
    for t in (TIER_OPEN, TIER_LAWFUL):
        assert is_trainable(_doc(t)), t
        assert_trainable(_doc(t))


# ── fail-closed ───────────────────────────────────────────────────────────────


def test_undeclared_tier_is_unknown_and_raises():
    doc = _doc()  # no `tier` key at all
    assert tier_of(doc) == TIER_UNKNOWN
    assert not is_trainable(doc)
    with pytest.raises(TierViolation, match="undeclared"):
        assert_trainable(doc)


def test_empty_tier_string_is_unknown_not_trainable():
    assert tier_of(_doc("")) == TIER_UNKNOWN
    with pytest.raises(TierViolation, match="undeclared"):
        assert_trainable(_doc(""))


def test_restricted_raises_and_says_where_it_belongs():
    with pytest.raises(TierViolation, match="never enter the weights"):
        assert_trainable(_doc(TIER_RESTRICTED))
    with pytest.raises(TierViolation, match="datastore"):
        assert_trainable(_doc(TIER_RESTRICTED))


def test_violation_message_names_the_offending_document():
    with pytest.raises(TierViolation, match=r"id='book-42'.*source='pearson'"):
        assert_trainable(_doc(TIER_RESTRICTED, id="book-42", source="pearson"))


def test_bogus_tier_raises():
    with pytest.raises(TierViolation):
        assert_trainable(_doc("opne"))


# ── synthetic-verified requires grounding ─────────────────────────────────────


def test_synthetic_verified_requires_grounded_on():
    bare = _doc(TIER_SYNTHETIC_VERIFIED)
    assert not is_trainable(bare)
    with pytest.raises(TierViolation, match="grounded_on"):
        assert_trainable(bare)


def test_synthetic_verified_with_grounding_is_trainable():
    doc = _doc(TIER_SYNTHETIC_VERIFIED, metadata={"grounded_on": ["src:pearson-fluids-ch7"]})
    assert is_trainable(doc)
    assert_trainable(doc)


def test_grounding_may_point_at_a_restricted_source():
    """The expression never transfers; the idea does. That is the whole point — but the
    grounding must be *recorded*, not laundered."""
    doc = _doc(TIER_SYNTHETIC_VERIFIED, metadata={"grounded_on": ["src:some-restricted-book"]})
    assert_trainable(doc)


# ── the record carries it, and the source declares it ─────────────────────────


def test_normalize_carries_tier_and_record_overrides_source_default():
    from lithos.data.documents import normalize

    doc = normalize({"text": "x"}, source="s", subset=None, language="en", license="l", tier="open")
    assert doc is not None and doc["tier"] == TIER_OPEN

    doc = normalize(
        {"text": "x", "tier": "restricted"},
        source="s",
        subset=None,
        language="en",
        license="l",
        tier="open",
    )
    assert doc is not None and doc["tier"] == TIER_RESTRICTED  # record wins, and it will fail


def test_normalize_defaults_to_unknown_when_source_omits_tier():
    from lithos.data.documents import normalize

    doc = normalize({"text": "x"}, source="s", subset=None, language="en", license="l")
    assert doc is not None and doc["tier"] == TIER_UNKNOWN


def test_document_source_rejects_a_typod_tier_at_config_load():
    from lithos.data.documents import DocumentSource
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DocumentSource(kind="jsonl", paths=["x"], tier="publik")  # type: ignore[arg-type]


# ── the gate, end to end through build_corpus ─────────────────────────────────


def _tiny_corpus(tmp_path, tier: str | None):
    p = tmp_path / "docs.jsonl"
    rec: dict = {"text": "alpha beta gamma delta"}
    if tier is not None:
        rec["tier"] = tier
    p.write_text(json.dumps(rec) + "\n")
    return p


def _build(tmp_path, tokenizer_path, *, src_tier, rec_tier=None, enforce=True):
    from lithos.data.pipeline import CorpusBuildConfig, build_corpus

    src = {
        "kind": "jsonl",
        "paths": [str(_tiny_corpus(tmp_path, rec_tier))],
        "source_name": "t",
        "license": "x",
    }
    if src_tier is not None:
        src["tier"] = src_tier
    cfg = CorpusBuildConfig(
        name="t",
        tokenizer_path=str(tokenizer_path),
        sources=[src],  # type: ignore[list-item]
        output_dir=str(tmp_path / "out"),
        seq_len=8,
        tokens_per_shard=64,
        tiers=TierPolicy(enforce=enforce),
    )
    return build_corpus(cfg)


@pytest.fixture
def tokenizer_path(tmp_path):
    """A minimal byte-level BPE, enough to tokenize the fixture text."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    trainer = trainers.BpeTrainer(vocab_size=300, special_tokens=["<unk>", "<s>", "</s>"])
    tok.train_from_iterator(["alpha beta gamma delta epsilon zeta"], trainer)
    p = tmp_path / "tok" / "tokenizer.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(p))
    return p


def test_build_corpus_rejects_restricted_documents(tmp_path, tokenizer_path):
    with pytest.raises(TierViolation, match="never enter the weights"):
        _build(tmp_path, tokenizer_path, src_tier="restricted")


def test_build_corpus_rejects_an_undeclared_source(tmp_path, tokenizer_path):
    with pytest.raises(TierViolation, match="undeclared"):
        _build(tmp_path, tokenizer_path, src_tier=None)


def test_build_corpus_rejects_a_restricted_record_inside_an_open_source(tmp_path, tokenizer_path):
    """Per-record, not per-source: a mixed shard cannot smuggle one restricted doc."""
    with pytest.raises(TierViolation, match="never enter the weights"):
        _build(tmp_path, tokenizer_path, src_tier="open", rec_tier="restricted")


def test_build_corpus_accepts_open_and_attests_it_in_the_manifest(tmp_path, tokenizer_path):
    manifest = _build(tmp_path, tokenizer_path, src_tier="open")
    assert manifest["tiers"]["counts"] == {"open": 1}
    assert manifest["tiers"]["policy"]["enforce"] is True
    assert "restricted" not in manifest["tiers"]["counts"]
    assert manifest["tiers"]["synthetic_grounded"] == 0


def test_disabling_the_gate_is_recorded_in_the_manifest(tmp_path, tokenizer_path):
    """Turning it off is permitted and auditable — the manifest says so."""
    manifest = _build(tmp_path, tokenizer_path, src_tier="restricted", enforce=False)
    assert manifest["tiers"]["policy"]["enforce"] is False
    assert manifest["tiers"]["counts"] == {"restricted": 1}


# ── the gate follows the gradient, not the stage ──────────────────────────────
#
# Same argument as the `tool_result` loss mask: a span that never contributed a gradient
# cannot be memorized from. SFT *targets* are gated; SFT *prompts* are not, because the
# loss mask zeroes them — a restricted textbook problem is a stimulus, never a target.
# RLVR needs no gate at all: the only gradient-bearing tokens are the policy's rollouts.


def test_prompt_source_permits_restricted_because_it_is_masked():
    from lithos.data.tiers import assert_prompt_source

    assert_prompt_source(TIER_RESTRICTED)  # a textbook problem statement
    assert_prompt_source(TIER_OPEN)


def test_prompt_source_still_rejects_undeclared_provenance():
    from lithos.data.tiers import assert_prompt_source

    with pytest.raises(TierViolation, match="undeclared or unknown"):
        assert_prompt_source(TIER_UNKNOWN)


def test_prompt_tier_defaults_to_target_tier():
    from lithos.posttrain.sft_corpus import SFTSourceSpec

    s = SFTSourceSpec(path="p", name="n", tier="open")
    assert s.masked_prompt_tier() == TIER_OPEN and s.target_tier() == TIER_OPEN


def _sft_cfg(tmp_path, tokenizer_path, *, tier, grounded_on=None, prompt_tier=None):
    from lithos.posttrain.sft_corpus import SFTCorpusBuildConfig

    p = tmp_path / "sft.jsonl"
    p.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "alpha beta"},
                ]
            }
        )
        + "\n"
    )
    src: dict = {"path": str(p), "name": "s1"}
    if tier is not None:
        src["tier"] = tier
    if grounded_on is not None:
        src["grounded_on"] = grounded_on
    if prompt_tier is not None:
        src["prompt_tier"] = prompt_tier
    return SFTCorpusBuildConfig(
        tokenizer_path=str(tokenizer_path),
        sources=[src],  # type: ignore[list-item]
        output_dir=str(tmp_path / "sftout"),
        seq_len=16,
        tokens_per_shard=64,
    )


def test_sft_source_spec_defaults_to_unknown_tier():
    from lithos.posttrain.sft_corpus import SFTSourceSpec

    assert SFTSourceSpec(path="p", name="n").tier == TIER_UNKNOWN


def test_build_sft_corpus_rejects_restricted(tmp_path, tokenizer_path):
    from lithos.posttrain.sft_corpus import build_sft_corpus

    with pytest.raises(TierViolation, match="never enter the weights"):
        build_sft_corpus(_sft_cfg(tmp_path, tokenizer_path, tier="restricted"))


def test_build_sft_corpus_rejects_an_undeclared_source(tmp_path, tokenizer_path):
    from lithos.posttrain.sft_corpus import build_sft_corpus

    with pytest.raises(TierViolation, match="undeclared"):
        build_sft_corpus(_sft_cfg(tmp_path, tokenizer_path, tier=None))


def test_build_sft_corpus_rejects_ungrounded_synthetic(tmp_path, tokenizer_path):
    from lithos.posttrain.sft_corpus import build_sft_corpus

    with pytest.raises(TierViolation, match="grounded_on"):
        build_sft_corpus(_sft_cfg(tmp_path, tokenizer_path, tier="synthetic-verified"))


def test_build_sft_corpus_fails_before_rendering_any_token(tmp_path, tokenizer_path):
    """The gate runs up front: a bad blend costs no work and writes no output dir."""
    from lithos.posttrain.sft_corpus import build_sft_corpus

    cfg = _sft_cfg(tmp_path, tokenizer_path, tier="restricted")
    with pytest.raises(TierViolation):
        build_sft_corpus(cfg)
    assert not (tmp_path / "sftout" / "tokenized").exists()


#: The real chat tokenizer — the positive path actually renders, so it needs the chat
#: special tokens. The negative paths raise at the gate before any tokenizer is touched.
_CHAT_TOKENIZER = Path("artifacts/tokenizer/fineweb-edu-32k/tokenizer.json")


@pytest.mark.skipif(not _CHAT_TOKENIZER.exists(), reason="chat tokenizer artifact absent")
def test_restricted_prompt_with_a_verified_derived_target_is_allowed(tmp_path):
    """The whole point. A restricted textbook problem as the *prompt* (loss-masked), and a
    teacher's sandbox-verified solution as the *target*. Read the book; write your own
    explanation. This is the path from a textbook to the weights, and it is one hop."""
    from lithos.posttrain.sft_corpus import build_sft_corpus

    manifest = build_sft_corpus(
        _sft_cfg(
            tmp_path,
            _CHAT_TOKENIZER,
            tier="synthetic-verified",
            grounded_on=["src:pearson-fluids-ch7"],
            prompt_tier="restricted",
        )
    )
    assert manifest["num_tokens"] > 0
    assert manifest["num_loss_tokens"] > 0  # the derived target trained; the prompt did not
    assert manifest["num_loss_tokens"] < manifest["num_tokens"]  # the prompt was masked


def test_a_restricted_prompt_cannot_rescue_a_restricted_target(tmp_path, tokenizer_path):
    """Declaring the prompt does not launder the target."""
    from lithos.posttrain.sft_corpus import build_sft_corpus

    with pytest.raises(TierViolation, match="never enter the weights"):
        build_sft_corpus(
            _sft_cfg(tmp_path, tokenizer_path, tier="restricted", prompt_tier="restricted")
        )
