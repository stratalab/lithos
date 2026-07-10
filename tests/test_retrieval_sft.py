"""Retrieval-aware SFT (`lithos/posttrain/retrieval_sft.py`, E2.5).

Closes the `untrained` cause the inline arm detects: teach the model to use a reference block
and to ignore an irrelevant one. The load-bearing test is
`test_the_sft_user_turn_is_byte_identical_to_what_the_server_builds` — if train and serve ever
render the block differently, the model is trained on a format it will never be served.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from lithos.posttrain.reference import ContextPlacement, build_messages
from lithos.posttrain.retrieval_sft import (
    ABSTAIN_ANSWER,
    Context,
    ExampleKind,
    RetrievalExample,
    grounded_source_ids,
    make_distractor,
    make_mixed,
    to_messages_record,
    write_retrieval_sft,
)

BERNOULLI = Context("Bernoulli: p + 0.5*rho*v**2 is constant.", source_id="src:fluids", relevant=True)
CELL = Context("The mitochondrion is the powerhouse of the cell.", source_id="src:bio", relevant=True)
GROUNDED = RetrievalExample(
    query="how does pressure relate to velocity?",
    answer="As velocity rises, pressure falls.",
    contexts=(BERNOULLI,),
    kind=ExampleKind.GROUNDED,
)


# ── the curriculum ─────────────────────────────────────────────────────────────


def test_grounded_example_puts_the_block_in_the_user_turn_and_the_answer_last():
    rec = to_messages_record(GROUNDED)
    roles = [m["role"] for m in rec["messages"]]
    assert roles == ["user", "assistant"]
    assert "Reference material:" in rec["messages"][0]["content"]
    assert rec["messages"][-1]["content"] == "As velocity rises, pressure falls."
    assert rec["kind"] == "grounded"


def test_distractor_example_abstains():
    ex = make_distractor("what is the boiling point of mercury?", [CELL])
    rec = to_messages_record(ex)
    assert ex.kind is ExampleKind.DISTRACTOR
    assert rec["messages"][-1]["content"] == ABSTAIN_ANSWER
    # the noise is present in the prompt but marked irrelevant
    assert "mitochondrion" in rec["messages"][0]["content"]
    assert all(not c.relevant for c in ex.contexts)


def test_mixed_example_keeps_the_answer_and_adds_noise_after_the_signal():
    ex = make_mixed(GROUNDED, [CELL])
    assert ex.kind is ExampleKind.MIXED
    assert ex.answer == GROUNDED.answer
    # relevant passage precedes the distractor -> the answer isn't always "passage [1]"
    assert ex.contexts[0].relevant and not ex.contexts[-1].relevant
    content = to_messages_record(ex)["messages"][0]["content"]
    assert "Bernoulli" in content and "mitochondrion" in content


def test_make_mixed_rejects_a_non_grounded_base():
    with pytest.raises(ValueError, match="GROUNDED"):
        make_mixed(make_distractor("q", [CELL]), [BERNOULLI])


# ── the seam: train renders exactly what serve renders ─────────────────────────


def test_the_sft_user_turn_is_byte_identical_to_what_the_server_builds():
    """Both go through reference.build_messages. If they ever diverge, this fails."""
    rec = to_messages_record(GROUNDED, placement=ContextPlacement.BLOCK)
    served = build_messages(
        GROUNDED.query,
        [c.text for c in GROUNDED.contexts],
        system=None,
        placement=ContextPlacement.BLOCK,
    )
    # the SFT record is the served prompt + one assistant turn appended
    assert rec["messages"][:-1] == served


def test_inline_placement_flows_through():
    rec = to_messages_record(GROUNDED, placement=ContextPlacement.INLINE)
    content = rec["messages"][0]["content"]
    assert "Reference material:" not in content  # inline = bare prose
    assert "Bernoulli" in content


# ── provenance: grounded_on excludes the noise ─────────────────────────────────


def test_grounded_source_ids_is_the_union_of_RELEVANT_sources_only():
    mixed = make_mixed(GROUNDED, [CELL])  # CELL is now irrelevant noise
    dist = make_distractor("q", [CELL])  # all irrelevant
    ids = grounded_source_ids([GROUNDED, mixed, dist])
    assert ids == ["src:fluids"]  # never src:bio -- nothing was learned from it
    assert "src:bio" not in ids


def test_a_distractor_only_set_grounds_on_nothing():
    assert grounded_source_ids([make_distractor("q", [CELL])]) == []


# ── end to end: the built shards train on the answer, not the block ────────────

_TOKENIZER = Path("artifacts/tokenizer/fineweb-edu-32k/tokenizer.json")


@pytest.mark.skipif(not _TOKENIZER.exists(), reason="chat tokenizer artifact absent")
def test_build_sft_corpus_masks_the_block_and_targets_only_the_answer(tmp_path):
    """The reference block is in the loss-masked prompt; only the assistant answer is a target.
    This is what makes the restricted-passage-as-prompt path legal (tier gate: prompt_tier)."""
    from lithos.posttrain.sft_corpus import SFTCorpusBuildConfig, build_sft_corpus

    path = tmp_path / "rsft.jsonl"
    write_retrieval_sft([GROUNDED, make_distractor("what boils mercury?", [CELL])], path)

    manifest = build_sft_corpus(
        SFTCorpusBuildConfig(
            tokenizer_path=str(_TOKENIZER),
            # the answer is the target -> synthetic-verified + grounded_on; the block is a
            # loss-masked prompt, so it may be restricted (a textbook passage).
            sources=[
                {
                    "path": str(path),
                    "name": "rsft",
                    "tier": "synthetic-verified",
                    "prompt_tier": "restricted",
                    "grounded_on": grounded_source_ids([GROUNDED]),
                }
            ],  # type: ignore[list-item]
            output_dir=str(tmp_path / "out"),
            seq_len=256,
            tokens_per_shard=4096,
        )
    )
    # some tokens train (the answers) and most do not (the reference blocks + prompts)
    assert 0 < manifest["num_loss_tokens"] < manifest["num_tokens"]
    # the block is large relative to the answer, so the masked fraction is high
    assert manifest["num_loss_tokens"] / manifest["num_tokens"] < 0.5


def test_stats_report_the_mix(tmp_path):
    exs = [GROUNDED, make_distractor("q1", [CELL]), make_mixed(GROUNDED, [CELL])]
    stats = write_retrieval_sft(exs, tmp_path / "rsft.jsonl")
    assert stats.n == 3
    assert stats.by_kind == {"grounded": 1, "distractor": 1, "mixed": 1}
    assert stats.grounded_on == ["src:fluids"]
    # the file is real messages-JSONL
    rows = [json.loads(line) for line in (tmp_path / "rsft.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and all("messages" in r for r in rows)
