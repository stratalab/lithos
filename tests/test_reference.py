"""The reference-rendering seam (`lithos/posttrain/reference.py`).

One implementation, both sides. `_CONTEXT_HEADER` used to live only in the serving path,
so the model read a `Reference material:` block it had never once seen in training — a
train/serve mismatch that gives C-CTX a third cause (`untrained`) predicting the same
numbers as `capability`.

The load-bearing test is `test_the_server_renders_exactly_what_training_would`: if the
composite ever re-invents the format, that test fails.
"""

from __future__ import annotations

import pytest
from lithos.posttrain.reference import (
    REFERENCE_FORMAT_VERSION,
    REFERENCE_HEADER,
    ContextPlacement,
    build_messages,
    render_inline,
    render_reference_block,
)

Q = "what is the pressure drop?"
C1 = "Bernoulli: p + 0.5*rho*v**2 is constant."
C2 = "Kirchhoff's current law: sum of currents at a node is zero."


# ── the formats differ in exactly the way the experiment needs ────────────────


def test_block_has_a_header_and_numbered_markers():
    out = render_reference_block([C1, C2])
    assert out.startswith(REFERENCE_HEADER)
    assert "[1] " + C1 in out and "[2] " + C2 in out


def test_inline_has_neither_header_nor_markers():
    """The point of the control: a shape the model has seen a billion times in pretraining."""
    out = render_inline([C1, C2])
    assert REFERENCE_HEADER not in out
    assert "[1]" not in out and "[2]" not in out
    assert C1 in out and C2 in out


def test_empty_contexts_render_to_nothing():
    assert render_reference_block([]) == ""
    assert render_inline([]) == ""


# ── build_messages ────────────────────────────────────────────────────────────


def test_no_contexts_collapses_to_the_plain_query_for_every_placement():
    """The `none` arm and an ordinary closed-book prompt must be the *same string*, not two
    strings that happen to match."""
    for placement in ContextPlacement:
        msgs = build_messages(Q, [], placement=placement)
        assert msgs == [{"role": "user", "content": Q}]


def test_block_placement_puts_the_block_before_the_query_in_the_user_turn():
    msgs = build_messages(Q, [C1], placement=ContextPlacement.BLOCK)
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    content = msgs[0]["content"]
    assert content.startswith(REFERENCE_HEADER)
    assert content.endswith(Q)


def test_inline_placement_puts_bare_prose_before_the_query():
    msgs = build_messages(Q, [C1], placement=ContextPlacement.INLINE)
    content = msgs[0]["content"]
    assert content == f"{C1}\n\n{Q}"
    assert REFERENCE_HEADER not in content


def test_system_placement_moves_the_block_into_a_system_turn():
    msgs = build_messages(Q, [C1], placement=ContextPlacement.SYSTEM)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert REFERENCE_HEADER in msgs[0]["content"]
    assert msgs[1]["content"] == Q


def test_system_placement_merges_with_a_caller_supplied_system_prompt():
    msgs = build_messages(Q, [C1], system="You are terse.", placement=ContextPlacement.SYSTEM)
    assert msgs[0]["content"].startswith("You are terse.")
    assert REFERENCE_HEADER in msgs[0]["content"]


def test_system_prompt_survives_block_and_inline_placements():
    for placement in (ContextPlacement.BLOCK, ContextPlacement.INLINE):
        msgs = build_messages(Q, [C1], system="You are terse.", placement=placement)
        assert [m["role"] for m in msgs] == ["system", "user"]
        assert msgs[0]["content"] == "You are terse."


def test_blank_contexts_are_dropped_not_rendered_as_empty_markers():
    msgs = build_messages(Q, ["", C1, ""], placement=ContextPlacement.BLOCK)
    assert "[2]" not in msgs[0]["content"]
    assert "[1] " + C1 in msgs[0]["content"]


def test_inline_and_block_carry_the_same_facts_differently():
    """Same information, different shape — the whole basis of the `inline` arm."""
    b = build_messages(Q, [C1], placement=ContextPlacement.BLOCK)[0]["content"]
    i = build_messages(Q, [C1], placement=ContextPlacement.INLINE)[0]["content"]
    assert C1 in b and C1 in i
    assert b != i
    assert len(b) > len(i)  # the header and markers cost tokens; C-CTX measures the cost


def test_format_version_is_exported_and_stable():
    assert REFERENCE_FORMAT_VERSION == "ref-v1"


def test_unknown_placement_raises():
    with pytest.raises(ValueError, match="unknown placement"):
        build_messages(Q, [C1], placement="sideways")  # type: ignore[arg-type]


# ── the seam: the server must not re-invent the format ────────────────────────


def test_the_server_renders_exactly_what_training_would():
    """`CompositeModel._build_messages` delegates here. If it ever forks, this fails.

    This is the same argument that made `validate_tir_record` shared rather than
    reimplemented: a format the server invents is a format the model never saw.
    """
    from types import SimpleNamespace

    from lithos.serve.composite import CompositeModel

    fake = SimpleNamespace(text=C1)
    built = CompositeModel._build_messages(
        SimpleNamespace(),  # type: ignore[arg-type]  # method uses no instance state
        Q,
        [fake],  # type: ignore[list-item]
        None,
        ContextPlacement.BLOCK,
    )
    assert built == build_messages(Q, [C1], system=None, placement=ContextPlacement.BLOCK)
