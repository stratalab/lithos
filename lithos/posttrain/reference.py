"""How retrieved passages are rendered into a prompt — **one implementation, both sides**.

Serving and training must agree on this byte for byte. They did not: `_CONTEXT_HEADER` used
to live in `lithos/serve/composite.py` and nowhere else, which meant the served model read a
``Reference material:`` block it had never once seen in training. That is not a cosmetic
mismatch. It is a **third cause** for C-CTX, one that predicts the same benchmark numbers as
the two we pre-registered:

    (a) capability   a small model cannot integrate a retrieved fact into a derivation
    (b) displacement the passages ate the context the reasoning needed
    (c) untrained    the model was never taught what a reference block *is*

`docs/composite-plan.md` §4 Gate 1 asks "has the baseline received every trick the composite
gets for free?" The mirror obligation is that the composite must have had a fair shot too.
Rendering lives here so SFT, RLVR rollouts, and the server import the same function, for the
same reason `validate_tir_record` is shared rather than reimplemented.

## Placements

``BLOCK``
    The served format. A header plus ``[n]``-numbered passages, then the query. Legible,
    citable, and **unfamiliar** to a model that has not been trained on it.
``INLINE``
    The passage as ordinary prose before the question — no header, no markers. This is a
    shape every LM has seen a billion times in pretraining (a document, then a question).
    It is the **control that detects cause (c)**: if a model can use a fact inline but not
    in a block, its failure is format, not capability.
``SYSTEM``
    The block, carried in the system turn instead of the user turn.

No new special tokens: the 32k tokenizer's specials are fixed at IDs 0–6 and the TIR block
above them. Reference material is plain text in an existing turn.

## An honest gap in the identity tuple

Changing ``REFERENCE_FORMAT_VERSION`` changes the model's output for identical weights,
datastore, policy, and tool env — yet ``ServedModelId`` has four components and none is the
prompt format. We do **not** quietly widen the doctrine's tuple here. The format version is
recorded per-episode instead, and the tension is written down rather than resolved by
stealth. See `docs/composite-plan.md`.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Final

#: Bump when the rendering changes. Recorded on every episode; a corpus of SFT data rendered
#: at ref-v1 is not interchangeable with a server rendering at ref-v2.
REFERENCE_FORMAT_VERSION: Final[str] = "ref-v1"

REFERENCE_HEADER: Final[str] = "Reference material:"


class ContextPlacement(StrEnum):
    BLOCK = "block"
    INLINE = "inline"
    SYSTEM = "system"


def render_reference_block(contexts: Sequence[str]) -> str:
    """``Reference material:\\n[1] …\\n[2] …`` — the numbered, citable form."""
    if not contexts:
        return ""
    lines = "\n".join(f"[{i}] {t}" for i, t in enumerate(contexts, start=1))
    return f"{REFERENCE_HEADER}\n{lines}"


def render_inline(contexts: Sequence[str]) -> str:
    """The passages as bare prose, no header and no markers."""
    return "\n\n".join(contexts)


def build_messages(
    query: str,
    contexts: Sequence[str] = (),
    *,
    system: str | None = None,
    placement: ContextPlacement = ContextPlacement.BLOCK,
) -> list[dict[str, str]]:
    """The chat messages for a (possibly retrieval-augmented) query.

    With no contexts every placement collapses to the plain query, so the ``none`` arm and an
    ordinary closed-book prompt are the same string — not two strings that happen to match.
    """
    contexts = [c for c in contexts if c]
    msgs: list[dict[str, str]] = []

    if not contexts:
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": query})
        return msgs

    if placement is ContextPlacement.SYSTEM:
        block = render_reference_block(contexts)
        merged = f"{system}\n\n{block}" if system else block
        msgs.append({"role": "system", "content": merged})
        msgs.append({"role": "user", "content": query})
        return msgs

    if system:
        msgs.append({"role": "system", "content": system})

    if placement is ContextPlacement.BLOCK:
        content = f"{render_reference_block(contexts)}\n\n{query}"
    elif placement is ContextPlacement.INLINE:
        content = f"{render_inline(contexts)}\n\n{query}"
    else:  # pragma: no cover - StrEnum is exhaustive
        raise ValueError(f"unknown placement {placement!r}")

    msgs.append({"role": "user", "content": content})
    return msgs
