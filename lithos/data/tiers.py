"""Acquisition tiers — the gate that decides what may enter the **weights**.

**The axis is how the bytes reached us, not what the license says.** A model's
copyright exposure divides into two independent questions: whether the *output*
reproduces protected expression, and whether the *copy made to train* was lawfully
obtained. Dedup, per-work epoch caps, and regurgitation evals defend the first.
Nothing in this codebase defended the second — this module does.

Explaining Bernoulli's theorem after reading a textbook is lawful: facts and ideas
are not copyrightable, only expression is. Downloading that textbook from a shadow
library is a separate act with separate liability, however transformatively it is
later used. So we tier by acquisition:

``open``
    Explicit permissive license or public domain. USPTO, NASA/DOE/NIST, OpenStax,
    LibreTexts, Stack Exchange (CC-BY-SA), ODC-By datasets.
``lawful``
    Freely and publicly distributed by the rightsholder, no explicit license.
    arXiv, vendor datasheets, GitHub issues, NPTEL, Stanford Online. Acquisition is
    clean; this is the least-contested case.
``restricted``
    Paywalled or shadow-library acquisition. Textbooks, standards bodies.
    **Never enters the weights.** It may live in the retrieval datastore, over
    copies the operator lawfully holds, where the model *consults and cites* it
    rather than memorizing it — the posture a scholar takes, and the one that
    survived Google Books.
``synthetic-verified``
    Machine-generated and verifier-gated. May be *grounded on* a ``restricted``
    source (a teacher reads the chapter, writes a worked problem, the sandbox
    verifies it). The expression never transfers; the idea does. Permitted in the
    weights **only** while carrying ``metadata.grounded_on`` — without it the
    provenance trace is not clean, merely laundered, and Petra would surface the
    grounding anyway.

**The gate belongs on tokens that receive gradient — nothing else.** This is the same
argument as the ``tool_result`` loss mask (``docs/tir-format.md`` §2–§4): a span that
never contributed a gradient cannot be memorized from, and cannot carry training-source
attribution. Applied consistently:

===========  ===========================================  ===================
stage        gradient-bearing tokens                      gated?
===========  ===========================================  ===================
pretrain     every token                                  all text
SFT          the completion only (prompt is loss-masked)  **targets only**
DPO          ``chosen`` and ``rejected``                  both
RLVR/GRPO    the policy's own rollouts                    **nothing external**
TIR          all but ``tool_result`` (masked)             targets only
===========  ===========================================  ===================

So a *restricted* problem statement may be an SFT prompt or an RLVR prompt — it is
never a training target. What it may not be is an assistant *target*, because that is
transcription, not teaching. Reading the textbook and copying the textbook into the
weights are different acts, and ``synthetic-verified`` + ``grounded_on`` is the path
from the first to the weights: a teacher reads the chapter, writes a worked problem in
its own words, the sandbox verifies it, and *that* is trained on.

Note the direction of risk, which is counterintuitive: **SFT memorizes harder than
pretraining.** Memorization tracks repetition, and SFT sees few tokens over several
epochs with the loss concentrated on the target (``SFTSourceSpec.repeats`` upsamples
deliberately). A paragraph seen once in a trillion-token stream is not the same hazard
as the same paragraph as a target, three times.

What this gate does **not** do: cure acquisition. The copy was made at download time and
barring the text from the weights does not un-make it. What it buys is (a) the corpus
manifest becomes an **attestation** — provably zero ``restricted`` documents entered the
weights — and (b) restricted expression is confined to the retrieval channel, where every
use is quotation with a citation.

Fail-closed by design: a source that does not declare its tier is ``unknown`` and
**cannot be trained on**. Stating a policy is not enforcing one — a policy you can
state is a policy you will eventually violate at 2am.

Chisel imports this module rather than reimplementing it, so the two repos cannot
drift on what "trainable" means. See ``docs/chisel-tier-gate.md``.
"""

from __future__ import annotations

from typing import Any, Final, Literal

Tier = Literal["open", "lawful", "restricted", "synthetic-verified", "unknown"]

TIER_OPEN: Final[Tier] = "open"
TIER_LAWFUL: Final[Tier] = "lawful"
TIER_RESTRICTED: Final[Tier] = "restricted"
TIER_SYNTHETIC_VERIFIED: Final[Tier] = "synthetic-verified"
TIER_UNKNOWN: Final[Tier] = "unknown"

ALL_TIERS: Final[frozenset[str]] = frozenset(
    {TIER_OPEN, TIER_LAWFUL, TIER_RESTRICTED, TIER_SYNTHETIC_VERIFIED, TIER_UNKNOWN}
)

#: Tiers whose text may be tokenized into training shards. ``restricted`` is excluded
#: (datastore only); ``unknown`` is excluded (fail-closed).
WEIGHTS_ALLOWED_TIERS: Final[frozenset[str]] = frozenset(
    {TIER_OPEN, TIER_LAWFUL, TIER_SYNTHETIC_VERIFIED}
)

#: Every tier may be indexed for retrieval — the model cites what it consults.
DATASTORE_ALLOWED_TIERS: Final[frozenset[str]] = ALL_TIERS - {TIER_UNKNOWN}


class TierViolation(ValueError):
    """A document whose acquisition tier bars it from the training corpus."""


def tier_of(doc: dict[str, Any]) -> str:
    """The document's acquisition tier; ``unknown`` when undeclared (fail-closed)."""
    tier = doc.get("tier")
    return tier if isinstance(tier, str) and tier else TIER_UNKNOWN


def is_trainable(doc: dict[str, Any], *, allowed: frozenset[str] = WEIGHTS_ALLOWED_TIERS) -> bool:
    """True when this document's text may enter the weights."""
    tier = tier_of(doc)
    if tier not in allowed:
        return False
    if tier == TIER_SYNTHETIC_VERIFIED:
        return bool(doc.get("metadata", {}).get("grounded_on"))
    return True


def assert_prompt_source(tier: str, *, where: str = "") -> None:
    """Validate a tier for **loss-masked** text: SFT prompts, RLVR problem statements.

    Any real tier passes, ``restricted`` included — a masked span receives no gradient,
    so the model is never trained to *produce* it. Only ``unknown`` fails, because an
    undeclared provenance cannot be attested either way.
    """
    if tier == TIER_UNKNOWN or tier not in ALL_TIERS:
        raise TierViolation(
            f"{where or 'prompt source'}: tier {tier!r} is undeclared or unknown. Masked "
            f"prompt text may be any declared tier (it receives no gradient), but its "
            f"provenance must still be stated."
        )


def assert_trainable(
    doc: dict[str, Any], *, allowed: frozenset[str] = WEIGHTS_ALLOWED_TIERS
) -> None:
    """Raise ``TierViolation`` if this document's **gradient-bearing** text may not enter
    the weights. For loss-masked spans (SFT/RLVR prompts) use ``assert_prompt_source``.

    Loud, not silent: a skipped record would quietly shrink a corpus, and a whole
    restricted shard would yield an empty build with no error — the same reasoning
    that makes an empty glob a ``FileNotFoundError`` in ``documents._expand_paths``.
    """
    tier = tier_of(doc)
    where = f"id={doc.get('id', '')!r} source={doc.get('source', '')!r}"

    if tier == TIER_UNKNOWN:
        raise TierViolation(
            f"{where}: acquisition tier is undeclared. Every source must declare "
            f"`tier` (one of {sorted(ALL_TIERS - {TIER_UNKNOWN})}). Fail-closed: an "
            f"undeclared source cannot enter the weights."
        )
    if tier not in ALL_TIERS:
        raise TierViolation(f"{where}: unknown tier {tier!r}; expected one of {sorted(ALL_TIERS)}")
    if tier == TIER_RESTRICTED:
        raise TierViolation(
            f"{where}: tier='restricted' may never enter the weights (paywalled or "
            f"shadow-library acquisition). Index it in the retrieval datastore instead, "
            f"over copies you lawfully hold, where the model cites what it consults."
        )
    if tier not in allowed:
        raise TierViolation(f"{where}: tier={tier!r} not in allowed tiers {sorted(allowed)}")
    if tier == TIER_SYNTHETIC_VERIFIED and not doc.get("metadata", {}).get("grounded_on"):
        raise TierViolation(
            f"{where}: tier='synthetic-verified' requires `metadata.grounded_on` "
            f"(the source_ids it was derived from). Without it the provenance trace is "
            f"laundered, not clean — and Petra surfaces the grounding regardless."
        )
