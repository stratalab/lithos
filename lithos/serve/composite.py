"""The composite model layer: a walking skeleton.

The thing that emits tokens is the weights **plus** something else, and the harness
cannot tell. This module is the seam where "something else" attaches.

Per `docs/composite-plan.md` (rev B, post-literature-sweep) the architecture is:

- **Retrieval lives ABOVE the token stream.** Passages are prepended to the prompt;
  they are *cited*, never interpolated into the decode loop. Every mechanism with
  positive evidence lives above the stream; every one with negative evidence lives
  below it. So there is no kNN-LM hook here, and no Moho.
- **The tool loop lives BELOW it** (`tir_rollout`): the model emits ``<|python|>``,
  the server pauses, the sandbox executes, the result is injected and decoding
  resumes. The *judgment* to call the tool is trained in; the *execution* never can
  be. That single clause is why TIR survives the absorption test.
- **The decode policy fixes the support.** Applied first, to the raw logits, and
  final because every later stage only removes mass. Enforced in
  `lithos/model/generation._apply_decode_policy`, not by convention.

Retrieval costs **context**, the scarcest resource a 500M has. ``CompositeResult``
therefore measures it: ``context_tokens`` is the C-CTX instrument
(`docs/composite-plan.md` §3) — the experiment that decides whether a small reasoner
is capability-limited or displacement-limited.

Stubs (`StubRetriever`, no policy) make the skeleton runnable end to end today; each
is replaced independently.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from lithos.model.generation import LogitsProcessor
from lithos.posttrain.chat_template import render_prompt, special_ids, tir_token_ids
from lithos.posttrain.reference import (
    REFERENCE_FORMAT_VERSION,
    ContextPlacement,
    build_messages,
)
from lithos.posttrain.sandbox import tool_env_sha
from lithos.posttrain.tir_rollout import tir_rollout

# Re-exported: the retrieval seam lives in `lithos.retrieval` so the dependency runs one
# way (serve -> retrieval). A retriever never needs to know it is being served.
from lithos.retrieval.types import Passage, RetrievedContext, Retriever, StubRetriever

__all__ = [
    "Citation",
    "CompositeModel",
    "CompositeResult",
    "ContextPlacement",
    "DenyTokensPolicy",
    "Passage",
    "RetrievedContext",
    "Retriever",
    "ServedModelId",
    "StubRetriever",
    "ToolCallRecord",
]

# ── identity ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ServedModelId:
    """A served model is no longer a weights file (`docs/composite-model-layer.md` §7.1).

    The four components *are* the parts list: weights = Lithos, datastore = StrataDB,
    decode policy = Lithos's decode-time enforcement, tool env = the sandbox. Evals, bisects,
    and incident reports
    record all four, or they are not about the same system.
    """

    weights_sha256: str
    datastore_version: str | None  # None = no retrieval attached
    decode_policy_version: str
    tool_env_sha: str

    def as_tuple(self) -> tuple[str, str | None, str, str]:
        return (
            self.weights_sha256,
            self.datastore_version,
            self.decode_policy_version,
            self.tool_env_sha,
        )

    def digest(self) -> str:
        """One stable id for the whole composite. Changing *any* component changes it."""
        blob = json.dumps(self.as_tuple(), separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def __str__(self) -> str:
        ds = self.datastore_version or "-"
        return (
            f"lithos:{self.weights_sha256[:12]}+ds:{ds}"
            f"+policy:{self.decode_policy_version}+tools:{self.tool_env_sha[:12]}"
        )


# ── decode policy ─────────────────────────────────────────────────────────────


class DenyTokensPolicy:
    """A minimal decode-policy primitive: these token ids may never be emitted.

    Only removes mass, so it satisfies the last-write invariant by construction.
    """

    def __init__(self, deny: set[int], *, version: str = "deny-v0") -> None:
        self.deny = frozenset(deny)
        self.version = version

    def __call__(self, logits: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
        if not self.deny:
            return logits
        out = logits.clone()
        idx = torch.tensor(sorted(self.deny), device=logits.device, dtype=torch.long)
        out[:, idx] = float("-inf")
        return out


# ── provenance (out of band, alongside the tokens) ────────────────────────────


@dataclass(frozen=True)
class Citation:
    """A retrieved fact is citable **by construction** — this is R1's real pitch."""

    source_id: str
    record_id: str
    text_sha256: str  # the parent document: the join key to Chisel and Petra
    tier: str
    tokens: int  # what this passage cost the context budget
    chunk_sha256: str = ""  # the exact span, when the retriever provides one


@dataclass(frozen=True)
class ToolCallRecord:
    runtime: str
    code: str
    output: str


@dataclass
class CompositeResult:
    """What the composite emits: tokens, plus the provenance the tokens don't carry."""

    model_id: ServedModelId
    text: str
    token_ids: list[int]
    action_mask: list[bool]
    citations: list[Citation] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    prompt_tokens: int = 0
    #: Of ``prompt_tokens``, how many the retrieved passages ate. **The C-CTX instrument.**
    context_tokens: int = 0
    #: Tokens the model was *allowed* to generate. Under a total budget L this is
    #: ``L - prompt_tokens`` — so prepended passages shrink it, and that is displacement.
    completion_budget: int = 0
    #: How the passages were rendered, and by which version of the renderer. Changing either
    #: changes the model's output for identical weights/datastore/policy/tool-env — an honest
    #: gap in the four-tuple, recorded per-response rather than papered over.
    context_placement: str = ""
    reference_format_version: str = REFERENCE_FORMAT_VERSION
    truncated: bool = False

    @property
    def completion_tokens(self) -> int:
        return len(self.token_ids) - self.prompt_tokens

    @property
    def reasoning_tokens(self) -> int:
        """Completion tokens the *policy* produced — injected tool results excluded.

        The quantity displaced when retrieved passages eat the context window.
        """
        return sum(1 for m in self.action_mask[self.prompt_tokens :] if m)

    def provenance(self) -> dict[str, Any]:
        """The out-of-band channel: which record, which tool, per response."""
        return {
            "served_model_id": self.model_id.as_tuple(),
            "digest": self.model_id.digest(),
            "citations": [c.__dict__ for c in self.citations],
            "tool_calls": [
                {"runtime": t.runtime, "code": t.code, "output": t.output} for t in self.tool_calls
            ],
            "context_tokens": self.context_tokens,
            "completion_budget": self.completion_budget,
            "reasoning_tokens": self.reasoning_tokens,
        }


# ── the composite ─────────────────────────────────────────────────────────────


class CompositeModel:
    """Weights + retrieval + tools + policy, behind one ``generate``.

    To the caller this is a model. Below the token stream it is four subsystems, and
    the identity tuple names each one.
    """

    def __init__(
        self,
        model: Any,
        tok: Any,
        *,
        weights_sha256: str,
        retriever: Retriever | None = None,
        datastore_version: str | None = None,
        policy: LogitsProcessor | None = None,
        policy_version: str = "none",
        device: str = "cpu",
    ) -> None:
        # A retriever that knows its own content hash supplies it; nobody should have to
        # restate a derived value, and a hand-written one can lie.
        if retriever is not None and datastore_version is None:
            datastore_version = getattr(retriever, "datastore_version", None)
        if retriever is not None and datastore_version is None:
            raise ValueError(
                "a retriever without a datastore_version is unevaluable: pin the version "
                "or the same prompt gives a different answer next month "
                "(docs/composite-model-layer.md §7.1)"
            )
        self.model = model
        self.tok = tok
        self.retriever = retriever
        self.policy = policy
        self.device = device
        self._sids = special_ids(tok)
        self._tir_ids = tir_token_ids(tok)
        self.id = ServedModelId(
            weights_sha256=weights_sha256,
            datastore_version=datastore_version,
            decode_policy_version=getattr(policy, "version", policy_version),
            tool_env_sha=tool_env_sha(),
        )

    def _build_messages(
        self,
        query: str,
        passages: Sequence[Passage],
        system: str | None,
        placement: ContextPlacement = ContextPlacement.BLOCK,
    ) -> list[dict[str, str]]:
        """Delegates to `lithos.posttrain.reference` — the ONE renderer that training also
        imports. The server does not get to invent a prompt format the model never saw."""
        return build_messages(
            query, [p.text for p in passages], system=system, placement=placement
        )

    def _fit_to_budget(
        self,
        query: str,
        ctx: RetrievedContext,
        system: str | None,
        budget: int,
        placement: ContextPlacement,
    ):
        """Drop lowest-ranked passages until the context fits ``budget`` tokens.

        The cost is measured as ``len(prompt_with) - len(prompt_without)``: exact, and
        immune to BPE merging across the passage/query seam. Measured per placement, since
        a block's header and ``[n]`` markers cost tokens that inline prose does not.
        """
        bare = len(render_prompt(self._build_messages(query, (), system, placement), self.tok))
        kept = list(ctx.passages)
        while kept:
            full = len(
                render_prompt(self._build_messages(query, kept, system, placement), self.tok)
            )
            if full - bare <= budget:
                return tuple(kept), full - bare, bare
            kept.pop()  # drop the last (lowest-ranked) passage and re-measure
        return (), 0, bare

    def generate(
        self,
        query: str,
        *,
        system: str | None = None,
        context_token_budget: int = 0,
        max_new: int = 128,
        total_token_budget: int | None = None,
        charge_context: bool = True,
        placement: ContextPlacement = ContextPlacement.BLOCK,
        max_tool_calls: int = 2,
        temperature: float = 1.0,
        top_p: float | None = 0.95,
        generator: torch.Generator | None = None,
        use_cache: bool = True,
    ) -> CompositeResult:
        """Generate one composite response.

        ``total_token_budget`` (*L*) caps ``prompt + completion``, as a real deployment's
        sequence length does. When set, the completion budget is ``L - prompt_tokens`` —
        so retrieved passages, being part of the prompt, **displace** reasoning tokens.

        ``charge_context=False`` is the **oracle arm** of C-CTX: the fact is delivered at
        zero context cost, which is what a context-free fact channel (kNN-LM) would buy.
        No mechanism can actually do this by prepending — that is the point. It is an
        upper bound, and the gap between it and the prepend arm *is* the displacement.

        ``placement`` selects the shared renderer's format (`lithos.posttrain.reference`).
        ``INLINE`` is the control for the third cause: a model that can use a fact as prose
        but not inside a ``Reference material:`` block is *untrained*, not incapable.
        """
        passages: tuple[Passage, ...] = ()
        context_tokens = 0
        if self.retriever is not None and context_token_budget > 0:
            ctx = self.retriever.retrieve(query, token_budget=context_token_budget)
            passages, context_tokens, _ = self._fit_to_budget(
                query, ctx, system, context_token_budget, placement
            )

        messages = self._build_messages(query, passages, system, placement)
        prompt_ids = render_prompt(messages, self.tok)

        if total_token_budget is not None:
            spent = len(prompt_ids) - (0 if charge_context else context_tokens)
            # A prompt that fills L leaves nothing to think with. That is not an error —
            # it is the displacement result, and it must be observable rather than raised.
            max_new = max(0, total_token_budget - spent)

        roll = tir_rollout(
            self.model,
            prompt_ids,
            self.tok,
            self._tir_ids,
            self._sids,
            device=self.device,
            max_new=max_new,
            max_tool_calls=max_tool_calls,
            temperature=temperature,
            top_p=top_p,
            generator=generator,
            use_cache=use_cache,
            logits_processor=self.policy,
        )

        # Per-passage cost, measured the same exact way as the total.
        cites: list[Citation] = []
        if passages:
            bare = len(render_prompt(self._build_messages(query, (), system, placement), self.tok))
            running = bare
            for i in range(1, len(passages) + 1):
                upto = len(
                    render_prompt(
                        self._build_messages(query, passages[:i], system, placement), self.tok
                    )
                )
                p = passages[i - 1]
                cites.append(
                    Citation(
                        source_id=p.source_id,
                        record_id=p.record_id,
                        text_sha256=p.text_sha256,
                        tier=p.tier,
                        tokens=upto - running,
                        chunk_sha256=p.chunk_sha256,
                    )
                )
                running = upto

        return CompositeResult(
            model_id=self.id,
            text=roll.completion_text,
            token_ids=roll.token_ids,
            action_mask=roll.action_mask,
            citations=cites,
            tool_calls=[
                ToolCallRecord(runtime=r, code=c, output=o)
                for (r, c), o in zip(roll.tool_calls, roll.tool_outputs, strict=True)
            ],
            prompt_tokens=len(prompt_ids),
            context_tokens=context_tokens,
            completion_budget=max_new,
            context_placement=placement.value if passages else "",
            truncated=roll.truncated,
        )
