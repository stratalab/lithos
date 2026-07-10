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
- **The decode policy (Verity) is the last write to the logits.** Enforced in
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
from typing import Any, Protocol, runtime_checkable

import torch

from lithos.data.tiers import DATASTORE_ALLOWED_TIERS
from lithos.model.generation import LogitsProcessor
from lithos.posttrain.chat_template import render_prompt, special_ids, tir_token_ids
from lithos.posttrain.sandbox import tool_env_sha
from lithos.posttrain.tir_rollout import tir_rollout

# ── identity ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ServedModelId:
    """A served model is no longer a weights file (`docs/composite-model-layer.md` §7.1).

    The four components *are* the parts list: weights = Lithos, datastore = StrataDB,
    decode policy = Verity, tool env = the sandbox. Evals, bisects, and incident reports
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


# ── retrieval (above the token stream) ────────────────────────────────────────


@dataclass(frozen=True)
class Passage:
    """One retrieved chunk. The three provenance keys are the join to Chisel and Petra."""

    text: str
    source_id: str
    record_id: str
    text_sha256: str
    tier: str
    score: float = 0.0


@dataclass(frozen=True)
class RetrievedContext:
    passages: tuple[Passage, ...] = ()
    #: Tokens the passages will consume once rendered into the prompt. Measured by the
    #: composite (BPE can merge across the seam), not guessed by the retriever.
    tokens_used: int = 0


@runtime_checkable
class Retriever(Protocol):
    """Anything that turns a query into passages, under a **token budget**."""

    version: str

    def retrieve(self, query: str, *, token_budget: int) -> RetrievedContext: ...


class StubRetriever:
    """Fixed passages, truncated to the budget. Enough to exercise every seam.

    Enforces the datastore half of the tier gate: `restricted` passages are welcome —
    the model *cites* what it consults, which is the whole point of moving books out of
    the weights (`docs/chisel-tier-gate.md`). `unknown` is not.
    """

    version = "stub-v0"

    def __init__(self, passages: Sequence[Passage]) -> None:
        for p in passages:
            if p.tier not in DATASTORE_ALLOWED_TIERS:
                raise ValueError(
                    f"passage {p.source_id!r} has tier={p.tier!r}; the datastore accepts "
                    f"{sorted(DATASTORE_ALLOWED_TIERS)} (restricted is allowed here — it is "
                    f"cited, never trained on)"
                )
        self._passages = tuple(passages)

    def retrieve(self, query: str, *, token_budget: int) -> RetrievedContext:
        if token_budget <= 0:
            return RetrievedContext()
        # A real retriever ranks by similarity to `query`; the stub preserves order and
        # lets the composite do the budget accounting, since only it owns the tokenizer.
        return RetrievedContext(passages=self._passages)


# ── decode policy (Verity) ────────────────────────────────────────────────────


class DenyTokensPolicy:
    """A minimal Verity primitive: these token ids may never be emitted.

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
    text_sha256: str
    tier: str
    tokens: int  # what this passage cost the context budget


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
            "reasoning_tokens": self.reasoning_tokens,
        }


# ── the composite ─────────────────────────────────────────────────────────────

_CONTEXT_HEADER = "Reference material:"


def _render_passage(p: Passage, n: int) -> str:
    return f"[{n}] {p.text}"


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
        self, query: str, passages: Sequence[Passage], system: str | None
    ) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        if passages:
            block = "\n".join(_render_passage(p, i + 1) for i, p in enumerate(passages))
            content = f"{_CONTEXT_HEADER}\n{block}\n\n{query}"
        else:
            content = query
        msgs.append({"role": "user", "content": content})
        return msgs

    def _fit_to_budget(self, query: str, ctx: RetrievedContext, system: str | None, budget: int):
        """Drop lowest-ranked passages until the context fits ``budget`` tokens.

        The cost is measured as ``len(prompt_with) - len(prompt_without)``: exact, and
        immune to BPE merging across the passage/query seam.
        """
        bare = len(render_prompt(self._build_messages(query, (), system), self.tok))
        kept = list(ctx.passages)
        while kept:
            full = len(render_prompt(self._build_messages(query, kept, system), self.tok))
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
        max_tool_calls: int = 2,
        temperature: float = 1.0,
        top_p: float | None = 0.95,
        generator: torch.Generator | None = None,
        use_cache: bool = True,
    ) -> CompositeResult:
        passages: tuple[Passage, ...] = ()
        context_tokens = 0
        if self.retriever is not None and context_token_budget > 0:
            ctx = self.retriever.retrieve(query, token_budget=context_token_budget)
            passages, context_tokens, _ = self._fit_to_budget(
                query, ctx, system, context_token_budget
            )

        messages = self._build_messages(query, passages, system)
        prompt_ids = render_prompt(messages, self.tok)

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
            bare = len(render_prompt(self._build_messages(query, (), system), self.tok))
            running = bare
            for i in range(1, len(passages) + 1):
                upto = len(
                    render_prompt(self._build_messages(query, passages[:i], system), self.tok)
                )
                cites.append(
                    Citation(
                        source_id=passages[i - 1].source_id,
                        record_id=passages[i - 1].record_id,
                        text_sha256=passages[i - 1].text_sha256,
                        tier=passages[i - 1].tier,
                        tokens=upto - running,
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
            truncated=roll.truncated,
        )
