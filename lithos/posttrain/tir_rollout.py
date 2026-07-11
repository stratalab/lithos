"""Multi-segment TIR rollout for RLVR (epic E4, Phase 12).

Runs one tool-integrated-reasoning episode to completion: the policy generates
until it emits ``<|/tool|>`` (pause + execute the call in the E1 sandbox, inject
``<|tool_result|>…<|end|>``, resume) or ``<|end|>`` (final answer). The returned
``action_mask`` marks which tokens are the *policy's* actions (True) vs the prompt
or the *environment's* injected tool results (False) — the GRPO loss uses it to
exclude tool-result tokens from the policy gradient and the KL (docs/tir-format.md
§4: those tokens are the sandbox's move, not the model's).

Sequential per rollout (v1); batched/vLLM rollouts are E5.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from lithos.model.generation import generate
from lithos.posttrain.chat_template import TOOL_CLOSE, TOOL_OPEN, TOOL_RESULT
from lithos.posttrain.sandbox import run_tool


@dataclass
class RolloutResult:
    """One completed episode. ``action_mask[i]`` is True iff ``token_ids[i]`` is a
    policy-generated action (False at prompt + every injected tool-result token)."""

    token_ids: list[int]
    action_mask: list[bool]
    completion_text: str  # decoded completion (prompt stripped), for the verifier
    num_tool_calls: int
    tool_calls: list[tuple[str, str]] = field(default_factory=list)  # (runtime, code) for the gaming screen
    truncated: bool = False  # hit the token/tool-call budget without a final <|end|>
    #: Strictly parallel to ``tool_calls``: what the sandbox actually returned. Feeds
    #: the composite's out-of-band provenance channel — a tool result is a `tool`-channel
    #: fact, never a parametric one (`docs/petra-composite-attribution.md` §2).
    tool_outputs: list[str] = field(default_factory=list)
    #: Segments that closed with ``<|/tool|>`` but carried no runtime tag. The error is
    #: injected into the context, but it is not a *call*, so it pairs with nothing.
    num_malformed_calls: int = 0


def parse_tool_call(
    segment: list[int], tir_ids: dict[str, int], tok
) -> tuple[str, str] | None:
    """Parse a generated segment ending in ``<|/tool|>`` into ``(runtime, code)``.

    Finds the *last* tool-open token (``<|python|>``/``<|octave|>``) by ID and
    decodes the span between it and the closer as raw source (never string-parsing
    structure). Returns None if there is no open tag before the close (malformed).
    """
    close_id = tir_ids[TOOL_CLOSE]
    if not segment or segment[-1] != close_id:
        return None
    open_to_runtime = {tir_ids[token]: runtime for runtime, token in TOOL_OPEN.items()}
    for i in range(len(segment) - 2, -1, -1):
        if segment[i] in open_to_runtime:
            code = tok.decode(segment[i + 1 : -1], skip_special_tokens=True)
            return open_to_runtime[segment[i]], code
    return None


def tir_rollout(
    model,
    prompt_ids: list[int],
    tok,
    tir_ids: dict[str, int],
    sids: dict[str, int],
    *,
    device: str = "cpu",
    max_new: int = 256,
    max_tool_calls: int = 4,
    temperature: float = 1.0,
    top_p: float | None = 0.95,
    generator: torch.Generator | None = None,
    timeout_s: float = 5.0,
    result_token_cap: int = 256,
    use_cache: bool = True,
) -> RolloutResult:
    """Generate one TIR episode, executing tool calls and injecting their results."""
    end_id = sids["<|end|>"]
    tool_close_id = tir_ids[TOOL_CLOSE]
    result_open_id = tir_ids[TOOL_RESULT]
    stop_ids = {tool_close_id, end_id}

    token_ids: list[int] = list(prompt_ids)
    action_mask: list[bool] = [False] * len(prompt_ids)
    tool_calls: list[tuple[str, str]] = []
    tool_outputs: list[str] = []
    num_malformed = 0
    budget = max_new
    completed = False  # emitted a final <|end|> within budget/tool caps

    # up to max_tool_calls tool-call segments + 1 final-answer segment
    for _ in range(max_tool_calls + 1):
        if budget <= 0:
            break
        out = generate(
            model,
            torch.tensor([token_ids], device=device),
            budget,
            temperature=temperature,
            top_p=top_p,
            stop_token_ids=stop_ids,
            generator=generator,
            use_cache=use_cache,
        )
        new = out[0].tolist()[len(token_ids) :]
        budget -= len(new)
        token_ids.extend(new)
        action_mask.extend([True] * len(new))  # model-generated policy actions
        if not new:
            break
        last = new[-1]
        if last == end_id:
            completed = True
            break
        if last != tool_close_id:
            break  # ran out of budget mid-generation (no stop token)
        if len(tool_calls) >= max_tool_calls:
            break  # cap reached and the model tried another tool call

        parsed = parse_tool_call(new, tir_ids, tok)
        if parsed is None:
            output = "[error: tool call had no <|python|>/<|octave|> tag]"
            num_malformed += 1  # the error is still injected; it just isn't a *call*
        else:
            runtime, code = parsed
            tool_calls.append((runtime, code))
            output = run_tool(runtime, code, timeout_s=timeout_s).output
            tool_outputs.append(output)  # keep strictly parallel to tool_calls
        # inject the result as the environment's move — NOT a policy action
        result_ids = [result_open_id, *tok.encode(output).ids[:result_token_cap], end_id]
        token_ids.extend(result_ids)
        action_mask.extend([False] * len(result_ids))

    completion_text = tok.decode(token_ids[len(prompt_ids) :], skip_special_tokens=True)
    truncated = not completed
    return RolloutResult(
        token_ids=token_ids,
        action_mask=action_mask,
        completion_text=completion_text,
        num_tool_calls=len(tool_calls),
        tool_calls=tool_calls,
        truncated=truncated,
        tool_outputs=tool_outputs,
        num_malformed_calls=num_malformed,
    )
