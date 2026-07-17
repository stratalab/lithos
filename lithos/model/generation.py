"""Autoregressive generation (PRD §6.1.10, §13.1).

Decoding methods: greedy, temperature, top-k, top-p. Uses the KV cache by
default; ``use_cache=False`` recomputes the full sequence each step and is the
reference path the cache is tested against.
"""

from __future__ import annotations

from typing import Literal, overload

import torch

from lithos.model.transformer import LithosForCausalLM


def _sample_next(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    greedy: bool,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """logits: (B, vocab) -> (next token ids (B, 1), sampler logprobs (B, 1)).

    The logprob is of the sampled token under the distribution it was *actually
    drawn from* — after temperature and top-k/top-p truncation + renormalization.
    That is the ``q`` an importance-sampling correction needs (``record.py``); the
    trainer's own forward pass gives ``p``. Greedy decoding is a delta
    distribution, so its logprob is 0.
    """
    if greedy or temperature == 0:
        token = logits.argmax(dim=-1, keepdim=True)
        return token, torch.zeros_like(token, dtype=logits.dtype)

    logits = logits / temperature

    if top_k is not None:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[:, -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_remove = cumulative > top_p
        # Keep at least the top token: shift the mask right by one.
        sorted_remove[:, 1:] = sorted_remove[:, :-1].clone()
        sorted_remove[:, 0] = False
        remove = torch.zeros_like(sorted_remove).scatter(1, sorted_idx, sorted_remove)
        logits = logits.masked_fill(remove, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    token = torch.multinomial(probs, num_samples=1, generator=generator)
    return token, torch.log(probs.gather(-1, token))


@overload
def generate(
    model: LithosForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = ...,
    top_k: int | None = ...,
    top_p: float | None = ...,
    greedy: bool = ...,
    eos_token_id: int | None = ...,
    stop_token_ids: set[int] | None = ...,
    use_cache: bool = ...,
    generator: torch.Generator | None = ...,
    return_logprobs: Literal[False] = ...,
) -> torch.Tensor: ...


@overload
def generate(
    model: LithosForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = ...,
    top_k: int | None = ...,
    top_p: float | None = ...,
    greedy: bool = ...,
    eos_token_id: int | None = ...,
    stop_token_ids: set[int] | None = ...,
    use_cache: bool = ...,
    generator: torch.Generator | None = ...,
    return_logprobs: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor]: ...


@torch.no_grad()
def generate(
    model: LithosForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    greedy: bool = False,
    eos_token_id: int | None = None,
    stop_token_ids: set[int] | None = None,
    use_cache: bool = True,
    generator: torch.Generator | None = None,
    return_logprobs: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Generate up to ``max_new_tokens`` tokens; returns prompt + completion.

    With ``return_logprobs=True`` returns ``(generated, logprobs)`` where
    ``logprobs`` is (B, num_generated): the **sampler's** log-probability of each
    generated token under the distribution it was drawn from (see ``_sample_next``)
    — recorded so RLVR rollout records can carry ``q`` for a later off-policy
    correction (``record.py``, docs/tinker-learnings.md T2). Positions where a
    finished row was padded with ``eos_token_id`` get 0.0 (forced, not sampled).

    A sequence finishes when it emits ``eos_token_id`` **or** any id in
    ``stop_token_ids`` — the latter lets a TIR rollout stop at ``<|/tool|>`` (to
    execute + resume) or ``<|end|>`` (done) and inspect which fired.

    NOTE: generation runs until **all** batch rows have finished. A row that
    finishes early is padded with ``eos_token_id`` if set, but with only
    ``stop_token_ids`` (no eos) an early-finished row keeps *sampling* until the
    batch ends — so a batched caller must trim each row at its first stop token
    (single-row callers like ``tir_rollout`` break immediately and are unaffected).

    Restores the model's prior train/eval mode on exit, so calling ``generate``
    mid-training (e.g. GRPO rollouts) does not silently leave the policy in eval
    mode for the subsequent loss forward — a correctness bug once dropout is on.
    """
    was_training = model.training
    model.eval()
    try:
        generated = input_ids
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        stop_ids: set[int] = set(stop_token_ids or ())
        if eos_token_id is not None:
            stop_ids.add(eos_token_id)

        kv_caches = model.init_kv_caches() if use_cache else None
        next_logits: torch.Tensor | None = None
        if use_cache:
            logits, _ = model(generated, kv_caches=kv_caches)
            next_logits = logits[:, -1, :]

        logprob_cols: list[torch.Tensor] = []
        for _ in range(max_new_tokens):
            if not use_cache:
                logits, _ = model(generated)
                next_logits = logits[:, -1, :]
            assert next_logits is not None

            next_token, next_logprob = _sample_next(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                greedy=greedy,
                generator=generator,
            )
            if eos_token_id is not None:
                keep_eos = torch.full_like(next_token, eos_token_id)
                next_token = torch.where(finished.unsqueeze(1), keep_eos, next_token)
                # a forced pad-eos was not sampled — its logprob is not the sampler's
                next_logprob = torch.where(
                    finished.unsqueeze(1), torch.zeros_like(next_logprob), next_logprob
                )

            generated = torch.cat((generated, next_token), dim=1)
            if return_logprobs:
                logprob_cols.append(next_logprob)

            if stop_ids:
                tok_col = next_token.squeeze(1)
                is_stop = torch.zeros_like(finished)
                for sid in stop_ids:
                    is_stop |= tok_col == sid
                finished = finished | is_stop
                if bool(finished.all()):
                    break

            if use_cache:
                logits, _ = model(next_token, kv_caches=kv_caches)
                next_logits = logits[:, -1, :]

        if return_logprobs:
            logprobs = (
                torch.cat(logprob_cols, dim=1)
                if logprob_cols
                else input_ids.new_zeros((input_ids.shape[0], 0), dtype=torch.float32)
            )
            return generated, logprobs
        return generated
    finally:
        if was_training:
            model.train()
