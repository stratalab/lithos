"""Autoregressive generation (PRD §6.1.10, §13.1).

Decoding methods: greedy, temperature, top-k, top-p. Uses the KV cache by
default; ``use_cache=False`` recomputes the full sequence each step and is the
reference path the cache is tested against.
"""

from __future__ import annotations

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
) -> torch.Tensor:
    """logits: (B, vocab) -> next token ids (B, 1)."""
    if greedy or temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

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
    return torch.multinomial(probs, num_samples=1, generator=generator)


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
    use_cache: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate up to ``max_new_tokens`` tokens; returns prompt + completion."""
    model.eval()
    generated = input_ids
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    kv_caches = model.init_kv_caches() if use_cache else None
    next_logits: torch.Tensor | None = None
    if use_cache:
        logits, _ = model(generated, kv_caches=kv_caches)
        next_logits = logits[:, -1, :]

    for _ in range(max_new_tokens):
        if not use_cache:
            logits, _ = model(generated)
            next_logits = logits[:, -1, :]
        assert next_logits is not None

        next_token = _sample_next(
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

        generated = torch.cat((generated, next_token), dim=1)

        if eos_token_id is not None:
            finished = finished | (next_token.squeeze(1) == eos_token_id)
            if bool(finished.all()):
                break

        if use_cache:
            logits, _ = model(next_token, kv_caches=kv_caches)
            next_logits = logits[:, -1, :]

    return generated
