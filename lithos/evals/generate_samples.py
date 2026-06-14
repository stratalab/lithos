"""Fixed-prompt sample generation and repetition checks (PRD §11.1)."""

from __future__ import annotations

from typing import Any

import torch
from tokenizers import Tokenizer

from lithos.model import LithosForCausalLM, generate


def repetition_score(token_ids: list[int], n: int = 3) -> float:
    """Fraction of repeated n-grams (0 = all distinct, ->1 = highly repetitive)."""
    if len(token_ids) < n:
        return 0.0
    ngrams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def generate_samples(
    model: LithosForCausalLM,
    tokenizer: Tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float | None = 0.95,
    greedy: bool = False,
    device: str = "cpu",
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Generate a completion per prompt with a repetition score for each."""
    # The RNG generator must live on the same device as the sampled probabilities.
    generator = torch.Generator(device=device).manual_seed(seed)
    results: list[dict[str, Any]] = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt).ids
        input_ids = torch.tensor([ids], device=device)
        out = generate(
            model,
            input_ids,
            max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            greedy=greedy,
            generator=generator,
        )
        new_ids = out[0].tolist()[len(ids) :]
        results.append(
            {
                "prompt": prompt,
                "completion": tokenizer.decode(new_ids),
                "n_new_tokens": len(new_ids),
                "repetition": repetition_score(new_ids),
            }
        )
    return results
