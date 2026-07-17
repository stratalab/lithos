"""The canonical post-training record (docs/tinker-learnings.md, T1+T2).

One shape serves every trainer: SFT consumes ``(tokens, weights)``; GRPO adds
``(logprobs, advantages)``; DPO is two records (chosen/rejected) sharing a masked
prompt. The *method* lives in how records are constructed and which loss consumes
them — not in per-trainer data shapes. (The design follows Tinker's ``Datum``:
"everything the loss needs is in the record".)

``weights`` is per-token and **float**: ``0.0`` = no gradient, ``1.0`` = full loss;
the boolean loss mask is the degenerate {0, 1} case. This array is also the
**gradient gate**: the tier doctrine ("only tokens that receive a gradient are
gated", ``lithos/data/tiers.py``) operates on exactly this vector — ``weight > 0``
⇔ gradient-bearing ⇔ tier-gated — so training and the attestation manifest share
one source of truth instead of parallel bookkeeping.

``logprobs`` is the **sampler's** per-token log-probability of each generated
token under the distribution it was actually sampled from (after decode policy,
temperature, top-k/top-p), recorded at generation time. The on-policy GRPO loss
does not read it yet — it is carried so that when rollout generation moves off the
trainer's own forward pass (batched/vLLM rollouts, E5) the sampler≠trainer
off-policy correction (importance sampling, p/q) is a loss-function swap, not a
data-format migration. ``0.0`` at non-action positions (prompt, injected tool
results, forced padding).

Alignment convention: all arrays are parallel to ``tokens``. Next-token training
shifts by one — ``labels()[i]`` is the target for predicting ``tokens[i + 1]``, so
consumers slicing auxiliary arrays against labels use ``values[1:]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

IGNORE_INDEX = -100  # matches F.cross_entropy(ignore_index=...) in the model


@dataclass
class TrainingRecord:
    """One training example: tokens + per-token loss weights (+ RL extras)."""

    tokens: list[int]
    weights: list[float]
    logprobs: list[float] | None = None  # sampler's, aligned to tokens; see module doc
    advantages: list[float] | None = None  # per-token; group-relative scalar broadcast in GRPO

    def __post_init__(self) -> None:
        n = len(self.tokens)
        if len(self.weights) != n:
            raise ValueError(f"tokens/weights length mismatch: {n} != {len(self.weights)}")
        bad = next((w for w in self.weights if w < 0 or not math.isfinite(w)), None)
        if bad is not None:
            raise ValueError(f"weights must be finite and >= 0, got {bad}")
        for name, values in (("logprobs", self.logprobs), ("advantages", self.advantages)):
            if values is not None and len(values) != n:
                raise ValueError(f"tokens/{name} length mismatch: {n} != {len(values)}")

    @classmethod
    def from_rendered(cls, rendered) -> TrainingRecord:
        """Lift a ``chat_template.Rendered`` conversation into a record."""
        return cls(tokens=list(rendered.input_ids), weights=list(rendered.weights))

    @property
    def num_loss_tokens(self) -> int:
        return sum(1 for w in self.weights if w > 0)

    def has_targets(self) -> bool:
        """True iff training on this record produces at least one gradient-bearing
        label — i.e. some *shifted* position carries weight (``weights[0]`` guards
        nothing: position 0 is never predicted)."""
        return len(self.tokens) >= 2 and any(w > 0 for w in self.weights[1:])

    def labels(self) -> list[int]:
        """Next-token labels for ``tokens[:-1]``: ``tokens[i + 1]`` where the shifted
        weight is > 0, else ``IGNORE_INDEX`` — so prompt, injected tool-result, and
        any other zero-weight span drops out of the loss (and the KL, via
        ``token_logprobs`` returning 0 at ignored positions)."""
        return [
            self.tokens[i + 1] if self.weights[i + 1] > 0 else IGNORE_INDEX
            for i in range(len(self.tokens) - 1)
        ]
