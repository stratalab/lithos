"""Direct Preference Optimization loss (Phase 11).

DPO fine-tunes a policy toward preferred responses without RL. Given a frozen
*reference* (the SFT model), it raises the policy's log-prob of the ``chosen``
response and lowers it for ``rejected``, measured *relative to the reference* so
the policy can't drift arbitrarily far. Two pure functions here — per-sequence
log-probs and the loss — keep the math unit-testable; the trainer wires them to
the model + a frozen reference.

    L = -E log σ( β · [ (logπ(y_w) - logπ_ref(y_w)) - (logπ(y_l) - logπ_ref(y_l)) ] )

where logπ(y) sums log-probs over the *response* tokens only (the prompt is masked).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lithos.posttrain.sft_dataset import IGNORE_INDEX


def sequence_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum log p(token) over the non-masked (response) positions of each sequence.

    ``logits``: (B, T, V), aligned with ``labels`` (B, T); ``labels`` use
    ``IGNORE_INDEX`` on prompt/padding positions (same convention as SFT).
    Returns (B,) total response log-prob per sequence.
    """
    b, t, v = logits.shape
    # Fused cross-entropy: never materialises the full (B,T,V) log-softmax (memory),
    # and under autocast computes in fp32 (stability). reduction="none" yields 0 at
    # ignore_index positions, so -CE summed over T is the response log-prob.
    nll = F.cross_entropy(
        logits.reshape(-1, v), labels.reshape(-1), ignore_index=IGNORE_INDEX, reduction="none"
    )
    return (-nll).reshape(b, t).sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    *,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """The DPO loss over a batch of (chosen, rejected) sequence log-probs.

    Returns ``(loss, metrics)`` where metrics include the implicit-reward accuracy
    (fraction of pairs the policy already ranks correctly) and the reward margin.
    """
    pi_logratio = policy_chosen_logps - policy_rejected_logps
    ref_logratio = ref_chosen_logps - ref_rejected_logps
    loss = -F.logsigmoid(beta * (pi_logratio - ref_logratio)).mean()

    chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    metrics = {
        "reward_accuracy": (chosen_reward > rejected_reward).float().mean().item(),
        "reward_margin": (chosen_reward - rejected_reward).mean().item(),
        "chosen_reward": chosen_reward.mean().item(),
        "rejected_reward": rejected_reward.mean().item(),
    }
    return loss, metrics
