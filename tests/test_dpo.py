"""Tests for the DPO loss + sequence log-probs (Phase 11)."""

import math

import torch
from lithos.posttrain.dpo import dpo_loss, sequence_logprobs
from lithos.posttrain.sft_dataset import IGNORE_INDEX


def test_sequence_logprobs_sums_only_unmasked():
    # 1 sequence, 3 positions, vocab 4. Make logits a one-hot-ish so log-softmax is known.
    logits = torch.zeros(1, 3, 4)
    logits[0, 0, 1] = 100.0  # position 0 -> ~log p(1) ~= 0
    logits[0, 1, 2] = 100.0  # position 1 -> ~log p(2) ~= 0
    logits[0, 2, 3] = 100.0
    labels = torch.tensor([[IGNORE_INDEX, 2, 3]])  # position 0 masked
    lp = sequence_logprobs(logits, labels)
    assert lp.shape == (1,)
    # positions 1 and 2 each ~ log(1) = 0; masked position contributes nothing
    assert abs(lp.item()) < 1e-3


def test_sequence_logprobs_value():
    # uniform logits over vocab=4 -> each token logprob = log(1/4)
    logits = torch.zeros(1, 2, 4)
    labels = torch.tensor([[1, 3]])  # both count
    lp = sequence_logprobs(logits, labels)
    assert math.isclose(lp.item(), 2 * math.log(0.25), rel_tol=1e-5)


def test_dpo_loss_zero_margin_is_log2():
    # policy == reference -> logits 0 -> loss = -log sigmoid(0) = log 2
    z = torch.zeros(4)
    loss, m = dpo_loss(z, z, z, z, beta=0.1)
    assert math.isclose(loss.item(), math.log(2), rel_tol=1e-5)
    assert m["reward_margin"] == 0.0


def test_dpo_loss_lower_when_policy_prefers_chosen():
    ref_c = torch.tensor([0.0])
    ref_r = torch.tensor([0.0])
    # policy raises chosen, lowers rejected relative to ref
    good_loss, gm = dpo_loss(torch.tensor([1.0]), torch.tensor([-1.0]), ref_c, ref_r, beta=0.1)
    # policy does the opposite
    bad_loss, bm = dpo_loss(torch.tensor([-1.0]), torch.tensor([1.0]), ref_c, ref_r, beta=0.1)
    assert good_loss.item() < math.log(2) < bad_loss.item()
    assert gm["reward_accuracy"] == 1.0 and bm["reward_accuracy"] == 0.0
    assert gm["reward_margin"] > 0 > bm["reward_margin"]


def test_dpo_loss_is_differentiable():
    pc = torch.tensor([0.5], requires_grad=True)
    pr = torch.tensor([0.2], requires_grad=True)
    loss, _ = dpo_loss(pc, pr, torch.zeros(1), torch.zeros(1), beta=0.1)
    loss.backward()
    assert pc.grad is not None and pr.grad is not None
    # increasing chosen logprob should decrease loss -> negative grad on pc
    assert pc.grad.item() < 0
