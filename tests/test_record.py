"""Tests for the canonical post-training record (lithos/posttrain/record.py, T1+T2).

The record is the one data shape all four trainers consume; these tests lock its
validation (fail-closed on malformed arrays), the labels() shift convention, and
the from_rendered lift.
"""

import math

import pytest
from lithos.posttrain.record import IGNORE_INDEX, TrainingRecord


def test_validates_lengths():
    with pytest.raises(ValueError, match="tokens/weights"):
        TrainingRecord(tokens=[1, 2, 3], weights=[1.0, 0.0])
    with pytest.raises(ValueError, match="tokens/logprobs"):
        TrainingRecord(tokens=[1, 2], weights=[1.0, 0.0], logprobs=[0.0])
    with pytest.raises(ValueError, match="tokens/advantages"):
        TrainingRecord(tokens=[1, 2], weights=[1.0, 0.0], advantages=[0.0, 0.0, 0.0])


def test_validates_weight_values():
    with pytest.raises(ValueError, match=">= 0"):
        TrainingRecord(tokens=[1, 2], weights=[1.0, -0.1])
    with pytest.raises(ValueError, match="finite"):
        TrainingRecord(tokens=[1, 2], weights=[1.0, math.nan])
    # fractional weights are legal — the float generalization is the point
    TrainingRecord(tokens=[1, 2], weights=[0.5, 1.0])


def test_labels_shift_and_ignore():
    # weights[i+1] gates the label for predicting tokens[i+1]
    rec = TrainingRecord(tokens=[10, 11, 12, 13], weights=[0.0, 0.0, 1.0, 1.0])
    assert rec.labels() == [IGNORE_INDEX, 12, 13]


def test_has_targets_ignores_position_zero():
    # position 0 is never predicted, so weight there alone trains nothing
    assert not TrainingRecord(tokens=[1, 2], weights=[1.0, 0.0]).has_targets()
    assert TrainingRecord(tokens=[1, 2], weights=[0.0, 1.0]).has_targets()
    assert not TrainingRecord(tokens=[1], weights=[1.0]).has_targets()


def test_num_loss_tokens_counts_positive_weights():
    rec = TrainingRecord(tokens=[1, 2, 3, 4], weights=[0.0, 0.5, 1.0, 0.0])
    assert rec.num_loss_tokens == 2


def test_from_rendered_round_trip():
    from lithos.posttrain.chat_template import Rendered

    r = Rendered(input_ids=[7, 8, 9], weights=[0.0, 1.0, 1.0])
    rec = TrainingRecord.from_rendered(r)
    assert rec.tokens == [7, 8, 9]
    assert rec.weights == [0.0, 1.0, 1.0]
    assert r.loss_mask == [False, True, True]  # the derived boolean view
