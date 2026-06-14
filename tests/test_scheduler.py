"""Tests for lithos.train.scheduler — warmup, cosine decay, min lr (PRD §9.4)."""

from lithos.train.config import ScheduleConfig
from lithos.train.scheduler import cosine_lr

CFG = ScheduleConfig(warmup_steps=10, max_steps=100, min_lr_ratio=0.1)


def test_warmup_ramps_linearly():
    assert abs(cosine_lr(0, CFG, 1.0) - 0.1) < 1e-9  # (0+1)/10
    assert abs(cosine_lr(4, CFG, 1.0) - 0.5) < 1e-9
    assert abs(cosine_lr(9, CFG, 1.0) - 1.0) < 1e-9  # peak at end of warmup


def test_cosine_decays_to_min():
    peak = cosine_lr(10, CFG, 1.0)
    mid = cosine_lr(55, CFG, 1.0)
    end = cosine_lr(100, CFG, 1.0)
    assert peak > mid > end
    assert abs(peak - 1.0) < 1e-9
    assert abs(mid - 0.55) < 1e-9
    assert abs(end - 0.1) < 1e-9  # min lr = base * min_lr_ratio


def test_past_max_steps_clamps_to_min():
    assert abs(cosine_lr(250, CFG, 1.0) - 0.1) < 1e-9
