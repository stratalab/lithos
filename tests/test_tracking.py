"""Tests for optional W&B tracking (PRD §15).

A fake ``wandb`` module exercises the real init/log/finish forwarding without
the dependency; the default (disabled) and off-main-rank paths are verified to
be true no-ops that never import or call wandb.
"""

import sys
import types

import pytest
from lithos.model import ModelConfig
from lithos.train.config import DataConfig, TrainConfig, WandbConfig
from lithos.train.tracking import Reporter, init_reporter


def _cfg(**wandb_over):
    return TrainConfig(
        run_name="t",
        model=ModelConfig(vocab_size=32, n_layers=1, hidden=16, n_heads=2, seq_len=8),
        data=DataConfig(corpus_manifest="x.json", seq_len=8),
        wandb=WandbConfig(**wandb_over),
    )


class _FakeRun:
    def __init__(self) -> None:
        self.logged: list[tuple[int, dict]] = []
        self.finished = False

    def log(self, data, step):
        self.logged.append((step, data))

    def finish(self):
        self.finished = True


class _FakeWandb(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("wandb")
        self.init_kwargs: dict | None = None
        self.run = _FakeRun()

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run


@pytest.fixture
def fake_wandb(monkeypatch):
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    # Keep the test hermetic: don't read the developer's real .env.
    monkeypatch.setattr("lithos.train.tracking.load_env", lambda: None)
    return fake


def test_disabled_is_noop(tmp_path):
    r = init_reporter(_cfg(enabled=False), run_id="r", run_dir=str(tmp_path), is_main=True)
    assert not r.enabled
    r.log({"step": 1, "train_loss": 2.0}, 1)  # must not raise
    r.finish()


def test_not_main_rank_never_inits(tmp_path, fake_wandb):
    r = init_reporter(_cfg(enabled=True), run_id="r", run_dir=str(tmp_path), is_main=False)
    assert not r.enabled
    assert fake_wandb.init_kwargs is None  # wandb.init never called off the main rank


def test_enabled_forwards_to_wandb(tmp_path, fake_wandb):
    cfg = _cfg(enabled=True, project="lithos-test", tags=["a"])
    r = init_reporter(cfg, run_id="2026_run", run_dir=str(tmp_path), is_main=True)
    assert r.enabled
    assert fake_wandb.init_kwargs["project"] == "lithos-test"
    assert fake_wandb.init_kwargs["name"] == "2026_run"
    assert fake_wandb.init_kwargs["group"] == "t"  # group defaults to run_name
    assert fake_wandb.init_kwargs["config"]["optim"]["lr"] == cfg.optim.lr  # full config logged

    # step/timestamp are bookkeeping (stripped); real metrics forwarded at the step.
    r.log({"step": 5, "timestamp": "z", "train_loss": 1.5, "grad_norm": 0.3}, 5)
    assert fake_wandb.run.logged == [(5, {"train_loss": 1.5, "grad_norm": 0.3})]

    # an all-bookkeeping record produces no wandb.log call at all
    r.log({"step": 6, "timestamp": "z"}, 6)
    assert len(fake_wandb.run.logged) == 1

    r.finish()
    assert fake_wandb.run.finished
    assert not r.enabled  # finish clears the run (idempotent)


def test_missing_wandb_dependency_raises_helpful_error(tmp_path, monkeypatch):
    monkeypatch.setattr("lithos.train.tracking.load_env", lambda: None)
    monkeypatch.setitem(sys.modules, "wandb", None)  # force ImportError on `import wandb`
    with pytest.raises(ImportError, match="wandb is not installed"):
        init_reporter(_cfg(enabled=True), run_id="r", run_dir=str(tmp_path), is_main=True)


def test_reporter_none_is_noop():
    r = Reporter(None)
    r.log({"train_loss": 1.0}, 0)
    r.finish()  # must not raise
