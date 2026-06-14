"""Integration tests for the training loop (PRD §9.10, §17.2).

Overfit a tiny model on tiny data, verify metrics logging, and verify that a
checkpoint resume reproduces the exact training trajectory.
"""

import json

from lithos.model import ModelConfig
from lithos.train import find_latest_checkpoint, train
from lithos.train.config import DataConfig, OptimConfig, ScheduleConfig, TrainConfig

from tests.helpers import make_tiny_corpus


def _losses(run):
    out = []
    for line in run.metrics.read_text().splitlines():
        rec = json.loads(line)
        if "train_loss" in rec:
            out.append((rec["step"], rec["train_loss"]))
    return out


def _cfg(tmp_path, manifest, name, max_steps, **over):
    base = dict(
        run_name=name,
        runs_dir=str(tmp_path / "runs"),
        device="cpu",
        precision="fp32",
        micro_batch_size=4,
        log_interval=1,
        model=ModelConfig(vocab_size=32, n_layers=2, hidden=64, n_heads=4, seq_len=32),
        data=DataConfig(corpus_manifest=manifest, seq_len=32),
        optim=OptimConfig(lr=1e-3),
        schedule=ScheduleConfig(warmup_steps=10, max_steps=max_steps, min_lr_ratio=0.1),
    )
    base.update(over)
    return TrainConfig(**base)


def test_tiny_model_overfits_tiny_data(tmp_path):
    manifest = make_tiny_corpus(tmp_path / "corpus")
    run = train(_cfg(tmp_path, manifest, "overfit", max_steps=500))
    losses = [loss for _, loss in _losses(run)]
    assert losses[0] > losses[-1]
    assert losses[-1] < 0.5  # memorized the deterministic pattern


def test_metrics_jsonl_and_run_artifacts(tmp_path):
    manifest = make_tiny_corpus(tmp_path / "corpus")
    run = train(_cfg(tmp_path, manifest, "metrics", max_steps=5))
    first = json.loads(run.metrics.read_text().splitlines()[0])
    for key in (
        "step",
        "tokens_seen",
        "train_loss",
        "learning_rate",
        "grad_norm",
        "throughput_tokens_per_sec",
        "timestamp",
    ):
        assert key in first
    assert run.resolved_config.exists()
    assert run.manifest.exists()
    man = json.loads(run.manifest.read_text())
    assert man["global_batch_size"] == 4
    assert man["tokens_per_step"] == 4 * 32


def test_checkpoint_resume_reproduces_trajectory(tmp_path):
    manifest = make_tiny_corpus(tmp_path / "corpus")
    full = train(_cfg(tmp_path, manifest, "full", max_steps=12, checkpoint_interval=6))
    full_losses = dict(_losses(full))

    part = train(_cfg(tmp_path, manifest, "part", max_steps=6, checkpoint_interval=6))
    ckpt = find_latest_checkpoint(part.root)
    assert ckpt is not None
    assert json.loads((ckpt / "meta.json").read_text())["step"] == 6

    resumed = train(
        _cfg(tmp_path, manifest, "resume", max_steps=12, checkpoint_interval=6),
        resume_from=str(ckpt),
    )
    resumed_losses = _losses(resumed)
    assert [s for s, _ in resumed_losses] == list(range(7, 13))
    for step, loss in resumed_losses:
        assert abs(loss - full_losses[step]) < 1e-4
