"""Integration tests for SFT wiring: config validation + weight-only init (Phase 11),
plus the packed-SFT loop path (E2)."""

import json

import pytest
import torch
from lithos.model import LithosForCausalLM
from lithos.model.config import ModelConfig
from lithos.posttrain.sft_corpus import SFTShardWriter
from lithos.train import train
from lithos.train.checkpoint import (
    load_model_from_checkpoint,
    load_model_weights,
    save_checkpoint,
)
from lithos.train.config import DataConfig, OptimConfig, ScheduleConfig, TrainConfig
from safetensors.torch import save_model


def _tiny_model_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        n_layers=2,
        hidden=32,
        n_heads=2,
        n_kv_heads=2,
        intermediate_size=64,
        seq_len=16,
        qk_norm=True,
        tie_embeddings=True,
    )


def test_sft_data_requires_tokenizer_path():
    with pytest.raises(ValueError, match="tokenizer_path"):
        DataConfig(kind="sft", corpus_manifest="data/sft/train.jsonl", seq_len=64)


def test_packed_data_defaults_need_no_tokenizer():
    cfg = DataConfig(corpus_manifest="corpus.json", seq_len=64)
    assert cfg.kind == "packed" and cfg.tokenizer_path is None


def test_sft_packed_needs_no_tokenizer():
    # sft_packed reads pre-rendered shards, so (unlike kind="sft") it must NOT
    # require a tokenizer at train time — the tokenizer is recorded in the manifest.
    cfg = DataConfig(kind="sft_packed", corpus_manifest="sft_manifest.json", seq_len=64)
    assert cfg.kind == "sft_packed" and cfg.tokenizer_path is None


def _write_packed_sft_shards(shard_dir, tokens, mask):
    shard_dir.mkdir(parents=True, exist_ok=True)
    w = SFTShardWriter(shard_dir, tokens_per_shard=1_000_000, dtype="uint16",
                       tokenizer_name="t", rel_base=shard_dir)
    w.add(list(tokens), [bool(m) for m in mask])
    shards = w.close()
    manifest = shard_dir / "sft_manifest.json"
    manifest.write_text(json.dumps({"kind": "sft_packed", "shards": shards}))
    return str(manifest)


def test_train_loop_drives_packed_sft(tmp_path):
    # A tiny model overfits a deterministic packed-SFT stream, proving the
    # kind="sft_packed" loader drives the unchanged train() loop + ignore_index path.
    seq_len = 16
    tokens = [1, 2, 3, 4, 5, 6, 7, 8] * 200  # learnable repeating pattern, vocab < 32
    manifest = _write_packed_sft_shards(tmp_path / "sft", tokens, [1] * len(tokens))

    cfg = TrainConfig(
        run_name="packed-sft",
        runs_dir=str(tmp_path / "runs"),
        device="cpu",
        precision="fp32",
        micro_batch_size=4,
        log_interval=1,
        model=ModelConfig(vocab_size=32, n_layers=2, hidden=64, n_heads=4, seq_len=32),
        data=DataConfig(kind="sft_packed", corpus_manifest=manifest, seq_len=seq_len),
        optim=OptimConfig(lr=1e-3),
        schedule=ScheduleConfig(warmup_steps=10, max_steps=120, min_lr_ratio=0.1),
    )
    run = train(cfg)
    losses = [
        json.loads(line)["train_loss"]
        for line in run.metrics.read_text().splitlines()
        if "train_loss" in json.loads(line)
    ]
    assert losses[0] > losses[-1]  # it learned
    assert losses[-1] < losses[0] * 0.7


def test_load_model_weights_roundtrip(tmp_path):
    torch.manual_seed(0)
    src = LithosForCausalLM(_tiny_model_cfg())
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    save_model(src, str(ckpt / "model.safetensors"))

    torch.manual_seed(1)  # a differently-initialised destination
    dst = LithosForCausalLM(_tiny_model_cfg())
    assert not all(torch.equal(a, b) for a, b in zip(src.parameters(), dst.parameters(), strict=False))

    load_model_weights(ckpt, dst)  # loads weights only — no optimizer/RNG/data state
    assert all(torch.equal(a, b) for a, b in zip(src.parameters(), dst.parameters(), strict=False))


def test_checkpoint_embeds_arch_and_loads_size_agnostic(tmp_path):
    import json

    src = LithosForCausalLM(_tiny_model_cfg())
    opt = torch.optim.AdamW(src.parameters(), lr=1e-3)
    ckpt = tmp_path / "step_000001"
    save_checkpoint(ckpt, model=src, optimizer=opt, step=1, tokens_seen=0, dataloader_state={}, meta={})

    # the architecture is embedded in meta.json (self-describing checkpoint)
    meta = json.loads((ckpt / "meta.json").read_text())
    assert meta["model"]["n_layers"] == 2 and meta["model"]["hidden"] == 32

    # the loader rebuilds the right model + weights WITHOUT being told the shape
    loaded = load_model_from_checkpoint(ckpt)
    assert all(torch.equal(a, b) for a, b in zip(src.parameters(), loaded.parameters(), strict=False))
