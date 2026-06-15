"""DPO trainer (Phase 11).

DPO isn't next-token CE, so it gets a custom step: load the SFT model as a
trainable **policy** and a frozen **reference**; per batch, compute chosen/rejected
sequence log-probs under both and apply ``dpo_loss``. It reuses the scaffolding
(optimizer, cosine schedule, checkpoint, run dir, logging). Single-process for now
— multi-GPU DPO is a later add (the win signal is reward-accuracy, watched on val).
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
import torch
from tokenizers import Tokenizer

from lithos.model import LithosForCausalLM
from lithos.posttrain.dpo import dpo_loss, sequence_logprobs
from lithos.posttrain.preference_dataset import PreferenceDataset
from lithos.train.checkpoint import load_model_weights, save_checkpoint
from lithos.train.config import TrainConfig
from lithos.train.distributed import cleanup_distributed, setup_distributed
from lithos.train.logging import JsonlWriter, RunDir, create_run_dir
from lithos.train.optim import build_optimizer
from lithos.train.scheduler import cosine_lr, set_lr
from lithos.train.tracking import Reporter, init_reporter
from lithos.utils.config import save_resolved_config
from lithos.utils.seed import seed_everything


def _autocast(device: str, precision: str) -> Any:
    if device.startswith("cuda") and precision in ("bf16", "fp16"):
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


def _stack_batch(dataset: PreferenceDataset, idx: Any, device: str):
    cols = [torch.stack([dataset[int(j)][k] for j in idx]) for k in range(4)]
    cx, cy, rx, ry = (c.to(device) for c in cols)
    return torch.cat([cx, rx]), torch.cat([cy, ry]), len(idx)  # x, y, B


def _policy_ref_logps(policy, reference, x, y, device, precision):
    # Compute log-probs INSIDE autocast so cross_entropy runs fused + fp32 without a
    # full fp32 logits copy (the OOM-avoiding path).
    with _autocast(device, precision):
        p_logits, _ = policy(x)
        p_lps = sequence_logprobs(p_logits, y)
    with torch.no_grad(), _autocast(device, precision):
        r_logits, _ = reference(x)
        r_lps = sequence_logprobs(r_logits, y)
    return p_lps, r_lps


@torch.no_grad()
def _eval_dpo(policy, reference, dataset, beta, B, device, precision, max_batches=20):
    was_training = policy.training
    policy.eval()
    accs, losses = [], []
    for b in range(min(max_batches, len(dataset) // B)):
        idx = list(range(b * B, b * B + B))
        x, y, bs = _stack_batch(dataset, idx, device)
        p_lps, r_lps = _policy_ref_logps(policy, reference, x, y, device, precision)
        loss, m = dpo_loss(p_lps[:bs], p_lps[bs:], r_lps[:bs], r_lps[bs:], beta=beta)
        accs.append(m["reward_accuracy"])
        losses.append(loss.item())
    if was_training:
        policy.train()
    return (sum(accs) / len(accs), sum(losses) / len(losses)) if accs else (0.0, 0.0)


def train_dpo(cfg: TrainConfig, *, resume_from: str | None = None) -> RunDir | None:
    dist = setup_distributed(cfg.device)
    device = dist.device
    seed_everything(cfg.seed)

    policy = LithosForCausalLM(cfg.model).to(device)
    reference = LithosForCausalLM(cfg.model).to(device)
    if cfg.init_from:
        load_model_weights(cfg.init_from, policy)
        load_model_weights(cfg.init_from, reference)
    reference.eval()
    reference.requires_grad_(False)  # frozen anchor
    optimizer = build_optimizer(policy, cfg.optim)

    tok = Tokenizer.from_file(cfg.data.tokenizer_path)
    dataset = PreferenceDataset(cfg.data.corpus_manifest, tok, cfg.data.seq_len)
    val_dataset = (
        PreferenceDataset(cfg.data.val_corpus_manifest, tok, cfg.data.seq_len)
        if cfg.data.val_corpus_manifest
        else None
    )

    run: RunDir | None = None
    metrics: JsonlWriter | None = None
    reporter = Reporter(None)
    if dist.is_main:
        run = create_run_dir(cfg.run_name, base=cfg.runs_dir)
        save_resolved_config(cfg, run.resolved_config)
        metrics = JsonlWriter(run.metrics)
        gpu = torch.cuda.get_device_name(0) if device.startswith("cuda") else None
        reporter = init_reporter(
            cfg, run_id=run.root.name, run_dir=str(run.root), is_main=True,
            runtime={"resolved_device": device, "gpu": gpu, "stage": "dpo", "pairs": len(dataset)},
        )

    B = cfg.micro_batch_size
    accum = cfg.gradient_accumulation_steps
    rng = np.random.RandomState(cfg.seed)

    def batch_indices():
        n = len(dataset)
        while True:
            perm = rng.permutation(n)
            for i in range(0, n - B + 1, B):
                yield perm[i : i + B]

    biter = batch_indices()
    policy.train()
    for step in range(cfg.schedule.max_steps):
        lr = cosine_lr(step, cfg.schedule, cfg.optim.lr)
        set_lr(optimizer, lr)
        optimizer.zero_grad()
        agg = {"loss": 0.0, "reward_accuracy": 0.0, "reward_margin": 0.0}
        for _ in range(accum):
            x, y, bs = _stack_batch(dataset, next(biter), device)
            p_lps, r_lps = _policy_ref_logps(policy, reference, x, y, device, cfg.precision)
            loss, m = dpo_loss(p_lps[:bs], p_lps[bs:], r_lps[:bs], r_lps[bs:], beta=cfg.dpo_beta)
            (loss / accum).backward()
            agg["loss"] += loss.item() / accum
            agg["reward_accuracy"] += m["reward_accuracy"] / accum
            agg["reward_margin"] += m["reward_margin"] / accum
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.optim.grad_clip)
        optimizer.step()

        if dist.is_main and (step % cfg.log_interval == 0 or step == cfg.schedule.max_steps - 1):
            rec = {"step": step, "learning_rate": lr, **{k: round(v, 4) for k, v in agg.items()}}
            metrics.write(rec)  # type: ignore[union-attr]
            reporter.log(rec, step)
        if val_dataset and dist.is_main and cfg.eval_interval and step % cfg.eval_interval == 0:
            acc, vloss = _eval_dpo(policy, reference, val_dataset, cfg.dpo_beta, B, device, cfg.precision)
            rec = {"step": step, "val_reward_accuracy": round(acc, 4), "val_loss": round(vloss, 4)}
            metrics.write(rec)  # type: ignore[union-attr]
            reporter.log(rec, step)

    if dist.is_main and run is not None:
        save_checkpoint(
            run.checkpoints / f"step_{cfg.schedule.max_steps:06d}",
            model=policy, optimizer=optimizer, step=cfg.schedule.max_steps, tokens_seen=0,
            dataloader_state={},
            meta={"stage": "dpo", "init_from": cfg.init_from, "tokenizer": cfg.data.tokenizer_path,
                  "prefs": cfg.data.corpus_manifest, "beta": cfg.dpo_beta},
        )
    cleanup_distributed()
    return run
