"""The explicit single-process training loop (PRD §9.2).

No hidden trainer abstraction (PRD §20.2): construct model -> load shards ->
forward -> loss -> backward -> clip -> optimizer step -> schedule -> log / eval /
checkpoint, with exact resume.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import subprocess
import time
from typing import Any

import torch

from lithos.data.dataloader import PackedDataLoader, PackedDataset
from lithos.model import LithosForCausalLM
from lithos.train.checkpoint import load_checkpoint, save_checkpoint
from lithos.train.config import TrainConfig
from lithos.train.logging import JsonlWriter, RunDir, create_run_dir
from lithos.train.optim import build_optimizer
from lithos.train.scheduler import cosine_lr, set_lr
from lithos.utils.config import save_resolved_config
from lithos.utils.device import resolve_device
from lithos.utils.io import read_json, write_json
from lithos.utils.seed import seed_everything


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    return out.stdout.strip() or None


def load_corpus_shards(manifest_path: str) -> tuple[list, dict[str, Any]]:
    man = read_json(manifest_path)
    shards = [(s["path"], s["num_tokens"], s["dtype"]) for s in man["shards"]]
    return shards, man


def _build_loader(
    manifest_path: str, seq_len: int, batch_size: int, seed: int
) -> tuple[PackedDataLoader, dict[str, Any]]:
    shards, man = load_corpus_shards(manifest_path)
    return PackedDataLoader(PackedDataset(shards, seq_len), batch_size, seed=seed), man


def _autocast(device: str, precision: str) -> Any:
    if device.startswith("cuda") and precision in ("bf16", "fp16"):
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, loader: PackedDataLoader, n_steps: int, device: str, precision: str
) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    for _ in range(n_steps):
        x, y = next(loader)
        with _autocast(device, precision):
            _, loss = model(x.to(device), targets=y.to(device))
        total += float(loss)
    if was_training:
        model.train()
    return total / max(1, n_steps)


def _run_manifest(
    cfg: TrainConfig, run: RunDir, model: LithosForCausalLM, device: str, corpus_man: dict[str, Any]
) -> dict[str, Any]:
    return {
        "run_id": run.root.name,
        "git_commit": _git_commit(),
        "resolved_config": str(run.resolved_config),
        "tokenizer": corpus_man.get("tokenizer"),
        "corpus": cfg.data.corpus_manifest,
        "num_parameters": model.num_parameters(),
        "sequence_length": cfg.data.seq_len,
        "micro_batch_size": cfg.micro_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "world_size": 1,
        "global_batch_size": cfg.global_batch_size,
        "tokens_per_step": cfg.tokens_per_step,
        "precision": cfg.precision,
        "device": device,
    }


def _save(
    run: RunDir,
    model: LithosForCausalLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokens_seen: int,
    loader: PackedDataLoader,
    cfg: TrainConfig,
    corpus_man: dict[str, Any],
) -> None:
    save_checkpoint(
        run.checkpoints / f"step_{step:06d}",
        model=model,
        optimizer=optimizer,
        step=step,
        tokens_seen=tokens_seen,
        dataloader_state=loader.state_dict(),
        meta={
            "tokenizer": corpus_man.get("tokenizer"),
            "corpus": cfg.data.corpus_manifest,
            "resolved_config": str(run.resolved_config),
        },
    )


def train(cfg: TrainConfig, *, resume_from: str | None = None) -> RunDir:
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    raw_model = LithosForCausalLM(cfg.model).to(device)
    raw_model.gradient_checkpointing = cfg.grad_checkpointing
    model: torch.nn.Module = raw_model
    if cfg.compile:
        model = torch.compile(raw_model)  # type: ignore[assignment]
    optimizer = build_optimizer(model, cfg.optim)

    loader, corpus_man = _build_loader(
        cfg.data.corpus_manifest, cfg.data.seq_len, cfg.micro_batch_size, cfg.seed
    )
    val_loader = None
    if cfg.data.val_corpus_manifest:
        val_loader, _ = _build_loader(
            cfg.data.val_corpus_manifest, cfg.data.seq_len, cfg.micro_batch_size, cfg.seed + 1
        )

    step = 0
    tokens_seen = 0
    if resume_from:
        state = load_checkpoint(resume_from, raw_model, optimizer)
        step = state["step"]
        tokens_seen = state["tokens_seen"]
        loader.load_state_dict(state["dataloader"])

    run = create_run_dir(cfg.run_name, base=cfg.runs_dir)
    save_resolved_config(cfg, run.resolved_config)
    write_json(run.manifest, _run_manifest(cfg, run, raw_model, device, corpus_man))
    metrics = JsonlWriter(run.metrics)

    use_scaler = device.startswith("cuda") and cfg.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    model.train()
    t0 = time.time()
    tokens_at_t0 = tokens_seen
    try:
        while step < cfg.schedule.max_steps:
            lr = cosine_lr(step, cfg.schedule, cfg.optim.lr)
            set_lr(optimizer, lr)
            optimizer.zero_grad(set_to_none=True)

            loss_sum = 0.0
            for _ in range(cfg.gradient_accumulation_steps):
                x, y = next(loader)
                with _autocast(device, cfg.precision):
                    _, loss = model(x.to(device), targets=y.to(device))
                loss = loss / cfg.gradient_accumulation_steps
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                loss_sum += loss.item()

            if scaler is not None:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            step += 1
            tokens_seen += cfg.tokens_per_step

            if step % cfg.log_interval == 0 or step == cfg.schedule.max_steps:
                now = time.time()
                elapsed = now - t0
                tps = (tokens_seen - tokens_at_t0) / elapsed if elapsed > 0 else 0.0
                t0, tokens_at_t0 = now, tokens_seen
                record: dict[str, Any] = {
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "train_loss": loss_sum,
                    "learning_rate": lr,
                    "grad_norm": float(grad_norm),
                    "throughput_tokens_per_sec": tps,
                    "timestamp": dt.datetime.now(dt.UTC).isoformat(),
                }
                if device.startswith("cuda"):
                    record["gpu_memory_allocated"] = int(torch.cuda.memory_allocated())
                metrics.write(record)

            if val_loader is not None and cfg.eval_interval and step % cfg.eval_interval == 0:
                vloss = evaluate(model, val_loader, cfg.eval_steps, device, cfg.precision)
                metrics.write(
                    {
                        "step": step,
                        "validation_loss": vloss,
                        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
                    }
                )

            if (
                cfg.checkpoint_interval
                and step % cfg.checkpoint_interval == 0
                and step < cfg.schedule.max_steps
            ):
                _save(run, raw_model, optimizer, step, tokens_seen, loader, cfg, corpus_man)
    except KeyboardInterrupt:
        _save(run, raw_model, optimizer, step, tokens_seen, loader, cfg, corpus_man)
        metrics.close()
        raise

    _save(run, raw_model, optimizer, step, tokens_seen, loader, cfg, corpus_man)
    metrics.close()
    return run
