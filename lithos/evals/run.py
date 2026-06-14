"""Evaluation orchestration (PRD §11): perplexity + samples -> report (+ export)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from safetensors.torch import load_model

from lithos.data.dataloader import PackedDataLoader, PackedDataset
from lithos.evals.config import EvalConfig
from lithos.evals.generate_samples import generate_samples
from lithos.evals.perplexity import compute_perplexity
from lithos.evals.report import write_eval_report
from lithos.model import LithosForCausalLM
from lithos.serve.export import export_hf
from lithos.tokenizer import DEFAULT_SPECIAL_TOKENS, load_tokenizer, special_token_ids
from lithos.train.config import TrainConfig
from lithos.utils.config import load_and_validate
from lithos.utils.device import resolve_device
from lithos.utils.io import read_json


def _read_shards(manifest_path: str) -> list:
    man = read_json(manifest_path)
    return [(s["path"], s["num_tokens"], s["dtype"]) for s in man["shards"]]


def load_model_from_checkpoint(checkpoint_path: str) -> tuple[LithosForCausalLM, TrainConfig]:
    """Reconstruct a model from a checkpoint dir + the run's resolved config."""
    ckpt = Path(checkpoint_path)
    resolved = ckpt.parent.parent / "resolved_config.yaml"
    train_cfg = load_and_validate(resolved, TrainConfig)
    model = LithosForCausalLM(train_cfg.model)
    load_model(model, str(ckpt / "model.safetensors"))
    model.eval()
    return model, train_cfg


def run_evaluation(
    model: LithosForCausalLM,
    tokenizer: Any,
    *,
    val_loader: PackedDataLoader | None = None,
    val_batches: int = 50,
    prompts: list[str] | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Compute the eval results dict (perplexity and/or samples)."""
    results: dict[str, Any] = {}
    if val_loader is not None:
        results["perplexity"] = compute_perplexity(model, val_loader, val_batches, device)
    if prompts:
        results["samples"] = generate_samples(
            model, tokenizer, list(prompts), device=device, **(sample_kwargs or {})
        )
    return results


def evaluate_checkpoint(cfg: EvalConfig, checkpoint_path: str) -> Path:
    """Full eval of a checkpoint: report (+ optional HF export). Returns report dir."""
    model, train_cfg = load_model_from_checkpoint(checkpoint_path)
    device = resolve_device("auto")
    model.to(device)
    tokenizer = load_tokenizer(cfg.tokenizer_path)

    val_loader = None
    if cfg.val_corpus_manifest:
        shards = _read_shards(cfg.val_corpus_manifest)
        val_loader = PackedDataLoader(PackedDataset(shards, train_cfg.data.seq_len), cfg.batch_size)

    results = run_evaluation(
        model,
        tokenizer,
        val_loader=val_loader,
        val_batches=cfg.eval_batches,
        prompts=cfg.prompts,
        sample_kwargs={"max_new_tokens": cfg.sample_max_new_tokens, "greedy": cfg.greedy},
        device=device,
    )
    reference = {
        "checkpoint": str(checkpoint_path),
        "tokenizer": cfg.tokenizer_path,
        "corpus": cfg.val_corpus_manifest,
        "num_parameters": model.num_parameters(),
        "sequence_length": train_cfg.data.seq_len,
    }
    out = write_eval_report(
        Path(cfg.output_dir) / cfg.name,
        name=cfg.name,
        results=results,
        model_reference=reference,
        config=cfg.model_dump(),
    )
    if cfg.export_dir:
        export_hf(
            model.cpu(),
            cfg.export_dir,
            tokenizer_path=cfg.tokenizer_path,
            special_ids=special_token_ids(tokenizer, DEFAULT_SPECIAL_TOKENS),
        )
    return out
