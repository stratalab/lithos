"""Evaluation orchestration (PRD §11): perplexity + samples -> report (+ export)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from safetensors.torch import load_model

from lithos.data.dataloader import PackedDataLoader, PackedDataset
from lithos.data.shard import read_shard_specs
from lithos.evals.benchmarks import run_benchmarks
from lithos.evals.config import EvalConfig
from lithos.evals.generate_samples import generate_samples
from lithos.evals.perplexity import compute_perplexity
from lithos.evals.report import write_eval_report
from lithos.evals.scorecard import append_entry
from lithos.model import LithosForCausalLM
from lithos.serve.export import export_hf
from lithos.tokenizer import DEFAULT_SPECIAL_TOKENS, load_tokenizer, special_token_ids
from lithos.train.config import TrainConfig
from lithos.utils.config import load_and_validate
from lithos.utils.device import resolve_device


def _read_shards(manifest_path: str) -> list:
    return read_shard_specs(manifest_path)


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
    """Full eval of a checkpoint: perplexity + samples + the frozen benchmark battery,
    written to a versioned report and (optionally) appended to a comparable scorecard.
    """
    model, train_cfg = load_model_from_checkpoint(checkpoint_path)
    device = resolve_device("auto")
    model.to(device)
    tokenizer = load_tokenizer(cfg.tokenizer_path)
    num_params = model.num_parameters()

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

    # Export once if needed (the benchmark harness loads via HF, or an explicit export
    # was requested), then run the frozen battery against that export directory.
    export_path = cfg.export_dir
    if cfg.benchmarks.enabled and export_path is None:
        export_path = str(Path(cfg.output_dir) / cfg.name / "hf_export")
    if export_path is not None:
        export_hf(
            model.cpu(),  # export reads weights on CPU; nothing below needs the GPU copy
            export_path,
            tokenizer_path=cfg.tokenizer_path,
            special_ids=special_token_ids(tokenizer, DEFAULT_SPECIAL_TOKENS),
            dtype=cfg.benchmarks.dtype,
        )
    if cfg.benchmarks.enabled:
        assert export_path is not None  # set above whenever benchmarks are enabled
        results["benchmarks"] = run_benchmarks(
            export_path,
            cfg.benchmarks.tasks,
            battery_version=cfg.benchmarks.battery_version,
            num_fewshot=cfg.benchmarks.num_fewshot,
            limit=cfg.benchmarks.limit,
            batch_size=cfg.benchmarks.batch_size,
            dtype=cfg.benchmarks.dtype,
            device=device,
        )

    reference = {
        "checkpoint": str(checkpoint_path),
        "tokenizer": cfg.tokenizer_path,
        "corpus": cfg.val_corpus_manifest,
        "num_parameters": num_params,
        "sequence_length": train_cfg.data.seq_len,
        "data_recipe": cfg.data_recipe,
    }
    out = write_eval_report(
        Path(cfg.output_dir) / cfg.name,
        name=cfg.name,
        results=results,
        model_reference=reference,
        config=cfg.model_dump(),
    )

    if cfg.scorecard_path:
        append_entry(
            cfg.scorecard_path,
            {
                "label": cfg.name,
                "timestamp": dt.datetime.now(dt.UTC).isoformat(),
                "checkpoint": str(checkpoint_path),
                "num_parameters": num_params,
                "data_recipe": cfg.data_recipe,
                "perplexity": results.get("perplexity"),
                "benchmarks": results.get("benchmarks"),
            },
        )
    return out
