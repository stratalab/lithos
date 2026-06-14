"""Data-recipe ablation harness (Phase 10): intervention -> proxy -> eval -> keep winners.

The loop that gives Phase 10 its acceptance criterion. A *variant* is a set of dotted-key
overrides to the base corpus-build config (e.g. ``quality.threshold=4.0`` or
``near_dedup=true``). For each variant the harness:

    1. builds the corpus variant,
    2. trains the SAME small proxy on it,
    3. scores it on the SAME frozen eval battery (a scorecard row labeled by the variant),

then diffs every variant against the baseline. Proxy and battery are held constant, so any
score gap is attributable to the data recipe — that's the whole point.

Real runs need a GPU (the proxy train); the orchestration is unit-tested with the three
heavy steps (build_corpus / train / evaluate_checkpoint) mocked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lithos.data.pipeline import CorpusBuildConfig, build_corpus
from lithos.evals.config import EvalConfig
from lithos.evals.run import evaluate_checkpoint
from lithos.evals.scorecard import diff, read_entries
from lithos.train.checkpoint import find_latest_checkpoint
from lithos.train.config import TrainConfig
from lithos.train.loop import train
from lithos.utils.config import load_and_validate


class AblationVariant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    corpus_overrides: list[str] = Field(default_factory=list)  # dotted-key overrides to the corpus


class AblationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    corpus_config: str  # base CorpusBuildConfig path
    train_config: str  # base TrainConfig path (the shared proxy)
    eval_config: str  # base EvalConfig path (the frozen battery)
    variants: list[AblationVariant]
    baseline: str | None = None  # variant to diff against (default: the first)
    work_dir: str = "runs/ablation"
    train_overrides: list[str] = Field(default_factory=list)  # applied to every proxy train
    eval_overrides: list[str] = Field(default_factory=list)


def run_variant(cfg: AblationConfig, variant: AblationVariant, scorecard_path: str) -> None:
    """Build the corpus variant, train the proxy, and score it onto the shared scorecard."""
    base = Path(cfg.work_dir) / cfg.name / variant.name

    corpus_cfg = load_and_validate(
        cfg.corpus_config,
        CorpusBuildConfig,
        [*variant.corpus_overrides, f"name={cfg.name}-{variant.name}", f"output_dir={base / 'corpus'}"],
    )
    build_corpus(corpus_cfg)
    corpus_manifest = str(Path(corpus_cfg.output_dir) / "corpus_manifest.json")

    train_cfg = load_and_validate(
        cfg.train_config,
        TrainConfig,
        [
            *cfg.train_overrides,
            f"data.corpus_manifest={corpus_manifest}",
            f"run_name={cfg.name}-{variant.name}",
            f"runs_dir={base / 'runs'}",
        ],
    )
    run = train(train_cfg)
    if run is None:
        raise RuntimeError("proxy train produced no run dir (distributed non-main rank?)")
    ckpt = find_latest_checkpoint(run.root)
    if ckpt is None:
        raise RuntimeError(f"proxy train wrote no checkpoint for variant {variant.name!r}")

    eval_cfg = load_and_validate(
        cfg.eval_config,
        EvalConfig,
        [
            *cfg.eval_overrides,
            f"name={variant.name}",
            f"data_recipe={variant.name}",
            f"scorecard_path={scorecard_path}",
            f"output_dir={base / 'eval'}",
        ],
    )
    evaluate_checkpoint(eval_cfg, str(ckpt))


def summarize(scorecard_path: str, *, baseline: str) -> dict[str, Any]:
    """Diff every variant against ``baseline`` (positive mean_delta = variant beats baseline)."""
    entries = {e["label"]: e for e in read_entries(scorecard_path)}
    if baseline not in entries:
        raise ValueError(f"baseline {baseline!r} not found in scorecard {scorecard_path}")
    base = entries[baseline]
    variants = {
        label: diff(base, e)  # diff(a, b) = b - a, so this is variant - baseline
        for label, e in entries.items()
        if label != baseline
    }
    winners = sorted(
        (lbl for lbl, d in variants.items() if (d.get("mean_delta") or 0) > 0),
        key=lambda lbl: variants[lbl]["mean_delta"],
        reverse=True,
    )
    return {
        "baseline": baseline,
        "baseline_mean": base.get("benchmarks", {}).get("mean"),
        "variants": variants,
        "winners": winners,  # variants that beat baseline, best first
    }


def run_ablation(cfg: AblationConfig) -> dict[str, Any]:
    """Run every variant and return the comparison summary."""
    if not cfg.variants:
        raise ValueError("ablation has no variants")
    scorecard_path = str(Path(cfg.work_dir) / cfg.name / "scorecard.jsonl")
    for variant in cfg.variants:
        run_variant(cfg, variant, scorecard_path)
    return summarize(scorecard_path, baseline=cfg.baseline or cfg.variants[0].name)
