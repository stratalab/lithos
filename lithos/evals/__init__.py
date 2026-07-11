"""Evaluation: perplexity, sample generation, lm-eval-harness path, reports (Phase 5)."""

from lithos.evals.benchmark_publish import (
    BenchmarkArtifact,
    freeze_benchmark,
    render_leaderboard,
    write_benchmark,
)
from lithos.evals.config import EvalConfig
from lithos.evals.generate_samples import generate_samples, repetition_score
from lithos.evals.perplexity import compute_perplexity
from lithos.evals.report import write_eval_report
from lithos.evals.run import evaluate_checkpoint, load_model_from_checkpoint, run_evaluation
from lithos.evals.tir_battery import run_tir_battery_eval, run_two_arm, summarize
from lithos.evals.tir_stats import paired_uplift

__all__ = [
    "BenchmarkArtifact",
    "EvalConfig",
    "compute_perplexity",
    "evaluate_checkpoint",
    "freeze_benchmark",
    "generate_samples",
    "load_model_from_checkpoint",
    "paired_uplift",
    "render_leaderboard",
    "repetition_score",
    "run_evaluation",
    "run_tir_battery_eval",
    "run_two_arm",
    "summarize",
    "write_benchmark",
    "write_eval_report",
]
