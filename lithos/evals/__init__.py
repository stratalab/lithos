"""Evaluation: perplexity, sample generation, lm-eval-harness path, reports (Phase 5)."""

from lithos.evals.config import EvalConfig
from lithos.evals.generate_samples import generate_samples, repetition_score
from lithos.evals.perplexity import compute_perplexity
from lithos.evals.report import write_eval_report
from lithos.evals.run import evaluate_checkpoint, load_model_from_checkpoint, run_evaluation

__all__ = [
    "EvalConfig",
    "compute_perplexity",
    "evaluate_checkpoint",
    "generate_samples",
    "load_model_from_checkpoint",
    "repetition_score",
    "run_evaluation",
    "write_eval_report",
]
