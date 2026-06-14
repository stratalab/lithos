"""Tests for lithos.evals.report — JSON/MD/config/reference output (PRD §11.3)."""

import json

from lithos.evals.report import write_eval_report


def test_write_eval_report_emits_all_files(tmp_path):
    results = {
        "perplexity": {"loss": 2.5, "perplexity": 12.18, "tokens": 1024},
        "samples": [{"prompt": "hi", "completion": "there", "n_new_tokens": 1, "repetition": 0.0}],
    }
    reference = {"checkpoint": "runs/x/checkpoints/step_100", "num_parameters": 123456}
    out = write_eval_report(
        tmp_path / "eval",
        name="base",
        results=results,
        model_reference=reference,
        config={"name": "base", "eval_batches": 20},
    )

    for fname in ("results.json", "results.md", "config.yaml", "model_reference.json"):
        assert (out / fname).is_file()

    assert json.loads((out / "results.json").read_text())["perplexity"]["perplexity"] == 12.18
    md = (out / "results.md").read_text()
    assert "Perplexity" in md
    assert "12.18" in md
    assert "123456" in md  # model reference rendered


def test_report_renders_benchmark_scores(tmp_path):
    results = {
        "benchmarks": {
            "battery_version": "v1",
            "num_fewshot": 0,
            "mean": 0.42,
            "tasks": {"arc_easy": {"metric": "acc_norm,none", "value": 0.55}},
        }
    }
    out = write_eval_report(
        tmp_path / "eval", name="base", results=results, model_reference={}, config={}
    )
    md = (out / "results.md").read_text()
    assert "Benchmarks (battery v1, 0-shot)" in md
    assert "arc_easy" in md and "0.5500" in md
    assert "mean" in md and "0.4200" in md
    # the full benchmark block also round-trips in results.json (not just the scorecard)
    assert json.loads((out / "results.json").read_text())["benchmarks"]["mean"] == 0.42
