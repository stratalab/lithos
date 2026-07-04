"""Tests for the lm-eval-harness wiring (PRD §11.2).

normalize_results is a pure function tested directly; run_benchmarks is exercised against
a fake `lm_eval` module so the wiring is proven without the heavy dependency.
"""

import sys
import types
from pathlib import Path

import pytest
import yaml
from lithos.evals.benchmarks import (
    BATTERY_VERSION,
    STEM_TASKS,
    normalize_results,
    run_benchmarks,
)


def test_normalize_prefers_acc_norm_and_computes_mean():
    raw = {
        "results": {
            "hellaswag": {
                "acc,none": 0.25,
                "acc_stderr,none": 0.01,
                "acc_norm,none": 0.26,  # preferred over acc
                "acc_norm_stderr,none": 0.01,
            },
            "piqa": {"acc,none": 0.55, "acc_stderr,none": 0.02},  # no acc_norm -> use acc
        }
    }
    out = normalize_results(raw)
    assert out["tasks"]["hellaswag"] == {"metric": "acc_norm,none", "value": 0.26}
    assert out["tasks"]["piqa"] == {"metric": "acc,none", "value": 0.55}
    assert out["num_tasks"] == 2
    assert out["mean"] == pytest.approx((0.26 + 0.55) / 2)


def test_normalize_empty():
    out = normalize_results({})
    assert out == {"tasks": {}, "mean": None, "num_tasks": 0}


def test_normalize_picks_exact_match_for_reasoning_tasks():
    # gsm8k/MATH report exact_match (no acc/acc_norm) — resolve it deterministically.
    raw = {"results": {"gsm8k": {
        "exact_match,strict-match": 0.42, "exact_match_stderr,strict-match": 0.01,
        "exact_match,flexible-extract": 0.44,
    }}}
    out = normalize_results(raw)
    assert out["tasks"]["gsm8k"] == {"metric": "exact_match,strict-match", "value": 0.42}


def test_stem_battery_covers_target_domains():
    joined = " ".join(STEM_TASKS)
    for token in ("gsm8k", "math", "physics", "chemistry", "electrical_engineering",
                  "mathematics", "computer_science"):
        assert token in joined, token


def test_stem_config_matches_constant():
    # The runnable config and the code constant must not drift.
    cfg = yaml.safe_load((Path("configs/eval/stem.yaml")).read_text())
    assert cfg["benchmarks"]["tasks"] == STEM_TASKS
    assert cfg["benchmarks"]["battery_version"] == "stem-v1"


class _FakeLmEval(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("lm_eval")
        self.call: dict | None = None

    def simple_evaluate(self, **kwargs):
        self.call = kwargs
        return {"results": {"piqa": {"acc,none": 0.6}, "sciq": {"acc,none": 0.7}}}


def test_run_benchmarks_forwards_and_normalizes(monkeypatch):
    fake = _FakeLmEval()
    monkeypatch.setitem(sys.modules, "lm_eval", fake)
    out = run_benchmarks(
        "/tmp/export", ["piqa", "sciq"], num_fewshot=0, limit=5, dtype="bfloat16", device="cpu"
    )
    # forwarded correctly to the harness
    assert fake.call["model"] == "hf"
    assert "pretrained=/tmp/export" in fake.call["model_args"]
    assert "dtype=bfloat16" in fake.call["model_args"]
    assert fake.call["tasks"] == ["piqa", "sciq"]
    assert fake.call["limit"] == 5
    # normalized + stamped with the frozen battery version
    assert out["tasks"]["piqa"]["value"] == 0.6
    assert out["battery_version"] == BATTERY_VERSION
    assert out["mean"] == pytest.approx(0.65)


def test_missing_lm_eval_raises_helpful_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "lm_eval", None)  # force ImportError on `import lm_eval`
    with pytest.raises(ImportError, match="lm-eval is required"):
        run_benchmarks("/tmp/x", ["piqa"])
