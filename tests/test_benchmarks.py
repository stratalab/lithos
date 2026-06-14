"""Tests for the lm-eval-harness wiring (PRD §11.2).

normalize_results is a pure function tested directly; run_benchmarks is exercised against
a fake `lm_eval` module so the wiring is proven without the heavy dependency.
"""

import sys
import types

import pytest
from lithos.evals.benchmarks import BATTERY_VERSION, normalize_results, run_benchmarks


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
