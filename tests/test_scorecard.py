"""Tests for the append-only benchmark scorecard (PRD §11.3)."""

import pytest
from lithos.evals.scorecard import append_entry, diff, read_entries


def _entry(label, hellaswag, piqa, version="v1"):
    return {
        "label": label,
        "benchmarks": {
            "battery_version": version,
            "mean": (hellaswag + piqa) / 2,
            "tasks": {
                "hellaswag": {"metric": "acc_norm", "value": hellaswag},
                "piqa": {"metric": "acc", "value": piqa},
            },
        },
    }


def test_append_and_read(tmp_path):
    p = tmp_path / "sc.jsonl"
    append_entry(p, _entry("a", 0.25, 0.50))
    append_entry(p, _entry("b", 0.30, 0.55))
    rows = read_entries(p)
    assert [r["label"] for r in rows] == ["a", "b"]


def test_read_missing_is_empty(tmp_path):
    assert read_entries(tmp_path / "nope.jsonl") == []


def test_diff_per_task_and_mean(tmp_path):
    d = diff(_entry("base", 0.25, 0.50), _entry("recipe-B", 0.30, 0.55))
    assert d["per_task"]["hellaswag"] == pytest.approx(0.05)
    assert d["per_task"]["piqa"] == pytest.approx(0.05)
    assert d["mean_delta"] == pytest.approx(0.05)
    assert d["a"] == "base" and d["b"] == "recipe-B"


def test_diff_refuses_cross_version():
    with pytest.raises(ValueError, match="battery versions"):
        diff(_entry("a", 0.25, 0.5, version="v1"), _entry("b", 0.3, 0.55, version="v2"))
