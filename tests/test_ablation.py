"""Tests for the data-recipe ablation harness (Phase 10).

`summarize` is tested directly; `run_ablation` is exercised with the three heavy steps
(build_corpus / train / evaluate_checkpoint) mocked, so the loop is validated without a GPU.
"""

import types

import pytest
from lithos.evals import ablation as ab
from lithos.evals.ablation import AblationConfig, AblationVariant, run_ablation, summarize
from lithos.evals.scorecard import append_entry


def _row(label, mean):
    return {
        "label": label,
        "benchmarks": {
            "battery_version": "v1",
            "mean": mean,
            "tasks": {"arc_easy": {"metric": "acc", "value": mean}},
        },
    }


def test_summarize_diffs_and_ranks_winners(tmp_path):
    sc = tmp_path / "sc.jsonl"
    append_entry(sc, _row("baseline", 0.50))
    append_entry(sc, _row("quality-4.0", 0.55))  # beats baseline
    append_entry(sc, _row("aggressive", 0.45))  # worse than baseline
    s = summarize(str(sc), baseline="baseline")
    assert s["baseline_mean"] == 0.50
    assert s["variants"]["quality-4.0"]["mean_delta"] == pytest.approx(0.05)
    assert s["variants"]["aggressive"]["mean_delta"] == pytest.approx(-0.05)
    assert s["winners"] == ["quality-4.0"]  # only the variant that beat baseline


def test_summarize_missing_baseline_raises(tmp_path):
    sc = tmp_path / "sc.jsonl"
    append_entry(sc, _row("a", 0.5))
    with pytest.raises(ValueError, match="baseline"):
        summarize(str(sc), baseline="does-not-exist")


def test_run_ablation_orchestration(tmp_path, monkeypatch):
    calls = {"build": [], "train": [], "eval": []}
    means = {"baseline": 0.40, "better": 0.50}

    def fake_load_and_validate(path, model_cls, overrides=None):
        ov = {o.split("=", 1)[0]: o.split("=", 1)[1] for o in (overrides or []) if "=" in o}
        return types.SimpleNamespace(
            output_dir=ov.get("output_dir", str(tmp_path / "c")),
            name=ov.get("name"),
            scorecard_path=ov.get("scorecard_path"),
            overrides=list(overrides or []),
        )

    def fake_train(cfg):
        calls["train"].append(cfg.overrides)
        return types.SimpleNamespace(root=tmp_path)

    def fake_eval(eval_cfg, ckpt):
        calls["eval"].append(eval_cfg.name)
        append_entry(eval_cfg.scorecard_path, _row(eval_cfg.name, means[eval_cfg.name]))

    monkeypatch.setattr(ab, "load_and_validate", fake_load_and_validate)
    monkeypatch.setattr(ab, "build_corpus", lambda c: calls["build"].append(c.overrides))
    monkeypatch.setattr(ab, "train", fake_train)
    monkeypatch.setattr(ab, "find_latest_checkpoint", lambda root: tmp_path / "step_1")
    monkeypatch.setattr(ab, "evaluate_checkpoint", fake_eval)

    cfg = AblationConfig(
        name="t",
        corpus_config="x.yaml",
        train_config="y.yaml",
        eval_config="z.yaml",
        baseline="baseline",
        work_dir=str(tmp_path / "work"),
        variants=[
            AblationVariant(name="baseline"),
            AblationVariant(name="better", corpus_overrides=["quality.threshold=4.0"]),
        ],
    )
    summary = run_ablation(cfg)

    # both variants ran, in order, through all three stages
    assert calls["eval"] == ["baseline", "better"]
    assert len(calls["build"]) == 2 and len(calls["train"]) == 2
    # the variant's corpus override actually reached the build step
    assert any("quality.threshold=4.0" in ovs for ovs in calls["build"])
    assert not any("quality.threshold=4.0" in ovs for ovs in [calls["build"][0]])  # not the baseline
    # and the comparison came out right
    assert summary["variants"]["better"]["mean_delta"] == pytest.approx(0.10)
    assert summary["winners"] == ["better"]
