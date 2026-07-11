"""Tests for the public-benchmark freeze/publish tooling (E8 Part B).

Covers the credibility artifacts that need no model: post-cutoff freeze, version-locked
content hash, canary embed/detect, the written bundle (round-trips through load_tasks +
registers prompts for decontam), and a leaderboard that shows losses.
"""

import json

from lithos.evals.benchmark_publish import (
    canary_line,
    content_sha256,
    find_canary,
    freeze_benchmark,
    frozen_task,
    render_leaderboard,
    write_benchmark,
)
from lithos.posttrain.taskbank import Task, load_tasks


def _num(id_, year, **kw):
    return Task(id=id_, prompt=f"prompt {id_}", kind="numeric", answer="1", year=year, **kw)


def test_freeze_takes_postcutoff_only():
    tasks = [
        _num("a", 2023, level="easy"),
        _num("b", 2025, level="hard"),
        Task(id="c", prompt="write f", kind="code", tests="assert f() == 3", year=2025, level="hard"),
    ]
    art = freeze_benchmark(
        tasks, version="tir-v1", cutoff_year=2024, canary_guid="deadbeef", created_at="2026-07-11"
    )
    assert [r["id"] for r in art.frozen_tasks] == ["b", "c"]  # post-2024 only, sorted
    assert art.manifest["num_tasks"] == 2
    assert art.manifest["cutoff_year"] == 2024
    # code task keeps `tests`, not `answer`; numeric keeps `answer`
    code = next(r for r in art.frozen_tasks if r["id"] == "c")
    assert code["tests"] == "assert f() == 3" and "answer" not in code


def test_freeze_empty_slice_raises():
    import pytest

    with pytest.raises(ValueError, match="no post-"):
        freeze_benchmark(
            [_num("a", 2020)], version="v", cutoff_year=2024, canary_guid="x", created_at="t"
        )


def test_content_hash_order_independent_and_content_sensitive():
    t1 = [_num("a", 2025), _num("b", 2025)]
    a1 = freeze_benchmark(t1, version="v", cutoff_year=2024, canary_guid="x", created_at="t")
    a2 = freeze_benchmark(
        list(reversed(t1)), version="v", cutoff_year=2024, canary_guid="x", created_at="t"
    )
    assert a1.manifest["content_sha256"] == a2.manifest["content_sha256"]  # order-independent

    t3 = [Task(id="a", prompt="CHANGED", kind="numeric", answer="1", year=2025), _num("b", 2025)]
    a3 = freeze_benchmark(t3, version="v", cutoff_year=2024, canary_guid="x", created_at="t")
    assert a3.manifest["content_sha256"] != a1.manifest["content_sha256"]  # content-sensitive
    # the canary GUID is provenance, not content — it must not move the hash
    assert content_sha256(a1.frozen_tasks) == content_sha256(a2.frozen_tasks)


def test_frozen_task_round_trips_through_task_from_record():
    from lithos.posttrain.taskbank import task_from_record

    t = Task(id="z", prompt="p", kind="numeric", answer="42", year=2025, level="hard", family_id="F1")
    back = task_from_record(frozen_task(t))
    assert back.id == "z" and back.answer == "42" and back.year == 2025 and back.family_id == "F1"


def test_canary_line_and_detect():
    line = canary_line("abc123")
    assert "abc123" in line
    assert find_canary(f"lorem\n{line}\nipsum", "abc123") is True
    assert find_canary("no canary here", "abc123") is False


def test_write_benchmark_bundle(tmp_path):
    tasks = [
        Task(id="b", prompt="compute the flux", kind="numeric", answer="2", year=2025, level="hard"),
        Task(id="c", prompt="write g", kind="code", tests="assert g() == 3", year=2025, level="hard"),
    ]
    art = freeze_benchmark(
        tasks, version="tir-v1", cutoff_year=2024, canary_guid="cafe", created_at="2026-07-11"
    )
    out = write_benchmark(tmp_path / "bench", art)

    for fname in ("benchmark.jsonl", "manifest.json", "canary.txt", "README.md", "decontam_probes.jsonl"):
        assert (out / fname).exists(), f"missing {fname}"

    # tasks round-trip through the loader the harness uses
    reloaded = load_tasks(out / "benchmark.jsonl")
    assert {t.id for t in reloaded} == {"b", "c"}
    assert next(t for t in reloaded if t.id == "c").tests == "assert g() == 3"

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["content_sha256"] == art.manifest["content_sha256"]
    assert "cafe" in (out / "canary.txt").read_text()

    readme = (out / "README.md").read_text()
    assert "tir-v1" in readme and "canary" in readme.lower() and "Sandbagged" in readme

    probes = [
        json.loads(line)["text"]
        for line in (out / "decontam_probes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert "compute the flux" in probes  # prompt registered for training decontam
    assert any("cafe" in p for p in probes)  # canary registered too


def _entry(label, off, on, uplift, lo, hi, sig):
    return {
        "label": label,
        "tir": {
            "n": 50,
            "battery_version": "tir-v1",
            "overall": {
                "solve_off": off, "solve_on": on, "uplift": uplift,
                "ci_low": lo, "ci_high": hi, "significant": sig,
            },
        },
    }


def test_leaderboard_shows_losses_and_sorts():
    entries = [
        _entry("Lithos-3B", 0.30, 0.60, 0.30, 0.20, 0.40, True),
        _entry("Qwen-8B", 0.50, 0.55, 0.05, -0.02, 0.12, False),
        _entry("WeakBaseline", 0.10, 0.08, -0.02, -0.10, 0.06, False),
    ]
    md = render_leaderboard(entries)
    # losses are not hidden: the negative-uplift model still appears
    assert "WeakBaseline" in md and "-0.020" in md
    # sorted by tools-on solve rate, descending
    assert md.index("Lithos-3B") < md.index("Qwen-8B") < md.index("WeakBaseline")
    assert "Losses are shown" in md
