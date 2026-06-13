"""Tests for lithos.train.logging — run-dir layout, no-clobber, JSONL writer."""

import json
from datetime import datetime

import pytest
from lithos.train.logging import JsonlWriter, create_run_dir

FIXED = datetime(2026, 6, 13, 12, 0, 0)


def test_create_run_dir_layout(tmp_path):
    run = create_run_dir("lithos-toy", base=tmp_path, now=FIXED)
    assert run.root.name == "2026-06-13_120000_lithos-toy"
    for sub in (run.samples, run.checkpoints, run.evals):
        assert sub.is_dir()


def test_create_run_dir_refuses_clobber(tmp_path):
    create_run_dir("dup", base=tmp_path, now=FIXED)
    with pytest.raises(FileExistsError):
        create_run_dir("dup", base=tmp_path, now=FIXED)


def test_jsonl_writer_appends_records(tmp_path):
    p = tmp_path / "metrics.jsonl"
    with JsonlWriter(p) as w:
        w.write({"step": 1, "loss": 2.0})
        w.write({"step": 2, "loss": 1.5})
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["step"] == 1
    assert json.loads(lines[1])["loss"] == 1.5
