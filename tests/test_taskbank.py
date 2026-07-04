"""Tests for the task bank + verify dispatch + year split (lithos/posttrain/taskbank.py)."""

import json
import sys

import pytest
from lithos.posttrain.taskbank import (
    Task,
    assert_disjoint,
    filter_by_level,
    load_tasks,
    split_by_year,
    task_from_record,
    verify,
    verify_batch,
)

_POSIX = sys.platform.startswith(("linux", "darwin"))


def test_task_validation():
    with pytest.raises(ValueError, match="unknown task kind"):
        Task(id="x", prompt="p", kind="bogus")
    with pytest.raises(ValueError, match="needs a `tests`"):
        Task(id="c", prompt="p", kind="code")
    with pytest.raises(ValueError, match="needs a `units`"):
        Task(id="u", prompt="p", kind="units")


def test_task_from_record_defaults():
    t = task_from_record({"id": 7, "prompt": "2+2?", "answer": 4})
    assert t.id == "7" and t.kind == "numeric" and t.answer == "4"


def test_load_tasks(tmp_path):
    path = tmp_path / "bank.jsonl"
    path.write_text(
        json.dumps({"id": "a", "prompt": "1+1?", "kind": "numeric", "answer": "2"})
        + "\n"
        + json.dumps({"id": "b", "prompt": "simplify", "kind": "symbolic", "answer": "x"})
        + "\n"
    )
    tasks = load_tasks(path)
    assert [t.id for t in tasks] == ["a", "b"]


def test_verify_numeric_and_symbolic():
    assert verify("the answer is 2", Task(id="a", prompt="1+1", answer="2")).correct
    sym = Task(id="s", prompt="factor", kind="symbolic", answer="x**2 - 1")
    assert verify("(x-1)*(x+1)", sym).correct


@pytest.mark.skipif(not _POSIX, reason="POSIX-only sandbox")
def test_verify_code():
    t = Task(
        id="c",
        prompt="write add",
        kind="code",
        tests="assert add(2, 2) == 4",
    )
    assert verify("def add(a, b):\n    return a + b", t).correct
    assert not verify("def add(a, b):\n    return a - b", t).correct


@pytest.mark.skipif(not _POSIX, reason="POSIX-only sandbox")
def test_verify_batch_preserves_order():
    tasks = [Task(id=str(i), prompt="p", answer=str(i)) for i in range(5)]
    items = [(f"answer {i}", t) for i, t in enumerate(tasks)]
    results = verify_batch(items, max_workers=4)
    assert len(results) == 5
    assert all(r.correct for r in results)


def test_verify_batch_empty():
    assert verify_batch([]) == []


def test_filter_by_level():
    tasks = [
        Task(id="a", prompt="p", answer="1", level="hs"),
        Task(id="b", prompt="p", answer="1", level="university"),
        Task(id="c", prompt="p", answer="1", level=None),
    ]
    assert [t.id for t in filter_by_level(tasks, ["hs"])] == ["a"]


def test_split_by_year():
    tasks = [
        Task(id="old", prompt="p", answer="1", year=2022),
        Task(id="new", prompt="p", answer="1", year=2025),
        Task(id="undated", prompt="p", answer="1", year=None),
    ]
    train, hold = split_by_year(tasks, cutoff_year=2023)
    assert {t.id for t in train} == {"old", "undated"}  # undated → train, never eval
    assert {t.id for t in hold} == {"new"}


def test_assert_disjoint():
    train = [Task(id="a", prompt="p", answer="1"), Task(id="b", prompt="p", answer="1")]
    ok_eval = [Task(id="c", prompt="p", answer="1")]
    assert_disjoint(train, ok_eval)  # no raise
    with pytest.raises(ValueError, match="share 1 task"):
        assert_disjoint(train, [Task(id="a", prompt="p", answer="1")])
