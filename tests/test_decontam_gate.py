"""Tests for the post-training decontamination gate (lithos/posttrain/decontam_gate.py, F2)."""

from lithos.data.decontam import write_probes
from lithos.posttrain.decontam_gate import (
    PostTrainDecontaminator,
    messages_text,
    prefs_text,
)

# A benchmark "probe" whose 13-gram signature we want to keep out of training data.
PROBE = (
    "A train leaves Chicago traveling west at sixty miles per hour while a second "
    "train departs from Denver heading east at forty miles per hour on the same track"
)


def _sft(text_user: str, text_assistant: str) -> dict:
    return {"messages": [{"role": "user", "content": text_user}, {"role": "assistant", "content": text_assistant}]}


def test_drops_contaminated_keeps_clean():
    gate = PostTrainDecontaminator([PROBE], n=13)
    records = [
        _sft("What is 2 + 2?", "4"),  # clean
        _sft(PROBE + " — when do they meet?", "In two hours."),  # leaked prompt
        _sft("Define entropy.", "A measure of disorder."),  # clean
    ]
    kept = gate.screen(records, messages_text)
    assert len(kept) == 2
    assert all("train leaves Chicago" not in messages_text(r) for r in kept)
    rep = gate.report()
    assert rep["read"] == 3 and rep["dropped"] == 1 and rep["kept"] == 2
    assert rep["drop_rate"] == round(1 / 3, 4)


def test_screens_assistant_side_leak():
    gate = PostTrainDecontaminator([PROBE], n=13)
    leaked_answer = _sft("Solve this classic puzzle.", PROBE + " so the answer follows.")
    assert gate.screen([leaked_answer], messages_text) == []


def test_prefs_text_extractor():
    rec = {
        "prompt": [{"role": "user", "content": "hello"}],
        "chosen": "hi there",
        "rejected": "go away",
    }
    text = prefs_text(rec)
    assert "hello" in text and "hi there" in text and "go away" in text


def test_from_probe_file(tmp_path):
    path = tmp_path / "probes.jsonl"
    write_probes(path, [PROBE])
    gate = PostTrainDecontaminator.from_probe_file(path, n=13)
    kept = gate.screen([_sft(PROBE, "leaked")], messages_text)
    assert kept == []


def test_clean_dataset_untouched():
    gate = PostTrainDecontaminator([PROBE], n=13)
    records = [_sft(f"Question {i}", f"Answer {i}") for i in range(10)]
    kept = gate.screen(records, messages_text)
    assert len(kept) == 10
    assert gate.report()["dropped"] == 0
