"""Tests for verifier-labeled DPO preferences (lithos/posttrain/verifier_prefs.py, E8)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from lithos.posttrain.taskbank import Task
from lithos.posttrain.verifier_prefs import build_verifier_prefs, make_pairs

_PROMPT = [{"role": "user", "content": "solve it"}]
TOKENIZER = "artifacts/tokenizer/fineweb-edu-32k/tokenizer.json"
_has_tok = Path(TOKENIZER).exists()


# ---- make_pairs ----


def test_pairs_correct_vs_incorrect():
    pairs = make_pairs(_PROMPT, [("right", True), ("wrong", False)])
    assert pairs == [{"prompt": _PROMPT, "chosen": "right", "rejected": "wrong"}]


def test_no_pair_when_all_correct_or_all_incorrect():
    assert make_pairs(_PROMPT, [("a", True), ("b", True)]) == []
    assert make_pairs(_PROMPT, [("a", False), ("b", False)]) == []
    assert make_pairs(_PROMPT, []) == []


def test_dedup_and_max_pairs():
    labeled = [("ok", True), ("ok", True), ("ok2", True), ("bad", False), ("bad2", False)]
    pairs = make_pairs(_PROMPT, labeled, max_pairs=5)
    # "ok" de-duped to one correct; two distinct correct (ok, ok2) x two incorrect -> 2 pairs
    assert len(pairs) == 2
    assert {p["chosen"] for p in pairs} == {"ok", "ok2"}
    assert all(p["rejected"] in {"bad", "bad2"} for p in pairs)
    # cap respected
    assert len(make_pairs(_PROMPT, labeled, max_pairs=1)) == 1


def test_blank_completions_ignored():
    pairs = make_pairs(_PROMPT, [("  ", True), ("real", True), ("nope", False)])
    assert len(pairs) == 1 and pairs[0]["chosen"] == "real"


# ---- build_verifier_prefs ----


def _tasks():
    return [
        Task(id="solvable", prompt="1+1?", answer="2"),  # model gets a mix -> a pair
        Task(id="too-hard", prompt="hard?", answer="42"),  # always wrong -> skipped
    ]


def test_build_prefs_orchestration():
    def sample(task):
        if task.id == "solvable":
            return ["the answer is 2", "the answer is 3"]  # one right, one wrong
        return ["dunno", "no idea"]  # all wrong

    def is_correct(text, task):
        return task.answer in text

    prefs, stats = build_verifier_prefs(_tasks(), sample, is_correct, samples_per_task=2)
    assert stats["tasks"] == 2
    assert stats["pairs"] == 1
    assert stats["skipped_no_mix"] == 1
    assert prefs[0]["chosen"] == "the answer is 2"
    assert prefs[0]["rejected"] == "the answer is 3"
    assert prefs[0]["prompt"] == [{"role": "user", "content": "1+1?"}]


def test_build_prefs_respects_sample_cap():
    def sample(task):
        return ["2 yes", "2 yes also", "3 no", "wrong 4"]

    def is_correct(text, task):
        return "2" in text

    _, stats = build_verifier_prefs(
        [Task(id="t", prompt="p", answer="2")], sample, is_correct,
        samples_per_task=2, max_pairs_per_task=3,
    )
    # only the first 2 completions are used -> "2 yes"(correct) + "2 yes also"(correct) => all correct => no pair
    assert stats["pairs"] == 0 and stats["skipped_no_mix"] == 1


# ---- the generated prefs feed the unchanged DPO dataset ----

_NAMES = ["<pad>", "<bos>", "<eos>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]


class FakeTok:
    def __init__(self):
        self._ids = {n: i for i, n in enumerate(_NAMES)}

    def token_to_id(self, token):
        return self._ids.get(token)

    def encode(self, text):
        return SimpleNamespace(ids=[100 + (ord(c) % 50) for c in text])


def test_generated_prefs_load_into_preference_dataset(tmp_path):
    from lithos.posttrain.preference_dataset import PreferenceDataset

    def sample(task):
        return ["answer 2 correct", "answer 9 wrong"]

    def is_correct(text, task):
        return "2" in text

    prefs, _ = build_verifier_prefs(
        [Task(id="t", prompt="what is 1+1", answer="2")], sample, is_correct, samples_per_task=2
    )
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(p) for p in prefs) + "\n")

    ds = PreferenceDataset(path, FakeTok(), seq_len=64)  # the unchanged DPO consumer
    assert ds.stats()["pairs"] == 1
    cx, _, _, ry = ds[0]
    assert cx.shape == (64,) and ry.shape == (64,)  # renders to the (x, y) DPO contract


@pytest.mark.skipif(not _has_tok, reason="tokenizer artifact absent")
def test_train_dpo_runs_on_verifier_prefs(tmp_path):
    # The E8 "Done" claim end-to-end: a real DPO step consumes generated prefs.
    # (Also the first end-to-end coverage of train_dpo itself.)
    from lithos.model.config import ModelConfig
    from lithos.posttrain.dpo_trainer import train_dpo
    from lithos.train.config import DataConfig, OptimConfig, ScheduleConfig, TrainConfig

    tasks = [Task(id=str(i), prompt=f"compute q{i}", answer=str(i)) for i in range(6)]

    def sample(task):
        return [f"the result is {task.answer} exactly", "totally wrong, no clue"]

    def is_correct(text, task):
        return task.answer in text  # "totally wrong..." has no digits -> always incorrect

    prefs, _ = build_verifier_prefs(tasks, sample, is_correct, samples_per_task=2)
    assert len(prefs) == 6  # every task yields a correct∧incorrect pair
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(p) for p in prefs) + "\n")

    cfg = TrainConfig(
        run_name="dpo-verifier",
        runs_dir=str(tmp_path / "runs"),
        device="cpu",
        precision="fp32",
        micro_batch_size=2,
        log_interval=1,
        dpo_beta=0.5,
        model=ModelConfig(vocab_size=32000, n_layers=2, hidden=32, n_heads=2, seq_len=64),
        data=DataConfig(kind="dpo", corpus_manifest=str(path), seq_len=64, tokenizer_path=TOKENIZER),
        optim=OptimConfig(lr=1e-4),
        schedule=ScheduleConfig(warmup_steps=1, max_steps=1),
    )
    run = train_dpo(cfg)
    assert run is not None
    recs = [json.loads(line) for line in run.metrics.read_text().splitlines()]
    assert any("loss" in r and "reward_accuracy" in r for r in recs)
