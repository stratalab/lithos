"""Tests for the TIR tool-uplift battery (lithos/evals/tir_battery.py, E8 Part A).

Three layers, matching docs/eval-tir-battery-plan.md §Verification:
1. ``verify_tir`` grades code by the executed tool code (not prose), answer-checked
   kinds by the completion.
2. ``paired_uplift`` stats: McNemar counts/p-value + clustered SE > naive.
3. End-to-end: a computation-heavy task solves WITH tools and fails WITHOUT, and the
   battery reports a positive uplift — the product thesis, quantified.

Reuses the scripted-model + round-trip-tokenizer harness pattern from
``test_tir_rollout`` (a real sandbox runs the decoded tool code).
"""

import sys
from types import SimpleNamespace

import pytest
import torch
from lithos.evals import tir_battery
from lithos.evals.tir_battery import (
    ArmOutcome,
    TaskOutcome,
    run_two_arm,
    sample_transcripts,
    summarize,
)
from lithos.evals.tir_stats import paired_uplift
from lithos.posttrain.chat_template import TIR_TOKENS, render_prompt, special_ids, tir_token_ids
from lithos.posttrain.taskbank import Task, verify_tir
from lithos.posttrain.tir_rollout import RolloutResult, tir_rollout

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith(("linux", "darwin")), reason="POSIX-only sandbox"
)

_SPECIALS = ["<pad>", "<bos>", "<eos>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
_ALL = _SPECIALS + list(TIR_TOKENS)


class RoundTripTok:
    """Fake tokenizer: specials/TIR at fixed low ids; each char maps to base+ord(c),
    so encode/decode round-trip (the sandbox runs the decoded code)."""

    def __init__(self) -> None:
        self._tok2id = {t: i for i, t in enumerate(_ALL)}
        self._id2tok = {i: t for t, i in self._tok2id.items()}
        self._base = len(_ALL)

    def token_to_id(self, token):
        return self._tok2id.get(token)

    def encode(self, text):
        return SimpleNamespace(ids=[self._base + ord(c) for c in text])

    def decode(self, ids, skip_special_tokens=True):
        out = []
        for i in ids:
            if i in self._id2tok:
                if not skip_special_tokens:
                    out.append(self._id2tok[i])
            else:
                out.append(chr(i - self._base))
        return "".join(out)

    @property
    def vocab(self):
        return self._base + 256


class ScriptedModel(torch.nn.Module):
    """Emits a fixed token sequence: each forward forces the next scripted token via a
    large logit at the last position. Use with use_cache=False (one call per token)."""

    def __init__(self, script, vocab) -> None:
        super().__init__()
        self.script = list(script)
        self.vocab = vocab
        self.calls = 0

    def forward(self, input_ids, kv_caches=None):
        b, t = input_ids.shape
        logits = torch.full((b, t, self.vocab), -30.0)
        nxt = self.script[min(self.calls, len(self.script) - 1)]
        logits[:, -1, nxt] = 30.0
        self.calls += 1
        return logits, None


def _ctx():
    tok = RoundTripTok()
    return tok, tir_token_ids(tok), special_ids(tok)


# --------------------------------------------------------------------------- #
# 1. verify_tir                                                               #
# --------------------------------------------------------------------------- #


def test_verify_tir_code_grades_by_tool_code():
    task = Task(id="c", prompt="write add", kind="code", tests="assert add(2, 3) == 5")
    good = [("python", "def add(a, b):\n    return a + b")]
    bad = [("python", "def add(a, b):\n    return a - b")]
    assert verify_tir("any prose", good, task).correct is True
    assert verify_tir("any prose", bad, task).correct is False
    # prose that merely *claims* the answer, with no executed code, must fail
    assert verify_tir("add(2, 3) == 5, obviously", [], task).correct is False
    # octave calls are not code-kind solutions
    assert verify_tir("x", [("octave", "disp(5)")], task).correct is False


def test_verify_tir_answer_kinds_use_completion():
    task = Task(id="n", prompt="2+2?", kind="numeric", answer="4")
    assert verify_tir("the answer is 4", [], task).correct is True
    assert verify_tir("the answer is 5", [], task).correct is False


# --------------------------------------------------------------------------- #
# 2. paired statistics                                                        #
# --------------------------------------------------------------------------- #


def test_paired_uplift_counts_and_pvalue():
    stat = paired_uplift([False, False, True], [True, True, True], [None, None, None])
    assert stat.mcnemar_on_gain == 2 and stat.mcnemar_on_loss == 0
    assert abs(stat.uplift - 2 / 3) < 1e-9
    assert stat.solve_off == pytest.approx(1 / 3) and stat.solve_on == 1.0
    # two-sided exact: b=2 gains of 2 discordant -> 2 * P(X<=0) = 2*(1/4) = 0.5
    assert stat.mcnemar_p == pytest.approx(0.5)


def test_paired_uplift_clustered_se_exceeds_naive():
    # 4 gains all in family "A", 4 ties as singletons: intra-cluster correlation makes
    # the clustered SE materially larger than the naive one (eval-plan §0.9).
    off = [False, False, False, False, True, True, True, True]
    on = [True, True, True, True, True, True, True, True]
    clusters = ["A", "A", "A", "A", None, None, None, None]
    stat = paired_uplift(off, on, clusters)
    assert stat.uplift == 0.5
    assert stat.se_clustered > stat.se_naive


def test_paired_uplift_empty():
    stat = paired_uplift([], [], [])
    assert stat.n == 0 and stat.uplift == 0.0 and stat.significant is False


# --------------------------------------------------------------------------- #
# 3. runner: orchestration + end-to-end uplift                                #
# --------------------------------------------------------------------------- #


def test_run_two_arm_off_disables_tools(monkeypatch):
    tok, tir, sids = _ctx()
    seen = []

    def fake_rollout(model, prompt_ids, tok_, tir_, sids_, *, max_tool_calls, **kw):
        seen.append(max_tool_calls)
        if max_tool_calls == 0:
            return RolloutResult(
                token_ids=[1], action_mask=[False], completion_text="i cannot compute",
                num_tool_calls=0, tool_calls=[], truncated=True, tool_outputs=[],
            )
        return RolloutResult(
            token_ids=[1], action_mask=[True], completion_text="the result is 4",
            num_tool_calls=1, tool_calls=[("python", "print(2 + 2)")],
            truncated=False, tool_outputs=["4"],
        )

    monkeypatch.setattr(tir_battery, "tir_rollout", fake_rollout)
    task = Task(id="t", prompt="2+2?", kind="numeric", answer="4")
    outcome = run_two_arm(object(), task, tok, tir, sids, max_tool_calls=4)

    assert seen == [0, 4]  # off arm first (tools disabled), then on arm
    assert outcome.off.correct is False and outcome.off.num_tool_calls == 0
    assert outcome.on.correct is True and outcome.on.num_tool_calls == 1


def test_end_to_end_positive_uplift_and_summary():
    tok, tir, sids = _ctx()
    py, close, end = tir["<|python|>"], tir["<|/tool|>"], sids["<|end|>"]
    task = Task(id="sq", prompt="what is 42 squared?", kind="numeric", answer="1764", level="hard")
    # tool call computes 42**2, then the model states the (tool-provided) answer
    script = [
        py, *tok.encode("print(42**2)").ids, close,
        *tok.encode("So it is 1764.").ids, end,
    ]
    prompt = render_prompt([{"role": "user", "content": task.prompt}], tok)

    def _roll(max_tool_calls):
        return tir_rollout(
            ScriptedModel(script, tok.vocab), prompt, tok, tir, sids, device="cpu",
            use_cache=False, temperature=0.0, max_new=200, max_tool_calls=max_tool_calls,
        )

    on, off = _roll(4), _roll(0)
    assert on.num_tool_calls == 1 and off.num_tool_calls == 0
    assert off.truncated  # off arm hit the tool call but the sandbox was disabled

    on_res = verify_tir(on.completion_text, on.tool_calls, task)
    off_res = verify_tir(off.completion_text, off.tool_calls, task)
    assert on_res.correct is True and off_res.correct is False  # the uplift, proven

    outcome = TaskOutcome(
        task_id=task.id, level=task.level, family_id=None,
        off=ArmOutcome(off_res.correct, off.num_tool_calls, off.num_malformed_calls, off.truncated),
        on=ArmOutcome(on_res.correct, on.num_tool_calls, on.num_malformed_calls, on.truncated),
        on_tool_calls=on.tool_calls, on_tool_outputs=on.tool_outputs, on_completion=on.completion_text,
    )
    summary = summarize([outcome])
    assert summary["overall"]["uplift"] == 1.0
    assert summary["overall"]["solve_on"] == 1.0 and summary["overall"]["solve_off"] == 0.0
    assert summary["per_tier"]["hard"]["n"] == 1
    assert summary["health"]["tool_call_rate"] == 1.0

    ts = sample_transcripts([outcome], 5)
    assert ts[0]["on_correct"] is True and ts[0]["off_correct"] is False
    assert ts[0]["tool_calls"] == [["python", "print(42**2)"]]
