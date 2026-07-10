"""C-CTX: the fork that picks the architecture (`lithos/evals/cctx.py`).

Two causes fit the literature, predict identical benchmark numbers, and imply opposite
architectures. These tests prove the *harness* can tell them apart — by constructing a
model that is capability-limited and one that is displacement-limited, and checking that
`diagnose` returns the right verdict for each.

The decision rule is pre-registered in code. If these tests pass, the rule cannot be
argued into shape once the real numbers arrive.
"""

from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
import torch
from lithos.evals.cctx import (
    Arm,
    diagnose,
    run_cctx,
    summarize,
    write_episodes,
)
from lithos.posttrain.chat_template import TIR_TOKENS
from lithos.posttrain.taskbank import Task
from lithos.retrieval import Datastore, DistractorRetriever, DocumentRetriever, HashingEmbedder

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith(("linux", "darwin")), reason="POSIX-only sandbox"
)

_SPECIALS = ["<pad>", "<bos>", "<eos>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
_ALL = _SPECIALS + list(TIR_TOKENS)


class RoundTripTok:
    def __init__(self):
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


@pytest.fixture
def tok():
    return RoundTripTok()


class AnswerAfterThinking(torch.nn.Module):
    """Emits ``think_tokens`` of filler, then an answer, then ``<|end|>``.

    Whether it *reaches* the answer depends entirely on how many tokens it is allowed to
    generate — which is exactly the resource retrieval spends. Starved, it emits only
    filler, the verifier finds no number, and the episode is wrong. That is displacement,
    mechanised.

    If the prompt carries the reference material it answers right; otherwise wrong. So:
    a **displacement-limited** model — it can use the fact, given room.

    Stateless by construction: the position in the script is derived from how many tokens
    follow the ``<|assistant|>`` header, so one instance serves every arm and budget
    without a reset anyone could forget to call.
    """

    def __init__(self, tok, *, think_tokens: int, right: str, wrong: str, cue: str):
        super().__init__()
        self.tok, self.vocab = tok, tok.vocab
        self.think = think_tokens
        self.cue, self.right, self.wrong = cue, right, wrong
        self._assistant = tok.token_to_id("<|assistant|>")
        self._end = tok.token_to_id("<|end|>")

    def _answer_for(self, prompt_text: str) -> str:
        return self.right if self.cue in prompt_text else self.wrong

    def _script(self, prompt_text: str) -> list[int]:
        filler = self.tok.encode("." * self.think).ids
        return [*filler, *self.tok.encode(self._answer_for(prompt_text)).ids, self._end]

    def forward(self, input_ids, kv_caches=None):
        ids = input_ids[0].tolist()
        header = max(i for i, x in enumerate(ids) if x == self._assistant)
        pos = len(ids) - 1 - header  # tokens emitted since the assistant turn opened
        prompt_text = self.tok.decode(ids[:header], skip_special_tokens=True)
        script = self._script(prompt_text)

        b, t = input_ids.shape
        logits = torch.full((b, t, self.vocab), -30.0)
        logits[:, -1, script[min(pos, len(script) - 1)]] = 30.0
        return logits, None


class IgnoresTheFact(AnswerAfterThinking):
    """Answers wrong however much room it has and however good the context.

    A **capability-limited** model: it had the fact and the room, and still could not use it.
    """

    def _answer_for(self, prompt_text: str) -> str:
        return self.wrong


def _store(tok, texts):
    emb = HashingEmbedder(dim=256)
    docs = [
        {
            "id": f"rec:{i}",
            "text": t,
            "source": f"src:{i}",
            "tier": "restricted",
            "metadata": {
                "source_id": f"src:{i}",
                "record_id": f"rec:{i}",
                "text_sha256": hashlib.sha256(t.encode()).hexdigest(),
            },
        }
        for i, t in enumerate(texts)
    ]
    return Datastore.build(docs, tok, emb, max_tokens=64, overlap_tokens=8), emb


CUE = "answer is"  # appears only in the reference block, never in a completion
CORPUS = [
    "the gravitational constant answer is 42 for this derivation",
    "mitochondria are organelles unrelated to the question at hand",
]
TASKS = [
    Task(id="t1", prompt="what is the gravitational constant answer", kind="numeric", answer="42")
]


def _run(model, tok, *, arms, budgets, distractor=False):
    store, emb = _store(tok, CORPUS)
    r = DocumentRetriever(store, emb, top_k=1)
    d = DistractorRetriever(store, emb, top_k=1) if distractor else None
    return run_cctx(
        model,
        tok,
        TASKS,
        weights_sha256="w" * 64,
        retriever=r,
        distractor_retriever=d,
        arms=arms,
        budgets=budgets,
        context_token_budget=128,
        max_tool_calls=0,
        use_cache=False,
        temperature=0.0,  # greedy: _sample_next short-circuits before top-p
    )


# ── the mechanic: one subtraction ─────────────────────────────────────────────


def test_oracle_gets_a_bigger_completion_budget_than_prepend(tok):
    """`prepend` charges the passage against the budget; `oracle` does not. Same prompt."""
    m = AnswerAfterThinking(tok, think_tokens=2, right="42", wrong="7", cue=CUE)
    recs = _run(m, tok, arms=(Arm.PREPEND, Arm.ORACLE), budgets=(256,))
    by_arm = {r.arm: r for r in recs}

    assert by_arm["prepend"].context_tokens > 0
    assert by_arm["prepend"].context_tokens == by_arm["oracle"].context_tokens
    assert by_arm["prepend"].prompt_tokens == by_arm["oracle"].prompt_tokens
    # ...and the ONLY difference is what the context was charged to.
    assert (
        by_arm["oracle"].completion_budget - by_arm["prepend"].completion_budget
        == by_arm["prepend"].context_tokens
    )


def test_none_arm_retrieves_nothing_and_is_unaffected_by_charging(tok):
    m = AnswerAfterThinking(tok, think_tokens=2, right="42", wrong="7", cue=CUE)
    recs = _run(m, tok, arms=(Arm.NONE,), budgets=(256,))
    assert recs[0].context_tokens == 0
    assert recs[0].cited_source_ids == ()
    assert recs[0].datastore_version is None


def test_a_prompt_that_fills_the_budget_starves_reasoning_rather_than_raising(tok):
    """Zero room to think is not an error — it is the displacement result, and must be seen."""
    m = AnswerAfterThinking(tok, think_tokens=2, right="42", wrong="7", cue=CUE)
    recs = _run(m, tok, arms=(Arm.PREPEND,), budgets=(8,))
    assert recs[0].completion_budget == 0
    assert recs[0].reasoning_tokens == 0
    assert recs[0].truncated


# ── the harness can tell the two causes apart ─────────────────────────────────


def test_a_displacement_limited_model_is_diagnosed_as_displacement(tok):
    """It can use the fact, but only if the passage isn't charged against its thinking room."""
    budgets = (96, 512)
    m = AnswerAfterThinking(tok, think_tokens=40, right="42", wrong="7", cue=CUE)
    recs = _run(m, tok, arms=(Arm.NONE, Arm.PREPEND, Arm.ORACLE), budgets=budgets)
    s = summarize(recs)
    d = diagnose(s, budgets=budgets)

    assert d.verdict == "displacement", d.rationale
    assert d.oracle_gain > 0  # a free fact helps
    assert d.displacement > 0  # charging it destroys the gain at the tight budget
    assert abs(d.converges) <= 0.05  # and the arms agree once there is room


def test_a_capability_limited_model_is_diagnosed_as_capability(tok):
    """It has the fact and the room, and still cannot use it. Retrieval never serves it."""
    budgets = (96, 512)
    m = IgnoresTheFact(tok, think_tokens=4, right="42", wrong="7", cue=CUE)
    recs = _run(m, tok, arms=(Arm.NONE, Arm.PREPEND, Arm.ORACLE), budgets=budgets)
    d = diagnose(summarize(recs), budgets=budgets)

    assert d.verdict == "capability", d.rationale
    assert d.oracle_gain <= 0.05


def test_missing_cells_are_inconclusive_not_guessed():
    d = diagnose({"none@64": {"accuracy": 1.0}}, budgets=(64, 512))
    assert d.verdict == "inconclusive" and "missing" in d.rationale


def test_a_pattern_fitting_neither_prediction_is_inconclusive():
    """Do not squint. If oracle helps but the arms never converge, say so."""
    s = {
        "none@64": {"accuracy": 0.0},
        "none@512": {"accuracy": 0.0},
        "prepend@64": {"accuracy": 0.0},
        "prepend@512": {"accuracy": 0.0},
        "oracle@64": {"accuracy": 1.0},
        "oracle@512": {"accuracy": 1.0},
    }
    assert diagnose(s, budgets=(64, 512)).verdict == "inconclusive"


# ── the distractor control ────────────────────────────────────────────────────


def test_distractor_returns_the_least_similar_passage(tok):
    store, emb = _store(tok, CORPUS)
    near = DocumentRetriever(store, emb, top_k=1).retrieve(
        "gravitational constant", token_budget=128
    )
    far = DistractorRetriever(store, emb, top_k=1).retrieve(
        "gravitational constant", token_budget=128
    )
    assert near.passages[0].source_id != far.passages[0].source_id
    assert far.passages[0].score < near.passages[0].score


def test_distractor_isolates_content_from_token_cost(tok):
    """`prepend - distractor` at the large budget: was it the content, or just the tokens?"""
    budgets = (96, 512)
    m = AnswerAfterThinking(tok, think_tokens=40, right="42", wrong="7", cue=CUE)
    recs = _run(
        m,
        tok,
        arms=(Arm.NONE, Arm.PREPEND, Arm.ORACLE, Arm.DISTRACTOR),
        budgets=budgets,
        distractor=True,
    )
    d = diagnose(summarize(recs), budgets=budgets)
    assert d.content_effect is not None
    assert d.content_effect > 0, (
        "the relevant passage must beat an irrelevant one of similar length"
    )


def test_distractor_arm_without_a_distractor_retriever_raises(tok):
    m = AnswerAfterThinking(tok, think_tokens=2, right="42", wrong="7", cue=CUE)
    with pytest.raises(ValueError, match="distractor arm needs"):
        _run(m, tok, arms=(Arm.DISTRACTOR,), budgets=(128,), distractor=False)


# ── records ───────────────────────────────────────────────────────────────────


def test_episodes_round_trip_to_jsonl(tmp_path, tok):
    m = AnswerAfterThinking(tok, think_tokens=2, right="42", wrong="7", cue=CUE)
    recs = _run(m, tok, arms=(Arm.NONE, Arm.ORACLE), budgets=(128,))
    p = write_episodes(recs, tmp_path / "episodes.jsonl")
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    assert len(rows) == len(recs)
    assert {r["arm"] for r in rows} == {"none", "oracle"}
    assert all("completion_budget" in r and "served_model_digest" in r for r in rows)


def test_summary_reports_starvation(tok):
    m = AnswerAfterThinking(tok, think_tokens=2, right="42", wrong="7", cue=CUE)
    recs = _run(m, tok, arms=(Arm.PREPEND,), budgets=(8,))
    s = summarize(recs)
    assert s["prepend@8"]["starved_frac"] == 1.0
