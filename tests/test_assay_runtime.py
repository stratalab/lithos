"""Tests for the assay TIR runtime (reserved ID 13, docs/tir-format.md §2).

Locks the four seams the new runtime touches: the tokenizer's reserved-block ID
assignment (lockstep with the renderer), the shared ingestion gate, the sandbox
executor shim, and the rollout parse. The shim tests EXECUTE real sympy in the
subprocess sandbox — this is the IR path running end to end.
"""

import json
import sys
from types import SimpleNamespace

import pytest
from lithos.posttrain.chat_template import (
    REQUIRED_SPECIAL_TOKENS,
    TIR_TOKENS,
    TOOL_OPEN,
    render_conversation,
    special_ids,
    tir_token_ids,
)
from lithos.posttrain.sandbox import run_assay, run_tool
from lithos.posttrain.tir_rollout import parse_tool_call
from lithos.posttrain.tir_validate import TOOL_RUNTIMES, validate_tir_record
from lithos.tokenizer.tokenizer_config import (
    DEFAULT_SPECIAL_TOKENS,
    STEM_SPECIAL_TOKENS,
    TIR_SPECIAL_TOKENS,
)

_POSIX = sys.platform.startswith(("linux", "darwin"))
sandbox = pytest.mark.skipif(not _POSIX, reason="POSIX-only sandbox")


# ---- the ID contract: tokenizer block == renderer names == tir-format §2 pins ----


def test_stem_specials_pin_the_documented_ids():
    # order IS the ID; these pins are docs/tir-format.md §2 verbatim
    assert STEM_SPECIAL_TOKENS.index("<|end|>") == 6
    assert STEM_SPECIAL_TOKENS.index("<think>") == 7
    assert STEM_SPECIAL_TOKENS.index("<|python|>") == 9
    assert STEM_SPECIAL_TOKENS.index("<|octave|>") == 10
    assert STEM_SPECIAL_TOKENS.index("<|/tool|>") == 11
    assert STEM_SPECIAL_TOKENS.index("<|tool_result|>") == 12
    assert STEM_SPECIAL_TOKENS.index("<|assay|>") == 13
    assert len(STEM_SPECIAL_TOKENS) == 16  # 0-6 chat + 7-15 reserved block


def test_tokenizer_block_covers_renderer_requirements():
    # every token the renderer can emit must exist in the STEM tokenizer's specials
    assert set(REQUIRED_SPECIAL_TOKENS) <= set(STEM_SPECIAL_TOKENS)
    # and the reserved block is exactly chat-block-disjoint
    assert not set(TIR_SPECIAL_TOKENS) & set(DEFAULT_SPECIAL_TOKENS)


def test_assay_is_a_runtime_everywhere():
    assert "assay" in TOOL_OPEN and TOOL_OPEN["assay"] == "<|assay|>"
    assert "assay" in TOOL_RUNTIMES
    assert "<|assay|>" in TIR_TOKENS


# ---- ingestion gate ----


IR = json.dumps(
    {"task": "differentiate.univariate",
     "inputs": {"expression": "sin(x)**3", "variable": "x"}, "missing_inputs": []}
)


def _assay_record(ir: str = IR) -> dict:
    return {
        "messages": [
            {"role": "user", "content": "Differentiate sin(x)^3."},
            {"role": "assistant", "segments": [
                {"type": "think", "text": "A templated task; route to assay."},
                {"type": "tool", "runtime": "assay", "code": ir},
                {"type": "tool_result", "output": '{"derivative": "3*sin(x)**2*cos(x)"}'},
                {"type": "text", "text": "The derivative is 3 sin^2(x) cos(x)."},
            ]},
        ]
    }


def test_validate_accepts_assay_runtime():
    validate_tir_record(_assay_record())  # must not raise


def test_validate_still_rejects_unknown_runtime():
    rec = _assay_record()
    rec["messages"][1]["segments"][1]["runtime"] = "wolfram"
    with pytest.raises(ValueError, match="unknown tool runtime"):
        validate_tir_record(rec)


# ---- render + rollout parse (fake tokenizer, same pattern as test_tir_rollout) ----


_ALL = [*DEFAULT_SPECIAL_TOKENS, *TIR_TOKENS]


class _Tok:
    def __init__(self):
        self._tok2id = {t: i for i, t in enumerate(_ALL)}
        self._base = len(_ALL)

    def token_to_id(self, token):
        return self._tok2id.get(token)

    def encode(self, text):
        return SimpleNamespace(ids=[self._base + ord(c) for c in text])

    def decode(self, ids, skip_special_tokens=True):
        inv = {i: t for t, i in self._tok2id.items()}
        return "".join(chr(i - self._base) for i in ids if i not in inv)


def test_render_assay_segment_masks_result_only():
    tok = _Tok()
    r = render_conversation(_assay_record()["messages"], tok)
    tir, sids = tir_token_ids(tok), special_ids(tok)
    open_pos = r.input_ids.index(tir["<|assay|>"])
    close_pos = r.input_ids.index(tir["<|/tool|>"])
    assert all(w == 1.0 for w in r.weights[open_pos : close_pos + 1])  # IR is learned
    res_pos = r.input_ids.index(tir["<|tool_result|>"])
    res_end = r.input_ids.index(sids["<|end|>"], res_pos)
    assert all(w == 0.0 for w in r.weights[res_pos : res_end + 1])  # result masked


def test_parse_tool_call_extracts_assay_ir():
    tok = _Tok()
    tir = tir_token_ids(tok)
    seg = [tir["<|assay|>"], *tok.encode(IR).ids, tir["<|/tool|>"]]
    runtime, code = parse_tool_call(seg, tir, tok)
    assert runtime == "assay"
    assert json.loads(code)["task"] == "differentiate.univariate"


# ---- the executor shim, actually executing ----


@sandbox
def test_run_assay_differentiates():
    res = run_tool("assay", IR)
    assert res.ok, res.output
    out = json.loads(res.output)
    assert out == {"derivative": "3*sin(x)**2*cos(x)"}


@sandbox
def test_run_assay_solves_and_limits():
    roots = run_assay(json.dumps({
        "task": "solve_equation.univariate",
        "inputs": {"expression": "x**2 + x - 6", "variable": "x"}}))
    assert roots.ok and set(json.loads(roots.output)["roots"]) == {"-3", "2"}
    lim = run_assay(json.dumps({
        "task": "limit.of_function",
        "inputs": {"expression": "5 - 2/x**2", "variable": "x", "point": "oo"}}))
    assert lim.ok and json.loads(lim.output)["limit_value"] == "5"


def test_run_assay_refuses_missing_inputs():
    res = run_assay(json.dumps({
        "task": "differentiate.univariate",
        "inputs": {"expression": "x", "variable": "x"},
        "missing_inputs": ["elastic_modulus"]}))
    assert not res.ok and "never invent" in res.output


def test_run_assay_refuses_unknown_template_with_fallback_hint():
    res = run_assay(json.dumps({"task": "beam_deflection", "inputs": {"expression": "x", "variable": "x"}}))
    assert not res.ok
    assert "not in the v0 registry" in res.output and "python runtime" in res.output


def test_run_assay_refuses_bad_json():
    res = run_assay("{not json")
    assert not res.ok and "not valid JSON" in res.output
