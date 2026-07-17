"""Tests for the multi-segment TIR rollout (lithos/posttrain/tir_rollout.py, E4).

A deterministic scripted "model" (forces a fixed token sequence) + a round-trip
fake tokenizer drive the rollout through a REAL sandbox execution, so the tool
call → execute → inject → resume loop and the action mask are verified end to end.
"""

import sys
from types import SimpleNamespace

import pytest
import torch
from lithos.posttrain.chat_template import TIR_TOKENS, special_ids, tir_token_ids
from lithos.posttrain.tir_rollout import parse_tool_call, tir_rollout

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith(("linux", "darwin")), reason="POSIX-only sandbox"
)

_SPECIALS = ["<pad>", "<bos>", "<eos>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
_ALL = _SPECIALS + list(TIR_TOKENS)


class RoundTripTok:
    """Fake tokenizer: specials/TIR at fixed low ids; each char maps to base+ord(c),
    so encode/decode round-trip (the sandbox runs the decoded code)."""

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


class ScriptedModel(torch.nn.Module):
    """Emits a fixed token sequence: each forward forces the next scripted token via
    a large logit at the last position. Use with use_cache=False (one call per token)."""

    def __init__(self, script, vocab):
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


def _ids(tok, tir, sids):
    return {
        "py": tir["<|python|>"], "oct": tir["<|octave|>"], "close": tir["<|/tool|>"],
        "result": tir["<|tool_result|>"], "end": sids["<|end|>"],
    }


def _rollout(script, tok, tir, sids, **kw):
    model = ScriptedModel(script, tok.vocab)
    prompt = [sids["<|user|>"], *tok.encode("go").ids, sids["<|end|>"], sids["<|assistant|>"]]
    return tir_rollout(
        model, prompt, tok, tir, sids, device="cpu", use_cache=False,
        max_new=kw.pop("max_new", 200), max_tool_calls=kw.pop("max_tool_calls", 4), **kw,
    ), len(prompt)


def test_parse_tool_call():
    tok, tir, _ = _ctx()
    seg = [tir["<|python|>"], *tok.encode("print(1)").ids, tir["<|/tool|>"]]
    rt, code = parse_tool_call(seg, tir, tok)
    assert rt == "python" and code == "print(1)"
    # no open tag before the close -> malformed
    assert parse_tool_call([*tok.encode("x").ids, tir["<|/tool|>"]], tir, tok) is None
    # not ending in close -> None
    assert parse_tool_call([tir["<|python|>"], *tok.encode("x").ids], tir, tok) is None


def test_rollout_executes_tool_and_masks_result():
    tok, tir, sids = _ctx()
    j = _ids(tok, tir, sids)
    script = [
        j["py"], *tok.encode("print(2+2)").ids, j["close"],   # tool call segment
        *tok.encode("The answer is 4.").ids, j["end"],        # answer segment (after injection)
    ]
    roll, plen = _rollout(script, tok, tir, sids)

    assert roll.num_tool_calls == 1
    assert roll.tool_calls == [("python", "print(2+2)")]
    assert not roll.truncated
    assert "4" in roll.completion_text  # the sandbox actually ran print(2+2)
    assert len(roll.token_ids) == len(roll.action_mask)

    # the injected tool-result span (<|tool_result|> ... <|end|>) is entirely masked
    r_open = roll.token_ids.index(j["result"])
    r_end = roll.token_ids.index(j["end"], r_open)  # the <|end|> that closes the result
    assert not any(roll.action_mask[r_open : r_end + 1])
    # the decoded result span carries the tool output
    assert "4" in tok.decode(roll.token_ids[r_open : r_end + 1])
    # the model-generated tool call IS a policy action (learned)
    call_open = roll.token_ids.index(j["py"])
    assert all(roll.action_mask[call_open : roll.token_ids.index(j["close"]) + 1])
    # prompt is masked
    assert not any(roll.action_mask[:plen])


def test_rollout_no_tool_plain_answer():
    tok, tir, sids = _ctx()
    j = _ids(tok, tir, sids)
    roll, plen = _rollout([*tok.encode("just 7").ids, j["end"]], tok, tir, sids)
    assert roll.num_tool_calls == 0
    assert not roll.truncated
    assert all(roll.action_mask[plen:])  # whole completion is a policy action


def test_rollout_malformed_tool_call_injects_error():
    tok, tir, sids = _ctx()
    j = _ids(tok, tir, sids)
    # <|/tool|> with no <|python|>/<|octave|> before it
    script = [*tok.encode("oops").ids, j["close"], *tok.encode("done").ids, j["end"]]
    roll, _ = _rollout(script, tok, tir, sids)
    assert roll.num_tool_calls == 0  # nothing parsed/executed
    assert j["result"] in roll.token_ids  # but an error result was still injected
    assert "error" in roll.completion_text.lower()


def test_rollout_respects_max_tool_calls():
    tok, tir, sids = _ctx()
    j = _ids(tok, tir, sids)
    one_call = [j["py"], *tok.encode("print(1)").ids, j["close"]]
    roll, _ = _rollout(one_call * 5, tok, tir, sids, max_tool_calls=2)
    assert roll.num_tool_calls == 2  # capped
    assert roll.truncated  # never reached a final <|end|>


def test_rollout_truncates_without_stop():
    tok, tir, sids = _ctx()
    # script emits plain tokens, never a stop token; max_new caps it
    roll, plen = _rollout([*tok.encode("aaaaaaaa").ids], tok, tir, sids, max_new=5)
    assert roll.truncated
    assert len(roll.token_ids) - plen <= 5


def test_rollout_logprobs_align_with_actions():
    # T2: the episode carries the sampler's per-token logprob, strictly parallel to
    # token_ids, and exactly 0.0 wherever the token was not the policy's move
    # (prompt + injected tool results).
    tok, tir, sids = _ctx()
    j = _ids(tok, tir, sids)
    script = [
        j["py"], *tok.encode("print(2+2)").ids, j["close"],
        *tok.encode("The answer is 4.").ids, j["end"],
    ]
    roll, _ = _rollout(script, tok, tir, sids)
    assert len(roll.logprobs) == len(roll.token_ids)
    for lp, is_action in zip(roll.logprobs, roll.action_mask, strict=True):
        if not is_action:
            assert lp == 0.0
        else:
            assert lp <= 0.0  # a real log-probability


def test_rollout_to_record():
    # T1: the episode lifts into the canonical record — weights mirror the action
    # mask, the scalar advantage broadcasts over action positions only, and the
    # injected tool-result span drops out of labels().
    from lithos.posttrain.record import IGNORE_INDEX

    tok, tir, sids = _ctx()
    j = _ids(tok, tir, sids)
    script = [
        j["py"], *tok.encode("print(2+2)").ids, j["close"],
        *tok.encode("ok").ids, j["end"],
    ]
    roll, _ = _rollout(script, tok, tir, sids)
    rec = roll.to_record(advantage=0.5)
    assert rec.tokens == roll.token_ids
    assert rec.weights == [1.0 if a else 0.0 for a in roll.action_mask]
    assert rec.advantages == [0.5 if a else 0.0 for a in roll.action_mask]
    assert rec.logprobs == roll.logprobs
    # the injected tool-result span produces only IGNORE_INDEX labels
    r_open = roll.token_ids.index(j["result"])
    r_end = roll.token_ids.index(j["end"], r_open)
    labels = rec.labels()
    assert all(labels[i - 1] == IGNORE_INDEX for i in range(r_open, r_end + 1))
