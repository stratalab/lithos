"""Tests for TIR episode rendering + masking (chat_template.py, E3).

Verifies the docs/tir-format.md §4 masking table token-for-token: think/tool/text
segments are learned; the tool_result span (incl. its closing <|end|>) is masked;
the turn-closing <|end|> is learned. Uses a FakeTok carrying the TIR tokens (the id
values are irrelevant — rendering resolves by name).
"""

from types import SimpleNamespace

import pytest
from lithos.posttrain.chat_template import (
    TIR_TOKENS,
    render_conversation,
    render_prompt,
    tir_token_ids,
)

_CORE = ["<pad>", "<bos>", "<eos>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
_ALL = _CORE + list(TIR_TOKENS)  # core 0-6, TIR 7-12


class TirTok:
    """FakeTok with the core specials and (optionally) the TIR tokens."""

    def __init__(self, *, tir: bool = True):
        names = _ALL if tir else _CORE
        self._ids = {n: i for i, n in enumerate(names)}

    def token_to_id(self, token):
        return self._ids.get(token)

    def encode(self, text):
        return SimpleNamespace(ids=[100 + (ord(c) % 50) for c in text])


def _id(name):
    return _ALL.index(name)


def _enc(s):
    return [100 + (ord(c) % 50) for c in s]


def _episode(*segments):
    return [{"role": "user", "content": "Q"}, {"role": "assistant", "segments": list(segments)}]


def test_tir_episode_segment_masking():
    tok = TirTok()
    r = render_conversation(
        _episode(
            {"type": "think", "text": "r"},
            {"type": "tool", "runtime": "python", "code": "c"},
            {"type": "tool_result", "output": "o"},
            {"type": "text", "text": "a"},
        ),
        tok,
    )
    assert r.input_ids == [
        _id("<bos>"),
        _id("<|user|>"), *_enc("Q"), _id("<|end|>"),
        _id("<|assistant|>"),
        _id("<think>"), *_enc("r"), _id("</think>"),
        _id("<|python|>"), *_enc("c"), _id("<|/tool|>"),
        _id("<|tool_result|>"), *_enc("o"), _id("<|end|>"),
        *_enc("a"),
        _id("<|end|>"),
    ]
    assert r.loss_mask == [
        False,                                    # bos
        False, *[False] * len(_enc("Q")), False,  # user turn
        False,                                    # assistant header
        True, *[True] * len(_enc("r")), True,     # <think> r </think>  — learned
        True, *[True] * len(_enc("c")), True,     # <|python|> c <|/tool|> — learned
        False, *[False] * len(_enc("o")), False,  # <|tool_result|> o <|end|> — MASKED
        *[True] * len(_enc("a")),                 # answer — learned
        True,                                     # turn-closing <|end|> — learned
    ]


def test_multi_tool_episode_masks_each_result():
    tok = TirTok()
    r = render_conversation(
        _episode(
            {"type": "tool", "runtime": "python", "code": "c1"},
            {"type": "tool_result", "output": "o1"},
            {"type": "tool", "runtime": "octave", "code": "c2"},
            {"type": "tool_result", "output": "o2"},
            {"type": "text", "text": "done"},
        ),
        tok,
    )
    learned = [tid for tid, m in zip(r.input_ids, r.loss_mask, strict=True) if m]
    # both tool calls + the final answer are learned; neither result is
    assert learned == [
        _id("<|python|>"), *_enc("c1"), _id("<|/tool|>"),
        _id("<|octave|>"), *_enc("c2"), _id("<|/tool|>"),
        *_enc("done"),
        _id("<|end|>"),
    ]
    # the tool_result marker never appears among learned tokens
    assert _id("<|tool_result|>") not in learned


def test_tool_result_output_never_a_learned_target():
    # Even though masking is positional, no <|tool_result|> or <|/tool|>-adjacent
    # result token should carry loss — assert the whole result span is masked.
    tok = TirTok()
    r = render_conversation(
        _episode(
            {"type": "tool", "runtime": "python", "code": "x"},
            {"type": "tool_result", "output": "SECRET"},
            {"type": "text", "text": "y"},
        ),
        tok,
    )
    ids, mask = r.input_ids, r.loss_mask
    start = ids.index(_id("<|tool_result|>"))
    end = start + len(_enc("SECRET")) + 2  # marker + output + closing <|end|>
    assert not any(mask[start:end])  # entire result span masked


def test_render_prompt_includes_prior_tir_turn_then_opens_assistant():
    tok = TirTok()
    msgs = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "segments": [{"type": "text", "text": "a"}]},
        {"role": "user", "content": "more"},
    ]
    ids = render_prompt(msgs, tok)
    assert ids[-1] == _id("<|assistant|>")  # ready for generation
    assert _id("<|assistant|>") in ids[:-1]  # the prior assistant turn rendered too


def test_missing_tir_tokens_errors_loudly():
    plain = TirTok(tir=False)  # only the 7 core specials, no TIR vocab
    with pytest.raises(ValueError, match="missing TIR tokens"):
        render_conversation(_episode({"type": "think", "text": "r"}), plain)
    with pytest.raises(ValueError, match="missing TIR tokens"):
        tir_token_ids(plain)


def test_unknown_segment_type_rejected():
    tok = TirTok()
    with pytest.raises(ValueError, match="unknown segment type"):
        render_conversation(_episode({"type": "bogus", "text": "x"}), tok)


def test_unknown_tool_runtime_rejected():
    tok = TirTok()
    with pytest.raises(ValueError, match="unknown tool runtime"):
        render_conversation(_episode({"type": "tool", "runtime": "ruby", "code": "x"}), tok)


@pytest.mark.parametrize(
    "seg, match",
    [
        ({"type": "think"}, "missing required field 'text'"),
        ({"type": "text"}, "missing required field 'text'"),
        ({"type": "tool", "runtime": "python"}, "missing required field 'code'"),
        ({"type": "tool_result"}, "missing required field 'output'"),
    ],
)
def test_missing_segment_field_errors_clearly(seg, match):
    with pytest.raises(ValueError, match=match):
        render_conversation(_episode(seg), TirTok())


def test_malformed_segments_containers_rejected():
    tok = TirTok()
    with pytest.raises(ValueError, match="'segments' must be a list"):
        render_conversation([{"role": "assistant", "segments": "think"}], tok)
    with pytest.raises(ValueError, match="segment 0 must be a dict"):
        render_conversation(_episode("think"), tok)


def test_segments_only_on_assistant():
    tok = TirTok()
    with pytest.raises(ValueError, match="segments are assistant-only"):
        render_conversation([{"role": "user", "segments": [{"type": "text", "text": "x"}]}], tok)


def test_content_and_segments_are_mutually_exclusive():
    tok = TirTok()
    with pytest.raises(ValueError, match="both 'content' and 'segments'"):
        render_conversation(
            [{"role": "assistant", "content": "c", "segments": [{"type": "text", "text": "s"}]}], tok
        )


def test_turn_missing_content_errors_clearly():
    tok = TirTok()
    with pytest.raises(ValueError, match="missing 'content'"):
        render_conversation([{"role": "assistant"}], tok)


def test_non_string_segment_field_errors_clearly():
    tok = TirTok()
    with pytest.raises(ValueError, match="must be a string, got int"):
        render_conversation(_episode({"type": "text", "text": 123}), tok)


def test_decontam_survives_malformed_segment():
    # decontam runs before the renderer validates — a messy record must not crash it
    from lithos.posttrain.decontam_gate import PostTrainDecontaminator, messages_text

    gate = PostTrainDecontaminator(["some benign probe text here"], n=13)
    messy = {"messages": [{"role": "assistant", "segments": [
        "not a dict", {"type": "text", "text": 123}, {"type": "text", "text": "ok"},
    ]}]}
    assert gate.screen([messy], messages_text) == [messy]  # kept (no leak), no crash


def test_flat_content_and_segments_coexist():
    # a flat assistant turn (no TIR tokens needed) then a segments turn
    tok = TirTok()
    r = render_conversation(
        [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "segments": [{"type": "think", "text": "r"}, {"type": "text", "text": "d"}]},
        ],
        tok,
    )
    learned = [tid for tid, m in zip(r.input_ids, r.loss_mask, strict=True) if m]
    assert learned == [
        *_enc("b"), _id("<|end|>"),                               # flat turn
        _id("<think>"), *_enc("r"), _id("</think>"), *_enc("d"), _id("<|end|>"),  # TIR turn
    ]


def test_tir_flows_through_build_xy():
    from lithos.posttrain.sft_dataset import IGNORE_INDEX, build_xy

    tok = TirTok()
    msgs = _episode(
        {"type": "think", "text": "r"},
        {"type": "tool", "runtime": "python", "code": "c"},
        {"type": "tool_result", "output": "leaked"},
        {"type": "text", "text": "a"},
    )
    pair = build_xy(msgs, tok, seq_len=128, pad_id=_id("<pad>"))
    assert pair is not None
    _, y = pair
    r = render_conversation(msgs, tok)
    # build_xy target[i] = ids[i+1] iff mask[i+1] — assert it faithfully propagates
    expected = [r.input_ids[i + 1] for i in range(len(r.input_ids) - 1) if r.loss_mask[i + 1]]
    assert [t for t in y if t != IGNORE_INDEX] == expected


def test_decontam_screens_tir_segment_leak():
    from lithos.posttrain.decontam_gate import PostTrainDecontaminator, messages_text

    probe = (
        "a train leaves chicago traveling west at sixty miles per hour while a second "
        "train departs denver heading east at forty on the same track"
    )
    gate = PostTrainDecontaminator([probe], n=13)
    leaked = {
        "messages": [
            {"role": "user", "content": "solve the classic puzzle"},
            {"role": "assistant", "segments": [
                {"type": "think", "text": probe + " so they meet..."},
                {"type": "text", "text": "in two hours"},
            ]},
        ]
    }
    # the leak is in an assistant SEGMENT, which the old messages_text missed
    assert gate.screen([leaked], messages_text) == []
