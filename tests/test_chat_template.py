"""Tests for the SFT chat template + loss masking (Phase 11)."""

import pytest
from lithos.posttrain.chat_template import (
    ROLE_TOKEN,
    Rendered,
    render_conversation,
    render_prompt,
    special_ids,
)

# Fixed IDs mirror DEFAULT_SPECIAL_TOKENS (order == id).
_NAMES = ["<pad>", "<bos>", "<eos>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]


class FakeTok:
    """Minimal stand-in: special tokens at fixed low IDs, content as char codes >=100."""

    def __init__(self, *, drop: str | None = None):
        self._ids = {n: i for i, n in enumerate(_NAMES) if n != drop}

    def token_to_id(self, token: str):
        return self._ids.get(token)

    def encode(self, text: str):
        class _Enc:
            ids = [100 + (ord(c) % 50) for c in text]

        return _Enc()


def _id(name):
    return _NAMES.index(name)


def test_render_masks_everything_but_assistant_response():
    tok = FakeTok()
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    r = render_conversation(msgs, tok)
    assert isinstance(r, Rendered)
    assert len(r.input_ids) == len(r.loss_mask)

    user_c = tok.encode("hi").ids
    asst_c = tok.encode("yo").ids
    # <bos> <|user|> hi <|end|> <|assistant|> yo <|end|>
    assert r.input_ids == [
        _id("<bos>"), _id("<|user|>"), *user_c, _id("<|end|>"),
        _id("<|assistant|>"), *asst_c, _id("<|end|>"),
    ]
    # loss ONLY on the assistant content + its closing <|end|>
    assert r.loss_mask == [
        False, False, *[False] * len(user_c), False,   # bos, user header+content+end
        False, *[True] * len(asst_c), True,             # asst header(masked), content, end(learned)
    ]


def test_multi_turn_learns_every_assistant_turn_only():
    tok = FakeTok()
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
    ]
    r = render_conversation(msgs, tok)
    # the only learned tokens are the two assistant contents (b, d) and their <|end|>s
    learned = [tid for tid, m in zip(r.input_ids, r.loss_mask) if m]
    assert learned == [
        *tok.encode("b").ids, _id("<|end|>"),
        *tok.encode("d").ids, _id("<|end|>"),
    ]
    # no system/user token is ever learned
    assert not any(
        m and tid in (_id("<|system|>"), _id("<|user|>")) for tid, m in zip(r.input_ids, r.loss_mask)
    )


def test_render_prompt_opens_assistant_turn_with_no_reply():
    tok = FakeTok()
    msgs = [{"role": "user", "content": "x"}]
    ids = render_prompt(msgs, tok)
    # ends with an OPEN assistant header and no content after it
    assert ids[-1] == _id("<|assistant|>")
    assert ids == [_id("<bos>"), _id("<|user|>"), *tok.encode("x").ids, _id("<|end|>"), _id("<|assistant|>")]


def test_add_bos_false_omits_bos():
    tok = FakeTok()
    r = render_conversation([{"role": "user", "content": "x"}], tok, add_bos=False)
    assert r.input_ids[0] == _id("<|user|>")


def test_unknown_role_rejected():
    tok = FakeTok()
    with pytest.raises(ValueError, match="unknown role"):
        render_conversation([{"role": "tool", "content": "x"}], tok)


def test_missing_special_tokens_errors_loudly():
    with pytest.raises(ValueError, match="missing chat special tokens"):
        special_ids(FakeTok(drop="<|assistant|>"))
