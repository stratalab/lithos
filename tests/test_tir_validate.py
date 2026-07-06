"""Tests for the standalone TIR ingestion validator (tir_validate.py) — the shared gate.

Guards the Chisel<->Lithos contract (docs/chisel-f7-response.md): every golden fixture
must validate AND render, and every malformed case must be rejected by BOTH the standalone
validator and the renderer — so the two paths cannot drift. tir_golden.jsonl is the shared
golden set both repos test against.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from lithos.posttrain.chat_template import TIR_TOKENS, render_conversation
from lithos.posttrain.tir_validate import validate_tir_record

_CORE = ["<pad>", "<bos>", "<eos>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]


class _TirTok:
    """Fake tokenizer carrying the core + TIR special tokens (id values arbitrary)."""

    def __init__(self) -> None:
        self._ids = {n: i for i, n in enumerate(_CORE + list(TIR_TOKENS))}

    def token_to_id(self, token: str) -> int | None:
        return self._ids.get(token)

    def encode(self, text: str) -> SimpleNamespace:
        return SimpleNamespace(ids=[100 + (ord(c) % 50) for c in text])


FIXTURES = Path(__file__).parent / "fixtures" / "tir_golden.jsonl"


def _golden() -> list[dict]:
    with open(FIXTURES, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_golden_fixtures_validate_and_render() -> None:
    tok = _TirTok()
    records = _golden()
    assert records, "golden fixtures must not be empty"
    for rec in records:
        validate_tir_record(rec)  # the standalone gate accepts it
        render_conversation(rec["messages"], tok)  # and the renderer agrees


# (malformed record, expected error substring) — the ingestion gate must reject each.
_MALFORMED = [
    ({"messages": []}, "'messages' is empty"),
    ({"messages": "nope"}, "'messages' must be a list"),
    ({"messages": [{"role": "assistant"}]}, "missing 'content'"),
    ({"messages": [{"role": "user", "segments": [{"type": "text", "text": "x"}]}]}, "segments are assistant-only"),
    ({"messages": [{"role": "assistant", "content": "c", "segments": []}]}, "both 'content' and 'segments'"),
    ({"messages": [{"role": "assistant", "segments": [{"type": "bogus", "text": "x"}]}]}, "unknown segment type"),
    ({"messages": [{"role": "assistant", "segments": [{"type": "tool", "runtime": "ruby", "code": "x"}]}]}, "unknown tool runtime"),
    ({"messages": [{"role": "assistant", "segments": [{"type": "tool_result"}]}]}, "missing required field 'output'"),
    ({"messages": [{"role": "assistant", "segments": [{"type": "text", "text": 5}]}]}, "must be a string, got int"),
    ({"messages": [{"role": "assistant", "segments": "nope"}]}, "'segments' must be a list"),
]


@pytest.mark.parametrize("rec, match", _MALFORMED)
def test_malformed_rejected_by_validator(rec: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_tir_record(rec)


@pytest.mark.parametrize("rec, match", _MALFORMED)
def test_renderer_rejects_the_same_message_level_cases(rec: dict, match: str) -> None:
    # message-level malformations must ALSO fail the renderer (the two paths agree);
    # record-envelope-only cases (empty / non-list messages) the renderer never sees.
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or not msgs:
        pytest.skip("record-level envelope case; the renderer takes a messages list")
    with pytest.raises(ValueError):
        render_conversation(msgs, _TirTok())
