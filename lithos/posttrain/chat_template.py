"""Chat template + loss masking for SFT (Phase 11).

Renders a messages-format conversation into token ids using the tokenizer's
built-in chat special tokens (``DEFAULT_SPECIAL_TOKENS``), and marks which tokens
the model should be trained to *predict* — only the assistant's response (content
+ its closing ``<|end|>``). System/user turns and the assistant role *header* are
masked out of the loss, because at inference we *supply* the prompt and the
``<|assistant|>`` header; the model only needs to learn the response and to stop.

Template ``lithos-chat-v1``::

    <bos> <|system|> {system} <|end|> <|user|> {user} <|end|> <|assistant|> {reply} <|end|> ...

Special tokens are inserted **by ID**, never by string-parsing the rendered text,
so the loss mask is exact regardless of how the content happens to tokenize.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from lithos.posttrain.tir_validate import validate_tir_message

CHAT_TEMPLATE_VERSION = "lithos-chat-v1"

# message role -> the special token that opens that turn
ROLE_TOKEN = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}
_SPECIALS = ("<bos>", "<eos>", "<pad>", "<|end|>", *ROLE_TOKEN.values())

# TIR (tool-integrated reasoning) tokens — docs/tir-format.md §2. These live in the
# reserved block (IDs 7-15) of the STEM tokenizer, NOT in today's fineweb-edu-32k,
# so they are resolved lazily (only when a TIR episode is rendered), unlike the
# always-required core specials above.
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
TOOL_CLOSE, TOOL_RESULT = "<|/tool|>", "<|tool_result|>"
TOOL_OPEN = {"python": "<|python|>", "octave": "<|octave|>"}  # runtime identity in the open tag
TIR_TOKENS = (THINK_OPEN, THINK_CLOSE, *TOOL_OPEN.values(), TOOL_CLOSE, TOOL_RESULT)


class _Encoding(Protocol):
    ids: list[int]


class TokenizerLike(Protocol):
    """The slice of the ``tokenizers.Tokenizer`` API we rely on."""

    def token_to_id(self, token: str) -> int | None: ...
    def encode(self, text: str) -> _Encoding: ...


@dataclass
class Rendered:
    """A rendered conversation: ``input_ids`` and a per-token ``loss_mask``.

    ``loss_mask[j]`` is True iff ``input_ids[j]`` is an assistant token the model
    should be trained to produce (the dataset turns this into ``-100`` labels).
    """

    input_ids: list[int]
    loss_mask: list[bool]


def special_ids(tok: TokenizerLike) -> dict[str, int]:
    """Resolve the chat special-token IDs, erroring loudly if any are missing."""
    ids = {name: tok.token_to_id(name) for name in _SPECIALS}
    missing = [name for name, tid in ids.items() if tid is None]
    if missing:
        raise ValueError(
            f"tokenizer is missing chat special tokens {missing}; "
            "SFT needs the DEFAULT_SPECIAL_TOKENS vocab (IDs 0-6)"
        )
    return {name: int(tid) for name, tid in ids.items()}  # type: ignore[arg-type]


def tir_token_ids(tok: TokenizerLike) -> dict[str, int]:
    """Resolve the TIR special-token IDs, erroring loudly if any are missing.

    Called only when an assistant turn carries ``segments`` (tool/think), so
    non-TIR SFT keeps working on a tokenizer without the TIR vocab.
    """
    ids = {name: tok.token_to_id(name) for name in TIR_TOKENS}
    missing = [name for name, tid in ids.items() if tid is None]
    if missing:
        raise ValueError(
            f"tokenizer is missing TIR tokens {missing}; rendering a tool-use episode "
            "needs the STEM tokenizer's reserved block (docs/tir-format.md §2)"
        )
    return {name: int(tid) for name, tid in ids.items()}  # type: ignore[arg-type]


def _encode_segments(
    segments: list[dict], tok: TokenizerLike, tir: dict[str, int], sids: dict[str, int]
) -> tuple[list[int], list[bool]]:
    """Encode an assistant turn's TIR segments to (ids, mask), per docs/tir-format.md
    §4: think/text/tool are learned; the tool_result span (incl. its closing
    ``<|end|>``) is masked — the sandbox wrote it, the model must not learn to
    predict it. All tokens inserted by ID (never string-parsed). Segment structure is
    guaranteed by the caller's ``validate_tir_message`` (the shared standalone gate)."""
    ids: list[int] = []
    mask: list[bool] = []

    def emit(token_id: int, learn: bool) -> None:
        ids.append(token_id)
        mask.append(learn)

    def emit_text(text: str, learn: bool) -> None:
        enc = tok.encode(text).ids
        ids.extend(enc)
        mask.extend([learn] * len(enc))

    for seg in segments:  # structure validated by validate_tir_message before we get here
        stype = seg["type"]
        if stype == "think":
            emit(tir[THINK_OPEN], True)
            emit_text(seg["text"], True)
            emit(tir[THINK_CLOSE], True)
        elif stype == "text":
            emit_text(seg["text"], True)
        elif stype == "tool":
            emit(tir[TOOL_OPEN[seg["runtime"]]], True)
            emit_text(seg["code"], True)
            emit(tir[TOOL_CLOSE], True)
        else:  # tool_result (validated)
            emit(tir[TOOL_RESULT], False)
            emit_text(seg["output"], False)
            emit(sids["<|end|>"], False)  # result closer — masked
    return ids, mask


def _encode_turn(
    msg: dict, tok: TokenizerLike, sids: dict[str, int]
) -> tuple[list[int], list[bool]]:
    """Encode one message to (ids, mask): role header (masked) + body + ``<|end|>``.
    An assistant turn carries either flat ``content`` or a TIR ``segments`` list.
    Envelope + segment structure are gated by ``validate_tir_message`` (shared)."""
    validate_tir_message(msg)
    role = msg["role"]
    ids: list[int] = [sids[ROLE_TOKEN[role]]]
    mask: list[bool] = [False]  # role header — always masked (supplied at inference)
    if "segments" in msg:
        seg_ids, seg_mask = _encode_segments(msg["segments"], tok, tir_token_ids(tok), sids)
        ids.extend(seg_ids)
        mask.extend(seg_mask)
        ids.append(sids["<|end|>"])
        mask.append(True)  # turn terminator — learned (the model learns to stop)
    else:
        learn = role == "assistant"
        content_ids = tok.encode(msg["content"]).ids
        ids.extend(content_ids)
        mask.extend([learn] * len(content_ids))
        ids.append(sids["<|end|>"])
        mask.append(learn)
    return ids, mask


def render_conversation(
    messages: list[dict[str, str]], tok: TokenizerLike, *, add_bos: bool = True
) -> Rendered:
    """Render a full conversation for **training** (every turn present).

    Loss falls only on assistant tokens the model should produce — flat content,
    or TIR think/tool/answer segments — plus each assistant turn's closing
    ``<|end|>``. BOS, role headers, system/user turns, and injected tool results
    are masked. See docs/tir-format.md §4.
    """
    sids = special_ids(tok)
    out: list[int] = []
    mask: list[bool] = []
    if add_bos:
        out.append(sids["<bos>"])
        mask.append(False)
    for msg in messages:
        turn_ids, turn_mask = _encode_turn(msg, tok, sids)
        out.extend(turn_ids)
        mask.extend(turn_mask)
    return Rendered(out, mask)


def render_prompt(
    messages: list[dict[str, str]], tok: TokenizerLike, *, add_bos: bool = True
) -> list[int]:
    """Render a prompt for **generation**: the conversation so far, ending with an
    open ``<|assistant|>`` header for the model to continue. Used by inference, not
    training.
    """
    sids = special_ids(tok)
    out: list[int] = [sids["<bos>"]] if add_bos else []
    for msg in messages:
        turn_ids, _ = _encode_turn(msg, tok, sids)
        out.extend(turn_ids)
    out.append(sids["<|assistant|>"])  # open the assistant turn; model continues
    return out
