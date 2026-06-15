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

CHAT_TEMPLATE_VERSION = "lithos-chat-v1"

# message role -> the special token that opens that turn
ROLE_TOKEN = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}
_SPECIALS = ("<bos>", "<eos>", "<pad>", "<|end|>", *ROLE_TOKEN.values())


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


def render_conversation(
    messages: list[dict[str, str]], tok: TokenizerLike, *, add_bos: bool = True
) -> Rendered:
    """Render a full conversation for **training** (every turn present).

    Loss falls only on assistant content + the ``<|end|>`` that closes each
    assistant turn; everything else (BOS, role headers, system/user turns) is
    masked.
    """
    sids = special_ids(tok)
    out: list[int] = []
    mask: list[bool] = []
    if add_bos:
        out.append(sids["<bos>"])
        mask.append(False)
    for msg in messages:
        role = msg["role"]
        if role not in ROLE_TOKEN:
            raise ValueError(f"unknown role {role!r}; expected one of {list(ROLE_TOKEN)}")
        learn = role == "assistant"
        content_ids = tok.encode(msg["content"]).ids
        # role header — always masked (supplied at inference)
        out.append(sids[ROLE_TOKEN[role]])
        mask.append(False)
        # content — learned only for assistant turns
        out.extend(content_ids)
        mask.extend([learn] * len(content_ids))
        # turn terminator — learned for assistant (so the model learns to stop)
        out.append(sids["<|end|>"])
        mask.append(learn)
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
        role = msg["role"]
        if role not in ROLE_TOKEN:
            raise ValueError(f"unknown role {role!r}; expected one of {list(ROLE_TOKEN)}")
        out.append(sids[ROLE_TOKEN[role]])
        out.extend(tok.encode(msg["content"]).ids)
        out.append(sids["<|end|>"])
    out.append(sids["<|assistant|>"])  # open the assistant turn; model continues
    return out
