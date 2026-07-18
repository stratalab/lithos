"""Standalone structural validation for TIR / messages records (docs/tir-format.md §7).

The **shared ingestion gate** for tool-integrated-reasoning training data: both Chisel
(before emitting a trace) and Lithos (before rendering one) call ``validate_tir_record``,
so a malformed segment fails LOUD at the boundary rather than silently poisoning the SFT
loss mask. A ``tool_result`` span is masked by token ID; a mistyped or misplaced segment
would train the model on the sandbox's output — the worst kind of data bug, invisible
until it degrades the model. This module is pure structure — **no tokenizer required** —
so a producer can gate its output offline.

Schema (authoritative; renders to the wire format in docs/tir-format.md §2-§4):

    record   = {"messages": [message, ...]}                       # non-empty
    message  = {"role": "system"|"user"|"assistant", "content": str}
             | {"role": "assistant", "segments": [segment, ...]}  # exactly one of content|segments
    segment  = {"type": "think",       "text": str}                # learned
             | {"type": "text",        "text": str}                # learned
             | {"type": "tool",        "runtime": "python"|"octave"|"assay", "code": str}  # learned
             | {"type": "tool_result", "output": str}              # MASKED from the loss

For ``runtime="assay"`` the ``code`` payload is the Assay IR as JSON text (task +
inputs + missing_inputs) rather than source code — the template owns the method;
the model only names the task and fills the slots (docs/tir-format.md §2).

``chat_template.render_conversation`` renders a validated record to (input_ids, loss_mask);
it calls the same validators, so the two paths cannot drift (guarded by test_tir_validate).
"""

from __future__ import annotations

from typing import Any

VALID_ROLES = ("system", "user", "assistant")
SEGMENT_TYPES = ("think", "text", "tool", "tool_result")
TOOL_RUNTIMES = ("python", "octave", "assay")
# required string field(s) per segment type ("runtime" is whitelist-checked separately)
_SEGMENT_FIELDS: dict[str, tuple[str, ...]] = {
    "think": ("text",),
    "text": ("text",),
    "tool": ("runtime", "code"),
    "tool_result": ("output",),
}


def _require_str(seg: dict[str, Any], key: str, stype: Any) -> None:
    if key not in seg:
        raise ValueError(f"{stype!r} segment missing required field {key!r}")
    if not isinstance(seg[key], str):
        raise ValueError(
            f"{stype!r} segment field {key!r} must be a string, got {type(seg[key]).__name__}"
        )


def validate_tir_segments(segments: Any) -> None:
    """Validate an assistant turn's ``segments`` list. Raises ``ValueError`` on any
    malformation (non-list, non-dict segment, unknown type/runtime, missing/mistyped
    field)."""
    if not isinstance(segments, list):
        raise ValueError(f"'segments' must be a list, got {type(segments).__name__}")
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            raise ValueError(f"segment {i} must be a dict, got {type(seg).__name__}")
        stype = seg.get("type")
        if stype not in SEGMENT_TYPES:
            raise ValueError(f"unknown segment type {stype!r}; expected one of {list(SEGMENT_TYPES)}")
        if stype == "tool" and seg.get("runtime") not in TOOL_RUNTIMES:
            raise ValueError(
                f"unknown tool runtime {seg.get('runtime')!r}; expected one of {list(TOOL_RUNTIMES)}"
            )
        for key in _SEGMENT_FIELDS[stype]:
            if key != "runtime":  # runtime already checked against the whitelist above
                _require_str(seg, key, stype)


def validate_tir_message(msg: Any) -> None:
    """Validate one message's envelope: a valid role and **exactly one** of
    ``content``|``segments`` (segments assistant-only). Recurses into segments."""
    if not isinstance(msg, dict):
        raise ValueError(f"message must be a dict, got {type(msg).__name__}")
    role = msg.get("role")
    if role not in VALID_ROLES:
        raise ValueError(f"unknown role {role!r}; expected one of {list(VALID_ROLES)}")
    has_segments, has_content = "segments" in msg, "content" in msg
    if has_segments and role != "assistant":
        raise ValueError(f"{role!r} turn cannot have 'segments' — segments are assistant-only")
    if has_segments and has_content:
        raise ValueError("assistant turn has both 'content' and 'segments'; use exactly one")
    if not has_segments and not has_content:
        raise ValueError(f"{role!r} turn missing 'content'")
    if has_content and not isinstance(msg["content"], str):
        raise ValueError(f"'content' must be a string, got {type(msg['content']).__name__}")
    if has_segments:
        validate_tir_segments(msg["segments"])


def validate_tir_record(record: Any) -> None:
    """The shared ingestion gate: validate a full ``{"messages": [...]}`` record.

    Raises ``ValueError`` with a precise reason on any malformation; returns ``None``
    if the record is well-formed. Call it before emitting (Chisel) and before rendering
    (Lithos). Structural only — it does not judge content quality.
    """
    if not isinstance(record, dict):
        raise ValueError(f"record must be a dict, got {type(record).__name__}")
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"record 'messages' must be a list, got {type(messages).__name__}")
    if not messages:
        raise ValueError("record 'messages' is empty")
    for msg in messages:
        validate_tir_message(msg)
