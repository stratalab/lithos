"""Tokenizer quality evaluation: per-domain compression, vocab health, roundtrip (PRD §7.2).

Extends ``inspect_tokenizer``'s single-blob fertility to the evaluation the STEM
tokenizer decision needs: compression measured *per domain* and compared against
reference tokenizers (a FineWeb-trained vocab looks fine on prose and only shows
its weakness on LaTeX/code), plus vocabulary-usage, special-token-stability, and
lossless-roundtrip checks. Driven by ``scripts/eval_tokenizer.py``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from tokenizers import Tokenizer

from lithos.tokenizer.tokenizer_config import DEFAULT_SPECIAL_TOKENS


def encode_ids(tok: Tokenizer, text: str) -> list[int]:
    """Token ids for ``text`` with no special tokens, so counts compare across tokenizers."""
    return tok.encode(text, add_special_tokens=False).ids


def compression_stats(tok: Tokenizer, texts: Iterable[str]) -> dict[str, Any]:
    """Compression over a text sample. ``bytes_per_token`` is the primary metric:
    byte-denominated, so it stays meaningful on code/LaTeX where "word" is fuzzy."""
    docs = tokens = words = chars = nbytes = 0
    for text in texts:
        docs += 1
        tokens += len(encode_ids(tok, text))
        words += len(text.split())
        chars += len(text)
        nbytes += len(text.encode("utf-8"))
    return {
        "docs": docs,
        "tokens": tokens,
        "words": words,
        "chars": chars,
        "bytes": nbytes,
        "tokens_per_word": tokens / max(words, 1),
        "chars_per_token": chars / max(tokens, 1),
        "bytes_per_token": nbytes / max(tokens, 1),
    }


def vocab_usage(tok: Tokenizer, texts: Iterable[str], *, rare_threshold: int = 5) -> dict[str, Any]:
    """How much of the vocabulary actually fires on a sample.

    Dead vocab slots are wasted embedding rows; tokens seen but only rarely are
    undertrained-token ("glitch token") candidates. Only meaningful on a large
    sample — on tiny probes most of the vocab is legitimately absent.
    """
    counts: Counter[int] = Counter()
    for text in texts:
        counts.update(encode_ids(tok, text))
    vocab_size = tok.get_vocab_size()
    used = len(counts)
    return {
        "vocab_size": vocab_size,
        "used": used,
        "used_fraction": used / max(vocab_size, 1),
        "rare_used": sum(1 for c in counts.values() if c <= rare_threshold),
        "rare_threshold": rare_threshold,
        "sample_tokens": sum(counts.values()),
    }


def roundtrip_failures(tok: Tokenizer, texts: Iterable[str]) -> list[dict[str, Any]]:
    """Texts where decode(encode(x)) != x. Byte-level BPE should make this empty;
    any failure means a normalizer/decoder setting is silently corrupting input."""
    failures = []
    for i, text in enumerate(texts):
        decoded = tok.decode(encode_ids(tok, text))
        if decoded != text:
            failures.append({"index": i, "text": text[:120], "decoded": decoded[:120]})
    return failures


def special_token_check(tok: Tokenizer, special_tokens: list[str] | None = None) -> dict[str, Any]:
    """Special tokens must sit at fixed low IDs (their list order) across retrains
    (PRD §7.1) so checkpoints and chat templates survive a tokenizer swap."""
    special_tokens = special_tokens if special_tokens is not None else DEFAULT_SPECIAL_TOKENS
    ids = {t: tok.token_to_id(t) for t in special_tokens}
    ok = all(ids[t] == i for i, t in enumerate(special_tokens))
    return {"ids": ids, "stable_low_ids": ok}


def segmentation_rows(tok: Tokenizer, probes: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tokenize short probe snippets ({text, note}) for eyeballing/diffing splits."""
    rows = []
    for probe in probes:
        enc = tok.encode(probe["text"], add_special_tokens=False)
        rows.append(
            {
                "text": probe["text"],
                "note": probe.get("note", ""),
                "n_tokens": len(enc.ids),
                "tokens": enc.tokens,
            }
        )
    return rows


def compare_tokenizers(
    domains: dict[str, list[str]], tokenizers: dict[str, Tokenizer]
) -> dict[str, dict[str, dict[str, Any]]]:
    """Per-domain compression for several tokenizers on identical text.

    Returns ``{domain: {tokenizer_name: {tokens, bytes_per_token}}}``. Fewer
    tokens on the same bytes == better compression; vocab sizes differ, so judge
    a small vocab by how little it loses, not by absolute parity.
    """
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for domain, texts in domains.items():
        nbytes = sum(len(t.encode("utf-8")) for t in texts)
        out[domain] = {}
        for name, tok in tokenizers.items():
            tokens = sum(len(encode_ids(tok, t)) for t in texts)
            out[domain][name] = {
                "tokens": tokens,
                "bytes_per_token": nbytes / max(tokens, 1),
            }
    return out


def evaluate_tokenizer(
    tok: Tokenizer,
    *,
    domains: dict[str, list[str]],
    segmentation_probes: Iterable[dict[str, Any]] = (),
    adversarial_texts: Iterable[str] = (),
    special_tokens: list[str] | None = None,
) -> dict[str, Any]:
    """Full intrinsic evaluation report (tiers 1-2; tier 3 is the bpb ablation)."""
    all_texts = [t for texts in domains.values() for t in texts]
    roundtrip_texts = all_texts + list(adversarial_texts)
    return {
        "vocab": vocab_usage(tok, all_texts),
        "special_tokens": special_token_check(tok, special_tokens),
        "domains": {name: compression_stats(tok, texts) for name, texts in domains.items()},
        "roundtrip": {
            "checked": len(roundtrip_texts),
            "failures": roundtrip_failures(tok, roundtrip_texts),
        },
        "segmentation": segmentation_rows(tok, segmentation_probes),
    }
