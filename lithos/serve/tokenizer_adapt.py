"""Give a Qwen base tokenizer the Lithos chat + TIR special tokens.

v1 post-trains a Qwen3 base (`docs/v1-on-qwen.md`). Qwen's tokenizer does not carry our
named chat specials (``<|system|>``, ``<|end|>``, …) or the whole TIR block, and the chat
template and ``tir_rollout`` insert those **by name** — so they must resolve to stable ids.
This adds exactly the missing ones and reports the vocab size the imported model must cover.

**One source of truth.** The token *strings* come from ``chat_template.REQUIRED_SPECIAL_TOKENS``,
so the tokenizer and the renderer cannot drift on what must exist — the same discipline that
made ``validate_tir_record`` shared rather than reimplemented.

**Reuse over duplication.** Qwen3 already tokenizes ``<think>`` / ``</think>`` for its own
reasoning mode. If the base already has a token, we keep its id — the pretrained model already
knows that embedding — rather than mint a colliding synonym.

**Why this preserves import parity.** New tokens are appended at ids ≥ Qwen's current vocab.
``load_qwen3(hf_model, vocab_size=…)`` grows the embedding to cover them; ``_pad_rows``
zero-inits the new rows, and the logit mask (`transformer`, ``>= cfg.vocab_size``) leaves the
added ids **valid** rather than masked. The rename/copy of Qwen's real rows is untouched, so
logits over the original vocab are bit-for-bit what they were. The added rows are trained
during SFT.

The object we operate on and return is the **backend** ``tokenizers.Tokenizer`` (via
``hf_tokenizer.backend_tokenizer``), because that is what exposes ``token_to_id`` and an
``encode(...).ids`` — exactly the ``TokenizerLike`` slice the composite and chat template use.
A HF ``PreTrainedTokenizerFast`` wrapper exposes neither in that shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lithos.posttrain.chat_template import REQUIRED_SPECIAL_TOKENS
from lithos.utils.io import ensure_dir, write_json

#: The sidecar naming the model this tokenizer was cut for. The augmented tokenizer's
#: ``vocab_size`` and the imported model's must agree, or shard ids can index past the
#: embedding — so the contract is written down next to the artifact, not left implicit.
ADAPT_MANIFEST_NAME = "adapt.json"
TOKENIZER_FILE_NAME = "tokenizer.json"


@dataclass(frozen=True)
class AugmentResult:
    """The augmented tokenizer plus everything the caller needs to import a model for it."""

    tokenizer: Any  # a tokenizers.Tokenizer, mutated in place and returned for convenience
    ids: dict[str, int] = field(default_factory=dict)  # special -> id, all of REQUIRED_*
    added: tuple[str, ...] = ()  # minted new (need fresh embedding rows)
    reused: tuple[str, ...] = ()  # already present in the base (e.g. Qwen's <think>)
    base_vocab_size: int = 0  # the base tokenizer's size before augmentation
    vocab_size: int = 0  # after augmentation (== get_vocab_size())


def augment_tokenizer(backend: Any) -> AugmentResult:
    """Add the missing Lithos specials to a ``tokenizers.Tokenizer`` (mutates in place).

    Verifies every required special resolves to a distinct id and encodes **atomically**
    (a special that split into pieces would break insert-by-id and the loss mask).
    """
    base_vocab = int(backend.get_vocab_size())

    added: list[str] = []
    reused: list[str] = []
    for tok in REQUIRED_SPECIAL_TOKENS:
        if backend.token_to_id(tok) is not None:
            reused.append(tok)
        else:
            added.append(tok)

    if added:
        backend.add_special_tokens(added)

    ids: dict[str, int] = {}
    for tok in REQUIRED_SPECIAL_TOKENS:
        tid = backend.token_to_id(tok)
        if tid is None:  # pragma: no cover - add_special_tokens just guaranteed this
            raise ValueError(f"special token {tok!r} did not resolve after augmentation")
        enc = backend.encode(tok).ids
        if enc != [tid]:
            raise ValueError(
                f"special token {tok!r} is not atomic: encodes to {enc}, not [{tid}]. "
                f"An added special that splits would break insert-by-id and the loss mask."
            )
        ids[tok] = tid

    if len(set(ids.values())) != len(ids):
        clash = {t: i for t, i in ids.items()}
        raise ValueError(f"special tokens collided on ids after augmentation: {clash}")

    return AugmentResult(
        tokenizer=backend,
        ids=ids,
        added=tuple(added),
        reused=tuple(reused),
        base_vocab_size=base_vocab,
        vocab_size=int(backend.get_vocab_size()),
    )


def import_vocab_size(hf_config: Any, result: AugmentResult) -> int:
    """The ``vocab_size`` to pass to ``load_qwen3`` so every added special is a valid token.

    ``max(config.vocab_size, highest_special_id + 1)``: if the added specials fit under the
    config's vocab (Qwen ships spare embedding rows below ``config.vocab_size``), no growth is
    needed and they reuse those rows; otherwise the embedding grows to cover them.
    """
    highest = max(result.ids.values())
    return max(int(hf_config.vocab_size), highest + 1)


def adapt_qwen(hf_model: Any, backend_tokenizer: Any) -> tuple[Any, AugmentResult]:
    """Augment the tokenizer and import the Qwen3 model sized to fit the added specials.

    Returns ``(LithosForCausalLM, AugmentResult)``. Parity over Qwen's original vocab is
    preserved; the new special rows are zero-initialised and trained during SFT.
    """
    from lithos.serve.hf_import import load_qwen3

    result = augment_tokenizer(backend_tokenizer)
    vsize = import_vocab_size(hf_model.config, result)
    model = load_qwen3(hf_model, vocab_size=vsize)
    return model, result


def save_augmented_tokenizer(
    base_tokenizer: Any, out_dir: str | Path, *, base_model: str | None = None
) -> AugmentResult:
    """Augment a base tokenizer and write it as a build artifact: ``tokenizer.json`` plus an
    ``adapt.json`` sidecar recording the vocab size, the added/reused specials, and the base
    model. The existing SFT/RLVR builds consume the ``tokenizer.json`` unchanged — "retokenize
    on the Qwen tokenizer" is just pointing ``tokenizer_path`` at this directory.

    ``base_tokenizer`` may be a live ``tokenizers.Tokenizer`` or a path to a ``tokenizer.json``
    (e.g. Qwen's). It is augmented in place; the original file on disk is not touched.
    """
    if isinstance(base_tokenizer, (str, Path)):
        from tokenizers import Tokenizer

        backend = Tokenizer.from_file(str(base_tokenizer))
    else:
        backend = base_tokenizer

    result = augment_tokenizer(backend)
    out = ensure_dir(out_dir)
    backend.save(str(out / TOKENIZER_FILE_NAME))
    write_json(
        out / ADAPT_MANIFEST_NAME,
        {
            "base_model": base_model,
            "base_vocab_size": result.base_vocab_size,
            "vocab_size": result.vocab_size,
            "added": list(result.added),
            "reused": list(result.reused),
            "special_ids": result.ids,
            "required_special_tokens": list(REQUIRED_SPECIAL_TOKENS),
        },
    )
    return result


def assert_tokenizer_matches_model(tok: Any, model_vocab_size: int) -> None:
    """Guard the train/serve contract: every id the tokenizer can emit must index a real
    model row. A tokenizer larger than the model's vocab would index past the embedding.
    """
    tvocab = int(tok.get_vocab_size())
    if tvocab > model_vocab_size:
        raise ValueError(
            f"tokenizer vocab ({tvocab}) exceeds the model's vocab ({model_vocab_size}): "
            f"token ids would index past the embedding. Import the model with "
            f"vocab_size >= {tvocab} (see import_vocab_size)."
        )
