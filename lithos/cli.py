"""The ``lithos`` command — one entrypoint over the training / eval / data library.

    lithos train     --config configs/train/100m.yaml
    lithos sft        --config configs/sft/lithos-100m-packed.yaml
    lithos dpo        --config configs/dpo/lithos-100m-verifier.yaml
    lithos grpo       --config configs/grpo/lithos-tir-toy.yaml
    lithos eval       --config configs/eval/lithos-100m.yaml --checkpoint <dir>
    lithos tokenize   --config configs/data/smoke.yaml
    lithos tokenizer  --config configs/tokenizer/bpe-32k.yaml

This covers the **consumer** path (the training contract). Producer/data commands
(acquisition, extraction, curation) live in ``scripts/`` and migrate to Chisel — see
``docs/chisel-producer-migration.md``.

Imports are lazy per-subcommand so ``lithos --help`` stays instant and doesn't pull
torch. The ``scripts/*.py`` entrypoints are thin shims over these functions, kept so
``torchrun … scripts/train_model.py`` still works for distributed launches.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence


def _train(argv: Sequence[str]) -> int:
    from lithos.train.config import TrainConfig
    from lithos.train.loop import train
    from lithos.utils.config import load_and_validate

    ap = argparse.ArgumentParser(prog="lithos train", description="Pretrain a model.")
    ap.add_argument("--config", required=True, help="Path to a training YAML config.")
    ap.add_argument("--resume", default=None, help="Checkpoint directory to resume from.")
    ap.add_argument("--override", nargs="*", default=[], help="Dotted-key overrides.")
    a = ap.parse_args(argv)
    train(load_and_validate(a.config, TrainConfig, a.override), resume_from=a.resume)
    return 0


def _sft(argv: Sequence[str]) -> int:
    from lithos.train.entry import train_from_config

    ap = argparse.ArgumentParser(prog="lithos sft", description="Supervised fine-tuning.")
    ap.add_argument("--config", required=True, help="Path to an SFT YAML config.")
    ap.add_argument("--override", nargs="*", default=[], help="Dotted-key overrides.")
    ap.add_argument("--resume-from", default=None, help="Resume a paused SFT run.")
    a = ap.parse_args(argv)
    train_from_config(a.config, a.override, resume_from=a.resume_from)
    return 0


def _dpo(argv: Sequence[str]) -> int:
    from lithos.posttrain.dpo_trainer import train_dpo
    from lithos.train.config import TrainConfig
    from lithos.utils.config import load_and_validate

    ap = argparse.ArgumentParser(prog="lithos dpo", description="DPO preference tuning.")
    ap.add_argument("--config", required=True, help="Path to a DPO YAML config.")
    ap.add_argument("--override", nargs="*", default=[], help="Dotted-key overrides.")
    a = ap.parse_args(argv)
    train_dpo(load_and_validate(a.config, TrainConfig, a.override))
    return 0


def _grpo(argv: Sequence[str]) -> int:
    from lithos.posttrain.grpo_trainer import train_grpo
    from lithos.train.config import TrainConfig
    from lithos.utils.config import load_and_validate

    ap = argparse.ArgumentParser(prog="lithos grpo", description="GRPO / RLVR tuning.")
    ap.add_argument("--config", required=True, help="Path to a GRPO YAML config.")
    ap.add_argument("--override", nargs="*", default=[], help="Dotted-key overrides.")
    a = ap.parse_args(argv)
    train_grpo(load_and_validate(a.config, TrainConfig, a.override))
    return 0


def _eval(argv: Sequence[str]) -> int:
    from lithos.evals.config import EvalConfig
    from lithos.evals.run import evaluate_checkpoint
    from lithos.utils.config import load_and_validate

    ap = argparse.ArgumentParser(prog="lithos eval", description="Evaluate a checkpoint.")
    ap.add_argument("--config", required=True, help="Path to an eval YAML config.")
    ap.add_argument("--checkpoint", required=True, help="Checkpoint directory to evaluate.")
    ap.add_argument("--override", nargs="*", default=[], help="Dotted-key overrides.")
    a = ap.parse_args(argv)
    evaluate_checkpoint(load_and_validate(a.config, EvalConfig, a.override), a.checkpoint)
    return 0


def _tokenize(argv: Sequence[str]) -> int:
    from lithos.data.pipeline import CorpusBuildConfig, build_corpus
    from lithos.utils.config import load_and_validate

    ap = argparse.ArgumentParser(prog="lithos tokenize", description="Build tokenized corpus shards.")
    ap.add_argument("--config", required=True, help="Path to a corpus-build YAML config.")
    ap.add_argument("--override", nargs="*", default=[], help="Dotted-key overrides.")
    a = ap.parse_args(argv)
    build_corpus(load_and_validate(a.config, CorpusBuildConfig, a.override))
    return 0


def _tokenizer(argv: Sequence[str]) -> int:
    from lithos.tokenizer.data_source import resolve_texts
    from lithos.tokenizer.tokenizer_config import TokenizerTrainConfig
    from lithos.tokenizer.train_tokenizer import (
        build_manifest,
        sample_report,
        save_tokenizer,
        train_tokenizer,
    )
    from lithos.utils.config import load_and_validate

    ap = argparse.ArgumentParser(prog="lithos tokenizer", description="Train the BPE tokenizer.")
    ap.add_argument("--config", required=True, help="Path to a tokenizer training YAML config.")
    ap.add_argument("--out", default=None, help="Override the output directory.")
    ap.add_argument("--max-documents", type=int, default=None, help="Override data.max_documents.")
    ap.add_argument("--override", nargs="*", default=[], help="Dotted-key overrides.")
    a = ap.parse_args(argv)

    cfg = load_and_validate(a.config, TokenizerTrainConfig, a.override)
    if a.out:
        cfg.output_dir = a.out
    if a.max_documents is not None:
        cfg.data.max_documents = a.max_documents
    sources, texts = resolve_texts(cfg.data)
    print(f"Training {cfg.tokenizer.full_name} (vocab={cfg.tokenizer.vocab_size}) from {sources}...")
    tok, stats = train_tokenizer(cfg.tokenizer, texts)
    out = save_tokenizer(
        tok, cfg.tokenizer, cfg.output_dir,
        build_manifest(cfg.tokenizer, stats, sources), sample_report(tok, cfg.report_samples),
    )
    print(f"Saved {tok.get_vocab_size()}-token tokenizer to {out} "
          f"(docs={stats['num_documents']:,}, chars={stats['approx_chars']:,})")
    return 0


COMMANDS: dict[str, tuple[Callable[[Sequence[str]], int], str]] = {
    "train": (_train, "pretrain a model"),
    "sft": (_sft, "supervised fine-tuning"),
    "dpo": (_dpo, "DPO preference tuning"),
    "grpo": (_grpo, "GRPO / RLVR tuning"),
    "eval": (_eval, "evaluate a checkpoint"),
    "tokenize": (_tokenize, "build tokenized corpus shards"),
    "tokenizer": (_tokenizer, "train the BPE tokenizer"),
}


def main(argv: Sequence[str] | None = None) -> int:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print("usage: lithos <command> [options]   (try `lithos <command> --help`)\n")
        print("commands:")
        for name, (_fn, desc) in COMMANDS.items():
            print(f"  {name:<10} {desc}")
        return 0
    cmd, rest = args[0], args[1:]
    if cmd not in COMMANDS:
        print(f"lithos: unknown command {cmd!r}; choose from {', '.join(COMMANDS)}")
        return 2
    return COMMANDS[cmd][0](rest)


if __name__ == "__main__":
    raise SystemExit(main())
