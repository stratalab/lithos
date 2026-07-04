#!/usr/bin/env python
"""Evaluate tokenizer quality: per-domain compression, references, vocab health (PRD §7.2).

python scripts/eval_tokenizer.py --tokenizer artifacts/tokenizer/fineweb-edu-32k
python scripts/eval_tokenizer.py --tokenizer ... --sample math=data/samples/math.jsonl \
    --references gpt2 Qwen/Qwen2.5-7B

Probe sets under corpus/probes/ give a fast, checked-in baseline; for meaningful
vocab-usage numbers pass large held-out samples via --sample (probes are tiny, so
most of the vocab is legitimately absent on them). The definitive tier-3 test is
a per-domain bits-per-byte ablation (scripts/run_ablation.py), not this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lithos.tokenizer.evaluate import compare_tokenizers, evaluate_tokenizer
from lithos.tokenizer.inspect_tokenizer import load_tokenizer
from lithos.utils.io import write_json
from rich.console import Console
from rich.table import Table
from tokenizers import Tokenizer

DEFAULT_PROBES = Path("corpus/probes")
DEFAULT_REFERENCES = ["gpt2", "Qwen/Qwen2.5-7B"]


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def resolve_tokenizer_path(raw: str) -> Path:
    path = Path(raw)
    return path / "tokenizer.json" if path.is_dir() else path


def load_domains(
    probes_dir: Path, samples: list[str], max_documents: int
) -> tuple[dict[str, list[str]], list[str], list[dict]]:
    """Assemble domain texts from probe files, then --sample overrides on top."""
    domains: dict[str, list[str]] = {}
    adversarial: list[str] = []
    segmentation: list[dict] = []
    for path in sorted(probes_dir.glob("*.jsonl")):
        texts = [r["text"] for r in read_jsonl(path)]
        if path.stem == "adversarial":
            adversarial = texts
        elif path.stem == "segmentation":
            segmentation = read_jsonl(path)
        else:
            domains[path.stem] = texts
    for spec in samples:
        name, _, raw_path = spec.partition("=")
        if not raw_path:
            raise SystemExit(f"--sample expects name=path, got {spec!r}")
        records = read_jsonl(Path(raw_path), limit=max_documents)
        domains[name] = [r["text"] for r in records if isinstance(r.get("text"), str)]
    return domains, adversarial, segmentation


def load_references(names: list[str], console: Console) -> dict[str, Tokenizer]:
    refs: dict[str, Tokenizer] = {}
    for name in names:
        try:
            refs[name] = Tokenizer.from_pretrained(name)
        except Exception as e:
            console.print(f"[yellow]skipping reference {name!r}: {e}[/yellow]")
    return refs


def print_report(
    console: Console, report: dict, comparison: dict, ours_name: str, ref_names: list[str]
) -> None:
    table = Table(title="Per-domain compression (bytes/token: higher = better)")
    table.add_column("domain")
    table.add_column("docs", justify="right")
    table.add_column("tokens", justify="right")
    table.add_column("tok/word", justify="right")
    table.add_column("bytes/tok", justify="right")
    for ref in ref_names:
        table.add_column(f"vs {ref}", justify="right")
    for domain, stats in report["domains"].items():
        row = [
            domain,
            f"{stats['docs']:,}",
            f"{stats['tokens']:,}",
            f"{stats['tokens_per_word']:.3f}",
            f"{stats['bytes_per_token']:.3f}",
        ]
        for ref in ref_names:
            # tokens ratio ours/ref on identical bytes: <1.00 means we compress better
            ratio = comparison[domain][ours_name]["tokens"] / comparison[domain][ref]["tokens"]
            row.append(f"{ratio:.2f}x")
        table.add_row(*row)
    console.print(table)

    vocab = report["vocab"]
    console.print(
        f"vocab: {vocab['used']:,}/{vocab['vocab_size']:,} tokens fired on the sample "
        f"({vocab['used_fraction']:.1%}); {vocab['rare_used']:,} fired <= {vocab['rare_threshold']} times "
        f"(undertrained-token candidates). Meaningful only on large --sample inputs."
    )

    special = report["special_tokens"]
    status = "[green]OK[/green]" if special["stable_low_ids"] else "[red]MISMATCH[/red]"
    console.print(f"special tokens at fixed low IDs: {status} {special['ids']}")

    failures = report["roundtrip"]["failures"]
    if failures:
        console.print(
            f"[red]roundtrip: {len(failures)}/{report['roundtrip']['checked']} FAILED[/red]"
        )
        for f in failures[:10]:
            console.print(f"  [red]#{f['index']}[/red] {f['text']!r} -> {f['decoded']!r}")
    else:
        console.print(
            f"roundtrip: [green]{report['roundtrip']['checked']}/{report['roundtrip']['checked']} lossless[/green]"
        )

    seg = Table(title="Segmentation probes")
    seg.add_column("text", max_width=42)
    seg.add_column("n", justify="right")
    seg.add_column("tokens", max_width=70)
    for row in report["segmentation"]:
        seg.add_row(row["text"], str(row["n_tokens"]), "|".join(row["tokens"]))
    console.print(seg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate tokenizer quality (tiers 1-2).")
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json or its directory.")
    ap.add_argument("--probes", default=str(DEFAULT_PROBES), help="Probe-set directory.")
    ap.add_argument(
        "--sample",
        action="append",
        default=[],
        help="name=path.jsonl — large held-out sample; adds/overrides that probe domain.",
    )
    ap.add_argument("--max-documents", type=int, default=5000, help="Doc cap per --sample.")
    ap.add_argument(
        "--references",
        nargs="*",
        default=DEFAULT_REFERENCES,
        help="HF tokenizers to compare against (pass --references with nothing to disable).",
    )
    ap.add_argument(
        "--out", default=None, help="Report JSON path (default: <tokenizer dir>/eval_report.json)."
    )
    args = ap.parse_args()

    console = Console()
    tok_path = resolve_tokenizer_path(args.tokenizer)
    tok = load_tokenizer(tok_path)
    ours_name = tok_path.parent.name or "ours"

    domains, adversarial, segmentation = load_domains(
        Path(args.probes), args.sample, args.max_documents
    )
    if not domains:
        sys.exit(f"no domain probe files found in {args.probes}")

    report = evaluate_tokenizer(
        tok, domains=domains, segmentation_probes=segmentation, adversarial_texts=adversarial
    )

    refs = load_references(args.references, console)
    comparison = compare_tokenizers(domains, {ours_name: tok, **refs})
    report["references"] = comparison

    print_report(console, report, comparison, ours_name, list(refs))

    out = Path(args.out) if args.out else tok_path.parent / "eval_report.json"
    write_json(out, report)
    console.print(f"report written to {out}")


if __name__ == "__main__":
    main()
