#!/usr/bin/env python3
"""Inventory the P0 corpus sources — per-source and per-domain doc + token counts.

Reads a source catalog (configs/data/p0-sources.yaml) and reports, per source and
rolled up by domain: document count, estimated token count, and on-disk size. This
is the number the mix decision needs — *tokens per domain* — which raw bytes don't
give (math and code tokenize very differently from prose).

Doc counts are exact where cheap (Parquet row-group metadata; Stack Exchange
`_manifest.json` documents_out) and sampled-estimated for other JSONL. Token counts
are ESTIMATES: the first --sample docs of each source are tokenized with the current
32k tokenizer and scaled by the doc count. They shift once the STEM tokenizer is
retrained, but the relative per-domain proportions — what the mix sweep needs — hold.

  uv run python scripts/inventory_corpus.py                 # full inventory table
  uv run python scripts/inventory_corpus.py --json out.json # + machine-readable dump
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lithos.data.documents import _open_text  # noqa: E402

CATALOG = REPO / "configs" / "data" / "p0-sources.yaml"
TOKENIZER = REPO / "artifacts" / "tokenizer" / "fineweb-edu-32k" / "tokenizer.json"


def _files(patterns: list[str]) -> list[str]:
    out: list[str] = []
    for p in patterns:
        out.extend(sorted(glob.glob(p, recursive=True)))
    return out


def _bytes(files: list[str]) -> int:
    return sum(Path(f).stat().st_size for f in files)


def parquet_doc_count(files: list[str]) -> int:
    import pyarrow.parquet as pq

    return sum(pq.ParquetFile(f).metadata.num_rows for f in files)


def parquet_sample(files: list[str], field: str, k: int) -> list[str]:
    import pyarrow.parquet as pq

    out: list[str] = []
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=256, columns=[field]):
            for v in batch.column(0).to_pylist():
                if v:
                    out.append(v)
                if len(out) >= k:
                    return out
    return out


def jsonl_sample(files: list[str], field: str, k: int) -> list[str]:
    out: list[str] = []
    for f in files:
        with _open_text(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                v = json.loads(line).get(field)
                if v:
                    out.append(v)
                if len(out) >= k:
                    return out
    return out


def jsonl_doc_count(files: list[str]) -> tuple[int, str]:
    """Exact from a sibling _manifest.json (documents_out) if present, else estimate
    from up to 3 sampled shards scaled to the shard count."""
    parent = Path(files[0]).parent
    man = parent / "_manifest.json"
    if man.exists():
        m = json.loads(man.read_text())
        if "documents_out" in m:
            return int(m["documents_out"]), "exact"
    sample = files[: min(3, len(files))]
    counted = 0
    for f in sample:
        with _open_text(f) as fh:
            counted += sum(1 for line in fh if line.strip())
    per_shard = counted / len(sample)
    return int(per_shard * len(files)), "est"


def human(n: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.1f}{unit}"
    return f"{n:.0f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", type=Path, default=CATALOG)
    ap.add_argument("--sample", type=int, default=1000, help="docs sampled per source for token/doc")
    ap.add_argument("--json", type=Path, default=None, help="write the full inventory here")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOKENIZER))
    catalog = yaml.safe_load(args.catalog.read_text())["sources"]

    rows: list[dict[str, Any]] = []
    for spec in catalog:
        files = _files(spec["paths"])
        if not files:
            rows.append({**spec, "docs": 0, "count_method": "MISSING", "tokens": 0,
                         "gb": 0.0, "tok_per_doc": 0.0, "n_files": 0})
            continue
        field = spec.get("text_field", "text")
        if spec["kind"] == "parquet":
            docs, method = parquet_doc_count(files), "exact"
            sample = parquet_sample(files, field, args.sample)
        else:
            docs, method = jsonl_doc_count(files)
            sample = jsonl_sample(files, field, args.sample)
        tok_per_doc = 0.0
        if sample:
            enc = tok.encode_batch(sample)
            tok_per_doc = sum(len(e.ids) for e in enc) / len(sample)
        rows.append({
            "source_name": spec["source_name"], "domain": spec["domain"], "kind": spec["kind"],
            "docs": docs, "count_method": method, "tok_per_doc": round(tok_per_doc, 1),
            "tokens": int(docs * tok_per_doc), "gb": round(_bytes(files) / 1e9, 1),
            "n_files": len(files),
        })

    # ---- per-source table ----
    print(f"\n{'source':<32}{'domain':<9}{'docs':>9}{'tok/doc':>9}{'est tokens':>13}{'GB':>8}  count")
    print("-" * 92)
    for r in sorted(rows, key=lambda r: (r["domain"], -r["tokens"])):
        print(f"{r['source_name']:<32}{r['domain']:<9}{human(r['docs']):>9}"
              f"{r['tok_per_doc']:>9.0f}{human(r['tokens']):>13}{r['gb']:>7.1f}G  {r['count_method']}")

    # ---- per-domain rollup ----
    dom: dict[str, dict[str, float]] = defaultdict(lambda: {"docs": 0, "tokens": 0, "gb": 0.0})
    for r in rows:
        dom[r["domain"]]["docs"] += r["docs"]
        dom[r["domain"]]["tokens"] += r["tokens"]
        dom[r["domain"]]["gb"] += r["gb"]
    total_tok = sum(d["tokens"] for d in dom.values()) or 1
    print(f"\n{'DOMAIN':<12}{'docs':>10}{'est tokens':>14}{'% tokens':>10}{'GB':>9}")
    print("-" * 56)
    for d in sorted(dom, key=lambda d: -dom[d]["tokens"]):
        v = dom[d]
        print(f"{d:<12}{human(v['docs']):>10}{human(v['tokens']):>14}"
              f"{100 * v['tokens'] / total_tok:>9.1f}%{v['gb']:>8.1f}G")
    print("-" * 56)
    tot_docs = sum(r["docs"] for r in rows)
    tot_gb = sum(r["gb"] for r in rows)
    print(f"{'TOTAL':<12}{human(tot_docs):>10}{human(total_tok):>14}{'100.0%':>10}{tot_gb:>8.1f}G")
    print("\nNOTE: token counts are ESTIMATES (32k tokenizer, first-N sample per source); "
          "relative per-domain proportions are what the mix sweep uses.")

    if args.json:
        args.json.write_text(json.dumps(
            {"sources": rows, "domains": {d: dict(v) for d, v in dom.items()},
             "total_docs": tot_docs, "total_tokens": int(total_tok)}, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
