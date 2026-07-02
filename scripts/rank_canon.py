#!/usr/bin/env python3
"""Rank canon candidates from the topic-graph citation mine (doc §1.7).

Combines the two noise-killing signals validated on the first real run:
- **PPR-weighted counts** (from the miner): a cite from a field's core page
  outweighs thousands from the family tail (star catalogs, hobbyist pages).
- **Cross-domain specificity** (computed here): a work cited in every family
  (biographical dictionaries, philosophy encyclopedias) is generic reference,
  not domain canon — score = weighted * (weighted_d / weighted_total).

Reads data/topicgraph/out/canon_<domain>.csv, re-marks coverage against the
CURRENT corpus/seed_index.csv, writes corpus/canon_candidates_<domain>.csv
(the reviewable growth queue for the index).

  uv run python scripts/rank_canon.py            # all domains, top 200
  uv run python scripts/rank_canon.py --domain physics --top 500
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lithos.data.topicgraph import mark_seed_index_coverage

log = logging.getLogger("rank-canon")

DOMAINS = ["math", "physics", "cs", "eng", "chem"]
_WIKILINK_RE = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]")


def clean_title(title: str) -> str:
    """Strip wiki markup ([[X]] / [[X|Y]] / bold quotes) from a cite title."""
    return _WIKILINK_RE.sub(r"\1", title).replace("'''", "").replace("''", "").strip()


def load_books(out_dir: Path, domains: list[str]) -> tuple[dict, dict]:
    """(weighted[key][domain], meta[key]) for ISBN-bearing cite-book rows."""
    weighted: dict[str, dict[str, float]] = defaultdict(dict)
    meta: dict[str, dict] = {}
    for d in domains:
        path = out_dir / f"canon_{d}.csv"
        if not path.exists():
            log.warning("[%s] missing %s — skipping", d, path)
            continue
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r["kind"] != "cite book" or not r["isbn"]:
                    continue
                weighted[r["key"]][d] = float(r["weighted"])
                meta.setdefault(r["key"], r)
    return weighted, meta


def rank_domain(
    domain: str,
    weighted: dict[str, dict[str, float]],
    meta: dict[str, dict],
    *,
    min_citations: int = 3,
    min_specificity: float = 0.45,
) -> list[dict]:
    """Score = PPR-weight x cross-domain specificity; filtered and sorted."""
    rows = []
    for key, per in weighted.items():
        if domain not in per:
            continue
        r = meta[key]
        if int(r["citations"]) < min_citations:
            continue
        spec = per[domain] / sum(per.values())
        if spec < min_specificity:
            continue
        rows.append(
            {
                "score": per[domain] * spec,
                "weighted": per[domain],
                "specificity": round(spec, 3),
                "citations": int(r["citations"]),
                "title": clean_title(r["title"]),
                "author": r["author"],
                "year": r["year"],
                "isbn": r["isbn"],
                "key": key,
            }
        )
    rows.sort(key=lambda x: -x["score"])
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", action="append", choices=DOMAINS)
    p.add_argument("--out-dir", type=Path, default=Path("data/topicgraph/out"))
    p.add_argument("--dest-dir", type=Path, default=Path("corpus"))
    p.add_argument("--seed-index", type=Path, default=Path("corpus/seed_index.csv"))
    p.add_argument("--top", type=int, default=200)
    p.add_argument("--min-citations", type=int, default=3)
    p.add_argument("--min-specificity", type=float, default=0.45)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    domains = args.domain or DOMAINS
    weighted, meta = load_books(args.out_dir, domains)

    fieldnames = [
        "citations", "specificity", "title", "author", "year", "isbn",
        "in_seed_index", "weighted", "score", "key",
    ]
    for d in domains:
        rows = rank_domain(
            d, weighted, meta,
            min_citations=args.min_citations, min_specificity=args.min_specificity,
        )[: args.top]
        if args.seed_index.exists():
            rows = mark_seed_index_coverage(rows, args.seed_index)
        dest = args.dest_dir / f"canon_candidates_{d}.csv"
        with open(dest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        covered = sum(bool(r.get("in_seed_index")) for r in rows)
        log.info("[%s] %d candidates (%d already indexed) → %s", d, len(rows), covered, dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
