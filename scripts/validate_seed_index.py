#!/usr/bin/env python3
"""Validate corpus/seed_index.csv and report coverage.

Schema check (enums, required fields, doctrine invariants) + a coverage
report: est. tokens by domain / tier / priority. The index is the
enforcement surface for the sourcing doctrine (docs/data-construction.md
S1.5-1.7), so a malformed row is a policy bug, not a typo.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

INDEX = Path(__file__).resolve().parent.parent / "corpus" / "seed_index.csv"

COLUMNS = [
    "id", "kind", "domain", "subfield", "level", "title", "creator",
    "canonical_id", "tier", "license_note", "est_tokens", "priority",
    "epoch_cap", "route", "status",
]
KINDS = {"corpus", "dump", "series", "book", "reference", "notes", "problems"}
DOMAINS = {"code", "math", "physics", "eng", "chem", "general", "xdomain"}
LEVELS = {"intro", "ug", "grad", "research", "mixed"}
TIERS = {"green", "grey", "mixed"}
PRIORITIES = {"P0", "P1", "P2"}
ROUTES = {"hf", "dump", "arxiv-src", "free-web", "pd", "scrape", "lawful-copy"}
STATUSES = {"indexed", "acquired", "extracted", "shipped"}

SUFFIX = {"K": 1e3, "M": 1e6, "B": 1e9}


def parse_tokens(s: str) -> float:
    s = s.strip()
    if not s or s[-1] not in SUFFIX:
        raise ValueError(f"est_tokens needs K/M/B suffix: {s!r}")
    return float(s[:-1]) * SUFFIX[s[-1]]


def fmt(n: float) -> str:
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return str(int(n))


def main() -> int:
    errors: list[str] = []
    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    with INDEX.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            errors.append(f"header mismatch: {reader.fieldnames}")
        for i, row in enumerate(reader, start=2):
            where = f"line {i} ({row.get('id', '?')})"
            if row["id"] in seen_ids:
                errors.append(f"{where}: duplicate id")
            seen_ids.add(row["id"])
            for col, allowed in (
                ("kind", KINDS), ("domain", DOMAINS), ("level", LEVELS),
                ("tier", TIERS), ("priority", PRIORITIES),
                ("route", ROUTES), ("status", STATUSES),
            ):
                if row[col] not in allowed:
                    errors.append(f"{where}: bad {col}={row[col]!r}")
            for col in ("title", "creator", "canonical_id"):
                if not row[col].strip():
                    errors.append(f"{where}: empty {col}")
            try:
                row["_tokens"] = parse_tokens(row["est_tokens"])  # type: ignore[assignment]
            except ValueError as e:
                errors.append(f"{where}: {e}")
                row["_tokens"] = 0.0  # type: ignore[assignment]
            # Doctrine invariants: grey books/series/references/notes are
            # epoch-capped; nothing sourced from a non-public route.
            if row["tier"] == "grey" and row["kind"] in {"book", "series", "reference", "notes", "problems"}:
                if row["epoch_cap"] in ("", "-"):
                    errors.append(f"{where}: grey {row['kind']} needs an epoch_cap")
            if row["epoch_cap"] not in ("", "-"):
                try:
                    if int(row["epoch_cap"]) <= 0:
                        raise ValueError
                except ValueError:
                    errors.append(f"{where}: epoch_cap must be a positive int or '-'")
            rows.append(row)

    if errors:
        print(f"INVALID — {len(errors)} error(s):")
        for e in errors:
            print(f"  {e}")
        return 1

    by_domain: dict[str, float] = defaultdict(float)
    by_tier: dict[str, float] = defaultdict(float)
    by_priority: dict[str, float] = defaultdict(float)
    canon_tokens = 0.0
    canon_count = 0
    for row in rows:
        t = row["_tokens"]  # type: ignore[index]
        by_domain[row["domain"]] += t
        by_tier[row["tier"]] += t
        by_priority[row["priority"]] += t
        if row["kind"] in {"book", "series", "reference", "notes"}:
            canon_tokens += t
            canon_count += 1

    print(f"OK — {len(rows)} rows\n")
    print("Tokens by domain:")
    for k, v in sorted(by_domain.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<8} {fmt(v):>8}")
    print("Tokens by tier:")
    for k, v in sorted(by_tier.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<8} {fmt(v):>8}")
    print("Tokens by priority:")
    for k in sorted(by_priority):
        print(f"  {k:<8} {fmt(by_priority[k]):>8}")
    print(f"Book canon: {canon_count} works, {fmt(canon_tokens)} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
