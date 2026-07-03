#!/usr/bin/env python3
"""Build the physics-eng pilot labeling set from the local wiki dump.

Samples family articles across PPR-rank strata (core / mid / tail) so the
pilot has genuine quality spread — core physics articles should score 3-4 on
the rubric, biography/history tail pages 1-2. That spread is what tests
whether the rubric separates quantitative substance from topic-adjacency.

  uv run python scripts/make_pilot_set.py --domain physics --per-stratum 150
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lithos.data.topicgraph import iter_xml_pages

log = logging.getLogger("pilot")

XML = Path("data/topicgraph/dumps/enwiki-latest-pages-articles-multistream.xml.bz2")
_MARKUP_RE = re.compile(r"\{\{[^{}]*\}\}|\[\[(?:[^|\]]*\|)?([^\]]*)\]\]|<[^>]+>|'{2,}")


def strip_markup(wikitext: str) -> str:
    """Rough wikitext → text (repeated template strip + link resolution)."""
    text = wikitext
    for _ in range(3):  # nested templates
        text = _MARKUP_RE.sub(lambda m: m.group(1) or "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", default="physics")
    p.add_argument("--per-stratum", type=int, default=150)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--family-dir", type=Path, default=Path("data/topicgraph/out"))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    fam_lines = (args.family_dir / f"family_{args.domain}.tsv").read_text().splitlines()
    titles = [line.split("\t")[0].replace("_", " ") for line in fam_lines if line]
    n = len(titles)
    strata = {
        "core": set(titles[: n // 100]),               # top 1%
        "mid": set(titles[n // 4 : n // 4 + n // 20]),  # around the 25th percentile
        "tail": set(titles[-n // 20 :]),                # bottom 5%
    }
    quota = {s: args.per_stratum for s in strata}
    out_path = args.out or Path(f"data/pilot/{args.domain}-eng.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with open(out_path, "w") as f:
        for title, wikitext in iter_xml_pages(XML):
            stratum = next((s for s, ts in strata.items() if title in ts), None)
            if stratum is None or quota[stratum] <= 0:
                continue
            text = strip_markup(wikitext)
            if len(text) < 500:
                continue
            quota[stratum] -= 1
            kept += 1
            f.write(json.dumps({"id": f"{stratum}:{title}", "text": text[:20000],
                                "stratum": stratum, "title": title}) + "\n")
            if all(q <= 0 for q in quota.values()):
                break
    log.info("pilot set: %d docs (remaining quota %s) → %s", kept, quota, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
