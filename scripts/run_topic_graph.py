#!/usr/bin/env python3
"""Run the Wikipedia topic-graph job (docs/data-construction.md §1.7).

Stages (each writes under --out-dir, default data/topicgraph/out):
  download   fetch the needed enwiki dumps into --dumps-dir (resumable skip)
  graph      build link graph → PPR per domain → family_<d>.tsv + vocab_<d>.txt
  citations  mine {{cite ...}} across each family → canon_<d>.csv
             (ranked canon candidates, coverage-marked vs corpus/seed_index.csv)

Typical:  uv run python scripts/run_topic_graph.py download
          uv run python scripts/run_topic_graph.py graph
          uv run python scripts/run_topic_graph.py citations

Full-scale memory note: graph stage wants ~32 GB RAM on complete enwiki dumps
(see lithos/data/topicgraph.py); --max-edges gives a cheap smoke run.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import urllib.request
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lithos.data import topicgraph as tg

log = logging.getLogger("topicgraph")

DUMP_BASE = "https://dumps.wikimedia.org/enwiki/latest/"
SQL_DUMPS = {
    "page": "enwiki-latest-page.sql.gz",
    "redirect": "enwiki-latest-redirect.sql.gz",
    "linktarget": "enwiki-latest-linktarget.sql.gz",
    "pagelinks": "enwiki-latest-pagelinks.sql.gz",
}
XML_DUMP = "enwiki-latest-pages-articles-multistream.xml.bz2"


def download(dumps_dir: Path, with_xml: bool) -> None:
    dumps_dir.mkdir(parents=True, exist_ok=True)
    names = list(SQL_DUMPS.values()) + ([XML_DUMP] if with_xml else [])
    for name in names:
        dest = dumps_dir / name
        if dest.exists():
            log.info("have %s (%.1f GB) — skipping", name, dest.stat().st_size / 1e9)
            continue
        url = DUMP_BASE + name
        log.info("downloading %s ...", url)
        tmp = dest.with_suffix(dest.suffix + ".part")
        # Wikimedia 403s the default urllib UA; their policy wants a descriptive one.
        req = urllib.request.Request(
            url, headers={"User-Agent": "LithosCorpusBot/0.1 (offline research corpus build)"}
        )
        with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 22):
                f.write(chunk)
        tmp.rename(dest)
        log.info("done: %s (%.1f GB)", name, dest.stat().st_size / 1e9)


def run_graph(cfg: dict, dumps_dir: Path, out_dir: Path, max_edges: int | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    graph = tg.build_link_graph(
        dumps_dir / SQL_DUMPS["page"],
        dumps_dir / SQL_DUMPS["redirect"],
        dumps_dir / SQL_DUMPS["pagelinks"],
        dumps_dir / SQL_DUMPS["linktarget"],
        max_edges=max_edges,
    )
    for domain, dcfg in cfg["domains"].items():
        seeds = []
        for title in dcfg["seeds"]:
            idx = graph.resolve(title)
            if idx is None:
                log.warning("[%s] seed not found: %r", domain, title)
            else:
                seeds.append(idx)
        log.info("[%s] %d/%d seeds resolved", domain, len(seeds), len(dcfg["seeds"]))
        scores = tg.personalized_pagerank(
            graph,
            seeds,
            alpha=float(cfg.get("alpha", 0.85)),
            max_iter=int(cfg.get("max_iter", 40)),
        )
        family = tg.select_family(graph, scores, int(dcfg.get("top_n", cfg.get("top_n", 60000))))
        fam_path = out_dir / f"family_{domain}.tsv"
        with open(fam_path, "w") as f:
            for title, score in family:
                f.write(f"{title}\t{score:.6e}\n")
        vocab = tg.family_vocabulary(graph, (t for t, _ in family))
        (out_dir / f"vocab_{domain}.txt").write_text("\n".join(vocab) + "\n")
        log.info("[%s] family=%d vocab=%d → %s", domain, len(family), len(vocab), fam_path)


def run_citations(cfg: dict, dumps_dir: Path, out_dir: Path, seed_index: Path) -> None:
    xml_path = dumps_dir / XML_DUMP
    if not xml_path.exists() and xml_path.with_suffix("").exists():
        xml_path = xml_path.with_suffix("")  # uncompressed variant (smoke runs)
    families: dict[str, dict[str, float]] = {}
    for domain in cfg["domains"]:
        fam_path = out_dir / f"family_{domain}.tsv"
        if not fam_path.exists():
            log.warning("[%s] no family file (run `graph` first) — skipping", domain)
            continue
        fam: dict[str, float] = {}
        for line in fam_path.read_text().splitlines():
            if line:
                title, score = line.split("\t")
                fam[title] = float(score)  # PPR score = citation weight
        families[domain] = fam
    if not families:
        return
    # One pass over the ~26GB dump for ALL families, PPR-weighted counts.
    per_domain = tg.mine_citations_multi(xml_path, families)
    fieldnames = [
        "weighted", "citations", "kind", "title", "author", "year", "isbn", "doi", "key",
        "in_seed_index",
    ]
    for domain, candidates in per_domain.items():
        if seed_index.exists():
            candidates = tg.mark_seed_index_coverage(candidates, seed_index)
        out_path = out_dir / f"canon_{domain}.csv"
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(candidates)
        log.info("[%s] %d canon candidates → %s", domain, len(candidates), out_path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["download", "graph", "citations", "all"])
    p.add_argument("--config", type=Path, default=Path("configs/topicgraph/seeds.yaml"))
    p.add_argument("--dumps-dir", type=Path, default=Path("data/topicgraph/dumps"))
    p.add_argument("--out-dir", type=Path, default=Path("data/topicgraph/out"))
    p.add_argument("--seed-index", type=Path, default=Path("corpus/seed_index.csv"))
    p.add_argument("--max-edges", type=int, default=None, help="cap edges (smoke runs)")
    p.add_argument("--skip-xml", action="store_true", help="download: skip the 20GB+ XML dump")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(args.config.read_text())

    if args.stage in ("download", "all"):
        download(args.dumps_dir, with_xml=not args.skip_xml)
    if args.stage in ("graph", "all"):
        run_graph(cfg, args.dumps_dir, args.out_dir, args.max_edges)
    if args.stage in ("citations", "all"):
        run_citations(cfg, args.dumps_dir, args.out_dir, args.seed_index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
