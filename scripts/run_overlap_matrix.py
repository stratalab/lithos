#!/usr/bin/env python3
"""Estimate the pairwise overlap matrix for the math corpora (doc §1.8 warning).

Streams a shuffled sample from each corpus over HF `datasets` (no bulk
download — ~1-3GB total network for the default sample size), caches per-corpus
signatures under --work-dir, then cross-matches all pairs and writes report.md.

Typical:
  uv run python scripts/run_overlap_matrix.py --list          # inspect configs/fields
  uv run python scripts/run_overlap_matrix.py --sample-size 2000   # smoke
  uv run python scripts/run_overlap_matrix.py                 # full (200k/corpus)

Cached samples are reused (delete <work-dir>/<name>.npz to re-stream one).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lithos.data.minhash import MinHasher
from lithos.data.overlap import (
    TEXT_FIELD_CANDIDATES,
    URL_FIELD_CANDIDATES,
    CorpusSpec,
    SampleSigs,
    build_sample,
    format_report,
    iter_pairs,
    pair_overlap,
)

log = logging.getLogger("overlap")

# The four CC-mined math corpora from corpus/seed_index.csv. total_docs are
# refreshed from the datasets-server size API at runtime when reachable.
CORPORA = [
    CorpusSpec(name="openwebmath", hf_id="open-web-math/open-web-math",
               total_docs=6_315_233),
    CorpusSpec(name="finemath", hf_id="HuggingFaceTB/finemath",
               config_prefer="finemath-3plus", total_docs=21_400_000),
    # GATED — request access at the dataset page (stratalab account), then it
    # streams like the others.
    CorpusSpec(name="nemotron-cc-math", hf_id="nvidia/Nemotron-CC-Math-v1",
               config_prefer="4plus", total_docs=52_000_000, total_docs_approx=True),
    # Organized by subdirectory, not configs; megamath-web is the CC-mined text.
    # No per-dir row count from the size API → static estimate, flagged approx.
    CorpusSpec(name="megamath", hf_id="LLM360/MegaMath", data_dir="megamath-web",
               total_docs=121_000_000, total_docs_approx=True),
]


def resolve_config(spec: CorpusSpec) -> str | None:
    from datasets import get_dataset_config_names

    if spec.data_dir:
        return None  # data_dir datasets bypass config resolution
    names = get_dataset_config_names(spec.hf_id)
    if spec.config:
        return spec.config
    if len(names) <= 1:
        return names[0] if names else None
    if spec.config_prefer:
        for n in names:
            if spec.config_prefer in n:
                return n
    log.warning("[%s] multiple configs %s — using %r", spec.name, names, names[0])
    return names[0]


def fetch_total_docs(spec: CorpusSpec, config: str | None) -> int:
    """datasets-server size API; fall back to the spec's static estimate."""
    if spec.data_dir:
        # size API is config-scoped and would count ALL subdirs — use the static
        # per-dir estimate (flagged approx in the report).
        assert spec.total_docs is not None
        return spec.total_docs
    url = f"https://datasets-server.huggingface.co/size?dataset={spec.hf_id}"
    if config:
        url += f"&config={config}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.load(r)
        node = data["size"]["config"] if config else data["size"]["dataset"]
        total = int(node["num_rows"])
        log.info("[%s] total_docs=%s (datasets-server)", spec.name, f"{total:,}")
        return total
    except Exception as e:
        if spec.total_docs is None:
            raise RuntimeError(f"{spec.name}: no total_docs and size API failed: {e}") from e
        log.warning("[%s] size API failed (%s) — using static %s", spec.name, e,
                    f"{spec.total_docs:,}")
        return spec.total_docs


def detect_fields(record: dict, spec: CorpusSpec) -> tuple[str, str | None]:
    text_field = spec.text_field or next(
        (f for f in TEXT_FIELD_CANDIDATES if isinstance(record.get(f), str)), None
    )
    if text_field is None:
        raise RuntimeError(f"{spec.name}: no text field among {list(record)}")
    url_field = spec.url_field or next(
        (f for f in URL_FIELD_CANDIDATES if isinstance(record.get(f), str)), None
    )
    return text_field, url_field


def stream_sample(spec: CorpusSpec, sample_size: int, work_dir: Path) -> SampleSigs:
    import numpy as np

    cache = work_dir / f"{spec.name}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        n = int(z["sigs"].shape[0])
        if n >= sample_size:
            log.info("[%s] cached sample (%d docs) — reusing", spec.name, n)
            return SampleSigs(
                name=spec.name, total_docs=int(z["total_docs"]), sigs=z["sigs"],
                text_hashes=z["text_hashes"],
                url_hashes=z["url_hashes"] if "url_hashes" in z.files else None,
            )
        log.info("[%s] cache too small (%d < %d) — re-streaming", spec.name, n, sample_size)

    from datasets import load_dataset

    config = resolve_config(spec)
    total = fetch_total_docs(spec, config)
    ds = load_dataset(
        spec.hf_id, name=config, data_dir=spec.data_dir, split=spec.split, streaming=True
    )
    ds = ds.shuffle(seed=1, buffer_size=10_000)
    it = iter(ds)
    first = next(it)
    text_field, url_field = detect_fields(first, spec)
    log.info("[%s] config=%s text_field=%r url_field=%r", spec.name, config, text_field, url_field)

    def docs():
        yield first
        yield from it

    sample = build_sample(
        spec.name,
        tqdm(docs(), total=sample_size, desc=spec.name, unit="doc"),
        total_docs=total, sample_size=sample_size,
        text_field=text_field, url_field=url_field,
        hasher=MinHasher(),
    )
    arrays = {"sigs": sample.sigs, "text_hashes": sample.text_hashes,
              "total_docs": np.asarray(total)}
    if sample.url_hashes is not None:
        arrays["url_hashes"] = sample.url_hashes
    np.savez_compressed(cache, **arrays)
    log.info("[%s] sampled %d docs → %s", spec.name, sample.n, cache)
    return sample


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample-size", type=int, default=200_000)
    p.add_argument("--work-dir", type=Path, default=Path("data/overlap"))
    p.add_argument("--corpus", action="append", help="restrict to named corpora")
    p.add_argument("--list", action="store_true", help="print configs + first-record fields")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    specs = [s for s in CORPORA if not args.corpus or s.name in args.corpus]

    if args.list:
        from datasets import load_dataset

        for spec in specs:
            config = resolve_config(spec)
            ds = load_dataset(
                spec.hf_id, name=config, data_dir=spec.data_dir, split=spec.split, streaming=True
            )
            rec = next(iter(ds))
            fields = {k: type(v).__name__ for k, v in rec.items()}
            print(f"{spec.name}: config={config} data_dir={spec.data_dir} fields={fields}")
        return 0

    args.work_dir.mkdir(parents=True, exist_ok=True)
    samples = [stream_sample(s, args.sample_size, args.work_dir) for s in specs]

    approx = {s.name for s in specs if s.total_docs_approx}
    results = []
    for a, b in iter_pairs(samples):
        log.info("matching %s x %s ...", a.name, b.name)
        r = pair_overlap(a, b)
        for name in approx & {r.a, r.b}:
            r.notes.append(f"N_{name} is approximate — estimates toward it scale with it")
        results.append(r)

    report = format_report(results)
    out = args.work_dir / "report.md"
    out.write_text(report)
    print("\n" + report)
    log.info("report → %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
