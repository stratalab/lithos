#!/usr/bin/env python3
"""Train the v0 quality classifier for a domain (docs/quality-classifiers.md §4).

Joins labels (data/labels/*.jsonl) with document texts and trains the hashed
n-gram linear model. Texts come from --texts (a jsonl with matching ids) or
--recover-hf: re-stream the same shuffled HF source the labels came from and
match records by the content-hash doc_id (streaming with a fixed seed is
deterministic, so the pilot's unsaved texts are recoverable).

  uv run python scripts/train_quality_classifier.py --domain physics-eng \\
      --labels data/labels/pilot-physics-eng.jsonl --texts data/pilot/physics-eng.jsonl
  uv run python scripts/train_quality_classifier.py --domain math \\
      --labels data/labels/pilot-math.jsonl --recover-hf HuggingFaceTB/finemath:finemath-3plus
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lithos.data.overlap import TEXT_FIELD_CANDIDATES, get_field
from lithos.data.quality_classifier import train

log = logging.getLogger("train-quality")


def load_labels(path: Path) -> tuple[dict[str, int], int]:
    labels = {}
    version = None
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            labels[r["doc_id"]] = r["score"]
            version = r["rubric_version"]
    return labels, version


def texts_from_jsonl(path: Path, wanted: set[str]) -> dict[str, str]:
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("id") in wanted:
                out[r["id"]] = r["text"]
    return out


def texts_from_hf(spec: str, wanted: set[str], *, scan_limit: int = 60_000) -> dict[str, str]:
    """Re-stream the labeling source (same seed) and match content hashes."""
    from scripts.label_quality import iter_hf  # same shuffle seed as labeling

    out: dict[str, str] = {}
    for i, rec in enumerate(iter_hf(spec)):
        text = next(
            (t for f in TEXT_FIELD_CANDIDATES if isinstance(t := get_field(rec, f), str)), None
        )
        if not text:
            continue
        doc_id = str(rec.get("id") or hashlib.sha1(text.encode()).hexdigest()[:16])
        if doc_id in wanted:
            out[doc_id] = text
        if len(out) >= len(wanted) or i >= scan_limit:
            break
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", required=True)
    p.add_argument("--labels", type=Path, required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--texts", type=Path, help="jsonl with id+text (from --save-texts)")
    src.add_argument("--recover-hf", help="re-stream this HF source to recover texts")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--dim-bits", type=int, default=18)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    labels, version = load_labels(args.labels)
    wanted = set(labels)
    if args.texts:
        texts = texts_from_jsonl(args.texts, wanted)
    else:
        texts = texts_from_hf(args.recover_hf, wanted)
    matched = [(texts[i], labels[i]) for i in wanted if i in texts]
    log.info("[%s] %d/%d labels matched to texts", args.domain, len(matched), len(labels))
    if len(matched) < 20:
        log.error("too few matches to train")
        return 1

    model = train([t for t, _ in matched], [s for _, s in matched],
                  domain=args.domain, rubric_version=version, dim_bits=args.dim_bits)
    out = args.out or Path(f"data/classifiers/{args.domain}-v0.npz")
    model.save(out)
    log.info("[%s] saved → %s", args.domain, out)
    print(json.dumps({"domain": args.domain, **model.metrics}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
