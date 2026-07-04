#!/usr/bin/env python3
"""Extract Stack Exchange ``.7z`` dumps into canonical Q&A JSONL(.zst).

Stage-2 (Extract) of the data funnel for the ``dump``-route Stack Exchange
sources (doc §1.2). Reads the archives acquired by ``scripts/acquire/acquire.py``
(``<staging>/stackexchange-dumps/*.7z``) and writes, per site, sharded canonical
records a ``kind: jsonl`` DocumentSource can read straight into the pipeline.

Typical (after the P0 StackExchange download lands in /data):
  uv run --extra data python scripts/extract_stackexchange.py \
      --input-dir /data/corpus-staging/stackexchange-dumps \
      --out /data/corpus-staging/stackexchange-extracted

  # one archive, quick smoke with a row cap is not supported here — use --limit
  uv run --extra data python scripts/extract_stackexchange.py \
      --archive /data/corpus-staging/stackexchange-dumps/physics.stackexchange.com.7z \
      --out /data/corpus-staging/stackexchange-extracted

Idempotent: a site whose ``_manifest.json`` already exists in --out is skipped
(pass --force to re-extract). Stack Overflow is the outlier (~100 GB of XML);
point --tmpdir at a volume with room for the staging db if --out is tight.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lithos.data.stackexchange import ExtractParams, extract_archive, infer_site  # noqa: E402

log = logging.getLogger("extract_se")


def _archives(args: argparse.Namespace) -> list[Path]:
    if args.archive:
        return [Path(a) for a in args.archive]
    root = Path(args.input_dir)
    found = sorted(root.glob("*.7z"))
    if not found:
        raise SystemExit(f"no .7z archives in {root}")
    return found


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--archive", action="append", help="one .7z archive (repeatable)")
    src.add_argument("--input-dir", help="directory of *.7z archives to extract")
    p.add_argument("--out", required=True, help="output root (per-site subdirs written under it)")
    p.add_argument("--tmpdir", default=None, help="dir for the staging db (default: the site out dir)")
    p.add_argument("--min-answer-score", type=int, default=1)
    p.add_argument("--min-question-score", type=int, default=None)
    p.add_argument("--max-answers", type=int, default=5)
    p.add_argument("--keep-unanswered", action="store_true",
                   help="emit questions with no kept answer (default: skip)")
    p.add_argument("--license", default="cc-by-sa-4.0")
    p.add_argument("--shard-size", type=int, default=50_000)
    p.add_argument("--force", action="store_true", help="re-extract even if a manifest exists")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    params = ExtractParams(
        min_answer_score=args.min_answer_score,
        min_question_score=args.min_question_score,
        max_answers=args.max_answers,
        require_answer=not args.keep_unanswered,
        license=args.license,
    )
    out_root = Path(args.out)
    archives = _archives(args)
    log.info("extracting %d archive(s) → %s", len(archives), out_root)

    failures: list[str] = []
    for arc in archives:
        site = infer_site(arc)
        manifest = out_root / site / "_manifest.json"
        if manifest.exists() and not args.force:
            log.info("[%s] manifest exists — skipping (use --force)", site)
            continue
        try:
            log.info("[%s] extracting from %s", site, arc.name)
            m = extract_archive(arc, out_root, params=params, tmpdir=args.tmpdir,
                                shard_size=args.shard_size)
            log.info("[%s] %d questions in → %d docs out (%d answers kept), %d shard(s)",
                     site, m["questions_in"], m["documents_out"], m["answers_kept"],
                     len(m["files"]))
        except Exception as e:  # keep going; report at the end
            log.error("[%s] FAILED: %s", site, e)
            failures.append(site)

    if failures:
        log.error("failed: %s", ", ".join(failures))
        return 1
    log.info("done — %d archive(s)", len(archives))
    print(json.dumps({"extracted": [infer_site(a) for a in archives]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
