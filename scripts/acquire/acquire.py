#!/usr/bin/env python3
"""Acquire raw corpora onto R2 (the VM→R2 pattern; bulk never transits home).

Reads corpus/seed_index.csv (the enforcement surface) + corpus/acquisition.yaml
(per-id mechanics), builds a plan, and per item: download to scratch → upload
to <R2>/raw/<id>/ → write a provenance manifest → clean scratch. Idempotent:
items whose manifest already exists on R2 are skipped.

Typical (on the acquisition VM, after scripts/acquire/bootstrap_vm.sh):
  uv run python scripts/acquire/acquire.py --wave p0 --dry-run   # plan + sizes
  uv run python scripts/acquire/acquire.py --wave p0             # execute
  uv run python scripts/acquire/acquire.py --id megamath         # one item

Doctrine hooks: every id must exist in seed_index.csv (or declare index_ids —
post-training sets live in doc §2.2 instead); the manifest records source,
revision-ish info, byte/file counts, and timestamps for provenance (§0.4).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

log = logging.getLogger("acquire")

INDEX = REPO / "corpus" / "seed_index.csv"
SPECS = REPO / "corpus" / "acquisition.yaml"


@dataclass
class Job:
    id: str
    route: str  # hf | dump
    est_gb: float
    download_argv: list[list[str]]  # commands that populate scratch/<id>/
    notes: str = ""
    gated: bool = False
    index_rows: list[str] = field(default_factory=list)


def r2_dest(base_uri: str) -> str:
    """LITHOS_STORAGE_BASE_URI (s3://bucket[/prefix]) → rclone r2:bucket[/prefix]."""
    if not base_uri.startswith("s3://"):
        raise ValueError(f"expected s3:// uri, got {base_uri!r}")
    return "r2:" + base_uri.removeprefix("s3://").rstrip("/")


def build_plan(
    index_ids: set[str],
    cfg: dict,
    *,
    wave: str | None = None,
    only_ids: list[str] | None = None,
    scratch: Path,
) -> list[Job]:
    specs: dict = cfg["specs"]
    if wave is not None:
        if wave not in cfg["waves"]:
            raise ValueError(f"unknown wave {wave!r}; have {list(cfg['waves'])}")
        ids = list(cfg["waves"][wave])
    elif only_ids:
        ids = list(only_ids)
    else:
        raise ValueError("pass --wave or --id")

    jobs: list[Job] = []
    for id_ in ids:
        if id_ not in specs:
            raise ValueError(f"{id_!r} has no spec in acquisition.yaml")
        spec = specs[id_]
        # Doctrine check: the id (or its declared index_ids) must be indexed.
        rows = spec.get("index_ids", [id_]) or []
        missing = [r for r in rows if r not in index_ids]
        if missing:
            raise ValueError(f"{id_}: not in seed_index.csv: {missing} — index before acquiring")

        dest_dir = scratch / id_
        if spec["route"] == "hf":
            argv = ["hf", "download", spec["repo"], "--repo-type", "dataset",
                    "--local-dir", str(dest_dir)]
            for pat in spec.get("include", []):
                argv += ["--include", pat]
            download = [argv]
        elif spec["route"] == "dump":
            # Prefer aria2c (multi-connection, resumable); fall back to wget -c
            # where aria2c isn't installed. Both do HTTP range-resume on the same
            # files, so switching mid-download is safe.
            if shutil.which("aria2c"):
                download = [
                    ["aria2c", "-x8", "-s8", "--continue=true", "-d", str(dest_dir), url]
                    for url in spec["urls"]
                ]
            else:
                download = [
                    ["wget", "--continue", "--tries=0", "--progress=dot:giga",
                     "-P", str(dest_dir), url]
                    for url in spec["urls"]
                ]
        else:
            raise ValueError(f"{id_}: unsupported route {spec['route']!r}")

        jobs.append(Job(
            id=id_, route=spec["route"], est_gb=float(spec.get("est_gb", 0)),
            download_argv=download, notes=spec.get("notes", ""),
            gated="GATED" in spec.get("notes", ""), index_rows=rows,
        ))
    return jobs


def load_index_ids(path: Path = INDEX) -> set[str]:
    with open(path, newline="") as f:
        return {row["id"] for row in csv.DictReader(f)}


def _run(argv: list[str]) -> None:
    log.info("$ %s", " ".join(argv))
    subprocess.run(argv, check=True)


def _dir_stats(root: Path) -> tuple[int, int]:
    files = [p for p in root.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def manifest_exists(dest: str, id_: str) -> bool:
    r = subprocess.run(["rclone", "lsf", f"{dest}/raw/{id_}/_manifest.json"],
                       capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() != ""


def download_item(job: Job, scratch: Path) -> Path:
    """Download into scratch/<id>/ and write the provenance manifest locally."""
    local = scratch / job.id
    local.mkdir(parents=True, exist_ok=True)
    for argv in job.download_argv:
        _run(argv)
    n_files, n_bytes = _dir_stats(local)
    if n_files == 0:
        raise RuntimeError(f"{job.id}: download produced no files")
    manifest = {
        "id": job.id, "route": job.route, "index_rows": job.index_rows,
        "commands": [" ".join(a) for a in job.download_argv],
        "files": n_files, "bytes": n_bytes,
        "acquired_at": datetime.now(UTC).isoformat(),
        "notes": job.notes,
    }
    (local / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("[%s] downloaded %d files, %.1f GB → %s", job.id, n_files, n_bytes / 1e9, local)
    return local


def upload_item(
    job: Job, dest: str, scratch: Path, *, keep_scratch: bool = False, bwlimit: str | None = None
) -> None:
    """Push scratch/<id>/ (+ manifest) to <dest>/raw/<id>/, then clean scratch."""
    local = scratch / job.id
    mpath = local / "_manifest.json"
    if not local.is_dir() or not any(local.iterdir()):
        raise RuntimeError(f"{job.id}: nothing in {local} — download first")
    if not mpath.exists():  # e.g. downloaded before the split existed
        n_files, n_bytes = _dir_stats(local)
        mpath.write_text(json.dumps({
            "id": job.id, "route": job.route, "index_rows": job.index_rows,
            "files": n_files, "bytes": n_bytes,
            "acquired_at": datetime.now(UTC).isoformat(), "notes": job.notes,
        }, indent=2))
    limit = ["--bwlimit", bwlimit] if bwlimit else []
    _run(["rclone", "copy", str(local), f"{dest}/raw/{job.id}/",
          "--transfers", "8", "--checkers", "16", "--progress", *limit,
          "--exclude", "_manifest.json"])
    # Manifest lands last: its presence on R2 marks the mirror complete.
    _run(["rclone", "copyto", str(mpath), f"{dest}/raw/{job.id}/_manifest.json", *limit])
    if not keep_scratch:
        shutil.rmtree(local)
        log.info("[%s] scratch cleaned", job.id)


def execute(
    job: Job, dest: str | None, scratch: Path, *,
    mode: str = "full",  # full | download | upload
    keep_scratch: bool = False, bwlimit: str | None = None,
) -> None:
    if mode in ("full", "upload"):
        assert dest is not None
        if manifest_exists(dest, job.id):
            log.info("[%s] manifest already on R2 — skipping", job.id)
            return
    if mode in ("full", "download"):
        if mode == "download" and (scratch / job.id / "_manifest.json").exists():
            log.info("[%s] already downloaded — skipping", job.id)
            return
        download_item(job, scratch)
    if mode in ("full", "upload"):
        assert dest is not None
        upload_item(job, dest, scratch, keep_scratch=keep_scratch, bwlimit=bwlimit)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wave", choices=["p0", "posttrain", "physeng", "p1"])
    p.add_argument("--id", action="append", dest="ids")
    # Prefer the big data volume for scratch when present (falls back to $HOME).
    _default_scratch = Path("/data/corpus-staging") if Path("/data").is_mount() \
        else Path.home() / "acquire-scratch"
    p.add_argument("--scratch", type=Path, default=_default_scratch)
    p.add_argument("--dest", default=None,
                   help="rclone dest (default: from LITHOS_STORAGE_BASE_URI)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--keep-scratch", action="store_true")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--no-upload", action="store_true",
                      help="download to scratch only (upload later)")
    mode.add_argument("--upload-only", action="store_true",
                      help="push previously-downloaded scratch dirs to R2")
    p.add_argument("--bwlimit", default=None,
                   help='rclone upload cap, e.g. "10M" or "08:00,2M 23:00,off" '
                        "(for trickle-uploading over a home connection)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(SPECS.read_text())
    jobs = build_plan(load_index_ids(), cfg, wave=args.wave, only_ids=args.ids,
                      scratch=args.scratch)

    total = sum(j.est_gb for j in jobs)
    print(f"\nPlan — {len(jobs)} item(s), est. {total:,.0f} GB "
          f"(~${total / 1000 * 15:.0f}/mo on R2):")
    for j in jobs:
        flags = " [GATED]" if j.gated else ""
        print(f"  {j.id:<28} {j.route:<5} ~{j.est_gb:>6,.0f} GB{flags}  {j.notes}")
    if args.dry_run:
        return 0

    mode_name = "download" if args.no_upload else "upload" if args.upload_only else "full"
    dest = None
    if mode_name != "download":
        dest = args.dest or r2_dest(os.environ["LITHOS_STORAGE_BASE_URI"])
    args.scratch.mkdir(parents=True, exist_ok=True)
    failures = []
    for j in jobs:
        try:
            execute(j, dest, args.scratch, mode=mode_name,
                    keep_scratch=args.keep_scratch or args.no_upload, bwlimit=args.bwlimit)
        except Exception as e:  # keep going; report at the end
            log.error("[%s] FAILED: %s", j.id, e)
            failures.append(j.id)
    if failures:
        log.error("failed: %s", ", ".join(failures))
        return 1
    log.info("wave complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
