"""Tests for the acquisition planner (scripts/acquire/acquire.py)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "acquire", Path(__file__).parent.parent / "scripts" / "acquire" / "acquire.py"
)
assert spec and spec.loader
acquire = importlib.util.module_from_spec(spec)
sys.modules["acquire"] = acquire  # dataclasses needs the module registered
spec.loader.exec_module(acquire)

CFG = {
    "waves": {"p0": ["alpha", "dumps"]},
    "specs": {
        "alpha": {"route": "hf", "repo": "org/alpha", "include": ["web/*"], "est_gb": 100},
        "dumps": {"route": "dump", "urls": ["https://x/a.7z", "https://x/b.7z"],
                  "est_gb": 50, "index_ids": ["a-row", "b-row"]},
        "gated1": {"route": "hf", "repo": "org/gated", "est_gb": 10,
                   "notes": "GATED - request access"},
        "unindexed": {"route": "hf", "repo": "org/x", "est_gb": 1},
        "badroute": {"route": "ftp", "est_gb": 1},
    },
}
INDEX_IDS = {"alpha", "a-row", "b-row", "gated1", "badroute"}
SCRATCH = Path("/scratch")


def test_wave_plan_builds_commands(monkeypatch):
    # Pin the dump downloader to aria2c regardless of what's installed on the box.
    monkeypatch.setattr(acquire.shutil, "which", lambda name: "/usr/bin/aria2c")
    jobs = acquire.build_plan(INDEX_IDS, CFG, wave="p0", scratch=SCRATCH)
    assert [j.id for j in jobs] == ["alpha", "dumps"]
    hf = jobs[0].download_argv[0]
    assert hf[:3] == ["hf", "download", "org/alpha"]
    assert "--include" in hf and "web/*" in hf
    assert str(SCRATCH / "alpha") in hf
    assert len(jobs[1].download_argv) == 2  # one aria2c per url
    assert jobs[1].download_argv[0][0] == "aria2c"


def test_dump_route_falls_back_to_wget_without_aria2c(monkeypatch):
    # When aria2c is absent, the dump route degrades to wget -c (still resumable).
    monkeypatch.setattr(acquire.shutil, "which", lambda name: None)
    jobs = acquire.build_plan(INDEX_IDS, CFG, only_ids=["dumps"], scratch=SCRATCH)
    argv = jobs[0].download_argv
    assert len(argv) == 2  # one wget per url
    assert argv[0][0] == "wget"
    assert "--continue" in argv[0]
    assert argv[0][-1] == "https://x/a.7z"


def test_index_ids_override_maps_to_index_rows():
    jobs = acquire.build_plan(INDEX_IDS, CFG, wave="p0", scratch=SCRATCH)
    assert jobs[1].index_rows == ["a-row", "b-row"]


def test_unindexed_id_rejected():
    with pytest.raises(ValueError, match="not in seed_index"):
        acquire.build_plan(INDEX_IDS, CFG, only_ids=["unindexed"], scratch=SCRATCH)


def test_gated_flag_from_notes():
    jobs = acquire.build_plan(INDEX_IDS, CFG, only_ids=["gated1"], scratch=SCRATCH)
    assert jobs[0].gated is True


def test_unknown_wave_and_route_and_missing_spec():
    with pytest.raises(ValueError, match="unknown wave"):
        acquire.build_plan(INDEX_IDS, CFG, wave="p9", scratch=SCRATCH)
    with pytest.raises(ValueError, match="unsupported route"):
        acquire.build_plan(INDEX_IDS, CFG, only_ids=["badroute"], scratch=SCRATCH)
    with pytest.raises(ValueError, match="no spec"):
        acquire.build_plan(INDEX_IDS, CFG, only_ids=["nope"], scratch=SCRATCH)
    with pytest.raises(ValueError, match="--wave or --id"):
        acquire.build_plan(INDEX_IDS, CFG, scratch=SCRATCH)


def test_download_upload_split(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    def fake_run(argv):
        calls.append(argv)
        if argv[0] == "hf":  # simulate the download populating scratch
            d = Path(argv[argv.index("--local-dir") + 1])
            d.mkdir(parents=True, exist_ok=True)
            (d / "part0.parquet").write_bytes(b"x" * 64)

    monkeypatch.setattr(acquire, "_run", fake_run)
    monkeypatch.setattr(acquire, "manifest_exists", lambda dest, id_: False)
    job = acquire.build_plan(INDEX_IDS, CFG, only_ids=["alpha"], scratch=tmp_path)[0]

    # download phase: no rclone, manifest written locally
    acquire.execute(job, None, tmp_path, mode="download")
    assert (tmp_path / "alpha" / "_manifest.json").exists()
    assert all(a[0] != "rclone" for a in calls)
    # idempotent
    n = len(calls)
    acquire.execute(job, None, tmp_path, mode="download")
    assert len(calls) == n

    # upload phase: rclone copy + manifest copyto, bwlimit propagated
    acquire.execute(job, "r2:bucket", tmp_path, mode="upload", bwlimit="10M")
    rclone = [a for a in calls if a[0] == "rclone"]
    assert len(rclone) == 2
    assert "--bwlimit" in rclone[0] and "10M" in rclone[0]
    assert rclone[1][:2] == ["rclone", "copyto"]
    assert not (tmp_path / "alpha").exists()  # scratch cleaned after upload


def test_upload_without_download_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(acquire, "manifest_exists", lambda dest, id_: False)
    job = acquire.build_plan(INDEX_IDS, CFG, only_ids=["alpha"], scratch=tmp_path)[0]
    with pytest.raises(RuntimeError, match="download first"):
        acquire.execute(job, "r2:bucket", tmp_path, mode="upload")


def test_r2_dest_parsing():
    assert acquire.r2_dest("s3://bucket") == "r2:bucket"
    assert acquire.r2_dest("s3://bucket/prefix/") == "r2:bucket/prefix"
    with pytest.raises(ValueError):
        acquire.r2_dest("gs://bucket")


def test_real_config_plans_all_waves_against_real_index():
    """The committed acquisition.yaml must be internally consistent with the
    committed seed_index.csv — every wave plans without error."""
    import yaml

    cfg = yaml.safe_load((Path(__file__).parent.parent / "corpus" / "acquisition.yaml").read_text())
    index_ids = acquire.load_index_ids()
    for wave in cfg["waves"]:
        jobs = acquire.build_plan(index_ids, cfg, wave=wave, scratch=SCRATCH)
        assert jobs, wave
        assert all(j.est_gb > 0 for j in jobs)
