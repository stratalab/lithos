"""Freeze + publish the TIR battery's post-cutoff slice as a public benchmark
(docs/eval-tir-battery-plan.md Part B; `eval-plan.md` §6).

Turns the internal hold-out into a *credibly* self-published benchmark. Credibility is
structural — §6's four suspicions each get a concrete artifact here, not a promise:

1. **"you trained on it"** → a **canary GUID** embedded in the set (grep-able in any
   corpus) + a **time-partition cutoff** (every task post-dates the cutoff) + every
   prompt emitted to the decontam probe list so the corpus build screens it out.
2. **"you graded it favourably"** → **executable grading only**: the set ships the
   answers/tests and the harness is the grader (no LLM judge anywhere).
3. **"you designed it around your model"** → a **version-locked content hash**; scores
   compare only within a version, and the version publishes *before* results.
4. **"sandbagged baselines"** → a **leaderboard** the anchors run through the identical
   `tir_battery` loop, **losses included** (a model that beats us stays on the board).

The freeze/hash/canary/card/leaderboard machinery is model-free (built + tested now);
populating the leaderboard is just Part A (`lithos tir-battery`) run per anchor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lithos.data.decontam import write_probes
from lithos.posttrain.taskbank import Task, split_by_year
from lithos.utils.io import atomic_write_text, ensure_dir, sha256_bytes, write_json

_CANARY_MARKER = "BENCHMARK DATA SHOULD NOT APPEAR IN TRAINING CORPORA. lithos-tir canary GUID"
_DEFAULT_TOL = 1e-6


def canary_line(guid: str) -> str:
    """The canary string embedded in the published set — its presence in a training
    corpus is direct evidence of contamination (the BIG-bench / GPQA convention)."""
    return f"{_CANARY_MARKER} {guid}"


def find_canary(text: str, guid: str) -> bool:
    """True iff this benchmark's canary GUID appears in ``text`` — the detection side."""
    return guid in text


def frozen_task(task: Task) -> dict[str, Any]:
    """A task's published record: exactly the fields the harness needs to score it,
    round-tripping through ``taskbank.task_from_record``. Optional fields are omitted
    when unset (a written ``null`` would decode to the string ``"None"``), which also
    keeps the content hash canonical."""
    rec: dict[str, Any] = {"id": task.id, "prompt": task.prompt, "kind": task.kind}
    if task.kind == "code":
        rec["tests"] = task.tests
    else:
        rec["answer"] = task.answer
    if task.units is not None:
        rec["units"] = task.units
    if task.tol != _DEFAULT_TOL:
        rec["tol"] = task.tol
    if task.level is not None:
        rec["level"] = task.level
    if task.year is not None:
        rec["year"] = task.year
    if task.family_id is not None:
        rec["family_id"] = task.family_id
    if task.metadata:
        rec["metadata"] = task.metadata
    return rec


def content_sha256(frozen: list[dict[str, Any]]) -> str:
    """Version-integrity hash over the canonical serialization of the frozen tasks —
    order-independent (sorted by id) and key-canonical, so the same content always
    yields the same version and any edit changes it."""
    canon = json.dumps(
        sorted(frozen, key=lambda r: r["id"]), sort_keys=True, ensure_ascii=False
    )
    return sha256_bytes(canon.encode("utf-8"))


def _breakdown(frozen: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    def _count(key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in frozen:
            k = str(r.get(key, "unspecified"))
            counts[k] = counts.get(k, 0) + 1
        return dict(sorted(counts.items()))

    domains: dict[str, int] = {}
    for r in frozen:
        d = str((r.get("metadata") or {}).get("domain", "unspecified"))
        domains[d] = domains.get(d, 0) + 1
    return {
        "by_kind": _count("kind"),
        "by_level": _count("level"),
        "by_year": _count("year"),
        "by_domain": dict(sorted(domains.items())),
    }


@dataclass(frozen=True)
class BenchmarkArtifact:
    """A frozen, version-locked benchmark ready to write to disk."""

    manifest: dict[str, Any]
    frozen_tasks: list[dict[str, Any]]
    canary_guid: str


def freeze_benchmark(
    tasks: list[Task],
    *,
    version: str,
    cutoff_year: int,
    canary_guid: str,
    created_at: str,
    name: str = "lithos-tir",
    license_id: str = "CC-BY-4.0",
) -> BenchmarkArtifact:
    """Freeze the **post-cutoff** slice of ``tasks`` into a versioned benchmark.

    The published set is exactly the ``split_by_year`` hold-out (family-aware), so every
    task provably post-dates ``cutoff_year`` — contamination is impossible by
    construction, not by inspection. Raises if the slice is empty.
    """
    _, hold = split_by_year(tasks, cutoff_year)
    if not hold:
        raise ValueError(
            f"no post-{cutoff_year} tasks to publish — the benchmark would be empty; "
            "check the bank's `year` stamps or lower cutoff_year"
        )
    frozen = sorted((frozen_task(t) for t in hold), key=lambda r: r["id"])
    manifest = {
        "benchmark": name,
        "version": version,
        "content_sha256": content_sha256(frozen),
        "canary_guid": canary_guid,
        "canary": canary_line(canary_guid),
        "cutoff_year": cutoff_year,
        "num_tasks": len(frozen),
        "license": license_id,
        "created_at": created_at,
        "grading": "executable-only",
        "breakdown": _breakdown(frozen),
    }
    return BenchmarkArtifact(manifest=manifest, frozen_tasks=frozen, canary_guid=canary_guid)


def benchmark_probe_texts(artifact: BenchmarkArtifact) -> list[str]:
    """Texts to register in the training decontam probe list: every prompt (the
    contamination-relevant content) plus the canary line itself."""
    return [r["prompt"] for r in artifact.frozen_tasks] + [artifact.manifest["canary"]]


def render_readme(manifest: dict[str, Any]) -> str:
    """The benchmark card — the four suspicions (§6) each answered with an artifact."""
    b = manifest["breakdown"]
    lines = [
        f"# {manifest['benchmark']} — {manifest['version']}",
        "",
        "A compact-STEM **tool-integrated reasoning (TIR)** benchmark: each problem is",
        "scored twice — reasoning **with** a code sandbox vs **without** — and the",
        "headline metric is **tool-uplift** (verified solve-rate with tools − without),",
        "reported per difficulty tier. Executable grading only; no LLM judge.",
        "",
        "## Provenance",
        f"- **version / content hash**: `{manifest['content_sha256']}` "
        "(scores compare only within this hash)",
        f"- **cutoff year**: {manifest['cutoff_year']} — every task post-dates it "
        "(contamination-resistant by construction)",
        f"- **tasks**: {manifest['num_tasks']}  ·  **license**: {manifest['license']}"
        f"  ·  **created**: {manifest['created_at']}",
        f"- **canary GUID**: `{manifest['canary_guid']}` — see `canary.txt`; its presence "
        "in a training corpus is direct evidence of contamination",
        "",
        "## Composition",
        f"- by kind: {b['by_kind']}",
        f"- by level: {b['by_level']}",
        f"- by domain: {b['by_domain']}",
        f"- by year: {b['by_year']}",
        "",
        "## Why you can trust it (the four suspicions, answered structurally)",
        "1. **\"You trained on it.\"** Every task post-dates the cutoff, the canary GUID is",
        "   embedded, and every prompt ships in `decontam_probes.jsonl` for corpus screening.",
        "2. **\"You graded it favourably.\"** Grading is executable only — the harness runs",
        "   the tool code / checks the value; there is no judge anywhere in scoring.",
        "3. **\"You designed it around your model.\"** The set is version-locked by content",
        "   hash and published before results; coverage follows the curriculum taxonomy, not",
        "   model behaviour.",
        "4. **\"Sandbagged baselines.\"** Baselines run through the identical harness with",
        "   losses published — a model that beats us stays on the leaderboard.",
        "",
        "## Run it",
        "```",
        "lithos tir-battery --config configs/eval/tir.yaml --checkpoint <model> \\",
        "  --override tir.task_bank=benchmark.jsonl tir.cutoff_year=0",
        "```",
        "(cutoff 0 scores the whole published set — it is already the post-cutoff slice.)",
        "",
    ]
    return "\n".join(lines)


def write_benchmark(out_dir: str | Path, artifact: BenchmarkArtifact) -> Path:
    """Write the publishable bundle: tasks, manifest, canary, README, decontam probes."""
    out = ensure_dir(out_dir)
    with (out / "benchmark.jsonl").open("w", encoding="utf-8") as f:
        for rec in artifact.frozen_tasks:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    write_json(out / "manifest.json", artifact.manifest)
    atomic_write_text(out / "canary.txt", artifact.manifest["canary"] + "\n")
    atomic_write_text(out / "README.md", render_readme(artifact.manifest))
    write_probes(out / "decontam_probes.jsonl", benchmark_probe_texts(artifact))
    return out


def render_leaderboard(entries: list[dict[str, Any]], *, sort_key: str = "solve_on") -> str:
    """Render the parity leaderboard from scorecard rows carrying a ``tir`` block (the
    output of `lithos tir-battery` per model). **Losses included** — every model that
    ran appears, sorted by tools-on solve rate, so a competitor win stays visible."""
    rows = []
    versions = set()
    for e in entries:
        tir = e.get("tir") or {}
        ov = tir.get("overall") or {}
        versions.add(tir.get("battery_version") or e.get("battery_version"))
        rows.append(
            {
                "label": e.get("label", "?"),
                "n": tir.get("n", 0),
                "solve_off": float(ov.get("solve_off", 0.0)),
                "solve_on": float(ov.get("solve_on", 0.0)),
                "uplift": float(ov.get("uplift", 0.0)),
                "ci_low": float(ov.get("ci_low", 0.0)),
                "ci_high": float(ov.get("ci_high", 0.0)),
                "sig": bool(ov.get("significant", False)),
            }
        )
    rows.sort(key=lambda r: r[sort_key], reverse=True)
    lines = [
        "| model | n | solve (off → on) | tool-uplift [95% CI] | sig |",
        "|---|---:|---|---|:-:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['n']} | {r['solve_off']:.3f} → {r['solve_on']:.3f} | "
            f"{r['uplift']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | "
            f"{'✓' if r['sig'] else '·'} |"
        )
    lines.append("")
    if len(versions) > 1:
        lines.append(
            f"> ⚠ mixed battery versions {sorted(v for v in versions if v)} — scores are "
            "only comparable within a version."
        )
    lines.append("_Losses are shown, not hidden: a model scoring above another stays on the board._")
    return "\n".join(lines)
