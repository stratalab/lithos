"""EX-6 — the-stack-stem-python scientific-import filter.

Turns the raw The-Stack-dedup **Python + Jupyter** slice into canonical records,
keeping only files that import the scientific-Python stack (numpy / scipy / torch /
sympy / astropy / …). That import is the signal we want: it separates genuine
numerical / physics / eng / math code from the generic web-app, CLI, and config
Python that dominates GitHub. On The Stack that selection lands roughly the ~150 GB
(of ~350 GB raw) worth keeping for the STEM code slice — the raw parquet is then
deleted, so only the filtered output occupies the drive.

Two filters, both auditable:
* **scientific-import** — a file is kept iff it imports at least one package in
  ``SCIENTIFIC_IMPORTS`` (detected by a syntax-error-tolerant regex, not the ``ast``
  module — The Stack is full of Python-2 and broken files that ``ast`` would reject).
  The matched packages are recorded in ``metadata.sci_imports`` for later mix analysis.
* **permissive-license** — defense in depth over The Stack's own filtering: every
  detected license must be permissive (``PERMISSIVE_LICENSES``). Green-tier doctrine.

Notebooks are converted to clean ``markdown + fenced-code`` text (cell *outputs*
dropped — base64 images and long dumps are token garbage), and the import test runs
over their code cells.

Flow (run *after* the hf download, as a separate post-step):
  uv run python -m lithos.data.stack_python \\
      --in  /data2/corpus-staging/the-stack-stem-python \\
      --out /data2/corpus-staging/the-stack-stem-python-filtered
  # then delete the raw parquet to reclaim the ~200 GB we filtered away.

Output: one ``<kind>-<shard>.jsonl.zst`` per input parquet (resumable — an existing
output shard is skipped), plus ``_sources.json`` with per-kind keep-rate stats so the
scientific-import set can be calibrated empirically before the full run.

pyarrow + zstandard only (both already core data deps).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import zstandard

log = logging.getLogger("lithos.stack_python")

CANON_ID = "the-stack-stem"  # corpus/seed_index.csv id these records anchor to (CH-12)

# Top-level packages whose import marks a file as scientific. Grouped for legibility;
# tune this set (and re-run a --limit-shards calibration) to move the keep rate.
SCIENTIFIC_IMPORTS: frozenset[str] = frozenset({
    # core numerics / arrays / dataframes
    "numpy", "scipy", "sympy", "mpmath", "gmpy2", "pandas", "xarray", "numba",
    "cupy", "dask", "zarr", "h5py", "netCDF4", "tables", "pyarrow",
    # ML / DL used as scientific compute
    "torch", "tensorflow", "jax", "flax", "sklearn", "statsmodels", "keras",
    # plotting / scientific viz
    "matplotlib", "seaborn", "plotly", "bokeh", "mayavi", "pyvista", "vtk",
    # symbolic / graph / optimization math
    "networkx", "cvxpy", "pyomo", "pulp", "nlopt", "galois", "sage",
    # physics / astronomy
    "astropy", "sunpy", "galpy", "yt", "qutip", "poliastro", "skyfield", "spacepy",
    "pint", "uncertainties",
    # chemistry / materials / molecular
    "rdkit", "ase", "pymatgen", "MDAnalysis", "openmm", "pyscf", "psi4", "cantera",
    # engineering: CFD / FEM / meshing / HPC
    "fenics", "dolfin", "dolfinx", "firedrake", "sfepy", "skfem", "petsc4py",
    "slepc4py", "mpi4py", "gmsh", "meshio", "control",
    # signal / image (scientific)
    "skimage", "cv2", "pywt",
})

# SPDX ids (lowercased) we accept as permissive. Copyleft (GPL/LGPL/AGPL/MPL) excluded.
PERMISSIVE_LICENSES: frozenset[str] = frozenset({
    "mit", "mit-0", "apache-2.0", "bsd-3-clause", "bsd-2-clause",
    "bsd-2-clause-patent", "isc", "0bsd", "unlicense", "cc0-1.0", "zlib",
    "python-2.0", "postgresql", "bsl-1.0", "ncsa", "wtfpl",
})

# Matches an import statement at (indented) line start, capturing the keyword and the
# rest of the line. Syntax-error tolerant: pure text scan, never parses the file.
_IMPORT_RE = re.compile(r"^[ \t]*(from|import)[ \t]+(.+)$", re.MULTILINE)

_PARQUET_COLS = [
    "hexsha", "content", "lang", "max_stars_repo_name", "max_stars_repo_path",
    "max_stars_repo_licenses", "max_stars_count", "size", "alphanum_fraction",
    "max_line_length",
]


def top_level_packages(code: str) -> set[str]:
    """Top-level package names imported by ``code`` (regex; tolerant of bad syntax)."""
    pkgs: set[str] = set()
    for kw, rest in _IMPORT_RE.findall(code):
        if kw == "from":
            toks = rest.split()
            if toks:
                pkgs.add(toks[0].split(".")[0])
        else:  # import a, b.c as d, e  -> a, b, e
            for part in rest.split(","):
                toks = part.strip().split()
                if toks:
                    pkgs.add(toks[0].split(".")[0])
    pkgs.discard("")  # relative imports ("from . import x") yield an empty top level
    return pkgs


def scientific_imports(code: str) -> set[str]:
    """Which ``SCIENTIFIC_IMPORTS`` a file imports (empty set → not scientific)."""
    return top_level_packages(code) & SCIENTIFIC_IMPORTS


def is_permissive(licenses: Any, *, require_license: bool = True) -> bool:
    """True if every detected license is permissive. Empty/unknown → ``require_license``."""
    if not licenses:
        return not require_license
    lics = [str(x).strip().lower() for x in licenses if str(x).strip()]
    if not lics:
        return not require_license
    return all(lic in PERMISSIVE_LICENSES for lic in lics)


def notebook_to_text(content: str) -> tuple[str | None, str]:
    """(.ipynb JSON) -> (rendered markdown+code text | None, code-only text).

    ``None`` rendered means the notebook could not be parsed. Cell outputs are
    dropped. Handles nbformat 4 (``cells``) and the legacy 3 (``worksheets``).
    """
    try:
        nb = json.loads(content)
    except (ValueError, TypeError):
        return None, ""
    if not isinstance(nb, dict):
        return None, ""
    cells = nb.get("cells")
    if cells is None:  # nbformat 3
        cells = [c for ws in nb.get("worksheets", []) for c in ws.get("cells", [])]
    if not isinstance(cells, list):
        return None, ""

    rendered: list[str] = []
    code: list[str] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        src = cell.get("source") or cell.get("input") or ""
        if isinstance(src, list):
            src = "".join(str(s) for s in src)
        if not isinstance(src, str) or not src.strip():
            continue
        if cell.get("cell_type") == "code":
            code.append(src)
            rendered.append(f"```python\n{src}\n```")
        elif cell.get("cell_type") == "markdown":
            rendered.append(src)
    return "\n\n".join(rendered), "\n".join(code)


def make_record(text: str, *, kind: str, hexsha: str, repo: str, path: str, lang: str,
                licenses: Any, stars: int, sci_imports: set[str]) -> dict[str, Any]:
    """Assemble one canonical record, anchored to the Canon via metadata.source_id."""
    lic_list = [str(x) for x in licenses] if licenses else []
    return {
        "id": f"stack-{kind}:{hexsha}",
        "text": text,
        "source": "the-stack-stem",
        "subset": f"python/{kind}",
        "language": "en",  # natural language of comments/markdown; code lang in metadata
        "license": lic_list[0].lower() if lic_list else "permissive",
        "metadata": {
            "source_id": CANON_ID,
            "lang": lang,
            "repo": repo,
            "path": path,
            "hexsha": hexsha,
            "licenses": lic_list,
            "stars": stars,
            "sci_imports": sorted(sci_imports),
            "extractor": "stack_python",
        },
    }


@dataclass
class Stats:
    total: int = 0
    kept: int = 0
    dropped_license: int = 0
    dropped_nonsci: int = 0
    dropped_empty: int = 0
    dropped_parse: int = 0
    kept_bytes: int = 0
    imports: dict[str, int] = field(default_factory=dict)

    def keep_rate(self) -> float:
        return self.kept / self.total if self.total else 0.0

    def merge(self, other: Stats) -> None:
        self.total += other.total
        self.kept += other.kept
        self.dropped_license += other.dropped_license
        self.dropped_nonsci += other.dropped_nonsci
        self.dropped_empty += other.dropped_empty
        self.dropped_parse += other.dropped_parse
        self.kept_bytes += other.kept_bytes
        for k, v in other.imports.items():
            self.imports[k] = self.imports.get(k, 0) + v


def iter_shard_records(parquet_path: Path, kind: str, *, require_license: bool,
                       min_chars: int, min_alphanum: float, stats: Stats,
                       ) -> Iterator[dict[str, Any]]:
    """Stream one parquet shard, yielding canonical records for kept files."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    cols = [c for c in _PARQUET_COLS if c in pf.schema_arrow.names]
    for batch in pf.iter_batches(batch_size=512, columns=cols):
        for row in batch.to_pylist():
            stats.total += 1
            content = row.get("content")
            if not isinstance(content, str) or len(content) < min_chars:
                stats.dropped_empty += 1
                continue
            alphanum = row.get("alphanum_fraction")
            if isinstance(alphanum, float) and alphanum < min_alphanum:
                stats.dropped_empty += 1
                continue
            if not is_permissive(row.get("max_stars_repo_licenses"),
                                 require_license=require_license):
                stats.dropped_license += 1
                continue

            if kind == "jupyter":
                text, code = notebook_to_text(content)
                if text is None:
                    stats.dropped_parse += 1
                    continue
                import_src = code
            else:
                text, import_src = content, content

            sci = scientific_imports(import_src)
            if not sci:
                stats.dropped_nonsci += 1
                continue
            if not text.strip():
                stats.dropped_empty += 1
                continue

            stats.kept += 1
            stats.kept_bytes += len(text.encode("utf-8"))
            for pkg in sci:
                stats.imports[pkg] = stats.imports.get(pkg, 0) + 1
            yield make_record(
                text, kind=kind, hexsha=str(row.get("hexsha", "")),
                repo=str(row.get("max_stars_repo_name", "")),
                path=str(row.get("max_stars_repo_path", "")),
                lang=str(row.get("lang", "")),
                licenses=row.get("max_stars_repo_licenses"),
                stars=int(row.get("max_stars_count") or 0), sci_imports=sci,
            )


def process_shard(parquet_path: Path, out_path: Path, kind: str, *,
                  require_license: bool, min_chars: int, min_alphanum: float) -> Stats:
    """Filter one parquet shard to ``out_path`` (.jsonl.zst). Returns its stats."""
    stats = Stats()
    tmp = out_path.with_suffix(out_path.suffix + ".partial")
    with open(tmp, "wb") as fh:
        w = zstandard.ZstdCompressor(level=10).stream_writer(fh)
        for rec in iter_shard_records(parquet_path, kind, require_license=require_license,
                                      min_chars=min_chars, min_alphanum=min_alphanum,
                                      stats=stats):
            w.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
        w.close()
    if stats.kept:
        tmp.replace(out_path)  # atomic: a finished shard only appears whole
    else:
        tmp.unlink(missing_ok=True)
    log.info("[%s] %s: %d/%d kept (%.1f%%), %.2f GB", kind, parquet_path.name,
             stats.kept, stats.total, 100 * stats.keep_rate(), stats.kept_bytes / 1e9)
    return stats


# The raw download uses HF's "jupyter-notebook" dir name; we emit the shorter "jupyter".
_KIND_DIRS = {"python": "python", "jupyter": "jupyter-notebook"}


def run(in_dir: Path, out_dir: Path, *, kinds: list[str], limit_shards: int | None,
        require_license: bool, min_chars: int, min_alphanum: float) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_kind: dict[str, Stats] = {}
    for kind in kinds:
        src = in_dir / "data" / _KIND_DIRS[kind]
        shards = sorted(src.glob("*.parquet"))
        if limit_shards:
            shards = shards[:limit_shards]
        if not shards:
            log.warning("[%s] no parquet under %s — skipping", kind, src)
            continue
        log.info("[%s] %d shard(s) under %s", kind, len(shards), src)
        agg = Stats()
        for shard in shards:
            out_path = out_dir / f"{kind}-{shard.stem}.jsonl.zst"
            if out_path.exists():
                log.info("[%s] %s already filtered — skipping", kind, out_path.name)
                continue
            agg.merge(process_shard(shard, out_path, kind, require_license=require_license,
                                    min_chars=min_chars, min_alphanum=min_alphanum))
        per_kind[kind] = agg
        top = sorted(agg.imports.items(), key=lambda kv: -kv[1])[:15]
        log.info("[%s] TOTAL %d/%d kept (%.1f%%), %.1f GB; top imports: %s",
                 kind, agg.kept, agg.total, 100 * agg.keep_rate(), agg.kept_bytes / 1e9,
                 ", ".join(f"{k}:{v}" for k, v in top))

    manifest = {
        "source": "the-stack-stem", "source_id": CANON_ID, "extractor": "stack_python",
        "filtered_at": datetime.now(UTC).isoformat(),
        "require_license": require_license, "min_chars": min_chars,
        "min_alphanum": min_alphanum,
        "kinds": {k: {
            "total": s.total, "kept": s.kept, "keep_rate": round(s.keep_rate(), 4),
            "dropped_license": s.dropped_license, "dropped_nonsci": s.dropped_nonsci,
            "dropped_empty": s.dropped_empty, "dropped_parse": s.dropped_parse,
            "kept_gb": round(s.kept_bytes / 1e9, 2),
            "top_imports": dict(sorted(s.imports.items(), key=lambda kv: -kv[1])[:25]),
        } for k, s in per_kind.items()},
    }
    (out_dir / "_sources.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_dir", required=True, type=Path,
                   help="raw the-stack-stem-python dir (contains data/python, data/jupyter-notebook)")
    p.add_argument("--out", required=True, type=Path, help="output dir for filtered .jsonl.zst")
    p.add_argument("--kinds", default="python,jupyter",
                   help="comma list of python,jupyter (default both)")
    p.add_argument("--limit-shards", type=int, default=None,
                   help="process only the first N shards per kind (calibration)")
    p.add_argument("--allow-unlicensed", action="store_true",
                   help="keep files with no detected license (default: drop — green tier)")
    p.add_argument("--min-chars", type=int, default=50)
    p.add_argument("--min-alphanum", type=float, default=0.25)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    bad = [k for k in kinds if k not in _KIND_DIRS]
    if bad:
        p.error(f"unknown kind(s) {bad}; choose from {list(_KIND_DIRS)}")

    m = run(args.in_dir, args.out, kinds=kinds, limit_shards=args.limit_shards,
            require_license=not args.allow_unlicensed, min_chars=args.min_chars,
            min_alphanum=args.min_alphanum)
    print(json.dumps(m, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
