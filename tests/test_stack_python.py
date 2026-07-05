"""Tests for EX-6 — the-stack-stem-python scientific-import filter.

The pure functions (import detection, license gate, notebook rendering, record
assembly) are unit-tested; ``process_shard`` is exercised end-to-end against a tiny
synthetic parquet built with pyarrow, matching the real The-Stack schema.
"""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
from lithos.data.documents import read_jsonl
from lithos.data.stack_python import (
    CANON_ID,
    is_permissive,
    make_record,
    notebook_to_text,
    process_shard,
    scientific_imports,
    top_level_packages,
)

# ---- import detection -------------------------------------------------------


def test_top_level_packages_forms() -> None:
    code = (
        "import numpy\n"
        "import os, scipy.integrate as si\n"
        "from astropy.units import Quantity\n"
        "    import torch  # indented, inside a function\n"
        "from . import helpers\n"        # relative -> ignored
        "importlib.reload(x)\n"          # not an import statement -> ignored
    )
    pkgs = top_level_packages(code)
    assert {"numpy", "os", "scipy", "astropy", "torch"} <= pkgs
    assert "importlib" not in pkgs   # 'import' must be followed by whitespace
    assert "" not in pkgs            # relative import yields no top-level name


def test_scientific_imports_positive_and_negative() -> None:
    assert scientific_imports("import numpy as np\nx = np.zeros(3)") == {"numpy"}
    assert scientific_imports("from sympy import Symbol") == {"sympy"}
    # generic app/CLI Python imports nothing scientific -> dropped
    assert scientific_imports("import os, sys, json\nimport argparse") == set()


# ---- license gate -----------------------------------------------------------


def test_is_permissive() -> None:
    assert is_permissive(["MIT"]) is True
    assert is_permissive(["Apache-2.0", "BSD-3-Clause"]) is True
    assert is_permissive(["GPL-3.0"]) is False
    assert is_permissive(["MIT", "GPL-3.0"]) is False          # any copyleft -> reject
    assert is_permissive([], require_license=True) is False    # unknown -> drop (green tier)
    assert is_permissive([], require_license=False) is True


# ---- notebook rendering -----------------------------------------------------


def test_notebook_to_text_renders_and_extracts_code() -> None:
    nb = json.dumps({
        "cells": [
            {"cell_type": "markdown", "source": ["# Title\n", "some prose"]},
            {"cell_type": "code", "source": "import torch\nx = torch.tensor([1.0])"},
            {"cell_type": "code", "source": "", "outputs": [{"data": "junk"}]},  # empty -> skipped
        ]
    })
    rendered, code = notebook_to_text(nb)
    assert rendered is not None
    assert "# Title" in rendered and "```python" in rendered
    assert "import torch" in code
    assert scientific_imports(code) == {"torch"}


def test_notebook_to_text_bad_json() -> None:
    assert notebook_to_text("{not valid json") == (None, "")


# ---- record shape -----------------------------------------------------------


def test_make_record_anchors_to_canon() -> None:
    r = make_record("import numpy", kind="python", hexsha="abc123", repo="me/proj",
                    path="src/sim.py", lang="Python", licenses=["MIT"], stars=42,
                    sci_imports={"numpy"})
    assert r["id"] == "stack-python:abc123"
    assert r["source"] == "the-stack-stem"
    assert r["metadata"]["source_id"] == CANON_ID   # CH-12 canon anchor
    assert r["metadata"]["sci_imports"] == ["numpy"]
    assert r["license"] == "mit"


# ---- end-to-end over a synthetic parquet ------------------------------------


def _write_parquet(path, rows: list[dict]) -> None:
    cols = ["hexsha", "content", "lang", "max_stars_repo_name", "max_stars_repo_path",
            "max_stars_repo_licenses", "max_stars_count", "alphanum_fraction"]
    table = pa.table({
        "hexsha": pa.array([r["hexsha"] for r in rows], pa.string()),
        "content": pa.array([r["content"] for r in rows], pa.string()),
        "lang": pa.array([r.get("lang", "Python") for r in rows], pa.string()),
        "max_stars_repo_name": pa.array([r.get("repo", "x/y") for r in rows], pa.string()),
        "max_stars_repo_path": pa.array([r.get("path", "a.py") for r in rows], pa.string()),
        "max_stars_repo_licenses": pa.array([r["licenses"] for r in rows],
                                            pa.list_(pa.string())),
        "max_stars_count": pa.array([r.get("stars", 0) for r in rows], pa.int64()),
        "alphanum_fraction": pa.array([r.get("alphanum", 0.8) for r in rows], pa.float64()),
    })
    assert set(cols) <= set(table.column_names)
    pq.write_table(table, path)


def test_process_shard_python(tmp_path) -> None:
    rows = [
        {"hexsha": "h1", "content": "import numpy as np\nprint(np.pi)", "licenses": ["MIT"]},
        {"hexsha": "h2", "content": "import os, sys\nprint('hi')", "licenses": ["MIT"]},        # non-sci
        {"hexsha": "h3", "content": "import scipy\nscipy.integrate", "licenses": ["GPL-3.0"]},  # copyleft
        {"hexsha": "h4", "content": "x=1", "licenses": ["MIT"]},                                # too short
    ]
    pqf = tmp_path / "data-00000-of-00001.parquet"
    _write_parquet(pqf, rows)
    out = tmp_path / "python-data-00000-of-00001.jsonl.zst"
    stats = process_shard(pqf, out, "python", require_license=True,
                          min_chars=10, min_alphanum=0.25)

    assert stats.total == 4
    assert stats.kept == 1
    assert stats.dropped_nonsci == 1
    assert stats.dropped_license == 1
    assert stats.dropped_empty == 1

    recs = list(read_jsonl([str(out)]))
    assert len(recs) == 1
    assert recs[0]["id"] == "stack-python:h1"
    assert recs[0]["metadata"]["sci_imports"] == ["numpy"]


def test_process_shard_jupyter(tmp_path) -> None:
    good_nb = json.dumps({"cells": [
        {"cell_type": "markdown", "source": "# Sim"},
        {"cell_type": "code", "source": "import scipy.integrate\nprint('go')"},
    ]})
    plain_nb = json.dumps({"cells": [
        {"cell_type": "code", "source": "import os\nprint('no science here at all')"},
    ]})
    rows = [
        {"hexsha": "nb1", "content": good_nb, "lang": "Jupyter Notebook", "licenses": ["MIT"]},
        {"hexsha": "nb2", "content": plain_nb, "lang": "Jupyter Notebook", "licenses": ["MIT"]},
        {"hexsha": "nb3", "content": "{broken json", "lang": "Jupyter Notebook", "licenses": ["MIT"]},
    ]
    pqf = tmp_path / "data-00000-of-00001.parquet"
    _write_parquet(pqf, rows)
    out = tmp_path / "jupyter-data-00000-of-00001.jsonl.zst"
    stats = process_shard(pqf, out, "jupyter", require_license=True,
                          min_chars=10, min_alphanum=0.0)

    assert stats.kept == 1
    assert stats.dropped_nonsci == 1
    assert stats.dropped_parse == 1
    recs = list(read_jsonl([str(out)]))
    assert recs[0]["id"] == "stack-jupyter:nb1"
    assert "```python" in recs[0]["text"]        # rendered, not raw ipynb JSON
    assert recs[0]["metadata"]["sci_imports"] == ["scipy"]
