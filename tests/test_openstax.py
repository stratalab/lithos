"""Tests for the OpenStax fetcher (lithos/data/openstax.py) — tarball -> records."""

from __future__ import annotations

import io
import tarfile

from lithos.data.openstax import _module_id, iter_records_from_tar

_CNXML = b"""<?xml version="1.0"?>
<document xmlns="http://cnx.rice.edu/cnxml" xmlns:m="http://www.w3.org/1998/Math/MathML">
 <content><section><title>Kinematics</title>
 <para>The velocity <m:math><m:msub><m:mi>v</m:mi><m:mn>0</m:mn></m:msub></m:math> is initial.</para>
 </section></content></document>"""

_EMPTY = b'<document xmlns="http://cnx.rice.edu/cnxml"><content/></document>'


def _make_tar(files: dict[str, bytes]) -> io.BytesIO:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return buf


def test_module_id() -> None:
    assert _module_id("osbooks-x-main/modules/m5501/index.cnxml") == "m5501"


def test_iter_records_extracts_module() -> None:
    tar = _make_tar({
        "osbooks-x-main/modules/m1/index.cnxml": _CNXML,
        "osbooks-x-main/README.md": b"ignore me",          # non-module -> ignored
        "osbooks-x-main/collections/x.collection.xml": b"<c/>",  # not a module
    })
    recs = list(iter_records_from_tar(tar, repo="osbooks-x", commit="abc1234567", domain="physics"))
    assert len(recs) == 1
    r = recs[0]
    assert r["id"] == "openstax:osbooks-x:m1"
    assert r["source"] == "openstax"
    assert r["subset"] == "osbooks-x"
    assert r["license"] == "cc-by-4.0"
    assert r["metadata"] == {
        "repo": "osbooks-x", "commit": "abc1234567", "module": "m1", "domain": "physics",
        "url": "https://github.com/openstax/osbooks-x",
    }
    assert "## Kinematics" in r["text"]
    assert "v_{0}" in r["text"]  # MathML -> LaTeX survived


def test_iter_records_skips_empty_modules() -> None:
    tar = _make_tar({
        "osbooks-x-main/modules/m1/index.cnxml": _CNXML,
        "osbooks-x-main/modules/m2/index.cnxml": _EMPTY,   # no text -> skipped
    })
    recs = list(iter_records_from_tar(tar, repo="osbooks-x", commit="c", domain="math"))
    assert [r["metadata"]["module"] for r in recs] == ["m1"]
