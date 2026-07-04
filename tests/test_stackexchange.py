"""Tests for the Stack Exchange dump extractor (lithos/data/stackexchange.py)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from lithos.data.documents import read_jsonl
from lithos.data.stackexchange import (
    ExtractParams,
    archive_has_posts,
    assemble_text,
    extract_archive,
    html_to_text,
    infer_site,
    iter_posts,
    parse_tags,
)

# A tiny Posts.xml exercising: accepted-answer ordering, the answer/question
# score filters, an unanswered question, an orphan answer, and a non-Q/A row.
POSTS_XML = b"""<?xml version="1.0" encoding="utf-8"?>
<posts>
  <row Id="1" PostTypeId="1" Title="Why is the sky blue?"
       Body="&lt;p&gt;Because of &lt;code&gt;scattering&lt;/code&gt;.&lt;/p&gt;"
       Tags="&lt;optics&gt;&lt;light&gt;" Score="42" AcceptedAnswerId="3"/>
  <row Id="2" PostTypeId="2" ParentId="1" Score="5"
       Body="&lt;p&gt;Higher-scored but not accepted.&lt;/p&gt;"/>
  <row Id="3" PostTypeId="2" ParentId="1" Score="2"
       Body="&lt;p&gt;Accepted.&lt;/p&gt;&lt;pre&gt;&lt;code&gt;I = I0 * lambda&lt;/code&gt;&lt;/pre&gt;"/>
  <row Id="4" PostTypeId="2" ParentId="1" Score="-1"
       Body="&lt;p&gt;Downvoted noise.&lt;/p&gt;"/>
  <row Id="5" PostTypeId="1" Title="Unanswered question" Body="&lt;p&gt;nobody knows&lt;/p&gt;" Score="1"/>
  <row Id="6" PostTypeId="2" ParentId="999" Score="9" Body="&lt;p&gt;orphan answer&lt;/p&gt;"/>
  <row Id="7" PostTypeId="1" Title="Zero-score question" Body="&lt;p&gt;meh&lt;/p&gt;" Score="0"/>
  <row Id="8" PostTypeId="2" ParentId="7" Score="3" Body="&lt;p&gt;good answer&lt;/p&gt;"/>
  <row Id="9" PostTypeId="4" Body="&lt;p&gt;tag wiki, ignored&lt;/p&gt;"/>
</posts>
"""


def _pack(tmp_path: Path, xml: bytes = POSTS_XML,
          archive: str = "physics.stackexchange.com.7z") -> Path:
    py7zr = pytest.importorskip("py7zr")
    path = tmp_path / archive
    with py7zr.SevenZipFile(str(path), "w") as z:
        z.writef(io.BytesIO(xml), "Posts.xml")
    return path


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw,expected", [
    ("<optics><light>", ["optics", "light"]),   # old angle-bracket form
    ("|optics|light|", ["optics", "light"]),     # new pipe form
    ("<single-tag>", ["single-tag"]),
    ("", []),
    (None, []),
])
def test_parse_tags(raw: str | None, expected: list[str]) -> None:
    assert parse_tags(raw) == expected


def test_html_to_text_inline_code() -> None:
    assert html_to_text("<p>Because of <code>scattering</code>.</p>") == "Because of `scattering`."


def test_html_to_text_code_block() -> None:
    out = html_to_text("<p>Here:</p><pre><code>a = 1\nb = 2</code></pre>")
    assert "```" in out
    assert "a = 1\nb = 2" in out


def test_html_to_text_entities_and_blocks() -> None:
    out = html_to_text("<p>x &amp; y</p><p>next line</p>")
    assert "x & y" in out
    assert "\n" in out  # paragraph boundary became a newline


def test_html_to_text_malformed_falls_back() -> None:
    # Unbalanced markup must not raise — a best-effort strip is fine.
    assert "unclosed" in html_to_text("<p>unclosed <b>bold")


def test_html_to_text_empty() -> None:
    assert html_to_text(None) == ""
    assert html_to_text("") == ""


@pytest.mark.parametrize("name,site", [
    ("physics.stackexchange.com.7z", "physics.stackexchange.com"),
    ("stackoverflow.com-Posts.7z", "stackoverflow.com"),
    ("stackoverflow.com-Comments.7z", "stackoverflow.com"),
    ("mathoverflow.net.7z", "mathoverflow.net"),
    ("/data/dumps/electronics.stackexchange.com.7z", "electronics.stackexchange.com"),
])
def test_infer_site(name: str, site: str) -> None:
    assert infer_site(name) == site


def test_assemble_text_structure() -> None:
    out = assemble_text("Title", "Question body", ["First answer", "Second answer"])
    assert out.startswith("Title\n\nQuestion body")
    assert out.count("Answer:") == 2


# --------------------------------------------------------------------------- #
# Streaming .7z → rows (needs py7zr)
# --------------------------------------------------------------------------- #

def test_iter_posts_streams_all_rows(tmp_path: Path) -> None:
    arc = _pack(tmp_path)
    rows = list(iter_posts(arc))
    assert len(rows) == 9
    ids = {r["Id"] for r in rows}
    assert ids == {str(i) for i in range(1, 10)}
    assert rows[0]["Title"] == "Why is the sky blue?"


def test_archive_has_posts(tmp_path: Path) -> None:
    py7zr = pytest.importorskip("py7zr")
    # A normal site dump has Posts.xml.
    assert archive_has_posts(_pack(tmp_path)) is True
    # Stack Overflow's Comments archive has only Comments.xml → skip, not fail.
    comments = tmp_path / "stackoverflow.com-Comments.7z"
    with py7zr.SevenZipFile(str(comments), "w") as z:
        z.writef(io.BytesIO(b"<comments><row Id='1'/></comments>"), "Comments.xml")
    assert archive_has_posts(comments) is False


# --------------------------------------------------------------------------- #
# End-to-end extraction
# --------------------------------------------------------------------------- #

def test_extract_archive_end_to_end(tmp_path: Path) -> None:
    arc = _pack(tmp_path)
    out = tmp_path / "extracted"
    manifest = extract_archive(arc, out, params=ExtractParams())

    site = "physics.stackexchange.com"
    assert manifest["site"] == site
    assert manifest["questions_in"] == 3          # Q1, Q5, Q7 (row 9 is type 4)
    # A2, A3, A8, and the orphan A6 all clear the score filter and stage; A4
    # (score -1) is dropped. The orphan is discarded later at join, not here.
    assert manifest["answers_kept"] == 4
    # Q5 (unanswered) and the orphan answer produce no document; Q1 and Q7 do.
    assert manifest["documents_out"] == 2

    docs = list(read_jsonl([str(p) for p in (out / site).glob("posts-*.jsonl.zst")]))
    by_id = {d["id"]: d for d in docs}
    assert set(by_id) == {f"{site}:1", f"{site}:7"}

    d1 = by_id[f"{site}:1"]
    assert d1["source"] == "stackexchange"
    assert d1["subset"] == site
    assert d1["license"] == "cc-by-sa-4.0"
    assert d1["metadata"]["tags"] == ["optics", "light"]
    assert d1["metadata"]["question_score"] == 42
    assert d1["metadata"]["accepted_answer_id"] == 3
    assert d1["metadata"]["url"] == f"https://{site}/q/1"
    # Accepted answer (id=3, score=2) must sort before the higher-scored id=2.
    assert d1["metadata"]["n_answers"] == 2
    assert d1["metadata"]["answer_scores"] == [2, 5]
    # Its body carries the fenced code block from the accepted answer.
    assert "```" in d1["text"]
    assert "Why is the sky blue?" in d1["text"]


def test_min_question_score_filter(tmp_path: Path) -> None:
    arc = _pack(tmp_path)
    out = tmp_path / "extracted"
    manifest = extract_archive(arc, out, params=ExtractParams(min_question_score=1))
    # Q7 has score 0 → dropped; only Q1 survives (Q5 still unanswered).
    assert manifest["documents_out"] == 1
    docs = list(read_jsonl([str(p) for p in (out / "physics.stackexchange.com").glob("*.zst")]))
    assert {d["id"] for d in docs} == {"physics.stackexchange.com:1"}


def test_extract_is_idempotent_manifest(tmp_path: Path) -> None:
    arc = _pack(tmp_path)
    out = tmp_path / "extracted"
    extract_archive(arc, out, params=ExtractParams())
    assert (out / "physics.stackexchange.com" / "_manifest.json").exists()
