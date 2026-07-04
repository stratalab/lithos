"""Stack Exchange dump extraction — raw ``.7z`` XML → canonical Q&A documents.

The archive.org Stack Exchange dumps ship one ``.7z`` per site (Stack Overflow
splits ``Posts``/``Comments`` into their own archives). Each holds ``Posts.xml``:
a flat list of ``<row>`` elements where ``PostTypeId=1`` is a question and
``PostTypeId=2`` an answer (linked to its question by ``ParentId``). Bodies are
XML-escaped HTML fragments.

This module turns that into the canonical record
``{id, text, source, subset, language, license, metadata}`` (``documents.py``),
pairing each question with its top answers. Two properties make it scale to the
Stack Overflow dump (~60M posts, ~100 GB of XML) on a commodity box:

* **Streaming decompression.** ``.7z`` is decompressed straight into an
  incremental XML parser via py7zr's writer-factory hook — no full-size temp
  file, bounded memory. A producer thread keeps the push-model decompressor
  behind a clean pull-model generator.
* **Disk-backed join.** Questions and answers are staged in a throwaway SQLite
  db, so pairing never has to hold the corpus in RAM.

py7zr is an optional (``data`` extra) dependency, imported lazily so importing
this module — and the rest of ``lithos.data`` — never requires it.
"""

from __future__ import annotations

import json
import os
import queue
import re
import sqlite3
import tempfile
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import zstandard
from lxml import etree, html

# Post types we care about (Posts.xml also carries tag-wikis, moderator posts, …).
QUESTION = 1
ANSWER = 2

# Table suffixes archive.org appends when a site is split across archives
# (Stack Overflow); stripped to recover the bare site name.
_TABLE_SUFFIX = re.compile(
    r"-(Posts|Comments|Users|Votes|Tags|Badges|PostHistory|PostLinks)$", re.IGNORECASE
)

# Block-level HTML tags whose boundaries become newlines in the flattened text.
_BLOCK = {
    "p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
    "blockquote", "tr", "table", "hr", "pre",
}

# Best-effort domain hint per known P0 site (the real domain tagger runs later;
# this is only a convenience field in metadata).
SITE_DOMAIN = {
    "stackoverflow.com": "code",
    "math.stackexchange.com": "math",
    "mathoverflow.net": "math",
    "physics.stackexchange.com": "physics",
    "electronics.stackexchange.com": "eng",
    "stats.stackexchange.com": "xdomain",
    "scicomp.stackexchange.com": "xdomain",
    "cstheory.stackexchange.com": "xdomain",
}


@dataclass(slots=True)
class ExtractParams:
    """Extraction-time filters (heavy quality filtering happens downstream)."""

    min_answer_score: int = 1  # drop zero/negative-scored answers (noise)
    min_question_score: int | None = None  # None → keep all answered questions
    max_answers: int = 5  # accepted first, then by score
    require_answer: bool = True  # a Q&A doc needs at least one answer
    license: str = "cc-by-sa-4.0"


# --------------------------------------------------------------------------- #
# HTML → text
# --------------------------------------------------------------------------- #

def parse_tags(raw: str | None) -> list[str]:
    """Parse a Tags attribute: old ``<a><b>`` or new ``|a|b|`` form → list."""
    if not raw:
        return []
    tags = re.findall(r"<([^<>]+)>", raw)
    if not tags:
        tags = [t for t in raw.split("|") if t]
    return [t.strip() for t in tags if t.strip()]


def _render(el: Any, out: list[str]) -> None:
    tag = el.tag if isinstance(el.tag, str) else ""
    if tag == "pre":  # code block — keep verbatim, fence it
        out.append("\n```\n")
        out.append(el.text_content())
        out.append("\n```\n")
        if el.tail:
            out.append(el.tail)
        return
    if tag == "code":  # inline code
        out.append(f"`{el.text_content().strip()}`")
        if el.tail:
            out.append(el.tail)
        return
    if tag in _BLOCK:
        out.append("\n")
    if el.text:
        out.append(el.text)
    for child in el:
        _render(child, out)
    if tag in _BLOCK:
        out.append("\n")
    if el.tail:
        out.append(el.tail)


def html_to_text(body: str | None) -> str:
    """Flatten a Stack Exchange HTML body to text, preserving code structure."""
    if not body:
        return ""
    try:
        frag = html.fragment_fromstring(body, create_parent="div")
    except (etree.ParserError, etree.XMLSyntaxError, ValueError):
        # Malformed fragment — fall back to a crude tag strip.
        return re.sub(r"\s+\n", "\n", re.sub(r"<[^>]+>", "", body)).strip()
    out: list[str] = []
    _render(frag, out)
    text = "".join(out)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Streaming .7z → <row> dicts
# --------------------------------------------------------------------------- #

def find_posts_member(names: list[str]) -> str:
    """Locate the Posts.xml entry inside an archive's member list."""
    for n in names:
        if n.replace("\\", "/").split("/")[-1].lower() == "posts.xml":
            return n
    raise ValueError(f"no Posts.xml in archive members: {names}")


def _drain(parser: Any, put: Any) -> None:
    """Emit finished <row> elements and prune the parsed tree to stay bounded."""
    for _event, el in parser.read_events():
        if el.tag == "row":
            put(dict(el.attrib))
        # Prune this element and any earlier siblings so the tree never grows.
        parent = el.getparent()
        el.clear()
        if parent is not None:
            while el.getprevious() is not None:
                del parent[0]


def iter_posts(archive_path: str | Path, member: str | None = None) -> Iterator[dict[str, str]]:
    """Yield each ``<row>`` of Posts.xml as an attribute dict, streaming from .7z.

    A producer thread decompresses and parses (push model); rows flow through a
    bounded queue so this is a clean, backpressured generator. Memory stays
    bounded regardless of archive size.
    """
    import py7zr  # lazy: only needed here, and only with the `data` extra
    from py7zr.io import WriterFactory

    rows: queue.Queue[dict[str, str] | None] = queue.Queue(maxsize=2000)
    sentinel: None = None
    err: dict[str, BaseException] = {}

    class _Writer:
        def __init__(self, feed: Any) -> None:
            self._feed, self._n = feed, 0

        def write(self, s: bytes | bytearray) -> int:
            b = bytes(s)
            self._feed(b)
            self._n += len(b)
            return len(b)

        def read(self, size: int | None = None) -> bytes:
            return b""

        def seek(self, offset: int, whence: int = 0) -> int:
            return 0

        def size(self) -> int:
            return self._n

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

    class _Factory(WriterFactory):
        def __init__(self, feed: Any) -> None:
            self._feed = feed

        def create(self, filename: str) -> Any:
            return _Writer(self._feed)

    def produce() -> None:
        try:
            parser = etree.XMLPullParser(events=("end",), huge_tree=True, recover=True)
            put = rows.put

            def feed(chunk: bytes) -> None:
                parser.feed(chunk)
                _drain(parser, put)

            with py7zr.SevenZipFile(str(archive_path), "r") as z:
                target = member or find_posts_member(z.getnames())
                z.extract(factory=_Factory(feed), targets=[target])
            parser.close()
            _drain(parser, put)
        except BaseException as e:  # surface in the consumer
            err["e"] = e
        finally:
            rows.put(sentinel)

    thread = threading.Thread(target=produce, daemon=True)
    thread.start()
    while True:
        item = rows.get()
        if item is sentinel:
            break
        yield item
    thread.join()
    if "e" in err:
        raise err["e"]


# --------------------------------------------------------------------------- #
# Q&A pairing + document assembly
# --------------------------------------------------------------------------- #

def infer_site(archive_path: str | Path) -> str:
    """``physics.stackexchange.com.7z`` / ``stackoverflow.com-Posts.7z`` → site."""
    stem = Path(archive_path).name
    if stem.lower().endswith(".7z"):
        stem = stem[:-3]
    return _TABLE_SUFFIX.sub("", stem)


def _to_int(v: str | None) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except ValueError:
        return None


def assemble_text(title: str, question: str, answers: list[str]) -> str:
    """Compose the Q&A document body (title + question + labeled answers)."""
    parts: list[str] = []
    if title.strip():
        parts.append(title.strip())
    if question.strip():
        parts.append(question.strip())
    for a in answers:
        if a.strip():
            parts.append(f"Answer:\n{a.strip()}")
    return "\n\n".join(parts)


def _stage_rows(conn: sqlite3.Connection, rows: Iterator[dict[str, str]], params: ExtractParams,
                ) -> tuple[int, int]:
    """Stream Posts rows into the staging db. Returns (questions_in, answers_kept)."""
    conn.executescript(
        "PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA temp_store=MEMORY;"
        "CREATE TABLE q(id INTEGER PRIMARY KEY, title TEXT, body TEXT, tags TEXT,"
        " accepted INTEGER, score INTEGER);"
        "CREATE TABLE a(id INTEGER, parent INTEGER, body TEXT, score INTEGER);"
    )
    q_batch: list[tuple[Any, ...]] = []
    a_batch: list[tuple[Any, ...]] = []
    n_q = n_a = 0

    def flush() -> None:
        if q_batch:
            conn.executemany("INSERT OR REPLACE INTO q VALUES (?,?,?,?,?,?)", q_batch)
            q_batch.clear()
        if a_batch:
            conn.executemany("INSERT INTO a VALUES (?,?,?,?)", a_batch)
            a_batch.clear()

    for r in rows:
        ptype = _to_int(r.get("PostTypeId"))
        if ptype == QUESTION:
            qid = _to_int(r.get("Id"))
            if qid is None:
                continue
            score = _to_int(r.get("Score")) or 0
            if params.min_question_score is not None and score < params.min_question_score:
                continue
            q_batch.append((
                qid, r.get("Title", "") or "", html_to_text(r.get("Body")),
                json.dumps(parse_tags(r.get("Tags"))), _to_int(r.get("AcceptedAnswerId")), score,
            ))
            n_q += 1
        elif ptype == ANSWER:
            parent = _to_int(r.get("ParentId"))
            aid = _to_int(r.get("Id"))
            if parent is None or aid is None:
                continue
            score = _to_int(r.get("Score")) or 0
            if score < params.min_answer_score:
                continue
            a_batch.append((aid, parent, html_to_text(r.get("Body")), score))
            n_a += 1
        if len(q_batch) >= 5000 or len(a_batch) >= 5000:
            flush()
    flush()
    conn.execute("CREATE INDEX a_parent ON a(parent)")
    return n_q, n_a


def _join_documents(conn: sqlite3.Connection, site: str, params: ExtractParams,
                    ) -> Iterator[dict[str, Any]]:
    """Pair each staged question with its top answers → canonical records."""
    domain = SITE_DOMAIN.get(site)
    qcur = conn.execute("SELECT id, title, body, tags, accepted, score FROM q ORDER BY id")
    for qid, title, qbody, tags_json, accepted, qscore in qcur:
        ans = conn.execute(
            "SELECT id, body, score FROM a WHERE parent=? ORDER BY (id=?) DESC, score DESC LIMIT ?",
            (qid, accepted if accepted is not None else -1, params.max_answers),
        ).fetchall()
        if not ans and params.require_answer:
            continue
        text = assemble_text(title, qbody, [b for _i, b, _s in ans])
        if not text:
            continue
        tags = json.loads(tags_json)
        yield {
            "id": f"{site}:{qid}",
            "text": text,
            "source": "stackexchange",
            "subset": site,
            "language": "en",
            "license": params.license,
            "metadata": {
                "site": site,
                "domain": domain,
                "question_id": qid,
                "question_score": qscore,
                "tags": tags,
                "n_answers": len(ans),
                "answer_scores": [s for _i, _b, s in ans],
                "accepted_answer_id": accepted,
                "url": f"https://{site}/q/{qid}",
            },
        }


@dataclass(slots=True)
class ShardWriter:
    """Write canonical records to sharded ``posts-NNNNN.jsonl.zst`` files."""

    out_dir: Path
    shard_size: int = 50_000
    _idx: int = 0
    _n_in_shard: int = 0
    _n_total: int = 0
    _bytes: int = 0
    _fh: Any = None
    _writer: Any = None
    files: list[str] = field(default_factory=list)

    def _open(self) -> None:
        name = f"posts-{self._idx:05d}.jsonl.zst"
        path = self.out_dir / name
        self._fh = open(path, "wb")  # noqa: SIM115 — long-lived; closed in close()
        self._writer = zstandard.ZstdCompressor(level=10).stream_writer(self._fh)
        self.files.append(name)

    def write(self, record: dict[str, Any]) -> None:
        if self._writer is None or self._n_in_shard >= self.shard_size:
            self.close()
            self._open()
            self._n_in_shard = 0
            self._idx += 1
        line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        self._bytes += self._writer.write(line)
        self._n_in_shard += 1
        self._n_total += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._fh.close()
            self._writer = None
            self._fh = None

    @property
    def n_total(self) -> int:
        return self._n_total


def extract_archive(archive_path: str | Path, out_dir: str | Path, *,
                    params: ExtractParams | None = None, tmpdir: str | Path | None = None,
                    shard_size: int = 50_000) -> dict[str, Any]:
    """Extract one site's ``.7z`` to canonical Q&A JSONL(.zst) + a manifest."""
    params = params or ExtractParams()
    site = infer_site(archive_path)
    out = Path(out_dir) / site
    out.mkdir(parents=True, exist_ok=True)

    db_dir = Path(tmpdir) if tmpdir else out
    db_dir.mkdir(parents=True, exist_ok=True)
    db_fd, db_path = tempfile.mkstemp(suffix=".sqlite", dir=str(db_dir))
    conn = sqlite3.connect(db_path)
    writer = ShardWriter(out, shard_size=shard_size)
    try:
        n_q, n_a = _stage_rows(conn, iter_posts(archive_path), params)
        for record in _join_documents(conn, site, params):
            writer.write(record)
        writer.close()
    finally:
        conn.close()
        os.close(db_fd)
        Path(db_path).unlink(missing_ok=True)

    manifest = {
        "site": site,
        "source_archive": str(Path(archive_path).name),
        "questions_in": n_q,
        "answers_kept": n_a,
        "documents_out": writer.n_total,
        "files": writer.files,
        "license": params.license,
        "params": {
            "min_answer_score": params.min_answer_score,
            "min_question_score": params.min_question_score,
            "max_answers": params.max_answers,
            "require_answer": params.require_answer,
        },
        "extracted_at": datetime.now(UTC).isoformat(),
    }
    (out / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
