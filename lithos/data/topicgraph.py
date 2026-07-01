"""Wikipedia topic-graph job (docs/data-construction.md §1.7).

Wikipedia's *link graph* — not its token count — is the tool: seed each STEM
domain with pre-made curation ("Outline of X", vital-article lists), expand by
personalized PageRank over the full link graph (backlinks + outlinks via
symmetrization), and threshold into per-domain **topic families**. Outputs, in
value order: citation-ranked canon candidates (grows `corpus/seed_index.csv`
with an objective priority signal), the stage-8 domain-tagging vocabulary,
a coverage checklist, and the graph-selected article slice itself.

Everything runs offline from the standard enwiki dumps (`page`, `redirect`,
`linktarget`, `pagelinks` SQL dumps + `pages-articles` XML for citations) —
no scraping, no API calls. Pure numpy; no scipy dependency.

Memory note: the full enwiki link graph is ~0.5-1B edges; building the CSR
plus PPR working arrays wants ~32 GB RAM. Like MinHash dedup, this is an
offline data-build step — use ``max_edges`` to smoke-test on less.
"""

from __future__ import annotations

import bz2
import csv
import gzip
import logging
import re
from array import array
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any
from xml.etree import ElementTree

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MediaWiki SQL dump parsing
# ---------------------------------------------------------------------------

_UNESCAPE_RE = re.compile(r"\\(.)")
_UNESCAPE_MAP = {"n": "\n", "t": "\t", "r": "\r", "0": "\0"}


def _unescape(s: str) -> str:
    return _UNESCAPE_RE.sub(lambda m: _UNESCAPE_MAP.get(m.group(1), m.group(1)), s)


def open_maybe_compressed(path: str | Path) -> IO[str]:
    """Open a dump file as text, transparently handling .gz / .bz2."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    if path.suffix == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def _iter_value_tuples(values: str) -> Iterator[list[str | None]]:
    """Parse the VALUES section of a MySQL INSERT: (a,'b',NULL),(...),...

    Handles backslash-escaped quotes inside strings. Yields one list per row;
    strings are unescaped, NULL becomes None, numbers stay as strings.
    """
    i = 0
    find = values.find
    while True:
        i = find("(", i)
        if i == -1:
            return
        row: list[str | None] = []
        i += 1
        while True:
            c = values[i]
            if c == "'":
                j = i + 1
                while True:
                    j = find("'", j)
                    if j == -1:
                        raise ValueError("unterminated string in SQL dump")
                    # A quote is the terminator iff preceded by an even number
                    # of backslashes.
                    k = j - 1
                    while values[k] == "\\":
                        k -= 1
                    if (j - 1 - k) % 2 == 0:
                        break
                    j += 1
                s = values[i + 1 : j]
                row.append(_unescape(s) if "\\" in s else s)
                i = j + 1
            else:
                j = i
                while values[j] not in ",)":
                    j += 1
                tok = values[i:j]
                row.append(None if tok == "NULL" else tok)
                i = j
            c = values[i]
            i += 1
            if c == ")":
                yield row
                break


def iter_sql_rows(path: str | Path, table: str) -> Iterator[list[str | None]]:
    """Stream rows from a MediaWiki `<table>.sql(.gz)` dump."""
    prefix = f"INSERT INTO `{table}` VALUES "
    with open_maybe_compressed(path) as f:
        for line in f:
            if line.startswith(prefix):
                yield from _iter_value_tuples(line[len(prefix) :])


# ---------------------------------------------------------------------------
# Table loaders (namespace 0 only throughout)
# ---------------------------------------------------------------------------


def load_pages(path: str | Path) -> tuple[dict[str, int], dict[int, str], set[int]]:
    """page table → (title→id, id→title, ids-that-are-redirects). ns0 only."""
    title_to_id: dict[str, int] = {}
    id_to_title: dict[int, str] = {}
    redirect_ids: set[int] = set()
    for row in iter_sql_rows(path, "page"):
        if row[1] != "0":
            continue
        page_id = int(row[0])  # type: ignore[arg-type]
        title = row[2] or ""
        title_to_id[title] = page_id
        id_to_title[page_id] = title
        if row[3] == "1":
            redirect_ids.add(page_id)
    logger.info("pages: %d ns0 (%d redirects)", len(title_to_id), len(redirect_ids))
    return title_to_id, id_to_title, redirect_ids


def load_redirects(path: str | Path) -> dict[int, str]:
    """redirect table → {source page id: target title}. ns0 targets only."""
    out: dict[int, str] = {}
    for row in iter_sql_rows(path, "redirect"):
        if row[1] == "0" and row[2] is not None:
            out[int(row[0])] = row[2]  # type: ignore[arg-type]
    logger.info("redirects: %d ns0 targets", len(out))
    return out


def load_linktargets(path: str | Path) -> dict[int, str]:
    """linktarget table → {lt_id: title}. ns0 only."""
    out: dict[int, str] = {}
    for row in iter_sql_rows(path, "linktarget"):
        if row[1] == "0" and row[2] is not None:
            out[int(row[0])] = row[2]  # type: ignore[arg-type]
    logger.info("linktargets: %d ns0", len(out))
    return out


# ---------------------------------------------------------------------------
# Link graph
# ---------------------------------------------------------------------------

_PL_NEW_RE = re.compile(r"\((\d+),(\d+),(\d+)\)")


@dataclass
class LinkGraph:
    """CSR adjacency over non-redirect ns0 articles."""

    titles: list[str]  # node idx → title (underscored)
    title_to_idx: dict[str, int]
    indptr: np.ndarray  # int64, len n+1
    indices: np.ndarray  # int32, len E
    # redirect alias → node idx (for seed resolution + tagging vocabulary)
    alias_to_idx: dict[str, int] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.titles)

    @property
    def num_edges(self) -> int:
        return int(self.indices.shape[0])

    def resolve(self, title: str) -> int | None:
        """Title (spaces or underscores) → node idx, following redirects."""
        t = title.replace(" ", "_")
        idx = self.title_to_idx.get(t)
        if idx is None:
            idx = self.alias_to_idx.get(t)
        return idx


def _resolve_redirect_map(
    title_to_id: dict[str, int],
    redirect_ids: set[int],
    redirects: dict[int, str],
    node_idx_by_id: dict[int, int],
    max_hops: int = 3,
) -> dict[int, int]:
    """Map every ns0 page id (article or redirect) to a node idx."""
    resolve: dict[int, int] = dict(node_idx_by_id)
    for rid in redirect_ids:
        target = redirects.get(rid)
        hops = 0
        while target is not None and hops < max_hops:
            tid = title_to_id.get(target)
            if tid is None:
                target = None
                break
            if tid in node_idx_by_id:
                resolve[rid] = node_idx_by_id[tid]
                break
            target = redirects.get(tid)  # double redirect
            hops += 1
    return resolve


def build_link_graph(
    page_path: str | Path,
    redirect_path: str | Path,
    pagelinks_path: str | Path,
    linktarget_path: str | Path | None = None,
    *,
    symmetrize: bool = True,
    max_edges: int | None = None,
) -> LinkGraph:
    """Build the article link graph from the four SQL dumps.

    Supports both pagelinks schemas: the post-2024 (pl_from, pl_from_namespace,
    pl_target_id) — requires ``linktarget_path`` — and the legacy
    (pl_from, pl_namespace, pl_title). Redirect pages are not nodes: links
    from/to a redirect are remapped to its target article.
    """
    title_to_id, id_to_title, redirect_ids = load_pages(page_path)
    redirects = load_redirects(redirect_path)

    articles = [pid for pid in id_to_title if pid not in redirect_ids]
    articles.sort()
    node_idx_by_id = {pid: i for i, pid in enumerate(articles)}
    titles = [id_to_title[pid] for pid in articles]
    title_to_idx = {t: i for i, t in enumerate(titles)}
    resolve = _resolve_redirect_map(title_to_id, redirect_ids, redirects, node_idx_by_id)
    alias_to_idx = {
        id_to_title[rid]: resolve[rid] for rid in redirect_ids if rid in resolve
    }
    linktargets = load_linktargets(linktarget_path) if linktarget_path else None

    src_buf = array("i")
    dst_buf = array("i")

    def _add(from_id: int, target_title: str) -> None:
        s = resolve.get(from_id)
        if s is None:
            return
        tid = title_to_id.get(target_title)
        if tid is None:
            return
        d = resolve.get(tid)
        if d is None or d == s:
            return
        src_buf.append(s)
        dst_buf.append(d)

    if linktargets is not None:
        # Fast path: all-integer rows, safe to regex straight off the line.
        prefix = "INSERT INTO `pagelinks` VALUES "
        with open_maybe_compressed(pagelinks_path) as f:
            for line in f:
                if not line.startswith(prefix):
                    continue
                for m in _PL_NEW_RE.finditer(line, len(prefix)):
                    if m.group(2) != "0":
                        continue
                    target_title = linktargets.get(int(m.group(3)))
                    if target_title is not None:
                        _add(int(m.group(1)), target_title)
                if max_edges is not None and len(src_buf) >= max_edges:
                    break
    else:
        for row in iter_sql_rows(pagelinks_path, "pagelinks"):
            if row[1] == "0" and row[2] is not None:
                _add(int(row[0]), row[2])  # type: ignore[arg-type]
            if max_edges is not None and len(src_buf) >= max_edges:
                break

    src = np.frombuffer(src_buf, dtype=np.int32).copy()
    dst = np.frombuffer(dst_buf, dtype=np.int32).copy()
    if symmetrize:
        src, dst = np.concatenate([src, dst]), np.concatenate([dst, src])

    n = len(titles)
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=indptr[1:])
    logger.info("graph: %d nodes, %d edges (symmetrize=%s)", n, len(dst), symmetrize)
    return LinkGraph(
        titles=titles,
        title_to_idx=title_to_idx,
        indptr=indptr,
        indices=dst.astype(np.int32, copy=False),
        alias_to_idx=alias_to_idx,
    )


# ---------------------------------------------------------------------------
# Personalized PageRank
# ---------------------------------------------------------------------------


def personalized_pagerank(
    graph: LinkGraph,
    seed_indices: Iterable[int],
    *,
    alpha: float = 0.85,
    max_iter: int = 40,
    tol: float = 1e-10,
) -> np.ndarray:
    """Power iteration: r = (1-alpha)*s + alpha*(P^T r + dangling*s).

    Dangling mass is returned to the seed vector (standard for PPR), so scores
    stay a probability distribution concentrated around the topic family.
    """
    n = graph.n
    seeds = np.fromiter(set(seed_indices), dtype=np.int64)
    if seeds.size == 0:
        raise ValueError("personalized_pagerank needs at least one resolved seed")
    s = np.zeros(n, dtype=np.float64)
    s[seeds] = 1.0 / seeds.size

    deg = np.diff(graph.indptr).astype(np.float64)
    dangling = deg == 0
    safe_deg = np.maximum(deg, 1.0)

    r = s.copy()
    for it in range(max_iter):
        contrib = r / safe_deg
        w = np.repeat(contrib, np.diff(graph.indptr))
        push = np.bincount(graph.indices, weights=w, minlength=n)
        r_new = (1.0 - alpha) * s + alpha * (push + r[dangling].sum() * s)
        delta = float(np.abs(r_new - r).sum())
        r = r_new
        if delta < tol:
            logger.info("ppr: converged at iter %d (Δ=%.2e)", it + 1, delta)
            break
    return r


def select_family(graph: LinkGraph, scores: np.ndarray, top_n: int) -> list[tuple[str, float]]:
    """Top-n (title, score) pairs with nonzero score, best first."""
    top_n = min(top_n, graph.n)
    idx = np.argpartition(-scores, top_n - 1)[:top_n]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    return [(graph.titles[i], float(scores[i])) for i in idx if scores[i] > 0.0]


def family_vocabulary(graph: LinkGraph, family_titles: Iterable[str]) -> list[str]:
    """Tagging vocabulary: family titles + every redirect alias into them."""
    members = {graph.title_to_idx[t] for t in family_titles if t in graph.title_to_idx}
    vocab = {graph.titles[i] for i in members}
    vocab.update(a for a, i in graph.alias_to_idx.items() if i in members)
    return sorted(t.replace("_", " ") for t in vocab)


# ---------------------------------------------------------------------------
# Citation mining (canon candidates)
# ---------------------------------------------------------------------------

_CITE_START_RE = re.compile(r"\{\{\s*(cite\s+(?:book|journal|conference|arxiv|web)|citation)\b", re.IGNORECASE)
_ISBN_CLEAN_RE = re.compile(r"[^0-9Xx]")


def _template_span(text: str, start: int) -> int | None:
    """End index (exclusive) of the {{...}} template opening at `start`."""
    depth = 0
    i = start
    n = len(text)
    while i < n - 1:
        pair = text[i : i + 2]
        if pair == "{{":
            depth += 1
            i += 2
        elif pair == "}}":
            depth -= 1
            i += 2
            if depth == 0:
                return i
        else:
            i += 1
    return None


def _split_params(body: str) -> list[str]:
    """Split template body on top-level '|' (nested {{ }} and [[ ]] aware)."""
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    i = 0
    n = len(body)
    while i < n:
        two = body[i : i + 2]
        if two in ("{{", "[["):
            depth += 1
            buf.append(two)
            i += 2
        elif two in ("}}", "]]"):
            depth -= 1
            buf.append(two)
            i += 2
        elif body[i] == "|" and depth == 0:
            parts.append("".join(buf))
            buf = []
            i += 1
        else:
            buf.append(body[i])
            i += 1
    parts.append("".join(buf))
    return parts


def extract_citations(wikitext: str) -> list[dict[str, str]]:
    """All {{cite book/journal/...}} / {{citation}} templates as field dicts."""
    out: list[dict[str, str]] = []
    for m in _CITE_START_RE.finditer(wikitext):
        end = _template_span(wikitext, m.start())
        if end is None:
            continue
        body = wikitext[m.start() + 2 : end - 2]
        parts = _split_params(body)
        fields: dict[str, str] = {"kind": parts[0].strip().lower().replace("  ", " ")}
        for p in parts[1:]:
            if "=" not in p:
                continue
            k, _, v = p.partition("=")
            k, v = k.strip().lower(), v.strip()
            if v:
                fields[k] = v
        out.append(fields)
    return out


def _candidate_key(c: dict[str, str]) -> str | None:
    isbn = c.get("isbn")
    if isbn:
        cleaned = _ISBN_CLEAN_RE.sub("", isbn).upper()
        if cleaned:
            return f"isbn:{cleaned}"
    doi = c.get("doi")
    if doi:
        return f"doi:{doi.lower()}"
    title = c.get("title")
    if not title:
        return None
    author = c.get("last") or c.get("last1") or c.get("author") or c.get("author1") or ""
    return f"title:{title.lower()}|{author.lower()}"


def iter_xml_pages(path: str | Path) -> Iterator[tuple[str, str]]:
    """Stream (title, wikitext) for ns0 pages from a pages-articles XML dump."""

    def local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    with open_maybe_compressed(path) as f:
        for _, elem in ElementTree.iterparse(f, events=("end",)):
            if local(elem.tag) != "page":
                continue
            ns_el = title_el = text_el = None
            for child in elem.iter():
                lc = local(child.tag)
                if lc == "ns" and ns_el is None:
                    ns_el = child
                elif lc == "title" and title_el is None:
                    title_el = child
                elif lc == "text":
                    text_el = child
            if ns_el is not None and ns_el.text == "0" and title_el is not None:
                yield title_el.text or "", (text_el.text if text_el is not None else "") or ""
            elem.clear()


def mine_citations(
    xml_path: str | Path, family_titles: Iterable[str]
) -> list[dict[str, Any]]:
    """Aggregate citations across a topic family → ranked canon candidates.

    Titles are matched with spaces (XML form); family titles may be underscored.
    """
    wanted = {t.replace("_", " ") for t in family_titles}
    counts: Counter[str] = Counter()
    examples: dict[str, dict[str, str]] = {}
    pages_seen = 0
    for title, text in iter_xml_pages(xml_path):
        if title not in wanted:
            continue
        pages_seen += 1
        for c in extract_citations(text):
            key = _candidate_key(c)
            if key is None:
                continue
            counts[key] += 1
            examples.setdefault(key, c)
    logger.info("citations: %d family pages scanned, %d unique works", pages_seen, len(counts))
    out = []
    for key, count in counts.most_common():
        c = examples[key]
        author = c.get("last") or c.get("last1") or c.get("author") or c.get("author1") or ""
        out.append(
            {
                "key": key,
                "citations": count,
                "kind": c.get("kind", ""),
                "title": c.get("title", ""),
                "author": author,
                "year": c.get("year") or c.get("date", ""),
                "isbn": c.get("isbn", ""),
                "doi": c.get("doi", ""),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Coverage checklist (candidates vs corpus/seed_index.csv)
# ---------------------------------------------------------------------------

_NORM_RE = re.compile(r"[^a-z0-9 ]")


def _norm_title(t: str) -> str:
    return _NORM_RE.sub("", t.lower()).strip()


def mark_seed_index_coverage(
    candidates: list[dict[str, Any]], seed_index_path: str | Path
) -> list[dict[str, Any]]:
    """Add `in_seed_index` to each candidate by normalized-title match.

    Containment counts as a match (length-guarded), so index annotations like
    "The Feynman Lectures on Physics (3 vols)" still cover the plain title.
    """
    with open(seed_index_path, newline="") as f:
        indexed = [_norm_title(row["title"]) for row in csv.DictReader(f)]
    exact = set(indexed)

    def _covered(title: str) -> bool:
        t = _norm_title(title)
        if t in exact:
            return True
        if len(t) < 10:  # too short for containment to mean anything
            return False
        return any(t in ix or (len(ix) >= 10 and ix in t) for ix in indexed)

    for c in candidates:
        c["in_seed_index"] = _covered(str(c["title"]))
    return candidates
