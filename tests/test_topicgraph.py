"""Tests for the Wikipedia topic-graph job (lithos/data/topicgraph.py)."""

from __future__ import annotations

import gzip
import itertools
from pathlib import Path

import numpy as np
import pytest
from lithos.data import topicgraph as tg

# ---------------------------------------------------------------------------
# A tiny synthetic wiki:
#   1 Physics            -> Quantum_mechanics
#   2 Quantum_mechanics  -> Physics
#   3 QM                 (redirect -> Quantum_mechanics), links -> Thermodynamics
#   4 Cooking            (no links; dangling)
#   5 Thermodynamics     -> Physics
#   6 O'Brien_(physics)  (escaped quote in title; no links)
# ---------------------------------------------------------------------------

PAGE_SQL = (
    "INSERT INTO `page` VALUES "
    "(1,0,'Physics',0),(2,0,'Quantum_mechanics',0),(3,0,'QM',1),"
    "(4,0,'Cooking',0),(5,0,'Thermodynamics',0),(6,0,'O\\'Brien_(physics)',0),"
    "(7,1,'Talk_page',0);\n"
)
REDIRECT_SQL = "INSERT INTO `redirect` VALUES (3,0,'Quantum_mechanics','','');\n"
LINKTARGET_SQL = (
    "INSERT INTO `linktarget` VALUES "
    "(10,0,'Physics'),(11,0,'Quantum_mechanics'),(13,0,'Thermodynamics'),(14,1,'Talk_page');\n"
)
# New (post-2024) pagelinks schema: (pl_from, pl_from_namespace, pl_target_id).
PAGELINKS_NEW_SQL = (
    "INSERT INTO `pagelinks` VALUES (1,0,11),(2,0,10),(3,0,13),(5,0,10),(7,1,10);\n"
)
# Legacy schema: (pl_from, pl_namespace, pl_title).
PAGELINKS_OLD_SQL = (
    "INSERT INTO `pagelinks` VALUES "
    "(1,0,'Quantum_mechanics'),(2,0,'Physics'),(3,0,'Thermodynamics'),(5,0,'Physics');\n"
)


@pytest.fixture
def dumps(tmp_path: Path) -> dict[str, Path]:
    paths = {}
    for name, content in [
        ("page", PAGE_SQL),
        ("redirect", REDIRECT_SQL),
        ("linktarget", LINKTARGET_SQL),
        ("pagelinks", PAGELINKS_NEW_SQL),
        ("pagelinks_old", PAGELINKS_OLD_SQL),
    ]:
        p = tmp_path / f"{name}.sql"
        p.write_text(content)
        paths[name] = p
    return paths


def _build(dumps: dict[str, Path], **kw) -> tg.LinkGraph:
    return tg.build_link_graph(
        dumps["page"], dumps["redirect"], dumps["pagelinks"], dumps["linktarget"], **kw
    )


# -- SQL parsing -------------------------------------------------------------


def test_value_tuple_parser_handles_escapes_and_nulls():
    rows = list(tg._iter_value_tuples("(1,'a\\'b',NULL,2.5),(2,'x\\\\',0,'')"))
    assert rows == [["1", "a'b", None, "2.5"], ["2", "x\\", "0", ""]]


def test_iter_sql_rows_filters_other_tables(tmp_path: Path):
    p = tmp_path / "page.sql"
    p.write_text("-- comment\nINSERT INTO `other` VALUES (9);\n" + PAGE_SQL)
    rows = list(tg.iter_sql_rows(p, "page"))
    assert len(rows) == 7
    assert rows[5][2] == "O'Brien_(physics)"


def test_gzip_transparent(tmp_path: Path):
    p = tmp_path / "page.sql.gz"
    with gzip.open(p, "wt") as f:
        f.write(PAGE_SQL)
    assert len(list(tg.iter_sql_rows(p, "page"))) == 7


def test_load_pages_ns0_and_redirect_flags(dumps):
    title_to_id, id_to_title, redirect_ids = tg.load_pages(dumps["page"])
    assert title_to_id["Physics"] == 1
    assert "Talk_page" not in title_to_id  # ns != 0 dropped
    assert redirect_ids == {3}
    assert id_to_title[6] == "O'Brien_(physics)"


# -- Graph build -------------------------------------------------------------


def test_graph_nodes_exclude_redirects(dumps):
    g = _build(dumps)
    assert g.n == 5  # Physics, QM, Cooking, Thermo, O'Brien
    assert "QM" not in g.title_to_idx
    assert g.resolve("QM") == g.resolve("Quantum mechanics")  # alias + space form


def test_redirect_source_links_remap(dumps):
    # Link (3 -> Thermodynamics) must become (Quantum_mechanics -> Thermodynamics).
    g = _build(dumps, symmetrize=False)
    qm, thermo = g.resolve("Quantum_mechanics"), g.resolve("Thermodynamics")
    out = set(g.indices[g.indptr[qm] : g.indptr[qm + 1]].tolist())
    assert thermo in out


def test_old_pagelinks_format(dumps):
    g_new = _build(dumps, symmetrize=False)
    g_old = tg.build_link_graph(
        dumps["page"], dumps["redirect"], dumps["pagelinks_old"], None, symmetrize=False
    )
    assert g_old.num_edges == g_new.num_edges == 4
    assert g_old.titles == g_new.titles


def test_symmetrize_doubles_edges(dumps):
    assert _build(dumps).num_edges == 2 * _build(dumps, symmetrize=False).num_edges


def test_namespace_links_dropped(dumps):
    # (7,1,10) comes from a ns=1 page: its pl_from resolves to nothing.
    g = _build(dumps, symmetrize=False)
    assert g.num_edges == 4


# -- PPR + selection ---------------------------------------------------------


def test_ppr_is_distribution_and_concentrates_on_seeds(dumps):
    g = _build(dumps)
    scores = tg.personalized_pagerank(g, [g.resolve("Physics")])
    assert scores.shape == (g.n,)
    assert abs(scores.sum() - 1.0) < 1e-6
    # Seed ranks first; connected pages outrank the isolated ones.
    assert int(np.argmax(scores)) == g.resolve("Physics")
    assert scores[g.resolve("Quantum_mechanics")] > scores[g.resolve("Cooking")]
    assert scores[g.resolve("Cooking")] == 0.0


def test_ppr_requires_seeds(dumps):
    with pytest.raises(ValueError):
        tg.personalized_pagerank(_build(dumps), [])


def test_select_family_orders_and_drops_zeros(dumps):
    g = _build(dumps)
    scores = tg.personalized_pagerank(g, [g.resolve("Physics")])
    family = tg.select_family(g, scores, top_n=10)
    titles = [t for t, _ in family]
    assert titles[0] == "Physics"
    assert "Cooking" not in titles  # zero score dropped even under top_n
    assert all(a >= b for (_, a), (_, b) in itertools.pairwise(family))


def test_family_vocabulary_includes_redirect_aliases(dumps):
    g = _build(dumps)
    vocab = tg.family_vocabulary(g, ["Quantum_mechanics", "Physics"])
    assert "Quantum mechanics" in vocab
    assert "QM" in vocab  # the redirect alias
    assert "Thermodynamics" not in vocab


# -- Citation mining ---------------------------------------------------------

WIKITEXT = """
The '''photon''' is fundamental.<ref>{{cite book |last=Griffiths |first=David
|title=Introduction to Elementary Particles |isbn=978-3-527-40601-2 |year=2008}}</ref>
See also.<ref>{{cite journal |last=Einstein |title=Zur Elektrodynamik bewegter
Körper |journal=Annalen der Physik |doi=10.1002/andp.19053221004 |year=1905}}</ref>
Again.<ref>{{Cite book|last=Griffiths|title=Introduction to Elementary Particles
|isbn=9783527406012}}</ref>
Nested: {{citation |title=Some Notes {{sic}} |author=Nobody |year=2000}}
"""


def test_extract_citations_kinds_and_fields():
    cites = tg.extract_citations(WIKITEXT)
    assert len(cites) == 4
    assert cites[0]["kind"] == "cite book"
    assert cites[0]["isbn"] == "978-3-527-40601-2"
    assert cites[1]["doi"] == "10.1002/andp.19053221004"
    assert cites[3]["kind"] == "citation"
    assert cites[3]["title"].startswith("Some Notes")  # nested template survived


def test_isbn_key_normalization_merges_duplicates(tmp_path: Path):
    xml = tmp_path / "articles.xml"
    xml.write_text(
        "<mediawiki>"
        "<page><title>Photon</title><ns>0</ns><revision><text>"
        + WIKITEXT.replace("&", "&amp;").replace("<ref>", "").replace("</ref>", "")
        + "</text></revision></page>"
        "<page><title>Skipped</title><ns>1</ns><revision><text>x</text></revision></page>"
        "</mediawiki>"
    )
    candidates = tg.mine_citations(xml, ["Photon"])
    # Two Griffiths cites share a normalized ISBN key -> one candidate, count 2.
    griffiths = [c for c in candidates if c["isbn"]]
    assert len(griffiths) == 1
    assert griffiths[0]["citations"] == 2
    assert candidates[0] is griffiths[0]  # ranked first


def test_coverage_marking(tmp_path: Path):
    idx = tmp_path / "seed_index.csv"
    idx.write_text(
        'id,title\nx,"Introduction to Elementary Particles"\n'
        'y,"The Feynman Lectures on Physics (3 vols)"\n'
    )
    cands = [
        {"title": "Introduction to Elementary Particles"},
        {"title": "Unknown Work"},
        {"title": "The Feynman Lectures on Physics"},  # containment match
        {"title": "on"},  # too short to containment-match anything
    ]
    marked = tg.mark_seed_index_coverage(cands, idx)
    assert marked[0]["in_seed_index"] is True
    assert marked[1]["in_seed_index"] is False
    assert marked[2]["in_seed_index"] is True
    assert marked[3]["in_seed_index"] is False
