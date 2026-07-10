"""Retrieval-in-context: chunk → embed → Datastore → DocumentRetriever.

R1 as rev B redefines it — above the token stream, cited, never interpolated into the
decode loop (`docs/composite-plan.md` §1).

The three tests that earn their keep:
  * `test_datastore_version_is_derived_from_content` — a corpus change moves the model's
    identity, which is what makes a corpus-caused regression bisectable (C5).
  * `test_eval_in_datastore_is_caught` — `docs/c0-spec.md` §5.1's "single highest-value
    line of code": if the eval set is retrievable, every number is worthless.
  * `test_restricted_is_welcome_and_unknown_is_not` — restricted content belongs in the
    datastore, cited on every use; it is barred only from the weights.
"""

from __future__ import annotations

import hashlib
import re
from types import SimpleNamespace

import numpy as np
import pytest
from lithos.retrieval import (
    Chunk,
    Datastore,
    DocumentRetriever,
    HashingEmbedder,
    NumpyExactIndex,
    Retriever,
    chunk_document,
)


class WordTok:
    """A whitespace tokenizer with exact decode round-trip — enough to chunk by tokens."""

    def encode(self, text):
        return SimpleNamespace(ids=[abs(hash(w)) % 10_000 for w in text.split()])

    def decode(self, ids, skip_special_tokens=True):  # pragma: no cover - unused
        raise NotImplementedError


class CharTok:
    """id = ord(c); decode is exact, so a token window is a real substring."""

    def encode(self, text):
        return SimpleNamespace(ids=[ord(c) for c in text])

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


def _doc(text, *, source_id="src:a", tier="open", rec="rec:a"):
    return {
        "id": rec,
        "text": text,
        "source": source_id,
        "tier": tier,
        "metadata": {
            "source_id": source_id,
            "record_id": rec,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        },
    }


# ── chunking ──────────────────────────────────────────────────────────────────


def test_chunking_covers_the_document_and_carries_provenance():
    tok = CharTok()
    text = "abcdefghij"
    chunks = chunk_document(_doc(text), tok, max_tokens=4, overlap_tokens=1)
    assert "".join(c.text for c in chunks) != ""
    # every character of the source appears in some chunk
    assert set(text) <= set("".join(c.text for c in chunks))
    for i, c in enumerate(chunks):
        assert c.chunk_index == i
        assert c.source_id == "src:a" and c.record_id == "rec:a" and c.tier == "open"
        assert c.text_sha256 == hashlib.sha256(text.encode()).hexdigest()
        assert c.chunk_sha256 == hashlib.sha256(c.text.encode()).hexdigest()
        assert 0 < c.n_tokens <= 4


def test_chunks_overlap_so_a_fact_on_a_boundary_survives():
    tok = CharTok()
    chunks = chunk_document(_doc("abcdefgh"), tok, max_tokens=4, overlap_tokens=2)
    assert len(chunks) >= 2
    # consecutive windows share `overlap_tokens` characters
    assert chunks[0].text[-2:] == chunks[1].text[:2]


def test_short_document_yields_exactly_one_chunk():
    chunks = chunk_document(_doc("abc"), CharTok(), max_tokens=64, overlap_tokens=8)
    assert len(chunks) == 1 and chunks[0].text == "abc"


def test_empty_document_yields_nothing():
    assert chunk_document(_doc(""), CharTok()) == []


def test_degenerate_chunk_params_raise():
    with pytest.raises(ValueError, match="must be <"):
        chunk_document(_doc("abc"), CharTok(), max_tokens=4, overlap_tokens=4)
    with pytest.raises(ValueError, match="positive"):
        chunk_document(_doc("abc"), CharTok(), max_tokens=0)


# ── embedding ─────────────────────────────────────────────────────────────────


def test_embeddings_are_unit_norm_and_deterministic():
    e = HashingEmbedder(dim=64)
    v = e.encode(["bernoulli principle fluid"])
    assert v.shape == (1, 64)
    assert np.isclose(np.linalg.norm(v[0]), 1.0, atol=1e-6)
    assert np.allclose(v, e.encode(["bernoulli principle fluid"]))


def test_embedder_does_not_use_pythons_salted_hash():
    """`hash()` on str is salted per process; a datastore built in one process would not
    match a query embedded in another, and it would look like bad retrieval."""
    import subprocess
    import sys

    code = (
        "from lithos.retrieval import HashingEmbedder;"
        "import numpy as np;"
        "print(float(HashingEmbedder(dim=32).encode(['bernoulli'])[0][:3].sum()))"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        ).stdout.strip()
        for seed in ("0", "1", "12345")
    }
    assert len(runs) == 1, f"embedding varied with PYTHONHASHSEED: {runs}"


def test_empty_text_embeds_to_zeros_not_nan():
    v = HashingEmbedder(dim=16).encode(["...", ""])
    assert not np.isnan(v).any()
    assert np.allclose(v, 0.0)


def test_similar_text_scores_higher_than_dissimilar():
    e = HashingEmbedder(dim=256)
    q = e.encode(["bernoulli principle in fluid dynamics"])[0]
    docs = e.encode(["bernoulli fluid dynamics pressure", "the mitochondrion is an organelle"])
    assert float(docs[0] @ q) > float(docs[1] @ q)


# ── the index ─────────────────────────────────────────────────────────────────


def test_exact_index_returns_nearest_first():
    vecs = np.eye(4, dtype=np.float32)
    idx = NumpyExactIndex(vecs)
    hits = idx.search(np.array([0, 1, 0, 0], dtype=np.float32), k=2)
    assert hits[0][0] == 1 and np.isclose(hits[0][1], 1.0)
    assert hits[1][1] <= hits[0][1]


def test_empty_index_returns_nothing():
    assert NumpyExactIndex(np.zeros((0, 4), dtype=np.float32)).search(np.zeros(4), 3) == []


def test_index_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        NumpyExactIndex(np.zeros(4, dtype=np.float32))


# ── the datastore: gates and identity ─────────────────────────────────────────


def _store(texts, *, tier="open", dim=128, max_tokens=64):
    e = HashingEmbedder(dim=dim)
    docs = [_doc(t, source_id=f"src:{i}", rec=f"rec:{i}", tier=tier) for i, t in enumerate(texts)]
    return (
        Datastore.build(
            docs, CharTok(), e, tokenizer_name="chartok", max_tokens=max_tokens, overlap_tokens=8
        ),
        e,
    )


def test_restricted_is_welcome_and_unknown_is_not():
    """Restricted content belongs here — cited on every use. It is barred from the weights,
    not from the datastore (`docs/chisel-tier-gate.md`)."""
    _store(["a textbook passage"], tier="restricted")  # allowed
    with pytest.raises(ValueError, match="tier='unknown'"):
        _store(["a passage of unclear provenance"], tier="unknown")


def test_datastore_version_is_derived_from_content():
    """Change one document and the identity moves — so C5 (bisect a corpus-caused
    regression) is possible, and results from two corpora cannot be silently pooled."""
    a, _ = _store(["alpha beta", "gamma delta"])
    b, _ = _store(["alpha beta", "gamma delta"])
    c, _ = _store(["alpha beta", "gamma DELTA"])
    assert a.version == b.version  # same content -> same identity
    assert a.version != c.version  # one edited doc -> different identity
    assert a.version.startswith("ds:")


def test_datastore_version_is_invariant_to_ingest_order():
    a, _ = _store(["alpha beta", "gamma delta"])
    b, _ = _store(["gamma delta", "alpha beta"])
    assert a.version == b.version


def test_datastore_version_moves_with_the_embedder_and_the_chunking():
    docs = [_doc("alpha beta gamma delta epsilon")]
    base = Datastore.build(docs, CharTok(), HashingEmbedder(dim=64), max_tokens=8, overlap_tokens=2)
    other_embedder = Datastore.build(
        docs, CharTok(), HashingEmbedder(dim=128), max_tokens=8, overlap_tokens=2
    )
    other_chunking = Datastore.build(
        docs, CharTok(), HashingEmbedder(dim=64), max_tokens=16, overlap_tokens=2
    )
    assert len({base.version, other_embedder.version, other_chunking.version}) == 3


def test_eval_in_datastore_is_caught():
    """If the eval set is retrievable, retrieval returns the answer verbatim."""
    text = "the answer to the held-out question is 42"
    store, _ = _store([text])
    eval_sha = hashlib.sha256(text.encode()).hexdigest()

    store.assert_disjoint_from(["deadbeef" * 8])  # unrelated eval set: fine
    with pytest.raises(ValueError, match="Retrieval would return the answer verbatim"):
        store.assert_disjoint_from([eval_sha])


def test_chunk_vector_count_mismatch_raises():
    c = Chunk("t", "s", "r", "h", "open", 0, "cs", 1)
    with pytest.raises(ValueError, match="1 chunks but 2 vectors"):
        Datastore([c], np.zeros((2, 4), dtype=np.float32), embedder_version="v", chunk_params={})


def test_manifest_attests_what_is_indexed_including_restricted():
    store, _ = _store(["a passage from a textbook"], tier="restricted")
    man = store.manifest()
    assert man["tiers"] == {"restricted": 1}
    assert man["num_documents"] == 1 and man["num_chunks"] >= 1
    assert man["datastore_version"] == store.version


def test_save_load_roundtrip_and_version_is_verified_on_load(tmp_path):
    store, _ = _store(["alpha beta gamma", "delta epsilon zeta"])
    store.save(tmp_path)
    back = Datastore.load(tmp_path)
    assert back.version == store.version
    assert [c.chunk_sha256 for c in back.chunks] == [c.chunk_sha256 for c in store.chunks]
    assert np.allclose(back.index.vectors, store.index.vectors)


def test_load_detects_content_tampering(tmp_path):
    """The stored chunk_sha256 is recomputed, never trusted — otherwise `version` would be
    a hash of a lie and every attestation built on it would inherit the lie."""
    store, _ = _store(["alpha beta gamma"])
    store.save(tmp_path)
    p = tmp_path / "chunks.jsonl"
    p.write_text(re.sub(r'"text":"[^"]*"', '"text":"TAMPERED"', p.read_text(), count=1))
    with pytest.raises(ValueError, match="content and its identity disagree"):
        Datastore.load(tmp_path)


def test_a_chunk_cannot_be_constructed_with_a_hash_of_other_text():
    bad = Chunk("hello", "s", "r", "h", "open", 0, hashlib.sha256(b"goodbye").hexdigest(), 1)
    with pytest.raises(ValueError, match="content and its identity disagree"):
        Datastore([bad], np.zeros((1, 4), dtype=np.float32), embedder_version="v", chunk_params={})


def test_load_detects_a_tampered_manifest(tmp_path):
    """Editing the recorded chunk params moves the true version; the stored one no longer matches."""
    store, _ = _store(["alpha beta gamma"])
    store.save(tmp_path)
    m = tmp_path / "datastore_manifest.json"
    m.write_text(m.read_text().replace('"max_tokens": 64', '"max_tokens": 99'))
    with pytest.raises(ValueError, match="datastore version mismatch on load"):
        Datastore.load(tmp_path)


# ── the retriever ─────────────────────────────────────────────────────────────


def test_retriever_satisfies_the_protocol_and_pins_the_datastore():
    store, e = _store(["alpha"])
    r = DocumentRetriever(store, e)
    assert isinstance(r, Retriever)
    assert r.datastore_version == store.version


def test_retriever_rejects_a_mismatched_embedder():
    store, _ = _store(["alpha"], dim=64)
    with pytest.raises(ValueError, match="embedder mismatch"):
        DocumentRetriever(store, HashingEmbedder(dim=128))


def test_retriever_finds_the_relevant_passage():
    store, e = _store(
        [
            "bernoulli principle pressure velocity fluid",
            "the mitochondrion is the powerhouse of the cell",
            "kirchhoff current law node analysis",
        ],
        max_tokens=256,
    )
    r = DocumentRetriever(store, e, top_k=1)
    ctx = r.retrieve("what does bernoulli say about fluid pressure?", token_budget=512)
    assert len(ctx.passages) == 1
    assert "bernoulli" in ctx.passages[0].text
    assert ctx.passages[0].score > 0
    assert ctx.tokens_used > 0


def test_zero_budget_retrieves_nothing():
    store, e = _store(["alpha beta"])
    assert DocumentRetriever(store, e).retrieve("alpha", token_budget=0).passages == ()


def test_budget_bounds_the_number_of_candidates():
    """A 40-passage answer to a 1-passage budget is wasted work and a misleading count."""
    store, e = _store([f"topic alpha document number {i}" for i in range(8)], max_tokens=32)
    r = DocumentRetriever(store, e, top_k=8)
    few = r.retrieve("topic alpha", token_budget=32)
    many = r.retrieve("topic alpha", token_budget=32 * 8)
    assert len(few.passages) < len(many.passages)


def test_noise_passages_are_not_returned():
    """A zero-similarity chunk only costs context. min_score drops it."""
    store, e = _store(["bernoulli fluid pressure", "zzz qqq"], max_tokens=256)
    r = DocumentRetriever(store, e, top_k=2)
    ctx = r.retrieve("xylophone unrelated quokka", token_budget=1024)
    assert all(p.score > 0 for p in ctx.passages)
