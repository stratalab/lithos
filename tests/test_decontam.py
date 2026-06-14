"""Tests for n-gram decontamination (PRD §8.9)."""

from lithos.data.decontam import (
    DecontaminationFilter,
    ngrams,
    read_probes,
    scan_contamination,
    write_probes,
)


def test_ngrams_basic():
    g = ngrams("The quick brown fox jumps", n=3)
    assert ("the", "quick", "brown") in g
    assert ("brown", "fox", "jumps") in g
    assert len(g) == 3  # 5 tokens -> 3 trigrams


def test_short_text_falls_back_to_full_gram():
    g = ngrams("only three words", n=13)
    assert g == {("only", "three", "words")}


def test_planted_contaminant_is_flagged_clean_is_not():
    examples = [
        # >= 13 tokens so a real 13-gram exists
        "the mitochondria is the powerhouse of the cell and produces atp for energy use",
        "an entirely separate sentence with totally different words never appearing in the held out corpus",
    ]
    corpus = [
        "intro filler text " + examples[0] + " trailing filler text",  # contains example 0 verbatim
        "the cat sat on the mat while the dog ran around the yard happily all day",  # unrelated
    ]
    rep = scan_contamination(examples, corpus, n=13)
    assert rep["contaminated_indices"] == [0]
    assert rep["num_examples"] == 2
    assert rep["num_contaminated"] == 1
    assert rep["rate"] == 0.5


def test_no_contamination_on_disjoint_corpus():
    examples = ["completely unique benchmark sentence number one with enough tokens to form a thirteen gram here"]
    corpus = ["a totally different document sharing no long spans with the benchmark example at all whatsoever ok"]
    rep = scan_contamination(examples, corpus, n=13)
    assert rep["num_contaminated"] == 0
    assert rep["rate"] == 0.0


def test_decontamination_filter_flags_and_passes():
    probes = ["the mitochondria is the powerhouse of the cell and produces atp for energy use"]
    f = DecontaminationFilter(probes, n=13)
    # a training doc containing a benchmark example is contaminated
    assert f.is_contaminated("preamble " + probes[0] + " postamble") is True
    # a clean doc passes
    assert f.is_contaminated("an unrelated document about gardening tools and the weather this weekend ok") is False
    s = f.stats()
    assert s["probe_examples"] == 1 and s["contaminated_docs"] == 1 and s["n"] == 13


def test_probes_roundtrip(tmp_path):
    texts = ["first benchmark probe text", "second probe with unicode café"]
    p = write_probes(tmp_path / "probes.jsonl", texts)
    assert read_probes(p) == texts
