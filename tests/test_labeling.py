"""Tests for the quality-labeling layer (lithos/data/labeling.py + rubrics)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from lithos.data.labeling import (
    LabelRecord,
    agreement,
    build_prompt,
    parse_label,
    score_histogram,
)

RUBRICS = yaml.safe_load(
    (Path(__file__).parent.parent / "configs" / "quality" / "rubrics.yaml").read_text()
)


# -- rubric config sanity ------------------------------------------------------


def test_rubrics_config_shape():
    assert RUBRICS["version"] == 1
    assert set(RUBRICS["domains"]) == {"physics-eng", "code", "math"}
    for dom in RUBRICS["domains"].values():
        # every level 0-5 must be described in every rubric
        for level in range(6):
            assert f"{level} -" in dom["rubric"], (dom["name"], level)
    assert "SCORE:" in RUBRICS["response_format"]


# -- prompt building -----------------------------------------------------------


def test_build_prompt_includes_rubric_and_truncates():
    rubric = RUBRICS["domains"]["physics-eng"]["rubric"]
    long_doc = "word " * 5000  # 25k chars > 8k budget
    msgs = build_prompt(rubric, RUBRICS["response_format"], long_doc)
    assert msgs[0]["role"] == "system"
    assert "SCORE:" in msgs[0]["content"]
    assert "QUANTITATIVE SUBSTANCE" in msgs[1]["content"]
    assert "[TRUNCATED]" in msgs[1]["content"]
    assert len(msgs[1]["content"]) < 10_000


def test_build_prompt_short_doc_not_marked_truncated():
    msgs = build_prompt("rubric", "fmt", "short doc")
    assert "[TRUNCATED]" not in msgs[1]["content"]


# -- response parsing ----------------------------------------------------------


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("SCORE: 4\nWHY: solid derivation with units", (4, "solid derivation with units")),
        ("score: 0\nwhy: spam", (0, "spam")),  # case-insensitive
        ("Sure! Here is my rating:\nSCORE: 5\nWHY: textbook grade.", (5, "textbook grade.")),
        ("SCORE: 3", (3, "")),  # missing WHY tolerated
        ("SCORE: 7\nWHY: x", None),  # out of range
        ("I think this is a 4.", None),  # no SCORE line
        ("", None),
    ],
)
def test_parse_label(response, expected):
    assert parse_label(response) == expected


# -- agreement + histogram -----------------------------------------------------


def test_agreement_stats():
    stats = agreement([0, 1, 2, 3, 4, 5], [0, 2, 2, 5, 4, 4])
    assert stats["exact"] == pytest.approx(3 / 6)
    assert stats["within_1"] == pytest.approx(5 / 6)
    assert stats["mean_abs_diff"] == pytest.approx(4 / 6)


def test_agreement_rejects_mismatched_or_empty():
    with pytest.raises(ValueError):
        agreement([1], [1, 2])
    with pytest.raises(ValueError):
        agreement([], [])


def test_score_histogram_covers_all_levels():
    assert score_histogram([0, 0, 5, 3]) == {0: 2, 1: 0, 2: 0, 3: 1, 4: 0, 5: 1}


def test_label_record_roundtrip():
    r = LabelRecord(doc_id="abc", domain="code", rubric_version=1, score=4,
                    justification="teaches", labeler="qwen3-32b", source="x.jsonl")
    d = r.to_json()
    assert d["rubric_version"] == 1
    assert LabelRecord(**d) == r
