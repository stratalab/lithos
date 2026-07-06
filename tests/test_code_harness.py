"""Shared code-harness golden fixtures — the `kind=code` contract with Chisel (F7 G2).

Every `(solution, tests, expect)` triple in `code_harness_golden.jsonl` must produce
`expect` when run through the real `check_code` at the pinned **CHECKER_IMPORT_SET =
{stdlib, numpy, scipy, sympy}**. Chisel mirrors this file byte-identical and proves the
same verdicts through its verify seam; the byte-identity + these tests catch drift, so a
Chisel-authored `tests` string provably runs identically under our runner *before* G2
mines a repo. Covers the tricky cases (empty-stub, wrong-answer, exception, timeout,
assert-with-message, seeded determinism) and proves numpy/scipy/sympy are actually present.
See `docs/chisel-f7-response.md` §3.
"""

import json
from pathlib import Path

import pytest
from lithos.posttrain.verifier import check_code

FIXTURES = Path(__file__).parent / "fixtures" / "code_harness_golden.jsonl"


def _load() -> list[dict]:
    with open(FIXTURES, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


_CASES = _load()


@pytest.mark.parametrize("case", _CASES, ids=[c["name"] for c in _CASES])
def test_code_harness_verdict(case: dict) -> None:
    result, _ = check_code(case["solution"], case["tests"], timeout_s=case.get("timeout_s", 5.0))
    expected = case["expect"] == "pass"
    assert result.correct is expected, (
        f"{case['name']}: expected {case['expect']!r}, got correct={result.correct} "
        f"(detail: {result.detail!r})"
    )


def test_fixture_set_is_nonempty_and_covers_both_verdicts() -> None:
    verdicts = {c["expect"] for c in _CASES}
    assert verdicts == {"pass", "fail"}, "fixtures must exercise both pass and fail"
    assert len(_CASES) >= 8, "keep the tricky-case coverage"
