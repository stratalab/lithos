"""Tests for the general per-task verifiers (lithos/posttrain/verifier.py, E1b).

The arithmetic test-bench MathVerifier keeps its own tests (test_verifier.py);
this covers the CheckResult-based numeric/symbolic/code/units primitives + the
RLVR shaping + the gaming pre-screen.
"""

import sys

import pytest
from lithos.posttrain.verifier import (
    CheckResult,
    check_code,
    check_numeric,
    check_symbolic,
    check_units,
    extract_final,
    extract_number,
    heuristic_gaming_check,
    shaped_reward,
)

_POSIX = sys.platform.startswith(("linux", "darwin"))


def test_extract_final_takes_post_think():
    assert extract_final("<think>lots of reasoning</think>  42 apples") == "42 apples"
    assert extract_final("no think block here") == "no think block here"


def test_extract_number_handles_formats():
    assert extract_number("the answer is 1,024") == 1024.0
    assert extract_number("x = 3.14159") == 3.14159
    assert extract_number("about 6.02e23 particles") == 6.02e23
    assert extract_number("negative -7 below") == -7.0
    assert extract_number("no digits") is None


def test_check_numeric_tolerance():
    assert check_numeric("result: 100.0000001", "100").correct
    assert not check_numeric("result: 101", "100").correct
    # absolute floor lets a zero target pass
    assert check_numeric("answer 0.0000000001", "0").correct
    bad = check_numeric("no number", "5")
    assert not bad.correct and bad.extracted is None


def test_check_numeric_reports_extracted():
    r = check_numeric("<think>...</think> the answer is 8", "8")
    assert r.correct
    assert r.extracted == "8.0"


def test_check_symbolic_equivalence():
    assert check_symbolic("1/2", "0.5").correct
    assert check_symbolic("(x-1)*(x+1)", "x**2 - 1").correct
    assert not check_symbolic("x + 1", "x - 1").correct
    assert not check_symbolic("", "x").correct  # empty response


def test_check_symbolic_uses_final_line():
    resp = "<think>expand it</think>\nThe factored form is\nx**2 - 1"
    assert check_symbolic(resp, "(x-1)*(x+1)").correct


def test_check_symbolic_bad_parse_is_not_correct():
    r = check_symbolic(")(+ not an expression", "x")
    assert not r.correct


@pytest.mark.skipif(not _POSIX, reason="POSIX-only sandbox")
def test_check_code_passes_and_fails():
    solution = "def add(a, b):\n    return a + b"
    ok, execu = check_code(solution, "assert add(2, 3) == 5\nassert add(-1, 1) == 0")
    assert ok.correct and execu.ok
    bad, execb = check_code(solution, "assert add(2, 3) == 6")
    assert not bad.correct and not execb.ok
    assert "AssertionError" in execb.stderr


@pytest.mark.skipif(not _POSIX, reason="POSIX-only sandbox")
def test_check_code_timeout_is_failure():
    r, _ = check_code("def f():\n    while True: pass", "f()", timeout_s=1.0)
    assert not r.correct


def test_check_units_guarded_without_pint():
    pytest.importorskip  # noqa: B018 — availability handled below
    try:
        import pint  # noqa: F401
    except ImportError:
        r = check_units("198.7 kPa", "198.7", units="kPa")
        assert not r.correct
        assert "pint" in r.detail
    else:
        r = check_units("the pressure is 198.7 kPa", "198.7", units="kPa")
        assert r.correct


def test_shaped_reward_orders_and_docks():
    correct = shaped_reward("42", CheckResult(True, extracted="42"))
    wrong_parseable = shaped_reward("41", CheckResult(False, extracted="41"))
    no_answer = shaped_reward("dunno", CheckResult(False, extracted=None))
    loopy = shaped_reward("42 42 42 42", CheckResult(True, extracted="42"))
    assert correct > wrong_parseable > no_answer
    assert loopy < correct


def test_heuristic_gaming_flags_hardcode_and_noop():
    assert heuristic_gaming_check("print(42)", "42") is not None  # literal answer, no math
    assert heuristic_gaming_check("", "42") == "empty tool call"
    assert heuristic_gaming_check("print(6 * 7)", "42") is None  # real computation
    assert heuristic_gaming_check("import sympy\nprint(sympy.factor('x**2-1'))", "x") is None
