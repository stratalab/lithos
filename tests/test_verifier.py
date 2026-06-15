"""Tests for the RLVR verifier + arithmetic task generator (Phase 11)."""

import re

from lithos.posttrain.verifier import MathVerifier, gen_arithmetic


def test_extracts_last_integer():
    v = MathVerifier()
    assert v.extract("The answer is 12.") == 12
    assert v.extract("3 + 5 = 8") == 8  # last int wins (model restates then answers)
    assert v.extract("1,024 apples") == 1024  # commas stripped
    assert v.extract("it is -4 below zero") == -4
    assert v.extract("no numbers here") is None


def test_correctness_is_strict_0_1():
    v = MathVerifier()
    assert v.correctness("The answer is 12", "12") == 1.0
    assert v.correctness("hmm, 11 maybe", "12") == 0.0
    assert v.correctness("no number at all", "12") == 0.0


def test_shaped_reward_orders_responses_and_docks_looping():
    v = MathVerifier()
    correct = v.reward("12", "12")            # correct + clean number
    wrong_clean = v.reward("11", "12")        # wrong but a clean number
    no_number = v.reward("uhh dunno", "12")   # no number at all
    loopy = v.reward("12 12 12 12 12", "12")  # correct but repetitive
    assert correct > wrong_clean > no_number  # shaping densifies the gradient
    assert correct > 1.0                      # correctness + format bonus
    assert loopy < correct                    # repetition is penalized


def test_gen_arithmetic_deterministic_and_sized():
    assert gen_arithmetic(50, seed=1) == gen_arithmetic(50, seed=1)
    assert len(gen_arithmetic(50, seed=1)) == 50


def test_gen_arithmetic_answers_are_correct():
    for t in gen_arithmetic(200, seed=3, ops="+-*"):
        m = re.match(r"What is (-?\d+) ([+\-*]) (-?\d+)\?", t["prompt"])
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        expected = {"+": a + b, "-": a - b, "*": a * b}[op]
        assert int(t["answer"]) == expected


def test_subtraction_stays_non_negative():
    for t in gen_arithmetic(200, seed=2, ops="-"):
        assert int(t["answer"]) >= 0
