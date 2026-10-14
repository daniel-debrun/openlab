"""Proof-arm shakedown, round 1 (2026-09-14): Bernoulli's inequality by induction with a real parameter.

- R1: a base case that keeps a real variable;
- R2: factors ≥ 0 by their conditions;
- R3: the multiplicative inductive step;
- R4: an exact one-variable polynomial check.

Also: the reading shown for a claim with no method, and controls that must never come out proved."""

import pytest

import openlab.engine as openlab


def verdict(src):
    r = list(openlab.iter_run(src, {}))[-1]
    return r.interpretation.split(":")[0], r


@pytest.mark.parametrize("src", [
    r"\forall n \ge 1, x \ge -1: (1+x)^n \ge 1 + n x \text{ by induction on n}",                  # Bernoulli: R1 + R3
    r"\forall n \ge 1, x \ge 0: (1+x)^n \ge 1 + n x \text{ by induction on n}",
    r"\forall n \ge 1, x \in \mathbb{R}: n x^2 \ge 0 \text{ by algebra}",                             # R2
    r"\forall n \ge 1, x \ge -1: (1+x)(1 + n x) \ge 1 + (n+1) x \text{ by algebra}",                 # R2: the step
    r"\forall n \ge 1, x \ge -1: (1+x)(1 + n x) \ge 1 + (n+1) x \text{ by algebra with } n x^2",     # a hint of that form
    r"\forall x \ge -1: (1+x)^6 \ge 1 + 6 x \text{ by algebra}",                                      # R4
    r"\forall x \ge -1: (1+x)^8 \ge 1 + 8 x \text{ by algebra}",
])
def test_proved(src):
    v, r = verdict(src)
    assert v.startswith("proved"), r.interpretation


@pytest.mark.parametrize("src", [
    r"\forall n \ge 1, x \in \mathbb{R}: (1+x)^n \ge 1 + n x \text{ by induction on n}",       # false: n = 3, x = −4
    r"\forall x \in \mathbb{R}: x^3 \ge x \text{ by algebra}",
    r"\forall x \ge -1: (1+x)^6 > 1 + 6 x \text{ by algebra}",                                 # equal at x = 0
    r"\forall n \ge 1, x \ge -1: n x \ge 0 \text{ by algebra}",
    r"\forall n \ge 1, x \ge 0: (1+x)^n \ge 1 + n x + x^2 \text{ by induction on n}",          # fails at n = 1
    r"\forall n \ge 1, x \ge -1: (1+x)^n \ge 1 + n x + x \text{ by induction on n}",
    r"\forall n \ge 1, x \ge -4: (1+x)^n \ge 1 + n x \text{ by induction on n}",               # false: n = 3, x = −4
    r"\forall x \ge -1: (1+x)^3 \ge 1 + 4 x \text{ by algebra}",
])
def test_never_proved(src):
    v, r = verdict(src)
    assert not v.startswith("proved"), r.interpretation


def test_counterexamples_are_exact():
    v, r = verdict(r"\forall x \ge -1: (1+x)^6 > 1 + 6 x \text{ by algebra}")
    assert v.startswith("disproved") and "x = 0" in " ".join(r.notes)
    v, r = verdict(r"\forall n \ge 1, x \ge 0: (1+x)^n \ge 1 + n x + x^2 \text{ by induction on n}")
    assert v.startswith("disproved") and "fails at n = 1" in " ".join(r.notes)


def test_a_polynomial_counterexample_is_a_short_number():
    v, r = verdict(r"\forall x \ge -1: (1+x)^6 \ge 1 + 6 x + 16 x^2 \text{ by algebra}")
    note = " ".join(r.notes)
    assert v.startswith("disproved") and "counterexample at x = " in note and len(note) < 200 and "sqrt" not in note


def test_a_claim_without_a_method_shows_how_it_was_read():
    r = list(openlab.iter_run(r"\forall n \ge 1, x \ge -1: (1+x)^n \ge 1 + n x", {}))[-1]
    assert "read as: ∀ n ≥ 1, x ≥ -1: (1 + x)**n ≥ 1 + n*x" in r.interpretation
