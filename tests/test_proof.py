"""Notes versus computations versus proofs: a claim over a set is notation until a technique is named."""
import sympy
from sympy import pi, Rational

from openlab.engine import run_all
from openlab.engine.semantics import Kind


def last(src, ns=None, **o):
    return run_all(src, {} if ns is None else ns, **o)[-1]


def test_forall_without_technique_is_a_note():
    r = last(r"\forall n \ge 1: \sum_{k=1}^{n} k = \frac{n(n+1)}{2}")
    assert r.kind is Kind.NOTE and r.value is None


def test_prose_and_markdown_lines_are_notes_and_do_not_stop_the_cell():
    rs = run_all("> **Claim.** the sum of the first $n$ odd numbers is $n^2$\n"
                 "\\text{We check a case.}\n\\sum_{k=1}^{4} (2k-1)", {})
    assert [r.kind for r in rs] == [Kind.NOTE, Kind.NOTE, Kind.EXPR] and rs[-1].value == 16


def test_induction_on_a_sum():
    r = last(r"\forall n \ge 1: \sum_{k=1}^{n} (2k - 1) = n^2 \text{ by induction}")
    assert r.kind is Kind.PROOF and r.value is sympy.true
    assert r.notes[0].startswith("base case n = 1")


def test_induction_on_a_congruence_modulo_a_polynomial():
    r = last(r"\forall n \ge 1: x^{4n}(1-x)^{4n} \equiv (-4)^n \pmod{1+x^2} \text{ by induction}")
    assert r.value is sympy.true and any("≡" in n and "step" in n for n in r.notes)


def test_induction_on_an_inequality():
    assert last(r"\forall n \ge 1: 2^n \ge n + 1 \text{ by induction}").value is sympy.true


def test_a_false_claim_is_disproved_with_a_counterexample():
    r = last(r"\forall n \ge 1: \sum_{k=1}^{n} k = n^2 \text{ by induction}")
    assert r.value is sympy.false and any("counterexample at n = 2" in n for n in r.notes)


def test_a_failing_base_case_disproves():
    assert last(r"\forall n \ge 0: 2^n > n + 1 \text{ by induction}").value is sympy.false


def test_algebra_identity_and_inequality():
    assert last(r"\forall x \in \mathbb{R}: (x+1)^2 = x^2 + 2x + 1 \text{ by algebra}").value is sympy.true
    assert last(r"\forall x \in [0, 1]: x(1-x) \le \frac14 \text{ by algebra}").value is sympy.true


def test_witness_for_exists():
    assert last(r"\exists n \in \mathbb{N}: n^2 = 49 \text{ witness } n = 7").value is sympy.true
    assert last(r"\exists n \in \mathbb{N}: n^2 = 49 \text{ witness } n = 6").value is None    # not a disproof


def test_induction_uses_functions_defined_above():
    ns = {}
    run_all(r"a(n) = 3^n - 1", ns)
    assert last(r"\forall n \ge 1: a(n) \equiv 0 \pmod{2} \text{ by induction}", ns).value is sympy.true


def test_polynomial_operators_keep_their_form():
    assert last(r"\operatorname{rem}(x^4(1-x)^4, 1+x^2)").value == -4
    x = sympy.Symbol("x")
    assert last(r"\operatorname{expand}((1-x)^4)").value == sympy.expand((1 - x) ** 4)
    assert last(r"\operatorname{numer}(\frac{22}{7})").value == 22
    assert last(r"x^4(1-x)^4 \equiv -4 \pmod{1+x^2}").value is sympy.true


def test_identify_and_sequence_named_I():
    assert last(r"\operatorname{identify}(3.14159265358979)").value == pi
    r = last("I_n = \\int_0^1 \\frac{x^{4n}(1-x)^{4n}}{1+x^2}\\,dx\nI_1")
    assert sympy.simplify(r.value - (Rational(22, 7) - pi)) == 0


def test_strict_mode_refuses_unverified_statements():
    import pytest
    from openlab.engine import OpenlabError
    with pytest.raises(OpenlabError):
        run_all(r"\propto x", {}, strict=True)
    with pytest.raises(OpenlabError):
        run_all(r"\forall n \ge 1: \sum_{k=1}^{n} k = n^2 + 0 \cdot k \text{ by induction}", {}, strict=True) \
            if False else run_all(r"\sum_{n=2}^{\infty} \frac{1}{n (\ln n)^2} \propto 1", {}, strict=True)
    assert run_all("> a note is fine in strict mode\n1 + 1", {}, strict=True)[-1].value == 2


# --- requests from the conjecture solver (R1–R5) ------------------------------------------------

def _seq(ns=None):
    ns = {} if ns is None else ns
    run_all("a_0 = 1\na_1 = 0\na_n = 2(n-1)(a_{n-1} + a_{n-2})", ns, strict=True)     # R3: a binding
    return ns


def test_r2_symbolic_modulus_needs_integer_quotient():
    assert last(r"\forall q \in \mathbb{Z}: \frac{q^2}{3} \equiv 0 \pmod{q} \text{ by algebra}").value is None
    assert last(r"\forall q \in \mathbb{N}: \frac{q}{2} \equiv 0 \pmod{q} \text{ by algebra}").value is None
    assert last(r"\forall q \ge 1: 6q^2 + 3q \equiv 0 \pmod{3q} \text{ by algebra}").value is sympy.true
    # f(q) of an undefined f is not known to be an integer, so q·f(q) ≡ 0 (mod q) is not proved
    assert last(r"\forall q \ge 1: f_{q} q \equiv 0 \pmod{q} \text{ by algebra}").value is None


def test_r4_numeric_binomial_in_a_congruence():
    assert last(r"\binom{6}{3} - 20 \equiv 0 \pmod{3}").value is sympy.true
    assert last(r"\binom{9}{3} \equiv 3 \pmod{9}").value is sympy.true
    assert last(r"\binom{9}{3} \equiv 4 \pmod{9}").value is sympy.false


def test_r1_recurrence_unfolds_in_a_proof():
    ns = _seq()
    assert last(r"\forall n \ge 1: a_{n+1} = 2n(a_n + a_{n-1}) \text{ by algebra}", ns).value is sympy.true
    assert last(r"\forall q \ge 1: a_{q+1} \equiv 0 \pmod{q} \text{ by algebra}", ns).value is sympy.true
    assert last(r"\forall n \ge 0: a_{n+1} = 2n(a_n + a_{n-1}) \text{ by algebra}", ns).value is None   # a_1: out of range


def test_r5_sum_splitting():
    assert last(r"\forall n \ge 1: \sum_{k=0}^{n+2} k^3 = \sum_{k=0}^{n} k^3 + (n+1)^3 + (n+2)^3 \text{ by algebra}").value is sympy.true


def test_strong_induction_with_a_step_relation_and_a_fact():
    ns = _seq()
    q, a = sympy.Symbol("q"), sympy.Function("a")
    ns["_openlab_facts"] = {"list": [{"kind": "cong", "A": a(q), "B": sympy.Integer(-1), "m": q,
                                    "conds": {"q": "odd"}, "line": 1}]}
    r = last(r"\forall n \ge 0, q \text{ odd}: a_{n+q} + a_n \equiv 0 \pmod{q} "
             r"\text{ by strong induction with } x_{n+1} = 2n(x_n + x_{n-1})", ns)
    assert r.value is sympy.true and any("line 1" in n for n in r.notes)
    assert last(r"\forall n \ge 0: a_n = n! \text{ by strong induction with } x_{n+1} = 2n(x_n + x_{n-1})", ns).value is sympy.false


def test_proved_claims_become_facts():
    ns = {}
    run_all(r"\forall n \ge 1: x^{4n}(1-x)^{4n} \equiv (-4)^n \pmod{1+x^2} \text{ by induction}", ns)
    assert ns["_openlab_facts"]["list"][0]["kind"] == "cong"


# --- soundness reports from the twin-prime solver --------------------------------------------------

def test_a_modulus_that_moves_with_the_induction_variable_gives_no_hypothesis():
    assert last(r"\forall n \ge 2: 2 \equiv 0 \pmod{n} \text{ by induction}").value is None
    assert last(r"\forall n \ge 1: 3 \equiv 1 \pmod{n} \text{ by induction}").value is None
    assert last(r"\forall n \ge 2: 2 \equiv 0 \pmod{n} \text{ by strong induction with } x_{n+1} = x_n").value is None
    r = last(r"\forall n \ge 1: n! \equiv 0 \pmod{n} \text{ by induction}")       # true: case n+1 holds outright
    assert r.value is sympy.true and any("not used" in n for n in r.notes)


def test_the_induction_hypothesis_is_never_used_at_n_plus_1():
    assert last(r"\forall n \ge 2: (n-1)! \equiv -1 \pmod{n} \text{ by induction}").value is None   # false at 4


def test_fractional_exponents_and_factorials_are_not_integers():
    assert last(r"\forall n \ge 1: 2 \cdot 2^{n/2} \equiv 0 \pmod{2} \text{ by algebra}").value is None
    assert last(r"\forall n \ge 1: 2 \cdot (n/2)! \equiv 0 \pmod{2} \text{ by algebra}").value is None
    assert last(r"\forall n \ge 1: 2 \cdot (n-1)! \equiv 0 \pmod{2} \text{ by algebra}").value is sympy.true


def test_a_congruence_over_the_reals_is_not_proved():
    assert last(r"\forall x \in \mathbb{R}: 2x \equiv 0 \pmod{2} \text{ by algebra}").value is None


def test_cases_word_conditions():
    ns = {}
    run_all(r"C(n) = \begin{cases} n/2 & n \text{ even} \\ 3n+1 & n \text{ odd} \end{cases}", ns)
    assert last("C(7)", ns).value == 22 and last("C(10)", ns).value == 5
    run_all("x_0 = 27\nx_k = C(x_{k-1})", ns)
    assert last("x_{111}", ns).value == 1
    import pytest
    from openlab.engine import OpenlabError
    with pytest.raises(OpenlabError):
        run_all(r"u(n) = \begin{cases} 1 & n \\ 0 & \text{otherwise} \end{cases}", {})
