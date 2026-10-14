"""Regression tests: the silent misreads that round-2 QA found (student, engineer and
adversarial personas). Each case is the minimal repro and the mathematically right answer."""
import os, sys
import pytest
import sympy as sp
sys.path.insert(0, os.environ.get("OPENLAB_SRC", "."))
import openlab.engine as openlab
from sympy import Symbol, Rational, exp, sin, Matrix, symbols, FiniteSet, Interval, oo
x, y, t, k, n, z = symbols("x y t k n z")

def last(r): return r[-1] if isinstance(r, list) else r
def run(src, ns=None, **o): return openlab.run(src, {} if ns is None else ns, **o)
def val(src, ns=None, **o): return last(run(src, ns, **o)).value
def num(v): return complex(sp.N(v))
def close(a, b, tol=1e-9): return abs(num(a) - num(b)) < tol
def raises(src, ns=None):
    try: run(src, ns); return False
    except openlab.OpenlabError: return True

def ns_after(src, ns=None):
    ns = {} if ns is None else ns
    run(src, ns); return ns

A2 = Matrix([[1, 2], [3, 4]])
CASES = [
 # --- student ---------------------------------------------------------------------------------
 ("\\left( line not glued",       lambda: (lambda r: r[0].value == 4 and r[1].value == 9)(run("g = 4\n\\left(1+2\\right)^2"))),
 ("1/(n(n+1)) with n = 2",        lambda: val(r"\frac{1}{n(n+1)}", {"n": 2}) == Rational(1, 6)),
 ("telescoping sum to oo",        lambda: val(r"\sum_{n=1}^\infty \frac{1}{n(n+1)}") == 1),
 ("telescoping sum to 3",         lambda: val(r"\sum_{k=1}^3 \frac{1}{k(k+1)}") == Rational(3, 4)),
 ("set equality",                 lambda: val(r"\{2,3\} = \{2,3\}") is sp.true),
 ("A cap B = {2,3}",              lambda: val("A = \\{1,2,3\\}\nB = \\{2,3,4\\}\nA \\cap B = \\{2,3\\}") is sp.true),
 ("(fg)(2)",                      lambda: val("f(x) = x^2\ng(x) = x + 1\n(fg)(2)") == 12),
 ("(f/g)(x)",                     lambda: sp.simplify(val("f(x) = x^2\ng(x) = x + 1\n(f/g)(x)") - x**2/(x + 1)) == 0),
 ("stated P not overwritten",     lambda: val("P(A) = 0.3\nP(B) = 0.5\nP(A \\cap B) = 0.1\nP(A \\cap B) = P(A)P(B)") is sp.false),
 ("P(A cup B) after that",        lambda: close(val("P(A) = 0.3\nP(B) = 0.5\nP(A \\cap B) = 0.1\nP(A \\cap B) = P(A)P(B)\nP(A \\cup B)"), 0.7)),
 ("|x| = x solved",               lambda: val(r"|x| = x") == Interval(0, oo)),
 ("\\text{at} x = 2",             lambda: val(r"\frac{d}{dx}(x^3) \text{ at } x = 2") == 12),
 ("stale x note",                 lambda: "bound" in last(run(r"x^2 - 5x + 6 = 0", {"x": 5})).interpretation),
 ("--for bound x",                lambda: val(r"x^2 - 5x + 6 = 0", {"x": 5}, solve_for="x") == [2, 3]),
 ("Var identity represented",     lambda: last(run(r"\operatorname{Var}(X) = E[X^2] - E[X]^2")).kind.name == "REPR"),
 ("P(n,k) defines",               lambda: val("P(n,k) = \\frac{n!}{(n-k)!}\nP(5,2)") == 20),
 ("(p land q)",                   lambda: val(r"(p \land q)", {"p": True, "q": False}) is sp.false),
 ("neg(3 > 2)",                   lambda: val(r"\neg(3 > 2)") is sp.false),
 ("recurrence a_n = a_{n-1}+2",   lambda: val("a_0 = 1\na_n = a_{n-1} + 2\na_5") == 11),
 ("ratio test limit",             lambda: val("a_n = \\frac{1}{2^n}\n\\lim_{n \\to \\infty} \\left|\\frac{a_{n+1}}{a_n}\\right|") == Rational(1, 2)),
 ("bar{mathrm{data}}",            lambda: val(r"\bar{\mathrm{data}}", {"data": [1, 2, 3]}) == 2),
 ("real cube root",               lambda: val(r"\sqrt[3]{-8}") == -2),
 ("v = 3 then v(t) = 9.8t",       lambda: close(val("v = 3\nv(t) = 9.8t\nv(2)"), 19.6)),
 ("total probability",            lambda: close(val("P(D) = 0.01\nP(T \\mid D) = 0.99\nP(T \\mid D') = 0.05\nP(T)"), 0.0594)),
 ("Bayes",                        lambda: close(val("P(D) = 0.01\nP(T \\mid D) = 0.99\nP(T \\mid D') = 0.05\nP(D \\mid T)"), 0.0099 / 0.0594)),
 ("lambda > 0 assumption",        lambda: last(run(r"\lambda > 0")).kind.name == "ASSUME"),
 ("|A| of a set",                 lambda: val("A = \\{1,2,3\\}\n|A|") == 3),
 ("trailing period",              lambda: val(r"x^2 = 4.") == [-2, 2]),
 ("[F]_0^2 bracket bar",          lambda: val(r"[\frac{x^3}{3}]_0^2") == Rational(8, 3)),
 # --- adversarial -----------------------------------------------------------------------------
 ("x^{-1}(x+1) product",          lambda: sp.simplify(val(r"x^{-1}(x+1)") - (x + 1) / x) == 0),
 ("x^2(y) product",               lambda: val(r"x^2(y)") == x**2 * y),
 ("I_0 is a name",                lambda: not isinstance(val("I(t) = I_0 \\sin(\\omega t)\nI(1)"), sp.MatrixBase)),
 ("sequence index i",             lambda: val("x_i = i^2\nx_3") == 9),
 ("scoped substitution",          lambda: val(r"\int_0^1 x\,dx + x", {"x": 3}) == Rational(7, 2)),
 ("scoped sum",                   lambda: val(r"\sum_{k=1}^3 k + k", {"k": 5}) == 11),
 ("no unit after a fraction",     lambda: val(r"\frac12\mathrm{m}v^2", {"m": 2, "v": 3}) == 9),
 ("unicode root",                 lambda: val("√16") == 4),
 ("1,000.5",                      lambda: close(val("1,000.5"), 1000.5)),
 ("{1,000}",                      lambda: val(r"\{1,000\}") == FiniteSet(1000)),
 ("T_0 not a sequence term",      lambda: "T_0" in str(val("T(t) = 20 + T_0 e^{-kt}\nT(0)"))),
 ("P(1+r)^t, P unbound",          lambda: val(r"P(1+r)^t").has(Symbol("P")) and not val(r"P(1+r)^t").has(sp.Function("P"))),
 ("qualified identity",           lambda: "positive" in last(run(r"(x^a)^b = x^{ab}")).latex_out),
 ("int x dx = x^2 is False",      lambda: val(r"\int x\,dx = x^2") is sp.false),
 ("int 2x dx = x^2 + 5 True",     lambda: val(r"\int 2x\,dx = x^2 + 5") is sp.true),
 ("e^A matrix exponential",       lambda: val(r"e^A", {"A": Matrix([[0, 1], [0, 0]])}) == Matrix([[1, 1], [0, 1]])),
 ("non-square A^2 refused",       lambda: raises(r"A^2", {"A": Matrix([[1, 2, 3], [4, 5, 6]])})),
 ("y = 2x+1, y = -x+4",           lambda: val("y = 2x + 1\ny = -x + 4") == [1]),
 ("circle and y = x",             lambda: len(val("x^2 + y^2 = 1\ny = x")) == 2),
 ("a = b = 2",                    lambda: (lambda ns: ns["a"] == 2 and ns["b"] == 2)(ns_after("a = b = 2"))),
 ("x = 1 or y = 2 binds nothing", lambda: "x" not in ns_after(r"x = 1 \text{ or } y = 2")),
 ("c'(x) of a constant",          lambda: val(r"c'(x)", {"c": 3}) == 0),
 ("cycle detected",               lambda: raises(r"a + 1", {"a": Symbol("b") + 1, "b": Symbol("a") - 1})),
 ("IC on next line",              lambda: sp.simplify(val("y' = -y\ny(0) = 5")[0] - 5 * exp(-x)) == 0),
 ("x^100 - 1 real roots",         lambda: val(r"x^{100} - 1 = 0") == [-1, 1]),
 ("tuple sum",                    lambda: list(val(r"(1,2) + (3,4)")) == [4, 6]),
 ("sympy function in ns",         lambda: val(r"f(\pi/2)", {"f": sp.sin}) == 1),
 ("-p 0 refused",                 lambda: raises(r"\pi", None) is False and (lambda: (_r(r"\pi", precision=0)))()),
 # --- engineer --------------------------------------------------------------------------------
 ("cubic: all three roots",       lambda: len(val(r"s^3 + 3s^2 + 4s + 2 = 0")) == 3),
 ("e^{At} exponential",           lambda: sp.simplify(val(r"e^{A t}", {"A": Matrix([[0, 1], [0, 0]])}) - Matrix([[1, t], [0, 1]])) == sp.zeros(2)),
 ("coupled ODEs",                 lambda: len(val("\\dot x = -y\n\\dot y = x")[0]) == 2),
 ("grad at a point",              lambda: val(r"\left. \nabla (x^2 + y^2 + z^2) \right|_{(1,2,3)}") == Matrix([2, 4, 6])),
 ("E bound, \\mathbb{E} operator",lambda: val("X \\sim N(0, 1)\n\\mathbb{E}[X]", {"E": 200}) == 0),
 ("P bound, P(X > 1)",            lambda: close(val("X \\sim N(0, 1)\nP(X > 1)", {"P": 25}), 0.158655253931457, 1e-9)),
 ("logistic ODE",                 lambda: val(r"\frac{dP}{dt} = rP(1 - \frac{P}{K})")[0].has(t)),
 ("units in where",               lambda: abs(float(val(r"\omega_0 = \frac{1}{\sqrt{LC}} \text{ where } L = 1\,\text{mH}, C = 1\,\mu\text{F}") / __import__("sympy.physics.units", fromlist=["hertz"]).hertz) - 31622.7766) < 0.01),
 ("m + s refused",                lambda: raises(r"5\,\text{m} + 2\,\text{s}")),
 ("km in m",                      lambda: "5000" in str(val(r"5\,\text{km}\text{ in }\text{m}"))),
 ("eig multiplicity",             lambda: list(val(r"\operatorname{eig}(A)", {"A": Matrix([[2, 1], [0, 2]])})) == [2, 2]),
 ("thin space ends sin arg",      lambda: val(r"\sin\theta \, t") == sin(Symbol("theta")) * t),
 ("x_{i+1} in a sum",             lambda: val(r"\sum_{i=1}^{3} (x_{i+1} - x_i)", {"x": [1, 4, 9, 16]}) == 15),
 ("0-based sum",                  lambda: val(r"\sum_{k=0}^{2} x_k", {"x": [1, 2, 3]}) == 6),
 ("V = IR solves for I",          lambda: val(r"V = I R", {"V": 12, "R": 3}) == [4]),
 ("V(r) = 0 solves for r",        lambda: val("V(r) = \\frac{kq}{r} - 1\nV(r) = 0") == [Symbol("k") * Symbol("q")]),
 ("dot x(3)",                     lambda: val("x(t) = t^3\n\\dot x(3)") == 27),
 ("approx sci notation",          lambda: val(r"6.022 \times 10^{23} \approx 6.02 \times 10^{23}") is sp.true),
 ("z bar z declared complex",     lambda: val("z \\in \\mathbb{C}\nz \\bar z") == sp.Abs(Symbol("z", complex=True))**2),
]

def _r(src, **o):
    try: run(src, **o); return False
    except openlab.OpenlabError: return True



@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_round2_regression(name, check):
    assert check(), name
