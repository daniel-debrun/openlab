"""The general-math lexicon: sets, logic, relations, decorations, stats, ODEs, vectors,
evaluation bars, cases, statement structure. Each block is a construct a student or
engineer types; the assertion is the mathematically right answer."""

import pytest
import sympy
from sympy import (FiniteSet, ImmutableMatrix as M, Interval, Matrix, Rational, S, Symbol,
                   symbols, sqrt, pi, sin, cos, exp)

import openlab.engine as openlab
from openlab.engine import Kind, run

x, y, z, t, n, k, p = symbols("x y z t n k p")


def val(src, ns=None, **o):
    r = run(src, {} if ns is None else ns, **o)
    return r.value if not isinstance(r, list) else [q.value for q in r]


# ---- numbers ------------------------------------------------------------------------
def test_numbers():
    assert val(r"50\%") == Rational(1, 2)
    assert val(r"1{,}000 + 1") == 1001
    assert val(r"1\,000\,000 + 1") == 1000001
    assert val(r"3e8") == 300000000
    assert val(r"1.5 \times 10^{-3}") == sympy.Float("0.0015")
    assert val(r"5!!") == 15
    assert val(r"7!!") == 105
    assert val(r"\sin(30^\circ)") == Rational(1, 2)
    assert val(r"45^{\circ}") == pi / 4
    assert val(r"\lg 1000") == 3
    assert val(r"\ln e") == 1
    assert val(r"\log_e x") == sympy.log(x)


# ---- functions: primes, inverse, nth derivative, composition -----------------------------
def test_function_modifiers():
    ns = {}
    run(r"f(x) = x^3", ns)
    assert val(r"f'(x)", ns) == 3 * x**2
    assert val(r"f''(2)", ns) == 12
    assert val(r"f^{(3)}(x)", ns) == 6
    assert val(r"f^{-1}(8)", ns) == 2
    run(r"g(x) = x + 1", ns)
    assert val(r"(f \circ g)(x)", ns) == (x + 1) ** 3
    assert val(r"\frac{df}{dx}", ns) == 3 * x**2         # a defined f used as an expression
    r = run(r"y'", {})
    assert isinstance(r.value, sympy.Derivative)


# ---- ODEs ----------------------------------------------------------------------------
def test_odes():
    r = run(r"\frac{dy}{dx} = 2x", {})
    assert r.kind is Kind.SOLVE and "ODE" in r.interpretation
    assert sympy.diff(r.value[0], x) == 2 * x
    r = run(r"y'' + y = 0", {})
    sol = r.value[0]
    assert sol.has(sin(x)) and sol.has(cos(x))
    r = run(r"\dot{x} = -k x", {})
    assert r.value[0].has(exp(-k * t))
    assert isinstance(val(r"\frac{d^2 y}{dx^2}"), sympy.Derivative)   # not 0


# ---- sets, intervals, logic ----------------------------------------------------------
def test_sets_and_logic():
    assert val(r"\{1, 2, 3\}") == FiniteSet(1, 2, 3)
    assert val(r"x \in \{1,2,3\}", {"x": 2}) is sympy.true
    assert val(r"4 \notin \{1,2,3\}") is sympy.true
    assert val(r"\{1,2\} \cup \{2,3\}") == FiniteSet(1, 2, 3)
    assert val(r"\{1,2\} \cap \{2,3\}") == FiniteSet(2)
    assert val(r"\{1,2,3\} \setminus \{2\}") == FiniteSet(1, 3)
    assert val(r"[0, 1]") == Interval(0, 1)
    assert val(r"(0, 1]") == Interval.Lopen(0, 1)
    assert val(r"[0, 1) \cup [1, 2]") == Interval(0, 2)
    assert val(r"\mathbb{R}") == S.Reals
    assert val(r"2 \in \mathbb{N}") is sympy.true
    assert val(r"\{1, 2\} \subseteq \{1, 2, 3\}") is sympy.true
    assert val(r"\emptyset") == S.EmptySet
    cs = val(r"\{ x \in \mathbb{R} \mid x^2 < 2 \}")
    assert isinstance(cs, sympy.ConditionSet)
    assert val(r"p \land q", {"p": True, "q": False}) is sympy.false
    assert val(r"p \lor q", {"p": True, "q": False}) is sympy.true
    assert val(r"\neg p", {"p": True}) is sympy.false
    assert val(r"p \implies q", {"p": False, "q": False}) is sympy.true
    assert val(r"p \iff q", {"p": True, "q": True}) is sympy.true
    r = run(r"\forall x \in \mathbb{R}, x^2 \ge 0", {})          # notation: a note until a technique is named
    assert r.kind is Kind.NOTE and "not executed" in r.interpretation
    r = run(r"\forall x \in \mathbb{R}, x^2 \ge 0 \text{ by algebra}", {})
    assert r.kind is Kind.PROOF and r.value is sympy.true


# ---- relations -----------------------------------------------------------------------
def test_relations():
    assert val(r"1 < 2 < 3") is sympy.true
    assert val(r"1 < 3 < 2") is sympy.false
    assert val(r"a \le b \le c", {"a": 1, "b": 2, "c": 3}) is sympy.true
    assert val(r"x := 5", {}) == 5
    assert val(r"f \triangleq x^2", {}) == x**2
    assert run(r"y \propto x", {}).kind is Kind.REPR
    assert val(r"17 \equiv 2 \pmod 5") is sympy.true
    assert val(r"17 \equiv 3 \pmod 5") is sympy.false
    sol = val(r"2x \equiv 1 \pmod 5")
    assert 3 in sol and 8 in sol and 4 not in sol
    assert val(r"3 \mid 12") is sympy.true
    assert val(r"5 \mid 12") is sympy.false
    assert val(r"x^2 + 2x + 1 = (x+1)^2") is sympy.true       # identity, not "no solution"
    assert val(r"\sin^2 x + \cos^2 x = 1") is sympy.true
    r = run(r"x^2 + 1 = 0", {})
    assert set(r.value) == {sympy.I, -sympy.I}


# ---- number theory & combinatorics -----------------------------------------------------
def test_number_theory():
    assert val(r"17 \bmod 5") == 2
    assert val(r"17 \mod 5") == 2
    assert val(r"\gcd(12, 18)") == 6
    assert val(r"\operatorname{lcm}(4, 6)") == 12
    assert val(r"{}^5C_2") == 10
    assert val(r"{5 \choose 2}") == 10
    assert val(r"\dbinom{5}{2}") == 10
    assert val(r"\max(1, 7, 3)") == 7
    assert val(r"\min(1, 7, 3)") == 1
    assert val(r"\max\{1, 7, 3\}") == 7
    assert val(r"\sup [0, 1)") == 1


# ---- optimisation & vector calculus ---------------------------------------------------------
def test_optim_and_nabla():
    assert val(r"\max_{x} (4 - x^2)") == 4
    assert val(r"\min_{x \in [0, 3]} (x - 1)^2") == 0
    assert val(r"\arg\max_x (4 - (x-2)^2)") == 2
    assert val(r"\nabla (x^2 + y^2)") == M([2 * x, 2 * y])
    assert val(r"\nabla \cdot \vec{F}", {"F": Matrix([x, y, z])}) == 3
    assert val(r"\nabla \times \vec{F}", {"F": Matrix([-y, x, 0])}) == M([0, 0, 2])
    assert val(r"\nabla^2 (x^2 + y^2)") == 4
    assert val(r"\partial_x \partial_y (x^2 y^3)") == 6 * x * y**2


# ---- evaluation bars & multiple integrals -------------------------------------------------
def test_evaluation_bars_and_integrals():
    assert val(r"\left. x^2 \right|_0^1") == 1
    assert val(r"x^2 + x \Big|_{x=2}") == 6
    assert val(r"\left[ x^3 \right]_1^2") == 7
    assert val(r"\iint x y \, dx \, dy") == x**2 * y**2 / 4
    assert val(r"\int_0^1 \int_0^1 x y \, dx \, dy") == Rational(1, 4)


# ---- decorations & names -------------------------------------------------------------
def test_decorations():
    ns = {}
    assert val(r"\bar{x} + \hat{\theta}", ns) == Symbol("xbar") + Symbol("thetahat")
    run(r"\bar{x} = 3", ns)
    assert ns["xbar"] == 3 and val(r"\bar{x} + 1", ns) == 4
    assert val(r"\bar{z}", {"z": 3 + 4 * sympy.I}) == 3 - 4 * sympy.I
    assert val(r"\overline{z}", {"z": 3 + 4 * sympy.I}) == 3 - 4 * sympy.I
    assert val(r"\bar{v}", {"v": [1, 2, 3, 6]}) == 3                    # mean of data
    assert val(r"\vec{v} \cdot \vec{v}", {"v": Matrix([1, 2, 2])}) == 9
    assert val(r"\mathbf{u} \times \mathbf{v}", {"u": Matrix([1, 0, 0]), "v": Matrix([0, 1, 0])}) == M([0, 0, 1])
    assert val(r"\Delta x = 5", {}) == 5
    assert val(r"x^{2}_{i}") == Symbol("x_i") ** 2
    assert val(r"\hbar \omega") == Symbol("hbar") * Symbol("omega")
    assert val(r"\mathbf{0} + 1") == 1
    assert val(r"x_{\text{max}} - 1", {"x_max": 4}) == 3


# ---- linear algebra -------------------------------------------------------------------
def test_linear_algebra():
    A = Matrix([[1, 2], [3, 4]])
    ns = {"A": A, "u": Matrix([1, 2, 2]), "v": Matrix([1, 0, 0])}
    assert val(r"\det A", ns) == -2                                  # mutable Matrix is visible
    assert val(r"|A|", ns) == -2
    assert val(r"\|u\|", ns) == 3
    assert val(r"|u|", ns) == 3
    assert val(r"\|A\|_F", ns) == sqrt(30)
    assert val(r"\langle u, v \rangle", ns) == 1
    assert val(r"A^\dagger", ns) == A.T
    assert val(r"A^*", ns) == A.T
    assert val(r"z^*", {"z": 1 + sympy.I}) == 1 - sympy.I
    assert val(r"\operatorname{tr}(A)", ns) == 5
    assert val(r"\operatorname{rank}(A)", ns) == 2
    assert val(r"\det(A - \lambda I)", ns).expand() == Symbol("lambda") ** 2 - 5 * Symbol("lambda") - 2
    assert val(r"I_2") == sympy.eye(2)
    assert sorted(val(r"\operatorname{eig}(A)", ns), key=str) == sorted(A.eigenvals().keys(), key=str)
    assert list(val(r"\operatorname{eig}(A)", {"A": Matrix([[2, 1], [0, 2]])})) == [2, 2]   # with multiplicity
    assert val(r"A \otimes I_2", ns).shape == (4, 4)
    assert val(r"\begin{array}{cc} 1 & 2 \\ 3 & 4 \end{array}") == M([[1, 2], [3, 4]])
    r = run(r"A x = b", {"A": A, "b": Matrix([5, 11])})
    assert r.value[0] == Matrix([1, 2])
    r = run(r"\begin{cases} x + y = 3 \\ x - y = 1 \end{cases}", {})
    assert r.value == [{x: 2, y: 1}]


# ---- piecewise ------------------------------------------------------------------------------
def test_cases_piecewise():
    ns = {}
    run(r"f(x) = \begin{cases} x^2 & \text{if } x \ge 0 \\ -x & \text{otherwise} \end{cases}", ns)
    assert val(r"f(3)", ns) == 9 and val(r"f(-3)", ns) == 3


# ---- statistics -------------------------------------------------------------------------
def test_stats():
    ns = {}
    r = run(r"X \sim \mathcal{N}(\mu, \sigma^2)", ns)
    assert "random variable" in r.interpretation
    assert val(r"\mathbb{E}[X]", ns) == Symbol("mu")
    assert val(r"\operatorname{Var}(X)", ns) == Symbol("sigma") ** 2
    run(r"Y \sim \operatorname{Bin}(10, 0.5)", ns)
    assert abs(float(val(r"P(Y = 5)", ns)) - 0.24609375) < 1e-9
    assert float(val(r"\mathbb{E}[Y]", ns)) == 5.0
    run(r"Z \sim N(0, 1)", ns)
    assert val(r"P(Z > 0)", ns) == Rational(1, 2)
    assert val(r"P(Z > 1 \mid Z > 0)", ns) is not None
    r = run(r"P(A \mid B)", {})
    assert "not known" in r.interpretation
    assert val(r"\frac{1}{n} \sum_{i=1}^{n} x_i", {"x": [2, 4, 6], "n": 3}) == 4


# ---- special functions ----------------------------------------------------------------
def test_special_functions():
    assert val(r"\Gamma(5)") == 24
    assert val(r"\zeta(2)") == pi**2 / 6
    assert val(r"\operatorname{erf}(0)") == 0
    assert val(r"J_0(0)") == 1
    assert val(r"\operatorname{sgn}(-3)") == -1
    assert val(r"\Re(z) + \Im(z)", {"z": 3 + 4 * sympy.I}) == 7
    assert val(r"\arg(i)") == pi / 2
    assert val(r"|3 + 4i|") == 5
    assert val(r"e^{i\pi} + 1") == 0


# ---- statement structure ---------------------------------------------------------------
def test_statement_structure():
    ns = {}
    out = val(r"x, y = 1, 2", ns)
    assert ns["x"] == 1 and ns["y"] == 2
    out = val(r"a = 2 \text{ and } b = 3", ns)
    assert out == [2, 3]
    out = val(r"c = a + b \quad \text{where } a = 10", ns)
    assert ns["c"] == 13
    out = val(r"x^2 = 4 \implies x = \pm 2", {})
    assert out[0] == [-2, 2] and out[1] is sympy.true        # the step is checked
    assert val(r"\text{let } q = 2", {}) == 2
    assert val(r"\therefore q = 2", {}) == 2
    pw = val(r"f(x) = x^2 \quad \text{for } x > 0", {}).expr
    assert isinstance(pw, sympy.Piecewise) and pw.subs(x, 2) == 4
    out = val("1 + 1\n+ 2", {})
    assert out == 4
    assert val(r"\begin{aligned} u &= 1 \\ w &= u + 1 \end{aligned}", {}) == [1, 2]
    assert val(r"\( 2 + 2 \)") == 4
    assert val(r"\Bigl( 1 + 2 \Bigr) \cdot 2") == 6
    assert val(r"x~+~1", {"x": 1}) == 2


def test_no_tracebacks_for_shape_errors():
    with pytest.raises(openlab.OpenlabError):
        run(r"A v", {"A": Matrix([[1, 2], [3, 4]]), "v": Matrix([1, 2, 3])})
    # a scalar formula on a data vector is applied entry by entry, not an error
    assert list(run(r"\frac{1}{2} m v^2", {"v": Matrix([1, 2, 3]), "m": 2}).value) == [1, 4, 9]
