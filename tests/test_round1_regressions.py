import pytest
"""Regression tests: every silent misread, hang and traceback that round-1 QA found
(student, engineer and adversarial personas). Each case is the minimal repro and the
mathematically right answer."""
import time
import sympy as sp
import openlab.engine as openlab
from sympy import Symbol, Rational, exp, sin, cos, Matrix, symbols
x, y, t, k, n = symbols("x y t k n")

def last(r): return r[-1] if isinstance(r, list) else r
def val(src, ns=None, **o):
    return last(openlab.run(src, {} if ns is None else ns, **o)).value
def interp(src, ns=None, **o):
    return last(openlab.run(src, {} if ns is None else ns, **o)).interpretation
def num(v): return complex(sp.N(v))

CASES = [
 # ODEs in prime notation, y on the RHS
 ("y' = ky",                 lambda: sp.simplify(val(r"y' = ky")[0].diff(x) - k*val(r"y' = ky")[0]) == 0),
 ("y' = -2y",                lambda: val(r"y' = -2y")[0].has(exp(-2*x))),
 ("y'' = -y",                lambda: val(r"y'' = -y")[0].has(sin(x)) and val(r"y'' = -y")[0].has(cos(x))),
 ("y''+2y'+y = e^{-t} in t", lambda: val(r"y'' + 2y' + y = e^{-t}")[0].has(t) and not val(r"y'' + 2y' + y = e^{-t}")[0].has(x)),
 ("ODE with IC",             lambda: sp.simplify(val(r"y' = -2y, \quad y(0) = 3")[0] - 3*exp(-2*x)) == 0),
 ("ODE with \\text{with} IC", lambda: sp.simplify(val(r"y' = -2y \text{ with } y(0) = 3")[0] - 3*exp(-2*x)) == 0),
 ("2nd-order IVP",           lambda: sp.simplify(val(r"y'' = -y, \; y(0) = 1, \; y'(0) = 0")[0] - cos(x)) == 0),
 ("L di/dt + R i = V",       lambda: val(r"L \frac{di}{dt} + R i = V")[0].has(t)),
 ("dT/dt = -k(T - T_a)",     lambda: val(r"\frac{dT}{dt} = -k(T - T_a)")[0].has(exp(-k*t))),
 ("PDE refused honestly",    lambda: last(openlab.run(r"\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}", {})).kind.name == "REPR"),
 ("x bound leaks into ODE",  lambda: sp.simplify(val(r"y' = x", {"x": 3})[0].diff(x) - x) == 0),
 # name(expr): product vs call
 ("P(1+r)^t with P bound",   lambda: abs(num(val(r"A = P(1 + r)^t", {"P": 1000, "r": 0.05, "t": 10})) - 1628.894626777442) < 1e-6),
 ("n(n+1) with n bound",     lambda: val(r"n(n+1)", {"n": 5}) == 30),
 ("x(x+1) = 0",              lambda: sorted(val(r"x(x+1) = 0")) == [-1, 0]),
 ("sum k = n(n+1)/2",        lambda: val(r"\sum_{k=1}^{n} k = \frac{n(n+1)}{2}") is sp.true),
 ("k(T - T_a) product",      lambda: "multiplication" in interp(r"k(T - T_a)")),
 # = rule
 ("f(x)=0 after define",     lambda: sorted(val("f(x) = x^2 - 4\nf(x) = 0")) == [-2, 2]),
 ("h(t)=0 then h(2)",        lambda: val("h(t) = -16t^2 + 64t\nh(t) = 0\nh(2)") == 64),
 ("f(x) = g(x)",             lambda: val("f(x) = x^2 + 1\ng(x) = 2x\nf(x) = g(x)") == [1]),
 ("y = 2x+1 then y = 5",     lambda: val("y = 2x + 1\ny = 5") == [2]),
 ("x = 4 after 2x = 6 check",lambda: val("2x + 4 = 10\n2x = 6\nx = 4") is sp.false),
 ("x^2=4 => x=±2",           lambda: val(r"x^2 = 4 \implies x = \pm 2") is sp.true),
 ("x = ±2 not bound",        lambda: "x" not in (lambda ns: (openlab.run(r"x = \pm 2", ns), ns)[1])({})),
 ("x := 3 forces bind",      lambda: (lambda ns: (openlab.run("y = 2x+1\ny := 5", ns), ns["y"])[1])({}) == 5),
 ("Gamma(x)=2 not a define", lambda: last(openlab.run(r"\Gamma(x) = 2", {})).kind.name != "DEFINE"),
 # approx
 ("x ≈ 3.0001 does not bind",lambda: (lambda ns: (openlab.run(r"x \approx 3.0001", ns), ns["x"])[1])({"x": 3}) == 3),
 ("pi ≈ 3 true (0 dp)",      lambda: val(r"\pi \approx 3") is sp.true),
 ("22/7 ≈ pi",               lambda: val(r"\frac{22}{7} \approx \pi") is sp.true),
 ("sqrt2 ≈ 1.42 false",      lambda: val(r"\sqrt{2} \approx 1.42") is sp.false),
 ("0.1+0.2 = 0.3",           lambda: val(r"0.1 + 0.2 = 0.3") is sp.true),
 # where / and ordering
 ("where ... and order",     lambda: num(val(r"E = m c^2 \text{ where } m = 1 \text{ and } c = 3e8", {"m": 2.5})) == 9e16),
 ("where with commas",       lambda: abs(num(val(r"\omega_0 = \frac{1}{\sqrt{LC}} \text{ where } L = 10^{-3}, C = 10^{-6}")) - 31622.7766) < 1e-3),
 (r"\; is not a separator",  lambda: val(r"\int_0^1 x^2 \; dx") == Rational(1, 3)),
 ("\\therefore chain",       lambda: val(r"x^2 = 4 \therefore x = \pm 2") is sp.true),
 # data vectors
 ("max(list)",               lambda: val(r"\max(x)", {"x": [1.9, 4.2, 3.3]}) == 4.2),
 ("x_1 from list",           lambda: val(r"x_1", {"x": [10, 20, 30]}) == 10),
 ("A_{12} from matrix",      lambda: val(r"A_{12}", {"A": Matrix([[1, 2], [3, 4]])}) == 2),
 ("max_i x_i",               lambda: val(r"\max_i x_i", {"x": [1, 7, 3]}) == 7),
 ("Var(data)",               lambda: val(r"\operatorname{Var}(x)", {"x": [2, 4, 4, 4, 5, 5, 7, 9]}) == Rational(32, 7)),
 ("a_n sequence then a_5",   lambda: val("a_n = 2n + 1\na_5") == 11),
 # vectors / matrices
 ("tuple vector dot",        lambda: val("u = (1, 2, 3)\nu \\cdot u") == 14),
 ("np vector ^2 elementwise",lambda: list(val(r"v^2", {"v": __import__("numpy").array([1, 2, 3])})) == [1, 4, 9]),
 ("matrix + 1 broadcast",    lambda: val(r"A + 1", {"A": Matrix([[1, 2], [3, 4]])}) == Matrix([[2, 3], [4, 5]])),
 ("(AB)^T",                  lambda: val(r"(A B)^T", {"A": Matrix([[1, 2], [3, 4]]), "B": Matrix([[0, 1], [1, 0]])}) == (Matrix([[1, 2], [3, 4]]) * Matrix([[0, 1], [1, 0]])).T),
 ("A^0 = I",                 lambda: val(r"A^0", {"A": Matrix([[1, 2], [3, 4]])}) == sp.eye(2)),
 ("literal M x = b",         lambda: val(r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} x = \begin{pmatrix} 1 \\ 2 \end{pmatrix}")[0] == Matrix([0, Rational(1, 2)])),
 ("v^T v scalar",            lambda: val(r"v^T v", {"v": Matrix([1, 2, 3])}) == 14),
 # identities
 ("(x+y)^2 identity",        lambda: val(r"(x+y)^2 = x^2 + 2xy + y^2") is sp.true),
 ("sqrt(x^2) = |x|",         lambda: val(r"\sqrt{x^2} = |x|") is sp.true),
 ("int x^2 dx = x^3/3 + C",  lambda: val(r"\int x^2 dx = \frac{x^3}{3} + C") is sp.true),
 # misc silent
 ("1,000 + 1",               lambda: val(r"1,000 + 1") == 1001),
 ("x_10 refused",            lambda: _raises(r"x_10 + 1")),
 ("units 5 km + 300 m",      lambda: "5.3" in str(sp.N(val(r"5 \text{ km} + 300 \text{ m}"))) or "53/10" in str(val(r"5 \text{ km} + 300 \text{ m}"))),
 ("2 m^2 + 3 m^2 units",     lambda: "meter**2" in str(val(r"2 \text{ m}^2 + 3 \text{ m}^2"))),
 ("x if x>0 otherwise 0",    lambda: val(r"x \text{ if } x > 0 \text{ otherwise } 0", {"x": -1}) == 0),
 ("d/dx of python lambda",   lambda: val(r"\frac{d}{dx} f", {"f": lambda z: z**3}) == 3*x**2),
 ("e everywhere",            lambda: val(r"\lim_{n\to\infty}(1+\frac{1}{n})^n = e") is sp.true),
 ("e^{2x}=5 real first",     lambda: val(r"e^{2x} = 5") == [sp.log(5)/2]),
 ("x^2+1=0 complex",         lambda: set(val(r"x^2 + 1 = 0")) == {sp.I, -sp.I}),
 ("eval bar x bound",        lambda: val(r"\left. x^3 \right|_0^2", {"x": 3}) == 8),
 ("f^{(n)} Taylor",          lambda: sp.expand(val("f(x) = e^x\n" + r"\sum_{n=0}^{3} \frac{f^{(n)}(0)}{n!} x^n") - (1 + x + x**2/2 + x**3/6)) == 0),
 ("f^2(x)",                  lambda: val("f(x) = x + 1\nf^2(2)") == 9),
 ("(f+g)(x)",                lambda: val("f(x) = x^2\ng(x) = 2x\n(f+g)(3)") == 15),
 ("\\det(A^{-1})",           lambda: val(r"\det(A^{-1})", {"A": Matrix([[1, 2], [3, 4]])}) == Rational(-1, 2)),
 ("P(A)=0.3 events",         lambda: abs(num(val("P(A) = 0.3\nP(B) = 0.5\n" + r"P(A \cap B) = 0.1" + "\n" + r"P(A \cup B)")) - 0.7) < 1e-12),
 ("P(A|B)",                  lambda: abs(num(val("P(A) = 0.3\nP(B) = 0.5\n" + r"P(A \cap B) = 0.1" + "\n" + r"P(A \mid B)")) - 0.2) < 1e-12),
 ("P(|X|<1.96)",             lambda: abs(num(val("X \\sim N(0, 1)\n" + r"P(|X| < 1.96)")) - 0.95) < 1e-3),
 ("Bin bare name",           lambda: last(openlab.run(r"X \sim Bin(10, 0.5)", {})).kind.name == "DEFINE"),
 ("Laplace",                 lambda: sp.simplify(val(r"\mathcal{L}\{e^{-2t}\}") - 1/(Symbol("s") + 2)) == 0),
 ("system over two lines",   lambda: val("x + y = 5\nx - y = 1") == [{x: 3, y: 2}]),
 ("100000! no crash",        lambda: isinstance(repr(openlab.run(r"100000!", {})), str)),
 ("long sum no traceback",   lambda: val(" + ".join(["x"] * 500)) == 500 * x),
 ("hang -> numeric/limit",   lambda: _no_hang(r"\int_0^1 \frac{1}{x^{20} + 1} dx")),
]

def _raises(src):
    try: openlab.run(src, {}); return False
    except openlab.OpenlabError: return True

def _no_hang(src):
    t0 = time.time()
    try: val(src, timeout=20)
    except openlab.OpenlabError: pass
    return time.time() - t0 < 25



@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_round1_regression(name, check):
    assert check(), name
