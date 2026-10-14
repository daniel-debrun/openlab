"""Round-3 regressions (student + adversarial + engineer QA): silent misreads (student report; engineer/adversarial appended later): minimal repro + right answer."""
import time
import sympy as sp
import openlab.engine as openlab
from sympy import Symbol, exp, sin, cos, cot, pi, sqrt, Matrix, symbols, oo
x, y, t, k, n, z = symbols("x y t k n z")

def last(r): return r[-1] if isinstance(r, list) else r
def run(src, ns=None, **o): return openlab.run(src, {} if ns is None else ns, **o)
def val(src, ns=None, **o): return last(run(src, ns, **o)).value
def res(src, ns=None, **o): return last(run(src, ns, **o))
def num(v): return complex(sp.N(v))
def close(a, b, tol=1e-9): return abs(num(a) - num(b)) < tol
def same(a, b): return sp.simplify(a - b) == 0
def setof(v): return set(sp.nsimplify(q) if isinstance(q, sp.Float) else q for q in v)
def raises(src, ns=None):
    try: run(src, ns); return False
    except openlab.OpenlabError: return True
def err(src, ns=None):
    try: run(src, ns); return None
    except openlab.OpenlabError as e: return e

def cells(*srcs, magic_ids=None):
    """Run cells in one namespace; each gets its own cell id (as the magic does)."""
    ns = {}
    out = []
    for i, s in enumerate(srcs):
        cid = magic_ids[i] if magic_ids else None
        out.append(openlab.run_all(s, ns, **({"cell_id": cid} if cid else {})))
    return out, ns

CASES = [
 # --- domains --------------------------------------------------------------------------------------
 ("sin x = 1/2, 0<=x<2pi",        lambda: val(r"\sin x = \frac{1}{2}, \quad 0 \le x < 2\pi") == [pi/6, 5*pi/6]),
 ("cos 2x = 1/2, x in [0,pi]",    lambda: val(r"\cos 2x = \frac{1}{2}, \; x \in [0, \pi]") == [pi/6, 5*pi/6]),
 ("x^2 = 4, x > 0",               lambda: val(r"x^2 = 4, \; x > 0") == [2]),
 ("binom(n,2) = 45, n > 0",       lambda: val(r"\binom{n}{2} = 45, \; n > 0") == [10]),
 ("sin x = 1/2, x in [0,2pi)",    lambda: val(r"\sin x = \frac12, \; x\in[0,2\pi)") == [pi/6, 5*pi/6]),
 ("for 0<=x<=2pi: two roots",     lambda: val(r"\sin x = -\frac{1}{2} \text{ for } 0 \le x \le 2\pi") == [7*pi/6, 11*pi/6]),
 ("sin 2x = 0 on [0,2pi)",        lambda: val(r"\sin 2x = 0 \text{ for } 0 \le x < 2\pi") == [0, pi/2, pi, 3*pi/2]),
 ("sin x = 1/2 on [0,4pi]",       lambda: len(val(r"\sin x = \frac12 \text{ for } 0 \le x \le 4\pi")) == 4),
 # --- \, dx after a function argument ---------------------------------------------------------------
 ("\\int \\sin x\\, dx",          lambda: same(val(r"\int \sin x\, dx"), -cos(x))),
 ("\\int_0^pi sin x\\, dx",       lambda: val(r"\int_0^{\pi} \sin x\, dx") == 2),
 ("\\int cos x \\; dx",           lambda: same(val(r"\int \cos x \; dx"), sin(x))),
 ("\\int x sin x\\, dx",          lambda: same(val(r"\int x\sin x\, dx"), sin(x) - x*cos(x))),
 # --- formula across cells ---------------------------------------------------------------------------
 ("y=2x+1 / y=7 across run()",    lambda: (lambda o: o[0][1][-1].value == [3])(cells("y = 2x + 1", "y = 7"))),
 ("y=2x+1 / y=7 across cells",    lambda: (lambda o: o[0][1][-1].value == [3])(cells("y = 2x + 1", "y = 7", magic_ids=["c1", "c2"]))),
 ("edited single-line cell",      lambda: (lambda o: o[1]["y"] == 3*x + 1)(cells("y = 2x + 1", "y = 3x + 1", magic_ids=["c1", "c1"]))),
 # --- series -----------------------------------------------------------------------------------------
 ("sum 1/sqrt n diverges",        lambda: val(r"\sum_{n=1}^{\infty} \frac{1}{\sqrt n}") == oo),
 ("sum 1/(n ln n) diverges",      lambda: val(r"\sum_{n=2}^{\infty} \frac{1}{n \ln n}") == oo),
 ("sum ln((n+1)/n) diverges",     lambda: val(r"\sum_{n=1}^{\infty} \ln\left(\frac{n+1}{n}\right)") == oo),
 ("sum (-1)^n/sqrt n numeric",    lambda: close(res(r"\sum_{n=1}^{\infty} \frac{(-1)^n}{\sqrt n}").numeric, -0.604898643421630, 1e-9)),
 ("sum sin n/n^2 numeric",        lambda: close(res(r"\sum_{n=1}^{\infty} \frac{\sin n}{n^2}").numeric, 1.01395913236077, 1e-9)),
 ("-n: 1/sqrt n not a decimal",   lambda: val(r"\sum_{n=1}^{\infty} \frac{1}{\sqrt n}", numeric=True) == oo),
 ("sum of a defined sequence",    lambda: val("a_n = \\frac{1}{2^n}\n\\sum_{n=1}^{\\infty} a_n") == 1),
 # --- derivatives ------------------------------------------------------------------------------------
 ("parametric dy/dx",             lambda: same(val("x(t) = \\cos t\ny(t) = \\sin t\n\\frac{dy}{dx}"), -cot(t))),
 ("critical points dfdx=0",       lambda: (lambda r: r[1].value == [-1, 1] and r[2].value == [0])(run("f(x,y) = x^3 - 3x + y^2\n\\frac{\\partial f}{\\partial x} = 0\n\\frac{\\partial f}{\\partial y} = 0"))),
 ("grad f = 0",                   lambda: len(val("f(x,y) = x^3 - 3x + y^2\n\\nabla f = 0")) == 2),
 ("Lagrange system",              lambda: len(val("f(x,y) = x + y\ng(x,y) = x^2 + y^2 - 2\n\\nabla f = \\lambda \\nabla g\ng(x,y) = 0")) == 2),
 ("implicit d/dx(x^2+y^2)=d/dx(25)", lambda: same(val(r"\frac{d}{dx}(x^2 + y^2) = \frac{d}{dx}(25)")[0], -x/y)),
 ("implicit follow-up dy/dx",     lambda: same(val("x^2 + y^2 = 25\n\\frac{dy}{dx}")[0], -x/y)),
 # --- numeric roots ----------------------------------------------------------------------------------
 ("tan x = x: no cloud at 0",     lambda: (lambda v: sum(1 for q in v if abs(float(q)) < 0.01) == 1 and any(abs(float(q) - 4.493409457909) < 1e-6 for q in v))(val(r"\tan x = x"))),
 ("x = cos x",                    lambda: close(val(r"x = \cos x")[0], 0.739085133215161)),
 # --- prose / period ---------------------------------------------------------------------------------
 ("prose word problem refused",   lambda: raises(r"\text{The sum of two numbers is } 15 \text{ and their difference is } 3.")),
 ("mid-line period",              lambda: (lambda r: r[0].kind.name == "DEFINE")(openlab.run_all(r"\text{Let } f(x) = 3x^2-12x+5. \text{ Find the vertex.}", {}))),
 # --- logic ------------------------------------------------------------------------------------------
 ("De Morgan tautology",          lambda: val(r"\neg(p \lor q) \iff \neg p \land \neg q") is sp.true),
 ("p or not p",                   lambda: val(r"p \lor \neg p") is sp.true),
 ("p and not p",                  lambda: val(r"p \land \neg p") is sp.false),
 # --- events -----------------------------------------------------------------------------------------
 ("edited P(A cap B) restates",   lambda: (lambda o: close(openlab.run(r"P(A \cup B)", o[1]).value, 0.5))(cells("P(A) = 0.3\nP(B) = 0.4", r"P(A \cap B) = 0.1", r"P(A \cap B) = 0.2", magic_ids=["c1", "c2", "c2"]))),
 ("named partition total P",      lambda: close(val("P(B_1) = 0.3\nP(B_2) = 0.7\nP(A | B_1) = 0.5\nP(A | B_2) = 0.2\nP(A)"), 0.29)),
 ("Bayes over named partition",   lambda: close(val("P(B_1) = 0.3\nP(B_2) = 0.7\nP(A | B_1) = 0.5\nP(A | B_2) = 0.2\nP(B_1 | A)"), 0.15/0.29)),
 # --- sets -------------------------------------------------------------------------------------------
 ("letter sets cap",              lambda: val("S = \\{a,b,c\\}\nT = \\{b,c,d\\}\nS \\cap T = \\{b,c\\}") is sp.true),
 ("letter sets setminus",         lambda: val("S = \\{a,b,c\\}\nT = \\{b,c,d\\}\nS \\setminus T = \\{a\\}") is sp.true),
 ("|A cup B|",                    lambda: val("A = \\{1,2,3\\}\nB = \\{3,4\\}\n|A \\cup B|") == 4),
 # --- recurrences ------------------------------------------------------------------------------------
 ("comma-form recurrence ICs",    lambda: val("c_n = c_{n-1} + c_{n-2}, c_0 = 0, c_1 = 1\nc_{10}") == 55),
 # --- limits -----------------------------------------------------------------------------------------
 ("lim sin(1/x) DNE",             lambda: str(val(r"\lim_{x \to 0} \sin\frac{1}{x}")) == "DNE"),
 # --- distributions ----------------------------------------------------------------------------------
 ("chi^2(3)",                     lambda: close(val("X \\sim \\chi^2(3)\n\\mathbb{E}[X]"), 3)),
 ("t(5) P fast",                  lambda: (lambda t0: close(val("T \\sim t(5)\nP(T > 2.015)"), 0.05, 1e-3) and time.time() - t0 < 10)(time.time())),
 ("P(mu-s < X < mu+s)",           lambda: close(val("X \\sim \\mathcal{N}(\\mu, \\sigma^2)\nP(\\mu - \\sigma < X < \\mu + \\sigma)"), 0.682689492137086, 1e-9)),
 ("Bin(n,p) E = np",              lambda: same(val("X \\sim \\operatorname{Bin}(n, p)\n\\mathbb{E}[X]"), n*Symbol("p"))),
 ("no silly N(0,1) note",         lambda: "1^2" not in res("X \\sim N(0, 1)").interpretation),
 # --- display ----------------------------------------------------------------------------------------
 ("polar complex 2e^{i pi/3}",    lambda: val(r"2e^{i\pi/3}") == 1 + sqrt(3)*sp.I),
 # --- where-conditions are local ---------------------------------------------------------------------
 ("where x > 0 is local",         lambda: (lambda ns: (run(r"\sqrt{x^2} \text{ where } x > 0", ns), val(r"x^2 = 4", ns))[1] == [-2, 2])({})),
]

A01 = Matrix([[0, 1], [0, 0]])
CASES += [
 # === adversarial ===================================================================================
 ("W1 f(x)=0 in its own cell",    lambda: (lambda o: o[0][1][-1].value == [-2, 2] and openlab.run("f(3)", o[1]).value == 5)(cells("f(x) = x^2 - 4", "f(x) = 0"))),
 ("W1 V = I R across cells",      lambda: (lambda o: o[0][2][-1].value == [3])(cells("V = 12", "R = 4", "V = I R"))),
 ("W2 sin x \\, + cos x",         lambda: same(val(r"\sin x \, + \cos x"), sin(x) + cos(x))),
 ("W2 ln x \\, - 1",              lambda: same(val(r"\ln x \, - 1"), sp.log(x) - 1)),
 ("W2 f(x) = sin x \\, + 1",      lambda: val("f(x) = \\sin x \\, + 1\nf(0)") == 1),
 ("W3 bar + x",                   lambda: val(r"\left. x^2 \right|_{x=1} + x", {"x": 3}) == 4),
 ("W3 d/dx x^2 + x",              lambda: val(r"\frac{d}{dx} x^2 + x", {"x": 3}) in (9, 2*x + 3)),
 ("W3 bar(d/dx) + x",             lambda: val(r"\left. \frac{d}{dx} x^2 \right|_{x=2} + x", {"x": 3}) == 7),
 ("W3 bar over unbound t",        lambda: val(r"\left. t^2 x \right|_0^2", {"x": 3}) == 12),
 ("W4a frac of units",            lambda: "meter" in str(val(r"\frac{10\,\text{m}}{2\,\text{s}}")) and "second" in str(val(r"\frac{10\,\text{m}}{2\,\text{s}}"))),
 ("W4a sqrt(4 m^2)",              lambda: "meter" in str(val(r"\sqrt{4\,\text{m}^2}"))),
 ("W4b ln of a unit refused",     lambda: raises(r"\ln(5\,\text{m})")),
 ("W4b 4 in = inch",              lambda: "inch" in str(val(r"4\,\text{in}"))),
 ("W5 sin^{-1} principal",        lambda: val("f(x) = \\sin x\nf^{-1}(\\frac{1}{2})") == pi/6),
 ("W5 cos^{-1} principal",        lambda: val("f(x) = \\cos x\nf^{-1}(0)") == pi/2),
 ("W5 x^2 inverse principal",     lambda: val("f(x) = x^2\nf^{-1}(4)") == 2),
 ("W6 2^A matrix function",       lambda: sp.simplify(val(r"2^{A}", {"A": A01}) - Matrix([[1, sp.log(2)], [0, 1]])) == sp.zeros(2)),
 ("W6 cos(A)",                    lambda: val(r"\cos(A)", {"A": A01}) == sp.eye(2)),
 ("W7 sin x = x only 0",          lambda: val(r"\sin x = x") == [0]),
 ("W8 identity not merged",       lambda: (lambda r: any(isinstance(q.value, list) and q.value and isinstance(q.value[0], dict) and q.value[0].get(x) == 2 for q in r))(openlab.run_all("x + y = 3\nx - y = 1\n\\sin^2 a + \\cos^2 a = 1", {}))),
 ("W9 x^2=9 => x=3 is False",     lambda: last(openlab.run_all(r"x^2 = 9 \implies x = 3", {})).value is sp.false),
 ("W10 g(1)=1 line kept",         lambda: len(openlab.run_all("g(x) = x^2\ng'(x) = 2x\ng(1) = 1\ng(2)", {})) == 4),
 ("W10 conflicting ICs refused",  lambda: raises("y' = y\ny(0) = 1\ny(0) = 2")),
 ("W11 x in Z, x^2 = 2",          lambda: val("x \\in \\mathbb{Z}\nx^2 = 2") == []),
 ("W12 inconsistent P refused",   lambda: raises("P(A) = 0.3\nP(B) = 0.5\nP(A \\cup B) = 0.9\nP(A \\cap B)")),
 ("W13 0-based x_{k+1}",          lambda: val(r"\sum_{k=0}^{2} x_{k+1}", {"x": [10, 20, 30, 40]}) == 60),
 ("W13 x_{2k} in sum",            lambda: val(r"\sum_{k=1}^{2} x_{2k}", {"x": [10, 20, 30, 40]}) == 60),
 ("W14 y = x^2 for x = 3",        lambda: val(r"y = x^2 \text{ for } x = 3") == 9),
 ("W14 v(t) = 3t at t = 2",       lambda: val(r"v(t) = 3t \text{ at } t = 2") == 6),
 # W15 (I_2 beside no matrix) is deliberately a name: engineers write currents I_1, I_2.
 ("W16 f shown as function",      lambda: isinstance(val("f(x) = x^2\nf", {"x": 0}), sp.Lambda)),
 ("W17 failed chain no binding",  lambda: (lambda ns: (raises("a = 3 = 4", ns) or True) and "a" not in ns)({})),
 ("E2 s = 2 + 3 = 5",             lambda: (lambda ns: (run("s = 2 + 3 = 5", ns), ns["s"] == 5)[1])({})),
 ("E2 A = pi r^2 = 4pi",          lambda: (lambda ns: (run(r"A = \pi r^2 = 4\pi", ns), ns["A"] == 4*pi)[1])({"r": 2})),
 ("E3 25 degC",                   lambda: "25" in str(val(r"25\,^\circ\text{C}"))),
 ("E3 100C - 20C",                lambda: "80" in str(val(r"100\,^\circ\text{C} - 20\,^\circ\text{C}"))),
 ("E4 x^2 = 4 s.t. x > 0",        lambda: val(r"x^2 = 4 \text{ s.t. } x > 0") == [2]),
 ("E4 max s.t.",                  lambda: val(r"\max_x -x^2 + 2x \text{ s.t. } x \ge 0") == 1),
 ("E5 int_0^x x dx",              lambda: (lambda e: e is None or "circular" not in str(e))(err(r"\int_0^x x\,dx", {"x": 3}))),
 ("E5 deep chain 60",             lambda: val("a_1", {**{f"a_{i}": Symbol(f"a_{i+1}") + 1 for i in range(1, 60)}, "a_60": 0}) == 59),
 ("E6 A^T A - I",                 lambda: val(r"A^T A - I", {"A": Matrix([[1, 2, 3], [4, 5, 6]])}) == Matrix([[1, 2, 3], [4, 5, 6]]).T * Matrix([[1, 2, 3], [4, 5, 6]]) - sp.eye(3)),
 ("E7 t = 2t - 3",                lambda: val("t = 2t - 3") == [3]),
 ("E8a x_{k+1} with k bound",     lambda: val(r"x_{k+1}", {"x": [10, 20, 30], "k": 1}) == 20),
 ("E8b recurrence IC after",      lambda: val("a_{n+1} = a_n + 3, \\; a_0 = 2\na_4") == 14),
 ("E8b P(bar A)",                 lambda: close(val("P(A) = 0.3\nP(\\bar A)"), 0.7)),
 ("E8b sin x / x = 1 no sol",     lambda: val(r"\frac{\sin x}{x} = 1") == []),
 ("E8b x>2 => x>1",               lambda: val(r"x > 2 \implies x > 1") is sp.true),
 ("E10 1,500/3",                  lambda: val(r"\frac{1,500}{3}") == 500),
 ("RAW openlabin not in display",   lambda: "openlabin" not in openlab.run(r"5\,\text{km} \text{ in } \text{m}", {})._repr_html_()),
]

import numpy as np
_A = np.array([[0., 1.], [-10., -2.]])
CASES += [
 # === engineer =====================================================================================
 ("WS1 C(sI-A)^{-1}B order",      lambda: same(val(r"C (sI - A)^{-1} B", {"A": _A, "B": np.array([[0.], [10.]]), "C": np.array([[1., 0.]])}), 10/(Symbol("s")**2 + 2*Symbol("s") + 10))),
 ("WS1 C(A+A)B order",            lambda: (val(r"C (A + A) B", {"A": np.array([[1., 2.], [3., 4.]]), "B": np.array([[0., 1.], [1., 0.]]), "C": np.array([[2., 0.], [0., 1.]])}) - Matrix([[8, 4], [8, 6]])).norm() < 1e-12),
 ("WS2 X_L with L bound",         lambda: close(val(r"X_L = 2\pi f L", {"f": 50.0, "L": 0.1}), 31.4159265358979)),
 ("WS2 h_f with f bound",         lambda: close(val(r"h_f = f \frac{L}{D} \frac{v^2}{2g}", {"f": 0.02, "L": 100.0, "D": 0.1, "v": 2.0, "g": 9.81}), 4.07747196738022)),
 ("WS3 m Delta T",                lambda: close(val(r"m \Delta T", {"m": 2.0, "Delta_T": 30.0}), 60)),
 ("WS3 Delta x / Delta t",        lambda: close(val(r"\Delta x / \Delta t", {"Delta_x": 10.0, "Delta_t": 2.0}), 5)),
 ("WS4 units in frac",            lambda: "ampere" in str(val(r"\frac{12\,\text{V}}{4\,\Omega}"))),
 ("WS4 kN/mm^2 in MPa",           lambda: "200" in str(val(r"\frac{10\,\text{kN}}{50\,\text{mm}^2} \text{ in } \text{MPa}"))),
 ("WS5 kWh in J",                 lambda: "3600000" in str(val(r"1\,\text{kWh} \text{ in } \text{J}"))),
 ("WS5 100F in K",                lambda: abs(float(val(r"100\,^\circ\text{F} \text{ in } \text{K}").args[0]) - 310.927777777778) < 1e-9),
 ("WS6 inverse Laplace delay",    lambda: val(r"\mathcal{L}^{-1}\left\{\frac{e^{-2s}}{s}\right\}").has(sp.Heaviside)),
 ("WS7 dot theta(0)",             lambda: val("\\theta(t) = \\sin(3t)\n\\dot\\theta(0)") == 3),
 ("WS8 ODE system ICs",           lambda: (lambda v: same(v[0][Symbol("x")] if Symbol("x") in v[0] else list(v[0].values())[0], cos(t)))(val("\\dot x = y, \\; x(0) = 1\n\\dot y = -x, \\; y(0) = 0"))),
 ("WS9 pinv(X) y",                lambda: (lambda v: v.shape == (2, 1) and close(v[1], 2.01))(val(r"\operatorname{pinv}(X) y", {"X": np.column_stack([np.ones(5), np.arange(5.)]), "y": np.array([1., 3., 5.2, 6.9, 9.1])}))),
 ("WS9 bar in a vector sum",      lambda: val(r"(1, 1) + \left.\nabla (x^2 + 3y^2)\right|_{(1,1)}") == Matrix([3, 7])),
 ("WS10 --for I on V = I R",      lambda: val("V = I R", solve_for="I") == [Symbol("V") / Symbol("R")]),
 ("WS13 y_n from data",           lambda: val("y_n = \\frac{x_n + x_{n-1}}{2}\ny_3", {"x": np.array([1., 3., 5., 7.])}) - 4 == 0),
 ("EB1 float SHM ODE",            lambda: val(r"\ddot x + 9.81 x = 0, \; x(0) = 0.1, \dot x(0) = 0")[0].has(sp.cos)),
 ("EB4 eig(I_3)",                 lambda: list(val(r"\operatorname{eig}(I_3)")) == [1, 1, 1]),
 ("EB4 currents I_1, I_2",        lambda: (lambda v: close(v[Symbol("I_1")], 5/11))(val(r"\begin{bmatrix}30 & -20\\-20 & 50\end{bmatrix}\begin{bmatrix}I_1\\I_2\end{bmatrix} = \begin{bmatrix}10\\0\end{bmatrix}")[0])),
 ("EB5 grad = 0",                 lambda: val(r"\nabla (x^2 + x y + y^2 - 3x) = 0") == [{x: 2, y: -1}]),
 ("EB5 min s.t.",                 lambda: val(r"\min_x x^2 \text{ s.t. } x \geq 1") == 1),
 ("EB5 argmin r > 0",             lambda: close(val(r"\arg\min_{r > 0} \left(2\pi r^2 + \frac{2V}{r}\right)", {"V": 1.0}), 0.541926070139289, 1e-6)),
 ("EB7 back-substitution",        lambda: len(val("\\omega_n^2 = 10\n2 \\zeta \\omega_n = 2")) == 2),
 ("EB8 corr",                     lambda: close(val(r"\operatorname{corr}(x, y)", {"x": [1, 2, 3], "y": [2, 4, 6]}), 1)),
 ("EB10 ODE solution bound",      lambda: same(val("y' = -2y, \\; y(0) = 3\ny(1)"), 3 * exp(-2))),
 ("EB11 Newton recurrence",       lambda: close(val("x_0 = 1\nx_{n+1} = x_n - \\frac{x_n^2 - 2}{2 x_n}\nx_4"), 1.41421356237469, 1e-12)),
 ("F3 float solve is fast",       lambda: (lambda t0: close(val(r"5 - 5 e^{-1000 t} = 4.5")[0], 0.00230258509299405) and time.time() - t0 < 5)(time.time())),
]


import pytest


@pytest.mark.parametrize("name,fn", CASES, ids=[c[0] for c in CASES])
def test_round3_regression(name, fn):
    assert fn()
