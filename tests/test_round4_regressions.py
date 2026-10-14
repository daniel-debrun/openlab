"""Round-4 regressions (student, engineer, adversarial QA): minimal repro + right answer."""
import sympy as sp
import openlab.engine as openlab
from sympy import Symbol, Rational, exp, sin, cos, pi, sqrt, Matrix, symbols, FiniteSet
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
 # === student ======================================================================================
 ("avg rate (f(3)-f(1))/(3-1)",   lambda: val("f(x) = x^2\n\\frac{f(3) - f(1)}{3 - 1}") == 4),
 ("MVT f'(c) = …",                lambda: set(val("f(x) = x^3\nf'(c) = \\frac{f(3) - f(0)}{3 - 0}")) == {-sqrt(3), sqrt(3)}),
 ("(f(4)+f(1))/(5-2)",            lambda: val("f(x) = x^2\n\\frac{f(4)+f(1)}{5-2}") == Rational(17, 3)),
 ("trig quadratic in sin",        lambda: val(r"2\sin^2 x - \sin x - 1 = 0, \; 0 \le x < 2\pi") == [pi/2, 7*pi/6, 11*pi/6]),
 ("trig quadratic in cos",        lambda: val(r"2\cos^2 x + \cos x - 1 = 0, \; 0 \le x < 2\pi") == [pi/3, pi, 5*pi/3]),
 ("int 1/x = ln|x| + C",          lambda: val(r"\int \frac{1}{x} \, dx = \ln|x| + C") is sp.true),
 ("int tan = -ln|cos| + C",       lambda: val(r"\int \tan x \, dx = -\ln|\cos x| + C") is sp.true),
 ("int x = x^2 still False",      lambda: val(r"\int x\,dx = x^2") is sp.false),
 ("sum k (1/2)^k",                lambda: val(r"\sum_{k=1}^{\infty} k \left(\frac{1}{2}\right)^k") == 2),
 ("sum k (0.5)^k",                lambda: close(val(r"\sum_{k=1}^{\infty} k (0.5)^k"), 2)),
 ("x (1/2)^2 product",            lambda: val(r"x \left(\frac{1}{2}\right)^2") == x/4),
 ("related rates dV/dt",          lambda: val("V = \\frac{4}{3}\\pi r^3\nr = 2\n\\frac{dr}{dt} = 1\n\\frac{dV}{dt}") == 16*pi),
 ("related rates dA/dt stated",   lambda: close(val("r = 3\n\\frac{dr}{dt} = 0.5\n\\frac{dA}{dt} = 2\\pi r \\frac{dr}{dt}"), 3*pi)),
 ("dy/dx = 2x still an ODE",      lambda: val(r"\frac{dy}{dx} = 2x")[0].has(Symbol("C1"))),
 ("30 C in F",                    lambda: "86" in str(val(r"30\,^\circ\text{C} \text{ in } ^\circ\text{F}"))),
 ("m x m in cm^2",                lambda: "120000*centimeter**2" == str(val(r"3\,\text{m} \times 4\,\text{m} \text{ in } \text{cm}^2"))),
 ("logistic ODE explicit",        lambda: close(val(r"\frac{dP}{dt} = 0.1 P \left(1 - \frac{P}{1000}\right), \; P(0) = 100")[0].subs(t, 10), 1000/(1 + 9*exp(-1)))),
 ("v_f = v_0 + a t",              lambda: val("v_0 = 5\na = 2\nv_f = v_0 + a \\cdot 3") == 11),
 ("T_f = T_0 - 20",               lambda: val("T_0 = 100\nT_f = T_0 - 20") == 80),
 ("Let x be … Then 3x - 7 = 29",  lambda: val(r"\text{Let } x \text{ be the number. Then } 3x - 7 = 29") == [12]),
 ("grad f(1,2) . u",              lambda: val("f(x,y) = x^2 y\n\\nabla f(1, 2) \\cdot \\left(\\frac{3}{5}, \\frac{4}{5}\\right)") == Rational(16, 5)),
 ("u . (v x w)",                  lambda: val("u = (1, 2, 3)\nv = (0, 1, 4)\nw = (2, 0, 1)\nu \\cdot (v \\times w)") == 11),
 ("log_2 x^3 = 3 log_2 x",        lambda: val(r"\log_2 (x^3) = 3\log_2 x") is sp.true),
 ("ln x^2 = 2 ln x qualified",    lambda: "positive" in res(r"\ln(x^2) = 2\ln x").latex_out),
 ("pi/3 in degrees",              lambda: val(r"\frac{\pi}{3} \text{ in } ^\circ") == 60),
 ("overline(3+4i)",               lambda: val(r"\overline{3 + 4i}") == 3 - 4*sp.I),
 ("z zbar = 25",                  lambda: val(r"(3 + 4i)\overline{(3 + 4i)}") == 25),
 ("-7 mod 3",                     lambda: val(r"-7 \bmod 3") == 2),
 ("(-8)^(1/3) noted",             lambda: "principal" in res(r"(-8)^{1/3}").interpretation),
 ("max … for 0<=x<=3",            lambda: val(r"\max_{x} (x^3 - 12x) \text{ for } 0 \le x \le 3") == 0),
 ("max_{0<=x<=3}",                lambda: val(r"\max_{0 \le x \le 3} (x^3 - 12x)") == 0),
 ("T(n) = 2T(n-1)+1 refused",     lambda: raises(r"T(n) = 2T(n-1) + 1")),
 ("k parameter system",           lambda: val("k x + y = 1\nx + k y = 1") == [{x: 1/(k + 1), y: 1/(k + 1)}]),
 ("sum (-1)^(n+1)/sqrt n",        lambda: close(res(r"\sum_{n=1}^\infty \frac{(-1)^{n+1}}{\sqrt n}").numeric, 0.604898643421630)),
 ("mid-line period",              lambda: [r.value for r in openlab.run_all("x + 1 = 5. y = 2", {})] == [[4], 2]),
 ("Solve: colon",                 lambda: val(r"\text{Solve}: x^2-16=0") == [-4, 4]),
 ("trailing ?",                   lambda: val(r"2x + 1 = 7?") == [3]),
 ("prose-only line skipped",      lambda: val("\\text{In 4 years Ann will be twice as old as she was 2 years ago:}\nx + 4 = 2(x - 2)") == [8]),
 ("Greek differential",           lambda: val(r"\int_0^\pi \sin\theta\,d\theta") == 2),
 ("U setminus (A cup B)",         lambda: val("U = \\{1,2,3,4,5\\}\nA = \\{1,2\\}\nB = \\{2,3\\}\nU \\setminus (A \\cup B)") == FiniteSet(4, 5)),
 ("|A x B|",                      lambda: val("A = \\{1,2,3\\}\nB = \\{4,5\\}\n|A \\times B|") == 6),
 ("on [0, 2pi]",                  lambda: val(r"\sin x = \frac{\sqrt 2}{2} \text{ on } [0, 2\pi]") == [pi/4, 3*pi/4]),
 ("\\vert, \\Vert",               lambda: val(r"\vert -5 \vert") == 5 and val(r"\Vert (3,4) \Vert") == 5),
 ("degrees in -> degrees note",   lambda: "30°" in res(r"2\sin\theta - 1 = 0, \; 0^\circ \le \theta < 360^\circ").interpretation),
]

import numpy as np
_Ac = np.array([[0., 1.], [-2., -3.]])
CASES += [
 # === engineer =====================================================================================
 ("N1 secant slope",              lambda: val("f(x) = x^3\n\\frac{f(3) - f(1)}{3 - 1}") == 13),
 ("N1 forward difference",        lambda: close(val("y(t) = e^{-t}\n\\frac{y(0.01) - y(0)}{0.01}"), (exp(-0.01) - 1) / 0.01)),
 ("N2 BC at y(L)",                lambda: same(val(r"y'' = 1, \; y(0) = 0, y(L) = 0", {"L": 2})[0], x**2/2 - x)),
 ("N2 BC at y(pi/4)",             lambda: same(val(r"y'' + 4y = 0, \; y(0) = 0, \; y(\pi/4) = 1")[0], sin(2*x))),
 ("N3 d/dtheta",                  lambda: same(val(r"\frac{d}{d\theta} \sin\theta"), cos(Symbol("theta")))),
 ("N3 d/dlambda MLE",             lambda: val(r"\frac{d}{d\lambda}\left(5 \ln\lambda - 5\lambda\right) = 0") == [1]),
 ("E2 d/dR_L",                    lambda: same(val(r"\frac{d}{dR_L}\left(R_L^2\right)"), 2*Symbol("R_L"))),
 ("E2 partial times factor",      lambda: same(val("f(x, y) = x y\n\\frac{\\partial f}{\\partial x}\\sigma_x"), y*Symbol("sigma_x"))),
 ("N4 block rank [B AB]",         lambda: val(r"\operatorname{rank}\begin{bmatrix} B & A B \end{bmatrix}", {"A": _Ac, "B": np.array([[0.], [1.]])}) == 2),
 ("N4 block diag",                lambda: val(r"\begin{bmatrix} A & 0 \\ 0 & A \end{bmatrix}", {"A": _Ac}).shape == (4, 4)),
 ("N5 rational char eq",          lambda: len(val(r"1 + \frac{6}{s(s+1)(s+2)} = 0")) == 3),
 ("N6 delta(x) own function",     lambda: val("\\delta(x) = x^2\n\\delta(3)") == 9),
 ("N6 P(v) power curve",          lambda: close(val("P(v) = \\frac12 \\rho A v^3\nP(10)", {"rho": 1.2, "A": 2.0}), 1200)),
 ("N7 defined f with domain",     lambda: val("y(t) = t^2\ny(t) = 4, \\; t > 0") == [2]),
 ("N8 micro coulomb",             lambda: "C" in str(val(r"3\,\mu\text{C}"))),
 ("N8 mAh / mA = 8 h",            lambda: close(val(r"\frac{2000\,\text{mAh}}{250\,\text{mA}}").args[0] if val(r"\frac{2000\,\text{mAh}}{250\,\text{mA}}").args[0] != 0 else 0, 28800)),
 ("N10 yhat_i in a sum",          lambda: close(val("\\hat y = X b\n\\sum_{i=1}^{4} (y_i - \\hat y_i)^2", {"X": np.column_stack([np.ones(4), np.arange(4.)]), "y": np.array([1., 2.9, 5.1, 7.]), "b": np.array([1., 2.])}), 0.02)),
 ("N10 Re split note",            lambda: "mathrm{Re}" in res(r"f_D = \frac{64}{Re}", {"Re": 1000.0}).interpretation),
 ("N11 max |eig|",                lambda: close(val(r"\max |\operatorname{eig}(A)|", {"A": np.array([[1., 0.], [0., -4.]])}), 4)),
 ("E1 omega_d not recurrence",    lambda: close(val(r"\omega_d = \omega_n \sqrt{1 - \zeta^2}", {"omega_n": 4.0, "zeta": 0.5}), 3.46410161513775)),
 ("E1 s_p not recurrence",        lambda: close(val(r"s_p = \sqrt{s_1^2 + s_2^2}", {"s_1": 3.0, "s_2": 4.0}), 5)),
 ("E3 Colebrook numeric",         lambda: close(val(r"\frac{1}{\sqrt{f}} = -2\log_{10}\left(\frac{0.0015}{3.7} + \frac{2.51}{10^5 \sqrt{f}}\right)")[0], 0.0237357308519095, 1e-9)),
 ("E4 Qdot = mdot c dT",          lambda: close(val(r"\dot Q = \dot m c \Delta T", {"mdot": 2.0, "c": 4186.0, "Delta_T": 10.0}), 83720)),
 ("E5 max of eig tuple",          lambda: close(val(r"\max(\operatorname{eig}(A))", {"A": np.array([[2., 1.], [1., 2.]])}), 3)),
 ("E8 Re^0.8 Pr^0.4",             lambda: close(val(r"\mathrm{Re}^{0.8} \mathrm{Pr}^{0.4}", {"Re": 1e4, "Pr": 5.0}), 1e4**0.8 * 5.0**0.4, 1e-6)),
 # === adversarial ==================================================================================
 ("A 25C in F",                   lambda: "77" in str(val(r"25\,^\circ\text{C} \text{ in } ^\circ\text{F}"))),
 ("A -40C in F",                  lambda: "-40" in str(val(r"-40\,^\circ\text{C} \text{ in } ^\circ\text{F}"))),
 ("A 20C + 5K",                   lambda: abs(float(val(r"20\,^\circ\text{C} + 5\,\text{K}").args[0]) - 298.15) < 1e-9),
 ("A 300K - 25C",                 lambda: abs(float(val(r"300\,\text{K} - 25\,^\circ\text{C}").args[0]) - 1.85) < 1e-9),
 ("A (100C-20C) in K diff",       lambda: "80" in str(val(r"(100\,^\circ\text{C} - 20\,^\circ\text{C}) \text{ in } \text{K}"))),
 ("A 100 m / 4 m",                lambda: val(r"100\,\text{m} / 4\,\text{m}") == 25),
 ("A 100 km / 2 h",               lambda: "50" in str(val(r"100\,\text{km} / 2\,\text{h}"))),
 ("A ln(5 m / 1 m)",              lambda: val(r"\ln(5\,\text{m} / 1\,\text{m})") == sp.log(5)),
 ("A 4 in in cm",                 lambda: "254" in str(val(r"4\,\text{in} \text{ in } \text{cm}"))),
 ("A 1 L in m^3",                 lambda: str(val(r"1\,\text{L} \text{ in } \text{m}^3")) == "meter**3/1000"),
 ("A 5 m in cm (one text)",       lambda: "500" in str(val(r"5 \text{ m in cm}"))),
 ("A e^{2 s} refused",            lambda: raises(r"e^{2\,\text{s}}")),
 ("A V = I R with units",         lambda: "ampere" in str(val("V = 12\\,\\text{V}\nR = 4\\,\\Omega\nV = I R"))),
 ("A where r = 3 is local",       lambda: (lambda ns: (run(r"A = \pi r^2 \text{ where } r = 3", ns), ns["r"] == 10 and ns["A"] == 9*pi)[1])({"r": 10})),
 ("A at x=1, y=2",                lambda: (lambda ns: (val(r"x^2 + y \text{ at } x = 1, y = 2", ns) == 3) and "y" not in ns)({})),
 ("A Z domain is local",          lambda: (lambda ns: (run(r"x^2 - 3 = 0, \; x \in \mathbb{Z}", ns), val("2x = 3", ns) == [Rational(3, 2)])[1])({})),
 ("A x = 3 < 4",                  lambda: (lambda ns: (run(r"x = 3 < 4", ns), ns["x"] == 3)[1])({})),
 ("A line magics distinct ids",   lambda: (lambda ns: [openlab.run_all("g(x) = x^2 - 9", ns, cell_id="c#1"), openlab.run_all("g(x) = 0", ns, cell_id="c#2")][1][-1].value == [-3, 3])({})),
 ("A sum sin n/n not diverges",   lambda: str(val(r"\sum_{n=1}^{\infty} \frac{\sin n}{n}")) != "DIVERGES"),
 ("A tiny value keeps magnitude", lambda: abs(float(res(r"\frac{1}{3} \times 10^{-20}").numeric) - 3.33333333333333e-21) < 1e-30),
 ("A huge factorial ratio",       lambda: val(r"(10^{100})! / (10^{100}-1)!") == 10**100),
 ("A 10-letter tautology",        lambda: val(r"p_{1} \lor p_{2} \lor p_{3} \lor p_{4} \lor p_{5} \lor p_{6} \lor p_{7} \lor p_{8} \lor p_{9} \lor \neg p_{1}") is sp.true),
 ("A inconsistent P refused",     lambda: raises("P(A) = 0.6\nP(B) = 0.7\nP(A \\cap B) = 0.1")),
 ("A f^{-1}(-8) for x^3",         lambda: val("f(x) = x^3\nf^{-1}(-8)") == -2),
 ("A arcsin x = 2 no solution",   lambda: val(r"\arcsin x = 2") == []),
 ("A sqrt 16 refused",            lambda: raises(r"\sqrt 16")),
 ("A x = 5 quad y = 6",           lambda: [r.value for r in openlab.run_all(r"x = 5 \quad y = 6", {})] == [5, 6]),
 ("A s.t. and",                   lambda: val(r"x^2 = 4 \text{ s.t. } x > 0 \text{ and } x < 1") == []),
 ("A A / vector refused",         lambda: raises(r"\frac{A}{B}", {"A": Matrix([[0, 1], [-2, -3]]), "B": Matrix([0, 1])})),
 ("A x^10 + 1 complex roots",     lambda: len(val(r"x^{10} + 1 = 0")) == 10),
 # === parser audit 2026-10-06: misreads that gave a confident wrong verdict =========================
 ("P 2 < 3 = 3 read whole",       lambda: val(r"2 < 3 = 3") is sp.true),
 ("P implies before =",           lambda: val(r"x \ge 0 \implies \sqrt{x^2} = x") is sp.true),
 ("P so ends a statement",        lambda: [r.value for r in openlab.run_all(r"x = -3 \text{ so } |x| = 3", {})] == [-3, sp.true]),
 ("P prose between terms refused", lambda: raises(r"\text{the value } x = 2 \text{ is a root of } x^2 - 4")),
 ("P a = 2, b = 3, a < b",        lambda: [r.value for r in openlab.run_all(r"a = 2, b = 3, a < b", {})] == [2, 3, sp.true]),
 ("P comma in a chain refused",   lambda: raises(r"\langle u, v \rangle = 0, \langle u, w \rangle = 1")),
 ("P bar over X_{t-1} is a name", lambda: "conjugate" not in res(r"\bar{X_{t-1}} = 2").interpretation),
]


import pytest


@pytest.mark.parametrize("name,fn", CASES, ids=[c[0] for c in CASES])
def test_round4_regression(name, fn):
    assert fn()


def test_a_unit_never_reaches_across_a_line():
    """`(…)` at a line's end and `\\mathrm{Hz}` (a unit spelling) starting the next line are two statements,
    not a quantity in hertz."""
    from openlab.engine.parse import statements
    cs = statements("a = (1 - 2)\n\\mathrm{Hz} = 3")
    assert len(cs) == 2 and cs[1].text.lstrip().startswith("\\mathrm{Hz}")
    assert "openlabunit" not in cs[0].text
