"""The single table that drives the parse tests, the ambiguity audit and the oracle test."""

from dataclasses import dataclass

import sympy
from sympy import (Abs, Derivative, ImmutableMatrix, Integral, Limit, Product, Rational, Sum,
                   binomial, ceiling, cos, exp, factorial, log, oo, pi, sin, sqrt, symbols,
                   tan, asin, sinh, Lt)

from openlab.engine.transform import Apply, Det, PlusMinus

x, y, z, a, b, c, n, k, i, t, s, d, r, A, T, f, g = symbols("x y z a b c n k i t s d r A T f g")
theta, alpha, Omega, mass = symbols("theta alpha Omega mass")
x_1, x_2, theta_1, theta_ij, x_ip1 = symbols("x_1 x_2 theta_1 theta_ij x_{i+1}")
e = sympy.Symbol("e")
ai = sympy.IndexedBase("a")[i]


@dataclass
class Case:
    src: str
    expected: object
    tags: tuple


CORPUS = [
    # arithmetic & implicit multiplication
    Case(r"2 + 3", 5, ("arith", "oracle")),
    Case(r"a x - b y", a * x - b * y, ("arith", "implicit", "oracle")),
    Case(r"2(x)", 2 * x, ("implicit",)),
    Case(r"(x+1)(x-1)", (x + 1) * (x - 1), ("implicit", "oracle")),
    Case(r"x y z", x * y * z, ("implicit", "oracle")),
    Case(r"2 x^2 y", 2 * x**2 * y, ("implicit", "oracle")),
    Case(r"-x^2", -x**2, ("arith", "oracle")),
    Case(r"x^-1", 1 / x, ("arith",)),
    Case(r"x^{-1}", 1 / x, ("arith", "oracle")),
    Case(r"a/b c", a / b * c, ("arith",)),
    Case(r"1/2 x", x / 2, ("arith",)),
    Case(r"x \cdot -3", -3 * x, ("arith",)),
    Case(r"a \cdot b \cdot c d", a * b * c * d, ("arith", "oracle")),
    Case(r"a \times b", a * b, ("arith", "oracle")),
    Case(r"a \div b", a / b, ("arith", "oracle")),
    Case(r"\frac{1}{2} + \frac{1}{3}", Rational(5, 6), ("arith", "oracle")),
    Case(r"\frac12", Rational(1, 2), ("arith", "oracle")),
    Case(r"\dfrac{a}{b}", a / b, ("arith",)),
    Case(r"2\pi r", 2 * pi * r, ("arith", "oracle")),
    Case(r"e^{-x^2}", e ** (-x**2), ("arith",)),
    Case(r"2^{n+1}", 2 ** (n + 1), ("arith", "oracle")),
    Case(r"5!", 120, ("arith", "oracle")),
    Case(r"(n+1)!", factorial(n + 1), ("arith", "oracle")),
    Case(r"\binom{5}{2}", 10, ("arith", "oracle")),
    Case(r"\binom{n}{k}", binomial(n, k), ("arith", "oracle")),
    Case(r"\sqrt{2}", sqrt(2), ("arith", "oracle")),
    Case(r"\sqrt 2 x", sqrt(2) * x, ("arith",)),
    Case(r"\sqrt[3]{27}", 3, ("arith", "oracle")),
    Case(r"\sqrt[n]{x}", x ** (1 / n), ("arith",)),
    Case(r"|x - 1|", Abs(x - 1), ("arith", "oracle")),
    Case(r"|x||y||z|", Abs(x) * Abs(y) * Abs(z), ("arith",)),
    Case(r"|x - |y||", Abs(x - Abs(y)), ("arith",)),
    Case(r"\left| x \right|", Abs(x), ("arith", "oracle")),
    Case(r"\lvert x \rvert", Abs(x), ("arith",)),
    Case(r"\lfloor 3.7 \rfloor", 3, ("arith", "oracle")),
    Case(r"\lceil x \rceil", ceiling(x), ("arith", "oracle")),
    Case(r"\left( x + 1 \right)^2", (x + 1) ** 2, ("arith", "oracle")),
    Case(r"\left[ x + 1 \right] 2", 2 * (x + 1), ("arith",)),
    Case(r"3.5 x", 3.5 * x, ("arith", "oracle")),
    Case(r"x \, y", x * y, ("arith",)),
    Case(r"a \pm b", PlusMinus(a, b), ("arith",)),
    Case(r"\frac{-b \pm \sqrt{b^2 - 4ac}}{2a}", PlusMinus(-b, sqrt(b**2 - 4 * a * c)) / (2 * a), ("arith",)),
    Case(r"\infty", oo, ("arith", "oracle")),
    # names
    Case(r"\theta_1 + \theta_{ij}", theta_1 + theta_ij, ("names",)),
    Case(r"x_{i+1}", x_ip1, ("names",)),
    Case(r"\alpha \Omega", alpha * Omega, ("names",)),
    Case(r"\mathrm{mass} g", mass * g, ("names",)),
    Case(r"\text{mass}", mass, ("names",)),
    Case(r"d + 1", d + 1, ("names",)),
    Case(r"d^2", d**2, ("names",)),
    Case(r"d(x)", Apply(d, x), ("names",)),
    # functions
    Case(r"\sin x \cos x", sin(x) * cos(x), ("func",)),
    Case(r"\sin 2x + 1", sin(2 * x) + 1, ("func",)),
    Case(r"\sin x y", sin(x * y), ("func",)),
    Case(r"2 \sin x \cos x", 2 * sin(x) * cos(x), ("func",)),
    Case(r"\sin(x) y", sin(x) * y, ("func",)),  # sympy reads sin(x*y); we document parens as delimiting
    Case(r"\sin x^2", sin(x**2), ("func",)),
    Case(r"\sin^2 x", sin(x) ** 2, ("func", "oracle")),
    Case(r"\sin^{-1} x", asin(x), ("func",)),
    Case(r"\arcsin x", asin(x), ("func", "oracle")),
    Case(r"\sinh x", sinh(x), ("func", "oracle")),
    Case(r"\tan\theta", tan(theta), ("func", "oracle")),
    Case(r"\ln x^2", log(x**2), ("func",)),
    Case(r"\log x", log(x), ("func",)),
    Case(r"\log_2 8", 3, ("func", "oracle")),
    Case(r"\log_{10} 1000", 3, ("func",)),
    Case(r"\exp(x) \exp(y)", exp(x) * exp(y), ("func", "oracle")),
    Case(r"\sin x \cdot y z", sin(x) * y * z, ("func",)),
    Case(r"2 \cdot \sin x y", 2 * sin(x * y), ("func",)),
    Case(r"- \sin x", -sin(x), ("func",)),
    Case(r"x - \sin x", x - sin(x), ("func", "oracle")),
    Case(r"x^2 \sin x", x**2 * sin(x), ("func",)),
    Case(r"f(x)", Apply(f, x), ("func",)),
    Case(r"f(x, y)", Apply(f, x, y), ("func",)),
    Case(r"x(y+1)", Apply(x, y + 1), ("func",)),
    Case(r"\operatorname{sinc}(x)", Apply(sympy.Symbol("sinc"), x), ("func",)),
    Case(r"\sin f(x)", sin(Apply(f, x)), ("func",)),
    Case(r"\det A", Det(A), ("matrix",)),
    # calculus
    Case(r"\int_0^1 x^2 dx", Integral(x**2, (x, 0, 1)), ("calc-int", "oracle")),
    Case(r"\int_0^1 x^2 \, dx", Integral(x**2, (x, 0, 1)), ("calc-int",)),
    Case(r"\int x^2 + 1 dx", Integral(x**2 + 1, x), ("calc-int",)),
    Case(r"\int x^2 dx + 1", Integral(x**2, x) + 1, ("calc-int",)),
    Case(r"\int_0^1 \int_0^x y dy dx", Integral(y, (y, 0, x), (x, 0, 1)), ("calc-int",)),
    Case(r"\int e^{-x} \mathrm{d}x", Integral(e ** (-x), x), ("calc-int",)),
    Case(r"\int_{-\infty}^{\infty} f(x) dx", Integral(Apply(f, x), (x, -oo, oo)), ("calc-int",)),
    Case(r"\int^1_0 x dx", Integral(x, (x, 0, 1)), ("calc-int",)),
    Case(r"\int_0^\pi \sin\theta d\theta", Integral(sin(theta), (theta, 0, pi)), ("calc-int",)),
    Case(r"2 \int_0^1 x dx + 1", 2 * Integral(x, (x, 0, 1)) + 1, ("calc-int",)),
    Case(r"\int\limits_0^1 x dx", Integral(x, (x, 0, 1)), ("calc-int",)),
    Case(r"\frac{d}{dx} x^2", Derivative(x**2, x), ("calc-deriv", "oracle")),
    Case(r"\frac{dy}{dx}", Derivative(y, x), ("calc-deriv",)),
    Case(r"\frac{d^2}{dx^2} \sin x", Derivative(sin(x), (x, 2)), ("calc-deriv",)),
    Case(r"\frac{d}{dx} \sin x \cos x", Derivative(sin(x) * cos(x), x), ("calc-deriv",)),
    Case(r"\frac{d}{dx} x^2 + 1", Derivative(x**2, x) + 1, ("calc-deriv",)),
    Case(r"\frac{\partial f}{\partial x}", Derivative(f, x), ("calc-deriv",)),
    Case(r"\frac{\partial^2 f}{\partial x \partial y}", Derivative(f, x, y), ("calc-deriv",)),
    Case(r"\frac{\partial}{\partial x} (x^2 y)", Derivative(x**2 * y, x), ("calc-deriv",)),
    Case(r"\partial_x f", Derivative(f, x), ("calc-deriv",)),
    Case(r"\lim_{x \to 0} \frac{\sin x}{x}", Limit(sin(x) / x, x, 0, "+-"), ("calc-lim",)),
    Case(r"\lim_{x \to 0^+} \frac{1}{x}", Limit(1 / x, x, 0, "+"), ("calc-lim", "oracle")),
    Case(r"\lim_{x\to\infty} \frac{1}{x}", Limit(1 / x, x, oo), ("calc-lim", "oracle")),
    Case(r"\lim_{n \rightarrow \infty} (1 + \frac{1}{n})^n", Limit((1 + 1 / n) ** n, n, oo), ("calc-lim",)),
    Case(r"\sum_{k=1}^{n} k^2", Sum(k**2, (k, 1, n)), ("sum", "oracle")),
    Case(r"\sum_{k=1}^n k", Sum(k, (k, 1, n)), ("sum",)),
    Case(r"\sum_{n=0}^{\infty} \frac{x^n}{n!}", Sum(x**n / factorial(n), (n, 0, oo)), ("sum", "oracle")),
    Case(r"\sum_{i=1}^{n} a_i", Sum(ai, (i, 1, n)), ("sum",)),
    Case(r"\prod_{k=1}^{n} k", Product(k, (k, 1, n)), ("sum", "oracle")),
    Case(r"\frac{1}{n}\sum_{i=1}^n x_i", Sum(sympy.IndexedBase("x")[i], (i, 1, n)) / n, ("sum",)),
    Case(r"\sum_{k=1}^{n} k + 1", Sum(k, (k, 1, n)) + 1, ("sum",)),
    Case(r"\sum_{k=1}^{n} k \cdot 2", Sum(2 * k, (k, 1, n)), ("sum",)),
    Case(r"\frac{1}{2} \sum_{k=1}^n k \cdot 2", Sum(2 * k, (k, 1, n)) / 2, ("sum",)),
    Case(r"\sin x \cdot 2", 2 * sin(x), ("func",)),
    Case(r"x \cdot \sin y \int z dz", x * sin(y) * Integral(z, z), ("calc-int",)),
    # matrices
    Case(r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}", ImmutableMatrix([[1, 2], [3, 4]]), ("matrix", "oracle")),
    Case(r"\begin{bmatrix} a & b \end{bmatrix}", ImmutableMatrix([[a, b]]), ("matrix", "oracle")),
    Case(r"\begin{vmatrix} 1 & 2 \\ 3 & 4 \end{vmatrix}", Det(ImmutableMatrix([[1, 2], [3, 4]])), ("matrix",)),
    Case(r"\begin{pmatrix} 1 \\ 0 \end{pmatrix}^T", ImmutableMatrix([[1, 0]]), ("matrix",)),
    Case(r"A^T", A**T, ("matrix",)),
    Case(r"2 \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}", ImmutableMatrix([[2, 0], [0, 2]]), ("matrix",)),
    # relations
    Case(r"x < 2", Lt(x, 2), ("relation", "oracle")),
    Case(r"x \le 2", sympy.Le(x, 2), ("relation", "oracle")),
    Case(r"x \neq 2", sympy.Ne(x, 2), ("relation", "oracle")),
]

# Statements that must be refused, with the phrase the hint has to contain.
REFUSED = [
    (r"sin(x)", "backslash"),
    (r"x**2", "^"),
    (r"a^b^c", "a^{b^c}"),
    (r"x^10", "braces"),
    (r"\int x^2", "differential"),
    (r"x dx", "d \\cdot x"),
    (r"2 d x", "d \\cdot x"),
    (r"\frac{d}{dx}", "differentiate"),
    (r"\sum_i x_i", "bounds"),
    (r"\lim x", "\\to"),
    (r"\int \frac{x \, dx}{x^2 + 1}", "after the fraction"),     # \frac{dx}{x} alone now reads (R7)
    (r"frac{1}{2}", "backslash"),
    (r"sqrt(2)", "backslash"),
    (r"pi r^2", "backslash"),
    (r"\begin{pmatrix} 1 & 2 \\ 3 \end{pmatrix}", "same number"),
    (r"\frac{d^2 y}{dx}", "d^2"),
    (r"\left( x + 1", "\\right"),
    (r"\frac{1}", "two arguments"),
    (r"\begin{pmatrix} 1 \end{bmatrix}", "\\end{pmatrix}"),
]
