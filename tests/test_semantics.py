import math

import numpy as np
import pytest
import sympy
from sympy import Lambda, Matrix, Rational, Symbol, sqrt

import openlab.engine as openlab
from openlab.engine import Kind, SemanticsError, run


def test_assign_and_substitute():
    ns = {}
    r = run(r"x = 3", ns)
    assert r.kind is Kind.ASSIGN and ns["x"] == 3 and r.bound == {"x": 3}
    r = run(r"x^2 + 1", ns)
    assert r.value == 10 and "with x = 3" in r.interpretation and r.interpretation.startswith("eval: x**2 + 1")


def test_define_and_call():
    ns = {}
    r = run(r"f(x) = x^2 - 2x + 1", ns)
    assert r.kind is Kind.DEFINE and isinstance(ns["f"], Lambda)
    assert run(r"f(3)", ns).value == 4
    y = Symbol("y")
    assert sympy.expand(run(r"f(y)", ns).value) == y**2 - 2 * y + 1


def test_define_ignores_bound_argument_name():
    ns = {"x": 3}
    run(r"g(x) = x + 1", ns)
    assert run(r"g(10)", ns).value == 11


def test_solve_one_unknown():
    r = run(r"x^2 - 5x + 6 = 0", {})
    assert r.kind is Kind.SOLVE and sorted(r.value) == [2, 3]
    assert "x = 2" in r.latex_out and "x = 3" in r.latex_out


def test_solve_two_unknowns_refused_and_for_flag():
    with pytest.raises(SemanticsError) as info:
        run(r"x^2 + w = 1", {})
    assert "--for" in info.value.hint
    r = run(r"x^2 + w = 1", {}, solve_for="w")
    assert r.value == [1 - Symbol("x") ** 2]


def test_check_relation():
    assert run(r"2 + 2 = 4", {}).value is sympy.true
    assert run(r"2 + 2 = 5", {}).value is sympy.false
    assert run(r"\sqrt{2} \approx 1.41", {}).value is sympy.true
    assert run(r"\sqrt{2} \approx 1.42", {}).value is sympy.false
    assert run(r"\pi \approx 3.14159", {}).value is sympy.true


def test_inequality():
    r = run(r"x^2 < 4", {})
    assert r.kind is Kind.RELATION
    x = Symbol("x")
    assert r.value == sympy.And(sympy.Lt(-2, x), sympy.Lt(x, 2))


def test_numeric_annotation():
    r = run(r"\sqrt{2}", {})
    assert r.value == sqrt(2) and abs(float(r.numeric) - math.sqrt(2)) < 1e-12
    assert r"\approx" in r.latex_out
    assert run(r"2 + 2", {}).numeric is None
    r = run(r"\pi", {}, numeric=True, precision=30)
    assert str(r.value).startswith("3.14159265358979323846264338328")


def test_calculus_evaluates():
    x = Symbol("x")
    assert run(r"\int_0^1 x^2 \, dx", {}).value == Rational(1, 3)
    assert run(r"\frac{d}{dx} \sin x \cos x", {}).value == sympy.cos(2 * x)
    assert run(r"\lim_{x \to 0} \frac{\sin x}{x}", {}).value == 1
    n = Symbol("n")
    assert sympy.expand(run(r"\sum_{k=1}^{n} k^2", {}).value - n * (n + 1) * (2 * n + 1) / 6) == 0
    r = run(r"F(t) = \int_0^t s^2 ds", {})
    assert run(r"F(2)", {"F": r.value}).value == Rational(8, 3)


def test_bound_variables_are_not_substituted():
    ns = {"x": 3}
    assert run(r"\int_0^1 x^2 dx", ns).value == Rational(1, 3)
    assert run(r"\frac{d}{dx} x^2", ns).value == 2 * Symbol("x")


def test_simplify_only_for_bare_expressions():
    ns = {}
    assert run(r"\sin^2 \theta + \cos^2 \theta", ns).value == 1
    r = run(r"y = x^2 - 2x + 1", ns)
    assert r.value == Symbol("x") ** 2 - 2 * Symbol("x") + 1  # stored as written
    assert run(r"\sin^2 \theta + \cos^2 \theta", ns, simplify=False).value != 1


def test_matrices():
    ns = {}
    run(r"A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}", ns)
    assert run(r"\det A", ns).value == -2
    assert run(r"A^T", ns).value == Matrix([[1, 3], [2, 4]])
    assert run(r"A^{-1}", ns).value == Matrix([[-2, 1], [Rational(3, 2), -Rational(1, 2)]])
    assert run(r"A \begin{pmatrix} 1 \\ 0 \end{pmatrix}", ns).value == Matrix([[1], [3]])
    run(r"B = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}", ns)
    assert run(r"A B - B A", ns).value == Matrix([[-1, -3], [3, 1]])
    assert run(r"\begin{vmatrix} 1 & 2 \\ 3 & 4 \end{vmatrix}", ns).value == -2


def test_numpy_and_python_callables():
    ns = {"g": math.sqrt, "v": np.array([1, 2, 3]), "M": np.eye(2) * 2, "h": lambda t: t**2 + 1}
    assert float(run(r"g(16)", ns).value) == 4.0
    assert run(r"h(3)", ns).value == 10
    assert run(r"h(z)", ns).value == Symbol("z") ** 2 + 1
    assert float(run(r"\det M", ns).value) == 4.0
    assert run(r"v^T v", ns).value == 14        # a 1×1 result is a number
    with pytest.raises(SemanticsError):
        run(r"g(z)", ns)


def test_apply_resolution_notes():
    ns = {"k": 2}
    r = run(r"k(x)", ns)
    assert r.value == 2 * Symbol("x") and "multiplication" in r.interpretation
    r = run(r"q(x)", ns)
    assert "undefined function" in r.interpretation
    with pytest.raises(openlab.ParseError) as info:   # multi-letter words are not names
        run(r"Sqrt(x)", ns)
    assert "backslash" in str(info.value)
    r = run(r"\operatorname{Sqrt}(x)", ns)
    assert "did you mean" in r.interpretation


def test_e_and_i_policies():
    assert run(r"e^{i\pi} + 1", {}).value == 0
    r = run(r"e^{i\pi} + 1", {}, complex_i=False)
    assert r.value != 0 and "plain symbol" in r.interpretation
    assert run(r"\ln e", {}).value == 1
    assert run(r"e^x", {"e": 2}).value == 2 ** Symbol("x")
    assert run(r"e", {}).value == sympy.E       # unbound e is Euler's number everywhere


def test_plus_minus():
    r = run(r"1 \pm 2", {})
    assert tuple(r.value) == (3, -1)


def test_multi_statement_cell_and_error_line():
    ns = {}
    results = run("a = 2\nb = a^2\n$$c = b + 1$$", ns)
    assert [r.value for r in results] == [2, 4, 5]
    assert ns["_openlab"].value == 5
    with pytest.raises(openlab.OpenlabError) as info:
        list(openlab.iter_run("a = 2\nb = sin(a)", ns))
    assert info.value.lineno == 2


def test_align_and_wrappers():
    ns = {}
    results = run(r"\begin{align} x &= 3 \\ y &= x + 1 \end{align}", ns)
    assert [r.value for r in results] == [3, 4]


def test_assignment_self_reference_refused():
    with pytest.raises(SemanticsError):
        run(r"x = x + 1", {})


def test_indexed_terms_from_namespace():
    ns = {"a_1": 1, "a_2": 2, "a_3": 3}
    assert run(r"\sum_{i=1}^{3} a_i", ns).value == 6


def test_text_and_latex_renderings():
    r = run(r"\int_0^1 x^2 dx", {})
    assert "interpreted as" in repr(r)
    assert "1/3" in repr(r)
    assert r._repr_latex_().startswith("$$")
    html = r._repr_html_()
    assert "interpreted as" in html and "\\int_0^1" in html
