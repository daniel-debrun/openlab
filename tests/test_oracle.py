"""Differential test against sympy's own LaTeX parser on the subset where it is unambiguous."""

import pytest
import sympy

from conftest import reeval
from corpus import CORPUS

parse_latex = pytest.importorskip("sympy.parsing.latex").parse_latex


def _oracle(src):
    try:
        got = parse_latex(src, backend="lark")
    except Exception as exc:  # the oracle cannot read it
        return None, f"raises {type(exc).__name__}"
    if not isinstance(got, sympy.Basic):
        return None, f"returns {type(got).__name__}"
    return got, ""


@pytest.mark.parametrize("case", [c for c in CORPUS if "oracle" in c.tags], ids=lambda c: c.src)
def test_agrees_with_sympy_parser(case):
    from openlab.engine import parse_expr
    oracle, why = _oracle(case.src)
    if oracle is None:
        pytest.skip(f"sympy oracle {why}")
    reeval_val = reeval(parse_expr(case.src)[0])
    if reeval_val == oracle or reeval_val.doit() == oracle.doit():
        return
    assert isinstance(reeval_val, sympy.Expr) and isinstance(oracle, sympy.Expr), f"openlab {reeval_val} vs sympy {oracle}"
    diff = sympy.simplify(reeval_val.doit() - oracle.doit())
    assert diff == 0, f"openlab {reeval_val} vs sympy {oracle}"
