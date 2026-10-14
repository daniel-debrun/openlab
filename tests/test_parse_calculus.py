import pytest

from conftest import same
from corpus import CORPUS
from openlab.engine import parse_expr

TAGS = ("calc-int", "calc-deriv", "calc-lim", "sum")


@pytest.mark.parametrize("case", [c for c in CORPUS if any(t in c.tags for t in TAGS)],
                         ids=lambda c: c.src)
def test_parse_calculus(case):
    got, _ = parse_expr(case.src)
    assert same(got, case.expected), f"{case.src!r} -> {got!r}, expected {case.expected!r}"


def test_differential_forms():
    forms = [r"\int x dx", r"\int x d x", r"\int x \, dx", r"\int x \mathrm{d}x",
             r"\int x \text{d}x", r"\int x d{x}", r"\int x\,\mathrm{d} x"]
    results = {str(parse_expr(f)[0]) for f in forms}
    assert results == {"Integral(x, x)"}
