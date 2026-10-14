import pytest

from conftest import same
from corpus import CORPUS
from openlab.engine import parse_expr


@pytest.mark.parametrize("case", [c for c in CORPUS if "matrix" in c.tags], ids=lambda c: c.src)
def test_parse_matrix(case):
    got, _ = parse_expr(case.src)
    assert same(got, case.expected), f"{case.src!r} -> {got!r}, expected {case.expected!r}"


def test_matrix_rows_may_span_lines():
    import openlab.engine as openlab
    r = openlab.run("A = \\begin{pmatrix}\n 1 & 2 \\\\\n 3 & 4\n\\end{pmatrix}", {})
    assert r.value.shape == (2, 2)
