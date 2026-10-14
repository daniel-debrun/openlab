import pytest

from conftest import same
from corpus import CORPUS, REFUSED
from openlab.engine import OpenlabError, parse_expr, parse_statement

TAGS = ("arith", "implicit", "names", "func", "relation")


@pytest.mark.parametrize("case", [c for c in CORPUS if any(t in c.tags for t in TAGS)],
                         ids=lambda c: c.src)
def test_parse(case):
    got, _notes = parse_expr(case.src)
    assert same(got, case.expected), f"{case.src!r} -> {got!r}, expected {case.expected!r}"


@pytest.mark.parametrize("src,phrase", REFUSED, ids=lambda v: v if isinstance(v, str) and "\\" in v or " " in str(v) else None)
def test_refused_with_hint(src, phrase):
    with pytest.raises(OpenlabError) as info:
        parse_statement(src)
    assert phrase in info.value.hint or phrase in info.value.message, str(info.value)


def test_notes_surface_policies():
    _, notes = parse_expr(r"\sin x y")
    assert any("sin(x*y)" in n for n in notes)
    _, notes = parse_expr(r"\sin^{-1} x")
    assert any("inverse" in n for n in notes)
    _, notes = parse_expr(r"\log x")
    assert any("natural" in n for n in notes)
