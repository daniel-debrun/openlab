import pytest

from openlab.engine.teach import did_you_mean, precheck, teaching_hint

POSITIVE = [
    (precheck, r"sin(x)", "backslash"),
    (precheck, r"x**2", "^"),
    (precheck, r"x^10", "braces"),
    (precheck, r"pi r^2", r"\pi"),
    (precheck, r"inf", r"\infty"),
    (precheck, r"1 + 2 + \cdots + n", "sum"),
    (precheck, r"\left( x", r"\right"),
    (precheck, r"{x", "brace"),
    (teaching_hint, r"\int x^2", "differential"),
    (teaching_hint, r"x dx", r"d \cdot x"),
    (teaching_hint, r"\frac{d}{dx}", "differentiate"),
    (teaching_hint, r"\lim x", r"\to"),
    (teaching_hint, r"\sum x", "bounds"),
    (teaching_hint, r"\int \frac{dx}{x}", "after the fraction"),
    (teaching_hint, r"\sin x (y+1)", "parenthes"),
    (teaching_hint, r"\frac{1}", "two arguments"),
    (teaching_hint, r"val\_bacc = 0.83", r"\mathrm{val\_bacc}"),
]

NEGATIVE = [
    (precheck, r"\sin(x)"),
    (precheck, r"\arcsin x"),
    (precheck, r"\sinh x"),
    (precheck, r"\text{sin} x"),
    (precheck, r"x^{10}"),
    (precheck, r"\int_0^1 x dx"),
    (precheck, r"\pi r^2"),
    (precheck, r"\left( x \right)"),
    (teaching_hint, r"\int x dx"),
    (teaching_hint, r"\frac{d}{dx} x"),
    (teaching_hint, r"\sum_{i=1}^n i"),
    (teaching_hint, r"d + 1"),
    (teaching_hint, r"\mathrm{val\_bacc} = 0.83"),          # already upright: nothing to teach
    (teaching_hint, r"x_1 + x_2"),
]


def test_a_column_name_with_an_underscore_is_an_error_with_the_idiom_to_use():
    """`val\\_bacc` is what a pandas habit types; the engine reads `\\` and stops, so the hint has to carry
    the upright form -- and the upright form must then run."""
    from openlab.engine.records import run_records
    bad = run_records(r"val\_bacc = 0.83", {}, strict=False, stop_on_error=False)[0]
    assert bad["verdict"] == "error" and r"\mathrm{val\_bacc}" in bad["hint"]
    ns = {}
    good = run_records(r"\mathrm{val\_bacc} = 0.83" "\n" r"\mathrm{val\_bacc} + 1", ns, strict=False,
                       stop_on_error=False)
    assert [r["verdict"] for r in good] == ["bound", "value"]
    assert float(good[1]["value"]) == pytest.approx(1.83)


@pytest.mark.parametrize("fn,src,phrase", POSITIVE, ids=lambda v: v if isinstance(v, str) else None)
def test_hint_fires(fn, src, phrase):
    out = fn(src)
    text = " ".join(out) if isinstance(out, tuple) else out
    assert text and phrase in text, (src, out)


@pytest.mark.parametrize("fn,src", NEGATIVE, ids=lambda v: v if isinstance(v, str) else None)
def test_hint_stays_quiet(fn, src):
    assert not fn(src), (src, fn(src))


def test_did_you_mean():
    assert r"\sqrt" in did_you_mean("Sqrt")
    assert did_you_mean("foo") == ""
