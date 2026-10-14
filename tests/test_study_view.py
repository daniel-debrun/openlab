"""Guarantee: a view at t never holds a row that became known after t, and a vintage dataset gives the value as
it was published at t."""

import math
from datetime import date, datetime, timedelta

import polars as pl
import pytest
from hypothesis import given, settings, strategies as st

from openlab.study.errors import StudyError
from openlab.study.spec import DatasetSpec
from openlab.study.durations import Duration
from openlab.study.view import Rule, Timeline, View, known_at, rule

D0 = datetime(2024, 1, 1)
RULES = ["date", "date - 1d + 18h", "published", "date + 2h - 30min"]
instants = st.builds(lambda m: D0 + timedelta(minutes=m), st.integers(-2 * 1440, 40 * 1440))


@settings(max_examples=200, deadline=None)
@given(rows=st.lists(st.tuples(st.integers(0, 30), st.integers(0, 3000), st.sampled_from("ab")), max_size=40),
       known=st.sampled_from(RULES), t=instants)
def test_view_never_holds_a_row_known_after_t(rows, known, t):
    df = pl.DataFrame({"id": list(range(len(rows))), "date": [D0.date() + timedelta(days=d) for d, _, _ in rows],
                       "published": [D0 + timedelta(days=d, minutes=lag) for d, lag, _ in rows],
                       "site": [k for _, _, k in rows]},
                      schema={"id": pl.Int64, "date": pl.Date, "published": pl.Datetime("us"), "site": pl.String})
    ds = DatasetSpec("x", time="date", key=("site",), known_at=known)
    out = View({"x": df}, {"x": ds}, t).frame("x")
    stamps = known_at(df, ds).to_list()
    assert all(k <= t for k in known_at(out, ds).to_list())
    assert sorted(out["id"]) == [i for i, k in enumerate(stamps) if k <= t]


@st.composite
def vintages(draw):
    """(key, date) observations, each with consecutive vintages; some end (the value was withdrawn)."""
    rows = []
    for key in draw(st.lists(st.sampled_from("abc"), min_size=1, max_size=3, unique=True)):
        for obs in draw(st.lists(st.integers(0, 20), min_size=1, max_size=4, unique=True)):
            starts = sorted(draw(st.lists(st.integers(obs, obs + 40), min_size=1, max_size=4, unique=True)))
            ends = [s - 1 for s in starts[1:]] + [draw(st.sampled_from([None, starts[-1] + 5]))]
            for s, e in zip(starts, ends):
                rows.append((key, obs, s, e, draw(st.floats(-5, 5))))
    return rows


@settings(max_examples=200, deadline=None)
@given(rows=vintages(), t=instants)
def test_vintages_return_the_value_as_published_at_t(rows, t):
    day = lambda n: (D0.date() + timedelta(days=n)).isoformat()             # noqa: E731
    df = pl.DataFrame({"series": [r[0] for r in rows], "date": [day(r[1]) for r in rows],
                       "start": [day(r[2]) for r in rows], "end": ["9999-12-31" if r[3] is None else day(r[3])
                                                                  for r in rows],
                       "value": [r[4] for r in rows]})
    ds = DatasetSpec("fred", time="date", key=("series",), vintages=("start", "end"))
    seen = View({"fred": df}, {"fred": ds}, t).frame("fred")
    assert seen["end"].null_count() == seen.height             # when the next revision came is not known at t
    got = {(k, d): v for k, d, v in seen.select("series", "date", "value").iter_rows()}
    today = (t.date() - D0.date()).days
    want = {}
    for key, obs, s, e, v in sorted(rows, key=lambda r: r[2]):              # dates are whole days, ends inclusive
        if s <= today and (e is None or today <= e):
            want[(key, day(obs))] = v
    assert got == want


@pytest.mark.parametrize("text, parsed", [
    ("date - 1d + 18h", Rule("date", Duration(delta=timedelta(hours=-6)))),
    ("date+2h", Rule("date", Duration(delta=timedelta(hours=2)))),
    ("report date - 30min + 1d", Rule("report date", Duration(delta=timedelta(days=1, minutes=-30)))),
    ("as-of - 1d", Rule("as-of", Duration(delta=timedelta(days=-1)))),
    ("published", Rule("published")),
    ("date - 1x", Rule("date - 1x")),                                         # not an expression: a column name
])
def test_known_at_expressions_parse(text, parsed):
    assert rule(DatasetSpec("x", time="date", known_at=text)) == parsed


def test_known_at_rules_apply():
    df = pl.DataFrame({"date": [date(2024, 1, 2)], "start": ["2024-01-05"], "end": ["2024-02-01"]})
    at = lambda **kw: known_at(df, DatasetSpec("x", **kw))[0]                # noqa: E731
    assert at(time="date") == datetime(2024, 1, 2)
    assert at(time="date", known_at="date - 1d + 18h") == datetime(2024, 1, 1, 18)
    assert at(time="date", vintages=("start", "end")) == datetime(2024, 1, 5)
    assert known_at(df, DatasetSpec("x")).null_count() == 1                  # a static table: known always
    assert View({"x": df}, {"x": DatasetSpec("x")}, datetime(1900, 1, 1)).frame("x").height == 1
    with pytest.raises(StudyError, match="no column `date - 1x`"):
        known_at(df, DatasetSpec("x", time="date", known_at="date - 1x"))


def test_series_at_moves_by_calendar_months():
    df = pl.DataFrame({"date": [date(2024, 1, 31), date(2024, 2, 29), date(2024, 3, 31)], "v": [1.0, 2.0, 3.0]})
    tl = Timeline({"x": df}, {"x": DatasetSpec("x", time="date")})
    s = tl.at(datetime(2024, 3, 31)).series("x", "v")
    assert s.at(datetime(2024, 3, 31)) == 3.0
    assert s.at(datetime(2024, 3, 31), back="1M") == 2.0                     # 2024-02-29, not 03-02
    assert s.at(datetime(2024, 3, 31), back="2M") == 1.0
    assert s.at(datetime(2024, 3, 30), back="10d") == 2.0
    assert math.isnan(s.at(datetime(2024, 1, 30)))
    assert tl.at(datetime(2024, 3, 30)).series("x", "v").at(datetime(2024, 3, 31), ahead="1d") == 2.0  # 03-31 unknown
