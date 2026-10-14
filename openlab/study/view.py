"""Point-in-time views: every row of a dataset became known at some time, and a view at t holds only the rows
known by t.

When a row became known is its dataset's `known_at` rule:
- a column name: that column holds the time (e.g. `published`);
- an expression `<column> (+|-) <n><unit> ...`, e.g. `date - 1d + 18h` (units of openlab.study.durations);
- `vintages = [start, end]`: each row is one published value, valid from start through end (a date end holds
  for its whole day; a null end is still valid). A view gives, per key and time, the latest-starting row valid
  at t.
A dataset with no `known_at` is known at its `time` column, when it happened. One with neither (a reference
table) is known always and returned whole. A row whose known-at time is null is never known.

Times compare as naive UTC: a date is its midnight, a zoned time is converted to UTC, and strings are parsed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import polars as pl

from .durations import Duration, parse, split_tail
from .errors import StudyError
from .spec import DatasetSpec

@dataclass(frozen=True)
class Rule:
    """A parsed known_at rule: a row is known at `column` + `shift`; with vintages it holds until `until`."""
    column: str
    shift: Duration = Duration()
    until: str = ""


def rule(ds: DatasetSpec) -> Rule | None:
    """The dataset's known_at rule, parsed (vintages take precedence over known_at); None for a static table."""
    if ds.vintages:
        return Rule(ds.vintages[0], until=ds.vintages[1])
    text = (ds.known_at or ds.time).strip()
    if not text:
        return None
    column, shift = split_tail(text)
    return Rule(column, shift)


def _parse(df: pl.DataFrame, col: str, ds: DatasetSpec) -> pl.Series:
    """Column `col` as dates or naive UTC datetimes."""
    if col not in df.columns:
        raise StudyError(f"data.{ds.name}: no column `{col}`",
                         "known_at is a column, `<column> +/- <n><unit> ...` (min, h, d, w, M, Y), or vintages; time, key and vintages "
                         "name columns", f"its columns: {', '.join(df.columns)}")
    s = df[col]
    try:
        if s.dtype == pl.String:
            try:
                s = s.str.to_date()
            except pl.exceptions.PolarsError:
                s = s.str.to_datetime(time_unit="us")
    except pl.exceptions.PolarsError:
        raise StudyError(f"data.{ds.name}.{col} does not parse as times", fix="use ISO dates or datetimes") from None
    if s.dtype == pl.Date:
        return s
    if isinstance(s.dtype, pl.Datetime):
        if s.dtype.time_zone:
            s = s.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        return s.cast(pl.Datetime("us"))
    raise StudyError(f"data.{ds.name}.{col} is {s.dtype}, not a time",
                     fix="make it a date, a datetime or an ISO string")


def _when(df: pl.DataFrame, col: str, ds: DatasetSpec) -> pl.Series:
    return _parse(df, col, ds).cast(pl.Datetime("us"))


def known_at(df: pl.DataFrame, ds: DatasetSpec) -> pl.Series:
    """When each row of `df` became known (naive UTC); all null for a static table, which is known always."""
    r = rule(ds)
    if r is None:
        return pl.Series("known_at", [None] * df.height, pl.Datetime("us"))
    return r.shift.on(_when(df, r.column, ds)).alias("known_at")


def _instant(t) -> datetime:
    """`t` (a date or datetime) as a naive UTC datetime."""
    if not isinstance(t, datetime):
        return datetime(t.year, t.month, t.day)
    return t.astimezone(timezone.utc).replace(tzinfo=None) if t.tzinfo else t


def _move(t: datetime, offset: str, sign: int) -> datetime:
    if not offset.strip():
        return t
    d = parse(offset)
    return t + d if sign > 0 else t - d


@dataclass(frozen=True)
class _Frame:
    ds: DatasetSpec
    df: pl.DataFrame                  # rows in the order they became known
    known: pl.Series | None           # None: a static table
    until: pl.Series | None = None    # vintages: valid before this time


def _annotate(df: pl.DataFrame, ds: DatasetSpec) -> _Frame:
    r = rule(ds)
    if r is None:
        return _Frame(ds, df, None)
    cols = {"__known": known_at(df, ds)}
    if r.until:
        if not ds.time:
            raise StudyError(f"data.{ds.name}.vintages needs data.{ds.name}.time",
                             "a vintage is a published value of the observation at that time")
        end = _parse(df, r.until, ds)
        step = timedelta(days=1) if end.dtype == pl.Date else timedelta(microseconds=1)   # the end is inclusive
        cols["__until"] = (end.cast(pl.Datetime("us")) + step).fill_null(datetime.max)
    a = df.with_columns(**cols).drop_nulls("__known").sort("__known", maintain_order=True)
    return _Frame(ds, a.drop(*cols), a["__known"], a["__until"] if r.until else None)


class Timeline:
    """The datasets annotated once with when each row became known, shared by every View over them: a runner
    that builds thousands of views makes one Timeline and calls `at(t)`."""

    def __init__(self, frames: dict, specs: dict):
        missing = [n for n in frames if n not in specs]
        if missing:
            raise StudyError(f"no dataset spec for {', '.join(missing)}", "a view needs each dataset's known_at rule")
        self.specs = specs
        self.frames = {n: _annotate(df, specs[n]) for n, df in frames.items()}

    def at(self, t) -> View:
        return View(self, self.specs, t)


class View:
    """What was known at `t`. `frames` is {name: frame}, or a Timeline (then `specs` is not used)."""

    def __init__(self, frames, specs: dict, t):
        self._timeline = frames if isinstance(frames, Timeline) else Timeline(frames, specs)
        self.t = _instant(t)

    def _get(self, name: str) -> _Frame:
        try:
            return self._timeline.frames[name]
        except KeyError:
            raise StudyError(f"no dataset `{name}` in this view",
                             f"datasets: {', '.join(self._timeline.frames) or 'none'}") from None

    def frame(self, name: str) -> pl.DataFrame:
        """The rows of `name` known at t, in the order they became known; for vintages, per key and time, the
        value as published at t. A static table is returned whole."""
        f = self._get(name)
        if f.known is None:
            return f.df
        n = f.known.search_sorted(self.t, side="right")
        df = f.df.head(n)
        if f.until is None:
            return df
        out = (df.filter(f.until.head(n) > self.t)
               .unique(subset=[*f.ds.key, f.ds.time], keep="last", maintain_order=True))
        end = f.ds.vintages[1]                       # when the next revision came is not known at t: hide it
        return out.with_columns(pl.lit(None, dtype=out.schema[end]).alias(end)) if end in out.columns else out

    def series(self, name: str, value: str, **where) -> Series:
        """Column `value` of `name` over its time column, as known at t; `where` keeps the rows whose columns
        equal the values given (e.g. `series="DGS2"`)."""
        ds = self._get(name).ds
        if not ds.time:
            raise StudyError(f"data.{name} has no time column", "a series is a value over time",
                             f"set data.{name}.time")
        df = self.frame(name)
        gone = [c for c in (value, *where) if c not in df.columns]
        if gone:
            raise StudyError(f"data.{name} has no column {', '.join(gone)}", f"its columns: {', '.join(df.columns)}")
        df = df.filter(**where) if where else df
        return Series(pl.DataFrame({"time": _when(df, ds.time, ds), "value": df[value].cast(pl.Float64)})
                      .fill_nan(None).drop_nulls().sort("time"))


class Series:
    """One value over time as known in a view: (time, value), sorted by time, with missing values dropped."""

    def __init__(self, values: pl.DataFrame):
        self._v = values

    def values(self) -> pl.DataFrame:
        return self._v

    def at(self, t, back: str = "", ahead: str = "") -> float:
        """The last value at or before t moved back by `back` or ahead by `ahead` ("3M", "10d", "1d12h": M calendar
        months, d days, h hours, min minutes); NaN if there is none."""
        when = _move(_move(_instant(t), back, -1), ahead, 1)
        i = self._v["time"].search_sorted(when, side="right")
        return float(self._v["value"][i - 1]) if i else math.nan
