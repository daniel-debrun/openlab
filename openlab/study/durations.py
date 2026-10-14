"""Durations in study.toml: "18h", "1d", "2w", "3M" (calendar months), "1Y", "30min", or signed sums of terms
such as "- 1d + 18h". They add to and subtract from datetimes; a month-end stays a month-end (Nov 30 + 1M =
Dec 31), so month-end decisions line up. This is the one duration parser of openlab.study."""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import timedelta

from .errors import StudyError

_UNITS = {"min": timedelta(minutes=1), "h": timedelta(hours=1), "d": timedelta(days=1), "w": timedelta(weeks=1),
          "M": 1, "Y": 12}
_TERM = r"\s*([+-]?)\s*(\d+)\s*(min|[hdwMY])"


def _add_months(t, n: int):
    if not n:
        return t
    y, m = divmod(t.month - 1 + n, 12)
    y, m = t.year + y, m + 1
    last = calendar.monthrange(y, m)[1]
    at_end = t.day == calendar.monthrange(t.year, t.month)[1]
    return t.replace(year=y, month=m, day=last if at_end else min(t.day, last))


@dataclass(frozen=True)
class Duration:
    months: int = 0
    delta: timedelta = timedelta(0)

    def __radd__(self, t):
        return _add_months(t, self.months) + self.delta

    def __rsub__(self, t):
        return _add_months(t, -self.months) - self.delta

    def on(self, expr):
        """Shift a polars datetime expression by this duration (months element-wise, the rest vectorised)."""
        if self.months:
            import polars as pl
            expr = expr.map_elements(lambda t: _add_months(t, self.months), return_dtype=pl.Datetime("us"))
        return expr + self.delta if self.delta else expr


def parse(text: str, signed: bool = False) -> Duration:
    """One term ("3M"), or with `signed` a sum of signed terms ("- 1d + 18h")."""
    text = str(text)
    terms = re.findall(_TERM, text) if re.fullmatch(rf"(?:{_TERM})+\s*", text) else []
    if not terms or (not signed and (len(terms) > 1 or terms[0][0])):
        raise StudyError(f"`{text}` is not a duration", "a whole number and a unit: min, h, d, w, M (calendar "
                         "months) or Y" + (", in signed terms such as - 1d + 18h" if signed else ""),
                         'e.g. "1d", "18h", "3M"')
    months, delta = 0, timedelta(0)
    for sign, n, unit in terms:
        k, u = (-1 if sign == "-" else 1) * int(n), _UNITS[unit]
        months, delta = (months + k * u, delta) if isinstance(u, int) else (months, delta + k * u)
    return Duration(months, delta)


def split_tail(text: str) -> tuple[str, Duration]:
    """`date - 1d + 18h` -> ("date", Duration); a bare name -> (name, zero)."""
    m = re.fullmatch(rf"(.+?)((?:{_TERM})+)\s*", str(text))
    if not m or not re.match(r"\s*[+-]", m[2]):
        return str(text).strip(), Duration()
    return m[1].strip(), parse(m[2], signed=True)
