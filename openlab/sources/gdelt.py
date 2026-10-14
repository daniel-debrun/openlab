"""GDELT DOC 2.0 (api.gdeltproject.org): how much of the world's monitored online news mentions a query, and how
positively, over time. No key; GDELT allows one request every 5 seconds per address."""
from __future__ import annotations

import urllib.error
import urllib.parse

import polars as pl

from . import http

BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
http.LIMITS.setdefault("api.gdeltproject.org", http.Limiter(1 / 6, name="api.gdeltproject.org"))


def _series(query: str, mode: str, window: dict, max_age_days: float) -> tuple:
    url = f"{BASE}?{urllib.parse.urlencode({'query': query, 'mode': mode, 'format': 'json', **window})}"
    try:
        d = http.get_json(url, max_age_days=max_age_days, tries=1)     # quick retries only extend a GDELT block
    except (ValueError, urllib.error.HTTPError) as e:   # throttle or bad query: plain text (HTTP 200) or 429
        if isinstance(e, urllib.error.HTTPError) and e.code != 429:
            raise
        raise RuntimeError("GDELT refused the request (one request per 5 s per address, or a malformed query); "
                           "wait and retry") from None
    return d.get("query_details", {}).get("date_resolution"), (d.get("timeline") or [{"data": []}])[0]["data"]


def timeline(query: str, start: str = "", end: str = "", timespan: str = "3m") -> pl.DataFrame:
    """Coverage of `query` (GDELT syntax: "exact phrase", OR, sourcecountry:, ...): date (or time, when GDELT
    serves hours), articles matching, all_articles monitored, share (articles / all_articles), tone (average,
    about -10 negative .. +10 positive). Between `start` and `end` (YYYY-MM-DD, from 2017) or the last `timespan`."""
    window = {"timespan": timespan}
    if start:
        window = {"startdatetime": start.replace("-", "") + "000000"}
        if end:
            window["enddatetime"] = end.replace("-", "") + "235959"
    age = 30.0 if end else 0.1
    res, vol = _series(query, "timelinevolraw", window, age)
    _, tone = _series(query, "timelinetone", window, age)
    if not vol:
        return pl.DataFrame()
    v = pl.DataFrame([{"t": x["date"], "articles": x["value"], "all_articles": x["norm"]} for x in vol])
    t = pl.DataFrame([{"t": x["date"], "tone": x["value"]} for x in tone], schema={"t": pl.Utf8, "tone": pl.Float64})
    d = v.join(t, on="t", how="left").with_columns(
        share=pl.col("articles") / pl.col("all_articles"),
        t=pl.col("t").str.to_datetime("%Y%m%dT%H%M%SZ", time_zone="UTC"))
    d = d.rename({"t": "time"}) if res in ("hour", "15min") else d.with_columns(pl.col("t").dt.date()).rename({"t": "date"})
    return d.select(d.columns[0], "articles", "all_articles", "share", "tone")
