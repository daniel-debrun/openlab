"""FRED's release calendar: when each statistical release lands (past and scheduled), and which series a
release carries. Needs `FRED_API_KEY` (in the environment or ~/.config/openlab/research.env).

Responses are cached by URL without the key, so the key is never written to disk. Scheduled (future) dates
are FRED's published calendar and can move; a past date is when the release actually came out."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse

import polars as pl

from . import http

BASE = "https://api.stlouisfed.org/fred"
_LIMIT = http.Limiter(2)                    # FRED allows 120 requests a minute per key


def _key() -> str:
    k = os.environ.get("FRED_API_KEY", "")
    if not k:
        raise RuntimeError("FRED_API_KEY is not set: request a free key at https://fredaccount.stlouisfed.org/apikey "
                           "and add FRED_API_KEY=... to ~/.config/openlab/research.env")
    return k


def _get(path: str, max_age_days: float = 1.0, **params) -> dict:
    public = f"{BASE}/{path}?{urllib.parse.urlencode({'file_type': 'json', **params})}"
    cached = http._key(public)
    if cached.exists() and (time.time() - cached.stat().st_mtime) / 86400 <= max_age_days:
        return json.loads(cached.read_text())["body"]
    _LIMIT.wait()
    try:
        body = json.loads(http.get_bytes(f"{public}&api_key={_key()}", cache=False, timeout=60))
    except urllib.error.HTTPError as e:
        msg = json.loads(e.read() or b"{}").get("error_message", "")
        raise RuntimeError(f"FRED {path}: HTTP {e.code} {msg}") from None
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps({"url": public, "fetched": time.strftime("%Y-%m-%dT%H:%M:%S"), "body": body}))
    return body


def _pages(path: str, field: str, **params) -> list:
    rows, offset = [], 0
    while True:
        b = _get(path, limit=1000, offset=offset, **params)
        rows += b.get(field, [])
        offset += 1000
        if offset >= b.get("count", 0):
            return rows


def release_dates(start: str = "", end: str = "", release_id: int | None = None) -> pl.DataFrame:
    """Release dates: date, release_id, release_name, for every release or one `release_id`, between `start` and
    `end` (YYYY-MM-DD; default today to 90 days ahead for all releases, the whole history for one). Future dates
    are the published schedule. Common ids: 10 CPI, 50 Employment Situation, 53 GDP, 54 Personal Income and
    Outlays, 101 FOMC press release (see `releases` on fred.stlouisfed.org)."""
    p = {"include_release_dates_with_no_data": "true", "sort_order": "asc"}
    if release_id is None:
        today = time.strftime("%Y-%m-%d")
        p |= {"realtime_start": start or today,
              "realtime_end": end or time.strftime("%Y-%m-%d", time.localtime(time.time() + 90 * 86400))}
        rows = []
        while True:                         # FRED hangs on any offset here: page by date window instead
            b = _get("releases/dates", limit=1000, **p)
            got = b.get("release_dates", [])
            if len(got) == b.get("count", 0):
                rows += got
                break
            last = got[-1]["date"]
            rows += [r for r in got if r["date"] < last]
            p["realtime_start"] = last
    else:
        p |= {"release_id": int(release_id), "realtime_start": start or "1776-07-04", "realtime_end": end or "9999-12-31"}
        rows = _pages("release/dates", "release_dates", **p)
        name = _get("release", max_age_days=30, release_id=int(release_id))["releases"][0]["name"]
        rows = [r | {"release_name": name} for r in rows]
    if not rows:
        return pl.DataFrame(schema={"date": pl.Date, "release_id": pl.Int64, "release_name": pl.Utf8})
    return (pl.DataFrame(rows).with_columns(pl.col("date").str.to_date())
            .select("date", "release_id", "release_name").sort("date", "release_id"))


def release_series(release_id: int) -> pl.DataFrame:
    """The series in one release: id, title, frequency, units, seasonal_adjustment, observation_start,
    observation_end (dates), last_updated, popularity."""
    rows = _pages("release/series", "seriess", release_id=int(release_id))
    cols = ["id", "title", "frequency", "units", "seasonal_adjustment", "observation_start", "observation_end",
            "last_updated", "popularity"]
    return (pl.DataFrame(rows).select(cols)
            .with_columns(pl.col("observation_start", "observation_end").str.to_date()).sort("id"))
