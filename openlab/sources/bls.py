"""Bureau of Labor Statistics (api.bls.gov, v2): any BLS time series by id, behind one key (`BLS_API_KEY`, in
the environment or ~/.config/openlab/research.env).

Common series ids:
    CUUR0000SA0    CPI-U, all items, US city average, not seasonally adjusted (CUSR0000SA0 seasonally adjusted)
    CUUR0000SA0L1E CPI-U less food and energy (core), NSA
    LNS14000000    unemployment rate, seasonally adjusted
    CES0000000001  total nonfarm payrolls, thousands, seasonally adjusted
    CES0500000003  average hourly earnings, total private, seasonally adjusted
    JTS000000000000000JOL  JOLTS job openings, seasonally adjusted

Responses are cached by request without the key (so the key is never written to disk)."""
from __future__ import annotations

import json
import os
import time
import urllib.request

import polars as pl

from . import http

URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
_LIMIT = http.Limiter(2)                    # keyed: 50 requests per 10 seconds, 500 a day


def _key() -> str:
    k = os.environ.get("BLS_API_KEY", "")
    if not k:
        raise RuntimeError("BLS_API_KEY is not set: register free at data.bls.gov/registrationEngine and add "
                           "BLS_API_KEY=... to ~/.config/openlab/research.env")
    return k


def _post(ids: list, start: int, end: int, max_age_days: float) -> list:
    req = {"seriesid": ids, "startyear": str(start), "endyear": str(end)}
    public = f"{URL}?{json.dumps(req, sort_keys=True)}"
    cached = http._key(public)
    if cached.exists() and (time.time() - cached.stat().st_mtime) / 86400 <= max_age_days:
        return json.loads(cached.read_text())["body"]
    _LIMIT.wait()
    data = json.dumps(req | {"registrationkey": _key()}).encode()
    r = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json",
                                                        "User-Agent": "openlab-research/0.1"})
    with urllib.request.urlopen(r, timeout=60) as resp:
        b = json.loads(resp.read())
    if b.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS: {b.get('status')} {'; '.join(b.get('message', []))[:300]}")
    body = b["Results"]["series"]
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps({"url": public, "fetched": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                  "messages": b.get("message", []), "body": body}))
    return body


def _month(period: str) -> int:
    kind, n = period[0], int(period[1:])
    return {"M": n, "Q": 3 * n - 2, "S": 6 * n - 5}.get(kind, 1)


def series(series_ids: str | list, start_year: int, end_year: int) -> pl.DataFrame:
    """Observations: series_id, date (first day of the period), period (M01..M12 monthly, Q01..Q04 quarterly,
    S01/S02 half-yearly, A01 annual), value, preliminary (footnote P), for one id or a list, `start_year` to
    `end_year`. Split into calls of 50 series and 20 years, the API's maximum."""
    ids = [series_ids] if isinstance(series_ids, str) else list(series_ids)
    rows = []
    for i in range(0, len(ids), 50):
        for y in range(int(start_year), int(end_year) + 1, 20):
            for s in _post(ids[i:i + 50], y, min(y + 19, int(end_year)), max_age_days=1.0):
                for o in s["data"]:
                    if o["period"] == "M13":            # annual average, only sent when asked for
                        continue
                    rows.append({"series_id": s["seriesID"], "year": int(o["year"]), "period": o["period"],
                                 "value": o["value"],
                                 "preliminary": any(f.get("code") == "P" for f in o.get("footnotes") or [])})
    if not rows:
        return pl.DataFrame(schema={"series_id": pl.Utf8, "date": pl.Date, "period": pl.Utf8, "value": pl.Float64,
                                    "preliminary": pl.Boolean})
    d = pl.DataFrame(rows)
    return (d.with_columns(date=pl.date(pl.col("year"), pl.col("period").map_elements(_month, return_dtype=pl.Int32), 1),
                           value=pl.col("value").cast(pl.Float64, strict=False))
            .select("series_id", "date", "period", "value", "preliminary").sort("series_id", "date"))
