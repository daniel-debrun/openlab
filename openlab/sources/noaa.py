"""NOAA NCEI climate data: GHCN daily summaries for a station from the Access Data Service (no token, wide rows,
multi-year spans in one request), and station search from the Climate Data Online v2 API, which needs a token
(`NOAA_TOKEN`; the misspelt `NOAA_TOEKN` is also read) sent as a header.

The daily data come from the Access Data Service rather than CDO v2 because CDO caps a request at 1000 long
rows and one year: a decade of one station is ~40 requests there and one here. Station search stays on CDO
because the Access search service has no station listing by place. The token never enters a URL, so the cache
(by URL) never holds it."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import polars as pl

from . import http

DAILY = "https://www.ncei.noaa.gov/access/services/data/v1"
CDO = "https://www.ncei.noaa.gov/cdo-web/api/v2"
_LIMIT = http.Limiter(4)                    # CDO allows 5 requests a second and 10,000 a day per token


def _token() -> str:
    t = os.environ.get("NOAA_TOKEN") or os.environ.get("NOAA_TOEKN", "")
    if not t:
        raise RuntimeError("NOAA_TOKEN is not set: request a free token at ncdc.noaa.gov/cdo-web/token "
                           "and add NOAA_TOKEN=... to ~/.config/openlab/research.env")
    return t


def _cdo(path: str, params: dict, max_age_days: float = 7.0) -> dict:
    url = f"{CDO}/{path}?{urllib.parse.urlencode(params)}"
    cached = http._key(url)
    if cached.exists() and (time.time() - cached.stat().st_mtime) / 86400 <= max_age_days:
        return json.loads(cached.read_text())["body"]
    _LIMIT.wait()
    req = urllib.request.Request(url, headers={"token": _token(), "User-Agent": "openlab-research/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"NOAA CDO {path}: HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from None
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps({"url": url, "fetched": time.strftime("%Y-%m-%dT%H:%M:%S"), "body": body}))
    return body


def daily(station: str, start: str, end: str, datatypes: str = "", units: str = "metric") -> pl.DataFrame:
    """GHCN daily summaries for one station (e.g. "USW00094728", Central Park; a "GHCND:" prefix is dropped):
    date, station, then one float column per element: TMAX, TMIN, TAVG (deg C), PRCP, SNOW, SNWD (mm), AWND
    (m/s), WT** weather-type flags (1 = occurred), ... `datatypes` restricts them (comma separated);
    `units` metric or standard."""
    p = {"dataset": "daily-summaries", "stations": station.removeprefix("GHCND:"), "startDate": start,
         "endDate": end, "format": "json", "units": units}
    if datatypes:
        p["dataTypes"] = datatypes
    rows = http.get_json(f"{DAILY}?{urllib.parse.urlencode(p)}", max_age_days=1.0, timeout=120)
    if not rows:
        return pl.DataFrame()
    d = pl.DataFrame(rows, infer_schema_length=None)
    vals = [c for c in d.columns if c not in ("STATION", "DATE")]
    return d.select(pl.col("DATE").str.to_date().alias("date"), pl.col("STATION").alias("station"),
                    *[pl.col(c).str.strip_chars().cast(pl.Float64, strict=False) for c in vals]).sort("date")


def stations(extent: tuple = (), location: str = "", name: str = "", dataset: str = "GHCND") -> pl.DataFrame:
    """Stations holding `dataset` inside `extent` (lat_min, lon_min, lat_max, lon_max) or a CDO `location`
    (e.g. "FIPS:36" New York State, "CITY:US360019" New York City, "ZIP:10024"), optionally filtered to names
    containing `name`: id, name, latitude, longitude, elevation (m), mindate, maxdate, datacoverage (0..1)."""
    p = {"datasetid": dataset, "limit": 1000}
    if extent:
        p["extent"] = ",".join(str(x) for x in extent)
    if location:
        p["locationid"] = location
    rows, total = [], None
    while total is None or len(rows) < total:
        b = _cdo("stations", {**p, "offset": len(rows) + 1})
        if not b.get("results"):
            break
        total = b["metadata"]["resultset"]["count"]
        rows += b["results"]
    if not rows:
        return pl.DataFrame()
    d = pl.DataFrame(rows, infer_schema_length=None).select(
        "id", "name", "latitude", "longitude", pl.col("elevation").cast(pl.Float64, strict=False),
        pl.col("mindate", "maxdate").str.to_date(), "datacoverage")
    if name:
        d = d.filter(pl.col("name").str.to_uppercase().str.contains(name.upper(), literal=True))
    return d.sort("id")
