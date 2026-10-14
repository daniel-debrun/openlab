"""FRED, the IMF and the World Bank: three macro sources behind one small surface.

They differ in exactly the way that matters for a backtest, so the difference is made explicit rather than
smoothed over:

* **FRED** has *vintages*. `observations(..., realtime_start=...)` returns the numbers as they stood on a
  date, and `release_dates()` says when each print landed. GDP for 2020Q1 was not knowable in April 2020 at
  the value it now shows. A macro signal built from final values is a signal built from the future, and FRED
  is the only one of the three that lets you avoid it. Needs a free key in `FRED_API_KEY`.
* **World Bank** is annual, wide country coverage, revised silently. Fine for cross-country structure, never
  for timing.
* **IMF** (DataMapper) is annual with forecasts included, which means a series can contain values for years
  that have not happened. `imf_series()` returns the year with each value so a forecast cannot be mistaken
  for an observation.
"""
from __future__ import annotations

import os

import polars as pl

from .http import get_json

FRED = "https://api.stlouisfed.org/fred"
WB = "https://api.worldbank.org/v2"
IMF_DM = "https://www.imf.org/external/datamapper/api/v1"


class NeedsKey(RuntimeError):
    pass


def fred_key() -> str:
    k = os.environ.get("FRED_API_KEY")
    if not k:
        raise NeedsKey("FRED needs a free API key: request one at "
                       "https://fredaccount.stlouisfed.org/apikey and export FRED_API_KEY=<key>")
    return k


def fred_series(series_id: str) -> dict:
    body = get_json(f"{FRED}/series?series_id={series_id}&file_type=json&api_key={fred_key()}",
                    max_age_days=30)
    return (body.get("seriess") or [{}])[0]


def fred_observations(series_id: str, start: str = "1900-01-01", end: str = "2100-01-01",
                      realtime_start: str = "", realtime_end: str = "") -> pl.DataFrame:
    """Observations, optionally *as they stood* on a real-time date.

    Leave the real-time arguments empty for the current (revised) series. Set both to a date to get the
    vintage a backtest standing on that date would have read."""
    url = (f"{FRED}/series/observations?series_id={series_id}&file_type=json&api_key={fred_key()}"
           f"&observation_start={start}&observation_end={end}")
    if realtime_start:
        url += f"&realtime_start={realtime_start}&realtime_end={realtime_end or realtime_start}"
    obs = []
    while True:                                      # FRED caps a reply at 100,000 rows: page by offset
        body = get_json(f"{url}&limit=100000&offset={len(obs)}", max_age_days=7)
        page = body.get("observations", [])
        obs += page
        if not page or len(obs) >= int(body.get("count", 0)):
            break
    if not obs:
        return pl.DataFrame()
    return (pl.DataFrame(obs)
            .with_columns(date=pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False),
                          value=pl.col("value").replace(".", None).cast(pl.Float64, strict=False),
                          series=pl.lit(series_id))
            .select("series", "date", "value", "realtime_start", "realtime_end"))


def fred_release_dates(series_id: str) -> pl.DataFrame:
    """When each vintage of this series was published -- the publication lag, measured rather than assumed."""
    s = fred_series(series_id)
    rid = s.get("id") and get_json(
        f"{FRED}/series/release?series_id={series_id}&file_type=json&api_key={fred_key()}",
        max_age_days=30).get("releases", [{}])[0].get("id")
    if not rid:
        return pl.DataFrame()
    body = get_json(f"{FRED}/release/dates?release_id={rid}&file_type=json&api_key={fred_key()}"
                    f"&include_release_dates_with_no_data=true", max_age_days=7)
    d = body.get("release_dates", [])
    return pl.DataFrame(d) if d else pl.DataFrame()


def fred_search(text: str, limit: int = 20) -> pl.DataFrame:
    body = get_json(f"{FRED}/series/search?search_text={text.replace(' ', '+')}&file_type=json"
                    f"&limit={limit}&api_key={fred_key()}", max_age_days=7)
    s = body.get("seriess", [])
    if not s:
        return pl.DataFrame()
    return pl.DataFrame(s).select([c for c in ("id", "title", "frequency", "units", "observation_start",
                                               "observation_end", "popularity") if c in pl.DataFrame(s).columns])


def worldbank(indicator: str, countries: str = "all", start: int = 1960, end: int = 2030) -> pl.DataFrame:
    """A World Bank indicator, paged to completion. `indicator` e.g. `NY.GDP.MKTP.CD`."""
    rows, page = [], 1
    while True:
        body = get_json(f"{WB}/country/{countries}/indicator/{indicator}"
                        f"?format=json&per_page=1000&page={page}&date={start}:{end}", max_age_days=30)
        if not isinstance(body, list) or len(body) < 2 or not body[1]:
            break
        rows += body[1]
        meta = body[0]
        if page >= int(meta.get("pages", 1)):
            break
        page += 1
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame([{"country": r["country"]["value"], "iso3": r.get("countryiso3code"),
                          "year": int(r["date"]), "value": r["value"], "indicator": indicator}
                         for r in rows])


def worldbank_indicators(search: str = "") -> pl.DataFrame:
    """World Bank indicator ids and names, optionally filtered by a case-insensitive search."""
    body = get_json(f"{WB}/indicator?format=json&per_page=1000", max_age_days=30)
    rows = body[1] if isinstance(body, list) and len(body) > 1 else []
    df = pl.DataFrame([{"id": r["id"], "name": r["name"]} for r in rows]) if rows else pl.DataFrame()
    return df.filter(pl.col("name").str.contains(f"(?i){search}")) if (search and df.height) else df


def imf_series(indicator: str, countries: str = "") -> pl.DataFrame:
    """An IMF DataMapper indicator (e.g. `NGDP_RPCH`, real GDP growth). Later years may be **forecasts**."""
    body = get_json(f"{IMF_DM}/{indicator}", max_age_days=30)
    vals = (body.get("values") or {}).get(indicator, {})
    want = {c.strip().upper() for c in countries.replace(";", ",").split(",") if c.strip()}
    rows = [{"iso3": iso, "year": int(y), "value": v, "indicator": indicator}
            for iso, series in vals.items() if not want or iso.upper() in want
            for y, v in series.items() if v is not None]
    # the DataMapper path ignores a country segment, so the filter is applied here rather than pretended at
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def imf_indicators() -> pl.DataFrame:
    """IMF DataMapper indicator ids, labels and units."""
    body = get_json(f"{IMF_DM}/indicators", max_age_days=30)
    ind = body.get("indicators", {})
    return pl.DataFrame([{"id": k, "label": v.get("label"), "unit": v.get("unit")}
                         for k, v in ind.items()]) if ind else pl.DataFrame()
