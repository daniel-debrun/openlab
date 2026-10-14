"""USDA National Agricultural Statistics Service Quick Stats (quickstats.nass.usda.gov): crop and livestock
production, acreage, yields, prices, stocks and condition by year and geography, behind `USDA_NASS_KEY`.

Responses are cached by URL without the key (so the key is never written to disk)."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse

import polars as pl

from . import http

BASE = "https://quickstats.nass.usda.gov/api/api_GET/"
_LIMIT = http.Limiter(2)
COLUMNS = {"year": "year", "reference_period_desc": "period", "week_ending": "week_ending", "freq_desc": "freq",
           "agg_level_desc": "geography", "state_alpha": "state", "county_name": "county",
           "location_desc": "location", "commodity_desc": "commodity", "short_desc": "description",
           "statisticcat_desc": "statistic", "unit_desc": "unit", "domain_desc": "domain",
           "domaincat_desc": "domain_category", "source_desc": "source", "Value": "value", "CV (%)": "cv_pct",
           "load_time": "load_time"}


def _key() -> str:
    k = os.environ.get("USDA_NASS_KEY", "")
    if not k:
        raise RuntimeError("USDA_NASS_KEY is not set: get a free key at quickstats.nass.usda.gov/api "
                           "and add USDA_NASS_KEY=... to ~/.config/openlab/research.env")
    return k


def _get(params: dict, max_age_days: float = 7.0) -> list:
    public = f"{BASE}?{urllib.parse.urlencode(params)}"
    cached = http._key(public)
    if cached.exists() and (time.time() - cached.stat().st_mtime) / 86400 <= max_age_days:
        return json.loads(cached.read_text())["body"]
    _LIMIT.wait()
    key = _key()
    try:
        body = json.loads(http.get_bytes(f"{public}&key={key}", cache=False, timeout=120))
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "replace").replace(key, "<key>")[:300]
        raise RuntimeError(f"NASS Quick Stats: HTTP {e.code}: {msg}") from None
    body = body.get("data", [])
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps({"url": public, "fetched": time.strftime("%Y-%m-%dT%H:%M:%S"), "body": body}))
    return body


def _number(c: str) -> pl.Expr:
    # "(D)" withheld to avoid disclosing a single operation, "(Z)" under half a unit, "(NA)", "(X)", "(S)": null
    return pl.col(c).str.replace_all(",", "").str.strip_chars().cast(pl.Float64, strict=False)


def quickstats(commodity: str, statistic: str = "", year=None, geography: str = "NATIONAL", state: str = "",
               **filters) -> pl.DataFrame:
    """A Quick Stats query: year, period (YEAR, a month, a week, or a forecast such as "YEAR - AUG FORECAST"),
    week_ending, freq, geography, state, county, location, commodity, description, statistic, unit, domain,
    domain_category, source (SURVEY or CENSUS), value, cv_pct, load_time. `commodity` e.g. CORN, `statistic`
    e.g. PRODUCTION, YIELD, AREA PLANTED, PRICE RECEIVED; `geography` NATIONAL, STATE or COUNTY; `state` a
    two-letter code. Any other Quick Stats field goes in `filters` (e.g. unit_desc="BU", year__GE=2015)."""
    p = {"commodity_desc": commodity.upper(), "agg_level_desc": geography.upper(), **filters, "format": "JSON"}
    if statistic:
        p["statisticcat_desc"] = statistic.upper()
    if year is not None:
        p["year"] = year
    if state:
        p["state_alpha"] = state.upper()
    rows = _get(p)
    if not rows:
        return pl.DataFrame()
    d = pl.DataFrame(rows, infer_schema_length=None).select(list(COLUMNS)).rename(COLUMNS)
    return d.with_columns(_number("value"), _number("cv_pct"), pl.col("week_ending").str.to_date("%Y-%m-%d", strict=False),
                          pl.col("load_time").str.to_datetime("%Y-%m-%d %H:%M:%S%.f", strict=False)
                          ).sort("year", "location", "period")
