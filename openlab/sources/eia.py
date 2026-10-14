"""U.S. Energy Information Administration API v2 (api.eia.gov): any series by route and facets, paged past the
5000-row limit, plus weekly crude and gasoline stocks and spot prices (WTI, Brent, Henry Hub), behind `EIA_API_KEY`.

Responses are cached by URL without the key (so the key is never written to disk)."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse

import polars as pl

from . import http

BASE = "https://api.eia.gov/v2"
PAGE = 5000
_LIMIT = http.Limiter(5)


def _key() -> str:
    k = os.environ.get("EIA_API_KEY", "")
    if not k:
        raise RuntimeError("EIA_API_KEY is not set: get a free key at eia.gov/opendata/register.php "
                           "and add EIA_API_KEY=... to ~/.config/openlab/research.env")
    return k


def _get(route: str, params: list, max_age_days: float = 1.0) -> dict:
    public = f"{BASE}/{route.strip('/')}/data/?{urllib.parse.urlencode(params)}"
    cached = http._key(public)
    if cached.exists() and (time.time() - cached.stat().st_mtime) / 86400 <= max_age_days:
        return json.loads(cached.read_text())["body"]["response"]
    _LIMIT.wait()
    key = _key()
    try:
        body = json.loads(http.get_bytes(f"{public}&api_key={key}", cache=False, timeout=60))
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "replace").replace(key, "<key>")[:300]
        raise RuntimeError(f"EIA {route}: HTTP {e.code}: {msg}") from None
    if "error" in body:
        raise RuntimeError(f"EIA {route}: {str(body['error']).replace(key, '<key>')[:300]}")
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps({"url": public, "fetched": time.strftime("%Y-%m-%dT%H:%M:%S"), "body": body}))
    return body["response"]


def series(route: str, frequency: str = "monthly", facets: dict | None = None, start: str = "", end: str = "",
           data: str = "value") -> pl.DataFrame:
    """Any API v2 data route (e.g. "petroleum/pri/spt", "electricity/retail-sales"): date (a Datetime for hourly
    frequencies), the route's facet and description columns, the `data` column as a float, and units.
    `facets` maps a facet to a value or list (e.g. {"series": ["RWTC"]}); `start`/`end` use the frequency's
    format (YYYY-MM-DD, YYYY-MM, YYYY)."""
    base = [("frequency", frequency), ("data[0]", data), ("sort[0][column]", "period"),
            ("sort[0][direction]", "asc"), ("length", PAGE)]
    for f, v in (facets or {}).items():
        base += [(f"facets[{f}][]", x) for x in ([v] if isinstance(v, str) else v)]
    base += [(k, v) for k, v in (("start", start), ("end", end)) if v]
    rows, total = [], None
    while total is None or len(rows) < total:
        r = _get(route, base + [("offset", len(rows))])
        total = int(r.get("total", 0))
        if not r["data"]:
            break
        rows += r["data"]
    if not rows:
        return pl.DataFrame()
    p = pl.col("period")
    date = (p.str.slice(0, 13).str.to_datetime("%Y-%m-%dT%H") if frequency.startswith(("hourly", "local-hourly"))
            else pl.when(p.str.contains("Q")).then(p.str.slice(0, 4) + "-" + (p.str.slice(-1).cast(pl.Int32) * 3 - 2)
                                                   .cast(pl.Utf8).str.zfill(2) + "-01")
            .when(p.str.len_chars() == 4).then(p + "-01-01")
            .when(p.str.len_chars() == 7).then(p + "-01").otherwise(p).str.to_date("%Y-%m-%d"))
    d = pl.DataFrame(rows, infer_schema_length=None)
    return (d.with_columns(date.alias("date"), pl.col(data).cast(pl.Float64, strict=False)).drop("period")
            .select("date", pl.exclude("date")).sort("date"))


SPOT = {"WTI": ("petroleum/pri/spt", "RWTC"), "BRENT": ("petroleum/pri/spt", "RBRTE"),
        "HENRY_HUB": ("natural-gas/pri/fut", "RNGWHHD")}


def spot(names=("WTI", "BRENT", "HENRY_HUB"), start: str = "2020-01-01") -> pl.DataFrame:
    """Daily spot prices: date, name (WTI Cushing and Brent in $/barrel, HENRY_HUB natural gas in $/MMBtu),
    value, units."""
    out = [series(SPOT[n][0], "daily", {"series": SPOT[n][1]}, start=start)
           .select("date", name=pl.lit(n), value="value", units="units") for n in names]
    return pl.concat(out).sort("name", "date")


STOCKS = {"WCESTUS1": "crude", "WGTSTUS1": "gasoline"}


def weekly_stocks(start: str = "2020-01-01") -> pl.DataFrame:
    """U.S. weekly ending stocks (the Wednesday petroleum status report): date (week ending Friday), name
    (crude excluding the SPR, total gasoline), value in thousand barrels, units."""
    d = series("petroleum/stoc/wstk", "weekly", {"series": list(STOCKS)}, start=start)
    return d.select("date", name=pl.col("series").replace_strict(STOCKS), value="value", units="units").sort("name", "date")
