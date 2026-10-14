"""US Census Bureau API (api.census.gov): the economic indicator time series (`timeseries/eits/*`: retail sales,
housing starts, durable goods, trade, ...) and a generic query for any Census dataset, behind `CENSUS_API_KEY`.

Responses are cached by URL without the key (so the key is never written to disk)."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse

import polars as pl

from . import http

BASE = "https://api.census.gov/data"
_LIMIT = http.Limiter(5)


def _key() -> str:
    k = os.environ.get("CENSUS_API_KEY", "")
    if not k:
        raise RuntimeError("CENSUS_API_KEY is not set: get a free key at api.census.gov/data/key_signup.html "
                           "and add CENSUS_API_KEY=... to ~/.config/openlab/research.env")
    return k


def _get(dataset: str, params: dict, max_age_days: float = 1.0) -> list:
    """One call: the rows as lists, header first ([] when nothing matches)."""
    public = f"{BASE}/{dataset}?{urllib.parse.urlencode(params)}"
    cached = http._key(public)
    if cached.exists() and (time.time() - cached.stat().st_mtime) / 86400 <= max_age_days:
        return json.loads(cached.read_text())["body"]
    _LIMIT.wait()
    key = _key()
    try:
        raw = http.get_bytes(f"{public}&key={key}", cache=False, timeout=60).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "replace").replace(key, "<key>")[:300]
        raise RuntimeError(f"Census {dataset}: HTTP {e.code}: {msg}") from None
    if raw.lstrip().startswith("<"):            # an invalid key is answered with an HTML page and status 200
        raise RuntimeError(f"Census {dataset}: refused (invalid key?)")
    body = json.loads(raw) if raw.strip() else []   # no matching rows: HTTP 204, empty body
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps({"url": public, "fetched": time.strftime("%Y-%m-%dT%H:%M:%S"), "body": body}))
    return body


def query(dataset: str, get, predicates: dict | None = None) -> pl.DataFrame:
    """Any Census API dataset (e.g. "timeseries/eits/marts", "2022/acs/acs5"): one string column per variable in
    `get` (a list or comma-separated string) and per predicate; cast them yourself."""
    get = get if isinstance(get, str) else ",".join(get)
    rows = _get(dataset, {"get": get, **(predicates or {})})
    if not rows:
        return pl.DataFrame()
    keep = [i for i, c in enumerate(rows[0]) if c not in rows[0][:i]]   # predicates are echoed as extra columns
    return pl.DataFrame([[r[i] for i in keep] for r in rows[1:]], schema=[rows[0][i] for i in keep], orient="row")


def indicator(program: str = "marts", start: str = "2015-01", seasonally_adj: bool = True, category: str = "",
              data_type: str = "") -> pl.DataFrame:
    """An economic indicator series, `program` one of the timeseries/eits datasets (marts retail sales, resconst
    housing starts, advm3 durable goods, ftd trade, m3, mtis, ressales, vip, qfr, ...): date, program,
    category_code, data_type_code, geo_level_code, seasonally_adj, value, is_error. `start` is YYYY-MM for
    monthly programs and YYYY for quarterly ones; `category` and `data_type` filter on their codes. Rows with is_error True are standard errors or
    other sampling measures, not levels. Quarterly programs are dated on the quarter's first day."""
    p = {"time": f"from {start}", "seasonally_adj": "yes" if seasonally_adj else "no"}
    if category:
        p["category_code"] = category
    if data_type:
        p["data_type_code"] = data_type
    if "default" not in http.get_json(f"{BASE}/timeseries/eits/{program}/geography.json"):
        p["for"] = "us:*"                       # advm3, m3 and a few others have no default geography
    d = query(f"timeseries/eits/{program}",
              "cell_value,data_type_code,category_code,time_slot_id,geo_level_code,error_data", p)
    if d.is_empty():
        return d
    t = pl.col("time")
    month = (pl.when(t.str.contains("Q")).then(t.str.slice(0, 4) + "-" +
                                               ((t.str.slice(-1).cast(pl.Int32) * 3 - 2).cast(pl.Utf8).str.zfill(2)))
             .when(t.str.len_chars() == 4).then(t + "-01").otherwise(t))
    return d.select(date=(month + "-01").str.to_date("%Y-%m-%d"), program=pl.lit(program),
                    category_code="category_code", data_type_code="data_type_code", geo_level_code="geo_level_code",
                    seasonally_adj=pl.col("seasonally_adj") == "yes",
                    value=pl.col("cell_value").cast(pl.Float64, strict=False),
                    is_error=pl.col("error_data") == "yes").sort("category_code", "data_type_code", "geo_level_code", "date")
