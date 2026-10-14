"""Bureau of Economic Analysis (apps.bea.gov/api): NIPA tables (GDP and its parts, income, prices, saving) and
the other BEA datasets that share the table/frequency/year shape, behind one key (`BEA_API_KEY`, in the
environment or ~/.config/openlab/research.env).

BEA echoes the key (UserID) back in every response: only the Results part is kept, and any error text is
scrubbed, so the key never reaches the cache or an error message. Cached by URL without the key."""
from __future__ import annotations

import json
import os
import time
import urllib.parse

import polars as pl

from . import http

BASE = "https://apps.bea.gov/api/data"
_LIMIT = http.Limiter(1)                    # 100 requests and 100 MB a minute, 30 errors a minute; over it, an hour's lockout


def _key() -> str:
    k = os.environ.get("BEA_API_KEY", "")
    if not k:
        raise RuntimeError("BEA_API_KEY is not set: register free at apps.bea.gov/api/signup (then activate it from "
                           "the e-mail) and add BEA_API_KEY=... to ~/.config/openlab/research.env")
    return k


def _get(max_age_days: float = 1.0, **params) -> dict:
    public = f"{BASE}?{urllib.parse.urlencode({'ResultFormat': 'JSON', **params})}"
    cached = http._key(public)
    if cached.exists() and (time.time() - cached.stat().st_mtime) / 86400 <= max_age_days:
        return json.loads(cached.read_text())["body"]
    _LIMIT.wait()
    k = _key()
    raw = http.get_bytes(f"{public}&UserID={k}", cache=False, timeout=120).decode("utf-8", "replace").replace(k, "")
    api = json.loads(raw)["BEAAPI"]
    res = api.get("Results") or {}
    err = api.get("Error") or (res.get("Error") if isinstance(res, dict) else None)
    if err:
        err = err[0] if isinstance(err, list) else err
        raise RuntimeError(f"BEA {params.get('method')}: {err.get('APIErrorDescription', err)}"[:300])
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps({"url": public, "fetched": time.strftime("%Y-%m-%dT%H:%M:%S"), "body": res}))
    return res


def table(table: str, frequency: str = "Q", years: str = "X", dataset: str = "NIPA") -> pl.DataFrame:
    """One BEA table, tidy: table, line, series_code, description, date (first day of the period), period
    ("2025", "2025Q1", "2025M01"), value, unit, unit_mult (value x 10^unit_mult). `frequency` A, Q or M (comma
    list allowed); `years` "2024,2025" or "X" (all; "LAST5" is refused for NIPA); `dataset` NIPA or NIUnderlyingDetail. Common
    tables: T10101 real GDP percent change, T10105 GDP in dollars, T20100 personal income, T20804 PCE price
    index percent change (parameters(dataset, "TableName") lists all)."""
    res = _get(method="GetData", DatasetName=dataset, TableName=table, Frequency=frequency, Year=years)
    rows = res.get("Data", [])
    if not rows:
        return pl.DataFrame()
    d = pl.DataFrame(rows, infer_schema_length=None)
    p = pl.col("TimePeriod")
    month = (pl.when(p.str.contains("Q")).then(p.str.slice(5).cast(pl.Int32, strict=False) * 3 - 2)
             .when(p.str.contains("M")).then(p.str.slice(5).cast(pl.Int32, strict=False)).otherwise(1))
    return (d.with_columns(date=pl.date(p.str.slice(0, 4).cast(pl.Int32), month, 1),
                           value=pl.col("DataValue").str.replace_all(",", "").cast(pl.Float64, strict=False),
                           line=pl.col("LineNumber").cast(pl.Int32), unit_mult=pl.col("UNIT_MULT").cast(pl.Int32))
            .select(pl.col("TableName").alias("table"), "line", pl.col("SeriesCode").alias("series_code"),
                    pl.col("LineDescription").alias("description"), "date", p.alias("period"), "value",
                    pl.col("CL_UNIT").alias("unit"), "unit_mult")
            .sort("line", "date"))


def parameters(dataset: str = "NIPA", parameter: str = "") -> pl.DataFrame:
    """A dataset's parameters (name, description, required, default ...), or with `parameter` the values it
    accepts (for NIPA "TableName": TableName and Description). `dataset` "" lists the datasets themselves."""
    if not dataset:
        return pl.DataFrame(_get(max_age_days=30, method="GetDataSetList").get("Dataset", []))
    if not parameter:
        return pl.DataFrame(_get(max_age_days=30, method="GetParameterList", DatasetName=dataset).get("Parameter", []))
    return pl.DataFrame(_get(max_age_days=7, method="GetParameterValues", DatasetName=dataset,
                             ParameterName=parameter).get("ParamValue", []))
