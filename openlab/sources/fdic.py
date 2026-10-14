"""FDIC BankFind (api.fdic.gov/banks): every FDIC-insured institution, active or closed, and its quarterly Call
Report financials. No key. Dollar amounts are in thousands, as the FDIC publishes them."""
from __future__ import annotations

import urllib.parse

import polars as pl

from . import http

BASE = "https://api.fdic.gov/banks"
http.LIMITS.setdefault("api.fdic.gov", http.Limiter(5, name="api.fdic.gov"))
INST_FIELDS = "CERT,NAME,CITY,STALP,ACTIVE,ASSET,DEP,BKCLASS,ESTYMD,ENDEFYMD,REPDTE"
FIN_FIELDS = "CERT,REPDTE,ASSET,DEP,LNLSNET,EQ,NETINC,ROA,ROE,NIMY"


def _get(path: str, max_age_days: float = 7.0, **params) -> pl.DataFrame:
    rows, offset = [], 0
    while True:
        q = urllib.parse.urlencode({**params, "limit": 10000, "offset": offset})
        d = http.get_json(f"{BASE}/{path}?{q}", max_age_days=max_age_days)
        rows += [x["data"] for x in d["data"]]
        offset += 10000
        if offset >= d["meta"]["total"]:
            d = pl.DataFrame(rows, infer_schema_length=None)
            return d.select([c for c in params["fields"].split(",") if c in d.columns])


def institutions(name: str = "", state: str = "", active: bool | None = True, filters: str = "",
                 fields: str = INST_FIELDS) -> pl.DataFrame:
    """Insured institutions: cert (the FDIC id), name, city, state, active, asset and dep (latest, $000), bkclass,
    established, ended (closed or merged; null while active), repdte. `name` matches words in the name; `filters`
    is raw FDIC syntax (e.g. 'ASSET:[1000000 TO *]'); `active=None` keeps closed institutions too."""
    f = [x for x in (f"STALP:{state.upper()}" if state else "", f"ACTIVE:{int(active)}" if active is not None else "",
                     filters) if x]
    p = {"fields": fields, "sort_by": "ASSET", "sort_order": "DESC"}
    if f:
        p["filters"] = " AND ".join(f)
    if name:
        p["search"] = f"NAME:{name}"
    d = _get("institutions", **p)
    if d.is_empty():
        return d
    d = d.rename({c: c.lower() for c in d.columns}).rename({"stalp": "state"}, strict=False)
    dates = [c for c in ("estymd", "endefymd", "repdte") if c in d.columns]
    d = d.with_columns(pl.col(dates).cast(pl.Utf8).str.to_date("%m/%d/%Y", strict=False))
    if "endefymd" in d.columns:                 # the FDIC writes 9999-12-31 for "still open"
        d = d.with_columns(pl.when(pl.col("endefymd").dt.year() < 9999).then(pl.col("endefymd")).alias("endefymd"))
    return d.rename({"estymd": "established", "endefymd": "ended"}, strict=False)


def financials(cert: int, start: str = "", fields: str = FIN_FIELDS) -> pl.DataFrame:
    """Quarterly Call Report figures for one institution: cert, date (quarter end), and the requested `fields`
    (default asset, dep, lnlsnet loans, eq equity, netinc year-to-date, roa, roe, nimy; amounts $000, ratios %)."""
    f = f"CERT:{int(cert)}" + (f" AND REPDTE:[{start.replace('-', '')} TO *]" if start else "")
    d = _get("financials", filters=f, fields=fields, sort_by="REPDTE", sort_order="ASC")
    if d.is_empty():
        return d
    d = d.rename({c: c.lower() for c in d.columns})
    return d.with_columns(pl.col("repdte").cast(pl.Utf8).str.to_date("%Y%m%d")).rename({"repdte": "date"})
