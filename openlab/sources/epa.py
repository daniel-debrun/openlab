"""EPA Greenhouse Gas Reporting Program (GHGRP) through Envirofacts (data.epa.gov/efservice): facility emissions
by year, gas and sector for every US facility over 25,000 tonnes CO2e a year, with its parent companies. No key."""
from __future__ import annotations

import urllib.parse

import polars as pl

from . import http

BASE = "https://data.epa.gov/efservice"
http.LIMITS.setdefault("data.epa.gov", http.Limiter(2, name="data.epa.gov"))
_PAGE = 10000


def _rows(path: str) -> list:
    out, start = [], 0
    while True:
        page = http.get_json(f"{BASE}/{path}/rows/{start}:{start + _PAGE - 1}/JSON", timeout=180)
        out += page
        if len(page) < _PAGE:
            return out
        start += _PAGE


def emissions(year: int | None = None, state: str = "", company: str = "", facility_id: int | None = None) -> pl.DataFrame:
    """Reported emissions, one row per facility, year, sector, subsector and gas: facility_id, facility_name,
    parent_company, state, city, naics_code, year, sector, sector_type (E emitter, S supplier), subsector_id,
    gas, co2e (metric tonnes CO2e). Filter by `year`, `state` (two letters), `company` (substring of the parent
    company, any case) or `facility_id`; at least one is needed."""
    f = [("FACILITY_ID", str(facility_id)) if facility_id else None, ("STATE", state.upper()) if state else None,
         ("YEAR", str(year)) if year else None,
         ("PARENT_COMPANY/CONTAINING", urllib.parse.quote(company)) if company else None]
    f = [x for x in f if x]
    if not f:
        raise ValueError("emissions needs at least one of year, state, company or facility_id")
    rows = _rows("PUB_DIM_FACILITY/" + "/".join(f"{k}/{v}" for k, v in f) + "/PUB_FACTS_SECTOR_GHG_EMISSION")
    if not rows:
        return pl.DataFrame()
    gas = {g["gas_id"]: g["gas_code"] for g in _rows("PUB_DIM_GHG")}
    sec = {s["sector_id"]: (s["sector_name"], s["sector_type"]) for s in _rows("PUB_DIM_SECTOR")}
    keep = [{"facility_id": r["facility_id"], "facility_name": r["facility_name"], "parent_company": r["parent_company"],
             "state": r["state"], "city": r["city"], "naics_code": r["naics_code"], "year": r["year"],
             "sector": sec.get(r["sector_id"], (None,))[0], "sector_type": sec.get(r["sector_id"], (None, None))[1],
             "subsector_id": r["subsector_id"], "gas": gas.get(r["gas_id"]), "co2e": r["co2e_emission"]}
            for r in rows if r["co2e_emission"] is not None]
    return pl.DataFrame(keep, infer_schema_length=None).sort("facility_id", "year", "sector", "gas")
