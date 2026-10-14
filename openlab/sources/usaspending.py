"""USAspending.gov: federal awards (contracts, grants, loans) by recipient, and a recipient's obligations over time.
No key. The search API is POST-only, so responses are cached by URL plus request body."""
from __future__ import annotations

import json
import time
import urllib.request

import polars as pl

from . import http

BASE = "https://api.usaspending.gov/api/v2"
_LIMIT = http.Limiter(4)
CONTRACTS = ("A", "B", "C", "D")


def _post(path: str, body: dict, max_age_days: float = 7.0) -> dict:
    url = f"{BASE}{path}"
    p = http._key(url + json.dumps(body, sort_keys=True))
    if p.exists() and (time.time() - p.stat().st_mtime) / 86400 <= max_age_days:
        return json.loads(p.read_text())["body"]
    _LIMIT.wait()
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "User-Agent": "openlab-research/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read())
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"url": url, "request": body, "fetched": time.strftime("%Y-%m-%dT%H:%M:%S"), "body": out}))
    return out


def _filters(recipient: str, start: str, end: str) -> dict:
    return {"recipient_search_text": [recipient], "time_period": [{"start_date": start, "end_date": end}]}


def awards(recipient: str, start: str, end: str, award_types: tuple = CONTRACTS, limit: int = 100) -> pl.DataFrame:
    """Awards to recipients matching `recipient` (name or UEI) with activity between `start` and `end`, largest
    first: award_id, recipient, amount, start_date, end_date, agency, description, award_key. `amount` is the
    award's lifetime total, not the window's share (use `over_time` for that). `award_types` are
    USAspending codes: contracts A-D (default), grants 02-05, loans 07-08 (one family per call)."""
    fields = ["Award ID", "Recipient Name", "Award Amount", "Start Date", "End Date", "Awarding Agency",
              "Description", "generated_internal_id"]
    rows, page = [], 1
    while len(rows) < limit:
        body = {"filters": {**_filters(recipient, start, end), "award_type_codes": list(award_types)},
                "fields": fields, "sort": "Award Amount", "order": "desc", "limit": min(100, limit), "page": page}
        r = _post("/search/spending_by_award/", body)
        rows += r["results"]
        if not r["page_metadata"].get("hasNext"):
            break
        page += 1
    if not rows:
        return pl.DataFrame()
    names = dict(zip(fields, ["award_id", "recipient", "amount", "start_date", "end_date", "agency", "description",
                              "award_key"]))
    return (pl.DataFrame([{names[f]: x.get(f) for f in fields} for x in rows[:limit]], infer_schema_length=None)
            .with_columns(pl.col("amount").cast(pl.Float64),
                          pl.col("start_date", "end_date").str.to_date(strict=False)))


def over_time(recipient: str, start: str, end: str, group: str = "quarter") -> pl.DataFrame:
    """Obligations to recipients matching `recipient` per federal fiscal period (`group`: fiscal_year, quarter,
    month): fiscal_year, quarter or month, obligations (all types), contracts, idvs, grants, loans, direct, other."""
    r = _post("/search/spending_over_time/", {"group": group, "filters": _filters(recipient, start, end)})
    kinds = {"Contract": "contracts", "Idv": "idvs", "Grant": "grants", "Loan": "loans", "Direct": "direct",
             "Other": "other"}
    rows = [{**{k: int(v) for k, v in x["time_period"].items()}, "obligations": x["aggregated_amount"],
             **{v: x.get(f"{k}_Obligations") for k, v in kinds.items()}} for x in r["results"]]
    return pl.DataFrame(rows).with_columns(pl.exclude("fiscal_year", "quarter", "month").cast(pl.Float64))
