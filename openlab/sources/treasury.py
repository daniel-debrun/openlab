"""US Treasury: the daily par yield curve (home.treasury.gov) and the Fiscal Data API
(api.fiscaldata.treasury.gov): average interest rates on the debt, debt to the penny, auction results, and any
other Fiscal Data endpoint through `dataset`. No key.

Fiscal Data serves every value as text, with "null" for missing; `dataset` converts each column by the type
the API declares for it (dates to pl.Date, amounts and rates to floats)."""
from __future__ import annotations

import io
import urllib.parse
from datetime import date

import polars as pl

from . import http

FISCAL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
CURVE = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv"
_LIMIT = http.Limiter(4)
_FLOAT = ("CURRENCY", "NUMBER", "PERCENTAGE")
_INT = ("INTEGER", "YEAR", "QUARTER", "MONTH", "DAY")


def yield_curve(start_year: int, end_year: int | None = None) -> pl.DataFrame:
    """Daily Treasury par yield curve rates: date, tenor ("1 Mo" .. "30 Yr"), years (the tenor in years), rate
    (percent), one row per date and tenor, from 1990. Tenors come and go (2 Mo from 2018, 4 Mo from 2022, 20 Yr
    absent 1987-1993, 30 Yr absent 2002-2006); missing tenors have no rows."""
    frames = []
    for y in range(int(start_year), int(end_year or start_year) + 1):
        _LIMIT.wait()
        url = f"{CURVE}/{y}/all?type=daily_treasury_yield_curve&field_tdr_date_value={y}&page&_format=csv"
        raw = http.get_bytes(url, cache=y < date.today().year)       # the current year changes daily
        if raw.strip():
            frames.append(pl.read_csv(io.BytesIO(raw), infer_schema_length=0)
                          .unpivot(index="Date", variable_name="tenor", value_name="rate"))
    if not frames:
        return pl.DataFrame(schema={"date": pl.Date, "tenor": pl.Utf8, "years": pl.Float64, "rate": pl.Float64})
    d = pl.concat(frames).filter(pl.col("rate").is_not_null() & (pl.col("rate") != "N/A"))
    n = d["tenor"].str.extract(r"^([\d.]+)").cast(pl.Float64)
    years = pl.when(d["tenor"].str.contains("Mo")).then(n / 12).otherwise(n)
    return (d.with_columns(date=pl.col("Date").str.to_date("%m/%d/%Y"), years=years,
                           rate=pl.col("rate").cast(pl.Float64))
            .select("date", "tenor", "years", "rate").sort("date", "years"))


def dataset(path: str, filter: str = "", fields: str = "", sort: str = "-record_date",
            max_rows: int = 10000) -> pl.DataFrame:
    """Any Fiscal Data endpoint by its `path` (e.g. "v2/accounting/od/debt_to_penny"; the list is at
    fiscaldata.treasury.gov/api-documentation): its columns, typed. `filter` is the API's syntax
    ("record_date:gte:2024-01-01,security_type:eq:Note"), `fields` a comma list, `sort` a column (- for
    descending); at most `max_rows` rows, newest first by default."""
    rows, types, page = [], {}, 1
    while len(rows) < max_rows:
        p = {"page[number]": page, "page[size]": min(10000, max_rows), "sort": sort}
        if filter:
            p["filter"] = filter
        if fields:
            p["fields"] = fields
        _LIMIT.wait()
        b = http.get_json(f"{FISCAL}/{path.strip('/')}?{urllib.parse.urlencode(p)}", max_age_days=1)
        if "data" not in b:
            raise RuntimeError(f"Fiscal Data {path}: {b.get('message') or b.get('error') or b}"[:300])
        rows += b["data"]
        types = b["meta"]["dataTypes"]
        if page >= b["meta"]["total-pages"]:
            break
        page += 1
    if not rows:
        return pl.DataFrame()
    d = pl.DataFrame(rows[:max_rows], schema=dict.fromkeys(rows[0], pl.Utf8)).with_columns(pl.all().replace("null", None))
    casts = []
    for c, t in types.items():
        if c not in d.columns:
            continue
        if t == "DATE":
            casts.append(pl.col(c).str.to_date("%Y-%m-%d", strict=False))
        elif t.startswith(_FLOAT):
            casts.append(pl.col(c).cast(pl.Float64, strict=False))
        elif t in _INT:
            casts.append(pl.col(c).cast(pl.Int64, strict=False))
    return d.with_columns(casts)


def avg_interest_rates(start: str = "2001-01-01") -> pl.DataFrame:
    """Average interest rate on Treasury debt, monthly from 2001: record_date, security_type_desc (Marketable,
    Non-marketable), security_desc (Treasury Bills, Notes, Bonds, TIPS, ...), avg_interest_rate_amt (percent)."""
    return dataset("v2/accounting/od/avg_interest_rates", f"record_date:gte:{start}",
                   "record_date,security_type_desc,security_desc,avg_interest_rate_amt", max_rows=100000)


def debt_to_penny(start: str = "1993-01-01") -> pl.DataFrame:
    """Total public debt outstanding, each business day from 1993: record_date, debt_held_public_amt,
    intragov_hold_amt, tot_pub_debt_out_amt (dollars)."""
    return dataset("v2/accounting/od/debt_to_penny", f"record_date:gte:{start}",
                   "record_date,debt_held_public_amt,intragov_hold_amt,tot_pub_debt_out_amt", max_rows=100000)


def auctions(start: str = "2020-01-01", security_type: str = "") -> pl.DataFrame:
    """Treasury auction results and announced upcoming auctions (results null until held), from 1979:
    auction_date, issue_date, maturity_date, cusip, security_type (Bill, Note, Bond, TIPS, FRN, CMB),
    security_term, offering_amt, total_tendered, total_accepted, bid_to_cover_ratio, high_yield,
    high_discnt_rate, int_rate, and the bidder split (primary_dealer/direct_bidder/indirect_bidder_accepted)."""
    cols = ("auction_date,issue_date,maturity_date,cusip,security_type,security_term,offering_amt,total_tendered,"
            "total_accepted,bid_to_cover_ratio,high_yield,high_discnt_rate,int_rate,primary_dealer_accepted,"
            "direct_bidder_accepted,indirect_bidder_accepted")
    f = f"auction_date:gte:{start}" + (f",security_type:eq:{security_type}" if security_type else "")
    return dataset("v1/accounting/od/auctions_query", f, cols, sort="-auction_date", max_rows=100000)
