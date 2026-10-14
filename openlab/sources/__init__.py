"""Data providers: every outside source behind one small surface, each declaring its limits as data.

A provider is a name, the endpoints it answers, and :class:`Limits` -- what the source cannot tell you. The
limits are the point. Every trap met in research so far was a property of the source, not of the analysis:
FRED's current values are not what was known on the day, a survey's coverage thins before some year. Written here once, they travel with the data: a dataset names its providers in ``%%source`` and
every reply about that dataset carries their limits, so the next user or agent reads them before the rows.

Nothing here is domain-specific by design: FRED, the World Bank and the IMF are general statistics, and a
new provider for any domain is a :func:`register` call. Importing :mod:`openlab` never imports this package.

    from openlab import sources
    sources.fetch("fred", "observations", series_id="GDPC1", realtime_start="2020-05-01")
    sources.describe("fred")["limits"]
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Limits:
    """What a source cannot tell you. Each field is a sentence; "" means nothing known to declare."""

    history: str = ""       # how far back, and where coverage thins
    frequency: str = ""     # the finest grain served
    delay: str = ""         # how late the data arrives relative to the event
    revisions: str = ""     # whether past values change, and whether the old values can be had
    survivorship: str = ""  # who is missing because they no longer exist
    timezone: str = ""      # what the timestamps mean
    licence: str = ""       # what you may do with it
    rate: str = ""          # request limits, and how this package keeps to them
    notes: tuple = ()       # any other trap, one sentence each

    def as_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items() if v}


@dataclass(frozen=True)
class Provider:
    name: str
    about: str
    limits: Limits
    endpoints: dict = field(default_factory=dict)   # endpoint name -> "module:function", imported on first use
    needs: tuple = ()                               # environment variables it cannot run without


PROVIDERS: dict = {}


def register(provider: Provider) -> Provider:
    PROVIDERS[provider.name] = provider
    return provider


def get(name: str) -> Provider:
    try:
        return PROVIDERS[name]
    except KeyError:
        raise KeyError(f"no provider {name!r}; the providers are {', '.join(sorted(PROVIDERS))}") from None


def _resolve(target: str) -> Callable:
    import importlib
    mod, fn = target.split(":")
    return getattr(importlib.import_module(mod), fn)


def describe(name: str) -> dict:
    """A provider as data: what it is, its endpoints with their first doc line, its limits, what it needs."""
    p = get(name)
    eps = {}
    for ep, target in p.endpoints.items():
        try:
            doc = (_resolve(target).__doc__ or "").strip().split("\n")[0]
        except ImportError as e:            # an endpoint whose extra is missing still lists, with the reason
            doc = f"(unavailable: {e})"
        eps[ep] = doc
    return {"name": p.name, "about": p.about, "endpoints": eps, "limits": p.limits.as_dict(),
            "needs": {v: bool(os.environ.get(v)) for v in p.needs}}


def fetch(name: str, endpoint: str, **kwargs):
    """Call one endpoint of one provider."""
    p = get(name)
    if endpoint not in p.endpoints:
        raise KeyError(f"{name} has no endpoint {endpoint!r}; it has {', '.join(p.endpoints)}")
    return _resolve(p.endpoints[endpoint])(**kwargs)


def limits(name: str) -> dict:
    return get(name).limits.as_dict()


_M = "openlab.sources.macro"

register(Provider(
    "fred", "Federal Reserve Bank of St. Louis economic series (800k+), with vintages: the values as they "
            "stood on any past date.",
    Limits(history="Series-dependent; some from the 1940s.",
           frequency="Series-dependent: daily to annual.",
           delay="Released on each series' own schedule; release_dates gives the measured lag.",
           revisions="Values are revised. Current values were not known at the time: pass realtime_start "
                     "for the vintage a backtest on that date would have read.",
           timezone="Dates are observation periods, not publication times.",
           licence="Mostly public domain; some series carry third-party copyright (see the series notes).",
           rate="Free API key required (FRED_API_KEY); kept under 10 requests/s, cached 7-30 days.",
           notes=("A series can be silently shorter than its title suggests; check the first date.",)),
    endpoints={"observations": f"{_M}:fred_observations", "series": f"{_M}:fred_series",
               "release_dates": f"{_M}:fred_release_dates", "search": f"{_M}:fred_search"},
    needs=("FRED_API_KEY",)))

register(Provider(
    "worldbank", "World Bank development indicators for every country (16k+ indicators).",
    Limits(history="Mostly from 1960.",
           frequency="Annual (a few quarterly).",
           delay="Months to years after the period.",
           revisions="Revised silently, with no vintages: fine for cross-country structure, never for timing.",
           licence="CC BY 4.0.",
           rate="No key; kept under 10 requests/s, cached 30 days.",
           notes=("Aggregates (regions, income groups) are mixed in with countries; filter on iso3.",)),
    endpoints={"indicator": f"{_M}:worldbank", "indicators": f"{_M}:worldbank_indicators"}))

register(Provider(
    "imf", "IMF DataMapper indicators (World Economic Outlook and others) by country.",
    Limits(history="Mostly from 1980.",
           frequency="Annual.",
           delay="WEO updates twice a year.",
           revisions="Revised silently, with no vintages.",
           licence="IMF terms: free with attribution.",
           rate="No key; kept under 5 requests/s, cached 30 days.",
           notes=("Series mix forecasts into the same column, out to five or more years ahead: a value "
                  "for a future year is a forecast.",
                  "The API ignores the country path segment; countries are filtered here.")),
    endpoints={"indicator": f"{_M}:imf_series", "indicators": f"{_M}:imf_indicators"}))

register(Provider(
    "census", "US Census Bureau: the economic indicator time series (retail sales, housing starts and permits, "
              "durable goods, trade, inventories, quarterly financial report, ...) and a generic query for any Census "
              "API dataset (ACS, CBP, ...).",
    Limits(history="Indicator series mostly from the 1990s (retail sales 1992-, housing 1959-); ACS and others by vintage.",
           frequency="Monthly, or quarterly (qfr, qss, qtax, ...); ACS annual.",
           delay="Released on the Census economic calendar, roughly 2-7 weeks after the reference month.",
           revisions="Values are as revised today (annual benchmark revisions change history); no vintages served.",
           timezone="Dates are reference periods (month or quarter start), not release times.",
           licence="US government work: public domain.",
           rate="With a key there is no published daily cap (500 a day per IP without one); kept to 5/s; cached by URL "
                "(key never stored).",
           notes=("`seasonally_adj` is a required predicate: adjusted and unadjusted series are separate calls.",
                  "Rows with is_error True are standard errors or other sampling measures, not levels (E_* codes).",
                  "Category and data type codes differ by program (resconst starts are ASTARTS/TOTAL, annual rate in "
                  "thousands); look them up with query() first.",
                  "Quarterly programs take `start` as YYYY; a YYYY-MM start is refused.")),
    endpoints={e: f"openlab.sources.census:{e}" for e in ("indicator", "query")},
    needs=("CENSUS_API_KEY",)))

register(Provider(
    "eia", "U.S. Energy Information Administration API v2: any energy series by route and facets (petroleum, natural "
           "gas, electricity, coal, STEO), weekly crude and gasoline stocks, and WTI, Brent and Henry Hub spot prices.",
    Limits(history="Spot prices from 1986 (WTI, Brent 1987, Henry Hub 1997); weekly stocks from 1982; varies by route.",
           frequency="Daily, weekly, monthly, quarterly, annual; hourly for electricity grid routes.",
           delay="Weekly petroleum status: Wednesday 10:30 ET for the week ending the previous Friday; daily spot "
                 "prices a few days late; monthly data 1-3 months late.",
           revisions="Values are as revised today; EIA keeps no vintages in the API.",
           timezone="Periods are dates (weekly = week ending Friday); hourly electricity periods are UTC unless the "
                    "frequency is local-hourly.",
           licence="US government work: public domain.",
           rate="No fixed published limit; EIA throttles bursts; kept to 5/s; 5000 rows a request, paged by offset; cached "
                "by URL (key never stored).",
           notes=("Facet names differ by route (series, duoarea, product, stateid, sectorid): read the route's metadata.",
                  "Weekly stocks are in thousand barrels (MBBL); crude excludes the Strategic Petroleum Reserve.")),
    endpoints={e: f"openlab.sources.eia:{e}" for e in ("series", "spot", "weekly_stocks")},
    needs=("EIA_API_KEY",)))

register(Provider(
    "noaa", "NOAA NCEI climate data: GHCN daily station summaries (temperature, precipitation, snow, wind, weather "
            "types) and station search by area, place or name.",
    Limits(history="GHCN daily from the 1800s for long stations (Central Park 1869-); most stations far shorter.",
           frequency="Daily.",
           delay="A few days; recent values are provisional.",
           revisions="Quality control can change or flag recent values; no vintages.",
           survivorship="Closed stations stay in the archive with their maxdate; station search lists them.",
           timezone="Dates are local station days (the observation day ends at the station's reading time).",
           licence="US government work: public domain.",
           rate="Daily data: NCEI Access Data Service, no token, no published limit. Station search: CDO v2, 5 a second and "
                "10,000 a day per token (NOAA_TOKEN, also read as NOAA_TOEKN), kept to 4/s and paged at 1000 rows; "
                "cached by URL (the token travels in a header, never stored).",
           notes=("Units are metric by default: deg C, mm, m/s (`units='standard'` for F and inches).",
                  "Element columns are present only when the station reports them; missing days are absent rows, not "
                  "nulls.",
                  "Station ids carry a 'GHCND:' prefix in CDO; daily() accepts either form.")),
    endpoints={e: f"openlab.sources.noaa:{e}" for e in ("daily", "stations")}))

register(Provider(
    "usda", "USDA NASS Quick Stats: crop and livestock production, area, yield, prices received, stocks and weekly "
            "crop condition and progress, by year and national, state or county geography.",
    Limits(history="Survey data from 1866 for major crops nationally; county data mostly from the 1970s; Census of "
                   "Agriculture every 5 years.",
           frequency="Annual, monthly and weekly (crop progress and condition).",
           delay="Weekly progress on Mondays for the week ending Sunday; annual estimates months after harvest.",
           revisions="Forecasts (\"YEAR - AUG FORECAST\") and final estimates are separate rows; load_time dates each.",
           timezone="Dates are reference periods; week_ending is a Sunday.",
           licence="US government work: public domain.",
           rate="No published rate limit; one query returns at most 50,000 rows (refused beyond); kept to 2/s; cached by "
                "URL (key never stored).",
           notes=("\"(D)\" values are withheld to avoid disclosing individual operations, and parse to null like (Z), (NA), "
                  "(X) and (S).",
                  "SURVEY and CENSUS rows for the same item differ: filter on source.",
                  "Commodity and statistic names are Quick Stats' uppercase labels (e.g. CORN, AREA PLANTED).")),
    endpoints={"quickstats": "openlab.sources.usda:quickstats"},
    needs=("USDA_NASS_KEY",)))

register(Provider(
    "fred_calendar", "FRED's release calendar: when each statistical release (CPI, jobs, GDP, ...) lands, past and "
                     "scheduled, and which series each release carries.",
    Limits(history="Release dates from each release's start (CPI from 1949); scheduled dates a few months ahead.",
           frequency="Per release date.",
           delay="Live: the calendar updates as releases are published.",
           revisions="Scheduled dates can move (e.g. a government shutdown); past dates are what happened.",
           timezone="Dates only, no release time (most BLS/BEA releases are 08:30 New York).",
           licence="FRED terms; the calendar is public information.",
           rate="Free key (FRED_API_KEY): 120 requests a minute, kept at 2 a second; cached 1 day by URL without the key.",
           notes=("FRED's /releases/dates hangs on any offset, so all-release ranges are paged by date window.",
                  "Release ids: 10 CPI, 50 Employment Situation, 53 GDP, 54 Personal Income and Outlays.")),
    endpoints={e: f"openlab.sources.fred_calendar:{e}" for e in ("release_dates", "release_series")},
    needs=("FRED_API_KEY",)))

register(Provider(
    "treasury", "US Treasury: the daily par yield curve, average interest rates on the debt, debt to the penny, "
                "auction results and upcoming auctions, and any Fiscal Data endpoint.",
    Limits(history="Yield curve from 1990; debt to the penny from 1993; average rates from 2001; auctions from 1979.",
           frequency="Yield curve and debt daily (business days); average rates monthly; auctions per auction.",
           delay="Yield curve after the close the same day; debt to the penny one business day late.",
           revisions="Par yields are not revised; Fiscal Data tables are occasionally restated without vintages.",
           timezone="Dates are US business days (record_date is the as-of date).",
           licence="US government work: public domain.",
           rate="No key; kept at 4 a second; Fiscal Data cached 1 day, past yield curve years cached for good.",
           notes=("Yield curve tenors change over time (2 Mo from 2018, 4 Mo and 1.5 Month added later, 20 Yr and 30 Yr "
                  "gaps); a missing tenor has no rows.",
                  "Amounts are dollars (DTS tables are millions); Fiscal Data's own types drive the conversion.",
                  "Upcoming auctions appear with null results until they are held.")),
    endpoints={e: f"openlab.sources.treasury:{e}" for e in
               ("yield_curve", "avg_interest_rates", "debt_to_penny", "auctions", "dataset")}))

register(Provider(
    "bls", "Bureau of Labor Statistics: any BLS series by id (CPI, unemployment, payrolls, wages, JOLTS, "
           "productivity).",
    Limits(history="Series-dependent; CPI-U from 1913, payrolls from 1939; 20 years per call (split here).",
           frequency="Monthly, quarterly, half-yearly or annual by series.",
           delay="Released on BLS's schedule (CPI about 2 weeks after the month; jobs the first Friday).",
           revisions="Payrolls are revised for two months and re-benchmarked yearly; seasonal factors are revised "
                     "every year; the API serves only current values (FRED/ALFRED has vintages).",
           timezone="Dates are the first day of the reference period, not publication dates.",
           licence="US government work: public domain.",
           rate="Free key (BLS_API_KEY): 500 requests a day, 50 series and 20 years per request; cached 1 day "
                "without the key.",
           notes=("`preliminary` marks values BLS footnotes as preliminary (P).",
                  "An unknown series id returns no rows rather than an error.")),
    endpoints={"series": "openlab.sources.bls:series"},
    needs=("BLS_API_KEY",)))

register(Provider(
    "bea", "Bureau of Economic Analysis: NIPA tables (GDP and its components, personal income and outlays, PCE "
           "prices, saving) and the dataset parameter lists.",
    Limits(history="NIPA from 1929 (annual), 1947 (quarterly), 1959 (monthly) by table.",
           frequency="Annual, quarterly or monthly by table.",
           delay="GDP advance estimate about 4 weeks after the quarter, then second and third estimates.",
           revisions="Every estimate is revised (monthly, then annual and comprehensive updates); the API serves "
                     "current values only.",
           timezone="Dates are the first day of the reference period, not publication dates.",
           licence="US government work: public domain.",
           rate="Free key (BEA_API_KEY): 100 requests, 100 MB and 30 errors a minute, else a 1-hour lockout; kept at 1 "
                "a second; cached 1 day without the key.",
           notes=("value is in units of `unit` x 10^unit_mult (unit_mult 6 = millions).",
                  "BEA echoes the key in responses: only Results are kept and the key is scrubbed.",
                  "The key must be activated from BEA's signup e-mail before any call works.")),
    endpoints={e: f"openlab.sources.bea:{e}" for e in ("table", "parameters")},
    needs=("BEA_API_KEY",)))

register(Provider(
    "usaspending", "USAspending.gov: federal contracts, grants and loans awarded to a recipient, and its obligations "
                   "per fiscal quarter or month.",
    Limits(history="Searchable from FY2008 (2007-10-01); older awards (from 2000-10) only through bulk downloads.",
           frequency="Per award; over_time by fiscal year, quarter or month.",
           delay="Agencies report contracts within days (DoD holds them back 90 days); grants and loans monthly.",
           revisions="Awards are modified over their life and amounts restated; no vintages are served.",
           survivorship="Complete: awards to merged or dissolved recipients remain under the name used then.",
           timezone="Dates are calendar dates; periods are federal fiscal years (FY2024 = Oct 2023 - Sep 2024).",
           licence="US government work: public domain.",
           rate="No published limit; kept under 4 requests/s, cached 7 days by URL and request body.",
           notes=("`recipient` is a text search: it also matches subsidiaries and parents carrying the name, and "
                  "misses ones that do not (Sikorsky appears under 'Lockheed Martin'; check the recipient column).",
                  "An award's `amount` is its lifetime total, not what was obligated inside the window.")),
    endpoints={e: f"openlab.sources.usaspending:{e}" for e in ("awards", "over_time")}))

register(Provider(
    "fdic", "FDIC BankFind: every FDIC-insured bank and savings institution, active or closed, and its quarterly "
            "Call Report financials.",
    Limits(history="Financials from 1984 Q1 (Call Reports); institutions back to their founding.",
           frequency="Quarterly (quarter-end report dates).",
           delay="A quarter's Call Reports appear about 2 months after quarter end.",
           revisions="Amended Call Reports overwrite the original; no vintages.",
           survivorship="Complete: failed and merged institutions are kept (`active=None`), with `ended` dates.",
           timezone="Dates are report or event dates.",
           licence="US government work: public domain.",
           rate="No key; no published limit; kept under 5 requests/s, cached 7 days.",
           notes=("Dollar amounts are in thousands.",
                  "netinc is year-to-date: difference consecutive quarters within a year for quarterly income.",
                  "Data are per insured institution (cert), not per holding company: JPMorgan Chase & Co. files "
                  "through several certs.")),
    endpoints={e: f"openlab.sources.fdic:{e}" for e in ("institutions", "financials")}))

register(Provider(
    "epa", "EPA Greenhouse Gas Reporting Program via Envirofacts: emissions by facility, year, sector and gas, "
           "with parent companies, for every large US emitter and fuel supplier.",
    Limits(history="Reporting years 2010-2023 (2024 not yet published on Envirofacts as of 2026-10-01).",
           frequency="Annual, per facility.",
           delay="A reporting year is published about 9-10 months after it ends.",
           revisions="Facilities resubmit and EPA republishes past years; no vintages.",
           survivorship="Facilities that stop reporting keep their past years.",
           timezone="Reporting (calendar) years.",
           licence="US government work: public domain.",
           rate="No key; kept under 2 requests/s, 10,000 rows per page, cached 30 days.",
           notes=("Only facilities over 25,000 t CO2e a year must report: small sources are absent.",
                  "sector_type S rows are quantities supplied (fuel, industrial gases), not emitted: drop them for "
                  "emissions totals, and drop gas BIOCO2, which EPA excludes from totals.",
                  "parent_company is free text with ownership shares ('X (50%);Y (50%)'): a substring match on a "
                  "company counts jointly-owned facilities in full.")),
    endpoints={"emissions": "openlab.sources.epa:emissions"}))

register(Provider(
    "gdelt", "GDELT DOC 2.0: the volume and average tone of worldwide online news coverage of any query, over time.",
    Limits(history="From January 2017 (GDELT's stated coverage; not checked here).",
           frequency="Daily over long windows; hourly or 15-minute over short ones (GDELT chooses).",
           delay="Updated every 15 minutes.",
           revisions="None announced; the monitored source set changes, so normalise by all_articles (`share`).",
           survivorship="Only sources GDELT monitors; coverage of non-English and paywalled news is partial.",
           timezone="UTC.",
           licence="GDELT terms: free for any use with citation of the GDELT Project.",
           rate="One request per 5 s per address; kept to one per 6 s; a breach brings a 429 block lasting "
                "minutes; each timeline is two requests; cached 30 days (closed windows) or 2 hours.",
           notes=("tone is GDELT's dictionary score of article text, not calibrated sentiment.",
                  "A shared network address (a university) shares GDELT's quota with everyone on it.")),
    endpoints={"timeline": "openlab.sources.gdelt:timeline"}))

register(Provider(
    "wikipedia", "Wikipedia pageviews: daily views of any article on any Wikimedia project, a public measure of "
                 "attention.",
    Limits(history="From 2015-07-01.",
           frequency="Daily.",
           delay="About a day.",
           revisions="None.",
           survivorship="Views are counted by exact title: a renamed article's history stays under its old title "
                        "and redirects are counted separately.",
           timezone="UTC days.",
           licence="CC0 (public domain).",
           rate="No key; User-Agent with contact (OPENLAB_SEC_CONTACT); kept under 10 requests/s, cached 30 days "
                "(12 hours for recent windows).",
           notes=("agent='user' still includes some undetected bots: spikes of one or two days are often automated.",)),
    endpoints={"pageviews": "openlab.sources.wikipedia:pageviews"}))

register(Provider(
    "snapshot", "Named, versioned copies of provider data on disk (Parquet + manifest): built once through the "
                "shared request queue, then read offline by any number of agents.",
    Limits(history="Whatever the spec pulled: the manifest records each dataset's arguments.",
           frequency="As the source.",
           delay="Frozen at build time (`fetched` per dataset): a snapshot does not update.",
           revisions="None after build; rebuild under a new name for new data.",
           survivorship="As the source.",
           timezone="As the source.",
           licence="As each source: the manifest carries every provider's limits, licence included.",
           rate="Local files; building uses each provider's shared queue and quotas.",
           notes=("Root: $OPENLAB_SNAPSHOTS, else ~/openlab-data/snapshots.",)),
    endpoints={"build": "openlab.sources.snapshot:build", "read": "openlab.sources.snapshot:read",
               "list": "openlab.sources.snapshot:listing"}))
