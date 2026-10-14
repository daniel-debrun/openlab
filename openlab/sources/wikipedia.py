"""Wikipedia pageviews (Wikimedia REST API): daily views of any article, a public measure of attention. No key;
Wikimedia requires a User-Agent with a contact, taken from `OPENLAB_SEC_CONTACT`."""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

import polars as pl

from . import http

BASE = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
_LIMIT = http.Limiter(10)


def _get(url: str, max_age_days: float) -> dict:
    p = http._key(url)
    if p.exists() and (time.time() - p.stat().st_mtime) / 86400 <= max_age_days:
        return json.loads(p.read_text())["body"]
    _LIMIT.wait()
    req = urllib.request.Request(url, headers={"User-Agent": f"openlab-research/0.1 ({http.sec_user_agent()})"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = json.loads(r.read())
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"url": url, "fetched": time.strftime("%Y-%m-%dT%H:%M:%S"), "body": body}))
    return body


def pageviews(article: str, start: str, end: str, project: str = "en.wikipedia", agent: str = "user") -> pl.DataFrame:
    """Daily views of one article between `start` and `end` (YYYY-MM-DD, from 2015-07-01): date, article, views.
    `agent` is user (people), automated, spider or all-agents; all devices are counted."""
    title = urllib.parse.quote(article.replace(" ", "_"), safe="")
    url = f"{BASE}/{project}/all-access/{agent}/{title}/daily/{start.replace('-', '')}/{end.replace('-', '')}"
    recent = end >= time.strftime("%Y-%m-%d", time.gmtime(time.time() - 3 * 86400))
    items = _get(url, 0.5 if recent else 30.0)["items"]
    return pl.DataFrame([{"date": x["timestamp"][:8], "article": article, "views": x["views"]} for x in items]
                        ).with_columns(pl.col("date").str.to_date("%Y%m%d"))
