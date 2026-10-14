"""One HTTP path for every public source: rate limited, cached on disk, retried, and honest about identity.

Each of the four providers has its own etiquette and all of it lands here rather than in four places:

* **SEC** publishes a rate limit (10 requests a second) and requires a declared `User-Agent` carrying a
  contact address. `OPENLAB_SEC_CONTACT` supplies it; unset, an SEC request is refused with the line that sets
  it. Nobody's personal address is sent anywhere without being put there deliberately.
* **FRED** needs a free key in `FRED_API_KEY`. Absent, its client raises with the URL to get one rather than
  failing at parse time.
* **IMF** and **World Bank** need no key and no identity.

The cache is the important part for research: a fundamentals panel is thousands of requests, a re-run must
not re-fetch them, and a point-in-time study must be able to prove it read what it read. Responses are
stored by URL hash under `$OPENLAB_SOURCE_CACHE` (default `~/.cache/openlab/sources`), with the fetch time beside
them.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

def _load_env(path: Path = Path.home() / ".config/openlab/research.env") -> None:
    """Read local credentials from ~/.config/openlab/research.env if the environment does not already carry
    them. Kept out of the repository and out of every command line: a key in a shell history or a git object
    is a key that has leaked."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_env()


def data_root(var: str, kind: str) -> Path:
    """A data root: `$var` (the environment, or the same key in research.env), else `~/openlab-data/<kind>`."""
    return Path(os.environ.get(var) or Path.home() / "openlab-data" / kind)


CACHE = Path(os.environ.get("OPENLAB_SOURCE_CACHE", str(Path.home() / ".cache/openlab/sources")))
SEC_PLACEHOLDER = "openlab-research (contact unset: export OPENLAB_SEC_CONTACT=you@example.com)"


def sec_contact_ok() -> bool:
    return bool(os.environ.get("OPENLAB_SEC_CONTACT"))


def sec_user_agent() -> str:
    return os.environ.get("OPENLAB_SEC_CONTACT") or SEC_PLACEHOLDER


def _sec_agent() -> str:
    """The SEC's required User-Agent; refused rather than sent without a contact."""
    if not sec_contact_ok():
        raise RuntimeError("the SEC requires a contact address: add OPENLAB_SEC_CONTACT=you@example.com to "
                           "~/.config/openlab/research.env (or export it)")
    return sec_user_agent()


#: Request quotas per source, as (calls, seconds) windows, from each provider's published free-tier terms. A
#: source with a quota refuses a call that would exceed it (QuotaExceeded) instead of burning the key.
QUOTAS = {"openlab.sources.alphavantage": ((25, 86400),),
          "openlab.sources.tiingo": ((50, 3600), (1000, 86400)),
          "openlab.sources.bls": ((500, 86400),),
          "openlab.sources.noaa": ((10000, 86400),),
          "openlab.sources.bea": ((100, 60),)}
LIMIT_DIR = Path(os.environ.get("OPENLAB_LIMIT_DIR", str(Path.home() / ".cache/openlab/limits")))


class QuotaExceeded(RuntimeError):
    pass


class Limiter:
    """A queue for one source, shared by every process on this machine: at most `per_second` requests leave
    for it, and none beyond its quota (QUOTAS). Agents running in parallel line up behind one file lock per
    source (~/.cache/openlab/limits/<name>.json), so ten agents send what one would, not ten times as much.

    `name` defaults to the calling module (openlab.sources.tiingo), so a provider's limiter is one queue
    whichever process creates it."""

    def __init__(self, per_second: float, name: str = ""):
        import sys
        self.min_gap = 1.0 / per_second
        self.name = name or sys._getframe(1).f_globals.get("__name__", "anon")
        self.quotas = QUOTAS.get(self.name, ())
        self.lock = threading.Lock()

    def _file(self) -> Path:
        LIMIT_DIR.mkdir(parents=True, exist_ok=True)
        return LIMIT_DIR / (self.name.replace("/", "_") + ".json")

    def wait(self) -> None:
        import fcntl
        with self.lock, self._file().open("a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)                # the queue: the next process waits here
            try:
                f.seek(0)
                st = json.loads(f.read() or "{}")
                now = time.time()
                if st.get("next", 0) > now:
                    time.sleep(st["next"] - now)
                    now = time.time()
                longest = max((w for _, w in self.quotas), default=0)
                calls = [t for t in st.get("calls", []) if now - t < longest]
                for limit, window in self.quotas:
                    used = [t for t in calls if now - t < window]
                    if len(used) >= limit:
                        reset = time.strftime("%H:%M", time.localtime(min(used) + window))
                        raise QuotaExceeded(f"{self.name}: quota of {limit} requests per {window // 3600 or 1}"
                                            f"{'h' if window >= 3600 else 'min'} used ({len(used)}); frees from {reset}. "
                                            "Read a snapshot (sources.fetch('snapshot', ...)) or wait.")
                if self.quotas:
                    calls.append(now)
                f.seek(0)
                f.truncate()
                f.write(json.dumps({"next": now + self.min_gap, "calls": calls}))
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


LIMITS = {h: Limiter(r, name=h) for h, r in (
    ("data.sec.gov", 8), ("www.sec.gov", 8), ("efts.sec.gov", 5), ("api.stlouisfed.org", 10),
    ("api.worldbank.org", 10), ("www.imf.org", 5), ("dataservices.imf.org", 3),
    # papers: arXiv asks for one request every 3 s; OpenAlex allows 10/s; Crossref's public pool ~5/s
    ("export.arxiv.org", 1 / 3), ("api.openalex.org", 8), ("api.crossref.org", 5))}


#: Query parameters that carry credentials: removed from a URL before it names or is written into a cache file.
SECRET_PARAMS = {"api_key", "apikey", "key", "token", "registrationkey", "userid", "access_token"}


def public_url(url: str) -> str:
    """The URL without its credential parameters, so no key ever reaches the disk cache."""
    u = urllib.parse.urlsplit(url)
    q = [(k, v) for k, v in urllib.parse.parse_qsl(u.query, keep_blank_values=True) if k.lower() not in SECRET_PARAMS]
    return urllib.parse.urlunsplit(u._replace(query=urllib.parse.urlencode(q)))


def _key(url: str) -> Path:
    url = public_url(url)
    h = hashlib.sha256(url.encode()).hexdigest()[:24]
    host = urllib.parse.urlparse(url).netloc.replace(":", "_")
    return CACHE / host / f"{h}.json"


RETRY = (429, 500, 502, 503, 504)


def _get(url: str, headers: dict, timeout: float, tries: int) -> bytes:
    """One GET's body, through the host's queue (LIMITS), retried on 429, 5xx and a dropped connection with
    exponential backoff and jitter: the SEC, arXiv and OpenAlex answer a burst with 429 rather than a ban, so
    backing off correctly is the whole of being a good citizen here."""
    host = urllib.parse.urlparse(url).netloc
    lim = LIMITS.get(host)
    headers = {**headers, "User-Agent": _sec_agent() if host.endswith("sec.gov") else "openlab-research/0.1"}
    for attempt in range(tries):
        if lim:
            lim.wait()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    import gzip
                    raw = gzip.decompress(raw)
                return raw
        except urllib.error.HTTPError as e:
            if e.code not in RETRY or attempt == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == tries - 1:
                raise
        time.sleep((2 ** attempt) + random.random())
    raise ValueError(f"tries must be at least 1, got {tries}")


def get_json(url: str, cache: bool = True, max_age_days: float = 30.0, tries: int = 4,
             timeout: float = 60.0) -> dict:
    """Fetch and parse JSON, from the cache when it is fresh enough."""
    p = _key(url)
    if cache and p.exists():
        age = (time.time() - p.stat().st_mtime) / 86400
        if age <= max_age_days:
            try:
                return json.loads(p.read_text())["body"]
            except (ValueError, KeyError):
                pass
    raw = _get(url, {"Accept": "application/json", "Accept-Encoding": "gzip, deflate"}, timeout, tries)
    body = json.loads(raw.decode("utf-8", "replace"))
    if cache:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")                      # a torn cache file would be served forever
        tmp.write_text(json.dumps({"url": public_url(url), "fetched": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                   "body": body}))
        tmp.replace(p)
    return body


def get_bytes(url: str, cache: bool = True, timeout: float = 120.0, tries: int = 4) -> bytes:
    """The same, for the archive endpoints that serve zip, tsv, XML or PDF rather than JSON."""
    p = _key(url).with_suffix(".bin")
    if cache and p.exists():
        return p.read_bytes()
    raw = _get(url, {}, timeout, tries)
    if cache:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(raw)
        tmp.replace(p)
    return raw
