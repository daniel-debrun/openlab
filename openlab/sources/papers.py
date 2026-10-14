"""Papers: search OpenAlex, arXiv and Crossref from one prompt, merge what they return, and follow citations.

* **OpenAlex**: about 250 million works across every field, updated daily, with abstracts, citation counts,
  open-access PDF links and the citation graph. No key.
* **arXiv**: the newest preprints, the day they are posted (CS, statistics, maths, physics, q-fin, econ).
* **Crossref**: DOI metadata from every publisher that registers them; wide, but with no abstracts for many.

Relevance: each source ranks by its own relevance, and ``merge`` fuses the lists by reciprocal rank (a paper
near the top of several lists beats one at the top of one), deduplicated by DOI, arXiv id and title. That is
a candidate list, not a judgment: reading the abstracts against the prompt is the agent's job
(a screener's, in a review). Recency: every search also asks each source for its newest matches, so a paper
posted this week is a candidate even before anything cites it.

A paper is a plain dict: ``id`` (``doi:…``, ``arxiv:…`` or ``openalex:W…``), ``title``, ``authors``,
``date``, ``year``, ``venue``, ``abstract``, ``cited_by``, ``pdf``, ``url``, ``doi``, ``arxiv``, ``openalex``,
``sources`` (which searches found it), ``score`` (the fused rank score) and, from ``search``,
``relevance`` (topical match of title and abstract to the queries, the first sort key). Where the index says,
it also carries ``version`` (``preprint`` or ``published``), ``retracted`` (OpenAlex) and ``journal_doi`` (the
DOI an arXiv preprint names for its published version).
"""
from __future__ import annotations

import datetime as dt
import os
import re
import subprocess
import tempfile
import urllib.parse
import xml.etree.ElementTree as ET

from .http import get_bytes, get_json

SOURCES = ("openalex", "arxiv", "crossref")
LIMITS = {
    "openalex": "Abstracts are rebuilt from OpenAlex's word index and are missing for some publishers; citation "
                "counts lag new papers by weeks.",
    "arxiv": "Preprints: not peer reviewed; an arXiv paper may since have a journal version under a DOI.",
    "crossref": "Metadata only, and many records carry no abstract; its relevance ranking is the weakest.",
}
FRESH_DAYS = 0.25                 # a search is re-asked after six hours: "up to date" is the point of the tool


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def _paper(**kw) -> dict:
    p = {"id": "", "title": "", "authors": [], "date": "", "year": None, "venue": "", "abstract": "",
         "cited_by": None, "pdf": "", "url": "", "doi": "", "arxiv": "", "openalex": "", "sources": []}
    p.update({k: v for k, v in kw.items() if v not in (None, "")})
    p["doi"] = (p["doi"] or "").lower().replace("https://doi.org/", "")
    if not p["arxiv"] and p["doi"].startswith("10.48550/arxiv."):
        p["arxiv"] = p["doi"].split("arxiv.", 1)[1]
    p["id"] = (f"doi:{p['doi']}" if p["doi"] and not p["arxiv"] else f"arxiv:{p['arxiv']}" if p["arxiv"]
               else f"openalex:{p['openalex']}" if p["openalex"] else f"title:{_norm_title(p['title'])[:80]}")
    if p["date"] and not p["year"]:
        p["year"] = int(p["date"][:4])
    return p


# --- OpenAlex ------------------------------------------------------------------------------------------------
def _oa_url(path: str, **params) -> str:
    # anonymous search is throttled when OpenAlex is busy; a free key (OPENALEX_API_KEY in research.env) is not
    if os.environ.get("OPENALEX_API_KEY"):
        params["api_key"] = os.environ["OPENALEX_API_KEY"]
    if os.environ.get("OPENALEX_MAILTO"):                       # the polite pool, only when deliberately set
        params["mailto"] = os.environ["OPENALEX_MAILTO"]
    return f"https://api.openalex.org/{path}?{urllib.parse.urlencode(params)}"


def _oa_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = sorted((i, w) for w, idx in inv.items() for i in idx)
    return " ".join(w for _, w in pos)


def _oa_paper(w: dict) -> dict:
    ids = w.get("ids") or {}
    loc = w.get("best_oa_location") or {}
    arxiv = ""
    for l in w.get("locations") or []:
        u = (l.get("landing_page_url") or "")
        if "arxiv.org/abs/" in u:
            arxiv = re.sub(r"v\d+$", "", u.split("arxiv.org/abs/", 1)[1])
    return _paper(title=w.get("title") or w.get("display_name") or "",
                  authors=[a["author"]["display_name"] for a in (w.get("authorships") or [])[:12]],
                  date=w.get("publication_date") or "", year=w.get("publication_year"),
                  venue=((w.get("primary_location") or {}).get("source") or {}).get("display_name") or "",
                  abstract=_oa_abstract(w.get("abstract_inverted_index")), cited_by=w.get("cited_by_count"),
                  pdf=loc.get("pdf_url") or "", url=w.get("doi") or ids.get("openalex") or "",
                  doi=w.get("doi") or "", arxiv=arxiv, openalex=(w.get("id") or "").rsplit("/", 1)[-1],
                  version="preprint" if w.get("type") in ("preprint", "posted-content") else "published",
                  retracted=bool(w.get("is_retracted")))


def _oa_query(query: str, words: int = 10) -> str:
    """The query as at most `words` content words: OpenAlex answers HTTP 400 to long prompts full of commas,
    colons, quotes or brackets, and ranks a short keyword query better anyway."""
    return " ".join(_words(query)[:words]) or query.strip()


def openalex(query: str, n: int = 25, since: str = "", open_access: bool = False, newest: bool = False) -> list:
    """Works matching `query`, by relevance (title, abstract and full text) or, with `newest`, by date (title and
    abstract only: date order over a full-text match surfaces anything that mentions the words once)."""
    f = [f"from_publication_date:{since}"] if since else []
    if open_access:
        f.append("is_oa:true")
    params = {"per-page": min(n, 200), "sort": "publication_date:desc" if newest else "relevance_score:desc"}
    query = _oa_query(query)
    if newest:
        f += [f"to_publication_date:{dt.date.today()}",         # a future-dated record is not the newest paper
              "title_and_abstract.search:" + query]
    else:
        params["search"] = query
    if f:
        params["filter"] = ",".join(f)
    body = get_json(_oa_url("works", **params), max_age_days=FRESH_DAYS)
    return [_oa_paper(w) for w in body.get("results", [])]


def openalex_work(key: str) -> dict:
    """One work by OpenAlex id (W…), DOI or arXiv id."""
    if key.startswith("W"):
        path = f"works/{key}"
    elif key.startswith("10."):
        path = f"works/https://doi.org/{key}"
    else:                                           # arXiv: OpenAlex keys it by its landing page, not its DOI
        url = f"http://arxiv.org/abs/{key}|https://arxiv.org/abs/{key}"
        hits = get_json(_oa_url("works", filter=f"locations.landing_page_url:{url}", **{"per-page": 1}),
                        max_age_days=FRESH_DAYS).get("results", [])
        if not hits:
            raise KeyError(f"arXiv {key} is not in OpenAlex yet")
        return hits[0]
    return get_json(_oa_url(path), max_age_days=FRESH_DAYS)


def _since(since: str) -> str:
    return f",from_publication_date:{since}" if since else ""


def works(ids: list, since: str = "") -> list:
    """The works with these OpenAlex ids (W… or their URLs), 50 a call, published on or after `since` if given.
    An id OpenAlex no longer serves is left out."""
    ids = [i.rsplit("/", 1)[-1] for i in ids]
    out = []
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        body = get_json(_oa_url("works", filter="openalex:" + "|".join(chunk) + _since(since),
                                **{"per-page": len(chunk)}), max_age_days=FRESH_DAYS)
        out += [_oa_paper(x) for x in body.get("results", [])]
    return out


def related(key: str, n: int = 20, related_works: bool = True, since: str = "") -> dict:
    """What a paper cites (its first n references), what cites it (most cited first), and OpenAlex's related works
    (unless `related_works` is false: they are not citation edges); `since` keeps works published from then on."""
    w = openalex_work(key)
    wid = w["id"].rsplit("/", 1)[-1]
    citing = get_json(_oa_url("works", filter=f"cites:{wid}" + _since(since), sort="cited_by_count:desc",
                              **{"per-page": n}), max_age_days=FRESH_DAYS)
    return {"paper": _oa_paper(w), "references": works((w.get("referenced_works") or [])[:n], since),
            "cited_by": [_oa_paper(x) for x in citing.get("results", [])],
            "related": works((w.get("related_works") or [])[:n], since) if related_works else []}


# --- arXiv ---------------------------------------------------------------------------------------------------
_ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def arxiv(query: str, n: int = 25, newest: bool = False) -> list:
    """Preprints matching `query` in any field, by relevance or, with `newest`, by submission date."""
    # arXiv ANDs every term and indexes no stopwords, so one "with" in the query empties the result
    terms = [w for w in re.findall(r"[\w-]+", query.lower()) if w not in _STOP][:8]
    words = " AND ".join(f"all:{w}" for w in terms)
    return _arxiv_feed({"search_query": words, "max_results": min(n, 100),
                        "sortBy": "submittedDate" if newest else "relevance", "sortOrder": "descending"})


def _arxiv_feed(params: dict) -> list:
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    root = ET.fromstring(get_bytes(url, cache=False, timeout=60))
    out = []
    for e in root.findall("a:entry", _ATOM):
        aid = re.sub(r"v\d+$", "", e.findtext("a:id", "", _ATOM).rsplit("/abs/", 1)[-1])
        doi = e.findtext("arxiv:doi", "", _ATOM)
        out.append(_paper(title=" ".join(e.findtext("a:title", "", _ATOM).split()),
                          authors=[a.findtext("a:name", "", _ATOM) for a in e.findall("a:author", _ATOM)][:12],
                          date=e.findtext("a:published", "", _ATOM)[:10],
                          venue="arXiv " + (e.find("arxiv:primary_category", _ATOM).get("term")
                                            if e.find("arxiv:primary_category", _ATOM) is not None else ""),
                          abstract=" ".join(e.findtext("a:summary", "", _ATOM).split()),
                          pdf=f"https://arxiv.org/pdf/{aid}", url=f"https://arxiv.org/abs/{aid}", arxiv=aid,
                          doi=doi, journal_doi=doi.lower(), version="preprint"))
    return out


# --- Crossref ------------------------------------------------------------------------------------------------
def crossref(query: str, n: int = 25, since: str = "") -> list:
    """Records matching `query` by Crossref's relevance (it has no usable newest-first: date order over its
    loose matching is noise). Records dated in the future (typos in deposited metadata) are dropped."""
    params = {"query.bibliographic": query, "rows": min(n, 100),
              "select": "DOI,title,author,issued,container-title,abstract,is-referenced-by-count,URL,link,type"}
    params["filter"] = f"from-pub-date:{since},until-pub-date:{dt.date.today()}" if since else \
        f"until-pub-date:{dt.date.today()}"
    body = get_json("https://api.crossref.org/works?" + urllib.parse.urlencode(params), max_age_days=FRESH_DAYS)
    out = []
    for it in body.get("message", {}).get("items", []):
        parts = ((it.get("issued") or {}).get("date-parts") or [[None]])[0]
        date = "-".join(f"{x:02d}" if i else str(x) for i, x in enumerate(parts) if x) if parts[0] else ""
        pdf = next((l["URL"] for l in it.get("link", []) if l.get("content-type") == "application/pdf"), "")
        out.append(_paper(title=" ".join((it.get("title") or [""])[0].split()),
                          authors=[f"{a.get('given', '')} {a.get('family', '')}".strip() for a in it.get("author", [])][:12],
                          date=date if len(date) == 10 else "", year=parts[0],
                          venue=(it.get("container-title") or [""])[0],
                          abstract=re.sub(r"<[^>]+>", "", it.get("abstract") or "").strip(),
                          cited_by=it.get("is-referenced-by-count"), pdf=pdf, url=it.get("URL", ""),
                          doi=it.get("DOI", ""),
                          version="preprint" if it.get("type") == "posted-content" else "published"))
    return out


def crossref_updates(dois: list) -> dict:
    """The update notices deposited against these DOIs (retraction, withdrawal, correction, ..., Retraction
    Watch's included), from Crossref's `updates` filter, 40 DOIs a call: {doi: [{"type", "notice", "date"}]}."""
    want = sorted({d.lower() for d in dois if d and "," not in d})     # a comma would split the filter
    out: dict = {}
    for i in range(0, len(want), 40):
        params = {"filter": ",".join(f"updates:{d}" for d in want[i:i + 40]), "rows": 1000,
                  "select": "DOI,update-to"}
        body = get_json("https://api.crossref.org/works?" + urllib.parse.urlencode(params), max_age_days=FRESH_DAYS)
        for it in body.get("message", {}).get("items", []):
            for u in it.get("update-to") or []:
                d = (u.get("DOI") or "").lower()
                if d in want:
                    date = "-".join(f"{x:02d}" for x in ((u.get("updated") or {}).get("date-parts") or [[]])[0])
                    out.setdefault(d, []).append({"type": u.get("type", ""), "notice": it["DOI"].lower(),
                                                  "date": date})
    return out


# --- merging -------------------------------------------------------------------------------------------------
def merge(lists: dict, k: int = 60) -> list:
    """Fuse ranked lists (``{label: [paper, ...]}``) by reciprocal rank, one record per paper: the same paper
    under its DOI, its arXiv id or its title is merged, keeping the richest fields."""
    by, alias = {}, {}
    for label, papers in lists.items():
        for rank, p in enumerate(papers):
            keys = [x for x in (p["doi"] and f"doi:{p['doi']}", p["arxiv"] and f"arxiv:{p['arxiv']}",
                                p["openalex"] and f"oa:{p['openalex']}", f"t:{_norm_title(p['title'])}") if x]
            hit = next((alias[x] for x in keys if x in alias), None)
            if hit is None:
                hit = p["id"]
                by[hit] = {**p, "sources": [], "score": 0.0}
            q = by[hit]
            for f, v in p.items():
                if f not in ("sources", "id") and v and (not q.get(f) or (f == "abstract" and len(v) > len(q[f]))):
                    q[f] = v
            q["sources"] = sorted(set(q["sources"]) | {label.split(":")[0]})
            q["score"] += 1.0 / (k + rank + 1)
            for x in keys:
                alias[x] = hit
    for q in by.values():
        q["id"] = _paper(**{f: q[f] for f in ("doi", "arxiv", "openalex", "title")})["id"]
        q["score"] = round(q["score"], 5)
    return sorted(by.values(), key=lambda p: -p["score"])


_STOP = frozenset("a an and the of for in on with to from by using via based is are or at as its does do did it this that "
                  "what how which than after before into over".split())


def _words(query: str) -> list:
    """The query's content words, in order, once each (no stop words, nothing under three letters)."""
    return list(dict.fromkeys(w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in _STOP and len(w) > 2))


def _on_topic(p: dict, query: str, share: float = 0.6) -> bool:
    """At least `share` of the query's content words appear in the title or abstract."""
    words = _words(query)
    text = f"{p['title']} {p['abstract']}".lower()
    return not words or sum(w in text for w in words) >= share * len(words)


def relevance(p: dict, query: str) -> float:
    """Topical match in [0, 1]: the share of the query's content words in the paper, a word in the title counting
    double one only in the abstract."""
    words = _words(query)
    if not words:
        return 0.0
    title, abstract = p["title"].lower(), p["abstract"].lower()
    return sum(2 if w in title else 1 if w in abstract else 0 for w in words) / (2 * len(words))


_ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5}|[a-z][a-z-]*(\.[A-Z]{2})?/\d{7})(v\d+)?")
_DOI = re.compile(r"10\.\d{4,9}/\S+")


def ask(src: str, query: str, n: int = 25, since: str = "", open_access: bool = False, newest: bool = False) -> list:
    """One query on one source, `since` applied (arXiv cannot filter by date, so its answer is cut here). A query
    that is a work's id (an arXiv id on arXiv, a DOI on OpenAlex) returns that work, so a review can target one."""
    q = query.strip()
    if src == "arxiv" and _ARXIV_ID.fullmatch(q):
        return [p for p in _arxiv_feed({"id_list": q, "max_results": 1}) if not since or p["date"] >= since]
    if src == "openalex" and _DOI.fullmatch(q):
        return [p for p in [_oa_paper(openalex_work(q))] if not since or p["date"] >= since]
    if src == "openalex":
        return openalex(query, n, since, open_access, newest)
    if src == "arxiv":
        return [p for p in arxiv(query, n, newest) if not since or p["date"] >= since]
    if src == "crossref":
        return crossref(query, n, since)
    raise ValueError(f"no source {src!r}: one of {', '.join(SOURCES)}")


def search(queries: list, n: int = 25, since: str = "", open_access: bool = False,
           sources: tuple = SOURCES, progress=None, stop=None) -> dict:
    """Every query against every source, by relevance and by newest, merged. A source that fails is reported
    in `failed` and the rest still answer. `progress(label, found, seconds, error)` is told after each call;
    `stop()` is asked before each, and a true answer ends the search with what it has (`stopped`)."""
    import time
    lists, failed, stopped = {}, [], False
    for q in queries:
        for src in sources:
            for newest in (False, True):
                if src == "crossref" and (newest or open_access):
                    continue
                if stop and stop():
                    stopped = True
                    break
                label = f"{src}:{'new' if newest else 'rel'}:{q}"
                t0, err = time.monotonic(), ""
                try:
                    lists[label] = ask(src, q, n, since, open_access, newest)
                except Exception as e:                  # noqa: BLE001 -- one source down must not blank the search
                    err = f"{type(e).__name__}: {str(e)[:120]}"
                    failed.append(f"{label}: {err}")
                if (newest or src == "crossref") and label in lists:   # date order, Crossref's loose match: on topic
                    lists[label] = [p for p in lists[label] if _on_topic(p, q)]
                if progress:
                    progress(label, len(lists.get(label, [])), round(time.monotonic() - t0, 1), err)
    papers = merge(lists)
    for p in papers:                                    # rank by topical match first, fused rank within a tier
        p["relevance"] = round(max((relevance(p, q) for q in queries), default=0.0), 2)
    papers.sort(key=lambda p: (-round(p["relevance"], 1), -p["score"]))
    return {"papers": papers, "failed": failed, "asked": len(lists), "stopped": stopped}


# --- full text -----------------------------------------------------------------------------------------------
def pdf(url: str) -> bytes:
    """An open-access PDF's bytes; a link that serves something else (a landing page, a bot check) is refused."""
    data = get_bytes(url, cache=False, timeout=90)
    if not data.startswith(b"%PDF"):
        raise ValueError(f"{url} did not serve a PDF (a landing page or a bot check?)")
    return data


def pdf_pages(data: bytes, layout: bool = False) -> list:
    """The text of each page (pdftotext ends every page with a form feed). `layout` keeps the columns and table
    alignment; without it the text is in reading order, which is what a quote is checked against."""
    with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
        f.write(data)
        f.flush()
        done = subprocess.run(["pdftotext", *(["-layout"] if layout else []), f.name, "-"], capture_output=True,
                              timeout=120)
    if done.returncode != 0:
        raise ValueError(done.stderr.decode("utf-8", "replace")[:200] or "pdftotext failed")
    pages = done.stdout.decode("utf-8", "replace").split("\f")
    return pages[:-1] if len(pages) > 1 and not pages[-1].strip() else pages
