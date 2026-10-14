"""Papers: a quick scan of the literature outside a review study (a planner's background, a lead's baselines).

The sources (OpenAlex, arXiv, Crossref; :mod:`openlab.sources.papers`) return a fused candidate list, saved as a
search in the workspace's ``papers/searches/<id>.json``; full texts read from open-access PDFs are kept in
``papers/text/``. A systematic review does not use these: its searches, screens and quotes go on the study record
(``study_discover``, ``study_screen``, ``study_extract``).
"""
from __future__ import annotations

import datetime as dt
import json
import re

from .registry import ToolContext, ToolError, tool

FOLDER = "papers"
ABSTRACT_CHARS = 700
TEXT_CHARS = 20000


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _search_file(ctx: ToolContext, sid: str):
    if not re.match(r"^[\w-]+$", sid):
        raise ToolError("not a search id", f"{sid!r}", "paper_search returns it as `id`")
    return ctx.path(f"{FOLDER}/searches/{sid}.json")


def _load(ctx: ToolContext, sid: str) -> dict:
    f = _search_file(ctx, sid)
    if not f.is_file():
        raise ToolError("no such search", f"{sid!r} is not saved", "paper_search returns it as `id`")
    return json.loads(f.read_text(encoding="utf-8"))


def _store(ctx: ToolContext, rec: dict) -> None:
    f = _search_file(ctx, rec["id"])
    f.parent.mkdir(parents=True, exist_ok=True)
    rec["updated"] = _now()
    f.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")


def _ordered(rec: dict) -> list:
    """Judged papers first by the agent's score, then the rest by topical relevance, then the fused rank score."""
    j = rec.get("judgments", {})
    return sorted(rec["candidates"], key=lambda p: (-(j[p["id"]]["score"] if p["id"] in j else -1),
                                                    -round(p.get("relevance", 0), 1), -p.get("score", 0)))


def _brief(p: dict, j: dict | None, abstracts: bool = True) -> dict:
    out = {k: p.get(k) for k in ("id", "title", "date", "venue", "cited_by", "sources", "pdf", "url")}
    out["authors"] = ", ".join(p.get("authors", [])[:4]) + (" et al." if len(p.get("authors", [])) > 4 else "")
    if p.get("new"):
        out["new"] = True
    if abstracts:
        a = p.get("abstract") or ""
        out["abstract"] = a[:ABSTRACT_CHARS] + ("…" if len(a) > ABSTRACT_CHARS else "")
    if j:
        out["judged"] = {"score": j["score"], "why": j["why"]}
    return out


@tool("research", mutates=True)
def paper_search(ctx: ToolContext, prompt: str, queries: str = "", since: str = "", n: int = 25,
                 open_access: bool = False, rerun: str = "", show: int = 30) -> dict:
    """Search OpenAlex (~250M works, every field), arXiv (newest preprints) and Crossref for papers on a prompt;
    each source is asked by relevance and by newest, and the lists are fused and deduplicated. Saved as a
    search (`id`) in papers/searches/: `paper_results` pages through it, `paper_related` follows a paper's
    citations, `paper_read` reads one.

    `queries` (separated by `;`) are extra phrasings to search besides the prompt (2-5: the key terms, synonyms,
    the method's and the field's names). `since` is YYYY-MM-DD. `rerun` re-asks a saved search, marking papers
    not seen before as `new`."""
    from openlab.sources import papers as P
    old = _load(ctx, rerun) if rerun else None
    if old:
        prompt, queries, since, open_access = old["prompt"], ";".join(old["queries"][1:]), old["since"], old["open_access"]
    if not prompt.strip():
        raise ToolError("no prompt", "say what papers you want", 'prompt="realized volatility forecasting with ML"')
    qs = [prompt.strip()] + [q.strip() for q in queries.split(";") if q.strip()]
    slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")[:40] or "search"
    sid = old["id"] if old else f"{dt.datetime.now():%Y%m%d-%H%M%S}-{slug}"
    r = P.search(qs, n=max(5, min(int(n), 100)), since=since, open_access=open_access)
    if not r["papers"] and r["failed"]:
        raise ToolError("every source failed", "; ".join(r["failed"])[:600],
                        "OpenAlex throttles anonymous search under load: retry in a minute, or set OPENALEX_API_KEY")
    seen = {p["id"] for p in old["candidates"]} if old else set()
    for p in r["papers"]:
        p["new"] = bool(old) and p["id"] not in seen
    if old:                                          # a paper found before and not now stays, with its judgment
        now = {p["id"] for p in r["papers"]}
        r["papers"] += [{**p, "new": False} for p in old["candidates"] if p["id"] not in now]
    rec = old or {"id": sid, "created": _now(), "judgments": {}}
    rec.update(prompt=prompt.strip(), queries=qs, since=since, open_access=open_access, failed=r["failed"],
               candidates=r["papers"], searched=_now(), stopped=r["stopped"])
    _store(ctx, rec)
    j = rec["judgments"]
    return {"id": rec["id"], "candidates": len(rec["candidates"]), "new": sum(p.get("new", False) for p in r["papers"]),
            "failed": r["failed"], "limits": P.LIMITS,
            "top": [_brief(p, j.get(p["id"])) for p in _ordered(rec)[:max(0, int(show))]]}


@tool("research")
def paper_results(ctx: ToolContext, search: str, n: int = 40, offset: int = 0, abstracts: bool = True) -> dict:
    """A saved search's candidates by fused rank (`offset`, `n`), with their abstracts unless `abstracts` is
    false."""
    rec = _load(ctx, search)
    j = rec.get("judgments", {})
    rows = _ordered(rec)
    page = rows[int(offset):int(offset) + max(1, int(n))]
    return {"id": rec["id"], "prompt": rec["prompt"], "queries": rec["queries"], "since": rec["since"],
            "searched": rec.get("searched"), "total": len(rows), "judged": len(j), "failed": rec.get("failed", []),
            "papers": [_brief(p, j.get(p["id"]), abstracts) for p in page]}


def _key(pid: str) -> str:
    for pre in ("doi:", "arxiv:", "openalex:"):
        if pid.startswith(pre):
            return pid[len(pre):]
    return pid


@tool("research", mutates=True)
def paper_related(ctx: ToolContext, id: str, n: int = 15, search: str = "") -> dict:
    """Snowball from one paper (`id` as the search shows it, a DOI, an arXiv id or an OpenAlex W-id): what it
    cites, what cites it (most cited first) and OpenAlex's related works. With `search`, the papers found are
    added to that search's candidates (source `citations`)."""
    from openlab.sources import papers as P
    try:
        r = P.related(_key(id), n=max(1, min(int(n), 50)))
    except KeyError as e:
        raise ToolError("not found", str(e).strip("'\""), "try its DOI") from None
    except Exception as e:                           # noqa: BLE001 -- the reply says which call failed
        raise ToolError("the lookup failed", f"{type(e).__name__}: {str(e)[:200]}", "retry in a minute") from None
    added = 0
    if search:
        rec = _load(ctx, search)
        have = {p["id"] for p in rec["candidates"]}
        for group in ("references", "cited_by", "related"):
            for p in r[group]:
                if p["id"] not in have:
                    rec["candidates"].append({**p, "sources": ["citations"], "score": 0.0, "new": True})
                    have.add(p["id"])
                    added += 1
        _store(ctx, rec)
    brief = lambda ps: [_brief(p, None) for p in ps]
    return {"paper": _brief(r["paper"], None), "references": brief(r["references"]),
            "cited_by": brief(r["cited_by"]), "related": brief(r["related"]),
            **({"added_to": search, "added": added} if search else {})}


@tool("research", mutates=True)
def paper_read(ctx: ToolContext, id: str, search: str = "", full_text: bool = False, offset: int = 0,
               chars: int = TEXT_CHARS) -> dict:
    """One paper whole: its record with the full abstract, and with `full_text` the text of its open-access PDF
    (`chars` from `offset`; kept in papers/text/ so a second read is local). `search` names where to find
    the record; without it the paper is looked up in OpenAlex."""
    from openlab.sources import papers as P
    p = None
    if search:
        p = next((x for x in _load(ctx, search)["candidates"] if x["id"] == id), None)
    if p is None:
        try:
            p = P._oa_paper(P.openalex_work(_key(id)))
        except Exception as e:                       # noqa: BLE001
            raise ToolError("not found", f"{id}: {type(e).__name__}", "pass the search it came from") from None
    out = {k: v for k, v in p.items() if k not in ("score", "new")}
    if not full_text:
        return out
    if not p.get("pdf"):
        raise ToolError("no open PDF", f"{id} has no open-access PDF link", "read the abstract, or open its url")
    txt = ctx.path(f"{FOLDER}/text/{re.sub(r'[^A-Za-z0-9._-]+', '_', id)}.txt")
    if not txt.is_file():
        try:
            data = P.pdf(p["pdf"])
        except Exception as e:                       # noqa: BLE001
            raise ToolError("the PDF did not download", f"{type(e).__name__}: {str(e)[:200]}", p["pdf"]) from None
        try:
            pages = P.pdf_pages(data, layout=True)
        except ValueError as e:
            raise ToolError("not a readable PDF", str(e), p.get("url", "")) from None
        txt.parent.mkdir(parents=True, exist_ok=True)
        txt.write_text("\f".join(pages) + "\f", encoding="utf-8")
    text = txt.read_text(encoding="utf-8", errors="replace")
    out.update(text=text[int(offset):int(offset) + int(chars)], text_chars=len(text), text_file=ctx.rel(txt))
    if int(offset) + int(chars) < len(text):
        out["next_offset"] = int(offset) + int(chars)
    return out
