"""The literature engine, part A: discovery to the sealed review protocol, full text by page, screening, and the
PRISMA flow, all on the study record.

openlab asks the indexes and records what they returned; the agent reads and judges, and a person settles what
the agent marks maybe. Deduplication, coverage, quotes and who may overrule whom are invariants of the schema
bodies (`Query`, `Source`, `Document`, `Coverage`, `Screen`): this module feeds them and never re-checks them.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from ..sources import papers
from . import schema
from . import spec as _spec
from .errors import StudyError
from .record import Node, Record
from .schema import Coverage, Document, Query, Recall, Screen, Source

RETRACTED = schema.RETRACTION_TYPES
STATUSES = ("all", "unscreened", "include", "exclude", "maybe", "conflict", "retracted")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _under(rec: Record) -> str:
    return (rec.last("amendment") or rec.last("plan")).id


def _err(e: Exception) -> str:
    return f"{type(e).__name__}: {str(e)[:200]}"


def _ids(p: dict) -> dict:
    """A paper's ids in stored form (one that does not parse is dropped). The DOI an arXiv record names is its
    journal version's, so it is that preprint's `published_as`, not its own id."""
    raw = {"doi": "" if p.get("journal_doi") else p.get("doi", ""), "arxiv": p.get("arxiv", ""),
           "openalex": p.get("openalex", "")}
    ids = {k: schema.norm_id(k, v) for k, v in raw.items() if v}
    return {k: v for k, v in ids.items() if schema.valid_id(k, v)}


def _remember(index: dict, n: Node) -> None:
    """Index a source by its ids, the DOI it was published as and its title (for the version links)."""
    p = n.payload
    for k in schema.IDS:
        if p.get(k):
            index.setdefault((k, p[k]), n)
    if p.get("published_as"):
        index.setdefault(("published_as", p["published_as"]), n)
    title = papers._norm_title(p["title"])
    if len(title) >= 20:
        index.setdefault(("title", title), n)


def _index(rec: Record) -> dict:
    index: dict = {}
    for n in rec.find("source"):
        _remember(index, n)
    return index


def _flags(batch: list, known=None) -> list:
    """(retracted, note, checks) per (ids, paper): OpenAlex's is_retracted where the paper came from OpenAlex,
    and Crossref's update notices on its DOI. A check that did not answer is not listed as run. `known`: the
    notices a search already fetched ({"notices", "checked"}: openlab.study.literature._Calls), asked only for the
    DOIs it did not check."""
    dois = [ids["doi"] for ids, _ in batch if "doi" in ids]
    have = known or {"notices": {}, "checked": set()}
    rest = [d for d in dois if d.lower() not in have["checked"]]
    try:
        notices, crossref = (papers.crossref_updates(rest) if rest else {}), True
    except Exception:                                   # noqa: BLE001 -- recorded as not checked, never as clean
        notices, crossref = {}, not rest
    notices = {**{d: have["notices"].get(d.lower(), []) for d in dois if d.lower() in have["checked"]}, **notices}
    out = []
    for ids, p in batch:
        checks, notes = [], []
        if "retracted" in p:
            checks.append("openalex")
            notes += [schema.retraction_note("openalex", "", "", "")] if p["retracted"] else []
        if "doi" in ids and crossref:
            checks.append("crossref")
            notes += [schema.retraction_note("crossref", u["type"], u["notice"], u["date"])
                      for u in notices.get(ids["doi"], []) if u["type"] in RETRACTED]
        out.append((bool(notes), "; ".join(notes), checks))
    return out


def _source(ids: dict, p: dict, index_name: str, flags: tuple) -> Source:
    kind = p.get("version") or ("preprint" if index_name == "arxiv" else "published")
    published_as = schema.norm_id("doi", p.get("journal_doi", "")) if kind == "preprint" else ""
    date = p.get("date", "") if _DATE.fullmatch(p.get("date", "")) else ""
    retracted, note, checks = flags
    return Source(title=p.get("title", ""), version_kind=kind, tier=schema.TIER_OF[index_name], retracted=retracted,
                  retraction_checked=checks, **ids,
                  published_as=published_as if schema.valid_id("doi", published_as) else "",
                  authors=list(p.get("authors", [])), year=int(date[:4]) if date else p.get("year"),
                  venue=p.get("venue", ""), published=date, retraction_note=note, abstract=p.get("abstract", ""),
                  url=p.get("url", ""), pdf=p.get("pdf", ""))


def _version_of(index: dict, src: Source) -> list:
    """A link to the other version on record: the published source this preprint names, the preprint that names
    this DOI, or failing both a source of the other kind with the same title."""
    found = [index.get(("doi", src.published_as)) if src.published_as else None,
             index.get(("published_as", src.doi)) if src.doi else None,
             index.get(("title", papers._norm_title(src.title)))]
    other = next((n for n in found if n is not None and n.payload["version_kind"] != src.version_kind), None)
    return [("version_of", other.id)] if other else []


def _record(rec: Record, index: dict, under: str, source: str, text: str, filters: dict, page: dict, found: list,
            error: str = "", seed: Node | None = None, claim: Node | None = None, known=None) -> tuple:
    """Write one query (`for` a claim in a contradiction search), then a source for each identity it returned that
    is not on record. Returns the query, the sources its hits resolve to, and how many were new."""
    hits = [(ids, p) for ids, p in ((_ids(p), p) for p in found) if ids]
    if len(hits) < len(found):
        page = {**page, "no_id": len(found) - len(hits)}           # returned with no id openlab can store
    q = rec.add(Query(source=source, text=text, filters=filters, n_hits=len(hits), hits=[i for i, _ in hits],
                      page=page, error=error), actor="openlab",
                links=[("under", under)] + ([("seed", seed.id)] if seed else [])
                + ([("for", claim.id)] if claim else []))
    new = [(ids, p) for ids, p in hits if schema.resolve(index, ids) is None]
    written = 0
    for (ids, p), flags in zip(new, _flags(new, known)):
        if schema.resolve(index, ids) is not None:                  # the same work twice in one answer
            continue
        body = _source(ids, p, source, flags)
        node = rec.add(body, actor="openlab", links=[("found_by", q.id)] + _version_of(index, body))
        _remember(index, node)
        written += 1
    return q, [schema.resolve(index, ids) for ids, _ in hits], written


def _lookup(seed: Node) -> str:
    """The id OpenAlex finds a seed by fastest: its W id, else its DOI, else its arXiv id."""
    p = seed.payload
    return p["openalex"] or p["doi"] or p["arxiv"]


def discover(study, progress=None) -> dict:
    """Run the sealed review's searches and record them: every protocol query on every protocol index, then
    `citation_hops` hops of references and citing works published since `review.since` (each direction from each
    seed is one query), a source for each identity not yet on record, and the coverage. The seeds are the sources
    the previous level reached, and once any source is screened include, only included sources seed (the first
    hop from all of them): snowballing follows what the review kept, not everything a query returned.
    `progress(line)` is told one line per call as it returns (see `search`)."""
    rec = study.record
    _spec.seal(study.folder, rec, "agent")
    review = _spec.check(study.folder, rec).review
    if not review:
        raise StudyError("study.toml has no [review]", "discovery runs the sealed review protocol",
                         "add [review] with question, sources, queries, include, exclude and coverage_floor")
    out = search(rec, review, review["queries"], included(rec), progress=progress)
    out.pop("sources")
    return out


def included(rec: Record) -> list:
    """The sources screening includes."""
    return [rec.get(i) for i, s in decisions(rec).items() if s.payload["decision"] == "include"]


WORKERS = int(os.environ.get("OPENLAB_DISCOVER_WORKERS") or 6)   # index calls in flight at once


class _Calls:
    """The index calls of one search: run on a bounded thread pool (each host's queue in openlab.sources.http still
    paces them and retries a 429 with backoff), the same call asked twice answered once, every call timed, and
    `progress` told one line per call as it returns."""

    def __init__(self, progress=None):
        self.progress, self.cache, self.timings, self.reused = progress, {}, [], 0
        self.lock = threading.Lock()
        self.known = {"notices": {}, "checked": set()}         # Crossref update notices, fetched beside the calls

    def _check(self, got) -> None:
        """Fetch the Crossref update notices of a call's hits now, in the worker, so recording asks for none."""
        hits = got if isinstance(got, list) else [x for v in (got or {}).values() if isinstance(v, list) for x in v]
        dois = sorted({i["doi"].lower() for i in (_ids(p) for p in hits if isinstance(p, dict)) if i.get("doi")})
        with self.lock:
            dois = [d for d in dois if d not in self.known["checked"]]
        if not dois:
            return
        try:
            notices = papers.crossref_updates(dois)
        except Exception:                                   # noqa: BLE001 -- recording asks again
            return
        with self.lock:
            self.known["notices"].update(notices)
            self.known["checked"].update(dois)

    def run(self, kind: str, jobs: list) -> list:
        """[(key, label, fn)] -> [(result, error)] in order; identical keys run once."""
        todo = {k: (label, fn) for k, label, fn in jobs if k not in self.cache}
        total, done = len(jobs), [0]

        def one(item):
            k, (label, fn) = item
            t0 = time.monotonic()
            try:
                got, err = fn(), ""
            except Exception as e:                          # noqa: BLE001 -- recorded as a failed call
                got, err = None, _err(e)
            secs = time.monotonic() - t0
            if got:
                self._check(got)
            with self.lock:
                self.cache[k] = (got, err)
                done[0] += 1
                hits = (sum(len(v) for v in got.values() if isinstance(v, list)) if isinstance(got, dict)
                        else len(got or []))
                self.timings.append({"call": label, "kind": kind, "hits": hits, "seconds": round(secs, 2),
                                     **({"error": err} if err else {})})
                if self.progress:
                    head = label if kind == "queries" else f"{label} {done[0]}/{len(todo)}"
                    self.progress(f"{head}: " + (f"failed: {err[:80]}" if err else f"{hits} hits") + f" ({secs:.1f} s)")
        if todo:
            with ThreadPoolExecutor(max_workers=max(1, min(WORKERS, len(todo)))) as pool:
                list(pool.map(one, list(todo.items())))
        reused = total - len(todo)
        self.reused += reused
        if reused and self.progress:
            self.progress(f"{reused} identical {kind} answered from this run's earlier call")
        return [self.cache[k] for k, _, _ in jobs]


def search(rec: Record, review: dict, texts: list, start: list, claim: Node | None = None, progress=None) -> dict:
    """Record `texts` on every index of the review, then `citation_hops` hops of citation queries and the coverage
    of them all: a discovery (`texts` the protocol's) or a claim's contradiction search (every query `for` it). The
    first hop is seeded by what the texts reached and `start`; once `start` has any source, only sources in it seed.
    The calls run concurrently (`_Calls`); they are recorded in protocol order. Returns the coverage, counts, failed
    calls, the sources reached, and each call's timing with the wall time and what concurrency saved."""
    t0 = time.monotonic()
    under, index, n, since = _under(rec), _index(rec), review["n"], review["since"]
    queries, level, new, errors, seen = [], [], 0, [], {}
    calls = _Calls(progress)
    asks = [(src, text) for src in review["sources"] for text in texts]
    jobs = [((src, text, n, since), f"{src} q{i}/{len(asks)}", (lambda s=src, t=text: papers.ask(s, t, n, since)))
            for i, (src, text) in enumerate(asks, 1)]
    for (src, text), (found, err) in zip(asks, calls.run("queries", jobs)):
        if err:
            errors.append(f"{src} `{text[:60]}`: {err}")
        q, reached, k = _record(rec, index, under, src, text, {"since": since, "n": n},
                                {"page": 1, "per_page": n, "sort": "relevance"}, found or [], err, claim=claim,
                                known=calls.known)
        queries.append(q)
        level += reached
        seen.update(dict.fromkeys(s.id for s in reached))
        new += k
    keep = {s.id for s in start}
    level += start
    seeded: set = set()
    for hop in range(1, review["citation_hops"] + 1):
        seeds = [s for s in {s.id: s for s in level}.values() if s.id not in seeded and (not keep or s.id in keep)]
        level = []
        seeded.update(s.id for s in seeds)
        jobs = [(("related", _lookup(seed), n, since), f"citations hop {hop}",
                 (lambda key=_lookup(seed): papers.related(key, n=n, related_works=False, since=since)))
                for seed in seeds]
        for seed, (got, err) in zip(seeds, calls.run("citation lookups", jobs)):
            if err:
                errors.append(f"citations of {schema.source_key(seed.payload)}: {err}")
            got = got or {}
            for direction, sort in (("references", "as listed"), ("cited_by", "cited_by_count desc")):
                q, reached, k = _record(rec, index, under, "citations", schema.source_key(seed.payload),
                                        {"direction": direction, "hop": hop, "n": n, "since": since},
                                        {"limit": n, "sort": sort}, got.get(direction, []), err, seed, claim,
                                        known=calls.known)
                queries.append(q)
                level += reached
                seen.update(dict.fromkeys(s.id for s in reached))
                new += k
    cov = rec.add(Coverage(**schema.coverage(rec, queries)), actor="openlab",
                  links=[("under", under)] + [("of", q.id) for q in queries])
    wall = time.monotonic() - t0
    serial = sum(t["seconds"] for t in calls.timings)
    return {"id": cov.id, "coverage": cov.payload, "queries": len(queries), "new_sources": new, "errors": errors,
            "sources": list(seen), "timings": calls.timings, "seconds": round(wall, 1),
            "calls_seconds": round(serial, 1), "saved_seconds": round(max(0.0, serial - wall), 1),
            "cached_calls": calls.reused}


def _source_node(rec: Record, source: str) -> Node:
    n = rec.get(source)
    if n.kind != "source":
        raise StudyError(f"{source} is a {n.kind}, not a source", fix="study_sources lists the sources")
    return n


def document(rec: Record, src: Node) -> Node | None:
    """The source's full text on record (the latest document), or None."""
    docs = [rec.get(d) for d, _, _ in rec.links(src.id, rel="of", incoming=True)]
    return next((d for d in reversed(docs) if d.kind == "document"), None)


def fetch_text(study, source: str) -> Node:
    """The source's open-access PDF as a document of page texts (pdftotext, in reading order and as laid out):
    its PDF link, else its arXiv PDF when the link serves no PDF. A source fetched before returns its document."""
    rec = study.record
    src = _source_node(rec, source)
    doc = document(rec, src)
    if doc is not None:
        return doc
    p = src.payload
    urls = list(dict.fromkeys(u for u in (p["pdf"], p["arxiv"] and f"https://arxiv.org/pdf/{p['arxiv']}") if u))
    if not urls:
        raise StudyError(f"{src.id} has no open-access PDF link", "openlab reads open-access full text only",
                         "screen it on its abstract")
    failed = []
    for url in urls:
        try:
            data = papers.pdf(url)
            texts = _texts(data)
            break
        except Exception as e:                          # noqa: BLE001 -- the reply says which link failed and why
            failed.append(f"{url}: {_err(e)}")
    else:
        raise StudyError(f"the full text of {src.id} could not be read", "; ".join(failed))
    return pdf_document(rec, src, data, url, texts)


def _texts(data: bytes) -> tuple:
    return papers.pdf_pages(data), papers.pdf_pages(data, layout=True)


def pdf_document(rec: Record, src: Node, data: bytes, url: str, texts: tuple = ()) -> Node:
    """The source's document from a PDF's bytes (fetched from `url`, or a file the study was given): its page texts
    in reading order and as laid out."""
    pages, layout = texts or _texts(data)
    blob = lambda texts: rec.put_blob(json.dumps(texts).encode("utf-8"))      # noqa: E731
    return rec.add(Document(pages=blob(pages), n_pages=len(pages), method="pdftotext", pdf=rec.put_blob(data),
                            url=url, layout=blob(layout)),
                   actor="openlab", links=[("of", src.id)])


def screen(study, source: str, decision: str, criterion: str, rationale: str, quote: str = "",
           person: bool = False) -> Node:
    """Record a screening decision on one source under the sealed review; the schema refuses a criterion of the
    wrong kind, a quote not in the source, an agent overruling a person and a maybe from a person."""
    rec = study.record
    _spec.check(study.folder, rec)
    src = _source_node(rec, source)
    return rec.add(Screen(decision=decision, criterion=criterion, rationale=rationale, quote=quote),
                   actor="person" if person else "agent", links=[("screens", src.id), ("under", _under(rec))])


def screening(rec: Record) -> tuple:
    """({source id: the screen that stands}, {source id: [the conflicting screens]}): schema.screen_standing per
    source. Agents that disagree leave no standing decision until a person decides."""
    by: dict = {}
    for s in rec.find("screen"):
        by.setdefault(rec.links(s.id, rel="screens")[0][2], []).append(s)
    stands, clashes = {}, {}
    for src, screens in by.items():
        node, clash = schema.screen_standing(screens)
        if node is not None:
            stands[src] = node
        if clash:
            clashes[src] = clash
    return stands, clashes


def decisions(rec: Record) -> dict:
    """{source id: the screen that stands} (a conflicted source has none)."""
    return screening(rec)[0]


def _option(n: Node) -> dict:
    return {"agent": n.session, "decision": n.payload["decision"], "criterion": n.payload["criterion"],
            "screen": n.id, "quote": n.payload["quote"]}


def flag_conflict(study, node: Node):
    """When the screen `node` makes its source's screens conflict, file a `screen_conflict` inbox item for a person
    (in the workspace's control store); returns the item id, or None. A person's screen instead cancels the source's
    pending conflict items."""
    rec = study.record
    src = rec.links(node.id, rel="screens")[0][2]
    screens = [rec.get(s) for s, _, _ in rec.links(src, rel="screens", incoming=True)]
    try:
        from ..agents.store import Store, root_of
        store = Store(root_of(study.folder))
    except Exception:                                   # noqa: BLE001 -- no control store: the record still says it
        return None
    where = _study_ref(store.root, study.folder)
    waiting = [i for i in store.inbox("pending") if i["kind"] in ("screen_conflict", "screen_maybe")
               and i["study"] == where and (i["args"] or {}).get("source") == src]
    pending = [i for i in waiting if i["kind"] == "screen_conflict"]
    if node.actor == "person":
        from ..agents.inbox import DECIDING
        for i in (i for i in waiting if i["id"] != DECIDING.get()):
            store.decide(i["id"], "cancelled", node.session, f"a person decided {src} ({node.id})")
        return None
    if node.payload["decision"] == "maybe":
        if any(i["kind"] == "screen_maybe" for i in waiting):
            return None
        s = rec.get(src).payload
        key = schema.source_key(s)
        return store.request(node.session, "screen_maybe", f"{node.session} cannot decide {key}: "
                             f"{node.payload['rationale']}", study=where,
                             args={"source": src, "key": key, "title": s.get("title", ""), "screen": node.id,
                                   "criterion": node.payload["criterion"], "quote": node.payload["quote"],
                                   "rationale": node.payload["rationale"]},
                             cost_note="the source waits for a person: approve with --pick include or --pick exclude")
    clash = schema.screen_standing(screens)[1]
    if not clash or pending:
        return None
    key = schema.source_key(rec.get(src).payload)
    reason = f"agents disagree on {key}: " + "; ".join(f"{n.session} {n.payload['decision']} "
                                                      f"({n.payload['criterion']})" for n in clash)
    return store.request(node.session, "screen_conflict", reason, study=where,
                         args={"source": src, "options": [_option(n) for n in clash]},
                         cost_note="no decision stands on this source until a person picks one")


def _study_ref(root, folder) -> str:
    try:
        return str(folder.resolve().relative_to(root))
    except ValueError:
        return str(folder.resolve())


def prisma(study) -> dict:
    """The PRISMA flow of the discovery the latest coverage judged (every discovery query, before any coverage;
    contradiction searches are not part of it), derived from
    the record and never stored: records identified per index and by citation searching, duplicates removed,
    records screened, excluded by criterion, waiting for a person, included, and the retracted sources (and
    which are included). Sources only an earlier search found are counted apart (`earlier_only`)."""
    rec = study.record
    discovery = [q for q in rec.find("query") if not rec.links(q.id, rel="for")]   # not a contradiction search's
    ids = {q.id for q in discovery}
    cov = next((c for c in reversed(rec.find("coverage")) if rec.links(c.id, rel="of")[0][2] in ids), None)
    queries = [rec.get(d) for _, _, d in rec.links(cov.id, rel="of")] if cov else discovery
    index, by, reached = schema.sources_by_id(rec), Counter(), {}
    for q in queries:
        by[q.payload["source"]] += q.payload["n_hits"]
        for h in q.payload["hits"]:
            s = schema.resolve(index, h)
            if s is not None:                           # None only for a discovery cut short before its sources
                reached[s.id] = s
    stands, clashes = screening(rec)
    final = {i: d for i, d in stands.items() if i in reached}
    clashes = {i: c for i, c in clashes.items() if i in reached}
    counts = Counter(s.payload["decision"] for s in final.values())
    excluded = Counter(s.payload["criterion"] for s in final.values() if s.payload["decision"] == "exclude")
    retracted = [i for i, s in reached.items() if schema.retracted(rec, s)]
    identified = sum(by.values())
    agents = Counter(s.session for s in rec.find("screen") if rec.links(s.id, rel="screens")[0][2] in reached)
    return {"coverage": cov.id if cov else None, "identified": identified, "by_source": dict(by),
            "duplicates_removed": identified - len(reached), "records": len(reached),
            "screened": counts["include"] + counts["exclude"], "awaiting_person": counts["maybe"] + len(clashes),
            "conflicts": len(clashes), "screened_by": dict(agents),
            "not_screened": len(reached) - len(final) - len(clashes), "excluded": counts["exclude"],
            "excluded_by_criterion": dict(excluded), "included": counts["include"], "retracted": len(retracted),
            "retracted_included": [i for i in retracted if i in final and final[i].payload["decision"] == "include"],
            "earlier_only": len(rec.find("source")) - len(reached)}


def sources(study, status: str = "all") -> list:
    """The sources, oldest first, with their standing decision; `status` is one of STATUSES."""
    if status not in STATUSES:
        raise StudyError(f"no status `{status}`", f"one of {', '.join(STATUSES)}")
    rec = study.record
    final, clashes = screening(rec)
    texts = {rec.links(d.id, rel="of")[0][2] for d in rec.find("document")}
    rows = []
    for s in rec.find("source"):
        p, f, retracted = s.payload, final.get(s.id), bool(schema.retracted(rec, s))
        decision, clash = f.payload["decision"] if f else "", clashes.get(s.id, [])
        if status == "all" or status == decision or (status == "unscreened" and not f and not clash) or \
                (status == "retracted" and retracted) or (status == "conflict" and clash):
            rows.append({"id": s.id, "key": schema.source_key(p), "title": p["title"], "year": p["year"],
                         "version": p["version_kind"], "tier": p["tier"], "retracted": retracted,
                         "decision": decision, "by": f.actor if f else "", "agent": f.session if f else "",
                         "conflict": [{k: v for k, v in _option(n).items() if k != "quote"} for n in clash],
                         "criterion": f.payload["criterion"] if f else "",
                         "rationale": f.payload["rationale"] if f else "", "quote": f.payload["quote"] if f else "",
                         "full_text": s.id in texts})
    return rows


def recall(study, gold: str, since: str | None = None) -> dict:
    """Recall of the study's sources against a gold set: the references (OpenAlex referenced_works) of a published
    review, by its DOI, published from `since` (default: the review protocol's) up to the gold review's own date.
    Records the recall node; the reply adds how many matches screening included, and the references missed."""
    rec = study.record
    review = _spec.check(study.folder, rec).review
    if not review:
        raise StudyError("study.toml has no [review]", "recall measures the review's search")
    doi = schema.norm_id("doi", gold)
    if not schema.valid_id("doi", doi):
        raise StudyError(f"`{gold}` is not a DOI", fix="find the review's DOI with an OpenAlex search")
    try:
        w = papers.openalex_work(doi)
        refs = papers.works(w.get("referenced_works") or [])
    except Exception as e:                              # noqa: BLE001 -- the reply says what failed
        raise StudyError(f"the references of {doi} could not be read from OpenAlex", _err(e)) from None
    since = review["since"] if since is None else since
    until = w.get("publication_date") or ""
    kept: dict = {}
    for p in refs:
        date = p.get("date", "")
        if not date or ((not since or date >= since) and (not until or date <= until)):
            kept.setdefault(schema.source_key(_ids(p)), (_ids(p), p))
    left_out = {"not_served": len(w.get("referenced_works") or []) - len(refs), "outside_dates": len(refs) - len(kept)}
    references = [ids for ids, _ in kept.values()]
    matched = schema.recall_matches(rec, references)
    node = rec.add(Recall(gold=doi, title=w.get("title") or "", filters={"since": since, "until": until},
                          references=references, left_out=left_out, matched=matched,
                          recall=round(len(matched) / len(references), 4) if references else 0.0),
                   actor="openlab", links=[("under", _under(rec))])
    final = decisions(rec)
    included = [s for s in matched.values() if s in final and final[s].payload["decision"] == "include"]
    return {"id": node.id, "recall": node.payload["recall"], "gold": doi, "title": node.payload["title"],
            "references": len(references), "matched": len(matched), "matched_included": len(included),
            "left_out": left_out, "filters": node.payload["filters"],
            "missed": [{"key": k, "title": p.get("title", ""), "year": p.get("year")}
                       for k, (_, p) in kept.items() if k not in matched]}
