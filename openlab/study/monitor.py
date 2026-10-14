"""Continuous research: a monitoring pass over a study, recorded as a `watch` node, and its
place in the user's crontab.

`openlab study monitor` runs the sealed review's discovery again (a source already on record is a duplicate the
schema refuses, so what it records is new), checks each source that evidence quotes or a paper cites for retraction (Crossref update
notices on its DOI, OpenAlex's flag on its W id) and records each notice as a `retraction` node, fills the outcomes
of the forecasts whose labels are known, and writes the `watch` node. Each new retraction's impact (the evidence,
claims, questions and papers resting on the source: graph.depends_on_source) is in the reply and, with the rest of
the news, in one notification: `notify.deliver` (email when SMTP is set, else `.openlab/outbox.jsonl`), and the
desktop as well with `notify = "desktop"`. A pass with no news notifies nothing.

    [monitor]
    cron = "0 6 * * 1"      when `openlab study monitor` runs, once installed (`openlab study monitor <folder> --schedule on`)
    notify = "outbox"       outbox | desktop

A source known to be retracted is not checked again. Contradiction searches are not re-run: their state is the
agent's reading of what they found, so the pass reports the evidence contradiction searches found since the last one.
"""

from __future__ import annotations

import re

from . import notify
from ..sources import papers
from . import graph, literature, schema
from . import spec as _spec
from .errors import StudyError
from .schema import Retraction, Watch

SHOWN = 20                  # ids per list a reply or a notification names
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def watched_sources(rec) -> list:
    """The sources a conclusion may rest on and not known to be retracted: those evidence quotes or a paper cites."""
    ids = {e.payload["source"] for e in rec.find("evidence")}
    ids |= {d for p in rec.find("paper") for _, _, d in rec.links(p.id, rel="renders") if d.startswith("source:")}
    return [s for s in (rec.get(i) for i in sorted(ids)) if not schema.retracted(rec, s)]


def _ask(fn, arg):
    """The index's answer, or None when it did not answer (recorded as not checked, never as clean)."""
    try:
        return fn(arg)
    except Exception:                                   # noqa: BLE001
        return None


def notices(sources: list) -> tuple:
    """({source id: [checks that answered]}, [(source, Retraction)]): Crossref's update notices on each DOI that
    withdraw it, and OpenAlex's flag on each W id."""
    checked, found = {s.id: [] for s in sources}, []
    dois = {s.payload["doi"]: s for s in sources if s.payload["doi"]}
    updates = _ask(papers.crossref_updates, list(dois)) if dois else None
    if updates is not None:
        for doi, s in dois.items():
            checked[s.id].append("crossref")
            found += [(s, Retraction(via="crossref", type=u["type"], notice=u["notice"],
                                     valid_from=u["date"] if _DATE.fullmatch(u["date"]) else ""))
                      for u in updates.get(doi, []) if u["type"] in schema.RETRACTION_TYPES]
    works = {s.payload["openalex"]: s for s in sources if s.payload["openalex"]}
    for w in (_ask(papers.works, list(works)) or []) if works else []:
        s = works.get(w["openalex"])
        if s is None:                                   # a work OpenAlex no longer serves is not checked
            continue
        checked[s.id].append("openalex")
        if w.get("retracted"):
            found.append((s, Retraction(via="openalex", type=schema.OPENALEX_FLAG, notice=w["openalex"])))
    return checked, found


def _n(k: int, word: str) -> str:
    return f"{k} {word}{'' if k == 1 else 's'}"


def _news(news: dict, impact: list) -> list:
    """The notification's lines: {alert, interpretation} per piece of news."""
    out = []
    if news["new_sources"]:
        k = len(news["new_sources"])
        out.append({"alert": f"{_n(k, 'new source')} {'matches' if k == 1 else 'match'} the protocol",
                    "interpretation": ", ".join(news["new_sources"][:SHOWN])})
    for i in impact:
        out.append({"alert": f"{i['key']} was retracted: {_n(len(i['claims']), 'claim')} and "
                             f"{_n(len(i['papers']), 'paper')} depend on it",
                    "interpretation": f"{i['note']}; claims {', '.join(i['claims'][:SHOWN]) or 'none'}; papers "
                                      f"{', '.join(i['papers']) or 'none'}"})
    if news["contradicting"]:
        out.append({"alert": f"contradiction searches found {_n(len(news['contradicting']), 'piece')} of evidence",
                    "interpretation": ", ".join(news["contradicting"][:SHOWN])})
    if news["outcomes"]:
        out.append({"alert": f"{_n(len(news['outcomes']), 'forecast outcome')} filled",
                    "interpretation": ", ".join(news["outcomes"][:SHOWN])})
    return out


def _impact(study, r) -> dict:
    rec = study.record
    src = rec.get(rec.links(r.id, rel="of")[0][2])
    dep = graph.depends_on_source(study, src.id)
    return {"retraction": r.id, "source": src.id, "key": schema.source_key(src.payload),
            "note": schema.retracted(rec, src), "valid_from": r.payload["valid_from"],
            **{k: dep[k] for k in ("evidence", "claims", "conclusions", "papers")}}


def watch(study) -> dict:
    """One monitoring pass, recorded as a watch node; the reply has the news, each new retraction's impact, the
    sources no check answered for, the failed calls, and how the notification was delivered (None: no news)."""
    rec = study.record
    spec = _spec.check(study.folder, rec)
    cov, errors = None, []
    if spec.review:
        d = literature.discover(study)
        cov, errors = rec.get(d["id"]), list(d["errors"])
    checked, found = notices(watched_sources(rec))
    for src, body in found:
        if any(r.payload["notice"] == body.notice for r in schema.retractions(rec, src)):
            continue                                    # the index listed the notice twice
        try:
            rec.add(body, actor="openlab", links=[("of", src.id)])
        except StudyError as e:
            errors.append(f"{src.id}: {e.error}")
    if rec.find("forecast"):
        from . import forward
        try:
            forward.fill(study)
        except StudyError as e:
            errors.append(f"fill: {e.error}")
    news = schema.watched(rec, cov)
    made = news.pop("retractions")
    node = rec.add(Watch(**news, checked=checked), actor="openlab",
                   links=([("of", cov.id)] if cov else []) + [("found", r) for r in made])
    impact = [_impact(study, rec.get(r)) for r in made]
    lines = _news(news, impact)
    summary = "; ".join(x["alert"] for x in lines)
    delivery = None
    if lines:
        delivery = notify.deliver(spec.name, lines, rec.dir / "outbox.jsonl", subject=f"openlab monitor of {spec.name}: "
                                  f"{summary}")
        extra = notify.NOTIFY[spec.monitor["notify"]]
        delivery += f"; {extra(f'openlab: {spec.name}', summary)}" if extra else ""
    return {"id": node.id, "summary": summary or "nothing new", "coverage": cov and cov.id,
            "new_sources": news["new_sources"], "retractions": impact, "contradicting": news["contradicting"],
            "outcomes": news["outcomes"], "checked": len(checked),
            "unchecked": [s for s, c in checked.items() if not c], "errors": errors, "delivery": delivery}


def schedule(study, action: str = "status") -> dict:
    """The study's `[monitor] cron` line in the user's crontab: `status`, `on` (install or update) or `off`."""
    from . import schedule as SC
    from .forward import crontab
    out = crontab(study, action, "monitor", SC.for_monitor, "0 6 * * 1")
    last = study.record.last("watch")
    out["last_watch"] = last and {"id": last.id, "at": last.at}
    return out
