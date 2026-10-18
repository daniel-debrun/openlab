"""Sealing a study, in plain words: what the seal freezes, what it costs, and whether a study is ready for it.

The seal is the study's first research run (openlab.study.runner.execute): study.toml is checked, the data is
materialised, a second study of a sealed question is refused, and then study.toml is written into the record as the
`plan` node; the run goes on and is counted as trial 1 of budget.trials. From then on an edit of study.toml is an
amendment (`openlab study amend`, with a reason), and an amendment that touches one of spec.PERSON_FIELDS needs the
person's signature. The holdout is not touched by the seal: only a person's look scores it.

`openlab study seal` and `study_seal` (and the Seal button of `openlab show`) run that first run on purpose, after
saying all this; `openlab study run` still seals implicitly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from . import spec as _spec
from .errors import StudyError

#: each of spec.PERSON_FIELDS in plain words (tests/test_study_seal.py keeps the two in step)
WHAT = {
    "split": "the split and holdout (research period, holdout, embargo)",
    "budget": "the budget (research trials, holdout looks)",
    "sandbox": "the sandbox (network, timeout, threads, cores, memory, GPUs)",
    "audit": "the look-ahead audit",
    "selection": "the rule that picks the final candidate",
    "forecast.gate": "the gate a forward forecast must pass",
    "evaluate.metrics[0]": "the primary metric",
    "data.*.known_at": "when each dataset's rows were known",
    "review.coverage_floor": "the review's coverage floor",
    "review.comparable_on": "what two review results must agree on to be compared",
    "inference": "the error budget tests spend (alpha-investing)",
    "study.fresh": "the fresh question declared (a trial budget and holdout of its own)",
    "metrics": "the study's own metrics (metrics.py, by its sha256)",
    "instruments": "the instrument versions the study pins",
    "data.*.instrument": "the instrument each dataset is read from",
}
COST = "the seal is the first research run: it is counted as trial 1 of the budget"
HOLDOUT = "the holdout is not touched: only a person's holdout look ever scores it"
AFTER = ("after the seal an edit of study.toml is an amendment (`openlab study amend STUDY --reason ...`); one that "
         "touches a protected field needs the person's signature (--person)")
WHY = ("sealing fixes the plan before any result is seen, so a result cannot be chosen by changing the split, the "
       "metric or the budget after the fact")


def _short(v, n: int = 90) -> str:
    s = v if isinstance(v, str) else json.dumps(v, default=str, sort_keys=True)
    return s if len(s) <= n else s[:n - 1] + "…"


def _split(raw: dict) -> str:
    sp = raw.get("split") or {}
    parts = [sp.get("scheme", "")]
    for k in ("research", "holdout"):
        if isinstance(sp.get(k), list) and len(sp[k]) == 2:
            parts.append(f"{k} {sp[k][0]} to {sp[k][1]}")
    if sp.get("fraction"):
        parts.append(f"holdout {sp['fraction']:g} of the rows (seed {sp.get('seed', 0)})")
    if sp.get("embargo"):
        parts.append(f"embargo {sp['embargo']}")
    return ", ".join(p for p in parts if p)


def _now(field: str, raw: dict) -> str:
    """The value a field holds in study.toml now, in one short line ('' when the study does not set it)."""
    if field == "split":
        return _split(raw)
    if field == "budget":
        b = raw.get("budget") or {}
        return f"{b.get('trials', 0)} trials, {b.get('holdout_looks', 1)} holdout look(s)" if b else ""
    if field == "evaluate.metrics[0]":
        return ((raw.get("evaluate") or {}).get("metrics") or [""])[0]
    if field == "metrics":
        return ", ".join(raw.get("metrics") or {})
    if field == "instruments":
        return ", ".join(f"{k} = {v}" for k, v in (raw.get("instruments") or {}).items())
    if field.startswith("data.*."):
        key = field.split(".")[-1]
        return ", ".join(f"{n}: {d[key]}" for n, d in (raw.get("data") or {}).items()
                         if isinstance(d, dict) and d.get(key))
    picked = _spec._pick(raw, _spec._tokens(field))
    return _short(next(iter(picked.values()))) if picked else ""


def terms(raw: dict) -> dict:
    """What sealing a study with spec `raw` (study.toml as a dict) freezes and costs, in plain words:
    {freezes: [{field, what, now}] (the protected fields it sets, and always the split, budget and primary metric),
    cost, holdout, after, why, trials, primary}."""
    always = ("split", "budget", "evaluate.metrics[0]")
    rows = []
    for f in _spec.PERSON_FIELDS:
        now = _now(f, raw)
        if now or f in always:
            rows.append({"field": f, "what": WHAT.get(f, f), "now": now or "(not set)"})
    b = raw.get("budget") or {}
    return {"freezes": rows, "cost": COST + (f" ({b['trials']})" if b.get("trials") else ""), "holdout": HOLDOUT,
            "after": AFTER, "why": WHY, "trials": b.get("trials", 0),
            "primary": ((raw.get("evaluate") or {}).get("metrics") or [""])[0]}


def lines(t: dict) -> list:
    """`terms` as the lines a person reads before the seal."""
    out = ["sealing freezes study.toml as the study's plan:"]
    out += [f"  {r['what']}: {r['now']}" for r in t["freezes"]]
    return out + [f"  and every other key of study.toml", t["cost"], t["holdout"], t["after"]]


def _by(node) -> str:
    return "the person" if node.actor == "person" else (node.session or node.actor)


def state(folder) -> dict:
    """{sealed, plan, at, by} of a study, read without creating its record."""
    folder = Path(folder)
    if not (folder / ".openlab" / "record.db").is_file():
        return {"sealed": False, "plan": "", "at": "", "by": ""}
    from .record import Record
    plan = next(iter(Record(folder).find("plan")), None)
    if plan is None:
        return {"sealed": False, "plan": "", "at": "", "by": ""}
    return {"sealed": True, "plan": plan.id, "at": plan.at, "by": _by(plan)}


def unsealed_line(rel: str) -> str:
    """The line an experiment or inbox item of an unsealed study carries."""
    return (f"the study {rel} is not sealed: its first research run seals study.toml (the split and holdout, budget, "
            f"instrument pins and primary metric) and spends trial 1; seal it with `openlab study seal {rel}` or the "
            "Seal button in openlab show")


def missing_data(study) -> list:
    """The datasets of the study's spec that are not materialised (no dataset node, or no file)."""
    from . import data
    rec = study.record
    return [n for n in study.spec.data if rec.last("dataset", name=n) is None or not data._path(study, n).is_file()]


def check(study, rel: str = "") -> dict:
    """The terms of sealing `study` now; a StudyError when it cannot be sealed: a review study, one already sealed
    (when and by which plan node), or one whose data is not materialised."""
    where = rel or study.folder.name
    plan = next(iter(study.record.find("plan")), None)
    if plan is not None:
        raise StudyError(f"{where} is already sealed: {plan.id} at {plan.at}, by {_by(plan)}",
                         "study.toml is frozen as that plan; a second seal would mean nothing",
                         f"`openlab study run {where}` for another counted trial; to change the plan, edit "
                         f"study.toml and record it with `openlab study amend {where} --reason ...` (the person "
                         "signs the protected fields with --person)")
    spec = _spec.load(study.folder)
    if spec.mode == "review":
        raise StudyError(f"{where} is a review study: its first discover seals the protocol, not a run",
                         "a review runs no experiment, so it has no first research run",
                         f"`openlab study discover {where}`")
    missing = missing_data(study)
    if missing:
        raise StudyError(f"the data of {where} is not materialised: {', '.join(missing)}",
                         "the seal is the first research run, and it runs on the registered data",
                         f"`openlab study materialise {where}` first, then seal")
    return terms(spec.raw)


def summary(out: dict) -> dict:
    """The seal's result in brief, for the page: trial, result id, the primary metric with its interval."""
    p = out.get("payload") or {}
    primary = next(iter(p.get("metrics") or {}), "")
    seal = out.get("seal") or {}
    return {"plan": seal.get("plan", ""), "result": out.get("id", ""), "trial": (out.get("trial") or {}).get("summary", ""),
            "metric": seal.get("primary") or primary,
            "value": ((p.get("metrics") or {}).get(seal.get("primary") or primary) or {})}


def overview(root, folders: Optional[list] = None) -> list:
    """Every study of the workspace with its seal: sealed ones {study, name, sealed, plan, at, by}; an unsealed one
    also `terms` and `ready` (and `blocked`, `fix` when it cannot be sealed now)."""
    from .workspace import studies
    base = Path(root).resolve()
    out = []
    for f in (folders if folders is not None else studies(root)):
        f = Path(f).resolve()
        try:
            rel = str(f.relative_to(base)) if base in f.parents else (f.name if f != base else ".")
        except ValueError:
            continue
        try:
            sp = _spec.load(f)
        except StudyError as e:
            out.append({"study": rel, "name": f.name, "sealed": False, "mode": "", "ready": False,
                        "blocked": e.error, "fix": e.fix})
            continue
        st = state(f)
        row = {"study": rel, "name": sp.name, "mode": sp.mode, **st}
        if not st["sealed"] and sp.mode != "review":
            row.update(terms=terms(sp.raw), ready=True, blocked="", fix="")
            try:
                if not (f / ".openlab" / "record.db").is_file():
                    raise StudyError(f"the data of {rel} is not materialised: {', '.join(sp.data)}",
                                     fix=f"`openlab study materialise {rel}` first, then seal")
                from . import open as _open
                check(_open(f), rel)
            except StudyError as e:
                row.update(ready=False, blocked=e.error, fix=e.fix)
        out.append(row)
    return out
