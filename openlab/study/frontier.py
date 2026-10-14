"""The research frontier: what to do next, derived from the other structures on every call
and never stored. Each item names the node to act on (`target`), why, the verb that acts on it, its estimated cost
and the information it is expected to give; items rank by information per hour of cost. `Study.take` writes an
action node linking an item's targets, so the loop's choices are on the record, and leases the item to the agent
that took it: another agent's take is refused (naming the holder) until the holder releases it, ends, or the lease
expires (`leases`). An approved proposal's open branches are items too (kind `branch`, verb `delegate`).

| kind | from | info | verb | cost |
|---|---|---|---|---|
| untested | a hypothesis with no test | 1 | test | run |
| untested | an in-scope question with no sub-question, hypothesis or claim | 1 | hypothesis | step |
| at_risk | a claim that needs a broken assumption, or rests on one that does | 1 | run | run |
| contradicted | a claim whose last contradiction search found evidence, and no person's decision since | 1 | approve | step |
| insufficient | a claim whose last contradiction search fell short of the review's floor | shortfall | contra | search |
| inconclusive | a claim or hypothesis whose tests are inconclusive | span | test | run |
| replication | an inconclusive replication | 1 | replicate | run |
| gap | an empty lattice cell: the declared coordinates by the first other text field | 1 | extract | search |
| stale_paper | the latest paper, with bindings stale or resting on a source retracted since | stale share | paper | run, or step |
| awaiting | the forecasts whose outcomes are not on record | 1 | fill | wait + run |
| branch | an open branch of the approved proposal (openlab.study.proposal) | 1 | delegate | its minutes |
| expand | a research run the search policy would grow (openlab.study.search) | P(pick) | run | the run's seconds |
| fuse | the best run of a line with no new best for `stagnation_branch` runs | 0.6 | run | its seconds |
| draft | the study, after `stagnation_global` runs with no new best | 0.5 | run | run |
| select | the study, once its clock reaches `[search] consolidate` | 1 | select | step |

Information is in [0, 1], 1 when nothing is known. `shortfall`: the mean, over the floor's requirements (queries per
index, hops), of the share not met. `span`: 2|diff| / (hi - lo) of the last test's paired interval, at most 1: near 1
a little more data decides it, near 0 the effect is too small to decide. `stale share`: the paper's bindings that are
stale or rest on a retracted source, over all of them. Budget: a `test` item is multiplied by the alpha-investing
wealth left (now over sealed, at most 1), and `test` and `run` items by 0 once the trial budget is used.

Cost is seconds: `run` the mean of the recorded research runs that were not cached (DEFAULT_RUN without one),
`search` the mean of the recorded searches, from their first query to their coverage (DEFAULT_SEARCH without one),
`step` one agent or person step (MIN_COST), `wait` until the first label is known; never below MIN_COST.

    score = info / (cost / 3600)        expected information per hour; ties keep the table's order
"""

from __future__ import annotations

import hashlib
from datetime import datetime

from . import clock
from . import claims as _claims, evidence as _evidence, graph, schema
from .errors import StudyError
from .paper import bindings

MIN_COST = 60.0
DEFAULT_RUN = 600.0
DEFAULT_SEARCH = 60.0
KINDS = ("untested", "at_risk", "contradicted", "insufficient", "inconclusive", "replication", "gap", "stale_paper",
         "awaiting", "branch", "select", "expand", "fuse", "draft")
TAKEN = ("kind", "target", "why", "verb", "info", "cost", "score")      # what an action taking an item records
LEASE_MINUTES = 30.0                                                    # how long a take holds an item
TEXT = 80                                                               # characters of a statement in `why`


def _item(kind: str, target: str, why: str, verb: str, info: float, cost: float, targets=(), key: str = "") -> dict:
    """An item; its id is stable while the record says the same thing about its target."""
    h = hashlib.sha256(f"{kind}|{target}|{key}".encode()).hexdigest()[:12]
    return {"id": f"frontier:{h}", "kind": kind, "target": target, "targets": list(targets) or [target], "why": why,
            "verb": verb, "info": info, "cost": cost}


def _cut(text: str) -> str:
    return text if len(text) <= TEXT else text[:TEXT - 3] + "..."


def _mean(xs: list, default: float) -> float:
    return sum(xs) / len(xs) if xs else default


def costs(rec) -> dict:
    """Seconds per run, search and step, from the record."""
    runs = [r.payload["seconds"] for r in rec.find("run", segment="research") if not r.payload["cached"]]
    spans = [clock.instant(c.at) - clock.instant(rec.get(q[0][2]).at)
             for c in rec.find("coverage") for q in [rec.links(c.id, rel="of")] if q]
    return {"run": max(MIN_COST, _mean(runs, DEFAULT_RUN)), "search": max(MIN_COST, _mean(spans, DEFAULT_SEARCH)),
            "step": MIN_COST}


def budget(study) -> dict:
    """The alpha-investing wealth now and as sealed, and the trials used of their budget: counted as the runner counts
    them (openlab.study.ledger.shared: distinct research runs and declared trials, of this study and its ledger)."""
    from .ledger import shared
    rec, spec = study.record, study.spec
    last = rec.last("test")
    used = shared(study.folder, rec)["trials"]
    return {"wealth": last.payload["wealth_after"] if last else spec.inference.get("wealth"),
            "initial": spec.inference.get("wealth"), "trials": {"used": used, "budget": spec.budget.get("trials", 0)}}


def _factor(verb: str, b: dict) -> float:
    if verb in ("test", "run") and b["trials"]["budget"] and b["trials"]["used"] >= b["trials"]["budget"]:
        return 0.0
    return max(0.0, min(1.0, b["wealth"] / b["initial"])) if verb == "test" and b["initial"] else 1.0


# --- the sources of items, in the table's order ---------------------------------------------------------------
def _branch_questions(study) -> set:
    """The question nodes of the approved proposal's branches: a branch is leased as a whole (its `branch` item)."""
    node = study.record.last("proposal")
    if node is None:
        return set()
    tree = node.payload["tree"]
    return {tree[b["id"]] for b in node.payload["proposal"]["branches"] if b["id"] in tree}


def _tree(study, c: dict) -> list:
    out, branches = [], _branch_questions(study)

    def walk(n: dict) -> None:
        if n["kind"] == "hypothesis" and n["status"] == "untested":
            out.append(_item("untested", n["id"], f"hypothesis `{_cut(n['text'])}` has no test", "test", 1, c["run"]))
        if n["kind"] == "question" and n["status"] == "untested" and not n["children"] and not n["claims"] \
                and n["id"] not in branches:
            out.append(_item("untested", n["id"], f"question `{_cut(n['text'])}` has no hypothesis or claim",
                             "hypothesis", 1, c["step"]))
        for k in n.get("children", []):
            walk(k)
    for q in _claims.tree(study)["questions"]:
        walk(q)
    return out


def _at_risk(study, c: dict) -> list:
    return [_item("at_risk", cid, f"it needs broken assumption {', '.join(a)}", "run", 1, c["run"])
            for cid, a in _claims.at_risk(study.record).items()]


def _shortfall(cov: dict) -> float:
    floor = cov["floor"]
    met = [min(1.0, n / floor["queries_per_source"]) for n in cov["queries"].values()]
    met += [min(1.0, cov["hops"] / floor["hops"])] if floor["hops"] else []
    return 1 - _mean(met, 0.0)


def _contra(study, c: dict) -> list:
    rec, out = study.record, []
    for claim in rec.find("claim"):
        searches = [s for s, _, _ in rec.links(claim.id, rel="for", incoming=True) if s.startswith("contra_search:")]
        if not searches:
            continue
        last = rec.get(searches[-1])
        if last.payload["state"] == "found":
            if any(rec.get(a).seq > last.seq for a, _, _ in rec.links(claim.id, rel="on", incoming=True)):
                continue                                # a person decided on the claim since (approvals are a person's)
            found = [d for _, _, d in rec.links(last.id, rel="found")]
            out.append(_item("contradicted", claim.id, f"its contradiction search found {', '.join(found)} and no "
                             "person has decided since", "approve", 1, c["step"], [claim.id, *found], last.id))
        elif last.payload["state"] == "insufficient":
            cov = rec.get(rec.links(last.id, rel="searched")[0][2]).payload
            short = _shortfall(cov)
            out.append(_item("insufficient", claim.id, f"its contradiction search met {1 - short:.0%} of the review's "
                             "floor", "contra", short, c["search"], key=last.id))
    return out


def _inconclusive(study, c: dict) -> list:
    rec, last, out = study.record, {}, []
    for t in rec.find("test"):
        last[rec.links(t.id, rel="of")[0][2]] = t
    for target, t in last.items():
        if schema.tested(rec, [target]) != "inconclusive":
            continue
        d = rec.get(rec.links(t.id, rel="from")[0][2]).payload["paired"].get(t.payload["baseline"], {})
        width = (d.get("hi") or 0) - (d.get("lo") or 0)
        span = min(1.0, 2 * abs(d["diff"]) / width) if d.get("diff") is not None and width > 0 else 1.0
        out.append(_item("inconclusive", target, f"its tests are inconclusive (last: p {t.payload['p']:.3g} against "
                         f"{t.payload['baseline']})", "test", span, c["run"], key=t.id))
    return out


def _replications(study, c: dict) -> list:
    return [_item("replication", r.id, "the replication is inconclusive", "replicate", 1, c["run"])
            for r in study.record.find("replication", verdict="inconclusive")]


def _gaps(study, c: dict) -> list:
    review = study.spec.review
    coords = review.get("coordinates") if review else None
    by = next((k for k, (t, _) in schema.extraction(review).items() if t == "text" and k not in coords), None) \
        if coords else None
    if by is None:
        return []
    out = []
    for row in _evidence.lattice(study, by)["rows"]:
        ev = [e["evidence"] for cells in row["cells"].values() for e in cells]
        where = ", ".join(f"{k} {v}" for k, v in row["coordinates"].items())
        out += [_item("gap", ev[0], f"no evidence for {by} {col} at {where}", "extract", 1, c["search"], ev,
                      f"{where}|{col}") for col in row["gaps"]]
    return out


def _paper(study, c: dict) -> list:
    rec = study.record
    paper = rec.last("paper")
    if paper is None:
        return []
    now, bound = bindings.current(rec, study.folder), paper.payload["bindings"]
    stale = {b for b in bound if bindings.stale(rec, rec.get(b), now)}
    gone = graph.retracted_since(study, paper)
    hit = stale | {b for g in gone for b in g["bindings"]}
    if not hit:
        return []
    why = ", ".join(([f"{len(stale)} stale bindings"] if stale else []) +
                    [f"{g['source']} retracted since the build" for g in gone])
    return [_item("stale_paper", paper.id, why, "paper", len(hit) / len(bound), c["run"] if stale else c["step"],
                  key=",".join(sorted(hit)))]


def _awaiting(study, c: dict) -> list:
    rec = study.record
    due = [f for f in rec.find("forecast") if not rec.links(f.id, rel="of", incoming=True)]
    if not due:
        return []
    first = min(due, key=lambda f: f.payload["label_at"])
    wait = max(0.0, (datetime.fromisoformat(first.payload["label_at"]) - clock.utc()).total_seconds())
    return [_item("awaiting", first.id, f"{len(due)} forecasts await their outcomes; the first label is known at "
                  f"{first.payload['label_at']} (UTC)", "fill", 1, wait + c["run"])]


def _branches(study, c: dict) -> list:
    from . import proposal
    return proposal.frontier_items(study, _item, c)


def _search(study, c: dict) -> list:
    from . import search
    return search.frontier_items(study, _item, c)


SOURCES = (_tree, _at_risk, _contra, _inconclusive, _replications, _gaps, _paper, _awaiting, _branches, _search)


def frontier(study, n: int | None = 20) -> dict:
    """The ranked items (the first `n`; None for all), how many there are, the budget and costs they were scored
    with, and the sources of items that could not be read (`skipped`, with why)."""
    rec, c, b = study.record, costs(study.record), budget(study)
    items, skipped = [], {}
    for make in SOURCES:
        try:
            items += make(study, c)
        except StudyError as e:
            skipped[make.__name__.strip("_")] = e.error
    taken = {a.payload["args"].get("item"): a.id for a in rec.find("action", verb="take") if a.payload["outcome"]["ok"]}
    held = leases(study)
    for i in items:
        i["info"] = round(i["info"] * _factor(i["verb"], b), 4)
        i["cost"] = round(max(MIN_COST, i["cost"]), 1)
        i["score"] = round(i["info"] * 3600 / i["cost"], 4)
        i["taken"] = taken.get(i["id"])
        i["held_by"] = {k: held[i["id"]][k] for k in ("agent", "expires")} if i["id"] in held else None
    order = {k: j for j, k in enumerate(KINDS)}
    items.sort(key=lambda i: (-i["score"], order[i["kind"]], i["target"]))
    return {"items": items if n is None else items[:n], "total": len(items), "budget": b,
            "costs": {k: round(v, 1) for k, v in c.items()}, **({"skipped": skipped} if skipped else {})}


def _ended(root, agent: str) -> bool:
    """The agent has ended (done, failed, stopped, lost) in the workspace's control store."""
    try:
        from ..agents.store import FINAL, Store
        if not (root / ".openlab" / "control.db").exists():
            return False
        a = Store(root).get(agent)
        return bool(a) and a["status"] in FINAL
    except Exception:                                   # noqa: BLE001 -- no store: the lease stands until it expires
        return False


def leases(study) -> dict:
    """{item id: {agent, expires, action}}: the frontier items leased now. `Study.take` leases an item to the agent
    that takes it (taking it again renews); the lease ends when that agent releases it (`Study.release`), when the
    agent ends (its status in the control store), or when it expires."""
    from ..agents.store import root_of
    rec = study.record
    acts = sorted(rec.find("action", verb="take") + rec.find("action", verb="release"), key=lambda n: n.seq)
    held: dict = {}
    for a in acts:
        if not a.payload["outcome"].get("ok"):
            continue
        args = a.payload["args"]
        if a.payload["verb"] == "release":
            held.pop(args.get("item"), None)
        elif args.get("lease"):
            held[args["item"]] = {**args["lease"], "action": a.id}
    now, root = clock.utc().isoformat(timespec="seconds"), root_of(study.folder)
    if held and (root / ".openlab" / "control.db").exists():
        try:                                            # a killed session's leases free once its silence ends it
            from ..agents.store import Store
            Store(root).reap()
        except Exception:                               # noqa: BLE001 -- the leases stand until they expire
            pass
    return {i: h for i, h in held.items() if h["expires"] > now and not _ended(root, h["agent"])}


def lease_until(minutes: float) -> str:
    from datetime import timedelta
    return (clock.utc() + timedelta(minutes=minutes)).isoformat(timespec="seconds")


def item(study, id: str) -> dict:
    """One item on the frontier now, by id; refused when it is not there."""
    found = next((i for i in frontier(study, None)["items"] if i["id"] == id), None)
    if found is None:
        raise StudyError(f"{id} is not on the frontier", "it was done, or the record changed since it was listed",
                         "study_frontier lists the items now")
    return found
