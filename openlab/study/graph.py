"""Read views over the record, for agents and the workspace UI; each is derived on call and never stored.

- `neighbourhood(study, id, depth)`: the nodes within `depth` links of a node and the typed edges between them.
- `trace(study, claim)`: the claim's provenance chain as a tree (claim -> evidence -> document -> source; claim ->
  result -> run -> datasets and plan, with the run's code and environment; a hypothesis -> its tests), each link
  with its relation, and what is missing along it.
- `traceability(study)`: the share of claims whose chain is complete and what each other
  claim is missing. A chain is broken where a ground would be refused today (evidence of a source retracted since,
  unless the claim says so or a person waived it), where a hypothesis has no test, or where a run records no code
  or environment.
- `depends_on_source(study, source)`: the evidence, claims, comparisons, papers and questions resting on a source
  (the impact of its retraction).
"""

from __future__ import annotations

from dataclasses import asdict

from . import schema
from .errors import StudyError
from .show import brief

NEIGHBOURS_MAX = 200        # nodes one neighbourhood carries; `withheld` counts the others it reached
DEPTH_MAX = 3
#: what a node rests on, by kind: (relation, followed against its direction)
CHAIN = {"claim": (("rests_on", False), ("tests", False), ("searched", False)), "evidence": (("from", False),),
         "document": (("of", False),), "result": (("of", False),), "run": (("uses", False), ("under", False)),
         "hypothesis": (("of", True),), "test": (("from", False),)}
#: the relations along which a conclusion rests on a source: the evidence quoting it and its document, the claims
#: resting on those, comparisons of the evidence and the papers rendering any of them
RESTS = ("of", "from", "rests_on", "compares", "renders")


def _brief(n) -> dict:
    return brief(asdict(n))


def neighbourhood(study, id: str, depth: int = 1) -> dict:
    """The nodes within `depth` links of `id` (either direction; action nodes left out, `study_log` has them) and
    the edges between them, at most NEIGHBOURS_MAX nodes."""
    rec, depth = study.record, max(0, min(depth, DEPTH_MAX))
    seen, todo, left = {id: rec.get(id)}, [id], set()
    for _ in range(depth):
        nxt = []
        for i in todo:
            for src, _, dst in rec.links(i) + rec.links(i, incoming=True):
                other = dst if src == i else src
                if other in seen or other.startswith("action:"):
                    continue
                if len(seen) >= NEIGHBOURS_MAX:
                    left.add(other)
                    continue
                seen[other] = rec.get(other)
                nxt.append(other)
        todo = nxt
    edges = {(s, r, d) for i in seen for s, r, d in rec.links(i) if d in seen}
    return {"center": id, "depth": depth, "nodes": [_brief(n) for n in sorted(seen.values(), key=lambda n: n.seq)],
            "edges": [{"src": s, "rel": r, "dst": d} for s, r, d in sorted(edges)],
            **({"withheld": len(left)} if left else {})}


def _facts(rec, n) -> dict:
    """What the chain shows of a node beyond its line: a run's code and environment, a dataset's hash, a source's
    id and retraction."""
    p = n.payload
    if n.kind == "run":
        return {"code": {k: v[:12] for k, v in p["code"].items()}, "env": p["env"], "audit": p["audit"]["verdict"]}
    if n.kind == "dataset":
        return {"sha256": p["sha256"][:12], "recipe_sha": p["recipe_sha"][:12]}
    if n.kind == "source":
        return {"key": schema.source_key(p), "retracted": schema.retracted(rec, n)}
    return {}


def _missing(rec, n) -> list:
    """What breaks the chain at this node."""
    if n.kind == "claim":
        out = []
        for _, _, g in rec.links(n.id, rel="rests_on"):
            try:
                schema._grounded(rec, n.payload["statement"], rec.get(g))
            except StudyError as e:
                out.append(f"{n.id} rests on {g}: {e.error} ({e.meaning})")
        return out
    if n.kind == "hypothesis" and not schema.tested(rec, [n.id]):
        return [f"{n.id} has no test"]
    if n.kind == "run":
        return [f"{n.id} records no {w}" for w, k in (("code", "code"), ("environment", "env")) if not n.payload[k]]
    return []


def _chain(rec, n, seen: set) -> tuple:
    """(tree, missing) below `n`; a node met twice is shown once, then by id."""
    out, missing = {**_brief(n), **_facts(rec, n), "links": []}, _missing(rec, n)
    for rel, back in CHAIN.get(n.kind, ()):
        ids = [s for s, _, _ in rec.links(n.id, rel=rel, incoming=True) if s.startswith("test:")] if back \
            else [d for _, _, d in rec.links(n.id, rel=rel)]
        for i in ids:
            if i in seen:
                out["links"].append({"rel": "tested_by" if back else rel, "node": {"id": i, "shown": "above"}})
                continue
            seen.add(i)
            sub, miss = _chain(rec, rec.get(i), seen)
            out["links"].append({"rel": "tested_by" if back else rel, "node": sub})
            missing += miss
    return out, missing


def _claim(rec, id: str):
    n = rec.get(id)
    if n.kind != "claim":
        raise StudyError(f"{id} is a {n.kind}, not a claim", fix="study_claims lists the claims")
    return n


def chain(rec, claim: str) -> tuple:
    """(tree, missing) of the claim's provenance chain, from the record alone."""
    c = _claim(rec, claim)
    return _chain(rec, c, {c.id})


def trace(study, claim: str) -> dict:
    """The claim's provenance chain as a tree, whether it is complete, and what is missing."""
    rec = study.record
    c = _claim(rec, claim)
    tree, missing = _chain(rec, c, {c.id})
    return {"claim": c.id, "statement": c.payload["statement"], "complete": not missing, "missing": missing,
            "chain": tree}


def traceability(study) -> dict:
    """The share of claims whose chain is complete, the same for the claims a person approved (the gate
    needs 100%), and each broken claim with what it is missing."""
    from .paper.lint import approved
    rec = study.record
    rows = [(c, _chain(rec, c, {c.id})[1], approved(rec, c)) for c in rec.find("claim")]
    ok = [r for r in rows if not r[1]]
    share = lambda k, n: round(k / n, 4) if n else None                           # noqa: E731
    return {"claims": len(rows), "traceable": len(ok), "share": share(len(ok), len(rows)),
            "approved": {"claims": sum(r[2] for r in rows), "traceable": sum(r[2] for r in ok),
                         "share": share(sum(r[2] for r in ok), sum(r[2] for r in rows))},
            "broken": [{"claim": c.id, "statement": c.payload["statement"], "missing": m} for c, m, _ in rows if m]}


def depends_on_source(study, source: str) -> dict:
    """What rests on a source: the evidence quoting it, the claims resting on that evidence (and on those claims),
    comparisons of the evidence, the papers rendering any of them or citing the source, and the questions and
    hypotheses those claims answer or test (`conclusions`)."""
    rec = study.record
    if rec.get(source).kind != "source":
        raise StudyError(f"{source} is not a source", fix="study_sources lists the sources")
    nodes = rec.impact(source, rels=RESTS)
    of = lambda kind: [n.id for n in nodes if n.kind == kind]                     # noqa: E731
    claims = of("claim")
    return {"source": source, "evidence": of("evidence"), "claims": claims, "comparisons": of("comparison"),
            "papers": of("paper"),
            "conclusions": sorted({d for c in claims for _, rel, d in rec.links(c) if rel in ("answers", "tests")})}


def retracted_since(study, paper) -> list:
    """The retractions recorded after `paper` of sources it rests on, each with the paper's bindings resting on
    the source (the source itself when cited)."""
    rec, out = study.record, []
    bound = set(paper.payload["bindings"])
    for r in rec.find("retraction"):
        if r.seq < paper.seq:
            continue
        src = rec.links(r.id, rel="of")[0][2]
        dep = depends_on_source(study, src)
        if paper.id in dep["papers"]:
            p = r.payload
            out.append({"retraction": r.id, "source": src,
                        "note": schema.retraction_note(p["via"], p["type"], p["notice"], p["valid_from"]),
                        "bindings": sorted(bound & {src, *dep["evidence"], *dep["claims"], *dep["comparisons"]})})
    return out
