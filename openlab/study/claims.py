"""Claims and how the research is organised: the question tree, the error budget (alpha-investing), the assumption
register, the dead-ends registry and contradiction searches.

The schema bodies (`Claim`, `Question`, `Hypothesis`, `Test`, `Assumption`, `Break`, `DeadEnd`, `ContraSearch`,
`Approval`) hold every invariant. This module fills in the fields openlab computes, writes the nodes, and derives the
views (the tree, the claims with their standing, the claims at risk), which are never stored.
"""

from __future__ import annotations

from . import literature, schema
from . import spec as _spec
from .errors import StudyError
from .record import Node, Record
from .schema import Approval, Assumption, Break, Claim, ContraSearch, DeadEnd, Hypothesis, Question

STATUS = {"supported": "answered", "refuted": "refuted", "inconclusive": "inconclusive"}   # a hypothesis's, by tests


def _nodes(rec: Record, pairs: list) -> dict:
    """{rel: [Node]} for (rel, id) pairs, as the schema's checks see them."""
    out: dict = {}
    for rel, i in pairs:
        out.setdefault(rel, []).append(rec.get(i))
    return out


def claim(study, statement: str, state: str, rests_on: list = (), value: float | None = None, expression: str = "",
          inputs: dict | None = None, tests: str = "", searched: str = "", needs: list = (),
          answers: list = ()) -> Node:
    """A claim, with the verdict and level openlab computes from what it rests on."""
    rec = study.record
    pairs = [("rests_on", i) for i in rests_on] + [("tests", tests)] * bool(tests) + \
        [("searched", searched)] * bool(searched) + [("needs", i) for i in needs] + [("answers", i) for i in answers]
    return rec.add(Claim(statement=statement, state=state, value=value, expression=expression, inputs=inputs or {},
                         **schema.claim_fields(rec, statement, state, _nodes(rec, pairs))), links=pairs)


def question(study, text: str, part_of: str = "", out_of_scope: str = "") -> Node:
    return study.record.add(Question(text=text, out_of_scope=out_of_scope),
                            links=[("part_of", part_of)] if part_of else [])


def hypothesis(study, statement: str, prediction: str, falsified_by: str, question: str, tags: list,
               mechanism: str = "", differs_from: list = (), difference: str = "") -> Node:
    return study.record.add(Hypothesis(statement=statement, prediction=prediction, mechanism=mechanism,
                                       falsified_by=falsified_by, tags=list(tags), difference=difference),
                            links=[("part_of", question)] + [("differs_from", d) for d in differs_from])


def test(study, of: str, result: str, baseline: str, alpha_j: float, family: str = "primary") -> Node:
    """Test a claim or hypothesis on a result's paired difference against `baseline` (one of its baselines, or
    `result:<id>`: another of the study's results), at level alpha_j, spending the error budget: p comes from the
    result (or both results' shared rows) and the ledger from the previous test."""
    rec = study.record
    p = schema.paired_p(rec.get(result), baseline, rec)
    return rec.add(schema.Test(family=family, baseline=baseline, alpha_j=alpha_j, p=p,
                               **schema.alpha_step(rec, alpha_j, p)),
                   links=[("of", of), ("from", result), *([("against", baseline)] if baseline.startswith("result:")
                                                          else [])])


def assume(study, statement: str, rests_on: str, breaks_if: str) -> Node:
    return study.record.add(Assumption(statement=statement, rests_on=rests_on, breaks_if=breaks_if))


def break_(study, assumption: str, reason: str, shown_by: list) -> Node:
    return study.record.add(Break(reason=reason), links=[("breaks", assumption)] + [("shown_by", s) for s in shown_by])


def dead_end(study, of: str, statement: str, reason: str, tags: list, shown_by: list) -> Node:
    return study.record.add(DeadEnd(statement=statement, reason=reason, tags=list(tags)),
                            links=[("of", of)] + [("shown_by", s) for s in shown_by])


def approve(study, on: str, decision: str, reason: str, person: bool = False) -> Node:
    """A person's approval, rejection or waiver; the schema refuses any other actor."""
    return study.record.add(Approval(decision=decision, reason=reason), actor="person" if person else "agent",
                            links=[("on", on)])


def _claim_node(rec: Record, id: str) -> Node:
    n = rec.get(id)
    if n.kind != "claim":
        raise StudyError(f"{id} is a {n.kind}, not a claim", fix="study_claims lists the claims")
    return n


def contra(study, claim: str, queries: list = (), found: list = ()) -> dict:
    """Search for what contradicts a claim and record the outcome. With `queries` (the agent's, recorded as queries
    `for` the claim) openlab runs them on the review's indexes, hops from the sources the claim's evidence quotes,
    and judges the coverage against the review's floor; without, the claim's last search is reused (to record the
    evidence `found` against it after reading). The state follows from the coverage and `found`."""
    rec = study.record
    c = _claim_node(rec, claim)
    search: dict = {}
    if queries:
        _spec.check(study.folder, rec)
        start = [rec.get(s) for s in sorted({n.payload["source"] for n in rec.why(c.id) if n.kind == "evidence"})]
        search = literature.search(rec, schema._review(rec), list(queries), start, claim=c)
        cov = rec.get(search["id"])
    else:
        last = [rec.get(s) for s, _, _ in rec.links(c.id, rel="for", incoming=True) if s.startswith("contra_search:")]
        if not last:
            raise StudyError(f"{c.id} has no contradiction search yet", fix="give the queries to search with")
        cov = rec.get(rec.links(last[-1].id, rel="searched")[0][2])
    node = rec.add(ContraSearch(state=schema.contra_state(cov, list(found))),
                   links=[("for", c.id), ("searched", cov.id)] + [("found", e) for e in found])
    return {"id": node.id, "state": node.payload["state"], "coverage": cov.id, **({"search": search} if search else {})}


# --- derived views ------------------------------------------------------------------------------------------
def at_risk(rec: Record) -> dict:
    """{claim id: [broken assumption ids]}: the claims that need a broken assumption, and every claim resting on
    them."""
    out: dict = {}
    for b in rec.find("break"):
        a = rec.links(b.id, rel="breaks")[0][2]
        for n in rec.impact(a, rels=("needs", "rests_on")):
            if n.kind == "claim" and a not in out.setdefault(n.id, []):
                out[n.id].append(a)
    return out


def _incoming(rec: Record, id: str, rel: str, kind: str) -> list:
    return [s for s, _, _ in rec.links(id, rel=rel, incoming=True) if s.startswith(f"{kind}:")]


def claims(study) -> list:
    """Every claim with its standing now: the verdict (tests written since count; `verdict_written` when it
    differs), level (reproductions written since count; `level_written` when it differs), the broken assumptions that
    put it at risk, the last person's decision and contradiction search."""
    rec = study.record
    risk = at_risk(rec)
    from .ledger import shared
    now = shared(rec.folder, rec)["trials"]
    rows = []
    for c in rec.find("claim"):
        p, verdict, level = c.payload, schema.current_verdict(rec, c), schema.current_level(rec, c)
        approval, contra_ = _incoming(rec, c.id, "on", "approval"), _incoming(rec, c.id, "for", "contra_search")
        rows.append({"id": c.id, "statement": p["statement"], "state": p["state"], "verdict": verdict,
                     **({"verdict_written": p["verdict"]} if verdict != p["verdict"] else {}), "level": level,
                     **({"level_written": p["level"]} if level != p["level"] else {}),
                     **({"level_note": note} if (note := schema.level_note(rec, c)) else {}),
                     "value": p["value"], "at_risk": risk.get(c.id, []),
                     "selected_from": p.get("selected_from", 0), "trials_now": now,
                     "approval": rec.get(approval[-1]).payload["decision"] if approval else None,
                     "contra": rec.get(contra_[-1]).payload["state"] if contra_ else None})
    return rows


def _combine(statuses: list) -> str:
    """A parent claims no more than its weakest child: untested while any child is, then inconclusive; once every
    child is decided, answered (refuted when every child was refuted)."""
    weakest = next((s for s in ("untested", "inconclusive") if s in statuses), None)
    return weakest or ("refuted" if set(statuses) == {"refuted"} else "answered")


def tree(study) -> dict:
    """The question tree, derived and never stored: a hypothesis's status from its tests, a question's from its
    in-scope children, and a question with none from the claims that answer it (answered when one is supported)."""
    rec = study.record

    def hypothesis_(h: Node) -> dict:
        return {"id": h.id, "kind": "hypothesis", "text": h.payload["statement"],
                "status": STATUS.get(schema.tested(rec, [h.id]), "untested"),
                "tests": _incoming(rec, h.id, "of", "test"), "dead_ends": _incoming(rec, h.id, "of", "dead_end"),
                "claims": _incoming(rec, h.id, "tests", "claim") + _incoming(rec, h.id, "answers", "claim")}

    def question_(q: Node) -> dict:
        kids = [question_(n) if n.kind == "question" else hypothesis_(n)
                for n in (rec.get(s) for s, _, _ in rec.links(q.id, rel="part_of", incoming=True))]
        answering = _incoming(rec, q.id, "answers", "claim")
        scoped = [k["status"] for k in kids if k["status"] != "out_of_scope"]
        if q.payload["out_of_scope"]:
            status = "out_of_scope"
        elif scoped:
            status = _combine(scoped)
        else:
            verdicts = [schema.current_verdict(rec, rec.get(c)) for c in answering]
            status = "answered" if "supported" in verdicts else "inconclusive" if verdicts else "untested"
        return {"id": q.id, "kind": "question", "text": q.payload["text"], "status": status, "children": kids,
                "claims": answering}

    return {"questions": [question_(q) for q in rec.find("question") if not rec.links(q.id, rel="part_of")]}
