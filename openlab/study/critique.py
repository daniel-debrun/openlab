"""Critique: an LLM reviewer's findings on a run, result, claim or paper, recorded as advice. Nothing reads a critique
into a verdict: verdicts stay computed (after FM Agent's LLM-judge feedback and MLEvolve's review agents, kept
advisory).

A critique records who wrote it (the model and a sha of its prompt), what it cost, and its findings, each anchored
where it applies (`model.py:57`, a result, a claim). Each finding is then answered (`finding`):

- `checked`, naming the check that settled it (`by`): a module check, a run (its truncation audit), a comparison or
  a test. The finding is confirmed when that check shows the problem: a red module check, a run whose audit found
  look-ahead, a comparison or test that went against the side the finding defended, as recorded;
- `dismissed`, with the reason (and, when one exists, the node that shows why).

A critique's track record is computed from its answers: findings confirmed, not confirmed, dismissed, still open.
"""

from __future__ import annotations

from .errors import StudyError


def record(study, on: str, model: str, prompt_sha: str, findings: list, cost_usd: float | None = None):
    from .schema import Critique
    rec = study.record
    target = rec.get(on)
    items = []
    for f in findings:
        if isinstance(f, str):
            f = {"text": f}
        if not str(f.get("text", "")).strip():
            raise StudyError("a finding says what it found")
        items.append({"where": str(f.get("where", "")), "text": str(f["text"]).strip()})
    return rec.add(Critique(model=model, prompt_sha=prompt_sha, findings=items, cost_usd=cost_usd),
                   actor="agent", links=[("on", target.id)])


def answer(study, critique: str, index: int, action: str, reason: str = "", by: str = ""):
    from .schema import Finding
    rec = study.record
    links = [("answers", critique)] + ([("by", by)] if by else [])
    return rec.add(Finding(index=index, action=action, reason=reason), actor="agent", links=links)


def confirmed(rec, node) -> bool | None:
    """Whether the check `node` shows a problem (None when it cannot say)."""
    p = node.payload
    if node.kind == "module_check":
        return not p["ok"]
    if node.kind == "run":
        return p["audit"].get("verdict") == "look-ahead"
    if node.kind in ("comparison", "test"):
        return None
    return None


def track(study, critique_id: str) -> dict:
    rec = study.record
    c = rec.get(critique_id)
    n = len(c.payload["findings"])
    status = {i: "open" for i in range(n)}
    for f in rec.find("finding"):
        if rec.links(f.id, rel="answers")[0][2] != critique_id:
            continue
        i = f.payload["index"]
        if f.payload["action"] == "dismissed":
            status[i] = "dismissed"
            continue
        by = rec.links(f.id, rel="by")
        verdict = confirmed(rec, rec.get(by[0][2])) if by else None
        status[i] = {True: "confirmed", False: "not confirmed", None: "checked"}[verdict]
    counts = {k: sum(1 for v in status.values() if v == k)
              for k in ("confirmed", "not confirmed", "checked", "dismissed", "open")}
    return {"critique": critique_id, "findings": n, **counts, "by_finding": status}


def listing(study) -> list:
    rec = study.record
    return [{"id": c.id, "on": rec.links(c.id, rel="on")[0][2], "model": c.payload["model"],
             "findings": c.payload["findings"], "track": track(study, c.id)} for c in rec.find("critique")]
