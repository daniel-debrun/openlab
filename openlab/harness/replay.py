"""Offline replay: every reference, probe and delivered run of a refinement step is logged as a node of a tree
(.openlab/harness/tree.jsonl): its step and task, the harness it ran and that harness's parent, its probe signals,
its deliverable and, when a task was graded later, its score. A refinement *policy* (when to keep the base harness,
which probe signal to trust, how many candidates to try) can then be screened against the recorded steps with no new
rollout: the policy picks a node of each step, and its replayed value is the mean score of what it picked.

The incumbent policy (keep the base harness) is always screened too, so a policy chosen this way is never worse than
the incumbent on the recorded history. That is all replay can say: it scores only branches that were recorded, so it
cannot credit a behaviour no step tried. It is a pre-screen for which policy to spend live rollouts on, never an
acceptance (openlab.harness.evaluate accepts).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

KINDS = ("reference", "probe", "deliver")


def _path(root) -> Path:
    from .versions import home
    return home(root) / "tree.jsonl"


def log(root, node: dict) -> dict:
    """Append a node: {step, task, kind (reference | probe | deliver), harness, parent, agent, signals, deliverable,
    usd, score (optional)}."""
    if node.get("kind") not in KINDS or not node.get("step"):
        raise ValueError(f"a tree node has a step and a kind of {', '.join(KINDS)}")
    p = _path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(node, sort_keys=True, default=str) + "\n")
    return node


def nodes(root) -> list:
    p = _path(root)
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.is_file() else []


def score(root, step: str, harness: str, value: float) -> int:
    """Attach a later grade to a step's nodes of `harness`; how many it scored."""
    rows, n = nodes(root), 0
    for r in rows:
        if r["step"] == step and r["harness"] == harness:
            r["score"], n = float(value), n + 1
    _path(root).write_text("".join(json.dumps(r, sort_keys=True, default=str) + "\n" for r in rows), encoding="utf-8")
    return n


def steps(rows: list) -> dict:
    out: dict = {}
    for r in rows:
        out.setdefault(r["step"], []).append(r)
    return out


# --- policies: (a step's nodes) -> the node it would have shipped ------------------------------------------------
def keep_base(step: list) -> dict | None:
    return next((r for r in step if r["kind"] == "reference"), None)


def by_signal(field: str, better: str = "low") -> Callable:
    """The probe with the best value of `field`, if it beats the reference's; else the reference."""
    def pick(step: list) -> dict | None:
        ref = keep_base(step)
        val = lambda r: (r.get("signals") or {}).get(field)                               # noqa: E731
        probes = [r for r in step if r["kind"] == "probe" and val(r) is not None]
        if not probes or ref is None or val(ref) is None:
            return ref
        best = min(probes, key=val) if better == "low" else max(probes, key=val)
        return best if (val(best) < val(ref) if better == "low" else val(best) > val(ref)) else ref
    pick.__name__ = f"by_signal:{field}"
    return pick


def delivered(step: list) -> dict | None:
    """What the step actually shipped."""
    return next((r for r in step if r["kind"] == "deliver"), keep_base(step))


def screen(rows: list, policies: dict | None = None) -> list:
    """Each policy's replayed value: the mean score of the nodes it picks over the steps where its pick was graded;
    best first. keep_base (the incumbent) is always among them."""
    from .signals import FIELDS
    policies = dict(policies or {})
    policies.setdefault("keep_base", keep_base)
    policies.setdefault("delivered", delivered)
    for f, (better, _) in FIELDS.items():
        if better in ("low", "high"):
            policies.setdefault(f"by_signal:{f}", by_signal(f, better))
    out = []
    for name, pick in policies.items():
        got = [p for p in (pick(s) for s in steps(rows).values()) if p is not None and p.get("score") is not None]
        out.append({"policy": name, "steps": len(got),
                    "value": round(sum(p["score"] for p in got) / len(got), 4) if got else None})
    return sorted(out, key=lambda r: (r["value"] is None, -(r["value"] or 0), r["policy"] != "keep_base"))
