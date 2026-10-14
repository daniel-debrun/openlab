"""The search over a study's runs, and the rule that picks its final candidate.

Runs form a tree (`Run.op`, `branches_from`, `refs`). This module scores the tree on every call (nothing is stored)
and offers the frontier its next steps, after MARS (UCT with a cost-shaped reward) and MLEvolve (exploration that
decays with the study's clock, elite-guided picks, stagnation triggers):

    G      = a run's primary metric, min-max normalised over the study's research runs (0 for a run whose audit
             is not clean or that has no result; 0.5 when all are equal), higher is better
    Q      = mean G over the run's subtree;  n = its size;  N = all research runs
    U      = sqrt(ln(N + 1) / n)
    c(t)   = c_hi until 0.3, linear to c_lo at 0.7, then c_lo (t: the study's clock, below)
    cost   = (seconds / budget seconds) ** cost_weight          (cost_weight -0.07: a little in favour of quick runs)
    uct    = (Q + c(t) * U) * cost
    w(t)   = 1 until explore_until, linear to w_lo at explore_until + 0.2, then w_lo
    elite  = the top 3 runs by G, picked with p proportional to 1 / rank
    P      = w(t) * [the run with the highest uct] + (1 - w(t)) * elite p      (the chance this policy picks it)

The clock `t` is the larger of the trials used over budget.trials and, with `[search] minutes`, the minutes of
non-cached research runs over it. Frontier items (openlab.study.frontier): `expand` (improve a valid run, or debug one
whose audit is not clean), info = P; `fuse` when a branch has had `stagnation_branch` expansions without a new best
(improve its best with refs to the other branches' best); `draft` when the study has had `stagnation_global` runs
without a new best; and, from `consolidate` (0.85) on, no draft and a `select` item: apply the selection rule.

`[search]` in study.toml tunes it (all optional): minutes, explore_until (0.5), consolidate (0.85), c_hi (1.414),
c_lo (0.5), w_lo (0.2), cost_weight (-0.07), stagnation_branch (3), stagnation_global (6). The items are advice: an
agent still chooses, and `study take` records what it took.

The selection rule, `[selection] rule` (a person's field, sealed before the first run, so a study cannot pick its
rule after seeing its results): `best_val` (the best primary metric over the research runs) or `seed_robust` (the
configuration, runs alike but for the seed, with the best mean over its seeds; `k`, default 3, seeds at least when
any configuration has them). `study select` applies it and records a `selection`: the run picked, the N candidates,
and the picked value adjusted for being the best of N (`adjusted`: the value moved toward worse by its standard
error times the expected maximum of N standard normals, a winner's-curse correction). With a rule declared, a holdout
look is refused unless the code to be looked at is the selected run's.
"""

from __future__ import annotations

import math
from functools import lru_cache

from .errors import StudyError

DEFAULTS = {"explore_until": 0.5, "consolidate": 0.85, "c_hi": math.sqrt(2), "c_lo": 0.5, "w_lo": 0.2,
            "cost_weight": -0.07, "stagnation_branch": 3, "stagnation_global": 6}
RULES = ("best_val", "seed_robust")


def settings(spec) -> dict:
    return {**DEFAULTS, **(spec.raw.get("search") or {})}


def c_of(t: float, s: dict) -> float:
    if t < 0.3:
        return s["c_hi"]
    if t < 0.7:
        return s["c_hi"] + (s["c_lo"] - s["c_hi"]) * (t - 0.3) / 0.4
    return s["c_lo"]


def w_of(t: float, s: dict) -> float:
    a = s["explore_until"]
    if t < a:
        return 1.0
    if t < a + 0.2:
        return 1.0 - (1.0 - s["w_lo"]) * (t - a) / 0.2
    return s["w_lo"]


def _primary(spec) -> tuple:
    from .metrics import METRICS
    m = spec.evaluate["metrics"][0]
    meta = METRICS.get(m)
    hib = meta.higher_is_better if meta is not None else None
    return m, hib


def nodes(study) -> list:
    """The research runs with their result's primary value (None without a result), oldest first: {id, run, parent,
    refs, op, value, valid, seconds, lo, hi, key}."""
    rec, spec = study.record, study.spec
    m, _ = _primary(spec)
    out = []
    for r in rec.find("run", segment="research"):
        res = rec.links(r.id, rel="of", incoming=True)
        met = (rec.get(res[0][0]).payload["metrics"].get(m) or {}) if res else {}
        parent = rec.links(r.id, rel="branches_from")
        out.append({"id": r.id, "result": res[0][0] if res else None, "parent": parent[0][2] if parent else None,
                    "refs": [d for _, _, d in rec.links(r.id, rel="refs")], "op": r.payload.get("op", ""),
                    "value": met.get("value"), "lo": met.get("lo"), "hi": met.get("hi"),
                    "valid": r.payload["audit"].get("verdict") in ("clean", "separated", "in-sample", "unchecked")
                    and isinstance(met.get("value"), (int, float)),
                    "seconds": r.payload["seconds"], "key": r.payload["key"], "seed": r.payload.get("seed"),
                    "config": (r.payload["spec_sha"], tuple(sorted(r.payload["code"].items())),
                               tuple(sorted(r.payload["data"].items())))})
    return out


def clock(study, ns: list | None = None) -> float:
    from .ledger import shared
    spec, s = study.spec, settings(study.spec)
    ns = nodes(study) if ns is None else ns
    t = 0.0
    budget = spec.budget.get("trials") or 0
    if budget:
        t = shared(study.folder, study.record)["trials"] / budget
    if s.get("minutes"):
        t = max(t, sum(n["seconds"] for n in ns) / 60 / float(s["minutes"]))
    return min(1.0, t)


def score(study) -> dict:
    """The tree scored now: {t, c, w, rows: [{id, op, value, G, Q, n, U, cost, uct, elite, P}], best, stagnant,
    since_best} (rows by P, then uct)."""
    spec, s = study.spec, settings(study.spec)
    ns = nodes(study)
    t = clock(study, ns)
    if not ns:
        return {"t": t, "c": c_of(t, s), "w": w_of(t, s), "rows": [], "best": None, "stagnant": {}, "since_best": 0}
    _, hib = _primary(spec)
    sign = -1.0 if hib is False else 1.0
    vals = [sign * n["value"] for n in ns if n["valid"]]
    lo, hi = (min(vals), max(vals)) if vals else (0.0, 0.0)
    G = {n["id"]: 0.0 if not n["valid"] else 0.5 if hi == lo else (sign * n["value"] - lo) / (hi - lo) for n in ns}
    kids: dict = {}
    for n in ns:
        if n["parent"]:
            kids.setdefault(n["parent"], []).append(n["id"])
    by = {n["id"]: n for n in ns}

    def sub(i: str) -> list:
        out, todo = [], [i]
        while todo:
            j = todo.pop()
            out.append(j)
            todo += [k for k in kids.get(j, []) if k in by]
        return out
    N, c, w = len(ns), c_of(t, s), w_of(t, s)
    budget_s = 60 * float(s["minutes"]) if s.get("minutes") else max(sum(n["seconds"] for n in ns), 1.0) * 4
    rows = []
    for n in ns:
        st = sub(n["id"])
        Q = sum(G[j] for j in st) / len(st)
        U = math.sqrt(math.log(N + 1) / len(st))
        cost = (max(n["seconds"], 1.0) / budget_s) ** s["cost_weight"]
        rows.append({"id": n["id"], "op": "improve" if n["valid"] else "debug", "value": n["value"], "G": G[n["id"]],
                     "Q": Q, "n": len(st), "U": U, "cost": cost, "uct": (Q + c * U) * cost, "elite": 0.0,
                     "seconds": n["seconds"]})
    elites = sorted((r for r in rows if by[r["id"]]["valid"]), key=lambda r: -r["G"])[:3]
    z = sum(1 / (i + 1) for i in range(len(elites)))
    for i, r in enumerate(elites):
        r["elite"] = (1 / (i + 1)) / z
    top = max(rows, key=lambda r: r["uct"])
    for r in rows:
        r["P"] = w * (1.0 if r is top else 0.0) + (1 - w) * r["elite"]
    rows.sort(key=lambda r: (-r["P"], -r["uct"]))
    # stagnation: per root line (a draft and what grew from it), and globally, in run order
    root = {}
    for n in ns:
        p = n["parent"]
        root[n["id"]] = root.get(p, p) if p in by else n["id"]
    stagnant, best_line, since_line = {}, {}, {}
    best, since = None, 0
    for n in ns:
        v = sign * n["value"] if n["valid"] else None
        rt = root[n["id"]]
        if v is not None and (rt not in best_line or v > best_line[rt]):
            best_line[rt], since_line[rt] = v, 0
        else:
            since_line[rt] = since_line.get(rt, 0) + 1
        if v is not None and (best is None or v > best):
            best, since = v, 0
        else:
            since += 1
    for rt, k in since_line.items():
        if k >= int(s["stagnation_branch"]):
            stagnant[rt] = k
    best_id = elites[0]["id"] if elites else None
    return {"t": t, "c": c, "w": w, "rows": rows, "best": best_id, "stagnant": stagnant, "since_best": since,
            "root": root}


def frontier_items(study, item, c: dict) -> list:
    """The search's items for openlab.study.frontier (`item` makes one, `c` the costs)."""
    sc = score(study)
    if not sc["rows"]:
        return []
    s = settings(study.spec)
    out = []
    consolidating = sc["t"] >= s["consolidate"]
    for r in sc["rows"][:5]:
        if r["P"] <= 0 and r is not sc["rows"][0]:
            continue
        why = (f"{r['op']} {r['id']}: P(pick) {r['P']:.2f} at t {sc['t']:.2f} (Q {r['Q']:.2f}, c·U "
               f"{sc['c'] * r['U']:.2f}, elite {r['elite']:.2f}); `study run --op {r['op']} --parent {r['id']}`")
        out.append(item("expand", r["id"], why, "run", max(r["P"], 0.05), r["seconds"], key=f"{sc['t']:.2f}"))
    by_line: dict = {}
    for r in sc["rows"]:
        by_line.setdefault(sc["root"][r["id"]], []).append(r)
    for rt, k in sc["stagnant"].items():
        line = by_line.get(rt, [])
        if not line:
            continue
        owned = max(line, key=lambda r: r["G"])
        others = [max(v, key=lambda r: r["G"])["id"] for o, v in by_line.items() if o != rt]
        if others:
            refs = " ".join(f"--ref {o}" for o in others[:2])
            out.append(item("fuse", owned["id"], f"the line from {rt} has had {k} runs without a new best: "
                            f"`study run --op fuse --parent {owned['id']} {refs}`", "run", 0.6, owned["seconds"],
                            targets=[owned["id"], *others[:2]], key=str(k)))
    if not consolidating and sc["since_best"] >= int(s["stagnation_global"]):
        out.append(item("draft", study.record.find("plan")[0].id, f"{sc['since_best']} runs without a new best: "
                        "start a new line (`study run --op draft`)", "run", 0.5, c["run"], key=str(sc["since_best"])))
    if consolidating:
        out.append(item("select", study.record.find("plan")[0].id, f"the clock is at {sc['t']:.2f}: consolidate; "
                        "apply the selection rule (`study select`)", "select", 1.0, c["step"], key=f"{sc['t']:.2f}"))
    return out


# --- the selection rule ------------------------------------------------------------------------------------------
@lru_cache(maxsize=64)
def expected_max(n: int) -> float:
    """E[max of n iid standard normals] (0 for n <= 1), by numerical integration."""
    if n <= 1:
        return 0.0
    from math import erf, exp, pi, sqrt
    phi = lambda x: exp(-x * x / 2) / sqrt(2 * pi)                       # noqa: E731
    Phi = lambda x: 0.5 * (1 + erf(x / sqrt(2)))                         # noqa: E731
    h, total, x = 0.01, 0.0, -8.0
    while x < 8.0:
        total += x * n * phi(x) * Phi(x) ** (n - 1) * h
        x += h
    return total


def rule(spec) -> dict:
    sel = spec.raw.get("selection") or {}
    return {"rule": sel.get("rule"), "k": int(sel.get("k", 3))}


def select(study) -> dict:
    """Apply the sealed selection rule to the research runs: {rule, picked (a run id), result, value, adjusted, n,
    candidates: [{run, value}], configs (seed_robust)}."""
    spec = study.spec
    r = rule(spec)
    if not r["rule"]:
        raise StudyError("the study declares no selection rule", "the final candidate would be picked after seeing "
                         "the results", 'a person declares [selection] rule = "best_val" or "seed_robust" by amendment')
    m, hib = _primary(spec)
    if hib is None:
        raise StudyError(f"the primary metric {m} has no order to select by")
    sign = 1.0 if hib else -1.0
    ns = [n for n in nodes(study) if n["valid"]]
    if not ns:
        raise StudyError("no research run with a clean audit and a result to select from")
    if r["rule"] == "best_val":
        pick = max(ns, key=lambda n: sign * n["value"])
        n_cand, value, se = len(ns), pick["value"], _se(pick)
        extra = {}
    else:                                                       # seed_robust: configurations alike but for the seed
        groups: dict = {}
        for n in ns:
            groups.setdefault(n["config"], []).append(n)
        want = min(r["k"], max(len(g) for g in groups.values()))
        ok = {k: g for k, g in groups.items() if len(g) >= want}
        mean = {k: sum(n["value"] for n in g) / len(g) for k, g in ok.items()}
        best = max(ok, key=lambda k: sign * mean[k])
        pick = max(ok[best], key=lambda n: sign * n["value"])
        n_cand, value = len(ok), mean[best]
        se = (sum(_se(n) ** 2 for n in ok[best]) ** 0.5) / len(ok[best])
        extra = {"configs": len(groups), "seeds": len(ok[best]), "runs_of_pick": [n["id"] for n in ok[best]]}
    adjusted = value - sign * se * expected_max(n_cand)
    return {"rule": r["rule"], "picked": pick["id"], "result": pick["result"], "metric": m, "value": value,
            "se": se, "adjusted": adjusted, "n": n_cand,
            "candidates": [{"run": n["id"], "value": n["value"]} for n in ns], **extra}


def _se(n: dict) -> float:
    if isinstance(n.get("lo"), (int, float)) and isinstance(n.get("hi"), (int, float)):
        return abs(n["hi"] - n["lo"]) / (2 * 1.96)
    return 0.0


def check_look(study, fp: dict) -> None:
    """With a selection rule declared, the code a holdout look runs must be the selected run's."""
    if not rule(study.spec)["rule"]:
        return
    sel = study.record.last("selection")
    if sel is None:
        raise StudyError("the study declares a selection rule, and nothing has been selected",
                         "the look must be on the run the rule picks", "run `openlab study select` first")
    picked = study.record.get(sel.payload["picked"])
    if picked.payload["code"] != fp["code"] or picked.payload["data"] != fp["data"]:
        raise StudyError(f"the code here is not {picked.id}, the run the selection rule picked",
                         "a look is on the selected candidate only",
                         "restore its code (`openlab study promote` from its branch, or check out its model.py)")


def record(study):
    """`select` applied and recorded as a selection node (its values are derived again on write)."""
    from .schema import Selection
    rec = study.record
    s = select(study)
    under = rec.last("amendment") or rec.last("plan")
    return rec.add(Selection(rule=s["rule"], picked=s["picked"], metric=s["metric"], value=s["value"],
                             adjusted=s["adjusted"], se=s["se"], n=s["n"], candidates=s["candidates"]),
                   actor="agent", links=[("picks", s["picked"]), ("under", under.id)])
