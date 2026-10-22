"""The offline harness evaluator: a candidate harness is accepted only on graded tasks no model saw while it was
made, against the pinned baseline and against the baseline given the same dollars as best-of-n samples.

Tasks are split once, by a salted hash of their names, into a development pool (whose transcripts and signals
editors and reflectors may read) and a private pool (which only gates acceptance and never enters any context);
canary tasks are ones no honest run can medal on, so a medal there means the answers were read. `dev_only` is what
anything feeding a model must pass rows through, and `tripwire` scans an evidence folder for grader output or
private-pool paths before a model is spawned on it.

`accept` gates a candidate, each gate a reason when it fails:

1. no leak attempt in any of its runs, no medal on a canary;
2. the trap suite (bench.harness traps) passed on this code;
3. a noise band measured from 2+ baseline seeds (never assumed): z times the standard error of the per-seed rate;
4. its private rate at least the best so far less the band (a regression floor);
5. growth paid by gain: a candidate whose always-on text grew must beat the baseline by more than the band;
6. at most `edit_cap` edits bundled in one candidate;
7. its predictions hold: of the tasks it said it would move, no more than half went the other way or stayed
   (else it is rolled back, however its aggregate moved);
8. out of distribution (rows in pool "ood"), it does not fall below the baseline by more than the band.

It reports whether the candidate also beats the matched best-of-n arm by more than the band: only then is
per-task refinement worth its four rollouts per step (openlab.harness.refine). Cost per accepted edit counts every
rejected proposal's spend.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from pathlib import Path

POOLS = ("dev", "private", "canary", "ood")
#: what in an evidence folder says a grader looked, or names the private pool
MARKERS = re.compile(r"\b(gold_medal|silver_medal|bronze_medal|any_medal|above_median|medal rate|grader|"
                     r"answers?\.csv|prepared/private|results\.jsonl|leaderboards?/)\b", re.I)


def split(tasks, private: float = 0.5, salt: str = "openlab") -> dict:
    """{dev: [...], private: [...]}: each task to one pool by a salted hash of its name (stable across runs)."""
    out = {"dev": [], "private": []}
    for t in sorted(set(tasks)):
        h = int(hashlib.sha256(f"{salt}|{t}".encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        out["private" if h < private else "dev"].append(t)
    return out


def fixed_pools(path: Path, tasks=None, private: float = 0.5, salt: str = "openlab", canaries=(), ood=()) -> dict:
    """The pools stored at `path`, made once from `tasks` (later calls read them: a pool never moves)."""
    path = Path(path)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    if tasks is None:
        raise ValueError(f"no pools at {path}: give the tasks to split")
    pools = {**split([t for t in tasks if t not in canaries and t not in ood], private, salt),
             "canary": sorted(canaries), "ood": sorted(ood), "salt": salt, "private_fraction": private}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pools, indent=1), encoding="utf-8")
    return pools


def dev_only(rows: list, pools: dict) -> list:
    """The rows a model may read: development-pool tasks only, with no grade on them."""
    dev = set(pools.get("dev") or [])
    keep = ("task", "competition", "arm", "seed", "sample", "valid", "spent", "leaks")
    return [{k: r[k] for k in keep if k in r} for r in rows if (r.get("task") or r.get("competition")) in dev]


def tripwire(folder) -> list:
    """What in `folder` would let a model see a grade or the private pool: [(file, the marker)]."""
    hits = []
    for f in sorted(Path(folder).rglob("*")):
        if not f.is_file() or f.stat().st_size > 50_000_000:
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        m = MARKERS.search(text)
        if m:
            hits.append((str(f.relative_to(folder)), m.group(0)))
    return hits


def _task(r: dict) -> str:
    return r.get("task") or r.get("competition") or ""


def _medal(r: dict) -> bool:
    return bool(r.get("medal", r.get("any_medal")))


def stats(rows: list, arm: str, tasks=None) -> dict:
    """An arm's medal rate on `tasks` (all when None): the mean over seeds of the per-seed rate, its standard error,
    the seeds and runs, valid share, leak attempts and spend."""
    rs = [r for r in rows if r.get("arm") == arm and (tasks is None or _task(r) in set(tasks))]
    per_seed: dict = {}
    for r in rs:
        per_seed.setdefault(r.get("seed", 0), []).append(_medal(r))
    rates = [sum(v) / len(v) for v in per_seed.values()]
    mean = statistics.mean(rates) if rates else None
    sem = statistics.stdev(rates) / math.sqrt(len(rates)) if len(rates) > 1 else None
    return {"arm": arm, "rate": mean, "sem": sem, "seeds": len(per_seed), "runs": len(rs),
            "valid": sum(bool(r.get("valid")) for r in rs) / len(rs) if rs else None,
            "leaks": sum(len(r.get("leaks") or []) for r in rs),
            "usd": round(sum(float((r.get("spent") or {}).get("usd") or r.get("usd") or 0) for r in rs), 4)}


def per_task(rows: list, arm: str, tasks) -> dict:
    """{task: the arm's medal rate on it over seeds}."""
    out = {}
    for t in tasks:
        rs = [r for r in rows if r.get("arm") == arm and _task(r) == t]
        if rs:
            out[t] = sum(_medal(r) for r in rs) / len(rs)
    return out


def traps_passed(result) -> tuple:
    """(passed, what failed) from a bench.harness result (its JSON, a path to it, or None)."""
    if result is None:
        return False, ["the trap suite was not run (python -m bench.harness --suite traps)"]
    if isinstance(result, (str, Path)):
        result = json.loads(Path(result).read_text(encoding="utf-8"))
    traps = result.get("traps") or {}                                  # {caught, total, missed, traps: [...]}
    items = traps.get("traps") or []
    failed = [t.get("trap", "?") for t in items if not t.get("caught")]
    if not items:
        return False, ["the result holds no traps"]
    return not failed, failed


def accept(rows: list, baseline: str, candidate: str, pools: dict, bestofn: str = "", best_so_far=None,
           predicted: dict | None = None, growth: float = 0.0, edits: int = 1, edit_cap: int = 3,
           traps=None, z: float = 2.0) -> dict:
    """Whether `candidate` is accepted over `baseline` on the private pool (see the module's gates)."""
    private, canaries, ood = pools.get("private") or [], set(pools.get("canary") or []), pools.get("ood") or []
    base, cand = stats(rows, baseline, private), stats(rows, candidate, private)
    reasons = []
    leaks = [r for r in rows if r.get("arm") == candidate and r.get("leaks")]
    if leaks:
        reasons.append(f"{len(leaks)} of its runs attempted a leak: rejected whatever it scored")
    if any(r.get("arm") == candidate and _task(r) in canaries and _medal(r) for r in rows):
        reasons.append("it medalled on a canary task, which no honest run can: it read the answers")
    ok, failed = traps_passed(traps)
    if not ok:
        reasons.append("the trap suite did not pass: " + ", ".join(failed))
    if base["sem"] is None or base["seeds"] < 2:
        band = None
        reasons.append("the noise band is measured, not assumed: run the baseline on the private pool with 2+ seeds")
    else:
        band = z * base["sem"]
    if cand["rate"] is None or base["rate"] is None:
        reasons.append("no private-pool runs of the baseline or the candidate")
    floor_from = base["rate"] if best_so_far is None else best_so_far
    delta = None if cand["rate"] is None or base["rate"] is None else cand["rate"] - base["rate"]
    if band is not None and delta is not None:
        if cand["rate"] < floor_from - band:
            reasons.append(f"its private rate {cand['rate']:.3f} is below the best so far {floor_from:.3f} less the "
                           f"noise band {band:.3f}")
        if growth > 0 and delta <= band:
            reasons.append(f"its always-on text grew by {growth:.0%} without a gain beyond the noise band "
                           f"({delta:+.3f} against {band:.3f})")
    if edits > edit_cap:
        reasons.append(f"it bundles {edits} edits; the cap is {edit_cap} (fewer edits per candidate)")
    flips = {}
    for t, way in ((predicted or {}).get("tasks") or {}).items():
        a, b = per_task(rows, baseline, [t]).get(t), per_task(rows, candidate, [t]).get(t)
        flips[t] = None if a is None or b is None else ("held" if (b > a if way == "up" else b < a) else "failed")
    judged = [v for v in flips.values() if v is not None]
    if judged and sum(v == "failed" for v in judged) * 2 > len(judged):
        reasons.append(f"its predictions failed on {sum(v == 'failed' for v in judged)} of {len(judged)} tasks: "
                       "rolled back, however its aggregate moved")
    ood_delta = None
    if ood:
        ob, oc = stats(rows, baseline, ood), stats(rows, candidate, ood)
        if ob["rate"] is not None and oc["rate"] is not None:
            ood_delta = oc["rate"] - ob["rate"]
            if band is not None and ood_delta < -band:
                reasons.append(f"out of distribution it falls {ood_delta:+.3f} against the baseline (band {band:.3f})")
    bo = stats(rows, bestofn, private) if bestofn else None
    beats = None
    if bo and bo["rate"] is not None and cand["rate"] is not None and band is not None:
        beats = cand["rate"] - bo["rate"] > band
    return {"accepted": not reasons, "reasons": reasons, "baseline": base, "candidate": cand, "bestofn": bo,
            "band": band, "delta": delta, "ood_delta": ood_delta, "flips": flips, "beats_bestofn": beats,
            "refine_worth_it": bool(not reasons and beats)}


def cost_per_accepted(proposals: list) -> dict:
    """{usd, accepted, per_accepted}: every proposal's spend, rejected ones included, per accepted edit."""
    usd = sum(float(p.get("usd") or 0) for p in proposals)
    n = sum(bool(p.get("accepted")) for p in proposals)
    return {"usd": round(usd, 4), "proposals": len(proposals), "accepted": n,
            "per_accepted": round(usd / n, 4) if n else None}
