"""Lessons: rules learned from the study's own runs, each resting on what the record shows (after MARS's comparative
reflective memory, with the comparison computed rather than written).

A lesson's basis is a comparison of two of the study's results (`study compare`: a paired difference on the rows both
scored) or, for a fix, the run that fixed its parent. Its direction must follow from the basis (schema.Lesson):

| direction | the basis shows |
|---|---|
| improves | the first side better, with an interval that excludes 0 |
| hurts | the second side better, with an interval that excludes 0 |
| no_gain | an interval inside ±margin (the metric's units; the margin is declared with the lesson) |
| fixes | a run whose parent had no result or an audit that was not clean, and which itself is valid |

A run that applies a lesson cites it (`study run --cite lesson:<id>`, or `<study>/lesson:<id>` from another study of
the workspace). A lesson's track record is computed from the runs that cite it: held when the citing run did better
than its parent on the primary metric. `find` ranks the workspace's lessons for a query (BM25 over rule and tags).
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from .errors import StudyError


def shows(rec, basis, direction: str, margin) -> dict:
    """What the basis shows for the direction ({metric, diff, lo, hi} or {fixed, by}); refused when it does not."""
    from .schema import LESSON_DIRECTIONS
    if direction not in LESSON_DIRECTIONS:
        raise StudyError(f"no lesson direction `{direction}`", f"one of {', '.join(LESSON_DIRECTIONS)}")
    if direction == "fixes":
        if basis.kind != "run":
            raise StudyError("a fix rests on the run that fixed its parent", "basis run:<id>")
        parent = rec.links(basis.id, rel="branches_from")
        if not parent:
            raise StudyError(f"{basis.id} has no parent to have fixed")
        p = rec.get(parent[0][2])
        broken = (not rec.links(p.id, rel="of", incoming=True)
                  or p.payload["audit"].get("verdict") in ("look-ahead",))
        valid = (rec.links(basis.id, rel="of", incoming=True)
                 and basis.payload["audit"].get("verdict") not in ("look-ahead",))
        if not broken or not valid:
            raise StudyError(f"{basis.id} does not show a fix", "its parent must have failed (no result, or an audit "
                             "that found look-ahead) and it must be valid")
        return {"fixed": p.id, "by": basis.id}
    if basis.kind != "comparison" or not basis.payload.get("paired"):
        raise StudyError("this lesson rests on a comparison of two of the study's results",
                         "`study compare RESULT_A RESULT_B` pairs them on the rows both scored", "basis comparison:<id>")
    pd = basis.payload["paired"]
    a, b = [d for _, _, d in rec.links(basis.id, rel="compares")]
    lo, hi = pd.get("lo"), pd.get("hi")
    if not all(isinstance(x, (int, float)) and not math.isnan(x) for x in (lo, hi)):
        raise StudyError("the comparison has no interval", "a lesson needs the paired interval")
    excludes = lo > 0 or hi < 0
    ok = {"improves": pd.get("better") == a and excludes, "hurts": pd.get("better") == b and excludes}
    if direction == "no_gain":
        if not isinstance(margin, (int, float)) or margin <= 0:
            raise StudyError("a no_gain lesson declares its margin", "how small a difference counts as none, in the "
                             "metric's units", "margin=0.005, say")
        ok["no_gain"] = -margin <= lo and hi <= margin
    if not ok[direction]:
        raise StudyError(f"the comparison does not show `{direction}`",
                         f"{pd.get('metric', basis.payload['metric'])}: diff {pd.get('diff'):.4g} [{lo:.4g}, {hi:.4g}], "
                         f"better {pd.get('better') or 'neither'}",
                         "record what it shows (a no_gain needs the interval inside its margin), or keep it as a note")
    return {"metric": basis.payload["metric"], "diff": pd.get("diff"), "lo": lo, "hi": hi}


def record(study, rule: str, direction: str, basis: str, tags: list, margin=None):
    from .schema import Lesson
    rec = study.record
    node = rec.get(basis)
    return rec.add(Lesson(rule=rule, direction=direction, tags=[t.strip() for t in tags if t.strip()],
                          margin=margin, shown=shows(rec, node, direction, margin)),
                   actor="agent", links=[("basis", node.id)])


def resolve(study, ref: str) -> tuple:
    """(study, lesson node) a citation names: lesson:<id> in this study, or <study folder>/lesson:<id>."""
    from .. import study as _study
    if "/" in ref:
        name, lid = ref.rsplit("/", 1)
        other = _study.open(Path(study.folder).parent / name)
        return other, other.record.get(lid)
    return study, study.record.get(ref)


def track(study, lesson_id: str) -> dict:
    """The runs that cite a lesson (in this study), and whether each did better than its parent."""
    from .search import _primary
    rec = study.record
    m, hib = _primary(study.spec)
    sign = -1.0 if hib is False else 1.0
    cited, held = [], 0
    for r in rec.find("run", segment="research"):
        if not any(c == lesson_id or c.endswith("/" + lesson_id) for c in r.payload.get("cites", [])):
            continue
        parent = rec.links(r.id, rel="branches_from")
        owned = rec.links(r.id, rel="of", incoming=True)
        theirs = rec.links(parent[0][2], rel="of", incoming=True) if parent else []
        v = (rec.get(owned[0][0]).payload["metrics"].get(m) or {}).get("value") if owned else None
        w = (rec.get(theirs[0][0]).payload["metrics"].get(m) or {}).get("value") if theirs else None
        better = None if v is None or w is None else sign * v > sign * w
        held += better is True
        cited.append({"run": r.id, "held": better})
    return {"cited": len(cited), "held": held, "runs": cited}


def listing(study) -> list:
    return [{"id": n.id, "rule": n.payload["rule"], "direction": n.payload["direction"], "tags": n.payload["tags"],
             "basis": study.record.links(n.id, rel="basis")[0][2], "shown": n.payload["shown"],
             "track": track(study, n.id)} for n in study.record.find("lesson")]


# --- retrieval across the workspace ------------------------------------------------------------------------------
def _tokens(text: str) -> list:
    return [w for w in re.split(r"[^a-z0-9\-]+", text.lower()) if w]


def find(root, query: str, n: int = 10, here: str = "") -> list:
    """The workspace's lessons ranked for `query` (BM25 over rule and tags, tags counted twice); each with its
    study, basis and track record."""
    from .. import study as _study
    from .workspace import studies
    docs = []
    for folder in studies(Path(root)):
        try:
            s = _study.open(folder)
            for item in listing(s):
                docs.append((s.folder.name, item, _tokens(item["rule"] + " " + " ".join(item["tags"]) * 2)))
        except Exception:                                           # noqa: BLE001 -- an unreadable study is skipped
            continue
    if not docs:
        return []
    q = _tokens(query)
    N, avg = len(docs), sum(len(d) for _, _, d in docs) / len(docs)
    df = {w: sum(1 for _, _, d in docs if w in d) for w in set(q)}
    scored = []
    for name, item, d in docs:
        s = 0.0
        for w in q:
            f = d.count(w)
            if f:
                idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * len(d) / avg))
        if s > 0:
            scored.append({"study": name, "cite": item["id"] if name == here else f"{name}/{item['id']}",
                           "score": round(s, 3), **item})
    return sorted(scored, key=lambda x: -x["score"])[:n]


def main(argv) -> int:
    """`openlab lessons find QUERY [--root DIR] [-n N] [--json]`."""
    import argparse
    import json
    p = argparse.ArgumentParser(prog="openlab lessons", description="The workspace's lessons, found by what you are "
                                "about to try.")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("find", help="rank the lessons of every study in the workspace for a query")
    f.add_argument("query")
    f.add_argument("--root", default=".")
    f.add_argument("-n", type=int, default=10)
    f.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    from .workspace import root
    hits = find(root(Path(a.root)), a.query, a.n)
    if a.json:
        print(json.dumps(hits, indent=1, default=str))
    elif not hits:
        print("no lesson matches")
    for h in hits if not a.json else []:
        t = h["track"]
        print(f"{h['score']:>6}  {h['cite']}  {h['direction']}: {h['rule']}\n        basis {h['basis']}; "
              f"tags {', '.join(h['tags'])}; cited {t['cited']}, held {t['held']}")
    return 0
