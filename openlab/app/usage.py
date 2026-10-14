r"""What each call cost, kept in ``.data/usage.jsonl``: the tool, the document, the wall time and the CPU the
session spent on it. ``usage`` sums it by tool or by document -- the cost of a research run, attributed.

openlab holds no model key, so it cannot see the tokens an agent spent; the agent's own CLI reports those
(Claude Code's ``/cost``). This is the other half: what the analysis itself cost, call by call.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..study import clock

MAX_BYTES = 8_000_000                                    # past this the oldest half is dropped


def _path(root: Path) -> Path:
    return Path(root) / ".data" / "usage.jsonl"


def log(root: Path, tool: str, env: dict, doc: str, cpu: float, by: str = "") -> None:
    if tool in ("usage", "tools"):
        return
    p = _path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {"at": clock.now(), "tool": tool, "doc": str(doc)[:200], "ok": bool(env.get("ok")),
           "ms": env.get("took_ms"), "cpu": round(cpu, 4), **({"by": by} if by else {}),
           **({"error": env.get("error")} if not env.get("ok") else {})}
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    if p.stat().st_size > MAX_BYTES:
        lines = p.read_text(encoding="utf-8").splitlines()
        p.write_text("\n".join(lines[len(lines) // 2:]) + "\n", encoding="utf-8")


def summary(root: Path, since: str = "", by: str = "tool", doc: str = "") -> dict:
    p = _path(root)
    rows = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if since and str(r.get("at", "")) < since:
                continue
            if doc and r.get("doc") != doc:
                continue
            rows.append(r)
    key = "doc" if by == "doc" else "tool"
    groups: dict = {}
    for r in rows:
        g = groups.setdefault(r.get(key) or "(none)", {"calls": 0, "errors": 0, "seconds": 0.0, "cpu": 0.0})
        g["calls"] += 1
        g["errors"] += 0 if r.get("ok") else 1
        g["seconds"] += (r.get("ms") or 0) / 1000
        g["cpu"] += r.get("cpu") or 0
    top = sorted(({key: k, **{m: (round(v, 3) if isinstance(v, float) else v) for m, v in g.items()}}
                  for k, g in groups.items()), key=lambda x: -x["seconds"])
    return {"calls": len(rows), "seconds": round(sum(g["seconds"] for g in groups.values()), 3),
            "cpu": round(sum(g["cpu"] for g in groups.values()), 3), "first": rows[0]["at"] if rows else None,
            "last": rows[-1]["at"] if rows else None, "by": key, "rows": top[:50],
            "note": "wall time and session CPU per call; an agent's tokens are its own CLI's to report (/cost)"}
