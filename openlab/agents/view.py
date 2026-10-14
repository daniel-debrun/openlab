"""What `openlab agents` and `openlab watch` show: every agent in tree order, its state and the flags a supervising
person cares about.

States: working, idle, waiting (on a person: a pending inbox item or a permission prompt), stuck (working with no
event for [agents] stuck_minutes), paused, done, failed, stopped, lost.
Flags: STUCK; RUN (an openlab run the agent started is in flight: it is working, not stuck); LOOP (the same refusal
[agents] loop_refusals times or more among its recent events); OVER (90% of a budget: $, minutes or turns); WAITING (a pending inbox item); DEAD-END (its hypothesis or question shares most of
its words with a dead end another agent recorded in the same study); BYPASS (a denied bypass attempt).
Per study, the record's own nodes give each agent's trials, claims and refusals. Trials are counted as `study
status` counts them: each distinct research run key once (to the agent that ran it first) plus the trials an agent
declared; a node copied from another study's record (same hash: `cp -r` of a study) counts once.
"""

from __future__ import annotations

import os
import re
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Optional

from .store import FINAL, Store, age, config

_WORD = re.compile(r"[a-z0-9]{4,}")
_STOP = {"that", "this", "with", "from", "than", "when", "will", "does", "have", "into", "more", "less", "over",
         "under", "about", "which", "their", "there", "these", "those", "should", "would", "could", "model"}


def words(text: str) -> set:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOP}


def overlap(a: str, b: str) -> float:
    wa, wb = words(a), words(b)
    return len(wa & wb) / min(len(wa), len(wb)) if wa and wb else 0.0


def tree_order(agents: list[dict]) -> list[tuple[dict, int]]:
    ids = {a["id"] for a in agents}
    kids: dict = {}
    for a in agents:
        kids.setdefault(a["parent"] if a["parent"] in ids else "", []).append(a)
    out, seen = [], set()

    def walk(pid: str, depth: int) -> None:
        for a in kids.get(pid, []):
            if a["id"] not in seen:
                seen.add(a["id"])
                out.append((a, depth))
                walk(a["id"], depth + 1)
    walk("", 0)
    out += [(a, 0) for a in agents if a["id"] not in seen]
    return out


def study_stats(root: Path, studies: list) -> tuple[dict, dict]:
    """({agent: {trials, claims, refusals, writes, last_write}}, {study folder: {"dead_ends": [(agent, text)],
    "hypotheses": [(agent, text)]}}) from the studies' records, read-only."""
    per, by_study = {}, {}
    seen_hashes: set = set()                    # a copied record's nodes keep their hashes: count them once
    seen_keys: set = set()
    for folder in studies:
        db = Path(folder) / ".openlab" / "record.db"
        if not db.exists():
            continue
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
        except sqlite3.Error:
            continue
        with closing(con):
            try:
                rows = con.execute("SELECT session, kind, payload, at, hash FROM nodes WHERE session != '' "
                                   "ORDER BY seq").fetchall()
            except sqlite3.Error:
                continue
        import json
        info = {"dead_ends": [], "hypotheses": []}
        keys: dict = {}
        for session, kind, payload, at, h in rows:
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            s = per.setdefault(session, {"trials": 0, "claims": 0, "refusals": 0, "writes": 0, "last_write": ""})
            s["writes"] += 1
            s["last_write"] = max(s["last_write"], at)
            if kind in ("run", "claim", "action", "dead_end", "hypothesis", "trial"):
                p = json.loads(payload)
                if kind == "run" and p.get("segment") == "research" and (folder, p.get("key")) not in seen_keys:
                    seen_keys.add((folder, p.get("key")))
                    keys.setdefault(session, set()).add((folder, p.get("key")))
                elif kind == "trial":
                    s["trials"] += int(p.get("n") or 0)
                elif kind == "claim":
                    s["claims"] += 1
                elif kind == "action" and not (p.get("outcome") or {}).get("ok", True):
                    s["refusals"] += 1
                elif kind == "dead_end":
                    info["dead_ends"].append((session, f"{p.get('statement', '')} {' '.join(p.get('tags') or [])}"))
                elif kind == "hypothesis":
                    info["hypotheses"].append((session, f"{p.get('statement', '')} {' '.join(p.get('tags') or [])}"))
        for session, ks in keys.items():
            per[session]["trials"] += len(ks)
        by_study[str(Path(folder).resolve())] = info
    return per, by_study


def _live_run(store: Store, agent: str) -> Optional[dict]:
    """The agent's openlab run in flight (its last `run` event with no `run_end` after it, its process alive)."""
    evs = store.events(agent, n=1, kinds=("run", "run_end"))
    if not evs or evs[-1]["kind"] != "run":
        return None
    try:
        pid = int(evs[-1]["payload"].get("pid") or 0)
        if pid <= 0:
            return None
        os.kill(pid, 0)
    except (OSError, ValueError):
        return None
    return evs[-1]


def snapshot(store: Store, all: bool = True, studies: Optional[list] = None) -> dict:
    cfg = config(store.root)
    store.reap()
    agents = store.agents()
    if studies is None:
        try:
            from ..study.workspace import studies as find
            studies = find(store.root)
        except Exception:                                           # noqa: BLE001
            studies = []
    stats, by_study = study_stats(store.root, studies)
    pending = Counter(i["agent"] for i in store.inbox("pending"))
    rows = []
    for a, depth in tree_order(agents):
        if not all and (a["status"] == "ended" or a["status"] in FINAL and a["status"] != "lost"
                        and age(a["updated"]) > 3600):              # an ended (quiet) session: --all shows it
            continue
        evs = store.events(a["id"], n=40)
        refusals = [e for e in evs if e["kind"] in ("refusal", "bypass_attempt", "blocked")]
        last_act = next((e for e in reversed(evs) if e["kind"] in (
            "tool", "say", "note", "stop", "result", "end", "prompt", "start", "run", "run_progress", "run_end")), None)
        b, s = a["budget"], a["spent"]
        idle_for = age(a["last_event_at"] or a["created"])
        status = a["status"]
        run = _live_run(store, a["id"]) if status not in FINAL else None
        if status == "paused" or (a["control"] == "pause" and status not in FINAL):
            state = "paused"
        elif status in FINAL:
            state = status
        elif pending[a["id"]] or status == "waiting":
            state = "waiting"
        elif run is not None:                       # waiting on its own openlab run: busy, not stuck
            state = "working"
        elif status == "running" and idle_for > float(cfg["stuck_minutes"]) * 60:
            state = "stuck"
        elif status == "running":
            state = "working"
        else:
            state = "idle"
        flags = []
        if state == "stuck":
            flags.append("STUCK")
        if run is not None:
            prog = store.events(a["id"], n=1, kinds=("run_progress",))
            queued = prog and prog[-1]["at"] >= run["at"] and prog[-1]["summary"].startswith("queued")
            flags.append(f"QUEUED {age(run['at']) / 60:.0f}m" if queued else f"RUN {age(run['at']) / 60:.0f}m")
        loops = Counter(e["summary"] for e in refusals if e["kind"] == "refusal")
        top = loops.most_common(1)
        if top and top[0][1] >= int(cfg["loop_refusals"]):
            flags.append(f"LOOP×{top[0][1]}")
        frac = []
        if b.get("usd"):
            frac.append((s.get("usd") or 0) / b["usd"])
        if b.get("turns"):
            frac.append((s.get("turns") or 0) / b["turns"])
        if b.get("minutes") and status not in FINAL:
            frac.append((age(a["created"]) / 60 - (s.get("paused_minutes") or 0)) / b["minutes"])
        if frac and max(frac) >= float(cfg["over_fraction"]):
            flags.append("OVER")
        if pending[a["id"]]:
            flags.append("WAITING")
        if any(e["kind"] == "bypass_attempt" for e in evs):
            flags.append("BYPASS")
        st = stats.get(a["id"], {})
        study = str((store.root / a["study"]).resolve()) if a["study"] else ""
        info = by_study.get(study) or {}
        owned = [t for who, t in info.get("hypotheses", []) if who == a["id"]] + ([a["question"]] if a["question"] else [])
        others = [(who, t) for who, t in info.get("dead_ends", []) if who != a["id"]]
        match = next(((who, t) for m in owned for who, t in others if overlap(m, t) >= 0.6), None)
        if match:
            flags.append("DEAD-END")
        rows.append({"agent": a, "depth": depth, "state": state, "flags": flags,
                     "usd": s.get("usd") or 0.0, "usd_estimated": bool(s.get("usd_estimated")),
                     "budget_usd": b.get("usd"), "frac": max(frac) if frac else 0.0,
                     "tokens": (s.get("tokens_in") or 0) + (s.get("tokens_out") or 0),
                     "trials": st.get("trials", 0), "claims": st.get("claims", 0),
                     "refusals": max(len([e for e in refusals if e["kind"] in ("refusal", "bypass_attempt")]),
                                     st.get("refusals", 0)),
                     "last": last_act, "last_refusal": refusals[-1] if refusals else None,
                     "idle_for": idle_for, "dead_end": match, "pending": pending[a["id"]]})
    kinds = {a["id"]: a["kind"] for a in agents}
    try:                                                            # the machine's queue (openlab.study.admission)
        from ..study.admission import tickets
        q = tickets()
        queue = {"running": sum(t["state"] == "running" for t in q), "waiting": sum(t["state"] == "waiting" for t in q)}
    except Exception:                                               # noqa: BLE001 -- the screen never fails on it
        queue = {"running": 0, "waiting": 0}
    return {"root": str(store.root), "rows": rows, "pending": sum(pending.values()), "queue": queue,
            "usd": sum(r["usd"] for r in rows if r["agent"]["kind"] != "claude-subagent"   # a spawned agent's
                       or kinds.get(r["agent"]["parent"]) == "claude-session"),           # total holds its subs
            "usd_estimated": any(r["usd_estimated"] and r["usd"] for r in rows),
            "counts": Counter(r["state"] for r in rows)}


def ago(seconds: float) -> str:
    if seconds == float("inf"):
        return "-"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.0f}d"


def tokens(n: int) -> str:
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}k" if n >= 1e3 else str(n)
