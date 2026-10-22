"""Probe signals: what one spawned agent's run did, read from the control store's events and the study records it
wrote, as rates, and keyed by the harness that briefed it (launch.json `harness_sha`). They need no verifier and no
model: refusals, bypass attempts, loops, stalls and budget holds are on the events; red module checks, look-ahead
audits, refused record writes and plan steps the record contradicts are on the records.

Signals are evidence for a person, a reflector and a composer, never an acceptance score: a harness is accepted on
graded private tasks (openlab.harness.evaluate). Every field is a rate, not a count: a count rewards doing more (a
delegation bonus paid per successful child was gamed by spawning children; per child it was not).

`contrast` sets two groups of runs side by side, field by field, and says which side a field favours only when the
difference clears the noise of the runs themselves: a defect seen in one run of a group is noise until it repeats.
`transcript` renders a run in the [ACTION] / [OBSERVATION] / [ERROR] grammar a reader can follow, cutting an
observation no shorter than 10,000 characters (a shorter cut invents defects that are not there).
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Optional

#: name -> (which way is better, what it measures)
FIELDS = {
    "tool_error_rate": ("low", "tool errors per tool call"),
    "refusal_rate": ("low", "openlab refusals per tool call"),
    "bypass_rate": ("low", "bypass attempts per tool call"),
    "loop_rate": ("low", "LOOP episodes (one refusal repeated [agents] loop_refusals times) per 100 tool calls"),
    "stuck_rate": ("low", "STUCK episodes (no event for [agents] stuck_minutes while working) per hour"),
    "hold_rate": ("low", "budget holds per hour"),
    "budget_used": ("low", "the largest share of its $ or turn budget used"),
    "write_refusal_rate": ("low", "refused record writes per record write"),
    "red_check_rate": ("low", "red module checks per module check"),
    "lookahead_rate": ("low", "research runs whose audit found look-ahead, per run"),
    "cached_share": ("none", "study_run calls answered from the cache (no trial spent), per call"),
    "plan_ok": ("high", "plan steps the record shows (ok), per step reported"),
    "plan_bad": ("low", "plan steps the record contradicts (bad), per step reported"),
    "dead_end_repeat": ("low", "1 when its question or a hypothesis repeats another agent's dead end"),
    "usd_per_ok_step": ("low", "$ per plan step the record shows"),
    "minutes_per_ok_step": ("low", "wall minutes per plan step the record shows"),
    "usd_per_claim": ("low", "$ per claim it recorded"),
    "deliverable": ("high", "1 when it ended done, with a summary, having written to the record or its plan"),
}
UNTRUSTED_TOOLS = ("WebFetch", "WebSearch", "mcp__openlab__openlab_study_fetch_text",
                   "mcp__openlab__openlab_study_discover")
OBSERVATION_MIN = 10_000


def _store(root):
    from ..agents.store import Store
    return Store(root)


def _launch(store, agent: str) -> dict:
    try:
        return json.loads((store.agent_dir(agent) / "launch.json").read_text())
    except (OSError, ValueError):
        return {}


def _t(at: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(at)
    except (TypeError, ValueError):
        return None


def _per(n: float, d: float) -> Optional[float]:
    return round(n / d, 4) if d else None


def _record_rows(root: Path, agent: str, study: str) -> list:
    """(kind, payload) of the record nodes the agent wrote in its study (and the studies under the workspace it
    named), read-only."""
    from ..study.workspace import studies
    folders = [Path(root) / study] if study and (Path(root) / study / ".openlab" / "record.db").exists() \
        else studies(Path(root))
    out = []
    for folder in folders:
        db = Path(folder) / ".openlab" / "record.db"
        if not db.exists():
            continue
        try:
            with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)) as con:
                out += [(k, json.loads(p)) for k, p in con.execute(
                    "SELECT kind, payload FROM nodes WHERE session=? ORDER BY seq", (agent,)).fetchall()]
        except sqlite3.Error:
            continue
    return out


def _plan(store, agent: str, study: str) -> dict:
    """Steps of its latest plan by what the record shows: ok, rep (its word only) or bad."""
    from ..agents import plan as P
    try:
        versions = P.read(store, agent)["versions"]
    except Exception:                                               # noqa: BLE001 -- no plan: no steps
        return {"ok": 0, "rep": 0, "bad": 0}
    counts = {"ok": 0, "rep": 0, "bad": 0}
    for node in versions[-1]["nodes"] if versions else []:
        try:
            state, _ = P.evidence(store, study, node)
        except Exception:                                           # noqa: BLE001 -- unreadable: its word only
            state = "rep"
        counts[state] = counts.get(state, 0) + 1
    return counts


def _repeats(root, a: dict) -> bool:
    """The DEAD-END flag of openlab watch: its question or a hypothesis of its own shares most of its words with a
    dead end another agent recorded in its study."""
    from ..agents.view import overlap, study_stats
    if not a["study"] or not (Path(root) / a["study"]).is_dir():
        return False
    _, by_study = study_stats(Path(root), [Path(root) / a["study"]])
    info = by_study.get(str((Path(root) / a["study"]).resolve())) or {}
    owned = [t for who, t in info.get("hypotheses", []) if who == a["id"]] + ([a["question"]] if a["question"] else [])
    return any(overlap(m, t) >= 0.6 for m in owned for who, t in info.get("dead_ends", []) if who != a["id"])


def of(root, agent: str) -> dict:
    """The probe signals of one agent's run, with what keys it: its role, harness version and harness_sha."""
    from ..agents.store import config
    store, cfg = _store(root), config(root)
    a = store.get(agent)
    if a is None:
        raise KeyError(f"no agent {agent!r}")
    launch = _launch(store, agent)
    evs = store.events(agent, n=10 ** 6)
    calls = int((a["spent"] or {}).get("tool_calls") or 0) or sum(e["kind"] == "tool" for e in evs)
    count = lambda *kinds: sum(e["kind"] in kinds for e in evs)                         # noqa: E731
    refusals = [e["summary"] for e in evs if e["kind"] == "refusal"]
    loops = sum(1 for s in set(refusals) if refusals.count(s) >= int(cfg["loop_refusals"]))
    times = [t for t in (_t(e["at"]) for e in evs) if t]
    start, end = (_t(a["created"]) or (times[0] if times else None)), (times[-1] if times else None)
    paused = float((a["spent"] or {}).get("paused_minutes") or 0)
    hours = max(((end - start).total_seconds() / 3600 - paused / 60) if start and end else 0.0, 0.0)
    stuck_s = float(cfg["stuck_minutes"]) * 60
    stuck = sum(1 for x, y in zip(times, times[1:]) if (y - x).total_seconds() > stuck_s)
    b, s = a["budget"] or {}, a["spent"] or {}
    used = [float(s.get(k) or 0) / float(b[k]) for k in ("usd", "turns") if b.get(k)]
    rows = _record_rows(root, agent, a["study"])
    kinds = [k for k, _ in rows]
    actions = [p for k, p in rows if k == "action"]
    checks = [p for k, p in rows if k == "module_check"]
    runs = [p for k, p in rows if k == "run" and p.get("segment") == "research"]
    run_calls = [p for p in actions if p.get("verb") == "run" and (p.get("outcome") or {}).get("ok")]
    plan = _plan(store, agent, a["study"])
    steps = sum(plan.values())
    usd, claims = float(s.get("usd") or 0), kinds.count("claim")
    minutes = hours * 60
    repeat = 1.0 if _repeats(root, a) else 0.0
    said = any(e["kind"] == "result" and (e["payload"] or {}).get("subtype") == "success" for e in evs) \
        or any(e["kind"] == "say" for e in evs)
    out = {
        "tool_error_rate": _per(count("tool_error"), calls),
        "refusal_rate": _per(count("refusal"), calls),
        "bypass_rate": _per(count("bypass_attempt"), calls),
        "loop_rate": _per(100 * loops, calls),
        "stuck_rate": _per(stuck, hours),
        "hold_rate": _per(count("budget_hold"), hours),
        "budget_used": round(max(used), 4) if used else None,
        "write_refusal_rate": _per(sum(1 for p in actions if not (p.get("outcome") or {}).get("ok", True)),
                                   len(actions)),
        "red_check_rate": _per(sum(1 for p in checks if not p.get("ok")), len(checks)),
        "lookahead_rate": _per(sum(1 for p in runs if (p.get("audit") or {}).get("verdict") == "look-ahead"),
                               len(runs)),
        "cached_share": _per(max(len(run_calls) - len(runs), 0), len(run_calls)),
        "plan_ok": _per(plan["ok"], steps),
        "plan_bad": _per(plan["bad"], steps),
        "dead_end_repeat": repeat,
        "usd_per_ok_step": _per(usd, plan["ok"]),
        "minutes_per_ok_step": _per(minutes, plan["ok"]),
        "usd_per_claim": _per(usd, claims),
        "deliverable": 1.0 if a["status"] == "done" and said and (rows or steps) else 0.0,
    }
    return {"agent": agent, "role": a["role"], "status": a["status"], "study": a["study"],
            "harness": launch.get("harness", ""), "harness_sha": launch.get("harness_sha", ""),
            "lessons": launch.get("lessons", []), "tag": launch.get("tag", ""), "tool_calls": calls,
            "hours": round(hours, 3), "usd": round(usd, 4), "untrusted": touched_untrusted(root, agent),
            "signals": out}


def touched_untrusted(root, agent: str) -> bool:
    """Whether the run read the open web or a fetched source (its output may carry text written to steer an agent)."""
    return any(e["tool"] in UNTRUSTED_TOOLS for e in _store(root).events(agent, n=10 ** 6, kinds=("tool",)))


def runs(root, role: str = "", harness_sha: str = "", harness: str = "", tag: str = "", ended: bool = True) -> list:
    """The signals of the workspace's spawned agents that match (ended ones only, by default)."""
    from ..agents.store import FINAL
    store, out = _store(root), []
    for a in store.agents():
        if a["kind"] not in ("spawned", "codex") or (role and a["role"] != role) or (ended and a["status"] not in FINAL):
            continue
        launch = _launch(store, a["id"])
        if (harness_sha and launch.get("harness_sha") != harness_sha) or (harness and launch.get("harness") != harness) \
                or (tag and launch.get("tag") != tag):
            continue
        out.append(of(root, a["id"]))
    return out


def groups(rows: list, key: str = "harness_sha") -> dict:
    """The rows by `key` (harness_sha, harness or role)."""
    out: dict = {}
    for r in rows:
        out.setdefault(r.get(key) or "", []).append(r)
    return out


def summary(rows: list) -> dict:
    """Per field: the mean over the runs that have a value, its standard error and n."""
    out = {}
    for f in FIELDS:
        xs = [r["signals"][f] for r in rows if r["signals"].get(f) is not None]
        mean = sum(xs) / len(xs) if xs else None
        se = (math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)) / math.sqrt(len(xs))) if len(xs) > 1 else None
        out[f] = {"mean": None if mean is None else round(mean, 4), "se": None if se is None else round(se, 4),
                  "n": len(xs)}
    return out


def contrast(ref: list, cand: list) -> dict:
    """Field by field, the candidate runs against the reference runs: each side's mean, the difference, and which
    side it favours. A difference counts only when both sides have 2+ runs and it exceeds twice their pooled
    standard error (`moved`); else it is noise, said as such."""
    a, b = summary(ref), summary(cand)
    rows = []
    for f, (better, what) in FIELDS.items():
        x, y = a[f], b[f]
        if x["mean"] is None or y["mean"] is None:
            rows.append({"field": f, "what": what, "ref": x["mean"], "cand": y["mean"], "delta": None,
                         "favours": "n/a"})
            continue
        delta = round(y["mean"] - x["mean"], 4)
        noise = math.sqrt((x["se"] or 0) ** 2 + (y["se"] or 0) ** 2) if x["n"] > 1 and y["n"] > 1 else None
        moved = noise is not None and abs(delta) > 2 * noise and delta != 0
        if not moved or better == "none":
            favours = "none (noise)" if delta and not moved else "none"
        else:
            favours = "cand" if (delta < 0) == (better == "low") else "ref"
        rows.append({"field": f, "what": what, "ref": x["mean"], "cand": y["mean"], "delta": delta,
                     "noise": None if noise is None else round(2 * noise, 4), "moved": moved, "favours": favours})
    return {"n": [len(ref), len(cand)], "ref_agents": [r["agent"] for r in ref], "cand_agents": [r["agent"] for r in cand],
            "rows": rows}


def save(root, c: dict, note: str = "") -> str:
    """Store a contrast, content-addressed (.openlab/harness/contrasts/<id>.json), and record it; its id (a harness
    lesson's basis cites it)."""
    from .versions import AGENT, home
    body = json.dumps({**c, "note": note}, sort_keys=True)
    cid = "hc-" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]
    d = home(root) / "contrasts"
    d.mkdir(parents=True, exist_ok=True)
    if not (d / f"{cid}.json").exists():
        (d / f"{cid}.json").write_text(body, encoding="utf-8")
        moved = [r["field"] for r in c["rows"] if r.get("moved")]
        _store(root).event(AGENT, "harness_contrast", summary=f"contrast {cid}: {c['n'][0]} vs {c['n'][1]} runs; "
                           f"moved: {', '.join(moved) or 'nothing'}", payload={"id": cid, "n": c["n"], "moved": moved,
                                                                               "note": note})
    return cid


def load_contrast(root, cid: str) -> dict:
    from .versions import home
    try:
        return json.loads((home(root) / "contrasts" / f"{cid}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise KeyError(f"no stored contrast {cid!r}") from None


def table(c: dict) -> list:
    """The contrast as text lines."""
    lines = [f"ref {c['n'][0]} run(s), cand {c['n'][1]} run(s)"]
    for r in c["rows"]:
        if r["delta"] is None:
            continue
        lines.append(f"  {r['field']:<22} {r['ref']!s:>9} -> {r['cand']!s:<9} {r['delta']:+.4g}  {r['favours']}")
    return lines


# --- the transcript, as a reader follows it -----------------------------------------------------------------------
def _cut(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + f"\n[... {len(text) - n} more characters cut]"


def transcript(root, agent: str, observation_max: int = OBSERVATION_MIN) -> str:
    """The run as [SAY] / [ACTION] / [OBSERVATION] / [ERROR] blocks, from its CLI transcript (claude stream-json or
    codex JSONL), else from its events."""
    from ..agents.supervisor import refusal
    n = max(int(observation_max), OBSERVATION_MIN)
    store = _store(root)
    path = store.agent_dir(agent) / "transcript.jsonl"
    out, names = [], {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines() if path.exists() else []
    for line in lines:
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        msg = obj.get("message") or {}
        if obj.get("type") == "assistant":
            for b in msg.get("content") or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text" and b.get("text", "").strip():
                    out.append(f"[SAY] {b['text'].strip()}")
                elif b.get("type") == "tool_use":
                    names[b.get("id")] = b.get("name")
                    out.append(f"[ACTION] {b.get('name')} {json.dumps(b.get('input') or {}, sort_keys=True)}")
        elif obj.get("type") == "user" and isinstance(msg.get("content"), list):
            for b in msg["content"]:
                if not isinstance(b, dict) or b.get("type") != "tool_result":
                    continue
                c = b.get("content")
                text = c if isinstance(c, str) else "".join(x.get("text", "") for x in c or [] if isinstance(x, dict))
                tag = "ERROR" if b.get("is_error") or refusal(text) else "OBSERVATION"
                out.append(f"[{tag}] {names.get(b.get('tool_use_id'), '?')}: {_cut(text, n)}")
        elif obj.get("type") == "result":
            out.append(f"[RESULT] {obj.get('subtype')}: {_cut(str(obj.get('result') or ''), n)}")
        elif obj.get("type") == "item.completed" or (obj.get("msg") or {}).get("type"):
            out.append(f"[ACTION] {json.dumps(obj, sort_keys=True)[:n]}")
    if not out:                                                     # no CLI transcript: the events say it
        tag = {"tool": "ACTION", "refusal": "ERROR", "tool_error": "ERROR", "bypass_attempt": "ERROR", "say": "SAY",
               "result": "RESULT", "end": "RESULT"}
        out = [f"[{tag[e['kind']]}] {e['summary']}" for e in store.events(agent, n=10 ** 6) if e["kind"] in tag]
    return "\n".join(out)
