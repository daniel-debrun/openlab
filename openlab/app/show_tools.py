"""What `openlab show` draws, in one read: every agent as `openlab watch` sees it plus what it is doing now, the people
declared in openlab.toml, each plan with its evidence, the event feed, the inbox, the machine's queue, the settings
and the studies' records. The page asks for it again whenever the control store changes."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .registry import ToolContext, tool

FEED_MAX = 80
LEDGER_MAX = 400
ANSWERED_MAX = 20
TAIL_BYTES = 1 << 20        # how much of a transcript's end is read for its last message
FINAL_STATES = ("done", "failed", "stopped", "lost", "ended")     # openlab.agents.store.FINAL
#: the event kinds the feed shows (a tool call, what an agent said, a plan, a request, a decision, a refusal, a run)
FEED_KINDS = ("start", "tool", "say", "note", "plan", "plan_node", "request", "decision", "decision_result", "refusal",
              "bypass_attempt", "tool_error", "run", "run_end", "tell", "told", "reply", "btw", "control", "end", "result", "spawn", "release")
APPROVALS = {"holdout": "a holdout look", "amend": "an amendment", "approval": "an approval", "budget": "more budget",
             "screen": "a screening decision", "screen_conflict": "a screening conflict", "screen_maybe": "a maybe",
             "fresh": "a fresh question", "continues": "a shared ledger", "compute": "another machine",
             "experiment": "an experiment's start", "other": "a request"}


def _store(ctx: ToolContext):
    from ..agents.store import Store, root_of
    return Store(root_of(ctx.root))


def _toml(root: Path) -> dict:
    p = Path(root) / "openlab.toml"
    if not p.is_file():
        return {}
    try:
        import tomllib
    except ImportError:                                                     # python 3.10
        import tomli as tomllib                                             # type: ignore
    try:
        return tomllib.loads(p.read_text(encoding="utf-8"))
    except Exception:                                                       # noqa: BLE001 -- a broken file is no file
        return {}


def _table(d: dict, key: str) -> dict:
    """`d[key]` when it is a table, else {} (a hand-edited openlab.toml may say `team = "x"`)."""
    v = d.get(key)
    return v if isinstance(v, dict) else {}


def _num(v) -> float:
    try:
        return float(v) if not isinstance(v, bool) else 0.0
    except (TypeError, ValueError):
        return 0.0


def people(root: Path) -> list:
    """The person who holds the key (`you`), then everyone declared under [team.<handle>] in openlab.toml (role,
    hours a week, away until)."""
    from ..agents.identity import person_label
    out = [{"id": "you", "name": person_label(), "role": "holds the person key", "state": "available", "hours": 0.0,
            "cap": 0.0, "key": True, "q": "signs looks, approvals and amendments"}]
    for handle, t in _table(_toml(root), "team").items():
        if not isinstance(t, dict) or handle == "you":
            continue
        away = str(t.get("away_until") or "")
        out.append({"id": str(handle), "role": str(t.get("role") or "contributor"), "state": "away" if away else "available",
                    **({"until": away} if away else {}), "hours": _num(t.get("hours_used")),
                    "cap": _num(t.get("hours")), "key": t.get("key") is True, "q": str(t.get("does") or "")})
    return out


def _now_of(store, row: dict, recent: Optional[list] = None) -> dict:
    """What the agent is doing now, from its own events: the tool call it is in, the run it waits on, between turns,
    waiting for the person, or ended."""
    a, state = row["agent"], row["state"]
    if state in ("done", "failed", "stopped", "lost", "ended"):
        evs = recent if recent is not None else store.events(a["id"], n=12)
        final = next((e for e in reversed(evs[-12:]) if e["kind"] in ("result", "end", "say")), None)
        return {"kind": "done", "op": f"exited: {a['status']}" + _limits_used(a), "since": a["updated"],
                "note": _final_text(store, a) or (final or row["last"] or {}).get("summary", "")}
    if state == "paused":
        why = a.get("control_reason") or ""
        if why.startswith("budget:"):
            return {"kind": "blocked", "op": "paused at its limit: raise it in the inbox", "since": a["updated"],
                    "note": why[7:].strip()}
        return {"kind": "idle", "op": "paused by the person", "since": a["updated"], "note": why}
    if state == "waiting":
        return {"kind": "blocked", "op": "waiting for the person", "since": a["last_event_at"] or a["updated"],
                "note": f"{row['pending']} inbox item{'s' if row['pending'] != 1 else ''} pending" if row["pending"] else ""}
    run = next((f for f in row["flags"] if f.startswith(("RUN", "QUEUED"))), None)
    evs = (recent if recent is not None else store.events(a["id"], n=8))[-8:]
    last = evs[-1] if evs else None
    if run:
        r = next((e for e in reversed(evs) if e["kind"] == "run"), last) or {}
        return {"kind": "blocked" if run.startswith("QUEUED") else "tool", "op": r.get("summary") or "an openlab run",
                "since": r.get("at") or a["updated"], "note": run.lower()}
    if state in ("working", "stuck") and last is not None and last["kind"] == "tool":
        return {"kind": "tool", "op": last["summary"] or last["tool"], "since": last["at"],
                "note": "no event for a while" if state == "stuck" else ""}
    if state in ("working", "stuck"):
        return {"kind": "tool", "op": "thinking (a model turn)", "since": (last or {}).get("at") or a["updated"], "note": ""}
    return {"kind": "idle", "op": "between turns", "since": a["last_event_at"] or a["updated"],
            "note": "it wakes on a tell, a comment or its parent"}


def _limits_used(a: dict) -> str:
    """For an ended agent, the limits it had (nearly) used, e.g. " at 226 of 240 min": why it may have wrapped up."""
    from ..agents.store import age
    b, s = a.get("budget") or {}, a.get("spent") or {}
    try:
        from datetime import datetime
        mins = (datetime.fromisoformat(a["updated"]) - datetime.fromisoformat(a["created"])).total_seconds() / 60
    except (KeyError, TypeError, ValueError):
        mins = age(a["created"]) / 60
    used = {"minutes": (mins - (s.get("paused_minutes") or 0), "{:.0f} of {:g} min"),
            "usd": (s.get("usd") or 0, "${:.2f} of ${:g}"), "turns": (s.get("turns") or 0, "{:.0f} of {:g} turns")}
    out = [f.format(n, float(b[k])) for k, (n, f) in used.items() if b.get(k) and n >= 0.9 * float(b[k])]
    return f" at {', '.join(out)}" if out else ""


def _final_text(store, a: dict) -> str:
    """The agent's last message in full (its transcript's result), for the person who reads why it stopped."""
    import json
    try:
        p = store.agent_dir(a["id"]) / "transcript.jsonl"
        if not p.is_file():
            return ""
        with open(p, "rb") as f:                            # the tail only: a transcript grows to megabytes
            f.seek(max(0, f.seek(0, 2) - TAIL_BYTES))
            tail = f.read().decode("utf-8", errors="replace")
        for line in reversed(tail.splitlines()[-200:]):
            if '"type":"result"' in line.replace(" ", ""):
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                return str(d.get("result") or "")[:4000] if isinstance(d, dict) else ""
    except (OSError, ValueError):
        return ""
    return ""


CONV_MAX = 1500             # entries of a conversation one reply carries at most (the latest kept)
CONV_TEXT = 8000            # characters of a message, or of a tool call's input, kept
CONV_OUT = 3000             # characters of a tool's result kept


def _clip(s, n: int) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n] + f"… ({len(s) - n} more characters)"


def _plan_sets(name: str, inp) -> list:
    """The (step, state) pairs a plan call sets: agent_plan_node one, agent_plan each step it lists (openlab_tool
    unwrapped)."""
    if not isinstance(inp, dict):
        return []
    short = name.rsplit("__", 1)[-1].removeprefix("openlab_")
    if short == "tool":
        short, inp = str(inp.get("name") or ""), inp.get("args") or {}
        if not isinstance(inp, dict):
            return []
    if short == "agent_plan_node":
        return [(inp.get("id"), inp.get("state") or "")]
    if short == "agent_plan":
        return [(n.get("id"), n.get("state") or "planned") for n in inp.get("nodes") or [] if isinstance(n, dict)]
    return []


def _result_text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") if b.get("type") == "text" else f"[{b.get('type')}]"
                     for b in content or [] if isinstance(b, dict))


def conversation(store, agent: str, node: str = "") -> dict:
    """Agent `agent`'s conversation from its transcript: its prompt, what it said, each tool call with its input and
    result, how it ended. With `node`, up to and including the call that last marked that step of its plan done
    (`cut`); the whole conversation while the step is not done."""
    import json
    from .registry import ToolError
    a = store.get(agent)
    if a is None:
        raise ToolError(f"no agent {agent!r}", "", "Activity lists the agents")
    d = store.dir / "agents" / agent.replace("/", "_")
    entries, calls = [], {}                         # calls: tool_use id -> (entry, the plan states it sets)
    prompt = d / "prompt.txt"
    if prompt.is_file():
        entries.append({"kind": "prompt", "at": a.get("created"), "text": _clip(prompt.read_text(encoding="utf-8", errors="replace"), CONV_TEXT)})
    state, cut, sub = "", None, {}
    try:
        f = open(d / "transcript.jsonl", encoding="utf-8", errors="replace")
    except OSError:
        f = None
    for line in f or []:
        if not line.startswith('{"type":"assistant"') and not line.startswith('{"type":"user"') \
                and not line.startswith('{"type":"result"'):
            continue
        try:
            o = json.loads(line)
        except ValueError:
            continue
        parent = o.get("parent_tool_use_id")
        if parent:                                  # a native sub-agent's turn: counted on the call that started it
            sub[parent] = sub.get(parent, 0) + 1
            continue
        at = o.get("timestamp")
        if o["type"] == "result":
            entries.append({"kind": "end", "at": at, "text": _clip(o.get("result") or o.get("subtype") or "", CONV_TEXT)})
            continue
        content = (o.get("message") or {}).get("content")
        for b in [{"type": "text", "text": content}] if isinstance(content, str) else content or []:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text" and (b.get("text") or "").strip():
                entries.append({"kind": "say" if o["type"] == "assistant" else "user", "at": at,
                                "text": _clip(b["text"], CONV_TEXT)})
            elif t == "tool_use":
                e = {"kind": "call", "at": at, "tool": str(b.get("name") or "").removeprefix("mcp__openlab__"),
                     "input": _clip(json.dumps(b.get("input"), ensure_ascii=False), CONV_TEXT)}
                entries.append(e)
                calls[b.get("id")] = (e, _plan_sets(str(b.get("name") or ""), b.get("input")))
            elif t == "tool_result" and b.get("tool_use_id") in calls:
                e, sets = calls[b["tool_use_id"]]
                e["result"] = _clip(_result_text(b.get("content")), CONV_OUT)
                e["error"] = bool(b.get("is_error"))
                for step, st in sets if node and not e["error"] else ():     # a refused call set nothing
                    if step == node:
                        if st == "done" and state != "done":
                            cut = e
                        state = st
    if f is not None:
        f.close()
    for tid, n in sub.items():
        if tid in calls:
            calls[tid][0]["sub"] = n
    total = len(entries)
    if node and state == "done" and cut is not None:
        entries = entries[:entries.index(cut) + 1]
        cut["cut"] = True
    shown = entries[-CONV_MAX:]
    return {"agent": agent, "role": a.get("role"), "status": a.get("status"), "node": node, "state": state,
            "cut": (cut or {}).get("at") if node and state == "done" and cut is not None else None,
            "entries": shown, "left_out": len(entries) - len(shown), "after": total - len(entries)}


@tool("research", human=True)
def show_conversation(ctx: ToolContext, agent: str, node: str = "") -> dict:
    """An agent's conversation from its transcript (prompt, messages, tool calls with input and result); with `node`,
    up to the call that last marked that step of its plan done. A person's read: agents never call it."""
    return conversation(_store(ctx), agent, node)


def _state(row: dict) -> str:
    if any(f.startswith("QUEUED") for f in row["flags"]):
        return "queued"
    return row["state"]


def _recent(store, ids: list) -> dict:
    """{agent: (its last 12 events, its last 5 refusals, {kind: count} of its refusals and bypass attempts, its last 8
    of them as {kind, at, text})}, read over one connection."""
    import json
    from contextlib import closing
    out = {}
    with closing(store._con()) as con:
        for a in ids:
            evs = con.execute("SELECT * FROM events WHERE agent=? ORDER BY seq DESC LIMIT 12", (a,)).fetchall()
            refs = con.execute("SELECT kind, at, summary FROM events WHERE agent=? AND kind IN ('refusal','bypass_attempt') "
                               "ORDER BY seq DESC LIMIT 8", (a,)).fetchall()
            n = dict(con.execute("SELECT kind, COUNT(*) FROM events WHERE agent=? AND kind IN ('refusal','bypass_attempt') "
                                 "GROUP BY kind", (a,)).fetchall())
            out[a] = ([{**dict(r), "payload": json.loads(r["payload"])} for r in reversed(evs)],
                      [r["summary"] for r in reversed(refs[:5])], n,
                      [{"kind": r["kind"], "at": r["at"], "text": r["summary"]} for r in reversed(refs)])
    return out


def sessions(store, snap: dict) -> list:
    out = []
    rows = [r for r in snap["rows"] if r["agent"]["kind"] != "claude-subagent"]   # a sub-agent's spend is its parent's
    recent = _recent(store, [r["agent"]["id"] for r in rows])
    for r in rows:
        a = r["agent"]
        s = a["spent"] or {}
        evs, refusals, n, attempts = recent[a["id"]]
        out.append({"id": a["id"], "role": a["role"] or a["kind"], "state": _state(r), "kind": a["kind"],
                    "usd": round(r["usd"], 4), "usd_estimated": r["usd_estimated"], "budget": r["budget_usd"] or 0,
                    "parent": a["parent"] or None, "study": a["study"] or "", "q": a["question"] or "",
                    "model": a["model"] or a["cli"] or "", "cli": a["cli"], "started": a["created"],
                    "heard": a["last_event_at"] or a["updated"], "turns": s.get("turns") or 0,
                    "tools": s.get("tool_calls") or 0, "flags": r["flags"], "refusals": refusals,
                    "refused": n.get("refusal", 0), "bypass": n.get("bypass_attempt", 0), "attempts": attempts,
                    "trials": r["trials"], "claims": r["claims"], "now": _now_of(store, r, evs),
                    "lead": store.is_lead(a), "held": a["control"] or ""})
    return out


_PLANS: dict = {}           # {(root, agent): (key, versions)}: a plan is read again only when it or what it cites changed


def _decided(n: dict, items: dict) -> None:
    """A step the agent left `waiting` on the person is drawn as the person decided it, once every inbox item it
    links is decided: dropped if one was denied, else done. The agent's word stays in `said`; its plan is not
    rewritten."""
    if n.get("state") != "waiting":
        return
    refs = [str(x)[6:] for x in (n.get("links") or []) if str(x).startswith("inbox:")]
    its = [items.get(int(r)) for r in refs if r.isascii() and r.isdigit() and len(r) <= 18]
    if not its or len(its) != len(refs) or any(i is None or i["status"] == "pending"
                                                 or (i["result"] or {}).get("running") for i in its):
        return
    n["said"] = n["state"]
    n["state"] = "dropped" if any(i["status"] == "denied" for i in its) else "done"
    by = "; ".join(f"inbox #{i['id']} {i['status']} by {i['decided_by']}" for i in its)
    n["evidence"] = f"the agent said waiting; {by}" + (f" — {n['evidence']}" if n.get("evidence") else "")


def _unfiled(n: dict) -> None:
    """A step the agent calls `waiting` with nothing linked (no inbox item, no ticket) waits on nothing anyone can
    act on: drawn as planned, saying so. (openlab now refuses such a step; older plans may hold one.)"""
    if n.get("state") != "waiting" or n.get("links"):
        return
    n["said"] = "waiting"
    n["state"] = "planned"
    n["evidence"] = ("the agent said waiting, but nothing is filed: no inbox item or ticket is linked, so nothing waits "
                     "on you" + (f" — {n['evidence']}" if n.get("evidence") else ""))


def _ended(n: dict, a: dict) -> None:
    """A step the agent left open (planned, active or waiting) when it ended is drawn as dropped: nobody works on it
    now. The agent's word stays in `said`; its plan is not rewritten."""
    if a.get("state") not in FINAL_STATES or n.get("state") not in ("planned", "active", "waiting"):
        return
    n["said"] = n["state"]
    n["state"] = "dropped"
    n["evidence"] = (f"the agent said {n['said']}; {a['id']} ended ({a['state']}) with the step open"
                     + (f" — {n['evidence']}" if n.get("evidence") else ""))


def _stamp(p: Path) -> tuple:
    try:
        st = p.stat()
        return st.st_size, st.st_mtime_ns
    except OSError:
        return None, None


def plans(store, agents: list, items: Optional[list] = None) -> dict:
    """Each agent's plan versions, every step with its evidence. Cached per agent by the plan file's size and mtime,
    the head of its study's record and the inbox (what the evidence reads)."""
    import copy

    from ..agents.plan import _path, _record, evidence, read
    items = {i["id"]: i for i in (items if items is not None else store.inbox(None))}
    inbox_key = tuple(sorted((k, i["status"]) for k, i in items.items()))
    out, records, heads = {}, {}, {}
    for a in agents:
        stamp = _stamp(_path(store, a["id"]))
        if stamp[0] is None:
            continue
        if a["study"] not in records:
            records[a["study"]] = rec = _record(store.root, a["study"])
            try:
                heads[a["study"]] = (rec.head(), _stamp(rec.path)) if rec is not None else ("", None)
            except Exception:                                               # noqa: BLE001
                heads[a["study"]] = ("?", None)
        key = (stamp, a["study"], heads[a["study"]], inbox_key)
        hit = _PLANS.get((str(store.root), a["id"]))
        if hit is not None and hit[0] == key:
            out[a["id"]] = copy.deepcopy(hit[1])
            continue
        versions = read(store, a["id"])["versions"]
        if not versions:
            continue
        rec = records[a["study"]]
        for v in versions:
            for n in v["nodes"]:
                n["ev"], n["evidence"] = evidence(store, a["study"], n, rec, items)
                if v is versions[-1]:
                    _decided(n, items)
        _PLANS[(str(store.root), a["id"])] = (key, copy.deepcopy(versions))
        out[a["id"]] = versions
    for a in agents:                                # after the cache: an agent's end changes no plan file
        for n in out.get(a["id"], [{}])[-1].get("nodes", []):
            _unfiled(n)
            _ended(n, a)
    return out


def feed(store) -> list:
    return [{"seq": e["seq"], "at": e["at"], "who": e["agent"], "kind": e["kind"],
             "text": e["summary"] or e["tool"] or e["kind"]} for e in reversed(store.events(n=FEED_MAX, kinds=FEED_KINDS))]


def _limit(i: dict) -> dict:
    """A budget request's one changed limit, for the approval form's field: {limit: {key, old, new, hold}}."""
    change = (i["args"] or {}).get("change") or {}
    if i["kind"] != "budget" or len(change) != 1:
        return {}
    (k, (old, new)), = change.items()
    return {"limit": {"key": k, "old": old, "new": new, "hold": bool((i["args"] or {}).get("hold"))}}


def inbox(store, every: Optional[list] = None) -> tuple:
    def item(i):
        args = i["args"] or {}
        q = i["kind"] == "question"
        return {"id": i["id"], "type": "choice" if q else "approval", "kind": i["kind"], "from": i["agent"],
                "at": i["created"], "person": not q, "what": i["reason"], "detail": i["cost_note"] or "",
                "study": i["study"], **({"choices": args.get("choices") or [], "multi": bool(args.get("multi")),
                                         "other": bool(args.get("other"))} if q else {"asks": APPROVALS.get(i["kind"], i["kind"])}),
                **_limit(i), **({"harness": lesson_evidence(store.root, args["lesson"])}
                                if i["kind"] == "harness" and args.get("action") == "lesson" and i["status"] == "pending" else {})}
    every = every if every is not None else store.inbox(None)
    pending = [item(i) for i in every if i["status"] == "pending"]
    done = [i for i in every if i["status"] != "pending"][-ANSWERED_MAX:]
    answered = [{**item(i), "decision": {"approved": "approve", "denied": "deny", "answered": "answer"}.get(i["status"], i["status"]),
                 "message": i["decision_reason"], "picks": (i["result"] or {}).get("picks") or [],
                 "other": (i["result"] or {}).get("other") or "", "answeredAt": i["decided_at"], "by": i["decided_by"]}
                for i in reversed(done)]
    return pending, answered


#: how many signals a lesson's evidence shows in the inbox: the ones it predicts, then the ones that moved
EVIDENCE_ROWS = 6


def lesson_evidence(root: Path, lid: str) -> dict:
    """What a harness lesson waiting in the inbox rests on, for the person deciding it: its cue and advice, and its
    basis contrast's signals (each side's mean, and whether the difference is past the noise), predicted ones first."""
    from ..harness import lessons, signals
    try:
        body = lessons.get(root, lid)
    except Exception:                                                       # noqa: BLE001 -- the item still draws
        return {}
    pred = (body.get("predicted") or {}).get("signals") or {}
    out = {"id": lid, "cue": body.get("cue", ""), "advice": body.get("advice", ""), "component": body.get("component", ""),
           "version": (body.get("edit") or {}).get("version", ""), "quarantined": lessons.status(root, lid) == "quarantined",
           "n": [0, 0], "rows": []}
    for cid in body.get("basis") or []:
        try:
            c = signals.load_contrast(root, cid)
        except KeyError:
            continue
        rows = [r for r in c["rows"] if r.get("delta") is not None]
        rows.sort(key=lambda r: (r["field"] not in pred, not r.get("moved")))
        out["n"], out["contrast"] = c["n"], cid
        out["rows"] = [{"what": r["what"], "ref": r["ref"], "cand": r["cand"], "moved": bool(r.get("moved")),
                        "favours": r.get("favours", ""), "predicted": pred.get(r["field"], "")}
                       for r in rows[:EVIDENCE_ROWS]]
        break
    return out


def harness(root: Path) -> dict:
    """The harness versions (Settings): each stored one with its note, when it was stored, whether it is active, and
    whether it may become active (it was active before, or an approved lesson supports it)."""
    from ..harness import versions
    try:
        default = versions.default_sha()
        return {"versions": [{**r, "may": r["active"] or r["sha"] == default or bool(versions._supported(root, r["sha"]))}
                             for r in versions.stored(root)]}
    except Exception:                                                       # noqa: BLE001 -- the page still draws
        return {"versions": []}


@tool("research", mutates=True, human=True)
def show_harness_use(ctx: ToolContext, ref: str, reason: str) -> dict:
    """A person makes a harness version active from openlab show (Settings), signed with the person key: new agents
    are briefed by it; running ones keep theirs. Refused for a version no approved lesson supports and that was
    never active."""
    from ..harness import HarnessError, versions
    from .registry import ToolError
    try:
        return versions.use(_store(ctx).root, ref, reason)
    except HarnessError as e:
        raise ToolError(e.problems[0], "; ".join(e.problems[1:]), "approve a lesson that supports it first") from None
    except PermissionError as e:
        raise ToolError(str(e)) from None


def queue() -> list:
    import time
    try:
        from ..study.admission import tickets
        rows = tickets()
    except Exception:                                                       # noqa: BLE001
        return []
    out = []
    for t in rows:
        since = t["started"] if t["state"] == "running" and t["started"] else t["at"]
        out.append({"n": t["id"], "state": t["state"], "for": f"{(time.time() - (since or time.time())) / 60:.1f}m",
                    "needs": f"{t['cores'] or 0}c {t['memory_gb'] or 0:g}GB {t['gpus'] or 0}gpu ~{t['minutes'] or 0:g}m",
                    "run": t["label"] or "", "agent": t["agent"] or "", "why": t["why"] or ""})
    return out


def box() -> dict:
    try:
        from ..boxcli import show
        d = show()
    except Exception:                                                       # noqa: BLE001
        d = {}
    gpus = d.get("gpus") or []
    sched = d.get("schedule") or {}
    import socket
    from datetime import datetime
    until = sched.get("available_until")
    return {"host": socket.gethostname(), "gpus": f"{len(gpus)}× {gpus[0].get('name', 'GPU')}" if gpus else "no GPU",
            "cores": d.get("cores"), "memory_gb": d.get("memory_gb"),
            "until": datetime.fromtimestamp(until).strftime("%Y-%m-%d %H:%M") if until else "",
            "reservations": [{"who": r.get("who") or "", "start": datetime.fromtimestamp(r["start"]).strftime("%m-%d %H:%M"),
                              "end": datetime.fromtimestamp(r["end"]).strftime("%m-%d %H:%M"), "gpus": r.get("gpus") or 0}
                             for r in sched.get("reservations") or []],
            "running": (d.get("queue") or {}).get("running", 0), "waiting": (d.get("queue") or {}).get("waiting", 0)}


MODELS = ["", "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-1"]
#: what Settings shows and edits: (key, table, label, kind, default, when, help, choices). `when`: "live" may change
#: at any time, from the next agent or check on; "new" is chosen in New session and fixed for the research's life, so
#: every agent of it ran under the same rules.
SETTINGS = [
    ("budget_usd", "agents", "Budget of a spawned agent, $", "usd", None, "live", "an agent spawned without a budget gets this", None),
    ("budget_minutes", "agents", "Budget of a spawned agent, minutes", "int", None, "live", "", None),
    ("budget_turns", "agents", "Budget of a spawned agent, turns", "int", None, "live", "", None),
    ("cap_usd", "agents", "Cap of a research, $", "usd", None, "live", "the most its lead and every branch together may budget", None),
    ("cap_minutes", "agents", "Cap of a research, minutes", "int", None, "live", "", None),
    ("max_children", "agents", "Children per agent", "int", None, "live", "live children one agent may have at once", None),
    ("max_depth", "agents", "Depth of delegation", "int", None, "live", "how deep a tree of delegated agents may go", None),
    ("model", "agents", "Model of spawned agents", "choice", "", "live", "blank: the agent CLI's own default", MODELS),
    ("stuck_minutes", "agents", "Flag STUCK after, minutes", "int", None, "live", "a working agent with no event this long", None),
    ("session_timeout_hours", "agents", "End a quiet session after, hours", "int", None, "live", "a Claude Code session with no event this long is ended", None),
    ("loop_refusals", "agents", "Flag LOOP after, refusals", "int", None, "live", "the same refusal this many times", None),
    ("compile_on_done", "show", "Compile the paper when the lead ends", "bool", True, "live", "the white paper and the code folder, into release/", None),
    ("permission_mode", "agents", "Agents' permission mode", "choice", None, "new", "what an agent may do without asking (claude --permission-mode)", ["acceptEdits", "default", "plan"]),
    ("allow_native_subagents", "agents", "Native sub-agents", "bool", None, "new", "let agents delegate with the CLI's own Agent/Task tool, outside openlab's budgets", None),
]


def _default(key: str, given):
    from ..agents.store import DEFAULTS
    return given if given is not None else DEFAULTS.get(key)


def settings(root: Path) -> list:
    """Each setting of Settings: {key, table, label, kind, value, default, is_default, when, help, choices, where}."""
    from ..agents.store import config
    c, t = config(root), _toml(root)
    rows = []
    for key, table, label, kind, default, when, help_, choices in SETTINGS:
        d = _default(key, default)
        if table == "agents":
            v = c.get(key, d)
            v = "" if v is None else v
        else:
            v = _table(t, table).get(key, d)
            v = v if isinstance(v, type(d)) else d
        rows.append({"key": key, "table": table, "label": label, "kind": kind, "value": v, "default": "" if d is None else d,
                     "is_default": v == ("" if d is None else d), "when": when, "help": help_, "choices": choices,
                     "where": f"openlab.toml [{table}]"})
    if c.get("notify"):
        rows.append({"key": "notify", "table": "agents", "label": "Notify", "kind": "text", "value": str(c["notify"]),
                     "default": "", "is_default": False, "when": "file", "help": "edit openlab.toml", "choices": None,
                     "where": "openlab.toml [agents]"})
    return rows


def _check(key: str, value):
    """`value` as the setting `key` takes it, or ValueError."""
    row = next((r for r in SETTINGS if r[0] == key), None)
    if row is None:
        raise ValueError(f"{key!r} is not a setting this page edits")
    kind, choices = row[3], row[7]
    if kind == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{key} is true or false")
        return value
    if kind == "choice":
        if value not in choices:
            raise ValueError(f"{key} is one of {', '.join(repr(c) for c in choices)}")
        return value
    try:
        v = float(value) if kind == "usd" else int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} is a number") from None
    if v < 0 or (kind == "int" and v < 1 and key not in ("max_depth",)):
        raise ValueError(f"{key} is {'at least 1' if kind == 'int' else 'not negative'}")
    return v


def _split_comment(rest: str) -> tuple:
    q = None
    for i, ch in enumerate(rest):
        if q:
            q = None if ch == q else q
        elif ch in "\"'":
            q = ch
        elif ch == "#":
            return rest[:i].rstrip(), rest[i:]
    return rest.rstrip(), ""


def toml_set(text: str, table: str, key: str, value) -> str:
    """`text` (a TOML file) with `key` of `[table]` set to `value`, every comment kept: the key's line rewritten (a
    commented-out one is uncommented), else a line added at the table's end, else the table appended."""
    import json
    import re
    lit = ("true" if value else "false") if isinstance(value, bool) else repr(value) if isinstance(value, (int, float)) \
        else json.dumps(value)
    lines = text.splitlines(keepends=True)
    hdr = re.compile(r"^\s*\[([^\[\]]+)\]\s*(#.*)?$")
    start, end = None, len(lines)
    for i, ln in enumerate(lines):
        m = hdr.match(ln)
        if m and start is not None:
            end = i
            break
        if m and m.group(1).strip() == table:
            start = i
    if start is None:
        return text + ("" if text.endswith("\n") or not text else "\n") + f"\n[{table}]\n{key} = {lit}\n"
    kre = re.compile(rf"^(\s*)(#\s*)?{re.escape(key)}\s*=\s*(.*?)\s*$")
    hit = None
    for i in range(start + 1, end):
        m = kre.match(lines[i].rstrip("\n"))
        if m and (hit is None or (lines[hit].lstrip().startswith("#") and not m.group(2))):
            hit = i
    if hit is not None:
        m = kre.match(lines[hit].rstrip("\n"))
        _, comment = _split_comment(m.group(3))
        body = f"{m.group(1)}{key} = {lit}"
        lines[hit] = (body.ljust(26) + " " + comment if comment else body) + "\n"
        return "".join(lines)
    at = end
    while at > start + 1 and not lines[at - 1].strip():
        at -= 1
    lines.insert(at, f"{key} = {lit}\n")
    return "".join(lines)


def set_setting(root: Path, key: str, value) -> tuple:
    """Write one setting into openlab.toml; returns (old value, new value). Refused unless the file still parses and
    reads the new value back."""
    try:
        import tomllib
    except ImportError:                                                     # python 3.10
        import tomli as tomllib                                             # type: ignore
    row = next(r for r in SETTINGS if r[0] == key)
    table = row[1]
    p = Path(root) / "openlab.toml"
    text = p.read_text(encoding="utf-8") if p.is_file() else ""
    try:
        old = _table(tomllib.loads(text), table).get(key, _default(key, row[4]))
    except ValueError:
        raise ValueError("openlab.toml does not parse: fix it by hand first") from None
    new = toml_set(text, table, key, value)
    try:
        back = _table(tomllib.loads(new), table).get(key)
    except ValueError:
        back = object()
    if back != value:
        raise ValueError(f"could not write {key} into openlab.toml safely: edit it by hand")
    tmp = p.with_name(".openlab.toml.save")
    tmp.write_text(new, encoding="utf-8")
    tmp.replace(p)
    return old, value


@tool("research", mutates=True, human=True)
def show_settings_set(ctx: ToolContext, key: str, value) -> dict:
    """A person changes a setting of this research (Settings in openlab show): one of those marked live, written into
    openlab.toml with its comments kept, and noted on the control store."""
    from .agent_tools import _person
    from .registry import ToolError
    row = next((r for r in SETTINGS if r[0] == key), None)
    if row is None:
        raise ToolError(f"{key!r} is not a setting this page edits", "", "Settings lists them")
    if row[5] != "live":
        raise ToolError(f"{key} is fixed for this research", "it is chosen in New session, so every agent of a "
                        "research runs under the same rules", "start a new session to change it")
    try:
        old, new = set_setting(_store(ctx).root, key, _check(key, value))
    except ValueError as e:
        raise ToolError(str(e)) from None
    st, by = _store(ctx), _person()
    st.event(by, "setting", summary=f"{by} set {key}: {old!r} -> {new!r}", payload={"key": key, "old": old, "new": new})
    return {"key": key, "old": old, "value": new, "settings": settings(st.root)}
_LEDGER: dict = {}          # {study folder: (key, count, rows)}: a record is read again only when it moved


def _ledger_rows(folder: Path, name: str) -> tuple:
    """(its node count, the newest LEDGER_MAX rows of one record, oldest first), cached by its head and file stamp."""
    import json
    import sqlite3
    from contextlib import closing

    from ..study.show import line
    db = Path(folder) / ".openlab" / "record.db"
    if not db.is_file():
        return 0, []
    with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)) as con:
        con.row_factory = sqlite3.Row
        head = con.execute("SELECT seq, hash FROM nodes ORDER BY seq DESC LIMIT 1").fetchone()
        key = (name, tuple(head) if head else None, _stamp(db))
        hit = _LEDGER.get(str(folder))
        if hit is not None and hit[0] == key:
            return hit[1], hit[2]
        total = con.execute("SELECT count(*) FROM nodes").fetchone()[0]
        raw = con.execute("SELECT * FROM nodes ORDER BY seq DESC LIMIT ?", (LEDGER_MAX,)).fetchall()
        tests, breaks, needs = {}, {}, {}                   # {hypothesis: its tests}, {assumption: breaks, claims}
        for h in (r["id"] for r in raw if r["kind"] == "hypothesis"):
            tests[h] = con.execute("SELECT n.id, n.payload FROM edges e JOIN nodes n ON n.id = e.src WHERE e.dst = ? "
                                   "AND e.rel = 'of' AND n.kind = 'test' ORDER BY n.seq", (h,)).fetchall()
        for a in (r["id"] for r in raw if r["kind"] == "assumption"):
            breaks[a] = [x[0] for x in con.execute("SELECT src FROM edges WHERE dst = ? AND rel = 'breaks' "
                                                   "ORDER BY rowid", (a,))]
            needs[a] = con.execute("SELECT count(*) FROM edges WHERE dst = ? AND rel = 'needs'", (a,)).fetchone()[0]
    rows = []
    for r in reversed(raw):
        try:
            p = json.loads(r["payload"])
        except ValueError:
            p = {}
        try:
            text = line({"kind": r["kind"], "payload": p})
        except Exception:                                                   # noqa: BLE001
            text = ""
        rows.append({"study": name, "seq": r["seq"], "at": r["at"], "who": r["session"] or r["actor"],
                     "actor": r["actor"], "kind": r["kind"], "id": r["id"], "text": text, "sig": r["actor"] == "person",
                     "bad": r["kind"] == "run" and isinstance(p, dict)
                     and (p.get("audit") or {}).get("verdict") == "look-ahead",
                     "prev": r["prev"], "hash": r["hash"]})
        if r["kind"] == "hypothesis":
            rows[-1]["tests"] = [_test_of(t) for t in tests.get(r["id"], [])]
        if r["kind"] == "assumption":
            rows[-1]["breaks"], rows[-1]["needed_by"] = breaks.get(r["id"], []), needs.get(r["id"], 0)
    _LEDGER[str(folder)] = (key, total, rows)
    return total, rows


def _test_of(row) -> dict:
    """A test of a hypothesis, as RAID shows it: whether it rejected the null, and its p."""
    import json
    try:
        p = json.loads(row[1])
    except ValueError:
        p = {}
    p = p if isinstance(p, dict) else {}
    return {"id": row[0], "rejected": p.get("rejected"), "p": p.get("p")}


def ledger(root: Path, folders: Optional[list] = None) -> dict:
    """The studies' records, newest last: each node's seq, time, agent, kind, id, line, whether a person signed it and
    whether the record disagrees with itself (a run whose audit found look-ahead); with each record's head. Only the
    newest LEDGER_MAX nodes of each record are read: the page shows LEDGER_MAX in all."""
    from ..study.workspace import studies
    rows, heads, total = [], [], 0
    base = Path(root).resolve()
    for folder in (folders if folders is not None else studies(root)):
        try:
            f = Path(folder).resolve()
            name = str(f.relative_to(base)) if base in f.parents else f.name
            n, owned = _ledger_rows(f, name)
        except Exception:                                                   # noqa: BLE001 -- a broken record is skipped
            continue
        heads.append({"study": name, "head": owned[-1]["hash"] if owned else "", "nodes": n})
        rows += owned
        total += n
    rows.sort(key=lambda x: x["at"])
    return {"entries": rows[-LEDGER_MAX:], "heads": heads, "total": total}


@tool("research")
def show_snapshot(ctx: ToolContext) -> dict:
    """Everything `openlab show` draws, in one read: agents and what each is doing now, people, plans with evidence,
    the feed, the inbox, the queue, the settings and the records."""
    from ..agents.store import now
    from ..agents.view import snapshot
    from ..study.workspace import studies
    store = _store(ctx)
    try:
        folders = studies(store.root)
    except Exception:                                                       # noqa: BLE001
        folders = []
    with store.reading():                                   # one connection for the hundred small reads below
        snap = snapshot(store, all=True, studies=folders)
        agents = sessions(store, snap)
        every = store.inbox(None)
        pending, answered = inbox(store, every)
        return {"now": now(), "workspace": str(store.root), "box": box(), "spend": round(snap["usd"], 4),
                "spend_estimated": snap["usd_estimated"], "sessions": agents, "people": people(store.root),
                "plans": plans(store, agents, every), "feed": feed(store), "inbox": pending, "answered": answered,
                "queue": queue(), "settings": settings(store.root), "ledger": ledger(store.root, folders),
                "experiments": experiments(store, folders), "instruments": instruments(store.root, folders),
                "studies": seals(store.root, folders), "harness": harness(store.root), "control_head": store.head()}


def seals(root: Path, folders: list) -> list:
    """Every study with its seal (openlab.study.sealing.overview): an unsealed one with what sealing it freezes and
    costs, and whether it can be sealed now (the Not sealed badge and the Seal button)."""
    from ..study import sealing
    try:
        return sealing.overview(root, folders)
    except Exception:                                                       # noqa: BLE001 -- the page still draws
        return []


@tool("research", mutates=True, human=True)
def show_study_seal(ctx: ToolContext, study: str, note: str = "") -> dict:
    """A person seals a study from openlab show (the Seal button, after its confirm), signed with the person key:
    the study's first research run, which writes study.toml into the record as the plan and is counted as trial 1;
    the holdout is not touched. Refused before anything runs when it is already sealed, a review study, or its data
    is not materialised. Returns the trial, the result id and the primary metric."""
    from .. import study as _study
    from ..study import sealing
    from ..study.errors import StudyError
    from .agent_tools import _person
    from .registry import ToolError
    folder = ctx.path(study or ".")
    if not (folder / "study.toml").is_file():
        raise ToolError(f"{study!r} is not a study of this research", "", "the Workspace lists the studies")
    rel = ctx.rel(folder) if folder != ctx.root else "."
    try:
        out = _study.open(folder).seal_run(note=note.strip() or "sealed in openlab show", actor="person", rel=rel)
    except StudyError as e:
        raise ToolError(e.error, e.meaning, e.fix) from None
    brief = {"study": rel, **sealing.summary(out), "audit": out.get("audit")}
    try:
        st, by = _store(ctx), _person()
        st.event(by, "note", summary=f"{by} sealed {rel}: {brief['plan']}, {brief['result']}",
                 payload={"study": rel, "plan": brief["plan"], "result": brief["result"]})
    except Exception:                                                       # noqa: BLE001 -- the seal stands
        pass
    return brief


def experiments(store, folders: list) -> list:
    """Every experiment of the studies, as `experiment status` reads it (openlab.study.experiments.view): the
    Experiments tab draws them as they are."""
    from ..study import experiments as E
    try:
        return E.overview(store.root, folders, store=store)
    except Exception as e:                                                  # noqa: BLE001 -- the page still draws
        return [{"study": "", "name": "", "state": "error", "spec_error": str(e)}]


def instruments(root: Path, folders: list) -> list:
    """Every registered instrument version, with its runs, its last run, the studies pinning it and its usage's head
    (the Experiments tab; the whole usage is `instrument_show` with usage_only)."""
    from ..study import experiments as E
    try:
        return E.instruments_overview(root, folders)
    except Exception:                                                       # noqa: BLE001
        return []


@tool("research")
def show_verify(ctx: ToolContext, study: str = "") -> dict:
    """Recompute the hash chains: each study's record (or one, by folder) and the control store's events; and check
    each agent's plan.jsonl line by line against the shas its plan events hold (`plans`)."""
    from ..agents.plan import verify as verify_plans
    from ..study.record import Record
    from ..study.workspace import studies
    out = []
    for folder in studies(ctx.root):
        name = Path(folder).name
        if study and Path(study).name != name:
            continue
        v = Record(folder).verify()
        out.append({"study": name, "ok": v["ok"], "nodes": v["nodes"], "broken_at": v["broken_at"], "head": v["head"]})
    store = _store(ctx)
    return {"records": out, "control": store.verify(), "plans": verify_plans(store)}


def changed(root: Path) -> Optional[tuple]:
    """A cheap fingerprint of the control store (the last event and the inbox), for the server's watcher. None when
    it cannot be read now: no store (yet, or the workspace was deleted), or a writer holds it locked or busy past a
    short wait. The watcher then keeps its last fingerprint and looks again next tick, so nothing is lost."""
    import sqlite3
    from contextlib import closing
    db = Path(root) / ".openlab" / "control.db"
    try:
        if not db.is_file():
            return None
        with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=0.25)) as con:
            seq = con.execute("SELECT max(seq) FROM events").fetchone()[0]
            inbox = con.execute("SELECT count(*), max(id), sum(status='pending') FROM inbox").fetchone()
        return (seq, *inbox)
    except (sqlite3.Error, OSError):
        return None


# --- the paper workspace: each study's white paper and code folder (openlab.study.release) ------------------------
def _studies(root: Path) -> list:
    from ..study.workspace import studies
    return [Path(f) for f in studies(root)]


def _jailed(ctx: ToolContext, f: Path) -> Optional[Path]:
    """`f` when it is a file inside the workspace and outside the vault (as ToolContext.path holds), else None: a
    section or README that is a symlink to /etc/passwd or the vault is not read."""
    from .registry import ToolError
    try:
        p = ctx.path(str(Path(f).relative_to(ctx.root)))
    except (ToolError, ValueError):
        return None
    return p if p.is_file() else None


def _read(ctx: ToolContext, f: Path, n: int = 20000) -> str:
    p = _jailed(ctx, f)
    return p.read_text(encoding="utf-8", errors="replace")[:n] if p is not None else ""


@tool("research")
def show_paper(ctx: ToolContext) -> dict:
    """Each study's paper workspace: its sections (the agents' prose), whether a paper.toml exists, the last release
    (paper built, refused or none; when; its files) and the README it wrote."""
    import json
    import time
    out = []
    for folder in _studies(ctx.root):
        rel = ctx.rel(folder) if folder != ctx.root else "."
        paper = folder / "paper"
        order = []
        if _jailed(ctx, paper / "paper.toml"):
            try:
                import tomllib
            except ImportError:                                             # python 3.10
                import tomli as tomllib                                     # type: ignore
            try:
                order = list(tomllib.loads(_read(ctx, paper / "paper.toml", 1 << 20)).get("sections") or [])
            except Exception:                                               # noqa: BLE001
                order = []
        secs = []
        one = _jailed(ctx, paper / "paper.tex")
        if one:
            secs.append({"name": "paper", "exists": True, "text": _read(ctx, one)})
        for name in [] if one else ["abstract", *map(str, order)]:
            f = paper / "sections" / f"{name}.tex"
            secs.append({"name": name, "exists": _jailed(ctx, f) is not None, "text": _read(ctx, f)})
        rel_dir = folder / "release"
        release = None
        manifest = _jailed(ctx, rel_dir / "MANIFEST.json")
        if manifest:
            try:
                m = json.loads(_read(ctx, manifest, 1 << 22))
            except ValueError:
                m = {}
            m = m if isinstance(m, dict) else {}
            release = {"paper": m.get("paper"), "head": m.get("record_head", ""), "files": sorted(m.get("files") or {}),
                       "at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(manifest.stat().st_mtime)),
                       "readme": _read(ctx, rel_dir / "README.md"),
                       "pdf": f"{rel}/release/paper.pdf" if _jailed(ctx, rel_dir / "paper.pdf") else ""}
        out.append({"study": rel, "has_paper": bool(one or _jailed(ctx, paper / "paper.toml")), "sections": secs,
                    "release": release})
    return {"studies": out}


@tool("research", mutates=True)
def show_compile(ctx: ToolContext, study: str) -> dict:
    """Compile a study's release: the white paper (when its bindings and lints pass) and the code folder, with the
    signed bundle and a manifest, into <study>/release/."""
    from .. import study as _study
    from ..study.release import build
    return build(_study.open(ctx.path(study)))


DRAFT_MAX = 1 << 20                     # bytes of a prose file the draft editor opens and saves
README = "README.md"                    # the study's own README: its prose opens the release README (release.py)


def _draft_folder(ctx: ToolContext, study: str) -> Path:
    from .registry import ToolError
    folder = ctx.path(study or ".")
    if not (folder / "study.toml").is_file():
        raise ToolError(f"{study!r} is not a study of this research", "", "show_paper lists the studies")
    return folder


def _draft_files(folder: Path, doc: str = "paper") -> list:
    """(path relative to the study, section name) of the files the draft editor opens: for the paper, the one
    document paper/paper.tex; or, in the older layout, paper.toml, the abstract and each section it names
    (openlab.study.paper.build._files), whether written yet or not. For the readme, the study's README.md."""
    from ..study.paper import build
    if doc == "readme":
        return [(README, "README")]
    if build.single(folder):
        return [(build.PAPER, "paper")]
    out = [(build.TOML, "paper.toml")]
    try:
        cfg = build.config((folder / build.TOML).read_text(encoding="utf-8"))
        out += [(rel, name or "abstract") for rel, name in build._files(cfg)]
    except Exception:                                                       # noqa: BLE001 -- no or a broken toml
        pass
    return out


def _sha(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _draft_path(ctx: ToolContext, folder: Path, file: str) -> Path:
    """The prose file `file` of the study, jailed: one the paper names (or paper.toml), never a symlink."""
    from .registry import ToolError
    files = {rel for rel, _ in _draft_files(folder)} | {README}
    if file not in files:
        raise ToolError(f"{file!r} is not a file of the paper or the README", "the editor opens the paper's LaTeX "
                        "(paper/paper.tex, or paper.toml and the sections it names) and the study's README.md",
                        "show_draft lists them")
    p = ctx.path(str((folder / file).relative_to(ctx.root)))
    if p.is_symlink() or (p.exists() and not p.is_file()):
        raise ToolError(f"{file} is not a regular file", "a link is never followed", "")
    return p


def _readme_parts(folder: Path) -> dict:
    """For the README editor: `record`, the record's part the last compile wrote (between release.RECORD_MARK and
    RECORD_END), `starter`, the text the editor opens when the study has no README.md yet (the last compile's
    README, or the name, the question and the marks), and the two `marks`."""
    from ..study import release
    mark, end = release.RECORD_MARK, release.RECORD_END
    p = folder / "release" / "README.md"
    text = p.read_text(encoding="utf-8", errors="replace") if p.is_file() and not p.is_symlink() else ""
    lines = text.rstrip().splitlines()
    if mark in lines:
        i = lines.index(mark)
        j = lines.index(end, i) if end in lines[i:] else len(lines)
        top, record = lines[:i], "\n".join(lines[i + 1:j]).strip()
    elif "## Data" in lines:                                                # a compile before the marks
        i = lines.index("## Data")
        top, record = lines[:i], "\n".join(lines[i:]).strip()
    else:
        try:
            import tomllib
        except ImportError:                                                 # python 3.10
            import tomli as tomllib                                         # type: ignore
        try:
            st = tomllib.loads((folder / "study.toml").read_text(encoding="utf-8")).get("study") or {}
        except Exception:                                                   # noqa: BLE001 -- a broken toml
            st = {}
        top, record = [f"# {st.get('name') or folder.name}", "", *str(st.get("question") or "").splitlines(), ""], ""
    starter = "\n".join([*top, mark, "", record or "(written from the record at each compile)", "", end]) + "\n"
    return {"record": record, "starter": starter, "marks": [mark, end]}


@tool("research")
def show_draft(ctx: ToolContext, study: str, doc: str = "paper") -> dict:
    """A study's draft for the editor, with each file's text and sha256 (the save checks it) and the comments on
    them: `doc` paper (paper/paper.tex, or paper.toml, the abstract and each section) or readme (the study's
    README.md, with the release README's `head` and record `tail` around it)."""
    from ..agents.comments import threads
    from .registry import ToolError
    if doc not in ("paper", "readme"):
        raise ToolError(f"doc is paper or readme, not {doc!r}")
    folder = _draft_folder(ctx, study)
    rel = ctx.rel(folder) if folder != ctx.root else "."
    files = []
    for f, name in _draft_files(folder, doc):
        p = _jailed(ctx, folder / f)
        text = p.read_text(encoding="utf-8", errors="replace") if p is not None and p.stat().st_size <= DRAFT_MAX else ""
        files.append({"file": f, "name": name, "exists": p is not None, "text": text, "sha": _sha(text) if p else "",
                      "too_big": p is not None and p.stat().st_size > DRAFT_MAX})
    names = {f["file"] for f in files}
    out = {"study": rel, "doc": doc, "files": files, "comments": [t for t in threads(_store(ctx), rel) if t["file"] in names]}
    if doc != "readme":
        return out
    parts = _readme_parts(folder)
    for f in files:
        if not f["exists"]:                         # opens as the README the last compile wrote; saved, it is README.md
            f["text"] = parts["starter"]
    return {**out, **parts}


@tool("research", mutates=True, human=True)
def show_draft_init(ctx: ToolContext, study: str) -> dict:
    """A person starts the study's paper from the draft editor: paper/paper.tex from openlab's template (the abstract
    and the usual headings), or the older paper.toml layout carried over into it. Told to the study's working
    agents."""
    from .. import study as _study
    from ..agents.comments import _tell
    from ..study.errors import StudyError
    from ..study.paper.build import init
    from .agent_tools import _person
    from .registry import ToolError
    folder = _draft_folder(ctx, study)
    try:
        r = init(_study.open(folder))
    except StudyError as e:
        raise ToolError(e.error, e.meaning, e.fix) from None
    rel = ctx.rel(folder) if folder != ctx.root else "."
    told = _tell(_store(ctx), rel, "the person started the paper as one document, paper/paper.tex: write the paper "
                 "there (study_paper draft=true renders it)" + ("; the section files are no longer read"
                                                                 if r["carried_over"] else "") + ".", _person())
    return {**r, "told": told}


@tool("research", mutates=True, human=True)
def show_draft_save(ctx: ToolContext, study: str, file: str, text: str, sha: str = "") -> dict:
    """A person saves a prose file of the paper from the draft editor. `sha`: the sha256 of the text the editor
    opened ('' for a file not written yet): refused when the file changed since, so an agent's edit is never lost.
    Told to the study's working agents."""
    from ..agents.comments import _tell
    from .agent_tools import _person
    from .registry import ToolError
    folder = _draft_folder(ctx, study)
    p = _draft_path(ctx, folder, file)
    if len(text.encode("utf-8")) > DRAFT_MAX:
        raise ToolError(f"{file} would be over {DRAFT_MAX} bytes", "", "split the section")
    old = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""
    if (_sha(old) if p.is_file() else "") != sha:
        raise ToolError("changed since you opened it", f"{file} was edited by someone else (an agent?) after you "
                        "opened it", "load theirs, or copy yours aside, load theirs and paste it back")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.save")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(p)
    rel = ctx.rel(folder) if folder != ctx.root else "."
    st, by = _store(ctx), _person()
    st.event(by, "draft_edit", summary=f"{by} edited {rel}/{file}: {len(old.splitlines())} -> {len(text.splitlines())} lines",
             payload={"study": rel, "file": file, "sha": _sha(text), "was": _sha(old)})
    what = f"{file}, the README's prose (it opens the release README)" if file == README else f"{file} of the paper"
    told = _tell(st, rel, f"the person edited {what} in openlab show: read it again before you edit it.", by)
    return {"file": file, "sha": _sha(text), "told": told}


@tool("research", human=True)
def show_draft_preview(ctx: ToolContext, study: str, files: Optional[dict] = None) -> dict:
    """The paper as it would build with the editor's unsaved text (`files`: {path: text}): the PDF (base64), the
    LaTeX errors on their prose lines, and every finding. Nothing is recorded, signed or written to paper/build."""
    import base64
    from .. import study as _study
    from ..study.errors import StudyError
    from .registry import ToolError
    folder = _draft_folder(ctx, study)
    names = {rel for rel, _ in _draft_files(folder)}
    over = {str(k): str(v) for k, v in (files or {}).items() if str(k) in names}
    try:
        from ..study.paper.build import preview
        r = preview(_study.open(folder), over)
    except StudyError as e:
        raise ToolError(e.error, e.meaning, e.fix) from None
    errors = [x for x in r["errors"] if not x["message"].startswith("==>")]
    return {**r, "errors": errors, "pdf": base64.b64encode(r["pdf"]).decode() if r["pdf"] else ""}


@tool("research", mutates=True, human=True)
def paper_comment(ctx: ToolContext, study: str, file: str, text: str, quote: str = "", start: int = 0,
                  end: int = 0) -> dict:
    """A person comments on the paper draft: on `quote` (characters start..end of `file`). Told to the study's
    working agents, who answer with agent_comment_reply."""
    from ..agents.comments import CommentError, comment
    from .agent_tools import _person
    from .registry import ToolError
    folder = _draft_folder(ctx, study)
    _draft_path(ctx, folder, file)
    try:
        return comment(_store(ctx), ctx.rel(folder) if folder != ctx.root else ".", file, quote, start, end, text,
                       _person())
    except CommentError as e:
        raise ToolError(e.error, "", e.try_) from None


@tool("research", mutates=True, human=True)
def paper_comment_reply(ctx: ToolContext, id: int, text: str = "", resolve: Optional[bool] = None) -> dict:
    """A person replies to comment #id, resolves it (resolve true) or reopens it (resolve false)."""
    from ..agents.comments import CommentError, reply
    from .agent_tools import _person
    from .registry import ToolError
    try:
        return reply(_store(ctx), id, text, _person(), resolve, person=True)
    except CommentError as e:
        raise ToolError(e.error, "", e.try_) from None


def summary(folder: Path) -> dict:
    """What a research's tab shows, read cheaply: live agents, pending inbox items, spend."""
    import json
    import sqlite3
    from contextlib import closing
    db = Path(folder) / ".openlab" / "control.db"
    if not db.is_file():
        return {"live": 0, "pending": 0, "usd": 0.0, "last": "", "lead": None}
    try:
        with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=1)) as con:
            live = con.execute("SELECT count(*) FROM agents WHERE status IN ('running','idle','waiting','paused')").fetchone()[0]
            pending = con.execute("SELECT count(*) FROM inbox WHERE status='pending'").fetchone()[0]
            usd = sum((json.loads(s or "{}").get("usd") or 0) for (s,) in con.execute(
                "SELECT spent FROM agents WHERE kind != 'claude-subagent'"))
            lead = con.execute("SELECT id, status, question, updated FROM agents WHERE role = 'lead' "
                               "ORDER BY created DESC LIMIT 1").fetchone()
            last = con.execute("SELECT max(updated) FROM agents").fetchone()[0] or ""
        return {"live": live, "pending": pending, "usd": round(usd, 4), "last": last,
                "lead": {"id": lead[0], "status": lead[1], "question": lead[2][:300], "at": lead[3]} if lead else None}
    except sqlite3.Error:
        return {"live": 0, "pending": 0, "usd": 0.0, "last": "", "lead": None}


def _release_stale(study: Path, ended: str) -> bool:
    """Whether a lead that ended at `ended` (ISO time) while the server was down still needs its study's release:
    the release's MANIFEST.json names a record head that is not the record's head now, and the lead ended after the
    manifest was written. A study with no release yet is history, never compiled on a restart (`show_compile`, or
    the Workspace's "Compile now", compiles it)."""
    import json
    from datetime import datetime
    man = Path(study) / "release" / "MANIFEST.json"
    try:
        head = json.loads(man.read_text(encoding="utf-8")).get("record_head", "")
        at = man.stat().st_mtime
        from ..study.record import Record
        if not (Path(study) / ".openlab" / "record.db").is_file():
            return False
        now_head = Record(Path(study)).head()
        return bool(now_head) and now_head != head and datetime.fromisoformat(ended).timestamp() > at
    except Exception:                                                       # noqa: BLE001
        return False


class ReleaseJob:
    """Compile one study's release for the leads that ended done: one build however many leads finished together.
    Every failure is logged as a `release` event of each lead; calling it never raises."""

    def __init__(self, folder: Path, study: Path, rel: str, agents: list):
        self.folder, self.study, self.rel, self.agents = Path(folder), Path(study), rel, list(agents)
        self.only_if_stale = False          # set by the server for a build folded in behind one that was running

    def current(self) -> bool:
        """The release already holds the record's head (so a second build would write the same thing)."""
        import json
        try:
            from ..study.record import Record
            man = json.loads((self.study / "release" / "MANIFEST.json").read_text(encoding="utf-8"))
            return man.get("record_head", "") == Record(self.study).head()
        except Exception:                                                   # noqa: BLE001
            return False

    def log(self, summary: str) -> None:
        from ..agents.store import Store
        try:
            st = Store(self.folder)
            for a in self.agents:
                st.event(a, "release", summary=summary)
        except Exception:                                                   # noqa: BLE001
            pass                                                            # the workspace is gone: nobody to tell

    def __call__(self) -> None:
        from .. import study as _study
        from ..study.release import build
        try:
            if self.only_if_stale and self.current():
                self.log(f"release already current in {self.rel}/release")
                return
            r = build(_study.open(self.study))
            self.log(f"release compiled: paper {r['paper']}" + (f" ({r['why'][:160]})" if r["why"] else "")
                     + f"; {len(r['files'])} files in {self.rel}/release")
        except Exception as e:                                              # noqa: BLE001
            self.log(f"release not compiled: {type(e).__name__}: {e}")


def finished_leads(folder: Path, seen: set, prime: bool = False) -> list:
    """The release compiles a study's leads finished since the last look: one job per study (two leads of a study
    that ended done together compile it once). [show] compile_on_done = false in openlab.toml turns it off (the leads
    are still noted, so turning it back on does not compile history).

    `seen` lives in the server's memory, so `prime` (the server's first look) notes who has already finished and
    compiles only what was missed while the server was down: a lead whose study's release names an older record
    head than the record's own and that ended after that release was written (`_release_stale`)."""
    from ..agents.store import Store
    if not (Path(folder) / ".openlab" / "control.db").is_file():             # deleted, or not a workspace yet
        return []
    off = (_toml(folder).get("show") or {}).get("compile_on_done") is False
    try:
        st = Store(folder)
        leads = [a for a in st.agents() if a["status"] == "done" and a["study"] and (a["role"] == "lead" or not a["parent"])]
    except Exception:                                                       # noqa: BLE001
        return []
    jobs: dict = {}
    for a in leads:
        if a["id"] in seen:
            continue
        seen.add(a["id"])
        study = (Path(folder) / a["study"]).resolve()
        if off or (prime and not _release_stale(study, a.get("updated") or "")):
            continue
        key = str(study)
        if key in jobs:
            jobs[key].agents.append(a["id"])
        else:
            jobs[key] = ReleaseJob(folder, study, a["study"], [a["id"]])
    return list(jobs.values())


# ---------- the Workspace: the research's files as a tree, each one opened, edited (prose) and commented on ----------

TREE_SKIP = {".openlab", ".git", "__pycache__", ".release.tmp", "node_modules", ".ipynb_checkpoints", ".pytest_cache",
             ".mypy_cache", ".ruff_cache"}
TREE_MAX = 500                          # entries of one folder listed
KINDS = {".tex": "tex", ".md": "md", ".pdf": "pdf", ".png": "image", ".jpg": "image", ".jpeg": "image",
         ".gif": "image", ".svg": "image", ".webp": "image"}
TEXT = {".py", ".toml", ".json", ".jsonl", ".csv", ".tsv", ".txt", ".yaml", ".yml", ".sh", ".log", ".cfg", ".ini",
        ".r", ".jl", ".sql", ".bib", ".sty", ".cls", ".html", ".js", ".css", ".xml", ".ipynb", ".env.example", ""}


def _kind(name: str) -> str:
    ext = Path(name).suffix.lower()
    return KINDS.get(ext) or ("text" if ext in TEXT else "other")


def _wpath(ctx: ToolContext, path: str) -> Path:
    """`path` of the research, jailed; never in .openlab."""
    from .registry import ToolError
    p = ctx.path(path or ".")
    if ".openlab" in p.relative_to(ctx.root).parts:
        raise ToolError("the .openlab folder is openlab's own", "read the record through the tools", "")
    return p


def _wrel(ctx: ToolContext, p: Path) -> str:
    r = p.relative_to(ctx.root).as_posix()
    return r if r not in ("", ".") else "."


def _study_of(ctx: ToolContext, p: Path) -> Optional[Path]:
    """The study folder (one with a study.toml) `p` is in, or None."""
    d = p if p.is_dir() else p.parent
    while True:
        if (d / "study.toml").is_file():
            return d
        if d == ctx.root or ctx.root not in d.parents:
            return None
        d = d.parent


def _open_counts(ctx: ToolContext) -> dict:
    from ..agents.comments import threads, where
    out: dict = {}
    for t in threads(_store(ctx)):
        if not t["resolved"]:
            out[where(t)] = out.get(where(t), 0) + 1
    return out


def _editable(ctx: ToolContext, p: Path, kind: str) -> bool:
    """Prose a person edits here: .tex and .md, not a link, not in a study's release/ (rewritten at each compile)."""
    if kind not in ("tex", "md") or p.is_symlink():
        return False
    s = _study_of(ctx, p)
    return not (s is not None and "release" in p.relative_to(s).parts[:1])


@tool("research")
def show_tree(ctx: ToolContext, path: str = "") -> dict:
    """One folder of the research for the Workspace's tree: its folders then files ({name, path, dir, size, kind,
    study, link, comments}), hidden and openlab's own folders left out; a study folder with no README.md lists it
    `virtual` (it opens as the README the last compile wrote); `open`: every file of the research with an open
    comment."""
    import os
    from .registry import ToolError
    d = _wpath(ctx, path)
    if not d.is_dir():
        raise ToolError(f"{path!r} is not a folder of the research")
    counts, rows = _open_counts(ctx), []
    with os.scandir(d) as it:
        items = sorted(it, key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.casefold()))
    for e in items:
        if e.name in TREE_SKIP or e.name.startswith("."):
            continue
        isdir = e.is_dir(follow_symlinks=False)
        rel = _wrel(ctx, d / e.name)
        try:
            size = 0 if isdir else e.stat(follow_symlinks=False).st_size
        except OSError:
            size = 0
        rows.append({"name": e.name, "path": rel, "dir": isdir, "size": size, "kind": "dir" if isdir else _kind(e.name),
                     "study": isdir and (d / e.name / "study.toml").is_file(), "link": e.is_symlink(),
                     "comments": sum(n for k, n in counts.items() if k == rel or k.startswith(rel + "/"))})
    if (d / "study.toml").is_file() and not (d / README).exists():
        rel = _wrel(ctx, d / README)
        rows.append({"name": README, "path": rel, "dir": False, "size": 0, "kind": "md", "study": False, "link": False,
                     "virtual": True, "comments": counts.get(rel, 0)})
    s = _study_of(ctx, d)
    return {"path": _wrel(ctx, d), "entries": rows[:TREE_MAX], "more": max(0, len(rows) - TREE_MAX),
            "study": _wrel(ctx, s) if s is not None else None, "open": sorted(counts)}


@tool("research")
def show_open(ctx: ToolContext, path: str) -> dict:
    """One file of the research for the Workspace: its kind (tex, md, pdf, image, text, other), size, text and sha256
    when it is text (the save checks it), whether a person edits it here, its study and its path in that study, whether
    it is the study's paper (it renders through show_draft_preview) or README (with the record's part: show_draft),
    and the comments on it."""
    from ..agents.comments import threads, where
    from .registry import ToolError
    p = _wpath(ctx, path)
    rel, kind = _wrel(ctx, p), _kind(p.name)
    s = _study_of(ctx, p)
    srel = _wrel(ctx, s) if s is not None else "."
    file = p.relative_to(s).as_posix() if s is not None else rel
    readme = s is not None and file == README
    if p.is_dir():
        raise ToolError(f"{path} is a folder", "show_tree lists it", "")
    if not p.exists() and not readme:
        raise ToolError(f"no file {path}", "", "show_tree lists the files")
    size = p.stat().st_size if p.is_file() else 0
    out = {"path": rel, "kind": kind, "size": size, "exists": p.is_file(), "text": None, "sha": "",
           "too_big": size > DRAFT_MAX, "editable": _editable(ctx, p, kind), "study": srel if s is not None else None,
           "file": file, "paper": False, "readme": readme,
           "comments": [t for t in threads(_store(ctx)) if where(t) == rel]}
    if kind in ("tex", "md", "text") and p.is_file() and not out["too_big"] and not p.is_symlink():
        data = p.read_bytes()
        if b"\0" in data[:8192]:
            out.update(kind="other", editable=False)
        else:
            text = data.decode("utf-8", errors="replace")
            out.update(text=text, sha=_sha(text))
    if s is not None and kind == "tex":
        out["paper"] = file in {f for f, _ in _draft_files(s)}
    if readme:
        parts = _readme_parts(s)
        out.update(parts)
        if not p.exists():
            out.update(text=parts["starter"], sha="", editable=True)
    return out


@tool("research", mutates=True, human=True)
def show_save(ctx: ToolContext, path: str, text: str, sha: str = "") -> dict:
    """A person saves a .tex or .md file of the research from the Workspace. `sha`: the sha256 of the text they
    opened ('' for a file not written yet): refused when the file changed since, so an agent's edit is never lost.
    Told to the live lead of its study (else every live lead)."""
    from ..agents.comments import _tell
    from .agent_tools import _person
    from .registry import ToolError
    p = _wpath(ctx, path)
    if p.is_dir() or not _editable(ctx, p, _kind(p.name)):
        raise ToolError(f"{path} is not edited here", "the Workspace edits .tex and .md files (not a study's "
                        "release/, rewritten at each compile)", "")
    if len(text.encode("utf-8")) > DRAFT_MAX:
        raise ToolError(f"{path} would be over {DRAFT_MAX} bytes", "", "split it")
    old = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""
    if (_sha(old) if p.is_file() else "") != sha:
        raise ToolError("changed since you opened it", f"{path} was edited by someone else (an agent?) after you "
                        "opened it", "load theirs, or copy yours aside, load theirs and paste it back")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.save")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(p)
    rel, s = _wrel(ctx, p), _study_of(ctx, p)
    st, by = _store(ctx), _person()
    st.event(by, "draft_edit", summary=f"{by} edited {rel}: {len(old.splitlines())} -> {len(text.splitlines())} lines",
             payload={"path": rel, "sha": _sha(text), "was": _sha(old)})
    told = _tell(st, _wrel(ctx, s) if s is not None else ".", f"the person edited {rel} in openlab show: read it again "
                 "before you edit it.", by)
    return {"path": rel, "sha": _sha(text), "told": told}


@tool("research", mutates=True, human=True)
def show_comment(ctx: ToolContext, path: str, text: str, quote: str = "", start: int = 0, end: int = 0) -> dict:
    """A person comments on a file of the research: on `quote` (characters start..end of its text) or, with no
    quote, on the whole file (a PDF, an image, data). Told to the live lead of its study (else every live lead) with
    the path, the quote and the comment; they answer with agent_comment_reply."""
    from ..agents.comments import CommentError, comment
    from .agent_tools import _person
    from .registry import ToolError
    p = _wpath(ctx, path)
    s = _study_of(ctx, p)
    file = p.relative_to(s).as_posix() if s is not None else _wrel(ctx, p)
    if p.is_dir() or not (p.exists() or (s is not None and file == README)):
        raise ToolError(f"no file {path}", "a comment is on a file of the research", "show_tree lists them")
    try:
        return comment(_store(ctx), _wrel(ctx, s) if s is not None else ".", file, quote if quote else "",
                       start, end, text, _person())
    except CommentError as e:
        raise ToolError(e.error, "", e.try_) from None
