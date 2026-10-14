"""The workspace control store: `<workspace>/.openlab/control.db`, SQLite in WAL mode, one writer at a time.

* `agents`: every agent openlab knows of -- spawned by openlab, a Claude Code session or sub-agent registered by a
  hook, a Codex run, a person's terminal -- with its place in the tree, its budget, what it spent and its status.
  `control` is what a person asked for (pause, stop); `status` is what the agent is doing.
* `events`: append-only and hash-chained like a study record: each row's hash covers its seq, agent, time, kind,
  tool, summary, payload and the previous row's hash, so editing a row breaks the chain from there on (`verify`).
* `inbox`: the requests that need a person (a holdout look, a protected amendment, an approval, more budget).
* `pending_calls`: a Claude Code hook saw an `mcp__openlab__*` call (or a sub-agent's Agent/Task call) and noted
  who made it; the MCP server claims the note to attribute the call (openlab.agents.identity.for_mcp_call).

Payloads are kept small (long strings and lists are cut); full transcripts live in `.openlab/agents/<id>/`.
This module imports nothing from openlab.study, so a hook stays fast.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

MARKER = "openlab.toml"
KINDS = ("spawned", "claude-session", "claude-subagent", "codex", "terminal")
STATUSES = ("running", "idle", "waiting", "paused", "done", "failed", "stopped", "lost", "ended")
LIVE = ("running", "idle", "waiting", "paused")
FINAL = ("done", "failed", "stopped", "lost", "ended")   # ended: a hook session gone quiet (reap)
INBOX_KINDS = ("holdout", "amend", "approval", "budget", "screen", "screen_conflict", "screen_maybe", "fresh",
               "continues", "compute", "question", "other")

#: [agents] defaults in openlab.toml
DEFAULTS = {"stuck_minutes": 5, "session_timeout_hours": 12, "max_depth": 3, "max_children": 6, "loop_refusals": 3,
            "over_fraction": 0.9,
            "budget_usd": 2.0, "budget_minutes": 60, "budget_turns": 200,
            "cap_usd": 20.0, "cap_minutes": 480,         # the most one proposal (its lead and all branches) may ask
            "permission_mode": "acceptEdits", "allow_native_subagents": False,
            "allowed_tools": ["mcp__openlab", "Read", "Edit", "Write", "Glob", "Grep", "Bash", "Agent", "Task",
                              "TodoWrite", "WebSearch", "WebFetch"]}

SUMMARY_MAX = 240           # characters of an event's one-line summary
STRING_MAX = 400            # characters of one string inside a payload
LIST_MAX = 20               # items of one list inside a payload
PAYLOAD_MAX = 4000          # characters of a payload's JSON
TELL_MAX = 3000             # characters of a direction (`openlab agent tell`), kept whole
CLAIM_WINDOW = 30.0         # seconds a pending MCP call waits to be claimed

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agents(id TEXT PRIMARY KEY, name TEXT NOT NULL DEFAULT '', role TEXT NOT NULL DEFAULT '',
    parent TEXT NOT NULL DEFAULT '', cli TEXT NOT NULL DEFAULT '', kind TEXT NOT NULL, study TEXT NOT NULL DEFAULT '',
    question TEXT NOT NULL DEFAULT '', cwd TEXT NOT NULL DEFAULT '', pid INTEGER, pgid INTEGER, supervisor INTEGER,
    session_id TEXT NOT NULL DEFAULT '', ext TEXT NOT NULL DEFAULT '', tool_use_id TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '', status TEXT NOT NULL, control TEXT NOT NULL DEFAULT '',
    control_reason TEXT NOT NULL DEFAULT '', exit_code INTEGER, budget TEXT NOT NULL DEFAULT '{}',
    spent TEXT NOT NULL DEFAULT '{}', created TEXT NOT NULL, updated TEXT NOT NULL,
    last_event_at TEXT NOT NULL DEFAULT '', token_sha TEXT NOT NULL DEFAULT '', decide TEXT NOT NULL DEFAULT '');
CREATE INDEX IF NOT EXISTS agents_session ON agents(session_id, ext);
CREATE INDEX IF NOT EXISTS agents_parent ON agents(parent);
CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY, agent TEXT NOT NULL, at TEXT NOT NULL, kind TEXT NOT NULL,
    tool TEXT NOT NULL DEFAULT '', summary TEXT NOT NULL DEFAULT '', payload TEXT NOT NULL DEFAULT '{}',
    prev TEXT NOT NULL, hash TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS events_agent ON events(agent, seq);
CREATE TABLE IF NOT EXISTS inbox(id INTEGER PRIMARY KEY AUTOINCREMENT, agent TEXT NOT NULL, study TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL, args TEXT NOT NULL DEFAULT '{}', reason TEXT NOT NULL, cost_note TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending', created TEXT NOT NULL, decided_by TEXT NOT NULL DEFAULT '',
    decided_at TEXT NOT NULL DEFAULT '', decision_reason TEXT NOT NULL DEFAULT '', result TEXT NOT NULL DEFAULT '{}');
CREATE TABLE IF NOT EXISTS pending_calls(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL DEFAULT '',
    agent TEXT NOT NULL, tool TEXT NOT NULL, sha TEXT NOT NULL, input TEXT NOT NULL DEFAULT '{}', at REAL NOT NULL,
    claimed_by TEXT NOT NULL DEFAULT '');
CREATE INDEX IF NOT EXISTS pending_tool ON pending_calls(tool, at);
"""

_JSON_COLS = ("budget", "spent")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def local_time(at: str, date: bool = False) -> str:
    """A stored UTC ISO time as the local wall time a person reads: HH:MM:SS (with the date when `date`, or when it
    is not today). Stored times stay UTC."""
    if not at:
        return "-"
    try:
        t = datetime.fromisoformat(at)
    except ValueError:
        return at
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    t = t.astimezone()
    today = t.date() == datetime.now().astimezone().date()
    return t.strftime("%Y-%m-%d %H:%M:%S" if date or not today else "%H:%M:%S")


def age(at: str) -> float:
    """Seconds since an ISO time (inf when empty)."""
    if not at:
        return float("inf")
    try:
        return max(0.0, (datetime.now(timezone.utc) - datetime.fromisoformat(at)).total_seconds())
    except ValueError:
        return float("inf")


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def token_sha(token: str) -> str:
    return hashlib.sha256(("openlab-agent-token|" + token).encode()).hexdigest()


def input_sha(obj) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def root_of(start: Optional[str | Path] = None) -> Path:
    """The workspace root, as openlab.study.workspace.root finds it (kept here so a hook need not import
    openlab.study): the nearest ancestor holding openlab.toml, else a study's parent folder, else the folder."""
    here = Path(start or os.environ.get("OPENLAB_WORKSPACE") or os.getcwd()).resolve()
    for d in (here, *here.parents):
        if (d / MARKER).is_file():
            return d
    return here.parent if (here / "study.toml").is_file() else here


def config(root: Path) -> dict:
    """[agents] of openlab.toml over DEFAULTS."""
    out = dict(DEFAULTS)
    p = Path(root) / MARKER
    if p.is_file():
        try:
            try:
                import tomllib
            except ImportError:                                     # Python 3.10
                import tomli as tomllib
            given = tomllib.loads(p.read_text(encoding="utf-8")).get("agents") or {}
        except Exception:                                           # noqa: BLE001 -- a broken file keeps the defaults
            given = {}
        for k, v in (given.items() if isinstance(given, dict) else ()):
            d = DEFAULTS.get(k)                                     # a value of the wrong type keeps the default
            if k not in DEFAULTS or d is None:
                out[k] = v
            elif isinstance(d, bool):
                out[k] = v if isinstance(v, bool) else d
            elif isinstance(d, (int, float)):
                out[k] = v if isinstance(v, (int, float)) and not isinstance(v, bool) else d
            elif isinstance(d, (list, tuple)):
                out[k] = list(v) if isinstance(v, (list, tuple)) else d
            else:
                out[k] = v if isinstance(v, type(d)) else d
    return out


def short(text, n: int = SUMMARY_MAX) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n - 1] + "…"


def bound(obj, depth: int = 0):
    """`obj` cut to size: long strings, long lists and deep nesting are shortened and say so."""
    if isinstance(obj, str):
        return obj if len(obj) <= STRING_MAX else obj[:STRING_MAX] + f"… [{len(obj) - STRING_MAX} more chars]"
    if depth > 4:
        return "…"
    if isinstance(obj, dict):
        items = list(obj.items())
        out = {str(k): bound(v, depth + 1) for k, v in items[:LIST_MAX * 2]}
        if len(items) > LIST_MAX * 2:
            out["…"] = f"{len(items) - LIST_MAX * 2} more keys"
        return out
    if isinstance(obj, (list, tuple)):
        out = [bound(v, depth + 1) for v in list(obj)[:LIST_MAX]]
        if len(obj) > LIST_MAX:
            out.append(f"… {len(obj) - LIST_MAX} more")
        return out
    if obj is None or isinstance(obj, (int, float, bool)):
        return obj
    return bound(str(obj), depth)


def _payload(obj) -> str:
    text = canonical(bound(obj or {}))
    return text if len(text) <= PAYLOAD_MAX else canonical({"truncated": text[:PAYLOAD_MAX]})


def _hash(seq: int, agent: str, at: str, kind: str, tool: str, summary: str, payload: str, prev: str) -> str:
    return hashlib.sha256(f"{seq}|{agent}|{at}|{kind}|{tool}|{summary}|{payload}|{prev}".encode()).hexdigest()


def alive(pid) -> bool:
    """A process with this pid exists and is not a zombie."""
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    try:
        with open(f"/proc/{int(pid)}/stat", "rb") as f:
            return f.read().rsplit(b")", 1)[1].split()[0] != b"Z"
    except OSError:
        return True


class _Kept(sqlite3.Connection):
    """A connection that `closing` leaves open until Store.reading ends."""
    kept = True
    thread = 0

    def close(self) -> None:
        if not self.kept:
            super().close()


class Store:
    def __init__(self, root: Optional[str | Path] = None):
        self.root = root_of(root) if root is None else Path(root).resolve()
        self.dir = self.root / ".openlab"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "control.db"
        with closing(self._con()) as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript(_SCHEMA)
            cols = {r[1] for r in con.execute("PRAGMA table_info(agents)")}
            if "token_sha" not in cols:                     # a store made before agent tokens
                con.execute("ALTER TABLE agents ADD COLUMN token_sha TEXT NOT NULL DEFAULT ''")
            if "decide" not in cols:                        # before an agent could be told to decide for itself
                con.execute("ALTER TABLE agents ADD COLUMN decide TEXT NOT NULL DEFAULT ''")

    def _con(self) -> sqlite3.Connection:
        shared = getattr(self, "_shared", None)
        if shared is not None and shared.thread == threading.get_ident():
            return shared
        con = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        con.row_factory = sqlite3.Row
        return con

    @contextmanager
    def reading(self):
        """One connection for every read (and write) in the block, on this thread: a view that asks the store a
        hundred small questions opens it once."""
        if getattr(self, "_shared", None) is not None:
            yield self
            return
        con = sqlite3.connect(self.path, timeout=30, isolation_level=None, factory=_Kept)
        con.row_factory = sqlite3.Row
        con.thread = threading.get_ident()
        self._shared = con
        try:
            yield self
        finally:
            self._shared = None
            con.kept = False
            con.close()

    def head(self) -> str:
        """The hash of the last event (the control store's head), without recomputing the chain."""
        with closing(self._con()) as con:
            row = con.execute("SELECT hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        return row["hash"] if row else ""

    @contextmanager
    def _write(self):
        con = self._con()
        try:
            con.execute("BEGIN IMMEDIATE")
            try:
                yield con
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise
        finally:
            con.close()

    def agent_dir(self, agent: str) -> Path:
        d = self.dir / "agents" / agent.replace("/", "_")
        d.mkdir(parents=True, exist_ok=True)
        return d

    # --- agents ----------------------------------------------------------------------------------------------
    @staticmethod
    def _agent(row) -> dict:
        d = dict(row)
        for k in _JSON_COLS:
            d[k] = json.loads(d[k] or "{}")
        return d

    def register(self, id: str, kind: str, **fields) -> dict:
        """Insert the agent, or update the fields given when it exists. Returns the row."""
        if kind not in KINDS:
            raise ValueError(f"kind is one of {', '.join(KINDS)}")
        for k in _JSON_COLS:
            if k in fields and not isinstance(fields[k], str):
                fields[k] = canonical(fields[k])
        t = now()
        with self._write() as con:
            row = con.execute("SELECT id FROM agents WHERE id=?", (id,)).fetchone()
            if row is None:
                cols = {"id": id, "kind": kind, "status": fields.pop("status", "running"), "created": t,
                        "updated": t, "name": fields.pop("name", id), **fields}
                con.execute(f"INSERT INTO agents({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                            list(cols.values()))
            elif fields:
                sets = ", ".join(f"{k}=?" for k in fields)
                con.execute(f"UPDATE agents SET {sets}, updated=? WHERE id=?", [*fields.values(), t, id])
        return self.get(id)

    def update(self, id: str, **fields) -> None:
        for k in _JSON_COLS:
            if k in fields and not isinstance(fields[k], str):
                fields[k] = canonical(fields[k])
        if not fields:
            return
        with self._write() as con:
            sets = ", ".join(f"{k}=?" for k in fields)
            con.execute(f"UPDATE agents SET {sets}, updated=? WHERE id=?", [*fields.values(), now(), id])

    def set_status(self, id: str, status: str, only_from: Iterable[str] = ()) -> bool:
        """Set the status; with `only_from`, only when the current status is one of them (atomic)."""
        if status not in STATUSES:
            raise ValueError(f"status is one of {', '.join(STATUSES)}")
        with self._write() as con:
            if only_from:
                q = ",".join("?" * len(tuple(only_from)))
                cur = con.execute(f"UPDATE agents SET status=?, updated=? WHERE id=? AND status IN ({q})",
                                  (status, now(), id, *only_from))
            else:
                cur = con.execute("UPDATE agents SET status=?, updated=? WHERE id=?", (status, now(), id))
            return cur.rowcount > 0

    def get(self, id: str) -> Optional[dict]:
        with closing(self._con()) as con:
            row = con.execute("SELECT * FROM agents WHERE id=?", (id,)).fetchone()
        return self._agent(row) if row else None

    def find(self, **where) -> list[dict]:
        cond = " AND ".join(f"{k}=?" for k in where)
        with closing(self._con()) as con:
            rows = con.execute("SELECT * FROM agents" + (f" WHERE {cond}" if cond else "") + " ORDER BY created",
                               list(where.values())).fetchall()
        return [self._agent(r) for r in rows]

    def agents(self, live_only: bool = False) -> list[dict]:
        rows = self.find()
        return [a for a in rows if a["status"] in LIVE] if live_only else rows

    def by_session(self, session_id: str, ext: str = "") -> Optional[dict]:
        found = self.find(session_id=session_id, ext=ext) if session_id else []
        return found[0] if found else None

    def children(self, id: str) -> list[dict]:
        return self.find(parent=id)

    def ancestors(self, id: str) -> list[dict]:
        """The agent's parent, its parent, ... (nearest first)."""
        out, seen = [], {id}
        a = self.get(id)
        while a and a["parent"] and a["parent"] not in seen:
            seen.add(a["parent"])
            a = self.get(a["parent"])
            if a:
                out.append(a)
        return out

    def depth(self, id: str) -> int:
        return len(self.ancestors(id))

    def blocked(self, id: str) -> Optional[tuple[str, str, str]]:
        """(agent id, what, reason) when the agent or an ancestor is paused or stopped by a person, else None."""
        a = self.get(id)
        for x in ([a] if a else []) + self.ancestors(id):
            if x["control"] in ("pause", "stop"):
                return x["id"], "paused" if x["control"] == "pause" else "stopped", x["control_reason"]
        return None

    def control(self, id: str, action: str, reason: str, by: str, sig: Optional[dict] = None) -> dict:
        """A person's pause, resume or stop. A spawned agent's supervisor acts on it (SIGSTOP/SIGCONT/kill);
        a hook-registered agent is held by the PreToolUse hook. Recorded as an event."""
        if action not in ("pause", "resume", "stop"):
            raise ValueError("action is pause, resume or stop")
        a = self.get(id)
        if a is None:
            raise KeyError(id)
        supervised = a["kind"] in ("spawned", "codex") and alive(a["supervisor"])
        if action == "resume":
            self.update(id, control="", control_reason="")
            if not supervised and a["status"] == "paused":
                self.set_status(id, "idle")
        else:
            self.update(id, control=action, control_reason=reason)
            if not supervised and a["status"] in LIVE:
                self.set_status(id, "paused" if action == "pause" else "stopped")
        self.event(id, f"control_{action}", summary=f"{action} by {by}: {reason}",
                   payload={"by": by, "reason": reason, **({"sig": sig} if sig else {})})
        return self.get(id)

    def add_spent(self, id: str, **delta) -> dict:
        with self._write() as con:
            row = con.execute("SELECT spent FROM agents WHERE id=?", (id,)).fetchone()
            if row is None:
                return {}
            spent = json.loads(row["spent"] or "{}")
            for k, v in delta.items():
                spent[k] = round(spent.get(k, 0) + v, 6) if isinstance(v, float) else spent.get(k, 0) + v
            con.execute("UPDATE agents SET spent=?, updated=? WHERE id=?", (canonical(spent), now(), id))
        return spent

    def set_spent(self, id: str, **values) -> dict:
        with self._write() as con:
            row = con.execute("SELECT spent FROM agents WHERE id=?", (id,)).fetchone()
            if row is None:
                return {}
            spent = {**json.loads(row["spent"] or "{}"), **values}
            con.execute("UPDATE agents SET spent=?, updated=? WHERE id=?", (canonical(spent), now(), id))
        return spent

    def reap(self) -> list[str]:
        """Mark `lost` the live agents whose process is gone without a final status: a spawned agent whose
        supervisor and CLI are both dead, a session whose claude process is dead. Returns their ids.

        A hook-registered agent (a Claude Code session or sub-agent) gets no final event when Claude Code is killed,
        so its silence decides: `idle` after [agents] stuck_minutes without an event, `ended` after
        [agents] session_timeout_hours (a sub-agent also ends with its session). Its leases then free."""
        lost, cfg = [], config(self.root)
        idle_s, end_s = float(cfg["stuck_minutes"]) * 60, float(cfg["session_timeout_hours"]) * 3600
        rows = self.agents(live_only=True)
        status = {a["id"]: a["status"] for a in self.agents()}
        for a in rows:
            if a["kind"] in ("spawned", "codex"):
                gone = a["supervisor"] and not alive(a["supervisor"]) and not alive(a["pid"])
            elif a["kind"] == "claude-session":
                gone = a["pid"] and not alive(a["pid"])
            else:
                gone = False
            if gone and self.set_status(a["id"], "lost", only_from=LIVE):
                self.event(a["id"], "lost", summary="its process is gone and it never reported an end")
                lost.append(a["id"])
                status[a["id"]] = "lost"
        for a in sorted(rows, key=lambda r: r["kind"] == "claude-subagent"):     # sessions first: subs end with them
            if a["kind"] not in ("claude-session", "claude-subagent") or status.get(a["id"]) not in LIVE:
                continue
            quiet = age(a["last_event_at"] or a["created"])
            orphan = a["kind"] == "claude-subagent" and status.get(a["parent"]) in FINAL
            if (quiet > end_s or orphan) and self.set_status(a["id"], "ended", only_from=LIVE):
                why = f"its session ended ({status.get(a['parent'])})" if orphan else \
                    f"no event for {quiet / 3600:.1f} h ([agents] session_timeout_hours)"
                self.event(a["id"], "ended", summary=why)
                lost.append(a["id"])
                status[a["id"]] = "ended"
            elif quiet > idle_s and status.get(a["id"]) == "running":
                self.set_status(a["id"], "idle", only_from=("running",))      # no event: its silence is the clock
        return lost

    # --- events ----------------------------------------------------------------------------------------------
    def event(self, agent: str, kind: str, tool: str = "", summary: str = "", payload: Optional[dict] = None,
              exact: bool = False) -> int:
        """Append an event. `exact`: the payload is kept whole (a direction, whose text is read and signed as is),
        refused above PAYLOAD_MAX rather than cut."""
        summary = short(summary)
        if exact:
            text = canonical(payload or {})
            if len(text) > PAYLOAD_MAX:
                raise ValueError(f"the event's payload is {len(text)} characters, above {PAYLOAD_MAX}")
        else:
            text = _payload(payload)
        with self._write() as con:
            last = con.execute("SELECT seq, hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
            seq, prev = (last["seq"] + 1, last["hash"]) if last else (1, "")
            at = now()
            con.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?)",
                        (seq, agent, at, kind, tool, summary, text, prev,
                         _hash(seq, agent, at, kind, tool, summary, text, prev)))
            con.execute("UPDATE agents SET last_event_at=? WHERE id=?", (at, agent))
        return seq

    def tell(self, agent: str, text: str, by: str, sig: Optional[dict] = None, ask: bool = False,
             via: str = "") -> int:
        """A direction for a working agent from a person or a supervising agent: a `tell` event, delivered into
        the agent's context at its next tool call (the PostToolUse hook, or the reply of its next openlab tool) and
        marked `told` then. `sig` ({key, value}): the person's signature over the direction, so the agent can trust
        it is the person's. `ask`: a person's prompt to a lead, which the lead answers with agent_reply while it
        works (`via`: where the person typed it, e.g. "show")."""
        if len(text) > TELL_MAX:
            raise ValueError(f"a direction is at most {TELL_MAX} characters (this one is {len(text)}): say it shorter, "
                             "or point to a file")
        return self.event(agent, "tell", summary=f"{by}: {text}",
                          payload={"text": text, "by": by, **({"sig": sig} if sig else {}),
                                   **({"ask": True} if ask else {}), **({"via": via} if via else {})}, exact=True)

    def reply(self, agent: str, text: str, to: int = 0) -> int:
        """An agent's answer to a person's prompt (`to`: the prompt's seq; 0 when it does not say)."""
        if len(text) > TELL_MAX:
            raise ValueError(f"a reply is at most {TELL_MAX} characters (this one is {len(text)}): say it shorter")
        return self.event(agent, "reply", summary=text, payload={"text": text, "to": int(to or 0)}, exact=True)

    def conversation(self, agent: str, n: int = 200) -> list:
        """The person's prompts to `agent` and its replies, oldest first: {seq, at, from, text, to, read, btw}. A
        prompt is read once it has been delivered (a `told` event names it); a side question (btw) is answered at
        once by a fork of the agent's session (openlab.agents.btw)."""
        evs = self.events(agent, n=1000, kinds=("tell", "told", "reply", "btw"))
        done = {s for e in evs if e["kind"] == "told" for s in e["payload"].get("seqs", [])}
        out = []
        for e in evs:
            p = e["payload"]
            if e["kind"] == "tell" and p.get("ask"):
                out.append({"seq": e["seq"], "at": e["at"], "from": p.get("by") or "person", "text": p.get("text", ""),
                            "to": 0, "read": e["seq"] in done, "btw": False})
            elif e["kind"] == "btw":                    # a side question: answered at once by a fork of its session
                out.append({"seq": e["seq"], "at": e["at"], "from": p.get("by") or "person", "text": p.get("text", ""),
                            "to": 0, "read": True, "btw": True})
            elif e["kind"] == "reply":
                out.append({"seq": e["seq"], "at": e["at"], "from": agent, "text": p.get("text", ""),
                            "to": p.get("to") or 0, "read": True, "btw": bool(p.get("btw"))})
        return out[-n:]

    def is_lead(self, a: Optional[dict]) -> bool:
        """A lead: an agent at the top of its tree (no parent; a lead role is one too). Only a lead takes the
        person's prompts; the agents under it hear from it."""
        return bool(a) and a["kind"] != "claude-subagent" and (not a["parent"] or a["role"] == "lead")

    def untold(self, agent: str) -> list:
        """The `tell` events of `agent` not delivered yet, oldest first."""
        evs = self.events(agent, n=1000, kinds=("tell", "told"))
        done = {s for e in evs if e["kind"] == "told" for s in e["payload"].get("seqs", [])}
        return [e for e in evs if e["kind"] == "tell" and e["seq"] not in done]

    def events(self, agent: Optional[str] = None, n: int = 50, after: int = 0,
               kinds: Iterable[str] = ()) -> list[dict]:
        """The last `n` events (of `agent`, of `kinds`, after seq `after`), oldest first."""
        cond, args = ["seq>?"], [after]
        if agent:
            cond.append("agent=?")
            args.append(agent)
        kinds = tuple(kinds)
        if kinds:
            cond.append(f"kind IN ({','.join('?' * len(kinds))})")
            args.extend(kinds)
        with closing(self._con()) as con:
            rows = con.execute(f"SELECT * FROM events WHERE {' AND '.join(cond)} ORDER BY seq DESC LIMIT ?",
                               [*args, n]).fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in reversed(rows)]

    def verify(self) -> dict:
        prev, n, broken = "", 0, None
        with closing(self._con()) as con:
            for r in con.execute("SELECT * FROM events ORDER BY seq"):
                n += 1
                h = _hash(r["seq"], r["agent"], r["at"], r["kind"], r["tool"], r["summary"], r["payload"], r["prev"])
                if broken is None and (r["seq"] != n or r["prev"] != prev or r["hash"] != h):
                    broken = r["seq"]
                prev = r["hash"]
        return {"ok": broken is None, "events": n, "broken_at": broken, "head": prev}

    # --- inbox -----------------------------------------------------------------------------------------------
    def request(self, agent: str, kind: str, reason: str, study: str = "", args: Optional[dict] = None,
                cost_note: str = "") -> int:
        if kind not in INBOX_KINDS:
            raise ValueError(f"kind is one of {', '.join(INBOX_KINDS)}")
        if not reason.strip():
            raise ValueError("a request needs a reason")
        with self._write() as con:
            cur = con.execute("INSERT INTO inbox(agent, study, kind, args, reason, cost_note, created) "
                              "VALUES (?,?,?,?,?,?,?)", (agent, study, kind, canonical(args or {}), reason,
                                                         cost_note, now()))
            item = cur.lastrowid
        self.event(agent, "request", summary=f"#{item} {kind}: {reason}",
                   payload={"item": item, "kind": kind, "study": study, "args": args or {}, "cost": cost_note})
        self.notify("request", agent, f"inbox #{item} from {agent} ({kind}): {short(reason, 160)}")
        return item

    def notify(self, kind: str, agent: str, text: str) -> None:
        """Run the person's [agents] notify command (openlab.toml) with `text` as its last argument, detached:
        OPENLAB_EVENT (request | end), OPENLAB_AGENT and OPENLAB_ROOT in its environment. A missing or failing command
        changes nothing: the inbox and the events hold the same."""
        import shlex
        import subprocess
        cmd = config(self.root).get("notify")
        cmd = shlex.split(cmd) if isinstance(cmd, str) else list(cmd or [])
        if not cmd:
            return
        try:
            subprocess.Popen([*map(str, cmd), text], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True,
                             env={**os.environ, "OPENLAB_EVENT": kind, "OPENLAB_AGENT": agent,
                                  "OPENLAB_ROOT": str(self.root)})
        except OSError:
            pass

    @staticmethod
    def _item(row) -> dict:
        return {**dict(row), "args": json.loads(row["args"]), "result": json.loads(row["result"] or "{}")}

    def item(self, id: int) -> Optional[dict]:
        with closing(self._con()) as con:
            row = con.execute("SELECT * FROM inbox WHERE id=?", (int(id),)).fetchone()
        return self._item(row) if row else None

    def inbox(self, status: Optional[str] = "pending", agent: Optional[str] = None) -> list[dict]:
        cond, args = [], []
        if status:
            cond.append("status=?")
            args.append(status)
        if agent:
            cond.append("agent=?")
            args.append(agent)
        with closing(self._con()) as con:
            rows = con.execute("SELECT * FROM inbox" + (f" WHERE {' AND '.join(cond)}" if cond else "")
                               + " ORDER BY id", args).fetchall()
        return [self._item(r) for r in rows]

    def decide(self, id: int, status: str, by: str, reason: str, result: Optional[dict] = None,
               sig: Optional[dict] = None) -> dict:
        """Close a pending item; `sig` is the person's signature over the decision (openlab.agents.inbox)."""
        if status not in ("approved", "denied", "cancelled", "answered"):
            raise ValueError("status is approved, denied, cancelled or answered")
        with self._write() as con:
            cur = con.execute("UPDATE inbox SET status=?, decided_by=?, decided_at=?, decision_reason=?, result=? "
                              "WHERE id=? AND status='pending'",
                              (status, by, now(), reason, canonical(result or {}), int(id)))
            if cur.rowcount == 0:
                raise ValueError(f"inbox item #{id} is not pending")
        it = self.item(id)
        self.event(it["agent"], "decision", summary=f"#{id} {it['kind']} {status} by {by}: {reason}",
                   payload={"item": int(id), "status": status, "by": by, "reason": reason, "result": result or {},
                            **({"sig": sig} if sig else {})})
        return it

    def set_result(self, id: int, result: dict) -> None:
        """The result of an approved item whose act ran after the decision (in the background)."""
        with self._write() as con:
            con.execute("UPDATE inbox SET result=? WHERE id=?", (canonical(result or {}), int(id)))
        it = self.item(id)
        ok = "error" not in (result or {})
        self.event(it["agent"], "decision_result", summary=f"#{id} {it['kind']} " + (
            "done" if ok else f"failed: {result.get('error')}"), payload={"item": int(id), "result": result or {}})

    # --- agent tokens (who is who) -----------------------------------------------------------------------------
    def issue_token(self, id: str) -> str:
        """A fresh secret for agent `id` (handed to its process's environment as OPENLAB_AGENT_TOKEN); the store
        keeps only its sha256, so a write that claims `id` without it is attributed `unverified:<id>`."""
        import secrets
        token = secrets.token_urlsafe(24)
        self.update(id, token_sha=token_sha(token))
        path = self.agent_dir(id) / "token"                 # for the hooks, which re-inject it per Bash call
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(token)
        return token

    def token(self, id: str) -> str:
        """The agent's token (issued once), or ""."""
        try:
            return (self.dir / "agents" / id.replace("/", "_") / "token").read_text().strip()
        except OSError:
            return ""

    def by_token(self, token: str) -> Optional[dict]:
        if not token:
            return None
        found = self.find(token_sha=token_sha(token))
        return found[0] if found else None

    # --- MCP attribution ---------------------------------------------------------------------------------------
    def pend(self, agent: str, tool: str, tool_input: dict, session_id: str = "", sha: str = "") -> None:
        """A hook saw `agent` call `tool` with `tool_input`; the MCP server claims it (or a sub-agent start)."""
        with self._write() as con:
            con.execute("DELETE FROM pending_calls WHERE at<?", (time.time() - 600,))
            con.execute("INSERT INTO pending_calls(session_id, agent, tool, sha, input, at) VALUES (?,?,?,?,?,?)",
                        (session_id, agent, tool, sha or input_sha(tool_input), canonical(bound(tool_input)),
                         time.time()))

    def claim(self, tool: str, args: dict, by: str = "mcp", window: float = CLAIM_WINDOW,
              match=None, among: Optional[set] = None) -> Optional[dict]:
        """The oldest unclaimed pending call of `tool` within `window` seconds whose input is `args` (same sha), or
        whose every key the server's `args` repeat (the server may fill in defaults); claimed atomically. `among`:
        only calls by these agents."""
        sha = input_sha(args)
        want = json.loads(canonical(args))
        with self._write() as con:
            rows = con.execute("SELECT * FROM pending_calls WHERE tool=? AND claimed_by='' AND at>? ORDER BY id",
                               (tool, time.time() - window)).fetchall()
            if among is not None:
                rows = [r for r in rows if r["agent"] in among]
            pick = next((r for r in rows if r["sha"] == sha), None)
            if pick is None:
                pick = next((r for r in rows if (match or _subset)(json.loads(r["input"]), want)), None)
            if pick is None:
                return None
            con.execute("UPDATE pending_calls SET claimed_by=? WHERE id=?", (by, pick["id"]))
        return {**dict(pick), "input": json.loads(pick["input"])}


def _subset(seen: dict, args: dict) -> bool:
    return all(k in args and args[k] == v for k, v in seen.items())
