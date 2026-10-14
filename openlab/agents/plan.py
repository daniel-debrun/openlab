"""An agent's plan: the steps it means to take, reported by the agent and checked against the record.

An agent reports its whole plan as a version (`agent_plan`), and each step's state as it moves (`agent_plan_node`).
Both are appended to `.openlab/agents/<id>/plan.jsonl` (a version line or a step line, never rewritten) and noted as
an event in the control store, whose chain covers the line's sha. `openlab show` draws the plan.

A step is drawn in two channels. Its fill is its state, which the agent reports. Its outline is its evidence, which
openlab computes from the record ids the step links to and the agent never types:

- `ok`: every link is on the record (a record node, an inbox item, a queue ticket), and none disagrees;
- `rep`: nothing is linked yet, so the state is the agent's word only;
- `bad`: the record disagrees: a link is not on the record, or the agent says done and a linked run's truncation
  audit found look-ahead.

Appends hold an fcntl lock on the agent's `plan.lock`, so two processes never interleave lines or number two
versions alike. `verify` checks every line's sha against the control store's events.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
import unicodedata
from collections import Counter
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from .store import PAYLOAD_MAX, Store, bound, canonical, now

STATES = ("planned", "active", "waiting", "done", "failed", "dropped")
KINDS = ("review", "data", "run", "branch", "test", "select", "person", "paper", "critique", "check", "lesson", "other")
_KIND_TRY = (f"kind is one of {', '.join(KINDS)} (reading or a survey is review, a claim is paper or test, a step "
             "that fits none is other)")
_STATE_TRY = f"state is one of {', '.join(STATES)}"
PAST = ("done", "failed", "dropped")
MAX_NODES = 60
MAX_LINKS = 12
_ID = re.compile(r"[A-Za-z0-9_.-]{1,40}")
_LINK = re.compile(r"[a-z_]+:[0-9A-Za-z#._-]{1,64}")
_DIR = re.compile(r"[A-Za-z0-9_.-]+")
_BIDI = dict.fromkeys([*range(0x202A, 0x202F), *range(0x2066, 0x206A)])    # direction overrides: a title reads as typed


class PlanError(ValueError):
    def __init__(self, error: str, try_: str = ""):
        super().__init__(error)
        self.error, self.try_ = error, try_


def _path(store: Store, agent: str) -> Path:
    """The agent's plan file. An id outside [A-Za-z0-9_.-] (one with '/') is percent-encoded, so `a/b` and `a_b`
    never share a file and no id (`..`) names a folder outside .openlab/agents."""
    agent = str(agent or "")
    if not agent:
        raise PlanError("no agent id", "run as an openlab agent (OPENLAB_AGENT)")
    name = agent if _DIR.fullmatch(agent) and agent.strip(".") else quote(agent, safe="").replace(".", "%2E")
    return store.agent_dir(name) / "plan.jsonl"


@contextmanager
def _locked(store: Store, agent: str):
    """The agent's plan lock, held from reading the plan to writing its event, so concurrent writers take turns."""
    with open(_path(store, agent).with_name("plan.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _seq(x, what: str) -> list:
    """A list given as a list (or one id as a string), never a string's characters."""
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if not isinstance(x, (list, tuple)):
        raise PlanError(f"{what} is a list of ids, not {type(x).__name__}")
    return list(x)


def _links(links) -> list:
    links = _seq(links, "links")
    if len(links) > MAX_LINKS:
        raise PlanError(f"a step links at most {MAX_LINKS} ids (this one has {len(links)})",
                        "link the ids the step rests on most; a run's id covers its datasets")
    links = list(dict.fromkeys(str(x).strip() for x in links if str(x).strip()))
    bad = [x for x in links if not _LINK.fullmatch(x)]
    if bad:
        raise PlanError(f"{bad[0]!r} is not a record id", "link record ids such as run:1a07c3, test:9b0e, "
                        "dataset:c84f, inbox:7 or ticket:43")
    bad = [x for x in links if x.startswith("inbox:") and not x[6:].isdigit()]
    if bad:
        raise PlanError(f"{bad[0]!r} is not an inbox id", "an inbox id is the item's number, such as inbox:7")
    return links


def _title(x) -> str:
    """A title as one line: NFC, whitespace collapsed, no control or direction-override characters, 120 at most."""
    t = unicodedata.normalize("NFC", str(x or "")).translate(_BIDI)
    return " ".join("".join(" " if unicodedata.category(c) == "Cc" else c for c in t).split())[:120]


def check(nodes, people: tuple = ()) -> list:
    """The nodes made exact, or PlanError: unique ids, a known kind and state, needs among the plan's own ids and
    no cycle, links shaped as record ids, an owner (when given) that is an agent id or a declared person."""
    if not isinstance(nodes, list) or not nodes:
        raise PlanError("a plan is a non-empty list of steps", 'nodes [{"id": "emos", "title": "Baseline: EMOS", '
                        '"kind": "run", "needs": ["data"], "state": "planned"}]')
    if len(nodes) > MAX_NODES:
        raise PlanError(f"a plan has at most {MAX_NODES} steps", "group small steps into one")
    out, ids = [], set()
    for n in nodes:
        if not isinstance(n, dict):
            raise PlanError("each step is an object with id, title, kind, needs and state")
        nid = str(n.get("id") or "")
        if not _ID.fullmatch(nid):
            raise PlanError(f"step id {nid!r} is not 1-40 letters, digits, _ . or -")
        if nid in ids:
            raise PlanError(f"step id {nid!r} appears twice")
        ids.add(nid)
        title = _title(n.get("title"))
        if not title:
            raise PlanError(f"step {nid} has no title")
        kind = str(n.get("kind") or "other")
        if kind not in KINDS:
            raise PlanError(f"step {nid}: kind {kind!r} is not a step kind", _KIND_TRY)
        state = str(n.get("state") or "planned")
        if state not in STATES:
            raise PlanError(f"step {nid}: state {state!r} is not a step state", _STATE_TRY)
        owner = n.get("owner")
        if owner is not None and not (isinstance(owner, str) and _ID.fullmatch(owner)):
            raise PlanError(f"step {nid}: owner {owner!r} is not 1-40 letters, digits, _ . or -",
                            "an owner is an agent id or a person declared in openlab.toml [team], such as you")
        needs = list(dict.fromkeys(str(x) for x in _seq(n.get("needs"), f"step {nid}: needs")))
        if nid in needs:
            raise PlanError(f"step {nid} needs itself", "a step needs only the steps it waits on")
        links = _links(n.get("links"))
        if state == "waiting" and not links:
            raise PlanError(f"step {nid} waits on nothing it links",
                            "file what it waits on first (openlab_agent_request) and link it, links [\"inbox:<id>\"] (or "
                            "a queue ticket:<id>); a step that waits for a date or for later stays planned, with a note")
        out.append({"id": nid, "title": title, "kind": kind, "needs": needs,
                    "state": state, "links": links, **({"owner": owner} if owner else {})})
    for n in out:
        missing = [x for x in n["needs"] if x not in ids]
        if missing:
            raise PlanError(f"step {n['id']} needs {', '.join(missing)}, which the plan does not have")
    by, done = {n["id"]: n for n in out}, set()

    def visit(i, path):
        if i in done:
            return
        if i in path:
            raise PlanError(f"the plan has a cycle through {i}", "a step cannot need itself, directly or not")
        for x in by[i]["needs"]:
            visit(x, path | {i})
        done.add(i)
    for i in by:
        visit(i, set())
    return out


def report(store: Store, agent: str, nodes, note: str = "") -> dict:
    """A new version of the agent's plan."""
    nodes = check(nodes)
    with _locked(store, agent):
        versions = read(store, agent)["versions"]
        line = {"v": len(versions) + 1, "at": now(), "note": str(note or "").strip()[:300], "nodes": nodes}
        sha = _append(store, agent, line)
        prev = versions[-1] if versions else None
        payload = {"v": line["v"], "steps": len(nodes), "sha": sha, "note": line["note"],
                   **({"changed": diff(prev, line)} if prev else {})}
        if len(canonical(bound(payload))) > PAYLOAD_MAX:            # the sha is kept whole, however long the titles
            payload = {"v": line["v"], "steps": len(nodes), "sha": sha, "note": line["note"][:200]}
        store.event(agent, "plan", summary=f"plan v{line['v']}: {len(nodes)} steps" + (f"; {line['note']}" if line["note"] else ""),
                    payload=payload)
    return {"v": line["v"], "steps": len(nodes), "sha": sha}


def step(store: Store, agent: str, id: str, state: str, links=None, note: str = "") -> dict:
    """One step of the agent's latest plan moves to `state`, with the record ids it now rests on."""
    if state not in STATES:
        raise PlanError(f"state {state!r} is not a step state", _STATE_TRY)
    links = _links(links) if links is not None else None
    with _locked(store, agent):
        plan = read(store, agent)
        if not plan["versions"]:
            raise PlanError("you have not reported a plan yet", "report it first with agent_plan")
        cur = plan["versions"][-1]
        node = next((n for n in cur["nodes"] if n["id"] == id), None)
        if node is None:
            raise PlanError(f"your plan v{cur['v']} has no step {id!r}", "steps: " + ", ".join(n["id"] for n in cur["nodes"]))
        if state == "waiting" and not (links if links is not None else node.get("links")):
            raise PlanError(f"step {id} waits on nothing it links",
                            "file what it waits on first (openlab_agent_request) and link it, links=[\"inbox:<id>\"] (or "
                            "a queue ticket:<id>); a step that waits for a date or for later stays planned, with a note")
        line = {"step": id, "at": now(), "state": state, "note": str(note or "").strip()[:300],
                **({"links": links} if links is not None else {})}
        sha = _append(store, agent, line)
        store.event(agent, "plan_node", summary=f"{id}: {node['state']} → {state}" + (f"; {line['note']}" if line["note"] else ""),
                    payload={"step": id, "from": node["state"], "to": state, "links": line.get("links"), "sha": sha})
    return {"step": id, "state": state, "v": cur["v"]}


def _append(store: Store, agent: str, line: dict) -> str:
    """One line, under the caller's lock. A cut last line (a writer killed mid-line) is ended first, so the new
    line never runs on from it."""
    text = canonical(line)
    p = _path(store, agent)
    with open(p, "ab+") as f:
        if f.tell():
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                f.write(b"\n")
        f.write((text + "\n").encode())
    return hashlib.sha256(text.encode()).hexdigest()


def _lines(p: Path):
    """(line number, text, its JSON object or None when it is not one) for each line of a plan file."""
    for i, raw in enumerate(p.read_bytes().decode("utf-8", "replace").split("\n"), 1):
        if not raw:
            continue
        try:
            line = json.loads(raw)
        except ValueError:
            line = None
        yield i, raw, line if isinstance(line, dict) else None


def _shaped(line: dict) -> bool:
    if "nodes" in line:
        return isinstance(line["nodes"], list) and "v" in line and "at" in line and all(
            isinstance(n, dict) and isinstance(n.get("id"), str) and n.get("state") in STATES for n in line["nodes"])
    return isinstance(line.get("step"), str) and line.get("state") in STATES and "at" in line


def read(store: Store, agent: str) -> dict:
    """{"versions": [{v, at, note, nodes}]}, each node with its state as of the next version, and the latest with
    every reported step applied; each node carries `t0` (first active) and `t1` (when it ended) where known."""
    p = _path(store, agent)
    versions: list = []
    if not p.is_file():
        return {"versions": versions}
    times: dict = {}
    for _, _, line in _lines(p):
        if line is None or not _shaped(line):                          # a cut or spoiled line: the rest still reads
            continue
        if "nodes" in line:
            for n in line["nodes"]:
                t = times.setdefault(n["id"], {})
                if n["state"] != "planned":
                    t.setdefault("t0", line["at"])
                if n["state"] in PAST:
                    t.setdefault("t1", line["at"])
            versions.append({"v": line["v"], "at": line["at"], "note": line.get("note", ""),
                             "nodes": [dict(n) for n in line["nodes"]]})
        elif "step" in line and versions:
            n = next((x for x in versions[-1]["nodes"] if x["id"] == line["step"]), None)
            if n is None:
                continue
            n["state"] = line["state"]
            if "links" in line:
                n["links"] = line["links"]
            if line.get("note"):
                n["note"] = line["note"]
            t = times.setdefault(n["id"], {})
            if line["state"] in ("active", "waiting"):
                t.setdefault("t0", line["at"])
                t.pop("t1", None)
            if line["state"] in PAST:
                t.setdefault("t0", line["at"])
                t["t1"] = line["at"]
    for v in versions:
        for n in v["nodes"]:
            n.update(times.get(n["id"], {}) if v is versions[-1] else {})
    return {"versions": versions}


def verify(store: Store) -> dict:
    """Every agent's plan.jsonl against the control store's plan and plan_node events, whose payloads hold each
    line's sha: a line no event names was edited or added by hand (`unrecorded`), an event whose line is gone was
    deleted (`missing`), lines out of their events' order were moved (`reordered`), and a line that is not JSON was
    cut or spoiled (`corrupt`). {"ok", "agents": [{agent, ok, lines, events, unrecorded, missing, corrupt,
    reordered}]}, line numbers from 1."""
    evs: dict = {}
    for e in store.events(n=10 ** 9, kinds=("plan", "plan_node")):
        evs.setdefault(e["agent"], []).append(e)
    out = []
    for agent in sorted({a["id"] for a in store.agents()} | set(evs)):
        try:
            p = _path(store, agent)
        except PlanError:
            continue
        owned = evs.get(agent, [])
        lines = list(_lines(p)) if p.is_file() else []
        if not lines and not owned:
            continue
        left = Counter(e["payload"].get("sha") for e in owned)
        unrecorded, order = [], []
        for i, raw, line in lines:
            if line is None:
                continue
            sha = hashlib.sha256(raw.encode()).hexdigest()
            if left[sha] > 0:
                left[sha] -= 1
                order.append(sha)
            else:
                unrecorded.append(i)
        missing = []
        for e in reversed(owned):                                       # the latest events of a sha are the unmatched
            sha = e["payload"].get("sha")
            if left[sha] > 0:
                left[sha] -= 1
                missing.append(e["seq"])
        kept = [e["payload"].get("sha") for e in owned if e["seq"] not in missing]
        corrupt = [i for i, _, line in lines if line is None]
        a = {"agent": agent, "lines": len(lines), "events": len(owned), "unrecorded": unrecorded,
             "missing": sorted(missing), "corrupt": corrupt, "reordered": order != kept}
        out.append({"ok": not (unrecorded or missing or corrupt or a["reordered"]), **a})
    return {"ok": all(a["ok"] for a in out), "agents": out}


def diff(prev: Optional[dict], cur: dict) -> dict:
    if not prev:
        return {"added": [n["title"] for n in cur["nodes"]], "removed": [], "changed": []}
    old = {n["id"]: n for n in prev["nodes"]}
    new = {n["id"]: n for n in cur["nodes"]}
    changed = []
    for i, n in new.items():
        o = old.get(i)
        if o and (o["state"], o.get("title"), o.get("needs"), o.get("owner")) != (n["state"], n.get("title"),
                                                                                    n.get("needs"), n.get("owner")):
            changed.append(n["title"])
    return {"added": [n["title"] for i, n in new.items() if i not in old],
            "removed": [n["title"] for i, n in old.items() if i not in new], "changed": changed}


# --- evidence: what the record says about a step ----------------------------------------------------------
def _record(root: Path, study: str):
    if not study:
        return None
    try:
        from ..study.record import Record
        folder = (Path(root) / study).resolve()
        return Record(folder) if (folder / ".openlab" / "record.db").is_file() else None
    except Exception:                                                   # noqa: BLE001 -- evidence never fails a view
        return None


def _resolve(rec, link: str):
    """(the record node a link names, or None; how many ids it matches): its id, or a unique prefix of one
    (run:1a07 for run:1a07c3e9f2b4). A prefix of two or more ids is ambiguous, never a guess."""
    try:
        return rec.get(link), 1
    except Exception:                                                   # noqa: BLE001
        pass
    try:
        with closing(rec._con()) as con:                                # a range on the id index: no scan, no LIKE wildcards
            rows = con.execute("SELECT id FROM nodes WHERE id >= ? AND id < ? LIMIT 2",
                               (link, link + "\U0010ffff")).fetchall()
        return (rec.get(rows[0]["id"]) if len(rows) == 1 else None), len(rows)
    except Exception:                                                   # noqa: BLE001 -- evidence never fails a view
        return None, 0


def _elsewhere(root: Path, study: str, link: str) -> tuple:
    """(the name of another study of the workspace whose record holds `link`, its node), or ("", None). A step of
    an agent with no study of its own (a workspace lead) rests on any study's record."""
    try:
        from ..study.workspace import studies
        base = Path(root).resolve()
        owned = (base / study).resolve() if study else None
        for folder in studies(base):
            if folder.resolve() == owned:
                continue
            rec = _record(base, str(folder.resolve().relative_to(base)))
            n = _resolve(rec, link)[0] if rec is not None else None
            if n is not None:
                return folder.name, n
    except Exception:                                                   # noqa: BLE001 -- evidence never fails a view
        pass
    return "", None


def _study_named(root: Path, ref: str) -> str:
    """'study NAME: its record holds N entries' for the study of the workspace with that folder name (or path under
    the workspace) and a record, else ''."""
    try:
        from ..study.record import Record
        from ..study.workspace import studies
        base = Path(root).resolve()
        for folder in studies(base):
            f = folder.resolve()
            if ref.strip("/") in (f.name, str(f.relative_to(base))) and (f / ".openlab" / "record.db").is_file():
                return f"study {ref}: its record holds {sum(Record(f).kinds().values())} entries"
    except Exception:                                                   # noqa: BLE001 -- evidence never fails a view
        pass
    return ""


def evidence(store: Store, study: str, node: dict, rec=None, items: Optional[dict] = None) -> tuple:
    """(ok | rep | bad, a line saying what the record shows) for one step. `items`: the inbox by id, read once by a
    caller that checks many steps."""
    links = node.get("links") or []
    if not links:
        return "rep", node.get("note") or ""
    rec = rec if rec is not None else _record(store.root, study)
    said, bad = [], []
    if not isinstance(links, list):
        links = [links]
    if len(links) > MAX_LINKS:                                          # a hand-edited line: check the first, say so
        bad.append(f"{len(links) - MAX_LINKS} links past the {MAX_LINKS} a step may hold were not checked")
        links = links[:MAX_LINKS]
    for link in links:
        kind, _, ref = str(link).partition(":")
        if kind == "inbox":
            if not (ref.isascii() and ref.isdigit() and len(ref) <= 18):
                bad.append(f"{link} is not an inbox id")
                continue
            it = items.get(int(ref)) if items is not None else store.item(int(ref))
            (said if it else bad).append(f"inbox #{ref}" + (f" {it['status']}" if it else " is not in the inbox"))
            continue
        if kind == "ticket":
            said.append(f"queue ticket {ref}")
            continue
        if kind == "study":                                             # the study itself: its record exists
            got = _study_named(store.root, ref)
            (said if got else bad).append(got or f"{link}: no study of that name has a record in the workspace")
            continue
        n, hits = _resolve(rec, str(link)) if rec is not None else (None, 0)
        if hits > 1:
            bad.append(f"{link} is ambiguous: it starts several record ids; link more of the id")
            continue
        where = ""
        if n is None:                                                   # another study of the workspace counts too
            other, n = _elsewhere(store.root, study, str(link))
            if n is None:
                bad.append(f"{link} is not on the record")
                continue
            where = f" (on study {other}'s record)"
        if n.kind == "run":
            verdict = (n.payload.get("audit") or {}).get("verdict")
            if verdict == "look-ahead" and node["state"] == "done":
                bad.append(f"{n.id}{where}: the audit found LOOK-AHEAD; the agent says done, the record disagrees")
                continue
            said.append(f"{n.id}{where}: audit {verdict}")
        else:
            try:
                from ..study.show import line
                said.append(f"{n.id}{where}: {line({'kind': n.kind, 'payload': n.payload})}")
            except Exception:                                           # noqa: BLE001
                said.append(n.id + where)
    return ("bad", "; ".join(bad + said)) if bad else ("ok", "; ".join(said))
