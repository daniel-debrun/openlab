"""The agent control plane as tools (openlab.agents): delegate a sub-question to a child agent, see the tree, wait for
children, ask a person, post progress. The caller is `identity.current()`: through the MCP, the agent that made
the call. Descriptions are kept short: every sub-agent pays for them in context."""
from __future__ import annotations

import re
import time
from collections import Counter
from pathlib import Path

from typing import Optional, Union

from .registry import ToolContext, ToolError, tool

WAIT_MAX = 300              # seconds one agent_wait call blocks at most
LOG_MAX = 200
BRIEF_MAX = 4000            # characters of a spawned agent's prompt in agent_tree's detail


def _store(ctx: ToolContext):
    from ..agents.store import Store, root_of
    return Store(root_of(ctx.root))


def _self_id() -> str:
    from ..agents.identity import current
    return current()


def _row(a: dict) -> dict:
    s, b = a["spent"], a["budget"]
    return {"id": a["id"], "role": a["role"], "parent": a["parent"] or None, "kind": a["kind"], "status": a["status"],
            "study": a["study"] or None, "question": a["question"] or None,
            "spent_usd": round(s.get("usd") or 0, 4), "budget_usd": b.get("usd"),
            "tokens": (s.get("tokens_in") or 0) + (s.get("tokens_out") or 0), "last_event_at": a["last_event_at"]}


@tool("research", mutates=True)
def agent_spawn(ctx: ToolContext, prompt: str, role: str = "agent", study: str = "", question: str = "",
                budget_usd: float = 0.0, minutes: float = 0.0, cli: str = "claude", model: str = "",
                decide: str = "") -> dict:
    """Delegate a sub-question to a new child agent (you are its parent); its budget comes out of yours. Roles
    with their own brief: adversary (attacks a result before it goes into a paper). `decide` "self": the child
    makes its own choices instead of asking the person (a child of such an agent always does)."""
    from ..agents.spawn import AgentError, spawn
    self_id = _self_id()
    parent = self_id if _store(ctx).get(self_id) else None
    try:
        a = spawn(_store(ctx).root, prompt, role=role, study=study or None, question=question or None,
                  parent=parent, cli=cli, model=model or None,
                  budget={"usd": budget_usd or None, "minutes": minutes or None}, decide=decide)
    except AgentError as e:
        raise ToolError(e.error, e.meaning, e.try_) from None
    return {**_row(a), "model": a["model"] or None,
            "next": "openlab_agent_wait(ids) waits for it; openlab_tool('agent_log', {'id': ...}) shows its events"}


@tool("research")
def agent_list(ctx: ToolContext, all: bool = False) -> dict:
    """The agents in this workspace (live ones unless all=true), with status, budget and spend."""
    st = _store(ctx)
    st.reap()
    return {"agents": [_row(a) for a in st.agents(live_only=not all)], "you": _self_id()}


@tool("research")
def agent_status(ctx: ToolContext, id: str = "") -> dict:
    """One agent (default: you): status, budget, spend, children, last events."""
    st = _store(ctx)
    a = st.get(id or _self_id())
    if a is None:
        raise ToolError(f"no agent {id or _self_id()!r}", "it is not registered in this workspace", "agent_list shows them")
    return {**_row(a), "budget": a["budget"], "spent": a["spent"], "control": a["control"] or None,
            "children": [_row(c) for c in st.children(a["id"])],
            "last": [f"{e['kind']}: {e['summary']}" for e in st.events(a["id"], n=5)]}


@tool("research")
def agent_log(ctx: ToolContext, id: str, n: int = 20) -> dict:
    """An agent's last n events (what it said, its tool calls, refusals)."""
    evs = _store(ctx).events(id, n=max(1, min(n, LOG_MAX)))
    return {"events": [{"seq": e["seq"], "at": e["at"], "kind": e["kind"], "tool": e["tool"] or None,
                        "summary": e["summary"]} for e in evs]}


def _ids(ids) -> list:
    """Agent ids as a list, an id, a comma-separated string, or a JSON list in a string."""
    if isinstance(ids, str):
        text = ids.strip()
        if text.startswith("["):
            import json
            try:
                ids = json.loads(text)
            except ValueError:
                ids = text.strip("[]").replace('"', "").replace("'", "")
    if isinstance(ids, (list, tuple)):
        return [str(i).strip() for i in ids if str(i).strip()]
    return [i.strip().strip("'\"") for i in str(ids or "").split(",") if i.strip().strip("'\"")]


def limits(st, id: str) -> Optional[str]:
    """An agent's budget as used so far, one line ("you: 46 of 60 min, $1.31 of $10, 30 of 200 turns"); None for
    an agent with no budget (or none registered)."""
    from ..agents.store import age
    a = st.get(id) if id else None
    if a is None:
        return None
    b, s, out = a["budget"] or {}, a["spent"] or {}, []
    if b.get("minutes"):
        used = max(0.0, age(a["created"]) / 60 - (s.get("paused_minutes") or 0))
        out.append(f"{used:.0f} of {float(b['minutes']):g} min")
    if b.get("usd"):
        out.append(f"{'~' if s.get('usd_estimated') else ''}${s.get('usd') or 0:.2f} of ${float(b['usd']):g}")
    if b.get("turns"):
        out.append(f"{s.get('turns') or 0} of {int(b['turns'])} turns")
    return "you: " + ", ".join(out) if out else None


_RUN = re.compile(r"run-[0-9a-f]{12}")              # a job id given without its job: prefix


def _job(ctx: ToolContext, i: str) -> tuple:
    """(job, study folder) of a `job:run-<hex>` id, or a refusal."""
    from ..study import detached
    job = i.split(":", 1)[-1].strip()
    job = job if job.startswith("run-") else f"run-{job}"
    folder = detached.find(ctx.root, job)
    if folder is None:
        raise ToolError(f"no run job {i}", "a job id is what study_run returned while its run was still going",
                        "study_runs lists the runs in flight, with their job ids")
    return job, folder


@tool("research")
def agent_wait(ctx: ToolContext, ids: Union[list, str], timeout: float = 120.0) -> dict:
    """Wait until the agents (ids) end, your inbox items ("inbox:7") are decided and your runs ("job:run-<hex>", a
    study_run still going) end, or timeout seconds pass (max 300); their final status, the person's answer, or the
    run's result (trial line, primary metric) or refusal. Mix them in one list."""
    from ..agents.store import FINAL
    from ..study import detached
    st, want = _store(ctx), _ids(ids)
    if not want:
        raise ToolError("no ids", "ids is a list of agent ids, inbox items (inbox:7) or runs (job:run-<hex>)",
                        'agent_wait(ids=["<id>"])')
    wait = max(0.0, min(float(timeout), WAIT_MAX))
    deadline = time.monotonic() + wait
    self_id = _self_id()

    def prompted() -> bool:                        # the person's message wakes a waiting lead to answer it
        return bool(st.get(self_id)) and any(e["payload"].get("ask") for e in st.untold(self_id))
    items = [i for i in want if i.startswith("inbox:")]
    bad = [i for i in items if not i.split(":", 1)[1].isdigit()]
    if bad:
        raise ToolError(f"{bad[0]!r} is not an inbox id", "an inbox id is the item's number, such as inbox:7")
    nums = [int(i.split(":", 1)[1]) for i in items]
    jobs = {i: _job(ctx, i) for i in want if i.startswith("job:") or (_RUN.fullmatch(i) and st.get(i) is None)}
    agents = [i for i in want if i not in items and i not in jobs]
    got, rows, ran = {}, {}, {}
    while True:
        got = {n: st.item(n) for n in nums}
        if any(v is None for v in got.values()):
            raise ToolError(f"no inbox item {', '.join(str(n) for n, v in got.items() if v is None)}")
        if agents:
            st.reap()
            rows = {i: st.get(i) for i in agents}
            missing = [i for i, a in rows.items() if a is None]
            if missing:
                raise ToolError(f"no agent {', '.join(missing)}", "", "agent_list shows the agents")
        ran = {i: detached.poll(folder, job) for i, (job, folder) in jobs.items()}
        done = (all(v["status"] != "pending" for v in got.values()) and all(a["status"] in FINAL for a in rows.values())
                and all(r["state"] != "running" for r in ran.values()))
        if done or time.monotonic() >= deadline or prompted():
            break
        time.sleep(1.0)
    out = {"done": done}
    if agents:
        out["agents"] = [{**_row(a), "result": next((e["summary"] for e in reversed(st.events(a["id"], n=30))
                                                     if e["kind"] in ("result", "end", "say")), None)}
                         for a in rows.values()]
    if items:
        out["items"] = [{k: v[k] for k in ("id", "kind", "status", "decision_reason", "result")} for v in got.values()]
    if jobs:
        out["runs"] = [{k: (_rel(ctx.root, v) if k == "study" else v) for k, v in r.items()} for r in ran.values()]
    if not done:
        out["next"] = "still waiting: call again to keep waiting"
    if float(timeout) > WAIT_MAX and not done:
        out["waited"] = f"{WAIT_MAX} s (the most one call waits)"
    if prompted():
        out.update(prompted=True, next="the person sent you a message (it follows): answer it with openlab_agent_reply, "
                   "then call openlab_agent_wait again")
    if you := limits(st, self_id):
        out["limits"] = you
    return out


@tool("research", mutates=True)
def agent_request(ctx: ToolContext, kind: str, reason: str, study: str = "", args: Optional[dict] = None,
                  cost_note: str = "") -> dict:
    """Ask a person for an act you cannot do (kind: holdout, amend, approval, budget, screen, fresh, continues,
    compute, question, other) with the exact change in args (budget: {"usd": new total}; approval: {"on", "decision"};
    question: {"choices": [...], "multi": false, "other": true}, answered with the person's pick or own words; fresh:
    {"why"}; compute (another machine, when a run needs more than this one has or the queue runs past its
    availability): {"gpus", "memory_gb", "cores", "hours"};
    continues: {"continues": "../study"}; amend: {"study.toml": {"split": {"scheme": "kfold", "k": 5}}}, the fields
    the person's approval writes before it records the amendment, so it holds even after you end); returns the inbox
    item id."""
    from ..agents.inbox import RequestError, check_request
    from ..agents.store import INBOX_KINDS
    if kind not in INBOX_KINDS:
        raise ToolError(f"unknown request kind {kind!r}", f"one of {', '.join(INBOX_KINDS)}")
    if not reason.strip():
        raise ToolError("a request needs a reason", "the person decides on it")
    st = _store(ctx)
    self_id = _self_id()
    if kind == "question" and (st.get(self_id) or {}).get("decide") == "self":
        raise ToolError("decide it yourself", "the person asked you (and the agents under you) to make your own "
                        "choices rather than ask them", "choose, say what you chose and why with agent_note (and in "
                        "your plan), and carry on; acts only the person signs still go to the inbox")
    try:
        args = check_request(st, self_id, kind, args, study)
    except RequestError as e:
        raise ToolError(e.error, "the person approves exactly what the request carries; this one carries nothing "
                        "that would change", e.try_) from None
    item = st.request(self_id, kind, reason, study=study, args=args, cost_note=cost_note)
    return {"item": item, "status": "pending",
            "next": f"a person decides in `openlab inbox`; check openlab_agent_inbox_item({item}) later"}


@tool("research")
def agent_inbox_item(ctx: ToolContext, id: int) -> dict:
    """An inbox item you filed: pending, approved or denied, with the person's reason and the result."""
    it = _store(ctx).item(id)
    if it is None:
        raise ToolError(f"no inbox item #{id}")
    return {k: it[k] for k in ("id", "kind", "study", "status", "reason", "decided_by", "decided_at",
                               "decision_reason", "result")} | {"decided": it["status"] != "pending"}


@tool("research")
def agent_comments(ctx: ToolContext, study: str = "", all: bool = False) -> dict:
    """The person's comments on the research's files, open ones only unless `all`: a lead sees every one, another
    agent its own study's (or `study`'s). Each with its `path` from the research's folder, the line it is on now,
    the quoted text (none: on the whole file), the comment and its replies."""
    from ..agents.comments import locate, threads, where
    st = _store(ctx)
    if not study:
        caller_row = st.get(_self_id()) or {}
        study = "" if st.is_lead(caller_row) else caller_row.get("study") or ""
    out = []
    for t in threads(st, study or None):
        if t["resolved"] and not all:
            continue
        line, f = None, Path(st.root) / (t["study"] or ".") / (t["file"] or "")
        try:
            text = f.read_text(encoding="utf-8", errors="replace") if f.is_file() and not f.is_symlink() else ""
            at = locate(text, t)
            line = text[:at[0]].count("\n") + 1 if at else None
        except OSError:
            pass
        out.append({**t, "path": where(t), "line": line, "moved": line is None and bool(t["quote"])})
    return {"comments": out, "open": sum(1 for t in out if not t["resolved"]),
            "next": "deal with it (or hand it to an agent under you), then agent_comment_reply(id, what you did, "
                    "resolve=true)" if out else ""}


@tool("research", mutates=True)
def agent_comment_reply(ctx: ToolContext, id: int, text: str, resolve: bool = False) -> dict:
    """Answer the person's comment #id on a file: say what you changed (or why not), and resolve it once it is
    dealt with. The person sees the thread in openlab show and can reopen it."""
    from ..agents.comments import CommentError, reply
    try:
        return reply(_store(ctx), id, text, _self_id(), True if resolve else None)
    except CommentError as e:
        raise ToolError(e.error, "", e.try_) from None


@tool("research", mutates=True)
def agent_plan(ctx: ToolContext, nodes: list, note: str = "") -> dict:
    """Report your whole plan as a new version: nodes [{id, title, kind, needs: [ids], state, links: [record ids],
    owner}]; report it first, and again whenever it changes. The person sees it in openlab show. kind: review, data,
    run, branch, test, select, person, paper, critique, check, lesson, other; state: planned, active, waiting, done,
    failed, dropped."""
    from ..agents.plan import PlanError, report
    try:
        return report(_store(ctx), _self_id(), nodes, note)
    except PlanError as e:
        raise ToolError(e.error, "a plan step is {id, title, kind, needs, state, links}", e.try_) from None


@tool("research", mutates=True)
def agent_plan_node(ctx: ToolContext, id: str, state: str, links: Optional[list] = None, note: str = "") -> dict:
    """One step of your plan moves (planned, active, waiting, done, failed, dropped), with the record ids it rests
    on (run:..., test:..., inbox:7, ticket:43); openlab checks them against the record."""
    from ..agents.plan import PlanError, step
    try:
        return step(_store(ctx), _self_id(), id, state, links, note)
    except PlanError as e:
        raise ToolError(e.error, "a step's state is planned, active, waiting, done, failed or dropped", e.try_) from None


@tool("research", mutates=True)
def agent_note(ctx: ToolContext, text: str) -> dict:
    """A short progress note on your agent's timeline (shown in openlab watch)."""
    st, self_id = _store(ctx), _self_id()
    seq = st.event(self_id, "note", summary=text)
    return {"seq": seq, "agent": self_id, **({"limits": you} if (you := limits(st, self_id)) else {})}


@tool("research", mutates=True)
def agent_reply(ctx: ToolContext, text: str, to: int = 0) -> dict:
    """Answer the person's message (prompt number `to`) in a few sentences, then carry on with your work."""
    st, self_id = _store(ctx), _self_id()
    if st.get(self_id) is None:
        raise ToolError("only a registered agent replies", f"{self_id!r} is not an agent of this workspace")
    if not str(text or "").strip():
        raise ToolError("the reply is empty", "", "say what the person asked about in a few sentences")
    try:
        seq = st.reply(self_id, str(text).strip(), int(to or 0))
    except ValueError as e:
        raise ToolError(str(e)) from None
    return {"seq": seq, "agent": self_id, "next": "carry on with your work"}


def _lead_or_refuse(st, id: str) -> dict:
    a = st.get(id)
    if a is None:
        raise ToolError(f"no agent {id!r}", "it is not registered in this workspace", "agent_tree lists them")
    if not st.is_lead(a):
        lead = next((x["id"] for x in reversed(st.ancestors(id)) if st.is_lead(x)), a["parent"])
        raise ToolError(f"{id} works under {lead}: only a lead takes your prompts",
                        "an agent under a lead hears from its lead, so the lead stays the one who directs it",
                        f"prompt {lead} instead; it can pass it on")
    return a


@tool("research", mutates=True, human=True)
def agent_prompt(ctx: ToolContext, id: str, text: str) -> dict:
    """A person's message to a lead while it works: it reads it at its next tool call, answers with agent_reply
    and carries on (nothing is stopped). Only a lead takes prompts. Advice, not a signed act."""
    from ..agents.store import FINAL
    st = _store(ctx)
    a = _lead_or_refuse(st, id)
    if a["status"] in FINAL:
        raise ToolError(f"{id} has ended ({a['status']})", "an ended agent reads nothing",
                        "start a lead to carry the research on")
    text = str(text or "").strip()
    if not text:
        raise ToolError("the message is empty")
    try:
        seq = st.tell(id, text, _person(), ask=True, via="show")
    except ValueError as e:
        raise ToolError(str(e)) from None
    held = st.blocked(id)
    return {"seq": seq, "id": id,
            "when": f"it is {held[1]}: it reads this once resumed" if held else "it reads this at its next tool call"}


@tool("research", mutates=True, human=True)
def agent_btw(ctx: ToolContext, id: str, text: str) -> dict:
    """A person's side question to a lead, answered now from its context (Claude Code's /btw): openlab forks the lead's
    session with no tools and asks it there; the lead keeps working and never sees the exchange. Only a lead takes
    one; an ended lead can still be asked. To change what the lead does, use agent_prompt."""
    from ..agents.btw import BtwError, ask
    st = _store(ctx)
    _lead_or_refuse(st, id)
    try:
        return ask(st, id, str(text or "").strip(), _person())
    except BtwError as e:
        raise ToolError(e.error, e.meaning, e.try_) from None


@tool("research", human=True)
def agent_prompts(ctx: ToolContext, id: str) -> dict:
    """The person's messages to a lead and its replies, oldest first, and whether it can take a prompt now (`open`)
    or a side question (`btw`: it has a Claude session to fork)."""
    from ..agents.store import FINAL
    st = _store(ctx)
    a = _lead_or_refuse(st, id)
    held = st.blocked(id)
    return {"id": id, "status": a["status"], "held": held[1] if held else None, "open": a["status"] not in FINAL,
            "btw": a["cli"] == "claude" and bool(a["session_id"]), "messages": st.conversation(id)}


@tool("research", mutates=True, human=True)
def agent_control_leads(ctx: ToolContext, action: str, reason: str) -> dict:
    """A person's pause or resume of every live lead at once (the agents under each are held with it); resume is
    signed, as agent_control's is."""
    if action not in ("pause", "resume"):
        raise ToolError(f"unknown action {action!r}", "action is pause or resume")
    st = _store(ctx)
    st.reap()
    leads = [a for a in st.agents(live_only=True) if st.is_lead(a)
             and (a["control"] != "pause" if action == "pause" else a["control"] == "pause")]
    done = [agent_control(ctx, a["id"], action, reason) for a in leads]
    return {"action": action, "leads": done}


@tool("research")
def box(ctx: ToolContext) -> dict:
    """This machine as openlab uses it (cores and memory under the container's limits, GPUs), the person's schedule
    (box.toml: available_until, reservations) and its queue of runs: what runs, what waits and why. Read it before
    starting several runs, or when a run says it is queued."""
    from ..boxcli import show
    from ..study import admission
    return {**show(), "tickets": admission.tickets()}


# --- a person's view of the tree: the web app's Agents view reads the snapshot `openlab watch` draws ----------

def _study_key(root, study: str) -> str:
    if not study:
        return ""
    p = Path(study)
    return str((p if p.is_absolute() else Path(root) / p).resolve())


def _rel(root, key: str) -> str:
    try:
        return str(Path(key).relative_to(Path(root).resolve())) or "."
    except ValueError:
        return key


def _event(e: Optional[dict]) -> Optional[dict]:
    return {"at": e["at"], "kind": e["kind"], "tool": e["tool"] or None, "summary": e["summary"]} if e else None


@tool("research")
def agent_tree(ctx: ToolContext, study: str = "", id: str = "", all: bool = True) -> dict:
    """Every agent in tree order with its state and flags (what `openlab watch` shows); `study` keeps one study's
    agents, `id` adds that agent's brief and budget."""
    from ..agents.view import snapshot
    st = _store(ctx)
    snap = snapshot(st, all=all)
    want = _study_key(ctx.root, study) if study else ""
    rows = [r for r in snap["rows"] if not want or _study_key(st.root, r["agent"]["study"]) == want]
    kept = {r["agent"]["id"] for r in rows}
    parent = {r["agent"]["id"]: r["agent"]["parent"] for r in snap["rows"]}

    def depth(aid: str) -> int:                     # kept ancestors only, so a filtered tree is still a tree
        n, seen, p = 0, {aid}, parent.get(aid)
        while p and p not in seen:
            seen.add(p)
            n += p in kept
            p = parent.get(p)
        return n
    out = []
    for r in rows:
        a = r["agent"]
        key = _study_key(st.root, a["study"])
        out.append({"id": a["id"], "parent": a["parent"] or None, "depth": depth(a["id"]) if want else r["depth"],
                    "role": a["role"], "kind": a["kind"], "status": a["status"], "state": r["state"],
                    "flags": r["flags"], "control": a["control"] or None,
                    "study": _rel(ctx.root, key) if key else None, "question": a["question"] or None,
                    "usd": round(r["usd"], 4), "usd_estimated": r["usd_estimated"], "budget_usd": r["budget_usd"],
                    "frac": round(r["frac"], 3), "tokens": r["tokens"], "trials": r["trials"], "claims": r["claims"],
                    "refusals": r["refusals"], "pending": r["pending"], "last": _event(r["last"]),
                    "last_refusal": _event(r["last_refusal"]),
                    "idle_s": None if r["idle_for"] == float("inf") else round(r["idle_for"], 1),
                    "dead_end": {"agent": r["dead_end"][0], "text": r["dead_end"][1]} if r["dead_end"] else None})
    res = {"agents": out, "pending": sum(x["pending"] for x in out) if want else snap["pending"],
           "usd": round(sum(x["usd"] for x in out) if want else snap["usd"], 4),
           "counts": dict(Counter(x["state"] for x in out))}
    if id:
        a = st.get(id)
        if a is None:
            raise ToolError(f"no agent {id!r}", "it is not registered in this workspace", "agent_tree lists them")
        f = st.dir / "agents" / id.replace("/", "_") / "prompt.txt"
        res["detail"] = {"id": id, "prompt": f.read_text(encoding="utf-8", errors="replace")[:BRIEF_MAX]
                         if f.is_file() else None,
                         "budget": a["budget"], "spent": a["spent"], "control": a["control"] or None,
                         "control_reason": a["control_reason"] or None, "cli": a["cli"] or None,
                         "model": a["model"] or None, "cwd": a["cwd"] or None, "created": a["created"],
                         "exit_code": a["exit_code"], "children": [c["id"] for c in st.children(id)]}
    return res


def _person() -> str:
    """Who acts in a request that carried the reviewer secret: the person whose key signs it (the web server opened
    the person key with the passphrase sent in the request: openlab.app.server), else the label of the person who
    started `openlab app`."""
    from ..agents.identity import person_label
    from ..study.personkey import current
    signer = current()
    return signer.label if signer is not None else person_label()


def _signer_or_refuse(what: str):
    from ..study.personkey import current
    signer = current()
    if signer is None:
        raise ToolError("passphrase required", f"{what} is a person's act, signed with your person key",
                        "send your passphrase as `passphrase` (the app asks for it)")
    return signer


@tool("research", mutates=True, human=True)
def agent_control(ctx: ToolContext, id: str, action: str, reason: str) -> dict:
    """A person's pause, resume or stop of an agent (its descendants are held with it), with the reason. A
    person's act: agents cannot call it."""
    if action not in ("pause", "resume", "stop"):
        raise ToolError(f"unknown action {action!r}", "action is pause, resume or stop")
    if not reason.strip():
        raise ToolError("a reason is required", "it is recorded on the agent's timeline")
    st = _store(ctx)
    if st.get(id) is None:
        raise ToolError(f"no agent {id!r}", "it is not registered in this workspace", "agent_tree lists them")
    if action == "resume":                                          # releasing a held agent is signed
        signer = _signer_or_refuse(f"resuming {id}")
        sig = {"key": signer.fingerprint, "value": signer.sign(f"openlab-resume|{id}|{reason.strip()}".encode())}
        a = st.control(id, action, reason.strip(), signer.label, sig=sig)
    else:
        a = st.control(id, action, reason.strip(), _person())
    return {"id": id, "status": a["status"], "control": a["control"] or None}


def _item(it: dict) -> dict:
    return {k: it[k] for k in ("id", "agent", "study", "kind", "args", "reason", "cost_note", "status", "created",
                               "decided_by", "decided_at", "decision_reason", "result")}


@tool("research")
def inbox_list(ctx: ToolContext, all: bool = False, study: str = "") -> dict:
    """The requests agents filed for a person (pending ones unless all=true), newest first: who asked, why, what
    it costs."""
    st = _store(ctx)
    rows = st.inbox(None if all else "pending")
    if study:
        want = _study_key(ctx.root, study)
        rows = [i for i in rows if i["study"] and _study_key(st.root, i["study"]) == want]
    from ..agents.inbox import requester_note
    return {"items": [{**_item(i), "requester_ended": requester_note(st, i) or None} for i in reversed(rows)],
            "pending": sum(i["status"] == "pending" for i in rows)}


@tool("research", mutates=True, human=True)
def inbox_answer(ctx: ToolContext, id: int, picks: Optional[list] = None, other: str = "", message: str = "") -> dict:
    """A person answers an agent's question: the choices picked, their own answer (`other`), and a message. Advice
    to the agent, not a signed act."""
    from ..agents.inbox import answer
    try:
        it = answer(_store(ctx), int(id), picks or [], other, message, _person())
    except ValueError as e:
        raise ToolError(str(e), "", "inbox_list shows the pending items") from None
    return _item(it)


@tool("research", mutates=True, human=True)
def inbox_decide(ctx: ToolContext, id: int, approve: bool, reason: str, pick: str = "") -> dict:
    """A person approves or denies an inbox item, with the reason; an approval performs the act asked for, as the
    person (a screen_conflict needs `pick`, the option that stands, from 1; a screen_maybe `pick` include or
    exclude). A person's act: agents cannot call it."""
    # Signed with the person key; the web server opens it with the `passphrase` the app's decide dialog
    # (ui/src/agents/personAct.ts) sends, and refuses (needs: "passphrase") without it.
    from ..agents.inbox import decide
    _signer_or_refuse(f"deciding inbox #{id}")
    try:
        it = decide(_store(ctx), int(id), bool(approve), reason, _person(), str(pick).strip() or None)
    except ValueError as e:
        raise ToolError(str(e), "", "inbox_list shows the pending items") from None
    return _item(it)
