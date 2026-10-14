"""The control plane's commands:

    openlab agents [ROOT] [--tree] [--all] [--json]         the agents, as a table or a tree
    openlab agent spawn PROMPT --role R [--study S] [--question Q] [--budget-usd X] [--minutes M] [--turns T]
                       [--cli claude|codex] [--model M] [--parent ID] [--cwd DIR] [--allowed-tools a,b]
                       [--permission-mode MODE]
    openlab agent status|log ID [-n N] [-f]
    openlab agent pause|resume|stop ID --reason R           (resume is a person's act)
    openlab inbox [--all] [--json]                          what needs a person
    openlab inbox approve|deny ITEM --reason R              a person's decision (signed: your passphrase)
    openlab inbox answer ITEM [--pick N ...] [--other TEXT] [-m MESSAGE]   answer an agent's question (unsigned advice)
    openlab inbox wait [--minutes M]                        until an agent asks something or a running agent ends
    openlab watch [ROOT] [--interval 2] [--once]            the live supervision screen for a tmux pane
    openlab hook EVENT                                      a Claude Code hook (stdin JSON)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import Optional

from .store import FINAL, Store, age, local_time, root_of


def _store(root) -> Store:
    return Store(root_of(root) if root else root_of())


def _bar(frac: float, width: int = 6) -> str:
    n = max(0, min(width, round(frac * width)))
    return "[" + "#" * n + "-" * (width - n) + "]"


def _money(r: dict) -> str:
    b = r["budget_usd"]
    return f"{'~' if r['usd_estimated'] else ''}${r['usd']:.2f}" + (f"/{b:g}" if b else "")


def _cut(text: str, n: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:max(0, n - 1)] + "…"


# --- openlab agents ---------------------------------------------------------------------------------------------
def agents_main(argv) -> int:
    p = argparse.ArgumentParser(prog="openlab agents", description="The agents in this workspace.")
    p.add_argument("root", nargs="?", default=None, help="the workspace (default: the one this folder is in)")
    p.add_argument("--tree", action="store_true", help="indent children under their parent")
    p.add_argument("--all", action="store_true", help="also agents that ended over an hour ago, "
                   "and sessions ended by silence")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    from .view import ago, snapshot, tokens
    snap = snapshot(_store(a.root), all=a.all or a.json)
    if a.json:
        print(json.dumps([{**{k: r["agent"][k] for k in ("id", "role", "kind", "parent", "status", "study",
                                                          "question", "budget", "spent", "last_event_at")},
                           "state": r["state"], "flags": r["flags"], "trials": r["trials"], "claims": r["claims"],
                           "refusals": r["refusals"]} for r in snap["rows"]], indent=1))
        return 0
    if not snap["rows"]:
        hidden = len(_store(a.root).agents())
        print(f"no live agents in {snap['root']} ({hidden} ended: `openlab agents --all` lists them)" if hidden else
              f"no agents in {snap['root']} yet: start `claude` here after `openlab init --claude`, or "
              "`openlab agent spawn PROMPT --role R`")
        return 0
    head = f"{'AGENT':<28} {'ROLE':<20} {'KIND':<15} {'STATE':<9} {'$ SPENT':<12} {'TOKENS':>7} {'TRIALS':>6} " \
           f"{'CLAIMS':>6} {'REFUSED':>7} {'LAST':>5}  STUDY / QUESTION"
    print(head)
    for r in snap["rows"]:
        ag = r["agent"]
        ind = ("  " * r["depth"] + ("└ " if r["depth"] else "")) if a.tree else ""
        where = " / ".join(x for x in (ag["study"], ag["question"]) if x)
        print(f"{ind + _cut(ag['id'], 28 - len(ind)):<28} {_cut(ag['role'], 20):<20} {ag['kind']:<15} {r['state']:<9} "
              f"{_money(r):<12} {tokens(r['tokens']):>7} {r['trials']:>6} {r['claims']:>6} {r['refusals']:>7} "
              f"{ago(r['idle_for']):>5}  {_cut(where, 60)}" + (f"  [{' '.join(r['flags'])}]" if r["flags"] else ""))
    return 0


# --- openlab agent ... -------------------------------------------------------------------------------------------
def _spawn(a) -> int:
    from .spawn import AgentError, spawn
    try:
        ag = spawn(root_of(a.root) if a.root else root_of(), a.prompt, role=a.role, study=a.study,
                   question=a.question, parent=a.parent, cli=a.cli, model=a.model,
                   budget={"usd": a.budget_usd, "minutes": a.minutes, "turns": a.turns}, cwd=a.cwd,
                   allowed_tools=a.allowed_tools.split(",") if a.allowed_tools else None,
                   permission_mode=a.permission_mode, decide=a.decide)
    except AgentError as e:
        print(f"error: {e.error}" + (f"\nmeaning: {e.meaning}" if e.meaning else "") + (f"\nfix: {e.try_}" if e.try_ else ""),
              file=sys.stderr)
        return 1
    b = ag["budget"]
    print(f"spawned {ag['id']} ({ag['cli']}{', ' + ag['model'] if ag['model'] else ''}, role {ag['role']}) "
          f"budget ${b.get('usd'):g}, {b.get('minutes'):g} min, "
          f"{b.get('turns')} turns\n  log: openlab agent log {ag['id']} -f\n  stop: openlab agent stop {ag['id']} "
          f"--reason ...")
    return 0


def _event_line(e: dict) -> str:
    t = local_time(e["at"])
    return f"{t} {e['agent']:<24} {e['kind']:<14} {e['summary']}"


def _log(a) -> int:
    st = _store(a.root)
    if st.get(a.id) is None:
        print(f"error: no agent {a.id}", file=sys.stderr)
        return 1
    evs = st.events(a.id, n=a.n)
    for e in evs:
        print(_event_line(e))
    if not a.follow:
        return 0
    last = evs[-1]["seq"] if evs else 0
    try:
        while True:
            for e in st.events(a.id, n=500, after=last):
                print(_event_line(e), flush=True)
                last = e["seq"]
            if (st.get(a.id) or {}).get("status") in FINAL and not st.events(a.id, n=1, after=last):
                return 0
            time.sleep(1)
    except KeyboardInterrupt:
        return 0


def _status(a) -> int:
    st = _store(a.root)
    ag = st.get(a.id)
    if ag is None:
        print(f"error: no agent {a.id}", file=sys.stderr)
        return 1
    print(json.dumps({**ag, "children": [c["id"] for c in st.children(a.id)]}, indent=1, default=str))
    return 0


def resume(st: Store, ag: dict, reason: str) -> dict:
    """Resume a paused agent: a person's act, signed with the person key (the passphrase), so an agent cannot
    release itself or a sibling. Raises person.NotAPerson."""
    from . import person
    with person.act(f"resume {ag['id']} (paused: {ag['control_reason'] or 'no reason given'})",
                    [f"reason: {reason}", f"role {ag['role'] or '-'}, status {ag['status']}"]) as signer:
        sig = {"key": signer.fingerprint,
               "value": signer.sign(f"openlab-resume|{ag['id']}|{reason}".encode("utf-8"))}
        return st.control(ag["id"], "resume", reason, signer.label, sig=sig)


def _control(a) -> int:
    from .person import NotAPerson
    from .identity import current
    st = _store(a.root)
    ag = st.get(a.id)
    if ag is None:
        print(f"error: no agent {a.id}", file=sys.stderr)
        return 1
    if a.verb == "resume":
        try:
            ag = resume(st, ag, a.reason)
        except NotAPerson as e:
            print(f"error: resuming an agent is a person's act: {e}", file=sys.stderr)
            return 1
    else:
        ag = st.control(a.id, a.verb, a.reason, current())
    note = {"pause": "its next tool call is refused" + (" and its process is stopped (SIGSTOP)"
                                                         if ag["kind"] in ("spawned", "codex") else ""),
            "resume": "it may continue", "stop": "it is told to end" + (" and its process is killed"
                                                                       if ag["kind"] in ("spawned", "codex") else "")}
    print(f"{a.verb}d {a.id}: {note[a.verb]}" if a.verb != "stop" else f"stopped {a.id}: {note['stop']}")
    return 0


def agent_main(argv) -> int:
    p = argparse.ArgumentParser(prog="openlab agent", description="Spawn, follow and control one agent.")
    p.add_argument("--root", default=None, help="the workspace (default: the one this folder is in)")
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("spawn", help="start an agent (claude -p or codex exec) under a supervisor")
    s.add_argument("prompt")
    s.add_argument("--role", default="agent")
    s.add_argument("--study", default=None, help="the study folder it works on (workspace-relative)")
    s.add_argument("--question", default=None)
    s.add_argument("--parent", default=None, help="the agent it reports to (its budget is a share of the parent's)")
    s.add_argument("--cli", default="claude", choices=("claude", "codex"))
    s.add_argument("--model", default=None)
    s.add_argument("--decide", choices=["self"], default="", help="the agent (and its children) make their own "
                   "choices instead of asking you questions; acts only you sign still come to your inbox")
    s.add_argument("--budget-usd", type=float, default=None)
    s.add_argument("--minutes", type=float, default=None)
    s.add_argument("--turns", type=int, default=None)
    s.add_argument("--cwd", default=None)
    s.add_argument("--allowed-tools", default=None, help="comma-separated (default: [agents] allowed_tools)")
    s.add_argument("--permission-mode", default=None, help="default: [agents] permission_mode (acceptEdits)")
    for verb in ("log", "status"):
        x = sub.add_parser(verb)
        x.add_argument("id")
        if verb == "log":
            x.add_argument("-n", type=int, default=40)
            x.add_argument("-f", "--follow", action="store_true")
    for verb, what in (("pause", "pause an agent (its next tool call is refused)"),
                       ("resume", "resume a paused agent: a person's act (asks for your passphrase)"),
                       ("stop", "stop an agent (its process is killed)")):
        x = sub.add_parser(verb, help=what, description=f"{what}. --reason is required: it goes on the record.")
        x.add_argument("id")
        x.add_argument("--reason", required=True, help="why (required; recorded)")
    x = sub.add_parser("tell", help="direct a working agent: the text reaches it at its next tool call",
                       description="Direct a working agent (a person, or a supervising agent): the text is recorded "
                       "and delivered into the agent's context at its next tool call.")
    x.add_argument("id")
    x.add_argument("text")
    x.add_argument("--person", action="store_true", help="sign it with your person key (it asks for the passphrase), so "
                   "the agent can trust it is yours; unsigned, it reaches the agent as advice")
    a = p.parse_args(argv)
    return {"spawn": _spawn, "log": _log, "status": _status, "tell": _tell}.get(a.verb, _control)(a)


def _tell(a) -> int:
    from .identity import current
    st = _store(a.root)
    ag = st.get(a.id)
    if ag is None:
        print(f"error: no agent {a.id}", file=sys.stderr)
        return 1
    if not a.text.strip():
        print("error: tell needs the text", file=sys.stderr)
        return 1
    by, sig = os.environ.get("OPENLAB_AGENT") or current() or "person", None
    if not st.is_lead(ag) and not any(x["id"] == by for x in st.ancestors(a.id)):
        lead = next((x["id"] for x in reversed(st.ancestors(a.id)) if st.is_lead(x)), ag["parent"])
        print(f"error: {a.id} works under {lead}: only its lead directs it\n"
              f"  try: openlab agent tell {lead} \"...\" (it can pass it on)", file=sys.stderr)
        return 1
    if a.person:                                    # signed: the agent can trust it is the person's direction
        from .person import NotAPerson, tell_message, unlock_interactive
        try:
            signer = unlock_interactive(f"direct agent {a.id}", [a.text.strip()])
        except NotAPerson as e:
            print(f"error: a signed direction is the person's: {e}", file=sys.stderr)
            return 1
        by, sig = signer.label, {"key": signer.fingerprint, "value": signer.sign(tell_message(a.id, a.text.strip()))}
    try:
        seq = st.tell(a.id, a.text.strip(), by, sig)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    live = ag["status"] in ("running", "idle", "waiting")
    print(f"told {a.id} (event {seq}): " + ("it reads it at its next tool call" if live else
                                           f"it is {ag['status']}; it reads it if it works again"))
    return 0


# --- openlab inbox ----------------------------------------------------------------------------------------------
def _item_line(it: dict, st: Optional[Store] = None) -> str:
    line = _item_text(it)
    if it["kind"] == "question":
        line += "".join(f"\n      {n}. {c}" for n, c in enumerate((it["args"] or {}).get("choices") or [], 1))
        if it["status"] == "pending":
            line += f"\n      answer with: openlab inbox answer {it['id']} " + _answer_usage(it)
    if st is not None:
        from .inbox import requester_note
        note = requester_note(st, it)
        line += f"\n      {note}" if note else ""
    return line


def _item_text(it: dict) -> str:
    a = dict(it["args"] or {})
    change = a.pop("change", None)
    if it["kind"] == "screen_conflict":
        from .inbox import options
        args = f"source={a.get('source')}; pick {options(it)}"
    elif it["kind"] == "question":
        r = it["result"] or {}
        args = ("pick any" if a.get("multi") else "pick one") + (" or answer in your own words" if a.get("other") else "")
        if it["status"] == "answered":
            args = "answer: " + "; ".join([*r.get("picks", []), *([f'"{r["other"]}"'] if r.get("other") else [])])
    elif it["kind"] == "screen_maybe":
        args = (f"source={a.get('key') or a.get('source')}; quote \"{a.get('quote') or ''}\"; "
                "pick include|exclude")
    else:
        args = ", ".join(f"{k}={v}" for k, v in a.items() if not (change and k in change))
    if change:
        args = ", ".join([*(f"{k} {o:g} -> {n:g}" for k, (o, n) in change.items()), *([args] if args else [])])
    return (f"#{it['id']:<4} {local_time(it['created'])} {it['status']:<9} {it['kind']:<9} {it['agent']:<24} "
            f"{it['study'] or '-':<14} "
            f"{it['reason']}" + (f"  ({args})" if args else "") + (f"  cost: {it['cost_note']}" if it["cost_note"] else "")
            + (f"  -> {it['decided_by']}: {it['decision_reason']}" if it["decided_by"] else ""))


def _answer_usage(it: dict) -> str:
    a = it["args"] or {}
    return ("--pick N [--pick N ...]" if a.get("multi") else "--pick N") + (" [--other TEXT]" if a.get("other") else "") \
        + " [-m MESSAGE]"


def inbox_main(argv) -> int:
    p = argparse.ArgumentParser(prog="openlab inbox", description="Requests from agents that need a person.")
    p.add_argument("--root", default=None)
    p.add_argument("--all", action="store_true", help="also decided items")
    p.add_argument("--json", action="store_true")
    sub = p.add_subparsers(dest="verb")
    w = sub.add_parser("wait", help="block until an agent asks the person something (a new inbox item: a "
                                    "request or a question) or a running agent ends; for a supervisor, person or agent, that steps away")
    w.add_argument("--minutes", type=float, default=0, help="give up after this long (exit 2); 0: no limit")
    for verb in ("approve", "deny"):
        x = sub.add_parser(verb, help=f"{verb} an item: a person's act (shows its effect, then asks for your "
                                      "passphrase)")
        x.add_argument("item", type=int)
        x.add_argument("--reason", required=True, help="why (required; recorded)")
        if verb == "approve":
            x.add_argument("--wait", action="store_true", help="run a long act (a holdout look) in this terminal "
                           "instead of the background")
            x.add_argument("--pick", default=None, help="a screen_conflict: the option that stands (1..n); a "
                           "screen_maybe: include or exclude; a budget request: your own new limit instead of the "
                           "one asked for")
            x.add_argument("--no-resume", action="store_true", help="when the requester has ended, do not resume "
                           "its work (by default the replication's builder or the proposal's execution goes on, "
                           "within the budget left)")
    x = sub.add_parser("answer", help="answer an agent's question: the choices you pick (numbered as `openlab inbox` "
                                      "lists them) and/or your own words; advice to the agent, not a signed act")
    x.add_argument("item", type=int)
    x.add_argument("--pick", action="append", default=[], metavar="N", help="a choice, from 1 (repeat it on a "
                   "question that takes several)")
    x.add_argument("--other", default="", metavar="TEXT", help="your own answer (when the question allows one)")
    x.add_argument("-m", "--message", default="", help="a message to the agent with the answer")
    a = p.parse_args(argv)
    st = _store(a.root)
    if a.verb == "wait":
        return _inbox_wait(st, a.minutes)
    if a.verb == "answer":
        return answer_cli(st, a.item, a.pick, a.other, a.message)
    if not a.verb:
        items = st.inbox(None if a.all else "pending")
        if a.json:
            print(json.dumps(items, indent=1))
        elif not items:
            print("the inbox is empty")
        else:
            print("\n".join(_item_line(i, st) for i in items))
        return 0
    return decide_cli(st, a.item, a.verb == "approve", a.reason, wait=getattr(a, "wait", False),
                      pick=getattr(a, "pick", None), resume=not getattr(a, "no_resume", True))


def _inbox_wait(st: Store, minutes: float = 0) -> int:
    """Wait for a new pending item or the end of an agent live now; print them (exit 0), or exit 2 on the limit."""
    from .store import LIVE
    seen = {i["id"] for i in st.inbox("pending")}
    live = {a["id"] for a in st.agents() if a["status"] in LIVE}
    t0 = time.time()
    while True:
        new = [i for i in st.inbox("pending") if i["id"] not in seen]
        ended = [a for a in st.agents() if a["id"] in live and a["status"] not in LIVE]
        if new or ended:
            for a in ended:
                print(f"ended  {a['id']} ({a['role'] or a['kind']}) {a['status']}")
            for i in new:
                print(_item_line(i, st))
            return 0
        if minutes > 0 and time.time() - t0 > minutes * 60:
            print(f"nothing new in {minutes:g} min", file=sys.stderr)
            return 2
        time.sleep(5)


def answer_cli(st: Store, item: int, picks, other: str = "", message: str = "") -> int:
    """The person's answer to an agent's question (openlab.agents.inbox.answer): `picks` are choice numbers from 1.
    Not signed: an answer is advice, and the agent decides what to do with it. An agent cannot answer (its own or
    another's question): the hooks refuse the command, and this refuses an agent's identity."""
    from .identity import UNATTRIBUTED, current, person_label
    from .inbox import answer

    def fail(msg: str) -> int:
        print(f"error: {msg}", file=sys.stderr)
        return 1
    who = current()
    if not (who.startswith("person:") or who == UNATTRIBUTED):
        return fail(f"answering an agent's question is the person's: {who} cannot (ask with agent_request instead)")
    it = st.item(item)
    if it is None:
        return fail(f"no inbox item #{item}")
    if it["kind"] != "question":
        return fail(f"#{item} is a {it['kind']} request, not a question: `openlab inbox approve|deny {item} --reason ...`")
    if it["status"] != "pending":
        return fail(f"#{item} is already {it['status']}")
    choices = (it["args"] or {}).get("choices") or []
    numbered = "; ".join(f"{n}. {c}" for n, c in enumerate(choices, 1))
    chosen = []
    for p in picks or []:
        try:
            n = int(str(p).strip())
        except ValueError:
            n = 0
        if not 1 <= n <= len(choices):
            return fail(f"--pick {p} is not one of the choices 1..{len(choices)} ({numbered})")
        if choices[n - 1] not in chosen:
            chosen.append(choices[n - 1])
    if not chosen and not str(other or "").strip():
        return fail(f"#{item} needs an answer: {_answer_usage(it)} ({numbered})")
    try:
        out = answer(st, item, chosen, other, message, by=who if who.startswith("person:") else person_label())
    except ValueError as e:
        return fail(f"{e}: {_answer_usage(it)}")
    print(_item_text(out))
    ag = st.get(out["agent"]) if out.get("agent") else None
    if ag and ag["status"] in FINAL:
        print(f"the asker ({ag['id']}) has ended: the answer stays on #{item} for whoever goes on (agent_inbox_item)")
    return 0


def decide_cli(st: Store, item: int, approve: bool, reason: str, wait: bool = True, pick=None,
               resume: bool = False) -> int:
    """A person's decision on an inbox item: shows exactly what approving does, asks for the passphrase, then does
    it. A long act (a holdout look) runs in the background unless `wait`; its result is posted to the item. With
    `resume`, an approval whose requester has ended resumes its work (openlab.agents.inbox.resume_argv)."""
    from . import person
    from .inbox import LONG, decide, describe, finish, start
    it = st.item(item)
    if it is None:
        print(f"error: no inbox item #{item}", file=sys.stderr)
        return 1
    if it["status"] != "pending":
        print(f"error: #{item} is already {it['status']}", file=sys.stderr)
        return 1
    if it["kind"] == "question":
        print(f"error: #{item} is a question, not a request to {'approve' if approve else 'deny'}: answer it with "
              f"`openlab inbox answer {item} {_answer_usage(it)}`", file=sys.stderr)
        return 1
    try:
        what, details = describe(st, it, approve)
        if pick is not None and it["kind"] == "screen_conflict":
            what = f"{what} -> option {pick} stands"
        elif it["kind"] == "screen_maybe" and approve:
            if str(pick or "").lower() not in ("include", "exclude"):
                print(f"error: #{item} is a maybe a person decides: `openlab inbox approve {item} --pick include` "
                      "(or exclude) --reason ...", file=sys.stderr)
                return 1
            what = f"{what} -> {str(pick).lower()}"
        with person.act(what, details):
            if approve and it["kind"] in LONG and not wait and hasattr(os, "fork"):
                out = start(st, item, reason)
                pid = os.fork()
                if pid == 0:                                        # the child runs the act, signed, and posts it
                    code = 0
                    try:
                        os.setsid()
                        finish(st, item, reason)
                    except BaseException:                           # noqa: BLE001
                        code = 1
                    finally:
                        os._exit(code)
                print(f"#{item} approved by {out['decided_by']}; running the {it['kind']} in the background "
                      f"(pid {pid}): `openlab inbox --all` shows its result, and the agent reads it with "
                      f"openlab_agent_inbox_item({item})")
                return 0
            if approve and it["kind"] in LONG:
                print(f"running the {it['kind']} for #{item}...", file=sys.stderr, flush=True)
            out = decide(st, item, approve, reason, pick=pick)
    except person.NotAPerson as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except Exception as e:                                          # noqa: BLE001 -- a study refusal, a bad item
        err = getattr(e, "error", None) or str(e)
        print(f"error: {err}" + (f"\nmeaning: {e.meaning}" if getattr(e, "meaning", "") else ""), file=sys.stderr)
        return 1
    print(f"#{item} {out['status']} by {out['decided_by']}" + (f": {json.dumps(out['result'])}" if out["result"] else ""))
    from .inbox import requester_note, resume_argv
    argv = resume_argv(st, out) if resume and out["status"] == "approved" else []
    if argv:
        print(f"the requester has ended; resuming: openlab {' '.join(argv)}", flush=True)
        from ..__main__ import main as openlab_main
        return openlab_main(argv)
    note = requester_note(st, out)
    if note:
        print(note)
    return 0


# --- openlab watch ----------------------------------------------------------------------------------------------
STATE_COLOUR = {"working": "32", "idle": "37", "waiting": "33;1", "stuck": "31;1", "paused": "35", "done": "90",
                "failed": "31", "stopped": "90", "lost": "31", "ended": "90"}


def frame(st: Store, width: int = 120, selected: int = -1, colour: bool = False, studies=None) -> tuple[str, list]:
    from .view import ago, snapshot, tokens
    snap = snapshot(st, all=False, studies=studies)
    c = snap["counts"]

    def paint(text: str, code: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if colour and code else text
    live = sum(c[k] for k in ("working", "idle", "stuck", "waiting"))
    lines = [paint(f"openlab watch  {_cut(snap['root'], 40)}", "1") +
             f"   agents: {live} live ({c['working']} working, {c['stuck']} stuck, {c['waiting']} waiting), "
             f"{c['paused']} paused   {'~' if snap['usd_estimated'] else ''}${snap['usd']:.2f} spent   inbox {snap['pending']}   {time.strftime('%H:%M:%S')}",
             ""]
    fw = max([len(" ".join(r["flags"])) for r in snap["rows"]] + [5])
    qw = max(10, width - 136 - fw)
    lines.append(f"  {'STATE':<8} {'AGENT':<24} {'ROLE':<20} {'$ / BUDGET':<20} {'TOKENS':>6} {'LAST ACTION':<32} "
                 f"{'FLAGS':<{fw}} QUESTION")
    for i, r in enumerate(snap["rows"]):
        ag = r["agent"]
        ind = "  " * r["depth"] + ("└ " if r["depth"] else "")
        last = r["last"]
        if last:
            what = last["summary"] if last["kind"] == "tool" else f"{last['kind']}: {last['summary']}"
            act = f"{ago(age(last['at']))} " + what.replace("mcp__openlab__openlab_", "")
        else:
            act = "-"
        money = f"{_money(r)} {_bar(r['frac'])}" if r["budget_usd"] else _money(r)
        state = paint(f"{r['state']:<8}", STATE_COLOUR.get(r["state"], ""))
        flags = paint(f"{' '.join(r['flags']):<{fw}}", "31;1" if r["flags"] else "")
        row = (f"{'>' if i == selected else ' '} {state} "
               f"{ind + _cut(ag['id'], 24 - len(ind)):<24} {_cut(ag['role'], 20):<20} {_cut(money, 20):<20} "
               f"{tokens(r['tokens']):>6} {_cut(act, 32):<32} {flags} {_cut(ag['question'] or ag['study'], qw)}")
        lines.append(row)
        if r["last_refusal"] and r["state"] not in FINAL:
            e = r["last_refusal"]
            label = {"bypass_attempt": "bypass attempt", "blocked": "blocked"}.get(e["kind"], "refusal")
            lines.append(paint(f"      ! {label} {ago(age(e['at']))} ago: {_cut(e['summary'], width - 30)}", "33"))
        if r["dead_end"]:
            who, text = r["dead_end"]
            lines.append(paint(f"      ! matches a dead end by {who}: {_cut(text, width - 40)}", "33"))
    if not snap["rows"]:
        lines.append("  no agents yet: start `claude` here after `openlab init --claude`, or `openlab agent spawn`")
    q = snap.get("queue") or {}
    lines += ["", f"inbox: {snap['pending']} pending" + ("  (openlab inbox)" if snap["pending"] else "")
              + (f"    runs on this machine: {q['running']} running, {q['waiting']} waiting  (openlab box queue)"
                 if q.get("running") or q.get("waiting") else "")
              + "    keys: q quit  j/k select  p pause/resume  s stop  a approve"]
    return "\n".join(lines), snap["rows"]


def watch_main(argv) -> int:
    p = argparse.ArgumentParser(prog="openlab watch", description="Live supervision of the workspace's agents.")
    p.add_argument("root", nargs="?", default=None)
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--once", action="store_true", help="print one frame and exit (no colour, no keys)")
    a = p.parse_args(argv)
    st = _store(a.root)
    try:
        from ..study.workspace import studies as find
        studies = find(st.root)
    except Exception:                                               # noqa: BLE001
        studies = []
    if a.once or not sys.stdout.isatty():
        print(frame(st, width=shutil.get_terminal_size((140, 40)).columns, studies=studies)[0])
        return 0
    return _live(st, a.interval, studies)


def _live(st: Store, interval: float, studies) -> int:
    import select
    import termios
    import tty
    fd = sys.stdin.fileno() if sys.stdin.isatty() else None
    old = termios.tcgetattr(fd) if fd is not None else None
    sel, rows, last_scan = 0, [], time.monotonic()
    try:
        if fd is not None:
            tty.setcbreak(fd)
        sys.stdout.write("\x1b[?25l")
        while True:
            if time.monotonic() - last_scan > 30:
                from ..study.workspace import studies as find
                studies, last_scan = find(st.root), time.monotonic()
            size = shutil.get_terminal_size((140, 40))
            text, rows = frame(st, width=size.columns, selected=sel, colour=True, studies=studies)
            out = [_cut_ansi(line, size.columns) for line in text.split("\n")][:size.lines - 1]
            sys.stdout.write("\x1b[H\x1b[2J" + "\n".join(out))
            sys.stdout.flush()
            if fd is None:
                time.sleep(interval)
                continue
            ready, _, _ = select.select([sys.stdin], [], [], interval)
            if not ready:
                continue
            key = sys.stdin.read(1)
            if key == "q":
                return 0
            if key in ("j", "B") and rows:
                sel = min(sel + 1, len(rows) - 1)
            elif key in ("k", "A") and rows:
                sel = max(sel - 1, 0)
            elif key in ("p", "s", "a"):
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
                sys.stdout.write("\x1b[?25h\n")
                try:
                    _key_act(st, key, rows[sel]["agent"] if rows and sel < len(rows) else None)
                finally:
                    tty.setcbreak(fd)
                    sys.stdout.write("\x1b[?25l")
    except KeyboardInterrupt:
        return 0
    finally:
        if old is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write("\x1b[?25h\n")


def _key_act(st: Store, key: str, ag) -> None:
    from .identity import current
    try:
        if key in ("p", "s") and ag is not None:
            verb = "stop" if key == "s" else ("resume" if ag["control"] == "pause" or ag["status"] == "paused"
                                               else "pause")
            reason = input(f"{verb} {ag['id']}: reason (empty cancels): ").strip()
            if reason and verb == "resume":
                resume(st, ag, reason)                              # a person's act: the passphrase
            elif reason:
                st.control(ag["id"], verb, reason, current())
        elif key == "a":
            items = st.inbox("pending")
            if not items:
                input("the inbox is empty (enter) ")
                return
            print("\n".join(_item_line(i, st) for i in items))
            pick = input("item to approve, or -N to deny (empty cancels): ").strip()
            if pick:
                reason = input("reason: ").strip()
                if reason:
                    decide_cli(st, abs(int(pick)), not pick.startswith("-"), reason, wait=False)
                    input("(enter) ")
    except (ValueError, EOFError) as e:
        if str(e):
            print(f"error: {e}")
            time.sleep(1.5)
    except Exception as e:                                          # noqa: BLE001 -- a person act refused
        print(f"error: {e}")
        time.sleep(1.5)


def _cut_ansi(line: str, width: int) -> str:
    """Cut a line with colour codes to `width` visible characters."""
    out, n, i = [], 0, 0
    while i < len(line):
        if line[i] == "\x1b":
            j = line.find("m", i)
            if j < 0:
                break
            out.append(line[i:j + 1])
            i = j + 1
            continue
        if n >= width:
            i += 1
            continue
        out.append(line[i])
        n += 1
        i += 1
    return "".join(out)


def hook_main(argv) -> int:
    from .hooks import main
    return main(argv)


# --- openlab person -----------------------------------------------------------------------------------------------
def person_main(argv) -> int:
    p = argparse.ArgumentParser(prog="openlab person", description=(
        "The person key: a passphrase-protected Ed25519 key that signs every person act (a holdout look, an "
        "approval, a person's amendment or screening, an inbox decision, resuming a paused agent). An agent on this "
        "account can read the encrypted file but cannot sign without the passphrase."))
    sub = p.add_subparsers(dest="verb", required=True)
    i = sub.add_parser("init", help="make the key (asks for your handle and a passphrase)")
    i.add_argument("--handle", default=None, help="your name or handle (default: asked, then your login)")
    sub.add_parser("show", help="the key's fingerprint, and what this workspace pins")
    sub.add_parser("rotate", help="a new key, endorsed by the old one (asks for both passphrases)")
    pn = sub.add_parser("pin", help="pin the key in this workspace (no folder) or in a study (a person act)")
    pn.add_argument("folder", nargs="?", default=None)
    pn.add_argument("--root", default=None, help="the workspace (default: the one this folder is in)")
    ac = sub.add_parser("act", help="run a person act on another machine (over ssh) that keeps the record, signed "
                        "here: the key never leaves this machine. e.g. openlab person act box -- study holdout demo "
                        "--reason final")
    ac.add_argument("host", help="the ssh host (an alias in ~/.ssh/config)")
    ac.add_argument("--openlab", default="openlab", help="the openlab command there (default: openlab on its PATH)")
    ac.add_argument("--cd", default="", help="the directory there to run it in (the workspace)")
    ac.add_argument("args", nargs="*", help="-- then the openlab command, as you would type it there")
    rest = []
    if argv[:1] == ["act"] and "--" in argv:            # the remote command is everything after --, options and all
        i = argv.index("--")
        argv, rest = argv[:i], argv[i + 1:]
    a = p.parse_args(argv)
    from . import person
    if a.verb == "act":
        from .remote import act_on
        args = [*a.args, *rest]
        if not args:
            p.error("person act needs the openlab command to run there, after --")
        return act_on(a.host, args, openlab=a.openlab, cd=a.cd)
    try:
        if a.verb == "init":
            if person.exists():
                print(f"error: a person key exists ({person.info()['fingerprint']}); `openlab person rotate` "
                      "replaces it", file=sys.stderr)
                return 1
            import getpass as _gp
            handle = a.handle or person._line(f"your handle [{_gp.getuser()}]: ", _gp.getuser())
            pw = person._ask("new passphrase (8+ characters): ")
            if pw != person._ask("again: "):
                print("error: the passphrases differ; nothing was written", file=sys.stderr)
                return 1
            k = person.create(handle, pw)
            print(f"person key for person:{k['handle']}\n  fingerprint {k['fingerprint']}\n  private (encrypted) "
                  f"{person.key_path()}\n  public {person.pub_path()}\nkeep the fingerprint somewhere an agent cannot "
                  "write (a commit, a note); pin it in a workspace with `openlab person pin`")
            return 0
        if a.verb == "show":
            k = person.info()
            if k is None:
                print("no person key here: `openlab person init` makes one")
                return 1
            ws = person.pin_state(os.getcwd())
            pin = ws["pin"]
            print(f"person:{k['handle']}  {k['fingerprint']}  (made {local_time(k['created'])})\n"
                  f"  workspace pin: {pin or 'none (openlab person pin)'}"
                  + ("" if not pin or pin == k["fingerprint"] else "   MISMATCH: this workspace pins another key")
                  + (f"\n  pin chain: {' -> '.join(ws['chain'])}" if len(ws["chain"]) > 1 else "")
                  + (f"\n  TAMPERED: {ws['conflict']}" if ws["conflict"] else "")
                  + "".join(f"\n  warning: control store event {x['seq']}: {x['why']}" for x in ws["invalid"])
                  + "\ncompare these with the fingerprint you keep where no agent can write")
            return 0 if not ws["conflict"] else 1
        if a.verb == "rotate":
            old = person._ask("current passphrase: ")
            new = person._ask("new passphrase (8+ characters): ")
            if new != person._ask("again: "):
                print("error: the passphrases differ; nothing was changed", file=sys.stderr)
                return 1
            pinned = person.workspace_pin(os.getcwd())
            k = person.rotate(old, new)
            print(f"rotated {k['rotated_from']} -> {k['fingerprint']}; studies pinned to the old key take the new "
                  "one, endorsed, at your next act there")
            if pinned == k["rotated_from"]:
                person.pin_workspace(os.getcwd(), person.unlock(new))
                print("  this workspace's pin is updated (a rotation endorsed by the old key)")
            elif pinned:
                print(f"  this workspace pins {pinned}, not the key rotated: it is unchanged")
            return 0
        if a.verb == "pin":
            if a.folder:
                from ..study import open as open_study
                s = open_study(a.folder)
                with person.act(f"pin your person key in study {s.spec.name}", [f"folder {s.folder}"]) as signer:
                    s.record._pin(signer)
                print(f"{s.spec.name} pins {signer.fingerprint}")
                return 0
            root = root_of(a.root) if a.root else root_of()
            with person.act(f"pin your person key in the workspace {root}") as signer:
                out = person.pin_workspace(root, signer)
            print(f"{out['workspace']} pins {out['fingerprint']}"
                  + (f" (rotated from {out['rotated_from']}, endorsed by it)" if out.get("rotated_from") else "")
                  + ("" if out["toml"] else " (control store only: no openlab.toml)"))
            return 0
    except person.NotAPerson as e:
        print(f"error: {getattr(e, 'error', None) or e}" + (f"\nmeaning: {e.meaning}" if getattr(e, "meaning", "")
                                                             else "") + (f"\nfix: {e.fix}" if getattr(e, "fix", "")
                                                                         else ""), file=sys.stderr)
        return 1
    except Exception as e:                                          # noqa: BLE001 -- a study refusal
        print(f"error: {getattr(e, 'error', None) or e}" + (f"\nmeaning: {e.meaning}" if getattr(e, "meaning", "")
                                                             else ""), file=sys.stderr)
        return 1
    return 2


COMMANDS = {"agents": agents_main, "agent": agent_main, "inbox": inbox_main, "watch": watch_main, "hook": hook_main,
            "person": person_main}


def main(argv) -> int:
    """`openlab agents|agent|inbox|watch|hook ...`: argv[0] is the subcommand."""
    return COMMANDS[argv[0]](list(argv[1:]))
