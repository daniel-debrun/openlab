"""Deciding an inbox item: the person's approval performs the act the agent asked for, as the person, and both the
act and the decision are recorded (the act on the study's record, the decision on the control store's events).
A decision is a person act: it is signed by the person key (openlab.agents.person.act), and `decide` refuses
without one.

* holdout: a holdout look on the study (`Study.holdout(reason, person=True)`); the result is the look's nodes.
* amend: a person's amendment of the study's edited study.toml (`Study.amend(reason, person=True)`); with args
  {"study.toml": {section: {key: value}}} approving writes those fields first (the agent need not be running).
* approval: `Study.approve(on, decision, reason, person=True)` with args {on, decision} (decision default approve).
* screen: a person's screening decision, args {source, decision, criterion, rationale, quote}.
* screen_conflict: agents disagree on a source (filed by openlab, args {source, options}); approving needs `pick`,
  the option that stands, which is recorded as the person's screen.
* screen_maybe: an agent marked a source maybe (filed by openlab, args {source, key, title, screen, criterion, quote,
  rationale}); approving needs `pick` include or exclude, recorded as the person's screen (under the agent's
  criterion when it is of that kind, else the review's first criterion of that kind).
* budget: the agent's budget raised to args {usd, minutes, turns} (the new limits; at least one, each above the
  current one). The request records the change ({"usd": [0.5, 1.0]}), and the approval prompt shows it. An agent at
  a limit is paused, not ended, and openlab files this request for it (`hold`, args.hold true): approving raises the
  limit (`pick`: another new limit than the one proposed) and resumes it; denying gives it a short grace to wrap up
  and resumes it, told to finish.
* fresh / continues: the same-data guard (openlab.study.ledger) refused a run; args {why} / {continues: "../demo"}.
  Approving writes `[study] fresh` / `continues` into study.toml and records it as a person's amendment (or the
  person's seal, before the first run).
* instrument: registering an instrument version (openlab.study.instruments), args {name, entry, source, runtime,
  python, node, binaries, env_files, env_cmds, env, capabilities, secrets, config, outputs, state, timeout, include,
  exclude, redact}: approving fingerprints it again, shows exactly what the version holds, and registers it.
* instrument_run: one run of a live-order instrument, args {instrument: NAME@VERSION, args: [...], study, note}:
  approving signs the authority for that run and runs it, as the person.
* other: the decision only.
A request that carries none of what its kind needs is refused when it is filed (`check_request`), so an approval
can never be a no-op. A denial does nothing but record why; the agent reads it with agent_inbox_item.
"""

from __future__ import annotations

import contextvars
from pathlib import Path
from typing import Optional

from .identity import acting_as
from .store import Store, canonical

BUDGET_KEYS = ("usd", "minutes", "turns")
#: the inbox item being decided now, so the act it performs does not settle that same item a second time
DECIDING: contextvars.ContextVar = contextvars.ContextVar("openlab_deciding", default=None)
#: kinds whose approval runs long (the holdout run): `openlab inbox approve` runs them in the background
LONG = ("holdout", "instrument_run")
#: what an instrument request's args may carry: the keyword arguments of openlab.study.instruments.prepare
INSTRUMENT_ARGS = ("name", "entry", "source", "runtime", "python", "node", "binaries", "env_files", "env_cmds", "env",
                   "capabilities", "secrets", "config", "outputs", "state", "timeout", "include", "exclude", "redact",
                   "version_probe")


class RequestError(ValueError):
    def __init__(self, error: str, try_: str = ""):
        super().__init__(error)
        self.error, self.try_ = error, try_


def _num(v) -> Optional[float]:
    try:
        return float(v) if v is not None and not isinstance(v, bool) else None
    except (TypeError, ValueError):
        return None


def check_request(store: Store, agent: str, kind: str, args: Optional[dict], study: str = "") -> dict:
    """The request's args made exact, or RequestError: a budget request names the new limits (and gets the change
    recorded), an approval names what it is on, a screening the source and decision, a study act its study."""
    args = dict(args or {})
    if kind == "budget":
        bad = [k for k in args if k not in BUDGET_KEYS]
        if bad:
            raise RequestError(f"a budget request has no {', '.join(bad)}",
                               "an agent's budget is usd, minutes or turns (the new limits); a study's trials or "
                               "looks are a person's amendment: file kind=amend with the edited study.toml")
        a = store.get(agent) or {"budget": {}}
        change = {}
        for k in BUDGET_KEYS:
            if k in args:
                new, old = _num(args[k]), _num((a["budget"] or {}).get(k)) or 0.0
                if new is None or new <= old:
                    raise RequestError(f"budget {k} {args[k]!r} is not above the current {old:g}",
                                       f'ask for the new limit, e.g. args {{"{k}": {round(old * 2, 4) or 1:g}}}')
                change[k] = [old, new]
        if not change:
            raise RequestError("a budget request names the new limits",
                               'e.g. args {"usd": 1.0} (the new total); nothing would change otherwise')
        return {**{k: v[1] for k, v in change.items()}, "change": change}
    if kind == "approval":
        if not args.get("on"):
            raise RequestError("an approval request names what it is on", 'args {"on": "<claim id>", "decision": '
                               '"approve"|"reject"|"waive"}')
        args.setdefault("decision", "approve")
    if kind == "screen" and not (args.get("source") and args.get("decision")):
        raise RequestError("a screening request names the source and the decision",
                           'args {"source": "<id>", "decision": "include"|"exclude", "criterion": "...", '
                           '"rationale": "..."}')
    if kind in ("screen_maybe", "screen_conflict"):
        raise RequestError(f"a {kind} item is filed by openlab when you screen",
                           "mark the source maybe (or disagree) with openlab_study_screen; it reaches the person")
    if kind == "fresh" and not str(args.get("why") or "").strip():
        raise RequestError("a fresh request says why this is a new question", 'args {"why": "<what is new about '
                           'it>"}; the person decides whether it gets a trial budget and a holdout of its own')
    if kind == "continues" and not str(args.get("continues") or "").strip():
        raise RequestError("a continues request names the study whose ledger to share",
                           'args {"continues": "../demo"} (the path the refusal named)')
    if kind == "amend" and _fields(args) is None and ("study.toml" in args or "fields" in args):
        raise RequestError("an amend request's study.toml is the fields it sets",
                           'args {"study.toml": {"split": {"scheme": "kfold", "k": 5}}}: section, key, value')
    if kind in ("holdout", "amend", "approval", "screen", "fresh", "continues") and not study:
        raise RequestError(f"a {kind} request names its study", "pass study=<folder>")
    if kind == "instrument":
        bad = [k for k in args if k not in INSTRUMENT_ARGS]
        if bad or not args.get("name") or not args.get("entry"):
            raise RequestError("an instrument request carries the exact registration",
                               'args {"name": "ibkr_paper", "entry": "python -m ibkr_paper", "source": '
                               '"instruments/ibkr_paper", "capabilities": "read,paper-order", ...}'
                               + (f"; unknown: {', '.join(bad)}" if bad else ""))
    if kind == "instrument_run":
        if not str(args.get("instrument") or "").strip() or not isinstance(args.get("args", []), list):
            raise RequestError("an instrument_run request names the version and its exact args",
                               'args {"instrument": "NAME@VERSION", "args": ["place", "--symbol", "F"], '
                               '"study": "<folder>"}')
    if kind == "compute":
        return compute_request(args)
    if kind == "question":
        return question(args)
    return args


#: what a denied hold leaves the agent to wrap up with, past what it has used
GRACE = {"minutes": 10.0, "usd": 0.1, "turns": 10}           # usd: a share of the limit
UNITS = {"minutes": "minutes", "usd": "$", "turns": "turns"}


def _limit_text(key: str, n) -> str:
    return f"${float(n):g}" if key == "usd" else f"{float(n):g} {UNITS[key]}"


def hold(store: Store, agent: str, key: str, used: str) -> Optional[int]:
    """`agent` reached its `key` limit: pause it (openlab holds it, as a person's pause would) and file a budget request
    for it, proposing twice the limit, unless one is pending already. Returns the inbox item (None when pending)."""
    a = store.get(agent)
    if a is None:
        return None
    store.control(agent, "pause", f"budget: {used}; waiting for the person to raise its limit", "openlab")
    store.event(agent, "budget_hold", summary=f"paused at its limit: {used}", payload={"key": key, "used": used})
    if any((i["args"] or {}).get("hold") for i in store.inbox("pending", agent=agent) if i["kind"] == "budget"):
        return None
    old = float((a["budget"] or {}).get(key) or 0)
    new = round(old * 2, 4) if key != "turns" else int(old * 2)
    what = {"minutes": "more time", "usd": "more money", "turns": "more turns"}[key]
    return store.request(agent, "budget", f"{agent} reached its limit ({used}) and is paused, not ended: it asks for "
                         f"{what}, {_limit_text(key, old)} -> {_limit_text(key, new)}. Approve to raise it and resume; "
                         "deny to let it wrap up.", study=a["study"] or "",
                         args={key: new, "change": {key: [old, new]}, "hold": True})


def _held_for_budget(a: Optional[dict]) -> bool:
    return bool(a) and a["control"] == "pause" and str(a["control_reason"] or "").startswith("budget:")


def question(args: dict) -> dict:
    """A question for the person: 2-8 choices, one (multi false) or any (multi true), and `other` when the person
    may type an answer of their own. Answering it is advice, not a person's act: it is not signed."""
    choices = [str(c).strip() for c in (args.get("choices") or []) if str(c).strip()]
    if not 2 <= len(choices) <= 8:
        raise RequestError("a question offers 2 to 8 choices",
                           'args {"choices": ["3 seeds", "5 seeds"], "multi": false, "other": true}')
    return {"choices": [c[:200] for c in choices], "multi": bool(args.get("multi")), "other": bool(args.get("other", True))}


def answer(store: Store, id: int, picks, other: str = "", message: str = "", by: str = "person") -> dict:
    """The person's answer to an agent's question: the choices picked and/or their own words, with a message."""
    item = store.item(id)
    if item is None:
        raise ValueError(f"no inbox item #{id}")
    if item["kind"] != "question":
        raise ValueError(f"#{id} is a {item['kind']} request, not a question: approve or deny it")
    if item["status"] != "pending":
        raise ValueError(f"#{id} is already {item['status']}")
    choices = (item["args"] or {}).get("choices") or []
    picks = [str(p) for p in (picks or [])]
    unknown = [p for p in picks if p not in choices]
    if unknown:
        raise ValueError(f"{unknown[0]!r} is not one of the choices")
    other = str(other or "").strip()
    if other and not (item["args"] or {}).get("other"):
        raise ValueError("this question takes only its choices")
    if not picks and not other:
        raise ValueError("pick a choice or type an answer")
    if len(picks) > 1 and not (item["args"] or {}).get("multi"):
        raise ValueError("this question takes one choice")
    return store.decide(id, "answered", by, str(message or "").strip(), {"picks": picks, "other": other})


COMPUTE_KEYS = ("cores", "memory_gb", "gpus", "hours")


def compute_request(args: dict) -> dict:
    """A request for another machine: what one would need (cores, memory_gb, gpus, hours; at least one), with what
    this machine has, its queue and its declared availability attached by openlab, not by the agent. Openlab never
    rents, pays for or stops a machine: the person answers by adding one (or declines)."""
    bad = [k for k in args if k not in COMPUTE_KEYS]
    need = {k: _num(args.get(k)) for k in COMPUTE_KEYS if args.get(k) is not None}
    if bad or not need or any(v is None or v <= 0 for v in need.values()):
        raise RequestError("a compute request says what another machine would need",
                           'args {"gpus": 1, "memory_gb": 64, "cores": 16, "hours": 6} (any of them, each above 0); '
                           "openlab adds what this machine has and its queue")
    try:
        from ..study import admission, machine
        sched, queue = admission.schedule(), admission.tickets()
        here = {**{k: v for k, v in machine.describe().items() if k in ("cores", "memory_gb")},
                "gpus": len(machine.gpus()), "available_until": sched["available_until"],
                "running": sum(1 for t in queue if t["state"] == "running"),
                "waiting": sum(1 for t in queue if t["state"] == "waiting")}
    except Exception as e:                                          # noqa: BLE001 -- the request still reaches the person
        here = {"error": f"{type(e).__name__}: {e}"}
    return {**need, "machine": here}


def _fields(args: dict) -> Optional[dict]:
    """The study.toml fields an amend request sets ({section: {key: value}}), or None (the agent edited the file)."""
    f = args.get("study.toml", args.get("fields"))
    return f if isinstance(f, dict) and f and all(isinstance(v, dict) and v for v in f.values()) else None


def _study(store: Store, item: dict):
    from .. import study
    if not item["study"]:
        raise ValueError(f"#{item['id']} names no study")
    folder = Path(item["study"])
    folder = folder if folder.is_absolute() else store.root / folder
    if not folder.is_dir():
        a = store.get(item["agent"]) or {}
        alt = Path(a.get("cwd") or store.root) / item["study"]
        folder = alt if alt.is_dir() else folder
    return study.open(folder)


def describe(store: Store, item: dict, approve: bool = True) -> tuple[str, list]:
    """(what the person signs, the details they read first): the exact effect of approving the item."""
    kind, args = item["kind"], item["args"] or {}
    head = f"{'approve' if approve else 'deny'} inbox #{item['id']} from {item['agent']}: {item['reason']}"
    if not approve:
        return head, [f"kind {kind}" + (f" on {item['study']}" if item["study"] else "")]
    details: list = []
    try:
        if kind == "holdout":
            from ..study.cli import _looking
            what, details = _looking(_study(store, item))
            head = f"{head} -> {what}"
        elif kind == "amend":
            from ..study.cli import _changes
            fields = _fields(args)
            details = ([f"study.toml [{sec}] {k} = {v!r}" for sec, kv in fields.items() for k, v in kv.items()]
                       + ["written into study.toml, then recorded as a person's amendment"]) if fields else \
                _changes(_study(store, item))
            head = f"{head} -> a person's amendment of {item['study']}"
        elif kind == "approval":
            details = [f"{args.get('decision', 'approve')} on {args.get('on')} in {item['study']}"]
        elif kind == "screen":
            details = [f"{args.get('decision')} {args.get('source')} ({args.get('criterion', '')})"]
        elif kind == "screen_conflict":
            details = [f"the decision that stands: {options(item)}"]
        elif kind == "screen_maybe":
            details = [f"source {args.get('key') or args.get('source')}: {args.get('title', '')}".rstrip(": "),
                       f"{item['agent']} marked it maybe ({args.get('criterion')}): {args.get('rationale')}",
                       f"quote: \"{args.get('quote') or '(none)'}\"",
                       "your pick (include or exclude) is recorded as your screen of it"]
        elif kind == "budget":
            details = [f"budget {k} {old:g} -> {new:g}" for k, (old, new) in (args.get("change") or {}).items()]
        elif kind == "compute":
            m = args.get("machine") or {}
            want = ", ".join(f"{args[k]:g} {k.replace('_gb', ' GB')}" for k in COMPUTE_KEYS if args.get(k))
            details = [f"another machine with {want}",
                       f"this one: {m.get('cores')} cores, {m.get('memory_gb')} GB, {m.get('gpus')} GPU(s); "
                       f"{m.get('running', 0)} runs running, {m.get('waiting', 0)} waiting",
                       "openlab rents nothing: approving records your answer; add the machine yourself "
                       "(see docs/box.md)"]
        elif kind == "instrument":
            from ..study import instruments
            prep = instruments.prepare(store.root, **{k: v for k, v in args.items() if k in INSTRUMENT_ARGS})
            head = f"{head} -> register instrument {prep.ident}"
            details = instruments.preview(prep)
        elif kind == "instrument_run":
            head = f"{head} -> run {args.get('instrument')} as the person"
            details = [f"args: {args.get('args') or []}", f"study: {args.get('study') or item['study'] or '(none)'}",
                       "a live-order instrument: your signature authorises this one run"]
        elif kind in ("fresh", "continues"):
            v = args.get("why") if kind == "fresh" else args.get("continues")
            details = [f'{item["study"]}/study.toml [study] {kind} = "{v}", recorded as a person\'s amendment'
                       + (": a trial budget and a holdout of its own" if kind == "fresh" else
                          f": one trial and holdout ledger with {v}")]
        else:
            details = ["the decision only: nothing is run"]
    except Exception as e:                                          # noqa: BLE001 -- the act itself will refuse
        details = [f"(cannot preview: {e})"]
    return head, details


def requester_note(store: Store, item: dict) -> str:
    """"" unless the agent that filed the item has ended while it waits (or after its approval): then the line that
    says how to go on (a resumed execution; the decision itself stays on the item for agent_inbox_item)."""
    from .store import FINAL
    a = store.get(item["agent"]) if item.get("agent") else None
    if not a or a["status"] not in FINAL or item["status"] not in ("pending", "approved"):
        return ""
    argv = resume_argv(store, item)
    if argv:
        return f"the requester has ended — run `openlab {' '.join(argv)}` to go on"
    return f"the requester ({a['id']}) has ended — spawn an agent to go on, or act on {item['study'] or 'FOLDER'} " \
           "yourself"


def resume_argv(store: Store, item: dict) -> list:
    """The openlab command that resumes the work of the item's ended requester ([] when there is none, or the
    requester is still running): the replication's builder or planner, the proposal's execution."""
    from .store import FINAL
    a = store.get(item["agent"]) if item.get("agent") else None
    study = item.get("study") or (a or {}).get("study") or ""
    if not a or a["status"] not in FINAL or not study:
        return []
    folder = store.root / study if not Path(study).is_absolute() else Path(study)
    if (folder / "replication.toml").is_file():             # a replication's planner or builder
        return ["replicate", "build" if a["role"] == "replicator" else "draft", study]
    if (folder / "proposal.toml").is_file():
        return ["propose", "execute", study, "--resume"]
    return []


def options(item: dict) -> str:
    """A screen_conflict's choices, numbered from 1."""
    return "; ".join(f"{i}: {o['agent']} {o['decision']} ({o['criterion']})"
                     for i, o in enumerate((item["args"] or {}).get("options") or [], 1))


def perform(store: Store, item: dict, reason: str, by: str, pick=None) -> dict:
    """Carry out an approved item as the person `by`; returns what it made."""
    kind, args = item["kind"], item["args"] or {}
    why = f"{reason} (inbox #{item['id']}, asked by {item['agent']}: {item['reason']})"
    if kind == "screen_conflict":
        opts = args.get("options") or []
        try:
            pick = int(pick) if pick not in (None, "") else None
        except (TypeError, ValueError):
            pick = None
        if pick is None or not 1 <= pick <= len(opts):
            raise ValueError(f"#{item['id']} needs the decision that stands: --pick 1..{len(opts)} ({options(item)})")
        o = opts[pick - 1]
        with acting_as(by):
            out = _study(store, item).screen(args["source"], o["decision"], o["criterion"], why, o.get("quote", ""),
                                             person=True)
        return {"id": out["id"], "kind": "screen", "summary": f"{o['decision']} ({o['criterion']}), as {o['agent']}"}
    if kind == "screen_maybe":
        decision = str(pick or "").strip().lower()
        if decision not in ("include", "exclude"):
            raise ValueError(f"#{item['id']} needs your decision on {args.get('key') or args.get('source')}: "
                             "--pick include or --pick exclude")
        s = _study(store, item)
        from ..study.schema import _review
        ids = [c["id"] for c in _review(s.record)[decision]]
        crit = args.get("criterion") if args.get("criterion") in ids else (ids[0] if ids else "")
        with acting_as(by):
            out = s.screen(args["source"], decision, crit, why, args.get("quote", ""), person=True)
        return {"id": out["id"], "kind": "screen", "summary": f"{decision} ({crit})"}
    with acting_as(by):
        if kind == "holdout":
            out = _study(store, item).holdout(why, person=True)
        elif kind == "amend":
            s = _study(store, item)
            fields = _fields(args)
            if fields is None:
                out = s.amend(why, person=True)
            else:
                from ..study.ledger import write_fields
                p = s.folder / "study.toml"
                before = p.read_text(encoding="utf-8")
                try:
                    write_fields(s.folder, fields)
                    out = s.amend(why, person=True)
                except Exception:
                    p.write_text(before, encoding="utf-8")
                    raise
        elif kind == "approval":
            out = _study(store, item).approve(args["on"], args.get("decision", "approve"), why, person=True)
        elif kind == "screen":
            out = _study(store, item).screen(args["source"], args["decision"], args.get("criterion", ""),
                                             args.get("rationale") or why, args.get("quote", ""), person=True)
        elif kind in ("fresh", "continues"):
            from ..study.ledger import declare
            s = _study(store, item)
            declare(s.folder, kind, args["why"] if kind == "fresh" else args["continues"])
            out = s.amend(why, person=True)
        elif kind == "instrument":
            from ..study import instruments
            kw = {k: v for k, v in args.items() if k in INSTRUMENT_ARGS}
            out = instruments.register(store.root, kw.pop("name"), kw.pop("entry"), reason=why, **kw)
            out = {**out, "summary": f"registered {instruments.ident(out['payload'])}"}
        elif kind == "instrument_run":
            from ..study import instruments
            out = instruments.run(store.root, args["instrument"], args.get("args") or [],
                                  study=args.get("study") or item["study"] or "", note=args.get("note", ""),
                                  reason=why)
            out = {**out, "summary": f"exit {out['payload']['exit']}"}
        elif kind == "budget":
            a = store.get(item["agent"])
            b = dict(a["budget"])
            change = args.get("change") or {k: [b.get(k), args[k]] for k in BUDGET_KEYS if args.get(k)}
            if not change:
                raise ValueError(f"#{item['id']} names no budget change; deny it and ask for a structured request")
            if pick not in (None, "") and len(change) == 1:          # the person's own new limit
                k, (old, _) = next(iter(change.items()))
                new = _num(pick)
                if new is None or new <= float(old or 0):
                    raise ValueError(f"#{item['id']}: the new {k} limit must be a number above {float(old or 0):g}")
                change = {k: [old, int(new) if k == "turns" else new]}
            for k, (_, new) in change.items():
                b[k] = new
            store.update(item["agent"], budget=b)
            told = ", ".join(f"{k} {_limit_text(k, o or 0)} -> {_limit_text(k, n)}" for k, (o, n) in change.items())
            if _held_for_budget(a):
                store.control(item["agent"], "resume", f"the person raised its budget (inbox #{item['id']})", by)
            store.tell(item["agent"], f"the person raised your budget: {told}. Carry on.", "openlab")
            return {"budget": b, "changed": change}
        else:
            return {}
    keep = ("id", "kind", "summary")
    return {k: out[k] for k in keep if isinstance(out, dict) and k in out} or {"result": str(out)[:300]}


def _signed(by: str, item: dict, status: str, reason: str):
    from ..study import personkey
    signer = personkey.current()
    if signer is None:
        raise PermissionError("an inbox decision is a person's act, signed with the person key; this process holds "
                              "none (the person decides in their own terminal: `openlab inbox approve|deny`)")
    msg = f"openlab-inbox|{item['id']}|{item['kind']}|{canonical(item['args'])}|{status}|{reason}".encode("utf-8")
    return signer, {"key": signer.fingerprint, "value": signer.sign(msg)}


def decide(store: Store, id: int, approve: bool, reason: str, by: str = "", pick=None) -> dict:
    item = store.item(id)
    if item is None:
        raise ValueError(f"no inbox item #{id}")
    if item["status"] != "pending":
        raise ValueError(f"#{id} is already {item['status']}")
    if item["kind"] == "question":
        raise ValueError(f"#{id} is a question: answer it (`openlab inbox answer {id} --pick N`, inbox_answer, "
                         "or openlab show) rather than approve or deny it")
    if not reason.strip():
        raise ValueError("a decision needs a reason")
    status = "approved" if approve else "denied"
    signer, sig = _signed(by, item, status, reason)
    by = signer.label
    token = DECIDING.set(id)
    try:
        result = perform(store, item, reason, by, pick) if approve else wrap_up(store, item, reason, by)
    finally:
        DECIDING.reset(token)
    return store.decide(id, status, by, reason, result, sig=sig)


def wrap_up(store: Store, item: dict, reason: str, by: str) -> dict:
    """A denied budget request of an agent openlab holds at its limit: a short grace past what it has used (GRACE),
    and it resumes told to finish now. $ and turns can run past the limit before the hold lands (a message's spend
    is counted when it arrives), so their grace starts from the larger of the limit and what was used; minutes are
    held within a tick. Any other denial does nothing."""
    a = store.get(item["agent"])
    if item["kind"] != "budget" or not (item["args"] or {}).get("hold") or not _held_for_budget(a):
        return {}
    b, spent = dict(a["budget"]), a.get("spent") or {}
    for k in (item["args"].get("change") or {}):
        cur = float(b.get(k) or 0)
        base = max(cur, float(spent.get(k) or 0)) if k in ("usd", "turns") else cur
        b[k] = (round(base + cur * GRACE[k], 4) if k == "usd" else
                int(base + GRACE[k]) if k == "turns" else cur + GRACE[k])
    store.update(item["agent"], budget=b)
    store.tell(item["agent"], f"the person declined more budget ({reason}). Wrap up now: record what you have (a "
               "dead end or a note for what is unfinished) and end with your summary; you have a short grace "
               f"({', '.join(f'{k} {_limit_text(k, v)}' for k, v in b.items() if k in item['args']['change'])}).",
               "openlab")
    store.control(item["agent"], "resume", f"the person declined more budget: wrap up (inbox #{item['id']})", by)
    return {"budget": b, "wrap_up": True}


def start(store: Store, id: int, reason: str) -> dict:
    """Approve now and run the act after (the background path): the item is approved with result
    {"running": true}; `finish` posts the result or the error to it."""
    item = store.item(id)
    if item is None or item["status"] != "pending":
        raise ValueError(f"#{id} is not pending")
    if not reason.strip():
        raise ValueError("a decision needs a reason")
    signer, sig = _signed("", item, "approved", reason)
    return store.decide(id, "approved", signer.label, reason, {"running": True}, sig=sig)


def finish(store: Store, id: int, reason: str) -> dict:
    item = store.item(id)
    token = DECIDING.set(id)
    try:
        result = perform(store, item, reason, item["decided_by"])
    except Exception as e:                                          # noqa: BLE001 -- posted to the item
        result = {"error": getattr(e, "error", None) or str(e), "meaning": getattr(e, "meaning", "")}
    finally:
        DECIDING.reset(token)
    store.set_result(id, result)
    return store.item(id)
