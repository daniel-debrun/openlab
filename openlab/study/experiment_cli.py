"""`openlab experiment new|list|status|request|start|kill|job|react`: a study's experiments from a shell.

`start` is the person's act: it shows what will run unattended (each job's instrument, args and schedule, the pass and
stop rules, the reaction agent's budget, the crontab lines) and asks for the passphrase, which signs it. `job` is what
the crontab lines run; `job --now` runs one at once (a person's act, marked off schedule). Everything else is open
to agents: `new` writes or checks a draft, `request` asks the person to start it, `kill` ends it with a reason, and
`react` records the conclusion (or, with --spawn, fires the reaction agent again when its spawn failed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import experiments as E
from .errors import StudyError


def _ws(a) -> Path:
    return E._ws(a.root or None)


def _new(a) -> dict:
    """Both paths return the keys of experiments.write (file, state, sha, spec; next when it can be requested), plus
    `written` (this call wrote the file), `ok` (the draft there is valid) and `error` (why not; empty when it is)."""
    ws, folder, rel = E.locate(_ws(a), a.study)
    p = E.spec_file(folder, a.name)
    if a.file:
        try:
            text = Path(a.file).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise StudyError(f"cannot read the spec file {a.file}", str(e)) from None
        return {**E.write(ws, rel, a.name, text), "written": True, "ok": True, "error": ""}
    if not p.is_file():
        if not E.NAME.fullmatch(a.name):
            raise StudyError(f"`{a.name}` is not an experiment name", "1-48 of a-z, 0-9, _ and -, starting with a letter")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(E.template(folder, a.name), encoding="utf-8")
        written = True
    else:
        written = False
    text, sha = E.read_spec(folder, a.name)
    try:
        spec, err = E.parse(text, folder, a.name, ws), ""
    except StudyError as e:
        spec, err = None, e.error + (f" ({e.meaning})" if e.meaning else "")
    state = E.state_of(E.nodes(E._record(folder, create=False), a.name), sha)
    out = {"file": str(p.relative_to(ws)), "state": state, "sha": sha, "spec": spec, "written": written,
           "ok": spec is not None, "error": err}
    if spec is not None and state == "draft":
        out["next"] = f"ask the person to start it: experiment_request(study={rel!r}, name={a.name!r}, reason=...)"
    return out


def _list(a) -> dict:
    return {"experiments": E.overview(_ws(a))}


def _status(a) -> dict:
    return E.status(_ws(a), a.study, a.name)


def _request(a) -> dict:
    return E.request(_ws(a), a.study, a.name, a.reason)


def _start(a) -> dict:
    from ..agents import person
    from .personkey import signing
    summary, details = E.describe(_ws(a), a.study, a.name, a.sha or "")
    try:
        signer = person.unlock_interactive(summary, [*details, f"reason: {a.reason}"])
    except person.NotAPerson as e:
        raise StudyError(f"starting an experiment is a person's act, signed with the person key: {e}",
                         "it runs unattended, outside the sandbox; the passphrase is the person's confirmation",
                         "the person runs it in their own terminal; an agent files experiment_request") from None
    with signing(signer):
        return E.start(_ws(a), a.study, a.name, a.reason, spec_sha=a.sha or "")


def _kill(a) -> dict:
    return E.kill(_ws(a), a.study, a.name, a.reason)


def _job(a) -> dict:
    return E.job(_ws(a), a.study, a.name, a.job, now=a.now)


def _react(a) -> dict:
    if a.spawn:
        return E.respawn(_ws(a), a.study, a.name)
    if not a.verdict or not a.conclusion:
        raise StudyError("a reaction has a verdict and a conclusion", fix="--verdict met|not_met|inconclusive "
                         "--conclusion \"...\" --rests-on ID ...  (or --spawn to fire the reaction agent again)")
    return E.react(_ws(a), a.study, a.name, a.verdict, a.conclusion, a.rests_on, a.records)


VERBS = {"new": _new, "list": _list, "status": _status, "request": _request, "start": _start, "kill": _kill,
         "job": _job, "react": _react}


def _bar(done: float, width: int = 20) -> str:
    n = int(round(max(0.0, min(1.0, done)) * width))
    return "#" * n + "." * (width - n)


def _dur(s: float) -> str:
    s = int(s)
    d, h, m = s // 86400, (s % 86400) // 3600, (s % 3600) // 60
    return f"{d}d {h}h" if d else f"{h}h {m}m" if h else f"{m}m"


def _text_status(o: dict) -> list:
    if not o.get("spec"):
        return [f"{o['study']}/{o['file']}: {o['state']}" + (f" ({o['spec_error']})" if o.get("spec_error") else "")]
    p = o["progress"]
    lines = [f"{o['name']} ({o['study']}): {o['state'].upper()}" + ("  [spec file edited after start: not run]"
                                                                     if o.get("edited") else ""),
             f"  purpose: {o['purpose']}"]
    if (o.get("study_seal") or {}).get("line"):
        lines.append(f"  {o['study_seal']['line']}")
    if o.get("spec_error"):
        lines.append(f"  spec: {o['spec_error']}")
    if o["plan_step"]:
        lines.append(f"  Roadmap: {o['owner'] or '?'} / {o['plan_step']}")
    if o.get("request"):
        r = o["request"]
        lines.append(f"  asked by {r['by']}" + (f", inbox #{r['inbox']} ({r.get('status', '?')})" if r["inbox"] else ""))
    if p["target"]:
        lines.append(f"  passes: [{_bar(p['pass_fraction'] or 0)}] {p['passes_ok']}/{p['target']} ok, "
                     f"{p['passes_failed']} failed")
    else:
        lines.append(f"  passes: {p['passes_ok']} ok, {p['passes_failed']} failed")
    if p["until"]:
        lines.append(f"  until {p['until']}" + (f": [{_bar(p['time_fraction'])}]" if p["time_fraction"] is not None
                                                 else ""))
    if p["started"]:
        lines.append(f"  started {p['started']}, {'ran' if p['ended'] else 'running'} {_dur(p['elapsed_s'])}"
                     + (f", last pass {p['last_pass']}" if p["last_pass"] else ""))
    for j in o["jobs"]:
        lines.append(f"  job {j['name']}: {j['instrument']} = {j['pin']} {' '.join(j['args'])}  \"{j['cron']}\" "
                     f"{j['tz'] or 'local'}" + (f"  next {j['next']}" if j["next"] else "")
                     + ("" if j["installed"] or o["state"] != "running" else "  [NOT INSTALLED in the crontab]"))
        for r in j["runs"][:3]:
            lines.append(f"    {r['slot']}: " + (f"{r['run']} exit {r['exit']}" if r["run"] else "no run")
                         + (f"  {r['error']}" if r["error"] else ""))
    if o["end"]:
        lines.append(f"  ended {o['end']['state']} at {o['end']['at']}: {o['end']['reason']}")
    rx = o["reaction"]
    if rx.get("agent") or rx.get("spawn_error"):
        lines.append(f"  reaction: {rx.get('agent') or 'not started'} ({rx.get('status') or rx.get('spawn_error')})")
    if rx.get("verdict"):
        lines.append(f"  conclusion ({rx['verdict']}, {rx['by']}): {rx['conclusion']}")
    return lines


def _text_list(o: dict) -> list:
    if not o["experiments"]:
        return ["no experiments in this workspace (openlab experiment new STUDY NAME writes one)"]
    out = []
    for x in o["experiments"]:
        p = x.get("progress") or {}
        out.append(f"{x['study']}/{x.get('name') or '?'}: {x['state']}"
                   + (f", {p.get('passes_ok', 0)}/{p.get('target') or '-'} passes" if p else "")
                   + (f"  ({x['spec_error']})" if x.get("spec_error") else "")
                   + ("  [study not sealed: openlab study seal " + x["study"] + "]"
                      if (x.get("study_seal") or {}).get("line") else ""))
    return out


def _text_new(o: dict) -> list:
    """`new`'s result (see _new): "wrote FILE (sha ...): valid", or why the draft is not valid yet."""
    valid = o.get("spec") is not None and not o.get("error")
    state = o.get("state") or "draft"
    lines = [("wrote " if o.get("written") else "") + f"{o['file']} (sha {o['sha'][:12]}): "
             + ("valid" if valid else f"not yet valid: {o.get('error') or '?'}")
             + ("" if state == "draft" else f"; it is {state}")]
    if o.get("next"):
        lines.append(f"  next: {o['next']}")
    return lines


def _text_reaction(r: dict) -> str:
    """fire()'s outcome: the reaction agent, or why it could not be spawned."""
    if r.get("ok"):
        return f"{r['agent']}" + (" (already fired)" if r.get("again") is False else "")
    return f"not started: {r.get('error') or '?'} (fire it again: openlab experiment react STUDY NAME --spawn)"


def _text_kill(o: dict) -> list:
    return [f"{o['id']}: {o['state']}", f"  reaction: {_text_reaction(o['reaction'])}",
            *([f"  note: {o['note']}"] if o.get("note") else []),
            *([f"  warning: {o['warning']}"] if o.get("warning") else [])]


def _text_start(o: dict) -> list:
    return [o["summary"], *[f"  {l}" for l in o["lines"]], f"  Roadmap: {o['plan_step']}"]


TEXT = {"new": _text_new, "list": _text_list, "status": _text_status,
        "request": lambda o: [f"{o['id']}: requested, inbox #{o['item']} (sha {o['sha'][:12]})", f"  {o['next']}",
                              *([f"  note: {o['note']}"] if o.get("note") else [])],
        "start": _text_start,
        "kill": _text_kill,
        "job": lambda o: [json.dumps(o, default=str)],
        "react": lambda o: [json.dumps(o, default=str)]}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="openlab experiment", description="A study's experiments: its pinned "
                                "instruments run on a schedule for a set number of passes, then an agent concludes.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default="", help="the workspace (default: the one the current folder is in)")
    common.add_argument("--json", action="store_true", help="the result as one JSON object")
    sub = p.add_subparsers(dest="verb", required=True)

    def one(verb: str, help_: str, name: bool = True):
        s = sub.add_parser(verb, parents=[common], help=help_)
        s.add_argument("study", help="the study's folder, from the workspace")
        if name:
            s.add_argument("name", help="the experiment (its file is STUDY/experiments/NAME.toml)")
        return s

    n = one("new", "write a draft from the template (or --file), or check the one there")
    n.add_argument("--file", default="", help="a spec file to check and save as the draft")
    sub.add_parser("list", parents=[common], help="every experiment of the workspace's studies")
    one("status", "one experiment: state, progress, jobs, runs, end, reaction")
    r = one("request", "ask the person to start it (an inbox item)")
    r.add_argument("--reason", required=True)
    s = one("start", "start it: installs its crontab lines (a person's act: asks for the passphrase)")
    s.add_argument("--reason", required=True, help="why it runs")
    s.add_argument("--sha", default="", help="the spec sha the request named (refused when the file changed)")
    k = one("kill", "end it now: its lines go, and the reaction agent is fired")
    k.add_argument("--reason", required=True)
    j = one("job", "run one job as its crontab line does (only at a minute it is due, once)")
    j.add_argument("job")
    j.add_argument("--now", action="store_true", help="run it now, off schedule (a person's act)")
    x = one("react", "record the conclusion of an ended experiment (or --spawn: fire the reaction agent again)")
    x.add_argument("--verdict", default="", choices=("", *E.EXPERIMENT_VERDICTS))
    x.add_argument("--conclusion", default="")
    x.add_argument("--rests-on", nargs="*", default=[], help="the pass and run ids it rests on")
    x.add_argument("--records", nargs="*", default=[], help="the record ids written on the way")
    x.add_argument("--spawn", action="store_true", help="fire the reaction agent again (when its spawn failed)")
    return p


def main(argv=None) -> int:
    a = _parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        out = VERBS[a.verb](a)
    except StudyError as e:
        if a.json:
            print(json.dumps({"ok": False, "error": e.error, "meaning": e.meaning, "fix": e.fix}))
        else:
            print("\n".join(f"{k}: {v}" for k, v in (("error", e.error), ("meaning", e.meaning), ("fix", e.fix))
                            if v), file=sys.stderr)
        return 1
    print(json.dumps(out, default=str) if a.json else "\n".join(TEXT[a.verb](out)))
    return 0
