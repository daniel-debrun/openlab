"""`openlab replicate PAPER [FOLDER]`: a paper turned into checked code, from its id alone.

    openlab replicate PAPER [FOLDER]          fetch the paper, and a planning agent drafts replication.toml; at a
                                             terminal, wait for it, show the plan and ask to approve and build
    openlab replicate draft FOLDER            a planning agent (again): it continues the plan as it is, until lint is clean
    openlab replicate lint FOLDER             what the plan still gets wrong (every quote checked on its page)
    openlab replicate show FOLDER             replication.md, the plan as you approve it
    openlab replicate approve FOLDER          a person's act, signed: the study made from the plan, then the build
    openlab replicate build FOLDER            spawn the builder (after approve --no-build, or once the last one ended)
    openlab replicate status FOLDER           the stage, the agents and their spend, the verdicts so far
    openlab replicate report FOLDER           README.md and results/ written from the record

PAPER is an arXiv id, a DOI or a PDF file. openlab.study.paperplan has the flow; `openlab study replicate` is the
manual form, with every parameter given by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import paperplan as R
from .errors import StudyError
from .personkey import signing

VERBS = ("draft", "lint", "show", "approve", "build", "status", "report")


def _tty() -> bool:
    return sys.stdin.isatty() and sys.stderr.isatty()


def _say(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def _wait(folder, agent: str) -> str:
    """Follow the planner until it ends, a line per change; the planner goes on if the person stops waiting."""
    from ..agents.store import LIVE, Store
    root, _ = R.workspace(folder)
    store, last, shown = Store(root), "", 0.0
    while True:
        a = store.get(agent) or {}
        usd = float((a.get("spent") or {}).get("usd", 0))
        line = f"  {agent} {a.get('status', '?')}, ${usd:.2f}"
        if line != last or time.monotonic() - shown > 30:
            _say(line)
            last, shown = line, time.monotonic()
        if a.get("status") not in LIVE:
            return a.get("status", "")
        time.sleep(3)


def _approve_and_build(folder, reason: str, build: bool, cli: str, model) -> dict:
    from ..agents import person
    raw = R.load(folder)
    try:
        signer = person.unlock_interactive(f"approve the replication plan of {raw.get('paper')}", (
            R.summary(raw) + "\nApproving makes the study from the plan (the target and tolerance are fixed before any "
            "run)" + (" and starts the build." if build else ".")).split("\n"))
    except person.NotAPerson as e:
        raise StudyError(str(e), "approving the plan is a person's act, signed with the person key",
                         "run it in your own terminal") from None
    with signing(signer):
        out = R.approve(folder, reason, signer.label)
    out["build"] = R.build(folder, cli=cli, model=model) if build else None
    return out


def _start(a) -> dict:
    out = R.start(a.paper, a.folder, a.budget_usd, a.minutes)
    folder = out["folder"]
    from .proposal import cli_available
    draft = a.draft_with
    if draft == "auto":
        draft = "claude" if cli_available("claude") else "none"
    out["drafting"] = R.draft(folder, cli=draft, usd=a.planner_budget_usd, model=a.model) if draft != "none" else None
    out["approved"] = None
    if out["drafting"] is None or a.no_wait or a.json or not _tty():
        return out
    _say(f"{out['paper']['key']}: {out['paper']['n_pages']} pages in {folder}/paper; the planner drafts the plan "
         "(Ctrl-C stops waiting; it goes on)")
    try:
        _wait(folder, out["drafting"]["id"])
    except KeyboardInterrupt:
        _say("\nstopped waiting; the planner goes on")
        return out
    lt = R.lint(folder)
    _say("\n" + (Path(lt["md"]).read_text(encoding="utf-8")))
    if not lt["ok"]:
        _say("\n".join(f"error: {p['error']}" for p in lt["problems"]))
        _say(f"the plan is not ready: fix replication.toml (or ask your agent), then `openlab replicate approve {folder}`")
        return out
    for w in lt["warnings"]:
        _say(f"warning: {w}")
    _say("approve this plan and start the build? [y/N] ")
    if sys.stdin.readline().strip().lower() not in ("y", "yes"):
        _say(f"not approved; when ready: openlab replicate approve {folder}")
        return out
    out["approved"] = _approve_and_build(folder, "approved at the terminal after reading replication.md", True,
                                         a.cli, a.model)
    return out


def _run(a) -> dict:
    if a.verb == "draft":
        return R.draft(a.folder, cli=a.cli, usd=a.planner_budget_usd, model=a.model, note=a.note)
    if a.verb == "lint":
        return R.lint(a.folder)
    if a.verb == "show":
        R.lint(a.folder)
        return {"md": (Path(a.folder) / R.MD).read_text(encoding="utf-8")}
    if a.verb == "approve":
        return _approve_and_build(a.folder, a.reason, not a.no_build, a.cli, a.model)
    if a.verb == "build":
        return R.build(a.folder, cli=a.cli, model=a.model, dry_run=a.dry_run)
    if a.verb == "status":
        return R.status(a.folder)
    return R.report(a.folder)


# --- text --------------------------------------------------------------------------------------------------------
def _build_lines(b: dict | None, folder: str) -> list:
    if not b:
        return [f"next: openlab replicate build {folder}"]
    return [f"building with {b['agent']} ({b['cli']}, ${b['budget']['usd']:g}, {b['budget']['minutes']:g} min)",
            f"  follow: openlab agent log {b['agent']} -f    watch: openlab watch    progress: openlab replicate status "
            f"{folder}", f"  when it ends: README.md and results/ in {folder} (openlab replicate report {folder})"]


def _start_text(r: dict) -> list:
    f = r["folder"]
    out = [f"{r['paper']['key']}: {r['paper']['n_pages']} pages fetched into {f}/paper"]
    d = r["drafting"]
    if r.get("approved"):
        ap = r["approved"]
        return out + [f"approved: {ap['id']} (sha {ap['sha'][:12]}); {len(ap['evidence'])} paper numbers recorded "
                      f"as evidence; target {ap['target']}"] + _build_lines(ap["build"], f)
    if d:
        out += [f"drafting the plan with {d['id']} ({d['cli']}, ${d['budget']['usd']:g})",
                f"  follow: openlab agent log {d['id']} -f"]
    else:
        out.append(f"fill {f}/replication.toml (or ask your agent to, until `openlab replicate lint {f}` is clean)")
    return out + [f"then: openlab replicate show {f}, and a person runs `openlab replicate approve {f}`"]


def _lint_text(r: dict) -> list:
    out = [f"status: {r['status']}"]
    out += [f"error: {p['error']}" + (f"\n  meaning: {p['meaning']}" if p["meaning"] else "")
            + (f"\n  fix: {p['fix']}" if p["fix"] else "") for p in r["problems"]]
    out += [f"warning: {w}" for w in r["warnings"]]
    return out + [("clean" if r["ok"] else f"{len(r['problems'])} problems") + f": {r['md']} rendered "
                  f"(sha {r['sha'][:12]})"]


def _approve_text(r: dict) -> list:
    if r.get("amends"):
        return [f"approved the changed plan: {r['id']} (sha {r['sha'][:12]}), amending {r['amends']}; "
                f"{len(r['quoted'])} claim(s) quoted again, the others keep their evidence"] + \
            _build_lines(r.get("build"), r["folder"])
    return [f"approved: {r['id']} (sha {r['sha'][:12]}); the study is made in {r['folder']}",
            f"  {len(r['evidence'])} paper numbers recorded as evidence, each quoted from its page; target {r['target']}"
            " (tolerance fixed)"] + _build_lines(r.get("build"), r["folder"])


def _status_text(r: dict) -> list:
    out = [f"{r['paper']['key']} in {r['folder']}: {r['stage']}"]
    out += [f"  agent {x['id']} ({x['role']}) {x['status']}, ${x['usd']:.2f} of ${x['budget'].get('usd', 0):g}"
            for x in r["agents"]]
    if r["stage"] == "draft":
        return out + [f"  plan: {'clean' if r['ok'] else str(r['problems']) + ' problems'} ({r['md']})"]
    out.append(f"  approved as {r['approval']}; {r['trials']} trial(s); rebuild: "
               + {True: "faithful", False: "NOT faithful", None: "no fidelity yet"}[r["rebuilt"]])
    out += [f"  fidelity {x['id']} {x['arm']}: paper {x['paper']:g}, ours {x['ours']:.4g} -> "
            f"{'within' if x['passed'] else 'outside'} {x['tolerance']:g}" for x in r["fidelity"]]
    out += [f"  claim {x['id']} {x['method']}: paper {x['paper']:g}, ours {x['ours']:.4g} -> {x['verdict']}"
            + (f" (openlab's corrected rule gives {x['rule_now']}: check the claim again)" if x.get("rule_now") else "")
            for x in r["replications"]]
    if r.get("pace"):
        p = r["pace"]
        out.append(f"  pace: {p['done']} of {p['planned']} planned runs, {p['run_minutes']:g} min a run (planned "
                   f"{p['planned_run_minutes']:g}); the runs left need ~{p['need']:.0f} min at {p['parallel']} at once,"
                   f" {p['left']:.0f} min left" + ("" if p["fits"] else " -- they do not fit: ask the person before "
                                                   "cutting runs"))
    if r.get("superseded"):
        out.append(f"  ({r['superseded']} earlier check(s) of the same numbers on record: openlab study replications)")
    return out + ([f"  report: {r['folder']}/README.md"] if r["report"] else [])


def _report_text(r: dict) -> list:
    out = [f"wrote {r['readme']} and {r['report']}"]
    out += [f"  {x['claim']} {x['method']} ({x['check']}): paper {x['paper']}, ours "
            f"{'' if x['ours'] is None else format(x['ours'], '.4g')} -> {x['verdict']}" for x in r["claims"]]
    return out


TEXT = {"start": _start_text, "draft": lambda r: [f"drafting the plan with {r['id']} ({r['cli']}, "
                                                    f"${r['budget']['usd']:g})", f"  follow: openlab agent log {r['id']} -f"], "lint": _lint_text, "show": lambda r: [r["md"].rstrip()], "approve": _approve_text,
        "build": lambda r: (["the builder's brief (nothing spawned):", "", r["brief"]] if r["dry_run"]
                            else (([f"resuming after {', '.join(r['resumed'])}"] if r.get("resumed") else [])
                                  + _build_lines(r, r["folder"]))),
        "status": _status_text, "report": _report_text}


def _parser(verb: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="openlab replicate", description="A paper turned into checked code: give the "
                                "paper; an agent drafts the plan, you approve it, agents build and verify it.")
    p.add_argument("--json", action="store_true", help="the reply as one JSON object")
    if verb == "start":
        p.add_argument("paper", help="an arXiv id, a DOI or a PDF file")
        p.add_argument("folder", nargs="?", default=None, help="the folder to make (default: the paper's id)")
        p.add_argument("--draft-with", default="auto", choices=("auto", "claude", "codex", "none"),
                       help="the planning agent's CLI (auto: claude when installed; none: you fill the plan)")
        p.add_argument("--planner-budget-usd", type=float, default=None, help=f"default {R.PLANNER_USD:g}")
        p.add_argument("--budget-usd", type=float, default=None, help="the build's $ the plan starts from (default 5)")
        p.add_argument("--minutes", type=float, default=None, help="the build's minutes (default 120)")
        p.add_argument("--no-wait", action="store_true", help="return once the planner is spawned")
    else:
        p.add_argument("verb", choices=VERBS)
        p.add_argument("folder")
    if verb == "draft":
        p.add_argument("--planner-budget-usd", type=float, default=None, help=f"default {R.PLANNER_USD:g}")
        p.add_argument("--note", default="", help="the person's direction to the planner (what to change, and why)")
    if verb in ("start", "draft", "approve", "build"):
        p.add_argument("--cli", default="claude", choices=("claude", "codex"), help="the agents' CLI")
        p.add_argument("--model", default=None, help="the agents' model (claude --model / codex -m)")
    if verb == "approve":
        p.add_argument("--reason", default="the plan as drafted, read in replication.md")
        p.add_argument("--no-build", action="store_true", help="approve only; `openlab replicate build` starts it")
    if verb == "build":
        p.add_argument("--dry-run", action="store_true", help="print the builder's brief; spawn nothing")
    return p


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    first = next((x for x in argv if not x.startswith("-")), "")
    verb = first if first in VERBS else "start"
    if not argv or argv in (["-h"], ["--help"]):
        print(__doc__.strip())
        return 0
    a = _parser(verb).parse_args(argv)
    try:
        out = _start(a) if verb == "start" else _run(a)
    except StudyError as e:
        if a.json:
            print(json.dumps({"ok": False, "error": e.error, "meaning": e.meaning, "fix": e.fix}))
        else:
            print("\n".join(f"{k}: {v}" for k, v in (("error", e.error), ("meaning", e.meaning), ("fix", e.fix)) if v),
                  file=sys.stderr)
        return 1
    print(json.dumps(out, default=str) if a.json else "\n".join(TEXT[verb](out)))
    return 1 if verb == "lint" and not out["ok"] else 0
