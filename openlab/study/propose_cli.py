"""`openlab propose new|lint|show|approve|amend|execute|status|stop FOLDER`: research from a prompt.

    openlab propose new FOLDER "<prompt>" [--from FILE] [--mode static|temporal|review] [--budget-usd X]
                       [--draft-with claude|codex|none]     scaffold proposal.toml (and draft it with a planning agent)
    openlab propose lint FOLDER                               refusals and warnings; renders proposal.md
    openlab propose show FOLDER                               proposal.md
    openlab propose approve FOLDER --reason R                 a person's act: study.toml, the question tree, the seal
    openlab propose amend FOLDER --reason R [--person]        record an edit of the approved proposal
    openlab propose execute FOLDER [--lead-cli claude] [--dry-run] [--resume]   spawn the lead with the execution brief
    openlab propose status FOLDER                             per branch: agents, spend, hypotheses, stop rule
    openlab propose stop FOLDER BRANCH --reason R [--kill]    a branch stopped by its stop rule (or killed)

Every person's act is signed with the person key: openlab.agents.person shows what will be signed and asks for the
passphrase.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import open as open_study
from . import proposal as P
from .errors import StudyError
from .personkey import signing


def _person(what: str, detail: str):
    """The person key, unlocked by the person's passphrase after they read `what` and `detail`."""
    from ..agents import person
    try:
        return person.unlock_interactive(what, detail.split("\n"))
    except person.NotAPerson as e:
        raise StudyError(str(e), "a proposal's approval and a protected amendment are a person's acts, signed with "
                         "the person key", "run it in your own terminal") from None


def _new(a) -> dict:
    prompt = Path(a.from_file).read_text(encoding="utf-8") if a.from_file else (a.prompt or a.prompt_opt or "")
    out = P.new(a.folder, prompt, a.mode, a.budget_usd, a.minutes)
    draft = a.draft_with
    if draft == "auto":
        draft = "claude" if P.cli_available("claude") else "none"
    out["drafting"] = P.draft(a.folder, prompt, cli=draft, usd=a.planner_budget_usd, model=a.model) \
        if draft != "none" else None
    return out


def _lint(a) -> dict:
    return P.lint(open_study(a.folder))


def _show(a) -> dict:
    s = open_study(a.folder)
    P.lint(s)                                                   # renders proposal.md from proposal.toml
    return {"md": (s.folder / P.MD).read_text(encoding="utf-8")}


def _same_data(s, c: dict) -> bool:
    """Ask the person at the terminal whether this proposal continues the study `c` that asks its question of the
    same data, or is a fresh question; the answer goes into proposal.toml. False when there is no terminal or no
    answer (the approval is then refused, naming both fixes)."""
    if not (sys.stdin.isatty() and sys.stderr.isatty()):
        return False
    f = c["fixes"]
    print(f"{s.folder.name} asks the question of {c['name']} ({c['folder']}) of the same data: "
          f"{c['trials']} trials and {c['looks']} holdout looks are spent there.\n"
          f"  c  continue it: {f['continues']}\n  f  a fresh question: {f['fresh']}\n"
          "continue or fresh? type c or f (anything else stops): ", end="", file=sys.stderr, flush=True)
    pick = sys.stdin.readline().strip().lower()
    if pick == "c":
        P.declare(s.folder, "continues", c["rel"])
    elif pick == "f":
        print("why is it a new question? ", end="", file=sys.stderr, flush=True)
        why = sys.stdin.readline().strip()
        if not why:
            return False
        P.declare(s.folder, "fresh", why)
    else:
        return False
    return True


def _approve(a) -> dict:
    s = open_study(a.folder)
    raw = P.load(s.folder)
    probs = P.problems(raw, P.cap(s.folder))
    if probs:
        raise P._refuse(probs)
    if P.approved(s.record) is None:
        pf = P.preflight(s, raw)
        if pf["conflicts"] and len(pf["conflicts"]) == len(pf["errors"]) and _same_data(s, pf["conflicts"][0]):
            raw, pf = P.load(s.folder), P.preflight(s, P.load(s.folder))
        if pf["errors"]:
            raise P._refuse(pf["errors"])
    signer = _person(f"approve the proposal of {s.folder.name}",
                     P.summary(raw) + "\nApproving writes study.toml from it, makes the question tree and seals the "
                     "plan.")
    with signing(signer):
        return P.approve(s, a.reason, signer.label)


def _amend(a) -> dict:
    s = open_study(a.folder)
    by = ""
    if a.person:
        node = P.approved(s.record)
        changes = P.diff(node.payload["proposal"], P.load(s.folder)) if node else {}
        signer = _person(f"amend the proposal of {s.folder.name}", "\n".join(
            ["changed: " + (", ".join(changes) or "nothing"), *P.study_effect_lines(s)]))
        with signing(signer):
            return P.amend(s, a.reason, a.person, signer.label)
    return P.amend(s, a.reason, a.person, by)


def _execute(a) -> dict:
    return P.execute(open_study(a.folder), cli=a.lead_cli, dry_run=a.dry_run, resume=a.resume, model=a.model)


def _status(a) -> dict:
    return P.status(open_study(a.folder))


def _stop(a) -> dict:
    return P.stop(open_study(a.folder), a.branch, a.reason, killed=a.kill, shown_by=a.shown_by)


VERBS = {"new": _new, "lint": _lint, "show": _show, "approve": _approve, "amend": _amend, "execute": _execute,
         "status": _status, "stop": _stop}


def _money(b: dict) -> str:
    return f"${b.get('usd', 0):g}, {b.get('minutes', 0):g} min, {b.get('trials', 0)} trials"


def _new_text(r: dict) -> list:
    out = [f"proposal in {r['folder']}: {', '.join(r['files'])} (mode {r['mode']}, budget ${r['budget']['usd']:g}, "
           f"{r['budget']['minutes']:g} min)"]
    d = r["drafting"]
    if d:
        out += [f"drafting with {d['id']} ({d['cli']}{', ' + d['model'] if d.get('model') else ''}, budget "
                f"${d['budget']['usd']:g}, {d['budget']['minutes']:g} min)",
                f"  follow: openlab agent log {d['id']} -f"]
    else:
        out.append(f"edit {r['folder']}/proposal.toml (or ask your agent: openlab_propose_lint until clean)")
    return out + [f"next: `openlab propose lint {r['folder']}`, then a person runs "
                  f"`openlab propose approve {r['folder']} --reason ...`"]


def _lint_text(r: dict) -> list:
    out = [f"status: {r['status']}"]
    out += [f"error: {p['error']}" + (f"\n  meaning: {p['meaning']}" if p["meaning"] else "")
            + (f"\n  fix: {p['fix']}" if p["fix"] else "") for p in r["problems"]]
    if r.get("drift"):
        out.append(f"  {r['drift']}")
    for c in r.get("conflicts") or []:
        out += [f"same data as {c['name']} ({c['folder']}): either", f"  continue it: {c['fixes']['continues']}",
                f"  or fresh: {c['fixes']['fresh']}"]
    out += [f"warning: {w}" for w in r["warnings"]]
    word = "clean" if r["ok"] else f"{len(r['problems'])} problems" if r["problems"] else "changed"
    out.append(f"{word}: {r['md']} rendered (sha {r['sha'][:12]})")
    return out


def _approve_text(r: dict) -> list:
    return [f"approved: {r['id']} (sha {r['sha'][:12]})",
            f"study.toml written; sealed as {r['plan']}" + (f", amended {r['amended']}" if r["amended"] else "")
            + f"; trials budget {r['trials']}",
            f"question tree: root {r['tree']['root']}, branches " + ", ".join(f"{b} {r['tree'][b]}" for b in r["branches"]),
            "next: `openlab propose execute FOLDER --dry-run` shows the lead's brief; without --dry-run it starts"]


def _execute_text(r: dict) -> list:
    rs = r.get("resume")
    head = [f"resuming: {', '.join(rs['left'])} left; finished: {', '.join(rs['finished']) or 'none'}; budget left "
            f"${rs['budget']['usd']:g}, {rs['budget']['minutes']:g} min (spent ${rs['spent']['usd']:g})"] if rs else []
    if r["dry_run"]:
        return head + ["the lead's brief (nothing spawned):", "", r["brief"], "", "the planned tree:"] + r["tree"]
    b = r["budget"]
    return head + [f"spawned the lead {r['lead']} ({r['cli']}{', ' + r['model'] if r.get('model') else ''}) with "
                   f"${b['usd']:g}, {b['minutes']:g} min for "
                   f"{r['proposal']}",
                   f"  follow: openlab agent log {r['lead']} -f    progress: openlab propose status FOLDER    "
                   f"tree: openlab agents --tree"] \
        + ([f"  working copies: {', '.join(r['copies'])} (one per experiment branch)"] if r.get("copies") else []) \
        + r["tree"]


def _status_text(r: dict) -> list:
    from ..agents.view import tokens
    sp, h = r["spent"], r["hypotheses"]
    out = [f"{r['title']}: {r['question']}",
           f"approved {r['approved_at'][:16]} by {r['approved_by']} ({r['proposal']})"
           + (f", {r['amendments']} amendments" if r["amendments"] else ""),
           f"spent ${sp['usd']:.2f} of ${r['budget']['usd']:g}, trials {r['trials_used']} of "
           f"{r['budget'].get('trials', 0)}, tokens {tokens(sp['tokens'])}"
           + "".join(f"; lead {x['id']} {x['status']} (${x['usd']:.2f}, {x['minutes']:.0f} min)" for x in r["lead"]),
           f"hypotheses: {h['tested']} tested, {h['killed']} killed, {h['untouched']} untouched"
           + "".join(f"; {n} {k}" for k, n in (r.get("verdicts") or {}).items() if n)]
    for b in r["branches"]:
        s, bud = b["spent"], b["budget"]
        out += ["", f"{b['id']} [{b['role']}] {b.get('label') or b['state'].upper()}  "
                f"${s['usd']:.2f}/{bud['usd']:g}  "
                f"{s['minutes']:.0f}/{bud['minutes']:g} min  trials {s['trials']}/{bud.get('trials', 0)}  "
                f"tokens {tokens(s['tokens'])}  {b['question']}"
                + (f"  (item held by {b['held_by']['agent']} until {b['held_by']['expires']} UTC)" if b["held_by"] else "")]
        out += [f"   agent {a['id']} ({a['role'] or a['kind']}) {a['status']}, ${a['usd']:.2f}, "
                f"{tokens(a['tokens'])} tokens, {a['minutes']:.0f} min" for a in b["agents"]] or ["   no agent yet"]
        for x in b["hypotheses"]:
            out.append(f"   {x['id']} {x.get('label') or x['state']}"
                       + (" (dead end recorded)" if x["verdict"] and x["dead_ends"] else "")
                       + (f" by {x['by']}" if x["by"] else "") + f": {x['statement']}")
            out += [f"      dead end {d['id']} by {d['by']}: {d['reason']}" for d in x["dead_ends"]]
        if b.get("blocked_by"):
            out += [f"   BLOCKED: {b['blocked_by']['reason']}", f"   fix: {b['blocked_by']['fix']}"]
        out += [f"   stop rule: {b['stop_rule']} -> {b['stop_state']}", f"   kill: {b['kill']}"]
    return out


TEXT = {"new": _new_text, "lint": _lint_text, "show": lambda r: [r["md"].rstrip()], "approve": _approve_text,
        "amend": lambda r: [f"amended: {r['id']} ({', '.join(r['changes'])})"
                            + (f"; a person's, for {', '.join(r['protected'])}" if r["protected"] else "")
                            + (f"; new nodes {', '.join(r['new_nodes'])}" if r["new_nodes"] else "")]
        + ([f"study.toml amended as {r['study_amendment']} ({', '.join(r['study_changes'])})"]
           if r.get("study_amendment") else []),
        "execute": _execute_text, "status": _status_text,
        "stop": lambda r: [f"branch {r['branch']} {'killed' if r['killed'] else 'stopped'}"]}


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("folder")
    common.add_argument("--json", action="store_true", help="the reply as one JSON object")
    p = argparse.ArgumentParser(prog="openlab propose", description="Research from a prompt: a proposal a person "
                                "approves, a sealed plan, and its execution by a tree of agents.")
    sub = p.add_subparsers(dest="verb", required=True)
    n = sub.add_parser("new", parents=[common], help="scaffold proposal.toml from a prompt (and draft it)")
    n.add_argument("prompt", nargs="?", default="", help="a paragraph: what you want to find out (before or after "
                   "the options)")
    n.add_argument("--prompt", dest="prompt_opt", default="", help="the paragraph, as an option")
    n.add_argument("--from", dest="from_file", default=None, help="read the prompt from a file")
    n.add_argument("--mode", default="static", choices=P.MODES, help="the study's mode (default static)")
    n.add_argument("--budget-usd", type=float, default=None, help="the proposal's total $ (default 5, at most the cap)")
    n.add_argument("--minutes", type=float, default=None, help="the proposal's total minutes (default 120)")
    n.add_argument("--draft-with", default="auto", choices=("auto", "claude", "codex", "none"),
                   help=f"a planning agent drafts it (${P.PLANNER_USD:g}, {P.PLANNER_MINUTES} min); auto: claude when "
                   "it is installed")
    n.add_argument("--model", default=None, help="the planning agent's model (claude --model / codex -m); default: "
                   "openlab.toml [agents] model, else the CLI's default")
    n.add_argument("--planner-budget-usd", type=float, default=None,
                   help=f"the planning agent's $ (default {P.PLANNER_USD:g}; any amount above 0)")
    sub.add_parser("lint", parents=[common], help="the proposal's refusals and warnings; renders proposal.md")
    sub.add_parser("show", parents=[common], help="proposal.md, rendered from proposal.toml")
    ap = sub.add_parser("approve", parents=[common], help="a person's approval: study.toml, the question tree, the seal")
    ap.add_argument("--reason", required=True)
    am = sub.add_parser("amend", parents=[common], help="record an edit of the approved proposal")
    am.add_argument("--reason", required=True)
    am.add_argument("--person", action="store_true", help="a person's amendment (budgets, stop rules, kill criteria, "
                    "falsifiers, the primary metric)")
    ex = sub.add_parser("execute", parents=[common], help="spawn the lead agent with the execution brief")
    ex.add_argument("--lead-cli", default="claude", choices=("claude", "codex"))
    ex.add_argument("--model", default=None, help="the lead's model (claude --model / codex -m), which its children "
                    "inherit; default: openlab.toml [agents] model, else the CLI's default")
    ex.add_argument("--dry-run", action="store_true", help="print the brief and the planned tree; spawn nothing")
    ex.add_argument("--resume", action="store_true", help="after the earlier agents ended (a person request, the "
                    "budget): a lead for the unfinished branches only, with the budget left")
    sub.add_parser("status", parents=[common], help="per branch: agents, spend, hypotheses, dead ends, stop rule")
    st = sub.add_parser("stop", parents=[common], help="record a branch stopped by its stop rule, or killed")
    st.add_argument("branch")
    st.add_argument("--reason", required=True, help="the rule that fired and what showed it")
    st.add_argument("--kill", "--killed", dest="kill", action="store_true", help="its kill criterion was met")
    st.add_argument("--shown-by", action="append", default=[], help="a result, evidence or test id (repeat)")
    return p


def main(argv=None) -> int:
    p = _parser()
    a, rest = p.parse_known_args(argv)
    if rest and a.verb == "new" and not a.prompt and not any(x.startswith("-") for x in rest):
        a.prompt = " ".join(rest)              # the prompt given after the options
    elif rest:
        p.error(f"unrecognized arguments: {' '.join(rest)}")
    try:
        out = VERBS[a.verb](a)
    except StudyError as e:
        if a.json:
            print(json.dumps({"ok": False, "error": e.error, "meaning": e.meaning, "fix": e.fix}))
        else:
            print("\n".join(f"{k}: {v}" for k, v in (("error", e.error), ("meaning", e.meaning), ("fix", e.fix)) if v),
                  file=sys.stderr)
        return 1
    print(json.dumps(out, default=str) if a.json else "\n".join(TEXT[a.verb](out)))
    return 1 if a.verb == "lint" and not out["ok"] else 0
