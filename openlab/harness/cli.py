"""`openlab harness`: the texts openlab briefs its agents with, as versions a person makes active, the probe signals
of the runs they briefed, the lessons drawn from them, and refinement.

    openlab harness show                          the active version, the stored ones, [harness] settings
    openlab harness check [DIR]                   validate a harness folder (default: the active version)
    openlab harness export REF DEST               a copy to edit;  install DIR [--note ...]: store it as a version
    openlab harness diff A B                      what changed between two versions
    openlab harness use REF --reason ...          make a version active (a person act; needs a supporting lesson)
    openlab harness rollback --reason ...         the version active before (a person act)
    openlab harness signals [AGENT] [--by KEY]    probe signals of one run, or grouped by harness_sha | harness | role
    openlab harness contrast --ref X --cand Y     two groups of runs side by side (--save: a lesson's basis)
    openlab harness transcript AGENT              a run as [ACTION] / [OBSERVATION] / [ERROR]
    openlab harness lessons [ID]                  harness lessons (one: its body and track record)
    openlab harness confirm ID | retire ID --reason ...
    openlab harness reflect AGENT                 run the reflector on an ended agent now (spends reflector_budget_usd)
    openlab harness refine UNIT SOURCE [--k N]    one Propose-Probe-Compose step (a person act; spends)
    openlab harness replay                        refinement policies screened on the recorded steps
    openlab harness cost                          what reflection, refinement and evaluation cost
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _root(a) -> Path:
    from ..agents.store import root_of
    return root_of(a.root) if a.root else root_of()


def _out(obj, as_json: bool, lines=None) -> None:
    if as_json or lines is None:
        print(json.dumps(obj, indent=1, default=str))
    else:
        print("\n".join(lines))


def _person(summary: str, details, fn):
    from ..agents import person
    with person.act(summary, details):
        return fn()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="openlab harness", description="The texts openlab briefs its agents with: "
                                "versions, probe signals, lessons, refinement.")
    p.add_argument("--root", default="", help="the workspace (default: the one around the current folder)")
    p.add_argument("--json", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show")
    c = sub.add_parser("check")
    c.add_argument("dir", nargs="?", default="")
    sub.add_parser("versions")
    e = sub.add_parser("export")
    e.add_argument("ref")
    e.add_argument("dest")
    i = sub.add_parser("install")
    i.add_argument("dir")
    i.add_argument("--note", default="")
    d = sub.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    u = sub.add_parser("use")
    u.add_argument("ref")
    u.add_argument("--reason", required=True)
    r = sub.add_parser("rollback")
    r.add_argument("--reason", required=True)
    s = sub.add_parser("signals")
    s.add_argument("agent", nargs="?", default="")
    s.add_argument("--role", default="")
    s.add_argument("--by", default="harness_sha", choices=("harness_sha", "harness", "role"))
    ct = sub.add_parser("contrast")
    ct.add_argument("--ref", required=True, help="a harness_sha or harness version (prefix) of the reference runs")
    ct.add_argument("--cand", required=True)
    ct.add_argument("--role", default="")
    ct.add_argument("--key", default="harness", choices=("harness", "harness_sha"))
    ct.add_argument("--save", action="store_true", help="store it, for a lesson's basis")
    t = sub.add_parser("transcript")
    t.add_argument("agent")
    ls = sub.add_parser("lessons")
    ls.add_argument("id", nargs="?", default="")
    cf = sub.add_parser("confirm")
    cf.add_argument("id")
    rt = sub.add_parser("retire")
    rt.add_argument("id")
    rt.add_argument("--reason", required=True)
    rf = sub.add_parser("reflect")
    rf.add_argument("agent")
    rn = sub.add_parser("refine")
    rn.add_argument("unit", choices=("propose-plan", "replicate-plan", "task"))
    rn.add_argument("source", help="the study folder (propose-plan), the replication folder (replicate-plan), or a "
                    "folder of files for a task")
    rn.add_argument("--prompt", default="", help="the task (task), or the person's prompt (propose-plan)")
    rn.add_argument("--k", type=int, default=1)
    rn.add_argument("--resume", default="", help="a step id to resume")
    rn.add_argument("--selector-cli", default="claude", choices=("claude", "codex"),
                    help="the blind selector's CLI at k > 1 (another model family than the editors where possible)")
    sub.add_parser("replay")
    sub.add_parser("cost")
    a = p.parse_args(argv)
    from . import HarnessError
    try:
        return _run(a)
    except (HarnessError, PermissionError, KeyError, ValueError) as err:
        problems = getattr(err, "problems", None) or [str(err).strip("'\"")]
        print("openlab harness: " + "\n  ".join(problems), file=sys.stderr)
        return 1
    except Exception as err:                                        # noqa: BLE001 -- NotAPerson and the like
        if type(err).__name__ == "NotAPerson":
            print(f"openlab harness: {err}", file=sys.stderr)
            return 1
        raise


def _run(a) -> int:
    from . import laws, read, settings, validate, versions
    root = _root(a)
    if a.cmd == "show":
        rows = versions.stored(root)
        _out({"active": versions.active_sha(root) or "default", "versions": rows, "settings": settings(root)}, a.json,
             [f"active: {versions.active_sha(root)[:12] or 'default (packaged)'}", "versions:"]
             + [f"  {'*' if v['active'] else ' '} {v['sha'][:12]}  {v['note']}" + (f"  (lesson {v['lesson']})"
                                                                                  if v["lesson"] else "") for v in rows]
             + [f"[harness] {k} = {v}" for k, v in settings(root).items()])
    elif a.cmd == "check":
        h = read(a.dir) if a.dir else versions.active(root)
        problems = validate(h, laws(root))
        _out({"sha": h.sha, "problems": problems}, a.json,
             [f"{h.sha[:12]}: " + ("valid" if not problems else f"{len(problems)} problem(s)")] + problems)
        return 1 if problems else 0
    elif a.cmd == "versions":
        _out(versions.stored(root), True)
    elif a.cmd == "export":
        print(versions.export(root, a.ref, a.dest))
    elif a.cmd == "install":
        _out(versions.install(root, a.dir, note=a.note), True)
    elif a.cmd == "diff":
        print(versions.diff(root, a.a, a.b), end="")
    elif a.cmd == "use":
        sha = versions.resolve(root, a.ref)
        details = [f"was {versions.active_sha(root)[:12] or 'default'}"] + versions.diff(
            root, versions.active_sha(root), sha).splitlines()[:60]
        _out(_person(f"make harness {sha[:12] or 'default'} active: {a.reason}", details,
                     lambda: versions.use(root, a.ref, a.reason)), True)
    elif a.cmd == "rollback":
        _out(_person(f"roll the harness back: {a.reason}", [], lambda: versions.rollback(root, a.reason)), True)
    elif a.cmd == "signals":
        from .signals import FIELDS, groups, of, runs, summary
        if a.agent:
            s = of(root, a.agent)
            _out(s, a.json, [f"{s['agent']} ({s['role']}, {s['status']}), harness {s['harness'][:12]}, "
                             f"harness_sha {s['harness_sha'][:12]}"]
                 + [f"  {k:<22} {v!s:>9}  {FIELDS[k][1]}" for k, v in s["signals"].items()])
        else:
            g = {k: {"runs": len(v), "summary": summary(v)} for k, v in groups(runs(root, role=a.role), a.by).items()}
            _out(g, a.json, [f"{k[:12] or '(none)'}: {v['runs']} run(s); "
                             + ", ".join(f"{f} {s['mean']}" for f, s in v["summary"].items() if s["mean"] is not None)
                             for k, v in g.items()])
    elif a.cmd == "contrast":
        from .signals import contrast, runs, save, table
        rows = runs(root, role=a.role)
        pick = lambda ref: [r for r in rows if (r.get(a.key) or "").startswith(ref)]        # noqa: E731
        c = contrast(pick(a.ref), pick(a.cand))
        cid = save(root, c, note=f"{a.key} {a.ref} vs {a.cand}" + (f", role {a.role}" if a.role else "")) \
            if a.save else ""
        _out({"id": cid, **c}, a.json, table(c) + ([f"stored as {cid}"] if cid else []))
    elif a.cmd == "transcript":
        from .signals import transcript
        print(transcript(root, a.agent))
    elif a.cmd == "lessons":
        from . import lessons
        if a.id:
            _out({**lessons.get(root, a.id), "status": lessons.status(root, a.id), "track": lessons.track(root, a.id)},
                 True)
        else:
            rows = lessons.listing(root)
            _out(rows, a.json, [f"{r['id']}  {r['status']}{' (stale)' if r['stale'] else ''}  {r['component']}: "
                                f"{r['advice']}" for r in rows] or ["no harness lessons"])
    elif a.cmd == "confirm":
        from . import lessons
        _out(lessons.confirm(root, a.id), True)
    elif a.cmd == "retire":
        from . import lessons
        _out(_person(f"retire harness lesson {a.id}: {a.reason}", [lessons.get(root, a.id)["advice"]],
                     lambda: lessons.retire(root, a.id, a.reason)), True)
    elif a.cmd == "reflect":
        from .reflect import evidence, spawn_reflector
        ev = evidence(root, a.agent)
        _out(spawn_reflector(root, a.agent, ev["dir"], float(settings(root)["reflector_budget_usd"]),
                             f"reflect:{a.agent}"), True)
    elif a.cmd == "refine":
        from .refine import budget, start
        b = budget(root)
        _out(_person(f"refine the harness on {a.unit} {a.source} (k={a.k})",
                     [f"budget ${b['usd']:g}: 4k rollouts, 3k editor and composer sessions, a reflector",
                      "on probe copies under .probes/; the study is not touched"],
                     lambda: start(root, a.unit, Path(a.source).resolve(), a.prompt, a.k, step=a.resume,
                                   selector_cli=a.selector_cli)), True)
    elif a.cmd == "replay":
        from .replay import nodes, screen
        rows = screen(nodes(root))
        _out(rows, a.json, [f"{r['policy']:<32} {r['value']!s:>8}  over {r['steps']} graded step(s)" for r in rows]
             + ["(a pre-screen on recorded steps only, never an acceptance)"])
    elif a.cmd == "cost":
        _out(cost(root), True)
    return 0


def cost(root) -> dict:
    """Harness overhead as its own line: the spend of reflectors and refinement agents (by their launch tag), and
    the evaluator's verdicts with cost per accepted edit."""
    from ..agents.store import Store
    from .evaluate import cost_per_accepted
    from .versions import AGENT
    store, by = Store(root), {}
    for ag in store.agents():
        try:
            tag = json.loads((store.agent_dir(ag["id"]) / "launch.json").read_text()).get("tag", "")
        except (OSError, ValueError):
            continue
        if tag.startswith(("refine:", "reflect:")):
            kind = tag.split(":", 1)[0]
            by[kind] = round(by.get(kind, 0.0) + float((ag["spent"] or {}).get("usd") or 0), 4)
    evals = [e["payload"] for e in store.events(AGENT, n=10 ** 6, kinds=("harness_eval",))]
    return {"overhead_usd": by, "evaluations": cost_per_accepted(evals)}


if __name__ == "__main__":
    sys.exit(main())
