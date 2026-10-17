"""`openlab instrument register|list|show|verify|run|runs|output`: the workspace's instruments from a shell.

`register` is a person's act: it fingerprints the instrument, probes its usage (`--help`), shows exactly what the
version holds (its files, its environment parts, its capabilities, its usage) and asks for the person's passphrase,
which signs the registration. `run` of a live-order instrument is a person's act too, signed for that one run.
Everything else is open to agents: `show` prints a version's usage, `run` prints the run's exit code, its redacted log
tails and its small text outputs, and `output` reads a recorded run's back.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from . import instruments
from .errors import StudyError


def _root(a) -> Path:
    return instruments.workspace_root(a.root or None)


def _person(summary: str, details):
    from ..agents import person
    from .personkey import signing
    try:
        signer = person.unlock_interactive(summary, details)
    except person.NotAPerson as e:
        raise StudyError(f"{summary} is a person's act, signed with the person key: {e}",
                         "the passphrase is the person's confirmation; an agent's shell does not have it",
                         "the person runs it in their own terminal; an agent files agent_request(kind=\"instrument\" "
                         "or \"instrument_run\") with the exact args") from None
    return signing(signer)


def _register(a) -> dict:
    kw = dict(source=a.source or "", runtime=a.runtime or "", python=a.python or "", node=a.node or "",
              binaries=a.binary, env_files=a.env_file, env_cmds=a.env_cmd, env=a.env, capabilities=a.capabilities,
              secrets=a.secrets, config=a.config or "", outputs=a.outputs, state=a.state, timeout=a.timeout,
              include=a.include, exclude=a.exclude, redact=a.redact, version_probe=not a.no_version_probe,
              usage=not a.no_usage, usage_args=a.usage_arg, usage_sub=a.usage_sub, usage_file=a.usage_file or "")
    prep = instruments.prepare(_root(a), a.name, a.entry, **kw)
    with _person(f"register instrument {prep.ident}", [*instruments.preview(prep), f"reason: {a.reason}"]):
        node = instruments.commit(_root(a), prep, a.reason)
    return {"id": node.id, "instrument": prep.ident, "snapshot": bool(prep.manifest["files"]),
            "files": len(prep.manifest["files"]), "capabilities": prep.manifest["capabilities"]}


def _list(a) -> dict:
    return {"instruments": instruments.listing(_root(a))}


def _show(a) -> dict:
    reg = instruments.Registry(_root(a))
    node = instruments.resolve(reg, a.ref)
    usage = instruments.usage_view(reg, node.payload)
    if a.usage:
        return {"instrument": instruments.ident(node.payload), "usage": usage, "only": True}
    out = {"id": node.id, "at": node.at, "payload": node.payload, "usage": usage}
    if a.verify:
        out["verify"] = instruments.verify(reg, node)
    return out


def _verify(a) -> dict:
    if not a.all and not a.ref:
        raise StudyError("name a version, or --all")
    reg = instruments.Registry(_root(a))
    nodes = instruments.versions(reg) if a.all else [instruments.resolve(reg, a.ref)]
    res = [instruments.verify(reg, n) for n in nodes]
    return {"ok": all(r["ok"] for r in res), "versions": res}


def _run(a) -> dict:
    root = _root(a)
    reg = instruments.Registry(root)
    st = instruments._open_study(reg, a.study) if a.study else None
    node = instruments.resolve(reg, a.ref, st)
    args = list(a.args or [])
    if set(node.payload["capabilities"]) & set(instruments.PERSON_CAPABILITIES):
        if not (a.reason or "").strip():
            raise StudyError("a live-order run needs a reason", fix="--reason says why this run")
        with _person(f"run {instruments.ident(node.payload)} (live-order)",
                     [f"args: {shlex.join(args) or '(none)'}", f"study: {a.study or '(none)'}", f"reason: {a.reason}",
                      "your signature authorises this one run"]):
            return instruments.run(root, a.ref, args, study=a.study or "", note=a.note or "", timeout=a.timeout,
                                   reason=a.reason)
    return instruments.run(root, a.ref, args, study=a.study or "", note=a.note or "", timeout=a.timeout,
                           reason=a.reason or "")


def _output(a) -> dict:
    return instruments.output(_root(a), a.run, a.file or "")


def _runs(a) -> dict:
    if not instruments.Registry.exists(_root(a)):
        return {"runs": []}
    return {"runs": instruments.runs(instruments.Registry(_root(a)), a.ref or "", a.study or "", a.n)}


VERBS = {"register": _register, "list": _list, "show": _show, "verify": _verify, "run": _run, "runs": _runs,
         "output": _output}


def _text_list(out: dict) -> list:
    if not out["instruments"]:
        return ["no instruments registered here (a person registers one: openlab instrument register NAME ...)"]
    lines = []
    for i in out["instruments"]:
        for v in i["versions"]:
            lines.append(f"{v['id']:<34} {','.join(v['capabilities']):<26} {v['runtime']:<7} "
                         f"{'external' if v['external'] else str(v['files']) + ' files':<10} {v['runs']:>4} runs  "
                         f"by {v['by']}  {v['reason'][:50]}")
    return lines


def _text_verify(out: dict) -> list:
    lines = []
    for v in out["versions"]:
        lines.append(f"{'ok' if v['ok'] else 'DRIFT'}: {v['instrument']}"
                     + (" (external: verified, never copied)" if v["external"] else "")
                     + (" (snapshot rebuilt from the content store)" if v["rebuilt"] else ""))
        lines += [f"  file {x}" for x in v["files"]] + ([f"  {v['config']}"] if v["config"] else [])
        for e in v["environment"]:
            lines.append(f"  {'ok' if e['ok'] else 'CHANGED'} {e['part']}")
            lines += [f"    {c}" for c in e["changes"][:12]]
    return lines


def _text_usage(u: dict, indent: str = "  ") -> list:
    return [indent + u["head"] + (":" if u["text"] else "")] + [indent + "  " + x for x in u["text"].splitlines()]


def _text_show(out: dict) -> list:
    if out.get("only"):
        u = out["usage"]
        return [u["text"].rstrip("\n")] if u["text"] else [u["head"]]
    m = out["payload"]
    lines = [f"{instruments.ident(m)}  ({out['id']}, {out['at']})", f"  reason: {m['reason']}",
             f"  entry: {shlex.join(m['entry'])}", f"  runtime: {m['runtime']}",
             f"  capabilities: {', '.join(m['capabilities'])}",
             f"  secrets: {', '.join(m['secrets']) or 'none'}; outputs: {', '.join(m['outputs']) or 'none'}; "
             f"state: {'yes' if m['state'] else 'no'}",
             f"  code: {len(m['files'])} files from {m['source']}" if m["files"] else
             "  code: none (an external command: verified, never copied)"]
    lines += [f"  environment: {p['kind']} {p['spec'].get('path') or p['spec'].get('cmd')}  {p['sha'][:12]}"
              for p in m["environment"]["parts"]]
    lines += _text_usage(out["usage"])
    if "verify" in out:
        lines += _text_verify({"versions": [out["verify"]]})
    return lines


def _text_run(out: dict) -> list:
    p = out["payload"]
    lines = [f"{out['id']}: {p['instrument']} exit {p['exit']}" + (" (timed out)" if p["timed_out"] else "")
             + f", {p['started']} -> {p['ended']}", "  outputs: " + (", ".join(
                 f"{k} ({v['bytes']} B, {v['sha256'][:12]})" for k, v in p["outputs"].items()) or "none"),
             f"  read it again: openlab instrument output {out['id']} [FILE]"]
    if out.get("study_run"):
        lines.append(f"  on the study's record: {out['study_run']}")
    if out.get("authority"):
        lines.append(f"  under the person's authority {out['authority']}")
    return lines + _text_view(out)


def _text_view(out: dict) -> list:
    """A run's returned output: the redacted log tails, then each declared output (its content, or why not)."""
    lines = []
    for k in ("stdout", "stderr"):
        t = out[k]
        if not t["text"]:
            lines.append(f"--- {k}: empty ---")
            continue
        shown = len(t["text"].splitlines())
        lines.append(f"--- {k}" + (f" (the last {shown} of {t['lines']} lines, {t['bytes']} B)" if t["truncated"]
                                   else f" ({t['lines']} lines)") + " ---")
        lines += t["text"].rstrip("\n").split("\n")
    for rel, o in out["outputs"].items():
        if "content" in o:
            lines.append(f"--- {rel} ({o['bytes']} B, sha256 {o['sha256'][:12]}) ---")
            lines += o["content"].rstrip("\n").split("\n") if o["content"] else []
        else:
            lines.append(f"--- {rel} ({o['bytes']} B, sha256 {o['sha256'][:12]}): {o['withheld']} ---")
    return lines


def _text_output(out: dict) -> list:
    if "file" in out:
        return [out["content"].rstrip("\n")]
    return [f"{out['id']}: {out['instrument']} exit {out['exit']}" + (" (timed out)" if out["timed_out"] else "")
            + f", {out['started']} -> {out['ended']}, args {shlex.join(out['args']) or '(none)'}"] + _text_view(out)


def _text_runs(out: dict) -> list:
    return [f"{r['id']}  {r['payload']['instrument']:<34} exit {r['payload']['exit']:>4}  {r['payload']['ended']}  "
            f"{shlex.join(r['payload']['args'])[:60]}" + (f"  [{r['payload']['study']}]" if r['payload']['study']
                                                            else "") for r in out["runs"]] or ["no runs recorded"]


TEXT = {"register": lambda o: [f"registered {o['instrument']} ({o['id']}): {o['files']} files, "
                               f"capabilities {', '.join(o['capabilities'])}"],
        "list": _text_list, "show": _text_show, "verify": _text_verify, "run": _text_run, "runs": _text_runs,
        "output": _text_output}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="openlab instrument", description="The workspace's instruments: registered, "
                                "immutable programs that measure or act outside the sandbox.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default="", help="the workspace (default: the one the current folder is in)")
    common.add_argument("--json", action="store_true", help="the result as one JSON object")
    sub = p.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("register", parents=[common], help="register a version (a person's act: asks for the "
                       "passphrase)")
    r.add_argument("name")
    r.add_argument("--entry", required=True, help='the command, as you would type it: "python -m pkg", '
                   '"node run.js", "./collect.sh", "mycli --flag"')
    r.add_argument("--source", default="", help="the instrument's code folder, snapshotted (none: an external "
                   "command, verified but never copied)")
    r.add_argument("--runtime", default="", choices=("", *instruments.RUNTIMES), help="default: from the entry")
    r.add_argument("--python", default="", help="an interpreter to fingerprint (default for a python entry: its own)")
    r.add_argument("--node", default="", help="a node binary to fingerprint")
    r.add_argument("--binary", action="append", default=[], help="another binary to fingerprint (repeatable)")
    r.add_argument("--env-file", action="append", default=[], help="a lockfile or any file whose sha256 is part of "
                   "the environment (repeatable)")
    r.add_argument("--env-cmd", action="append", default=[], help="a command whose stdout is hashed into the "
                   "environment, e.g. \"uv pip freeze\" (repeatable)")
    r.add_argument("--env", action="append", default=[], help="a fixed variable K=V the instrument runs with "
                   "(repeatable)")
    r.add_argument("--capabilities", default="read", help=f"comma-separated: {', '.join(instruments.CAPABILITIES)}")
    r.add_argument("--secrets", default="", help="comma-separated environment variables passed through by name, "
                   "never recorded")
    r.add_argument("--config", default="", help="a config file, copied and hashed into the version")
    r.add_argument("--outputs", default="", help="comma-separated globs of the files a run writes (in its output "
                   "folder) that are kept and hashed")
    r.add_argument("--state", action="store_true", help="give the instrument a state folder kept across runs")
    r.add_argument("--timeout", type=float, default=None, help=f"seconds a run may take (default "
                   f"{instruments.TIMEOUT:g})")
    r.add_argument("--include", action="append", default=[], help="a glob to snapshot even if excluded")
    r.add_argument("--exclude", action="append", default=[], help="a glob to leave out of the snapshot")
    r.add_argument("--redact", action="append", default=[], help="a regex masked in the stored logs (repeatable)")
    r.add_argument("--no-version-probe", action="store_true", help="do not hash a binary's --version output")
    r.add_argument("--usage-arg", action="append", default=[], help="what the usage probe adds to the entry "
                   "(repeatable, in order; default --help), e.g. help or -h")
    r.add_argument("--usage-sub", action="store_true", help="also probe each subcommand an argparse-style help lists "
                   f"as {{a,b,c}} (at most {instruments.USAGE_MAX_SUBCOMMANDS}, "
                   f"{instruments.USAGE_MAX_BYTES // 1024} KB in all)")
    r.add_argument("--usage-file", default="", help="a text file with the usage, instead of probing")
    r.add_argument("--no-usage", action="store_true", help="record no usage (for an entry that does not honour "
                   "--help: the probe runs it, in a throwaway copy, before you sign)")
    r.add_argument("--reason", required=True, help="what the instrument is for")
    sub.add_parser("list", parents=[common], help="every registered version")
    s = sub.add_parser("show", parents=[common], help="one version's manifest")
    s.add_argument("ref", help="NAME@VERSION (or NAME with one version)")
    s.add_argument("--verify", action="store_true", help="also check every hash again")
    s.add_argument("--usage", action="store_true", help="print only the version's usage")
    v = sub.add_parser("verify", parents=[common], help="check a version's files, config and environment again")
    v.add_argument("ref", nargs="?", default="")
    v.add_argument("--all", action="store_true", help="every version")
    u = sub.add_parser("run", parents=[common], help="run a version (verified first) and record the run; "
                       "arguments after --")
    u.add_argument("ref", help="NAME@VERSION, NAME with one version, or a study's alias with --study")
    u.add_argument("--study", default="", help="also record the run on this study (it must pin the version)")
    u.add_argument("--note", default="")
    u.add_argument("--reason", default="", help="why this run (required for a live-order instrument)")
    u.add_argument("--timeout", type=float, default=None)
    u.add_argument("args", nargs="*", help="after --: the instrument's own arguments")
    o = sub.add_parser("output", parents=[common], help="a recorded run's redacted logs and outputs, each sha "
                       "checked first")
    o.add_argument("run", help="the run's id (as `run` and `runs` print it)")
    o.add_argument("file", nargs="?", default="", help="one declared output, or stdout / stderr, printed whole "
                   f"(text, at most {instruments.READ_MAX // 2 ** 20} MB)")
    n = sub.add_parser("runs", parents=[common], help="recorded runs, newest first")
    n.add_argument("ref", nargs="?", default="")
    n.add_argument("--study", default="")
    n.add_argument("-n", type=int, default=20)
    return p


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    rest: list = []
    if "--" in argv:                                # the instrument's own arguments, never openlab's options
        i = argv.index("--")
        argv, rest = argv[:i], argv[i + 1:]
    a = _parser().parse_args(argv)
    if a.verb == "run":
        a.args = list(a.args or []) + rest
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
    if a.verb == "verify" and not out["ok"]:
        return 1
    if a.verb == "run" and out["payload"]["exit"] != 0:
        return 4
    return 0
