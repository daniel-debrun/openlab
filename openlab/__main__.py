"""``python -m openlab`` / ``openlab``: the research framework's command line.

    openlab study new|materialise|run|status|claim|paper|replicate|... FOLDER    a study (openlab study --help)
    openlab init [DIR]                    OPENLAB.md, openlab.toml and .mcp.json (--claude: hooks and roles)
    openlab app --root DIR                the workspace server: the web UI and the tool API
    openlab agents | agent | watch | inbox  the agents working on the studies, and what needs a person
    openlab instrument register|list|show|verify|run|runs  registered programs that act outside the sandbox
    openlab guide | tools | tool NAME     the tools your agent has, from a shell
    openlab doctor [--json]               what works here, and the one-line fixes
    openlab verify FILE.sig.json          check a signed file offline

Every subcommand is one row of COMMANDS, imported only when it runs: `openlab study ...` never loads the LaTeX
engine (sympy, lark) or the web server.
"""

from __future__ import annotations

import importlib
import sys

from . import __version__

#: name -> ("module:function", one line of help). The function takes the arguments after the name (all of argv for a
#: "module:function+" target, which reads the name itself) and returns the exit code. A new subcommand (e.g.
#: `agents`) is one row here.
COMMANDS: dict = {
    "study": ("openlab.study.cli:main", "a study: new, materialise, run, status, claim, paper, replicate, ... "
                                       "(openlab study --help)"),
    "propose": ("openlab.study.propose_cli:main", "research from a prompt: new, lint, approve (a person's), execute, "
                                                 "status (openlab propose --help)"),
    "replicate": ("openlab.study.paperplan_cli:main", "a paper turned into checked code: openlab replicate ARXIV_ID "
                                                     "(an agent drafts the plan, you approve it, agents build it)"),
    "agents": ("openlab.agents.cli:main+", "the agents, as a table or a tree: role, study, spend, trials, claims, refusals"),
    "agent": ("openlab.agents.cli:main+", "one agent: spawn, status, log, pause, resume, stop (openlab agent --help)"),
    "watch": ("openlab.agents.cli:main+", "the live supervision screen for a tmux pane, with its flags"),
    "inbox": ("openlab.agents.cli:main+", "what needs a person, and a person's approve / deny (agents' "
                                         "questions are answered in openlab show)"),
    "init": ("openlab.init:main", "set up a workspace: OPENLAB.md, openlab.toml, .mcp.json (--claude adds hooks and roles)"),
    "app": ("openlab.app.server:main", "the workspace server: web UI and tool API (openlab app --root DIR)"),
    "show": ("openlab.app.server:show_main", "the live view of the agents, their plans, the inbox and the record "
                                            "(openlab show --root DIR; over ssh, forward the port)"),
    "guide": ("openlab.thin_cli:main+", "the agent's guide, from a shell"),
    "tools": ("openlab.thin_cli:main+", "the tools an agent has, by section"),
    "tool": ("openlab.thin_cli:main+", "call one tool: openlab tool NAME [--help | key=value ...]"),
    "env": ("openlab.envcli:main", "the Python a study's code runs in; `openlab env add PKG` installs what a study needs "
                                  "without changing what is installed"),
    "lessons": ("openlab.study.lessons:main", "the workspace's lessons, found by what you are about to try: "
                                             "`openlab lessons find QUERY`"),
    "mirror": ("openlab.study.mirror:main", "the Weights & Biases mirror of a study's runs (never the record): "
                                           "`openlab mirror status|sync FOLDER`"),
    "bench": ("openlab.benchmark:main", "MLE-bench: the same agent with and without openlab, graded on your machine, "
                                       "which keeps the answers: `openlab bench mle --help`"),
    "box": ("openlab.boxcli:main", "this machine as openlab uses it (cores, memory, GPUs, the schedule in box.toml) and "
                                  "its queue of runs: `openlab box queue`"),
    "instrument": ("openlab.study.instrument_cli:main", "the workspace's instruments, registered immutable programs "
                                                       "that measure or act outside the sandbox: register (a person's), "
                                                       "list, show, verify, run, runs"),
    "doctor": ("openlab.doctor:main", "what works in this install, and the one line that fixes what does not"),
    "verify": ("openlab.__main__:verify_main", "check a signed file (FILE.sig.json) offline"),
    "hook": ("openlab.agents.cli:main+", "a Claude Code hook (stdin JSON; written by openlab init --claude)"),
    "person": ("openlab.agents.cli:main+", "your person key: init, show, rotate, pin (it signs every person act)"),
}


def verify_main(argv) -> int:
    """``openlab verify FILE.sig.json [--json]``: the signature over the manifest, and the file beside it unchanged."""
    import argparse
    import json
    from pathlib import Path
    ap = argparse.ArgumentParser(prog="openlab verify", description="Check a signed openlab file offline.")
    ap.add_argument("file", help="a .sig.json beside the file it signs (a forecast, a bundle, a paper)")
    ap.add_argument("--json", action="store_true", help="the result as one JSON object")
    a = ap.parse_args(argv)
    from .study.signing import SIG_SUFFIX, verify_file
    if not a.file.endswith(SIG_SUFFIX):
        print(f"openlab verify: {a.file} is not a {SIG_SUFFIX} file", file=sys.stderr)
        return 2
    try:
        r = verify_file(Path(a.file))
    except ImportError as e:
        print(f"openlab verify: {e.name or e} is not installed: pip install pynacl", file=sys.stderr)
        return 2
    except (OSError, ValueError) as e:
        print(f"cannot read {a.file}: {e}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(r))
    elif r["ok"]:
        run = (f", gate {'passed' if r['gate'].get('passed') else 'FAILED'}" if r.get("gate")
               else f", monitor {r['monitor']}" if r.get("monitor") else "")
        print(f"valid: {r['file']} from {r['study'] or r['notebook']}, key {r['key_id']}"
              + (f", expires {r['expires_at']}" if r.get("expires_at") else "") + run)
    else:
        print(f"INVALID: {r['why']}")
    return 0 if r["ok"] else 1


def _help() -> str:
    width = max(map(len, COMMANDS))
    rows = "\n".join(f"  {name:<{width}}  {what}" for name, (_, what) in COMMANDS.items())
    return (f"openlab {__version__}: a research framework you can trust. Your coding agent (Claude Code, Codex) runs\n"
            "the research; openlab owns the data, the splits, the runner, the counted trials, the person-only holdout\n"
            "looks and the hash-chained record of each study.\n\n"
            f"usage: openlab COMMAND [ARGS]\n\ncommands:\n{rows}\n\n"
            "The MCP server for an agent: openlab-mcp (claude mcp add openlab -- openlab-mcp).")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_help())
        return 0
    if argv[0] in ("-V", "--version", "version"):
        print(f"openlab {__version__}")
        return 0
    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:
        import difflib
        near = difflib.get_close_matches(name, list(COMMANDS), n=3, cutoff=0.7)
        print(f"openlab: no command {name!r}"
              + (f"; did you mean {' or '.join(repr(n) for n in near)}?" if near else "")
              + "\n`openlab --help` lists the commands", file=sys.stderr)
        return 2
    mod, _, fn = COMMANDS[name][0].partition(":")
    whole = fn.endswith("+")
    main_fn = getattr(importlib.import_module(mod), fn.rstrip("+"))
    return main_fn(argv if whole else rest) or 0


if __name__ == "__main__":
    sys.exit(main())
