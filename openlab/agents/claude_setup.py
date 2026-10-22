"""`openlab init --claude` and `openlab init --codex`: connect the person's agent CLI to the control plane.

--claude writes or merges, idempotently and keeping everything of the person's:
* `.claude/settings.json`: the openlab hooks (`python -m openlab.agents.hook <Event>`); any earlier openlab hook is
  replaced, every other hook is kept;
* `openlab.toml`: the workspace marker, with the [agents] defaults (stuck minutes, tree limits, default budgets);
* `.mcp.json`: the openlab MCP server (added beside the person's servers, never over one named openlab);
* `.claude/agents/openlab-{lead,screener,replicator,experimenter,reviewer}.md`: role definitions a lead can
  delegate to (an existing file is kept).
--codex writes an AGENTS.md section with the contract and the MCP config snippet for ~/.codex/config.toml.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from ..harness import load as load_harness

HOOK_MARK = "openlab.agents.hook"


def is_hook_command(command: str) -> bool:
    """A hook command openlab wrote: `python -m openlab.agents.hook EVENT`, `openlab hook EVENT` (openlab on PATH), or
    the $OPENLAB_HOOK_COMMAND launcher."""
    command = str(command or "")
    launcher = os.environ.get("OPENLAB_HOOK_COMMAND")
    return (HOOK_MARK in command or re.search(r"(^|[/\s])openlab hook ", command) is not None
            or bool(launcher and command.startswith(launcher + " ")))

#: (event, matcher or None, timeout s)
EVENTS = (("SessionStart", None, 10), ("SubagentStart", None, 10), ("SubagentStop", None, 10),
          ("PreToolUse", "*", 10), ("PostToolUse", "mcp__openlab__.*|Bash", 10), ("PostToolUseFailure", "*", 10),
          ("Stop", None, 10), ("Notification", None, 10), ("UserPromptSubmit", None, 10), ("SessionEnd", None, 5))

TOML = """# openlab workspace (openlab.toml): the folder whose studies and agents openlab manages together.
# Edited by a person only: the openlab hooks refuse an agent's edits of this file.

[agents]
stuck_minutes = 5          # openlab watch flags STUCK after this long with no event from a working agent
session_timeout_hours = 12 # a Claude Code session or sub-agent with no event this long is ended (its leases free)
max_depth = 3              # how deep a tree of delegated agents may go
max_children = 6           # live children per agent
loop_refusals = 3          # the same refusal this many times is flagged LOOP
budget_usd = 2.0           # default budget of a spawned agent ($, wall minutes, turns)
budget_minutes = 60
budget_turns = 200
cap_usd = 20.0             # the most one research proposal (its lead and every branch) may budget ($, minutes)
cap_minutes = 480
permission_mode = "acceptEdits"
# model = "haiku"          # the model spawned agents run on (claude --model / codex -m); unset: the CLI's default
# allowed_tools = ["mcp__openlab", "Read", "Edit", "Write", "Glob", "Grep", "Bash", "TodoWrite", "WebSearch", "WebFetch"]
allow_native_subagents = false  # spawned and budgeted agents delegate with openlab_agent_spawn, not Agent/Task
# notify = ["notify-send", "openlab"]  # run with the message as its last argument when an agent asks the person
#                                     # (an inbox request) and when a spawned agent ends; OPENLAB_EVENT says which
# bash_attribution = "input"   # a sub-agent's openlab Bash call gets OPENLAB_AGENT (input | allow | off)
"""

COMMON = load_harness().text("common.md")

#: role -> (description, body): the role files `init --claude` writes and the role a spawned agent is briefed as
ROLES = load_harness().roles


def role_file(role: str, h=None) -> str:
    """The Claude Code agent file of an openlab role, from harness `h` (default: the packaged one)."""
    desc, body = (h.roles if h else ROLES)[role]
    return f"---\nname: openlab-{role}\ndescription: {desc}\n---\n\n{body}\n\n{h.text('common.md') if h else COMMON}\n"


def hooks_settings() -> dict:
    from .spawn import hook_command
    out = {}
    for event, matcher, timeout in EVENTS:
        entry = {"hooks": [{"type": "command", "command": hook_command(event), "timeout": timeout}]}
        if matcher:
            entry = {"matcher": matcher, **entry}
        out[event] = [entry]
    return out


def _is_ours(entry: dict) -> bool:
    return any(is_hook_command(h.get("command", "")) for h in entry.get("hooks", []) if isinstance(h, dict))


def merge_settings(existing: dict) -> dict:
    """The person's settings with the openlab hooks in: earlier openlab entries replaced, the rest kept in place."""
    out = dict(existing or {})
    hooks = {k: list(v) for k, v in (out.get("hooks") or {}).items()}
    for event, entries in hooks_settings().items():
        hooks[event] = [e for e in hooks.get(event, []) if not (isinstance(e, dict) and _is_ours(e))] + entries
    out["hooks"] = hooks
    return out


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def init_claude(root: Path) -> dict:
    from ..harness.versions import active
    root = Path(root).resolve()
    harness = active(root)
    wrote, kept, updated = [], [], []
    sp = root / ".claude" / "settings.json"
    try:
        cur = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    except ValueError as e:
        raise SystemExit(f"{sp} is not valid JSON ({e}); fix it and run again") from None
    new = merge_settings(cur)
    if new != cur:
        _write_json(sp, new)
        (updated if cur else wrote).append(".claude/settings.json")
    else:
        kept.append(".claude/settings.json")
    toml = root / "openlab.toml"
    if not toml.exists():
        toml.write_text(TOML, encoding="utf-8")
        wrote.append("openlab.toml")
    elif "[agents]" not in toml.read_text(encoding="utf-8"):
        with open(toml, "a", encoding="utf-8") as f:
            f.write("\n" + TOML[TOML.index("[agents]"):])
        updated.append("openlab.toml")
    else:
        kept.append("openlab.toml")
    mp = root / ".mcp.json"
    try:
        mcp = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    except ValueError:
        mcp = None
    if mcp is None:
        kept.append(".mcp.json (not JSON; left alone)")
    elif "openlab" in (mcp.get("mcpServers") or {}):
        kept.append(".mcp.json")
    else:
        from .spawn import mcp_config
        mcp.setdefault("mcpServers", {})["openlab"] = mcp_config(root, launcher=True)["mcpServers"]["openlab"]
        _write_json(mp, mcp)
        (updated if mp.exists() and len(mcp["mcpServers"]) > 1 else wrote).append(".mcp.json")
    for role in ROLES:
        p = root / ".claude" / "agents" / f"openlab-{role}.md"
        if p.exists():
            kept.append(str(p.relative_to(root)))
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(role_file(role, harness), encoding="utf-8")
            wrote.append(str(p.relative_to(root)))
    (root / ".openlab").mkdir(exist_ok=True)
    return {"root": str(root), "wrote": wrote, "updated": updated, "kept": kept,
            "next": "start `claude` here: each session and sub-agent joins `openlab agents`; watch them with "
                    "`openlab watch` in a tmux pane; person acts arrive in `openlab inbox`"}


AGENTS_MD_MARK = "<!-- openlab:agents -->"


def init_codex(root: Path) -> dict:
    from ..harness.versions import active
    root = Path(root).resolve()
    common = active(root).text("common.md")
    env = f'{{ OPENLAB_WORKSPACE = "{root}" }}'
    text = f"""{AGENTS_MD_MARK}
## openlab (research record and supervision)

{common}

Codex connects to openlab's tools through MCP. Add to ~/.codex/config.toml:

```toml
[mcp_servers.openlab]
command = "{sys.executable}"
args = ["-m", "openlab.mcp_server"]
env = {env}
```

Identity: set `OPENLAB_AGENT=<name>` in the environment you start codex from (or spawn it with
`openlab agent spawn --cli codex ...`, which sets it), so your record writes name you.
"""
    p = root / "AGENTS.md"
    cur = p.read_text(encoding="utf-8") if p.exists() else ""
    if AGENTS_MD_MARK in cur:
        return {"root": str(root), "wrote": [], "kept": ["AGENTS.md"]}
    p.write_text((cur.rstrip() + "\n\n" if cur else "") + text, encoding="utf-8")
    return {"root": str(root), "wrote": ["AGENTS.md"] if not cur else [], "updated": ["AGENTS.md"] if cur else [],
            "kept": []}
