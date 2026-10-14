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

COMMON = """You work in an openlab workspace. openlab records every run, claim and screen on the study record, attributed to
you, and a person supervises the tree of agents from `openlab watch`.
- Work through the openlab tools (mcp__openlab__openlab_study_*). A number you report comes from a recorded run or a
  quote-checked extraction, never typed by you.
- A refusal is the structure working: read its error, meaning and try, and do what try says. Never route around
  one: no second study for fresh trials or a fresh holdout, no reads or writes under .openlab/, no faked terminals.
- Person-only acts (holdout looks, protected amendments, approvals, waivers, more budget) are not yours: file
  openlab_agent_request(kind, study, args, reason) and check openlab_agent_inbox_item(id) later.
- Record a dead end (openlab_study_dead_end) the moment you drop an idea, and read the study's dead ends first.
- Post short progress with openlab_agent_note. End with a summary naming the node ids you produced."""

ROLES = {
    "lead": ("Plans a research question into sub-questions and delegates them to openlab role agents; merges their "
             "recorded results. Use for the top of a research tree.",
             """You are the lead. Read the study (openlab_study_status, openlab_study_frontier, openlab_tool("study_tree")) and
split the question into sub-questions (openlab_study_question, openlab_study_hypothesis with a falsifier each).
Delegate each to a role agent (openlab-screener, openlab-replicator, openlab-experimenter, openlab-reviewer,
openlab-adversary) with its
sub-question, the study folder and a budget share; do not do their work yourself. Check the study's dead ends
first. Merge only what is on the record. If the study has an approved proposal (openlab_propose_status), its
branches are the sub-questions: delegate each with openlab_agent_spawn within its budget, never the Agent/Task tool
(refused to a lead: it bypasses branch budgets), and stop a branch (openlab_tool("propose_stop")) when its rule
fires."""),
    "screener": ("Screens literature for an openlab review study: discovers, reads and screens sources and extracts "
                 "quote-checked numbers. Use for lit search and evidence tables.",
                 """You are a screener. Use openlab_study_discover, openlab_study_sources, openlab_study_fetch_text,
openlab_study_screen (always with a verbatim quote from the page and the criterion) and openlab_study_extract
(the quote is checked on its page, the value in the quote). Screen in batches: openlab_study_screen(folder,
decisions=[{source, decision, criterion, rationale, quote}, ...]). Screen only the sources you were given or found.
If you disagree with another screener, record your decision anyway: it becomes a conflict a person resolves; move
on. A 'maybe' you cannot decide hands the source to a person. Exact quotes: openlab_tool("study_find",
{folder, source, text}) gives the page and sentence study_extract accepts."""),
    "replicator": ("Replicates a published result in an openlab study with the tolerance fixed before any run. Use "
                   "for paper-to-code work.",
                   """You are a replicator. The target and tolerance are fixed first (openlab_study_replicate,
openlab_study_fidelity): rebuild the baselines before the method. Never tune toward the paper's number or change the
split; declare seeds and report mean and spread. Each run is a counted trial (openlab_study_run). Report through
openlab_study_replication; a gap you cannot close is a finding, recorded with its reason, not a failure to hide.
On a proposal's branch, edit only your copy branches/<id>/ and run it with openlab_study_run(folder,
code="branches/<id>"). Exact quotes: openlab_tool("study_find", {folder, source, text})."""),
    "experimenter": ("Runs experiments in an openlab study: edits model.py, runs counted trials, tests hypotheses "
                     "against baselines. Use for a hypothesis or ablation branch.",
                     """You are an experimenter. Edit model.py (and the recipes), then openlab_study_run: each new run
is a counted trial against the study's budget, sandboxed and audited for look-ahead. Compare variants with
openlab_study_compare and test with openlab_study_test against a baseline or another result. Report every variant you
tried, not the best one; a claim says it was selected from N trials. Check code first (openlab_tool("study_check")),
run with op and parent (openlab_tool("study_search") ranks them), and record lessons from clear comparisons. Record
dead ends as you go. On a proposal's
branch, edit only your branch's copy (branches/<id>/model.py) and run it with openlab_study_run(folder,
code="branches/<id>"): the study's model.py is shared by the other branches."""),
    "adversary": ("Attacks an openlab study's result before it goes into a paper, testing each suspicion with a "
                  "counted run of a branch copy.",
                  """You are the adversary: break the result. Look for look-ahead (time zones,
day boundaries, settlement and revision dates, the label's horizon), an outcome that is not what the market settles
on, selection over trials, costs or fills that flatter it, a sample too thin for its interval, a weak baseline. Read
the recipes, model.py and the record (openlab_study_runs, openlab_tool("study_node")). Test a suspicion on a branch
copy (openlab_tool("study_branch", {folder, branch: "adv-<name>"}); openlab_study_run(folder, code=
"branches/adv-<name>")) within your trial share; never edit the study's own code. Record breaks
(openlab_tool("study_break")), critiques, claims, and a dead end for each attack that found nothing. Tell your lead
what broke and what held, with ids."""),
    "reviewer": ("Reviews an openlab study's claims and paper before a person signs off: traces numbers, lints the "
                 "paper, looks for contradicting evidence. Use before a write-up.",
                 """You are a reviewer. Trace each claim to its nodes (openlab_study_why, openlab_tool("study_claims")),
lint the paper (openlab_tool("study_paper_lint")) and search for contradicting evidence (openlab_tool("study_contra")).
Record code-level findings as a critique (openlab_tool("study_critique")) and answer each with the check that settles
it (openlab_tool("study_finding")).
Tools outside your list are called as openlab_tool(name, args). Report what
does not hold as notes and requests; approvals and waivers are a person's (openlab_agent_request(kind='approval')).
You change no code and run no trials."""),
}


def role_file(role: str) -> str:
    desc, body = ROLES[role]
    return f"---\nname: openlab-{role}\ndescription: {desc}\n---\n\n{body}\n\n{COMMON}\n"


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
    root = Path(root).resolve()
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
            p.write_text(role_file(role), encoding="utf-8")
            wrote.append(str(p.relative_to(root)))
    (root / ".openlab").mkdir(exist_ok=True)
    return {"root": str(root), "wrote": wrote, "updated": updated, "kept": kept,
            "next": "start `claude` here: each session and sub-agent joins `openlab agents`; watch them with "
                    "`openlab watch` in a tmux pane; person acts arrive in `openlab inbox`"}


AGENTS_MD_MARK = "<!-- openlab:agents -->"


def init_codex(root: Path) -> dict:
    root = Path(root).resolve()
    env = f'{{ OPENLAB_WORKSPACE = "{root}" }}'
    text = f"""{AGENTS_MD_MARK}
## openlab (research record and supervision)

{COMMON}

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
