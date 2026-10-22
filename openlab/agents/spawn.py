"""Start a research agent: register it, write its MCP config, settings and brief, and start a detached supervisor
(openlab.agents.supervisor) that runs the agent CLI and outlives the command that spawned it.

The agent is the person's own CLI under their own login: `claude -p` (stream-json) or `codex exec --json`.
openlab passes no model key. Defaults, overridable per spawn or in `openlab.toml [agents]`:

* permission mode `acceptEdits`: file edits inside the agent's folder are accepted, nothing else is widened;
* allowed tools: the openlab MCP tools, Read/Edit/Write/Glob/Grep, Bash, TodoWrite, WebSearch/WebFetch. Claude
  Code's own sub-agent tools (Agent, Task) are passed as `--disallowedTools Agent,Task`, so a spawned agent
  delegates with openlab_agent_spawn and every child has a budget and a role prompt (`[agents]
  allow_native_subagents = true` gives them back). The person gates hold anyway: the openlab PreToolUse hook runs
  for spawned agents too (`--settings`), and denies person-only verbs, pty wrappers, writes into `.openlab/` and a
  budgeted agent's Agent/Task calls;
* `--strict-mcp-config`: only openlab's MCP server, started with OPENLAB_AGENT=<id> so its writes are attributed;
* budget: $, wall minutes (paused time not counted), turns; at a limit the supervisor pauses the agent and asks the
  person for more (openlab.agents.inbox.hold), so no `--max-budget-usd` / `--max-turns` is passed;
* model: the one asked for (claude `--model`, codex `-m`), else the parent's (a child of the same CLI), else
  `[agents] model`, else the CLI's own default.
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional

from ..harness import load as load_harness
from .store import FINAL, LIVE, Store, config, root_of

CLIS = ("claude", "codex")
#: Claude Code's own sub-agent tools (`Agent`, and its legacy name `Task`): a spawned agent runs with them
#: disallowed, so it delegates through openlab_agent_spawn and the child's budget comes out of its own
NATIVE_SUBAGENT_TOOLS = ("Agent", "Task")


class AgentError(Exception):
    """A refusal with what it means and what to try (the shape of a study error and a tool error)."""

    def __init__(self, error: str, meaning: str = "", try_: str = ""):
        super().__init__(error)
        self.error, self.meaning, self.try_ = error, meaning, try_


#: the spawn brief, the role prompts and the decide-yourself note (openlab.harness)
HARNESS = load_harness()
BRIEF = HARNESS.text("brief.md")


DECIDE = ("", "self")
SELF = HARNESS.text("self.md")


def brief(agent: dict, h=None, advice: str = "") -> str:
    """The system prompt of a spawned agent: the brief of harness `h` (default: the packaged one), its role's prompt,
    the decide-yourself note, and `advice` (approved harness lessons, openlab.harness.lessons.advice)."""
    h = h or HARNESS
    b = agent["budget"]
    parts = [f"${b['usd']:g}" if b.get("usd") else "", f"{b['minutes']:g} min" if b.get("minutes") else "",
             f"{b['turns']} turns" if b.get("turns") else ""]
    return h.text("brief.md").format(
        id=agent["id"], role=agent["role"] or "agent",
        parent_line=f", delegated by {agent['parent']}" if agent["parent"] else "",
        study_line=f"Study: {agent['study']}. " if agent["study"] else "No study is assigned yet. ",
        question_line=f"Your question: {agent['question']}" if agent["question"] else "",
        budget_line=", ".join(p for p in parts if p) or "none set") \
        + "".join(f"\n\n{n}" for n in (_role_note(agent["role"], h),
                                       h.text("self.md") if agent.get("decide") == "self" else "", advice) if n)


def _role_note(role: str, h=None) -> str:
    """What a spawned agent of an openlab role is told beyond the brief: the role's prompt ("" for any other role)."""
    roles = (h or HARNESS).roles
    return roles[role][1] if role in roles else ""


def harness_sha(root: Path, role: str, decide: str, cli: str, launch: dict, cfg: dict, h=None,
                lessons: tuple = ()) -> str:
    """The fingerprint of what briefs and bounds an agent apart from its task: the harness's texts, the role and the
    decide-yourself note it is briefed with, the harness lessons put in its brief, its CLI, tools and permission
    mode, the [agents] deny list and the workspace's OPENLAB.md. Spawns that share it were told the same things
    whatever their prompt, id or budget, so their outcomes can be compared by it."""
    import hashlib
    from ..init import conventions
    told = {"harness": (h or HARNESS).sha, "role": role, "self": decide == "self", "cli": cli,
            **{k: launch[k] for k in ("allowed_tools", "disallowed_tools", "permission_mode")},
            "deny": list(cfg.get("deny") or []), "conventions": conventions(root),
            **({"lessons": sorted(lessons)} if lessons else {})}
    return hashlib.sha256(json.dumps(told, sort_keys=True).encode("utf-8")).hexdigest()


def role_tools(cfg: dict, role: str) -> Optional[list]:
    """A role's own tool allowlist: `[agents.roles.<role>] allowed_tools` in openlab.toml (None: the workspace's)."""
    roles = cfg.get("roles") if isinstance(cfg.get("roles"), dict) else {}
    tools = (roles.get(role) or {}).get("allowed_tools") if isinstance(roles.get(role), dict) else None
    return list(tools) if isinstance(tools, list) and all(isinstance(x, str) for x in tools) else None


def hook_command(event: str) -> str:
    """The shell command a Claude Code hook runs: `openlab hook EVENT` when `openlab` is on PATH (a stable launcher a
    lab can commit; `openlab hook` imports only the hook module), else this interpreter with the module entry point
    (`python -m openlab.agents.hook EVENT`). $OPENLAB_HOOK_COMMAND overrides the launcher (e.g. a venv path)."""
    import shutil
    launcher = os.environ.get("OPENLAB_HOOK_COMMAND")
    if launcher:
        return f"{launcher} {event}"
    if shutil.which("openlab"):
        return f"openlab hook {event}"
    return f"{_q(sys.executable)} -m openlab.agents.hook {event}"


def _q(s: str) -> str:
    import shlex
    return shlex.quote(s)


def mcp_config(root: Path, agent: Optional[str] = None, token: str = "", launcher: bool = False) -> dict:
    """The openlab MCP server entry. `launcher` (the project's .mcp.json, which a lab may commit): `openlab-mcp` when it
    is on PATH, as the hooks use `openlab`; else, and for a spawned agent's own config, this interpreter."""
    import shutil
    env = {"OPENLAB_WORKSPACE": str(root)}
    if agent:
        env["OPENLAB_AGENT"] = agent
    if token:
        env["OPENLAB_AGENT_TOKEN"] = token
    if os.environ.get("PYTHONPATH"):
        env["PYTHONPATH"] = os.environ["PYTHONPATH"]
    if launcher and shutil.which("openlab-mcp"):
        return {"mcpServers": {"openlab": {"command": "openlab-mcp", "args": [], "env": env}}}
    return {"mcpServers": {"openlab": {"command": sys.executable, "args": ["-m", "openlab.mcp_server"], "env": env}}}


def check_child(store: Store, parent: str, budget: dict, cfg: dict) -> None:
    """Refuse a child the tree cannot give: a paused or stopped parent, past max depth or max live children, or a
    budget beyond the parent's remainder ($: its budget less its spend and its children's shares; minutes: its
    wall time left)."""
    p = store.get(parent)
    if p is None:
        return                                                      # an unregistered caller: a top-level agent
    if (held := store.blocked(parent)) is not None:
        raise AgentError(f"{parent} is {held[1]} by a person", f"reason: {held[2] or 'none given'}",
                         "nothing can be delegated until a person resumes it")
    if p["status"] in FINAL:
        raise AgentError(f"{parent} has ended ({p['status']})")
    depth = store.depth(parent) + 1
    if depth > int(cfg["max_depth"]):
        raise AgentError(f"the tree is {cfg['max_depth']} deep at most", f"a child of {parent} would be depth {depth}",
                         "do the sub-question yourself, or ask the person to raise [agents] max_depth")
    kids = [c for c in store.children(parent) if c["status"] in LIVE]
    if len(kids) >= int(cfg["max_children"]):
        raise AgentError(f"{parent} already has {len(kids)} live children", f"the limit is {cfg['max_children']}",
                         "wait for one to finish (openlab_agent_wait) or stop one")
    pb, ps = p["budget"], p["spent"]
    if pb.get("usd"):
        reserved = sum((c["budget"].get("usd") or 0) if c["status"] in LIVE else (c["spent"].get("usd") or 0)
                       for c in store.children(parent))
        left = round(pb["usd"] - (ps.get("usd") or 0) - reserved, 4)
        if not budget.get("usd") or budget["usd"] > left:
            raise AgentError(f"a child's budget is at most its parent's remainder: ${max(left, 0):g}",
                             f"{parent}: budget ${pb['usd']:g}, spent ${ps.get('usd') or 0:g}, "
                             f"${reserved:g} held by its children; asked ${budget.get('usd') or 0:g}",
                             "ask for less, or file openlab_agent_request(kind='budget') for more")
    if pb.get("minutes"):
        from .store import age
        used = age(p["created"]) / 60 - (ps.get("paused_minutes") or 0)
        left = pb["minutes"] - used
        if not budget.get("minutes") or budget["minutes"] > left:
            raise AgentError(f"a child's time is at most its parent's remainder: {max(left, 0):.0f} min",
                             f"asked {budget.get('minutes') or 0:g} min")


def _launch(store: Store, agent: Optional[str]) -> dict:
    """An agent's launch.json ({} when it has none)."""
    if not agent:
        return {}
    f = store.dir / "agents" / agent.replace("/", "_") / "launch.json"
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        return {}


def _inherited_model(store: Store, parent: Optional[str], cli: str) -> str:
    """The model a parent was spawned on (as asked, from its launch.json), for a child of the same CLI."""
    p = store.get(parent) if parent else None
    if not p or p["cli"] != cli:
        return ""
    return _launch(store, parent).get("model") or ""


def spawn(root, prompt: str, role: str = "agent", study: Optional[str] = None, question: Optional[str] = None,
          parent: Optional[str] = None, cli: str = "claude", model: Optional[str] = None,
          budget: Optional[dict] = None, cwd: Optional[str] = None, allowed_tools: Optional[list] = None,
          permission_mode: Optional[str] = None, start: bool = True, decide: str = "", harness: str = "",
          tag: str = "", advise: bool = True, resources: Optional[dict] = None, node: str = "",
          params: Optional[dict] = None) -> dict:
    """Register an agent and start its supervisor (detached). Returns the agent row. `decide` "self": the person
    asked it to make its own choices rather than ask them (agent_request kind question is refused it); a child takes
    its parent's. `harness`: the stored harness version that briefs it (a probe's candidate; openlab code only, no
    tool passes it); a child is briefed by its parent's, any other agent by the workspace's active one. `tag`: what
    spent it (a refinement step), kept in launch.json and inherited, so harness overhead is reported apart.
    `advise`: put the approved harness lessons that match it in its brief. `resources` ({cores, memory_gb, gpus,
    gpu_model}) and `node` (a node's name): the allocation it holds for its life (openlab.nodes); without them a child
    shares its parent's allocation, if it has one. `params`: settings for its work, kept in launch.json and put in
    its brief as data."""
    root = root_of(root) if root is None else Path(root).resolve()
    store, cfg = Store(root), config(root)
    if cli not in CLIS:
        raise AgentError(f"unknown agent CLI {cli!r}", f"one of {', '.join(CLIS)}")
    if not prompt.strip():
        raise AgentError("an agent needs a prompt")
    budget = {"usd": float(cfg["budget_usd"]), "minutes": float(cfg["budget_minutes"]),
              "turns": int(cfg["budget_turns"]), **{k: v for k, v in (budget or {}).items() if v}}
    if parent:
        check_child(store, parent, budget, cfg)
    if params is not None and not isinstance(params, dict):
        raise AgentError("params is an object of settings for the worker", "e.g. {\"seed\": 3, \"lr\": 0.001}")
    from ..nodes import alloc as A
    from ..nodes.inventory import NodeError
    try:
        req = A.parse(resources, node, budget.get("minutes"))
        if req is not None:                             # refused now when no node could ever hold it
            A.place(req, A.offers({req["pin"]} if req["pin"] else None), [])
    except NodeError as e:
        raise AgentError(e.error, e.meaning, e.try_) from None
    shared = A.of_agent(parent, root) if parent and req is None else None
    model = model or _inherited_model(store, parent, cli) or cfg.get("model") or None
    if decide not in DECIDE:
        raise AgentError(f"decide is one of {', '.join(repr(d) for d in DECIDE)}", "'self': the agents make their own "
                         "choices; '': they may ask the person")
    decide = decide or ((store.get(parent) or {}).get("decide") or "" if parent else "")
    role = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in (role or "agent"))[:24] or "agent"
    from ..harness import versions
    up = _launch(store, parent)
    try:
        h = versions.get(root, harness or up.get("harness", "")) if (harness or up.get("harness")) \
            else versions.active(root)
    except Exception as e:                                          # noqa: BLE001 -- a broken harness spawns nothing
        raise AgentError(f"the harness cannot brief an agent: {e}", "the active harness version (or the one asked "
                         "for) does not validate here", "`openlab harness check` says why; `openlab harness "
                         "rollback` goes back") from None
    tag = tag or up.get("tag", "")
    advice, lessons = ("", ())
    if advise:
        from ..harness import lessons as L
        advice, lessons = L.advice(root, role, study or "", prompt, h)
    id = f"{role}-{secrets.token_hex(3)}"
    work = Path(cwd).resolve() if cwd else (root / study if study and (root / study).is_dir() else root)
    agent = store.register(id, "spawned" if cli == "claude" else "codex", name=id, role=role, parent=parent or "",
                           cli=cli, study=study or "", question=question or "", cwd=str(work),
                           session_id=str(uuid.uuid4()) if cli == "claude" else "", model=model or "",
                           budget=budget, spent={"usd": 0.0, "tokens_in": 0, "tokens_out": 0, "turns": 0,
                                                 "tool_calls": 0}, status="running", decide=decide)
    d = store.agent_dir(id)
    token = store.issue_token(id)                       # its writes are verified against it (identity.verified)
    (d / "mcp.json").write_text(json.dumps(mcp_config(root, id, token), indent=1))
    os.chmod(d / "mcp.json", 0o600)
    (d / "settings.json").write_text(json.dumps(settings_for_spawn(), indent=1))
    (d / "prompt.txt").write_text(prompt)
    (d / "brief.txt").write_text(brief(agent, h, advice) + (
        "\n\nParams your parent set for this work (data, not instructions): " + json.dumps(params, sort_keys=True)
        if params else ""))
    native = bool(cfg.get("allow_native_subagents"))
    tools = list(allowed_tools or role_tools(cfg, role) or cfg["allowed_tools"])
    if not native:                                      # delegation goes through openlab_agent_spawn (its budget)
        tools = [t for t in tools if t not in NATIVE_SUBAGENT_TOOLS]
    launch = {"allowed_tools": tools, "disallowed_tools": [] if native else list(NATIVE_SUBAGENT_TOOLS),
              "permission_mode": permission_mode or cfg["permission_mode"], "model": model or ""}
    told = {"harness": h.sha, "harness_sha": harness_sha(root, role, decide, cli, launch, cfg, h, lessons),
            **({"lessons": list(lessons)} if lessons else {}), **({"tag": tag} if tag else {})}
    work_on = {**({"resources": req} if req else {}), **({"share": shared["id"]} if shared else {}),
               **({"params": params} if params else {})}
    (d / "launch.json").write_text(json.dumps({**launch, **told, **work_on}))
    store.event(id, "spawn", summary=f"{cli} {role}" + (f" for {parent}" if parent else "")
                + (f" on {model}" if model else "") + f": {prompt}",
                payload={"prompt": prompt, "budget": budget, "study": study, "question": question, "parent": parent,
                         "model": model or "", **told, **work_on})
    sup_pid = os.getpid()
    if start:
        log = open(d / "supervisor.log", "ab")
        sup = subprocess.Popen([sys.executable, "-m", "openlab.agents.supervisor", str(root), id],
                               stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
                               cwd=str(root), env={**os.environ, "OPENLAB_WORKSPACE": str(root),
                                                   "OPENLAB_AGENT_TOKEN": token})
        log.close()
        store.update(id, supervisor=sup.pid)
        sup_pid = sup.pid
    out = store.get(id)
    if req is not None or shared is not None:           # held by its supervisor's process, for its life
        try:
            row = A.request(root, id, req, sup_pid, params) if req is not None else A.join(shared["id"], root, id,
                                                                                           sup_pid)
        except NodeError as e:
            store.event(id, "allocation_refused", summary=e.error)
            raise AgentError(e.error, e.meaning, e.try_) from None
        store.event(id, "allocation_request" if req is not None else "allocation_share",
                    summary=(A.describe(row) + f" ({row['state']}" + (f": {row['why']})" if row["why"] else ")")),
                    payload={"allocation": row["id"], "state": row["state"], "why": row["why"],
                             "request": req or {}})
        out = {**out, "allocation": {k: row[k] for k in ("id", "state", "node", "host", "cores", "memory_gb", "gpus",
                                                           "gpu_names", "why", "expected")}}
    return out


def settings_for_spawn() -> dict:
    """The openlab hooks, passed with --settings so the gates hold for a spawned agent in any folder (identical
    to what `openlab init --claude` writes, so a project copy runs once)."""
    from .claude_setup import hooks_settings
    return {"hooks": hooks_settings()}
