"""Who is acting: the agent id every record write carries (a study node's `session`).

`current()` looks, in order, at
1. the context variable set for one call (`acting_as(id)`): the MCP server sets it per tool call, the web app per
   request (the `x-openlab-agent` header);
2. `OPENLAB_AGENT` in the environment: a spawned agent's MCP server and CLI, and a Claude Code session's Bash
   commands (its SessionStart hook exports it through CLAUDE_ENV_FILE);
3. `person:<handle>` (the person key's handle, else the login) when the process runs at a terminal (stdin,
   stdout or stderr is a tty, or it has a controlling terminal: `at_terminal`) and the environment is not an
   agent's: a person typing commands, even with stdin redirected. A LABEL only: such writes stay actor `agent`; a
   person's act is the signed one (openlab.agents.person);
4. `unattributed`.

Identity values are agent labels, never a person: a claimed id starting with `person` or `terminal:` is
`unverified:<claimed>`. An agent openlab knows of has a token (OPENLAB_AGENT_TOKEN, issued at spawn or by the hooks;
the store keeps its hash): a claim of an agent's id without its token is `unverified:<claimed>` and a `spoof`
event, unless the token names a Claude Code session or sub-agent, whose hook-injected id then wins. This is
attribution, recorded and cross-checked, not authentication: an agent on the same OS account can read a sibling's
files. Person acts are authenticated (the person key).

The MCP server of an interactive Claude Code session cannot know which session or sub-agent called it (MCP carries
no such id), so the PreToolUse hook notes each `mcp__openlab__*` call in the control store and `for_mcp_call`
claims that note (same tool, same input, within 30 s).
"""

from __future__ import annotations

import contextvars
import getpass
import os
import sys
from contextlib import contextmanager
from typing import Optional

UNATTRIBUTED = "unattributed"
_ACTING: contextvars.ContextVar = contextvars.ContextVar("openlab_agent", default=None)


@contextmanager
def acting_as(agent: Optional[str]):
    """Run the block as `agent` (a no-op for None or "")."""
    if not agent:
        yield
        return
    token = _ACTING.set(agent)
    try:
        yield
    finally:
        _ACTING.reset(token)


def terminal() -> str:
    try:
        user = getpass.getuser()
    except Exception:                                               # noqa: BLE001
        user = "person"
    return f"terminal:{user}"


AGENT_ENV = ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CODEX_SANDBOX", "CODEX_SANDBOX_NETWORK_DISABLED")
_VERIFIED: dict = {}


def person_label() -> str:
    """`person:<handle>` from the person key's public file, else `person:<login>`."""
    try:
        import json
        from pathlib import Path
        h = json.loads((Path.home() / ".config" / "openlab" / "person.pub").read_text(encoding="utf-8"))["handle"]
    except Exception:                                               # noqa: BLE001
        h = terminal().split(":", 1)[1]
    return f"person:{h}"


def claimed(agent: str) -> str:
    """An id an agent (or a header) claims: never a person's."""
    agent = (agent or "").strip()
    low = agent.lower()
    if low.startswith(("person", "terminal:", "unverified:", "openlab")) or low in ("human", "reviewer"):
        return f"unverified:{agent}"
    return agent


def _store_for_check():
    from .store import Store, root_of
    root = root_of(os.environ.get("OPENLAB_WORKSPACE") or None)
    if not (root / ".openlab" / "control.db").is_file():
        return None
    return Store(root)


def verified(agent: str, token: str) -> str:
    """The id to record for a process claiming `agent` with `token` (see the module docstring). Logs a `spoof`
    event once per claim and token. Never raises."""
    key = (agent, token)
    if key in _VERIFIED:
        return _VERIFIED[key]
    out = agent
    try:
        st = _store_for_check()
        a = st.get(agent) if st is not None and agent else None
        owner = st.by_token(token) if st is not None and token else None
        if owner is not None and owner["id"] != agent:
            hook = owner["kind"] in ("claude-session", "claude-subagent")
            out = owner["id"] if hook else f"unverified:{agent}"
            st.event(owner["id"], "spoof", summary=f"claimed to be {agent or '(no one)'}; its token is "
                     f"{owner['id']}'s" + ("; recorded as " + owner["id"] if hook else ""),
                     payload={"claimed": agent, "token_of": owner["id"]})
        elif a is not None and a.get("token_sha") and owner is None:
            out = f"unverified:{agent}"
            st.event(agent, "spoof", summary=f"a process claimed to be {agent} without its token",
                     payload={"claimed": agent, "token": "none" if not token else "unknown"})
    except Exception:                                               # noqa: BLE001 -- attribution never fails a write
        pass
    _VERIFIED[key] = out
    return out


def current() -> str:
    ctx = _ACTING.get()
    if ctx:
        return ctx
    env = os.environ.get("OPENLAB_AGENT", "").strip()
    token = os.environ.get("OPENLAB_AGENT_TOKEN", "").strip()
    if env:
        agent = claimed(env)
        return agent if agent != env else verified(env, token)
    if token:                                                       # OPENLAB_AGENT unset, its token still here
        return verified("", token) or "unverified:"
    if any(os.environ.get(k) for k in AGENT_ENV):
        return "unverified:agent-terminal"                          # an agent's shell, without its id
    if at_terminal():
        return person_label()
    return UNATTRIBUTED


def at_terminal() -> bool:
    """Whether this process runs at a person's terminal: stdin, stdout or stderr is a tty, or the process has a
    controlling terminal (so `openlab ... < file` or `echo y | openlab ...` typed by a person still counts). An
    agent's shell has none of these (Claude Code and Codex run commands without a terminal; a spawned agent's
    supervisor starts a new session). A label only, never a person's authority."""
    for f in (sys.stdin, sys.stdout, sys.stderr):
        try:
            if f is not None and f.isatty():
                return True
        except (ValueError, OSError, AttributeError):
            continue
    try:
        fd = os.open("/dev/tty", os.O_RDONLY | os.O_NOCTTY)
    except OSError:
        return False
    os.close(fd)
    return True


def for_mcp_call(tool: str, args: Optional[dict]) -> str:
    """The agent behind one MCP tool call: the note a PreToolUse hook left for this tool and input (oldest first,
    within 30 s), else OPENLAB_AGENT (a spawned agent's server), else unattributed. With OPENLAB_AGENT set, only
    notes by that agent or its descendants count: its Claude sub-agents share its MCP server. Never raises."""
    env = os.environ.get("OPENLAB_AGENT", "").strip()
    try:
        from .store import Store
        st = Store()
        among = None
        if env:
            among, todo = {env}, [env]
            while todo:
                for c in st.children(todo.pop()):
                    if c["id"] not in among:
                        among.add(c["id"])
                        todo.append(c["id"])
        hit = st.claim(tool, dict(args or {}), by=f"mcp:{os.getpid()}", among=among)
        if hit:
            return hit["agent"]
    except Exception:                                               # noqa: BLE001 -- attribution never fails a call
        pass
    return env or UNATTRIBUTED


#: tools an agent may call while paused: reading its own state and the inbox
PAUSE_OK = ("tools", "usage", "agent_status", "agent_inbox_item", "agent_list", "agent_log")


def held(workspace, tool: str = "") -> Optional[dict]:
    """The refusal envelope when the acting agent (or an ancestor) is paused or stopped by a person, else None.
    Dispatch calls this first, so a paused agent's next openlab call is refused even without Claude Code hooks."""
    agent = current()
    if agent == UNATTRIBUTED or agent.startswith(("terminal:", "person:")) or tool in PAUSE_OK:
        return None
    agent = agent.split(":", 1)[1] if agent.startswith("unverified:") else agent   # a claimed id is held too
    try:
        from .store import Store, root_of
        root = root_of(workspace)
        if not (root / ".openlab" / "control.db").exists():
            return None
        hit = Store(root).blocked(agent)
    except Exception:                                               # noqa: BLE001 -- never fails a call
        return None
    if hit is None:
        return None
    who, what, reason = hit
    if what == "paused" and str(reason).startswith("budget:"):
        return {"ok": False, "tool": tool, "error": "paused at its limit",
                "meaning": f"{who} reached its limit ({reason[8:].strip()}); openlab asked the person for more",
                "try": "stop and wait: the person's approval in the inbox raises the limit and resumes it"}
    return {"ok": False, "tool": tool, "error": f"{what} by person",
            "meaning": f"{who} was {what} by a person: {reason or 'no reason given'}",
            "try": "stop and wait; a person resumes it with `openlab agent resume " + who + "`" if what == "paused"
            else "end your work now"}


def attribute_mcp(server) -> bool:
    """Wrap an MCP server's tool dispatch so every call runs `acting_as` its caller (mcp 1.x FastMCP and 2.x
    MCPServer both route calls through `server._tool_manager.call_tool(name, arguments, ...)`, with the client's
    own arguments). Returns False, changing nothing, if the server is not built that way."""
    manager = getattr(server, "_tool_manager", None)
    inner = getattr(manager, "call_tool", None)
    if inner is None or getattr(inner, "_openlab_attributed", False):
        return False

    async def call_tool(name, arguments, *a, **kw):
        with acting_as(for_mcp_call(name, arguments)):
            return await inner(name, arguments, *a, **kw)

    call_tool._openlab_attributed = True
    manager.call_tool = call_tool
    return True


def ensure_agent_node(record, agent: str) -> None:
    """The first time `agent` writes to a study, an `agent` node (who it is, its parent, role and CLI) so the study
    is self-contained for audit. Once per agent id per study; never fails the write that triggered it."""
    if not agent or agent == UNATTRIBUTED:
        return
    try:
        if record.find("agent", id=agent):
            return
        info = {"id": agent, "name": agent, "role": "", "parent": "", "cli": "",
                "kind": "person" if agent.startswith("person:") else "terminal"}
        if agent.startswith("unverified:"):
            info["kind"] = "unverified"
        elif not agent.startswith(("terminal:", "person:")):
            try:
                from .store import Store, root_of
                a = Store(root_of(record.folder)).get(agent) if (root_of(record.folder) / ".openlab" /
                                                                 "control.db").exists() else None
            except Exception:                                       # noqa: BLE001
                a = None
            info.update({k: a[k] or "" for k in ("name", "role", "parent", "cli", "kind")} if a else
                        {"kind": "unregistered"})
        record.add("agent", info, actor="openlab", session=agent)
    except Exception:                                               # noqa: BLE001
        pass
