r"""One dispatch, several skins: every action in the workspace is a tool, and the web UI, the CLI, the MCP
and the agent all call the same :func:`dispatch`. The agent can do what the user can because there is
nothing else to call.

A tool is a Python function whose first parameter is the :class:`ToolContext`; its JSON Schema is derived
from its annotations, its first docstring line is its description. Every call returns the same envelope:

    {"ok": true,  "tool": "study_status", "result": {...}, "took_ms": 12}
    {"ok": false, "tool": "ws_delete", "error": "confirm", "meaning": "...", "try": "..."}

A ``destructive`` tool runs only with ``confirm=true``: the first call explains what would be lost.
"""

from __future__ import annotations

import contextvars
import inspect
import os
import threading
import time
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

SECTIONS = ("workspace", "research", "study", "meta")


def _retry_hint(message: str) -> str:
    """What to try after a failure: a missing module names what to install, anything else says fix the input."""
    import re
    m = re.search(r"No module named '([^'.]+)", message)
    if not m:
        return "fix the input and call again"
    mod = m.group(1)
    dist = {"nacl": "pynacl", "sklearn": "scikit-learn", "yaml": "pyyaml"}.get(mod, mod)
    try:
        from importlib import metadata
        for req in metadata.requires("openlab-research") or []:
            name = re.split(r"[\s;<>=!~\[(]", req, maxsplit=1)[0].lower().replace("_", "-")
            if name == dist.lower():
                extra = re.search(r"extra\s*==\s*['\"]([^'\"]+)", req)
                return (f'install the "{extra.group(1)}" extra: pip install "openlab-research[{extra.group(1)}]"' if extra
                        else f"a core dependency is missing: pip install {dist} (or reinstall openlab)")
    except Exception:                                           # noqa: BLE001 -- openlab not installed as a package
        pass
    return f"install {dist} (pip install {dist}); `openlab doctor` lists what each feature needs"


@dataclass
class Tool:
    name: str
    section: str
    fn: Callable
    doc: str
    mutates: bool = False
    destructive: bool = False
    human: bool = False            # a person's decision: refused unless the caller proved it is the reviewer
    confirm: str = ""              # what a destructive call does, in the words of the person asked to confirm it

    @property
    def schema(self) -> dict:
        props, required = {}, []
        hints = typing.get_type_hints(self.fn)
        for pname, p in list(inspect.signature(self.fn).parameters.items())[1:]:        # skip ctx
            t = hints.get(pname, str)
            origin = typing.get_origin(t)
            if origin is typing.Union:                                                   # Optional[X]
                t = next((a for a in typing.get_args(t) if a is not type(None)), str)
            js = {str: "string", int: "integer", float: "number", bool: "boolean", dict: "object", list: "array"}.get(t, "string")
            props[pname] = {"type": js}
            if p.default is inspect.Parameter.empty:
                required.append(pname)
            else:
                props[pname]["default"] = p.default
        if self.destructive:
            props["confirm"] = {"type": "boolean", "default": False}
        return {"type": "object", "properties": props, "required": required}

    def to_record(self) -> dict:
        return {"name": self.name, "section": self.section, "description": self.doc, "mutates": self.mutates,
                "destructive": self.destructive, "human": self.human, "input_schema": self.schema}


TOOLS: dict = {}


def tool(section: str, *, mutates: bool = False, destructive: bool = False, human: bool = False,
         name: Optional[str] = None, confirm: str = ""):
    if section not in SECTIONS:
        raise ValueError(f"section is one of {', '.join(SECTIONS)}")

    def deco(fn):
        doc = (inspect.getdoc(fn) or "").strip().split("\n\n")[0].replace("\n", " ")
        TOOLS[name or fn.__name__] = Tool(name or fn.__name__, section, fn, doc, mutates or destructive, destructive,
                                          human, confirm)
        return fn
    return deco


class ToolError(Exception):
    """A refusal the caller can act on: what it means and what to try."""

    def __init__(self, error: str, meaning: str = "", try_: str = ""):
        super().__init__(error)
        self.error, self.meaning, self.try_ = error, meaning, try_


def person_file(path: str) -> bool:
    """A file only a person edits (the workspace's budgets and the agent CLI's hooks): no tool writes, moves or
    deletes it; the Claude Code hooks hold the same for an agent's Edit and Bash."""
    p = Path(path)
    return p.name == "openlab.toml" or (len(p.parts) >= 2 and p.parts[-2] == ".claude"
                                       and p.name in ("settings.json", "settings.local.json"))


def no_agent_code(what: str) -> None:
    """Refuse when ``OPENLAB_NO_AGENT_CODE=1`` (``openlab-mcp --http``: the agent is on another machine, and nothing
    it writes runs as code on this one, where openlab and its keys live)."""
    if os.environ.get("OPENLAB_NO_AGENT_CODE") == "1":
        raise ToolError(f"{what} is off", "this openlab serves an agent on another machine; its code does not run here",
                        "use the tools that run no code of yours: the study's reads, claims and literature verbs")


class ToolContext:
    """What the tools share: the workspace root (every path is jailed under it) and subscribers to events."""

    def __init__(self, root: str = "."):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.subscribers: list = []

    def path(self, rel: str) -> Path:
        p = (self.root / (rel or "")).resolve()
        if p != self.root and self.root not in p.parents:
            raise ToolError("outside the workspace", f"{rel!r} resolves outside {self.root}", "use a path inside the workspace")
        from ..study.vault import closed_dirs
        for v in closed_dirs(self.root):
            if p == v or v in p.parents:
                raise ToolError("the vault is closed", "held-out labels are not readable through the workspace",
                                "a study scores its holdout through `openlab study holdout` (a person's act)")
        return p

    def rel(self, p: Path) -> str:
        return str(p.resolve().relative_to(self.root))

    def emit(self, channel: str, payload: dict) -> None:
        event = {"channel": channel, **payload}
        for fn in list(self.subscribers):
            try:
                fn(event)
            except Exception:                                   # noqa: BLE001 -- a broken subscriber must not break a tool
                pass

    def close(self) -> None:
        """Nothing is held between calls: kept so a server can close its context."""


#: who called the tool being dispatched: "agent" through the MCP or the server's agent API, else "" (a person, a test)
CALLER: contextvars.ContextVar = contextvars.ContextVar("openlab_caller", default="")


def dispatch(ctx: ToolContext, name: str, args: Optional[dict] = None, human: bool = False, caller: str = "",
             agent: str = "") -> dict:
    """Run a tool and return the envelope. Never raises. ``human`` is set only by the web server, for a request
    carrying the reviewer secret it never hands to the agent terminal. Every call is metered (openlab.app.usage).
    ``agent``: the acting agent (the web server's x-openlab-agent header); else openlab.agents.identity.current()."""
    from ..agents.identity import acting_as, held           # attribution and a person's pause
    token = CALLER.set(caller)
    try:
        with acting_as(agent):
            if not human and (refused := held(ctx.root, name)) is not None:    # a person's request is never held
                return refused
            return _metered(ctx, name, args, human, caller)
    finally:
        CALLER.reset(token)


def _metered(ctx: ToolContext, name: str, args: Optional[dict], human: bool, caller: str) -> dict:
    _meter.cpu, _meter.doc = 0.0, (args or {}).get("doc") or (args or {}).get("path") or ""
    env = _dispatch(ctx, name, args, human)
    try:
        from .usage import log
        log(ctx.root, name, env, _meter.doc, _meter.cpu, caller)
    except Exception:                                            # noqa: BLE001 -- metering never fails a call
        pass
    return env


_meter = threading.local()


def _dispatch(ctx: ToolContext, name: str, args: Optional[dict] = None, human: bool = False) -> dict:
    t0 = time.monotonic()
    args = dict(args or {})
    t = TOOLS.get(name)
    if t is None:
        return {"ok": False, "tool": name, "error": "unknown tool", "meaning": f"there is no tool named {name!r}",
                "try": "call `tools` for the list"}
    if t.human and not human:
        return {"ok": False, "tool": name, "error": "a reviewer's act",
                "meaning": f"{name} records a person's decision, and an agent cannot make it for them",
                "try": "file it with agent_request (it lands in the person's inbox, in `openlab inbox` and the app's "
                       "Agents view), or ask the person to do it in the openlab app or with `openlab study holdout` / "
                       "`amend --person` / `screen --person` in their own terminal"}
    confirm = bool(args.pop("confirm", False))
    if t.destructive and not confirm:
        return {"ok": False, "tool": name, "error": "confirm",
                "meaning": f"{t.confirm}: this cannot be undone" if t.confirm else f"{name} cannot be undone: {t.doc}",
                "try": "call again with confirm=true"}
    params = inspect.signature(t.fn).parameters
    if "confirm" in params:                                     # a tool that decides for itself when to ask
        args["confirm"] = confirm
    unknown = [k for k in args if k not in params]
    if unknown:
        return {"ok": False, "tool": name, "error": "unknown argument", "meaning": f"{name} has no {', '.join(unknown)}",
                "try": f"its arguments are {', '.join(list(params)[1:]) or 'none'}"}
    if t.mutates and any(isinstance(args.get(k), str) and person_file(args[k]) for k in ("path", "to", "dest", "src")):
        return {"ok": False, "tool": name, "error": "a person's file", "meaning": "openlab.toml and .claude/settings*.json "
                "are edited by a person in their editor, never through a tool", "try": "read it with ws_read"}
    try:
        result = t.fn(ctx, **args)
    except ToolError as e:
        return {"ok": False, "tool": name, "error": e.error, "meaning": e.meaning, "try": e.try_}
    except TypeError as e:
        return {"ok": False, "tool": name, "error": "bad arguments", "meaning": str(e),
                "try": f"its arguments are {', '.join(list(params)[1:]) or 'none'}"}
    except Exception as e:                                       # noqa: BLE001
        return {"ok": False, "tool": name, "error": type(e).__name__, "meaning": str(e), "try": _retry_hint(str(e))}
    out = {"ok": True, "tool": name, "result": result, "took_ms": int((time.monotonic() - t0) * 1000)}
    if t.mutates:
        ctx.emit("workspace", {"tool": name, "args": {k: v for k, v in args.items() if k in ("path", "doc", "index", "name")}})
    return out
