"""An MCP server that gives an agent openlab's study tools: ``openlab-mcp`` / ``python -m openlab.mcp_server`` (stdio).

The ``mcp`` SDK is a core dependency (``pip install openlab-research``). Works with the ``mcp`` SDK 1.x (FastMCP) and 2.x
(MCPServer). Every tool is a tool of the workspace's dispatch (:mod:`openlab.app`): what the web UI and the CLI act
on is what the agent acts on, through the same refusals. When ``$OPENLAB_APP_URL`` names a running
``openlab app``, the calls go to it; otherwise they run in this process on the workspace ``$OPENLAB_WORKSPACE`` (the
working directory by default).

Claude Code: ``claude mcp add openlab -- openlab-mcp`` (or the venv's own ``python -m openlab.mcp_server``).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

from . import __version__

try:                                            # mcp 2.x
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:                             # mcp 1.x
    try:
        from mcp.server.fastmcp import FastMCP as _Server
    except ImportError as exc:                  # pragma: no cover
        raise ImportError("the MCP server needs the `mcp` package, a core dependency: pip install openlab-research "
                          "(or reinstall it)") from exc

from . import agent_guide as _agent_guide
from .agent_guide import GUIDE as _GUIDE

#: the MCP's tool list (OPENLAB_MCP_TOOLS): unset, the CORE tools and openlab_tools / openlab_tool for every other;
#: "grouped", one tool per section; "all", every tool on its own
_MODE = _agent_guide.tool_mode()
#: what an agent reads when it connects (the text lives in openlab.agent_guide, so `openlab guide` never imports mcp)
INSTRUCTIONS = _agent_guide.instructions(_MODE)
server = _Server("openlab", instructions=INSTRUCTIONS, version=__version__)

_STARTUP: dict = {"toolset": None, "stamp": None, "installed": None}


def _src_stamp() -> tuple:
    """The modification times of openlab's sources: when none changed, the installed tool set cannot have."""
    d = Path(__file__).resolve().parent
    return tuple(sorted((str(p), p.stat().st_mtime_ns) for p in d.rglob("*.py") if "__pycache__" not in p.parts))


def _installed_toolset() -> Optional[str]:
    """The tool-set hash a fresh process would serve (the installed code), or None when it cannot be read."""
    import subprocess
    try:
        r = subprocess.run([sys.executable, "-c", "from openlab.app.tools import toolset_hash; print(toolset_hash())"],
                           capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    out = r.stdout.strip().splitlines()
    return out[-1] if r.returncode == 0 and out else None


def _restart_hint() -> Optional[str]:
    """Say so when the installed tool set differs from the one this server started with: what an agent can do about
    it (a headless agent cannot restart its MCP server)."""
    if _STARTUP["toolset"] is None:
        return None
    stamp = _src_stamp()
    if stamp != _STARTUP["stamp"]:                  # sources changed: ask a fresh process once per change
        _STARTUP["stamp"], _STARTUP["installed"], _STARTUP["changed"] = stamp, _installed_toolset(), True
    new = _STARTUP["installed"]
    if new and new != _STARTUP["toolset"]:
        return (f"this MCP server runs tool set {_STARTUP['toolset']}; the installed openlab has {new}, so a refusal "
                "here may be the older code's: the CLI `openlab tool NAME key=value` runs the installed code; tell "
                "the person if it matters (they restart the MCP server)")
    if _STARTUP.get("changed"):
        return ("openlab's code changed after this MCP server started, so these tools run the older code: the CLI "
                "(`openlab study ...`, `openlab tool NAME key=value`) runs the current openlab; a refusal here that the "
                "CLI does not give is the older code's")
    return None


NOTICE_EVERY = 50                                   # tool results between two repeats of the same notice


def _notice() -> Optional[str]:
    """The restart hint on the first result that has it, then once every NOTICE_EVERY results, not on every one."""
    hint = _restart_hint()
    if hint is None:
        return None
    seen = _STARTUP.get("notice")
    if seen and seen[0] == hint and seen[1] < NOTICE_EVERY:
        _STARTUP["notice"] = (hint, seen[1] + 1)
        return None
    _STARTUP["notice"] = (hint, 1)
    return hint


def _told() -> Optional[str]:
    """The person's prompts and the directions for this agent not yet delivered, marked delivered: added to the reply
    of its openlab tool call, the one channel a Codex agent has (a Claude agent's PostToolUse hook is the other)."""
    try:
        from .agents.hooks import delivery
        from .agents.identity import current
        from .agents.store import Store, root_of
        self_id = current()
        st = Store(root_of(os.environ.get("OPENLAB_WORKSPACE") or None))
        a = st.get(self_id) if self_id else None
        return delivery(st, a) if a else None
    except Exception:                                               # noqa: BLE001 -- a tool reply never breaks
        return None


def _register_workspace_tools() -> None:
    """Expose the workspace's dispatch (openlab.app) through the MCP: what the UI can do, the agent can do.
    The MCP tool's signature is the dispatch tool's, without its context; the reply is the envelope."""
    import inspect
    import typing
    try:
        from .app import TOOLS
        from .app.client import call
    except ImportError:                                             # the app's extras are not installed
        return
    families: dict = {}
    from .app.tools import toolset_hash
    _STARTUP["toolset"], _STARTUP["stamp"] = toolset_hash(), _src_stamp()
    for t in TOOLS.values():
        mcp_name = f"openlab_{t.name}"
        if mcp_name == "openlab_tools" or t.human:                   # a reviewer's act is not an agent's tool
            continue
        sig = inspect.signature(t.fn)
        hints = typing.get_type_hints(t.fn)                         # the tools modules use string annotations
        params = [p.replace(annotation=hints.get(p.name, str)) for p in list(sig.parameters.values())[1:]]
        if t.destructive:
            params.append(inspect.Parameter("confirm", inspect.Parameter.KEYWORD_ONLY, default=False, annotation=bool))

        def make(tool_name):
            def fn(**kw) -> dict:
                out = call(tool_name, kw)
                hint = _notice()
                out = {**out, "openlab_updated": hint} if hint and isinstance(out, dict) else out
                told = _told()
                return {**out, "openlab_message": told} if told and isinstance(out, dict) else out
            return fn
        fn = make(t.name)
        fn.__name__ = mcp_name
        fn.__signature__ = sig.replace(parameters=params, return_annotation=dict)
        fn.__annotations__ = {p.name: p.annotation for p in params if p.annotation is not inspect.Parameter.empty}
        fn.__annotations__["return"] = dict
        if GROUPED:
            families.setdefault(t.section, []).append(t)
            continue
        server.tool(name=mcp_name, description=SHORT.get(t.name) or t.doc)(fn)
    for section, members in families.items():
        _register_family(section, members, call)
    server.tool(name="openlab_tools", description="Find tools not in your list: `query` (a word) or `section`; "
                "no argument lists names by section.")(openlab_tools)
    server.tool(name="openlab_tool", description="Call any tool by name: name, args={...}.")(openlab_tool)
    if not GROUPED and _MODE != "all":
        _serve_only(CORE | {"tools", "tool"})
    _compact_schemas()


def _compact(node):
    """A JSON schema without what an agent does not need to call the tool: titles, `Optional`'s null branch, empty
    `items` and open `additionalProperties`, and null or empty-string defaults (the required list says optional)."""
    if isinstance(node, list):
        return [_compact(x) for x in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for k, v in node.items():
        if k == "title" and isinstance(v, str):
            continue
        if k == "anyOf" and isinstance(v, list):
            kept = [x for x in v if x != {"type": "null"}]
            if len(kept) == 1 and isinstance(kept[0], dict):
                out.update(_compact(kept[0]))
                continue
        if (k == "items" and v == {}) or (k == "additionalProperties" and v is True):
            continue
        if k == "default" and v in (None, ""):
            continue
        out[k] = v if k == "default" else _compact(v)
    return out


def _compact_schemas() -> None:
    for t in server._tool_manager.list_tools():
        props = t.parameters.get("properties") or {}
        t.parameters = {**_compact(t.parameters), "properties": {k: _compact(v) for k, v in props.items()}}


#: the default MCP list and the one line each is described by (an agent pays for every character of this list in
#: every session and sub-agent: tests/test_mcp_surface.py holds it under a budget). The verbs the role briefs
#: (openlab.agents.claude_setup, openlab.agents.spawn, openlab.study.proposal) name; every other tool is a
#: openlab_tool call away, with its full description in openlab_tools.
SHORT = {
    "study_new": "Scaffold a study folder (template: temporal, static or review).",
    "study_materialise": "Build and register the study's datasets from its recipes (`name`: one only; "
                         "refresh=true reruns the recipes).",
    "study_run": "Run model.py sandboxed as a counted trial (the first run seals study.toml); returns the result: "
                 "metrics with intervals, baselines, paired differences (a run still going after 14 min: its job "
                 "id, for agent_wait).",
    "study_status": "The study at a glance: question, seal, datasets, trials and holdout looks used, last result.",
    "study_runs": "The study's runs, newest first, and those in flight: audit verdict, primary metric against each "
                  "baseline; every score: openlab_tool(\"study_node\", {folder, id}); act() per decision: "
                  "\"study_series\".",
    "study_why": "What a node (e.g. a result id) rests on, oldest first.",
    "study_amend": "Record your edit of study.toml since the seal, with the reason.",
    "study_compare": "Compare result `a` against `b` on `metric`: paired difference, interval, p.",
    "study_test": "Test `of` (hypothesis or claim) on `result` against `baseline` at `alpha_j`; spends alpha budget.",
    "study_claim": "A claim: state observed (rests_on results/evidence), derived (expression over inputs "
                   "{name: 'node#path'}, value), inferred (rests_on claims), hypothesized (tests) or unknown "
                   "(searched). Verdict and level are computed.",
    "study_discover": "Run the sealed [review] searches and citation hops; records sources and coverage.",
    "study_sources": "The study's sources (status: all, unscreened, include, exclude, maybe, retracted).",
    "study_fetch_text": "A source's open-access full text, by page (from 1), to quote from.",
    "study_screen": "Screen sources: decisions=[{source, decision: include|exclude|maybe, criterion, rationale, "
                    "quote}]; the quote (20+ chars) must be in the source.",
    "study_extract": "Record evidence: `quote` on `page` of a fetched source, `section`, and `fields` of the "
                     "[review.extract] schema read from the quote.",
    "study_question": "Add a question to the question tree (`part_of` a broader one).",
    "study_hypothesis": "Add a falsifiable hypothesis under `question`: prediction, falsified_by, tags.",
    "study_dead_end": "Record a dropped hypothesis or approach (`of`): reason, tags, shown_by ids.",
    "study_frontier": "What to do next, ranked by expected information per hour; each item names its verb.",
    "study_take": "Lease a frontier item for `minutes` before acting on it.",
    "study_replicate": "Start a study replicating a paper (arXiv id, DOI or PDF), its target and tolerance fixed "
                       "before any run.",
    "study_fidelity": "One rebuilt baseline (`arm` of result `ours`) against the paper's value (evidence `paper`).",
    "study_replication": "One claim of the paper (evidence `paper`) against result `ours`, gated by fidelity ids.",
    "agent_spawn": "Delegate a sub-question to a child agent; its budget comes out of yours.",
    "agent_request": "Ask the person for an act you can't (kind: holdout, amend, approval, budget, screen, fresh, "
                     "continues, compute, instrument, instrument_run, experiment, other) or a question (kind "
                     "question, args {choices, multi, other}); returns an inbox id.",
    "agent_plan": "Report your whole plan as a new version: nodes [{id, title, kind, needs, state, links, owner}]; again "
                  "whenever it changes. kind: review, data, run, branch, test, select, person, paper, critique, "
                  "check, lesson, other.",
    "agent_plan_node": "Move one plan step (planned, active, waiting, done, failed, dropped) with the record ids it "
                       "rests on.",
    "agent_inbox_item": "The state of an inbox item you filed, with the person's reason or answer.",
    "agent_wait": "Wait for agents (ids) to end, your inbox items (inbox:7) to be answered or your runs "
                  "(job:run-...) to end, mixed in one list, up to `timeout` s (300 at most).",
    "agent_note": "A short progress note on your timeline.",
    "agent_reply": "Answer the person's message (prompt number `to`) in a few sentences, then carry on.",
    "agent_comments": "The person's open comments on your study's paper draft: file, line, quote, text, replies.",
    "agent_comment_reply": "Answer the person's comment #id on the paper draft; resolve=true once the prose is fixed.",
    "propose_status": "Per-branch progress of the approved proposal: agents, spend, trials, hypotheses.",
}
CORE = frozenset(SHORT)

#: the most the default tool list and the instructions may cost a session, in characters (~4 per token: 5k tokens)
BUDGET_CHARS = 20_000

_LIST_MAX = 50                                                      # rows of one openlab_tools reply


def _args(schema: dict) -> str:
    """A tool's arguments on one line: `name:type` when required, `name=default` when not."""
    req = schema.get("required", [])
    return ", ".join(f"{k}:{v['type']}" if k in req else f"{k}={json.dumps(v.get('default'), default=str)}"
                     for k, v in schema["properties"].items())


def openlab_tools(section: str = "", query: str = "") -> dict:
    from .app.client import call
    env = call("tools", {"section": section})
    if not env.get("ok"):
        return env
    q = query.strip().lower()
    rows = [r for r in env["result"]["tools"] if not r["human"] and r["name"] not in _REMOVED
            and (q in r["name"] or q in r["description"].lower())]
    if section or q:
        res = {"tools": [{"name": r["name"], "section": r["section"], "doc": r["description"].split(". ")[0][:160],
                          "args": _args(r["input_schema"])} for r in rows[:_LIST_MAX]],
               "more": max(0, len(rows) - _LIST_MAX)}
    else:
        res = {"sections": {}}
        for r in rows:
            res["sections"].setdefault(r["section"], []).append(r["name"])
    env["result"] = {**res, "openlab": env["result"]["openlab"]}
    if hint := _notice():
        env["openlab_updated"] = hint
    return env


def openlab_tool(name: str, args: Optional[dict] = None) -> dict:
    from .app import TOOLS
    from .app.client import call
    name = name.removeprefix("openlab_")
    if TOOLS.get(name) is None or name in _REMOVED:
        return {"ok": False, "tool": name, "error": "unknown tool", "meaning": f"no agent tool {name!r}",
                "try": "openlab_tools(query=...) finds a tool by a word of its name or description"}
    out = call(name, dict(args or {}))                              # a person's act: dispatch refuses it
    hint = _notice()
    return {**out, "openlab_updated": hint} if hint and isinstance(out, dict) else out


def _serve_only(keep) -> None:
    """Take every MCP tool whose name (without openlab_) is not in `keep` off the list; a section's family stays
    and refuses its removed members itself."""
    from .app import TOOLS
    families = {t.section for t in TOOLS.values()} if GROUPED else set()
    for t in server._tool_manager.list_tools():
        base = t.name.removeprefix("openlab_")
        if base not in keep and base not in families:
            server.remove_tool(t.name)


#: one MCP tool per section instead of one per tool (OPENLAB_MCP_TOOLS=grouped): an agent's first tools/list is a
#: few short descriptions rather than every full schema, and openlab_tools(section=...) gives the arguments
GROUPED = _MODE == "grouped"


def _register_family(section: str, members: list, call) -> None:
    names = sorted(t.name for t in members)
    by_name = {t.name: t for t in members}
    lines = "; ".join(f"{t.name}: {t.doc.split('. ')[0][:90]}" for t in sorted(members, key=lambda t: t.name))

    def fn(tool: str, args: Optional[dict] = None) -> dict:
        name = tool.removeprefix("openlab_")
        if name not in by_name or name in _REMOVED:
            return {"ok": False, "tool": name, "error": "unknown tool", "meaning": f"{name!r} is not a {section} tool",
                    "try": f"one of: {', '.join(n for n in names if n not in _REMOVED)}"}
        return call(name, dict(args or {}))
    fn.__name__ = f"openlab_{section}"
    server.tool(name=f"openlab_{section}",
                description=f"The {section} tools, called by name: tool=NAME, args={{...}}. "
                            f"openlab_tools(section=\"{section}\") lists each one's arguments. {lines}")(fn)


#: tools taken away at startup (--http removes those not allowlisted); a family must refuse them too
_REMOVED: set = set()


_register_workspace_tools()

from .agents.identity import attribute_mcp  # noqa: E402 -- each call runs as the agent that made it
attribute_mcp(server)


@server.resource("openlab://guide", name="guide", description="The agent contract: the study, the verbs in "
                 "order, the rules the record enforces, the person's acts", mime_type="text/markdown")
def guide_resource() -> str:
    if _GUIDE.exists():
        return _GUIDE.read_text(encoding="utf-8")
    return INSTRUCTIONS


@server.resource("openlab://reference", name="reference", description="The guide's detail: study.toml, model.py, "
                 "the verbs, the refusals", mime_type="text/markdown")
def reference_resource() -> str:
    ref = _GUIDE.with_name("AGENTS_REFERENCE.md")
    return ref.read_text(encoding="utf-8") if ref.exists() else "not in this install"


#: what ``--http`` serves an agent on another machine: an allowlist, so a tool added later is not served until
#: someone checks it cannot run code, read the host or reach the network from here
AGENT_TOOLS = frozenset("""
    notes_add notes_list ws_delete ws_list ws_move ws_read ws_write
    agent_activity key_info tools tool usage
    study_new study_materialise study_run study_seal study_status study_why study_amend study_verify
    study_list study_runs study_node study_series study_log study_forecast study_fill study_forecasts
    study_screen study_prisma study_sources study_extract study_evidence study_compare study_lattice
    study_claim study_question study_hypothesis study_test study_assume study_break study_dead_end study_tree
    study_claims study_graph study_trace study_traceability study_frontier study_take
    study_reproduce study_bundle study_reproductions study_nodes study_impact study_branch
    instrument_list instrument_show instrument_verify instrument_runs experiment_list experiment_status
""".split())


def _serve_agent_tools() -> None:
    """``--http``: only AGENT_TOOLS, on the list and through openlab_tool and the families alike."""
    from .app import TOOLS
    _REMOVED.update(set(TOOLS) - AGENT_TOOLS)
    _serve_only(AGENT_TOOLS)


def _host_ok(host: str, allowed: list) -> bool:
    return any(host == a or (a.endswith(":*") and host.rsplit(":", 1)[0] == a[:-2]) for a in allowed)


def _mount_agent_api(allowed: list, token: str) -> None:
    """``--http`` also serves the tools as plain JSON for the ``openlab`` CLI (``openlab/thin_cli.py``), through the
    same gate as the MCP: GET /api/guide, GET /api/tools, POST /api/tool/{name}. Never a person's acts. Every
    request needs the bearer token and an allowed Host, and is refused from a browser page (an Origin header)
    that isn't an allowed host: a web page the user visits must not reach the tools."""
    import hmac
    import urllib.parse
    import anyio
    from starlette.requests import Request
    from starlette.responses import JSONResponse, PlainTextResponse
    from .app import TOOLS
    from .app.client import call
    hidden = set(TOOLS) - AGENT_TOOLS

    def refused(request: Request):
        origin = request.headers.get("origin")
        if not _host_ok(request.headers.get("host", ""), allowed) or \
                (origin and not _host_ok(urllib.parse.urlparse(origin).netloc, allowed)):
            return JSONResponse({"ok": False, "error": "host not allowed"}, status_code=421)
        if not hmac.compare_digest(request.headers.get("authorization", "").encode(), f"Bearer {token}".encode()):
            return JSONResponse({"ok": False, "error": "unauthorized", "meaning": "the agent API needs the token "
                                 "openlab-mcp printed", "try": "set OPENLAB_APP_TOKEN"}, status_code=401)
        return None

    @server.custom_route("/api/guide", methods=["GET"])
    async def guide(request: Request):
        if r := refused(request):
            return r
        from .thin_cli import local_guide
        return PlainTextResponse(local_guide())

    @server.custom_route("/api/tools", methods=["GET"])
    async def tools(request: Request):
        if r := refused(request):
            return r
        env = await anyio.to_thread.run_sync(call, "tools", {"section": request.query_params.get("section", "")})
        env["result"]["tools"] = [r for r in env["result"]["tools"] if not r["human"] and r["name"] not in hidden]
        return JSONResponse(env)

    @server.custom_route("/api/tool/{name}", methods=["POST"])
    async def tool(request: Request):
        if r := refused(request):
            return r
        name = request.path_params["name"]
        try:
            args = await request.json() if await request.body() else {}
        except ValueError:
            args = None
        if not isinstance(args, dict):
            return JSONResponse({"ok": False, "tool": name, "error": "bad arguments", "meaning": "the body must be "
                                 "a JSON object", "try": "openlab tool NAME key=value ..."}, status_code=400)
        t = TOOLS.get(name)
        if t is None or t.human or name in hidden:
            return JSONResponse({"ok": False, "tool": name, "error": "unknown tool", "meaning": f"no agent tool {name!r}",
                                 "try": "openlab tools"}, status_code=404)
        env = await anyio.to_thread.run_sync(call, name, args)
        return JSONResponse(json.loads(json.dumps(env, default=str)))       # numpy and dates as text


def main() -> None:
    """``openlab-mcp`` (stdio), or ``openlab-mcp --http HOST:PORT`` for an agent on another machine (a VM): openlab,
    its keys and its data stay here, and nothing the agent writes runs as code here (no spawned agents, no study
    runs; a study's model.py runs only for an agent on this machine, in the runner's sandbox)."""
    import argparse
    p = argparse.ArgumentParser(prog="openlab-mcp")
    p.add_argument("--http", metavar="HOST:PORT", help="serve streamable HTTP at http://HOST:PORT/mcp")
    p.add_argument("--allow-host", action="append", default=[], metavar="HOST:PORT",
                   help="a Host header to accept besides localhost, e.g. 10.0.2.2:8765 for a QEMU guest")
    args = p.parse_args()
    hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*", *args.allow_host]
    if args.http:
        os.environ["OPENLAB_NO_AGENT_CODE"] = "1"
        _serve_agent_tools()
        token = os.environ.get("OPENLAB_APP_TOKEN") or ""
        if not token:
            import secrets
            token = secrets.token_urlsafe(24)
            print(f"openlab-mcp: agent API token (give the agent OPENLAB_APP_TOKEN={token})", file=sys.stderr, flush=True)
        _mount_agent_api(hosts, token)
        host, port = args.http.rsplit(":", 1)
        from mcp.server.transport_security import TransportSecuritySettings
        server.run(transport="streamable-http", host=host, port=int(port),
                   transport_security=TransportSecuritySettings(allowed_hosts=hosts))
    else:
        server.run(transport="stdio")


if __name__ == "__main__":
    main()
