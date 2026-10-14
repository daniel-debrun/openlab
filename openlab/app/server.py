r"""The workspace server: the web UI's skin over the one dispatch.

    openlab app --root my_workspace            # http://127.0.0.1:8321  (prints the URL with its token)

    GET  /api/health
    GET  /api/tools                          every tool: section, description, input schema
    POST /api/tool/{name}                    body = the arguments; the reply is the dispatch envelope, verbatim
    POST /api/upload?path=NAME               body = a data file's bytes, written into the workspace
    WS   /ws/events                          one socket: every mutation, as it happens

The server computes nothing of its own: a button in the UI, ``openlab tool``, and an agent over the MCP all land
in :func:`openlab.app.dispatch`. When the MCP server is started with ``OPENLAB_APP_URL`` pointing here, the agent's
actions run in *this* process, so they appear in the open UI as they happen.

Auth: a bearer token (``OPENLAB_APP_TOKEN``, else minted per boot and printed). Bound to 127.0.0.1 unless told
otherwise; inside a container bind 0.0.0.0 and publish the port.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import secrets
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .. import __version__
from . import TOOLS, ToolContext, ToolError, dispatch

STATIC = Path(__file__).resolve().parent / "static"
SHOW = Path(__file__).resolve().parent / "show"            # openlab show: one page, no build step
WATCH_SECONDS = 1.0                                        # how often the watcher looks at the control store
COMPILE_WORKERS = 2                                        # release compiles at once (a study only ever one)
log = logging.getLogger("openlab.show")
#: what the upload route accepts: the data files a study's recipes read, and at most this many bytes
UPLOAD_SUFFIXES = (".csv", ".tsv", ".parquet", ".json", ".ndjson", ".jsonl", ".xlsx", ".xlsm", ".xls", ".sav",
                   ".zsav", ".por", ".dta", ".sas7bdat", ".xpt", ".nc", ".nc4", ".netcdf", ".h5", ".hdf5", ".zarr")
UPLOAD_MAX = 500 << 20
PLACEHOLDER = """<!doctype html><meta charset="utf-8"><title>openlab</title>
<body style="font:16px system-ui;margin:3rem;max-width:44rem">
<h1>openlab workspace</h1><p>The server is running; the web UI has not been built into this install.</p>
<p>Everything the UI does is available now: <code>GET /api/tools</code>, <code>POST /api/tool/{name}</code>,
<code>WS /ws/events</code>, the CLI (<code>openlab tool ws_list</code>) and the MCP.</p></body>"""


def write_mcp_json(root: Path) -> Optional[Path]:
    """Give the agent pane openlab's tools: a workspace ``.mcp.json`` running this interpreter's MCP server, unless
    the workspace already has one (the user's file is never overwritten). Returns the file written, else None."""
    import sys
    f = Path(root) / ".mcp.json"
    if f.exists():
        return None
    f.write_text(json.dumps({"mcpServers": {"openlab": {"command": sys.executable, "args": ["-m", "openlab.mcp_server"]}}},
                            indent=1) + "\n")
    return f


class Hub:
    """Fan one stream of events out to every connected socket (events arrive from worker threads)."""

    def __init__(self):
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.queues: set = set()

    def publish(self, event: dict) -> None:
        if self.loop is None:
            return
        for q in list(self.queues):
            self.loop.call_soon_threadsafe(q.put_nowait, event)


_NAME = __import__("re").compile(r"[a-z0-9][a-z0-9._-]{0,47}")


def home_name(home: Path) -> str:
    """The tab name of `home` itself: its folder name, with `~` appended while a workspace directly under it has that
    name, so the two never share a tab (the sub-workspace keeps its folder name: it is the one a POST made)."""
    home = Path(home).resolve()
    name = home.name or "workspace"
    while (home / name / "openlab.toml").is_file():
        name += "~"
    return name


def workspaces(home: Path) -> dict:
    """{name: folder}: `home` itself when it is a workspace (it has openlab.toml), and each folder directly under it
    that is one. Each is a research of its own: its agents, inbox, records and paper. openlab show draws one per tab.
    A symlink is never one: the server is jailed to home, and a link would hand a tab (and its acts) a folder outside.
    Only direct children: a workspace nested deeper (inside a study folder, say) is not a research of this root."""
    home = Path(home).resolve()
    key = home_name(home)
    out = {key: home} if (home / "openlab.toml").is_file() else {}
    try:
        entries = sorted(home.iterdir())
    except OSError:
        entries = []
    for d in entries:
        try:
            if (not d.name.startswith(".") and not d.is_symlink() and d.is_dir() and (d / "openlab.toml").is_file()
                    and d.name not in out):
                out[d.name] = d
        except OSError:                                         # a folder we may not read is not a workspace
            continue
    return out or {key: home}


def _cli_missing(cli: str, name: str) -> Optional[dict]:
    """A refusal when the agent CLI is not on this server: the supervisor would only fail later, out of sight."""
    from ..agents.spawn import CLIS
    if cli in CLIS and not (os.environ.get(f"OPENLAB_{cli.upper()}_BIN") or shutil.which(cli)):
        return {"ok": False, "error": f"{cli} is not installed", "meaning": f"there is no {cli} on this server's PATH, "
                "so the lead cannot run", "try": f"install {cli} and sign it in (or set OPENLAB_{cli.upper()}_BIN), "
                f"then POST /api/workspaces/{name}/lead with the prompt"}
    return None


def create_app(root: Optional[str] = None, token: Optional[str] = None) -> FastAPI:
    hub = Hub()
    ctx = ToolContext(root or os.environ.get("OPENLAB_WORKSPACE") or ".")
    home = ctx.root
    ctx.subscribers.append(lambda e: hub.publish({**e, "ws": home_name(home)}))      # every event names its research
    ctxs: dict = {}
    lead_lock = threading.Lock()

    def forget(ws: str) -> None:
        """Drop a cached context whose folder is gone (renamed or deleted at runtime)."""
        c = ctxs.pop(ws, None)
        if c is not None:
            with contextlib.suppress(Exception):
                c.close()

    def ctx_for(ws: str) -> ToolContext:
        """The context of workspace `ws` under home ("" or home's tab name: the server's root). The cache is checked
        against the disk on every call: a folder renamed or deleted is forgotten, and a name reused is a new context."""
        if not ws or ws == home_name(home):
            return ctx
        found = workspaces(home).get(ws)
        if ws in ctxs and (found is None or ctxs[ws].root != found):
            forget(ws)
        if found is None or found == home:
            raise HTTPException(404, f"no workspace {ws!r} under {home}")
        if ws not in ctxs or ctxs[ws].root != found.resolve():    # a folder deleted and made again is a new one
            c = ToolContext(str(found))
            c.subscribers.append(lambda e, ws=ws: hub.publish({**e, "ws": ws}))
            c.app_token = getattr(ctx, "app_token", "")
            ctxs[ws] = c
        return ctxs[ws]

    def start_lead(name: str, body: dict) -> dict:
        """Spawn the lead of research `name` with the prompt and budget in `body`: the envelope of agent_spawn."""
        cli = str(body.get("cli") or "claude")
        if (refused := _cli_missing(cli, name)) is not None:
            return refused
        return dispatch(ctx_for(name), "agent_spawn", {"prompt": str(body["prompt"]), "role": "lead",
                        "question": str(body.get("question") or "")[:300], "budget_usd": float(body.get("budget_usd") or 0),
                        "minutes": float(body.get("minutes") or 0), "model": str(body.get("model") or ""), "cli": cli,
                        "decide": "self" if body.get("decide") == "self" else ""})

    token = token if token is not None else (os.environ.get("OPENLAB_APP_TOKEN") or secrets.token_urlsafe(18))
    ctx.app_token = token                     # a research agent started by a tool reaches back here with it

    compiler = ThreadPoolExecutor(max_workers=COMPILE_WORKERS, thread_name_prefix="openlab-release")

    async def watch():
        """Agents write the control store from their own processes (hooks, the supervisor, the MCP): a change there
        is published as a `control` event (with its workspace), so the open page refreshes without polling. A new
        workspace under home is published as `workspaces`. When a study's lead ends done, its release is compiled.

        The compiles run on their own pool, never awaited here, so a slow one does not hold the loop up; one study
        compiles at most once at a time (a lead finishing while its study compiles is folded into one more build,
        made only when the record moved past what the release holds). A failed compile is a `release` event in the
        control store. Each workspace's look is guarded on its own: a deleted, locked or broken workspace is skipped
        this tick and never stops the others or the watcher. A lead missed while the server was down: see
        show_tools.finished_leads."""
        from .show_tools import changed, finished_leads
        loop = asyncio.get_running_loop()
        last: dict = {}
        names: Optional[tuple] = None
        seen: dict = {}
        compiling: dict = {}                    # study path: the job building it now
        queued: dict = {}                       # study path: the leads that ended while it built

        def failed(job, e: BaseException) -> None:
            note = getattr(job, "log", None)
            log.warning("release job for %s failed: %r", getattr(job, "study", job), e)
            if note is not None:
                with contextlib.suppress(Exception):
                    loop.run_in_executor(compiler, note, f"release not compiled: {type(e).__name__}: {e}")

        def finish(fut, job, key) -> None:
            compiling.pop(key, None)
            if not fut.cancelled() and fut.exception() is not None:
                failed(job, fut.exception())
            nxt = queued.pop(key, None)
            if nxt is not None:
                nxt.only_if_stale = True
                start(nxt)

        def start(job) -> None:
            key = str(getattr(job, "study", id(job)))
            if key in compiling:
                if key in queued:
                    queued[key].agents.extend(a for a in job.agents if a not in queued[key].agents)
                else:
                    queued[key] = job
                return
            try:
                fut = loop.run_in_executor(compiler, job)
            except RuntimeError as e:           # the pool is shut down: the server is stopping
                failed(job, e)
                return
            compiling[key] = job
            fut.add_done_callback(lambda f, job=job, key=key: finish(f, job, key))

        async def look(name: str, folder: Path) -> None:
            fp = await loop.run_in_executor(None, changed, folder)
            if fp is None or fp == last.get(name):
                return
            if name in last:
                hub.publish({"channel": "control", "ws": name, "seq": fp[0]})
                jobs = await loop.run_in_executor(None, finished_leads, folder, seen.setdefault(name, set()))
            else:                               # the first look: history, but what was missed while down
                jobs = await loop.run_in_executor(None, finished_leads, folder, seen.setdefault(name, set()), True)
            last[name] = fp
            for job in jobs:
                start(job)

        app.state.compiling = compiling
        while True:
            try:
                found = await loop.run_in_executor(None, workspaces, home)
                if names is not None and tuple(found) != names:
                    hub.publish({"channel": "workspaces", "names": list(found)})
                names = tuple(found)
                for gone in set(last) - set(found):     # deleted: forget it, a new one of that name starts afresh
                    last.pop(gone, None)
                    seen.pop(gone, None)
                for gone in [w for w in ctxs if w not in found]:     # renamed or deleted at runtime
                    forget(gone)
                for name, folder in found.items():
                    try:
                        await look(name, folder)
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:              # noqa: BLE001
                        log.warning("watcher: workspace %s: %r", name, e)
            except asyncio.CancelledError:
                raise
            except Exception as e:                      # noqa: BLE001
                log.warning("watcher: %r", e)
            await asyncio.sleep(WATCH_SECONDS)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.loop = asyncio.get_running_loop()
        watcher = asyncio.ensure_future(watch())
        yield
        watcher.cancel()
        compiler.shutdown(wait=False)
        ctx.close()

    app = FastAPI(title="openlab workspace", version=__version__, lifespan=lifespan)
    reviewer = secrets.token_urlsafe(18)          # printed in the browser URL only; never in the agent's env
    app.state.ctx, app.state.token, app.state.hub, app.state.reviewer = ctx, token, hub, reviewer
    app.state.ctx_for = ctx_for

    def authed(authorization: str = Header(default=""), t: str = Query(default="")):
        given = authorization.split(" ", 1)[1].strip() if authorization.lower().startswith("bearer ") else t
        if token and not secrets.compare_digest(given or "", token):
            raise HTTPException(401, "missing or wrong token")

    @app.get("/api/health")
    def health():
        return {"ok": True, "version": __version__, "root": str(ctx.root), "tools": len(TOOLS)}

    @app.get("/api/doctor", dependencies=[Depends(authed)])
    def doctor():
        """`openlab doctor` for this workspace: the start screen's readiness line."""
        from ..doctor import check
        return check(ctx.root).to_record()

    @app.get("/api/guide", dependencies=[Depends(authed)])
    def guide():
        from fastapi.responses import PlainTextResponse
        from ..thin_cli import local_guide
        return PlainTextResponse(local_guide())

    @app.get("/api/tools", dependencies=[Depends(authed)])
    def tools(section: str = ""):
        return dispatch(ctx, "tools", {"section": section})

    @app.get("/api/workspaces", dependencies=[Depends(authed)])
    def list_workspaces():
        """Each research under home, with what the tab strip shows: live agents, pending inbox items, spend."""
        from .show_home import read
        from .show_tools import summary
        closed = set(read(home)["closed"])
        ws = [{"name": n, "path": str(f), "closed": n in closed, **summary(f)} for n, f in workspaces(home).items()]
        default = home_name(home) if (home / "openlab.toml").is_file() and home_name(home) not in closed else ""
        from .show_tools import settings
        return {"home": str(home), "default": default or next((w["name"] for w in ws if not w["closed"]), ""),
                "workspaces": ws, "settings": settings(home / ".openlab-show-defaults")}     # no such folder: defaults

    def person_only(x_openlab_reviewer: str = Header(default="")):
        if not (x_openlab_reviewer and secrets.compare_digest(x_openlab_reviewer, reviewer)):
            raise HTTPException(403, "a person's act: open the page from the link openlab show printed")

    @app.post("/api/workspaces/{name}/tab", dependencies=[Depends(authed)])
    async def tab(name: str, request: Request):
        """Show or hide a research's tab ({"open": false} closes it). Its agents and record are untouched."""
        from .show_home import set_open
        body = await request.json()
        if name not in workspaces(home):
            return JSONResponse({"ok": False, "error": "no such research", "meaning": f"{name} is not under {home}"},
                                status_code=404)
        closed = set_open(home, name, bool(body.get("open")))
        hub.publish({"channel": "workspaces", "names": list(workspaces(home))})
        return {"ok": True, "result": {"name": name, "closed": closed}}

    @app.get("/api/person", dependencies=[Depends(authed)])
    def person_check(x_openlab_reviewer: str = Header(default="")):
        """Whether this page's link still carries this run's reviewer secret: a new run of openlab show makes a new
        one, and a page opened from an older link may read but not act."""
        return {"ok": True, "person": bool(x_openlab_reviewer) and secrets.compare_digest(x_openlab_reviewer, reviewer)}

    @app.get("/api/machines", dependencies=[Depends(authed)])
    async def machines_now():
        """Every machine the Machine queue lists, with its usage: this one (local) and the remote ones declared."""
        from .show_home import machines
        return {"ok": True, "result": await asyncio.get_running_loop().run_in_executor(None, machines, home, 0.8)}

    @app.post("/api/machines", dependencies=[Depends(authed), Depends(person_only)])
    async def machines_edit(request: Request):
        """Declare a remote machine ({"add": {name, ssh, port}}) or drop one ({"remove": name}): a person's edit."""
        from .show_home import add_machine, remove_machine
        body = await request.json()
        try:
            if isinstance(body.get("add"), dict):
                a = body["add"]
                out = add_machine(home, a.get("name"), a.get("ssh"), int(a.get("port") or 0))
            elif body.get("remove"):
                out = remove_machine(home, str(body["remove"]))
            else:
                raise ValueError('send {"add": {"name": ..., "ssh": ...}} or {"remove": name}')
        except (ValueError, TypeError) as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
        hub.publish({"channel": "workspaces", "names": list(workspaces(home))})
        return {"ok": True, "result": out}

    @app.post("/api/workspaces", dependencies=[Depends(authed)])
    async def new_workspace(request: Request):
        """A new research: a folder under home set up as its own workspace (openlab init with the Claude hooks), and
        its lead agent spawned there with the prompt and budget given."""
        try:
            body = await request.json()
        except json.JSONDecodeError:
            body = None
        if not isinstance(body, dict):
            return JSONResponse({"ok": False, "error": "bad arguments", "meaning": "the body must be a JSON object",
                                 "try": 'send {"name": ..., "prompt": ...}'}, status_code=400)
        name = str(body.get("name") or "").strip().lower()
        if not _NAME.fullmatch(name) or name == home.name:      # home's own name is the server's root, not a tab
            return JSONResponse({"ok": False, "error": "bad name", "meaning": "a research's name is its folder: a-z, 0-9, "
                                 ". _ -, at most 48", "try": "e.g. stations-2026"}, status_code=400)
        chosen, bad = [], ""
        from .show_tools import _check
        for key, value in (body.get("settings") or {}).items() if isinstance(body.get("settings"), dict) else ():
            try:
                chosen.append((key, _check(key, value)))
            except ValueError as e:
                bad = str(e)
        if bad:
            return JSONResponse({"ok": False, "error": "bad setting", "meaning": bad, "try": "fix it in the form"},
                                status_code=400)
        folder = home / name
        try:
            if folder.is_symlink():                             # a dangling link with the name: taken, not ours
                raise FileExistsError(str(folder))
            folder.mkdir()                                      # the lock: of two POSTs with one name, one makes it
        except FileExistsError:
            return JSONResponse({"ok": False, "error": "exists", "meaning": f"{folder} is already there",
                                 "try": "pick another name, or open its tab"}, status_code=409)
        except OSError as e:
            return JSONResponse({"ok": False, "error": "cannot make the folder", "meaning": str(e),
                                 "try": "check the root folder is writable"}, status_code=500)

        def make():
            from ..init import init
            try:
                init(folder)
            except Exception as e:                              # noqa: BLE001 -- a half-made folder is not left behind
                shutil.rmtree(folder, ignore_errors=True)
                return {"ok": False, "error": "openlab init failed", "meaning": f"{type(e).__name__}: {e}",
                        "try": "the folder was removed; try again"}, False
            try:
                from ..agents.claude_setup import init_claude
                init_claude(folder)
            except Exception:                                   # noqa: BLE001 -- hooks are optional; the record is not
                pass
            from .show_home import set_open
            set_open(home, name, True)                          # a name closed before opens again
            from .show_tools import set_setting
            for key, value in chosen:                           # the research's settings, chosen in New session
                try:
                    set_setting(folder, key, value)
                except ValueError:                              # checked above; a hand-broken template keeps its own
                    pass
            if not str(body.get("prompt") or "").strip():
                return {"ok": True, "result": {"name": name, "agent": None}}, True
            try:
                return start_lead(name, body), True
            except Exception as e:                              # noqa: BLE001 -- the workspace stands without its lead
                return {"ok": False, "error": f"{type(e).__name__}: {e}", "meaning": "", "try": ""}, True
        env, made = await asyncio.get_running_loop().run_in_executor(None, make)
        if not made:
            return JSONResponse(env, status_code=500)
        hub.publish({"channel": "workspaces", "names": list(workspaces(home))})
        if env.get("ok"):
            res = env.get("result") or {}
            return {"ok": True, "result": {"name": name, "agent": res.get("id") if "id" in res else res.get("agent")}}
        return {**env, "made": True, "result": {"name": name, "agent": None},
                "note": f"the workspace {name} was made; the lead did not start: {env.get('error')}",
                "try": env.get("try") or f"POST /api/workspaces/{name}/lead with the prompt to start it"}

    @app.post("/api/workspaces/{name}/lead", dependencies=[Depends(authed)])
    async def new_lead(name: str, request: Request):
        """Start the lead of an existing research that has none running (a retry after its spawn failed)."""
        body = await request.json()
        if not str(body.get("prompt") or "").strip():
            return JSONResponse({"ok": False, "error": "no prompt", "meaning": "a lead needs a prompt",
                                 "try": 'POST {"prompt": "...", "budget_usd": 5}'}, status_code=400)
        c = ctx_for(name)

        def go():
            from ..agents.store import Store
            with lead_lock:                                     # two retries at once start one lead
                st = Store(c.root)
                st.reap()
                live = [a["id"] for a in st.agents(live_only=True) if a.get("role") == "lead"]
                if live:
                    return {"ok": False, "error": "a lead is running", "meaning": f"{name} is led by {live[0]}",
                            "try": "open its tab; a second agent goes through agent_spawn"}, 409
                return start_lead(name, body), 200
        env, status = await asyncio.get_running_loop().run_in_executor(None, go)
        if env.get("ok"):
            res = env.get("result") or {}
            return {"ok": True, "result": {"name": name, "agent": res.get("id")}}
        return JSONResponse(env, status_code=status)

    @app.get("/api/file", dependencies=[Depends(authed)])
    def file(path: str = Query(...), ws: str = ""):
        """A file of a workspace (a paper's PDF, a release's README or code), jailed to the workspace and the vault
        closed, as ToolContext.path holds every tool."""
        c = ctx_for(ws)
        try:
            p = c.path(path)
        except ToolError as e:
            raise HTTPException(403, e.meaning or e.error)
        if ".openlab" in p.relative_to(c.root).parts:          # the record and the control store are openlab's only
            raise HTTPException(403, "the .openlab folder is openlab's own: read the record through the tools")
        if not p.is_file():
            raise HTTPException(404, f"no file {path}")
        return FileResponse(p, headers={"Cache-Control": "no-cache"})

    @app.post("/api/tool/{name}", dependencies=[Depends(authed)])
    async def call(name: str, request: Request, x_openlab_reviewer: str = Header(default=""),
                   x_openlab_caller: str = Header(default=""), x_openlab_agent: str = Header(default=""),
                   ws: str = Query(default="")):
        ctx = ctx_for(ws)
        try:
            args = await request.json() if (await request.body()) else {}
        except json.JSONDecodeError:
            return JSONResponse({"ok": False, "tool": name, "error": "bad json", "meaning": "the body is not JSON",
                                 "try": "send a JSON object of arguments"}, status_code=400)
        if not isinstance(args, dict):
            return JSONResponse({"ok": False, "tool": name, "error": "bad arguments", "meaning": "the body must be an object",
                                 "try": "send a JSON object of arguments"}, status_code=400)
        human = bool(x_openlab_reviewer) and secrets.compare_digest(x_openlab_reviewer, reviewer)
        caller = "agent" if x_openlab_caller == "agent" else ""
        passphrase = args.pop("passphrase", None)
        from ..agents.identity import claimed
        agent = claimed(x_openlab_agent[:120]) if x_openlab_agent else ""
        t = TOOLS.get(name)
        light = (name == "agent_control" and args.get("action") in ("pause", "stop")    # holding an agent: unsigned
                 or name == "agent_control_leads" and args.get("action") == "pause"
                 or name in ("inbox_answer", "agent_prompt", "agent_prompts",     # a question's answer, a prompt,
                             "agent_btw")                                         # a side question: advice
                 or name in ("paper_comment", "paper_comment_reply",              # the draft: comments, the prose
                             "show_draft_save", "show_draft_preview",             # and its preview, unsigned;
                             "show_draft_init",                                   # starting it from the template
                             "show_settings_set",                                 # a setting: logged, unsigned
                             "show_conversation",                                 # reading a transcript
                             "show_save", "show_comment"))                        # the Workspace: prose, comments
        if human and t is not None and t.human and not (light and not passphrase):
            # A person's act is signed with the person key. The browser sends the passphrase in the request
            # body (never stored, never logged); the server proves the person by opening the key with it and runs
            # the act signed. Without it the act is refused, reviewer secret or not.
            if not passphrase:
                return {"ok": False, "tool": name, "error": "passphrase", "needs": "passphrase",
                        "meaning": f"{name} is a person's act, signed with your person key",
                        "try": "send your passphrase as `passphrase` in the request body (the app asks for it)"}

            def signed():
                from ..agents import person
                try:
                    with person.with_passphrase(str(passphrase), summary=f"{name} from the app"):
                        return dispatch(ctx, name, args, human, caller, agent)
                except person.NotAPerson as e:
                    return {"ok": False, "tool": name, "error": "passphrase", "needs": "passphrase", "meaning": str(e),
                            "try": "check the passphrase (`openlab person show` names the key)"}
            return await asyncio.get_running_loop().run_in_executor(None, signed)
        return await asyncio.get_running_loop().run_in_executor(None, dispatch, ctx, name, args, human, caller,
                                                                agent)     # the acting agent

    @app.post("/api/upload", dependencies=[Depends(authed)])
    async def upload(request: Request, path: str = Query(...), overwrite: bool = False):
        """Bring your own file: the raw body written to `path` in the workspace. A data format only, at most
        UPLOAD_MAX bytes, inside the workspace and outside the vault (ToolContext.path), never over an existing
        file unless `overwrite`. The reply is a tool envelope."""
        def refuse(error: str, meaning: str, try_: str, status: int = 400):
            return JSONResponse({"ok": False, "tool": "upload", "error": error, "meaning": meaning, "try": try_},
                                status_code=status)
        if not path.lower().endswith(UPLOAD_SUFFIXES):
            return refuse("not a data file", f"{path!r} is not a format openlab reads",
                          f"upload one of {', '.join(UPLOAD_SUFFIXES)}")
        if int(request.headers.get("content-length") or 0) > UPLOAD_MAX:
            return refuse("too large", f"{path!r} is over {UPLOAD_MAX >> 20} MB", "put it in the workspace folder "
                          "on disk", 413)
        try:
            p = ctx.path(path)
        except ToolError as e:
            return refuse(e.error, e.meaning, e.try_, 403)
        if p.exists() and not overwrite:
            return refuse("exists", f"{ctx.rel(p)} is already in the workspace", "replace it, or rename your file", 409)
        p.parent.mkdir(parents=True, exist_ok=True)
        part, size = p.with_name(f".{p.name}.upload"), 0
        try:
            with open(part, "wb") as f:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > UPLOAD_MAX:
                        return refuse("too large", f"{path!r} is over {UPLOAD_MAX >> 20} MB", "put it in the "
                                      "workspace folder on disk", 413)
                    f.write(chunk)
            os.replace(part, p)
        finally:
            part.unlink(missing_ok=True)
        ctx.emit("workspace", {"tool": "upload", "args": {"path": ctx.rel(p)}})
        return {"ok": True, "tool": "upload", "result": {"path": ctx.rel(p), "bytes": size}}

    @app.websocket("/ws/events")
    async def events(ws: WebSocket, t: str = ""):
        if token and not secrets.compare_digest(t or "", token):
            await ws.close(code=4401)
            return
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue()
        hub.queues.add(q)
        gone = asyncio.ensure_future(ws.receive())          # a closed page ends the handler, not the next event
        try:
            await ws.send_json({"channel": "hello", "version": __version__})
            while True:
                nxt = asyncio.ensure_future(q.get())
                done, _ = await asyncio.wait({nxt, gone}, return_when=asyncio.FIRST_COMPLETED)
                if gone in done:
                    nxt.cancel()
                    msg = gone.result()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    gone = asyncio.ensure_future(ws.receive())  # a client message: ignored, keep listening
                    if nxt.done() and not nxt.cancelled():
                        await ws.send_json(nxt.result())
                    continue
                await ws.send_json(nxt.result())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            gone.cancel()
            hub.queues.discard(q)

    @app.websocket("/ws/terminal")
    async def terminal(ws: WebSocket, t: str = ""):
        """The agent pane: a terminal running the user's own agent CLI (``$OPENLAB_AGENT_CMD``, else ``claude`` if
        it is installed; with neither it closes with 4404) in the workspace. openlab holds no model key: the user signs the CLI in.
        The child gets OPENLAB_APP_URL / OPENLAB_APP_TOKEN, so its openlab MCP acts on this server's sessions."""
        if token and not secrets.compare_digest(t or "", token):
            await ws.close(code=4401)
            return
        # on unless switched off: the agent is the point of the product, and the pane is behind the same
        # token as every other route. OPENLAB_APP_TERMINAL=0 keeps a shared or headless server from offering it.
        if os.environ.get("OPENLAB_APP_TERMINAL", "1") in ("0", "false"):
            await ws.close(code=4403)
            return
        import fcntl
        import pty
        import shlex
        import signal
        import struct
        import subprocess
        import termios
        from ..doctor import agent_cmd
        await ws.accept()
        cmd = agent_cmd()
        if not cmd:                               # never a bare shell: the pane says how to install an agent
            await ws.close(code=4404)
            return
        with contextlib.suppress(OSError):
            write_mcp_json(ctx.root)
        master, slave = pty.openpty()
        env = {**os.environ, "TERM": "xterm-256color", "OPENLAB_APP_TOKEN": token or "",
               "OPENLAB_APP_URL": os.environ.get("OPENLAB_APP_SELF") or f"http://127.0.0.1:{os.environ.get('OPENLAB_APP_PORT', '8321')}",
               "OPENLAB_WORKSPACE": str(ctx.root)}
        proc = subprocess.Popen(shlex.split(cmd), stdin=slave, stdout=slave, stderr=slave, cwd=str(ctx.root), env=env,
                                start_new_session=True, close_fds=True)
        os.close(slave)
        loop = asyncio.get_running_loop()
        out: asyncio.Queue = asyncio.Queue()

        def readable():
            try:
                data = os.read(master, 65536)
            except OSError:
                data = b""
            out.put_nowait(data)
            if not data:
                loop.remove_reader(master)
        loop.add_reader(master, readable)

        async def pump():
            while True:
                data = await out.get()
                if not data:
                    break
                await ws.send_text(data.decode("utf-8", "replace"))
        pumping = asyncio.ensure_future(pump())
        try:
            while True:
                msg = await ws.receive_text()
                if msg.startswith('{"resize"'):
                    try:
                        cols, rows = json.loads(msg)["resize"]
                        fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", int(rows), int(cols), 0, 0))
                        continue
                    except (ValueError, KeyError, OSError):
                        pass
                os.write(master, msg.encode())
        except (WebSocketDisconnect, RuntimeError, OSError):
            pass
        finally:
            pumping.cancel()
            with contextlib.suppress(Exception):
                loop.remove_reader(master)
            with contextlib.suppress(ProcessLookupError, OSError):
                os.killpg(proc.pid, signal.SIGHUP)
            with contextlib.suppress(OSError):
                os.close(master)

    @app.get("/show", include_in_schema=False)
    @app.get("/show/{path:path}", include_in_schema=False)
    def show(path: str = ""):
        f = (SHOW / (path or "index.html")).resolve()
        if SHOW in f.parents and f.is_file():
            return FileResponse(f, headers={"Cache-Control": "no-cache"})
        raise HTTPException(404)

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith(("api/", "ws/")):
            raise HTTPException(404)
        f = (STATIC / path).resolve()
        if path and STATIC in f.parents and f.is_file():
            return FileResponse(f)
        index = STATIC / "index.html"
        return FileResponse(index) if index.is_file() else HTMLResponse(PLACEHOLDER)

    return app


def show_main(argv=None) -> int:
    """`openlab show`: the same server, opened on the live view (/show)."""
    return main(argv, prog="openlab show", page="show/")


def main(argv=None, prog: str = "openlab app", page: str = "") -> int:
    import argparse
    p = argparse.ArgumentParser(prog=prog, description="Serve the openlab workspace (web UI + API)." if not page
                                else "Serve the live view of the agents, their plans, the inbox and the record.")
    p.add_argument("--root", default=os.environ.get("OPENLAB_WORKSPACE", "."), help="the workspace folder; openlab show also draws a tab for each workspace directly under it")
    p.add_argument("--host", default=os.environ.get("OPENLAB_APP_HOST", "127.0.0.1"),
                   help="the address to bind (default: $OPENLAB_APP_HOST, else 127.0.0.1)")
    p.add_argument("--port", type=int, default=int(os.environ.get("OPENLAB_APP_PORT", "8321")),
                   help="the port (default: $OPENLAB_APP_PORT, else 8321)")
    p.add_argument("--token", default=None, help="the bearer token (default: $OPENLAB_APP_TOKEN, else minted)")
    a = p.parse_args(argv)
    r = Path(a.root).expanduser()
    if not r.exists():
        p.error(f"--root {a.root}: no such folder (give the folder that holds your researches)")
    if not r.is_dir():
        p.error(f"--root {a.root} is a file, not a folder (give the folder that holds your researches)")
    import uvicorn
    os.environ["OPENLAB_APP_PORT"] = str(a.port)
    os.environ.setdefault("OPENLAB_APP_SELF", f"http://127.0.0.1:{a.port}")
    app = create_app(a.root, a.token)
    print(f"openlab workspace {Path(a.root).resolve()}\n  http://{a.host}:{a.port}/{page}?t={app.state.token}"
          f"&r={app.state.reviewer}", flush=True)
    if page and a.host in ("127.0.0.1", "localhost"):
        print(f"  over ssh: ssh -L {a.port}:localhost:{a.port} HOST, then open the link here", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning",
                timeout_graceful_shutdown=3)            # an open page or terminal never holds a stop (SIGTERM) for long
    return 0
