"""The workspace server and its other skins: HTTP, the events socket, the CLI, and the MCP over a live server."""

import asyncio
import json
import os
import socket
import sys
import threading
import time

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
from fastapi.testclient import TestClient   # noqa: E402
from starlette.websockets import WebSocketDisconnect   # noqa: E402


@pytest.fixture
def client(tmp_path):
    from openlab.app.server import create_app
    with TestClient(create_app(str(tmp_path), token="tok")) as c:
        yield c, tmp_path


H = {"authorization": "Bearer tok"}


def test_the_api_is_the_dispatch_envelope(client):
    c, root = client
    assert c.get("/api/health").json()["ok"]
    assert c.get("/api/tools").status_code == 401                                    # a token is required
    names = [t["name"] for t in c.get("/api/tools", headers=H).json()["result"]["tools"]]
    assert {"ws_list", "study_run", "paper_search"} <= set(names)
    env = c.post("/api/tool/ws_write", headers=H, json={"path": "a.txt", "text": "A"}).json()
    assert env["ok"] and env["result"]["path"] == "a.txt" and (root / "a.txt").exists()
    env = c.post("/api/tool/ws_read", headers=H, json={"path": "../x"}).json()
    assert not env["ok"] and env["error"] == "outside the workspace" and env["try"]
    assert c.post("/api/tool/nope", headers=H, json={}).json()["error"] == "unknown tool"
    doc = c.get("/api/doctor", headers=H).json()                                      # the start screen's readiness
    assert c.get("/api/doctor").status_code == 401 and {"ok", "rows"} <= set(doc)
    assert any(r["check"] == "workspace" for r in doc["rows"])                       # for the workspace
    index = c.get("/")                                                                # the SPA, or its placeholder
    assert index.status_code == 200 and "openlab" in index.text


def test_upload_writes_a_data_file_inside_the_workspace_only(client):
    c, root = client
    up = lambda path, body=b"a,b\n1,2\n", **q: c.post("/api/upload", params={"path": path, **q}, content=body, headers=H)
    assert c.post("/api/upload", params={"path": "x.csv"}, content=b"a\n").status_code == 401
    env = up("d/x.csv").json()
    assert env["ok"] and env["result"] == {"path": "d/x.csv", "bytes": 8} and (root / "d/x.csv").read_bytes() == b"a,b\n1,2\n"
    assert up("d/x.csv", b"new\n").json()["error"] == "exists" and (root / "d/x.csv").read_bytes() == b"a,b\n1,2\n"
    assert up("d/x.csv", b"new\n", overwrite=1).json()["ok"] and (root / "d/x.csv").read_bytes() == b"new\n"
    assert up("../x.csv").json()["error"] == "outside the workspace" and not (root.parent / "x.csv").exists()
    assert up("run.py").json()["error"] == "not a data file"


def test_events_stream_workspace_mutations(client):
    c, root = client
    with c.websocket_connect("/ws/events?t=tok") as ws:
        assert ws.receive_json()["channel"] == "hello"
        env = c.post("/api/tool/ws_write", headers=H, json={"path": "n.txt", "text": "x"}).json()
        assert env["ok"]
        seen = ws.receive_json()
    assert seen["channel"] == "workspace" and seen["tool"] == "ws_write" and seen["args"]["path"] == "n.txt"


def test_cli_tool_runs_the_same_dispatch(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OPENLAB_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("OPENLAB_APP_URL", raising=False)
    import importlib
    import openlab.app.client as C
    importlib.reload(C)
    from openlab.__main__ import main
    assert main(["tool", "ws_write", "path=cli.txt", "text=x", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["result"]["path"] == "cli.txt"
    assert main(["tool", "ws_list"]) == 0                               # lines and a table by default
    out = capsys.readouterr().out
    assert "cli.txt" in out and not out.lstrip().startswith("{")
    assert main(["tool", "ws_read", "path=../x"]) == 1
    assert capsys.readouterr().out.startswith("error: outside the workspace\nmeaning:")
    assert main(["tool"]) == 0 and "study_run" in capsys.readouterr().out
    assert main(["tools"]) == 0
    out = capsys.readouterr().out
    assert len(out.splitlines()) <= 19 and "study (" in out                # a line or two per section, wrapped and "study_holdout" not in out
    assert main(["tool", "study_run", "--help"]) == 0 and "arguments: folder:string" in capsys.readouterr().out


def test_the_mcp_acts_on_the_running_server(tmp_path):
    """An agent over the MCP, pointed at a live workspace server: its actions land in the server's workspace."""
    pytest.importorskip("mcp")
    import uvicorn
    from openlab.app.server import create_app
    app = create_app(str(tmp_path), token="tok")
    sock = socket.socket(); sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]; sock.close()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(200):
        try:
            if httpx.get(base + "/api/health", timeout=1).status_code == 200:
                break
        except Exception:
            time.sleep(0.05)
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def go():
        env = {**os.environ, "OPENLAB_APP_URL": base, "OPENLAB_APP_TOKEN": "tok"}
        params = StdioServerParameters(command=sys.executable, args=["-m", "openlab.mcp_server"], env=env)
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                names = [t.name for t in (await s.list_tools()).tools]
                J = lambda res: json.loads(res.content[0].text)
                made = J(await s.call_tool("openlab_tool", {"name": "ws_write",
                                                           "args": {"path": "agent_made.txt", "text": "by the agent"}}))
                gone = J(await s.call_tool("openlab_tool", {"name": "ws_delete", "args": {"path": "agent_made.txt"}}))
                return names, made, gone
    names, made, gone = asyncio.run(go())
    srv.should_exit = True
    assert {"openlab_tool", "openlab_study_run", "openlab_agent_spawn"} <= set(names) and "openlab_run" not in names
    assert made["ok"] and (tmp_path / "agent_made.txt").exists()                         # written by the server
    assert not gone["ok"] and gone["error"] == "confirm"                                # the ceremony holds for agents too


def test_the_agent_pane_is_a_terminal_in_the_workspace(tmp_path, monkeypatch):
    from openlab.app.server import create_app
    monkeypatch.setenv("OPENLAB_APP_TERMINAL", "0")
    with TestClient(create_app(str(tmp_path), token="tok")) as c:
        with pytest.raises(Exception):                                   # off when switched off
            with c.websocket_connect("/ws/terminal?t=tok") as ws:
                ws.receive_text()
    monkeypatch.delenv("OPENLAB_APP_TERMINAL")                            # on by default
    monkeypatch.delenv("OPENLAB_AGENT_CMD", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))                            # no agent CLI: closed, never a shell
    with TestClient(create_app(str(tmp_path), token="tok")) as c:
        with c.websocket_connect("/ws/terminal?t=tok") as ws:
            with pytest.raises(WebSocketDisconnect) as gone:
                ws.receive_text()
    assert gone.value.code == 4404
    monkeypatch.setenv("OPENLAB_AGENT_CMD", "/bin/sh")
    with TestClient(create_app(str(tmp_path), token="tok")) as c:
        with c.websocket_connect("/ws/terminal?t=tok") as ws:
            ws.send_text('{"resize":[100,30]}')
            ws.send_text("echo agent-$((40+2)); pwd; echo $OPENLAB_APP_TOKEN\n")
            seen = ""
            for _ in range(50):
                seen += ws.receive_text()
                if "agent-42" in seen and "tok" in seen.split("agent-42")[-1]:
                    break
    assert "agent-42" in seen and str(tmp_path) in seen                  # a real shell, in the workspace, with the server's token


def test_every_api_route_and_socket_needs_the_token(client):
    c, root = client
    from fastapi.routing import APIRoute
    routes = [(m, r.path) for r in c.app.routes if isinstance(r, APIRoute) and r.path.startswith("/api/")
              for m in r.methods if m != "HEAD"]
    assert len(routes) >= 8
    for m, path in routes:
        if path == "/api/health":                                          # the one open route: no workspace data
            continue
        url = path.replace("{name}", "show_snapshot")
        for headers, q in [({}, {}), ({"authorization": "Bearer nope"}, {}), ({}, {"t": "nope"}),
                           ({"authorization": "Basic tok"}, {})]:
            r = c.request(m, url, headers=headers, params={"path": "x.csv", "ws": "", **q}, json={"name": "zz"})
            assert r.status_code == 401, (m, path, headers, q)
    assert not (root / "zz").exists()
    for sock in ("/ws/events", "/ws/events?t=nope", "/ws/terminal", "/ws/terminal?t=tok2"):
        with pytest.raises(WebSocketDisconnect) as gone:
            with c.websocket_connect(sock) as s:
                s.receive_text()
        assert gone.value.code == 4401


def test_the_reviewer_secret_is_never_in_an_agents_env_or_a_snapshot(tmp_path, monkeypatch):
    from openlab.app.server import create_app
    monkeypatch.setenv("OPENLAB_AGENT_CMD", "/bin/sh")
    app = create_app(str(tmp_path), token="tok")
    secret = app.state.reviewer
    assert secret and secret != app.state.token and secret not in json.dumps(dict(os.environ))
    assert secret not in json.dumps(vars(app.state.ctx), default=str)  # nothing a tool or a spawn reads holds it
    with TestClient(app) as c:
        snap = c.post("/api/tool/show_snapshot", headers=H).json()
        assert snap["ok"] and secret not in json.dumps(snap)
        with c.websocket_connect("/ws/terminal?t=tok") as ws:
            ws.send_text("env; echo end-$((40+2))\n")
            seen = ""
            for _ in range(200):
                seen += ws.receive_text()
                if "end-42" in seen:
                    break
    assert "end-42" in seen and "OPENLAB_APP_TOKEN=tok" in seen and secret not in seen
