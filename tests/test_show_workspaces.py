"""openlab show over many researches: the tabs under one root, a new research whose lead fails to start and its
retry, two new researches with one name, the root's own tab, events naming their research, a folder renamed or
deleted while the server runs, and a --root that is not a folder."""

import shutil
import threading

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from openlab.app import server  # noqa: E402
from openlab.app.server import home_name, workspaces  # noqa: E402

H = {"authorization": "Bearer tok"}


def _ws(folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "openlab.toml").write_text("")
    return folder


def test_a_lead_that_does_not_start_leaves_a_workspace_and_can_be_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(server.shutil, "which", lambda name: None)      # no claude on this server
    monkeypatch.delenv("OPENLAB_CLAUDE_BIN", raising=False)
    with TestClient(server.create_app(str(tmp_path), token="tok")) as c:
        env = c.post("/api/workspaces", headers=H, json={"name": "gbm", "prompt": "forecast it"}).json()
        assert not env["ok"] and env["made"] and env["result"] == {"name": "gbm", "agent": None}
        assert "was made; the lead did not start: claude is not installed" in env["note"]
        assert "/api/workspaces/gbm/lead" in env["try"]
        assert (tmp_path / "gbm" / "openlab.toml").is_file()
        assert "gbm" in [w["name"] for w in c.get("/api/workspaces", headers=H).json()["workspaces"]]
        assert c.post("/api/tool/show_snapshot?ws=gbm", headers=H).json()["ok"]
        # the retry: a failing spawn says why; a working one starts the lead; a second is refused
        assert c.post("/api/workspaces/gbm/lead", headers=H, json={}).status_code == 400
        assert c.post("/api/workspaces/gbm/lead", headers=H, json={"prompt": "go"}).json()["error"] == \
            "claude is not installed"
        calls = []

        def fake(name, body):
            calls.append(name)
            from openlab.agents.store import Store
            Store(tmp_path / name).register("lead-x", "spawned", role="lead", status="running")
            return {"ok": True, "result": {"id": "lead-x"}}
        monkeypatch.setattr(server, "_cli_missing", lambda cli, name: None)
        monkeypatch.setattr("openlab.app.server.dispatch", lambda ctx, tool, args: fake(ctx.root.name, args))
        monkeypatch.setattr("openlab.agents.store.Store.reap", lambda self: None)
        assert c.post("/api/workspaces/gbm/lead", headers=H, json={"prompt": "go"}).json() == \
            {"ok": True, "result": {"name": "gbm", "agent": "lead-x"}}
        r = c.post("/api/workspaces/gbm/lead", headers=H, json={"prompt": "go"})
        assert r.status_code == 409 and "lead-x" in r.json()["meaning"] and calls == ["gbm"]
        assert c.post("/api/workspaces/nope/lead", headers=H, json={"prompt": "go"}).status_code == 404


def test_a_spawn_that_raises_still_reports_the_workspace_made(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_cli_missing", lambda cli, name: None)

    def boom(ctx, tool, args):
        raise RuntimeError("spawn broke")
    monkeypatch.setattr("openlab.app.server.dispatch", boom)
    with TestClient(server.create_app(str(tmp_path), token="tok")) as c:
        env = c.post("/api/workspaces", headers=H, json={"name": "w", "prompt": "p"}).json()
        assert env["made"] and "spawn broke" in env["note"] and (tmp_path / "w" / "openlab.toml").is_file()


def test_two_new_researches_with_one_name_make_one(tmp_path):
    with TestClient(server.create_app(str(tmp_path), token="tok")) as c:
        go, out = threading.Barrier(4), []

        def post():
            go.wait()
            out.append(c.post("/api/workspaces", headers=H, json={"name": "same", "prompt": ""}).status_code)
        ts = [threading.Thread(target=post) for _ in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert sorted(out) == [200, 409, 409, 409]
        assert (tmp_path / "same" / "openlab.toml").is_file()


def test_the_roots_own_tab_never_takes_a_sub_workspaces_name(tmp_path):
    home = _ws(tmp_path / "research")
    _ws(home / "research")                       # a research named like the root
    _ws(home / "other")
    found = workspaces(home)
    assert found == {"research~": home, "other": home / "other", "research": home / "research"}
    assert home_name(home) == "research~"
    with TestClient(server.create_app(str(home), token="tok")) as c:
        r = c.get("/api/workspaces", headers=H).json()
        assert r["default"] == "research~" and len({w["name"] for w in r["workspaces"]}) == 3
        snap = lambda ws: c.post(f"/api/tool/show_snapshot?ws={ws}", headers=H).json()["result"]["workspace"]
        assert snap("research").endswith("research/research") and not snap("research~").endswith("research/research")


def test_a_workspace_nested_in_a_study_is_not_a_tab(tmp_path):
    _ws(tmp_path)
    (tmp_path / "study1").mkdir()
    _ws(tmp_path / "study1" / "inner")
    _ws(tmp_path / ".hidden")
    assert list(workspaces(tmp_path)) == [tmp_path.name]
    with TestClient(server.create_app(str(tmp_path), token="tok")) as c:
        assert c.post("/api/tool/show_snapshot?ws=inner", headers=H).status_code == 404


def test_every_event_names_its_research_the_roots_too(tmp_path):
    _ws(tmp_path)
    _ws(tmp_path / "sub")
    app = server.create_app(str(tmp_path), token="tok")
    seen = []
    app.state.hub.publish = seen.append
    app.state.ctx.emit("workspace", {"tool": "x"})
    app.state.ctx_for("sub").emit("workspace", {"tool": "y"})
    app.state.ctx_for(tmp_path.name).emit("workspace", {"tool": "z"})
    assert [e["ws"] for e in seen] == [tmp_path.name, "sub", tmp_path.name]


def test_a_folder_renamed_or_deleted_at_runtime_is_forgotten(tmp_path):
    _ws(tmp_path / "a")
    with TestClient(server.create_app(str(tmp_path), token="tok")) as c:
        snap = lambda ws: c.post(f"/api/tool/show_snapshot?ws={ws}", headers=H)
        assert snap("a").json()["ok"]
        (tmp_path / "a").rename(tmp_path / "b")
        assert snap("a").status_code == 404 and not (tmp_path / "a").exists()   # not recreated by a stale context
        assert snap("b").json()["result"]["workspace"].endswith("/b")
        shutil.rmtree(tmp_path / "b")
        assert snap("b").status_code == 404 and not (tmp_path / "b").exists()
        _ws(tmp_path / "x")
        assert snap("x").json()["ok"]
        (tmp_path / "x").rename(tmp_path / "y")
        _ws(tmp_path / "x")                       # the name reused by another folder: a new context
        (tmp_path / "x" / "marker").write_text("")
        assert snap("x").json()["result"]["workspace"].endswith("/x")
        assert c.get("/api/file?ws=x&path=marker&t=tok").status_code == 200


@pytest.mark.parametrize("make", ["file", "missing"])
def test_a_root_that_is_not_a_folder_is_a_clear_error(tmp_path, capsys, make):
    root = tmp_path / "r"
    if make == "file":
        root.write_text("not a folder")
    with pytest.raises(SystemExit) as e:
        server.show_main(["--root", str(root)])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert ("is a file, not a folder" if make == "file" else "no such folder") in err
    assert root.exists() == (make == "file")      # a missing root is not created
