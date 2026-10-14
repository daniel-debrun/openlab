"""The show server's watcher and its release compiles: a restart neither recompiles history nor misses a lead that
ended while it was down; one study compiles once at a time; a slow or failing compile, a locked store, a broken or
deleted workspace never stop the watcher; [show] compile_on_done = false stops the compiles."""

import json
import shutil
import sqlite3
import threading
import time

import pytest

from openlab.agents.store import Store
from openlab.app import ToolContext, dispatch


def _until(cond, seconds=5.0):
    end = time.time() + seconds
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def _summaries(root, agent):
    return [e["summary"] for e in Store(root).events(agent, n=50)]


@pytest.fixture
def server(monkeypatch):
    pytest.importorskip("fastapi")
    from openlab.app import server
    monkeypatch.setattr(server, "WATCH_SECONDS", 0.03)
    return server


@pytest.fixture
def fake_build(monkeypatch):
    """A release build that takes `delay` seconds and counts its calls (the real one is tested in test_show)."""
    import openlab.study.release as release
    calls, lock = [], threading.Lock()
    state = {"delay": 0.0, "raise": None}

    def build(study):
        with lock:
            calls.append(study.folder)
        time.sleep(state["delay"])
        if state["raise"]:
            raise state["raise"]
        rel = study.folder / "release"
        rel.mkdir(exist_ok=True)
        (rel / "MANIFEST.json").write_text(json.dumps({"record_head": ""}))
        return {"paper": "none", "why": "", "files": ["MANIFEST.json"]}
    monkeypatch.setattr(release, "build", build)
    return calls, state


def _study(root, name="s"):
    (root / name).mkdir(parents=True, exist_ok=True)
    (root / name / "study.toml").write_text("")
    return root / name


def _ws(root, toml=""):
    root.mkdir(parents=True, exist_ok=True)
    (root / "openlab.toml").write_text(toml)
    Store(root).event("boot", "note", summary="the store exists")
    return root


def _done(root, *ids):
    """End leads done in one write, then poke the store so the watcher looks (as the supervisor does)."""
    con = sqlite3.connect(root / ".openlab" / "control.db", isolation_level=None)
    con.execute(f"UPDATE agents SET status='done' WHERE id IN ({','.join('?' * len(ids))})", ids)
    con.close()
    Store(root).event(ids[0], "end", summary="done")


def test_a_restart_compiles_only_what_ended_while_it_was_down(tmp_path):
    from openlab import study
    from openlab.app.show_tools import finished_leads
    from openlab.study.release import build
    ctx = ToolContext(str(tmp_path))
    for name, args in [("study_new", {"folder": "s", "template": "static"}), ("study_materialise", {"folder": "s"}),
                       ("study_run", {"folder": "s"})]:
        assert dispatch(ctx, name, args)["ok"]
    st = Store(tmp_path)
    st.register("old", "spawned", role="lead", status="done", study="s")
    build(study.open(tmp_path / "s"))                                   # the release the server last compiled
    assert finished_leads(tmp_path, set(), prime=True) == []            # nothing new since: no recompile
    time.sleep(0.05)
    st.register("missed", "spawned", role="lead", status="done", study="s")
    assert finished_leads(tmp_path, set(), prime=True) == []            # ended after it, but the record has not moved
    assert dispatch(ctx, "study_run", {"folder": "s", "note": "while the server was down"})["ok"]
    jobs = finished_leads(tmp_path, set(), prime=True)
    assert len(jobs) == 1 and jobs[0].agents == ["missed"]              # 'old' ended before the release: history
    # a lead that ended before the release was written is history, even though the record moved since
    con = sqlite3.connect(tmp_path / ".openlab" / "control.db", isolation_level=None)
    con.execute("UPDATE agents SET updated='2000-01-01T00:00:00.000+00:00' WHERE id='missed'")
    con.close()
    assert finished_leads(tmp_path, set(), prime=True) == []


def test_two_leads_of_a_study_ending_together_compile_once(tmp_path, server, fake_build):
    from fastapi.testclient import TestClient
    calls, state = fake_build
    root = _ws(tmp_path / "w")
    _study(root)
    st = Store(root)
    for a in ("l1", "l2", "l3"):
        st.register(a, "spawned", role="lead", status="running", study="s")
    state["delay"] = 0.6
    with TestClient(server.create_app(str(root), token="tok")) as c:
        time.sleep(0.15)                                                # the first look (prime) is done
        _done(root, "l1", "l2")
        assert _until(lambda: len(calls) == 1)
        time.sleep(0.1)
        _done(root, "l3")                                               # ends while the study compiles
        assert _until(lambda: any("release" in s for s in _summaries(root, "l3")))
        assert _until(lambda: not c.app.state.compiling)
    assert len(calls) == 1                                              # the folded build saw the release current
    assert any("release compiled" in s for s in _summaries(root, "l1"))
    assert any("release compiled" in s for s in _summaries(root, "l2"))
    assert any("already current" in s for s in _summaries(root, "l3"))


def test_a_slow_compile_does_not_hold_the_watcher_up(tmp_path, server, fake_build, monkeypatch):
    from fastapi.testclient import TestClient
    from openlab.app import show_tools
    calls, state = fake_build
    state["delay"] = 1.5
    looks = []
    real = show_tools.changed
    monkeypatch.setattr(show_tools, "changed", lambda folder: (looks.append(1), real(folder))[1])
    root = _ws(tmp_path / "w")
    _study(root)
    Store(root).register("l1", "spawned", role="lead", status="running", study="s")
    with TestClient(server.create_app(str(root), token="tok")):
        time.sleep(0.15)
        _done(root, "l1")
        assert _until(lambda: len(calls) == 1)
        n = len(looks)
        time.sleep(0.5)                                                 # the build is still sleeping
        assert len(looks) >= n + 5


def test_a_failing_compile_is_a_control_store_event_and_the_watcher_goes_on(tmp_path, server, fake_build, monkeypatch):
    from fastapi.testclient import TestClient
    from openlab.app import show_tools
    calls, state = fake_build
    state["raise"] = RuntimeError("latex exploded")
    root = _ws(tmp_path / "w")
    _study(root)
    _study(root, "t")
    st = Store(root)
    st.register("l1", "spawned", role="lead", status="running", study="s")
    st.register("l2", "spawned", role="lead", status="running", study="t")
    with TestClient(server.create_app(str(root), token="tok")) as c:
        time.sleep(0.15)
        _done(root, "l1")
        assert _until(lambda: any("latex exploded" in s for s in _summaries(root, "l1")))
        # an exception escaping the job itself is logged too, and the next lead still compiles
        monkeypatch.setattr(show_tools.ReleaseJob, "__call__", lambda self: (_ for _ in ()).throw(ValueError("boom")))
        _done(root, "l2")
        assert _until(lambda: any("release not compiled: ValueError: boom" in s for s in _summaries(root, "l2")))
        with c.websocket_connect("/ws/events?t=tok") as sock:
            assert sock.receive_json()["channel"] == "hello"
            Store(root).event("l1", "note", summary="still watched")
            assert sock.receive_json()["channel"] == "control"


def test_one_broken_workspace_does_not_stop_the_others(tmp_path, server, monkeypatch):
    from fastapi.testclient import TestClient
    from openlab.app import show_tools
    real = show_tools.changed

    def changed(folder):
        if folder.name == "bad":
            raise RuntimeError("a broken workspace")
        return real(folder)
    monkeypatch.setattr(show_tools, "changed", changed)
    _ws(tmp_path / "bad")
    good = _ws(tmp_path / "good")
    with TestClient(server.create_app(str(tmp_path), token="tok")) as c:
        with c.websocket_connect("/ws/events?t=tok") as sock:
            assert sock.receive_json()["channel"] == "hello"
            time.sleep(0.15)
            Store(good).event("a", "note", summary="hello")
            ev = sock.receive_json()
            assert ev["channel"] == "control" and ev["ws"] == "good"


def test_changed_is_none_while_the_store_is_locked_and_reads_again_after(tmp_path):
    from openlab.app.show_tools import changed
    Store(tmp_path).event("a", "note", summary="x")
    before = changed(tmp_path)
    con = sqlite3.connect(tmp_path / ".openlab" / "control.db", isolation_level=None)
    con.execute("PRAGMA locking_mode=EXCLUSIVE")
    con.execute("BEGIN EXCLUSIVE")
    t = time.time()
    assert changed(tmp_path) is None and time.time() - t < 1.0         # busy: no fingerprint, no long stall
    con.execute("ROLLBACK")
    con.close()
    assert changed(tmp_path) == before                                  # the same: the watcher publishes nothing
    assert changed(tmp_path / "nowhere") is None


def test_a_workspace_deleted_while_watched_is_dropped_not_recreated(tmp_path, server, fake_build):
    from fastapi.testclient import TestClient
    calls, _ = fake_build
    gone = _ws(tmp_path / "gone")
    _study(gone)
    Store(gone).register("l1", "spawned", role="lead", status="running", study="s")
    stay = _ws(tmp_path / "stay")
    with TestClient(server.create_app(str(tmp_path), token="tok")) as c:
        time.sleep(0.15)
        with c.websocket_connect("/ws/events?t=tok") as sock:
            assert sock.receive_json()["channel"] == "hello"
            shutil.rmtree(gone)
            ev = sock.receive_json()
            assert ev["channel"] == "workspaces" and ev["names"] == ["stay"]
            Store(stay).event("a", "note", summary="still here")
            ev = sock.receive_json()
            assert ev["channel"] == "control" and ev["ws"] == "stay"
        time.sleep(0.15)
    assert not gone.exists() and calls == []


def test_compile_on_done_false_stops_the_compiles(tmp_path, server, fake_build):
    from fastapi.testclient import TestClient
    calls, _ = fake_build
    root = _ws(tmp_path / "w", "[show]\ncompile_on_done = false\n")
    _study(root)
    Store(root).register("l1", "spawned", role="lead", status="running", study="s")
    with TestClient(server.create_app(str(root), token="tok")) as c:
        with c.websocket_connect("/ws/events?t=tok") as sock:
            assert sock.receive_json()["channel"] == "hello"
            time.sleep(0.15)
            _done(root, "l1")
            assert sock.receive_json()["channel"] == "control"          # still watched
            time.sleep(0.3)
    assert calls == [] and not (root / "s" / "release").exists()
    # turning it back on does not compile the leads that ended while it was off
    from openlab.app.show_tools import finished_leads
    seen = set()
    finished_leads(root, seen)
    (root / "openlab.toml").write_text("")
    assert finished_leads(root, seen) == []


def test_a_lead_spawned_without_a_study_takes_the_one_it_makes_and_its_end_compiles(tmp_path):
    from openlab.app.show_tools import finished_leads
    _ws(tmp_path)
    st = Store(tmp_path)
    st.register("lead1", "spawned", role="lead", status="running")
    st.register("other", "spawned", role="lead", status="running", study="first")
    ctx = ToolContext(str(tmp_path))
    assert dispatch(ctx, "study_new", {"folder": "s", "template": "static"}, agent="lead1")["ok"]
    assert dispatch(ctx, "study_new", {"folder": "t", "template": "static"}, agent="lead1")["ok"]
    assert dispatch(ctx, "study_new", {"folder": "u", "template": "static"}, agent="other")["ok"]
    assert st.get("lead1")["study"] == "s"                              # its first study, kept
    assert st.get("other")["study"] == "first"                          # an assigned study is never replaced
    st.set_status("lead1", "done")
    jobs = finished_leads(tmp_path, set())
    assert [(j.rel, j.agents) for j in jobs] == [("s", ["lead1"])]


def test_a_lead_that_runs_in_a_study_another_made_takes_it(tmp_path):
    _ws(tmp_path)
    st = Store(tmp_path)
    st.register("maker", "spawned", role="lead", status="done", study="s")
    st.register("lead2", "spawned", role="lead", status="running")
    ctx = ToolContext(str(tmp_path))
    assert dispatch(ctx, "study_new", {"folder": "s", "template": "static"}, agent="maker")["ok"]
    dispatch(ctx, "study_run", {"folder": "s"}, agent="lead2")         # refused or not, it worked in s
    assert st.get("lead2")["study"] == "s"
