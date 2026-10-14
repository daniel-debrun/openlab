"""openlab show's home: closing and resuming tabs, the machines with their usage, and a research's settings."""

import pytest

from openlab.agents.store import Store
from openlab.app import ToolContext, dispatch

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from openlab.app import server, show_home  # noqa: E402

H = {"authorization": "Bearer tok"}


def ws(home, name):
    (home / name).mkdir()
    (home / name / "openlab.toml").write_text("[agents]\nbudget_usd = 2.0           # default budget\n")


def test_a_closed_tab_is_hidden_and_resume_opens_it(tmp_path):
    ws(tmp_path, "a"), ws(tmp_path, "b")
    with TestClient(server.create_app(str(tmp_path), token="tok")) as c:
        r = c.get("/api/workspaces", headers=H).json()
        assert [(w["name"], w["closed"]) for w in r["workspaces"]] == [("a", False), ("b", False)]
        assert {s["key"] for s in r["settings"]} >= {"budget_usd", "permission_mode"}       # New session's defaults
        assert c.post("/api/workspaces/a/tab", headers=H, json={"open": False}).json()["ok"]
        r = c.get("/api/workspaces", headers=H).json()
        assert r["default"] == "b" and [w["closed"] for w in r["workspaces"]] == [True, False]
        assert (tmp_path / "a" / "openlab.toml").is_file()                   # the research itself is untouched
        assert c.post("/api/workspaces/nope/tab", headers=H, json={"open": False}).status_code == 404
        assert c.post("/api/workspaces/a/tab", headers=H, json={"open": True}).json()["result"]["closed"] == []
    (tmp_path / show_home.FILE).write_text("not json")
    assert show_home.read(tmp_path) == {"closed": [], "machines": []}


def test_machines_are_declared_by_the_person_and_read_safely(tmp_path):
    app = server.create_app(str(tmp_path), token="tok")
    P = {**H, "x-openlab-reviewer": app.state.reviewer}
    with TestClient(app) as c:
        add = {"add": {"name": "gpu-box", "ssh": "gpu-box", "port": 47956}}
        assert c.post("/api/machines", headers=H, json=add).status_code == 403      # a person's edit
        assert c.post("/api/machines", headers=P, json=add).json()["ok"]
        for bad in ({"name": "x", "ssh": "-oProxyCommand=sh"}, {"name": "local", "ssh": "h"}, {"name": "gpu-box", "ssh": "h"},
                    {"name": "y", "ssh": "h; rm -rf /"}, {"name": "z", "ssh": "h", "port": 70000}):
            assert c.post("/api/machines", headers=P, json={"add": bad}).status_code == 400, bad
        ms = c.get("/api/machines", headers=H).json()["result"]
        assert [(m["name"], m["where"]) for m in ms] == [("local", "local"), ("gpu-box", "remote")]
        assert "queue" in ms[0] and "queue" not in ms[1]
        assert c.post("/api/machines", headers=P, json={"remove": "gpu-box"}).json()["result"] == []


def test_usage_is_parsed_from_the_probe():
    out = ("box\n8\n2.50 2.00 1.50 3/900 1234\nMemTotal:       16000000 kB\nMemAvailable:    6000000 kB\n"
           "/dev/sda1 1000000 400000 600000 40% /\nGPUS\nRTX 4090, 97, 21504, 24564\n")
    d = show_home.parse(out)
    assert d["host"] == "box" and d["cores"] == 8 and d["load"] == [2.5, 2.0, 1.5]
    assert d["mem_total_gb"] == 15.3 and d["mem_used_gb"] == 9.5 and d["disk_used_gb"] == 0.4
    assert d["gpus"] == [{"name": "RTX 4090", "util": 97.0, "mem_used_gb": 21.0, "mem_total_gb": 24.0}]


def test_a_person_changes_a_live_setting_never_a_fixed_one_and_comments_stay(tmp_path):
    ws(tmp_path, "a")
    ctx = ToolContext(str(tmp_path / "a"))
    assert not dispatch(ctx, "show_settings_set", {"key": "budget_usd", "value": 3})["ok"]       # not an agent
    env = dispatch(ctx, "show_settings_set", {"key": "budget_usd", "value": 3.5}, human=True)
    assert env["ok"], env
    assert "budget_usd = 3.5" in (tmp_path / "a" / "openlab.toml").read_text() and "# default budget" in \
        (tmp_path / "a" / "openlab.toml").read_text()
    row = next(r for r in env["result"]["settings"] if r["key"] == "budget_usd")
    assert row["value"] == 3.5 and not row["is_default"] and row["default"] == 2.0
    assert Store(tmp_path / "a").events(None, kinds=("setting",))[-1]["payload"] == {"key": "budget_usd", "old": 2.0,
                                                                                      "new": 3.5}
    assert dispatch(ctx, "show_settings_set", {"key": "compile_on_done", "value": False}, human=True)["ok"]
    assert "[show]\ncompile_on_done = false" in (tmp_path / "a" / "openlab.toml").read_text()
    for key, value, why in [("permission_mode", "plan", "fixed for this research"), ("budget_usd", -1, "not negative"),
                            ("model", "gpt-5", "is one of"), ("cap_usd", "lots", "a number"), ("deny", "x", "not a setting")]:
        env = dispatch(ctx, "show_settings_set", {"key": key, "value": value}, human=True)
        assert not env["ok"] and why in env["error"], (key, env)


def test_a_new_session_writes_its_chosen_settings(tmp_path):
    with TestClient(server.create_app(str(tmp_path), token="tok")) as c:
        bad = c.post("/api/workspaces", headers=H, json={"name": "x", "settings": {"permission_mode": "yolo"}})
        assert bad.status_code == 400 and not (tmp_path / "x").exists()
        env = c.post("/api/workspaces", headers=H, json={"name": "y", "settings": {"permission_mode": "plan",
                                                                                    "cap_usd": 50}}).json()
        assert env["ok"], env
    text = (tmp_path / "y" / "openlab.toml").read_text()
    assert 'permission_mode = "plan"' in text and "cap_usd = 50" in text and "# openlab workspace" in text


def test_a_page_learns_its_link_is_from_an_earlier_run(tmp_path):
    app = server.create_app(str(tmp_path), token="tok")
    with TestClient(app) as c:
        assert c.get("/api/person", headers={**H, "x-openlab-reviewer": app.state.reviewer}).json()["person"]
        assert not c.get("/api/person", headers={**H, "x-openlab-reviewer": "old-run"}).json()["person"]
        assert not c.get("/api/person", headers=H).json()["person"]
