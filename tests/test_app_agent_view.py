"""The web app's Agents view tools: agent_tree (the watch snapshot as JSON, filtered by study), inbox_list, and the
person's acts agent_control and inbox_decide, which need the reviewer (dispatch human=True)."""

import pytest

from openlab.app import ToolContext, dispatch
from openlab.agents.store import Store


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "openlab.toml").write_text("[agents]\nstuck_minutes = 5\n")
    st = Store(tmp_path)
    st.register("lead", "claude-session", role="session", question="does x predict y?", budget={"usd": 2.0},
                spent={"usd": 0.4, "tokens_in": 1000, "tokens_out": 200})
    st.register("lead.lit", "claude-subagent", role="screener", parent="lead", study="lit")
    st.register("lead.exp", "claude-subagent", role="experimenter", parent="lead", study="demo")
    st.register("lead.exp.sub", "claude-subagent", role="helper", parent="lead.exp", study="demo")
    d = st.agent_dir("lead")
    (d / "prompt.txt").write_text("find out whether x predicts y")
    st.event("lead.exp", "tool", tool="mcp__openlab__openlab_study_run", summary="study_run demo")
    st.request("lead.exp", "budget", "need one more dollar", study="demo", args={"usd": 1.0}, cost_note="$1")
    return ToolContext(str(tmp_path)), st


def test_agent_tree_is_the_watch_snapshot_in_tree_order(ws):
    ctx, _ = ws
    env = dispatch(ctx, "agent_tree", {"id": "lead"})
    assert env["ok"], env
    res = env["result"]
    assert [a["id"] for a in res["agents"]] == ["lead", "lead.lit", "lead.exp", "lead.exp.sub"]
    assert [a["depth"] for a in res["agents"]] == [0, 1, 1, 2]
    exp = next(a for a in res["agents"] if a["id"] == "lead.exp")
    assert exp["state"] == "waiting" and "WAITING" in exp["flags"] and exp["study"] == "demo"
    assert exp["last"]["tool"] == "mcp__openlab__openlab_study_run"
    assert res["pending"] == 1 and res["detail"]["prompt"].startswith("find out")
    assert res["detail"]["children"] == ["lead.lit", "lead.exp"]


def test_agent_tree_by_study_keeps_a_tree(ws):
    ctx, _ = ws
    res = dispatch(ctx, "agent_tree", {"study": "demo"})["result"]
    assert [(a["id"], a["depth"]) for a in res["agents"]] == [("lead.exp", 0), ("lead.exp.sub", 1)]
    assert dispatch(ctx, "inbox_list", {"study": "lit"})["result"]["items"] == []


def test_control_and_decide_are_a_persons_acts(ws, person_key):
    ctx, st = ws
    for name, args in (("agent_control", {"id": "lead.exp", "action": "pause", "reason": "look"}),
                       ("inbox_decide", {"id": 1, "approve": True, "reason": "fine"})):
        env = dispatch(ctx, name, args)
        assert not env["ok"] and env["error"] == "a reviewer's act"
    env = dispatch(ctx, "agent_control", {"id": "lead.exp", "action": "pause", "reason": "look"}, human=True)
    assert env["ok"] and st.get("lead.exp")["control"] == "pause"
    tree = dispatch(ctx, "agent_tree", {})["result"]["agents"]
    assert next(a for a in tree if a["id"] == "lead.exp")["state"] == "paused"
    assert not dispatch(ctx, "agent_control", {"id": "lead.exp", "action": "nap", "reason": "x"}, human=True)["ok"]

    items = dispatch(ctx, "inbox_list", {})["result"]["items"]
    assert [(i["id"], i["agent"], i["kind"], i["cost_note"]) for i in items] == [(1, "lead.exp", "budget", "$1")]
    env = dispatch(ctx, "inbox_decide", {"id": 1, "approve": True, "reason": "fine"}, human=True)
    assert not env["ok"] and env["error"] == "passphrase required"          # the reviewer secret alone signs nothing
    resume = dispatch(ctx, "agent_control", {"id": "lead.exp", "action": "resume", "reason": "ok"}, human=True)
    assert not resume["ok"] and resume["error"] == "passphrase required"
    with person_key.act():                                                 # the server opened the key
        env = dispatch(ctx, "inbox_decide", {"id": 1, "approve": True, "reason": "fine"}, human=True)
        assert env["ok"] and env["result"]["status"] == "approved"
        assert st.get("lead.exp")["budget"]["usd"] == 1.0
        again = dispatch(ctx, "inbox_decide", {"id": 1, "approve": False, "reason": "no"}, human=True)
        assert not again["ok"] and "already approved" in again["error"]
        assert dispatch(ctx, "agent_control", {"id": "lead.exp", "action": "resume", "reason": "ok"}, human=True)["ok"]
    assert dispatch(ctx, "inbox_list", {"all": True})["result"]["items"][0]["decided_by"] == "person:sam"

    item = st.request("lead.exp", "screen_conflict", "agents disagree on src:1", study="demo",
                      args={"source": "src:1", "options": [{"agent": "a", "decision": "include", "criterion": "i1"},
                                                           {"agent": "b", "decision": "exclude", "criterion": "e1"}]})
    with person_key.act():
        env = dispatch(ctx, "inbox_decide", {"id": item, "approve": True, "reason": "a is right"}, human=True)
    assert not env["ok"] and "pick" in env["error"]


def test_agent_activity_reads_the_control_store(ws, tmp_path):
    ctx, st = ws
    st.event("lead.lit", "refusal", tool="mcp__openlab__openlab_study_holdout", summary="a reviewer's act")
    res = dispatch(ctx, "agent_activity", {"last": 5})["result"]
    first = res["events"][0]
    assert (first["agent"], first["ok"], first["error"]) == ("lead.lit", False, "a reviewer's act")
    assert any(e["agent"] == "lead.exp" and e["ok"] for e in res["events"])
    assert res["waiting"] == 1 and res["refused"] == 1
    assert not (tmp_path / ".data" / "laws.jsonl").exists()
