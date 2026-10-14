"""The person prompts a lead while it works: the lead reads it at its next tool call, answers with agent_reply and
carries on; the agents under a lead hear only from it; the leads pause and play together."""

import pytest

from openlab.agents import hooks as H
from openlab.agents.store import Store
from openlab.app import ToolContext, dispatch


@pytest.fixture
def ws(tmp_path, monkeypatch):
    (tmp_path / "openlab.toml").write_text("")
    st = Store(tmp_path)
    st.register("lead-1", "spawned", role="lead", status="running", study="s")
    st.register("lead-2", "spawned", role="lead", status="running", study="t")
    st.register("exp-1", "spawned", role="experimenter", parent="lead-1", status="running", study="s")
    st.register("gone", "spawned", role="lead", status="done", study="s")
    monkeypatch.delenv("OPENLAB_AGENT", raising=False)
    return ToolContext(str(tmp_path)), st


def person(ctx, name, args):
    return dispatch(ctx, name, args, human=True)


def test_a_prompt_reaches_the_lead_once_and_its_reply_joins_the_conversation(ws, monkeypatch):
    ctx, st = ws
    env = person(ctx, "agent_prompt", {"id": "lead-1", "text": "why is this taking so long?"})
    assert env["ok"], env
    seq = env["result"]["seq"]
    assert "next tool call" in env["result"]["when"]
    talk = person(ctx, "agent_prompts", {"id": "lead-1"})["result"]
    assert [(m["text"], m["read"]) for m in talk["messages"]] == [("why is this taking so long?", False)]
    text = H.delivery(st, st.get("lead-1"))
    assert f"[prompt {seq}]" in text and "from openlab show" in text and "openlab_agent_reply" in text
    assert "you were not stopped" in text and H.delivery(st, st.get("lead-1")) is None      # delivered once
    monkeypatch.setenv("OPENLAB_AGENT", "lead-1")
    r = dispatch(ctx, "agent_reply", {"text": "the third branch's runs are slow; done in ~10 min", "to": seq})
    assert r["ok"], r
    msgs = person(ctx, "agent_prompts", {"id": "lead-1"})["result"]["messages"]
    assert [m["from"] for m in msgs][-1] == "lead-1" and msgs[0]["read"] and msgs[-1]["to"] == seq


def test_only_a_lead_takes_prompts_and_only_from_the_person(ws, monkeypatch):
    ctx, st = ws
    env = person(ctx, "agent_prompt", {"id": "exp-1", "text": "hurry up"})
    assert not env["ok"] and "works under lead-1" in env["error"] and "lead-1" in env["try"]
    assert not st.untold("exp-1")
    ended = person(ctx, "agent_prompt", {"id": "gone", "text": "hello?"})
    assert not ended["ok"] and "has ended" in ended["error"]
    monkeypatch.setenv("OPENLAB_AGENT", "lead-2")
    assert not dispatch(ctx, "agent_prompt", {"id": "lead-1", "text": "stop now"})["ok"]   # an agent: not its tool
    assert not st.untold("lead-1")


def test_a_person_tells_a_lead_and_only_its_lead_tells_the_agents_under_it(ws, monkeypatch, capsys):
    from openlab.agents.cli import agent_main
    ctx, st = ws
    root = str(st.root)
    assert agent_main(["--root", root, "tell", "exp-1", "use the other split"]) == 1
    assert "works under lead-1" in capsys.readouterr().err and not st.untold("exp-1")
    assert agent_main(["--root", root, "tell", "lead-1", "use the other split"]) == 0
    monkeypatch.setenv("OPENLAB_AGENT", "lead-2")                    # another lead: not exp-1's
    monkeypatch.setattr("openlab.agents.identity.verified", lambda a, t: a)
    assert agent_main(["--root", root, "tell", "exp-1", "drop it"]) == 1
    monkeypatch.setenv("OPENLAB_AGENT", "lead-1")                    # its own lead
    assert agent_main(["--root", root, "tell", "exp-1", "drop it"]) == 0
    assert [e["payload"]["text"] for e in st.untold("exp-1")] == ["drop it"]


def test_a_waiting_lead_wakes_for_the_persons_message(ws, monkeypatch):
    ctx, st = ws
    person(ctx, "agent_prompt", {"id": "lead-1", "text": "status?"})
    monkeypatch.setenv("OPENLAB_AGENT", "lead-1")
    env = dispatch(ctx, "agent_wait", {"ids": ["exp-1"], "timeout": 30})
    assert env["ok"] and env["result"]["prompted"] and not env["result"]["done"]
    assert "openlab_agent_reply" in env["result"]["next"]


def test_the_mcp_reply_carries_the_message_for_an_agent_without_hooks(ws, monkeypatch):
    from openlab import mcp_server
    ctx, st = ws
    person(ctx, "agent_prompt", {"id": "lead-2", "text": "finish up quickly"})
    monkeypatch.setenv("OPENLAB_WORKSPACE", str(st.root))
    monkeypatch.setenv("OPENLAB_AGENT", "lead-2")
    monkeypatch.setattr("openlab.agents.identity.verified", lambda a, t: a)
    text = mcp_server._told()
    assert "finish up quickly" in text and mcp_server._told() is None


def test_the_leads_pause_and_play_together_and_the_agents_under_them_are_held(ws):
    ctx, st = ws
    env = person(ctx, "agent_control_leads", {"action": "pause", "reason": "lunch"})
    assert env["ok"], env
    assert sorted(a["id"] for a in env["result"]["leads"]) == ["lead-1", "lead-2"]
    assert st.get("lead-1")["control"] == "pause" and st.get("exp-1")["control"] == ""
    assert st.blocked("exp-1")[0] == "lead-1"                       # held with its lead
    held = person(ctx, "agent_prompt", {"id": "lead-1", "text": "still there?"})
    assert held["ok"] and "once resumed" in held["result"]["when"]
    again = person(ctx, "agent_control_leads", {"action": "resume", "reason": "back"})
    assert not again["ok"] and "passphrase" in again["error"]       # a resume is signed


def test_a_persons_request_is_never_held_by_a_paused_agent_in_the_servers_environment(ws, monkeypatch):
    ctx, st = ws
    assert person(ctx, "agent_control", {"id": "lead-1", "action": "pause", "reason": "lunch"})["ok"]
    monkeypatch.setenv("OPENLAB_AGENT", "lead-1")                    # e.g. openlab show started from its shell
    monkeypatch.setattr("openlab.agents.identity.verified", lambda a, t: a)
    assert not dispatch(ctx, "agent_note", {"text": "x"})["ok"]       # the paused agent itself: held
    assert person(ctx, "agent_prompts", {"id": "lead-1"})["ok"]      # the person: never held


def test_a_side_question_is_answered_now_by_a_fork_of_the_leads_session(ws, tmp_path, monkeypatch):
    import json
    from agents_fake import install
    from openlab.agents import btw
    ctx, st = ws
    install(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_ARGV", str(tmp_path / "argv.json"))
    st.update("lead-1", cli="claude", session_id="sess-1", model="claude-opus-4-5", cwd=str(tmp_path))
    env = person(ctx, "agent_btw", {"id": "lead-1", "text": "why is this taking so long?"})
    assert env["ok"], env
    assert env["result"]["answer"] == "side answer to: why is this taking so long?" and env["result"]["usd"] == 0.004
    argv = json.loads((tmp_path / "argv.json").read_text())["argv"]
    assert argv[argv.index("--resume") + 1] == "sess-1" and "--fork-session" in argv and "--no-session-persistence" in argv
    assert argv[argv.index("--tools") + 1] == "" and argv[argv.index("--setting-sources") + 1] == "user"
    assert argv[argv.index("--model") + 1] == "claude-opus-4-5" and argv[argv.index("-p") + 1].startswith(btw.FRAME)
    assert not st.untold("lead-1")                                  # the lead itself is not told
    assert st.get("lead-1")["spent"]["btw_usd"] == 0.004 and not st.get("lead-1")["spent"].get("usd")
    msgs = person(ctx, "agent_prompts", {"id": "lead-1"})["result"]["messages"]
    assert [(m["from"] == "lead-1", m["btw"]) for m in msgs] == [(False, True), (True, True)]
    assert person(ctx, "agent_btw", {"id": "gone", "text": "why did you stop?"})["ok"] is False   # no session: refused
    st.update("gone", cli="claude", session_id="sess-2", cwd=str(tmp_path))
    assert person(ctx, "agent_btw", {"id": "gone", "text": "why did you stop?"})["ok"]           # an ended lead: asked
    assert not person(ctx, "agent_btw", {"id": "exp-1", "text": "hi"})["ok"]                   # under a lead: never
    monkeypatch.setenv("OPENLAB_AGENT", "lead-2")
    assert not dispatch(ctx, "agent_btw", {"id": "lead-1", "text": "hi"})["ok"]               # an agent: not its tool


def test_a_side_questions_fork_never_joins_the_tree():
    import os
    from openlab.agents import hooks as H
    os.environ["OPENLAB_BTW"] = "1"
    try:
        assert H.handle("SessionStart", {"session_id": "fork", "cwd": "/nowhere"}) is None
        assert H.handle("PreToolUse", {"tool_name": "Bash", "tool_input": {"command": "openlab study holdout x"}})
    finally:
        del os.environ["OPENLAB_BTW"]
