"""Guarantee (round 4): a spawned agent or a proposal's lead delegates through openlab_agent_spawn, so every child
has a branch budget and a role prompt. Claude Code's own Agent/Task tool is disallowed on the spawned CLI's command
line and refused by the PreToolUse hook to a budgeted agent; a plain interactive session keeps its native sub-agents
(tracked as before). `[agents] allow_native_subagents = true` gives them back."""

import json
from pathlib import Path

import pytest

from agents_fake import final, install
from openlab.agents import hooks
from openlab.agents.spawn import spawn
from openlab.agents.store import Store

SID = "5e55i0n0-aaaa-bbbb-cccc-000000000009"


@pytest.fixture
def ws(tmp_path, monkeypatch):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "openlab.toml").write_text("[agents]\nbudget_usd = 1.0\n")
    install(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_ARGV", str(tmp_path / "argv.json"))
    monkeypatch.delenv("OPENLAB_WORKSPACE", raising=False)
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1]))   # the supervisor runs this openlab
    return root


def argv(tmp_path) -> list:
    return json.loads((tmp_path / "argv.json").read_text())["argv"]


def test_a_spawned_claude_runs_with_agent_and_task_disallowed(ws, tmp_path):
    a = spawn(ws, "execute the proposal", role="lead")
    final(Store(ws), a["id"])
    cmd = argv(tmp_path)
    assert cmd[cmd.index("--disallowedTools") + 1] == "Agent,Task"
    allowed = cmd[cmd.index("--allowedTools") + 1].split(",")
    assert "Agent" not in allowed and "Task" not in allowed and "mcp__openlab" in allowed
    assert "openlab_agent_spawn" in cmd[cmd.index("--append-system-prompt") + 1]


def test_allow_native_subagents_gives_them_back(ws, tmp_path):
    (ws / "openlab.toml").write_text("[agents]\nbudget_usd = 1.0\nallow_native_subagents = true\n")
    a = spawn(ws, "x")
    final(Store(ws), a["id"])
    cmd = argv(tmp_path)
    assert "--disallowedTools" not in cmd and "Agent" in cmd[cmd.index("--allowedTools") + 1].split(",")


def pre(ws, tool="Agent", sid=SID, **extra):
    return hooks.handle("PreToolUse", {"session_id": sid, "cwd": str(ws), "tool_name": tool, "tool_use_id": "tu1",
                                       "tool_input": {"subagent_type": "general-purpose", "description": "b1",
                                                      "prompt": "do branch b1"}, **extra})


def decision(out):
    return (out or {}).get("hookSpecificOutput", {}).get("permissionDecision")


@pytest.mark.parametrize("tool", ["Agent", "Task"])
def test_the_hook_refuses_native_subagents_to_a_lead_and_a_budgeted_spawned_agent(ws, tool):
    st = Store(ws)
    st.register("lead-abc123", "spawned", role="lead", cli="claude", session_id=SID, status="running",
                budget={"usd": 0.4})
    out = pre(ws, tool)
    assert decision(out) == "deny"
    assert "delegate with openlab_agent_spawn so the branch budget applies" in \
        out["hookSpecificOutput"]["permissionDecisionReason"]
    assert st.events("lead-abc123", kinds=("refusal",))[-1]["tool"] == tool
    st.register("experimenter-1", "spawned", role="experimenter", cli="claude", session_id="sid-2",
                status="running", budget={"usd": 0.2})
    assert decision(pre(ws, tool, sid="sid-2")) == "deny"


def test_a_lead_session_attached_by_role_is_refused_and_a_plain_session_is_not(ws):
    st = Store(ws)
    st.register("cc-leadsess", "claude-session", role="lead", cli="claude", session_id="sid-lead", status="running")
    assert decision(pre(ws, sid="sid-lead")) == "deny"
    out = pre(ws, sid="sid-plain")                                  # a plain interactive session: allowed, tracked
    assert decision(out) is None
    a = st.by_session("sid-plain")
    assert a["kind"] == "claude-session" and st.events(a["id"], kinds=("tool",))[-1]["tool"] == "Agent"


def test_the_config_allows_native_subagents_to_a_lead(ws):
    (ws / "openlab.toml").write_text("[agents]\nallow_native_subagents = true\n")
    Store(ws).register("lead-x", "spawned", role="lead", cli="claude", session_id=SID, status="running",
                       budget={"usd": 0.4})
    assert decision(pre(ws)) is None


def test_the_lead_brief_and_role_say_so():
    from openlab.agents.claude_setup import role_file
    from openlab.study.proposal import LEAD
    assert "Never delegate with Claude Code's own Agent or Task tool" in LEAD
    assert "never the Agent/Task tool" in role_file("lead")
