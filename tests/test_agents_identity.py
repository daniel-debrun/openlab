"""Who is acting: the precedence of identity.current(), every record node naming its agent, one agent node per
agent per study, and the MCP server attributing a call to the session or sub-agent the hook saw make it."""

import asyncio
import sys

import pytest

from openlab.agents import identity
from openlab.agents.identity import acting_as, current
from openlab.agents.store import Store
from openlab.study.record import Record


def test_precedence(monkeypatch):
    for k in ("OPENLAB_AGENT", "OPENLAB_AGENT_TOKEN", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CODEX_SANDBOX",
              "CODEX_SANDBOX_NETWORK_DISABLED"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    assert current() == "unattributed"
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    assert current().startswith("person:")                         # a label: such writes stay actor agent
    monkeypatch.setenv("OPENLAB_AGENT", "cc-1234abcd")
    assert current() == "cc-1234abcd"
    with acting_as("cc-1234abcd.ab12cd"):
        assert current() == "cc-1234abcd.ab12cd"
        with acting_as(None):
            assert current() == "cc-1234abcd.ab12cd"
    assert current() == "cc-1234abcd"


def test_record_add_fills_session_and_writes_one_agent_node_per_study(tmp_path, monkeypatch):
    ws = tmp_path
    (ws / "openlab.toml").write_text("")
    st = Store(ws)
    st.register("lead-1", "claude-session", role="lead", cli="claude")
    st.register("lead-1.abc123", "claude-subagent", role="openlab-screener", parent="lead-1", cli="claude")
    rec = Record(ws / "demo")
    monkeypatch.setenv("OPENLAB_AGENT", "lead-1")
    q = rec.add("question", {"text": "does x predict y?"})
    assert q.session == "lead-1"
    with acting_as("lead-1.abc123"):
        q2 = rec.add("question", {"text": "which sources measure x?"}, links=[("part_of", q.id)])
        rec.add("question", {"text": "and y?"}, links=[("part_of", q.id)])
    assert q2.session == "lead-1.abc123"
    agents = rec.find("agent")
    assert [a.payload["id"] for a in agents] == ["lead-1", "lead-1.abc123"]
    sub = agents[1].payload
    assert sub["role"] == "openlab-screener" and sub["parent"] == "lead-1" and agents[1].actor == "openlab"
    assert rec.verify()["ok"]
    from openlab.study.errors import StudyError
    with pytest.raises(StudyError, match="already"):
        rec.add("agent", dict(sub), actor="openlab")


def test_why_names_the_sub_agent(tmp_path, monkeypatch, capsys):
    from openlab.study import cli
    from openlab import study
    ws = tmp_path
    (ws / "openlab.toml").write_text("")
    Store(ws).register("cc-1.abc123", "claude-subagent", role="openlab-experimenter", parent="cc-1")
    s = study.new(ws / "demo", "static")
    with acting_as("cc-1.abc123"):
        q = s.record.add("question", {"text": "q?"})
        h = s.record.add("hypothesis", {"statement": "x helps", "prediction": "lower error", "mechanism": "m",
                                        "falsified_by": "no change", "tags": ["x"]}, links=[("part_of", q.id)])
    assert cli.main(["why", str(ws / "demo"), h.id]) == 0
    out = capsys.readouterr().out
    assert "by cc-1.abc123 (openlab-experimenter)" in out and "agent  " not in out


def test_mcp_call_claims_the_hooks_note(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENLAB_AGENT", raising=False)
    monkeypatch.setenv("OPENLAB_WORKSPACE", str(tmp_path))
    st = Store(tmp_path)
    st.pend("cc-1.abc123", "openlab_study_run", {"folder": "demo", "note": "x only"})
    assert identity.for_mcp_call("openlab_study_run", {"folder": "demo", "note": "x only"}) == "cc-1.abc123"
    assert identity.for_mcp_call("openlab_study_run", {"folder": "demo", "note": "x only"}) == "unattributed"
    monkeypatch.setenv("OPENLAB_AGENT", "spawned-1")                      # a spawned agent's server
    assert identity.for_mcp_call("openlab_study_run", {}) == "spawned-1"
    st.register("spawned-1", "spawned")
    st.register("spawned-1.abc", "claude-subagent", parent="spawned-1")
    st.pend("cc-other", "openlab_study_status", {"folder": "demo"})       # another session's call: not ours
    st.pend("spawned-1.abc", "openlab_study_status", {"folder": "demo"})  # its sub-agent shares its MCP server
    assert identity.for_mcp_call("openlab_study_status", {"folder": "demo"}) == "spawned-1.abc"
    assert identity.for_mcp_call("openlab_study_status", {"folder": "demo"}) == "spawned-1"


def test_attribute_mcp_wraps_the_tool_manager(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENLAB_AGENT", raising=False)
    monkeypatch.setenv("OPENLAB_WORKSPACE", str(tmp_path))
    seen = []

    class Manager:
        async def call_tool(self, name, arguments, context=None, convert_result=False):
            seen.append(current())
            return "ok"

    class Server:
        _tool_manager = Manager()
    srv = Server()
    assert identity.attribute_mcp(srv) and not identity.attribute_mcp(srv)
    Store(tmp_path).pend("cc-9", "openlab_agent_note", {"text": "hi"})
    assert asyncio.run(srv._tool_manager.call_tool("openlab_agent_note", {"text": "hi"}, None, convert_result=True)) == "ok"
    assert seen == ["cc-9"]


def test_real_mcp_server_attributes_and_refuses_when_paused(tmp_path, monkeypatch):
    """The openlab MCP server end to end (in process): a hook's note attributes the call, a pause refuses it."""
    pytest.importorskip("mcp")
    monkeypatch.delenv("OPENLAB_AGENT", raising=False)
    monkeypatch.setenv("OPENLAB_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("OPENLAB_MCP_INPROCESS", "1")
    (tmp_path / "openlab.toml").write_text("")
    import openlab.app.client as client
    monkeypatch.setattr(client, "_LOCAL", None)
    from openlab import mcp_server
    st = Store(tmp_path)
    st.register("cc-77", "claude-session")
    st.pend("cc-77", "openlab_agent_note", {"text": "progress"})

    async def go(args):
        return await mcp_server.server.call_tool("openlab_agent_note", args)
    asyncio.run(go({"text": "progress"}))
    note = st.events("cc-77", kinds=("note",))
    assert note and note[-1]["summary"] == "progress"
    st.control("cc-77", "pause", "wait for the person", "terminal:sam")
    st.pend("cc-77", "openlab_agent_spawn", {"prompt": "x"})
    out = asyncio.run(mcp_server.server.call_tool("openlab_agent_spawn", {"prompt": "x"}))
    assert "paused by person" in str(out) and "wait for the person" in str(out)


# --- the person's own terminal (a person's own writes once read `unattributed`) -------------------------------------
ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
AGENT_VARS = ("OPENLAB_AGENT", "OPENLAB_AGENT_TOKEN", "OPENLAB_WORKSPACE", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT",
              "CODEX_SANDBOX", "CODEX_SANDBOX_NETWORK_DISABLED")
WRITE = ("import sys; from openlab.study.record import Record; "
         "print('SESSION', Record(sys.argv[1]).add('question', {'text': 'q'}).session)")


def _in_pty(folder, home, stdin_null: bool = False) -> str:
    """Write one node from a child whose controlling terminal is a fresh pty (a person's terminal, no agent
    variables); return what it printed."""
    import os
    import pty
    import select
    env = {k: v for k, v in os.environ.items() if k not in AGENT_VARS}
    env.update({"HOME": str(home), "PYTHONPATH": str(ROOT)})
    pid, fd = pty.fork()
    if pid == 0:                                                    # the child: never returns into pytest
        try:
            if stdin_null:
                os.dup2(os.open(os.devnull, os.O_RDONLY), 0)        # `openlab ... < /dev/null` typed at the terminal
            os.execve(sys.executable, [sys.executable, "-c", WRITE, str(folder)], env)
        finally:
            os._exit(127)
    out = b""
    while True:
        r, _, _ = select.select([fd], [], [], 60)
        if not r:
            break
        try:
            chunk = os.read(fd, 4096)
        except OSError:
            break
        if not chunk:
            break
        out += chunk
    os.waitpid(pid, 0)
    return out.decode(errors="replace")


@pytest.mark.parametrize("stdin_null", [False, True])
def test_a_write_from_the_persons_terminal_is_labelled_person(tmp_path, person_key, stdin_null):
    from openlab.agents import person
    home = person.config_dir().parent.parent                        # the fixture's home: the key's handle is sam
    out = _in_pty(tmp_path / "s", home, stdin_null)
    assert "SESSION person:sam" in out, out


def test_a_write_with_no_terminal_and_no_agent_is_unattributed(tmp_path):
    import os
    import subprocess
    env = {k: v for k, v in os.environ.items() if k not in AGENT_VARS}
    env["PYTHONPATH"] = str(ROOT)
    p = subprocess.run([sys.executable, "-c", WRITE, str(tmp_path / "s")], env=env, capture_output=True, text=True,
                       stdin=subprocess.DEVNULL, start_new_session=True, timeout=60)
    assert "SESSION unattributed" in p.stdout, p.stderr
