"""openlab show's conversation: an agent's transcript as the page reads it, up to the call that marked a step done."""

import json

from openlab.agents.store import Store
from openlab.app import ToolContext, dispatch


def _line(kind, blocks, at, parent=None):
    return json.dumps({"type": kind, "message": {"content": blocks}, "timestamp": at, "parent_tool_use_id": parent},
                      separators=(",", ":"))


def _call(i, name, inp, at):
    return _line("assistant", [{"type": "tool_use", "id": f"t{i}", "name": name, "input": inp}], at)


def _res(i, text, at, error=False):
    return _line("user", [{"type": "tool_result", "tool_use_id": f"t{i}", "content": [{"type": "text", "text": text}],
                           "is_error": error}], at)


def _ws(tmp_path, lines):
    st = Store(tmp_path)
    st.register("lead-1", "spawned", role="lead", status="done")
    d = st.agent_dir("lead-1")
    (d / "prompt.txt").write_text("Run trial 2.")
    (d / "transcript.jsonl").write_text("\n".join(
        ['{"type":"system","subtype":"init"}', *lines,
         json.dumps({"type": "result", "subtype": "success", "result": "All done."}, separators=(",", ":"))]) + "\n")
    return ToolContext(str(tmp_path))


PLAN = "mcp__openlab__openlab_agent_plan"
NODE = "mcp__openlab__openlab_agent_plan_node"
LINES = [
    _line("assistant", [{"type": "thinking", "thinking": ""}, {"type": "text", "text": "Reading the rules."}], "T1"),
    _call(1, PLAN, {"nodes": [{"id": "rules", "title": "Rules", "state": "active"}]}, "T2"), _res(1, "v1", "T2"),
    _call(2, "Bash", {"command": "ls"}, "T3"), _res(2, "x" * 10_000, "T3"),
    _line("assistant", [{"type": "text", "text": "sub"}], "T3", parent="t2"),
    _call(3, NODE, {"id": "rules", "state": "done"}, "T4"), _res(3, "refused", "T4", error=True),
    _call(4, NODE, {"id": "rules", "state": "done", "note": "ok"}, "T5"), _res(4, "ok", "T5"),
    _line("assistant", [{"type": "text", "text": "Next: the run."}], "T6"),
    _call(5, "mcp__openlab__openlab_tool", {"name": "agent_plan_node", "args": {"id": "run", "state": "done"}}, "T7"),
    _res(5, "ok", "T7"),
]


def test_up_to_the_call_that_marked_the_step_done(tmp_path):
    env = dispatch(_ws(tmp_path, LINES), "show_conversation", {"agent": "lead-1", "node": "rules"}, human=True)
    r = env["result"]
    assert r["state"] == "done" and r["cut"] == "T5"
    kinds = [e["kind"] for e in r["entries"]]
    assert kinds == ["prompt", "say", "call", "call", "call", "call"]          # the refused done is not the cut
    last = r["entries"][-1]
    assert last["cut"] and last["result"] == "ok" and '"note": "ok"' in last["input"]
    assert r["entries"][4]["error"] is True
    bash = r["entries"][3]
    assert bash["sub"] == 1 and len(bash["result"]) < 3100                      # a sub-agent's turn counted, not shown
    assert r["after"] == 3                                                     # the say, the run's call, the end


def test_a_step_through_openlab_tool_and_the_whole_conversation(tmp_path):
    ctx = _ws(tmp_path, LINES)
    r = dispatch(ctx, "show_conversation", {"agent": "lead-1", "node": "run"}, human=True)["result"]
    assert r["cut"] == "T7" and r["entries"][-1]["tool"] == "openlab_tool"
    r = dispatch(ctx, "show_conversation", {"agent": "lead-1"}, human=True)["result"]
    assert r["cut"] is None and r["entries"][-1] == {"kind": "end", "at": None, "text": "All done."}
    r = dispatch(ctx, "show_conversation", {"agent": "lead-1", "node": "never"}, human=True)["result"]
    assert r["cut"] is None and r["after"] == 0 and r["entries"][-1]["kind"] == "end"


def test_an_unknown_agent_and_an_agent_cannot_read_it(tmp_path):
    ctx = _ws(tmp_path, LINES)
    env = dispatch(ctx, "show_conversation", {"agent": "lead-9"}, human=True)
    assert not env["ok"] and "no agent" in env["error"]
    assert not (tmp_path / ".openlab" / "agents" / "lead-9").exists()
    assert not dispatch(ctx, "show_conversation", {"agent": "lead-1"})["ok"]                  # a person's read
