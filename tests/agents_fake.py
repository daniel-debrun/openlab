"""Fake agent CLIs for the control-plane tests: a `claude` that prints Claude Code stream-json and a `codex` that
prints `codex exec --json` JSONL, chosen by FAKE_MODE. Nothing here calls a model."""

import stat
import sys
import time
from pathlib import Path

CLAUDE = r'''#!{python}
import json, os, sys, time
args = sys.argv[1:]
if os.environ.get("FAKE_ARGV"):
    with open(os.environ["FAKE_ARGV"], "w") as f:
        json.dump({"argv": args, "agent": os.environ.get("OPENLAB_AGENT"), "ws": os.environ.get("OPENLAB_WORKSPACE"),
                   "cwd": os.getcwd(), "allocation": os.environ.get("OPENLAB_ALLOCATION"),
                   "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"), "capacity": os.environ.get("OPENLAB_BOX_CAPACITY"),
                   "affinity": sorted(os.sched_getaffinity(0)),
                   "bg_ceiling": os.environ.get("CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS")}, f)
mode = os.environ.get("FAKE_MODE", "normal")
if "--fork-session" in args:                       # a side question (openlab.agents.btw): one JSON answer
    q = args[args.index("-p") + 1]
    print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "total_cost_usd": 0.004,
                      "session_id": "fork", "result": "side answer to: " + q.split("\n\n", 1)[-1],
                      "env_btw": os.environ.get("OPENLAB_BTW")}), flush=True)
    sys.exit(0)
sid = args[args.index("--session-id") + 1] if "--session-id" in args else "none"
def out(o):
    print(json.dumps(o), flush=True)
def usage(i, o):
    return {"input_tokens": i, "output_tokens": o, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
out({"type": "system", "subtype": "init", "session_id": sid, "model": "claude-haiku-4-5", "cwd": os.getcwd(),
     "tools": ["Bash"], "mcp_servers": [{"name": "openlab", "status": "connected"}]})
if mode == "fail":
    sys.exit(3)
if mode == "normal":
    out({"type": "assistant", "parent_tool_use_id": None, "session_id": sid, "message": {"id": "m1", "model": "claude-haiku-4-5",
         "content": [{"type": "text", "text": "Looking at the study first."},
                     {"type": "tool_use", "id": "tu1", "name": "mcp__openlab__openlab_study_status", "input": {"folder": "demo"}}],
         "usage": usage(1000, 50)}})
    out({"type": "user", "parent_tool_use_id": None, "message": {"content": [{"type": "tool_result", "tool_use_id": "tu1",
         "content": [{"type": "text", "text": json.dumps({"ok": False, "tool": "study_status", "error": "demo is not a study",
                                                        "meaning": "no study.toml", "try": "study_new"})}]}]}})
    out({"type": "assistant", "parent_tool_use_id": None, "message": {"id": "m2", "model": "claude-haiku-4-5",
         "content": [{"type": "tool_use", "id": "toolu_task1", "name": "Agent",
                      "input": {"subagent_type": "openlab-screener", "description": "screen the sources", "prompt": "go"}}],
         "usage": usage(1200, 40)}})
    out({"type": "assistant", "parent_tool_use_id": "toolu_task1", "message": {"id": "s1", "model": "claude-haiku-4-5",
         "content": [{"type": "text", "text": "Screening three sources."},
                     {"type": "tool_use", "id": "tu2", "name": "mcp__openlab__openlab_study_screen", "input": {"folder": "lit"}}],
         "usage": usage(800, 30)}})
    out({"type": "user", "parent_tool_use_id": None, "message": {"content": [{"type": "tool_result",
         "tool_use_id": "toolu_task1", "content": "screened 3 sources"}]}})
    out({"type": "assistant", "parent_tool_use_id": None, "message": {"id": "m3", "model": "claude-haiku-4-5",
         "content": [{"type": "text", "text": "Done: recorded nothing yet."}], "usage": usage(1500, 20)}})
    out({"type": "result", "subtype": "success", "is_error": False, "num_turns": 3,
         "total_cost_usd": 0.0246 if "--resume" in args else 0.0123,          # a resumed session counts it all
         "session_id": sid, "result": "Done: recorded nothing yet.", "usage": usage(3700, 110)})
elif mode == "slow":
    for i in range(int(os.environ.get("FAKE_N", "300"))):
        out({"type": "assistant", "parent_tool_use_id": None, "message": {"id": f"t{i}", "model": "claude-haiku-4-5",
             "content": [{"type": "text", "text": f"tick {i}"}], "usage": usage(10, 1)}})
        time.sleep(0.1)
    out({"type": "result", "subtype": "success", "is_error": False, "num_turns": 1, "total_cost_usd": 0.001,
         "session_id": sid, "result": "slow done", "usage": usage(10, 1)})
elif mode == "spend":
    out({"type": "assistant", "parent_tool_use_id": None, "message": {"id": "big", "model": "claude-sonnet-4-5",
         "content": [{"type": "text", "text": "thinking a lot"}], "usage": usage(10, 1000000)}})
    time.sleep(60)
'''

CODEX = r'''#!{python}
import json, os, sys
def out(o):
    print(json.dumps(o), flush=True)
out({"type": "thread.started", "thread_id": "th_1"})
out({"type": "turn.started"})
out({"type": "item.completed", "item": {"id": "i1", "type": "agent_message", "text": "Reading the study."}})
out({"type": "item.completed", "item": {"id": "i2", "type": "mcp_tool_call", "server": "openlab", "tool": "openlab_study_run",
     "arguments": {"folder": "demo"}, "status": "completed",
     "result": {"content": [{"type": "text", "text": json.dumps({"ok": False, "error": "the trial budget is spent", "meaning": "20 of 20"})}]}}})
out({"type": "item.completed", "item": {"id": "i3", "type": "command_execution", "command": "ls", "aggregated_output": "a b", "exit_code": 0}})
out({"type": "turn.completed", "usage": {"input_tokens": 2000, "cached_input_tokens": 500, "output_tokens": 300}})
out({"type": "item.completed", "item": {"id": "i4", "type": "agent_message", "text": "Finished."}})
'''


def install(tmp: Path, monkeypatch) -> dict:
    """Write the fakes and point OPENLAB_CLAUDE_BIN / OPENLAB_CODEX_BIN at them."""
    bins = {}
    for name, text in (("claude", CLAUDE), ("codex", CODEX)):
        p = tmp / f"fake-{name}"
        p.write_text(text.replace("{python}", sys.executable))
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
        bins[name] = p
    monkeypatch.setenv("OPENLAB_CLAUDE_BIN", str(bins["claude"]))
    monkeypatch.setenv("OPENLAB_CODEX_BIN", str(bins["codex"]))
    monkeypatch.delenv("OPENLAB_AGENT", raising=False)
    return bins


def wait(store, agent: str, cond, timeout: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        a = store.get(agent)
        if cond(a):
            return a
        time.sleep(0.1)
    raise AssertionError(f"timed out: {store.get(agent)}\nevents: "
                         f"{[(e['kind'], e['summary']) for e in store.events(agent, n=30)]}")


def final(store, agent: str, timeout: float = 20.0) -> dict:
    from openlab.agents.store import FINAL
    return wait(store, agent, lambda a: a["status"] in FINAL, timeout)
