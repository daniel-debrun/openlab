"""Spawned agents under a supervisor, with fake CLIs: the stream becomes events and spend, the budget is enforced,
pause is SIGSTOP, stop kills the group, a dead supervisor is detected as lost, and the tree has limits."""

import json
import os
import re
import signal
import time

import pytest

from agents_fake import final, install, wait
from openlab.agents.claude_setup import ROLES
from openlab.agents.spawn import AgentError, relaunch, spawn
from openlab.agents.store import Store


@pytest.fixture
def ws(tmp_path, monkeypatch):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "openlab.toml").write_text("[agents]\nbudget_usd = 1.0\n")
    install(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_ARGV", str(tmp_path / "argv.json"))
    yield root
    for a in Store(root).agents():                  # a failed test never leaves a (paused) fake agent behind
        for pid, group in ((a["pgid"], True), (a["supervisor"], False)):
            for sig in (signal.SIGCONT, signal.SIGKILL):
                try:
                    os.killpg(pid, sig) if group else os.kill(pid, sig)
                except (ProcessLookupError, PermissionError, TypeError, OSError):
                    pass


def test_an_ended_lead_is_relaunched_in_its_own_session_and_its_spend_carries_on(ws, tmp_path):
    a = spawn(ws, "look at the demo study", role="lead")
    st = Store(ws)
    final(st, a["id"])
    first = json.loads((tmp_path / "argv.json").read_text())
    assert first["bg_ceiling"] == "0"                   # claude -p waits on its background tasks, not 10 min
    st.register("busy-1", "spawned", cli="claude", session_id="s-1", status="running")
    with pytest.raises(AgentError, match="busy-1 is running"):
        relaunch(st, ws, "busy-1", "go on", "person:test")
    relaunch(st, ws, a["id"], "go on: the replicator still trains", "person:test")
    done = wait(st, a["id"], lambda x: x["status"] == "done" and x["spent"]["usd"] > 0.02)
    argv = json.loads((tmp_path / "argv.json").read_text())["argv"]
    assert argv[argv.index("--resume") + 1] == a["session_id"] and "--session-id" not in argv
    assert argv[argv.index("-p") + 1] == "go on: the replicator still trains"
    assert done["spent"]["usd"] == pytest.approx(2 * 0.0123)
    assert "relaunch" in [e["kind"] for e in st.events(a["id"], n=100)]


def test_spawn_parses_the_stream_into_events_and_spend(ws, tmp_path):
    a = spawn(ws, "look at the demo study", role="lead", question="does x predict y?")
    st = Store(ws)
    done = final(st, a["id"])
    assert done["status"] == "done" and done["exit_code"] == 0
    s = done["spent"]
    assert s["usd"] == pytest.approx(0.0123) and s["usd_estimated"] is False
    assert s["turns"] == 3 and s["tool_calls"] == 2 and s["tokens_in"] >= 3700
    kinds = [e["kind"] for e in st.events(a["id"], n=100)]
    for k in ("spawn", "launch", "start", "say", "tool", "refusal", "result", "end"):
        assert k in kinds, k
    refusal = next(e for e in st.events(a["id"], n=100) if e["kind"] == "refusal")
    assert "demo is not a study" in refusal["summary"]
    # the sub-agent message (parent_tool_use_id) made a child in the tree, with its own tool call and tokens
    kids = st.children(a["id"])
    assert len(kids) == 1 and kids[0]["kind"] == "claude-subagent" and kids[0]["role"] == "openlab-screener"
    assert kids[0]["status"] == "done" and kids[0]["spent"]["tool_calls"] == 1 and kids[0]["spent"]["tokens_in"] == 800
    assert any(e["tool"] == "mcp__openlab__openlab_study_screen" for e in st.events(kids[0]["id"]))
    # the CLI got the attribution and the limits
    launched = json.loads((tmp_path / "argv.json").read_text())
    argv = launched["argv"]
    assert launched["agent"] == a["id"] and launched["ws"] == str(ws)
    assert argv[argv.index("--session-id") + 1] == a["session_id"]
    assert "--strict-mcp-config" in argv
    assert "--max-budget-usd" not in argv and "--max-turns" not in argv   # openlab pauses at a limit, not the CLI
    mcp = json.loads(open(argv[argv.index("--mcp-config") + 1]).read())
    assert mcp["mcpServers"]["openlab"]["env"]["OPENLAB_AGENT"] == a["id"]
    brief = argv[argv.index("--append-system-prompt") + 1]
    role = ROLES["lead"][1]                                             # a role's prompt follows the brief
    assert a["id"] in brief and "does x predict y?" in brief and brief.endswith(role)
    assert len(brief.split()) - len(role.split()) < 300
    settings = json.loads(open(argv[argv.index("--settings") + 1]).read())
    from openlab.agents.claude_setup import is_hook_command
    cmd = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]      # either launcher form, PATH-dependent
    assert is_hook_command(cmd) and cmd.endswith("PreToolUse")
    transcript = st.agent_dir(a["id"]) / "transcript.jsonl"
    assert len(transcript.read_text().splitlines()) == 8
    assert st.verify()["ok"]


def test_a_failing_cli_ends_failed(ws, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "fail")
    a = spawn(ws, "x")
    done = final(Store(ws), a["id"])
    assert done["status"] == "failed" and done["exit_code"] == 3


def test_an_agent_at_its_minutes_limit_is_paused_and_asks_the_person_for_more(ws, monkeypatch, person_key):
    from openlab.agents.inbox import decide
    monkeypatch.setenv("FAKE_MODE", "slow")
    st = Store(ws)
    a = spawn(ws, "x", budget={"minutes": 0.03})                  # 1.8 s
    held = wait(st, a["id"], lambda x: x["status"] == "paused")
    assert held["control"] == "pause" and held["control_reason"].startswith("budget: 0 of 0.03 minutes")
    items = [i for i in st.inbox("pending", agent=a["id"]) if i["kind"] == "budget"]
    assert len(items) == 1 and items[0]["args"]["hold"] and items[0]["args"]["change"] == {"minutes": [0.03, 0.06]}
    assert "is paused, not ended" in items[0]["reason"] and "0.03 minutes -> 0.06 minutes" in items[0]["reason"]
    time.sleep(1.0)                                                # still held: one request, no end
    assert st.get(a["id"])["status"] == "paused" and len(st.inbox("pending", agent=a["id"])) == 1
    with person_key.act():
        decide(st, items[0]["id"], True, "go on", pick="5")        # the person's own new limit
    back = wait(st, a["id"], lambda x: x["status"] == "running")
    assert back["budget"]["minutes"] == 5.0 and back["control"] == ""
    assert any("raised your budget: minutes 0.03 minutes -> 5 minutes" in e["payload"]["text"] for e in st.untold(a["id"]))
    st.control(a["id"], "stop", "enough", "terminal:sam")
    assert final(st, a["id"])["status"] == "stopped"


def test_an_agent_at_its_usd_limit_is_paused_on_the_running_estimate_and_a_denial_lets_it_wrap_up(
        ws, monkeypatch, person_key):
    from openlab.agents.inbox import decide
    monkeypatch.setenv("FAKE_MODE", "spend")
    st = Store(ws)
    a = spawn(ws, "x", budget={"usd": 0.5})
    held = wait(st, a["id"], lambda x: x["status"] == "paused")
    assert held["spent"]["usd"] >= 0.5 and held["spent"]["usd_estimated"]
    item = st.inbox("pending", agent=a["id"])[0]
    assert item["args"]["change"] == {"usd": [0.5, 1.0]}
    with person_key.act():
        decide(st, item["id"], False, "that is enough")
    back = wait(st, a["id"], lambda x: x["status"] == "running")
    spent = back["spent"]["usd"]                                    # ~$15: the one message ran past the limit
    assert back["budget"]["usd"] == round(max(0.5, spent) + 0.05, 4)          # a short grace past what it used
    time.sleep(1.0)
    assert st.get(a["id"])["status"] == "running"                  # not held again at once: it can wrap up
    told = " ".join(e["payload"]["text"] for e in st.untold(a["id"]))
    assert "declined more budget" in told and "Wrap up now" in told
    st.control(a["id"], "stop", "enough", "terminal:sam")
    final(st, a["id"])


def test_an_agent_is_told_at_80_percent_of_a_limit(ws, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "slow")
    st = Store(ws)
    a = spawn(ws, "x", budget={"turns": 10})
    held = wait(st, a["id"], lambda x: x["status"] == "paused")
    texts = [e["payload"]["text"] for e in st.events(a["id"], n=500, kinds=("tell",))]
    assert any(re.search(r"you have used (8|9|10) of 10 turns", t) and 'kind="budget"' in t for t in texts), texts
    from openlab.agents.hooks import _sender
    assert _sender(st, st.get(a["id"]), {"by": "openlab"}) == "openlab (your supervisor)"
    assert re.match(r"budget: 1[1-9] of 10 turns", held["control_reason"])      # paused past the limit, not ended
    st.control(a["id"], "stop", "enough", "terminal:sam")
    final(st, a["id"])


def test_pause_is_sigstop_and_does_not_count_toward_minutes_resume_continues_stop_kills(ws, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "slow")
    st = Store(ws)
    a = spawn(ws, "x", budget={"minutes": 0.08})                  # 4.8 s of running time
    wait(st, a["id"], lambda x: x["pid"] and x["spent"].get("turns", 0) >= 3)
    st.control(a["id"], "pause", "checking something", "terminal:sam")
    wait(st, a["id"], lambda x: x["status"] == "paused")
    time.sleep(0.3)
    n = st.get(a["id"])["spent"]["turns"]
    time.sleep(5.5)                                                # longer than the whole minutes budget
    paused = st.get(a["id"])
    assert paused["status"] == "paused" and paused["spent"]["turns"] == n      # SIGSTOP: no output at all
    st.control(a["id"], "resume", "go on", "terminal:sam")
    wait(st, a["id"], lambda x: x["status"] == "running" and x["spent"]["turns"] > n)
    st.control(a["id"], "stop", "enough", "terminal:sam")
    done = final(st, a["id"])
    assert done["status"] == "stopped"
    pgid = done["pgid"]
    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, 0)
    kinds = [e["kind"] for e in st.events(a["id"], n=500)]
    assert {"control_pause", "paused", "control_resume", "resumed", "control_stop", "stopping", "end"} <= set(kinds)


def test_a_dead_supervisor_is_lost(ws, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "slow")
    st = Store(ws)
    a = spawn(ws, "x")
    a = wait(st, a["id"], lambda x: x["pid"])
    os.kill(a["supervisor"], signal.SIGKILL)
    os.killpg(a["pgid"], signal.SIGKILL)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not st.reap():
        time.sleep(0.2)
    assert st.get(a["id"])["status"] == "lost"


def test_codex_adapter_reads_its_jsonl(ws):
    a = spawn(ws, "run the demo", role="experimenter", cli="codex")
    st = Store(ws)
    done = final(st, a["id"])
    assert done["status"] == "done" and done["kind"] == "codex"
    s = done["spent"]
    assert s["tokens_in"] == 2000 and s["tokens_out"] == 300 and s["turns"] == 1 and s["tool_calls"] == 2
    evs = st.events(a["id"], n=50)
    assert any(e["kind"] == "refusal" and "trial budget is spent" in e["summary"] for e in evs)
    assert any(e["kind"] == "say" and e["summary"] == "Finished." for e in evs)
    cmd = json.loads((st.agent_dir(a["id"]) / "command.json").read_text())
    assert any(f'OPENLAB_AGENT="{a["id"]}"' in c for c in cmd)


def test_tree_limits_depth_children_and_budget_share(ws):
    st = Store(ws)
    lead = spawn(ws, "lead", role="lead", budget={"usd": 1.0, "minutes": 60}, start=False)
    with pytest.raises(AgentError, match="remainder"):
        spawn(ws, "child", parent=lead["id"], budget={"usd": 1.5, "minutes": 10}, start=False)
    c1 = spawn(ws, "child", parent=lead["id"], budget={"usd": 0.6, "minutes": 10}, start=False)
    with pytest.raises(AgentError, match=r"\$0\.4"):                # 1.0 - 0.6 held by the live child
        spawn(ws, "child", parent=lead["id"], budget={"usd": 0.5, "minutes": 10}, start=False)
    with pytest.raises(AgentError, match="time"):
        spawn(ws, "child", parent=lead["id"], budget={"usd": 0.1, "minutes": 120}, start=False)
    g = spawn(ws, "grandchild", parent=c1["id"], budget={"usd": 0.1, "minutes": 5}, start=False)
    gg = spawn(ws, "depth 3", parent=g["id"], budget={"usd": 0.05, "minutes": 4}, start=False)
    with pytest.raises(AgentError, match="deep"):
        spawn(ws, "depth 4", parent=gg["id"], budget={"usd": 0.01, "minutes": 1}, start=False)
    for i in range(5):
        spawn(ws, f"kid {i}", parent=lead["id"], budget={"usd": 0.01, "minutes": 1}, start=False)
    with pytest.raises(AgentError, match="live children"):          # max_children = 6
        spawn(ws, "seventh", parent=lead["id"], budget={"usd": 0.01, "minutes": 1}, start=False)
    st.control(c1["id"], "pause", "hold on", "terminal:sam")
    with pytest.raises(AgentError, match="paused by a person"):
        spawn(ws, "under a paused parent", parent=g["id"], budget={"usd": 0.01, "minutes": 1}, start=False)


def test_agent_spawn_tool_uses_the_callers_identity_as_parent(ws, monkeypatch):
    from openlab.agents.identity import acting_as
    from openlab.app import ToolContext, dispatch
    monkeypatch.setenv("FAKE_MODE", "slow")                     # the child still runs, so its share is held
    monkeypatch.setenv("FAKE_N", "40")
    lead = spawn(ws, "lead", role="lead", budget={"usd": 1.0, "minutes": 60}, start=False)
    ctx = ToolContext(str(ws))
    with acting_as(lead["id"]):
        out = dispatch(ctx, "agent_spawn", {"prompt": "screen", "role": "screener", "budget_usd": 0.3, "minutes": 5})
        assert out["ok"], out
        too_much = dispatch(ctx, "agent_spawn", {"prompt": "x", "budget_usd": 0.9, "minutes": 5})
    child = Store(ws).get(out["result"]["id"])
    assert child["parent"] == lead["id"] and child["role"] == "screener"
    assert not too_much["ok"] and "remainder" in too_much["error"]
    final(Store(ws), child["id"])


def test_an_adversary_is_briefed_to_attack_and_a_lead_told_to_decide_passes_it_to_its_children(tmp_path):
    from openlab.agents.spawn import brief, spawn
    from openlab.app import ToolContext, dispatch
    (tmp_path / "openlab.toml").write_text("")
    lead = spawn(tmp_path, "run trial 2", role="lead", start=False, decide="self")
    assert lead["decide"] == "self" and "make your own decisions" in brief(lead)
    adv = spawn(tmp_path, "attack it", role="adversary", parent=lead["id"], start=False,
                budget={"usd": 0.5, "minutes": 10})
    b = brief(adv)
    assert adv["decide"] == "self" and "You are the adversary" in b and "study_branch" in b
    plain = spawn(tmp_path, "x", role="experimenter", start=False)
    assert plain["decide"] == "" and "own decisions" not in brief(plain) and "adversary" not in brief(plain)
    ctx = ToolContext(str(tmp_path))
    q = {"kind": "question", "reason": "which?", "args": {"choices": ["a", "b"]}}
    env = dispatch(ctx, "agent_request", q, agent=adv["id"])
    assert not env["ok"] and env["error"] == "decide it yourself"
    assert dispatch(ctx, "agent_request", {**q}, agent=plain["id"])["ok"]
    with pytest.raises(Exception, match="decide is one of"):
        spawn(tmp_path, "x", start=False, decide="maybe")


def test_a_branch_copy_runs_as_its_own_code(tmp_path):
    from openlab.app import ToolContext, dispatch
    (tmp_path / "openlab.toml").write_text("")
    ctx = ToolContext(str(tmp_path))
    assert dispatch(ctx, "study_new", {"folder": "s", "template": "static"})["ok"]
    env = dispatch(ctx, "study_branch", {"folder": "s", "branch": "adv-tz"})
    assert env["ok"], env
    assert (tmp_path / "s" / "branches" / "adv-tz" / "model.py").read_text() == (tmp_path / "s" / "model.py").read_text()
    assert not dispatch(ctx, "study_branch", {"folder": "s", "branch": "Bad Name"})["ok"]
