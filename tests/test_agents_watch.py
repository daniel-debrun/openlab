"""`openlab watch --once` and `openlab agents`: the tree in order, states, and the flags (STUCK, LOOP, OVER, WAITING,
DEAD-END, BYPASS)."""

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from openlab.agents import view
from openlab.agents.cli import frame, main
from openlab.agents.store import Store


def _age(st, agent, minutes):
    old = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat(timespec="milliseconds")
    con = sqlite3.connect(st.path)
    con.execute("UPDATE agents SET last_event_at=?, created=? WHERE id=?", (old, old, agent))
    con.commit()
    con.close()


@pytest.fixture
def demo(tmp_path):
    (tmp_path / "openlab.toml").write_text("[agents]\nstuck_minutes = 5\n")
    st = Store(tmp_path)
    st.register("cc-lead0001", "claude-session", role="session", question="does x predict y?",
                budget={"usd": 2.0}, spent={"usd": 0.4, "tokens_in": 120000, "tokens_out": 3000})
    st.register("cc-lead0001.scr001", "claude-subagent", role="openlab-screener", parent="cc-lead0001",
                question="which papers measure x", study="lit")
    st.register("cc-lead0001.exp001", "claude-subagent", role="openlab-experimenter", parent="cc-lead0001",
                question="lagged x features beat persistence", study="demo")
    st.register("replicator-aa11", "spawned", role="replicator", budget={"usd": 1.0, "minutes": 60},
                spent={"usd": 0.95, "usd_estimated": True}, question="replicate table 2")
    st.register("cc-other002", "claude-session", role="session")
    st.event("cc-lead0001", "tool", tool="mcp__openlab__openlab_study_status", summary="mcp__openlab__openlab_study_status demo")
    for _ in range(3):
        st.event("cc-lead0001.exp001", "refusal", tool="mcp__openlab__openlab_study_amend",
                 summary="mcp__openlab__openlab_study_amend: budget can only be amended by a person")
    st.event("cc-lead0001.scr001", "tool", tool="Read", summary="Read paper.pdf")
    _age(st, "cc-lead0001.scr001", 12)
    st.update("replicator-aa11", last_event_at=(datetime.now(timezone.utc) - timedelta(minutes=12)).isoformat())
    st.request("cc-lead0001.exp001", "holdout", "the model is final; one look", study="demo", cost_note="look 1 of 1")
    st.event("cc-other002", "bypass_attempt", tool="Bash", summary="Bash: a holdout look (openlab study holdout) is a person's act")
    return tmp_path, st


def test_watch_once_renders_tree_states_and_flags(demo, monkeypatch, capsys):
    root, st = demo
    monkeypatch.setattr(view, "study_stats", lambda root, studies: ({}, {
        str((root / "demo").resolve()): {"dead_ends": [("cc-lead0001.scr001", "lagged features of x beat persistence lag")],
                                         "hypotheses": []}}))
    text, rows = frame(st, width=200, studies=[])
    order = [r["agent"]["id"] for r in rows]
    assert order == ["cc-lead0001", "cc-lead0001.scr001", "cc-lead0001.exp001", "replicator-aa11", "cc-other002"]
    by = {r["agent"]["id"]: r for r in rows}
    assert by["replicator-aa11"]["state"] == "stuck" and "STUCK" in by["replicator-aa11"]["flags"]
    assert by["cc-lead0001.scr001"]["state"] == "idle"          # a hook agent's silence: idle, then ended (reap)
    exp = by["cc-lead0001.exp001"]
    assert exp["state"] == "waiting" and {"LOOP×3", "WAITING", "DEAD-END"} <= set(exp["flags"])
    assert "OVER" in by["replicator-aa11"]["flags"]
    assert "BYPASS" in by["cc-other002"]["flags"]
    assert "└ cc-lead0001.scr001" in text and "inbox: 1 pending" in text
    assert "! refusal" in text and "budget can only be amended" in text
    assert "! bypass attempt" in text and "matches a dead end by cc-lead0001.scr001" in text
    assert "~$0.95/1 [######]" in text
    monkeypatch.chdir(root)
    assert main(["watch", "--once"]) == 0
    assert "openlab watch" in capsys.readouterr().out


def test_agents_table_tree_and_json(demo, monkeypatch, capsys):
    root, st = demo
    monkeypatch.chdir(root)
    assert main(["agents", "--tree"]) == 0
    out = capsys.readouterr().out
    assert "  └ cc-lead0001.exp001" in out and "openlab-experimenter" in out and "LOOP×3" in out
    assert main(["agents", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {r["id"] for r in rows} >= {"cc-lead0001", "replicator-aa11"}


def test_agents_counts_trials_and_claims_from_the_study_records(tmp_path, monkeypatch, capsys):
    from openlab.agents.identity import acting_as
    from openlab.study.record import Record
    (tmp_path / "openlab.toml").write_text("")
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "study.toml").write_text("")
    st = Store(tmp_path)
    st.register("cc-a.exp", "claude-subagent", role="openlab-experimenter")
    rec = Record(tmp_path / "demo")
    with acting_as("cc-a.exp"):
        rec.add("action", {"verb": "run", "args": {}, "outcome": {"ok": False, "error": "refused"}})
    per, _ = view.study_stats(tmp_path, [tmp_path / "demo"])
    assert per["cc-a.exp"]["refusals"] == 1 and per["cc-a.exp"]["writes"] == 2   # its agent node + the action
    monkeypatch.chdir(tmp_path)
    assert main(["agents"]) == 0
    assert "cc-a.exp" in capsys.readouterr().out


def test_control_cli(demo, monkeypatch, capsys, person_key):
    from openlab.agents import person
    root, st = demo
    monkeypatch.chdir(root)
    assert main(["agent", "pause", "cc-lead0001", "--reason", "lunch"]) == 0
    assert st.get("cc-lead0001")["status"] == "paused"
    person_key.answers("")                                          # an agent's shell has no passphrase to type
    assert main(["agent", "resume", "cc-lead0001", "--reason", "back"]) == 1
    assert st.get("cc-lead0001")["status"] == "paused"
    person_key.answers()
    assert main(["agent", "resume", "cc-lead0001", "--reason", "back"]) == 0
    ev = [e for e in st.events("cc-lead0001") if e["kind"] == "control_resume"][-1]
    assert ev["payload"]["by"] == "person:sam" and ev["payload"]["sig"]["key"] == person.info()["fingerprint"]
    assert main(["agent", "log", "cc-lead0001"]) == 0
    assert "control_resume" in capsys.readouterr().out
