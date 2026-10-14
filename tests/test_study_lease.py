"""A frontier item is an exclusive lease: one agent holds it, another agent's take is refused naming the holder,
the holder's take renews it, and it frees on release, when the holder ends (control store) or when it expires."""

from datetime import timedelta

import pytest

from openlab import study
from openlab.agents.identity import acting_as
from openlab.agents.store import Store
from openlab.app.registry import ToolContext, dispatch
from openlab.study import cli, frontier
from openlab.study.errors import StudyError


@pytest.fixture
def s(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENLAB_AGENT", raising=False)
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "openlab.toml").write_text("")
    st = study.new(ws / "s", "static")
    st.question("Does feature x carry signal?")
    return st


def first(s):
    return s.frontier()["items"][0]["id"]


def test_two_agents_cannot_take_the_same_item(s, capsys):
    item = first(s)
    with acting_as("exp-a"):
        out = s.take(item)
    assert out["lease"]["agent"] == "exp-a" and not out["renewed"]
    assert s.frontier()["items"][0]["held_by"]["agent"] == "exp-a"
    with acting_as("exp-b"), pytest.raises(StudyError, match="is taken by exp-a until") as e:
        s.take(item)
    assert "one agent at a time" in e.value.meaning
    assert not s.record.last("action", verb="take").payload["outcome"]["ok"]
    with acting_as("exp-a"):                                        # the holder renews
        again = s.take(item, minutes=90)
    assert again["renewed"] and again["lease"]["expires"] > out["lease"]["expires"]
    with acting_as("exp-b"), pytest.raises(StudyError, match="held by exp-a"):
        s.release(item)
    with acting_as("exp-a"):
        s.release(item)
    assert s.frontier()["items"][0]["held_by"] is None
    with acting_as("exp-b"):
        assert s.take(item)["lease"]["agent"] == "exp-b"
    with acting_as("exp-b"):
        s.release(item)
        with pytest.raises(StudyError, match="not leased"):
            s.release(item)
        s.take(item)
    env = dispatch(ToolContext(str(s.folder.parent)), "study_take", {"folder": "s", "item": item}, agent="exp-c")
    assert not env["ok"] and "exp-b" in env["error"]
    assert cli.main(["frontier", str(s.folder)]) == 0


def test_a_lease_frees_when_its_agent_ends_or_it_expires(s, monkeypatch):
    item = first(s)
    st = Store(s.folder.parent)
    st.register("exp-a", "spawned", role="experimenter", status="running")
    with acting_as("exp-a"):
        s.take(item, minutes=5)
    with acting_as("exp-b"), pytest.raises(StudyError, match="taken by exp-a"):
        s.take(item)
    st.set_status("exp-a", "done")                                  # the holder finished: released
    with acting_as("exp-b"):
        s.take(item, minutes=5)
    real = frontier.clock.utc
    monkeypatch.setattr(frontier.clock, "utc", lambda: real() + timedelta(minutes=6))
    assert frontier.leases(s) == {}                                 # expired
    with acting_as("exp-c"):
        assert s.take(item)["lease"]["agent"] == "exp-c"


# --- a person releases any lease; a killed Claude Code session's leases free --------------------------------------
def test_a_person_releases_another_agents_lease(s, person_key, capsys):
    item = first(s)
    with acting_as("exp-b"):
        s.take(item)
    with acting_as("exp-a"), pytest.raises(StudyError, match="held by exp-b"):
        s.release(item)
    assert cli.main(["release", str(s.folder), item, "--person"]) == 0
    assert "released" in capsys.readouterr().out and s.frontier()["items"][0]["held_by"] is None
    act = s.record.last("action", verb="release")
    assert act.actor == "person" and act.payload["outcome"]["ok"]
    from openlab.agents.hooks import gate
    assert gate("Bash", {"command": f"openlab study release s {item} --person"})


def test_without_the_person_key_release_person_is_refused(s, person_home, capsys):
    item = first(s)
    with acting_as("exp-b"):
        s.take(item)
    assert cli.main(["release", str(s.folder), item, "--person"]) == 1
    assert "person's act" in capsys.readouterr().err and s.frontier()["items"][0]["held_by"]["agent"] == "exp-b"


def _quiet(st, agent, seconds):
    from datetime import datetime, timezone
    at = datetime.fromtimestamp(datetime.now(timezone.utc).timestamp() - seconds, timezone.utc)
    st.update(agent, last_event_at=at.isoformat(timespec="milliseconds"))


def test_quiet_hook_sessions_go_idle_then_end_and_free_their_leases(s, capsys):
    from openlab.agents import cli as acli
    st = Store(s.folder.parent)
    st.register("cc-x", "claude-session", role="session", cli="claude", session_id="x", status="running")
    st.register("cc-x.sub", "claude-subagent", role="openlab-experimenter", parent="cc-x", cli="claude",
                session_id="x", ext="t1", status="running")
    item = first(s)
    with acting_as("cc-x.sub"):
        s.take(item)
    _quiet(st, "cc-x", 6 * 60)
    _quiet(st, "cc-x.sub", 6 * 60)
    st.reap()
    assert st.get("cc-x")["status"] == "idle" and st.get("cc-x.sub")["status"] == "idle"     # not STUCK forever
    assert s.frontier()["items"][0]["held_by"]["agent"] == "cc-x.sub"
    _quiet(st, "cc-x", 13 * 3600)                                  # past [agents] session_timeout_hours (12)
    assert s.frontier()["items"][0]["held_by"] is None              # the session ended, and its sub-agent with it
    assert st.get("cc-x")["status"] == "ended" and st.get("cc-x.sub")["status"] == "ended"
    assert any(e["kind"] == "ended" for e in st.events("cc-x.sub"))
    assert acli.agents_main([str(s.folder.parent)]) == 0
    assert "cc-x" not in capsys.readouterr().out
    assert acli.agents_main([str(s.folder.parent), "--all"]) == 0
    assert "cc-x" in capsys.readouterr().out
    with acting_as("exp-z"):
        assert s.take(item)["lease"]["agent"] == "exp-z"
