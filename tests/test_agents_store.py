"""The control store: hash-chained events with tamper detection, bounded payloads, the inbox, the MCP claim."""

import sqlite3
import time

import pytest

from openlab.agents import inbox, person
from openlab.agents.store import Store, bound


def test_events_are_hash_chained_and_an_edit_is_caught(tmp_path):
    st = Store(tmp_path)
    st.register("a1", "terminal")
    for i in range(5):
        st.event("a1", "note", summary=f"note {i}", payload={"i": i})
    assert st.verify() == {"ok": True, "events": 5, "broken_at": None, "head": st.verify()["head"]}
    con = sqlite3.connect(st.path)
    con.execute("UPDATE events SET summary='rewritten' WHERE seq=3")
    con.commit()
    con.close()
    v = st.verify()
    assert not v["ok"] and v["broken_at"] == 3


def test_payloads_and_summaries_are_bounded(tmp_path):
    st = Store(tmp_path)
    st.register("a1", "terminal")
    st.event("a1", "tool", summary="x" * 5000, payload={"input": {"content": "y" * 100000, "list": list(range(500))}})
    e = st.events("a1")[-1]
    assert len(e["summary"]) <= 240
    assert len(str(e["payload"])) < 6000
    assert bound([1] * 100)[-1] == "… 80 more"


def test_tree_helpers_and_control(tmp_path):
    st = Store(tmp_path)
    st.register("lead", "claude-session")
    st.register("kid", "claude-subagent", parent="lead")
    st.register("grandkid", "claude-subagent", parent="kid")
    assert [a["id"] for a in st.ancestors("grandkid")] == ["kid", "lead"] and st.depth("grandkid") == 2
    assert st.blocked("grandkid") is None
    st.control("lead", "pause", "lunch", "terminal:sam")
    assert st.blocked("grandkid") == ("lead", "paused", "lunch")
    assert st.get("lead")["status"] == "paused"
    st.control("lead", "resume", "back", "terminal:sam")
    assert st.blocked("grandkid") is None and st.get("lead")["status"] == "idle"


def test_mcp_claim_matches_tool_and_input_once(tmp_path):
    st = Store(tmp_path)
    st.pend("screener-1", "openlab_study_screen", {"folder": "lit", "source": "s1"})
    st.pend("screener-2", "openlab_study_screen", {"folder": "lit", "source": "s2"})
    # the server sees the client's arguments plus defaults: a subset match still finds the right caller
    hit = st.claim("openlab_study_screen", {"folder": "lit", "source": "s2", "quote": ""})
    assert hit["agent"] == "screener-2"
    assert st.claim("openlab_study_screen", {"folder": "lit", "source": "s2"}) is None      # claimed once
    assert st.claim("openlab_study_screen", {"folder": "lit", "source": "s1"})["agent"] == "screener-1"
    st.pend("old", "openlab_study_run", {"folder": "x"})
    assert st.claim("openlab_study_run", {"folder": "x"}, window=0.0) is None              # too old


def test_inbox_request_decide_and_budget_act(tmp_path, person_key):
    st = Store(tmp_path)
    st.register("exp-1", "spawned", budget={"usd": 1.0, "minutes": 10})
    args = inbox.check_request(st, "exp-1", "budget", {"usd": 1.5})
    assert args == {"usd": 1.5, "change": {"usd": [1.0, 1.5]}}
    item = st.request("exp-1", "budget", "need 20 more trials of compute", args=args, cost_note="$1 -> $1.5")
    assert [i["id"] for i in st.inbox()] == [item]
    with pytest.raises(PermissionError, match="signed with the person key"):      # no person, no decision
        inbox.decide(st, item, True, "fine")
    with person_key.act():
        out = inbox.decide(st, item, True, "fine")
    assert out["status"] == "approved" and out["decided_by"] == "person:sam"
    assert st.get("exp-1")["budget"]["usd"] == 1.5
    assert st.inbox() == []
    with pytest.raises(ValueError, match="already approved"), person_key.act():
        inbox.decide(st, item, False, "no")
    second = st.request("exp-1", "holdout", "look at the holdout", study="demo")
    with person_key.act():
        denied = inbox.decide(st, second, False, "too early")
    assert denied["status"] == "denied" and st.item(second)["decision_reason"] == "too early"
    kinds = [e["kind"] for e in st.events("exp-1")]
    assert kinds.count("request") == 2 and kinds.count("decision") == 2


def test_inbox_cli_needs_a_person(tmp_path, monkeypatch, capsys, person_home):
    from conftest import PASSPHRASE
    from openlab.agents.cli import main
    st = Store(tmp_path)
    st.register("exp-1", "spawned")
    item = st.request("exp-1", "other", "may the GPU box be used?")
    monkeypatch.chdir(tmp_path)
    assert main(["inbox", "approve", str(item), "--reason", "ok"]) == 1          # no person key here
    assert "openlab person init" in capsys.readouterr().err
    person.create("sam", PASSPHRASE)
    asked = []
    monkeypatch.setattr(person, "_ask", lambda prompt: asked.append(prompt) or "")   # no passphrase typed
    assert main(["inbox", "approve", str(item), "--reason", "ok"]) == 1
    assert "cancelled" in capsys.readouterr().err and st.item(item)["status"] == "pending"
    monkeypatch.setattr(person, "_ask", lambda prompt: PASSPHRASE)
    assert main(["inbox", "approve", str(item), "--reason", "ok"]) == 0
    assert st.item(item)["status"] == "approved"
    assert main(["inbox"]) == 0 and "empty" in capsys.readouterr().out


def test_an_amend_request_carries_its_fields_and_the_approval_writes_them(tmp_path, person_key):
    """The agent that asked may have ended: approving writes the fields it named, then records the amendment."""
    from openlab import study
    from openlab.study import spec
    s = study.new(tmp_path / "s", "static")
    spec.seal(s.folder, s.record, "agent")
    st = Store(tmp_path)
    st.register("exp-1", "spawned", budget={"usd": 1.0})
    with pytest.raises(inbox.RequestError, match="fields it sets"):
        inbox.check_request(st, "exp-1", "amend", {"study.toml": {"split": "kfold"}}, study="s")
    args = inbox.check_request(st, "exp-1", "amend", {"study.toml": {"split": {"k": 4, "reseed": True}}}, study="s")
    item = st.request("exp-1", "amend", "four folds", study="s", args=args)
    head, details = inbox.describe(st, st.item(item))
    assert "study.toml [split] k = 4" in details
    toml = (s.folder / "study.toml").read_text()
    with person_key.act():
        out = inbox.decide(st, item, True, "four folds, reseeded")
    assert out["status"] == "approved"
    s = study.open(s.folder)
    assert s.spec.split["k"] == 4 and s.spec.split["reseed"] is True
    assert s.record.last("amendment").payload["person"]
    assert (s.folder / "study.toml").read_text() != toml and "# " in (s.folder / "study.toml").read_text()
    bad = st.request("exp-1", "amend", "a field openlab refuses", study="s",
                     args={"study.toml": {"split": {"scheme": "nonsense"}}})
    before = (s.folder / "study.toml").read_text()
    with pytest.raises(Exception), person_key.act():
        inbox.decide(st, bad, True, "try")
    assert (s.folder / "study.toml").read_text() == before


def test_approving_an_ended_builders_request_resumes_the_build(tmp_path, monkeypatch, capsys, person_home):
    """The person approves after the builder ended: the build goes on from there (unless --no-resume); a denial
    and a request from a live agent resume nothing."""
    from conftest import PASSPHRASE
    from openlab import __main__ as top
    from openlab.agents.cli import main
    (tmp_path / "paper1").mkdir()
    (tmp_path / "paper1" / "replication.toml").write_text("")
    st = Store(tmp_path)
    st.register("replicator-1", "spawned", role="replicator", study="paper1")
    person.create("sam", PASSPHRASE)
    monkeypatch.setattr(person, "_ask", lambda prompt: PASSPHRASE)
    monkeypatch.chdir(tmp_path)
    ran = []
    real = top.main
    monkeypatch.setattr(top, "main", lambda argv: ran.append(argv) or 0 if argv[0] == "replicate" else real(argv))
    live = st.request("replicator-1", "other", "more seeds?", study="paper1")
    assert main(["inbox", "approve", str(live), "--reason", "yes"]) == 0 and ran == []
    st.set_status("replicator-1", "done")
    first, second, third = (st.request("replicator-1", "other", f"q{i}", study="paper1") for i in range(3))
    assert main(["inbox", "approve", str(first), "--reason", "yes"]) == 0
    assert ran == [["replicate", "build", "paper1"]] and "resuming" in capsys.readouterr().out
    assert main(["inbox", "approve", str(second), "--reason", "yes", "--no-resume"]) == 0
    assert main(["inbox", "deny", str(third), "--reason", "no"]) == 0 and len(ran) == 1


def test_a_request_runs_the_persons_notify_command(tmp_path):
    out = tmp_path / "told"
    (tmp_path / "openlab.toml").write_text(
        f'[agents]\nnotify = ["sh", "-c", \'printf "%s|%s" "$OPENLAB_EVENT" "$1" > {out}\', "x"]\n')
    st = Store(tmp_path)
    st.register("exp-1", "spawned")
    item = st.request("exp-1", "other", "may the GPU box be used?")
    for _ in range(100):
        if out.is_file() and out.read_text():
            break
        time.sleep(0.05)
    assert out.read_text() == f"request|inbox #{item} from exp-1 (other): may the GPU box be used?"
    (tmp_path / "openlab.toml").write_text('[agents]\nnotify = ["/no/such/command"]\n')
    st.request("exp-1", "other", "a missing command changes nothing")


def test_inbox_wait_returns_on_a_new_request_or_an_agent_ending(tmp_path, monkeypatch, capsys):
    import threading
    from openlab.agents.cli import main
    st = Store(tmp_path)
    st.register("exp-1", "spawned")
    st.set_status("exp-1", "running")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    assert main(["inbox", "wait", "--minutes", "0.0005"]) == 2
    threading.Timer(0.05, lambda: st.request("exp-1", "other", "a question")).start()
    assert main(["inbox", "wait", "--minutes", "1"]) == 0 and "a question" in capsys.readouterr().out
    threading.Timer(0.05, lambda: st.set_status("exp-1", "done")).start()
    assert main(["inbox", "wait", "--minutes", "1"]) == 0 and "ended  exp-1" in capsys.readouterr().out


def test_a_compute_request_carries_what_this_machine_has_and_approving_runs_nothing(tmp_path, person_key,
                                                                                    monkeypatch):
    from openlab.study import machine
    monkeypatch.setenv("OPENLAB_BOX_DIR", str(tmp_path / "box"))
    monkeypatch.setattr(machine, "gpus", lambda: [{"index": 0, "name": "RTX 3090", "memory_mb": 24576}])
    st = Store(tmp_path)
    st.register("exp-1", "spawned", budget={"usd": 1.0})
    for bad in ({}, {"gpus": 0}, {"tpus": 8}):
        with pytest.raises(inbox.RequestError, match="what another machine would need"):
            inbox.check_request(st, "exp-1", "compute", bad)
    args = inbox.check_request(st, "exp-1", "compute", {"gpus": 2, "hours": 6})
    assert args["gpus"] == 2 and args["hours"] == 6 and args["machine"]["gpus"] == 1          # openlab adds this one
    assert args["machine"]["cores"] >= 1 and args["machine"]["waiting"] == 0
    item = st.request("exp-1", "compute", "the DBM needs two GPUs at once", args=args)
    head, details = inbox.describe(st, st.inbox()[0])
    assert details[0] == "another machine with 2 gpus, 6 hours" and "rents nothing" in details[2]
    with person_key.act():
        out = inbox.decide(st, item, True, "added gpu-box-2")
    assert out["status"] == "approved" and st.get("exp-1")["budget"] == {"usd": 1.0}       # nothing else changed
