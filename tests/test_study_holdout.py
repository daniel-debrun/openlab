"""Guarantee: a holdout look is a person's act, recorded with the study's hashes before anything is scored and
counted against the budget; an agent cannot take one through the Study API, the tools or the CLI."""

import io
import sys
import types

import pytest

from openlab import study
from openlab.app import ToolContext, dispatch
from openlab.study import cli
from openlab.study.errors import StudyError
from openlab.study.fingerprint import fingerprint, key


@pytest.fixture
def made(tmp_path):
    s = study.new(tmp_path / "s", "temporal")
    s.materialise()
    s.seal()
    return s


@pytest.fixture
def runner(monkeypatch):
    """A stand-in for openlab.study.runner: notes the looks on record when it starts, then writes the holdout run
    and result its look allows (or fails, with `fail`)."""
    mod = types.ModuleType("openlab.study.runner")
    mod.seen, mod.fail = [], False

    def run(s, *, note="", actor="agent", segment="research", look=None, code_dir=None):
        rec = s.record
        mod.seen.append({"look": look, "on_record": [n.id for n in rec.find("look")], "actor": actor,
                         "segment": segment})
        if mod.fail:
            raise StudyError("model.py raised", "ValueError: boom", "fix model.py")
        fp = fingerprint(s)
        links = [("under", (rec.last("amendment") or rec.last("plan")).id), ("after", look),
                 *[("uses", n.id) for n in rec.find("dataset")]]
        node = rec.add("run", {"segment": segment, "key": key(fp, segment), "cached": False, "spec_sha": fp["spec_sha"],
                               "code": fp["code"], "data": fp["data"], "env": fp["env"], "sandbox": "stub",
                               "audit": {"verdict": "unchecked", "checked": 0}, "seconds": 0.0}, actor=actor, links=links)
        return rec.add("result", {"metrics": {"mae": {"value": 0.5, "lo": 0.4, "hi": 0.6}}}, actor=actor,
                       links=[("of", node.id)])
    mod.run = run
    monkeypatch.setitem(sys.modules, "openlab.study.runner", mod)
    return mod


class Tty(io.StringIO):
    def isatty(self):
        return True


def test_an_agent_cannot_take_a_look(made, runner, tmp_path, monkeypatch, capsys):
    with pytest.raises(StudyError, match="only a person"):
        made.holdout("peek", person=False)
    act = made.record.last("action")
    assert act.actor == "agent" and act.payload["verb"] == "holdout" and not act.payload["outcome"]["ok"]
    ctx = ToolContext(str(tmp_path))
    for name in ("study_holdout", "study_amend_person"):
        env = dispatch(ctx, name, {"folder": "s", "reason": "peek"})
        assert not env["ok"] and env["error"] == "a reviewer's act"
    monkeypatch.setattr(sys, "stdin", io.StringIO("s\n"))         # the name piped in is not a person at a terminal
    assert cli.main(["holdout", str(made.folder), "--reason", "peek"]) == 1
    assert cli.main(["amend", str(made.folder), "--reason", "peek", "--person"]) == 1
    assert capsys.readouterr().err.count("in their own terminal") == 2
    assert made.record.find("look") == [] and runner.seen == []


def test_a_person_look_is_recorded_before_the_run(made, runner, person_key):
    with person_key.act():
        r = made.holdout("the final check", person=True)
    look = made.record.last("look")
    assert runner.seen == [{"look": look.id, "on_record": [look.id], "actor": "person", "segment": "holdout"}]
    fp = fingerprint(made)
    assert look.actor == "person" and look.payload["n"] == 1
    assert (look.payload["spec_sha"], look.payload["code"], look.payload["data"]) == \
        (fp["spec_sha"], fp["code"], fp["data"]) and look.payload["data"]
    run = made.record.links(r["id"], rel="of")[0][2]
    assert made.record.links(run, rel="after") == [(run, "after", look.id)]
    assert made.record.last("action").payload["outcome"] == {"ok": True, "id": r["id"]}


def test_the_look_budget_is_enforced(made, runner, person_key):
    with person_key.act():
        made.holdout("first", person=True)
        with pytest.raises(StudyError, match="budget is spent"):
            made.holdout("second", person=True)
    assert len(made.record.find("look")) == 1 and len(runner.seen) == 1


def test_a_failed_run_leaves_the_look_spent(made, runner, person_key):
    runner.fail = True
    with pytest.raises(StudyError, match="look 1 .* is spent") as e, person_key.act():
        made.holdout("final", person=True)
    assert "fix model.py" in e.value.fix
    assert len(made.record.find("look")) == 1 and made.status()["looks"]["used"] == 1
    runner.fail = False
    with pytest.raises(StudyError, match="budget is spent"), person_key.act():
        made.holdout("again", person=True)


def test_a_person_acts_through_a_terminal_or_the_reviewer_tool(made, runner, tmp_path, monkeypatch, capsys,
                                                               person_key):
    toml = made.folder / "study.toml"
    toml.write_text(toml.read_text().replace("holdout_looks = 1", "holdout_looks = 2"))
    person_key.answers("wrong one", "wrong two", "wrong three")      # the passphrase is the confirmation
    assert cli.main(["amend", str(made.folder), "--reason", "two looks", "--person"]) == 1
    assert "wrong passphrases" in capsys.readouterr().err and made.record.find("amendment") == []
    person_key.answers("wrong one")                                  # a second try is fine
    assert cli.main(["amend", str(made.folder), "--reason", "two looks", "--person"]) == 0
    assert made.record.last("amendment").actor == "person"
    assert cli.main(["holdout", str(made.folder), "--reason", "first", "--json"]) == 0
    env = dispatch(ToolContext(str(tmp_path)), "study_holdout", {"folder": "s", "reason": "second"}, human=True)
    assert not env["ok"]                                             # the reviewer secret alone signs nothing
    with person_key.act():
        env = dispatch(ToolContext(str(tmp_path)), "study_holdout", {"folder": "s", "reason": "second"}, human=True)
    assert env["ok"] and env["result"]["kind"] == "result"
    assert [n.payload["n"] for n in made.record.find("look")] == [1, 2]
