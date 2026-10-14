"""Guarantee: a workspace that pins the person key takes another key only through a rotation the pinned
key signed. An adversarial agent's bypass -- an agent makes its own key under another HOME, pins it, and spends the holdout
as "person:sam" -- is refused at every step, nothing is spent, and verify and doctor say what happened. An edited
openlab.toml that disagrees with the control store's pin chain refuses person acts until it is restored."""

import json

import pytest

from conftest import PASSPHRASE
from test_study_holdout import runner  # noqa: F401 -- the holdout run stand-in (a fixture)
from openlab import study
from openlab.agents import hooks, person
from openlab.agents.cli import person_main
from openlab.agents.store import Store
from openlab.study import cli as study_cli
from openlab.study import personkey
from openlab.study.errors import StudyError

AGENT_PASS = "agentpass99"


@pytest.fixture
def ws(tmp_path, person_key, monkeypatch, capsys):
    """A workspace pinned to sam's key by `openlab person pin`, with a sealed study `demo` (one holdout look)."""
    root = tmp_path / "r4"
    root.mkdir()
    (root / "openlab.toml").write_text("[agents]\nbudget_usd = 1.0\n")
    monkeypatch.chdir(root)
    assert person_main(["pin"]) == 0
    s = study.new(root / "demo", "temporal")
    s.materialise()
    s.seal()
    capsys.readouterr()
    return root


def agent_home(tmp_path, monkeypatch):
    """`HOME=/tmp/ah openlab person init --handle sam`: the agent's own key and passphrase."""
    home = tmp_path / "ah"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    info = person.create("sam", AGENT_PASS)
    monkeypatch.setattr(person, "_ask", lambda prompt: AGENT_PASS)
    return info


def test_an_adversarial_repin_sequence_is_refused_and_nothing_is_spent(ws, tmp_path, monkeypatch, capsys, runner):
    sam = person.info()["fingerprint"]
    toml = (ws / "openlab.toml").read_text()
    assert f'fingerprint = "{sam}"' in toml
    owned = agent_home(tmp_path, monkeypatch)
    assert owned["fingerprint"] != sam
    # HOME=/tmp/ah openlab person pin
    assert person_main(["pin"]) == 1
    err = capsys.readouterr().err
    assert f"error: this workspace pins the person key {sam}" in err
    assert "meaning:" in err and "never endorsed it" in err and "fix:" in err and "openlab person rotate" in err
    assert (ws / "openlab.toml").read_text() == toml
    st = person.pin_state(ws)
    assert st["pin"] == sam and st["chain"] == [sam] and not st["conflict"]
    # (sleep 3; echo demo; sleep 1; echo agentpass99) | HOME=/tmp/ah script -qc "openlab study holdout demo ..."
    assert study_cli.main(["holdout", str(ws / "demo"), "--reason", "forged"]) == 1
    err = capsys.readouterr().err
    assert f"this workspace pins the person key {sam}" in err
    s = study.open(ws / "demo")
    assert s.record.find("look") == [] and s.record.find("person_key") == [] and runner.seen == []
    assert s.status()["looks"]["used"] == 0
    v = s.verify()
    assert v["ok"] and v["signatures"]["signed"] == 0 and v["workspace"]["pin"] == sam
    assert study_cli.main(["verify", str(ws / "demo")]) == 0
    assert f"workspace pin: {sam}" in capsys.readouterr().out


def test_a_rotation_signed_by_the_pinned_key_moves_the_pin(ws, monkeypatch, capsys, runner):
    old = person.info()["fingerprint"]
    typed = [PASSPHRASE, "a new passphrase", "a new passphrase"]
    monkeypatch.setattr(person, "_ask", lambda prompt: typed.pop(0) if typed else "a new passphrase")
    assert person_main(["rotate"]) == 0
    assert "this workspace's pin is updated" in capsys.readouterr().out
    new = person.info()["fingerprint"]
    st = person.pin_state(ws)
    assert st["chain"] == [old, new] and st["pin"] == new and not st["conflict"] and not st["invalid"]
    assert f'fingerprint = "{new}"' in (ws / "openlab.toml").read_text()
    s = study.open(ws / "demo")
    with personkey.signing(person.unlock("a new passphrase")):
        s.holdout("the final check", person=True)
    v = s.verify()
    assert v["ok"] and v["signatures"]["pinned"] == new and v["workspace"]["chain"] == [old, new]
    with personkey.signing(person.unlock("a new passphrase")):     # re-pinning the same key adds nothing
        person.pin_workspace(ws, person.unlock("a new passphrase"))
    assert len(Store(ws).events(None, n=100, kinds=("person_pin",))) == 2


def test_an_edited_openlab_toml_refuses_person_acts_and_shows_in_verify_and_doctor(ws, tmp_path, person_key,
                                                                                    monkeypatch, capsys, runner):
    sam = person.info()["fingerprint"]
    sam_signer = person_key.signer
    sam_home = person.config_dir().parent.parent
    owned = agent_home(tmp_path, monkeypatch)
    toml = ws / "openlab.toml"
    toml.write_text(toml.read_text().replace(sam, owned["fingerprint"]))   # hooks off: the agent edits the file
    st = person.pin_state(ws)
    assert st["conflict"] and "openlab.toml was edited" in st["conflict"] and st["pin"] == sam
    s = study.open(ws / "demo")
    for signer in (person.unlock(AGENT_PASS), sam_signer):          # neither key acts on a tampered workspace
        with personkey.signing(signer), pytest.raises(StudyError, match="tampered|pins the person key"):
            s.holdout("forged", person=True)
    assert s.record.find("look") == [] and runner.seen == []
    assert person_main(["pin"]) == 1                                # the agent's key cannot "repair" it either
    v = s.verify()
    assert not v["ok"] and v["workspace"]["conflict"]
    assert study_cli.main(["verify", str(ws / "demo")]) == 1
    assert "TAMPERED: openlab.toml [person] fingerprint is" in capsys.readouterr().out
    from openlab import doctor
    rows = {r["check"]: r for r in doctor.check(ws).rows}
    assert rows["person pin"]["ok"] is False and "TAMPERED" in rows["person pin"]["detail"]
    # the person restores it with the pinned key
    monkeypatch.setenv("HOME", str(sam_home))
    monkeypatch.setattr(person, "_ask", lambda prompt: PASSPHRASE)
    assert person_main(["pin"]) == 0
    assert not person.pin_state(ws)["conflict"]
    with person_key.act():
        s.holdout("the final check", person=True)
    assert s.verify()["ok"]


def test_a_pin_event_written_straight_into_the_control_store_is_ignored(ws, tmp_path, monkeypatch):
    sam = person.info()["fingerprint"]
    owned = agent_home(tmp_path, monkeypatch)
    Store(ws).event("workspace", "person_pin", summary="forged", payload={
        "fingerprint": owned["fingerprint"], "public": owned["public"], "handle": "sam", "rotates": sam,
        "hops": [{"old": sam, "old_public": person.unlock(AGENT_PASS).public, "new_public": owned["public"],
                  "sig": "AAAA"}]})
    st = person.pin_state(ws)
    assert st["pin"] == sam and st["chain"] == [sam]
    assert st["invalid"] and "never endorsed" in st["invalid"][0]["why"]


def test_a_forged_endorsement_in_the_key_file_is_checked_not_trusted(ws, tmp_path, monkeypatch, runner):
    sam = person.info()["fingerprint"]
    agent_home(tmp_path, monkeypatch)
    kp = person.key_path()
    d = json.loads(kp.read_text())
    d["endorsements"] = [{"old": sam, "old_public": "x", "new_public": d["public"], "sig": "AAAA"}]
    kp.write_text(json.dumps(d))
    signer = person.unlock(AGENT_PASS)
    assert person.endorsed(sam, None, signer.endorsements, signer.public) is False
    s = study.open(ws / "demo")
    with personkey.signing(signer), pytest.raises(StudyError, match="never endorsed"):
        s.holdout("forged", person=True)
    with pytest.raises(person.PinRefused):
        person.pin_workspace(ws, signer)


def test_verify_fails_a_study_signed_by_a_key_the_workspace_never_pinned(tmp_path, person_key, monkeypatch):
    root = tmp_path / "w"
    root.mkdir()
    (root / "openlab.toml").write_text("[agents]\n")
    s = study.new(root / "s", "temporal")
    from nacl.signing import SigningKey
    other = personkey.Signer(SigningKey.generate(), "mallory")
    with personkey.signing(other):                                  # before any workspace pin: TOFU on its key
        s.record._pin(other)
    with person_key.act() as signer:
        person.pin_workspace(root, signer)
    v = s.verify()
    assert not v["ok"] and any("but the workspace pins" in x["why"] for x in v["signatures"]["invalid"])


@pytest.mark.parametrize("cmd", [
    "HOME=/tmp/ah openlab person init --handle sam",
    "HOME=/tmp/ah openlab person pin",
    "openlab person pin",
    "openlab person rotate",
    "openlab person act box -- study holdout demo --reason final",
    "python -m openlab person init",
    "cd /w && openlab person pin demo",
    "env HOME=/tmp/ah openlab study status demo",
    "export HOME=/tmp/ah; openlab study run demo",
    "XDG_CONFIG_HOME=/tmp/x openlab study run demo",
    "OPENLAB_WORKSPACE=/tmp/other openlab study run demo",
    "python -c 'from openlab.agents import person; person.pin_workspace(\".\", s)'",
])
def test_the_hooks_deny_person_key_commands_and_env_overrides(cmd):
    assert hooks.gate("Bash", {"command": cmd}), cmd


@pytest.mark.parametrize("cmd", ["openlab person show", "openlab study run demo", "echo $HOME", "ls ~",
                                 "cd $HOME/project && openlab study status demo"])
def test_reading_the_pin_and_plain_commands_pass(cmd):
    assert hooks.gate("Bash", {"command": cmd}) is None, cmd


def test_a_denied_person_command_is_a_bypass_attempt(tmp_path):
    (tmp_path / "openlab.toml").write_text("")
    out = hooks.handle("PreToolUse", {"session_id": "sess-1", "cwd": str(tmp_path), "tool_name": "Bash",
                                      "tool_input": {"command": "HOME=/tmp/ah openlab person pin"}})
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    ev = Store(tmp_path).events(None, n=10, kinds=("bypass_attempt",))
    assert len(ev) == 1 and "person" in ev[0]["summary"]


def test_a_study_sealed_under_the_workspace_key_refuses_another_after_the_pin_stores_are_wiped(ws, monkeypatch):
    """A residual bypass: with hooks off, an agent wipes both pin stores (the control store's chain and
    openlab.toml) and pins its own key on a study no person has signed yet. The plan names the key the study was
    sealed under, so the first pin must be that key: refused when written, and flagged by verify when forced in."""
    s = study.open(ws / "demo")
    sealed = s.record.find("plan")[0].payload["person_key"]
    assert sealed["fingerprint"] == person.workspace_pin(ws)
    (ws / ".openlab" / "control.db").unlink()
    (ws / "openlab.toml").write_text("[agents]\nbudget_usd = 1.0\n")
    assert person.workspace_pin(ws) is None
    from nacl.signing import SigningKey
    mallory = personkey.Signer(SigningKey.generate(), "mallory")
    with personkey.signing(mallory), pytest.raises(StudyError, match="sealed under the person key"):
        s.record._pin(mallory)
    assert not s.record.find("person_key")
    from openlab.study import schema
    monkeypatch.setattr(schema.PersonKey, "check", lambda self, rec, actor, links: None)
    with personkey.signing(mallory):                                # forced in past the schema (a raw write)
        s.record.add(schema.PersonKey(fingerprint=mallory.fingerprint, public=mallory.public, handle="mallory"),
                     actor="person")
    v = s.verify()
    assert not v["ok"] and any("sealed under" in x["why"] for x in v["signatures"]["invalid"])
