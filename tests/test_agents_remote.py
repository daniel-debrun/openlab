"""Guarantees of person acts signed on the person's machine for a record on another (openlab.agents.remote): the box
runs the act with no person key; every signature it needs is asked over the channel and made on the person's
machine, which signs only what a person act signs (never a key rotation) and shows each one; the record on the box
then verifies against the person's key, which never left their machine. The box here is a local process with a home
of its own (no key) behind a stand-in for ssh."""

import shutil
import sys
from pathlib import Path

import pytest

from conftest import PASSPHRASE
from openlab import study
from openlab.agents import person, remote
from openlab.study import personkey

ROOT = str(Path(__file__).resolve().parents[1])


def box_transport(home: Path) -> list:
    """`ssh -T HOST` for a box that is this machine under another HOME (no person key there)."""
    return ["env", f"HOME={home}", f"PYTHONPATH={ROOT}", "OPENLAB_MODULE_CACHE=off", "sh", "-c"]


@pytest.fixture
def box(tmp_path, person_key):
    """A workspace on the 'box' and its home, with no person key; the person's key is unlocked here."""
    home, root = tmp_path / "boxhome", tmp_path / "boxws"
    home.mkdir()
    root.mkdir()
    (root / "openlab.toml").write_text("[agents]\n")
    return {"home": home, "root": root, "unlock": lambda summary, details: person.unlock(PASSPHRASE, summary)}


def act(box, *argv, out=None):
    return remote.act_on("box", ["-m", "openlab", *argv], openlab=sys.executable, cd=str(box["root"]),
                         transport=box_transport(box["home"]), unlock=box["unlock"])


def test_what_the_persons_machine_signs_and_refuses():
    node = personkey.message("look", '{"n":1}', [["of", "plan:x"]], "person:sam", "ab12")
    assert remote.describe(node) == 'a record node: look {"n":1}'
    assert remote.describe(person.tell_message("lead-1", "stop at 3 runs")).startswith("a direction to an agent")
    assert remote.describe(personkey.rotation_message("ed25519:aa", "PUB")) is None      # key changes stay local
    assert remote.describe(b"anything else") is None


def test_the_box_pins_its_workspace_to_the_key_on_the_persons_machine(box, capsys):
    assert act(box, "person", "pin") == 0
    out = capsys.readouterr().out
    assert "signed on this machine: the workspace's pin" in out
    state = person.pin_state(box["root"])
    assert state["pin"] == person.info()["fingerprint"] and not state["conflict"]
    assert not (box["home"] / ".config" / "openlab").exists() or not any((box["home"] / ".config").rglob("*.key*"))


def test_a_request_for_a_key_rotation_is_refused(box, capsys):
    script = ("import sys; sys.path.insert(0, %r); from openlab.agents import remote; from openlab.study import personkey\n"
              "s = remote.hello('rotate the key')\n"
              "try:\n    s.sign(personkey.rotation_message(s.fingerprint, 'NEWPUB')); print('SIGNED')\n"
              "except Exception as e:\n    print('REFUSED', e)\n" % ROOT)
    rc = remote.act_on("box", ["-c", script], openlab=sys.executable, transport=box_transport(box["home"]),
                       unlock=box["unlock"])
    out = capsys.readouterr().out
    assert rc == 0 and "REFUSED" in out and "key changes stay local" in out and "SIGNED" not in out


@pytest.mark.skipif(not shutil.which("bwrap"), reason="runs are tested under bwrap")
def test_a_holdout_look_on_the_box_is_signed_on_the_persons_machine(box, capsys):          # ~20 s
    assert act(box, "person", "pin") == 0
    s = study.new(box["root"] / "demo", "temporal")
    s.run(note="research")                                          # an agent's run, on the box
    assert act(box, "study", "holdout", "demo", "--reason", "final") == 0
    out = capsys.readouterr().out
    assert "signed on this machine: a record node: look" in out
    s = study.open(box["root"] / "demo")
    look = s.record.find("look")[0]
    assert look.payload["sig"]["key"] == person.info()["fingerprint"]
    assert s.record.find("run", segment="holdout") and s.verify()["ok"]


def test_the_cli_passes_its_own_options_and_the_whole_remote_command(monkeypatch):
    from openlab.agents import cli
    got = {}
    monkeypatch.setattr(remote, "act_on", lambda host, args, openlab, cd: got.update(host=host, args=args,
                                                                                     openlab=openlab, cd=cd) or 0)
    assert cli.person_main(["act", "gpu-box", "--openlab", "/root/openlab/.venv/bin/openlab", "--cd", "/root/w", "--",
                            "study", "holdout", "gpu", "--reason", "final"]) == 0
    assert got == {"host": "gpu-box", "args": ["study", "holdout", "gpu", "--reason", "final"],
                   "openlab": "/root/openlab/.venv/bin/openlab", "cd": "/root/w"}
    cli.person_main(["act", "box", "--", "person", "pin"])
    assert got["args"] == ["person", "pin"] and got["openlab"] == "openlab" and got["cd"] == ""
