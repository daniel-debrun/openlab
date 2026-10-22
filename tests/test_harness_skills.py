"""A harness's skills (openlab.harness.skills): a skill is loaded by name through the harness_skill tool, its script
runs only sandboxed (no network, the study's record masked, writes only to its stage), and the critique track record
now reads comparisons and tests (openlab.study.critique.confirmed)."""

import shutil
from types import SimpleNamespace

import pytest

from harness_fake import candidate
from openlab import harness as H
from openlab.harness import versions as V

bwrap = pytest.mark.skipif(not shutil.which("bwrap"), reason="scripts run only under bwrap")


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "openlab.toml").write_text("")
    return tmp_path


def _skilled(ws, script="print('hello from the sandbox')"):
    c = candidate(ws, ws / "cand", extra="- To tidy a folder: the tidy skill (openlab_tool(\"harness_skill\", "
                                         "{\"name\": \"tidy\"})).")
    (c / "skills" / "tidy").mkdir(parents=True)
    (c / "skills" / "tidy" / "SKILL.md").write_text("---\nname: tidy\n---\nRun its script; read what it wrote.\n")
    (c / "skills" / "tidy" / "script.py").write_text(script + "\n")
    return V.install(ws, c)["sha"]


def test_a_skill_needs_a_trigger_and_its_script_must_compile(ws):
    c = candidate(ws, ws / "c2")
    (c / "skills" / "quux").mkdir(parents=True)
    (c / "skills" / "quux" / "script.py").write_text("def (:\n")
    out = "\n".join(H.validate(H.read(c)))
    assert "has no SKILL.md beside it" in out
    (c / "skills" / "quux" / "SKILL.md").write_text("quux\n")
    out = "\n".join(H.validate(H.read(c)))
    assert "does not compile" in out and "names the skill 'quux'" in out


@bwrap
def test_a_skill_script_runs_sandboxed_with_no_network_and_the_record_masked(ws, monkeypatch):
    from openlab.agents.spawn import spawn
    from openlab.app import ToolContext, dispatch
    probe = ("import os, socket, sys\nprint('args', sys.argv[1:])\nprint('record', os.path.exists("
             "os.path.join(STUDY, '.openlab', 'secret.txt')))\ntry:\n    socket.create_connection(('1.1.1.1', 53), 2)\n"
             "    print('net yes')\nexcept OSError:\n    print('net no')\nopen('out.txt', 'w').write('ok')\n")
    study = ws / "s"
    (study / ".openlab").mkdir(parents=True)
    (study / ".openlab" / "secret.txt").write_text("holdout")
    sha = _skilled(ws, f"STUDY = {str(study)!r}\n" + probe)
    a = spawn(ws, "x", role="experimenter", start=False, harness=sha)
    monkeypatch.setenv("OPENLAB_AGENT", a["id"])
    ctx = ToolContext(str(ws))
    sk = dispatch(ctx, "harness_skill", {"name": "tidy"}, agent=a["id"])
    assert sk["ok"] and sk["result"]["script"] and "Run its script" in sk["result"]["skill"]
    out = dispatch(ctx, "harness_script", {"name": "tidy", "args": ["--fast"], "study": "s"}, agent=a["id"])
    assert out["ok"], out
    r = out["result"]
    assert r["exit"] == 0 and r["sandbox"] == "bwrap" and r["files"] == ["out.txt"]
    assert "args ['--fast']" in r["stdout"] and "record False" in r["stdout"] and "net no" in r["stdout"]
    assert not dispatch(ctx, "harness_skill", {"name": "nope"}, agent=a["id"])["ok"]
    assert not dispatch(ctx, "harness_script", {"name": "tidy", "study": "../.."}, agent=a["id"])["ok"]


def test_comparisons_and_tests_settle_a_critique_finding():
    from openlab.study.critique import confirmed

    class Rec:
        def __init__(self, nodes, edges):
            self.nodes, self.edges = nodes, edges

        def get(self, id):
            return self.nodes[id]

        def links(self, id, rel=None, incoming=False):
            return [e for e in self.edges if (e[2] if incoming else e[0]) == id and (rel is None or e[1] == rel)]
    n = lambda id, kind, **p: SimpleNamespace(id=id, kind=kind, payload=p)                 # noqa: E731
    nodes = {"r1": n("r1", "result"), "r2": n("r2", "result"), "run1": n("run1", "run", audit={}),
             "cmpA": n("cmpA", "comparison", verdict="worse"), "cmpB": n("cmpB", "comparison", verdict="worse"),
             "cmpC": n("cmpC", "comparison", verdict="equal"), "t1": n("t1", "test", rejected=False),
             "t2": n("t2", "test", rejected=True), "claim": n("claim", "claim")}
    edges = [("r1", "of", "run1"), ("cmpA", "compares", "r1"), ("cmpA", "compares", "r2"),
             ("cmpB", "compares", "r2"), ("cmpB", "compares", "r1"), ("cmpC", "compares", "r1"),
             ("cmpC", "compares", "r2"), ("t1", "from", "r1"), ("t2", "from", "r1"), ("claim", "rests_on", "r1")]
    rec = Rec(nodes, edges)
    on_run = nodes["run1"]
    assert confirmed(rec, nodes["cmpA"], on_run) is True                 # its result came out worse
    assert confirmed(rec, nodes["cmpB"], on_run) is False                # the other side came out worse
    assert confirmed(rec, nodes["cmpC"], on_run) is None
    assert confirmed(rec, nodes["t1"], nodes["claim"]) is True and confirmed(rec, nodes["t2"], on_run) is False
    assert confirmed(rec, nodes["cmpA"], nodes["r2"]) is False and confirmed(rec, nodes["cmpA"], None) is None
