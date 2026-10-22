"""A refinement step (openlab.harness.refine) end to end with a stand-in runner (no model): reference, two scoped
editors (one refused once and retried), probes on copies, a composer's decision, delivery, a blind selector at k=2
and a reflector; the ledger is 4k rollouts and 3k improver sessions; a resumed step redoes nothing; the selected
harness is stored, never made active; starting is the person's act."""

import json
from pathlib import Path

import pytest

from harness_fake import run
from openlab import harness as H
from openlab.harness import refine as RF
from openlab.harness import versions as V


@pytest.fixture
def ws(tmp_path, monkeypatch):
    (tmp_path / "openlab.toml").write_text("")
    from openlab.agents import spawn as spawn_mod
    real = spawn_mod.spawn
    monkeypatch.setattr(spawn_mod, "spawn", lambda *a, **k: real(*a, **{**k, "start": False}))    # the reflector
    return tmp_path


class Runner:
    """Plays each role: executors write an answer, editors edit ./harness, the composer adopts inv, the selector
    picks B."""

    def __init__(self):
        self.calls = []

    def __call__(self, root, role, prompt, cwd, harness, usd, minutes, tag, study="", cli="claude"):
        self.calls.append(role)
        cwd = Path(cwd)
        h = cwd / "harness"
        if role == "agent":
            (cwd / "answer.txt").write_text(f"made under {harness[:12]}")
        elif role == "harness-editor-inv":
            (h / "common.md").write_text((h / "common.md").read_text() + "- Check the answer file exists.\n")
            (cwd / "notes.md").write_text("common.md: a check before finishing (reference [RESULT]).")
        elif role == "harness-editor-dom":
            if "previous edit was refused" not in prompt:
                (h / "brief.md").write_text((h / "brief.md").read_text().replace("{budget_line}", "{budget}"))
            else:
                (h / "brief.md").write_text(H.load().files["brief.md"] + "\n")
                (h / "roles" / "screener.md").write_text((h / "roles" / "screener.md").read_text() + "One more.\n")
            (cwd / "notes.md").write_text("brief.md: budget wording.")
        elif role == "composer":
            assert (cwd / "candidates" / "inv" / "diff.patch").read_text().count("Check the answer") == 1
            (cwd / "decision.json").write_text(json.dumps({"decision": "adopt", "candidate": "inv", "edits": [
                {"state": "kept", "edit": "common.md check", "evidence": "reference transcript, [RESULT]"},
                {"state": "dropped", "edit": "roles/screener.md", "evidence": "no moment shows it"}]}))
        elif role == "selector":
            assert (cwd / "options" / "A" / "answer.txt").exists()
            (cwd / "choice.json").write_text(json.dumps({"choice": "B", "why": "it checks"}))
        return run(root, role=role, harness=harness, cwd=str(cwd), tag=tag, prompt=prompt, calls=4,
                   refusals=0 if harness != "default" else 1, budget={"usd": usd, "minutes": minutes})


def test_a_step_is_four_rollouts_and_three_sessions_and_stores_but_never_activates(ws, person_key):
    src = ws / "task"
    src.mkdir()
    (src / "README.md").write_text("Write answer.txt.")
    runner = Runner()
    with pytest.raises(PermissionError, match="person's own terminal"):
        RF.start(ws, "task", src, "write the answer", runner=runner)
    with person_key.act():
        out = RF.start(ws, "task", src, "write the answer", runner=runner)
    led = out["ledger"]
    assert (led["executor_rollouts"], led["improver_sessions"], led["reflector"]) == (4, 4, 1)  # dom retried once
    assert runner.calls.count("harness-editor-dom") == 2 and runner.calls.count("agent") == 4
    assert out["decisions"] == ["adopt"] and out["final"] != V.default_sha() and V.active_sha(ws) == ""
    inv = [v for v in V.stored(ws) if "g0 inv" in v["note"]]
    assert inv and inv[0]["sha"] == out["final"]
    assert "Check the answer" in V.get(ws, out["final"]).files["common.md"]
    assert Path(out["deliverable"]).is_relative_to(ws / ".probes")
    assert (Path(out["deliverable"]) / "answer.txt").read_text() == f"made under {out['final'][:12]}"
    assert not (src / "answer.txt").exists()                           # the source is never written
    from openlab.harness.replay import nodes
    assert [n["kind"] for n in nodes(ws)] == ["reference", "probe", "probe", "deliver"]
    state = json.loads((V.home(ws) / "refine" / out["step"] / "state.json").read_text())
    assert {"g0-reference", "g0-propose-inv", "g0-propose-dom", "g0-compose", "g0-deliver", "reflect"} <= set(
        state["done"]) and len(state["sessions"]) == 3
    n = len(runner.calls)
    with person_key.act():
        again = RF.start(ws, "task", src, "write the answer", runner=runner, step=out["step"])
    assert len(runner.calls) == n and again["final"] == out["final"]       # resumed: nothing redone
    assert led["budgeted_usd"] <= RF.budget(ws)["usd"]


def test_at_k2_a_blind_selector_may_pick_a_group_or_abstain(ws, person_key):
    src = ws / "task"
    src.mkdir()
    (ws / "openlab.toml").write_text("[harness]\nrefine_budget_usd = 8.0\n[agents]\ncap_usd = 40.0\n")
    runner = Runner()
    with person_key.act():
        out = RF.start(ws, "task", src, "write the answer", k=2, runner=runner)
    assert out["ledger"]["executor_rollouts"] == 8 and out["ledger"]["selector"] == 1 and out["selected"] == "B"
    with person_key.act(), pytest.raises(H.HarnessError, match="k_max"):
        RF.start(ws, "task", src, "x", k=9, runner=runner)


def test_an_evidence_folder_showing_a_grade_is_never_read_by_a_model(ws, person_key):
    src = ws / "task"
    src.mkdir()
    (src / "notes.txt").write_text("last time: gold_medal true")
    runner = Runner()

    def leaky(root, role, *a, **k):
        if role == "agent":
            (Path(a[1]) / "results.txt").write_text("any_medal: true")
        return runner(root, role, *a, **k)
    with person_key.act(), pytest.raises(H.HarnessError, match="shows a grade"):
        RF.start(ws, "task", src, "x", runner=leaky)
    assert "harness-editor-inv" not in runner.calls
