"""The harness as data (openlab.harness): the texts openlab briefs agents with are files, checked before use, and a
spawn records which harness briefed it. Nothing here calls a model."""

import json
import shutil

import pytest

from openlab import harness as H


def test_the_packaged_harness_is_valid_and_its_texts_are_the_ones_in_use():
    from openlab import agent_guide, mcp_server
    from openlab.agents import claude_setup, spawn
    from openlab.study import paperplan, proposal
    h = H.read()
    assert H.validate(h) == [] and h.sha == H.load().sha and h.skills == ()
    assert spawn.BRIEF == h.text("brief.md") and spawn.SELF == h.text("self.md")
    assert claude_setup.COMMON == h.text("common.md") and agent_guide.BRIEF == h.text("guide.md")
    assert list(claude_setup.ROLES) == list(H.ROLES) and claude_setup.ROLES == h.roles
    assert proposal.LEAD == h.text("workflows/propose_lead.md")
    assert paperplan.BUILD == h.text("workflows/replicate_build.md")
    assert mcp_server.SHORT == h.short and not h.text("brief.md").endswith("\n")


def test_every_openlab_role_is_briefed_with_its_prompt(tmp_path):
    from openlab.agents.spawn import brief, spawn
    (tmp_path / "openlab.toml").write_text("")
    for role, (_, body) in H.load().roles.items():
        assert body in brief(spawn(tmp_path, "x", role=role, start=False)), role
    planner = brief(spawn(tmp_path, "x", role="planner", start=False))
    assert not any(body in planner for _, body in H.load().roles.values())


def test_a_spawn_records_the_harness_that_briefed_it(tmp_path):
    from openlab.agents.spawn import HARNESS, spawn
    from openlab.agents.store import Store
    (tmp_path / "openlab.toml").write_text("")
    st = Store(tmp_path)

    def told(a):
        launch = json.loads((st.agent_dir(a["id"]) / "launch.json").read_text())
        ev = next(e for e in st.events(a["id"], n=10) if e["kind"] == "spawn")
        assert launch["harness"] == ev["payload"]["harness"] == HARNESS.sha
        assert launch["harness_sha"] == ev["payload"]["harness_sha"]
        return launch["harness_sha"]
    one = told(spawn(tmp_path, "does x predict y?", role="screener", start=False))
    # the same role, tools and workspace: the same harness, whatever the task, id or budget
    assert told(spawn(tmp_path, "another task", role="screener", start=False, budget={"usd": 0.3})) == one
    assert told(spawn(tmp_path, "x", role="reviewer", start=False)) != one
    assert told(spawn(tmp_path, "x", role="screener", start=False, decide="self")) != one
    assert told(spawn(tmp_path, "x", role="screener", start=False, allowed_tools=["Read"])) != one
    (tmp_path / "OPENLAB.md").write_text("Use polars, never pandas.")
    assert told(spawn(tmp_path, "x", role="screener", start=False)) != one


@pytest.fixture
def copy(tmp_path):
    d = tmp_path / "h"
    shutil.copytree(H.DEFAULT, d, ignore=shutil.ignore_patterns("__pycache__"))
    return d


def _problems(d):
    return "\n".join(H.validate(H.read(d)))


def test_a_copy_has_the_default_fingerprint_until_a_file_changes(copy):
    assert H.read(copy).sha == H.read().sha and _problems(copy) == ""
    (copy / "common.md").write_text((copy / "common.md").read_text() + "- Be brief.\n")
    assert H.read(copy).sha != H.read().sha and _problems(copy) == ""


def test_a_harness_holds_advice_only(copy):
    (copy / "hooks.json").write_text("{}")
    (copy / "skills" / "walk").mkdir(parents=True)
    (copy / "skills" / "walk" / "run.py").write_text("print(1)")
    out = _problems(copy)
    assert "hooks.json is not part of a harness" in out and "skills/walk/run.py is not part of a harness" in out
    (copy / "roles" / "reviewer.md").unlink()
    assert "roles/reviewer.md is missing" in _problems(copy)


def test_a_text_must_fill_the_fields_openlab_gives_it(copy):
    b = copy / "brief.md"
    b.write_text(b.read_text().replace("{budget_line}", "{budget}"))
    assert "brief.md uses the fields" in _problems(copy)
    b.write_text(b.read_text().replace("{budget}", "{budget_line} {"))
    assert "brief.md:" in _problems(copy)                           # an unpaired brace


def test_role_files_text_caps_and_benchmark_words(copy):
    (copy / "roles" / "lead.md").write_text("You are the lead.\n")
    (copy / "brief.md").write_text((copy / "brief.md").read_text() + "x" * H.ALWAYS_ON_MAX)
    (copy / "common.md").write_text("Check the rubric before you finish.\n")
    (copy / "guide.md").write_text("Read aerial-cactus-identification's data.\n")
    out = _problems(copy)
    assert "roles/lead.md: a role file starts with ---" in out and f"over {H.ALWAYS_ON_MAX}" in out
    assert "common.md says 'rubric'" in out and "guide.md says 'aerial-cactus-identification'" in out


def test_a_skill_must_be_named_where_an_agent_always_reads(copy):
    (copy / "skills" / "walk-forward").mkdir(parents=True)
    (copy / "skills" / "walk-forward" / "SKILL.md").write_text("---\nname: walk-forward\n---\nSplit by time.\n")
    assert "no always-on text (brief.md, common.md) names the skill 'walk-forward'" in _problems(copy)
    (copy / "common.md").write_text((copy / "common.md").read_text() + "- Features over time: the walk-forward skill.\n")
    assert _problems(copy) == "" and H.read(copy).skills == ("walk-forward",)
