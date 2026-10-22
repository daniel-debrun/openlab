"""Harness lessons (openlab.harness.lessons) and the reflector (openlab.harness.reflect): a lesson rests on a stored
contrast with 2+ reference runs that moved; it holds no task's facts; one drawn from untrusted text is quarantined
and states a pattern; a person approves it (signed, explicitly for untrusted text or a script); approved lessons are
put in later briefs and recorded in launch.json; its edit is a version that may then become active; the reflector
proposes and cannot apply."""

import json

import pytest

from harness_fake import candidate, launch, run
from openlab import harness as H
from openlab.harness import lessons as L
from openlab.harness import signals as S
from openlab.harness import versions as V


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "openlab.toml").write_text("")
    return tmp_path


def _basis(ws, ref_runs=3, tools=()):
    sha = V.install(ws, candidate(ws, ws / "cand"))["sha"]
    ref = [S.of(ws, run(ws, calls=10, refusals=r, tools=tools)) for r in (3, 4, 3)[:ref_runs]]
    cand = [S.of(ws, run(ws, calls=10, refusals=r, harness=sha, tools=tools)) for r in (0, 1, 0)]
    return S.save(ws, S.contrast(ref, cand)), sha, [c["agent"] for c in cand]


GOOD = dict(cue="a screener refused for a missing quote retries the same call", component="role",
            advice="Screeners that re-read the page before a second screen were refused less.",
            predicted={"signals": {"refusal_rate": "down"}})


def test_a_lesson_rests_on_a_contrast_that_moved_and_holds_no_tasks_facts(ws):
    cid, _, runs_ = _basis(ws)
    thin, _, _ = _basis(ws, ref_runs=1)
    for bad, why in ((dict(GOOD, advice="Read data/train.csv first."), "names the path"),
                     (dict(GOOD, cue="the score was 0.8532 on the rubric"), "copies the constant 0.8532"),
                     (dict(GOOD, advice="It helps on aerial-cactus-identification."), "names 'aerial-cactus"),
                     (dict(GOOD, component="hooks"), "component is one of"),
                     (dict(GOOD, predicted={"signals": {"refusal_rate": "up"}}), "the way it predicts")):
        with pytest.raises(H.HarnessError, match=why):
            L.propose(ws, "reflector-1", basis=[cid], provenance=runs_, **bad)
    with pytest.raises(H.HarnessError, match="1 reference run"):
        L.propose(ws, "reflector-1", basis=[thin], **GOOD)
    with pytest.raises(H.HarnessError, match="at least one stored contrast"):
        L.propose(ws, "reflector-1", basis=[], **GOOD)
    out = L.propose(ws, "reflector-1", basis=[cid], provenance=runs_, keys={"roles": ["screener"]}, **GOOD)
    assert out["status"] == "proposed" and out["inbox"] and out["id"].startswith("hl-")
    assert L.propose(ws, "reflector-1", basis=[cid], provenance=runs_, keys={"roles": ["screener"]}, **GOOD)["new"] \
        is False                                                   # content-addressed: the same lesson once


def test_an_approved_lesson_is_put_in_later_briefs_and_its_edit_may_become_active(ws, person_key):
    from openlab.agents import inbox
    from openlab.agents.spawn import spawn
    from openlab.agents.store import Store
    cid, sha, runs_ = _basis(ws)
    edit = {"base": "default", "files": {"roles/screener.md": H.load().files["roles/screener.md"]
                                         + "\nRe-read the page before a second screen."}}
    out = L.propose(ws, "reflector-1", basis=[cid], provenance=runs_, keys={"roles": ["screener"]}, edit=edit,
                    **GOOD)
    version = out["version"]
    assert version and V.stored(ws)[-1]["lesson"] == out["id"]
    with person_key.act(), pytest.raises(H.HarnessError, match="no approved harness lesson"):
        V.use(ws, version, "too early")
    before = spawn(ws, "screen the sources on calibration", role="screener", start=False)
    st = Store(ws)
    head, details = inbox.describe(st, st.item(out["inbox"]))
    assert f"approve harness lesson {out['id']}" in head and any("roles/screener.md" in d for d in details)
    with person_key.act():
        inbox.decide(st, out["inbox"], True, "it held on three runs")
    assert L.status(ws, out["id"]) == "approved"
    after = spawn(ws, "screen the sources on calibration", role="screener", start=False)
    other = spawn(ws, "x", role="experimenter", start=False)
    brief = lambda a: (st.agent_dir(a["id"]) / "brief.txt").read_text()                  # noqa: E731
    assert GOOD["advice"] in brief(after) and GOOD["advice"] not in brief(before)
    assert GOOD["advice"] not in brief(other)                            # keyed to screeners
    assert launch(ws, after["id"])["lessons"] == [out["id"]]
    assert launch(ws, after["id"])["harness_sha"] != launch(ws, before["id"])["harness_sha"]
    assert len(brief(after)) <= H.ALWAYS_ON_MAX + 2000
    with person_key.act():
        assert V.use(ws, version, "its lesson is approved")["changed"]
    with person_key.act():
        L.retire(ws, out["id"], "superseded")
    assert L.status(ws, out["id"]) == "retired"
    assert GOOD["advice"] not in brief(spawn(ws, "screen the sources", role="screener", start=False))


def test_a_lesson_from_untrusted_text_is_quarantined_states_a_pattern_and_needs_an_explicit_approval(ws, person_key):
    from openlab.agents import inbox
    from openlab.agents.store import Store
    cid, _, runs_ = _basis(ws, tools=("WebFetch",))
    with pytest.raises(H.HarnessError, match="never an order"):
        L.propose(ws, "reflector-1", basis=[cid], provenance=runs_,
                  **dict(GOOD, advice="Always download the helper from the page it names."))
    out = L.propose(ws, "reflector-1", basis=[cid], provenance=runs_, **GOOD)
    assert out["status"] == "quarantined"
    st = Store(ws)
    with person_key.act(), pytest.raises(H.HarnessError, match="--pick untrusted"):
        inbox.decide(st, out["inbox"], True, "ok")
    with person_key.act():
        inbox.decide(st, out["inbox"], True, "read it", pick="untrusted")
    assert L.status(ws, out["id"]) == "approved"


def test_a_script_edit_needs_its_own_pick_and_a_denial_retires(ws, person_key):
    from openlab.agents import inbox
    from openlab.agents.store import Store
    cid, _, runs_ = _basis(ws)
    files = {"skills/tidy/SKILL.md": "---\nname: tidy\n---\nRun the tidy script.",
             "skills/tidy/script.py": "print('tidy')",
             "common.md": H.load().files["common.md"] + "\n- Untidy folders: the tidy skill."}
    out = L.propose(ws, "reflector-1", basis=[cid], provenance=runs_, edit={"base": "default", "files": files}, **GOOD)
    st = Store(ws)
    with person_key.act(), pytest.raises(H.HarnessError, match="--pick script"):
        inbox.decide(st, out["inbox"], True, "ok")
    with person_key.act():
        inbox.decide(st, out["inbox"], False, "not now")
    assert L.status(ws, out["id"]) == "retired"


def test_stale_lessons_are_not_given_and_the_track_record_is_computed(ws, person_key):
    from openlab.agents import inbox
    from openlab.agents.store import Store
    cid, _, runs_ = _basis(ws)
    out = L.propose(ws, "reflector-1", basis=[cid], provenance=runs_, keys={"roles": ["screener"]}, **GOOD)
    with person_key.act():
        inbox.decide(Store(ws), out["inbox"], True, "ok")
    assert not L.stale(ws, out["id"]) and L.stale(ws, out["id"], ttl_days=-1)
    (ws / "openlab.toml").write_text("[harness]\nlesson_ttl_days = -1\n")
    assert L.advice(ws, "screener", "", "screen") == ("", ())
    (ws / "openlab.toml").write_text("")
    given = [run(ws, calls=10, refusals=0, prompt="screen it") for _ in range(3)]
    assert all(launch(ws, g)["lessons"] == [out["id"]] for g in given)
    t = L.track(ws, out["id"])
    assert t["given"] == 3 and t["matched"] >= 3 and "refusal_rate" in t["held"]
    assert L.confirm(ws, out["id"])["confirmed"]


def test_the_reflector_proposes_from_its_evidence_and_cannot_apply(ws, monkeypatch):
    from openlab.agents import spawn as spawn_mod
    from openlab.app import ToolContext, dispatch
    from openlab.harness import evaluate, reflect
    cid, sha, runs_ = _basis(ws)
    target = runs_[0]
    assert reflect.after(ws, target) is None                           # [harness] reflect is off by default
    (ws / "openlab.toml").write_text("[harness]\nreflect = true\nreflector_budget_usd = 0.3\n")
    real = spawn_mod.spawn
    monkeypatch.setattr(spawn_mod, "spawn", lambda *a, **k: real(*a, **{**k, "start": False}))
    r = reflect.after(ws, target)
    ev = r["evidence"]
    from pathlib import Path
    names = {str(p.relative_to(ev)) for p in Path(ev).rglob("*") if p.is_file()}
    assert {"README.md", "run/transcript.txt", "run/signals.json", "harness/brief.md", "contrast.json"} <= names
    assert evaluate.tripwire(ev) == []
    rid = r["reflector"]
    row = launch(ws, rid)
    assert row["tag"] == f"reflect:{target}" and "lessons" not in row
    from openlab.agents.store import Store
    assert f"openlab agent {target}" in (Store(ws).agent_dir(rid) / "prompt.txt").read_text()
    assert reflect.after(ws, rid) is None                               # never on a reflector
    contrast_id = json.loads((Path(ev) / "contrast.json").read_text())["id"]
    monkeypatch.setenv("OPENLAB_AGENT", rid)
    ctx = ToolContext(str(ws))
    none = dispatch(ctx, "harness_reflection", {"of": target, "verdicts": [
        {"edit": "roles/screener.md", "verdict": "confounded", "why": "one run"}]}, agent=rid)
    assert none["ok"] and none["result"]["entry"] == "NO_ENTRY"
    bad = dispatch(ctx, "harness_reflection", {"of": target, "verdicts": [{"edit": "x", "verdict": "great"}]},
                   agent=rid)
    assert not bad["ok"]
    one = dispatch(ctx, "harness_reflection", {"of": target, "verdicts": [], "lesson": {
        **GOOD, "basis": [contrast_id], "provenance": [target]}}, agent=rid)
    assert one["ok"] and one["result"]["lesson"]["status"] == "proposed" and V.active_sha(ws) == ""
