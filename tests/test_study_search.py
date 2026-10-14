"""Guarantees of the search, the selection rule and lessons (openlab.study.search, openlab.study.lessons): the tree is
scored by the documented formula on the study's clock and offered to the frontier as expand / fuse / draft / select
items; a sealed rule picks the final candidate, its record derived and not typed, and the holdout look is on that
run only; a lesson's direction must follow from the comparison it rests on, its track record is computed from the
runs that cite it, and the workspace's lessons are found by what one is about to try."""

import math
import shutil

import pytest

from openlab import study as vs
from openlab.study import search
from openlab.study.errors import StudyError

SHA = "a" * 64


def sealed_study(tmp_path, extra: str = "", name: str = "s"):
    s = vs.new(tmp_path / name, "temporal")
    toml = s.folder / "study.toml"
    toml.write_text(toml.read_text() + extra)
    s.materialise()
    s.seal()
    return vs.open(s.folder)


def put(s, mae: float, parent: str = "", op: str = "", seconds: float = 60.0, seed=None, code: str = SHA,
        verdict: str = "clean"):
    """A research run and its result written straight into the record; returns the run id."""
    rec = s.record
    plan, ds = rec.last("plan"), rec.last("dataset")
    links = [("under", plan.id), ("uses", ds.id)] + ([("branches_from", parent)] if parent else [])
    run = rec.add("run", {"segment": "research", "key": f"k{len(rec.find('run'))}", "cached": False,
                          "spec_sha": plan.payload["sha"], "code": {"model.py": code},
                          "data": {"panel": ds.payload["sha256"]}, "env": {"python": "3.11"}, "sandbox": "stub",
                          "seconds": seconds, "op": op, "seed": seed,
                          "audit": {"verdict": verdict, "checked": 2 if verdict == "clean" else 0,
                                    "mismatches": [] if verdict != "look-ahead" else [{"t": "x", "why": "y"}]}},
                  links=links)
    rec.add("result", {"metrics": {"mae": {"value": mae, "lo": mae - 0.02, "hi": mae + 0.02}}}, links=[("of", run.id)])
    return run.id


def test_the_tree_is_scored_by_the_formula_and_offered_to_the_frontier(tmp_path):
    s = sealed_study(tmp_path)
    r1 = put(s, 0.50, op="draft")
    r2 = put(s, 0.45, r1, "improve")
    put(s, 0.46, r2, "improve")
    r4 = put(s, 0.44, r2, "improve")                                  # the best (mae: lower is better)
    sc = search.score(s)
    assert sc["t"] == pytest.approx(4 / 20) and sc["c"] == pytest.approx(math.sqrt(2)) and sc["w"] == 1.0
    rows = {r["id"]: r for r in sc["rows"]}
    assert rows[r4]["G"] == 1.0 and rows[r1]["G"] == 0.0 and rows[r2]["n"] == 3
    N = 4
    for r in sc["rows"]:
        assert r["U"] == pytest.approx(math.sqrt(math.log(N + 1) / r["n"]))
        assert r["uct"] == pytest.approx((r["Q"] + sc["c"] * r["U"]) * r["cost"])
    top = max(sc["rows"], key=lambda r: r["uct"])
    assert top["P"] == 1.0 and sum(r["P"] for r in sc["rows"]) == pytest.approx(1.0)       # w = 1: the UCT pick
    assert sc["best"] == r4
    items = [i for i in s.frontier(n=None)["items"] if i["kind"] == "expand"]
    assert items and items[0]["target"] == top["id"] and f"--parent {top['id']}" in items[0]["why"]


def test_stagnation_offers_a_fuse_and_a_draft_and_late_in_the_budget_a_select(tmp_path):
    s = sealed_study(tmp_path)
    a = put(s, 0.40, op="draft")
    b = put(s, 0.60, op="draft")                                      # a second line
    x = a
    for v in (0.41, 0.42, 0.43, 0.44, 0.45, 0.46):                  # six runs, none better than 0.40
        x = put(s, v, x, "improve")
    sc = search.score(s)
    assert sc["since_best"] >= 6 and sc["root"][x] == a and a in sc["stagnant"]
    kinds = {i["kind"]: i for i in s.frontier(n=None)["items"]}
    assert "fuse" in kinds and f"--ref {b}" in kinds["fuse"]["why"] and "draft" in kinds
    for _ in range(10):                                              # 18 of 20 trials: past consolidate (0.85)
        put(s, 0.5, a, "improve")
    kinds = {i["kind"] for i in s.frontier(n=None)["items"]}
    assert "select" in kinds and "draft" not in kinds
    assert search.w_of(0.9, search.settings(s.spec)) == pytest.approx(0.2)


def test_the_clock_reads_declared_minutes(tmp_path):
    s = sealed_study(tmp_path, "\n[search]\nminutes = 10\n")
    put(s, 0.5, op="draft", seconds=300)
    assert search.clock(s) == pytest.approx(0.5)                     # 5 of 10 minutes beats 1 of 20 trials


def test_a_sealed_rule_picks_the_candidate_and_its_record_cannot_be_typed(tmp_path):
    s = sealed_study(tmp_path, '\n[selection]\nrule = "best_val"\n')
    put(s, 0.50, op="draft")
    best = put(s, 0.40, op="draft")
    put(s, 0.45, op="draft")
    out = s.select()
    sel = s.record.get(out["id"]).payload
    assert sel["picked"] == best and sel["n"] == 3 and sel["value"] == 0.40
    assert sel["adjusted"] == pytest.approx(0.40 + (0.04 / (2 * 1.96)) * search.expected_max(3))   # moved toward worse
    from openlab.study.schema import Selection
    with pytest.raises(StudyError, match="does not follow from the record"):
        s.record.add(Selection(**{**sel, "value": 0.30}), actor="agent",
                     links=[("picks", best), ("under", s.record.last("plan").id)])


def test_seed_robust_prefers_the_configuration_with_the_best_mean_over_seeds(tmp_path):
    s = sealed_study(tmp_path, '\n[selection]\nrule = "seed_robust"\nk = 3\n')
    lucky = [put(s, v, op="draft", seed=i, code="b" * 64) for i, v in enumerate((0.30, 0.60, 0.60))]   # mean 0.50
    steady = [put(s, v, op="draft", seed=i, code="c" * 64) for i, v in enumerate((0.42, 0.43, 0.44))]  # mean 0.43
    out = search.select(s)
    assert out["picked"] in steady and out["value"] == pytest.approx(0.43) and out["n"] == 2
    assert search.select(s)["picked"] not in lucky
    assert search.expected_max(1) == 0 and search.expected_max(2) == pytest.approx(1 / math.sqrt(math.pi), abs=1e-3)
    assert search.expected_max(3) == pytest.approx(0.8463, abs=1e-3)


def test_without_a_rule_select_is_refused_and_with_one_the_look_is_on_the_selected_run(tmp_path, person_key):
    s = sealed_study(tmp_path)
    put(s, 0.5, op="draft")
    with pytest.raises(StudyError, match="declares no selection rule"):
        s.select()
    s = sealed_study(tmp_path, '\n[selection]\nrule = "best_val"\n', name="r")
    put(s, 0.5, op="draft")                                          # its code is SHA, not the folder's model.py
    with person_key.act(), pytest.raises(StudyError, match="nothing has been selected"):
        s.holdout("final", person=True)
    s.select()
    with person_key.act(), pytest.raises(StudyError, match="is not run:"):
        s.holdout("final", person=True)
    assert not s.record.find("look")                                 # refused before the look was taken


@pytest.mark.skipif(not shutil.which("bwrap"), reason="runs are tested under bwrap")
def test_a_lesson_follows_from_its_comparison_is_cited_and_found_across_the_workspace(tmp_path):   # ~25 s
    s = vs.new(tmp_path / "w" / "demo", "temporal")
    (s.folder / "study.toml").write_text((s.folder / "study.toml").read_text() + "\n[audit]\nn = 1\n")
    (tmp_path / "w" / "openlab.toml").write_text("[agents]\n")
    base = s.run(note="both features", op="draft")
    model = s.folder / "model.py"
    model.write_text(model.read_text().replace('COLUMNS = ["x", "y_now"]', 'COLUMNS = ["x"]'))
    run1 = s.record.links(base["id"], rel="of")[0][2]
    variant = s.run(note="x only", op="improve", parent=run1)
    cmp = s.compare(variant["id"], base["id"], "mae")
    pd = s.record.get(cmp["id"]).payload["paired"]
    truth = ("improves" if pd["better"] == variant["id"] else "hurts") if pd["lo"] > 0 or pd["hi"] < 0 else None
    wrong = "improves" if truth == "hurts" else "hurts"
    with pytest.raises(StudyError, match=f"does not show `{wrong}`"):
        s.lesson("dropping y_now changes the error", wrong, cmp["id"], ["features"])
    assert truth is not None                                         # the template's y_now matters, clearly
    lesson = s.lesson("keep the current value y_now as a feature for this panel", truth, cmp["id"],
                      ["features", "lagged-target"])
    shown = s.record.get(lesson["id"]).payload["shown"]
    assert shown["lo"] == pd["lo"] and shown["hi"] == pd["hi"]
    model.write_text(model.read_text().replace('COLUMNS = ["x"]', 'COLUMNS = ["y_now"]'))     # a new variant
    with pytest.raises(StudyError, match="no lesson `lesson:nope`"):
        s.run(note="cites nothing real", op="improve", parent=run1, cites=["lesson:nope"])
    from openlab.study.lessons import find, track
    assert track(s, lesson["id"])["cited"] == 0
    s.run(note="y_now only", op="improve", parent=run1, cites=[lesson["id"]])
    t = track(s, lesson["id"])
    assert t["cited"] == 1 and t["runs"][0]["held"] in (True, False)
    hits = find(tmp_path / "w", "lagged target feature")
    assert hits and hits[0]["id"] == lesson["id"] and hits[0]["study"] == "demo" and hits[0]["cite"].startswith("demo/")
    assert find(tmp_path / "w", "transformer attention") == []
