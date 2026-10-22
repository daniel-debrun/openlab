"""The offline evaluator (openlab.harness.evaluate) and replay (openlab.harness.replay): fixed pools, a private pool
no model reads, a measured noise band, the gates (leaks, canaries, traps, floor, growth, edit cap, predicted flips,
out of distribution), the matched best-of-n comparison; replay screens policies with the incumbent always in."""

import json

import pytest

from openlab.harness import evaluate as E
from openlab.harness import replay as R

TRAPS = {"traps": {"caught": 2, "total": 2, "missed": [], "traps": [{"trap": "a", "caught": True},
                                                                   {"trap": "b", "caught": True}]}}


def rows(arm, medals_by_seed, tasks=("p1", "p2", "p3", "p4"), **kw):
    """One row per (seed, task): medals_by_seed[seed][i] for task i."""
    return [{"arm": arm, "seed": s, "task": t, "medal": bool(m), "valid": True, **kw}
            for s, ms in enumerate(medals_by_seed) for t, m in zip(tasks, ms)]


POOLS = {"dev": ["d1"], "private": ["p1", "p2", "p3", "p4"], "canary": ["c1"], "ood": []}
BASE = rows("openlab", [(1, 0, 0, 0), (1, 1, 0, 0), (0, 1, 0, 0)])           # 0.25, 0.5, 0.25


def test_pools_are_fixed_and_a_model_reads_only_the_development_pool(tmp_path):
    tasks = [f"t{i}" for i in range(40)]
    a, b = E.split(tasks), E.split(list(reversed(tasks)))
    assert a == b and 8 < len(a["private"]) < 32 and not set(a["dev"]) & set(a["private"])
    p = E.fixed_pools(tmp_path / "pools.json", tasks, canaries=["c1"])
    assert E.fixed_pools(tmp_path / "pools.json") == p and p["canary"] == ["c1"]
    view = E.dev_only([{"task": a["dev"][0], "arm": "x", "medal": True, "score": 0.9},
                       {"task": a["private"][0], "arm": "x", "medal": True}], p)
    assert view == [{"task": a["dev"][0], "arm": "x"}]
    (tmp_path / "ev").mkdir()
    (tmp_path / "ev" / "notes.md").write_text("the run's any_medal was true")
    assert E.tripwire(tmp_path / "ev") == [("notes.md", "any_medal")]


def test_a_candidate_is_accepted_only_through_every_gate():
    cand = rows("openlab@c", [(1, 1, 1, 0), (1, 1, 0, 0), (1, 1, 1, 0)])
    bo = rows("openlab*3", [(1, 1, 1, 0), (1, 1, 0, 0), (1, 1, 0, 0)])
    out = E.accept(BASE + cand + bo, "openlab", "openlab@c", POOLS, bestofn="openlab*3", traps=TRAPS)
    assert out["accepted"] and out["band"] == pytest.approx(2 * 0.0833, abs=1e-3) and out["delta"] > 0
    assert out["beats_bestofn"] is False and not out["refine_worth_it"]       # not by more than the band
    weak = rows("openlab*2", [(1, 0, 0, 0)] * 3)
    out = E.accept(BASE + cand + weak, "openlab", "openlab@c", POOLS, bestofn="openlab*2", traps=TRAPS)
    assert out["beats_bestofn"] and out["refine_worth_it"]
    no_traps = E.accept(BASE + cand, "openlab", "openlab@c", POOLS)
    assert not no_traps["accepted"] and "trap suite was not run" in no_traps["reasons"][0]
    one_seed = E.accept(rows("openlab", [(1, 0, 0, 0)]) + cand, "openlab", "openlab@c", POOLS, traps=TRAPS)
    assert any("measured, not assumed" in r for r in one_seed["reasons"])
    worse = rows("openlab@w", [(0, 0, 0, 0)] * 3)
    assert any("below the best so far" in r for r in E.accept(BASE + worse, "openlab", "openlab@w", POOLS,
                                                               traps=TRAPS)["reasons"])
    same = rows("openlab@s", [(1, 0, 0, 0), (1, 1, 0, 0), (0, 1, 0, 0)])
    grew = E.accept(BASE + same, "openlab", "openlab@s", POOLS, traps=TRAPS, growth=0.2)
    assert any("grew by 20%" in r for r in grew["reasons"])
    assert any("bundles 5 edits" in r for r in E.accept(BASE + cand, "openlab", "openlab@c", POOLS, traps=TRAPS,
                                                         edits=5)["reasons"])


def test_leaks_canaries_failed_predictions_and_a_fall_out_of_distribution_each_reject():
    cand = rows("openlab@c", [(1, 1, 1, 0), (1, 1, 0, 0), (1, 1, 1, 0)])
    leak = [dict(cand[0], leaks=[{"why": "a Kaggle download"}])] + cand[1:]
    assert "attempted a leak" in E.accept(BASE + leak, "openlab", "openlab@c", POOLS, traps=TRAPS)["reasons"][0]
    canary = cand + [{"arm": "openlab@c", "seed": 0, "task": "c1", "medal": True}]
    assert any("canary" in r for r in E.accept(BASE + canary, "openlab", "openlab@c", POOLS, traps=TRAPS)["reasons"])
    flips = E.accept(BASE + cand, "openlab", "openlab@c", POOLS, traps=TRAPS,
                     predicted={"tasks": {"p4": "up", "p1": "up", "p3": "down"}})
    assert flips["flips"] == {"p4": "failed", "p1": "held", "p3": "failed"} and not flips["accepted"]
    pools = {**POOLS, "ood": ["o1", "o2"]}
    ood = rows("openlab", [(1, 1)] * 3, tasks=("o1", "o2")) + rows("openlab@c", [(0, 0)] * 3, tasks=("o1", "o2"))
    out = E.accept(BASE + cand + ood, "openlab", "openlab@c", pools, traps=TRAPS)
    assert out["ood_delta"] == -1.0 and any("out of distribution" in r for r in out["reasons"])


def test_cost_per_accepted_edit_counts_the_rejected():
    assert E.cost_per_accepted([{"usd": 3, "accepted": False}, {"usd": 2, "accepted": True}]) == {
        "usd": 5, "proposals": 2, "accepted": 1, "per_accepted": 5.0}
    assert E.traps_passed({"traps": {"traps": [{"trap": "x", "caught": False}]}}) == (False, ["x"])


def test_replay_screens_policies_with_the_incumbent_always_in(tmp_path):
    (tmp_path / "openlab.toml").write_text("")
    for step, (ref, inv, dom) in {"s1": (0.5, 0.7, 0.4), "s2": (0.6, 0.5, 0.9)}.items():
        R.log(tmp_path, {"step": step, "kind": "reference", "harness": "base", "signals": {"refusal_rate": 0.3},
                         "score": ref})
        R.log(tmp_path, {"step": step, "kind": "probe", "harness": "inv", "signals": {"refusal_rate": 0.1},
                         "score": inv})
        R.log(tmp_path, {"step": step, "kind": "probe", "harness": "dom", "signals": {"refusal_rate": 0.5},
                         "score": dom})
    out = {r["policy"]: r for r in R.screen(R.nodes(tmp_path))}
    assert out["keep_base"]["value"] == 0.55 and out["by_signal:refusal_rate"]["value"] == 0.6
    assert R.screen(R.nodes(tmp_path))[0]["value"] >= out["keep_base"]["value"]
    assert R.score(tmp_path, "s1", "dom", 0.95) == 1
    with pytest.raises(ValueError):
        R.log(tmp_path, {"step": "s3", "kind": "guess"})
    json.dumps(R.nodes(tmp_path))
