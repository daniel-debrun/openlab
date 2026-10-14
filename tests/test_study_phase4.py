"""Guarantees of module checks, ensembles and critiques (openlab.study.preflight, .ensemble, .critique): a check
runs the runner's own child job on a little data and records a module_check keyed by the code, a run of code whose
latest check is red is refused, and the lint and determinism checks catch what they name; an ensemble of recorded
runs is a counted, cached run with its members as refs, weighted only by cross-fitting; a critique is advice whose
findings are answered and whose track record is computed, and nothing reads it into a verdict."""

import shutil

import pytest

from openlab import study as vs
from openlab.study import critique, preflight
from openlab.study.errors import StudyError

bwrap = pytest.mark.skipif(not shutil.which("bwrap"), reason="runs are tested under bwrap")


def temporal(tmp_path, name="t"):
    s = vs.new(tmp_path / name, "temporal")
    (s.folder / "study.toml").write_text((s.folder / "study.toml").read_text() + "\n[audit]\nn = 1\n")
    s.materialise()
    return s


def edit_model(s, old, new):
    p = s.folder / "model.py"
    text = p.read_text()
    assert old in text
    p.write_text(text.replace(old, new))


def test_the_lint_names_what_a_run_must_not_touch(tmp_path):
    (tmp_path / "model.py").write_text('import requests\nP = "../other/data.parquet"\nV = open(".openlab/record.db")\n'
                                       '# a comment about the vault is fine\n')
    found = {(f, ln, why.split(" (")[0]) for f, ln, why in preflight.lint(tmp_path, network=False)}
    assert ("model.py", 1, "a network library") in found and ("model.py", 2, "a path outside the study") in found
    assert ("model.py", 3, "the study's record") in found and not any(ln == 4 for _, ln, _ in found)
    assert not [x for x in preflight.lint(tmp_path, network=True) if "network" in x[2]]


@bwrap
def test_a_good_model_passes_and_a_broken_one_is_refused_until_it_is_fixed(tmp_path):            # ~20 s
    s = temporal(tmp_path)
    good = s.record.get(s.check()["id"]).payload
    assert good["ok"] and [c["name"] for c in good["checks"]] == ["lint", "import", "smoke", "determinism"]
    edit_model(s, "        return self.m.predict(", "        raise RuntimeError('broken predict')\n        return self.m.predict(")
    red = s.record.get(s.check()["id"]).payload
    smoke = next(c for c in red["checks"] if c["name"] == "smoke")
    assert not red["ok"] and not smoke["ok"] and "broken predict" in smoke["detail"]
    with pytest.raises(StudyError, match="module check is red"):
        s.run(note="broken")
    assert not s.record.find("run")                                  # refused before anything ran: no trial
    edit_model(s, "        raise RuntimeError('broken predict')\n", "")
    s.run(note="fixed")                                              # other code: its check does not bind


@bwrap
def test_a_model_that_predicts_differently_twice_fails_determinism(tmp_path):                   # ~15 s
    s = temporal(tmp_path)
    edit_model(s, "        return self\n",         # entropy no seed fixes, at fit time: steady within a run, not across
               "        import os\n        self.off = int.from_bytes(os.urandom(2), 'big') / 65536\n        return self\n")
    edit_model(s, "        return self.m.predict(X.select(COLUMNS).to_numpy())",
               "        return self.m.predict(X.select(COLUMNS).to_numpy()) + self.off")
    out = s.record.get(s.check()["id"]).payload
    det = next(c for c in out["checks"] if c["name"] == "determinism")
    assert not out["ok"] and not det["ok"] and "seed every random" in det["detail"]


@bwrap
def test_an_ensemble_is_a_counted_cached_run_and_weights_are_cross_fitted(tmp_path):            # ~25 s
    s = temporal(tmp_path)
    a = s.run(note="both")
    edit_model(s, 'COLUMNS = ["x", "y_now"]', 'COLUMNS = ["x"]')
    b = s.run(note="x only")
    used = s.status()["trials"]["used"]
    mean = s.ensemble([a["id"], b["id"]], "mean")
    run = s.record.get(s.record.links(mean["id"], rel="of")[0][2])
    assert run.payload["op"] == "ensemble" and len(s.record.links(run.id, rel="refs")) == 1
    assert s.status()["trials"]["used"] == used + 1
    assert s.ensemble([b["id"], a["id"]], "mean")["id"] == mean["id"]           # the same ensemble: cached
    assert s.status()["trials"]["used"] == used + 1
    with pytest.raises(StudyError, match="rank ensemble is for probabilities"):
        s.ensemble([a["id"], b["id"]], "rank")
    w = s.ensemble([a["id"], b["id"]], "weighted")
    wrun = s.record.get(s.record.links(w["id"], rel="of")[0][2])
    weights = wrun.payload["env"]["weights"]
    assert set(weights) == {"fit_on_even", "fit_on_odd"} and all(abs(sum(v) - 1) < 1e-9 for v in weights.values())
    with pytest.raises(StudyError, match="at least two different runs"):
        s.ensemble([a["id"], a["id"]], "mean")


@bwrap
def test_a_critique_is_advice_answered_by_checks_with_a_computed_track_record(tmp_path):        # ~20 s
    s = temporal(tmp_path)
    r = s.run(note="first")
    run = s.record.links(r["id"], rel="of")[0][2]
    c = s.critique(run, "claude-sonnet-5-5", "9b2e" + "0" * 60,
                   [{"where": "model.py:24", "text": "predict may fail on new sites"},
                    {"where": "model.py:9", "text": "y_now could carry the label"},
                    "the learning rate looks high"], cost_usd=0.04)
    edit_model(s, "        return self.m.predict(", "        raise RuntimeError('new site')\n        return self.m.predict(")
    red = s.check()
    s.finding(c["id"], 0, "checked", "a module check of the change", by=red["id"])
    s.finding(c["id"], 1, "checked", "the truncation audit replays it", by=run)
    s.finding(c["id"], 2, "dismissed", "there is no learning rate in a linear regression")
    t = critique.track(s, c["id"])
    assert (t["confirmed"], t["not confirmed"], t["dismissed"], t["open"]) == (1, 1, 1, 0)
    with pytest.raises(StudyError):
        s.finding(c["id"], 7, "checked", "no such finding")
    assert all("verdict" not in k for k in s.record.get(c["id"]).payload)       # advice carries no verdict


def test_a_check_that_times_out_says_so_first():
    from openlab.study.sandbox import Result
    late = Result(ok=False, output=None, stdout="", sandbox="bwrap", seconds=120.5, timed_out=True,
                  stderr="timed out after 120 s ([sandbox] timeout)\nstep 2 of 3: fitting")
    assert preflight._failed(late, 120, "the child failed") == \
        "timed out after 120 s (the check's limit): step 2 of 3: fitting"
    broken = Result(ok=False, output=None, stdout="", sandbox="bwrap", seconds=1.0,
                    stderr="Traceback ...\nValueError: no column y\nthe child wrote no result.json")
    assert preflight._failed(broken, 120, "the child failed") == "ValueError: no column y"


def test_code_changed_since_its_check_is_named_and_checked_code_is_not():
    class Rec:
        def __init__(self, checks):
            self.checks = checks

        def find(self, kind):
            return self.checks

    class N:
        def __init__(self, id, code):
            self.id, self.payload = id, {"code": code}

    checked = {"model.py": "a", "util.py": "b"}
    assert preflight.unchecked(Rec([]), checked) is None                         # never checked: no warning
    assert preflight.unchecked(Rec([N("module_check:1", checked)]), checked) is None
    warn = preflight.unchecked(Rec([N("module_check:1", checked)]), {"model.py": "c", "util.py": "b"})
    assert warn.startswith("model.py changed since its last check (module_check:1)") and "study_check" in warn


@bwrap
def test_a_run_of_code_edited_after_its_check_warns_and_still_runs(tmp_path):                    # ~20 s
    s = temporal(tmp_path)
    assert s.record.get(s.check()["id"]).payload["ok"]
    first = s.run(note="checked")["trial"]
    assert "warning" not in first
    edit_model(s, 'COLUMNS = ["x", "y_now"]', 'COLUMNS = ["x"]')
    trial = s.run(note="edited after the check")["trial"]
    head, warn = trial["summary"].split("\n")
    assert head.startswith("trial 2 of") and warn == f"warning: {trial['warning']}"
    assert trial["warning"].startswith("model.py changed since its last check")
    assert "warning" not in s.run(note="edited after the check")["trial"]       # cached: nothing ran


def test_a_study_inside_another_study_is_refused(tmp_path):
    (tmp_path / "openlab.toml").write_text("")
    outer = vs.new(tmp_path / "wxmodels", "static")
    with pytest.raises(StudyError, match="is inside the study") as e:
        vs.new(outer.folder / "temp", "static")
    assert "sibling" in e.value.fix and "code hash" in e.value.meaning
    assert not (outer.folder / "temp").exists()
    assert vs.new(tmp_path / "temp", "static")                                    # a sibling is fine
