"""A study's own metrics and labels with several outputs: study.toml declares the metric, metrics.py holds its code,
the seal holds that code's sha, and the code runs only in the sandbox (a run, a reproduction, a comparison). A label
may be a vector per row (e.g. the next H steps of a series), scored element by element."""

import hashlib
import shutil
import sys

import numpy as np
import pytest

import openlab.study as vs
from openlab.study import metrics, spec as _spec
from openlab.study.errors import StudyError

OWN = '''
[metrics.mse_h]
better = "lower"
description = "the mean squared error over every output of every row"
'''

METRICS_PY = '''import numpy as np


def mse_h(y, pred):
    return float(np.nanmean((np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)) ** 2))
'''

VECTOR = '''"""Two outputs per site: tomorrow's y and twice it."""

import numpy as np
import polars as pl


def features(view, t):
    return (view.frame("panel").sort("date").group_by("site")
            .agg(pl.col("x").last(), pl.col("y").last().alias("y_now")).sort("site"))


def label(view, t):
    last = view.frame("panel").sort("date").group_by("site").agg(pl.col("y").last()).sort("site")
    return last.select("site", pl.concat_list(pl.col("y"), 2 * pl.col("y")).alias("y"))


class Zero:
    def fit(self, X, y):
        self.width = y.shape[1]
        return self

    def predict(self, X):
        return np.zeros((X.height, self.width))


class Model:
    def fit(self, X, y):
        A = np.column_stack([X["x"].to_numpy(), X["y_now"].to_numpy(), np.ones(X.height)])
        self.w = np.linalg.lstsq(A, y, rcond=None)[0]
        return self

    def predict(self, X):
        return np.column_stack([X["x"].to_numpy(), X["y_now"].to_numpy(), np.ones(X.height)]) @ self.w
'''


def _own(tmp_path, code: str = METRICS_PY) -> vs.Study:
    s = vs.new(tmp_path / "own", "temporal")
    toml = s.folder / "study.toml"
    toml.write_text(toml.read_text().replace('metrics = ["mae", "rmse"]', 'metrics = ["mse_h", "rmse"]')
                    .replace('baselines = ["persistence", "climatology"]', 'baselines = ["model:Zero"]')
                    + OWN + "\n[audit]\nn = 2\n")
    (s.folder / "metrics.py").write_text(code)
    (s.folder / "model.py").write_text(VECTOR)
    return s


def test_a_declared_metric_has_a_direction_and_is_never_scored_here():
    metrics.declare({"own_loss": {"better": "lower", "reads": ["point"]}})
    m = metrics.METRICS["own_loss"]
    assert m.study and m.direction == "lower" and metrics.better("own_loss", 1.0, 2.0)
    with pytest.raises(StudyError, match="runs only in the study's sandbox"):
        metrics.score("own_loss", np.zeros(3), np.ones(3))
    with pytest.raises(StudyError, match="one of openlab's metrics"):
        metrics.declare({"rmse": {"better": "higher"}})


def test_the_spec_holds_the_metric_codes_sha(tmp_path):
    s = _own(tmp_path)
    raw = _spec.load(s.folder).raw
    sha = hashlib.sha256((s.folder / "metrics.py").read_bytes()).hexdigest()
    assert raw["metrics"]["mse_h"]["code_sha"] == sha
    toml = s.folder / "study.toml"
    toml.write_text(toml.read_text().replace('better = "lower"', f'better = "lower"\ncode_sha = "{sha}"'))
    with pytest.raises(StudyError, match="written by openlab"):
        _spec.load(s.folder)
    toml.write_text(toml.read_text().replace(f'code_sha = "{sha}"\n', ""))
    (s.folder / "metrics.py").unlink()
    with pytest.raises(StudyError, match="has no metrics.py"):
        _spec.load(s.folder)


@pytest.mark.skipif(not shutil.which("bwrap"), reason="the study's own code runs under bwrap")
def test_a_vector_label_scored_by_the_studys_own_metric_in_the_sandbox(tmp_path):    # ~15 s
    s = _own(tmp_path)
    res = s.run()["payload"]
    run = s.record.last("run")
    assert run.payload["audit"]["verdict"] == "clean" and run.payload["sandbox"] == "bwrap"
    own, rmse = res["metrics"]["mse_h"], res["metrics"]["rmse"]
    assert own["value"] == pytest.approx(rmse["value"] ** 2) and own["lo"] <= own["value"] <= own["hi"]
    assert res["paired"]["model:Zero"]["metric"] == "mse_h" and res["paired"]["model:Zero"]["better"] == "model"
    assert "openlab_study_metrics" not in sys.modules                 # never imported in this process
    sha = s.spec.raw["metrics"]["mse_h"]["code_sha"]
    assert s.record.blob(sha) == (s.folder / "metrics.py").read_bytes()

    (r,) = s.reproduce()["reproductions"]                            # rescored in the sandbox from the sealed code
    assert r["matched"] and r["metrics"]["mse_h"]["diff"] == 0.0, r

    (s.folder / "metrics.py").write_text(METRICS_PY.replace("** 2", "** 2 * 0.5"))
    with pytest.raises(StudyError, match="metrics.mse_h.code_sha"):
        s.run()
    with pytest.raises(StudyError, match="metrics can only be amended by a person"):
        s.amend("a gentler metric")


def test_a_metric_of_row_sums_bootstraps_its_row_statistics(tmp_path):
    from openlab.study.inference import interval, paired
    (tmp_path / "metrics.py").write_text(METRICS_PY + '''

def mse_r(y, pred):
    return mse_h(y, pred)


def mse_r_rows(y, pred):
    return np.column_stack([np.sum((y - pred) ** 2, axis=1), np.full(len(y), y.shape[1])])


def mse_r_total(sums):
    return float(sums[0] / sums[1])


def bad(y, pred):
    return mse_h(y, pred)


def bad_rows(y, pred):
    return np.ones((len(y), 2))


def bad_total(sums):
    return float(sums[0])
''')
    decl = {"better": "lower"}
    metrics.adopt(tmp_path, {"mse_h": decl, "mse_r": decl, "bad": decl})
    rng = np.random.default_rng(1)
    y, a, b = rng.normal(size=(300, 4)), rng.normal(size=(300, 4)), rng.normal(scale=2, size=(300, 4))
    blocks = np.arange(300) // 7
    assert interval("mse_r", y, a, blocks=blocks) == pytest.approx(interval("mse_h", y, a, blocks=blocks), rel=1e-9)
    pr, ph = paired("mse_r", y, a, b, blocks=blocks), paired("mse_h", y, a, b, blocks=blocks)
    assert pr["better"] == ph["better"] == "a" and pr["diff"] == pytest.approx(ph["diff"])
    assert (pr["lo"], pr["hi"], pr["p"]) == pytest.approx((ph["lo"], ph["hi"], ph["p"]), rel=1e-9)
    with pytest.raises(StudyError, match="give the metric exactly"):
        interval("bad", y, a)


def test_a_paired_difference_uses_only_the_rows_both_methods_scored():
    from openlab.study.inference import paired
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])               # a NaN: a row the method did not score
    b = np.array([2.0, np.nan, 3.0, 5.0, 6.0, 7.0])
    d = paired("mae", y, a, b, draws=200)
    assert d["dropped"] == 2 and d["diff"] == pytest.approx(-1.0)      # rows 0, 3, 4, 5: mae 0 against 1
    assert "dropped" not in paired("mae", y, y.copy(), y + 1, draws=50)
