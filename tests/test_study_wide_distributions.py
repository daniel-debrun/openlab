"""A distribution over several outputs per row (e.g. retrieved analogues of the next H steps, with weights): the walk
writes each step's predictions to disk and the runner keeps the predictive mean and the per-row statistics of the
study's metrics, so a forecast too wide to hold is still scored, bootstrapped, paired and audited. A model with
update(X, y) is refitted on the new rows alone, and the audit's fresh fits check it."""

import shutil

import numpy as np
import polars as pl
import pytest

import openlab.study as vs
from openlab.study import metrics
from openlab.study.errors import StudyError

ANALOGUES = '''"""Two outputs per site (tomorrow's y and twice it), forecast by the site's last K labels as weighted members."""

import numpy as np
import polars as pl

K = 4


def features(view, t):
    return (view.frame("panel").sort("date").group_by("site")
            .agg(pl.col("x").last(), pl.col("y").last().alias("y_now")).sort("site"))


def label(view, t):
    last = view.frame("panel").sort("date").group_by("site").agg(pl.col("y").last()).sort("site")
    return last.select("site", pl.concat_list(pl.col("y"), 2 * pl.col("y")).alias("y"))


class Archive:
    """Every label seen, per site; fit stores them, update appends (the same archive a fresh fit would build)."""
    FITS = 0

    def fit(self, X, y):
        type(self).FITS += 1
        self.past = {}
        return self.update(X, y)

    def update(self, X, y):
        for site, row in zip(X["site"].to_list(), np.asarray(y, dtype=float)):
            self.past.setdefault(site, []).append(row)
        return self

    def predict(self, X):
        cols = {}
        for k in range(K):
            m = [self.past.get(s, [np.zeros(2)])[-1 - k] if len(self.past.get(s, [])) > k else
                 self.past.get(s, [np.zeros(2)])[-1] for s in X["site"].to_list()]
            cols[f"m{k}"] = np.asarray(m)
            cols[f"w{k}"] = np.full(X.height, float(K - k))
        return pl.DataFrame({c: v for c, v in cols.items()})


Model = Archive


class Flat(Archive):
    """The same archive, equal weights."""

    def predict(self, X):
        P = super().predict(X)
        return P.select([c for c in P.columns if c.startswith("m")])
'''


def _wide(tmp_path, metrics_line='metrics = ["crps_empirical", "coverage90", "mae"]') -> vs.Study:
    s = vs.new(tmp_path / "wide", "temporal")
    toml = s.folder / "study.toml"
    toml.write_text(toml.read_text().replace('metrics = ["mae", "rmse"]', metrics_line)
                    .replace('prediction = "point"', 'prediction = "ensemble"')
                    .replace('baselines = ["persistence", "climatology"]', 'baselines = ["model:Flat"]')
                    .replace('refit = "30d"', 'refit = "1d"') + "\n[audit]\nn = 3\n")
    (s.folder / "model.py").write_text(ANALOGUES)
    return s


def test_weighted_members_over_several_outputs_score_as_their_empirical_distribution():
    rng = np.random.default_rng(1)
    y, x, w = rng.normal(size=(9, 3)), rng.normal(size=(9, 3, 5)), rng.random((9, 5))
    P = pl.DataFrame({**{f"m{i}": x[:, :, i] for i in range(5)}, **{f"w{i}": w[:, i] for i in range(5)}})
    ww = w / w.sum(1, keepdims=True)
    brute = np.mean([(ww[i] * abs(x[i, j] - y[i, j])).sum()
                     - 0.5 * (ww[i][:, None] * ww[i][None] * abs(x[i, j][:, None] - x[i, j][None])).sum()
                     for i in range(9) for j in range(3)])
    assert metrics.score("crps", y, P) == pytest.approx(brute) == metrics.score("crps_empirical", y, P)
    R = metrics.row_stats("crps_empirical", y, P)
    assert R.shape == (9, 2) and metrics.METRICS["crps_empirical"].total(R.sum(0)) == pytest.approx(brute)
    with pytest.raises(ValueError, match="one weight wK per member"):
        metrics.score("crps", y, P.drop("w4"))


@pytest.mark.skipif(not shutil.which("bwrap"), reason="the study's code runs under bwrap")
def test_a_wide_distribution_is_reduced_step_by_step_and_still_audited(tmp_path):
    s = _wide(tmp_path)
    res = s.run()["payload"]
    run = s.record.last("run")
    assert run.payload["audit"]["verdict"] == "clean", run.payload["audit"]
    m = res["metrics"]
    assert m["crps_empirical"]["lo"] <= m["crps_empirical"]["value"] <= m["crps_empirical"]["hi"]
    assert 0 <= m["coverage90"]["value"] <= 1 and m["mae"]["value"] > 0
    assert res["baselines"]["model:Flat"]["crps_empirical"]["value"] is not None
    assert res["paired"]["model:Flat"]["metric"] == "crps_empirical"
    import io
    P = pl.read_parquet(io.BytesIO(s.record.blob(run.payload["predictions"])))
    assert "m0" not in P.columns and P.schema["pred"] == pl.Array(pl.Float32, 2)
    assert {"@crps_empirical.0", "b:model:Flat:@coverage90.1"} <= set(P.columns)
    with pytest.raises(StudyError, match="rmse was not scored on this run"):
        metrics.score("rmse", P["y"].to_numpy(), P.select("pred", "@mae.0", "@mae.1"))


@pytest.mark.skipif(not shutil.which("bwrap"), reason="the study's code runs under bwrap")
def test_train_from_trains_on_earlier_decisions_and_scores_only_the_research_span(tmp_path):
    s = _wide(tmp_path)
    toml = s.folder / "study.toml"
    toml.write_text(toml.read_text().replace("research = [2024-01-01, 2024-12-31]",
                                             "research = [2024-10-01, 2024-12-31]\ntrain_from = 2024-01-01"))
    s.run()
    run = s.record.last("run")
    import io
    P = pl.read_parquet(io.BytesIO(s.record.blob(run.payload["predictions"])))
    from datetime import datetime
    assert P["_t"].min() == datetime(2024, 10, 1) and run.payload["skipped"] == 0
    assert run.payload["audit"]["verdict"] == "clean"
    late = _wide(tmp_path / "late")
    toml = late.folder / "study.toml"
    toml.write_text(toml.read_text().replace("research = [2024-01-01, 2024-12-31]",
                                             "research = [2024-10-01, 2024-12-31]\ntrain_from = 2024-11-01"))
    with pytest.raises(StudyError, match="train_from` must come before"):
        late.run()
