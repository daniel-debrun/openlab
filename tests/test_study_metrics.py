"""The study's metrics agree with their textbook definitions."""

import numpy as np
import pytest

from openlab.study.metrics import score


def test_metrics_agree_with_their_definitions():
    y = [0, 0, 1, 1, 1, 0]
    assert score("accuracy", y, [0, 1, 1, 1, 0, 0]) == pytest.approx(4 / 6)
    assert score("balanced_accuracy", y, [0, 1, 1, 1, 0, 0]) == pytest.approx(2 / 3)
    assert score("macro_f1", y, [0, 1, 1, 1, 0, 0]) == pytest.approx(2 / 3)
    assert score("auc", y, [0.1, 0.4, 0.35, 0.8, 0.7, 0.2]) == pytest.approx(8 / 9)
    assert score("auc", y, [0.5] * 6) == pytest.approx(0.5)
    assert score("logloss", [0, 1], [0.5, 0.5]) == pytest.approx(np.log(2))
    r = [1.0, 2.0, 3.0, 4.0]
    assert score("rmse", r, [1, 2, 3, 6]) == pytest.approx(1.0)
    assert score("mae", r, [1, 2, 3, 6]) == pytest.approx(0.5)
    assert score("r2", r, r) == pytest.approx(1.0) and score("r2", r, [2.5] * 4) == pytest.approx(0.0)


def _bands():
    """The same 90% interval three ways: a lower/upper frame, the 5% and 95% quantiles, and a gaussian."""
    import polars as pl
    from scipy.special import ndtri
    lo, hi = np.array([0.0, 1.0, 2.0, 3.0]), np.array([2.0, 2.0, 4.0, 7.0])
    mu, sd = (lo + hi) / 2, (hi - lo) / 2 / ndtri(0.95)
    return (pl.DataFrame({"lower": lo, "upper": hi}), pl.DataFrame({"q0.05": lo, "q0.5": mu, "q0.95": hi}),
            pl.DataFrame({"mu": mu, "sigma": sd}))


def test_interval_width_and_coverage_read_the_central_interval():
    y = [1.0, 3.0, 3.0, 5.0]                            # inside, outside, inside, inside
    interval, quantiles, gaussian = _bands()
    for pred in (quantiles, gaussian):
        assert score("width90", y, pred) == pytest.approx(2.25)
        assert score("coverage90", y, pred) == pytest.approx(0.75)
    assert score("width90", y, interval, level=0.9) == pytest.approx(2.25)
    assert score("coverage90", y, interval, level=0.9) == pytest.approx(0.75)
    assert score("interval_width", y, interval, level=0.9) == pytest.approx(2.25)
    assert score("coverage", y, interval, level=0.9) == pytest.approx(0.75)
    assert score("coverage", y, quantiles, level=0.9) == pytest.approx(0.75)


def test_an_interval_at_an_undeclared_or_other_level_is_refused():
    from openlab.study.errors import StudyError
    y = [1.0, 3.0, 3.0, 5.0]
    interval, quantiles, _ = _bands()
    with pytest.raises(StudyError, match="does not say its level") as e:
        score("width90", y, interval)
    assert e.value.fix == "declare it: evaluate.level = 0.9"
    with pytest.raises(StudyError, match="the predicted one is at 0.8"):
        score("coverage90", y, interval, level=0.8)
    with pytest.raises(StudyError, match="declares none"):
        score("interval_width", y, quantiles)
    with pytest.raises(ValueError, match="q0.1 and q0.9"):
        score("coverage", y, quantiles, level=0.8)


def test_a_run_scores_a_lower_upper_prediction_at_the_declared_level():
    import polars as pl
    from types import SimpleNamespace
    from openlab.study import runner
    y = np.random.default_rng(0).normal(size=200)
    P = pl.DataFrame({"_row": np.arange(200), "y": y, "lower": np.full(200, -1.645), "upper": np.full(200, 1.645),
                      "b:model:Wide:lower": np.full(200, -3.0), "b:model:Wide:upper": np.full(200, 3.0)})
    spec = SimpleNamespace(split={}, evaluate={"metrics": ["width90", "coverage90"], "level": 0.9,
                                               "baselines": ["model:Wide"]})
    out = runner._evaluate(spec, P, 0)
    assert out["metrics"]["width90"]["value"] == pytest.approx(3.29)
    assert out["baselines"]["model:Wide"]["width90"]["value"] == pytest.approx(6.0)
    assert out["paired"]["model:Wide"]["better"] == "model"                 # width90: shorter is better
    spec.evaluate.pop("level")
    assert "does not say its level" in runner._evaluate(spec, P, 0)["metrics"]["width90"]["reason"]
