"""The baselines a model has to beat. Each gets the model's ``X`` (with ``_key`` and, in temporal mode, ``_t``)
and ``y``, and returns what a model returns: an array, or a ``mu``/``sigma`` frame when the study predicts a
gaussian."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import polars as pl

from .errors import StudyError

if TYPE_CHECKING:
    from .spec import Spec


def _gaussian(spec: "Spec") -> bool:
    return spec.evaluate.get("prediction") == "gaussian"


def _out(mu: np.ndarray, sd: np.ndarray, gaussian: bool):
    return pl.DataFrame({"mu": mu, "sigma": sd}) if gaussian else mu


def _need_time(X: pl.DataFrame, name: str) -> None:
    if "_t" not in X.columns:
        raise StudyError(f"the {name} baseline needs decision times", "a static study has no _t",
                         "use majority or mean in a static study")


class Majority:
    """The most common training class (the smallest on a tie); with ``prob`` predictions, the training share of
    the positive class (the one that sorts last, as the metrics read it)."""

    def __init__(self, spec: "Spec"):
        self.prob = spec.evaluate.get("prediction") == "prob"

    def fit(self, X, y):
        values, counts = np.unique(np.asarray(y), return_counts=True)
        self.top = values[np.argmax(counts)]
        positive = max(str(v) for v in values)
        self.rate = float(np.mean([str(v) == positive for v in y]))
        return self

    def predict(self, X):
        return np.full(len(X), self.rate) if self.prob else np.full(len(X), self.top)


class Mean:
    """The training mean (and sd, ddof 1, for a gaussian)."""

    def __init__(self, spec: "Spec"):
        self.gaussian = _gaussian(spec)

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.mu, self.sd = float(y.mean()), float(y.std(ddof=1))
        return self

    def predict(self, X):
        return _out(np.full(len(X), self.mu), np.full(len(X), self.sd), self.gaussian)


class Persistence:
    """Per ``_key``, the latest training label at or before the row's ``_t``; the overall mean where there is
    none."""

    def __init__(self, spec: "Spec"):
        pass

    def fit(self, X, y):
        _need_time(X, "persistence")
        self.train = X.select("_key", "_t").with_columns(_y=pl.Series(np.asarray(y, dtype=float))).sort("_t")
        self.mean = float(np.mean(np.asarray(y, dtype=float)))
        return self

    def predict(self, X):
        _need_time(X, "persistence")
        rows = X.select("_key", "_t").with_row_index("_i").sort("_t")
        hit = rows.join_asof(self.train, on="_t", by="_key", strategy="backward", check_sortedness=False)
        return hit.sort("_i")["_y"].fill_null(self.mean).to_numpy()


class Climatology:
    """Per ``_key``, the mean (and sd) of the training labels within 15 days of the row's day of year, any year.
    Fewer than 2 such labels: the key's own; an unseen key or one with fewer than 2: all training labels."""

    WINDOW = 15

    def __init__(self, spec: "Spec"):
        self.gaussian = _gaussian(spec)

    def fit(self, X, y):
        _need_time(X, "climatology")
        self.key, self.day = X["_key"].to_numpy(), X["_t"].dt.ordinal_day().to_numpy()
        self.y = np.asarray(y, dtype=float)
        self.floor = max(1e-6, 0.01 * float(np.std(self.y)))
        return self

    def predict(self, X):
        _need_time(X, "climatology")
        key, day = X["_key"].to_numpy(), X["_t"].dt.ordinal_day().to_numpy()
        mu, sd = np.empty(len(X)), np.empty(len(X))
        for k in np.unique(key):
            at, owned = key == k, self.key == k
            if owned.sum() < 2:
                owned = np.ones(len(self.y), dtype=bool)
            ty, td = self.y[owned], self.day[owned]
            gap = np.abs(day[at][:, None] - td[None, :])
            near = np.minimum(gap, 365.25 - gap) <= self.WINDOW             # day of year wraps at the new year
            near[near.sum(axis=1) < 2] = True                               # too few near: the key's own labels
            n = near.sum(axis=1)
            m = near @ ty / n
            mu[at] = m
            sd[at] = np.sqrt(np.maximum((near @ ty ** 2 - n * m ** 2) / (n - 1), 0.0))
        sd = np.maximum(sd, self.floor)        # equal labels give sd 0; a gaussian needs a spread
        return _out(mu, sd, self.gaussian)


BASELINES: dict[str, Callable[["Spec"], object]] = {
    "majority": Majority, "mean": Mean, "persistence": Persistence, "climatology": Climatology}


def make(name: str, spec: "Spec", model_module=None) -> object:
    """A fresh baseline: a built-in by name, or ``model:Name``, a class in the study's model.py."""
    if name.startswith("model:"):
        cls = getattr(model_module, name[len("model:"):], None)
        if cls is None:
            raise StudyError(f"baseline {name!r} names no class in model.py", "",
                             f"define class {name[len('model:'):]} in model.py with fit and predict")
        return cls()
    if name not in BASELINES:
        raise StudyError(f"no baseline named {name!r}", "", f"one of {', '.join(BASELINES)}, or model:ClassName")
    return BASELINES[name](spec)
