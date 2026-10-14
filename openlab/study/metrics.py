r"""One versioned registry of metrics: how a prediction is scored against the truth.

A metric declares the prediction kinds it reads and a version that a run records. The kinds:

- ``point`` and ``prob``: a 1-D array, one value per row (a class or a number, or the positive class's
  probability).
- ``gaussian``: a frame with ``mu`` and ``sigma``.
- ``quantiles``: a frame with ``q0.1``, ``q0.5``, ... (the level after the ``q``).
- ``ensemble``: a frame with members ``m0``, ``m1``, ..., and optionally their weights ``w0``, ``w1``, ... (a weighted
  empirical distribution, e.g. retrieved analogues with softmax weights; each row's weights are normalised to sum
  to 1).
- ``interval``: a frame with ``lower`` and ``upper``, one central interval per row (e.g. conformalized quantile
  regression). It does not say its level, so the study declares it (``evaluate.level``).
- ``series``: per-decision values with no truth (``y=None``), e.g. the value of acting on each decision.
- ``frame``: any other prediction frame, columns of the model's own (a staircase region's cells, a set's bounds):
  only a study's own metric reads it (``reads = ["frame"]``), since only it knows what the columns mean.

A label with several outputs per row (y 2-D, e.g. the next H steps of C series) takes the same frames with each
column holding one value per output (an array column): ``mu``, ``q0.1`` or ``m0`` is then rows x outputs. A metric
averages over the rows and the outputs alike, and the rows stay the unit of the bootstrap: each built-in mean-type
metric has per-row statistics (``rows``/``total``), so its interval resamples rows without scoring again.

The core holds no domain metrics: a package registers its own (:func:`register`). A package may also register the older way, ``openlab.extensions.METRICS[name] = (fn(y, yhat), higher_is_better)``;
such a metric is read here as a ``point``/``prob`` metric of version 1.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import numpy as np

from .. import extensions as X
from .errors import StudyError


#: a target-valued metric whose nominal value is the study's declared ``evaluate.level`` (``coverage``)
LEVEL = "level"


@dataclass(frozen=True)
class Metric:
    """A metric is ordered one of three ways (``direction``): ``higher`` or ``lower`` is better, or ``target``:
    best at its ``nominal`` value, so better is closer to it (coverage90: 0.9). With neither, it has no order."""
    name: str
    fn: Callable                       # fn(y, pred, **ctx) -> float
    higher_is_better: Optional[bool]   # None: target-valued (with a nominal) or unordered
    prediction: tuple                  # the kinds it reads
    version: int
    nominal: object = None             # a target-valued metric's best value: a number, or LEVEL
    unit: str = ""                     # "fraction": a share (a paper may print it in percent); "": its own units
    study: bool = False                # a study's own metric (study.toml [metrics.NAME], code in its metrics.py)
    rows: Optional[Callable] = None    # rows(y, pred) -> per-row statistics (rows x k), and total(their column
    total: Optional[Callable] = None   # sums) -> the metric: the bootstrap resamples the statistics, not y and pred

    @property
    def direction(self) -> str:
        if self.nominal is not None:
            return "target"
        return {True: "higher", False: "lower"}.get(self.higher_is_better, "none")


class _Registry(X.Registry):
    """name -> Metric. A read first loads the packages that register (``openlab.extensions.load``) and adopts the
    metrics they registered the older way. A name already here keeps its meaning."""

    def _adopt(self) -> None:
        for name in list(X.METRICS):
            if name not in self._d:
                fn, hib = X.METRICS[name]
                self._d[name] = Metric(name, _legacy(fn), bool(hib), ("point", "prob"), 1)

    def __getitem__(self, name):
        self._adopt()
        return self._d[name]

    def __iter__(self):
        self._adopt()
        return iter(self._d)

    def __len__(self):
        self._adopt()
        return len(self._d)


def _legacy(fn):
    return lambda y, pred, **_: float(fn(list(y), list(pred)))


METRICS: dict = _Registry()


def register(m: Metric) -> None:
    """Add a metric. A name keeps one meaning: registering a different metric under it is refused."""
    old = METRICS._d.get(m.name)                   # not METRICS.get: a read would load the registering packages
    if old is not None and old != m:
        raise StudyError(f"a metric named {m.name!r} is already registered", "a metric name keeps one meaning",
                         "register yours under another name")
    METRICS._d[m.name] = m


# --- a study's own metrics ---------------------------------------------------------------------------------------
#: where a study's own metrics live, beside model.py. study.toml declares each ([metrics.NAME]: better, reads, unit)
#: and the plan seals the file's sha, so the metric is fixed before the first result, like the rest of the plan.
STUDY_FILE = "metrics.py"
KINDS = ("point", "prob", "gaussian", "quantiles", "ensemble", "interval", "frame")


@dataclass(frozen=True)
class _Sandboxed:
    """A study metric's stand-in outside the sandbox: openlab never runs a study's code in its own process."""
    name: str

    def __call__(self, y, pred, **_):
        raise StudyError(f"{self.name} is the study's own metric", "its code (metrics.py) runs only in the study's "
                         "sandbox", "openlab scores it inside a run, a reproduction or a comparison")


def _declared(name: str, decl: dict, fn, rows=None, total=None) -> Metric:
    return Metric(name, fn, decl["better"] == "higher", tuple(decl.get("reads", ["point"])), 1,
                  unit=decl.get("unit", ""), study=True, rows=rows, total=total)


def _same(a: Metric, b: Metric) -> bool:
    return (a.higher_is_better, a.prediction, a.unit) == (b.higher_is_better, b.prediction, b.unit)


def declare(decls: dict) -> None:
    """Register a study's own metrics by their declaration ([metrics.NAME] in study.toml): the parent knows each
    one's direction, the predictions it reads and its unit, and scoring it here is refused (its code runs in the
    study's sandbox: `adopt`). A name openlab or a package already gives a meaning is refused."""
    for name, d in decls.items():
        m, old = _declared(name, d, _Sandboxed(name)), METRICS._d.get(name)
        if old is not None and not old.study:
            raise StudyError(f"study.toml: [metrics.{name}] is the name of one of openlab's metrics",
                             "a metric name keeps one meaning", f"name the study's own otherwise, e.g. {name}_study")
        if old is None or not _same(old, m):
            METRICS._d[name] = m


def _takes_keywords(fn) -> bool:
    import inspect
    try:
        return any(p.kind is p.VAR_KEYWORD for p in inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return False


def _call(fn):
    """fn(y, pred, **ctx) when fn takes keywords, else fn(y, pred); a float either way."""
    if _takes_keywords(fn):
        return lambda y, pred, **ctx: float(fn(y, pred, **ctx))
    return lambda y, pred, **_: float(fn(y, pred))


def adopt(code_dir, decls: dict) -> None:
    """In the sandbox: the study's metrics.py imported from `code_dir` (the run's snapshot) and each declared metric
    registered with its function. `fn(y, pred)` gets numpy arrays (2-D for a label with several outputs) or the
    prediction frame, and returns a float. A metric that is a function of sums over rows may also define
    `<name>_rows(y, pred)` (per-row statistics, rows x k) and `<name>_total(sums)` (the metric from their column
    sums): its intervals then resample those statistics (`row_stats`), which is much faster for wide labels."""
    import importlib.util
    from pathlib import Path
    if not decls:
        return
    path = Path(code_dir) / STUDY_FILE
    if not path.is_file():
        raise StudyError(f"study.toml declares {', '.join(decls)}, and the study has no {STUDY_FILE}",
                         fix=f"define each declared metric as a function in {STUDY_FILE}")
    spec = importlib.util.spec_from_file_location("openlab_study_metrics", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name, d in decls.items():
        fn = getattr(mod, name, None)
        if not callable(fn):
            raise StudyError(f"{STUDY_FILE} defines no function {name}", f"study.toml declares [metrics.{name}]",
                             f"def {name}(y, pred): ... returning a float")
        rows, total = getattr(mod, f"{name}_rows", None), getattr(mod, f"{name}_total", None)
        if callable(rows) != callable(total):
            raise StudyError(f"{STUDY_FILE} defines one of {name}_rows and {name}_total", "the bootstrap needs both: "
                             "the per-row statistics and the metric from their sums", "define both, or neither")
        METRICS._d[name] = _declared(name, d, _call(fn), _rows(rows) if callable(rows) else None,
                                     (lambda s, f=total: float(f(s))) if callable(total) else None)


def _rows(fn):
    """rows(y, pred, **ctx) as a 2-D float array (rows x k)."""
    kw = _takes_keywords(fn)

    def rows(y, pred, **ctx):
        R = np.asarray(fn(y, pred, **ctx) if kw else fn(y, pred), dtype=float)
        return R[:, None] if R.ndim == 1 else R
    return rows


def row_stats(name: str, y, pred, **ctx):
    """A decomposable metric's per-row statistics (rows x k), or None for a metric without them. They are used
    only after this check: the metric's total of their column sums is the metric itself on these rows."""
    m = METRICS[name]
    R = _kept(name, pred)
    if R is not None:
        return R
    if m.rows is None:
        return None
    R = m.rows(y, pred, **ctx)
    if R.shape[0] != len(pred):
        raise StudyError(f"{name}_rows gave {R.shape[0]} rows for {len(pred)}", "one row of statistics per row")
    a, b = m.total(R.sum(axis=0)), score(name, y, pred, **ctx)
    if not (math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) or (math.isnan(a) and math.isnan(b))):
        raise StudyError(f"{name}_total of {name}_rows' sums is {a!r}, and {name} is {b!r}", "the bootstrap resamples "
                         "the per-row statistics only when they give the metric exactly", "fix them in metrics.py")
    return R


def nominal(name: str, level: Optional[float] = None) -> Optional[float]:
    """A target-valued metric's nominal value: its own, or for one at the declared level, ``level`` (None when the
    study declared none). None for a metric with a direction."""
    m = METRICS.get(name)
    if m is None or m.nominal is None:
        return None
    return (None if level is None else float(level)) if m.nominal == LEVEL else float(m.nominal)


def better(name: str, a: float, b: float, level: Optional[float] = None) -> Optional[bool]:
    """Is ``a`` better than ``b`` on ``name``: True, False, or None for a tie or no order. A target-valued metric
    is better closer to its nominal value (at ``level`` for one at the declared level; no order without it)."""
    m = METRICS[name]
    if m.nominal is not None:
        nom = nominal(name, level)
        if nom is None or abs(a - nom) == abs(b - nom):
            return None
        return abs(a - nom) < abs(b - nom)
    if m.higher_is_better is None or a == b:
        return None
    return (a > b) == m.higher_is_better


def best(name: str, values, level: Optional[float] = None) -> Optional[float]:
    """The best of ``values`` on ``name`` (None when the metric has no order)."""
    m, values = METRICS[name], list(values)
    if m.nominal is not None:
        nom = nominal(name, level)
        return None if nom is None else min(values, key=lambda v: abs(v - nom))
    return None if m.higher_is_better is None else (max if m.higher_is_better else min)(values)


def arrow(name: str, level: Optional[float] = None) -> str:
    """The direction in words: "higher is better", "lower is better", "best at 0.9" (or "best at evaluate.level")."""
    m = METRICS.get(name)
    if m is None or m.direction == "none":
        return "no direction"
    if m.direction == "target":
        nom = nominal(name, level)
        return f"best at {nom:g}" if nom is not None else "best at evaluate.level"
    return f"{m.direction} is better"


def _kept(name: str, pred):
    """The per-row statistics a run kept of `name` (columns @<name>.0, ...) when it reduced a wide distribution
    step by step (openlab.study.runner._reduce); None for a prediction kept whole."""
    if not _is_frame(pred) or not any(c.startswith("@") for c in pred.columns):
        return None
    cols = sorted((c for c in pred.columns if c.startswith(f"@{name}.")), key=lambda c: int(c.rsplit(".", 1)[1]))
    if not cols:
        kept = sorted({c[1:].rsplit(".", 1)[0] for c in pred.columns if c.startswith("@")})
        raise StudyError(f"{name} was not scored on this run", "its predictions were a distribution over several "
                         f"outputs, kept as the per-row statistics of {', '.join(kept)} only",
                         "declare the metric in evaluate.metrics before the run")
    return np.column_stack([np.asarray(pred[c].to_numpy(), dtype=float) for c in cols])


def score(name: str, y, pred, **ctx) -> float:
    """``name`` on these predictions. ``ctx`` reaches the metric (e.g. ``periods_per_year``); a ``prediction``
    in it (the spec's ``evaluate.prediction``) says whether an array is ``point`` or ``prob``."""
    m = METRICS.get(name)
    if m is None:
        raise StudyError(f"no metric named {name!r}", "", f"one of {', '.join(METRICS)}")
    R = _kept(name, pred)
    if R is not None:
        return float(m.total(R.sum(axis=0)))
    got = _kind(y, pred)
    if ctx.get("prediction") in got:
        got = (ctx["prediction"],)
    if not set(got) & set(m.prediction):
        raise StudyError(f"{name} does not score a {' or '.join(got)} prediction",
                         f"{name} reads {', '.join(m.prediction)}",
                         "pick a metric for this prediction kind, or change what predict returns")
    return float(m.fn(y, pred, **ctx))


# --- prediction kinds ------------------------------------------------------------------------------------
_QUANT, _MEMBER, _WEIGHT = re.compile(r"q(0?\.\d+)"), re.compile(r"m\d+"), re.compile(r"w\d+")


def _is_frame(pred) -> bool:
    return hasattr(pred, "columns")


def _frame_kind(pred) -> str:
    cols = list(pred.columns)
    if {"mu", "sigma"} <= set(cols):
        return "gaussian"
    if any(_QUANT.fullmatch(c) for c in cols):
        return "quantiles"
    if any(_MEMBER.fullmatch(c) for c in cols):
        return "ensemble"
    if {"lower", "upper"} <= set(cols):
        return "interval"
    raise StudyError(f"a prediction frame with columns {cols} is no known kind", "",
                     "return mu and sigma (gaussian), q0.1, q0.5, ... (quantiles), m0, m1, ... (ensemble) or lower and "
                     "upper (interval)")


def _kind(y, pred) -> tuple:
    if y is None:
        return ("series",)
    if not _is_frame(pred):
        return ("point", "prob")
    try:
        return (_frame_kind(pred),)
    except StudyError:                              # columns of the model's own (a region's cells, a set's bounds)
        return ("frame",)


def _col(pred, c: str) -> np.ndarray:
    """A prediction column as floats: rows, or rows x outputs for an array (or list) column."""
    v = pred[c]
    dt = str(getattr(v, "dtype", ""))
    if dt.startswith("List"):
        return np.asarray(v.to_list(), dtype=float)
    return np.asarray(v.to_numpy() if hasattr(v, "to_numpy") else v, dtype=float)


def _gauss(pred) -> tuple:
    mu, sd = _col(pred, "mu"), _col(pred, "sigma")
    if np.any(sd <= 0):
        raise ValueError("a gaussian prediction needs sigma > 0 on every row")
    return mu, sd


def _quantiles(pred) -> tuple:
    """``(levels, values)``: the levels ascending, values with one level per entry of the last axis (rows x levels,
    or rows x outputs x levels)."""
    cols = sorted((float(c[1:]), c) for c in pred.columns if _QUANT.fullmatch(c))
    return np.array([lv for lv, _ in cols]), np.stack([_col(pred, c) for _, c in cols], axis=-1)


def _member_cols(pred) -> list:
    return sorted((c for c in pred.columns if _MEMBER.fullmatch(c)), key=lambda c: int(c[1:]))


def _members(pred) -> np.ndarray:
    """The members, one per entry of the last axis (rows x M, or rows x outputs x M)."""
    return np.stack([_col(pred, c) for c in _member_cols(pred)], axis=-1)


def _weights(pred, x: np.ndarray):
    """The members' weights broadcast to `x`'s shape and normalised to sum to 1, or None for equal weights. A weight
    column wK goes with the member mK: per row (one value) or per row and output."""
    ws = [c for c in pred.columns if _WEIGHT.fullmatch(c)]
    if not ws:
        return None
    members = _member_cols(pred)
    if sorted(ws, key=lambda c: int(c[1:])) != [f"w{c[1:]}" for c in members]:
        raise ValueError("a weighted ensemble gives one weight wK per member mK")
    w = np.stack([_col(pred, f"w{c[1:]}") for c in members], axis=-1)
    if w.ndim < x.ndim:                                 # one weight per row: the same for every output
        w = np.broadcast_to(w[:, None, :], x.shape)
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("an ensemble's weights are finite and not negative")
    total = w.sum(axis=-1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError("an ensemble's weights sum to more than 0 on every row")
    return w / total


# --- class and probability metrics (the vault's) ---------------------------------------------------------
def _cls(v) -> str:
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)


def _labels(y, yhat) -> tuple:
    return np.asarray([_cls(v) for v in y]), np.asarray([_cls(v) for v in yhat])  # 1.0 is class 1


def accuracy(y, yhat, **_) -> float:
    y, yhat = _labels(y, yhat)
    return float((y == yhat).mean())


def hit_rate(y, p, **_) -> float:
    """A probability's call at 0.5 against the binary outcome: the share called right."""
    t = _positive("hit_rate", y)
    return float(((np.asarray(p, dtype=float) > 0.5) == (t == 1)).mean())


def balanced_accuracy(y, yhat, **_) -> float:
    y, yhat = _labels(y, yhat)
    return float(np.mean([(yhat[y == c] == c).mean() for c in np.unique(y)]))


def macro_f1(y, yhat, **_) -> float:
    y, yhat = _labels(y, yhat)
    f1 = []
    for c in np.unique(y):
        tp = float(((y == c) & (yhat == c)).sum())
        fp, fn = float(((y != c) & (yhat == c)).sum()), float(((y == c) & (yhat != c)).sum())
        f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1))


def _precision_recall(name: str, y, yhat) -> tuple:
    classes = sorted({str(v) for v in y})
    if len(classes) != 2:
        raise ValueError(f"{name} needs a binary target; this one has {len(classes)} classes")
    t = np.asarray([str(v) == classes[1] for v in y])
    norm = [_cls(v) for v in yhat]
    if set(norm) <= set(classes):                           # class labels
        pred = np.asarray([v == classes[1] for v in norm])
    else:                                                   # a probability of the positive class, cut at 0.5
        pred = np.asarray(yhat, dtype=float) >= 0.5
    tp, fp, fn = float((t & pred).sum()), float((~t & pred).sum()), float((t & ~pred).sum())
    return tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0


def precision(y, yhat, **_) -> float:
    return _precision_recall("precision", y, yhat)[0]


def recall(y, yhat, **_) -> float:
    return _precision_recall("recall", y, yhat)[1]


def f1(y, yhat, **_) -> float:
    p, r = _precision_recall("f1", y, yhat)
    return 2 * p * r / (p + r) if p + r else 0.0


def _positive(name: str, y) -> np.ndarray:
    """1.0 where the truth is the positive class (the one that sorts last), else 0.0."""
    classes = sorted({str(v) for v in y})
    if len(classes) != 2:
        raise ValueError(f"{name} needs a binary target; this one has {len(classes)} classes")
    return np.asarray([1.0 if str(v) == classes[1] else 0.0 for v in y])


def brier(y, yhat, **_) -> float:
    yhat = np.asarray(yhat, dtype=float)
    return float(((yhat - _positive("brier", y)) ** 2).mean())


def logloss(y, yhat, **_) -> float:
    q = np.clip(np.asarray(yhat, dtype=float), 1e-15, 1 - 1e-15)
    t = _positive("logloss", y)
    return float(-(t * np.log(q) + (1 - t) * np.log(1 - q)).mean())


def auc(y, yhat, **_) -> float:
    yhat = np.asarray(yhat, dtype=float)
    t = _positive("auc", y)
    from scipy.stats import rankdata
    ranks = rankdata(yhat)                                                # ties share the mean rank
    pos, neg = t.sum(), (1 - t).sum()
    if pos == 0 or neg == 0:
        raise ValueError("auc needs both classes in the holdout")
    return float((ranks[t == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


# --- numeric metrics ---------------------------------------------------------------------------------------
def _ensemble_mean(p) -> np.ndarray:
    x = _members(p)
    w = _weights(p, x)
    return x.mean(axis=-1) if w is None else (w * x).sum(axis=-1)


_MEAN = {"gaussian": lambda p: _col(p, "mu"),
         # the mean of evenly spaced quantiles approximates the predictive mean
         "quantiles": lambda p: _quantiles(p)[1].mean(axis=-1),
         "ensemble": _ensemble_mean}


def _point(pred) -> np.ndarray:
    """The prediction as one number per row: the array itself, or a frame's predictive mean."""
    return _MEAN[_frame_kind(pred)](pred) if _is_frame(pred) else np.asarray(pred, dtype=float)


def _sq_err(y, pred, **_) -> np.ndarray:
    return (np.asarray(y, dtype=float) - _point(pred)) ** 2


def _abs_err(y, pred, **_) -> np.ndarray:
    return np.abs(np.asarray(y, dtype=float) - _point(pred))


def rmse(y, pred, **_) -> float:
    return float(np.sqrt(_sq_err(y, pred).mean()))


def mae(y, pred, **_) -> float:
    return float(_abs_err(y, pred).mean())


def mape(y, yhat, **_) -> float:
    yhat, y = np.asarray(yhat, dtype=float), np.asarray(y, dtype=float)
    if np.any(y == 0):
        raise ValueError("mape divides by the truth, and some held-out values are 0; use mae")
    return float(np.abs((y - yhat) / y).mean())


def r2(y, yhat, **_) -> float:
    yhat, y = np.asarray(yhat, dtype=float), np.asarray(y, dtype=float)
    ss = float(((y - y.mean()) ** 2).sum())
    return float(1 - ((y - yhat) ** 2).sum() / ss) if ss else float("nan")


# --- probabilistic metrics ---------------------------------------------------------------------------------
def _pinball_rows(y, pred, **_) -> np.ndarray:
    """Each row's (and output's) pinball loss averaged over its quantile levels."""
    levels, q = _quantiles(pred)
    u = np.asarray(y, dtype=float)[..., None] - q
    return np.maximum(levels * u, (levels - 1) * u).mean(axis=-1)


def pinball(y, pred, **_) -> float:
    """The pinball (quantile) loss, averaged over the levels and the rows."""
    return float(_pinball_rows(y, pred).mean())


def _crps_gaussian(y, pred) -> np.ndarray:
    """Closed form: sigma (z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi)), z = (y - mu) / sigma."""
    from scipy.special import ndtr
    mu, sd = _gauss(pred)
    z = (y - mu) / sd
    return sd * (z * (2 * ndtr(z) - 1) + 2 * np.exp(-z * z / 2) / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi))


def _crps_ensemble(y, pred) -> np.ndarray:
    """The fair CRPS (Ferro 2014): mean |x_i - y| - sum_ij |x_i - x_j| / (2 M (M - 1)), unbiased for the
    distribution the M members are drawn from. With one member it is the absolute error. A weighted ensemble is a
    distribution in itself (no fair form): its CRPS is that of the weighted empirical distribution."""
    x = _members(pred)
    if _weights(pred, x) is not None:
        return _crps_weighted(y, pred)
    x = np.sort(x, axis=-1)
    m = x.shape[-1]
    err = np.abs(x - np.asarray(y, dtype=float)[..., None]).mean(axis=-1)
    if m < 2:
        return err
    w = 2 * np.arange(1, m + 1) - m - 1                    # sum_ij |x_i - x_j| = 2 sum_i w_i x_(i), x sorted
    return err - (x @ w) / (m * (m - 1))


def _crps_weighted(y, pred) -> np.ndarray:
    """The CRPS of the members' empirical distribution (weights wK, else equal): sum_i w_i |x_i - y| - 1/2 sum_ij
    w_i w_j |x_i - x_j|, as properscoring's crps_ensemble and most papers that score samples compute it."""
    x = _members(pred)
    w = _weights(pred, x)
    w = np.full(x.shape, 1.0 / x.shape[-1]) if w is None else w
    i = np.argsort(x, axis=-1)
    x, w = np.take_along_axis(x, i, axis=-1), np.take_along_axis(w, i, axis=-1)
    err = (w * np.abs(x - np.asarray(y, dtype=float)[..., None])).sum(axis=-1)
    c = np.cumsum(w, axis=-1)
    spread = 2 * (w * x * ((c - w) - (1 - c))).sum(axis=-1)   # sum_ij w_i w_j |x_i - x_j|, x sorted
    return err - spread / 2


_CRPS = {"gaussian": _crps_gaussian,
         "ensemble": _crps_ensemble,
         # CRPS = 2 * integral over levels of the pinball loss; evenly spaced levels approximate the integral
         "quantiles": lambda y, p: 2 * _pinball_rows(y, p)}


def _crps_elems(y, pred, **_) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if not _is_frame(pred):
        return np.abs(y - np.asarray(pred, dtype=float))
    return _CRPS[_frame_kind(pred)](y, pred)


def crps(y, pred, **_) -> float:
    """The continuous ranked probability score (lower is better, in the target's units). A point prediction
    is a one-point distribution, whose CRPS is the absolute error. An unweighted ensemble is scored by the fair
    CRPS; `crps_empirical` scores the members' empirical distribution instead."""
    return float(_crps_elems(y, pred).mean())


def _crps_empirical_elems(y, pred, **_) -> np.ndarray:
    if _is_frame(pred) and _frame_kind(pred) == "ensemble":
        return _crps_weighted(y, pred)
    return _crps_elems(y, pred)


def crps_empirical(y, pred, **_) -> float:
    """The CRPS of an ensemble's empirical distribution (weighted by wK when given): what a paper computes from its
    samples or retrieved analogues with the standard (not the fair) estimator. Other kinds score as `crps`."""
    return float(_crps_empirical_elems(y, pred).mean())


def _band_quantiles(pred, lo: float, hi: float) -> tuple:
    levels, q = _quantiles(pred)
    at = [np.flatnonzero(np.isclose(levels, v)) for v in (lo, hi)]
    if not all(len(i) for i in at):
        raise ValueError(f"a central interval needs the quantiles q{lo:g} and q{hi:g}")
    return q[..., at[0][0]], q[..., at[1][0]]


def _band_gaussian(pred, lo: float, hi: float) -> tuple:
    from scipy.special import ndtri
    mu, sd = _gauss(pred)
    return mu + ndtri(lo) * sd, mu + ndtri(hi) * sd


def _band_ensemble(pred, lo: float, hi: float) -> tuple:
    """The members' quantiles at plotting position k / (M + 1): a calibrated ensemble (the truth exchangeable
    with its M members) then covers the level. Beyond 1/(M+1) they clamp to the extreme members. A weighted
    ensemble's band is its weighted empirical distribution's lo and hi quantiles (the first member whose cumulative
    weight reaches each level)."""
    x = _members(pred)
    w = _weights(pred, x)
    if w is None:
        b = np.quantile(x, [lo, hi], axis=-1, method="weibull")
        return b[0], b[1]
    i = np.argsort(x, axis=-1)
    x, c = np.take_along_axis(x, i, axis=-1), np.cumsum(np.take_along_axis(w, i, axis=-1), axis=-1)
    at = [np.minimum((c < q - 1e-12).sum(axis=-1), x.shape[-1] - 1)[..., None] for q in (lo, hi)]
    return np.take_along_axis(x, at[0], axis=-1)[..., 0], np.take_along_axis(x, at[1], axis=-1)[..., 0]


_BAND = {"gaussian": _band_gaussian, "quantiles": _band_quantiles, "ensemble": _band_ensemble}


def _central(name: str, pred, at, level) -> tuple:
    """(lower, upper) of each row's central interval at ``at``, a metric's own level (coverage90: 0.9), or with
    ``at`` None at ``level``, the study's declared ``evaluate.level``. A ``lower``/``upper`` frame does not say its
    level: the declared one must be there and be the metric's."""
    want = level if at is None else at
    if want is None:
        raise StudyError(f"{name} reads the interval at the study's declared level", "the study declares none",
                         "set evaluate.level, e.g. 0.9")
    if not 0 < want < 1:
        raise StudyError(f"evaluate.level {want} is no interval level", "a central interval's level is in (0, 1)")
    kind = _frame_kind(pred)
    if kind != "interval":
        return _BAND[kind](pred, (1 - want) / 2, (1 + want) / 2)
    if level is None:
        raise StudyError(f"{name} reads the {want:g} interval, and a lower/upper prediction does not say its level",
                         "openlab cannot tell a 90% interval from an 80% one", f"declare it: evaluate.level = {want:g}")
    if not math.isclose(level, want):
        raise StudyError(f"{name} reads the {want:g} interval, and the predicted one is at {level:g} "
                         "(evaluate.level)", "", "use coverage and interval_width (at the declared level)")
    lo, hi = _col(pred, "lower"), _col(pred, "upper")
    if np.any(hi < lo):
        raise ValueError("an interval prediction needs lower <= upper on every row")
    return lo, hi


def _covered(y, pred, at: Optional[float] = None, level: Optional[float] = None, name: str = "coverage",
             **_) -> np.ndarray:
    lo, hi = _central(name, pred, at, level)
    y = np.asarray(y, dtype=float)
    return ((y >= lo) & (y <= hi)).astype(float)


def coverage(y, pred, at: Optional[float] = None, level: Optional[float] = None, name: str = "coverage",
             **_) -> float:
    """The share of truths inside the central interval at ``at`` (0.9: from the 5% to the 95% quantile), or with
    ``at`` None at the declared ``level``. A calibrated forecast covers that share of them."""
    return float(_covered(y, pred, at, level, name).mean())


def _widths(y, pred, at: Optional[float] = None, level: Optional[float] = None, name: str = "interval_width",
            **_) -> np.ndarray:
    lo, hi = _central(name, pred, at, level)
    return hi - lo


def width(y, pred, at: Optional[float] = None, level: Optional[float] = None, name: str = "interval_width",
          **_) -> float:
    """The mean length of the central interval at ``at``, or at the declared ``level`` (in the target's units;
    shorter is better at the same coverage)."""
    return float(_widths(y, pred, at, level, name).mean())


def _mean_rows(elems):
    """rows(y, pred) of a metric that is the mean of `elems` over rows and outputs: each row's sum and count."""
    def rows(y, pred, **ctx):
        e = np.asarray(elems(y, pred, **ctx), dtype=float)
        e = e.reshape(len(e), -1)
        return np.column_stack([e.sum(axis=1), np.full(len(e), e.shape[1], dtype=float)])
    return rows


def _mean_total(s) -> float:
    return float(s[0] / s[1])


def _rms_total(s) -> float:
    return float(np.sqrt(s[0] / s[1]))


# --- series metrics (y is None; pred holds one value per decision) -----------------------------------------
def mean(y, pred, **_) -> float:
    return float(np.mean(np.asarray(pred, dtype=float)))


def total(y, pred, **_) -> float:
    return float(np.sum(np.asarray(pred, dtype=float)))


_ROWS, _FRAMES = ("point", "prob"), ("gaussian", "quantiles", "ensemble")
_BANDS = _FRAMES + ("interval",)                 # what has a central interval
#: what the core registers, in this order (the vault's thirteen first)
_BUILTIN = (
    Metric("accuracy", accuracy, True, ("point",), 1, unit="fraction"),
    Metric("balanced_accuracy", balanced_accuracy, True, ("point",), 1, unit="fraction"),
    Metric("macro_f1", macro_f1, True, ("point",), 1, unit="fraction"),
    Metric("auc", auc, True, ("prob",), 1, unit="fraction"),
    Metric("hit_rate", hit_rate, True, ("prob",), 1, unit="fraction"),
    Metric("logloss", logloss, False, ("prob",), 1),
    Metric("rmse", rmse, False, ("point",) + _FRAMES, 1, rows=_mean_rows(_sq_err), total=_rms_total),
    Metric("mae", mae, False, ("point",) + _FRAMES, 1, rows=_mean_rows(_abs_err), total=_mean_total),
    Metric("r2", r2, True, ("point",), 1),
    Metric("brier", brier, False, ("prob",), 1),
    Metric("mape", mape, False, ("point",), 1, unit="fraction"),
    Metric("precision", precision, True, _ROWS, 1, unit="fraction"),
    Metric("recall", recall, True, _ROWS, 1, unit="fraction"),
    Metric("f1", f1, True, _ROWS, 1, unit="fraction"),
    Metric("crps", crps, False, ("point",) + _FRAMES, 1, rows=_mean_rows(_crps_elems), total=_mean_total),
    Metric("crps_empirical", crps_empirical, False, ("point",) + _FRAMES, 1, rows=_mean_rows(_crps_empirical_elems),
           total=_mean_total),
    Metric("pinball", pinball, False, ("quantiles",), 1, rows=_mean_rows(_pinball_rows), total=_mean_total),
    Metric("coverage90", partial(coverage, at=0.9, name="coverage90"), None, _BANDS, 1, nominal=0.9,
           unit="fraction", rows=_mean_rows(partial(_covered, at=0.9, name="coverage90")), total=_mean_total),
    Metric("coverage", coverage, None, _BANDS, 1, nominal=LEVEL, unit="fraction",   # at evaluate.level
           rows=_mean_rows(_covered), total=_mean_total),
    Metric("width90", partial(width, at=0.9, name="width90"), False, _BANDS, 1,
           rows=_mean_rows(partial(_widths, at=0.9, name="width90")), total=_mean_total),
    Metric("interval_width", width, False, _BANDS, 1, rows=_mean_rows(_widths), total=_mean_total),  # at evaluate.level
    Metric("mean", mean, True, ("series",), 1),
    Metric("total", total, True, ("series",), 1),
)
for _m in _BUILTIN:
    register(_m)
