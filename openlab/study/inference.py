"""How sure a score is: bootstrap intervals, paired differences against a baseline, Diebold-Mariano, Holm.

Every bootstrap draws rows with replacement, or with ``blocks`` (one block id per row) whole blocks, which keeps
the dependence inside a block (e.g. a week of daily decisions). A draw on which the metric is undefined (a
ValueError, or NaN: one class only for auc) is skipped; with fewer than half the draws left the answer is NaN.
"""

from __future__ import annotations

import math

import numpy as np

from .errors import StudyError
from .metrics import METRICS, better, nominal, row_stats, score


def _resamples(n: int, blocks, draws: int, seed: int):
    """The row indices of each draw. With one block per row this is the plain bootstrap, draw for draw."""
    rng = np.random.default_rng(seed)
    if blocks is None:
        for _ in range(draws):
            yield rng.integers(0, n, n)
        return
    _, inv = np.unique(np.asarray(blocks), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    rows = np.split(order, np.cumsum(np.bincount(inv))[:-1])     # rows[k]: the rows of block k
    for _ in range(draws):
        yield np.concatenate([rows[k] for k in rng.integers(0, len(rows), len(rows))])


def _take(x, i):
    return None if x is None else x[i] if hasattr(x, "columns") else np.asarray(x)[i]


def _try(metric: str, y, pred, ctx: dict):
    try:
        v = score(metric, y, pred, **ctx)
    except ValueError:
        return None
    return None if math.isnan(v) else v


def _total(metric: str, R, i):
    """A decomposable metric on the rows `i` of its per-row statistics R (None where it is undefined)."""
    try:
        v = METRICS[metric].total(R[i].sum(axis=0))
    except ValueError:
        return None
    return None if math.isnan(v) else v


def _band(vals: list, draws: int, alpha: float) -> tuple:
    if len(vals) < draws // 2:
        return math.nan, math.nan
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def interval(metric: str, y, pred, *, blocks=None, draws: int = 1000, seed: int = 0, alpha: float = 0.05,
             **ctx) -> tuple:
    """``(lo, hi)``: the percentile bootstrap interval of ``metric`` at level ``1 - alpha``."""
    vals, R = [], row_stats(metric, y, pred, **ctx)
    for i in _resamples(len(pred), blocks, draws, seed):
        v = _try(metric, _take(y, i), _take(pred, i), ctx) if R is None else _total(metric, R, i)
        if v is not None:
            vals.append(v)
    return _band(vals, draws, alpha)


def paired(metric: str, y, pred_a, pred_b, *, blocks=None, draws: int = 1000, seed: int = 0,
           alpha: float = 0.05, **ctx) -> dict:
    """The difference ``metric(a) - metric(b)`` on the same rows, bootstrapped on the same draws.
    ``p`` is two-sided: twice the smaller share of draws on either side of 0. ``better`` is ``"a"`` or ``"b"``
    by the point values and the metric's direction: for a target-valued metric (coverage90) the one closer to its
    ``nominal`` value, which the reply carries (None for a tie or a metric with no order)."""
    sa, sb = score(metric, y, pred_a, **ctx), score(metric, y, pred_b, **ctx)
    diff = sa - sb
    ds, Ra = [], row_stats(metric, y, pred_a, **ctx)
    Rb = None if Ra is None else row_stats(metric, y, pred_b, **ctx)
    for i in _resamples(len(pred_a), blocks, draws, seed):
        if Ra is None:
            a, b = _try(metric, _take(y, i), _take(pred_a, i), ctx), _try(metric, _take(y, i), _take(pred_b, i), ctx)
        else:
            a, b = _total(metric, Ra, i), _total(metric, Rb, i)
        if a is not None and b is not None:
            ds.append(a - b)
    lo, hi = _band(ds, draws, alpha)
    d = np.asarray(ds)
    side = min(int((d <= 0).sum()), int((d >= 0).sum()))
    p = math.nan if math.isnan(lo) else min(1.0, 2 * (side + 1) / (len(d) + 1))   # +1: p is never 0 from finite draws
    win = better(metric, sa, sb, ctx.get("level"))
    nom = nominal(metric, ctx.get("level"))
    return {"diff": diff, "lo": lo, "hi": hi, "p": p, "better": None if win is None else "a" if win else "b",
            **({"nominal": nom} if nom is not None else {})}


def _t_tail(t: float, df: float) -> tuple:
    """``(p, exact)``: two-sided p for a t, from scipy when it is there, else the normal approximation."""
    try:
        from scipy import stats
        return float(2 * stats.t.sf(abs(t), df)), True
    except Exception:                                      # noqa: BLE001
        return float(math.erfc(abs(t) / math.sqrt(2.0))), False


def diebold_mariano(loss_a, loss_b, h: int = 1) -> dict:
    """Are two forecasts' expected losses equal? ``d = loss_a - loss_b`` per decision; its mean over a HAC
    standard error (autocovariances to lag ``h - 1``, the forecast horizon in decisions), scaled by Harvey,
    Leybourne and Newbold's (1997) small-sample correction and read against t with n - 1 df. A negative
    ``diff`` means a has the lower loss."""
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    dbar = float(d.mean())
    if dbar == 0:
        return {"stat": 0.0, "p": 1.0, "diff": 0.0, "n": n, "h": h}
    e = d - dbar
    var = (e @ e + 2 * sum(e[k:] @ e[:-k] for k in range(1, h))) / n / n
    if var <= 0:
        raise StudyError("the loss difference has no usable variance",
                         f"its HAC variance at lag {h - 1} is {var:.3g}", "use more decisions, or a smaller h")
    stat = dbar / math.sqrt(float(var)) * math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return {"stat": stat, "p": _t_tail(stat, n - 1)[0], "diff": dbar, "n": n, "h": h}

