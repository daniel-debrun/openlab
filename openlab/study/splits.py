"""Splits. Temporal: which decisions each segment scores, when the model refits, and which decisions it may
train on. Static: the validation folds and the holdout carve, dealt by the fold functions below.

`split.train_from` (a date before the research start): decisions from then on train the fits but are never
predicted or scored, e.g. the paper's train and validation windows ahead of its test set.

The embargo is the horizon: a decision t is in a segment only if t + horizon <= the segment end, so no label
crosses a split edge; a fit at s trains only on decisions t_i with t_i + horizon <= s, so every training label
was known when the fit ran.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import polars as pl

from .durations import Duration, parse
from .errors import StudyError
from .spec import Spec


# --- the folds: k-fold, stratified, grouped (indices into the frame's rows) ----------------------------------
class CVError(ValueError):
    def __init__(self, error: str, meaning: str = "", try_: str = ""):
        super().__init__(error)
        self.error, self.meaning, self.try_ = error, meaning, try_


def _folds_kfold(n: int, k: int, seed: int) -> list:
    order = np.random.default_rng(seed).permutation(n)
    return [(np.setdiff1d(order, part), part) for part in np.array_split(order, k)]


def _folds_stratified(y: list, k: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))                       # ties in the target broken at random
    order = perm[np.argsort(np.asarray(y)[perm], kind="stable")]
    idx = np.empty(len(y), dtype=int)
    for i in range(0, len(y), k):                        # each run of k neighbours in the target spans k folds
        chunk = order[i:i + k]
        idx[chunk] = rng.permutation(k)[:len(chunk)]
    return [(np.where(idx != f)[0], np.where(idx == f)[0]) for f in range(k)]


def _folds_group(keys: list, k: int, seed: int) -> list:
    groups = sorted({str(v) for v in keys})
    if len(groups) < k:
        raise CVError("too few groups", f"{len(groups)} groups for {k} folds", f"folds={max(2, len(groups))} or fewer")
    order = np.random.default_rng(seed).permutation(len(groups))
    fold_of = {groups[g]: i % k for i, g in enumerate(order)}
    idx = np.array([fold_of[str(v)] for v in keys])
    return [(np.where(idx != f)[0], np.where(idx == f)[0]) for f in range(k)]


CADENCES = {"daily": "1d", "weekly": "1w", "monthly": "1M"}       # weekly: every 7 days from the research start


@dataclass(frozen=True)
class Step:
    t: datetime
    refit: bool
    train_until: datetime                   # the fit time s: labels known by s
    train_from: datetime | None = None      # window start, exclusive; None is an expanding window

    def trains_on(self, t_i: datetime, h: Duration) -> bool:
        """Whether the decision at t_i is a training row for this step's fit."""
        return t_i + h <= self.train_until and (self.train_from is None or t_i > self.train_from)


def _dt(v) -> datetime:
    if isinstance(v, datetime):
        return v
    if isinstance(v, date):
        return datetime(v.year, v.month, v.day)
    return datetime.fromisoformat(str(v))


def horizon(spec: Spec) -> Duration:
    """The target horizon, which is also the embargo."""
    if spec.split.get("embargo", "horizon") != "horizon":
        raise StudyError('study.toml: `split.embargo` can only be "horizon"', "the embargo is the target horizon",
                         "remove split.embargo")
    if "horizon" not in spec.target:
        raise StudyError("study.toml: `target.horizon` is required in temporal mode", fix='e.g. horizon = "1d"')
    return parse(spec.target["horizon"])


def _span(v, key: str) -> tuple:
    start, end = _dt(v[0]), _dt(v[1])
    if start > end:
        raise StudyError(f"study.toml: `{key}` starts after it ends")
    return start, end


def segments(spec: Spec) -> dict:
    s = spec.split
    if "research" not in s:
        raise StudyError("study.toml: `split.research` is required in temporal mode", fix="research = [start, end]")
    seg = {"research": _span(s["research"], "split.research"),
           "holdout": _span(s["holdout"], "split.holdout") if "holdout" in s else None}
    if seg["holdout"] and seg["holdout"][0] <= seg["research"][1]:
        raise StudyError("study.toml: `split.holdout` must start after `split.research` ends",
                         "an overlap puts holdout periods in research")
    return seg


def last_end(spec: Spec) -> datetime:
    """The end of the study's last segment: a forward decision comes after it."""
    return max(v[1] for v in segments(spec).values() if v)


def decision_times(spec: Spec, frames: dict, until: datetime | None = None) -> list[datetime]:
    """The decision times inside the segments (with `until`: from the research start through `until`, for forward
    runs): a cadence (monthly is month-ends), or the distinct values of `<dataset>.<column>`."""
    spans = [v for v in segments(spec).values() if v]
    if until is not None:
        spans = [(spans[0][0], until)]
    if spec.split.get("train_from") is not None:          # training-only decisions before the research start
        tf = _dt(spec.split["train_from"])
        if tf >= spans[0][0]:
            raise StudyError("study.toml: `split.train_from` must come before `split.research` starts",
                             "its decisions only train the fits; the research span is what is scored")
        spans = [(tf, spans[0][0]), *spans]
    how = spec.split.get("decisions")
    if how in CADENCES:
        lo, hi, every, ts = spans[0][0], spans[-1][1], parse(CADENCES[how]), []
        t = lo.replace(day=calendar.monthrange(lo.year, lo.month)[1]) if how == "monthly" else lo
        while t <= hi:
            ts.append(t)
            t = t + every
    else:
        name, _, col = (how or "").partition(".")
        if name not in frames or col not in frames[name].columns:
            raise StudyError(f"study.toml: `split.decisions` = {how!r} is not a cadence or a dataset column",
                             fix='"daily", "weekly", "monthly" or "<dataset>.<column>"')
        ts = sorted({_dt(v) for v in frames[name].get_column(col).drop_nulls().to_list()})
    return [t for t in ts if any(a <= t <= b for a, b in spans)]


def _refits(split: dict, ts: list, segment: str) -> tuple[list, set]:
    """(the scored decisions, the refit times among them)."""
    if split["scheme"] == "time" and segment == "research":
        k = split.get("k", 5)
        if len(ts) < k + 1:
            raise StudyError(f"{len(ts)} research decisions for {k} time folds", fix="fewer folds (split.k)")
        blocks = np.array_split(np.arange(len(ts)), k + 1)[1:]
        return ts[blocks[0][0]:], {ts[b[0]] for b in blocks}
    every = parse(split["refit"]) if "refit" in split else None
    refits, last = set(), None
    for t in ts:
        if last is None or (every is not None and t >= last + every):
            refits.add(t)
            last = t
    return ts, refits


def temporal_steps(spec: Spec, times: list, segment: str) -> list[Step]:
    """The scored steps of one segment, in time order.

    walk_forward refits at the segment's first decision and again once `split.refit` has passed (no refit: once).
    time cuts the research decisions into k + 1 blocks and refits at the start of blocks 1..k, the ones scored;
    its holdout refits as walk_forward does. `split.window` ("expanding", or a duration) bounds the training
    decisions to (s - window, s]."""
    span = segments(spec).get(segment)
    if span is None:
        raise StudyError(f"the study has no {segment} segment", fix=f"add split.{segment} = [start, end]")
    h = horizon(spec)
    ts = sorted({t for t in map(_dt, times) if span[0] <= t and t + h <= span[1]})
    if not ts:
        raise StudyError(f"no decision falls in the {segment} segment",
                         "a decision t counts only if t + horizon <= the segment end")
    ts, refits = _refits(spec.split, ts, segment)
    window = spec.split.get("window", "expanding")
    width = None if window == "expanding" else parse(window)
    steps, s = [], None
    for t in ts:
        s = t if t in refits else s
        steps.append(Step(t, t in refits, s, None if width is None else s - width))
    return steps


def _column(df: pl.DataFrame, name, key: str) -> list:
    if not name or name not in df.columns:
        raise StudyError(f"study.toml: `{key}` must name a column of the data", f"it is {name!r}")
    return df.get_column(name).to_list()


def _folds(df: pl.DataFrame, spec: Spec, scheme: str, k: int, seed: int | None = None) -> list:
    seed = spec.split.get("seed", 0) if seed is None else seed
    try:
        if scheme == "group":
            return _folds_group(_column(df, spec.split.get("by"), "split.by"), k, seed)
        if scheme == "stratified":
            return _folds_stratified(_column(df, spec.target.get("column"), "target.column"), k, seed)
        return _folds_kfold(df.height, k, seed)
    except CVError as e:
        raise StudyError(e.error, e.meaning, e.try_) from None


def static_folds(df: pl.DataFrame, spec: Spec, seed: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train, validation) row indices: kfold, stratified on target.column, or whole split.by groups. With
    `split.reseed = true`, `seed` (the run's) deals them, so a paper's "mean over 20 random splits" is one run per
    seed; the holdout carve keeps split.seed either way."""
    return _folds(df, spec, spec.split["scheme"], spec.split.get("k", 5),
                  seed if spec.split.get("reseed") else None)


def carve(df: pl.DataFrame, spec: Spec) -> tuple[np.ndarray, np.ndarray]:
    """(research rows, holdout rows): one of k = 1 / split.fraction folds, of whole split.by groups when set,
    else stratified under the stratified scheme, else random. No fraction: no holdout."""
    if spec.split.get("labels") == "person":       # the person keeps the holdout's labels: its rows have none here
        from .custody import unlabelled
        hold = unlabelled(df, spec.target["column"])
        return np.setdiff1d(np.arange(df.height), hold), hold
    f = spec.split.get("fraction")
    if f is None:
        return np.arange(df.height), np.array([], dtype=int)
    k = round(1 / f)
    if abs(k * f - 1) > 1e-9:
        raise StudyError(f"study.toml: `split.fraction` = {f} is not 1/k", "the holdout is one of k equal folds",
                         "use 0.5, 0.25, 0.2, 0.1, ...")
    by, scheme = spec.split.get("by"), spec.split["scheme"]
    research, holdout = _folds(df, spec, "group" if by else "stratified" if scheme == "stratified" else "kfold", k)[0]
    return np.sort(research), np.sort(holdout)
