"""Ensembles: a combination of recorded research runs, recorded as a run of its own and counted as a trial (after
AIBuildAI's aggregator, with the combination on the record).

`openlab study ensemble FOLDER --members RUN,RUN[,...] --method mean|rank|weighted` combines the members' recorded
predictions on the rows all of them scored:

- `mean` averages each prediction column;
- `rank` averages rank-percentiles (probabilities only: it changes the scale);
- `weighted` fits non-negative weights summing to 1 (a grid of 0.1 on the simplex, at most 4 members) that best the
  primary metric, cross-fitted: the rows are split in two, weights fit on one half predict the other, so no row is
  predicted by weights fit on it.

Members must share the protocol (spec, data, segment) and the labels, and none may have an audit that found
look-ahead. The ensemble is a run with op `ensemble`, its first member as parent and the others as refs; its key
hashes the members' keys and the method, so the same ensemble twice is cached and spends no trial. Its result is
scored by the study's own evaluation. It has no fitted model of its own, so it is a research candidate: the
selection rule considers single runs, whose code a holdout look can run.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import time

import numpy as np
import polars as pl

from .errors import StudyError

METHODS = ("mean", "rank", "weighted")


def _frame(rec, run) -> pl.DataFrame:
    return pl.read_parquet(io.BytesIO(rec.blob(run.payload["predictions"])))


def _combine(frames: list, cols: list, method: str, weights=None) -> pl.DataFrame:
    base = frames[0]
    out = {}
    for c in cols:
        M = np.column_stack([f[c].to_numpy().astype(float) for f in frames])
        if method == "rank":
            M = np.column_stack([pl.Series(M[:, j]).rank("average").to_numpy() / len(M) for j in range(M.shape[1])])
        w = np.full(M.shape[1], 1 / M.shape[1]) if weights is None else np.asarray(weights)
        out[c] = M @ w
    return base.with_columns(**{c: pl.Series(c, v) for c, v in out.items()})


def _grid(k: int):
    steps = range(11)
    for combo in itertools.product(steps, repeat=k - 1):
        if sum(combo) <= 10:
            yield [c / 10 for c in combo] + [(10 - sum(combo)) / 10]


def build(study, members: list, method: str = "mean"):
    """Combine, score and record the ensemble; returns its result node (an earlier identical one when cached)."""
    from . import runner, schema
    from .metrics import score
    from .spec import sealed
    rec = study.record
    if method not in METHODS:
        raise StudyError(f"no ensemble method `{method}`", f"one of {', '.join(METHODS)}")
    runs = []
    for m in members:
        n = rec.get(m)
        if n.kind == "result":
            n = rec.get(rec.links(n.id, rel="of")[0][2])
        if n.kind != "run" or n.payload["segment"] != "research" or not rec.links(n.id, rel="of", incoming=True):
            raise StudyError(f"{m} is not a research run with a result")
        if n.payload["audit"].get("verdict") == "look-ahead":
            raise StudyError(f"{n.id}'s audit found look-ahead", "an ensemble of it would carry the leak")
        runs.append(n)
    if len({r.id for r in runs}) < 2:
        raise StudyError("an ensemble combines at least two different runs")
    first = runs[0].payload
    if any((r.payload["spec_sha"], r.payload["data"]) != (first["spec_sha"], first["data"]) for r in runs):
        raise StudyError("the members ran under different protocols or data", "an ensemble combines runs of one "
                         "protocol", "choose runs of the current sealed spec and data")
    spec = sealed(rec)
    if method == "rank" and spec.evaluate.get("prediction") != "prob":
        raise StudyError("a rank ensemble is for probabilities", "it changes the scale of a point prediction",
                         "use mean or weighted")
    if method == "weighted" and len(runs) > 4:
        raise StudyError("a weighted ensemble takes at most 4 members", "the weight grid grows fast")
    key = hashlib.sha256(json.dumps({"members": sorted(r.payload["key"] for r in runs), "method": method})
                         .encode()).hexdigest()
    done = runner._cached(rec, key)
    if done is not None:
        return done
    runner._budget(study, spec, key)
    t0 = time.monotonic()
    frames = [_frame(rec, r) for r in runs]
    ids = [c for c in ("_t", "_row", "_key") if all(c in f.columns for f in frames)]
    cols = runner._cols(frames[0], "")
    if any(sorted(runner._cols(f, "")) != sorted(cols) for f in frames):
        raise StudyError("the members predict different things", "an ensemble combines like predictions")
    keyset = frames[0].select(ids)
    for f in frames[1:]:
        keyset = keyset.join(f.select(ids), on=ids, how="inner")
    frames = [keyset.join(f, on=ids, how="left") for f in frames]
    ys = [f["y"].to_numpy() for f in frames]
    if not all(np.array_equal(ys[0], y, equal_nan=True) if y.dtype.kind in "fc" else np.array_equal(ys[0], y)
               for y in ys[1:]):
        raise StudyError("the members scored different labels on the same rows")
    weights = None
    if method == "weighted":
        from .metrics import METRICS
        m = spec.evaluate["metrics"][0]
        sign = -1.0 if METRICS[m].higher_is_better else 1.0
        ctx = runner.score_ctx(spec.evaluate)

        def loss(fitted: list, w: list) -> float:
            C = runner._known(_combine(fitted, cols, "mean", w), "y")
            v = score(m, C["y"].to_numpy(), runner._pred(C, ""), **ctx)
            return sign * v if v is not None and not np.isnan(v) else float("inf")
        half = pl.Series(np.arange(keyset.height) % 2 == 0)
        parts, weights = [], {}
        for name, fit, apply in (("fit_on_even", half, ~half), ("fit_on_odd", ~half, half)):
            fitted = [f.filter(fit) for f in frames]
            best = min(_grid(len(runs)), key=lambda w: loss(fitted, w))
            weights[name] = best                                    # weights fit on one half predict the other
            parts.append(_combine([f.filter(apply) for f in frames], cols, "mean", best))
        P = pl.concat(parts).sort(ids)
    else:
        P = _combine(frames, cols, method)
    ev = runner._evaluate(spec, runner._scored(spec, "research", P), 0)
    verdicts = {r.payload["audit"]["verdict"] for r in runs}
    v = verdicts.pop() if len(verdicts) == 1 else "unchecked"
    audit = {"verdict": v, "checked": min(r.payload["audit"].get("checked", 0) for r in runs) if v == "clean" else 0,
             "mismatches": []}
    if v not in ("clean", "separated", "in-sample"):
        audit = {"verdict": "unchecked", "checked": 0, "mismatches": []}
    buf = io.BytesIO()
    P.write_parquet(buf)
    rest = runs[1:]
    under = rec.last("amendment") or rec.last("plan")
    uses = [d for _, _, d in rec.links(runs[0].id, rel="uses")]
    run = rec.add(schema.Run(
        segment="research", key=key, cached=False, spec_sha=first["spec_sha"],
        code={"ensemble": hashlib.sha256(json.dumps([method, [r.payload["code"] for r in runs], weights],
                                                    sort_keys=True).encode()).hexdigest()},
        data=first["data"], env={"ensemble": method, "members": [r.id for r in runs], "weights": weights},
        sandbox="ensemble", audit=audit, seconds=time.monotonic() - t0, stdout=rec.put_blob(b""),
        predictions=rec.put_blob(buf.getvalue()), note=f"{method} ensemble of {', '.join(r.id for r in runs)}",
        op="ensemble"), actor="agent",
        links=[("under", under.id), *[("uses", u) for u in uses], ("branches_from", runs[0].id),
               *[("refs", r.id) for r in rest]])
    return rec.add(schema.Result(**ev), actor="agent", links=[("of", run.id)])
