"""Variant against variant: two of the study's own results compared on the rows both scored.

A study's results are comparable with each other when their runs share the protocol: the same segment, the same
datasets (by sha) and the same sealed data, target and split sections. Then each row (a temporal decision and
entity, or a static row) is scored for both, and the difference of the metric is bootstrapped on the same draws
(whole blocks of decisions where the spec says block_bootstrap), as the runner pairs the model with its baselines.
Results from another protocol (an amended split, other data) are refused, naming what differs. Evidence from the
literature is compared under [review] comparable_on instead (schema.comparison).
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from .errors import StudyError

if TYPE_CHECKING:
    from .record import Node, Record

PROTOCOL = ("data", "target", "split")       # the sealed sections that decide which rows a run scores, and how


def _run(rec: "Record", result: "Node") -> "Node":
    if result.kind != "result":
        raise StudyError(f"{result.id} is a {result.kind}", "variants are compared by their result ids",
                         "openlab study runs lists them")
    return rec.get(rec.links(result.id, rel="of")[0][2])


def protocol_differs(a: "Node", b: "Node") -> list:
    """What differs between two runs' protocols (segment, datasets, the sealed data/target/split sections)."""
    pa, pb = a.payload, b.payload
    out = [f"segment: {pa['segment']} vs {pb['segment']}"] if pa["segment"] != pb["segment"] else []
    if pa["data"] != pb["data"]:
        out.append("datasets: " + ", ".join(f"{k} {str(pa['data'].get(k))[:12]} vs {str(pb['data'].get(k))[:12]}"
                                            for k in sorted({*pa["data"], *pb["data"]})
                                            if pa["data"].get(k) != pb["data"].get(k)))
    sa, sb = pa.get("choices", {}).get("spec", {}), pb.get("choices", {}).get("spec", {})
    out += [f"the sealed [{s}] section" for s in PROTOCOL if sa.get(s) != sb.get(s)]
    return out


def _frame(rec: "Record", run: "Node") -> pl.DataFrame:
    if not run.payload.get("predictions"):
        raise StudyError(f"{run.id} records no predictions", "a paired comparison needs both runs' rows")
    return pl.read_parquet(io.BytesIO(rec.blob(run.payload["predictions"])))


def paired(rec: "Record", a: "Node", b: "Node", metric: str) -> dict:
    """`a` minus `b` (two result nodes) on `metric`, on the rows both runs scored: {metric, values, diff, lo, hi, p,
    better (a result id, or None), n, only_a, only_b, interval}. Refused across protocols."""
    from . import runner
    from .metrics import METRICS
    from .spec import sealed
    if a.id == b.id:
        raise StudyError("a comparison sets two different results side by side")
    ra, rb = _run(rec, a), _run(rec, b)
    differ = protocol_differs(ra, rb)
    if differ:
        raise StudyError(f"incomparable: {a.id} and {b.id} were run under different protocols",
                         "; ".join(differ), "compare results of the same sealed data, target and split; an amended "
                         "protocol starts a new line of results")
    spec = sealed(rec, at=rec.links(ra.id, rel="under")[0][2])     # declares the study's own metrics
    if METRICS.get(metric) is None:
        raise StudyError(f"no metric `{metric}` in openlab.study.metrics", fix=f"one of {', '.join(METRICS)}")
    seg = ra.payload["segment"]
    A, B = (runner._scored(spec, seg, _frame(rec, r)) for r in (ra, rb))
    ids = [c for c in ("_t", "_row", "_key") if c in A.columns and c in B.columns]
    ca, cb = runner._cols(A, ""), runner._cols(B, "")
    if sorted(ca) != sorted(cb):
        raise StudyError("the two runs predict different things", f"{a.id}: {', '.join(ca)}; {b.id}: {', '.join(cb)}")
    J = A.select(*ids, "y", *ca).join(B.select(*ids, pl.col("y").alias("y_b"), *[pl.col(c).alias(f"b:{c}") for c in cb]),
                                      on=ids, how="inner")
    J = runner._known(J, "y")
    if J.height == 0:
        raise StudyError("the two runs share no scored row", fix="compare runs over the same segment")
    ya, yb = J["y"].to_numpy(), J["y_b"].to_numpy()
    same = np.isclose(ya, yb, rtol=1e-12, atol=0, equal_nan=True) if ya.dtype.kind in "fiu" else ya == yb
    if not np.all(same):
        raise StudyError("the two runs scored different labels on the same rows",
                         f"{int(np.size(same) - np.sum(same))} of {len(ya)} rows differ", "their data or target differ")
    temporal = "_t" in J.columns and spec.evaluate.get("interval") != "bootstrap"
    args = {"metric": metric, "ids": ids, "a": ca, "b": cb}
    d = (runner.sandboxed(rec, spec, f"{__name__}:_paired_rows", {"J": J}, **args) if runner.own_metrics(spec, [metric])
         else _paired_rows(spec, J, **args))
    return {"metric": metric, **d, "better": {"a": a.id, "b": b.id}.get(d["better"]), "n": J.height,
            "only_a": A.height - J.height, "only_b": B.height - J.height,
            "interval": "block_bootstrap" if temporal else "bootstrap"}


def _paired_rows(spec, J: pl.DataFrame, metric: str, ids: list, a: list, b: list) -> dict:
    """Each run's `metric` on the joined rows J, and a minus b bootstrapped on the same draws (blocks of decisions
    in a temporal study): {values, diff, lo, hi, p, better ("a", "b" or None), nominal?}."""
    from . import runner
    from .inference import paired as boot
    from .metrics import score
    ev = spec.evaluate
    temporal = "_t" in J.columns and ev.get("interval") != "bootstrap"
    blocks = runner._blocks(J["_t"], ev.get("block", 7)) if temporal else None
    ya = J["y"].to_numpy()
    pa, pb = runner._pred(J.select(*ids, "y", *a), ""), runner._pred(J.select(*[f"b:{c}" for c in b]), "b:")
    ctx = {k: v for k, v in runner.score_ctx(ev).items() if v is not None}
    d = boot(metric, ya, pa, pb, blocks=blocks, seed=spec.split.get("seed", 0), alpha=ev.get("alpha", 0.05), **ctx)
    num = runner._num
    return {"values": [num(score(metric, ya, pa, **ctx)), num(score(metric, ya, pb, **ctx))],
            **{k: num(d[k]) for k in ("diff", "lo", "hi", "p")}, "better": d["better"],
            **({"nominal": d["nominal"]} if "nominal" in d else {})}
