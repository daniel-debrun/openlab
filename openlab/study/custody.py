"""Holdout labels kept by the person: a static study whose held-out labels never reach the machine that keeps the
record (a shared box, where every process can read every file).

The study declares it in study.toml:

    [split]
    labels = "person"      # the holdout rows are the rows whose target is empty in the data
    labels_key = "id"      # the column that names a row in the person's labels file

The data on the box carries the holdout rows' features with an empty target (a Kaggle-style test set). Research runs
never see those rows. A holdout look on the box predicts them and records the run, its predictions and no result,
since there is nothing there to score them with. On the person's machine, `openlab study score HOST FOLDER --labels
FILE` fetches the predictions (`openlab study holdout-export`), joins the person's labels on `labels_key`, computes the
result with openlab's own metric code, and writes it into the box's record as a person act signed there
(`openlab study holdout-score`, through `openlab person act`). The result carries the labels' sha, so every later look
is scored against the same labels; the box checks what it can (the run, the rows, the signature) and cannot read the
labels.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import polars as pl

from .errors import StudyError


def person_labels(spec) -> bool:
    return spec.split.get("labels") == "person"


def check_spec(spec) -> None:
    """A study whose labels the person keeps: static, with a key column, scored by openlab's own metrics."""
    if not person_labels(spec):
        return
    from .runner import own_metrics
    if spec.mode != "static":
        raise StudyError('split.labels = "person" is for a static study', "a temporal holdout is a later period of "
                         "the same data, so its labels arrive with it", 'remove split.labels, or use mode = "static"')
    if not spec.split.get("labels_key"):
        raise StudyError('split.labels = "person" needs split.labels_key', "the column that names a row in the "
                         "person's labels file", 'e.g. split.labels_key = "id"')
    if spec.split.get("fraction") is not None:
        raise StudyError('split.labels = "person" takes no split.fraction', "the holdout is the rows whose target "
                         "is empty in the data", "remove split.fraction")
    if own_metrics(spec):
        raise StudyError('split.labels = "person" is scored with openlab\'s own metrics only',
                         "the person's machine scores the holdout, and a study's metrics.py runs only in its sandbox",
                         "use built-in metrics in evaluate.metrics")


def unlabelled(df: pl.DataFrame, target: str) -> np.ndarray:
    """The rows whose target is empty (null, or NaN in a float column)."""
    col = df[target]
    mask = col.is_null()
    if col.dtype in (pl.Float32, pl.Float64):
        mask = mask | col.is_nan()
    return np.flatnonzero(mask.to_numpy())


def labels_sha(keys: list, labels: list) -> str:
    pairs = sorted(zip(map(str, keys), map(str, labels)))
    return hashlib.sha256(json.dumps(pairs).encode("utf-8")).hexdigest()


# --- on the machine that keeps the record ------------------------------------------------------------------------
def export(study, run_id: str = "") -> dict:
    """The predictions of a holdout run awaiting its score (the latest, or `run_id`), with each row's key: what the
    person's machine scores. Predictions only: nothing here is a label."""
    from .data import frames
    rec, spec = study.record, study.spec
    if not person_labels(spec):
        raise StudyError("this study keeps its holdout labels itself", 'only a study with split.labels = "person" '
                         "is scored on the person's machine")
    runs = [r for r in rec.find("run", segment="holdout") if not rec.links(r.id, rel="of", incoming=True)]
    run = next((r for r in runs if r.id == run_id), None) if run_id else (runs[-1] if runs else None)
    if run is None:
        raise StudyError(f"no holdout run of {study.spec.name} awaits a score" + (f" ({run_id})" if run_id else ""),
                         "a holdout look records one; its score is written once", "take a look first")
    P = pl.read_parquet(io.BytesIO(rec.blob(run.payload["predictions"])))
    key = spec.split["labels_key"]
    (name,) = spec.data
    df = frames(study)[name]
    P = P.with_columns(_key=df[key].gather(P["_row"].to_list()).cast(pl.Utf8)).drop("y", strict=False)
    buf = io.BytesIO()
    P.write_parquet(buf)
    return {"run": run.id, "study": spec.name, "spec": spec.raw, "seed": run.payload.get("seed") or 0,
            "rows": P.height, "predictions": base64.b64encode(buf.getvalue()).decode("ascii")}


def record_score(study, run_id: str, body: dict) -> object:
    """The person's score of a holdout run, as its result (a person node, so signed): refused unless the run is
    this study's holdout run awaiting a score, the body scores every row it predicted, and earlier scores used the
    same labels."""
    from . import schema
    rec = study.record
    run = rec.get(run_id)
    if run.kind != "run" or run.payload["segment"] != "holdout" or not person_labels(study.spec):
        raise StudyError(f"{run_id} is not a holdout run of a study whose labels the person keeps")
    scored = body.get("scored") or {}
    P = pl.read_parquet(io.BytesIO(rec.blob(run.payload["predictions"])))
    if int(body.get("n", -1)) != P.height or scored.get("rows") != P.height:
        raise StudyError(f"the score covers {body.get('n')} rows; the run predicted {P.height}",
                         "a holdout score is over every held-out row", "score it again with labels for every row")
    for r in rec.find("result"):
        s = r.payload.get("scored") or {}
        if s.get("labels_sha") and s["labels_sha"] != scored.get("labels_sha"):
            raise StudyError("these labels are not the ones the study's earlier looks were scored against",
                             f"labels sha {scored.get('labels_sha', '')[:12]} vs {s['labels_sha'][:12]}",
                             "score with the same labels file")
    return rec.add(schema.Result(**body), actor="person", links=[("of", run.id)])


# --- on the person's machine -------------------------------------------------------------------------------------
def read_labels(path: Path, key: str, target: str) -> pl.DataFrame:
    p = Path(path)
    df = pl.read_parquet(p) if p.suffix == ".parquet" else pl.read_csv(p, infer_schema_length=10000)
    missing = [c for c in (key, target) if c not in df.columns]
    if missing:
        raise StudyError(f"the labels file has no {', '.join(missing)}", f"its columns: {', '.join(df.columns)}",
                         f"a file with the columns {key} and {target}")
    return df.select(pl.col(key).cast(pl.Utf8).alias("_key"), pl.col(target).alias("y"))


def score(exported: dict, labels: pl.DataFrame) -> dict:
    """The result body of an exported holdout run against the person's labels: openlab's evaluation, here."""
    from .runner import _evaluate
    from .spec import build
    from .. import __version__
    import tempfile
    P = pl.read_parquet(io.BytesIO(base64.b64decode(exported["predictions"])))
    J = P.join(labels, on="_key", how="left")
    if J["y"].null_count():
        miss = J.filter(pl.col("y").is_null())["_key"].head(5).to_list()
        raise StudyError(f"{J['y'].null_count()} predicted rows have no label in the file", f"e.g. keys {miss}",
                         "a labels file with every held-out row")
    with tempfile.TemporaryDirectory() as d:
        spec = build(exported["spec"], Path(d))
    check_spec(spec)
    body = _evaluate(spec, J.drop("_key"), int(exported.get("seed") or 0))
    return {**body, "scored": {"by": "person", "labels_sha": labels_sha(J["_key"].to_list(), J["y"].to_list()),
                               "rows": J.height, "openlab": __version__}}


def encode(body: dict) -> str:
    return base64.b64encode(json.dumps(body).encode("utf-8")).decode("ascii")


def decode(text: str) -> dict:
    return json.loads(base64.b64decode(text.encode("ascii")))
