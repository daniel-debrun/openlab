"""Forward runs: a forecast at the latest decision time by a registered model, gated and signed for a program
downstream, and its outcome once the label is known.

`forecast` refreshes the data when `[schedule] refresh = true`, takes the latest decision time at or before now
(after the study's segments), computes the features at t and predicts with the pinned model (`[forecast] model`:
"latest" or a version) in a sandboxed child, which is the only place the model is unpickled. It checks the
predictions against `[forecast] gate` (`%%expect` predicates: SQL over a table `t`), writes `forecasts/<t>.csv`,
signs it (`<file>.sig.json`: the study, spec and code hashes, data, model, t, the gate and an expiry from
`[forecast] expires`) and records a `forecast` node. Every invariant of that node is checked before anything runs or
is written. A failed gate is recorded and signed as failed, so a consumer refuses it, and is delivered as an alert.

`fill` records an `outcome` for each forecast whose label is known: its label time has passed and every dataset with
a time has rows known by then. The labels are computed in a child, from the data as last materialised.

Acting on a forecast (a broker, an order) stays outside openlab: that program reads the signed file and checks it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import math
import os
from datetime import datetime, time, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from . import clock, registry
from . import data, fingerprint, runner, schema, splits
from . import spec as _spec
from .durations import parse
from .errors import StudyError
from .metrics import score
from .record import Node
from .view import known_at, rule

if TYPE_CHECKING:
    from . import Study


def forecast(study: "Study", *, now: datetime | None = None, model_version: int | None = None,
             actor: str = "openlab") -> Node:
    """The forecast at the latest decision time at or before `now` (naive UTC; default now), as a recorded,
    signed file; returns its node."""
    from .. import __version__
    from .signing import SIG_SUFFIX, sign_file
    rec = study.record
    spec = _spec.check(study.folder, rec)
    if spec.mode != "temporal":
        raise StudyError("only a temporal study forecasts", "a static study has no decision times")
    model = _model(rec, spec, model_version)
    if "keys" not in model.payload:
        raise StudyError(f"model v{model.payload['version']} does not record its key columns",
                         "it was registered before forward runs existed", "run the study again to register a new model")
    for name in spec.data:
        data.materialise(study, name, refresh=spec.schedule.get("refresh", False))
    fp, frames = fingerprint.fingerprint(study), data.frames(study)
    now = (now.astimezone(timezone.utc).replace(tzinfo=None) if now and now.tzinfo else now) or clock.utc()
    times = splits.decision_times(spec, frames, until=now)
    if not times:
        raise StudyError(f"no decision time at or before {now.isoformat()}")
    t, run = times[-1], rec.get(rec.links(model.id, rel="from")[0][2])
    label_at, rel = t + splits.horizon(spec), f"forecasts/{_stamp(t)}.csv"
    m = model.payload
    links = [("under", (rec.last("amendment") or rec.last("plan")).id), ("by", model.id),
             *[("uses", rec.last("dataset", name=n).id) for n in fp["data"]]]
    body = schema.Forecast(
        t=t.isoformat(), label_at=label_at.isoformat(), prospective=label_at > clock.utc(),
        model={"name": m["name"], "version": m["version"], "run": run.id, "sha256": ""}, spec_sha=fp["spec_sha"],
        code=fp["code"], data=fp["data"], rows=0, file=rel, sha256="", gate={"passed": True, "checks": []},
        signed=rel + SIG_SUFFIX, expires="")
    _valid(rec, body, actor, links)                 # the outputs are not known yet; every other rule is checked now

    blob = registry.Registry(m["name"], arm="study", root=str(rec.dir / "mlflow")).packed(m["name"], m["version"])
    r = runner._Run(study, spec, "forecast", spec.split.get("seed", 0))
    with r.stage(runner._cut(frames, spec.data, t)) as stage:
        (stage / "model.pkl").write_bytes(blob)
        r.child(stage, f"{__name__}:forecast_job", t=t.isoformat(), keys=m["keys"])
        P = pl.read_parquet(stage / "predictions.parquet")
    gate = _gate(P, spec.forecast["gate"])
    csv, made = P.write_csv().encode(), datetime.now().astimezone()
    body = dataclasses.replace(
        body, prospective=label_at > clock.utc(), model={**body.model, "sha256": hashlib.sha256(blob).hexdigest()},
        rows=P.height, sha256=hashlib.sha256(csv).hexdigest(), gate=gate,
        expires=(made + parse(spec.forecast["expires"])).isoformat(timespec="seconds"))
    _valid(rec, body, actor, links)

    rec.put_blob(csv)
    path = study.folder / rel
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    tmp.write_bytes(csv)
    os.replace(tmp, path)
    sign_file(path, {"kind": "openlab-forecast", "study": spec.name, "file": rel, "t": body.t, "spec_sha": body.spec_sha,
                     "code": body.code, "data": body.data, "model": body.model, "gate": gate,
                     "made_at": made.isoformat(timespec="seconds"), "expires_at": body.expires,
                     "openlab_version": __version__})
    node = rec.add(body, actor=actor, links=links)
    if not gate["passed"]:
        from .notify import deliver
        deliver(f"{spec.name} (forecast {body.t})", [{"alert": c["expect"], "interpretation": c.get("error") or
                                                      f"held {c.get('actual', 'false')}"}
                                                     for c in gate["checks"] if not c["ok"]], rec.dir / "outbox.jsonl")
    return node


def fill(study: "Study", *, actor: str = "openlab") -> list[Node]:
    """An outcome node for each forecast without one whose label is known; returns them (none when none is due)."""
    rec = study.record
    spec = _spec.check(study.folder, rec)
    now = clock.utc()
    due = [f for f in rec.find("forecast") if datetime.fromisoformat(f.payload["label_at"]) <= now
           and not rec.links(f.id, rel="of", incoming=True)]
    if not due:
        return []
    for name in spec.data:
        data.materialise(study, name)
    frames = data.frames(study)
    caught = _caught_up(frames, spec.data)
    due = [f for f in due if caught is None or datetime.fromisoformat(f.payload["label_at"]) <= caught]
    if not due:
        return []
    r = runner._Run(study, spec, "fill", spec.split.get("seed", 0))
    with r.stage(frames) as stage:
        r.child(stage, f"{__name__}:label_job", times=sorted({f.payload["t"] for f in due}))
        L = pl.read_parquet(stage / "labels.parquet")
    used = fingerprint.data(rec, set(spec.data))
    uses = [("uses", rec.last("dataset", name=n).id) for n in used]
    out = []
    for f in due:
        P = (pl.read_csv(io.BytesIO(rec.blob(f.payload["sha256"])), schema_overrides={"_key": pl.String})
             .with_columns(pl.col("_key").fill_null("")))           # no key columns: _key is "", which CSV reads as null
        y = L.filter(pl.col("_t") == datetime.fromisoformat(f.payload["t"])).select("_key", "y")
        S = runner._known(P.join(y, on="_key", how="left"), "y")
        buf = io.BytesIO()
        S.write_parquet(buf)
        out.append(rec.add(schema.Outcome(forecast=f.id, label_at=f.payload["label_at"],
                                          metrics=metrics_of(rec, spec, S), n=S.height, data=used,
                                          labels=rec.put_blob(buf.getvalue())),
                           actor=actor, links=[("of", f.id), *uses]))
    return out


def schedule(study: "Study", action: str = "status") -> dict:
    """The study's `[schedule] cron` line in the user's crontab: `status`, `on` (install or update) or `off`."""
    from . import schedule as SC
    out = crontab(study, action, "schedule", SC.for_study, "7 * * * *")
    last = study.record.last("forecast")
    out["last_forecast"] = last and {"id": last.id, "at": last.at, "t": last.payload["t"],
                                     "gate": last.payload["gate"]["passed"]}
    return out


def crontab(study: "Study", action: str, section: str, line, example: str) -> dict:
    """`status`, `on` (install or update) or `off` of the crontab line `line(folder, cron)` makes for the study's
    `[section] cron`; returns its status."""
    from . import schedule as SC
    cron = getattr(study.spec, section).get("cron")
    root, doc, job, log = line(study.folder, cron or "")
    if action == "on":
        if not cron:
            raise StudyError(f"study.toml has no [{section}] cron", "there is nothing to install",
                             f'add cron = "{example}" under [{section}] (by amendment once sealed)')
        SC.on(root, doc, job, log)
    elif action == "off":
        SC.off(root, doc)
    elif action != "status":
        raise StudyError(f"no schedule action `{action}`", "status, on or off")
    return SC.status(root, doc, job if cron else None, log)


def _model(rec, spec, version: int | None) -> Node:
    """The pinned model: `version`, else `[forecast] model`; "latest" is the newest model of a research run."""
    want = version or spec.forecast["model"]
    if want == "latest":
        found = next((m for m in reversed(rec.find("model"))
                      if rec.get(rec.links(m.id, rel="from")[0][2]).payload["segment"] == "research"), None)
    else:
        found = rec.last("model", version=want)
    if found is None:
        raise StudyError("no model to forecast with" if want == "latest" else f"no model version {want}",
                         fix="a research run registers one: `openlab study run <folder>`")
    return found


def _valid(rec, body: schema.Forecast, actor: str, links: list) -> None:
    """Every rule the record applies to `body` when it is written, checked now."""
    schema.check(rec, schema.body(body), actor, [(rel, rec.get(i)) for rel, i in links])


def _stamp(t: datetime) -> str:
    return f"{t:%Y-%m-%d}" if t.time() == time() else f"{t:%Y-%m-%dT%H%M%S}"


def _gate(P: pl.DataFrame, checks: list) -> dict:
    """Each gate predicate over the predictions (a table named `t`), as `%%expect` lines are checked."""
    from .checks import check_frame
    out = [check_frame(P, c) for c in checks]
    return {"passed": all(c["ok"] for c in out), "checks": out}


def _caught_up(frames: dict, specs: dict) -> datetime | None:
    """The time by which every dataset with a time has rows known (None when none has a time)."""
    ends = [known_at(df, specs[n]).max() for n, df in frames.items() if rule(specs[n]) is not None]
    ends = [e for e in ends if e is not None]
    return min(ends) if ends else None


def metrics_of(rec, spec, S: pl.DataFrame) -> dict:
    """`_metrics`, in the sandbox when a metric is the study's own."""
    if runner.own_metrics(spec):
        return runner.sandboxed(rec, spec, f"{__name__}:_metrics", {"S": S})
    return _metrics(spec, S)


def _metrics(spec, S: pl.DataFrame) -> dict:
    """The spec's metrics on one forecast's labelled rows, without intervals."""
    if S.height == 0:
        return {m: {"value": None, "reason": "no row of this forecast has a label"} for m in spec.evaluate["metrics"]}
    y, pred, out = S["y"].to_numpy(), runner._pred(S, ""), {}
    for m in spec.evaluate["metrics"]:
        try:
            v = score(m, y, pred, prediction=spec.evaluate.get("prediction"))
            out[m] = {"value": v} if math.isfinite(v) else {"value": None, "reason": f"{m} is undefined on these rows"}
        except (StudyError, ValueError) as e:
            out[m] = {"value": None, "reason": str(e)}
    return out


# --- the children ------------------------------------------------------------------------------------------
def forecast_job(stage: Path, *, spec: dict, seed: int, t: str, keys: list) -> Path:
    """The pinned model's predictions at t, on features keyed as its training rows were; the model is unpickled
    here, in the sandbox."""
    import pickle
    w = runner._Walk(stage, spec, seed)
    w.keys = keys
    out = stage / "predictions.parquet"
    w.predict({"": pickle.loads((stage / "model.pkl").read_bytes())}, datetime.fromisoformat(t)).write_parquet(out)
    return out


def label_job(stage: Path, *, spec: dict, seed: int, times: list) -> Path:
    """Each decision's label at t + horizon, keyed as its features are: _t, _key, y."""
    w = runner._Walk(stage, spec, seed)
    out = stage / "labels.parquet"
    pl.concat([w.labelled(t).select("_t", "_key", pl.col("__y").alias("y")) for t in runner._times(times)],
              how="vertical_relaxed").write_parquet(out)
    return out
