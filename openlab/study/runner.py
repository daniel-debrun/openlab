"""A study run, from the sealed spec to a recorded result.

The parent (this process) seals and checks the spec, materialises the data, stages what the segment may see, runs
the study's code in sandboxed children, replays sampled steps on data cut at their time (the truncation audit),
scores the predictions, and writes the run, result and model nodes through the schema. The study's code only runs
in a child: the parent reads back parquet files and a model the child packed, and never unpickles it.

The children's entries:
- `temporal_job` walks a segment's steps: features at t, labels at t + horizon, Model and each baseline refitted
  where a step refits (seeded with the run seed), each step's predictions, then act() per decision.
- `static_job` fits on one fold's training rows and predicts its validation rows (or the holdout rows).
- `replay_job` replays one temporal step on data cut at its time.
- `score_job` scores with the study's own metrics (metrics.py, as sealed, runs only in a child: `sandboxed`).

A label may have several outputs (label() returns y as a list or array per row, e.g. the next 96 steps of a
series): y is then a fixed-width array column, predict returns a 2-D array (rows x outputs), and a metric gets 2-D
arrays. The rows stay the unit of the bootstrap.

A temporal step whose prediction is a distribution over several outputs (array columns of mu/sigma, qX, mK and wK:
e.g. 100 analogues of the next 720 steps of 7 series) is too wide to keep: the walk writes each step's predictions,
with its labels, to disk, and this process reduces each step file while the walk goes on (and deletes it) to the
predictive mean (`pred`, float32) and the per-row statistics of every metric the study scores (`@<metric>.<j>`,
the metric's rows()). The result, its
intervals and later paired comparisons read those; a metric not declared before the run cannot be scored on it. The
truncation audit reduces each replay the same way before comparing.

act(pred_row, view, t) gets the decision's prediction row as a dict (with several rows, a dict of lists) and the
label view; its value is kept once per decision. It may return a dict of named values instead: the first is the
policy's value, and the others are kept per decision (columns act.<name>) and summarised by their mean.

A Model with no fit (a fixed rule, or a model trained elsewhere) is used as is from the first decision.
"""

from __future__ import annotations

import bisect
import contextlib
import fcntl
import hashlib
import importlib
import io
import json
import math
import os
import random
import secrets
import re
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from . import registry
from . import admission, baselines, data, fingerprint, ledger, machine, sandbox, schema, splits
from . import spec as _spec
from .durations import parse
from .errors import StudyError
from .inference import interval, paired
from .metrics import METRICS, _point, nominal, score
from .record import Node, canonical
from .splits import Step
from .view import Timeline, known_at, rule

if TYPE_CHECKING:
    from . import Study

ID = ("_t", "_key", "_row", "y")                   # the columns of a predictions frame that are no prediction
SPEC_CHOICES = ("data", "target", "split", "evaluate")
TOL = 1e-9                                         # the audit's relative tolerance
WORKERS = int(os.environ.get("OPENLAB_AUDIT_WORKERS") or machine.cores())   # sandboxed children at once
_ROOM = threading.Lock()


def _available() -> tuple:
    """(available, total) memory in bytes, under the container's limit when it sets one (openlab.study.machine)."""
    return machine.memory()


def _room(wait: float = 600.0) -> None:
    """Wait (up to `wait` s) until a fifth of the memory, and at least 1.5 GB, is available, one starter at a time
    (with a pause after it when memory is tight, so the child it starts takes its share before the next one looks):
    an audit replay is started only when it fits beside the walk. The audit replays as many steps either way."""
    with _ROOM:
        end = time.monotonic() + wait
        while time.monotonic() < end:
            avail, total = _available()
            if avail is None or avail >= max(1.5e9, total / 5):
                break
            time.sleep(1.0)
        avail, total = _available()
        if avail is not None and avail < 0.4 * total:            # tight: let the last child take its share first
            time.sleep(2.0)


class _Timeline:
    """A run on the timeline of the agent that started it (`run`, then `run_end`), when an agent of the
    workspace's control store did: `openlab watch` shows the agent running a trial, not stuck, while it waits."""

    def __init__(self, study, segment: str, note: str, key: str):
        self.store = self.agent = None
        try:
            from ..agents.identity import current
            from ..agents.store import Store, root_of
            agent, root = current(), root_of(study.folder)
            if (root / ".openlab" / "control.db").is_file() and Store(root).get(agent):
                self.store, self.agent = Store(root), agent
                self.store.event(agent, "run", summary=f"{segment} run of {study.folder.name}: {note}"[:300],
                                 payload={"pid": os.getpid(), "key": key, "study": str(study.folder)})
        except Exception:                           # noqa: BLE001 -- the timeline never stops a run
            self.store = None

    def progress(self, line: str) -> None:
        if self.store is None:
            return
        try:
            self.store.event(self.agent, "run_progress", summary=line)
        except Exception:                           # noqa: BLE001
            pass

    def end(self, summary: str, ok: bool) -> None:
        if self.store is None:
            return
        try:
            self.store.event(self.agent, "run_end", summary=summary, payload={"ok": ok, "pid": os.getpid()})
        except Exception:                           # noqa: BLE001
            pass


# --- the parent ----------------------------------------------------------------------------------------------
def run(study: "Study", *, note: str = "", actor: str = "agent", segment: str = "research",
        look: str | None = None, code_dir: Path | None = None, seed: int | None = None) -> Node:
    """Run the study on `segment` and return its result node. A research run with the key of an earlier one
    returns that run's result; a holdout run needs `look`, the person's look node, written before. `code_dir`
    holds model.py when it is not the study folder."""
    return execute(study, note=note, actor=actor, segment=segment, look=look, code_dir=code_dir, seed=seed)[0]


def execute(study: "Study", *, note: str = "", actor: str = "agent", segment: str = "research",
            look: str | None = None, code_dir: Path | None = None, branch: str = "",
            seed: int | None = None, parent: str = "", op: str = "", refs: tuple = (), cites: tuple = ()) -> tuple:
    """`run`, and what the run cost: (result node, trial), trial = {cached, key, k, budget, changed, summary}.
    `seed`: the model's seed for this run (default the spec's split.seed), e.g. one run per seed of a paper that
    averages over seeds; the split's folds and holdout keep split.seed, and the seed is part of the run's key.
    `summary` is one line: "cached: no trial spent (key ...)", or "trial k of N: new key because X changed (a→b)"
    naming what differs from the closest earlier research run. `op` and `parent` place the run in the search tree
    (schema.RUN_OPS: a draft has no parent; improve, debug and fuse name one; a fuse `refs` runs of other branches);
    without them it branches from the branch's (or study's) last run. A cached run keeps the place it was recorded
    at."""
    if segment not in ("research", "holdout"):
        raise StudyError(f"no segment `{segment}`", "research or holdout")
    t0, rec = time.monotonic(), study.record
    if cites:                                  # the lessons it applies must exist
        from .lessons import resolve
        for c in cites:
            try:
                _, ln = resolve(study, c)
            except Exception:                                        # noqa: BLE001
                ln = None
            if ln is None or ln.kind != "lesson":
                raise StudyError(f"no lesson `{c}`", "a run cites lessons on record", "`openlab lessons find` lists them")
    if parent or op or refs:                   # the run's place in the search, refused before anything runs
        schema._run_op(op, segment, {"branches_from": [_run_node(rec, parent)] if parent else [],
                                     "refs": [_run_node(rec, x) for x in refs]})
    ledger.not_a_copy(study)
    was_sealed = _spec.sealed(rec) is not None
    spec = _spec.check(study.folder, rec) if was_sealed else _spec.load(study.folder)
    _preflight(spec)
    for name in spec.data:
        data.materialise(study, name)
    ledger.same_question(study, spec)          # before the seal: a refused first run leaves study.toml unsealed
    if not was_sealed:
        _spec.seal(study.folder, rec, actor)
        spec = _spec.check(study.folder, rec)
    if segment == "holdout" and look is None:
        raise StudyError("a holdout run needs a look", "a person records the look before the holdout is touched",
                         "take it with `openlab study holdout`")
    with fingerprint.staged(study, code_dir) as snap:
        fp = fingerprint.fingerprint(study, snap)
        from .preflight import refuse_red
        refuse_red(rec, fp["code"])                    # this exact code's latest module check is red: refused here
        given, seed = seed, spec.split.get("seed", 0) if seed is None else int(seed)
        key = fingerprint.key(fp, segment, seed)
        with _in_flight(rec, key if segment == "research" else None):
            if segment == "research":
                done = _cached(rec, key) or _cached_legacy(rec, fp, seed)
                if done is not None:
                    return done, _trial(rec, spec, fp, key, cached=done)
                _budget(study, spec, key)
                why = _changed(rec, fp)                # before this run is on record
            timeline = _Timeline(study, segment, note, key)
            lease = contextlib.ExitStack()
            try:
                needs = admission.needs_of(spec, rec)
                adm = lease.enter_context(admission.admit(needs, label=f"{study.folder.name} {segment}",
                                                          agent=timeline.agent or "", on_wait=_waiting(timeline)))
                r = _Run(study, spec, segment, seed, snap)
                r.gpus = lease.enter_context(machine.lease_gpus(needs.gpus))
                r.waited = adm["waited"]
                if spec.sandbox.get("cores"):                # a declared share caps the run's children
                    r.workers = min(WORKERS, needs.cores)
                r.timeline = timeline
                r.registry = _Registry(rec.dir, _model_name(spec))    # MLflow opens while the children run
                out = r.temporal() if spec.mode == "temporal" else r.static()
                kept = segment == "holdout" and spec.split.get("labels") == "person"
                ev = None if kept else r.evaluate(_scored(spec, segment, out.P))   # kept: scored by the person
                node = _record(r, out, ev, fp, key, note=note, actor=actor, look=look,
                               seconds=time.monotonic() - t0, branch=branch, seed=given, parent=parent, op=op,
                               refs=tuple(refs), cites=tuple(cites))
            except BaseException as e:
                timeline.end(f"failed: {getattr(e, 'error', None) or e}"[:300], ok=False)
                raise
            finally:
                lease.close()                          # the GPUs are held for the whole run, released also on failure
            m = spec.evaluate["metrics"][0]
            v = (node.payload.get("metrics", {}).get(m) or {}).get("value")
            if node.kind == "run":
                print(f"holdout predictions recorded ({node.id}); the person holds the labels: score them on their "
                      f"machine with `openlab study score HOST {study.folder.name} --labels FILE`", file=sys.stderr)
            timeline.end(f"{node.id}: {m} " + (f"{v:.4g}" if isinstance(v, (int, float)) else "n/a")
                         + f" after {(time.monotonic() - t0) / 60:.1f} min", ok=True)
            if segment != "research":
                return node, {"cached": False, "key": key, "summary": f"holdout run under look {look}: no research "
                                                                      "trial spent"}
            return node, _trial(rec, spec, fp, key, changed=why)


def _run_node(rec, ref: str) -> Node:
    """The research run `ref` names: run:<id>, or a result of it (result:<id>)."""
    try:
        n = rec.get(ref)
    except Exception:                                               # noqa: BLE001
        n = None
    if n is not None and n.kind == "result":
        n = rec.get(rec.links(n.id, rel="of")[0][2])
    if n is None or n.kind != "run":
        raise StudyError(f"no run `{ref}` in this study", "a parent or ref is a run:<id> (or its result:<id>)",
                         "list them with `openlab study runs`")
    return n


def _box(gpus: list | None = None, waited: float = 0.0) -> dict:
    """The machine a run ran on: its name, the cores and memory openlab used there, the GPUs the run held, and how
    long it waited to be admitted (openlab.study.admission)."""
    import socket
    m = machine.describe()
    return {"host": socket.gethostname(), "cores": m["cores"], "memory_gb": m["memory_gb"],
            **({"gpus": list(gpus)} if gpus else {}), **({"waited_s": round(waited, 1)} if waited >= 0.5 else {})}


def _waiting(timeline: "_Timeline"):
    """What a run says while it waits to be admitted: a line on stderr when the wait starts (and each 10 minutes),
    and on the agent's timeline each minute, so `openlab watch` shows it queued, not stuck."""
    said = {"stderr": -1.0, "timeline": -1.0}

    def on_wait(why: str, seconds: float) -> None:
        if said["stderr"] < 0 or seconds - said["stderr"] >= 600:
            print(f"queued ({seconds / 60:.0f} min): {why}; `openlab box queue` shows the machine's queue",
                  file=sys.stderr, flush=True)
            said["stderr"] = seconds
        if said["timeline"] < 0 or seconds - said["timeline"] >= 60:
            timeline.progress(f"queued {seconds / 60:.0f} min: {why}")
            said["timeline"] = seconds
    return on_wait


def _changed(rec, fp: dict) -> dict:
    """What makes this fingerprint a new key: its differences from the closest earlier research run (the fewest
    components differing; the latest among equals). {} for the study's first run."""
    best = None
    for r in rec.find("run", segment="research"):
        diff = fingerprint.changes(r.payload, fp)
        if best is None or len(diff) <= len(best[1]):
            best = (r, diff)
    if best is None:
        return {}
    return {"run": best[0].id, "components": [c for c, _ in best[1]], "details": dict(best[1])}


def _trial(rec, spec, fp: dict, key: str, cached: Node | None = None, changed: dict | None = None) -> dict:
    """The trial line of a research run: cached, or which trial it was and why its key is new."""
    budget = spec.budget.get("trials")
    if cached is not None:
        run = rec.get(rec.links(cached.id, rel="of")[0][2])
        return {"cached": True, "key": run.payload["key"], "run": run.id,
                "summary": f"cached: no trial spent (key {run.payload['key'][:12]}): the code, data, spec and environment of {run.id}"
                           f"{' (' + repr(run.payload.get('note')) + ')' if run.payload.get('note') else ''}"}
    sh = ledger.shared(rec.folder, rec)
    k, changed = sh["trials"], changed or {}
    shared_with = ", ".join(f"{v['trials']} in {n}" for n, v in sh["by"].items())
    head = f"trial {k} of {budget}" + (f" ({shared_with})" if shared_with else "")
    comps = changed.get("components", [])
    if not changed:
        why = "the study's first run"
    elif not comps:
        why = f"new key, though nothing differs from {changed['run']} that openlab hashes"
    elif set(comps) <= {"env packages", "python"}:
        why = ("new key because only the environment changed: " +
               "; ".join(f"{c} ({changed['details'][c]})" for c in comps) +
               f"; the code, data and spec are those of {changed['run']}")
    else:
        why = ("new key because " + " and ".join(f"{c} ({changed['details'][c]})" for c in comps) +
               f" changed since {changed['run']}")
    return {"cached": False, "key": key, "k": k, "budget": budget, "changed": changed, "summary": f"{head}: {why}"}


def _runs_dir(rec) -> Path:
    d = rec.dir / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _held(path: Path) -> bool:
    """Whether another process holds the lock file at `path` (a run in flight)."""
    try:
        with open(path) as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return True
            fcntl.flock(f, fcntl.LOCK_UN)
            return False
    except OSError:
        return False


@contextmanager
def _in_flight(rec, key: str | None):
    """Hold .openlab/runs/<key>.lock while a research run is in flight. A second run of the same key waits for the
    first and then finds its result cached (no second trial, no duplicate run); runs of different keys proceed side
    by side, each on its own snapshot, and `_budget` counts the ones in flight."""
    if key is None:
        yield
        return
    path = _runs_dir(rec) / f"{key}.lock"
    with open(path, "a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print(f"openlab: an identical run (key {key[:12]}) is in flight; waiting for it", file=sys.stderr,
                  flush=True)
            fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
            path.unlink(missing_ok=True)


def in_flight(rec, but: str = "") -> list:
    """The keys of the research runs in flight in this study (their lock is held), except `but`."""
    d = rec.dir / "runs"
    if not d.is_dir():
        return []
    return [p.stem for p in sorted(d.glob("*.lock")) if p.stem != but and _held(p)]


def _preflight(spec) -> None:
    """Refuse a misnamed metric, one that cannot score the declared prediction, or a missing model registry, before
    any trial is spent."""
    from importlib.util import find_spec
    from .custody import check_spec
    check_spec(spec)
    if find_spec("mlflow") is None:
        raise StudyError("a run registers its model, and mlflow is not installed", "nothing ran",
                         'pip install "openlab-research[study]"')
    kind = spec.evaluate.get("prediction")
    for m in [*spec.evaluate["metrics"], *spec.evaluate.get("policy_metrics", [])]:
        if METRICS.get(m) is None:
            raise StudyError(f"study.toml: no metric named {m!r}", fix=f"one of {', '.join(METRICS)}")
    bad = [m for m in spec.evaluate["metrics"] if kind and kind not in METRICS[m].prediction]
    if bad:
        raise StudyError(f"{', '.join(bad)} cannot score a {kind} prediction",
                         "; ".join(f"{m} reads {', '.join(METRICS[m].prediction)}" for m in bad),
                         "change evaluate.metrics or evaluate.prediction")


def _cached(rec, key: str) -> Node | None:
    for r in rec.find("run", key=key, segment="research"):
        of = rec.links(r.id, rel="of", incoming=True)
        if of:
            return rec.get(of[0][0])
    return None


def _cached_legacy(rec, fp: dict, seed: int) -> Node | None:
    """A run recorded before the key hashed only the packages the code can import (its env has no `packages`),
    with this run's spec, code and data under the same installed packages: the same run, so no trial."""
    old = [r for r in rec.find("run", segment="research", spec_sha=fp["spec_sha"])
           if "packages" not in r.payload["env"] and r.payload["code"] == fp["code"] and r.payload["data"] == fp["data"]]
    if not old:
        return None
    legacy = {**fp, "env": {**fp["env"], "packages_sha": fingerprint.all_packages_sha()}}
    return _cached(rec, fingerprint.key(legacy, "research", seed))


def _budget(study: "Study", spec, key: str = "") -> None:
    """Refuse a new trial once the budget is spent: the trials of this study and of every study sharing its ledger
    (openlab.study.ledger), and the runs in flight here."""
    rec = study.record
    sh = ledger.shared(study.folder, rec)
    used, budget = sh["trials"], spec.budget["trials"]
    others = ", ".join(f"{v['trials']} in {k}" for k, v in sh["by"].items())
    if used >= budget:
        raise StudyError(f"the trial budget is spent: {used} of {budget} trials" + (f" ({others})" if others else ""),
                         "every distinct run and declared variant is a counted trial, so the number of tries stays "
                         "honest",
                         "a person raises budget.trials (a person-only field) in study.toml and signs it: "
                         'openlab study amend STUDY --person --reason "..."')
    flying = [k for k in in_flight(rec, but=key) if not rec.find("run", key=k, segment="research")]
    if used + len(flying) >= budget:
        raise StudyError(f"the last trials are taken by runs in flight: {used} of {budget} used, {len(flying)} "
                         "running now", "each run in flight will count a trial when it is recorded",
                         "wait for them (openlab study runs shows them once recorded), or a person raises budget.trials "
                         'in study.toml and signs it: openlab study amend STUDY --person --reason "..."')


@dataclass
class _Out:
    P: pl.DataFrame          # one row per prediction: _t or _row, _key, y, the model's columns, b:<baseline>:<col>, act
    model: tuple             # (pickle, pack info) of the final fit, as registry.pack made them in the child
    params: dict             # model.PARAMS
    stdout: str
    audit: dict
    skipped: int = 0


class _Run:
    """One run's parent side: stages, children and the truncation audit."""

    def __init__(self, study: "Study", spec, segment: str, seed: int, code_dir: Path | None = None):
        self.study, self.spec, self.segment, self.seed = study, spec, segment, seed
        self.code_dir = Path(code_dir or study.folder)      # what the children import model.py from
        self.workers = WORKERS                              # sandboxed children at once ([sandbox] cores caps it)
        self.sandboxes: set = set()
        self._local, self._servers = threading.local(), []   # the audit's replay servers, one per worker thread

    def frames(self) -> dict:
        """Every dataset of the spec, loaded and checked against its sha."""
        return data.frames(self.study)

    def labels(self, name: str, target: str, df: pl.DataFrame, hold: np.ndarray) -> list:
        """A static study's holdout labels, from the vault."""
        return _vault(self.study, self.spec, name, target, df, hold)

    @contextmanager
    def stage(self, frames: dict):
        """A fresh stage dir holding `frames` as data/<name>.parquet; deleted afterwards."""
        root = self.study.record.dir / "stage"
        root.mkdir(parents=True, exist_ok=True)
        d = Path(tempfile.mkdtemp(prefix=f"{self.segment}-", dir=root))
        try:
            (d / "data").mkdir()
            for name, df in frames.items():
                df.write_parquet(d / "data" / f"{name}.parquet")
            yield d
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def child(self, stage: Path, entry: str, *, must: bool = True, **args) -> sandbox.Result:
        """Run `entry` (a function of this module, or "module:function") in a sandboxed child."""
        entry = entry if ":" in entry else f"{__name__}:{entry}"
        res = sandbox.run({"entry": entry, "args": {"spec": self.spec.raw, "seed": self.seed, **args}},
                          study_dir=self.code_dir, stage_dir=stage,
                          network=bool(self.spec.sandbox.get("network", False)), timeout=self.spec.sandbox.get("timeout"),
                          threads=threads(self.spec), gpus=getattr(self, "gpus", None))
        self.sandboxes.add(res.sandbox)
        if must and not res.ok:
            code = "metrics.py" if entry.endswith(":evaluate_job") else "model.py"
            raise StudyError(f"{code} failed in {entry}", res.stderr[-3000:],
                             f"fix {code} and run again; a failed run is not a trial")
        return res

    def evaluate(self, P: pl.DataFrame) -> dict:
        """The result body (`_evaluate`): here, or in a sandboxed child when a metric it scores is the study's
        own, since the study's code runs nowhere else."""
        if not own_metrics(self.spec):
            return _evaluate(self.spec, P, self.seed)
        return sandboxed(self.study.record, self.spec, f"{__name__}:_evaluate", {"P": P}, seed=self.seed)

    def temporal(self) -> _Out:
        """The segment's walk in one child and, side by side with it, the truncation audit's replays (in parallel,
        at most WORKERS children at once)."""
        span = splits.segments(self.spec)[self.segment]
        if span is None:
            raise StudyError("the study has no holdout segment", fix="add split.holdout = [start, end]")
        full = self.frames()
        seen = _cut(full, self.spec.data, span[1])
        times = splits.decision_times(self.spec, seen)
        steps = splits.temporal_steps(self.spec, times, self.segment)
        times = [t for t in times if t <= steps[-1].t]
        pool = ThreadPoolExecutor(max_workers=max(1, self.workers - 1))
        try:
            plan = self._audit_plan(steps)
            early = {i: pool.submit(self._replay, full, times, steps[i]) for i in plan["order"][:plan["n"]]}
            with self.stage(seen) as stage:
                reducer = _Reducer(self.spec, stage / "parts")      # a wide distribution, reduced as it is written
                progress = _Progress(stage / "progress.json", getattr(self, "timeline", None))
                try:
                    res = self.child(stage, "temporal_job", times=_iso(times), steps=[_step_out(s) for s in steps])
                finally:
                    progress.stop()
                    parts = reducer.finish()
                P = pl.read_parquet(stage / "predictions.parquet")
                if parts is not None:
                    P = _aligned(P, parts)
                model, params = _packed(stage)
            audit = self.audit(full, P, times, steps, plan=plan, early=early, pool=pool)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            self._close_servers()
        return _Out(P, model, params, res.stdout, audit, len(steps) - P["_t"].n_unique())

    def _audit_plan(self, steps: list) -> dict:
        """The audit's draw: a fresh seed (recorded, so the model cannot know which steps are replayed) and the
        order it puts the steps in. The audit replays the first `audit.n` steps in that order that predicted."""
        draw = secrets.randbits(32)
        return {"n": self.spec.audit.get("n", 20), "seed": draw,
                "order": [int(i) for i in np.random.default_rng(draw).permutation(len(steps))]}

    def _replay(self, full: dict, times: list, step: Step) -> tuple:
        """(ok, stderr, the replay's predictions or None) of one step replayed on data cut at its time, in this
        thread's server (started on first use: one import per server, a fresh fork per replay). It starts only once
        the machine has memory to spare (`_room`), so a wide study's replays queue rather than exhaust it."""
        _room()
        srv = getattr(self._local, "server", None)
        if srv is None or not srv.alive():
            root = self.study.record.dir / "stage"
            root.mkdir(parents=True, exist_ok=True)
            srv = self._local.server = sandbox.Server(
                study_dir=self.code_dir, stage_dir=tempfile.mkdtemp(prefix="replay-", dir=root),
                network=bool(self.spec.sandbox.get("network", False)), timeout=self.spec.sandbox.get("timeout"),
                preload=fingerprint.imports(self.code_dir), threads=threads(self.spec),
                gpus=getattr(self, "gpus", None))
            self._servers.append(srv)
        sub = srv.clear()
        (sub / "data").mkdir()
        for name, df in _cut(full, self.spec.data, step.t).items():
            df.write_parquet(sub / "data" / f"{name}.parquet")
        res = srv.run({"entry": f"{__name__}:replay_job",
                       "args": {"spec": self.spec.raw, "seed": self.seed, "step": _step_out(step),
                                "times": _iso([t for t in times if t <= step.t])}}, sub)
        self.sandboxes.add(res.sandbox)
        answered = (sub / "result.json").is_file()
        out = pl.read_parquet(sub / "predictions.parquet") if res.ok else None
        shutil.rmtree(sub, ignore_errors=True)
        if not res.ok and not answered:         # the server failed, not the model: the replay again, in a child of its own
            with self.stage(_cut(full, self.spec.data, step.t)) as stage:
                res = self.child(stage, "replay_job", must=False, times=_iso([t for t in times if t <= step.t]),
                                 step=_step_out(step))
                out = pl.read_parquet(stage / "predictions.parquet") if res.ok else None
        return res.ok, res.stderr, out

    def _close_servers(self) -> None:
        for srv in self._servers:
            srv.close()
            shutil.rmtree(srv.stage, ignore_errors=True)
        self._servers.clear()

    def audit(self, full: dict, P: pl.DataFrame, times: list, steps: list, *, plan: dict, early: dict,
              pool: ThreadPoolExecutor) -> dict:
        """The truncation audit: the first `audit.n` steps of the drawn order that predicted are replayed in
        children whose data is cut at the step's time (a uniform draw of n of the steps that predicted). Replays
        start beside the main child, on the first n steps of the order; a drawn step that turns out not to have
        predicted is replaced by the next one that did. A replay that predicts otherwise, or fails, is look-ahead."""
        n, done = plan["n"], set(P["_t"].unique().to_list())
        pick = [i for i in plan["order"] if steps[i].t in done][:n]
        for i, f in early.items():
            if i not in pick:
                f.cancel()
        jobs = {i: early.get(i) or pool.submit(self._replay, full, times, steps[i]) for i in pick}
        bad = []
        for i in sorted(pick, key=lambda i: steps[i].t):
            ok, stderr, replay = jobs[i].result()
            step = steps[i]
            run = P.filter(pl.col("_t") == step.t)
            if ok and _reduced(run) and replay.height == run.height and replay.height:
                ids = run.sort("_key", maintain_order=True)
                replay = _reduce(self.spec, replay.sort("_key", maintain_order=True).drop("_t", "_key")
                                 .hstack(ids.select("_t", "_key", "y")))
            why = (_differs(run, replay) if ok else f"the replay failed: {stderr.strip()[-300:]}")
            if why:
                bad.append({"t": step.t.isoformat(), "why": why})
        verdict = "look-ahead" if bad else "clean" if len(pick) else "unchecked"
        return {"n": n, "seed": plan["seed"], "checked": len(pick), "mismatches": bad, "verdict": verdict}

    def static(self) -> _Out:
        """Research: one child per fold (that fold's training rows and validation features only), then one fit on
        every research row for the model. Holdout: one fit on the research rows predicting the holdout rows, scored
        against the labels in the vault."""
        spec = self.spec
        if len(spec.data) != 1:
            raise StudyError("a static study reads one table", f"data has {', '.join(spec.data)}", "keep one [data.*]")
        (name,), target = spec.data, spec.target.get("column")
        df = self.frames()[name]
        if target not in df.columns:
            raise StudyError(f"study.toml: `target.column` = {target!r} is not a column of data.{name}",
                             f"its columns: {', '.join(df.columns)}")
        df = outputs(df, target, f"data.{name}.{target}")
        research, hold = splits.carve(df, spec)
        kept = spec.split.get("labels") == "person"        # the person keeps them: none here, and no vault
        labels = [None] * len(hold) if kept else self.labels(name, target, df, hold)
        R = df[research]
        insample = spec.split.get("scheme") == "insample"
        if self.segment == "research" and insample:  # the paper's protocol: fit on every row, score those rows
            jobs = [(R, R, research, R[target])]
        elif self.segment == "research":
            jobs = [(R[tr], R[va], research[va], R[va][target]) for tr, va in splits.static_folds(R, spec, self.seed)]
            jobs.append((R, None, None, None))
        elif len(hold):
            jobs = [(R, df[hold], hold, pl.Series("y", labels, dtype=df[target].dtype))]
        else:
            raise StudyError("the study has no holdout", fix="add split.fraction (e.g. 0.2) before the first run")
        def one(i: int) -> tuple:                   # each fold in a child of its own, at most WORKERS at once
            train, valid, rows, y = jobs[i]
            last = i == len(jobs) - 1
            with self.stage({"train": train} if valid is None else {"train": train, "valid": valid.drop(target)}) as st:
                res = self.child(st, "static_job", target=target, pack=last)
                part = (pl.read_parquet(st / "predictions.parquet").with_columns(_row=pl.Series(rows), y=y)
                        if valid is not None else None)
                return part, _packed(st) if last else None, res.stdout
        with ThreadPoolExecutor(max_workers=max(1, min(self.workers, len(jobs)))) as pool:
            done = list(pool.map(one, range(len(jobs))))
        parts = [p for p, _, _ in done if p is not None]
        model, params = done[-1][1]
        return _Out(pl.concat(parts), model, params, "".join(o for _, _, o in done),
                    {"n": 0, "checked": 0, "mismatches": [],
                     "verdict": "in-sample" if insample and self.segment == "research" else "separated"})


def _vault(study: "Study", spec, name: str, target: str, df: pl.DataFrame, hold: np.ndarray) -> list:
    """The holdout labels, in the vault outside the study from the first run on (keyed by the plan's id); a later
    carve must hold the same rows and labels."""
    if not len(hold):
        return []
    from . import vault
    root, doc, labels = study.folder, study.record.find("plan")[0].id, df[target].gather(hold).to_list()
    try:
        vault.create(root, doc, "holdout", {"labels": labels, "rows": hold.tolist(), "train": "research",
                                            "test": "holdout"}, frame=name, target=target,
                     metric=spec.evaluate["metrics"][0], budget=max(1, spec.budget["holdout_looks"]),
                     fraction=spec.split.get("fraction"), seed=spec.split.get("seed"))
    except FileExistsError:
        pass
    except ValueError as e:
        raise StudyError(f"the holdout labels cannot go to the vault: {e}") from None
    kept = vault.load(root, doc, "holdout")
    if kept["rows"] != hold.tolist() or kept["labels_sha"] != vault.labels_sha(labels):
        raise StudyError("the holdout carve no longer matches the labels in the vault",
                         "the data or the split changed after the first run, so research would hold held-out rows",
                         "restore the data the study was first run with")
    return kept["labels"]


def _packed(stage: Path) -> tuple:
    meta = json.loads((stage / "model.json").read_text(encoding="utf-8"))
    return ((stage / "model.pkl").read_bytes(), meta["pack"]), meta["params"]


def _cut(frames: dict, specs: dict, until: datetime) -> dict:
    """Each dataset's rows known by `until`; a table with no time is whole."""
    return {n: df if rule(specs[n]) is None else df.filter(known_at(df, specs[n]) <= until) for n, df in frames.items()}


def _differs(run: pl.DataFrame, replay: pl.DataFrame) -> str:
    """"" if the replay predicted what the run did at this step: the same keys and values (relative tolerance TOL,
    NaN equal to NaN); else what differs."""
    if run.height == replay.height == 0:
        return ""
    a, b = run.sort("_key", maintain_order=True), replay.sort("_key", maintain_order=True) if replay.height else replay
    if a.height != b.height or a["_key"].to_list() != b["_key"].to_list():
        return f"the run predicted {a.height} rows here, the replay {b.height}"
    cols = _cols(a, "") + [c for c in a.columns if c.startswith("b:model:") and c in b.columns]
    for c in cols:
        if c not in b.columns:
            return f"the replay has no column {c}"
        x, z = a[c].to_numpy(), b[c].to_numpy()
        same = (np.isclose(x, z, rtol=TOL, atol=0, equal_nan=True) if x.dtype.kind in "fiu" and z.dtype.kind in "fiu"
                else x == z)
        if not np.all(same):
            return f"{c} differs on {int(np.size(same) - np.sum(same))} of {len(x)} rows"
    return ""


# --- wide distributions, reduced step by step ------------------------------------------------------------------
_DIST = re.compile(r"mu|sigma|lower|upper|q0?\.\d+|m\d+|w\d+")


def _wide(P: pl.DataFrame) -> bool:
    """Whether one step's predictions hold a distribution over several outputs (an array column of mu/sigma,
    lower/upper, qX, mK or wK), too wide to keep for a whole walk."""
    return any(c not in ID and isinstance(P.schema[c], (pl.Array, pl.List)) and _DIST.fullmatch(c.rsplit(":", 1)[-1])
               for c in P.columns)


def _reduced(P: pl.DataFrame) -> bool:
    return any(c.rsplit(":", 1)[-1].startswith("@") for c in P.columns)


def _prefixes(P: pl.DataFrame) -> list:
    """"" (the model) and each baseline's b:<name>: among the prediction columns."""
    base = []
    for c in P.columns:
        if c.startswith("b:"):
            p = c[:len(c) - len(c.rsplit(":", 1)[-1])]
            if p not in base:
                base.append(p)
    return ([""] if _cols(P, "") else []) + base


def _reduce(spec, part: pl.DataFrame) -> pl.DataFrame:
    """One stretch of a walk's predictions (with _t, _key and y) as the predictive mean of each arm (`pred`, float32)
    and the per-row statistics of every metric the study scores (`@<metric>.<j>`); rows with no label get NaN
    statistics."""
    ids = part.select("_t", "_key", "y")
    part = part.drop("_t", "_key", "y")
    names, ctx = spec.evaluate["metrics"], {k: v for k, v in score_ctx(spec.evaluate).items() if v is not None}
    lacking = [m for m in names if METRICS.get(m) is None or METRICS[m].rows is None]
    if lacking:
        raise StudyError(f"{', '.join(lacking)} cannot score a distribution over several outputs step by step",
                         "this run's predictions are too wide to keep whole, so each step is reduced to the per-row "
                         "statistics of the study's metrics, and these have none (rows/total)",
                         "score metrics that average over rows and outputs: crps, crps_empirical, mae, rmse, "
                         "pinball, coverage90, width90 (or a study metric with <name>_rows and <name>_total)")
    if own_metrics(spec, names):
        raise StudyError("a study's own metric cannot score a distribution over several outputs step by step",
                         "metrics.py runs only in its own sandbox, and this run's predictions are too wide to keep",
                         "score it with openlab's metrics (crps, crps_empirical, coverage90, width90, mae, rmse)")
    known = ids["y"].is_not_null().to_numpy()
    cols = [ids["_t"], ids["_key"], ids["y"]]
    y = ids["y"].filter(pl.Series(known)).to_numpy() if known.any() else None
    for prefix in _prefixes(part):
        pred = _pred(part, prefix)
        point = np.asarray(_point(pred) if hasattr(pred, "columns") else pred, dtype=np.float32)
        cols.append(pl.Series(f"{prefix}pred", point))
        sub = (pred.filter(pl.Series(known)) if hasattr(pred, "columns") else pred[known]) if y is not None else None
        for m in names:
            R = METRICS[m].rows(y, sub, **ctx) if y is not None else np.zeros((0, 1))
            full = np.full((len(known), R.shape[1]), np.nan)
            full[known] = R
            cols += [pl.Series(f"{prefix}@{m}.{j}", full[:, j]) for j in range(R.shape[1])]
    return pl.DataFrame(cols)


class _Progress:
    """Prints the walk's progress (step k of n, the minutes so far and an estimate of those left) every minute while
    it runs, and puts it on the agent's timeline every five, so a long run is seen to move."""

    def __init__(self, path: Path, timeline=None):
        self.path, self.timeline, self.t0 = path, timeline, time.monotonic()
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _line(self) -> str:
        try:
            p = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return ""
        mins = (time.monotonic() - self.t0) / 60
        left = mins * (p["steps"] - p["step"]) / p["step"] if p["step"] else None
        return (f"walk: step {p['step']} of {p['steps']}, {mins:.0f} min so far"
                + (f", about {left:.0f} min left" if left is not None else ""))

    def _loop(self) -> None:
        k = 0
        while not self.done.wait(60):
            k += 1
            line = self._line()
            if line:
                print(f"openlab: {line}", file=sys.stderr, flush=True)
                if self.timeline is not None and k % 5 == 0:
                    self.timeline.progress(line)

    def stop(self) -> None:
        self.done.set()


class _Reducer:
    """Reduces the walk's step files (`parts/0000001.parquet`, ..., each renamed into place once written) in order
    while the walk runs, deleting each, so a wide distribution never fills the disk."""

    def __init__(self, spec, folder: Path):
        self.spec, self.folder, self.out, self.error, self.n = spec, folder, [], None, 0
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _next(self) -> Path | None:
        p = self.folder / f"{self.n:07d}.parquet"
        return p if p.is_file() else None

    def _loop(self) -> None:
        try:
            while True:
                p = self._next()
                if p is None:
                    if self.done.is_set() and self._next() is None:
                        return
                    time.sleep(0.2)
                    continue
                self.out.append(_reduce(self.spec, pl.read_parquet(p)))
                p.unlink()
                self.n += 1
        except BaseException as e:                  # noqa: BLE001 -- raised in the run's thread by finish()
            self.error = e

    def finish(self):
        """The reduced rows (None when the walk wrote no step file), once every file is read."""
        self.done.set()
        self.thread.join()
        if self.error is not None:
            raise self.error
        return pl.concat(self.out, how="vertical_relaxed") if self.out else None


def _aligned(P: pl.DataFrame, R: pl.DataFrame) -> pl.DataFrame:
    """The reduced rows, checked against the walk's own list of the rows it predicted (its _t and _key)."""
    if R.height != P.height or R["_t"].to_list() != P["_t"].to_list() or R["_key"].to_list() != P["_key"].to_list():
        raise StudyError(f"the walk wrote {R.height} predicted rows and listed {P.height}",
                         "the step files are incomplete or out of order")
    return R


# --- evaluation ---------------------------------------------------------------------------------------------
def _scored(spec, segment: str, P: pl.DataFrame) -> pl.DataFrame:
    """The rows the result scores: a temporal segment's first `split.burn_in` (declared before the seal: the
    model's warm-up) runs and is kept in the predictions, but is not scored."""
    burn = spec.split.get("burn_in")
    if not burn or "_t" not in P.columns:
        return P
    start = splits.segments(spec)[segment][0]
    return P.filter(pl.col("_t") >= start + parse(burn))


def own_metrics(spec, names=None) -> list:
    """The study's own metrics (study.toml [metrics.*]) among `names` (default: what its evaluation scores)."""
    ev = spec.evaluate
    return [m for m in (names or [*ev["metrics"], *ev.get("policy_metrics", [])])
            if METRICS.get(m) is not None and METRICS[m].study]


def sandboxed(rec, spec, call: str, frames: dict, **args):
    """`call(spec, **frames, **args)` (an openlab.study function that scores) in a sandboxed child holding the
    study's own metrics: metrics.py exactly as the spec seals it (its code_sha, from the record's content store),
    never imported in this process. The reply is what `call` returns, through JSON."""
    from .metrics import STUDY_FILE
    shas = {m["code_sha"] for m in (spec.raw.get("metrics") or {}).values()}
    root = rec.dir / "stage"
    root.mkdir(parents=True, exist_ok=True)
    code, stage = (Path(tempfile.mkdtemp(prefix=p, dir=root)) for p in ("metrics-", "score-"))
    try:
        for sha in shas:                                # one file, so one sha
            (code / STUDY_FILE).write_bytes(rec.blob(sha))
        (stage / "frames").mkdir()
        for name, df in frames.items():
            df.write_parquet(stage / "frames" / f"{name}.parquet")
        res = sandbox.run({"entry": f"{__name__}:score_job", "args": {"spec": spec.raw, "seed": 0, "code": str(code),
                                                                      "call": call, "args": args}},
                          study_dir=code, stage_dir=stage, timeout=spec.sandbox.get("timeout"), threads=threads(spec))
        if not res.ok:
            raise StudyError("metrics.py failed while scoring", res.stderr[-3000:],
                             "fix metrics.py (a person amends it once the study has run) and score again")
        return json.loads((stage / "scores.json").read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(code, ignore_errors=True)
        shutil.rmtree(stage, ignore_errors=True)


def score_ctx(ev: dict) -> dict:
    """What a metric reads from the spec's [evaluate] besides the predictions: the declared prediction kind (point
    or prob for an array) and the interval level (coverage, interval_width)."""
    return {"prediction": ev.get("prediction"), **({"level": ev["level"]} if "level" in ev else {})}


def _evaluate(spec, P: pl.DataFrame, seed: int) -> dict:
    """The Result body: each metric with its bootstrap interval for the model and each baseline, the model minus
    each baseline on the primary metric, and the policy metrics when model.py acts. Rows with no label are not
    scored. Temporal intervals resample blocks of `evaluate.block` (7) decisions."""
    ev = spec.evaluate
    S = _known(P, "y")
    y, names = S["y"].to_numpy(), ev["metrics"]
    ctx = score_ctx(ev)
    temporal = "_t" in S.columns and ev.get("interval") != "bootstrap"
    boot = {"seed": seed, "alpha": ev.get("alpha", 0.05),
            "blocks": _blocks(S["_t"], ev.get("block", 7)) if temporal else None}
    model, base = _pred(S, ""), {b: _pred(S, f"b:{b}:") for b in ev.get("baselines", [])}
    groups = {}
    if ev.get("by") == "key":                        # each entity's own scores (a station, a dataset, a class)
        for (g,), G in S.group_by("_key", maintain_order=True):
            gy, gb = G["y"].to_numpy(), {**boot, "blocks": _blocks(G["_t"], ev.get("block", 7)) if temporal else None}
            groups[str(g)] = {"metrics": {m: _score(m, gy, _pred(G, ""), gb, **ctx) for m in names},
                              "baselines": {b: {m: _score(m, gy, _pred(G, f"b:{b}:"), gb, **ctx) for m in names}
                                            for b in ev.get("baselines", [])}, "n": G.height}
    return {"metrics": {m: _score(m, y, model, boot, **ctx) for m in names},
            **({"groups": groups} if groups else {}),
            "baselines": {b: {m: _score(m, y, p, boot, **ctx) for m in names} for b, p in base.items()},
            "paired": {b: _paired(names[0], y, model, p, boot, **ctx) for b, p in base.items()},
            "policy": _policy(spec, P, {**boot, "blocks": None}, temporal) if "act" in P.columns else {},
            "n": S.height}


def _policy(spec, P: pl.DataFrame, boot: dict, blocked: bool) -> dict:
    """act()'s values, one per decision, scored by evaluate.policy_metrics for the model and each policy baseline."""
    arms = ["act", *(f"act:{p}" for p in spec.evaluate.get("policy_baselines", []))]
    D = P.unique("_t", keep="first", maintain_order=True).sort("_t")
    acted = D.height
    D = D.filter(pl.all_horizontal(pl.col(a).is_finite() for a in arms))   # every arm on the same decisions
    ppy = _per_year(spec, D["_t"])
    if blocked:
        boot = {**boot, "blocks": _blocks(D["_t"], spec.evaluate.get("block", 7))}
    names = spec.evaluate.get("policy_metrics", ["mean"])

    def scores(col: str) -> dict:
        return {m: _score(m, None, D[col].to_numpy(), boot, periods_per_year=ppy) for m in names}

    def mean(col: str) -> dict:                     # an extra value of act(), over the decisions where it is finite
        S = D.filter(pl.col(col).is_finite())
        if S.height == 0:
            return {"value": None, "reason": f"act() gave no finite {col[len('act.'):]}"}
        return _score("mean", None, S[col].to_numpy(),
                      {**boot, "blocks": _blocks(S["_t"], spec.evaluate.get("block", 7))} if blocked else boot)
    paired = {p: _paired_series(names[0], D["act"].to_numpy(), D[f"act:{p}"].to_numpy(), boot, ppy)
              for p in spec.evaluate.get("policy_baselines", [])}
    extra = {c[len("act."):]: mean(c) for c in D.columns if c.startswith("act.")}
    return {"decisions": D.height, "skipped": acted - D.height, "periods_per_year": ppy, "metrics": scores("act"),
            "paired": paired, **({"series": extra} if extra else {}),
            "baselines": {p: scores(f"act:{p}") for p in spec.evaluate.get("policy_baselines", [])}}


def _paired_series(metric: str, a, b, boot: dict, ppy) -> dict:
    """The model's policy minus a policy baseline's on the first policy metric, on the same resamples."""
    try:
        return paired(metric, None, a, b, seed=boot["seed"], alpha=boot["alpha"], blocks=boot.get("blocks"),
                      periods_per_year=ppy)
    except StudyError as e:
        return {"diff": None, "reason": e.error}


def _per_year(spec, t: pl.Series) -> float | None:
    """Decisions per year: from the cadence of split.decisions, else from the span of the decision times."""
    how = spec.split.get("decisions")
    if how in splits.CADENCES:
        d = parse(splits.CADENCES[how])
        return 12 / d.months if d.months else timedelta(days=365.25) / d.delta
    days = (t.max() - t.min()).total_seconds() / 86400 if t.len() > 1 else 0
    return (t.len() - 1) * 365.25 / days if days else None


def _score(metric: str, y, pred, boot: dict, **ctx) -> dict:
    """{value, lo, hi}, or {value: None, reason} when the metric cannot score this prediction."""
    try:
        v = score(metric, y, pred, **ctx)
        lo, hi = interval(metric, y, pred, **boot, **ctx)
    except (StudyError, ValueError) as e:
        return {"value": None, "reason": str(e)}
    if not math.isfinite(v):
        return {"value": None, "reason": f"{metric} is undefined on these rows"}
    nom = nominal(metric, ctx.get("level"))           # a target-valued metric: the value it is best at
    return {"value": v, "lo": _num(lo), "hi": _num(hi), **({"nominal": nom} if nom is not None else {})}


def _paired(metric: str, y, a, b, boot: dict, **ctx) -> dict:
    """The model minus the baseline on `metric`, bootstrapped on the same draws; `better` names the winner."""
    try:
        d = paired(metric, y, a, b, **boot, **ctx)
    except (StudyError, ValueError) as e:
        return {"metric": metric, "diff": None, "reason": str(e)}
    out = {"metric": metric, **{k: _num(d[k]) for k in ("diff", "lo", "hi", "p")},
           "better": {"a": "model", "b": "baseline"}.get(d["better"]),
           **({"nominal": d["nominal"]} if "nominal" in d else {}),
           **({"dropped": d["dropped"]} if d.get("dropped") else {})}       # rows one of the two did not score
    return out if out["diff"] is not None else {**out, "reason": f"{metric} is undefined on these rows"}


def _num(x) -> float | None:
    return float(x) if x is not None and math.isfinite(x) else None


def _known(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """The rows whose `col` is known: not null, and not NaN."""
    df = df.with_columns(pl.col(col).fill_nan(None)) if df.schema[col].is_float() else df
    return df.drop_nulls(col)


def _blocks(t: pl.Series, size: int) -> np.ndarray:
    """A block id per row: consecutive runs of `size` decision times."""
    return ((t.rank("dense").cast(pl.Int64) - 1) // size).to_numpy()


def _cols(P: pl.DataFrame, prefix: str) -> list:
    """The prediction columns under `prefix`: b:<baseline>: for a baseline, "" for the model's own."""
    if prefix:
        return [c for c in P.columns if c.startswith(prefix)]
    return [c for c in P.columns if c not in ID and not c.startswith(("b:", "act:", "act.")) and c != "act"]


def _pred(P: pl.DataFrame, prefix: str):
    """The prediction under `prefix` as the metrics read it: an array for `pred`, else a frame."""
    cols = {c: c[len(prefix):] for c in _cols(P, prefix)}
    return P[next(iter(cols))].to_numpy() if list(cols.values()) == ["pred"] else P.select(list(cols)).rename(cols)


def threads(spec) -> int:
    """The thread pools a run's libraries get ([sandbox] threads, default 1), recorded in the run's env."""
    return int(spec.sandbox.get("threads", 1))


# --- the record ---------------------------------------------------------------------------------------------
def _record(r: _Run, out: _Out, ev: dict, fp: dict, key: str, *, note: str, actor: str, look: str | None,
            seconds: float, branch: str = "", seed: int | None = None, parent: str = "", op: str = "",
            refs: tuple = (), cites: tuple = ()) -> Node:
    """The run, its result, and the final model (registered in the study's MLflow) as a model node. The run branches
    from `parent` when given (none for a draft), else from the branch's last run (else the study's last)."""
    rec, spec = r.study.record, r.spec
    buf = io.BytesIO()
    out.P.write_parquet(buf)
    if parent or op:
        prev = _run_node(rec, parent) if parent else None
    else:
        prev = (rec.last("run", segment="research", branch=branch) if branch else None) or \
            rec.last("run", segment="research")
    links = [("under", (rec.last("amendment") or rec.last("plan")).id),
             *[("uses", rec.last("dataset", name=n).id) for n in fp["data"]],
             *([("branches_from", prev.id)] if prev else []), *[("refs", _run_node(rec, x).id) for x in refs],
             *([("after", look)] if look else [])]
    used = [*spec.evaluate["metrics"], *(spec.evaluate.get("policy_metrics", ["mean"]) if ev and ev["policy"] else [])]
    choices = {"spec": {s: hashlib.sha256(canonical(spec.raw.get(s, {})).encode()).hexdigest() for s in SPEC_CHOICES},
               "code": fp["code"], "params": out.params}
    run = rec.add(schema.Run(
        segment=r.segment, key=key, cached=False, spec_sha=fp["spec_sha"], code=fp["code"], data=fp["data"],
        env={**fp["env"], "threads": threads(spec)}, sandbox="unsandboxed" if "unsandboxed" in r.sandboxes else "bwrap",
        audit=out.audit,
        seconds=seconds, choices=choices, stdout=rec.put_blob(out.stdout.encode()),
        predictions=rec.put_blob(buf.getvalue()), metric_versions={m: METRICS[m].version for m in used}, note=note,
        skipped=out.skipped, branch=branch, seed=seed, op=op, cites=list(cites),
        box=_box(getattr(r, "gpus", None), getattr(r, "waited", 0.0))), actor=actor, links=links)
    result = rec.add(schema.Result(**ev), actor=actor, links=[("of", run.id)]) if ev is not None else run
    name = _model_name(spec)
    warm = getattr(r, "registry", None) or _Registry(rec.dir, name)
    with warm.lock():
        entry = warm.get().log_packed(name, *out.model,
                                      note=f"{run.id} ({r.segment}, key {key[:12]}, spec {fp['spec_sha'][:12]})")
    rec.add(schema.Model(name=name, version=entry.version, uri=f"models:/study-{name}/{entry.version}",
                         keys=out.model[1].get("keys", [])), actor=actor, links=[("from", run.id)])
    from . import mirror
    mirror.after_run(r.study, run)                  # the W&B view, detached: never waits, never fails a run
    return result


def _model_name(spec) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", spec.name).strip("_-")[:100] or "model"


def quiet_mlflow() -> None:
    """MLflow prints an agent hint on import and its schema migrations on a new database: none of it belongs in a
    run's output. Set before mlflow is imported, and its loggers held at WARNING after."""
    import logging
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    os.environ.setdefault("MLFLOW_LOGGING_LEVEL", "WARNING")
    import mlflow  # noqa: F401
    for name in ("mlflow", "alembic"):
        logging.getLogger(name).setLevel(logging.WARNING)


class _Registry(threading.Thread):
    """The study's model registry (MLflow under .openlab/mlflow), opened in the background: importing mlflow and
    creating its database take seconds, which overlap the run's children. Opening and writing hold
    .openlab/mlflow.lock, so two runs finishing at once do not race on MLflow's database."""

    def __init__(self, dir: Path, name: str):
        super().__init__(daemon=True)
        self.dir, self.name, self.reg, self.err = Path(dir), name, None, None
        self.start()

    @contextmanager
    def lock(self):
        with open(self.dir / "mlflow.lock", "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def run(self) -> None:
        try:
            quiet_mlflow()
            with self.lock():
                self.reg = registry.Registry(self.name, arm="study", root=str(self.dir / "mlflow"))
        except BaseException as e:              # noqa: BLE001 -- raised where the registry is used
            self.err = e

    def get(self):
        self.join()
        if self.err is not None:
            raise self.err
        return self.reg


# --- shared by parent and children --------------------------------------------------------------------------
def _study_baselines(spec) -> list:
    """The baselines the study's own code defines (model:<Class>): replayed by the truncation audit like the model,
    since a fidelity rests on them. openlab's built-in baselines are its own code."""
    return [b for b in spec.evaluate.get("baselines", []) if b.startswith("model:")]


def _train_times(times: list, step: Step, h) -> list:
    """The decision times (sorted) that `step`'s fit trains on."""
    return [t for t in times[:bisect.bisect_right(times, step.train_until)] if step.trains_on(t, h)]


def _iso(times: list) -> list:
    return [t.isoformat() for t in times]


def _times(xs: list) -> list:
    return [datetime.fromisoformat(x) for x in xs]


def _step_out(s: Step) -> list:
    return [s.t.isoformat(), s.refit, s.train_until.isoformat(), s.train_from and s.train_from.isoformat()]


def _step_in(v: list) -> Step:
    return Step(datetime.fromisoformat(v[0]), v[1], datetime.fromisoformat(v[2]), v[3] and datetime.fromisoformat(v[3]))


# --- the children ------------------------------------------------------------------------------------------
def _frame(out, what: str) -> pl.DataFrame:
    """features()/label() output as a frame: a frame as is, a dict as one row, a scalar label as y."""
    if isinstance(out, pl.DataFrame):
        return out
    if isinstance(out, dict):
        return pl.DataFrame([out])
    if what == "label":
        return pl.DataFrame({"y": [out]})
    raise StudyError(f"{what}() returned a {type(out).__name__}", "a dict for one entity, or a polars DataFrame")


def outputs(df: pl.DataFrame, col: str, what: str) -> pl.DataFrame:
    """A label with several outputs per row (a list column) as a fixed-width array column; one width for all."""
    if not isinstance(df.schema[col], pl.List):
        return df
    widths = sorted(df[col].list.len().drop_nulls().unique().to_list())
    if len(widths) > 1:
        raise StudyError(f"{what} returned outputs of {', '.join(map(str, widths))} values",
                         "a label's outputs have one width (pad a shorter one with NaN)")
    return df.with_columns(pl.col(col).cast(pl.Array(pl.Float64, widths[0] if widths else 0)))


def _key(keys: list) -> pl.Expr:
    """The entity key as one string ("" when there is none)."""
    return pl.concat_str([pl.col(k).cast(pl.String) for k in keys], separator="|") if keys else pl.lit("")


def _fitted(model, spec, X: pl.DataFrame, y, seed: int, with_model: bool = True, with_baselines: bool = False) -> dict:
    """{"": Model} and/or {"b:<name>:": each baseline}, each made and fitted (when it has fit) after seeding random
    and numpy."""
    makers = {"": model.Model} if with_model else {}
    if with_baselines or not with_model:
        makers.update({f"b:{b}:": partial(baselines.make, b, spec, model) for b in spec.evaluate.get("baselines", [])})
    out = {}
    for prefix, make in makers.items():
        random.seed(seed)
        np.random.seed(seed)
        out[prefix] = make()
        if hasattr(out[prefix], "fit"):
            out[prefix].fit(X, y)
    return out


def _predict(fitted: dict, X: pl.DataFrame) -> pl.DataFrame:
    """Each fitted model's predictions for X as columns: the model's as returned (an array is `pred`), each
    baseline's under b:<name>:."""
    cols = []
    for prefix, m in fitted.items():
        p = m.predict(X)
        if not isinstance(p, pl.DataFrame):
            a = np.asarray(p)
            if a.ndim not in (1, 2):
                raise StudyError(f"predict returned an array of shape {a.shape}",
                                 "one value per row, one row of outputs per row (a label with several outputs), or "
                                 "a frame (mu, sigma / q0.1, ... / m0, ... / lower, upper)")
            p = pl.DataFrame({"pred": a})
        if p.height != X.height:
            raise StudyError(f"predict returned {p.height} rows for {X.height}")
        cols += [p[c].alias(prefix + c) for c in p.columns]
    return pl.DataFrame(cols)


def _pack(stage: Path, model, fitted, keys) -> None:
    """The final model as registry.pack makes it (with the key columns its rows were keyed on), and model.PARAMS,
    for the parent to register."""
    blob, info = registry.pack(fitted)
    (stage / "model.pkl").write_bytes(blob)
    (stage / "model.json").write_text(json.dumps({"pack": {**info, "keys": list(keys or [])},
                                                  "params": getattr(model, "PARAMS", {})}, default=str),
                                      encoding="utf-8")


class _Walk:
    """A temporal child's view of its stage: one Timeline, model.py, and features, labels and training rows per
    decision time, each computed once."""

    def __init__(self, stage: Path, spec: dict, seed: int):
        self.spec = _spec.build(spec, stage)
        self.model = importlib.import_module("model")
        frames = {n: pl.read_parquet(stage / "data" / f"{n}.parquet") for n in self.spec.data}
        self.tl, self.h, self.seed = Timeline(frames, self.spec.data), splits.horizon(self.spec), seed
        self.keys: list | None = None              # the entity key: the columns label() returns besides y
        self._x, self._rows = {}, {}
        self._last: dict = {}                      # "model" / "base" -> (fitted, the times it trained on)

    def x(self, t: datetime) -> pl.DataFrame:
        if t not in self._x:
            out = _frame(self.model.features(self.tl.at(t), t), "features")
            self._x[t] = out.with_columns(_t=pl.lit(t, pl.Datetime("us")))
        return self._x[t]

    def keyed(self, t: datetime) -> pl.DataFrame:
        """X at t: _t, _key, then the columns features() returned."""
        return self.x(t).with_columns(_key=_key(self.keys)).select("_t", "_key", pl.exclude("_t", "_key"))

    def labelled(self, t: datetime) -> pl.DataFrame:
        """X at t with the label at t + horizon as __y, null where label() has no row for the key."""
        if t not in self._rows:
            lab = _frame(self.model.label(self.tl.at(t + self.h), t), "label")
            if "y" not in lab.columns:
                raise StudyError("label() returned no y", "a value, a dict with y, or a frame of the key columns and y")
            keys = [c for c in lab.columns if c != "y"]
            self.keys = keys if self.keys is None else self.keys
            if keys != self.keys:
                raise StudyError(f"label() returned the key columns {keys} at {t}, {self.keys} before")
            X, lab = self.keyed(t), outputs(lab.rename({"y": "__y"}), "__y", "label()")
            gone = [k for k in keys if k not in X.columns]
            if gone:
                raise StudyError(f"features() has no key column {', '.join(gone)}",
                                 "labels join to features on the columns label() returns besides y")
            if not keys and (X.height > 1 or lab.height != 1):
                raise StudyError(f"label() returned {lab.height} values for {X.height} rows of features at {t}",
                                 "with no key columns there is one entity", "return a frame of the key columns plus y")
            self._rows[t] = (X.join(lab, on=keys, how="left", validate="m:1", maintain_order="left") if keys
                             else X.with_columns(__y=pl.lit(lab["__y"][0])))
        return self._rows[t]

    def fit(self, times: list, step: Step, with_baselines: bool) -> dict | None:
        """The model fitted on this step's training rows, and the baselines on every row known by then (the window
        is the model's choice, not theirs); None while the model has no rows. A Model with no fit needs none."""
        if hasattr(self.model.Model, "fit"):
            fitted = self._refit("model", _train_times(times, step, self.h), with_model=True)
        else:
            fitted = _fitted(self.model, self.spec, None, None, self.seed)
        if fitted is not None and with_baselines and self.spec.evaluate.get("baselines"):
            base = self._refit("base", _train_times(times, replace(step, train_from=None), self.h), with_model=False)
            fitted = None if base is None else {**fitted, **base}
        return fitted

    def _refit(self, group: str, ts: list, with_model: bool) -> dict | None:
        """A fit on the rows of `ts`; or, when every model of the last fit has `update(X, y)` and `ts` only adds
        decisions to the ones it trained on (an expanding window), the last fit updated with the new rows alone.
        The truncation audit fits from scratch at the steps it replays, so an update that a fresh fit would not
        reproduce shows as a look-ahead mismatch."""
        last, before = self._last.get(group, (None, None))
        if last is not None and len(before) <= len(ts) and ts[:len(before)] == before and \
                all(callable(getattr(m, "update", None)) for m in last.values()):
            new = ts[len(before):]
            T = _known(pl.concat([self.labelled(t) for t in new], how="vertical_relaxed"), "__y") if new else None
            if T is not None and T.height:
                X, y = T.drop("__y"), T["__y"].to_numpy()
                for m in last.values():
                    m.update(X, y)
            self._last[group] = (last, ts)
            return last
        fitted = self._fit(ts, with_model)
        self._last[group] = (fitted, ts) if fitted is not None else (None, None)
        return fitted

    def _fit(self, ts: list, with_model: bool) -> dict | None:
        T = _known(pl.concat([self.labelled(t) for t in ts], how="vertical_relaxed"), "__y") if ts else None
        if T is None or T.height == 0:
            return None
        return _fitted(self.model, self.spec, T.drop("__y"), T["__y"].to_numpy(), self.seed, with_model)

    def predict(self, fitted: dict, t: datetime) -> pl.DataFrame:
        X = self.keyed(t)
        return pl.concat([X.select("_t", "_key"), _predict(fitted, X)], how="horizontal")

    def acts(self, P: pl.DataFrame, act) -> dict:
        """act() per decision with the label view, for the model and each policy baseline, repeated on the
        decision's rows."""
        fns = {"act": act}
        for p in self.spec.evaluate.get("policy_baselines", []):
            fn = getattr(baselines.make(p, self.spec, self.model), "act", None)
            if fn is None:
                raise StudyError(f"policy baseline {p} has no act", "a policy baseline is scored by its own act(pred_row, "
                                 "view, t), on its own forecast when it is also in evaluate.baselines",
                                 f"give {p.split(':')[-1]} an act method")
            fns[f"act:{p}"] = fn
        # each policy acts on its own forecast: a policy baseline that is also a forecast baseline (evaluate.baselines)
        # gets its b:<name>: predictions under the model's column names; any other gets the model's
        own = {"act": {c: c for c in _cols(P, "")}}
        for p in self.spec.evaluate.get("policy_baselines", []):
            pre = f"b:{p}:"
            own[f"act:{p}"] = {c: c[len(pre):] for c in _cols(P, pre)} or own["act"]
        values: dict = {}
        for (t,), rows in P.group_by("_t", maintain_order=True):
            view = self.tl.at(t + self.h)
            for c, fn in fns.items():
                R = rows.select("_t", "_key", *[pl.col(a).alias(b) for a, b in own[c].items()])
                row = R.row(0, named=True) if R.height == 1 else R.to_dict(as_series=False)
                for k, v in _named(c, fn(row, view, t)).items():
                    values.setdefault(k, []).extend([float(v)] * rows.height)
        uneven = [k for k, v in values.items() if len(v) != P.height]
        if uneven:
            raise StudyError(f"act() returned {', '.join(uneven)} at some decisions only",
                             fix="return the same names at every decision")
        return {c: pl.Series(v, dtype=pl.Float64) for c, v in values.items()}


def _named(col: str, value) -> dict:
    """act()'s return under `col`: a number, or a dict whose first value is the policy's and the rest col.<name>."""
    if not isinstance(value, dict):
        return {col: value}
    (_, first), *rest = value.items()
    return {col: first, **{f"{col}.{k}": v for k, v in rest}}


def temporal_job(stage: Path, *, spec: dict, seed: int, times: list, steps: list) -> Path:
    """Walk the steps: refit where a step refits, predict each step's rows once a fit exists; then the labels of the
    scored rows and act() per decision (so no label view opens while features are still being computed)."""
    from .registry import check_deterministic
    w, times = _Walk(stage, spec, seed), _times(times)
    fitted, final, out, wide = None, None, [], None
    shown = time.monotonic()
    for i, step in enumerate(map(_step_in, steps)):
        if time.monotonic() - shown > 10:           # how far the walk is, for the parent to show
            (stage / "progress.json").write_text(json.dumps({"step": i, "steps": len(steps)}))
            shown = time.monotonic()
        if step.refit:
            fitted = w.fit(times, step, with_baselines=True)
        if fitted is not None:
            p = w.predict(fitted, step.t)
            if wide is None and (wide := _wide(p)):
                (stage / "parts").mkdir(exist_ok=True)
            if wide:                                # each step with its labels to disk; the parent reduces it
                ids = w.labelled(step.t).select("_t", "_key", pl.col("__y").alias("y"))
                tmp = stage / "parts" / f"{len(out):07d}.tmp"
                ids.hstack(p.drop("_t", "_key")).write_parquet(tmp)
                os.replace(tmp, tmp.with_suffix(".parquet"))
                p = ids.select("_t", "_key")
            out.append(p)
            final = (fitted[""], step.t)
    if not out:
        raise StudyError("no step had training rows", "a fit needs decisions whose labels are known by the refit",
                         "check that label() returns the key columns features() returns, plus y")
    P = pl.concat(out, how="vertical_relaxed")
    ts = P["_t"].unique(maintain_order=True)
    y = pl.concat([w.labelled(t).select("__y") for t in ts], how="vertical_relaxed")
    # a Model with no fit predicted before label() named the key columns: key its rows now
    P = P.with_columns(y=y["__y"], _key=pl.concat([w.keyed(t).select("_key") for t in ts])["_key"])
    act = getattr(w.model, "act", None)
    ev = w.spec.evaluate
    if act is None and (ev.get("policy_metrics") or ev.get("policy_baselines")):
        hint = " (Model has an act method: openlab calls a module-level act)" if hasattr(getattr(w.model, "Model", None),
                                                                                         "act") else ""
        raise StudyError(f"evaluate declares a policy, and model.py defines no act(){hint}",
                         "policy_metrics and policy_baselines score act()'s value per decision; with no act the "
                         "policy would silently go unscored",
                         "define def act(pred_row, view, t) in model.py, or remove the policy fields by amendment")
    if act is not None and wide:
        raise StudyError("act() reads each decision's prediction, and this one is a distribution over several outputs "
                         "too wide to keep", "the walk scores it step by step", "act on a narrower prediction")
    if act is not None:
        P = P.with_columns(**w.acts(P, act))
    check_deterministic(final[0], w.keyed(final[1]))
    P.write_parquet(stage / "predictions.parquet")
    _pack(stage, w.model, final[0], w.keys)
    return stage / "predictions.parquet"


def replay_job(stage: Path, *, spec: dict, seed: int, times: list, step: list) -> Path:
    """One step again on data cut at its time: the training rows its fit used, the fit, the step's predictions."""
    w, step = _Walk(stage, spec, seed), _step_in(step)
    fitted = w.fit(_times(times), step, with_baselines=bool(_study_baselines(w.spec)))
    if fitted is not None and w.keys is None:       # a Model with no fit: label() names the keys (no later data here)
        w.labelled(step.t)
    out = stage / "predictions.parquet"
    (pl.DataFrame() if fitted is None else w.predict(fitted, step.t)).write_parquet(out)
    return out


def _static_x(model, frame: pl.DataFrame, keys: tuple) -> pl.DataFrame:
    """Static X: features(frame) when model.py has it (row-wise), else the frame; with _key first."""
    f = getattr(model, "features", None)
    X = _frame(f(frame), "features") if f else frame
    if X.height != frame.height:
        raise StudyError(f"features(frame) returned {X.height} rows for {frame.height}", "static features are row-wise")
    return X.with_columns(_key=frame.with_columns(_key=_key(list(keys)))["_key"]).select("_key", pl.exclude("_key"))


def score_job(stage: Path, *, spec: dict, seed: int, code: str, call: str, args: dict) -> Path:
    """`sandboxed`'s child: the study's metrics.py adopted from `code`, then call(spec, **frames, **args), its reply
    written as scores.json."""
    from .metrics import adopt
    module, _, fn = call.partition(":")
    if not module.startswith("openlab.study.") or not fn:
        raise ValueError(f"{call!r} is not an openlab.study function")
    s = _spec.build(spec, stage)
    adopt(code, spec.get("metrics", {}))
    frames = {p.stem: pl.read_parquet(p) for p in sorted((stage / "frames").glob("*.parquet"))}
    out = stage / "scores.json"
    out.write_text(json.dumps(getattr(importlib.import_module(module), fn)(s, **frames, **args)), encoding="utf-8")
    return out


def static_job(stage: Path, *, spec: dict, seed: int, target: str, pack: bool) -> Path:
    """Fit Model and the baselines on data/train.parquet and predict data/valid.parquet (staged without the target)
    when there is one; with `pack`, keep the fitted model."""
    from .registry import check_deterministic
    s, model = _spec.build(spec, stage), importlib.import_module("model")
    keys = next(iter(s.data.values())).key
    train, valid, out = stage / "data" / "train.parquet", stage / "data" / "valid.parquet", stage / "predictions.parquet"
    T = outputs(pl.read_parquet(train), target, f"data column {target}")
    fitted = _fitted(model, s, _static_x(model, T.drop(target), keys), T[target].to_numpy(), seed,
                     with_baselines=valid.is_file())
    if valid.is_file():
        V = _static_x(model, pl.read_parquet(valid), keys)
        pl.concat([V.select("_key"), _predict(fitted, V)], how="horizontal").write_parquet(out)
        check_deterministic(fitted[""], V)
    if pack:
        _pack(stage, model, fitted[""], keys)
    return out if valid.is_file() else stage / "model.json"
