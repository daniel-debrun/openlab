"""Independent reproduction (trust level "reproduced"): a recorded run executed again from what the record holds only,
and a bundle a third party reproduces from.

`reproduce` takes a run's code from the content store (`Run.code`), the materialised datasets whose sha256 the run
recorded (refused when a file no longer matches), the sealed spec at the run's `under` node (the plan and the
amendments up to it) and the seed that spec fixes (the run's key must follow from all of them), and runs the run's
children again through the runner, in fresh sandboxes, without the truncation audit (the run has its verdict). The
predictions go to the content store and a `reproduction` node, whose schema compares them with the run's.

`bundle` writes a tar of the record (a consistent copy), its blobs, the materialised data and the study's files
(spec, code, recipes, paper sources and builds, signed files with their signatures), with MANIFEST.json (each file's
sha256 and the record head) signed beside it. `from_bundle` checks the signature, every file and the hash chain in a
temporary directory, and reproduces every run there.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import shutil
import sqlite3
import tarfile
import tempfile
import time
from contextlib import closing
from pathlib import Path, PurePosixPath

import numpy as np
import polars as pl

from . import data, fingerprint, runner, schema
from . import spec as _spec
from .errors import StudyError
from .metrics import score
from .record import Node, Record

MANIFEST = "MANIFEST.json"
SIG = MANIFEST + ".sig.json"
SKIP = {".openlab", ".git", "data", "__pycache__", "release", ".release.tmp"}    # study files not bundled: the record goes in below; raw inputs
RECORD = ("blobs", "data")                           # (the materialised data goes in), history and caches


class _Replay(runner._Run):
    """A recorded run's parent side again: its code from the content store, its data as it recorded it, a static
    holdout's labels from that data (not the vault), and no truncation audit."""

    def __init__(self, study, spec, segment: str, code_dir: Path, frames: dict, seed: int | None = None):
        super().__init__(study, spec, segment, spec.split.get("seed", 0) if seed is None else seed)
        self.code_dir, self._frames = code_dir, frames

    def frames(self) -> dict:
        return self._frames

    def labels(self, name, target, df, hold) -> list:
        return df[target].gather(hold).to_list()

    def _audit_plan(self, steps: list) -> dict:
        return {"n": 0, "seed": 0, "order": []}

    def audit(self, *args, **kw) -> dict:
        return {}


def _spec_of(rec: Record, run: Node):
    return _spec.sealed(rec, at=rec.links(run.id, rel="under")[0][2])


def _code(rec: Record, code: dict, into: Path) -> None:
    """Write the run's code files from the content store (a path leaving `into` is refused)."""
    for rel, sha in code.items():
        parts = PurePosixPath(rel).parts
        if PurePosixPath(rel).is_absolute() or ".." in parts:
            raise StudyError(f"the run records the code file {rel!r}", "a code path stays inside the study")
        path = into.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(rec.blob(sha))


def _one(study, run: Node, actor: str) -> Node:
    t0, rec, p = time.monotonic(), study.record, run.payload
    if run.kind != "run":
        raise StudyError(f"{run.id} is a {run.kind}, not a run", fix="study_runs lists the runs")
    if not p["predictions"]:
        raise StudyError(f"{run.id} records no predictions", "there is nothing to compare a reproduction with")
    spec = _spec_of(rec, run)
    fp = {k: p[k] for k in ("spec_sha", "code", "data", "env")}
    seed = p.get("seed")
    if fingerprint.key(fp, p["segment"], spec.split.get("seed", 0) if seed is None else seed) != p["key"]:
        raise StudyError(f"{run.id}'s key does not follow from its spec, code, data, environment and seed",
                         "the record does not say how the run was made", "run `openlab study verify`")
    frames = {name: data.load(study, name, sha=sha) for name, sha in p["data"].items()}
    root = rec.dir / "stage"
    root.mkdir(parents=True, exist_ok=True)
    code = Path(tempfile.mkdtemp(prefix="code-", dir=root))
    try:
        _code(rec, p["code"], code)
        r = _Replay(study, spec, p["segment"], code, frames, seed)
        out = r.temporal() if spec.mode == "temporal" else r.static()
        env = {**fingerprint.env(code), "threads": int(spec.sandbox.get("threads", 1))}   # as the run records it
    finally:
        shutil.rmtree(code, ignore_errors=True)
    buf = io.BytesIO()
    out.P.write_parquet(buf)
    sha = rec.put_blob(buf.getvalue())
    return rec.add(schema.Reproduction(
        run=run.id, predictions=sha, **compare(rec, run, sha), env=env, env_diff=env_diff(p["env"], env),
        sandbox="unsandboxed" if "unsandboxed" in r.sandboxes else "bwrap", seconds=time.monotonic() - t0),
        actor=actor, links=[("of", run.id)])


def reproduce(study, run: str = "", every: bool = False, actor: str = "agent") -> list:
    """Reproduce one run (default: the latest) or `every` run of the study; returns the reproduction nodes."""
    rec = study.record
    runs = rec.find("run") if every else [rec.get(run)] if run else rec.find("run")[-1:]
    if not runs:
        raise StudyError("the study has no run to reproduce", fix="run it first (openlab study run)")
    return [_one(study, r, actor) for r in runs]


# --- the comparison (the Reproduction schema recomputes it) ---------------------------------------------------
def _num(x) -> float | None:
    return float(x) if x is not None and math.isfinite(x) else None


def _differ(a: pl.DataFrame, b: pl.DataFrame) -> tuple:
    """(what differs, "" when nothing; the largest absolute difference between numbers both have)."""
    if "_key" in b.columns and "_key" not in a.columns:     # a static run from before openlab kept its rows' keys
        b = b.drop("_key")
    if a.columns != b.columns or a.height != b.height:
        return f"the run predicted {a.height} rows of {a.columns}, the reproduction {b.height} of {b.columns}", None
    worst, bad = 0.0, []
    for c in a.columns:
        x, z = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype.kind in "fiub" and z.dtype.kind in "fiub":
            x, z = x.astype(float), z.astype(float)
            same = np.isclose(x, z, rtol=runner.TOL, atol=0, equal_nan=True)
            both = np.isfinite(x) & np.isfinite(z)
            worst = max(worst, float(np.abs(x[both] - z[both]).max()) if both.any() else 0.0)
        else:
            same = np.asarray(x == z, dtype=bool)
        if not same.all():
            bad.append(f"{c} differs on {int((~same).sum())} of {len(same)} rows")
    return "; ".join(bad), worst


def _scored(spec, segment: str, P: pl.DataFrame) -> dict:
    """{metric: value} of the spec's metrics on P, as the runner scores a run (None where it cannot)."""
    try:
        S = runner._known(runner._scored(spec, segment, P), "y")
        y, pred = S["y"].to_numpy(), runner._pred(S, "")
    except Exception:                                   # noqa: BLE001 -- a frame the metrics cannot read scores None
        return {}
    out = {}
    for m in spec.evaluate["metrics"]:
        try:
            out[m] = _num(score(m, y, pred, prediction=spec.evaluate.get("prediction")))
        except (StudyError, ValueError):
            out[m] = None
    return out


def compare(rec: Record, run: Node, predictions: str) -> dict:
    """The reproduction's predictions (blob `predictions`) against the run's: `differs` (every row and column must
    match: numbers within the truncation audit's relative tolerance, NaN equal to NaN), `max_abs_diff`, and each of
    the run's metrics scored again on the reproduction ({run, reproduced, diff}); `matched` when nothing differs."""
    p = run.payload
    a, b = (pl.read_parquet(io.BytesIO(rec.blob(s))) for s in (p["predictions"], predictions))
    differs, worst = _differ(a, b)
    result = next((rec.get(s) for s, _, _ in rec.links(run.id, rel="of", incoming=True) if s.startswith("result:")),
                  None)
    spec = _spec_of(rec, run)
    again = (runner.sandboxed(rec, spec, f"{__name__}:_scored", {"P": b}, segment=p["segment"])
             if runner.own_metrics(spec) else _scored(spec, p["segment"], b))
    metrics = {}
    for m, v in (result.payload["metrics"] if result else {}).items():
        was, got = v.get("value"), again.get(m)
        metrics[m] = {"run": was, "reproduced": got, "diff": _num(got - was) if None not in (was, got) else None}
    off = [m for m, v in metrics.items() if (v["run"] is None) != (v["reproduced"] is None) or
           (v["diff"] is not None and not math.isclose(v["run"], v["reproduced"], rel_tol=runner.TOL))]
    differs = differs or "; ".join(f"{m}: run {metrics[m]['run']}, reproduced {metrics[m]['reproduced']}" for m in off)
    return {"matched": not differs, "max_abs_diff": worst, "differs": differs, "metrics": metrics}


def env_diff(was: dict, now: dict) -> dict:
    """{key: {run, now}} for each environment field that differs; `packages` as one line naming the distributions
    that differ."""
    out = {k: {"run": was.get(k), "now": now.get(k)} for k in sorted({*was, *now})
           if was.get(k) != now.get(k) and k != "packages"}
    if "packages" in was and "packages" in now and was["packages"] != now["packages"]:
        out["packages"] = {"run": "", "now": fingerprint.package_changes(was["packages"], now["packages"])}
    return out


def summary(n: Node) -> dict:
    p = n.payload
    return {"id": n.id, "run": p["run"], "matched": p["matched"], "max_abs_diff": p["max_abs_diff"],
            "differs": p["differs"], "metrics": p["metrics"], "env_diff": p["env_diff"], "sandbox": p["sandbox"],
            "seconds": p["seconds"], "at": n.at}


def reproductions(study) -> dict:
    """Every reproduction on record, and how many of the runs have a matched one."""
    rec = study.record
    runs = rec.find("run")
    return {"reproductions": [summary(n) for n in rec.find("reproduction")],
            "runs": len(runs), "reproduced": sum(schema.reproduced(rec, r) for r in runs)}


# --- bundles -------------------------------------------------------------------------------------------------
def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _linked(folder: Path, p: Path) -> bool:
    """Whether `p` is reached through a symlink below `folder`: not bundled, since it may point out of the study."""
    rel = p.relative_to(folder).parts
    return any((folder.joinpath(*rel[:i + 1])).is_symlink() for i in range(len(rel)))


def bundle(study, out) -> dict:
    """Write the study as a tar a third party can reproduce from (see the module docstring); returns its path,
    sha256, file count, the record head it holds and the signing key's id."""
    from .signing import sign_file
    rec, folder, out = study.record, study.folder, Path(out).resolve()
    with tempfile.TemporaryDirectory(prefix="openlab-bundle-") as tmp:
        db = Path(tmp) / "record.db"
        with closing(sqlite3.connect(rec.path)) as src, closing(sqlite3.connect(db)) as dst:
            src.backup(dst)                             # a consistent copy, WAL included
        with closing(sqlite3.connect(db)) as con:
            row = con.execute("SELECT hash FROM nodes ORDER BY seq DESC LIMIT 1").fetchone()
        files = {".openlab/record.db": db}
        files |= {f".openlab/{d}/{p.name}": p for d in RECORD for p in sorted((rec.dir / d).glob("*")) if p.is_file()}
        files |= {p.relative_to(folder).as_posix(): p for p in sorted(folder.rglob("*"))
                  if p.is_file() and p != out and not SKIP & set(p.relative_to(folder).parts) and not _linked(folder, p)}
        manifest = {"kind": "openlab-bundle", "study": study.spec.name, "head": row[0] if row else "",
                    "files": {rel: _sha(p) for rel, p in files.items()}}
        mpath = Path(tmp) / MANIFEST
        mpath.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
        sig = sign_file(mpath, {k: manifest[k] for k in ("kind", "study", "head")})
        out.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(out, "w", dereference=True) as t:
            for rel, p in [*files.items(), (MANIFEST, mpath), (SIG, sig)]:
                t.add(p, arcname=rel)
        key = json.loads(sig.read_text(encoding="utf-8"))["key_id"]
    return {"bundle": str(out), "sha256": _sha(out), "bytes": out.stat().st_size, "files": len(files),
            "head": manifest["head"], "key_id": key}


def _unpack(tar: Path, root: Path) -> dict:
    """Extract a bundle into `root` and check it: the manifest's signature, that the bundle holds exactly the files
    the manifest lists with their sha256s, and the record's hash chain up to the manifest's head."""
    from .signing import verify_file
    try:
        with tarfile.open(tar) as t:
            t.extractall(root, filter="data")
    except (OSError, tarfile.TarError) as e:
        raise StudyError(f"{tar} is not a readable bundle", str(e)) from None
    if not (root / SIG).is_file():
        raise StudyError(f"{tar} has no signed manifest", f"a bundle carries {MANIFEST} and {SIG}")
    v = verify_file(root / SIG)
    if not v["ok"]:
        raise StudyError("the bundle's manifest is not as it was signed", v["why"])
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    held = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} - {MANIFEST, SIG}
    bad = sorted(held ^ set(manifest["files"])) + [r for r, s in manifest["files"].items()
                                                   if r in held and _sha(root / r) != s]
    if bad:
        raise StudyError("the bundle does not hold what its manifest lists", ", ".join(bad[:10]))
    chain = Record(root).verify()
    if not chain["ok"] or chain["head"] != manifest["head"]:
        raise StudyError("the bundle's record is not the record its manifest names",
                         f"chain ok {chain['ok']}, head {chain['head'][:12]} vs {manifest['head'][:12]}")
    return {**{k: manifest[k] for k in ("study", "head")}, "key_id": v["key_id"]}


def from_bundle(tar, actor: str = "agent") -> dict:
    """Reproduce every run of a bundle from the bundle alone, in a temporary directory deleted afterwards; returns
    the bundle's study, head and signing key id, and each reproduction."""
    from . import Study
    with tempfile.TemporaryDirectory(prefix="openlab-reproduce-") as d:
        meta = _unpack(Path(tar), Path(d))
        return {"bundle": str(tar), **meta, **Study(Path(d)).reproduce(every=True, actor=actor)}
