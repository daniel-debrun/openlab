"""Registered datasets: each is named in study.toml, materialised to `.openlab/data/<name>.parquet`, recorded as a
`dataset` node with its sha256 and card, and re-checked against that sha on every load.

A dataset comes from one source kind, each a registered `Source`: a provider endpoint (`openlab.sources`), a
snapshot on disk, or a file made by a recipe the study holds. The recipe runs (with this Python, in the study
folder) when the file is missing or on refresh, and its source goes into the content store under its sha256. A
file without a recipe is refused: nothing would say how it was made. A recipe edited after its file was made is
not re-run without refresh; `stale` says so.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import polars as pl

from .. import sources
from ..sources import snapshot
from .errors import StudyError
from .record import Node, canonical
from .spec import DatasetSpec
from .view import Timeline

if TYPE_CHECKING:
    from . import Study

READERS = {".parquet": pl.read_parquet, ".csv": pl.read_csv}


@dataclass(frozen=True)
class Source:
    load: Callable                                  # (study folder, DatasetSpec, refresh) -> a frame
    fields: tuple                                   # the DatasetSpec fields that say where the data came from
    limits: Callable = lambda ds: {}                # what the source cannot tell you, as `sources.Limits` data


def _file(folder: Path, ds: DatasetSpec, refresh: bool) -> pl.DataFrame:
    """The file, made first by its recipe when it is missing or on refresh."""
    if not ds.recipe:
        raise StudyError(f"data.{ds.name}: a file needs the recipe that makes it", "nothing would say how it was made",
                         f'add recipe = "recipes/<script>.py" to data.{ds.name}')
    path, recipe = folder / ds.file, folder / ds.recipe
    if not recipe.is_file():
        raise StudyError(f"data.{ds.name}: no recipe {ds.recipe}", fix="write it, or fix the path in study.toml")
    if refresh or not path.is_file():
        done = subprocess.run([sys.executable, str(recipe)], cwd=folder, capture_output=True, text=True)
        if done.returncode:
            raise StudyError(f"recipe {ds.recipe} failed (exit {done.returncode})", done.stderr[-2000:],
                             "fix the recipe and materialise again")
        if not path.is_file():
            raise StudyError(f"recipe {ds.recipe} did not write {ds.file}",
                             fix=f"make it write {ds.file}, or point data.{ds.name}.file at what it writes")
    read = READERS.get(path.suffix)
    if read is None:
        raise StudyError(f"data.{ds.name}: cannot read {path.suffix or 'extensionless'} files",
                         f"readable: {', '.join(READERS)}")
    return read(path)


def _snapshot_limits(ds: DatasetSpec) -> dict:
    """The limits of the provider the snapshot pulled from, as its manifest recorded them."""
    return snapshot._manifest(ds.snapshot)["datasets"].get(ds.dataset or ds.name, {}).get("limits", {})


SOURCES = {
    "provider": Source(lambda folder, ds, refresh: sources.fetch(ds.provider, ds.endpoint, **ds.args),
                       ("provider", "endpoint", "args"), lambda ds: sources.limits(ds.provider)),
    "snapshot": Source(lambda folder, ds, refresh: snapshot.read(ds.snapshot, ds.dataset or ds.name),
                       ("snapshot", "dataset"), _snapshot_limits),
    "file": Source(_file, ("file", "recipe")),
}


def _source(ds: DatasetSpec) -> tuple[str, Source]:
    kind = next(k for k in SOURCES if getattr(ds, k))
    return kind, SOURCES[kind]


def _frame(out) -> pl.DataFrame:
    """A source's reply as a polars frame."""
    out = getattr(out, "frame", out)                # providers that wrap their frame (yahoo) carry .frame
    if type(out).__module__.split(".")[0] == "pandas":
        out = pl.from_pandas(out, include_index=True)
    if not isinstance(out, pl.DataFrame):
        raise StudyError(f"the source returned a {type(out).__name__}, not a table")
    return out


def _path(study: "Study", name: str) -> Path:
    return study.record.dir / "data" / f"{name}.parquet"


def card(df: pl.DataFrame, ds: DatasetSpec) -> dict:
    """What to read before the rows: schema, rows, time coverage, missing values per column (null or NaN), keys,
    licence (the spec's, else the source's) and the source's declared limits."""
    limits = _source(ds)[1].limits(ds)
    missing = df.select([(pl.col(c).is_null() | (pl.col(c).is_nan() if t.is_float() else False)).sum().alias(c)
                         for c, t in df.schema.items()])
    return {"rows": df.height, "schema": {c: str(t) for c, t in df.schema.items()},
            "missing": missing.row(0, named=True),
            "time": {"column": ds.time, "min": str(df[ds.time].min()), "max": str(df[ds.time].max())}
            if ds.time else None,
            "key": {"columns": list(ds.key), "distinct": df.select(ds.key).n_unique()} if ds.key else None,
            "licence": ds.licence or limits.get("licence", ""), "limits": limits}


def _check(df: pl.DataFrame, ds: DatasetSpec) -> None:
    """Fail now rather than mid-run: the named columns exist and the known_at rule applies."""
    if df.height == 0:
        raise StudyError(f"data.{ds.name} has no rows", fix="check the source's arguments (dates, ids) or the recipe")
    gone = [c for c in (ds.time, *ds.key, *ds.vintages) if c and c not in df.columns]
    if gone:
        raise StudyError(f"data.{ds.name} has no column {', '.join(gone)}", f"its columns: {', '.join(df.columns)}",
                         "fix study.toml or the source")
    Timeline({ds.name: df}, {ds.name: ds})


def materialise(study: "Study", name: str, refresh: bool = False) -> Node:
    """Read dataset `name` from its source, write `.openlab/data/<name>.parquet`, and record a `dataset` node; an
    unchanged sha returns the last node. `refresh` re-runs a file's recipe."""
    spec = study.spec
    if name not in spec.data:
        raise StudyError(f"no dataset `{name}` in study.toml", f"datasets: {', '.join(spec.data) or 'none'}")
    ds = spec.data[name]
    kind, src = _source(ds)
    source = {"kind": kind, **{f: getattr(ds, f) for f in src.fields}}
    last, out = study.record.last("dataset", name=name), _path(study, name)
    if (kind == "provider" and not refresh and last is not None
            and json.loads(canonical(last.payload["source"])) == json.loads(canonical(source))
            and out.is_file() and hashlib.sha256(out.read_bytes()).hexdigest() == last.payload["sha256"]):
        return last        # a provider's pull is frozen: re-fetched only on refresh (or a changed definition)
    try:
        df = _frame(src.load(study.folder, ds, refresh))
    except StudyError:
        raise
    except Exception as e:                          # noqa: BLE001 -- any source failure becomes one clear error
        raise StudyError(f"data.{name}: the {kind} could not be read", f"{type(e).__name__}: {e}") from e
    _check(df, ds)
    buf = io.BytesIO()
    df.write_parquet(buf)
    data = buf.getvalue()
    sha = hashlib.sha256(data).hexdigest()
    out = _path(study, name)
    out.parent.mkdir(exist_ok=True)
    tmp = out.with_name(f".{name}.{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, out)
    if last is not None and last.payload["sha256"] == sha:
        return last
    recipe_sha = study.record.put_blob((study.folder / ds.recipe).read_bytes()) if ds.recipe else ""
    return study.record.add("dataset", {"name": name, "source": source, "sha256": sha, "rows": df.height,
                                        "columns": df.columns, "card": card(df, ds), "recipe_sha": recipe_sha})


def load(study: "Study", name: str, sha: str = "") -> pl.DataFrame:
    """The materialised dataset, once its bytes are checked against the sha its last dataset node recorded (or
    `sha`, the one a run recorded)."""
    node, path = study.record.last("dataset", name=name), _path(study, name)
    if node is None or not path.is_file():
        raise StudyError(f"dataset {name} is not materialised", fix=f"materialise it: Study.materialise({name!r})")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != (sha or node.payload["sha256"]):
        if sha:
            raise StudyError(f"dataset {name} is not the data the run used",
                             f"{path.name} no longer matches the sha256 {sha[:12]} the run recorded",
                             "restore the data the run was made with; a run is reproduced from its own data only")
        raise StudyError(f"dataset {name} changed since it was registered",
                         f"{path.name} no longer matches the sha256 in {node.id}",
                         f"re-materialise it: Study.materialise({name!r})")
    return pl.read_parquet(io.BytesIO(data))


def stale(study: "Study", names=None, how: str = "refresh=true") -> list:
    """A warning per dataset (of `names`, default all) whose recipe changed after the file it made: materialise
    without refresh reads the old file."""
    out = []
    for name in names or list(study.spec.data):
        ds = study.spec.data.get(name)
        if ds is None or not ds.recipe or not ds.file:
            continue
        recipe, path = study.folder / ds.recipe, study.folder / ds.file
        try:
            if recipe.stat().st_mtime > path.stat().st_mtime:
                out.append(f"{ds.recipe} changed after {ds.file} was made: materialise with {how} to rebuild it")
        except OSError:
            continue
    return out


def frames(study: "Study") -> dict[str, pl.DataFrame]:
    """Every dataset in the spec, loaded and checked."""
    return {name: load(study, name) for name in study.spec.data}
