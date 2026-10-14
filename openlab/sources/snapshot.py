"""Snapshots: named, versioned copies of provider data on disk, so agents read files instead of calling APIs.

A snapshot is built once from a spec (a list of pulls, each a provider endpoint and its arguments) through
the shared request queue, then read by any number of agents without touching the network::

    {"name": "macro-2026-10",
     "datasets": [{"dataset": "unrate", "provider": "fred", "endpoint": "observations",
                   "args": {"series_id": "UNRATE", "start": "2015-01-01", "end": "2024-12-31"}}]}

Each dataset is `<root>/<name>/<dataset>.parquet`; `manifest.json` records per dataset the provider, endpoint,
arguments, rows, columns, sha256, fetch time and the provider's limits, so a result can name exactly what it
read. A build resumes: datasets already present are kept unless `redo`; `workers` runs pulls in parallel; it prints one progress line per pull, records an
empty pull as failed, and several builds of one snapshot may run at once (the manifest is locked). Root: `$OPENLAB_SNAPSHOTS` (the
environment, or the same key in ~/.config/openlab/research.env), else `~/openlab-data/snapshots`."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import polars as pl


def root() -> Path:
    from .http import data_root
    return data_root("OPENLAB_SNAPSHOTS", "snapshots")


def _manifest(name: str) -> dict:
    f = root() / name / "manifest.json"
    return json.loads(f.read_text()) if f.is_file() else {"name": name, "datasets": {}}


def build(spec, redo: bool = False, workers: int = 1) -> dict:
    """Build (or complete) a snapshot from `spec`: a dict, a JSON string, or a path to a JSON file. Each pull
    goes through the provider's shared queue; a pull that fails is recorded and the rest still run. `workers` > 1
    runs that many pulls at once (threads): each provider's rate queue still spaces its own requests, so this
    speeds up pulls from different providers or slow server-side queries, and costs memory per pull in flight."""
    from concurrent.futures import ThreadPoolExecutor
    from . import fetch, limits
    if isinstance(spec, (str, Path)):
        spec = json.loads(Path(spec).read_text()) if Path(str(spec)).is_file() else json.loads(str(spec))
    import fcntl
    name = spec["name"]
    d = root() / name
    d.mkdir(parents=True, exist_ok=True)
    done, failed = [], {}

    def pull(i: int, item: dict) -> None:
        ds = item["dataset"]
        out = d / f"{ds}.parquet"
        if out.is_file() and not redo and ds in _manifest(name)["datasets"]:
            return
        print(f"[snapshot {name}] {i}/{len(spec['datasets'])} {ds}: {item['provider']}.{item['endpoint']}", flush=True)
        try:
            df = fetch(item["provider"], item["endpoint"], **item.get("args", {}))
            df = getattr(df, "frame", df)                # providers that return a wrapper (yahoo) carry .frame
        except Exception as e:                       # noqa: BLE001 -- one pull must not stop the snapshot
            failed[ds] = f"{type(e).__name__}: {str(e)[:300]}"
            return
        if df.height == 0:                           # an empty pull is a failure to say, not a dataset to keep
            failed[ds] = "0 rows: check the arguments (dates, symbols) or the source's coverage"
            return
        df.write_parquet(out)
        rec = {"provider": item["provider"], "endpoint": item["endpoint"], "args": item.get("args", {}),
               "rows": df.height, "columns": df.columns, "sha256": hashlib.sha256(out.read_bytes()).hexdigest()[:16],
               "fetched": dt.datetime.now().replace(microsecond=0).isoformat(), "limits": limits(item["provider"])}
        with (d / ".manifest.lock").open("w") as lock:   # concurrent builds (and workers) each add their entries
            fcntl.flock(lock, fcntl.LOCK_EX)
            man = _manifest(name)
            man["datasets"][ds] = rec
            (d / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
        done.append(ds)

    with ThreadPoolExecutor(max(1, int(workers))) as ex:
        list(ex.map(pull, range(1, len(spec["datasets"]) + 1), spec["datasets"]))
    man = _manifest(name)
    (d / "spec.json").write_text(json.dumps(spec, indent=1))
    return {"snapshot": name, "path": str(d), "built": done, "failed": failed,
            "datasets": {k: v["rows"] for k, v in man["datasets"].items()}}


def read(snapshot: str, dataset: str, columns: str = "") -> pl.DataFrame:
    """One dataset of a snapshot, from disk (no network). `columns` (comma separated) reads only those."""
    f = root() / snapshot / f"{dataset}.parquet"
    if not f.is_file():
        have = ", ".join(_manifest(snapshot)["datasets"]) or "none"
        raise FileNotFoundError(f"snapshot {snapshot} has no dataset {dataset!r} (it has: {have})")
    cols = [c.strip() for c in columns.split(",") if c.strip()] or None
    return pl.read_parquet(f, columns=cols)


def listing(snapshot: str = "") -> pl.DataFrame:
    """Every snapshot (name, datasets, rows, newest fetch), or with `snapshot` its datasets with provenance."""
    if snapshot:
        man = _manifest(snapshot)
        return pl.DataFrame([{"dataset": k, "provider": v["provider"], "endpoint": v["endpoint"],
                              "args": json.dumps(v["args"]), "rows": v["rows"], "sha256": v["sha256"],
                              "fetched": v["fetched"]} for k, v in man["datasets"].items()])
    rows = []
    for d in sorted(root().glob("*/manifest.json")) if root().is_dir() else []:
        m = json.loads(d.read_text())
        rows.append({"snapshot": m["name"], "datasets": len(m["datasets"]),
                     "rows": sum(v["rows"] for v in m["datasets"].values()),
                     "newest": max((v["fetched"] for v in m["datasets"].values()), default="")})
    return pl.DataFrame(rows)
