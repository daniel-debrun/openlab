"""A Weights & Biases mirror of a study's runs: a view for dashboards and teammates, never the record.

After a run is on the record, openlab starts a detached process (`python -m openlab.study.mirror push FOLDER RUN`) that
writes one offline W&B run for it under the study's .openlab/wandb and then tries to sync it. A W&B problem or outage
therefore never slows or stops a run; what did not sync stays on disk for `openlab mirror sync`.

What it sends: the run's metrics (value and interval) and paired differences as the summary; its key, op, parent,
refs, branch, seed, segment, audit verdict, the spec, code and data shas, the machine and the packages as config.
Never data: not the rows, not the predictions, not the holdout.

When: the `wandb` package is installed (`pip install "openlab-research[mirror]"`), and the workspace's openlab.toml says so:

    [mirror]
    wandb = true            # default: on when W&B has a key here (WANDB_API_KEY or ~/.netrc), else off
    project = "openlab"      # default "openlab"
    entity = ""             # default: the key's own

and the study has not opted out (study.toml `[mirror] wandb = false`). OPENLAB_MIRROR=off turns it off. The W&B run id is the run's key, so a cached
rerun maps to the same W&B run; the group is the study, the job type the op (else the segment).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:                                         # pragma: no cover -- python 3.10
    import tomli as tomllib


def _has_key() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    netrc = Path.home() / ".netrc"
    try:
        return "api.wandb.ai" in netrc.read_text(encoding="utf-8")
    except OSError:
        return False


def _installed() -> bool:
    import importlib.util
    return importlib.util.find_spec("wandb") is not None


def settings(folder: Path) -> dict:
    """{on, why, project, entity, sync} for the study at `folder`."""
    from .workspace import root
    ws = Path(root(folder))
    raw = {}
    if (ws / "openlab.toml").is_file():
        raw = tomllib.loads((ws / "openlab.toml").read_text(encoding="utf-8")).get("mirror", {})
    study = {}
    if (Path(folder) / "study.toml").is_file():
        try:
            study = tomllib.loads((Path(folder) / "study.toml").read_text(encoding="utf-8")).get("mirror", {})
        except tomllib.TOMLDecodeError:
            study = {}
    want = raw.get("wandb")
    out = {"project": raw.get("project") or "openlab", "entity": raw.get("entity") or "",
           "sync": raw.get("sync", True) is not False}
    if os.environ.get("OPENLAB_MIRROR", "").lower() == "off":
        return {**out, "on": False, "why": "OPENLAB_MIRROR=off"}
    if study.get("wandb") is False:
        return {**out, "on": False, "why": "the study opts out ([mirror] wandb = false)"}
    if want is False:
        return {**out, "on": False, "why": "openlab.toml [mirror] wandb = false"}
    if not _installed():
        return {**out, "on": False, "why": 'wandb is not installed (pip install "openlab-research[mirror]")'}
    if want is None and not _has_key():
        return {**out, "on": False, "why": "no W&B key here (WANDB_API_KEY or ~/.netrc); [mirror] wandb = true "
                                           "mirrors offline anyway"}
    return {**out, "on": True, "why": ""}


def after_run(study, run) -> None:
    """Start the mirror of a recorded run, detached; never raises and never waits."""
    try:
        s = settings(study.folder)
        if not s["on"]:
            return
        log = study.record.dir / "mirror.log"
        with open(log, "a", encoding="utf-8") as f:
            subprocess.Popen([sys.executable, "-m", "openlab.study.mirror", "push", str(study.folder), run.id],
                             stdin=subprocess.DEVNULL, stdout=f, stderr=f, start_new_session=True,
                             env={**os.environ, "WANDB_SILENT": "true"})
    except Exception:                                               # noqa: BLE001 -- the mirror never stops a run
        pass


def describe(study, run) -> dict:
    """{id, name, group, job_type, tags, config, summary} of the W&B run for `run` (pure: from the record)."""
    rec, p = study.record, run.payload
    res = rec.links(run.id, rel="of", incoming=True)
    result = rec.get(res[0][0]).payload if res else {}
    parent = rec.links(run.id, rel="branches_from")
    summary = {}
    for m, v in (result.get("metrics") or {}).items():
        if isinstance(v, dict) and isinstance(v.get("value"), (int, float)):
            summary[m] = v["value"]
            for k in ("lo", "hi"):
                if isinstance(v.get(k), (int, float)):
                    summary[f"{m}.{k}"] = v[k]
    for b, d in (result.get("paired") or {}).items():
        for k in ("diff", "lo", "hi", "p"):
            if isinstance(d.get(k), (int, float)):
                summary[f"vs {b}.{k}"] = d[k]
    config = {"run": run.id, "key": p["key"], "segment": p["segment"], "op": p.get("op", ""),
              "parent": parent[0][2] if parent else None, "refs": [d for _, _, d in rec.links(run.id, rel="refs")],
              "branch": p.get("branch", ""), "seed": p.get("seed"), "audit": p["audit"].get("verdict"),
              "sandbox": p["sandbox"], "spec_sha": p["spec_sha"], "code": p["code"], "data": p["data"],
              "box": p.get("box", {}), "packages": p["env"].get("packages", {}), "python": p["env"].get("python")}
    return {"id": f"{p['segment'][0]}{p['key'][:40]}", "name": run.id, "group": study.spec.name,
            "job_type": p.get("op") or p["segment"],
            "tags": [t for t in (p["segment"], p.get("branch"), p.get("op"), p["audit"].get("verdict")) if t],
            "config": config, "summary": summary}


def push(folder: str, run_id: str) -> dict:
    """Write the offline W&B run for `run_id`, then sync it when settings allow; the outcome is appended to
    .openlab/mirror.jsonl. Run by after_run in a detached process."""
    from .. import study as _study
    s = _study.open(folder)
    st = settings(s.folder)
    d = describe(s, s.record.get(run_id))
    out = {"at": time.time(), "run": run_id, "wandb_id": d["id"], "synced": False}
    if not st["on"]:
        return {**out, "skipped": st["why"]}
    os.environ.setdefault("WANDB_SILENT", "true")
    try:
        import wandb
        wdir = s.record.dir / "wandb"
        wdir.mkdir(exist_ok=True)
        r = wandb.init(project=st["project"], entity=st["entity"] or None, id=d["id"], name=d["name"],
                       group=d["group"], job_type=d["job_type"], tags=d["tags"], config=d["config"], dir=str(wdir),
                       mode="offline")
        r.summary.update(d["summary"])
        run_dir = getattr(r, "dir", "")
        r.finish()
        out["dir"] = str(Path(run_dir).parent) if run_dir else ""
        if st["sync"] and _has_key() and out["dir"]:
            p = subprocess.run([sys.executable, "-m", "wandb", "sync", out["dir"]], capture_output=True, text=True,
                               timeout=600)          # this Python's wandb, whatever PATH the detached process has
            out["synced"] = p.returncode == 0
            if p.returncode:
                out["error"] = (p.stderr or p.stdout)[-500:]
    except Exception as e:                                          # noqa: BLE001 -- logged, retried by `sync`
        out["error"] = f"{type(e).__name__}: {e}"[:500]
    finally:
        with open(s.record.dir / "mirror.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")
    return out


def pending(folder) -> list:
    """The study's runs whose mirror has not synced: never pushed, or whose last attempt failed (by mirror.jsonl);
    each {run, and the last attempt's fields when there was one}."""
    from .. import study as _study
    s = _study.open(folder)
    path = s.record.dir / "mirror.jsonl"
    last = {}
    for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
        o = json.loads(line)
        last[o["run"]] = o
    return [last.get(r.id, {"run": r.id}) for r in s.record.find("run") if not last.get(r.id, {}).get("synced")]


def sync(folder) -> list:
    """Push again each run that has not synced (the W&B id is the run's key, so a run is never doubled); the
    outcomes."""
    return [push(str(folder), o["run"]) for o in pending(folder)]


def main(argv) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="openlab mirror", description="The W&B mirror of a study's runs (never the record).")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, help_ in (("status", "whether the study mirrors, and what has not synced"),
                        ("sync", "sync the offline W&B runs that did not")):
        sub.add_parser(name, help=help_).add_argument("folder")
    ps = sub.add_parser("push", help=argparse.SUPPRESS)
    ps.add_argument("folder")
    ps.add_argument("run")
    a = p.parse_args(argv)
    if a.cmd == "push":
        print(json.dumps(push(a.folder, a.run)))
        return 0
    if a.cmd == "sync":
        out = sync(a.folder)
        print(f"{sum(o['synced'] for o in out)} of {len(out)} synced" if out else "nothing to sync")
        return 0 if all(o["synced"] for o in out) else 1
    st = settings(Path(a.folder).resolve())
    print(f"mirror: {'on' if st['on'] else 'off'}" + (f" ({st['why']})" if st["why"] else
                                                      f", project {st['project']}" + (f", entity {st['entity']}" if st['entity'] else "")))
    left = pending(a.folder)
    print(f"{len(left)} run(s) not synced" + (": `openlab mirror sync " + a.folder + "`" if left else ""))
    errors = [o for o in left if o.get("error")]
    if errors:
        print(f"last error ({errors[-1]['run']}): {errors[-1]['error'][-300:]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
