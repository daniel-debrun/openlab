"""A research run that outlives the agent that asked for it.

An agent's `study_run` (through the MCP, or the workspace server) is served by a process that ends with the agent
(its MCP server) or with the server (a restart of `openlab show`). A run lasting hours inside it died with it, at the
step it had reached, with nothing on the record. So an agent's run goes to a process of its own, in its own session:
`python -m openlab.study.detached STUDY JOB`. The caller waits for it up to WAIT and returns its result as before; a
run still going then is returned as a job (`job:run-<hex>`) that `agent_wait` waits on, since an agent's MCP client
aborts a call that answers nothing for 30 minutes (and the workspace server's client for 15). If the caller ends
first, the run goes on and records its result on the study's record all the same (the agent's `run` event names
the run's pid, so `openlab show` shows it working until it ends). A person's `openlab study run` stays in their
terminal.

The job is a file under the study's `.openlab/jobs/`: the run's arguments, who asked (the acting agent, as
attributed to the call), its pid and start; the run writes its result or its refusal next to it (`.out`), and its
stderr (the walk's progress) to `.log`. The caller that waited reads the result and removes the files; a job
collected later by `agent_wait` leaves its log and a small `.done` summary, so a second wait says the same. The
workspace's `.openlab/runs/<job>` names the study a job id belongs to. Files of jobs ended over a day ago are
removed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

POLL = 0.5
WAIT = 840.0                    # seconds a caller waits before it returns the job: under the clients' timeouts
KEEP = 86_400.0                 # seconds an ended job's files are kept
_PROCS: dict = {}               # this process's children, so an ended one is reaped (not a live-looking zombie)


def _wait() -> float:
    try:
        return float(os.environ.get("OPENLAB_RUN_WAIT") or WAIT)
    except ValueError:
        return WAIT


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jobs(folder: Path) -> Path:
    return Path(folder).resolve() / ".openlab" / "jobs"


def _index(folder: Path) -> Path:
    from ..agents.store import root_of
    return root_of(folder) / ".openlab" / "runs"


def _write(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, default=str), encoding="utf-8")
    tmp.replace(path)


def _read(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def alive(pid, job: str = "") -> bool:
    """Whether the run's process is still going (an ended child not yet reaped is not)."""
    p = _PROCS.get(job)
    if p is not None:
        return p.poll() is None
    try:
        pid = int(pid or 0)
        if pid <= 0:
            return False
        os.kill(pid, 0)
    except (ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    try:                                                    # Linux: not a zombie, and still a detached run
        stat = Path(f"/proc/{pid}/stat").read_text()
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
        return stat.rsplit(")", 1)[1].split()[0] != "Z" and b"openlab.study.detached" in cmd
    except (OSError, IndexError):
        return True


def _files(jobs: Path, job: str) -> tuple:
    base = jobs / job
    return base.with_suffix(".json"), base.with_suffix(".out"), base.with_suffix(".log"), base.with_suffix(".done")


def _remove(folder: Path, job: str, keep_done: bool = False) -> None:
    spec, out, log, done = _files(_jobs(folder), job)
    for p in (spec, out) if keep_done else (spec, out, log, done):     # a collected job's log stays a day
        p.unlink(missing_ok=True)
    if not keep_done:
        (_index(folder) / job).unlink(missing_ok=True)


def _prune(folder: Path) -> None:
    """Remove the files of jobs that ended over a day ago (collected or not); a live job's are kept."""
    jobs, cut = _jobs(folder), time.time() - KEEP
    for p in list(jobs.glob("run-*.*")):
        job = p.name.split(".", 1)[0]
        try:
            if p.stat().st_mtime > cut:
                continue
        except OSError:
            continue
        spec = _read(jobs / f"{job}.json") or {}
        if spec and not (jobs / f"{job}.out").is_file() and alive(spec.get("pid"), job):
            continue
        _remove(folder, job)


def run(folder: Path, kwargs: dict, actor: str) -> dict:
    """Run the study in a detached process and wait up to WAIT for its result (a StudyError when it refuses); a
    run still going then is returned as its job, {running, job, pid, started, next}."""
    from .errors import StudyError
    folder = Path(folder).resolve()
    jobs = _jobs(folder)
    jobs.mkdir(parents=True, exist_ok=True)
    _prune(folder)
    job = f"run-{uuid.uuid4().hex[:12]}"
    spec, out, log, _ = _files(jobs, job)
    meta = {"kwargs": kwargs, "actor": actor, "started": _now()}
    _write(spec, meta)
    with open(log, "ab") as f:
        proc = subprocess.Popen([sys.executable, "-m", "openlab.study.detached", str(folder), str(spec)],
                                cwd=str(folder), stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                                start_new_session=True, env=os.environ.copy())
    _PROCS[job] = proc
    _write(spec, {**meta, "pid": proc.pid})
    try:
        idx = _index(folder)
        idx.mkdir(parents=True, exist_ok=True)
        (idx / job).write_text(str(folder), encoding="utf-8")
    except OSError:
        pass
    end = time.monotonic() + _wait()
    while proc.poll() is None:
        if time.monotonic() >= end:
            return {"running": True, "job": f"job:{job}", "pid": proc.pid, "started": meta["started"],
                    "next": f"still running: agent_wait(ids=['job:{job}']) waits for it (call again until done); "
                            "study_runs lists it while it runs"}
        time.sleep(POLL)
    data = _outcome(folder, job, proc.returncode)
    _remove(folder, job)
    if "error" in data:
        raise StudyError(data["error"], data.get("meaning", ""), data.get("fix", ""))
    return data["result"]


def _outcome(folder: Path, job: str, code=None) -> dict:
    """The ended run's result or refusal, as the run wrote it (a refusal when it wrote nothing)."""
    _, out, log, _ = _files(_jobs(folder), job)
    data = _read(out)
    if data is None:
        tail = log.read_text(encoding="utf-8", errors="replace")[-600:] if log.is_file() else ""
        data = {"error": "the run's process ended without a result",
                "meaning": (f"exit {code}: " if code is not None else "") + tail.strip(), "fix": f"its log is {log}"}
    return data


def find(root, job: str) -> Optional[Path]:
    """The study folder job `job` (`run-<hex>`) runs in, from the workspace's index."""
    from ..agents.store import root_of
    try:
        text = (root_of(root) / ".openlab" / "runs" / job).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return Path(text) if text else None


def brief(data: dict) -> dict:
    """A run's result in a few fields: its trial line, its result id, the audit, its primary metric."""
    if "error" in data:
        return {"ok": False, "error": data["error"], "meaning": data.get("meaning", ""), "try": data.get("fix", "")}
    r = data.get("result") or {}
    p = r.get("payload") or {}
    name, m = next(iter((p.get("metrics") or {}).items()), (None, None))
    m = m if isinstance(m, dict) else {}
    return {"ok": True, "result": r.get("id"), "trial": (r.get("trial") or {}).get("summary"), "audit": r.get("audit"),
            "metric": {"name": name, **{k: m[k] for k in ("value", "lo", "hi") if k in m}} if name else None}


def poll(folder: Path, job: str) -> dict:
    """Job `job` now: {state: running|done|failed, ...}; an ended job is collected (its brief kept in `.done`)."""
    spec, out, _, done = _files(_jobs(folder), job)
    kept = _read(done)
    if kept is not None:
        return kept
    meta = _read(spec)
    if meta is None:
        return {"job": f"job:{job}", "state": "gone", "error": f"no job {job}",
                "meaning": "it is not in the study's jobs (ended over a day ago, or never started)"}
    if not out.is_file() and alive(meta.get("pid"), job):
        return {"job": f"job:{job}", "state": "running", **_row(folder, job, meta)}
    p = _PROCS.pop(job, None)
    b = brief(_outcome(folder, job, p.returncode if p is not None else None))
    res = {"job": f"job:{job}", "state": "done" if b["ok"] else "failed", "study": str(folder), **b}
    _write(done, res)
    _remove(folder, job, keep_done=True)
    return res


def _progress(log: Path) -> str:
    """The walk's (or the queue's) last progress line in the run's stderr."""
    try:
        with open(log, "rb") as f:
            f.seek(max(0, f.seek(0, 2) - 4096))
            lines = f.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    for line in reversed(lines):
        line = line.strip()
        if line.startswith(("openlab: walk", "queued")):
            return line.removeprefix("openlab: ")[:200]
    return ""


def _row(folder: Path, job: str, meta: dict) -> dict:
    from ..agents.store import age
    return {"by": meta.get("actor") or None, "started": meta.get("started"),
            "minutes": round(age(meta.get("started") or "") / 60, 1),
            "note": (meta.get("kwargs") or {}).get("note") or "",
            "progress": _progress(_files(_jobs(folder), job)[2]) or None, "pid": meta.get("pid")}


def live(folder: Path) -> list:
    """The study's runs in flight (an agent's detached run whose process goes on), oldest first."""
    jobs, out = _jobs(folder), []
    for spec in sorted(jobs.glob("run-*.json")) if jobs.is_dir() else []:
        job = spec.stem
        meta = _read(spec)
        if meta is None or (jobs / f"{job}.out").is_file() or not alive(meta.get("pid"), job):
            continue
        out.append({"job": f"job:{job}", **_row(folder, job, meta)})
    return sorted(out, key=lambda r: r["started"] or "")


def line(r: dict) -> str:
    """One in-flight run as a line of `openlab study runs` / `status`."""
    return (f"running {r['job']}" + (f" by {r['by']}" if r.get("by") else "") + f", started {r['started']} "
            f"({r['minutes']:.0f} min)" + (f": {r['progress']}" if r.get("progress") else "")
            + (f"  \"{r['note']}\"" if r.get("note") else ""))


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    folder, job = Path(argv[0]), Path(argv[1])
    spec = json.loads(job.read_text(encoding="utf-8"))
    from .. import study
    from ..agents.identity import acting_as
    from .errors import StudyError
    try:
        with acting_as(spec.get("actor") or ""):
            data = {"result": study.open(folder).run(**spec["kwargs"])}
    except StudyError as e:
        data = {"error": e.error, "meaning": e.meaning, "fix": e.fix}
    _write(job.with_suffix(".out"), data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
