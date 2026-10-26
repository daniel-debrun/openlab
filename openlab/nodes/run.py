"""Commands inside an allocation: `openlab node exec` and the runs it leaves.

A run copies the caller's folder to the node (rsync; the record, keys, the person's files and venvs never go), starts
the command detached in the allocation (a local or ssh node: its cores with taskset and its GPUs with
CUDA_VISIBLE_DEVICES; a Slurm node: a job step with `srun --jobid --overlap`), follows its output for a while, and
when it has ended copies new and changed files back (newer than here, under a size cap; never .git). A run outlives
the call that started it: `wait`, `logs`, `stop` and `pull` reach it again. Every start and end is an event of the
agent that ran it.
"""

from __future__ import annotations

import hashlib
import secrets
import shlex
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path
from typing import Optional

from . import alloc as A
from .inventory import EXCLUDE, Node, NodeError, get, home

FOLLOW = 90.0                       # seconds `exec` follows a run before leaving it running
POLL = 2.0
CHUNK = 256 * 1024
PULL_MAX_MB = 200


def _tag(cwd: Path) -> str:
    return f"{cwd.name or 'root'}-{hashlib.sha1(str(cwd).encode()).hexdigest()[:6]}"


def where(a: dict, cwd: Path) -> tuple:
    """(node, work dir, runs dir) for allocation `a` and a caller in `cwd`: a local allocation works in `cwd` itself."""
    n = get(a["node"])
    if n.backend == "local":
        return n, str(cwd), str(home() / "allocations" / a["id"] / "runs")
    return n, n.path(f"{a['id']}/{_tag(cwd)}"), n.path(f"{a['id']}/.runs")


def _excludes(n: Node, pull: bool = False) -> list:
    out = []
    for e in list(EXCLUDE) + list(n.exclude or []) + ([".git/"] if pull else []):
        out += ["--exclude", e]
    return out


def _rsync(args: list, timeout: float = 3600) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["rsync", *args], capture_output=True, text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        raise NodeError(f"rsync took over {timeout:.0f} s", " ".join(args[-2:])) from None


def push(n: Node, cwd: Path, wdir: str) -> dict:
    """Copy `cwd` to `wdir` on the node (nothing is deleted there: outputs made on the node stay). A file newer there
    is left alone (--update): a run still writing it, whose earlier copy came back here, keeps its open file; rsync
    would otherwise rename a new copy over it, and on a shared filesystem the run's handle goes stale."""
    n.sh(f"mkdir -p {shlex.quote(wdir)}", timeout=60)
    t0 = time.monotonic()
    p = _rsync(["-a", "--update", "--stats", "-e", n.rsh(), *_excludes(n), f"{cwd}/", f"{n.host}:{wdir}/"])
    if p.returncode:
        raise NodeError(f"copying {cwd} to {n.name} failed: {(p.stderr or p.stdout).strip()[-400:]}")
    sent = next((ln.split(":", 1)[1].strip() for ln in p.stdout.splitlines() if ln.startswith("Total bytes sent")), "")
    return {"seconds": round(time.monotonic() - t0, 1), "sent": sent}


def pull(n: Node, wdir: str, cwd: Path, max_mb: float = PULL_MAX_MB) -> dict:
    """Copy what is newer on the node back into `cwd` (files over `max_mb` are left there, and named)."""
    if n.backend == "local":
        return {"files": [], "skipped": []}
    p = _rsync(["-a", "--update", f"--max-size={int(max_mb)}M", "--out-format=%n", "-e", n.rsh(),
                *_excludes(n, pull=True), f"{n.host}:{wdir}/", f"{cwd}/"])
    if p.returncode not in (0, 24):                                 # 24: a file vanished while it was read
        raise NodeError(f"copying results back from {n.name} failed: {(p.stderr or p.stdout).strip()[-400:]}")
    files = [ln for ln in p.stdout.splitlines() if ln and not ln.endswith("/")]
    big = n.sh(f"cd {shlex.quote(wdir)} && find . -type f -size +{int(max_mb)}M -not -path './.git/*' | head -20",
               timeout=60, check=False).split()
    return {"files": files, "skipped": [b[2:] if b.startswith("./") else b for b in big]}


def _script(a: dict, n: Node, wdir: str, rdir: str, rid: str, cmd: str) -> str:
    """The run's own script on the node: the command in the allocation, its exit code written beside its log."""
    q = shlex.quote
    if n.backend == "slurm":
        inner = (f"srun --jobid={q(a['ref'])} --overlap --ntasks=1 --cpus-per-task={a['cores']} "
                 f"--chdir={q(wdir)} bash -c {q(cmd)}")
    else:
        pre = f"taskset -c {','.join(str(c) for c in a['cpu_list'])} " if a["cpu_list"] and n.backend == "ssh" else ""
        inner = f"cd {q(wdir)} && {pre}bash -c {q(cmd)}"
    env = f"export OPENLAB_ALLOCATION={q(a['id'])} OPENLAB_RUN={q(rid)}\n"
    if n.backend == "ssh":
        env += f"export CUDA_VISIBLE_DEVICES={q(','.join(str(g) for g in a['gpu_list']))}\n"
    return f"{env}{inner}\necho $? > {q(rdir)}/{rid}.exit\n"


def start(a: dict, agent: str, workspace, cwd: Path, cmd: str, sync: bool = True) -> dict:
    """Copy `cwd` to the node and start `cmd` there, detached; the run's row."""
    if a is None or a["state"] != "running":
        raise NodeError("your allocation is not running" + (f" ({a['state']}: {a['why']})" if a else ""),
                        "a command runs inside a running allocation",
                        "wait for it (openlab node list says where it stands), or ask for it again")
    n, wdir, rdir = where(a, cwd)
    pushed = push(n, cwd, wdir) if sync and n.backend != "local" else {}
    rid = "r-" + secrets.token_hex(3)
    q = shlex.quote
    script = _script(a, n, wdir, rdir, rid, cmd)
    launcher = (f"mkdir -p {q(rdir)} && cat > {q(rdir)}/{rid}.sh <<'OPENLAB_EOF'\n{script}OPENLAB_EOF\n"
                f"(cd {q(wdir)} && exec setsid bash {q(rdir)}/{rid}.sh) > {q(rdir)}/{rid}.log 2>&1 < /dev/null & "
                f"echo $! > {q(rdir)}/{rid}.pid; cat {q(rdir)}/{rid}.pid")       # nothing holds the call's pipes
    pid = int((n.sh(launcher, timeout=60).strip().splitlines() or ["0"])[-1] or 0)
    with closing(A._db()) as con:
        con.execute("INSERT INTO runs (id, alloc, agent, workspace, cwd, wdir, rdir, cmd, state, pid, at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)", (rid, a["id"], agent, str(workspace or ""), str(cwd), wdir, rdir,
                                                       cmd, "running", pid, time.time()))
    _event(agent, workspace, "node_run", f"{a['node']}: {cmd[:200]} (run {rid})",
           {"run": rid, "allocation": a["id"], "node": a["node"], "host": a.get("host") or "", "command": cmd,
            "cwd": str(cwd), "gpus": a["gpu_list"], "gpu_names": a["gpu_names"], "pushed": pushed})
    return {**get_run(rid), "pushed": pushed}


def get_run(rid: str) -> Optional[dict]:
    with closing(A._db()) as con:
        r = con.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()
    return dict(r) if r else None


def runs(aid: Optional[str] = None, agent: Optional[str] = None, all: bool = True) -> list:
    with closing(A._db()) as con:
        q, args = "SELECT * FROM runs WHERE 1=1", []
        if aid:
            q, args = q + " AND alloc=?", args + [aid]
        if agent:
            q, args = q + " AND agent=?", args + [agent]
        if not all:
            q += " AND state='running'"
        return [dict(r) for r in con.execute(q + " ORDER BY at", args)]


def peek(r: dict, offset: int) -> tuple:
    """(exit code or None, the log's bytes from `offset`, at most CHUNK) of run `r`."""
    a = A.get_alloc(r["alloc"])
    n = get(a["node"]) if a else None
    if n is None:
        return None, b""
    q = shlex.quote
    base = f"{q(r['rdir'])}/{r['id']}"
    script = (f"cat {base}.exit 2>/dev/null || echo -; printf '\\036'; "
              f"tail -c +{offset + 1} {base}.log 2>/dev/null | head -c {CHUNK}")
    argv = ["bash", "-c", script] if n.backend == "local" else n.ssh() + ["bash -c " + q(script)]
    try:
        p = subprocess.run(argv, capture_output=True, timeout=60, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return None, b""
    head, _, body = p.stdout.partition(b"\x1e")
    code = head.decode(errors="replace").strip()
    return (int(code) if code.lstrip("-").isdigit() and code != "-" else None), body


def follow(r: dict, seconds: float, out=None, offset: int = 0) -> tuple:
    """Stream run `r`'s log to `out` for up to `seconds`; (exit code or None if still running, bytes read)."""
    out = out or sys.stdout
    end = time.monotonic() + seconds
    while True:
        code, chunk = peek(r, offset)
        while chunk:
            offset += len(chunk)
            out.write(chunk.decode(errors="replace"))
            out.flush()
            if len(chunk) < CHUNK:
                break
            code, chunk = peek(r, offset)
        if code is not None:
            return code, offset
        if time.monotonic() >= end:
            return None, offset
        time.sleep(POLL)


def finish(r: dict, code: int, do_pull: bool = True, max_mb: float = PULL_MAX_MB) -> dict:
    """Record run `r`'s end and copy its results back."""
    a = A.get_alloc(r["alloc"])
    got = {"files": [], "skipped": []}
    if do_pull and a is not None:
        try:
            got = pull(get(a["node"]), r["wdir"], Path(r["cwd"]), max_mb)
        except NodeError as e:
            got = {"files": [], "skipped": [], "error": e.error}
    with closing(A._db()) as con:
        first = con.execute("UPDATE runs SET state='done', exit=?, ended=COALESCE(ended, ?), pulled=? WHERE id=? AND "
                            "state='running'", (code, time.time(), time.time(), r["id"])).rowcount
        if not first:
            con.execute("UPDATE runs SET pulled=? WHERE id=?", (time.time(), r["id"]))
    r2 = get_run(r["id"])
    if first:
        _event(r["agent"], r["workspace"], "node_run_end",
               f"{(a or {}).get('node', '?')}: run {r['id']} exit {code} in {(r2['ended'] - r2['at']):.0f} s, "
               f"{len(got['files'])} file(s) back", {"run": r["id"], "allocation": r["alloc"], "exit": code,
                                                     "seconds": round(r2["ended"] - r2["at"], 1), **got})
    return {**r2, **got}


def stop(r: dict) -> None:
    """Stop run `r` (its whole process group on the node; for Slurm, the step's srun)."""
    a = A.get_alloc(r["alloc"])
    if a is None or not r.get("pid"):
        return
    n = get(a["node"])
    base = f"{shlex.quote(r['rdir'])}/{r['id']}"
    n.sh(f"kill -TERM -{int(r['pid'])} 2>/dev/null; sleep 2; kill -KILL -{int(r['pid'])} 2>/dev/null; "
         f"test -f {base}.exit || echo 143 > {base}.exit", timeout=60, check=False)


def stop_all(a: dict) -> None:
    for r in runs(a["id"], all=False):
        try:
            stop(r)
        except NodeError:
            pass
        with closing(A._db()) as con:
            con.execute("UPDATE runs SET state='stopped', ended=? WHERE id=? AND state='running'", (time.time(), r["id"]))


def _event(agent: str, workspace, kind: str, summary: str, payload: dict) -> None:
    if not agent or not workspace:
        return
    try:
        from ..agents.store import Store
        Store(Path(workspace)).event(agent, kind, summary=summary, payload=payload)
    except Exception:                                               # noqa: BLE001 -- the run stands either way
        pass

