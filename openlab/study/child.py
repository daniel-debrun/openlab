"""The process a study's code runs in: `python -m openlab.study.child <stage_dir>/job.json`.

With `--serve <stage_dir> <sandboxed> [module ...]` it is a server instead (`sandbox.Server`): one import, then a
fresh fork per job.

The job names an entry, "openlab.study.<module>:<function>", and its args. The child puts the study folder on
sys.path (so model.py imports) and calls entry(stage_dir, **args); the entry writes its output into the stage dir
and returns its path. The child then writes result.json there: ok, output, error.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import socket
import sys
import traceback
from pathlib import Path

ALLOWED = "openlab.study."


def _entry(name: str):
    module, _, fn = name.partition(":")
    if not module.startswith(ALLOWED) or not fn:
        raise ValueError(f"entry {name!r} is not a {ALLOWED}<module>:<function>")
    return getattr(importlib.import_module(module), fn)


def main(job_path: str) -> int:
    stage = Path(job_path).parent
    try:
        job = json.loads(Path(job_path).read_text(encoding="utf-8"))
        sys.path.insert(0, job["study"])
        out = _entry(job["entry"])(stage, **job.get("args", {}))
        res = {"ok": True, "output": str(out) if out else None, "error": ""}
    except Exception:
        res = {"ok": False, "output": None, "error": traceback.format_exc()}
    (stage / "result.json").write_text(json.dumps(res), encoding="utf-8")
    return 0 if res["ok"] else 1


def _mounts() -> set:
    """The sandbox's mount points (from /proc/self/mounts, octal escapes decoded)."""
    try:
        lines = Path("/proc/self/mounts").read_text().splitlines()
    except OSError:
        return set()
    return {re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), ln.split()[1]) for ln in lines if ln.split()}


def _wipe(d: Path, mounts: set) -> None:
    """Empty `d` (the sandbox's private /tmp) of what earlier jobs wrote, leaving every mount point under it (the
    stage dir and the study bound under /tmp) and the dirs leading to them in place."""
    for p in d.iterdir():
        s = str(p)
        if s in mounts:
            continue
        if any(m.startswith(s + "/") for m in mounts):
            _wipe(p, mounts)
            continue
        try:
            shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink()
        except OSError:
            pass


def serve(stage: str, sandboxed: str, *preload: str) -> int:
    """The server of `sandbox.Server`: import the runner and `preload`, then per line on stdin (a sub dir of
    `stage`) clear /tmp (private to the sandbox; never outside one) and fork; the fork runs that sub dir's job.json
    as `main` does, with stdin closed and its output in the sub dir. Each line is answered "<sub> <exit code>"."""
    out = os.fdopen(os.dup(1), "w", buffering=1)
    importlib.import_module("openlab.study.runner")
    for m in preload:
        try:
            importlib.import_module(m)
        except Exception:                           # noqa: BLE001 -- the job's own import says what is missing
            pass
    root, mounts = Path(stage), _mounts() if sandboxed == "1" else set()
    for line in sys.stdin:
        sub = line.strip()
        if sandboxed == "1":
            _wipe(Path("/tmp"), mounts)
        pid = os.fork()
        if pid == 0:
            code = 1
            try:
                d = root / sub
                os.chdir(d)
                os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
                for fd, name in ((1, "stdout.txt"), (2, "stderr.txt")):
                    os.dup2(os.open(d / name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644), fd)
                code = main(str(d / "job.json"))
            finally:
                try:
                    sys.stdout.flush()
                    sys.stderr.flush()
                finally:
                    os._exit(code)
        _, status = os.waitpid(pid, 0)
        out.write(f"{sub} {os.waitstatus_to_exitcode(status)}\n")
    return 0


def _outcome(fn) -> str:
    try:
        fn()
        return "ok"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def probe(stage: Path, read: list = (), module: str = "") -> Path:
    """What the sandbox lets through: a connection to 1.1.1.1:53, reading each path in `read`, writing into the
    stage dir, importing `module`. Each outcome is "ok" or the error."""
    out = {"connect": _outcome(lambda: socket.create_connection(("1.1.1.1", 53), timeout=3).close()),
           "read": {p: _outcome(lambda p=p: Path(p).read_bytes()) for p in read},
           "write": _outcome(lambda: (stage / "probe_write.txt").write_text("x")),
           "import": _outcome(lambda: importlib.import_module(module)) if module else ""}
    path = stage / "probe.json"
    path.write_text(json.dumps(out), encoding="utf-8")
    return path


if __name__ == "__main__":
    sys.exit(serve(*sys.argv[2:]) if sys.argv[1] == "--serve" else main(sys.argv[1]))
