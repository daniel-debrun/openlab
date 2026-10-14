"""Run a study job in a child process (`openlab.study.child`). Under bwrap the child has no network, sees the
system, Python and openlab read-only, the study read-only with its record, raw data and git history masked (a
nested study's too), and writes only the stage dir. Where bwrap does not start, the child runs directly and the
run is `unsandboxed`.
"""

from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import openlab

from . import machine

SYSTEM = ("/usr", "/lib", "/lib64", "/bin", "/etc")
MASKED = (".openlab", "data", ".git")        # the record and blobs; input files, which may hold holdout rows; history
#: folders not walked for nested masks: environments and caches, large and holding no study
UNWALKED = frozenset(("node_modules", ".venv", "venv", "site-packages", "__pycache__", ".tox", ".nox", ".cache",
                      ".mypy_cache", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints"))
PKG = Path(openlab.__file__).resolve().parent
THREADS = ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


@dataclass(frozen=True)
class Result:
    ok: bool
    output: Path | None
    stdout: str
    stderr: str
    sandbox: str                            # "bwrap" | "unsandboxed"
    seconds: float
    timed_out: bool = False                 # stopped at its timeout: stderr then says so first


def _python() -> str:
    return os.path.abspath(sys.executable)  # not resolved: a venv's python finds its pyvenv.cfg by this path


def _python_dirs() -> list[str]:
    """The interpreter and its environment, at the paths Python uses and at their real paths (a venv's python
    links into its base install, sometimes through a link of its own)."""
    paths = (sys.prefix, sys.exec_prefix, sys.base_prefix, sys.base_exec_prefix, os.path.dirname(_python()))
    return sorted({q for p in paths for q in (os.path.abspath(p), os.path.realpath(p))})


DEV = Path("/dev")


def gpu_devices(dev: Path | None = None) -> list[str]:
    """The NVIDIA device files a GPU job needs inside the sandbox (the control and UVM devices and one per GPU).
    All are bound: which GPUs the job uses is CUDA_VISIBLE_DEVICES's to say, as numbering follows the driver's."""
    return sorted(str(p) for p in (dev or DEV).glob("nvidia*") if p.exists())


def bwrap_args(study_dir: Path, stage_dir: Path, network: bool = False, gpus: list | None = None) -> list[str]:
    """bwrap's options. Order matters: /tmp before anything bound under it, the masks after the study, and the
    stage (under .openlab/stage) after the masks. `gpus`: the run's leased GPUs (the NVIDIA devices are bound only
    then)."""
    study, stage = Path(study_dir).resolve(), str(Path(stage_dir).resolve())
    a = ["--die-with-parent", "--unshare-pid", "--new-session", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for d in gpu_devices() if gpus else []:
        a += ["--dev-bind", d, d]
    if not network:
        a.append("--unshare-net")
    for d in SYSTEM:
        a += ["--ro-bind-try", d, d]
    for d in [*_python_dirs(), str(PKG), str(study)]:
        a += ["--ro-bind", d, d]
    for m in masks(study):
        a += ["--tmpfs", str(m)]
    return a + ["--bind", stage, stage, "--chdir", stage]


def masks(study: Path) -> list[Path]:
    """The real dirs a run must not see: the study's own .openlab, data and .git, and below it every .openlab and .git
    and every data dir beside a study.toml (a nested study's record and input files, which may hold holdout rows).
    One walk that never enters a masked dir, an environment or a cache; a symlink's target is not bound, so only
    real dirs need masking."""
    study, out = Path(study), []
    for d, dirs, files in os.walk(study):
        a_study = d == str(study) or "study.toml" in files
        here = [x for x in dirs if x in (".openlab", ".git") or (x == "data" and a_study)]
        out += [Path(d) / x for x in here]
        dirs[:] = [x for x in dirs if x not in here and x not in UNWALKED]
    return [m for m in out if m.is_dir() and not m.is_symlink()]


def _env(threads: int = 1, gpus: list | None = None) -> dict:
    """The child's whole environment: nothing of the parent's, so no credentials. `threads` ([sandbox] threads,
    default 1) caps the libraries' thread pools, so the numbers do not depend on the machine. `gpus`: the GPUs the
    run leased; a run that leased none sees none (CUDA_VISIBLE_DEVICES=""), so it cannot take another run's."""
    return {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "PYTHONPATH": str(PKG.parent), "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",            # replays of a step (the truncation audit) must match
            "CUDA_VISIBLE_DEVICES": ",".join(str(int(g)) for g in gpus or []),
            **{k: str(int(threads)) for k in THREADS}}


def _bwrap(study_dir: Path, stage_dir: Path, network: bool, gpus: list | None = None) -> tuple[list[str], str]:
    """(the bwrap prefix, "") if Python starts under it here, else ([], why not). Checked before the job runs, so
    the job's own code cannot make the run fall back to unsandboxed."""
    exe = shutil.which("bwrap")
    if not exe:
        return [], "bwrap is not installed"
    pre = [exe, *bwrap_args(study_dir, stage_dir, network, gpus)]
    try:
        p = subprocess.run([*pre, _python(), "-c", ""], capture_output=True, text=True, env=_env(), timeout=60)
    except subprocess.TimeoutExpired:
        return [], "bwrap did not start within 60s"
    return (pre, "") if p.returncode == 0 else ([], f"bwrap failed to start: {p.stderr.strip()}")


def _text(x) -> str:
    return x.decode(errors="replace") if isinstance(x, bytes) else x or ""


def _stopped(timeout) -> str:
    return f"timed out after {timeout:g} s ([sandbox] timeout)"


def _said(why: str, stderr: str, error: str, note: str) -> str:
    """The child's stderr as the run reports it. A timeout is said first, before the child's last words (kept
    short, so a reply that shows the tail still shows it), and replaces "no result.json", which it caused."""
    if note:
        return "\n".join(x for x in (note, why, stderr.strip()[-2500:]) if x)
    return "\n".join(x for x in (why, stderr, error) if x)


def _outcome(stage: Path) -> tuple[bool, Path | None, str]:
    try:
        r = json.loads((stage / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, None, "the child wrote no result.json"
    out = (stage / r["output"]).resolve() if r.get("output") else None
    if out is not None and not out.is_relative_to(stage):
        return False, None, f"the output {out} is outside the stage dir"
    return bool(r.get("ok")), out, r.get("error") or ""


class Server:
    """One sandboxed child that imports once and forks a fresh process per job: the truncation audit's replays pay
    Python's and the libraries' import once per server instead of once per replay.

    The server starts under the same bwrap options as `run` (its stage dir `stage_dir` is the only thing it may
    write), imports openlab's runner and `preload` (the libraries the study's code imports, never the study's own
    modules: no study code runs before a job), then waits. Per job the parent empties the stage dir, stages the
    job's data in one sub dir and sends its name; the server clears its private /tmp and forks, and the fork runs
    the job in that sub dir with fd 0 closed and its output in stdout.txt / stderr.txt. A job sees its own data
    only, starts from the state of a fresh import, and leaves nothing for the next one."""

    def __init__(self, *, study_dir: Path, stage_dir: Path, network: bool = False, timeout: float | None = None,
                 preload: list | tuple = (), threads: int = 1, gpus: list | None = None):
        self.study, self.stage, self.timeout = Path(study_dir).resolve(), Path(stage_dir).resolve(), timeout
        self.stage.mkdir(parents=True, exist_ok=True)
        self.prefix, self.why = _bwrap(self.study, self.stage, network, gpus)
        self.sandbox = "bwrap" if self.prefix else "unsandboxed"
        cmd = [*self.prefix, _python(), "-m", "openlab.study.child", "--serve", str(self.stage),
               "1" if self.prefix else "0", *preload]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  env=_env(threads, gpus), cwd=self.stage, text=True, bufsize=1)
        self.n = 0

    def alive(self) -> bool:
        return self.p.poll() is None

    def clear(self) -> Path:
        """Empty the stage dir and return a fresh sub dir for the next job's files."""
        for p in self.stage.iterdir():
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() and not p.is_symlink() else p.unlink(missing_ok=True)
        self.n += 1
        sub = self.stage / f"job{self.n}"
        sub.mkdir()
        return sub

    def run(self, job: dict, sub: Path) -> Result:
        """Run `job` in the sub dir `clear` made (its data staged there)."""
        sub = Path(sub).resolve()
        (sub / "job.json").write_text(json.dumps({**job, "study": str(self.study)}), encoding="utf-8")
        t0, code, note = time.monotonic(), None, ""
        try:
            with machine.slot():                    # the machine's children, at most one per core
                self.p.stdin.write(sub.name + "\n")
                self.p.stdin.flush()
                ready, _, _ = select.select([self.p.stdout], [], [], self.timeout)
                line = self.p.stdout.readline() if ready else ""
            if not ready:
                note = _stopped(self.timeout)
                self.close()
            elif line.split()[:1] == [sub.name]:
                code = int(line.split()[1])
            else:
                note = "the sandboxed server stopped"
        except (OSError, ValueError, IndexError):
            note = "the sandboxed server stopped"
        seconds = time.monotonic() - t0
        read = lambda f: (sub / f).read_text(errors="replace") if (sub / f).is_file() else ""     # noqa: E731
        ok, out, error = _outcome(sub)
        stopped = note.startswith("timed out")
        return Result(ok=ok and code == 0, output=out, stdout=read("stdout.txt"),
                      stderr=_said(self.why, read("stderr.txt"), "\n".join(x for x in (error, note) if x),
                                   note if stopped else ""),
                      sandbox=self.sandbox, seconds=seconds, timed_out=stopped)

    def close(self) -> None:
        try:
            self.p.stdin.close()
        except OSError:
            pass
        try:
            self.p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.p.wait()


def run(job: dict, *, study_dir: Path, stage_dir: Path, network: bool = False,
        timeout: float | None = None, threads: int = 1, gpus: list | None = None) -> Result:
    """Run `job` ({"entry", "args"}) in a child; its output is a file in `stage_dir`."""
    stage = Path(stage_dir).resolve()
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "result.json").unlink(missing_ok=True)
    path = stage / "job.json"
    path.write_text(json.dumps({**job, "study": str(Path(study_dir).resolve())}), encoding="utf-8")
    prefix, why = _bwrap(study_dir, stage, network, gpus)
    t0, note = time.monotonic(), ""
    try:
        with machine.slot():                        # the machine's children, at most one per core
            p = subprocess.run([*prefix, _python(), "-m", "openlab.study.child", str(path)], capture_output=True,
                               text=True, env=_env(threads, gpus), cwd=stage, timeout=timeout)
        stdout, stderr, code = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        stdout, stderr, code, note = _text(e.stdout), _text(e.stderr), None, _stopped(timeout)
    seconds = time.monotonic() - t0
    ok, out, error = _outcome(stage)
    return Result(ok=ok and code == 0, output=out, stdout=stdout, stderr=_said(why, stderr, error, note),
                  sandbox="bwrap" if prefix else "unsandboxed", seconds=seconds, timed_out=bool(note))
