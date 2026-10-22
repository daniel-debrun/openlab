"""A harness's skills: skills/<name>/SKILL.md, a procedure an agent loads when its always-on text says it applies
(the validator holds every skill to a trigger line there), and, beside it, an optional skills/<name>/script.py.

A script is the one kind of code a harness carries, so it runs only one way: `run`, in the bubblewrap sandbox with
no network, the study (if one is named) read-only with its record, raw data and history masked, and a fresh stage
directory as the only place it can write. Where the sandbox does not start, the script does not run: unlike a study
run there is no unsandboxed fallback, because nothing about a script was checked beyond its compiling.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from . import Harness, HarnessError

OUT_MAX = 20_000            # characters of a script's stdout returned
ERR_MAX = 4_000


def skill(h: Harness, name: str) -> str:
    rel = f"skills/{name}/SKILL.md"
    if rel not in h.files:
        raise HarnessError([f"the harness has no skill {name!r}" + (f"; it has {', '.join(h.skills)}" if h.skills
                                                                      else "; it has none")])
    return h.files[rel]


def run(root, h: Harness, name: str, args: list | None = None, study: str = "", timeout: float = 120.0) -> dict:
    """Run skill `name`'s script sandboxed: {exit, stdout, stderr, files (what it wrote to its stage), stage}."""
    from ..study.sandbox import _bwrap, _env, _python
    rel = f"skills/{name}/script.py"
    if rel not in h.files:
        raise HarnessError([f"the skill {name!r} has no script"])
    study_dir = (Path(root) / study).resolve() if study else None
    if study_dir is not None and (not study_dir.is_dir() or Path(root).resolve() not in study_dir.parents):
        raise HarnessError([f"{study!r} is not a folder of this workspace"])
    stage = Path(tempfile.mkdtemp(prefix="openlab-script-"))
    empty = Path(tempfile.mkdtemp(prefix="openlab-script-ro-")) if study_dir is None else None
    try:
        shutil.copyfile(Path(h.path) / rel, stage / "script.py")
        prefix, why = _bwrap(study_dir or empty, stage, network=False)
        if not prefix:
            raise HarnessError([f"a harness script runs only in the sandbox, which does not start here: {why}"])
        try:
            p = subprocess.run([*prefix, _python(), "script.py", *[str(a) for a in args or []]], capture_output=True,
                               text=True, env=_env(), cwd=stage, timeout=timeout)
            code, out, err = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired as e:
            code, out, err = None, (e.stdout or ""), f"timed out after {timeout:g} s"
            out = out.decode(errors="replace") if isinstance(out, bytes) else out
        files = sorted(str(f.relative_to(stage)) for f in stage.rglob("*") if f.is_file() and f.name != "script.py")
        return {"exit": code, "stdout": out[-OUT_MAX:], "stderr": err[-ERR_MAX:], "files": files, "stage": str(stage),
                "sandbox": "bwrap"}
    finally:
        if empty is not None:
            shutil.rmtree(empty, ignore_errors=True)
