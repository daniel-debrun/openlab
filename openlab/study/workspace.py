"""The workspace: the folder that holds a researcher's studies and the agents working on them.

The root is the nearest ancestor (the start folder included) holding `openlab.toml`, which `openlab init` writes.
Without one, a study's workspace is the folder it sits in, and any other folder is its own workspace. The control
store for agents lives at `<root>/.openlab/control.db`. A study's own record stays in `<study>/.openlab/record.db`.
"""

from __future__ import annotations

import os
from pathlib import Path

MARKER = "openlab.toml"
_SKIP = {"node_modules", "__pycache__", "site-packages", "venv"}


def is_study(folder: Path) -> bool:
    try:
        return (Path(folder) / "study.toml").is_file()
    except OSError:                                 # an unreadable folder is not a study
        return False


def root(start: Path | str | None = None) -> Path:
    """The workspace that `start` (default: the working directory) belongs to."""
    here = Path(start or os.getcwd()).resolve()
    for d in (here, *here.parents):
        if (d / MARKER).is_file():
            return d
    return here.parent if is_study(here) else here


def studies(ws: Path | str, depth: int = 3) -> list[Path]:
    """Every study folder under the workspace, at most `depth` levels down. Hidden and vendor folders are
    skipped, and a study's release/ (openlab.study.release)."""
    base, out = Path(ws).resolve(), []

    def walk(d: Path, left: int) -> None:
        study = is_study(d)
        if study:
            out.append(d)
        if left == 0:
            return
        try:
            kids = sorted(p for p in d.iterdir() if p.is_dir())
        except OSError:
            return
        for k in kids:
            if not k.name.startswith(".") and k.name not in _SKIP and not (study and k.name == "release"):
                walk(k, left - 1)                       # a study's release/ holds a copy of its code, not a study

    walk(base, depth)
    return out
