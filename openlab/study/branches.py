"""Per-branch working copies: parallel experimenters on one study each edit their own model.py.

`branches/<id>/` in the study folder holds a branch's copy of model.py and the study's own modules it imports
(made by `make` when the proposal is executed; an existing copy is kept, it is the branch's work). A run of the
branch's code is `Study.run(code="branches/<id>")`: the runner snapshots that folder instead of the study's, the run
records `branch`, and its key holds that code's hash as any run's does. The study's code (its fingerprint, the main
model.py) never includes `branches/`, so one branch's edit moves neither the main code nor another branch's.
A paper binding of a branch's run is stale when that branch's current code moved on (the main model.py's for a run
of the main code). `promote` copies a branch's code over the main code: an agent's act, recorded.
"""

from __future__ import annotations

import ast
import hashlib
import re
import shutil
from pathlib import Path

from .errors import StudyError

DIR = "branches"
ENTRY = "model.py"
_ID = re.compile(r"[a-z][a-z0-9_-]{0,23}")


def local_modules(folder: Path, entry: str = ENTRY) -> list:
    """`entry` and the study's own modules it imports, transitively (relative paths of .py files and packages)."""
    folder = Path(folder)
    out, todo = [], [entry]
    while todo:
        rel = todo.pop()
        if rel in out:
            continue
        p = folder / rel
        if not p.exists():
            continue
        out.append(rel)
        files = [p] if p.is_file() else sorted(p.rglob("*.py"))
        for f in files:
            try:
                tree = ast.parse(f.read_bytes())
            except (SyntaxError, ValueError, OSError):
                continue
            for node in ast.walk(tree):
                names = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                         [node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else [])
                for n in names:
                    top = n.split(".")[0]
                    if (folder / f"{top}.py").is_file():
                        todo.append(f"{top}.py")
                    elif (folder / top / "__init__.py").is_file():
                        todo.append(top)
    return sorted(out)


def path(study, branch: str) -> Path:
    if not (isinstance(branch, str) and _ID.fullmatch(branch)):
        raise StudyError(f"no branch {branch!r}", "a branch id is a lower-case letter, then letters, digits, - or _")
    return Path(study.folder) / DIR / branch


def make(study, branch: str) -> dict:
    """branches/<branch>/ with a copy of model.py and the local modules it imports; kept as is when it exists."""
    d = path(study, branch)
    if (d / ENTRY).is_file():
        return {"branch": branch, "dir": f"{DIR}/{branch}", "made": False, "files": sorted(_files(d))}
    files = local_modules(study.folder)
    if ENTRY not in files:
        raise StudyError(f"{study.folder.name}/{ENTRY} is missing", "a branch's working copy starts from it",
                         f"write {ENTRY} first")
    d.mkdir(parents=True, exist_ok=True)
    for rel in files:
        src = Path(study.folder) / rel
        if src.is_dir():
            shutil.copytree(src, d / rel, ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)
        else:
            shutil.copyfile(src, d / rel)
    return {"branch": branch, "dir": f"{DIR}/{branch}", "made": True, "files": files}


def _files(d: Path) -> dict:
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*.py"))
            if "__pycache__" not in p.parts}


def resolve(study, code: str) -> tuple:
    """(the code folder, the branch id) of a run's `code` argument: "" for the study's own code, else
    `branches/<id>` (or just `<id>`) holding model.py."""
    if not code or code in (".", "./"):
        return None, ""
    rel = Path(code)
    if rel.is_absolute():
        try:
            rel = rel.resolve().relative_to(Path(study.folder).resolve())
        except ValueError:
            raise StudyError(f"{code} is outside the study", f"a branch's code is {DIR}/<id> in {study.folder}") \
                from None
    parts = rel.parts
    branch = parts[1] if len(parts) == 2 and parts[0] == DIR else parts[0] if len(parts) == 1 else ""
    d = path(study, branch) if branch else None
    if d is None or not (d / ENTRY).is_file():
        have = sorted(p.name for p in (Path(study.folder) / DIR).glob("*") if (p / ENTRY).is_file()) \
            if (Path(study.folder) / DIR).is_dir() else []
        raise StudyError(f"no branch code at {code}", f"a branch's code is {DIR}/<id>/{ENTRY}; branches: "
                         f"{', '.join(have) or 'none'}",
                         "the branch copies are made when the proposal is executed (openlab propose execute)")
    return d, branch


def current(folder: Path) -> dict:
    """{branch: {relative path: sha256}} of every branch's code now (what its runs' bindings must still match)."""
    from .fingerprint import code
    root = Path(folder) / DIR
    if not root.is_dir():
        return {}
    return {p.name: code(p) for p in sorted(root.iterdir()) if (p / ENTRY).is_file()}


def promote(study, branch: str) -> dict:
    """Copy the branch's code over the study's own (each of its .py files to the same path in the study)."""
    d = path(study, branch)
    if not (d / ENTRY).is_file():
        raise StudyError(f"branch {branch} has no working copy", f"no {DIR}/{branch}/{ENTRY}")
    files = _files(d)
    before = {rel: hashlib.sha256((Path(study.folder) / rel).read_bytes()).hexdigest()
              for rel in files if (Path(study.folder) / rel).is_file()}
    for rel in files:
        dst = Path(study.folder) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(d / rel, dst)
    changed = sorted(r for r, h in files.items() if before.get(r) != h)
    return {"branch": branch, "files": files, "changed": changed}
