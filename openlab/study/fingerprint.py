"""What identifies a study's state: the effective spec, every code file, the registered data and the environment.
The runner keys runs on it and a look records it, so both compute it the same way (one function, here)."""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path

from .record import canonical

SKIP = {".openlab", "data", "recipes", ".git", "paper", "__pycache__", "branches", "release", ".release.tmp"}   # branches/: openlab.study.branches; release/: openlab.study.release


def code(folder: Path, record=None) -> dict:
    """{relative path: sha256} of every .py file in the study outside data, recipes and the record; with a
    record, each file's source is also kept in the content store, so code never exists without its source."""
    out = {}
    for p in sorted(Path(folder).rglob("*.py")):
        rel = p.relative_to(folder)
        if SKIP & set(rel.parts):
            continue
        data = p.read_bytes()
        out[str(rel)] = record.put_blob(data) if record is not None else hashlib.sha256(data).hexdigest()
    return out


def snapshot(folder: Path, into: Path) -> Path:
    """Copy the study's code into `into`: every .py file `code` hashes, and the other files outside data, recipes,
    paper, the record and hidden folders (what model.py may read beside it). A run imports only from its snapshot,
    so an edit made while it runs (another agent's next variant) reaches neither its children nor its audit
    replays, and the code it records (`code(into)`, equal to `code(folder)` at the copy) is the code that ran."""
    folder, into = Path(folder), Path(into)
    for d, dirs, files in os.walk(folder):
        rel = Path(d).relative_to(folder)
        dirs[:] = sorted(x for x in dirs if x not in SKIP)
        hidden = any(x.startswith(".") for x in rel.parts)
        for f in files:
            src = Path(d) / f
            if src.is_file() and (f.endswith(".py") or not (hidden or f.startswith("."))):
                (into / rel).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, into / rel / f)
    return into


@contextmanager
def staged(study, code_dir: Path | None = None):
    """The study's code (from `code_dir` when it is not the study folder) snapshotted into a run-owned dir under
    .openlab/stage, deleted afterwards. A run's fingerprint, its children and its audit replays all read this copy,
    and a look records the copy its holdout run then runs."""
    root = study.record.dir / "stage"
    root.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="code-", dir=root))
    try:
        yield snapshot(Path(code_dir or study.folder), d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def data(record, names=None) -> dict:
    """{dataset name: sha256} of the latest dataset node per name (only `names`, when given)."""
    return {n.payload["name"]: n.payload["sha256"] for n in record.find("dataset")
            if names is None or n.payload["name"] in names}


CORE = ("numpy", "polars")     # the runner's numeric stack: it stages, scores and bootstraps every run


def imports(code_dir: Path) -> list:
    """The modules the study's code imports that are not its own (dotted names, from each .py file's import
    statements; relative imports and the study's own modules left out)."""
    import ast
    code_dir = Path(code_dir)
    local = {p.stem for p in code_dir.glob("*.py")} | {p.name for p in code_dir.iterdir() if p.is_dir()}
    out = set()
    for rel in code(code_dir):
        try:
            tree = ast.parse((code_dir / rel).read_bytes())
        except (SyntaxError, ValueError, OSError):
            continue
        for node in ast.walk(tree):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                     [node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else [])
            out.update(n for n in names if n.split(".")[0] not in local)
    return sorted(out)


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requires(dist) -> list:
    """The names a distribution requires outright: extras left out, environment markers evaluated here."""
    out = []
    for r in dist.requires or []:
        try:
            from packaging.requirements import Requirement
            req = Requirement(r)
            if req.marker is not None and not req.marker.evaluate({"extra": ""}):
                continue
            out.append(req.name)
        except Exception:                           # noqa: BLE001 -- no packaging, or a line it cannot parse
            if "extra" not in r.partition(";")[2]:
                out.append(re.split(r"[\s;<>=!~\[(]", r.strip(), maxsplit=1)[0])
    return out


def _owners(module: str, dists: dict) -> list:
    """The distributions that install the top-level module `module`: by top_level.txt, by the distribution's own
    name, else by the files it installed (sklearn is scikit-learn's); none for the standard library."""
    if module in sys.builtin_module_names or module in getattr(sys, "stdlib_module_names", ()):
        return []
    if _norm(module) in dists:
        return [_norm(module)]
    named = [n for n, d in dists.items() if module in (d.read_text("top_level.txt") or "").split()]
    if named:
        return named
    marks = (f"{module}/", f"{module}.py,", f"{module}.")
    for n, d in dists.items():
        lines = "\n" + (d.read_text("RECORD") or "")
        if any(f"\n{m}" in lines for m in marks):
            return [n]
    return []


def packages(code_dir: Path | None = None) -> dict:
    """{distribution: version} of what a run can execute: the distributions the study's code imports, numpy and
    polars, and everything those require (extras aside). A tool installed into a shared environment for something
    else (pytest, a linter) is not in it, so it cannot change a run's key."""
    dists = {}
    for d in metadata.distributions():
        name = d.metadata["Name"]
        if name:
            dists.setdefault(_norm(name), d)
    roots = {m.split(".")[0] for m in (imports(code_dir) if code_dir else [])} | set(CORE)
    todo = [_norm(n) for r in sorted(roots) for n in _owners(r, dists)]
    out: dict = {}
    while todo:
        n = todo.pop()
        if n in out or n not in dists:
            continue
        out[n] = dists[n].version
        todo += [_norm(x) for x in _requires(dists[n])]
    return dict(sorted(out.items()))


def all_packages_sha() -> str:
    """The packages hash runs carried before `packages`: every installed distribution's name and version."""
    dists = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions() if d.metadata["Name"])
    return hashlib.sha256("\n".join(dists).encode()).hexdigest()


def package_changes(was: dict, now: dict, most: int = 6) -> str:
    """The distributions that differ, e.g. "scikit-learn 1.7.1→1.7.2, joblib added"."""
    out = [f"{n} {was[n]}→{now[n]}" if n in was and n in now else f"{n} {'added' if n in now else 'removed'}"
           for n in sorted({*was, *now}) if was.get(n) != now.get(n)]
    return ", ".join(out[:most]) + (f" and {len(out) - most} more" if len(out) > most else "")


def changes(was: dict, now: dict) -> list:
    """What differs between two fingerprints (a recorded run's fields and a new one), as (component, detail) pairs
    in key order: spec, code (per file), data (per dataset), env packages, python."""
    out = []
    if was.get("spec_sha") != now.get("spec_sha"):
        out.append(("spec", f"{was.get('spec_sha', '')[:8]}→{now.get('spec_sha', '')[:8]}"))
    for kind in ("code", "data"):
        a, b = was.get(kind, {}), now.get(kind, {})
        diff = [f"{f} {a[f][:8]}→{b[f][:8]}" if f in a and f in b else f"{f} {'added' if f in b else 'removed'}"
                for f in sorted({*a, *b}) if a.get(f) != b.get(f)]
        if diff:
            out.append((kind, ", ".join(diff)))
    ea, eb = was.get("env", {}), now.get("env", {})
    if ea.get("packages_sha") != eb.get("packages_sha"):
        detail = (package_changes(ea["packages"], eb["packages"]) if "packages" in ea and "packages" in eb else
                  f"packages_sha {ea.get('packages_sha', '')[:8]}→{eb.get('packages_sha', '')[:8]}" + (
                      "; openlab now hashes only the packages the code can import, not every installed one (a "
                      "one-time change)" if "packages" in eb else ""))
        out.append(("env packages", detail or "the set of packages hashed"))
    if ea.get("python") != eb.get("python"):
        out.append(("python", f"{ea.get('python')}→{eb.get('python')}"))
    return out


def env(code_dir: Path | None = None) -> dict:
    """The interpreter, platform, the packages a run can execute (`packages`, with `packages_sha` over them; the
    run key holds the sha) and a hash of openlab's own source."""
    pk = packages(code_dir)
    return {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
            "packages_sha": hashlib.sha256(canonical(pk).encode()).hexdigest(), "packages": pk,
            "openlab_sha": _openlab_sha()}


def _openlab_sha() -> str:
    """A hash of openlab's own source, so a reproduction can name a framework change (e.g. a baseline's fix). It is
    recorded, not part of the run key: updating openlab must not cost a study a trial."""
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        h.update(str(p.relative_to(root)).encode() + b"\0" + p.read_bytes())
    return h.hexdigest()


def fingerprint(study, code_dir: Path | None = None) -> dict:
    """{spec_sha, code, data, env} for the study as it stands (the spec must be sealed); `code_dir` is where the
    code is when it is not the study folder (one branch of the forking-paths tree run beside the others)."""
    from .spec import sealed
    spec = sealed(study.record)
    return {"spec_sha": spec.sha() if spec else "", "code": code(code_dir or study.folder, study.record),
            "data": data(study.record, set(spec.data) if spec else None), "env": env(code_dir or study.folder)}


def key(fp: dict, segment: str, seed: int = 0) -> str:
    """A run's identity: same spec, code, data, interpreter and packages, segment and seed = same run."""
    essential = {k: fp[k] for k in ("spec_sha", "code", "data")}
    essential["env"] = {k: fp["env"][k] for k in ("python", "packages_sha")}
    return hashlib.sha256(canonical({**essential, "segment": segment, "seed": seed}).encode()).hexdigest()
