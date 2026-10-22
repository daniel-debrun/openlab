"""A workspace's harness versions: each a validated harness folder stored by its fingerprint under
.openlab/harness/versions/<sha>/ (the hooks refuse an agent's reads and writes there), and the one agents are briefed
by, named in .openlab/harness/active (none: the packaged harness).

Storing a version (`install`) checks it and records it on the control store's hash chain with its parent and the
lesson it rests on. Making one active (`use`) and going back (`rollback`) are person acts, signed with the person key
and recorded the same way. A version may become active only if it is the packaged one, was active before, or rests on
an approved harness lesson (openlab.harness.lessons): a harness enters use only with a lesson that supports it.

A spawned agent records the version that briefed it (launch.json `harness`); its children are briefed by the same
version, so a probe's whole tree runs the candidate it tests.
"""

from __future__ import annotations

import difflib
import shutil
import tempfile
from pathlib import Path

from . import DEFAULT, Harness, HarnessError, laws, load, read, validate

AGENT = "harness"           # the control store's events of the harness are filed under this name


def home(root) -> Path:
    return Path(root) / ".openlab" / "harness"


def path(root, sha: str) -> Path:
    return home(root) / "versions" / sha


def _store(root):
    from ..agents.store import Store
    return Store(root)


def default_sha() -> str:
    return load().sha


def resolve(root, ref: str) -> str:
    """The full sha a reference names: "default" (or "") is the packaged harness, else a stored version's sha or a
    prefix of one (at least 6 characters)."""
    ref = (ref or "").strip()
    if ref in ("", "default") or ref == default_sha():
        return ""
    d = home(root) / "versions"
    hits = sorted(p.name for p in d.iterdir() if p.name.startswith(ref)) if d.is_dir() and len(ref) >= 6 else []
    if len(hits) != 1:
        raise HarnessError([f"no single stored harness version {ref!r}" + (f" (it names {len(hits)})" if hits else "")
                            + ": `openlab harness versions` lists them"])
    return hits[0]


def get(root, ref: str = "") -> Harness:
    """The harness `ref` names (see resolve), checked: its files still have the fingerprint it is stored under, and
    it keeps the workspace's laws."""
    sha = resolve(root, ref)
    if not sha:
        h = load()
    else:
        h = read(path(root, sha))
        if h.sha != sha:
            raise HarnessError([f"harness version {sha[:12]} changed on disk (its files now hash to {h.sha[:12]}); "
                                "restore it, or make another version active"])
    problems = validate(h, laws(root))
    if problems:
        raise HarnessError([f"harness {sha[:12] or 'default'}: {p}" for p in problems])
    return h


def active_sha(root) -> str:
    """The active version's sha ("" for the packaged harness)."""
    try:
        return (home(root) / "active").read_text(encoding="utf-8").strip()
    except (OSError, TypeError):
        return ""


def active(root=None) -> Harness:
    """The harness the workspace's agents are briefed by."""
    return get(root, active_sha(root)) if root is not None else load()


def workflow(root, name: str) -> str:
    """A workflow brief (workflows/<name>.md) of the workspace's active harness."""
    return active(root).text(f"workflows/{name}.md")


def install(root, src, parent: str = "", note: str = "", lesson: str = "") -> dict:
    """Store the harness folder `src` as a version (refused unless it validates); its sha, and whether it is new."""
    h = read(src)
    problems = validate(h, laws(root))
    if problems:
        raise HarnessError(problems)
    if h.sha == default_sha():
        return {"sha": h.sha, "path": str(DEFAULT), "new": False, "default": True}
    dest = path(root, h.sha)
    new = not dest.is_dir()
    if new:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(dir=dest.parent, prefix=".tmp-"))
        for rel in h.files:
            (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(h.path) / rel, tmp / rel)
        tmp.rename(dest)
        _store(root).event(AGENT, "harness_version", summary=f"version {h.sha[:12]}" + (f": {note}" if note else ""),
                           payload={"sha": h.sha, "parent": parent or active_sha(root) or default_sha(),
                                    "note": note, "lesson": lesson, "files": len(h.files)})
    return {"sha": h.sha, "path": str(dest), "new": new}


def history(root) -> list:
    """The person's changes of the active version, oldest first."""
    return [e for e in _store(root).events(AGENT, n=10 ** 6, kinds=("harness_use",))]


def stored(root) -> list:
    """Every stored version with its parent, note, lesson and when it was stored; the active one marked."""
    on = active_sha(root)
    rows = {e["payload"]["sha"]: {"sha": e["payload"]["sha"], "parent": e["payload"].get("parent", ""),
                                  "note": e["payload"].get("note", ""), "lesson": e["payload"].get("lesson", ""),
                                  "at": e["at"]}
            for e in _store(root).events(AGENT, n=10 ** 6, kinds=("harness_version",))}
    out = [{"sha": default_sha(), "parent": "", "note": "the packaged harness", "lesson": "", "at": "",
            "active": not on}]
    return out + [{**r, "active": r["sha"] == on} for r in rows.values() if path(root, r["sha"]).is_dir()]


def _supported(root, sha: str) -> str:
    """Why `sha` may become active ("" if it may not): it was active before, or its lesson is approved."""
    if any(e["payload"].get("sha") == sha for e in history(root)):
        return "it was active before"
    row = next((r for r in stored(root) if r["sha"] == sha), None)
    if row and row["lesson"]:
        from .lessons import status
        if status(root, row["lesson"]) == "approved":
            return f"it rests on the approved lesson {row['lesson']}"
    return ""


def use(root, ref: str, reason: str) -> dict:
    """Make a version active (a person act: signed by the person key active in this process)."""
    from ..study import personkey
    signer = personkey.current()
    if signer is None:
        raise PermissionError("making a harness version active is a person's act, signed with the person key: "
                              "`openlab harness use SHA --reason ...` in the person's own terminal")
    if not reason.strip():
        raise HarnessError(["a change of the active harness needs a reason"])
    sha = resolve(root, ref)
    get(root, sha)
    prev = active_sha(root)
    if sha == prev:
        return {"sha": sha or default_sha(), "changed": False}
    if sha and not _supported(root, sha):
        raise HarnessError([f"version {sha[:12]} rests on no approved harness lesson and was never active: a harness "
                            "enters use only with a lesson that supports it (`openlab harness lessons`)"])
    msg = f"openlab-harness|use|{sha or 'default'}|{prev or 'default'}|{reason}".encode("utf-8")
    sig = {"key": signer.fingerprint, "value": signer.sign(msg)}
    f = home(root) / "active"
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(sha, encoding="utf-8")
    tmp.replace(f)
    _store(root).event(AGENT, "harness_use", summary=f"{signer.label}: harness {sha[:12] or 'default'} active "
                       f"(was {prev[:12] or 'default'}): {reason}",
                       payload={"sha": sha, "prev": prev, "reason": reason, "by": signer.label, "sig": sig},
                       exact=True)
    return {"sha": sha or default_sha(), "prev": prev or default_sha(), "changed": True, "by": signer.label}


def rollback(root, reason: str) -> dict:
    """Make the version active before the current one active again (a person act)."""
    seen = [e["payload"].get("prev", "") for e in history(root) if e["payload"].get("sha") == active_sha(root)]
    if not seen:
        raise HarnessError(["there is no earlier active harness to go back to"])
    return use(root, seen[-1] or "default", reason)


def diff(root, a: str, b: str) -> str:
    """A unified diff of two harnesses' files (a to b)."""
    ha, hb = get(root, a), get(root, b)
    out = []
    for rel in sorted(set(ha.files) | set(hb.files)):
        x, y = ha.files.get(rel), hb.files.get(rel)
        if x != y:
            out += difflib.unified_diff((x or "").splitlines(True) if x is not None else [],
                                        (y or "").splitlines(True) if y is not None else [],
                                        f"a/{rel}" if x is not None else "/dev/null",
                                        f"b/{rel}" if y is not None else "/dev/null")
    return "".join(line if line.endswith("\n") else line + "\n" for line in out)


def export(root, ref: str, dest) -> Path:
    """A copy of a harness's files in `dest` (a new folder), to edit into a candidate."""
    h = get(root, ref)
    dest = Path(dest)
    for rel in h.files:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(h.path) / rel, dest / rel)
    return dest


def changed(root, a: str, b: str) -> list:
    """The files that differ between two harnesses."""
    ha, hb = get(root, a), get(root, b)
    return sorted(rel for rel in set(ha.files) | set(hb.files) if ha.files.get(rel) != hb.files.get(rel))
