"""The vault: a study's holdout labels, kept outside the study folder from its first run on, so a later carve is
checked against them (the same rows, the same labels) and the agent's working copy never needs them.

Where: ``$OPENLAB_VAULT/<workspace key>/`` when set, else ``~/.local/share/openlab/vault/<workspace key>/`` (the
key is a short hash of the study folder). Locally the vault is tamper-evident, not tamper-proof: it is a file
the same OS user can read. `closed_dirs` names the directories the workspace's file tools refuse.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from . import clock


def higher_is_better(metric: str) -> bool:
    from .metrics import METRICS
    return METRICS[metric].higher_is_better


# --- file names: one injective name per key (a/b and a_b never share a file) ----------------------------------
#: The marker between a flattened name and its fingerprint. Never produced by flattening (it is not in the
#: safe set), which is what makes "has a fingerprint" a decidable question about a file name.
FINGERPRINT = "~"

#: Hex digits of the key's sha1. Eight is ~4 billion: a workspace would need ~100k documents whose flat
#: names agree before a collision is as likely as a coin landing on its edge, and the cost of being wrong
#: here is two logs merged, not data lost.
FINGERPRINT_CHARS = 8

#: Characters of the flat part that are kept. A file name is capped at 255 *bytes* on every filesystem
#: anyone runs this on, and the callers add up to ``.tests.jsonl`` after it -- a name over the limit is an
#: OSError that the best-effort writers swallow, i.e. a log that silently never appears.
NAME_CHARS = 120


def _flat(key: str) -> str:
    """The key with everything a file name should not carry replaced by ``_``. Lossy on purpose."""
    return "".join(ch if (ch.isascii() and ch.isalnum()) or ch in "-_." else "_" for ch in key)


def derived_name(key: str, fallback: str = "scratch") -> str:
    """The one file name (without a suffix) that this key's derived state lives under.

    ``key`` is a workspace path, or a session name -- whatever the caller keys its store on. The empty key
    is the ``fallback``, because a store still needs a name for the document that is not a file yet."""
    if not key:
        return fallback
    flat = _flat(key)
    # A name of dots only (``.``, ``..``) is not a name at all -- it addresses a directory -- and one over
    # the byte budget cannot be written. Both are keys that the flat form fails to stand for, so both take
    # the fingerprint that does.
    unchanged = flat == key and len(flat) <= NAME_CHARS and flat.strip(".") != ""
    if unchanged:
        return flat
    fp = hashlib.sha1(key.encode("utf-8")).hexdigest()[:FINGERPRINT_CHARS]
    return f"{flat[:NAME_CHARS]}{FINGERPRINT}{fp}"


# --- the store ------------------------------------------------------------------------------------------
def _root_key(root: Path) -> str:
    """Which workspace a holdout belongs to, as a short stable name."""
    return hashlib.sha1(str(root).encode()).hexdigest()[:12]


def vault_dir(root: Path) -> Path:
    """Where this workspace's holdouts live -- always a directory *per workspace*.

    ``$OPENLAB_VAULT`` names the place the vaults live, not one vault: two workspaces pointed at it used to
    share a flat namespace, so ``scratch__churn.json`` from one would answer for the other, and the second
    carve would be refused as "already exists" against labels that belong to somebody else's frame. The
    workspace key is in the path either way now."""
    base = os.environ.get("OPENLAB_VAULT")
    if base:
        return (Path(base).expanduser().resolve() / _root_key(root))
    home = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return (home / "openlab" / "vault" / _root_key(root)).resolve()


def _legacy_dir(root: Path) -> Optional[Path]:
    """The flat ``$OPENLAB_VAULT`` layout, before the workspace key went into the path. Read, never written:
    a holdout carved under the old layout keeps its labels, its attempts and its budget."""
    base = os.environ.get("OPENLAB_VAULT")
    return Path(base).expanduser().resolve() if base else None


def closed_dirs(root: Path) -> list:
    """Every directory the workspace must refuse to read or list: this workspace's vault, and -- when
    ``$OPENLAB_VAULT`` puts the vaults somewhere inside the workspace -- the base that holds all of them.
    Refusing only the leaf would let a file tool walk the parent and see which workspaces have holdouts."""
    out = [vault_dir(root)]
    base = _legacy_dir(root)
    if base is not None:
        out.append(base)
    return out


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s) or "scratch"


def _file(root: Path, doc: str, name: str) -> Path:
    """Where this holdout is written. Always the current layout: a record read from an old one is migrated
    on its next write, so the attempt that moves it is the same attempt that counts it.

    The name goes through `derived_name`, which is injective: ``a/b.openlab`` and ``a_b.openlab`` used to
    flatten to one file, so two documents' holdouts could answer for each other."""
    return vault_dir(root) / f"{derived_name(doc, 'scratch')}__{derived_name(name, 'scratch')}.json"


def _old_names(root: Path, doc: str, name: str) -> list:
    """Every path an older layout may have written this holdout to: the flat name in this workspace's
    directory, and the flat name in the pre-workspace-key ``$OPENLAB_VAULT`` base."""
    flat = f"{_safe(doc)}__{_safe(name)}.json"
    out = [vault_dir(root) / flat]
    old = _legacy_dir(root)
    if old is not None:
        out.append(old / flat)
    return [q for q in out if q.resolve() != _file(root, doc, name).resolve()]


def _owns(q: Path, doc: str, name: str) -> bool:
    """A flat-named record is only this holdout's if the record says so: the flat name is ambiguous, the
    ``doc``/``name`` stored inside it are not."""
    try:
        rec = json.loads(q.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return rec.get("doc") == doc and rec.get("name") == name


def _read_file(root: Path, doc: str, name: str) -> Optional[Path]:
    """Where this holdout actually is: the current path, else an old flat one whose record names it."""
    p = _file(root, doc, name)
    if p.exists():
        return p
    for q in _old_names(root, doc, name):
        if q.exists() and _owns(q, doc, name):
            return q
    return None


def _write(p: Path, rec: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(p.parent, 0o700)
    except OSError:
        pass
    fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")    # mkstemp makes it 0600; one per writer
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec))
    os.replace(tmp, p)


def public(rec: dict) -> dict:
    """A holdout as anyone may see it: everything but the labels."""
    left = rec["budget"] - len(rec["attempts"])
    return {"name": rec["name"], "doc": rec["doc"], "frame": rec["frame"], "target": rec["target"],
            "metric": rec["metric"], "higher_is_better": higher_is_better(rec["metric"]), "n": len(rec["labels"]),
            "train": rec["train"], "test": rec["test"], "budget": rec["budget"], "remaining": left,
            "fraction": rec.get("fraction"), **{k: rec[k] for k in ("seed", "split", "col", "group") if k in rec},
            "attempts":
                [{**{k: a[k] for k in ("attempt", "score", "interval", "time")},
                  "labels_reachable": a.get("labels_reachable")} for a in rec["attempts"]],
            "labels_sha": rec["labels_sha"], "created": rec["created"], "closed": rec.get("closed")}


def create(root: Path, doc: str, name: str, carved: dict, frame: str, target: str, metric: str, budget: int,
           fraction: Optional[float] = None, seed: Optional[int] = None, terms: Optional[dict] = None) -> dict:
    from .metrics import METRICS as _ALL
    if metric not in _ALL:
        raise ValueError(f"metric is one of {', '.join(_ALL)}")
    if budget < 1:
        raise ValueError("budget is at least 1")
    if _read_file(root, doc, name) is not None:
        raise FileExistsError(f"a holdout named {name!r} already exists for this document; its labels and its "
                              "attempt count are kept. Carve another under a new name")
    p = _file(root, doc, name)
    labels = carved["labels"]
    rec = {"name": name, "doc": doc, "frame": frame, "target": target, "metric": metric, "budget": int(budget),
           "fraction": fraction, "seed": seed, **(terms or {}),  # kept so a later carve under this name is checked
           "notes": carved.get("notes") or [],                 # its warnings, shown again on every re-apply
           "rows": carved["rows"], "labels": labels, "train": carved["train"], "test": carved["test"],
           "labels_sha": labels_sha(labels), "attempts": [],
           "created": clock.now()}
    _write(p, rec)
    return public(rec)


def labels_sha(labels: list) -> str:
    """The fingerprint of a holdout's labels, in their held-out row order. Short on purpose: it identifies a
    label set to anyone who already has it and is useless to anyone who does not."""
    return hashlib.sha256(json.dumps(labels).encode()).hexdigest()[:16]


def load(root: Path, doc: str, name: str) -> dict:
    p = _read_file(root, doc, name)
    if p is None:
        raise KeyError(f"no holdout named {name!r} for this document")
    return json.loads(p.read_text(encoding="utf-8"))
