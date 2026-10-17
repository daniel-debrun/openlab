"""Instruments: named, versioned programs that measure or act outside the sandbox (a broker's paper account, a data
collector, a scraper), which research code and agents use only through openlab, so every use is on the record and
bound to an exact version.

An instrument wraps any command: `python -m pkg`, `node script.js`, a shell script, a CLI binary, `npx` or `uv run`.
Its version is the sha256 of a manifest that holds everything it runs with:

* the entry (an argv, never one runtime's field), the declared capabilities, the secrets it may read (by name), its
  fixed environment variables, its outputs, whether it keeps state, its timeout and its log redaction patterns;
* its code: the files of its source folder (tracked or listed; no venvs, caches, .git or state), each by sha256. They
  are copied into the workspace's content store and into a read-only snapshot, and every run executes the snapshot,
  never the live folder. An instrument with no source folder (a system CLI) has no snapshot: it cannot be copied,
  only verified;
* its environment fingerprint, pluggable per runtime and hashed part by part: python (the interpreter, its version,
  and the sha256 of every installed distribution's files, an editable install's tree included), node (the binary,
  `node --version`, node_modules), a binary (its resolved path, its sha256 and its `--version` output), any lockfile
  (`--env-file`) and any command whose stdout is hashed (`--env-cmd`);
* its config file's sha256;
* its usage: the entry's `--help` (or `--usage-arg`s, and with `--usage-sub` each argparse subcommand's), probed at
  register in a throwaway copy of the run's snapshot and environment (10 s each, redacted), kept in the content store
  by its sha256. A failed probe is recorded as unavailable, not refused; a registration from before usage capture
  has none and hashes as it always did.

Registering is a person's act: an `instrument` node signed with the person key, on the workspace's instrument record
(`<workspace>/.openlab/instruments/record.db`, append-only and hash-chained like a study's, so the team shares one).
An agent files `agent_request(kind="instrument", args={...})`. A version never changes: before each run every file,
the config and every environment part are checked again, and drift refuses the run, naming what changed; new code or a
new environment is a new registration.

A run (`openlab instrument run NAME@v -- ARGS`) executes outside bwrap (instruments need the network or a broker), in
a per-run output folder, with a minimal environment: PATH, HOME, the locale, the declared secrets and fixed variables,
and OPENLAB_INSTRUMENT_* (its id, the snapshot, the output folder, the state folder when it keeps state, the config).
It records an `instrument_run` node: the version, args, actor, start and end, exit code, the sha256 of its redacted
stdout and stderr (kept in the content store) and of every declared output (kept there too). With a study, the run is
also written on that study's record, and only an instrument the study pins (`[instruments] alias = "NAME@v"`, a
person's field) can run there. A study's dataset can come from an instrument's output (`[data.x] instrument = "alias"`,
`output = "file"`): materialise takes the latest successful recorded run of the pinned version that wrote it, checks
its sha256 and registers it like any dataset.

A run returns to its caller its exit code, the redacted stdout and stderr tails and each declared output's content
when it is text of at most INLINE_MAX (else its size, sha and why it is withheld); `output(RUN, FILE)` reads a
recorded run back from the content store, each sha checked first.

Capabilities: `read`, `write`, `network`, `paper-order` (agents may run it) and `live-order`, which makes every run a
person's act: a signed `instrument_authority` node, written before the run with its exact args, that one run links.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import secrets as _secrets
import shlex
import shutil
import stat
import subprocess
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import personkey
from .errors import StudyError
from .record import _SCHEMA, Node, Record, canonical

NAME = re.compile(r"[a-z][a-z0-9_-]{0,47}")
REF = re.compile(r"([a-z][a-z0-9_-]{0,47})(?:@([0-9a-f]{12}))?")
PIN = re.compile(r"[a-z][a-z0-9_-]{0,47}@[0-9a-f]{12}")
CAPABILITIES = ("read", "write", "network", "paper-order", "live-order")
PERSON_CAPABILITIES = ("live-order",)       # every run of one is a person's act, signed for that run
RUNTIMES = ("python", "node", "binary", "none")
#: never copied into a snapshot: environments, caches, VCS folders, state and the usual secret files
DEFAULT_EXCLUDE = (".git", ".hg", ".svn", ".openlab", ".venv", "venv", "env", "__pycache__", "*.pyc", "*.pyo",
                   ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules", "*.egg-info", ".tox", "build",
                   "dist", "state", ".DS_Store", ".ipynb_checkpoints", ".env", ".env.*", "*.key", "*.pem")
MAX_FILES, MAX_BYTES = 5000, 200 * 2 ** 20          # a snapshot's limits: an instrument is a program, not a dataset
OUTPUT_MAX_BYTES = 1024 * 2 ** 20                   # what one run's declared outputs may store
TIMEOUT = 3600.0
PROBE_TIMEOUT = 60.0
USAGE_ARGS = ("--help",)                            # the usage probe: the entry with these, at register
USAGE_TIMEOUT = 10.0                                # seconds one usage probe may take
USAGE_MAX_SUBCOMMANDS, USAGE_MAX_BYTES = 40, 64 * 1024
TAIL_LINES, TAIL_BYTES = 200, 32 * 1024             # what a run returns of each log
INLINE_MAX = 256 * 1024                             # an output returned inline (text, at most this size)
READ_MAX = 4 * 2 ** 20                              # an output read whole by `instrument output RUN FILE`
WITHHELD = "too large; materialise it in a study"
PREVIEW_USAGE_LINES = 60                            # of the usage, in what the person reads before signing
BASE_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "USER", "LOGNAME", "TMPDIR", "TERM")
PLACEHOLDERS = ("snapshot", "out", "state", "config")
ENV_PREFIX = "OPENLAB_INSTRUMENT"
PIN_FIX = ("a person pins it under [instruments] in study.toml and signs that amendment (an agent files an "
           "amend request with the exact field)")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --- the workspace's instrument record --------------------------------------------------------------------------------
def workspace_root(start=None) -> Path:
    from .workspace import root
    return root(start)


class Registry(Record):
    """The workspace's instrument record: `<workspace>/.openlab/instruments/record.db`, the same append-only,
    hash-chained, signed record a study keeps (its person nodes are pinned to the workspace's person key), with its
    content store beside it (`blobs/`), the read-only snapshots (`snapshots/`), each run's folder (`runs/`) and each
    instrument's state (`state/<name>/`)."""

    def __init__(self, root=None):
        self.root = workspace_root(root)
        self.folder = self.root                     # Record._pin checks the workspace's pin from here
        self.dir = self.root / ".openlab" / "instruments"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "record.db"
        with closing(self._con()) as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript(_SCHEMA)

    @staticmethod
    def exists(root=None) -> bool:
        return (workspace_root(root) / ".openlab" / "instruments" / "record.db").is_file()

    def snapshot_dir(self, ident: str) -> Path:
        return self.dir / "snapshots" / ident

    def config_path(self, ident: str, suffix: str) -> Path:
        return self.dir / "snapshots" / f"{ident}.config{suffix}"

    def state_dir(self, name: str) -> Path:
        return self.dir / "state" / name


def ident(payload: dict) -> str:
    return f"{payload['name']}@{payload['version']}"


# --- the environment fingerprint: one function per part kind ----------------------------------------------------------
_PY_PROBE = r'''
import hashlib, json, os, sys
from importlib import metadata
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
SKIP = {".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache", ".mypy_cache", ".ruff_cache",
        ".tox", "build", "dist", ".openlab"}
def tree(root):
    h, n = hashlib.sha256(), 0
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in SKIP and not x.endswith(".egg-info"))
        for f in sorted(files):
            if f.endswith((".pyc", ".pyo")):
                continue
            p = os.path.join(d, f)
            try:
                h.update(os.path.relpath(p, root).encode() + b"\0" + sha(p).encode() + b"\n")
                n += 1
            except OSError:
                pass
    return h.hexdigest(), n
dists = {}
for d in metadata.distributions():
    name = (d.metadata["Name"] or "").lower().replace("_", "-")
    if not name or name in dists:
        continue
    h, files, editable = hashlib.sha256(), 0, ""
    direct = d.read_text("direct_url.json")
    if direct:
        try:
            u = json.loads(direct)
            if (u.get("dir_info") or {}).get("editable"):
                url = u.get("url", "")
                editable = url[7:] if url.startswith("file://") else url
        except ValueError:
            pass
    for f in sorted(d.files or [], key=str):
        s = str(f)
        if s.endswith((".pyc", ".pyo")) or "__pycache__" in s:
            continue
        try:
            h.update(s.encode() + b"\0" + sha(d.locate_file(f)).encode() + b"\n")
            files += 1
        except OSError:
            h.update(s.encode() + b"\0missing\n")
    rec = {"version": d.version, "sha": h.hexdigest()}
    if editable and os.path.isdir(editable):
        rec["editable"] = editable
        rec["tree"] = tree(editable)[0]
    dists[name] = rec
print(json.dumps({"version": sys.version.split()[0], "implementation": sys.implementation.name,
                  "executable": os.path.realpath(sys.executable), "dists": dists}, sort_keys=True))
'''


def _which(name: str, path: Optional[str] = None) -> Optional[str]:
    if "/" in name:
        p = Path(name).expanduser()
        return str(p.absolute()) if p.is_file() else None
    found = shutil.which(name, path=path)
    return str(Path(found).absolute()) if found else None


def _probe(argv: list, cwd=None, timeout: float = PROBE_TIMEOUT) -> tuple:
    """(exit code, stdout+stderr) of a fingerprint probe; a probe that cannot start or hangs says so."""
    try:
        p = subprocess.run(argv, cwd=cwd, capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL)
        return p.returncode, (p.stdout + p.stderr).decode("utf-8", "replace")
    except FileNotFoundError:
        return 127, f"{argv[0]}: not found"
    except subprocess.TimeoutExpired:
        return -1, f"timed out after {timeout:g}s"
    except OSError as e:
        return 126, f"{argv[0]}: {e}"


def _fp_python(spec: dict) -> dict:
    exe = _which(spec["path"])
    if exe is None:
        return {"path": spec["path"], "missing": True}
    code, out = _probe([exe, "-I", "-c", _PY_PROBE])
    if code != 0:
        return {"path": exe, "error": out.strip()[-500:]}
    info = json.loads(out)
    return {"path": exe, "executable": info["executable"], "version": info["version"],
            "implementation": info["implementation"], "binary_sha256": sha_file(info["executable"]),
            "dists": info["dists"]}


def _tree(root: Path, skip=(".cache",)) -> dict:
    h, n = hashlib.sha256(), 0
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in skip)
        for f in sorted(files):
            p = Path(d) / f
            try:
                h.update(str(p.relative_to(root)).encode() + b"\0" + sha_file(p).encode() + b"\n")
                n += 1
            except OSError:
                continue
    return {"sha256": h.hexdigest(), "files": n}


def _fp_node(spec: dict) -> dict:
    exe = _which(spec.get("path") or "node")
    if exe is None:
        return {"path": spec.get("path") or "node", "missing": True}
    code, out = _probe([exe, "--version"])
    detail = {"path": exe, "real": os.path.realpath(exe), "sha256": sha_file(exe), "version": out.strip()[:200],
              "exit": code}
    modules = spec.get("node_modules")
    if modules:
        detail["node_modules"] = _tree(Path(modules)) if Path(modules).is_dir() else {"missing": True}
    return detail


def _fp_binary(spec: dict) -> dict:
    exe = _which(spec["path"])
    if exe is None:
        return {"path": spec["path"], "missing": True}
    detail = {"path": exe, "real": os.path.realpath(exe), "sha256": sha_file(exe)}
    if spec.get("version_args") is not None:
        code, out = _probe([exe, *spec["version_args"]], timeout=10)
        detail.update({"version": out.strip()[:500], "exit": code})
    return detail


def _fp_file(spec: dict) -> dict:
    p = Path(spec["path"])
    return {"path": str(p), "sha256": sha_file(p)} if p.is_file() else {"path": str(p), "missing": True}


def _fp_cmd(spec: dict) -> dict:
    try:
        p = subprocess.run(["sh", "-c", spec["cmd"]], cwd=spec.get("cwd") or None, capture_output=True,
                           timeout=PROBE_TIMEOUT, stdin=subprocess.DEVNULL)
        out, code = p.stdout, p.returncode
    except subprocess.TimeoutExpired:
        out, code = b"", -1
    return {"cmd": spec["cmd"], "sha256": sha_bytes(out), "exit": code,
            "head": out.decode("utf-8", "replace")[:300]}


#: kind -> the function from a part's spec (what was declared) to its detail (what was found); a new runtime is a row
FINGERPRINTS = {"python": _fp_python, "node": _fp_node, "binary": _fp_binary, "file": _fp_file, "cmd": _fp_cmd}


def fingerprint_part(kind: str, spec: dict) -> dict:
    detail = FINGERPRINTS[kind](spec)
    return {"kind": kind, "spec": spec, "detail": detail, "sha": sha_bytes(canonical(detail).encode())}


def environment_sha(parts: list) -> str:
    return sha_bytes(canonical([[p["kind"], p["spec"], p["sha"]] for p in parts]).encode())


def _label(part: dict) -> str:
    s = part["spec"]
    return f"{part['kind']} {s.get('path') or s.get('cmd') or ''}".strip()


def _part_changes(old: dict, new: dict) -> list:
    """What differs between a part's registered detail and the one found now, as short lines."""
    a, b = old["detail"], new["detail"]
    if b.get("missing"):
        return [f"{b.get('path') or b.get('cmd')} is gone"]
    if b.get("error"):
        return [f"the probe failed: {b['error'][-200:]}"]
    out = []
    if old["kind"] == "python":
        for k in ("executable", "version", "implementation", "binary_sha256"):
            if a.get(k) != b.get(k):
                out.append(f"{k}: {a.get(k)} -> {b.get(k)}")
        da, db = a.get("dists") or {}, b.get("dists") or {}
        for n in sorted(set(da) | set(db)):
            x, y = da.get(n), db.get(n)
            if x is None:
                out.append(f"{n} {y['version']} was installed")
            elif y is None:
                out.append(f"{n} {x['version']} was removed")
            elif x["version"] != y["version"]:
                out.append(f"{n} {x['version']} -> {y['version']}")
            elif x.get("sha") != y.get("sha") or x.get("tree") != y.get("tree"):
                out.append(f"{n} {x['version']}: its installed files changed")
        return out or ["the interpreter's environment changed"]
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            out.append(f"{k}: {str(a.get(k))[:80]} -> {str(b.get(k))[:80]}")
    return out


# --- what a registration is made of ----------------------------------------------------------------------------------
def _matches(rel: str, patterns) -> bool:
    parts = rel.split("/")
    for p in patterns:
        p = p.strip("/")
        if "/" in p:
            if fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(rel, p + "/*"):
                return True
        elif any(fnmatch.fnmatch(x, p) for x in parts):
            return True
    return False


def _git_files(src: Path) -> Optional[list]:
    try:
        top = subprocess.run(["git", "-C", str(src), "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                             timeout=30)
        if top.returncode != 0:
            return None
        p = subprocess.run(["git", "-C", str(src), "ls-files", "-z", "--cached", "--others", "--exclude-standard",
                            "--", "."], capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if p.returncode != 0:
        return None
    return [x for x in p.stdout.decode("utf-8", "surrogateescape").split("\0") if x]


def collect(src: Path, include=(), exclude=()) -> list:
    """The source folder's files that make up the instrument's code, relative paths, sorted: git's tracked and
    untracked-but-not-ignored files in a work tree, else every file; never DEFAULT_EXCLUDE or `exclude`, except what
    `include` names. A symlink to a folder is not followed."""
    src = Path(src)
    skip = DEFAULT_EXCLUDE + tuple(exclude)
    listed = _git_files(src)
    if listed is None:
        listed = []
        for d, dirs, files in os.walk(src):
            rel_d = os.path.relpath(d, src)
            rel_d = "" if rel_d == "." else rel_d + "/"
            dirs[:] = sorted(x for x in dirs if not (Path(d) / x).is_symlink() and not _matches(rel_d + x, skip))
            listed += [rel_d + f for f in files]
    out = []
    for rel in sorted(set(listed)):
        p = src / rel
        if not p.is_file() or _matches(rel, (".git", ".openlab")):
            continue
        if _matches(rel, skip) and not _matches(rel, include):
            continue
        out.append(rel)
    for pat in include:
        for p in sorted(src.glob(pat)):
            rel = str(p.relative_to(src))
            if p.is_file() and rel not in out and not _matches(rel, (".git", ".openlab")):
                out.append(rel)
    return sorted(out)


def _shebang(path: Path) -> Optional[list]:
    try:
        with open(path, "rb") as f:
            line = f.readline(256)
    except OSError:
        return None
    if not line.startswith(b"#!"):
        return None
    argv = line[2:].decode("utf-8", "replace").strip().split()
    if argv and Path(argv[0]).name == "env":
        argv = [a for a in argv[1:] if not a.startswith("-")]
    return argv or None


def detect_runtime(argv0: str) -> str:
    base = Path(argv0).name
    if re.fullmatch(r"(python|pypy)[0-9.]*", base):
        return "python"
    if base in ("node", "nodejs", "npx", "npm", "pnpm", "yarn", "bun"):
        return "node"
    return "binary"


@dataclass
class Prepared:
    """A registration, computed and not yet written: the manifest whose sha is the version, and the files."""
    name: str
    manifest: dict
    sha: str
    files: dict = field(default_factory=dict)     # rel -> absolute source path
    source: str = ""
    config_file: str = ""
    usage_text: str = ""                          # the probed (or supplied) usage, redacted; its sha is in the manifest

    @property
    def version(self) -> str:
        return self.sha[:12]

    @property
    def ident(self) -> str:
        return f"{self.name}@{self.version}"


MANIFEST_KEYS = ("name", "entry", "runtime", "capabilities", "secrets", "env", "outputs", "state", "timeout",
                 "redact", "files", "links", "config", "config_suffix", "environment")


def manifest_sha(payload: dict) -> str:
    """The version's full sha256: of the manifest's fields, canonical (the reason and source path are not part of
    what runs). The usage (how the entry says it is called, by its sha) is part of it when it was recorded; a
    registration from before usage capture has none, and hashes as it always did."""
    fields = {k: payload[k] for k in MANIFEST_KEYS}
    if payload.get("usage"):
        fields["usage"] = payload["usage"]
    return sha_bytes(canonical(fields).encode())


def _list(v) -> list:
    if v is None:
        return []
    if isinstance(v, str):
        return [x.strip() for x in v.split(",") if x.strip()]
    out = []
    for x in v:
        out += _list(x) if isinstance(x, str) and "," in x else ([str(x).strip()] if str(x).strip() else [])
    return out


def _fixed_env(env) -> dict:
    items = env.items() if isinstance(env, dict) else [tuple(x.split("=", 1)) for x in (env or [])]
    fixed = {}
    for kv in items:
        k, v = kv[0], kv[1] if len(kv) > 1 else ""
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k) or k.startswith(ENV_PREFIX):
            raise StudyError(f"`{k}` is not an environment variable an instrument may fix",
                             f"a name of letters, digits and _, not {ENV_PREFIX}_* (openlab sets those)")
        fixed[k] = str(v)
    return fixed


def prepare(root, name: str, entry, *, source: str = "", runtime: str = "", python: str = "", node: str = "",
            binaries=(), env_files=(), env_cmds=(), env=None, capabilities=("read",), secrets=(), config: str = "",
            outputs=(), state: bool = False, timeout: Optional[float] = None, include=(), exclude=(), redact=(),
            version_probe: bool = True, usage: bool = True, usage_args=(), usage_sub: bool = False,
            usage_file: str = "") -> Prepared:
    """The registration of `name` as it would be written: the files collected, every environment part
    fingerprinted, the usage probed (`usage_args`, default `--help`, in a throwaway copy of the run's snapshot and
    environment; `usage_sub` also each argparse subcommand's; `usage_file` supplies it instead; `usage=False` skips
    it), the manifest hashed. Nothing is written into the workspace."""
    if not NAME.fullmatch(name or ""):
        raise StudyError(f"`{name}` is not an instrument name", "a name is lowercase letters, digits, - and _, "
                         "starting with a letter (at most 48)", "e.g. ibkr_paper")
    argv = shlex.split(entry) if isinstance(entry, str) else [str(x) for x in (entry or [])]
    if not argv:
        raise StudyError("an instrument needs an entry", "the command it runs, as an argv",
                         '--entry "python -m mypkg" (or "node run.js", "./collect.sh", "mycli")')
    caps = sorted(set(_list(capabilities))) or ["read"]
    bad = [c for c in caps if c not in CAPABILITIES]
    if bad:
        raise StudyError(f"unknown capability {', '.join(bad)}", f"capabilities: {', '.join(CAPABILITIES)}")
    if runtime and runtime not in RUNTIMES:
        raise StudyError(f"unknown runtime `{runtime}`", f"runtimes: {', '.join(RUNTIMES)}")
    fixed = _fixed_env(env)
    secret_names = sorted(set(_list(secrets)))
    for s in secret_names:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", s):
            raise StudyError(f"`{s}` is not an environment variable name")
    clash = sorted(set(secret_names) & set(fixed))
    if clash:
        raise StudyError(f"{', '.join(clash)} is both a secret and a fixed variable")
    redact = list(redact or [])
    for rx in redact:
        try:
            re.compile(rx)
        except re.error as e:
            raise StudyError(f"redact pattern {rx!r} does not compile", str(e)) from None

    src = Path(source).expanduser().resolve() if source else None
    files: dict = {}
    manifest_files: dict = {}
    links: dict = {}
    if src is not None:
        if not src.is_dir():
            raise StudyError(f"no folder {source}", fix="--source names the instrument's code folder")
        rels = collect(src, include, exclude)
        if not rels:
            raise StudyError(f"{source} has no files to snapshot", "every file was excluded",
                             "check --exclude, or --include what the instrument needs")
        if len(rels) > MAX_FILES:
            raise StudyError(f"{source} has {len(rels)} files to snapshot", f"an instrument's code is at most "
                             f"{MAX_FILES} files", "exclude data and dependencies (--exclude); dependencies belong to "
                             "the environment (a venv, node_modules), which is fingerprinted, not copied")
        total = 0
        for rel in rels:
            p = src / rel
            total += p.stat().st_size
            manifest_files[rel] = {"sha256": sha_file(p), "mode": "x" if os.access(p, os.X_OK) else ""}
            files[rel] = str(p)
        if total > MAX_BYTES:
            raise StudyError(f"{source} holds {total / 2 ** 20:.0f} MB to snapshot", f"at most {MAX_BYTES // 2 ** 20} "
                             "MB: an instrument is a program, not a dataset", "exclude its data (--exclude)")

    # the entry: a token naming one of the snapshot's files (or, written as a path with a /, one of its folders) runs
    # from the snapshot; a path under the source that is not snapshotted (a venv's interpreter) is part of the
    # environment, by its absolute path. A module name (`-m pkg`) and every other word stay as written.
    out_argv = []
    for i, tok in enumerate(argv):
        cand = src / tok if src is not None and tok and not tok.startswith("-") and "{" not in tok else None
        if cand is not None and (i == 0 or argv[i - 1] != "-m") and cand.exists():
            rel = os.path.relpath(os.path.normpath(cand.absolute()), src)       # a venv's python is a symlink out
            pathlike = "/" in tok or tok.startswith(".")
            if not rel.startswith(".."):
                if rel in manifest_files or (pathlike and any(r.startswith(rel.rstrip("/") + "/")
                                                              for r in manifest_files)):
                    out_argv.append("{snapshot}/" + rel if rel != "." else "{snapshot}")
                    continue
                if pathlike:
                    out_argv.append(os.path.normpath(cand.absolute()))
                    continue
        out_argv.append(tok)
    for tok in out_argv:
        for ph in re.findall(r"\{([a-z_]+)\}", tok):
            if ph not in PLACEHOLDERS:
                raise StudyError(f"unknown placeholder {{{ph}}} in the entry", "placeholders: "
                                 + ", ".join("{" + p + "}" for p in PLACEHOLDERS))
            if ph == "snapshot" and src is None:
                raise StudyError("{snapshot} in the entry of an instrument with no source folder")
            if ph == "state" and not state:
                raise StudyError("{state} in the entry of an instrument that keeps no state", fix="add --state")
            if ph == "config" and not config:
                raise StudyError("{config} in the entry of an instrument with no config", fix="add --config FILE")
    exe0 = out_argv[0]
    rt = runtime or detect_runtime(exe0)

    # the environment parts: a path is stored absolute (under the source if it is there), a bare name is looked up on
    # PATH at every check
    def _abs(path: str) -> str:
        if not path or "/" not in path or path.startswith("{"):
            return path
        c = src / path if src is not None and (src / path).exists() else Path(path).expanduser()
        return os.path.normpath(c.absolute())
    python, node, exe0 = _abs(python), _abs(node), _abs(exe0)
    out_argv[0] = exe0                              # a run's working folder is its output folder: no relative argv[0]
    probe = ["--version"] if version_probe else None
    specs: list = []
    if rt == "python":
        specs.append(("python", {"path": python or exe0}))
    elif python:
        specs.append(("python", {"path": python}))
    nm = str(src / "node_modules") if src is not None and (src / "node_modules").is_dir() else ""
    if rt == "node":
        base = Path(exe0).name
        specs.append(("node", {"path": exe0 if base in ("node", "nodejs") else (node or "node"),
                               **({"node_modules": nm} if nm else {})}))
        if base not in ("node", "nodejs"):
            specs.append(("binary", {"path": exe0, "version_args": probe}))
        if nm:
            links["node_modules"] = nm
    elif node:
        specs.append(("node", {"path": node}))
    if rt == "binary":
        if exe0.startswith("{snapshot}/"):
            rel0 = exe0[len("{snapshot}/"):]
            interp = _shebang(Path(files.get(rel0, "")))
            if interp:
                specs.append(("binary", {"path": interp[0], "version_args": None}))
            elif not manifest_files.get(rel0, {}).get("mode"):
                raise StudyError(f"{rel0} is neither executable nor a #! script",
                                 fix=f'chmod +x it, or name its interpreter first: --entry "sh {rel0}"')
        else:
            specs.append(("binary", {"path": exe0, "version_args": probe}))
    for b in _list(binaries):
        specs.append(("binary", {"path": _abs(b), "version_args": probe}))
    for f in env_files or ():
        specs.append(("file", {"path": str(Path(f).expanduser().resolve())}))
    for c in env_cmds or ():
        specs.append(("cmd", {"cmd": c, "cwd": str(src) if src else ""}))
    seen, parts = set(), []
    for kind, spec in specs:
        key = canonical([kind, spec])
        if key in seen:
            continue
        seen.add(key)
        part = fingerprint_part(kind, spec)
        d = part["detail"]
        if d.get("missing"):
            raise StudyError(f"{kind} {spec.get('path') or spec.get('cmd')} is not found",
                             "an instrument's environment is fingerprinted when it is registered",
                             "install it, or name it by its full path")
        if d.get("error"):
            raise StudyError(f"the {kind} probe of {spec.get('path')} failed", d["error"][-300:])
        parts.append(part)

    cfg_sha, cfg_file = "", ""
    if config:
        cp = Path(config).expanduser().resolve()
        if not cp.is_file():
            raise StudyError(f"no config file {config}")
        cfg_sha, cfg_file = sha_file(cp), str(cp)
    manifest = {"name": name, "entry": out_argv, "runtime": rt, "capabilities": caps, "secrets": secret_names,
                "env": fixed, "outputs": _list(outputs), "state": bool(state),
                "timeout": float(timeout or TIMEOUT), "redact": redact, "files": manifest_files, "links": links,
                "config": cfg_sha, "config_suffix": Path(cfg_file).suffix if cfg_file else "",
                "environment": {"parts": parts, "sha": environment_sha(parts)}}
    manifest["usage"], usage_text = _usage(manifest, files, cfg_file, usage=usage, args=usage_args, sub=usage_sub,
                                           supplied=usage_file)
    sha = manifest_sha(manifest)
    return Prepared(name, manifest, sha, files, str(src) if src else "", cfg_file, usage_text)


# --- the usage: how the entry says it is called, probed at register --------------------------------------------------
_SUBCOMMANDS = re.compile(r"(?:^|(?<=[\s\[(]))\{([A-Za-z0-9_][A-Za-z0-9_.:-]*(?:,[A-Za-z0-9_][A-Za-z0-9_.:-]*)+)\}")


def parse_subcommands(text: str) -> list:
    """The subcommands an argparse-style help lists, `{a,b,c}`, in order; an option's choices
    (`--runtime {a,b}`) are not subcommands."""
    for mt in _SUBCOMMANDS.finditer(text):
        before = text[:mt.start()].rstrip().rsplit(None, 1)
        if before and before[-1].lstrip("[(").startswith("-"):
            continue
        return list(dict.fromkeys(mt.group(1).split(",")))[:USAGE_MAX_SUBCOMMANDS]
    return []


def _usage_probe(argv: list, cwd, env: dict) -> tuple:
    """(exit code or None, output, reason): one usage probe, stdout and stderr together, in its own process group so
    a timeout stops everything it started."""
    import signal
    try:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, start_new_session=True)
    except FileNotFoundError:
        return None, "", f"{argv[0]}: not found"
    except OSError as e:
        return None, "", f"{argv[0]}: {e}"
    try:
        out, _ = proc.communicate(timeout=USAGE_TIMEOUT)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        out, _ = proc.communicate()
        return None, out.decode("utf-8", "replace"), f"timed out after {USAGE_TIMEOUT:g}s"
    text = out.decode("utf-8", "replace")
    return proc.returncode, text, "" if proc.returncode == 0 else f"exit {proc.returncode}"


def _cap(text: str, limit: int) -> str:
    data = text.encode("utf-8")
    if len(data) <= limit:
        return text
    return data[:limit].decode("utf-8", "ignore") + f"\n[usage cut at {limit // 1024} KB]\n"


def _usage(m: dict, files: dict, cfg_file: str, *, usage: bool, args, sub: bool, supplied: str) -> tuple:
    """(the manifest's usage record, its redacted text). The probe runs the entry with `args` (default --help) in a
    throwaway copy of what a run gets (the snapshot, read-only; an empty output and state folder; the config; the
    minimal environment with the secrets that are set), for at most USAGE_TIMEOUT seconds each. A failed probe is
    recorded as unavailable, with why; it never refuses the registration."""
    import tempfile
    args = shlex.split(args) if isinstance(args, str) else [str(a) for a in (args or ())]
    args = args or list(USAGE_ARGS)
    secret_values = {k: os.environ[k] for k in m["secrets"] if os.environ.get(k)}
    if supplied:
        path = Path(supplied).expanduser()
        if not path.is_file():
            raise StudyError(f"no usage file {supplied}", fix="--usage-file names a text file with the usage")
        text = _cap(_redact(path.read_bytes().decode("utf-8", "replace"), m, secret_values), USAGE_MAX_BYTES)
        return ({"status": "supplied", "args": [], "reason": f"from {path.name}", "subcommands": [],
                 "sha256": sha_bytes(text.encode("utf-8")), "bytes": len(text.encode("utf-8"))}, text)
    if not usage:
        return {"status": "skipped", "args": [], "reason": "--no-usage", "subcommands": [], "sha256": "",
                "bytes": 0}, ""
    tmp = Path(tempfile.mkdtemp(prefix="openlab-usage-"))
    try:
        snap = None
        if files:
            snap = tmp / "snapshot"
            for rel, path in files.items():
                data = Path(path).read_bytes()
                if sha_bytes(data) != m["files"][rel]["sha256"]:
                    raise StudyError(f"{rel} changed while it was being registered", fix="register again")
                q = snap / rel
                q.parent.mkdir(parents=True, exist_ok=True)
                q.write_bytes(data)
                q.chmod(0o555 if m["files"][rel].get("mode") == "x" else 0o444)
            for rel, target in (m.get("links") or {}).items():
                (snap / rel).parent.mkdir(parents=True, exist_ok=True)
                os.symlink(target, snap / rel)
            _readonly(snap)
        out = tmp / "out"
        out.mkdir()
        state_dir = tmp / "state" if m["state"] else None
        if state_dir is not None:
            state_dir.mkdir()
        cfg = None
        if cfg_file:
            cfg = tmp / f"config{m.get('config_suffix', '')}"
            shutil.copyfile(cfg_file, cfg)
        argv, env = _launch(m, snap, out, state_dir, cfg, secret_values)
        code, text, reason = _usage_probe(argv + args, out, env)
        text = _cap(_redact(text, m, secret_values), USAGE_MAX_BYTES)
        subs = parse_subcommands(text) if sub and code == 0 else []
        for name in subs:
            if len(text.encode("utf-8")) >= USAGE_MAX_BYTES:
                text += f"\n[{USAGE_MAX_BYTES // 1024} KB of usage: the other subcommands were not probed]\n"
                break
            c, t, why = _usage_probe(argv + [name, *args], out, env)
            head = f"{m['name']} {name} {' '.join(args)}"
            text += f"\n--- {head} ---\n" if c == 0 else f"\n--- {head}: unavailable ({why}) ---\n"
            text += _redact(t, m, secret_values)
        text = _cap(text, USAGE_MAX_BYTES)
    finally:
        _writable(tmp)
        shutil.rmtree(tmp, ignore_errors=True)
    data = text.encode("utf-8")
    return ({"status": "ok" if code == 0 else "unavailable", "args": args, "reason": reason, "subcommands": subs,
             "sha256": sha_bytes(data) if data else "", "bytes": len(data)}, text)


def usage_head(u: dict) -> str:
    """The usage's one-line state, as `show` and the registration preview print it."""
    if not u:
        return "usage: not recorded (registered before usage capture)"
    if u["status"] == "skipped":
        return "usage: not captured (--no-usage)"
    if u["status"] == "unavailable":
        return f"usage: unavailable ({u['reason']})"
    how = " ".join(u["args"]) if u["status"] == "ok" else u["reason"]
    subs = f", {len(u['subcommands'])} subcommands" if u.get("subcommands") else ""
    return f"usage ({how}{subs}, sha256 {u['sha256'][:12]})"


def usage_view(reg: "Registry", m: dict) -> dict:
    """A version's usage: its record, its one-line state and its text, read from the content store (the store
    refuses bytes that no longer match the sha)."""
    u = dict(m.get("usage") or {})
    text = ""
    if u.get("sha256"):
        text = reg.blob(u["sha256"]).decode("utf-8", "replace")
    return {**u, "status": u.get("status", "not recorded"), "head": usage_head(m.get("usage") or {}), "text": text}


def preview(p: Prepared) -> list:
    """The lines a person reads before signing a registration."""
    m = p.manifest
    lines = [f"version {p.ident} (sha256 {p.sha})", f"entry: {shlex.join(m['entry'])}", f"runtime: {m['runtime']}",
             f"capabilities: {', '.join(m['capabilities'])}"
             + ("  -- every run is a person's act (live-order)" if set(m["capabilities"]) & set(PERSON_CAPABILITIES)
                else "")]
    if m["secrets"]:
        lines.append(f"secrets passed by name: {', '.join(m['secrets'])}")
    if m["env"]:
        lines.append("fixed env: " + ", ".join(f"{k}={v}" for k, v in m["env"].items()))
    if m["outputs"]:
        lines.append(f"outputs: {', '.join(m['outputs'])}")
    if p.source:
        size = sum(Path(f).stat().st_size for f in p.files.values())
        lines.append(f"snapshot of {p.source}: {len(m['files'])} files, {size / 1024:.0f} KB")
        lines += [f"  {r}" for r in list(m["files"])[:40]]
        if len(m["files"]) > 40:
            lines.append(f"  ... {len(m['files']) - 40} more")
    else:
        lines.append("no source folder: an external command, verified before each run, never copied")
    for part in m["environment"]["parts"]:
        d = part["detail"]
        what = d.get("version") or (f"{len(d['dists'])} distributions" if "dists" in d else "") or \
            d.get("sha256", "")[:12]
        lines.append(f"environment: {_label(part)}" + (f" ({str(what).splitlines()[0][:60]})" if what else ""))
    if m["config"]:
        lines.append(f"config: {p.config_file} (sha256 {m['config'][:12]})")
    lines.append(f"state folder: {'yes' if m['state'] else 'no'}; timeout {m['timeout']:g}s")
    text = p.usage_text.splitlines()
    lines.append(usage_head(m.get("usage") or {}) + (":" if text else ""))
    lines += [f"  | {x}" for x in text[:PREVIEW_USAGE_LINES]]
    if len(text) > PREVIEW_USAGE_LINES:
        lines.append(f"  | ... {len(text) - PREVIEW_USAGE_LINES} more lines (all of it is kept; "
                     "`openlab instrument show REF --usage` prints it)")
    return lines


def commit(root, p: Prepared, reason: str) -> Node:
    """Write a prepared registration: its files into the content store and a read-only snapshot, then the signed
    `instrument` node. A person's act: refused without the person key unlocked."""
    if not (reason or "").strip():
        raise StudyError("a registration needs a reason", fix="--reason says what the instrument is for")
    if personkey.current() is None:
        raise StudyError(f"registering instrument {p.ident} is a person's act, signed with the person key",
                         "this process holds no unlocked person key, so it is not the person acting",
                         "the person registers it in their own terminal; an agent files "
                         'agent_request(kind="instrument", args={...}) with the exact registration')
    reg = Registry(root)
    found = reg.find("instrument", name=p.name, version=p.version)
    if found:
        raise StudyError(f"{p.ident} is already registered ({found[0].id})", "a version is its manifest's sha: the "
                         "same code, environment and settings are the same version", "use it as it is")
    for rel, path in p.files.items():
        data = Path(path).read_bytes()
        if sha_bytes(data) != p.manifest["files"][rel]["sha256"]:
            raise StudyError(f"{rel} changed while it was being registered", fix="register again")
        reg.put_blob(data)
    if p.config_file:
        data = Path(p.config_file).read_bytes()
        if sha_bytes(data) != p.manifest["config"]:
            raise StudyError("the config changed while it was being registered", fix="register again")
        reg.put_blob(data)
    usage = p.manifest.get("usage") or {}
    if usage.get("sha256"):
        if reg.put_blob(p.usage_text.encode("utf-8")) != usage["sha256"]:
            raise StudyError("the usage text is not the one the manifest names", fix="register again")
    payload = {**{k: p.manifest[k] for k in MANIFEST_KEYS}, "version": p.version, "sha": p.sha, "source": p.source,
               "reason": reason.strip(), **({"usage": usage} if usage else {})}
    node = reg.add("instrument", payload, actor="person")
    _materialise_snapshot(reg, node.payload)
    return node


def register(root, name: str, entry, *, reason: str, **kw) -> dict:
    """prepare and commit, as the person (the person key must be unlocked: `personkey.signing`)."""
    return _node(commit(root, prepare(root, name, entry, **kw), reason))


def _readonly(path: Path) -> None:
    w = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    for d, dirs, files in sorted(os.walk(path), key=lambda t: -len(t[0])):
        for f in files:
            p = Path(d) / f
            if not p.is_symlink():
                p.chmod(p.stat().st_mode & ~w)
        Path(d).chmod(Path(d).stat().st_mode & ~w)


def _writable(path: Path) -> None:
    for d, dirs, files in os.walk(path):
        Path(d).chmod(Path(d).stat().st_mode | stat.S_IWUSR)
        for f in files:
            p = Path(d) / f
            if not p.is_symlink():
                p.chmod(p.stat().st_mode | stat.S_IWUSR)


def _materialise_snapshot(reg: Registry, m: dict) -> Optional[Path]:
    """The version's snapshot folder (and its config copy), rebuilt from the content store when missing. None for an
    external instrument (no files)."""
    if m.get("config"):
        cp = reg.config_path(ident(m), m.get("config_suffix", ""))
        if not cp.is_file():
            cp.parent.mkdir(parents=True, exist_ok=True)
            tmp = cp.with_name(f".{cp.name}.{os.getpid()}")
            tmp.write_bytes(reg.blob(m["config"]))
            tmp.chmod(0o444)
            os.replace(tmp, cp)
    if not m["files"]:
        return None
    snap = reg.snapshot_dir(ident(m))
    if snap.is_dir():
        return snap
    snap.parent.mkdir(parents=True, exist_ok=True)
    tmp = snap.with_name(f".{snap.name}.{os.getpid()}.{_secrets.token_hex(3)}")
    for rel, f in m["files"].items():
        p = tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(reg.blob(f["sha256"]))
        p.chmod(0o555 if f.get("mode") == "x" else 0o444)
    for rel, target in (m.get("links") or {}).items():
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        os.symlink(target, tmp / rel)
    _readonly(tmp)
    try:
        os.rename(tmp, snap)
    except OSError:                                 # another process made it first
        _writable(tmp)
        shutil.rmtree(tmp, ignore_errors=True)
    return snap


# --- finding a version ------------------------------------------------------------------------------------------------
def _node(n: Node) -> dict:
    return {"id": n.id, "seq": n.seq, "kind": n.kind, "actor": n.actor, "agent": n.session, "at": n.at,
            "payload": n.payload}


def versions(reg: Registry, name: str = "") -> list:
    return reg.find("instrument", name=name) if name else reg.find("instrument")


def resolve(reg: Registry, ref: str, study=None) -> Node:
    """The registered version `ref` names: NAME@VERSION, or a bare NAME with one version; with `study`, also a pin's
    alias, and only a version the study pins."""
    pins = dict(study.spec.raw.get("instruments") or {}) if study is not None else None
    want = (ref or "").strip()
    if pins is not None and want in pins:
        want = pins[want]
    m = REF.fullmatch(want)
    if not m:
        raise StudyError(f"`{ref}` names no instrument", "an instrument is named NAME@VERSION (or a study's alias)",
                         "`openlab instrument list` lists them")
    name, version = m.group(1), m.group(2)
    if pins is not None:
        pinned = [p for p in pins.values() if p.split("@")[0] == name]
        if version is None and len(pinned) == 1:
            version = pinned[0].split("@")[1]
        if f"{name}@{version}" not in pins.values():
            raise StudyError(f"{study.spec.name} does not pin {name}{'@' + version if version else ''}",
                             f"its [instruments]: {', '.join(f'{k} = {v}' for k, v in pins.items()) or 'none'}; a "
                             "study runs and reads only the versions it pins", PIN_FIX)
    found = versions(reg, name)
    if not found:
        raise StudyError(f"no instrument `{name}` is registered here", f"workspace {reg.root}",
                         "a person registers it (openlab instrument register, in their own terminal)")
    if version is None:
        if len(found) > 1:
            raise StudyError(f"`{name}` has {len(found)} versions", "a run is bound to one version: "
                             + ", ".join(ident(n.payload) for n in found), f"name one: {ident(found[-1].payload)}")
        return found[0]
    hit = [n for n in found if n.payload["version"] == version]
    if not hit:
        raise StudyError(f"no version {version} of `{name}`", "registered: "
                         + ", ".join(ident(n.payload) for n in found))
    return hit[0]


# --- verification -----------------------------------------------------------------------------------------------------
def verify(reg: Registry, node: Node, environment: bool = True) -> dict:
    """Every hash of a version again: the snapshot's files (none missing, changed or added), its config, and each
    environment part. {instrument, ok, files: [problems], config, environment: [{part, ok, changes}], external}."""
    m = node.payload
    out = {"instrument": ident(m), "ok": True, "files": [], "config": "", "environment": [], "rebuilt": False,
           "external": not m["files"]}
    if manifest_sha(m) != m["sha"] or m["version"] != m["sha"][:12]:
        out.update(ok=False, files=["the registration does not hash to its version: the record is inconsistent"])
        return out
    if m["files"]:
        snap = reg.snapshot_dir(ident(m))
        if not snap.is_dir():
            out["rebuilt"] = True
            try:
                _materialise_snapshot(reg, m)
            except StudyError as e:
                out.update(ok=False, files=[f"the snapshot is gone and cannot be rebuilt: {e.error}"])
                return out
        links = m.get("links") or {}
        seen = set()
        for d, dirs, files in os.walk(snap):
            for x in list(dirs) + list(files):
                p = Path(d) / x
                rel = str(p.relative_to(snap))
                if p.is_symlink():
                    if x in dirs:
                        dirs.remove(x)
                    seen.add(rel)
                    if links.get(rel) != os.readlink(p):
                        out["files"].append(f"{rel}: a link that is not the registered one")
                    continue
                if x in dirs:
                    continue
                seen.add(rel)
                want = m["files"].get(rel)
                if want is None:
                    out["files"].append(f"{rel}: added to the snapshot")
                elif sha_file(p) != want["sha256"]:
                    out["files"].append(f"{rel}: changed")
        out["files"] += [f"{rel}: missing from the snapshot" for rel in [*m["files"], *links] if rel not in seen]
    if m.get("config"):
        cp = reg.config_path(ident(m), m.get("config_suffix", ""))
        if not cp.is_file():
            _materialise_snapshot(reg, m)
        if sha_file(cp) != m["config"]:
            out["config"] = "the config copy changed"
    if environment:
        for part in m["environment"]["parts"]:
            now = fingerprint_part(part["kind"], part["spec"])
            ok = now["sha"] == part["sha"]
            out["environment"].append({"part": _label(part), "ok": ok,
                                       "changes": [] if ok else _part_changes(part, now)})
    out["ok"] = not out["files"] and not out["config"] and all(e["ok"] for e in out["environment"])
    return out


def drift(v: dict) -> str:
    """A verification's problems, as one line."""
    lines = list(v["files"]) + ([v["config"]] if v["config"] else [])
    for e in v["environment"]:
        if not e["ok"]:
            lines.append(f"{e['part']}: " + "; ".join(e["changes"][:6])
                         + (f"; and {len(e['changes']) - 6} more" if len(e["changes"]) > 6 else ""))
    return "; ".join(lines)


# --- running ----------------------------------------------------------------------------------------------------------
def _launch(m: dict, snap, out, state_dir, cfg, secret_values: dict) -> tuple:
    """The entry's argv (placeholders filled) and the minimal environment a run gets: PATH, HOME and the locale, the
    fixed variables, the secrets, OPENLAB_INSTRUMENT_*, and for a python entry PYTHONPATH at the snapshot. The usage
    probe at register gets the same, in a throwaway copy."""
    values = {"snapshot": str(snap or ""), "out": str(out), "state": str(state_dir or ""), "config": str(cfg or "")}
    argv = [re.sub(r"\{(" + "|".join(PLACEHOLDERS) + r")\}", lambda x: values[x.group(1)], t) for t in m["entry"]]
    env = {k: os.environ[k] for k in BASE_ENV if k in os.environ}
    env.update(m["env"])
    env.update(secret_values)
    env.update({ENV_PREFIX: f"{m['name']}@{m.get('version') or 'unregistered'}",
                f"{ENV_PREFIX}_VERSION": m.get("version") or "", f"{ENV_PREFIX}_OUT": str(out),
                f"{ENV_PREFIX}_CAPABILITIES": ",".join(m["capabilities"])})
    if snap is not None:
        env[f"{ENV_PREFIX}_SNAPSHOT"] = str(snap)
    if state_dir is not None:
        env[f"{ENV_PREFIX}_STATE"] = str(state_dir)
    if cfg is not None:
        env[f"{ENV_PREFIX}_CONFIG"] = str(cfg)
    if m["runtime"] == "python":
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        if snap is not None:
            env["PYTHONPATH"] = os.pathsep.join(x for x in (str(snap), m["env"].get("PYTHONPATH", "")) if x)
    return argv, env


def _redact(text: str, m: dict, secret_values: dict) -> str:
    for name, value in secret_values.items():
        if value and len(value) >= 4:
            text = text.replace(value, f"[secret {name}]")
    for rx in m.get("redact") or []:
        text = re.sub(rx, "[redacted]", text)
    return text


def _as_text(data: bytes) -> Optional[str]:
    """The bytes as text (UTF-8, no NUL), or None: a binary output (parquet, an image) is not returned inline."""
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _outputs(reg: Registry, out: Path, patterns: list, m: Optional[dict] = None,
             secret_values: Optional[dict] = None) -> dict:
    """The declared outputs the run wrote, each kept as written (`sha256`, what a dataset reads). A text output that
    the redaction changes (a secret's value, a redact pattern) also keeps its redacted copy (`redacted`), the one a
    caller is shown."""
    found: dict = {}
    total = 0
    for pat in patterns:
        for p in sorted(out.glob(pat)):
            rel = str(p.relative_to(out))
            if not p.is_file() or p.is_symlink() or rel in found:
                continue
            data = p.read_bytes()
            total += len(data)
            if total > OUTPUT_MAX_BYTES:
                raise StudyError(f"the run's outputs pass {OUTPUT_MAX_BYTES // 2 ** 20} MB",
                                 "an instrument's declared outputs are kept in the content store",
                                 "declare fewer or smaller outputs")
            found[rel] = {"sha256": reg.put_blob(data), "bytes": len(data)}
            text = _as_text(data) if m is not None and len(data) <= READ_MAX else None
            if text is not None:
                masked = _redact(text, m, secret_values or {})
                if masked != text:
                    found[rel]["redacted"] = reg.put_blob(masked.encode("utf-8"))
    return found


def _tail(text: str) -> dict:
    """The last TAIL_LINES lines of a log, at most TAIL_BYTES: {text, lines, bytes, truncated}."""
    lines = text.splitlines(keepends=True)
    tail = "".join(lines[-TAIL_LINES:])
    data = tail.encode("utf-8")
    if len(data) > TAIL_BYTES:
        tail = data[-TAIL_BYTES:].decode("utf-8", "ignore")
        cut = tail.find("\n")
        tail = tail[cut + 1:] if 0 <= cut < len(tail) - 1 else tail
    nbytes = len(text.encode("utf-8"))
    return {"text": tail, "lines": len(lines), "bytes": nbytes, "truncated": len(tail.encode("utf-8")) < nbytes}


def _secrets_here(m: dict) -> dict:
    return {k: os.environ[k] for k in m["secrets"] if os.environ.get(k)}


def _output_text(reg: Registry, rel: str, o: dict, m: dict, secret_values: dict, limit: int) -> tuple:
    """(text or None, why withheld): a declared output's content as a caller sees it, its sha checked first (the
    kept file and its redacted copy both), redacted again with the patterns and the secrets set here."""
    data = reg.blob(o["sha256"])                    # the content store refuses bytes that no longer match the sha
    if len(data) > limit:
        return None, WITHHELD
    text = _as_text(data)
    if text is None:
        return None, "not text; materialise it in a study"
    if o.get("redacted"):
        text = reg.blob(o["redacted"]).decode("utf-8", "replace")
    return _redact(text, m, secret_values), ""


def run_view(reg: Registry, payload: dict, m: dict, secret_values: Optional[dict] = None) -> dict:
    """What a run's caller reads: its exit code, the redacted stdout and stderr tails, and each declared output's
    content when it is text and at most INLINE_MAX (else its size, sha and why it is withheld). Each sha is checked
    against the content store first."""
    sv = _secrets_here(m) if secret_values is None else secret_values
    view: dict = {"exit": payload["exit"], "timed_out": payload["timed_out"]}
    for k in ("stdout", "stderr"):
        view[k] = _tail(_redact(reg.blob(payload[k]).decode("utf-8", "replace"), m, sv))
    outs: dict = {}
    for rel, o in payload["outputs"].items():
        text, why = _output_text(reg, rel, o, m, sv, INLINE_MAX)
        outs[rel] = {"bytes": o["bytes"], "sha256": o["sha256"], **({"content": text} if why == "" else
                                                                     {"withheld": why})}
    view["outputs"] = outs
    return view


def output(root, run_id: str, file: str = "") -> dict:
    """A recorded run's redacted logs and outputs, read back from the content store with each sha checked (a
    changed blob is refused). Without `file`: as `run` returned them (log tails, small text outputs inline). With
    `file` (a declared output, or `stdout` / `stderr`): that one whole, if it is text and at most READ_MAX. Nothing
    but the instrument record's runs and their blobs is read."""
    reg = Registry(root)
    try:
        node = reg.get(run_id)
    except StudyError:
        node = None
    if node is None or node.kind != "instrument_run":
        raise StudyError(f"no instrument run `{run_id}` on the instrument record",
                         "a run is named by the id `instrument run` printed (on the instrument record)",
                         "`openlab instrument runs` lists them")
    p = node.payload
    m = reg.get(p["registration"]).payload
    sv = _secrets_here(m)
    head = {"id": node.id, "instrument": p["instrument"], "args": p["args"], "study": p["study"],
            "started": p["started"], "ended": p["ended"]}
    if not file:
        return {**head, **run_view(reg, p, m, sv)}
    if file in p["outputs"]:
        o = p["outputs"][file]
        text, why = _output_text(reg, file, o, m, sv, READ_MAX)
        if why:
            raise StudyError(f"{file} is {o['bytes']} bytes: {why}" if why == WITHHELD else f"{file}: {why}",
                             f"its sha256 is {o['sha256']}; `instrument output` prints text of at most "
                             f"{READ_MAX // 2 ** 20} MB", f"declare a study dataset from it ([data.x] instrument = "
                             f"\"alias\", output = \"{file}\") and materialise it")
        return {**head, "file": file, "bytes": o["bytes"], "sha256": o["sha256"], "content": text}
    if file in ("stdout", "stderr"):
        text = _redact(reg.blob(p[file]).decode("utf-8", "replace"), m, sv)
        return {**head, "file": file, "bytes": len(text.encode("utf-8")), "sha256": p[file], "content": text}
    raise StudyError(f"run {node.id} has no output `{file}`",
                     f"its outputs: {', '.join(p['outputs']) or 'none'}; and its logs: stdout, stderr")


def _open_study(reg: Registry, study: str):
    from .. import study as _study
    folder = Path(study) if Path(study).is_absolute() else reg.root / study
    return _study.open(folder)


def run(root, ref: str, args=(), *, study: str = "", note: str = "", timeout: Optional[float] = None,
        reason: str = "") -> dict:
    """Run a registered version (after verifying it) and record the run. A `live-order` instrument runs only as the
    person (the person key unlocked, and a reason): a signed authority for this one run is written first."""
    reg = Registry(root)
    args = [str(a) for a in (args or [])]
    try:
        st = _open_study(reg, study) if study else None
        node = resolve(reg, ref, st)
        m = node.payload
        live = bool(set(m["capabilities"]) & set(PERSON_CAPABILITIES))
        if live and personkey.current() is None:
            raise StudyError(f"{ident(m)} can place live orders: each run is a person's act, signed for that run",
                             "this process holds no unlocked person key, so it is not the person acting",
                             "the person runs it in their own terminal; an agent files "
                             f'agent_request(kind="instrument_run", args={{"instrument": "{ident(m)}", "args": '
                             f"{json.dumps(args)}}}, reason=...)")
        if live and not (reason or note).strip():
            raise StudyError("a live-order run needs a reason", fix="--reason says why this run")
        v = verify(reg, node)
        if not v["ok"]:
            raise StudyError(f"{ident(m)} is not the version that was registered", drift(v),
                             "a version never changes: register the instrument again (a new version), then pin it")
        secret_values = {}
        for s in m["secrets"]:
            if s not in os.environ:
                raise StudyError(f"the secret {s} is not set", f"{ident(m)} reads it from the environment of the "
                                 "process that runs it", f"export {s} in the shell that runs the instrument")
            secret_values[s] = os.environ[s]
    except StudyError as e:
        reg.add("action", {"verb": "instrument_run", "args": {"ref": ref, "args": args, "study": study},
                           "outcome": {"ok": False, "error": str(e)}}, actor="agent")
        raise
    authority = None
    if live:
        authority = reg.add("instrument_authority", {"instrument": ident(m), "args": args, "study": study,
                                                     "reason": (reason or note).strip()}, actor="person")
    snap = _materialise_snapshot(reg, m)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = reg.dir / "runs" / ident(m) / f"{stamp}-{_secrets.token_hex(3)}"
    out = run_dir / "out"
    out.mkdir(parents=True)
    state_dir = reg.state_dir(m["name"]) if m["state"] else None
    if state_dir is not None:
        state_dir.mkdir(parents=True, exist_ok=True)
    cfg = reg.config_path(ident(m), m.get("config_suffix", "")) if m.get("config") else None
    argv, env = _launch(m, snap, out, state_dir, cfg, secret_values)
    argv += args
    started = _now()
    timed_out = False
    with open(run_dir / "stdout.log", "wb") as fo, open(run_dir / "stderr.log", "wb") as fe:
        try:
            p = subprocess.run(argv, cwd=out, env=env, stdout=fo, stderr=fe, stdin=subprocess.DEVNULL,
                               timeout=float(timeout or m["timeout"]))
            code = p.returncode
        except subprocess.TimeoutExpired:
            code, timed_out = -9, True
        except OSError as e:
            fe.write(f"openlab: the instrument did not start: {e}\n".encode())
            code = 126
    ended = _now()
    logs = {}
    for k in ("stdout", "stderr"):
        text = _redact((run_dir / f"{k}.log").read_bytes().decode("utf-8", "replace"), m, secret_values)
        (run_dir / f"{k}.log").write_text(text, encoding="utf-8")
        logs[k] = reg.put_blob(text.encode("utf-8"))
    payload = {"instrument": ident(m), "registration": node.id, "args": args, "note": note, "study": study,
               "started": started, "ended": ended, "exit": code, "timed_out": timed_out, "stdout": logs["stdout"],
               "stderr": logs["stderr"], "outputs": _outputs(reg, out, m["outputs"], m, secret_values),
               "environment": m["environment"]["sha"], "capabilities": m["capabilities"],
               "run_dir": str(run_dir.relative_to(reg.root)), "registry_run": ""}
    actor = "person" if live else "agent"
    rn = reg.add("instrument_run", payload, actor=actor, links=[("under", authority.id)] if authority else ())
    res = {**_node(rn), "authority": authority.id if authority else "", "log": str(run_dir / "stdout.log"),
           **run_view(reg, payload, m, secret_values)}
    if st is not None:
        res["study_run"] = st._act("instrument_run", {"instrument": ident(m), "args": args}, actor, lambda: _node(
            st.record.add("instrument_run", {**payload, "registry_run": rn.id}, actor=actor)))["id"]
    return res


def runs(reg: Registry, ref: str = "", study: str = "", n: int = 20) -> list:
    found = reg.find("instrument_run")
    if ref:
        m = REF.fullmatch(ref)
        if not m:
            raise StudyError(f"`{ref}` names no instrument")
        found = [r for r in found if r.payload["instrument"].split("@")[0] == m.group(1)
                 and (m.group(2) is None or r.payload["instrument"].endswith("@" + m.group(2)))]
    if study:
        found = [r for r in found if r.payload["study"] == study]
    return [_node(r) for r in reversed(found)][:n]


def listing(root=None) -> list:
    """Every registered instrument with its versions, oldest first."""
    if not Registry.exists(root):
        return []
    reg = Registry(root)
    counts: dict = {}
    for r in reg.find("instrument_run"):
        counts[r.payload["instrument"]] = counts.get(r.payload["instrument"], 0) + 1
    by: dict = {}
    for n in versions(reg):
        m = n.payload
        by.setdefault(m["name"], []).append({
            "id": ident(m), "node": n.id, "at": n.at, "by": (m.get("sig") or {}).get("by", n.session),
            "capabilities": m["capabilities"], "runtime": m["runtime"], "external": not m["files"],
            "files": len(m["files"]), "runs": counts.get(ident(m), 0), "reason": m["reason"]})
    return [{"name": k, "versions": v} for k, v in by.items()]


# --- datasets from an instrument ----------------------------------------------------------------------------------------
@dataclass
class Output:
    """A frame read from an instrument run's output, with where it came from (merged into the dataset's source)."""
    frame: object
    provenance: dict


READABLE = (".parquet", ".csv", ".json", ".jsonl", ".ndjson")


def dataset_output(folder: Path, pin: str, output: str) -> Output:
    """The latest successful recorded run of `pin` that wrote `output`, read after its sha256 is checked."""
    import io
    import polars as pl
    reg = Registry(folder)
    recorded = reg.find("instrument_run", instrument=pin)
    hit = next((r for r in reversed(recorded) if r.payload["exit"] == 0 and output in r.payload["outputs"]), None)
    if hit is None:
        raise StudyError(f"no successful run of {pin} wrote {output}", f"{len(recorded)} recorded runs of {pin}",
                         f"run it first: openlab instrument run {pin} --study STUDY -- ARGS")
    sha = hit.payload["outputs"][output]["sha256"]
    data = reg.blob(sha)                            # the content store refuses bytes that no longer match the sha
    readers = {".parquet": pl.read_parquet, ".csv": pl.read_csv, ".json": pl.read_json,
               ".jsonl": pl.read_ndjson, ".ndjson": pl.read_ndjson}
    suffix = Path(output).suffix.lower()
    if suffix not in readers:
        raise StudyError(f"cannot read {output} as a table", f"readable: {', '.join(READABLE)}")
    return Output(readers[suffix](io.BytesIO(data)), {"run": hit.id, "output_sha256": sha,
                                                      "run_ended": hit.payload["ended"]})
