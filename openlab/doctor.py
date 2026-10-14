r"""``openlab doctor``: what works here for studies and agents, what does not, and the one line that fixes it.

First the workspace (openlab.toml, the Claude Code hooks, the person key, the agent CLIs, the agents working here),
then the install (the MCP, the sandbox, the extras), then where things live.

The failure modes are quiet rather than loud: without bwrap a study's runs are recorded as `unsandboxed`, without
mlflow `openlab study run` cannot keep its model, without the built UI ``openlab app`` serves an API and a blank page.
Each of those is one row here with the command that repairs it.

It never prints a key. Whether an agent CLI is signed in is between the user and that CLI; this reports only
whether the binary is on PATH, and for each data provider's key only whether it is set.

Paths are reported **for the current directory**, since that is what decides them: run it in the workspace you
are asking about.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

#: extra -> the modules it is supposed to provide, and what stops working without it. The MCP (mcp) and the web
#: workspace (fastapi, uvicorn) are core dependencies: the "core" row checks them.
EXTRAS = {
    "study": (("polars", "numpy", "sklearn", "mlflow", "cloudpickle"),
              "`openlab study run`: the data, the runner, the templates' scikit-learn and the model store (mlflow)"),
}

#: the agent CLIs openlab works with: the user's own, never a key of openlab's
AGENT_CLIS = ("claude", "codex")


def agent_cmd() -> Optional[str]:
    """The agent pane's command: ``$OPENLAB_AGENT_CMD``, else ``claude`` if it is on PATH, else None."""
    return os.environ.get("OPENLAB_AGENT_CMD") or shutil.which("claude")


def _have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):            # a namespace package shadowed by a broken install
        return False


class Report:
    """Rows, and whether any of them is a problem."""

    def __init__(self) -> None:
        self.rows: list = []

    def add(self, name: str, ok: Optional[bool], detail: str, fix: str = "", note: str = "") -> None:
        """``ok`` is True, False, or None for a row that is information rather than a verdict.

        ``fix`` is strictly the command that repairs a failure -- never a remark -- so an agent reading
        ``--json`` can act on every non-empty one. A remark goes in ``note``."""
        self.rows.append({"check": name, "ok": ok, "detail": detail, "fix": fix, "note": note})

    @property
    def broken(self) -> list:
        return [r for r in self.rows if r["ok"] is False]

    def to_record(self) -> dict:
        return {"ok": not self.broken, "python": sys.version.split()[0], "openlab": _version(),
                "cwd": str(Path.cwd()), "rows": self.rows}


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("openlab-research")
    except Exception:                            # noqa: BLE001 -- running from a checkout without an install
        return "(from source)"


def check(here: Optional[Path] = None) -> Report:
    """Every row, in the order a person reads them: can it run, can it run a study, where does it put things.

    ``here`` is the workspace the paths are resolved for (default: the current directory)."""
    r = Report()
    v = sys.version_info
    r.add("python", v >= (3, 10), f"{v.major}.{v.minor}.{v.micro} at {sys.executable}",
          "" if v >= (3, 10) else "openlab needs Python 3.10 or later; install one and re-create the venv")
    r.add("openlab", True, f"{_version()} from {Path(__file__).resolve().parent}")

    core = [m for m in ("sympy", "lark", "nacl", "fastapi", "uvicorn") if not _have(m)]
    r.add("core", not core, "the equation gate (sympy/lark: derived claims, the paper's equations), signing "
          "(pynacl) and the web workspace (fastapi/uvicorn) are importable" if not core else
          f"missing {', '.join(core)}",
          "" if not core else "pip install --force-reinstall openlab   # core dependencies are missing")

    for name, ok, where, note in _workspace(here or Path.cwd()):
        r.add(name, ok, where, note=note)
    pin = _pin(here or Path.cwd())
    if pin is not None:
        r.add("person pin", *pin)

    found = {c: shutil.which(c) for c in AGENT_CLIS}
    have = [c for c, p in found.items() if p]
    r.add("agent CLI", bool(have) or bool(os.environ.get("OPENLAB_AGENT_CMD")),
          "; ".join(f"{c}: {'found (' + found[c] + ')' if found[c] else 'not found'}" for c in AGENT_CLIS)
          + ("" if have else " -- openlab runs the research through your own agent CLI"),
          "" if have or os.environ.get("OPENLAB_AGENT_CMD") else
          "npm install -g @anthropic-ai/claude-code   (or @openai/codex; openlab never holds a key)")

    ok, detail = _mcp()
    r.add("MCP", ok, detail, "" if ok else "pip install --force-reinstall openlab (mcp is a core dependency), then "
          "`claude mcp add openlab -- openlab-mcp`")

    for name, (mods, what) in EXTRAS.items():
        missing = [m for m in mods if not _have(m)]
        r.add(f"extra: {name}", not missing,
              what if not missing else f"{what} -- missing {', '.join(missing)}",
              "" if not missing else f'pip install "openlab-research[{name}]"')

    ok, detail = _sandbox()
    r.add("sandbox", ok, detail, "" if ok else _sandbox_fix())

    detail, note = _machine()
    r.add("machine", None, detail, note=note)
    gpus = _gpus()
    if gpus is not None:
        r.add("GPUs", None, gpus, note="what nvidia-smi lists; a run sees the ones CUDA_VISIBLE_DEVICES leaves it")

    r.add("p-values", _have("scipy"),
          "scipy is here: t p-values are exact" if _have("scipy") else
          "scipy is missing: t p-values (Diebold-Mariano) fall back to the normal approximation, optimistic on "
          "small samples", "" if _have("scipy") else "pip install scipy")

    ui = Path(__file__).resolve().parent / "app" / "static" / "index.html"
    r.add("web UI", ui.is_file(),
          f"built, at {ui.parent}" if ui.is_file() else "not in this install: `openlab app` serves the API and a "
          "blank page",
          "" if ui.is_file() else "build it: cd openlab/app/ui && npm install && npm run build")

    for name, where, note in _paths(here or Path.cwd()):
        r.add(name, None, where, note=note)
    return r


def _sandbox() -> tuple:
    """Does Python start under bwrap here, the way a study's runner starts its child?"""
    import tempfile
    try:
        from .study.sandbox import _bwrap
    except Exception as exc:                     # noqa: BLE001 -- the study extra is missing
        return False, f"the study runner does not import: {type(exc).__name__}: {exc}"
    with tempfile.TemporaryDirectory() as d:
        study, stage = Path(d) / "study", Path(d) / "stage"
        study.mkdir()
        stage.mkdir()
        try:
            prefix, why = _bwrap(study, stage, False)
        except Exception as exc:                 # noqa: BLE001
            return False, f"bwrap check failed: {type(exc).__name__}: {exc}"
    return (True, "bwrap starts Python: a study's runs are sandboxed (no network, the record masked)") if prefix \
        else (False, f"{why}: a study's runs would be recorded as unsandboxed")


def _userns_blocked() -> bool:
    """Whether this machine refuses a user namespace outright (a container's seccomp profile): `unshare -U` says
    "Operation not permitted"."""
    import subprocess
    if not shutil.which("unshare"):
        return False
    try:
        p = subprocess.run(["unshare", "-U", "-r", "true"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return p.returncode != 0 and "not permitted" in (p.stderr or "").lower()


def _sandbox_fix() -> str:
    """The fix for a missing sandbox: install bwrap, unless this is a container that forbids user namespaces, where
    no install helps."""
    if _userns_blocked():
        return ("this container forbids user namespaces (its seccomp profile), so no sandbox can start here: runs "
                "are recorded as unsandboxed; keep the vault and the record on another machine")
    return ("install bubblewrap (apt install bubblewrap) and allow unprivileged user namespaces; without it runs are "
            "recorded as unsandboxed")


def _machine() -> tuple:
    """(detail, note): the cores and memory openlab uses (the container's limits when it sets them)."""
    try:
        from .study.machine import box_dir, describe
    except Exception as exc:                     # noqa: BLE001
        return f"unknown: {type(exc).__name__}: {exc}", ""
    m = describe()
    detail = f"{m['cores']} cores, {m['memory_gb']} GB" + (
        f" (the container's limits; the host reports {m['host_cores']} cores, {m['host_memory_gb']} GB)"
        if m["limited"] else "")
    return detail, f"at most {m['cores']} sandboxed children run at once on this machine (slots in {box_dir()})"


def _gpus() -> Optional[str]:
    """The GPUs nvidia-smi lists (name, memory), or None without nvidia-smi."""
    import subprocess
    if not shutil.which("nvidia-smi"):
        return None
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return "nvidia-smi did not answer"
    rows = [r.strip() for r in p.stdout.splitlines() if r.strip()]
    return "; ".join(rows) if p.returncode == 0 and rows else "nvidia-smi lists none"


def _mcp() -> tuple:
    """Does the MCP entry build its tool list? That is what fails when an extra is missing, and it fails at
    import rather than at connect, so a user sees it as "the MCP does not start"."""
    if not _have("mcp"):
        return False, "the `mcp` package is not installed"
    try:
        import openlab.mcp_server as M
        mgr = getattr(M.server, "_tool_manager", None)
        n = len(getattr(mgr, "_tools", None) or getattr(M.server, "_tools", None) or [])
        return True, ("`openlab-mcp` imports and registers "
                      + (f"{n} tools" if n else "its tools"))
    except Exception as exc:                     # noqa: BLE001
        return False, f"importing openlab.mcp_server raised {type(exc).__name__}: {exc}"


def _hooks(ws: Path) -> Optional[Path]:
    """The Claude Code settings file that runs openlab's hooks (the workspace's, else the user's), or None."""
    from .agents.claude_setup import is_hook_command
    for p in (ws / ".claude" / "settings.json", ws / ".claude" / "settings.local.json",
              Path.home() / ".claude" / "settings.json"):
        try:
            hooks = dict(json.loads(p.read_text(encoding="utf-8")).get("hooks") or {})
        except (OSError, ValueError, AttributeError, TypeError):
            continue
        if any(is_hook_command(h.get("command", "")) for entries in hooks.values() if isinstance(entries, list)
               for e in entries if isinstance(e, dict) for h in e.get("hooks", []) if isinstance(h, dict)):
            return p
    return None


def _workspace(here: Path) -> list:
    """The workspace this directory belongs to: openlab.toml and its studies, the hooks, the person key, and the
    agents working here (read from the control store, never started). Each row is (name, ok, detail, note): ok is
    True for a part that is set up, None for information (a part not set up yet is not a failure)."""
    out = []
    from .study.workspace import root, studies
    ws = root(here)
    marker = (ws / "openlab.toml").is_file()
    try:
        found = studies(ws)
    except OSError as exc:                       # a folder under the workspace this user cannot read
        out.append(("workspace", None, str(ws), f"its studies could not be listed: {type(exc).__name__}: {exc}"))
    else:
        out.append(("workspace", True if marker else None,
                    f"{ws} ({'openlab.toml' if marker else 'no openlab.toml: `openlab init` writes it'})",
                    f"{len(found)} stud{'y' if len(found) == 1 else 'ies'}"
                    + (f": {', '.join(str(p.relative_to(ws)) or '.' for p in found[:8])}" if found else
                       " -- start one with `openlab study new NAME`")))
    hooks = _hooks(ws)
    out.append(("hooks", True if hooks else None, f"yes ({hooks})" if hooks else "no",
                "Claude Code reports each agent's calls and refuses a person's acts to it" if hooks else
                "`openlab init --claude` installs them: without them sub-agents are not tracked or gated"))
    from .agents.person import pub_path
    pub = pub_path()
    out.append(("person key", True if pub.is_file() else None, f"yes ({pub})" if pub.is_file() else "no",
                "signs every person act (holdout looks, approvals, protected amendments)" if pub.is_file() else
                "`openlab person init` makes one: a person's acts are refused without it"))
    if (ws / ".openlab" / "control.db").is_file():
        try:
            from .agents.store import LIVE, Store
            st = Store(ws)
            live = [a for a in st.agents() if a["status"] in LIVE]
            waiting = len(st.inbox("pending"))
            out.append(("agents", None, f"{len(live)} live, {waiting} waiting in the inbox",
                        "`openlab agents` shows the tree, `openlab inbox` what needs you"))
        except Exception as exc:                 # noqa: BLE001 -- a report, never a failure
            out.append(("agents", None, f"unreadable ({type(exc).__name__})",
                        "the control store in .openlab/control.db"))
    return out


def _pin(here: Path) -> Optional[tuple]:
    """(ok, detail, fix, note) of the workspace's person-key pin: FAIL when openlab.toml disagrees with the control
    store's pin chain (an edited pin); None when nothing is pinned and there is no key."""
    try:
        from .agents.person import info, pin_state
        s, k = pin_state(here), info()
    except Exception as exc:                     # noqa: BLE001 -- a report, never a failure
        return None, f"unreadable ({type(exc).__name__})", "", ""
    if s["conflict"]:
        return (False, f"TAMPERED: {s['conflict']}",
                f'restore fingerprint = "{s["pin"]}" under [person] in openlab.toml by hand, or run `openlab person '
                "pin` with the pinned key", "")
    if not s["pin"]:
        return (None, "none", "", "`openlab person pin` pins your key here: a pinned workspace takes another key only "
                "through a rotation the pinned key signs") if k else None
    owned = k and k["fingerprint"] == s["pin"]
    return (True if owned else None, s["pin"] + (f" (chain {' -> '.join(s['chain'])})" if len(s["chain"]) > 1 else "")
            + ("" if owned or not k else f"; your key here is {k['fingerprint']}: MISMATCH"), "",
            "compare it with the fingerprint you keep where no agent can write"
            + "".join(f"; warning: control store event {x['seq']}: {x['why']}" for x in s["invalid"]))


def _paths(here: Path) -> list:
    """Where the records resolve. The vault and the signing key deliberately sit outside the workspace."""
    out = []
    try:
        import stat
        from .study.vault import vault_dir
        v = vault_dir(here)
        note = "held-out labels: outside the workspace on purpose, and keyed by the study folder"
        if v.exists():
            mode = stat.S_IMODE(os.stat(v).st_mode)
            if mode & 0o077:
                note += f" -- WARNING: permissions {oct(mode)} let other users read it; chmod 700 {v}"
        out.append(("vault", str(v.parent), note))
    except Exception as exc:                     # noqa: BLE001
        out.append(("vault", f"unavailable ({type(exc).__name__})", ""))
    try:
        import nacl  # noqa: F401
        from .study.signing import key_info
        k = key_info()
        out.append(("signing key", k["path"], f"id {k['key_id']}: publish it so readers can check your signed files"
                    if k["exists"] else "none yet: made on the first signature"))
    except ImportError:
        out.append(("signing key", "unavailable", "PyNaCl is not installed: `pip install pynacl` to sign"))

    from .sources import PROVIDERS
    from .sources.http import data_root          # importing it loads ~/.config/openlab/research.env
    out.append(("snapshots", str(data_root("OPENLAB_SNAPSHOTS", "snapshots")),
                "$OPENLAB_SNAPSHOTS, else research.env, else ~/openlab-data/snapshots"))
    keys: dict = {}
    for p in PROVIDERS.values():
        for var in p.needs:
            keys.setdefault(var, []).append(p.name)
    have = [var for var in sorted(keys) if os.environ.get(var)]
    if have:                                    # a data provider's key: shown only to someone who set one
        out.append(("provider keys", f"set: {', '.join(have)}", "~/.config/openlab/research.env or the environment"))
    return out


def render(r: Report) -> str:
    """The table a person reads: one line per check, the fix under the ones that failed."""
    mark = {True: "ok  ", False: "FAIL", None: "--  "}
    width = max(len(row["check"]) for row in r.rows)
    lines = [f"openlab {_version()}  ·  python {sys.version.split()[0]}  ·  {Path.cwd()}", ""]
    for row in r.rows:
        lines.append(f"  {mark[row['ok']]}  {row['check']:<{width}}  {row['detail']}")
        if row["ok"] is False and row["fix"]:
            lines.append(f"        {'':<{width}}  -> {row['fix']}")
        elif row["note"]:
            lines.append(f"        {'':<{width}}  {row['note']}")
    lines.append("")
    lines.append("all good" if not r.broken else
                 f"{len(r.broken)} to fix: {', '.join(x['check'] for x in r.broken)}")
    return "\n".join(lines)


def main(argv) -> int:
    """``openlab doctor [--json]``. Exit 0 when nothing is broken, so it can gate a setup script."""
    import argparse
    import json
    p = argparse.ArgumentParser(prog="openlab doctor",
                                description="What works here for running studies and agents, and the one line "
                                            "that fixes what does not.")
    p.add_argument("--json", action="store_true", help="the same rows, for an agent")
    a = p.parse_args(argv)
    r = check()
    print(json.dumps(r.to_record(), indent=1) if a.json else render(r))
    return 0 if not r.broken else 1
