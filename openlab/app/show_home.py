"""What `openlab show` keeps for its home (the folder of researches), apart from any one research: the tabs the
person closed, and the machines the Machine queue lists, with each one's usage.

`<home>/.openlab-show.json`:

    {"closed": ["weather"], "machines": [{"name": "gpu-box", "ssh": "gpu-box", "port": 0}]}

A closed tab is only hidden: the research, its agents and its record are untouched, and New session -> Resume opens it
again. A machine is this one (always listed, `local`) or a remote one reached over ssh by a host or a ~/.ssh/config
alias (`remote`); openlab reads its usage, never starts, stops or rents it.

Usage is read by one small shell script (CPU load and cores, memory, disk, GPUs through nvidia-smi), run here with sh
and on a remote with `ssh -o BatchMode=yes`: it never asks for a password or accepts an unknown host key (run `ssh
HOST` once by hand). Each machine's reading is cached and refreshed in the background, so a page never waits on ssh.
"""

from __future__ import annotations

import fcntl
import json
import re
import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

FILE = ".openlab-show.json"
HOST = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._@-]{0,120}")        # a host, user@host or an ssh alias; never an option
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,47}")
FRESH = {"local": 5.0, "remote": 30.0}                           # seconds a reading is kept
SSH_TIMEOUT = 15
SCRIPT = ("hostname; nproc; cat /proc/loadavg; grep -E '^(MemTotal|MemAvailable):' /proc/meminfo; df -Pk / | tail -1; "
          "echo GPUS; nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,"
          "nounits 2>/dev/null; true")


# --- the home's file ------------------------------------------------------------------------------------------------
@contextmanager
def _locked(home: Path):
    with open(Path(home) / f"{FILE}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def read(home: Path) -> dict:
    """The home's file: {closed: [names], machines: [{name, ssh, port}]}; a missing or broken file is empty."""
    try:
        d = json.loads((Path(home) / FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        d = {}
    d = d if isinstance(d, dict) else {}
    closed = [str(x) for x in d.get("closed") or [] if isinstance(x, str)]
    machines = [m for m in d.get("machines") or [] if isinstance(m, dict) and NAME.fullmatch(str(m.get("name", "")))
                and HOST.fullmatch(str(m.get("ssh", "")))]
    return {"closed": closed, "machines": [{"name": m["name"], "ssh": m["ssh"], "port": int(m.get("port") or 0)}
                                           for m in machines]}


def _write(home: Path, d: dict) -> None:
    p = Path(home) / FILE
    tmp = p.with_name(f".{FILE}.tmp")
    tmp.write_text(json.dumps(d, indent=1) + "\n", encoding="utf-8")
    tmp.replace(p)


def set_open(home: Path, name: str, open_: bool) -> list:
    """Show (open_) or hide a research's tab; returns the closed names. Nothing is written when it is so already."""
    if (name in read(home)["closed"]) != bool(open_):
        return read(home)["closed"]
    with _locked(home):
        d = read(home)
        closed = [x for x in d["closed"] if x != name] + ([] if open_ else [name])
        _write(home, {**d, "closed": closed})
        return closed


def add_machine(home: Path, name: str, ssh: str, port: int = 0) -> list:
    """Declare a remote machine reached by `ssh` (a host, user@host or an ~/.ssh/config alias)."""
    name, ssh = str(name or "").strip(), str(ssh or "").strip()
    if not NAME.fullmatch(name) or name == "local":
        raise ValueError("a machine's name is a-z, A-Z, 0-9, . _ -, at most 48, and not 'local'")
    if not HOST.fullmatch(ssh):
        raise ValueError("ssh is a host, user@host or an alias from ~/.ssh/config (letters, digits, . _ @ -)")
    if not 0 <= int(port or 0) <= 65535:
        raise ValueError("the port is 1 to 65535, or 0 for ssh's own")
    with _locked(home):
        d = read(home)
        if any(m["name"] == name for m in d["machines"]):
            raise ValueError(f"there is a machine named {name} already")
        machines = d["machines"] + [{"name": name, "ssh": ssh, "port": int(port or 0)}]
        _write(home, {**d, "machines": machines})
        return machines


def remove_machine(home: Path, name: str) -> list:
    with _locked(home):
        d = read(home)
        machines = [m for m in d["machines"] if m["name"] != name]
        _write(home, {**d, "machines": machines})
        _CACHE.pop(name, None)
        return machines


# --- usage ----------------------------------------------------------------------------------------------------------
def parse(out: str) -> dict:
    """The script's output as {host, cores, load: [1, 5, 15 min], mem_total_gb, mem_used_gb, disk_total_gb,
    disk_used_gb, gpus: [{name, util, mem_used_gb, mem_total_gb}]}."""
    head, _, gpus = out.partition("GPUS")
    lines = [ln.strip() for ln in head.strip().splitlines() if ln.strip()]
    d: dict = {"host": lines[0] if lines else "", "gpus": []}
    try:
        d["cores"] = int(lines[1])
        d["load"] = [float(x) for x in lines[2].split()[:3]]
    except (IndexError, ValueError):
        pass
    mem = {}
    for ln in lines:
        m = re.match(r"(MemTotal|MemAvailable):\s+(\d+)", ln)
        if m:
            mem[m.group(1)] = int(m.group(2)) / 1024 / 1024
    if "MemTotal" in mem:
        d["mem_total_gb"] = round(mem["MemTotal"], 1)
        d["mem_used_gb"] = round(mem["MemTotal"] - mem.get("MemAvailable", 0), 1)
    df = next((ln.split() for ln in lines if ln.startswith("/") and len(ln.split()) >= 5), None)
    if df:
        try:
            d["disk_total_gb"], d["disk_used_gb"] = round(int(df[1]) / 1024 / 1024, 1), round(int(df[2]) / 1024 / 1024, 1)
        except ValueError:
            pass
    for ln in gpus.strip().splitlines():
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) == 4:
            try:
                d["gpus"].append({"name": parts[0], "util": float(parts[1]), "mem_used_gb": round(float(parts[2]) / 1024, 1),
                                  "mem_total_gb": round(float(parts[3]) / 1024, 1)})
            except ValueError:
                continue
    return d


def _probe(m: dict) -> dict:
    if m["where"] == "local":
        argv = ["sh", "-c", SCRIPT]
    else:
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"] + (["-p", str(m["port"])] if m.get("port") else []) \
            + ["--", m["ssh"], SCRIPT]
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=SSH_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"ok": False, "error": f"no answer in {SSH_TIMEOUT} s" if isinstance(e, subprocess.TimeoutExpired) else str(e)}
    if r.returncode and "GPUS" not in r.stdout:
        err = (r.stderr.strip().splitlines() or [f"exit {r.returncode}"])[-1]
        return {"ok": False, "error": err[:200]}
    return {"ok": True, **parse(r.stdout)}


_CACHE: dict = {}            # {machine name: (time read, reading)}
_BUSY: set = set()
_LOCK = threading.Lock()


def _refresh(m: dict) -> None:
    try:
        got = _probe(m)
    finally:
        with _LOCK:
            _BUSY.discard(m["name"])
    _CACHE[m["name"]] = (time.time(), {**got, "at": time.strftime("%H:%M:%S")})


def usage(home: Path, wait: float = 0.0) -> list:
    """Every machine with its last reading: this one first (`local`), then the remote ones declared. A reading older
    than FRESH is refreshed in the background (`wait`: seconds to wait for a machine never read yet)."""
    machines = [{"name": "local", "where": "local", "ssh": "", "port": 0, "host": socket.gethostname()}] + \
        [{**m, "where": "remote", "host": m["ssh"]} for m in read(home)["machines"]]
    started = []
    for m in machines:
        hit = _CACHE.get(m["name"])
        if hit is None or time.time() - hit[0] > FRESH[m["where"]]:
            with _LOCK:
                if m["name"] in _BUSY:
                    continue
                _BUSY.add(m["name"])
            t = threading.Thread(target=_refresh, args=(m,), daemon=True, name=f"openlab-usage-{m['name']}")
            t.start()
            if hit is None:
                started.append(t)
    end = time.time() + wait
    for t in started:
        t.join(max(0.0, end - time.time()))
    out = []
    for m in machines:
        hit = _CACHE.get(m["name"])
        out.append({**m, **(hit[1] if hit else {"ok": None, "error": "", "at": ""})})
    return out


def local_queue() -> dict:
    """This machine's queue and schedule (openlab.study.admission, box.toml): running, waiting, until, reservations."""
    from .show_tools import box, queue
    b = box()
    return {"running": b["running"], "waiting": b["waiting"], "until": b["until"], "reservations": b["reservations"],
            "tickets": queue()}


def machines(home: Path, wait: float = 0.0) -> list:
    out = usage(home, wait)
    try:
        q = local_queue()
    except Exception:                                                       # noqa: BLE001
        q = {}
    for m in out:
        if m["where"] == "local":
            m["queue"] = q
    return out


def known(home: Path, name: str) -> Optional[dict]:
    return next((m for m in read(home)["machines"] if m["name"] == name), None)
