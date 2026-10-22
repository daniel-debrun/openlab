"""The person's inventory of nodes: this machine, and the `[[node]]` entries of box.toml (the box dir,
OPENLAB_BOX_DIR, default ~/.openlab/box), which only the person edits.

    [[node]]                       # a machine reached over SSH; openlab places work on it by its own bookkeeping
    name = "vast-3060"
    host = "root@66.96.91.70"
    port = 41615
    workdir = "openlab-nodes"      # where allocations' folders go (relative: under the login's home)
    unsandboxed = "ok"             # the person's OK for runs without bwrap there (needed when the node has none)
    # optional: cores, memory_gb, gpus (cut what the probe finds), keep_cores, keep_memory_gb, gpu_model, exclude

    [[node]]                       # a Slurm cluster, through its login node: one allocation is one job
    name = "vulcan"
    backend = "slurm"
    host = "vulcan"
    account = "aip-mtaylor3"
    workdir = "/scratch/USER/openlab-nodes"    # on a filesystem the compute nodes share
    # optional: partitions = ["gpubase_bygpu_b1", ...] (default: every partition sinfo lists), gpu_model

`probe(node)` reads what a node offers (cores and memory under its cgroup's limits, GPUs, bwrap) and keeps it in
nodes.json beside box.toml; `offer(node)` is that, cut to the entry's own numbers and less what it keeps back.
OPENLAB_NODE_SSH replaces the ssh program (tests run the "remote" side locally).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import tomllib
except ModuleNotFoundError:                                         # pragma: no cover -- python 3.10
    import tomli as tomllib

from ..study import machine

LOCAL = "local"
BACKENDS = ("local", "ssh", "slurm")
NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,30}$")
FRESH = 600                                                         # seconds a probe stays current
#: what never goes to a node: the record, the keys and the person's files (and what is rebuilt there)
EXCLUDE = (".openlab/", ".claude/", ".mcp.json", "openlab.toml", ".env", "*.key", ".venv/", "venv/", "__pycache__/",
           "*.pyc", "node_modules/")


class NodeError(Exception):
    """A refusal with what it means and what to try (the shape of an agent error)."""

    def __init__(self, error: str, meaning: str = "", try_: str = ""):
        super().__init__(error)
        self.error, self.meaning, self.try_ = error, meaning, try_


@dataclass
class Node:
    name: str
    backend: str
    host: str = ""
    port: int = 0
    workdir: str = "openlab-nodes"
    cores: Optional[int] = None
    memory_gb: Optional[float] = None
    gpus: Optional[int] = None
    gpu_model: str = ""
    keep_cores: int = 0
    keep_memory_gb: float = 0.0
    account: str = ""
    partitions: list = field(default_factory=list)
    exclude: list = field(default_factory=list)
    unsandboxed: str = ""
    ssh_options: list = field(default_factory=list)

    @property
    def remote(self) -> bool:
        return self.backend != "local"

    def ssh(self) -> list:
        """The ssh argv up to and including the host (the command follows)."""
        argv = [os.environ.get("OPENLAB_NODE_SSH") or "ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                "-o", "ClearAllForwardings=yes", "-o", "LogLevel=ERROR", "-o", "ServerAliveInterval=30"]
        for o in self.ssh_options:
            argv += ["-o", o]
        if self.port:
            argv += ["-p", str(self.port)]
        return argv + [self.host]

    def sh(self, script: str, timeout: float = 120, check: bool = True) -> str:
        """Run `script` with bash on the node (here for the local node); its stdout."""
        argv = ["bash", "-c", script] if not self.remote else self.ssh() + ["bash -c " + shlex.quote(script)]
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            raise NodeError(f"{self.name} did not answer in {timeout:.0f} s", f"running: {script[:80]}",
                            f"check that `ssh {self.host}` works from this machine") from None
        if check and p.returncode:
            raise NodeError(f"on {self.name}: {(p.stderr or p.stdout).strip()[-600:] or f'exit {p.returncode}'}",
                            f"running: {script[:160]}")
        return p.stdout

    def rsh(self) -> str:
        """The -e argument rsync reaches the node with."""
        return " ".join(shlex.quote(a) for a in self.ssh()[:-1])

    def path(self, rel: str) -> str:
        """A path under the node's workdir, as rsync and the shell there read it."""
        return f"{self.workdir.rstrip('/')}/{rel}" if self.workdir else rel


def home() -> Path:
    """The person's box dir, where box.toml, nodes.json and the allocations live. Inside a local allocation
    OPENLAB_BOX_DIR is the allocation's own (its runs' queue), and OPENLAB_NODES_HOME keeps the person's."""
    return Path(os.environ.get("OPENLAB_NODES_HOME") or machine.box_dir())


def box_toml(box: Optional[Path] = None) -> Path:
    return (box or home()) / "box.toml"


def nodes(box: Optional[Path] = None) -> list:
    """This machine (`local`) and the box.toml `[[node]]` entries, checked."""
    path = box_toml(box)
    raw = tomllib.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    b = raw.get("box", {})
    out = [Node(LOCAL, "local", keep_cores=int(b.get("keep_cores", 0)),
                keep_memory_gb=float(b.get("keep_memory_gb", 0)))]
    for e in raw.get("node", []):
        name = str(e.get("name") or "")
        if not NAME.match(name) or name in [n.name for n in out]:
            raise NodeError(f"box.toml: [[node]] name {name!r} is missing, repeated or not lower-case letters, digits "
                            "and dashes", str(path))
        backend = str(e.get("backend") or ("ssh" if e.get("host") else ""))
        if backend not in ("ssh", "slurm") or not e.get("host"):
            raise NodeError(f"box.toml: node {name} needs a host (and backend \"ssh\", the default, or \"slurm\")",
                            str(path))
        known = {f for f in Node.__dataclass_fields__} - {"name", "backend"}
        extra = sorted(set(e) - known - {"name", "backend"})
        if extra:
            raise NodeError(f"box.toml: node {name} has unknown keys {', '.join(extra)}", str(path))
        out.append(Node(name=name, backend=backend, **{k: e[k] for k in known if k in e}))
    return out


def get(name: str, box: Optional[Path] = None) -> Node:
    for n in nodes(box):
        if n.name == name:
            return n
    raise NodeError(f"no node {name!r}", "the inventory is this machine and box.toml's [[node]] entries",
                    "`openlab node list` shows them")


# --- what a node offers ------------------------------------------------------------------------------------------
PROBE = r"""
import json, math, os, shutil, socket, subprocess, sys
def rd(p):
    try:
        return open(p).read().strip()
    except OSError:
        return ""
cpus = sorted(os.sched_getaffinity(0))
q = rd("/sys/fs/cgroup/cpu.max").split()
n = len(cpus)
if len(q) == 2 and q[0] != "max":
    n = min(n, max(1, math.floor(int(q[0]) / int(q[1]) + 1e-9)))
info = dict(l.split(":", 1) for l in open("/proc/meminfo") if ":" in l)
mem = int(info["MemTotal"].split()[0]) * 1024
lim = rd("/sys/fs/cgroup/memory.max")
if lim.isdigit():
    mem = min(mem, int(lim))
gpus = []
if shutil.which("nvidia-smi"):
    p = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    for line in p.stdout.splitlines() if p.returncode == 0 else []:
        i, name, m = [x.strip() for x in line.split(",")]
        gpus.append({"index": int(i), "name": name, "memory_mb": int(m)})
print(json.dumps({"host": socket.gethostname(), "cpus": cpus[:n], "cores": n, "memory_gb": round(mem / 2**30, 1),
                  "gpus": gpus, "bwrap": bool(shutil.which("bwrap")), "rsync": bool(shutil.which("rsync")),
                  "taskset": bool(shutil.which("taskset")), "python": sys.version.split()[0]}))
"""

SINFO = "sinfo -h -o '%P|%l|%c|%m|%G|%a'"


def _minutes(limit: str) -> Optional[int]:
    """A Slurm time limit (D-HH:MM:SS, HH:MM:SS, MM:SS, MM, infinite) in minutes; None when unlimited."""
    if limit in ("infinite", "UNLIMITED", ""):
        return None
    days, _, rest = limit.rpartition("-")
    parts = [int(x) for x in rest.split(":")]
    h, m = (parts[0], parts[1]) if len(parts) == 3 else (0, parts[0]) if len(parts) <= 2 else (0, 0)
    return int(days or 0) * 1440 + h * 60 + m


def parse_sinfo(text: str) -> list:
    """`sinfo -h -o '%P|%l|%c|%m|%G|%a'` as [{name, default, minutes, cores, memory_gb, gpus, gpu_model, up}], one row
    per partition (its largest node shape)."""
    out = {}
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 6:
            continue
        name, limit, cpus, mem, gres, avail = parts
        gpus, model = 0, ""
        for g in gres.split(","):
            m = re.match(r"gpu:(?:([A-Za-z0-9_.-]+):)?(\d+)", g)
            if m:
                gpus, model = int(m.group(2)), m.group(1) or ""
        row = {"name": name.rstrip("*"), "default": name.endswith("*"), "minutes": _minutes(limit),
               "cores": int(re.sub(r"\D", "", cpus) or 0), "memory_gb": round(int(re.sub(r"\D", "", mem) or 0) / 1024, 1),
               "gpus": gpus, "gpu_model": model, "up": avail == "up"}
        old = out.get(row["name"])
        if old is None or (row["gpus"], row["cores"], row["memory_gb"]) > (old["gpus"], old["cores"], old["memory_gb"]):
            out[row["name"]] = row
    return list(out.values())


def _cache_path() -> Path:
    return home() / "nodes.json"


def _cache() -> dict:
    try:
        return json.loads(_cache_path().read_text())
    except (OSError, ValueError):
        return {}


def probe(node: Node, fresh: float = FRESH) -> dict:
    """What `node` has, read on it (cached `fresh` seconds in nodes.json): for local and ssh {host, cpus, cores,
    memory_gb, gpus: [{index, name, memory_mb}], bwrap, ...}; for slurm {partitions: [...]}."""
    cache = _cache()
    hit = cache.get(node.name)
    if hit and time.time() - hit.get("probed_at", 0) < fresh and hit.get("backend") == node.backend:
        return hit
    if node.backend == "local":
        d = _local()
    elif node.backend == "ssh":
        d = json.loads(node.sh("python3 -c " + shlex.quote(PROBE), timeout=60).strip().splitlines()[-1])
    else:
        d = {"partitions": parse_sinfo(node.sh(SINFO, timeout=60))}
    d = {**d, "backend": node.backend, "probed_at": time.time()}
    cache = _cache()
    cache[node.name] = d
    home().mkdir(parents=True, exist_ok=True)
    tmp = _cache_path().with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=1))
    os.replace(tmp, _cache_path())
    return d


def _local() -> dict:
    """This machine whole (not an allocation's share: a lead inside one places its children on the machine)."""
    import shutil
    import socket
    try:
        cpus = sorted(int(x) for x in _ranges(Path("/sys/devices/system/cpu/online").read_text().strip()))
    except (OSError, ValueError):
        cpus = list(range(os.cpu_count() or 1))
    quota = machine.cgroup_cpus()
    n = min(len(cpus), max(1, int(quota + 1e-9))) if quota else len(cpus)
    total = machine.memory(share=False)[1]
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    gpus = []
    if shutil.which("nvidia-smi"):
        p = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, env=env)
        for line in p.stdout.splitlines() if p.returncode == 0 else []:
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 3 and parts[0].isdigit():
                gpus.append({"index": int(parts[0]), "name": parts[1], "memory_mb": int(parts[2] or 0)})
    return {"host": socket.gethostname(), "cpus": cpus[:n], "cores": n,
            "memory_gb": round(total / 2 ** 30, 1) if total else 0.0, "gpus": gpus,
            "bwrap": bool(shutil.which("bwrap")), "rsync": bool(shutil.which("rsync")),
            "taskset": bool(shutil.which("taskset"))}


def _ranges(text: str) -> list:
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        elif part:
            out.append(int(part))
    return out


def offer(node: Node, d: Optional[dict] = None) -> dict:
    """What openlab may hand out on `node`: {cpus, cores, memory_gb, gpus: [{index, name, memory_mb}]} for local and
    ssh nodes (the probe, cut to the entry's numbers, less what it keeps back: the highest-numbered cores), or
    {partitions} for slurm (each partition's largest node: what one allocation can be at most)."""
    d = d if d is not None else probe(node)
    if node.backend == "slurm":
        parts = [p for p in d.get("partitions", []) if p["up"] and (not node.partitions or p["name"] in node.partitions)]
        return {"partitions": parts}
    cpus = list(d.get("cpus") or [])
    if node.cores is not None:
        cpus = cpus[:max(0, int(node.cores))]
    if node.keep_cores:
        cpus = cpus[:max(0, len(cpus) - int(node.keep_cores))]
    mem = float(d.get("memory_gb") or 0)
    if node.memory_gb is not None:
        mem = min(mem, float(node.memory_gb)) if mem else float(node.memory_gb)
    mem = max(0.0, mem - float(node.keep_memory_gb or 0))
    gpus = list(d.get("gpus") or [])
    if node.gpus is not None:
        gpus = gpus[:max(0, int(node.gpus))]
    return {"cpus": cpus, "cores": len(cpus), "memory_gb": round(mem, 1), "gpus": gpus}
