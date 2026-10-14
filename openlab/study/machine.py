"""What this machine lets openlab use, and a per-machine gate on sandboxed children.

In a container, `os.cpu_count()` and /proc/meminfo describe the host, not what the container may use: a vast.ai or
cloud container may see 96 cores and 251 GB while its cgroup allows 23 CPUs and 120 GB. `cores()` and `memory()` read
the limits that bind (the cgroup's CPU quota and CPU set, its memory limit and use; v2 or v1), so a run never plans
for, or starts, more than the machine allows.

`slot()` holds one of `cores()` lock files under ~/.openlab/box/slots (OPENLAB_BOX_DIR) while a sandboxed child runs, so
every openlab process on the machine together runs at most that many children. A lock is a flock: it goes with the
process that held it, so a killed run never leaves a slot taken.
"""

from __future__ import annotations

import fcntl
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path

CGROUP = Path("/sys/fs/cgroup")
UNLIMITED = 1 << 60                   # a cgroup v1 limit at or past this is "no limit" (it prints 2^63 rounded down)


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _int(text: str):
    try:
        return int(text)
    except ValueError:
        return None


def cgroup_cpus(root: Path = CGROUP):
    """The CPUs the cgroup's quota allows (quota / period, a float), or None when it sets none."""
    v2 = _read(root / "cpu.max").split()
    if len(v2) == 2 and v2[0] != "max" and _int(v2[0]) and _int(v2[1]):
        return int(v2[0]) / int(v2[1])
    for d in (root / "cpu", root / "cpu,cpuacct"):
        quota, period = _int(_read(d / "cpu.cfs_quota_us")), _int(_read(d / "cpu.cfs_period_us"))
        if quota and quota > 0 and period:
            return quota / period
    return None


def cores(root: Path = CGROUP) -> int:
    """The cores openlab may use at once: the CPUs this process may run on, cut to the cgroup's quota (at least 1)."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    quota = cgroup_cpus(root)
    if quota is not None:
        n = min(n, max(1, math.floor(quota + 1e-9)))
    return max(1, n)


def _meminfo() -> tuple:
    try:
        info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line)
        return int(info["MemAvailable"].split()[0]) * 1024, int(info["MemTotal"].split()[0]) * 1024
    except (OSError, KeyError, ValueError):
        return None, None


def _stat(path: Path) -> dict:
    out = {}
    for line in _read(path).splitlines():
        k, _, v = line.partition(" ")
        if _int(v) is not None:
            out[k] = int(v)
    return out


def cgroup_memory(root: Path = CGROUP):
    """(available, limit) in bytes under the cgroup's memory limit, or None when it sets none. Reclaimable page
    cache (inactive files) counts as available, as the kernel reclaims it before it kills."""
    limit = _int(_read(root / "memory.max"))                        # v2 ("max" when unlimited: no int)
    if limit is not None:
        used = _int(_read(root / "memory.current")) or 0
        used -= _stat(root / "memory.stat").get("inactive_file", 0)
    else:
        limit = _int(_read(root / "memory" / "memory.limit_in_bytes"))
        if limit is None or limit >= UNLIMITED:
            return None
        used = _int(_read(root / "memory" / "memory.usage_in_bytes")) or 0
        st = _stat(root / "memory" / "memory.stat")
        used -= st.get("total_inactive_file", st.get("inactive_file", 0))
    if limit <= 0 or limit >= UNLIMITED:
        return None
    return max(0, limit - max(0, used)), limit


def memory(root: Path = CGROUP) -> tuple:
    """(available, total) memory in bytes: the host's (/proc/meminfo), cut to the cgroup's limit when it sets one;
    (None, None) where neither can be read."""
    avail, total = _meminfo()
    cg = cgroup_memory(root)
    if cg is None:
        return avail, total
    if total is None:
        return cg
    return min(avail, cg[0]), min(total, cg[1])


def describe(root: Path = CGROUP) -> dict:
    """{cores, memory_gb, host_cores, host_memory_gb, limited}: what openlab uses, and what the host reports."""
    avail, total = memory(root)
    host = _meminfo()[1]
    n = cores(root)
    return {"cores": n, "memory_gb": round(total / 2 ** 30, 1) if total else None,
            "host_cores": os.cpu_count() or 1, "host_memory_gb": round(host / 2 ** 30, 1) if host else None,
            "limited": n < (os.cpu_count() or 1) or bool(total and host and total < host)}


# --- the per-machine gate ---------------------------------------------------------------------------------------
def box_dir() -> Path:
    return Path(os.environ.get("OPENLAB_BOX_DIR") or Path.home() / ".openlab" / "box")


@contextmanager
def slot(wait: float = 0.25):
    """Hold one of the machine's `cores()` child slots for the block, waiting (polling every `wait` s) until one is
    free. OPENLAB_BOX_SLOTS overrides the count; 0 turns the gate off."""
    env = os.environ.get("OPENLAB_BOX_SLOTS")
    n = int(env) if env not in (None, "") else cores()
    if n <= 0:
        yield None
        return
    d = box_dir() / "slots"
    d.mkdir(parents=True, exist_ok=True)
    while True:
        for i in range(n):
            f = open(d / f"{i}.lock", "a+")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                f.close()
                continue
            try:
                yield i
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
                f.close()
            return
        time.sleep(wait)


# --- GPUs --------------------------------------------------------------------------------------------------------
def gpus() -> list:
    """[{index, name, memory_mb}] of the GPUs nvidia-smi lists, cut to CUDA_VISIBLE_DEVICES when the parent sets it
    (openlab never hands out a GPU its own process was not given); [] without nvidia-smi."""
    import shutil
    import subprocess
    if not shutil.which("nvidia-smi"):
        return []
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return []
    out = []
    for line in p.stdout.splitlines() if p.returncode == 0 else []:
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 3 and _int(parts[0]) is not None:
            out.append({"index": int(parts[0]), "name": parts[1], "memory_mb": _int(parts[2])})
    given = os.environ.get("CUDA_VISIBLE_DEVICES")
    if given is not None:
        allowed = {_int(x.strip()) for x in given.split(",") if x.strip()}
        out = [g for g in out if g["index"] in allowed]
    return out


@contextmanager
def lease_gpus(n: int, wait: float = 1.0, timeout: float | None = None):
    """Hold `n` of this machine's GPUs for the block (a lock file each under the box dir, like the core slots),
    waiting until that many are free; yields their indices. n <= 0 yields []. Refused at once when the machine has
    fewer than `n`; after `timeout` seconds of waiting, refused too."""
    if n <= 0:
        yield []
        return
    have = gpus()
    if len(have) < n:
        from .errors import StudyError
        raise StudyError(f"the run asks for {n} GPU(s); this machine has {len(have)}",
                         "[sandbox] gpus is what one run holds for its whole length",
                         "run it on a machine with enough GPUs, or lower [sandbox] gpus (a person's amendment)")
    d = box_dir() / "gpus"
    d.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    while True:
        held = []
        for g in have:
            f = open(d / f"{g['index']}.lock", "a+")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                f.close()
                continue
            held.append((g["index"], f))
            if len(held) == n:
                break
        if len(held) == n:
            try:
                yield [i for i, _ in held]
            finally:
                for _, f in held:
                    fcntl.flock(f, fcntl.LOCK_UN)
                    f.close()
            return
        for _, f in held:                                    # not enough free: let go of the partial set and wait
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()
        if timeout is not None and time.monotonic() - start > timeout:
            from .errors import StudyError
            raise StudyError(f"no {n} GPU(s) came free in {timeout:.0f} s", "other runs hold them")
        time.sleep(wait)
