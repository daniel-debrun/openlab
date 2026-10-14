"""Guarantees of openlab.study.machine: in a container, openlab uses the cgroup's CPU quota and memory limit, not the
host's counts (a vast.ai container saw 96 cores and 251 GB under a 23-CPU, 120.8 GB cgroup); and the per-machine
gate never lets more sandboxed children run at once than it has slots, across threads and processes."""

import threading
import time

import pytest

from openlab.study import machine

GB = 2 ** 30


def v1(tmp_path, quota="2304000", period="100000", limit=str(int(120.8 * GB)), usage=str(40 * GB),
       inactive=str(10 * GB)):
    """A cgroup v1 tree as the vast.ai container has it."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "cpu").mkdir()
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text(quota + "\n")
    (tmp_path / "cpu" / "cpu.cfs_period_us").write_text(period + "\n")
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "memory.limit_in_bytes").write_text(limit + "\n")
    (tmp_path / "memory" / "memory.usage_in_bytes").write_text(usage + "\n")
    (tmp_path / "memory" / "memory.stat").write_text(f"cache 1\ntotal_inactive_file {inactive}\n")
    return tmp_path


def test_a_v1_container_quota_and_limit_bind(tmp_path, monkeypatch):
    root = v1(tmp_path)
    monkeypatch.setattr(machine.os, "sched_getaffinity", lambda pid: set(range(96)))
    monkeypatch.setattr(machine, "_meminfo", lambda: (215 * GB, 251 * GB))
    assert machine.cgroup_cpus(root) == pytest.approx(23.04)
    assert machine.cores(root) == 23
    avail, total = machine.memory(root)
    assert total == int(120.8 * GB) and avail == int(120.8 * GB) - 30 * GB


def test_a_v2_container_and_no_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(machine.os, "sched_getaffinity", lambda pid: set(range(16)))
    monkeypatch.setattr(machine, "_meminfo", lambda: (50 * GB, 64 * GB))
    (tmp_path / "cpu.max").write_text("400000 100000\n")
    (tmp_path / "memory.max").write_text(str(8 * GB) + "\n")
    (tmp_path / "memory.current").write_text(str(3 * GB) + "\n")
    (tmp_path / "memory.stat").write_text(f"anon 1\ninactive_file {GB}\n")
    assert machine.cores(tmp_path) == 4 and machine.memory(tmp_path) == (6 * GB, 8 * GB)
    (tmp_path / "cpu.max").write_text("max 100000\n")
    (tmp_path / "memory.max").write_text("max\n")
    assert machine.cores(tmp_path) == 16 and machine.memory(tmp_path) == (50 * GB, 64 * GB)
    unlimited = v1(tmp_path / "v1", quota="-1", limit=str(9223372036854771712))
    assert machine.cores(unlimited) == 16 and machine.memory(unlimited) == (50 * GB, 64 * GB)


def test_a_fractional_quota_below_one_core_still_gives_one(tmp_path, monkeypatch):
    monkeypatch.setattr(machine.os, "sched_getaffinity", lambda pid: {0, 1})
    assert machine.cores(v1(tmp_path, quota="50000")) == 1


def test_the_gate_runs_at_most_its_slots_at_once(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENLAB_BOX_DIR", str(tmp_path))
    monkeypatch.setenv("OPENLAB_BOX_SLOTS", "2")
    live, peak, lock = [0], [0], threading.Lock()

    def job():
        with machine.slot(wait=0.01):
            with lock:
                live[0] += 1
                peak[0] = max(peak[0], live[0])
            time.sleep(0.05)
            with lock:
                live[0] -= 1

    threads = [threading.Thread(target=job) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 2 and live[0] == 0


def test_a_slot_held_by_another_process_is_waited_for(tmp_path, monkeypatch):
    import fcntl
    import subprocess
    import sys
    monkeypatch.setenv("OPENLAB_BOX_DIR", str(tmp_path))
    monkeypatch.setenv("OPENLAB_BOX_SLOTS", "1")
    (tmp_path / "slots").mkdir()
    holder = subprocess.Popen([sys.executable, "-c", "import fcntl, sys, time; f = open(sys.argv[1], 'a+'); "
                               "fcntl.flock(f, fcntl.LOCK_EX); print('held', flush=True); time.sleep(0.6)",
                               str(tmp_path / "slots" / "0.lock")], stdout=subprocess.PIPE, text=True)
    assert holder.stdout.readline().strip() == "held"
    t0 = time.monotonic()
    with machine.slot(wait=0.02):
        waited = time.monotonic() - t0
    holder.wait()
    assert waited > 0.3                                     # it waited for the other process to let go
    with open(tmp_path / "slots" / "0.lock", "a+") as f:    # and released it after the block
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_zero_slots_turns_the_gate_off(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENLAB_BOX_DIR", str(tmp_path))
    monkeypatch.setenv("OPENLAB_BOX_SLOTS", "0")
    with machine.slot() as s:
        assert s is None
    assert not (tmp_path / "slots").exists()
