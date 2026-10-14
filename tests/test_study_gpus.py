"""Guarantees of GPUs in runs: `[sandbox] gpus = N` leases N specific GPUs for the run's whole length (a lock file
each, shared by every openlab process on the machine), refused at once when the machine has fewer; the run's children
see only the GPUs it leased (CUDA_VISIBLE_DEVICES), a run that leased none sees none, and the sandbox binds the NVIDIA
devices only for a run that leased some; the run records the GPUs it held."""

import shutil
import subprocess
import threading
import time

import pytest

from openlab import study as vs
from openlab.study import machine, sandbox
from openlab.study.errors import StudyError

TWO = [{"index": 0, "name": "RTX 3090", "memory_mb": 24576}, {"index": 1, "name": "RTX 3090", "memory_mb": 24576}]


@pytest.fixture
def box(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENLAB_BOX_DIR", str(tmp_path / "box"))
    monkeypatch.setattr(machine, "gpus", lambda: TWO)


def test_two_runs_lease_different_gpus_and_a_third_waits(box):
    got, release = [], threading.Event()

    def hold():
        with machine.lease_gpus(1, wait=0.01) as g:
            got.append(g)
            release.wait(5)

    a, b = threading.Thread(target=hold), threading.Thread(target=hold)
    a.start(), b.start()
    for _ in range(200):
        if len(got) == 2:
            break
        time.sleep(0.01)
    assert sorted(g[0] for g in got) == [0, 1]
    with pytest.raises(StudyError, match="came free"):          # both held: a third waits, here until its timeout
        with machine.lease_gpus(1, wait=0.01, timeout=0.2):
            pass
    release.set()
    a.join(), b.join()
    with machine.lease_gpus(2, wait=0.01) as both:                # released: both are free again
        assert both == [0, 1]


def test_asking_for_more_gpus_than_the_machine_has_is_refused_at_once(box):
    with pytest.raises(StudyError, match="asks for 3 GPU"):
        with machine.lease_gpus(3):
            pass
    with machine.lease_gpus(0) as none:
        assert none == []


def test_the_gpus_listed_respect_the_parents_cuda_visible_devices(monkeypatch):
    out = "0, NVIDIA GeForce RTX 3090, 24576\n1, NVIDIA GeForce RTX 3090, 24576\n"
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, out, ""))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert [g["index"] for g in machine.gpus()] == [0, 1] and machine.gpus()[0]["memory_mb"] == 24576
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert [g["index"] for g in machine.gpus()] == [1]


def test_a_child_sees_only_its_runs_gpus_and_the_sandbox_binds_the_devices_only_then(tmp_path, monkeypatch):
    assert sandbox._env()["CUDA_VISIBLE_DEVICES"] == ""            # leased none: sees none
    assert sandbox._env(1, [1])["CUDA_VISIBLE_DEVICES"] == "1"
    dev = tmp_path / "dev"
    dev.mkdir()
    for f in ("nvidia0", "nvidia1", "nvidiactl", "nvidia-uvm", "null"):
        (dev / f).touch()
    monkeypatch.setattr(sandbox, "DEV", dev)
    with_gpu = sandbox.bwrap_args(tmp_path, tmp_path / "stage", gpus=[1])
    binds = [with_gpu[i + 1] for i, x in enumerate(with_gpu) if x == "--dev-bind"]
    assert binds == sorted(str(dev / f) for f in ("nvidia0", "nvidia1", "nvidiactl", "nvidia-uvm"))
    assert "--dev-bind" not in sandbox.bwrap_args(tmp_path, tmp_path / "stage")


def gpu_study(tmp_path, n):
    s = vs.new(tmp_path / "g", "temporal")
    toml = s.folder / "study.toml"
    toml.write_text(toml.read_text().replace("[sandbox]\n", f"[sandbox]\ngpus = {n}\n", 1) + "\n[audit]\nn = 1\n")
    return s


def test_a_run_asking_for_gpus_the_machine_lacks_is_refused_before_it_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENLAB_BOX_DIR", str(tmp_path / "box"))
    monkeypatch.setattr(machine, "gpus", lambda: [])
    s = gpu_study(tmp_path, 1)
    with pytest.raises(StudyError, match="needs 1 GPU; the machine has 0"):    # admission refuses it first
        s.run(note="needs a GPU")
    assert not s.record.find("run")


@pytest.mark.skipif(not shutil.which("bwrap"), reason="runs are tested under bwrap")
def test_a_run_records_the_gpus_it_held_and_lets_them_go(tmp_path, box):          # ~10 s
    s = gpu_study(tmp_path, 1)
    out = s.run(note="on a GPU")
    run = s.record.get(s.record.links(out["id"], rel="of")[0][2])
    assert run.payload["box"]["gpus"] in ([0], [1])
    with machine.lease_gpus(2, wait=0.01, timeout=1) as both:     # the run let its GPU go
        assert both == [0, 1]


@pytest.mark.skipif(not shutil.which("bwrap"), reason="runs are tested under bwrap")
def test_a_run_waits_while_another_holds_the_only_gpu_and_records_its_wait(tmp_path, monkeypatch):       # ~10 s
    from openlab.study import admission
    monkeypatch.setenv("OPENLAB_BOX_DIR", str(tmp_path / "box"))
    monkeypatch.setattr(machine, "gpus", lambda: TWO[:1])
    s = gpu_study(tmp_path / "a", 1)
    holding, done = threading.Event(), threading.Event()

    def other_run():                                    # another run holds the machine's only GPU for 3 s
        with admission.admit(admission.Needs(gpus=1, minutes=0.05), label="other", wait=0.05):
            holding.set()
            time.sleep(3)
        done.set()

    threading.Thread(target=other_run).start()
    assert holding.wait(10)
    out = s.run(note="waits its turn")
    assert done.is_set()
    run = s.record.get(s.record.links(out["id"], rel="of")[0][2]).payload
    assert run["box"]["gpus"] == [0] and run["box"]["waited_s"] > 1
    assert [t["state"] for t in admission.tickets(all=True)] == ["done", "done"]
