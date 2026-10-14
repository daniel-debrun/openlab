"""Guarantees of admission (openlab.study.admission): a run starts only when its cores, memory and GPUs fit beside the
running runs and the person's reservations; first come, first served, except that a younger run that fits may go
first when it ends before the oldest waiting run could start or leaves room for it; no run is admitted that would end
after the machine's declared availability, or that needs more than the machine has; a ticket whose process is gone
never blocks the queue."""

import os
import subprocess
import sys
import threading
import time

import pytest

from openlab.study import admission as A
from openlab.study.errors import StudyError

CAP = {"cores": 23, "memory_gb": 120.0, "gpus": 2}
NOW = 1_000_000.0
EMPTY = {"available_until": None, "keep_cores": 0, "keep_memory_gb": 0.0, "reservations": []}


def t(id=0, cores=1, memory_gb=0.0, gpus=0, minutes=None, started=None):
    return {"id": id, "cores": cores, "memory_gb": memory_gb, "gpus": gpus, "minutes": minutes, "started": started}


def test_a_run_that_fits_starts_and_one_that_does_not_waits_for_what_it_lacks():
    running = [t(1, gpus=2, minutes=30, started=NOW - 60)]
    assert A.decide(t(2, gpus=0), [], running, CAP, EMPTY, NOW) == (True, "")
    ok, why = A.decide(t(2, gpus=1), [], running, CAP, EMPTY, NOW)
    assert not ok and "gpus" in why
    ok, why = A.decide(t(2, memory_gb=100), [], [t(1, memory_gb=30, started=NOW)], CAP, EMPTY, NOW)
    assert not ok and "memory_gb" in why


def test_a_running_run_past_its_expected_end_still_holds_what_it_holds():
    overran = [t(1, gpus=2, minutes=10, started=NOW - 3600)]
    assert not A.decide(t(2, gpus=1), [], overran, CAP, EMPTY, NOW)[0]


def test_first_come_first_served_with_backfill():
    running = [t(1, gpus=1, minutes=60, started=NOW)]                # one GPU free; ends at NOW + 1 h
    head = t(2, gpus=2, minutes=30)                                  # waits for both GPUs: starts at NOW + 1 h
    short = t(3, gpus=1, minutes=20)                                 # fits now, ends before the head could start
    long_ = t(4, gpus=1, minutes=120)                                # fits now, but would hold the head back
    assert A.decide(short, [head], running, CAP, EMPTY, NOW) == (True, "")
    ok, why = A.decide(long_, [head], running, CAP, EMPTY, NOW)
    assert not ok and "behind ticket 2" in why
    small = t(5, cores=2, minutes=None)                              # leaves the head's GPUs alone: room for it
    assert A.decide(small, [head], running, CAP, EMPTY, NOW) == (True, "")


def test_reservations_hold_their_resources_in_their_window():
    sched = {**EMPTY, "reservations": [{"who": "sam", "start": NOW + 3600, "end": NOW + 7200, "cores": 0,
                                        "memory_gb": 0, "gpus": 1}]}
    ok, _ = A.decide(t(1, gpus=2, minutes=30), [], [], CAP, sched, NOW)       # ends before the window
    assert ok
    ok, why = A.decide(t(1, gpus=2, minutes=90), [], [], CAP, sched, NOW)     # crosses it
    assert not ok and "gpus" in why
    assert not A.decide(t(1, gpus=2), [], [], CAP, sched, NOW)[0]             # unknown length: counts it
    assert A.decide(t(1, gpus=2), [], [], CAP, sched, NOW + 7300)[0]          # after it


def test_no_run_is_admitted_past_the_machines_availability_or_beyond_its_size():
    sched = {**EMPTY, "available_until": NOW + 3600}
    assert A.decide(t(1, minutes=30), [], [], CAP, sched, NOW)[0]
    with pytest.raises(StudyError, match="end after the machine stops being available"):
        A.decide(t(1, minutes=90), [], [], CAP, sched, NOW)
    with pytest.raises(StudyError, match="past its declared availability"):
        A.decide(t(1), [], [], CAP, sched, NOW + 4000)
    with pytest.raises(StudyError, match="needs 3 GPUs; the machine has 2"):
        A.decide(t(1, gpus=3), [], [], CAP, EMPTY, NOW)
    keep = {**EMPTY, "keep_cores": 3}
    with pytest.raises(StudyError, match="needs 21 cores; the machine has 23"):
        A.decide(t(1, cores=21), [], [], CAP, keep, NOW)


def test_the_schedule_reads_box_toml(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENLAB_BOX_DIR", str(tmp_path))
    (tmp_path / "box.toml").write_text('[box]\navailable_until = 2026-10-20T08:00:00\nkeep_cores = 2\n\n'
                                       '[[reservation]]\nwho = "sam"\nstart = 2026-10-12T14:00:00\n'
                                       'end = 2026-10-12T18:00:00\ngpus = 1\n')
    s = A.schedule()
    assert s["keep_cores"] == 2 and s["reservations"][0]["gpus"] == 1
    assert s["reservations"][0]["end"] - s["reservations"][0]["start"] == 4 * 3600
    assert s["available_until"] > s["reservations"][0]["end"]
    monkeypatch.setenv("OPENLAB_BOX_DIR", str(tmp_path / "none"))
    assert A.schedule() == EMPTY


@pytest.fixture
def box(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENLAB_BOX_DIR", str(tmp_path))
    monkeypatch.setattr(A, "capacity", lambda: {"cores": 4, "memory_gb": 16.0, "gpus": 1})
    return tmp_path


def test_admitted_tickets_run_and_a_second_gpu_run_waits_its_turn(box):
    order, first_in = [], threading.Event()

    def run(name, hold):
        with A.admit(A.Needs(gpus=1, minutes=1), label=name, wait=0.02):
            order.append(name)
            first_in.set()
            time.sleep(hold)

    a = threading.Thread(target=run, args=("a", 0.3))
    a.start()
    first_in.wait(5)
    waits = []
    with A.admit(A.Needs(gpus=1, minutes=1), label="b", wait=0.02, on_wait=lambda why, s: waits.append(why)) as adm:
        order.append("b")
        assert adm["waited"] > 0.1
    a.join()
    assert order == ["a", "b"] and waits and "gpus" in waits[0]
    assert [r["state"] for r in A.tickets(all=True)] == ["done", "done"]


def test_a_ticket_whose_process_is_gone_does_not_block(box):
    code = ("import os, sys; os.environ['OPENLAB_BOX_DIR'] = sys.argv[1]; from openlab.study import admission as A; "
            "A.capacity = lambda: {'cores': 4, 'memory_gb': 16.0, 'gpus': 1}; "
            "cm = A.admit(A.Needs(gpus=1)); cm.__enter__(); print('in', flush=True); os._exit(0)")
    p = subprocess.Popen([sys.executable, "-c", code, str(box)], stdout=subprocess.PIPE, text=True,
                         env={**os.environ, "PYTHONPATH": os.getcwd()})
    assert p.stdout.readline().strip() == "in"
    p.wait()                                                        # died holding the GPU, its ticket `running`
    with A.admit(A.Needs(gpus=1), wait=0.02, timeout=5) as adm:
        assert adm["ticket"]
    assert [r["state"] for r in A.tickets(all=True)] == ["gone", "done"]


def test_admission_off(box, monkeypatch):
    monkeypatch.setenv("OPENLAB_ADMISSION", "off")
    with A.admit(A.Needs(gpus=5)) as adm:
        assert adm["ticket"] is None


def test_openlab_box_shows_the_machine_its_schedule_and_its_queue(box, capsys):
    from openlab import boxcli
    (box / "box.toml").write_text('[box]\nkeep_cores = 1\n\n[[reservation]]\nwho = "sam"\nstart = 2026-10-12T14:00:00\n'
                                  'end = 2026-10-12T18:00:00\ngpus = 1\n')
    assert boxcli.main([]) == 0
    out = capsys.readouterr().out
    assert "kept back  1 cores" in out and "reserved   2026-10-12 14:00 to 2026-10-12 18:00 by sam" in out
    with A.admit(A.Needs(gpus=1, minutes=5), label="demo research", agent="lead-1"):
        assert boxcli.main(["queue"]) == 0
        line = capsys.readouterr().out.splitlines()[1]
        assert "running" in line and "1c 0GB 1gpu ~5m" in line and "demo research" in line and "lead-1" in line
    assert boxcli.main(["queue"]) == 0 and capsys.readouterr().out.strip() == "the queue is empty"
