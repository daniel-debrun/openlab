"""Admission: which run starts now on this machine, and which waits.

A run takes a ticket in the machine's queue (~/.openlab/box/queue.db, OPENLAB_BOX_DIR) with what it needs: cores,
memory, GPUs (the study's [sandbox] cores, memory_gb, gpus) and its expected minutes (the study's measured pace). It
starts when that fits beside the runs already running, first come first served, with one exception (backfill): a
younger ticket may start first when it fits now and, by the running runs' expected ends, it ends before the oldest
waiting ticket could start or leaves room for it then.

The machine's schedule is the person's, in box.toml beside the queue:

    [box]
    available_until = 2026-10-20T08:00:00   # optional: no run is admitted that would end after it
    keep_cores = 2                          # optional: held back for the person's own work
    keep_memory_gb = 8

    [[reservation]]                         # a window the person keeps resources in
    who = "sam"
    start = 2026-10-12T14:00:00
    end = 2026-10-12T18:00:00
    gpus = 1                                # and/or cores, memory_gb

A run whose window (now to now + its minutes) crosses a reservation counts the reserved resources as taken; a run of
unknown length counts every reservation that has not ended. A ticket whose process is gone is dropped, so a killed run
never blocks the queue. Within an admitted run, each sandboxed child still holds one of the machine's per-core slots
(openlab.study.machine.slot).
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:                                         # pragma: no cover -- python 3.10
    import tomli as tomllib

from . import machine
from .errors import StudyError


@dataclass(frozen=True)
class Needs:
    cores: int = 1
    memory_gb: float = 0.0
    gpus: int = 0
    minutes: float | None = None            # expected length; None when there is no measured pace yet


def needs_of(spec, rec=None) -> Needs:
    """A run's needs from its study: [sandbox] cores (default 1: the per-core slots share the rest), memory_gb,
    gpus, and the median minutes of the study's recorded non-cached runs."""
    sb = spec.sandbox or {}
    minutes = None
    if rec is not None:
        secs = sorted(r.payload["seconds"] for r in rec.find("run") if not r.payload.get("cached"))
        if secs:
            minutes = secs[len(secs) // 2] / 60
    return Needs(cores=max(1, int(sb.get("cores") or 1)), memory_gb=float(sb.get("memory_gb") or 0),
                 gpus=int(sb.get("gpus") or 0), minutes=minutes)


# --- the machine's schedule --------------------------------------------------------------------------------------
def _when(v) -> float | None:
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        return v.timestamp() if v.tzinfo else v.astimezone().timestamp()
    return datetime.fromisoformat(str(v)).astimezone().timestamp()


def schedule(box: Path | None = None) -> dict:
    """box.toml read: {available_until, keep_cores, keep_memory_gb, reservations: [{who, start, end, cores,
    memory_gb, gpus}]}, times as epoch seconds; empty when there is no box.toml."""
    path = (box or machine.box_dir()) / "box.toml"
    raw = tomllib.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    b = raw.get("box", {})
    res = [{"who": r.get("who", ""), "start": _when(r.get("start")), "end": _when(r.get("end")),
            "cores": int(r.get("cores", 0)), "memory_gb": float(r.get("memory_gb", 0)), "gpus": int(r.get("gpus", 0))}
           for r in raw.get("reservation", [])]
    return {"available_until": _when(b.get("available_until")), "keep_cores": int(b.get("keep_cores", 0)),
            "keep_memory_gb": float(b.get("keep_memory_gb", 0)), "reservations": res}


def capacity() -> dict:
    """What runs here may use: the machine (or, inside a worker's local allocation, its share), less what the live
    local allocations hold when this process is outside them (openlab.nodes.alloc)."""
    total = machine.memory()[1]
    cap = {"cores": machine.cores(), "memory_gb": total / 2 ** 30 if total else float("inf"),
           "gpus": len(machine.gpus())}
    if not machine.share():
        try:
            from ..nodes.alloc import local_held
            for r in local_held():
                cap = {"cores": max(1, cap["cores"] - (r["cores"] or 0)),
                       "memory_gb": max(0.0, cap["memory_gb"] - (r["memory_gb"] or 0)),
                       "gpus": max(0, cap["gpus"] - (r["gpus"] or 0))}
        except Exception:                                           # noqa: BLE001 -- no allocations: the machine
            pass
    return cap


# --- the queue ---------------------------------------------------------------------------------------------------
def _db() -> sqlite3.Connection:
    d = machine.box_dir()
    d.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(d / "queue.db", timeout=30, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE IF NOT EXISTS tickets (id INTEGER PRIMARY KEY, at REAL, label TEXT, agent TEXT,
                   cores INTEGER, memory_gb REAL, gpus INTEGER, minutes REAL, state TEXT, pid INTEGER, started REAL,
                   ended REAL, why TEXT)""")
    return con


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _expected_end(t, now: float) -> float:
    """When a running ticket is expected to end; an unknown length counts as never (it blocks to the end)."""
    return t["started"] + t["minutes"] * 60 if t["minutes"] is not None else float("inf")


def decide(owned: dict, waiting: list, running: list, cap: dict, sched: dict, now: float) -> tuple[bool, str]:
    """(start now?, why not). Pure: `owned` and the tickets are dicts with cores, memory_gb, gpus, minutes (and
    started for the running ones); `waiting` are the tickets older than `owned` still waiting, oldest first."""
    end = now + owned["minutes"] * 60 if owned["minutes"] is not None else None
    until = sched["available_until"]
    if until is not None and (end or now) > until:
        raise StudyError("this run would end after the machine stops being available" if end else
                         "the machine is past its declared availability",
                         f"box.toml says available until {datetime.fromtimestamp(until):%Y-%m-%d %H:%M}"
                         + (f"; this run is expected to take {owned['minutes']:.0f} min" if end else ""),
                         "ask the person for another machine (an inbox request of kind compute), or for a later "
                         "available_until")

    def free(at: float, then_end: float | None, extra=()) -> dict:
        f = {"cores": cap["cores"] - sched["keep_cores"], "memory_gb": cap["memory_gb"] - sched["keep_memory_gb"],
             "gpus": cap["gpus"]}
        for t in [*[t for t in running if at <= now or _expected_end(t, now) > at], *extra]:   # now: every one
            for k in f:
                f[k] -= t[k]
        for r in sched["reservations"]:
            if r["end"] is not None and r["end"] <= at:
                continue
            if then_end is not None and r["start"] is not None and r["start"] >= then_end:
                continue
            for k in f:
                f[k] -= r[k]
        return f

    def fits(t: dict, f: dict) -> bool:
        return all(t[k] <= f[k] + 1e-9 for k in ("cores", "memory_gb", "gpus"))

    for k, one, many in (("cores", "core", "cores"), ("memory_gb", "GB", "GB"), ("gpus", "GPU", "GPUs")):
        if owned[k] > (cap[k] - (sched["keep_cores"] if k == "cores" else sched["keep_memory_gb"] if
                                k == "memory_gb" else 0)) + 1e-9:
            raise StudyError(f"this run needs {owned[k]:g} {one if owned[k] == 1 else many}; the machine has "
                             f"{cap[k]:g} for runs",
                             "it would wait forever", "ask the person for a bigger machine (an inbox request of "
                             "kind compute), or lower [sandbox] " + k)
    now_free = free(now, end)
    if not fits(owned, now_free):
        short = [k for k in ("cores", "memory_gb", "gpus") if owned[k] > now_free[k] + 1e-9]
        return False, "waiting for " + ", ".join(short) + " held by running runs or reserved"
    if not waiting:
        return True, ""
    head = waiting[0]
    # the head's earliest start: the first running run's end after which it fits
    ends = sorted({_expected_end(t, now) for t in running} | {now})
    head_start = next((e for e in ends if e != float("inf") and fits(head, free(e, None))), float("inf"))
    if end is not None and end <= head_start:
        return True, ""                                             # ends before the head could start
    if head_start != float("inf") and fits(head, free(head_start, None, extra=[owned])):
        return True, ""                                             # leaves room for the head when it starts
    return False, f"waiting behind ticket {head['id']} (first come, first served)"


@contextmanager
def admit(needs: Needs, label: str = "", agent: str = "", wait: float = 2.0, on_wait=None, timeout: float | None = None):
    """Hold a place among the machine's running runs for the block. Yields {ticket, waited} once admitted;
    `on_wait(reason, seconds)` is called while it waits. OPENLAB_ADMISSION=off turns admission off."""
    if os.environ.get("OPENLAB_ADMISSION", "").lower() == "off":
        yield {"ticket": None, "waited": 0.0}
        return
    cap = capacity()
    with closing(_db()) as con:
        cur = con.execute("INSERT INTO tickets (at, label, agent, cores, memory_gb, gpus, minutes, state, pid) "
                          "VALUES (?,?,?,?,?,?,?,?,?)", (time.time(), label, agent, needs.cores, needs.memory_gb,
                                                         needs.gpus, needs.minutes, "waiting", os.getpid()))
        tid, t0 = cur.lastrowid, time.monotonic()
        try:
            while True:
                con.execute("BEGIN IMMEDIATE")
                try:
                    rows = [dict(r) for r in con.execute("SELECT * FROM tickets WHERE state IN ('waiting','running')")]
                    for r in rows:
                        if r["id"] != tid and not _alive(r["pid"]):
                            con.execute("UPDATE tickets SET state='gone', ended=? WHERE id=?", (time.time(), r["id"]))
                    rows = [r for r in rows if r["id"] == tid or _alive(r["pid"])]
                    owned = next(r for r in rows if r["id"] == tid)
                    waiting = sorted((r for r in rows if r["state"] == "waiting" and r["id"] < tid),
                                     key=lambda r: r["id"])
                    running = [r for r in rows if r["state"] == "running"]
                    ok, why = decide(owned, waiting, running, cap, schedule(), time.time())
                    if ok:
                        con.execute("UPDATE tickets SET state='running', started=?, why='' WHERE id=?",
                                    (time.time(), tid))
                    else:
                        con.execute("UPDATE tickets SET why=? WHERE id=?", (why, tid))
                    con.execute("COMMIT")
                except BaseException:
                    con.execute("ROLLBACK")
                    raise
                if ok:
                    break
                waited = time.monotonic() - t0
                if on_wait is not None:
                    on_wait(why, waited)
                if timeout is not None and waited > timeout:
                    raise StudyError(f"not admitted after {waited:.0f} s", why)
                time.sleep(wait)
            yield {"ticket": tid, "waited": time.monotonic() - t0}
            con.execute("UPDATE tickets SET state='done', ended=? WHERE id=?", (time.time(), tid))
        except BaseException:
            con.execute("UPDATE tickets SET state='gone', ended=? WHERE id=? AND state!='done'", (time.time(), tid))
            raise


def tickets(all: bool = False) -> list:
    """The machine's queue: running and waiting tickets (with `all`, the ended ones too), oldest first."""
    with closing(_db()) as con:
        q = "SELECT * FROM tickets" + ("" if all else " WHERE state IN ('waiting','running')") + " ORDER BY id"
        rows = [dict(r) for r in con.execute(q)]
    return [r for r in rows if all or _alive(r["pid"])]


def as_json(rows: list) -> str:
    return json.dumps(rows, indent=1)
