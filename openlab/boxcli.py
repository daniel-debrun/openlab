"""`openlab box`: this machine as openlab uses it (cores, memory, GPUs, the person's schedule in box.toml) and its
queue of runs (openlab.study.admission)."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime


def _at(ts) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "-"


def show() -> dict:
    from .study import admission, machine
    m, s = machine.describe(), admission.schedule()
    return {"box_dir": str(machine.box_dir()), **m, "gpus": machine.gpus(), "schedule": s,
            "queue": {k: sum(1 for t in admission.tickets() if t["state"] == k) for k in ("running", "waiting")}}


def render_show(d: dict) -> str:
    s = d["schedule"]
    lines = [f"cores      {d['cores']}" + (f"  (the host reports {d['host_cores']})" if d["host_cores"] != d["cores"] else ""),
             f"memory     {d['memory_gb']} GB" + (f"  (the host reports {d['host_memory_gb']} GB)"
                                                  if d["host_memory_gb"] != d["memory_gb"] else ""),
             "gpus       " + ("; ".join(f"{g['index']}: {g['name']} {g['memory_mb']} MiB" for g in d["gpus"]) or "none"),
             f"available  until {_at(s['available_until'])}" if s["available_until"] else "available  (no end declared)",
             f"kept back  {s['keep_cores']} cores, {s['keep_memory_gb']:g} GB" if s["keep_cores"] or s["keep_memory_gb"]
             else "kept back  nothing",
             *[f"reserved   {_at(r['start'])} to {_at(r['end'])} by {r['who'] or '?'}: "
               f"{r['cores']} cores, {r['memory_gb']:g} GB, {r['gpus']} GPU(s)" for r in s["reservations"]],
             f"queue      {d['queue']['running']} running, {d['queue']['waiting']} waiting  (openlab box queue)",
             f"schedule   {d['box_dir']}/box.toml"]
    return "\n".join(lines)


def render_queue(rows: list) -> str:
    if not rows:
        return "the queue is empty"
    now = time.time()
    out = [f"{'#':>4}  {'state':<8} {'for':>7}  {'needs':<26} {'run':<28} {'agent':<20} why"]
    for t in rows:
        since = (t["started"] or t["at"]) if t["state"] in ("running", "waiting") else (t["ended"] or t["at"])
        span = (now - since) / 60 if t["state"] in ("running", "waiting") else ((t["ended"] or now) - (t["started"] or t["at"])) / 60
        needs = f"{t['cores']}c {t['memory_gb']:g}GB {t['gpus']}gpu" + (f" ~{t['minutes']:.0f}m" if t["minutes"] else "")
        out.append(f"{t['id']:>4}  {t['state']:<8} {span:>6.1f}m  {needs:<26} {(t['label'] or '')[:28]:<28} "
                   f"{(t['agent'] or '')[:20]:<20} {t['why'] or ''}")
    return "\n".join(out)


def main(argv) -> int:
    p = argparse.ArgumentParser(prog="openlab box", description="This machine as openlab uses it, and its queue of runs.")
    sub = p.add_subparsers(dest="cmd")
    sh = sub.add_parser("show", help="cores, memory and GPUs openlab uses here, the schedule (box.toml), the queue")
    sh.add_argument("--json", action="store_true")
    q = sub.add_parser("queue", help="the runs running and waiting on this machine")
    q.add_argument("--all", action="store_true", help="the ended ones too")
    q.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    if a.cmd == "queue":
        from .study import admission
        rows = admission.tickets(all=a.all)
        print(json.dumps(rows, indent=1) if a.json else render_queue(rows))
        return 0
    d = show()
    print(json.dumps(d, indent=1) if getattr(a, "json", False) else render_show(d))
    return 0
