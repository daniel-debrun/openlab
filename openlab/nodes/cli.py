"""`openlab node`: the nodes openlab places workers on, their allocations, and commands run inside one.

    openlab node list                 the nodes, what each offers and has free, and the live allocations
    openlab node probe [NAME]         read the nodes again (cores and memory under their limits, GPUs)
    openlab node allocations [--all]  the allocations, the ended ones too with --all
    openlab node exec -- CMD ...      run CMD inside your allocation (a worker's: from OPENLAB_ALLOCATION)
    openlab node runs | logs RUN | wait RUN | stop RUN | pull [RUN]
    openlab node release ALLOC        end an allocation (the person's)

An agent reaches only the allocation it holds; the person names one with --allocation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from . import alloc as A
from . import run as R
from .inventory import NodeError, nodes, offer, probe


def _at(ts) -> str:
    return datetime.fromtimestamp(ts).strftime("%m-%d %H:%M") if ts else "-"


def _err(e: NodeError) -> int:
    print(f"openlab node: {e.error}" + (f"\n  {e.meaning}" if e.meaning else "") + (f"\n  try: {e.try_}" if e.try_
                                                                                     else ""), file=sys.stderr)
    return 2


def _caller(a) -> tuple:
    """(agent, workspace, allocation row) of the caller: an agent's own allocation (OPENLAB_ALLOCATION, which it must
    hold), or the person's --allocation."""
    agent, ws = os.environ.get("OPENLAB_AGENT", ""), os.environ.get("OPENLAB_WORKSPACE", "")
    aid = getattr(a, "allocation", "") or os.environ.get("OPENLAB_ALLOCATION", "")
    if not aid:
        raise NodeError("no allocation", "a worker spawned with resources holds one (OPENLAB_ALLOCATION)",
                        "ask your parent for a worker with resources, or (the person) pass --allocation ID")
    if agent and not A.holds(agent, aid):
        raise NodeError(f"{agent} does not hold allocation {aid}", "an agent uses only its own allocation")
    row = A.get_alloc(aid)
    if row is None:
        raise NodeError(f"no allocation {aid}")
    return agent, ws, row


def _run_of(a, rid: str) -> dict:
    agent, _, row = _caller(a)
    r = R.get_run(rid)
    if r is None or (agent and r["alloc"] != row["id"]):
        raise NodeError(f"no run {rid} in allocation {row['id']}", "", "`openlab node runs` lists them")
    return r


def render_nodes() -> str:
    held = A.rows()
    out = []
    for n in nodes():
        try:
            d = probe(n)
            o = offer(n, d)
        except NodeError as e:
            out.append(f"{n.name:<12} {n.backend:<6} unreachable: {e.error}")
            continue
        if n.backend == "slurm":
            parts = ", ".join(f"{p['name']} ({p['cores']}c {p['memory_gb']:g}GB {p['gpus']} {p['gpu_model'] or 'gpu'}"
                              f", {p['minutes'] // 60 if p['minutes'] else '∞'}h)" for p in o["partitions"][:12])
            out.append(f"{n.name:<12} slurm  via {n.host}, account {n.account or '-'}: {parts}"
                       + (" ..." if len(o["partitions"]) > 12 else ""))
        else:
            f = A._free(n, o, held)
            gpus = ", ".join(f"{g['index']}:{g['name'].replace('NVIDIA ', '').replace('GeForce ', '')}"
                             for g in o["gpus"]) or "no GPU"
            sandbox = "" if d.get("bwrap") else ("  unsandboxed (the person's OK)" if n.unsandboxed else
                                                "  unsandboxed: needs the person's OK (unsandboxed = \"ok\")")
            out.append(f"{n.name:<12} {n.backend:<6} {d.get('host', '')[:16]:<16} {o['cores']:>3} cores "
                       f"{o['memory_gb']:>6g} GB  {gpus}   free: {len(f['cpus'])} cores, {f['memory_gb']:g} GB, "
                       f"{len(f['gpus'])} GPU{sandbox}")
    out.append("")
    out.append(render_allocations(held) if held else "no live allocations")
    return "\n".join(out)


def render_allocations(rows: list) -> str:
    out = [f"{'id':<10} {'state':<9} {'agent':<22} {'holds':<34} {'since':<11} why"]
    for r in rows:
        holds = f"{r['node'] or '?'}: {r['cores']}c {r['memory_gb']:g}GB" + (f" {r['gpus']}gpu" if r["gpus"] else "")
        out.append(f"{r['id']:<10} {r['state']:<9} {r['agent'][:22]:<22} {holds[:34]:<34} "
                   f"{_at(r['started'] or r['at']):<11} {(r['why'] or '')[:90]}")
    return "\n".join(out)


def main(argv) -> int:
    p = argparse.ArgumentParser(prog="openlab node", description="The nodes openlab places workers on, their "
                                "allocations, and commands run inside one.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="the nodes, what each offers and has free, and the live allocations") \
        .add_argument("--json", action="store_true")
    pr = sub.add_parser("probe", help="read the nodes again")
    pr.add_argument("name", nargs="?")
    al = sub.add_parser("allocations", help="the allocations")
    al.add_argument("--all", action="store_true")
    al.add_argument("--json", action="store_true")
    ex = sub.add_parser("exec", help="run a command inside your allocation: `openlab node exec -- CMD ...`")
    ex.add_argument("--detach", action="store_true", help="start it and return at once")
    ex.add_argument("--follow", type=float, default=R.FOLLOW, help="seconds to follow its output (default 90)")
    ex.add_argument("--no-sync", action="store_true", help="do not copy this folder to the node first")
    ex.add_argument("--no-pull", action="store_true", help="do not copy results back")
    ex.add_argument("--pull-max-mb", type=float, default=R.PULL_MAX_MB)
    ex.add_argument("--allocation", default="")
    ex.add_argument("command", nargs=argparse.REMAINDER)
    for name, h in (("runs", "the runs in your allocation"), ("logs", "a run's output"),
                    ("wait", "follow a run until it ends, then copy its results back"), ("stop", "stop a run"),
                    ("pull", "copy results back again (a run's, or your folder's on the node)")):
        sp = sub.add_parser(name, help=h)
        sp.add_argument("--allocation", default="")
        if name in ("logs", "wait", "stop"):
            sp.add_argument("run")
        if name == "pull":
            sp.add_argument("run", nargs="?")
            sp.add_argument("--max-mb", type=float, default=R.PULL_MAX_MB)
        if name == "logs":
            sp.add_argument("--bytes", type=int, default=20000, help="the last N bytes (default 20000)")
        if name == "wait":
            sp.add_argument("--timeout", type=float, default=540.0)
        if name == "runs":
            sp.add_argument("--json", action="store_true")
    rl = sub.add_parser("release", help="end an allocation (the person's)")
    rl.add_argument("allocation")
    a = p.parse_args(argv)
    try:
        return _main(a)
    except NodeError as e:
        return _err(e)


def _main(a) -> int:
    if a.cmd == "list":
        if a.json:
            out = []
            for n in nodes():
                try:
                    out.append({"name": n.name, "backend": n.backend, "host": n.host, "offer": offer(n)})
                except NodeError as e:
                    out.append({"name": n.name, "backend": n.backend, "error": e.error})
            print(json.dumps({"nodes": out, "allocations": A.rows()}, indent=1, default=str))
        else:
            print(render_nodes())
        return 0
    if a.cmd == "probe":
        for n in nodes():
            if a.name and n.name != a.name:
                continue
            try:
                d = probe(n, fresh=0)
                print(f"{n.name}: " + json.dumps({k: v for k, v in d.items() if k not in ("probed_at",)}))
            except NodeError as e:
                print(f"{n.name}: unreachable: {e.error}")
        return 0
    if a.cmd == "allocations":
        rows = A.rows(all=a.all)
        print(json.dumps(rows, indent=1, default=str) if a.json else render_allocations(rows) if rows else
              "no allocations")
        return 0
    if a.cmd == "release":
        if os.environ.get("OPENLAB_AGENT"):
            raise NodeError("releasing an allocation is the person's", "a worker's allocation ends when it does")
        r = A.release(a.allocation, "released by the person")
        print(f"{a.allocation}: {r['state'] if r else 'no such allocation'}")
        return 0
    agent, ws, row = _caller(a)
    if a.cmd == "exec":
        cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
        if not cmd:
            raise NodeError("nothing to run", "", "openlab node exec -- python train.py --epochs 3")
        line = cmd[0] if len(cmd) == 1 else " ".join(_quote(c) for c in cmd)
        r = R.start(row, agent, ws, Path.cwd(), line, sync=not a.no_sync)
        node = row["node"] + (f" ({row['host']})" if row["backend"] == "slurm" else "")
        print(f"run {r['id']} on {node}" + (f", folder copied in {r['pushed']['seconds']} s" if r.get("pushed") else ""),
              flush=True)
        if a.detach:
            print(f"started; `openlab node wait {r['id']}` follows it, `openlab node logs {r['id']}` shows its output")
            return 0
        code, _ = R.follow(r, a.follow)
        if code is None:
            print(f"\n[still running after {a.follow:.0f} s: run {r['id']}; `openlab node wait {r['id']} --timeout 540` "
                  f"follows it on (give the call a 600000 ms timeout), `openlab node stop {r['id']}` stops it]")
            return 0
        return _ended(r, code, not a.no_pull, a.pull_max_mb)
    if a.cmd == "runs":
        rows = [x for x in R.runs(row["id"])]
        for x in rows:
            if x["state"] == "running":
                code, _ = R.peek(x, 1 << 62)
                if code is not None:
                    x.update(R.finish(x, code))
        if a.json:
            print(json.dumps(rows, indent=1, default=str))
        else:
            for x in rows:
                span = (x["ended"] or time.time()) - x["at"]
                print(f"{x['id']}  {x['state']:<8} exit {x['exit'] if x['exit'] is not None else '-':>4}  "
                      f"{span / 60:6.1f} min  {x['cmd'][:100]}")
            if not rows:
                print("no runs in this allocation")
        return 0
    if a.cmd == "logs":
        r = _run_of(a, a.run)
        code, _ = R.peek(r, 0)
        _, body = _tail(r, a.bytes)
        sys.stdout.write(body.decode(errors="replace"))
        print(f"\n[run {r['id']}: " + (f"exit {code}]" if code is not None else "running]"))
        return 0
    if a.cmd == "wait":
        r = _run_of(a, a.run)
        code, _ = R.follow(r, a.timeout)
        if code is None:
            print(f"\n[run {r['id']} still running after {a.timeout:.0f} s]")
            return 0
        return _ended(r, code, True, R.PULL_MAX_MB)
    if a.cmd == "stop":
        r = _run_of(a, a.run)
        R.stop(r)
        code, _ = R.peek(r, 1 << 62)
        R.finish(r, code if code is not None else 143, do_pull=False)
        print(f"run {r['id']} stopped")
        return 0
    if a.cmd == "pull":
        if a.run:
            r = _run_of(a, a.run)
            got = R.pull(*_node_dirs(row, Path(r["cwd"])), a.max_mb)
        else:
            got = R.pull(*_node_dirs(row, Path.cwd()), a.max_mb)
        _report(got)
        return 0
    return 2


def _node_dirs(row: dict, cwd: Path) -> tuple:
    n, wdir, _ = R.where(row, cwd)
    return n, wdir, cwd


def _tail(r: dict, n: int) -> tuple:
    from .inventory import get
    import shlex
    import subprocess
    a = A.get_alloc(r["alloc"])
    node = get(a["node"])
    script = f"tail -c {int(n)} {shlex.quote(r['rdir'])}/{r['id']}.log 2>/dev/null"
    argv = ["bash", "-c", script] if node.backend == "local" else node.ssh() + ["bash -c " + shlex.quote(script)]
    p = subprocess.run(argv, capture_output=True, timeout=60, stdin=subprocess.DEVNULL)
    return None, p.stdout


def _ended(r: dict, code: int, do_pull: bool, max_mb: float) -> int:
    done = R.finish(r, code, do_pull=do_pull, max_mb=max_mb)
    print(f"\n[run {r['id']} ended: exit {code}]")
    if do_pull:
        _report(done)
    return code if 0 <= code < 256 else 1


def _report(got: dict) -> None:
    files = got.get("files") or []
    print(f"{len(files)} file(s) copied back" + (": " + ", ".join(files[:12]) + (" ..." if len(files) > 12 else "")
                                                if files else ""))
    if got.get("skipped"):
        print("left on the node (over the size cap): " + ", ".join(got["skipped"]))
    if got.get("error"):
        print(f"copying back failed: {got['error']}")


def _quote(s: str) -> str:
    import shlex
    return shlex.quote(s)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
