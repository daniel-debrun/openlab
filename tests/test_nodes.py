"""Worker allocations (openlab.nodes): a worker spawned with resources holds cores, memory and whole GPUs on one node
of the person's inventory for its life. Placement packs GPU work, keeps CPU work off GPU nodes, honours a pin and a
GPU model, queues oldest first with the reason, sends what our nodes cannot start now to Slurm, and refuses what no
node could ever hold. `openlab node exec` runs a command inside the allocation on its node and copies results back;
the record, keys and person files never go. The supervisor waits for the allocation, runs the agent inside it and
lets go at its end; a child shares its parent's. A spawned agent's direct ssh or Slurm command is refused.

The "remote" nodes here are this machine behind a fake ssh (OPENLAB_NODE_SSH) and fake Slurm commands."""

import io
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agents_fake import final, install
from openlab.agents import hooks
from openlab.agents.spawn import AgentError, spawn
from openlab.agents.store import Store
from openlab.nodes import alloc as A
from openlab.nodes import cli as C
from openlab.nodes import run as R
from openlab.nodes.inventory import Node, NodeError, nodes, offer, parse_sinfo

FAKE_SSH = r'''#!{python}
import os, sys
args, i = sys.argv[1:], 0
while i < len(args) and args[i].startswith("-"):
    i += 2 if args[i] in ("-o", "-p", "-i", "-l", "-F", "-J", "-L", "-R", "-D") else 1
host, rest = args[i], args[i + 1:]
env = dict(os.environ)
if os.environ.get("FAKE_SLURM_BIN"):
    env["PATH"] = os.environ["FAKE_SLURM_BIN"] + os.pathsep + env["PATH"]
os.execvpe("bash", ["bash", "-c", " ".join(rest)], env)
'''

SBATCH = r'''#!/bin/bash
echo "$@" >> "$FAKE_SLURM/sbatch.args"
echo 4242
'''
SQUEUE = r'''#!/bin/bash
cat "$FAKE_SLURM/state" 2>/dev/null
'''
SCANCEL = r'''#!/bin/bash
echo "$@" >> "$FAKE_SLURM/scancel.args"
'''
SRUN = r'''#!/bin/bash
echo "$@" >> "$FAKE_SLURM/srun.args"
while [ $# -gt 0 ]; do
  case "$1" in
    --chdir=*) cd "${1#--chdir=}"; shift ;;
    --*) shift ;;
    *) break ;;
  esac
done
export CUDA_VISIBLE_DEVICES=0,1
exec "$@"
'''

SINFO = "\n".join([
    "gpu_short*|3:00:00|64|515000|gpu:l40s:4(S:0-1)|up",
    "gpu_long|7-00:00:00|64|515000|gpu:l40s:4(S:0-1)|up",
    "cpu_short|3:00:00|64|128000|(null)|up",
    "gpu_bynode|3:00:00|64|515000|gpu:l40s:4(S:0-1)|up",
])


def _exe(path: Path, text: str) -> Path:
    path.write_text(text.replace("{python}", sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture
def box(tmp_path, monkeypatch):
    """A person's box dir with two fake ssh nodes (one with two fake GPUs) and a fake Slurm node."""
    d = tmp_path / "box"
    d.mkdir()
    monkeypatch.setenv("OPENLAB_BOX_DIR", str(d))
    monkeypatch.delenv("OPENLAB_NODES_HOME", raising=False)
    monkeypatch.delenv("OPENLAB_BOX_CAPACITY", raising=False)
    monkeypatch.delenv("OPENLAB_ALLOCATION", raising=False)
    monkeypatch.setenv("OPENLAB_NODE_SSH", str(_exe(tmp_path / "fake-ssh", FAKE_SSH)))
    slurm = tmp_path / "slurm"
    (slurm / "bin").mkdir(parents=True)
    for name, text in (("sbatch", SBATCH), ("squeue", SQUEUE), ("scancel", SCANCEL), ("srun", SRUN)):
        _exe(slurm / "bin" / name, text)
    monkeypatch.setenv("FAKE_SLURM", str(slurm))
    monkeypatch.setenv("FAKE_SLURM_BIN", str(slurm / "bin"))
    remote = tmp_path / "remote"
    (d / "box.toml").write_text(f"""
[box]
keep_cores = 0

[[node]]
name = "gpubox"
host = "root@gpubox"
port = 2222
workdir = "{remote}/gpubox"
unsandboxed = "ok"

[[node]]
name = "cpubox"
host = "cpubox"
workdir = "{remote}/cpubox"
unsandboxed = "ok"

[[node]]
name = "cluster"
backend = "slurm"
host = "login"
account = "lab-acct"
workdir = "{remote}/cluster"
unsandboxed = "ok"
partitions = ["gpu_short", "gpu_long", "cpu_short"]
""")
    now = time.time()
    (d / "nodes.json").write_text(json.dumps({
        "local": {"backend": "local", "probed_at": now, "host": "here", "cpus": [0, 1], "cores": 2,
                  "memory_gb": 8.0, "gpus": [], "bwrap": True},
        "gpubox": {"backend": "ssh", "probed_at": now, "host": "gpubox", "cpus": [0, 1, 2, 3], "cores": 4,
                   "memory_gb": 32.0, "bwrap": False,
                   "gpus": [{"index": 0, "name": "NVIDIA GeForce RTX 3060", "memory_mb": 12288},
                            {"index": 1, "name": "NVIDIA GeForce RTX 3060", "memory_mb": 12288}]},
        "cpubox": {"backend": "ssh", "probed_at": now, "host": "cpubox", "cpus": list(range(16)), "cores": 16,
                   "memory_gb": 64.0, "gpus": [], "bwrap": False},
        "cluster": {"backend": "slurm", "probed_at": now, "partitions": parse_sinfo(SINFO)}}))
    return d


def req(**kw):
    r = {"cores": 1, "memory_gb": 0.0, "gpus": 0, "gpu_model": "", "minutes": 60.0, "pin": ""}
    r.update(kw)
    return r


def sleeping(arg: str) -> bool:
    """Whether a `sleep ARG` process runs (its argv exactly, so no shell line that mentions it counts)."""
    for d in Path("/proc").iterdir():
        try:
            if (d / "cmdline").read_bytes().split(b"\0")[:2] == [b"sleep", arg.encode()]:
                return True
        except OSError:
            continue
    return False


def live_pid():
    """A process that outlives the test's placement calls, as a supervisor does."""
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    return p


# --- placement ---------------------------------------------------------------------------------------------------
def test_sinfo_and_squeue_are_read():
    parts = {p["name"]: p for p in parse_sinfo(SINFO)}
    assert parts["gpu_short"] == {"name": "gpu_short", "default": True, "minutes": 180, "cores": 64,
                                  "memory_gb": 502.9, "gpus": 4, "gpu_model": "l40s", "up": True}
    assert parts["gpu_long"]["minutes"] == 7 * 1440 and parts["cpu_short"]["gpus"] == 0
    q = A.parse_squeue("PENDING|Priority|(null)|2026-10-23T09:00:00\n")
    assert q["state"] == "PENDING" and q["reason"] == "Priority" and q["start"] is not None
    assert A.parse_squeue("") is None


def test_placement_packs_gpu_work_and_keeps_cpu_work_off_gpu_nodes(box):
    offs = A.offers()
    assert set(offs) == {"local", "gpubox", "cpubox", "cluster"}
    p = A.place(req(gpus=1, cores=2), offs, [])
    assert p["start"].name == "gpubox" and p["cpus"] == [0, 1] and p["gpus"][0]["index"] == 0
    held = [{"node": "gpubox", "state": "running", "cpu_list": [0, 1], "gpu_list": [0], "memory_gb": 8.0,
             "agent": "w1", "started": time.time(), "minutes": 30}]
    p2 = A.place(req(gpus=1, cores=2), offs, held)
    assert p2["start"].name == "gpubox" and p2["cpus"] == [2, 3] and p2["gpus"][0]["index"] == 1
    cpu = A.place(req(cores=4, memory_gb=16), offs, [])
    assert cpu["start"].name == "cpubox"                      # not the GPU box, and not this small machine
    small = A.place(req(cores=1), offs, [])
    assert small["start"].name == "local"                     # the tightest fit without GPUs
    assert A.place(req(gpus=1, gpu_model="rtx 3060"), offs, [])["start"].name == "gpubox"
    assert A.place(req(cores=2, pin="cpubox"), offs, [])["start"].name == "cpubox"


def test_what_our_nodes_cannot_start_goes_to_slurm_and_what_none_could_hold_is_refused(box):
    offs = A.offers()
    full = [{"node": "gpubox", "state": "running", "cpu_list": [0, 1], "gpu_list": [0, 1], "memory_gb": 8.0,
             "agent": "w1", "started": time.time(), "minutes": 30}]
    s = A.place(req(gpus=1), offs, full)
    assert s["slurm"].name == "cluster" and s["partition"]["name"] == "gpu_short"
    assert A.place(req(gpus=4, gpu_model="L40S", minutes=600), offs, [])["partition"]["name"] == "gpu_long"
    assert A.place(req(cores=8, memory_gb=100, pin="cluster"), offs, [])["partition"]["name"] == "cpu_short"
    no_slurm = {k: v for k, v in offs.items() if k != "cluster"}
    w = A.place(req(gpus=1), no_slurm, full)
    assert "waiting for 1 core, 0 GB, 1 GPU" in w["wait"] and "gpubox: GPUs held by w1" in w["wait"]
    assert "first holder's budget ends" in w["wait"]
    with pytest.raises(NodeError, match="no node could ever hold 1 core, 0 GB, 8 GPUs"):
        A.place(req(gpus=8), offs, [])
    with pytest.raises(NodeError, match="no node could ever hold .*A100"):
        A.place(req(gpus=1, gpu_model="A100"), offs, [])
    with pytest.raises(NodeError, match="no node 'nope'"):
        A.place(req(pin="nope"), offs, [])


def test_an_unsandboxed_node_is_offered_only_with_the_persons_ok(box):
    (box / "box.toml").write_text((box / "box.toml").read_text().replace('unsandboxed = "ok"\n', "", 1))
    assert "gpubox" not in A.offers() and "cpubox" in A.offers()
    with pytest.raises(NodeError, match="unknown keys"):
        (box / "box.toml").write_text('[[node]]\nname = "x"\nhost = "h"\ncolour = "red"\n')
        nodes()


# --- the lifecycle -----------------------------------------------------------------------------------------------
def test_requests_queue_oldest_first_and_the_last_holder_ends_an_allocation(box, tmp_path):
    procs = [live_pid() for _ in range(4)]
    try:
        r1 = A.request(tmp_path, "w1", req(gpus=1, pin="gpubox"), procs[0].pid)
        r2 = A.request(tmp_path, "w2", req(gpus=1, pin="gpubox"), procs[1].pid)
        assert (r1["state"], r1["gpu_list"]) == ("running", [0]) and (r2["state"], r2["gpu_list"]) == ("running", [1])
        r3 = A.request(tmp_path, "w3", req(gpus=1, pin="gpubox"), procs[2].pid)
        assert r3["state"] == "queued" and "held by w1, w2" in r3["why"]
        A.join(r1["id"], tmp_path, "w1-child", procs[3].pid)
        assert A.of_agent("w1-child")["id"] == r1["id"] and A.holds("w1-child", r1["id"])
        assert A.leave(r1["id"], "w1")["state"] == "running"      # its child still holds it
        assert A.step(r3["id"])["state"] == "queued"
        ended = A.leave(r1["id"], "w1-child")
        assert ended["state"] == "ended" and set(ended["usage"]) == {"hours", "gpu_hours", "core_hours"}
        r3 = A.step(r3["id"])
        assert (r3["state"], r3["gpu_list"]) == ("running", [0])
        procs[1].kill()
        procs[1].wait()
        assert [r["id"] for r in A.rows()] == [r3["id"]]           # w2's holder is gone: its allocation ended
        assert A.get_alloc(r2["id"])["why"] == "its holders' processes are gone"
    finally:
        for p in procs:
            p.kill()


def test_a_slurm_allocation_is_a_job_and_its_commands_are_steps(box, tmp_path):
    p = live_pid()
    state = Path(os.environ["FAKE_SLURM"]) / "state"
    try:
        state.write_text("PENDING|Priority|(null)|2026-10-23T09:00:00\n")
        a = A.request(tmp_path, "w1", req(gpus=2, gpu_model="L40S", memory_gb=40, cores=8, pin="cluster",
                                          minutes=90), p.pid)
        assert a["state"] == "starting" and a["ref"] == "4242" and "pending (Priority), expected to start" in a["why"]
        args = (Path(os.environ["FAKE_SLURM"]) / "sbatch.args").read_text()
        for want in ("--partition=gpu_short", "--time=105", "--cpus-per-task=8", "--mem=40G", "--gres=gpu:l40s:2",
                     "--account=lab-acct", "--job-name=openlab-" + a["id"], "sleep infinity"):
            assert want in args, (want, args)
        state.write_text("RUNNING|None|gpu-node-7|2026-10-22T09:00:00\n")
        a = A.step(a["id"])
        assert (a["state"], a["host"], a["gpu_list"], a["gpu_names"]) == ("running", "gpu-node-7", [0, 1],
                                                                           ["l40s", "l40s"])
        work = tmp_path / "job"
        work.mkdir()
        (work / "train.py").write_text("print('trained')\n")
        r = R.start(a, "w1", tmp_path / "ws", work, "python3 train.py && echo $CUDA_VISIBLE_DEVICES > gpus.txt")
        out = io.StringIO()
        code, _ = R.follow(r, 20, out=out)
        assert code == 0 and "trained" in out.getvalue()
        done = R.finish(r, code)
        assert "gpus.txt" in done["files"] and (work / "gpus.txt").read_text().strip() == "0,1"
        srun = (Path(os.environ["FAKE_SLURM"]) / "srun.args").read_text()
        assert "--jobid=4242 --overlap" in srun and "--cpus-per-task=8" in srun
        A.release(a["id"], "test")
        assert "4242" in (Path(os.environ["FAKE_SLURM"]) / "scancel.args").read_text()
        state.write_text("")
        gone = A.request(tmp_path, "w2", req(gpus=1, pin="cluster"), p.pid)
        assert gone["state"] == "failed" and "is gone" in gone["why"]
    finally:
        p.kill()


def test_exec_copies_the_folder_runs_inside_the_allocation_and_copies_results_back(box, tmp_path, capsys):
    p = live_pid()
    ws = tmp_path / "ws"
    ws.mkdir()
    try:
        a = A.request(ws, "w1", req(cores=2, gpus=1, pin="gpubox"), p.pid)
        work = ws / "job"
        (work / ".openlab").mkdir(parents=True)
        (work / ".openlab" / "record.db").write_text("the record")
        (work / ".env").write_text("SECRET=1")
        (work / "openlab.toml").write_text("[agents]")
        (work / "main.py").write_text("import os\nprint('gpus', os.environ['CUDA_VISIBLE_DEVICES'])\n"
                                      "open('result.json', 'w').write('{\"loss\": 0.1}')\n")
        r = R.start(a, "w1", ws, work, "python3 main.py; sleep 0.5; echo second line")
        remote = Path(r["wdir"])
        assert (remote / "main.py").exists()
        assert not (remote / ".openlab").exists() and not (remote / ".env").exists() \
            and not (remote / "openlab.toml").exists()
        out = io.StringIO()
        code, _ = R.follow(r, 20, out=out)
        assert code == 0 and "gpus 0" in out.getvalue() and "second line" in out.getvalue()
        done = R.finish(r, code)
        assert done["files"] == ["result.json"] and json.loads((work / "result.json").read_text()) == {"loss": 0.1}
        kinds = [e["kind"] for e in Store(ws).events("w1", n=20)]
        assert "node_run" in kinds and "node_run_end" in kinds
        t0 = time.monotonic()
        slow = R.start(a, "w1", ws, work, "sleep 31.25")
        assert time.monotonic() - t0 < 10                          # started detached
        code, _ = R.follow(slow, 1, out=io.StringIO())
        assert code is None
        assert sleeping("31.25")
        A.release(a["id"], "test")                                  # its runs are stopped with it
        assert R.get_run(slow["id"])["state"] == "stopped"
        assert not sleeping("31.25")
    finally:
        p.kill()


def test_the_cli_reaches_only_the_callers_allocation(box, tmp_path, monkeypatch, capsys):
    p = live_pid()
    ws = tmp_path / "ws"
    (ws / "job").mkdir(parents=True)
    try:
        a = A.request(ws, "w1", req(cores=1, pin="cpubox"), p.pid)
        monkeypatch.chdir(ws / "job")
        monkeypatch.setenv("OPENLAB_AGENT", "w1")
        monkeypatch.setenv("OPENLAB_WORKSPACE", str(ws))
        monkeypatch.setenv("OPENLAB_ALLOCATION", a["id"])
        assert C.main(["exec", "--", "echo", "hello from the node"]) == 0
        out = capsys.readouterr().out
        assert "on cpubox" in out and "hello from the node" in out and "ended: exit 0" in out
        assert C.main(["exec", "--", "sh", "-c", "exit 3"]) == 3
        assert C.main(["runs"]) == 0 and "exit    3" in capsys.readouterr().out
        assert C.main(["release", a["id"]]) == 2                    # the person's
        monkeypatch.setenv("OPENLAB_AGENT", "w2")
        assert C.main(["runs"]) == 2 and "w2 does not hold allocation" in capsys.readouterr().err
    finally:
        p.kill()


# --- spawn and the supervisor ------------------------------------------------------------------------------------
@pytest.fixture
def ws(tmp_path, monkeypatch, box):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "openlab.toml").write_text("[agents]\nbudget_usd = 1.0\n")
    install(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_ARGV", str(tmp_path / "argv.json"))
    monkeypatch.setenv("OPENLAB_ALLOC_POLL", "0.2")
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1]))
    return root


def test_spawn_records_the_request_refuses_the_impossible_and_a_child_shares(ws):
    lead = spawn(ws, "lead", role="lead", budget={"usd": 1.0, "minutes": 60}, start=False)
    with pytest.raises(AgentError, match="no node could ever hold"):
        spawn(ws, "too big", parent=lead["id"], budget={"usd": 0.1, "minutes": 10}, resources={"gpus": 9},
              start=False)
    with pytest.raises(AgentError, match="unknown keys colour"):
        spawn(ws, "odd", parent=lead["id"], budget={"usd": 0.1, "minutes": 10}, resources={"colour": 1}, start=False)
    w = spawn(ws, "train", parent=lead["id"], budget={"usd": 0.3, "minutes": 30},
              resources={"gpus": 1, "cores": 2}, params={"seed": 3}, start=False)
    assert w["allocation"]["state"] == "running" and w["allocation"]["node"] == "gpubox"
    d = Store(ws).agent_dir(w["id"])
    launch = json.loads((d / "launch.json").read_text())
    assert launch["resources"]["gpus"] == 1 and launch["params"] == {"seed": 3}
    assert '"seed": 3' in (d / "brief.txt").read_text()
    kid = spawn(ws, "eval", parent=w["id"], budget={"usd": 0.1, "minutes": 10}, start=False)
    assert kid["allocation"]["id"] == w["allocation"]["id"]
    assert json.loads((Store(ws).agent_dir(kid["id"]) / "launch.json").read_text())["share"] == w["allocation"]["id"]
    plain = spawn(ws, "read", parent=lead["id"], budget={"usd": 0.1, "minutes": 10}, start=False)
    assert "allocation" not in plain


def test_the_supervisor_runs_a_worker_inside_its_local_allocation_and_lets_go_at_its_end(ws, tmp_path):
    a = spawn(ws, "work", resources={"cores": 1, "memory_gb": 1}, node="local")
    final(Store(ws), a["id"])
    seen = json.loads((tmp_path / "argv.json").read_text())
    alloc = A.get_alloc(a["allocation"]["id"])
    assert seen["allocation"] == alloc["id"] and seen["cuda"] == ""
    assert json.loads(seen["capacity"]) == {"cores": 1, "memory_gb": 1.0, "gpus": 0}
    assert seen["affinity"] == alloc["cpu_list"]
    brief = seen["argv"][seen["argv"].index("--append-system-prompt") + 1]
    assert "Your allocation: " + alloc["id"] + " on local" in brief
    assert alloc["state"] == "ended"
    kinds = [e["kind"] for e in Store(ws).events(a["id"], n=50)]
    assert {"allocation_request", "allocation_start", "allocation_end"} <= set(kinds)


def test_a_worker_waits_for_its_allocation_and_a_person_can_stop_it_meanwhile(ws, tmp_path):
    hold = live_pid()
    try:
        first = A.request(ws, "other", req(gpus=2, pin="gpubox"), hold.pid)
        assert first["state"] == "running"
        a = spawn(ws, "wait for a GPU", resources={"gpus": 1}, node="gpubox")
        assert a["allocation"]["state"] == "queued" and "held by other" in a["allocation"]["why"]
        deadline = time.time() + 15
        while time.time() < deadline and Store(ws).get(a["id"])["status"] != "waiting":
            time.sleep(0.1)
        assert Store(ws).get(a["id"])["status"] == "waiting"
        Store(ws).update(a["id"], control="stop", control_reason="not needed")
        final(Store(ws), a["id"])
        st = Store(ws).get(a["id"])
        assert st["status"] == "stopped" and A.get_alloc(a["allocation"]["id"])["state"] == "ended"
        ev = [e for e in Store(ws).events(a["id"], n=50) if e["kind"] == "end"][0]
        assert "no allocation" in ev["summary"] and "not needed" in ev["summary"]
    finally:
        hold.kill()


# --- the hook ----------------------------------------------------------------------------------------------------
SID = "5e55i0n0-aaaa-bbbb-cccc-0000000000aa"


def test_a_spawned_agent_cannot_reach_another_machine_or_a_scheduler_directly(ws):
    st = Store(ws)
    st.register("worker-abc123", "spawned", role="experimenter", cli="claude", session_id=SID, status="running",
                budget={"usd": 0.4})

    def bash(cmd):
        out = hooks.handle("PreToolUse", {"session_id": SID, "cwd": str(ws), "tool_name": "Bash",
                                          "tool_use_id": "t", "tool_input": {"command": cmd}})
        return (out or {}).get("hookSpecificOutput", {})

    for cmd in ("ssh vulcan nvidia-smi", "cd x && sbatch job.sh", "timeout 60 srun -n1 python a.py",
                "rsync -a out/ box:/tmp/out/", "scp model.pt root@1.2.3.4:"):
        got = bash(cmd)
        assert got.get("permissionDecision") == "deny", cmd
        assert "openlab node exec" in got["permissionDecisionReason"]
    for cmd in ("openlab node exec -- python train.py", "rsync -a out/ backup/", "grep sbatch README.md",
                "echo ssh"):
        assert bash(cmd).get("permissionDecision") != "deny", cmd
    assert any(e["kind"] == "refusal" and "ssh" in e["summary"] for e in st.events("worker-abc123", n=20))


def test_offer_cuts_to_the_entry_and_keeps_back_what_the_person_keeps():
    n = Node("n", "ssh", host="h", cores=3, memory_gb=10, gpus=0, keep_cores=1, keep_memory_gb=2)
    o = offer(n, {"cpus": [0, 1, 2, 3, 4], "memory_gb": 64, "gpus": [{"index": 0, "name": "G", "memory_mb": 1}]})
    assert o == {"cpus": [0, 1], "cores": 2, "memory_gb": 8.0, "gpus": []}
