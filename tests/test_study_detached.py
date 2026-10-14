"""An agent's run goes to a process of its own and outlives the agent: its result reaches the record even when the
process that asked for it is gone."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from openlab.agents.store import Store
from openlab.app import ToolContext, dispatch

REPO = Path(__file__).resolve().parents[1]


def _study(tmp_path):
    (tmp_path / "openlab.toml").write_text("")
    ctx = ToolContext(str(tmp_path))
    assert dispatch(ctx, "study_new", {"folder": "s", "template": "static"})["ok"]
    assert dispatch(ctx, "study_materialise", {"folder": "s"})["ok"]
    return ctx


def test_an_agents_run_is_detached_and_returns_its_result(tmp_path, monkeypatch):
    ctx = _study(tmp_path)
    Store(tmp_path).register("lead-1", "spawned", role="lead", status="running")
    monkeypatch.setenv("OPENLAB_AGENT", "lead-1")
    monkeypatch.setattr("openlab.agents.identity.verified", lambda a, t: a)
    env = dispatch(ctx, "study_run", {"folder": "s"}, caller="agent")
    assert env["ok"], env
    assert env["result"]["trial"]["summary"].startswith("trial 1")
    runs = Store(tmp_path).events("lead-1", kinds=("run",))
    assert runs and runs[-1]["payload"]["pid"] != os.getpid()      # it ran in a process of its own
    assert not list((tmp_path / "s" / ".openlab" / "jobs").glob("*"))  # its job files are cleaned up
    bad = dispatch(ctx, "study_run", {"folder": "nope"}, caller="agent")
    assert not bad["ok"]


def test_the_run_finishes_and_is_recorded_after_its_caller_is_killed(tmp_path):
    _study(tmp_path)
    caller = subprocess.Popen([sys.executable, "-c", (
        "from openlab.app import ToolContext, dispatch\n"
        f"dispatch(ToolContext({str(tmp_path)!r}), 'study_run', {{'folder': 's'}}, caller='agent')")],
        env={**os.environ, "PYTHONPATH": str(REPO)}, start_new_session=True)
    jobs = tmp_path / "s" / ".openlab" / "jobs"
    end = time.time() + 30
    while time.time() < end and not list(jobs.glob("*.json")):
        time.sleep(0.1)
    assert list(jobs.glob("*.json")), "the job never started"
    os.killpg(caller.pid, 9)                                         # the agent's MCP server dies mid-run
    caller.wait()
    end, runs = time.time() + 120, []
    while time.time() < end and not runs:
        out = subprocess.run([sys.executable, "-m", "openlab", "study", "runs", "s", "--json"], cwd=tmp_path,
                             capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO)}).stdout
        runs = json.loads(out or "{}").get("runs") or []
        time.sleep(0.5)
    assert len(runs) == 1 and runs[0]["id"].startswith("run:")     # recorded although its caller is gone


def test_a_run_still_going_comes_back_as_a_job_that_agent_wait_collects(tmp_path, monkeypatch, capsys):
    from openlab.study import cli as study_cli
    ctx = _study(tmp_path)
    Store(tmp_path).register("lead-1", "spawned", role="lead", status="running", budget={"minutes": 60, "usd": 10})
    monkeypatch.setenv("OPENLAB_AGENT", "lead-1")
    monkeypatch.setattr("openlab.agents.identity.verified", lambda a, t: a)
    monkeypatch.setenv("OPENLAB_RUN_WAIT", "0.05")                   # the caller stops waiting at once
    env = dispatch(ctx, "study_run", {"folder": "s", "note": "slow one"}, caller="agent")
    assert env["ok"], env
    r = env["result"]
    assert r["running"] and r["job"].startswith("job:run-") and "agent_wait" in r["next"]
    live = dispatch(ctx, "study_runs", {"folder": "s"})["result"]["running"]  # in flight: listed, not "no runs"
    assert [x["job"] for x in live] == [r["job"]] and live[0]["by"] == "lead-1" and live[0]["note"] == "slow one"
    assert dispatch(ctx, "study_status", {"folder": "s"})["result"]["running"][0]["job"] == r["job"]
    assert study_cli.main(["runs", str(tmp_path / "s")]) == 0
    out = capsys.readouterr().out
    assert f"running {r['job']} by lead-1" in out and "no runs yet" not in out
    env = dispatch(ctx, "agent_wait", {"ids": [r["job"]], "timeout": 120}, caller="agent")
    assert env["ok"], env
    w = env["result"]
    assert w["done"] and w["runs"][0]["state"] == "done" and w["runs"][0]["trial"].startswith("trial 1")
    assert w["runs"][0]["result"].startswith("result:") and w["runs"][0]["metric"]["value"] is not None
    assert w["limits"].startswith("you: 0 of 60 min, $0.00 of $10")
    again = dispatch(ctx, "agent_wait", {"ids": r["job"].removeprefix("job:"), "timeout": 0})["result"]
    assert again["runs"][0]["result"] == w["runs"][0]["result"]     # a second wait says the same
    jobs = tmp_path / "s" / ".openlab" / "jobs"
    assert not list(jobs.glob("*.json")) and not list(jobs.glob("*.out")) and list(jobs.glob("*.done"))
    assert dispatch(ctx, "study_runs", {"folder": "s"})["result"]["running"] == []
    assert not dispatch(ctx, "agent_wait", {"ids": ["job:run-000000000000"], "timeout": 0})["ok"]


def test_a_run_that_died_without_a_result_is_a_failed_job_and_old_jobs_are_removed(tmp_path):
    from openlab.study import detached
    folder = tmp_path / "s"
    jobs = folder / ".openlab" / "jobs"
    jobs.mkdir(parents=True)
    (tmp_path / "openlab.toml").write_text("")
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()                                                     # a pid that has ended
    (jobs / "run-aaaaaaaaaaaa.json").write_text(json.dumps({"kwargs": {}, "actor": "x", "pid": gone.pid,
                                                            "started": "2026-01-01T00:00:00+00:00"}))
    (jobs / "run-aaaaaaaaaaaa.log").write_text("Traceback: killed\n")
    assert detached.live(folder) == []
    got = detached.poll(folder, "run-aaaaaaaaaaaa")
    assert got["state"] == "failed" and "without a result" in got["error"] and "killed" in got["meaning"]
    old = time.time() - 2 * detached.KEEP
    for p in jobs.iterdir():
        os.utime(p, (old, old))
    detached._prune(folder)
    assert not list(jobs.iterdir())


def test_materialise_warns_of_an_edited_recipe_and_refresh_reruns_it(tmp_path, capsys):
    from openlab.study import cli as study_cli
    import openlab.study as vs
    ctx = _study(tmp_path)
    ds = next(iter(vs.open(tmp_path / "s").spec.data.values()))
    recipe, data = tmp_path / "s" / ds.recipe, tmp_path / "s" / ds.file
    now = time.time()
    os.utime(data, (now - 120, now - 120))
    os.utime(recipe, (now - 60, now - 60))                          # the recipe edited after its file was made
    env = dispatch(ctx, "study_materialise", {"folder": "s"})
    assert env["ok"] and f"{ds.recipe} changed after {ds.file}" in env["result"]["warnings"][0]
    assert "refresh=true" in env["result"]["warnings"][0]
    assert study_cli.main(["materialise", str(tmp_path / "s")]) == 0
    assert "--refresh" in capsys.readouterr().out
    assert study_cli.main(["materialise", str(tmp_path / "s"), "--refresh"]) == 0
    assert "warning" not in capsys.readouterr().out                 # the recipe ran again: the file is newer
    os.utime(data, (now - 120, now - 120))
    env = dispatch(ctx, "study_materialise", {"folder": "s", "refresh": True})
    assert env["ok"] and "warnings" not in env["result"]
