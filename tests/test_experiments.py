"""Experiments (openlab.study.experiments): a study's pinned instruments run on a schedule for a set number of passes,
then the experiment ends and an agent is fired to conclude it.

The spec is checked strictly; the start is the person's signed act; a job runs only at a minute its schedule fires,
once; passes, failures and the stop rule are counted on the record; a kill removes the lines at once; the reaction
agent is spawned at the end. No test touches the real crontab (OPENLAB_CRONTAB names a fake) or spawns a real agent
(experiments.SPAWNER records the call)."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from openlab.study import experiments as E
from openlab.study import instruments as I
from openlab.study.errors import StudyError

REPO = Path(__file__).resolve().parents[1]
UTC = dt.timezone.utc
SCRIPT = """#!/bin/sh
case "$1" in fail) echo "boom" >&2; exit 3;; esac
n=$(cat "$OPENLAB_INSTRUMENT_STATE/n" 2>/dev/null || echo 0); n=$((n + 1)); echo $n > "$OPENLAB_INSTRUMENT_STATE/n"
echo "{\\"job\\": \\"$1\\", \\"n\\": $n}" > out.json
echo "ran $1 $n"
"""


@pytest.fixture(autouse=True)
def fake_crontab(tmp_path, monkeypatch):
    """The crontab is a file: `crontab -l` prints it, `crontab -` replaces it."""
    fake, tab = tmp_path / "crontab", tmp_path / "crontab.txt"
    fake.write_text(f'#!/bin/sh\nif [ "$1" = "-l" ]; then cat "{tab}" 2>/dev/null || exit 1; else cat > "{tab}"; fi\n')
    fake.chmod(0o755)
    monkeypatch.setenv("OPENLAB_CRONTAB", str(fake))
    tab.write_text("0 1 * * * echo the person's own line\n")
    return tab


@pytest.fixture(autouse=True)
def spawned(monkeypatch):
    calls = []

    def spawner(root, **kw):
        calls.append({"root": str(root), **kw})
        return {"id": f"reviewer-{len(calls):06x}"}
    monkeypatch.setattr(E, "SPAWNER", spawner)
    return calls


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "openlab.toml").write_text("# a test workspace\n")
    return root


def register(ws, person, name="tender", capabilities="read,paper-order", script=SCRIPT):
    src = ws / f"{name}_src"
    src.mkdir()
    (src / "run.sh").write_text(script)
    (src / "run.sh").chmod(0o755)
    with person.act():
        out = I.register(ws, name, "./run.sh", source=str(src), state=True, outputs="out.json",
                         capabilities=capabilities, reason="a fake instrument", usage=False)
    return I.ident(out["payload"])


def make_study(ws, pins: dict, folder="tender"):
    from openlab import study
    s = study.new(ws / folder, "temporal")
    with open(s.folder / "study.toml", "a") as f:
        f.write("\n[instruments]\n" + "".join(f'{k} = "{v}"\n' for k, v in pins.items()))
    return s.folder


SPEC = """[experiment]
name = "{name}"
purpose = "does the scan find tenders the export confirms; judged on the exported rows"
plan_step = "paper-exp"

[[jobs]]
name = "scan"
instrument = "tender"
args = ["scan"]
cron = "*/10 9-20 * * 1-5"
tz = "America/New_York"

[[jobs]]
name = "export"
instrument = "tender"
args = [{export}]
cron = "30 23 * * 1-5"
tz = "America/New_York"

[pass]
job = "{pass_job}"

[stop]
{stop}

[react]
role = "reviewer"
prompt = "Write the conclusion."
budget = {{ usd = 5, minutes = 45, turns = 80 }}
"""


def spec(name="paper_trial", pass_job="export", stop="passes = 3", export='"export"') -> str:
    return SPEC.format(name=name, pass_job=pass_job, stop=stop, export=export)


def save(folder: Path, text: str, name="paper_trial") -> Path:
    p = folder / "experiments" / f"{name}.toml"
    p.parent.mkdir(exist_ok=True)
    p.write_text(text)
    return p


# a weekday, Monday 2026-10-19: 13:30 UTC is 09:30 in New York (EDT), 03:30 UTC on the 20th is 23:30 on the 19th
def ny(day: int, hh: int, mm: int) -> dt.datetime:
    from zoneinfo import ZoneInfo
    return dt.datetime(2026, 10, day, hh, mm, 5, tzinfo=ZoneInfo("America/New_York")).astimezone(UTC)


@pytest.fixture
def lab(ws, person_key):
    pin = register(ws, person_key)
    folder = make_study(ws, {"tender": pin})
    return ws, folder, pin, person_key


def started(lab, text=None, name="paper_trial"):
    ws, folder, pin, person = lab
    save(folder, text or spec(name=name), name)
    from openlab.agents.identity import acting_as
    from openlab.agents.store import Store
    Store(ws).register("lead-1", "spawned", role="lead", status="running", budget={"usd": 5})
    with acting_as("lead-1"):
        req = E.request(ws, "tender", name, "the forward paper trial")
    with person.act():
        out = E.start(ws, "tender", name, "go", spec_sha=req["sha"])
    return req, out


# --- the spec --------------------------------------------------------------------------------------------------------
def test_a_valid_spec_parses_to_its_checked_form(lab):
    ws, folder, pin, _ = lab
    s = E.parse(spec(stop="passes = 20\nuntil = 2026-12-31\nmax_failures = 2"), folder, "paper_trial", ws)
    assert [j["name"] for j in s["jobs"]] == ["scan", "export"] and s["jobs"][0]["pin"] == pin
    assert s["pass"] == {"job": "export"} and s["stop"]["passes"] == 20 and s["stop"]["max_failures"] == 2
    assert s["stop"]["until"] == "2027-01-01T00:00:00-05:00"          # the end of Dec 31 in the jobs' zone
    assert s["react"] == {"role": "reviewer", "prompt": "Write the conclusion.",
                          "budget": {"usd": 5.0, "minutes": 45.0, "turns": 80}}
    assert E.parse(spec(stop="until = 2026-12-31"), folder, "paper_trial", ws)["stop"]["max_failures"] == 3


@pytest.mark.parametrize("edit, why", [
    (lambda t: t + "\n[extra]\nx = 1\n", "has no key extra"),
    (lambda t: t.replace('tz = "America/New_York"\n\n[[jobs]]', 'tz = "America/New_York"\nretries = 2\n\n[[jobs]]'),
     "has no key retries"),
    (lambda t: t.replace("turns = 80", "turns = 80, gpus = 1"), "budget has no key gpus"),
    (lambda t: t.replace('instrument = "tender"\nargs = ["scan"]', 'instrument = "other"\nargs = ["scan"]'),
     "does not pin `other`"),
    (lambda t: t.replace('"*/10 9-20 * * 1-5"', '"*/10 25 * * 1-5"'), "hour"),
    (lambda t: t.replace('"*/10 9-20 * * 1-5"', '"*/10 9-20 * *"'), "5 fields"),
    (lambda t: t.replace('tz = "America/New_York"', 'tz = "Mars/Olympus"', 1), "not a time zone"),
    (lambda t: t.replace("passes = 3", "max_failures = 2"), "needs passes or until"),
    (lambda t: t.replace('name = "paper_trial"', 'name = "other_name"'), "not 'paper_trial'"),
    (lambda t: t.replace('job = "export"', 'job = "nope"'), "is not one of the jobs"),
    (lambda t: t.replace('role = "reviewer"', 'role = "wizard"'), "not an openlab role"),
    (lambda t: t.replace("usd = 5, ", ""), "names usd and minutes"),
    (lambda t: t.replace('args = ["scan"]', 'args = "scan"'), "args is a list of strings"),
    (lambda t: t.replace("[pass]\n", "").replace('job = "export"\n', ""), r"no \[pass\]"),
    (lambda t: t.replace("purpose = ", "# purpose = "), "purpose"),
    (lambda t: t.replace("passes = 3", "passes = 0"), "at least 1"),
    (lambda t: t.replace("passes = 3", 'until = "2026-12-31"'), "a date"),
])
def test_the_spec_is_checked_strictly(lab, edit, why):
    ws, folder, _, _ = lab
    with pytest.raises(StudyError, match=why):
        E.parse(edit(spec()), folder, "paper_trial", ws)


def test_a_live_order_instrument_is_refused_in_an_experiment(ws, person_key):
    live = register(ws, person_key, name="broker", capabilities="read,live-order")
    folder = make_study(ws, {"tender": live})
    with pytest.raises(StudyError, match="can place live orders"):
        E.parse(spec(), folder, "paper_trial", ws)


def test_a_spec_given_as_a_table_is_written_as_toml(lab):
    ws, folder, _, _ = lab
    table = {"experiment": {"name": "t2", "purpose": "p"},
             "jobs": [{"name": "a", "instrument": "tender", "args": ["scan"], "cron": "0 17 * * *", "tz": "UTC"}],
             "pass": {"job": "a"}, "stop": {"passes": 2, "until": "2026-12-31"},
             "react": {"role": "reviewer", "budget": {"usd": 1, "minutes": 10}}}
    out = E.write(ws, "tender", "t2", E.to_toml(table))
    assert out["spec"]["stop"]["until"].startswith("2027-01-01T00:00:00+00:00")
    assert (folder / "experiments" / "t2.toml").is_file()


# --- the lifecycle and the person gate --------------------------------------------------------------------------------
def test_draft_requested_then_only_the_person_starts_it(lab, fake_crontab):
    ws, folder, pin, person = lab
    save(folder, spec())
    assert E.status(ws, "tender", "paper_trial")["state"] == "draft"
    from openlab.agents.identity import acting_as
    from openlab.agents.store import Store
    with acting_as("lead-1"):
        req = E.request(ws, "tender", "paper_trial", "the forward paper trial")
    st = Store(ws)
    item = st.item(req["item"])
    assert item["kind"] == "experiment" and item["args"] == {"study": "tender", "name": "paper_trial",
                                                            "spec_sha": req["sha"]}
    v = E.status(ws, "tender", "paper_trial")
    assert v["state"] == "requested" and v["request"]["by"] == "lead-1" and v["request"]["status"] == "pending"
    with pytest.raises(StudyError, match="person's act"):
        E.start(ws, "tender", "paper_trial", "go")                        # no person key in this process
    from openlab.study.record import Record
    rec = Record(folder)
    assert rec.find("action", verb="experiment_start")[0].payload["outcome"]["ok"] is False
    assert not rec.find("experiment_start")
    assert fake_crontab.read_text() == "0 1 * * * echo the person's own line\n"
    with pytest.raises(StudyError, match="can only be written by person"):
        rec.add("experiment_start", {"name": "paper_trial", "sha": req["sha"], "spec": v["spec"], "lines": [],
                                     "reason": "forged"}, actor="agent", links=[("starts", req["id"])])
    from openlab.agents.inbox import decide
    with person.act():
        done = decide(st, req["item"], True, "run it")
    assert done["status"] == "approved" and done["result"]["summary"].startswith("started paper_trial")
    start = rec.find("experiment_start")[0]
    assert start.actor == "person" and start.payload["sig"]["key"] == person.signer.fingerprint
    lines = fake_crontab.read_text().splitlines()
    assert lines[0] == "0 1 * * * echo the person's own line" and len(lines) == 3
    tag = f"# openlab-experiment:{ws}::tender::paper_trial::"
    assert lines[1].endswith(tag + "scan") and lines[2].endswith(tag + "export")
    assert " experiment job tender paper_trial scan --root " in lines[1] and "flock -n" in lines[1]
    assert lines[2].startswith(E.line_cron("30 23 * * 1-5", "America/New_York") + " ")
    v = E.status(ws, "tender", "paper_trial")
    assert v["state"] == "running" and all(j["installed"] and j["next"] for j in v["jobs"])
    assert rec.verify()["ok"]
    with person.act(), pytest.raises(StudyError, match="already started"):
        E.start(ws, "tender", "paper_trial", "again")


def test_a_spec_changed_after_the_request_or_the_start_is_refused(lab):
    ws, folder, _, person = lab
    save(folder, spec())
    from openlab.agents.identity import acting_as
    with acting_as("lead-1"):
        req = E.request(ws, "tender", "paper_trial", "trial")
    save(folder, spec(stop="passes = 4"))
    assert E.status(ws, "tender", "paper_trial")["state"] == "draft"          # edited since its request
    with person.act(), pytest.raises(StudyError, match="changed since it was requested"):
        E.start(ws, "tender", "paper_trial", "go", spec_sha=req["sha"])
    with person.act():
        E.start(ws, "tender", "paper_trial", "go")                            # the person starts the file as it is
    with pytest.raises(StudyError, match="has started"):
        E.write(ws, "tender", "paper_trial", spec(stop="passes = 9"))
    save(folder, spec(stop="passes = 9"))                                     # by hand: flagged, not run
    v = E.status(ws, "tender", "paper_trial")
    assert v["edited"] and v["spec"]["stop"]["passes"] == 4
    with acting_as("lead-1"), pytest.raises(StudyError, match="has started"):
        E.request(ws, "tender", "paper_trial", "again")


# --- running: jobs, passes, the stop rule, the reaction ----------------------------------------------------------------
def test_a_job_runs_only_at_its_minute_once_and_counts_passes_to_the_end(lab, fake_crontab, spawned):
    ws, folder, pin, person = lab
    from openlab.agents import plan
    from openlab.agents.store import Store
    st = Store(ws)
    st.register("lead-1", "spawned", role="lead", status="running", budget={"usd": 5})
    plan.report(st, "lead-1", [{"id": "paper-exp", "title": "The paper trial", "kind": "run", "state": "planned"}])
    req, out = started(lab)
    assert plan.read(st, "lead-1")["versions"][-1]["nodes"][0]["state"] == "active"
    assert E.job(ws, "tender", "paper_trial", "scan", at=ny(19, 9, 35))["ran"] is False     # not due
    r = E.job(ws, "tender", "paper_trial", "scan", at=ny(19, 9, 30))
    assert r["ran"] and r["exit"] == 0 and r["run"].startswith("instrument_run:") and not r["pass"]
    assert E.job(ws, "tender", "paper_trial", "scan", at=ny(19, 9, 31))["why"].startswith("scan already ran")
    from openlab.study.record import Record
    rec = Record(folder)
    run = rec.get(r["run"])
    assert run.payload["instrument"] == pin and run.payload["args"] == ["scan"]          # as instrument run --study
    assert run.payload["note"] == "experiment paper_trial, job scan" and run.payload["registry_run"]
    for day in (19, 20):
        e = E.job(ws, "tender", "paper_trial", "export", at=ny(day, 23, 30))
        assert e["ran"] and e["pass"]
    v = E.status(ws, "tender", "paper_trial")
    assert v["progress"]["passes_ok"] == 2 and v["progress"]["pass_fraction"] == pytest.approx(2 / 3)
    assert v["jobs"][1]["runs"][0]["run"] and v["jobs"][1]["runs"][0]["registry_run"]
    assert not spawned
    last = E.job(ws, "tender", "paper_trial", "export", at=ny(21, 23, 30))
    assert last["ended"]["state"] == "concluded" and "3 passes" in last["ended"]["reason"]
    assert fake_crontab.read_text() == "0 1 * * * echo the person's own line\n"           # its lines, gone
    assert len(spawned) == 1 and last["reaction"] == {"ok": True, "agent": "reviewer-000001"}
    call = spawned[0]
    assert call["role"] == "reviewer" and call["study"] == "tender" and call["budget"] == {
        "usd": 5.0, "minutes": 45.0, "turns": 80}
    assert "Write the conclusion." in call["prompt"]
    assert "You are a reviewer" not in call["prompt"]               # the role's prompt is in its brief, once
    assert f"record {r['run']}; output {run.payload['registry_run']}" in call["prompt"]
    assert I.output(ws, run.payload["registry_run"])["exit"] == 0                  # the output id reads back
    assert r["run"] in call["prompt"] and "experiment_react" in call["prompt"] and "openlab_agent_plan" in call["prompt"]
    assert E.status(ws, "tender", "paper_trial")["state"] == "reacting"
    after = E.job(ws, "tender", "paper_trial", "scan", at=ny(22, 9, 30))
    assert after["ran"] is False and "has ended" in after["why"]
    passes = [p.id for p in rec.find("experiment_pass")]
    with pytest.raises(StudyError, match="verdict"):
        E.react(ws, "tender", "paper_trial", "great", "x", passes)
    out = E.react(ws, "tender", "paper_trial", "met", "the export confirmed every scan", passes)
    v = E.status(ws, "tender", "paper_trial")
    assert v["state"] == "reacted" and v["reaction"]["verdict"] == "met" and set(v["reaction"]["rests_on"]) == set(passes)
    step = plan.read(st, "lead-1")["versions"][-1]["nodes"][0]
    assert step["state"] == "done" and out["id"] in step["links"]
    with pytest.raises(StudyError, match="already has its reaction"):
        E.react(ws, "tender", "paper_trial", "met", "again", passes)
    assert rec.verify()["ok"]


def test_failed_passes_in_a_row_end_it_failed(lab, spawned):
    ws, folder, _, _ = lab
    started(lab, spec(export='"fail"', stop="passes = 5\nmax_failures = 2"))
    a = E.job(ws, "tender", "paper_trial", "export", at=ny(19, 23, 30))
    assert a["ran"] and a["error"] == "exit 3" and a["pass"] and "ended" not in a
    b = E.job(ws, "tender", "paper_trial", "export", at=ny(20, 23, 30))
    assert b["ended"]["state"] == "failed" and "2 failed passes in a row" in b["ended"]["reason"]
    v = E.status(ws, "tender", "paper_trial")
    assert v["progress"]["passes_failed"] == 2 and len(v["failures"]) == 2 and v["end"]["state"] == "failed"
    assert len(spawned) == 1


def test_a_refused_run_is_a_failed_pass_that_says_why(lab):
    ws, folder, pin, _ = lab
    started(lab, spec(stop="passes = 5"))
    snap = I.Registry(ws).snapshot_dir(pin) / "run.sh"
    snap.parent.chmod(0o755)
    snap.chmod(0o755)
    snap.write_text("#!/bin/sh\necho changed\n")                              # the version drifted
    r = E.job(ws, "tender", "paper_trial", "export", at=ny(19, 23, 30))
    assert r["ran"] and not r["run"] and r["error"].startswith("refused:")
    from openlab.study.record import Record
    p = Record(folder).find("experiment_pass")[0].payload
    assert p["ok"] is False and p["runs"] == [] and "refused" in p["error"]


def test_a_cycle_pass_waits_for_every_job(lab):
    ws, folder, _, _ = lab
    started(lab, spec(pass_job="cycle", stop="passes = 2"))
    assert not E.job(ws, "tender", "paper_trial", "scan", at=ny(19, 9, 30))["pass"]
    assert not E.job(ws, "tender", "paper_trial", "scan", at=ny(19, 9, 40))["pass"]
    one = E.job(ws, "tender", "paper_trial", "export", at=ny(19, 23, 30))
    assert one["pass"]
    from openlab.study.record import Record
    p = Record(folder).get(one["pass"]).payload
    assert p["job"] == "cycle" and p["ok"] and len(p["runs"]) == 2         # the latest scan and the export
    assert not E.job(ws, "tender", "paper_trial", "export", at=ny(20, 23, 30))["pass"]
    assert E.job(ws, "tender", "paper_trial", "scan", at=ny(21, 9, 30))["ended"]["state"] == "concluded"


def test_until_ends_it_at_the_first_job_after(lab, spawned, monkeypatch):
    ws, folder, _, _ = lab
    monkeypatch.setattr(E, "_now", lambda: ny(19, 12, 0))       # started the Monday before, whatever today is
    started(lab, spec(stop="until = 2026-10-20T12:00:00-04:00"))
    assert E.job(ws, "tender", "paper_trial", "scan", at=ny(20, 9, 30))["ran"]
    r = E.job(ws, "tender", "paper_trial", "scan", at=ny(20, 12, 10))
    assert r["ran"] is False and r["ended"]["state"] == "concluded" and "until" in r["ended"]["reason"]
    assert len(spawned) == 1


def test_the_person_runs_a_job_now_and_its_pass_is_off_schedule(lab):
    ws, folder, _, _ = lab
    started(lab)
    r = E.job(ws, "tender", "paper_trial", "export", now=True)
    from openlab.study.record import Record
    assert Record(folder).get(r["pass"]).payload["off_schedule"] is True


# --- kill -------------------------------------------------------------------------------------------------------------
def test_a_kill_removes_the_lines_ends_it_and_fires_the_reaction(lab, fake_crontab, spawned):
    ws, folder, _, _ = lab
    started(lab)
    assert len(fake_crontab.read_text().splitlines()) == 3
    from openlab.agents.identity import acting_as
    with pytest.raises(StudyError, match="needs a reason"):
        E.kill(ws, "tender", "paper_trial", " ")
    with acting_as("lead-1"):
        out = E.kill(ws, "tender", "paper_trial", "the scan is wrong")
    assert out["state"] == "killed" and out["reaction"]["ok"]
    assert fake_crontab.read_text() == "0 1 * * * echo the person's own line\n"
    v = E.status(ws, "tender", "paper_trial")
    assert v["end"]["state"] == "killed" and v["end"]["by"] == "lead-1" and "the scan is wrong" in v["end"]["reason"]
    assert v["state"] == "reacting" and len(spawned) == 1 and "killed by lead-1" in spawned[0]["prompt"]
    with pytest.raises(StudyError, match="already ended"):
        E.kill(ws, "tender", "paper_trial", "again")


def test_a_failed_spawn_is_recorded_and_can_be_fired_again(lab, monkeypatch):
    ws, folder, _, _ = lab
    started(lab)

    def broken(root, **kw):
        raise RuntimeError("no claude here")
    monkeypatch.setattr(E, "SPAWNER", broken)
    out = E.kill(ws, "tender", "paper_trial", "stop")
    assert out["reaction"] == {"ok": False, "error": "no claude here"}
    v = E.status(ws, "tender", "paper_trial")
    assert v["state"] == "killed" and v["reaction"]["spawn_error"] == "no claude here"
    monkeypatch.setattr(E, "SPAWNER", lambda root, **kw: {"id": "reviewer-abc"})
    assert E.respawn(ws, "tender", "paper_trial") == {"ok": True, "agent": "reviewer-abc"}
    assert E.respawn(ws, "tender", "paper_trial")["again"] is False                       # once
    assert E.status(ws, "tender", "paper_trial")["state"] == "reacting"


def test_a_crontab_that_cannot_be_written_fails_the_start_and_never_blocks_a_kill(lab, tmp_path, monkeypatch, spawned):
    ws, folder, _, person = lab
    good = os.environ["OPENLAB_CRONTAB"]
    broken = tmp_path / "broken-crontab"
    broken.write_text('#!/bin/sh\nif [ "$1" = "-l" ]; then exit 1; fi\necho "no crontab for you" >&2; exit 1\n')
    broken.chmod(0o755)
    save(folder, spec())
    monkeypatch.setenv("OPENLAB_CRONTAB", str(broken))
    with person.act(), pytest.raises(StudyError, match="could not be installed"):
        E.start(ws, "tender", "paper_trial", "go")
    v = E.status(ws, "tender", "paper_trial")
    assert v["state"] == "failed" and "could not be installed" in v["end"]["reason"] and not spawned
    monkeypatch.setenv("OPENLAB_CRONTAB", good)
    save(folder, spec(name="second"), "second")
    with person.act():
        E.start(ws, "tender", "second", "go")
    readonly = tmp_path / "readonly-crontab"                                  # it lists the lines, and cannot write
    readonly.write_text(f'#!/bin/sh\nif [ "$1" = "-l" ]; then cat "{tmp_path / "crontab.txt"}"; exit 0; fi\nexit 1\n')
    readonly.chmod(0o755)
    monkeypatch.setenv("OPENLAB_CRONTAB", str(readonly))
    out = E.kill(ws, "tender", "second", "stop it")
    assert out["state"] == "killed" and "could not be removed" in out["warning"] and len(spawned) == 1
    monkeypatch.setenv("OPENLAB_CRONTAB", good)
    late = E.job(ws, "tender", "second", "scan", at=ny(19, 9, 30))           # the line left behind runs nothing
    assert late["ran"] is False and late["removed"] == 2


# --- the record's rules ------------------------------------------------------------------------------------------------
def test_only_openlab_counts_passes_and_ends_by_rule(lab):
    ws, folder, _, _ = lab
    started(lab)
    from openlab.study.record import Record
    rec = Record(folder)
    start = rec.find("experiment_start")[0]
    with pytest.raises(StudyError, match="can only be written by openlab"):
        rec.add("experiment_pass", {"name": "paper_trial", "n": 1, "ok": True, "job": "export", "runs": []},
                actor="agent", links=[("on", start.id)])
    with pytest.raises(StudyError, match="by its own rules"):
        rec.add("experiment_end", {"name": "paper_trial", "state": "concluded", "reason": "I say so", "by": "x",
                                   "passes_ok": 0, "passes_failed": 0}, actor="agent", links=[("ends", start.id)])
    with pytest.raises(StudyError, match="next pass"):
        rec.add("experiment_pass", {"name": "paper_trial", "n": 2, "ok": False, "job": "export", "runs": [],
                                    "error": "x"}, actor="openlab", session="t", links=[("on", start.id)])


# --- time zones ---------------------------------------------------------------------------------------------------------
def test_a_line_fires_at_least_when_its_job_is_due_in_its_own_zone(monkeypatch):
    import time
    monkeypatch.setenv("TZ", "Europe/London")
    time.tzset()
    try:
        assert E.line_cron("30 23 * * 1-5", "Europe/London") == "30 23 * * 1-5"
        assert E.line_cron("30 23 * * 1-5", "America/New_York") == "30 * * * *"
        assert E.line_cron("30 23 * * 1-5", "Asia/Kolkata") == "* * * * *"           # a half-hour offset
        assert E.line_cron("30 23 * * 1-5", "") == "30 23 * * 1-5"
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()
    assert E.next_fire("30 23 * * 1-5", "America/New_York", ny(19, 12, 0)) == "2026-10-19T23:30-04:00"
    from openlab.study import schedule
    with pytest.raises(ValueError, match="never matches within a year"):
        schedule.next_run("0 0 29 2 *", dt.datetime(2026, 3, 1))                  # the next Feb 29 is in 2028
    assert schedule.next_run("0 0 1 1 *", dt.datetime(2026, 3, 1)) == dt.datetime(2027, 1, 1)
    assert schedule.next_run("*/15 9 * * 1", dt.datetime(2026, 10, 17, 12)) == dt.datetime(2026, 10, 19, 9, 0)


def test_a_late_start_still_runs_its_minute(lab):
    ws, folder, _, _ = lab
    started(lab)
    r = E.job(ws, "tender", "paper_trial", "export", at=ny(19, 23, 31))      # cron ran it a minute late
    assert r["ran"] and E.job(ws, "tender", "paper_trial", "export", at=ny(19, 23, 32))["ran"] is False


# --- the crontab line, run as cron runs it ------------------------------------------------------------------------------
def test_the_installed_line_runs_the_job(lab, fake_crontab, tmp_path):
    ws, folder, _, _ = lab
    every = spec().replace('"*/10 9-20 * * 1-5"', '"* * * * *"').replace('tz = "America/New_York"', 'tz = ""', 1)
    started(lab, every)
    line = next(l for l in fake_crontab.read_text().splitlines() if l.endswith("::scan"))
    assert line.startswith("* * * * * cd ")
    cmd = line[len("* * * * * "):line.rfind(" # openlab-experiment:")].replace("\\%", "%")
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "PYTHONPATH": str(REPO),
           "OPENLAB_CRONTAB": os.environ["OPENLAB_CRONTAB"], "OPENLAB_EXPERIMENT_SPAWN": "register"}
    subprocess.run(["sh", "-c", cmd], env=env, check=True, timeout=120)
    log = (folder / ".openlab" / "experiments" / "paper_trial" / "cron.log").read_text()
    v = E.status(ws, "tender", "paper_trial")
    assert v["jobs"][0]["n_runs"] == 1 and v["jobs"][0]["runs"][0]["ok"], log


# --- the CLI and the tools ----------------------------------------------------------------------------------------------
def test_the_cli_writes_a_template_and_the_person_starts_with_the_passphrase(lab, capsys, fake_crontab):
    from openlab.study import experiment_cli
    ws, folder, _, person = lab
    root = ["--root", str(ws)]
    assert experiment_cli.main(["new", "tender", "trial2", *root]) == 0
    out = capsys.readouterr().out
    assert "wrote tender/experiments/trial2.toml" in out and "valid" in out
    assert E.parse((folder / "experiments" / "trial2.toml").read_text(), folder, "trial2", ws)["jobs"][0]["instrument"] \
        == "tender"
    person.answers()
    assert experiment_cli.main(["start", "tender", "trial2", "--reason", "go", *root]) == 0
    assert "started trial2" in capsys.readouterr().out
    assert experiment_cli.main(["status", "tender", "trial2", *root]) == 0
    assert "RUNNING" in capsys.readouterr().out
    assert experiment_cli.main(["list", *root, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["experiments"][0]["state"] == "running"
    assert experiment_cli.main(["kill", "tender", "trial2", "--reason", "done", *root]) == 0
    assert experiment_cli.main(["react", "tender", "trial2", "--spawn", *root, "--json"]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["again"] is False


def test_every_cli_verb_prints_its_text_without_json(lab, capsys, fake_crontab, spawned, tmp_path, monkeypatch):
    """B10: the text output of each verb reads the keys its function returns (`new --file` crashed on 'written')."""
    from openlab.agents.store import Store
    from openlab.study import experiment_cli
    ws, folder, _, person = lab
    root = ["--root", str(ws)]

    def run(*argv, code=0):
        assert experiment_cli.main([*argv, *root]) == code
        o = capsys.readouterr()
        return o.out, o.err

    draft = tmp_path / "paper_trial.toml"                                       # new --file: a valid spec
    draft.write_text(spec())
    out, _ = run("new", "tender", "paper_trial", "--file", str(draft))
    assert out.startswith("wrote tender/experiments/paper_trial.toml (sha ") and "): valid" in out
    assert "next: ask the person to start it: experiment_request(study='tender', name='paper_trial'" in out
    assert experiment_cli.main(["new", "tender", "paper_trial", "--file", str(draft), *root, "--json"]) == 0
    j = json.loads(capsys.readouterr().out)
    assert {"file", "state", "sha", "spec", "written", "ok", "error"} <= set(j) and j["written"] and j["ok"]
    bad = tmp_path / "bad.toml"                                                 # new --file: an invalid spec
    bad.write_text(spec().replace("passes = 3", "max_failures = 2"))
    out, err = run("new", "tender", "paper_trial", "--file", str(bad), code=1)
    assert out == "" and "needs passes or until" in err and save(folder, spec()).read_text() == spec()
    _, err = run("new", "tender", "paper_trial", "--file", str(tmp_path / "missing.toml"), code=1)
    assert "cannot read the spec file" in err
    out, _ = run("new", "tender", "paper_trial")                                # new: the draft there, unwritten
    assert out.startswith("tender/experiments/paper_trial.toml (sha ") and "): valid" in out
    out, _ = run("new", "tender", "trial2")                                     # new: the template, written
    assert out.startswith("wrote tender/experiments/trial2.toml") and "): valid" in out
    save(folder, spec(name="broken").replace("passes = 3", "max_failures = 2"), "broken")
    out, _ = run("new", "tender", "broken")                                     # new: an invalid draft there
    assert "tender/experiments/broken.toml" in out and "not yet valid: " in out and "needs passes or until" in out
    assert "next:" not in out
    out, _ = run("list")
    assert "tender/paper_trial: draft" in out and "tender/broken: draft" in out
    out, _ = run("status", "tender", "paper_trial")
    assert "paper_trial (tender): DRAFT" in out

    Store(ws).register("lead-1", "spawned", role="lead", status="running", budget={"usd": 5})
    monkeypatch.setenv("OPENLAB_AGENT", "lead-1")
    out, _ = run("request", "tender", "paper_trial", "--reason", "the forward paper trial")
    assert ": requested, inbox #1 (sha " in out and "the person starts it: approving inbox #1" in out
    out, _ = run("new", "tender", "paper_trial")
    assert "): valid; it is requested" in out and "next:" not in out
    monkeypatch.delenv("OPENLAB_AGENT")
    person.answers()
    out, _ = run("start", "tender", "paper_trial", "--reason", "go")
    assert "started paper_trial: 2 crontab line(s)" in out and out.count("::paper_trial::") == 2 and "Roadmap: " in out
    out, _ = run("job", "tender", "paper_trial", "scan")
    assert json.loads(out)["ran"] in (True, False)

    def no_spawn(root, **kw):
        raise RuntimeError("no agents here")
    monkeypatch.setattr(E, "SPAWNER", no_spawn)
    out, _ = run("kill", "tender", "paper_trial", "--reason", "done")
    assert ": killed" in out and "reaction: not started: no agents here" in out and "{" not in out
    monkeypatch.setattr(E, "SPAWNER", lambda root, **kw: {"id": "reviewer-000001"})
    out, _ = run("react", "tender", "paper_trial", "--spawn")
    assert json.loads(out) == {"ok": True, "agent": "reviewer-000001"}
    end = E.status(ws, "tender", "paper_trial")["end"]["id"]
    out, _ = run("react", "tender", "paper_trial", "--verdict", "inconclusive", "--conclusion", "killed early",
                 "--rests-on", end)
    assert json.loads(out)["state"] == "reacted"


def test_the_cli_kill_text_names_the_reaction_agent(lab, capsys, spawned):
    from openlab.study import experiment_cli
    ws, _, _, _ = lab
    started(lab)
    assert experiment_cli.main(["kill", "tender", "paper_trial", "--reason", "done", "--root", str(ws)]) == 0
    assert capsys.readouterr().out.splitlines()[1] == "  reaction: reviewer-000001"


def test_the_tools_set_up_request_read_kill_and_conclude(lab, monkeypatch):
    from openlab.agents.store import Store
    from openlab.app import ToolContext, dispatch
    ws, folder, _, person = lab
    st = Store(ws)
    st.register("lead-1", "spawned", role="lead", status="running", budget={"usd": 5})
    monkeypatch.setenv("OPENLAB_AGENT", "lead-1")
    ctx = ToolContext(str(ws))
    bad = dispatch(ctx, "experiment_new", {"study": "tender", "name": "x1", "toml": spec(name="x1") + "[nope]\n"})
    assert not bad["ok"] and "no key nope" in bad["error"]
    ok = dispatch(ctx, "experiment_new", {"study": "tender", "name": "x1", "toml": spec(name="x1")})
    assert ok["ok"], ok
    req = dispatch(ctx, "experiment_request", {"study": "tender", "name": "x1", "reason": "trial"})
    assert req["ok"] and st.item(req["result"]["item"])["agent"] == "lead-1"
    filed = dispatch(ctx, "agent_request", {"kind": "experiment", "reason": "again", "study": "tender",
                                            "args": {"name": "x1"}})
    assert filed["ok"] and st.item(filed["result"]["item"])["args"]["spec_sha"] == req["result"]["sha"]
    assert not dispatch(ctx, "agent_request", {"kind": "experiment", "reason": "r", "args": {"name": "nope",
                                                                                            "study": "tender"}})["ok"]
    lst = dispatch(ctx, "experiment_list", {})["result"]["experiments"]
    assert lst[0]["name"] == "x1" and lst[0]["state"] == "requested" and lst[0]["passes"] == "0/3"
    with person.act():
        E.start(ws, "tender", "x1", "go")
    assert [st.item(i)["status"] for i in (req["result"]["item"], filed["result"]["item"])] == ["cancelled"] * 2
    k = dispatch(ctx, "experiment_kill", {"study": "tender", "name": "x1", "reason": "enough"})
    assert k["ok"] and k["result"]["state"] == "killed"
    s = dispatch(ctx, "experiment_status", {"study": "tender", "name": "x1"})["result"]
    r = dispatch(ctx, "experiment_react", {"study": "tender", "name": "x1", "verdict": "inconclusive",
                                           "conclusion": "killed before any pass", "rests_on": [s["end"]["id"]]})
    assert r["ok"], r
    assert dispatch(ctx, "experiment_status", {"study": "tender", "name": "x1"})["result"]["reaction"]["by"] == "lead-1"


def test_the_hooks_keep_the_start_and_an_off_schedule_job_the_persons(tmp_path, monkeypatch, capsys):
    from openlab.agents import hooks
    start = "openlab experiment " + "start tender paper_trial --reason go"
    g = hooks.guard("Bash", {"command": start})
    assert g and g[0] == "bypass" and "starting an experiment" in g[1]
    g = hooks.guard("Bash", {"command": "openlab experiment job tender paper_trial scan --" + "now"})
    assert g and "off schedule" in g[1]
    for ok in ("openlab experiment job tender paper_trial scan", "openlab experiment kill tender p --reason 'start over'",
               "openlab experiment status tender paper_trial", "openlab experiment " + "start --help",
               "openlab experiment " + "start tender x -h"):
        assert hooks.guard("Bash", {"command": ok}) is None, ok
    from openlab import __main__ as cli
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(home)
    with pytest.raises(SystemExit) as e:
        cli.main(["experiment", "start", "--help"])
    assert e.value.code == 0 and "usage:" in capsys.readouterr().out and list(home.iterdir()) == []


# --- what openlab show reads ---------------------------------------------------------------------------------------------
def test_the_show_snapshot_carries_the_experiments_and_the_instruments(lab):
    from openlab.app import ToolContext, dispatch
    ws, folder, pin, _ = lab
    started(lab)
    E.job(ws, "tender", "paper_trial", "scan", at=ny(19, 9, 30))
    snap = dispatch(ToolContext(str(ws)), "show_snapshot", {})["result"]
    x = snap["experiments"][0]
    assert x["name"] == "paper_trial" and x["state"] == "running" and x["plan_step"] == "paper-exp"
    assert x["jobs"][0]["runs"][0]["run"] and x["jobs"][0]["next"]
    i = snap["instruments"][0]
    assert i["id"] == pin and i["pinned_by"] == [{"study": "tender", "alias": "tender"}] and i["runs"] == 1
    assert i["last_run"]["exit"] == 0 and i["usage_head"]


# --- the worked example of docs/experiments.md, run verbatim -------------------------------------------------------------
def test_the_worked_example_of_the_docs(ws, person_key, tmp_path, capsys, fake_crontab, spawned, monkeypatch):
    import re
    from openlab.agents import plan
    from openlab.agents.inbox import decide
    from openlab.agents.store import Store
    from openlab.app import ToolContext, dispatch
    from openlab.study import experiment_cli
    doc = (REPO / "docs" / "experiments.md").read_text()
    example = doc[doc.index("## Worked example"):doc.index("## What openlab show draws")]
    toml = re.findall(r"```toml\n(.*?)```", example, re.S)[0]
    toml = "\n".join(line[3:] if line.startswith("   ") else line for line in toml.splitlines()) + "\n"
    pin = register(ws, person_key)
    make_study(ws, {"tender": pin}, folder="tender_trial")
    root = ["--root", str(ws)]
    draft = tmp_path / "paper_trial.toml"
    draft.write_text(toml)
    assert experiment_cli.main(["new", "tender_trial", "paper_trial", "--file", str(draft), *root, "--json"]) == 0
    capsys.readouterr()
    st = Store(ws)
    st.register("lead-1", "spawned", role="lead", status="running", budget={"usd": 5})
    plan.report(st, "lead-1", [{"id": "paper-exp", "title": "The forward check", "kind": "run", "state": "planned"}])
    monkeypatch.setenv("OPENLAB_AGENT", "lead-1")
    req = dispatch(ToolContext(str(ws)), "experiment_request", {"study": "tender_trial", "name": "paper_trial",
                                                               "reason": "the forward check"})["result"]
    monkeypatch.delenv("OPENLAB_AGENT")
    with person_key.act():
        decide(st, req["item"], True, "go")
    lines = fake_crontab.read_text().splitlines()[1:]
    assert [l.rsplit("::", 1)[1] for l in lines] == ["scan", "export"]
    E.job(ws, "tender_trial", "paper_trial", "scan", at=ny(19, 9, 30))
    E.job(ws, "tender_trial", "paper_trial", "export", at=ny(19, 23, 30))
    end = E.job(ws, "tender_trial", "paper_trial", "export", at=ny(20, 23, 30))
    assert end["ended"] == {**end["ended"], "state": "concluded", "reason": "stop rule met: 2 passes"}
    assert fake_crontab.read_text().splitlines() == ["0 1 * * * echo the person's own line"]
    assert experiment_cli.main(["status", "tender_trial", "paper_trial", *root]) == 0
    out = capsys.readouterr().out
    assert "paper_trial (tender_trial): REACTING" in out and "2/2 ok, 0 failed" in out and "Roadmap: lead-1 / paper-exp" in out
    assert "asked by lead-1, inbox #1 (approved)" in out and "stop rule met: 2 passes" in out
    assert spawned[0]["role"] == "reviewer" and spawned[0]["budget"] == {"usd": 2.0, "minutes": 20.0}
    passes = [p["id"] for p in E.status(ws, "tender_trial", "paper_trial")["passes"]]
    assert experiment_cli.main(["react", "tender_trial", "paper_trial", "--verdict", "met", "--conclusion",
                                "both exports confirmed the scans", "--rests-on", *passes, *root]) == 0
    assert E.status(ws, "tender_trial", "paper_trial")["state"] == "reacted"
    assert plan.read(st, "lead-1")["versions"][-1]["nodes"][0]["state"] == "done"
