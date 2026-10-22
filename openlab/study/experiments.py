"""Experiments: a study's pinned instruments run on a schedule, unattended, for a set number of passes; then the
experiment ends and an agent is fired to draw its conclusion.

An experiment is a file, `<study>/experiments/<name>.toml` (`parse` checks it strictly):

* `[experiment]` its name (the file's), its purpose (what it tests and how it will be judged), and optionally the
  Roadmap step it implements (`plan_step`, of `owner`: by default the agent that asked for it);
* `[[jobs]]` each runs one instrument alias the study pins (`[instruments]` in study.toml) with fixed args, on a cron
  schedule in a time zone. A `live-order` instrument is refused: each of its runs is a person's own act;
* `[pass]` what one pass is: one successful run of a named job, or `cycle` (every job ran once since the last pass);
* `[stop]` when it ends: after `passes` passes, at `until`, or after `max_failures` failed passes in a row
  (default 3, 0 for none). `passes` or `until` is required;
* `[react]` the role, prompt and budget of the agent fired when it ends.

Its life is on the study's record:

    draft -> requested -> running -> concluded | killed | failed -> reacting -> reacted

* `experiment`: the spec (its file's sha256 and its checked form), who asked for it and the inbox item. An agent asks
  with `request` (the tool `experiment_request`), which files `agent_request(kind="experiment")`.
* `experiment_start`: a person node, signed. The experiment runs unattended and outside the sandbox, so starting it is
  the person's act: approving the inbox item, or `openlab experiment start STUDY NAME`. It installs one tagged line per
  job in the user's crontab (`# openlab-experiment:<workspace>::<study>::<name>::<job>`; only these lines are ever
  touched, under a lock), each running `openlab experiment job STUDY NAME JOB` at its minute.
* each job run is an instrument run recorded on the study exactly as `instrument run --study` records one, plus an
  `action` (`experiment_job`) naming its slot (the scheduled minute; a job runs once per slot, and only at a minute
  its schedule fires, unless a person runs it `--now`, which the pass marks as off schedule);
* `experiment_pass`: openlab counts a pass when the pass rule is met: its number, ok or failed, the runs it rests on;
* `experiment_end`: the stop rule met (concluded), the failure limit (failed), or killed by an agent or the person
  with a reason. The crontab lines are removed first; then the reaction agent is spawned (an `action`
  `experiment_react_spawn` names it);
* `experiment_reaction`: the agent's verdict against the purpose (met, not_met, inconclusive), its conclusion, what
  it rests on and the record nodes it wrote.

An experiment runs the spec it was started with (the start node holds it). A spec file edited afterwards is not run,
and is flagged; a changed experiment is a new one, under a new name. When the spec names a plan step, the owner's step
moves to active at the start and to done (failed, for a failed experiment) when the reaction is recorded.

Test seams: `OPENLAB_CRONTAB` names the crontab program (openlab.study.schedule), and `SPAWNER` (or
`OPENLAB_EXPERIMENT_SPAWN=register`, which registers the agent and starts nothing) replaces the spawn.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shlex
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

from . import schedule
from .errors import StudyError
from .schema import EXPERIMENT_ENDS, EXPERIMENT_VERDICTS

NAME = re.compile(r"[a-z][a-z0-9_-]{0,47}")
STEP = re.compile(r"[A-Za-z0-9_.-]{1,40}")                     # an agent's plan step id (openlab.agents.plan)
FOLDER = "experiments"
TAG = "# openlab-experiment:"
STATES = ("draft", "requested", "running", "concluded", "killed", "failed", "reacting", "reacted")
DEFAULT_MAX_FAILURES = 3
GRACE_MINUTES = 2               # a job started late by cron (a loaded machine) still runs its scheduled minute
RUNS_SHOWN = 8                 # of each job, in a status
PROMPT_RUNS = 40                # runs listed in the reaction agent's prompt
SECTIONS = ("experiment", "jobs", "pass", "stop", "react")
KEYS = {"experiment": ("name", "purpose", "plan_step", "owner"), "pass": ("job",),
        "stop": ("passes", "until", "max_failures", "tz"), "react": ("role", "prompt", "budget")}
JOB_KEYS = ("name", "instrument", "args", "cron", "tz", "timeout")
BUDGET_KEYS = ("usd", "minutes", "turns")
#: a test's spawner: called as SPAWNER(root, prompt=..., role=..., study=..., question=..., budget=...) -> agent row
SPAWNER: Optional[Callable] = None


# --- where things are --------------------------------------------------------------------------------------------------
def _ws(root) -> Path:
    from .instruments import workspace_root
    return workspace_root(root or None)


def locate(root, study: str) -> tuple:
    """(workspace, the study's folder, its path from the workspace) for a study named by its path."""
    ws = _ws(root)
    folder = Path(study) if Path(study).is_absolute() else ws / study
    folder = folder.resolve()
    if not (folder / "study.toml").is_file():
        raise StudyError(f"{study} is not a study (no study.toml)", f"workspace {ws}",
                         "name the study by its folder, from the workspace")
    try:
        rel = str(folder.relative_to(ws))
    except ValueError:
        raise StudyError(f"{study} is not in the workspace {ws}") from None
    return ws, folder, rel


def spec_file(folder: Path, name: str) -> Path:
    return Path(folder) / FOLDER / f"{name}.toml"


def _dir(folder: Path, name: str) -> Path:
    d = Path(folder) / ".openlab" / "experiments" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _record(folder: Path, create: bool = True):
    from .record import Record
    if not create and not (Path(folder) / ".openlab" / "record.db").is_file():
        return None
    return Record(folder)


def _session(name: str) -> str:
    return f"experiment:{name}"


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _iso(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).isoformat(timespec="seconds")


def _parse_iso(s: str) -> Optional[dt.datetime]:
    try:
        t = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


# --- time zones --------------------------------------------------------------------------------------------------------
def _zone(tz: str):
    if not tz:
        return None
    from zoneinfo import ZoneInfo
    return ZoneInfo(tz)


def _wall(t: dt.datetime, tz: str) -> dt.datetime:
    """`t` as a naive wall-clock time in `tz` (this machine's local time when empty)."""
    z = _zone(tz)
    return (t.astimezone(z) if z else t.astimezone()).replace(tzinfo=None)


def _aware(w: dt.datetime, tz: str) -> dt.datetime:
    z = _zone(tz)
    return w.replace(tzinfo=z) if z else w.astimezone()


def next_fire(cron: str, tz: str, after: Optional[dt.datetime] = None) -> str:
    """The next minute the job fires, in its own zone, as ISO with the offset."""
    return _aware(schedule.next_run(cron, _wall(after or _now(), tz)), tz).isoformat(timespec="minutes")


def line_cron(cron: str, tz: str, now: Optional[dt.datetime] = None) -> str:
    """The schedule the crontab line runs on. cron reads its lines in this machine's local time, so a job in another
    zone gets a line that fires at least at every minute it is due (its own minutes when the zones differ by whole
    hours all year, else every minute), and `job` runs only at a minute its schedule fires in its zone."""
    z = _zone(tz)
    if z is None:
        return cron
    now = now or _now()
    diffs = set()
    for d in range(0, 400):
        t = now + dt.timedelta(days=d)
        diffs.add(t.astimezone(z).utcoffset() - t.astimezone().utcoffset())
    if diffs == {dt.timedelta(0)}:
        return cron
    if all(x.total_seconds() % 3600 == 0 for x in diffs):
        return f"{cron.split()[0]} * * * *"
    return "* * * * *"


# --- the spec ----------------------------------------------------------------------------------------------------------
def _toml(text: str, where: str) -> dict:
    try:
        import tomllib
    except ImportError:                                                     # python 3.10
        import tomli as tomllib                                             # type: ignore
    try:
        return tomllib.loads(text)
    except Exception as e:                                                  # noqa: BLE001 -- a parse error, shown
        raise StudyError(f"{where} is not valid TOML", str(e), "fix the file") from None


def _keys(table, allowed: tuple, where: str) -> dict:
    if not isinstance(table, dict):
        raise StudyError(f"{where} is not a table")
    bad = sorted(set(table) - set(allowed))
    if bad:
        raise StudyError(f"{where} has no key {', '.join(bad)}", f"its keys: {', '.join(allowed)}",
                         "an experiment's spec is checked strictly, so a typo never silently changes what runs")
    return table


def _text(table: dict, key: str, where: str, required: bool = True) -> str:
    v = table.get(key, "")
    if not isinstance(v, str) or (required and not v.strip()):
        raise StudyError(f"{where} {key} is {'a non-empty string' if required else 'a string'}", f"got {v!r}")
    return v.strip()


def _int(table: dict, key: str, where: str, lo: int) -> Optional[int]:
    if key not in table:
        return None
    v = table[key]
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise StudyError(f"{where} {key} is a whole number of at least {lo}", f"got {v!r}")
    return v


def _num(table: dict, key: str, where: str) -> Optional[float]:
    if key not in table:
        return None
    v = table[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
        raise StudyError(f"{where} {key} is a positive number", f"got {v!r}")
    return float(v)


def _tz(table: dict, where: str) -> str:
    tz = table.get("tz", "")
    if not isinstance(tz, str):
        raise StudyError(f"{where} tz is a time zone name", f"got {tz!r}")
    if tz:
        try:
            _zone(tz)
        except Exception:                                                   # noqa: BLE001 -- ZoneInfoNotFoundError
            raise StudyError(f"{where} tz {tz!r} is not a time zone", "an IANA name, such as America/New_York or "
                             "UTC") from None
    return tz


def _pins(folder: Path) -> dict:
    """The study's `[instruments]` pins (sealed, when it is)."""
    from . import spec as _spec
    rec = _record(folder, create=False)
    sp = (_spec.sealed(rec) if rec is not None else None) or _spec.load(folder)
    return dict(sp.raw.get("instruments") or {})


def _until(v, zone_name: str) -> str:
    """`until` as an ISO time with its offset: a date is the end of that day in the experiment's zone."""
    z = _zone(zone_name)
    if isinstance(v, dt.datetime):
        t = v if v.tzinfo else (v.replace(tzinfo=z) if z else v.astimezone())
    elif isinstance(v, dt.date):
        w = dt.datetime.combine(v + dt.timedelta(days=1), dt.time())
        t = w.replace(tzinfo=z) if z else w.astimezone()
    else:
        raise StudyError("[stop] until is a date or a date and time", "until = 2026-12-31, or "
                         "until = 2026-12-31T16:00:00-05:00", "without quotes: a TOML date")
    return t.isoformat(timespec="seconds")


def parse(text: str, folder: Path, name: str, root=None) -> dict:
    """The spec's checked form, or StudyError naming what is wrong: unknown keys at any level, an alias the study does
    not pin, a version not registered, a live-order instrument, a bad cron or time zone, no stop rule."""
    where = f"{FOLDER}/{name}.toml"
    if not NAME.fullmatch(name or ""):
        raise StudyError(f"`{name}` is not an experiment name", "1-48 of a-z, 0-9, _ and -, starting with a letter")
    raw = _keys(_toml(text, where), SECTIONS, where)
    for sec in SECTIONS:
        if sec not in raw:
            raise StudyError(f"{where} has no [{sec}]" if sec != "jobs" else f"{where} has no [[jobs]]",
                             "an experiment has [experiment], [[jobs]], [pass], [stop] and [react]",
                             "`openlab experiment new STUDY NAME` writes a template")
    e = _keys(raw["experiment"], KEYS["experiment"], "[experiment]")
    if e.get("name") != name:
        raise StudyError(f"[experiment] name is {e.get('name')!r}, not {name!r}", "the name is the file's name")
    out: dict = {"name": name, "purpose": _text(e, "purpose", "[experiment]"),
                 "plan_step": _text(e, "plan_step", "[experiment]", required=False),
                 "owner": _text(e, "owner", "[experiment]", required=False)}
    if out["plan_step"] and not STEP.fullmatch(out["plan_step"]):
        raise StudyError(f"[experiment] plan_step {out['plan_step']!r} is not a plan step id",
                         "the id of a step of the owner's Roadmap plan (1-40 letters, digits, _ . or -)")
    jobs = raw["jobs"]
    if not isinstance(jobs, list) or not jobs:
        raise StudyError("[[jobs]] is a list of jobs", "each [[jobs]] table runs one pinned instrument")
    pins = _pins(folder)
    from . import instruments
    reg = instruments.Registry(_ws(root or folder)) if instruments.Registry.exists(_ws(root or folder)) else None
    out["jobs"], seen = [], set()
    for i, j in enumerate(jobs):
        w = f"[[jobs]] #{i + 1}"
        _keys(j, JOB_KEYS, w)
        jn = _text(j, "name", w)
        w = f"job {jn}"
        if not NAME.fullmatch(jn) or jn == "cycle":
            raise StudyError(f"{w}: `{jn}` is not a job name", "1-48 of a-z, 0-9, _ and -; `cycle` is the pass rule's")
        if jn in seen:
            raise StudyError(f"two jobs are named {jn}")
        seen.add(jn)
        alias = _text(j, "instrument", w)
        if alias not in pins:
            raise StudyError(f"{w}: the study does not pin `{alias}`",
                             f"its [instruments]: {', '.join(f'{k} = {v}' for k, v in pins.items()) or 'none'}; a job "
                             "runs an alias the study pins, so it runs one exact version", instruments.PIN_FIX)
        pin = pins[alias]
        if reg is None:
            raise StudyError(f"{w}: no instrument is registered in this workspace",
                             "a job runs a registered version", "a person registers it (openlab instrument register)")
        m = instruments.resolve(reg, pin).payload
        if set(m["capabilities"]) & set(instruments.PERSON_CAPABILITIES):
            raise StudyError(f"{w}: {pin} can place live orders", "each live-order run is a person's act, signed for "
                             "that run; an experiment runs unattended", "run a paper-order version instead")
        args = j.get("args", [])
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise StudyError(f"{w}: args is a list of strings", f"got {args!r}")
        cron = _text(j, "cron", w)
        try:
            schedule.parse_cron(cron)
        except ValueError as ex:
            raise StudyError(f"{w}: {ex}", "five fields: minute hour day-of-month month day-of-week") from None
        out["jobs"].append({"name": jn, "instrument": alias, "pin": pin, "args": list(args), "cron": cron,
                            "tz": _tz(j, w), "timeout": _num(j, "timeout", w), "capabilities": list(m["capabilities"]),
                            "secrets": list(m["secrets"])})
    p = _keys(raw["pass"], KEYS["pass"], "[pass]")
    pj = _text(p, "job", "[pass]")
    if pj != "cycle" and pj not in seen:
        raise StudyError(f"[pass] job {pj!r} is not one of the jobs", f"jobs: {', '.join(sorted(seen))}; or `cycle`: "
                         "a pass when every job ran once since the last")
    out["pass"] = {"job": pj}
    s = _keys(raw["stop"], KEYS["stop"], "[stop]")
    tz = _tz(s, "[stop]") or out["jobs"][0]["tz"]
    stop = {"passes": _int(s, "passes", "[stop]", 1), "until": _until(s["until"], tz) if "until" in s else "",
            "max_failures": DEFAULT_MAX_FAILURES if "max_failures" not in s else _int(s, "max_failures", "[stop]", 0),
            "tz": tz}
    if not stop["passes"] and not stop["until"]:
        raise StudyError("[stop] needs passes or until", "an experiment that runs unattended has a set end",
                         "passes = 20, and/or until = 2026-12-31")
    out["stop"] = stop
    r = _keys(raw["react"], KEYS["react"], "[react]")
    from ..agents.claude_setup import ROLES
    role = _text(r, "role", "[react]")
    if role not in ROLES:
        raise StudyError(f"[react] role {role!r} is not an openlab role", f"roles: {', '.join(ROLES)}")
    b = _keys(r.get("budget", {}), BUDGET_KEYS, "[react] budget")
    budget = {"usd": _num(b, "usd", "[react] budget"), "minutes": _num(b, "minutes", "[react] budget"),
              "turns": _int(b, "turns", "[react] budget", 1)}
    if not budget["usd"] or not budget["minutes"]:
        raise StudyError("[react] budget names usd and minutes", "the reaction agent runs unattended, so its spend is "
                         "set before the person signs", "budget = { usd = 5, minutes = 45, turns = 80 }")
    out["react"] = {"role": role, "prompt": _text(r, "prompt", "[react]", required=False),
                    "budget": {k: v for k, v in budget.items() if v}}
    return out


def read_spec(folder: Path, name: str) -> tuple:
    """(the spec file's text, the sha256 of its bytes)."""
    p = spec_file(folder, name)
    if not p.is_file():
        raise StudyError(f"no experiment `{name}` here", f"{p} does not exist",
                         "write it (openlab experiment new STUDY NAME, or the experiment_new tool)")
    data = p.read_bytes()
    return data.decode("utf-8"), hashlib.sha256(data).hexdigest()


def _q(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (dt.date, dt.datetime)):
        return v.isoformat()
    if isinstance(v, list):
        return "[" + ", ".join(_q(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {_q(x)}" for k, x in v.items()) + " }"
    return json.dumps(str(v), ensure_ascii=False)


_DATE = re.compile(r"\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:\d{2})?)?")


def _until_q(v) -> str:
    """`until` given as text (JSON has no dates) is written as a TOML date when it is one."""
    return v if isinstance(v, str) and _DATE.fullmatch(v.strip()) else _q(v)


def to_toml(spec: dict) -> str:
    """A spec given as a table ({experiment: {...}, jobs: [...], pass, stop, react}) as the file's text."""
    if not isinstance(spec, dict):
        raise StudyError("a spec is a table", '{"experiment": {...}, "jobs": [...], "pass": {...}, "stop": {...}, '
                         '"react": {...}}')
    lines = []
    for sec in SECTIONS:
        if sec not in spec:
            continue
        if sec == "jobs":
            for j in spec["jobs"] if isinstance(spec["jobs"], list) else [spec["jobs"]]:
                lines += ["", "[[jobs]]"] + [f"{k} = {_q(v)}" for k, v in (j or {}).items()]
            continue
        lines += ["", f"[{sec}]"] + [f"{k} = {_until_q(v) if (sec, k) == ('stop', 'until') else _q(v)}"
                                     for k, v in (spec[sec] or {}).items()]
    extra = sorted(set(spec) - set(SECTIONS))
    for sec in extra:                                   # kept, so parse names the unknown section
        lines += ["", f"[{sec}]"]
    return "\n".join(lines).lstrip("\n") + "\n"


def template(folder: Path, name: str) -> str:
    pins = _pins(folder)
    alias = next(iter(pins), "ALIAS")
    return f'''# An experiment of this study: openlab runs its jobs on schedule, counts passes and ends it by [stop].
# docs/experiments.md. Keys are checked strictly; edit, then `openlab experiment status STUDY {name}`.

[experiment]
name = "{name}"
purpose = "what it tests, and how it will be judged at the end"
# plan_step = "step-id"                 # the Roadmap step it implements (of the agent that asks for it)

[[jobs]]
name = "run"
instrument = "{alias}"                  # an alias under [instruments] in study.toml: {", ".join(pins) or "none yet"}
args = []
cron = "0 17 * * 1-5"
tz = "America/New_York"

[pass]
job = "run"                             # a pass: one successful run of this job; or "cycle"

[stop]
passes = 20                             # and/or until = 2026-12-31; max_failures = 3 (failed passes in a row)

[react]
role = "reviewer"
prompt = "Read the experiment's runs and outputs and write its conclusion against its purpose."
budget = {{ usd = 5, minutes = 45, turns = 80 }}
'''


# --- the record: an experiment's nodes, and its state ------------------------------------------------------------------
def _job_actions(rec, name: str) -> list:
    return [n for n in rec.find("action", verb="experiment_job") if (n.payload.get("args") or {}).get("experiment")
            == name]


def nodes(rec, name: str) -> dict:
    """Every node of the experiment on the record: experiment (all versions), start, passes, end, reaction, job
    actions and reaction spawns."""
    if rec is None:
        return {"experiment": [], "start": None, "passes": [], "end": None, "reaction": None, "jobs": [],
                "spawns": []}
    last = lambda k: (rec.find(k, name=name) or [None])[-1]               # noqa: E731
    return {"experiment": rec.find("experiment", name=name), "start": last("experiment_start"),
            "passes": rec.find("experiment_pass", name=name), "end": last("experiment_end"),
            "reaction": last("experiment_reaction"), "jobs": _job_actions(rec, name),
            "spawns": [n for n in rec.find("action", verb="experiment_react_spawn")
                       if (n.payload.get("args") or {}).get("experiment") == name]}


def state_of(ns: dict, file_sha: str = "") -> str:
    if ns["reaction"]:
        return "reacted"
    if ns["end"]:
        return "reacting" if any(s.payload["outcome"].get("ok") for s in ns["spawns"]) else ns["end"].payload["state"]
    if ns["start"]:
        return "running"
    if ns["experiment"] and (not file_sha or ns["experiment"][-1].payload["sha"] == file_sha):
        return "requested"
    return "draft"


def _counts(passes: list) -> dict:
    ok = sum(1 for p in passes if p.payload["ok"])
    run = 0
    for p in reversed(passes):
        if p.payload["ok"]:
            break
        run += 1
    return {"ok": ok, "failed": len(passes) - ok, "in_a_row": run}


def _running(rec, name: str) -> tuple:
    ns = nodes(rec, name)
    if ns["start"] is None:
        raise StudyError(f"the experiment {name} has not started", f"its state: {state_of(ns)}",
                         "the person starts it (openlab experiment start, or approving its inbox request)")
    return ns, ns["start"].payload["spec"]


# --- the crontab -------------------------------------------------------------------------------------------------------
def prefix(ws: Path, rel: str, name: str) -> str:
    return f"{TAG}{ws}::{rel}::{name}::"


def cron_lines(ws: Path, folder: Path, rel: str, spec: dict) -> list:
    """One line per job: at its minutes, `openlab experiment job` for it, serialised per job with flock -n (a run
    still going skips the next), logged to the experiment's cron.log, with the PATH of the shell that started it."""
    q, name, d = shlex.quote, spec["name"], _dir(folder, spec["name"])
    path = os.environ.get("PATH") or "/usr/bin:/bin"
    out = []
    for j in spec["jobs"]:
        cmd = (f"cd {q(str(ws))} && PATH={q(path)} flock -n {q(str(d / (j['name'] + '.lock')))} {q(sys.executable)} "
               f"-m openlab experiment job {q(rel)} {q(name)} {q(j['name'])} --root {q(str(ws))} "
               f">> {q(str(d / 'cron.log'))} 2>&1").replace("%", "\\%")        # % ends a crontab command
        out.append(f"{line_cron(j['cron'], j['tz'])} {cmd} {prefix(ws, rel, name)}{j['name']}")
    return out


def installed(ws: Path, rel: str, name: str) -> list:
    try:
        return schedule.tagged(prefix(ws, rel, name))
    except OSError:
        return []


# --- the person's start ------------------------------------------------------------------------------------------------
def _agent() -> str:
    from ..agents.identity import current
    return current()


def preview(ws: Path, folder: Path, rel: str, spec: dict, sha: str) -> list:
    """What the person reads before signing the start."""
    st, lines = spec["stop"], [f"experiment {spec['name']} on {rel}: {spec['purpose']}",
                               f"spec {FOLDER}/{spec['name']}.toml, sha256 {sha[:16]}"]
    for j in spec["jobs"]:
        try:
            nxt = next_fire(j["cron"], j["tz"])
        except ValueError:
            nxt = "never within a year"
        lines.append(f"job {j['name']}: {j['instrument']} = {j['pin']} ({', '.join(j['capabilities'])}) "
                     f"args {shlex.join(j['args']) or '(none)'}; cron \"{j['cron']}\" {j['tz'] or 'local time'}; "
                     f"next {nxt}")
        if j["secrets"]:
            lines.append(f"  it reads the secrets {', '.join(j['secrets'])}: cron's environment must hold them "
                         "(VAR=value lines at the top of your crontab), or its runs fail")
    pj = spec["pass"]["job"]
    lines.append("pass: " + ("every job ran once since the last pass (cycle)" if pj == "cycle" else
                             f"one successful run of {pj}"))
    lines.append("stop: " + "; ".join(x for x in (
        f"{st['passes']} passes" if st["passes"] else "", f"at {st['until']}" if st["until"] else "",
        f"failed after {st['max_failures']} failed passes in a row" if st["max_failures"] else "no failure limit")
        if x))
    b = spec["react"]["budget"]
    lines.append(f"then a {spec['react']['role']} agent reacts, with ${b['usd']:g}, {b['minutes']:g} min"
                 + (f", {b['turns']} turns" if b.get("turns") else "") + ": " + (spec["react"]["prompt"][:200]
                                                                                   or "(the default prompt)"))
    if spec["plan_step"]:
        lines.append(f"Roadmap: step {spec['plan_step']} of {spec['owner'] or 'the agent that asked'}")
    lines.append(f"installs {len(spec['jobs'])} crontab line(s), tagged {prefix(ws, rel, spec['name'])}<job>, that "
                 "run unattended, outside the sandbox, until it ends")
    return lines


def describe(root, study: str, name: str, spec_sha: str = "") -> tuple:
    """(one line, the preview) of starting an experiment, for the inbox and the CLI."""
    ws, folder, rel = locate(root, study)
    text, sha = read_spec(folder, name)
    spec = parse(text, folder, name, ws)
    lines = preview(ws, folder, rel, spec, sha)
    from .sealing import state, unsealed_line
    if not state(folder)["sealed"]:
        lines.append(unsealed_line(rel))
    if spec_sha and spec_sha != sha:
        lines.insert(0, f"the spec changed since it was requested (sha {spec_sha[:12]} -> {sha[:12]}): approving "
                        "is refused; the agent requests it again")
    return f"start experiment {name} on {rel}", lines


def check_request(root, args: dict) -> dict:
    """An experiment request's args made exact ({study, name, spec_sha}), its spec checked now."""
    study, name = str(args.get("study") or "").strip(), str(args.get("name") or "").strip()
    if not study or not name or set(args) - {"study", "name", "spec_sha"}:
        raise StudyError("an experiment request names the study and the experiment",
                         'args {"study": "studies/tender", "name": "paper_trial"} (or use the experiment_request tool)')
    ws, folder, rel = locate(root, study)
    text, sha = read_spec(folder, name)
    parse(text, folder, name, ws)
    if args.get("spec_sha") and args["spec_sha"] != sha:
        raise StudyError("spec_sha is not the spec file's", f"the file's sha256 is {sha}")
    return {"study": rel, "name": name, "spec_sha": sha}


def request(root, study: str, name: str, reason: str) -> dict:
    """An agent asks the person to start the experiment: the inbox item (kind experiment) and the `experiment` node."""
    from ..agents.store import Store
    if not (reason or "").strip():
        raise StudyError("a request needs a reason", "what the experiment is for, for the person who signs it")
    ws, folder, rel = locate(root, study)
    text, sha = read_spec(folder, name)
    spec = parse(text, folder, name, ws)
    rec = _record(folder)
    if rec.find("experiment_start", name=name):
        raise StudyError(f"the experiment {name} has started", "a started experiment runs the spec it was started with",
                         "a changed experiment is a new one: save it under a new name")
    agent = _agent()
    item = Store(ws).request(agent, "experiment", reason, study=rel, args={"study": rel, "name": name,
                                                                           "spec_sha": sha})
    node = rec.add("experiment", {"name": name, "sha": sha, "spec": spec, "file": f"{FOLDER}/{name}.toml",
                                  "requested_by": agent, "inbox": item, "reason": reason.strip()}, actor="agent")
    from .sealing import state, unsealed_line
    return {"id": node.id, "item": item, "state": "requested", "sha": sha,
            "next": f"the person starts it: approving inbox #{item}, or `openlab experiment start {rel} {name}`",
            **({} if state(folder)["sealed"] else {"note": unsealed_line(rel)})}


def _step(ws: Path, owner: str, step: str, state: str, links: list, note: str) -> str:
    """Move the owner's Roadmap step (openlab.agents.plan); '' when done, else why not. Never fails the caller."""
    if not owner or not step or owner.startswith(("person:", "terminal:", "unverified:", "experiment:")) \
            or owner == "unattributed":
        return "no plan step to move"
    try:
        from ..agents import plan
        from ..agents.store import Store
        plan.step(Store(ws), owner, step, state, links, note)
        return ""
    except Exception as e:                                                  # noqa: BLE001 -- the step is a view
        return getattr(e, "error", None) or str(e)


def _owner(ns: dict, spec: dict) -> str:
    if spec.get("owner"):
        return spec["owner"]
    return ns["experiment"][-1].payload["requested_by"] if ns["experiment"] else ""


def _settle(ws: Path, rel: str, name: str, node) -> None:
    """A start the person made directly closes the experiment's other pending requests (the one being approved now
    is settled by its own decision)."""
    try:
        from ..agents.inbox import DECIDING
        from ..agents.store import Store
        store = Store(ws)
        for i in store.inbox("pending"):
            a = i["args"] or {}
            if i["kind"] == "experiment" and a.get("study") == rel and a.get("name") == name \
                    and i["id"] != DECIDING.get():
                store.decide(i["id"], "cancelled", node.session, f"the person started {name} ({node.id})")
    except Exception:                                                       # noqa: BLE001 -- the start stands
        pass


def start(root, study: str, name: str, reason: str, spec_sha: str = "") -> dict:
    """The person starts the experiment (a signed act): its crontab lines are installed and it runs."""
    from . import instruments, personkey
    ws, folder, rel = locate(root, study)
    rec = _record(folder)
    if personkey.current() is None:
        rec.add("action", {"verb": "experiment_start", "args": {"experiment": name}, "outcome": {
            "ok": False, "error": "an unsigned attempt to start an experiment (recorded as the agent's)"}},
                actor="agent")
        raise StudyError(f"starting the experiment {name} is a person's act, signed with the person key",
                         "it runs unattended, outside the sandbox, on the user's crontab; this process holds no "
                         "unlocked person key",
                         "an agent files experiment_request (agent_request kind experiment); the person approves it "
                         "in the inbox or runs `openlab experiment start STUDY NAME` in their own terminal")
    if not (reason or "").strip():
        raise StudyError("starting an experiment needs a reason")
    text, sha = read_spec(folder, name)
    if spec_sha and spec_sha != sha:
        raise StudyError(f"the spec of {name} changed since it was requested",
                         f"requested sha {spec_sha[:12]}, the file's {sha[:12]}",
                         "deny the request; the agent requests the current spec again")
    spec = parse(text, folder, name, ws)
    ns = nodes(rec, name)
    if ns["start"] is not None:
        raise StudyError(f"the experiment {name} has already started", f"its state: {state_of(ns)}",
                         "a changed experiment is a new one: save it under a new name")
    if spec["stop"]["until"] and _parse_iso(spec["stop"]["until"]) <= _now():
        raise StudyError(f"[stop] until {spec['stop']['until']} has passed", "it would end at once")
    reg = instruments.Registry(ws)
    for pin in sorted({j["pin"] for j in spec["jobs"]}):
        v = instruments.verify(reg, instruments.resolve(reg, pin))
        if not v["ok"]:
            raise StudyError(f"{pin} is not the version that was registered", instruments.drift(v),
                             "a version never changes: register it again, pin it, and save a new experiment")
    exp = ns["experiment"][-1] if ns["experiment"] else None
    if exp is None or exp.payload["sha"] != sha:
        exp = rec.add("experiment", {"name": name, "sha": sha, "spec": spec, "file": f"{FOLDER}/{name}.toml",
                                     "requested_by": exp.payload["requested_by"] if exp else _agent(),
                                     "inbox": 0, "reason": reason.strip()}, actor="person")
    lines = cron_lines(ws, folder, rel, spec)
    node = rec.add("experiment_start", {"name": name, "sha": sha, "spec": spec, "lines": lines,
                                        "reason": reason.strip()}, actor="person", links=[("starts", exp.id)])
    try:
        schedule.replace_tagged(prefix(ws, rel, name), lines)
    except Exception as e:                                                  # noqa: BLE001 -- recorded, then raised
        _end(ws, folder, rel, rec, name, "failed", f"its crontab lines could not be installed: {e}", "openlab",
             "openlab", react=False)                # nothing ran, so no agent is fired to react
        raise StudyError(f"the crontab lines of {name} could not be installed", str(e),
                         "check `crontab -l` works for this user; the experiment ended failed") from None
    _settle(ws, rel, name, node)
    moved = _step(ws, _owner(nodes(rec, name), spec), spec["plan_step"], "active", [node.id],
                  f"experiment {name} started")
    return {"id": node.id, "state": "running", "lines": lines, "plan_step": moved or "moved to active",
            "summary": f"started {name}: {len(lines)} crontab line(s)"}


# --- running: one job ----------------------------------------------------------------------------------------------------
@contextmanager
def _locked(folder: Path, name: str):
    with open(_dir(folder, name) / "experiment.lock", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _pass_due(spec: dict, ns: dict) -> Optional[list]:
    """The job actions the next pass rests on, when the pass rule is met by the latest one, else None."""
    acts, pj = ns["jobs"], spec["pass"]["job"]
    if not acts:
        return None
    last_pass = ns["passes"][-1].seq if ns["passes"] else 0
    window = [a for a in acts if a.seq > last_pass]
    if pj != "cycle":
        latest = window[-1] if window else None
        return [latest] if latest is not None and latest.payload["args"]["job"] == pj else None
    names = [j["name"] for j in spec["jobs"]]
    latest = {}
    for a in window:
        latest[a.payload["args"]["job"]] = a
    return [latest[n] for n in names] if all(n in latest for n in names) else None


def _stop_reason(spec: dict, counts: dict, t: dt.datetime) -> Optional[tuple]:
    st = spec["stop"]
    if st["passes"] and counts["ok"] >= st["passes"]:
        return "concluded", f"stop rule met: {counts['ok']} passes"
    if st["max_failures"] and counts["in_a_row"] >= st["max_failures"]:
        return "failed", f"{counts['in_a_row']} failed passes in a row (max_failures = {st['max_failures']})"
    if st["until"] and t >= _parse_iso(st["until"]):
        return "concluded", f"stop rule met: until {st['until']}"
    return None


def job(root, study: str, name: str, job_name: str, now: bool = False, at: Optional[dt.datetime] = None) -> dict:
    """Run one job of a running experiment, as its crontab line does: at a minute its schedule fires (in its zone),
    once per such minute; `now` runs it at once (a person's act; its pass is marked off schedule). Then a pass when the
    pass rule is met, and the end when the stop rule is; at the end the lines go and the reaction agent is fired."""
    from ..agents.identity import acting_as
    from . import instruments
    ws, folder, rel = locate(root, study)
    rec = _record(folder)
    ns, spec = _running(rec, name)
    j = next((x for x in spec["jobs"] if x["name"] == job_name), None)
    if j is None:
        raise StudyError(f"{name} has no job {job_name}", f"jobs: {', '.join(x['name'] for x in spec['jobs'])}")
    t = at or _now()
    wall = _wall(t, j["tz"]).replace(second=0, microsecond=0)
    due_at = next((m for m in (wall - dt.timedelta(minutes=k) for k in range(GRACE_MINUTES + 1))
                   if schedule.matches(j["cron"], m)), None)
    if not now and due_at is None:
        return {"ran": False, "why": f"not due: {job_name} fires at \"{j['cron']}\" {j['tz'] or 'local time'}"}
    slot = f"now {_iso(t)}" if now else f"{due_at.isoformat(timespec='minutes')} {j['tz'] or 'local'}"
    ended = None
    with _locked(folder, name):
        ns = nodes(rec, name)
        if ns["end"] is not None:                   # a line left behind: it runs nothing, and goes
            try:
                gone = len(schedule.replace_tagged(prefix(ws, rel, name), []))
            except Exception:                                               # noqa: BLE001
                gone = 0
            return {"ran": False, "why": f"{name} has ended ({ns['end'].payload['state']})", "removed": gone}
        if any(a.payload["args"].get("job") == job_name and a.payload["args"].get("slot") == slot for a in ns["jobs"]):
            return {"ran": False, "why": f"{job_name} already ran at {slot}"}
        why = _stop_reason(spec, _counts(ns["passes"]), t)
        if why is None:
            with acting_as(_session(name)):
                try:
                    r = instruments.run(ws, j["instrument"], j["args"], study=rel,
                                        note=f"experiment {name}, job {job_name}", timeout=j["timeout"])
                    run_id, reg_id, code = r.get("study_run", ""), r["id"], r["payload"]["exit"]
                    error = "" if code == 0 else ("timed out" if r["payload"]["timed_out"] else f"exit {code}")
                except Exception as e:                                      # noqa: BLE001 -- a failed run, recorded
                    run_id, reg_id, code, error = "", "", None, f"refused: {getattr(e, 'error', None) or e}"
            act = rec.add("action", {"verb": "experiment_job", "args": {
                "experiment": name, "job": job_name, "slot": slot, "off_schedule": bool(now)},
                "outcome": {"ok": not error, "id": run_id or None, "registry_run": reg_id, "exit": code,
                            "error": error}},
                actor="openlab", session=_session(name), links=[("did", run_id)] if run_id else ())
            ns = nodes(rec, name)
            due = _pass_due(spec, ns)
            if due is not None:
                runs = [a.payload["outcome"]["id"] for a in due if a.payload["outcome"].get("id")]
                errs = [f"{a.payload['args']['job']}: {a.payload['outcome']['error']}" for a in due
                        if a.payload["outcome"].get("error")]
                rec.add("experiment_pass", {"name": name, "n": len(ns["passes"]) + 1, "ok": not errs,
                                            "job": spec["pass"]["job"], "runs": runs, "error": "; ".join(errs),
                                            "off_schedule": any(a.payload["args"].get("off_schedule") for a in due)},
                        actor="openlab", session=_session(name),
                        links=[("on", ns["start"].id)] + [("rests_on", x) for x in runs])
                ns = nodes(rec, name)
            why = _stop_reason(spec, _counts(ns["passes"]), t)
            out = {"ran": True, "action": act.id, "run": run_id, "exit": code, "error": error,
                   "pass": ns["passes"][-1].id if due is not None else ""}
        else:
            out = {"ran": False, "why": why[1]}
        if why is not None:
            ended, err = _end(ws, folder, rel, rec, name, why[0], why[1], "openlab", "openlab", react=False)
            out["ended"] = {"state": why[0], "reason": why[1], "id": ended.id, **({"warning": err} if err else {})}
    if ended is not None:
        out["reaction"] = fire(ws, folder, rel, name)
    return out


def _end(ws, folder, rel, rec, name: str, state: str, reason: str, by: str, actor: str, react: bool = True):
    """Remove the lines, then write the end (under the experiment's lock when the caller holds it). The end is written
    even when the crontab cannot be edited: a line left behind finds the experiment ended when it fires, runs nothing
    and tries to remove the lines again. Returns (the end node, the crontab's error or '')."""
    ns = nodes(rec, name)
    err = ""
    try:
        schedule.replace_tagged(prefix(ws, rel, name), [])
    except Exception as e:                                                  # noqa: BLE001 -- reported, not fatal
        err = f"its crontab lines could not be removed: {e}"
    c = _counts(ns["passes"])
    node = rec.add("experiment_end", {"name": name, "state": state, "reason": reason, "by": by,
                                      "passes_ok": c["ok"], "passes_failed": c["failed"]},
                   actor=actor, session=_session(name) if actor == "openlab" else "",
                   links=[("ends", ns["start"].id)])
    if react:
        fire(ws, folder, rel, name)
    return node, err


def kill(root, study: str, name: str, reason: str) -> dict:
    """End a running experiment now (an agent or the person, with the reason): its lines go at once (before waiting
    for a job that is running), the end names who killed it, and the reaction agent is fired."""
    from . import personkey
    if not (reason or "").strip():
        raise StudyError("killing an experiment needs a reason", "it is on the record, and the reaction agent reads it")
    ws, folder, rel = locate(root, study)
    rec = _record(folder)
    ns, spec = _running(rec, name)
    if ns["end"] is not None:
        raise StudyError(f"the experiment {name} has already ended", f"{ns['end'].payload['state']}: "
                         f"{ns['end'].payload['reason']}")
    try:
        schedule.replace_tagged(prefix(ws, rel, name), [])  # no new job starts from here on
    except Exception:                                       # noqa: BLE001 -- the end below says so
        pass
    by = _agent()
    with _locked(folder, name):                             # a job still running finishes and is counted first
        ns = nodes(rec, name)
        if ns["end"] is not None:
            return {"id": ns["end"].id, "state": ns["end"].payload["state"], "reaction": fire(ws, folder, rel, name),
                    "note": f"it ended on its own first: {ns['end'].payload['reason']}"}
        node, err = _end(ws, folder, rel, rec, name, "killed", f"killed by {by}: {reason.strip()}", by,
                         "person" if personkey.current() is not None else "agent", react=False)
    return {"id": node.id, "state": "killed", "reaction": fire(ws, folder, rel, name), **({"warning": err} if err else {})}


# --- the reaction -----------------------------------------------------------------------------------------------------
def _spawn(root: Path, **kw) -> dict:
    if SPAWNER is not None:
        return SPAWNER(root, **kw)
    from ..agents.spawn import spawn
    return spawn(root, start=os.environ.get("OPENLAB_EXPERIMENT_SPAWN") != "register", **kw)


def reaction_prompt(rel: str, spec: dict, ns: dict) -> str:
    """The task of the agent fired at the experiment's end (its role's prompt is in its brief)."""
    end, c = ns["end"].payload, _counts(ns["passes"])
    runs = [a for a in ns["jobs"]][-PROMPT_RUNS:]
    st = spec["stop"]
    lines = [f"The experiment `{spec['name']}` of the study {rel} has ended: {end['state']} ({end['reason']}).",
             f"Its purpose: {spec['purpose']}",
             f"Passes: {c['ok']} ok and {c['failed']} failed" + (f", of {st['passes']} wanted" if st["passes"] else "")
             + (f"; until {st['until']}" if st["until"] else "") + f". Started {ns['start'].at}, ended {ns['end'].at}.",
             "Its runs, oldest first (job, slot, exit or error, the run on the study's record, its output's id):"]
    lines += [f"- {a.payload['args']['job']} {a.payload['args']['slot']}: "
              + (a.payload["outcome"].get("error") or f"exit {a.payload['outcome'].get('exit')}")
              + f"; record {a.payload['outcome'].get('id') or '(none)'}"
              + (f"; output {a.payload['outcome']['registry_run']}" if a.payload["outcome"].get("registry_run") else "")
              for a in runs]
    if len(ns["jobs"]) > len(runs):
        lines.append(f"(and {len(ns['jobs']) - len(runs)} earlier runs: openlab_tool(\"experiment_status\"))")
    lines += ["",
              "Read a run's logs and outputs by its output id: openlab_tool(\"instrument_output\", {\"run\": \"<output "
              "id>\"}) (or `openlab instrument output <output id> [FILE]`); a reaction rests on the record ids. The whole "
              f"picture with openlab_tool(\"experiment_status\", {{\"study\": \"{rel}\", \"name\": \"{spec['name']}\"}}).",
              "1. Report your plan first with openlab_agent_plan (the Roadmap): reading the runs, judging, recording.",
              "2. Judge the outcome against the purpose. Record what the record allows on the way (a question, a "
              "hypothesis, a dead end, a claim grounded in results), and say what you could not establish.",
              "3. Close with openlab_tool(\"experiment_react\", {\"study\": \"" + rel + "\", \"name\": \"" + spec["name"]
              + "\", \"verdict\": \"met\" | \"not_met\" | \"inconclusive\", \"conclusion\": \"...\", \"rests_on\": "
              "[the pass and run ids it rests on], \"records\": [the ids you wrote]}).",
              ]
    if spec["plan_step"]:
        lines.append(f"It implements step {spec['plan_step']} of {_owner(ns, spec) or 'its owner'}'s Roadmap; openlab "
                     "moves that step when your reaction is recorded.")
    if spec["react"]["prompt"]:
        lines += ["", spec["react"]["prompt"]]
    return "\n".join(lines)


def fire(ws: Path, folder: Path, rel: str, name: str) -> dict:
    """Spawn the reaction agent of an ended experiment (once: a spawn that worked is not repeated)."""
    rec = _record(folder)
    ns = nodes(rec, name)
    if ns["end"] is None:
        raise StudyError(f"the experiment {name} has not ended", "the reaction agent is fired when it ends")
    done = next((s for s in ns["spawns"] if s.payload["outcome"].get("ok")), None)
    if done is not None:
        return {"ok": True, "agent": done.payload["outcome"]["agent"], "again": False}
    spec = ns["start"].payload["spec"]
    r = spec["react"]
    try:
        a = _spawn(ws, prompt=reaction_prompt(rel, spec, ns), role=r["role"], study=rel,
                   question=f"conclude the experiment {name}: {spec['purpose']}"[:300], budget=dict(r["budget"]))
        outcome = {"ok": True, "agent": a["id"]}
    except Exception as e:                                                  # noqa: BLE001 -- recorded; react --spawn
        outcome = {"ok": False, "error": getattr(e, "error", None) or str(e)}
    rec.add("action", {"verb": "experiment_react_spawn", "args": {"experiment": name, "role": r["role"]},
                       "outcome": outcome}, actor="openlab", session=_session(name))
    return outcome


def respawn(root, study: str, name: str) -> dict:
    ws, folder, rel = locate(root, study)
    return fire(ws, folder, rel, name)


def react(root, study: str, name: str, verdict: str, conclusion: str, rests_on: list, records: Optional[list] = None
          ) -> dict:
    """Record the conclusion of an ended experiment; then the owner's plan step moves to done (failed, for a failed
    experiment)."""
    if verdict not in EXPERIMENT_VERDICTS:
        raise StudyError(f"a verdict is one of {', '.join(EXPERIMENT_VERDICTS)}", f"got {verdict!r}")
    ws, folder, rel = locate(root, study)
    rec = _record(folder)
    ns = nodes(rec, name)
    if ns["end"] is None:
        raise StudyError(f"the experiment {name} has not ended", f"its state: {state_of(ns)}",
                         "its reaction is written after it ends (its stop rule, or a kill)")
    rests = [str(x) for x in (rests_on or [])]
    if not rests:
        raise StudyError("a reaction rests on the experiment's passes or runs",
                         f"e.g. rests_on [{', '.join(p.id for p in ns['passes'][-3:]) or ns['end'].id}]")
    agent = _agent()
    node = rec.add("experiment_reaction", {"name": name, "agent": agent, "verdict": verdict,
                                           "conclusion": conclusion.strip()}, actor="agent",
                   links=[("on", ns["end"].id)] + [("rests_on", x) for x in rests]
                   + [("records", str(x)) for x in (records or [])])
    spec = ns["start"].payload["spec"]
    moved = _step(ws, _owner(ns, spec), spec["plan_step"], "failed" if ns["end"].payload["state"] == "failed"
                  else "done", [ns["start"].id, ns["end"].id, node.id], f"experiment {name}: {verdict}")
    return {"id": node.id, "state": "reacted", "plan_step": moved or "moved"}


# --- reading ------------------------------------------------------------------------------------------------------------
def _job_view(a) -> dict:
    p, o = a.payload["args"], a.payload["outcome"]
    return {"job": p["job"], "slot": p["slot"], "at": a.at, "ok": bool(o.get("ok")), "exit": o.get("exit"),
            "run": o.get("id") or "", "registry_run": o.get("registry_run") or "", "error": o.get("error") or "",
            "off_schedule": bool(p.get("off_schedule"))}


def _registry_run(rec, run_id: str) -> str:
    try:
        return rec.get(run_id).payload.get("registry_run") or ""
    except StudyError:
        return ""


def view(ws: Path, folder: Path, rel: str, name: str, rec=None, store=None, at: Optional[dt.datetime] = None) -> dict:
    """An experiment as `status`, the tools and openlab show read it: state, spec, progress, jobs (next time, last
    runs), passes, failures, the end, the reaction agent and its conclusion."""
    rec = rec if rec is not None else _record(folder, create=False)
    t = at or _now()
    out: dict = {"study": rel, "name": name, "file": f"{FOLDER}/{name}.toml"}
    from .sealing import state, unsealed_line
    seal = state(folder)
    out["study_seal"] = {**seal, **({} if seal["sealed"] else {"line": unsealed_line(rel)})}
    sha, spec, err = "", None, ""
    try:
        text, sha = read_spec(folder, name)
    except StudyError:
        text = None
    ns = nodes(rec, name)
    if ns["start"] is not None:
        spec = ns["start"].payload["spec"]
    elif text is not None:
        try:
            spec = parse(text, folder, name, ws)
        except StudyError as e:
            err = e.error + (f" ({e.meaning})" if e.meaning else "")
            spec = ns["experiment"][-1].payload["spec"] if ns["experiment"] else None
    elif ns["experiment"]:
        spec = ns["experiment"][-1].payload["spec"]
    state = state_of(ns, sha)
    out.update(state=state, spec=spec, spec_error=err, sha=sha,
               edited=bool(ns["start"] and sha and sha != ns["start"].payload["sha"]),
               missing_file=text is None)
    if spec is None:
        return out
    out.update(purpose=spec["purpose"], plan_step=spec["plan_step"], owner=_owner(ns, spec))
    exp = ns["experiment"][-1] if ns["experiment"] else None
    out["request"] = {"id": exp.id, "by": exp.payload["requested_by"], "inbox": exp.payload["inbox"],
                      "at": exp.at} if exp else None
    if exp and exp.payload["inbox"] and store is not None:
        it = store.item(exp.payload["inbox"])
        out["request"]["status"] = it["status"] if it else "gone"
    c = _counts(ns["passes"])
    st = spec["stop"]
    started = _parse_iso(ns["start"].at) if ns["start"] else None
    ended = _parse_iso(ns["end"].at) if ns["end"] else None
    until = _parse_iso(st["until"]) if st["until"] else None
    elapsed = ((ended or t) - started).total_seconds() if started else 0
    out["progress"] = {"passes_ok": c["ok"], "passes_failed": c["failed"], "in_a_row": c["in_a_row"],
                       "target": st["passes"], "until": st["until"], "max_failures": st["max_failures"],
                       "started": ns["start"].at if started else "", "ended": ns["end"].at if ended else "",
                       "elapsed_s": round(elapsed), "last_pass": ns["passes"][-1].at if ns["passes"] else "",
                       "pass_fraction": min(1.0, c["ok"] / st["passes"]) if st["passes"] else None,
                       "time_fraction": (min(1.0, max(0.0, elapsed / (until - started).total_seconds()))
                                         if started and until and until > started else None)}
    jobs = []
    lines = installed(ws, rel, name) if state == "running" else []
    for j in spec["jobs"]:
        acts = [a for a in ns["jobs"] if a.payload["args"]["job"] == j["name"]]
        row = {k: j[k] for k in ("name", "instrument", "pin", "args", "cron", "tz", "capabilities")}
        row["runs"] = [_job_view(a) for a in acts[-RUNS_SHOWN:]][::-1]
        for r in row["runs"]:
            r["registry_run"] = r["registry_run"] or (_registry_run(rec, r["run"]) if r["run"] else "")
        row["n_runs"], row["n_failed"] = len(acts), sum(1 for a in acts if not a.payload["outcome"].get("ok"))
        row["installed"] = any(l.endswith(f"::{j['name']}") for l in lines)
        try:
            row["next"] = next_fire(j["cron"], j["tz"], t) if state == "running" else ""
        except ValueError:
            row["next"] = ""
        jobs.append(row)
    out["jobs"] = jobs
    out["pass_rule"] = spec["pass"]["job"]
    out["passes"] = [{"id": p.id, "n": p.payload["n"], "ok": p.payload["ok"], "at": p.at, "runs": p.payload["runs"],
                      "error": p.payload.get("error", ""), "off_schedule": p.payload.get("off_schedule", False)}
                     for p in ns["passes"]][-50:]
    out["failures"] = [_job_view(a) for a in ns["jobs"] if not a.payload["outcome"].get("ok")][-10:][::-1]
    out["start"] = {"id": ns["start"].id, "at": ns["start"].at, "reason": ns["start"].payload["reason"],
                    "by": ns["start"].session} if ns["start"] else None
    out["end"] = ({"id": ns["end"].id, "at": ns["end"].at, **{k: ns["end"].payload[k] for k in ("state", "reason",
                                                                                                "by")}}
                  if ns["end"] else None)
    spawn = ns["spawns"][-1].payload["outcome"] if ns["spawns"] else None
    reaction = {"agent": (spawn or {}).get("agent", ""), "spawn_error": (spawn or {}).get("error", ""),
                "role": spec["react"]["role"], "budget": spec["react"]["budget"]}
    if reaction["agent"] and store is not None:
        a = store.get(reaction["agent"])
        reaction["status"] = a["status"] if a else "unknown"
        reaction["usd"] = round(((a or {}).get("spent") or {}).get("usd") or 0, 4)
    if ns["reaction"]:
        p = ns["reaction"].payload
        reaction.update(id=ns["reaction"].id, at=ns["reaction"].at, by=p["agent"], verdict=p["verdict"],
                        conclusion=p["conclusion"],
                        records=[d for _, rel_, d in rec.links(ns["reaction"].id) if rel_ == "records"],
                        rests_on=[d for _, rel_, d in rec.links(ns["reaction"].id) if rel_ == "rests_on"])
    out["reaction"] = reaction
    return out


def names(folder: Path) -> list:
    """The experiments of a study: its spec files and the names on its record."""
    found = {p.stem for p in (Path(folder) / FOLDER).glob("*.toml") if NAME.fullmatch(p.stem)}
    rec = _record(folder, create=False)
    if rec is not None:
        found |= {n.payload["name"] for n in rec.find("experiment")}
    return sorted(found)


def status(root, study: str, name: str) -> dict:
    from ..agents.store import Store
    ws, folder, rel = locate(root, study)
    if name not in names(folder):
        raise StudyError(f"no experiment `{name}` in {rel}", f"its experiments: {', '.join(names(folder)) or 'none'}")
    return view(ws, folder, rel, name, store=Store(ws))


def overview(root, folders: Optional[list] = None, store=None) -> list:
    """Every experiment of the workspace's studies, study by study (a study that cannot be read is one `error` row)."""
    from .workspace import studies
    ws = _ws(root)
    if store is None:
        try:
            from ..agents.store import Store
            store = Store(ws) if (ws / ".openlab" / "control.db").is_file() else None
        except Exception:                                                   # noqa: BLE001
            store = None
    out = []
    for f in (folders if folders is not None else studies(ws)):
        f = Path(f).resolve()
        try:
            rel = str(f.relative_to(ws))
        except ValueError:
            continue
        try:
            ns = names(f)
            rec = _record(f, create=False) if ns else None
            for n in ns:
                out.append(view(ws, f, rel, n, rec=rec, store=store))
        except Exception as e:                                              # noqa: BLE001 -- one bad study is shown
            out.append({"study": rel, "name": "", "state": "error", "spec_error": str(e)})
    return out


def instruments_overview(root, folders: Optional[list] = None) -> list:
    """Every registered version: capabilities, runtime, runs, its last run, the studies that pin it (and under which
    alias) and the head of its usage (the whole usage: `instrument show --usage`)."""
    from . import instruments
    from .workspace import studies
    ws = _ws(root)
    if not instruments.Registry.exists(ws):
        return []
    reg = instruments.Registry(ws)
    pinned: dict = {}
    for f in (folders if folders is not None else studies(ws)):
        try:
            rel = str(Path(f).resolve().relative_to(ws))
            for alias, pin in _pins(Path(f)).items():
                pinned.setdefault(pin, []).append({"study": rel, "alias": alias})
        except Exception:                                                   # noqa: BLE001 -- a broken study.toml
            continue
    last: dict = {}
    for r in reg.find("instrument_run"):
        last[r.payload["instrument"]] = {"id": r.id, "at": r.payload["ended"], "exit": r.payload["exit"],
                                         "args": r.payload["args"], "study": r.payload["study"]}
    out = []
    for item in instruments.listing(ws):
        for v in item["versions"]:
            node = reg.get(v["node"])
            u = instruments.usage_view(reg, node.payload)
            out.append({**v, "name": item["name"], "last_run": last.get(v["id"]), "pinned_by": pinned.get(v["id"], []),
                        "usage_head": u["head"], "usage_lines": u["text"].splitlines()[:12],
                        "usage_more": max(0, len(u["text"].splitlines()) - 12)})
    return out


def write(root, study: str, name: str, text: str) -> dict:
    """Write (or replace) a draft's file, checked first; refused once the experiment has been requested and started."""
    ws, folder, rel = locate(root, study)
    rec = _record(folder, create=False)
    if rec is not None and rec.find("experiment_start", name=name):
        raise StudyError(f"the experiment {name} has started", "it runs the spec it was started with",
                         "a changed experiment is a new one: write it under a new name")
    spec = parse(text, folder, name, ws)
    p = spec_file(folder, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return {"file": str(p.relative_to(ws)), "state": "draft", "sha": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "spec": spec, "next": f"ask the person to start it: experiment_request(study={rel!r}, name={name!r}, "
                                  "reason=...)"}
