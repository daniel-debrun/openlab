"""A study's forward runs on a schedule kept in the user's crontab.

A study declares `[schedule] cron = "7 * * * *"` in study.toml; its line, tagged ``# openlab:<parent>::<study>``,
runs `openlab study forecast` and then `openlab study fill` on the study folder with this Python, logged to
`<study>/.openlab/cron.log` (`for_study`). `on` writes that one line into the user's crontab, so `off` and
`status` find exactly that line and nothing else in the crontab is touched; each run is serialised with flock.
`status` reads the declared schedule, whether the installed line matches it and the next run time."""
from __future__ import annotations

import datetime as dt
import os
import shlex
import subprocess
import sys
from pathlib import Path

_RANGES = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 7))
_NAMES = ("minute", "hour", "day of month", "month", "day of week")


def _field(text: str, lo: int, hi: int, name: str) -> set:
    out = set()
    for part in text.split(","):
        rng, _, step = part.partition("/")
        if step and not step.isdigit():
            raise ValueError(f"cron {name}: bad step {part!r}")
        if rng == "*":
            a, b = lo, hi
        elif "-" in rng:
            a, b = (int(x) for x in rng.split("-", 1))
        elif rng.isdigit():
            a = b = int(rng)
            if step:
                b = hi
        else:
            raise ValueError(f"cron {name}: {part!r} is not a number, range, list or */step")
        if not (lo <= a <= hi and lo <= b <= hi and a <= b):
            raise ValueError(f"cron {name}: {part!r} is outside {lo}-{hi}")
        out |= set(range(a, b + 1, int(step) if step else 1))
    return out


def parse_cron(expr: str) -> tuple:
    """The five fields as sets of allowed values; raises ValueError naming the field that is wrong."""
    parts = expr.split()
    if len(parts) != 5:
        raise ValueError(f"cron needs 5 fields (minute hour day month weekday), got {len(parts)}: {expr!r}")
    sets = [_field(p, lo, hi, n) for p, (lo, hi), n in zip(parts, _RANGES, _NAMES)]
    sets[4] = {d % 7 for d in sets[4]}                                    # 7 is Sunday as well as 0
    return tuple(sets) + (parts[2] != "*", parts[4] != "*")


def next_run(expr: str, after: dt.datetime | None = None) -> dt.datetime:
    """The first minute strictly after `after` (local time, default now) that the expression matches."""
    mi, hr, dom, mon, dow, dom_set, dow_set = parse_cron(expr)
    t = (after or dt.datetime.now()).replace(second=0, microsecond=0) + dt.timedelta(minutes=1)
    for _ in range(366 * 24 * 60):
        day_ok = ((t.day in dom) or (t.isoweekday() % 7 in dow)) if dom_set and dow_set else \
                 (t.day in dom and t.isoweekday() % 7 in dow)
        if t.month in mon and day_ok and t.hour in hr and t.minute in mi:
            return t
        t += dt.timedelta(minutes=1)
    raise ValueError(f"cron {expr!r} never matches within a year")


def _tag(root: Path, doc: str) -> str:
    return f"# openlab:{root}::{doc}"


def cron_line(root: Path, doc: str, spec: dict, log: Path) -> str:
    root = Path(root)
    lock = log.with_suffix(".lock")
    return (f"{spec['cron']} cd {shlex.quote(str(root))} && flock -n {shlex.quote(str(lock))} {spec['command']} "
            f">> {shlex.quote(str(log))} 2>&1 {_tag(root, doc)}")


def for_study(folder: Path, cron: str) -> tuple:
    """(root, doc, spec, cron log) of a study's schedule, for `on`, `off` and `status`: keyed by the study's folder,
    its line runs forecast, then fill (whether or not a forecast was due)."""
    folder = Path(folder).resolve()
    openlab, f = f"{shlex.quote(sys.executable)} -m openlab study", shlex.quote(str(folder))
    cmd = "sh -c " + shlex.quote(f"{openlab} forecast {f}; {openlab} fill {f}")
    return folder.parent, folder.name, {"cron": cron, "output": "", "command": cmd}, folder / ".openlab" / "cron.log"


def for_monitor(folder: Path, cron: str) -> tuple:
    """(root, doc, spec, log) of a study's monitoring line, apart from its forward line: it runs `openlab study
    monitor`, tagged ``# openlab:<parent>::<study>:watch`` (the tag of earlier installs, so `on` replaces one) and
    logged to `<study>/.openlab/monitor.log`."""
    folder = Path(folder).resolve()
    cmd = f"{shlex.quote(sys.executable)} -m openlab study monitor {shlex.quote(str(folder))}"
    return folder.parent, f"{folder.name}:watch", {"cron": cron, "output": "", "command": cmd}, \
        folder / ".openlab" / "monitor.log"


def _program() -> str:
    """The crontab program; `OPENLAB_CRONTAB` names another (a test's fake), so no check touches the real one."""
    return os.environ.get("OPENLAB_CRONTAB") or "crontab"


def _crontab() -> list:
    r = subprocess.run([_program(), "-l"], capture_output=True, text=True)
    return r.stdout.splitlines() if r.returncode == 0 else []


def _write(lines: list) -> None:
    subprocess.run([_program(), "-"], input="\n".join(lines) + "\n", text=True, check=True)


def installed(root: Path, doc: str) -> str | None:
    tag = _tag(Path(root), doc)
    return next((l for l in _crontab() if l.endswith(tag)), None)


def on(root: Path, doc: str, spec: dict, log: Path) -> str:
    """Install (or replace) this study's line; returns it."""
    tag, line = _tag(Path(root), doc), cron_line(root, doc, spec, log)
    log.parent.mkdir(parents=True, exist_ok=True)
    _write([l for l in _crontab() if not l.endswith(tag)] + [line])
    return line


def off(root: Path, doc: str) -> bool:
    tag = _tag(Path(root), doc)
    lines = _crontab()
    keep = [l for l in lines if not l.endswith(tag)]
    if len(keep) != len(lines):
        _write(keep)
    return len(keep) != len(lines)


def status(root: Path, doc: str, spec: dict | None, cron_log: Path) -> dict:
    """Declared vs installed, and the next run."""
    line = installed(root, doc)
    out = {"kind": "schedule", "declared": spec, "installed": line is not None,
           "matches": bool(spec) and line == cron_line(root, doc, spec, cron_log), "line": line,
           "cron_log": str(cron_log)}
    if spec and line:
        out["next_run"] = next_run(spec["cron"]).isoformat(timespec="minutes")
    out["state"] = ("off" if not line else "on" if out["matches"] else "stale")
    return out
