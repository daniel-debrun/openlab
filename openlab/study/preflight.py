"""Module checks: a broken model.py fails in seconds, before a full run spends minutes (after MARS's per-module unit
tests and MLEvolve's pre-execution review).

`openlab study check FOLDER [--code branches/<id>]` runs, on a snapshot of the code:

| check | what it does |
|---|---|
| import | model.py (and the modules it imports) imports in the sandbox, and defines Model with fit and predict |
| lint | the code names no path a run must not read (the vault, the record, the holdout's labels, `..`), and opens no network when the sandbox has none |
| smoke | the runner's own child job on a little data: a sample of research rows (static), or three steps from the middle of the research segment (temporal: the first steps have no labels to fit on yet); under `[sandbox] timeout`, else 120 s |
| determinism | the smoke job again gives the same predictions |

The outcome is a `module_check` node keyed by the code's shas; a run of that exact code is refused while its latest
check is red. A check scores no research rows and spends no trial.
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path

import polars as pl

from .errors import StudyError

SMOKE_TIMEOUT = 120.0
FORBIDDEN = [(re.compile(r"\.openlab\b"), "the study's record (.openlab)"),
             (re.compile(r"\bvault\b", re.I), "the vault of holdout labels"),
             (re.compile(r"""['"][^'"]*\.\./"""), "a path outside the study (..)"),
             (re.compile(r"\bholdout\b[^\n]*\.(parquet|csv)", re.I), "a holdout data file")]
NETWORK = re.compile(r"^\s*(import|from)\s+(requests|urllib|httpx|socket|aiohttp)\b", re.M)


def lint(code_dir: Path, network: bool) -> list:
    """[(file, line, why)] of what the code must not do."""
    out = []
    for p in sorted(Path(code_dir).glob("*.py")):
        text = p.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            for rx, why in FORBIDDEN:
                if rx.search(line) and not line.lstrip().startswith("#"):
                    out.append((p.name, i, why))
        if not network:
            for m in NETWORK.finditer(text):
                out.append((p.name, text[:m.start()].count("\n") + 1, f"a network library ({m.group(2)}) with "
                                                                       "[sandbox] network = false"))
    return out


def import_job(stage: Path, *, spec: dict, seed: int) -> Path:
    """In the sandbox: import model and check its interface."""
    import importlib
    import json
    m = importlib.import_module("model")
    M = getattr(m, "Model", None)
    missing = [] if M is not None else ["Model"]
    if M is not None:
        missing += [f"Model.{a}" for a in ("fit", "predict") if not callable(getattr(M, a, None))]
    out = stage / "import.json"
    out.write_text(json.dumps({"missing": missing}))
    return out


def _failed(res, limit: float, default: str) -> str:
    """Why a check's child failed, in one line: a timeout first (the child's last line after it), else the
    child's last line ("no result.json" only when the child said nothing else)."""
    lines = [x for x in (res.stderr or "").strip().splitlines() if x.strip()]
    said = [x for x in lines if not x.startswith(("timed out after", "the child wrote no result.json"))]
    if getattr(res, "timed_out", False):
        return (f"timed out after {limit:g} s (the check's limit)" + (f": {said[-1]}" if said else ""))[:300]
    return ((said or lines)[-1] if lines else default)[:300]


def _smoke(r, spec) -> tuple:
    """(seconds, predictions frame) of the runner's child job on a little data."""
    from . import runner, splits
    t0 = time.monotonic()
    if spec.mode == "static":
        (name,), target = spec.data, spec.target.get("column")
        df = runner.outputs(r.frames()[name], target, f"data.{name}.{target}")
        research, _ = splits.carve(df, spec)
        R = df[research]
        n = min(R.height, max(100, math.ceil(0.05 * R.height)))
        S = R.sample(n=n, seed=0, shuffle=True)
        train, valid = S.head(max(1, n - min(20, n // 4))), S.tail(min(20, n // 4) or 1)
        with r.stage({"train": train, "valid": valid.drop(target)}) as st:
            res = r.child(st, "static_job", must=False, target=target, pack=False)
            P = pl.read_parquet(st / "predictions.parquet") if res.ok else None
    else:
        span = splits.segments(spec)["research"]
        full = r.frames()
        seen = runner._cut(full, spec.data, span[1])
        times = splits.decision_times(spec, seen)
        steps = splits.temporal_steps(spec, times, "research")
        refits = [i for i, st in enumerate(steps) if st.refit]   # predictions start at a refit with labels to fit on
        start = next((i for i in refits if i >= len(steps) // 2), refits[-1] if refits else 0)
        steps = steps[start:start + 3]
        times = [t for t in times if t <= steps[-1].t]
        with r.stage(seen) as st:
            res = r.child(st, "temporal_job", must=False, times=runner._iso(times),
                          steps=[runner._step_out(s) for s in steps])
            P = pl.read_parquet(st / "predictions.parquet") if res.ok else None
    return time.monotonic() - t0, P, res


def check(study, code: str = ""):
    """Run the checks on a snapshot of the study's code (or a branch's) and record a module_check node."""
    from . import fingerprint, runner, schema
    from . import spec as _spec
    from .branches import resolve
    rec = study.record
    spec = _spec.check(study.folder, rec) if _spec.sealed(rec) is not None else _spec.load(study.folder)
    code_dir, _ = resolve(study, code) if code else (None, "")
    results = []
    with fingerprint.staged(study, code_dir) as snap:
        fp = fingerprint.fingerprint(study, snap)
        bad = lint(snap, bool(spec.sandbox.get("network", False)))
        results.append({"name": "lint", "ok": not bad, "seconds": 0.0,
                        "detail": "; ".join(f"{f}:{ln}: {why}" for f, ln, why in bad[:10]) or "nothing a run must not "
                                                                                            "touch"})
        timeout = float(spec.sandbox.get("timeout") or SMOKE_TIMEOUT)
        raw = {**spec.raw, "sandbox": {**spec.raw.get("sandbox", {}), "timeout": min(timeout, SMOKE_TIMEOUT)}}
        smoke_spec = _spec.build(raw, study.folder)
        r = runner._Run(study, smoke_spec, "research", spec.split.get("seed", 0), snap)
        t0 = time.monotonic()
        with r.stage({}) as st:
            res = r.child(st, "openlab.study.preflight:import_job", must=False)
            import json
            missing = json.loads((st / "import.json").read_text())["missing"] if res.ok else None
        limit = min(timeout, SMOKE_TIMEOUT)
        if not res.ok or missing:
            detail = _failed(res, limit, "the child failed") if not res.ok else f"missing {', '.join(missing or [])}"
            results.append({"name": "import", "ok": False, "seconds": round(time.monotonic() - t0, 2), "detail": detail})
        else:
            results.append({"name": "import", "ok": True, "seconds": round(time.monotonic() - t0, 2),
                            "detail": "model.py imports and defines Model.fit and Model.predict"})
            s1, P1, res1 = _smoke(r, smoke_spec)
            if P1 is None:
                results.append({"name": "smoke", "ok": False, "seconds": round(s1, 2),
                                "detail": _failed(res1, limit, "the child failed")})
            else:
                results.append({"name": "smoke", "ok": True, "seconds": round(s1, 2),
                                "detail": f"{P1.height} predictions from a little data"})
                s2, P2, _ = _smoke(r, smoke_spec)
                same = P2 is not None and P1.equals(P2)
                results.append({"name": "determinism", "ok": same, "seconds": round(s2, 2),
                                "detail": "the same predictions twice" if same else
                                "a second smoke run predicted differently: seed every random generator from the seed"})
        r._close_servers()
    return rec.add(schema.ModuleCheck(code=fp["code"], checks=results, ok=all(c["ok"] for c in results)),
                   actor="agent")


def refuse_red(rec, code: dict) -> None:
    """Refuse a run of code whose latest module check is red."""
    checks = [n for n in rec.find("module_check") if n.payload["code"] == code]
    if checks and not checks[-1].payload["ok"]:
        red = [c for c in checks[-1].payload["checks"] if not c["ok"]]
        raise StudyError(f"this code's module check is red ({checks[-1].id}): "
                         + "; ".join(f"{c['name']}: {c['detail']}" for c in red)[:400],
                         "a run of code that fails its checks would fail later, or mislead",
                         "fix it and run `openlab study check` again")


def unchecked(rec, code: dict) -> str | None:
    """A one-line warning when this code has no module check while an earlier version of it has one (the code was
    edited after its check), naming the files that differ from the last check; else None. Not a refusal."""
    checks = rec.find("module_check")
    if not checks or any(n.payload["code"] == code for n in checks):
        return None
    last = checks[-1].payload["code"]
    files = sorted(f for f in set(code) | set(last) if code.get(f) != last.get(f))
    named = ", ".join(files[:3]) + (f" and {len(files) - 3} more" if len(files) > 3 else "")
    return (f"{named or 'the code'} changed since its last check ({checks[-1].id}); this run's code has no check: "
            "run study_check")
