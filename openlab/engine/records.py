"""The agent surface: every openlab result as one machine-readable record.

An agent (or a script) does not read rendered HTML. It reads a list of records, one per statement,
each with a ``verdict`` it cannot forge and an ``interpretation`` it can check against the words::

    >>> from openlab.engine.records import run_records
    >>> run_records(r"\\int_0^1 x^2 \\, dx")[0]["value"]
    '1/3'

Used by the modules and the paper's equation gate (:mod:`openlab.engine.modules`). Strict mode is the
default here: a statement that was only represented, or a check left undecided, is an error record,
because an agent should never mistake "shown" for "checked".
"""

from __future__ import annotations

import re
import time
from typing import Any, Optional

import sympy

from .errors import OpenlabError
from .semantics import Kind, Result, iter_run

#: Every verdict a record can carry, and what it means. The proof and claim words come from openlab's
#: checker; the rest describe what a statement did. See docs/AGENTS.md.
VERDICTS = {
    "proved": "the claim holds for every element of its set, by the named technique",
    "disproved": "a counterexample was found; it is in `notes`",
    "not proved": "the technique could not decide the claim; the reason is in `notes` (this is not a disproof)",
    "supported": "an empirical claim passed its test at the record's bar (never 'proved')",
    "refuted": "an empirical claim failed its test",
    "inconclusive": "the data do not decide the claim (too few rows, or between the bars)",
    "true": "a relation or identity checked True",
    "false": "a relation or identity checked False",
    "value": "an expression evaluated; see `value` and `numeric`",
    "solved": "an equation was solved; `value` lists the solutions",
    "bound": "a name was bound (assignment, definition, sequence, random variable, assumption)",
    "note": "prose or a claim with no technique: shown, never run (counts for nothing)",
    "represented": "valid math openlab does not execute; only an error under strict mode",
    "undecided": "a check openlab could neither confirm nor refute; only an error under strict mode",
    "error": "the statement did not run; `message` and `hint` say why and what to write instead",
}

_PROOF_WORDS = ("not proved", "disproved", "proved", "refuted", "supported", "inconclusive")


def _text(v: Any) -> Optional[str]:
    if v is None:
        return None
    try:
        return sympy.sstr(v) if isinstance(v, sympy.Basic) else str(v)
    except Exception:       # a value that cannot print is still a value
        return repr(v)


def _written(src: str) -> str:
    """The statement as written: openlab's internal ``\\openlabnote{…}`` is what ``\\text{…}`` becomes."""
    return re.sub(r"\\openlabnote\{([^{}]*)\}", r"\\text{\1}", src or "").strip()


def verdict_of(r: Result) -> str:
    """The verdict word of one result."""
    k = r.kind
    if k is Kind.NOTE:
        return "note"
    if k is Kind.PROOF:
        head = (r.interpretation or "").split(":", 1)[0].strip().lower()
        for w in _PROOF_WORDS:
            if head.startswith(w):
                return w
        if r.value is None:
            return "not proved"
        return {True: "proved", False: "disproved"}.get(bool(r.value), "not proved")
    if k is Kind.REPR:
        return "represented"
    if k is Kind.RELATION:
        if isinstance(r.value, sympy.logic.boolalg.BooleanAtom):
            return "true" if bool(r.value) else "false"
        if isinstance(r.value, bool):
            return "true" if r.value else "false"
        if r.value is None or isinstance(r.value, (list, sympy.Set)):
            return "value"
        return "undecided"
    if k is Kind.SOLVE:
        return "solved"
    if k in (Kind.ASSIGN, Kind.DEFINE, Kind.ASSUME):
        return "bound"
    return "value"


def _method_of(interpretation: str) -> Optional[str]:
    """``proved by induction on n: claim | …`` (core) or ``proved: claim by causality`` (panel)."""
    head = interpretation.split(" | ", 1)[0]
    m = re.match(r"^(?:not proved|disproved|proved|refuted|supported|inconclusive)(?: by (.+?))?:\s*(.*)$", head)
    if not m:
        return None
    if m.group(1):
        return m.group(1).strip()
    tail = re.search(r"\bby ([a-z][a-z ]*?)\s*$", m.group(2))
    return tail.group(1).strip() if tail else None


def to_record(r: Result) -> dict:
    """One statement's result as a plain dict (JSON-serialisable)."""
    rec = {
        "line": r.lineno,
        "src": _written(r.src),
        "kind": r.kind.name.lower(),
        "verdict": verdict_of(r),
        "interpretation": r.interpretation or "",
        "value": _text(r.value),
        "numeric": _text(r.numeric),
        "notes": [str(n) for n in (r.notes or [])],
        "bound": sorted(str(k) for k in (r.bound or {}) if isinstance(k, str)),
    }
    if rec["verdict"] in _PROOF_WORDS:
        method = _method_of(r.interpretation or "")
        if method:
            rec["method"] = method
    table = getattr(r, "table", None)
    if table:
        rec["table"] = [[str(a), _text(b)] for a, b in table]
    caption = getattr(r, "caption", "")
    if caption:
        rec["caption"] = caption
    return rec


def error_record(err: BaseException, lineno: Optional[int] = None, lines: Optional[list] = None) -> dict:
    """``lines``: the cell's source lines, so ``src`` is what was written, not openlab's internal form."""
    ln = getattr(err, "lineno", None) or lineno
    src = getattr(err, "src", "") or ""
    if lines and ln and 0 < ln <= len(lines) and lines[ln - 1].strip():
        src = lines[ln - 1].strip()
    rec = {"line": ln, "src": src,
           "kind": "error", "verdict": "error", "error": type(err).__name__,
           "message": getattr(err, "message", None) or str(err), "hint": getattr(err, "hint", "") or ""}
    return rec


def run_records(src: str, ns: Optional[dict] = None, *, strict: bool = True, panel: Optional[str] = None,
                stop_on_error: bool = True, **opts) -> list[dict]:
    """Run a cell and return its records. The run stops at the first error (the statements after it
    depend on the ones before), unless ``stop_on_error`` is False, in which case the failing line is
    recorded and the rest of the cell is run line by line. ``panel`` is refused: the panel dialect is gone."""
    ns = {} if ns is None else ns
    if panel:
        raise OpenlabError("the panel dialect is not part of openlab")
    run_cell = iter_run
    opts = dict(opts, strict=strict)
    out: list[dict] = []
    t0 = time.perf_counter()
    try:
        for r in run_cell(src, ns, **opts):
            rec = to_record(r)
            rec["seconds"] = round(time.perf_counter() - t0, 3)
            t0 = time.perf_counter()
            out.append(rec)
    except OpenlabError as err:
        lines = src.split("\n")
        out.append(error_record(err, lines=lines))
        if not stop_on_error:
            done = {r["line"] for r in out if r.get("line")}
            failed = err.lineno or 0
            for i in range(failed + 1, len(lines) + 1):
                if i in done or not lines[i - 1].strip():
                    continue
                for rec in run_records(lines[i - 1], ns, strict=strict, panel=panel, stop_on_error=True):
                    rec["line"] = i
                    out.append(rec)
    except Exception as exc:                # noqa: BLE001 -- an agent needs a record, not a traceback
        out.append(error_record(exc, lines=src.split("\n")))
    return out


def summary(records: list[dict]) -> dict:
    """Counts by verdict, and whether the run is clean (no error, nothing undecided or represented)."""
    counts: dict[str, int] = {}
    for r in records:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    bad = {"error", "undecided", "represented", "not proved", "disproved", "refuted", "false"}
    return {"statements": len(records), "verdicts": counts,
            "clean": not any(v in bad for v in counts)}
