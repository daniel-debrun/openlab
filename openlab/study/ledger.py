"""Budgets belong to the question and the data, not to the folder.

A study's trials and holdout looks are counted in its record. Two things would let a folder escape them:

- **A copied record.** `cp -r demo demo2` forks demo's record: the copy would start counting from demo's trials
  with its own holdout. `.openlab/origin` names the folder a record was made in; a record whose origin is another
  folder that still holds the same record (the same first node) is a copy, and runs in it are refused, naming the
  original. In a workspace, two studies holding the same record are refused the same way when neither is the
  origin.
- **A second study of the same question.** A new study whose datasets (by sha256) and target equal a sealed
  sibling's in the workspace would get a fresh trial budget and a fresh holdout. Its runs are refused until it
  either declares `[study] continues = "../demo"` (its trials and looks then count against one shared budget with
  demo's, and demo's against it), or a person declares a fresh question: `[study] fresh = "<why>"`, recorded by a
  person (in the plan a person sealed, or a person's amendment).

`shared` gives the shared counts (what `status` prints as `trials 4 of 20 (3 in demo)`) and `trials` the total a
budget is checked against.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

from . import workspace
from .errors import StudyError
from .record import canonical

if TYPE_CHECKING:
    from . import Study

ORIGIN = "origin"


def _genesis(folder: Path) -> str:
    """The hash of a record's first node ("" when it has none), read without creating a record."""
    db = Path(folder) / ".openlab" / "record.db"
    if not db.is_file():
        return ""
    try:
        with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30)) as con:
            row = con.execute("SELECT hash FROM nodes ORDER BY seq LIMIT 1").fetchone()
    except sqlite3.Error:
        return ""
    return row[0] if row else ""


def claim_origin(folder: Path) -> None:
    """Write .openlab/origin for a record made here (once)."""
    f = Path(folder) / ".openlab" / ORIGIN
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(str(Path(folder).resolve()), encoding="utf-8")


def not_a_copy(study: "Study") -> None:
    """Refuse a run in a copied record (see the module's docstring); a moved study takes its new folder as origin."""
    here, self_genesis = study.folder.resolve(), _genesis(study.folder)
    f = study.record.dir / ORIGIN
    origin = Path(f.read_text(encoding="utf-8").strip()) if f.is_file() else None
    if origin is not None and origin != here and self_genesis and _genesis(origin) == self_genesis:
        raise StudyError(f"{here} holds a copy of the record of {origin}",
                         "a copied record would start again from the original's trials with a holdout of its own, so "
                         "the trial and holdout budgets would no longer count every try",
                         f"run in {origin}; for a new question start a study with `openlab study new` (a study on "
                         "the same data declares [study] continues = \"<path to the original>\")")
    if origin is None:
        twins = [s for s in workspace.studies(workspace.root(here)) if s.resolve() != here and _genesis(s) == self_genesis and self_genesis]
        claimed = [s for s in twins if (s / ".openlab" / ORIGIN).is_file()
                   and Path((s / ".openlab" / ORIGIN).read_text(encoding="utf-8").strip()) == s.resolve()]
        if claimed or twins:
            raise StudyError(f"{here} holds the same record as {(claimed or twins)[0]}",
                             "one record in two folders would count one budget twice, and neither names the folder "
                             "the record was made in", "keep the study in one folder: remove the copy, then run "
                             "again (a new question is a new study: openlab study new)")
    if origin != here:
        f.write_text(str(here), encoding="utf-8")


def _plan(folder: Path):
    """(the sealed spec's raw dict, dataset shas, record genesis) of a sibling study, read-only; None if unsealed."""
    db = Path(folder) / ".openlab" / "record.db"
    if not db.is_file():
        return None
    try:
        with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30)) as con:
            plan = con.execute("SELECT payload FROM nodes WHERE kind='plan' ORDER BY seq LIMIT 1").fetchone()
            if plan is None:
                return None
            amends = [json.loads(r[0]) for r in con.execute(
                "SELECT payload FROM nodes WHERE kind='amendment' ORDER BY seq")]
            data = {}
            for (p,) in con.execute("SELECT payload FROM nodes WHERE kind='dataset' ORDER BY seq"):
                d = json.loads(p)
                data[d["name"]] = d["sha256"]
            trials = {json.loads(p)["key"] for (p,) in con.execute(
                "SELECT payload FROM nodes WHERE kind='run' AND json_extract(payload, '$.segment')='research'")}
            declared = sum(json.loads(p)["n"] for (p,) in con.execute("SELECT payload FROM nodes WHERE kind='trial'"))
            looks = con.execute("SELECT COUNT(*) FROM nodes WHERE kind='look'").fetchone()[0]
    except sqlite3.Error:
        return None
    from .spec import _apply
    raw = json.loads(plan[0])["spec"]
    for a in amends:
        raw = _apply(raw, a["changes"])
    return {"raw": raw, "data": data, "trials": len(trials) + declared, "looks": looks}


def _continues(folder: Path, raw: dict) -> Path | None:
    c = raw.get("study", {}).get("continues")
    return (Path(folder) / c).resolve() if c else None


def _copy(folder: Path) -> bool:
    """`folder` holds a copy of another folder's record: its origin names that folder, which holds the same record."""
    f = Path(folder) / ".openlab" / ORIGIN
    if not f.is_file():
        return False
    origin, self_genesis = Path(f.read_text(encoding="utf-8").strip()), _genesis(folder)
    return origin != Path(folder).resolve() and bool(self_genesis) and _genesis(origin) == self_genesis


def siblings(folder: Path) -> list:
    """The workspace's other studies with a record of their own: a copy of a record (refused at run by
    `not_a_copy`) is not a study, and a record held in two folders counts once."""
    here = Path(folder).resolve()
    out, seen = [], {_genesis(here)} - {""}
    for s in workspace.studies(workspace.root(here)):
        s = s.resolve()
        if s == here or _copy(s):
            continue
        g, f = _genesis(s), s / ".openlab" / ORIGIN
        own = f.is_file() and Path(f.read_text(encoding="utf-8").strip()) == s
        if g and g in seen and not own:     # an unclaimed second folder of one record (an older record made in the
            continue                        # same second as another can share a first node: its origin tells)
        seen.add(g)
        out.append(s)
    return out


def members(folder: Path, rec, raw: dict | None = None) -> dict:
    """{folder: plan info} of the other studies sharing this study's ledger: those it continues (transitively), and
    those in the workspace that continue it or one of them. `raw`: this study's spec as it is to be sealed (default:
    the sealed one)."""
    from .spec import sealed
    here = Path(folder).resolve()
    if raw is None:
        spec = sealed(rec)
        if spec is None:
            return {}
        raw = spec.raw
    plans = {s: _plan(s) for s in siblings(here)}
    plans = {k: v for k, v in plans.items() if v}
    group, todo = {here}, [here]
    edges = {k: _continues(k, v["raw"]) for k, v in plans.items()}
    edges[here] = _continues(here, raw)
    while todo:
        cur = todo.pop()
        linked = {edges.get(cur)} | {k for k, c in edges.items() if c == cur}
        for k in linked - {None} - group:
            if k not in plans:
                p = None if _copy(k) else _plan(k)
                if p is None:
                    continue
                plans[k] = p
                edges[k] = _continues(k, p["raw"])
            group.add(k)
            todo.append(k)
    return {k: plans[k] for k in sorted(group - {here})}


def shared(folder: Path, rec) -> dict:
    """{"trials", "looks"} counted against this study's budgets (its own and its ledger's), and "by": {study name:
    {trials, looks}} of the others."""
    from .schema import trials_used
    others = members(folder, rec)
    by = {v["raw"]["study"]["name"]: {"trials": v["trials"], "looks": v["looks"], "folder": str(k)}
          for k, v in others.items()}
    return {"trials": trials_used(rec) + sum(v["trials"] for v in by.values()),
            "looks": len(rec.find("look")) + sum(v["looks"] for v in by.values()), "by": by}


def _fresh(rec) -> str:
    """The fresh question a person declared ([study] fresh in a plan a person sealed or a person's amendment)."""
    from .personkey import node_signed            # a person's declaration is one signed by the person key
    plan = rec.find("plan")
    if plan and node_signed(rec, plan[0]) and plan[0].payload["spec"].get("study", {}).get("fresh"):
        return plan[0].payload["spec"]["study"]["fresh"]
    for a in reversed(rec.find("amendment")):
        if "study.fresh" in a.payload["changes"]:
            return a.payload["changes"]["study.fresh"]["new"] if node_signed(rec, a) else ""
    return ""


def same_target(folder: Path, rec, raw: dict) -> list:
    """[(folder, plan info)] of the sealed siblings outside this study's ledger (spec `raw`) asking about the same
    target: the ones whose datasets must be compared."""
    here = Path(folder).resolve()
    target = canonical(raw.get("target", {}))
    shared_with = set(members(here, rec, raw))
    out = []
    for s in siblings(here):
        if s in shared_with:
            continue
        p = _plan(s)
        if p and canonical(p["raw"].get("target", {})) == target:
            out.append((s, p))
    return out


def conflicts(folder: Path, rec, raw: dict, owned: list) -> list:
    """The siblings whose datasets (by sha256; `owned` is this study's, sorted) and target equal this study's (spec
    `raw`) and that do not share its ledger, each {name, folder, rel, trials, looks}."""
    here = Path(folder).resolve()
    if not owned:
        return []
    out = []
    for s, p in same_target(here, rec, raw):
        if sorted(p["data"][n] for n in p["raw"].get("data", {}) if n in p["data"]) == owned:
            rel = Path("..") / s.name if s.parent == here.parent else s
            out.append({"name": p["raw"]["study"]["name"], "folder": str(s), "rel": str(rel), "trials": p["trials"],
                        "looks": p["looks"]})
    return out


def fixes(c: dict) -> dict:
    """The two ways out of a same-question conflict `c`: continue its ledger, or a person's fresh question."""
    return {"continues": f'[study] continues = "{c["rel"]}": its {c["trials"]} trials and {c["looks"]} holdout '
                         "looks count against this study's budget too, and this study's against it",
            "fresh": '[study] fresh = "<why this is a new question>", declared by a person: a budget and a holdout '
                     "of its own"}


def conflict_error(name: str, c: dict, owned: list, sealed: bool) -> StudyError:
    how = ("record it with `openlab study amend`" if sealed else
           "edit study.toml (before the first run nothing needs recording)")
    return StudyError(f"{name} asks the question of {c['name']} ({c['folder']}) of the same data",
                      f"the same datasets (sha {', '.join(x[:12] for x in owned)}) and target; a second study "
                      f"would start a fresh trial budget and a fresh holdout ({c['trials']} trials and "
                      f"{c['looks']} holdout looks are spent there)",
                      f'declare [study] continues = "{c["rel"]}" (its trials and looks then count here too): {how}; '
                      'or a person declares a new question: [study] fresh = "<why>", recorded with '
                      "`openlab study amend --person` (before the first run that seals study.toml as the person's); "
                      f'an agent asks a person: openlab_agent_request(kind="continues", args={{"continues": '
                      f'"{c["rel"]}"}}) or kind="fresh", args={{"why": "..."}}')


def same_question(study: "Study", spec) -> None:
    """Refuse a study whose datasets and target equal a sealed sibling's, unless it shares that study's ledger or a
    person declared a fresh question (see the module's docstring). `spec` may be study.toml before the seal."""
    from .fingerprint import data
    from .spec import sealed
    owned = sorted(data(study.record, set(spec.data)).values())
    if not owned or _fresh(study.record):
        return
    found = conflicts(study.folder, study.record, spec.raw, owned)
    if found:
        raise conflict_error(spec.name, found[0], owned, sealed(study.record) is not None)


def declare(folder: Path, key: str, value: str) -> None:
    """Write `[study] continues` or `fresh` into study.toml (replacing the line when it is there); the caller
    records the change (an amendment once sealed: a person's for fresh)."""
    import re
    if key not in ("continues", "fresh") or not str(value).strip():
        raise StudyError(f"cannot declare study.{key}", "continues (a study folder) or fresh (why), with a value")
    p = Path(folder) / "study.toml"
    lines = p.read_text(encoding="utf-8").splitlines()
    line = f"{key} = {json.dumps(str(value).strip(), ensure_ascii=False)}"
    head = next((i for i, x in enumerate(lines) if re.match(r"\s*\[study\]\s*(#.*)?$", x)), None)
    if head is None:
        raise StudyError("study.toml has no [study] section")
    end = next((i for i in range(head + 1, len(lines)) if re.match(r"\s*\[", lines[i])), len(lines))
    at = next((i for i in range(head + 1, end) if re.match(rf"\s*{key}\s*=", lines[i])), None)
    if at is not None:
        lines[at] = line
    else:
        last = max([i for i in range(head + 1, end) if lines[i].strip()] or [head])
        lines.insert(last + 1, line)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise StudyError(f"cannot write {v!r} into study.toml", "a string, number, boolean or a list of them")


def write_fields(folder: Path, fields: dict) -> list:
    """Set {section: {key: value}} in study.toml ("data.main" for [data.main]), line by line so the comments stay:
    a key's line is replaced, a missing key is added at the end of its section, a missing section at the end of the
    file. The caller records the change (an amendment). Returns the "section.key" names written."""
    import re
    p = Path(folder) / "study.toml"
    lines = p.read_text(encoding="utf-8").splitlines()
    done = []
    for sec, kv in fields.items():
        if not isinstance(kv, dict) or not kv:
            raise StudyError(f"study.toml field {sec!r} is not a table of keys",
                             '{"split": {"scheme": "kfold", "k": 5}}: a section, then its keys and values')
        hdr = re.compile(rf"\s*\[{re.escape(sec)}\]\s*(#.*)?$")
        head = next((i for i, x in enumerate(lines) if hdr.match(x)), None)
        if head is None:
            lines += ["", f"[{sec}]"]
            head = len(lines) - 1
        for k, v in kv.items():
            end = next((i for i in range(head + 1, len(lines)) if re.match(r"\s*\[", lines[i])), len(lines))
            line = f"{k} = {_toml_value(v)}"
            at = next((i for i in range(head + 1, end) if re.match(rf"\s*{re.escape(k)}\s*=", lines[i])), None)
            if at is not None:
                lines[at] = line
            else:
                last = max([i for i in range(head + 1, end) if lines[i].strip()] or [head])
                lines.insert(last + 1, line)
            done.append(f"{sec}.{k}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return done
