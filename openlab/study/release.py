"""A study's release: the white paper and the code folder, compiled from the record when the research is done.

`release/` beside the study holds what a reader takes away, and nothing openlab does not stand behind:

    release/
      README.md            the study's own README.md when it has one (written by the agents or the person), else its
                           name and question; and, between RECORD_MARK and RECORD_END (where README.md puts the two
                           lines, else at its end), the record's part, rewritten at each compile: the data, the
                           result as the record holds it (beside its baselines), the dead ends and claims, and how
                           to reproduce it
      paper.pdf            the paper (openlab.study.paper), built only when every binding resolves and every finding
                           of its lints is fixed or waived by a person; else the README says why there is none
      code/                model.py, metrics.py, recipes/, study.toml: the code the last result ran
      <name>.bundle.tar    the signed bundle a third party reproduces from (openlab study reproduce --bundle)
      MANIFEST.json        every other file's sha256, the record head, and whether the paper was built

It is rebuilt whole each time, so it never mixes two compiles: the new folder is written under `.release.tmp/` (a
name of its own per compile, beside the per-study lock that lets one compile run at a time) and swapped in only when
it is complete, so a compile that fails leaves the previous release as it was. `.release.tmp/` and `release/` are in
the SKIP sets of openlab.study.fingerprint and openlab.study.reproduce: the release is never the study's code.

code/ holds regular files only: a symlink (model.py pointing at /etc/passwd, or out of the study), a special file, a
file over MAX_BYTES or a binary file is not copied, and the README names each one; the bundle holds the study.
`verify(folder)` rechecks a release against its MANIFEST.json.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

from .errors import StudyError

CODE = ("model.py", "metrics.py", "study.toml", "recipes")         # README.md: the release README is made from it
MAX_BYTES = 5 * 1024 * 1024                     # a code/ file above this is left to the bundle
PROSE_MAX = 1 << 20                             # bytes of the study's README.md the release README is made from
RECORD_MARK = "<!-- openlab: the record's part, rewritten at each compile: edits between these marks are replaced -->"
RECORD_END = "<!-- openlab: end of the record's part -->"
WORK = ".release.tmp"                           # each compile's folder, the lock, and the release being swapped out


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _num(v) -> str:
    return f"{v:.4g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def _iv(m: dict) -> str:
    return f" [{_num(m['lo'])}, {_num(m['hi'])}]" if m.get("lo") is not None and m.get("hi") is not None else ""


def _scores(metrics: dict) -> str:
    """`brier 0.0856 [0.08, 0.09], log_loss 0.31 [...]`: each score with its interval."""
    return ", ".join(f"{k} {_num(m['value'])}{_iv(m)}" for k, m in (metrics or {}).items()
                     if isinstance(m, dict) and m.get("value") is not None) or "no score"


def _metric_lines(metrics: dict) -> list:
    return [f"  - {k}: {_num(m['value'])}{_iv(m)}" for k, m in (metrics or {}).items()
            if isinstance(m, dict) and "value" in m]


def _diff(who: str, b: str, d: dict) -> str:
    """One paired difference: the model (or its policy) minus baseline `b`, its interval, p and which is better."""
    if d.get("diff") is None:
        return f"  - {who} minus {b}: not computed ({d.get('reason') or 'undefined on these rows'})"
    better = {"model": "the model", "baseline": f"the baseline {b}", None: "neither (the difference is not clear)"}
    p = f", p = {_num(d['p'])}" if isinstance(d.get("p"), (int, float)) else ""
    return (f"  - {who} minus {b}{' on ' + d['metric'] if d.get('metric') else ''}: {_num(d['diff'])}{_iv(d)}{p}; "
            f"better: {better.get(d.get('better'), d.get('better'))}")


def _comparison(payload: dict) -> list:
    """A result's metrics, each baseline's, the paired differences, and its policy's, if it acts."""
    out = _metric_lines(payload.get("metrics"))
    out += [f"  - baseline {b}: {_scores(ms)}" for b, ms in (payload.get("baselines") or {}).items()]
    out += [_diff("model", b, d) for b, d in (payload.get("paired") or {}).items()]
    pol = payload.get("policy") or {}
    if pol.get("metrics"):
        out.append(f"  - policy over {pol.get('decisions', '?')} decisions: {_scores(pol['metrics'])}")
        out += [f"  - policy baseline {b}: {_scores(ms)}" for b, ms in (pol.get("baselines") or {}).items()]
        out += [_diff("policy", b, d) for b, d in (pol.get("paired") or {}).items()]
    return out


def _result_lines(study, status: dict) -> list:
    """The results the study's tests rest on, each with its tests and its comparison; then the latest result when it
    is another one (a check run after the test, say), named as such, its numbers left to the record. With no test,
    the latest result and its comparison. A comparison is the model's metrics, each baseline's, the paired
    differences and which is better, and the policy's, if it acts: shown whether or not the model won."""
    rec = study.record
    last = status.get("last_result") or {}
    if not last:
        return ["No result is on the record yet: run the study (openlab study run) and compile again."]
    tests = {}
    for t in rec.find("test"):
        src = [d for _, _, d in rec.links(t.id, rel="from") if d.startswith("result:")]
        if src:
            tests.setdefault(src[0], []).append(t)
    out = []
    for rid, ts in tests.items():
        said = [f"tested by `{t.id}` against {t.payload.get('baseline') or 'its baseline'}: "
                + ("rejected the null" if t.payload.get("rejected") else "did not reject the null")
                + (f" (p = {_num(t.payload['p'])})" if isinstance(t.payload.get("p"), (int, float)) else "")
                for t in ts]
        out += [f"- result `{rid}`, {said[0]}", *[f"  - also {x}" for x in said[1:]],
                *_comparison(rec.get(rid).payload)]
    if not tests:
        try:
            payload = rec.get(last["id"]).payload
        except Exception:                                               # noqa: BLE001 -- the status's summary
            payload = last
        return [f"- result `{last.get('id', '')}` (untested)", *_comparison(payload)]
    if last.get("id") not in tests:
        out += ["", f"The latest result, `{last.get('id', '')}`, is a later run that no test names "
                f"(its run: `{last.get('run', '')}`); see the record for what it checked."]
    return out


LISTED = 10                                     # dead ends and claims the README lists, each


def _record_lines(study) -> list:
    """The study's dead ends and claims, one line each (at most LISTED of each, newest last), if it has any."""
    rec, out = study.record, []
    ends = rec.find("dead_end")
    if ends:
        out += ["## Dead ends", *[f"- {d.payload['statement']}: {d.payload['reason']} (`{d.id}`)"
                                  for d in ends[-LISTED:]]]
        out += [f"- and {len(ends) - LISTED} earlier"] if len(ends) > LISTED else []
        out.append("")
    claims = rec.find("claim")
    if claims:
        try:
            rows = {r["id"]: r for r in study.claims()}
        except Exception:                                               # noqa: BLE001 -- the written state stands
            rows = {}
        lines = []
        for c in claims[-LISTED:]:
            r = rows.get(c.id, {})
            how = ", ".join(x for x in (c.payload.get("state"), r.get("verdict") or c.payload.get("verdict"),
                                        r.get("level") or c.payload.get("level")) if x)
            lines.append(f"- {c.payload['statement']} ({how}; `{c.id}`)")
        out += ["## Claims", *lines, *([f"- and {len(claims) - LISTED} earlier"] if len(claims) > LISTED else []), ""]
    return out


def _skip_why(p: Path) -> str:
    """Why `p` is not copied into code/, or '' to copy it."""
    if p.is_symlink():
        return f"a symlink (to {os.readlink(p)}), not followed"
    if not p.is_file():
        return "not a regular file"
    size = p.stat().st_size
    if size > MAX_BYTES:
        return f"{size} bytes, over {MAX_BYTES}"
    with open(p, "rb") as f:
        if b"\0" in f.read(8192):
            return "a binary file"
    return ""


def _copy_code(folder: Path, code: Path) -> list:
    """Copy the CODE entries of `folder` into `code`, regular text files only; returns [(path, why)] not copied."""
    skipped = []

    def one(src: Path, dst: Path, rel: str):
        if src.is_dir() and not src.is_symlink():
            for k in sorted(src.iterdir()):
                if k.name != "__pycache__" and not k.name.endswith(".pyc"):
                    one(k, dst / k.name, f"{rel}/{k.name}")
            return
        why = _skip_why(src)
        if why:
            skipped.append((rel, why))
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst, follow_symlinks=False)

    code.mkdir()
    for name in CODE:
        src = folder / name
        if src.is_symlink() or src.exists():
            one(src, code / name, name)
    return skipped


def _files(root: Path) -> list:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink())


def verify(folder) -> dict:
    """Recheck a release folder against its MANIFEST.json: {ok, missing, changed, extra}. Every file but the manifest
    must be listed, be a regular file, and have the listed sha256."""
    folder = Path(folder)
    try:
        listed = json.loads((folder / "MANIFEST.json").read_text(encoding="utf-8"))["files"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        return {"ok": False, "error": f"no readable MANIFEST.json: {e}", "missing": [], "changed": [], "extra": []}
    there = set(_files(folder)) - {"MANIFEST.json"}
    missing, extra = sorted(set(listed) - there), sorted(there - set(listed))
    changed = sorted(f for f in set(listed) & there
                     if (folder / f).is_symlink() or not (folder / f).is_file() or _sha(folder / f) != listed[f])
    return {"ok": not (missing or changed or extra), "missing": missing, "changed": changed, "extra": extra}


def build(study) -> dict:
    """Compile the release of `study` (an openlab.study.Study). Returns {folder, paper: built | refused | error | none,
    why, findings, files, skipped}; never raises for a paper that is refused or fails, since the code folder stands
    without it. One compile per study runs at a time (the others wait); one that raises leaves release/ as it was."""
    folder = Path(study.folder)
    work = folder / WORK
    work.mkdir(exist_ok=True)
    with open(work / "lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            tmp = work / f"new-{uuid.uuid4().hex}"
            tmp.mkdir()
            try:
                r = _compile(study, folder, tmp)
                _swap(tmp, folder / "release", work)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            return r
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _swap(tmp: Path, out: Path, work: Path):
    """Put `tmp` in the place of `out`; if that fails, `out` is put back as it was."""
    old = None
    if out.exists() or out.is_symlink():
        old = work / f"old-{uuid.uuid4().hex}"
        out.rename(old)
    try:
        tmp.rename(out)
    except BaseException:
        if old is not None:
            old.rename(out)
        raise
    if old is not None:
        if old.is_symlink():
            old.unlink()
        else:
            shutil.rmtree(old, ignore_errors=True)


def _prose(folder: Path) -> list:
    """The lines of the study's README.md ([] when it has none, or it is a link, a special file or over
    PROSE_MAX)."""
    p = folder / "README.md"
    if p.is_symlink() or not p.is_file() or p.stat().st_size > PROSE_MAX:
        return []
    lines = p.read_text(encoding="utf-8", errors="replace").rstrip().splitlines()
    return lines if any(x.strip() for x in lines) else []


def place(lines: list, block: list) -> list:
    """`lines` with the record's part `block` (its marks included) in place of the one between their marks, or
    after them when they have none (a start mark with no end: to the end)."""
    marks = [x.strip() for x in lines]
    if RECORD_MARK not in marks:
        return [*lines, "", *block]
    i = marks.index(RECORD_MARK)
    j = marks.index(RECORD_END, i) if RECORD_END in marks[i:] else len(lines) - 1
    return [*lines[:i], *block, *lines[j + 1:]]


def _compile(study, folder: Path, tmp: Path) -> dict:
    status = study.status()

    paper, findings, why = "none", [], "the study has no paper (paper/paper.tex): the agents wrote no paper"
    if (folder / "paper" / "paper.tex").is_file() or (folder / "paper" / "paper.toml").is_file():
        try:
            study.paper()
            pdf = folder / "paper" / "build" / "main.pdf"
            paper, why = "error", "the paper build wrote no paper/build/main.pdf"
            if pdf.is_file() and not pdf.is_symlink():
                shutil.copyfile(pdf, tmp / "paper.pdf")
                paper, why = "built", ""
        except StudyError as e:                                         # refused: findings listed
            paper, why = "refused", str(e.args[0] if e.args else e)
            try:
                lint = study.paper_lint()
                findings = [f"{f.get('rule', '')}: {f.get('message', '')}" for f in lint.get("findings", [])
                            if not f.get("waived")][:30]
            except Exception:                                           # noqa: BLE001
                pass
        except Exception as e:                                          # noqa: BLE001 -- a fault, not a refusal
            paper, why = "error", f"the paper build failed: {type(e).__name__}: {e}"

    skipped = _copy_code(folder, tmp / "code")

    bundle = tmp / f"{folder.name}.bundle.tar"
    try:
        study.bundle(bundle)
    except Exception as e:                                              # noqa: BLE001
        bundle.unlink(missing_ok=True)
        bundle = None
        why = (why + "; " if why else "") + f"no bundle: {e}"

    head = study.record.head()
    trials, looks = status.get("trials") or {}, status.get("looks") or {}
    data = [f"- {d['name']}: {d.get('rows')} rows, sha256 `{(d.get('sha') or '')[:16]}`" for d in status.get("datasets", [])]
    shipped = _files(tmp / "code")
    untried = not study.record.find("run")
    nothing = ["**No runs yet: nothing has been tested.** This release holds the plan and the code, no result.", ""]
    prose = _prose(folder)
    record = [*(nothing if untried and prose else []),
              "## Data", *(data or ["- none recorded"]), "",
              "## Result", *_result_lines(study, status), "",
              f"Trials used: {trials.get('used', 0)} of {trials.get('budget') or 'no budget'}. "
              f"Holdout looks: {looks.get('used', 0)} of {looks.get('budget', 0)}.", "",
              *_record_lines(study),
              "## The paper", "paper.pdf: every number in it is bound to the record." if paper == "built"
              else f"There is no paper.pdf: {why}.", *[f"- {f}" for f in findings], "",
              "## Code", "code/ holds " + (", ".join(f"`{f}`" for f in shipped) or "nothing") + ".",
              *(["", "Not copied into code/ (the bundle holds the study as it is):"]
                + [f"- `{p}`: {w}" for p, w in skipped] if skipped else []), "",
              "## Reproduce", f"`openlab study reproduce --bundle {folder.name}.bundle.tar`" if bundle else
              "No bundle was written.", "", f"Record head: `{head}`"]
    block = [RECORD_MARK, "", *record, "", RECORD_END]
    readme = place(prose, block) if prose else \
        [f"# {status.get('name') or folder.name}", "", *(nothing if untried else []), status.get("question") or "", "",
         *block]
    (tmp / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    files = _files(tmp)
    manifest = {"study": folder.name, "record_head": head, "paper": paper, "skipped": dict(skipped),
                "files": {f: _sha(tmp / f) for f in files}}
    (tmp / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return {"folder": str(folder / "release"), "paper": paper, "why": why, "findings": findings,
            "files": files + ["MANIFEST.json"], "skipped": [p for p, _ in skipped]}
