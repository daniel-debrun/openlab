"""Harness lessons: what a run showed about the harness, kept as advice a later agent is given, each resting on a
computed contrast of runs (openlab.harness.signals), after the study lessons' rule that a lesson's comparison is
computed rather than written (openlab.study.lessons).

A lesson holds:

- `cue`: the shape of the situation it applies to, not one task's facts;
- `component`: the part of the harness it is about (brief, common, role, skill, tool_guidance, workflow);
- `advice`: the sentence an agent is given (an observed pattern);
- `edit` (optional): the harness edit it proposes, {base: harness sha, files: {path: new text}}; proposing stores the
  edited harness as a version (openlab.harness.versions), which may become active only once the lesson is approved;
- `basis`: stored contrast ids, each with 2+ reference runs and a field that moved beyond the runs' noise;
- `predicted`: the signal fields ({field: "up" | "down"}) and evaluator tasks ({task: "up" | "down"}) it expects to
  move, which the evaluator checks (a prediction that fails rolls the change back);
- `provenance`: the runs it was drawn from, and `untrusted`, set when any of them read the open web or a fetched
  source. Such a lesson is quarantined: its advice must state a pattern, not give an order, and only a person's
  explicit approval (`--pick untrusted`) lets it out;
- `keys`: the roles and study templates it is for (empty: any).

Lessons are content-addressed (`hl-<sha>`; .openlab/harness/lessons/<id>.json) and their states are events on the
control store's hash chain: proposed (or quarantined) by an agent, approved by a person through the inbox (signed;
an edit that adds a script needs `--pick script`), retired by a person, confirmed by openlab when a later comparison
of spawns given the lesson against matched spawns without it moves its predicted fields its way. Entries are added
or retired, never rewritten. A lesson not confirmed for [harness] lesson_ttl_days is stale and no longer given.

Approved lessons are pushed: at spawn, the best-matching few (BM25 over cue and advice, filtered by role and template;
[harness] advice of them) are put in the brief under the always-on cap, and launch.json lists them, so their track
record is computed from the spawns that had them.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import ALWAYS_ON_MAX, SCRIPT, HarnessError, _eval_words, settings

COMPONENTS = ("brief", "common", "role", "skill", "tool_guidance", "workflow")
STATES = ("proposed", "quarantined", "approved", "retired")
VERDICTS = ("helped", "hurt", "no observed effect", "confounded")
CUE_MAX, ADVICE_MAX = 300, 600
BLOCK_MAX = 1200            # characters of advice one brief gets
#: words that open an order rather than an observation (a quarantined lesson's advice may not start a sentence so)
IMPERATIVE = {"run", "use", "always", "never", "do", "don't", "dont", "call", "execute", "delete", "remove",
              "ignore", "fetch", "download", "read", "write", "set", "install", "send", "open", "copy", "upload",
              "disable", "enable", "skip", "bypass", "override", "please", "must", "you"}
PATH = re.compile(r"(?:[\w.-]+/)+[\w.-]+|\b[\w-]+\.(?:csv|parquet|json|py|toml|txt|pdf|tex|md|jsonl|db)\b")
CONSTANT = re.compile(r"\b\d+\.\d{2,}\b|\b\d{4,}\b")


def _home(root) -> Path:
    from .versions import home
    return home(root) / "lessons"


def _store(root):
    from ..agents.store import Store
    return Store(root)


def _id(body: dict) -> str:
    keep = {k: body[k] for k in sorted(body) if k not in ("by", "created")}
    return "hl-" + hashlib.sha256(json.dumps(keep, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def get(root, lid: str) -> dict:
    try:
        return json.loads((_home(root) / f"{lid}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise HarnessError([f"no harness lesson {lid!r}: `openlab harness lessons` lists them"]) from None


def _events(root, kinds=("harness_lesson", "harness_lesson_status", "harness_lesson_confirmed")) -> list:
    from .versions import AGENT
    return _store(root).events(AGENT, n=10 ** 6, kinds=kinds)


def status(root, lid: str) -> str:
    st = ""
    for e in _events(root, ("harness_lesson", "harness_lesson_status")):
        if e["payload"].get("id") == lid:
            st = e["payload"]["status"]
    return st


def leaks(text: str) -> list:
    """What would tie a lesson to one task: a benchmark's name or grading word, a file path, a copied constant."""
    low, out = text.lower(), []
    out += [f"names {w!r}" for w in _eval_words() if re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", low)]
    out += [f"names the path {m.group(0)!r}" for m in PATH.finditer(text)]
    out += [f"copies the constant {m.group(0)}" for m in CONSTANT.finditer(text)]
    return out


def orders(text: str) -> list:
    """The sentences of `text` that give an order rather than state a pattern."""
    out = []
    for s in re.split(r"(?<=[.!?;])\s+|\n+", text):
        w = re.match(r"[\W_]*([A-Za-z']+)", s)
        if w and w.group(1).lower() in IMPERATIVE:
            out.append(s.strip()[:80])
    return out


def _supported(root, basis: list, predicted: dict) -> list:
    """Why the basis does not support the lesson ([] when it does)."""
    from .signals import load_contrast
    if not basis:
        return ["a lesson rests on at least one stored contrast (`openlab harness contrast ... --save`)"]
    out, moved_any = [], False
    want = (predicted or {}).get("signals") or {}
    for cid in basis:
        try:
            c = load_contrast(root, cid)
        except KeyError as e:
            out.append(str(e))
            continue
        if c["n"][0] < 2:
            out.append(f"{cid} has {c['n'][0]} reference run(s): a lesson needs 2+, so a one-run defect is not taken "
                       "for a pattern")
            continue
        for r in c["rows"]:
            if not r.get("moved"):
                continue
            if not want:
                moved_any = True
            elif r["field"] in want and (r["delta"] > 0) == (want[r["field"]] == "up"):
                moved_any = True
    if not out and not moved_any:
        out.append("no field of its contrasts moved beyond the runs' noise" + (" the way it predicts" if want else ""))
    return out


def propose(root, by: str, cue: str, component: str, advice: str, basis: list, predicted: dict | None = None,
            provenance: list | None = None, keys: dict | None = None, edit: dict | None = None,
            verdicts: list | None = None) -> dict:
    """Check and store a lesson, store its edited harness as a version, and file the inbox item a person decides;
    {id, status, inbox, version}."""
    from ..agents.inbox import check_request
    from . import read
    from .signals import FIELDS, touched_untrusted
    cue, advice = (cue or "").strip(), (advice or "").strip()
    problems = []
    if not cue or len(cue) > CUE_MAX:
        problems.append(f"a lesson's cue is the shape of the situation, in 1 to {CUE_MAX} characters")
    if not advice or len(advice) > ADVICE_MAX:
        problems.append(f"a lesson's advice is 1 to {ADVICE_MAX} characters")
    if component not in COMPONENTS:
        problems.append(f"component is one of {', '.join(COMPONENTS)}")
    problems += [f"the cue or advice {x}: a lesson holds the shape of a situation, not one task's facts"
                 for x in leaks(cue + "\n" + advice)]
    predicted = {"signals": dict((predicted or {}).get("signals") or {}),
                 "tasks": dict((predicted or {}).get("tasks") or {})}
    problems += [f"predicted signal {f!r} is not a probe signal" for f in predicted["signals"] if f not in FIELDS]
    problems += [f"predicted {k} {f!r}: {d!r} is not up or down" for k in ("signals", "tasks")
                 for f, d in predicted[k].items() if d not in ("up", "down")]
    problems += _supported(root, list(basis or []), predicted)
    for v in verdicts or []:
        if not isinstance(v, dict) or v.get("verdict") not in VERDICTS:
            problems.append(f"each verdict is {{edit, verdict, why}} with verdict one of {', '.join(VERDICTS)}")
            break
    provenance = [str(p) for p in provenance or []]
    untrusted = any(touched_untrusted(root, p) for p in provenance if _store(root).get(p))
    if untrusted and orders(advice):
        problems.append("a lesson drawn from a run that read the open web or a fetched source states the pattern it "
                        f"saw, never an order: {'; '.join(orders(advice))}")
    keys = {"roles": [str(r) for r in (keys or {}).get("roles") or []],
            "templates": [str(t) for t in (keys or {}).get("templates") or []]}
    version, scripts = "", False
    if edit:
        from . import versions
        import tempfile
        files = edit.get("files") or {}
        if not isinstance(files, dict) or not files:
            problems.append("an edit is {base: <harness sha>, files: {path: new text}}")
        elif not problems:
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    versions.export(root, edit.get("base") or "default", tmp)
                    base = read(tmp)
                    for rel, text in files.items():
                        target = Path(tmp) / rel
                        if text is None:
                            target.unlink(missing_ok=True)
                            continue
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(str(text) + "\n", encoding="utf-8")
                    scripts = any(SCRIPT.fullmatch(rel) and base.files.get(rel) != (None if text is None else str(text))
                                  for rel, text in files.items())
                    version = versions.install(root, tmp, parent=base.sha, note="proposed with a harness lesson",
                                               lesson="")["sha"]
                except HarnessError as e:
                    problems += [f"the edit: {p}" for p in e.problems]
    if problems:
        raise HarnessError(problems)
    body = {"cue": cue, "component": component, "advice": advice, "basis": list(basis), "predicted": predicted,
            "provenance": {"runs": provenance, "untrusted": untrusted}, "keys": keys,
            "edit": {"base": edit.get("base") or "default", "version": version, "scripts": scripts} if edit else {},
            "verdicts": list(verdicts or []), "by": by,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    lid = _id(body)
    d = _home(root)
    d.mkdir(parents=True, exist_ok=True)
    if status(root, lid):
        return {"id": lid, "status": status(root, lid), "inbox": None, "version": version, "new": False}
    (d / f"{lid}.json").write_text(json.dumps(body, indent=1, sort_keys=True), encoding="utf-8")
    st = "quarantined" if untrusted else "proposed"
    from .versions import AGENT
    _store(root).event(AGENT, "harness_lesson", summary=f"{lid} {st} by {by}: {advice}",
                       payload={"id": lid, "status": st, "by": by, "version": version})
    if version:
        _store(root).event(AGENT, "harness_version", summary=f"version {version[:12]} rests on {lid}",
                           payload={"sha": version, "parent": edit.get("base") or "", "note": f"lesson {lid}",
                                    "lesson": lid, "files": len(edit.get("files") or {})})
    args = check_request(_store(root), by, "harness", {"action": "lesson", "lesson": lid})
    item = _store(root).request(by, "harness", f"approve harness lesson {lid}: {advice[:160]}", args=args,
                                cost_note="advice in later briefs" + (f"; harness version {version[:12]}"
                                                                      if version else ""))
    return {"id": lid, "status": st, "inbox": item, "version": version, "new": True}


def decide(root, lid: str, approve: bool, by: str, reason: str, pick=None) -> dict:
    """The person's decision on a lesson (inbox perform calls it, inside the signed decision)."""
    st = status(root, lid)
    if st not in ("proposed", "quarantined"):
        raise HarnessError([f"{lid} is {st or 'unknown'}, not waiting for a decision"])
    body = get(root, lid)
    if approve and st == "quarantined" and pick != "untrusted":
        raise HarnessError([f"{lid} was drawn from runs that read untrusted text: approve it with --pick untrusted "
                            "once you have read its advice and provenance"])
    if approve and (body.get("edit") or {}).get("scripts") and pick != "script":
        raise HarnessError([f"{lid}'s edit adds or changes a skill script (code an agent can run, sandboxed): read "
                            "the diff (`openlab harness diff`) and approve it with --pick script"])
    new = "approved" if approve else "retired"
    from .versions import AGENT
    _store(root).event(AGENT, "harness_lesson_status", summary=f"{lid} {new} by {by}: {reason}",
                       payload={"id": lid, "status": new, "by": by, "reason": reason})
    return {"id": lid, "status": new}


def retire(root, lid: str, reason: str) -> dict:
    """Retire an approved lesson (a person act, signed): it is no longer given."""
    from ..study import personkey
    signer = personkey.current()
    if signer is None:
        raise PermissionError("retiring a harness lesson is a person's act: `openlab harness retire ID --reason ...`")
    if status(root, lid) not in ("approved", "proposed", "quarantined"):
        raise HarnessError([f"{lid} is {status(root, lid) or 'unknown'}"])
    msg = f"openlab-harness|retire|{lid}|{reason}".encode("utf-8")
    from .versions import AGENT
    _store(root).event(AGENT, "harness_lesson_status", summary=f"{lid} retired by {signer.label}: {reason}",
                       payload={"id": lid, "status": "retired", "by": signer.label, "reason": reason,
                                "sig": {"key": signer.fingerprint, "value": signer.sign(msg)}}, exact=True)
    return {"id": lid, "status": "retired"}


def _since(root, lid: str) -> str:
    """When the lesson was last approved or confirmed."""
    at = ""
    for e in _events(root, ("harness_lesson_status", "harness_lesson_confirmed")):
        if e["payload"].get("id") == lid and (e["kind"] == "harness_lesson_confirmed"
                                              or e["payload"].get("status") == "approved"):
            at = e["at"]
    return at


def stale(root, lid: str, ttl_days: float | None = None) -> bool:
    ttl = float(settings(root)["lesson_ttl_days"] if ttl_days is None else ttl_days)
    at = _since(root, lid)
    try:
        return not at or datetime.now(timezone.utc) - datetime.fromisoformat(at) > timedelta(days=ttl)
    except ValueError:
        return True


def listing(root) -> list:
    d = _home(root)
    if not d.is_dir():
        return []
    rows = []
    for f in sorted(d.glob("hl-*.json")):
        lid = f.stem
        try:
            body = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            continue
        st = status(root, lid)
        rows.append({"id": lid, "status": st, "stale": st == "approved" and stale(root, lid), **body})
    return rows


# --- push retrieval ----------------------------------------------------------------------------------------------
def _tokens(text: str) -> list:
    return [w for w in re.split(r"[^a-z0-9\-]+", text.lower()) if len(w) > 2]


def template(root, study: str) -> str:
    """A study's template: replicate (replication.toml), review, temporal or static (study.toml mode)."""
    if not study:
        return ""
    f = Path(root) / study
    if (f / "replication.toml").is_file():
        return "replicate"
    try:
        from . import _toml
        spec = _toml((f / "study.toml").read_text(encoding="utf-8"))
    except Exception:                                               # noqa: BLE001 -- not a study: no template
        return ""
    return "review" if spec.get("review") else str(spec.get("mode") or "")


def advice(root, role: str, study: str, prompt: str, h=None) -> tuple:
    """(the block put in a spawned agent's brief, the lesson ids in it): the approved, fresh lessons for its role
    and template that best match its task, within [harness] advice and what the always-on cap leaves."""
    d = _home(root)
    if not d.is_dir():
        return "", ()
    n = int(settings(root)["advice"])
    tpl = template(root, study)
    pool = [r for r in listing(root) if r["status"] == "approved" and not r["stale"]
            and (not r["keys"]["roles"] or role in r["keys"]["roles"])
            and (not r["keys"]["templates"] or tpl in r["keys"]["templates"])]
    if not pool or n <= 0:
        return "", ()
    docs = [(r, _tokens(r["cue"] + " " + r["advice"] + " " + r["component"])) for r in pool]
    q = _tokens(f"{prompt} {role} {tpl}")
    N, avg = len(docs), sum(len(t) for _, t in docs) / len(docs) or 1
    df = {w: sum(1 for _, t in docs if w in t) for w in set(q)}
    scored = []
    for r, t in docs:
        s = sum(math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5)) * t.count(w) * 2.2
                / (t.count(w) + 1.2 * (0.25 + 0.75 * len(t) / avg)) for w in q if t.count(w))
        keyed = role in r["keys"]["roles"] or tpl in r["keys"]["templates"]
        if s > 0 or keyed:
            scored.append((s + (1.0 if keyed else 0.0), r))
    scored.sort(key=lambda x: (-x[0], x[1]["id"]))
    room = BLOCK_MAX
    if h is not None:
        from ..agents.spawn import _role_note
        room = min(room, ALWAYS_ON_MAX - len(h.text("brief.md")) - len(_role_note(role, h)) - len(h.text("self.md")))
    head = "Harness lessons a person approved (advice drawn from earlier runs, not rules; ids {}):"
    lines, ids = [], []
    for _, r in scored[:n]:
        line = f"- {r['advice']} (when: {r['cue']})"
        if len(head) + 40 + sum(len(x) + 1 for x in lines) + len(line) > room:
            break
        lines.append(line)
        ids.append(r["id"])
    if not ids:
        return "", ()
    return head.format(", ".join(ids)) + "\n" + "\n".join(lines), tuple(ids)


# --- track record ------------------------------------------------------------------------------------------------
def track(root, lid: str) -> dict:
    """Spawns given the lesson against matched spawns without it (same role and harness version), by signals."""
    from .signals import contrast, runs
    body = get(root, lid)
    given = [r for r in runs(root) if lid in r["lessons"]]
    pairs = {(r["role"], r["harness"]) for r in given}
    matched = [r for r in runs(root) if (r["role"], r["harness"]) in pairs and lid not in r["lessons"]]
    c = contrast(matched, given)
    want = body["predicted"]["signals"]
    held = [r["field"] for r in c["rows"] if r.get("moved") and r["field"] in want
            and (r["delta"] > 0) == (want[r["field"]] == "up")]
    failed = [r["field"] for r in c["rows"] if r.get("moved") and r["field"] in want
              and (r["delta"] > 0) != (want[r["field"]] == "up")]
    return {"id": lid, "given": len(given), "matched": len(matched), "held": held, "failed": failed, "contrast": c}


def confirm(root, lid: str) -> dict:
    """Record the lesson confirmed when its track record moves a predicted field its way and none against it."""
    t = track(root, lid)
    ok = bool(t["held"]) and not t["failed"]
    if ok:
        from .signals import save
        from .versions import AGENT
        cid = save(root, t["contrast"], note=f"track record of {lid}")
        _store(root).event(AGENT, "harness_lesson_confirmed", summary=f"{lid} confirmed: {', '.join(t['held'])}",
                           payload={"id": lid, "held": t["held"], "contrast": cid})
    return {**{k: v for k, v in t.items() if k != "contrast"}, "confirmed": ok}


# --- reflections -------------------------------------------------------------------------------------------------
def reflection(root, by: str, of: str, verdicts: list, lesson: dict | None = None) -> dict:
    """A reflector's record on a run: its per-edit verdicts and at most one lesson (proposed), or NO_ENTRY."""
    bad = [v for v in verdicts or [] if not isinstance(v, dict) or v.get("verdict") not in VERDICTS]
    if bad:
        raise HarnessError([f"each verdict is {{edit, verdict, why}} with verdict one of {', '.join(VERDICTS)}"])
    out = propose(root, by, verdicts=verdicts, **lesson) if lesson else None
    body = {"of": of, "by": by, "verdicts": verdicts or [], "entry": out["id"] if out else "NO_ENTRY"}
    rid = "hr-" + hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    from .versions import AGENT, home
    d = home(root) / "reflections"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{rid}.json").write_text(json.dumps(body, indent=1, sort_keys=True), encoding="utf-8")
    _store(root).event(AGENT, "harness_reflection", summary=f"{rid} on {of} by {by}: {body['entry']}",
                       payload={"id": rid, "of": of, "entry": body["entry"], "verdicts": len(body["verdicts"])})
    return {"id": rid, "entry": body["entry"], **({"lesson": out} if out else {})}
