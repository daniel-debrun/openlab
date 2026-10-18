"""`openlab study new|materialise|run|runs|status|why|amend|holdout|verify|forecast|fill|schedule|discover|fetch|screen|prisma|
extract|evidence|compare|lattice|recall|claim|question|hypothesis|test|assume|break|dead-end|contra|tree|claims|approve|
paper|paper-lint|verify-paper|trace|traceability|frontier|take|monitor|fidelity|replication|replications|reproduce|bundle
<folder>` and `openlab study replicate <paper> <folder>`: the study verbs from a shell.

Each verb is a `Study` method, which writes the verb's action node; this file parses, asks a person to confirm a
person's act, and prints. A person's act (`holdout`, `amend --person`, `screen --person`, `approve`) shows what it
will sign and asks for the person's passphrase, which unlocks the person key that signs its nodes, so an
agent cannot run it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import new, open as open_study, replicate
from .errors import StudyError
from .show import brief, line, selected


def _confirm(study, verb: str, what: str, details=()):
    """A person's act: show exactly what will be signed and ask for the person's passphrase, which unlocks
    the person key; returns the context in which the act's person nodes are signed. Refused without the key, a
    terminal or the passphrase, so an agent cannot run it."""
    from ..agents import person
    from .personkey import signing
    try:
        signer = person.unlock_interactive(what, details)
    except person.NotAPerson as e:
        raise StudyError(f"`openlab study {verb}` is a person's act, signed with the person key: {e}",
                         "the passphrase is the person's confirmation; an agent's shell does not have it",
                         f"a person must run `openlab study {verb} ...` in their own terminal") from None
    return signing(signer)


def _verify_text(r: dict) -> list:
    """The chain, then the person signatures: forged or unpinned ones fail; legacy unsigned ones warn."""
    sig = r.get("signatures") or {}
    chain = r["broken_at"] is None
    out = [f"{'ok' if r['ok'] else 'FAILED'}: {r['nodes']} nodes, head {r['head'][:12]}" if chain
           else f"BROKEN at seq {r['broken_at']} of {r['nodes']}: the record was edited"]
    out.append(f"person signatures: {sig.get('signed', 0)} valid" + (f", key {sig['pinned']}" if sig.get("pinned")
                                                                    else ", no person key pinned yet"))
    ws = r.get("workspace")
    if ws is not None:
        out.append(f"workspace pin: {ws['pin'] or 'none (openlab person pin)'}"
                   + (f" (pin chain {' -> '.join(ws['chain'])})" if len(ws["chain"]) > 1 else "")
                   + ("" if not ws["pin"] or not sig.get("pinned") or sig["pinned"] == ws["pin"]
                      else f"; the study pins {sig['pinned']}"))
        if ws["conflict"]:
            out.append(f"  TAMPERED: {ws['conflict']}")
        out += [f"  warning: control store event {x['seq']}: {x['why']}" for x in ws.get("invalid") or []]
    out += [f"  INVALID seq {x['seq']} {x['id']}: {x['why']}" for x in sig.get("invalid", [])]
    out += [f"  warning: seq {x['seq']} {x['id']} ({x['session'] or 'no session'}): {x['why']}"
            for x in sig.get("legacy", [])]
    return out


def _looking(s) -> tuple:
    """What a holdout look will sign: the look number, the study, the code and data hashes."""
    from .fingerprint import fingerprint
    looks, fp = s.status()["looks"], fingerprint(s)
    code = ", ".join(f"{k} {v[:12]}" for k, v in sorted(fp["code"].items())) or "none"
    data = ", ".join(f"{k} {v[:12]}" for k, v in sorted(fp["data"].items())) or "none"
    return (f"holdout look {looks['used'] + 1} of {looks['budget']} on {s.spec.name}",
            [f"spec sha {fp['spec_sha'][:12]}", f"code sha {code}", f"data sha {data}"])


def _changes(s) -> list:
    """The fields an amendment would change (for the person to read before signing)."""
    try:
        from . import spec as _spec
        sealed = _spec.sealed(s.record)
        if sealed is None:
            return []
        now, was = _spec.load(s.folder).raw, sealed.raw
        out = []

        def walk(a, b, path):
            if isinstance(a, dict) and isinstance(b, dict):
                for k in sorted(set(a) | set(b)):
                    walk(a.get(k), b.get(k), f"{path}.{k}" if path else k)
            elif a != b:
                out.append(f"{path}: {b!r} -> {a!r}")
        walk(now, was, "")
        return out[:12] or ["no field changed"]
    except Exception:                                               # noqa: BLE001 -- the prompt never fails the act
        return []


def _show(a) -> dict:
    """One node in full: its kind, id, author, time, links and body (a result as its scores, with every group)."""
    rec = open_study(a.folder).record
    n = rec.get(a.id)
    if n is None:
        raise StudyError(f"no node {a.id} in this study", fix="openlab study runs / openlab study status list the ids")
    return {"id": n.id, "kind": n.kind, "actor": n.actor, "by": n.session, "at": n.at, "payload": n.payload,
            "links": [{"rel": rel, "to": dst} for _, rel, dst in rec.links(n.id)]}


def _show_text(r: dict) -> list:
    head = [f"{r['id']}  {r['kind']}  by {r['by'] or r['actor']}  at {r['at'][:19]}"]
    head += [f"  {x['rel']} -> {x['to']}" for x in r["links"]]
    if r["kind"] == "result":
        return head + _result_text({**r, "kind": "result"})[1:]
    return head + json.dumps(r["payload"], indent=1, default=str).splitlines()


def _why(a) -> dict:
    """The node and what it depends on, each naming the agent that wrote it (with its role, from the study's
    agent nodes)."""
    from dataclasses import asdict
    s = open_study(a.folder)
    roles = {n.payload["id"]: n.payload for n in s.record.find("agent")}

    def by(b: dict) -> dict:
        p = roles.get(b.get("agent") or "")
        who = b.get("agent") or b["actor"]
        return {**b, "by": f"{who} ({p['role'] or p['kind']})" if p and (p["role"] or p["kind"]) else who}
    return {"id": a.id, "node": [by(brief(asdict(s.record.get(a.id))))],
            "depends_on": [by(brief(n)) for n in s.why(a.id)]}


def _holdout(a) -> dict:
    s = open_study(a.folder)
    what, details = _looking(s)
    with _confirm(s, "holdout", what, [*details, f"reason: {a.reason}"]):
        print("running the holdout look...", file=sys.stderr, flush=True)
        return s.holdout(a.reason, person=True)


def _search(a) -> dict:
    return open_study(a.folder).search()


def _search_text(sc: dict) -> list:
    out = [f"clock t {sc['t']:.2f}   c(t) {sc['c']:.2f}   w(t) {sc['w']:.2f}   best {sc['best'] or '-'}   "
           f"{sc['since_best']} runs since a new best" + (f"   stagnant lines: {', '.join(sc['stagnant'])}"
                                                           if sc["stagnant"] else "")]
    if not sc["rows"]:
        return out + ["no research run yet"]
    out.append(f"  {'run':<18} {'op':<8} {'value':>9} {'Q':>5} {'n':>3} {'c·U':>5} {'cost':>5} {'uct':>5} "
               f"{'elite':>5} {'P':>5}")
    for r in sc["rows"]:
        v = f"{r['value']:.4g}" if isinstance(r["value"], (int, float)) else "-"
        out.append(f"  {r['id']:<18} {r['op']:<8} {v:>9} {r['Q']:>5.2f} {r['n']:>3} {sc['c'] * r['U']:>5.2f} "
                   f"{r['cost']:>5.2f} {r['uct']:>5.2f} {r['elite']:>5.2f} {r['P']:>5.2f}")
    return out


def _select(a) -> dict:
    return open_study(a.folder).select()


def _lesson(a) -> dict:
    return open_study(a.folder).lesson(a.rule, a.direction, a.basis, a.tags.split(","), a.margin)


def _lessons(a) -> dict:
    return {"lessons": open_study(a.folder).lessons()}


def _lessons_text(out: dict) -> list:
    if not out["lessons"]:
        return ["no lessons yet: `openlab study lesson FOLDER --rule ... --direction ... --basis comparison:...`"]
    return [f"{x['id']}  {x['direction']:<8} {x['rule']}\n    basis {x['basis']} {x['shown']}; tags "
            f"{', '.join(x['tags'])}; cited {x['track']['cited']}, held {x['track']['held']}" for x in out["lessons"]]


def _check(a) -> dict:
    return open_study(a.folder).check(a.code)


def _check_text(n: dict) -> list:
    p = n["payload"]
    return [f"{n['id']}  {'ok' if p['ok'] else 'red'}"] + [
        f"  {'ok  ' if c['ok'] else 'FAIL'} {c['name']:<12} {c['seconds']:>6.2f}s  {c['detail']}" for c in p["checks"]]


def _ensemble(a) -> dict:
    return open_study(a.folder).ensemble(a.members.split(","), a.method)


def _critiques(a) -> dict:
    return {"critiques": open_study(a.folder).critiques()}


def _critiques_text(out: dict) -> list:
    if not out["critiques"]:
        return ["no critiques"]
    lines = []
    for c in out["critiques"]:
        t = c["track"]
        lines.append(f"{c['id']} on {c['on']} by {c['model']}: {t['findings']} findings, {t['confirmed']} confirmed, "
                     f"{t['not confirmed']} not, {t['dismissed']} dismissed, {t['open']} open")
        lines += [f"  {i}. [{t['by_finding'][i]}] {f['where']} {f['text']}" for i, f in enumerate(c["findings"])]
    return lines


def _holdout_export(a) -> dict:
    """The predictions of a holdout run awaiting the person's score (openlab.study.custody.export)."""
    from .custody import export
    return export(open_study(a.folder), a.run)


def _holdout_score(a) -> dict:
    """The person's score of a holdout run whose labels they keep, recorded as its result: a person's act."""
    from .custody import decode
    s = open_study(a.folder)
    body = decode(a.result)
    m = s.spec.evaluate["metrics"][0]
    v = (body.get("metrics", {}).get(m) or {})
    details = [f"run {a.run}: {m} {v.get('value')} [{v.get('lo')}, {v.get('hi')}] over {body.get('n')} rows",
               f"labels sha {(body.get('scored') or {}).get('labels_sha', '')[:12]}, scored with openlab "
               f"{(body.get('scored') or {}).get('openlab')}"]
    with _confirm(s, "holdout-score", f"record the holdout score of {s.spec.name}", details):
        return s.holdout_score(a.run, body)


def _score(a) -> dict:
    """On the person's machine: fetch a holdout run's predictions from HOST, score them against the labels file,
    and record the score there as a person act signed here (openlab.study.custody)."""
    import shlex
    import subprocess
    from pathlib import Path
    from .custody import encode, read_labels, score
    from ..agents.remote import act_on
    remote = f"{shlex.quote(a.openlab)} study holdout-export {shlex.quote(a.folder)} --json" + (
        f" --run {shlex.quote(a.run)}" if a.run else "")
    if a.cd:
        remote = f"cd {shlex.quote(a.cd)} && {remote}"
    p = subprocess.run(["ssh", "-T", a.host, remote], capture_output=True, text=True)
    try:
        exported = json.loads(p.stdout)
    except ValueError:
        raise StudyError(f"{a.host} sent no predictions", (p.stderr or p.stdout)[-500:]) from None
    if exported.get("ok") is False:
        raise StudyError(exported["error"], exported.get("meaning", ""), exported.get("fix", ""))
    spec = exported["spec"]
    labels = read_labels(Path(a.labels), spec["split"]["labels_key"], spec["target"]["column"])
    body = score(exported, labels)
    m = spec["evaluate"]["metrics"][0]
    print(f"scored here: {m} {body['metrics'][m]['value']:.4g} over {body['n']} rows; recording it on {a.host}",
          file=sys.stderr)
    rc = act_on(a.host, ["study", "holdout-score", a.folder, "--run", exported["run"], "--result", encode(body)],
                openlab=a.openlab, cd=a.cd)
    if rc:
        raise StudyError(f"the score was not recorded on {a.host}", f"exit {rc}")
    return {"run": exported["run"], "metrics": body["metrics"], "n": body["n"], "scored": body["scored"]}


def _release(a) -> dict:
    """Give back a frontier item; with --person, any agent's lease (a dead agent's, say), as the person."""
    s = open_study(a.folder)
    if not a.person:
        return s.release(a.item)
    from .frontier import leases
    holder = leases(s).get(a.item)
    with _confirm(s, "release --person", f"release {a.item} of {s.spec.name}",
                  [f"held by {holder['agent']} until {holder['expires']} UTC" if holder else "not leased now"]):
        return s.release(a.item, actor="person")


def _amend(a) -> dict:
    s = open_study(a.folder)
    if a.person:
        with _confirm(s, "amend --person", f"a person's amendment of {s.spec.name}",
                      _changes(s) + [f"reason: {a.reason}"]):
            return s.amend(a.reason, person=True)
    return s.amend(a.reason, person=a.person)


def _fetch(a) -> dict:
    """The source's full text (fetched once); with --page, that page's text to quote from."""
    s = open_study(a.folder)
    d = s.fetch_text(a.source)
    out = {"id": d["id"], "source": a.source, "n_pages": d["payload"]["n_pages"], "url": d["payload"]["url"]}
    if a.page:
        key = "layout" if a.layout else "pages"
        if not d["payload"].get(key):
            raise StudyError("this document has no layout text", "it was fetched before openlab kept one",
                             "drop --layout")
        pages = json.loads(s.record.blob(d["payload"][key]))
        if not 1 <= a.page <= len(pages):
            raise StudyError(f"no page {a.page}", f"the document has pages 1..{len(pages)}")
        out.update(page=a.page, text=pages[a.page - 1])
    return out


def _screen(a) -> dict:
    if not a.quote.strip() and not a.person:
        raise StudyError("screen needs --quote", "every agent's screening decision quotes the passage of the source "
                         "it rests on, and openlab checks the passage is there",
                         'add --quote "<20+ characters of its abstract or full text>" (openlab study fetch FOLDER SOURCE '
                         "--page N prints a page)")
    s = open_study(a.folder)
    if a.person:
        with _confirm(s, "screen --person", f"a person's screening decision on {a.source}: {a.decision}",
                      [f"criterion {a.criterion}", f"rationale: {a.rationale}"]):
            return s.screen(a.source, a.decision, a.criterion, a.rationale, a.quote, person=True)
    return s.screen(a.source, a.decision, a.criterion, a.rationale, a.quote, person=a.person)


def _json(text: str, flag: str, example: str):
    try:
        return json.loads(text)
    except ValueError as e:
        raise StudyError(f"{flag} is not JSON", str(e), f"{flag} '{example}'") from None


def _extract(a) -> dict:
    fields = _json(a.fields, "--fields", '{"crps": 0.9, "method": "EMOS"}')
    units = {}
    for u in a.unit:
        k, eq, v = u.partition("=")
        if not eq or not k.strip() or not v.strip():
            raise StudyError(f"--unit {u} is not FIELD=UNIT", fix="e.g. --unit coverage90=percent")
        units[k.strip()] = v.strip()
    return open_study(a.folder).extract(a.source, a.page, a.quote, fields, a.kind, a.section, units, a.header)


def _claim(a) -> dict:
    inputs = _json(a.inputs, "--inputs", '{"m": "result:0123456789ab#metrics.mae.value"}')
    return open_study(a.folder).claim(a.statement, a.state, a.rests_on, a.value, a.expression, inputs, a.tests,
                                      a.searched, a.needs, a.answers)


def _approve(a) -> dict:
    s = open_study(a.folder)
    with _confirm(s, "approve", f"a person's {a.decision} on {a.on} in {s.spec.name}", [f"reason: {a.reason}"]):
        return s.approve(a.on, a.decision, a.reason, person=True)


def _tree_text(nodes: list, depth: int = 0) -> list:
    out = []
    for n in nodes:
        out.append(f"{'  ' * depth}{n['status']:<12} {n['id']}  {n['text']}")
        out += _tree_text(n.get("children", []), depth + 1)
    return out


def _monitor(a) -> dict:
    s = open_study(a.folder)
    return s.monitor_schedule(a.schedule) if a.schedule else s.monitor()


def _reproduce(a) -> dict:
    if a.bundle:
        from .reproduce import from_bundle
        return from_bundle(a.bundle)
    if not a.folder:
        raise StudyError("reproduce needs a study folder or --bundle <tar>")
    return open_study(a.folder).reproduce(a.run, a.all)


def _new(a) -> dict:
    s = new(a.folder, a.template)
    return {"folder": str(s.folder), "name": s.spec.name, "template": a.template}


def _evidence(a) -> dict:
    s = open_study(a.folder)
    if a.table or a.csv:
        return {"table": s.evidence_table(), "csv": a.csv}
    return {"evidence": s.evidence(a.source)}


def _evidence_text(r: dict) -> list:
    if "table" not in r:
        return [f"{e['id']}  {e['key']}  {line({'kind': 'evidence', 'payload': e})}"
                for e in r["evidence"]] or ["no evidence on record"]
    from .evidence import TABLE_COLUMNS
    if r["csv"]:
        import csv
        import io
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=TABLE_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(r["table"])
        return [buf.getvalue().rstrip("\n")]
    if not r["table"]:
        return ["no evidence on record"]
    cols = ("cite", "year", "method", "dataset", "metric", "value", "page", "quote", "agent")
    cells = [[("" if row[c] is None else f"{row[c]:.4g}" if isinstance(row[c], float) else str(row[c]))
              for c in cols] for row in r["table"]]
    width = [min(48, max(len(c), *(len(x[i]) for x in cells))) for i, c in enumerate(cols)]
    fit = lambda xs: "  ".join((x if len(x) <= w else x[:w - 1] + "…").ljust(w)        # noqa: E731
                               for x, w in zip(xs, width)).rstrip()
    return [fit([c.upper() for c in cols])] + [fit(x) for x in cells]


def _sources(a) -> dict:
    return {"sources": open_study(a.folder).sources(a.status)}


def _source_line(s: dict) -> str:
    if s["conflict"]:
        who = "CONFLICT " + "; ".join(f"{c['agent']} {c['decision']} ({c['criterion']})" for c in s["conflict"])
    elif s["decision"]:
        who = f"{s['decision']} ({s['criterion']}) by {s['agent'] or s['by']}"
    else:
        who = "unscreened"
    return f"{s['id']}  {s['key']}  {s['title'][:70]} ({s['year']}){'  RETRACTED' if s['retracted'] else ''}  {who}"


def _sources_text(r: dict) -> list:
    return [_source_line(s) for s in r["sources"]] or ["no sources"]


def _replicate(a) -> dict:
    s = replicate(a.paper, a.folder, a.mode, a.methods.split(","), a.metric, a.tolerance, a.max_tolerance, a.kind,
                  a.code, a.paper_unit)
    return {"folder": str(s.folder), **s.replications()}


def _run(a) -> dict:
    return open_study(a.folder).run(note=a.note, code=a.code, seed=a.seed, parent=a.parent, op=a.op, refs=a.ref,
                                    cites=a.cite)


def _seal(a) -> dict:
    """The first research run, on purpose: refused before anything runs when the study cannot be sealed; else what
    sealing freezes and costs is printed first (stderr under --json), then the run."""
    from . import sealing
    s = open_study(a.folder)
    try:
        terms = s.seal_terms(a.folder)
    except StudyError:
        return s.seal_run(note=a.note, rel=a.folder)        # refused the same way, and the refusal recorded
    print("\n".join(sealing.lines(terms) + ["running the first research run now ..."]),
          file=sys.stderr if a.json else sys.stdout, flush=True)
    return s.seal_run(note=a.note, rel=a.folder)


def _seal_text(r: dict) -> list:
    st = r.get("seal") or {}
    return [f"sealed: study.toml is {st.get('plan', 'the plan')} ({st.get('at', '')}); edits now go through "
            "`openlab study amend`"] + _result_text(r)


def _materialise(a) -> dict:
    """The datasets built, and a warning per recipe edited after the file it made (rebuilt only with --refresh)."""
    from .data import stale
    s = open_study(a.folder)
    out = s.materialise(a.name, refresh=a.refresh)
    warnings = stale(s, [a.name] if a.name else None, how="--refresh")
    return {**out, **({"warnings": warnings} if warnings else {})}


def _running(folder) -> list:
    """The study's runs in flight (an agent's detached run)."""
    from .detached import live
    return live(Path(folder))


def running(r: dict) -> str:
    from .detached import line as one
    return one(r)


VERBS = {
    "new": _new,
    "materialise": lambda a: _materialise(a),
    "run": _run,
    "seal": _seal,
    "status": lambda a: {**open_study(a.folder).status(), "running": _running(a.folder)},
    "why": lambda a: _why(a),
    "show": _show,
    "amend": _amend,
    "holdout": _holdout,
    "holdout-export": _holdout_export,
    "search": _search,
    "check": _check,
    "ensemble": _ensemble,
    "critiques": _critiques,
    "select": _select,
    "lesson": _lesson,
    "lessons": _lessons,
    "holdout-score": _holdout_score,
    "score": _score,
    "verify": lambda a: open_study(a.folder).verify(),
    "forecast": lambda a: open_study(a.folder).forecast(model_version=a.model),
    "fill": lambda a: open_study(a.folder).fill(),
    "schedule": lambda a: open_study(a.folder).schedule(a.action),
    "discover": lambda a: open_study(a.folder).discover(
        progress=None if a.json else lambda line: print(line, file=sys.stderr, flush=True)),
    "fetch": _fetch,
    "find": lambda a: open_study(a.folder).find(a.source, a.text, a.around, a.n),
    "runs": lambda a: {**open_study(a.folder).runs(), "running": _running(a.folder)},
    "paper-lint": lambda a: open_study(a.folder).paper_lint(),
    "screen": _screen,
    "prisma": lambda a: open_study(a.folder).prisma(),
    "extract": _extract,
    "evidence": _evidence,
    "sources": _sources,
    "compare": lambda a: open_study(a.folder).compare(a.a, a.b, a.metric),
    "lattice": lambda a: open_study(a.folder).lattice(a.by),
    "recall": lambda a: open_study(a.folder).recall(a.gold, a.since),
    "paper": lambda a: _paper(a),
    "verify-paper": lambda a: open_study(a.folder).verify_paper(),
    "claim": _claim,
    "question": lambda a: open_study(a.folder).question(a.text, a.part_of, a.out_of_scope),
    "hypothesis": lambda a: open_study(a.folder).hypothesis(a.statement, a.prediction, a.falsified_by, a.question,
                                                            a.tag, a.mechanism, a.differs_from, a.difference),
    "test": lambda a: open_study(a.folder).test(a.of, a.result, a.baseline, a.alpha, a.family),
    "assume": lambda a: open_study(a.folder).assume(a.statement, a.rests_on, a.breaks_if),
    "break": lambda a: open_study(a.folder).break_(a.assumption, a.reason, a.shown_by),
    "dead-end": lambda a: open_study(a.folder).dead_end(a.of, a.statement, a.reason, a.tag, a.shown_by),
    "contra": lambda a: open_study(a.folder).contra(a.claim, a.query, a.found),
    "tree": lambda a: open_study(a.folder).tree(),
    "claims": lambda a: {"claims": open_study(a.folder).claims()},
    "approve": _approve,
    "trace": lambda a: open_study(a.folder).trace(a.claim),
    "traceability": lambda a: open_study(a.folder).traceability(),
    "frontier": lambda a: open_study(a.folder).frontier(a.n),
    "take": lambda a: open_study(a.folder).take(a.item, minutes=a.minutes),
    "release": _release,
    "promote": lambda a: open_study(a.folder).promote(a.branch),
    "monitor": _monitor,
    "replicate": _replicate,
    "fidelity": lambda a: open_study(a.folder).fidelity(a.paper, a.ours, a.arm, a.tolerance, a.metric, a.justification,
                                                        a.group),
    "replication": lambda a: open_study(a.folder).replication(a.paper, a.ours, a.gated_by, a.arm, a.tolerance, a.metric,
                                                              a.justification, a.group),
    "replications": lambda a: open_study(a.folder).replications(),
    "reproduce": _reproduce,
    "bundle": lambda a: open_study(a.folder).bundle(a.out),
}


def _status_text(s: dict) -> list:
    out = [f"{s['name']} ({s['mode']}): {s['question']}",
           f"sealed {s['sealed_at']} ({s['plan']}): study.toml is frozen; an edit needs `openlab study amend`"
           if s["sealed"] else "not sealed: the first research run seals (freezes) study.toml and is trial 1; "
           "`openlab study seal STUDY` says what it freezes, then runs it"]
    out += [f"amended {m['at']} by {m['actor']} ({', '.join(m['changed'])}): {m['reason']}" for m in s["amendments"]]
    out += [f"dataset {d['name']}: {d['rows']} rows, sha {d['sha'][:12]}" for d in s["datasets"]]
    if s["mode"] != "review":                       # a review runs no trials
        def where(x: dict) -> str:
            return f" ({', '.join(f'{n} in {k}' for k, n in x.get('shared', {}).items())})" if x.get("shared") else ""
        out.append(f"trials {s['trials']['used']} of {s['trials']['budget']}{where(s['trials'])}, "
                   f"holdout looks {s['looks']['used']} of {s['looks']['budget']}{where(s['looks'])} "
                   "(a look scores the held-out rows once; a person's act)")
    out += [running(x) for x in s.get("running", [])]
    if s["last_result"]:
        r = s["last_result"]
        out.append(f"last result {r['id']} ({r['at']}): {line({'kind': 'result', 'payload': r})}"
                   + (f" ({selected(s['trials']['used'])})" if r.get("segment") == "research" else ""))
    return out


def _runs_text(r: dict) -> list:
    shared = ", ".join(f"{n} in {k}" for k, n in r["shared"].items())
    out = [f"{r['trials']} of {r['budget']} trials" + (f" ({shared})" if shared else "")
           + f": a result chosen among them is {selected(r['trials'])}"]
    branched = any(x.get("branch") for x in r["runs"])          # a branch column once a branch's copy ran
    if branched:
        out.append(f"{'':>9}  {'run':<16}  branch")
    for x in r["runs"]:
        m = x["metric"] or {}
        head = (f"trial {x['trial']:>3}" if x["trial"] else f"{x['segment']:>9}") + f"  {x['id']}  " \
            + (f"{x.get('branch') or 'main':<6}  " if branched else "") + f"code v{x['code_version']}"
        what = (f"  {m['name']} {_fmt(m)}" if m.get("name") and "value" in m else "")
        pd = next(iter((x["paired"] or {}).items()), None)
        out.append(head + f"  audit {x['audit']}" + what
                   + (f"  vs {pd[0]} {pd[1]['diff']:+.3g} (p {pd[1]['p']:.2g})" if pd and pd[1].get("diff") is not None
                      and pd[1].get("p") is not None else "")
                   + (f"  cached x{x['cached_reruns']}" if x["cached_reruns"] else "")
                   + (f"  changed {', '.join(x['changed'])}" if x["changed"] else "")
                   + (f"  \"{x['note']}\"" if x["note"] else ""))
    out += [running(x) for x in r.get("running", [])]
    return out if r["runs"] or r.get("running") else out + ["no runs yet: openlab study run"]


def _lint_text(r: dict) -> list:
    out = [("ok: the paper builds" if r["ok"] else f"{len(r['findings'])} open finding(s): the paper does not build")
           + f"; {len(r['bindings'])} bindings, results {selected(r['trials'])}"]
    out += [f"  {f['location']} [{f['rule']}] {f['message']} (id {f['id']})" for f in r["findings"]]
    out += [f"  ablation {b['macro']} at {b['at']}: {b['earlier'] or 'the current code'}" for b in r["bindings"]
            if b.get("ablation")]
    out += [f"  waived {f['location']} [{f['rule']}] by {f['waived_by']}" for f in r["waived"]]
    return out


def _paper(a) -> dict:
    """`study paper`: build, record and sign; --draft renders it as it stands; --init starts paper.tex."""
    from .paper import build
    s = open_study(a.folder)
    if getattr(a, "init", False):
        return build.init(s)
    if getattr(a, "draft", False):
        return build.draft(s)
    return s.paper()


def _paper_text(r: dict) -> list:
    if "carried_over" in r:
        return [f"wrote {r['file']}" + (f", carried over {', '.join(r['carried_over'])}" if r["carried_over"] else ""),
                r["next"]]
    if "compiled" in r:
        head = [f"draft {r['draft']}" if r["compiled"] else f"the draft does not compile: {r['error']}"]
        head += [f"  {e['file']}:{e['line']}: {e['message']}" for e in r["latex_errors"]]
        head += [f"  {f['file']}:{f['line']} [{f['rule']}] {f['message']}" for f in r["findings"]]
        return head + ([f"  ... {r['more_findings']} more"] if r["more_findings"] else []) + [r["next"]]
    return _node_text(r) + [f"pdf paper/build/main.pdf, signed {r['payload']['signed']}"]


def _node_text(n: dict) -> list:
    level = n.get("kind") == "claim" and (n.get("payload") or {}).get("level")
    return [f"{n['id']}  {line(n)}"] + (LEVELS if level else [])


def _fmt(m: dict) -> str:
    if m.get("value") is None:
        return f"n/a ({m.get('reason', '')})"
    iv = f" [{m['lo']:.4g}, {m['hi']:.4g}]" if m.get("lo") is not None else ""
    return f"{m['value']:.4g}{iv}"


def _result_text(r: dict) -> list:
    """A run's result in full: the audit verdict, every metric with its interval, each baseline, and the model
    minus each baseline on the primary metric."""
    if r.get("kind") != "result":
        return _node_text(r)
    p = r["payload"]
    out = [r["trial"]["summary"]] if r.get("trial") else []
    out += [f"{r['id']}  audit: {r.get('audit', '?')}  scored rows: {p.get('n', '?')}"
            + (f"  branch {r['branch']}" if r.get('branch') else '')]
    out += [f"  {k:<14} {_fmt(m)}" for k, m in p["metrics"].items()]
    for b, ms in p.get("baselines", {}).items():
        out.append(f"  baseline {b}: " + ", ".join(f"{k} {_fmt(m)}" for k, m in ms.items()))
    for b, d in p.get("paired", {}).items():
        out.append(f"  model - {b} ({d.get('metric', '')}): " + _diff(d))
    for g, body in (p.get("groups") or {}).items():          # evaluate.by: each entity's own scores
        out.append(f"  group {g} ({body.get('n', '?')} rows): "
                   + ", ".join(f"{k} {_fmt(m)}" for k, m in body.get("metrics", {}).items()))
        out += [f"    baseline {b}: " + ", ".join(f"{k} {_fmt(m)}" for k, m in ms.items())
                for b, ms in body.get("baselines", {}).items()]
    pol = p.get("policy") or {}
    if pol:
        out.append(f"  policy over {pol.get('decisions', '?')} decisions"
                   + (f" ({pol['skipped']} without a value in every arm left out)" if pol.get("skipped") else "") + ":")
        out += [f"    {k:<12} {_fmt(m)}" for k, m in pol.get("metrics", {}).items()]
        out += [f"    {k:<12} {_fmt(m)} (act's {k}, mean per decision)" for k, m in (pol.get("series") or {}).items()]
        for b, ms in pol.get("baselines", {}).items():
            out.append(f"    baseline {b}: " + ", ".join(f"{k} {_fmt(m)}" for k, m in ms.items()))
        for b, d in pol.get("paired", {}).items():
            out.append(f"    policy - {b}: " + _diff(d))
    return out


def _diff(d: dict) -> str:
    return (f"{d['diff']:.4g} [{d['lo']:.4g}, {d['hi']:.4g}], p {d['p']:.3g}, better: {d['better']}"
            + (f" (closer to {d['nominal']:g})" if d.get("nominal") is not None and d.get("better") else "")
            if d.get("diff") is not None else f"n/a ({d.get('reason', '')})")


def _replications_text(r: dict) -> list:
    t = r["target"]
    out = [f"{t['key']} ({t['kind']}): {t['title']}",
           "rebuild: " + {True: "faithful (the last check of every baseline number passed)", False: "NOT faithful",
                          None: "no fidelity yet"}[r["rebuilt"]]]
    out += [f"  fidelity {f['id']}  {f['method'] or f['arm']}: " + line({"kind": "fidelity", "payload": f})
            for f in r["fidelity"]]
    out += [f"  claim {c['id']}  {c['method'] or c['arm']}: " + line({"kind": "replication", "payload": c})
            for c in r["replications"]]
    return out


def _forecast_text(n: dict) -> list:
    p = n["payload"]
    return _node_text(n) + [f"signed {p['signed']}, expires {p['expires']}"] + [
        f"gate FAILED: {c['expect']}" + (f" ({c.get('error') or 'held ' + str(c.get('actual', 'false'))})")
        for c in p["gate"]["checks"] if not c["ok"]]


def _lattice_text(r: dict) -> list:
    out = ["the lattice: the evidence as a table, a row per protocol and a column per method; a gap is a cell no "
           "extracted source fills",
           f"{len(r['rows'])} protocols x {len(r['columns'])} {r['by']}: {r['filled']} of {r['cells']} cells filled, "
           f"{r['gaps']} gaps"]
    for row in r["rows"]:
        out.append("  " + ", ".join(f"{k} {v}" for k, v in row["coordinates"].items()))
        out += [f"    {col}: " + "; ".join(", ".join(f"{k} {e[k]}" for k in r["values"] if k in e) + f" (p.{e['page']})"
                                          for e in cells) for col, cells in row["cells"].items()]
        out += [f"    gap: {', '.join(row['gaps'])}"] if row["gaps"] else []
    return out


def _cron_text(s: dict, section: str) -> list:
    out = [f"{section} {s['state']}: " + (f"cron {s['declared']['cron']}" if s["declared"] else
                                          f"study.toml has no [{section}] cron")]
    return out + ([f"next run {s['next_run']}"] if s.get("next_run") else [])


def _schedule_text(s: dict) -> list:
    out = _cron_text(s, "schedule")
    f = s["last_forecast"]
    return out + ([f"last forecast {f['id']} at {f['t']}: gate {'passed' if f['gate'] else 'FAILED'}"] if f else [])


def _chain_text(n: dict, rel: str = "", depth: int = 0) -> list:
    """A trace's tree, one node a line, indented under what rests on it."""
    head = "  " * depth + (f"{rel} -> " if rel else "") + n["id"]
    if "shown" in n:
        return [f"{head}  (shown above)"]
    out = [f"{head}  {n['summary']}"]
    if n["kind"] == "run":
        out.append("  " * (depth + 2) + f"code {n['code']}, env {n['env']}")
    for link in n["links"]:
        out += _chain_text(link["node"], link["rel"], depth + 1)
    return out


def _traceability_text(r: dict) -> list:
    pct = lambda x: f" ({x:.0%})" if x is not None else ""                    # noqa: E731
    return [f"{r['traceable']} of {r['claims']} claims traceable{pct(r['share'])}; approved: "
            f"{r['approved']['traceable']} of {r['approved']['claims']}{pct(r['approved']['share'])}"] + \
        [f"{b['claim']}  {m}" for b in r["broken"] for m in b["missing"]]


def _frontier_text(r: dict) -> list:
    head = ["score = info / hours: how much a step should tell us (0 to 1) per hour it costs"] if r["items"] else []
    return head + [f"{i['score']:>9.3f}  {i['kind']:<12} {i['target']} -> {i['verb']} (info {i['info']:.3g}, cost "
            f"{i['cost']:.0f} s){' HELD by ' + i['held_by']['agent'] if i.get('held_by') else ''}: {i['why']}  "
            f"[{i['id']}]" for i in r["items"]] \
        or ["the frontier is empty"]


def _monitor_text(r: dict) -> list:
    if "state" in r:                                                    # --schedule
        w = r["last_watch"]
        return _cron_text(r, "monitor") + ([f"last pass {w['id']} at {w['at']}"] if w else [])
    out = [f"{r['id']}  {r['summary']}"]
    out += [f"  retracted {i['key']} ({i['note']}): claims {', '.join(i['claims']) or 'none'}; papers "
            f"{', '.join(i['papers']) or 'none'}" for i in r["retractions"]]
    out += [f"  no retraction check answered for {len(r['unchecked'])} sources"] if r["unchecked"] else []
    return out + r["errors"] + ([f"notified: {r['delivery']}"] if r["delivery"] else [])


def _reproduce_text(r: dict) -> list:
    out = [f"bundle {r['bundle']} ({r['study']}, head {r['head'][:12]}, signed by key {r['key_id']})"
           ] if "bundle" in r else []
    for x in r["reproductions"]:
        out.append(f"{x['id']}  {x['run']}: " + ("matched" if x["matched"] else f"DIFFERS: {x['differs']}")
                   + (f" (max |diff| {x['max_abs_diff']:.3g})" if x["max_abs_diff"] else "")
                   + f", {x['seconds']:.1f} s")
        out += [f"  env {k}: {d['run']} -> {d['now']}" for k, d in x["env_diff"].items()]
    return out


TEXT = {
    "new": lambda r: [f"new study {r['name']} in {r['folder']}",
                      f"next: edit [review] in study.toml, then `openlab study discover {r['folder']}` (the first "
                      f"discover seals the protocol); the steps are in {r['folder']}/README.md"
                      if r["template"] == "review" else
                      f"next: `openlab study materialise {r['folder']}`, then `openlab study seal {r['folder']}` "
                      "(the first research run: it freezes study.toml and is trial 1)"],
    "materialise": lambda r: [x for n in r.get("datasets", [r]) for x in _node_text(n)]
                             + [f"warning: {w}" for w in r.get("warnings", [])],
    "run": _result_text,
    "seal": _seal_text,
    "status": _status_text,
    "why": lambda r: [f"{n['id']}  by {n.get('by') or n['actor']}  {n['summary']}" for n in r.get("node", [])]
                     + ([f"  {n['id']}  {n.get('by') or n['actor']:<6}  {n['summary']}" for n in r["depends_on"]]
                        or [f"  {r['id']} depends on nothing"]),
    "show": _show_text,
    "amend": _node_text,
    "holdout": _node_text,
    "holdout-export": lambda out: [json.dumps(out)],
    "search": _search_text,
    "check": _check_text,
    "ensemble": _node_text,
    "critiques": _critiques_text,
    "select": _node_text,
    "lesson": _node_text,
    "lessons": _lessons_text,
    "holdout-score": _node_text,
    "score": lambda out: [f"{out['run']} scored: " + ", ".join(f"{k} {v.get('value'):.4g} [{v.get('lo'):.4g}, "
                                                                f"{v.get('hi'):.4g}]" for k, v in out["metrics"].items()
                                                                if isinstance(v.get('value'), (int, float)))
                          + f" over {out['n']} rows (labels sha {out['scored']['labels_sha'][:12]})"],
    "verify": lambda r: _verify_text(r),
    "forecast": _forecast_text,
    "fill": lambda r: [x for n in r["outcomes"] for x in _node_text(n)] or ["no forecast's label is known yet"],
    "schedule": _schedule_text,
    "discover": lambda r: [f"{r['id']}  {line({'kind': 'coverage', 'payload': r['coverage']})}",
                           f"{r['queries']} queries recorded, {r['new_sources']} new sources"]
    + ([f"took {r['seconds']:g} s: the {len(r['timings'])} index calls took {r['calls_seconds']:g} s between them, so "
        f"running them side by side saved {r['saved_seconds']:g} s"
        + (f"; {r['cached_calls']} identical calls answered once" if r.get("cached_calls") else "")]
       if "seconds" in r else []) + r["errors"],
    "fetch": lambda r: [f"{r['id']}  {r['n_pages']} pages from {r['url']}"]
                       + ([f"--- page {r['page']} of {r['n_pages']}", r["text"]] if "page" in r else []),
    "runs": _runs_text,
    "find": lambda r: [f"p.{h['page']} ({h['form']}): \"{h['quote']}\"" for h in r["hits"]]
    + ([f"{r['total'] - len(r['hits'])} more (--n)"] if r["total"] > len(r["hits"]) else [])
    + ([f"next: {r['next']}"] if r["hits"] else [f"{r['needle']!r} is not in the fetched text of {r['source']}"]),
    "paper-lint": _lint_text,
    "screen": _node_text,
    "prisma": lambda r: [f"{k}: {v}" for k, v in r.items()],
    "extract": _node_text,
    "evidence": _evidence_text,
    "sources": _sources_text,
    "compare": lambda r: _node_text(r) + ([
        f"  {r['payload']['paired']['n']} shared rows ({r['payload']['paired']['interval']}); better: "
        f"{r['payload']['paired']['better'] or 'neither'}"] if r["payload"].get("paired") else []),
    "lattice": _lattice_text,
    "recall": lambda r: [f"{r['id']}  recall {r['recall']:.1%}: {r['matched']} of {r['references']} references of "
                         f"{r['gold']} ({r['title']}) on record, {r['matched_included']} of them included",
                         f"filters {r['filters']}; left out {r['left_out']}"]
                        + [f"  missed {m['key']} ({m['year']}): {m['title']}" for m in r["missed"]],
    "paper": _paper_text,
    "verify-paper": lambda r: [f"valid: {r['paper']} at head {r['head'][:12]}, key {r['key_id']}, signed {r['signed_at']}"
                               if r["ok"] else f"INVALID: {r['why']}"],
    **dict.fromkeys(("claim", "question", "hypothesis", "test", "assume", "break", "dead-end", "approve"), _node_text),
    "contra": lambda r: [f"{r['id']}  contradiction search: {r['state']} (coverage {r['coverage']})"]
                        + ([f"{r['search']['queries']} queries, {r['search']['new_sources']} new sources, "
                            f"{len(r['search']['sources'])} reached"] + r["search"]["errors"] if "search" in r else []),
    "tree": lambda r: _tree_text(r["questions"]) or ["no questions on record"],
    "trace": lambda r: _chain_text(r["chain"]) + ([f"MISSING: {m}" for m in r["missing"]] or ["the chain is complete"]),
    "traceability": _traceability_text,
    "frontier": _frontier_text,
    "take": lambda r: [f"{'renewed' if r['renewed'] else 'took'} {r['item']['id']} ({r['item']['kind']} on "
                       f"{r['item']['target']}): action {r['action']}, leased to {r['lease']['agent']} until "
                       f"{r['lease']['expires']} UTC", f"next: {r['item']['verb']} ({r['item']['why']})"],
    "promote": lambda r: [f"promoted branch {r['branch']}: {', '.join(r['changed']) or 'nothing changed'} copied over the "
                          "study's code"],
    "release": lambda r: [f"released {r['item']} (held by {r['released']}): action {r['id']}"
                          if r.get("id") else f"released {r['item']}"],
    "monitor": _monitor_text,
    "replicate": lambda r: [f"replication study in {r['folder']}"] + _replications_text(r)
                           + [f"next: the steps in {r['folder']}/README.md"],
    "fidelity": _node_text,
    "replication": _node_text,
    "replications": _replications_text,
    "reproduce": _reproduce_text,
    "bundle": lambda r: [f"wrote {r['bundle']}: {r['files']} files, {r['bytes']} bytes, head {r['head'][:12]}, "
                         f"signed by key {r['key_id']}"],
    "claims": lambda r: [f"{c['id']}  {c['state']}, {c['verdict']}" + (f", {c['level']}" if c["level"] else "")
                         + (f", {selected(c['selected_from'])}" + (
                             f" ({c['trials_now']} now)" if c["trials_now"] != c["selected_from"] else "")
                            if c.get("selected_from") else "")
                         + (f", AT RISK ({', '.join(c['at_risk'])})" if c["at_risk"] else "")
                         + (f", {c['approval']} by a person" if c["approval"] else "")
                         + (f", contradiction search {c['contra']}" if c["contra"] else "") + f": {c['statement']}"
                         + (f"\n    ({c['level_note']})" if c.get("level_note") else "")
                         for c in r["claims"]] + LEVELS if r["claims"] else ["no claims on record"],
}


#: the words a study's output uses, one line each (`openlab study --help`)
GLOSSARY = """words:
  seal          the first research run (`openlab study run`, or `openlab study seal` on purpose) freezes
                study.toml: the split and holdout, budget, instrument pins, primary metric. It spends trial 1 and
                never touches the holdout; after it an edit needs `openlab study amend` with a reason, and the
                person's signature for those protected fields
  look          one scoring of the held-out rows; only a person spends one, counted against budget.holdout_looks
  trial         a distinct research run, counted against budget.trials (an identical rerun is cached and free)
  alpha budget  the error budget tests spend (alpha-investing): a rejection earns some back, a miss costs
  frontier      what to do next, ranked by information per hour: what a step should tell us over its time
  lattice       the evidence as a table (protocol x method); a gap is a cell no extracted source fills
  level         how a claim's numbers were made: recorded (research runs), measured (the holdout),
                reproduced (every run behind it re-ran and matched)"""

#: the claim levels, printed under a list of claims
LEVELS = ["levels: recorded = from research runs; measured = from the holdout; reproduced = its runs re-ran and matched"]

def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("folder")
    common.add_argument("--json", action="store_true", help="the reply as one JSON object")
    p = argparse.ArgumentParser(prog="openlab study", description="A study is a folder: study.toml, model.py, "
                                "recipes/, data/ and its record. These verbs act on it.", epilog=GLOSSARY,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    sub.add_parser("new", parents=[common], help="scaffold a study").add_argument(
        "--template", default="temporal", choices=("temporal", "static", "review"),
        help="temporal or static (an experiment), or review (a literature review: the protocol and the steps)")
    mat = sub.add_parser("materialise", parents=[common], help="build and register the datasets")
    mat.add_argument("name", nargs="?", help="one dataset (default: all)")
    mat.add_argument("--refresh", action="store_true", help="run every recipe again (and re-fetch providers), "
                     "not only for a missing file")
    rn = sub.add_parser("run", parents=[common], help="a research run (seals study.toml first)")
    rn.add_argument("--note", default="")
    rn.add_argument("--code", default="", help="a branch's working copy, branches/<id> (default: the study's model.py)")
    rn.add_argument("--seed", type=int, default=None, help="the model's seed for this run (default split.seed; the "
                    "folds and holdout keep split.seed): one run per seed when the paper averages over seeds")
    rn.add_argument("--op", default="", choices=["draft", "improve", "debug", "fuse", "ensemble"],
                    help="how this run is made from its parent: a draft starts a new line (no --parent)")
    rn.add_argument("--parent", default="", help="the run it is made from (run:<id> or its result:<id>)")
    rn.add_argument("--ref", action="append", default=[], help="a run of another branch a fuse borrows from "
                    "(repeat it)")
    rn.add_argument("--cite", action="append", default=[], help="a lesson this run applies (repeat it): "
                    "lesson:<id>, or <study>/lesson:<id>")
    sl = sub.add_parser("seal", parents=[common], help="seal study.toml now: the first research run (trial 1), "
                        "after saying what it freezes; refused once sealed or before the data is materialised")
    sl.add_argument("--note", default="")
    sub.add_parser("promote", parents=[common], help="copy a branch's working copy over the study's model.py "
                   "(recorded)").add_argument("branch", help="the branch id (branches/<id>)")
    sub.add_parser("status", parents=[common], help="the summary")
    sub.add_parser("why", parents=[common], help="what a node depends on").add_argument("id")
    sub.add_parser("show", parents=[common], help="one node in full (a result: every score, each group's too)"
                   ).add_argument("id")
    amend = sub.add_parser("amend", parents=[common], help="record the edit of study.toml since the seal")
    amend.add_argument("--reason", required=True)
    amend.add_argument("--person", action="store_true", help="a person's amendment (split, budget, ...)")
    sub.add_parser("search", parents=[common], help="the search over the runs, scored now: the clock, c(t), w(t), "
                   "each run's Q, U, cost, UCT and P(pick)")
    sub.add_parser("select", parents=[common], help="apply the sealed [selection] rule and record what it picks")
    ck = sub.add_parser("check", parents=[common], help="module checks in seconds: lint, import, smoke, determinism")
    ck.add_argument("--code", default="", help="a branch's copy, branches/<id> (default: the study's code)")
    en = sub.add_parser("ensemble", parents=[common], help="combine recorded runs into a counted run of its own")
    en.add_argument("--members", required=True, help="comma-separated run or result ids")
    en.add_argument("--method", default="mean", choices=["mean", "rank", "weighted"])
    sub.add_parser("critiques", parents=[common], help="the critiques, their findings and track records")
    le = sub.add_parser("lesson", parents=[common], help="a lesson resting on a comparison (or a fixing run)")
    le.add_argument("--rule", required=True, help="what to do, in a sentence")
    le.add_argument("--direction", required=True, choices=["improves", "hurts", "no_gain", "fixes"])
    le.add_argument("--basis", required=True, help="comparison:<id> (study compare), or run:<id> for a fix")
    le.add_argument("--tags", required=True, help="comma-separated: what it is about")
    le.add_argument("--margin", type=float, default=None, help="no_gain: how small a difference counts as none")
    sub.add_parser("lessons", parents=[common], help="the study's lessons, what each rests on, its track record")
    he = sub.add_parser("holdout-export", parents=[common], help="the predictions of a holdout run awaiting the "
                        "person's score (labels kept by the person), as JSON")
    he.add_argument("--run", default="", help="the holdout run (default: the latest awaiting a score)")
    hs = sub.add_parser("holdout-score", parents=[common], help="record the person's score of such a run (a "
                        "person's act; `openlab study score` runs it)")
    hs.add_argument("--run", required=True)
    hs.add_argument("--result", required=True, help="the result body, base64 JSON")
    sc = sub.add_parser("score", help="on your machine: score a holdout run on HOST against your labels file and "
                        "record it there, signed here (split.labels = \"person\")")
    sc.add_argument("host", help="the ssh host that keeps the record")
    sc.add_argument("folder", help="the study folder there (relative to --cd)")
    sc.add_argument("--labels", required=True, help="your labels: CSV or parquet with split.labels_key and the target")
    sc.add_argument("--run", default="", help="the holdout run (default: the latest awaiting a score)")
    sc.add_argument("--openlab", default="openlab", help="the openlab command there")
    sc.add_argument("--cd", default="", help="the directory there (the workspace)")
    sc.add_argument("--json", action="store_true")
    sub.add_parser("holdout", parents=[common], help="a person's holdout look: scores the held-out rows once, "
                   "counted against the look budget").add_argument(
        "--reason", required=True)
    sub.add_parser("verify", parents=[common], help="recompute the record's hash chain")
    sub.add_parser("forecast", parents=[common], help="a forward forecast, gated and signed (exit 1: gate failed)"
                   ).add_argument("--model", type=int, default=None, help="a model version (default: [forecast] model)")
    sub.add_parser("fill", parents=[common], help="record the outcomes of forecasts whose labels are known")
    sub.add_parser("schedule", parents=[common], help="the [schedule] cron line in your crontab").add_argument(
        "action", nargs="?", default="status", choices=("status", "on", "off"))
    sub.add_parser("discover", parents=[common], help="run the sealed [review]'s searches and record coverage")
    fe = sub.add_parser("fetch", parents=[common], help="a source's open-access full text, by page")
    fe.add_argument("source")
    fe.add_argument("--page", type=int, default=0, help="print this page's text (from 1), to quote from")
    fe.add_argument("--layout", action="store_true", help="the page as laid out, table rows intact")
    fi = sub.add_parser("find", parents=[common], help="where a source's full text says a phrase or number: page "
                        "and the exact sentence, ready for extract --quote")
    fi.add_argument("source")
    fi.add_argument("text", help="the words or the number to look for (0.90 finds 0.9)")
    fi.add_argument("--around", type=int, default=0, help="sentences of context on each side")
    fi.add_argument("--n", type=int, default=10, help="hits to show")
    sub.add_parser("runs", parents=[common], help="every trial: metric, audit, code version, note, cached reruns")
    sub.add_parser("paper-lint", parents=[common], help="the paper's bindings and findings, without building it")
    screen = sub.add_parser("screen", parents=[common], help="a screening decision on one source")
    screen.add_argument("source")
    screen.add_argument("--decision", required=True, choices=("include", "exclude", "maybe"))
    screen.add_argument("--criterion", required=True, help="an include or exclude criterion id of the review")
    screen.add_argument("--rationale", required=True)
    screen.add_argument("--quote", default="", help="required (only a person's decision may omit it): a passage (20+ "
                        "characters) of its abstract or full text, as the source has it; openlab checks it is there "
                        "(openlab study fetch --page N prints a page)")
    screen.add_argument("--person", action="store_true", help="a person's decision (resolves a maybe)")
    sub.add_parser("prisma", parents=[common], help="the PRISMA flow, from the record")
    ext = sub.add_parser("extract", parents=[common], help="evidence quoted from one page of a source's full text")
    ext.add_argument("source")
    ext.add_argument("--page", type=int, required=True, help="the page of the fetched document (from 1)")
    ext.add_argument("--quote", required=True, help="the passage or table row, as the page has it")
    ext.add_argument("--fields", default="{}", help='the values read from it, as JSON: {"crps": 0.9, ...}')
    ext.add_argument("--header", default="", help="a table whose methods are its columns: the header line as laid "
                     "out; the method may be read from it when each number sits under its column")
    ext.add_argument("--kind", default="", choices=("", "text", "passage", "table", "figure"),
                     help="default: table or figure when --section says so (\"Table 2\"), else text (a passage)")
    ext.add_argument("--section", required=True, help='where on the page, e.g. "Table 2"')
    ext.add_argument("--unit", action="append", default=[], metavar="FIELD=UNIT",
                     help="how the paper writes a number: percent or fraction (repeat), e.g. coverage90=percent when "
                     "the table's header says %%; a number the quote writes with %% is percent without it")
    ev = sub.add_parser("evidence", parents=[common], help="the evidence on record, or the evidence matrix (--table)")
    ev.add_argument("--source", default="", help="one source's")
    ev.add_argument("--table", action="store_true", help="the evidence matrix: source, year, method, dataset, "
                    "metric, value, page, quote, agent")
    ev.add_argument("--csv", action="store_true", help="the matrix as CSV (implies --table)")
    sub.add_parser("sources", parents=[common], help="the sources with their standing decision and its agent"
                   ).add_argument("--status", default="all", choices=("all", "unscreened", "include", "exclude", "maybe",
                                                                       "conflict", "retracted"))
    cmp = sub.add_parser("compare", parents=[common], help="two results on one metric (same protocol only): two of "
                         "the study's results paired on their shared rows, or evidence under [review] comparable_on")
    cmp.add_argument("a", help="a result or evidence id")
    cmp.add_argument("b", help="a result or evidence id")
    cmp.add_argument("--metric", required=True, help="a metric of openlab.study.metrics, e.g. crps")
    sub.add_parser("lattice", parents=[common], help="the evidence as a table: protocols x methods, with the "
                   "gaps").add_argument(
        "--by", default="method", help="the text field whose values are the columns")
    rc = sub.add_parser("recall", parents=[common], help="recall against a published review's references")
    rc.add_argument("--gold", required=True, help="the review's DOI")
    rc.add_argument("--since", default=None, help="keep references published from this date (default: review.since)")
    pa = sub.add_parser("paper", parents=[common], help="build, record and sign the paper (paper/paper.tex)")
    pa.add_argument("--draft", action="store_true", help="render it as it stands into paper/build/draft.pdf, with its "
                    "findings; record nothing")
    pa.add_argument("--init", action="store_true", help="start paper/paper.tex from the template (an older "
                    "paper.toml layout is carried over)")
    sub.add_parser("verify-paper", parents=[common], help="check the signed paper against the record")
    cl = sub.add_parser("claim", parents=[common], help="a claim; openlab computes its verdict and level")
    cl.add_argument("statement")
    cl.add_argument("--state", required=True, choices=("observed", "derived", "inferred", "hypothesized", "unknown"))
    cl.add_argument("--rests-on", action="append", default=[], help="evidence, result or claim id (repeat)")
    cl.add_argument("--value", type=float, default=None, help="its number: an input's (observed) or computed (derived)")
    cl.add_argument("--expression", default="", help="derived: LaTeX over the inputs, e.g. '{a} - {b}'")
    cl.add_argument("--inputs", default="{}", help='JSON name -> "<node-id>#<dotted path>"')
    cl.add_argument("--tests", default="", help="hypothesized: the hypothesis id")
    cl.add_argument("--searched", default="", help="unknown: the coverage id")
    cl.add_argument("--needs", action="append", default=[], help="an assumption id (repeat)")
    cl.add_argument("--answers", action="append", default=[], help="a question or hypothesis id (repeat)")
    qu = sub.add_parser("question", parents=[common], help="a question of the question tree")
    qu.add_argument("text")
    qu.add_argument("--part-of", default="", help="the broader question's id")
    qu.add_argument("--out-of-scope", default="", help="why it is recorded but not pursued")
    hy = sub.add_parser("hypothesis", parents=[common], help="a falsifiable hypothesis under a question")
    hy.add_argument("statement")
    hy.add_argument("--question", required=True)
    hy.add_argument("--prediction", required=True)
    hy.add_argument("--falsified-by", required=True)
    hy.add_argument("--mechanism", default="")
    hy.add_argument("--tag", action="append", required=True, help="what it is about (repeat)")
    hy.add_argument("--differs-from", action="append", default=[], help="a dead end it differs from (repeat)")
    hy.add_argument("--difference", default="", help="what is different from those dead ends now")
    te = sub.add_parser("test", parents=[common], help="test a claim or hypothesis, spending the error budget")
    te.add_argument("of")
    te.add_argument("--result", required=True)
    te.add_argument("--baseline", required=True, help="the result's paired entry (a baseline's name), or "
                    "result:<id>: another of the study's results, paired on their shared rows")
    te.add_argument("--alpha", type=float, required=True,
                    help="alpha_j, at most b / (1 + b) for b the alpha budget left (the error budget tests spend)")
    te.add_argument("--family", default="primary")
    asm = sub.add_parser("assume", parents=[common], help="an assumption claims may need")
    asm.add_argument("statement")
    asm.add_argument("--rests-on", required=True, help="what it rests on")
    asm.add_argument("--breaks-if", required=True, help="the test that could break it")
    br = sub.add_parser("break", parents=[common], help="an assumption shown not to hold")
    br.add_argument("assumption")
    br.add_argument("--reason", required=True)
    br.add_argument("--shown-by", action="append", required=True, help="a result or evidence id (repeat)")
    de = sub.add_parser("dead-end", parents=[common], help="a falsified hypothesis or abandoned run")
    de.add_argument("of", help="the hypothesis or run id")
    de.add_argument("--statement", required=True)
    de.add_argument("--reason", required=True)
    de.add_argument("--tag", action="append", required=True, help="(repeat; a hypothesis's tags included)")
    de.add_argument("--shown-by", action="append", required=True, help="a result, evidence or test id (repeat)")
    co = sub.add_parser("contra", parents=[common], help="search for what contradicts a claim")
    co.add_argument("claim")
    co.add_argument("--query", action="append", default=[], help="a search for it (repeat; none: reuse the last)")
    co.add_argument("--found", action="append", default=[], help="evidence that contradicts it (repeat)")
    sub.add_parser("tree", parents=[common], help="the question tree with derived statuses")
    sub.add_parser("claims", parents=[common], help="the claims with their verdict, level (recorded, measured, "
                   "reproduced) and risk")
    ap = sub.add_parser("approve", parents=[common], help="a person's approval, rejection or waiver")
    ap.add_argument("on", help="a claim, paper or (waive only) evidence id")
    ap.add_argument("--decision", required=True, choices=("approve", "reject", "waive"))
    ap.add_argument("--reason", required=True)
    sub.add_parser("trace", parents=[common], help="a claim's provenance chain").add_argument("claim")
    sub.add_parser("traceability", parents=[common], help="the share of claims whose chain is complete")
    sub.add_parser("frontier", parents=[common], help="what to do next, ranked by information per hour (what a "
                   "step should tell us over what it costs)").add_argument(
        "--n", type=int, default=20, help="items to show")
    tk = sub.add_parser("take", parents=[common], help="take a frontier item (recorded, and leased to you)")
    tk.add_argument("item")
    tk.add_argument("--minutes", type=float, default=None, help="the lease (default 30); taking it again renews")
    rl = sub.add_parser("release", parents=[common], help="give back a frontier item you hold")
    rl.add_argument("item")
    rl.add_argument("--person", action="store_true", help="a person releases any agent's lease (signed)")
    sub.add_parser("monitor", parents=[common], help="a literature monitoring pass: new sources, retractions, "
                   "outcomes").add_argument(
        "--schedule", choices=("status", "on", "off"), default=None, help="the [monitor] cron line instead")
    rp = sub.add_parser("replicate", help="a study that replicates one paper: pinned, scaffolded, its target set")
    rp.add_argument("paper", help="its arXiv id, DOI or PDF file")
    rp.add_argument("folder")
    rp.add_argument("--json", action="store_true", help="the reply as one JSON object")
    rp.add_argument("--mode", required=True, choices=("temporal", "static"))
    rp.add_argument("--methods", required=True, help="the paper's methods as class names, comma-separated; the last "
                    "is Model, the others its baselines")
    rp.add_argument("--metric", required=True, help="the metric its claims are in, e.g. crps")
    rp.add_argument("--tolerance", type=float, required=True, help="how close a rebuilt value must be to the paper's, "
                    "in the metric's own unit (coverage: a fraction, so 0.02 is two points)")
    rp.add_argument("--paper-unit", default="", choices=("", "percent", "fraction"),
                    help="how the paper prints shares such as coverage or accuracy (percent: 92.62 is 0.9262), for "
                    "evidence whose quote has no %% sign")
    rp.add_argument("--max-tolerance", type=float, default=0.0, help="the widest tolerance a justification may ask "
                    "for (default: --tolerance)")
    rp.add_argument("--kind", default="replication", choices=("replication", "reproduction"),
                    help="rebuilt from the text, or run from the authors' code (--code)")
    rp.add_argument("--code", default="", help="the authors' code (a reproduction runs it)")
    for verb, what in (("fidelity", "a baseline of the paper rebuilt within tolerance?"),
                       ("replication", "one claim of the paper against the rebuild; openlab computes the verdict")):
        v = sub.add_parser(verb, parents=[common], help=what)
        v.add_argument("--paper", required=True, help="the evidence id of the paper's value")
        v.add_argument("--ours", required=True, action="append", help="the result id (repeat, or comma-separate: the "
                       "paper's number is the mean over several runs, e.g. one per horizon or seed)")
        v.add_argument("--arm", default="model", help="model, or a baseline the run scored (e.g. model:EMOS_gl)")
        v.add_argument("--tolerance", type=float, default=None, help="default: the target's")
        v.add_argument("--metric", default="", help="default: the primary metric")
        v.add_argument("--justification", default="", help="why, for a tolerance wider than the target's")
        v.add_argument("--group", default="", help="one group's scores (evaluate.by = \"key\": a station, a dataset)")
        if verb == "replication":
            v.add_argument("--gated-by", action="append", required=True, help="a fidelity id (repeat)")
    sub.add_parser("replications", parents=[common], help="the replication report: fidelity and verdicts")
    rd = sub.add_parser("reproduce", help="run recorded runs again from the record alone and compare")
    rd.add_argument("folder", nargs="?", default="")
    rd.add_argument("--json", action="store_true", help="the reply as one JSON object")
    one = rd.add_mutually_exclusive_group()
    one.add_argument("--run", default="", help="a run id (default: the latest run)")
    one.add_argument("--all", action="store_true", help="every run")
    one.add_argument("--bundle", default="", help="a bundle tar: reproduce every run in a temp dir, from it alone")
    sub.add_parser("bundle", parents=[common], help="a signed tar to reproduce the study from").add_argument("out")
    return p


def main(argv=None) -> int:
    a = _parser().parse_args(argv)
    try:
        out = VERBS[a.verb](a)
    except StudyError as e:
        if a.json:
            print(json.dumps({"ok": False, "error": e.error, "meaning": e.meaning, "fix": e.fix}))
        else:
            print("\n".join(f"{k}: {v}" for k, v in (("error", e.error), ("meaning", e.meaning), ("fix", e.fix))
                            if v), file=sys.stderr)
        return 1
    print(json.dumps(out, default=str) if a.json else "\n".join(TEXT[a.verb](out)))
    failed = (a.verb in ("verify", "verify-paper", "paper-lint") and not out["ok"]) or (
        a.verb == "forecast" and not out["payload"]["gate"]["passed"]) or (
        a.verb == "reproduce" and not all(x["matched"] for x in out["reproductions"]))
    return 1 if failed else 0
