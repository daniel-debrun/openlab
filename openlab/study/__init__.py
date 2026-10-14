"""A study: a folder whose research openlab runs and records (study.toml, model.py, .openlab/record.db)."""

from __future__ import annotations

import importlib
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from . import schema
from . import spec as _spec
from .errors import StudyError
from .record import Node, Record, canonical
from .spec import Spec

TEMPLATES = ("temporal", "static", "review")


def _node(n: Node) -> dict:
    return {"id": n.id, "seq": n.seq, "kind": n.kind, "actor": n.actor, "agent": n.session, "at": n.at,
            "payload": n.payload}


def _later(name: str):
    """A module built in a later step; a clear error until it exists."""
    try:
        return importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as e:
        if e.name != f"{__name__}.{name}":
            raise
        raise StudyError(f"`study {name}` is not built yet") from None


class Study:
    def __init__(self, folder: Path):
        self.folder = Path(folder).resolve()
        self.record = Record(self.folder)
        try:                        # the study's own metrics declared before anything reads a metric by name
            _spec.sealed(self.record)
        except StudyError:
            pass

    @property
    def spec(self) -> Spec:
        """The effective sealed spec once sealed; study.toml before that."""
        return _spec.sealed(self.record) or _spec.load(self.folder)

    def _experiment(self, name: str):
        """An experiment engine (runner, holdout, forward); a review study runs none."""
        if self.spec.mode == "review":
            raise StudyError("a review study runs no experiment", "study.mode is review: [study] and [review] only",
                             "run the experiment in a study of its own (openlab study new)")
        return _later(name)

    def _act(self, verb: str, args: dict, actor: str, fn: Callable[[], dict]) -> dict:
        """Run a state-changing verb and write its action node, refusals included."""
        from .personkey import current as _signer
        if actor == "person" and _signer() is None:     # an unsigned person act is refused before it starts
            e = StudyError(f"`{verb}` is a person's act and must be signed with the person key",
                           "this process holds no unlocked person key, so it is not the person acting",
                           "the person runs it in their own terminal, where openlab asks for their passphrase "
                           "(once, first: `openlab person init`)")
            self.record.add("action", {"verb": verb, "args": args, "outcome": {
                "ok": False, "error": f"{e.error} (an unsigned attempt; recorded as the agent's)"}}, actor="agent")
            raise e
        try:
            out = fn()
        except Exception as e:
            self.record.add("action", {"verb": verb, "args": args, "outcome": {"ok": False, "error": str(e)}},
                            actor=actor)
            raise
        made = [out["id"]] if out.get("id") else list(out.get("ids", []))
        outcome = {"ok": True, "id": out.get("id"), **({"ids": out["ids"]} if "ids" in out else {})}
        self.record.add("action", {"verb": verb, "args": args, "outcome": outcome},
                        actor=actor, links=[("did", i) for i in made])
        return out

    def materialise(self, name: str | None = None, refresh: bool = False) -> dict:
        """Materialise one dataset, or every dataset in the spec when `name` is None."""
        from . import data

        def go() -> dict:
            if name is not None:
                return _node(data.materialise(self, name, refresh))
            nodes = [_node(data.materialise(self, n, refresh)) for n in self.spec.data]
            return {"datasets": nodes, "ids": [n["id"] for n in nodes]}
        return self._act("materialise", {"name": name, "refresh": refresh}, "agent", go)

    def seal(self, actor: str = "agent") -> dict:
        return self._act("seal", {}, actor, lambda: _node(_spec.seal(self.folder, self.record, actor)))

    def run(self, note: str = "", actor: str = "agent", code: str = "", seed: int | None = None, parent: str = "",
            op: str = "", refs: list | tuple = (), cites: list | tuple = ()) -> dict:
        """A research run: its result node, with `trial` first (cached, or which trial it spent and why its key is
        new) and the run's audit verdict. `code`: a branch's working copy (`branches/<id>`, openlab.study.branches)
        instead of the study's own model.py. `op`, `parent`, `refs`: the run's place in the search tree
        (openlab.study.runner.execute)."""
        def go() -> dict:
            from .branches import resolve
            code_dir, branch = resolve(self, code)
            node, trial = self._experiment("runner").execute(self, note=note, actor=actor, code_dir=code_dir,
                                                             branch=branch, seed=seed, parent=parent, op=op,
                                                             refs=tuple(refs), cites=tuple(cites))
            run = self.record.get(self.record.links(node.id, rel="of")[0][2])
            from .preflight import unchecked
            warn = None if trial.get("cached") else unchecked(self.record, run.payload["code"])
            if warn:                                    # a second line: the first still says the trial
                trial = {**trial, "warning": warn, "summary": f"{trial['summary']}\nwarning: {warn}"}
            return {"trial": trial, **_node(node), "audit": run.payload["audit"].get("verdict"),
                    **({"branch": branch} if branch else {})}
        return self._act("run", {"note": note, **({"code": code} if code else {}),
                                 **({"seed": seed} if seed is not None else {}),
                                 **({"parent": parent} if parent else {}), **({"op": op} if op else {}),
                                 **({"refs": list(refs)} if refs else {}),
                                 **({"cites": list(cites)} if cites else {})}, actor, go)

    def promote(self, branch: str, actor: str = "agent") -> dict:
        """Copy a branch's working copy over the study's own code (openlab.study.branches.promote), recorded."""
        br = _later("branches")
        try:
            d = br.path(self, branch)
            files = br._files(d) if d.is_dir() else {}
        except StudyError:
            files = {}
        return self._act("promote", {"branch": branch, "files": files}, actor, lambda: br.promote(self, branch))

    def amend(self, reason: str, person: bool = False) -> dict:
        """Record the edit of study.toml since the seal. A person's amendment before the first seal seals study.toml
        as the person's plan (how a person declares [study] fresh before any run: openlab.study.ledger)."""
        actor = "person" if person else "agent"

        def go() -> dict:
            if person and _spec.sealed(self.record) is None:
                if not reason.strip():
                    raise StudyError("an amendment needs a reason", fix="say why the plan changes")
                return _node(_spec.seal(self.folder, self.record, "person"))
            return _node(_spec.amend(self.folder, self.record, reason, person=person, actor=actor))
        return self._act("amend", {"reason": reason, "person": person}, actor, go)

    def holdout(self, reason: str, person: bool) -> dict:
        return self._act("holdout", {"reason": reason, "person": person}, "person" if person else "agent",
                         lambda: _node(self._experiment("holdout").look(self, reason=reason, person=person)))

    def search(self) -> dict:
        """The search over the runs, scored now (openlab.study.search): the clock, c(t), w(t), each run's numbers."""
        return _later("search").score(self)

    def select(self) -> dict:
        """Apply the sealed selection rule and record what it picked (openlab.study.search)."""
        return self._act("select", {}, "agent", lambda: _node(_later("search").record(self)))

    def lesson(self, rule: str, direction: str, basis: str, tags: list, margin: float | None = None) -> dict:
        """A lesson resting on a comparison (or, for a fix, a run): its direction is checked against the basis."""
        args = {"rule": rule, "direction": direction, "basis": basis, "tags": list(tags), "margin": margin}
        return self._act("lesson", args, "agent", lambda: _node(_later("lessons").record(self, **args)))

    def lessons(self) -> list:
        """The study's lessons with what each rests on and its track record."""
        return _later("lessons").listing(self)

    def check(self, code: str = "") -> dict:
        """Module checks of the study's code (or a branch's, `code`), recorded (openlab.study.preflight)."""
        return self._act("check", {"code": code} if code else {}, "agent",
                         lambda: _node(_later("preflight").check(self, code)))

    def ensemble(self, members: list, method: str = "mean") -> dict:
        """A combination of recorded runs, scored and recorded as a counted run (openlab.study.ensemble)."""
        args = {"members": list(members), "method": method}
        return self._act("ensemble", args, "agent", lambda: _node(_later("ensemble").build(self, **args)))

    def critique(self, on: str, model: str, prompt_sha: str, findings: list, cost_usd: float | None = None) -> dict:
        """A reviewer's findings, recorded as advice (openlab.study.critique)."""
        args = {"on": on, "model": model, "prompt_sha": prompt_sha, "findings": list(findings), "cost_usd": cost_usd}
        return self._act("critique", args, "agent", lambda: _node(_later("critique").record(self, **args)))

    def finding(self, critique: str, index: int, action: str, reason: str = "", by: str = "") -> dict:
        """The answer to one finding: checked (by a check) or dismissed (with the reason)."""
        args = {"critique": critique, "index": index, "action": action, "reason": reason, "by": by}
        return self._act("finding", args, "agent", lambda: _node(_later("critique").answer(self, **args)))

    def critiques(self) -> list:
        return _later("critique").listing(self)

    def holdout_score(self, run: str, body: dict) -> dict:
        """The person's score of a holdout run whose labels they keep, as its result (openlab.study.custody)."""
        args = {"run": run, "scored": body.get("scored"), "n": body.get("n")}
        return self._act("holdout_score", args, "person",
                         lambda: _node(_later("custody").record_score(self, run, body)))

    def forecast(self, now=None, model_version: int | None = None, actor: str = "openlab") -> dict:
        """A forward forecast at the latest decision time, gated, signed and recorded (openlab.study.forward)."""
        return self._act("forecast", {"now": now and now.isoformat(), "model_version": model_version}, actor,
                         lambda: _node(self._experiment("forward").forecast(self, now=now,
                                                                            model_version=model_version, actor=actor)))

    def fill(self, actor: str = "openlab") -> dict:
        """The outcomes of the forecasts whose labels are now known."""
        def go() -> dict:
            nodes = [_node(n) for n in self._experiment("forward").fill(self, actor=actor)]
            return {"outcomes": nodes, "ids": [n["id"] for n in nodes]}
        return self._act("fill", {}, actor, go)

    def schedule(self, action: str = "status", actor: str = "agent") -> dict:
        """The `[schedule] cron` line in the user's crontab: status, on or off."""
        forward = self._experiment("forward")
        if action == "status":                                          # a read: no action node
            return forward.schedule(self, action)
        return self._act("schedule", {"action": action}, actor, lambda: forward.schedule(self, action))

    def discover(self, progress=None) -> dict:
        """Run the sealed [review]'s searches (sealing study.toml first): queries, new sources, coverage; the calls
        run concurrently, `progress(line)` told one line per call."""
        return self._act("discover", {}, "agent", lambda: _later("literature").discover(self, progress=progress))

    def fetch_text(self, source: str) -> dict:
        return self._act("fetch_text", {"source": source}, "agent",
                         lambda: _node(_later("literature").fetch_text(self, source)))

    def screen(self, source: str, decision: str, criterion: str, rationale: str, quote: str = "",
               person: bool = False) -> dict:
        args = {"source": source, "decision": decision, "criterion": criterion, "rationale": rationale,
                "quote": quote, "person": person}
        lit = _later("literature")

        def go() -> dict:
            node = lit.screen(self, source, decision, criterion, rationale, quote, person=person)
            item = lit.flag_conflict(self, node)        # a conflict, or an agent's maybe: an inbox item
            return {**_node(node), **({"maybe" if decision == "maybe" else "conflict": item} if item else {})}
        return self._act("screen", args, "person" if person else "agent", go)

    def screen_many(self, decisions: list) -> dict:
        """Several agent screening decisions in one call, each {source, decision, criterion, rationale, quote}; each
        is its own screen and action node (a refusal refuses only that one)."""
        out = []
        for d in decisions:
            if not isinstance(d, dict):
                out.append({"ok": False, "error": "a decision is {source, decision, criterion, rationale, quote}"})
                continue
            try:
                n = self.screen(str(d.get("source", "")), str(d.get("decision", "")), str(d.get("criterion", "")),
                                str(d.get("rationale", "")), str(d.get("quote", "")))
                out.append({"ok": True, "source": d.get("source"), "id": n["id"],
                            **({"conflict": n["conflict"]} if n.get("conflict") else {}),
                            **({"inbox": n["maybe"]} if n.get("maybe") else {})})
            except StudyError as e:
                out.append({"ok": False, "source": d.get("source"), "error": e.error, "meaning": e.meaning})
        return {"decisions": out, "recorded": sum(r["ok"] for r in out), "refused": sum(not r["ok"] for r in out)}

    def prisma(self) -> dict:
        return _later("literature").prisma(self)

    def sources(self, status: str = "all") -> list[dict]:
        return _later("literature").sources(self, status)

    def recall(self, gold: str, since: str | None = None) -> dict:
        """Recall of the sources against a published review's references (openlab.study.literature.recall)."""
        return self._act("recall", {"gold": gold, "since": since}, "agent",
                         lambda: _later("literature").recall(self, gold, since))

    def extract(self, source: str, page: int, quote: str, fields: dict, kind: str = "",
                section: str = "", units: dict | None = None, header: str = "") -> dict:
        args = {"source": source, "page": page, "quote": quote, "fields": fields, "kind": kind, "section": section,
                **({"units": units} if units else {}), **({"header": header} if header else {})}
        return self._act("extract", args, "agent", lambda: _node(_later("evidence").extract(
            self, source, page, quote, fields, kind, section, units, header)))

    def find(self, source: str, text: str, around: int = 0, n: int = 10) -> dict:
        """Where a source's full text says `text` (a phrase or a number): page and exact sentence(s), each a quote
        `extract` accepts (openlab.study.evidence.find). The full text is fetched first if it is not on record."""
        lit = _later("literature")
        if lit.document(self.record, lit._source_node(self.record, source)) is None:
            self.fetch_text(source)
        return _later("evidence").find(self, source, text, around, n)

    def evidence(self, source: str = "") -> list[dict]:
        return _later("evidence").evidence(self, source)

    def evidence_table(self) -> list[dict]:
        """The evidence matrix (openlab.study.evidence.table): one row per number read, with its page, quote and agent."""
        return _later("evidence").table(self)

    def compare(self, a: str, b: str, metric: str) -> dict:
        return self._act("compare", {"a": a, "b": b, "metric": metric}, "agent",
                         lambda: _node(_later("evidence").compare(self, a, b, metric)))

    def lattice(self, by: str = "method") -> dict:
        return _later("evidence").lattice(self, by)

    def paper(self, person: bool = False) -> dict:
        """Build the paper (openlab.study.paper): refused while a binding is unresolved or stale or a finding of
        the lints and gates has no person's waiver; else the PDF, its Paper node and the signed bundle."""
        return self._act("paper", {"person": person}, "person" if person else "agent",
                         lambda: _node(_later("paper.build").build(self, person=person)))

    def paper_lint(self) -> dict:
        """The paper's bindings and findings, without building it."""
        return _later("paper.build").lint_report(self)

    def verify_paper(self) -> dict:
        return _later("paper.build").verify(self)

    def _organise(self, verb: str, args: dict, actor: str = "agent", fn: str = "") -> dict:
        """A verb of openlab.study.claims (claims, questions, tests, assumptions, dead ends) as one action node."""
        def go() -> dict:
            out = getattr(_later("claims"), fn or verb)(self, **args)
            return out if isinstance(out, dict) else _node(out)
        return self._act(verb, args, actor, go)

    def claim(self, statement: str, state: str, rests_on=(), value: float | None = None, expression: str = "",
              inputs: dict | None = None, tests: str = "", searched: str = "", needs=(), answers=()) -> dict:
        return self._organise("claim", {"statement": statement, "state": state, "rests_on": list(rests_on),
                                        "value": value, "expression": expression, "inputs": inputs or {},
                                        "tests": tests, "searched": searched, "needs": list(needs),
                                        "answers": list(answers)})

    def question(self, text: str, part_of: str = "", out_of_scope: str = "") -> dict:
        return self._organise("question", {"text": text, "part_of": part_of, "out_of_scope": out_of_scope})

    def hypothesis(self, statement: str, prediction: str, falsified_by: str, question: str, tags, mechanism: str = "",
                   differs_from=(), difference: str = "") -> dict:
        return self._organise("hypothesis", {"statement": statement, "prediction": prediction,
                                             "falsified_by": falsified_by, "question": question, "tags": list(tags),
                                             "mechanism": mechanism, "differs_from": list(differs_from),
                                             "difference": difference})

    def test(self, of: str, result: str, baseline: str, alpha_j: float, family: str = "primary") -> dict:
        return self._organise("test", {"of": of, "result": result, "baseline": baseline, "alpha_j": alpha_j,
                                       "family": family})

    def assume(self, statement: str, rests_on: str, breaks_if: str) -> dict:
        return self._organise("assume", {"statement": statement, "rests_on": rests_on, "breaks_if": breaks_if})

    def break_(self, assumption: str, reason: str, shown_by) -> dict:
        return self._organise("break", {"assumption": assumption, "reason": reason, "shown_by": list(shown_by)},
                              fn="break_")

    def dead_end(self, of: str, statement: str, reason: str, tags, shown_by) -> dict:
        return self._organise("dead_end", {"of": of, "statement": statement, "reason": reason, "tags": list(tags),
                                           "shown_by": list(shown_by)})

    def contra(self, claim: str, queries=(), found=()) -> dict:
        return self._organise("contra", {"claim": claim, "queries": list(queries), "found": list(found)})

    def approve(self, on: str, decision: str, reason: str, person: bool = False) -> dict:
        return self._organise("approve", {"on": on, "decision": decision, "reason": reason, "person": person},
                              "person" if person else "agent")

    def fidelity(self, paper: str, ours, arm: str = "model", tolerance: float | None = None, metric: str = "",
                 justification: str = "", group: str = "") -> dict:
        """Whether a baseline of the target paper was rebuilt (openlab.study.replication.fidelity)."""
        args = {"paper": paper, "ours": ours, "arm": arm, "tolerance": tolerance, "metric": metric,
                "justification": justification, **({"group": group} if group else {})}
        return self._act("fidelity", args, "agent", lambda: _node(_later("replication").fidelity(self, **args)))

    def replication(self, paper: str, ours, gated_by, arm: str = "model", tolerance: float | None = None,
                    metric: str = "", justification: str = "", group: str = "") -> dict:
        """One claim of the target paper against the rebuild, with its computed verdict."""
        args = {"paper": paper, "ours": ours, "gated_by": list(gated_by), "arm": arm, "tolerance": tolerance,
                "metric": metric, "justification": justification, **({"group": group} if group else {})}
        return self._act("replication", args, "agent", lambda: _node(_later("replication").replication(self, **args)))

    def replications(self) -> dict:
        return _later("replication").replications(self)

    def tree(self) -> dict:
        return _later("claims").tree(self)

    def claims(self) -> list[dict]:
        return _later("claims").claims(self)

    def why(self, id: str) -> list[dict]:
        return [_node(n) for n in self.record.why(id)]

    def neighbourhood(self, id: str, depth: int = 1) -> dict:
        """The nodes within `depth` links of a node and the typed edges between them (openlab.study.graph)."""
        return _later("graph").neighbourhood(self, id, depth)

    def trace(self, claim: str) -> dict:
        """A claim's provenance chain as a tree, and what is missing along it."""
        return _later("graph").trace(self, claim)

    def traceability(self) -> dict:
        return _later("graph").traceability(self)

    def depends_on_source(self, source: str) -> dict:
        return _later("graph").depends_on_source(self, source)

    def frontier(self, n: int | None = 20) -> dict:
        """What to do next, ranked by expected information per hour (openlab.study.frontier)."""
        return _later("frontier").frontier(self, n)

    def take(self, item: str, actor: str = "agent", minutes: float | None = None) -> dict:
        """Take a frontier item: its action node links the item's targets, so the choice is on the record, and
        leases the item to the acting agent for `minutes` (default 30): another agent's take is refused, naming the
        holder, until the holder releases it, ends, or the lease expires. The holder taking it again renews it."""
        from ..agents.identity import current
        fr = _later("frontier")
        self_id, minutes = current(), float(minutes or fr.LEASE_MINUTES)
        try:
            found, refused = fr.item(self, item), None
            holder = fr.leases(self).get(item)
            if holder and holder["agent"] != self_id:
                refused = StudyError(f"{item} is taken by {holder['agent']} until {holder['expires']} UTC",
                                     "a frontier item is leased to one agent at a time",
                                     "take another item; this one frees when that agent releases it or ends, or "
                                     "the lease expires")
        except StudyError as e:
            found, refused, holder = None, e, None
        lease = {"agent": self_id, "expires": fr.lease_until(minutes)}
        args = {"item": item, **({k: found[k] for k in fr.TAKEN} if found else {}), "lease": lease}

        def go() -> dict:
            if refused:
                raise refused
            return {"item": found, "ids": found["targets"]}
        out = self._act("take", args, actor, go)
        return {**out, "action": self.record.last("action", verb="take").id, "lease": lease,
                "renewed": bool(holder)}

    def release(self, item: str, actor: str = "agent") -> dict:
        """Give back a frontier item this agent holds (a person may release any)."""
        from ..agents.identity import current
        fr = _later("frontier")
        holder, self_id = fr.leases(self).get(item), current()

        def go() -> dict:
            if holder is None:
                raise StudyError(f"{item} is not leased now", "nothing to release")
            if holder["agent"] != self_id and actor != "person":
                raise StudyError(f"{item} is held by {holder['agent']}", "only its holder or a person releases it")
            return {"item": item, "released": holder["agent"]}
        return self._act("release", {"item": item}, actor, go)

    def monitor(self, actor: str = "openlab") -> dict:
        """One monitoring pass (openlab.study.monitor): discovery again, retraction checks, outcomes, a notification."""
        return self._act("monitor", {}, actor, lambda: _later("monitor").watch(self))

    def monitor_schedule(self, action: str = "status", actor: str = "agent") -> dict:
        """The `[monitor] cron` line in the user's crontab: status, on or off."""
        monitor = _later("monitor")
        if action == "status":                                          # a read: no action node
            return monitor.schedule(self, action)
        return self._act("monitor_schedule", {"action": action}, actor, lambda: monitor.schedule(self, action))

    def reproduce(self, run: str = "", every: bool = False, actor: str = "agent") -> dict:
        """Run a recorded run (default: the latest) or `every` run again from the record alone and compare it with
        what was recorded (openlab.study.reproduce)."""
        rep = _later("reproduce")

        def go() -> dict:
            nodes = rep.reproduce(self, run, every, actor)
            return {"reproductions": [rep.summary(n) for n in nodes], "ids": [n.id for n in nodes]}
        return self._act("reproduce", {"run": run, "every": every}, actor, go)

    def bundle(self, out) -> dict:
        """The study as a signed tar a third party can reproduce from (openlab.study.reproduce.bundle)."""
        return self._act("bundle", {"out": str(out)}, "agent", lambda: _later("reproduce").bundle(self, out))

    def reproductions(self) -> dict:
        return _later("reproduce").reproductions(self)

    def verify(self) -> dict:
        """The hash chain and every person signature, against the workspace's pin chain when there is one: every key
        that signs the study's person nodes must be one the workspace pinned, and openlab.toml must agree with the
        control store's pin chain (`workspace`: {pin, chain, conflict, invalid})."""
        try:
            from ..agents.person import pin_state
            ws = pin_state(self.folder)
        except Exception:                                           # noqa: BLE001 -- verify never needs the workspace
            ws = None
        out = self.record.verify(expected_key=ws["chain"] if ws and ws["chain"] else None)
        if ws is not None:
            out["workspace"] = {k: ws[k] for k in ("pin", "chain", "first", "toml", "conflict", "invalid")}
            if ws["conflict"]:
                out["ok"] = False
        return out

    def status(self) -> dict:
        rec, spec = self.record, self.spec
        plan = next(iter(rec.find("plan")), None)
        datasets = {n.payload.get("name"): n for n in rec.find("dataset")}          # the latest per name
        last = rec.last("result")
        from .ledger import shared
        ledger = shared(self.folder, rec)
        return {
            "name": spec.name, "question": spec.question, "mode": spec.mode,
            "sealed": plan is not None, "plan": plan and plan.id, "sealed_at": plan and plan.at,
            "amendments": [{"id": n.id, "at": n.at, "actor": n.actor, "reason": n.payload["reason"],
                            "changed": list(n.payload["changes"])} for n in rec.find("amendment")],
            "datasets": [{"name": k, "rows": n.payload.get("rows"), "sha": n.payload.get("sha256")}
                         for k, n in datasets.items()],
            "trials": {"used": ledger["trials"], "budget": spec.budget.get("trials", 0), "own": schema.trials_used(rec),
                       "shared": {k: v["trials"] for k, v in ledger["by"].items()}},     # a review has none
            "looks": {"used": ledger["looks"], "budget": spec.budget.get("holdout_looks", 0),
                      "own": len(rec.find("look")), "shared": {k: v["looks"] for k, v in ledger["by"].items()}},
            "last_result": last and self._summary(last),
        }

    def runs(self) -> dict:
        """Every run, oldest first: its trial number (research runs, by distinct key), segment, note, code version
        (v1, v2, ... by distinct code), the files that changed from the run it branches from, the audit verdict, its
        result's primary metric with interval and the paired differences, and how many reruns were cached on it.
        `trials` is the count a result is selected from (with the studies sharing this one's ledger)."""
        import hashlib
        from .ledger import shared
        rec, spec = self.record, self.spec
        primary = spec.evaluate["metrics"][0] if spec.mode != "review" else ""
        versions, keys, rows = {}, {}, []
        asked: dict = {}
        for a in rec.find("action", verb="run"):
            if a.payload["outcome"].get("ok") and a.payload["outcome"].get("id"):
                asked[a.payload["outcome"]["id"]] = asked.get(a.payload["outcome"]["id"], 0) + 1
        for r in rec.find("run"):
            p = r.payload
            code = hashlib.sha256(canonical(p["code"]).encode()).hexdigest()
            versions.setdefault(code, len(versions) + 1)
            if p["segment"] == "research":
                keys.setdefault(p["key"], len(keys) + 1)
            res = rec.links(r.id, rel="of", incoming=True)
            result = rec.get(res[0][0]) if res else None
            parent = rec.links(r.id, rel="branches_from")
            prev = rec.get(parent[0][2]).payload["code"] if parent else {}
            rows.append({
                "id": r.id, "at": r.at, "actor": r.actor, "segment": p["segment"], "branch": p.get("branch", ""),
                "trial": keys.get(p["key"]) if p["segment"] == "research" else None, "note": p.get("note", ""),
                "code_version": versions[code], "code": {f: h[:8] for f, h in p["code"].items()},
                "changed": sorted(f for f in {*p["code"], *prev} if p["code"].get(f) != prev.get(f)) if parent else [],
                "audit": p["audit"].get("verdict"), "seconds": p["seconds"],
                "result": result and result.id,
                "metric": result and {"name": primary, **result.payload["metrics"].get(primary, {})},
                "paired": result and {b: {k: d.get(k) for k in ("diff", "lo", "hi", "p", "better")}
                                      for b, d in result.payload.get("paired", {}).items()},
                "cached_reruns": max(0, asked.get(result.id, 0) - 1) if result else 0,
                "op": p.get("op", ""), "parent": parent[0][2] if parent else None,
                "refs": [d for _, _, d in rec.links(r.id, rel="refs")], "box": p.get("box", {}).get("host", "")})
        sh = shared(self.folder, rec)
        return {"primary": primary, "trials": sh["trials"], "budget": spec.budget.get("trials", 0),
                "shared": {k: v["trials"] for k, v in sh["by"].items()}, "runs": rows}

    def _summary(self, result: Node) -> dict:
        """A result with its primary metric (value and interval) and its run's truncation-audit verdict."""
        run = self.record.get(self.record.links(result.id, rel="of")[0][2])
        primary = self.spec.evaluate["metrics"][0]
        return {"id": result.id, "at": result.at, "run": run.id, "segment": run.payload["segment"],
                "primary": {"metric": primary, **result.payload["metrics"].get(primary, {})},
                "audit": run.payload["audit"].get("verdict"), "metrics": result.payload["metrics"]}


def open(folder) -> Study:
    folder = Path(folder)
    if not (folder / "study.toml").is_file():
        raise StudyError(f"{folder} is not a study (no study.toml)", fix="create one with `openlab study new <folder>`")
    return Study(folder)


def new(folder, template: str = "temporal") -> Study:
    """Scaffold a runnable study from a template: study.toml, model.py, recipes/, data/, and git; a review: study.toml
    with the protocol, and a README with the steps."""
    if template not in TEMPLATES:
        raise StudyError(f"no template `{template}`", f"templates: {', '.join(TEMPLATES)}")
    folder, src = Path(folder).resolve(), Path(__file__).parent / "templates" / template
    _not_nested(folder)
    files = [p.relative_to(src) for p in src.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    clash = [str(f) for f in [*files, Path(".gitignore")] if (folder / f).exists()]
    if clash:
        raise StudyError(f"{folder} already has {', '.join(clash)}", fix="choose a new folder; nothing was written")
    for f in files:
        (folder / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / f, folder / f)
    toml = folder / "study.toml"
    toml.write_text(toml.read_text(encoding="utf-8").replace('"__name__"', canonical(folder.name)), encoding="utf-8")
    if template != "review":                                        # a review has no data of its own
        (folder / "data").mkdir(exist_ok=True)
    (folder / ".gitignore").write_text(".openlab/stage/\n", encoding="utf-8")
    git = shutil.which("git")
    init = bool(git) and subprocess.run([git, "-C", str(folder), "rev-parse", "--is-inside-work-tree"],
                                        capture_output=True).returncode != 0
    if init:
        subprocess.run([git, "init", "-q", str(folder)], check=True)
    study = Study(folder)
    from .ledger import claim_origin
    claim_origin(folder)
    import secrets                              # the first node names this record alone (openlab.study.ledger tells
    study._act("new", {"template": template, "git_init": init, "record": secrets.token_hex(8)}, "agent",
               lambda: {})                      # records apart by it), even for two studies made in one second
    return study


def _not_nested(folder: Path) -> None:
    """Refuse a study inside another study's folder (any ancestor up to the workspace root holds study.toml)."""
    for d in folder.parents:
        if (d / "study.toml").is_file():
            raise StudyError(f"{folder} is inside the study {d}",
                             "a study inside another's folder is part of the outer study's code: the outer study's "
                             "runs would see its files (its model.py, its recipes' outputs), and the outer code hash "
                             "would change with every edit of the inner one",
                             f"make it a sibling folder, e.g. {d.parent / folder.name}; nothing was written")
        if (d / "openlab.toml").is_file():
            return


def replicate(src: str, folder, mode: str, methods, metric: str, tolerance: float, max_tolerance: float = 0.0,
              kind: str = "replication", code: str = "", paper_unit: str = "") -> Study:
    """A study replicating the paper `src` names (arXiv id, DOI or PDF): openlab.study.replication.new."""
    return _later("replication").new(src, folder, mode, list(methods), metric, tolerance, max_tolerance, kind, code,
                                     paper_unit)
