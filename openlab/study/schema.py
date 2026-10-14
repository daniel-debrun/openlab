"""The record's data model: a typed body per node kind, the links each kind may or must carry, and the
invariants checked before a node is written. A write that breaks any of them is refused, never stored.

A kind with no body here cannot be written at all, so every new feature starts as a schema.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import typing
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, ClassVar, Literal, Optional

from .clock import utc
from .errors import StudyError

if TYPE_CHECKING:
    from .record import Node, Record

ANY = ("*",)


@dataclass(frozen=True)
class Link:
    """A relation a kind may carry: to which kinds, and how many (max None = any number)."""
    rel: str
    to: tuple
    min: int = 0
    max: Optional[int] = None


@dataclass(frozen=True)
class Body:
    KIND: ClassVar[str] = ""
    LINKS: ClassVar[tuple] = ()
    ACTORS: ClassVar[tuple] = ("agent", "person", "openlab")

    def check(self, rec: "Record", actor: str, links: dict) -> None:
        """Invariants that need the record; `links` maps rel -> [Node]. Raise StudyError to refuse."""


_SHA = re.compile(r"[0-9a-f]{64}")


def _sha(value: str, what: str) -> None:
    if value and not _SHA.fullmatch(value):
        raise StudyError(f"{what} is not a sha256", f"got {value[:20]!r}")


AUDIT_VERDICTS = ("clean", "separated", "look-ahead", "unchecked", "in-sample")


def _audit(audit: dict) -> None:
    """The truncation audit's verdict follows from what it checked: look-ahead iff a replay mismatched, clean only
    after at least one replay; `separated` is a static run, whose folds were staged apart (each child saw only its
    training rows and the validation features), so nothing needs replaying; audit.n = 0 is unchecked."""
    verdict, checked, bad = audit.get("verdict"), audit.get("checked", 0), audit.get("mismatches", [])
    if verdict not in AUDIT_VERDICTS:
        raise StudyError("run.audit has no verdict", f"one of {', '.join(AUDIT_VERDICTS)}; got {verdict!r}")
    if verdict in ("separated", "in-sample") and (checked or bad):
        raise StudyError(f"run.audit says {verdict} but lists replays", f"{verdict} is for static runs")
    if (verdict == "look-ahead") != bool(bad) or (verdict == "clean") != (checked > 0 and not bad):
        raise StudyError(f"run.audit says {verdict} after {checked} checks and {len(bad)} mismatches")


def _scores(where: str, scores: dict) -> None:
    """{metric: {value, lo, hi}}; a value that could not be computed is None with the reason."""
    for name, m in scores.items():
        if not isinstance(m, dict) or "value" not in m:
            raise StudyError(f"{where} {name} has no value", "a metric is {value, lo, hi}")
        if m["value"] is None and not m.get("reason"):
            raise StudyError(f"{where} {name} has no value and no reason", "a metric that was not computed says why")


# --- phase 1 -------------------------------------------------------------------------------------------------
#: how a research run was made from its parent: a fresh draft, an improvement, a fix of a failed or buggy parent, a
#: fusion with runs of other branches, an ensemble of runs. "" is a run recorded before operators.
RUN_OPS = ("", "draft", "improve", "debug", "fuse", "ensemble")


def _run_op(op: str, segment: str, links: dict) -> None:
    if op not in RUN_OPS:
        raise StudyError(f"no run op `{op}`", f"one of {', '.join(o for o in RUN_OPS if o)}")
    parent, refs = links.get("branches_from", []), links.get("refs", [])
    if op and segment != "research":
        raise StudyError("only a research run has an op", "a holdout run is the look's code, not a step of the search")
    if op == "draft" and parent:
        raise StudyError("a draft has no parent", "it starts a new line of runs; improve, debug or fuse one instead")
    if op in ("improve", "debug", "fuse", "ensemble") and not parent:
        raise StudyError(f"{'an' if op[0] in 'aeiou' else 'a'} {op} run needs its parent",
                         "give the run it was made from (parent=run:...)")
    if refs and op not in ("fuse", "ensemble"):
        raise StudyError("only a fuse or an ensemble refs other runs", f"this run's op is {op or 'unset'}")
    for n in [*parent, *refs]:
        if n.payload.get("segment") != "research":
            raise StudyError(f"{n.id} is a holdout run", "the search moves over research runs only")
    if parent and parent[0].id in {n.id for n in refs}:
        raise StudyError("a run's parent is not also one of its refs")

@dataclass(frozen=True)
class Plan(Body):
    """The sealed study.toml (canonical raw dict) and its sha. `person_key`: the person key the study was sealed under
    ({fingerprint, public}: the study's own pin, else the workspace's), so the study's first pin must be that key or a
    rotation it endorsed, even when the workspace's pin stores are gone later (empty: no key was pinned anywhere)."""
    KIND: ClassVar[str] = "plan"
    spec: dict
    sha: str
    person_key: dict = field(default_factory=dict)

    def check(self, rec, actor, links):
        _sha(self.sha, "plan sha")
        if rec.find("plan"):
            raise StudyError("the study is already sealed", "a study has one plan; change it by amendment")


@dataclass(frozen=True)
class Amendment(Body):
    KIND: ClassVar[str] = "amendment"
    LINKS: ClassVar[tuple] = (Link("amends", ("plan", "amendment"), 1, 1),)
    reason: str
    changes: dict
    person_fields: list
    person: bool
    sha: str
    prev_sha: str

    def check(self, rec, actor, links):
        if not self.reason.strip():
            raise StudyError("an amendment needs a reason")
        if (self.person or self.person_fields) and actor != "person":
            raise StudyError(f"{', '.join(self.person_fields) or 'this amendment'} can only be amended by a person")
        _latest_spec_node(rec, links["amends"][0], "an amendment must amend the latest plan or amendment")


@dataclass(frozen=True)
class Dataset(Body):
    """A materialised dataset: where it came from, its bytes' sha256, and its card."""
    KIND: ClassVar[str] = "dataset"
    name: str
    source: dict
    sha256: str
    rows: int
    columns: list
    card: dict
    recipe_sha: str = ""

    def check(self, rec, actor, links):
        _sha(self.sha256, "dataset sha256")
        _sha(self.recipe_sha, "recipe sha")
        if self.source.get("kind") == "file" and not self.recipe_sha:
            raise StudyError(f"dataset {self.name} is a file with no recipe", "nothing says how it was made")


@dataclass(frozen=True)
class Run(Body):
    """One execution of model.py over a segment: one leaf of the forking-paths tree. `choices` names the analytic
    choices that identify the path (spec sections, code, declared model parameters); `branches_from` links the run
    it varies: the parent the agent chose with `op` (a draft has none; improve, debug and fuse have one; a fuse also
    `refs` runs of other branches it borrows from), else the branch's (or study's) last run. `box`: the machine that
    ran it (openlab.study.machine). A holdout run is only valid after a person's look that carries the same code,
    data and spec."""
    KIND: ClassVar[str] = "run"
    LINKS: ClassVar[tuple] = (Link("under", ("plan", "amendment"), 1, 1), Link("uses", ("dataset",), 1),
                              Link("after", ("look",), 0, 1), Link("branches_from", ("run",), 0, 1),
                              Link("refs", ("run",), 0, None))
    segment: Literal["research", "holdout"]
    key: str
    cached: bool
    spec_sha: str
    code: dict
    data: dict
    env: dict
    sandbox: str
    audit: dict
    seconds: float
    choices: dict = field(default_factory=dict)
    stdout: str = ""
    predictions: str = ""
    metric_versions: dict = field(default_factory=dict)
    note: str = ""
    skipped: int = 0                        # temporal steps not scored: no training rows yet
    branch: str = ""                        # the proposal branch whose working copy ran (openlab.study.branches)
    seed: Optional[int] = None              # the model's seed when the run gave one (else the spec's split.seed)
    op: str = ""                            # how it was made from its parent (RUN_OPS); "" before operators
    cites: list = field(default_factory=list)   # the lessons it applies: lesson:<id> here, or <study>/lesson:<id>
    box: dict = field(default_factory=dict) # {host, cores, memory_gb}: the machine it ran on

    def check(self, rec, actor, links):
        _sha(self.stdout, "stdout blob")
        _run_op(self.op, self.segment, links)
        _sha(self.predictions, "predictions blob")
        _audit(self.audit)
        under = _latest_spec_node(rec, links["under"][0], "a run must run under the current sealed spec")
        if under.payload["sha"] != self.spec_sha:
            raise StudyError("the run's spec_sha is not the sealed spec's sha")
        budget = _trial_budget(rec, under)
        if self.segment == "research" and budget is not None and not rec.find("run", key=self.key):
            if trials_used(rec) >= budget:            # checked inside the record's write lock: two agents at once
                                                      # cannot both take the last trial
                raise StudyError(f"the trial budget is spent: {trials_used(rec)} of {budget} trials",
                                 "a person must amend budget.trials before another run")
        if self.audit["verdict"] in ("separated", "in-sample"):
            from .spec import sealed
            spec = sealed(rec)
            if spec is None or spec.mode != "static":
                raise StudyError(f"only a static run is `{self.audit['verdict']}`", "a temporal run is cleared by the "
                                 "truncation audit's replays, not by declaration")
            if (self.audit["verdict"] == "in-sample") != (spec.split.get("scheme") == "insample"):
                raise StudyError("a run is `in-sample` exactly when the sealed split.scheme is insample")
        _uses("run", self.data, links)
        looks = links.get("after", [])
        if self.segment == "research":
            if looks:
                raise StudyError("a research run cannot link a holdout look")
            return
        if not looks:
            raise StudyError("a holdout run needs a look", "a person records the look before the holdout is touched")
        look = looks[0].payload
        if (look["code"], look["data"], look["spec_sha"]) != (self.code, self.data, self.spec_sha):
            raise StudyError("the holdout run differs from its look", "code, data or spec changed after the look; "
                             "that is a new look")
        if rec.links(looks[0].id, rel="after", incoming=True):
            raise StudyError("this look was already used by a holdout run", "one look, one run")


def _person_labels(rec) -> bool:
    """Whether the study's sealed split keeps its holdout labels with the person (split.labels = "person")."""
    from .spec import sealed
    try:
        spec = sealed(rec)
        return spec is not None and spec.split.get("labels") == "person"
    except StudyError:                              # a plan that does not replay: read it as sealed
        plan = rec.find("plan")
        return bool(plan) and plan[0].payload.get("spec", {}).get("split", {}).get("labels") == "person"


@dataclass(frozen=True)
class Result(Body):
    """A run's evaluation over n scored rows: metrics as {name: {value, lo, hi}}; baselines as {baseline:
    {name: {value, lo, hi}}}; paired as {baseline: {metric, diff, lo, hi, p, better}}, the model minus the
    baseline on the primary metric; policy as {decisions, periods_per_year, metrics, baselines} over act()."""
    KIND: ClassVar[str] = "result"
    LINKS: ClassVar[tuple] = (Link("of", ("run",), 1, 1),)
    metrics: dict
    baselines: dict = field(default_factory=dict)
    paired: dict = field(default_factory=dict)
    policy: dict = field(default_factory=dict)
    n: int = 0
    groups: dict = field(default_factory=dict)     # evaluate.by: {group: {metrics, baselines, n}}, each entity's own
    scored: dict = field(default_factory=dict)     # scored by the person, who keeps the labels: {by, labels_sha, rows,
                                                   # openlab} (openlab.study.custody)

    def check(self, rec, actor, links):
        _scores("metric", self.metrics)
        run = links["of"][0]
        kept = run.payload.get("segment") == "holdout" and _person_labels(rec)
        if kept and (actor != "person" or not self.scored.get("labels_sha")):
            raise StudyError("this holdout's labels are kept by the person: only the person's score is its result",
                             "the machine holding the record has no labels to score it with",
                             "the person runs `openlab study score HOST FOLDER --labels FILE` on their machine")
        if self.scored and not kept:
            raise StudyError("a person's score is only for a holdout whose labels the person keeps")
        for g, body in self.groups.items():
            _scores(f"group {g}:", body.get("metrics", {}))
            for b, scores in body.get("baselines", {}).items():
                _scores(f"group {g} baseline {b}:", scores)
        for b, scores in [*self.baselines.items(), *self.policy.get("baselines", {}).items()]:
            _scores(f"baseline {b}:", scores)
        _scores("policy metric", self.policy.get("metrics", {}))
        _scores("policy series", self.policy.get("series", {}))
        for b, p in self.paired.items():
            if p.get("diff") is None and not p.get("reason"):
                raise StudyError(f"the paired difference against {b} has no value and no reason")
        if rec.links(links["of"][0].id, rel="of", incoming=True):
            raise StudyError("this run already has a result", "a result is written once per run")


@dataclass(frozen=True)
class Look(Body):
    """A person's decision to score the holdout, written before anything is scored; n counts looks."""
    KIND: ClassVar[str] = "look"
    LINKS: ClassVar[tuple] = (Link("under", ("plan", "amendment"), 1, 1),)
    ACTORS: ClassVar[tuple] = ("person",)
    n: int
    reason: str
    spec_sha: str
    code: dict
    data: dict

    def check(self, rec, actor, links):
        if not self.reason.strip():
            raise StudyError("a look needs a reason")
        if self.n != len(rec.find("look")) + 1:
            raise StudyError(f"look number {self.n} is out of order", f"{len(rec.find('look'))} looks are on record")
        _latest_spec_node(rec, links["under"][0], "a look is taken under the current sealed spec")


@dataclass(frozen=True)
class Model(Body):
    """A run's final fit, registered in the study's MLflow; `keys` are the entity key columns label() returned, which
    a forward run keys its features on."""
    KIND: ClassVar[str] = "model"
    LINKS: ClassVar[tuple] = (Link("from", ("run",), 1, 1),)
    name: str
    version: int
    uri: str
    keys: list = field(default_factory=list)


@dataclass(frozen=True)
class Action(Body):
    """The flight recorder: one node per verb, refusals included."""
    KIND: ClassVar[str] = "action"
    LINKS: ClassVar[tuple] = (Link("did", ANY),)
    verb: str
    args: dict
    outcome: dict


@dataclass(frozen=True)
class Forecast(Body):
    """One forward decision: what a registered model predicted at t from the data known then, the gate's checks on
    those predictions ({passed, checks: [{expect, ok, ...}]}), and the signed file a program downstream acts on
    (`file`, its sha256 also naming its copy in the content store). The model must come from a research run whose
    truncation audit is clean, and model.py must be the code it was fitted with. `label_at` (t + horizon) is when
    its outcome can be known; `prospective` says the forecast was written before then. One per decision and model."""
    KIND: ClassVar[str] = "forecast"
    LINKS: ClassVar[tuple] = (Link("under", ("plan", "amendment"), 1, 1), Link("by", ("model",), 1, 1),
                              Link("uses", ("dataset",), 1))
    t: str
    label_at: str
    prospective: bool
    model: dict                             # {name, version, run, sha256 of its pickle}
    spec_sha: str
    code: dict
    data: dict
    rows: int
    file: str                               # relative to the study, as `signed` is
    sha256: str
    gate: dict
    signed: str
    expires: str

    def check(self, rec, actor, links):
        from .spec import sealed
        from .splits import horizon, last_end
        _sha(self.sha256, "forecast sha256")
        _sha(self.model.get("sha256", ""), "model sha256")
        under = _latest_spec_node(rec, links["under"][0], "a forecast is made under the current sealed spec")
        if under.payload["sha"] != self.spec_sha:
            raise StudyError("the forecast's spec_sha is not the sealed spec's sha")
        _uses("forecast", self.data, links)
        by = links["by"][0]
        run = forecastable(rec, by)
        want = {"name": by.payload["name"], "version": by.payload["version"], "run": run.id}
        if {k: self.model.get(k) for k in want} != want:
            raise StudyError("the forecast's model is not the model it links", f"{self.model} vs {want}")
        if self.code != run.payload["code"]:
            raise StudyError(f"model.py is not the code {by.payload['name']} v{by.payload['version']} was fitted with",
                             "its features would not be the ones the model learned from",
                             "restore that code, or run again and forecast with the new model")
        checks = self.gate.get("checks")
        if not isinstance(checks, list) or self.gate.get("passed") is not all(c.get("ok") is True for c in checks):
            raise StudyError("forecast.gate is not {passed, checks}", "passed is true exactly when every check is ok")
        t, spec = datetime.fromisoformat(self.t), sealed(rec)
        if t <= last_end(spec):
            raise StudyError(f"decision {self.t} is inside the study's segments", "a forecast there would score research "
                             "or holdout periods outside the trial and look budgets", "forward decisions come after "
                             f"{last_end(spec).isoformat()}")
        if self.label_at != (t + horizon(spec)).isoformat():
            raise StudyError("forecast.label_at is not t + horizon")
        if self.prospective != (datetime.fromisoformat(self.label_at) > utc()):
            raise StudyError("forecast.prospective must say whether its label is still unknown now")
        if forecast_of(rec, self.t, by.id):
            raise StudyError(f"{by.payload['name']} v{by.payload['version']} already forecast decision {self.t}",
                             "one forecast per decision and model, so none is quietly replaced",
                             "wait for the next decision time")


@dataclass(frozen=True)
class Outcome(Body):
    """A forecast's realised labels, once known: the spec's metrics on its n labelled rows (one decision, so no
    intervals), from the data it links; `labels` is the content-store blob of the rows scored. Written once per
    forecast, and only after its label_at has passed."""
    KIND: ClassVar[str] = "outcome"
    LINKS: ClassVar[tuple] = (Link("of", ("forecast",), 1, 1), Link("uses", ("dataset",), 1))
    forecast: str
    label_at: str
    metrics: dict
    n: int
    data: dict
    labels: str

    def check(self, rec, actor, links):
        f = links["of"][0]
        _sha(self.labels, "labels blob")
        _uses("outcome", self.data, links)
        _scores("outcome metric", self.metrics)
        if self.forecast != f.id or self.label_at != f.payload["label_at"]:
            raise StudyError("the outcome's forecast and label_at are not those of the forecast it links")
        if datetime.fromisoformat(self.label_at) > utc():
            raise StudyError(f"the label of {f.id} is not known until {self.label_at} (UTC)",
                             "an outcome is written only once its label can be known")
        if rec.links(f.id, rel="of", incoming=True):
            raise StudyError(f"{f.id} already has an outcome", "an outcome is written once per forecast")


SCHEMAS: dict = {c.KIND: c for c in (Plan, Amendment, Dataset, Run, Result, Look, Model, Action, Forecast, Outcome)}
KINDS = tuple(SCHEMAS)


def _trial_budget(rec: "Record", under: "Node"):
    """budget.trials of the spec the run is under (None when the spec declares none; a sealed study always does)."""
    if under.kind == "plan":
        return (under.payload["spec"].get("budget") or {}).get("trials")
    from .spec import sealed
    return sealed(rec).budget.get("trials")


def _latest_spec_node(rec: "Record", node: "Node", why: str) -> "Node":
    latest = rec.last("amendment") or rec.last("plan")
    if latest is None or latest.id != node.id:
        raise StudyError(why, f"{node.id} is not the latest plan/amendment")
    return node


def _uses(kind: str, data: dict, links: dict) -> None:
    """The node's data hashes are those of the datasets it links."""
    used = {n.payload["name"]: n.payload["sha256"] for n in links.get("uses", [])}
    if used != data:
        raise StudyError(f"the {kind}'s data hashes do not match the datasets it links", f"{used} vs {data}")


def _in_sample(rec: "Record", results) -> bool:
    """Whether any of the results was scored on the rows that trained it (split.scheme = "insample")."""
    return any(run_of(rec, r).payload["audit"]["verdict"] == "in-sample"
               for r in (results if isinstance(results, (list, tuple)) else [results]))


def _audited(run: "Node", name: str, what: str) -> None:
    """Refuse a run whose truncation audit is not clean; `what` names what would rest on it."""
    verdict = run.payload["audit"]["verdict"]
    if verdict not in ("clean", "separated", "in-sample"):          # in-sample: said so by the verdict it backs
        raise StudyError(f"{name} comes from a run whose truncation audit is {verdict}",
                         f"its features may read data from after t, so {what} cannot be trusted"
                         if verdict == "look-ahead" else "nothing checked that its features stay before t",
                         "fix model.py and run again" if verdict == "look-ahead" else "run again with audit.n > 0")


def forecastable(rec: "Record", model: "Node") -> "Node":
    """The research run `model` was fitted by, when its truncation audit is clean; else StudyError."""
    run = rec.get(rec.links(model.id, rel="from")[0][2])
    name = f"{model.payload['name']} v{model.payload['version']}"
    if run.payload["segment"] != "research":
        raise StudyError(f"{name} comes from a {run.payload['segment']} run", "forecasts use a research run's model")
    _audited(run, name, "its forecasts")
    return run


def forecast_of(rec: "Record", t: str, model_id: str) -> "Node | None":
    """The forecast on record for decision t by this model."""
    return next((f for f in rec.find("forecast", t=t) if rec.links(f.id, rel="by")[0][2] == model_id), None)


# --- phase 2: literature ---------------------------------------------------------------------------------------
INDEXES = ("openalex", "arxiv", "crossref")
#: the source tiers, most trusted first; model-generated never counts as evidence
TIERS = ("primary", "publisher", "doi_metadata", "scholarly_index", "secondary", "web", "model")
TIER_OF = {"openalex": "scholarly_index", "citations": "scholarly_index", "crossref": "doi_metadata",
           "arxiv": "publisher"}                    # a record's tier, by the search that returned it
IDS = ("doi", "arxiv", "openalex")
_ID_FORM = {"doi": re.compile(r"10\.\d{4,9}/\S+"), "openalex": re.compile(r"W\d+"),
            "arxiv": re.compile(r"\d{4}\.\d{4,5}|[a-z][a-z-]*(\.[A-Z]{2})?/\d{7}")}
DIRECTIONS = ("references", "cited_by")
RETRACTION_CHECKS = ("openalex", "crossref")
MIN_QUOTE = 20                                      # characters: a shorter quote is found almost anywhere
FIELD_TYPES = ("number", "int", "text", "list")    # of an extracted value ([review.extract])
FREE_TYPES = ("text", "list")                       # may be declared free; a number is always checked
UNITS = ("percent", "fraction")                     # the unit a paper writes a number in (evidence, target)


def norm_id(kind: str, value: str) -> str:
    """An id in its one stored form: a DOI lower-case with no resolver prefix, an arXiv id with no version, an
    OpenAlex work id with no URL."""
    v = (value or "").strip()
    if kind == "doi":
        return re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", v, flags=re.I).lower()
    if kind == "arxiv":
        return re.sub(r"v\d+$", "", re.sub(r"^(https?://arxiv\.org/(abs|pdf)/|arxiv:)", "", v, flags=re.I))
    return v.rsplit("/", 1)[-1].upper()


def valid_id(kind: str, value: str) -> bool:
    return bool(value) and value == norm_id(kind, value) and bool(_ID_FORM[kind].fullmatch(value))


def identity(d: dict) -> dict:
    """{kind: id} for the ids `d` carries; refused unless there is one and each is in its stored form, so the
    same work cannot hide under two spellings."""
    out = {k: d[k] for k in IDS if d.get(k)}
    bad = [f"{k} {v!r}" for k, v in out.items() if not valid_id(k, v)]
    if bad or not out:
        raise StudyError(f"not an id in its stored form: {', '.join(bad)}" if bad else "a source needs an id",
                         "a DOI lower-case with no https://doi.org/, an arXiv id with no version, an OpenAlex W id")
    return out


def source_key(d: dict) -> str:
    """One id as `kind:id`, the DOI first."""
    k = next(k for k in IDS if d.get(k))
    return f"{k}:{d[k]}"


def sources_by_id(rec: "Record") -> dict:
    """{(kind, id): source node} over the record's sources."""
    out: dict = {}
    for n in rec.find("source"):
        for k in IDS:
            if n.payload.get(k):
                out.setdefault((k, n.payload[k]), n)
    return out


def resolve(index: dict, ids: dict):
    """The source on record sharing any of `ids`, or None."""
    return next((index[(k, v)] for k, v in ids.items() if (k, v) in index), None)


def _review(rec: "Record") -> dict:
    from . import spec                                 # spec imports the record, which imports this module
    s = spec.sealed(rec)
    if s is None or not s.review:
        raise StudyError("the sealed spec has no [review]", "the literature engine works to a sealed protocol",
                         "add a [review] section to study.toml (by amendment once sealed)")
    return s.review


@dataclass(frozen=True)
class Query(Body):
    """One search as asked and as answered: a protocol query on one index, or one citation hop from a `seed`
    source (source `citations`, text the seed's id, filters its direction and hop). A contradiction search's
    query links the claim it is `for`, and its text is the agent's, not the protocol's. `hits` are the identities
    returned, in rank order; a call that failed keeps its error and has no hits."""
    KIND: ClassVar[str] = "query"
    LINKS: ClassVar[tuple] = (Link("under", ("plan", "amendment"), 1, 1), Link("seed", ("source",), 0, 1),
                              Link("for", ("claim",), 0, 1))
    ACTORS: ClassVar[tuple] = ("agent", "openlab")
    source: Literal["openalex", "arxiv", "crossref", "citations"]
    text: str
    filters: dict
    n_hits: int
    hits: list
    page: dict
    error: str = ""

    def check(self, rec, actor, links):
        _latest_spec_node(rec, links["under"][0], "a query runs under the current sealed review")
        review = _review(rec)
        if self.n_hits != len(self.hits) or (self.error and self.hits):
            raise StudyError(f"the query says {self.n_hits} hits and lists {len(self.hits)}",
                             "a failed query has none")
        for h in self.hits:
            if not isinstance(h, dict) or set(h) - set(IDS):
                raise StudyError("a hit is an identity", f"{{doi, arxiv, openalex}}; got {h!r}")
            identity(h)
        seed = links.get("seed", [])
        if self.source != "citations":
            if seed:
                raise StudyError("only a citation hop has a seed")
            own = bool(links.get("for")) and bool(self.text.strip())     # a contradiction search's own query
            if self.source not in review["sources"] or not (own or self.text in review["queries"]):
                raise StudyError(f"{self.source} `{self.text}` is not in the sealed review",
                                 f"sources: {', '.join(review['sources'])}; queries: {'; '.join(review['queries'])}",
                                 "add it to [review] by amendment, with a reason")
            want = {"since": review["since"], "n": review["n"]}
            if self.filters != want:
                raise StudyError("the query's filters are not the review's", f"{self.filters} vs {want}")
            return
        f = self.filters
        if not seed or self.text != source_key(seed[0].payload):
            raise StudyError("a citation hop names its seed source", "link `seed` and give its id as the text")
        if set(f) != {"direction", "hop", "n", "since"} or f["direction"] not in DIRECTIONS or f["n"] != review["n"] \
                or f["since"] != review["since"] or type(f["hop"]) is not int \
                or not 1 <= f["hop"] <= review["citation_hops"]:
            raise StudyError("a citation hop's filters are its direction, hop, n and since, within the review",
                             f"direction {' or '.join(DIRECTIONS)}, hop 1..{review['citation_hops']}, "
                             f"n {review['n']}, since {review['since']!r}; got {f}")


@dataclass(frozen=True)
class Source(Body):
    """One version of a work, as the index that found it described it. Its ids (DOI, arXiv, OpenAlex; stored
    form) are unique in the record: a source sharing any of them with one on record is a duplicate and is
    refused. `published` is the valid time (when it appeared); the node's `at` is when the study learned of it.
    A preprint may name the DOI it was `published_as`; `version_of` links a version of the other kind.
    `retraction_checked` names the checks behind `retracted`."""
    KIND: ClassVar[str] = "source"
    LINKS: ClassVar[tuple] = (Link("found_by", ("query",), 1, 1), Link("version_of", ("source",), 0, 1))
    ACTORS: ClassVar[tuple] = ("openlab",)
    title: str
    version_kind: Literal["preprint", "published"]
    tier: Literal[TIERS]
    retracted: bool
    retraction_checked: list
    doi: str = ""
    arxiv: str = ""
    openalex: str = ""
    published_as: str = ""
    authors: list = field(default_factory=list)
    year: Optional[int] = None
    venue: str = ""
    published: str = ""
    retraction_note: str = ""
    abstract: str = ""
    url: str = ""
    pdf: str = ""

    def check(self, rec, actor, links):
        owned = identity({k: getattr(self, k) for k in IDS})
        if self.published_as and (self.version_kind != "preprint" or not valid_id("doi", self.published_as)):
            raise StudyError("published_as is the stored-form DOI a preprint names for its published version")
        if self.published and (not re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.published)
                               or (self.year is not None and int(self.published[:4]) != self.year)):
            raise StudyError(f"published {self.published!r} is not an ISO date in year {self.year}")
        if self.retracted and not self.retraction_note.strip():
            raise StudyError("a retracted source says where the flag came from (retraction_note)")
        if not set(self.retraction_checked) <= set(RETRACTION_CHECKS):
            raise StudyError("retraction_checked names checks openlab runs", ", ".join(RETRACTION_CHECKS))
        q = links["found_by"][0].payload
        if not any(set(owned.items()) & set(h.items()) for h in q["hits"]):
            raise StudyError("the query it names did not return this source")
        if TIER_OF.get(q["source"], self.tier) != self.tier:
            raise StudyError(f"a record found on {q['source']} is tier {TIER_OF[q['source']]}")
        for k, v in owned.items():
            same = rec.find("source", **{k: v})                # indexed: a lookup, not a scan of every source
            if same:
                raise StudyError(f"this source is already on record as {same[0].id}", f"same {k} {v}",
                                 "a work is recorded once; finding it again is a duplicate")
        for v in links.get("version_of", []):
            if v.payload["version_kind"] == self.version_kind:
                raise StudyError("version_of links a preprint and a published version", f"both are {self.version_kind}")


@dataclass(frozen=True)
class Document(Body):
    """A source's full text by page: `pages` is the blob of a JSON list of page texts in reading order, and
    `layout` the same pages as laid out (table rows intact), both extracted by `method` from the PDF blob `pdf`
    (fetched from `url`)."""
    KIND: ClassVar[str] = "document"
    LINKS: ClassVar[tuple] = (Link("of", ("source",), 1, 1),)
    ACTORS: ClassVar[tuple] = ("openlab",)
    pages: str
    n_pages: int
    method: Literal["pdftotext"]
    pdf: str
    url: str
    layout: str = ""

    def check(self, rec, actor, links):
        import json
        if not (_SHA.fullmatch(self.pages) and _SHA.fullmatch(self.pdf)) or (self.layout and not _SHA.fullmatch(self.layout)):
            raise StudyError("a document's pages, layout and pdf are blob shas")
        rec.blob(self.pdf)
        for name, sha in (("pages", self.pages), ("layout", self.layout)):
            try:
                pages = json.loads(rec.blob(sha)) if sha else []
            except ValueError:
                pages = None
            if sha and not (isinstance(pages, list) and all(isinstance(p, str) for p in pages)
                            and len(pages) == self.n_pages > 0):
                raise StudyError(f"the {name} blob is not a list of {self.n_pages} page texts")


def coverage(rec: "Record", queries: list) -> dict:
    """The coverage of these query nodes, recomputed from them and the record's sources: distinct protocol
    queries answered per index, hops and citation edges followed, failed calls, records returned, the unique
    sources they resolve to (each must be on record) and the verdict against the sealed floor."""
    review, index = _review(rec), sources_by_id(rec)
    texts: dict = {s: set() for s in review["sources"]}
    hops = edges = records = failed = 0
    reached = set()
    for q in queries:
        p = q.payload
        if p["error"]:
            failed += 1
            continue
        records += p["n_hits"]
        if p["source"] == "citations":
            edges += p["n_hits"]
            hops = max(hops, p["filters"]["hop"])
        else:
            texts.setdefault(p["source"], set()).add(p["text"])
        for h in p["hits"]:
            node = resolve(index, h)
            if node is None:
                raise StudyError(f"{q.id} returned {source_key(h)}, which has no source on record",
                                 "coverage counts recorded sources only")
            reached.add(node.id)
    floor = review["coverage_floor"]
    per = {s: len(t) for s, t in texts.items()}
    ok = all(per[s] >= floor["queries_per_source"] for s in review["sources"]) and hops >= floor["hops"]
    return {"queries": per, "hops": hops, "edges": edges, "records": records, "unique": len(reached),
            "failed": failed, "floor": floor, "verdict": "sufficient" if ok else "insufficient"}


@dataclass(frozen=True)
class Coverage(Body):
    """How much one search searched (`coverage`), with the verdict against the sealed review's floor: a discovery
    (protocol queries) or one claim's contradiction search (queries `for` it), never a mix. Every number is
    recomputed from the linked queries and the record on write, so coverage cannot be typed."""
    KIND: ClassVar[str] = "coverage"
    LINKS: ClassVar[tuple] = (Link("under", ("plan", "amendment"), 1, 1), Link("of", ("query",), 1))
    queries: dict
    hops: int
    edges: int
    records: int
    unique: int
    failed: int
    floor: dict
    verdict: Literal["sufficient", "insufficient"]

    def check(self, rec, actor, links):
        _latest_spec_node(rec, links["under"][0], "coverage is judged under the current sealed review")
        if len({tuple(d for _, _, d in rec.links(q.id, rel="for")) for q in links["of"]}) > 1:
            raise StudyError("a coverage judges one search",
                             "a discovery's queries, or one claim's contradiction search")
        want = coverage(rec, links["of"])
        bad = [f"{k}: typed {getattr(self, k)}, recomputed {v}" for k, v in want.items() if getattr(self, k) != v]
        if bad:
            raise StudyError("the coverage does not match its queries", "; ".join(bad))


def _flat(text: str) -> str:
    """Text as a quote is matched: NFKC (ligatures), no soft hyphens, the unicode minus as '-', a word broken
    across lines ("post-\\nprocessing") joined, a hyphen between letters dropped (so "post-processing" and
    "postprocessing" match either way), whitespace collapsed."""
    import unicodedata
    t = unicodedata.normalize("NFKC", text).replace("­", "").replace("−", "-")
    t = re.sub(r"(?<=[^\W\d_])-\s*\n\s*(?=[^\W\d_])", "", t)
    t = re.sub(r"(?<=[^\W\d_])-(?=[^\W\d_])", "", t)
    return " ".join(t.split())


def _quote(quote: str) -> str:
    """The quote in matching form, refused when too short to mean anything."""
    q = _flat(quote)
    if len(q) < MIN_QUOTE:
        raise StudyError(f"the quote is too short to check ({len(q)} characters)",
                         f"at least {MIN_QUOTE}: quote the passage the decision rests on")
    return q


def _quoted(rec: "Record", src: "Node", quote: str) -> None:
    """Refuse unless the quote, whitespace aside, is in the source's abstract or its full-text pages."""
    import json
    q = _quote(quote)
    if q in _flat(src.payload["abstract"]):
        return
    for d, _, _ in rec.links(src.id, rel="of", incoming=True):
        doc = rec.get(d)
        if doc.kind == "document" and q in _flat(" ".join(json.loads(rec.blob(doc.payload["pages"])))):
            return
    raise StudyError("the quote is not in the source", "openlab looks for it, whitespace aside, in the abstract "
                     "and the full-text pages", "quote the abstract, or fetch the full text first (study_fetch_text)")


@dataclass(frozen=True)
class Screen(Body):
    """A screening decision on one source under the sealed review: include or exclude citing one of the
    review's criteria of that kind, with a rationale and a quote openlab finds in the source (an agent must
    quote; a person may). `maybe`, the agent's only, hands the source to a person, whose decision then stands:
    an agent can neither resolve a maybe nor overrule a person."""
    KIND: ClassVar[str] = "screen"
    LINKS: ClassVar[tuple] = (Link("screens", ("source",), 1, 1), Link("under", ("plan", "amendment"), 1, 1))
    ACTORS: ClassVar[tuple] = ("agent", "person")
    decision: Literal["include", "exclude", "maybe"]
    criterion: str
    rationale: str
    quote: str = ""

    def check(self, rec, actor, links):
        _latest_spec_node(rec, links["under"][0], "a screening decision is made under the current sealed review")
        review = _review(rec)
        ids = {d: [c["id"] for c in review[d]] for d in ("include", "exclude")}
        allowed = ids["include"] + ids["exclude"] if self.decision == "maybe" else ids[self.decision]
        if self.criterion not in allowed:
            what = "a criterion" if self.decision == "maybe" else f"an {self.decision} criterion"
            raise StudyError(f"`{self.criterion}` is not {what} of the sealed review", f"those are {', '.join(allowed)}")
        if not self.rationale.strip():
            raise StudyError("a screening decision needs a rationale")
        if self.decision == "maybe" and actor != "agent":
            raise StudyError("only the agent marks a source maybe", "a person decides include or exclude")
        src = links["screens"][0]
        before = [rec.get(s) for s, _, _ in rec.links(src.id, rel="screens", incoming=True)]
        if actor == "agent" and any(n.actor == "person" for n in before):
            raise StudyError("a person has decided this source", "an agent cannot overrule a person's decision")
        if actor == "agent" and before and before[-1].payload["decision"] == "maybe":
            raise StudyError("this source waits for a person", "the agent marked it maybe; a person resolves it")
        clash = screen_standing(before)[1]
        if actor == "agent" and clash:
            raise StudyError("the screens of this source conflict; a person resolves it",
                             "; ".join(f"{n.session} {n.payload['decision']} ({n.payload['criterion']})" for n in clash),
                             "screen other sources; the conflict waits in `openlab inbox` (screen_conflict)")
        if actor == "agent" or self.quote.strip():
            _quoted(rec, src, self.quote)


def screen_standing(screens: list) -> tuple:
    """(the screen that stands or None, the conflicting screens) over one source's screens, oldest first. A person's
    latest decision stands over any agent's. Among agents, each agent's latest decision counts (an agent may change
    its mind): a maybe hands the source to a person and stands; agents that disagree (include against exclude) are
    a conflict, and no decision stands until a person decides. Otherwise the latest stands."""
    people = [s for s in screens if s.actor == "person"]
    if people:
        return people[-1], []
    if not screens:
        return None, []
    if screens[-1].payload["decision"] == "maybe":
        return screens[-1], []
    latest: dict = {}
    for s in screens:
        latest[s.session] = s
    if len({s.payload["decision"] for s in latest.values()} - {"maybe"}) > 1:
        return None, sorted(latest.values(), key=lambda s: s.seq)
    return screens[-1], []


def extraction(review: dict) -> dict:
    """The sealed extraction schema ([review.extract]) as {field: (type, free)}."""
    return {k: (v, False) if isinstance(v, str) else (v["type"], v.get("free", False))
            for k, v in review["extract"].items()}


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


_TYPES = {"number": (_number, "a number"),
          "int": (lambda v: isinstance(v, int) and not isinstance(v, bool), "an integer"),
          "text": (lambda v: isinstance(v, str) and v.strip() != "", "a non-empty string"),
          "list": (lambda v: isinstance(v, list) and len(v) > 0 and all(_number(x) or _TYPES["text"][0](x) for x in v),
                   "a non-empty list of numbers and strings")}
#: a number as written: thousands separators allowed, and a minus only where it is a sign (not "0.5-0.6")
_NUMERAL = re.compile(r"(?<![\w.])(-?)((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)")


def _numbers(text: str) -> set:
    """The numbers written in `text`, by value: "0.90" is 0.9, "1,234" is 1234, "−2" is -2."""
    return {Decimal(sign + digits.replace(",", "")) for sign, digits in _NUMERAL.findall(_flat(text))}


def _percents(text: str) -> set:
    """The numbers `text` writes as percentages ("92.62%", "92.62 %"), by value."""
    flat = _flat(text)
    return {Decimal(m[1] + m[2].replace(",", "")) for m in _NUMERAL.finditer(flat)
            if flat[m.end():].lstrip().startswith("%")}


def extraction_units(review: dict) -> dict:
    """{field: unit} for the number fields [review.extract] declares with a unit (percent or fraction)."""
    return {k: v["unit"] for k, v in review["extract"].items() if isinstance(v, dict) and v.get("unit")}


def evidence_units(quote: str, fields: dict, declared: dict, schema_units: dict) -> dict:
    """The unit of each number field: as declared on the extract, else as [review.extract] declares it, else
    percent when the quote writes that number with a % sign. A field with none is in the metric's own unit."""
    out = {}
    for k, v in fields.items():
        unit = declared.get(k) or schema_units.get(k) or (
            "percent" if _number(v) and Decimal(repr(v)) in _percents(quote) else "")
        if unit:
            out[k] = unit
    return {**out, **{k: u for k, u in declared.items() if k not in out}}


def _number_spans(line: str, value) -> list:
    want = Decimal(repr(value))
    out = []
    for m in re.finditer(r"[-+\u2212]?\d[\d,]*(?:\.\d+)?", line):
        try:
            if Decimal(m[0].replace(",", "").replace("\u2212", "-")) == want:
                out.append(m.span())
        except ArithmeticError:
            continue
    return out


def _units(line: str) -> list:
    """The labels of one laid-out header line: runs of text split where two or more spaces fall, with their spans."""
    return [(m[0], m.span()) for m in re.finditer(r"\S+(?: \S+)*", line)]


def label_parts(label: str) -> list:
    """A column's label by its header levels: "SC2 / γ=1−α" is the group SC2, the column γ=1−α under it."""
    return [p.strip() for p in label.split(" / ") if p.strip()]


def in_header(value: str, header: str) -> bool:
    """Whether every level of a column label is written in the header (its lines), case aside."""
    flat = _flat(header).casefold()
    return all(_flat(p).casefold() in flat for p in label_parts(value))


def _centre(span) -> float:
    return (span[0] + span[1]) / 2


def _align(units: list, cells: list) -> dict:
    """Header labels to row cells, in order (a cell may go unlabelled, a label unmatched at a cost), minimising the
    total distance between centres: {label index: cell index}."""
    n, m, skip = len(units), len(cells), 1000.0
    inf = float("inf")
    cost = [[inf] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    for b in range(m + 1):
        cost[0][b] = 0.0
    for a in range(1, n + 1):
        cost[a][0], back[a][0] = cost[a - 1][0] + skip, "u"
        for b in range(1, m + 1):
            opts = [(cost[a - 1][b - 1] + abs(_centre(units[a - 1]) - _centre(cells[b - 1])), "m"),
                    (cost[a][b - 1], "c"), (cost[a - 1][b] + skip, "u")]
            cost[a][b], back[a][b] = min(opts, key=lambda o: o[0])
    out, a, b = {}, n, m
    while a > 0:
        step = back[a][b]
        if step == "m":
            out[a - 1] = b - 1
            a, b = a - 1, b - 1
        elif step == "c":
            b -= 1
        else:
            a -= 1
    return out


def _groups(groups: list, leaves: list) -> dict:
    """Which group label each leaf label falls under: the leaves cut into contiguous runs, one per group (a group may
    have none), so each run's span is centred under its group: {leaf index: group index}."""
    n, g = len(leaves), len(groups)
    inf = float("inf")
    best = [[inf] * (g + 1) for _ in range(n + 1)]
    back = [[None] * (g + 1) for _ in range(n + 1)]
    for k in range(g + 1):
        best[0][k] = 0.0
    for a in range(1, n + 1):
        for k in range(1, g + 1):
            for start in range(a):                  # leaves start..a-1 under group k-1
                span = (leaves[start][0], leaves[a - 1][1])
                c = best[start][k - 1] + abs(_centre(span) - _centre(groups[k - 1]))
                if c < best[a][k]:
                    best[a][k], back[a][k] = c, ("run", start)
            if best[a][k - 1] < best[a][k]:          # group k-1 has no leaf
                best[a][k], back[a][k] = best[a][k - 1], ("empty", None)
    out, a, k = {}, n, g
    while a > 0 and k > 0:
        kind, start = back[a][k]
        if kind == "run":
            for x in range(start, a):
                out[x] = k - 1
            a = start
        k -= 1
    return out


def _same_label(unit: str, part: str) -> bool:
    return _flat(unit).casefold() == _flat(part).casefold()


def column_check(layout: str, row: str, header: str, label: str, value) -> str:
    """Why `value` is not shown under the column `label` ("" when it is). On the laid-out page, `header` is one or
    more lines above the line `row` is on (one per header level, "\n" between them), and `label` names the column by
    level (" / " between them, e.g. "SC2 / γ=1−α": the group, then the column under it). The lowest level's labels
    are aligned to the row's cells in order (nearest centres, a cell may be unlabelled), so a header set a few
    characters off its columns still names them; a group label covers the run of columns centred under it. How a
    table whose methods are its columns names a number's method."""
    lines = layout.splitlines()
    fr = _flat(row)
    rows = [i for i, x in enumerate(lines) if fr and fr in _flat(x)]
    if not rows:
        return "the row is not one laid-out line of the page: quote the row as one line of the page's layout"
    heads = []
    for h in [x for x in header.split("\n") if x.strip()]:
        at = [i for i, x in enumerate(lines) if _flat(h) and _flat(h) in _flat(x)]
        if not at:
            return f"the header line {h.strip()!r} is not one laid-out line of the page"
        heads.append(at)
    parts = label_parts(label)
    if not parts:
        return "the method has no label"
    for r in rows:
        above = sorted({i for at in heads for i in at if i < r})
        cells = [m.span() for m in re.finditer(r"\S+", lines[r])]
        nums = _number_spans(lines[r], value)
        owned = {k for k, (a, b) in enumerate(cells) if any(a <= x and y <= b for x, y in nums)}
        leaf = next((i for i in reversed(above) if any(_same_label(t, parts[-1]) for t, _ in _units(lines[i]))), None)
        if leaf is None or not owned:
            continue
        units = _units(lines[leaf])
        to_cell = _align([sp for _, sp in units], cells)
        hits = [u for u, (t, _) in enumerate(units) if _same_label(t, parts[-1]) and to_cell.get(u) in owned]
        for part in reversed(parts[:-1]):             # each group level above, nearest first
            group = next((i for i in reversed(above) if i < leaf and any(_same_label(t, part) for t, _ in _units(lines[i]))),
                         None)
            if group is None:
                hits = []
                break
            gunits = _units(lines[group])
            under = _groups([sp for _, sp in gunits], [sp for _, sp in units])
            hits = [u for u in hits if under.get(u) is not None and _same_label(gunits[under[u]][0], part)]
        if hits:
            return ""
    if not any(i < r for at in heads for i in at for r in rows):
        return "the header is not above the row on the page"
    missing = [p for p in parts if not any(_same_label(t, p) for at in heads for i in at for t, _ in _units(lines[i]))]
    if missing:
        return f"the header does not show `{missing[0]}` as a column label"
    return f"{value:g} is not under the column `{label}` (the header's labels, aligned in order to the row's cells, "\
        "put another label over it)"


def _in_quote(value, quote: str) -> bool:
    """A string is in the quote (case and _flat aside); a number is written in it."""
    if isinstance(value, str):
        return _flat(value).casefold() in _flat(quote).casefold()
    return Decimal(repr(value)) in _numbers(quote)


@dataclass(frozen=True)
class Evidence(Body):
    """What a source's full text says, quoted from one page: a passage, a table row or a figure caption, with the
    values read from it as fields of the sealed extraction schema ([review.extract]). openlab checks the quote is
    on that page of the document, in reading order or as laid out (whitespace, ligatures and line-break hyphens
    aside), that each field is declared and has its declared type, that each number is written in the quote (0.90
    is 0.9, 1,234 is 1234, − is -), and that each text value is in the quote, case aside, unless the schema
    declares the field free. A table whose columns name what its numbers are (the methods across, a dataset per row)
    gives `header`, its header line on the same page: a text value may then be in the header instead, when each
    number of the evidence sits under that value's column in the page's layout. Bitemporal: `valid_from` is the
    source's publication date (valid time), the node's `at` when the study read it."""
    KIND: ClassVar[str] = "evidence"
    LINKS: ClassVar[tuple] = (Link("from", ("document",), 1, 1), Link("of", ("source",), 1, 1),
                              Link("under", ("plan", "amendment"), 1, 1))
    ACTORS: ClassVar[tuple] = ("agent", "person")
    source: str
    page: int
    section: str
    kind: Literal["passage", "table", "figure"]
    quote: str
    fields: dict
    valid_from: str = ""
    units: dict = field(default_factory=dict)      # {number field: percent or fraction}, as the paper writes it
    header: str = ""                               # a table's header line, when a text value is a column's label

    def _check_units(self, schema_units: dict) -> None:
        """Each unit names a number field and is percent or fraction, and agrees with [review.extract] and with the
        quote: a number written with % is percent."""
        for name, unit in self.units.items():
            v = self.fields.get(name)
            if not _number(v) or unit not in UNITS:
                raise StudyError(f"unit `{name}={unit}` does not fit", f"a unit names a number field of the evidence "
                                 f"and is one of {', '.join(UNITS)}")
            if schema_units.get(name, unit) != unit:
                raise StudyError(f"`{name}` is in {schema_units[name]} by [review.extract], not {unit}")
        written = _percents(self.quote)
        bad = [k for k, v in self.fields.items() if _number(v) and Decimal(repr(v)) in written
               and self.units.get(k) != "percent"]
        if bad:
            raise StudyError(f"the quote writes {bad[0]} as a percentage ({self.fields[bad[0]]:g}%)",
                             "a value's unit is the one its quote shows", f"declare {bad[0]}=percent")

    def check(self, rec, actor, links):
        import json
        _latest_spec_node(rec, links["under"][0], "evidence is extracted under the current sealed review")
        schema = extraction(_review(rec))
        src, doc = links["of"][0], links["from"][0]
        if self.source != src.id or rec.links(doc.id, rel="of")[0][2] != src.id:
            raise StudyError("evidence names one source", "its `source`, its `of` link and its document's source")
        if self.valid_from != src.payload["published"]:
            raise StudyError("evidence.valid_from is its source's publication date",
                             f"{src.payload['published']!r}, got {self.valid_from!r}")
        if not self.section.strip():
            raise StudyError("evidence names its section", 'e.g. "Table 2" or "4.1 Results"')
        forms = [json.loads(rec.blob(doc.payload[k])) for k in ("pages", "layout") if doc.payload.get(k)]
        n = doc.payload["n_pages"]
        if not 1 <= self.page <= n:
            raise StudyError(f"the document has no page {self.page}", f"its pages are 1..{n}")
        q = _quote(self.quote)
        on = [str(i + 1) for i in range(n) if any(q in _flat(f[i]) for f in forms)]
        if str(self.page) not in on:
            raise StudyError(f"the quote is not on page {self.page}",
                             f"it is on page {', '.join(on)}" if on else "openlab looks for it, whitespace, ligatures "
                             "and line-break hyphens aside, in the page's text and its layout (table rows intact)",
                             "quote the page as study_fetch_text gives it (layout=true for a table row)")
        if self.header:
            _quote(self.header)
            for line in [x for x in self.header.split("\n") if x.strip()]:
                if not any(_flat(line) in _flat(f[self.page - 1]) for f in forms):
                    raise StudyError(f"the header line {line.strip()!r} is not on page {self.page}", "the header "
                                     "lines name the columns of the table the quote is a row of",
                                     "quote each as the page's layout has it")
        numbers = [v for k, v in self.fields.items() if k in schema and schema[k][0] in ("number", "int")
                   and _number(v)]
        for name, value in self.fields.items():
            if name not in schema:
                raise StudyError(f"`{name}` is not a field of the sealed extraction schema",
                                 f"fields: {', '.join(schema) or 'none (add [review.extract])'}")
            typ, free = schema[name]
            ok, what = _TYPES[typ]
            if not ok(value):
                raise StudyError(f"field `{name}` must be {what}", f"got {value!r}")
            missing = [] if free else [x for x in (value if typ == "list" else [value]) if not _in_quote(x, self.quote)]
            if missing and self.header and typ == "text" and in_header(value, self.header):
                layout = json.loads(rec.blob(doc.payload["layout"])) if doc.payload.get("layout") else None
                if layout is None:
                    raise StudyError("a value read from a table's header needs the page's layout",
                                     "this document was fetched without it", "quote a row that names it")
                why = next((w for w in (column_check(layout[self.page - 1], self.quote, self.header, value, v)
                                        for v in numbers) if w), "")
                if why:
                    raise StudyError(f"the value of `{name}` is in the header, but {why}", "a number read from a "
                                     "table whose columns are its methods belongs to the column it sits under",
                                     "quote the header line and the row as the page's layout has them")
                missing = [] if numbers else missing
            if missing:
                raise StudyError(f"the value of `{name}` is not in the quote", f"{missing[0]!r} is not "
                                 + ("written in it" if _number(missing[0]) else "in it, case aside"),
                                 "quote the text that states it" + ("" if typ in ("number", "int") else
                                                                    ", or declare the field free in [review.extract]"))
        self._check_units(extraction_units(_review(rec)))


def coord_key(v):
    """A coordinate value as compared: text by its matching form, case aside; a list item by item."""
    if isinstance(v, str):
        return _flat(v).casefold()
    return tuple(coord_key(x) for x in v) if isinstance(v, list) else v


def _side(node: "Node", metric: str, coordinates: list) -> tuple:
    """(coordinates, value of `metric`) of one side of a comparison. A study's result carries no coordinates yet
    (its runs do not record a protocol in these terms), so it is comparable with nothing until they do."""
    if node.kind == "evidence":
        f = node.payload["fields"]
        return {c: f[c] for c in coordinates if c in f}, f.get(metric)
    if node.kind == "result":
        return {}, (node.payload["metrics"].get(metric) or {}).get("value")
    raise StudyError(f"{node.id} is a {node.kind}", "a comparison compares evidence or a study's result")


def _verdict(metric: str, va, vb, level=None) -> str:
    """The first value against the second by the metric's order: higher or lower is better, or for a target-valued
    metric closer to its nominal value (at `level` for one at the declared level). Incomparable with no order."""
    from .metrics import METRICS, better, nominal
    m = METRICS[metric]
    unordered = m.direction == "none" or (m.direction == "target" and nominal(metric, level) is None)
    if va is None or vb is None or unordered:
        return "incomparable"
    win = better(metric, va, vb, level)
    return "equal" if win is None else "better" if win else "worse"


def comparison(rec: "Record", sides: list, metric: str) -> dict:
    """The comparison of two sides on `metric`, derived from them: the shared comparable_on coordinates (`dims`),
    the two values and the verdict for the first side by the metric's direction. Sides that differ on (or lack) a
    comparable_on coordinate are refused, naming each. Two of the study's own results need no [review]: they are
    compared on the rows both scored, with the paired difference (`paired`: diff, interval, p), and refused unless
    their runs share the protocol (openlab.study.variants)."""
    from .metrics import METRICS
    if len(sides) == 2 and all(n.kind == "result" for n in sides):
        from .variants import paired
        d = paired(rec, sides[0], sides[1], metric)
        va, vb = d["values"]
        verdict = _verdict(metric, va, vb, d.get("nominal"))
        return {"dims": {"protocol": "same sealed data, target and split", "rows": d["n"]}, "verdict": verdict,
                "metric": metric, "values": [va, vb], "paired": {k: v for k, v in d.items() if k not in ("metric", "values")}}
    on = _review(rec)["comparable_on"]
    if not on:
        raise StudyError("the sealed review declares no comparable_on", "without it openlab cannot tell comparable "
                         "results from incomparable ones", "add [review] comparable_on (a person's amendment)")
    if len({n.id for n in sides}) != 2:
        raise StudyError("a comparison sets two different results side by side")
    (ca, va), (cb, vb) = (_side(n, metric, on) for n in sides)
    differ = [c for c in on if c not in ca or c not in cb or coord_key(ca[c]) != coord_key(cb[c])]
    if differ:
        raise StudyError(f"incomparable: the two sides differ on {', '.join(differ)}",
                         "; ".join(f"{c}: {ca.get(c, 'missing')!r} vs {cb.get(c, 'missing')!r}" for c in differ),
                         "compare results under the same protocol; study_lattice shows which exist")
    if metric not in METRICS:
        raise StudyError(f"no metric `{metric}` in openlab.study.metrics", "its direction decides the verdict")
    bad = [n.id for n, v in zip(sides, (va, vb)) if not _number(v)]
    if bad:
        raise StudyError(f"{', '.join(bad)} has no {metric} value", "evidence carries it as a number field of "
                         "that name; a result as a computed metric")
    verdict = _verdict(metric, va, vb)
    return {"dims": {c: ca[c] for c in on}, "verdict": verdict, "metric": metric, "values": [va, vb], "paired": {}}


@dataclass(frozen=True)
class Comparison(Body):
    """Two results side by side on one metric: values a paper reports (evidence) or a study's result. They must
    agree on every coordinate the sealed review names comparable_on, so results under different protocols cannot
    be compared at all; the refusal names the coordinates that differ. `dims` (the shared coordinates), `values`
    and `verdict` (the first side against the second, by the metric's direction: higher, lower, or closer to a
    target-valued metric's nominal value; incomparable only for a metric with no order) are derived from the sides
    on write, so none can be typed."""
    KIND: ClassVar[str] = "comparison"
    LINKS: ClassVar[tuple] = (Link("compares", ("evidence", "result"), 2, 2), Link("under", ("plan", "amendment"), 1, 1))
    dims: dict
    verdict: Literal["better", "worse", "equal", "incomparable"]
    metric: str
    values: list
    paired: dict = field(default_factory=dict)      # two of the study's results: a - b on their shared rows

    def check(self, rec, actor, links):
        _latest_spec_node(rec, links["under"][0], "a comparison is made under the current sealed review")
        if all(n.kind == "result" for n in links["compares"]):
            for n in links["compares"]:
                _audited(run_of(rec, n), n.id, "a comparison on it")
        want = comparison(rec, links["compares"], self.metric)
        bad = [f"{k}: typed {getattr(self, k)!r}, derived {v!r}" for k, v in want.items() if getattr(self, k) != v]
        if bad:
            raise StudyError("the comparison does not follow from its sides", "; ".join(bad))


def recall_matches(rec: "Record", references: list) -> dict:
    """{reference key: source id} for each gold reference that shares an id with a source on record (a
    preprint's published_as counting as its DOI)."""
    index = sources_by_id(rec)
    for n in rec.find("source"):
        if n.payload["published_as"]:
            index.setdefault(("doi", n.payload["published_as"]), n)
    found = {source_key(r): resolve(index, r) for r in references}
    return {k: n.id for k, n in found.items() if n is not None}


@dataclass(frozen=True)
class Recall(Body):
    """How much of a gold set the study's search found: the references of a published review (`gold`, its DOI;
    OpenAlex referenced_works) within `filters` (since, until), each matched to a source on record by a shared
    id. `left_out` counts the references the filters (or OpenAlex) dropped. `matched` and `recall` are
    recomputed from the record on write."""
    KIND: ClassVar[str] = "recall"
    LINKS: ClassVar[tuple] = (Link("under", ("plan", "amendment"), 1, 1),)
    ACTORS: ClassVar[tuple] = ("openlab",)
    gold: str
    title: str
    filters: dict
    references: list
    left_out: dict
    matched: dict
    recall: float

    def check(self, rec, actor, links):
        _latest_spec_node(rec, links["under"][0], "recall is measured under the current sealed review")
        if not valid_id("doi", self.gold):
            raise StudyError("the gold review is named by its DOI in stored form", f"got {self.gold!r}")
        if not self.references:
            raise StudyError("the gold set is empty", "OpenAlex lists no references in the filters for it")
        keys = [source_key(identity(r)) for r in self.references]
        if len(set(keys)) != len(keys):
            raise StudyError("the gold set lists a reference twice")
        want = recall_matches(rec, self.references)
        if self.matched != want or self.recall != round(len(want) / len(keys), 4):
            raise StudyError("the recall does not match the record",
                             f"{len(want)} of {len(keys)} references are on record")


SCHEMAS.update({c.KIND: c for c in (Query, Source, Document, Coverage, Screen, Evidence, Comparison, Recall)})
KINDS = tuple(SCHEMAS)


# --- phase 3: claims -------------------------------------------------------------------------------------------
CLAIM_STATES = ("observed", "derived", "inferred", "hypothesized", "unknown")
#: what a claim of each state rests on: (relation, the kinds it points to, exactly one)
GROUNDS = {"observed": ("rests_on", ("evidence", "result"), False),
           "derived": ("rests_on", ("claim", "result", "evidence"), False),
           "inferred": ("rests_on", ("claim",), False),
           "hypothesized": ("tests", ("hypothesis",), True),
           "unknown": ("searched", ("coverage",), True)}
TOLERANCE = 1e-9                                    # relative: a claim's value against the value it is bound to
_REF = re.compile(r"([a-z_]+:[0-9a-f]{12})#([^\s#]+)")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def lookup(rec: "Record", ref: str) -> tuple:
    """(node, value) at `<node-id>#<dotted path>`; a list item is named by its index."""
    m = _REF.fullmatch(ref) if isinstance(ref, str) else None
    if m is None:
        raise StudyError(f"`{ref}` is not <node-id>#<dotted path>", 'e.g. "result:0123456789ab#metrics.mae.value"')
    node = rec.get(m[1])
    v = node.payload
    for k in m[2].split("."):
        if isinstance(v, dict) and k in v:
            v = v[k]
        elif isinstance(v, list) and k.isdigit() and int(k) < len(v):
            v = v[int(k)]
        else:
            raise StudyError(f"{node.id} has no `{m[2]}`", f"there is no `{k}` on the way")
    return node, v


def run_of(rec: "Record", result: "Node") -> "Node":
    return rec.get(rec.links(result.id, rel="of")[0][2])


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=TOLERANCE, abs_tol=1e-12)


def evaluate(expression: str, values: dict) -> float:
    """The logic checker: the LaTeX engine's value of `expression` with each `{name}` replaced by its value.
    Refused unless every value is used and the engine returns one finite number."""
    from ..engine import run
    unused = [k for k in values if "{%s}" % k not in expression]
    if not expression.strip() or unused:
        raise StudyError(f"the expression does not use {', '.join(unused)}" if unused else "the expression is empty",
                         "a derived claim's expression is LaTeX with each input written as {name}")
    src = re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}",
                 lambda m: f"{{({values[m[1]]!r})}}" if m[1] in values else m[0], expression)
    try:
        r = run(src)
        v = float(r.numeric if r.numeric is not None else r.value) if r.kind.name == "EXPR" else math.nan
    except Exception as e:                              # noqa: BLE001 -- the engine's refusal is the claim's
        raise StudyError("the LaTeX engine cannot evaluate the expression to a number",
                         f"{src}: {type(e).__name__}: {e}"[:300],
                         "write one expression over the inputs' {names}") from None
    if not math.isfinite(v):
        raise StudyError("the expression is not one finite number", f"{src} gives {r.value}")
    return v


def _waived(rec: "Record", evidence: "Node") -> bool:
    return any(rec.get(a).payload["decision"] == "waive" for a, _, _ in rec.links(evidence.id, rel="on", incoming=True))


def _grounded(rec: "Record", statement: str, node: "Node") -> None:
    """Refuse a ground no claim may rest on: a result whose run's truncation audit is not clean, or evidence from a
    retracted source that the statement does not mention and no person waived."""
    if node.kind == "result":
        _audited(run_of(rec, node), node.id, "a claim on it")
    if node.kind == "evidence":
        src = rec.get(node.payload["source"])
        why = retracted(rec, src)
        if why and not re.search(r"retract", statement, re.I) and not _waived(rec, node):
            raise StudyError(f"{node.id} quotes {src.id}, which is retracted", why,
                             "say so in the statement, or ask a person to waive it (study_approve, decision waive)")


def reproduced(rec: "Record", run: "Node") -> bool:
    """Whether a reproduction of the run matched it."""
    return any(rec.get(r).payload["matched"] for r, _, _ in rec.links(run.id, rel="of", incoming=True)
               if r.startswith("reproduction:"))


def _results(rec: "Record", node: "Node") -> list:
    """The results a ground rests on: a result itself, a claim's through its own grounds, else none."""
    if node.kind == "result":
        return [node]
    if node.kind != "claim":
        return []
    rel = GROUNDS[node.payload["state"]][0]
    return [r for _, _, d in rec.links(node.id, rel=rel) for r in _results(rec, rec.get(d))]


def signed_look(rec: "Record", run: "Node") -> bool:
    """Whether a holdout run's look is a person's look signed by the key the study pins: an unsigned look
    (a legacy record, or one an agent wrote) never makes a claim `measured`."""
    from .personkey import node_signed
    looks = [d for _, _, d in rec.links(run.id, rel="after")]
    return bool(looks) and node_signed(rec, rec.get(looks[0]))


def level_note(rec: "Record", claim: "Node") -> str:
    """Why a claim on holdout results is `recorded`, not `measured`, when its look is not signed; else ""."""
    grounds = [rec.get(d) for _, _, d in rec.links(claim.id, rel=GROUNDS[claim.payload["state"]][0])]
    runs = [run_of(rec, r) for g in grounds for r in _results(rec, g)]
    bad = [r.id for r in runs if r.payload["segment"] == "holdout" and not signed_look(rec, r)]
    return (f"recorded, not measured: the holdout look of {', '.join(bad)} is not a signed person look"
            if bad else "")


def claim_level(rec: "Record", grounds: list) -> str:
    """A claim's level from the results its grounds rest on: measured when each comes from the holdout, else
    recorded, with "+reproduced" when a reproduction matched the run of each (the two are independent); "" when it
    rests on no result."""
    runs = [run_of(rec, r) for g in grounds for r in _results(rec, g)]
    if not runs:
        return ""
    base = "measured" if all(r.payload["segment"] == "holdout" and signed_look(rec, r) for r in runs) else "recorded"
    return base + ("+reproduced" if all(reproduced(rec, r) for r in runs) else "")


def current_level(rec: "Record", claim: "Node") -> str:
    """A claim's level now: reproductions written after it count too."""
    return claim_level(rec, [rec.get(d) for _, _, d in rec.links(claim.id, rel=GROUNDS[claim.payload["state"]][0])])


def _outcome(rec: "Record", test: "Node") -> str:
    """A test of the model against a baseline: supported if it rejected with the model better, refuted if it
    rejected with the baseline better, inconclusive otherwise."""
    if not test.payload["rejected"]:
        return "inconclusive"
    paired = rec.get(rec.links(test.id, rel="from")[0][2]).payload["paired"][test.payload["baseline"]]
    return {"model": "supported", "baseline": "refuted"}.get(paired.get("better"), "inconclusive")


def tested(rec: "Record", ids: list) -> Optional[str]:
    """The verdict of the tests of these claims or hypotheses: None when there are none; supported (refuted) when
    every rejection found the model (the baseline) better; inconclusive otherwise."""
    tests = [rec.get(t) for i in ids for t, _, _ in rec.links(i, rel="of", incoming=True) if t.startswith("test:")]
    if not tests:
        return None
    found = {_outcome(rec, t) for t in tests} - {"inconclusive"}
    return found.pop() if len(found) == 1 else "inconclusive"


def claim_verdict(rec: "Record", state: str, links: dict, own: Optional[str] = None) -> str:
    """A claim's verdict from its links. Tests decide when there are any: of the claim itself (`own`, once it is
    written), of the hypothesis it tests and, for an inferred claim, of the hypotheses it answers. Otherwise an
    observed claim is supported by its checked ground, a derived claim takes its weakest premise claim's verdict
    (supported when it has none), and the rest are untested."""
    ids = ([own] if own else []) + [n.id for n in links.get("tests", [])]
    if state == "inferred":
        ids += [n.id for n in links.get("answers", []) if n.kind == "hypothesis"]
    verdict = tested(rec, ids)
    if verdict:
        return verdict
    if state == "observed":
        return "supported"
    if state == "derived":
        premises = {current_verdict(rec, n) for n in links.get("rests_on", []) if n.kind == "claim"}
        return next((v for v in ("refuted", "inconclusive", "untested") if v in premises), "supported")
    return "untested"


def current_verdict(rec: "Record", claim: "Node") -> str:
    """A claim's verdict now: the tests written after it count too."""
    links: dict = {}
    for _, rel, dst in rec.links(claim.id):
        links.setdefault(rel, []).append(rec.get(dst))
    return claim_verdict(rec, claim.payload["state"], links, own=claim.id)


def selected_from(rec: "Record", grounds: list) -> int:
    """The research trials a claim resting on results was selected from: the study's trial count when it is
    written (0 when it rests on no result). Printed as "selected from N trials" wherever the claim is."""
    if not any(_results(rec, g) for g in grounds):
        return 0
    from .ledger import shared
    return shared(rec.folder, rec)["trials"]


def claim_fields(rec: "Record", statement: str, state: str, links: dict) -> dict:
    """The fields of a claim openlab computes on write: its verdict, its level and the trials it was selected from
    (refusing a bad ground)."""
    grounds = links.get(GROUNDS.get(state, ("rests_on",))[0], [])
    for n in grounds:
        _grounded(rec, statement, n)
    return {"verdict": claim_verdict(rec, state, links), "level": claim_level(rec, grounds),
            "selected_from": selected_from(rec, grounds)}


@dataclass(frozen=True)
class Claim(Body):
    """A sentence the paper may print, resting on what its state demands (GROUNDS): observed on evidence or a
    result; derived on claims, results or evidence through `expression`; inferred on claims; hypothesized on the
    hypothesis it tests; unknown on the coverage of a search. A number is never typed: an observed value is its one
    input's, a derived value what the LaTeX engine makes of `expression` with each `{name}` read from `inputs`
    (name -> "<node-id>#<dotted path>", over exactly the nodes it rests on), within TOLERANCE. A result whose run's
    truncation audit is not clean, or evidence from a retracted source the statement does not mention and no person
    waived, is refused as a ground. `verdict` and `level` (reproduced once every result it rests on has a matched
    reproduction, else measured from the holdout, recorded from research) are computed on write (claim_fields), so
    typing them otherwise is refused; current_verdict and current_level give them now."""
    KIND: ClassVar[str] = "claim"
    LINKS: ClassVar[tuple] = (Link("rests_on", ("evidence", "result", "claim")), Link("tests", ("hypothesis",), 0, 1),
                              Link("searched", ("coverage",), 0, 1), Link("needs", ("assumption",)),
                              Link("answers", ("question", "hypothesis")))
    statement: str
    state: Literal[CLAIM_STATES]
    verdict: Literal["supported", "refuted", "inconclusive", "untested"]
    level: Literal["recorded", "measured", "recorded+reproduced", "measured+reproduced", ""] = ""
    value: Optional[float] = None
    expression: str = ""
    inputs: dict = field(default_factory=dict)
    selected_from: int = 0                  # the research trials its results were selected from (computed)

    def check(self, rec, actor, links):
        if not self.statement.strip():
            raise StudyError("a claim needs a statement", "the sentence the paper may print")
        rel, kinds, one = GROUNDS[self.state]
        stray = [r for r in ("rests_on", "tests", "searched") if r != rel and links.get(r)]
        if stray:
            raise StudyError(f"{self.state} claims cannot link `{stray[0]}`",
                             f"they rest on `{rel}` -> {' or '.join(kinds)}")
        grounds = links.get(rel, [])
        if not grounds or (one and len(grounds) > 1) or any(n.kind not in kinds for n in grounds):
            raise StudyError(f"{self.state} claims rest on {'one' if one else 'at least one'} `{rel}` link to "
                             f"{' or '.join(kinds)}", f"got {', '.join(n.id for n in grounds) or 'none'}")
        want = claim_fields(rec, self.statement, self.state, links)
        self._bound(rec, grounds)
        typed = [f"{k}: typed {getattr(self, k)!r}, computed {v!r}" for k, v in want.items() if getattr(self, k) != v]
        if typed:
            raise StudyError("a claim's verdict, level and selected_from are computed from what it rests on, never typed",
                             "; ".join(typed), "leave them to openlab (study_claim)")

    def _bound(self, rec, grounds) -> None:
        """The claim's number is bound to its inputs, and its inputs to its grounds."""
        bad = [k for k, r in self.inputs.items() if not _NAME.fullmatch(k) or not isinstance(r, str)]
        if bad:
            raise StudyError(f"input `{bad[0]}` is not name -> \"<node-id>#<dotted path>\"")
        read = {k: lookup(rec, r) for k, r in self.inputs.items()}
        ids = {n.id for n in grounds}
        for k, (node, v) in read.items():
            if node.id not in ids:
                raise StudyError(f"input `{k}` reads {node.id}, which the claim does not rest on",
                                 fix="link it rests_on")
            if not _number(v):
                raise StudyError(f"input `{k}` ({self.inputs[k]}) is not a number", f"got {v!r}")
        values = {k: v for k, (_, v) in read.items()}
        if self.state == "derived":
            unread = sorted(ids - {n.id for n, _ in read.values()})
            if unread:
                raise StudyError("a derived claim rests on what it computes from",
                                 f"no input reads {', '.join(unread)}")
            if self.value is None:
                raise StudyError("a derived claim states its value", "openlab checks it against the expression's")
            got = evaluate(self.expression, values)
            if not _close(self.value, got):
                raise StudyError(f"the value {self.value!r} is not what the expression gives",
                                 f"the LaTeX engine evaluates {self.expression} to {got!r}", "state the computed value")
            return
        if self.expression:
            raise StudyError("only a derived claim has an expression")
        if self.state == "observed" and (self.value is not None or values):
            if self.value is None or len(values) != 1 or not _close(self.value, next(iter(values.values()))):
                raise StudyError("an observed number is its one input's", f"value {self.value!r}, inputs {values}",
                                 'bind it to the field it is read from: inputs {"v": "<node-id>#<dotted path>"}')
        elif self.value is not None or values:
            raise StudyError(f"{self.state} claims carry no number",
                             "a number is observed (read from a node) or derived (computed by the LaTeX engine)")


@dataclass(frozen=True)
class Approval(Body):
    """A person's decision on a claim or a paper (approve, reject or waive), with the reason. On evidence it can
    only waive its source's retraction, so that the evidence may ground a claim."""
    KIND: ClassVar[str] = "approval"
    LINKS: ClassVar[tuple] = (Link("on", ("claim", "paper", "evidence"), 1, 1),)
    ACTORS: ClassVar[tuple] = ("person",)
    decision: Literal["approve", "reject", "waive"]
    reason: str

    def check(self, rec, actor, links):
        if not self.reason.strip():
            raise StudyError("an approval needs a reason")
        on = links["on"][0]
        if on.kind == "evidence" and (self.decision != "waive" or not retracted(rec, rec.get(on.payload["source"]))):
            raise StudyError("on evidence a person only waives its source's retraction",
                             "the decision is waive and the source is retracted")


@dataclass(frozen=True)
class Paper(Body):
    """A build of the study's paper: the .tex and .pdf blobs, the node ids its binding macros resolved (each linked
    `renders`), the lint findings, the signature ("" until signed) and the record head it was built at (the head
    when the paper is written). `sources` is the blob of the prose and paper.toml it was built from ({path: text});
    `figures` the figures with the code that drew them. On write the prose is analysed again against the record
    (openlab.study.paper.build.analyse): a binding that does not resolve or is stale refuses the paper, and so does
    any finding of the lints and gates without a person's waiver; `lint` holds the waived findings."""
    KIND: ClassVar[str] = "paper"
    LINKS: ClassVar[tuple] = (Link("renders", ANY),)
    tex_sha: str
    pdf_sha: str
    bindings: list
    lint: list
    head: str
    sources: str
    signed: str = ""
    figures: dict = field(default_factory=dict)

    def check(self, rec, actor, links):
        import json
        from .paper.build import analyse
        from .paper.lint import Finding, waiver
        if not (_SHA.fullmatch(self.tex_sha) and _SHA.fullmatch(self.head)):
            raise StudyError("a paper's tex_sha and head are sha256s")
        _sha(self.pdf_sha, "pdf sha")
        if self.head != rec.head():
            raise StudyError("a paper names the record head it was built at", f"the head is {rec.head()[:12]}, "
                             f"the paper names {self.head[:12]}", "build it again")
        if sorted(set(self.bindings)) != sorted({n.id for n in links.get("renders", [])}):
            raise StudyError("a paper renders exactly the nodes it binds", "one `renders` link per bound node id")
        bounds, found = analyse(rec, rec.folder, json.loads(rec.blob(self.sources)))[1:]
        bad = [f for f in found if not f.waivable]
        if bad:
            raise StudyError(f"{len(bad)} bindings of the paper do not resolve or are stale",
                             "; ".join(f"{f.location}: {f.message}" for f in bad[:5]), "fix the prose or run again")
        if sorted({b.node for b in bounds if b.node}) != sorted(set(self.bindings)):
            raise StudyError("a paper binds exactly the nodes its prose binds", "paper.bindings differs from its sources")
        waived = {d["id"]: d.get("waived_by") for d in self.lint}
        for f in found + [Finding.of(d) for d in self.lint]:
            w = waiver(rec, f)
            if w is None or waived.get(f.id) != w.id:
                raise StudyError(f"finding {f.id} has no person's waiver", f"{f.location}: {f.message}",
                                 "fix the prose; only a person's waiver lets a finding pass")


def _tags(tags: list, what: str) -> set:
    """Tags as compared: stripped and case-folded, from a non-empty list of non-empty strings."""
    if not tags or not all(isinstance(t, str) and t.strip() for t in tags):
        raise StudyError(f"a {what} needs tags", 'a list of words naming what it is about, e.g. ["emos", "lead time"]')
    return {t.strip().casefold() for t in tags}


def _said(body: Body, *names: str) -> None:
    empty = [n for n in names if not getattr(body, n).strip()]
    if empty:
        raise StudyError(f"a {body.KIND.replace('_', ' ')} needs {' and '.join(empty)}")


@dataclass(frozen=True)
class Question(Body):
    """A question, part of a broader one (the question tree). `out_of_scope` says why it
    is recorded but not pursued. Its status is derived from its children (claims.tree), never stored."""
    KIND: ClassVar[str] = "question"
    LINKS: ClassVar[tuple] = (Link("part_of", ("question",), 0, 1),)
    text: str
    out_of_scope: str = ""

    def check(self, rec, actor, links):
        _said(self, "text")


@dataclass(frozen=True)
class Hypothesis(Body):
    """A falsifiable answer to a question: the prediction, its mechanism, what would falsify it, and tags. A dead
    end sharing a tag with it must be linked `differs_from`, with what is different now (`difference`), or the
    hypothesis is refused: history cannot silently repeat."""
    KIND: ClassVar[str] = "hypothesis"
    LINKS: ClassVar[tuple] = (Link("part_of", ("question",), 1, 1), Link("differs_from", ("dead_end",)))
    statement: str
    prediction: str
    mechanism: str
    falsified_by: str
    tags: list
    difference: str = ""

    def check(self, rec, actor, links):
        _said(self, "statement", "prediction", "falsified_by")
        tags, differs = _tags(self.tags, "hypothesis"), {n.id for n in links.get("differs_from", [])}
        if differs and not self.difference.strip():
            raise StudyError("a hypothesis that differs from a dead end says how (difference)")
        for d in rec.find("dead_end"):
            shared = tags & _tags(d.payload["tags"], "dead end")
            if shared and d.id not in differs:
                raise StudyError(f"dead end {d.id} shares {', '.join(sorted(shared))} with this hypothesis",
                                 f"tried before: {d.payload['statement']}; it failed because {d.payload['reason']}",
                                 "link it differs_from and say what is different now (difference), "
                                 "or drop the hypothesis")


def paired_p(result: "Node", baseline: str, rec: "Optional[Record]" = None) -> float:
    """The p of a result's paired difference against `baseline`: one of its baselines, or `result:<id>`, another
    of the study's results under the same protocol (computed from both runs' rows, openlab.study.variants)."""
    if baseline.startswith("result:") and rec is not None:
        from .spec import sealed
        from .variants import paired as variant
        other = rec.get(baseline)
        _audited(run_of(rec, other), other.id, "a test against it")
        spec = sealed(rec)
        d = variant(rec, result, other, spec.evaluate["metrics"][0])
        if d["p"] is None:
            raise StudyError(f"the paired difference of {result.id} and {other.id} has no p",
                             f"{d['metric']} is undefined on their shared rows")
        return d["p"]
    paired = result.payload.get("paired", {})
    if (paired.get(baseline) or {}).get("p") is None:
        raise StudyError(f"{result.id} has no paired p against `{baseline}`",
                         f"its paired entries: {', '.join(paired) or 'none'}")
    return paired[baseline]["p"]


def alpha_step(rec: "Record", alpha_j: float, p: float) -> dict:
    """The next test's entry in the alpha-investing ledger (Foster and Stine 2008): the wealth before (the last
    test's wealth after, or the sealed [inference] wealth), whether p rejects at alpha_j, and the wealth after (a
    rejection earns the payout; otherwise the test costs alpha_j / (1 - alpha_j)). Refused when alpha_j asks more
    than the wealth pays for (wealth / (1 + wealth)), so the wealth never goes below 0 and a spent budget refuses."""
    from .spec import sealed                            # spec imports the record, which imports this module
    s = sealed(rec)
    inf = s.inference if s else {}
    if not inf:
        raise StudyError("the study has no error budget", "a test spends the sealed [inference] budget",
                         "tests run in an experiment study, which has [inference] by default")
    last = rec.last("test")
    w = last.payload["wealth_after"] if last else inf["wealth"]
    rejected, payable = p <= alpha_j, 0 < alpha_j <= w / (1 + w)
    after = (w + inf["payout"] if rejected else w - alpha_j / (1 - alpha_j)) if payable else -1.0
    if after < 0:                                       # also a cost that rounds below 0 at the cap
        raise StudyError("the error budget is spent" if w <= 0 else
                         f"alpha_j {alpha_j} is more than the alpha budget left pays for",
                         f"alpha budget left: {w:.6g}; it pays a test at most alpha_j {w / (1 + w):.6g}",
                         "a person may amend [inference], with the reason" if w <= 0 else "test at a smaller alpha_j")
    return {"wealth_before": w, "rejected": rejected, "wealth_after": after}


@dataclass(frozen=True)
class Test(Body):
    """One test spending the study's error budget: a claim or hypothesis tested at level
    alpha_j on the paired difference of a result's model against `baseline`: one of its baselines, or another of
    the study's results (`result:<id>`, linked `against`; a variant against a variant, on their shared rows). Its p is that entry's (never typed)
    and its ledger fields follow alpha_step from the previous test. One test per target, result and baseline: a
    second look at the same p cannot buy a rejection."""
    KIND: ClassVar[str] = "test"
    LINKS: ClassVar[tuple] = (Link("of", ("claim", "hypothesis"), 1, 1), Link("from", ("result",), 1, 1),
                              Link("against", ("result",), 0, 1))
    family: str
    baseline: str
    alpha_j: float
    p: float
    rejected: bool
    wealth_before: float
    wealth_after: float

    def check(self, rec, actor, links):
        _said(self, "family")
        res, of = links["from"][0], links["of"][0]
        _audited(run_of(rec, res), res.id, "a test on it")
        against = [n.id for n in links.get("against", [])]
        if self.baseline.startswith("result:") != bool(against) or (against and against != [self.baseline]):
            raise StudyError("a test against another result links it (`against`)", f"baseline {self.baseline}, "
                             f"against {', '.join(against) or 'nothing'}")
        p = paired_p(res, self.baseline, rec)
        if self.p != p:
            raise StudyError("a test's p is its result's",
                             f"typed {self.p!r}; {res.id} has p {p!r} against {self.baseline}")
        for t in rec.find("test", baseline=self.baseline):
            if {rel: d for _, rel, d in rec.links(t.id)} == {"of": of.id, "from": res.id}:
                raise StudyError(f"{of.id} was already tested on {res.id} against {self.baseline} ({t.id})",
                                 "a second look at the same p at a higher alpha_j would buy a rejection",
                                 "test it on a new result")
        want = alpha_step(rec, self.alpha_j, self.p)
        typed = [f"{k}: typed {getattr(self, k)!r}, ledger {v!r}" for k, v in want.items() if getattr(self, k) != v]
        if typed:
            raise StudyError("the test does not follow the alpha-investing ledger", "; ".join(typed))


@dataclass(frozen=True)
class Assumption(Body):
    """An assumption claims may need: what it rests on and the test that could break it."""
    KIND: ClassVar[str] = "assumption"
    statement: str
    rests_on: str
    breaks_if: str

    def check(self, rec, actor, links):
        _said(self, "statement", "rests_on", "breaks_if")


@dataclass(frozen=True)
class Break(Body):
    """An assumption shown not to hold, by results or evidence; every claim that needs it, and every claim resting on
    those, is then at risk (claims.at_risk)."""
    KIND: ClassVar[str] = "break"
    LINKS: ClassVar[tuple] = (Link("breaks", ("assumption",), 1, 1), Link("shown_by", ("result", "evidence"), 1))
    reason: str

    def check(self, rec, actor, links):
        _said(self, "reason")


@dataclass(frozen=True)
class DeadEnd(Body):
    """A falsified hypothesis or an abandoned run, kept with the reason and what showed it.
    A dead end of a hypothesis carries its tags, so a later hypothesis on the same thing has to face it. Of a
    question, it is an operational dead end: a way of answering it dropped (a data source that cannot serve, an
    approach that does not run). In a replication it may also be of an approach to the target paper (`of` the
    target, a quoted passage, the source, a fidelity or replication): a reading of the method dropped, shown by the
    paper's own text or a result. A claim may show it too (the finding that closed the idea)."""
    KIND: ClassVar[str] = "dead_end"
    LINKS: ClassVar[tuple] = (Link("of", ("hypothesis", "question", "run", "target", "evidence", "source", "fidelity",
                                          "replication"), 1, 1),
                              Link("shown_by", ("result", "evidence", "test", "comparison", "fidelity",
                                                "replication", "source", "document", "claim"), 1))
    statement: str
    reason: str
    tags: list

    def check(self, rec, actor, links):
        _said(self, "statement", "reason")
        tags, of = _tags(self.tags, "dead end"), links["of"][0]
        missing = _tags(of.payload["tags"], "hypothesis") - tags if of.kind == "hypothesis" else set()
        if missing:
            raise StudyError("a dead end of a hypothesis carries its tags", f"missing {', '.join(sorted(missing))}")


def contra_state(coverage: "Node", found: list) -> str:
    return "found" if found else "none_found" if coverage.payload["verdict"] == "sufficient" else "insufficient"


@dataclass(frozen=True)
class ContraSearch(Body):
    """A search for what contradicts a claim: the coverage of its own queries (each `for` the
    claim) and the evidence found against it. `state` follows: found when it links evidence, none_found when nothing
    was found by a search the review's floor judges sufficient, insufficient otherwise."""
    KIND: ClassVar[str] = "contra_search"
    LINKS: ClassVar[tuple] = (Link("for", ("claim",), 1, 1), Link("searched", ("coverage",), 1, 1),
                              Link("found", ("evidence",)))
    state: Literal["found", "none_found", "insufficient"]

    def check(self, rec, actor, links):
        claim, cov, found = links["for"][0], links["searched"][0], links.get("found", [])
        if any(rec.links(q, rel="for") != [(q, "for", claim.id)] for _, _, q in rec.links(cov.id, rel="of")):
            raise StudyError(f"{cov.id} is not a search for {claim.id}", "a contradiction search records its own "
                             "queries, each linked to the claim", "run study_contra with queries")
        own = {d for _, _, d in rec.links(claim.id, rel="rests_on")} & {n.id for n in found}
        if own:
            raise StudyError(f"the claim rests on {', '.join(sorted(own))}",
                             "what a claim rests on does not contradict it")
        want = contra_state(cov, found)
        if self.state != want:
            raise StudyError(f"the search's state is {want}", f"typed {self.state}: found links evidence; none_found "
                             "needs a sufficient search; otherwise insufficient")


SCHEMAS.update({c.KIND: c for c in (Claim, Approval, Paper, Question, Hypothesis, Test, Assumption, Break, DeadEnd,
                                    ContraSearch)})
KINDS = tuple(SCHEMAS)


# --- phases 5-6: retractions and monitoring ---------------------------------------------------------------------
RETRACTION_TYPES = ("retraction", "withdrawal", "removal", "partial_retraction")   # Crossref update types that withdraw
OPENALEX_FLAG = "is_retracted"


def retraction_note(via: str, type_: str, notice: str, date: str) -> str:
    """A retraction as a source's note says it."""
    return "OpenAlex is_retracted" if via == "openalex" else f"Crossref {type_} notice {notice} ({date})"


def retractions(rec: "Record", source: "Node") -> list:
    """The retraction nodes of a source, oldest first."""
    return [rec.get(r) for r, _, _ in rec.links(source.id, rel="of", incoming=True) if r.startswith("retraction:")]


def retracted(rec: "Record", source: "Node") -> str:
    """Why a source is retracted, "" when it is not: the flag it was recorded with, else each notice found since."""
    if source.payload["retracted"]:
        return source.payload["retraction_note"]
    return "; ".join(retraction_note(r.payload["via"], r.payload["type"], r.payload["notice"], r.payload["valid_from"])
                     for r in retractions(rec, source))


@dataclass(frozen=True)
class Retraction(Body):
    """A retraction found on a source after it was recorded (bitemporal): `valid_from` is the
    notice's date (valid time; "" when the index gives none, as OpenAlex's flag does) and the node's `at` is when
    openlab detected it (record time). `via` names the check: a Crossref update notice on the source's DOI (`notice`
    its DOI, `type` one of RETRACTION_TYPES) or OpenAlex's flag on its work (`notice` the W id, `type` is_retracted).
    One per source and notice."""
    KIND: ClassVar[str] = "retraction"
    LINKS: ClassVar[tuple] = (Link("of", ("source",), 1, 1),)
    ACTORS: ClassVar[tuple] = ("openlab",)
    via: Literal[RETRACTION_CHECKS]
    type: str
    notice: str
    valid_from: str = ""

    def check(self, rec, actor, links):
        src = links["of"][0]
        p = src.payload
        ok = (self.type in RETRACTION_TYPES and valid_id("doi", self.notice) and bool(p["doi"])) \
            if self.via == "crossref" else (self.type == OPENALEX_FLAG and self.notice == p["openalex"] != "")
        if not ok:
            raise StudyError(f"not a {self.via} retraction of {src.id}",
                             f"crossref: a notice DOI of type {', '.join(RETRACTION_TYPES)} on a source with a DOI; "
                             f"openalex: type {OPENALEX_FLAG}, the notice the source's W id")
        if self.valid_from and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.valid_from):
            raise StudyError(f"valid_from {self.valid_from!r} is not an ISO date", "the notice's date, or \"\"")
        if any(r.payload["notice"] == self.notice for r in retractions(rec, src)):
            raise StudyError(f"{src.id} already has retraction notice {self.notice}", "one per source and notice")


def watched(rec: "Record", coverage: "Optional[Node]") -> dict:
    """What a monitoring pass reports, from the record: the sources first recorded by the discovery `coverage`
    judges, and the retractions, the evidence contradiction searches found and the outcomes written since the
    previous pass."""
    prev = rec.last("watch")
    since = prev.seq if prev else 0
    queries = [d for _, _, d in rec.links(coverage.id, rel="of")] if coverage else []
    return {"new_sources": [s for q in queries for s, _, _ in rec.links(q, rel="found_by", incoming=True)],
            "contradicting": sorted({d for c in rec.find("contra_search") if c.seq > since
                                     for _, _, d in rec.links(c.id, rel="found")}),
            "outcomes": [o.id for o in rec.find("outcome") if o.seq > since],
            "retractions": [r.id for r in rec.find("retraction") if r.seq > since]}


@dataclass(frozen=True)
class Watch(Body):
    """One monitoring pass: the sealed review's discovery run again (`of` its coverage; none
    in a study without a review) and the sources it recorded first (`new_sources`); the retractions found since the
    previous pass (each linked `found`); the evidence contradiction searches found and the forecast outcomes filled
    since then (`contradicting`, `outcomes`); and the checks that answered for each source it watched (`checked`:
    {source id: [checks]}, an empty list unchecked). All but `checked` are recomputed from the record on write."""
    KIND: ClassVar[str] = "watch"
    LINKS: ClassVar[tuple] = (Link("of", ("coverage",), 0, 1), Link("found", ("retraction",)))
    ACTORS: ClassVar[tuple] = ("openlab",)
    new_sources: list
    contradicting: list
    outcomes: list
    checked: dict

    def check(self, rec, actor, links):
        cov = (links.get("of") or [None])[0]
        prev = rec.last("watch")
        if cov is not None and (cov.seq < (prev.seq if prev else 0) or any(
                rec.links(q, rel="for") for _, _, q in rec.links(cov.id, rel="of"))):
            raise StudyError(f"{cov.id} is not a discovery since the previous pass",
                             "a watch judges the discovery it ran, not a contradiction search or an older one")
        want = watched(rec, cov)
        found = want.pop("retractions")
        typed = [f"{k}: typed {getattr(self, k)}, recomputed {v}" for k, v in want.items() if getattr(self, k) != v]
        if typed or sorted(n.id for n in links.get("found", [])) != sorted(found):
            raise StudyError("the watch does not match the record", "; ".join(typed) or
                             f"it links the retractions found since the previous pass: {', '.join(found) or 'none'}")
        bad = [s for s, c in self.checked.items() if not s.startswith("source:") or not set(c) <= set(RETRACTION_CHECKS)]
        if bad:
            raise StudyError(f"watch.checked names {bad[0]}", "{source id: the retraction checks that answered}")


SCHEMAS.update({c.KIND: c for c in (Retraction, Watch)})
# --- phase 4: replication --------------------------------------------------------------------------------------
REPLICATION_VERDICTS = ("replicated", "weaker", "not_replicated", "inconclusive")
NOT_FAITHFUL = "rebuild not faithful"


def replication_tier(lo: float, hi: float, paper: float, reference: Optional[float] = None,
                     near: bool = False, better: str = "") -> str:
    """A claim's verdict from our interval [lo, hi]. The reference is what the claim improves on (0 for a
    difference): an interval holding it is inconclusive if it also holds the paper's value (too wide to tell), else
    not_replicated; one on the other side of it is not_replicated. Otherwise replicated when it holds the paper's
    value (or `near`: our value is within tolerance of it) or lies beyond it, away from the reference, and weaker
    when it falls short. With no reference (a level, not an effect) there is no side to lose.

    `better` ("higher" or "lower", the metric's direction) says which side of the reference is an improvement. A
    paper's value on the worse side claims no improvement (a method that ties or trails its best baseline): it is a
    level, replicated when we hold it or do better, not_replicated when we fall short of it."""
    has_paper = near or lo <= paper <= hi
    if better in ("higher", "lower") and reference is not None and (paper > reference) != (better == "higher"):
        ahead = lo > paper if better == "higher" else hi < paper
        return "replicated" if has_paper or ahead else "not_replicated"
    ref = 0.0 if reference is None else reference
    if reference is not None and lo <= reference <= hi:
        return "inconclusive" if has_paper else "not_replicated"
    if reference is not None and (lo > ref) != (paper > ref):
        return "not_replicated"
    beyond = lo > paper if paper >= ref else hi < paper
    return "replicated" if has_paper or beyond else "weaker"


def _typed(body: Body, want: dict, what: str) -> None:
    """Refuse a body whose computed fields were typed otherwise."""
    bad = [f"{k}: typed {getattr(body, k)!r}, computed {v!r}" for k, v in want.items() if getattr(body, k) != v]
    if bad:
        raise StudyError(f"{what}, never typed", "; ".join(bad), "leave them to openlab")


def _within(target: "Node", tolerance: float, justification: str) -> None:
    """The target's tolerance policy: up to `tolerance`, or up to `max_tolerance` with a justification."""
    t = target.payload
    if not 0 < tolerance <= t["max_tolerance"]:
        raise StudyError(f"tolerance {tolerance} is outside the target's policy", f"0 < tolerance <= "
                         f"{t['max_tolerance']} (max_tolerance), set before any run")
    if tolerance > t["tolerance"] and not justification.strip():
        raise StudyError(f"tolerance {tolerance} is wider than the target's {t['tolerance']}",
                         "a wider tolerance, up to max_tolerance, says why", "give a justification")


def _paper(rec: "Record", target: "Node", evidence: "Node", metric: str, ours: Optional[float] = None) -> tuple:
    """(the target paper's value of `metric` in the metric's own unit, the value as the paper writes it when that
    differs, e.g. "92.62%"). A share (coverage, accuracy) the paper writes in percent is divided by 100: percent is
    the evidence's unit (declared, from [review.extract], or a % sign in its quote) or, for a share, the target's
    paper_unit. A share above 1 with no unit is refused rather than compared."""
    from .metrics import METRICS
    want = rec.links(target.id, rel="of")[0][2]
    if evidence.payload["source"] != want:
        raise StudyError(f"{evidence.id} quotes {evidence.payload['source']}, not the target paper {want}")
    v = evidence.payload["fields"].get(metric)
    if not _number(v):
        raise StudyError(f"{evidence.id} has no {metric} value", "the paper's value is the evidence's field named "
                         "by the metric", f"extract the row with {metric} in its fields")
    m = METRICS.get(metric)
    share = m is not None and m.unit == "fraction"
    unit = evidence.payload.get("units", {}).get(metric) or (target.payload.get("paper_unit", "") if share else "")
    if not unit and Decimal(repr(v)) in _percents(evidence.payload["quote"]):
        unit = "percent"                            # evidence extracted before openlab recorded its units
    if unit == "percent":
        if not share:
            raise StudyError(f"{evidence.id} gives {metric} in percent, and {metric} is not a share",
                             f"{metric} is in its own units; openlab converts percent only for a share such as "
                             "coverage or accuracy", f"extract {metric} as the number it is, without unit=percent")
        return float(Decimal(repr(v)) / 100), f"{v:g}%"
    if share and unit != "fraction" and abs(v) > 1:
        theirs = f" (ours: {ours:.4g})" if ours is not None else ""
        raise StudyError(f"the paper's {metric} is {v:g}{theirs}: its unit is ambiguous",
                         f"{metric} is a share between 0 and 1, and {v:g} reads as a percentage, but neither the quote "
                         "(no % sign), the extract nor the target says so; openlab will not compare a fraction with a "
                         "percentage", f"declare unit=percent on the extract (openlab study extract ... --unit "
                         f"{metric}=percent) or on the target (openlab study replicate ... --paper-unit percent)")
    return v, ""


def _ours(rec: "Record", result, metric: str, arm: str, group: str = "") -> dict:
    """{value, lo, hi} of `metric` for one arm of a result (the model, or a baseline it scored), from a run whose
    truncation audit is clean or that was separated. Several results (a paper's number averaged over horizons,
    datasets or seeds, one run each): the uniform mean of their values, with `n` the number of results. Its interval
    holds two uncertainties. The rows': the mean of the runs' interval bounds (conservative: as wide as if the runs'
    errors moved together). The seeds': runs on the same protocol (data, target, split) that differ only in their
    seed are draws of one quantity, so their mean has a standard error sd/sqrt(n), taken at Student's t with n - 1
    degrees of freedom; runs on different protocols (datasets, horizons) are fixed parts of the mean, not draws. The
    two are added in quadrature on each side; `seed_half` is the seeds' part."""
    if isinstance(result, (list, tuple)):
        if len(result) == 1:
            return _ours(rec, result[0], metric, arm, group)
        if len({r.id for r in result}) != len(result):
            raise StudyError("the same result is named twice", "a mean over runs counts each run once")
        each = [_ours(rec, r, metric, arm, group) for r in result]
        n = len(each)
        out = {"value": sum(m["value"] for m in each) / n, "n": n}
        if all(m.get("lo") is not None and m.get("hi") is not None for m in each):
            half = _seed_half(rec, result, [m["value"] for m in each])
            lo, hi = sum(m["lo"] for m in each) / n, sum(m["hi"] for m in each) / n
            v = out["value"]
            out.update(lo=v - math.hypot(v - lo, half), hi=v + math.hypot(hi - v, half), seed_half=half)
        if each[0].get("nominal") is not None:
            out["nominal"] = each[0]["nominal"]
        return out
    _audited(run_of(rec, result), result.id, "a replication verdict on it")
    p = result.payload
    if group:
        groups = p.get("groups") or {}
        if group not in groups:
            raise StudyError(f"{result.id} scored no group `{group}`",
                             f"its groups: {', '.join(groups)}" if groups else "the run scored no groups",
                             "set evaluate.by = \"key\" before the run, and name a group it scored")
        p = groups[group]
    scores = p["metrics"] if arm == "model" else p["baselines"].get(arm)
    if scores is None:
        raise StudyError(f"{result.id} scored no `{arm}`", f"its arms: model, {', '.join(p['baselines'])}")
    m = scores.get(metric) or {}
    if m.get("value") is None:
        raise StudyError(f"{result.id} has no {metric} for {arm}", m.get("reason", "the run did not score it"))
    return m


def _seed_half(rec: "Record", results: list, values: list) -> float:
    """The half-width the seeds add to a mean over `results` (see _ours): per protocol with several runs, t times
    sd/sqrt(n_g), weighted n_g/N as the uniform mean weights it, added in quadrature over protocols."""
    from scipy.stats import t as student
    by: dict = {}
    for r, v in zip(results, values):
        key = json.dumps(_protocol(rec, r, ""), sort_keys=True, default=str)
        by.setdefault(key, []).append(v)
    from .spec import sealed
    sp = sealed(rec)
    alpha = sp.evaluate.get("alpha", 0.05) if sp is not None else 0.05
    N, total = len(values), 0.0
    for vs in by.values():
        k = len(vs)
        if k > 1:
            mean = sum(vs) / k
            se = math.sqrt(sum((x - mean) ** 2 for x in vs) / (k - 1) / k)
            total += (k / N * float(student.ppf(1 - alpha / 2, k - 1)) * se) ** 2
    return math.sqrt(total)


def _protocol(rec: "Record", result, metric: str) -> tuple:
    """What a fidelity certifies: the data, target and split of the run behind the result, and the metric's
    version (for several results, the set of theirs)."""
    if isinstance(result, (list, tuple)):
        return tuple(sorted(json.dumps(_protocol(rec, r, metric), sort_keys=True, default=str) for r in result))
    run = run_of(rec, result).payload
    spec = run.get("choices", {}).get("spec", {})
    return (run["data"], [spec.get(k) for k in ("data", "target", "split")], run["metric_versions"].get(metric))


def fidelity_fields(rec: "Record", links: dict, metric: str, arm: str, tolerance: float, justification: str,
                    group: str = "") -> dict:
    """A fidelity's computed fields: the paper's value (in the metric's unit, and as written when that differs),
    ours, and whether they agree within tolerance (in the metric's unit)."""
    target = links["for"][0]
    _within(target, tolerance, justification)
    ours = _ours(rec, links["ours"], metric, arm, group)["value"]
    paper, written = _paper(rec, target, links["paper"][0], metric, ours)
    return {"paper": paper, "ours": ours, "passed": abs(ours - paper) <= tolerance + 1e-12, "paper_written": written,
            "in_sample": _in_sample(rec, links["ours"])}


def replication_fields(rec: "Record", links: dict, metric: str, arm: str, tolerance: float,
                       justification: str, group: str = "") -> dict:
    """A replication's computed fields: the paper's value, ours with its interval, the reference (the best paper
    value among the gating fidelities: what the claim improves on) and the verdict, forced to inconclusive (rebuild
    not faithful) unless every gating fidelity passed.

    A target-valued metric (coverage90, best at 0.9) states a level, not an effect: the claim replicates when our
    value is within tolerance of the paper's (or the paper's is inside our interval), else not_replicated. Its
    reference is the gating baseline closest to the nominal value, and `vs_reference` says whether ours is closer to
    the nominal value than that baseline, or n/a with the reason."""
    from .metrics import METRICS, best, nominal
    target, evidence, result, gates = links["for"][0], links["paper"][0], links["ours"], links["gated_by"]
    _within(target, tolerance, justification)
    ours = _ours(rec, result, metric, arm, group)
    for g in gates:
        if g.payload.get("group", "") != group:
            raise StudyError(f"{g.id} checks the group `{g.payload.get('group', '') or 'all rows'}`, this claim "
                             f"`{group or 'all rows'}`", "a fidelity gates the claims of its own group",
                             "gate it by the fidelities of the same group")
    paper, written = _paper(rec, target, evidence, metric, ours["value"])
    if ours.get("lo") is None or ours.get("hi") is None:
        raise StudyError(f"{', '.join(r.id for r in result)} has no interval for {metric} of {arm}",
                         "a verdict needs our interval")
    m = METRICS.get(metric)
    if m is None or m.direction == "none":
        raise StudyError(f"{metric} has no direction", "the reference is the best gating baseline, by the metric's "
                         "direction", "register the metric with higher_is_better, or with the nominal value it is "
                         "best at")
    for g in gates:
        if rec.links(g.id, rel="for")[0][2] != target.id or g.payload["metric"] != metric:
            raise StudyError(f"{g.id} is not a {metric} fidelity of this target")
        if rec.links(g.id, rel="paper")[0][2] == evidence.id:
            raise StudyError(f"{g.id} checks the very value this claims", "a claim cannot gate itself")
        if _protocol(rec, [rec.get(d) for _, _, d in rec.links(g.id, rel="ours")], metric) != \
                _protocol(rec, result, metric):
            raise StudyError(f"{g.id} certifies another protocol", "a fidelity vouches only for runs with the same "
                             "data, target, split and metric version", "check the baselines on this run's protocol")
    failed = [g.id for g in gates if not g.payload["passed"]]
    near = abs(ours["value"] - paper) <= tolerance + 1e-12
    out = {"paper": paper, "ours": ours["value"], "lo": ours["lo"], "hi": ours["hi"], "paper_written": written,
           "in_sample": _in_sample(rec, result),
           "reason": f"{NOT_FAITHFUL}: {', '.join(failed)} failed" if failed else ""}
    baselines = [_paper(rec, target, rec.get(rec.links(g.id, rel="paper")[0][2]), metric)[0] for g in gates]
    if m.direction == "target":
        nom = ours.get("nominal", nominal(metric))         # the result records the level it was scored at
        reference = best(metric, baselines, nom)
        holds = near or ours["lo"] <= paper <= ours["hi"]
        return {**out, "reference": reference, "nominal": nom,
                "verdict": "inconclusive" if failed else "replicated" if holds else "not_replicated",
                "vs_reference": _vs_reference(metric, ours["value"], reference, nom)}
    reference = best(metric, baselines)
    if paper == reference:
        raise StudyError(f"the paper's {metric} equals its baseline's ({paper})", "there is no effect to replicate")
    verdict = "inconclusive" if failed else replication_tier(ours["lo"], ours["hi"], paper, reference, near=near,
                                                                better=m.direction)
    return {**out, "reference": reference, "nominal": None, "verdict": verdict, "vs_reference": ""}


def _vs_reference(metric: str, ours: float, reference, nom) -> str:
    """Whether ours beats the paper's best baseline on a target-valued metric: closer to its nominal value."""
    if nom is None:
        return (f"n/a: {metric} is best at the declared level, and the result does not record one "
                "(set evaluate.level and run again)")
    da, db = abs(ours - nom), abs(reference - nom)
    if da == db:
        return f"n/a: as far from {nom:g} as the baseline's {reference:.4g} ({da:.4g} away)"
    return f"{'closer to' if da < db else 'farther from'} {nom:g} than the baseline ({da:.4g} vs {db:.4g} away)"


@dataclass(frozen=True)
class Target(Body):
    """The paper a study replicates (`of` its source): a reproduction (from the authors' `code`) or a replication
    (rebuilt from the text), and the tolerance policy every fidelity and replication is held to: `tolerance`, or up
    to `max_tolerance` with a justification, in the metric's own unit. One per study, written before any run, so the
    policy cannot be fitted to the results. `paper_unit` percent says the paper prints its shares (coverage,
    accuracy) as percentages, for evidence that does not say."""
    KIND: ClassVar[str] = "target"
    LINKS: ClassVar[tuple] = (Link("of", ("source",), 1, 1),)
    kind: Literal["reproduction", "replication"]
    tolerance: float
    max_tolerance: float
    code: str = ""
    paper_unit: str = ""

    def check(self, rec, actor, links):
        if self.paper_unit not in ("", *UNITS):
            raise StudyError(f"paper_unit {self.paper_unit!r} is no unit", f"one of {', '.join(UNITS)}")
        if rec.find("run"):
            raise StudyError("the target is set before any run", "a tolerance set after the results could be fitted "
                             "to them", "start the replication in a new study")
        if rec.find("target"):
            raise StudyError("the study already has a target", "a study replicates one paper")
        if not 0 < self.tolerance <= self.max_tolerance:
            raise StudyError("a target needs 0 < tolerance <= max_tolerance")
        if self.kind == "reproduction" and not self.code.strip():
            raise StudyError("a reproduction runs the authors' code", "name it (code)")


@dataclass(frozen=True)
class Fidelity(Body):
    """Did we rebuild it? One baseline of the target paper reproduced: the paper's value of `metric` (the evidence
    it links) against ours (`arm` of the result it links: "model" or a baseline the run scored), passed when they
    agree within the tolerance. paper, ours and passed are computed on write (fidelity_fields), never typed."""
    KIND: ClassVar[str] = "fidelity"
    LINKS: ClassVar[tuple] = (Link("for", ("target",), 1, 1), Link("paper", ("evidence",), 1, 1),
                              Link("ours", ("result",), 1))
    metric: str
    arm: str
    tolerance: float
    paper: float                        # in the metric's unit
    ours: float
    passed: bool
    justification: str = ""
    paper_written: str = ""             # the paper's value as written, when its unit differs: "92.62%"
    group: str = ""                     # evaluate.by: the group (entity) of the result it reads, "" for all rows
    in_sample: bool = False             # ours was scored on the rows that trained it (as the paper's protocol does)

    def check(self, rec, actor, links):
        _typed(self, fidelity_fields(rec, links, self.metric, self.arm, self.tolerance, self.justification,
                                     self.group),
               "a fidelity's numbers come from its evidence and result")


@dataclass(frozen=True)
class Replication(Body):
    """Does it hold? One claim of the target paper: its value of `metric` (evidence) against ours with our interval
    (`arm` of a result), gated by fidelities of the paper's baselines on the same protocol. The verdict is
    replication_tier against the reference, the best gating baseline's paper value, and it is inconclusive (rebuild
    not faithful) unless every gating fidelity passed: a bad rebuild cannot fail the paper. For a target-valued
    metric (coverage90) the claim is a level: replicated within tolerance of the paper's value. Every number, the
    verdict and the reason are computed on write (replication_fields), never typed."""
    KIND: ClassVar[str] = "replication"
    LINKS: ClassVar[tuple] = (Link("for", ("target",), 1, 1), Link("paper", ("evidence",), 1, 1),
                              Link("ours", ("result",), 1), Link("gated_by", ("fidelity",), 1))
    metric: str
    arm: str
    tolerance: float
    paper: float
    ours: float
    lo: float
    hi: float
    reference: Optional[float]
    verdict: Literal[REPLICATION_VERDICTS]
    reason: str = ""
    justification: str = ""
    nominal: Optional[float] = None     # a target-valued metric's best value (coverage90: 0.9)
    vs_reference: str = ""              # a target-valued metric: closer to nominal than the reference, or n/a
    paper_written: str = ""             # the paper's value as written, when its unit differs: "90.33%"
    group: str = ""                     # evaluate.by: the group (entity) of the result it reads, "" for all rows
    in_sample: bool = False             # ours was scored on the rows that trained it (as the paper's protocol does)

    def check(self, rec, actor, links):
        _typed(self, replication_fields(rec, links, self.metric, self.arm, self.tolerance, self.justification,
                                        self.group),
               "a replication's numbers and verdict come from its evidence, result and fidelities")


SCHEMAS.update({c.KIND: c for c in (Target, Fidelity, Replication)})
KINDS = tuple(SCHEMAS)


# --- phase 7: reproduction --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Reproduction(Body):
    """A recorded run executed again from what the record holds only (openlab.study.reproduce): its code blobs, the
    data whose sha it recorded, the sealed spec at its `under` node and that spec's seed. `predictions` is the blob of
    the reproduction's predictions; `matched`, `max_abs_diff`, `differs` (what does not match, "" when it all does)
    and `metrics` ({metric: {run, reproduced, diff}}, the run's metrics scored again on the reproduction) are
    recomputed from the two predictions blobs on write, and `env_diff` from the run's environment and `env`, so none
    can be typed. A matched reproduction of every result a claim rests on makes the claim's level reproduced."""
    KIND: ClassVar[str] = "reproduction"
    LINKS: ClassVar[tuple] = (Link("of", ("run",), 1, 1),)
    run: str
    predictions: str
    matched: bool
    max_abs_diff: Optional[float]
    differs: str
    metrics: dict
    env: dict
    env_diff: dict
    sandbox: str
    seconds: float

    def check(self, rec, actor, links):
        from .reproduce import compare, env_diff
        run = links["of"][0]
        if self.run != run.id:
            raise StudyError("a reproduction names the run it links", f"{self.run} vs {run.id}")
        if not _SHA.fullmatch(self.predictions) or not run.payload["predictions"]:
            raise StudyError("a reproduction compares predictions blobs", "the run's and its own")
        want = {**compare(rec, run, self.predictions), "env_diff": env_diff(run.payload["env"], self.env)}
        typed = [f"{k}: typed {getattr(self, k)!r}, recomputed {v!r}" for k, v in want.items() if getattr(self, k) != v]
        if typed:
            raise StudyError("a reproduction's match is recomputed from the predictions, never typed", "; ".join(typed))


SCHEMAS.update({Reproduction.KIND: Reproduction})
KINDS = tuple(SCHEMAS)


# --- phase 7: declared trials -----------------------------------------------------------------------------------
def trials_used(rec: "Record") -> int:
    """The study's trials: its distinct research runs (by key) and the variants declared as tried elsewhere."""
    return (len({n.payload["key"] for n in rec.find("run", segment="research")})
            + sum(n.payload["n"] for n in rec.find("trial")))


@dataclass(frozen=True)
class Trial(Body):
    """`n` variants tried outside the runner (a sweep in a script), declared so they count like runs: against the
    trial budget and in every correction over the study's trials. Refused past the budget."""
    KIND: ClassVar[str] = "trial"
    n: int
    note: str

    def check(self, rec, actor, links):
        from .spec import sealed
        _said(self, "note")
        if self.n < 1:
            raise StudyError("a declaration counts at least one trial", f"n is {self.n}")
        spec = sealed(rec)
        budget = spec.budget.get("trials") if spec else None
        if not budget:
            raise StudyError("declared trials count against a sealed trial budget", "this study has none")
        used = trials_used(rec)
        if used + self.n > budget:
            raise StudyError(f"the trial budget is spent: {used} of {budget} used, {self.n} declared",
                             "every variant tried counts, so the number of tries stays honest",
                             "a person must amend budget.trials (a person-only field) first")


SCHEMAS.update({Trial.KIND: Trial})
KINDS = tuple(SCHEMAS)


# --- agents -------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Agent(Body):
    """An agent that wrote to this study, written by openlab before its first node, so the study names its agents
    (and their tree) without the workspace's control store. Every node's `session` is an agent id."""
    KIND: ClassVar[str] = "agent"
    ACTORS: ClassVar[tuple] = ("openlab",)
    id: str
    name: str
    role: str
    parent: str
    cli: str
    kind: str

    def check(self, rec, actor, links):
        if rec.find("agent", id=self.id):
            raise StudyError(f"agent {self.id} is already in the record", "an agent node is written once per study")


SCHEMAS.update({Agent.KIND: Agent})
KINDS = tuple(SCHEMAS)


# --- research from a prompt ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class Proposal(Body):
    """A research proposal as signed off (proposal.toml, canonical) and its sha: the question, the hypotheses with
    their falsifiers, the branches with a budget, a stop rule and a kill criterion each, the data, metrics and
    baselines (openlab.study.proposal). `tree` maps "root", each branch id and each hypothesis id to the question and
    hypothesis nodes made for it, linked `plans`. The first proposal is a person's approval; a later one amends the
    latest with a reason, and needs a person when it touches a `protected` field (budgets, stop rules, kill
    criteria, falsifiers, predictions, the primary metric, a removed branch or hypothesis)."""
    KIND: ClassVar[str] = "proposal"
    LINKS: ClassVar[tuple] = (Link("amends", ("proposal",), 0, 1), Link("plans", ("question", "hypothesis")))
    ACTORS: ClassVar[tuple] = ("agent", "person")
    proposal: dict
    sha: str
    tree: dict
    reason: str
    protected: list
    prev_sha: str = ""

    def check(self, rec, actor, links):
        import hashlib
        import json
        _sha(self.sha, "proposal sha")
        text = json.dumps(self.proposal, sort_keys=True, separators=(",", ":"), default=str)
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != self.sha:
            raise StudyError("a proposal's sha is the sha256 of its canonical text")
        if not self.reason.strip():
            raise StudyError("a proposal needs a reason", "why it is approved, or why it changes")
        prior, amends = rec.find("proposal"), links.get("amends", [])
        if prior and not amends:
            raise StudyError("the proposal is already approved", f"approved as {prior[0].id}",
                             "record the change as an amendment: openlab propose amend FOLDER --reason ...")
        if amends and amends[0].id != prior[-1].id:
            raise StudyError("an amendment amends the latest proposal", f"the latest is {prior[-1].id}")
        if amends and self.prev_sha != amends[0].payload["sha"]:
            raise StudyError("an amendment names the sha it amends (prev_sha)")
        if not prior and actor != "person":
            raise StudyError("a proposal is approved by a person", "an agent drafts it; a person signs it off",
                             "ask the person to run `openlab propose approve FOLDER --reason ...`")
        if self.protected and actor != "person":
            raise StudyError(f"{', '.join(self.protected)} can only be amended by a person",
                             "budgets, stop rules, kill criteria, falsifiers and the primary metric protect the plan",
                             "undo that part of the edit, or a person signs it: openlab propose amend FOLDER --person "
                             '--reason "..."')
        planned = {n.id for n in links.get("plans", [])}
        missing = sorted(set(self.tree.values()) - planned - {i for p in prior for i in p.payload["tree"].values()})
        if missing:
            raise StudyError("a proposal links `plans` to every node of its tree", f"not linked: {', '.join(missing)}")


SCHEMAS.update({Proposal.KIND: Proposal})
KINDS = tuple(SCHEMAS)

# --- the person key ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PersonKey(Body):
    """The person key this study pins: the public key (base64) and its fingerprint, signed by itself like every
    person node (openlab.study.personkey). The first one pins the key; a later one is a rotation: `rotates` names the
    pinned fingerprint and `endorsement` is the pinned key's signature over the new public key."""
    KIND: ClassVar[str] = "person_key"
    ACTORS: ClassVar[tuple] = ("person",)
    fingerprint: str
    public: str
    handle: str
    rotates: str = ""
    endorsement: str = ""

    def check(self, rec, actor, links):
        from .personkey import check_rotation, fingerprint
        try:
            fp = fingerprint(self.public)
        except Exception:                                           # noqa: BLE001
            raise StudyError("person_key.public is not a base64 Ed25519 key") from None
        if fp != self.fingerprint:
            raise StudyError("person_key.fingerprint is not its key's", f"{self.fingerprint} vs {fp}")
        pins = rec.find("person_key")
        if not pins:
            plan = rec.find("plan")
            sealed = plan[0].payload.get("person_key") or {} if plan else {}
            if not sealed:
                if self.rotates or self.endorsement:
                    raise StudyError("the first person_key pins a key; there is nothing to rotate from")
                return
            if self.fingerprint == sealed["fingerprint"] and not (self.rotates or self.endorsement):
                return
            if self.rotates != sealed["fingerprint"] or not check_rotation(sealed["fingerprint"], sealed["public"],
                                                                           self.public, self.endorsement):
                raise StudyError(f"this study was sealed under the person key {sealed['fingerprint']}",
                                 f"a first pin of another key ({self.fingerprint}) needs a rotation signed by it",
                                 "use the key the study was sealed under, or rotate from it (`openlab person rotate`)")
            return
        last = pins[-1].payload
        if self.fingerprint == last["fingerprint"]:
            raise StudyError(f"the study already pins {self.fingerprint}")
        if self.rotates != last["fingerprint"] or not check_rotation(last["fingerprint"], last["public"],
                                                                     self.public, self.endorsement):
            raise StudyError(f"this study pins the person key {last['fingerprint']}",
                             f"a new key ({self.fingerprint}) is accepted only with a rotation signed by the pinned "
                             "key", "run `openlab person rotate` with the old passphrase, or use the pinned key")


SCHEMAS.update({PersonKey.KIND: PersonKey})
KINDS = tuple(SCHEMAS)


# --- phase 3: the search's selection and lessons (openlab.study.search, openlab.study.lessons) ------------------------
@dataclass(frozen=True)
class Selection(Body):
    """The sealed selection rule applied: the run it picked among N candidates, the picked value and that value
    adjusted for being the best of N. Every field is derived from the record on write, so none can be typed."""
    KIND: ClassVar[str] = "selection"
    LINKS: ClassVar[tuple] = (Link("picks", ("run",), 1, 1), Link("under", ("plan", "amendment"), 1, 1))
    rule: str
    picked: str
    metric: str
    value: float
    adjusted: float
    se: float
    n: int
    candidates: list = field(default_factory=list)

    def check(self, rec, actor, links):
        from types import SimpleNamespace
        from .search import select
        from .spec import sealed
        _latest_spec_node(rec, links["under"][0], "a selection is made under the current sealed spec")
        want = select(SimpleNamespace(record=rec, spec=sealed(rec)))
        typed = {"rule": self.rule, "picked": self.picked, "metric": self.metric, "n": self.n}
        bad = [f"{k}: typed {v!r}, derived {want[k]!r}" for k, v in typed.items() if want[k] != v]
        bad += [f"{k}: typed {getattr(self, k)!r}, derived {want[k]!r}" for k in ("value", "adjusted", "se")
                if abs(getattr(self, k) - want[k]) > 1e-9 * max(1.0, abs(want[k]))]
        if links["picks"][0].id != self.picked:
            bad.append(f"picks links {links['picks'][0].id}, not {self.picked}")
        if bad:
            raise StudyError("the selection does not follow from the record", "; ".join(bad))


LESSON_DIRECTIONS = ("improves", "hurts", "no_gain", "fixes")


@dataclass(frozen=True)
class Lesson(Body):
    """A rule learned from two runs, resting on what the record shows about them. `basis` is a comparison of two of
    the study's results (improves, hurts, no_gain) or, for `fixes`, the run that fixed its parent. The direction must
    follow from it: improves needs the first side better with an interval that excludes 0, hurts the second, no_gain
    an interval inside ±margin (in the metric's units), fixes a parent whose audit was not clean or that had no
    result and a child that is valid. `tags` scope it for retrieval (openlab.study.lessons)."""
    KIND: ClassVar[str] = "lesson"
    LINKS: ClassVar[tuple] = (Link("basis", ("comparison", "run"), 1, 1),)
    rule: str
    direction: Literal["improves", "hurts", "no_gain", "fixes"]
    tags: list
    margin: Optional[float] = None
    shown: dict = field(default_factory=dict)       # what the basis shows, derived on write: {metric, diff, lo, hi}

    def check(self, rec, actor, links):
        from .lessons import shows
        if not self.rule.strip():
            raise StudyError("a lesson states its rule")
        if not self.tags:
            raise StudyError("a lesson carries tags", "they scope it, so the next study that shares one finds it")
        want = shows(rec, links["basis"][0], self.direction, self.margin)
        if want != self.shown:
            raise StudyError("the lesson's `shown` does not follow from its basis", f"derived {want}")


SCHEMAS.update({Selection.KIND: Selection, Lesson.KIND: Lesson})
KINDS = tuple(SCHEMAS)


# --- phase 4: module checks and critiques (openlab.study.preflight, openlab.study.critique) ----------------------------
@dataclass(frozen=True)
class ModuleCheck(Body):
    """The checks of one snapshot of the study's code (`code`: its files' shas): lint, import, smoke, determinism,
    each {name, ok, seconds, detail}. `ok` is derived: every check passed. A run of exactly this code is refused
    while its latest check is red."""
    KIND: ClassVar[str] = "module_check"
    code: dict
    checks: list
    ok: bool

    def check(self, rec, actor, links):
        if not self.checks:
            raise StudyError("a module check lists its checks")
        if self.ok != all(c.get("ok") for c in self.checks):
            raise StudyError("module_check.ok does not follow from its checks", "it is true exactly when every check "
                             "passed")


@dataclass(frozen=True)
class Critique(Body):
    """A reviewer's findings on a run, result, claim or paper: advice. `model` and `prompt_sha` say who wrote it,
    `findings` are [{where, text}]. Nothing reads a critique into a verdict."""
    KIND: ClassVar[str] = "critique"
    LINKS: ClassVar[tuple] = (Link("on", ("run", "result", "claim", "paper"), 1, 1),)
    model: str
    prompt_sha: str
    findings: list
    cost_usd: Optional[float] = None

    def check(self, rec, actor, links):
        if not self.model.strip():
            raise StudyError("a critique names the model that wrote it")
        if not self.findings:
            raise StudyError("a critique has at least one finding")


@dataclass(frozen=True)
class Finding(Body):
    """The answer to one finding of a critique: `checked` (by a module check, a run's audit, a comparison or a test
    that settled it) or `dismissed` (with the reason)."""
    KIND: ClassVar[str] = "finding"
    LINKS: ClassVar[tuple] = (Link("answers", ("critique",), 1, 1),
                              Link("by", ("module_check", "run", "comparison", "test"), 0, 1))
    index: int
    action: Literal["checked", "dismissed"]
    reason: str

    def check(self, rec, actor, links):
        c = links["answers"][0]
        if not 0 <= self.index < len(c.payload["findings"]):
            raise StudyError(f"{c.id} has no finding {self.index}", f"its findings are 0..{len(c.payload['findings']) - 1}")
        if self.action == "checked" and not links.get("by"):
            raise StudyError("a checked finding names the check that settled it", "by=<module_check, run, "
                             "comparison or test id>")
        if self.action == "dismissed" and not self.reason.strip():
            raise StudyError("a dismissed finding says why")


SCHEMAS.update({c.KIND: c for c in (ModuleCheck, Critique, Finding)})
KINDS = tuple(SCHEMAS)

#: kinds only a person writes: each is signed by the person key, and `verify` flags one written by anyone else
PERSON_ONLY = tuple(k for k, c in SCHEMAS.items() if tuple(c.ACTORS) == ("person",))


# --- validation ----------------------------------------------------------------------------------------------
def body(kind_or_body, payload: Optional[dict] = None) -> Body:
    """A typed body from a Body instance or (kind, payload); unknown kinds, missing or extra fields and
    wrongly typed values are refused."""
    if isinstance(kind_or_body, Body):
        b = kind_or_body
    else:
        cls = SCHEMAS.get(kind_or_body)
        if cls is None:
            raise StudyError(f"unknown node kind `{kind_or_body}`", f"kinds: {', '.join(KINDS)}",
                             "a new kind needs a schema in openlab/study/schema.py first")
        names = {f.name for f in dataclasses.fields(cls)}
        if isinstance(payload, dict) and "sig" in payload:          # a person node's signature (personkey.SIG)
            payload = {k: v for k, v in payload.items() if k != "sig"}
        extra = sorted(set(payload or {}) - names)
        if extra:
            raise StudyError(f"{cls.KIND} has no field {', '.join(extra)}", f"fields: {', '.join(sorted(names))}")
        try:
            b = cls(**(payload or {}))
        except TypeError as e:
            raise StudyError(f"{cls.KIND} is missing a field", str(e)) from None
    _typecheck(b)
    return b


def _typecheck(b: Body) -> None:
    hints = typing.get_type_hints(type(b))
    for f in dataclasses.fields(b):
        if not _is(getattr(b, f.name), hints[f.name]):
            raise StudyError(f"{b.KIND}.{f.name} has the wrong type", f"expected {hints[f.name]}, got "
                             f"{type(getattr(b, f.name)).__name__}")


def _is(v, t) -> bool:
    origin, args = typing.get_origin(t), typing.get_args(t)
    if origin is Literal:
        return v in args
    if origin is typing.Union:
        return any(_is(v, a) for a in args)
    if t is type(None):
        return v is None
    if t is float:
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t is int:
        return isinstance(v, int) and not isinstance(v, bool)
    return isinstance(v, origin or t)


def check(rec: "Record", b: Body, actor: str, links: list) -> None:
    """Every rule for writing `b`: actor, declared links (relation, target kinds, counts), then the invariants."""
    if actor not in b.ACTORS:
        raise StudyError(f"a {b.KIND} can only be written by {' or '.join(b.ACTORS)}", f"actor was {actor}")
    by_rel: dict = {}
    for rel, node in links:
        by_rel.setdefault(rel, []).append(node)
    rules = {l.rel: l for l in b.LINKS}
    for rel, nodes in by_rel.items():
        rule = rules.get(rel)
        if rule is None:
            raise StudyError(f"a {b.KIND} cannot link `{rel}`", f"its links: {', '.join(rules) or 'none'}")
        bad = [n.id for n in nodes if rule.to != ANY and n.kind not in rule.to]
        if bad:
            raise StudyError(f"{b.KIND} `{rel}` must point to {' or '.join(rule.to)}", f"got {', '.join(bad)}")
    for rule in b.LINKS:
        n = len(by_rel.get(rule.rel, []))
        if n < rule.min or (rule.max is not None and n > rule.max):
            raise StudyError(f"a {b.KIND} needs {rule.min}..{rule.max if rule.max is not None else 'n'} "
                             f"`{rule.rel}` links", f"got {n}")
    b.check(rec, actor, by_rel)
