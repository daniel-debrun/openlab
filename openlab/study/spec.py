"""`study.toml`: load and validate it, seal it into the record, and amend it with a recorded diff.

After the seal the effective spec is the plan node plus its amendments, replayed in order. Editing the file
without amending is caught by `check`; amendments that touch `PERSON_FIELDS` need a person.
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:                                  # Python 3.10
    import tomli as tomllib

from .metrics import KINDS, STUDY_FILE, declare
from .notify import NOTIFY
from .schedule import parse_cron
from .durations import parse as parse_duration
from .errors import StudyError
from .record import Node, Record, canonical
from .schema import FIELD_TYPES, FREE_TYPES, INDEXES, UNITS

PERSON_FIELDS = ("split", "budget", "sandbox", "audit", "selection", "forecast.gate", "evaluate.metrics[0]", "data.*.known_at",
                 "review.coverage_floor", "review.comparable_on", "inference", "study.fresh",
                 "metrics")                                  # amend needs a person; metrics: the study's own
REVIEW_FIELDS = ("review.coverage_floor", "review.comparable_on")  # guarded even while a replication is planned

MODES = {"temporal": ("walk_forward", "time"), "static": ("kfold", "stratified", "group", "insample"), "review": ()}
PREDICTIONS = ("point", "prob", "gaussian", "quantiles", "ensemble", "interval", "frame")

# (test, what the value must be) for each kind of value
_STR =(lambda v: isinstance(v, str) and v.strip() != "", "a non-empty string")
_STRS = (lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v), "a list of strings")
_PAIR = (lambda v: isinstance(v, list) and len(v) == 2, "a [start, end] pair")
_DICT = (lambda v: isinstance(v, dict), "a table")
_BOOL = (lambda v: isinstance(v, bool), "true or false")
_NUM = (lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0, "a positive number")
_FRAC = (lambda v: isinstance(v, float) and 0 < v < 1, "a fraction between 0 and 1")


def _int(lo: int):
    return lambda v: type(v) is int and v >= lo, f"an integer >= {lo}"


def _one_of(*opts):
    return lambda v: v in opts, "one of " + ", ".join(opts)


def _distinct(options: tuple = ()):
    """A non-empty list of distinct strings (from `options` when given)."""
    return (lambda v: _STRS[0](v) and len(v) > 0 and len(set(v)) == len(v) and (not options or set(v) <= set(options)),
            "a non-empty list of distinct " + (", ".join(options) if options else "strings"))


_CRITERIA = (lambda v: isinstance(v, list) and len(v) > 0 and all(
    isinstance(c, dict) and set(c) == {"id", "text"} and all(_STR[0](x) for x in c.values()) for c in v),
    'a non-empty list of criteria, e.g. [{id = "I1", text = "..."}]')
_FLOOR = (lambda v: isinstance(v, dict) and set(v) == {"queries_per_source", "hops"}
          and _int(1)[0](v["queries_per_source"]) and _int(0)[0](v["hops"]),
          "{queries_per_source = <integer >= 1>, hops = <integer >= 0>}")


def _field(v) -> bool:
    """An extraction field: its type, or {type, free} where free (text and list only) skips the quote check, or
    {type = "number", unit} where unit (percent or fraction) is how the paper writes it."""
    if isinstance(v, str):
        return v in FIELD_TYPES
    return isinstance(v, dict) and set(v) <= {"type", "free", "unit"} and v.get("type") in FIELD_TYPES and \
        ("free" not in v or (isinstance(v["free"], bool) and v["type"] in FREE_TYPES)) and \
        ("unit" not in v or (v["unit"] in UNITS and v["type"] == "number"))


_EXTRACT = (lambda v: isinstance(v, dict) and len(v) > 0 and all(
    re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k) and _field(f) for k, f in v.items()),
    f'a table of field = one of {", ".join(FIELD_TYPES)}, or {{type = "text", free = true}} for a text or list '
    'value not checked against its quote, or {type = "number", unit = "percent"} for a number the paper writes in '
    "percent")


def _parses(parse, what: str):
    """A string that `parse` reads."""
    def ok(v) -> bool:
        try:
            return isinstance(v, str) and parse(v) is not None
        except (ValueError, StudyError):
            return False
    return ok, what


# The allowed keys of each section and what each value must be. `data.*` is one dataset table.
KEYS = {
    "study": {"name": _STR, "question": _STR, "mode": _one_of(*MODES), "continues": _STR, "fresh": _STR},
    "data.*": {"provider": _STR, "endpoint": _STR, "args": _DICT, "snapshot": _STR, "dataset": _STR, "file": _STR,
               "recipe": _STR, "time": _STR, "key": _STRS, "known_at": _STR, "vintages": _PAIR, "licence": _STR},
    "target": {"horizon": _STR, "column": _STR},
    "split": {"scheme": _one_of(*(s for v in MODES.values() for s in v)), "decisions": _STR, "refit": _STR,
              "window": _STR, "research": _PAIR, "holdout": _PAIR, "embargo": _STR, "fraction": _FRAC,
              "seed": _int(0), "by": _STR, "k": _int(2), "burn_in": _STR, "reseed": (lambda v: isinstance(v, bool), "true or false"),
              "train_from": (lambda v: hasattr(v, "year") or isinstance(v, str), "a date"),
              "labels": _one_of("person"), "labels_key": _STR},
    "evaluate": {"metrics": (lambda v: _STRS[0](v) and len(v) > 0, "a non-empty list of metric names"),
                 "prediction": _one_of(*PREDICTIONS), "baselines": _STRS, "interval": _STR,
                 "alpha": (lambda v: isinstance(v, float) and 0 < v < 1, "between 0 and 1"),
                 # the central interval's level coverage and interval_width read, and a lower/upper prediction is at
                 "level": (lambda v: isinstance(v, float) and 0 < v < 1, "between 0 and 1, e.g. 0.9"),
                 "policy_metrics": _STRS, "policy_baselines": _STRS, "block": _int(1),
                 "by": _one_of("key")},
    "budget": {"trials": _int(1), "holdout_looks": _int(0)},
    # the study's own metrics, one table each, the function in metrics.py; code_sha is metrics.py's sha256, which
    # openlab writes (so the seal holds the code, and an edit of it is an amendment)
    "metrics.*": {"better": _one_of("lower", "higher"), "reads": _distinct(KINDS), "unit": _one_of("", "fraction"),
                  "description": _STR, "code_sha": _STR},
    # threads: what the model's libraries may use (POLARS/OMP/OPENBLAS/MKL); 1 by default, so a run's numbers do not
    # depend on the machine's core count. A higher value is recorded in the run's env.
    "sandbox": {"network": _BOOL, "timeout": _NUM, "threads": _int(1), "gpus": _int(0), "cores": _int(1),
                "memory_gb": _NUM},
    "audit": {"n": _int(0)},
    "schedule": {"cron": _parses(parse_cron, "a five-field cron expression (minute hour day month weekday)"),
                 "refresh": _BOOL},
    "forecast": {"model": (lambda v: v == "latest" or _int(1)[0](v), '"latest" or a model version (an integer >= 1)'),
                 "gate": _STRS, "expires": _parses(parse_duration, 'a duration such as "1d" or "6h"')},
    # the review protocol: `n` hits per query on each index; a citation hop follows references and citing works
    "review": {"question": _STR, "scope": _STR, "sources": _distinct(INDEXES), "queries": _distinct(),
               "since": (lambda v: isinstance(v, str) and bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", v)), "a date"),
               "include": _CRITERIA, "exclude": _CRITERIA, "citation_hops": _int(0), "n": _int(1),
               "coverage_floor": _FLOOR,
               # extraction: the fields an evidence node may carry; coordinates name the protocol among them, and
               # two results are compared only if they agree on every comparable_on coordinate
               "extract": _EXTRACT, "coordinates": _distinct(), "comparable_on": _distinct()},
    # the error budget tests spend (alpha-investing): initial wealth and the payout of a rejection
    "inference": {"alpha": _FRAC, "wealth": _NUM, "payout": _NUM},
    # continuous research: when `openlab study monitor` runs, and where its notification goes besides the outbox
    "monitor": {"cron": _parses(parse_cron, "a five-field cron expression (minute hour day month weekday)"),
                "notify": _one_of(*NOTIFY)},
    "mirror": {"wandb": _BOOL},
    "search": {"minutes": _NUM, "explore_until": _NUM, "consolidate": _NUM, "c_hi": _NUM, "c_lo": _NUM, "w_lo": _NUM,
               "cost_weight": _NUM, "stagnation_branch": _int(1), "stagnation_global": _int(1)},
    "selection": {"rule": _one_of("best_val", "seed_robust"), "k": _int(1)},
}
FREE = ()                                                    # later phases: any keys
SECTIONS = ("study", "data", "target", "split", "evaluate", "budget", "sandbox", "schedule", "forecast", "review",
            "claims", "audit", "inference", "monitor", "metrics", "mirror", "search", "selection")
_EXPERIMENT = ("split.scheme", "evaluate.metrics", "budget.trials")
#: per mode, what study.toml must have beyond study.name, question and mode, and (ONLY) the sections it may have:
#: a review runs no experiment, so it has [study] and [review] and none of the experiment's sections
REQUIRED = {"temporal": _EXPERIMENT, "static": _EXPERIMENT, "review": ("review.question",)}
ONLY = {"review": ("study", "review", "claims", "monitor", "mirror")}
DEFAULTS = {"budget": {"holdout_looks": 1}, "forecast": {"model": "latest", "gate": [], "expires": "1d"},
            "monitor": {"notify": "outbox"}}
REVIEW_REQUIRED = ("question", "sources", "queries", "include", "exclude", "coverage_floor")
REVIEW_DEFAULTS = {"scope": "", "since": "", "citation_hops": 1, "n": 25, "extract": {}, "coordinates": [],
                   "comparable_on": []}
SOURCES = ("provider", "snapshot", "file")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    provider: str = ""
    endpoint: str = ""
    args: dict = field(default_factory=dict)
    snapshot: str = ""
    dataset: str = ""
    file: str = ""
    recipe: str = ""
    time: str = ""
    key: tuple = ()
    known_at: str = ""
    vintages: tuple = ()
    licence: str = ""


@dataclass(frozen=True)
class Spec:
    raw: dict
    folder: Path
    name: str
    question: str
    mode: str
    data: dict
    target: dict
    split: dict
    evaluate: dict
    budget: dict
    sandbox: dict
    schedule: dict
    review: dict
    claims: list
    audit: dict
    forecast: dict
    inference: dict
    monitor: dict

    def sha(self) -> str:
        return hashlib.sha256(canonical(self.raw).encode("utf-8")).hexdigest()


def _bad(key: str, meaning: str, fix: str = "") -> StudyError:
    return StudyError(f"study.toml: `{key}` {meaning}", fix=fix or "fix study.toml")


def _keys(table: dict, allowed: dict, where: str) -> None:
    for k, v in table.items():
        if k not in allowed:
            near = difflib.get_close_matches(k, allowed, n=1)
            raise _bad(f"{where}.{k}", "is not a known key",
                       f"did you mean `{where}.{near[0]}`?" if near else f"known keys: {', '.join(allowed)}")
        ok, desc = allowed[k]
        if not ok(v):
            raise _bad(f"{where}.{k}", f"must be {desc}")


def _validate(raw: dict) -> None:
    for sec, body in raw.items():
        if sec not in SECTIONS:
            near = difflib.get_close_matches(sec, SECTIONS, n=1)
            raise _bad(sec, "is not a known section", f"did you mean `{near[0]}`?" if near else
                       f"sections: {', '.join(SECTIONS)}")
        if sec == "claims":
            if not isinstance(body, list):
                raise _bad(sec, "must be a list")
        elif not isinstance(body, dict):
            raise _bad(sec, "must be a table")
        elif sec == "data":
            for name, ds in body.items():
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or not isinstance(ds, dict):
                    raise _bad(f"data.{name}", "must be a table named like an identifier (letters, digits, _)")
                _keys(ds, KEYS["data.*"], f"data.{name}")
                if sum(bool(ds.get(s)) for s in SOURCES) != 1:
                    raise _bad(f"data.{name}", "needs exactly one of provider, snapshot or file")
                if ds.get("file") and not ds.get("recipe"):
                    raise _bad(f"data.{name}.recipe", "is required for a file",
                               "add the script that makes the file: recipe = \"recipes/<script>.py\"")
        elif sec == "metrics":
            for name, m in body.items():
                if not name.isidentifier() or not isinstance(m, dict):
                    raise _bad(f"metrics.{name}", "must be a table named like a Python function")
                _keys(m, KEYS["metrics.*"], f"metrics.{name}")
                if "better" not in m:
                    raise _bad(f"metrics.{name}.better", "is required", 'better = "lower" or "higher"')
        elif sec not in FREE:
            _keys(body, KEYS[sec], sec)
    for key in ("study.name", "study.question", "study.mode"):
        if key.split(".")[1] not in raw.get("study", {}):
            raise _bad(key, "is required")
    mode = raw["study"]["mode"]
    extra = [s for s in raw if s not in ONLY.get(mode, SECTIONS)]
    if extra:
        raise _bad(extra[0], f"has no place in a {mode} study", f"a {mode} study has only "
                   + ", ".join(f"[{s}]" for s in ONLY[mode]) + "; run experiments in their own study")
    for key in REQUIRED[mode]:
        sec, k = key.split(".")
        if k not in raw.get(sec, {}):
            raise _bad(key, "is required")
    if MODES[mode] and raw["split"]["scheme"] not in MODES[mode]:
        raise _bad("split.scheme", f"must be one of {', '.join(MODES[mode])} in {mode} mode")
    if "review" in raw:
        _review(raw["review"])
    if "inference" in raw:
        _inference(raw["inference"])


def _inference(raw: dict) -> dict:
    """[inference] with its defaults: alpha 0.05, wealth alpha (1 - alpha) and payout alpha. Foster and Stine's
    alpha-investing controls mFDR at alpha when the wealth and the payout are at most alpha."""
    a = raw.get("alpha", 0.05)
    out = {"alpha": a, "wealth": a * (1 - a), "payout": a, **raw}
    big = [k for k in ("wealth", "payout") if out[k] > a]
    if big:
        raise _bad(f"inference.{big[0]}", f"is more than inference.alpha ({a})", "the error budget controls false "
                   "discoveries at alpha only when the wealth and the payout are at most alpha")
    return out


def _review(review: dict) -> None:
    for k in REVIEW_REQUIRED:
        if k not in review:
            raise _bad(f"review.{k}", "is required")
    ids = [c["id"] for c in review["include"] + review["exclude"]]
    if len(set(ids)) != len(ids):
        raise _bad("review.exclude", "repeats a criterion id: include and exclude ids must all differ")
    if review["coverage_floor"]["hops"] > review.get("citation_hops", REVIEW_DEFAULTS["citation_hops"]):
        raise _bad("review.coverage_floor", "asks for more hops than review.citation_hops follows",
                   "the search could never be sufficient: lower the floor or raise citation_hops")
    for key, within, of in (("coordinates", review.get("extract", {}), "review.extract"),
                            ("comparable_on", review.get("coordinates", []), "review.coordinates")):
        stray = [f for f in review.get(key, []) if f not in within]
        if stray:
            raise _bad(f"review.{key}", f"names {', '.join(stray)}, not in {of}")


def build(raw: dict, folder: Path) -> Spec:
    """A validated Spec from a canonical raw dict (the form a plan stores and a run's child receives)."""
    _validate(raw)
    allowed = ONLY.get(raw["study"]["mode"], SECTIONS)
    sec = {s: {**(DEFAULTS.get(s, {}) if s in allowed else {}), **raw.get(s, {})} for s in KEYS
           if s not in ("data.*", "metrics.*", "inference")}
    declare(raw.get("metrics", {}))
    data = {name: DatasetSpec(name=name, **{k: tuple(v) if k in ("key", "vintages") else v for k, v in ds.items()})
            for name, ds in raw.get("data", {}).items()}
    return Spec(raw=raw, folder=Path(folder), name=raw["study"]["name"], question=raw["study"]["question"],
                mode=raw["study"]["mode"], data=data, target=sec["target"], split=sec["split"],
                evaluate=sec["evaluate"], budget=sec["budget"], sandbox=sec["sandbox"],
                schedule=sec["schedule"], claims=raw.get("claims", []),
                review={**REVIEW_DEFAULTS, **raw["review"]} if "review" in raw else {},
                audit=sec["audit"], forecast=sec["forecast"],
                inference=_inference(raw.get("inference", {})) if "inference" in allowed else {},
                monitor=sec["monitor"])


def load(folder: Path) -> Spec:
    """Parse and validate `<folder>/study.toml`. Dates are kept as ISO strings, the form the seal stores."""
    path = Path(folder) / "study.toml"
    try:
        parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise StudyError(f"no study.toml in {folder}", fix="create one with `openlab study new <folder>`") from None
    except tomllib.TOMLDecodeError as e:
        raise StudyError("study.toml does not parse", str(e)) from None
    return build(_coded(json.loads(canonical(parsed)), Path(folder)), folder)


def _coded(raw: dict, folder: Path) -> dict:
    """raw with each declared metric's code_sha: the sha256 of the study's metrics.py, as it is now."""
    decls = raw.get("metrics")
    if not isinstance(decls, dict) or not decls:
        return raw
    if any(isinstance(m, dict) and "code_sha" in m for m in decls.values()):
        raise _bad("metrics.*.code_sha", "is written by openlab", "remove it: openlab hashes metrics.py itself")
    path = folder / STUDY_FILE
    if not path.is_file():
        raise _bad("metrics", f"declares {', '.join(decls)}, and the study has no {STUDY_FILE}",
                   f"define each as a function in {STUDY_FILE}: def NAME(y, pred) -> float")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return {**raw, "metrics": {n: {**m, "code_sha": sha} if isinstance(m, dict) else m for n, m in decls.items()}}


def _tokens(path: str) -> list:
    return re.findall(r"\[\d+\]|[^.\[\]]+", path)


def _pick(v, toks: list, path: str = "") -> dict:
    """{concrete dotted path: value} for a field pattern such as `data.*.known_at` or `evaluate.metrics[0]`."""
    if not toks:
        return {path: v}
    t, rest, out = toks[0], toks[1:], {}
    if isinstance(v, dict):
        for k in (v if t == "*" else [t] if t in v else []):
            out.update(_pick(v[k], rest, f"{path}.{k}" if path else k))
    elif isinstance(v, list) and t.startswith("[") and int(t[1:-1]) < len(v):
        out.update(_pick(v[int(t[1:-1])], rest, path + t))
    return out


def _diff(old, new, path: str = "") -> dict:
    """{dotted path: {"old", "new"}} down through tables; None is absent (TOML has no null)."""
    if isinstance(old, dict) and isinstance(new, dict) and not any("." in k for k in {*old, *new}):
        out = {}
        for k in sorted({*old, *new}):
            out.update(_diff(old.get(k), new.get(k), f"{path}.{k}" if path else k))
        return out
    return {} if old == new else {path: {"old": old, "new": new}}


def _apply(raw: dict, changes: dict) -> dict:
    raw = copy.deepcopy(raw)
    for path, ch in changes.items():
        *parents, leaf = path.split(".")
        node = raw
        for k in parents:
            node = node.setdefault(k, {})
        if ch["new"] is None:
            node.pop(leaf, None)
        else:
            node[leaf] = ch["new"]
    return raw


def _plan(rec: Record) -> Node | None:
    plans = rec.find("plan")
    return plans[0] if plans else None


def sealed(rec: Record, at: str = "") -> Spec | None:
    """The effective spec: the plan node with every amendment replayed (up to the plan or amendment `at`), or None
    before the seal."""
    plan = _plan(rec)
    if plan is None:
        return None
    raw, sha = plan.payload["spec"], plan.payload["sha"]
    nodes = [plan] + rec.find("amendment")
    if at:
        upto = [i for i, n in enumerate(nodes) if n.id == at]
        if not upto:
            raise StudyError(f"{at} is not the plan or an amendment of this study")
        nodes = nodes[:upto[0] + 1]
    for node in nodes:
        if node.kind == "amendment":
            raw, sha = _apply(raw, node.payload["changes"]), node.payload["sha"]
        if hashlib.sha256(canonical(raw).encode("utf-8")).hexdigest() != sha:
            raise StudyError(f"the sealed spec does not replay at {node.id}", "the record is inconsistent",
                             "run `openlab study verify`")
    return build(raw, rec.folder)


def seal(folder: Path, rec: Record, actor: str) -> Node:
    """Seal study.toml as the plan; idempotent (an existing plan node is returned as is)."""
    plan = _plan(rec)
    if plan is not None:
        return plan
    spec = load(folder)
    _keep_metrics(folder, rec, spec.raw)
    return rec.add("plan", {"spec": spec.raw, "sha": spec.sha(), "person_key": _sealed_key(folder, rec)}, actor=actor,
                   once=True)


def _sealed_key(folder: Path, rec: Record) -> dict:
    """{fingerprint, public} of the person key a study is sealed under: its own pin, else the workspace's (a pin
    whose stores disagree is not taken); {} when neither pins one."""
    pins = rec.find("person_key")
    if pins:
        return {"fingerprint": pins[-1].payload["fingerprint"], "public": pins[-1].payload["public"]}
    try:
        from ..agents.person import pin_state, pinned_public
        state = pin_state(folder)
        public = pinned_public(folder) if state["pin"] and not state["conflict"] else None
    except Exception:                                               # noqa: BLE001 -- a study without the agents package
        return {}
    if not public:
        return {}
    from .personkey import fingerprint
    return {"fingerprint": state["pin"], "public": public} if fingerprint(public) == state["pin"] else {}


def _keep_metrics(folder: Path, rec: Record, raw: dict) -> None:
    """metrics.py into the record's content store, under the sha the spec seals (code_sha): what scores the study's
    own metrics is read from there, never from the folder."""
    if raw.get("metrics"):
        rec.put_blob((Path(folder) / STUDY_FILE).read_bytes())


def check(folder: Path, rec: Record) -> Spec:
    """The effective spec, if study.toml still matches it; else a StudyError naming the keys that differ."""
    eff = sealed(rec)
    if eff is None:
        raise StudyError("the study is not sealed yet", fix="the first run seals study.toml")
    diff = _diff(eff.raw, load(folder).raw)
    if diff:
        raise StudyError("study.toml was edited after the seal", "changed: " + ", ".join(diff),
                         "undo the edit, or record it with `openlab study amend <folder> --reason ...`")
    return eff


def _planning(rec: Record) -> bool:
    """A replication whose experiment has not run yet. Its seal came early, to pin the target paper, and its
    pre-registration is the target's tolerance policy; the agent transcribes the paper's protocol into the experiment
    until the first run, from which the person fields are guarded as in any study."""
    return bool(rec.find("target")) and not rec.find("run") and not rec.find("look")


def amend(folder: Path, rec: Record, reason: str, *, person: bool, actor: str) -> Node:
    """Record the edit of study.toml since the effective spec as an amendment node, with the reason."""
    eff = sealed(rec)
    if eff is None:
        raise StudyError("nothing to amend: the study is not sealed", fix="edit study.toml freely until the first run")
    if not reason.strip():
        raise StudyError("an amendment needs a reason", fix="say why the plan changes")
    cur = load(folder)
    diff = _diff(eff.raw, cur.raw)
    guarded = REVIEW_FIELDS if _planning(rec) else PERSON_FIELDS
    touched = [f for f in guarded if _pick(eff.raw, _tokens(f)) != _pick(cur.raw, _tokens(f))]
    if not diff:
        raise StudyError("study.toml matches the sealed spec", "there is nothing to amend")
    if touched and not person:
        raise StudyError(f"{', '.join(touched)} can only be amended by a person",
                         f"changed: {', '.join(diff)}; split, budget, sandbox, the primary metric and known_at "
                         "protect the result",
                         "undo that part of the edit, or a person signs it: "
                         'openlab study amend STUDY --person --reason "..."')
    prev = rec.last("amendment") or _plan(rec)
    _keep_metrics(folder, rec, cur.raw)
    return rec.add("amendment", {"reason": reason, "changes": diff, "person_fields": touched, "person": person,
                                 "sha": cur.sha(), "prev_sha": eff.sha()},
                   actor=actor, links=[("amends", prev.id)])
