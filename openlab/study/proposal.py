"""Research from a prompt: a proposal a person signs off, a sealed plan, and its execution by a tree of agents.

`proposal.toml` in the study folder is the proposal, edited by the researcher or an agent and checked by openlab;
`proposal.md` is rendered from it for reading. It holds the question, a short background, the hypotheses (each with
its prediction and the falsifier: the result that would refute it), the branches (each with the question or
hypotheses it serves, a method sketch, its data needs, a budget of $, minutes and trials, a stop rule, a kill
criterion and the role of the agent that does it), the data, the metrics (the primary first), the baselines, an
optional literature-review protocol, the total budget and the risks.

* `problems` lists every refusal (error, meaning, fix); `lint` adds the warnings and renders proposal.md.
* `approve` (a person's act) writes the `proposal` node with the canonical sha, writes study.toml from it (mode,
  [evaluate], [budget] trials = the sum of the branches', [review] for a literature branch), makes the question
  tree (the root question, one question per branch, the hypotheses under the branch that serves them, each with its
  falsifier as what decides it) and seals the study.
* After approval an edit of proposal.toml is refused (`drift`) until it is recorded with `amend`, which needs a
  person when it touches a protected field.
* `brief` is the lead agent's execution brief; `status` is the progress per branch: agents and their state, $,
  minutes, trials, the hypotheses tested, killed and untouched, dead ends, and the stop rule.

A branch is a frontier item (kind `branch`, verb `delegate`) while it is open; `stop` records a branch stopped by
its stop rule or killed by its kill criterion.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:                                  # Python 3.10
    import tomli as tomllib

from . import schema
from . import spec as _spec
from .errors import StudyError
from .record import Node, canonical

FILE, MD = "proposal.toml", "proposal.md"
ROLES = ("screener", "replicator", "experimenter", "reviewer")
KIND_OF = {"screener": "literature", "replicator": "experiment", "experimenter": "experiment", "reviewer": "review"}
MODES = tuple(_spec.MODES)
_ID = re.compile(r"[a-z][a-z0-9_-]{0,23}")
TOP = ("title", "question", "background", "mode", "metrics", "baselines", "risks", "budget", "hypotheses",
       "branches", "data", "review", "continues", "fresh")
HYPOTHESIS = ("id", "statement", "prediction", "falsifier", "mechanism", "tags")
BRANCH = ("id", "question", "serves", "role", "method", "data_needs", "budget", "stop", "kill")
BUDGET = ("usd", "minutes", "trials", "holdout_looks")
DATA = ("name", "source", "recipe", "licence")
REVIEW = ("question", "sources", "queries", "include", "exclude", "coverage_floor", "citation_hops", "n", "since",
          "extract")
#: amendments touching these (dotted, with * for an id) need a person; so does removing a branch or hypothesis
PROTECTED = ("budget", "budget.*", "branches.*.budget", "branches.*.budget.*", "branches.*.stop", "branches.*.kill",
             "hypotheses.*.falsifier", "hypotheses.*.prediction", "review.coverage_floor", "review.coverage_floor.*",
             "fresh")
PLANNER_USD, PLANNER_MINUTES = 0.5, 15


def _p(where: str, error: str, meaning: str = "", fix: str = "") -> dict:
    return {"where": where, "error": f"proposal.toml: `{where}` {error}" if where else f"proposal.toml: {error}",
            "meaning": meaning, "fix": fix or "edit proposal.toml"}


def _text(v) -> bool:
    return isinstance(v, str) and v.strip() != ""


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0


def _count(v) -> bool:
    return type(v) is int and v >= 0


def _strs(v) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


def path(folder) -> Path:
    return Path(folder) / FILE


def load(folder) -> dict:
    """proposal.toml parsed, in canonical form (what the sha covers)."""
    p = path(folder)
    try:
        parsed = tomllib.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise StudyError(f"no {FILE} in {folder}", "a proposal is drafted first",
                         'start one with `openlab propose new FOLDER "<your question>"`') from None
    except tomllib.TOMLDecodeError as e:
        raise StudyError(f"{FILE} does not parse", str(e), "fix the TOML syntax") from None
    return json.loads(canonical(parsed))


def sha(raw: dict) -> str:
    return hashlib.sha256(canonical(raw).encode("utf-8")).hexdigest()


def cap(folder) -> dict:
    """The workspace cap on one proposal's total budget ([agents] cap_usd, cap_minutes in openlab.toml)."""
    from ..agents.store import config, root_of
    cfg = config(root_of(Path(folder)))
    return {"usd": float(cfg["cap_usd"]), "minutes": float(cfg["cap_minutes"])}


def _unknown(table: dict, allowed: tuple, where: str) -> list:
    import difflib
    out = []
    for k in table:
        if k not in allowed:
            near = difflib.get_close_matches(k, allowed, n=1)
            out.append(_p(f"{where}.{k}" if where else k, "is not a known key",
                          f"known keys: {', '.join(allowed)}", f"did you mean `{near[0]}`?" if near else ""))
    return out


def _budget(b, where: str, trials_needed: bool) -> list:
    if not isinstance(b, dict):
        return [_p(where, "is required: a table {usd, minutes, trials}", "every branch and the whole proposal have a "
                   "budget, so no agent can spend open-endedly", f"add {where} = {{usd = 1.0, minutes = 30, trials = 5}}")]
    out = _unknown(b, BUDGET, where)
    for k in ("usd", "minutes"):
        if not _num(b.get(k)):
            out.append(_p(f"{where}.{k}", "must be a positive number", "dollars of agent spend" if k == "usd" else
                          "wall minutes of agent time"))
    if "trials" not in b or not _count(b.get("trials")) or (trials_needed and not b.get("trials")):
        out.append(_p(f"{where}.trials", "must be an integer" + (" >= 1" if trials_needed else " >= 0"),
                      "the runs this branch may spend from the study's trial budget" if where != "budget" else
                      "the study's trial budget (study.toml budget.trials is the sum of the branches')"))
    if "holdout_looks" in b and (where != "budget" or not _count(b["holdout_looks"])):
        out.append(_p(f"{where}.holdout_looks", "is the total's only", "a whole number of person's holdout looks"))
    return out


def problems(raw: dict, limit: dict | None = None) -> list:
    """Every refusal of the proposal, each {where, error, meaning, fix}; empty when it may be approved."""
    out = _unknown(raw, TOP, "")
    for k in ("title", "question", "background"):
        if not _text(raw.get(k)):
            out.append(_p(k, "is required", "the person signs off on what it says",
                          "the question in one sentence" if k == "question" else ""))
    mode = raw.get("mode")
    if mode not in MODES:
        out.append(_p("mode", f"must be one of {', '.join(MODES)}", "temporal or static for an experiment, review for "
                      "a literature review only"))
    for k in ("metrics", "baselines", "risks"):
        if k in raw and not _strs(raw[k]):
            out.append(_p(k, "must be a list of strings"))
    for k, what in (("continues", 'the study folder whose trial ledger this one continues, e.g. "../demo"'),
                    ("fresh", "why this is a new question on data another study already uses")):
        if k in raw and not _text(raw[k]):
            out.append(_p(k, "must be a non-empty string", what))
    hyps = raw.get("hypotheses", [])
    hyps = hyps if isinstance(hyps, list) else []
    if not isinstance(raw.get("hypotheses", []), list):
        out.append(_p("hypotheses", "must be [[hypotheses]] tables"))
    seen: set = set()
    for i, h in enumerate(hyps):
        where = f"hypotheses.{h.get('id') if isinstance(h, dict) and _text(h.get('id')) else i}"
        if not isinstance(h, dict):
            out.append(_p(where, "must be a table"))
            continue
        out += _unknown(h, HYPOTHESIS, where)
        if not (isinstance(h.get("id"), str) and _ID.fullmatch(h["id"])):
            out.append(_p(f"{where}.id", "must be a short id: a lower-case letter, then letters, digits, - or _"))
        elif h["id"] in seen:
            out.append(_p(f"{where}.id", "repeats an id"))
        seen.add(h.get("id"))
        for k, what in (("statement", "the claim"), ("prediction", "what you expect to observe if it holds")):
            if not _text(h.get(k)):
                out.append(_p(f"{where}.{k}", "is required", what))
        if not _text(h.get("falsifier")):
            out.append(_p(f"{where}.falsifier", "is required: every hypothesis has a falsifier",
                          "the test or result that would refute it; a hypothesis nothing could refute is not tested",
                          'e.g. falsifier = "a paired test against majority with p >= 0.05 on the research split"'))
        if "tags" in h and not _strs(h["tags"]):
            out.append(_p(f"{where}.tags", "must be a list of strings"))
    hids = {h.get("id") for h in hyps if isinstance(h, dict)}
    branches = raw.get("branches")
    if not isinstance(branches, list) or not branches:
        out.append(_p("branches", "is required: at least one [[branches]] table",
                      "a branch is the unit an agent is assigned: a question, a budget, a stop rule and a kill "
                      "criterion"))
        branches = []
    bseen: set = set()
    kinds = set()
    for i, b in enumerate(branches):
        where = f"branches.{b.get('id') if isinstance(b, dict) and _text(b.get('id')) else i}"
        if not isinstance(b, dict):
            out.append(_p(where, "must be a table"))
            continue
        out += _unknown(b, BRANCH, where)
        if not (isinstance(b.get("id"), str) and _ID.fullmatch(b["id"])):
            out.append(_p(f"{where}.id", "must be a short id: a lower-case letter, then letters, digits, - or _"))
        elif b["id"] in bseen:
            out.append(_p(f"{where}.id", "repeats an id"))
        bseen.add(b.get("id"))
        role = b.get("role")
        if role not in ROLES:
            out.append(_p(f"{where}.role", f"must be one of {', '.join(ROLES)}", "the openlab role agent that does it"))
        kind = KIND_OF.get(role, "")
        kinds.add(kind)
        for k, what in (("question", "the sub-question this branch answers"), ("method", "a sketch of what is done")):
            if not _text(b.get(k)):
                out.append(_p(f"{where}.{k}", "is required", what))
        if not _text(b.get("stop")):
            out.append(_p(f"{where}.stop", "is required: every branch has a stop rule",
                          "when to stop exploring, so no agent spends hours on a dead end",
                          'e.g. stop = "3 variants in a row without a gain of 0.005 auc"'))
        if not _text(b.get("kill")):
            out.append(_p(f"{where}.kill", "is required: every branch has a kill criterion",
                          "the result that kills the branch; it is recorded as a dead end its siblings see",
                          'e.g. kill = "no variant beats majority at p < 0.05"'))
        serves = b.get("serves", [])
        if not _strs(serves):
            out.append(_p(f"{where}.serves", "must be a list of hypothesis ids"))
        else:
            stray = [s for s in serves if s not in hids]
            if stray:
                out.append(_p(f"{where}.serves", f"names {', '.join(stray)}, not a hypothesis id",
                              f"hypotheses: {', '.join(sorted(str(h) for h in hids)) or 'none'}"))
        if "data_needs" in b and not isinstance(b["data_needs"], str):
            out.append(_p(f"{where}.data_needs", "must be a string"))
        out += _budget(b.get("budget"), f"{where}.budget", kind == "experiment")
    total = raw.get("budget")
    out += _budget(total, "budget", "experiment" in kinds)
    if "experiment" in kinds:
        if mode == "review":
            out.append(_p("mode", "is review, but a branch runs experiments",
                          "a review study runs no experiment (roles experimenter, replicator)",
                          "set mode to temporal or static, or run the experiment in a study of its own"))
        if not (_strs(raw.get("metrics")) and raw.get("metrics")):
            out.append(_p("metrics", "is required: an experiment branch needs the primary metric",
                          "the first metric is primary: it decides the hypotheses and is sealed with the plan",
                          'e.g. metrics = ["auc", "brier"]'))
    data = raw.get("data", [])
    if not isinstance(data, list):
        out.append(_p("data", "must be [[data]] tables"))
        data = []
    for i, d in enumerate(data):
        where = f"data.{d.get('name') if isinstance(d, dict) and _text(d.get('name')) else i}"
        if not isinstance(d, dict):
            out.append(_p(where, "must be a table"))
            continue
        out += _unknown(d, DATA, where)
        if not (isinstance(d.get("name"), str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", d["name"])):
            out.append(_p(f"{where}.name", "must be named like an identifier", "its study.toml [data.NAME]"))
        if not (_text(d.get("source")) or _text(d.get("recipe"))):
            out.append(_p(where, "needs a source or a recipe", "where the data comes from, or the script that makes it"))
    review = raw.get("review")
    if "literature" in kinds and not isinstance(review, dict):
        out.append(_p("review", "is required for a literature branch (role screener)",
                      "the review protocol: queries, include and exclude criteria, the coverage floor",
                      '[review] queries = [...], include = [{id = "I1", text = "..."}], exclude = [...], '
                      "coverage_floor = {queries_per_source = 2, hops = 1}"))
    if isinstance(review, dict):
        out += _review(review, raw)
    elif review is not None:
        out.append(_p("review", "must be a table"))
    if isinstance(total, dict) and all(isinstance(b, dict) and isinstance(b.get("budget"), dict) for b in branches):
        for k, word in (("usd", "$"), ("trials", "trials")):
            parts = [b["budget"].get(k) for b in branches]
            if isinstance(total.get(k), (int, float)) and all(isinstance(x, (int, float)) for x in parts):
                if sum(parts) > total[k] + 1e-9:
                    out.append(_p(f"budget.{k}", f"is less than the branches' sum ({word} {sum(parts):g} > "
                                  f"{total[k]:g})", "the branches' budgets come out of the total",
                                  "raise the total or lower a branch's share"))
        mins = total.get("minutes")
        for b in branches:
            m = b["budget"].get("minutes")
            if isinstance(mins, (int, float)) and isinstance(m, (int, float)) and m > mins:
                out.append(_p(f"branches.{b.get('id')}.budget.minutes", f"is more than the total ({m:g} > {mins:g})",
                              "branches run in parallel: each one's wall time is at most the total's"))
    if limit and isinstance(total, dict):
        for k, word in (("usd", "$"), ("minutes", "minutes")):
            if isinstance(total.get(k), (int, float)) and total[k] > limit[k]:
                out.append(_p(f"budget.{k}", f"is more than the workspace cap ({word} {total[k]:g} > {limit[k]:g})",
                              "openlab.toml [agents] cap_usd and cap_minutes bound one proposal",
                              "lower the total, or ask the person to raise the cap in openlab.toml"))
    return out


def _review(r: dict, raw: dict) -> list:
    out = _unknown(r, REVIEW, "review")
    if not (_strs(r.get("queries")) and r.get("queries")):
        out.append(_p("review.queries", "is required: a non-empty list of searches"))
    if "sources" in r and not (_strs(r["sources"]) and set(r["sources"]) <= set(schema.INDEXES) and r["sources"]):
        out.append(_p("review.sources", f"must be a non-empty list of {', '.join(schema.INDEXES)}"))
    for k in ("include", "exclude"):
        v = r.get(k)
        if not (isinstance(v, list) and v and all(isinstance(c, dict) and set(c) == {"id", "text"} and
                                                  all(_text(x) for x in c.values()) for c in v)):
            out.append(_p(f"review.{k}", "is required: criteria like [{id = \"I1\", text = \"...\"}]",
                          "every screening decision cites one"))
    f = r.get("coverage_floor")
    if not (isinstance(f, dict) and set(f) == {"queries_per_source", "hops"} and type(f["queries_per_source"]) is int
            and f["queries_per_source"] >= 1 and type(f["hops"]) is int and f["hops"] >= 0):
        out.append(_p("review.coverage_floor", "is required: {queries_per_source = N, hops = H}",
                      "the least search that counts as sufficient"))
    return out


def warnings(raw: dict) -> list:
    """What does not refuse the proposal but leaves part of it idle."""
    out = []
    served = {s for b in raw.get("branches") or [] if isinstance(b, dict) and _strs(b.get("serves"))
              for s in b["serves"]}
    for h in raw.get("hypotheses") or []:
        if isinstance(h, dict) and h.get("id") and h["id"] not in served:
            out.append(f"hypothesis {h['id']} is served by no branch: it will stay untouched")
    if any(isinstance(b, dict) and KIND_OF.get(b.get("role")) == "experiment" for b in raw.get("branches") or []) \
            and not raw.get("data"):
        out.append("no [[data]] is declared for the experiment branches")
    return out


# --- the drafting skeleton ---------------------------------------------------------------------------------------
def _toml(v) -> str:
    """One TOML value: strings, numbers, booleans, lists and inline tables."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_key(k)} = {_toml(x)}" for k, x in v.items()) + "}"
    return json.dumps(str(v))


def _key(k: str) -> str:
    return k if re.fullmatch(r"[A-Za-z0-9_-]+", k) else json.dumps(k)


def _long(text: str) -> str:
    """A multi-line string as a TOML literal block where it can be, else a basic string."""
    text = text.strip()
    if "\n" in text and "'''" not in text:
        return "'''\n" + text + "'''"
    return _toml(text)


def first_question(prompt: str) -> str:
    """The prompt's first question (or first sentence), as the proposal's question to start from."""
    text = " ".join(prompt.split())
    m = re.search(r"[^.?!]*\?", text)
    if m and len(m.group(0).strip()) <= 300:
        return m.group(0).strip()
    m = re.match(r"(.{1,300}?[.!])(\s|$)", text)
    return (m.group(1) if m else text[:300]).strip()


def skeleton(prompt: str, mode: str = "static", usd: float = 5.0, minutes: float = 120.0) -> str:
    """proposal.toml to draft from: the prompt as the question and background, every field the proposal needs,
    each with a comment saying what goes in it."""
    return f"""# The research proposal. Edit this file; proposal.md is rendered from it by `openlab propose lint FOLDER`.
# A person approves it with `openlab propose approve FOLDER --reason ...`: that writes study.toml from it, makes the
# question tree and seals the plan. After that a change is an amendment (`openlab propose amend`).

title = ""                       # a short title
question = {_toml(first_question(prompt))}
background = {_long(prompt)}
mode = {_toml(mode)}                  # temporal | static (an experiment) | review (a literature review only)
metrics = []                     # the primary metric first, e.g. ["auc", "brier"]
baselines = []                   # e.g. ["majority"]
risks = []                       # what could make the result wrong or useless

[budget]                         # the total: every branch's share comes out of it
usd = {usd:g}
minutes = {minutes:g}
trials = 20
holdout_looks = 1

[[hypotheses]]
id = "h1"
statement = ""                   # the claim
prediction = ""                  # what you expect to observe if it holds
falsifier = ""                   # the test or result that would refute it
tags = []                        # words naming what it is about

[[branches]]
id = "b1"
question = ""                    # the sub-question this branch answers
serves = ["h1"]                  # the hypotheses it tests
role = "experimenter"            # screener | replicator | experimenter | reviewer
method = ""                      # a sketch of what the agent does
data_needs = ""
stop = ""                        # when to stop exploring, e.g. "3 variants in a row without a gain"
kill = ""                        # the result that kills the branch (recorded as a dead end)
budget = {{usd = 1.0, minutes = 60, trials = 10}}

[[data]]
name = "table"                   # its study.toml [data.NAME]
source = ""                      # where it comes from
recipe = "recipes/make_data.py"  # the script that makes it

# A literature branch (role screener) needs the review protocol:
# [review]
# queries = ["...", "..."]
# sources = ["openalex", "arxiv"]
# include = [{{id = "I1", text = "..."}}]
# exclude = [{{id = "E1", text = "..."}}]
# coverage_floor = {{queries_per_source = 2, hops = 1}}
"""


# --- proposal.md -------------------------------------------------------------------------------------------------
def _cell(v) -> str:
    return " ".join(str(v or "").split()).replace("|", "\\|")


def _money(b: dict) -> str:
    parts = [f"${b['usd']:g}" if isinstance(b.get("usd"), (int, float)) else "",
             f"{b['minutes']:g} min" if isinstance(b.get("minutes"), (int, float)) else "",
             f"{b['trials']} trials" if isinstance(b.get("trials"), int) else ""]
    return ", ".join(p for p in parts if p) or "none"


def render(raw: dict, status: str = "draft: not approved") -> str:
    """proposal.md: the proposal for reading. Never edited by hand; it is written again from proposal.toml."""
    g = lambda k, d="": raw.get(k) or d                                       # noqa: E731
    out = [f"# {g('title', '(untitled)')}", "",
           f"> Rendered from {FILE} by openlab; edit that file, not this one. Status: {status}.", "",
           f"**Question.** {g('question', '(none)')}", "", f"**Background.** {g('background', '(none)')}", "",
           f"**Mode.** {g('mode', '?')}", "", "## Hypotheses", ""]
    hyps = [h for h in g("hypotheses", []) if isinstance(h, dict)]
    if hyps:
        out += ["| id | statement | prediction | falsified by |", "|---|---|---|---|"]
        out += [f"| {_cell(h.get('id'))} | {_cell(h.get('statement'))} | {_cell(h.get('prediction'))} | "
                f"{_cell(h.get('falsifier'))} |" for h in hyps]
    else:
        out.append("None.")
    out += ["", "## Branches", ""]
    for b in (b for b in g("branches", []) if isinstance(b, dict)):
        out += [f"### {b.get('id', '?')}: {b.get('question', '')} ({b.get('role', '?')})", "",
                f"- serves: {', '.join(b.get('serves') or []) or 'no hypothesis'}",
                f"- method: {b.get('method', '')}", f"- data needs: {b.get('data_needs', '') or 'none stated'}",
                f"- budget: {_money(b.get('budget') or {})}", f"- stop rule: {b.get('stop', '')}",
                f"- kill criterion: {b.get('kill', '')}", ""]
    data = [d for d in g("data", []) if isinstance(d, dict)]
    out += ["## Data", ""] + ([f"- `{d.get('name', '?')}`: {d.get('source', '')}"
                                + (f" (recipe `{d['recipe']}`)" if d.get("recipe") else "") for d in data] or ["None."])
    m = g("metrics", [])
    out += ["", "## Metrics and baselines", "",
            f"Primary: {m[0] if m else '(none)'}" + (f"; also {', '.join(m[1:])}" if len(m) > 1 else "") + ".",
            f"Baselines: {', '.join(g('baselines', [])) or 'none'}."]
    r = raw.get("review")
    if isinstance(r, dict):
        out += ["", "## Literature review protocol", "", f"- queries: {'; '.join(r.get('queries') or [])}",
                f"- indexes: {', '.join(r.get('sources') or ['openalex', 'arxiv'])}"]
        out += [f"- {k}: " + "; ".join(f"{c.get('id')} {c.get('text')}" for c in r.get(k) or [] if isinstance(c, dict))
                for k in ("include", "exclude")]
        out.append(f"- coverage floor: {r.get('coverage_floor')}")
    total = raw.get("budget") or {}
    shares = [b.get("budget") or {} for b in g("branches", []) if isinstance(b, dict)]
    usd = sum(s.get("usd") or 0 for s in shares if isinstance(s.get("usd"), (int, float)))
    trials = sum(s.get("trials") or 0 for s in shares if isinstance(s.get("trials"), int))
    out += ["", "## Budget", "", f"Total: {_money(total)}" + (f", {total['holdout_looks']} holdout looks"
                                                               if isinstance(total.get("holdout_looks"), int) else "")
            + f". The branches take ${usd:g} and {trials} trials of it.", "", "## Risks", ""]
    out += [f"- {x}" for x in g("risks", [])] or ["None stated."]
    return "\n".join(out) + "\n"


def write_md(folder, raw: dict, status: str) -> Path:
    p = Path(folder) / MD
    p.write_text(render(raw, status), encoding="utf-8")
    return p


# --- the record: approval, drift, amendment ----------------------------------------------------------------------
def approved(rec) -> Node | None:
    """The proposal as it stands: the latest proposal node (the approval or its latest amendment)."""
    return rec.last("proposal")


def _norm(raw: dict) -> dict:
    """The proposal keyed by id for diffing: hypotheses, branches and data become tables by id or name."""
    out = dict(raw)
    for k, key in (("hypotheses", "id"), ("branches", "id"), ("data", "name")):
        if isinstance(raw.get(k), list):
            out[k] = {str(x.get(key, i)) if isinstance(x, dict) else str(i): x for i, x in enumerate(raw[k])}
    return out


def diff(old: dict, new: dict) -> dict:
    return _spec._diff(_norm(old), _norm(new))


def protected(changes: dict, old: dict, new: dict) -> list:
    """The changed paths an amendment needs a person for."""
    out = [p for p in changes if any(fnmatch.fnmatchcase(p, pat) for pat in PROTECTED)]
    out += [p for p, ch in changes.items() if re.fullmatch(r"(branches|hypotheses)\.[^.]+", p) and ch["new"] is None]
    if (old.get("metrics") or [None])[0] != (new.get("metrics") or [None])[0]:
        out.append("metrics[0]")
    return sorted(set(out))


def drift(folder, rec) -> dict:
    """{} when proposal.toml is what was approved (or there is no approval), else the changes since."""
    node = approved(rec)
    if node is None:
        return {}
    return diff(node.payload["proposal"], load(folder))


def _drift_error(changes: dict) -> StudyError:
    lines = [f"{p}: {_short(c['old'])} -> {_short(c['new'])}" for p, c in list(changes.items())[:8]]
    return StudyError(f"the approved proposal changed: {', '.join(changes)}", "; ".join(lines),
                      "record an amendment with `openlab propose amend FOLDER --reason ...` (a person's act for "
                      "budgets, stop rules, kill criteria, falsifiers and the primary metric), or undo the edit")


def _short(v) -> str:
    text = "absent" if v is None else json.dumps(v, ensure_ascii=False, default=str)
    return text if len(text) <= 60 else text[:57] + "..."


def check_unchanged(folder, rec) -> None:
    changes = drift(folder, rec)
    if changes:
        raise _drift_error(changes)


def lint(study) -> dict:
    """Problems and warnings of proposal.toml, its standing (draft, approved, or changed since approval), and
    proposal.md rendered again. `ok` is true when it may be approved (a draft) or is unchanged (approved)."""
    folder, rec = study.folder, study.record
    raw = load(folder)
    probs = problems(raw, cap(folder))
    node = approved(rec)
    changes = diff(node.payload["proposal"], raw) if node else {}
    pf = {"errors": [], "warnings": [], "conflicts": []}
    if node is None and not probs:
        pf = preflight(study, raw)
        probs += pf["errors"]
    elif node is None:
        pf["warnings"].append("the approval checks (study.toml, model.py, data, sibling studies on the same data) "
                              "run once the proposal itself is clean")
    if node is None:
        status = "draft: not approved"
    elif changes:
        status = f"approved as {node.id}, CHANGED since: {', '.join(changes)} (record an amendment)"
    else:
        status = f"approved as {node.id} ({node.at[:16]}, by {node.session})"
    write_md(folder, raw, status)
    return {"ok": not probs and not changes, "problems": probs, "warnings": warnings(raw) + pf["warnings"],
            "conflicts": pf["conflicts"], "status": status, "approved": node.id if node else None,
            "changed": list(changes), "sha": sha(raw), "md": str(Path(folder) / MD),
            **({"drift": _drift_error(changes).meaning} if changes else {})}


def summary(raw: dict) -> str:
    """The budget summary a person reads before approving."""
    lines = [f"{raw.get('title')}: {raw.get('question')}", f"total budget: {_money(raw.get('budget') or {})}"]
    for b in raw.get("branches") or []:
        lines.append(f"  {b['id']} [{b['role']}] {_money(b['budget'])}: {b['question']}")
        lines.append(f"      stop: {b['stop']} | kill: {b['kill']}")
    lines.append(f"hypotheses: {', '.join(h['id'] for h in raw.get('hypotheses') or []) or 'none'}; "
                 f"primary metric: {(raw.get('metrics') or ['none'])[0]}")
    if raw.get("continues"):
        lines.append(f"continues {raw['continues']}: one trial and holdout ledger with it")
    if raw.get("fresh"):
        lines.append(f"FRESH QUESTION (you declare it by approving): {raw['fresh']}")
    return "\n".join(lines)


def _refuse(probs: list) -> StudyError:
    p = probs[0]
    more = f" ({len(probs) - 1} more: `openlab propose lint` lists them)" if len(probs) > 1 else ""
    return StudyError(p["error"] + more, p["meaning"], p["fix"])


# --- study.toml from the proposal --------------------------------------------------------------------------------
_HEADER = re.compile(r"^\s*\[\[?\s*([A-Za-z0-9_\-\".]+?)\s*\]\]?\s*(#.*)?$")


def _blocks(text: str) -> list:
    """[(section or "", lines)] in file order; a section's sub-tables ([review.extract]) are blocks of their own."""
    out, cur, lines = [], "", []
    for line in text.splitlines():
        m = _HEADER.match(line)
        if m:
            out.append((cur, lines))
            cur, lines = m.group(1).strip('"'), [line]
        else:
            lines.append(line)
    out.append((cur, lines))
    return out


def _section(name: str, body: dict) -> list:
    lines = [f"[{name}]"]
    lines += [f"{_key(k)} = {_toml(v)}" for k, v in body.items() if not isinstance(v, dict) or k != "extract"]
    if isinstance(body.get("extract"), dict) and body["extract"]:
        lines += ["", f"[{name}.extract]"] + [f"{_key(k)} = {_toml(v)}" for k, v in body["extract"].items()]
    return lines + [""]


def plan_toml(folder, raw: dict) -> str:
    """study.toml as the approved proposal makes it: [study] question and mode, [evaluate] metrics and baselines,
    [budget] trials (the branches' sum) and holdout looks, and [review] for a literature branch; every other section
    is kept as written (data, target, split...). A review study keeps only [study], [review] and its notes."""
    p = Path(folder) / "study.toml"
    text = p.read_text(encoding="utf-8") if p.is_file() else ""
    try:
        cur = tomllib.loads(text) if text else {}
    except tomllib.TOMLDecodeError as e:
        raise StudyError("study.toml does not parse", str(e), "fix it, then approve again") from None
    mode = raw["mode"]
    want: dict = {"study": {"name": (cur.get("study") or {}).get("name") or Path(folder).name,
                            "question": raw["question"], "mode": mode}}
    for k in ("continues", "fresh"):           # the same-data ledger: the proposal's word, else study.toml's
        v = raw.get(k) or (cur.get("study") or {}).get(k)
        if v:
            want["study"][k] = v
    kinds = {KIND_OF[b["role"]] for b in raw["branches"]}
    if mode != "review":
        ev = dict(cur.get("evaluate") or {})
        ev["metrics"] = list(raw.get("metrics") or ev.get("metrics") or [])
        if raw.get("baselines"):
            ev["baselines"] = list(raw["baselines"])
        bud = dict(cur.get("budget") or {})
        bud["trials"] = max(1, sum(b["budget"].get("trials") or 0 for b in raw["branches"]))
        bud["holdout_looks"] = raw["budget"].get("holdout_looks", bud.get("holdout_looks", 1))
        want["evaluate"], want["budget"] = ev, bud
    if isinstance(raw.get("review"), dict) and ("literature" in kinds or mode == "review"):
        r = raw["review"]
        rv = dict(cur.get("review") or {})
        rv.update({"question": r.get("question") or raw["question"],
                   "sources": r.get("sources") or rv.get("sources") or ["openalex", "arxiv"],
                   "queries": r["queries"], "include": r["include"], "exclude": r["exclude"],
                   "coverage_floor": r["coverage_floor"],
                   **{k: r[k] for k in ("citation_hops", "n", "since", "extract") if k in r}})
        want["review"] = rv
    keep = _spec.ONLY.get(mode)
    out, placed = [], set()
    for sec, lines in _blocks(text):
        top = sec.split(".")[0]
        if top in want:
            if top not in placed:
                out += _section(top, want[top])
                placed.add(top)
            continue
        if sec and keep is not None and top not in keep:
            continue                                            # a review keeps no experiment sections
        out += lines
    missing = [s for s in want if s not in placed]
    if missing:
        head = [x for s in missing if s == "study" for x in _section(s, want[s])]
        tail = [x for s in missing if s != "study" for x in _section(s, want[s])]
        out = head + out + ([""] if out and out[-1].strip() else []) + tail
    new = "\n".join(out).rstrip() + "\n"
    try:
        _spec.build(json.loads(canonical(tomllib.loads(new))), Path(folder))
    except StudyError as e:
        raise StudyError(f"study.toml would not be valid after approval: {e.error}", e.meaning,
                         "edit study.toml's experiment sections (data, target, split) to fit the proposal's mode "
                         "and data, then approve again; nothing was written") from None
    except tomllib.TOMLDecodeError as e:                        # pragma: no cover -- our own writer
        raise StudyError("study.toml would not parse after approval", str(e)) from None
    return new


# --- what would block execution, checked before approval ---------------------------------------------------------
def _data_sha(study, ds) -> str:
    """The sha256 a dataset would be registered under ("" when it cannot be known before a run: a provider not yet
    fetched). A file is made by its recipe when missing, as materialise would; nothing is recorded."""
    import io
    from . import data
    kind, src = data._source(ds)
    last = study.record.last("dataset", name=ds.name)
    if kind == "provider":
        return last.payload["sha256"] if last else ""
    df = data._frame(src.load(study.folder, ds, False))
    data._check(df, ds)
    buf = io.BytesIO()
    df.write_parquet(buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def preflight(study, raw: dict) -> dict:
    """What would stop the approved plan's execution, found now (`raw` passes `problems`): study.toml as approval would
    write it, model.py for the experiment branches, each dataset's recipe, file or snapshot, the review protocol of a
    literature branch, and the siblings asking the same question of the same data (openlab.study.ledger), unless the
    proposal says `continues` (one shared trial ledger) or `fresh` (a new question, a person's word by approving).
    {"errors": [problem], "warnings": [text], "conflicts": [{name, folder, rel, trials, looks, fixes}]}."""
    from . import ledger
    folder = Path(study.folder)
    errors, warns, found = [], [], []
    kinds = {KIND_OF[b["role"]] for b in raw["branches"]}
    try:
        text = plan_toml(folder, raw)
    except StudyError as e:
        return {"errors": [{"where": "study.toml", "error": e.error, "meaning": e.meaning, "fix": e.fix}],
                "warnings": [], "conflicts": []}
    spec = _spec.build(json.loads(canonical(tomllib.loads(text))), folder)
    if "experiment" in kinds:
        roles = sorted({b["role"] for b in raw["branches"] if KIND_OF[b["role"]] == "experiment"})
        if not (folder / "model.py").is_file():
            errors.append(_p("", f"{folder.name}/model.py is missing, and the {' and '.join(roles)} branches run it",
                             "every run imports model.py (features and label in temporal mode, and class Model)",
                             "write model.py (the template: `openlab study new X --template static` or temporal)"))
        declared = set(spec.data)
        for d in raw.get("data") or []:
            if d.get("name") not in declared:
                errors.append(_p(f"data.{d.get('name')}", "is not a dataset of study.toml",
                                 f"the runs read study.toml's [data.NAME] only (declared: "
                                 f"{', '.join(sorted(declared)) or 'none'})",
                                 f"add [data.{d.get('name')}] to study.toml with its file and recipe, snapshot or "
                                 "provider"))
            elif d.get("recipe") and not (folder / d["recipe"]).is_file():
                errors.append(_p(f"data.{d['name']}.recipe", f"names {d['recipe']}, which does not exist",
                                 "a data file is made by its recipe", "write the recipe, or fix the path"))
        shas, unknown = {}, []
        rivals = [] if raw.get("fresh") else ledger.same_target(folder, study.record, spec.raw)
        for name, ds in spec.data.items():
            if ds.recipe and not (folder / ds.recipe).is_file():
                errors.append({"where": f"study.toml data.{name}", "error": f"data.{name}: no recipe {ds.recipe}",
                               "meaning": "a data file is made by its recipe, and none would run",
                               "fix": "write it, or fix the path in study.toml"})
                continue
            if ds.snapshot and not Path(ds.snapshot).expanduser().exists():
                errors.append({"where": f"study.toml data.{name}", "error": f"data.{name}: no snapshot {ds.snapshot}",
                               "meaning": "the dataset would not load at the first run",
                               "fix": "fix the snapshot path, or point data at a file with its recipe"})
                continue
            if not rivals:
                continue                                # nothing to compare: the data need not be made now
            try:
                shas[name] = _data_sha(study, ds)
            except StudyError as e:
                errors.append({"where": f"study.toml data.{name}", "error": e.error, "meaning": e.meaning,
                               "fix": e.fix or "fix the recipe or the data section"})
                continue
            except Exception as e:                  # noqa: BLE001 -- any source failure is one clear finding
                errors.append({"where": f"study.toml data.{name}", "error": f"data.{name} could not be read",
                               "meaning": f"{type(e).__name__}: {e}", "fix": "fix the recipe or the data section"})
                continue
            if not shas[name]:
                unknown.append(name)
        if unknown:
            warns.append(f"data {', '.join(unknown)} is fetched at the first run, so the same-data check against "
                         "sibling studies runs then")
        elif shas and len(shas) == len(spec.data):
            owned = sorted(shas.values())
            for c in ledger.conflicts(folder, study.record, spec.raw, owned):
                found.append({**c, "fixes": ledger.fixes(c)})
                errors.append(_p("", f"{spec.name} asks the question of {c['name']} ({c['folder']}) of the same "
                                 f"data ({c['trials']} trials and {c['looks']} holdout looks spent there)",
                                 "a second study of one question would start a fresh trial budget and a fresh "
                                 "holdout, and its first run would be refused",
                                 f'add `continues = "{c["rel"]}"` to proposal.toml (one shared trial ledger), or '
                                 '`fresh = "<why this is a new question>"` (a person declares it by approving)'))
    if raw.get("fresh"):
        warns.append(f"fresh: approving declares a new question on data other studies may use: {raw['fresh']}")
    if "literature" in kinds and not spec.review.get("queries"):
        errors.append(_p("review", "is empty, and a screener branch screens what it finds",
                         "study.toml's [review] needs queries, criteria and a coverage floor",
                         "fill [review] in proposal.toml"))
    return {"errors": errors, "warnings": warns, "conflicts": found}


def declare(folder, key: str, value: str) -> None:
    """Write a top-level `key = "value"` (continues or fresh) into proposal.toml, before its first table."""
    if key not in ("continues", "fresh") or not value.strip():
        raise StudyError(f"cannot declare {key}", "continues or fresh, with a value")
    p = path(folder)
    lines, line = p.read_text(encoding="utf-8").splitlines(), f"{key} = {_toml(value.strip())}"
    at = next((i for i, x in enumerate(lines) if re.match(rf"\s*{key}\s*=", x)), None)
    if at is not None:
        lines[at] = line
    else:
        first = next((i for i, x in enumerate(lines) if _HEADER.match(x)), len(lines))
        lines.insert(first, line)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- approval ---------------------------------------------------------------------------------------------------
def _tree_nodes(study, raw: dict, tree: dict, by: str) -> list:
    """Write what the proposal adds to the question tree (the root question, a question per branch, each hypothesis
    under the first branch that serves it, with its falsifier) as the person `by`; returns the new node ids and
    fills `tree`."""
    from ..agents.identity import acting_as
    from .personkey import current as _signer
    from .schema import Hypothesis, Question
    rec, made = study.record, []
    actor = "person" if _signer() is not None else "agent"     # an agent's unprotected amendment adds agent nodes
    with acting_as(by):
        if "root" not in tree:
            tree["root"] = rec.add(Question(text=raw["question"]), actor=actor).id
            made.append(tree["root"])
        for b in raw["branches"]:
            if b["id"] not in tree:
                q = rec.add(Question(text=f"{b['id']}: {b['question']}"), actor=actor,
                            links=[("part_of", tree["root"])])
                tree[b["id"]], made = q.id, made + [q.id]
        for h in raw.get("hypotheses") or []:
            if h["id"] in tree:
                continue
            home = next((b["id"] for b in raw["branches"] if h["id"] in (b.get("serves") or [])), "root")
            tags = list(h.get("tags") or []) or [h["id"]]
            n = rec.add(Hypothesis(statement=h["statement"], prediction=h["prediction"],
                                   mechanism=h.get("mechanism", ""), falsified_by=h["falsifier"], tags=tags),
                        actor=actor, links=[("part_of", tree[home])])
            tree[h["id"]], made = n.id, made + [n.id]
    return made


def _dead_end_clash(rec, raw: dict) -> None:
    """Refuse before writing anything: a hypothesis sharing a tag with a recorded dead end."""
    for h in raw.get("hypotheses") or []:
        tags = {t.strip().casefold() for t in (h.get("tags") or [h["id"]])}
        for d in rec.find("dead_end"):
            shared = tags & {t.strip().casefold() for t in d.payload["tags"]}
            if shared:
                raise StudyError(f"hypothesis {h['id']} shares {', '.join(sorted(shared))} with dead end {d.id}",
                                 f"tried before: {d.payload['statement']}; it failed because {d.payload['reason']}",
                                 "change its tags and say in its statement what is different now, or drop it")


def approve(study, reason: str, by: str) -> dict:
    """The person `by` approves proposal.toml (the CLI asked them to confirm): study.toml is written from it, the
    question tree made, the proposal node written (actor person) and the study sealed (or, if it already was, the
    change to study.toml recorded as a person's amendment)."""
    folder, rec = study.folder, study.record
    if not reason.strip():
        raise StudyError("an approval needs a reason", fix="say why this plan is the one to run")
    raw = load(folder)
    probs = problems(raw, cap(folder))
    if probs:
        raise _refuse(probs)
    node = approved(rec)
    if node is not None:
        raise StudyError("the proposal is already approved", f"approved as {node.id} at {node.at}",
                         "a change is an amendment: openlab propose amend FOLDER --reason ...")
    blocked = preflight(study, raw)["errors"]
    if blocked:
        raise _refuse(blocked)
    _dead_end_clash(rec, raw)
    text = plan_toml(folder, raw)
    from ..agents.identity import acting_as
    (Path(folder) / "study.toml").write_text(text, encoding="utf-8")
    tree: dict = {}
    made = _tree_nodes(study, raw, tree, by)
    with acting_as(by):
        p = rec.add(schema.Proposal(proposal=raw, sha=sha(raw), tree=tree, reason=reason, protected=[]),
                    actor="person", links=[("plans", i) for i in made])
        if _spec.sealed(rec) is None:
            plan = study.seal(actor="person")
            sealed = {"plan": plan["id"], "amended": None}
        elif _spec._diff(_spec.sealed(rec).raw, _spec.load(folder).raw):
            am = study.amend(f"the approved proposal {p.id}: {reason}", person=True)
            sealed = {"plan": _spec._plan(rec).id, "amended": am["id"]}
        else:
            sealed = {"plan": _spec._plan(rec).id, "amended": None}
    write_md(folder, raw, f"approved as {p.id} ({p.at[:16]}, by {by})")
    return {"id": p.id, "sha": p.payload["sha"], "tree": tree, **sealed, "branches": [b["id"] for b in raw["branches"]],
            "trials": _spec.load(folder).budget.get("trials", 0)}


def amend(study, reason: str, person: bool, by: str = "") -> dict:
    """Record the edit of proposal.toml since the approved proposal: a new proposal node amending it, with the new
    branches' and hypotheses' nodes. A protected change needs a person; study.toml is not changed (amend it on its own
    when the plan's sealed fields change)."""
    folder, rec = study.folder, study.record
    node = approved(rec)
    if node is None:
        raise StudyError("there is no approved proposal to amend", fix="a person approves it first: openlab propose approve")
    if not reason.strip():
        raise StudyError("an amendment needs a reason", fix="say why the plan changes")
    raw = load(folder)
    probs = problems(raw, cap(folder))
    if probs:
        raise _refuse(probs)
    old = node.payload["proposal"]
    changes = diff(old, raw)
    if not changes:
        raise StudyError("proposal.toml matches the approved proposal", "there is nothing to amend")
    guarded = protected(changes, old, raw)
    if guarded and not person:
        raise StudyError(f"{', '.join(guarded)} can only be amended by a person", "changed: " + ", ".join(changes),
                         "undo that part of the edit, or ask a person to run `openlab propose amend FOLDER --person`")
    from ..agents.identity import acting_as, current
    who = by or current()
    tree = dict(node.payload["tree"])
    _dead_end_clash(rec, {"hypotheses": [h for h in raw.get("hypotheses") or [] if h["id"] not in tree]})
    text, carried = study_effect(study, raw)
    toml = Path(folder) / "study.toml"
    was = toml.read_text(encoding="utf-8") if toml.is_file() else None
    am = None
    with acting_as(who):
        if carried:                             # the proposal's budget, metrics, question... reach study.toml
            toml.write_text(text, encoding="utf-8")
            try:
                am = study.amend(f"the amended proposal: {reason}", person=person)
            except BaseException:
                if was is not None:
                    toml.write_text(was, encoding="utf-8")
                raise
    made = _tree_nodes(study, raw, tree, who)
    with acting_as(who):
        p = rec.add(schema.Proposal(proposal=raw, sha=sha(raw), tree=tree, reason=reason, protected=guarded,
                                    prev_sha=node.payload["sha"]), actor="person" if person else "agent",
                    links=[("amends", node.id)] + [("plans", i) for i in made])
    write_md(folder, raw, f"approved as {rec.find('proposal')[0].id}, amended as {p.id} ({p.at[:16]}, by {who})")
    return {"id": p.id, "changes": list(changes), "protected": guarded, "new_nodes": made,
            "study_amendment": am and am["id"], "study_changes": list(carried)}


def study_effect(study, raw: dict) -> tuple:
    """(study.toml as the proposal `raw` makes it, the changes that carries against the sealed spec: {} when the
    proposal's amendment leaves study.toml's sealed fields as they are)."""
    text = plan_toml(study.folder, raw)
    eff = _spec.sealed(study.record)
    if eff is None:
        return text, {}
    new = json.loads(canonical(tomllib.loads(text)))
    return text, _spec._diff(eff.raw, _spec.build(new, Path(study.folder)).raw)


def study_effect_lines(study) -> list:
    """What `propose amend` would change in study.toml, as lines for the person to read before signing."""
    try:
        _, carried = study_effect(study, load(study.folder))
    except StudyError as e:
        return [f"study.toml: {e.error}"]
    return [f"study.toml {k}: {_short(v['old'])} -> {_short(v['new'])} (recorded as the study's amendment too)"
            for k, v in carried.items()]


def require(study) -> Node:
    """The approved proposal, unchanged since; refused otherwise."""
    node = approved(study.record)
    if node is None:
        raise StudyError("the proposal is not approved", "only an approved, sealed plan is executed",
                         "a person runs `openlab propose approve FOLDER --reason ...` first")
    check_unchanged(study.folder, study.record)
    return node


# --- branches: state, frontier items, stop ------------------------------------------------------------------------
def branch_item_id(question: str, branch: str) -> str:
    return f"frontier:{hashlib.sha256(f'branch|{question}|{branch}'.encode()).hexdigest()[:12]}"


def _stops(rec) -> dict:
    """{branch id: the latest successful branch_stop action}."""
    return {a.payload["args"].get("branch"): a for a in rec.find("action", verb="branch_stop")
            if a.payload["outcome"].get("ok")}


def _hyp_rows(study, node: Node, qid: str, served: list) -> list:
    """The branch's hypotheses (those it serves and any under its question), each tested, killed or untouched."""
    rec = study.record
    tree = node.payload["tree"]
    ids = [tree[h] for h in served if h in tree]
    ids += [s for s, _, _ in rec.links(qid, rel="part_of", incoming=True) if s.startswith("hypothesis:") and s not in ids]
    names = {v: k for k, v in tree.items()}
    rows = []
    for hid in ids:
        h = rec.get(hid)
        tests = [rec.get(s) for s, _, _ in rec.links(hid, rel="of", incoming=True) if s.startswith("test:")]
        ends = [rec.get(s) for s, _, _ in rec.links(hid, rel="of", incoming=True) if s.startswith("dead_end:")]
        verdict = schema.tested(rec, [hid]) if tests else None
        state = "tested" if tests else "killed" if ends else "untouched"
        by = (ends or tests or [None])[-1]
        rows.append({"id": names.get(hid, hid), "node": hid, "statement": h.payload["statement"], "state": state,
                     "label": verdict or state, "verdict": verdict, "falsifier": h.payload["falsified_by"],
                     "by": by.session if by else "",
                     "dead_ends": [{"id": e.id, "statement": e.payload["statement"], "reason": e.payload["reason"],
                                    "by": e.session} for e in ends]})
    return rows


def branch_states(study) -> list:
    """Each branch of the approved proposal with its question node, frontier item id and state: stopped or killed
    (a branch_stop), killed (every hypothesis it serves has a dead end), done (its question is answered or refuted
    in the tree), else open."""
    rec = study.record
    node = approved(rec)
    if node is None:
        return []
    stops = _stops(rec)
    from . import claims
    status = {}

    def walk(n):
        status[n["id"]] = n["status"]
        for k in n.get("children", []):
            walk(k)
    for q in claims.tree(study)["questions"]:
        walk(q)
    out = []
    for b in node.payload["proposal"]["branches"]:
        qid = node.payload["tree"].get(b["id"], "")
        hyps = _hyp_rows(study, node, qid, b.get("serves") or []) if qid else []
        stop = stops.get(b["id"])
        supported = [h["id"] for h in hyps if h["verdict"] == "supported"]
        if stop:                        # a kill criterion cannot have been met by a branch whose hypothesis held
            state = "killed" if stop.payload["args"].get("killed") and not supported else "stopped"
        elif hyps and all(h["state"] == "killed" for h in hyps):
            state = "killed"            # every hypothesis it serves was dropped as a dead end, untested
        elif status.get(qid) in ("answered", "refuted"):
            state = "done"
        else:
            state = "open"
        if state == "killed":           # a killed branch's untested hypotheses die with it
            hyps = [{**h, "state": "killed", "label": "killed"} if h["state"] == "untouched" else h for h in hyps]
        out.append({"branch": b, "question": qid, "item": branch_item_id(qid, b["id"]), "state": state,
                    "label": STATE_LABELS.get(state, state.upper()), "hypotheses": hyps, "stop": stop})
    return out


#: how a branch's state reads on the status screen: a fired stop rule is not a kill
STATE_LABELS = {"stopped": "STOPPED (stop rule)", "killed": "KILLED (kill criterion)"}


def frontier_items(study, make, costs: dict) -> list:
    """The open branches as frontier items (kind `branch`, verb `delegate`), for openlab.study.frontier."""
    out = []
    for s in branch_states(study):
        if s["state"] != "open":
            continue
        b = s["branch"]
        item = make("branch", s["question"], f"branch {b['id']} ({b['role']}) of the approved proposal: "
                    f"{b['question'][:60]}", "delegate", 1, float(b["budget"]["minutes"]) * 60, key=b["id"])
        out.append(item)
    return out


def stop(study, branch: str, reason: str, killed: bool = False, shown_by=()) -> dict:
    """Record that a branch stopped (its stop rule fired) or was killed (its kill criterion was met)."""
    node = require(study)
    ids = [b["id"] for b in node.payload["proposal"]["branches"]]

    def go() -> dict:
        if branch not in ids:
            raise StudyError(f"no branch {branch!r} in the approved proposal", f"branches: {', '.join(ids)}")
        if not reason.strip():
            raise StudyError("a stopped branch says why", fix="name the stop rule or kill criterion that fired, and "
                             "the result that showed it")
        for i in shown_by:
            study.record.get(i)
        if killed:
            s = next(x for x in branch_states(study) if x["branch"]["id"] == branch)
            held = [h["id"] for h in s["hypotheses"] if h["verdict"] == "supported"]
            if held:
                raise StudyError(f"branch {branch} was not killed: {', '.join(held)} is supported",
                                 "killed means its kill criterion was met; a branch whose hypothesis held stopped",
                                 f"record it as stopped: propose_stop(folder, branch={branch!r}, reason) without "
                                 "killed")
        return {"branch": branch, "killed": killed, "ids": list(shown_by)}
    return study._act("branch_stop", {"branch": branch, "reason": reason, "killed": killed,
                                      "shown_by": list(shown_by)}, "agent", go)


# --- drafting -----------------------------------------------------------------------------------------------------
def cli_available(cli: str = "claude") -> bool:
    import os
    import shutil
    return bool(os.environ.get(f"OPENLAB_{cli.upper()}_BIN") or shutil.which(cli))


def _where(folder) -> tuple:
    """(workspace root, the study folder relative to it)."""
    from ..agents.store import root_of
    folder = Path(folder).resolve()
    root = root_of(folder)
    try:
        return root, str(folder.relative_to(root)) or "."
    except ValueError:
        return root, str(folder)


PLANNER = """Draft the research proposal for the study {rel} (an openlab study folder) from the person's prompt below.

1. Read {rel}/proposal.toml (a skeleton; each field says what goes in it) and {rel}/study.toml.
2. If it helps, scan the literature briefly (WebSearch, or openlab_tool("paper_search")) for the background, the
   baselines and the risks. Keep it short: a few searches, no full reading.
3. Fill proposal.toml: title, question, a short background, hypotheses (statement, prediction, falsifier: the result
   that would refute it), branches (question, the hypotheses it serves, role {roles}, method, data needs,
   budget {{usd, minutes, trials}}, a stop rule, a kill criterion), data, metrics (the primary first), baselines, a
   [review] protocol if a branch is a literature review (role screener), the total budget, risks. The total is at most
   ${cap_usd:g} and {cap_minutes:g} minutes; the branches' $ and trials add up to at most the total.
   Make study.toml's data, target and split fit the proposal (mode {mode}).
4. Run `openlab propose lint {rel}` (or openlab_tool("propose_lint")) and fix what it refuses, until it is clean.
5. Stop there. Approval is the person's act: do not approve, run, materialise or create any other study.
Finish with five lines: the question, the hypotheses, the branches with their budgets, and anything unresolved.

The person's prompt:
{prompt}"""


def new(folder, prompt: str, mode: str = "static", usd: float | None = None, minutes: float | None = None) -> dict:
    """Scaffold a proposal: the study folder (from the mode's template) if it is not one yet, proposal.toml with the
    prompt as its question and background, and proposal.md."""
    from .. import study as _study
    folder = Path(folder).resolve()
    if not prompt.strip():
        raise StudyError("a proposal starts from a prompt", "a paragraph saying what you want to find out",
                         'openlab propose new FOLDER "<your paragraph>" (or --from FILE)')
    if mode not in MODES:
        raise StudyError(f"no mode `{mode}`", f"one of {', '.join(MODES)}")
    if path(folder).exists():
        raise StudyError(f"{folder} already has {FILE}", "a study has one proposal", "edit it, or choose a new folder")
    made = []
    if not (folder / "study.toml").is_file():
        _study.new(folder, mode)
        made.append("study.toml")
    s = _study.open(folder)
    if approved(s.record) is not None:
        raise StudyError(f"{folder} already has an approved proposal", fix="amend it instead")
    limit = cap(folder)
    usd = min(float(usd) if usd else 5.0, limit["usd"])
    minutes = min(float(minutes) if minutes else 120.0, limit["minutes"])
    path(folder).write_text(skeleton(prompt, mode, usd, minutes), encoding="utf-8")
    raw = load(folder)
    write_md(folder, raw, "draft: not approved")
    s._act("propose_new", {"mode": mode, "usd": usd, "minutes": minutes, "prompt": prompt[:2000]}, "agent", lambda: {})
    return {"folder": str(folder), "files": made + [FILE, MD], "mode": mode, "budget": {"usd": usd, "minutes": minutes},
            "problems": len(problems(raw, limit))}


def draft(folder, prompt: str, cli: str = "claude", usd: float | None = None, minutes: float = PLANNER_MINUTES,
          model: str | None = None) -> dict:
    """Spawn the planning agent (role planner, a small budget: `usd`, default $0.50, any amount above 0) that fills
    proposal.toml and lints it until clean; `model` as `openlab agent spawn --model`."""
    from ..agents.spawn import AgentError, spawn
    usd = PLANNER_USD if usd is None else float(usd)
    if usd <= 0:
        raise StudyError(f"the planner's budget must be above $0 (asked ${usd:g})",
                         "a planning agent spends on its model calls", "--planner-budget-usd 0.2, say")
    root, rel = _where(folder)
    raw = load(folder)
    limit = cap(folder)
    text = PLANNER.format(rel=rel, roles=" | ".join(ROLES), cap_usd=limit["usd"], cap_minutes=limit["minutes"],
                          mode=raw.get("mode", "static"), prompt=prompt.strip())
    try:
        a = spawn(root, text, role="planner", study=rel, question=f"draft the proposal: {first_question(prompt)}",
                  cli=cli, budget={"usd": usd, "minutes": minutes}, model=model)
    except AgentError as e:
        raise StudyError(e.error, e.meaning, e.try_) from None
    return {"id": a["id"], "cli": cli, "budget": a["budget"], "model": _asked_model(root, a)}


def _asked_model(root, a: dict) -> str:
    """The model an agent was spawned on, as asked ("" for the CLI's default)."""
    from ..agents.store import Store
    try:
        return json.loads((Store(root).agent_dir(a["id"]) / "launch.json").read_text()).get("model") or ""
    except (OSError, ValueError):
        return ""


# --- execution ----------------------------------------------------------------------------------------------------
LEAD = """Execute the approved research proposal of the study {rel}: {title}
It is signed off by a person (proposal {pid}, sha {sha}) and the plan is sealed: do not edit proposal.toml or the
sealed fields of study.toml (an edit is refused until a person records an amendment).

Question: {question} (question node {root})

Hypotheses (each decided by its falsifier):
{hyps}

Branches (each is one frontier item; give each to one child agent):
{branches}

How to run it:
1. Delegate each branch with openlab_agent_spawn(prompt, role=<the branch's role>, study="{rel}",
   question="<branch id>: <branch question>", budget_usd=<the branch's $>, minutes=<the branch's minutes>), never more
   than the branch's budget. Never delegate with Claude Code's own Agent or Task tool: it is refused to a lead,
   because such a sub-agent runs outside the branch budget and the role prompt. The child's prompt carries the branch's method, data needs, hypotheses (node ids and
   falsifiers), trial share, stop rule, kill criterion and frontier item, and tells it to lease the item first with
   openlab_study_take(folder="{rel}", item=<item>) and to stop at its trial share.
2. A child that meets its kill criterion records each hypothesis it killed as a dead end, with its evidence:
   openlab_study_dead_end(folder="{rel}", of=<hypothesis node>, statement, reason, tags, shown_by=[result, evidence or
   test ids]), then openlab_tool("propose_stop", {{folder: "{rel}", branch: <id>, reason, killed: true}}).
   Siblings read the dead ends before they start anything.
3. When a stop rule fires (or a branch's budget is spent), the branch stops: openlab_tool("propose_stop",
   {{folder: "{rel}", branch: <id>, reason}}) and the child ends.
4. Person acts (holdout looks, protected amendments, more budget, a fresh question) go to openlab_agent_request; do
   other work meanwhile. If nothing can go on without the person, say so and end: once they decide, they resume
   the run (`openlab propose execute --resume`), and a new lead gets the unfinished branches.
5. openlab_agent_wait waits for children; openlab_propose_status(folder="{rel}") shows each branch's agents, spend,
   trials and hypotheses.
6. Finish with a summary: the hypotheses tested, killed and untouched (with node ids), each branch's state and spend,
   and what is left open."""


def _hyp_line(h: dict, tree: dict) -> str:
    return f"- {h['id']} ({tree.get(h['id'], '?')}): {h['statement']} | falsified if: {h['falsifier']}"


def _branch_line(b: dict, s: dict) -> str:
    bud = b["budget"]
    return (f"- {b['id']} [{b['role']}] {b['question']} (question {s['question']}, item {s['item']})\n"
            f"    serves: {', '.join(b.get('serves') or []) or 'none'}; method: {b['method']}"
            + (f"; data: {b['data_needs']}" if b.get("data_needs") else "") + "\n"
            f"    budget: ${bud['usd']:g}, {bud['minutes']:g} min, {bud.get('trials', 0)} trials; "
            f"stop when: {b['stop']}; kill if: {b['kill']}"
            + (f"\n    code: its own working copy branches/{b['id']}/ (model.py and the modules it imports): the child "
               f"edits only that copy and runs it with openlab_study_run(folder, code=\"branches/{b['id']}\"); never "
               "the study's model.py, which other branches share" if KIND_OF[b["role"]] == "experiment" else ""))


RESUME = """This RESUMES the execution of the study {rel}: earlier agents stopped ({why}).
Finished branches (do not redo them): {finished}
Person decisions since the agents asked: {decisions}
Budget left for this run: ${usd:g}, {minutes:g} min (the proposal's total less what the earlier agents spent).
Only the branches below are left; read openlab_propose_status(folder="{rel}") and the study's dead ends first.

"""


def brief(study, only=None, preamble: str = "") -> str:
    """The lead's execution brief, from the approved proposal; `only`: the branch ids left (a resumed run), after
    `preamble` (what finished and what a person decided)."""
    node = require(study)
    raw, tree = node.payload["proposal"], node.payload["tree"]
    states = {s["branch"]["id"]: s for s in branch_states(study)}
    _, rel = _where(study.folder)
    branches = [b for b in raw["branches"] if only is None or b["id"] in only]
    served = {h for b in branches for h in b.get("serves") or []}
    hyps = [h for h in raw.get("hypotheses") or [] if only is None or h["id"] in served]
    return preamble + LEAD.format(rel=rel, title=raw["title"], pid=node.id, sha=node.payload["sha"][:12],
                                  question=raw["question"], root=tree.get("root", "?"),
                                  hyps="\n".join(_hyp_line(h, tree) for h in hyps) or "- none",
                                  branches="\n".join(_branch_line(b, states[b["id"]]) for b in branches))


FINISHED = ("done", "killed", "stopped")


def _decisions(study) -> list:
    """The inbox items the study's agents filed, as lines (decided ones say by whom and why)."""
    from ..agents.store import Store
    root, rel = _where(study.folder)
    if not (root / ".openlab" / "control.db").exists():
        return []
    owned = {a["id"] for a in Store(root).agents() if a["study"] == rel}
    return [f"#{i['id']} {i['kind']} from {i['agent']} {i['status']}"
            + (f" by {i['decided_by']}: {i['decision_reason']}" if i["decided_by"] else "") + f" (asked: {i['reason']})"
            for i in Store(root).inbox(None) if i["agent"] in owned or i["study"] in (rel, str(study.folder))]


def resume_plan(study) -> dict:
    """What `execute --resume` would run: the unfinished branches, the finished ones with their state, the person
    decisions since, and the budget left (the total less the spend of every earlier agent of the study)."""
    node = require(study)
    raw = node.payload["proposal"]
    st = status(study)
    finished = [f"{b['id']} {b['state']}" for b in st["branches"] if b["state"] in FINISHED]
    left = [b["id"] for b in st["branches"] if b["state"] not in FINISHED]
    usd = round(float(raw["budget"]["usd"]) - st["spent"]["usd"], 4)
    used_min = sum(x["minutes"] for x in st["lead"])
    minutes = round(float(raw["budget"]["minutes"]) - used_min, 1)
    return {"left": left, "finished": finished, "decisions": _decisions(study), "budget": {"usd": usd,
            "minutes": minutes}, "spent": {"usd": st["spent"]["usd"], "minutes": round(used_min, 1)},
            "blocked": [b["blocked"] for b in st["branches"] if b.get("blocked")]}


def planned_tree(study) -> list:
    """The tree execution follows, as lines: the lead, then each branch (state, budget, item) and its hypotheses."""
    node = require(study)
    raw = node.payload["proposal"]
    tot = raw["budget"]
    lines = [f"lead (role lead, budget ${tot['usd']:g}, {tot['minutes']:g} min): {raw['question']}"]
    for s in branch_states(study):
        b = s["branch"]
        lines.append(f"  └ {b['id']} [{b['role']}] {s['state']}, ${b['budget']['usd']:g}, {b['budget']['minutes']:g} "
                     f"min, {b['budget'].get('trials', 0)} trials, item {s['item']}: {b['question']}")
        lines += [f"      {h['id']} {h['label']}: {h['statement']}" for h in s["hypotheses"]]
    return lines


def live_lead(study) -> dict | None:
    from ..agents.store import LIVE, Store
    root, rel = _where(study.folder)
    if not (root / ".openlab" / "control.db").exists():
        return None
    return next((a for a in Store(root).find(role="lead", study=rel) if a["status"] in LIVE), None)


def execute(study, cli: str = "claude", dry_run: bool = False, resume: bool = False,
            model: str | None = None) -> dict:
    """Spawn the lead agent with the execution brief and the proposal's total budget (refused unless the proposal
    is approved and unchanged, and no lead of this study is live). `dry_run`: the brief and the tree, nothing spawned.
    `resume`: after the earlier agents ended (on a person request, say), a lead for the unfinished branches only,
    with the budget left, told what finished and what a person decided since."""
    from ..agents.spawn import AgentError, spawn
    node = require(study)
    raw = node.payload["proposal"]
    root, rel = _where(study.folder)
    budget = {"usd": float(raw["budget"]["usd"]), "minutes": float(raw["budget"]["minutes"])}
    plan = None
    if resume:
        if not study.record.find("action", verb="execute"):
            raise StudyError("there is no execution to resume", "no lead was spawned for this proposal yet",
                             f"start it: openlab propose execute {rel}")
        plan = resume_plan(study)
        if not plan["left"]:
            raise StudyError("every branch is finished", "; ".join(plan["finished"]),
                             f"read the results: openlab propose status {rel}")
        if plan["budget"]["usd"] <= 0.01 or plan["budget"]["minutes"] <= 0.5:
            raise StudyError(f"the proposal's budget is spent: ${plan['spent']['usd']:g} of ${budget['usd']:g}, "
                             f"{plan['spent']['minutes']:g} of {budget['minutes']:g} min",
                             "a resumed run gets the total less what the earlier agents spent",
                             "a person raises budget.usd or budget.minutes in proposal.toml and records it with "
                             f"`openlab propose amend {rel} --person --reason ...`, then resumes")
        budget = plan["budget"]
        why = "; ".join(plan["blocked"]) or "their leads ended"
        text = brief(study, only=plan["left"], preamble=RESUME.format(
            rel=rel, why=why, finished=", ".join(plan["finished"]) or "none",
            decisions="; ".join(plan["decisions"]) or "none", usd=budget["usd"], minutes=budget["minutes"]))
    else:
        text = brief(study)
    tree = planned_tree(study)
    if dry_run:
        return {"dry_run": True, "brief": text, "tree": tree, "proposal": node.id, "budget": budget,
                "resume": plan}
    lead = live_lead(study)
    if lead is not None:
        raise StudyError(f"the proposal is already executing: lead {lead['id']} is {lead['status']}",
                         "one lead per study", f"follow it with `openlab agent log {lead['id']} -f`, or stop it first")
    from .branches import make
    copies = [make(study, b["id"]) for b in raw["branches"] if KIND_OF[b["role"]] == "experiment"
              and (plan is None or b["id"] in plan["left"])]           # each experiment branch edits its own copy
    try:
        a = spawn(root, text, role="lead", study=rel, question=raw["question"], cli=cli, budget=budget, model=model)
    except AgentError as e:
        raise StudyError(e.error, e.meaning, e.try_) from None
    asked = _asked_model(root, a)
    study._act("execute", {"proposal": node.id, "lead": a["id"], "cli": cli, "budget": a["budget"], "model": asked,
                           "copies": {c["branch"]: c["dir"] for c in copies},
                           **({"resume": {"left": plan["left"], "finished": plan["finished"]}} if plan else {})},
               "agent", lambda: {})
    return {"dry_run": False, "lead": a["id"], "cli": cli, "budget": a["budget"], "proposal": node.id, "tree": tree,
            "resume": plan, "copies": [c["dir"] for c in copies], "model": asked}


# --- progress -----------------------------------------------------------------------------------------------------
def _minutes(a: dict) -> float:
    from datetime import datetime
    from ..agents.store import FINAL, age
    try:
        start = datetime.fromisoformat(a["created"])
        end = datetime.fromisoformat(a["updated"]) if a["status"] in FINAL else None
    except ValueError:
        return 0.0
    span = ((end - start).total_seconds() if end else age(a["created"])) / 60
    return max(0.0, span - float(a["spent"].get("paused_minutes") or 0))


def _team(top: list, kids: dict) -> dict:
    team, todo = {a["id"]: a for a in top}, list(top)
    while todo:
        for c in kids.get(todo.pop()["id"], []):
            if c["id"] not in team:
                team[c["id"]] = c
                todo.append(c)
    return team


def _rule(s: dict, used: dict) -> str:
    if s["stop"]:
        return f"fired: {s['stop'].payload['args']['reason']}"
    over = [k for k, v in used.items() if v >= 1]
    if over:
        return f"budget spent ({', '.join(over)}): the branch should stop"
    near = [k for k, v in used.items() if v >= 0.9]
    return f"near its budget ({', '.join(near)})" if near else "not fired"


def status(study) -> dict:
    """Progress per branch of the approved proposal: its state, its agents (from the control store: those spawned
    for it, `question` = "<branch id>: ...", and those that leased its item, with their descendants) and their state,
    $ / minutes / trials spent of its budget, its hypotheses tested, killed and untouched (and by whom), its dead
    ends, and the stop rule's state."""
    from ..agents.store import FINAL, LIVE, Store
    rec = study.record
    node = approved(rec)
    if node is None:
        raise StudyError("the proposal is not approved", fix="openlab propose lint shows the draft")
    raw = node.payload["proposal"]
    root, rel = _where(study.folder)
    store = Store(root) if (root / ".openlab" / "control.db").exists() else None
    agents = store.agents() if store else []
    inbox = store.inbox(None) if store else []
    kids: dict = {}
    for a in agents:
        kids.setdefault(a["parent"], []).append(a)
    holders: dict = {}
    for t in rec.find("action", verb="take"):
        if t.payload["outcome"].get("ok"):
            holders.setdefault(t.payload["args"].get("item"), set()).add(t.session)
    runs: dict = {}
    for r in rec.find("run", segment="research"):
        runs.setdefault(r.session, set()).add(r.payload["key"])
    from .frontier import leases
    leased = leases(study)
    rows, totals = [], {"usd": 0.0, "trials": 0, "tokens": 0}
    counts = {"tested": 0, "killed": 0, "untouched": 0}
    verdicts = {"supported": 0, "refuted": 0, "inconclusive": 0}
    for s in branch_states(study):
        b = s["branch"]
        top = [a for a in agents if a["study"] == rel and (a["question"] or "").startswith(f"{b['id']}:")]
        top += [a for a in agents if a["id"] in holders.get(s["item"], set()) and a not in top]
        team = _team(top, kids)
        paid = [a for a in team.values() if a["kind"] != "claude-subagent"]       # a session's spend holds its subs
        usd = round(sum(a["spent"].get("usd") or 0 for a in paid), 4)
        tokens = sum((a["spent"].get("tokens_in") or 0) + (a["spent"].get("tokens_out") or 0) for a in paid)
        sessions = set(team) | holders.get(s["item"], set())
        trials = len({k for sess in sessions for k in runs.get(sess, set())})
        minutes = round(max([_minutes(a) for a in top] or [0.0]), 1)
        bud = b["budget"]
        used = {"usd": usd / bud["usd"] if bud.get("usd") else 0.0,
                "minutes": minutes / bud["minutes"] if bud.get("minutes") else 0.0,
                "trials": trials / bud["trials"] if bud.get("trials") else 0.0}
        live = [a for a in team.values() if a["status"] in LIVE]
        for h in s["hypotheses"]:
            counts[h["state"]] += 1
            if h["verdict"] in verdicts:
                verdicts[h["verdict"]] += 1
        totals = {"usd": totals["usd"] + usd, "trials": totals["trials"] + trials, "tokens": totals["tokens"] + tokens}
        holder = leased.get(s["item"])
        state = s["state"] if s["state"] != "open" else "working" if live else "held" if holder else \
            "unassigned" if not team else "idle"
        asked = [i for i in inbox if i["agent"] in team]
        blocked = None
        if state == "idle" and asked:                   # its agents ended on a person request
            i = asked[-1]
            state = "blocked"
            if i["status"] == "pending":
                fix = (f"a person decides it: `openlab inbox approve {i['id']} --reason ...` (or deny), then "
                       f"`openlab propose execute {rel} --resume`")
            else:
                fix = f"`openlab propose execute {rel} --resume` (a lead for the unfinished branches)"
            blocked = {"item": i["id"], "kind": i["kind"], "status": i["status"], "agent": i["agent"],
                       "reason": f"{i['agent']} ended on inbox #{i['id']} ({i['kind']}, {i['status']}): {i['reason']}",
                       "fix": fix}
        rows.append({"id": b["id"], "role": b["role"], "question": b["question"], "question_node": s["question"],
                     "item": s["item"], "state": state, "label": STATE_LABELS.get(state, state.upper()),
                     "budget": bud,
                     "spent": {"usd": usd, "minutes": minutes, "trials": trials, "tokens": tokens},
                     "agents": [{"id": a["id"], "role": a["role"], "kind": a["kind"], "status": a["status"],
                                 "parent": a["parent"], "usd": round(a["spent"].get("usd") or 0, 4),
                                 "tokens": (a["spent"].get("tokens_in") or 0) + (a["spent"].get("tokens_out") or 0),
                                 "minutes": round(_minutes(a), 1)} for a in team.values()],
                     "hypotheses": s["hypotheses"], "dead_ends": [d for h in s["hypotheses"] for d in h["dead_ends"]],
                     "stop_rule": b["stop"], "kill": b["kill"], "stop_state": _rule(s, used),
                     "live": [a["id"] for a in live if a["status"] not in FINAL],
                     "held_by": holder and {"agent": holder["agent"], "expires": holder["expires"]},
                     "blocked": blocked and blocked["reason"], "blocked_by": blocked})
    lead = [a for a in agents if a["role"] == "lead" and a["study"] == rel]
    for a in lead:                                              # the lead's own spend counts against the total
        totals["usd"] += a["spent"].get("usd") or 0
        totals["tokens"] += (a["spent"].get("tokens_in") or 0) + (a["spent"].get("tokens_out") or 0)
    first = rec.find("proposal")[0]
    return {"proposal": node.id, "title": raw["title"], "question": raw["question"], "approved_at": first.at,
            "approved_by": first.session, "amendments": len(rec.find("proposal")) - 1, "budget": raw["budget"],
            "spent": {**totals, "usd": round(totals["usd"], 4)}, "hypotheses": counts, "verdicts": verdicts,
            "lead": [{"id": a["id"], "status": a["status"], "usd": round(a["spent"].get("usd") or 0, 4),
                      "minutes": round(_minutes(a), 1)} for a in lead],
            "branches": rows, "trials_used": schema.trials_used(rec)}
