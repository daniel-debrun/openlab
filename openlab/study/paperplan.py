"""Paper -> checked code from one command: `openlab replicate PAPER`.

The person gives the paper (an arXiv id, a DOI or a PDF) and nothing else. openlab fetches it into a new folder
(paper/paper.pdf, its page texts in paper/pages.txt, paper/source.json) and a planning agent drafts
`replication.toml`: what the paper claims and which numbers will be checked (each quoted from its page), the
methods (the paper's own and its baselines), the data and protocol, the metric, and a tolerance justified from the
paper. openlab checks the draft (`problems`: every quote is on its page and writes its number, the metric is one
openlab scores, the tolerance has a reason, the budget fits the cap) and renders replication.md.

* `approve` (a person's act, signed) builds the study from the plan: the paper pinned, the target and its tolerance
  written before any run, each claim recorded as evidence quoted from its page, and the approval node with the
  plan's sha.
* `build` spawns the builder agent with the plan's budget: baselines first (fidelity), then the paper's claims
  (replication, gated by the fidelities), the equations bound to the code (the equation gate).
* `report` writes README.md and results/ from the record: the verdicts are computed, never typed.
"""

from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Optional

try:
    import tomllib
except ModuleNotFoundError:                                  # Python 3.10
    import tomli as tomllib

from ..harness import load as load_harness
from . import schema
from .errors import StudyError
from .record import canonical

FILE, MD = "replication.toml", "replication.md"
PDF, PAGES, SOURCE = "paper/paper.pdf", "paper/pages.txt", "paper/source.json"
MODES = ("static", "temporal")
ROLES = ("baseline", "proposed")
KINDS = ("replication", "reproduction")
TOP = ("paper", "title", "summary", "kind", "code", "mode", "model", "methods", "metric", "tolerance", "data", "claims",
       "budget", "risks", "unresolved")
METHOD = ("name", "label", "role", "page", "description")
TOLERANCE = ("value", "max", "justification", "page", "quote")
DATA = ("name", "source", "licence", "protocol", "page", "quote")
CLAIM = ("id", "method", "value", "page", "section", "setting", "quote", "unit", "header")
BUDGET = ("usd", "minutes", "trials", "runs", "run_minutes", "parallel", "estimate")
FIT = 0.8                   # the planned compute may take this share of the build's minutes; the rest is the agent's
PLANNER_USD, PLANNER_MINUTES = 3.0, 30
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,39}")


def _p(where: str, error: str, meaning: str = "", fix: str = "") -> dict:
    return {"where": where, "error": f"{FILE}: `{where}` {error}" if where else f"{FILE}: {error}",
            "meaning": meaning, "fix": fix or f"edit {FILE}"}


def _text(v) -> bool:
    return isinstance(v, str) and v.strip() != ""


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# --- the paper, fetched once into the folder ---------------------------------------------------------------------
def fetch(src: str, folder: Path) -> dict:
    """Pin the paper `src` names and keep it in `folder`: paper/paper.pdf, paper/pages.txt (each page as laid out,
    under a `=== page N ===` line) and paper/source.json ({key, kind, id, url, n_pages})."""
    from ..sources import papers
    from .replication import target_id
    kind, ident, data, path = target_id(src)
    url = path.as_uri() if path else ""
    if data is None:
        urls = [f"https://arxiv.org/pdf/{ident}"] if kind == "arxiv" else []
        if kind == "doi":
            try:
                w = papers.openalex_work(f"doi:{ident}") or {}
                urls = [u for u in (w.get("pdf"), w.get("arxiv") and f"https://arxiv.org/pdf/{w['arxiv']}") if u]
            except Exception:                           # noqa: BLE001 -- the refusal below says what to do
                urls = []
        failed = []
        for u in urls:
            try:
                data, url = papers.pdf(u), u
                break
            except Exception as e:                      # noqa: BLE001 -- the reply says which link failed and why
                failed.append(f"{u}: {e}")
        if data is None:
            raise StudyError(f"no open-access PDF of {kind}:{ident} could be fetched", "; ".join(failed) or
                             "the index lists no PDF link for it", "download the PDF and give its path instead")
    (folder / "paper").mkdir(parents=True, exist_ok=True)
    (folder / PDF).write_bytes(data)
    layout = papers.pdf_pages(data, layout=True)
    (folder / PAGES).write_text("".join(f"=== page {i} ===\n{t.rstrip()}\n\n" for i, t in enumerate(layout, 1)),
                                encoding="utf-8")
    meta = {"key": f"{kind}:{ident}", "kind": kind, "id": ident, "url": url, "n_pages": len(layout)}
    (folder / SOURCE).write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return meta


def source(folder) -> dict:
    try:
        return json.loads((Path(folder) / SOURCE).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise StudyError(f"{folder} has no {SOURCE}", "the paper is fetched into the folder first",
                         "openlab replicate PAPER [FOLDER]") from None


def pages(folder) -> tuple:
    """(reading-order pages, laid-out pages) of the folder's PDF, as a quote is checked against them."""
    from ..sources import papers
    data = (Path(folder) / PDF).read_bytes()
    return papers.pdf_pages(data), papers.pdf_pages(data, layout=True)


def on_page(texts: tuple, page: int, quote: str) -> bool:
    i = page - 1
    q = schema._flat(quote)
    return any(0 <= i < len(t) and q in schema._flat(t[i]) for t in texts)


def _pages_with(texts: tuple, quote: str) -> list:
    q = schema._flat(quote)
    return [i + 1 for i in range(len(texts[0])) if any(i < len(t) and q in schema._flat(t[i]) for t in texts)]


# --- the plan ------------------------------------------------------------------------------------------------------
def path(folder) -> Path:
    return Path(folder) / FILE


def load(folder) -> dict:
    p = path(folder)
    try:
        parsed = tomllib.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise StudyError(f"no {FILE} in {folder}", "the plan is drafted first", "openlab replicate PAPER [FOLDER]") \
            from None
    except tomllib.TOMLDecodeError as e:
        raise StudyError(f"{FILE} does not parse", str(e), "fix the TOML syntax") from None
    return _coded(json.loads(canonical(parsed)), Path(folder))


def machine() -> dict:
    """{cores, memory}: what the planner scales the paper's timings to (the container's limits when it sets them,
    openlab.study.machine)."""
    from . import machine as _machine
    total = _machine.memory()[1]
    gb = total / 2 ** 30 if total else 0.0
    return {"cores": _machine.cores(), "memory": f"{gb:.0f} GB" if gb else "an unknown amount"}


def own(raw: dict) -> bool:
    """Whether the plan's metric is the study's own (metric.better set; its function in metrics.py)."""
    return isinstance(raw.get("metric"), dict) and "better" in raw["metric"]


def _coded(raw: dict, folder: Path) -> dict:
    """The plan with its own metric's code_sha (metrics.py's sha256, written by openlab): the plan's sha, and so the
    person's approval, covers the metric's code."""
    from .metrics import STUDY_FILE
    if not own(raw):
        return raw
    met = {k: v for k, v in raw["metric"].items() if k != "code_sha"}
    if (folder / STUDY_FILE).is_file():
        met["code_sha"] = hashlib.sha256((folder / STUDY_FILE).read_bytes()).hexdigest()
    return {**raw, "metric": met}


def metric_code(folder, raw: dict) -> str:
    """metrics.py as the plan's own metric has it ("" when the metric is one of openlab's)."""
    from .metrics import STUDY_FILE
    p = Path(folder) / STUDY_FILE
    return p.read_text(encoding="utf-8") if own(raw) and p.is_file() else ""


def sha(raw: dict) -> str:
    return hashlib.sha256(canonical(raw).encode("utf-8")).hexdigest()


def skeleton(meta: dict, usd: float = 5.0, minutes: float = 120.0) -> str:
    return f'''# The replication plan of {meta["key"]}, drafted from the paper and approved by a person
# (`openlab replicate approve FOLDER`). Every number to check is quoted from its page: openlab checks the quote is on
# that page of paper/paper.pdf and writes the number. `openlab replicate lint FOLDER` lists what is still wrong.

paper = "{meta["key"]}"        # pinned by openlab: do not edit
title = ""                     # the paper's title
summary = ""                   # its contribution in two or three sentences
kind = "replication"           # replication (rebuilt from the text) | reproduction (from the authors' code)
code = ""                      # the authors' code release, if any (a reproduction needs it)
mode = "static"                # static: rows, a fixed test split or k-fold | temporal: forecasts at decision times
# model = ""                   # with several proposed methods: the one openlab runs as model.py's Model
risks = []                     # what may keep the numbers from replicating (data access, compute, missing details)
unresolved = []                # what the paper leaves open that the person should know before approving

# One table per method: the baselines the paper compares against, and its own method(s) (role proposed; the first
# proposed is the model openlab runs, any other is run beside it, and each proposed method's numbers are claims).
[[methods]]
name = ""                      # a Python class name, e.g. EMOS_gl
label = ""                     # as the paper prints it in its tables, e.g. "EMOS-gl"
role = "baseline"              # baseline | proposed
page = 1                       # where the paper defines it
description = ""               # what it is, in a line, with the equation numbers it implements

[metric]
name = ""                      # the paper's headline metric, as openlab names it (openlab study --help lists them)
paper_unit = ""                # "percent" when the paper prints shares such as coverage as percentages
# When openlab scores none the paper uses, the study defines its own: set better = "lower" (or "higher"), a
# description, and reads = ["point"] (the predictions it reads), and write def <name>(y, pred) -> float in
# metrics.py beside this file. Its code is part of the plan the person approves.

[tolerance]                    # how close our number must be to the paper's, in the metric's own unit
value = 0.0
max = 0.0                      # the most a justified fidelity may allow (at least value)
justification = ""             # why: the spread the paper reports over seeds, the rounding of its tables, test size
page = 0                       # optional: the page the justification quotes
quote = ""

[data]
name = ""
source = ""                    # where the data comes from (URL, release, how it is built)
licence = ""
protocol = ""                  # the split, the evaluation set and the preprocessing, as the paper states them
page = 0                       # optional: the page the protocol quotes
quote = ""

# One table per number to check: each baseline's (fidelity: the rebuild is faithful) and the paper's own (the claim).
[[claims]]
id = ""                        # short, e.g. t2-emos
method = ""                    # a method's name above
value = 0.0                    # the number as the paper prints it
page = 1
section = ""                   # e.g. "Table 2"
setting = ""                   # the column or condition, e.g. "test 2016"
quote = ""                     # the row or sentence as the page has it, with the method's label and the number
# header = ""                  # a table whose methods are its COLUMNS: the header line with the method's label, as
#                              # laid out (paper/pages.txt); openlab checks the number sits under that label's column.
#                              # Several header levels: a list of the lines, and the method's label by level,
#                              # label = "SC2 / γ=1−α" (the group, then the column under it)

[budget]                       # for the build, at most the workspace cap
usd = {usd:g}
minutes = {minutes:g}
trials = 20
runs = 0                       # the counted runs the protocol needs (seeds x datasets x settings)
run_minutes = 0.0              # one run's wall minutes (every method, every fold) on this machine, estimated
parallel = 1                   # runs going at once (each then gets a share of the cores)
estimate = ""                  # where run_minutes comes from: the paper's timings, the data size, the cores
'''


def _check_quote(texts: tuple, where: str, page, quote, need: bool) -> list:
    if not _text(quote):
        return [_p(f"{where}.quote", "is empty", "the number is checked against the page it is quoted from",
                   "quote the row or sentence as the page has it (paper/pages.txt)")] if need else []
    if len(schema._flat(quote)) < schema.MIN_QUOTE:
        return [_p(f"{where}.quote", f"is shorter than {schema.MIN_QUOTE} characters", "a short quote is found almost "
                   "anywhere", "quote the whole row or sentence")]
    if type(page) is not int or not 1 <= page <= len(texts[0]):
        return [_p(f"{where}.page", f"is not a page of the paper (1..{len(texts[0])})")]
    if on_page(texts, page, quote):
        return []
    elsewhere = _pages_with(texts, quote)
    return [_p(f"{where}.quote", f"is not on page {page}", "openlab looks for it, whitespace aside, in the page's text "
               "in reading order and as laid out", f"it is on page {elsewhere[0]}" if elsewhere else
               "copy it from paper/pages.txt exactly (a table row as one line)")]


def _header(c: dict) -> str:
    """A claim's header as the record keeps it: its lines (a string, or a list of the table's header lines, one per
    level) joined by newlines."""
    h = c.get("header")
    return "\n".join(str(x) for x in h) if isinstance(h, list) else (h if isinstance(h, str) else "")


def problems(raw: dict, folder, limit: dict | None = None) -> list:
    """Every refusal of the plan (where, error, meaning, fix), checked against the paper in `folder`."""
    from .metrics import METRICS
    import difflib
    out = []
    for k in raw:
        if k not in TOP:
            near = difflib.get_close_matches(k, TOP, n=1)
            out.append(_p(k, "is not a known key", f"known keys: {', '.join(TOP)}",
                          f"did you mean `{near[0]}`?" if near else ""))
    meta = source(folder)
    texts = pages(folder)
    if raw.get("paper") != meta["key"]:
        out.append(_p("paper", f"must be {meta['key']}", "the paper the folder was made for", f'paper = "{meta["key"]}"'))
    for k in ("title", "summary"):
        if not _text(raw.get(k)):
            out.append(_p(k, "is empty"))
    if raw.get("kind", "replication") not in KINDS:
        out.append(_p("kind", f"is not one of {', '.join(KINDS)}"))
    elif raw.get("kind") == "reproduction" and not _text(raw.get("code")):
        out.append(_p("code", "is empty", "a reproduction runs the authors' code", "name the release, or kind = "
                      '"replication"'))
    if raw.get("mode") not in MODES:
        out.append(_p("mode", f"is not one of {', '.join(MODES)}"))
    methods = raw.get("methods") if isinstance(raw.get("methods"), list) else []
    names = []
    for i, m in enumerate(methods):
        w = f"methods[{i}]"
        out += [_p(f"{w}.{k}", "is not a known key", f"known keys: {', '.join(METHOD)}") for k in m if k not in METHOD]
        n = m.get("name", "")
        if not (isinstance(n, str) and n.isidentifier()) or n in ("Model", "features", "label"):
            out.append(_p(f"{w}.name", f"`{n}` is not a class name", fix="e.g. EMOS_gl"))
        elif n in names:
            out.append(_p(f"{w}.name", f"`{n}` names two methods"))
        names.append(n)
        if not _text(m.get("label")):
            out.append(_p(f"{w}.label", "is empty", "the method as the paper prints it; each claim's quote shows it"))
        if m.get("role") not in ROLES:
            out.append(_p(f"{w}.role", f"is not one of {', '.join(ROLES)}"))
        if not _text(m.get("description")):
            out.append(_p(f"{w}.description", "is empty"))
    proposed = [m["name"] for m in methods if m.get("role") == "proposed"]
    if not proposed:
        out.append(_p("methods", "name no proposed method", "the paper's own method is the model; the others are its "
                      "baselines", "give the paper's method role = \"proposed\" (several, when the paper proposes a "
                      "family: the first is the model, each one's numbers are claims)"))
    met = raw.get("metric") or {}
    if own(raw):
        out += _own_metric(met, folder)
    elif met.get("name") not in METRICS:
        near = difflib.get_close_matches(str(met.get("name", "")), list(METRICS), n=1)
        out.append(_p("metric.name", f"`{met.get('name', '')}` is not a metric openlab scores",
                      f"one of {', '.join(METRICS)}", f"did you mean `{near[0]}`?" if near else
                      "pick the closest, and say so under unresolved"))
    if met.get("paper_unit", "") not in ("", *schema.UNITS):
        out.append(_p("metric.paper_unit", f"is not one of {', '.join(schema.UNITS)}"))
    tol = raw.get("tolerance") or {}
    v, mx = tol.get("value"), tol.get("max", tol.get("value"))
    if not (_num(v) and v > 0):
        out.append(_p("tolerance.value", "must be above 0", "how close our number must be to the paper's, fixed "
                      "before any run"))
    elif not (_num(mx) and mx >= v):
        out.append(_p("tolerance.max", "must be at least tolerance.value"))
    if len(str(tol.get("justification", "")).strip()) < 20:
        out.append(_p("tolerance.justification", "is missing", "the tolerance is held to every comparison, so its "
                      "reason is part of the plan", "say why: the spread over seeds the paper reports, the rounding "
                      "of its tables, the size of its test set"))
    out += _check_quote(texts, "tolerance", tol.get("page"), tol.get("quote"), False)
    data = raw.get("data") or {}
    for k in ("name", "source", "protocol"):
        if not _text(data.get(k)):
            out.append(_p(f"data.{k}", "is empty"))
    out += _check_quote(texts, "data", data.get("page"), data.get("quote"), False)
    claims = raw.get("claims") if isinstance(raw.get("claims"), list) else []
    if not claims:
        out.append(_p("claims", "is empty", "the numbers the replication checks", "add a [[claims]] table per number"))
    labels = {m.get("name"): m.get("label", "") for m in methods}
    ids = []
    for i, c in enumerate(claims):
        w = f"claims[{i}]"
        out += [_p(f"{w}.{k}", "is not a known key", f"known keys: {', '.join(CLAIM)}") for k in c if k not in CLAIM]
        cid = c.get("id", "")
        if not (isinstance(cid, str) and _ID.fullmatch(cid)) or cid in ids:
            out.append(_p(f"{w}.id", f"`{cid}` is not a fresh short id", fix="e.g. t2-emos"))
        ids.append(cid)
        if c.get("method") not in names:
            out.append(_p(f"{w}.method", f"`{c.get('method', '')}` is not a method above"))
        if not _num(c.get("value")):
            out.append(_p(f"{w}.value", "is not a number"))
        if not _text(c.get("section")):
            out.append(_p(f"{w}.section", "is empty", fix='e.g. "Table 2"'))
        if c.get("unit", "") not in ("", *schema.UNITS):
            out.append(_p(f"{w}.unit", f"is not one of {', '.join(schema.UNITS)}"))
        bad = _check_quote(texts, w, c.get("page"), c.get("quote"), True)
        out += bad
        if bad or not _text(c.get("quote")):
            continue
        if _num(c.get("value")) and Decimal(repr(c["value"])) not in schema._numbers(c["quote"]):
            out.append(_p(f"{w}.value", f"{c['value']:g} is not written in the quote", "each number is read from its "
                          "quote", "copy the number as printed"))
        lab = labels.get(c.get("method"), "")
        head = _header(c)
        if head:
            hb = [] if len(schema._flat(head)) >= schema.MIN_QUOTE else [_p(f"{w}.header", f"is shorter than "
                  f"{schema.MIN_QUOTE} characters", "a short header is found almost anywhere", "give every header "
                  "line of the table (a list, one per level)")]
            for k, line in enumerate(x for x in head.split("\n") if x.strip()):
                if not hb and type(c.get("page")) is int and not on_page(texts, c["page"], line):
                    hb.append(_p(f"{w}.header", f"line {k + 1} is not on page {c['page']}", fix="copy it from "
                                 "paper/pages.txt exactly"))
            out += hb
            why = "" if hb or not (_text(lab) and _num(c.get("value"))) else \
                schema.column_check(texts[1][c["page"] - 1], c["quote"], head, lab, c["value"])
            if why:
                out.append(_p(f"{w}.header", why, "a table whose methods are its columns shows the method in its "
                              "header: the number must sit under that column", "quote the header line and the row "
                              "as paper/pages.txt lays them out, each as one line"))
        elif _text(lab) and schema._flat(lab).casefold() not in schema._flat(c["quote"]).casefold():
            out.append(_p(f"{w}.quote", f"does not show the method's label `{lab}`", "the evidence names the method "
                          "the number belongs to", "quote the row with its label, or, when the methods are the "
                          "table's columns, give `header`: the header line with the label (openlab checks the number "
                          "is under it)"))
    if len(proposed) > 1 and raw.get("model") not in proposed:
        out.append(_p("model", f"is not one of the proposed methods ({', '.join(proposed)})", "with several proposed "
                      "methods, the plan names the one openlab runs as model.py's Model; the others run beside it",
                      f'model = "{proposed[0]}"'))
    if proposed and claims and not any(c.get("method") in proposed for c in claims):
        out.append(_p("claims", f"check no number of a proposed method ({', '.join(proposed)})",
                      "the paper's own result is what the replication tests"))
    bud = raw.get("budget") or {}
    for k in ("usd", "minutes"):
        if not (_num(bud.get(k)) and bud[k] > 0):
            out.append(_p(f"budget.{k}", "must be above 0"))
        elif limit and bud[k] > limit[k]:
            out.append(_p(f"budget.{k}", f"{bud[k]:g} is above the workspace cap {limit[k]:g}",
                          "openlab.toml [agents] cap_usd / cap_minutes", f"at most {limit[k]:g}"))
    if "trials" in bud and not (type(bud["trials"]) is int and bud["trials"] > 0):
        out.append(_p("budget.trials", "must be a whole number above 0"))
    out += [_p(f"budget.{k}", "is not a known key", f"known keys: {', '.join(BUDGET)}") for k in bud if k not in BUDGET]
    out += _compute(bud)
    return out


def _compute(bud: dict) -> list:
    """The plan's compute against its minutes: `runs` counted runs of `run_minutes` each, `parallel` at once, must fit
    FIT of the build's minutes (the rest is the agent's reading, coding and checking)."""
    out = []
    if not (type(bud.get("runs")) is int and bud["runs"] > 0):
        out.append(_p("budget.runs", "must say how many counted runs the protocol needs (a whole number)",
                      "seeds x datasets x settings, as the claims need them, plus a few for failures",
                      "runs = 40"))
    if not (_num(bud.get("run_minutes")) and bud["run_minutes"] > 0):
        out.append(_p("budget.run_minutes", "must estimate the wall minutes of one counted run (every method, "
                      "every fold) on this machine", "from the paper's reported timings, scaled by the data size "
                      "and this machine's cores; say how in budget.estimate", "run_minutes = 15"))
    par = bud.get("parallel", 1)
    if not (type(par) is int and par > 0):
        out.append(_p("budget.parallel", "must be a whole number above 0", "the runs going at once (each then gets "
                      "a share of the cores, so its run_minutes is the share's)"))
    if not str(bud.get("estimate") or "").strip():
        out.append(_p("budget.estimate", "must say where run_minutes comes from", "the paper's timing table, a "
                      "measured run of the same methods, the data size and the cores; what would make it longer"))
    if out or not _num(bud.get("minutes")):
        return out
    need = bud["runs"] * bud["run_minutes"] / par
    if need > FIT * bud["minutes"]:
        out.append(_p("budget.minutes", f"{bud['runs']} runs x {bud['run_minutes']:g} min / {par} at once = "
                      f"{need:.0f} min of compute, more than {FIT:.0%} of the {bud['minutes']:g} min budget",
                      "fewer seeds or settings (and a tolerance that says so), more minutes, or more runs at once",
                      f"minutes = {need / FIT:.0f}"))
    if type(bud.get("trials")) is int and bud["runs"] > bud["trials"]:
        out.append(_p("budget.trials", f"{bud['trials']} is fewer than the {bud['runs']} runs the plan needs",
                      "every run is a counted trial", f"trials = {bud['runs'] + max(5, bud['runs'] // 4)}"))
    return out


def _own_metric(met: dict, folder) -> list:
    """The refusals of a metric the study defines itself: a free function name, a direction, what it reads, a
    description, and the function in metrics.py."""
    import ast
    from .metrics import KINDS, METRICS, STUDY_FILE
    out, name = [], met.get("name", "")
    if not (isinstance(name, str) and name.isidentifier()) or (name in METRICS and not METRICS[name].study):
        out.append(_p("metric.name", f"`{name}` cannot name the study's own metric", "a Python function name that is "
                      "not one of openlab's metrics", f"e.g. {name}_study" if _text(name) else "e.g. mse_avg"))
    if met.get("better") not in ("lower", "higher"):
        out.append(_p("metric.better", "is not lower or higher", "the metric's direction decides every verdict"))
    reads = met.get("reads", ["point"])
    if not (isinstance(reads, list) and reads and set(reads) <= set(KINDS)):
        out.append(_p("metric.reads", f"is not a list of {', '.join(KINDS)}"))
    if not _text(met.get("description")):
        out.append(_p("metric.description", "is empty", "the person approves the metric by its description and its "
                      "code", "say what it computes, as the paper defines it"))
    path = Path(folder) / STUDY_FILE
    if not path.is_file():
        return out + [_p("metric", f"is the study's own, and there is no {STUDY_FILE}", "its code is part of the "
                         "plan the person approves", f"write def {name}(y, pred) -> float in {path}")]
    try:
        tree = ast.parse(path.read_bytes())
    except SyntaxError as e:
        return out + [_p("metric", f"{STUDY_FILE} does not parse: {e}")]
    if not any(isinstance(n, ast.FunctionDef) and n.name == name for n in tree.body):
        out.append(_p("metric.name", f"{STUDY_FILE} defines no function {name}", fix=f"def {name}(y, pred): ..."))
    return out


def warnings(raw: dict) -> list:
    methods = raw.get("methods") or []
    checked = {c.get("method") for c in raw.get("claims") or []}
    out = [f"baseline {m['name']} has no number to check: its rebuild cannot be shown faithful, and the claims it "
           "should gate are inconclusive without it" for m in methods if m.get("role") == "baseline"
           and m.get("name") not in checked]
    if not any(m.get("role") == "baseline" for m in methods):
        out.append("no baseline: nothing shows the rebuild faithful before the paper's own number is compared")
    out += [f"unresolved: {u}" for u in raw.get("unresolved") or []]
    return out


def _primary(raw: dict) -> str:
    """The proposed method openlab runs as model.py's Model (`model`, which a plan with several proposed methods
    names; else the one proposed); any other is an arm beside it."""
    proposed = [m["name"] for m in raw["methods"] if m["role"] == "proposed"]
    return raw["model"] if raw.get("model") in proposed else proposed[0]


def _role(raw: dict, method: str) -> str:
    return next((m["role"] for m in raw["methods"] if m["name"] == method), "baseline")


def _arm(raw: dict, method: str) -> str:
    return "model" if method == _primary(raw) else f"model:{method}"


def _check(raw: dict, method: str) -> str:
    """How a claim of `method` is checked: a proposed method's number is a replication, a baseline's a fidelity."""
    return "replication" if _role(raw, method) == "proposed" else "fidelity"


def render(raw: dict, status: str, code: str = "") -> str:
    met, tol, data, bud = raw.get("metric") or {}, raw.get("tolerance") or {}, raw.get("data") or {}, \
        raw.get("budget") or {}
    roles = {m.get("name"): m for m in raw.get("methods") or []}
    out = [f"# Replication plan: {raw.get('title') or raw.get('paper')}", "", f"*{status}* · {raw.get('paper')}"
           + (f" · code: {raw['code']}" if raw.get("code") else ""), "", raw.get("summary", ""), "",
           "## Methods", "", "| Method | As printed | Role | Page | What it is |", "|---|---|---|---|---|"]
    out += [f"| {m.get('name')} | {m.get('label')} | {m.get('role')} | {m.get('page')} | {m.get('description')} |"
            for m in raw.get("methods") or []]
    out += ["", "## Numbers to check", "", "| Claim | Method | Paper | Where | Checked as | Quote |",
            "|---|---|---|---|---|---|"]
    for c in raw.get("claims") or []:
        role = (roles.get(c.get("method")) or {}).get("role")
        q = str(c.get("quote", ""))
        out.append(f"| {c.get('id')} | {c.get('method')} | {c.get('value')} | {c.get('section')}, p.{c.get('page')}"
                   + (f", {c['setting']}" if c.get("setting") else "") + " | "
                   + ("replication (the paper's claim)" if role == "proposed" else "fidelity (baseline rebuilt)")
                   + f" | {(q[:70] + '…') if len(q) > 70 else q} |")
    out += ["", "## Metric and tolerance", "",
            f"{met.get('name')}" + (f" (the paper prints it in {met['paper_unit']})" if met.get("paper_unit") else "")
            + f": ours must be within **{tol.get('value')}** of the paper's (up to {tol.get('max', tol.get('value'))} "
            "with a justification), fixed before any run.", "", f"Why: {tol.get('justification', '')}"]
    if own(raw):
        out += ["", f"{met.get('name')} is the study's own metric ({met.get('better')} is better, reads "
                f"{', '.join(met.get('reads', ['point']))}): {met.get('description', '')}", "",
                f"Its code, metrics.py (sha {str(met.get('code_sha', 'missing'))[:12]}), is part of what you approve. "
                "openlab runs it only in the study's sandbox, and from the first run an edit of it is your amendment:",
                "", "```python", code.rstrip(), "```"]
    out += ["", "## Data and protocol", "", f"{data.get('name')}: {data.get('source')}"
            + (f" (licence {data['licence']})" if data.get("licence") else ""), "", data.get("protocol", "")]
    out += ["", "## Budget", "", f"${bud.get('usd', 0):g}, {bud.get('minutes', 0):g} min, {bud.get('trials', 20)} "
            "trials for the build"]
    if raw.get("risks"):
        out += ["", "## Risks", ""] + [f"- {r}" for r in raw["risks"]]
    if raw.get("unresolved"):
        out += ["", "## Unresolved", ""] + [f"- {u}" for u in raw["unresolved"]]
    return "\n".join(out) + "\n"


def write_md(folder, raw: dict, status: str) -> Path:
    p = Path(folder) / MD
    p.write_text(render(raw, status, metric_code(folder, raw)), encoding="utf-8")
    return p


def approved(folder):
    """The person's approval action of the folder's study, or None (no study yet, or not approved)."""
    if not (Path(folder) / "study.toml").is_file():
        return None
    from . import open as open_study
    rec = open_study(folder).record
    return next((n for n in reversed(rec.find("action", verb="replication_plan"))
                 if n.payload["outcome"].get("ok")), None)


def _amended_code(folder, raw: dict, node) -> bool:
    """Whether the plan differs from its approval only in its metric's code_sha, and a person's signed amendment of
    the study sealed metrics.py as it is now: that edit was approved once, on the record, and is not asked again."""
    old = node.payload["args"]["plan"]
    if not own(raw) or not own(old) or sha({**raw, "metric": {**raw["metric"], "code_sha": old["metric"].get(
            "code_sha")}}) != node.payload["args"]["sha"]:
        return False
    return any(new == raw["metric"].get("code_sha") for _, new in _code_amendments(folder, node))


def _code_amendments(folder, node) -> list:
    """(id, new code_sha) of each person's amendment of metrics.py since the plan's approval."""
    from . import open as open_study
    out = []
    for a in open_study(folder).record.find("amendment"):
        ch = a.payload.get("changes")
        if a.payload.get("person") and a.at >= node.at and isinstance(ch, dict):
            out += [(a.id, v.get("new")) for k, v in ch.items()
                    if k.startswith("metrics.") and k.endswith(".code_sha") and isinstance(v, dict)][:1]
    return out


def _changed(folder, raw: dict, node) -> bool:
    return node.payload["args"]["sha"] != sha(raw) and not _amended_code(folder, raw, node)


def lint(folder) -> dict:
    raw = load(folder)
    probs = problems(raw, folder, cap(folder))
    node = approved(folder)
    if node is not None and _changed(folder, raw, node):
        probs.insert(0, _p("", "changed since its approval", f"approved as {node.id} with sha "
                           f"{node.payload['args']['sha'][:12]}", "put it back, or a person approves the change: "
                           f"openlab replicate approve {folder} (the paper, metric, tolerance and mode stay fixed)"))
    status = f"approved as {node.id}" if node is not None else "draft: not approved"
    md = write_md(folder, raw, status)
    return {"ok": not probs, "problems": probs, "warnings": warnings(raw), "md": str(md), "sha": sha(raw),
            "status": status}


def cap(folder) -> dict:
    from ..agents.store import config
    cfg = config(workspace(folder)[0])
    return {"usd": float(cfg["cap_usd"]), "minutes": float(cfg["cap_minutes"])}


def workspace(folder) -> tuple:
    """(workspace root, folder relative to it): the nearest openlab.toml above, else the folder's parent (where a
    study's agents are kept, before and after study.toml exists)."""
    from ..agents.store import MARKER
    folder = Path(folder).resolve()
    root = next((d for d in folder.parents if (d / MARKER).is_file()), folder.parent)
    return root, str(folder.relative_to(root))


# --- start: fetch and draft ---------------------------------------------------------------------------------------
PLANNER = load_harness().text("workflows/replicate_planner.md")


def start(src: str, folder=None, usd: float | None = None, minutes: float | None = None) -> dict:
    """The folder for the paper `src` names (default: its id, in the current directory) with the paper fetched and
    the plan's skeleton."""
    from .replication import target_id
    kind, ident, _, _ = target_id(src)
    folder = Path(folder or re.sub(r"[^\w.-]+", "_", ident)).resolve()
    if path(folder).exists() or (folder / "study.toml").exists():
        raise StudyError(f"{folder} already holds a replication", fix=f"openlab replicate status {folder}, or choose a "
                         "new folder")
    folder.mkdir(parents=True, exist_ok=True)
    meta = fetch(src, folder)
    limit = cap(folder)
    usd = min(float(usd) if usd else 5.0, limit["usd"])
    minutes = min(float(minutes) if minutes else 120.0, limit["minutes"])
    path(folder).write_text(skeleton(meta, usd, minutes), encoding="utf-8")
    write_md(folder, load(folder), "draft: not approved")
    return {"folder": str(folder), "paper": meta, "budget": {"usd": usd, "minutes": minutes}}


CONTINUE = load_harness().text("workflows/replicate_continue.md")


def draft(folder, cli: str = "claude", usd: float | None = None, model: str | None = None, note: str = "") -> dict:
    """Spawn the planning agent (role planner, $3 by default) that fills replication.toml until lint is clean. A
    plan already drafted (an earlier planner ran out, or the person wants it revised) is continued, not redone."""
    from ..agents.spawn import AgentError, spawn
    from .metrics import METRICS
    usd = PLANNER_USD if usd is None else float(usd)
    if usd <= 0:
        raise StudyError(f"the planner's budget must be above $0 (asked ${usd:g})")
    root, rel = workspace(folder)
    meta, limit = source(folder), cap(folder)
    if approved(folder) is not None:
        raise StudyError("the plan is already approved", "an approved plan is the one built",
                         f"openlab replicate status {folder}")
    from ..harness.versions import workflow
    text = workflow(root, "replicate_planner").format(folder=Path(folder).resolve(), key=meta["key"], metrics=", ".join(METRICS),
                          cap_usd=limit["usd"], cap_minutes=limit["minutes"], **machine(), fit=FIT)
    if load(folder) != load_skeleton(folder):
        lt = lint(folder)
        text = workflow(root, "replicate_continue").format(folder=Path(folder).resolve(), lint="\n".join(
            f"- {p['error']}" for p in lt["problems"]) or "- clean") + text
    if note.strip():
        text = f"The person who approves the plan says:\n{note.strip()}\nAct on it first.\n\n" + text
    try:
        a = spawn(root, text, role="planner", study=rel, question=f"plan the replication of {meta['key']}", cli=cli,
                  budget={"usd": usd, "minutes": PLANNER_MINUTES}, model=model)
    except AgentError as e:
        raise StudyError(e.error, e.meaning, e.try_) from None
    return {"id": a["id"], "cli": cli, "budget": a["budget"]}


# --- approve: the study, built from the plan ----------------------------------------------------------------------
def load_skeleton(folder) -> dict:
    """The plan as `start` writes it, before any planner (so `draft` knows whether to continue)."""
    meta = source(folder)
    raw = tomllib.loads(skeleton(meta))
    bud = (load(folder).get("budget") or {})
    return json.loads(canonical({**raw, "budget": {**raw["budget"], **{k: bud[k] for k in ("usd", "minutes")
                                                                         if k in bud}}}))


def summary(raw: dict) -> str:
    tol, met = raw["tolerance"], raw["metric"]
    roles = {m["name"]: m["role"] for m in raw["methods"]}
    return "\n".join([
        f"{raw['title']} ({raw['paper']})",
        "methods: " + ", ".join(f"{m['name']} ({m['role']})" for m in raw["methods"]),
        f"claims: {sum(roles[c['method']] == 'proposed' for c in raw['claims'])} of the paper's own, "
        f"{sum(roles[c['method']] == 'baseline' for c in raw['claims'])} baseline numbers",
        f"metric {met['name']}, tolerance {tol['value']:g} (up to {tol.get('max', tol['value']):g}): "
        f"{tol['justification']}",
        f"build budget ${raw['budget']['usd']:g}, {raw['budget']['minutes']:g} min",
        *([f"metric {met['name']} is the study's own ({met['better']} is better): metrics.py, sha "
           f"{met.get('code_sha', 'missing')[:12]}, shown in full in replication.md"] if own(raw) else []),
        *[f"unresolved: {u}" for u in raw.get("unresolved") or []]])


def approve(folder, reason: str, by: str) -> dict:
    """The person `by` approves replication.toml (the CLI showed it and they signed): the study is made from it
    (the paper pinned, the target and tolerance written before any run, each claim recorded as evidence quoted from
    its page) and the approval recorded with the plan's sha (a person's node). A failure part way (a quote the record
    refuses, the index unreachable) removes what it made, so the plan can be fixed and approved again."""
    from .personkey import current
    if current() is None:                               # refused before anything is made
        raise StudyError("approving the plan is a person's act and must be signed with the person key",
                         "this process holds no unlocked person key", f"run `openlab replicate approve {folder}` in "
                         "your own terminal")
    folder = Path(folder).resolve()
    raw = load(folder)
    probs = problems(raw, folder, cap(folder))
    if probs:
        raise StudyError(f"the plan is not ready: {len(probs)} problem{'s' if len(probs) != 1 else ''}",
                         "\n".join(p["error"] for p in probs[:8]), f"openlab replicate lint {folder}")
    if (folder / "study.toml").exists():
        return _reapprove(folder, raw, reason, by)
    before = _entries(folder)
    try:
        return _approve(folder, raw, reason, by)
    except BaseException:                               # nothing half-made: the person can fix the plan and approve again
        _rollback(folder, before)
        raise


def _entries(folder: Path) -> set:
    return {p.relative_to(folder) for p in folder.rglob("*")}


def _rollback(folder: Path, before: set) -> None:
    """Remove what a failed approval made in `folder` (the study's files and record), keeping all that was there."""
    import shutil
    for rel in sorted(_entries(folder) - before, key=lambda r: len(r.parts)):
        p = folder / rel
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists() or p.is_symlink():
            p.unlink(missing_ok=True)


def _approve(folder: Path, raw: dict, reason: str, by: str) -> dict:
    from . import replication
    from ..agents.identity import acting_as
    meta = source(folder)
    primary = _primary(raw)
    methods = [m for m in raw["methods"] if m["name"] != primary] + \
        [m for m in raw["methods"] if m["name"] == primary]          # the last is model.py's Model
    tol, met = raw["tolerance"], raw["metric"]
    s = replication.new(meta["key"].split(":", 1)[1], folder, raw["mode"], [m["name"] for m in methods], met["name"],
                        float(tol["value"]), float(tol.get("max", tol["value"])), raw.get("kind", "replication"),
                        raw.get("code", ""), met.get("paper_unit", ""), pdf_file=folder / PDF, pdf_url=meta["url"],
                        own={k: met[k] for k in ("better", "reads", "description") if k in met} if own(raw) else None)
    src = s.record.last("source").id
    labels = {m["name"]: m["label"] for m in raw["methods"]}
    evidence = {}
    for c in raw["claims"]:
        fields = {"method": labels[c["method"]], met["name"]: c["value"], "setting": c.get("setting", "")}
        e = s.extract(src, c["page"], c["quote"], fields, "", c["section"],
                      {met["name"]: c["unit"]} if c.get("unit") else None, _header(c))
        evidence[c["id"]] = e["id"]
    with acting_as(by):                                 # the person's node, signed by the CLI's signer
        s._act("replication_plan", {"sha": sha(raw), "reason": reason, "plan": raw, "evidence": evidence,
                                    "target": s.record.last("target").id}, "person", lambda: {})
    act = s.record.find("action", verb="replication_plan")[-1]
    write_md(folder, raw, f"approved as {act.id} ({act.at[:16]}, by {by})")
    return {"id": act.id, "sha": sha(raw), "evidence": evidence, "target": s.record.last("target").id,
            "folder": str(folder)}


#: what the first approval fixed before any run: a later approval may change the claims, roles, budget and notes,
#: never these
FIXED = ("paper", "kind", "code", "mode", "metric", "tolerance")


def _reapprove(folder: Path, raw: dict, reason: str, by: str) -> dict:
    """A person approves a changed plan for a study already made from an earlier one: the paper, metric, tolerance
    and mode stay as first approved (they were fixed before any run), and so does the model (the first proposed
    method); claims kept unchanged keep their evidence, new or changed ones are quoted again from their pages."""
    from . import open as open_study
    from ..agents.identity import acting_as
    node = approved(folder)
    if node is None:
        raise StudyError("this folder holds a study no plan was approved for", fix="start a new folder")
    if sha(raw) == node.payload["args"]["sha"]:
        raise StudyError("the plan is already approved as it is", fix=f"openlab replicate status {folder}")
    old = node.payload["args"]["plan"]
    moved = [k for k in FIXED if canonical(old.get(k)) != canonical(raw.get(k))]
    if moved:
        raise StudyError(f"{', '.join(moved)} cannot change after the first approval",
                         "they were fixed before any run, so no result can have shaped them",
                         "put them back; a different paper, metric or tolerance is a new replication in a new folder")
    if _primary(raw) != _primary(old):
        raise StudyError(f"the model is {_primary(old)}", "model.py's Model was made from the first approval",
                         f'model = "{_primary(old)}"')
    s = open_study(folder)
    src, met = s.record.last("source").id, raw["metric"]
    labels, prev = {m["name"]: m["label"] for m in raw["methods"]}, {c["id"]: c for c in old["claims"]}
    evidence, quoted = {}, []
    for c in raw["claims"]:
        if c["id"] in node.payload["args"]["evidence"] and canonical(prev.get(c["id"])) == canonical(c):
            evidence[c["id"]] = node.payload["args"]["evidence"][c["id"]]
            continue
        fields = {"method": labels[c["method"]], met["name"]: c["value"], "setting": c.get("setting", "")}
        evidence[c["id"]] = s.extract(src, c["page"], c["quote"], fields, "", c["section"],
                                      {met["name"]: c["unit"]} if c.get("unit") else None, _header(c))["id"]
        quoted.append(c["id"])
    with acting_as(by):
        s._act("replication_plan", {"sha": sha(raw), "reason": reason, "plan": raw, "evidence": evidence,
                                    "target": s.record.last("target").id, "amends": node.id}, "person", lambda: {})
    act = s.record.find("action", verb="replication_plan")[-1]
    write_md(folder, raw, f"approved as {act.id} ({act.at[:16]}, by {by}), amending {node.id}")
    return {"id": act.id, "sha": sha(raw), "evidence": evidence, "target": s.record.last("target").id,
            "folder": str(folder), "amends": node.id, "quoted": quoted}


def require(folder) -> tuple:
    """(the study, the approved plan, the approval action), or a refusal."""
    from . import open as open_study
    node = approved(folder)
    if node is None:
        raise StudyError("the plan is not approved", "a person approves the plan before it is built",
                         f"openlab replicate approve {folder}")
    raw = load(folder)
    if _changed(folder, raw, node):
        raise StudyError(f"{FILE} changed since its approval", f"approved with sha {node.payload['args']['sha'][:12]}",
                         f"put it back, or a person approves the change: openlab replicate approve {folder}")
    return open_study(folder), raw, node


# --- build: the builder agent -------------------------------------------------------------------------------------
BUILD = load_harness().text("workflows/replicate_build.md")


def brief(study, raw: dict, node) -> str:
    ev = node.payload["args"]["evidence"]
    proposed = _primary(raw)
    methods = "\n".join(f"- {m['name']} ({m['role']}, printed \"{m['label']}\", p.{m['page']}): {m['description']}"
                        for m in raw["methods"])
    claims = "\n".join(f"- {c['id']}: {c['method']} = {c['value']} ({c['section']}, p.{c['page']}"
                       + (f", {c['setting']}" if c.get("setting") else "") + f"), evidence {ev[c['id']]}, arm "
                       f"{_arm(raw, c['method'])}: "
                       + ("replication, gated by the baselines' fidelities" if _check(raw, c["method"]) == "replication"
                          else "fidelity") for c in raw["claims"])
    from ..harness.versions import workflow
    tol, bud = raw["tolerance"], raw["budget"]
    return workflow(workspace(study.folder)[0], "replicate_build").format(title=raw["title"], key=raw["paper"], folder=study.folder, md=MD, file=FILE, aid=node.id,
                        sha=node.payload["args"]["sha"][:12], tol=tol["value"], metric=raw["metric"]["name"],
                        max=tol.get("max", tol["value"]), methods=methods, claims=claims,
                        data_name=raw["data"]["name"], data_source=raw["data"]["source"],
                        protocol=raw["data"]["protocol"], proposed=proposed, usd=bud["usd"], minutes=bud["minutes"],
                        trials=bud.get("trials", 20))


def live_builder(folder) -> dict | None:
    from ..agents.store import LIVE, Store
    root, rel = workspace(folder)
    if not (root / ".openlab" / "control.db").exists():
        return None
    return next((a for a in Store(root).find(role="replicator", study=rel) if a["status"] in LIVE), None)


RESUME = load_harness().text("workflows/replicate_resume.md")


def _earlier(folder) -> list:
    from ..agents.store import Store
    root, rel = workspace(folder)
    if not (root / ".openlab" / "control.db").exists():
        return []
    return Store(root).find(role="replicator", study=rel)


def _worked(store, a: dict) -> float:
    """Minutes from a builder's start to its last act (a lost builder's machine may have been off long after); a
    live one's, to now."""
    from datetime import datetime, timezone
    from ..agents.store import LIVE
    try:
        start = datetime.fromisoformat(a["created"])
        if a["status"] in LIVE:
            return max(0.0, (datetime.now(timezone.utc) - start).total_seconds() / 60)
        evs = [e for e in store.events(a["id"], n=5) if e["kind"] not in ("lost", "end")] or [{"at": a["created"]}]
        return max(0.0, (datetime.fromisoformat(evs[-1]["at"]) - start).total_seconds() / 60)
    except (ValueError, TypeError):
        return 0.0


def pace(s, raw: dict, folder) -> Optional[dict]:
    """The build's measured pace against the plan's compute: the counted research runs so far, their median wall
    minutes, and whether the plan's runs left fit the minutes left (at `parallel` at once). None before a run, or
    for a plan that predates budget.runs."""
    from statistics import median
    from ..agents.store import Store
    bud = raw.get("budget") or {}
    done = [r for r in s.runs()["runs"] if r.get("segment") == "research" and r.get("seconds")]
    if not done or not bud.get("runs"):
        return None
    mins = median(r["seconds"] for r in done) / 60
    par = bud.get("parallel", 1) or 1
    root, _ = workspace(folder)
    store = Store(root)
    left = float(bud["minutes"]) - sum(_worked(store, a) for a in _earlier(folder))
    need = max(0, bud["runs"] - len(done)) * mins / par
    return {"done": len(done), "planned": bud["runs"], "run_minutes": round(mins, 1),
            "planned_run_minutes": bud.get("run_minutes"), "parallel": par, "need": round(need, 1),
            "left": round(left, 1), "fits": need <= left}


def resume_preamble(s, raw: dict, earlier: list) -> tuple:
    """(the preamble a resumed builder reads first, the budget left)."""
    from ..agents.store import Store
    root, _ = workspace(s.folder)
    store = Store(root)

    def worked(a: dict) -> float:
        return _worked(store, a)
    usd = round(float(raw["budget"]["usd"]) - sum(float((a.get("spent") or {}).get("usd", 0)) for a in earlier), 2)
    minutes = round(float(raw["budget"]["minutes"]) - sum(worked(a) for a in earlier), 1)
    why = "; ".join(f"{a['id']} {a['status']}" + (" (its process ended without reporting: the machine stopped?)"
                                                   if a["status"] == "lost" else "") for a in earlier)
    rep = s.replications()
    record = (f"{len(s.runs()['runs'])} run(s); "
              + ("; ".join(f"fidelity {f['id']} {f['arm']}: paper {f['paper']:g}, ours {f['ours']:.4g} "
                           f"({'within' if f['passed'] else 'outside'} {f['tolerance']:g})" for f in rep["fidelity"])
                 or "no fidelity yet")
              + "".join(f"; claim {r['id']}: {r['verdict']}" for r in rep["replications"]))
    notes = [e for a in earlier for e in store.events(a["id"], n=2, kinds=("note",))]
    lines = "\n".join(f"- {e['agent']}: {(e['payload'].get('text') or e['summary'])[:600]}" for e in notes) or "- none"
    from ..harness.versions import workflow
    return workflow(root, "replicate_resume").format(why=why, folder=s.folder, record=record, notes=lines, usd=usd, minutes=minutes), \
        {"usd": usd, "minutes": minutes}


def build(folder, cli: str = "claude", model: str | None = None, dry_run: bool = False) -> dict:
    """Spawn the builder (role replicator) with the brief and the plan's budget; refused unless approved and
    unchanged, and no builder of this study is live. After earlier builders ended (the budget, a crash, the machine
    off) the new one resumes: told what is on record and their last notes, with the budget they left."""
    from ..agents.spawn import AgentError, spawn
    s, raw, node = require(folder)
    text = brief(s, raw, node)
    budget = {"usd": float(raw["budget"]["usd"]), "minutes": float(raw["budget"]["minutes"])}
    earlier = [a for a in _earlier(folder) if a["status"] not in ("running", "idle", "waiting", "paused")]
    if earlier:
        pre, budget = resume_preamble(s, raw, earlier)
        text = pre + text
        if budget["usd"] <= 0.01 or budget["minutes"] <= 0.5:
            raise StudyError(f"the build's budget is spent: ${budget['usd']:g}, {budget['minutes']:g} min left",
                             "a resumed builder gets the plan's budget less what the earlier ones spent",
                             "a person raises [budget] in replication.toml and approves it again (a new plan)")
    if dry_run:
        return {"dry_run": True, "brief": text, "budget": budget, "resumed": [a["id"] for a in earlier]}
    live = live_builder(folder)
    if live is not None:
        raise StudyError(f"the build is running: {live['id']} is {live['status']}", "one builder per replication",
                         f"follow it: openlab agent log {live['id']} -f")
    root, rel = workspace(folder)
    try:
        a = spawn(root, text, role="replicator", study=rel, question=f"build the replication of {raw['paper']}",
                  cli=cli, budget=budget, model=model)
    except AgentError as e:
        raise StudyError(e.error, e.meaning, e.try_) from None
    s._act("replicate_build", {"plan": node.id, "agent": a["id"], "cli": cli, "budget": a["budget"]}, "agent",
           lambda: {})
    return {"dry_run": False, "agent": a["id"], "cli": cli, "budget": a["budget"], "folder": str(s.folder),
            "resumed": [x["id"] for x in earlier]}


# --- status and report ---------------------------------------------------------------------------------------------
def _agents(folder) -> list:
    from ..agents.store import Store
    root, rel = workspace(folder)
    if not (root / ".openlab" / "control.db").exists():
        return []
    return [{"id": a["id"], "role": a["role"], "status": a["status"], "usd": float((a.get("spent") or {}).get("usd", 0)),
             "budget": a.get("budget") or {}} for a in Store(root).agents() if a["study"] == rel]


def _equations(s) -> dict:
    try:
        r = s.paper_lint()
    except StudyError as e:
        return {"checked": None, "reason": e.error, "bound": 0, "failing": []}
    bound = [b for b in r["bindings"] if b["macro"].startswith("\\labeq") and not b["error"]]
    failing = [f["message"] for f in r["findings"] if f["rule"] == "eq.gate"]
    return {"checked": True, "bound": len(bound), "failing": failing}


def status(folder) -> dict:
    folder = Path(folder).resolve()
    out = {"folder": str(folder), "paper": source(folder), "agents": _agents(folder)}
    node = approved(folder)
    if node is None:
        lt = lint(folder)
        return {**out, "stage": "draft", "ok": lt["ok"], "problems": len(lt["problems"]), "md": lt["md"]}
    from . import open as open_study
    s = open_study(folder)
    rep = s.replications()
    latest = lambda rows: list({r["evidence"]: r for r in rows}.values())      # noqa: E731 -- each claim's last check
    return {**out, "stage": "approved", "approval": node.id, "trials": len(s.runs()["runs"]),
            "rebuilt": rep["rebuilt"], "fidelity": latest(rep["fidelity"]), "replications": latest(rep["replications"]),
            "superseded": len(rep["fidelity"]) + len(rep["replications"]) - len(latest(rep["fidelity"]))
            - len(latest(rep["replications"])),
            "report": (folder / "results" / "report.json").is_file(), "pace": pace(s, load(folder), folder)}


def _fmt(v) -> str:
    return "" if v is None else f"{v:.4g}" if isinstance(v, float) else str(v)


_DEVIATIONS = re.compile(r"^## Protocol and deviations\s*$(.*?)(?=^## |\Z)", re.M | re.S)


def report(folder) -> dict:
    """README.md and results/report.json from the record: per claim the paper's number, ours, and the verdict
    openlab computed; whether the rebuild is faithful; the equations checked; the trials run."""
    s, raw, node = require(folder)
    folder = s.folder
    rep, eqs = s.replications(), _equations(s)
    runs = s.runs()
    by_ev = {e: cid for cid, e in node.payload["args"]["evidence"].items()}
    claim = {c["id"]: c for c in raw["claims"]}
    rows = []
    latest = {r["evidence"]: r for r in rep["fidelity"] + rep["replications"]}   # each claim's last check
    for r in sorted(latest.values(), key=lambda r: "verdict" not in r):
        c = claim.get(by_ev.get(r["evidence"]), {})
        rows.append({"claim": c.get("id", r["evidence"]), "method": c.get("method", r["method"]),
                     "section": r["section"], "page": r["page"], "setting": c.get("setting", ""),
                     "check": "replication" if "verdict" in r else "fidelity", "paper": r.get("paper"),
                     "ours": r.get("ours"), "lo": r.get("lo"), "hi": r.get("hi"),
                     "verdict": r.get("verdict") or ("faithful" if r.get("passed") else "not faithful"),
                     "rule_now": r.get("rule_now", ""),
                     "id": r["id"], "result": r["result"], "results": r.get("results") or [r["result"]],
                     "group": r.get("group", ""), "in_sample": bool(r.get("in_sample"))})
    done = {x["claim"] for x in rows}
    rows += [{"claim": c["id"], "method": c["method"], "section": c["section"], "page": c["page"],
              "setting": c.get("setting", ""), "check": _check(raw, c["method"]), "paper": c["value"], "ours": None, "lo": None, "hi": None,
              "verdict": "not run", "id": None, "result": None, "results": [], "group": ""}
             for c in raw["claims"] if c["id"] not in done]
    tol, met = raw["tolerance"], raw["metric"]
    data = {"paper": raw["paper"], "title": raw["title"], "approval": node.id, "plan_sha": node.payload["args"]["sha"],
            "code_amendments": [i for i, _ in _code_amendments(folder, node)], "metric": met["name"], "tolerance": tol["value"], "max_tolerance": tol.get("max", tol["value"]),
            "rebuilt": rep["rebuilt"], "claims": rows, "equations": eqs, "trials": runs["trials"]}
    (folder / "results").mkdir(exist_ok=True)
    (folder / "results" / "report.json").write_text(json.dumps(data, indent=1, default=str) + "\n", encoding="utf-8")
    old = (folder / "README.md").read_text(encoding="utf-8") if (folder / "README.md").is_file() else ""
    m = _DEVIATIONS.search(old)
    deviations = m[1].strip() if m and m[1].strip() and not m[1].strip().startswith("The split chosen and why") \
        else "None recorded."
    (folder / "README.md").write_text(_readme(raw, data, deviations), encoding="utf-8")
    return {"folder": str(folder), "readme": str(folder / "README.md"), "report": str(folder / "results/report.json"),
            **data}


def _readme(raw: dict, d: dict, deviations: str) -> str:
    rebuilt = {True: "**faithful**: every baseline was rebuilt within tolerance of the paper's number",
               False: "**not faithful**: a baseline missed the paper's number, so the claims it gates are inconclusive",
               None: "not shown yet: no baseline has been compared"}[d["rebuilt"]]
    eq = d["equations"]
    eqline = (f"{eq['bound']} equation(s) bound to the code and checked on probe inputs"
              + (f"; {len(eq['failing'])} disagree: " + "; ".join(eq["failing"][:3]) if eq["failing"] else ", all agree")
              ) if eq["checked"] else f"not checked ({eq.get('reason', '')})"
    out = [f"# {raw['title']}", "", f"{raw['paper']} · rebuilt and checked with openlab", "", raw["summary"], "",
           "## Verdicts", "", "| Claim | Method | Where | Paper | Ours | Check | Verdict |", "|---|---|---|---|---|---|---|"]
    for r in d["claims"]:
        iv = f" [{_fmt(r['lo'])}, {_fmt(r['hi'])}]" if r["lo"] is not None else ""
        iv += f" (mean of {len(r['results'])} runs)" if len(r.get("results") or []) > 1 else ""
        iv += f" ({r['group']})" if r.get("group") else ""
        iv += " (in-sample)" if r.get("in_sample") else ""
        out.append(f"| {r['claim']} | {r['method']} | {r['section']}, p.{r['page']}"
                   + (f", {r['setting']}" if r["setting"] else "") + f" | {_fmt(r['paper'])} | {_fmt(r['ours'])}{iv} | "
                   f"{r['check']} | {r['verdict']}"
                   + (f" (recorded; the corrected rule gives {r['rule_now']})" if r.get('rule_now') else "") + " |")
    if any(r.get("in_sample") for r in d["claims"]):
        out += ["", "**In-sample:** the numbers marked in-sample were scored on the rows that trained the model "
                "(split.scheme = insample), as the paper's protocol does. They are optimistic: a model scores the rows "
                "it learned better than new ones. A held-out run, if recorded, is the honest number."]
    out += ["", f"- Rebuild: {rebuilt}.",
            f"- Tolerance: {d['tolerance']:g} on {d['metric']} (up to {d['max_tolerance']:g} with a justification), "
            f"fixed before any run: {raw['tolerance']['justification']}",
            f"- Equations: {eqline}.",
            f"- Trials: {d['trials']} run(s), every one on record; a result chosen among them says so.",
            f"- Plan: approved by a person ({d['approval']}, sha {d['plan_sha'][:12]}); see {MD}."
            + (f" Its metric's code (metrics.py) was since amended by a person: {', '.join(d['code_amendments'])}."
               if d.get("code_amendments") else ""),
            "", "Each verdict is computed by openlab from the paper's quoted number (checked on its page) and our run; "
            "none is typed by an agent. A claim is inconclusive while a baseline that gates it is not faithful.",
            "", "## Layout", "",
            "- `methods/`: one file per method, each citing the page and equations it implements",
            "- `model.py`: the entry point openlab runs (`Model` is the paper's method; the others are its baselines)",
            "- `recipes/make_data.py` and `data/`: the data, as the paper's release gives it",
            "- `paper/`: the paper (`paper.pdf`, `pages.txt`) and `equations.tex`, each equation bound to its function",
            "- `results/report.json`: the table above, with the record ids behind each row",
            f"- `study.toml` and `{FILE}`: the protocol and the approved plan",
            "- `.openlab/`: the record every number above is computed from",
            "", "## Run it again", "", "```bash",
            "openlab study reproduce . --all     # re-runs every recorded run and checks the numbers match",
            "openlab study replications .        # the verdicts, from the record",
            "openlab study verify .              # the record's hash chain and signatures",
            "openlab study bundle . --out replication.tar   # a signed bundle another machine can reproduce", "```",
            "", "## Protocol and deviations", "", deviations, ""]
    return "\n".join(out)
