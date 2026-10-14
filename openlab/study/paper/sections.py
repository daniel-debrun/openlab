"""The paper's LaTeX: each bound macro rendered from its node, and the sections openlab generates from the record
(Data, Methods, Literature, Results, Limitations, Reproducibility, Checklist), the bibliography and the waivers.
Every number here comes from the record, so none of it is linted."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from pathlib import Path

from .tex import esc
from .checklists import CHECKLISTS, MET, NA, NOT_MET, TO_STATE
from ..claims import at_risk
from ..errors import StudyError
from ..metrics import METRICS
from ..schema import current_level, current_verdict, retracted, run_of as _run
from ..ledger import shared
from ..show import selected
from .bindings import runs, value
from .lint import approved

#: the code that draws each figure kind, by version: a figure records the version that drew it
FIGURES = {"spec_curve": 1, "reliability": 1}
TABLE_ROWS = 40                 # rows of a generated table before it counts the rest


@dataclass
class Ctx:
    study: object
    digits: int
    build: Path
    primary: list               # the result nodes that are current: the latest research result, then holdout results
    equations: dict             # {macro: why it failed, "" when it matches}
    checklist: str = "neurips"  # a list of checklists.CHECKLISTS
    figures: dict = field(default_factory=dict)     # {result id: {file, sha256, generator}}
    cited: dict = field(default_factory=dict)       # {source id: node}, in citation order
    tables: set = field(default_factory=set)        # results already tabled (and labelled)

    @property
    def rec(self):
        return self.study.record

    @property
    def spec(self):
        return self.study.spec


# --- values and tables -----------------------------------------------------------------------------------------
def num(v, digits: int) -> str:
    """A number as LaTeX math (no $): an int exactly, a float to `digits` significant digits."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return rf"\text{{{esc(v)}}}"
    if isinstance(v, int):
        return str(v)
    if not math.isfinite(v):
        return r"\text{n/a}"
    s = f"{v:.{digits}g}"
    if "e" in s:
        m, e = s.split("e")
        return rf"{m}\times 10^{{{int(e)}}}"
    return s


def show(v, digits: int) -> str:
    """A value as it prints: a number, an estimate with its interval as "v [lo, hi]", or text."""
    if isinstance(v, dict):
        main = v.get("value", v.get("diff"))
        if main is None:
            return r"\text{n/a}" if "reason" in v else "--"
        iv = rf"\ [{num(v['lo'], digits)},\ {num(v['hi'], digits)}]" if v.get("lo") is not None and v.get("hi") is not None else ""
        return rf"\ensuremath{{{num(main, digits)}{iv}}}"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return rf"\ensuremath{{{num(v, digits)}}}"
    return esc(v if not isinstance(v, bool) else str(v).lower())


def tabular(headers: list, rows: list, align: str = "") -> str:
    shown = rows[:TABLE_ROWS] + ([[rf"\emph{{{len(rows) - TABLE_ROWS} more rows}}"] + [""] * (len(headers) - 1)]
                                 if len(rows) > TABLE_ROWS else [])
    return (rf"\begin{{tabular}}{{{align or 'l' * len(headers)}}}\toprule" + "\n" + " & ".join(headers) + r" \\\midrule"
            + "\n" + "\n".join(" & ".join(r) + r" \\" for r in shown) + "\n" + r"\bottomrule\end{tabular}")


def _metric_head(name: str) -> str:
    m = METRICS.get(name)
    if m is not None and m.direction == "target" and m.nominal != "level":
        return esc(name) + rf" $\to {m.nominal:g}$"                     # best at its nominal value
    return esc(name) + {True: r" $\uparrow$", False: r" $\downarrow$"}.get(m.higher_is_better if m else None, "")


def table(ctx: Ctx, result) -> str:
    r"""\labtab: the result's metrics with intervals for the model and each baseline, then the model minus each
    baseline on the primary metric (paired), then the policy metrics when model.py acts."""
    p, d = result.payload, ctx.digits
    run = _run(ctx.rec, result)
    names = list(p["metrics"])
    rows = [["model"] + [show(p["metrics"][k], d) for k in names]]
    rows += [[esc(b)] + [show(s.get(k, {}), d) for k in names] for b, s in p.get("baselines", {}).items()]
    parts = [tabular([""] + [_metric_head(k) for k in names], rows, "l" + "r" * len(names))]
    if p.get("paired"):
        pr = [[esc(b), show({"diff": x.get("diff"), "lo": x.get("lo"), "hi": x.get("hi")}, d),
               show(x.get("p"), d) if x.get("p") is not None else "--", esc(x.get("better") or x.get("reason", ""))]
              for b, x in p["paired"].items()]
        metric = next(iter(p["paired"].values())).get("metric", names[0])
        parts.append(tabular([f"model $-$ baseline ({esc(metric)})", "difference [interval]", "$p$", "better"], pr,
                             "lrrl"))
    if p.get("policy"):
        pol = p["policy"]
        pn = list(pol.get("metrics", {}))
        prow = [["model"] + [show(pol["metrics"][k], d) for k in pn]]
        prow += [[esc(b)] + [show(s.get(k, {}), d) for k in pn] for b, s in pol.get("baselines", {}).items()]
        parts.append(tabular([f"policy ({pol.get('decisions')} decisions)"] + [esc(k) for k in pn], prow,
                             "l" + "r" * len(pn)))
    ev = ctx.spec.evaluate
    cap = (rf"Result \texttt{{{esc(result.id)}}}: {p.get('n', '?')} scored rows of the {esc(run.payload['segment'])} "
           rf"segment; run \texttt{{{esc(run.id)}}}, truncation audit {esc(run.payload['audit'].get('verdict'))}. "
           rf"Intervals: {esc(ev.get('interval', 'bootstrap'))}, $\alpha = {ev.get('alpha', 0.05)}$. "
           r"Arrows give each metric's better direction ($\to v$: best at $v$). " + _selected(ctx, run))
    label = "" if result.id in ctx.tables else rf"\label{{tab:{result.id}}}"
    ctx.tables.add(result.id)
    return (r"\begin{table}[htbp]\centering\small" + "\n" + "\n\n\\medskip\n".join(parts) + "\n"
            + rf"\caption{{{cap}}}{label}" + "\n" + r"\end{table}")


def _selected(ctx: Ctx, run) -> str:
    """"Selected from N research trials." (the study's trial count at build time); for a holdout run, the research
    trials its model was chosen on."""
    from ..ledger import shared
    n = shared(ctx.rec.folder, ctx.rec)["trials"]
    if run.payload["segment"] == "holdout":
        return f"Its model was chosen on {n} research trial{'s' if n != 1 else ''}."
    return f"Selected from {n} research trial{'s' if n != 1 else ''}."


# --- figures ---------------------------------------------------------------------------------------------------
def _primary(ctx: Ctx) -> str:
    return ctx.spec.evaluate["metrics"][0]


def _spec_curve(ctx: Ctx, result, ax) -> str:
    """The primary metric of every research run with a result, sorted, the bound result in black; the bound
    result's baselines as dashed lines."""
    rec, metric = ctx.rec, _primary(ctx)
    pts = []
    for run in rec.find("run", segment="research"):
        made = rec.links(run.id, rel="of", incoming=True)
        if made:
            res = rec.get(made[0][0])
            m = res.payload["metrics"].get(metric, {})
            if m.get("value") is not None:
                pts.append((m["value"], m.get("lo"), m.get("hi"), res.id == result.id))
    pts.sort(key=lambda t: t[0])
    for i, (v, lo, hi, is_model) in enumerate(pts, 1):
        err = [[v - lo], [hi - v]] if lo is not None and hi is not None else None
        ax.errorbar([i], [v], yerr=err, fmt="o", ms=5 if is_model else 3.5, color="black" if is_model else "0.6", lw=1, capsize=0)
    for b, s in result.payload.get("baselines", {}).items():
        bv = s.get(metric, {}).get("value")
        if bv is not None:
            ax.axhline(bv, color="0.35", lw=0.8, ls="--")
            ax.annotate(b, (len(pts) + 0.6, bv), fontsize=7, color="0.25", va="center", annotation_clip=False)
    ax.set_xlim(0.4, len(pts) + 0.6)
    ax.set_xticks(range(1, len(pts) + 1))
    ax.set_xlabel("research runs, sorted")
    ax.set_ylabel(metric)
    return (f"Specification curve: {metric} of each of the {len(pts)} research runs on record, sorted, with its "
            f"interval; the bound result in black, its baselines dashed.")


def _reliability(ctx: Ctx, result, ax) -> str:
    """Observed frequency against mean predicted probability in ten equal bins of the run's predictions."""
    import numpy as np
    import polars as pl
    run = _run(ctx.rec, result)
    P = pl.read_parquet(io.BytesIO(ctx.rec.blob(run.payload["predictions"])))
    if "pred" not in P.columns:
        raise StudyError(f"{run.id} has no probability column `pred`")
    P = P.drop_nulls(["y", "pred"])
    p, y = P["pred"].to_numpy(), P["y"].to_numpy()
    bins = np.clip((p * 10).astype(int), 0, 9)
    full = [k for k in range(10) if (bins == k).any()]
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--")
    ax.plot([p[bins == k].mean() for k in full], [y[bins == k].mean() for k in full], "o-", color="black", lw=1, ms=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency")
    return f"Reliability diagram of the {len(p)} scored predictions, in ten equal bins; the diagonal is perfect calibration."


def figure(ctx: Ctx, result) -> str:
    r"""\labfig: a reliability diagram for a probability prediction, else the specification curve, drawn from the
    record into build/figures/ as PDF; the figure and the version of the code that drew it are recorded."""
    from matplotlib.figure import Figure
    from ... import __version__
    kind = "reliability" if ctx.spec.evaluate.get("prediction") == "prob" else "spec_curve"
    name = f"figures/{result.id.replace(':', '-')}.pdf"
    if result.id not in ctx.figures:
        fig = Figure(figsize=(5.2, 3.0))
        ax = fig.subplots()
        caption = (_reliability if kind == "reliability" else _spec_curve)(ctx, result, ax)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="0.9", lw=0.5)
        fig.tight_layout()
        (ctx.build / "figures").mkdir(exist_ok=True)
        gen = f"openlab {__version__} paper.{kind} v{FIGURES[kind]}"
        fig.savefig(ctx.build / name, metadata={"Creator": gen, "CreationDate": None})
        sha = ctx.rec.put_blob((ctx.build / name).read_bytes())
        ctx.figures[result.id] = {"file": name, "sha256": sha, "generator": gen, "caption": caption}
    f = ctx.figures[result.id]
    return (r"\begin{figure}[htbp]\centering" + rf"\includegraphics[width=0.85\linewidth]{{{name}}}"
            + rf"\caption{{{esc(f['caption'])} Result \texttt{{{esc(result.id)}}}; drawn by {esc(f['generator'])}.}}"
            + rf"\label{{fig:{result.id}}}\end{{figure}}")


# --- claims, citations, equations ------------------------------------------------------------------------------
def claim(ctx: Ctx, node) -> str:
    r"""\labclaim: the statement with its state, its verdict now, its level now (reproduced, measured from the
    holdout, recorded from research) and whether a person approved it."""
    p = node.payload
    n = shared(ctx.rec.folder, ctx.rec)["trials"] if p.get("selected_from") else 0
    marks = [p["state"], current_verdict(ctx.rec, node), current_level(ctx.rec, node),
             selected(n) if n else "",
             "approved by a person" if approved(ctx.rec, node) else "not approved by a person"]
    return rf"\emph{{{esc(p['statement'])}}} \textup{{[{esc('; '.join(m for m in marks if m))}]}}"


def cite(ctx: Ctx, node) -> str:
    r"""\labcite: a natbib citation, flagged when the source is retracted."""
    ctx.cited.setdefault(node.id, node)
    return rf"\citep{{{node.id}}}" + (r" \textbf{[retracted]}" if retracted(ctx.rec, node) else "")


def render(ctx: Ctx, b) -> str:
    """One resolved binding as LaTeX."""
    m = b.macro
    if m.name == "labeq":
        return rf"\begin{{equation}}\label{{eq:{m.args[0]}}}{m.args[2]}\end{{equation}}"
    if m.name == "lab":
        return show(value(ctx.rec, m.args[0]), ctx.digits)
    return {"labtab": table, "labfig": figure, "labclaim": claim, "labcite": cite}[m.name](ctx, ctx.rec.get(b.node))


# --- generated sections ------------------------------------------------------------------------------------------
def _items(pairs) -> str:
    return r"\begin{description}\small" + "\n" + "\n".join(rf"\item[{esc(k)}] {v}" for k, v in pairs) + "\n" + r"\end{description}"


def _kv(d: dict) -> str:
    return "; ".join(f"{esc(k)} = {esc(', '.join(map(str, v)) if isinstance(v, list) else v)}"
                     for k, v in d.items()) or "--"


def data(ctx: Ctx) -> str:
    """A card per dataset: source, sha256, rows, time coverage, keys, licence and each column's type and missing
    values."""
    out = []
    for name in ctx.spec.data:
        n = ctx.rec.last("dataset", name=name)
        if n is None:
            out.append(rf"\paragraph{{{esc(name)}}} not materialised.")
            continue
        p, c = n.payload, n.payload["card"]
        facts = [("source", _kv(p["source"])), ("sha256", rf"\texttt{{{p['sha256'][:16]}}}"), ("rows", str(p["rows"]))]
        if c.get("time"):
            facts.append(("time", f"{esc(c['time']['column'])} from {esc(c['time']['min'])} to {esc(c['time']['max'])}"))
        if c.get("key"):
            facts.append(("key", f"{esc(', '.join(c['key']['columns']))} ({c['key']['distinct']} distinct)"))
        facts.append(("licence", esc(c.get("licence") or "not stated")))
        cols = [[rf"\texttt{{{esc(k)}}}", esc(t), str(c.get("missing", {}).get(k, ""))] for k, t in c["schema"].items()]
        out.append(rf"\paragraph{{{esc(name)}}} (\texttt{{{esc(n.id)}}})" + "\n" + _items(facts) + "\n"
                   + r"\begin{center}\small" + tabular(["column", "type", "missing"], cols, "llr") + r"\end{center}")
    return "\n\n".join(out)


def methods(ctx: Ctx) -> str:
    """The sealed spec as run: question, each section, the embargo rule, the primary metric; then the amendments."""
    rec, spec = ctx.rec, ctx.spec
    plan = rec.find("plan")[0]
    out = [rf"The plan was sealed on {esc(plan.at)} (\texttt{{{esc(plan.id)}}}, spec sha256 "
           rf"\texttt{{{plan.payload['sha'][:16]}}}) before any result. Its question: \emph{{{esc(spec.question)}}}"]
    pairs = []
    for name, ds in spec.raw.get("data", {}).items():
        pairs.append((f"data.{name}", _kv(ds)))
    for sec in ("target", "split", "evaluate", "budget", "audit", "sandbox"):
        if spec.raw.get(sec):
            pairs.append((sec, _kv(spec.raw[sec])))
    out.append(_items(pairs))
    if spec.mode == "temporal":
        out.append(r"Embargo: a decision at $t$ belongs to a segment only if $t + h$ is not after the segment's end, "
                   r"and a row decided at $t_i$ trains a refit at $s$ only if $t_i + h \le s$, with $h$ the horizon "
                   rf"({esc(spec.target.get('horizon'))}).")
    if spec.evaluate.get("metrics"):
        out.append(rf"The primary metric, fixed by the seal, is {esc(spec.evaluate['metrics'][0])}.")
    amends = rec.find("amendment")
    out.append(r"\paragraph{Amendments} " + ("None: the study ran as sealed." if not amends else ""))
    if amends:
        out.append(_items([(a.at, f"by {esc(a.actor)} ({esc(', '.join(a.payload['changes']))}): {esc(a.payload['reason'])}")
                           for a in amends]))
    return "\n\n".join(out)


def literature(ctx: Ctx) -> str:
    """When the study has a review: the PRISMA flow, the coverage statement, the extraction table and the
    comparison lattice with its empty cells."""
    from .. import evidence, literature as lit
    if not ctx.spec.review:
        return ""
    fl = lit.prisma(ctx.study)
    flow = [("identified", f"{fl['identified']} records ({_kv(fl['by_source'])})"),
            ("duplicates removed", str(fl["duplicates_removed"])), ("records screened", str(fl["screened"])),
            ("excluded", f"{fl['excluded']} ({_kv(fl['excluded_by_criterion'])})"), ("included", str(fl["included"])),
            ("waiting for a person", str(fl["awaiting_person"])), ("not screened", str(fl["not_screened"])),
            ("retracted", f"{fl['retracted']} ({len(fl['retracted_included'])} included)")]
    out = [r"\paragraph{Flow} The PRISMA flow, from the record:" + "\n" + _items(flow)]
    cov = ctx.rec.last("coverage")
    out.append(r"\paragraph{Coverage} " + (
        f"The search was judged {esc(cov.payload['verdict'])} against the sealed floor ({_kv(cov.payload['floor'])}): "
        f"{_kv(cov.payload['queries'])} distinct queries per index, {cov.payload['hops']} citation hops, "
        f"{cov.payload['unique']} unique sources. A method missing below was not found in the searched sources; "
        "that is not evidence that it does not exist." if cov else "No search is on record."))
    ev = evidence.evidence(ctx.study)
    if ev:
        rows = [[esc(e["key"]), str(e["page"]), esc(e["section"]), esc("; ".join(f"{k} = {v}" for k, v in e["fields"].items()))]
                for e in ev]
        out.append(r"\paragraph{Extraction}" + "\n" + r"\begin{center}\small"
                   + tabular(["source", "page", "section", "values"], rows, "lrlp{8cm}") + r"\end{center}")
    try:
        lat = evidence.lattice(ctx.study)
    except StudyError as e:
        out.append(rf"\paragraph{{Comparison lattice}} Not drawn: {esc(e.error)}.")
    else:
        rows = [[esc(", ".join(f"{v}" for v in r["coordinates"].values()))]
                + [esc("; ".join(", ".join(f"{k} {c[k]}" for k in lat["values"] if k in c) for c in r["cells"][col]))
                   if col in r["cells"] else "--" for col in lat["columns"]] for r in lat["rows"]]
        out.append(rf"\paragraph{{Comparison lattice}} {lat['filled']} of {lat['cells']} cells filled; an empty cell "
                   "(--) is a combination not found in the extracted sources." + "\n" + r"\begin{center}\small"
                   + tabular([esc(", ".join(lat["coordinates"]))] + [esc(c) for c in lat["columns"]], rows) + r"\end{center}")
    return "\n\n".join(out)


def results(ctx: Ctx) -> str:
    """A table per primary result: the latest research result of the current code, spec and data, then each
    holdout result."""
    if not ctx.primary:
        return ("No result of the study's current code, spec and data is on record; run the study "
                r"(\texttt{openlab study run}) before reporting.")
    out = []
    for r in ctx.primary:
        run = _run(ctx.rec, r)
        out.append(rf"\paragraph{{{esc(run.payload['segment']).capitalize()} segment}} "
                   + (f"Holdout look {ctx.rec.get(ctx.rec.links(run.id, rel='after')[0][2]).payload['n']}."
                      if run.payload["segment"] == "holdout" else "Research runs are the ones the model was chosen on. ")
                   + _selected(ctx, run)
                   + " Table~\\ref{tab:" + r.id + "}.\n" + ("" if r.id in ctx.tables else table(ctx, r)))
    return "\n\n".join(out)


TRUST = (r"\paragraph{Trust levels} What each kind of statement here rests on. A number bound to a result was "
         r"computed by openlab from sandboxed runs under the sealed spec, with the truncation audit stated beside it. A "
         r"citation's quote was found in the source's text. A claim prints its state (observed, derived, inferred, "
         r"hypothesized or unknown), its verdict and whether a person approved it; an inferred claim is a reading, "
         r"not a measurement. The prose around them is the author's: openlab checked that it types no number and that "
         r"its wording does not outrun the claims, and nothing more. A waived finding is printed in the appendix.")


def limitations(ctx: Ctx) -> str:
    """What the record says limits the results: audit verdicts, trials and looks against their budgets,
    amendments, the sandbox, coverage, claims at risk, and the trust levels."""
    rec, st = ctx.rec, ctx.study.status()
    items = []
    for r in ctx.primary:
        run = _run(rec, r)
        a = run.payload["audit"]
        items.append((f"audit ({run.payload['segment']})", f"{esc(a.get('verdict'))}: {a.get('checked', 0)} of "
                      f"{a.get('n', 0)} sampled steps replayed on data cut at their time"
                      + (f", {len(a.get('mismatches', []))} differed" if a.get("mismatches") else "")
                      + (r"; nothing checked that the features stay before $t$" if a.get("verdict") == "unchecked" else "")))
        if run.payload["sandbox"] != "bwrap":
            items.append(("sandbox", f"{esc(run.id)} ran {esc(run.payload['sandbox'])}: no isolation from the network "
                          "or the record"))
    if ctx.spec.mode != "review":
        items.append(("trials", f"{st['trials']['used']} of {st['trials']['budget']} research runs; each distinct run "
                      "is a counted trial and the model was chosen among them"))
        items.append(("holdout looks", f"{st['looks']['used']} of {st['looks']['budget']}"
                      + ("" if st["looks"]["used"] else ": no holdout result, so every number is from the segment the "
                         "model was chosen on")))
    items.append(("amendments", f"{len(st['amendments'])}" + (" (listed in Methods)" if st["amendments"] else "")))
    cov = rec.last("coverage")
    if cov is not None and cov.payload["verdict"] != "sufficient":
        items.append(("coverage", f"the search is {esc(cov.payload['verdict'])} against its sealed floor"))
    risk = at_risk(rec)
    items.append(("claims at risk", "; ".join(f"{esc(c)} (broken: {esc(', '.join(a))})" for c, a in risk.items())
                  if risk else "none: no assumption a claim needs is broken on record"))
    return _items(items) + "\n\n" + TRUST


def reproducibility(ctx: Ctx) -> str:
    """The record head, code and data hashes, the environment, the equation checks and the commands."""
    from ... import __version__
    rec = ctx.rec
    v = rec.verify()
    run = _run(rec, ctx.primary[0]) if ctx.primary else rec.last("run")
    pairs = [("record", rf"{v['nodes']} nodes, head \texttt{{{v['head'][:16]}}} before this paper; hash chain "
              + ("intact" if v["ok"] else f"BROKEN at seq {v['broken_at']}"))]
    if run is not None:
        p = run.payload
        pairs += [(f"code {k}", rf"\texttt{{{s[:16]}}}") for k, s in p["code"].items()]
        pairs += [(f"data {k}", rf"\texttt{{{s[:16]}}}") for k, s in p["data"].items()]
        pairs += [("environment", f"Python {esc(p['env'].get('python'))}, {esc(p['env'].get('platform'))}, packages "
                   rf"\texttt{{{esc(p['env'].get('packages_sha', '')[:16])}}}; openlab {esc(__version__)}"),
                  ("run", rf"\texttt{{{esc(run.id)}}}, {p['seconds']:.0f} s, sandbox {esc(p['sandbox'])}, seed "
                   f"{ctx.spec.split.get('seed', 0)}")]
    pairs += [(f"equation {m.args[0]}", f"{esc(m.args[1])}: " + (esc(why) or "matches the LaTeX on the probes"))
              for m, why in ctx.equations.items()]
    folder = esc(ctx.study.folder.name)
    pairs.append(("commands", rf"\texttt{{openlab study materialise {folder}}}; \texttt{{openlab study run {folder}}}; "
                  rf"\texttt{{openlab study paper {folder}}}; \texttt{{openlab study verify-paper {folder}}}"))
    return _items(pairs) + "\n\nFull hashes are in the signed manifest, \\texttt{paper/build/main.pdf.sig.json}."


def _checks(ctx: Ctx) -> dict:
    """The checklist items the record can answer: id -> (status, detail)."""
    rec, r = ctx.rec, ctx.primary[0] if ctx.primary else None
    run = _run(rec, r) if r else None
    verdict = run.payload["audit"].get("verdict") if run else None
    holdout = [x for x in ctx.primary if _run(rec, x).payload["segment"] == "holdout"]
    tests = rec.find("test")
    return {
        "claims": (MET, "every number in the prose is bound to the record and the wording lint passed"),
        "limitations": (MET, "the Limitations section is generated from the record"),
        "errorbars": (MET, "every metric is reported with its interval") if r and all(
            m.get("lo") is not None for m in r.payload["metrics"].values()) else (NOT_MET, "a metric has no interval"),
        "heldout": (MET, f"{len(holdout)} holdout result(s)") if holdout else (NOT_MET, "no holdout look on record"),
        "significance": (MET, f"{len(tests)} tests on record") if tests else (NA, "no tests on record"),
        "leakage": (MET if verdict == "clean" else NOT_MET, f"truncation audit {verdict}"),
        "code": (MET, "code hashes in Reproducibility; sources in the record's content store"),
        "compute": (MET, f"{run.payload['seconds']:.0f} s, sandbox {run.payload['sandbox']}") if run else (TO_STATE, ""),
    }


def checklist(ctx: Ctx) -> str:
    """The items of a checklists.py list the record can answer; the rest are left for the author to state."""
    source, items = CHECKLISTS[ctx.checklist]
    known = _checks(ctx)
    rows = [[esc(text), *(esc(x) for x in known.get(i, (TO_STATE, "not in the record")))] for i, text, _ in items]
    return (rf"Items of the {esc(source.split(',')[0])}, read from the record." + "\n" + r"\begin{center}\small"
            + tabular(["item", "status", "from the record"], rows, "p{5.5cm}lp{7cm}") + r"\end{center}")


GENERATED = {"data": data, "methods": methods, "literature": literature, "results": results,
             "limitations": limitations, "reproducibility": reproducibility, "checklist": checklist}


def bibtex(ctx: Ctx) -> str:
    """A BibTeX entry per cited source, from its node."""
    out = []
    for sid, n in ctx.cited.items():
        p = n.payload
        fields = {"title": "{" + esc(p["title"]) + "}", "author": " and ".join(esc(a) for a in p.get("authors", [])),
                  "year": str(p.get("year") or ""), "journal": esc(p.get("venue", "")), "doi": p.get("doi", ""),
                  "howpublished": f"arXiv:{p['arxiv']}" if p.get("arxiv") and not p.get("venue") else "",
                  "url": p.get("url", ""), "note": "RETRACTED" if retracted(ctx.rec, n) else ""}
        body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items() if v)
        out.append(f"@{'article' if p.get('venue') else 'misc'}{{{sid},\n{body}\n}}")
    return "\n\n".join(out) + "\n"


def waivers(rec, waived: list) -> str:
    """The appendix: each finding a person waived, with the waiver."""
    if not waived:
        return r"\section{Waivers}No finding of the lints or gates was waived."
    rows = []
    for f, a in waived:
        rows.append(rf"\item \texttt{{{esc(f.id)}}} ({esc(f.location)}): {esc(f.message)}\\ Waived by a person "
                    rf"(\texttt{{{esc(a.id)}}}, {esc(a.at)}): {esc(a.payload.get('reason', ''))}")
    return r"\section{Waivers}A person let these findings pass." + "\n" + r"\begin{itemize}" + "\n" + "\n".join(rows) \
        + "\n" + r"\end{itemize}"


def current_results(rec, now: dict) -> list:
    """The results whose runs match the study now: the latest research one, then each holdout one."""
    from .bindings import stale
    ok = [r for r in rec.find("result") if not stale(rec, r, now)]
    research = [r for r in ok if runs(rec, r)[0].payload["segment"] == "research"]
    return research[-1:] + [r for r in ok if runs(rec, r)[0].payload["segment"] == "holdout"]
