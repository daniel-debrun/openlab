"""Paper -> repo: a study that replicates one paper.

`new` pins the paper (a review that finds it by its id, its source and its full text by page), writes the target
with its tolerance policy before any run, and scaffolds the rebuild: model.py with one class per method,
paper/equations.tex, a data recipe stub and a README with the next steps. `fidelity` and `replication` write the
schema bodies (`Target`, `Fidelity`, `Replication`), whose numbers and verdicts openlab computes from the evidence,
results and fidelities they link; `replications` is the derived report.
"""

from __future__ import annotations

import re
import shlex
import time
import string
from pathlib import Path

from . import literature, schema
from . import _node, new as new_study
from .errors import StudyError
from .record import Node
from .schema import Fidelity, Replication, Target

TEMPLATE = Path(__file__).parent / "templates" / "replicate"
INDEX = {"arxiv": "arxiv", "doi": "openalex"}       # the index a review finds a paper on, by the kind of its id
_ARXIV_STAMP = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5})")
_DOI_TEXT = re.compile(r"\b10\.\d{4,9}/[^\s\"<>]+")


def _fill(template: str, **values) -> str:
    return string.Template((TEMPLATE / template).read_text(encoding="utf-8")).substitute(values)


def _pdf_id(data: bytes) -> tuple:
    """(kind, id) a PDF names on its first two pages: the arXiv stamp, else the first DOI."""
    from ..sources import papers
    text = " ".join(papers.pdf_pages(data)[:2])
    m = _ARXIV_STAMP.search(text)
    if m:
        return "arxiv", m[1]
    m = _DOI_TEXT.search(text)
    doi = schema.norm_id("doi", m[0].rstrip(".,;)]")) if m else ""
    if not schema.valid_id("doi", doi):
        raise StudyError("the PDF names no arXiv id or DOI on its first pages", "the target is pinned by its id",
                         "give its arXiv id or DOI instead")
    return "doi", doi


def target_id(src: str) -> tuple:
    """(kind, id, PDF bytes or None, PDF path or None) of the paper `src` names: an arXiv id, a DOI, or a PDF file."""
    path = Path(src).expanduser()
    if path.suffix.lower() == ".pdf" or path.is_file():
        if not path.is_file():
            raise StudyError(f"no PDF at {src}")
        data = path.read_bytes()
        if not data.startswith(b"%PDF"):
            raise StudyError(f"{src} is not a PDF")
        return (*_pdf_id(data), data, path.resolve())
    for kind in ("doi", "arxiv"):
        ident = schema.norm_id(kind, src)
        if schema.valid_id(kind, ident):
            return kind, ident, None, None
    raise StudyError(f"`{src}` is not an arXiv id, a DOI or a PDF file", fix="e.g. 1805.09091, "
                     "10.1175/MWR-D-18-0187.1 or paper.pdf")


def _model_py(key: str, methods: list, mode: str) -> str:
    head = (f'"""The methods of {key}, rebuilt: one class per method, each with fit and predict.\n\nopenlab scores '
            'Model as the model and every class evaluate.baselines names ("model:<Name>") as its\nbaseline, on the '
            'same rows. Equations: paper/equations.tex."""\n\n')
    if mode == "temporal":
        head += ('\ndef features(view, t):\n    """What the methods see at decision t: one row per entity, with its '
                 'key columns."""\n    raise NotImplementedError("features: the paper\'s predictors, from '
                 'view.frame(...)")\n\n\ndef label(view, t):\n    """The outcome the paper scores at t + horizon: '
                 'the key columns and y."""\n    raise NotImplementedError("label: the paper\'s target, from '
                 'view.frame(...)")\n\n')
    body = "".join(f'\nclass {m}:\n    """{m} as the paper defines it (cite the page and equation)."""\n\n'
                   f'    def fit(self, X, y):\n        raise NotImplementedError("{m}.fit")\n\n'
                   f'    def predict(self, X):\n        raise NotImplementedError("{m}.predict")\n\n' for m in methods)
    return head + body + f"\nModel = {methods[-1]}    # the paper's own method; the others are its baselines\n"


def _declaration(metric: str, own: dict) -> str:
    """study.toml's [metrics.<metric>]: the study's own metric, its function in metrics.py."""
    import json
    keys = {k: own[k] for k in ("better", "reads", "unit", "description") if own.get(k) not in (None, "", [])}
    return (f"\n# The study's own metric: def {metric}(y, pred) in metrics.py, scored in the sandbox. The seal holds "
            "metrics.py's sha;\n# from the first run on, an edit of it is a person's amendment.\n"
            f"[metrics.{metric}]\n" + "".join(f"{k} = {json.dumps(v)}\n" for k, v in keys.items()))


def _equations(key: str, methods: list) -> str:
    return (f"% The equations of {key}, one per formula a method implements. The paper builder binds each to the\n"
            "% function that computes it and checks the two agree on probe inputs:\n"
            "%   \\labeq{<name>}{model.<function>}{<the equation as the paper writes it>}\n"
            "% Uncomment a line once its function exists in model.py, and cite the page.\n"
            + "".join(f"% \\labeq{{{m}}}{{model.<function>}}{{<equation, page N>}}\n" for m in methods))


def new(src: str, folder, mode: str, methods: list, metric: str, tolerance: float, max_tolerance: float = 0.0,
        kind: str = "replication", code: str = "", paper_unit: str = "", pdf_file=None, pdf_url: str = "",
        own: dict | None = None):
    """A replication study of the paper `src` names, in a new folder: scaffolded, its review sealed, the paper's
    source and full text recorded, and the target written with its tolerance policy (in the metric's unit).
    `paper_unit` percent: the paper prints shares such as coverage as percentages. `pdf_file`: the paper's PDF
    already fetched (from `pdf_url`), read instead of fetching it again. `own`: `metric` is the study's own, with
    this declaration (better, reads, unit, description) and its function in the folder's metrics.py."""
    from .metrics import METRICS, STUDY_FILE
    if paper_unit not in ("", *schema.UNITS):
        raise StudyError(f"--paper-unit {paper_unit} is no unit", f"one of {', '.join(schema.UNITS)}")
    methods = list(dict.fromkeys(m.strip() for m in methods if m.strip()))
    bad = [m for m in methods if not m.isidentifier() or m in ("Model", "features", "label")]
    if not methods or bad:
        raise StudyError(f"methods must be class names: {', '.join(bad) or 'none given'}",
                         fix="e.g. EMOS_gl,NN_aux_emb")
    if own:
        if not metric.isidentifier() or (metric in METRICS and not METRICS[metric].study):
            raise StudyError(f"`{metric}` cannot name the study's own metric", "it must be a Python function name "
                             "that is not one of openlab's metrics", f"e.g. {metric}_study")
        if not (Path(folder) / STUDY_FILE).is_file():
            raise StudyError(f"the study's own metric {metric} needs {Path(folder) / STUDY_FILE}",
                             fix=f"def {metric}(y, pred): ... returning a float")
    elif metric not in METRICS:
        raise StudyError(f"no metric `{metric}`", f"one of {', '.join(METRICS)}")
    id_kind, ident, pdf, path = target_id(src)
    if pdf is None and pdf_file is not None:
        path = Path(pdf_file).resolve()
        pdf = path.read_bytes()
    key = f"{id_kind}:{ident}"
    study = new_study(folder, mode)
    f = study.folder
    baselines = ", ".join(f'"model:{m}"' for m in methods[:-1])
    (f / "study.toml").write_text(_fill("study.toml", name=f.name, key=key, mode=mode, index=INDEX[id_kind],
                                        ident=ident, metric=metric, baselines=baselines,
                                        experiment=(TEMPLATE / f"experiment_{mode}.toml").read_text(encoding="utf-8"))
                                  + (_declaration(metric, own) if own else ""), encoding="utf-8")
    (f / "model.py").write_text(_model_py(key, methods, mode), encoding="utf-8")
    (f / "recipes" / "make_data.py").write_text(_fill("make_data.py", key=key), encoding="utf-8")
    (f / "paper").mkdir(exist_ok=True)
    (f / "paper" / "equations.tex").write_text(_equations(key, methods), encoding="utf-8")
    (f / "paper" / "paper.toml").write_text(              # so `study paper-lint` checks the equations from the start
        f'title = "Replication of {key}"\nauthors = []\nabstract = "equations.tex"   # the equation bindings, checked '
        'by paper-lint\nsections = ["methods"]\n', encoding="utf-8")
    study.discover()                                    # seals study.toml
    src_node = study.record.last("source", **{id_kind: ident})
    for wait in (3, 10, 30):                            # an index answers empty now and then (arXiv's API does)
        if src_node is not None:
            break
        time.sleep(wait)
        study.discover()
        src_node = study.record.last("source", **{id_kind: ident})
    if src_node is None:
        raise StudyError(f"{INDEX[id_kind]} did not return {key}", fix="check the id; then `openlab study discover`")
    if pdf is None:
        doc = study.fetch_text(src_node.id)
    else:
        doc = study._act("fetch_text", {"source": src_node.id, "file": str(path)}, "agent", lambda: _node(
            literature.pdf_document(study.record, src_node, pdf, pdf_url or path.as_uri())))
    study._act("target", {"source": src_node.id, "kind": kind, "tolerance": tolerance, "max_tolerance": max_tolerance,
                          **({"paper_unit": paper_unit} if paper_unit else {})},
               "agent", lambda: _node(study.record.add(
                   Target(kind=kind, tolerance=tolerance, max_tolerance=max_tolerance or tolerance, code=code,
                          paper_unit=paper_unit), links=[("of", src_node.id)])))
    (f / "README.md").write_text(_fill(
        "README.md", title=src_node.payload["title"] or key, key=key, kind=kind, source=src_node.id,
        pages=doc["payload"]["n_pages"], document=f"{doc['id']} from {doc['payload']['url']}", tolerance=tolerance,
        metric=metric, max_tolerance=max_tolerance or tolerance, folder=shlex.quote(str(f))), encoding="utf-8")
    return study


def _target(rec) -> Node:
    t = rec.last("target")
    if t is None:
        raise StudyError("the study has no target", "fidelity and replication check a target paper",
                         "start a replication with `openlab study replicate`")
    return t


def _links(rec, pairs: list) -> dict:
    out: dict = {}
    for rel, i in pairs:
        out.setdefault(rel, []).append(rec.get(i))
    return out


def _args(study, metric: str, tolerance) -> tuple:
    """The target, the metric (default: the primary) and the tolerance (default: the target's)."""
    t = _target(study.record)
    return t, metric or study.spec.evaluate["metrics"][0], t.payload["tolerance"] if tolerance is None else tolerance


def _results(ours) -> list:
    """`ours` as result ids: one id, a comma-separated list, or a list (a paper number averaged over several runs)."""
    ids = [y.strip() for x in ([ours] if isinstance(ours, str) else ours) for y in str(x).split(",")]
    ids = [x for x in ids if x]
    if not ids:
        raise StudyError("`ours` names no result", fix="a result id, or several (a mean over runs)")
    return ids


def fidelity(study, paper: str, ours, arm: str = "model", tolerance: float | None = None, metric: str = "",
             justification: str = "", group: str = "") -> Node:
    """Record whether a baseline of the paper was rebuilt: the paper's value (evidence `paper`) against ours
    (`arm` of result `ours`, or the mean over several results: the paper's number averaged over horizons, datasets
    or seeds, one run each) within tolerance."""
    rec = study.record
    t, metric, tolerance = _args(study, metric, tolerance)
    pairs = [("for", t.id), ("paper", paper), *[("ours", r) for r in _results(ours)]]
    computed = schema.fidelity_fields(rec, _links(rec, pairs), metric, arm, tolerance, justification, group)
    return rec.add(Fidelity(metric=metric, arm=arm, tolerance=tolerance, justification=justification, group=group,
                            **computed), links=pairs)


def replication(study, paper: str, ours, gated_by: list, arm: str = "model", tolerance: float | None = None,
                metric: str = "", justification: str = "", group: str = "") -> Node:
    """Record one claim of the paper against our rebuild (one result, or the mean over several), gated by
    fidelities over the same results' protocols; openlab computes the verdict."""
    rec = study.record
    t, metric, tolerance = _args(study, metric, tolerance)
    pairs = [("for", t.id), ("paper", paper), *[("ours", r) for r in _results(ours)]] + \
        [("gated_by", g) for g in gated_by]
    computed = schema.replication_fields(rec, _links(rec, pairs), metric, arm, tolerance, justification, group)
    return rec.add(Replication(metric=metric, arm=arm, tolerance=tolerance, justification=justification,
                               group=group, **computed), links=pairs)


def _row(rec, n: Node) -> dict:
    ev = rec.get(rec.links(n.id, rel="paper")[0][2])
    return {"id": n.id, "method": ev.payload["fields"].get("method", ""), "page": ev.payload["page"],
            "section": ev.payload["section"], "evidence": ev.id, "result": rec.links(n.id, rel="ours")[0][2],
            "results": [d for _, _, d in rec.links(n.id, rel="ours")],
            "gated_by": [d for _, _, d in rec.links(n.id, rel="gated_by")], **n.payload,
            **({"rule_now": now} if n.kind == "replication" and (now := _rule_now(n.payload)) else {})}


def _rule_now(p: dict) -> str:
    """The verdict openlab's current rule gives the recorded numbers, when it differs from the recorded one (the rule
    was corrected after the check): "" otherwise. The record keeps what was computed; a new check records the fix."""
    from .metrics import METRICS
    m = METRICS.get(p.get("metric"))
    if m is None or p.get("reason") or p.get("nominal") is not None or p.get("reference") is None \
            or p.get("lo") is None or p.get("verdict") == "inconclusive":
        return ""
    near = abs(p["ours"] - p["paper"]) <= p["tolerance"] + 1e-12
    now = schema.replication_tier(p["lo"], p["hi"], p["paper"], p["reference"], near=near, better=m.direction)
    return now if now != p["verdict"] else ""


def replications(study) -> dict:
    """The replication report, derived from the record: the target, each fidelity (rebuilt or not) and each claim's
    verdict, with whether the rebuild as a whole is faithful (the last check of every baseline number passed; the
    earlier failures stay on record and are counted in failed_before). A claim's own verdict still rests on the
    fidelities it names."""
    rec = study.record
    t = _target(rec)
    src = rec.get(rec.links(t.id, rel="of")[0][2])
    fid = [_row(rec, n) for n in rec.find("fidelity")]
    latest = list({r["evidence"]: r for r in fid}.values())    # each baseline number's last check decides the rebuild
    return {"target": {"id": t.id, "source": src.id, "key": schema.source_key(src.payload),
                       "title": src.payload["title"], **t.payload},
            "rebuilt": all(r["passed"] for r in latest) if latest else None, "fidelity": fid,
            "failed_before": sum(not r["passed"] for r in fid) - sum(not r["passed"] for r in latest),
            "replications": [_row(rec, n) for n in rec.find("replication")]}
