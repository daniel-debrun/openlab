"""Harness benchmarks: what Openlab's structure catches and keeps, computed from study records.
Deterministic and with no model call: the studies are the templates the harness builds and runs itself, the
literature a canned corpus (no network), and every number is read from the records.

    python -m bench.harness                                      # every suite on the harness's own studies
    python -m bench.harness --suite traps,reproducibility --study studies/weather_emos

- traceability: across the studies, the share of claims (and of approved claims) whose chain is complete.
- traps: planted traps run through the real framework; each must be refused or flagged by the structure.
- contradiction: a corpus with planted contradicting papers; recall = the share a recorded contradiction search
  ends `found` on (the harness reads only what the search reached, as an agent would).
- reproducibility: every run of the studies executed again from its record; the share matched.
- cost: per study, the runs' wall time, actions, trials and holdout looks against their budgets, reproductions.

A study given with --study is copied to a temp dir first: the harness never writes into it. Results go to
bench/results/harness/<UTC time>.json. Needs bwrap (the runner's sandbox).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path

SUITES = ("traceability", "traps", "contradiction", "reproducibility", "cost")
OUT = Path(__file__).resolve().parent / "results" / "harness"

# --- the canned literature --------------------------------------------------------------------------------------
# (arxiv id, title, abstract, date, the one page of full text, arxiv ids it references)
CORPUS = [
    ("1805.09091", "Neural networks for postprocessing ensemble weather forecasts",
     "Ensemble postprocessing corrects systematic errors; local EMOS is a strong baseline.", "2018-05-23",
     "Table 2. Local EMOS reaches a mean CRPS of 0.90 at 24h lead time across stations.", ()),
    ("2001.00002", "Distributional regression networks for ensemble postprocessing",
     "Ensemble postprocessing with distributional regression networks.", "2020-01-02",
     "Table 3. DRN reaches a mean CRPS of 0.85 at 48h lead time across stations.", ()),
    # the contradiction corpus: four claim sources (B), four planted conflicts (K), one bridge paper (D)
    ("2101.00001", "EMOS for station temperature calibration", "Station temperature calibration with EMOS.",
     "2021-01-04", "EMOS improves the CRPS of station temperature forecasts by 0.12 at 24h lead time.", ()),
    ("2101.00002", "Quantile methods for station temperature calibration",
     "Station temperature calibration with quantile methods.", "2021-01-05",
     "Quantile regression forests match EMOS on station temperature at every lead time we tried.", ()),
    ("2101.00003", "Spread correction in station temperature calibration",
     "Station temperature calibration of ensemble spread.", "2021-01-06",
     "Ensemble underdispersion is removed by the spread correction at all stations.", ("2001.00103",)),
    ("2101.00004", "Neural station temperature calibration", "Station temperature calibration with neural nets.",
     "2021-01-07", "The neural postprocessing generalises to unseen stations without loss of skill.", ()),
    ("2201.00101", "When EMOS degrades temperature forecasts", "EMOS degrades temperature forecasts in mountains.",
     "2022-01-03", "At mountain stations EMOS raises the CRPS of temperature forecasts by 0.05 at 24h.", ()),
    ("2201.00102", "Forests against regression at the stations", "A comparison at the stations.", "2022-01-04",
     "Quantile regression forests lose to EMOS at long lead times on station temperature.", ("2101.00002",)),
    ("2001.00103", "Underdispersion that stays", "A study of ensemble spread.", "2020-01-08",
     "Underdispersion persists after spread correction at coastal stations in winter.", ()),
    ("2301.00104", "Neural nets that do not travel", "A study of transfer.", "2023-01-09",
     "Neural postprocessing loses skill at stations it was not trained on.", ("2201.00105",)),
    ("2201.00105", "A survey of station calibration", "A survey.", "2022-01-10",
     "We survey calibration methods for station forecasts.", ("2101.00004",)),
]
# claim source -> (the agent's contradiction query, the planted conflict, the route that reaches it, its passage)
PLANTED = {
    "2101.00001": ("emos degrades temperature", "2201.00101", "query"),
    "2101.00002": ("forests outperform emos", "2201.00102", "cited_by hop"),
    "2101.00003": ("spread fixed everywhere", "2001.00103", "references hop"),
    "2101.00004": ("transfer across stations", "2301.00104", "two citation hops (the review follows one)"),
}


def _paper(c: tuple) -> dict:
    from openlab.sources import papers
    key, title, abstract, date, _, _ = c
    return papers._paper(title=title, arxiv=key, doi=f"10.5555/lab.{key}", date=date, abstract=abstract,
                         pdf=f"https://arxiv.org/pdf/{key}", version="preprint")


@contextlib.contextmanager
def canned():
    """openlab.sources.papers answering from CORPUS: a query returns the papers whose title and abstract hold all its
    words; citations follow the corpus's references; full text is the corpus's page; no retraction notices."""
    from openlab.sources import papers
    by = {c[0]: c for c in CORPUS}

    def ask(src, query, n=25, since="", **_):
        words = query.lower().split()
        return [_paper(c) for c in CORPUS if all(w in f"{c[1]} {c[2]}".lower() for w in words)][:n]

    def related(key, n=20, related_works=True, since=""):
        paper_row = next(c for c in CORPUS if key in (c[0], f"10.5555/lab.{c[0]}"))
        return {"references": [_paper(by[r]) for r in paper_row[5]][:n],
                "cited_by": [_paper(c) for c in CORPUS if paper_row[0] in c[5]][:n]}

    fake = {"ask": ask, "related": related, "crossref_updates": lambda dois: {},
            "pdf": lambda url: url.encode(), "pdf_pages": lambda data, layout=False: [by[data.decode()[-10:]][4]]}
    real = {k: getattr(papers, k) for k in fake}
    for k, v in fake.items():
        setattr(papers, k, v)
    try:
        yield
    finally:
        for k, v in real.items():
            setattr(papers, k, v)


# --- the studies ---------------------------------------------------------------------------------------------------
BASE = """
[review]
question = "Which post-processing methods beat EMOS for station forecasts?"
sources = ["arxiv"]
queries = ["ensemble postprocessing"]
include = [{id = "I1", text = "post-processing of ensemble weather forecasts"}]
exclude = [{id = "E1", text = "not about weather"}]
since = "2000-01-01"
citation_hops = 0
n = 10
coverage_floor = {queries_per_source = 1, hops = 0}
coordinates = ["lead"]
comparable_on = ["lead"]

[review.extract]
method = "text"
crps = "number"
lead = "text"

[audit]
n = 2
"""
CONTRA = """
[study]
name = "harness_contra"
question = "Is EMOS the right station temperature calibration?"
mode = "review"

[review]
question = "Is EMOS the right station temperature calibration?"
sources = ["arxiv"]
queries = ["station temperature calibration"]
include = [{id = "I1", text = "calibration of station temperature forecasts"}]
exclude = [{id = "E1", text = "not about temperature"}]
since = "2000-01-01"
citation_hops = 1
n = 10
coverage_floor = {queries_per_source = 1, hops = 1}
"""


class Harness:
    """The studies the suites read, each built once in the work dir (external ones copied there)."""

    def __init__(self, work: Path, external: list):
        self.work, self.external = work, [Path(e).resolve() for e in external]

    def _new(self, name: str, template: str, edit=lambda t: t):
        import openlab.study as vs
        s = vs.new(self.work / name, template)
        toml = s.folder / "study.toml"
        toml.write_text(edit(toml.read_text()))
        return s

    @cached_property
    def base(self):
        """The temporal template with a review: one run, two sources with evidence, four claims, one approved."""
        from openlab.study import schema
        s = self._new("harness_temporal", "temporal",
                      lambda t: t.replace("holdout = [2025-01-01, 2025-06-30]", "holdout = [2025-01-01, 2025-05-31]")
                      + BASE)
        s.run()
        s.discover()
        ev = []
        for key, fields in (("1805.09091", {"method": "Local EMOS", "crps": 0.9, "lead": "24h"}),
                            ("2001.00002", {"method": "DRN", "crps": 0.85, "lead": "48h"})):
            src = s.record.last("source", arxiv=key).id
            s.fetch_text(src)
            ev.append(s.extract(src, 1, next(c[4] for c in CORPUS if c[0] == key), fields, "table", "Table")["id"])
        res = s.record.last("result")
        m = res.payload
        a, b = m["metrics"]["mae"]["value"], m["baselines"]["persistence"]["mae"]["value"]
        c1 = s.claim("The model's mean absolute error over the research segment is on record.", "observed", [res.id],
                     value=a, inputs={"v": f"{res.id}#metrics.mae.value"})["id"]
        s.claim("Local EMOS reaches the CRPS its paper reports at 24h.", "observed", [ev[0]], value=0.9,
                inputs={"v": f"{ev[0]}#fields.crps"})
        inputs = {"a": f"{res.id}#metrics.mae.value", "b": f"{res.id}#baselines.persistence.mae.value"}
        c3 = s.claim("The model's error minus persistence's.", "derived", [res.id],
                     value=schema.evaluate("{a} - {b}", {"a": a, "b": b}), expression="{a} - {b}", inputs=inputs)["id"]
        s.claim("The model forecasts better than persistence on this panel.", "inferred", [c1, c3])
        s.approve(c1, "approve", "checked against its run", person=True)
        return s

    @cached_property
    def static(self):
        s = self._new("harness_static", "static", lambda t: t.replace("k = 5", "k = 3"))
        res = s.record.get(s.run()["id"])
        s.claim("The classifier's AUC is on record.", "observed", [res.id],
                value=res.payload["metrics"]["auc"]["value"], inputs={"v": f"{res.id}#metrics.auc.value"})
        return s

    @cached_property
    def contra(self):
        """A review: four claims, each on evidence from its source."""
        import openlab.study as vs
        folder = self.work / "harness_contra"
        folder.mkdir()
        (folder / "study.toml").write_text(CONTRA)
        s = vs.open(folder)
        s.discover()
        for key in PLANTED:
            src = s.record.last("source", arxiv=key).id
            s.fetch_text(src)
            e = s.extract(src, 1, next(c[4] for c in CORPUS if c[0] == key), {}, "passage", "Results")["id"]
            s.claim(next(c[4] for c in CORPUS if c[0] == key), "observed", [e])
        return s

    @cached_property
    def studies(self) -> dict:
        import openlab.study as vs
        out = {"harness_temporal": self.base, "harness_static": self.static, "harness_contra": self.contra}
        for src in self.external:
            dst = self.work / "external" / src.name
            shutil.copytree(src, dst, symlinks=True)
            out[src.name] = vs.open(dst)
        return out

    def copy(self, name: str):
        """A fresh copy of the base study for one trap."""
        import openlab.study as vs
        dst = self.work / "traps" / name
        shutil.copytree(self.base.folder, dst)
        return vs.open(dst)


# --- traps -------------------------------------------------------------------------------------------------------
LEAKY = '''
from datetime import timedelta

import polars as pl
from sklearn.linear_model import LinearRegression


def features(view, t):
    """Tomorrow's y, read from the staged file behind the view's back."""
    nxt = pl.read_parquet("data/panel.parquet").filter(pl.col("date") == (t + timedelta(days=1)).date())
    now = view.frame("panel").group_by("site").agg(pl.col("x").last()).sort("site")
    return (now.join(nxt.select("site", pl.col("y").alias("y_next")), on="site", how="left")
            .with_columns(pl.col("y_next").fill_null(0.0)))


def label(view, t):
    return view.frame("panel").group_by("site").agg(pl.col("y").last()).sort("site")


class Model:
    def fit(self, X, y):
        self.m = LinearRegression().fit(X.select("x", "y_next").to_numpy(), y)
        return self

    def predict(self, X):
        return self.m.predict(X.select("x", "y_next").to_numpy())
'''
PROBE = '''
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
for p in (HERE / "data" / "panel.parquet", HERE / ".openlab" / "data" / "panel.parquet", Path("data/panel.parquet")):
    try:
        last = pl.read_parquet(p)["date"].max()
    except Exception as e:
        print(f"BLOCKED {p}: {type(e).__name__}")
    else:
        print(f"{'VISIBLE' if str(last) >= '2025-01-01' else 'RESEARCH-ONLY'} {p}: rows to {last}")
'''
PAPER = ('title = "Harness"\nauthors = ["openlab harness"]\nabstract = "sections/abstract.tex"\n'
         'sections = ["discussion"]\n')


def _refused(fn, expect: str) -> dict:
    """Caught when `fn` is refused with `expect` in the refusal."""
    from openlab.study.errors import StudyError
    try:
        fn()
    except StudyError as e:
        said = f"{e.error} ({e.meaning})" if e.meaning else e.error
        return {"caught": expect in said, "by": said[:240]}
    return {"caught": False, "by": "accepted"}


def _flagged(s, discussion: str, rule: str) -> dict:
    """Caught when the paper's lint (what study_paper refuses on) has a finding of `rule`."""
    d = s.folder / "paper" / "sections"
    d.mkdir(parents=True, exist_ok=True)
    (s.folder / "paper" / "paper.toml").write_text(PAPER)
    (d / "abstract.tex").write_text("We ask whether a daily signal predicts the next day's outcome.\n")
    (d / "discussion.tex").write_text(discussion + "\n")
    found = s.paper_lint()["findings"]
    hit = [f for f in found if f["rule"] == rule]
    return {"caught": bool(hit), "by": f"{rule}: {hit[0]['message'][:200]}" if hit else
            f"findings: {', '.join(f['rule'] for f in found) or 'none'}"}


def _ids(s) -> dict:
    rec = s.record
    claims = rec.find("claim")
    return {"result": rec.last("result").id, "evidence": [e.id for e in rec.find("evidence")],
            "inferred": next(c for c in claims if c.payload["state"] == "inferred"),
            "on_evidence": next(c for c in claims if c.payload["state"] == "observed" and c.payload["inputs"]["v"]
                                .startswith("evidence:")),
            "source": rec.get(rec.find("evidence")[0].payload["source"])}


def _again(s, node, change: dict, actor: str = "agent") -> dict:
    """Write `node` again with `change` applied to its payload, under the same links."""
    links = [(rel, d) for _, rel, d in s.record.links(node.id) if rel != "branches_from"]
    return s.record.add(node.kind, {**node.payload, **change}, actor=actor, links=links)


# each trap: on its own study (`h`, the harness) or on a fresh copy of the base study (`s`)
def look_ahead(h) -> dict:
    s = h._new("traps/look_ahead", "temporal", lambda t: t + "\n[audit]\nn = 2\n")
    (s.folder / "model.py").write_text(LEAKY)
    s.run()
    v = s.record.last("run").payload["audit"]["verdict"]
    return {"caught": v == "look-ahead", "by": f"truncation audit: {v}"}


def holdout_read(h) -> dict:
    s = h._new("traps/holdout_read", "temporal", lambda t: t + "\n[audit]\nn = 0\n")
    model = s.folder / "model.py"
    model.write_text(PROBE + model.read_text())
    s.run()
    out = s.record.blob(s.record.last("run").payload["stdout"]).decode()
    return {"caught": "VISIBLE" not in out and "BLOCKED" in out, "by": "; ".join(out.split("\n")[:3])}


def typed_number(s) -> dict:
    return _flagged(s, "The error fell by 12 percent over the year.", "number.typed")


def ungrounded_quote(s) -> dict:
    return _flagged(s, rf"It helps \labcite{{{_ids(s)['source'].id}}}{{post-processing always removes every error}}.",
                    "cite.quote")


def novelty(s) -> dict:
    return _flagged(s, "This is a novel way to forecast.", "wording.novelty")


def cross_protocol(s) -> dict:
    return _refused(lambda: s.compare(*_ids(s)["evidence"], "crps"), "incomparable")


def duplicate_source(s) -> dict:
    return _refused(lambda: _again(s, s.record.last("source"), {}, actor="openlab"), "already on record")


def typed_coverage(s) -> dict:
    cov = s.record.last("coverage")
    return _refused(lambda: _again(s, cov, {"unique": cov.payload["unique"] + 5}, actor="openlab"),
                    "does not match its queries")


def typed_verdict(s) -> dict:
    return _refused(lambda: _again(s, _ids(s)["inferred"], {"verdict": "supported"}), "never typed")


def agent_look(s) -> dict:
    return _refused(lambda: s.holdout("one peek", person=False), "only a person")


def agent_approval(s) -> dict:
    return _refused(lambda: s.approve(_ids(s)["inferred"].id, "approve", "looks right"),
                    "can only be written by person")


def retracted_ground(s) -> dict:
    from openlab.study.schema import Retraction
    ids = _ids(s)
    s.record.add(Retraction(via="crossref", type="retraction", notice="10.5555/lab.retraction", valid_from="2026-01-02"),
                 actor="openlab", links=[("of", ids["source"].id)])
    return _flagged(s, rf"\labclaim{{{ids['on_evidence'].id}}}", "claim.untraceable")


def separated(s) -> dict:
    audit = {"n": 0, "checked": 0, "mismatches": [], "verdict": "separated"}
    return _refused(lambda: _again(s, s.record.last("run"), {"key": "forged", "audit": audit}), "only a static run")


def alpha_overdraw(s) -> dict:
    ids = _ids(s)
    return _refused(lambda: s.test(ids["inferred"].id, ids["result"], "persistence", 0.9),
                    "more than the wealth pays for")


def reforecast(s) -> dict:
    s.forecast(now=datetime(2025, 6, 10, 12))
    return _refused(lambda: s.forecast(now=datetime(2025, 6, 10, 18)), "already forecast decision")


def spec_edit(s) -> dict:
    toml = s.folder / "study.toml"
    toml.write_text(toml.read_text().replace("trials = 20", "trials = 21"))
    return _refused(s.run, "edited after the seal")


TRAPS = [   # (name, what is planted, trap, on a copy of the base study)
    ("look_ahead_feature", "features read tomorrow's y from the staged file", look_ahead, False),
    ("holdout_read_attempt", "model.py reads the raw data, the materialised data and its stage", holdout_read, False),
    ("typed_number_in_paper", "a number typed in the prose", typed_number, True),
    ("ungrounded_citation_quote", "a citation quoting what the source does not say", ungrounded_quote, True),
    ("novelty_wording", "\"novel\" in the prose", novelty, True),
    ("cross_protocol_comparison", "a 24h result compared with a 48h one", cross_protocol, True),
    ("duplicate_source", "the same work recorded twice", duplicate_source, True),
    ("typed_coverage", "a coverage claiming more unique sources than its queries reached", typed_coverage, True),
    ("typed_verdict", "an inferred claim typed supported", typed_verdict, True),
    ("agent_holdout_look", "an agent spending the holdout", agent_look, True),
    ("agent_approval", "an agent approving its own claim", agent_approval, True),
    ("retracted_ground_in_paper", "a paper printing a claim whose source was retracted since", retracted_ground, True),
    ("temporal_run_separated", "a temporal run declaring its audit `separated`", separated, True),
    ("alpha_budget_overdraw", "a test at alpha_j above what the error budget pays for", alpha_overdraw, True),
    ("reforecast_same_t", "a second forecast of the same decision by the same model", reforecast, True),
    ("unamended_spec_edit", "study.toml edited after the seal without an amendment", spec_edit, True),
]


# --- suites ------------------------------------------------------------------------------------------------------
def _share(k: int, n: int):
    return round(k / n, 4) if n else None


def traceability(h) -> dict:
    rows = []
    for name, s in h.studies.items():
        t = s.traceability()
        rows.append({"study": name, "claims": t["claims"], "traceable": t["traceable"],
                     "approved": t["approved"]["claims"], "approved_traceable": t["approved"]["traceable"],
                     "broken": [b["claim"] for b in t["broken"]]})
    n, k = sum(r["claims"] for r in rows), sum(r["traceable"] for r in rows)
    an, ak = sum(r["approved"] for r in rows), sum(r["approved_traceable"] for r in rows)
    return {"claims": n, "traceable": k, "share": _share(k, n), "approved": an, "approved_traceable": ak,
            "approved_share": _share(ak, an), "studies": rows}


def traps(h) -> dict:
    rows = []
    for name, what, fn, on_copy in TRAPS:
        try:
            r = fn(h.copy(name) if on_copy else h)
        except Exception as e:                              # noqa: BLE001 -- a trap that errors is not caught
            r = {"caught": False, "by": f"error: {type(e).__name__}: {e}"[:240]}
        rows.append({"trap": name, "what": what, **r})
    return {"caught": sum(r["caught"] for r in rows), "total": len(rows),
            "missed": [r["trap"] for r in rows if not r["caught"]], "traps": rows}


def contradiction(h) -> dict:
    """Each claim's contradiction search with the agent's query; the harness extracts the planted passage only from a
    planted paper the search reached, records it found, and counts the searches that end found on it."""
    s, rows = h.contra, []
    rec = s.record
    for key, (query, planted, route) in PLANTED.items():
        src = rec.last("source", arxiv=key).id
        claim = next(c for c in rec.find("claim")
                     if rec.get(rec.links(c.id, rel="rests_on")[0][2]).payload["source"] == src)
        out = s.contra(claim.id, [query])
        hit = rec.last("source", arxiv=planted)
        if hit is not None and hit.id in out["search"]["sources"]:
            s.fetch_text(hit.id)
            e = s.extract(hit.id, 1, next(c[4] for c in CORPUS if c[0] == planted), {}, "passage", "Results")["id"]
            out = s.contra(claim.id, found=[e])
        last = rec.get(out["id"])
        found = bool(hit) and any(rec.get(d).payload["source"] == hit.id for _, _, d in rec.links(last.id, rel="found"))
        rows.append({"claim": claim.id, "planted": f"arxiv:{planted}", "route": route, "state": last.payload["state"],
                     "found": found})
    k = sum(r["found"] for r in rows)
    return {"planted": len(rows), "found": k, "recall": _share(k, len(rows)), "conflicts": rows}


def reproducibility(h) -> dict:
    from openlab.study.errors import StudyError
    rows = []
    for name, s in h.studies.items():
        for run in s.record.find("run"):
            try:
                (r,) = s.reproduce(run.id)["reproductions"]
                rows.append({"study": name, "run": run.id, "matched": r["matched"], "max_abs_diff": r["max_abs_diff"],
                             "differs": r["differs"], "env_diff": list(r["env_diff"]),
                             "seconds": round(r["seconds"], 1)})
            except StudyError as e:
                rows.append({"study": name, "run": run.id, "matched": False, "refused": e.error})
    k = sum(r["matched"] for r in rows)
    return {"runs": len(rows), "matched": k, "share": _share(k, len(rows)), "rows": rows}


def cost(h) -> dict:
    rows = []
    for name, s in h.studies.items():
        rec, st = s.record, s.status()
        runs, reps = rec.find("run"), rec.find("reproduction")
        rows.append({"study": name, "nodes": len(rec.find()), "actions": len(rec.find("action")), "runs": len(runs),
                     "run_seconds": round(sum(r.payload["seconds"] for r in runs), 1),
                     "trials": st["trials"], "looks": st["looks"], "reproductions": len(reps),
                     "reproduce_seconds": round(sum(r.payload["seconds"] for r in reps), 1)})
    return {"studies": rows}


FNS = {"traceability": traceability, "traps": traps, "contradiction": contradiction,
       "reproducibility": reproducibility, "cost": cost}


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def text(report: dict) -> list:
    out = []
    if "traceability" in report:
        t = report["traceability"]
        out.append(f"traceability     {t['traceable']} of {t['claims']} claims ({_pct(t['share'])}); approved "
                   f"{t['approved_traceable']} of {t['approved']} ({_pct(t['approved_share'])})")
    if "traps" in report:
        t = report["traps"]
        out.append(f"traps            {t['caught']} of {t['total']} caught")
        out += [f"  {'caught' if r['caught'] else 'MISSED':7} {r['trap']:28} {r['by'][:110]}" for r in t["traps"]]
    if "contradiction" in report:
        c = report["contradiction"]
        out.append(f"contradiction    recall {c['found']} of {c['planted']} planted conflicts ({_pct(c['recall'])})")
        out += [f"  {'found' if r['found'] else 'MISSED':7} {r['planted']:18} {r['route']} ({r['state']})"
                for r in c["conflicts"]]
    if "reproducibility" in report:
        r = report["reproducibility"]
        out.append(f"reproducibility  {r['matched']} of {r['runs']} runs matched ({_pct(r['share'])})")
        out += [f"  {'matched' if x['matched'] else 'DIFFERS':7} {x['study']:18} {x['run']} "
                + (x.get("refused") or x.get("differs") or f"max |diff| {x['max_abs_diff']}") for x in r["rows"]]
    if "cost" in report:
        out.append("cost")
        out += [f"  {x['study']:18} {x['runs']} runs {x['run_seconds']} s, {x['actions']} actions, trials "
                f"{x['trials']['used']}/{x['trials']['budget']}, looks {x['looks']['used']}/{x['looks']['budget']}, "
                f"{x['reproductions']} reproductions {x['reproduce_seconds']} s" for x in report["cost"]["studies"]]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.harness", description="Openlab's own benchmarks, computed from "
                                 "study records")
    ap.add_argument("--suite", default=",".join(SUITES), help=f"comma-separated, of {', '.join(SUITES)}")
    ap.add_argument("--study", action="append", default=[], help="a study to include (copied first; never written)")
    ap.add_argument("--out", default="", help="the results file (default bench/results/harness/<UTC time>.json)")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    a = ap.parse_args(argv)
    suites = [x.strip() for x in a.suite.split(",") if x.strip()]
    if set(suites) - set(SUITES):
        ap.error(f"no suite {', '.join(sorted(set(suites) - set(SUITES)))}")
    work = Path(tempfile.mkdtemp(prefix="openlab-harness-"))
    os.environ.update(OPENLAB_VAULT=str(work / "vault"), OPENLAB_SIGNING_KEY=str(work / "signing.key"),
                      OPENLAB_ALERT_DRY_RUN="1", MLFLOW_DISABLE_AGENT_HINT="1")
    at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    try:
        with canned():
            h = Harness(work, a.study)
            report = {"at": at, "external": [str(p) for p in h.external], **{s: FNS[s](h) for s in suites}}
    finally:
        shutil.rmtree(work, ignore_errors=True)
    out = Path(a.out) if a.out else OUT / f"{at}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps(report, default=str) if a.json else "\n".join(text(report) + [f"written to {out}"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
