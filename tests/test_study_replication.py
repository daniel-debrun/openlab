"""Guarantees of a replication study, enforced by the record's schema: a fidelity's and a replication's numbers and
verdict are computed from the evidence, result and fidelities they link, never typed; a replication gated by a failed
fidelity is inconclusive (rebuild not faithful), so a bad rebuild cannot fail the paper; a paper value inside our
interval replicates; a look-ahead run backs nothing; the tolerance policy is fixed before any run. The indexes and the
PDF are canned: no network."""

import pytest

from openlab import study
from openlab.app.registry import ToolContext, dispatch
from openlab.sources import papers
from openlab.study import cli, replication
from openlab.study.errors import StudyError
from openlab.study.schema import replication_tier

PDF = "https://arxiv.org/pdf/1805.09091"
PAPER = papers._paper(title="Neural networks for post-processing ensemble weather forecasts", arxiv="1805.09091",
                      doi="10.1175/MWR-D-18-0187.1", journal_doi="10.1175/mwr-d-18-0187.1", version="preprint",
                      date="2018-05-23", pdf=PDF, abstract="Ensemble weather predictions require post-processing.")
TABLE = ("Table 2: Mean CRPS for raw and post-processed ECMWF ensemble forecasts.\n"
         " Raw ensemble 1.16 1.16\n EMOS-gl Global EMOS 1.01 1.00\n EMOS-loc Local EMOS 0.90 0.90\n"
         " NN-aux-emb . . . and station embeddings 0.82 0.78\n")
SHA = "a" * 64
METHODS = ["Raw", "EMOS_gl", "EMOS_loc", "NN_aux_emb"]


@pytest.fixture
def canned(monkeypatch):
    monkeypatch.setattr(papers, "ask", lambda src, q, n, since: [PAPER] if (src, q) == ("arxiv", "1805.09091") else [])
    monkeypatch.setattr(papers, "crossref_updates", lambda dois: {})
    monkeypatch.setattr(papers, "pdf", lambda url: b"%PDF " + url.encode())
    monkeypatch.setattr(papers, "pdf_pages", lambda data, layout=False: ["arXiv:1805.09091v1 [stat.ML]", TABLE])


@pytest.fixture
def s(tmp_path, canned):
    return study.replicate("arXiv:1805.09091", tmp_path / "rl18", "temporal", METHODS, "crps", 0.02, 0.03)


def evidence(s, quote, method, crps):
    src = s.record.last("source").id
    return s.extract(src, 2, quote, {"method": method, "crps": crps, "setting": "training 2015"}, "table",
                     "Table 2")["id"]


def result(s, *, audit="clean", nn=(0.82, 0.80, 0.84), raw=1.15, gl=1.005, loc=(0.90, 0.88, 0.92), data=SHA):
    """A run and its result written straight into the record, as the runner would: Model is NN_aux_emb and the
    other methods are its baselines."""
    rec = s.record
    under = rec.last("amendment") or rec.last("plan")
    ds = rec.last("dataset", sha256=data) or rec.add("dataset", {"name": "main", "source": {"kind": "provider"},
                                                                 "sha256": data, "rows": 1, "columns": ["y"],
                                                                 "card": {}})
    bad = [{"t": "2016-03-01"}] if audit == "look-ahead" else []
    run = rec.add("run", {"segment": "research", "key": f"k{len(rec.find('run'))}", "cached": False,
                          "spec_sha": under.payload["sha"], "code": {}, "data": {"main": data}, "env": {},
                          "sandbox": "stub", "seconds": 0.0, "metric_versions": {"crps": 1},
                          "audit": {"verdict": audit, "checked": 3, "mismatches": bad}},
                  links=[("under", under.id), ("uses", ds.id)])
    cell = lambda v: {"crps": {"value": v[0], "lo": v[1], "hi": v[2]}}          # noqa: E731
    return rec.add("result", {"metrics": cell(nn), "n": 100, "baselines": {
        "model:Raw": cell((raw, raw - 0.01, raw + 0.01)), "model:EMOS_gl": cell((gl, gl - 0.01, gl + 0.01)),
        "model:EMOS_loc": cell(loc)}}, links=[("of", run.id)]).id


@pytest.fixture
def table(s):
    return {"raw": evidence(s, "Raw ensemble 1.16 1.16", "Raw ensemble", 1.16),
            "gl": evidence(s, "EMOS-gl Global EMOS 1.01 1.00", "EMOS-gl", 1.01),
            "loc": evidence(s, "EMOS-loc Local EMOS 0.90 0.90", "EMOS-loc", 0.9),
            "nn": evidence(s, "NN-aux-emb . . . and station embeddings 0.82 0.78", "NN-aux-emb", 0.82)}


def gates(s, table, r):
    return [s.fidelity(table["raw"], r, "model:Raw")["id"], s.fidelity(table["gl"], r, "model:EMOS_gl")["id"]]


def test_replicate_pins_the_paper_and_scaffolds_the_rebuild(s):
    rec, f = s.record, s.folder
    src = rec.last("source")
    assert (src.payload["arxiv"], src.payload["published_as"]) == ("1805.09091", "10.1175/mwr-d-18-0187.1")
    assert rec.last("document").payload["n_pages"] == 2 and rec.find("plan")
    t = rec.last("target")
    assert (t.payload["kind"], t.payload["tolerance"], t.payload["max_tolerance"]) == ("replication", 0.02, 0.03)
    assert rec.links(t.id, rel="of")[0][2] == src.id
    model = (f / "model.py").read_text()
    assert all(f"class {m}:" in model for m in METHODS) and "Model = NN_aux_emb" in model and "def label(" in model
    assert s.spec.evaluate["baselines"] == ["model:Raw", "model:EMOS_gl", "model:EMOS_loc"]
    assert s.spec.review["queries"] == ["1805.09091"] and s.spec.review["citation_hops"] == 0
    assert all(f"\\labeq{{{m}}}" in (f / "paper" / "equations.tex").read_text() for m in METHODS)
    assert "SystemExit" in (f / "recipes" / "make_data.py").read_text()
    assert src.id in (f / "README.md").read_text() and s.verify()["ok"]


def test_an_agent_plans_the_experiment_until_the_first_run(s):
    toml = s.folder / "study.toml"
    toml.write_text(toml.read_text().replace('horizon = "1d"', 'horizon = "2d"').replace("trials = 20", "trials = 30"))
    st = s.status()
    assert st["sealed"] and st["planning"] and st["person_fields"] == ["review.coverage_floor", "review.comparable_on"]
    assert "until the first run" in "\n".join(cli._status_text(st))
    assert s.amend("the paper's 48 h lead time")["payload"]["person_fields"] == []
    result(s)
    st = s.status()
    assert not st["planning"] and "budget" in st["person_fields"] and "frozen" in "\n".join(cli._status_text(st))
    toml.write_text(toml.read_text().replace("trials = 30", "trials = 40"))
    with pytest.raises(StudyError, match="budget can only be amended by a person"):
        s.amend("more trials")


def test_an_ungated_replication_is_refused_saying_what_to_do_instead(s, table):
    r = result(s)
    with pytest.raises(StudyError, match="at least one gating fidelity") as e:
        s.replication(table["nn"], r, [])
    assert "study_claim" in e.value.fix and not s.record.find("replication")


def test_a_typed_number_or_verdict_is_refused(s, table):
    r = result(s)
    links = [("for", s.record.last("target").id), ("paper", table["raw"]), ("ours", r)]
    body = {"metric": "crps", "arm": "model:Raw", "tolerance": 0.02, "paper": 1.16, "ours": 1.15, "passed": True}
    with pytest.raises(StudyError, match="never typed") as e:
        s.record.add("fidelity", {**body, "paper": 1.0}, links=links)
    assert "paper: typed 1.0, computed 1.16" in e.value.meaning
    g = gates(s, table, r)
    want = s.replication(table["nn"], r, g)["payload"]
    assert want["verdict"] == "replicated"
    links = [("for", s.record.last("target").id), ("paper", table["nn"]), ("ours", r)] + [("gated_by", i) for i in g]
    with pytest.raises(StudyError, match="never typed") as e:
        s.record.add("replication", {**want, "verdict": "not_replicated"}, links=links)
    assert "verdict: typed 'not_replicated', computed 'replicated'" in e.value.meaning


def test_a_failed_fidelity_forces_inconclusive(s, table):
    r = result(s, raw=1.25, nn=(1.00, 0.98, 1.02))                    # the raw ensemble is not rebuilt
    g = gates(s, table, r)
    assert [s.record.get(i).payload["passed"] for i in g] == [False, True]
    out = s.replication(table["nn"], r, g)["payload"]
    assert out["verdict"] == "inconclusive" and out["reason"] == f"rebuild not faithful: {g[0]} failed"
    links = [("for", s.record.last("target").id), ("paper", table["nn"]), ("ours", r)] + [("gated_by", i) for i in g]
    for typed in ("not_replicated", "weaker"):
        with pytest.raises(StudyError, match="never typed"):
            s.record.add("replication", {**out, "verdict": typed, "reason": ""}, links=links)
    assert replication.replications(s)["rebuilt"] is False


@pytest.mark.parametrize("nn, verdict", [
    ((0.83, 0.80, 0.86), "replicated"),         # the paper's 0.82 is inside our interval
    ((0.79, 0.78, 0.80), "replicated"),         # better than the paper, beyond it
    ((0.95, 0.94, 0.96), "weaker"),             # still beats EMOS-gl (1.01), but short of 0.82
    ((1.03, 1.02, 1.04), "not_replicated"),     # no better than EMOS-gl
    ((0.90, 0.70, 1.10), "inconclusive"),       # holds both 0.82 and 1.01: too wide to tell
])
def test_the_verdict_follows_our_interval(s, table, nn, verdict):
    r = result(s, nn=nn)
    out = s.replication(table["nn"], r, gates(s, table, r))["payload"]
    assert (out["verdict"], out["reference"], out["paper"]) == (verdict, 1.01, 0.82)


def test_a_look_ahead_run_backs_no_fidelity_or_replication(s, table):
    bad = result(s, audit="look-ahead")
    with pytest.raises(StudyError, match="truncation audit is look-ahead"):
        s.fidelity(table["raw"], bad, "model:Raw")
    r = result(s)
    with pytest.raises(StudyError, match="truncation audit is look-ahead"):
        s.replication(table["nn"], bad, gates(s, table, r))


def test_the_tolerance_policy_is_fixed_before_any_run(s, table):
    r = result(s, raw=1.185)
    with pytest.raises(StudyError, match="the target is set before any run"):
        s.record.add("target", {"kind": "replication", "tolerance": 0.5, "max_tolerance": 0.5},
                     links=[("of", s.record.last("source").id)])
    with pytest.raises(StudyError, match="wider than the target's 0.02"):
        s.fidelity(table["raw"], r, "model:Raw", tolerance=0.03)
    with pytest.raises(StudyError, match="outside the target's policy"):
        s.fidelity(table["raw"], r, "model:Raw", tolerance=0.05, justification="close enough")
    assert s.fidelity(table["raw"], r, "model:Raw")["payload"]["passed"] is False
    assert s.fidelity(table["raw"], r, "model:Raw", tolerance=0.03,
                      justification="the paper's raw CRPS uses 50 members; ours a Gaussian fit")["payload"]["passed"]


def test_a_fidelity_vouches_only_for_its_own_protocol(s, table):
    g = gates(s, table, result(s))
    other = result(s, data="b" * 64)
    with pytest.raises(StudyError, match="certifies another protocol"):
        s.replication(table["nn"], other, g)
    with pytest.raises(StudyError, match="cannot gate itself"):
        s.replication(table["gl"], result(s), g)


def test_target_id_reads_ids_and_pdfs(tmp_path, canned):
    assert replication.target_id("https://doi.org/10.1175/MWR-D-18-0187.1")[:2] == ("doi", "10.1175/mwr-d-18-0187.1")
    assert replication.target_id("arXiv:1805.09091v1")[:2] == ("arxiv", "1805.09091")
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF fake")
    assert replication.target_id(str(pdf))[:2] == ("arxiv", "1805.09091")
    with pytest.raises(StudyError, match="not an arXiv id, a DOI or a PDF"):
        replication.target_id("not a paper")


def test_the_cli_and_the_tools_reach_the_verbs(tmp_path, canned, capsys):
    assert cli.main(["replicate", "1805.09091", str(tmp_path / "c"), "--mode", "static", "--methods", "A,B",
                     "--metric", "crps", "--tolerance", "0.02"]) == 0
    assert "replication study in" in capsys.readouterr().out
    s = study.open(tmp_path / "c")
    assert s.spec.mode == "static" and s.spec.evaluate["baselines"] == ["model:A"]
    out = dispatch(ToolContext(str(tmp_path)), "study_replications", {"folder": "c"})["result"]
    assert out["target"]["key"] == "arxiv:1805.09091" and out["rebuilt"] is None


def test_replication_tier_keeps_the_old_rule():
    assert replication_tier(0.01, 0.03, 0.02, 0.0) == "replicated"
    assert replication_tier(-0.01, 0.03, 0.02, 0.0) == "inconclusive"
    assert replication_tier(-0.03, -0.01, 0.02, 0.0) == "not_replicated"
    assert replication_tier(0.001, 0.005, 0.02, 0.0) == "weaker"
    assert replication_tier(0.90, 0.92, 0.93) == "weaker" and replication_tier(0.94, 0.95, 0.93) == "replicated"


def test_a_method_that_trails_its_best_baseline_is_a_level_and_falling_further_short_does_not_replicate():
    """energyad: the paper's DBM (AUROC 0.713) trails its AutoEncoder (0.731). Ours at 0.648 lies further from the
    AE than the paper's, which the old rule read as "beyond the paper, away from the reference"."""
    assert replication_tier(0.640, 0.656, 0.713, 0.731, better="higher") == "not_replicated"
    assert replication_tier(0.700, 0.720, 0.713, 0.731, better="higher") == "replicated"
    assert replication_tier(0.640, 0.656, 0.713, 0.731, near=True, better="higher") == "replicated"
    assert replication_tier(0.725, 0.735, 0.713, 0.731, better="higher") == "replicated"
    # lower is better (CRPS): a paper value above its best baseline's trails it
    assert replication_tier(1.30, 1.35, 1.20, 1.10, better="lower") == "not_replicated"
    # an improvement keeps the old tiers whatever the direction
    assert replication_tier(0.76, 0.78, 0.75, 0.731, better="higher") == "replicated"
    assert replication_tier(0.735, 0.74, 0.75, 0.731, better="higher") == "weaker"
    assert replication_tier(1.00, 1.02, 1.05, 1.10, better="lower") == "replicated"
    assert replication_tier(1.07, 1.08, 1.05, 1.10, better="lower") == "weaker"


def test_a_number_averaged_over_runs_is_checked_against_their_mean(s, table):
    """One run per horizon (data differs), the paper's number their mean: fidelity and replication take every result,
    and a gate must cover the same set of protocols."""
    r1 = result(s, raw=1.14, nn=(0.80, 0.78, 0.82))
    r2 = result(s, raw=1.20, nn=(0.86, 0.84, 0.88), data="c" * 64)
    f = s.fidelity(table["raw"], f"{r1},{r2}", "model:Raw")
    assert f["payload"]["ours"] == pytest.approx(1.17) and f["payload"]["passed"]
    assert [d for _, _, d in s.record.links(f["id"], rel="ours")] == [r1, r2]
    g = [f["id"], s.fidelity(table["gl"], [r1, r2], "model:EMOS_gl")["id"]]
    out = s.replication(table["nn"], [r2, r1], g)["payload"]
    assert (out["ours"], out["lo"], out["hi"]) == pytest.approx((0.83, 0.81, 0.85)) and out["verdict"] == "replicated"
    with pytest.raises(StudyError, match="certifies another protocol"):
        s.replication(table["nn"], r1, g)
    with pytest.raises(StudyError, match="named twice"):
        s.fidelity(table["raw"], [r1, r1], "model:Raw")
    rep = replication.replications(s)
    assert rep["rebuilt"] is True and rep["replications"][0]["results"] == [r2, r1]


def test_a_mean_over_seeds_holds_the_seeds_spread_too(s, table):
    """Runs on the same protocol that differ only in their seed are draws: the mean's interval adds t * sd/sqrt(n)
    to the rows' interval, in quadrature. Runs on different data (above) are fixed parts and add nothing."""
    from scipy.stats import t
    from openlab.study.schema import _ours
    vals = (0.78, 0.82, 0.86)
    rs = [result(s, nn=(v, v - 0.01, v + 0.01)) for v in vals]
    o = _ours(s.record, [s.record.get(r) for r in rs], "crps", "model")
    half = t.ppf(0.975, 2) * 0.04 / 3 ** 0.5
    assert o["value"] == pytest.approx(0.82) and o["seed_half"] == pytest.approx(half)
    assert (o["lo"], o["hi"]) == pytest.approx((0.82 - (0.01 ** 2 + half ** 2) ** 0.5, 0.82 + (0.01 ** 2 + half ** 2) ** 0.5))
    # a paper value 0.9 lies outside the rows' interval [0.81, 0.83] but inside the seeds': not a miss to call
    assert o["lo"] < 0.73 and o["hi"] > 0.9


def test_the_rebuild_is_judged_by_each_numbers_last_check(s, table):
    r = result(s, raw=1.25)
    assert s.fidelity(table["raw"], r, "model:Raw")["payload"]["passed"] is False
    assert replication.replications(s)["rebuilt"] is False
    r2 = result(s, raw=1.16)
    s.fidelity(table["raw"], r2, "model:Raw")
    rep = replication.replications(s)
    assert rep["rebuilt"] is True and rep["failed_before"] == 1 and len(rep["fidelity"]) == 2
