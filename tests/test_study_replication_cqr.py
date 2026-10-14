"""Paper -> code on an interval method (conformalized quantile regression, arXiv 1905.03222): a coverage claim is a
level, judged by closeness to the paper's value and, against a baseline, to the nominal 0.9; a width claim is lower is
better. The index and the PDF are canned: no network."""

import numpy as np
import pytest

from openlab import study
from openlab.sources import papers
from openlab.study.errors import StudyError
from openlab.study.inference import paired
from openlab.study.metrics import METRICS, best, better

PAPER = papers._paper(title="Conformalized Quantile Regression", arxiv="1905.03222", doi="", journal_doi="",
                      version="preprint", date="2019-05-08", pdf="https://arxiv.org/pdf/1905.03222",
                      abstract="Conformal prediction is a technique for constructing prediction intervals.")
TABLE = ("Table 1: Average length and coverage of the 90% prediction intervals over 11 datasets.\n"
         " Quantile Random Forests length 1.20 coverage 0.9262\n"
         " CQR Random Forests length 1.00 coverage 0.9033\n"
         "Table 2: Coverage (%) of the same intervals.\n"
         " Quantile Random Forests coverage 92.62% length 1.20\n"
         " CQR Random Forests coverage 90.33% length 1.00\n"
         " 90.33 90.05 *92.62 *88.51 CQR Random Forests CQR Neural Net *Quantile Random Forests\n")
SHA = "c" * 64


@pytest.fixture
def s(tmp_path, monkeypatch):
    monkeypatch.setattr(papers, "ask", lambda src, q, n, since: [PAPER] if (src, q) == ("arxiv", "1905.03222") else [])
    monkeypatch.setattr(papers, "crossref_updates", lambda dois: {})
    monkeypatch.setattr(papers, "pdf", lambda url: b"%PDF " + url.encode())
    monkeypatch.setattr(papers, "pdf_pages", lambda data, layout=False: ["arXiv:1905.03222v2 [stat.ME]", TABLE])
    s = study.replicate("1905.03222", tmp_path / "cqr", "static", ["QuantileReg", "CQR"], "coverage90", 0.02, 0.05)
    toml = s.folder / "study.toml"
    toml.write_text(toml.read_text().replace('coverage90 = "number"', 'coverage90 = "number"\nwidth90 = "number"')
                    .replace('metrics = ["coverage90"]', 'metrics = ["coverage90", "width90"]'))
    s.amend("the paper reports interval length too")
    return s


def evidence(s, quote, method, **fields):
    return s.extract(s.record.last("source").id, 2, quote, {"method": method, **fields}, "table", "Table 1")["id"]


@pytest.fixture
def table(s):
    qr, cqr = "Quantile Random Forests length 1.20 coverage 0.9262", "CQR Random Forests length 1.00 coverage 0.9033"
    return {"qr": evidence(s, qr, "Quantile Random Forests", coverage90=0.9262, width90=1.2),
            "cqr": evidence(s, cqr, "CQR Random Forests", coverage90=0.9033, width90=1.0)}


def result(s, cqr=(0.8981, 0.8796, 0.9155), qr=(0.9200, 0.9010, 0.9380), cqr_width=(1.02, 0.98, 1.06),
           qr_width=(1.21, 1.17, 1.25)):
    """A static run's result written straight into the record, as the runner writes it (coverage90 with its
    nominal value)."""
    rec = s.record
    under = rec.last("amendment") or rec.last("plan")
    ds = rec.last("dataset", sha256=SHA) or rec.add("dataset", {"name": "main", "source": {"kind": "provider"},
                                                                "sha256": SHA, "rows": 1, "columns": ["y"],
                                                                "card": {}})
    run = rec.add("run", {"segment": "research", "key": f"k{len(rec.find('run'))}", "cached": False,
                          "spec_sha": under.payload["sha"], "code": {}, "data": {"main": SHA}, "env": {},
                          "sandbox": "stub", "seconds": 0.0, "metric_versions": {"coverage90": 1, "width90": 1},
                          "audit": {"verdict": "separated", "checked": 0, "mismatches": []}},
                  links=[("under", under.id), ("uses", ds.id)])

    def cells(cov, wid):
        return {"coverage90": {"value": cov[0], "lo": cov[1], "hi": cov[2], "nominal": 0.9},
                "width90": {"value": wid[0], "lo": wid[1], "hi": wid[2]}}
    return rec.add("result", {"metrics": cells(cqr, cqr_width), "n": 1030,
                              "baselines": {"model:QuantileReg": cells(qr, qr_width)}}, links=[("of", run.id)]).id


def test_the_registry_orders_three_ways():
    assert [METRICS[m].direction for m in ("crps", "auc", "coverage90", "coverage", "width90", "interval_width")] \
        == ["lower", "higher", "target", "target", "lower", "lower"]
    assert better("coverage90", 0.8981, 0.8204) is True and better("coverage90", 0.95, 0.89) is False
    assert better("coverage90", 0.88, 0.92) is None                          # equally far from 0.9
    assert better("coverage", 0.79, 0.85, level=0.8) is True and better("coverage", 0.79, 0.85) is None
    assert best("coverage90", [0.9262, 0.8851, 0.9033]) == 0.9033 and best("width90", [1.2, 1.0]) == 1.0


def test_a_paired_comparison_on_coverage_prefers_the_one_closer_to_nominal():
    import polars as pl
    rng = np.random.default_rng(1)
    y = rng.normal(size=400)
    calibrated = pl.DataFrame({"q0.05": np.full(400, -1.645), "q0.95": np.full(400, 1.645)})
    narrow = pl.DataFrame({"q0.05": np.full(400, -1.0), "q0.95": np.full(400, 1.0)})
    d = paired("coverage90", y, calibrated, narrow, draws=200)
    assert d["diff"] > 0 and d["better"] == "a" and d["nominal"] == 0.9
    over = pl.DataFrame({"q0.05": np.full(400, -3.0), "q0.95": np.full(400, 3.0)})   # covers ~1.0: farther from 0.9
    assert paired("coverage90", y, over, calibrated, draws=200)["better"] == "b"
    assert "nominal" not in paired("width90", y, narrow, calibrated, draws=200)


def test_a_coverage_claim_replicates_within_tolerance_and_beats_the_baseline_by_closeness(s, table):
    r = result(s)
    gate = s.fidelity(table["qr"], r, "model:QuantileReg")
    assert gate["payload"]["passed"]                                         # 0.92 vs 0.9262
    out = s.replication(table["cqr"], r, [gate["id"]])["payload"]
    assert (out["verdict"], out["paper"], out["reference"], out["nominal"]) == ("replicated", 0.9033, 0.9262, 0.9)
    assert out["vs_reference"] == "closer to 0.9 than the baseline (0.0019 vs 0.0262 away)"
    far = result(s, cqr=(0.85, 0.83, 0.87))
    out = s.replication(table["cqr"], far, [s.fidelity(table["qr"], far, "model:QuantileReg")["id"]])["payload"]
    assert out["verdict"] == "not_replicated" and out["vs_reference"].startswith("farther from 0.9")
    from openlab.study.show import line
    assert "best at" not in line({"kind": "replication", "payload": out}) and "ours farther from 0.9" in \
        line({"kind": "replication", "payload": out})


def test_a_width_claim_is_lower_is_better(s, table):
    r = result(s)
    gate = s.fidelity(table["qr"], r, "model:QuantileReg", metric="width90")
    out = s.replication(table["cqr"], r, [gate["id"]], metric="width90")["payload"]
    assert (out["verdict"], out["reference"], out["nominal"], out["vs_reference"]) == ("replicated", 1.2, None, "")
    wide = result(s, cqr_width=(1.23, 1.21, 1.25))
    gate = s.fidelity(table["qr"], wide, "model:QuantileReg", metric="width90")
    assert s.replication(table["cqr"], wide, [gate["id"]], metric="width90")["payload"]["verdict"] == "not_replicated"


ROW = "90.33 90.05 *92.62 *88.51 CQR Random Forests CQR Neural Net *Quantile Random Forests"   # as Table 1 lays out


def test_a_percent_in_the_quote_is_recorded_and_normalised_before_comparing(s):
    qr = s.extract(s.record.last("source").id, 2, "Quantile Random Forests coverage 92.62% length 1.20",
                   {"method": "Quantile Random Forests", "coverage90": 92.62, "width90": 1.2}, "table", "Table 2")
    assert qr["payload"]["units"] == {"coverage90": "percent"}               # from the % sign; width90 has none
    r = result(s)
    f = s.fidelity(qr["id"], r, "model:QuantileReg")["payload"]
    assert (f["paper"], f["paper_written"], f["ours"], f["passed"]) == (0.9262, "92.62%", 0.92, True)
    from openlab.study.show import line
    assert line({"kind": "fidelity", "payload": f}) == ("rebuilt: model:QuantileReg coverage90 0.92 vs the paper's 0.9262 "
                                                        "(92.62% as printed), tolerance 0.02 as a fraction")
    w = s.fidelity(qr["id"], r, "model:QuantileReg", metric="width90", tolerance=0.05,
                   justification="interval length varies more")["payload"]
    assert (w["paper"], w["paper_written"]) == (1.2, "") and "in width90's units" in line({"kind": "fidelity",
                                                                                            "payload": w})
    cqr = s.extract(s.record.last("source").id, 2, "CQR Random Forests coverage 90.33% length 1.00",
                    {"method": "CQR Random Forests", "coverage90": 90.33}, "table", "Table 2")["id"]
    out = s.replication(cqr, r, [s.fidelity(qr["id"], r, "model:QuantileReg")["id"]])["payload"]
    assert (out["paper"], out["paper_written"], out["reference"], out["verdict"]) == (0.9033, "90.33%", 0.9262,
                                                                                       "replicated")


def test_an_ambiguous_unit_is_refused_until_declared(s):
    src = s.record.last("source").id
    bare = s.extract(src, 2, ROW, {"method": "Quantile Random Forests", "coverage90": 92.62}, "table", "Table 1")
    assert bare["payload"]["units"] == {}
    r = result(s, qr=(0.8204, 0.7971, 0.8417))
    with pytest.raises(StudyError, match=r"the paper's coverage90 is 92.62 \(ours: 0.8204\): its unit is ambiguous") as e:
        s.fidelity(bare["id"], r, "model:QuantileReg")
    assert "no % sign" in e.value.meaning
    assert e.value.fix == ("declare unit=percent on the extract (openlab study extract ... --unit coverage90=percent) "
                           "or on the target (openlab study replicate ... --paper-unit percent)")
    declared = s.extract(src, 2, ROW, {"method": "Quantile Random Forests", "coverage90": 92.62}, "table",
                         "Table 1", units={"coverage90": "percent"})
    f = s.fidelity(declared["id"], r, "model:QuantileReg")["payload"]
    assert (f["paper"], f["paper_written"], f["passed"]) == (0.9262, "92.62%", False)
    qrf = {"method": "Quantile Random Forests"}
    with pytest.raises(StudyError, match="writes coverage90 as a percentage"):
        s.extract(src, 2, "Quantile Random Forests coverage 92.62% length 1.20", {**qrf, "coverage90": 92.62},
                  "table", "Table 2", units={"coverage90": "fraction"})
    with pytest.raises(StudyError, match="unit `width90=pct` does not fit"):
        s.extract(src, 2, ROW, {**qrf, "width90": 92.62}, "table", "Table 1", units={"width90": "pct"})
    pct = s.extract(src, 2, ROW, {**qrf, "width90": 92.62}, "table", "Table 1", units={"width90": "percent"})
    with pytest.raises(StudyError, match="width90 is not a share"):
        s.fidelity(pct["id"], r, "model:QuantileReg", metric="width90", tolerance=0.05, justification="wide")


def test_the_target_can_say_the_paper_prints_shares_in_percent(tmp_path, s):
    from openlab.study import cli
    folder = tmp_path / "cqr_pct"
    assert cli.main(["replicate", "1905.03222", str(folder), "--mode", "static", "--methods", "QuantileReg,CQR",
                     "--metric", "coverage90", "--tolerance", "0.02", "--paper-unit", "percent"]) == 0
    t = study.open(folder)
    assert t.record.last("target").payload["paper_unit"] == "percent"
    ev = t.extract(t.record.last("source").id, 2, ROW, {"method": "Quantile Random Forests", "coverage90": 92.62},
                   "table", "Table 1")
    from openlab.study.replication import _target
    from openlab.study.schema import _paper
    assert _paper(t.record, _target(t.record), t.record.get(ev["id"]), "coverage90") == (0.9262, "92.62%")
    assert cli.main(["extract", str(folder), t.record.last("source").id, "--page", "2", "--quote", ROW, "--section",
                     "Table 1", "--fields", '{"method": "CQR Random Forests", "coverage90": 90.33}', "--unit",
                     "coverage90=percent"]) == 0
    assert t.record.last("evidence").payload["units"] == {"coverage90": "percent"}


def test_a_typed_coverage_verdict_is_refused(s, table):
    r = result(s)
    g = s.fidelity(table["qr"], r, "model:QuantileReg")["id"]
    want = s.replication(table["cqr"], r, [g])["payload"]
    links = [("for", s.record.last("target").id), ("paper", table["cqr"]), ("ours", r), ("gated_by", g)]
    with pytest.raises(StudyError, match="never typed"):
        s.record.add("replication", {**want, "vs_reference": "closer, by assertion"}, links=links)
