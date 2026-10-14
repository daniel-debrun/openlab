"""Guarantees of extraction, comparison and recall, enforced by the record's schema: evidence is quoted from the
page it names, its fields are the sealed schema's with their types, its numbers and text are in the quote; two
results are compared only under the same protocol, with the verdict the metric's direction gives; the lattice
shows its empty cells; recall is recomputed from the record. Canned documents and indexes: no network."""

import pytest

from openlab import study
from openlab.app.registry import ToolContext, dispatch
from openlab.sources import papers
from openlab.study import spec
from openlab.study.errors import StudyError

TOML = """
[study]
name = "ev"
question = "Which post-processing methods beat EMOS?"
mode = "review"

[review]
question = "Which post-processing methods beat EMOS at stations?"
sources = ["arxiv"]
queries = ["ensemble postprocessing"]
include = [{id = "I1", text = "post-processing of ensemble weather forecasts"}]
exclude = [{id = "E1", text = "not about weather"}]
since = "2005-01-01"
citation_hops = 0
n = 5
coverage_floor = {queries_per_source = 1, hops = 0}
coordinates = ["dataset", "lead_time", "metric"]
comparable_on = ["dataset", "lead_time", "metric"]

[review.extract]
dataset = {type = "text", free = true}
lead_time = {type = "text", free = true}
metric = {type = "text", free = true}
method = "text"
crps = "number"
skill = "number"
stations = "int"
"""

P = papers._paper
A = P(title="Neural networks for postprocessing ensemble weather forecasts", arxiv="1805.09091", version="preprint",
      abstract="Ensemble weather predictions require statistical post-processing of systematic errors.",
      date="2018-05-23", pdf="https://arxiv.org/pdf/1805.09091")
B = P(title="Postprocessing of ensemble weather forecasts using permutation-invariant neural networks",
      arxiv="2309.04452", version="preprint", abstract="Statistical postprocessing is used to translate ensembles.",
      date="2023-09-08", pdf="https://arxiv.org/pdf/2309.04452")
TABLE_A = ("Table 2: Mean CRPS for raw and post-processed ECMWF ensemble forecasts.\n"
           " Model       Description              Mean CRPS\n Raw ensemble                         1.16\n"
           " EMOS-gl     Global EMOS              1.01\n EMOS-loc    Local EMOS               0.90\n")
TEXT = {  # reading order (table columns scrambled) and layout (rows intact), per PDF
    A["pdf"]: ["Neural networks for postprocessing ensemble weather forecasts\n",
               "Table 2: Mean CRPS for raw and post-processed ECMWF ensemble forecasts.\nModel\nRaw ensemble\n"
               "1.16\n1.01\n0.90\nEMOS-gl\nEMOS-loc\n",
               "Adding local station information in EMOS-\nloc improves the global score over 1,234 stations; the "
               "worst skill score is −0.05.\n"],
    B["pdf"]: ["Table 4. Mean CRPS in K for the EUPPBench reforecast data.\n Method CRPS\n EMOS 0.82\n DRN 0.67\n"]}
LAYOUT = {A["pdf"]: [TEXT[A["pdf"]][0], TABLE_A, TEXT[A["pdf"]][2]], B["pdf"]: TEXT[B["pdf"]]}
PROTO_A = {"dataset": "ECMWF EPS at German stations, 2016", "lead_time": "48 h", "metric": "CRPS (K)"}
PROTO_B = {"dataset": "EUPPBench reforecasts 2013-2016", "lead_time": "24 h", "metric": "CRPS (K)"}


@pytest.fixture
def made(tmp_path, monkeypatch):
    folder = tmp_path / "s"
    folder.mkdir()
    (folder / "study.toml").write_text(TOML, encoding="utf-8")
    monkeypatch.setattr(papers, "ask", lambda src, q, n, since: [A, B])
    monkeypatch.setattr(papers, "crossref_updates", lambda dois: {})
    monkeypatch.setattr(papers, "pdf", lambda url: b"%PDF " + url.encode())
    monkeypatch.setattr(papers, "pdf_pages", lambda data, layout=False: (LAYOUT if layout else TEXT)[data[5:].decode()])
    s = study.open(folder)
    s.discover()
    return s


def src(s, arxiv):
    return s.record.last("source", arxiv=arxiv).id


ROW = "EMOS-loc Local EMOS 0.90"


def test_a_quote_not_on_the_page_it_names_is_refused(made):
    a = src(made, "1805.09091")
    with pytest.raises(StudyError, match="has no full text"):
        made.extract(a, 2, ROW, {"method": "EMOS-loc", "crps": 0.9}, "table", "Table 2")
    made.fetch_text(a)
    with pytest.raises(StudyError, match="not on page 3") as e:
        made.extract(a, 3, ROW, {"method": "EMOS-loc", "crps": 0.9}, "table", "Table 2")
    assert "on page 2" in e.value.meaning
    with pytest.raises(StudyError, match="not on page 2"):
        made.extract(a, 2, "EMOS-loc Local EMOS 0.85", {}, "table", "Table 2")
    with pytest.raises(StudyError, match="no page 9"):
        made.extract(a, 9, ROW, {}, "table", "Table 2")
    ev = made.extract(a, 2, ROW, {**PROTO_A, "method": "EMOS-loc", "crps": 0.9}, "table", "Table 2")   # layout row
    p = made.record.get(ev["id"]).payload
    assert p["valid_from"] == "2018-05-23" and p["fields"]["crps"] == 0.9
    made.extract(a, 3, "station information in EMOS-loc improves the global score", {"method": "EMOS-loc"},
                 "passage", "5.1")                          # a word broken across lines, in reading order


def test_a_number_not_written_in_the_quote_is_refused(made):
    a = src(made, "1805.09091")
    made.fetch_text(a)
    with pytest.raises(StudyError, match="value of `crps` is not in the quote"):
        made.extract(a, 2, ROW, {"method": "EMOS-loc", "crps": 0.91}, "table", "Table 2")
    passage = "the global score over 1,234 stations; the worst skill score is −0.05"
    with pytest.raises(StudyError, match="value of `skill`"):
        made.extract(a, 3, passage, {"skill": 0.05}, "passage", "5.1")               # the sign is read
    made.extract(a, 3, passage, {"skill": -0.05, "stations": 1234}, "passage", "5.1")
    with pytest.raises(StudyError, match="value of `method` is not in the quote"):
        made.extract(a, 2, ROW, {"method": "EMOS-gl", "crps": 0.9}, "table", "Table 2")
    made.extract(a, 2, ROW, {"method": "emos-LOC", "dataset": "anything: declared free"}, "table", "Table 2")


def test_an_undeclared_field_or_a_wrong_type_is_refused(made):
    a = src(made, "1805.09091")
    made.fetch_text(a)
    with pytest.raises(StudyError, match="`rmse` is not a field of the sealed extraction schema"):
        made.extract(a, 2, ROW, {"rmse": 0.9}, "table", "Table 2")
    with pytest.raises(StudyError, match="field `crps` must be a number"):
        made.extract(a, 2, ROW, {"crps": "0.90"}, "table", "Table 2")
    with pytest.raises(StudyError, match="field `stations` must be an integer"):
        made.extract(a, 3, "the global score over 1,234 stations", {"stations": 1234.0}, "passage", "5.1")
    with pytest.raises(StudyError, match="evidence names its section"):
        made.extract(a, 2, ROW, {"crps": 0.9}, "table", "")
    with pytest.raises(StudyError, match="evidence.kind has the wrong type"):
        made.extract(a, 2, ROW, {"crps": 0.9}, "chart", "Table 2")
    refused = made.record.last("action")
    assert refused.payload["verb"] == "extract" and not refused.payload["outcome"]["ok"]


@pytest.fixture
def rows(made):
    a, b = src(made, "1805.09091"), src(made, "2309.04452")
    made.fetch_text(a)
    made.fetch_text(b)
    ex = lambda s, page, quote, method, crps, proto: made.extract(           # noqa: E731
        s, page, quote, {**proto, "method": method, "crps": crps}, "table", "Table")["id"]
    return {"loc": ex(a, 2, ROW, "EMOS-loc", 0.9, PROTO_A),
            "gl": ex(a, 2, "EMOS-gl Global EMOS 1.01", "EMOS-gl", 1.01, PROTO_A),
            "emos": ex(b, 1, "Method CRPS EMOS 0.82", "EMOS", 0.82, PROTO_B),
            "drn": ex(b, 1, "Method CRPS EMOS 0.82 DRN 0.67", "DRN", 0.67, PROTO_B)}


def test_a_cross_protocol_comparison_is_refused_naming_the_coordinates(made, rows):
    with pytest.raises(StudyError, match="incomparable: the two sides differ on dataset, lead_time") as e:
        made.compare(rows["loc"], rows["emos"], "crps")
    assert "'48 h' vs '24 h'" in e.value.meaning
    env = dispatch(ToolContext(str(made.folder.parent)), "study_compare",
                   {"folder": "s", "a": rows["loc"], "b": rows["emos"], "metric": "crps"})
    assert not env["ok"] and "dataset, lead_time" in env["error"]
    assert not made.record.find("comparison")


def test_a_comparable_comparison_takes_the_metrics_direction(made, rows):
    c = made.compare(rows["loc"], rows["gl"], "crps")["payload"]
    assert c["verdict"] == "better" and c["values"] == [0.9, 1.01] and c["dims"] == PROTO_A    # lower CRPS wins
    assert made.compare(rows["drn"], rows["emos"], "crps")["payload"]["verdict"] == "better"
    assert made.compare(rows["gl"], rows["loc"], "crps")["payload"]["verdict"] == "worse"
    rec = made.record
    links = [("compares", rows["gl"]), ("compares", rows["loc"]), ("under", rec.last("plan").id)]
    with pytest.raises(StudyError, match="does not follow from its sides"):
        rec.add("comparison", {"dims": PROTO_A, "verdict": "better", "metric": "crps", "values": [1.01, 0.9]},
                links=links)
    with pytest.raises(StudyError, match="no metric `skill_score`"):
        made.compare(rows["loc"], rows["gl"], "skill_score")


def test_the_lattice_reports_the_empty_cells(made, rows, tmp_path):
    lat = made.lattice()
    assert lat["columns"] == ["EMOS-loc", "EMOS-gl", "EMOS", "DRN"] and lat["values"] == ["crps", "skill", "stations"]
    assert [r["coordinates"] for r in lat["rows"]] == [PROTO_A, PROTO_B]
    assert [r["gaps"] for r in lat["rows"]] == [["EMOS", "DRN"], ["EMOS-loc", "EMOS-gl"]]
    assert (lat["cells"], lat["filled"], lat["gaps"]) == (8, 4, 4)
    assert lat["rows"][0]["cells"]["EMOS-loc"][0]["crps"] == 0.9
    with pytest.raises(StudyError, match="`crps` is not a text field"):
        made.lattice("crps")
    assert dispatch(ToolContext(str(made.folder.parent)), "study_lattice", {"folder": "s"})["result"]["gaps"] == 4
    assert len(made.evidence(src(made, "2309.04452"))) == 2


def test_recall_is_recomputed_from_the_record(made, monkeypatch):
    work = {"title": "A review", "publication_date": "2021-06-01",
            "referenced_works": ["https://openalex.org/W10", "W11", "W12", "W13"]}
    monkeypatch.setattr(papers, "openalex_work", lambda key: work)
    monkeypatch.setattr(papers, "works", lambda ids, since="": [
        P(title="Neural networks", openalex="W10", arxiv="1805.09091", date="2018-05-23"),
        P(title="Not found", openalex="W11", doi="10.1000/zz", date="2010-01-01"),
        P(title="Too old", openalex="W12", date="1990-01-01")])                   # W13: OpenAlex serves none
    out = made.recall("https://doi.org/10.1175/BAMS-D-19-0308.1")
    assert (out["recall"], out["references"], out["matched"], out["matched_included"]) == (0.5, 2, 1, 0)
    assert out["left_out"] == {"not_served": 1, "outside_dates": 1} and out["missed"][0]["key"] == "doi:10.1000/zz"
    node = made.record.get(out["id"])
    assert node.payload["gold"] == "10.1175/bams-d-19-0308.1"
    with pytest.raises(StudyError, match="does not match the record"):
        made.record.add("recall", {**node.payload, "recall": 1.0}, actor="openlab",
                        links=[("under", made.record.last("plan").id)])


def test_a_review_study_has_no_experiment(made, tmp_path):
    with pytest.raises(StudyError, match="a review study runs no experiment"):
        made.run()
    assert {k: made.status()["trials"][k] for k in ("used", "budget")} == {"used": 0, "budget": 0}
    (tmp_path / "study.toml").write_text(TOML + '\n[split]\nscheme = "kfold"\n', encoding="utf-8")
    with pytest.raises(StudyError, match="`split` has no place in a review study"):
        spec.load(tmp_path)
    (tmp_path / "study.toml").write_text(TOML.split("[review]")[0], encoding="utf-8")
    with pytest.raises(StudyError, match="`review.question` is required"):
        spec.load(tmp_path)
    (tmp_path / "study.toml").write_text(TOML.replace('comparable_on = ["dataset"', 'comparable_on = ["split"'),
                                         encoding="utf-8")
    with pytest.raises(StudyError, match="`review.comparable_on` names split, not in review.coordinates"):
        spec.load(tmp_path)
    (tmp_path / "study.toml").write_text(TOML.replace('crps = "number"', 'crps = {type = "number", free = true}'),
                                         encoding="utf-8")
    with pytest.raises(StudyError, match="`review.extract` must be"):
        spec.load(tmp_path)
