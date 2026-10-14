"""Guarantees of the literature engine, enforced by the record's schema: a work is recorded once, coverage is
recomputed (never typed), a screening quote must be in the source, a criterion must fit its decision, an agent
cannot overrule a person, and the PRISMA flow is derived from the record. Index calls are canned: no network."""

import io
import sys

import pytest

from openlab.study import literature

from openlab import study
from openlab.app.registry import ToolContext, dispatch
from openlab.sources import papers
from openlab.study import cli, spec
from openlab.study.errors import StudyError
from openlab.study.schema import Source

TOML = """
[study]
name = "lit"
question = "q?"
mode = "static"

[split]
scheme = "kfold"
k = 5
fraction = 0.2

[evaluate]
metrics = ["auc"]

[budget]
trials = 3

[review]
question = "Does EMOS beat the raw ensemble at stations?"
sources = ["openalex", "arxiv"]
queries = ["ensemble postprocessing", "EMOS"]
include = [{id = "I1", text = "post-processing of ensemble weather forecasts"}]
exclude = [{id = "E1", text = "not about weather"}, {id = "E2", text = "no evaluation"}]
n = 5
coverage_floor = {queries_per_source = 2, hops = 1}
"""

P = papers._paper
A = P(title="Ensemble model output statistics for weather", doi="10.1000/a", openalex="W1", version="published",
      retracted=False, abstract="We calibrate ensemble forecasts with EMOS\nand show gains over raw ensembles at "
      "stations.", date="2005-05-01")
B = P(title="Neural network postprocessing of ensemble weather forecasts", arxiv="2001.00001", doi="10.1000/b",
      journal_doi="10.1000/b", version="preprint", abstract="A network post-processes ensemble output.",
      date="2020-01-01", pdf="https://arxiv.org/pdf/2001.00001")
C = P(title="Neural network postprocessing of ensemble weather forecasts", doi="10.1000/b", openalex="W3",
      version="published", retracted=True, abstract="Published version.", date="2021-03-01")
D = P(title="Station forecasts from ensembles", doi="10.1000/d", openalex="W4", version="published", retracted=False)
E = P(title="Probabilistic forecasting review", doi="10.1000/e", openalex="W5", version="published", retracted=False,
      pdf="https://example.org/e.pdf")
F = P(title="Citing work", openalex="W6", version="published", retracted=False)
PAGES = ["Page one.\n", "Page two: stations calibrated with EMOS\n beat the raw ensemble in every season.\n"]


@pytest.fixture
def made(tmp_path, monkeypatch):
    folder = tmp_path / "s"
    folder.mkdir()
    (folder / "study.toml").write_text(TOML, encoding="utf-8")
    answers = {("openalex", "ensemble postprocessing"): [A, C], ("openalex", "EMOS"): [A, D],
               ("arxiv", "ensemble postprocessing"): [B], ("arxiv", "EMOS"): [B]}
    monkeypatch.setattr(papers, "ask", lambda src, q, n, since: answers[(src, q)])
    monkeypatch.setattr(papers, "related", lambda key, n, related_works, since: {
        "references": [E], "cited_by": [C, F]} if key == "W1" else {"references": [], "cited_by": []})
    monkeypatch.setattr(papers, "crossref_updates", lambda dois: {"10.1000/d": [
        {"type": "retraction", "notice": "10.1000/notice", "date": "2020-01-02"}]})
    monkeypatch.setattr(papers, "pdf", lambda url: b"%PDF-1.4 canned")
    monkeypatch.setattr(papers, "pdf_pages", lambda data, layout=False: PAGES)
    return study.open(folder)


def by_doi(s, doi):
    return s.record.last("source", doi=doi)


def test_discover_records_the_search_its_versions_retractions_and_coverage(made, tmp_path):
    out = made.discover()
    rec = made.record
    cov = rec.get(out["id"]).payload
    assert {k: cov[k] for k in ("queries", "hops", "edges", "records", "unique", "failed", "verdict")} == \
        {"queries": {"openalex": 2, "arxiv": 2}, "hops": 1, "edges": 3, "records": 9, "unique": 6, "failed": 0,
         "verdict": "sufficient"}
    assert out["queries"] == 12 and out["new_sources"] == 6 and len(rec.find("source")) == 6
    pre, pub = rec.last("source", arxiv="2001.00001"), by_doi(made, "10.1000/b")
    assert pre.payload["doi"] == "" and pre.payload["published_as"] == "10.1000/b" and pre.payload["tier"] == "publisher"
    assert rec.links(pre.id, rel="version_of") == [(pre.id, "version_of", pub.id)]
    assert pub.payload["retracted"] and pub.payload["retraction_note"] == "OpenAlex is_retracted"
    d = by_doi(made, "10.1000/d").payload
    assert d["retracted"] and "Crossref retraction notice 10.1000/notice" in d["retraction_note"]
    assert by_doi(made, "10.1000/a").payload["retraction_checked"] == ["openalex", "crossref"]
    assert pre.payload["retraction_checked"] == []
    act = rec.last("action")
    assert act.payload["verb"] == "discover" and rec.links(act.id, rel="did")[0][2] == out["id"]
    again = dispatch(ToolContext(str(tmp_path)), "study_discover", {"folder": "s"})["result"]
    assert again["new_sources"] == 0 and rec.get(again["id"]).payload["unique"] == 6     # a re-run: no new source
    assert rec.verify()["ok"]


def test_a_duplicate_or_unnormalised_id_is_refused_by_the_record(made):
    made.discover()
    rec = made.record
    q = rec.find("query", source="openalex", text="ensemble postprocessing")[0]
    body = dict(title="x", version_kind="published", tier="scholarly_index", retracted=False, retraction_checked=[])
    with pytest.raises(StudyError, match="already on record"):
        rec.add(Source(**body, doi="10.1000/a"), actor="openlab", links=[("found_by", q.id)])
    with pytest.raises(StudyError, match="stored form"):
        rec.add(Source(**body, doi="https://doi.org/10.1000/A"), actor="openlab", links=[("found_by", q.id)])
    with pytest.raises(StudyError, match="did not return"):
        rec.add(Source(**body, doi="10.1000/zzz"), actor="openlab", links=[("found_by", q.id)])
    with pytest.raises(StudyError, match="only be written by openlab"):
        rec.add(Source(**body, doi="10.1000/a"), actor="agent", links=[("found_by", q.id)])


def test_typed_coverage_that_disagrees_with_its_queries_is_refused(made):
    out = made.discover()
    rec = made.record
    cov = rec.get(out["id"])
    links = [(rel, dst) for _, rel, dst in rec.links(cov.id)]
    for field, wrong in (("unique", 7), ("records", 10), ("edges", 2), ("verdict", "insufficient"),
                         ("queries", {"openalex": 3, "arxiv": 2})):
        with pytest.raises(StudyError, match="does not match its queries"):
            rec.add("coverage", {**cov.payload, field: wrong}, actor="openlab", links=links)
    some = [l for l in links if l[0] == "under"] + [l for l in links if l[0] == "of"][:4]   # no hops: insufficient
    with pytest.raises(StudyError, match="verdict: typed sufficient, recomputed insufficient"):
        rec.add("coverage", {**cov.payload, "hops": 0, "edges": 0, "records": 6, "unique": 4}, links=some)
    assert rec.add("coverage", cov.payload, actor="openlab", links=links).payload == cov.payload


def test_a_screen_quote_must_be_in_the_abstract_or_the_full_text(made, tmp_path):
    made.discover()
    a, e = by_doi(made, "10.1000/a"), by_doi(made, "10.1000/e")
    made.screen(a.id, "include", "I1", "EMOS at stations", "calibrate ensemble forecasts with EMOS and show gains")
    with pytest.raises(StudyError, match="not in the source"):
        made.screen(e.id, "include", "I1", "about EMOS", "stations calibrated with EMOS beat the raw ensemble")
    with pytest.raises(StudyError, match="too short"):
        made.screen(e.id, "include", "I1", "about EMOS", "EMOS")
    doc = made.fetch_text(e.id)
    read = dispatch(ToolContext(str(tmp_path)), "study_fetch_text", {"folder": "s", "source": e.id, "page": 2})
    assert read["result"]["id"] == doc["id"] and read["result"]["text"] == PAGES[1]      # fetched once, then read
    assert doc["payload"]["n_pages"] == 2
    made.screen(e.id, "include", "I1", "about EMOS", "stations calibrated with EMOS beat the raw ensemble")
    bad = {**doc["payload"], "n_pages": 3}
    with pytest.raises(StudyError, match="not a list of 3 page texts"):
        made.record.add("document", bad, actor="openlab", links=[("of", e.id)])


def test_the_cli_fetches_a_page_and_screen_says_quote_is_required(made, capsys):
    made.discover()
    e = by_doi(made, "10.1000/e").id
    assert cli.main(["fetch", str(made.folder), e, "--page", "2"]) == 0
    out = capsys.readouterr().out
    assert "--- page 2 of 2" in out and PAGES[1] in out
    assert cli.main(["fetch", str(made.folder), e, "--page", "3"]) == 1
    assert "no page 3" in capsys.readouterr().err
    assert cli.main(["screen", str(made.folder), e, "--decision", "include", "--criterion", "I1",
                     "--rationale", "about EMOS"]) == 1
    assert "screen needs --quote" in capsys.readouterr().err
    assert "required" in cli._parser()._subparsers._group_actions[0].choices["screen"].format_help()


def test_a_criterion_must_fit_its_decision(made):
    made.discover()
    a = by_doi(made, "10.1000/a").id
    quote = "calibrate ensemble forecasts with EMOS"
    with pytest.raises(StudyError, match="`I1` is not an exclude criterion"):
        made.screen(a, "exclude", "I1", "off topic", quote)
    with pytest.raises(StudyError, match="`E1` is not an include criterion"):
        made.screen(a, "include", "E1", "on topic", quote)
    with pytest.raises(StudyError, match="`X9` is not a criterion"):
        made.screen(a, "maybe", "X9", "unsure", quote)
    made.screen(a, "maybe", "E2", "does it evaluate?", quote)


class Tty(io.StringIO):
    def isatty(self):
        return True


def test_an_agent_cannot_overrule_a_person_or_resolve_its_own_maybe(made, tmp_path, monkeypatch, capsys, person_key):
    made.discover()
    a = by_doi(made, "10.1000/a").id
    quote = "calibrate ensemble forecasts with EMOS"
    made.screen(a, "maybe", "E2", "does it evaluate?", quote)
    with pytest.raises(StudyError, match="waits for a person"):
        made.screen(a, "include", "I1", "it does", quote)
    with pytest.raises(StudyError, match="only the agent marks a source maybe"), person_key.act():
        made.screen(a, "maybe", "E2", "unsure", quote, person=True)
    ctx = ToolContext(str(tmp_path))
    args = {"folder": "s", "source": a, "decision": "exclude", "criterion": "E2", "rationale": "no evaluation"}
    env = dispatch(ctx, "study_screen_person", args)
    assert not env["ok"] and env["error"] == "a reviewer's act"
    monkeypatch.setattr(sys, "stdin", io.StringIO("lit\n"))       # piped, not a person at a terminal
    person_key.answers("")                                         # and no passphrase typed
    assert cli.main(["screen", str(made.folder), a, "--decision", "exclude", "--criterion", "E2",
                     "--rationale", "no evaluation", "--person"]) == 1
    assert "in their own terminal" in capsys.readouterr().err
    assert not dispatch(ctx, "study_screen_person", args, human=True)["ok"]   # the reviewer secret alone: unsigned
    with person_key.act():
        assert dispatch(ctx, "study_screen_person", args, human=True)["ok"]   # a person's decision, no quote
    with pytest.raises(StudyError, match="a person has decided"):
        made.screen(a, "include", "I1", "it does", quote)
    with person_key.act():
        made.screen(a, "include", "I1", "on reflection", "", person=True)      # a person may change their mind
    assert made.sources("include")[0]["by"] == "person"


def test_prisma_is_derived_from_the_record(made, tmp_path, person_key):
    made.discover()
    ids = {d: by_doi(made, d).id for d in ("10.1000/a", "10.1000/b", "10.1000/d")}
    pre = made.record.last("source", arxiv="2001.00001").id
    made.screen(ids["10.1000/a"], "include", "I1", "EMOS", "calibrate ensemble forecasts with EMOS")
    with person_key.act():
        made.screen(ids["10.1000/b"], "include", "I1", "published version", "", person=True)
        made.screen(ids["10.1000/d"], "exclude", "E2", "no evaluation", "", person=True)
    made.screen(pre, "maybe", "E2", "abstract is thin", "A network post-processes ensemble output.")
    cov = made.discover()["id"]                          # a re-run is the search now; its records are not added twice
    assert made.prisma() == {
        "coverage": cov, "identified": 9, "by_source": {"openalex": 4, "arxiv": 2, "citations": 3},
        "duplicates_removed": 3, "records": 6, "screened": 3, "awaiting_person": 1, "conflicts": 0,
        "screened_by": {"unattributed": 2, "person:sam": 2}, "not_screened": 2, "excluded": 1,
        "excluded_by_criterion": {"E2": 1}, "included": 2, "retracted": 2, "retracted_included": [ids["10.1000/b"]],
        "earlier_only": 0}
    env = dispatch(ToolContext(str(tmp_path)), "study_sources", {"folder": "s", "status": "unscreened", "n": 1})
    assert env["ok"] and env["result"]["total"] == 2 and len(env["result"]["sources"]) == 1
    assert dispatch(ToolContext(str(tmp_path)), "study_prisma", {"folder": "s"})["result"]["included"] == 2


def test_queries_follow_the_sealed_protocol_and_the_floor_is_a_persons(made):
    made.discover()
    rec = made.record
    under = rec.last("plan").id
    q = {"source": "openalex", "text": "EMOS", "filters": {"since": "", "n": 5}, "n_hits": 0, "hits": [], "page": {}}
    rec.add("query", q, links=[("under", under)])
    with pytest.raises(StudyError, match="not in the sealed review"):
        rec.add("query", {**q, "text": "something else"}, links=[("under", under)])
    with pytest.raises(StudyError, match="filters are not the review's"):
        rec.add("query", {**q, "filters": {"since": "", "n": 50}}, links=[("under", under)])
    with pytest.raises(StudyError, match="names its seed"):
        rec.add("query", {**q, "source": "citations", "filters": {"direction": "references", "hop": 1, "n": 5, "since": ""}},
                links=[("under", under)])
    toml = made.folder / "study.toml"
    toml.write_text(TOML.replace('"EMOS"]', '"EMOS", "ensemble calibration"]'), encoding="utf-8")
    made.amend("a synonym the first search surfaced")
    toml.write_text(toml.read_text().replace("queries_per_source = 2", "queries_per_source = 1"), encoding="utf-8")
    with pytest.raises(StudyError, match="review.coverage_floor can only be amended by a person"):
        made.amend("lower the bar")


@pytest.mark.parametrize("old, new, key", [
    ('sources = ["openalex", "arxiv"]', 'sources = ["scholar"]', "review.sources"),
    ('include = [{id = "I1", text = "post-processing of ensemble weather forecasts"}]\n', "", "review.include"),
    ('{id = "E2"', '{id = "I1"', "review.exclude"),
    ("hops = 1}", "hops = 2}", "review.coverage_floor"),
    ('n = 5', 'n = 0', "review.n"),
])
def test_the_review_protocol_is_validated(tmp_path, old, new, key):
    (tmp_path / "study.toml").write_text(TOML.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(StudyError) as e:
        spec.load(tmp_path)
    assert f"`{key}`" in e.value.error


def test_hops_seed_from_the_included_sources_once_any_is_included_and_keep_since(made, monkeypatch):
    (made.folder / "study.toml").write_text(TOML.replace("n = 5", 'n = 5\nsince = "2005-01-01"'), encoding="utf-8")
    calls = []
    monkeypatch.setattr(papers, "related", lambda key, n, related_works, since: calls.append((key, since)) or {
        "references": [], "cited_by": []})
    made.discover()
    assert len(calls) == 4 and {s for _, s in calls} == {"2005-01-01"}        # nothing included: every source
    made.screen(by_doi(made, "10.1000/a").id, "include", "I1", "EMOS", "calibrate ensemble forecasts with EMOS")
    calls.clear()
    made.discover()
    assert calls == [("W1", "2005-01-01")]
    q = made.record.find("query", source="citations")[-1].payload
    assert q["filters"] == {"direction": "cited_by", "hop": 1, "n": 5, "since": "2005-01-01"}


def test_a_source_is_looked_up_by_id_through_the_index(made):
    from contextlib import closing
    made.discover()
    with closing(made.record._con()) as con:
        plan = con.execute("EXPLAIN QUERY PLAN SELECT * FROM nodes WHERE kind=? AND json_extract(payload, '$.doi')=? "
                           "ORDER BY seq", ("source", "10.1000/a")).fetchall()
    assert any("nodes_doi" in str(tuple(r)) for r in plan)
    assert [n.payload["title"] for n in made.record.find("source", doi="10.1000/a", retracted=False)] == [A["title"]]


def test_a_fetch_is_retried_after_a_429_and_not_after_a_404(monkeypatch):
    import urllib.error
    from openlab.sources import http
    calls = []

    class Reply(io.BytesIO):
        headers: dict = {}

    def urlopen(req, timeout):
        calls.append(req.full_url)
        if len(calls) == 1 or "missing" in req.full_url:
            raise urllib.error.HTTPError(req.full_url, 404 if "missing" in req.full_url else 429, "no", {}, None)
        return Reply(b"ok")
    monkeypatch.setattr(http.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    assert http.get_bytes("https://example.org/a", cache=False) == b"ok" and len(calls) == 2
    with pytest.raises(urllib.error.HTTPError):
        http.get_bytes("https://example.org/missing", cache=False)
    assert len(calls) == 3


def test_discover_runs_its_calls_side_by_side_with_progress_timings_and_one_answer_per_identical_call(
        made, monkeypatch, capsys):
    import time
    from openlab.study import cli
    asked = []
    slow = {("openalex", "ensemble postprocessing"): [A, C], ("openalex", "EMOS"): [A, D],
            ("arxiv", "ensemble postprocessing"): [B], ("arxiv", "EMOS"): [B]}

    def ask(src, q, n, since):
        asked.append((src, q))
        time.sleep(0.3)
        return slow[(src, q)]
    monkeypatch.setattr(papers, "ask", ask)
    assert cli.main(["discover", str(made.folder)]) == 0
    io = capsys.readouterr()
    assert "openalex q2/4: 2 hits (0.3 s)" in io.err and "arxiv q4/4: 1 hits" in io.err
    assert "citations hop 1 1/" in io.err
    assert sorted(asked) == sorted(slow)                           # each distinct call once
    assert "running them side by side saved" in io.out
    out = made.discover()                                          # the reply carries every call's timing
    qs = [t for t in out["timings"] if t["kind"] == "queries"]
    assert len(qs) == 4 and all(t["seconds"] >= 0.3 for t in qs) and out["cached_calls"] == 0
    assert out["seconds"] < out["calls_seconds"]                   # 4 slow calls took less than 4 x 0.3 s
    assert made.record.verify()["ok"]
    calls, n = literature._Calls(), []                             # the same call twice in one search: asked once
    got = calls.run("queries", [(("k",), "x", lambda: n.append(1) or [A]), (("k",), "x", lambda: n.append(1) or [A])])
    assert got == [([A], ""), ([A], "")] and n == [1] and calls.reused == 1


def test_find_gives_the_page_and_the_exact_sentence_extract_accepts(made, monkeypatch, capsys):
    text = ["Intro page.\n", "We study adaptive conformal inference. The realised cover-\nage of ACI was 0.90 on "
            "the electricity data. Short one.\n"]
    layout = ["Intro page.\n", "Method      Coverage   Width\nACI         0.90       1.23\nSPCI        0.91       1.10\n"]
    monkeypatch.setattr(papers, "pdf_pages", lambda data, layout_=False, **kw: layout if kw.get("layout") else text)
    made.discover()
    src = by_doi(made, "10.1000/e").id                       # E has an open-access PDF
    out = made.find(src, "0.9")                              # fetched on first use; a number is found by value
    assert out["total"] == 2 and [h["page"] for h in out["hits"]] == [2, 2]
    sentence = next(h for h in out["hits"] if h["form"] == "text")["quote"]
    assert sentence == "The realised coverage of ACI was 0.90 on the electricity data."
    row = next(h for h in out["hits"] if h["form"] == "table row")["quote"]
    assert row == "Method Coverage Width ACI 0.90 1.23"            # a row under 20 characters takes its header
    assert made.find(src, "realised coverage")["hits"][0]["quote"] == sentence
    assert made.find(src, "nothing like this")["total"] == 0
    toml = made.folder / "study.toml"
    toml.write_text(toml.read_text() + '\n[review.extract]\nmethod = "text"\nvalue = "number"\n')
    made.amend("the extraction schema")
    ev = made.extract(src, 2, sentence, {"method": "ACI", "value": 0.9}, "passage", "4 Results")
    assert ev["payload"]["quote"] == sentence                    # the quote find gave is accepted as is
    assert cli.main(["find", str(made.folder), src, "0.90", "--around", "1"]) == 0
    printed = capsys.readouterr().out
    assert 'p.2 (text): "We study adaptive conformal inference. The realised coverage of ACI was 0.90' in printed
    assert "next: openlab study extract" in printed
    env = dispatch(ToolContext(str(made.folder.parent)), "study_find", {"folder": "s", "source": src, "text": "SPCI"})
    assert env["ok"] and env["result"]["hits"][0]["quote"] == "ACI 0.90 1.23 SPCI 0.91 1.10"
