"""Literature as a first-class template: `study new --template review` discovers offline-safe (canned indexes),
screening is batched and attributed per agent, two agents that disagree make a screen_conflict for a person instead
of last-write-wins, and the evidence matrix names the agent behind each number."""

import pytest

from openlab import study
from openlab.agents.identity import acting_as
from openlab.agents.inbox import decide
from openlab.agents.store import Store
from openlab.app.registry import ToolContext, dispatch
from openlab.sources import papers
from openlab.study import cli
from openlab.study.errors import StudyError

P = papers._paper
A = P(title="Ensemble model output statistics for station weather forecasts", doi="10.1000/a", openalex="W1",
      version="published", retracted=False, authors=["Tilmann Gneiting", "Adrian Raftery", "Anton Westveld"],
      abstract="We calibrate ensemble forecasts with EMOS and show gains over raw ensembles at stations.",
      date="2005-05-01", pdf="https://example.org/a.pdf")
B = P(title="Neural network postprocessing of ensemble weather forecasts", arxiv="2001.00001",
      version="preprint", authors=["Stephan Rasp"], abstract="A network post-processes ensemble output at stations.",
      date="2020-01-01")
C = P(title="Station forecasts from ensembles with distributional regression", doi="10.1000/c", openalex="W3",
      version="published", retracted=False, abstract="Distributional regression for station ensemble forecasts.",
      date="2019-03-01")
PAGES = ["Title page.\n", "Table 2: Mean CRPS.\n EMOS-loc Local EMOS 0.90\n Raw ensemble 1.16\n"]
QA = "calibrate ensemble forecasts with EMOS"
QB = "A network post-processes ensemble output"
QC = "Distributional regression for station"


@pytest.fixture
def lit(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENLAB_AGENT", raising=False)
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "openlab.toml").write_text("")
    monkeypatch.setattr(papers, "ask", lambda src, q, n, since: [A, C] if src == "openalex" else [B])
    monkeypatch.setattr(papers, "related", lambda key, n, related_works, since: {"references": [], "cited_by": []})
    monkeypatch.setattr(papers, "crossref_updates", lambda dois: {})
    monkeypatch.setattr(papers, "pdf", lambda url: b"%PDF canned")
    monkeypatch.setattr(papers, "pdf_pages", lambda data, layout=False: PAGES)
    return study.new(ws / "lit", "review")


def ids(s):
    return {k: s.record.last("source", **{f: v}).id for k, (f, v) in
            {"a": ("doi", "10.1000/a"), "b": ("arxiv", "2001.00001"), "c": ("doi", "10.1000/c")}.items()}


def test_the_review_template_scaffolds_a_protocol_and_discovers_offline(lit, capsys):
    assert lit.spec.mode == "review" and lit.spec.review["coverage_floor"] == {"queries_per_source": 2, "hops": 1}
    assert (lit.folder / "README.md").read_text().count("openlab study") >= 5
    assert not (lit.folder / "model.py").exists() and not (lit.folder / "data").exists()
    out = lit.discover()
    assert out["new_sources"] == 3 and out["coverage"]["verdict"] == "sufficient"
    assert lit.status()["sealed"]
    assert cli.main(["new", str(lit.folder.parent / "lit2"), "--template", "review"]) == 0
    assert "openlab study discover" in capsys.readouterr().out
    with pytest.raises(StudyError, match="no template"):
        study.new(lit.folder.parent / "x", "nope")


def test_batch_screening_is_attributed_per_agent(lit, capsys):
    lit.discover()
    i = ids(lit)
    ctx = ToolContext(str(lit.folder.parent))
    with acting_as("screener-a1"):
        env = dispatch(ctx, "study_screen", {"folder": "lit", "decisions": [
            {"source": i["a"], "decision": "include", "criterion": "I1", "rationale": "EMOS at stations", "quote": QA},
            {"source": i["b"], "decision": "exclude", "criterion": "E2", "rationale": "no numbers", "quote": QB},
            {"source": i["c"], "decision": "include", "criterion": "E1", "rationale": "wrong kind", "quote": QC}]})
    assert env["ok"], env
    r = env["result"]
    assert (r["recorded"], r["refused"]) == (2, 1) and "not an include criterion" in r["decisions"][2]["error"]
    with acting_as("screener-b2"):                                  # the single form still works
        assert dispatch(ctx, "study_screen", {"folder": "lit", "source": i["c"], "decision": "include",
                                              "criterion": "I1", "rationale": "stations", "quote": QC})["ok"]
    rows = {s["id"]: s for s in lit.sources()}
    assert rows[i["a"]]["agent"] == "screener-a1" and rows[i["c"]]["agent"] == "screener-b2"
    assert all(n.session in ("screener-a1", "screener-b2") for n in lit.record.find("screen"))
    pr = lit.prisma()
    assert pr["screened_by"] == {"screener-a1": 2, "screener-b2": 1} and pr["included"] == 2 and pr["conflicts"] == 0
    assert cli.main(["sources", str(lit.folder)]) == 0
    assert "by screener-a1" in capsys.readouterr().out


def test_agents_that_disagree_make_a_conflict_for_a_person(lit, capsys, person_key):
    lit.discover()
    i = ids(lit)
    with acting_as("screener-a1"):
        lit.screen(i["a"], "include", "I1", "EMOS at stations", QA)
    with acting_as("screener-a1"):                                  # one agent changing its mind is no conflict
        lit.screen(i["a"], "include", "I1", "still in", QA)
    with acting_as("screener-b2"):
        out = lit.screen(i["a"], "exclude", "E2", "no evaluation", QA)
    assert out["conflict"]
    st = Store(lit.folder.parent)
    item = st.item(out["conflict"])
    assert item["kind"] == "screen_conflict" and item["study"] == "lit" and item["agent"] == "screener-b2"
    assert [o["agent"] for o in item["args"]["options"]] == ["screener-a1", "screener-b2"]
    assert "screener-a1 include (I1)" in item["reason"] and "screener-b2 exclude (E2)" in item["reason"]
    # no decision stands: the source is in conflict, not last-write-wins
    row = next(s for s in lit.sources() if s["id"] == i["a"])
    assert row["decision"] == "" and len(row["conflict"]) == 2
    assert [s["id"] for s in lit.sources("conflict")] == [i["a"]]
    pr = lit.prisma()
    assert pr["conflicts"] == 1 and pr["awaiting_person"] == 1 and pr["included"] == 0
    with acting_as("screener-c3"), pytest.raises(StudyError, match="conflict"):
        lit.screen(i["a"], "include", "I1", "agreed", QA)
    with acting_as("screener-c3"):                                  # a second conflict is not filed twice
        lit.screen(i["b"], "include", "I1", "ok", QB)
    with acting_as("screener-a1"):
        assert lit.screen(i["b"], "exclude", "E2", "no", QB)["conflict"]
    assert len(st.inbox("pending")) == 2
    # the person picks the decision that stands; it is recorded as the person's screen
    with pytest.raises(ValueError, match="--pick 1..2"), person_key.act():
        decide(st, item["id"], True, "read it")
    with person_key.act():
        done = decide(st, item["id"], True, "it has an evaluation table", pick=1)
    assert done["status"] == "approved" and done["result"]["kind"] == "screen"
    row = next(s for s in lit.sources() if s["id"] == i["a"])
    assert (row["decision"], row["by"], row["agent"], row["conflict"]) == ("include", "person", "person:sam", [])
    # a person's screen made directly cancels the pending conflict item of that source
    with person_key.act():
        lit.screen(i["b"], "exclude", "E2", "no evaluation", "", person=True)
    assert st.inbox("pending") == [] and st.inbox("cancelled")[0]["args"]["source"] == i["b"]
    assert cli.main(["prisma", str(lit.folder)]) == 0
    assert "screened_by" in capsys.readouterr().out


def test_an_agents_maybe_lands_in_the_inbox_and_the_person_picks(lit, capsys, person_key):
    """Round 4: the screener's maybe showed in PRISMA as awaiting_person but `openlab inbox` was empty. It is now a
    screen_maybe item with the source, quote and rationale; `openlab inbox approve N --pick include|exclude` (signed)
    records the person's screen."""
    from openlab.agents.cli import inbox_main
    lit.discover()
    i = ids(lit)
    ctx = ToolContext(str(lit.folder.parent))
    with acting_as("openlab-screener-9c6864"):
        env = dispatch(ctx, "study_screen", {"folder": "lit", "decisions": [
            {"source": i["b"], "decision": "maybe", "criterion": "I1", "rationale": "a network, but no stations "
             "table is visible", "quote": QB}]})
    assert env["ok"] and env["result"]["decisions"][0]["inbox"]
    st = Store(lit.folder.parent)
    item = st.item(env["result"]["decisions"][0]["inbox"])
    assert item["kind"] == "screen_maybe" and item["study"] == "lit" and item["agent"] == "openlab-screener-9c6864"
    assert item["args"]["source"] == i["b"] and item["args"]["quote"] == QB
    assert "no stations table" in item["args"]["rationale"] and "no stations table" in item["reason"]
    assert lit.prisma()["awaiting_person"] == 1
    with acting_as("openlab-screener-9c6864"):                       # filed once per source
        with pytest.raises(StudyError, match="waits for a person"):
            lit.screen(i["b"], "maybe", "I1", "again", QB)
    assert len(st.inbox("pending")) == 1
    assert inbox_main(["--root", str(lit.folder.parent)]) == 0
    out = capsys.readouterr().out
    assert "screen_maybe" in out and "pick include|exclude" in out and QB in out
    # approving needs the pick; with it, the person's screen is signed and the source is decided
    assert inbox_main(["--root", str(lit.folder.parent), "approve", str(item["id"]), "--reason", "read it"]) == 1
    assert "--pick include" in capsys.readouterr().err
    assert inbox_main(["--root", str(lit.folder.parent), "approve", str(item["id"]), "--pick", "exclude",
                       "--reason", "no station evaluation"]) == 0
    assert f"#{item['id']} approved by person:sam" in capsys.readouterr().out
    row = next(s for s in lit.sources() if s["id"] == i["b"])
    assert (row["decision"], row["by"], row["agent"]) == ("exclude", "person", "person:sam")
    screen = lit.record.last("screen")
    assert screen.actor == "person" and screen.payload["criterion"] == "E1" and "sig" in screen.payload
    assert lit.prisma()["awaiting_person"] == 0 and st.item(item["id"])["status"] == "approved"
    assert lit.verify()["ok"]


def test_a_persons_screen_settles_a_pending_maybe(lit, person_key):
    lit.discover()
    i = ids(lit)
    with acting_as("screener-a1"):
        out = lit.screen(i["a"], "maybe", "I1", "unsure", QA)
    st = Store(lit.folder.parent)
    assert st.item(out["maybe"])["kind"] == "screen_maybe"
    with person_key.act():
        lit.screen(i["a"], "include", "I1", "it is in", "", person=True)
    assert st.inbox("pending") == [] and st.item(out["maybe"])["status"] == "cancelled"


def test_the_evidence_table_names_source_page_quote_and_agent(lit, capsys):
    lit.discover()
    a = ids(lit)["a"]
    lit.fetch_text(a)
    with acting_as("screener-a1"):
        lit.extract(a, 2, "EMOS-loc Local EMOS 0.90", {"method": "EMOS-loc", "dataset": "German stations 2016",
                                                       "metric": "CRPS", "value": 0.9}, "table", "Table 2")
    rows = lit.evidence_table()
    assert len(rows) == 1
    r = rows[0]
    assert (r["cite"], r["year"], r["method"], r["dataset"], r["metric"], r["value"], r["page"]) == \
        ("Gneiting et al. 2005", 2005, "EMOS-loc", "German stations 2016", "CRPS", 0.9, 2)
    assert r["quote"] == "EMOS-loc Local EMOS 0.90" and r["agent"] == "screener-a1" and r["source"] == "doi:10.1000/a"
    assert cli.main(["evidence", str(lit.folder), "--table"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].split()[:6] == ["CITE", "YEAR", "METHOD", "DATASET", "METRIC", "VALUE"]
    assert "Gneiting et al. 2005" in out and "screener-a1" in out
    assert cli.main(["evidence", str(lit.folder), "--csv"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("cite,year,method,dataset,metric,value,page") and "0.9" in lines[1]
    env = dispatch(ToolContext(str(lit.folder.parent)), "study_evidence", {"folder": "lit", "table": True})
    assert env["ok"] and env["result"]["table"][0]["agent"] == "screener-a1"
