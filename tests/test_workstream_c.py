"""Workstream C (2026-10-03), no network: snapshot.build(workers=N) on a fake provider, paper-search query cleanup
and relevance."""

import json
import threading
import time

import polars as pl
import pytest


# --- snapshot.build(workers=N) ----------------------------------------------------------------------------
@pytest.fixture
def fake_provider(tmp_path, monkeypatch):
    import openlab.sources as S
    monkeypatch.setenv("OPENLAB_SNAPSHOTS", str(tmp_path / "snaps"))
    calls, lock = [], threading.Lock()

    def fetch(provider, endpoint, **args):
        with lock:
            calls.append(args["n"])
        time.sleep(0.01)                               # let the workers overlap
        if endpoint == "boom":
            raise RuntimeError("provider down")
        return pl.DataFrame({"x": list(range(args["n"]))})
    monkeypatch.setattr(S, "fetch", fetch)
    monkeypatch.setattr(S, "limits", lambda name: {"per_second": 1})
    return calls


def _spec(name):
    ds = [{"dataset": f"d{i}", "provider": "fake", "endpoint": "rows", "args": {"n": i + 1}} for i in range(8)]
    ds += [{"dataset": "bad", "provider": "fake", "endpoint": "boom", "args": {"n": 1}},
           {"dataset": "empty", "provider": "fake", "endpoint": "rows", "args": {"n": 0}}]
    return {"name": name, "datasets": ds}


def test_snapshot_workers_match_one_worker(fake_provider, tmp_path):
    from openlab.sources import snapshot
    par = snapshot.build(_spec("par"), workers=4)
    seq = snapshot.build(_spec("seq"), workers=1)
    assert par["datasets"] == seq["datasets"] == {f"d{i}": i + 1 for i in range(8)}
    assert set(par["failed"]) == {"bad", "empty"} and "provider down" in par["failed"]["bad"]
    assert "0 rows" in par["failed"]["empty"]
    man = json.loads((tmp_path / "snaps" / "par" / "manifest.json").read_text())
    assert sorted(man["datasets"]) == sorted(f"d{i}" for i in range(8))   # every concurrent write kept, once each
    assert snapshot.read("par", "d3").height == 4


def test_snapshot_resume_skips_built_datasets(fake_provider):
    from openlab.sources import snapshot
    snapshot.build(_spec("r"), workers=4)
    n = len(fake_provider)
    again = snapshot.build(_spec("r"), workers=4)
    assert again["built"] == [] and len(fake_provider) - n == 2       # only the two failures are retried


# --- paper search -----------------------------------------------------------------------------------------
def test_oa_query_strips_punctuation_and_caps_words():
    from openlab.sources.papers import _oa_query
    q = 'Momentum: "crash" risk (equity) markets; volatility, scaling [Barroso] daniel moskowitz returns factor alpha'
    out = _oa_query(q)
    assert out.split() == ["momentum", "crash", "risk", "equity", "markets", "volatility", "scaling", "barroso",
                           "daniel", "moskowitz"]
    assert not set(out) & set(':"();,[]')
    assert _oa_query("a of") == "a of"                 # nothing left: the query as given


def test_relevance_title_beats_abstract():
    from openlab.sources.papers import _paper, relevance
    q = "momentum crash"
    t = _paper(title="Momentum crash", abstract="")
    a = _paper(title="Other", abstract="a momentum crash study")
    half = _paper(title="Momentum", abstract="")
    assert relevance(t, q) == 1.0 and relevance(a, q) == 0.5 and relevance(half, q) == 0.5
    assert relevance(_paper(title="Gardening"), q) == 0.0


def test_search_orders_by_relevance_and_drops_off_topic_crossref(monkeypatch):
    from openlab.sources import papers
    P = papers._paper
    oa = [P(title="Other", abstract="momentum crash evidence", doi="10.1/a"),
          P(title="Momentum crash risk", abstract="", doi="10.1/b")]
    monkeypatch.setattr(papers, "openalex", lambda q, n, since, oa_, newest: [] if newest else oa)
    monkeypatch.setattr(papers, "arxiv", lambda q, n, newest: [])
    monkeypatch.setattr(papers, "crossref", lambda q, n, since: [P(title="Gardening tips", doi="10.1/c"),
                                                                  P(title="Momentum crash again", doi="10.1/d")])
    out = papers.search(["momentum crash"])
    titles = [p["title"] for p in out["papers"]]
    assert "Gardening tips" not in titles and out["failed"] == []
    assert titles.index("Momentum crash risk") < titles.index("Other")
