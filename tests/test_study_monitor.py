"""Guarantees of continuous research: a monitoring pass records new sources and each retraction notice once, as a
`retraction` node linked to its source (bitemporal: the notice's date, and when openlab found it); its impact lists
the claims and papers resting on the source; a second pass duplicates nothing; the signed paper then fails to verify,
naming the retraction, and a rebuild prints the citation as retracted; the watch node cannot be typed; the [monitor]
cron line touches only its own crontab line (a fake crontab). The indexes are canned: no network."""

import json

import pytest

from openlab import study
from openlab.app.registry import ToolContext, dispatch
from openlab.study.signing import sign_file
from openlab.sources import papers
from openlab.study import cli
from openlab.study.errors import StudyError
from openlab.study.paper import sections
from openlab.study.schema import Retraction

TOML = """
[study]
name = "w"
question = "Which post-processing methods beat EMOS?"
mode = "review"

[review]
question = "Which post-processing methods beat EMOS?"
sources = ["openalex"]
queries = ["ensemble postprocessing"]
include = [{id = "I1", text = "post-processing of ensemble weather forecasts"}]
exclude = [{id = "E1", text = "not about weather"}]
since = "2005-01-01"
citation_hops = 0
n = 5
coverage_floor = {queries_per_source = 1, hops = 0}

[review.extract]
method = "text"
crps = "number"

[monitor]
cron = "0 6 * * 1"
"""

P = papers._paper
A = P(title="Neural networks for postprocessing ensemble weather forecasts", doi="10.1000/a", openalex="W1",
      version="published", date="2018-05-23", pdf="https://example.org/a.pdf", retracted=False,
      abstract="Ensemble weather predictions require statistical post-processing of systematic errors.")
B = P(title="Station forecasts from ensembles", doi="10.1000/b", openalex="W2", version="published",
      date="2020-01-02", retracted=False, abstract="Stations.")
D = P(title="A new look at ensemble postprocessing", doi="10.1000/d", openalex="W4", version="published",
      date="2026-09-20", retracted=False, abstract="New.")
PAGES = ["Table 2: Mean CRPS.\n EMOS-loc Local EMOS 0.90\n"]
NOTICE = {"type": "retraction", "notice": "10.1000/notice", "date": "2026-09-01"}
SHA = "a" * 64


@pytest.fixture
def s(tmp_path, monkeypatch):
    hits, notices = [A, B], {}
    monkeypatch.setattr(papers, "ask", lambda src, q, n, since: list(hits))
    monkeypatch.setattr(papers, "crossref_updates", lambda dois: {d: notices.get(d, []) for d in dois})
    monkeypatch.setattr(papers, "works", lambda ids, since="": [{"openalex": i, "retracted": False} for i in ids])
    monkeypatch.setattr(papers, "pdf", lambda url: b"%PDF a")
    monkeypatch.setattr(papers, "pdf_pages", lambda data, layout=False: PAGES)
    monkeypatch.setenv("OPENLAB_ALERT_DRY_RUN", "1")
    monkeypatch.setenv("OPENLAB_SIGNING_KEY", str(tmp_path / "signing.key"))
    folder = tmp_path / "s"
    folder.mkdir()
    (folder / "study.toml").write_text(TOML, encoding="utf-8")
    st = study.open(folder)
    st.discover()                                           # seals study.toml
    st.hits, st.notices = hits, notices
    return st


def paper(s, claim: str, src: str):
    """A paper node rendering the claim and citing the source, with its signed PDF, as `study paper` leaves them."""
    rec = s.record
    prose = {"paper/paper.toml": 'title = "t"\nauthors = ["a"]\nabstract = "a.tex"\nsections = ["results"]\n',
             "paper/a.tex": f"\\labclaim{{{claim}}} \\labcite{{{src}}}{{statistical post-processing of systematic errors}}"}
    pdf = b"%PDF-1.5 the paper"
    node = rec.add("paper", {"tex_sha": SHA, "pdf_sha": rec.put_blob(pdf), "bindings": sorted([claim, src]),
                             "lint": [], "head": rec.head(), "sources": rec.put_blob(json.dumps(prose).encode())},
                   links=[("renders", claim), ("renders", src)])
    build = s.folder / "paper" / "build"
    build.mkdir(parents=True)
    (build / "main.pdf").write_bytes(pdf)
    sign_file(build / "main.pdf", {"kind": "openlab-paper", "paper": node.id, "head": node.hash})
    return node


def test_a_retraction_is_recorded_once_with_its_impact_and_the_paper_goes_stale(s, capsys):
    rec = s.record
    src = rec.last("source", doi="10.1000/a").id
    s.fetch_text(src)
    ev = s.extract(src, 1, "EMOS-loc Local EMOS 0.90", {"method": "EMOS-loc", "crps": 0.9}, "table", "Table 2")["id"]
    claim = s.claim("Local EMOS scores a CRPS of 0.90.", "observed", [ev])["id"]
    p = paper(s, claim, src)
    assert s.verify_paper()["ok"]

    s.hits.append(D)
    s.notices["10.1000/a"] = [NOTICE, {"type": "correction", "notice": "10.1000/fix", "date": "2026-08-01"}]
    w = s.monitor()
    [r] = rec.find("retraction")
    assert (r.actor, r.payload) == ("openlab", {"via": "crossref", "type": "retraction", "notice": "10.1000/notice",
                                               "valid_from": "2026-09-01"})
    assert rec.links(r.id, rel="of") == [(r.id, "of", src)]
    [hit] = w["retractions"]
    assert (hit["retraction"], hit["source"], hit["evidence"], hit["claims"], hit["papers"]) == \
        (r.id, src, [ev], [claim], [p.id])
    assert w["new_sources"] == [rec.last("source", doi="10.1000/d").id]
    node = rec.get(w["id"])
    assert node.payload["checked"] == {src: ["crossref", "openalex"]}
    assert [(rel, d) for _, rel, d in rec.links(node.id)] == [("of", w["coverage"]), ("found", r.id)]
    assert w["summary"] == "1 new source matches the protocol; doi:10.1000/a was retracted: 1 claim and 1 paper " \
                           "depend on it"
    sent = json.loads((rec.dir / "outbox.jsonl").read_text().splitlines()[-1])
    assert w["summary"] in sent["subject"] and claim in sent["body"]
    assert rec.last("action").payload["verb"] == "monitor" and rec.links(rec.last("action").id) == \
        [(rec.last("action").id, "did", node.id)]

    v = s.verify_paper()
    assert not v["ok"] and "the paper is stale" in v["why"] and "10.1000/notice" in v["why"]
    assert v["retracted"] == [{"retraction": r.id, "source": src, "note": "Crossref retraction notice 10.1000/notice "
                               "(2026-09-01)", "bindings": sorted([claim, src])}]
    assert cli.main(["verify-paper", str(s.folder)]) == 1 and "stale" in capsys.readouterr().out
    ctx = sections.Ctx(s, 3, s.folder, [], {})
    assert sections.cite(ctx, rec.get(src)).endswith(r"\textbf{[retracted]}") and "RETRACTED" in sections.bibtex(ctx)
    with pytest.raises(StudyError, match="which is retracted") as e:      # a new claim on it must say so
        s.claim("Local EMOS scores 0.90.", "observed", [ev])
    assert "10.1000/notice" in e.value.meaning
    assert [x["retracted"] for x in s.sources("retracted")] == [True] and s.prisma()["retracted"] == 1

    assert cli.main(["monitor", str(s.folder)]) == 0                       # a second pass duplicates nothing
    assert "nothing new" in capsys.readouterr().out
    assert len(rec.find("retraction")) == 1 and len(rec.find("watch")) == 2
    assert rec.links(rec.last("watch").id, rel="found") == [] and len(rec.find("source", doi="10.1000/d")) == 1
    with pytest.raises(StudyError, match="already has retraction notice"):
        rec.add(Retraction(via="crossref", type="retraction", notice="10.1000/notice"), actor="openlab",
                links=[("of", src)])


def test_the_schema_refuses_a_typed_watch_or_retraction(s):
    rec = s.record
    b = rec.last("source", doi="10.1000/b").id
    body = {"via": "openalex", "type": "is_retracted", "notice": "W9"}
    with pytest.raises(StudyError, match="not a openalex retraction"):
        rec.add("retraction", body, actor="openlab", links=[("of", b)])
    with pytest.raises(StudyError, match="can only be written by openlab"):
        rec.add("retraction", {**body, "notice": "W2"}, actor="agent", links=[("of", b)])
    r = rec.add("retraction", {**body, "notice": "W2"}, actor="openlab", links=[("of", b)])
    empty = {"new_sources": [], "contradicting": [], "outcomes": [], "checked": {}}
    with pytest.raises(StudyError, match="does not match the record"):
        rec.add("watch", empty, actor="openlab")                          # it must link the retraction found
    with pytest.raises(StudyError, match="does not match the record"):
        rec.add("watch", {**empty, "outcomes": ["outcome:000000000000"]}, actor="openlab", links=[("found", r.id)])
    cov = rec.find("coverage")[0].id
    new = [rec.find("source")[0].id, rec.find("source")[1].id]
    assert rec.add("watch", {**empty, "new_sources": new}, actor="openlab", links=[("of", cov), ("found", r.id)])
    with pytest.raises(StudyError, match="not a discovery since the previous pass"):
        rec.add("watch", {**empty, "new_sources": new}, actor="openlab", links=[("of", cov)])


def test_the_monitor_schedule_touches_only_its_own_crontab_line(s, tmp_path, monkeypatch):
    fake, tab = tmp_path / "crontab", tmp_path / "crontab.txt"
    fake.write_text(f'#!/bin/sh\nif [ "$1" = "-l" ]; then cat "{tab}" 2>/dev/null || exit 1; else cat > "{tab}"; fi\n')
    fake.chmod(0o755)
    monkeypatch.setenv("OPENLAB_CRONTAB", str(fake))
    tab.write_text("0 1 * * * echo owned\n")
    assert s.monitor_schedule()["state"] == "off"
    on = dispatch(ToolContext(str(s.folder.parent)), "study_monitor", {"folder": "s", "schedule": "on"})["result"]
    owned, line = tab.read_text().splitlines()
    assert owned == "0 1 * * * echo owned" and on["state"] == "on" and on["next_run"]
    assert line.startswith("0 6 * * 1 ") and line.endswith(f"# openlab:{s.folder.parent}::{s.folder.name}:watch")
    assert "study monitor" in line and "monitor.log" in line
    s.monitor_schedule("off")
    assert tab.read_text().splitlines() == ["0 1 * * * echo owned"] and s.monitor_schedule()["state"] == "off"


def test_a_paper_claim_resting_on_a_retracted_source_fails_the_traceability_gate(s):
    from openlab.study.paper import lint
    rec = s.record
    src = rec.last("source", doi="10.1000/a").id
    s.fetch_text(src)
    ev = s.extract(src, 1, "EMOS-loc Local EMOS 0.90", {"method": "EMOS-loc", "crps": 0.9}, "table", "Table 2")["id"]
    claim = s.claim("Local EMOS scores a CRPS of 0.90.", "observed", [ev])["id"]
    b = type("B", (), {"macro": type("M", (), {"name": "labclaim", "file": "a.tex", "line": 1,
                                                "text": f"\\labclaim{{{claim}}}"})(), "node": claim})()
    assert lint.traced(rec, [b]) == []
    s.notices["10.1000/a"] = [NOTICE]
    s.monitor()
    [f] = lint.traced(rec, [b])
    assert f.rule == "claim.untraceable" and claim in f.message and f.waivable
