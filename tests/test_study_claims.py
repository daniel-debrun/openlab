"""Guarantees of claims and the research structures, enforced by the record's schema: a verdict or a level is
computed, never typed; a derived number is what the LaTeX engine makes of its inputs; a claim cannot rest on a
look-ahead run or, silently, on a retracted source; tests spend the alpha-investing budget by its arithmetic, with
the result's p; a broken assumption puts its claims at risk; a hypothesis must face the dead ends it repeats; a
contradiction search's state follows its coverage; only a person approves; the question tree is derived from its
leaves. Runs are written into the record directly and the indexes are canned: no network."""

import io
import json
import sys

import pytest

from openlab import study
from openlab.app.registry import ToolContext, dispatch
from openlab.sources import papers
from openlab.study import cli, spec
from openlab.study.errors import StudyError
from conftest import PASSPHRASE

TOML = """
[study]
name = "c"
question = "Does the model beat persistence?"
mode = "temporal"

[data.panel]
file = "data/panel.parquet"
recipe = "recipes/make_data.py"
time = "date"
key = ["site"]
known_at = "date"

[target]
horizon = "1d"

[split]
scheme = "walk_forward"
decisions = "daily"
research = [2024-01-01, 2024-12-31]
holdout = [2025-01-01, 2025-06-30]

[evaluate]
metrics = ["mae"]

[budget]
trials = 20
holdout_looks = 2

[inference]
alpha = 0.5
wealth = 0.25
payout = 0.5

[review]
question = "Which post-processing methods beat EMOS?"
sources = ["arxiv"]
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
"""

P = papers._paper
A = P(title="Neural networks for postprocessing ensemble weather forecasts", arxiv="1805.09091", version="preprint",
      date="2018-05-23", pdf="https://arxiv.org/pdf/1805.09091", abstract="Ensembles need post-processing.")
B = P(title="A retracted study of ensemble postprocessing at stations", arxiv="2309.04452", version="preprint",
      date="2023-09-08", pdf="https://arxiv.org/pdf/2309.04452", abstract="Retracted.", retracted=True)
C = P(title="When EMOS fails: a negative result for station postprocessing", arxiv="2401.00001", version="preprint",
      date="2024-01-02", pdf="https://arxiv.org/pdf/2401.00001", abstract="EMOS fails here.")
TEXT = {A["pdf"]: ["Table 2: Mean CRPS.\n EMOS-loc Local EMOS 0.90\n EMOS-gl Global EMOS 1.01\n"],
        B["pdf"]: ["Table 4. Mean CRPS in K.\n EMOS 0.82\n DRN 0.67\n"],
        C["pdf"]: ["Table 1. Local EMOS is worse than the raw ensemble: CRPS 1.20 against 1.16.\n"]}
SHA = "a" * 64


@pytest.fixture
def s(tmp_path, monkeypatch):
    folder = tmp_path / "s"
    folder.mkdir()
    (folder / "study.toml").write_text(TOML, encoding="utf-8")
    monkeypatch.setattr(papers, "ask", lambda src, q, n, since: [A, B] if q == "ensemble postprocessing" else [C])
    monkeypatch.setattr(papers, "crossref_updates", lambda dois: {})
    monkeypatch.setattr(papers, "pdf", lambda url: b"%PDF " + url.encode())
    monkeypatch.setattr(papers, "pdf_pages", lambda data, layout=False: TEXT[data[5:].decode()])
    st = study.open(folder)
    st.discover()                                           # seals study.toml
    return st


def source(s, arxiv):
    return s.record.last("source", arxiv=arxiv).id


def evidence(s, arxiv, quote, fields):
    src = source(s, arxiv)
    s.fetch_text(src)
    return s.extract(src, 1, quote, fields, "table", "Table")["id"]


def result(s, *, audit="clean", segment="research", p=0.01, better="model", mae=0.5):
    """A run and its result written straight into the record, as the runner would (code = the study's own)."""
    from openlab.study.fingerprint import code
    rec = s.record
    if not (s.folder / "model.py").exists():
        (s.folder / "model.py").write_text("class Model: ...\n")
    plan = rec.last("plan")
    ds = rec.last("dataset") or rec.add("dataset", {"name": "panel", "source": {"kind": "provider"}, "sha256": SHA,
                                                    "rows": 1, "columns": ["y"], "card": {}})
    after = []
    if segment == "holdout":
        from openlab.agents import person
        from openlab.study.personkey import signing
        with signing(person.unlock(PASSPHRASE) if person.exists() else None):
            look = rec.add("look", {"n": len(rec.find("look")) + 1, "reason": "final",
                                    "spec_sha": plan.payload["sha"], "code": code(s.folder), "data": {"panel": SHA}},
                           actor="person" if person.exists() else "agent", links=[("under", plan.id)])
        after = [("after", look.id)]
    bad = [{"t": "2024-03-01"}] if audit == "look-ahead" else []
    run = rec.add("run", {"segment": segment, "key": f"k{len(rec.find('run'))}", "cached": False,
                          "spec_sha": plan.payload["sha"], "code": code(s.folder), "data": {"panel": SHA},
                          "env": {"python": "3.11"},
                          "sandbox": "stub", "seconds": 0.0,
                          "audit": {"verdict": audit, "checked": 0 if audit == "unchecked" else 3, "mismatches": bad}},
                  links=[("under", plan.id), ("uses", ds.id)] + after)
    return rec.add("result", {"metrics": {"mae": {"value": mae, "lo": mae - 0.1, "hi": mae + 0.1}},
                              "paired": {"persistence": {"metric": "mae", "diff": -0.1, "lo": -0.2, "hi": 0.0, "p": p,
                                                         "better": better}}}, links=[("of", run.id)]).id


def observed(s, r, **kw):
    return s.claim("The model's MAE is 0.5.", "observed", [r], **kw)


def test_an_agent_cannot_type_a_verdict(s):
    r = result(s)
    c = observed(s, r)
    assert (c["payload"]["verdict"], c["payload"]["level"]) == ("supported", "recorded")
    links = [("rests_on", c["id"])]
    body = {"statement": "So the model learns the station climate.", "state": "inferred"}
    with pytest.raises(StudyError, match="computed from what it rests on, never typed") as e:
        s.record.add("claim", {**body, "verdict": "supported"}, links=links)
    assert "typed 'supported', computed 'untested'" in e.value.meaning
    with pytest.raises(StudyError, match="never typed"):
        s.record.add("claim", {**body, "verdict": "untested", "level": "measured"}, links=links)
    assert s.record.add("claim", {**body, "verdict": "untested", "level": "recorded",
                               "selected_from": 1}, links=links)
    with pytest.raises(StudyError, match="inferred claims carry no number"):
        s.claim("It is 0.5.", "inferred", [c["id"]], value=0.5)
    with pytest.raises(StudyError, match="observed claims cannot link `tests`"):
        s.claim("x", "observed", [r], tests=_hypothesis(s, s.question("q?")["id"]))
    with pytest.raises(StudyError, match="hypothesized claims rest on one `tests` link"):
        s.claim("x", "hypothesized")
    unknown = s.claim("No study tests EMOS outside Europe.", "unknown", searched=s.record.find("coverage")[0].id)
    assert (unknown["payload"]["verdict"], unknown["payload"]["level"]) == ("untested", "")


def test_a_paper_renders_exactly_what_it_binds_at_the_current_head(s, person_key):
    c = observed(s, result(s))["id"]
    rec = s.record
    prose = {"paper/paper.toml": 'title = "t"\nauthors = ["a"]\nabstract = "a.tex"\nsections = ["results"]\n',
             "paper/a.tex": f"\\labclaim{{{c}}}"}
    body = {"tex_sha": SHA, "pdf_sha": "", "bindings": [c], "lint": [],
            "sources": rec.put_blob(json.dumps(prose).encode())}
    with pytest.raises(StudyError, match="renders exactly the nodes it binds"):
        rec.add("paper", {**body, "head": rec.head()}, links=[("renders", c), ("renders", rec.last("result").id)])
    stale = rec.head()
    rec.add("question", {"text": "Anything new?"})
    with pytest.raises(StudyError, match="names the record head it was built at"):
        rec.add("paper", {**body, "head": stale}, links=[("renders", c)])
    paper = rec.add("paper", {**body, "head": rec.head()}, links=[("renders", c)])
    assert paper.payload["signed"] == ""
    with pytest.raises(StudyError, match="can only be written by person"):
        rec.add("approval", {"decision": "approve", "reason": "reads well"}, links=[("on", paper.id)])
    with person_key.act():
        assert rec.add("approval", {"decision": "approve", "reason": "reads well"}, actor="person",
                       links=[("on", paper.id)])


def test_a_derived_claim_is_checked_by_the_latex_engine(s):
    r = result(s, mae=0.5)
    ev = evidence(s, "1805.09091", "EMOS-loc Local EMOS 0.90", {"method": "EMOS-loc", "crps": 0.9})
    inputs = {"m": f"{r}#metrics.mae.value", "e": f"{ev}#fields.crps"}
    frac = r"\frac{{m} - {e}}{{e}}"
    with pytest.raises(StudyError, match="not what the expression gives") as e:
        s.claim("The model is 40% below EMOS-loc.", "derived", [r, ev], value=-0.4, expression=frac, inputs=inputs)
    assert "-0.444" in e.value.meaning
    c = s.claim("The model is 44% below EMOS-loc.", "derived", [r, ev], value=-0.4 / 0.9, expression=frac,
                inputs=inputs)["payload"]
    assert (c["verdict"], c["level"]) == ("supported", "recorded")
    with pytest.raises(StudyError, match="no input reads"):
        s.claim("x", "derived", [r, ev], value=0.5, expression="{m}", inputs={"m": inputs["m"]})
    with pytest.raises(StudyError, match="does not rest on"):
        s.claim("x", "derived", [r], value=0.9, expression="{e}", inputs={"e": inputs["e"]})
    with pytest.raises(StudyError, match="does not use e"):
        s.claim("x", "derived", [r, ev], value=0.5, expression="{m}", inputs=inputs)
    with pytest.raises(StudyError, match="cannot evaluate the expression to a number"):
        s.claim("x", "derived", [r], value=0.5, expression="{m} + y", inputs={"m": inputs["m"]})
    with pytest.raises(StudyError, match="has no `metrics.crps.value`"):
        s.claim("x", "derived", [r], value=0.5, expression="{m}", inputs={"m": f"{r}#metrics.crps.value"})
    assert s.record.last("action").payload["verb"] == "claim" and not s.record.last("action").payload["outcome"]["ok"]


def test_an_observed_claim_on_a_look_ahead_run_is_refused(s, person_key):
    for audit in ("look-ahead", "unchecked"):
        with pytest.raises(StudyError, match=f"truncation audit is {audit}"):
            observed(s, result(s, audit=audit))
    h = result(s, segment="holdout", mae=0.5)
    c = observed(s, h, value=0.5, inputs={"v": f"{h}#metrics.mae.value"})["payload"]
    assert (c["level"], c["value"]) == ("measured", 0.5)
    with pytest.raises(StudyError, match="an observed number is its one input's"):
        observed(s, h, value=0.6, inputs={"v": f"{h}#metrics.mae.value"})
    with pytest.raises(StudyError, match="an observed number is its one input's"):
        observed(s, h, value=0.5)
    both = s.claim("The model's MAE is 0.5 on research and holdout.", "observed", [h, result(s)])["payload"]
    assert both["level"] == "recorded"                                  # the weakest ground


def test_a_claim_on_a_retracted_source_says_so_or_a_person_waives_it(s, tmp_path, person_key):
    ev = evidence(s, "2309.04452", "Mean CRPS in K. EMOS 0.82", {"method": "EMOS", "crps": 0.82})
    with pytest.raises(StudyError, match="which is retracted") as e:
        s.claim("EMOS scores a CRPS of 0.82.", "observed", [ev])
    assert "OpenAlex is_retracted" in e.value.meaning
    assert s.claim("A retracted paper reports a CRPS of 0.82 for EMOS.", "observed", [ev])["payload"]["verdict"] == \
        "supported"
    with pytest.raises(StudyError, match="can only be written by person"):
        s.approve(ev, "waive", "the retraction concerns Table 5", person=False)
    with pytest.raises(StudyError, match="only waives its source's retraction"), person_key.act():
        s.approve(ev, "approve", "fine", person=True)
    with person_key.act():
        s.approve(ev, "waive", "the retraction concerns Table 5 only", person=True)
    assert s.claim("EMOS scores a CRPS of 0.82.", "observed", [ev])["payload"]["verdict"] == "supported"


def test_an_agent_cannot_approve(s, tmp_path, monkeypatch, capsys, person_home):
    c = observed(s, result(s))["id"]
    with pytest.raises(StudyError, match="can only be written by person"):
        s.approve(c, "approve", "looks right")
    act = s.record.last("action")
    assert (act.actor, act.payload["verb"], act.payload["outcome"]["ok"]) == ("agent", "approve", False)
    args = {"folder": "s", "on": c, "decision": "approve", "reason": "checked the run"}
    env = dispatch(ToolContext(str(tmp_path)), "study_approve", args)
    assert not env["ok"] and env["error"] == "a reviewer's act"
    monkeypatch.setattr(sys, "stdin", io.StringIO("c\n"))
    assert cli.main(["approve", str(s.folder), c, "--decision", "approve", "--reason", "x"]) == 1
    assert "in their own terminal" in capsys.readouterr().err
    assert s.record.find("approval") == []
    env = dispatch(ToolContext(str(tmp_path)), "study_approve", args, human=True)
    assert not env["ok"] and "signed with the person key" in env["meaning"] + env["error"]   # reviewer secret alone
    from openlab.agents import person
    from openlab.study.personkey import signing
    person.create("sam", PASSPHRASE)
    with signing(person.unlock(PASSPHRASE)):
        env = dispatch(ToolContext(str(tmp_path)), "study_approve", args, human=True)
    assert env["ok"] and env["result"]["decision"] == "approve"
    assert dispatch(ToolContext(str(tmp_path)), "study_claims", {"folder": "s"})["result"]["claims"][0]["approval"] \
        == "approve"


def _hypothesis(s, q, tags=("emos",), **kw):
    return s.hypothesis("The model beats persistence.", "lower MAE than persistence", "a paired test that does not "
                        "reject", q, list(tags), **kw)["id"]


def test_the_alpha_investing_arithmetic_is_enforced(s):
    h = _hypothesis(s, s.question("Does the model beat persistence?")["id"])
    r1, r2, r3 = result(s, p=0.01), result(s, p=0.3), result(s, p=0.2)
    t1 = s.test(h, r1, "persistence", 0.1)["payload"]                   # rejected: earns the payout
    assert (t1["wealth_before"], t1["rejected"], t1["wealth_after"]) == (0.25, True, 0.75)
    t2 = s.test(h, r2, "persistence", 0.2)["payload"]                   # costs 0.2 / 0.8
    assert (t2["wealth_before"], t2["rejected"], t2["wealth_after"]) == (0.75, False, 0.5)
    with pytest.raises(StudyError, match="more than the alpha budget left pays for") as e:
        s.test(h, r3, "persistence", 0.4)
    assert "at most alpha_j 0.333333" in e.value.meaning
    with pytest.raises(StudyError, match="already tested on"):
        s.test(h, r2, "persistence", 0.3)
    links = [("of", h), ("from", r3)]
    body = {"family": "primary", "baseline": "persistence", "alpha_j": 0.25, "p": 0.2, "rejected": True,
            "wealth_before": 0.5, "wealth_after": 1.0}
    with pytest.raises(StudyError, match="a test's p is its result's"):
        s.record.add("test", {**body, "p": 0.001}, links=links)
    for typed in ({"wealth_before": 0.75}, {"rejected": False}, {"wealth_after": 1.5}):
        with pytest.raises(StudyError, match="does not follow the alpha-investing ledger"):
            s.record.add("test", {**body, **typed}, links=links)
    with pytest.raises(StudyError, match="no paired p against `climatology`"):
        s.test(h, r3, "climatology", 0.1)
    with pytest.raises(StudyError, match="truncation audit is look-ahead"):
        s.test(h, result(s, audit="look-ahead"), "persistence", 0.1)
    assert s.record.add("test", body, links=links).payload["wealth_after"] == 1.0


def test_a_spent_budget_refuses_the_next_test(s):
    h = _hypothesis(s, s.question("q?")["id"])
    assert s.test(h, result(s, p=0.3), "persistence", 0.2)["payload"]["wealth_after"] == 0.0   # 0.25 - 0.2 / 0.8
    with pytest.raises(StudyError, match="the error budget is spent"):
        s.test(h, result(s, p=0.001), "persistence", 0.01)
    assert len(s.record.find("test")) == 1


def test_inference_is_a_persons_budget(tmp_path):
    (tmp_path / "study.toml").write_text(TOML.replace("wealth = 0.25", "wealth = 0.75"), encoding="utf-8")
    with pytest.raises(StudyError, match="`inference.wealth` is more than inference.alpha"):
        spec.load(tmp_path)
    (tmp_path / "study.toml").write_text(TOML.split("[inference]")[0] + TOML.split("payout = 0.5")[1], encoding="utf-8")
    assert spec.load(tmp_path).inference == {"alpha": 0.05, "wealth": 0.05 * 0.95, "payout": 0.05}
    assert "inference" in spec.PERSON_FIELDS


def test_breaking_an_assumption_puts_dependent_claims_at_risk(s):
    r = result(s)
    a = s.assume("Station errors are independent.", "the stations are far apart", "a spatial correlation test")["id"]
    c1 = observed(s, r, needs=[a])["id"]
    c1v = observed(s, r, value=0.5, inputs={"v": f"{r}#metrics.mae.value"}, needs=[a])["id"]
    c2 = s.claim("Twice the MAE is 1.0.", "derived", [c1v], value=1.0, expression=r"2 \cdot {x}",
                 inputs={"x": f"{c1v}#value"})["id"]
    c3 = s.claim("So the errors are small.", "inferred", [c2])["id"]
    free = observed(s, r)["id"]
    assert all(not row["at_risk"] for row in s.claims())
    b = s.break_(a, "neighbouring stations' errors correlate at 0.6", [r])
    assert b["kind"] == "break"
    risk = {row["id"]: row["at_risk"] for row in s.claims()}
    assert risk == {c1: [a], c1v: [a], c2: [a], c3: [a], free: []}


def test_a_hypothesis_repeating_a_dead_end_is_refused(s):
    q = s.question("Does EMOS help?")["id"]
    h1 = _hypothesis(s, q, tags=("emos", "lead time"))
    r = result(s, p=0.4)
    with pytest.raises(StudyError, match="carries its tags"):
        s.dead_end(h1, "EMOS at 48 h did not beat persistence", "too few stations", ["emos"], [r])
    d = s.dead_end(h1, "EMOS at 48 h did not beat persistence", "too few stations", ["emos", "lead time"], [r])["id"]
    with pytest.raises(StudyError, match=f"dead end {d} shares emos") as e:
        _hypothesis(s, q, tags=("EMOS ",))
    assert "too few stations" in e.value.meaning
    with pytest.raises(StudyError, match="says how"):
        _hypothesis(s, q, tags=("emos",), differs_from=[d])
    assert _hypothesis(s, q, tags=("emos",), differs_from=[d], difference="ten times the stations")
    assert _hypothesis(s, q, tags=("crps",))


def test_a_dead_end_may_be_of_a_question_and_shown_by_a_claim(s):
    q = s.question("Can the station feed serve hourly forecasts?")["id"]
    c = observed(s, result(s))["id"]
    d = s.dead_end(q, "the hourly station feed", "it lags by six hours", ["feed"], [c])
    links = s.record.links(d["id"])
    assert ("of", q) in [(rel, dst) for _, rel, dst in links] and ("shown_by", c) in [(r, x) for _, r, x in links]
    from openlab.study import claims
    assert claims.tree(s)["questions"][-1]["status"] == "untested"       # the tree reads it no differently
    with pytest.raises(StudyError, match="needs tags"):
        _hypothesis(s, q, tags=())


def test_a_contradiction_searchs_state_follows_its_coverage(s, monkeypatch):
    ev = evidence(s, "1805.09091", "EMOS-loc Local EMOS 0.90", {"method": "EMOS-loc", "crps": 0.9})
    c = s.claim("Local EMOS scores a CRPS of 0.90.", "observed", [ev])["id"]
    first = s.contra(c, ["EMOS fails at stations"])
    assert first["state"] == "none_found" and first["search"]["sources"] == [source(s, "2401.00001")]
    discovery = s.record.find("coverage")[0]
    assert s.prisma()["coverage"] == discovery.id                       # a contradiction search is not the discovery
    against = evidence(s, "2401.00001", "Local EMOS is worse than the raw ensemble: CRPS 1.20 against 1.16",
                       {"method": "Local EMOS", "crps": 1.2})
    found = s.contra(c, found=[against])
    assert found["state"] == "found" and found["coverage"] == first["coverage"]
    with pytest.raises(StudyError, match="does not contradict it"):
        s.contra(c, found=[ev])
    rec = s.record
    with pytest.raises(StudyError, match="the search's state is none_found"):
        rec.add("contra_search", {"state": "found"}, links=[("for", c), ("searched", first["coverage"])])
    with pytest.raises(StudyError, match="is not a search for"):
        rec.add("contra_search", {"state": "none_found"}, links=[("for", c), ("searched", discovery.id)])

    def down(src, q, n, since):
        raise OSError("arXiv is down")
    monkeypatch.setattr(papers, "ask", down)
    assert s.contra(c, ["EMOS fails again"])["state"] == "insufficient"  # a failed query searched nothing
    assert [row["contra"] for row in s.claims()] == ["insufficient"]


def test_the_question_tree_is_derived_from_its_leaves(s):
    root = s.question("Is the model useful?")["id"]
    q1 = s.question("Does it beat the baselines?", part_of=root)["id"]
    s.question("Does it beat NWP?", part_of=root, out_of_scope="no NWP archive")
    q3 = s.question("Is it calibrated?", part_of=root)["id"]
    h1, h2 = _hypothesis(s, q1, tags=("persistence",)), _hypothesis(s, q1, tags=("climatology",))
    status = lambda t: {n["id"]: n["status"] for n in _flat(t["questions"])}   # noqa: E731
    assert set(status(s.tree()).values()) == {"untested", "out_of_scope"}
    s.test(h1, result(s, p=0.01), "persistence", 0.1)                   # rejected, the model better
    st = status(s.tree())
    assert (st[h1], st[h2], st[q1], st[root]) == ("answered", "untested", "untested", "untested")
    s.test(h2, result(s, p=0.3), "persistence", 0.2)                    # not rejected
    s.claim("The model is calibrated.", "observed", [result(s)], answers=[q3])
    st = status(s.tree())
    assert (st[h2], st[q1], st[q3], st[root]) == ("inconclusive", "inconclusive", "answered", "inconclusive")
    hyp = s.claim("The model beats climatology.", "hypothesized", tests=h2)["payload"]
    assert hyp["verdict"] == "inconclusive"
    tree = dispatch(ToolContext(str(s.folder.parent)), "study_tree", {"folder": "s"})["result"]
    assert tree["questions"][0]["id"] == root and len(tree["questions"][0]["children"]) == 3


def _flat(nodes):
    for n in nodes:
        yield n
        yield from _flat(n.get("children", []))


def test_a_claims_verdict_now_counts_the_tests_written_since(s):
    r = result(s, p=0.01, better="baseline")
    c = observed(s, r, value=0.5, inputs={"v": f"{r}#metrics.mae.value"})["id"]
    s.test(c, r, "persistence", 0.1)                                    # rejected, the baseline better
    row = s.claims()[0]
    assert (row["verdict"], row["verdict_written"]) == ("refuted", "supported")
    d = s.claim("Half the MAE is 0.25.", "derived", [c], value=0.25, expression=r"\frac{{v}}{2}",
                inputs={"v": f"{c}#value"})["payload"]
    assert d["verdict"] == "refuted"                                    # no better than its premise
