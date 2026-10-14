"""Guarantees of the graph views and the research frontier: the frontier lists untested leaves, claims at risk and
empty lattice cells, scored by the documented formula (information per hour, budget-adjusted) and ranked by it;
taking an item writes an action node linked to its targets; a claim's trace follows its chain to the source or the
run, and traceability names the claims whose chain is broken (a hypothesis with no test, a source retracted since).
Runs are written into the record directly and the indexes are canned: no network."""

import pytest

from openlab import study
from openlab.app.registry import ToolContext, dispatch
from openlab.sources import papers
from openlab.study import cli, frontier
from openlab.study.errors import StudyError
from openlab.study.schema import Retraction

TOML = """
[study]
name = "f"
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

[evaluate]
metrics = ["mae"]

[budget]
trials = 20

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
coordinates = ["dataset"]

[review.extract]
method = "text"
dataset = "text"
crps = "number"
"""

P = papers._paper
A = P(title="Neural networks for postprocessing ensemble weather forecasts", arxiv="1805.09091", doi="10.1000/a",
      version="preprint", date="2018-05-23", pdf="https://arxiv.org/pdf/1805.09091", abstract="Ensembles need it.")
C = P(title="Gridded postprocessing of ensemble weather forecasts", arxiv="2401.00001", doi="10.1000/c",
      version="preprint", date="2024-01-02", pdf="https://arxiv.org/pdf/2401.00001", abstract="Gridded EMOS.")
TEXT = {A["pdf"]: ["Table 2: Mean CRPS on the station set.\n EMOS 0.90\n DRN 0.82\n"],
        C["pdf"]: ["Table 1: Mean CRPS on the gridded set.\n EMOS 1.20\n"]}
SHA = "a" * 64


@pytest.fixture
def s(tmp_path, monkeypatch):
    folder = tmp_path / "s"
    folder.mkdir()
    (folder / "study.toml").write_text(TOML, encoding="utf-8")
    monkeypatch.setattr(papers, "ask", lambda src, q, n, since: [A, C])
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


def lattice(s) -> list:
    """Evidence for EMOS and DRN on the station set and EMOS on the gridded set: DRN on the gridded set is a gap."""
    station = "Mean CRPS on the station set. EMOS 0.90 DRN 0.82"
    return [evidence(s, "1805.09091", station, {"method": "EMOS", "dataset": "station set", "crps": 0.9}),
            evidence(s, "1805.09091", station, {"method": "DRN", "dataset": "station set", "crps": 0.82}),
            evidence(s, "2401.00001", "Mean CRPS on the gridded set. EMOS 1.20",
                     {"method": "EMOS", "dataset": "gridded set", "crps": 1.2})]


def result(s, p=0.01, diff=-0.1, lo=-0.2, hi=0.0, seconds=120.0):
    """A run (with its code and environment) and its result written straight into the record."""
    rec = s.record
    plan = rec.last("plan")
    ds = rec.last("dataset") or rec.add("dataset", {"name": "panel", "source": {"kind": "provider"}, "sha256": SHA,
                                                    "rows": 1, "columns": ["y"], "card": {}})
    run = rec.add("run", {"segment": "research", "key": f"k{len(rec.find('run'))}", "cached": False,
                          "spec_sha": plan.payload["sha"], "code": {"model.py": SHA}, "data": {"panel": SHA},
                          "env": {"python": "3.11"}, "sandbox": "stub", "seconds": seconds,
                          "audit": {"verdict": "clean", "checked": 3, "mismatches": []}},
                  links=[("under", plan.id), ("uses", ds.id)])
    return rec.add("result", {"metrics": {"mae": {"value": 0.5, "lo": 0.4, "hi": 0.6}},
                              "paired": {"persistence": {"metric": "mae", "diff": diff, "lo": lo, "hi": hi, "p": p,
                                                         "better": "model"}}}, links=[("of", run.id)]).id


def hypothesis(s, q, tag="emos"):
    return s.hypothesis("The model beats persistence.", "lower MAE than persistence", "a paired test that does not "
                        "reject", q, [tag])["id"]


def test_the_frontier_lists_untested_leaves_risks_and_gaps_ranked_by_information_per_hour(s):
    q = s.question("Does the model beat persistence?")["id"]
    h = hypothesis(s, q)
    r = result(s)
    a = s.assume("Station errors are independent.", "the stations are far apart", "a spatial correlation test")["id"]
    c = s.claim("The model's MAE is 0.5.", "observed", [r], needs=[a])["id"]
    s.break_(a, "neighbouring stations' errors correlate", [r])
    station, _, gridded = lattice(s)
    out = s.frontier(n=None)
    items = out["items"]
    by = {(i["kind"], i["target"]): i for i in items}
    assert (by[("untested", h)]["verb"], by[("untested", h)]["info"], by[("untested", h)]["cost"]) == ("test", 1, 120)
    assert (by[("at_risk", c)]["verb"], by[("at_risk", c)]["info"], by[("at_risk", c)]["cost"]) == ("run", 1, 120)
    assert a in by[("at_risk", c)]["why"] and out["costs"]["run"] == 120
    [gap] = [i for i in items if i["kind"] == "gap"]
    assert gap["target"] == gridded and "method DRN" in gap["why"] and "gridded set" in gap["why"]
    assert (gap["verb"], gap["cost"]) == ("extract", frontier.MIN_COST)     # the search took under a minute
    for i in items:
        assert i["score"] == round(i["info"] * 3600 / i["cost"], 4) and i["taken"] is None
    ranked = [i["kind"] for i in items if i["kind"] not in ("expand", "fuse", "draft", "select")]    # the search's
    assert ranked == ["gap", "untested", "at_risk"]     # score 60, 30, 30: ties in table order  (test_study_search)
    assert dispatch(ToolContext(str(s.folder.parent)), "study_frontier", {"folder": "s", "n": 1})["result"][
        "items"] == items[:1]


def test_an_inconclusive_test_and_the_spent_budget_scale_the_information(s):
    q = s.question("Does the model beat persistence?")["id"]
    h1, h2 = hypothesis(s, q, "emos"), hypothesis(s, q, "drn")
    r = result(s, p=0.3, diff=-0.1, lo=-0.25, hi=0.05)
    s.test(h1, r, "persistence", 0.1)                       # not rejected: costs 0.1 / 0.9 of the wealth
    out = s.frontier(n=None)
    left = out["budget"]["wealth"] / out["budget"]["initial"]
    assert out["budget"]["wealth"] == pytest.approx(0.25 - 0.1 / 0.9)
    by = {(i["kind"], i["target"]): i for i in out["items"]}
    assert by[("inconclusive", h1)]["info"] == round(2 * abs(-0.1) / (0.05 - -0.25) * left, 4)
    assert by[("untested", h2)]["info"] == round(left, 4)


def test_taking_an_item_records_an_action_linked_to_its_target(s, capsys):
    q = s.question("Is the model calibrated?")["id"]
    item = s.frontier()["items"][0]
    assert (item["kind"], item["target"], item["verb"], item["taken"]) == ("untested", q, "hypothesis", None)
    out = s.take(item["id"])
    act = s.record.get(out["action"])
    assert (act.payload["verb"], act.payload["args"]["kind"], act.payload["args"]["score"]) == \
        ("take", "untested", item["score"])
    assert [d for _, _, d in s.record.links(act.id, rel="did")] == [q]
    assert s.frontier()["items"][0]["taken"] == act.id
    with pytest.raises(StudyError, match="is not on the frontier"):
        s.take("frontier:000000000000")
    assert not s.record.last("action").payload["outcome"]["ok"]
    env = dispatch(ToolContext(str(s.folder.parent)), "study_take", {"folder": "s", "item": item["id"]})
    assert env["ok"] and env["result"]["target"] == q and env["result"]["action"].startswith("action:")
    assert cli.main(["frontier", str(s.folder)]) == 0 and item["id"] in capsys.readouterr().out


def test_traceability_names_the_claims_whose_chain_is_broken(s):
    r = result(s)
    on_run = s.claim("The model's MAE is 0.5.", "observed", [r])["id"]
    ev = lattice(s)[0]
    on_ev = s.claim("EMOS scores a CRPS of 0.90 on the station set.", "observed", [ev])["id"]
    h = hypothesis(s, s.question("Does the model beat persistence?")["id"])
    hyp = s.claim("The model beats persistence.", "hypothesized", tests=h)["id"]

    def path(node):
        while node.get("links"):
            link = node["links"][0]
            yield link["rel"], link["node"]["kind"]
            node = link["node"]
    t = s.trace(on_ev)
    assert t["complete"] and list(path(t["chain"])) == [("rests_on", "evidence"), ("from", "document"),
                                                        ("of", "source")]
    run = s.trace(on_run)["chain"]["links"][0]["node"]["links"][0]["node"]
    assert run["kind"] == "run" and run["code"] == {"model.py": SHA[:12]} and run["env"] == {"python": "3.11"}
    assert [(x["rel"], x["node"]["kind"]) for x in run["links"]] == [("uses", "dataset"), ("under", "plan")]
    tr = s.traceability()
    assert (tr["claims"], tr["traceable"], [b["claim"] for b in tr["broken"]]) == (3, 2, [hyp])
    assert tr["broken"][0]["missing"] == [f"{h} has no test"]
    src = source(s, "1805.09091")
    s.record.add(Retraction(via="crossref", type="retraction", notice="10.1000/notice", valid_from="2026-09-01"),
                 actor="openlab", links=[("of", src)])
    tr = dispatch(ToolContext(str(s.folder.parent)), "study_traceability", {"folder": "s"})["result"]
    assert sorted(b["claim"] for b in tr["broken"]) == sorted([on_ev, hyp]) and tr["share"] == round(1 / 3, 4)
    assert "which is retracted" in s.trace(on_ev)["missing"][0]
    g = dispatch(ToolContext(str(s.folder.parent)), "study_graph", {"folder": "s", "id": ev})["result"]
    assert {"src": ev, "rel": "of", "dst": src} in g["edges"] and on_ev in {n["id"] for n in g["nodes"]}
    assert not any(n["kind"] == "action" for n in g["nodes"])


def test_the_frontier_counts_trials_as_the_runner_does(s):
    """Declared variants and distinct run keys count; a second run of one key does not (openlab.study.ledger.shared)."""
    result(s)
    s.record.add("trial", {"n": 3, "note": "a sweep run outside openlab"})
    run = s.record.find("run")[0]
    s.record.add("run", {**run.payload, "cached": False},
                 links=[("under", s.record.last("plan").id), ("uses", s.record.last("dataset").id)])
    assert frontier.budget(s)["trials"]["used"] == 4
