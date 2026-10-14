"""`openlab replicate PAPER`: the paper alone in, a checked plan a person approves, a study built from it and a
README whose verdicts come from the record. The index and the PDF are canned: no network, no agent."""

import json

import pytest

from openlab import study
from openlab.sources import papers
from openlab.study import paperplan as R
from openlab.study import paperplan_cli
from openlab.study.errors import StudyError

PAPER = papers._paper(title="Neural networks for post-processing ensemble weather forecasts", arxiv="1805.09091",
                      doi="10.1175/MWR-D-18-0187.1", journal_doi="10.1175/mwr-d-18-0187.1", version="preprint",
                      date="2018-05-23", pdf="https://arxiv.org/pdf/1805.09091",
                      abstract="Ensemble weather predictions require post-processing.")
TABLE = ("Table 2: Mean CRPS for raw and post-processed ECMWF ensemble forecasts.\n"
         " Raw ensemble 1.16 1.16\n EMOS-gl Global EMOS 1.01 1.00\n"
         " NN-aux-emb . . . and station embeddings 0.82 0.78\n"
         "Across five seeds the CRPS of the network varies by about 0.01.\n")
PLAN = """
paper = "arxiv:1805.09091"
title = "Neural networks for post-processing ensemble weather forecasts"
summary = "A network with station embeddings post-processes ensemble forecasts better than EMOS."
mode = "static"
risks = ["the ECMWF data needs a licence"]
unresolved = []

[[methods]]
name = "Raw"
label = "Raw ensemble"
role = "baseline"
page = 2
description = "the raw ensemble"

[[methods]]
name = "EMOS_gl"
label = "EMOS-gl"
role = "baseline"
page = 2
description = "global EMOS, eq. 3"

[[methods]]
name = "NN_aux_emb"
label = "NN-aux-emb"
role = "proposed"
page = 2
description = "the network with auxiliary inputs and station embeddings, eq. 5"

[metric]
name = "crps"

[tolerance]
value = 0.02
max = 0.03
justification = "twice the seed-to-seed spread of about 0.01 the paper reports"
page = 2
quote = "Across five seeds the CRPS of the network varies by about 0.01."

[data]
name = "ECMWF ensemble forecasts, German stations"
source = "the paper's data release"
protocol = "train 2015, test 2016"

[[claims]]
id = "t2-raw"
method = "Raw"
value = 1.16
page = 2
section = "Table 2"
setting = "test 2016"
quote = "Raw ensemble 1.16 1.16"

[[claims]]
id = "t2-gl"
method = "EMOS_gl"
value = 1.01
page = 2
section = "Table 2"
setting = "test 2016"
quote = "EMOS-gl Global EMOS 1.01 1.00"

[[claims]]
id = "t2-nn"
method = "NN_aux_emb"
value = 0.82
page = 2
section = "Table 2"
setting = "test 2016"
quote = "NN-aux-emb . . . and station embeddings 0.82 0.78"

[budget]
usd = 3
minutes = 60
trials = 10
runs = 4
run_minutes = 5
estimate = "Table 4 times one fit at 2 min on 8 cores; 4 here, three methods"
"""


@pytest.fixture
def canned(monkeypatch):
    monkeypatch.setattr(papers, "ask", lambda src, q, n, since: [PAPER] if (src, q) == ("arxiv", "1805.09091") else [])
    monkeypatch.setattr(papers, "crossref_updates", lambda dois: {})
    monkeypatch.setattr(papers, "pdf", lambda url: b"%PDF " + url.encode())
    monkeypatch.setattr(papers, "pdf_pages", lambda data, layout=False: ["arXiv:1805.09091v1 [stat.ML]", TABLE])


@pytest.fixture
def started(tmp_path, canned, monkeypatch):
    monkeypatch.delenv("OPENLAB_AGENT", raising=False)
    out = R.start("1805.09091", tmp_path / "rl18")
    return tmp_path / "rl18", out


@pytest.fixture
def planned(started):
    folder, _ = started
    R.path(folder).write_text(PLAN, encoding="utf-8")
    return folder


def edit(folder, old, new):
    p = R.path(folder)
    p.write_text(p.read_text().replace(old, new, 1))


def test_start_fetches_the_paper_and_writes_a_skeleton(started):
    folder, out = started
    assert out["paper"]["key"] == "arxiv:1805.09091" and out["paper"]["n_pages"] == 2
    assert (folder / R.PDF).read_bytes().startswith(b"%PDF")
    assert "=== page 2 ===" in (folder / R.PAGES).read_text() and "Raw ensemble 1.16" in (folder / R.PAGES).read_text()
    raw = R.load(folder)
    assert raw["paper"] == "arxiv:1805.09091" and not (folder / "study.toml").exists()
    errs = " ".join(p["error"] for p in R.problems(raw, folder))
    assert "`title` is empty" in errs and "tolerance.justification" in errs and "metric.name" in errs
    with pytest.raises(StudyError, match="already holds a replication"):
        R.start("1805.09091", folder)


def test_a_clean_plan_lints_clean_and_renders(planned):
    r = R.lint(planned)
    assert r["ok"], r["problems"]
    md = (planned / R.MD).read_text()
    assert "fidelity (baseline rebuilt)" in md and "replication (the paper's claim)" in md and "0.02" in md


@pytest.mark.parametrize("old,new,error", [
    ('quote = "Raw ensemble 1.16 1.16"', 'quote = "Raw ensemble 1.17 1.16 here"', "is not on page 2"),
    ("value = 1.01", "value = 1.02", "1.02 is not written in the quote"),
    ('label = "EMOS-gl"', 'label = "EMOS-global"', "does not show the method's label"),
    ('name = "crps"', 'name = "crpss"', "is not a metric openlab scores"),
    ('justification = "twice the seed-to-seed spread of about 0.01 the paper reports"', 'justification = ""',
     "tolerance.justification"),
    ("max = 0.03", "max = 0.01", "must be at least tolerance.value"),
    ('role = "proposed"', 'role = "baseline"', "name no proposed method"),
    ('paper = "arxiv:1805.09091"', 'paper = "arxiv:1805.00001"', "must be arxiv:1805.09091"),
    ("page = 2\nquote = \"Across", "page = 1\nquote = \"Across", "is not on page 1"),
])
def test_lint_refuses_what_the_paper_does_not_say(planned, old, new, error):
    edit(planned, old, new)
    errs = [p["error"] for p in R.lint(planned)["problems"]]
    assert any(error in e for e in errs), errs


def test_approve_needs_a_person_and_builds_the_study(planned, person_key):
    with pytest.raises(StudyError, match="person"):
        R.approve(planned, "ok", "person:sam")
    assert not (planned / "study.toml").exists()                # nothing made by an unsigned attempt
    tmp = planned
    with person_key.act():
        out = R.approve(tmp, "read it", "person:sam")
    s = study.open(tmp)
    t = s.record.last("target")
    assert (t.payload["tolerance"], t.payload["max_tolerance"]) == (0.02, 0.03)
    assert s.spec.evaluate["baselines"] == ["model:Raw", "model:EMOS_gl"] and "Model = NN_aux_emb" in \
        (tmp / "model.py").read_text()
    ev = {k: s.record.get(v).payload for k, v in out["evidence"].items()}
    assert ev["t2-gl"]["fields"] == {"method": "EMOS-gl", "crps": 1.01, "setting": "test 2016"}
    act = R.approved(tmp)
    assert act.actor == "person" and act.payload["args"]["sha"] == R.sha(R.load(tmp)) and s.verify()["ok"]
    assert "approved as" in (tmp / R.MD).read_text()
    with person_key.act(), pytest.raises(StudyError, match="already approved"):
        R.approve(tmp, "again", "person:sam")
    edit(tmp, "usd = 3", "usd = 4")
    with pytest.raises(StudyError, match="changed since its approval"):
        R.build(tmp, dry_run=True)
    # a person approves the change: claims and roles may move, what was fixed before any run may not
    edit(tmp, 'name = "EMOS_gl"\nlabel = "EMOS-gl"\nrole = "baseline"', 'name = "EMOS_gl"\nlabel = "EMOS-gl"\nrole = "proposed"')
    assert any("`model` is not one of the proposed" in p["error"] for p in R.lint(tmp)["problems"])
    edit(tmp, 'mode = "static"', 'mode = "static"\nmodel = "NN_aux_emb"')
    with person_key.act():
        again = R.approve(tmp, "EMOS-gl is a proposed variant too; more budget", "person:sam")
    assert again["amends"] == act.id and again["quoted"] == [] and again["evidence"] == out["evidence"]
    assert R.approved(tmp).id == again["id"] and R.lint(tmp)["ok"]
    brief = R.build(tmp, dry_run=True)["brief"]
    assert "arm model:EMOS_gl: replication" in brief and brief.count("Model = NN_aux_emb") == 1
    edit(tmp, "value = 0.02", "value = 0.025")
    with person_key.act(), pytest.raises(StudyError, match="tolerance cannot change after the first approval"):
        R.approve(tmp, "wider", "person:sam")


@pytest.fixture
def built(planned, person_key):
    with person_key.act():
        R.approve(planned, "read it", "person:sam")
    return study.open(planned)


def test_the_builders_brief_carries_the_plan(built):
    r = R.build(built.folder, dry_run=True)
    ev = R.approved(built.folder).payload["args"]["evidence"]
    assert ev["t2-nn"] in r["brief"] and "arm model:EMOS_gl: fidelity" in r["brief"]
    assert "Model = NN_aux_emb" in r["brief"] and "never the tolerance" in r["brief"] and r["budget"]["usd"] == 3


def _result(s, nn, raw, gl, data="a" * 64, seconds=0.0):
    rec = s.record
    under = rec.last("amendment") or rec.last("plan")
    ds = rec.last("dataset", sha256=data) or rec.add("dataset", {"name": "main", "source": {"kind": "provider"},
                                                                 "sha256": data, "rows": 1, "columns": ["y"],
                                                                 "card": {}})
    run = rec.add("run", {"segment": "research", "key": f"k{len(rec.find('run'))}", "cached": False,
                          "spec_sha": under.payload["sha"], "code": {}, "data": {"main": data}, "env": {},
                          "sandbox": "stub", "seconds": seconds, "metric_versions": {"crps": 1},
                          "audit": {"verdict": "clean", "checked": 3, "mismatches": []}},
                  links=[("under", under.id), ("uses", ds.id)])
    cell = lambda v: {"crps": {"value": v, "lo": v - 0.02, "hi": v + 0.02}}       # noqa: E731
    return rec.add("result", {"metrics": cell(nn), "n": 100, "baselines": {"model:Raw": cell(raw),
                                                                         "model:EMOS_gl": cell(gl)}},
                   links=[("of", run.id)]).id


def test_status_shows_the_builds_pace_against_the_plan(built, capsys):
    """The plan said 4 runs of 5 min in 60; two runs took 40 min each, so the 2 left need 80: status says so."""
    assert R.status(built.folder)["pace"] is None
    for _ in range(2):
        _result(built, 1.0, 1.1, 1.05, seconds=2400.0)
    p = R.status(built.folder)["pace"]
    assert (p["done"], p["run_minutes"], p["need"], p["fits"]) == (2, 40.0, 80.0, False)
    assert paperplan_cli.main(["status", str(built.folder)]) == 0
    assert "the runs left need ~80 min at 1 at once, 60 min left -- they do not fit" in capsys.readouterr().out
    assert "shows the pace against the plan's estimate" in R.build(built.folder, dry_run=True)["brief"]


def test_the_report_writes_verdicts_from_the_record(built, capsys):
    s, folder = built, built.folder
    ev = R.approved(folder).payload["args"]["evidence"]
    readme = folder / "README.md"
    readme.write_text(readme.read_text().replace("The split chosen and why, and every deviation from the paper.",
                                                 "We trained on 2015 only, as the paper did."))
    r = _result(s, 0.83, 1.15, 1.005)
    g = [s.fidelity(ev["t2-raw"], r, "model:Raw")["id"], s.fidelity(ev["t2-gl"], r, "model:EMOS_gl")["id"]]
    s.replication(ev["t2-nn"], r, g)
    r2 = _result(s, 0.83, 1.15, 1.006)                                  # a second trial: one row per claim, its last
    g = [s.fidelity(ev["t2-raw"], r2, "model:Raw")["id"], s.fidelity(ev["t2-gl"], r2, "model:EMOS_gl")["id"]]
    s.replication(ev["t2-nn"], r2, g)
    out = R.report(folder)
    assert len(out["claims"]) == 3 and all(x["result"] == r2 for x in out["claims"])
    rows = {x["claim"]: x for x in out["claims"]}
    assert rows["t2-nn"]["verdict"] == "replicated" and rows["t2-gl"]["verdict"] == "faithful"
    assert out["rebuilt"] is True
    text = readme.read_text()
    assert "| t2-nn | NN_aux_emb | Table 2, p.2, test 2016 | 0.82 | 0.83" in text and "**faithful**" in text
    assert "We trained on 2015 only" in text and "## Run it again" in text
    assert json.loads((folder / "results" / "report.json").read_text())["claims"][0]["claim"] == "t2-nn"
    assert paperplan_cli.main(["status", str(folder)]) == 0
    assert "claim" in capsys.readouterr().out


def test_the_cli_routes_a_paper_and_the_verbs(tmp_path, canned, monkeypatch, capsys):
    monkeypatch.delenv("OPENLAB_AGENT", raising=False)
    assert paperplan_cli.main(["1805.09091", str(tmp_path / "p"), "--draft-with", "none"]) == 0
    assert "fill" in capsys.readouterr().out
    assert paperplan_cli.main(["lint", str(tmp_path / "p")]) == 1
    from openlab.__main__ import COMMANDS
    assert COMMANDS["replicate"][0] == "openlab.study.paperplan_cli:main"


def test_a_builder_after_a_lost_one_resumes_with_the_budget_left(built):
    from openlab.agents.store import Store
    root, rel = R.workspace(built.folder)
    store = Store(root)
    store.register("replicator-aaaaaa", "spawned", role="replicator", study=rel, status="running",
                   budget={"usd": 3.0, "minutes": 60.0}, spent={"usd": 1.25})
    assert R.live_builder(built.folder)["id"] == "replicator-aaaaaa"
    with pytest.raises(StudyError, match="the build is running"):
        R.build(built.folder)
    store.event("replicator-aaaaaa", "note", summary="trial 1 failed two fidelities; trial 2 running")
    store.set_status("replicator-aaaaaa", "lost")
    r = R.build(built.folder, dry_run=True)
    assert r["resumed"] == ["replicator-aaaaaa"] and r["budget"]["usd"] == 1.75
    assert r["brief"].startswith("This RESUMES the build") and "trial 2 running" in r["brief"]
    assert "machine stopped" in r["brief"] and "0 run(s); no fidelity yet" in r["brief"]


def test_the_plan_says_how_long_its_runs_take_and_they_must_fit_its_minutes(planned):
    """energyad: 20 NSL-KDD runs of 87 min each did not fit 480 min; lint now asks before the person approves."""
    edit(planned, "runs = 4", "runs = 40")
    edit(planned, "run_minutes = 5", "run_minutes = 87")
    probs = {p["where"]: p for p in R.lint(planned)["problems"]}
    assert "3480 min of compute" in probs["budget.minutes"]["error"] and "budget.trials" in probs
    edit(planned, "runs = 40", "runs = 40\nparallel = 10")
    edit(planned, "minutes = 60", "minutes = 480")
    edit(planned, "trials = 10", "trials = 45")
    assert R.lint(planned)["ok"], R.lint(planned)["problems"]
    edit(planned, 'estimate = "Table 4', 'estimate = ""\n# "Table 4')
    assert "budget.estimate" in {p["where"] for p in R.lint(planned)["problems"]}


def test_a_plan_may_define_its_own_metric_and_its_code_is_approved_with_it(planned, person_key):
    edit(planned, 'name = "crps"', 'name = "crps_own"\nbetter = "lower"\n'
         'description = "the CRPS of the ensemble, as the paper defines it (eq. 2)"')
    errs = [p["error"] for p in R.lint(planned)["problems"]]
    assert any("there is no metrics.py" in e for e in errs), errs
    code = "import numpy as np\n\n\ndef crps_own(y, pred):\n    return float(np.mean(np.abs(y - pred)))\n"
    (planned / "metrics.py").write_text(code)
    r = R.lint(planned)
    assert r["ok"], r["problems"]
    md = (planned / R.MD).read_text()
    assert "crps_own is the study's own metric (lower is better" in md and "def crps_own(y, pred):" in md
    with person_key.act():
        out = R.approve(planned, "read it, code included", "person:sam")
    s = study.open(planned)
    assert s.spec.raw["metrics"]["crps_own"]["better"] == "lower" and s.spec.evaluate["metrics"] == ["crps_own"]
    assert s.record.get(out["evidence"]["t2-gl"]).payload["fields"]["crps_own"] == 1.01
    (planned / "metrics.py").write_text(code.replace("np.abs", "np.square"))
    assert "changed since its approval" in R.lint(planned)["problems"][0]["error"]
    from openlab.study import spec
    spec.seal(planned, s.record, "agent")
    with pytest.raises(StudyError, match="changed since its approval"):
        R.require(planned)
    s.amend("an agent's edit of the metric is not the person's", person=False)
    assert "changed since its approval" in R.lint(planned)["problems"][0]["error"]
    (planned / "metrics.py").write_text(code.replace("np.abs", "np.fabs"))
    with person_key.act():
        study.open(planned).amend("same values, faster", person=True)
    assert R.lint(planned)["ok"] and R.require(planned)[2].id == out["id"]
