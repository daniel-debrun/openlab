"""Research from a prompt: a proposal is refused until every hypothesis has a falsifier and every branch a budget, a
stop rule and a kill criterion within the total and the workspace cap; a person's approval writes the proposal node,
study.toml, the question tree and the seal; an edit after approval is refused until amended (protected fields by a
person); execute spawns the lead with the brief and the total budget; status reads per branch what was tested,
killed or left untouched, by which agent, at what spend."""

import json

import pytest

from agents_fake import final, install
from openlab import study
from openlab.agents import person
from openlab.study import personkey
from openlab.agents.identity import acting_as
from openlab.agents.store import Store
from openlab.app.registry import ToolContext, dispatch
from openlab.study import proposal as P, propose_cli
from openlab.study.errors import StudyError
from openlab.study.schema import Proposal

PROMPT = ("Does tenure predict churn better than the majority baseline? We have 2,000 customers with five features. "
          "Tenure likely carries most of the signal.")

FULL = """
title = "Tenure and churn"
question = "Does tenure predict churn better than the majority baseline?"
background = "2,000 customers, five features."
mode = "static"
metrics = ["auc", "brier"]
baselines = ["majority"]
risks = ["the label may leak through a billing feature"]

[budget]
usd = 4.0
minutes = 90
trials = 12
holdout_looks = 1

[[hypotheses]]
id = "h1"
statement = "A model on tenure alone beats majority on AUC."
prediction = "a paired p below 0.05 against majority"
falsifier = "a paired test against majority with p >= 0.05"
tags = ["tenure"]

[[hypotheses]]
id = "h2"
statement = "The other four features add no AUC over tenure alone."
prediction = "AUC gain under 0.01"
falsifier = "the five-feature model beats tenure-only by 0.01 AUC or more"
tags = ["features"]

[[hypotheses]]
id = "h3"
statement = "Prior work ranks tenure as the strongest churn predictor."
prediction = "most included studies rank tenure first"
falsifier = "fewer than half of the included studies rank tenure first"
tags = ["literature"]

[[branches]]
id = "b1"
question = "Does tenure alone beat majority?"
serves = ["h1"]
role = "experimenter"
method = "logistic regression on tenure"
data_needs = "the churn table"
stop = "3 variants in a row without an AUC gain of 0.005"
kill = "no variant beats majority at p < 0.05"
budget = {usd = 1.5, minutes = 45, trials = 6}

[[branches]]
id = "b2"
question = "Do the other features add signal?"
serves = ["h2"]
role = "experimenter"
method = "gradient boosting on all five features"
stop = "trial share spent"
kill = "the five-feature model never beats tenure-only"
budget = {usd = 1.5, minutes = 45, trials = 6}

[[branches]]
id = "b3"
question = "What does prior work say about tenure?"
serves = ["h3"]
role = "screener"
method = "screen the discovered sources"
stop = "coverage floor met and every source screened"
kill = "fewer than 3 relevant studies found"
budget = {usd = 0.5, minutes = 30, trials = 0}

[[data]]
name = "table"
source = "synthetic churn table"
recipe = "recipes/make_data.py"

[review]
queries = ["customer churn tenure", "churn prediction features"]
include = [{id = "I1", text = "predicts customer churn"}]
exclude = [{id = "E1", text = "not about churn"}]
coverage_floor = {queries_per_source = 2, hops = 0}
"""
SHA = "a" * 64


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENLAB_AGENT", raising=False)
    root = tmp_path / "ws"
    root.mkdir()
    (root / "openlab.toml").write_text("[agents]\ncap_usd = 10.0\n")
    return root


@pytest.fixture
def drafted(ws):
    P.new(ws / "churn", PROMPT)
    (ws / "churn" / P.FILE).write_text(FULL, encoding="utf-8")
    return study.open(ws / "churn")


@pytest.fixture
def approved(drafted, person_key):
    with person_key.act():
        P.approve(drafted, "the plan to run", "person:sam")
    return drafted


def errors(raw, limit=None):
    return [p["error"] for p in P.problems(raw, limit)]


def edit(s, old, new):
    p = s.folder / P.FILE
    text = p.read_text()
    assert old in text
    p.write_text(text.replace(old, new, 1))


def test_new_scaffolds_a_study_and_a_skeleton_that_lint_refuses(ws, capsys):
    assert propose_cli.main(["new", str(ws / "churn"), PROMPT, "--draft-with", "none"]) == 0
    assert "next: `openlab propose lint" in capsys.readouterr().out
    raw = P.load(ws / "churn")
    assert raw["question"] == "Does tenure predict churn better than the majority baseline?"
    assert "Tenure likely carries" in raw["background"] and raw["mode"] == "static"
    assert (ws / "churn" / "study.toml").is_file() and (ws / "churn" / P.MD).is_file()
    assert propose_cli.main(["lint", str(ws / "churn")]) == 1
    out = capsys.readouterr().out
    assert "`hypotheses.h1.falsifier` is required: every hypothesis has a falsifier" in out
    assert "`branches.b1.stop` is required" in out and "`branches.b1.kill` is required" in out
    assert "`metrics` is required: an experiment branch needs the primary metric" in out
    with pytest.raises(StudyError, match="already has proposal.toml"):
        P.new(ws / "churn", PROMPT)


def test_the_proposal_refusals(drafted):
    raw = P.load(drafted.folder)
    assert P.problems(raw, P.cap(drafted.folder)) == []
    bad = json.loads(json.dumps(raw))
    del bad["hypotheses"][0]["falsifier"]
    del bad["branches"][1]["budget"]
    bad["branches"][2]["stop"] = ""
    bad["branches"][0]["kill"] = " "
    bad["branches"][0]["serve"] = ["h1"]
    got = errors(bad)
    assert any("hypotheses.h1.falsifier` is required" in e for e in got)
    assert any("branches.b2.budget` is required" in e for e in got)
    assert any("branches.b3.stop` is required" in e for e in got)
    assert any("branches.b1.kill` is required" in e for e in got)
    assert any("branches.b1.serve` is not a known key" in e for e in got)
    over = json.loads(json.dumps(raw))
    over["branches"][0]["budget"]["usd"] = 3.0
    over["branches"][0]["budget"]["trials"] = 9
    over["branches"][1]["budget"]["minutes"] = 120
    got = errors(over)
    assert "proposal.toml: `budget.usd` is less than the branches' sum ($ 5 > 4)" in got
    assert "proposal.toml: `budget.trials` is less than the branches' sum (trials 15 > 12)" in got
    assert any("branches.b2.budget.minutes` is more than the total" in e for e in got)
    assert any("more than the workspace cap ($ 4 > 3)" in e for e in errors(raw, {"usd": 3.0, "minutes": 480}))
    nometric = {**raw, "metrics": []}
    assert any("an experiment branch needs the primary metric" in e for e in errors(nometric))
    norev = {k: v for k, v in raw.items() if k != "review"}
    assert any("`review` is required for a literature branch" in e for e in errors(norev))
    assert any("`mode` is review, but a branch runs experiments" in e for e in errors({**raw, "mode": "review"}))
    assert "hypothesis h9" not in " ".join(P.warnings(raw))
    raw["hypotheses"].append({"id": "h9", "statement": "s", "prediction": "p", "falsifier": "f"})
    assert P.warnings(raw) == ["hypothesis h9 is served by no branch: it will stay untouched"]
    md = P.render(raw)
    for part in ("# Tenure and churn", "| h1 |", "### b1: Does tenure alone beat majority? (experimenter)",
                 "- stop rule: 3 variants", "- kill criterion: no variant", "Primary: auc", "## Literature review",
                 "Total: $4, 90 min, 12 trials, 1 holdout looks. The branches take $3.5 and 12 trials"):
        assert part in md, part
    with pytest.raises(StudyError, match="falsifier"):
        bad_s = drafted
        edit(bad_s, 'falsifier = "a paired test against majority with p >= 0.05"', 'falsifier = ""')
        P.approve(bad_s, "go", "terminal:sam")
    assert drafted.record.find("proposal") == [] and P.approved(drafted.record) is None


def test_approval_writes_the_node_study_toml_the_tree_and_the_seal(approved):
    s, rec = approved, approved.record
    node = P.approved(rec)
    assert node.actor == "person" and node.session == "person:sam" and personkey.node_signed(rec, node)
    assert node.payload["sha"] == P.sha(P.load(s.folder))
    spec = s.spec
    assert spec.question == "Does tenure predict churn better than the majority baseline?"
    assert spec.evaluate["metrics"] == ["auc", "brier"] and spec.evaluate["baselines"] == ["majority"]
    assert spec.budget == {"trials": 12, "holdout_looks": 1}
    assert spec.review["queries"] == ["customer churn tenure", "churn prediction features"]
    plan = rec.last("plan")
    assert plan is not None and plan.actor == "person"                      # approval sealed it
    tree = node.payload["tree"]
    root = rec.get(tree["root"])
    assert root.kind == "question" and root.actor == "person"
    for b in ("b1", "b2", "b3"):
        assert rec.links(tree[b], rel="part_of")[0][2] == tree["root"]
    h1 = rec.get(tree["h1"])
    assert h1.payload["falsified_by"] == "a paired test against majority with p >= 0.05"
    assert rec.links(h1.id, rel="part_of")[0][2] == tree["b1"]
    assert {d for _, _, d in rec.links(node.id, rel="plans")} == set(tree.values())
    assert "Status: approved as proposal:" in (s.folder / P.MD).read_text()
    with pytest.raises(StudyError, match="already approved"):
        P.approve(s, "again", "person:sam")
    with pytest.raises(StudyError, match="study.toml was edited after the seal"):
        (s.folder / "study.toml").write_text((s.folder / "study.toml").read_text().replace("trials = 12", "trials = 50"))
        from openlab.study import spec as _spec
        _spec.check(s.folder, rec)


def test_an_agent_cannot_approve(drafted, person_home):
    raw = P.load(drafted.folder)
    with pytest.raises(StudyError, match="approved by a person"):
        drafted.record.add(Proposal(proposal=raw, sha=P.sha(raw), tree={}, reason="test", protected=[]), actor="agent")
    with pytest.raises(StudyError, match="signed with the person key"):
        drafted.record.add(Proposal(proposal=raw, sha=P.sha(raw), tree={}, reason="test", protected=[]), actor="person")
    with pytest.raises(StudyError, match="openlab person init"):
        propose_cli._person("approve", "")
    assert propose_cli.main(["approve", str(drafted.folder), "--reason", "x"]) == 1
    assert drafted.record.find("proposal") == []


def test_an_edit_after_approval_is_refused_until_amended(approved, monkeypatch, capsys):
    s = approved
    edit(s, "usd = 1.5, minutes = 45, trials = 6}", "usd = 2.0, minutes = 45, trials = 6}")
    edit(s, "usd = 4.0", "usd = 4.5")
    out = P.lint(s)
    assert not out["ok"] and out["changed"] == ["branches.b1.budget.usd", "budget.usd"]
    assert "branches.b1.budget.usd: 1.5 -> 2.0" in out["drift"]
    with pytest.raises(StudyError, match="the approved proposal changed: branches.b1.budget.usd") as e:
        P.execute(s, dry_run=True)
    assert "record an amendment" in e.value.fix
    with pytest.raises(StudyError, match="can only be amended by a person"):
        P.amend(s, "more for b1", person=False)
    assert propose_cli.main(["amend", str(s.folder), "--reason", "more for b1", "--person"]) == 0   # passphrase
    assert "a person's, for branches.b1.budget.usd, budget.usd" in capsys.readouterr().out
    assert P.lint(s)["ok"]
    # an unprotected change an agent may record; a new branch gets its question node
    edit(s, 'risks = ["the label', 'risks = ["class imbalance", "the label')
    text = (s.folder / P.FILE).read_text() + ('\n[[branches]]\nid = "b4"\nquestion = "Is it calibrated?"\n'
                                               'role = "reviewer"\nmethod = "reliability diagram"\nstop = "one look"\n'
                                               'kill = "none"\nbudget = {usd = 0.4, minutes = 10, trials = 0}\n')
    (s.folder / P.FILE).write_text(text)
    with acting_as("lead-1"):
        out = P.amend(s, "a calibration check", person=False)
    assert out["protected"] == [] and len(out["new_nodes"]) == 1
    node = P.approved(s.record)
    assert node.actor == "agent" and node.payload["tree"]["b4"] == out["new_nodes"][0]
    assert s.record.links(node.id, rel="amends")[0][2] == s.record.find("proposal")[1].id


def test_one_signed_proposal_amendment_carries_holdout_looks_into_study_toml(approved, person_key, capsys):
    """Round 4: `propose amend --person` changing holdout_looks reached proposal.toml but not study.toml, so the
    person had to sign a second `study amend --person`. One signed act now writes both, and status agrees."""
    s = approved
    assert s.status()["looks"]["budget"] == 1
    edit(s, "holdout_looks = 1", "holdout_looks = 2")
    seen = []
    person_key.mp.setattr(person, "_tty_out", lambda text: seen.append(text))
    assert propose_cli.main(["amend", str(s.folder), "--reason", "one more look for b3", "--person"]) == 0
    out = capsys.readouterr().out
    assert "study.toml amended as amendment:" in out and "budget.holdout_looks" in out
    assert any("study.toml budget.holdout_looks: 1 -> 2" in t for t in seen)    # shown before the passphrase
    assert "holdout_looks = 2" in (s.folder / "study.toml").read_text()
    s = study.open(s.folder)
    assert s.status()["looks"]["budget"] == 2
    am = s.record.last("amendment")
    assert am.actor == "person" and am.payload["person"] and "budget.holdout_looks" in am.payload["changes"]
    assert personkey.node_signed(s.record, am) and P.lint(s)["ok"] and s.verify()["ok"]
    with pytest.raises(StudyError, match="matches the sealed spec"):     # nothing left for a second amendment
        s.amend("again", person=False)


def test_an_agents_unprotected_proposal_amendment_leaves_study_toml_alone(approved):
    s = approved
    before = (s.folder / "study.toml").read_text()
    edit(s, 'risks = ["the label', 'risks = ["class imbalance", "the label')
    with acting_as("lead-1"):
        out = P.amend(s, "a risk", person=False)
    assert out["study_amendment"] is None and (s.folder / "study.toml").read_text() == before


def test_execute_dry_run_prints_the_brief_and_the_planned_tree(approved, capsys):
    out = P.execute(approved, dry_run=True)
    brief = out["brief"]
    tree = P.approved(approved.record).payload["tree"]
    states = {x["branch"]["id"]: x for x in P.branch_states(approved)}
    for part in ("openlab_agent_spawn", "never more", "openlab_study_dead_end", 'openlab_tool("propose_stop"',
                 "openlab_agent_request", "tested, killed and untouched", tree["h1"], states["b1"]["item"],
                 "budget: $1.5, 45 min, 6 trials; stop when: 3 variants in a row without an AUC gain of 0.005; "
                 "kill if: no variant beats majority at p < 0.05",
                 "falsified if: a paired test against majority with p >= 0.05"):
        assert part in brief, part
    assert len(brief.split()) < 900
    assert out["tree"][0].startswith("lead (role lead, budget $4, 90 min)")
    assert any(line.strip().startswith("└ b3 [screener] open") for line in out["tree"])
    items = {i["id"]: i for i in approved.frontier(None)["items"]}
    assert all(states[b]["item"] in items and items[states[b]["item"]]["kind"] == "branch" for b in states)
    assert not any(i["kind"] == "untested" and i["target"] == tree["b1"] for i in items.values())
    assert approved.record.find("action", verb="execute") == []             # nothing recorded, nothing spawned
    assert propose_cli.main(["execute", str(approved.folder), "--dry-run"]) == 0
    assert "the planned tree:" in capsys.readouterr().out


def test_execute_spawns_the_lead_with_the_total_budget(approved, ws, tmp_path, monkeypatch):
    install(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_ARGV", str(tmp_path / "argv.json"))
    out = P.execute(approved, cli="claude")
    st = Store(ws)
    lead = st.get(out["lead"])
    assert lead["role"] == "lead" and lead["study"] == "churn" and lead["budget"]["usd"] == 4.0
    assert lead["budget"]["minutes"] == 90.0
    done = final(st, lead["id"])
    assert done["status"] == "done"
    argv = json.loads((tmp_path / "argv.json").read_text())["argv"]
    assert "--max-budget-usd" not in argv                         # openlab pauses at the limit, not the CLI
    prompt = argv[argv.index("-p") + 1]
    assert prompt.startswith("Execute the approved research proposal of the study churn") and "b3 [screener]" in prompt
    assert "role: lead" in argv[argv.index("--append-system-prompt") + 1]
    act = approved.record.last("action", verb="execute")
    assert act.payload["args"]["lead"] == lead["id"]
    assert out["copies"] == ["branches/b1", "branches/b2"] and act.payload["args"]["copies"]["b1"] == "branches/b1"
    assert (approved.folder / "branches" / "b2" / "model.py").is_file()                 # each experiment branch's own
    assert not (approved.folder / "branches" / "b3").exists()                           # a screener runs no code
    assert 'code="branches/b1"' in prompt and "edits only that copy" in prompt
    st.register("lead-live", "spawned", role="lead", study="churn", status="running")
    with pytest.raises(StudyError, match="already executing: lead lead-live"):
        P.execute(approved)


def result(s, key, p=0.01, better="model"):
    rec = s.record
    plan = rec.last("plan")
    ds = rec.last("dataset") or rec.add("dataset", {"name": "table", "source": {"kind": "provider"}, "sha256": SHA,
                                                    "rows": 1, "columns": ["y"], "card": {}})
    run = rec.add("run", {"segment": "research", "key": key, "cached": False, "spec_sha": plan.payload["sha"],
                          "code": {"model.py": SHA}, "data": {"table": SHA}, "env": {"python": "3.11"},
                          "sandbox": "stub", "seconds": 5.0, "audit": {"verdict": "separated", "checked": 0,
                                                                       "mismatches": []}},
                  links=[("under", plan.id), ("uses", ds.id)])
    return rec.add("result", {"metrics": {"auc": {"value": 0.7, "lo": 0.65, "hi": 0.75}},
                              "paired": {"majority": {"metric": "auc", "diff": 0.2, "lo": 0.15, "hi": 0.25, "p": p,
                                                      "better": better}}}, links=[("of", run.id)]).id


def test_status_reads_each_branch_off_the_record_and_the_control_store(approved, ws, capsys):
    s, st = approved, Store(ws)
    tree = P.approved(s.record).payload["tree"]
    states = {x["branch"]["id"]: x for x in P.branch_states(s)}
    st.register("lead-1", "spawned", role="lead", study="churn", status="running",
                budget={"usd": 4.0}, spent={"usd": 0.2})
    st.register("exp-1", "spawned", role="experimenter", parent="lead-1", study="churn", status="done",
                question="b1: Does tenure alone beat majority?", spent={"usd": 0.31, "tokens_in": 9000, "tokens_out": 500})
    st.register("exp-1.sub", "claude-subagent", role="openlab-experimenter", parent="exp-1", status="done",
                spent={"usd": 0.05, "tokens_in": 100, "tokens_out": 10})
    st.register("exp-2", "spawned", role="experimenter", parent="lead-1", study="churn", status="running",
                question="b2: Do the other features add signal?", spent={"usd": 0.4, "tokens_in": 1000})
    with acting_as("exp-1"):
        r1 = result(s, "k1")
        result(s, "k2")
        s.test(tree["h1"], r1, "majority", 0.04)
    with acting_as("exp-1.sub"):
        result(s, "k3")
    with acting_as("exp-2"):
        r2 = result(s, "k4", p=0.5)
        s.dead_end(tree["h2"], "boosting adds nothing", "never beat tenure-only", ["features"], [r2])
        P.stop(s, "b2", "kill criterion met: the five-feature model never beat tenure-only", killed=True,
               shown_by=[r2])
    with acting_as("scr-1"):
        s.take(states["b3"]["item"])
    out = P.status(s)
    by = {b["id"]: b for b in out["branches"]}
    assert out["hypotheses"] == {"tested": 1, "killed": 1, "untouched": 1}
    b1 = by["b1"]
    assert b1["state"] == "done" and {a["id"] for a in b1["agents"]} == {"exp-1", "exp-1.sub"}   # h1 supported
    assert b1["spent"]["usd"] == 0.31 and b1["spent"]["trials"] == 3 and b1["spent"]["tokens"] == 9500
    assert b1["hypotheses"][0]["state"] == "tested" and b1["hypotheses"][0]["verdict"] == "supported"
    assert b1["hypotheses"][0]["by"] == "exp-1" and b1["stop_state"] == "not fired"
    b2 = by["b2"]
    assert b2["state"] == "killed" and b2["hypotheses"][0]["state"] == "killed"
    assert b2["dead_ends"][0]["by"] == "exp-2" and b2["stop_state"].startswith("fired: kill criterion met")
    b3 = by["b3"]
    assert b3["state"] == "held" and b3["held_by"]["agent"] == "scr-1"                   # scr-1 holds the lease
    assert b3["hypotheses"][0]["state"] == "untouched" and b3["agents"] == []
    assert out["lead"][0]["id"] == "lead-1" and out["spent"]["trials"] == 4
    items = {i["id"] for i in s.frontier(None)["items"] if i["kind"] == "branch"}
    assert items == {states["b3"]["item"]}                              # b1 done, b2 killed
    assert propose_cli.main(["status", str(s.folder)]) == 0
    text = capsys.readouterr().out
    assert "hypotheses: 1 tested, 1 killed, 1 untouched" in text and "b2 [experimenter] KILLED" in text
    assert "h1 supported by exp-1" in text and "dead end dead_end:" in text
    env = dispatch(ToolContext(str(ws)), "propose_status", {"folder": "churn"})
    assert env["ok"] and env["result"]["hypotheses"]["killed"] == 1


def test_status_words_are_truthful_stopped_refuted_killed(approved, ws, capsys):
    """Round 4: a branch whose stop rule fired read KILLED, and a hypothesis whose falsifier fired read `killed
    (inconclusive)`. Now: STOPPED (stop rule) for a stop, KILLED only for a met kill criterion, a hypothesis reads its
    verdict, and a killed branch's untested hypothesis reads killed."""
    s = approved
    tree = P.approved(s.record).payload["tree"]
    with acting_as("exp-1"):
        r1 = result(s, "k1")
        s.test(tree["h1"], r1, "majority", 0.04)                    # supported
        with pytest.raises(StudyError, match="b1 was not killed: h1 is supported"):
            P.stop(s, "b1", "kill criterion met", killed=True)
        P.stop(s, "b1", "stop rule: 3 variants in a row without an AUC gain of 0.005")
    with acting_as("exp-2"):
        r2 = result(s, "k2", p=0.001, better="baseline")
        s.test(tree["h2"], r2, "majority", 0.04)                    # the falsifier fired: refuted
        s.dead_end(tree["h2"], "the five features", "majority wins", ["features"], [r2])
        P.stop(s, "b2", "kill criterion met: never beats tenure-only", killed=True, shown_by=[r2])
    with acting_as("scr-1"):
        P.stop(s, "b3", "kill criterion met: fewer than 3 relevant studies", killed=True)
    by = {b["id"]: b for b in P.status(s)["branches"]}
    assert by["b1"]["state"] == "stopped" and by["b1"]["label"] == "STOPPED (stop rule)"
    assert by["b1"]["hypotheses"][0]["label"] == "supported"
    assert by["b2"]["state"] == "killed" and by["b2"]["hypotheses"][0]["label"] == "refuted"
    assert by["b3"]["state"] == "killed" and by["b3"]["hypotheses"][0]["label"] == "killed"   # untested, branch killed
    assert propose_cli.main(["status", str(s.folder)]) == 0
    text = capsys.readouterr().out
    assert "b1 [experimenter] STOPPED (stop rule)" in text and "b1 [experimenter] KILLED" not in text
    assert "b2 [experimenter] KILLED (kill criterion)" in text and "h2 refuted (dead end recorded)" in text
    assert "h3 killed" in text and "killed (inconclusive)" not in text and "; 1 supported; 1 refuted" in text
    tree_lines = P.planned_tree(s)
    assert any(line.strip().startswith("h2 refuted:") for line in tree_lines)


def test_an_inconclusive_test_reads_inconclusive_even_with_a_dead_end(approved, ws):
    s = approved
    tree = P.approved(s.record).payload["tree"]
    with acting_as("exp-2"):
        r = result(s, "k1", p=0.5)
        s.test(tree["h2"], r, "majority", 0.04)
        s.dead_end(tree["h2"], "the five features", "no gain", ["features"], [r])
    h2 = next(b for b in P.status(s)["branches"] if b["id"] == "b2")["hypotheses"][0]
    assert h2["state"] == "tested" and h2["label"] == "inconclusive"


def test_trials_over_the_share_say_the_stop_rule_should_fire(approved, ws):
    s, st = approved, Store(ws)
    st.register("exp-1", "spawned", role="experimenter", study="churn", status="running",
                question="b1: Does tenure alone beat majority?")
    with acting_as("exp-1"):
        for k in range(6):
            result(s, f"t{k}")
    b1 = next(b for b in P.status(s)["branches"] if b["id"] == "b1")
    assert b1["state"] == "working" and b1["stop_state"].startswith("budget spent (trials)")


def test_propose_tools(ws):
    ctx = ToolContext(str(ws))
    env = dispatch(ctx, "propose_new", {"folder": "lab", "prompt": PROMPT, "mode": "review"})
    assert env["ok"] and env["result"]["mode"] == "review"
    assert study.open(ws / "lab").spec.mode == "review"
    lint = dispatch(ctx, "propose_lint", {"folder": "lab"})
    assert lint["ok"] and not lint["result"]["ok"] and lint["result"]["problems"]
    show = dispatch(ctx, "propose_show", {"folder": "lab"})
    assert show["ok"] and show["result"]["md"].startswith("# (untitled)")
    assert not dispatch(ctx, "propose_status", {"folder": "lab"})["ok"]


def test_new_drafts_with_a_planning_agent_on_a_small_budget(ws, tmp_path, monkeypatch, capsys):
    install(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_ARGV", str(tmp_path / "argv.json"))
    assert propose_cli.main(["new", str(ws / "churn"), PROMPT, "--draft-with", "claude"]) == 0
    assert "drafting with planner-" in capsys.readouterr().out
    st = Store(ws)
    planner = st.find(role="planner")[0]
    assert planner["study"] == "churn" and planner["budget"]["usd"] == 0.5 and planner["budget"]["minutes"] == 15
    final(st, planner["id"])
    prompt = json.loads((tmp_path / "argv.json").read_text())["argv"]
    prompt = prompt[prompt.index("-p") + 1]
    assert "openlab propose lint churn" in prompt and "do not approve" in prompt and "Tenure likely carries" in prompt


# --- approval checks: what would block execution is found before approval ----------------------------------------
@pytest.fixture
def sibling(ws):
    """demo: a sealed study of the same template data and target (its dataset registered, no run needed)."""
    d = study.new(ws / "demo", "static")
    d.seal()
    d.materialise()
    return d


def test_lint_and_approve_find_a_sibling_on_the_same_data(sibling, drafted, capsys, person_key):
    out = P.lint(drafted)
    assert not out["ok"] and out["conflicts"][0]["name"] == "demo" and out["conflicts"][0]["rel"] == "../demo"
    assert any("churn asks the question of demo" in p["error"] and 'continues = "../demo"' in p["fix"]
               and "fresh =" in p["fix"] for p in out["problems"])
    assert propose_cli.main(["lint", str(drafted.folder)]) == 1
    text = capsys.readouterr().out
    assert "same data as demo" in text and "continue it:" in text and "or fresh:" in text
    with pytest.raises(StudyError, match="asks the question of demo"):
        P.approve(drafted, "go", "terminal:sam")
    assert drafted.record.find("proposal") == [] and not drafted.record.find("plan")      # nothing written
    P.declare(drafted.folder, "continues", "../demo")
    assert P.lint(drafted)["ok"]
    with person_key.act():
        P.approve(drafted, "continue demo's ledger", "person:sam")
    assert drafted.spec.raw["study"]["continues"] == "../demo"
    from openlab.study import ledger
    assert set(ledger.members(drafted.folder, drafted.record)) == {sibling.folder}


def test_a_fresh_question_declared_in_the_proposal_is_the_persons_by_approving(sibling, drafted, person_key):
    P.declare(drafted.folder, "fresh", "churn at a new site")
    out = P.lint(drafted)
    assert out["ok"] and any("fresh: approving declares a new question" in w for w in out["warnings"])
    assert "FRESH QUESTION (you declare it by approving): churn at a new site" in P.summary(P.load(drafted.folder))
    with person_key.act():
        P.approve(drafted, "a new site", "person:sam")
    from openlab.study import ledger
    assert drafted.record.last("plan").actor == "person" and ledger._fresh(drafted.record) == "churn at a new site"
    edit(drafted, 'fresh = "churn at a new site"', 'fresh = "another"')
    assert P.protected(P.diff(P.approved(drafted.record).payload["proposal"], P.load(drafted.folder)),
                       {}, {}) == ["fresh"]


def test_the_approval_screen_asks_continue_or_fresh(sibling, drafted, monkeypatch, capsys, person_key):
    import io

    class Tty(io.StringIO):
        def isatty(self):
            return True
    monkeypatch.setattr("sys.stdin", Tty("c\n"))
    monkeypatch.setattr("sys.stderr", Tty())
    assert propose_cli.main(["approve", str(drafted.folder), "--reason", "go"]) == 0
    assert P.load(drafted.folder)["continues"] == "../demo" and P.approved(drafted.record) is not None


def test_lint_finds_a_missing_model_and_recipe_and_an_empty_review(drafted):
    (drafted.folder / "model.py").unlink()
    (drafted.folder / "recipes" / "make_data.py").unlink()
    errs = [p["error"] for p in P.lint(drafted)["problems"]]
    assert any("churn/model.py is missing, and the experimenter branches run it" in e for e in errs)
    assert any("`data.table.recipe` names recipes/make_data.py, which does not exist" in e for e in errs)
    with pytest.raises(StudyError, match="model.py is missing"):
        P.approve(drafted, "go", "terminal:sam")
    edit(drafted, 'name = "table"', 'name = "other"')
    assert any("`data.other` is not a dataset of study.toml" in p["error"] for p in P.lint(drafted)["problems"])


# --- resume: after a person acts on what blocked a branch, execution goes on --------------------------------------
def test_a_branch_blocked_on_a_person_request_is_resumed(approved, ws, tmp_path, monkeypatch, person_key, capsys):
    from openlab.agents import inbox
    s, st = approved, Store(ws)
    approved._act("execute", {"proposal": P.approved(s.record).id, "lead": "lead-1"}, "agent", lambda: {})
    st.register("lead-1", "spawned", role="lead", study="churn", status="done", spent={"usd": 0.5})
    st.register("exp-1", "spawned", role="experimenter", parent="lead-1", study="churn", status="done",
                question="b1: Does tenure alone beat majority?", spent={"usd": 0.25})
    item = st.request("exp-1", "fresh", "demo uses the same data; a person must choose", study="churn",
                      args=inbox.check_request(st, "exp-1", "fresh", {"why": "tenure alone, a new question"}, "churn"))
    with acting_as("exp-2"):
        P.stop(s, "b2", "the plan changed: b2 is not needed", shown_by=[])
    b1 = next(b for b in P.status(s)["branches"] if b["id"] == "b1")
    assert b1["state"] == "blocked" and f"inbox #{item} (fresh, pending)" in b1["blocked"]
    assert f"openlab inbox approve {item}" in b1["blocked_by"]["fix"]
    assert propose_cli.main(["status", str(s.folder)]) == 0
    out = capsys.readouterr().out
    assert "b1 [experimenter] BLOCKED" in out and "   BLOCKED: exp-1 ended on inbox" in out and "   fix: " in out
    assert inbox.requester_note(st, st.item(item)).startswith("the requester has ended")
    with person_key.act():
        inbox.decide(st, item, True, "yes, a new question")
    assert 'fresh = "tenure alone, a new question"' in (s.folder / "study.toml").read_text()
    from openlab.study import ledger
    assert ledger._fresh(s.record) == "tenure alone, a new question"
    seen = dispatch(ToolContext(str(ws)), "agent_inbox_item", {"id": item})["result"]
    assert seen["status"] == "approved" and seen["decided"] and seen["decision_reason"] == "yes, a new question"
    b1 = next(b for b in P.status(s)["branches"] if b["id"] == "b1")
    assert "--resume" in b1["blocked_by"]["fix"]
    dry = P.execute(s, dry_run=True, resume=True)
    assert dry["resume"]["left"] == ["b1", "b3"] and dry["resume"]["finished"] == ["b2 stopped"]
    assert dry["budget"]["usd"] == 3.25                                     # 4 less the lead's and exp-1's spend
    brief = dry["brief"]
    assert brief.startswith("This RESUMES") and "b2 stopped" in brief and f"#{item} fresh from exp-1 approved" in brief
    assert "- b1 [experimenter]" in brief and "- b2 [" not in brief and "- b3 [screener]" in brief
    install(tmp_path, monkeypatch)
    out = P.execute(s, resume=True)
    lead = st.get(out["lead"])
    assert lead["budget"]["usd"] == 3.25 and s.record.last("action", verb="execute").payload["args"]["resume"]
    final(st, lead["id"])


def test_resume_is_refused_with_nothing_to_resume_or_no_budget(approved, ws):
    with pytest.raises(StudyError, match="no execution to resume"):
        P.execute(approved, dry_run=True, resume=True)
    approved._act("execute", {"lead": "lead-1"}, "agent", lambda: {})
    Store(ws).register("lead-1", "spawned", role="lead", study="churn", status="done", spent={"usd": 4.0})
    with pytest.raises(StudyError, match="budget is spent") as e:
        P.execute(approved, dry_run=True, resume=True)
    assert "propose amend churn --person" in e.value.fix


def test_fresh_and_continues_are_inbox_kinds(ws):
    from openlab.agents import inbox
    st = Store(ws)
    st.register("exp-1", "spawned", role="experimenter", status="running")
    with pytest.raises(inbox.RequestError, match="says why"):
        inbox.check_request(st, "exp-1", "fresh", {}, "churn")
    with pytest.raises(inbox.RequestError, match="names the study whose ledger"):
        inbox.check_request(st, "exp-1", "continues", {}, "churn")
    assert inbox.check_request(st, "exp-1", "continues", {"continues": "../demo"}, "churn") == {"continues": "../demo"}


# --- model and budget controls at every spawn point ---------------------------------------------------------------
def test_the_planner_takes_a_model_and_a_budget(ws, tmp_path, monkeypatch, capsys):
    install(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_ARGV", str(tmp_path / "argv.json"))
    assert propose_cli.main(["new", str(ws / "churn"), PROMPT, "--draft-with", "claude", "--model", "haiku",
                             "--planner-budget-usd", "0.2"]) == 0
    assert "(claude, haiku, budget $0.2," in capsys.readouterr().out
    st = Store(ws)
    planner = st.find(role="planner")[0]
    assert planner["budget"]["usd"] == 0.2
    final(st, planner["id"])
    argv = json.loads((tmp_path / "argv.json").read_text())["argv"]
    assert argv[argv.index("--model") + 1] == "haiku"
    with pytest.raises(StudyError, match="above \\$0"):
        P.draft(ws / "churn", PROMPT, usd=0)


def test_execute_takes_a_model_its_children_inherit_and_the_workspace_default(approved, ws, tmp_path, monkeypatch):
    from openlab.agents import spawn as S
    from openlab.agents.supervisor import command
    install(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_ARGV", str(tmp_path / "argv.json"))
    (ws / "openlab.toml").write_text('[agents]\ncap_usd = 10.0\nmodel = "sonnet"\n')
    st = Store(ws)
    out = P.execute(approved, cli="claude")
    final(st, out["lead"])
    argv = json.loads((tmp_path / "argv.json").read_text())["argv"]
    assert argv[argv.index("--model") + 1] == "sonnet" and out["model"] == "sonnet"      # [agents] model
    out = P.execute(approved, cli="claude", model="haiku")
    final(st, out["lead"])
    argv = json.loads((tmp_path / "argv.json").read_text())["argv"]
    assert argv[argv.index("--model") + 1] == "haiku"
    assert approved.record.last("action", verb="execute").payload["args"]["model"] == "haiku"
    st.update(out["lead"], status="running")
    child = S.spawn(ws, "b1 child", role="experimenter", parent=out["lead"], budget={"usd": 1.0, "minutes": 10},
                    start=False)
    d = st.agent_dir(child["id"])
    cmd = command(ws, child, d)
    assert cmd[cmd.index("--model") + 1] == "haiku"                                    # inherited from the lead
    cx = S.spawn(ws, "a codex agent", cli="codex", model="gpt-5-mini", start=False)
    cmd = command(ws, cx, st.agent_dir(cx["id"]))
    assert cmd[cmd.index("-m") + 1] == "gpt-5-mini"
    env = dispatch(ToolContext(str(ws)), "agent_spawn", {"prompt": "x", "role": "r", "model": "opus",
                                                          "budget_usd": 0.1, "minutes": 1})
    assert env["ok"] and env["result"]["model"] == "opus"


def test_the_prompt_may_follow_the_options_or_come_as_an_option(ws, capsys):
    assert propose_cli.main(["new", str(ws / "a"), "--draft-with", "none", "--mode", "static", PROMPT]) == 0
    assert P.load(ws / "a")["question"] == "Does tenure predict churn better than the majority baseline?"
    assert propose_cli.main(["new", str(ws / "b"), "--draft-with", "none", "--prompt", PROMPT]) == 0
    assert "Tenure likely carries" in P.load(ws / "b")["background"]
    with pytest.raises(SystemExit):
        propose_cli.main(["new", str(ws / "c"), "--draft-with", "none", PROMPT, "--bogus"])


def test_the_project_mcp_json_uses_the_openlab_mcp_launcher_when_installed(ws, monkeypatch):
    import shutil
    import sys
    from openlab.agents import spawn as S
    real = shutil.which
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/openlab-mcp" if name == "openlab-mcp" else real(name))
    assert S.mcp_config(ws, launcher=True)["mcpServers"]["openlab"]["command"] == "openlab-mcp"
    assert S.mcp_config(ws, "exp-1")["mcpServers"]["openlab"]["command"] == sys.executable      # a spawned agent's
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert S.mcp_config(ws, launcher=True)["mcpServers"]["openlab"]["args"] == ["-m", "openlab.mcp_server"]


def test_an_extract_kind_defaults_from_its_section():
    from openlab.study.evidence import kind_of
    assert kind_of("", "Table 2") == "table" and kind_of("", "Fig. 3") == "figure"
    assert kind_of("", "4.1 Results") == "passage" and kind_of("text", "Table 2") == "passage"
    assert kind_of("table", "4.1 Results") == "table"
    assert kind_of("chart", "") == "chart"                         # the record's schema refuses it, as before
