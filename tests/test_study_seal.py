"""The seal on purpose (B11): `openlab study seal` runs a study's first research run, which writes study.toml into
the record as the plan and is trial 1, after saying what it freezes; it is refused once sealed (naming when and the
plan node) and before the data is materialised. The snapshot of openlab show carries each study's seal, the page's
Seal is a person's act through the server's own protections (bearer token, reviewer secret, passphrase), and an
experiment of an unsealed study says so in its preview, its status and its request. Temp workspaces only."""

import shutil

import pytest

import openlab.study as vs
from openlab.agents.store import Store
from openlab.app import ToolContext, dispatch
from openlab.study import cli, sealing
from openlab.study import spec as _spec
from openlab.study.errors import StudyError

runs = pytest.mark.skipif(not shutil.which("bwrap"), reason="a research run is sandboxed under bwrap")


def _ws(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "openlab.toml").write_text("# a test workspace\n")
    Store(root)
    return root


def test_every_protected_field_is_said_in_plain_words():
    assert set(sealing.WHAT) == set(_spec.PERSON_FIELDS)
    t = sealing.terms({"split": {"scheme": "walk_forward", "research": ["2020-01-01", "2023-12-31"],
                                 "holdout": ["2024-01-01", "2024-12-31"], "embargo": "5d"},
                       "budget": {"trials": 12, "holdout_looks": 1}, "evaluate": {"metrics": ["net_mean", "nw_t"]},
                       "instruments": {"tender": "tender@0123456789ab"},
                       "data": {"panel": {"known_at": "date"}}})
    got = {r["field"]: r["now"] for r in t["freezes"]}
    assert got["split"] == "walk_forward, research 2020-01-01 to 2023-12-31, holdout 2024-01-01 to 2024-12-31, embargo 5d"
    assert got["budget"] == "12 trials, 1 holdout look(s)"
    assert got["evaluate.metrics[0]"] == "net_mean"
    assert got["instruments"] == "tender = tender@0123456789ab"
    assert got["data.*.known_at"] == "panel: date"
    assert "trial 1" in t["cost"] and "not touched" in t["holdout"] and "--person" in t["after"]
    text = "\n".join(sealing.lines(t))
    assert "the split and holdout" in text and "the primary metric: net_mean" in text


@runs
def test_seal_refuses_unmaterialised_data_then_seals_as_trial_1_then_refuses_again(tmp_path, capsys):
    s = vs.new(tmp_path / "s1", "static")
    folder = str(s.folder)
    assert cli.main(["seal", folder]) == 1
    err = capsys.readouterr().err
    assert "not materialised: table" in err and "openlab study materialise" in err
    assert not sealing.state(s.folder)["sealed"]
    assert vs.open(s.folder).record.last("action", verb="seal").payload["outcome"]["ok"] is False   # recorded
    vs.open(s.folder).materialise()
    capsys.readouterr()

    assert cli.main(["seal", folder, "--note", "the plan as agreed"]) == 0
    out = capsys.readouterr().out
    before, after = out.split("running the first research run now", 1)
    assert "sealing freezes study.toml" in before and "the primary metric: auc" in before
    assert "trial 1 of the budget (20)" in before and "holdout is not touched" in before
    assert "openlab study amend" in before
    st = sealing.state(s.folder)
    assert st["sealed"] and st["plan"] in after and "trial 1 of 20" in after
    runs_ = vs.open(s.folder).runs()
    assert runs_["trials"] == 1 and runs_["runs"][0]["trial"] == 1 and runs_["runs"][0]["note"] == "the plan as agreed"
    assert runs_["runs"][0]["result"] in after

    assert cli.main(["seal", folder]) == 1
    err = capsys.readouterr().err
    assert f"already sealed: {st['plan']} at {st['at']}" in err and "openlab study amend" in err
    assert vs.open(s.folder).runs()["trials"] == 1                       # the refusal spent nothing
    assert vs.open(s.folder).verify()["ok"]


def test_a_review_study_is_sealed_by_its_discover_not_a_run(tmp_path):
    s = vs.new(tmp_path / "rev", "review")
    with pytest.raises(StudyError, match="review study"):
        s.seal_terms()


def test_the_mcp_tool_refuses_before_anything_runs(tmp_path):
    root = _ws(tmp_path)
    vs.new(root / "s1", "static")
    env = dispatch(ToolContext(str(root)), "study_seal", {"folder": "s1"})
    assert not env["ok"] and "not materialised" in env["error"] and "materialise" in env["try"]
    from openlab.mcp_server import AGENT_TOOLS
    assert "study_seal" in AGENT_TOOLS
    assert not sealing.state(root / "s1")["sealed"]


@runs
def test_the_snapshot_says_which_studies_are_sealed_and_what_sealing_one_freezes(tmp_path):
    root = _ws(tmp_path)
    ready = vs.new(root / "ready", "static")
    ready.materialise()
    raw = vs.new(root / "raw", "static")                               # no data yet
    kinds = raw.record.kinds()
    done = vs.new(root / "done", "static")
    done.materialise()
    done.seal_run(note="sealed by the agent")
    snap = dispatch(ToolContext(str(root)), "show_snapshot", {})
    assert snap["ok"]
    by = {s["study"]: s for s in snap["result"]["studies"]}
    assert by["ready"]["sealed"] is False and by["ready"]["ready"] is True
    fields = [r["field"] for r in by["ready"]["terms"]["freezes"]]
    assert {"split", "budget", "evaluate.metrics[0]"} <= set(fields)
    assert by["ready"]["terms"]["trials"] == 20 and by["ready"]["terms"]["primary"] == "auc"
    assert by["raw"]["ready"] is False and "not materialised" in by["raw"]["blocked"] and "materialise" in by["raw"]["fix"]
    assert vs.open(root / "raw").record.kinds() == kinds                # the snapshot wrote nothing
    assert by["done"]["sealed"] and by["done"]["plan"].startswith("plan:") and "terms" not in by["done"]


@pytest.fixture
def server_ws(tmp_path, person_key):
    pytest.importorskip("fastapi")
    from openlab.app import server
    home = tmp_path / "home"
    root = home / "lab"
    root.mkdir(parents=True)
    (root / "openlab.toml").write_text("")
    Store(root)
    s = vs.new(root / "s1", "static")
    s.materialise()
    return home, root, server.create_app(str(home), token="tok")


@runs
def test_the_pages_seal_is_a_persons_act_through_the_servers_protections(server_ws):
    from fastapi.testclient import TestClient
    from conftest import PASSPHRASE
    home, root, app = server_ws
    url, H = "/api/tool/show_study_seal?ws=lab", {"authorization": "Bearer tok"}
    P = {**H, "x-openlab-reviewer": app.state.reviewer}
    with TestClient(app) as c:
        assert c.post(url, json={"study": "s1"}).status_code == 401                      # no token
        env = c.post(url, headers=H, json={"study": "s1"}).json()
        assert not env["ok"] and env["error"] == "a reviewer's act"                      # no reviewer secret
        env = c.post(url, headers={**H, "x-openlab-reviewer": "guess"}, json={"study": "s1"}).json()
        assert not env["ok"]
        env = c.post(url, headers=P, json={"study": "s1"}).json()
        assert not env["ok"] and env["needs"] == "passphrase"                            # the secret alone is not a person
        env = c.post(url, headers=P, json={"study": "s1", "passphrase": "wrong one"}).json()
        assert not env["ok"] and env["error"] == "passphrase"
        assert not sealing.state(root / "s1")["sealed"]                                  # nothing ran
        env = c.post(url, headers=P, json={"study": "s1", "note": "from the page", "passphrase": PASSPHRASE}).json()
        assert env["ok"], env
        r = env["result"]
        assert r["study"] == "s1" and r["trial"].startswith("trial 1 of 20") and r["result"].startswith("result:")
        assert r["metric"] == "auc" and isinstance(r["value"]["value"], float) and r["plan"].startswith("plan:")
        env = c.post(url, headers=P, json={"study": "s1", "passphrase": PASSPHRASE}).json()
        assert not env["ok"] and "already sealed" in env["error"] and r["plan"] in env["error"]
    st = sealing.state(root / "s1")
    assert st["sealed"] and st["by"] == "the person" and st["plan"] == r["plan"]
    s = vs.open(root / "s1")
    assert s.record.get(st["plan"]).actor == "person" and s.verify()["ok"]
    assert s.runs()["trials"] == 1


def test_an_experiment_of_an_unsealed_study_says_so_in_its_preview_status_and_request(tmp_path, person_key,
                                                                                      monkeypatch):
    import test_experiments as T
    from openlab.agents.identity import acting_as
    from openlab.study import experiment_cli
    from openlab.study import experiments as E
    fake, tab = tmp_path / "crontab", tmp_path / "crontab.txt"
    fake.write_text(f'#!/bin/sh\nif [ "$1" = "-l" ]; then cat "{tab}" 2>/dev/null || exit 1; else cat > "{tab}"; fi\n')
    fake.chmod(0o755)
    monkeypatch.setenv("OPENLAB_CRONTAB", str(fake))
    ws = _ws(tmp_path)
    pin = T.register(ws, person_key)
    folder = T.make_study(ws, {"tender": pin})
    T.save(folder, T.spec())
    Store(ws).register("lead-1", "spawned", role="lead", status="running", budget={"usd": 5})
    with acting_as("lead-1"):
        req = E.request(ws, "tender", "paper_trial", "the forward paper trial")
    assert "tender is not sealed" in req["note"] and "openlab study seal tender" in req["note"]
    _, preview = E.describe(ws, "tender", "paper_trial", req["sha"])
    assert any("tender is not sealed" in line and "Seal button" in line for line in preview)
    status = E.status(ws, "tender", "paper_trial")
    assert status["study_seal"]["sealed"] is False and "not sealed" in status["study_seal"]["line"]
    text = "\n".join(experiment_cli._text_status(status))
    assert "the study tender is not sealed" in text
    listed = "\n".join(experiment_cli._text_list({"experiments": E.overview(ws)}))
    assert "[study not sealed: openlab study seal tender]" in listed
    snap = dispatch(ToolContext(str(ws)), "show_snapshot", {})["result"]
    x = next(e for e in snap["experiments"] if e["name"] == "paper_trial")
    assert x["study_seal"]["sealed"] is False                           # the Experiments card's line
    item = next(i for i in snap["inbox"] if i["kind"] == "experiment")
    assert item["study"] == "tender" and any(s["study"] == "tender" and not s["sealed"] for s in snap["studies"])

    s = vs.open(folder)                                                 # sealed: the lines are gone
    _spec.seal(folder, s.record, "agent")
    _, preview = E.describe(ws, "tender", "paper_trial", req["sha"])
    assert not any("not sealed" in line for line in preview)
    status = E.status(ws, "tender", "paper_trial")
    assert status["study_seal"]["sealed"] and "line" not in status["study_seal"]
    assert "not sealed" not in "\n".join(experiment_cli._text_status(status))
