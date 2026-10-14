"""openlab show: an agent's plan and its evidence, a question for the person and its answer, the snapshot the page
draws, and the server that serves the page and tells it when the control store changes."""

import time

import pytest

from openlab.agents import plan as P
from openlab.agents.store import Store
from openlab.app import ToolContext, dispatch


@pytest.fixture
def ws(tmp_path, monkeypatch):
    (tmp_path / "openlab.toml").write_text('[team.sam]\nrole = "meteorologist"\nhours = 10\n')
    st = Store(tmp_path)
    st.register("lead-1", "spawned", role="lead", status="running", budget={"usd": 6})
    monkeypatch.setenv("OPENLAB_AGENT", "lead-1")
    return ToolContext(str(tmp_path)), st


STEPS = [{"id": "data", "title": "Data", "kind": "data", "state": "done", "links": ["inbox:1"]},
         {"id": "emos", "title": "Baseline", "kind": "run", "needs": ["data"], "state": "active"},
         {"id": "look", "title": "Holdout look", "kind": "person", "needs": ["emos"], "owner": "you"}]


def test_a_plan_is_checked_before_it_is_kept():
    for bad, words in [([], "non-empty"), ([{"id": "a", "title": "A", "needs": ["b"]}], "does not have"),
                       ([{"id": "a", "title": "A"}, {"id": "a", "title": "B"}], "twice"),
                       ([{"id": "a", "title": "A", "needs": ["b"]}, {"id": "b", "title": "B", "needs": ["a"]}], "cycle"),
                       ([{"id": "a", "title": "A", "state": "finished"}], "state"),
                       ([{"id": "a", "title": "A", "links": ["not a record id"]}], "record id")]:
        with pytest.raises(P.PlanError, match=words):
            P.check(bad)


def test_versions_and_steps_build_the_plan_and_its_times(ws):
    ctx, st = ws
    assert dispatch(ctx, "agent_plan", {"nodes": STEPS, "note": "first"})["result"]["v"] == 1
    assert dispatch(ctx, "agent_plan_node", {"id": "emos", "state": "done", "links": ["run:abcd"]})["ok"]
    env = dispatch(ctx, "agent_plan_node", {"id": "nope", "state": "done"})
    assert not env["ok"] and "no step" in env["error"]
    v = P.read(st, "lead-1")["versions"]
    emos = next(n for n in v[-1]["nodes"] if n["id"] == "emos")
    assert emos["state"] == "done" and emos["links"] == ["run:abcd"] and emos["t0"] <= emos["t1"]
    kinds = [e["kind"] for e in st.events("lead-1")]
    assert "plan" in kinds and "plan_node" in kinds                     # on the hash-chained control store too
    assert st.verify()["ok"]


def test_evidence_is_computed_from_the_links_never_typed(ws):
    ctx, st = ws
    assert P.evidence(st, "", {"state": "active", "links": []})[0] == "rep"         # the agent's word only
    assert P.evidence(st, "", {"state": "done", "links": ["run:abcd"]})[0] == "bad"  # not on any record
    item = st.request("lead-1", "budget", "more", args={"usd": 9})
    ev, text = P.evidence(st, "", {"state": "done", "links": [f"inbox:{item}"]})
    assert ev == "ok" and "pending" in text


def test_a_question_is_answered_by_the_person_unsigned_and_never_approved(ws):
    ctx, st = ws
    env = dispatch(ctx, "agent_request", {"kind": "question", "reason": "How many seeds?", "args": {"choices": ["3", "5"]}})
    item = env["result"]["item"]
    assert not dispatch(ctx, "agent_request", {"kind": "question", "reason": "x", "args": {"choices": ["only one"]}})["ok"]
    assert dispatch(ctx, "inbox_answer", {"id": item, "picks": ["5"]})["error"] == "a reviewer's act"   # an agent cannot
    assert not dispatch(ctx, "inbox_answer", {"id": item, "picks": ["7"]}, human=True)["ok"]           # not a choice
    from openlab.agents.inbox import decide
    with pytest.raises(ValueError, match="question"):
        decide(st, item, True, "yes")
    env = dispatch(ctx, "inbox_answer", {"id": item, "picks": ["5"], "other": "then decide", "message": "ok"}, human=True)
    assert env["ok"] and env["result"]["status"] == "answered"
    got = dispatch(ctx, "agent_inbox_item", {"id": item})["result"]
    assert got["decided"] and got["result"] == {"picks": ["5"], "other": "then decide"} and got["decision_reason"] == "ok"


def test_the_snapshot_holds_what_the_page_draws(ws):
    ctx, st = ws
    dispatch(ctx, "agent_plan", {"nodes": STEPS})
    dispatch(ctx, "agent_request", {"kind": "question", "reason": "Seeds?", "args": {"choices": ["3", "5"], "multi": True}})
    st.event("lead-1", "tool", tool="study_status", summary="study_status churn")
    s = dispatch(ctx, "show_snapshot", {})["result"]
    lead = next(x for x in s["sessions"] if x["id"] == "lead-1")
    assert lead["state"] == "waiting" and lead["now"]["op"] == "waiting for the person"   # its question is pending
    assert [p["id"] for p in s["people"]] == ["you", "sam"] and s["people"][1]["cap"] == 10
    nodes = {n["id"]: n for n in s["plans"]["lead-1"][-1]["nodes"]}
    assert nodes["emos"]["ev"] == "rep" and nodes["look"]["owner"] == "you"
    assert s["inbox"][0]["type"] == "choice" and s["inbox"][0]["multi"] and s["feed"][0]["who"] == "lead-1"


def test_the_server_serves_the_page_and_publishes_control_store_changes(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from openlab.app import server
    server.WATCH_SECONDS = 0.05
    Store(tmp_path).register("a-1", "spawned", status="running")
    with TestClient(server.create_app(str(tmp_path), token="tok")) as c:
        page = c.get("/show/")
        assert page.status_code == 200 and "live.js" in page.text
        assert c.get("/show/live.js").status_code == 200
        for sneaky in ("/show/..%2fserver.py", "/show/%2e%2e/server.py", "/show/..%2F__init__.py"):
            assert "def create_app" not in c.get(sneaky).text and "import" not in c.get(sneaky).text[:200]
        with c.websocket_connect("/ws/events?t=tok") as sock:
            assert sock.receive_json()["channel"] == "hello"
            time.sleep(0.2)
            Store(tmp_path).event("a-1", "note", summary="from the agent's own process")
            assert sock.receive_json()["channel"] == "control"


def test_a_release_is_the_code_folder_with_the_paper_when_it_builds(tmp_path):
    from openlab.app.server import workspaces
    from openlab.study.release import build
    from openlab.study.workspace import studies
    ctx = ToolContext(str(tmp_path))
    for name, args in [("study_new", {"folder": "s", "template": "static"}), ("study_materialise", {"folder": "s"}),
                       ("study_run", {"folder": "s"})]:
        assert dispatch(ctx, name, args)["ok"]
    from openlab import study
    r = build(study.open(tmp_path / "s"))
    assert r["paper"] == "none" and {"README.md", "code/model.py", "s.bundle.tar", "MANIFEST.json"} <= set(r["files"])
    assert studies(tmp_path) == [tmp_path / "s"]                      # release/code is not a second study
    # the release is never the study's code: a later result is not stale because of it
    assert dispatch(ctx, "study_run", {"folder": "s"})["result"].get("id")
    assert "README.md" in dispatch(ctx, "show_paper", {})["result"]["studies"][0]["release"]["files"]
    (tmp_path / "w2").mkdir(); (tmp_path / "w2" / "openlab.toml").write_text("")
    assert "w2" in workspaces(tmp_path)


def test_a_new_research_is_its_own_workspace_and_files_stay_inside(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from openlab.app import server
    H = {"authorization": "Bearer tok"}
    with TestClient(server.create_app(str(tmp_path), token="tok")) as c:
        assert c.post("/api/workspaces", headers=H, json={"name": "../x"}).status_code == 400
        env = c.post("/api/workspaces", headers=H, json={"name": "gbm", "prompt": ""}).json()
        assert env["ok"] and (tmp_path / "gbm" / "openlab.toml").is_file()
        assert [w["name"] for w in c.get("/api/workspaces", headers=H).json()["workspaces"]][-1] == "gbm"
        assert c.post("/api/workspaces", headers=H, json={"name": "gbm"}).status_code == 409
        assert c.post("/api/tool/show_snapshot?ws=gbm", headers=H).json()["result"]["workspace"].endswith("gbm")
        assert c.post("/api/tool/show_snapshot?ws=nope", headers=H).status_code == 404
        (tmp_path / "gbm" / "a.txt").write_text("A")
        assert c.get("/api/file?ws=gbm&path=a.txt&t=tok").text == "A"
        assert c.get("/api/file?ws=gbm&path=../openlab.toml&t=tok").status_code == 403
        assert c.get("/api/file?ws=gbm&path=a.txt").status_code == 401


def test_files_are_jailed_to_the_workspace_symlinks_and_the_control_store_too(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from openlab.app import server
    home, outside = tmp_path / "home", tmp_path / "outside"
    (home / "gbm").mkdir(parents=True); (home / "gbm" / "openlab.toml").write_text("")
    outside.mkdir(); (outside / "secret.txt").write_text("SECRET")
    (home / "gbm" / "a.txt").write_text("A")
    (home / "gbm" / "link.txt").symlink_to(outside / "secret.txt")
    (home / "gbm" / "dir").symlink_to(outside)
    (home / "gbm" / "passwd").symlink_to("/etc/passwd")
    (home / "gbm" / ".openlab").mkdir(); (home / "gbm" / ".openlab" / "control.db").write_text("DB")
    with TestClient(server.create_app(str(home), token="tok")) as c:
        assert c.get("/api/file?ws=gbm&path=a.txt&t=tok").text == "A"
        for bad in ("link.txt", "dir/secret.txt", "passwd", "/etc/passwd", "../../outside/secret.txt",
                    "..%2f..%2foutside%2fsecret.txt", ".openlab/control.db", "./.openlab/control.db"):
            r = c.get(f"/api/file?ws=gbm&path={bad}&t=tok")
            assert r.status_code in (403, 404) and "SECRET" not in r.text and "root:" not in r.text and "DB" != r.text
        assert c.get("/api/file?path=gbm/.openlab/control.db&t=tok").status_code == 403   # from home's own tab too
        for sneaky in ("/show/../server.py", "/show/%2e%2e/%2e%2e/registry.py", "/show//etc/passwd"):
            assert "root:" not in c.get(sneaky).text and "def " not in c.get(sneaky).text


def test_a_research_name_is_one_plain_folder_under_home(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from openlab.app import server
    home = tmp_path / "home"
    home.mkdir(); (home / "openlab.toml").write_text("")
    (home / "dangling").symlink_to(tmp_path / "nowhere")
    H = {"authorization": "Bearer tok"}
    with TestClient(server.create_app(str(home), token="tok")) as c:
        for bad in (".", "..", "a/b", "a\\b", "/abs", "", " ", ".hidden", "-x", "été", "x́", "a b", "a\nb",
                    "x" * 49, "home"):                                     # home's own name is the root, not a tab
            r = c.post("/api/workspaces", headers=H, json={"name": bad, "prompt": ""})
            assert r.status_code == 400 and r.json()["error"] == "bad name", bad
        for body in ([1], "gbm", None):
            r = c.post("/api/workspaces", headers=H, json=body)
            assert r.status_code == 400 and r.json()["try"]
        assert c.post("/api/workspaces", headers=H, content=b"{not json").status_code == 400
        assert c.post("/api/workspaces", headers=H, json={"name": "dangling"}).status_code == 409
        assert c.post("/api/workspaces", headers=H, json={"name": "x" * 48, "prompt": ""}).json()["ok"]
    assert sorted(p.name for p in home.iterdir()) == ["dangling", "openlab.toml", "x" * 48]


def test_a_tab_is_a_workspace_folder_now_never_a_cached_or_linked_one(tmp_path):
    pytest.importorskip("fastapi")
    import shutil
    from fastapi.testclient import TestClient
    from openlab.app import server
    home, elsewhere = tmp_path / "home", tmp_path / "elsewhere"
    for d in (home / "gbm", home / "plain", elsewhere):
        d.mkdir(parents=True)
    (home / "gbm" / "openlab.toml").write_text(""); (elsewhere / "openlab.toml").write_text("")
    (home / "linked").symlink_to(elsewhere)
    H = {"authorization": "Bearer tok"}
    with TestClient(server.create_app(str(home), token="tok")) as c:
        snap = lambda ws: c.post(f"/api/tool/show_snapshot?ws={ws}", headers=H)     # noqa: E731
        assert snap("gbm").json()["ok"]
        for bad in ("plain", "linked", "..", ".", "nope", "gbm/..", "%2e%2e"):
            assert snap(bad).status_code == 404, bad                          # not a workspace: refused, never made
        assert not (home / "plain" / ".openlab").exists() and not (elsewhere / ".openlab").exists()
        assert "linked" not in [w["name"] for w in c.get("/api/workspaces", headers=H).json()["workspaces"]]
        shutil.rmtree(home / "gbm")                                           # deleted after it was cached
        assert snap("gbm").status_code == 404
        assert c.get("/api/file?ws=gbm&path=openlab.toml&t=tok").status_code == 404
        assert not (home / "gbm").exists()                                    # the cached context did not remake it


def test_a_persons_act_needs_the_passphrase_in_every_workspace(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from openlab.app import server
    (tmp_path / "gbm").mkdir(); (tmp_path / "gbm" / "openlab.toml").write_text("")
    Store(tmp_path / "gbm").register("a-1", "spawned", status="paused")
    app = server.create_app(str(tmp_path), token="tok")
    H = {"authorization": "Bearer tok"}
    args = {"id": "a-1", "action": "resume", "reason": "go on"}
    with TestClient(app) as c:
        env = c.post("/api/tool/agent_control?ws=gbm", headers={**H, "x-openlab-reviewer": app.state.reviewer},
                     json=args).json()
        assert not env["ok"] and env["needs"] == "passphrase"                 # the reviewer secret alone is not a person
        env = c.post("/api/tool/agent_control?ws=gbm", headers=H, json=args).json()
        assert not env["ok"] and env["error"] == "a reviewer's act"           # no secret: not the person's at all
        env = c.post("/api/tool/agent_control?ws=gbm", headers={**H, "x-openlab-reviewer": "guess"}, json=args).json()
        assert not env["ok"]
    assert [a["status"] for a in Store(tmp_path / "gbm").agents() if a["id"] == "a-1"] == ["paused"]


def test_the_snapshot_and_the_paper_read_nothing_from_the_vault_or_outside(tmp_path, monkeypatch):
    import json
    from openlab.study.vault import vault_dir
    monkeypatch.setenv("OPENLAB_VAULT", str(tmp_path / "ws" / "vaults"))    # vaults inside the workspace: closed
    root = tmp_path / "ws"
    root.mkdir(); (root / "openlab.toml").write_text("")
    vault = vault_dir(root)
    vault.mkdir(parents=True); (vault / "scratch__h.json").write_text('{"labels": "HELDOUT-LABELS"}')
    (tmp_path / "outside.txt").write_text("OUTSIDE-TEXT")
    s = root / "s"
    (s / "paper" / "sections").mkdir(parents=True); (s / "release").mkdir()
    (s / "study.toml").write_text("")
    (s / "paper" / "paper.toml").write_text('sections = ["intro", "../../../outside"]\n')
    (s / "paper" / "sections" / "abstract.tex").symlink_to(vault / "scratch__h.json")
    (s / "paper" / "sections" / "intro.tex").symlink_to(tmp_path / "outside.txt")
    (s / "release" / "MANIFEST.json").write_text('{"paper": "none", "files": {}}')
    (s / "release" / "README.md").symlink_to(vault / "scratch__h.json")
    ctx = ToolContext(str(root))
    Store(root).register("a-1", "spawned", status="running")
    paper = dispatch(ctx, "show_paper", {})
    assert paper["ok"] and paper["result"]["studies"][0]["release"] is not None
    snap = dispatch(ctx, "show_snapshot", {})
    assert snap["ok"]
    for env in (paper, snap):
        text = json.dumps(env)
        assert "HELDOUT-LABELS" not in text and "OUTSIDE-TEXT" not in text
    assert not any(sec["exists"] for sec in paper["result"]["studies"][0]["sections"])


def test_a_lead_that_ends_done_gets_its_release_compiled(tmp_path):
    from openlab.app.show_tools import finished_leads
    ctx = ToolContext(str(tmp_path))
    for name, args in [("study_new", {"folder": "s", "template": "static"}), ("study_materialise", {"folder": "s"}),
                       ("study_run", {"folder": "s"})]:
        dispatch(ctx, name, args)
    st = Store(tmp_path)
    st.register("old", "spawned", role="lead", status="done", study="s")
    st.register("lead", "spawned", role="lead", status="running", study="s")
    seen = set()
    assert finished_leads(tmp_path, seen, prime=True) == [] and "old" in seen      # history is not recompiled
    st.set_status("lead", "done")
    jobs = finished_leads(tmp_path, seen)
    assert len(jobs) == 1 and finished_leads(tmp_path, seen) == []
    jobs[0]()
    assert (tmp_path / "s" / "release" / "README.md").is_file()
    assert "release compiled" in st.events("lead", n=1)[-1]["summary"]


def test_an_agent_waits_for_the_persons_answer(ws):
    ctx, st = ws
    item = dispatch(ctx, "agent_request", {"kind": "question", "reason": "Which data?", "args": {"choices": ["a", "b"]}})["result"]["item"]
    env = dispatch(ctx, "agent_wait", {"ids": [f"inbox:{item}"], "timeout": 0.5})["result"]
    assert not env["done"] and env["items"][0]["status"] == "pending" and env["next"]
    dispatch(ctx, "inbox_answer", {"id": item, "picks": ["b"], "message": "use b"}, human=True)
    env = dispatch(ctx, "agent_wait", {"ids": f"inbox:{item}"})["result"]
    assert env["done"] and env["items"][0]["result"]["picks"] == ["b"] and env["items"][0]["decision_reason"] == "use b"
    assert not dispatch(ctx, "agent_wait", {"ids": ["inbox:999"], "timeout": 0.1})["ok"]


def _steps(root, agent, k):
    st = Store(root)
    for i in range(k):
        P.step(st, agent, "data", "active", note=f"{i}")


def test_two_processes_append_without_tearing_a_line(ws):
    import multiprocessing as mp
    ctx, st = ws
    P.report(st, "lead-1", STEPS)
    procs = [mp.get_context("fork").Process(target=_steps, args=(st.root, "lead-1", 15)) for _ in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    lines = P._path(st, "lead-1").read_text().splitlines()
    assert len(lines) == 31 and all(isinstance(P.json.loads(x), dict) for x in lines)
    assert P.verify(st)["ok"]


def test_a_cut_last_line_is_skipped_and_the_next_append_starts_fresh(ws):
    ctx, st = ws
    P.report(st, "lead-1", STEPS)
    p = P._path(st, "lead-1")
    with open(p, "ab") as f:
        f.write('{"step":"emos","at":"x","state":"do'.encode() + "é".encode()[:1])   # killed mid-character
    assert len(P.read(st, "lead-1")["versions"]) == 1
    P.step(st, "lead-1", "emos", "done")
    v = P.read(st, "lead-1")["versions"][-1]
    assert next(n for n in v["nodes"] if n["id"] == "emos")["state"] == "done"
    for junk in ("[1, 2]", '{"nodes": "x"}', '{"step": "emos"}'):
        with open(p, "a") as f:
            f.write(junk + "\n")
    assert P.read(st, "lead-1")["versions"][-1]["v"] == 1
    got = P.verify(st)["agents"][0]
    assert not got["ok"] and got["corrupt"] == [2, 4] and got["unrecorded"] == [5, 6]


def test_verify_finds_an_edited_a_deleted_and_a_moved_line(ws):
    ctx, st = ws
    P.report(st, "lead-1", STEPS)
    P.step(st, "lead-1", "emos", "done", links=["run:abcd"])
    P.step(st, "lead-1", "look", "active")
    assert P.verify(st)["ok"] and dispatch(ctx, "show_verify", {})["result"]["plans"]["ok"]
    p = P._path(st, "lead-1")
    one, two, three = p.read_text().splitlines()
    p.write_text("\n".join([one, two.replace("done", "failed"), three]) + "\n")
    got = P.verify(st)["agents"][0]
    assert got["unrecorded"] == [2] and got["missing"] == [st.events("lead-1", kinds=("plan_node",))[0]["seq"]]
    p.write_text("\n".join([one, three]) + "\n")
    got = P.verify(st)["agents"][0]
    assert got["missing"] and not got["unrecorded"]
    p.write_text("\n".join([one, three, two]) + "\n")
    got = dispatch(ctx, "show_verify", {})["result"]["plans"]
    assert not got["ok"] and got["agents"][0]["reordered"] and not got["agents"][0]["missing"]


def test_a_step_is_checked_field_by_field():
    for bad, words in [([{"id": "a", "title": "A", "needs": ["a"]}], "needs itself"),
                       ([{"id": "a", "title": "A", "owner": ""}], "owner"),
                       ([{"id": "a", "title": "A", "owner": "x" * 41}], "owner"),
                       ([{"id": "a", "title": "A", "owner": "sam smith"}], "owner"),
                       ([{"id": "a", "title": "A", "owner": 7}], "owner"),
                       ([{"id": "a", "title": "A", "links": ["inbox:seven"]}], "inbox id"),
                       ([{"id": "a", "title": "A", "links": [f"run:{i:04x}" for i in range(13)]}], "at most"),
                       ([{"id": "a", "title": "‮\n\t"}], "no title")]:
        with pytest.raises(P.PlanError, match=words):
            P.check(bad)
    ok = P.check([{"id": "a", "title": "Café — 天気 \U0001f327️\nline two‮", "owner": "sam.k-1",
                   "needs": "b", "links": "run:abcd"}, {"id": "b", "title": "B"}])
    assert ok[0]["title"] == "Café — 天気 \U0001f327️ line two"
    assert ok[0]["needs"] == ["b"] and ok[0]["links"] == ["run:abcd"] and ok[0]["owner"] == "sam.k-1"


def test_unicode_titles_round_trip_and_long_ones_keep_the_sha_in_the_event(ws):
    ctx, st = ws
    nodes = [{"id": f"s{i}", "title": "天気予報" * 30} for i in range(60)]
    P.report(st, "lead-1", nodes)
    P.report(st, "lead-1", [dict(n, title="été " + n["title"]) for n in nodes])
    v = P.read(st, "lead-1")["versions"]
    assert v[-1]["nodes"][0]["title"].startswith("été 天") and len(v[-1]["nodes"][0]["title"]) == 120
    assert all(e["payload"]["sha"] for e in st.events("lead-1", kinds=("plan",)))
    assert P.verify(st)["ok"]


def test_agent_ids_with_a_slash_keep_their_own_plan(ws):
    ctx, st = ws
    agents = ("team/a", "team_a", "..", "a/../../b")
    for agent in agents:
        P.report(st, agent, [{"id": "x", "title": agent}])
    for agent in agents:
        assert P.read(st, agent)["versions"][0]["nodes"][0]["title"] == agent
        assert P._path(st, agent).resolve().parent.parent == (st.dir / "agents").resolve()
    assert P.verify(st)["ok"]


def test_evidence_says_ambiguous_foreign_odd_and_long_links(ws, tmp_path):
    from contextlib import closing
    ctx, st = ws
    for name, args in [("study_new", {"folder": "s", "template": "static"}), ("study_materialise", {"folder": "s"}),
                       ("study_run", {"folder": "s"}), ("study_new", {"folder": "t", "template": "static"}),
                       ("study_materialise", {"folder": "t"})]:
        assert dispatch(ctx, name, args)["ok"]
    rec = P._record(tmp_path, "s")
    with closing(rec._con()) as con:
        ids = [r["id"] for r in con.execute("SELECT id FROM nodes")]
    run = next(i for i in ids if i.startswith("run:"))
    ev, text = P.evidence(st, "s", {"state": "done", "links": [run[:-2]]}, rec)
    assert ev == "ok" and run in text                                     # a unique prefix resolves
    kinds: dict = {}
    for i in ids:
        kinds.setdefault(i.split(":")[0], []).append(i)
    kind = next(k for k, v in kinds.items() if len(v) > 1)
    ev, text = P.evidence(st, "s", {"state": "done", "links": [kind + ":"]}, rec)
    assert ev == "bad" and "ambiguous" in text and "not on the record" not in text
    ev, text = P.evidence(st, "t", {"state": "done", "links": [run]})
    assert ev == "ok" and run in text and "on study s's record" in text   # another study of the workspace counts
    ev, text = P.evidence(st, "", {"state": "done", "links": [run]})
    assert ev == "ok" and run in text                                     # a workspace lead rests on any study
    ev, text = P.evidence(st, "", {"state": "done", "links": ["study:s"]})
    assert ev == "ok" and "study s: its record holds" in text             # the study itself, once it has a record
    ev, text = P.evidence(st, "", {"state": "done", "links": ["study:nope"]})
    assert ev == "bad" and "no study of that name" in text
    for odd in ("inbox:seven", "inbox:99999999999999999999999", "inbox:٣"):
        ev, text = P.evidence(st, "", {"state": "done", "links": [odd]})
        assert ev == "bad" and "not an inbox id" in text
    ev, text = P.evidence(st, "s", {"state": "done", "links": [run] * 5000}, rec)
    assert ev == "bad" and "4988 links past the 12" in text


# --- the release, hardened: links, odd files, concurrent compiles, failures, the manifest -------------------------
def _ran(tmp_path, folder="s", run=True):
    from openlab import study
    ctx = ToolContext(str(tmp_path))
    steps = [("study_new", {"folder": folder, "template": "static"}), ("study_materialise", {"folder": folder})]
    for name, args in steps + ([("study_run", {"folder": folder})] if run else []):
        assert dispatch(ctx, name, args)["ok"], name
    return study.open(tmp_path / folder)


def test_a_release_never_follows_a_link_nor_ships_a_binary_or_huge_file(tmp_path, monkeypatch):
    import tarfile
    import openlab.study.release as R
    s = _ran(tmp_path)
    out = tmp_path / "outside"; out.mkdir(); (out / "secret.py").write_text("SECRET = 1\n")
    (s.folder / "metrics.py").symlink_to("/etc/passwd")
    (s.folder / "recipes" / "x.py").symlink_to(out / "secret.py")
    (s.folder / "recipes" / "dir").symlink_to(out, target_is_directory=True)
    (s.folder / "recipes" / "blob.bin").write_bytes(b"\x00\x01" * 10)
    (s.folder / "recipes" / "huge.py").write_text("#" * 200_000)
    monkeypatch.setattr(R, "MAX_BYTES", 100_000)
    r = R.build(s)
    rel = tmp_path / "s" / "release"
    assert set(r["skipped"]) == {"metrics.py", "recipes/x.py", "recipes/dir", "recipes/blob.bin", "recipes/huge.py"}
    assert not any(p.is_symlink() for p in rel.rglob("*")) and not (rel / "code" / "metrics.py").exists()
    readme = (rel / "README.md").read_text()
    assert "`recipes/x.py`: a symlink" in readme and "/etc/passwd" in readme and "a binary file" in readme
    with tarfile.open(rel / "s.bundle.tar") as t:                       # the bundle does not follow them either
        names = t.getnames()
    assert "model.py" in names and not {"metrics.py", "recipes/x.py", "recipes/dir/secret.py"} & set(names)
    assert R.verify(rel)["ok"]


def test_a_release_without_metrics_or_a_result_says_so(tmp_path):
    from openlab.study.release import build, verify
    s = _ran(tmp_path, run=False)
    assert not (s.folder / "metrics.py").exists()
    r = build(s)
    readme = (tmp_path / "s" / "release" / "README.md").read_text()
    assert "No result is on the record yet" in readme and "code/metrics.py" not in r["files"]
    assert "code/model.py" in r["files"] and verify(tmp_path / "s" / "release")["ok"]


def test_the_manifest_lists_every_file_and_verify_rechecks_it(tmp_path):
    import json
    from openlab.study.release import _sha, build, verify
    build(_ran(tmp_path))
    rel = tmp_path / "s" / "release"
    m = json.loads((rel / "MANIFEST.json").read_text())["files"]
    on_disk = {p.relative_to(rel).as_posix() for p in rel.rglob("*") if p.is_file()} - {"MANIFEST.json"}
    assert set(m) == on_disk and all(_sha(rel / f) == h for f, h in m.items())
    assert verify(rel) == {"ok": True, "missing": [], "changed": [], "extra": []}
    (rel / "code" / "model.py").write_text("tampered\n"); (rel / "extra.txt").write_text("x"); (rel / "README.md").unlink()
    v = verify(rel)
    assert not v["ok"] and v["changed"] == ["code/model.py"] and v["extra"] == ["extra.txt"] and v["missing"] == ["README.md"]
    assert not verify(tmp_path)["ok"]                                   # no manifest


def test_a_failed_compile_leaves_the_last_release_and_no_temp_behind(tmp_path, monkeypatch):
    from pathlib import Path
    import openlab.study.release as R
    s = _ran(tmp_path)
    R.build(s)
    rel = tmp_path / "s" / "release"
    before = (rel / "MANIFEST.json").read_text()

    def boom(*a, **k):
        raise RuntimeError("disk full")
    monkeypatch.setattr(R, "_files", boom)
    with pytest.raises(RuntimeError):
        R.build(s)
    monkeypatch.undo()
    assert (rel / "MANIFEST.json").read_text() == before and R.verify(rel)["ok"]
    assert [p.name for p in (tmp_path / "s" / ".release.tmp").iterdir()] == ["lock"]
    real = Path.rename

    def fail_swap(self, target):                                        # the swap itself fails: the old release returns
        if Path(target).name == "release" and self.name.startswith("new-"):
            raise OSError("rename failed")
        return real(self, target)
    monkeypatch.setattr(Path, "rename", fail_swap)
    with pytest.raises(OSError):
        R.build(s)
    monkeypatch.undo()
    assert (rel / "MANIFEST.json").read_text() == before and R.verify(rel)["ok"]
    assert [p.name for p in (tmp_path / "s" / ".release.tmp").iterdir()] == ["lock"]


def test_a_paper_that_fails_is_reported_and_the_code_still_compiles(tmp_path, monkeypatch):
    from openlab.study.errors import StudyError
    from openlab.study.release import build
    s = _ran(tmp_path)
    (s.folder / "paper").mkdir(); (s.folder / "paper" / "paper.toml").write_text("")

    def crash():
        raise KeyError("bindings")
    monkeypatch.setattr(s, "paper", crash)
    r = build(s)
    assert r["paper"] == "error" and "KeyError" in r["why"] and "code/model.py" in r["files"]
    assert "the paper build failed: KeyError" in (tmp_path / "s" / "release" / "README.md").read_text()

    def refuse():
        raise StudyError("paper refused", "a binding is stale")
    monkeypatch.setattr(s, "paper", refuse)
    monkeypatch.setattr(s, "paper_lint", lambda: {"findings": [{"rule": "stale", "message": "rerun"}]})
    r = build(s)
    assert r["paper"] == "refused" and r["why"] == "paper refused" and r["findings"] == ["stale: rerun"]


def test_two_compiles_at_once_leave_one_whole_release(tmp_path):
    import threading
    from openlab.study.release import build, verify
    s = _ran(tmp_path)
    errs = []

    def go():
        try:
            build(s)
        except Exception as e:                                          # noqa: BLE001
            errs.append(e)
    ts = [threading.Thread(target=go) for _ in range(3)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs and verify(tmp_path / "s" / "release")["ok"]
    assert [p.name for p in (tmp_path / "s" / ".release.tmp").iterdir()] == ["lock"]


def test_a_study_at_the_workspace_root_compiles_its_release(tmp_path):
    from openlab import study
    from openlab.study.release import build, verify
    _ran(tmp_path, folder=".", run=False)                              # (study_run refuses a study at the root)
    r = build(study.open(tmp_path))
    assert (tmp_path / "release" / "README.md").is_file() and f"{tmp_path.name}.bundle.tar" in r["files"]
    assert verify(tmp_path / "release")["ok"] and not (tmp_path / "release" / "release").exists()


def test_the_release_is_never_in_the_bundle_nor_the_code_fingerprint(tmp_path):
    import tarfile
    from openlab.study import fingerprint
    from openlab.study.release import build
    s = _ran(tmp_path)
    before = fingerprint.code(s.folder)
    build(s)
    (s.folder / ".release.tmp" / "new-x").mkdir()
    (s.folder / ".release.tmp" / "new-x" / "m.py").write_text("x = 1\n")
    assert fingerprint.code(s.folder) == before
    s.bundle(tmp_path / "b.tar")
    with tarfile.open(tmp_path / "b.tar") as t:
        assert not [n for n in t.getnames() if n.split("/")[0] in ("release", ".release.tmp")]


def test_a_closed_page_frees_its_event_socket(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from openlab.app import server
    app = server.create_app(str(tmp_path), token="tok")
    with TestClient(app) as c:
        with c.websocket_connect("/ws/events?t=tok") as sock:
            assert sock.receive_json()["channel"] == "hello"
            assert len(app.state.hub.queues) == 1
        for _ in range(50):                                 # the handler sees the close without waiting for an event
            if not app.state.hub.queues:
                break
            time.sleep(0.02)
        assert not app.state.hub.queues


def test_a_step_left_waiting_on_a_decided_inbox_item_is_drawn_as_decided():
    from openlab.app.show_tools import _decided
    items = {1: {"id": 1, "status": "denied", "decided_by": "person:p", "result": {}},
             2: {"id": 2, "status": "approved", "decided_by": "person:p", "result": {}},
             3: {"id": 3, "status": "pending", "decided_by": "", "result": {}},
             4: {"id": 4, "status": "approved", "decided_by": "person:p", "result": {"running": True}}}

    def drawn(state, links, evidence="inbox"):
        n = {"state": state, "links": links, "evidence": evidence}
        _decided(n, items)
        return n

    n = drawn("waiting", ["inbox:1"])
    assert (n["state"], n["said"]) == ("dropped", "waiting") and "inbox #1 denied by person:p" in n["evidence"]
    assert drawn("waiting", ["inbox:2", "result:abc"])["state"] == "done"
    assert drawn("waiting", ["inbox:2", "inbox:1"])["state"] == "dropped"      # one denial drops it
    for state, links in [("waiting", ["inbox:3"]), ("waiting", ["inbox:2", "inbox:3"]), ("waiting", ["inbox:4"]),
                         ("waiting", ["inbox:9"]), ("waiting", ["result:abc"]), ("active", ["inbox:1"])]:
        n = drawn(state, links)
        assert n["state"] == state and "said" not in n, (state, links)


def test_the_release_readme_reports_the_tested_result_not_a_later_check():
    from types import SimpleNamespace as NS
    from openlab.study.release import _result_lines
    m = lambda v: {"auc": {"value": v, "lo": v - 0.02, "hi": v + 0.02}}             # noqa: E731
    nodes = {"result:a": NS(id="result:a", payload={"metrics": m(0.905)}),
             "test:t": NS(id="test:t", payload={"baseline": "model:LogReg", "rejected": True, "p": 0.002})}
    rec = NS(find=lambda k: [nodes["test:t"]] if k == "test" else [], get=nodes.__getitem__,
             links=lambda i, rel=None: [("test:t", "from", "result:a")] if i == "test:t" else [])
    status = {"last_result": {"id": "result:z", "run": "run:z", "metrics": m(0.859)}}
    text = "\n".join(_result_lines(NS(record=rec), status))
    assert "result `result:a`, tested by `test:t` against model:LogReg: rejected the null (p = 0.002)" in text
    assert "auc: 0.905" in text and "0.859" not in text and "`result:z`, is a later run that no test names" in text
    rec.find = lambda k: []                                                         # no test: the latest result
    assert "\n".join(_result_lines(NS(record=rec), status)).startswith("- result `result:z`")


def test_the_ledger_carries_an_assumptions_breaks_for_raid(tmp_path):
    from openlab.app.show_tools import ledger
    ctx = ToolContext(str(tmp_path))
    for name, args in [("study_new", {"folder": "s", "template": "static"}), ("study_materialise", {"folder": "s"})]:
        assert dispatch(ctx, name, args)["ok"]
    res = dispatch(ctx, "study_run", {"folder": "s"})["result"]["id"]
    a = dispatch(ctx, "study_assume", {"folder": "s", "statement": "rows are independent", "rests_on": "the data card",
                                       "breaks_if": "duplicates cross folds"})["result"]["id"]
    row = lambda: next(e for e in ledger(tmp_path)["entries"] if e["id"] == a)     # noqa: E731
    assert row()["breaks"] == [] and row()["needed_by"] == 0
    env = dispatch(ctx, "study_break", {"folder": "s", "assumption": a, "reason": "duplicates found", "shown_by": [res]})
    assert env["ok"], env
    assert row()["breaks"] == [env["result"]["id"]]


def test_a_step_an_ended_agent_left_open_is_drawn_as_dropped():
    from openlab.app.show_tools import _ended
    a = {"id": "lead-1", "state": "done"}
    for state in ("planned", "active", "waiting"):
        n = {"id": "x", "state": state, "evidence": "run:1"}
        _ended(n, a)
        assert (n["state"], n["said"]) == ("dropped", state) and "lead-1 ended (done) with the step open" in n["evidence"]
    n = {"id": "y", "state": "done"}
    _ended(n, a)
    assert n["state"] == "done" and "said" not in n
    n = {"id": "z", "state": "active"}
    _ended(n, {"id": "lead-2", "state": "working"})                     # a live agent's word stands
    assert n["state"] == "active" and "said" not in n


def test_an_agent_at_its_limit_says_so_and_an_ended_one_says_what_it_used():
    from openlab.app.show_tools import _limits_used, _now_of, _limit
    a = {"id": "lead-1", "status": "paused", "updated": "2026-10-13T23:00:00+00:00", "last_event_at": None,
         "control_reason": "budget: 240 of 240 minutes; waiting for the person to raise its limit"}
    now = _now_of(None, {"agent": a, "state": "paused", "flags": [], "pending": 1, "last": None}, [])
    assert now["op"] == "paused at its limit: raise it in the inbox" and now["note"].startswith("240 of 240 minutes")
    ended = {"created": "2026-10-13T19:29:49+00:00", "updated": "2026-10-13T23:15:41+00:00",
             "budget": {"minutes": 240, "usd": 90, "turns": 200}, "spent": {"usd": 9.2, "turns": 175}}
    assert _limits_used(ended) == " at 226 of 240 min"
    assert _limits_used({**ended, "updated": "2026-10-13T20:00:00+00:00"}) == ""
    item = {"kind": "budget", "args": {"minutes": 480, "change": {"minutes": [240, 480]}, "hold": True}}
    assert _limit(item) == {"limit": {"key": "minutes", "old": 240, "new": 480, "hold": True}}


def test_a_waiting_step_with_nothing_filed_is_drawn_as_planned_and_refused_now(tmp_path):
    from openlab.agents import plan as P
    from openlab.agents.store import Store
    from openlab.app.show_tools import _unfiled
    n = {"id": "p4", "state": "waiting", "links": [], "evidence": "Not filing now"}
    _unfiled(n)
    assert (n["state"], n["said"]) == ("planned", "waiting") and "nothing is filed" in n["evidence"]
    n = {"id": "p5", "state": "waiting", "links": ["inbox:2"]}
    _unfiled(n)
    assert n["state"] == "waiting" and "said" not in n
    st = Store(tmp_path)
    st.register("lead-1", "spawned", role="lead")
    with pytest.raises(P.PlanError, match="waits on nothing it links"):
        P.report(st, "lead-1", [{"id": "a", "title": "Ask", "state": "waiting"}])
    P.report(st, "lead-1", [{"id": "a", "title": "Ask", "state": "planned"}])
    with pytest.raises(P.PlanError, match="waits on nothing it links") as e:
        P.step(st, "lead-1", "a", "waiting", note="later")
    assert "stays planned" in e.value.try_
    assert P.step(st, "lead-1", "a", "waiting", links=["inbox:1"])["state"] == "waiting"


def test_a_wrong_kind_or_state_names_the_valid_ones(ws):
    ctx, st = ws
    env = dispatch(ctx, "agent_plan", {"nodes": [{"id": "lit", "title": "Read", "kind": "survey"}]})
    assert not env["ok"] and "survey" in env["error"] and "review, data, run" in env["try"]
    env = dispatch(ctx, "agent_plan", {"nodes": [{"id": "lit", "title": "Read", "kind": "review", "state": "doing"}]})
    assert not env["ok"] and "planned, active, waiting" in env["try"]
    assert dispatch(ctx, "agent_plan", {"nodes": STEPS})["ok"]
    env = dispatch(ctx, "agent_plan_node", {"id": "emos", "state": "finished"})
    assert not env["ok"] and "planned, active, waiting" in env["try"]


def test_an_agents_refusals_and_bypass_attempts_are_counted_by_kind(ws):
    ctx, st = ws
    for _ in range(5):
        st.event("lead-1", "refusal", tool="Read", summary="Read .openlab/record.jsonl: read the record through the tools")
    st.event("lead-1", "bypass_attempt", tool="Bash", summary="Bash echo x > .openlab/record.jsonl: a write into .openlab/")
    s = dispatch(ctx, "show_snapshot", {})["result"]
    lead = next(x for x in s["sessions"] if x["id"] == "lead-1")
    assert lead["refused"] == 5 and lead["bypass"] == 1
    assert [x["kind"] for x in lead["attempts"]] == ["refusal"] * 5 + ["bypass_attempt"]   # oldest first
    assert lead["attempts"][-1]["text"].startswith("Bash echo") and lead["attempts"][0]["at"]
    assert len(lead["refusals"]) == 5 and "BYPASS" in lead["flags"] and any(f.startswith("LOOP") for f in lead["flags"])
    for _ in range(10):
        st.event("lead-1", "refusal", tool="Read", summary="again")
    lead = next(x for x in dispatch(ctx, "show_snapshot", {})["result"]["sessions"] if x["id"] == "lead-1")
    assert lead["refused"] == 15 and len(lead["attempts"]) == 8                # the count is all of them; the list the last 8
