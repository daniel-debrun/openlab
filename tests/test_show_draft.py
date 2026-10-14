"""The paper draft in openlab show: the person edits the LaTeX prose and comments on it; the agents hear both."""

import json

import pytest

from openlab.agents.store import Store
from openlab.app import ToolContext, dispatch

TOML = 'title = "T"\nauthors = ["A"]\nabstract = "sections/abstract.tex"\nsections = ["intro", "results"]\n'
INTRO = "We ask a question.\nThe forest ranks good wines better.\nThat is all.\n"


@pytest.fixture
def ws(tmp_path, monkeypatch):
    (tmp_path / "openlab.toml").write_text("")
    st = Store(tmp_path)
    st.register("lead-1", "spawned", role="lead", status="running", study="s")
    st.register("other", "spawned", role="lead", status="running", study="t")
    st.register("gone", "spawned", role="lead", status="done", study="s")
    monkeypatch.setenv("OPENLAB_AGENT", "lead-1")
    ctx = ToolContext(str(tmp_path))
    assert dispatch(ctx, "study_new", {"folder": "s", "template": "static"})["ok"]
    (tmp_path / "s" / "paper" / "sections").mkdir(parents=True)
    (tmp_path / "s" / "paper" / "paper.toml").write_text(TOML)
    (tmp_path / "s" / "paper" / "sections" / "intro.tex").write_text(INTRO)
    return ctx, st


def person(ctx, name, args):
    return dispatch(ctx, name, args, human=True)


def told(st, agent):
    return [e["payload"]["text"] for e in st.untold(agent)]


def test_the_draft_lists_the_papers_files_with_their_text_and_sha(ws):
    ctx, _ = ws
    env = dispatch(ctx, "show_draft", {"study": "s"})
    assert env["ok"], env
    files = {f["file"]: f for f in env["result"]["files"]}
    assert list(files) == ["paper/paper.toml", "paper/sections/abstract.tex", "paper/sections/intro.tex",
                           "paper/sections/results.tex"]
    assert files["paper/sections/intro.tex"]["text"] == INTRO and files["paper/sections/intro.tex"]["sha"]
    assert not files["paper/sections/results.tex"]["exists"] and files["paper/sections/results.tex"]["sha"] == ""
    assert not dispatch(ctx, "show_draft", {"study": "nope"})["ok"]


def test_a_person_saves_prose_only_over_the_text_they_opened(ws, tmp_path):
    ctx, st = ws
    f = dispatch(ctx, "show_draft", {"study": "s"})["result"]["files"][2]
    args = {"study": "s", "file": f["file"], "text": INTRO + "More.\n", "sha": f["sha"]}
    assert not dispatch(ctx, "show_draft_save", args)["ok"]                  # an agent: not through this tool
    env = person(ctx, "show_draft_save", args)
    assert env["ok"], env
    assert (tmp_path / "s" / f["file"]).read_text() == INTRO + "More.\n" and env["result"]["told"] == ["lead-1"]
    assert "edited paper/sections/intro.tex" in told(st, "lead-1")[0] and not told(st, "other")
    assert st.events(None, kinds=("draft_edit",))[-1]["payload"]["file"] == f["file"]
    stale = person(ctx, "show_draft_save", {**args, "text": "edited"})            # the sha it opened is old now
    assert not stale["ok"] and "changed since you opened it" in stale["error"]
    new = person(ctx, "show_draft_save", {"study": "s", "file": "paper/sections/results.tex", "text": "R.\n", "sha": ""})
    assert new["ok"] and (tmp_path / "s" / "paper" / "sections" / "results.tex").read_text() == "R.\n"
    for file in ("model.py", "paper/sections/../../model.py", "paper/sections/other.tex", "/etc/passwd"):
        env = person(ctx, "show_draft_save", {"study": "s", "file": file, "text": "x", "sha": ""})
        assert not env["ok"] and "not a file of the paper" in env["error"], file
    model = (tmp_path / "s" / "model.py").read_text()
    (tmp_path / "s" / "paper" / "sections" / "abstract.tex").symlink_to(tmp_path / "s" / "model.py")
    env = person(ctx, "show_draft_save", {"study": "s", "file": "paper/sections/abstract.tex", "text": "x", "sha": ""})
    assert not env["ok"] and (tmp_path / "s" / "model.py").read_text() == model     # a link is never written through


def test_a_comment_reaches_the_studys_working_agents_who_answer_and_resolve_it(ws, tmp_path):
    ctx, st = ws
    quote = "The forest ranks good wines better."
    args = {"study": "s", "file": "paper/sections/intro.tex", "text": "Say by how much.", "quote": quote,
            "start": INTRO.index(quote), "end": INTRO.index(quote) + len(quote)}
    assert not dispatch(ctx, "paper_comment", args)["ok"]                    # an agent cannot comment as the person
    env = person(ctx, "paper_comment", args)
    assert env["ok"], env
    c = env["result"]
    assert c["id"] == 1 and c["told"] == ["lead-1"] and not c["resolved"]
    msg = told(st, "lead-1")[0]
    assert "comment #1" in msg and quote in msg and "agent_comment_reply(1" in msg
    assert not told(st, "other") and not told(st, "gone")

    got = dispatch(ctx, "agent_comments", {})["result"]                       # the agent's own study by default
    assert got["open"] == 1 and got["comments"][0]["line"] == 2
    intro = tmp_path / "s" / "paper" / "sections" / "intro.tex"
    intro.write_text("Moved down.\n\n" + INTRO)
    assert dispatch(ctx, "agent_comments", {})["result"]["comments"][0]["line"] == 4      # found where it is now
    intro.write_text("Rewritten entirely.\n")
    assert dispatch(ctx, "agent_comments", {})["result"]["comments"][0]["moved"]

    env = dispatch(ctx, "agent_comment_reply", {"id": 1, "text": "Added the difference and its interval.",
                                                "resolve": True})
    assert env["ok"] and env["result"]["resolved"] and env["result"]["resolved_by"] == "lead-1"
    assert [r["by"] for r in env["result"]["replies"]] == ["lead-1", "lead-1"]
    assert dispatch(ctx, "agent_comments", {})["result"]["open"] == 0
    assert dispatch(ctx, "agent_comments", {"all": True})["result"]["comments"][0]["resolved"]

    n = len(told(st, "lead-1"))
    env = person(ctx, "paper_comment_reply", {"id": 1, "text": "Not the interval intended.", "resolve": False})
    assert env["ok"] and not env["result"]["resolved"]
    assert len(told(st, "lead-1")) == n + 1 and "Not the interval intended." in told(st, "lead-1")[-1]
    assert not dispatch(ctx, "paper_comment_reply", {"id": 1, "text": "as the person"})["ok"]
    assert not dispatch(ctx, "agent_comment_reply", {"id": 9, "text": "x"})["ok"]

    log = tmp_path / ".openlab" / "comments.jsonl"
    log.write_text(log.read_text() + "not json\n" + json.dumps({"op": "reply", "id": "x"}) + "\n")
    shown = dispatch(ctx, "show_draft", {"study": "s"})["result"]["comments"]
    assert [c["id"] for c in shown] == [1] and len(shown[0]["replies"]) == 4
    assert {e["payload"]["op"] for e in st.events(None, n=100, kinds=("comment",))} == {"new", "reply", "resolve",
                                                                                        "reopen"}


def test_the_page_comments_and_saves_with_the_reviewer_secret_and_no_passphrase(ws, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from openlab.app import server
    app = server.create_app(str(tmp_path), token="tok")
    H = {"authorization": "Bearer tok"}
    args = {"study": "s", "file": "paper/sections/intro.tex", "text": "A note.", "quote": "That is all."}
    with TestClient(app) as c:
        assert not c.post("/api/tool/paper_comment", headers=H, json=args).json()["ok"]        # no reviewer secret
        env = c.post("/api/tool/paper_comment", headers={**H, "x-openlab-reviewer": app.state.reviewer}, json=args).json()
        assert env["ok"], env
        assert env["result"]["by"]


def test_a_person_starts_the_paper_as_one_document_and_the_draft_is_that_one_file(ws, tmp_path):
    ctx, st = ws
    assert not dispatch(ctx, "show_draft_init", {"study": "s"})["ok"]         # an agent: study_paper_init instead
    env = person(ctx, "show_draft_init", {"study": "s"})
    assert env["ok"], env
    assert env["result"]["carried_over"] == ["paper/sections/intro.tex"] and env["result"]["told"] == ["lead-1"]
    text = (tmp_path / "s" / "paper" / "paper.tex").read_text()
    assert "\\section{Intro}\nWe ask a question." in text and "\\begin{abstract}" in text
    files = dispatch(ctx, "show_draft", {"study": "s"})["result"]["files"]
    assert [(f["file"], f["name"]) for f in files] == [("paper/paper.tex", "paper")]
    papers = dispatch(ctx, "show_paper", {})["result"]["studies"]
    assert [p["sections"][0]["name"] for p in papers if p["study"] == "s"] == ["paper"]
    assert not person(ctx, "show_draft_init", {"study": "s"})["ok"]           # once
    f = files[0]
    env = person(ctx, "show_draft_save", {"study": "s", "file": f["file"], "text": f["text"] + "%\n", "sha": f["sha"]})
    assert env["ok"], env
    env = dispatch(ctx, "study_paper_init", {"folder": "s"})
    assert not env["ok"] and "exists" in env["error"]


def test_the_readme_opens_in_the_editor_as_the_last_compile_wrote_it(ws, tmp_path):
    ctx, st = ws
    env = dispatch(ctx, "show_draft", {"study": "s", "doc": "readme"})
    assert env["ok"], env
    r = env["result"]
    mark, end = r["marks"]
    f = r["files"][0]
    assert f["file"] == "README.md" and not f["exists"] and f["sha"] == ""
    assert f["text"].startswith("# ") and mark in f["text"] and r["record"] == ""      # not compiled yet
    assert not dispatch(ctx, "show_draft", {"study": "s", "doc": "notes"})["ok"]
    assert dispatch(ctx, "show_compile", {"study": "s"})["ok"]
    r = dispatch(ctx, "show_draft", {"study": "s", "doc": "readme"})["result"]
    text = r["files"][0]["text"]                                               # the compiled README, whole
    assert r["record"].startswith("## Data") and r["record"] in text
    assert text == (tmp_path / "s" / "release" / "README.md").read_text()
    edited = text.replace(mark, "Plain words.\n\n" + mark)
    saved = person(ctx, "show_draft_save", {"study": "s", "file": "README.md", "text": edited, "sha": ""})
    assert saved["ok"], saved
    assert any("README's prose" in t for t in told(st, "lead-1"))
    assert person(ctx, "paper_comment", {"study": "s", "file": "README.md", "text": "say more", "quote": "Plain",
                                         "start": edited.index("Plain"), "end": edited.index("Plain") + 5})["ok"]
    assert dispatch(ctx, "show_compile", {"study": "s"})["ok"]
    r = dispatch(ctx, "show_draft", {"study": "s", "doc": "readme"})["result"]
    assert r["files"][0]["exists"] and [t["file"] for t in r["comments"]] == ["README.md"]
    assert all(t["file"] != "README.md" for t in dispatch(ctx, "show_draft", {"study": "s"})["result"]["comments"])
    out = (tmp_path / "s" / "release" / "README.md").read_text()
    assert out.count(mark) == 1 and out.index("Plain words.") < out.index(mark)
