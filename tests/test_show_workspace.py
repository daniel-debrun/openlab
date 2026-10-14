"""openlab show's Workspace: the research's files as a tree, each opened (prose edited), and a comment on any file
reaching the live lead with its path, the quoted text and what the person wrote."""

import pytest

from openlab.agents.store import Store
from openlab.app import ToolContext, dispatch


@pytest.fixture
def ws(tmp_path, monkeypatch):
    (tmp_path / "openlab.toml").write_text("")
    st = Store(tmp_path)
    st.register("lead-1", "spawned", role="lead", status="running", study="s")
    monkeypatch.setenv("OPENLAB_AGENT", "lead-1")
    ctx = ToolContext(str(tmp_path))
    assert dispatch(ctx, "study_new", {"folder": "s", "template": "static"})["ok"]
    (tmp_path / "s" / "paper").mkdir()
    (tmp_path / "s" / "paper" / "paper.tex").write_text("\\documentclass{article}\n\\begin{document}\nHello.\n\\end{document}\n")
    (tmp_path / "s" / "release").mkdir()
    (tmp_path / "s" / "release" / "README.md").write_text("# s\n")
    (tmp_path / "s" / "fig.png").write_bytes(b"\x89PNG\r\n\x1a\n\0\0")
    (tmp_path / "NOTES.md").write_text("Top notes.\n")
    (tmp_path / ".hidden").write_text("x")
    return ctx, st, tmp_path


def person(ctx, name, args):
    return dispatch(ctx, name, args, human=True)


def told(st, agent):
    return [e["payload"]["text"] for e in st.untold(agent)]


def test_the_tree_lists_the_research_without_openlabs_own_or_hidden_files(ws):
    ctx, _, _ = ws
    root = dispatch(ctx, "show_tree", {})["result"]
    names = {e["name"]: e for e in root["entries"]}
    assert "s" in names and names["s"]["dir"] and names["s"]["study"] and "NOTES.md" in names
    assert ".openlab" not in names and ".hidden" not in names and names["NOTES.md"]["kind"] == "md"
    s = dispatch(ctx, "show_tree", {"path": "s"})["result"]
    by = {e["name"]: e for e in s["entries"]}
    assert by["README.md"]["virtual"] and by["fig.png"]["kind"] == "image" and s["study"] == "s"
    assert not dispatch(ctx, "show_tree", {"path": ".openlab"})["ok"]
    assert not dispatch(ctx, "show_tree", {"path": "../"})["ok"]


def test_a_file_opens_with_its_kind_and_whether_a_person_edits_it_here(ws):
    ctx, _, _ = ws
    tex = dispatch(ctx, "show_open", {"path": "s/paper/paper.tex"})["result"]
    assert tex["kind"] == "tex" and tex["editable"] and tex["paper"] and tex["study"] == "s" and "Hello." in tex["text"]
    assert tex["file"] == "paper/paper.tex"
    rel = dispatch(ctx, "show_open", {"path": "s/release/README.md"})["result"]
    assert rel["kind"] == "md" and not rel["editable"]                       # rewritten at each compile
    readme = dispatch(ctx, "show_open", {"path": "s/README.md"})["result"]
    assert readme["readme"] and not readme["exists"] and readme["editable"] and readme["marks"][0] in readme["text"]
    png = dispatch(ctx, "show_open", {"path": "s/fig.png"})["result"]
    assert png["kind"] == "image" and png["text"] is None and not png["editable"]
    model = dispatch(ctx, "show_open", {"path": "s/model.py"})["result"]
    assert model["kind"] == "text" and not model["editable"] and model["text"]
    notes = dispatch(ctx, "show_open", {"path": "NOTES.md"})["result"]
    assert notes["study"] is None and notes["editable"]
    for bad in (".openlab/control.db", "../x", "s/nope.md", "s"):
        assert not dispatch(ctx, "show_open", {"path": bad})["ok"], bad


def test_a_person_saves_prose_over_the_text_they_opened_and_the_lead_hears(ws):
    ctx, st, tmp = ws
    tex = dispatch(ctx, "show_open", {"path": "s/paper/paper.tex"})["result"]
    assert not dispatch(ctx, "show_save", {"path": "s/paper/paper.tex", "text": "x", "sha": tex["sha"]})["ok"]  # an agent
    env = person(ctx, "show_save", {"path": "s/paper/paper.tex", "text": tex["text"] + "% more\n", "sha": tex["sha"]})
    assert env["ok"], env
    assert any("edited s/paper/paper.tex" in t for t in told(st, "lead-1"))
    assert not person(ctx, "show_save", {"path": "s/paper/paper.tex", "text": "y", "sha": tex["sha"]})["ok"]   # changed
    assert not person(ctx, "show_save", {"path": "s/model.py", "text": "y", "sha": ""})["ok"]                  # code
    assert not person(ctx, "show_save", {"path": "s/release/README.md", "text": "y", "sha": ""})["ok"]
    assert person(ctx, "show_save", {"path": "s/README.md", "text": "Mine.\n", "sha": ""})["ok"]
    assert (tmp / "s" / "README.md").read_text() == "Mine.\n"


def test_a_comment_on_text_or_on_a_whole_file_reaches_the_lead_with_path_quote_and_comment(ws):
    ctx, st, _ = ws
    assert not dispatch(ctx, "show_comment", {"path": "s/paper/paper.tex", "text": "x"})["ok"]              # an agent
    env = person(ctx, "show_comment", {"path": "s/paper/paper.tex", "text": "Say more.", "quote": "Hello.",
                                       "start": 42, "end": 48})
    assert env["ok"], env
    msg = told(st, "lead-1")[-1]
    assert "s/paper/paper.tex" in msg and "> Hello." in msg and "Their comment: Say more." in msg
    assert "agent_comment_reply(1" in msg and env["result"]["told"] == ["lead-1"]
    env = person(ctx, "show_comment", {"path": "s/fig.png", "text": "Label the axes."})
    assert env["ok"], env
    msg = told(st, "lead-1")[-1]
    assert "s/fig.png (the whole file)" in msg and "Label the axes." in msg
    got = dispatch(ctx, "agent_comments", {})["result"]
    assert [c["path"] for c in got["comments"]] == ["s/paper/paper.tex", "s/fig.png"]
    assert dispatch(ctx, "show_open", {"path": "s/fig.png"})["result"]["comments"][0]["text"] == "Label the axes."
    assert dispatch(ctx, "show_tree", {})["result"]["open"] == ["s/fig.png", "s/paper/paper.tex"]
    assert not person(ctx, "show_comment", {"path": "s/none.txt", "text": "x"})["ok"]


def test_a_comment_on_the_researchs_own_file_goes_to_every_live_lead_and_waits_when_none_is(ws):
    ctx, st, _ = ws
    st.register("lead-2", "spawned", role="lead", status="running", study="t")
    env = person(ctx, "show_comment", {"path": "NOTES.md", "text": "Update these."})
    assert sorted(env["result"]["told"]) == ["lead-1", "lead-2"]
    env = person(ctx, "show_comment", {"path": "s/paper/paper.tex", "text": "Only s."})
    assert env["result"]["told"] == ["lead-1"]                             # its study has a live lead
    st.set_status("lead-1", "done")
    env = person(ctx, "show_comment", {"path": "s/paper/paper.tex", "text": "Anyone."})
    assert env["result"]["told"] == ["lead-2"]                             # none of its own: every live lead
    st.set_status("lead-2", "done")
    env = person(ctx, "show_comment", {"path": "NOTES.md", "text": "Later."})
    assert env["ok"] and env["result"]["told"] == [] and not env["result"]["resolved"]   # waits, open
