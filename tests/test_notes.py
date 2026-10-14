r"""The scratchpad: `NOTES.md` in the workspace root, and the two tools over it.

An idea arrives while you are reading a chart, and anything that makes you leave the chart to record it loses
it. What is pinned here is the part of that answer a later change could quietly break:

* a note is **appended**, and nothing is read first. `ws_write` nearly does this and is the wrong tool for one
  reason: it writes whole files, so appending would mean `ws_read` then `ws_write`, and `ws_read` is a window
  -- the "read truncated, write back, lose the tail" failure the run-log commit documented;
* the reply is **bounded and counted**, like every other reply on this surface;
* and the file is **the user's own markdown**, so the parser is written for a file somebody edited by hand:
  prose, a heading, a reordered line, a missing date, a dash typed as `--`. Nothing is dropped for being
  malformed, because a note you can still read is not a parse error.
"""
import pathlib

import pytest

from openlab.app import ToolContext, dispatch


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    c = ToolContext(str(tmp_path))
    (tmp_path / "churn.openlab").write_text("x = 1\n", encoding="utf-8")
    yield c
    c.close()


def ok(env):
    assert env["ok"], env
    return env["result"]


def test_a_note_is_appended_and_nothing_is_read_first(ctx, tmp_path, monkeypatch):
    """The whole reason `notes_add` exists rather than a `ws_read` + `ws_write` pair."""
    p = tmp_path / "NOTES.md"
    before = "# Notes\n\n- [ ] 2026-09-01 12:00 — an older idea\n"
    p.write_text(before, encoding="utf-8")

    # If this tool read the file in order to write it, the tail of a long scratchpad would be at the mercy of
    # whatever window the read used. It does not, so a read here is a failure.
    def no(*a, **kw):
        raise AssertionError("notes_add read the file it was appending to")

    monkeypatch.setattr(pathlib.Path, "read_text", no)
    r = ok(dispatch(ctx, "notes_add", {"text": "try a deeper tree", "doc": "churn.openlab", "tag": "#next"}))
    monkeypatch.undo()

    after = p.read_text(encoding="utf-8")
    assert after.startswith(before)                      # nothing that was there was rewritten
    added = after[len(before):].strip()
    assert added == r["markdown"]
    assert added.startswith("- [ ] ") and added.endswith("— try a deeper tree")
    assert "· churn.openlab" in added and "· #next" in added
    assert r["path"] == "NOTES.md"                        # the root, not under .data: it is theirs to grep

    # A file whose last line lost its newline must not take the next note onto the end of it.
    p.write_text(after.rstrip("\n"), encoding="utf-8")
    ok(dispatch(ctx, "notes_add", {"text": "and another"}))
    assert len([l for l in p.read_text(encoding="utf-8").splitlines() if l.startswith("- [")]) == 3


def test_notes_list_filters_and_is_bounded(ctx):
    """`doc`, `tag`, `open_only` -- and a ceiling, whatever `last` asked for."""
    import openlab.app.tools as T

    for i in range(4):
        ok(dispatch(ctx, "notes_add", {"text": f"idea {i}", "doc": "churn.openlab" if i % 2 else "", "tag": "next" if i == 1 else ""}))
    r = ok(dispatch(ctx, "notes_list", {}))
    assert [n["text"] for n in r["notes"]][0] == "idea 3"         # newest first
    assert (r["total"], r["open"], r["done"]) == (4, 4, 0)
    assert r["notes"][0]["line"] > 0                               # addressable: what an edit to the file needs
    # `openable` is the document still being there, which is what the promote affordance turns on
    assert [n["openable"] for n in r["notes"]] == [True, False, True, False]

    assert [n["text"] for n in ok(dispatch(ctx, "notes_list", {"doc": "churn.openlab"}))["notes"]] == ["idea 3", "idea 1"]
    assert [n["text"] for n in ok(dispatch(ctx, "notes_list", {"tag": "#next"}))["notes"]] == ["idea 1"]

    two = ok(dispatch(ctx, "notes_list", {"last": 2}))
    assert len(two["notes"]) == 2 and two["notes_withheld"] == 2
    assert two["total"] == 4                                      # counted before `last` narrowed anything
    assert len(ok(dispatch(ctx, "notes_list", {"last": 10_000}))["notes"]) <= T.NOTES_MAX

    # done is plain text in the file, so it is an edit to the file -- and `open_only` drops it
    text = (ctx.root / "NOTES.md").read_text(encoding="utf-8").replace("- [ ] ", "- [x] ", 1)
    (ctx.root / "NOTES.md").write_text(text, encoding="utf-8")
    r = ok(dispatch(ctx, "notes_list", {"open_only": True}))
    assert (r["open"], r["done"]) == (3, 1) and len(r["notes"]) == 3

    for bad in ({"text": "  "}, {"text": "x" * (T.NOTE_CHARS + 1)}, {"text": "ok", "tag": "two words"}):
        assert not dispatch(ctx, "notes_add", bad)["ok"]
    assert ok(dispatch(ctx, "notes_add", {"text": "a\npasted\nparagraph"}))["text"] == "a pasted paragraph"


def test_a_hand_edited_notes_file_still_parses(ctx):
    """The file is markdown somebody writes in: what is not a note is left alone, and what is one survives
    having been typed rather than generated."""
    (ctx.root / "NOTES.md").write_text(
        "# Notes\n"
        "\n"
        "Some prose about this file, with a - dash - in it.\n"
        "## a heading\n"
        "- [ ] 2026-09-25 16:44 · churn.openlab · #next — the generated shape\n"
        "* [x] 2026-09-24 09:12 · trial2/model.openlab -- an ascii dash and a star bullet\n"
        "  - [ ] 2026-09-23 indented, dated, no separator at all\n"
        "- [ ] no date, no document, just the words — and a dash inside them\n"
        "- [x] done\n",
        encoding="utf-8")
    rows = ok(dispatch(ctx, "notes_list", {}))
    assert rows["total"] == 5 and (rows["open"], rows["done"]) == (3, 2)      # prose and headings are not notes
    by_line = {n["line"]: n for n in rows["notes"]}

    assert by_line[5]["text"] == "the generated shape"
    assert (by_line[5]["at"], by_line[5]["doc"], by_line[5]["tag"]) == ("2026-09-25 16:44", "churn.openlab", "next")

    assert by_line[6]["done"] and by_line[6]["doc"] == "trial2/model.openlab"
    assert by_line[6]["text"] == "an ascii dash and a star bullet"

    assert by_line[7]["at"] == "2026-09-23" and by_line[7]["text"] == "indented, dated, no separator at all"

    # No head to read, so every word after the box is the note -- including the dash somebody wrote mid-sentence
    assert by_line[8]["text"] == "no date, no document, just the words — and a dash inside them"
    assert by_line[8]["at"] == "" and by_line[8]["doc"] == ""
    assert by_line[9] == {"at": "", "doc": "", "tag": "", "text": "done", "done": True, "line": 9,
                          "openable": False}
