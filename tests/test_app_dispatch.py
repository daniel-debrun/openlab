"""openlab.app: the one dispatch every skin shares."""

import pytest

from openlab.app import TOOLS, ToolContext, dispatch


@pytest.fixture
def ctx(tmp_path):
    c = ToolContext(str(tmp_path))
    yield c
    c.close()


def ok(env):
    assert env["ok"], env
    return env["result"]


def test_tools_sections_and_version_handshake(ctx):
    for name, sec in {"paper_search": "research", "agent_spawn": "research", "study_run": "study",
                      "ws_list": "workspace", "agent_activity": "meta"}.items():
        assert TOOLS[name].section == sec, name
    info = ok(dispatch(ctx, "tools", {"section": "study"}))["openlab"]
    assert set(info) == {"version", "commit", "toolset"} and len(info["toolset"]) == 12


def test_mcp_says_restart_when_the_installed_tool_set_changes(monkeypatch):
    pytest.importorskip("mcp")
    from openlab import mcp_server as M
    assert M._STARTUP["toolset"] and M._restart_hint() is None
    monkeypatch.setitem(M._STARTUP, "stamp", ())                # sources changed on disk...
    monkeypatch.setattr(M, "_installed_toolset", lambda: "000000000000")    # ...and the tool set with them
    hint = M._restart_hint()
    assert "openlab tool NAME key=value" in hint and "/mcp" not in hint      # advice a headless agent can act on
    monkeypatch.setitem(M._STARTUP, "notice", None)
    shown = [M._notice() for _ in range(2 * M.NOTICE_EVERY + 1)]
    assert shown[0] == hint and shown.count(hint) == 3                   # once, then once per NOTICE_EVERY results


def test_refusal_hints_name_the_right_next_step(tmp_path):
    from openlab.app.registry import _retry_hint
    assert _retry_hint("bad x") == "fix the input and call again"
    assert "install" in _retry_hint("No module named 'mlflow'") and "pynacl" in _retry_hint("No module named 'nacl'")
    from openlab.app.server import write_mcp_json
    import json
    import sys
    f = write_mcp_json(tmp_path)
    assert json.loads(f.read_text())["mcpServers"]["openlab"] == {"command": sys.executable, "args": ["-m", "openlab.mcp_server"]}
    assert write_mcp_json(tmp_path) is None                     # never overwrites the user's file


def test_every_tool_has_a_section_a_description_and_a_schema():
    assert len(TOOLS) >= 20
    for t in TOOLS.values():
        rec = t.to_record()
        assert rec["description"] and rec["section"] and rec["input_schema"]["type"] == "object", t.name
    assert TOOLS["ws_delete"].schema["properties"]["confirm"]["type"] == "boolean"
    assert TOOLS["ws_read"].schema["required"] == ["path"]


def test_the_envelope_explains_refusals(ctx):
    assert dispatch(ctx, "nope")["error"] == "unknown tool"
    env = dispatch(ctx, "ws_read", {"path": "../etc/passwd"})
    assert not env["ok"] and env["error"] == "outside the workspace" and env["try"]
    env = dispatch(ctx, "ws_read", {"pth": "x"})
    assert env["error"] == "unknown argument" and "path" in env["try"]
    ok(dispatch(ctx, "ws_write", {"path": "a.txt", "text": "hi"}))
    env = dispatch(ctx, "ws_delete", {"path": "a.txt"})
    assert env["error"] == "confirm" and (ctx.root / "a.txt").exists()
    assert ok(dispatch(ctx, "ws_delete", {"path": "a.txt", "confirm": True})) == {"deleted": "a.txt"}


def test_ws_read_is_a_window_onto_a_long_file_and_says_it(ctx):
    """Read a file, edit it, write it back, and a silent cut loses the tail: so the reply says what it did not
    send and `chars=-1` reads whole."""
    ok(dispatch(ctx, "ws_write", {"path": "big.txt", "text": "y" * 300_000}))
    got = ok(dispatch(ctx, "ws_read", {"path": "big.txt"}))
    assert len(got["text"]) == 200_000 and got["total_chars"] == 300_000 and got["truncated"]
    assert got["chars_withheld"] == 100_000 and "offset=200000" in got["read_with"]
    rest = ok(dispatch(ctx, "ws_read", {"path": "big.txt", "offset": 200_000}))
    assert len(rest["text"]) == 100_000 and "truncated" not in rest
    whole = ok(dispatch(ctx, "ws_read", {"path": "big.txt", "chars": -1}))
    assert len(whole["text"]) == 300_000 and "truncated" not in whole and "total_chars" not in whole


# --- what `hidden` shows, and what it never shows ----------------------------------------------------
def test_hidden_shows_dot_names_but_never_git_or_the_vault(ctx, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENLAB_VAULT", str(tmp_path / ".vault"))
    (tmp_path / ".git").mkdir()
    (tmp_path / ".data").mkdir()
    (tmp_path / ".vault").mkdir()
    (tmp_path / ".notes.txt").write_text("x")
    ok(dispatch(ctx, "ws_write", {"path": "open.txt", "text": "x"}))
    plain = {e["name"] for e in ok(dispatch(ctx, "ws_list", {}))["entries"]}
    assert plain == {"open.txt"}
    shown = {e["name"] for e in ok(dispatch(ctx, "ws_list", {"hidden": True}))["entries"]}
    assert shown == {".data", ".notes.txt", "open.txt"}                          # never .git, never the vault
