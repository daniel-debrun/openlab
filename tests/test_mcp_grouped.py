"""OPENLAB_MCP_TOOLS=grouped: one MCP tool per section, so an agent's first tools/list is small (B8)."""
import asyncio
import importlib
import json
import sys


def _fresh(monkeypatch, mode):
    if mode:
        monkeypatch.setenv("OPENLAB_MCP_TOOLS", mode)
    else:
        monkeypatch.delenv("OPENLAB_MCP_TOOLS", raising=False)
    sys.modules.pop("openlab.mcp_server", None)
    return importlib.import_module("openlab.mcp_server")


def test_grouped_lists_one_tool_per_section_and_dispatches_by_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import openlab.app.client as C
    monkeypatch.setattr(C, "_LOCAL", None)                       # a context another test left behind has its root
    M = _fresh(monkeypatch, "grouped")
    try:
        tools = asyncio.run(M.server.list_tools())
        names = {t.name for t in tools}
        assert len(tools) <= 8 and {"openlab_workspace", "openlab_study", "openlab_tools"} <= names
        assert "openlab_ws_write" not in names
        size = sum(len(json.dumps(t.model_dump(mode="json"))) for t in tools)
        out = asyncio.run(M.server.call_tool("openlab_workspace", {"tool": "openlab_ws_write", "args": {"path": "a.txt", "text": "x"}}))
        assert '"ok": true' in str(out) and (tmp_path / "a.txt").exists()     # with or without the prefix
        bad = asyncio.run(M.server.call_tool("openlab_study", {"tool": "ws_write"}))
        assert "is not a study tool" in str(bad)
        assert "grouped by section" in M.INSTRUCTIONS
    finally:
        full = _fresh(monkeypatch, "all")
        _fresh(monkeypatch, "")                                                   # leave the default module behind
    full_size = sum(len(json.dumps(t.model_dump(mode="json"))) for t in asyncio.run(full.server.list_tools()))
    assert size * 3 < full_size
