"""The default MCP list is the CORE tools and the generic pair; openlab_tool reaches the rest through the same gates."""
import asyncio
import importlib
import json
import re
import sys

import pytest

pytest.importorskip("mcp")


@pytest.fixture
def M(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENLAB_MCP_TOOLS", raising=False)
    monkeypatch.delenv("OPENLAB_APP_URL", raising=False)
    import openlab.app.client as C
    monkeypatch.setattr(C, "_LOCAL", None)                       # a context another test left behind has its root
    sys.modules.pop("openlab.mcp_server", None)
    yield importlib.import_module("openlab.mcp_server")
    sys.modules.pop("openlab.mcp_server", None)                   # leave a default module, without --http's removals
    importlib.import_module("openlab.mcp_server")


def _names(M):
    return {t.name.removeprefix("openlab_") for t in asyncio.run(M.server.list_tools())}


def test_the_default_list_is_the_core_and_the_pair(M):
    from openlab.app import TOOLS
    assert M.CORE <= set(TOOLS) and not any(TOOLS[n].human for n in M.CORE)      # no drift, no person's act
    assert _names(M) == M.CORE | {"tools", "tool"}
    assert "openlab_tool(name, args)" in M.INSTRUCTIONS


def _surface(M) -> str:
    """What a client pays for before its first call: each tool's name, description and input schema, and the
    server's instructions."""
    keep = ("name", "title", "description", "inputSchema", "outputSchema", "annotations")
    rows = [{k: v for k, v in t.model_dump(by_alias=True, exclude_none=True, mode="json").items() if k in keep}
            for t in asyncio.run(M.server.list_tools())]
    return json.dumps(rows) + M.INSTRUCTIONS


def test_the_default_surface_stays_under_the_context_budget(M):
    size = len(_surface(M))
    assert size <= M.BUDGET_CHARS, f"the default MCP costs {size} chars (~{size // 4} tokens), over {M.BUDGET_CHARS}"
    assert all(len(d) <= 240 for d in M.SHORT.values())                 # one line each


def test_every_tool_a_brief_names_is_listed_or_called_through_openlab_tool(M):
    from openlab.agents import claude_setup, spawn
    from openlab.app import TOOLS
    from openlab.study import proposal
    texts = [M.INSTRUCTIONS, claude_setup.COMMON, spawn.BRIEF, proposal.LEAD, proposal.PLANNER,
             *(body for _, body in claude_setup.ROLES.values())]
    for text in texts:
        for name in re.findall(r"\bopenlab_([a-z_]+)", text):
            assert name in M.CORE | {"tool", "tools"}, f"a brief names openlab_{name}, which is not in the default list"
        for name in re.findall(r'openlab_tool\("([a-z_]+)"', text):
            assert name in TOOLS and not TOOLS[name].human, name


def test_openlab_tool_reaches_the_rest_and_refuses_a_persons_act(M, tmp_path):
    assert "ws_write" not in _names(M)
    assert "ws_write" in [r["name"] for r in M.openlab_tools(query="ws_write")["result"]["tools"]]
    assert M.openlab_tool("ws_write", {"path": "a.txt", "text": "x"})["ok"] and (tmp_path / "a.txt").exists()
    assert M.openlab_tool("openlab_ws_delete", {"path": "a.txt"})["error"] == "confirm"     # the ceremony holds
    assert M.openlab_tool("study_holdout", {"folder": "s"})["error"] == "a reviewer's act"
    assert "study_holdout" not in str(M.openlab_tools(section="study"))
    assert M.openlab_tool("ws_write", {"path": "openlab.toml", "text": "x"})["error"] == "a person's file"


def test_under_http_openlab_tool_keeps_to_the_allowlist(M, monkeypatch):
    monkeypatch.setenv("OPENLAB_NO_AGENT_CODE", "1")             # as main() sets it for --http
    M._serve_agent_tools()
    assert {"tools", "tool"} <= _names(M) <= M.AGENT_TOOLS
    for name in ("agent_spawn", "paper_search", "study_discover"):     # not allowlisted
        assert M.openlab_tool(name, {})["error"] == "unknown tool"
    assert "agent_spawn" not in [r["name"] for r in M.openlab_tools(section="research")["result"]["tools"]]
    assert M.openlab_tool("notes_list", {})["ok"]                # allowlisted but not core: reachable
    assert M.openlab_tool("study_run", {"folder": "s"})["error"] == "study_run is off"    # agent code stays off
