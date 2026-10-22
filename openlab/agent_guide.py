"""The agent contract in brief: the instructions the MCP gives an agent when it connects, which `openlab guide` also
prints. It lives apart from :mod:`openlab.mcp_server` (which re-exports ``INSTRUCTIONS``) so reading it never imports
the mcp SDK.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .harness import load as load_harness

GUIDE = Path(__file__).resolve().parent / "AGENTS.md"          # packaged; docs/AGENTS.md links here

BRIEF = load_harness().text("guide.md")


def tool_mode() -> str:
    """The MCP's tool list (OPENLAB_MCP_TOOLS): "" (the CORE tools and openlab_tools / openlab_tool for every other),
    "grouped" (one tool per section) or "all" (every tool on its own)."""
    return os.environ.get("OPENLAB_MCP_TOOLS", "").strip().lower()


def env_line() -> str:
    """Which Python and which library versions this server computes with, so a reply can be reproduced."""
    from importlib.metadata import PackageNotFoundError, version

    def v(pkg):
        try:
            return version(pkg)
        except PackageNotFoundError:
            return "not installed"
    return (f"This server runs {sys.executable}: polars {v('polars')}, numpy {v('numpy')}, "
            f"scikit-learn {v('scikit-learn')}.")


def conventions() -> str:
    """The workspace's OPENLAB.md, as a section of the instructions ("" when it has none)."""
    from .init import conventions as read
    text = read(Path(os.environ.get("OPENLAB_WORKSPACE") or "."))
    return f"\n\nThis project's conventions (OPENLAB.md, written by the person; follow them):\n{text}" if text else ""


def active_harness():
    """The harness of the workspace this server serves (OPENLAB_WORKSPACE): its active version, else the packaged
    one (a version that does not validate here falls back to the packaged one: the MCP always starts)."""
    root = os.environ.get("OPENLAB_WORKSPACE")
    if not root:
        return load_harness()
    try:
        from .harness.versions import active
        return active(Path(root))
    except Exception:                                               # noqa: BLE001 -- spawn refuses it, loudly
        return load_harness()


def instructions(mode: str | None = None) -> str:
    """The instructions for the tool list ``mode`` (default: the one OPENLAB_MCP_TOOLS names)."""
    mode = tool_mode() if mode is None else mode
    text = active_harness().text("guide.md") + "\n" + env_line()
    if mode == "grouped":
        text += ("\nThe workspace tools are grouped by section: a name used above such as openlab_study_run "
                 'is openlab_study(tool="study_run", args={...}); each family lists its members, and '
                 'openlab_tools(section=...) their arguments.')
    elif mode != "all":
        text += ("\nTools not in your list: openlab_tools(query=...) finds them, openlab_tool(name, args) "
                 "calls them.")
    return text + conventions()
