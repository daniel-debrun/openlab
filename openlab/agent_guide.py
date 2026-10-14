"""The agent contract in brief: the instructions the MCP gives an agent when it connects, which `openlab guide` also
prints. It lives apart from :mod:`openlab.mcp_server` (which re-exports ``INSTRUCTIONS``) so reading it never imports
the mcp SDK.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

GUIDE = Path(__file__).resolve().parent / "AGENTS.md"          # packaged; docs/AGENTS.md links here

BRIEF = """Openlab keeps the research record: you write the model code; Openlab runs, counts and audits it.
A study is a folder: study.toml (the plan), model.py (features, label, Model.fit/predict), recipes/ (data scripts)
and .openlab/ (the record; never read or write there). Order: study_new; edit the files; study_materialise; study_run (each
run is a counted trial; the first seals study.toml); study_status, study_runs, study_why to read; study_amend to
record an edit with its reason. Report through the record (study_claim, study_test): a number comes from a run or
quote-checked evidence, never typed.
A refusal is the structure working: read error, meaning and try, and do what try says. Never route around one: no
second study for fresh trials or a fresh holdout, no faked terminals.
Person-only acts (holdout looks, protected amendments, screens over yours, approvals, waivers, more budget) are signed
with the person's passphrase-protected key: the record refuses one without it and verify flags forgeries, so no
shell of yours can perform one; attempts are recorded with your id. Ask with agent_request, the exact change in args.
Read openlab://guide for the full contract."""


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


def instructions(mode: str | None = None) -> str:
    """The instructions for the tool list ``mode`` (default: the one OPENLAB_MCP_TOOLS names)."""
    mode = tool_mode() if mode is None else mode
    text = BRIEF + "\n" + env_line()
    if mode == "grouped":
        text += ("\nThe workspace tools are grouped by section: a name used above such as openlab_study_run "
                 'is openlab_study(tool="study_run", args={...}); each family lists its members, and '
                 'openlab_tools(section=...) their arguments.')
    elif mode != "all":
        text += ("\nTools not in your list: openlab_tools(query=...) finds them, openlab_tool(name, args) "
                 "calls them.")
    return text + conventions()
