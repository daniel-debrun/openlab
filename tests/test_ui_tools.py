"""Every tool the web UI calls exists in the dispatch registry.

The UI reaches the server only through POST /api/tool/{name}; a name the registry does not have is a button that
always answers "unknown tool". This scans the UI source (openlab/app/ui/src) two ways:

* the call sites: `callTool("x"`, `runTool<T>("x"`, `callConfirmed("x"`, `personAct("x"`, `act("x"` and
  `useRead<T>(folder, "x"`;
* any string literal shaped like a tool of a tool family (`study_*`, `agent_*`, ...) or of a family that was
  removed (`data_*`, `dash_*`, `bench_*`, ...), so a name passed through a variable is caught too.
"""

import re
from pathlib import Path

from openlab.app import TOOLS

SRC = Path(__file__).resolve().parents[1] / "openlab" / "app" / "ui" / "src"

#: a call of the dispatch, with the tool name as the first string argument (after an optional leading argument
#: such as a study folder); generic type arguments may sit between the name and the parenthesis
CALL = re.compile(r"\b(?:callTool|runTool|callConfirmed|personAct|useRead|act)\b\s*(?:<[^()]*?>)?\s*\(\s*"
                  r"(?:[A-Za-z_][\w.]*(?:\s*\|\|\s*\"[^\"]*\")?\s*,\s*)?\"([a-z][a-z0-9_]*)\"")
LITERAL = re.compile(r"[\"'`]([a-z]+_[a-z0-9_]*[a-z0-9])[\"'`]")
#: families of tools that no longer exist: a literal of one of these is a call to a missing tool
REMOVED = {"data", "dash", "bench", "holdout", "run", "rq", "trials", "snapshot", "viz", "module", "modules",
           "registry", "journal", "names", "submissions", "friction", "frame", "model", "verify", "parse"}


def _sources():
    return [(p, p.read_text(encoding="utf-8")) for p in sorted(SRC.rglob("*")) if p.suffix in (".ts", ".tsx")]


def test_every_call_site_names_a_tool_the_server_has():
    found = {}
    for p, text in _sources():
        for m in CALL.finditer(text):
            found.setdefault(m.group(1), []).append(p.relative_to(SRC).as_posix())
    assert len(found) >= 20, f"the scan found only {sorted(found)}: the CALL pattern no longer matches the UI"
    missing = {name: where for name, where in found.items() if name not in TOOLS}
    assert not missing, f"the UI calls tools the registry does not have: {missing}"


def test_no_literal_names_a_missing_tool():
    families = {n.split("_")[0] for n in TOOLS if "_" in n} | REMOVED
    bad = {}
    for p, text in _sources():
        for m in LITERAL.finditer(text):
            name = m.group(1)
            if name.split("_")[0] in families and name not in TOOLS:
                bad.setdefault(name, []).append(p.relative_to(SRC).as_posix())
    assert not bad, f"names of tools the registry does not have: {bad}"


def test_the_agents_view_reads_and_acts_through_the_control_plane_tools():
    text = "\n".join(t for p, t in _sources() if "agents" in p.parts)
    for name in ("agent_tree", "agent_log", "inbox_list", "agent_control", "inbox_decide"):
        assert f'"{name}"' in text, name
    assert TOOLS["agent_control"].human and TOOLS["inbox_decide"].human
    assert not TOOLS["agent_tree"].mutates and not TOOLS["inbox_list"].mutates
