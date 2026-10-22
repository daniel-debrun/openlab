"""Recorded agent runs for the harness tests: an agent registered as openlab spawns it (launch.json, harness_sha),
its events written as its supervisor would, and its end. Nothing here calls a model or starts a process."""

import json
import shutil
from pathlib import Path

from openlab import harness as H


def run(root, role="screener", harness="", calls=10, refusals=0, errors=0, tools=(), status="done", prompt="x",
        cwd=None, tag="", say="done: nothing recorded", **kw) -> str:
    """An ended spawned agent with `calls` tool calls, `refusals` of them refused and `errors` failed."""
    from openlab.agents.spawn import spawn
    from openlab.agents.store import Store
    a = spawn(root, prompt, role=role, start=False, harness=harness, cwd=cwd, tag=tag, **kw)
    st = Store(root)
    for i in range(calls):
        st.event(a["id"], "tool", tool=tools[i % len(tools)] if tools else "mcp__openlab__openlab_study_status",
                 summary=f"call {i}")
    for i in range(refusals):
        st.event(a["id"], "refusal", tool="mcp__openlab__openlab_study_run", summary="study_run: refused the same way")
    for i in range(errors):
        st.event(a["id"], "tool_error", tool="Bash", summary=f"Bash: error {i}")
    st.add_spent(a["id"], tool_calls=calls, usd=0.01 * calls)
    st.event(a["id"], "say", summary=say)
    st.event(a["id"], "result", summary="success", payload={"subtype": "success"})
    st.set_status(a["id"], status)
    return a["id"]


def candidate(root, tmp, extra="- Before you finish, read your own plan once more.", rel="common.md") -> Path:
    """A copy of the packaged harness with one sentence added to `rel`."""
    d = Path(tmp)
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(H.DEFAULT, d, ignore=shutil.ignore_patterns("__pycache__"))
    (d / rel).write_text((d / rel).read_text() + extra + "\n")
    return d


def launch(root, agent) -> dict:
    from openlab.agents.store import Store
    return json.loads((Store(root).agent_dir(agent) / "launch.json").read_text())
