r"""openlab.app -- the workspace: one dispatch behind the web UI, the CLI, the MCP and the agent.

    from openlab.app import ToolContext, dispatch
    ctx = ToolContext("my_workspace")
    dispatch(ctx, "study_new", {"folder": "churn", "template": "static"})
    dispatch(ctx, "study_run", {"folder": "churn"})
"""

from . import tools as _tools  # noqa: F401 -- registers the tools
from . import agent_tools as _agent_tools  # noqa: F401,E402 -- the agent control plane's tools
from . import propose_tools as _propose_tools  # noqa: F401,E402 -- research from a prompt
from . import show_tools as _show_tools  # noqa: F401,E402 -- what openlab show draws
from . import instrument_tools as _instrument_tools  # noqa: F401,E402 -- registered programs outside the sandbox
from . import experiment_tools as _experiment_tools  # noqa: F401,E402 -- scheduled instrument runs, then a conclusion
from . import harness_tools as _harness_tools  # noqa: F401,E402 -- skills, probe signals, harness lessons
from .registry import TOOLS, Tool, ToolContext, ToolError, dispatch

__all__ = ["TOOLS", "Tool", "ToolContext", "ToolError", "dispatch"]
