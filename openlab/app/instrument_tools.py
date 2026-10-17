"""The workspace's instruments as tools (openlab.study.instruments): list, show, verify, run and read runs. Registering
a version is the person's act (an agent files agent_request kind "instrument"), and so is any run of a live-order
instrument (kind "instrument_run"): this process holds no person key, so the record refuses both here."""

from __future__ import annotations

from typing import Optional

from .registry import ToolContext, no_agent_code, tool
from .study_tools import _errors


@tool("workspace")
def instrument_list(ctx: ToolContext) -> dict:
    """The registered instruments and their versions: capabilities, runtime, snapshot or external, runs, reason."""
    from ..study import instruments
    with _errors():
        return {"instruments": instruments.listing(ctx.root)}


@tool("workspace")
def instrument_show(ctx: ToolContext, ref: str, verify: bool = False) -> dict:
    """One version's manifest (`ref`: NAME@VERSION, or NAME with one version); verify=true checks every hash again."""
    from ..study import instruments
    with _errors():
        reg = instruments.Registry(ctx.root)
        node = instruments.resolve(reg, ref)
        out = {"id": node.id, "at": node.at, "payload": node.payload}
        if verify:
            out["verify"] = instruments.verify(reg, node)
        return out


@tool("workspace")
def instrument_verify(ctx: ToolContext, ref: str) -> dict:
    """Check a version's snapshot, config and environment parts against its registration; lists what drifted."""
    from ..study import instruments
    with _errors():
        reg = instruments.Registry(ctx.root)
        return instruments.verify(reg, instruments.resolve(reg, ref))


@tool("workspace", mutates=True)
def instrument_run(ctx: ToolContext, ref: str, args: Optional[list] = None, study: str = "", note: str = "") -> dict:
    """Run a registered version (verified first) with `args`; recorded, and with `study` also on that study's record
    (it must pin the version: `ref` may be its alias). A live-order instrument is refused: file agent_request kind
    instrument_run."""
    from ..study import instruments
    no_agent_code("running an instrument")
    with _errors():
        return instruments.run(ctx.root, ref, [str(a) for a in (args or [])],
                               study=str(ctx.path(study).relative_to(ctx.root)) if study else "", note=note)


@tool("workspace")
def instrument_runs(ctx: ToolContext, ref: str = "", study: str = "", n: int = 20) -> dict:
    """Recorded runs, newest first (of `ref`, or on `study`): args, exit, start and end, output hashes."""
    from ..study import instruments
    with _errors():
        if not instruments.Registry.exists(ctx.root):
            return {"runs": []}
        return {"runs": instruments.runs(instruments.Registry(ctx.root), ref, study, n)}
