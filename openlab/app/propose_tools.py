"""Research from a prompt as tools (openlab.study.proposal), so the researcher's own Claude Code or Codex can draft
the proposal, and the lead agent can read progress and stop a branch. Approval and protected amendments are a
person's acts (`openlab propose approve`), never tools. Descriptions are short: every agent pays for them."""
from __future__ import annotations

from contextlib import contextmanager

from .registry import ToolContext, ToolError, tool


@contextmanager
def _errors():
    from ..study.errors import StudyError
    try:
        yield
    except StudyError as e:
        raise ToolError(e.error, e.meaning, e.fix) from None


def _open(ctx: ToolContext, folder: str):
    from .. import study
    return study.open(ctx.path(folder))


@tool("study", mutates=True)
def propose_new(ctx: ToolContext, folder: str, prompt: str, mode: str = "static", budget_usd: float = 0.0) -> dict:
    """Scaffold a research proposal from a prompt: the study folder, proposal.toml (to fill in) and proposal.md."""
    from ..study import proposal
    with _errors():
        out = proposal.new(ctx.path(folder), prompt, mode, budget_usd or None)
    return {**out, "folder": folder, "next": "fill proposal.toml, then propose_lint until it is clean; a person "
            "approves it with `openlab propose approve`"}


@tool("study")
def propose_lint(ctx: ToolContext, folder: str) -> dict:
    """Check proposal.toml: every refusal (error, meaning, fix) and warning; renders proposal.md."""
    from ..study import proposal
    with _errors():
        out = proposal.lint(_open(ctx, folder))
    return {k: out[k] for k in ("ok", "status", "problems", "warnings", "changed", "sha") if k in out}


@tool("study")
def propose_show(ctx: ToolContext, folder: str) -> dict:
    """The proposal as rendered for reading (proposal.md) and its standing."""
    from ..study import proposal
    with _errors():
        s = _open(ctx, folder)
        out = proposal.lint(s)
        return {"status": out["status"], "ok": out["ok"], "md": (s.folder / proposal.MD).read_text(encoding="utf-8")}


@tool("study")
def propose_status(ctx: ToolContext, folder: str) -> dict:
    """Progress per branch of the approved proposal: agents and their state, $ / minutes / trials of its budget,
    hypotheses tested, killed and untouched, dead ends, the stop rule."""
    from ..study import proposal
    with _errors():
        out = proposal.status(_open(ctx, folder))
    for b in out["branches"]:
        b["hypotheses"] = [{k: h[k] for k in ("id", "node", "state", "verdict", "by")} for h in b["hypotheses"]]
    return out


@tool("study", mutates=True)
def propose_stop(ctx: ToolContext, folder: str, branch: str, reason: str, killed: bool = False,
                 shown_by: str = "") -> dict:
    """Record a branch stopped by its stop rule, or killed (killed=true) by its kill criterion; shown_by: comma-
    separated result, evidence or test ids."""
    from ..study import proposal
    ids = [i.strip() for i in str(shown_by).split(",") if i.strip()]
    with _errors():
        return proposal.stop(_open(ctx, folder), branch, reason, killed, ids)
