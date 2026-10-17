"""A study's experiments as tools (openlab.study.experiments): list, read, write a draft, ask the person to start one,
kill one, and record the conclusion of an ended one. Starting an experiment is the person's act (it runs unattended,
outside the sandbox): this process holds no person key, so the record refuses it here, and experiment_request files
the inbox item the person approves."""

from __future__ import annotations

from typing import Optional

from .registry import ToolContext, ToolError, no_agent_code, tool
from .study_tools import _errors


def _study(ctx: ToolContext, study: str) -> str:
    if not study:
        raise ToolError("name the study", "its folder, from the workspace")
    return str(ctx.path(study).relative_to(ctx.root))


@tool("workspace")
def experiment_list(ctx: ToolContext) -> dict:
    """Every experiment of the workspace's studies: state (draft, requested, running, concluded, killed, failed,
    reacting, reacted), purpose, passes done of the target, its Roadmap step."""
    from ..study import experiments as E
    with _errors():
        rows = E.overview(ctx.root)
    keep = ("study", "name", "state", "purpose", "plan_step", "owner", "spec_error", "edited")
    return {"experiments": [{**{k: r.get(k) for k in keep if k in r},
                             **({"passes": f"{r['progress']['passes_ok']}/{r['progress']['target'] or '-'}"}
                                if r.get("progress") else {})} for r in rows]}


@tool("workspace")
def experiment_status(ctx: ToolContext, study: str, name: str) -> dict:
    """One experiment: its state and spec, progress (passes, time to `until`), each job's next time and last runs
    (exit, run id: read one with instrument_output), the passes, failures, how it ended and the reaction agent with its
    conclusion."""
    from ..study import experiments as E
    with _errors():
        return E.status(ctx.root, _study(ctx, study), name)


@tool("workspace", mutates=True)
def experiment_new(ctx: ToolContext, study: str, name: str, toml: str = "", spec: Optional[dict] = None) -> dict:
    """Write a draft experiment, STUDY/experiments/NAME.toml, checked first: `toml` (the file's text) or `spec` (a
    table: {experiment: {name, purpose, plan_step}, jobs: [{name, instrument (a pinned alias), args, cron, tz}], pass:
    {job}, stop: {passes, until, max_failures}, react: {role, prompt, budget: {usd, minutes, turns}}}). Refused once
    the experiment has started: a changed experiment is a new name. Then experiment_request asks the person."""
    from ..study import experiments as E
    if bool(toml) == bool(spec):
        raise ToolError("give the spec as toml or as spec, one of the two",
                        "docs/experiments.md has the format; experiment_status reads a draft back")
    with _errors():
        return E.write(ctx.root, _study(ctx, study), name, toml or E.to_toml(spec))


@tool("workspace", mutates=True)
def experiment_request(ctx: ToolContext, study: str, name: str, reason: str) -> dict:
    """Ask the person to start the experiment (it runs unattended, outside the sandbox, so the start is theirs):
    files agent_request kind experiment with the spec's sha and records the request; returns the inbox id. Link it
    on your Roadmap step (inbox:<id>) and wait for it with agent_wait."""
    from ..study import experiments as E
    with _errors():
        return E.request(ctx.root, _study(ctx, study), name, reason)


@tool("workspace", mutates=True)
def experiment_kill(ctx: ToolContext, study: str, name: str, reason: str) -> dict:
    """End a running experiment now, with the reason: its crontab lines go at once, the end names you, and the
    reaction agent is fired."""
    from ..study import experiments as E
    no_agent_code("killing an experiment")
    with _errors():
        return E.kill(ctx.root, _study(ctx, study), name, reason)


@tool("workspace", mutates=True)
def experiment_react(ctx: ToolContext, study: str, name: str, verdict: str, conclusion: str, rests_on: list,
                     records: Optional[list] = None) -> dict:
    """Record the conclusion of an ended experiment: verdict (met, not_met, inconclusive) against its purpose, the
    conclusion in words, rests_on (its pass, run and end ids) and records (the ids you wrote on the way: claims, dead
    ends, questions). One per experiment; its Roadmap step then moves to done."""
    from ..study import experiments as E
    with _errors():
        return E.react(ctx.root, _study(ctx, study), name, verdict, conclusion, list(rests_on or []),
                       list(records or []))
