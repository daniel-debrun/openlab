"""The harness as tools (openlab.harness): a skill's procedure and its sandboxed script, a run's probe signals, and a
harness lesson or a reflection filed for the person's decision. Nothing here changes the active harness or approves
a lesson: those are the person's, signed (`openlab harness use`, `openlab inbox approve`)."""

from __future__ import annotations

from typing import Optional

from .registry import ToolContext, ToolError, no_agent_code, tool


def _caller_harness(ctx: ToolContext):
    """The harness that briefed the calling agent (its launch.json), else the workspace's active one."""
    from ..agents.identity import current
    from ..agents.store import Store, root_of
    from ..harness import versions
    store = Store(root_of(ctx.root))
    import json
    try:
        sha = json.loads((store.agent_dir(current()) / "launch.json").read_text()).get("harness", "")
    except (OSError, ValueError, TypeError):
        sha = ""
    return store.root, (versions.get(store.root, sha) if sha else versions.active(store.root))


def _errors(fn):
    from ..harness import HarnessError
    try:
        return fn()
    except HarnessError as e:
        raise ToolError(e.problems[0], "; ".join(e.problems[1:]), "fix what it names and call again") from None
    except (KeyError, PermissionError) as e:
        raise ToolError(str(e).strip("'\"")) from None


@tool("research")
def harness_skill(ctx: ToolContext, name: str) -> dict:
    """A skill of your harness: its procedure (SKILL.md), when your brief says it applies."""
    from ..harness import skills
    root, h = _caller_harness(ctx)
    return _errors(lambda: {"name": name, "skill": skills.skill(h, name),
                            "script": f"skills/{name}/script.py" in h.files})


@tool("research", mutates=True)
def harness_script(ctx: ToolContext, name: str, args: Optional[list] = None, study: str = "") -> dict:
    """Run a skill's script in the sandbox (no network; `study` read-only with its record and data masked); returns
    its exit, output and the files it wrote to its stage."""
    from ..harness import skills
    no_agent_code("a harness script")
    root, h = _caller_harness(ctx)
    return _errors(lambda: skills.run(root, h, name, args or [], study))


@tool("research")
def harness_signals(ctx: ToolContext, agent: str = "") -> dict:
    """An ended agent's probe signals (rates from its events and records), keyed by the harness that briefed it."""
    from ..agents.identity import current
    from ..harness import signals
    root, _ = _caller_harness(ctx)
    return _errors(lambda: signals.of(root, agent or current()))


@tool("research", mutates=True)
def harness_lesson_propose(ctx: ToolContext, cue: str, component: str, advice: str, basis: list,
                           predicted: Optional[dict] = None, provenance: Optional[list] = None,
                           keys: Optional[dict] = None, edit: Optional[dict] = None) -> dict:
    """Propose a harness lesson for the person's approval: cue (the situation's shape), component, advice (an
    observed pattern), basis (stored contrast ids), predicted {signals: {field: up|down}, tasks}, provenance (run
    ids), keys {roles, templates}, edit {base, files: {path: text}}. Returns its id and inbox item."""
    from ..agents.identity import current
    from ..harness import lessons
    root, _ = _caller_harness(ctx)
    return _errors(lambda: lessons.propose(root, current(), cue, component, advice, basis, predicted, provenance,
                                           keys, edit))


@tool("research", mutates=True)
def harness_reflection(ctx: ToolContext, of: str, verdicts: list, lesson: Optional[dict] = None) -> dict:
    """Record your reflection on run `of`: verdicts [{edit, verdict: helped|hurt|no observed effect|confounded, why}]
    and at most one lesson (harness_lesson_propose's arguments), or none (NO_ENTRY)."""
    from ..agents.identity import current
    from ..harness import lessons
    root, _ = _caller_harness(ctx)
    return _errors(lambda: lessons.reflection(root, current(), of, verdicts, lesson))
