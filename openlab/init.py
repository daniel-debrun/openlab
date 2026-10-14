r"""``openlab init``: set a folder up as an openlab project, the way ``/init`` sets up a repository for an agent.

Writes, only where the file does not exist yet:

* ``OPENLAB.md`` -- the project's analysis conventions, which every agent connected to openlab's MCP receives with
  its instructions (the outcome definitions, the holdout policy, what counts as a finding here);
* ``openlab.toml`` -- the workspace marker, with the [agents] defaults (budgets, tree limits); a person's file;
* ``.mcp.json`` -- the MCP server entry, so an agent CLI started here finds openlab's tools.

``--claude`` adds the Claude Code hooks (``.claude/settings.json``) and the role agents; ``--codex`` an AGENTS.md
section (:mod:`openlab.agents.claude_setup`).

Nothing existing is overwritten; the reply says what was written and what was already there.
"""

from __future__ import annotations

from pathlib import Path

CONVENTIONS = """# Analysis conventions

openlab's MCP gives this file to every agent working here, with its instructions. Write what a new analyst
would need on day one; keep it short.

## Data
- Where the data live, and which file is the source of truth:
- What one row is (a customer, a visit, a day):
- Known quirks (late-arriving rows, a changed definition, test accounts to drop):

## Outcomes and definitions
- The outcome(s), defined exactly:
- Units, currencies, time zones:

## How results are judged here
- Every experiment is a study (`openlab study new`); its holdout and primary metric are:
- A finding is a claim on a study's record; anything else is a lead.
- Report every run and test, not only the ones that cleared the bar.

## Do not
- Do not peek at the holdout to choose a model.
"""

def init(root: Path) -> dict:
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    from .agents.claude_setup import TOML
    wrote, kept = [], []
    for name, text in (("OPENLAB.md", CONVENTIONS), ("openlab.toml", TOML)):
        p = root / name
        if p.exists():
            kept.append(name)
        else:
            p.write_text(text, encoding="utf-8")
            wrote.append(name)
    try:
        from .app.server import write_mcp_json
        (wrote if write_mcp_json(root) else kept).append(".mcp.json")
    except ImportError:
        pass
    return {"root": str(root), "wrote": wrote, "kept": kept,
            "next": "fill in OPENLAB.md; then `openlab study new NAME`, `openlab app`, or start your agent CLI here "
                    "(it reads .mcp.json)"}


def conventions(root: Path, limit: int = 6000) -> str:
    """The project's OPENLAB.md, for the MCP instructions (cut at `limit` characters)."""
    p = Path(root) / "OPENLAB.md"
    try:
        text = p.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    return text[:limit] + ("\n[... cut: read OPENLAB.md for the rest]" if len(text) > limit else "")


def _files(paths: list) -> str:
    """Files for a person to read: the role agents under .claude/agents/ on one entry, the rest by name."""
    roles = [p[len(".claude/agents/"):].removesuffix(".md") for p in paths if p.startswith(".claude/agents/")]
    rest = [p for p in paths if not p.startswith(".claude/agents/")]
    return ", ".join(rest + ([f".claude/agents/ ({', '.join(roles)})"] if roles else []))


def render(r: dict) -> str:
    """What `openlab init` did, in a few lines: what it wrote, what it kept as it was, and the next step."""
    lines = [f"openlab workspace at {r['root']}"]
    for k, what in (("wrote", "wrote"), ("updated", "updated"), ("kept", "kept (already here, unchanged)")):
        if r.get(k):
            lines.append(f"  {what}: {_files(r[k])}")
    lines.append(f"next: {r['next']}")
    return "\n".join(lines)


def main(argv) -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser(prog="openlab init", description="Set this folder up as an openlab project.")
    ap.add_argument("dir", nargs="?", default=".")
    ap.add_argument("--claude", action="store_true", help="also the Claude Code hooks and the role agents")
    ap.add_argument("--codex", action="store_true", help="also AGENTS.md guidance and the Codex MCP snippet")
    ap.add_argument("--json", action="store_true", help="the same report as JSON, for a script or an agent")
    a = ap.parse_args(argv)
    r = init(Path(a.dir))
    if a.claude or a.codex:                                     # openlab.agents.claude_setup
        from .agents.claude_setup import init_claude, init_codex
        for extra in ([init_claude(Path(a.dir))] if a.claude else []) + ([init_codex(Path(a.dir))] if a.codex else []):
            for k in ("wrote", "kept", "updated"):
                r.setdefault(k, []).extend(extra.get(k, []))
            r["next"] = extra.get("next", r["next"])
        for k in ("wrote", "updated", "kept"):
            r[k] = list(dict.fromkeys(r.get(k, [])))               # each file once, in the order it was done
        r["kept"] = [k for k in r["kept"] if k not in r["wrote"] and k not in r["updated"]]
    print(json.dumps(r, indent=1) if a.json else render(r))
    return 0
