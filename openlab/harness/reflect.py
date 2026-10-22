"""The reflector: after a spawned agent ends, a cheap agent reads the run's evidence and records per-edit verdicts and
at most one harness lesson, or NO_ENTRY (openlab.harness.lessons.reflection). It proposes; it cannot apply: a lesson
waits in the inbox for the person, and nothing it writes changes a harness.

It runs when the person turns it on ([harness] reflect = true in openlab.toml) and costs [harness]
reflector_budget_usd per run, tagged `reflect:<agent>` so the spend is reported as harness overhead. It never runs on
the reflector, the refinement roles, or an agent of a refinement step (which reflects on the step as a whole).

The evidence is a folder outside .openlab/ (the hooks refuse reads there): the run's brief, task, transcript in the
[ACTION] / [OBSERVATION] / [ERROR] grammar, probe signals, a stored contrast of runs of its role by harness, and a
copy of the harness that briefed it. The meta prompts (meta/*.md) are openlab's, not a harness's: no editor can
rewrite the rules it is judged by.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

META = Path(__file__).resolve().parent / "meta"
META_ROLES = ("reflector", "harness-editor-inv", "harness-editor-dom", "composer", "selector")


def meta(name: str, **fields) -> str:
    return (META / f"{name}.md").read_text(encoding="utf-8").removesuffix("\n").format(**fields)


def _store(root):
    from ..agents.store import Store
    return Store(root)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_folder(root, agent: str, dest: Path) -> dict:
    """Write one run's evidence into `dest` (brief.txt, prompt.txt, transcript.txt, signals.json, deliverable.txt);
    its signals."""
    from .signals import of, transcript
    store = _store(root)
    d = store.agent_dir(agent)
    for name in ("brief.txt", "prompt.txt"):
        if (d / name).exists():
            _write(dest / name, (d / name).read_text(encoding="utf-8", errors="replace"))
    _write(dest / "transcript.txt", transcript(root, agent))
    sig = of(root, agent)
    _write(dest / "signals.json", json.dumps(sig, indent=1))
    last = [e for e in store.events(agent, n=200) if e["kind"] in ("result", "end")]
    _write(dest / "deliverable.txt", "\n".join(e["summary"] for e in last) or "(it ended with no summary)")
    return sig


def evidence(root, agent: str, dest=None) -> dict:
    """The reflector's evidence for `agent`: {dir, contrast} (the stored contrast id, or "")."""
    from . import versions
    from .signals import contrast, groups, runs, save, table
    dest = Path(dest or tempfile.mkdtemp(prefix=f"openlab-reflect-{agent}-"))
    sig = run_folder(root, agent, dest / "run")
    versions.export(root, sig["harness"], dest / "harness")
    peers = groups([r for r in runs(root, role=sig["role"]) if r["agent"] != agent])
    mine = [r for r in peers.pop(sig["harness_sha"], [])] + [sig]
    ref = max(peers.values(), key=len, default=[])
    cid = ""
    if ref:
        c = contrast(ref, mine)
        cid = save(root, c, note=f"reflection on {agent}: its harness against the most used other one of its role")
        _write(dest / "contrast.json", json.dumps({"id": cid, **c}, indent=1))
        _write(dest / "contrast.txt", "\n".join(table(c)))
    _write(dest / "README.md", "\n".join([
        f"# Evidence on openlab agent {agent} (role {sig['role']}, harness {sig['harness'][:12]})", "",
        "- run/brief.txt: the system prompt it was given; run/prompt.txt: its task",
        "- run/transcript.txt: what it did, as [SAY] / [ACTION] / [OBSERVATION] / [ERROR] / [RESULT] blocks",
        "- run/signals.json: its probe signals (rates; openlab.harness.signals.FIELDS says what each means)",
        "- run/deliverable.txt: how it ended",
        "- harness/: the harness that briefed it (brief.md, common.md, roles/, workflows/, tools/short.toml, skills/)",
        (f"- contrast.json, contrast.txt: stored contrast {cid}: {len(mine)} run(s) of its role on its harness "
         f"(cand) against {len(ref)} on another (ref); cite {cid} as a lesson's basis") if cid else
        "- no contrast: no run of its role used another harness, so no lesson can rest on this evidence yet"]))
    return {"dir": str(dest), "contrast": cid}


def after(root, agent: str) -> dict | None:
    """Spawn the reflector on an ended agent when [harness] reflect is on and the agent is one to reflect on."""
    from . import settings
    cfg = settings(root)
    if not cfg["reflect"]:
        return None
    store = _store(root)
    a = store.get(agent)
    if a is None or a["kind"] not in ("spawned", "codex") or a["role"] in META_ROLES:
        return None
    try:
        tag = json.loads((store.agent_dir(agent) / "launch.json").read_text()).get("tag", "")
    except (OSError, ValueError):
        tag = ""
    if tag.startswith(("refine:", "reflect:")):
        return None
    ev = evidence(root, agent)
    return spawn_reflector(root, agent, ev["dir"], float(cfg["reflector_budget_usd"]), f"reflect:{agent}")


def spawn_reflector(root, of: str, folder: str, usd: float, tag: str, label: str = "") -> dict:
    from ..agents.spawn import spawn
    a = _store(root).get(of) or {"role": "agent", "status": "ended"}
    prompt = meta("reflect", agent=label or of, role=a["role"], status=a["status"], dir=folder)
    row = spawn(root, prompt, role="reflector", question=f"reflect on {label or of}", cwd=folder,
                budget={"usd": usd, "minutes": 15}, tag=tag, advise=False, harness="default")
    return {"reflector": row["id"], "evidence": folder}
