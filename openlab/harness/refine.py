"""Per-task refinement (`openlab harness refine`): one Propose-Probe-Compose step on a bounded unit of work, run on
probe copies, never on the study itself.

    reference  the unit run once with the base harness (the workspace's active one), in a probe copy
    propose    two editors, each in a folder holding a copy of the harness and the reference run's evidence:
               harness-editor-inv edits procedure and tooling only, harness-editor-dom practitioner knowledge only
               (meta/edit_inv.md, meta/edit_dom.md); they edit the copy and write notes, and can apply nothing
    validate   each edited harness against openlab.harness.validate (one retry, told why it was refused)
    probe      each valid candidate stored as a version and run on the unit in a probe copy of its own
    compose    a composer reads the pool (reference, each candidate's diff, notes, probe run and signal contrast)
               and writes keep_base, adopt <candidate> or merge, each kept or dropped edit citing its evidence
    deliver    the unit run with the composed harness
    select     at k > 1, a blind selector (solving first, comparing pairwise) picks among the groups' deliverables,
               option A being the base harness's, so it can abstain
    reflect    the reflector on the step: per-edit verdicts and at most one harness lesson for the person

A step costs exactly 4k executor rollouts and 3k improver sessions (plus the selector at k > 1 and the reflector);
the ledger asserts it. What it selects is a harness, stored as a version: it becomes active only through a person
(`openlab harness use`, with an approved lesson), and the unit's deliverable stays in the probe copy for the person to
read, never written into the study. Every probe's record writes stay in its copy; the runs they made are counted in
the ledger (`probe_trials`).

Units: `propose-plan` (a proposal's planner, on a copy of its study), `replicate-plan` (a replication's planner, on a
copy of its folder) and `task` (a prompt, on a copy of a folder of files). Starting a step is a person act (it spends:
[harness] refine_budget_usd, at most refine_max_fraction of [agents] cap_usd), signed and recorded. Each stage writes
a completion marker in .openlab/harness/refine/<step>/state.json, and each improver session is cached by the sha of
its role, mission, harness and evidence, so a step resumed after a crash redoes nothing it finished.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

from . import HarnessError, laws, read, settings, validate
from .reflect import meta, run_folder

UNITS = ("propose-plan", "replicate-plan", "task")
EDITORS = (("inv", "harness-editor-inv", "edit_inv"), ("dom", "harness-editor-dom", "edit_dom"))
SHARES = {"executor": 0.6, "improver": 0.3, "rest": 0.1}       # of the step's budget: rollouts, editors+composer, rest
IGNORE = shutil.ignore_patterns(".openlab", "release", "branches", "__pycache__", ".git")


def _store(root):
    from ..agents.store import Store
    return Store(root)


def _home(root, step: str) -> Path:
    from .versions import home
    return home(root) / "refine" / step


def _sha(*parts) -> str:
    return hashlib.sha256("\x00".join(map(str, parts)).encode("utf-8")).hexdigest()


def _tree_sha(folder: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(Path(folder).rglob("*")):
        if f.is_file():
            h.update(f"{f.relative_to(folder)}\0".encode() + f.read_bytes())
    return h.hexdigest()


def budget(root) -> dict:
    """The step's $ and how it is shared: executors, improvers (editors and composer), the rest (selector,
    reflector)."""
    from ..agents.store import config
    s = settings(root)
    usd = min(float(s["refine_budget_usd"]), float(s["refine_max_fraction"]) * float(config(root)["cap_usd"]))
    return {"usd": round(usd, 4), **{k: round(usd * v, 4) for k, v in SHARES.items()},
            "minutes": float(s["probe_minutes"])}


# --- the runner: spawn an agent and wait for it to end -----------------------------------------------------------
def spawn_and_wait(root, role: str, prompt: str, cwd: str, harness: str, usd: float, minutes: float, tag: str,
                   study: str = "", cli: str = "claude") -> str:
    """Spawn an agent and block until it ends (its minutes and 10 more at most); its id."""
    from ..agents.spawn import spawn
    from ..agents.store import FINAL
    a = spawn(root, prompt, role=role, study=study or None, cwd=cwd, harness=harness, tag=tag, advise=False,
              cli=cli, budget={"usd": usd, "minutes": minutes})
    store, deadline = _store(root), time.monotonic() + (minutes + 10) * 60
    while time.monotonic() < deadline:
        if (store.get(a["id"]) or {}).get("status") in FINAL:
            return a["id"]
        time.sleep(2)
    store.control(a["id"], "stop", "the refinement step's time is up", "openlab")
    return a["id"]


Runner = Callable[..., str]


# --- the unit: what the executor is asked to do, on which copy, and what it delivers ------------------------------
def _prepare(root, unit: str, source: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    if source.is_dir():
        shutil.copytree(source, dest, ignore=IGNORE)
    else:
        dest.mkdir(parents=True)


def unit_prompt(root, unit: str, h, copy: Path, text: str, source: Path) -> str:
    """The executor's task for `unit` on probe copy `copy`, from harness `h`'s own workflow brief."""
    rel = str(copy.relative_to(root))
    if unit == "propose-plan":
        from ..study import proposal as P
        raw = P.load(copy)
        limit = P.cap(source)
        return h.text("workflows/propose_planner.md").format(
            rel=rel, roles=" | ".join(P.ROLES), cap_usd=limit["usd"], cap_minutes=limit["minutes"],
            mode=raw.get("mode", "static"), prompt=(text or raw.get("question") or "").strip())
    if unit == "replicate-plan":
        from ..study import paperplan as R
        from ..study.metrics import METRICS
        limit = R.cap(source)
        return h.text("workflows/replicate_planner.md").format(
            folder=copy, key=R.source(copy)["key"], metrics=", ".join(METRICS), cap_usd=limit["usd"],
            cap_minutes=limit["minutes"], **R.machine(), fit=R.FIT)
    return text


def deliverable(unit: str, copy: Path) -> dict:
    """{name: text} of what the unit made: the drafted plan, or a task folder's text files (small ones)."""
    if unit == "propose-plan":
        names = ["proposal.toml"]
    elif unit == "replicate-plan":
        names = ["replication.toml"]
    else:
        names = sorted(str(f.relative_to(copy)) for f in copy.rglob("*") if f.is_file() and f.stat().st_size < 200_000)
    out = {}
    for n in names[:50]:
        try:
            out[n] = (copy / n).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
    return out


def _write_tree(folder: Path, files: dict) -> None:
    for name, text in files.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text(text, encoding="utf-8")


# --- the step ----------------------------------------------------------------------------------------------------
class Step:
    def __init__(self, root, step: str, unit: str, source, text: str = "", k: int = 1, runner: Optional[Runner] = None,
                 selector_cli: str = "claude"):
        if unit not in UNITS:
            raise HarnessError([f"a unit is one of {', '.join(UNITS)}"])
        self.root, self.id, self.unit, self.text, self.k = Path(root), step, unit, text, int(k)
        self.source = Path(source).resolve() if source else Path(tempfile.mkdtemp(prefix="openlab-task-"))
        self.run_agent = runner or spawn_and_wait
        self.selector_cli = selector_cli
        self.home = _home(root, step)
        self.home.mkdir(parents=True, exist_ok=True)
        self.state = json.loads((self.home / "state.json").read_text()) if (self.home / "state.json").exists() else {}
        self.state.setdefault("ledger", {"executor": [], "improver": [], "selector": [], "reflector": []})
        self.probes = Path(root) / ".probes" / step
        self.budget = budget(root)

    def save(self) -> None:
        (self.home / "state.json").write_text(json.dumps(self.state, indent=1, sort_keys=True), encoding="utf-8")

    def stage(self, name: str, fn: Callable[[], dict]) -> dict:
        """Run a stage once: its result is kept under its completion marker."""
        if name in self.state.get("done", {}):
            return self.state["done"][name]
        out = fn()
        self.state.setdefault("done", {})[name] = out
        self.save()
        return out

    def _agent(self, kind: str, role: str, prompt: str, cwd: Path, harness: str, study: str = "",
               cli: str = "claude") -> str:
        # an improver's share leaves room for each editor's one retry: 3 sessions and 2 retries per group
        share = {"executor": self.budget["executor"] / (4 * self.k), "improver": self.budget["improver"] / (5 * self.k),
                 "selector": self.budget["rest"] / 2, "reflector": self.budget["rest"] / 2}[kind]
        planned = sum(float(x["usd"]) for v in self.state["ledger"].values() for x in v) + share
        if planned > self.budget["usd"] + 1e-6:
            raise HarnessError([f"the step's budget ${self.budget['usd']:g} is spent"])
        aid = self.run_agent(self.root, role, prompt, str(cwd), harness, round(share, 4), self.budget["minutes"],
                             f"refine:{self.id}", study=study, cli=cli)
        self.state["ledger"][kind].append({"agent": aid, "role": role, "usd": round(share, 4)})
        self.save()
        return aid

    # executor: the unit on a probe copy, with harness `sha`
    def execute(self, label: str, sha: str) -> dict:
        from .versions import get
        copy = self.probes / label / (self.source.name or "task")
        _prepare(self.root, self.unit, self.source, copy)
        h = get(self.root, sha)
        prompt = unit_prompt(self.root, self.unit, h, copy, self.text, self.source)
        role = {"propose-plan": "planner", "replicate-plan": "planner"}.get(self.unit, "agent")
        study = str(copy.relative_to(self.root)) if self.unit != "task" else ""
        aid = self._agent("executor", role, prompt, copy, sha or "default", study=study)
        from . import replay
        from .signals import of
        sig = of(self.root, aid)
        kind = "reference" if label.endswith("reference") else "deliver" if label.endswith("deliver") else "probe"
        replay.log(self.root, {"step": self.id, "task": self.unit, "kind": kind, "label": label, "agent": aid,
                               "harness": h.sha, "parent": self.state.get("base", ""), "signals": sig["signals"],
                               "usd": sig["usd"], "deliverable": str(copy)})
        return {"agent": aid, "copy": str(copy), "harness": h.sha}

    def _evidence_run(self, dest: Path, run: dict) -> None:
        run_folder(self.root, run["agent"], dest)
        _write_tree(dest / "deliverable", deliverable(self.unit, Path(run["copy"])))

    def _cached(self, label: str, role: str, mission: str, folder: Path, sha: str, outputs: tuple) -> bool:
        """Run an improver session unless its sha (role, mission, harness, evidence) is cached; then copy its
        outputs into `folder`. Whether it was cached."""
        from .evaluate import tripwire
        hits = tripwire(folder)
        if hits:
            raise HarnessError([f"the evidence for {label} shows a grade or the private pool ({hits[0][0]}: "
                                f"{hits[0][1]}): no model reads it"])
        key = _sha(role, mission, sha, _tree_sha(folder))
        cache = self.home.parent.parent / "cache" / key
        if not (cache / ".done").exists():
            self._agent("improver", role, mission, folder, "default")         # a retry is a spawn of its own
            cache.mkdir(parents=True, exist_ok=True)
            for o in outputs:
                src = folder / o
                if src.is_dir():
                    shutil.copytree(src, cache / o, dirs_exist_ok=True)
                elif src.exists():
                    shutil.copyfile(src, cache / o)
            (cache / ".done").write_text(label, encoding="utf-8")
            return False
        for o in outputs:
            src = cache / o
            if src.is_dir():
                shutil.rmtree(folder / o, ignore_errors=True)
                shutil.copytree(src, folder / o)
            elif src.exists():
                shutil.copyfile(src, folder / o)
        return True

    def propose(self, g: int, name: str, role: str, prompt: str, reference: dict, base: str) -> dict:
        from .versions import export, install
        folder = Path(tempfile.mkdtemp(prefix=f"openlab-edit-{self.id}-g{g}-{name}-"))
        export(self.root, base, folder / "harness")
        self._evidence_run(folder / "reference", reference)
        (folder / "README.md").write_text(
            "- harness/: the harness to edit, in place\n- reference/: the run made with it: brief.txt, prompt.txt, "
            "transcript.txt, signals.json, deliverable.txt and deliverable/ (what it made)\n- notes.md: yours to "
            "write\n", encoding="utf-8")
        mission = meta(prompt)
        problems: list = []
        for attempt in range(2):
            text = mission if not problems else mission + "\n\nYour previous edit was refused:\n- " + "\n- ".join(
                problems) + "\nFix exactly that, in ./harness, and update ./notes.md."
            self._cached(f"g{g}-{name}-{attempt}", role, text, folder, base, ("harness", "notes.md"))
            problems = validate(read(folder / "harness"), laws(self.root))
            if not problems:
                break
        self.state.setdefault("sessions", []).append(f"g{g}-{name}")
        if problems:
            return {"name": name, "valid": False, "problems": problems, "folder": str(folder)}
        sha = install(self.root, folder / "harness", parent=base, note=f"refine {self.id} g{g} {name}")["sha"]
        notes = (folder / "notes.md").read_text(encoding="utf-8") if (folder / "notes.md").exists() else ""
        return {"name": name, "valid": True, "sha": sha, "notes": notes, "folder": str(folder)}

    def compose(self, g: int, reference: dict, cands: list, base: str) -> dict:
        from .signals import contrast, of, table
        from .versions import diff, export, install
        folder = Path(tempfile.mkdtemp(prefix=f"openlab-compose-{self.id}-g{g}-"))
        export(self.root, base, folder / "base")
        self._evidence_run(folder / "reference", reference)
        ref_sig = [of(self.root, reference["agent"])]
        for c in cands:
            d = folder / "candidates" / c["name"]
            d.mkdir(parents=True, exist_ok=True)
            (d / "diff.patch").write_text(diff(self.root, base, c["sha"]), encoding="utf-8")
            (d / "notes.md").write_text(c.get("notes", ""), encoding="utf-8")
            self._evidence_run(d / "run", c["probe"])
            (folder / "contrast").mkdir(exist_ok=True)
            (folder / "contrast" / f"{c['name']}.txt").write_text(
                "\n".join(table(contrast(ref_sig, [of(self.root, c["probe"]["agent"])])))
                + "\n(one run a side: every difference is within noise until it repeats)\n", encoding="utf-8")
        self._cached(f"g{g}-compose", "composer", meta("compose"), folder, base, ("decision.json", "merged"))
        self.state.setdefault("sessions", []).append(f"g{g}-compose")
        try:
            decision = json.loads((folder / "decision.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            decision = {"decision": "keep_base", "why": "the composer wrote no readable decision"}
        names = {c["name"]: c["sha"] for c in cands}
        final = base
        if decision.get("decision") == "adopt" and decision.get("candidate") in names:
            final = names[decision["candidate"]]
        elif decision.get("decision") == "merge" and (folder / "merged").is_dir():
            problems = validate(read(folder / "merged"), laws(self.root))
            if problems:
                decision = {**decision, "decision": "keep_base", "refused": problems}
            else:
                final = install(self.root, folder / "merged", parent=base, note=f"refine {self.id} g{g} merged")["sha"]
        elif decision.get("decision") != "keep_base":
            decision = {**decision, "decision": "keep_base", "refused": ["not keep_base, adopt <candidate> or merge"]}
        return {"decision": decision, "final": final, "folder": str(folder)}

    def select(self, base_run: dict, delivered: list) -> dict:
        """The blind selector over the groups' deliverables, option A the base harness's (so it can abstain)."""
        folder = Path(tempfile.mkdtemp(prefix=f"openlab-select-{self.id}-"))
        from . import load
        (folder / "task.txt").write_text(unit_prompt(self.root, self.unit, load(), Path(base_run["copy"]), self.text,
                                                     self.source), encoding="utf-8")
        others = list(delivered)
        random.Random(self.id).shuffle(others)
        options = {"A": base_run, **{chr(ord("B") + i): d for i, d in enumerate(others)}}
        for letter, run in options.items():
            _write_tree(folder / "options" / letter, deliverable(self.unit, Path(run["copy"])))
        self._agent("selector", "selector", meta("select"), folder, "default", cli=self.selector_cli)
        try:
            choice = json.loads((folder / "choice.json").read_text(encoding="utf-8")).get("choice", "A")
        except (OSError, ValueError):
            choice = "A"
        pick = options.get(choice, options["A"])
        return {"choice": choice if choice in options else "A", "harness": pick["harness"], "copy": pick["copy"],
                "folder": str(folder)}

    def run(self) -> dict:
        from .versions import active_sha, default_sha
        self.state.setdefault("base", active_sha(self.root) or default_sha())
        base = self.state["base"]
        groups = []
        for g in range(self.k):
            ref = self.stage(f"g{g}-reference", lambda: self.execute(f"g{g}-reference", base))
            cands = []
            for name, role, prompt in EDITORS:
                c = self.stage(f"g{g}-propose-{name}", lambda: self.propose(g, name, role, prompt, ref, base))
                if c["valid"]:
                    c = {**c, "probe": self.stage(f"g{g}-probe-{name}",
                                                  lambda: self.execute(f"g{g}-probe-{name}", c["sha"]))}
                else:                       # no valid candidate: its rollout is a second base run (a noise check)
                    c = {**c, "probe": self.stage(f"g{g}-probe-{name}",
                                                  lambda: self.execute(f"g{g}-probe-{name}-invalid", base))}
                cands.append(c)
            comp = self.stage(f"g{g}-compose", lambda: self.compose(g, ref, [c for c in cands if c["valid"]], base))
            dl = self.stage(f"g{g}-deliver", lambda: self.execute(f"g{g}-deliver", comp["final"]))
            groups.append({"reference": ref, "candidates": cands, "compose": comp, "deliver": dl})
        chosen = groups[0]["deliver"] if self.k == 1 else self.stage(
            "select", lambda: self.select(groups[0]["reference"], [g["deliver"] for g in groups]))
        led, sessions = self.state["ledger"], self.state.get("sessions", [])
        if len(led["executor"]) != 4 * self.k or len(sessions) != 3 * self.k:     # the cost of a step is fixed
            raise HarnessError([f"the ledger holds {len(led['executor'])} rollouts and {len(sessions)} improver "
                                f"sessions; a step at k={self.k} is {4 * self.k} and {3 * self.k}"])
        report = self.stage("reflect", lambda: self.reflect(groups, chosen))
        return self.summary(groups, chosen, report)

    def reflect(self, groups: list, chosen: dict) -> dict:
        from .evaluate import tripwire
        from .reflect import spawn_reflector
        folder = Path(groups[-1]["compose"]["folder"])
        if tripwire(folder):
            return {"reflector": None, "why": "its evidence shows a grade"}
        share = self.budget["rest"] / 2
        r = spawn_reflector(self.root, chosen.get("agent") or groups[-1]["deliver"]["agent"], str(folder), share,
                            f"refine:{self.id}", label=f"refinement step {self.id}")
        self.state["ledger"]["reflector"].append({"agent": r["reflector"], "role": "reflector", "usd": round(share, 4)})
        self.save()
        return r

    def summary(self, groups: list, chosen: dict, reflection: dict) -> dict:
        led = self.state["ledger"]
        store = _store(self.root)
        spent = {k: round(sum(float(((store.get(x["agent"]) or {}).get("spent") or {}).get("usd") or 0) for x in v), 4)
                 for k, v in led.items()}
        trials = 0
        for d in self.probes.glob("*/*/.openlab/record.db") if self.probes.exists() else []:
            import sqlite3
            from contextlib import closing
            with closing(sqlite3.connect(f"file:{d}?mode=ro", uri=True)) as con:
                trials += con.execute("SELECT COUNT(*) FROM nodes WHERE kind='run'").fetchone()[0]
        out = {"step": self.id, "unit": self.unit, "k": self.k, "base": self.state["base"],
               "final": chosen["harness"], "decisions": [g["compose"]["decision"].get("decision") for g in groups],
               "selected": chosen.get("choice"), "deliverable": chosen["copy"],
               "ledger": {"executor_rollouts": len(led["executor"]), "improver_sessions": len(led["improver"]),
                          "selector": len(led["selector"]), "reflector": len(led["reflector"]),
                          "budgeted_usd": round(sum(float(x["usd"]) for v in led.values() for x in v), 4),
                          "spent_usd": spent, "probe_trials": trials},
               "reflection": reflection}
        from .versions import AGENT
        store.event(AGENT, "harness_refine_end", summary=f"refinement {self.id}: {out['decisions']} -> "
                    f"{out['final'][:12]}; {out['ledger']['executor_rollouts']} rollouts, "
                    f"${sum(spent.values()):.4f}", payload={k: out[k] for k in ("step", "unit", "k", "base", "final")})
        return out


def start(root, unit: str, source, text: str = "", k: int = 1, runner: Optional[Runner] = None,
          step: str = "", selector_cli: str = "claude") -> dict:
    """Start (or resume, by `step`) a refinement step: a person act, signed."""
    from ..study import personkey
    from .versions import AGENT
    signer = personkey.current()
    if signer is None:
        raise PermissionError("a refinement step spends the person's budget: `openlab harness refine ...` in the "
                              "person's own terminal")
    s = settings(root)
    if not 1 <= int(k) <= int(s["k_max"]):
        raise HarnessError([f"k is 1 to [harness] k_max ({s['k_max']})"])
    new = not step
    step = step or f"rf-{time.strftime('%Y%m%d-%H%M%S')}-{_sha(unit, source, text)[:6]}"
    if new:
        b = budget(root)
        msg = f"openlab-harness|refine|{step}|{unit}|{source}|{k}|{b['usd']}".encode("utf-8")
        _store(root).event(AGENT, "harness_refine", summary=f"{signer.label}: refinement {step} of {unit} (k={k}, "
                           f"${b['usd']:g})", payload={"step": step, "unit": unit, "source": str(source), "k": int(k),
                                                      "budget": b, "by": signer.label,
                                                      "sig": {"key": signer.fingerprint, "value": signer.sign(msg)}},
                           exact=True)
    return Step(root, step, unit, source, text, k, runner, selector_cli).run()
