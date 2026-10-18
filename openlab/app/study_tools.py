"""The study verbs (openlab.study) as tools, for the app, the CLI and every agent through the MCP. Each verb is a
`Study` method, which writes the verb's action node; the read-only views (study_list, study_runs, study_node,
study_nodes, study_log, study_forecasts, study_prisma, study_sources, study_evidence, study_lattice, study_tree,
study_claims, study_paper_lint, study_graph, study_trace, study_impact, study_traceability, study_frontier,
study_replications, study_reproductions) read the record, and nothing here writes it. Folders are workspace paths."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .registry import CALLER, ToolContext, ToolError, no_agent_code, tool

WHY_MAX = 200               # nodes one `study_why` reply carries; `depends_on_withheld` counts the older rest
LIST_MAX = 100              # studies one `study_list` reply names
RUNS_MAX = 200              # runs one `study_runs` reply carries at most
LOG_MAX = 500               # actions one `study_log` reply carries at most
FORECASTS_MAX = 200         # forecasts one `study_forecasts` reply carries at most
NODE_ITEMS = 40             # entries of a list or dict `study_node` shows before it counts the rest
NODE_DEPTH = 4              # nesting `study_node` shows before it counts a list or dict instead
NODE_CHARS = 2_000          # of a text `study_node` shows
SOURCES_MAX = 200           # sources one `study_sources` reply carries at most
TITLE_CHARS = 160           # of a title `study_sources` shows
PAGE_CHARS = 12_000         # of a page `study_fetch_text` shows
ERRORS_MAX = 20             # failed calls one `study_discover` reply names
EVIDENCE_MAX = 200          # evidence nodes one `study_evidence` reply carries at most
LATTICE_ROWS = 100          # protocol rows one `study_lattice` reply carries
MISSED_MAX = 50             # missed references one `study_recall` reply names
FINDINGS_MAX = 300          # bindings and findings one `study_paper_lint` reply carries at most
CLAIMS_MAX = 200            # claims one `study_claims` reply carries at most
REACHED_MAX = 100           # sources one `study_contra` reply names
FRONTIER_MAX = 100          # items one `study_frontier` reply carries at most
BROKEN_MAX = 100            # broken claims one `study_traceability` reply carries at most
WATCH_MAX = 50              # ids per list one `study_monitor` reply names
NODES_MAX = 200             # nodes one `study_nodes` reply carries at most
STATS = ("value", "lo", "hi", "diff", "p", "better", "nominal")


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


def _assign(folder: Path) -> None:
    """The acting agent's first study (the one it made, or first ran or built a paper in), when it was spawned with
    none: so its row names the study, and the release compiles when it ends done (show_tools.finished_leads). Never fails the study it was made for."""
    try:
        from ..agents.identity import current
        from ..agents.store import Store, root_of
        root = root_of(folder)
        if not (root / ".openlab" / "control.db").is_file():
            return
        st = Store(root)
        a = st.get(current())
        if a is not None and not a["study"]:
            st.update(a["id"], study=str(Path(folder).resolve().relative_to(root)))
    except Exception:                                                       # noqa: BLE001
        pass


@tool("study", mutates=True)
def study_new(ctx: ToolContext, folder: str, template: str = "temporal") -> dict:
    """Scaffold a study in a new workspace folder from a template (temporal, static, or review: a literature
    review's protocol and steps): study.toml (the plan,
    sealed by the first run), model.py, recipes/make_data.py, data/, and git."""
    from .. import study
    with _errors():
        s = study.new(ctx.path(folder), template)
        _assign(s.folder)
        return {"folder": ctx.rel(s.folder), "name": s.spec.name, "template": template,
                "next": "edit study.toml and model.py, then study_materialise and study_run"}


@tool("study", mutates=True)
def study_materialise(ctx: ToolContext, folder: str, name: str = "", refresh: bool = False) -> dict:
    """Build a study's datasets (or the one named): run the recipe when its file is missing (every recipe again on
    refresh=true; a provider is re-fetched only then) or fetch the source, store the parquet and register its sha256
    and data card. A changed file is caught on every later load; a recipe edited after its file was made is named
    in `warnings`."""
    from ..study.data import stale
    from ..study.show import brief
    no_agent_code("study_materialise")
    with _errors():
        s = _open(ctx, folder)
        out = s.materialise(name or None, refresh=bool(refresh))
        warnings = stale(s, [name] if name else None)
    return {"datasets": [brief(n) for n in out.get("datasets", [out])], **({"warnings": warnings} if warnings else {})}


@tool("study", mutates=True)
def study_run(ctx: ToolContext, folder: str, note: str = "", code: str = "", seed: Optional[int] = None,
              parent: str = "", op: str = "", refs: Optional[list] = None, cites: Optional[list] = None) -> dict:
    """A research run of model.py in the sandbox: seals study.toml on the first run, counts a trial against the
    budget (an identical run is cached, not counted) and returns the result node: metrics with intervals,
    baselines and paired differences. `trial.summary` says first whether it was cached or which trial it spent and
    which key component changed (code, data, spec, env packages, python). Never touches the holdout. `code`: your
    branch's working copy, branches/<id> (a proposal branch edits only its own copy). `seed`: the model's seed for
    this run (a paper averaged over seeds: one run per seed, each a trial; the folds and holdout keep split.seed).
    `op` and `parent`: the run's place in the search tree: draft (a new line, no parent), improve or debug (of
    `parent`, a run:<id> or its result:<id>), fuse (of `parent`, borrowing from the runs in `refs`); without them the
    run follows the branch's last run. `cites`: the lessons this run applies (lesson:<id>, or <study>/lesson:<id>
    from another study; lessons_find finds them), so each lesson's track record counts it. An agent's run still going
    after 14 minutes comes back as {running, job}: agent_wait(ids=["job:run-..."]) waits for its result."""
    no_agent_code("study_run")
    kw = dict(note=note, code=code, seed=seed, parent=parent, op=op, refs=refs or [], cites=cites or [])
    with _errors():
        if CALLER.get() == "agent":                 # its own process: the run outlives the agent (study.detached)
            from ..agents.identity import current
            from ..study import detached
            _assign(_open(ctx, folder).folder)          # refusals that need no run (not a study) come back at once
            return detached.run(ctx.path(folder), kw, current())
        s = _open(ctx, folder)
        _assign(s.folder)
        return s.run(**kw)


@tool("study", mutates=True)
def study_seal(ctx: ToolContext, folder: str, note: str = "") -> dict:
    """Seal study.toml on purpose: the study's first research run, which writes study.toml into the record as the
    plan and is counted as trial 1 (study_run does the same implicitly). Refused before anything runs when the study
    is already sealed (says when, by which plan node), is a review study, or its data is not materialised. `terms`:
    what the seal froze (the split and holdout, budget, instrument pins, primary metric and the other protected
    fields; a later edit is a study_amend, the person signs the protected ones) and what it cost. The holdout is
    never touched. An agent's seal still going after 14 minutes comes back as {running, job}, as study_run's."""
    no_agent_code("study_seal")
    with _errors():
        from ..study.errors import StudyError
        s = _open(ctx, folder)
        try:
            terms = s.seal_terms(ctx.rel(s.folder))     # refused here, before a process is started
        except StudyError:
            s.seal_run(note=note, rel=ctx.rel(s.folder))    # refused the same way, and the refusal recorded
            raise
        _assign(s.folder)
        if CALLER.get() == "agent":
            from ..agents.identity import current
            from ..study import detached
            out = detached.run(ctx.path(folder), {"note": note, "rel": ctx.rel(s.folder)}, current(), verb="seal")
        else:
            out = s.seal_run(note=note, rel=ctx.rel(s.folder))
    return {**out, "terms": terms}


SERIES_MAX = 500            # decisions one `study_series` reply carries at most


@tool("study")
def study_series(ctx: ToolContext, folder: str, result: str, columns: Optional[list] = None, n: int = 200,
                 offset: int = 0) -> dict:
    """A research result's policy series, one value per decision time `_t`, oldest first: act()'s value (`act`),
    its named values (`act.<name>`) and each policy baseline's (`act:<name>`); `columns` keeps some, `n` decisions
    from `offset` (at most 500; `more` counts the rest). No labels or predictions; a holdout result's are refused."""
    import io
    import math
    import polars as pl
    n, offset = max(1, min(int(n), SERIES_MAX)), max(0, int(offset))
    with _errors():
        from ..study.errors import StudyError
        rec = _open(ctx, folder).record
        node = rec.get(result)
        if node.kind == "result":
            node = rec.get(rec.links(node.id, rel="of")[0][2])
        if node.kind != "run":
            raise StudyError(f"{result} is a {node.kind}", "a policy series is a result's", "study_runs lists them")
        if node.payload["segment"] != "research":
            raise StudyError(f"{result} is a {node.payload['segment']} result",
                             "the held-out rows are scored once, by the person: their per-decision values stay there",
                             "read a research result's series")
        if not node.payload.get("predictions"):
            raise StudyError(f"{node.id} records no predictions")
        P = pl.read_parquet(io.BytesIO(rec.blob(node.payload["predictions"])))
        acts = [c for c in P.columns if c == "act" or c.startswith(("act.", "act:"))]
        if not acts:
            raise StudyError(f"{result} has no policy series", "model.py defined no act() when it ran",
                             "define act(pred_row, view, t) in model.py; its value is kept per decision")
        want = [c for c in acts if not columns or c in columns or c.removeprefix("act.") in columns]
        if not want:
            raise StudyError(f"no column {', '.join(map(str, columns))}", f"its series: {', '.join(acts)}")
    D = P.select("_t", *want).unique("_t", keep="first", maintain_order=True).sort("_t")
    page = D.slice(offset, n)

    def val(v):
        return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else v
    return {"result": result, "run": node.id, "decisions": D.height, "offset": offset,
            "more": max(0, D.height - offset - page.height), "columns": want,
            "t": [str(t) if not isinstance(t, (int, float)) else t for t in page["_t"].to_list()],
            **{c: [val(v) for v in page[c].to_list()] for c in want}}


@tool("study")
def study_search(ctx: ToolContext, folder: str) -> dict:
    """The search over the study's runs, scored now: the clock t (trials and minutes used), the exploration weight
    c(t), the chance w(t) of the UCT pick against an elite one, and per run its Q, U, cost factor, UCT and P(pick);
    stagnant lines. study_frontier turns it into expand / fuse / draft / select items."""
    with _errors():
        sc = _open(ctx, folder).search()
        return {**{k: sc[k] for k in ("t", "c", "w", "best", "stagnant", "since_best")},
                "rows": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in sc["rows"]]}


@tool("study", mutates=True)
def study_select(ctx: ToolContext, folder: str) -> dict:
    """Apply the study's sealed selection rule ([selection] rule: best_val or seed_robust) and record what it
    picked among N, with its value adjusted for being the best of N. The holdout look is then on that run only."""
    with _errors():
        return _open(ctx, folder).select()


@tool("study", mutates=True)
def study_lesson(ctx: ToolContext, folder: str, rule: str, direction: str, basis: str, tags: list,
                 margin: Optional[float] = None) -> dict:
    """A lesson from two runs: `rule` (what to do), `direction` improves | hurts | no_gain | fixes, `basis` a
    comparison:<id> of two of the study's results (study_compare; improves needs the first side better with an
    interval excluding 0, hurts the second, no_gain the interval inside ±margin) or, for fixes, the run:<id> that
    fixed its failed parent; `tags` scope it for the next study. Refused when the basis does not show it."""
    with _errors():
        return _open(ctx, folder).lesson(rule, direction, basis, tags, margin)


@tool("study", mutates=True)
def study_check(ctx: ToolContext, folder: str, code: str = "") -> dict:
    """Module checks of the study's code (or a branch's copy, code="branches/<id>") in seconds: lint (no path a run
    must not read, no network), import (Model.fit and Model.predict), smoke (the runner's own job on a little data)
    and determinism. Recorded; a run of exactly this code is refused while its latest check is red."""
    with _errors():
        return _open(ctx, folder).check(code)


@tool("study", mutates=True)
def study_ensemble(ctx: ToolContext, folder: str, members: list, method: str = "mean") -> dict:
    """Combine recorded research runs (run or result ids, one protocol) into a run of its own, counted as a trial:
    method mean, rank (probabilities) or weighted (weights cross-fitted on halves of the rows). The same members and
    method again are cached."""
    with _errors():
        return _open(ctx, folder).ensemble(members, method)


@tool("study", mutates=True)
def study_critique(ctx: ToolContext, folder: str, on: str, model: str, prompt_sha: str, findings: list,
                   cost_usd: Optional[float] = None) -> dict:
    """A reviewer's findings on a run, result, claim or paper (`on`), recorded as advice: findings
    [{where: "model.py:57", text: "..."}]; `model` and `prompt_sha` say who wrote it. Nothing reads it into a verdict;
    answer each finding with study_finding."""
    with _errors():
        return _open(ctx, folder).critique(on, model, prompt_sha, findings, cost_usd)


@tool("study", mutates=True)
def study_finding(ctx: ToolContext, folder: str, critique: str, index: int, action: str, reason: str = "",
                  by: str = "") -> dict:
    """Answer finding `index` of a critique: action checked (by: the module_check, run, comparison or test that
    settled it) or dismissed (reason: why). The critique's track record is computed from these answers."""
    with _errors():
        return _open(ctx, folder).finding(critique, index, action, reason, by)


@tool("study")
def lessons_find(ctx: ToolContext, query: str, folder: str = "", n: int = 10) -> dict:
    """The workspace's lessons ranked for `query` (words of what you are about to try), each with its study, what
    it rests on and its track record (runs that cited it, and how many did better than their parent). Cite the ones
    you apply in study_run(cites=[...])."""
    from ..study.lessons import find
    with _errors():
        here = _open(ctx, folder).folder.name if folder else ""
        return {"lessons": find(ctx.root, query, n, here=here)}


@tool("study", mutates=True)
def study_branch(ctx: ToolContext, folder: str, branch: str) -> dict:
    """A working copy of the study's model.py (and the study's modules it imports) in branches/<branch>/, to edit
    and run as its own code (study_run code="branches/<branch>") without moving the study's: an experimenter's
    variant, an adversary's attack. An existing copy is kept as it is (it is that branch's work)."""
    from ..study import branches
    from ..study.errors import StudyError
    with _errors():
        s = _open(ctx, folder)
        if not branches._ID.fullmatch(branch):
            raise StudyError(f"branch {branch!r}: 1-24 lowercase letters, digits, - or _, starting with a letter")
        r = branches.make(s, branch)
    return {**r, "next": f"edit {folder}/{r['dir']}/model.py, then study_run(folder, code=\"{r['dir']}\")"}


@tool("study", mutates=True)
def study_promote(ctx: ToolContext, folder: str, branch: str) -> dict:
    """Copy a branch's working copy (branches/<id>) over the study's own model.py and modules; recorded."""
    no_agent_code("study_promote")
    with _errors():
        return _open(ctx, folder).promote(branch)


@tool("study")
def study_status(ctx: ToolContext, folder: str) -> dict:
    """A study's summary: its question, the seal and amendments, datasets (with their node id and time and key
    coverage), trials and holdout looks used of their budgets, the last result, and how many nodes of each kind
    the record holds (`kinds`), and the runs in flight (`running`)."""
    with _errors():
        s = _open(ctx, folder)
        out = s.status()
        for d in out["datasets"]:
            n = s.record.last("dataset", name=d["name"])
            d.update(id=n.id, coverage={k: n.payload["card"].get(k) for k in ("time", "key")})
        return {**out, "kinds": s.record.kinds(), "running": _live(s.folder)}


def _stats(d, metric: str):
    """One metric's numbers ({value, lo, hi} or a paired {diff, lo, hi, p, better}) from a dict keyed by metric,
    or from the dict itself when it already holds one metric's."""
    if isinstance(d, dict) and isinstance(d.get(metric), dict):
        d = d[metric]
    if not isinstance(d, dict):
        return None
    return {k: d[k] for k in STATS if k in d} or None


def _audit(a: dict) -> str:
    """The truncation audit in one word: its verdict (e.g. clean, look-ahead), or not audited."""
    return a.get("verdict") or {True: "clean", False: "look-ahead"}.get(a.get("ok")) or "not audited"


def _summary(ctx: ToolContext, folder) -> dict:
    from .. import study
    from ..study import spec as _spec
    from ..study.errors import StudyError
    row = {"folder": ctx.rel(folder)}
    try:
        if not (folder / ".openlab" / "record.db").is_file():          # listing a study must not create its record
            sp = _spec.load(folder)
            return {**row, "name": sp.name, "question": sp.question, "sealed": False, "runs": 0, "last": None}
        s = study.open(folder)
        rec, sp = s.record, s.spec
        last = rec.last("result")
        if last is not None:
            run = rec.get(rec.links(last.id, rel="of")[0][2])
            primary = sp.evaluate["metrics"][0]
            last = {"metric": primary, **(_stats(last.payload["metrics"], primary) or {}),
                    "segment": run.payload["segment"], "at": last.at}
        return {**row, "name": sp.name, "question": sp.question, "sealed": bool(rec.find("plan")),
                "runs": len(rec.find("run")), "last": last}
    except StudyError as e:
        return {**row, "error": e.error, "meaning": e.meaning}


@tool("study")
def study_list(ctx: ToolContext) -> dict:
    """The studies in the workspace (every folder with a study.toml): name, question, whether its spec is sealed,
    the runs on record and the last result's primary metric. A study whose study.toml does not load says why."""
    from .tools import _walk
    files, _ = _walk(ctx, suffixes={".toml"})
    folders = sorted(p.parent for p in files if p.name == "study.toml")
    out = {"studies": [_summary(ctx, f) for f in folders[:LIST_MAX]]}
    if len(folders) > LIST_MAX:
        out["studies_withheld"] = len(folders) - LIST_MAX
    return out


def _run_row(rec, run, primary: str) -> dict:
    p = run.payload
    parent = rec.links(run.id, rel="branches_from")
    made = rec.links(run.id, rel="of", incoming=True)
    row = {"id": run.id, "at": run.at, "actor": run.actor, "segment": p["segment"], "cached": p["cached"],
           "audit": _audit(p["audit"]), "seconds": p["seconds"], "note": p.get("note", ""),
           "branch": p.get("branch") or None,
           "branches_from": parent[0][2] if parent else None, "result": None}
    if made:
        res = rec.get(made[0][0])
        r = res.payload
        row["result"] = {"id": res.id, "metric": _stats(r["metrics"], primary),
                         "baselines": {b: _stats(v, primary) for b, v in r.get("baselines", {}).items()},
                         "paired": {b: _stats(v, primary) for b, v in r.get("paired", {}).items()}}
    return row


@tool("study")
def study_runs(ctx: ToolContext, folder: str, n: int = 50) -> dict:
    """A study's runs, newest first: segment, cached, the truncation audit's verdict, seconds, note, the run it
    branches from, and its result's primary metric with interval, each baseline's, and the paired differences
    against each baseline. `selected_from`: the trials a result chosen among these was selected from (with the
    studies sharing this one's ledger). `study_node` on a result id gives every metric. `running`: the runs in
    flight, each with its job id, who started it, when, and its last progress."""
    from ..study.metrics import METRICS
    n = max(1, min(n, RUNS_MAX))
    with _errors():
        s = _open(ctx, folder)
        rec, sp = s.record, s.spec
        primary = sp.evaluate["metrics"][0]
        runs = rec.find("run")
        rows = [_run_row(rec, r, primary) for r in reversed(runs[-n:])]
        from ..study.ledger import shared
        trials = shared(s.folder, rec)["trials"]
    m = METRICS.get(primary)
    return {"primary": primary, "higher_is_better": m.higher_is_better if m else None,
            "direction": m.direction if m else "none", "nominal": m.nominal if m else None, "selected_from": trials,
            "baselines": list(sp.evaluate.get("baselines", [])), "runs": rows, "total": len(runs),
            "running": _live(s.folder)}


def _live(folder) -> list:
    """The study's runs in flight: an agent's detached run, with who started it, when, and its last progress."""
    from ..study.detached import live
    return live(folder)


def _trim(v, depth: int = 0):
    """A payload cut to size: long text, lists and dicts keep their head and count what was left out."""
    if isinstance(v, str):
        return v if len(v) <= NODE_CHARS else f"{v[:NODE_CHARS]}... ({len(v) - NODE_CHARS} more chars)"
    if isinstance(v, (list, dict)) and depth >= NODE_DEPTH:
        return f"({len(v)} {'items' if isinstance(v, list) else 'keys'})"
    if isinstance(v, list):
        head = [_trim(x, depth + 1) for x in v[:NODE_ITEMS]]
        return head + [f"... {len(v) - NODE_ITEMS} more"] if len(v) > NODE_ITEMS else head
    if isinstance(v, dict):
        head = {k: _trim(x, depth + 1) for k, x in list(v.items())[:NODE_ITEMS]}
        if len(v) > NODE_ITEMS:
            head["..."] = f"{len(v) - NODE_ITEMS} more keys"
        return head
    return v


@tool("study")
def study_node(ctx: ToolContext, folder: str, id: str) -> dict:
    """One record node: id, kind, actor, time, its body (long text, lists and dicts cut, counting what was left
    out) and the links it carries."""
    with _errors():
        rec = _open(ctx, folder).record
        n = rec.get(id)
        links = [{"rel": rel, "to": dst} for _, rel, dst in rec.links(id)]
    return {"id": n.id, "seq": n.seq, "kind": n.kind, "actor": n.actor, "at": n.at, "payload": _trim(n.payload),
            "links": links}


@tool("study")
def study_nodes(ctx: ToolContext, folder: str, kind: str, n: int = 50) -> dict:
    """The record's nodes of one kind (e.g. test, assumption, break, dead_end, watch, retraction, paper), newest
    first, each as study_node gives it (its body cut to size, the links it carries; `links_withheld` counts those
    past the first 40)."""
    n = max(1, min(n, NODES_MAX))

    def row(x) -> dict:
        links = [{"rel": rel, "to": dst} for _, rel, dst in rec.links(x.id)]
        return {"id": x.id, "seq": x.seq, "kind": x.kind, "actor": x.actor, "at": x.at, "payload": _trim(x.payload),
                "links": links[:NODE_ITEMS], **({"links_withheld": len(links) - NODE_ITEMS}
                                               if len(links) > NODE_ITEMS else {})}
    with _errors():
        rec = _open(ctx, folder).record
        found = rec.find(kind)
        rows = [row(x) for x in reversed(found[-n:])]
    return {"nodes": rows, "total": len(found)}


@tool("study")
def study_log(ctx: ToolContext, folder: str, n: int = 100) -> dict:
    """The study's flight recorder, newest first: one line per verb run on it (by an agent, a person or openlab),
    with what it did or why it was refused."""
    from ..study.show import brief
    n = max(1, min(n, LOG_MAX))
    with _errors():
        acts = _open(ctx, folder).record.find("action")
    return {"actions": [{**brief(asdict(a)), "verb": a.payload["verb"], "ok": bool(a.payload["outcome"].get("ok"))}
                        for a in reversed(acts[-n:])], "total": len(acts)}


@tool("study")
def study_why(ctx: ToolContext, folder: str, id: str) -> dict:
    """What a record node (e.g. a result id) depends on, oldest first: each node's id, kind, actor, time and a
    one-line summary."""
    from ..study.show import brief
    with _errors():
        nodes = _open(ctx, folder).why(id)
    out = {"id": id, "depends_on": [brief(n) for n in nodes[-WHY_MAX:]]}
    if len(nodes) > WHY_MAX:
        out["depends_on_withheld"] = len(nodes) - WHY_MAX
    return out


def _amend(ctx: ToolContext, folder: str, reason: str, person: bool) -> dict:
    from ..study.show import brief
    with _errors():
        return brief(_open(ctx, folder).amend(reason, person=person))


@tool("study", mutates=True)
def study_amend(ctx: ToolContext, folder: str, reason: str) -> dict:
    """Record the edit of study.toml since the seal as an amendment, with the reason. Split, budget, sandbox,
    the primary metric and known_at need a person's amendment (study_amend_person)."""
    return _amend(ctx, folder, reason, person=False)


@tool("study")
def study_verify(ctx: ToolContext, folder: str) -> dict:
    """Recompute the study record's hash chain: ok, the node count, the head, and the first edited seq if any."""
    with _errors():
        return _open(ctx, folder).verify()


@tool("study", mutates=True)
def study_forecast(ctx: ToolContext, folder: str, model_version: int = 0) -> dict:
    """A forward forecast: refreshes the data when [schedule] refresh = true, takes the latest decision time (after
    the study's segments), predicts with the pinned registered model ([forecast] model, or `model_version`) in the
    sandbox, checks the [forecast] gate and writes forecasts/<t>.csv with a signed .sig.json for a program downstream.
    One per decision and model; a failed gate is recorded and signed as failed. Returns the forecast node."""
    no_agent_code("study_forecast")
    with _errors():
        return _open(ctx, folder).forecast(model_version=model_version or None, actor="agent")


@tool("study", mutates=True)
def study_fill(ctx: ToolContext, folder: str) -> dict:
    """Record the outcome of every forecast whose label is now known (its label time has passed and the data has
    caught up): the spec's metrics on its rows, once per forecast."""
    no_agent_code("study_fill")
    with _errors():
        return _open(ctx, folder).fill(actor="agent")


def _forecast_row(rec, f) -> dict:
    p = f.payload
    made = rec.links(f.id, rel="of", incoming=True)
    row = {"id": f.id, "at": f.at, "actor": f.actor, "t": p["t"], "model": f"{p['model']['name']} v{p['model']['version']}",
           "rows": p["rows"], "gate": p["gate"]["passed"], "failed": [c for c in p["gate"]["checks"] if not c["ok"]],
           "prospective": p["prospective"], "label_at": p["label_at"], "file": p["file"], "signed": p["signed"],
           "expires": p["expires"], "outcome": None}
    if made:
        o = rec.get(made[0][0])
        row["outcome"] = {"id": o.id, "at": o.at, "n": o.payload["n"], "metrics": o.payload["metrics"]}
    return row


@tool("study")
def study_forecasts(ctx: ToolContext, folder: str, n: int = 50) -> dict:
    """A study's forward forecasts, newest first: decision time, model version, rows, whether the gate passed (and
    the checks that failed), whether it was made before its label could be known (prospective), the signed file and
    its expiry, and the outcome's metrics once filled."""
    n = max(1, min(n, FORECASTS_MAX))
    with _errors():
        rec = _open(ctx, folder).record
        found = rec.find("forecast")
        rows = [_forecast_row(rec, f) for f in reversed(found[-n:])]
    return {"forecasts": rows, "total": len(found)}


@tool("study", mutates=True)
def study_schedule(ctx: ToolContext, folder: str, action: str = "status") -> dict:
    """The study's [schedule] cron line in the user's crontab: `status` (default: declared, installed, matching,
    next run, last forecast), `on` (install or update) or `off`. Only the study's own tagged line is touched; it runs
    `openlab study forecast`, then `openlab study fill`, serialised with flock and logged to .openlab/cron.log."""
    if action == "on":
        no_agent_code("study_schedule")
    with _errors():
        return _open(ctx, folder).schedule(action, actor="agent")


@tool("study", mutates=True, human=True)
def study_amend_person(ctx: ToolContext, folder: str, reason: str) -> dict:
    """A person's amendment of study.toml, which may change split, budget, sandbox, the primary metric and
    known_at. A person's decision: agents cannot call it."""
    return _amend(ctx, folder, reason, person=True)


@tool("study", mutates=True)
def study_discover(ctx: ToolContext, folder: str) -> dict:
    """Run the sealed [review] protocol's searches (sealing study.toml on first use): every query on every index
    it names, then `citation_hops` hops of references and citing works. Records each query, a source per new
    identity (a duplicate is refused by the record) with its retraction flags, and a coverage node judged against
    the protocol's coverage floor. Returns the coverage and counts; study_sources lists the sources."""
    with _errors():
        out = _open(ctx, folder).discover()
    if len(out["errors"]) > ERRORS_MAX:
        out = {**out, "errors": out["errors"][:ERRORS_MAX], "errors_withheld": len(out["errors"]) - ERRORS_MAX}
    t = out.get("timings") or []
    queries = [x for x in t if x["kind"] == "queries"]
    slow = sorted((x for x in t if x["kind"] != "queries"), key=lambda x: -x["seconds"])[:5]
    return {**out, "timings": queries + slow, "timings_withheld": len(t) - len(queries) - len(slow)}


@tool("study", mutates=True)
def study_fetch_text(ctx: ToolContext, folder: str, source: str, page: int = 0, layout: bool = False) -> dict:
    """A source's open-access full text, split by page and kept in the record (fetched once). With `page` (from
    1) the reply carries that page's text, to quote from in study_screen or study_extract; `layout` gives the
    page as laid out, with table rows intact."""
    import json
    with _errors():
        s = _open(ctx, folder)
        d = s.fetch_text(source)
        out = {"id": d["id"], "source": source, "n_pages": d["payload"]["n_pages"], "url": d["payload"]["url"]}
        if page:
            if layout and not d["payload"].get("layout"):
                raise ToolError("this document has no layout text", "it was fetched before openlab kept one")
            pages = json.loads(s.record.blob(d["payload"]["layout" if layout else "pages"]))
            if not 1 <= page <= len(pages):
                raise ToolError(f"no page {page}", f"the document has pages 1..{len(pages)}")
            text = pages[page - 1]
            out.update(page=page, text=text[:PAGE_CHARS], **({"text_withheld": len(text) - PAGE_CHARS}
                                                             if len(text) > PAGE_CHARS else {}))
    return out


@tool("study", mutates=True)
def study_find(ctx: ToolContext, folder: str, source: str, text: str, around: int = 0, n: int = 10) -> dict:
    """Where a source's full text says a phrase or number: each hit's page and the exact sentence(s), a quote
    study_extract accepts as is (the text is fetched first if needed)."""
    with _errors():
        return _open(ctx, folder).find(source, text, around, max(1, min(n, 50)))


def _screen(ctx: ToolContext, folder, source, decision, criterion, rationale, quote, person: bool) -> dict:
    from ..study.show import brief
    with _errors():
        out = _open(ctx, folder).screen(source, decision, criterion, rationale, quote, person=person)
    return {**brief(out), **({"conflict": out["conflict"]} if out.get("conflict") else {})}


@tool("study", mutates=True)
def study_screen(ctx: ToolContext, folder: str, source: str = "", decision: str = "", criterion: str = "",
                 rationale: str = "", quote: str = "", decisions: Optional[list] = None) -> dict:
    """Screen a source under the sealed review, or several: `decisions` = [{source, decision, criterion, rationale,
    quote}, ...]. `decision`: include (an include criterion id), exclude (an exclude criterion id) or maybe (hands
    it to a person). `quote` (required): 20+ characters openlab finds in its abstract or full text. Each decision
    names you; agents that disagree on a source make a conflict a person resolves."""
    if decisions:
        if source:
            raise ToolError("give one source or decisions, not both")
        with _errors():
            return _open(ctx, folder).screen_many(list(decisions))
    if not source:
        raise ToolError("a screening decision needs a source", "or a list of decisions", "study_sources lists them")
    return _screen(ctx, folder, source, decision, criterion, rationale, quote, person=False)


@tool("study", mutates=True, human=True)
def study_screen_person(ctx: ToolContext, folder: str, source: str, decision: str, criterion: str, rationale: str,
                        quote: str = "") -> dict:
    """A person's screening decision (include or exclude), which resolves an agent's maybe and stands over any
    agent decision. A person's decision: agents cannot call it."""
    return _screen(ctx, folder, source, decision, criterion, rationale, quote, person=True)


@tool("study")
def study_prisma(ctx: ToolContext, folder: str) -> dict:
    """The review's PRISMA flow, derived from the record: records identified per index and by citation search,
    duplicates removed, screened, excluded by criterion, waiting for a person, included, and retracted sources;
    with the sealed review's include and exclude criteria ({id, text})."""
    from ..study import spec as _spec
    with _errors():
        s = _open(ctx, folder)
        sealed = _spec.sealed(s.record)
        review = sealed.review if sealed else {}
        return {**s.prisma(), "criteria": {d: list(review.get(d, [])) for d in ("include", "exclude")}}


@tool("study")
def study_sources(ctx: ToolContext, folder: str, status: str = "all", n: int = 30) -> dict:
    """A study's sources, oldest first: id, one id key, title, year, version, trust tier, retraction flag, the
    standing decision, whose, its criterion, rationale and quote, and whether its full text is fetched. `status`:
    all, unscreened, include, exclude, maybe (waiting for a person) or retracted. study_node on an id gives its
    abstract."""
    n = max(1, min(n, SOURCES_MAX))
    with _errors():
        rows = _open(ctx, folder).sources(status)
    return {"sources": [{**r, "title": r["title"][:TITLE_CHARS], "quote": _trim(r["quote"])} for r in rows[:n]],
            "total": len(rows)}


@tool("study", mutates=True)
def study_extract(ctx: ToolContext, folder: str, source: str, page: int, quote: str, fields: dict, section: str,
                  kind: str = "", units: Optional[dict] = None, header: str = "") -> dict:
    """Record evidence from a source's fetched full text (study_fetch_text first): `quote` from page `page` (from
    1) as that page has it, `section` (e.g. "Table 2"), `kind` text, table or figure (default: from the section),
    and `fields`: values of the sealed [review.extract] schema read from the quote, e.g. {"method": "EMOS", "crps":
    0.9}. `units`: how the paper writes a number, {field: "percent" or "fraction"}, e.g. {"coverage90": "percent"}
    when the table's header says %; a number the quote writes with % is percent without it. Refused unless
    the quote is on that page, each field is declared with its type, each number is written in the quote and each
    text value is in it (case aside) unless declared free. A table whose methods are its columns: give `header`,
    the header line as laid out; a text value (the method) may be in it when each number sits under its column."""
    from ..study.show import brief
    with _errors():
        return brief(_open(ctx, folder).extract(source, page, quote, fields, kind, section, units, header))


@tool("study")
def study_evidence(ctx: ToolContext, folder: str, source: str = "", n: int = 50, table: bool = False) -> dict:
    """The evidence on record, newest first, or one source's: id, the source and its id key, page, section, kind,
    the quote, the field values and valid_from (the source's publication date). `table`: the evidence matrix
    instead (cite, year, method, dataset, metric, value, page, quote, agent)."""
    n = max(1, min(n, EVIDENCE_MAX))
    if table:
        with _errors():
            rows = _open(ctx, folder).evidence_table()
        return {"table": rows[:n], "total": len(rows)}
    with _errors():
        rows = _open(ctx, folder).evidence(source)
    return {"evidence": [{**r, "quote": _trim(r["quote"])} for r in reversed(rows[-n:])], "total": len(rows)}


@tool("study", mutates=True)
def study_compare(ctx: ToolContext, folder: str, a: str, b: str, metric: str) -> dict:
    """Compare two results (`a` against `b`) on a metric of openlab.study.metrics (e.g. crps). Two of the study's
    own results (variants) need no [review]: paired on the rows both scored, with diff, interval and p, refused
    unless their runs share the sealed data, target and split. Evidence is refused, naming the coordinates, unless
    both sides carry the same value of every [review] comparable_on coordinate. The verdict (better, worse, equal)
    follows the metric's direction."""
    with _errors():
        n = _open(ctx, folder).compare(a, b, metric)
    return {"id": n["id"], **n["payload"]}


@tool("study")
def study_lattice(ctx: ToolContext, folder: str, by: str = "method") -> dict:
    """The comparison lattice, derived from the evidence: a row per combination of the review's coordinates, a
    column per value of the text field `by`, each cell the numbers read for it, and each row's gaps (columns no
    evidence fills: not found in the extracted sources, which is not the same as novel)."""
    with _errors():
        out = _open(ctx, folder).lattice(by)
    if len(out["rows"]) > LATTICE_ROWS:
        out = {**out, "rows": out["rows"][:LATTICE_ROWS], "rows_withheld": len(out["rows"]) - LATTICE_ROWS}
    return out


@tool("study", mutates=True)
def study_recall(ctx: ToolContext, folder: str, gold: str, since: str = "") -> dict:
    """Recall of the study's sources against a gold set: the references (OpenAlex) of a published review given
    by its DOI, published from `since` (default: the review's) up to the review's date. Records it; the reply
    adds how many matches were screened include and the references missed."""
    with _errors():
        out = _open(ctx, folder).recall(gold, since or None)
    if len(out["missed"]) > MISSED_MAX:
        out = {**out, "missed": out["missed"][:MISSED_MAX], "missed_withheld": len(out["missed"]) - MISSED_MAX}
    return out


@tool("study", mutates=True)
def study_paper(ctx: ToolContext, folder: str, draft: bool = False) -> dict:
    r"""Build the study's paper: paper/paper.tex, one LaTeX document (study_paper_init writes it from the template),
    or the older paper/paper.toml with paper/sections/<name>.tex. Its numbers, tables, figures, claims, citations and
    equations are bound to the record (\lab{<id>#<path>}, \labtab{<result>}, \labfig{<result>}, \labclaim{<claim>},
    \labcite{<source>}{<quote>}, \labeq{<name>}{<module.fn>}{<latex>}); \labsection{<name>} puts in a section openlab
    writes (data, methods, literature, results, limitations, reproducibility, checklist). Any package may be loaded,
    and any image of the study's folder included. `draft`: render it as it stands into paper/build/draft.pdf and list
    the findings, recording nothing. Without it, refused, listing the findings, while a binding is unresolved or
    stale, a number is typed in prose, the wording outruns the claims, a quote is not in its source or an equation
    disagrees with its function (only a person can waive a finding); else builds paper/build/main.pdf, records the
    Paper node and signs the bundle."""
    no_agent_code("study_paper")
    with _errors():
        s = _open(ctx, folder)
        _assign(s.folder)
        if draft:
            from ..study.paper.build import draft as render
            r = render(s)
            return {**r, "draft": f"{folder}/{r['draft']}" if r["draft"] else ""}
        n = s.paper()
    p = n["payload"]
    return {"id": n["id"], "pdf": f"{folder}/paper/build/main.pdf", "signed": f"{folder}/{p['signed']}",
            "bindings": len(p["bindings"]), "waived": [f["id"] for f in p["lint"]], "figures": list(p["figures"])}


@tool("study", mutates=True)
def study_paper_init(ctx: ToolContext, folder: str) -> dict:
    """Start the paper as one LaTeX document, paper/paper.tex, from openlab's template: the abstract and the usual
    headings (introduction, data, methods, results, discussion, limitations, reproducibility). A paper in the older
    layout (paper.toml and section files) is carried over into it. Refused when paper.tex exists."""
    from ..study.paper.build import init
    with _errors():
        s = _open(ctx, folder)
        _assign(s.folder)
        r = init(s)
    return {**r, "file": f"{folder}/{r['file']}"}


@tool("study")
def study_paper_lint(ctx: ToolContext, folder: str, n: int = 100) -> dict:
    """Check the paper without building it: each binding (resolved, unresolved or stale) and each finding of the
    typed-number and wording lints and the citation and equation gates, with its id, rule, location and message,
    and whether a person waived it. `ok` is whether study_paper would pass them."""
    no_agent_code("study_paper_lint")               # \labeq runs model.py's functions in the sandbox
    with _errors():
        out = _open(ctx, folder).paper_lint()
    n = max(1, min(n, FINDINGS_MAX))
    for k in ("bindings", "findings"):
        if len(out[k]) > n:
            out = {**out, k: out[k][:n], f"{k}_withheld": len(out[k]) - n}
    return out


def _organise(ctx: ToolContext, folder: str, verb: str, *args, **kw) -> dict:
    """A claims verb of the study; the reply is the node written, its id and its body."""
    with _errors():
        n = getattr(_open(ctx, folder), verb)(*args, **kw)
    return {"id": n["id"], **n["payload"]}


@tool("study", mutates=True)
def study_claim(ctx: ToolContext, folder: str, statement: str, state: str, rests_on: Optional[list] = None,
                value: Optional[float] = None, expression: str = "", inputs: Optional[dict] = None, tests: str = "",
                searched: str = "", needs: Optional[list] = None, answers: Optional[list] = None) -> dict:
    """A claim the paper may print. `state` and what it rests on: observed (`rests_on` evidence or results, whose
    run's truncation audit is clean), derived (`rests_on` claims, results or evidence, `expression` LaTeX over
    `inputs` {name: "<node-id>#<dotted path>"} written {name}, and `value`, which the LaTeX engine must reproduce),
    inferred (`rests_on` claims), hypothesized (`tests` a hypothesis id) or unknown (`searched` a coverage id). An
    observed `value` needs its one input. Optional: `needs` assumption ids, `answers` question or hypothesis ids.
    openlab computes the verdict and level (reproduced once every result it rests on has a matched reproduction,
    else measured from the holdout, recorded from research): neither is typed."""
    return _organise(ctx, folder, "claim", statement, state, rests_on or [], value, expression, inputs or {}, tests,
                     searched, needs or [], answers or [])


@tool("study", mutates=True)
def study_question(ctx: ToolContext, folder: str, text: str, part_of: str = "", out_of_scope: str = "") -> dict:
    """A question of the study's question tree, optionally `part_of` a broader question; `out_of_scope` says why
    it is recorded but not pursued. Its status is derived (study_tree)."""
    return _organise(ctx, folder, "question", text, part_of, out_of_scope)


@tool("study", mutates=True)
def study_hypothesis(ctx: ToolContext, folder: str, statement: str, prediction: str, falsified_by: str,
                     question: str, tags: list, mechanism: str = "", differs_from: Optional[list] = None,
                     difference: str = "") -> dict:
    """A falsifiable hypothesis under a question: its prediction, what would falsify it, the mechanism and `tags`.
    Refused while a dead end shares a tag with it, unless it lists that dead end in `differs_from` and says in
    `difference` what is different now."""
    return _organise(ctx, folder, "hypothesis", statement, prediction, falsified_by, question, tags, mechanism,
                     differs_from or [], difference)


@tool("study", mutates=True)
def study_test(ctx: ToolContext, folder: str, of: str, result: str, baseline: str, alpha_j: float,
               family: str = "primary") -> dict:
    """Test a claim or hypothesis (`of`) on a result's paired difference against `baseline` (a baseline's name, or
    result:<id> for another of the study's results, paired on their shared rows), at level `alpha_j`,
    spending the error budget (alpha-investing): p is the result's; a rejection earns [inference] payout, otherwise
    the test costs alpha_j / (1 - alpha_j). alpha_j may be at most wealth / (1 + wealth); a spent budget refuses.
    One test per target, result and baseline."""
    return _organise(ctx, folder, "test", of, result, baseline, alpha_j, family)


@tool("study", mutates=True)
def study_assume(ctx: ToolContext, folder: str, statement: str, rests_on: str, breaks_if: str) -> dict:
    """An assumption claims may need (study_claim `needs`): what it rests on and the test that could break it."""
    return _organise(ctx, folder, "assume", statement, rests_on, breaks_if)


@tool("study", mutates=True)
def study_break(ctx: ToolContext, folder: str, assumption: str, reason: str, shown_by: list) -> dict:
    """Record that an assumption does not hold, `shown_by` result or evidence ids: every claim that needs it, and
    every claim resting on those, is then at risk (study_claims)."""
    return _organise(ctx, folder, "break_", assumption, reason, shown_by)


@tool("study", mutates=True)
def study_dead_end(ctx: ToolContext, folder: str, of: str, statement: str, reason: str, tags: list,
                   shown_by: list) -> dict:
    """A falsified hypothesis or abandoned run (`of`; in a replication also the target, a quoted evidence, the
    source, a fidelity or replication: a dropped reading of the paper), with the reason and what showed it
    (`shown_by` result, evidence, test, comparison, fidelity, replication, source or document ids). It carries the hypothesis's tags, so a later hypothesis sharing one must say how it
    differs."""
    return _organise(ctx, folder, "dead_end", of, statement, reason, tags, shown_by)


@tool("study", mutates=True)
def study_contra(ctx: ToolContext, folder: str, claim: str, queries: Optional[list] = None,
                 found: Optional[list] = None) -> dict:
    """Search for what contradicts a claim. With `queries` openlab runs them on the sealed review's indexes (and
    citation hops from the sources the claim quotes), records them and judges their coverage against the review's
    floor; the reply names the sources reached, to read and extract from. Without queries it reuses the claim's last
    search, to record the evidence `found` against it. State: found (evidence linked), none_found (a sufficient
    search found nothing) or insufficient."""
    with _errors():
        out = _open(ctx, folder).contra(claim, queries or [], found or [])
    s = out.get("search")
    if s and len(s["sources"]) > REACHED_MAX:
        s = {**s, "sources": s["sources"][:REACHED_MAX], "sources_withheld": len(s["sources"]) - REACHED_MAX}
        out = {**out, "search": s}
    return out


@tool("study")
def study_tree(ctx: ToolContext, folder: str) -> dict:
    """The question tree, derived from the record: each question with its sub-questions and hypotheses, and each
    node's status (answered, refuted, inconclusive, untested, out_of_scope). A parent is no better than its weakest
    in-scope child; a hypothesis's status comes from its tests."""
    with _errors():
        return _open(ctx, folder).tree()


@tool("study")
def study_claims(ctx: ToolContext, folder: str, n: int = 100) -> dict:
    """The claims, oldest first: state, verdict now (tests written since count), level, value, the broken
    assumptions that put it at risk, a person's last decision on it and its last contradiction search."""
    n = max(1, min(n, CLAIMS_MAX))
    with _errors():
        rows = _open(ctx, folder).claims()
    return {"claims": [{**r, "statement": _trim(r["statement"])} for r in rows[:n]], "total": len(rows)}


@tool("study", mutates=True, human=True)
def study_approve(ctx: ToolContext, folder: str, on: str, decision: str, reason: str) -> dict:
    """A person's decision on a claim or paper: approve, reject or waive, with the reason; on evidence, a waiver of
    its source's retraction. A person's decision: agents cannot call it."""
    return _organise(ctx, folder, "approve", on, decision, reason, person=True)


@tool("study", mutates=True, human=True)
def study_holdout(ctx: ToolContext, folder: str, reason: str) -> dict:
    """A person's holdout look: recorded with the study's hashes before anything is scored, counted against the
    look budget, then the holdout run; returns its result. A person's decision: agents cannot call it."""
    with _errors():
        return _open(ctx, folder).holdout(reason, person=True)


@tool("study")
def study_graph(ctx: ToolContext, folder: str, id: str, depth: int = 1) -> dict:
    """The record around a node, for a graph view: the nodes within `depth` links (1 to 3, either direction; action
    nodes left out, study_log has them; at most 200, `withheld` counts the rest), each with a one-line summary, and
    the typed edges between them."""
    with _errors():
        return _open(ctx, folder).neighbourhood(id, depth)


@tool("study")
def study_trace(ctx: ToolContext, folder: str, claim: str) -> dict:
    """A claim's provenance chain as a tree: claim -> evidence -> document -> source, claim -> result -> run ->
    datasets and plan (with the run's code hashes and environment), a hypothesis -> its tests; each link with its
    relation, whether the chain is complete and what is missing (a source retracted since, a hypothesis with no test,
    a run with no code or environment)."""
    with _errors():
        return _open(ctx, folder).trace(claim)


@tool("study")
def study_impact(ctx: ToolContext, folder: str, source: str) -> dict:
    """What rests on a source (the impact of its retraction): the evidence quoting it, the claims resting on that
    evidence, comparisons of it, the papers rendering any of them, and the questions and hypotheses those claims
    answer or test (`conclusions`)."""
    with _errors():
        return _open(ctx, folder).depends_on_source(source)


@tool("study")
def study_traceability(ctx: ToolContext, folder: str, n: int = 50) -> dict:
    """The share of claims whose provenance chain is complete (study_trace), the same for the claims a person
    approved (the gate is 100%), and each broken claim with what it is missing."""
    n = max(1, min(n, BROKEN_MAX))
    with _errors():
        out = _open(ctx, folder).traceability()
    if len(out["broken"]) > n:
        out = {**out, "broken": out["broken"][:n], "broken_withheld": len(out["broken"]) - n}
    return out


@tool("study")
def study_frontier(ctx: ToolContext, folder: str, n: int = 20) -> dict:
    """What to do next, derived from the record and ranked by expected information per hour: untested hypotheses
    and questions, claims at risk, contradictions found and not decided, insufficient contradiction searches,
    inconclusive tests, empty lattice cells, a stale paper, forecasts awaiting outcomes. Each item: id, kind, target,
    why, the verb to use, info (0 to 1), cost (seconds), score (info per hour) and the action that took it, if any.
    Take one with study_take, then act on it with its verb."""
    with _errors():
        return _open(ctx, folder).frontier(max(1, min(n, FRONTIER_MAX)))


@tool("study", mutates=True)
def study_take(ctx: ToolContext, folder: str, item: str, minutes: float = 30.0) -> dict:
    """Take a frontier item by its id: recorded, and leased to you for `minutes` (taking it again renews); another
    agent's take is refused until you release it or end. Then act on it with the verb it names."""
    with _errors():
        out = _open(ctx, folder).take(item, minutes=minutes)
    return {"action": out["action"], **out["item"], "lease": out["lease"], "renewed": out["renewed"]}


@tool("study", mutates=True)
def study_release(ctx: ToolContext, folder: str, item: str) -> dict:
    """Give back a frontier item you hold, so another agent can take it."""
    with _errors():
        return _open(ctx, folder).release(item)


@tool("study", mutates=True)
def study_monitor(ctx: ToolContext, folder: str, schedule: str = "") -> dict:
    """One monitoring pass: runs the sealed review's discovery again (new sources), checks the sources evidence
    quotes or papers cite for retraction (Crossref notices, OpenAlex's flag) and records each as a retraction node,
    fills forecast outcomes, and notifies the news ([monitor] notify). Returns the news, each new retraction's impact
    (the claims, questions and papers resting on the source) and how it was delivered. With `schedule` (status, on,
    off): the [monitor] cron line in the user's crontab instead."""
    with _errors():
        s = _open(ctx, folder)
        if schedule == "on" or (not schedule and s.record.find("forecast")):
            no_agent_code("study_monitor")              # filling an outcome runs model.py
        if schedule:
            return s.monitor_schedule(schedule, actor="agent")
        out = s.monitor(actor="agent")
    for k in ("new_sources", "contradicting", "outcomes", "unchecked"):
        if len(out[k]) > WATCH_MAX:
            out = {**out, k: out[k][:WATCH_MAX], f"{k}_withheld": len(out[k]) - WATCH_MAX}
    return out


@tool("study", mutates=True)
def study_replicate(ctx: ToolContext, folder: str, paper: str, mode: str, methods: list, metric: str,
                    tolerance: float, max_tolerance: float = 0.0, kind: str = "replication", code: str = "",
                    paper_unit: str = "") -> dict:
    """Start a study that replicates one paper (`paper`: its arXiv id, DOI or a PDF in the workspace) in a new
    folder: a review that finds the paper by its id is sealed, its source and full text (by page) recorded, and the
    target written with the tolerance policy on `metric` (`tolerance`, or up to `max_tolerance` with a
    justification, in the metric's own unit), before any run. Scaffolds model.py with one class per method
    (`methods`, class names; the last is Model, the others its baselines), paper/equations.tex, a data recipe stub and
    README.md with the next steps. `mode`: temporal or static. `kind`: replication (from the text) or reproduction
    (the authors' `code`). `paper_unit` percent: the paper prints shares such as coverage as percentages."""
    from .. import study
    src = str(ctx.path(paper)) if paper.lower().endswith(".pdf") else paper
    with _errors():
        s = study.replicate(src, ctx.path(folder), mode, methods, metric, tolerance, max_tolerance, kind, code,
                            paper_unit)
        return {"folder": ctx.rel(s.folder), **s.replications(), "next": f"{ctx.rel(s.folder)}/README.md"}


@tool("study", mutates=True)
def study_fidelity(ctx: ToolContext, folder: str, paper: str, ours: str, arm: str = "model",
                   tolerance: Optional[float] = None, metric: str = "", justification: str = "",
                   group: str = "") -> dict:
    """Did we rebuild it? One baseline of the target paper: its value (evidence `paper`, the field named by the
    metric, default the primary) against ours (`arm` of result `ours`: model, or a baseline the run scored such as
    model:EMOS_gl; several result ids, comma-separated, when the paper's number is a mean over runs such as one per
    horizon or seed: openlab averages them), within `tolerance` (default the target's; wider, up to its max, needs a `justification`). openlab
    computes paper, ours and passed; the result's run must be clean or separated. `group`: one entity's scores (a
    station, a dataset) when the study sets evaluate.by = "key"."""
    return _organise(ctx, folder, "fidelity", paper, ours, arm, tolerance, metric, justification, group)


@tool("study", mutates=True)
def study_replication(ctx: ToolContext, folder: str, paper: str, ours: str, gated_by: list, arm: str = "model",
                      tolerance: Optional[float] = None, metric: str = "", justification: str = "",
                      group: str = "") -> dict:
    """Does it hold? One claim of the target paper (evidence `paper`) against ours with its interval (`arm` of
    result `ours`, or comma-separated results whose mean the paper prints), gated by fidelities (`gated_by`) over the
    same protocols. openlab computes the verdict against the
    best gating baseline: replicated, weaker, not_replicated or inconclusive; with any gating fidelity failed it is
    inconclusive (rebuild not faithful), so a bad rebuild cannot fail the paper."""
    return _organise(ctx, folder, "replication", paper, ours, gated_by, arm, tolerance, metric, justification, group)


@tool("study")
def study_replications(ctx: ToolContext, folder: str) -> dict:
    """The replication report, derived from the record: the target paper and its tolerance policy, each fidelity
    (rebuilt or not), each claim's verdict with the paper's value, ours and our interval, and whether the rebuild is
    faithful (every fidelity passed)."""
    with _errors():
        return _open(ctx, folder).replications()


@tool("study", mutates=True)
def study_reproduce(ctx: ToolContext, folder: str = "", run: str = "", all_runs: bool = False,
                    bundle: str = "") -> dict:
    """Run recorded runs again from what the record holds only (the code blobs, the data whose sha the run recorded,
    the sealed spec at the run and its seed) in fresh sandboxes, and record a reproduction of each: matched or what
    differs, the largest prediction difference, each metric again and the environment differences. One run (`run`,
    default the latest) or `all_runs`; or every run of a `bundle` tar (study_bundle), in a temp dir, from it alone.
    A claim whose results all have a matched reproduction is at level reproduced."""
    no_agent_code("study_reproduce")
    with _errors():
        if bundle:
            from ..study.reproduce import from_bundle
            return from_bundle(ctx.path(bundle))
        return _open(ctx, folder).reproduce(run, all_runs)


@tool("study", mutates=True)
def study_bundle(ctx: ToolContext, folder: str, out: str) -> dict:
    """Write the study as a signed tar a third party can reproduce from (study_reproduce with `bundle`): the record,
    its blobs, the materialised data, the study's files and their signatures, and a signed manifest of every file's
    sha256 and the record head."""
    no_agent_code("study_bundle")
    with _errors():
        made = _open(ctx, folder).bundle(ctx.path(out))
    return {**made, "bundle": ctx.rel(ctx.path(out))}


@tool("study")
def study_reproductions(ctx: ToolContext, folder: str) -> dict:
    """The study's reproductions (each: the run, matched or what differs, the largest prediction difference, the
    metrics again, the environment differences) and how many of its runs have a matched one."""
    with _errors():
        return _open(ctx, folder).reproductions()
