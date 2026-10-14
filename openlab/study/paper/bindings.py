r"""What a paper's prose binds: its macros, parsed, and each resolved against the record and checked for staleness;
the equation gate (`\labeq`), which runs the study's Python function in a sandboxed child.

    \lab{<node id>#<dotted path>}        a value of a node's payload
    \labtab{<result id>}  \labfig{<result id>}  \labclaim{<claim id>}
    \labcite{<source id>}{<quote>}       \labeq{<name>}{<module.fn>}{<latex>}

A result binding is stale once the study's code, spec or data moved on from the run it comes from: the headline
must be the current model. `\lab[ablation]{...}` (also \labtab, \labfig, \labclaim) binds an earlier variant on
purpose, for an ablation table: it may come from earlier code, but not from another spec or other data (an
ablation varies the model under the same protocol).

Nothing here prints LaTeX (sections.py does), so a child can import this module without the app.
"""

from __future__ import annotations

import importlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .. import fingerprint, schema
from .. import spec as _spec
from ..errors import StudyError

MACROS = {"lab": 1, "labclaim": 1, "labfig": 1, "labtab": 1, "labcite": 2, "labeq": 3}
OPTIONS = {"ablation": ("lab", "labclaim", "labfig", "labtab")}      # \lab[ablation]{...}: an earlier variant on purpose
KINDS = {"labclaim": "claim", "labfig": "result", "labtab": "result", "labcite": "source"}
_START = re.compile(r"\\(lab(?:claim|fig|tab|cite|eq)?)(?![A-Za-z])")
EQ_TIMEOUT = 60


@dataclass(frozen=True)
class Macro:
    name: str
    args: tuple
    file: str
    line: int
    start: int
    end: int
    error: str = ""                 # a syntax error: the macro is unresolved
    option: str = ""                # [ablation]

    @property
    def text(self) -> str:
        return f"\\{self.name}" + (f"[{self.option}]" if self.option else "") + "".join("{" + a + "}" for a in self.args)


@dataclass(frozen=True)
class Bound:
    """A macro and the node it binds ("" for an equation); `error` says why it does not resolve, `stale` why the
    node no longer describes the study's code, spec or data."""
    macro: Macro
    node: str = ""
    error: str = ""
    stale: str = ""
    earlier: str = ""               # an [ablation] binding of an earlier variant: why it is not the current model


def mask_comments(text: str) -> str:
    """The text with each comment (an unescaped % to the end of its line) blanked, offsets kept."""
    return re.sub(r"(?<!\\)%[^\n]*", lambda m: " " * len(m.group(0)), text)


def _group(text: str, i: int) -> tuple:
    """(content, end) of the brace group at i (spaces before it allowed), or (None, i)."""
    while i < len(text) and text[i] in " \t":
        i += 1
    if i >= len(text) or text[i] != "{":
        return None, i
    depth = 0
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0) if text[j - 1] != "\\" or j == i else 0
        if depth == 0:
            return text[i + 1:j], j + 1
    return None, i


def parse(text: str, file: str) -> list:
    """The binding macros of `text` (comments already masked), in order."""
    out = []
    for m in _START.finditer(text):
        name, args, end, option = m.group(1), [], m.end(), ""
        opt = re.match(r"\[([^\]\n]*)\]", text[end:])
        if opt:
            option, end = opt.group(1).strip(), end + opt.end()
        for _ in range(MACROS[name]):
            arg, end = _group(text, end)
            if arg is None:
                break
            args.append(arg.strip())
        line = text.count("\n", 0, m.start()) + 1
        error = "" if len(args) == MACROS[name] else (
            f"\\{name} takes {MACROS[name]} brace argument{'s' if MACROS[name] > 1 else ''}")
        if option and name not in OPTIONS.get(option, ()):
            error = error or (f"\\{name} has no option [{option}]" + (
                f"; [ablation] binds {', '.join(chr(92) + n for n in OPTIONS['ablation'])}" if option == "ablation"
                else "; the one option is [ablation]"))
        out.append(Macro(name, tuple(args), file, line, m.start(), end, error, option))
    return out


def current(rec, folder: Path) -> dict:
    """The study's spec sha, code hashes and data hashes as they stand: what a bound run must still match (a run of
    a branch's working copy, that branch's code: openlab.study.branches)."""
    from ..branches import current as branch_code
    spec = _spec.sealed(rec)
    return {"spec_sha": spec.sha() if spec else "", "code": fingerprint.code(folder),
            "data": fingerprint.data(rec, set(spec.data) if spec else None), "branches": branch_code(folder)}


def runs(rec, node) -> list:
    """The runs a node's numbers come from: a run itself, a result's or model's run, or the runs of the results
    a claim or comparison rests on (results only: a run's earlier branches are not what it rests on)."""
    if node.kind == "run":
        return [node]
    if node.kind == "result":
        return [schema.run_of(rec, node)]
    if node.kind == "model":
        return [rec.get(rec.links(node.id, rel="from")[0][2])]
    return [schema.run_of(rec, n) for n in rec.why(node.id) if n.kind == "result"]


def _differ(a: dict, b: dict) -> str:
    return ", ".join(sorted(k for k in {*a, *b} if a.get(k) != b.get(k)))


def stale(rec, node, now: dict, code: bool = True) -> str:
    """Why `node` no longer describes the study ("" when it does): a run it comes from ran other code, spec or
    data than the study has now (`code=False`: other spec or data only, as an ablation binding is checked)."""
    for run in runs(rec, node):
        p = run.payload
        owned = (now.get("branches") or {}).get(p.get("branch") or "", now["code"])     # the run's branch's code now
        what = [f"code ({_differ(p['code'], owned)})" if code and p["code"] != owned else "",
                "spec" if p["spec_sha"] != now["spec_sha"] else "",
                f"data ({_differ(p['data'], now['data'])})" if p["data"] != now["data"] else ""]
        what = [w for w in what if w]
        if what:
            return f"{node.id} comes from {run.id}, whose {' and '.join(what)} differ from the study's now"
    return ""


def value(rec, ref: str):
    """The printable value at `<node-id>#<dotted path>` (schema.lookup): a number, text, true/false, or an estimate
    with its interval ({value | diff, lo, hi}); a table or a missing value is refused, saying why."""
    path = ref.partition("#")[2]
    v = schema.lookup(rec, ref)[1]
    if isinstance(v, dict) and ("value" in v or "diff" in v):
        main = v.get("value", v.get("diff"))
        if main is None:
            raise StudyError(f"`{path}` has no value: {v.get('reason', 'not computed')}")
        return v
    if isinstance(v, (dict, list)):
        raise StudyError(f"`{path}` is a {'table' if isinstance(v, dict) else 'list'}, not a value",
                         fix="name one of its entries" + (f": {path}.{', {path}.'.join(map(str, list(v)[:12]))}"
                                                          if isinstance(v, dict) and v else ""))
    if v is None:
        raise StudyError(f"`{path}` is empty")
    return v


def resolve(rec, macro: Macro, now: dict) -> Bound:
    """The macro bound to its node: unresolved (error) unless the node exists, is of the macro's kind and the
    path names a value; stale when the node's runs no longer match the study."""
    if macro.error:
        return Bound(macro, error=macro.error)
    if macro.name == "labeq":
        return Bound(macro)
    want = KINDS.get(macro.name)
    try:
        if macro.name == "lab":
            value(rec, macro.args[0])
        node = rec.get(macro.args[0].partition("#")[0])
        if want and node.kind != want:
            raise StudyError(f"{node.id} is a {node.kind}, not a {want}")
    except StudyError as e:
        return Bound(macro, error=e.error)
    if macro.option == "ablation":
        return Bound(macro, node=node.id, stale=stale(rec, node, now, code=False), earlier=stale(rec, node, now))
    return Bound(macro, node=node.id, stale=stale(rec, node, now))


# --- the equation gate ---------------------------------------------------------------------------------------
def equations(study, macros: list) -> dict:
    r"""{macro: ""} when `<module.fn>` from the study's code agrees with the LaTeX on the engine.modules probes, else
    {macro: why not}. The LaTeX is read by the engine; the function runs in one sandboxed child for all macros."""
    import numpy as np
    import sympy as sp
    from ...engine import modules
    from .. import sandbox
    out, todo = {}, []
    for m in macros:
        name, tex = m.args[0], m.args[2]
        mod, ns = modules.Module(name=name, kind="custom", definition=tex, fn=name), {}
        try:
            modules._definitions(mod, ns)
        except Exception as e:                  # noqa: BLE001 -- the engine's refusal is the gate's finding
            out[m] = str(e)
            continue
        if not isinstance(ns.get(name), sp.Lambda):
            out[m] = f"the LaTeX does not define a function {name}" + (f"; it defines {', '.join(ns)}" if ns else "")
            continue
        mod.arity = len(ns[name].variables)
        todo.append((m, mod, ns, modules._probes(mod)))
    if not todo:
        return out
    root = study.record.dir / "stage"
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="paper-", dir=root))
    try:
        calls = [{"fn": m.args[1], "probes": [p.tolist() for p in probes]} for m, _, _, probes in todo]
        res = sandbox.run({"entry": f"{__name__}:eq_job", "args": {"calls": calls}}, study_dir=study.folder,
                          stage_dir=stage, timeout=EQ_TIMEOUT)
        got = json.loads((stage / "equations.json").read_text()) if res.ok else None
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    for i, (m, mod, ns, _) in enumerate(todo):
        if got is None:
            out[m] = f"the child failed: {res.stderr.strip()[-300:]}"
        elif "error" in got[i]:
            out[m] = f"{m.args[1]} failed on the probes: {got[i]['error']}"
        else:
            mod.binding = lambda *_, v=np.asarray(got[i]["values"], dtype=float): v
            try:
                modules._check_binding(mod, ns)
                out[m] = ""
            except Exception as e:              # noqa: BLE001 -- a disagreement, or a LaTeX the check cannot evaluate
                out[m] = (str(e).replace("the binding", m.args[1]).replace("): binding ", f"): {m.args[1]} ")
                          or f"the check failed: {type(e).__name__}")
    return out


def eq_job(stage: Path, *, calls: list) -> Path:
    """In the child: each `module.fn` of the study evaluated on its probes (vectorised, else point by point)."""
    import numpy as np
    out = []
    for c in calls:
        try:
            module, _, name = c["fn"].rpartition(".")
            if not module:
                raise ValueError(f"name the function as module.fn (e.g. model.{name})")
            f = getattr(importlib.import_module(module), name)
            xs = [np.asarray(p, dtype=float) for p in c["probes"]]
            try:
                v = np.asarray(f(*xs), dtype=float)
                if v.shape != xs[0].shape:
                    raise ValueError
            except Exception:                   # noqa: BLE001 -- a scalar function: one probe at a time
                v = np.asarray([float(f(*row)) for row in zip(*(x.tolist() for x in xs))])
            out.append({"values": v.tolist()})
        except Exception as e:                  # noqa: BLE001 -- reported as the gate's finding
            out.append({"error": f"{type(e).__name__}: {e}"})
    path = stage / "equations.json"
    path.write_text(json.dumps(out), encoding="utf-8")
    return path
