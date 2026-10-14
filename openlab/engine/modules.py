r"""Hardened modules: a named piece of ML/RL mathematics -- an activation, a loss, a policy objective, a
shape rule, an update rule, a schedule -- defined in LaTeX, with claims openlab verifies and a NumPy binding
that must agree with the definition.

    from openlab.engine import modules as qm
    m = qm.register_module("SWISH", r"\operatorname{SWISH}(x) = \frac{x}{1 + e^{-x}}",
                           claims=[r"\operatorname{SWISH}(0) = 0",
                                   r"\lim_{x \to -\infty} \operatorname{SWISH}(x) = 0"],
                           binding=lambda x: x / (1 + np.exp(-x)), kind="activation")
    m.status            # 'hardened' -- every claim proved or true, and the binding matches the definition

Then, in a cell, ``\text{module SWISH}`` binds it (the definition is run there and the claims are verified
again; the note shows the status) and ``\operatorname{SWISH}(x)`` works in any statement.

Verdict tiers are never blurred: a claim carries openlab's verdict (``proved``, ``true``, or why not); the
binding check is reported as *matches on N probes*, never as a proof. A module whose definition openlab
cannot read, or whose binding disagrees with the definition, is refused at registration. A module with a
claim that did not verify is registered as **marked**, with the reason -- usable, but it says so.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

#: What a module can be.
KINDS = ("activation", "loss", "rl", "norm", "shape", "optimizer", "schedule", "risk", "custom")
#: A claim counts only with one of these verdicts.
OK = ("proved", "true")
#: A definition line may land as any of these.
_DEFINED = ("bound", "value", "solved", "true", "proved")
_NAME = re.compile(r"^[A-Z]{2,24}$")
_PROBES = 64


@dataclass
class Claim:
    """One claim of a module: its LaTeX, and what openlab said."""
    tex: str
    verdict: str = ""
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict in OK

    def __str__(self):
        return f"{self.verdict or '?':<10} {self.tex}" + (f"  -- {self.note}" if self.note and not self.ok else "")


@dataclass
class Module:
    """A hardened module. Build one with :func:`register_module`; read them with :func:`describe`."""
    name: str
    kind: str
    definition: str
    claims: list = field(default_factory=list)
    fn: str = ""                                  # the function the definition binds (defaults to name)
    binding: Optional[Callable] = None
    arity: int = 1
    probe: tuple = ()                             # per argument: (lo, hi) floats, ("int", lo, hi), ("choice", a, b, …)
    reference: Optional[Callable] = None          # executable twin of a recurrence (sequence modules)
    requires: tuple = ()                          # modules whose definitions this one uses
    params: dict = field(default_factory=dict)    # the configured constants, for the record
    doc: str = ""
    status: str = ""                              # hardened | marked
    binding_check: str = ""
    fingerprint: str = ""
    builtin: bool = False
    probe_ns: Optional[Callable] = None           # names to bind before the claims (arrays for sequence claims)

    @property
    def text(self) -> str:
        """The module as openlab statements: the definition, then every claim."""
        return "\n".join([self.definition.strip()] + [c.tex for c in self.claims])

    @property
    def failed(self) -> list:
        return [c for c in self.claims if not c.ok]

    def to_record(self) -> dict:
        return {"name": self.name, "kind": self.kind, "status": self.status, "fn": self.fn, "arity": self.arity,
                "definition": self.definition.strip(), "params": dict(self.params), "requires": list(self.requires),
                "claims": [{"tex": c.tex, "verdict": c.verdict, "note": c.note} for c in self.claims],
                "binding": self.binding_check, "fingerprint": self.fingerprint, "doc": self.doc, "builtin": self.builtin}

    def __str__(self):
        head = f"{self.name} ({self.kind}, {self.status}) [{self.fingerprint}]"
        if self.params:
            head += "  " + ", ".join(f"{k} = {v}" for k, v in self.params.items())
        lines = [head, "  " + self.definition.strip().replace("\n", "\n  ")]
        lines += ["  " + str(c) for c in self.claims]
        lines.append(f"  binding: {self.binding_check}")
        return "\n".join(lines)


#: The library: name -> Module.
LIB: dict = {}


# --- verification -----------------------------------------------------------------------------------
def _run(text: str, ns: dict) -> list:
    from .records import run_records
    return run_records(text, ns, strict=False, stop_on_error=False)


def _definitions(module: Module, ns: dict) -> list:
    """Run the definitions of what the module requires, then its own, into ``ns``; the module's records."""
    for req in module.requires:
        dep = LIB.get(req.upper())
        if dep is None:
            raise ValueError(f"{module.name} requires {req}, which is not a module")
        _definitions(dep, ns)
    recs = _run(module.definition.strip(), ns)
    bad = [r for r in recs if r["verdict"] not in _DEFINED]
    if bad:
        r = bad[0]
        why = r.get("message") or (r["notes"][0] if r.get("notes") else r["verdict"])
        raise ValueError(f"{module.name}: openlab cannot read the definition (line {r['line']}: {why})")
    return recs


def _probes(module: Module):
    import numpy as np
    rng = np.random.default_rng(0)
    spec = module.probe or ((-4.0, 4.0),) * module.arity
    if len(spec) != module.arity:
        raise ValueError(f"{module.name}: probe has {len(spec)} ranges for {module.arity} arguments")
    out = []
    for s in spec:
        if s[0] == "int":
            out.append(rng.integers(int(s[1]), int(s[2]) + 1, size=_PROBES).astype("int64"))
        elif s[0] == "choice":
            out.append(rng.choice(np.asarray(s[1:], dtype="float64"), size=_PROBES))
        else:
            out.append(rng.uniform(float(s[0]), float(s[1]), size=_PROBES))
    return out


def _lambdify(fn, arity: int) -> Callable:
    """A NumPy evaluator of the definition's Lambda, falling back to point-wise SymPy evaluation."""
    import numpy as np
    import sympy as sp
    args = list(fn.args[0]) if isinstance(fn, sp.Lambda) else None
    if args is None or len(args) != arity:
        raise ValueError(f"the definition binds a {type(fn).__name__} of {len(args) if args else '?'} argument(s), "
                         f"not a function of {arity}")
    expr = fn.expr
    erf = np.vectorize(math.erf)
    try:
        f = sp.lambdify(args, expr, modules=[{"erf": erf}, "numpy"])
        f(*[np.zeros(2) + 0.5 for _ in args])              # smoke: some printers cannot do Max/Piecewise on arrays
        return f
    except Exception:                                    # noqa: BLE001
        def point(*xs):
            vals = []
            for row in zip(*[np.asarray(x).ravel() for x in xs]):
                v = expr.subs(dict(zip(args, [sp.Float(float(r)) for r in row])))
                vals.append(float(sp.N(v)))
            return np.asarray(vals)
        return point


def _close(a, b) -> bool:
    import numpy as np
    a, b = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=1e-6, atol=1e-7, equal_nan=True))


def _check_binding(module: Module, ns: dict) -> str:
    import numpy as np
    if module.binding is None:
        return "no binding"
    probes = _probes(module)
    if module.reference is not None:
        got, want = module.binding(*probes), module.reference(*probes)
        what = "the reference recurrence"
    else:
        fn = ns.get(module.fn)
        if fn is None:
            raise ValueError(f"{module.name}: the definition does not bind {module.fn}")
        got, want = module.binding(*probes), _lambdify(fn, module.arity)(*probes)
        what = "the definition"
    got, want = np.asarray(got, dtype="float64"), np.asarray(want, dtype="float64")
    if not _close(got, want):
        i = int(np.argmax(~np.isclose(got, want, rtol=1e-6, atol=1e-7, equal_nan=True))) if got.shape == want.shape else 0
        at = ", ".join(f"{float(p.ravel()[i]):.4g}" for p in probes) if got.shape == want.shape else "?"
        raise ValueError(f"{module.name}: the binding disagrees with {what} (at ({at}): binding {got.ravel()[i] if got.shape == want.shape else got.shape}, "
                         f"{what} {want.ravel()[i] if got.shape == want.shape else want.shape})")
    return f"matches {what} on {_PROBES} probes"


def _fingerprint(module: Module) -> str:
    from .. import __version__
    src = ""
    for f in (module.binding, module.reference):
        if f is not None:
            try:
                src += inspect.getsource(f)
            except (OSError, TypeError):
                src += repr(f)
    deps = "".join(LIB[r].fingerprint for r in module.requires if r in LIB)
    h = hashlib.sha256((module.text + "\n" + src + "\n" + deps + "\n" + __version__).encode()).hexdigest()
    return h[:12]


def _cache_file():
    """Where verified verdicts are kept between processes: ``$OPENLAB_MODULE_CACHE`` (``off`` disables it), else
    ``~/.cache/openlab/modules.json``. The key is the fingerprint -- definition, claims, binding source, what the
    module requires, and the engine version -- so an entry can only ever be replayed for the same module."""
    import os
    from pathlib import Path
    where = os.environ.get("OPENLAB_MODULE_CACHE", "")
    if where.lower() in ("off", "0", "false"):
        return None
    return Path(where) if where else Path.home() / ".cache" / "openlab" / "modules.json"


def _cache_read() -> dict:
    import json
    f = _cache_file()
    try:
        return json.loads(f.read_text()) if f is not None and f.is_file() else {}
    except (OSError, ValueError):
        return {}


def _cache_write(fp: str, module: "Module") -> None:
    import json
    import os
    f = _cache_file()
    if f is None:
        return
    try:
        data = _cache_read()
        data[fp] = {"name": module.name, "status": module.status, "binding": module.binding_check,
                    "claims": [[c.verdict, c.note] for c in module.claims]}
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data))
        os.replace(tmp, f)
    except OSError:
        pass


def verify(module, ns: Optional[dict] = None, force: bool = False) -> Module:
    """Verify a module (or a module name): run its definition, verify every claim, test the binding.
    Sets ``status``, each claim's verdict, ``binding_check`` and ``fingerprint``; raises if refused.
    Verdicts already established for this exact fingerprint are replayed from the cache (``force=True``
    proves them again); the definition always runs, so the module's functions are bound in ``ns``."""
    if isinstance(module, str):
        module = describe(module)
    ns = {} if ns is None else ns
    _definitions(module, ns)
    if module.probe_ns is not None:
        ns.update(module.probe_ns(*_probes(module)))
    fp = _fingerprint(module)
    hit = None if force else _cache_read().get(fp)
    if hit and len(hit.get("claims", [])) == len(module.claims):
        for c, (verdict, note) in zip(module.claims, hit["claims"]):
            c.verdict, c.note = verdict, note
        module.status, module.binding_check, module.fingerprint = hit["status"], hit["binding"], fp
        return module
    module.binding_check = _check_binding(module, ns)
    for c in module.claims:
        recs = _run(c.tex, ns)
        r = recs[0] if recs else {"verdict": "error", "message": "no statement"}
        if len(recs) > 1:                       # a claim may bind a value first; its last statement decides
            worst = next((x for x in recs[:-1] if x["verdict"] not in OK + ("bound", "value")), None)
            r = worst or recs[-1]
        c.verdict = r["verdict"]
        c.note = "" if c.ok else (r.get("message") or (r["notes"][0] if r.get("notes") else "") or r["verdict"])
    module.status = "hardened" if not module.failed else "marked"
    module.fingerprint = fp
    _cache_write(fp, module)
    return module


# --- the API ---------------------------------------------------------------------------------------
def register_module(name: str, definition: str, claims: Sequence[str] = (), *, binding: Optional[Callable] = None,
                    kind: str = "custom", fn: Optional[str] = None, arity: int = 1, probe: Sequence = (),
                    reference: Optional[Callable] = None, requires: Sequence[str] = (), params: Optional[dict] = None,
                    doc: str = "", replace: bool = False, builtin: bool = False, probe_ns=None) -> Module:
    """Register a hardened module. ``definition`` and ``claims`` are openlab statements; ``binding`` a NumPy
    function of ``arity`` arrays, tested against the definition on probes in the given ranges (or against
    ``reference``, an executable twin, for a recurrence). Raises ``ValueError`` when refused."""
    key = name.upper()
    if not _NAME.match(key):
        raise ValueError(f"a module name is 2-24 letters (got {name!r})")
    if kind not in KINDS:
        raise ValueError(f"kind is one of {', '.join(KINDS)}")
    if key in LIB and not replace:
        raise ValueError(f"{key} is already a module; pass replace=True to override it")
    m = Module(name=key, kind=kind, definition=definition, claims=[Claim(c) for c in claims], fn=(fn or key),
               binding=binding, arity=arity, probe=tuple(probe), reference=reference, requires=tuple(r.upper() for r in requires),
               params=dict(params or {}), doc=doc, builtin=builtin, probe_ns=probe_ns)
    verify(m)
    LIB[key] = m
    return m


def unregister_module(name: str) -> None:
    LIB.pop(name.upper(), None)


def _ensure_builtins() -> None:
    """No module is built in: the library holds what was registered in this process."""


def library(kind: Optional[str] = None) -> list:
    """Every module (of a kind), built-in first, then by name."""
    _ensure_builtins()
    ms = [m for m in LIB.values() if kind is None or m.kind == kind]
    return sorted(ms, key=lambda m: (not m.builtin, m.kind, m.name))


def describe(name: str) -> Module:
    _ensure_builtins()
    key = name.upper()
    if key not in LIB:
        raise KeyError(f"no module named {key}; the library has {', '.join(sorted(LIB)) or 'nothing'}")
    return LIB[key]


def bind(name: str, ns: dict) -> Module:
    """Run the module's definition (and what it requires) into ``ns`` and verify its claims there, so
    ``\\operatorname{NAME}(…)`` works in later statements. Returns the module with fresh verdicts."""
    m = describe(name)
    return verify(m, ns)


def records(kind: Optional[str] = None) -> list:
    return [m.to_record() for m in library(kind)]
