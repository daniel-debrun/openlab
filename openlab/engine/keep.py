r"""Directives inside a cell: bind a module, list the modules.

    \text{module } GELU                               bind a module (openlab.engine.modules): its claims verified here
    \text{modules}                                    the library of modules and their status

The statement registry (``\text{registry}``, ``keep as``, ``\text{use}``, ``\text{lemmas}``, ``\text{lock}``) is
not part of openlab: a study's record keeps what a study shows. Those directives are recognised and refused, so a
cell that uses one says why instead of misreading it.
"""

from __future__ import annotations

import re
from typing import Optional

from .errors import SemanticsError

KINDS = ("lemma", "theorem", "proof", "counterexample", "conjecture", "definition", "spec", "result", "module")
_DIRECTIVE = re.compile(r"^\s*\\text\s*\{\s*(registry|use|lemmas|search|lock|modules|module)\b\s*([^{}]*)\}\s*(.*?)\s*$")
_KEEP = re.compile(r"\\(?:text|openlabnote)\s*\{\s*keep\b(?:\s+as)?\s*(?:(" + "|".join(KINDS) + r")\b)?\s*([^{}]*)\}\s*([A-Za-z0-9_.\-\\]+)?")


def _clean(s: str) -> str:
    s = s.replace("\\_", "_").replace("\\", "").strip().strip("`'\"")
    return re.sub(r"\s*_\s*", "_", s)              # `gauss\_sum` reaches us as `gauss _sum`


def split(src: str) -> tuple:
    """``(src with directive lines blanked, {lineno: (kind, argument)})``."""
    lines = src.split("\n")
    found = {}
    for k, line in enumerate(lines):
        m = _DIRECTIVE.match(line)
        if m:
            found[k + 1] = (m.group(1), _clean((m.group(2) + " " + m.group(3)).strip()))
            lines[k] = ""
    return "\n".join(lines), found


def strip_keep(text: str) -> tuple:
    """``(text without the keep annotation, (kind, name) or None)``."""
    m = _KEEP.search(text)
    if not m:
        return text, None
    kind = m.group(1) or "lemma"
    name = _clean(m.group(2) or m.group(3) or "")
    if not name:
        raise SemanticsError("`keep` needs a name", hint=r"`… \text{ keep as theorem gauss\_sum}`.")
    return (text[: m.start()] + text[m.end():]).rstrip(), (kind, name)



def _note_result(src: str, lineno: int, interpretation: str, notes: list):
    from .semantics import Kind, Result
    return Result(src=src, kind=Kind.NOTE, expr=None, value=None, numeric=None, interpretation=interpretation,
                  lineno=lineno, latex_in=r"\text{" + src.replace("\\", "") + "}", notes=notes)


_GONE = "the statement registry is not part of openlab"
_GONE_HINT = "a study's record keeps what a study shows: claims, runs and results (openlab study claim)"


def run_directive(kind: str, arg: str, ns: dict, lineno: int):
    if kind == "module":
        from .modules import bind
        if not re.match(r"^[A-Za-z]{2,24}$", arg):
            raise SemanticsError("`module` takes the name of a module", hint=r"`\text{module } GELU`; `\text{modules}` lists them.")
        try:
            m = bind(arg, ns)
        except KeyError as exc:
            raise SemanticsError(str(exc), hint="register one with openlab.engine.modules.register_module.") from exc
        n_ok = sum(c.ok for c in m.claims)
        return _note_result(f"module {m.name}", lineno, f"module {m.name} ({m.kind}) {m.status} [{m.fingerprint}]: "
                            f"{n_ok}/{len(m.claims)} claims verified here; binding {m.binding_check}"
                            + (f"; {', '.join(f'{k} = {v}' for k, v in m.params.items())}" if m.params else ""),
                            [m.definition.strip()] + [str(c) for c in m.claims])
    if kind == "modules":
        from .modules import library
        ms = library(arg if arg in ("activation", "loss", "rl", "norm", "shape", "optimizer", "schedule", "custom") else None)
        return _note_result(f"modules {arg}".strip(), lineno, f"{len(ms)} module(s)",
                            [f"{m.name} ({m.kind}, {m.status}): {sum(c.ok for c in m.claims)}/{len(m.claims)} claims -- {m.doc}" for m in ms])
    if kind in ("registry", "use", "lemmas", "search", "lock"):
        raise SemanticsError(f"`{kind}`: {_GONE}", hint=_GONE_HINT)
    raise SemanticsError(f"unknown directive `{kind}`")


def before(directives: dict, lineno: Optional[int], ns: dict):
    """Run (and yield) every directive that comes before ``lineno`` (all of them when None)."""
    for k in sorted(directives):
        if lineno is None or k < lineno:
            kind, arg = directives.pop(k)
            yield run_directive(kind, arg, ns, k)


def log(ns: dict, keep: tuple, text: str, panel: Optional[str] = None) -> str:
    """``keep as``: refused, the statement registry is gone."""
    raise SemanticsError(f"`keep`: {_GONE}", hint=_GONE_HINT)
