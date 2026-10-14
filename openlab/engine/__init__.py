r"""The LaTeX engine: openlab's checker behind a study's derived claims and the paper's equation gate.

Internal: a study calls it, a person or an agent does not. A derived claim's expression is evaluated here
(:func:`openlab.study.schema.evaluate`), and the paper's ``\labeq`` equations are read here and compared with the
study's code on probes (:mod:`openlab.engine.modules`, :func:`openlab.study.paper.bindings.equations`).

>>> from openlab import engine
>>> engine.run(r"\int_0^1 x^2 \, dx").value
1/3

Loaded on first use (sympy and lark are heavy), so ``import openlab.engine`` is cheap until a name is read.
"""

_LAZY = {"parse_expr": "parse", "parse_statement": "parse", "Kind": "semantics", "Result": "semantics",
         "iter_run": "semantics", "run": "semantics", "run_all": "semantics",
         "OpenlabError": "errors", "ParseError": "errors", "SemanticsError": "errors",
         "StrayDifferentialError": "errors", "UnsupportedError": "errors",
         "run_records": "records", "VERDICTS": "records"}


def __getattr__(name):
    import importlib
    if name in _LAZY:
        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module 'openlab.engine' has no attribute {name!r}")


__all__ = list(_LAZY)
