"""The one-line transparency string: *how openlab read the statement*."""

from __future__ import annotations

import sympy
from sympy import Lambda
from sympy.matrices import MatrixBase
from sympy.printing.str import StrPrinter


class _OneLine(StrPrinter):
    """``sstr`` with matrices on one line, so the interpretation stays one line."""

    def _print_MatrixBase(self, expr):
        rows = ", ".join("[" + ", ".join(self._print(e) for e in expr.row(i)) + "]"
                         for i in range(expr.rows))
        return f"Matrix([{rows}])"

    _print_ImmutableDenseMatrix = _print_MutableDenseMatrix = _print_MatrixBase


class _Inv(sympy.Function):
    """``A**(-1)`` for a matrix-like factor: printed as a power and never reordered."""
    is_commutative = False

    def _sympystr(self, printer):
        return f"{printer._print(self.args[0])}**(-1)"


def _matrixlike(b) -> bool:
    return isinstance(b, MatrixBase) or (isinstance(b, sympy.Symbol) and len(b.name) == 1 and b.name.isupper())


def pyform(expr) -> str:
    """A SymPy expression in the order it was typed, paste-able into a Python cell."""
    try:
        if isinstance(expr, sympy.Basic) and expr.has(sympy.Pow):
            expr = expr.replace(lambda n: n.is_Pow and n.exp == -1 and _matrixlike(n.base),
                                lambda n: _Inv(n.base))
    except Exception:   # presentation only
        pass
    try:
        return _OneLine({"order": "none"}).doprint(expr)
    except Exception:   # printing is best effort
        return str(expr)


def short(v) -> str:
    """A compact rendering of a namespace value for the ``with ...`` clause."""
    if isinstance(v, MatrixBase):
        return f"Matrix({v.rows}×{v.cols})"
    if isinstance(v, Lambda):
        args = ", ".join(str(a) for a in v.args[0])
        return f"({args}) ↦ {pyform(v.expr)}"
    s = pyform(v) if isinstance(v, sympy.Basic) else repr(v)
    return s if len(s) <= 40 else s[:37] + "..."


def build(prefix: str, body: str, subs: dict, notes: list[str]) -> str:
    out = f"{prefix} {body}".strip()
    if subs:
        out += " | with " + ", ".join(f"{k} = {short(v)}" for k, v in subs.items())
    uniq = list(dict.fromkeys(n for n in notes if n))
    if uniq:
        out += " | note: " + "; ".join(uniq)
    return out
