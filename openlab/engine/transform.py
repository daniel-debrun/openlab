r"""Turn the Lark parse tree into SymPy objects.

Arithmetic is built with ``evaluate=False`` so the interpretation line can show the
expression *as typed*; evaluation happens later in ``semantics``. Marker heads defer the
decisions that need the namespace (see the ``Marker`` subclasses below): what ``f(x)``
means, whether ``a \times b`` is a cross product, what ``\bar{x}`` is, and so on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import sympy
from sympy import (Add, Derivative, ImmutableMatrix, Integer, Integral, Limit, Mul, Pow,
                   Product, S, Sum, Symbol, Tuple)
from sympy.matrices import MatrixBase
from lark import Token, Transformer, v_args

from sympy.printing.precedence import PRECEDENCE

from . import units as _units
from .errors import ParseError, UnsupportedError
from .grammar import GREEK_NAMES

#: Display names for decorated symbols: ``Symbol('xbar')`` renders as ``\bar{x}``.
DISPLAY_NAMES: dict = {}
#: ``\mathbb{E}``, ``\mathbb{P}``, ``\mathbb{V}``: the operators, whatever E, P, V are bound to.
OPERATOR_NAMES = {"E": "𝔼", "P": "ℙ", "V": "𝕍"}
for _k, _v in OPERATOR_NAMES.items():
    DISPLAY_NAMES[Symbol(_v)] = rf"\mathbb{{{_k}}}"


class Marker(sympy.Function):
    """Base for unevaluated heads resolved in :mod:`semantics`."""

    _fmt = "{name}({args})"
    _lfmt = None

    def _sympystr(self, printer):
        return self._fmt.format(name=self.__class__.__name__,
                                args=", ".join(printer._print(a) for a in self.args),
                                **{f"a{i}": printer._print(a) for i, a in enumerate(self.args)})

    def _latex(self, printer):
        fmt = self._lfmt or self._fmt
        return fmt.format(name=self.__class__.__name__,
                          args=", ".join(printer._print(a) for a in self.args),
                          **{f"a{i}": printer._print(a) for i, a in enumerate(self.args)})


class Apply(Marker):
    """``f(args)``: a call, a product, or an undefined function. ``args[0]`` is the head;
    the keyword-like modifier (prime count / inverse / n-th derivative) is ``args[-1]``
    when it is a :class:`CallMod`."""

    def _sympystr(self, printer):
        head, *args = self.args
        mod = ""
        if args and isinstance(args[-1], CallMod):
            mod, args = args[-1].text(), args[:-1]
        return f"{printer._print(head)}{mod}({', '.join(printer._print(a) for a in args)})"

    _latex = _sympystr


class CallMod(sympy.Function):
    """``('prime', n)`` / ``('inverse',)`` / ``('nderiv', n)`` / ``('power', n)`` attached
    to an Apply: ``f'(x)``, ``f^{-1}(x)``, ``f^{(n)}(x)``, ``f^2(x)``."""

    def text(self):
        kind = str(self.args[0])
        if kind == "prime":
            return "'" * int(self.args[1])
        if kind == "inverse":
            return "^{-1}"
        if kind == "iter":
            return f"^(∘{self.args[1]})"
        if kind == "power":
            return f"^{self.args[1]}"
        return f"^({self.args[1]})"


class BinMarker(Marker):
    """A deferred binary operation, printed with its operands parenthesised as needed."""
    _op = "*"
    _lop = r"\cdot"
    _prec = PRECEDENCE["Mul"]

    @property
    def precedence(self):          # so 1/(a·b) prints its denominator in parentheses
        return self._prec

    def _sympystr(self, printer):
        return f" {self._op} ".join(printer.parenthesize(a, self._prec) for a in self.args)

    def _latex(self, printer):
        return f" {self._lop} ".join(printer.parenthesize(a, self._prec) for a in self.args)


class Det(Marker):
    _fmt = "det({a0})"
    _lfmt = r"\det\left({a0}\right)"


class CallPow(Marker):
    """``name(arg)^e`` written with an explicit exponent: the power of a call -- or, when the
    name turns out to be a multiplier, ``name·arg^e`` (``P(1+r)^t``). A division never makes
    one, so ``\\frac{1}{n(n+1)}`` is not affected."""

    def _sympystr(self, printer):
        return f"{printer._print(self.args[0])}**{printer.parenthesize(self.args[1], PRECEDENCE['Pow'])}"

    def _latex(self, printer):
        return f"{printer._print(self.args[0])}^{{{printer._print(self.args[1])}}}"


class RealRoot(sympy.Function):
    """``\\sqrt[n]{x}`` for odd n: the real root, so ``\\sqrt[3]{-8} = -2``."""

    @classmethod
    def eval(cls, x, n):
        if x.is_number and x.is_real and getattr(n, "is_Integer", False) and x.is_negative is not None:
            r = Pow(abs(x), sympy.Rational(1, int(n)))
            return -r if x.is_negative else r
        return None

    def _sympystr(self, printer):
        return f"root({printer._print(self.args[0])}, {printer._print(self.args[1])})"

    def _latex(self, printer):
        return rf"\sqrt[{printer._print(self.args[1])}]{{{printer._print(self.args[0])}}}"


class Bars(Marker):
    """``|x|``: absolute value, determinant of a square matrix, or norm of a vector."""
    _fmt = "|{a0}|"
    _lfmt = r"\left| {a0} \right|"


class AbsEach(Marker):
    """``\\max |\\operatorname{eig}(A)|``: the absolute value of each entry (inside \\max/\\min)."""
    _fmt = "|{a0}|"
    _lfmt = r"\left| {a0} \right|"


class Compose(BinMarker):
    _op, _lop = "∘", r"\circ"


class PlusMinus(Marker):
    _fmt = "({a0} ± {a1})"
    _lfmt = r"\left({a0} \pm {a1}\right)"


class Star(BinMarker):
    """Deferred product with a matrix on one side and a symbol on the other."""
    _op, _lop = "*", ""


class Plus(BinMarker):
    """Deferred sum with a matrix on one side (scalars broadcast)."""
    _op, _lop, _prec = "+", "+", PRECEDENCE["Add"]


class Dot(BinMarker):
    """``a \\cdot b``: dot product for vectors, plain product otherwise."""
    _op, _lop = "·", r"\cdot"


class Times(BinMarker):
    """``a \\times b``: cross product for 3-vectors, plain/matrix product otherwise."""
    _op, _lop = "×", r"\times"


class Kron(BinMarker):
    _op, _lop = "⊗", r"\otimes"


class MPow(Marker):
    """A matrix literal to an integer power, computed later (under the time limit)."""

    def _sympystr(self, printer):
        return f"{printer.parenthesize(self.args[0], PRECEDENCE['Pow'])}**{printer._print(self.args[1])}"

    def _latex(self, printer):
        return f"{printer._print(self.args[0])}^{{{printer._print(self.args[1])}}}"


class Transform(Marker):
    """``\\mathcal{L}\\{f\\}`` / ``\\mathcal{F}[f]``: Transform(kind, f, inverse[, var])."""

    def _sympystr(self, printer):
        inv = "^-1" if self.args[2] else ""
        return f"{self.args[0]}{inv}{{{printer._print(self.args[1])}}}"

    def _latex(self, printer):
        inv = "^{-1}" if self.args[2] else ""
        return rf"\mathcal{{{self.args[0]}}}{inv}\left\{{{printer._print(self.args[1])}\right\}}"


class Inner(Marker):
    _fmt = "<{args}>"
    _lfmt = r"\left\langle {args} \right\rangle"


class Norm(Marker):
    _fmt = "norm({args})"
    _lfmt = r"\left\| {a0} \right\|"


class Adjoint(Marker):
    """``A^\\dagger`` / ``A^*`` / ``\\bar z``: conjugate transpose or complex conjugate."""

    def _sympystr(self, printer):
        return f"{printer.parenthesize(self.args[0], PRECEDENCE['Pow'])}^H"

    def _latex(self, printer):
        return f"{printer.parenthesize(self.args[0], PRECEDENCE['Pow'])}^{{\\dagger}}"


class Prime(Marker):
    """``y'`` (n primes) on something that is not a call."""

    def _sympystr(self, printer):
        return printer.parenthesize(self.args[0], PRECEDENCE["Pow"]) + "'" * int(self.args[1])

    _latex = _sympystr


class Deco(Marker):
    """``\\bar{x}`` etc. on a symbol: ``Deco(Symbol('x'), Symbol('bar'))`` — a named
    quantity (``xbar``) unless the base is bound and the decoration means conjugate/mean."""
    _fmt = "{a1}({a0})"

    def _latex(self, printer):
        return rf"\{self.args[1]}{{{printer._print(self.args[0])}}}"


class TupleSet(Marker):
    """``\\{(k, a) : bounds and conditions\\}``: the tuples of whole numbers in the bounded box that pass."""

    def _sympystr(self, printer):
        return f"{{{printer._print(self.args[0])} : {', '.join(printer._print(c) for c in self.args[1])}}}"

    def _latex(self, printer):
        return (rf"\left\{{{printer._print(self.args[0])} : "
                rf"{', '.join(printer._print(c) for c in self.args[1])}\right\}}")


class Pick(sympy.Function):
    """``Pick(condition, value, neutral)``: value where the condition holds, else neutral -- the filter
    of \\sum_{p \\text{ prime}} and the Iverson bracket [P]. Not a Piecewise, so SymPy never folds a
    condition on a summation index out of its sum."""
    _is_pick = True

    @classmethod
    def eval(cls, c, value, neutral):
        if c is S.true:
            return value
        if c is S.false:
            return neutral
        return None

    def _sympystr(self, printer):
        c, v, n = self.args
        if v == 1 and n == 0:
            return f"[{printer._print(c)}]"
        return f"({printer._print(v)} if {printer._print(c)})"

    def _latex(self, printer):
        c, v, n = self.args
        if v == 1 and n == 0:
            return rf"\left[{printer._print(c)}\right]"
        return rf"{printer._print(v)}\,\left[{printer._print(c)}\right]" if n == 0 else \
            rf"\begin{{cases}} {printer._print(v)} & {printer._print(c)} \\ {printer._print(n)} & \text{{otherwise}} \end{{cases}}"


class SetImage(Marker):
    """``\\{f(p) : p \\in S, conditions\\}``: args (Lambda(p, (f(p), cond…)), S)."""

    def _sympystr(self, printer):
        lam, S_ = self.args
        head, *conds = lam.expr
        extra = "".join(", " + printer._print(c) for c in conds)
        return f"{{{printer._print(head)} : {lam.variables[0]} in {printer._print(S_)}{extra}}}"

    def _latex(self, printer):
        lam, S_ = self.args
        head, *conds = lam.expr
        extra = "".join(", " + printer._print(c) for c in conds)
        return rf"\left\{{{printer._print(head)} : {printer._print(lam.variables[0])} \in {printer._print(S_)}{extra}\right\}}"


class Iverson(Marker):
    """``[a \\mid b]``: 1 if the (deferred) condition holds, else 0; args (condition, wanted truth)."""

    def _sympystr(self, printer):
        c = printer._print(self.args[0])
        return f"[{c}]" if self.args[1] is S.true else f"[{c.replace('|', '∤', 1)}]"

    def _latex(self, printer):
        c = printer._print(self.args[0])
        return rf"\left[{c}\right]" if self.args[1] is S.true else rf"\left[{c.replace('mid', 'nmid', 1)}\right]"


class IntRange(Marker):
    """``\\{a, a+s, \\ldots, c\\}``: the whole numbers from a to c in steps of s."""
    _fmt = "{{{a0}, …, {a1}}}"
    _lfmt = r"\left\{{{a0}, \ldots, {a1}\right\}}"


class SetSum(Marker):
    """``\\sum_{p \\in S} f(p)`` / ``\\prod``: (op, Lambda(p, f(p)), S), computed once S is finite."""

    def _sympystr(self, printer):
        op, lam, S_ = self.args
        return f"{op}({printer._print(lam.expr)} for {lam.variables[0]} in {printer._print(S_)})"

    def _latex(self, printer):
        op, lam, S_ = self.args
        return (rf"\\{op}_{{{printer._print(lam.variables[0])} \\in {printer._print(S_)}}} "
                rf"{printer._print(lam.expr)}")


class EvalSubs(Marker):
    _fmt = "({a0})|_{a1}={a2}"
    _lfmt = r"\left. {a0} \right|_{{{a1}={a2}}}"


class EvalBetween(Marker):
    _fmt = "({a0})|_{a1}^{a2}"
    _lfmt = r"\left. {a0} \right|_{{{a1}}}^{{{a2}}}"


class Rel(Marker):
    """A relation openlab can only represent (``\\propto``, ``\\sim`` ...)."""

    def _sympystr(self, printer):
        return f"{printer._print(self.args[1])} {self.args[0]} {printer._print(self.args[2])}"

    def _latex(self, printer):
        return rf"{printer._print(self.args[1])} \{self.args[0]} {printer._print(self.args[2])}"


class Mid(Marker):
    """``a \\mid b``: divisibility, or a conditional inside ``P(...)``."""
    _fmt = "{a0} | {a1}"
    _lfmt = r"{a0} \mid {a1}"


class Quant(Marker):
    """``\\forall x \\in S, P``: represented, never executed."""

    def _sympystr(self, printer):
        return f"{self.args[0]} {printer._print(self.args[1])}: {printer._print(self.args[2])}"

    def _latex(self, printer):
        return rf"\{self.args[0]} {printer._print(self.args[1])} \colon\; {printer._print(self.args[2])}"


class System(Marker):
    """Simultaneous equations from ``\\begin{cases}``."""
    _fmt = "system({args})"

    def _latex(self, printer):
        rows = r" \\ ".join(printer._print(a) for a in self.args)
        return rf"\begin{{cases}} {rows} \end{{cases}}"


class Optim(Marker):
    """``Optim(op, body, var_or_domain_or_None)``."""

    def _sympystr(self, printer):
        op, body, spec = self.args
        sp = "" if spec == S.Zero else f"_{{{printer._print(spec)}}}"
        return f"{op}{sp} {printer._print(body)}"

    def _latex(self, printer):
        op, body, spec = self.args
        sp = "" if spec == S.Zero else f"_{{{printer._print(spec)}}}"
        return rf"\{op}{sp} {printer._print(body)}"


class Nabla(Marker):
    """``Nabla(kind, body)``: grad / div / curl / laplacian."""

    def _sympystr(self, printer):
        return f"{self.args[0]}({printer._print(self.args[1])})"


class Subset(Marker):
    _fmt = "{a0} ⊆ {a1}"
    _lfmt = r"{a0} \subseteq {a1}"


class Congruence(Marker):
    _fmt = "{a0} ≡ {a1} (mod {a2})"
    _lfmt = r"{a0} \equiv {a1} \pmod{{{a2}}}"


_FUNCS = {
    "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan, "cot": sympy.cot,
    "sec": sympy.sec, "csc": sympy.csc, "sinh": sympy.sinh, "cosh": sympy.cosh,
    "tanh": sympy.tanh, "coth": sympy.coth, "sech": sympy.sech, "csch": sympy.csch,
    "arcsin": sympy.asin, "arccos": sympy.acos, "arctan": sympy.atan, "arccot": sympy.acot,
    "arcsec": sympy.asec, "arccsc": sympy.acsc,
    "arcsinh": sympy.asinh, "arccosh": sympy.acosh, "arctanh": sympy.atanh,
    "exp": sympy.exp, "Re": sympy.re, "Im": sympy.im, "arg": sympy.arg, "sgn": sympy.sign,
    "lg": lambda x: sympy.log(x, 10),
    "max": sympy.Max, "min": sympy.Min, "gcd": sympy.gcd, "lcm": sympy.lcm,
}
_INVERSE = {
    "sin": sympy.asin, "cos": sympy.acos, "tan": sympy.atan, "cot": sympy.acot,
    "sec": sympy.asec, "csc": sympy.acsc, "sinh": sympy.asinh, "cosh": sympy.acosh,
    "tanh": sympy.atanh, "coth": sympy.acoth, "sech": sympy.asech, "csch": sympy.acsch,
}
_DECO_SUFFIX = {"bar": "bar", "overline": "bar", "hat": "hat", "widehat": "hat",
                "tilde": "tilde", "widetilde": "tilde", "underline": "under",
                "breve": "breve", "check": "check", "acute": "acute", "grave": "grave",
                "mathring": "ring"}
_DECO_STRIP = {"vec", "mathbf", "boldsymbol", "bm", "mathcal", "mathfrak", "mathsf",
               "mathtt", "mathbb"}
_BLACKBOARD_SETS = {"R": S.Reals, "N": S.Naturals0, "Z": S.Integers, "Q": S.Rationals,
                    "C": S.Complexes}

_DIFF_RE = re.compile(
    rf"(?:\\mathrm\{{d\}}|\\text\{{d\}}|d)\s*(?:\{{([a-zA-Z])\}}|([a-zA-Z])|\\({GREEK_NAMES}))"
    r"(?:_(?:\{{([a-zA-Z0-9]+)\}}|([a-zA-Z0-9])))?")
_DECO_RE = re.compile(r"\\([a-zA-Z]+)(?![a-zA-Z])\s*(?:\{((?:[^{}]|\{[^{}]*\})*)\}|(\\[a-zA-Z]+)|([a-zA-Z]))")


@dataclass
class _FPow:
    value: sympy.Basic


@dataclass
class _DPow:
    n: int


@dataclass
class _Limits:
    lo: sympy.Basic
    hi: sympy.Basic


@dataclass
class _Args:
    items: list


@dataclass
class _Pmod:
    n: sympy.Basic


@dataclass
class _NormSub:
    kind: str


def clean_name(body: str) -> str:
    """``\\text{tot\\_sum}`` -> ``tot_sum``; spaces become underscores."""
    return body.replace("\\_", "_").replace("\\", "").strip().replace(" ", "_")


def _is_matrix(x) -> bool:
    return isinstance(x, MatrixBase)


def _call(fn, *args):
    """``fn(*args)`` unevaluated when ``fn`` allows it (``sin(pi/2)`` stays ``sin(pi/2)``)."""
    try:
        return fn(*args, evaluate=False)
    except TypeError:
        return fn(*args)


def _flat(cls, a, b):
    left = list(a.args) if isinstance(a, cls) else [a]
    return cls(*left, b, evaluate=False)


def _mat_now(a, b, add=False) -> bool:
    if _is_matrix(a) and _is_matrix(b):
        return False     # deferred (Star / Plus) so the interpretation shows the product
    if add:          # matrix + scalar broadcasts: decided (with a note) in semantics
        return False
    return ((_is_matrix(a) and getattr(b, "is_number", False))
            or (_is_matrix(b) and getattr(a, "is_number", False)))


def _guard_set(a, b, what):
    if isinstance(a, sympy.Set) or isinstance(b, sympy.Set):
        raise UnsupportedError(f"a set cannot be used in {what}",
                               hint=r"sets combine with \cup, \cap, \setminus and are tested with \in / \subseteq.")


def _vecify(a):
    """A parenthesised tuple in arithmetic is a vector: (1, 2) + (3, 4)."""
    return ImmutableMatrix(list(a)) if isinstance(a, Tuple) else a


def _add(a, b):
    a, b = _vecify(a), _vecify(b)
    _guard_set(a, b, "a sum")
    if _mat_now(a, b, add=True):
        return a + b
    if _is_matrix(a) or _is_matrix(b):
        return Plus(a, b)
    return _flat(Add, a, b)


def _mul(a, b):
    a, b = _vecify(a), _vecify(b)
    _guard_set(a, b, "a product")
    if _mat_now(a, b):
        return a * b
    if _is_matrix(a) or _is_matrix(b):
        return Star(a, b)
    return _flat(Mul, a, b)


def _pow(a, b):
    if isinstance(a, sympy.Set):
        if getattr(b, "is_Integer", False) and b > 0:
            return sympy.ProductSet(*[a] * int(b))
        return Rel(Symbol("power"), a, b)
    _guard_set(a, b, "a power")
    if _is_matrix(a):
        if b == Symbol("T"):
            return a.T
        if getattr(b, "is_Integer", False):
            return MPow(a, b)
        raise UnsupportedError(
            f"a matrix can only be raised to an integer power or transposed (^T), not ^{b}")
    return Pow(a, b, evaluate=False)


def _div(a, b):
    if _is_matrix(a) or _is_matrix(b):
        return a / b
    return Mul(a, Pow(b, S.NegativeOne, evaluate=False), evaluate=False)


def _mulall(factors):
    out = factors[0]
    for f in factors[1:]:
        out = _mul(out, f)
    return out


def diff_var(tok: str) -> Symbol:
    m = _DIFF_RE.fullmatch(str(tok).strip())
    if not m:
        raise ParseError(f"could not read the differential {tok!r}")
    name = m.group(1) or m.group(2) or m.group(3)
    sub = m.group(4) or m.group(5)
    return Symbol(f"{name}_{sub}" if sub else name)


def _first(parts, cls):
    return next((x for x in parts if isinstance(x, cls)), None)


@v_args(inline=True)
class ToSympy(Transformer):
    """Parse tree -> SymPy. Collects human-readable ``notes`` about policy decisions."""

    def __init__(self):
        super().__init__()
        self.notes: list[str] = []

    def note(self, msg: str) -> None:
        if msg not in self.notes:
            self.notes.append(msg)

    # --- names and literals ------------------------------------------------
    def sym(self, tok):
        return Symbol(str(tok))

    def greek(self, tok):
        name = str(tok)[1:]
        if name in ("imath", "jmath"):
            return sympy.I
        return Symbol(name)

    def textname(self, tok):
        body = re.match(r"\\[a-zA-Z]+\{(.*)\}", str(tok)).group(1)
        name = clean_name(body)
        if name in ("True", "true"):
            return sympy.true
        if name in ("False", "false"):
            return sympy.false
        return Symbol(name)

    def deco(self, tok):
        m = _DECO_RE.match(str(tok))
        cmd = m.group(1)
        inner = m.group(2) if m.group(2) is not None else (m.group(3) or m.group(4))
        inner = inner.strip()
        font = re.fullmatch(r"\\(?:mathrm|text|mathit|textit|mathbf|operatorname)\s*\{(.*)\}", inner)
        if font:
            # A font wrapper says "this is one name", so the expression test below must not see inside it:
            # `\bar{\mathrm{t2band}}` is the mean of a column called t2band, not the conjugate of 2·b·a·n·d.
            inner = clean_name(font.group(1))
        elif cmd in ("bar", "overline") and re.search(r"[-+*/^()]|\\frac|\\sqrt|\d\s*[a-zA-Z]", inner):
            from .parse import parse_expr                # \overline{3 + 4i}: the conjugate of an expression
            e = parse_expr(inner)[0]
            if not isinstance(e, Symbol):                # the `-` in `\bar{X_{t-1}}` is inside a name
                self.note(r"\overline{…} of an expression read as its complex conjugate")
                return sympy.conjugate(e, evaluate=False)
        if inner.startswith("\\"):
            base = Symbol(inner[1:])
        else:
            base = Symbol(clean_name(inner))
        if inner.isdigit():
            return Integer(inner)
        if cmd == "mathbb" and base.name in OPERATOR_NAMES:
            return Symbol(OPERATOR_NAMES[base.name])
        if cmd == "mathbb" and base.name in _BLACKBOARD_SETS:
            return _BLACKBOARD_SETS[base.name]
        if cmd in _DECO_STRIP:
            return base
        if cmd in ("dot", "ddot", "dddot"):
            n = {"dot": 1, "ddot": 2, "dddot": 3}[cmd]
            return Derivative(base, (Symbol("t"), n))
        return Deco(base, Symbol(cmd))

    def deltaq(self, tok):
        return Symbol("Delta")   # combined with the following symbol in `imul` (see below)

    def subscripted(self, base, sub):
        s = str(sub).strip()
        if s.startswith("\\"):
            s = s[1:]
        s = s.replace("\\", "").replace(" ", "")
        if isinstance(base, Deco):
            inner, cmd = base.args
            return Deco(self.subscripted(inner, sub), cmd)
        if not isinstance(base, Symbol):
            raise UnsupportedError(f"cannot subscript {base}")
        if re.fullmatch(r"[A-Za-z0-9]+", s):
            return Symbol(f"{base.name}_{s}")
        return Symbol(f"{base.name}_{{{s}}}")

    def supsub(self, base, exponent, sub):
        return _pow(self.subscripted(base, sub), exponent)

    def number(self, tok):
        text = str(tok).replace("\\,", "")
        if "e" in text.lower():
            mant, ex = re.split(r"[eE]", text)
            if "." in mant:
                return sympy.Float(text)
            return Integer(mant) * Integer(10) ** Integer(ex)
        return Integer(text) if "." not in text else sympy.Float(text)

    def const(self, tok):
        return {r"\pi": sympy.pi, r"\infty": sympy.oo}.get(str(tok), S.EmptySet)

    def subchar(self, tok):
        s = str(tok)
        return Integer(s) if s.isdigit() else Symbol(s)

    def group(self, e):
        return e

    def tuple_(self, *items):
        return Tuple(*items)

    # --- arithmetic --------------------------------------------------------
    def add(self, a, b):
        return _add(a, b)

    def sub(self, a, b):
        return _add(a, -_vecify(b))

    def pm(self, a, op, b):
        return PlusMinus(a, b) if str(op) == r"\pm" else PlusMinus(-a, -b)

    def mul(self, a, op, b):
        if str(op) == r"\cdot" and not (a.is_number and b.is_number):
            return Dot(a, b)
        return _mul(a, b)

    def times(self, a, _op, b):
        if a.is_number and b.is_number:
            return _mul(a, b)
        return Times(a, b)

    def imul(self, a, b):
        if isinstance(a, Compose):          # (f \circ g)(x) and (f \circ g) x are applications
            return Apply(a, b)
        if isinstance(a, Transform) and isinstance(b, Symbol):   # \mathcal{L}\{f\}(s)
            return Transform(a.args[0], a.args[1], a.args[2], b)
        if isinstance(a, Mul) and a.args and a.args[-1] == Symbol("Delta") and (
                isinstance(b, Symbol) or (isinstance(b, Pow) and isinstance(b.base, Symbol))):
            return _flat(Mul, Mul(*a.args[:-1], evaluate=False) if len(a.args) > 2 else a.args[0],
                         self.imul(a.args[-1], b))          # m c \Delta T: the Delta is T's
        if isinstance(a, Mul) and a.args and isinstance(b, Symbol) and isinstance(a.args[-1], Pow) \
                and a.args[-1].base == Symbol("Delta") and a.args[-1].exp == S.NegativeOne:
            inv = Pow(self.imul(Symbol("Delta"), b), S.NegativeOne, evaluate=False)   # \Delta x / \Delta t
            return Mul(*a.args[:-1], inv, evaluate=False)
        if a == Symbol("Delta") and isinstance(b, Pow) and isinstance(b.base, Symbol) \
                and not b.base.name.startswith("Delta"):         # \Delta x^2 = (\Delta x)^2
            return Pow(self.imul(a, b.base), b.exp, evaluate=False)
        if a == Symbol("Delta") and isinstance(b, Symbol) and not b.name.startswith("Delta"):
            name = f"Delta_{b.name}"
            DISPLAY_NAMES[Symbol(name)] = rf"\Delta {sympy.latex(b)}"
            return Symbol(name)
        if isinstance(b, Mul) and b.args and a.is_Integer and isinstance(b.args[0], Integer) \
                and b.args[-1].is_Pow and b.args[-1].exp == S.NegativeOne:
            self.note(rf"read as {a}·({sympy.sstr(b)}); a mixed number is {a} + \frac{{...}}{{...}}")
        return _mul(a, b)

    def div(self, a, _op, b):
        return _div(a, b)

    def modop(self, a, _op, b):
        return _call(sympy.Mod, a, b)

    def kron(self, a, _op, b):
        return Kron(a, b)

    def signed(self, sign, x):
        if str(sign) == "-" and isinstance(x, sympy.Mod) and x.args[0].is_Number and x.args[0] > 0:
            self.note(f"-{x.args[0]} mod {x.args[1]} read as (-{x.args[0]}) mod {x.args[1]}")
            return sympy.Mod(-x.args[0], x.args[1])      # -7 \bmod 3 is 2, as a CAS reads it
        return -x if str(sign) == "-" else x

    def pow(self, a, b):
        if isinstance(a, Apply):          # f(x)^2 or P(1+r)^t: decided once names are known
            return CallPow(a, b)
        return _pow(a, b)

    def specialexp(self, a, tok):
        t = str(tok).replace("{", "").replace("}", "").replace(" ", "")
        if t == r"\circ":
            self.note("° read as degrees (converted to radians)")
            return _mul(a, Mul(sympy.pi, Pow(Integer(180), S.NegativeOne, evaluate=False), evaluate=False))
        if t in (r"\top", r"\intercal"):
            return _pow(a, Symbol("T"))
        if r"\prime" in t:
            return Prime(a, Integer(t.count(r"\prime")))
        return Adjoint(a)

    def factorial(self, x, _tok=None):
        return _call(sympy.factorial, x)

    def factorial2(self, x, _tok=None):
        return sympy.factorial2(x)

    def pm_unary(self, _op, b):
        return PlusMinus(S.Zero, b)

    def compose(self, a, _op, b):
        return Compose(a, b)

    def divides(self, a, b):
        return Mid(a, b)

    def ndivides(self, a, b):
        return sympy.Not(Mid(a, b))

    def given(self, a, b):
        return Mid(a, b)

    def optim_char(self, tok, sub, body):
        return self.optim(tok, self.subchar(sub), body)

    def optim_greek(self, tok, sub, body):
        return self.optim(tok, self.greek(sub), body)

    def optspec_to(self, var, _to, target):
        return sympy.Tuple(var, target)

    def quant_bare(self, tok, spec):
        return Quant(Symbol(str(tok)[1:]), spec, sympy.true)

    def prime(self, x, tok):
        return Prime(x, Integer(len(str(tok))))

    def percent(self, x):
        return Mul(x, Pow(Integer(100), S.NegativeOne, evaluate=False), evaluate=False)

    def frac(self, _tok, n, d):
        return _div(n, d)

    def binom(self, *parts):
        n, k = [p for p in parts if not isinstance(p, Token)]
        return _call(sympy.binomial, n, k)

    def ncr(self, n, k):
        return sympy.binomial(n, k)

    def npr(self, n, k):
        return _div(_call(sympy.factorial, n), _call(sympy.factorial, _add(n, -k)))

    def sqrt(self, _tok, x):
        return Pow(x, S.Half, evaluate=False)

    def nroot(self, _tok, n, x):
        if getattr(n, "is_Integer", False) and int(n) % 2 == 1:
            if getattr(x, "is_number", False) and getattr(x, "is_negative", False):
                self.note(rf"\\sqrt[{n}] of a negative number is read as the real root")
            return RealRoot(x, n, evaluate=False)
        return Pow(x, Pow(n, S.NegativeOne, evaluate=False), evaluate=False)

    def absval(self, e):
        if isinstance(e, sympy.Set):
            return Bars(e)
        if e.is_number:
            return sympy.Abs(e)
        return Bars(e)

    def normsub(self, spec):
        s = str(spec) if isinstance(spec, Token) else sympy.sstr(spec)
        return _NormSub(s.strip())

    def norm(self, e, *rest):
        kind = rest[0].kind if rest else "2"
        return Norm(e, Symbol(kind))

    def floor(self, e):
        return _call(sympy.floor, e)

    def ceil(self, e):
        return _call(sympy.ceiling, e)

    def inner(self, args):
        return Inner(*args)

    # --- relations, sets, logic ---------------------------------------------
    _RELS = {"<": sympy.Lt, ">": sympy.Gt, r"\le": sympy.Le, r"\leq": sympy.Le,
             r"\ge": sympy.Ge, r"\geq": sympy.Ge, r"\ne": sympy.Ne, r"\neq": sympy.Ne,
             "=": sympy.Eq, r"\approx": sympy.Eq}

    def _rel(self, op, a, b):
        op = str(op)
        if op in self._RELS:
            return self._RELS[op](a, b, evaluate=False)
        if op in (r"\in", r"\notin"):
            c = sympy.Contains(a, b, evaluate=False) if isinstance(b, sympy.Set) else Rel(Symbol("in"), a, b)
            return sympy.Not(c) if op == r"\notin" else c
        if op in (r"\subset", r"\subseteq"):
            return Subset(a, b)
        if op in (r"\supset", r"\supseteq"):
            return Subset(b, a)
        if op == r"\mid":
            return Mid(a, b)
        if op == r"\nmid":
            return sympy.Not(Mid(a, b))
        if op == r"\colon":
            return Rel(Symbol("colon"), a, b)
        return Rel(Symbol(op[1:]), a, b)

    def chain(self, first, *rest):
        parts = [first]
        rels = []
        for i in range(0, len(rest), 2):
            rels.append(self._rel(rest[i], parts[-1], rest[i + 1]))
            parts.append(rest[i + 1])
        if len(rels) == 1:
            return rels[0]
        return sympy.And(*rels, evaluate=False)

    def pmod(self, n):
        return _Pmod(n)

    def congruence(self, a, _eq, b, pm):
        return Congruence(a, b, pm.n)

    def setop(self, a, op, b):
        op = str(op)
        f = {r"\cup": sympy.Union, r"\cap": sympy.Intersection, r"\setminus": sympy.Complement,
             r"\sqcup": sympy.Union, r"\triangle": sympy.SymmetricDifference}[op]
        if isinstance(a, sympy.Set) and isinstance(b, sympy.Set):
            return f(a, b, evaluate=False)
        return Rel(Symbol(op[1:]), a, b)

    def logic(self, a, op, b):
        op = str(op)
        f = {r"\land": sympy.And, r"\wedge": sympy.And, r"\lor": sympy.Or, r"\vee": sympy.Or,
             r"\implies": sympy.Implies, r"\Rightarrow": sympy.Implies,
             r"\Longrightarrow": sympy.Implies, r"\iff": sympy.Equivalent,
             r"\Leftrightarrow": sympy.Equivalent, r"\Longleftrightarrow": sympy.Equivalent,
             r"\oplus": sympy.Xor, r"\veebar": sympy.Xor}[op]
        try:
            return f(a, b, evaluate=False)
        except TypeError:
            raise ParseError(f"`{op}` needs a statement on each side (like x > 0), not "
                             f"{sympy.sstr(a)} and {sympy.sstr(b)}",
                             hint=r"connect conditions or equations: `x > 0 \land x < 5`.") from None

    def lnot(self, _tok, a):
        return sympy.Not(a, evaluate=False)

    def quant(self, tok, spec, body):
        return Quant(Symbol(str(tok)[1:]), spec, body)

    def emptyset(self, *_):
        return S.EmptySet

    def finiteset(self, *parts):
        return sympy.FiniteSet(*[x for x in parts if not isinstance(x, Token)])

    def condset_multi(self, _l, var, *rest):
        conds = [c for c in rest[:-1] if not isinstance(c, Token)]
        img = self._set_image(var, conds)
        if img is not None:
            return img
        return self.condset(_l, var, sympy.And(*conds), rest[-1])

    @staticmethod
    def _membership(c):
        if isinstance(c, sympy.Contains) and isinstance(c.args[0], Symbol):
            return c.args[0], c.args[1]
        if isinstance(c, Rel) and str(c.args[0]) == "in" and isinstance(c.args[1], Symbol):
            return c.args[1], c.args[2]
        return None

    def _set_image(self, head, conds):
        """``\\{(p, p+2) : p \\in S, …\\}``: a head that is not a plain variable (or conditions openlab must
        resolve first), with a membership giving the domain -> SetImage, listed once S is finite."""
        m = self._membership(head)
        if m is not None and isinstance(head, Rel):         # \{n \in \{4, \ldots, 60\} : …\}
            v, S_ = m
            return SetImage(sympy.Lambda(v, Tuple(v, *conds)), S_)
        plain = isinstance(head, Symbol) and not any(isinstance(c, Marker) or (isinstance(c, sympy.Basic) and c.has(Marker))
                                                     for c in conds)
        if plain or isinstance(head, (sympy.Contains, sympy.core.relational.Relational, sympy.And)):
            return None
        for c in conds:
            m = self._membership(c)
            if m and m[0] in getattr(head, "free_symbols", {head}):
                v, S_ = m
                rest = [x for x in conds if x is not c]
                return SetImage(sympy.Lambda(v, Tuple(head, *rest)), S_)
        if isinstance(head, Tuple) and all(isinstance(x, Symbol) for x in head):   # \{(k, a) : 1 \le k \le 40, …\}
            return TupleSet(head, Tuple(*conds))
        if not isinstance(head, Symbol):
            raise UnsupportedError("a set of expressions needs the domain of its variable",
                                   hint=r"write `\{(p, p+2) : p \in \operatorname{primes}(200), …\}`.")
        return None

    def intrange(self, _l, a, _dots, c, _r):          # \{1, \ldots, n\}
        return IntRange(a, c, Integer(1))

    def intrange_step(self, _l, a, b, _dots, c, _r):  # \{1, 3, \ldots, 99\}
        return IntRange(a, c, Add(b, -a))

    def iverson_mid(self, a, b):                      # [a | b]
        return Iverson(Mid(a, b), sympy.true)

    def iverson_nmid(self, a, b):                     # [a ∤ b]
        return Iverson(Mid(a, b), sympy.false)

    def iverson(self, cond):                          # [P]: 1 if P holds, else 0
        return Pick(cond, Integer(1), Integer(0))

    def condset(self, _l, var, cond, _r):
        img = self._set_image(var, [cond])
        if img is not None:
            return img
        base = S.UniversalSet
        if isinstance(var, sympy.Contains):
            var, base = var.args
        elif isinstance(var, (sympy.core.relational.Relational, sympy.And)):   # \{k \ge 1 : …\}
            letters = [s_ for s_ in var.free_symbols if isinstance(s_, Symbol)]
            if len(letters) == 1:
                cond, var = sympy.And(var, cond), letters[0]
        if not isinstance(var, Symbol):
            raise UnsupportedError("set-builder notation needs a variable before the bar",
                                   hint=r"write `\{ x \mid x > 0 \}` or `\{ x \in \mathbb{R} : x^2 < 2 \}`.")
        return sympy.ConditionSet(var, cond, base)

    def interval_cc(self, a, b):
        return sympy.Interval(a, b)

    def interval_co(self, a, b):
        return sympy.Interval.Ropen(a, b)

    def interval_oc(self, a, b):
        return sympy.Interval.Lopen(a, b)

    # --- named functions ---------------------------------------------------
    def fpow(self, v):
        return _FPow(Integer(str(v)) if isinstance(v, Token) else v)

    def bareargs(self, *factors):
        return _Args(list(factors))

    def funcchain(self, *funcs):
        return _mulall(list(funcs))

    def _bare_arg(self, name, args: _Args):
        factors = args.items
        arg = _mulall(factors)
        symbolic = [f for f in factors if not getattr(f, "is_number", False)]
        if len(symbolic) >= 2:
            self.note(rf"\{name} {' '.join(map(str, factors))} read as {name}({sympy.sstr(arg)}); "
                      rf"parenthesise (`\{name}(x) y`) if you meant a product")
        return arg

    def _named(self, name: str, fpow: Optional[_FPow], base, args: list):
        p = fpow.value if fpow is not None else None
        if name == "det":
            return Det(args[0])
        if name in ("Pr", "deg", "dim", "ker"):
            return Apply(Symbol(name), *args)
        if name in ("max", "min", "gcd", "lcm"):
            # evaluated in semantics once names are resolved (a single argument may be a data
            # vector); kept whole here so the interpretation shows what was typed
            return Apply(Symbol(name), *args)
        if name in ("ln", "log"):
            if base is not None:
                if getattr(base, "is_number", False) and base in (0, 1):
                    raise UnsupportedError(f"a logarithm needs a base other than 0 and 1 (got {base})")
                r = _call(sympy.log, args[0], base)
            else:
                r = _call(sympy.log, args[0])
                if name == "log":
                    self.note(r"\log read as the natural log (write \log_{10} x for base 10)")
        else:
            fn = _FUNCS[name]
            if p == -1 and name in _INVERSE:
                self.note(rf"\{name}^{{-1}} read as the inverse function, not 1/{name}")
                return _call(_INVERSE[name], args[0])
            r = _call(fn, *args)
        if p is not None and p != -1:
            r = Pow(r, p, evaluate=False)
        elif p == -1:
            r = Pow(r, S.NegativeOne, evaluate=False)
        return r

    def func_bare(self, tok, *parts):
        fpow, args = _first(parts, _FPow), _first(parts, _Args)
        base = _first(parts, sympy.Basic)
        name = str(tok)[1:]
        return self._named(name, fpow, base, [self._bare_arg(name, args)])

    def func_nest(self, tok, *parts):
        fpow = _first(parts, _FPow)
        inner = [x for x in parts if isinstance(x, sympy.Basic)][-1]
        return self._named(str(tok)[1:], fpow, None, [inner])

    def func_base_nest(self, tok, base, *parts):     # \log_2 \log_3 2
        fpow = _first(parts, _FPow)
        inner = [x for x in parts if isinstance(x, sympy.Basic)][-1]
        return self._named(str(tok)[1:], fpow, base, [inner])

    def func_base_bare(self, tok, base, *parts):
        fpow, args = _first(parts, _FPow), _first(parts, _Args)
        name = str(tok)[1:]
        return self._named(name, fpow, base, [self._bare_arg(name, args)])

    def func_paren(self, tok, *parts):
        fpow = _first(parts, _FPow)
        args = _first(parts, list)
        return self._named(str(tok)[1:], fpow, None, args)

    def func_base_paren(self, tok, base, *parts):
        fpow = _first(parts, _FPow)
        args = _first(parts, list)
        return self._named(str(tok)[1:], fpow, base, args)

    @staticmethod
    def _opname(tok) -> Symbol:
        return Symbol(clean_name(re.match(r"\\operatorname\*?\{(.*)\}", str(tok)).group(1)))

    def op_bare(self, tok, args: _Args):
        return Apply(self._opname(tok), _mulall(args.items))

    def op_paren(self, tok, args):
        return Apply(self._opname(tok), *args)

    def op_sub_bare(self, tok, sub, args: _Args):
        return Apply(self._opname(tok), sub, _mulall(args.items))

    def op_sub_paren(self, tok, sub, args):
        return Apply(self._opname(tok), sub, *args)

    def cm_prime(self, tok):
        return CallMod(Symbol("prime"), Integer(len(str(tok))))

    def cm_inverse(self):
        return CallMod(Symbol("inverse"))

    def cm_nderiv(self, n):
        n = str(n)
        return CallMod(Symbol("nderiv"), Integer(n) if n.isdigit() else Symbol(n))

    def cm_iter(self, _tok, n):                   # T^{\circ k}(x): T applied k times
        return CallMod(Symbol("iter"), n)

    def cm_power(self, n):
        return CallMod(Symbol("power"), Integer(str(n)))

    def cm_none(self):
        return None

    def call(self, fname, mod, args):
        if mod is None:
            return Apply(fname, *args)
        return Apply(fname, *args, mod)

    def arglist(self, *exprs):
        return list(exprs)

    # --- big operators -----------------------------------------------------
    def limits(self, lo, hi):
        return _Limits(lo, hi)

    def limits_rev(self, hi, lo):
        return _Limits(lo, hi)

    def integral(self, tok, *parts):
        lim = _first(parts, _Limits)
        body = _first(parts, sympy.Basic)
        diffs = [diff_var(x) for x in parts if isinstance(x, Token) and x.type == "DIFF"]
        signs = {r"\iint": 2, r"\iiint": 3}.get(str(tok), 1)
        if len(diffs) < signs:
            raise UnsupportedError(
                f"`{tok}` needs {signs} differentials; a single `d{diffs[0]}` names a region, not variables",
                hint=r"write the variables and bounds: `\int_0^1 \int_0^1 \int_0^1 f \, dx \, dy \, dz`.")
        if str(tok) == r"\oint":
            self.note(r"\oint read as an ordinary integral over the given bounds")
        if lim is not None:
            if len(diffs) > 1:
                raise UnsupportedError("one set of bounds for several differentials",
                                       hint=r"write nested integrals: `\int_0^1 \int_0^1 x y \, dx \, dy`.")
            return Integral(body, (diffs[0], lim.lo, lim.hi))
        return Integral(body, *diffs)

    def integral_one(self, tok, *parts):
        return self.integral(tok, *parts, S.One)

    def integral_region(self, _tok, region, *parts):
        raise UnsupportedError(f"an integral over a region `{region}` cannot be computed",
                               hint=r"give explicit bounds: `\int_0^1 \int_0^{1-x} f \, dy \, dx`.")

    @staticmethod
    def _index_body(body, var: Symbol):
        suffix = "_" + var.name
        rep = {s: sympy.IndexedBase(s.name[: -len(suffix)])[var]
               for s in body.free_symbols
               if isinstance(s, Symbol) and s.name.endswith(suffix) and len(s.name) > len(suffix)}
        shifted = re.compile(rf"(.+)_\{{{re.escape(var.name)}\s*([+-])\s*(\d+)\}}")   # x_{i+1}, x_{i-1}
        for s in body.free_symbols:
            m = shifted.fullmatch(getattr(s, "name", ""))
            if m:
                off = int(m.group(3)) * (1 if m.group(2) == "+" else -1)
                rep[s] = sympy.IndexedBase(m.group(1))[var + off]
        scaled = re.compile(rf"([A-Za-z]+)_\{{?(\d+)\s*{re.escape(var.name)}(?:\s*([+-])\s*(\d+))?\}}?")
        for s in body.free_symbols:                               # x_{2k}, x_{2k+1}
            m = scaled.fullmatch(getattr(s, "name", ""))
            if m and s not in rep:
                off = int(m.group(4) or 0) * (1 if m.group(3) != "-" else -1)
                rep[s] = sympy.IndexedBase(m.group(1))[int(m.group(2)) * var + off]
        return body.xreplace(rep) if rep else body

    def sum_prod(self, tok, var, lo, hi, body):
        op = Sum if str(tok) == r"\sum" else Product
        return op(self._index_body(body, var), (var, lo, hi))

    def sum_prod_rev(self, tok, hi, var, lo, body):
        return self.sum_prod(tok, var, lo, hi, body)

    _SUM_REL = re.compile(r"(\\leq?|\\geq?|\\lt|\\gt|<|>)(?![a-zA-Z])")
    _SUM_VAR = r"(?:[A-Za-z]|\\[A-Za-z]+)"

    @staticmethod
    def _split0(text):
        parts, depth, cur = [], 0, ""
        for ch in text:
            depth += ch in "{(["
            depth -= ch in "})]"
            if ch == "," and depth == 0:
                parts.append(cur)
                cur = ""
            else:
                cur += ch
        return [p for p in (parts + [cur]) if p.strip()]

    def sum_cond(self, tok, sub, body):
        """``\\sum_{p \\le x, p \\in \\mathbb{P}}``, ``\\prod_{p \\mid n, p > 2}``, ``\\sum_{k \\in S}``: the
        index and its domain read from the conditions under the operator."""
        from .parse import parse_expr
        op = "sum" if str(tok) == r"\sum" else "prod"
        text = re.sub(r"^_\s*\{", "", str(sub))[:-1]
        if re.fullmatch(rf"\s*{self._SUM_VAR}\s*", text):           # \sum_{i} x_i: no domain at all
            return self.sum_unbounded(tok)
        text = re.sub(r"\\[,;:!]|\\quad|\\qquad|\\ |~|\\text\{\s*and\s*\}|\\land|\\wedge", " ", text)

        def px(t):
            return parse_expr(t.strip())[0]
        V = self._SUM_VAR
        facts = {"prime": set(), "div": [], "set": [], "ne": [], "chains": []}
        for piece in self._split0(text):
            pc = piece.strip()
            m = re.fullmatch(rf"({V})\s*\\in\s*\\mathbb\s*\{{?\s*P\s*\}}?", pc)
            if m:
                facts["prime"].add(m.group(1))
                continue
            m = re.fullmatch(rf"({V})\s*\\mid\s*(.+)", pc, re.S)
            if m:
                facts["div"].append((m.group(1), px(m.group(2))))
                continue
            m = re.fullmatch(rf"({V})\s*\\in\s*(.+)", pc, re.S)
            if m:
                facts["set"].append((m.group(1), px(m.group(2))))
                continue
            m = re.fullmatch(rf"({V})\s*\\neq?\s*(.+)", pc, re.S)
            if m:
                facts["ne"].append((m.group(1), px(m.group(2))))
                continue
            bits = self._SUM_REL.split(pc)
            if len(bits) >= 3:
                facts["chains"].append(([b.strip() for b in bits[0::2]], bits[1::2]))
                continue
            raise UnsupportedError(f"`\\{op}_{{{text.strip()}}}`: could not read the condition `{pc}`",
                                   hint=r"write bounds (`2 < p \le 1000`), `p \text{ prime}`, `d \mid n`, or `k \in S`.")
        named = facts["prime"] | {v for v, _ in facts["div"]} | {v for v, _ in facts["set"]} | {v for v, _ in facts["ne"]}
        letters = set()
        for items, _ in facts["chains"]:
            letters |= {it for it in items if re.fullmatch(V, it)}
        body_names = {s_.name for s_ in getattr(body, "free_symbols", ())}
        if len(named) == 1:
            vname = next(iter(named))
        else:
            mids = {items[1] for items, _ in facts["chains"] if len(items) == 3 and re.fullmatch(V, items[1])}
            pool = (named or mids or letters)
            inbody = {v for v in pool if px(v).name in body_names or any(
                n_.endswith("_" + px(v).name) for n_ in body_names)}
            cands = inbody or pool
            if len(cands) != 1:
                raise UnsupportedError(f"`\\{op}_{{{text.strip()}}}`: which letter is the index?",
                                       hint=rf"write it as `\{op}_{{k=1}}^{{n}}` or make the index clear, e.g. `p \le x, p \text{{ prime}}`.")
            vname = next(iter(cands))
        var = px(vname)
        if vname in ("p", "q") and vname not in facts["prime"] and not facts["set"]:
            self.note(f"{vname} runs over every whole number in range (not only primes); "
                      rf"add `{vname} \text{{ prime}}` for primes")
        lo = hi = None
        conds = []
        los, his = [], []
        for items, ops in facts["chains"]:
            if vname not in items:
                for a_, o_, b_ in zip(items, ops, items[1:]):
                    conds.append(self._sum_rel(o_, px(a_), px(b_)))
                continue
            i = items.index(vname)
            if i > 0:
                a_, o_ = px(items[i - 1]), ops[i - 1]
                (los if o_ in ("<", r"\lt", r"\le", r"\leq") else his).append(self._bound(o_, a_, before=True))
            if i + 1 < len(items):
                b_, o_ = px(items[i + 1]), ops[i]
                (his if o_ in ("<", r"\lt", r"\le", r"\leq") else los).append(self._bound(o_, b_, before=False))
        from .semantics import _NT
        if vname in facts["prime"]:
            conds.append(sympy.Eq(_NT["isprime"](var), 1))
            los.append(Integer(2))
        for v_, n_ in facts["div"]:
            conds.append(sympy.Eq(sympy.Mod(n_, var), 0))
            los.append(Integer(1))
            his.append(sympy.Abs(n_))
        for v_, e_ in facts["ne"]:
            conds.append(sympy.Ne(var, e_))
        neutral = Integer(0) if op == "sum" else Integer(1)
        body = self._index_body(body, var)
        fbody = Pick(sympy.And(*conds), body, neutral) if conds else body
        if facts["set"]:                                  # k ∈ S: the other conditions filter the terms
            S_ = facts["set"][0][1]
            if los or his:
                fbody = Pick(sympy.And(*([var >= l_ for l_ in los] + [var <= h_ for h_ in his])), fbody, neutral)
            return SetSum(Symbol(op), sympy.Lambda(var, fbody), S_)
        if not los or not his:
            raise UnsupportedError(f"`\\{op}_{{{text.strip()}}}` needs {'a lower' if not los else 'an upper'} bound "
                                   f"for {vname} to be computable",
                                   hint=rf"give both ends (`2 \le {vname} \le N`), or say `{vname} \text{{ prime}}` "
                                        r"(primes start at 2) or `d \mid n` (divisors run from 1 to n).")
        lo = los[0] if len(los) == 1 else sympy.Max(*los)
        hi = his[0] if len(his) == 1 else sympy.Min(*his)
        return (Sum if op == "sum" else Product)(fbody, (var, lo, hi))

    @staticmethod
    def _sum_rel(o, a, b):
        return {"<": sympy.Lt, r"\lt": sympy.Lt, r"\le": sympy.Le, r"\leq": sympy.Le,
                ">": sympy.Gt, r"\gt": sympy.Gt, r"\ge": sympy.Ge, r"\geq": sympy.Ge}[o](a, b)

    @staticmethod
    def _bound(o, e, before):
        """The whole-number bound on the index from `e o index` (before) or `index o e`."""
        strict = o in ("<", ">", r"\lt", r"\gt")
        lower = (o in ("<", r"\lt", r"\le", r"\leq")) == before
        if lower:
            return sympy.floor(e) + 1 if strict else sympy.ceiling(e)
        return sympy.ceiling(e) - 1 if strict else sympy.floor(e)

    def sum_unbounded(self, tok, *_):
        op = str(tok)
        raise UnsupportedError(
            f"`{op}` needs bounds to be computable",
            hint=rf"write `{op}_{{i=1}}^{{n}} x_i` -- an unbounded `{op}_i` has no domain to run over.")

    def limit(self, _tok, var, _to, target, *rest):
        direction = "+-"
        body = None
        for x in rest:
            if isinstance(x, Token) and x.type == "LIMDIR":
                direction = "+" if "+" in str(x) else "-"
            else:
                body = x
        return Limit(body, var, target, direction)

    def dpow(self, tok):
        return _DPow(int(str(tok)))

    def deriv_op(self, *parts):
        return self._deriv(parts)

    def deriv_of(self, *parts):
        return self._deriv(parts)

    def _deriv(self, parts):
        parts = list(parts)
        di = next(i for i, x in enumerate(parts) if isinstance(x, Token) and x.type == "DIFF")
        var = diff_var(parts[di])
        n_num = next((x.n for x in parts[:di] if isinstance(x, _DPow)), 1)
        n_den = next((x.n for x in parts[di:] if isinstance(x, _DPow)), 1)
        if n_num != n_den:
            raise ParseError(f"derivative orders disagree: d^{n_num} over d{var}^{n_den}",
                             hint=rf"write `\frac{{d^{n_num}}}{{d{var}^{n_num}}}`.")
        body = _first(parts, sympy.Basic)
        return Derivative(body, (var, n_num))

    def pvar(self, _tok, sym, *rest):
        return (sym, rest[0].n if rest else 1)

    def pvars(self, *pv):
        return list(pv)

    def partial_op(self, *parts):
        return self._partial(parts)

    def partial_of(self, *parts):
        return self._partial(parts)

    def _partial(self, parts):
        pv = _first(parts, list)
        body = _first(parts, sympy.Basic)
        order = next((x.n for x in parts if isinstance(x, _DPow)), None)
        total = sum(n for _, n in pv)
        if order is not None and order != total:
            raise ParseError(f"partial derivative orders disagree: ∂^{order} over {total} variables",
                             hint=r"`\frac{\partial^2 f}{\partial x \partial y}` -- the top order must equal the number of variables below.")
        return Derivative(body, *pv)

    def partial_sub(self, _tok, sym, body):
        return Derivative(body, sym)

    def optim(self, tok, spec, body):
        op = re.sub(r"\\|\s", "", str(tok))
        return Optim(Symbol(op), body, spec)

    def optim_plain(self, tok, body):
        op = re.sub(r"\\|\s", "", str(tok))
        if isinstance(body, Bars):              # \max |\lambda_i|-style: entry by entry, then the max
            body = AbsEach(body.args[0])
        return Optim(Symbol(op), body, S.Zero)

    def grad(self, _tok, body):
        if isinstance(body, (Dot, Times)):     # \nabla f(1, 2) \cdot \vec u: the gradient, then the dot
            return body.func(self.grad(_tok, body.args[0]), body.args[1])
        return Nabla(Symbol("grad"), body)

    def divergence(self, _tok, _op, body):
        return Nabla(Symbol("div"), body)

    def curl(self, _tok, _op, body):
        return Nabla(Symbol("curl"), body)

    def laplacian(self, _tok, n, body):
        if str(n) != "2":
            raise UnsupportedError(rf"\nabla^{n} is not defined")
        return Nabla(Symbol("laplacian"), body)

    # --- evaluation bars -------------------------------------------------------
    def eval_subs(self, var, val):
        return ("subs", var, val)

    def eval_subs2(self, var, lo, var2, hi):
        return ("between", lo, hi, var)

    def eval_pass(self, spec):
        return spec

    def eval_subs_hi(self, var, lo, hi):          # \left. F \right|_{n=1}^{5} = F(5) − F(1)
        return ("between", lo, hi, var)

    def eval_between(self, lim: _Limits):
        return ("between", lim.lo, lim.hi, None)

    def eval_at(self, val):
        return ("at", val)

    def evalpair(self, var, val):
        return (var, val)

    def eval_multi(self, *pairs):
        return ("multi", [p[0] for p in pairs], [p[1] for p in pairs])

    def eval_point(self, *vals):
        return ("point", vals)

    def eval_tuple(self, *items):
        k = len(items) // 2
        return ("multi", list(items[:k]), list(items[k:]))

    def evalat(self, expr, *rest):
        spec = rest[-1]
        if spec[0] == "subs":
            return EvalSubs(expr, spec[1], spec[2])
        if spec[0] == "at":
            return EvalSubs(expr, S.Zero, spec[1])   # variable decided in semantics
        if spec[0] == "multi":
            return EvalSubs(expr, Tuple(*spec[1]), Tuple(*spec[2]))
        if spec[0] == "point":
            return EvalSubs(expr, S.Zero, Tuple(*spec[1]))
        var = spec[3] if spec[3] is not None else S.Zero
        return EvalBetween(expr, spec[1], spec[2], var)

    def evalat_bracket(self, expr, lim: _Limits):
        return EvalBetween(expr, lim.lo, lim.hi, S.Zero)

    def transform_(self, tok, body):
        t = str(tok)
        kind = "L" if re.search(r"\{\s*L\s*\}", t) else "F"
        if isinstance(body, sympy.FiniteSet) and len(body) == 1:
            body = next(iter(body))
        return Transform(Symbol(kind), body, sympy.true if "-1" in t else sympy.false)

    def unit(self, tok):
        text = re.match(r"\\openlabunit\{(.*)\}", str(tok)).group(1)
        q = _units.parse_unit(text)
        if q is None:
            raise UnsupportedError(f"unknown unit `{text}`")
        return q

    # --- matrices, cases ---------------------------------------------------------
    def mrow(self, *exprs):
        return list(exprs)

    def matrix(self, begin, *parts):
        rows = [p for p in parts if isinstance(p, list)]
        end = next(p for p in parts if isinstance(p, Token) and p.type == "MATEND")
        env_b = re.match(r"\\begin\{(\w+)\}", str(begin)).group(1)
        env_e = re.match(r"\\end\{(\w+)\}", str(end)).group(1)
        if env_b != env_e:
            raise ParseError(rf"\begin{{{env_b}}} is closed by \end{{{env_e}}}",
                             hint=rf"close it with \end{{{env_b}}}.")
        if len({len(r) for r in rows}) != 1:
            raise ParseError("matrix rows have different lengths",
                             hint=r"rows are separated by `\\` and entries by `&`; every row needs the same number of entries.")
        M = ImmutableMatrix(rows)
        if env_b == "vmatrix":
            return Det(M)
        if env_b == "Vmatrix":
            return Norm(M, Symbol("2"))
        return M

    def crow_cond(self, value, cond):
        return (value, cond)

    def crow_plain(self, value):
        return (value, None)

    def cases(self, *rows):
        rows = [r for r in rows if isinstance(r, tuple)]
        conds = [r for r in rows if r[1] is not None]
        plain = [r for r in rows if r[1] is None]
        if not conds and plain and all(isinstance(v, sympy.Eq) for v, _ in plain):
            return System(*[v for v, _ in plain])
        if not conds:
            raise UnsupportedError("`cases` rows need a condition (`value & condition`) or must all be equations",
                                   hint=r"piecewise: `\begin{cases} x^2 & x \ge 0 \\ -x & \text{otherwise} \end{cases}`; a system: `\begin{cases} x + y = 3 \\ x - y = 1 \end{cases}`.")
        if len(plain) > 1 or (plain and rows[-1][1] is not None):
            raise ParseError("only the last `cases` row may lack a condition (the `otherwise` branch)")
        for _, c in conds:
            if isinstance(c, sympy.Expr) and not getattr(c, "is_Relational", False):
                raise ParseError(f"the `cases` condition `{sympy.sstr(c)}` is not a condition (true or false)",
                                 hint=r"write a comparison or a congruence, e.g. `n \bmod 2 = 0`, `x \ge 0`, "
                                      r"or the words `n \text{ even}` / `n \text{ odd}`.")
        pieces = [(v, c) for v, c in conds] + [(v, sympy.true) for v, _ in plain]
        if not plain and len(conds) >= 2:
            res = [self._residue(c) for _, c in conds]
            if all(r is not None for r in res) and len({(x, m) for x, m, _ in res}) == 1:
                x, m, _ = res[0]
                if sorted(r for _, _, r in res) == list(range(m)):
                    pieces[-1] = (pieces[-1][0], sympy.true)
                    self.note(f"the conditions cover every remainder of {sympy.sstr(x)} mod {m}, so for whole "
                              f"numbers the last row is the `otherwise` case")
        return sympy.Piecewise(*pieces, evaluate=False)

    @staticmethod
    def _residue(c):
        """``Eq(Mod(x, m), r)`` -> (x, m, r mod m) for a whole-number modulus, else None."""
        if isinstance(c, sympy.Eq):
            for a, b in (c.args, c.args[::-1]):
                if isinstance(a, sympy.Mod) and a.args[1].is_Integer and a.args[1] > 1 and b.is_Integer:
                    m = int(a.args[1])
                    return a.args[0], m, int(b) % m
        return None
