r"""Execute parsed statements against a namespace.

The ``=`` rule (decided on the parsed left-hand side and the namespace):

* ``x = ...``      a name        -> ASSIGN (bind x). Two exceptions, both shown in the
                                    interpretation line: if x already holds a formula and the
                                    right side is a plain value it is an equation
                                    (``y = 2x+1`` then ``y = 5`` solves for x); if x was just
                                    solved for in this cell it is a check of that solution.
* ``x := ...``     always binds (also ``\coloneqq``, ``\triangleq``)
* ``x, y = 1, 2``  names         -> ASSIGN each
* ``f(x) = ...``   a call        -> DEFINE, unless f is already a function and the right side
                                    is constant or uses other functions (``h(t) = 0``,
                                    ``f(x) = g(x)``): then it is an equation to solve
* ``a_n = 2n+1``   a sequence    -> DEFINE a_n (so ``a_5`` works)
* ``P(A) = 0.3``   an event      -> recorded; ``P(A \cap B)``, ``P(A \mid B)``, ``P(A^c)`` derive
* anything else    ``x^2 = 4``   -> SOLVE: real solutions first; an ODE when it contains y', dy/dx,
                                    ẋ (initial conditions after a comma or ``\text{with}``); a
                                    system for a matrix equation. Identities are checked instead.
* ``a = b = c``                  -> every link checked
* ``a \approx b``                -> checked to the decimals written; never binds
* no ``=``                       -> EXPR / RELATION
* ``X \sim N(0,1)``              -> a random variable (sympy.stats)
* ``x \in \mathbb{R}``           -> ASSUME: later solving and simplifying use it
* ``\forall``, ``\propto``, ...  -> REPR: represented, not executed

Names resolve against the namespace transitively (``y = x^2``, then ``x = 3``, then ``y`` is 9),
never inside the variables of an integral, sum, limit, derivative or evaluation bar.
"""

from __future__ import annotations

import functools
import math
from collections import ChainMap
from fractions import Fraction
import numbers
import re
import signal
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Optional

import mpmath
import sympy
from . import fasteval as _fast
from .. import extensions as _extensions
from sympy import (Basic, Derivative, Eq, Expr, Float, Function, Integer, Integral, Lambda,
                   Limit, Mul, Pow, Product, Subs, Sum, Symbol, Transpose, Tuple)
from sympy.core.function import AppliedUndef
from sympy.core.relational import Relational
from sympy.logic.boolalg import BooleanAtom, BooleanFunction
from sympy.matrices import ImmutableMatrix, MatrixBase

try:
    from sympy.matrices.exceptions import NonInvertibleMatrixError, NonSquareMatrixError, ShapeError
except ImportError:  # older SymPy
    from sympy.matrices.common import NonInvertibleMatrixError, NonSquareMatrixError, ShapeError

from . import interp, render
from . import units as _units
from .errors import OpenlabError, MultipleUnknowns, ParseError, SemanticsError, UnsupportedError
from .grammar import GREEK_NAMES
from .parse import Clause, Statement, display_src, parse_expr, parse_statement, statements
from .teach import did_you_mean
from .transform import (DISPLAY_NAMES, Adjoint, Apply, Bars, CallMod, Compose, Congruence, Deco,
                        Det, Dot, EvalBetween, EvalSubs, Inner, Kron, Marker, Mid, MPow, Nabla,
                        Norm, Optim, Plus, PlusMinus, Prime, Quant, Rel, Star, Subset, System,
                        Times, Transform, CallPow, RealRoot, AbsEach, IntRange, SetSum, Iverson, SetImage, Pick,
                        TupleSet)



def _exact(e):
    """Every decimal as the exact fraction it denotes (1.0000000001 stays 10000000001/10^10; nsimplify rounds it)."""
    return e.xreplace({f: sympy.Rational(str(f)) for f in e.atoms(Float)}) if isinstance(e, Basic) else e

OUT_OF_TIME = "the identity check ran out of time (4 s)"


class Kind(Enum):
    EXPR = "eval:"
    ASSIGN = "assign"
    DEFINE = "define"
    SOLVE = "solve for"
    RELATION = "check:"
    REPR = "represent:"
    ASSUME = "assume"
    NOTE = "note:"          # prose, markdown, a claim with no technique: shown, never run
    PROOF = "prove"         # a claim over a set, proved by a named technique


class Transposed(Marker):
    """``A^T`` when ``T`` is also a bound name: still a transpose. ``args[1]``: conjugate?"""

    def _sympystr(self, printer):
        return f"{printer._print(self.args[0])}^{'H' if self.args[1] else 'T'}"


class NthDerivAt(sympy.Function):
    """The n-th derivative of ``body`` in ``u`` at ``a``, for an order n that is still a
    symbol; it becomes concrete when a finite sum fixes n (Taylor coefficients)."""

    @classmethod
    def eval(cls, body, u, n, a):
        if getattr(n, "is_Integer", False) and n >= 0:
            return sympy.diff(body, u, int(n)).subs(u, a)
        return None


def _set_digits(n):
    """Python refuses to print integers over 4300 digits; openlab prints factorials."""
    get = getattr(sys, "get_int_max_str_digits", None)
    if get is None or n is None:
        return None
    old = get()
    sys.set_int_max_str_digits(n)
    return old


#: Names that are never read from the namespace.
IGNORED = {"In", "Out", "exit", "quit", "get_ipython", "_openlab", "_", "__", "___",
           "_i", "_ii", "_iii", "_ih", "_oh", "_dh"}

_CALCULUS = (Integral, Derivative, Sum, Product, Limit, Transpose)
_STATS_OPS = {"E", "Var", "Cov", "SD", "std", "P", "Pr", "Prob", "Std", "var", "mean", "𝔼", "ℙ", "𝕍"}
_PROB_HEADS = {"P", "Pr", "Prob", "ℙ"}
_OP_ALIAS = {"𝔼": "E", "ℙ": "P", "𝕍": "Var"}


def _callable_value(raw) -> bool:
    """An openlab, SymPy or Python function (not a number, not a class)."""
    if isinstance(raw, (Lambda, sympy.FunctionClass)):
        return True
    return callable(raw) and not isinstance(raw, (Basic, type))
#: Letters conventionally used as function names: `f(x+1)` is a call even when f is unbound.
FUNCTION_LETTERS = {"f", "g", "h", "F", "G", "H", "u", "y", "phi", "psi", "Phi", "Psi",
                    "varphi", "rho"}
_QUIET_HEADS = _STATS_OPS | FUNCTION_LETTERS | {"v", "w", "L", "x", "r", "s"}
_GREEK = set(GREEK_NAMES.split("|"))
_SET_OPS = {"cup": sympy.Union, "cap": sympy.Intersection, "setminus": sympy.Complement,
            "sqcup": sympy.Union, "triangle": sympy.SymmetricDifference}
#: The value of a limit that does not exist.
DNE = Symbol("DNE")
DISPLAY_NAMES[DNE] = r"\text{does not exist}"
#: The value of a divergent series whose terms change sign.
DIVERGES = Symbol("DIVERGES")
DISPLAY_NAMES[DIVERGES] = r"\text{diverges}"


def _std_normal_cdf(z):
    return (1 + sympy.erf(z / sympy.sqrt(2))) / 2


def _operatorname_latex(self, printer, exp=None):
    r"""\operatorname{isprime}(k): SymPy's printer would read a trailing "prime" as a prime mark."""
    name = type(self).__name__
    head = r"\gcd" if name == "gcd" else rf"\operatorname{{{name}}}"
    out = rf"{head}\left({', '.join(printer._print(a) for a in self.args)}\right)"
    return rf"{out}^{{{exp}}}" if exp is not None else out


def _binomial_sums(expr):
    """The binomial theorem: Σ_{j=0}^{k} C(k, j)·c·r^j = c·(1 + r)^k (k a whole number ≥ 0;
    so x^j y^{k−j} gives (x + y)^k). SymPy only sums it under a convergence condition |x| ≤ 1."""
    def fix(node):
        if not (isinstance(node, Sum) and len(node.limits) == 1):
            return node
        j, lo, hi = node.limits[0]
        if lo != 0 or not (getattr(hi, "is_nonnegative", False) and getattr(hi, "is_integer", False)):
            return node
        jj = sympy.Dummy(j.name, integer=True, nonnegative=True)
        f = sympy.expand_power_base(sympy.expand(node.function.xreplace({j: jj})), force=False)
        j = jj
        terms = sympy.Add.make_args(f)
        out = []
        for t in terms:
            bins = [b for b in Mul.make_args(t) if isinstance(b, sympy.binomial)
                    and b.args[1] == j and sympy.expand(b.args[0] - hi) == 0]
            if len(bins) != 1:
                return node
            rest = Mul(*[x for x in Mul.make_args(t) if x is not bins[0]])
            try:
                r = sympy.simplify(sympy.powsimp(rest.xreplace({j: j + 1}) / rest, force=True))
            except Exception:
                return node
            if r.has(j):
                return node
            term = sympy.powsimp(rest.xreplace({j: 0}) * (1 + r) ** hi, force=True)   # k whole: (ab)^k = a^k b^k
            out.append(term.replace(lambda p: isinstance(p, Pow) and p.exp == hi,
                                    lambda p: Pow(sympy.cancel(p.base), hi)))
        return sympy.Add(*out)
    if not isinstance(expr, Basic) or not expr.has(sympy.binomial):
        return expr
    return expr.replace(lambda n: isinstance(n, Sum) and n.function.has(sympy.binomial), fix)


def _max_digits(e) -> int:
    """The number of digits of the largest integer written inside ``e``."""
    best = 0
    for a in e.atoms(sympy.Rational) if isinstance(e, Basic) else ():
        best = max(best, int(abs(a.p)).bit_length() * 30103 // 100000 + 1, int(a.q).bit_length() * 30103 // 100000 + 1)
    return best


def _outer_optims(e) -> list:
    """The max/min/argmax nodes of ``e`` not inside another one."""
    if isinstance(e, Optim):
        return [e]
    out = []
    for a in getattr(e, "args", ()):
        if isinstance(a, Basic):
            out.extend(_outer_optims(a))
    return out


def _free_vars(e) -> set:
    """The letters of ``e`` that are variables: not the head of a call T(…) nor a modifier tag."""
    if not isinstance(e, Basic):
        return set()
    out = {s_ for s_ in e.free_symbols if isinstance(s_, Symbol)}
    for a in e.atoms(Apply):
        h = a.args[0]
        if isinstance(h, Symbol) and not any(h in x.free_symbols for x in a.args[1:]
                                             if not isinstance(x, CallMod)):
            out.discard(h)
    for c in e.atoms(CallMod):
        out.discard(c.args[0])
    for ix in e.atoms(sympy.Indexed):            # c[n]: the list's name is not a variable
        out.discard(ix.base.label)
    for o in e.atoms(Optim, Nabla):              # the operation's tag (min, grad, …)
        out.discard(o.args[0])
    for o in e.atoms(Optim):                     # \max_{2 ≤ j ≤ n−1}: j is bound inside
        v_ = _optim_var(o.args[2])
        if v_ is not None and not any(v_ in x.free_symbols for x in _outside_of(e, o)):
            out.discard(v_)
    return out


def _optim_var(spec):
    """The index of a max/min spec: `j`, `j ∈ S`, or the bare side of `a ≤ j ≤ b`."""
    if isinstance(spec, Symbol):
        return spec
    if isinstance(spec, sympy.Contains) or (isinstance(spec, Rel) and str(spec.args[0]) == "in"):
        v_ = spec.args[0] if isinstance(spec, sympy.Contains) else spec.args[1]
        return v_ if isinstance(v_, Symbol) else None
    if isinstance(spec, (sympy.core.relational.Relational, sympy.And)):
        bare = None
        for c_ in (sympy.And.make_args(spec) if isinstance(spec, sympy.And) else (spec,)):
            sides = {x for x in (getattr(c_, "lhs", None), getattr(c_, "rhs", None)) if isinstance(x, Symbol)}
            bare = sides if bare is None else bare & sides
        return next(iter(bare)) if bare and len(bare) == 1 else None
    return None


def _outside_of(e, node):
    """The parts of e outside ``node`` (to see whether a letter bound in node is also used elsewhere)."""
    if e == node:
        return []
    return [e.xreplace({node: sympy.Dummy()})]


def _finite_sums(expr, cap=200_000, settle=None, hook=None):
    """A sum or product with no free letters whose ranges are finite once the outer indices are
    known (the inner range may depend on them): add or multiply its terms one by one. SymPy's
    ``doit`` looks for a closed form first and gives up on terms like isprime(k)·isprime(2m − k)."""
    count = [0]

    def run(f, limits, is_sum):
        if not limits:
            count[0] += 1
            if count[0] > cap:
                raise OverflowError
            v = f.doit() if isinstance(f, Basic) else f
            if settle is not None and isinstance(v, Basic) and not v.is_number:
                v = settle(v)
            if isinstance(v, Basic) and (_free_vars(v) or v.has(Sum, Product, Marker)):
                raise ValueError
            return v
        k, a, b = limits[-1]
        a, b = sympy.sympify(a).doit(), sympy.sympify(b).doit()
        if not (a.is_Integer and b.is_Integer):
            raise ValueError
        acc = Integer(0) if is_sum else Integer(1)
        for v in range(int(a), int(b) + 1):
            sub = {k: Integer(v)}
            inner = [(kk, aa.xreplace(sub), bb.xreplace(sub)) for kk, aa, bb in limits[:-1]]
            t = run(f.xreplace(sub), inner, is_sum)
            acc = acc + t if is_sum else acc * t
        return acc

    def explicit(node):
        try:
            return _fast.evaluate(node, hook=hook)  # plain whole-number arithmetic: exact and quick
        except (_fast.Unsupported, KeyError, ZeroDivisionError, OverflowError, ValueError, TypeError):
            pass
        v = _vector_sum(node)                      # integer summands: the innermost index as a vector, exact
        if v is not None:
            return v
        try:
            return run(node.function, list(node.limits), isinstance(node, Sum))
        except (ValueError, OverflowError, TypeError):
            return node
    return expr.replace(lambda n: isinstance(n, (Sum, Product)) and not _free_vars(n), explicit)


_VECTOR_OK = (sympy.Add, sympy.Mul, sympy.Pow, sympy.Mod, Integer, Symbol)


def _integer_body(f, indices) -> bool:
    """Integer arithmetic in the indices only: +, ×, non-negative integer powers, mod, integer literals."""
    for n in sympy.preorder_traversal(f):
        if isinstance(n, sympy.Pow):
            if not (n.exp.is_Integer and n.exp >= 0):
                return False
        elif isinstance(n, Symbol):
            if n not in indices:
                return False
        elif not isinstance(n, _VECTOR_OK):
            return False
    return True


def _vector_or_keep(node):
    v = _vector_sum(node)
    return node if v is None else v


def _vector_sum(node, cap=50_000_000):
    """``Σ … f(i, j, k)`` with an integer-arithmetic summand: evaluate the innermost index as one NumPy
    object array (Python integers, so exact), the outer indices in Python loops. About 100x the term-by-term
    path; None when the shape does not fit."""
    try:
        import numpy as np
    except ImportError:                                 # pragma: no cover
        return None
    limits = list(node.limits)
    indices = {k for k, _, _ in limits}
    if not limits or not _integer_body(node.function, indices):
        return None
    inner, outer = limits[0], limits[1:]
    if any(not (sympy.sympify(a).is_Integer and sympy.sympify(b).is_Integer) for _, a, b in outer):
        return None
    total = 1
    for _, a, b in outer:
        total *= max(0, int(b) - int(a) + 1)
    k, lo, hi = inner
    try:
        f = sympy.lambdify([k] + [o[0] for o in outer], node.function, modules=[{"Mod": np.mod}, "numpy"])
        flo = sympy.lambdify([o[0] for o in outer], lo, "math")
        fhi = sympy.lambdify([o[0] for o in outer], hi, "math")
    except Exception:                                   # noqa: BLE001
        return None
    is_sum = isinstance(node, Sum)
    acc = 0 if is_sum else 1
    import itertools
    ranges = [range(int(a), int(b) + 1) for _, a, b in outer]
    seen = 0
    for combo in itertools.product(*ranges) if outer else [()]:
        a, b = flo(*combo), fhi(*combo)
        if not (isinstance(a, int) and isinstance(b, int)):
            return None
        n = b - a + 1
        if n <= 0:
            continue
        seen += n
        if seen > cap:
            return None
        ks = np.arange(a, b + 1, dtype=object)
        vals = f(ks, *combo)
        if not isinstance(vals, np.ndarray):            # the summand does not depend on the inner index
            block = vals * n if is_sum else vals ** n
        else:
            block = vals.sum() if is_sum else int(np.prod(vals))
        acc = acc + block if is_sum else acc * block
    return Integer(int(acc))


def _int_reducer(name, op):
    """gcd / lcm as a SymPy function that evaluates once its arguments are integers, so a term
    gcd(k, 10) inside a sum or product computes when the index is substituted."""
    def eval_(cls, *a):
        if a and all(getattr(x, "is_Integer", False) for x in a):
            return functools.reduce(op, a)
        if name == "gcd" and len(a) == 2:               # gcd(6k + 1, 6) = gcd(1, 6) for whole-number k
            for c_, e_ in (a, a[::-1]):
                if getattr(c_, "is_Integer", False) and c_ != 0 and isinstance(e_, Expr) and e_.free_symbols \
                        and all(s_.is_integer for s_ in e_.free_symbols):
                    r_ = sympy.Mod(sympy.expand(e_), abs(c_))
                    if r_.is_Integer:
                        return sympy.gcd(r_, c_)
        return None
    py = math.gcd if name == "gcd" else math.lcm
    return type(name, (sympy.Function,), {"eval": classmethod(eval_), "_latex": _operatorname_latex,
                                          "_pyfn": staticmethod(lambda *a: functools.reduce(py, a))})


_GCD, _LCM = _int_reducer("gcd", sympy.gcd), _int_reducer("lcm", sympy.lcm)


def _gcd(*a):
    return _GCD(*a)


def _lcm(*a):
    return _LCM(*a)


def _extreme(fn):
    def f(*a):
        if len(a) == 1 and isinstance(a[0], (MatrixBase, Tuple)):
            return fn(*list(a[0]))
        if len(a) == 1 and isinstance(a[0], sympy.Set):
            return a[0].sup if fn is sympy.Max else a[0].inf
        return fn(*a)
    return f


def _median(*a):
    v = a[0] if len(a) == 1 else a                 # median(v) of a vector, or median(a, b, c) of numbers
    xs = sorted(list(v), key=lambda x: float(x))
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def _var(v):
    n = len(v)
    m = sum(v) / n
    return sum((x - m) ** 2 for x in v) / (n - 1)


def _corr(a, b):
    """Pearson correlation of two data vectors."""
    a, b = list(a), list(b)
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    sab = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return sab / sympy.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))


def _percentile(v, p):
    """The p-th percentile with linear interpolation (NumPy's default)."""
    xs = sorted(list(v), key=float)
    pos = (len(xs) - 1) * sympy.nsimplify(p) / 100
    lo = int(sympy.floor(pos))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _mode(v):
    xs = list(v)
    return max(set(xs), key=xs.count)


def _hessian(f, *vs):
    vars_ = list(vs) or sorted((q for q in f.free_symbols if isinstance(q, Symbol)), key=lambda q: q.name)
    return ImmutableMatrix(sympy.hessian(f, vars_))


def _nt(n, fn, name, low=0):
    """A number-theory function of a whole number; symbolic arguments stay symbolic."""
    n = sympy.sympify(n)
    if not n.is_number:
        return Function(name)(n)
    if not n.is_Integer:
        n = n.doit()                               # 2^{32} + 1 arrives unevaluated
    if not (n.is_Integer and n >= low):
        raise SemanticsError(f"\\operatorname{{{name}}} needs a whole number ≥ {low} (got {interp.pyform(n)})")
    return fn(int(n))


_SIEVE = {"n": 0, "bits": bytearray()}


def _sieve_to(n: int):
    """A byte sieve of Eratosthenes up to n (grown on demand, at most 2·10⁸)."""
    n = min(max(n, 1 << 16), 200_000_000)
    if n <= _SIEVE["n"]:
        return
    b = bytearray([1]) * (n + 1)
    b[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if b[i]:
            b[i * i::i] = bytes(len(range(i * i, n + 1, i)))
    _SIEVE["n"], _SIEVE["bits"] = n, b


def _is_prime_int(v: int) -> int:
    if v < 2:
        return 0
    if v <= _SIEVE["n"]:
        return _SIEVE["bits"][v]
    if v <= 200_000_000:
        _sieve_to(max(2 * v, 1 << 20))
        return _SIEVE["bits"][v]
    return 1 if sympy.isprime(v) else 0


def _nt_function(name, fn, low=0, doc=""):
    """A SymPy function that evaluates for a whole-number argument (so it works inside sums)."""
    def eval_(cls, n):
        if n.is_number and not n.is_Integer and n.has(sympy.Add, sympy.Mul, sympy.Pow):
            n = n.doit()                           # an unevaluated 2^{32} + 1
        if n.is_Integer:
            if low is not None and n < low:
                raise SemanticsError(f"\\operatorname{{{name}}} needs a whole number ≥ {low} (got {n})")
            return fn(int(n))
        return None                                # symbolic, or a non-integer point: left as written
    def pyfn(v):
        out = fn(v)
        if not getattr(out, "is_Integer", False):
            raise _fast.Unsupported(name)
        return int(out)
    return type(name, (sympy.Function,), {"eval": classmethod(eval_), "__doc__": doc, "nargs": 1,
                                          "_latex": _operatorname_latex, "_pyfn": staticmethod(pyfn),
                                          "_low": low})



_NT = {
    "isprime": _nt_function("isprime", lambda v: Integer(_is_prime_int(v)), low=None,
                            doc="1 if n is prime, else 0 (the indicator of the primes on ℤ, so it can be summed)"),
    "primepi": _nt_function("primepi", lambda v: Integer(sympy.primepi(v))),
    "prime": _nt_function("prime", lambda v: Integer(sympy.prime(v)), low=1),
    "nextprime": _nt_function("nextprime", lambda v: Integer(sympy.nextprime(v))),
    "prevprime": _nt_function("prevprime", lambda v: Integer(sympy.prevprime(v)), low=3),
    "totient": _nt_function("totient", lambda v: Integer(sympy.totient(v)), low=1),
    "sigma": _nt_function("sigma", lambda v: Integer(sympy.divisor_sigma(v)), low=1),
    "omega": _nt_function("omega", lambda v: Integer(len(sympy.factorint(v))), low=1),
    "Omega": _nt_function("Omega", lambda v: Integer(sum(sympy.factorint(v).values())), low=1),
}


def _frac_part(name, idx):
    """numer / denom: of a number or an explicit fraction; a sequence entry c_n or an unknown letter waits
    (SymPy would say denom(c_n) = 1 before n is known)."""
    def eval_(cls, e):
        if isinstance(e, Basic) and (e.has(sympy.Indexed, AppliedUndef, Marker)
                                     or (isinstance(e, Symbol))):
            return None
        return sympy.fraction(sympy.together(e))[idx]
    return type(name, (sympy.Function,), {"eval": classmethod(eval_), "nargs": 1, "_frac_part": idx,
                                          "_latex": _operatorname_latex})


_NUMER, _DENOM = _frac_part("numer", 0), _frac_part("denom", 1)


def _cf_terms(x, n):
    """The first n partial quotients of a real number, each confirmed at two precisions."""
    n = int(sympy.sympify(n))
    if n < 1 or n > 2000:
        raise SemanticsError("\\operatorname{cf}: ask for between 1 and 2000 terms")
    x = sympy.sympify(x).doit() if isinstance(x, Basic) else sympy.sympify(x)
    if not (isinstance(x, Expr) and x.is_number and x.is_real):
        raise SemanticsError("\\operatorname{cf} needs a real number")
    if x.is_Rational:
        return list(sympy.continued_fraction(x))[:n]

    def run(dps):
        v = sympy.N(x, dps)
        out = []
        with mpmath.workdps(dps):
            v = mpmath.mpf(str(v))
            for _ in range(n):
                a = int(mpmath.floor(v))
                out.append(a)
                frac = v - a
                if frac == 0 or abs(frac) < mpmath.mpf(10) ** (-dps + 10):
                    break
                v = 1 / frac
        return out
    prev = None
    dps = max(60, 3 * n)
    for _ in range(6):
        cur = run(dps)
        if prev is not None and cur[:n] == prev[:n] and len(cur) >= min(n, len(prev)):
            return cur[:n]
        prev, dps = cur, dps * 2
    raise SemanticsError("\\operatorname{cf}: the partial quotients did not settle (is the number rational?)")


def _convergents(x, n):
    a = _cf_terms(x, n)
    out, (p0, q0, p1, q1) = [], (1, 0, a[0], 1)
    out.append(sympy.Rational(p1, q1))
    for ai in a[1:]:
        p0, q0, p1, q1 = p1, q1, ai * p1 + p0, ai * q1 + q0
        out.append(sympy.Rational(p1, q1))
    return Tuple(*out)


def _nt_range(a, b=None):
    """\\operatorname{primes}(x): the primes ≤ x; \\operatorname{primes}(a, b): the primes in [a, b]."""
    lo, hi = (2, a) if b is None else (a, b)
    lo, hi = sympy.sympify(lo), sympy.sympify(hi)
    if not (lo.is_Integer and hi.is_Integer):
        raise SemanticsError("\\operatorname{primes} needs whole-number bounds")
    if hi - lo > 10**7:
        raise SemanticsError("\\operatorname{primes}: a range of more than 10^7 numbers is too large to list")
    return sympy.FiniteSet(*sympy.primerange(int(lo), int(hi) + 1))


class _PRIMES(sympy.Function):
    """\\operatorname{primes}(x) / (a, b): the finite set of primes, once the bounds are whole numbers."""

    @classmethod
    def eval(cls, *args):
        vals = [a.doit() if isinstance(a, Basic) and a.is_number else a for a in args]
        if all(getattr(v, "is_Integer", False) for v in vals):
            return _nt_range(*vals)
        return None

    def _latex(self, printer, exp=None):
        return rf"\operatorname{{primes}}\left({', '.join(printer._print(a) for a in self.args)}\right)"

    def _sympystr(self, printer):
        return f"primes({', '.join(printer._print(a) for a in self.args)})"


def _gen(*exprs):
    """The main variable of a polynomial operation: x if present, else the only letter."""
    free = sorted(set().union(*[getattr(e, "free_symbols", set()) for e in exprs]), key=str)
    for c in ("x", "t", "s", "z", "y"):
        if Symbol(c) in free:
            return Symbol(c)
    if len(free) == 1:
        return free[0]
    raise SemanticsError("which variable? the polynomial has several letters",
                         hint=f"the letters are {', '.join(map(str, free)) or 'none'}; openlab uses x, t, s, z or y first.")


def _polyop(name):
    def f(p, q):
        x = _gen(p, q)
        return {"rem": sympy.rem, "quo": sympy.quo}[name](sympy.expand(p), sympy.expand(q), x)
    return f


def _identify(v):
    """A decimal as a closed form: 3.14159… -> π, 0.6931… -> ln 2 (mpmath's identify)."""
    x = sympy.N(v, 30)
    if not x.is_number:
        raise SemanticsError("identify needs a number")
    guess = mpmath.identify(mpmath.mpf(str(x)), ["pi", "e", "log(2)", "log(3)", "sqrt(2)", "sqrt(3)", "euler"])
    if guess is None:
        raise SemanticsError(f"no simple closed form recognised for {sympy.N(v, 15)}",
                             hint="identify tries rationals and simple combinations of π, e, ln 2, ln 3, √2, √3, γ.")
    return sympy.sympify(guess.replace("euler", "EulerGamma"), locals={"e": sympy.E})


def _eigs(M):
    """Eigenvalues with their multiplicity: [[2, 1], [0, 2]] -> (2, 2)."""
    return Tuple(*[k for k, m in M.eigenvals().items() for _ in range(m)])


def _proj(u, v):
    """The projection of v onto u: ``\\operatorname{proj}_u v``."""
    return (u.dot(v) / u.dot(u)) * u


# Functions a general audience writes by name: resolved when the name is unbound. A package adds more through
# openlab.extensions.FUNCS; a built-in name wins.
KNOWN_FUNCTIONS = ChainMap({
    "Gamma": sympy.gamma, "zeta": sympy.zeta, "erf": sympy.erf, "erfc": sympy.erfc,
    "sgn": sympy.sign, "sign": sympy.sign, "sinc": sympy.sinc, "atan2": sympy.atan2,
    "floor": sympy.floor, "ceil": sympy.ceiling, "abs": sympy.Abs, "exp": sympy.exp,
    "log": sympy.log, "ln": sympy.log, "lg": lambda x: sympy.log(x, 10), "sqrt": sympy.sqrt,
    "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan, "cot": sympy.cot, "sec": sympy.sec,
    "csc": sympy.csc, "sinh": sympy.sinh, "cosh": sympy.cosh, "tanh": sympy.tanh,
    "arcsin": sympy.asin, "arccos": sympy.acos, "arctan": sympy.atan, "arccot": sympy.acot,
    "asin": sympy.asin, "acos": sympy.acos, "atan": sympy.atan,
    "delta": sympy.DiracDelta, "Beta": sympy.beta, "Ei": sympy.Ei, "Si": sympy.Si,
    "Ci": sympy.Ci, "li": sympy.li, "Li": sympy.Li, "W": sympy.LambertW,
    "Heaviside": sympy.Heaviside, "Phi": _std_normal_cdf,
    "gcd": _gcd, "lcm": _lcm, "max": _extreme(sympy.Max), "min": _extreme(sympy.Min),
    "Re": sympy.re, "Im": sympy.im, "arg": sympy.arg, "conj": sympy.conjugate,
    "binom": sympy.binomial, "mod": sympy.Mod,
    # linear algebra (operate on matrices)
    "tr": lambda M: M.trace(), "trace": lambda M: M.trace(), "rank": lambda M: M.rank(),
    "det": lambda M: M.det(), "inv": lambda M: M.inv(), "transpose": lambda M: M.T,
    "adj": lambda M: M.adjugate(), "rref": lambda M: M.rref()[0], "pinv": lambda M: M.pinv(),
    "nullspace": lambda M: sympy.Matrix.hstack(*M.nullspace()) if M.nullspace() else sympy.zeros(M.rows, 0),
    "diag": lambda *d: sympy.diag(*d), "eye": lambda n: sympy.eye(n),
    "eig": lambda M: _eigs(M), "eigvals": lambda M: _eigs(M), "eigenvalues": lambda M: _eigs(M),
    "eigvecs": lambda M: sympy.Matrix.hstack(*[v for _, _, vs in M.eigenvects() for v in vs]),
    "eigenvectors": lambda M: sympy.Matrix.hstack(*[v for _, _, vs in M.eigenvects() for v in vs]),
    "norm": lambda M: M.norm(), "kron": lambda A, B: sympy.kronecker_product(A, B),
    "dim": lambda M: sympy.Tuple(*M.shape), "size": lambda M: Integer(len(M)),
    "proj": _proj,
    "ker": lambda M: sympy.Matrix.hstack(*M.nullspace()) if M.nullspace() else sympy.zeros(M.cols, 0),
    "null": lambda M: sympy.Matrix.hstack(*M.nullspace()) if M.nullspace() else sympy.zeros(M.cols, 0),
    "cond": lambda M: M.condition_number(), "hessian": _hessian, "Hessian": _hessian,
    "corr": _corr, "percentile": _percentile, "quantile": lambda v, q: _percentile(v, 100 * sympy.nsimplify(q)),
    "mode": _mode,
    # number theory (whole-number arguments; they evaluate as soon as the argument is a number)
    "isprime": _NT["isprime"], "primepi": _NT["primepi"], "prime": _NT["prime"],
    "nextprime": _NT["nextprime"], "prevprime": _NT["prevprime"], "totient": _NT["totient"],
    "phi": _NT["totient"], "sigma": _NT["sigma"], "omega": _NT["omega"], "Omega": _NT["Omega"],
    "divisors": lambda n: _nt(n, lambda v: sympy.FiniteSet(*sympy.divisors(v)), "divisors", low=1),
    "factorint": lambda n: _nt(n, lambda v: Tuple(*[Tuple(p_, e_) for p_, e_ in sorted(sympy.factorint(v).items())]),
                               "factorint", low=1),
    "primes": lambda a, b=None: _PRIMES(a) if b is None else _PRIMES(a, b),
    "cf": lambda x, n: Tuple(*[Integer(t) for t in _cf_terms(x, n)]),
    "convergents": _convergents,
    # algebra on demand (the result keeps the form asked for)
    "rem": _polyop("rem"), "quo": _polyop("quo"), "expand": sympy.expand, "factor": sympy.factor,
    "cancel": sympy.cancel, "apart": sympy.apart, "together": sympy.together,
    "numer": lambda e: _NUMER(e), "denom": lambda e: _DENOM(e),
    "identify": _identify, "degree": lambda p: Integer(sympy.degree(p, _gen(p))),
    # data vectors
    "sum": lambda *a: sum(a[0] if len(a) == 1 else a), "len": lambda v: Integer(len(v)), "median": _median,
}, _extensions.FUNCS)
# Subscript-indexed function families: J_0(x) is besselj(0, x).
_FAMILIES = {"J": sympy.besselj, "Y": sympy.bessely, "I": sympy.besseli, "K": sympy.besselk,
             "P": sympy.legendre, "H": sympy.hermite, "L": sympy.laguerre,
             "T": sympy.chebyshevt, "U": sympy.chebyshevu}


# --- time limits -------------------------------------------------------------------------

class _Timeout(BaseException):
    """Raised by the alarm; a BaseException so SymPy's broad ``except Exception`` can't eat it."""

    def __init__(self, token=None):
        super().__init__("time limit")
        if token is None:                       # injected into a worker thread: no arguments are possible
            token = _PENDING.get(threading.get_ident())
        self.token = token


_ALARMS: list = []          # stack of (deadline, token) -- the main thread (SIGALRM)
_PREV_HANDLER: list = [None]
_TSTACKS: dict = {}         # thread ident -> stack of (deadline, token) -- worker threads (timer + injection)
_PENDING: dict = {}         # thread ident -> the token an injected _Timeout carries


def _on_alarm(signum, frame):
    raise _Timeout(_ALARMS[-1][1] if _ALARMS else None)


def _inject(tid: int, exc) -> None:
    """Raise ``exc`` (a class) in thread ``tid`` at its next bytecode; ``None`` clears a pending one."""
    import ctypes
    fn = ctypes.pythonapi.PyThreadState_SetAsyncExc
    fn.argtypes = [ctypes.c_ulong, ctypes.py_object]
    fn(ctypes.c_ulong(tid), ctypes.py_object(exc) if exc is not None else ctypes.py_object())


@contextmanager
def _thread_time_limit(seconds: float, token):
    """The worker-thread form: a timer injects ``_Timeout`` into this thread (signals cannot reach it)."""
    tid = threading.get_ident()
    stack = _TSTACKS.setdefault(tid, [])
    now = time.monotonic()
    deadline = min(now + seconds, stack[-1][0]) if stack else now + seconds
    stack.append((deadline, token))

    def fire():
        if stack and any(t is token for _, t in stack):
            _PENDING[tid] = stack[-1][1]
            _inject(tid, _Timeout)
    timer = threading.Timer(max(deadline - now, 1e-3), fire)
    timer.daemon = True
    timer.start()
    try:
        yield token
    finally:
        timer.cancel()
        stack.pop()
        if not stack:
            _TSTACKS.pop(tid, None)
            _PENDING.pop(tid, None)
            _inject(tid, None)                    # a shot that landed after the body ended is discarded


@contextmanager
def time_limit(seconds: Optional[float]):
    """Interrupt the body after ``seconds``: SIGALRM on the main thread (POSIX), an injected exception on
    any other thread. Nested limits never extend an outer one; the body sees ``_Timeout(token)``."""
    token = object()
    if not (seconds and seconds > 0):
        yield token
        return
    if not (hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread()):
        with _thread_time_limit(float(seconds), token) as t:
            yield t
        return
    now = time.monotonic()
    deadline = now + seconds
    if _ALARMS:
        deadline = min(deadline, _ALARMS[-1][0])
    else:
        _PREV_HANDLER[0] = signal.signal(signal.SIGALRM, _on_alarm)
    _ALARMS.append((deadline, token))
    signal.setitimer(signal.ITIMER_REAL, max(deadline - now, 1e-3))
    try:
        yield token
    finally:
        _ALARMS.pop()
        if _ALARMS:
            signal.setitimer(signal.ITIMER_REAL, max(_ALARMS[-1][0] - time.monotonic(), 1e-3))
        else:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, _PREV_HANDLER[0] or signal.SIG_DFL)


# --- results ---------------------------------------------------------------------------

@dataclass
class Result:
    src: str
    kind: Kind
    expr: Basic                      # the parsed expression
    value: Any                       # what it evaluated to
    numeric: Optional[Any]           # a decimal approximation when that adds information
    interpretation: str
    bound: dict = field(default_factory=dict)
    latex_in: str = ""
    latex_out: str = ""
    notes: list = field(default_factory=list)
    lineno: int = 1
    quiet: bool = False              # display option: hide the interpretation line

    def _repr_html_(self):
        return render.result_html(self)

    def _repr_latex_(self):
        return render.result_latex(self)

    def __repr__(self):
        return render.result_text(self)

    __str__ = __repr__


def L(x) -> str:
    """LaTeX with decorated-symbol display names; huge integers in scientific form."""
    if isinstance(x, Integer) and x.p.bit_length() > 200:
        return sympy.latex(x.evalf(15))
    if isinstance(x, sympy.Rational) and max(x.p.bit_length(), x.q.bit_length()) > 200:
        return sympy.latex(x.evalf(15))
    if x is sympy.zoo:
        return r"\text{undefined (division by zero)}"
    if x is sympy.nan:
        return r"\text{undefined}"
    return sympy.latex(x, symbol_names=DISPLAY_NAMES)


def coerce(v) -> Optional[Basic]:
    """A namespace value as a SymPy object, or ``None`` if it has no meaning in math."""
    if v is None:
        return None
    if isinstance(v, bool):
        return sympy.true if v else sympy.false
    if isinstance(v, MatrixBase):
        return ImmutableMatrix(v)
    if isinstance(v, Basic):
        return v
    if hasattr(v, "to_numpy") and not isinstance(v, (str, bytes)):     # pandas Series / DataFrame
        try:
            return coerce(v.to_numpy())
        except Exception:
            return None
    if isinstance(v, (int, float, complex, Fraction)) or isinstance(v, numbers.Number):
        try:
            return sympy.sympify(v)
        except (sympy.SympifyError, TypeError):
            return None
    try:
        import numpy as np
        if isinstance(v, np.ndarray) and v.ndim in (1, 2) and v.dtype.kind in "biufc":
            return ImmutableMatrix(v.tolist())
        if isinstance(v, np.generic) and v.dtype.kind in "biufc":
            return sympy.sympify(v.item())
    except ImportError:
        pass
    if isinstance(v, (list, tuple, range)) and len(v):
        items = list(v)
        flat = [x for row in items for x in (row if isinstance(row, (list, tuple)) else [row])]
        if not all(isinstance(x, (numbers.Number, Basic)) and not isinstance(x, bool) for x in flat):
            return None
        try:
            return ImmutableMatrix(items)
        except (TypeError, ValueError):
            return None
    if isinstance(v, (set, frozenset)):
        if all(isinstance(x, (numbers.Number, Basic)) for x in v):
            return sympy.FiniteSet(*v)
    return None


def _is_vector(m) -> bool:
    return isinstance(m, MatrixBase) and (m.cols == 1 or m.rows == 1)


def _as_column(m):
    return m if m.cols == 1 else m.T


def _vec(a):
    """A parenthesised tuple of numbers used as a vector."""
    if isinstance(a, Tuple):
        return ImmutableMatrix(list(a))
    return a


_LINALG = {"eig", "eigvals", "eigenvalues", "eigvecs", "eigenvectors", "tr", "trace", "rank", "det",
           "inv", "adj", "rref", "pinv", "nullspace", "ker", "cond", "norm", "dim", "transpose"}
_FORM_FUNCS = {"expand", "factor", "cancel", "apart", "together", "rem", "quo", "numer", "denom"}
_MATRIX_FUNCS = (sympy.sin, sympy.cos, sympy.tan, sympy.sinh, sympy.cosh, sympy.tanh, sympy.log)


def _ordered_replace(expr, pred, fn):
    """``expr.replace(pred, fn)`` that rebuilds products and sums without re-sorting their
    factors: ``C (sI - A)^{-1} B`` must stay C·(…)·B when C, A, B are matrices."""
    if not isinstance(expr, Basic) or isinstance(expr, MatrixBase):
        return expr
    if expr.args:
        args = [_ordered_replace(a, pred, fn) for a in expr.args]
        if not all(a is b for a, b in zip(args, expr.args)):
            if isinstance(expr, (Mul, sympy.Add)):
                expr = expr.func(*args, evaluate=False)
            else:
                expr = expr.func(*args)
    return fn(expr) if pred(expr) else expr


def _letter_sets(a, b) -> bool:
    """Two finite sets written with letters (``\\{a, b\\}``): compared element by element, since
    SymPy cannot know that the letters differ."""
    if not (isinstance(a, sympy.FiniteSet) and isinstance(b, sympy.FiniteSet)):
        return False
    elems = list(a.args) + list(b.args)
    return any(isinstance(x, Symbol) for x in elems) and all(
        isinstance(x, Symbol) or getattr(x, "is_number", False) for x in elems)


def _is_rv(x) -> bool:
    from sympy.stats.rv import RandomSymbol
    return isinstance(x, Basic) and x.has(RandomSymbol)


def _evalf(v, p):
    if isinstance(v, (list, tuple)):
        return type(v)(_evalf(x, p) for x in v)
    if isinstance(v, dict):
        return {k: _evalf(x, p) for k, x in v.items()}
    if isinstance(v, Tuple):
        return Tuple(*[_evalf(x, p) for x in v])
    if isinstance(v, (BooleanAtom, sympy.Set, Lambda)) or not isinstance(v, Basic):
        return v
    try:
        return v.evalf(p)
    except Exception:
        return v


def _shape_error(exc, a=None, b=None) -> SemanticsError:
    shapes = ""
    if isinstance(a, MatrixBase) and isinstance(b, MatrixBase):
        shapes = f" ({a.rows}×{a.cols} and {b.rows}×{b.cols})"
    return SemanticsError(f"the matrix shapes do not fit{shapes}",
                          hint="a product A B needs A's columns to equal B's rows; a sum needs equal shapes.")


class Executor:
    def __init__(self, ns: dict, *, numeric: bool = False, precision: int = 15,
                 complex_i: bool = True, solve_for: Optional[str] = None,
                 simplify: bool = True, timeout: float = 30.0, cell_id: Optional[str] = None,
                 strict: bool = False):
        if precision < 1:
            raise ParseError(f"-p needs at least 1 digit (got {precision})")
        self.ns = ns
        self.numeric = numeric
        self.precision = precision
        self.complex_i = complex_i
        self.solve_for = solve_for
        self.simplify = simplify
        self.timeout = timeout
        t_ = timeout or 30.0
        self.budget = max(1.0, min(10.0, t_ / 2)) if t_ <= 30 else 0.9 * t_   # -t 600: 540 s for one step
        self.solved: dict = {}           # unknown -> (solutions, lineno), this cell only
        self.last_solve = None           # (unknown, solutions, lineno) of the latest solve
        self._indep: Optional[Symbol] = None     # independent variable for y' in this statement
        self._dep: set = set()                   # dependent (differentiated) names
        self.cell_assigned: set = set()          # names given a formula earlier in this cell
        self.cell_lines: list = []               # the cell, for recognising an edited re-run
        self._qualifier = ""                     # "for positive values" on a qualified identity
        self.cell_id = cell_id                   # the notebook cell being run (from the magic)
        self.strict = strict                     # a statement that is only represented is an error
        self.keep_sums = False                   # the prover: leave Sum/Product unevaluated
        self._keep_form = False                  # expand/factor/apart asked for a form: keep it
        self.underdetermined_ok = False          # solve a matrix equation with spare unknowns anyway
        self._convergent: set = set()            # infinite sums known to converge (summed numerically)
        self._maybe: set = set()                 # undecided sums: summed only when the methods agree
        self._event_notes: list = []             # how a probability was derived
        self._columns: dict = {}                 # frame.column -> its values, read once per cell

    # --- namespace ---------------------------------------------------------------------
    def _lookup(self, name: str):
        if name in IGNORED or name.startswith("_"):
            return None
        if name not in self.ns:
            v = self._frame_column(name) if "." in name else None
            return self._fit_ref(name) if v is None else v
        if name in getattr(self, "_shadow", ()):     # a claim's bound variable hides a notebook name
            return None
        return self.ns[name]

    def _frame_column(self, name: str):
        r"""``frame.column`` as the column's values, the way a test already reads ``a="book.r"``.

        Without this the only dotted name a statement could see was one a ``%%py`` cell had bound by hand:
        ``\bar{\mathrm{book.r}}`` bound the *symbol* ``book.rbar`` and said nothing, which reads as a mean
        and is not one. Nulls are dropped, as ``data_bind`` drops them -- a mean over a missing value is not
        a mean. The values are cached for the cell because every resolution pass looks its names up again,
        and a few hundred thousand rows should cross out of polars once."""
        cache = self._columns
        if name not in cache:                # the misses are cached too: most dotted names are not columns
            frames = self.ns.get("frames")
            fr, _, col = name.partition(".")
            df = frames.get(fr) if isinstance(frames, dict) else None
            known = df is not None and hasattr(df, "get_column") and col in getattr(df, "columns", ())
            cache[name] = [v for v in df.get_column(col).to_list() if v is not None] if known else None
        return cache[name]

    def _fit_ref(self, name: str):
        r"""``\hat\beta_{m,x}``, ``\mathrm{se}_{m,x}``, ``p_{m,x}``: the estimate, se or p of term ``x`` in the
        model fit named ``m`` (in ``_openlab_fits``), read from the fit itself -- computed, as a data mean is, not
        bound."""
        m = _FIT_REF.fullmatch(name)
        fits = self.ns.get("_openlab_fits")
        if not m or not isinstance(fits, dict) or m.group(2) not in fits or (m.group(1) == "beta") != bool(m.group(4)):
            return None
        terms = [t["term"] for t in fits[m.group(2)]["terms"]]
        if not terms:
            raise SemanticsError(f"a {fits[m.group(2)].get('model', 'model')} has no coefficients: model "
                                 f"{m.group(2)} has importances, not estimates", hint="score it on a holdout")
        hit = next((t for t in fits[m.group(2)]["terms"] if _term_key(t["term"]) == _term_key(m.group(3))), None)
        if hit is None:
            raise SemanticsError(f"model {m.group(2)} has no term `{m.group(3)}`; its terms: {', '.join(terms)}",
                                 hint=fit_refer(m.group(2), terms))
        return hit["estimate" if m.group(1) == "beta" else m.group(1)]

    def _reg(self, key: str) -> dict:
        reg = self.ns.get(key)
        if not isinstance(reg, dict):
            reg = self.ns[key] = {}
        return reg

    def _is_function_value(self, s) -> bool:
        return _callable_value(self._lookup(getattr(s, "name", "")))

    @staticmethod
    def bound_variables(expr: Basic) -> set:
        out: set = set()
        for node in expr.atoms(Derivative):
            out.update(node.variables)
        for node in expr.atoms(Integral):      # definite ones are scoped instead (see _rebuild)
            out.update(lim[0] for lim in node.limits if len(lim) == 1)
        for node in expr.atoms(Lambda):
            out.update(node.variables)
        for node in expr.atoms(sympy.ConditionSet):
            out.add(node.sym)
        for node in expr.atoms(Optim):             # only the index: in \max_{r \in Q}, Q is looked up
            v_ = _optim_var(node.args[2])
            if v_ is not None:
                out.add(v_)
            elif isinstance(node.args[2], Symbol):
                out.add(node.args[2])
        for node in expr.atoms(Quant):
            out.update(s for s in node.args[1].free_symbols if isinstance(s, Symbol))
        return out

    def _implicit_vars(self, expr: Basic) -> set:
        """Variables of evaluation bars and transforms: never substituted inside them."""
        out: set = set()
        for node in expr.atoms(EvalSubs, EvalBetween, Transform):
            out |= self._node_vars(node)
        return out

    def _bar_var(self, node) -> Optional[Symbol]:
        """The variable an evaluation bar with no ``x =`` acts on: the one unbound free letter,
        else x, else t (``\\left. t^2 x \\right|_0^2`` with x bound is over t)."""
        body = node.args[0]
        free = {s for s in body.free_symbols if isinstance(s, Symbol)}
        unbound = {s for s in free if self._lookup(s.name) is None}
        for pool in (unbound, free):
            if len(pool) == 1:
                return next(iter(pool))
            for c in ("x", "t"):
                if Symbol(c) in pool:
                    return Symbol(c)
        return None

    def _node_vars(self, node) -> set:
        """The names a scoping node binds inside itself."""
        if isinstance(node, (EvalSubs, EvalBetween)):
            var = node.args[1] if isinstance(node, EvalSubs) else node.args[3]
            if isinstance(var, Symbol):
                return {var}
            if isinstance(var, Tuple):
                return {v for v in var if isinstance(v, Symbol)}
            point = node.args[2] if isinstance(node, EvalSubs) else None
            if isinstance(point, Tuple):
                return {Symbol(c) for c in "xyz"[:len(point)]}
            v = self._bar_var(node)
            return {v} if v is not None else set()
        if isinstance(node, Transform):
            out = {Symbol("t"), Symbol("s"), Symbol("omega"), Symbol("x")} & node.free_symbols
            if len(node.args) > 3 and isinstance(node.args[3], Symbol):
                out.add(node.args[3])
            return out
        if isinstance(node, Derivative):
            return set(node.variables)
        if isinstance(node, Integral):
            return {lim[0] for lim in node.limits if len(lim) == 1}
        if isinstance(node, Optim):
            v_ = _optim_var(node.args[2])               # \max_{r \in Q}: r is bound, Q is a name to look up
            return {v_} if v_ is not None else set()
        if isinstance(node, Quant):
            return {s for s in node.args[1].free_symbols if isinstance(s, Symbol)}
        if isinstance(node, sympy.ConditionSet):
            return {node.sym}
        return set()

    _SCOPES = (EvalSubs, EvalBetween, Transform, Derivative, Optim, Quant, sympy.ConditionSet)

    def _outside_free(self, expr: Basic) -> set:
        """Free names outside every scoping node (``x`` in ``\\frac{d}{dx} x^2 + x``)."""
        def is_scope(n):
            return isinstance(n, self._SCOPES) or (isinstance(n, Integral) and any(len(l) == 1 for l in n.limits))
        rep = {}
        for n in sympy.preorder_traversal(expr):
            if is_scope(n):
                rep[n] = sympy.Dummy()
        masked = expr.xreplace(rep) if rep else expr
        return {s for s in masked.free_symbols if isinstance(s, Symbol)}

    @staticmethod
    def _heads(expr: Basic) -> set:
        heads = {a.args[0] for a in expr.atoms(Apply) if isinstance(a.args[0], Symbol)}
        heads |= {a.args[0].expr for a in expr.atoms(Apply)          # \dot x(3): x is the function
                  if isinstance(a.args[0], Derivative) and isinstance(a.args[0].expr, Symbol)}
        for c in expr.atoms(Compose):
            heads |= {a for a in c.args if isinstance(a, Symbol)}
        heads |= {ib.label for ib in expr.atoms(sympy.IndexedBase)}
        return heads

    @staticmethod
    def _differentiated(expr: Basic) -> set:
        """Names that are differentiated in the statement (the unknown functions of an ODE)."""
        out = {d.expr for d in expr.atoms(Derivative) if isinstance(d.expr, Symbol)}
        out |= {p.args[0] for p in expr.atoms(Prime) if isinstance(p.args[0], Symbol)}
        for a in expr.atoms(Apply):
            if a.args and isinstance(a.args[-1], CallMod) and str(a.args[-1].args[0]) in ("prime", "nderiv") \
                    and isinstance(a.args[0], Symbol):
                out.add(a.args[0])
        return out

    @staticmethod
    def _context_var(expr: Basic) -> Optional[Symbol]:
        for node in expr.atoms(Derivative):
            return node.variables[0]
        for node in expr.atoms(Integral):
            return node.variables[0]
        for node in expr.atoms(Limit):
            return node.args[1]
        return None

    # --- decorated names -----------------------------------------------------------------
    def _deco_name(self, node: Deco) -> str:
        base, cmd = node.args
        suffix = {"bar": "bar", "overline": "bar", "hat": "hat", "widehat": "hat",
                  "tilde": "tilde", "widetilde": "tilde", "underline": "under",
                  "breve": "breve", "check": "check", "acute": "acute", "grave": "grave",
                  "mathring": "ring"}.get(str(cmd), str(cmd))
        m = re.fullmatch(r"([A-Za-z]+)_(.+)", base.name)
        if m and self._lookup(m.group(1) + suffix) is not None:     # \hat y_i with yhat bound: yhat_i
            return f"{m.group(1)}{suffix}_{m.group(2)}"
        return f"{base.name}{suffix}"

    def _data_mean(self, node) -> bool:
        """Is this ``\\bar{x}`` with ``x`` bound to data (a list, an array, a column)? Then it is the mean of
        that data -- a statistic openlab computes -- and never a free name: it cannot be assigned, and a
        separately bound ``xbar`` cannot stand in for it. Otherwise a false mean could be bound and 'verified'."""
        if not isinstance(node, Deco):
            return False
        base, cmd = node.args
        if not (isinstance(base, Symbol) and str(cmd) in ("bar", "overline")):
            return False
        return isinstance(coerce(self._lookup(base.name)), MatrixBase)

    def _resolve_deco(self, node: Deco, notes: list):
        base, cmd = node.args
        if isinstance(base, sympy.Indexed):              # \hat y_i inside a sum: entry i of yhat
            nm = self._deco_name(Deco(Symbol(str(base.base.label)), cmd))
            if self._lookup(nm) is not None:
                return None, sympy.IndexedBase(nm)[base.indices]
            return Symbol(f"{nm}_{'_'.join(map(str, base.indices))}"), None
        name = self._deco_name(node)
        sym = Symbol(name)
        DISPLAY_NAMES[sym] = rf"\{cmd}{{{L(base)}}}"
        if self._data_mean(node):                        # p is data: \bar{p} is its mean, whatever `pbar` holds
            bv = coerce(self._lookup(base.name))
            notes.append(rf"\bar{{{base}}} read as the mean of the {len(bv)} values of {base}"
                         + (f" (a separate name `{name}` is bound too; it is not the mean and is not used)"
                            if self._lookup(name) is not None else ""))
            return None, sum(bv) / len(bv)
        if self._lookup(name) is not None:
            return sym, None
        if isinstance(base, Symbol) and str(cmd) in ("bar", "overline") and \
                self._reg("_openlab_assume").get(base.name, {}).get("complex"):
            notes.append(rf"\bar{{{base}}} read as the complex conjugate of {base}")
            return None, sympy.conjugate(Symbol(base.name, complex=True))
        bv = coerce(self._lookup(base.name)) if isinstance(base, Symbol) else None
        if bv is not None and str(cmd) in ("bar", "overline"):
            if isinstance(bv, MatrixBase):
                notes.append(rf"\bar{{{base}}} read as the mean of the {len(bv)} values of {base}")
                return None, sum(bv) / len(bv)
            if bv.is_number:
                notes.append(rf"\bar{{{base}}} read as the complex conjugate of {base}")
                return None, sympy.conjugate(bv)
        return sym, None

    # --- resolution ------------------------------------------------------------------------
    def resolve(self, expr: Basic, exclude=frozenset(), notes: Optional[list] = None):
        notes = [] if notes is None else notes
        exclude = set(exclude)
        if not isinstance(expr, Basic):
            return expr, {}
        expr = self._rewrite_events(expr, notes)
        expr = self._rewrite_dot_names(expr, notes)
        expr = self._rewrite_rates(expr, notes)
        if expr.has(Deco):
            rep = {}
            for node in expr.atoms(Deco):
                sym, val = self._resolve_deco(node, notes)
                rep[node] = val if (val is not None and sym is None) else sym
            expr = expr.xreplace(rep)
            if expr.has(Sum):                  # \sum_i (y_i - \hat y_i)^2: yhat_i is an entry of yhat
                from .transform import ToSympy
                expr = expr.replace(lambda n: isinstance(n, Sum) and len(n.limits) == 1,
                                    lambda n: Sum(ToSympy._index_body(n.function, n.limits[0][0]), *n.limits))
        dependent = self._differentiated(expr) | self._dep
        expr = self._pre_rewrite(expr, exclude, notes)
        seen: set = set()
        shown: dict = {}
        for _ in range(300):
            heads = self._heads(expr)
            # a name bound by an operator is left alone inside it, and substituted outside it
            bound = (self.bound_variables(expr) | self._implicit_vars(expr)) - self._outside_free(expr)
            subs = self._value_subs(expr, heads, bound, exclude, notes, seen)
            changed = False
            if subs:
                new = self._subst(expr, subs, notes)
                seen.update(k for k in subs if isinstance(k, Symbol))
                shown.update(subs)
                changed = new != expr
                expr = new
            if expr.has(Apply) or self._late_calls(expr):
                new = self._apply_all(expr, notes)
                if new != expr:
                    expr, changed = new, True
            if not changed:
                break
        else:
            raise SemanticsError("circular definitions: some names are defined in terms of each other",
                                 hint="rebind one of them to a value, e.g. `a := 3`.")
        expr = self._policies(expr, dependent, notes, shown, exclude)
        expr = self._assume(expr, exclude)
        return expr, {k: v for k, v in shown.items() if isinstance(k, (Symbol, sympy.Indexed))}

    def _value_subs(self, expr, heads, bound, exclude, notes, seen) -> dict:
        subs: dict = {}
        ctx = self._context_var(expr)
        for s in sorted(expr.free_symbols, key=lambda s: getattr(s, "name", "")):
            if not isinstance(s, Symbol) or s in heads or s in bound or s.name in exclude:
                continue
            raw = self._lookup(s.name)
            if raw is None:
                eye = self._identity_for(s, expr, notes)
                if eye is not None:
                    subs[s] = eye
                    continue
                el = self._element(s.name, notes)
                if el is not None:
                    subs[s] = el
                continue
            v = coerce(raw)
            if v is None:
                if not callable(raw):
                    notes.append(f"{s.name} is bound to a {type(raw).__name__}, not a number; "
                                 f"it is treated as a symbol here")
                continue
            if isinstance(v, Lambda) and expr == s:
                subs[s] = v                       # `f` on its own: the function itself
                continue
            if isinstance(v, Lambda):
                body = v.expr
                if len(v.variables) == 1 and ctx is not None and v.variables[0] != ctx:
                    body = body.xreplace({v.variables[0]: ctx})
                notes.append(f"{s.name} used as an expression: read as {interp.pyform(body)}")
                v = body
            if isinstance(v, Basic) and s in v.free_symbols:
                continue
            subs[s] = v
        for ix in expr.atoms(sympy.Indexed):
            v = self._indexed_value(ix)
            if v is not None:
                subs[ix] = v
        return subs

    def _identity_for(self, s, expr, notes):
        """An unbound ``I`` beside square matrices is the identity of their size; ``I_n`` (n ≥ 2)
        is the n×n identity when matrices are around or it stands alone (``I_0`` stays a name).
        Substituted together with the matrices, so never broadcast as a scalar."""
        mats = list(expr.atoms(MatrixBase))
        for other in expr.free_symbols:
            v = coerce(self._lookup(getattr(other, "name", "")))
            if isinstance(v, MatrixBase):
                mats.append(v)
        m = re.fullmatch(r"I_(\d+)", s.name)
        if m and self._lookup("I") is not None:
            return None                  # I_n = \int … defined a sequence: I_3 is its term
        if m:
            k = int(m.group(1))
            if any(s in M.free_symbols for M in expr.atoms(MatrixBase)):
                return None              # [I_1; I_2] = …: currents, not identities
            in_la = any(isinstance(a, Apply) and getattr(a.args[0], "name", "") in _LINALG and s in a.free_symbols
                        for a in expr.atoms(Apply)) or any(s in d.free_symbols for d in expr.atoms(Det))
            return ImmutableMatrix(sympy.eye(k)) if k >= 2 and (mats or expr == s or in_la) else None
        if s.name != "I" or not mats:
            return None
        sizes = {M.rows for M in mats if M.rows == M.cols}
        if len(sizes) != 1:
            return None          # sized later, from the term it is added to or multiplied with
        n = sizes.pop()
        notes.append(f"I read as the {n}×{n} identity")
        return ImmutableMatrix(sympy.eye(n))

    def _element(self, name: str, notes: list):
        """``x_1`` / ``A_{12}`` / ``A_{1,2}`` / ``a_5`` from a bound vector, matrix or sequence."""
        m = re.fullmatch(r"(.+?)_(?:\{(.+)\}|(.+))", name)
        if not m:
            return None
        base, idx = m.group(1), (m.group(2) or m.group(3))
        rec = self._reg("_openlab_recur").get(base)
        if rec is not None and re.fullmatch(r"\d+", idx):
            return self._iterate(base, rec, int(idx), notes)
        bv = _vec(coerce(self._lookup(base)))
        if bv is None:
            return None
        parts = [p.strip() for p in idx.split(",")] if "," in idx else [idx]

        def num(p):
            if re.fullmatch(r"-?\d+", p):
                return int(p)
            val = coerce(self._lookup(p))
            if val is not None and getattr(val, "is_Integer", False):
                return int(val)
            if not re.search(r"[A-Za-z]", p) or not re.fullmatch(r"[\w+\-*\s{}]+", p):
                return None
            try:                                  # x_{k+1}, x_{2k} with k bound
                e = parse_expr(p)[0]
            except OpenlabError:
                return None
            vals = {q: coerce(self._lookup(q.name)) for q in e.free_symbols if isinstance(q, Symbol)}
            if not vals or any(v is None for v in vals.values()):
                return None
            e = e.xreplace(vals)
            return int(e) if getattr(e, "is_Integer", False) else None
        rec = self._reg("_openlab_recur").get(base)
        if rec is not None and re.fullmatch(r"\d+", idx):
            return self._iterate(base, rec, int(idx), notes)
        if isinstance(bv, Lambda) and len(bv.variables) == 1 and base in self._reg("_openlab_seqs"):
            k = num(idx)
            if k is None:
                try:
                    kexpr = parse_expr(idx)[0]
                except OpenlabError:
                    return None
                notes.append(f"{name} is the term at {idx} of the sequence {base}")
                return bv(kexpr)
            notes.append(f"{name} is term {k} of the sequence {base}")
            return bv(k)
        if not isinstance(bv, MatrixBase):
            return None
        if _is_vector(bv) and len(parts) == 1:
            k = num(parts[0])
            if k is None:
                return None
            if not 1 <= k <= len(bv):
                raise SemanticsError(f"{name}: {base} has entries 1..{len(bv)} (openlab counts from 1)")
            notes.append(f"{name} is entry {k} of {base} (counting from 1)")
            return bv[k - 1]
        if len(parts) == 1 and re.fullmatch(r"\d\d", parts[0]):
            parts = list(parts[0])
        if len(parts) == 2:
            i, j = num(parts[0]), num(parts[1])
            if i is None or j is None:
                return None
            if not (1 <= i <= bv.rows and 1 <= j <= bv.cols):
                raise SemanticsError(f"{name}: {base} is {bv.rows}×{bv.cols} (openlab counts from 1)")
            notes.append(f"{name} is row {i}, column {j} of {base}")
            return bv[i - 1, j - 1]
        return None

    def _iterate(self, base, rec, k, notes):
        n, shift, rhs, vals = rec
        a = Function(base)
        vals = dict(vals)
        if k > 100000:
            raise SemanticsError(f"{base}_{k} would take {k} steps to iterate", hint="ask for an earlier term.")
        m = max(vals) + 1
        while k not in vals and m <= k:
            nn = m - shift
            expr = rhs.xreplace({n: nn})
            rep = {f: vals.get(int(f.args[0])) for f in expr.atoms(AppliedUndef)
                   if f.func == a and f.args[0].is_Integer}
            if any(v is None for v in rep.values()):
                raise SemanticsError(f"{base}_{k} needs earlier initial values")
            v = expr.xreplace(rep)
            vals[m] = v.evalf(self.precision) if v.has(Float) else v
            m += 1
        if k not in vals:
            raise SemanticsError(f"{base}_{k} comes before the initial values")
        notes.append(f"{base}_{k} computed by iterating the recurrence")
        return vals[k]

    def _indexed_value(self, ix: sympy.Indexed):
        seq = self._lookup(str(ix.base.label))
        if isinstance(seq, Lambda) and len(ix.indices) == 1 and len(seq.variables) == 1 \
                and str(ix.base.label) in self._reg("_openlab_seqs"):
            return seq(ix.indices[0])            # \sum_{n=1}^\infty a_n of a defined sequence
        if not all(getattr(k, "is_Integer", False) for k in ix.indices):
            return None
        name = f"{ix.base.label}_{'_'.join(str(k) for k in ix.indices)}"
        v = coerce(self._lookup(name))
        if v is not None:
            return v
        base = _vec(coerce(self._lookup(str(ix.base.label))))
        if isinstance(base, MatrixBase) and len(ix.indices) == 1:
            k = int(ix.indices[0])
            if 1 <= k <= len(base):
                return base[k - 1]
            raise SemanticsError(f"{ix.base.label}[{k}] is out of range: {ix.base.label} has {len(base)} entries (counting from 1)")
        if isinstance(base, Lambda) and len(ix.indices) == 1 and str(ix.base.label) in self._reg("_openlab_seqs"):
            return base(ix.indices[0])
        return None

    # --- substitution that understands matrices ------------------------------------------------
    def _subst(self, expr, subs, notes):
        return self._rebuild(expr, subs, notes)

    def _rebuild(self, node, subs, notes):
        """Substitute bottom-up, understanding matrices and scopes: the variable of a definite
        integral, sum, product, limit or function is never replaced inside it (``x = 3`` then
        ``\\int_0^1 x\\,dx + x`` is 1/2 + 3)."""
        if not subs:
            return node
        if node in subs:
            return subs[node]
        if isinstance(node, (Integral, Sum, Product)) and node.limits:
            vars_ = [lim[0] for lim in node.limits]
            f = self._rebuild(node.function, {k: v for k, v in subs.items() if k not in vars_}, notes)
            lims = []
            for i, lim in enumerate(node.limits):
                # the bounds of limit i see the variables of the limits outside it, not its own:
                # \int_0^x x\,dx with x = 3 runs to 3
                outer = {k: v for k, v in subs.items() if k not in vars_[i + 1:]}
                lims.append(Tuple(lim[0], *[self._rebuild(b, outer, notes) for b in lim[1:]]))
            if f is node.function and all(a == b for a, b in zip(lims, node.limits)):
                return node
            return node.func(f, *lims)
        if isinstance(node, self._SCOPES):
            inner = self._node_vars(node)
            if inner & set(subs):
                keep = {k: v for k, v in subs.items() if k not in inner}
                if isinstance(node, Derivative):
                    return Derivative(self._rebuild(node.expr, keep, notes), *node.variable_count)
                if isinstance(node, (EvalSubs, EvalBetween)):
                    a = list(node.args)
                    a[0] = self._rebuild(a[0], keep, notes)
                    rng = (2,) if isinstance(node, EvalSubs) else (1, 2)
                    for j in rng:
                        a[j] = self._rebuild(a[j], subs, notes)
                    return node.func(*a)
                args = [self._rebuild(a, keep, notes) for a in node.args]
                return node.func(*args)
        if isinstance(node, Limit):
            f, var, pt, d = node.args
            return Limit(self._rebuild(f, {k: v for k, v in subs.items() if k != var}, notes),
                         var, self._rebuild(pt, subs, notes), str(d))
        if isinstance(node, Lambda):
            inner = {k: v for k, v in subs.items() if k not in node.variables}
            return Lambda(node.variables, self._rebuild(node.expr, inner, notes))
        if isinstance(node, MatrixBase):
            if not node.free_symbols:
                return node
            entries = [self._rebuild(e, subs, notes) for e in node]
            if any(isinstance(e, MatrixBase) for e in entries):
                return self._blocks(node.rows, node.cols, entries, notes)
            return ImmutableMatrix(node.rows, node.cols, entries)
        if not isinstance(node, Basic) or not node.args:
            return node
        args = [self._rebuild(a, subs, notes) for a in node.args]
        if all(a is b for a, b in zip(args, node.args)):
            return node
        return self._combine(node, args, notes)

    @staticmethod
    def _blocks(rows, cols, entries, notes):
        """A matrix written with matrix entries is a block matrix: [B  AB], [[A, 0], [0, A]]."""
        grid = [entries[r * cols:(r + 1) * cols] for r in range(rows)]
        heights = [max((e.rows for e in row if isinstance(e, MatrixBase)), default=None) for row in grid]
        widths = [max((grid[r][c].cols for r in range(rows) if isinstance(grid[r][c], MatrixBase)), default=None)
                  for c in range(cols)]
        if None in heights or None in widths:
            raise SemanticsError("a block matrix needs a matrix in every block row and column",
                                 hint="write zero blocks explicitly, e.g. `0` beside a matrix in the same row and column.")
        out_rows = []
        for r, row in enumerate(grid):
            blocks = []
            for c, e in enumerate(row):
                h, w = heights[r], widths[c]
                if isinstance(e, MatrixBase):
                    if e.shape != (h, w):
                        raise SemanticsError(f"block ({r + 1}, {c + 1}) is {e.rows}×{e.cols}, but its row and "
                                             f"column need {h}×{w}")
                    blocks.append(sympy.Matrix(e))
                elif e == 0:
                    blocks.append(sympy.zeros(h, w))
                elif h == w:
                    blocks.append(e * sympy.eye(h))            # a scalar block on the diagonal: s·I
                else:
                    raise SemanticsError(f"block ({r + 1}, {c + 1}) is the number {e}, but the block is {h}×{w}")
            out_rows.append(sympy.Matrix.hstack(*blocks))
        notes.append("a matrix of matrices read as a block matrix")
        return ImmutableMatrix(sympy.Matrix.vstack(*out_rows))

    def _combine(self, node, args, notes):
        mats = [a for a in args if isinstance(a, MatrixBase)]
        if not mats:
            if isinstance(node, (Mul, sympy.Add)):
                return node.func(*args, evaluate=False)
            try:
                return node.func(*args)
            except TypeError as exc:                          # `Var(y) < 1.02` with y now a data vector: the
                if isinstance(node, Relational) and "comparison" in str(exc):   # statistic is evaluated later
                    return node.func(*args, evaluate=False)
                raise
        if isinstance(node, Relational) and all(m.shape == (1, 1) for m in mats):   # a one-row column vs a number
            notes.append("a 1×1 value compared as a number")
            return self._combine(node, [a[0, 0] if isinstance(a, MatrixBase) else a for a in args], notes)
        pending = [a for a in args if not isinstance(a, MatrixBase) and isinstance(a, Basic)
                   and (isinstance(a, (Apply, Marker)) or a.has(Apply, EvalSubs, EvalBetween, Nabla, Transposed, MPow))]
        if pending and (node.is_Mul or node.is_Add):    # pinv(X) y: keep the order, combine once it is a matrix
            marker = Star if node.is_Mul else Plus
            out = args[0]
            for a in args[1:]:
                out = marker(out, a)
            return out
        try:
            if isinstance(node, Pow):
                return self._mat_pow(args[0], args[1], notes)
            if node.is_Add:
                return self._mat_add(args, notes)
            if node.is_Mul:
                I_ = Symbol("I")
                if I_ in args and self._lookup("I") is None:          # A I with A 2×3: I is the identity
                    args = [a for a in args if a != I_]
                    notes.append("I read as the identity of the size that fits")
                out = args[0]
                for a in args[1:]:
                    out = self._mat_mul(out, a)
                return out
            if isinstance(node, sympy.Function) and not isinstance(node, (Marker, AppliedUndef, CallMod)) \
                    and len(args) == 1:
                M = args[0]
                if node.func is sympy.exp and M.rows == M.cols:
                    notes.append("e^A read as the matrix exponential")
                    return M.exp()
                if M.rows == M.cols and M.rows > 1 and node.func in _MATRIX_FUNCS:
                    return self._matrix_function(M, node.func, notes)
                notes.append(f"{node.func.__name__} applied to each entry")
                return M.applyfunc(node.func)
            return node.func(*args)
        except (ShapeError, NonSquareMatrixError) as exc:
            raise _shape_error(exc, *mats[:2]) from None

    @staticmethod
    def _mat_mul(a, b):
        if isinstance(a, MatrixBase) and isinstance(b, MatrixBase):
            if a.cols != b.rows:
                raise _shape_error(None, a, b)
        return a * b

    def _mat_add(self, args, notes):
        mats = [a for a in args if isinstance(a, MatrixBase)]
        scal = [a for a in args if not isinstance(a, MatrixBase)]
        shape = mats[0].shape
        if any(m.shape != shape for m in mats):
            raise _shape_error(None, mats[0], next(m for m in mats if m.shape != shape))
        total = functools.reduce(lambda x, y: x + y, mats)
        I_ = Symbol("I")
        if scal and shape[0] == shape[1] and self._lookup("I") is None and \
                any(isinstance(a, Basic) and I_ in a.free_symbols for a in scal):    # A^T A - I
            s_ = sympy.Add(*scal)
            c = s_.coeff(I_)
            if not c.has(I_):
                total = total + c * sympy.eye(shape[0])
                notes.append(f"I read as the {shape[0]}×{shape[0]} identity")
                rest = sympy.expand(s_ - c * I_)
                scal = [rest] if rest != 0 else []
        if scal:
            s = sympy.Add(*scal)
            notes.append(f"{interp.pyform(s)} added to every entry")
            total = total + s * sympy.ones(*shape)
        return total

    def _mat_pow(self, b, e, notes):
        T, H = Symbol("T"), Symbol("H")
        if isinstance(b, MatrixBase):
            if e == T:
                return b.T
            if e == H:
                return b.H
            if b.rows == b.cols:
                if getattr(e, "is_Integer", False):
                    try:
                        return b ** int(e)
                    except NonInvertibleMatrixError:
                        raise SemanticsError("the matrix is singular (determinant 0): it has no inverse") from None
                try:
                    return b ** e
                except Exception:
                    raise SemanticsError(f"cannot raise a square matrix to the power {interp.pyform(e)}",
                                         hint="integer powers, ^T, ^{-1} and e^{A} are supported.") from None
            if not _is_vector(b) or (getattr(e, "is_negative", False)):
                raise SemanticsError(f"a {b.rows}×{b.cols} matrix has no powers or inverse (it is not square)",
                                     hint=r"use \operatorname{pinv}(A) for a pseudo-inverse, or A^T A.")
            notes.append(f"^{interp.pyform(e)} taken entry by entry")
            return b.applyfunc(lambda z: z ** e)
        if isinstance(e, MatrixBase):
            if (b == sympy.E or (b == Symbol("e") and self._lookup("e") is None)) and e.rows == e.cols:
                notes.append("e^A read as the matrix exponential")
                return e.exp()
            if e.rows == e.cols and e.rows > 1 and getattr(b, "is_number", False) and b.is_positive:
                notes.append(f"{interp.pyform(b)}^A read as the matrix function e^(A ln {interp.pyform(b)})")
                return ImmutableMatrix((e * sympy.log(b)).exp())
            notes.append("power taken entry by entry")
            return e.applyfunc(lambda z: b ** z)
        return Pow(b, e)

    def _matrix_function(self, M, fn, notes):
        """sin A, cos A, ln A, … of a square matrix: the matrix function (its power series), as
        e^A already is -- not the function of each entry."""
        x = sympy.Dummy("x")
        try:
            out = ImmutableMatrix(M.analytic_func(fn(x), x))
        except Exception as exc:
            raise SemanticsError(f"{fn.__name__} of this matrix could not be computed ({type(exc).__name__})",
                                 hint=f"for {fn.__name__} of each entry, write the entries out.") from None
        notes.append(f"{fn.__name__}(A) is the matrix function (a power series in A), not "
                     f"{fn.__name__} of each entry")
        return out.applyfunc(sympy.simplify)

    # --- rewrites that must see the namespace before substitution ------------------------------------
    def _pre_rewrite(self, expr, exclude, notes):
        expr = self._rewrite_events(expr, notes)
        expr = self._rewrite_parametric(expr, notes)
        expr = self._rewrite_transpose(expr, notes)
        expr = self._rewrite_products(expr, notes)
        expr = self._rewrite_pointwise(expr, notes)
        expr = self._rewrite_nabla_point(expr, notes)
        expr = self._rewrite_callables(expr, exclude, notes)
        return expr

    @staticmethod
    def _split_apply(node: Apply):
        head, *args = node.args
        mod = None
        if args and isinstance(args[-1], CallMod):
            mod, args = args[-1], args[:-1]
        return head, args, mod

    def _stat_sides(self, rel) -> bool:
        """An inequality with a statistic of data (``Var``, ``std``, ``corr`` …) applied on a side."""
        if not isinstance(rel, Relational) or isinstance(rel, (sympy.Eq, sympy.Ne)):
            return False
        names = _STATS_OPS | {"corr", "percentile", "quantile", "mode", "median"}
        return any(str(n.args[0]) in names for n in rel.atoms(Apply))

    def _eventish(self, a) -> bool:
        if isinstance(a, (Relational, Mid, BooleanFunction, sympy.Not, Prime)):
            return True
        if isinstance(a, Rel) and str(a.args[0]) in _SET_OPS:
            return True
        if getattr(a, "is_Pow", False) and a.exp == Symbol("c"):
            return True
        return _is_rv(a)

    def _matrixish(self, b) -> bool:
        if isinstance(b, MatrixBase) or b.atoms(MatrixBase):
            return True
        return any(isinstance(coerce(self._lookup(s.name)), MatrixBase)
                   for s in b.free_symbols if isinstance(s, Symbol))

    def _call_kind(self, head, args, mod) -> tuple[str, str]:
        """Is ``head(args)`` a function call or a product?"""
        kind = str(mod.args[0]) if mod is not None else None
        if len(args) != 1 or kind in ("prime", "nderiv"):
            return "call", ""
        if isinstance(head, (Deco, Compose, Lambda, Derivative)):
            return "call", ""
        if not isinstance(head, Symbol):
            return ("product", "") if getattr(head, "is_number", False) else ("call", "")
        name, a = head.name, args[0]
        raw = self._lookup(name)
        functionish = _callable_value(raw) or (raw is None and (
            name in FUNCTION_LETTERS or name in KNOWN_FUNCTIONS or name in _STATS_OPS
            or re.fullmatch(r"[A-Z]_\d+", name) is not None))
        if kind == "iter":
            return "call", ""
        if kind in ("power", "inverse"):          # x^2(y), x^{-1}(x+1): powers of a factor
            if functionish:
                return "call", ""
            if len(name) == 1 and name.islower():
                return "product", ""       # x^4(1 - x)^4: obviously a product
            return "product", (f"{name}{mod.text()}(…) read as a product; write "
                               f"\\left({name}(x)\\right)^2 if {name} is a function")
        if raw is not None:
            if _callable_value(raw):
                return "call", ""
            v = coerce(raw)
            if v is None or _is_rv(v) or isinstance(v, sympy.Set):
                return "call", ""
            if name in _STATS_OPS and self._eventish(a):
                return "call", f"{name}(…) read as a probability ({name} is also bound, to {interp.short(v)}, but not used here)"
            return "product", f"{name}(…) read as multiplication because {name} = {interp.short(v)}"
        if head in self._dep and not isinstance(a, Symbol):
            return "product", (f"{name} is the unknown function here, so {name}({interp.pyform(a)}) "
                               f"is read as {name}·({interp.pyform(a)})")
        if name == "e":
            return "product", "e(…) read as Euler's number times (…)"
        if name in _PROB_HEADS and (a.is_number or isinstance(a, (sympy.Add, sympy.Mul, Pow))) \
                and not self._eventish(a):
            return "product", f"{name}({interp.pyform(a)}) read as multiplication (a probability needs an event)"
        if functionish:
            return "call", ""
        if raw is None and len(name) == 1 and name.islower() and getattr(a, "is_number", False):
            return "product", (f"{name}({interp.pyform(a)}) read as multiplication ({name} is not a function; "
                               f"define {name}(x) = … first to call it)")
        if head in a.free_symbols:
            return "product", f"{name}({interp.pyform(a)}) read as multiplication ({name} appears inside)"
        single = len(name) == 1 or name in _GREEK
        if single and not (isinstance(a, Symbol) or a.is_number):
            return "product", (f"{name}({interp.pyform(a)}) read as multiplication; "
                               f"define {name}(x) = … first to make {name} a function")
        return "call", ""

    def _rewrite_products(self, expr, notes):
        if not expr.has(Apply):
            return expr

        def kind(n):
            return self._call_kind(*self._split_apply(n))

        def base_of(n):
            head, args, mod = self._split_apply(n)
            if mod is not None and str(mod.args[0]) == "power":
                return Pow(head, mod.args[1], evaluate=False), args[0]
            if mod is not None and str(mod.args[0]) == "inverse":
                return Pow(head, sympy.S.NegativeOne, evaluate=False), args[0]
            return head, args[0]

        def callpow(n):
            ap, e = n.args
            if isinstance(ap, Apply) and kind(ap)[0] == "product":
                k, why = kind(ap)
                if why:
                    notes.append(why)
                head, arg = base_of(ap)
                return Mul(head, Pow(arg, e, evaluate=False), evaluate=False)
            return Pow(ap, e, evaluate=False)

        def prod(n):
            k, why = kind(n)
            if why:
                notes.append(why)
            head, arg = base_of(n)
            return Mul(head, arg, evaluate=False)

        def postfix_prod(n):                      # q (2j-1)!! is q · (2j-1)!!, not (q(2j-1))!!
            ap = n.args[0]
            k, why = kind(ap)
            if why:
                notes.append(why)
            head, arg = base_of(ap)
            return Mul(head, n.func(arg), evaluate=False)

        expr = _ordered_replace(expr, lambda n: isinstance(n, (sympy.factorial, sympy.factorial2))
                                and isinstance(n.args[0], Apply) and kind(n.args[0])[0] == "product", postfix_prod)
        expr = _ordered_replace(expr, lambda n: isinstance(n, CallPow), callpow)
        expr = _ordered_replace(expr, lambda n: isinstance(n, Apply) and kind(n)[0] == "product", prod)
        for n in expr.atoms(Apply):
            k, why = kind(n)
            if why:
                notes.append(why)
        return expr

    def _rewrite_parametric(self, expr, notes):
        """``x(t) = \\cos t``, ``y(t) = \\sin t``, then ``\\frac{dy}{dx}``: the parametric derivative
        (dy/dt)/(dx/dt), applied n times for d^n y/dx^n."""
        if not expr.has(Derivative):
            return expr

        def fns(n):
            if not (isinstance(n, Derivative) and isinstance(n.expr, Symbol) and len(n.variable_count) == 1):
                return None
            v = n.variable_count[0][0]
            fy, fx = self._lookup(n.expr.name), self._lookup(getattr(v, "name", ""))
            if isinstance(fy, Lambda) and isinstance(fx, Lambda) and len(fy.variables) == len(fx.variables) == 1:
                return fy, fx
            return None

        def pd(n):
            fy, fx = fns(n)
            v, k = n.variable_count[0]
            t = fx.variables[0]
            dxdt = sympy.diff(fx.expr, t)
            cur = fy.expr.xreplace({fy.variables[0]: t})
            for _ in range(int(k)):
                cur = sympy.simplify(sympy.diff(cur, t) / dxdt)
            notes.append(f"{n.expr} and {v} are both functions of {t}: d{n.expr}/d{v} is the parametric "
                         f"derivative (d{n.expr}/d{t}) / (d{v}/d{t})")
            return cur
        return expr.replace(lambda n: fns(n) is not None, pd)

    def _rewrite_transpose(self, expr, notes):
        T, H = Symbol("T"), Symbol("H")

        def is_tp(n):
            return (n.is_Pow and n.exp in (T, H) and self._lookup(n.exp.name) is not None
                    and self._matrixish(n.base))

        def tp(n):
            notes.append(f"^{n.exp} read as the {'transpose' if n.exp == T else 'conjugate transpose'} "
                         f"({n.exp} is also bound, but not used here)")
            return Transposed(n.base, sympy.true if n.exp == H else sympy.false)
        return _ordered_replace(expr, is_tp, tp)

    def _rewrite_pointwise(self, expr, notes):
        """``(f+g)(x)``, ``(fg)(x)``, ``(f/g)(x)``: an expression built only from functions,
        followed by an argument, applies each function to it."""
        def split(n):
            if not n.is_Mul or len(n.args) < 2:
                return None
            *prefix, arg = n.args
            if len(prefix) == 1 and isinstance(prefix[0], (Symbol, Apply)):
                return None
            f = prefix[0] if len(prefix) == 1 else Mul(*prefix, evaluate=False)
            if not isinstance(f, Basic) or f.is_number or f.has(Apply, AppliedUndef, *_CALCULUS, Marker):
                return None          # (f(3) - f(1))/(3 - 1): the functions are already applied
            syms = [s for s in f.free_symbols if isinstance(s, Symbol)]
            if not syms or not all(self._is_function_value(s) for s in syms):
                return None
            if isinstance(arg, Symbol) and self._is_function_value(arg):
                return None
            return f, arg

        def pw(n):
            f, arg = split(n)
            args = list(arg) if isinstance(arg, Tuple) else [arg]
            notes.append(f"({interp.pyform(f)})({interp.pyform(arg)}) read as applying each function")
            return f.xreplace({s: Apply(s, *args) for s in f.free_symbols if isinstance(s, Symbol)})
        return expr.replace(lambda n: split(n) is not None, pw) if expr.has(Mul) else expr

    def _rewrite_nabla_point(self, expr, notes):
        def is_np(n):
            if not (isinstance(n, Nabla) and str(n.args[0]) == "grad" and isinstance(n.args[1], Apply)):
                return False
            head, args, mod = self._split_apply(n.args[1])
            lam = self._lookup(getattr(head, "name", ""))
            return (isinstance(lam, Lambda) and mod is None and len(args) == len(lam.variables)
                    and all(getattr(a, "is_number", False) for a in args))

        def np_(n):
            head, args, _ = self._split_apply(n.args[1])
            lam = self._lookup(head.name)
            pt = dict(zip(lam.variables, args))
            notes.append(f"∇{head.name} evaluated at ({', '.join(map(str, args))})")
            return ImmutableMatrix([sympy.diff(lam.expr, v).subs(pt) for v in lam.variables])
        return expr.replace(is_np, np_) if expr.has(Nabla) else expr

    def _rewrite_callables(self, expr, exclude, notes):
        rep = {}
        heads = self._heads(expr)
        ctx = self._context_var(expr) or Symbol("x")
        for s in expr.free_symbols:
            if not isinstance(s, Symbol) or s in heads or s.name in exclude:
                continue
            raw = self._lookup(s.name)
            if raw is None or isinstance(raw, Lambda) or not _callable_value(raw):
                continue
            try:
                out = coerce(raw(ctx))
            except Exception:
                out = None
            if out is None:
                raise SemanticsError(f"{s.name} is a Python function; it cannot be used as a formula",
                                     hint=f"call it with numbers ({s.name}(2)), or define it as a statement: "
                                          f"`{s.name}(x) = …`.")
            rep[s] = out
            notes.append(f"{s.name} (a Python function) used as {s.name}({ctx}) = {interp.pyform(out)}")
        return expr.xreplace(rep) if rep else expr

    # --- events: P(A), P(A \cap B), P(A \mid B) ------------------------------------------------------
    def _complement_of(self, ev):
        if isinstance(ev, Prime) and ev.args[1] == 1:
            return ev.args[0]
        if isinstance(ev, sympy.Not):
            return ev.args[0]
        if getattr(ev, "is_Pow", False) and ev.exp == Symbol("c"):
            return ev.base
        if isinstance(ev, Deco) and str(ev.args[1]) in ("bar", "overline"):
            return ev.args[0]
        return None

    def _ev_key(self, ev) -> Optional[str]:
        comp = self._complement_of(ev)
        if comp is not None:
            k = self._ev_key(comp)
            return None if k is None else f"not({k})"
        if isinstance(ev, Symbol):
            return ev.name
        if isinstance(ev, Rel):
            op = str(ev.args[0])
            ka, kb = self._ev_key(ev.args[1]), self._ev_key(ev.args[2])
            if ka is None or kb is None:
                return None
            if op in ("cap", "cup"):
                return f"{op}({','.join(sorted([ka, kb]))})"
            if op == "setminus":
                return f"cap({','.join(sorted([ka, f'not({kb})']))})"
            return None
        if isinstance(ev, Mid):
            ka, kb = self._ev_key(ev.args[0]), self._ev_key(ev.args[1])
            return None if ka is None or kb is None else f"given({ka}|{kb})"
        if isinstance(ev, Relational) and not _is_rv(ev):
            return interp.pyform(ev)
        return None

    def _prob(self, ev, depth: int = 0):
        reg = self._reg("_openlab_events")
        key = self._ev_key(ev)
        if key is None or depth > 6:
            return None
        if key in reg:
            return reg[key]
        if f"not({key})" in reg:
            return 1 - reg[f"not({key})"]
        if isinstance(ev, Symbol):                 # total probability over a stated partition
            for k, v in list(reg.items()):
                m = re.fullmatch(rf"given\({re.escape(key)}\|(.+)\)", k)
                if not m:
                    continue
                cond = m.group(1)
                other = cond[4:-1] if cond.startswith("not(") else f"not({cond})"
                g2, pc = reg.get(f"given({key}|{other})"), self._pk(cond)
                if g2 is not None and pc is not None:
                    return v * pc + g2 * (1 - pc)
            parts = []                             # B_1, B_2, …: stated conditionals whose P sum to 1
            for k, v in reg.items():
                m = re.fullmatch(rf"given\({re.escape(key)}\|(.+)\)", k)
                if m and self._pk(m.group(1)) is not None:
                    parts.append((m.group(1), self._pk(m.group(1)), v))
            if len(parts) >= 2:
                try:
                    whole = sympy.simplify(sum(p for _, p, _ in parts) - 1) == 0
                except Exception:
                    whole = False
                if whole:
                    self._event_notes.append(
                        f"total probability over {', '.join(c for c, _, _ in parts)} (their probabilities sum "
                        f"to 1, so they are taken as a partition)")
                    return sum(p * v for _, p, v in parts)
        comp = self._complement_of(ev)
        if comp is not None:
            v = self._prob(comp, depth + 1)
            return None if v is None else 1 - v
        if isinstance(ev, Rel):
            op, a, b = str(ev.args[0]), ev.args[1], ev.args[2]
            pa, pb = self._prob(a, depth + 1), self._prob(b, depth + 1)
            if op == "cup":
                pab = self._prob(Rel(Symbol("cap"), a, b), depth + 1)
                if None not in (pa, pb, pab):
                    return pa + pb - pab
            if op == "cap":
                ka, kb = self._ev_key(a), self._ev_key(b)
                if f"given({ka}|{kb})" in reg and pb is not None:
                    return reg[f"given({ka}|{kb})"] * pb
                if f"given({kb}|{ka})" in reg and pa is not None:
                    return reg[f"given({kb}|{ka})"] * pa
                cup = reg.get(f"cup({','.join(sorted([ka, kb]))})")
                if cup is not None and None not in (pa, pb):
                    return pa + pb - cup
        if isinstance(ev, Mid):
            a, b = ev.args
            pab = self._prob(Rel(Symbol("cap"), a, b), depth + 1)
            pb = self._prob(b, depth + 1)
            if None not in (pab, pb) and pb != 0:
                return pab / pb
        return None

    def _is_event_call(self, n) -> bool:
        if not isinstance(n, Apply) or not isinstance(n.args[0], Symbol) or n.args[0].name not in _PROB_HEADS:
            return False
        raw = self._lookup(n.args[0].name)
        if _callable_value(raw):
            return False
        _, args, mod = self._split_apply(n)
        if len(args) != 1 or mod is not None:
            return False
        ev = args[0]
        if any(_is_rv(coerce(self._lookup(s.name))) for s in ev.free_symbols if isinstance(s, Symbol)):
            return False          # a random variable: sympy.stats handles it
        if raw is not None and not self._eventish(ev):
            return False          # P bound to a number: P(A) is a product
        return self._ev_key(ev) is not None

    def _rewrite_events(self, expr, notes):
        if not expr.has(Apply):
            return expr

        def val(n):
            ev = n.args[1]
            self._event_notes = []
            v = self._prob(ev)
            notes.extend(self._event_notes)
            if v is None:
                notes.append(f"{n.args[0]}({interp.pyform(ev)}) is not known yet (state it, e.g. P(A) = 0.3)")
                return Function(n.args[0].name)(ev)
            if isinstance(v, Expr) and v.is_number and v.is_real and not (-1e-12 <= v <= 1 + 1e-12):
                raise SemanticsError(f"the stated probabilities are inconsistent: they give "
                                     f"{n.args[0]}({interp.pyform(ev)}) = {interp.pyform(v)}",
                                     hint="check the stated values; a probability lies between 0 and 1.")
            notes.append(f"{n.args[0]}({interp.pyform(ev)}) = {interp.pyform(v)} from the stated probabilities")
            return v
        return expr.replace(self._is_event_call, val)

    # --- function application ------------------------------------------------------
    def _late_calls(self, expr) -> bool:
        recs = self._reg("_openlab_recur")
        return any(self._is_function_value(Symbol(f.func.__name__))
                   or (f.func.__name__ in recs and len(f.args) == 1 and f.args[0].is_Integer)
                   for f in expr.atoms(AppliedUndef))

    def _apply_all(self, expr, notes):
        if isinstance(expr, MatrixBase):            # [f(1) f(2) f(3)]: calls inside the entries
            if any(isinstance(e, Basic) and e.has(Apply, AppliedUndef) for e in expr):
                return ImmutableMatrix(expr.rows, expr.cols, [self._apply_all(e, notes) for e in expr])
            return expr
        if isinstance(expr, Basic) and any(isinstance(M, MatrixBase) for M in expr.atoms(MatrixBase)) \
                and expr.has(Apply):
            expr = expr.xreplace({M: self._apply_all(M, notes) for M in expr.atoms(MatrixBase)})
        recs = self._reg("_openlab_recur")
        if recs and expr.has(AppliedUndef):
            rep = {f: self._iterate(f.func.__name__, recs[f.func.__name__], int(f.args[0]), notes)
                   for f in expr.atoms(AppliedUndef)
                   if f.func.__name__ in recs and len(f.args) == 1 and f.args[0].is_Integer}
            if rep:
                expr = expr.xreplace(rep)
        if expr.has(Apply):
            expr = _ordered_replace(expr, lambda n: isinstance(n, Apply),
                                    lambda n: self._apply(n.args[0], n.args[1:], notes))
        rep = {}
        for f in expr.atoms(AppliedUndef):
            if self._is_function_value(Symbol(f.func.__name__)):
                rep[f] = self._apply(Symbol(f.func.__name__), f.args, notes)
        return expr.xreplace(rep) if rep else expr

    def _apply(self, head, args, notes: list):
        mod = args[-1] if args and isinstance(args[-1], CallMod) else None
        if mod is not None:
            args = args[:-1]
        if mod is not None and str(mod.args[0]) == "iter":
            return self._iterate_call(head, list(args), mod, notes)
        power = mod.args[1] if mod is not None and str(mod.args[0]) == "power" else None
        if power is not None:
            mod = None
        out = self._apply_inner(head, list(args), mod, notes)
        return out ** power if power is not None else out

    def _iterate_call(self, head, args, mod, notes):
        """``T^{\\circ k}(x)``: T applied k times (k a whole number; a symbolic k waits)."""
        k = sympy.sympify(mod.args[1])
        k = k.doit() if isinstance(k, Basic) else k
        if not k.is_Integer and isinstance(k, Basic) and not _free_vars(k):
            k = self._settle(k, [])                   # T^{∘κ(2)}: the count is computed first
        if not k.is_number:
            return Apply(head, *args, mod)            # T^{∘j}(27) inside \max_j: computed per j
        if not (k.is_Integer and k >= 0):
            raise SemanticsError(f"{head}^(∘{k}) needs a whole number of repetitions (got {interp.pyform(k)})")
        if len(args) != 1:
            raise SemanticsError(f"{head}^(∘{k}) iterates a function of one argument")
        if k > 100_000:
            raise SemanticsError(f"{head}^(∘{k}): too many repetitions to compute one by one")
        val = args[0]
        fast = self._fast_orbit(head, val, int(k))
        if fast is not None:
            return fast
        reg = self._reg("_openlab_assume")                 # m ∈ ℤ: each step's parity is decided at once
        rep = {s_: Symbol(s_.name, **reg[s_.name]) for s_ in getattr(val, "free_symbols", ())
               if isinstance(s_, Symbol) and s_.name in reg and not s_.assumptions0.get("real")}
        val = val.xreplace(rep) if rep else val
        start = val
        for step_ in range(int(k)):
            val = self._apply_inner(head, [val], None, notes)
            if isinstance(val, Expr) and not val.is_number and not val.has(sympy.Piecewise):
                val = sympy.expand(val)                  # 3(64m + 7)/2 + 1/2 -> 96m + 11: parity decides
            elif int(k) > 1 and isinstance(val, Basic) and val.has(sympy.Piecewise):
                # an undecided case (r not known to be a whole number, or its parity unknown): each further step
                # would double the cases -- keep T^{∘k}(r) as written until r is a number
                return Apply(head, *[a.xreplace({v: k_ for k_, v in rep.items()}) if rep else a for a in [start]], mod)
        return val.xreplace({v: k_ for k_, v in rep.items()}) if rep and isinstance(val, Basic) else val

    _OPAQUE = (Sum, Product, sympy.Piecewise, sympy.Mod, sympy.floor, sympy.ceiling, Pick)

    def _opaque_terms(self, f) -> bool:
        """Terms no closed-form summation handles: nested sums, remainders, number-theory functions."""
        if not isinstance(f, Basic):
            return False
        return f.has(*self._OPAQUE) or any(hasattr(type(n), "_pyfn") for n in sympy.preorder_traversal(f))

    @staticmethod
    def _small_range(node, cap=5000):
        if len(node.limits) != 1:
            return False
        k, a, b = node.limits[0]
        a, b = sympy.sympify(a), sympy.sympify(b)
        return bool(a.is_Integer and b.is_Integer and 0 <= b - a < cap)

    @staticmethod
    def _written_out(node):
        k, a, b = node.limits[0]
        terms = [node.function.xreplace({k: Integer(v)}) for v in range(int(a), int(b) + 1)]
        return sympy.Add(*terms) if isinstance(node, Sum) else sympy.Mul(*terms)

    def _fast_or_keep(self, node):
        try:
            return _fast.evaluate(node, hook=self._fast_hook)
        except (_fast.Unsupported, KeyError, ZeroDivisionError, OverflowError, ValueError, TypeError):
            return node

    def _fast_hook(self, e, names):
        """Compile openlab's own nodes for :mod:`fasteval`: a call of a defined one-letter function,
        an iterate T^{∘k}(x), and a search \\min\\{k ≥ k0 : P(k)\\}. None = not compilable."""
        if isinstance(e, Apply) and isinstance(e.args[0], Symbol):
            fn = self._lookup(e.args[0].name)
            if not isinstance(fn, Lambda):
                return None
            args = list(e.args[1:])
            mod = args.pop() if args and isinstance(args[-1], CallMod) else None
            if len(args) != len(fn.variables):
                return None
            vnames = [v.name for v in fn.variables]
            body = _fast.compile_expr(fn.expr, set(vnames), self._fast_hook)
            fargs = [_fast.compile_expr(a, names, self._fast_hook) for a in args]
            if mod is None:
                return lambda env: body(dict(zip(vnames, [f(env) for f in fargs])))
            if str(mod.args[0]) != "iter" or len(vnames) != 1:
                return None
            fk = _fast.compile_expr(mod.args[1], names, self._fast_hook)
            orbits: dict = {}
            name = vnames[0]

            step = getattr(body, "direct", None) or (lambda v: body({name: v}))

            def iterate(env):
                x, k = fargs[0](env), _fast._as_int(fk(env), "a repetition count")
                if k < 0 or k > 1_000_000:
                    raise _fast.Unsupported("repetitions out of range")
                orbit = orbits.get(x)
                if orbit is None:
                    if len(orbits) > 200_000:
                        orbits.clear()                 # keep memory bounded on long sweeps
                    orbit = orbits[x] = [x]
                while len(orbit) <= k:
                    orbit.append(step(orbit[-1]))
                return orbit[k]
            return iterate
        if isinstance(e, sympy.Indexed) and len(e.indices) == 1:   # c[n] of a bound list: looked up
            vals = coerce(self._lookup(str(e.base.label)))
            if isinstance(vals, (Tuple, MatrixBase)):
                items = []
                for x in (list(vals) if isinstance(vals, Tuple) else list(vals)):
                    if getattr(x, "is_Integer", False):
                        items.append(int(x))
                    elif getattr(x, "is_Rational", False):
                        items.append(Fraction(int(x.p), int(x.q)))
                    else:
                        return None
                fi = _fast.compile_expr(e.indices[0], names, self._fast_hook)
                lab = str(e.base.label)

                def look(env):
                    k_ = _fast._as_int(fi(env), "an index")
                    if not 1 <= k_ <= len(items):
                        raise SemanticsError(f"{lab}[{k_}] is out of range: {lab} has {len(items)} entries "
                                             f"(counting from 1)")
                    return items[k_ - 1]
                return look
            return None
        if isinstance(e, SetSum):                      # Σ_{p ∈ S} f(p): S finite, or primes(…) of the variables
            op, lam, S_ = e.args
            body = _fast.compile_expr(lam.expr, set(names) | {lam.variables[0].name}, self._fast_hook)
            vname, is_sum = lam.variables[0].name, str(op) == "sum"
            if isinstance(S_, sympy.FiniteSet) or (isinstance(S_, sympy.Range) and S_.is_finite_set):
                elems = [int(x) for x in S_] if all(getattr(x, "is_Integer", False) for x in S_) else None
                if elems is None:
                    return None
                get = lambda env: elems
            elif isinstance(S_, _PRIMES):
                fa = [_fast.compile_expr(a, names, self._fast_hook) for a in S_.args]

                def get(env):
                    b = [_fast._as_int(f(env), "a bound") for f in fa]
                    lo, hi = (2, b[0]) if len(b) == 1 else (b[0], b[1])
                    return list(sympy.primerange(lo, hi + 1))
            else:
                return None

            def setsum(env):
                e2 = dict(env)
                acc = 0 if is_sum else 1
                for x in get(env):
                    e2[vname] = x
                    t_ = body(e2)
                    acc = acc + t_ if is_sum else acc * t_
                return acc
            if is_sum:
                def stream(env):
                    e2 = dict(env)
                    for x in get(env):
                        e2[vname] = x
                        yield body(e2)
                setsum.stream, setsum.nonneg = stream, _fast.nonneg_static(lam.expr)
            return setsum
        if isinstance(e, Optim) and str(e.args[0]) in ("max", "min", "argmax", "argmin") \
                and not isinstance(e.args[1], sympy.ConditionSet):      # max over a range of another letter
            op_, body_, spec_ = str(e.args[0]), e.args[1], e.args[2]
            rng = self._compiled_range_spec(spec_, names)
            if rng is None:
                return None
            var_, get = rng
            fb = _fast.compile_expr(body_, set(names) | {var_.name}, self._fast_hook)

            def agg(env):
                e2 = dict(env)
                best = bk = None
                for k_ in get(env):
                    e2[var_.name] = k_
                    v_ = fb(e2)
                    if best is None or (v_ > best if op_ in ("max", "argmax") else v_ < best):
                        best, bk = v_, k_
                if best is None:
                    if op_ == "max":
                        return float("-inf")               # max ∅ = −∞ (so the first n is a record)
                    if op_ == "min":
                        return float("inf")
                    raise _fast.Unsupported("an empty range")
                return bk if op_.startswith("arg") else best
            return agg
        if isinstance(e, Bars) and isinstance(e.args[0], sympy.ConditionSet):   # \#\{r : 0 ≤ r ≤ 2^k − 1, …\}
            cs = e.args[0]
            rng = self._compiled_range(cs.sym, cs.condition, names)
            if rng is None:
                return None
            flo, fhi = rng
            fc = _fast.compile_expr(cs.condition, set(names) | {cs.sym.name}, self._fast_hook)
            vn = cs.sym.name

            def count(env):
                e2 = dict(env)
                lo_, hi_ = flo(env), fhi(env)
                if hi_ - lo_ > 5_000_000:
                    raise _fast.Unsupported("too many to count")
                n_ = 0
                for k_ in range(lo_, hi_ + 1):
                    e2[vn] = k_
                    n_ += 1 if fc(e2) else 0
                return n_
            return count
        if isinstance(e, Optim) and str(e.args[0]) == "min" and isinstance(e.args[1], sympy.ConditionSet):
            cs = e.args[1]
            var = cs.sym
            lo, hi = self._int_bounds(var, cs.base_set, cs.condition)
            if lo is None or isinstance(lo, sympy.Set) or not self._is_whole(var, cs.base_set, cs.condition):
                return None
            direct = self._direct_orbit_search(cs, var, lo, hi, names)
            if direct is not None:
                return direct
            cond = _fast.compile_expr(cs.condition, set(names) | {var.name}, self._fast_hook)
            top = hi if hi is not None else lo + 1_000_000
            keys = sorted(n_.name for n_ in _free_vars(cs.condition) if n_.name != var.name)
            memo: dict = {}

            def search(env):
                key = tuple(env.get(k_) for k_ in keys)
                if key in memo:
                    return memo[key]
                e2 = dict(env)
                for k in range(lo, top + 1):
                    e2[var.name] = k
                    if cond(e2):
                        memo[key] = k
                        return k
                raise _fast.Unsupported("no k found")
            return search
        return None

    def _direct_orbit_search(self, cs, var, lo, hi, names):
        """min{k ≥ k0 : T^{∘k}(x) ⋈ y} for a map T with an integer fast path: one loop over the orbit
        (a stopping time or total stopping time costs one integer step per k)."""
        parts = sympy.And.make_args(cs.condition) if isinstance(cs.condition, sympy.And) else (cs.condition,)
        rels = [c_ for c_ in parts if not (isinstance(c_, Relational) and c_.free_symbols == {var}
                                           and (c_.lhs - c_.rhs).is_polynomial(var))]
        if len(rels) != 1 or not isinstance(rels[0], Relational) or type(rels[0]) not in _fast._REL:
            return None
        r_ = rels[0]
        side_it = [x for x in (r_.lhs, r_.rhs) if isinstance(x, Apply) and any(
            isinstance(a, CallMod) and str(a.args[0]) == "iter" and a.args[1] == var for a in x.args)]
        if len(side_it) != 1:
            return None
        it = side_it[0]
        other = r_.rhs if it is r_.lhs else r_.lhs
        if var in other.free_symbols:
            return None
        fn = self._lookup(it.args[0].name) if isinstance(it.args[0], Symbol) else None
        if not (isinstance(fn, Lambda) and len(fn.variables) == 1):
            return None
        try:
            body = _fast.compile_expr(fn.expr, {fn.variables[0].name}, self._fast_hook)
            fx = _fast.compile_expr(it.args[1], names, self._fast_hook)
            fo = _fast.compile_expr(other, names, self._fast_hook)
        except _fast.Unsupported:
            return None
        step = getattr(body, "direct", None)
        if step is None:
            return None
        op = _fast._REL[type(r_)]
        it_left = it is r_.lhs
        top = hi if hi is not None else lo + 1_000_000
        memo: dict = {}

        def search(env):
            x, y = fx(env), fo(env)
            key = (x, y)
            if key in memo:
                return memo[key]
            v = x
            for _ in range(lo):
                v = step(v)
            k = lo
            while k <= top:
                if (op(v, y) if it_left else op(y, v)):
                    if len(memo) < 500_000:
                        memo[key] = k
                    return k
                v = step(v)
                k += 1
            raise _fast.Unsupported("no k found")
        return search

    def _compiled_range(self, var, cond, names):
        """Whole-number bounds of var from the comparisons of cond with var alone on one side (the other side
        in the letters ``names``): (lo(env), hi(env)) as compiled functions, else None."""
        los, his = [], []
        for c_ in (sympy.And.make_args(cond) if isinstance(cond, sympy.And) else (cond,)):
            if not isinstance(c_, Relational) or isinstance(c_, (sympy.Eq, sympy.Ne)):
                continue
            for side, other, flip in ((c_.lhs, c_.rhs, False), (c_.rhs, c_.lhs, True)):
                if side == var and var not in other.free_symbols:
                    try:
                        fo = _fast.compile_expr(other, names, self._fast_hook)
                    except _fast.Unsupported:
                        return None
                    kind = type(c_)
                    ge = (kind in (sympy.GreaterThan, sympy.StrictGreaterThan)) != flip
                    strict = kind in (sympy.StrictGreaterThan, sympy.StrictLessThan)
                    if ge:
                        los.append((fo, strict))
                    else:
                        his.append((fo, strict))
        if not los or not his:
            return None

        def lo(env):
            out = []
            for fo, strict in los:
                v_ = Fraction(fo(env))
                out.append(math.floor(v_) + 1 if strict else math.ceil(v_))
            return max(out)

        def hi(env):
            out = []
            for fo, strict in his:
                v_ = Fraction(fo(env))
                out.append(math.ceil(v_) - 1 if strict else math.floor(v_))
            return min(out)
        return lo, hi

    def _compiled_range_spec(self, spec, names):
        """The index and its values for a max/min spec: `a ≤ j ≤ b` (bounds in other letters) or `j ∈ S`."""
        if isinstance(spec, sympy.Contains) and isinstance(spec.args[0], Symbol) and \
                isinstance(spec.args[1], (sympy.FiniteSet, sympy.Range)) and spec.args[1].is_finite_set:
            vals = [int(x) for x in spec.args[1]]
            return spec.args[0], (lambda env: vals)
        if not isinstance(spec, (Relational, sympy.And)):
            return None
        parts = sympy.And.make_args(spec) if isinstance(spec, sympy.And) else (spec,)
        bare = None
        for c_ in parts:
            sides = {x for x in (getattr(c_, "lhs", None), getattr(c_, "rhs", None)) if isinstance(x, Symbol)}
            bare = sides if bare is None else bare & sides
        if not bare or len(bare) != 1:
            return None
        var = next(iter(bare))
        rng = self._compiled_range(var, spec, set(names) - {var.name})
        if rng is None:
            return None
        lo, hi = rng
        return var, (lambda env: range(lo(env), hi(env) + 1))

    def _fast_orbit(self, head, x, k):
        """T^{∘k}(x) for a number x and a whole-number map T: iterated in plain Python, the orbit
        remembered so a search over k extends it one step at a time."""
        if not (isinstance(x, Basic) and x.is_Rational and isinstance(head, Symbol)):
            return None
        fn = self._lookup(head.name)
        if not (isinstance(fn, Lambda) and len(fn.variables) == 1):
            return None
        cache = self.__dict__.setdefault("_orbits", {})
        key = (head.name, fn, x)
        if key not in cache:
            try:
                step = _fast.compile_expr(fn.expr, {fn.variables[0].name})
            except _fast.Unsupported:
                return None
            cache[key] = (step, [int(x) if x.is_Integer else x])
        step, orbit = cache[key]
        name = fn.variables[0].name
        try:
            while len(orbit) <= k:
                orbit.append(step({name: orbit[-1]}))
        except (_fast.Unsupported, ZeroDivisionError, OverflowError, ValueError, TypeError):
            return None
        return _fast.to_sympy(orbit[k])

    def _alpha_safe(self, fn, args):
        """σ(r + 2^k) with σ(n) = min{k : …}: rename the letters σ binds inside (a search index, a sum
        index, a max index) that occur in the arguments, so the argument's k is not captured."""
        avoid = set()
        for a in args:
            if isinstance(a, Basic):
                avoid |= {s_.name for s_ in a.free_symbols if isinstance(s_, Symbol)}
        body = fn.expr
        if not avoid or not isinstance(body, Basic):
            return fn
        taken = avoid | {s_.name for s_ in body.atoms(Symbol)}

        def fresh(sym):
            nm = sym.name + "′"
            while nm in taken:
                nm += "′"
            taken.add(nm)
            return Symbol(nm, **sym.assumptions0)

        def rename(node):
            if isinstance(node, sympy.ConditionSet) and node.sym.name in avoid:
                f_ = fresh(node.sym)
                return sympy.ConditionSet(f_, node.condition.xreplace({node.sym: f_}), node.base_set)
            if isinstance(node, (Sum, Product)) and any(l[0].name in avoid for l in node.limits):
                rep_ = {l[0]: fresh(l[0]) for l in node.limits if l[0].name in avoid}
                return node.xreplace(rep_)
            if isinstance(node, Optim):
                v_ = _optim_var(node.args[2])
                if v_ is not None and v_.name in avoid:
                    return node.xreplace({v_: fresh(v_)})
            return node
        new = body.replace(lambda n: isinstance(n, (sympy.ConditionSet, Sum, Product, Optim)), rename)
        return fn if new == body else Lambda(fn.variables, new)

    def _apply_inner(self, head, args, mod, notes):
        if isinstance(head, Lambda):
            return self._apply_lambda(head, args, mod, notes)
        if isinstance(head, Compose):
            f, g = head.args
            inner_v = self._apply(g, args, notes)
            return self._apply(f, [inner_v], notes)
        if isinstance(head, Derivative) and isinstance(head.expr, Symbol):
            fn = self._lookup(head.expr.name)
            var, n = head.variable_count[0]
            if isinstance(fn, Lambda):
                return sympy.diff(fn.expr, fn.variables[0], n).subs(fn.variables[0], args[0])
            f = Function(head.expr.name)
            return Subs(Derivative(f(var), (var, n)), var, args[0])
        if isinstance(head, Symbol):
            name = head.name
            v = self._lookup(name)
            if v is not None:
                if isinstance(v, Lambda):
                    return self._apply_lambda(v, args, mod, notes)
                if isinstance(v, sympy.FunctionClass) and mod is not None:
                    u = Symbol("_u")
                    return self._apply_lambda(Lambda(u, v(u)), args, mod, notes)
                if _callable_value(v):
                    if mod is not None:
                        raise UnsupportedError(f"cannot take a derivative/inverse of the Python function {name}",
                                               hint=f"define it as a statement (`{name}(x) = …`) to differentiate it.")
                    return self._call_python(name, v, args, notes)
                cv = coerce(v)
                if cv is None:
                    notes.append(f"{name} is bound to a {type(v).__name__}; {name}(…) is treated as an undefined function")
                elif mod is not None and str(mod.args[0]) in ("prime", "nderiv") and not isinstance(cv, (MatrixBase, sympy.Set)):
                    notes.append(f"{name} = {interp.short(cv)} is a constant, so its derivative is 0")
                    return sympy.S.Zero
                elif not (_is_rv(cv) or isinstance(cv, sympy.Set)
                          or (name in _STATS_OPS and args and self._eventish(args[0]))):
                    if len(args) == 1:
                        notes.append(f"{name}(…) read as multiplication because {name} = {interp.short(cv)}")
                        return cv * args[0]
                    raise SemanticsError(f"{name} = {interp.short(cv)} is a value, not a function of {len(args)} arguments")
            failed = self._reg("_openlab_failed").get(name)
            if failed and (name in KNOWN_FUNCTIONS or name in _STATS_OPS):
                raise SemanticsError(f"your definition of {name} (`{failed}`) failed, so {name}(…) is not defined here; "
                                     f"openlab will not fall back to the built-in {name}",
                                     hint="fix that definition, or rename your function.")
            stats = self._stats_apply(name, args, notes)
            if stats is not None:
                return stats
            fam = re.fullmatch(r"([A-Z])_(\d+)", name)
            if fam and fam.group(1) in _FAMILIES and len(args) == 1:
                notes.append(f"{name} read as the {_FAMILIES[fam.group(1)].__name__} function")
                return _FAMILIES[fam.group(1)](Integer(fam.group(2)), args[0])
            known = KNOWN_FUNCTIONS.get(name)
            if known is not None:
                if name == "Phi":
                    notes.append("Φ read as the standard normal CDF")
                if name == "sinc":
                    notes.append("sinc x = sin(x)/x (unnormalised; NumPy's sinc is sin(πx)/(πx))")
                if mod is not None:
                    u = Symbol("_u")
                    return self._apply_lambda(Lambda(u, sympy.sympify(known(u))), args, mod, notes)
                if name in _FORM_FUNCS:
                    self._keep_form = True
                try:
                    return sympy.sympify(known(*args))
                except (TypeError, ValueError, AttributeError) as exc:
                    raise SemanticsError(f"{name}({', '.join(map(interp.pyform, args))}) failed: {exc}") from None
            f = Function(name)
            if mod is not None:
                return self._apply_undefined_mod(f, args, mod)
            if name not in _QUIET_HEADS:
                notes.append(did_you_mean(name) or f"{name} is an undefined function")
            return f(*args)
        if isinstance(head, MatrixBase):
            if len(args) == 1:
                return self._mat_mul(head, _vec(args[0]))
            raise SemanticsError("a matrix cannot be applied to arguments",
                                 hint=r"write `A \cdot v` or `A v` for a product.")
        if isinstance(head, Basic) and head.is_number:
            notes.append(f"read as multiplication ({interp.pyform(head)} is a number)")
            return head * sympy.Mul(*args)
        return Function(str(head))(*args)

    def _apply_lambda(self, fn: Lambda, args, mod, notes):
        if mod is None:
            fn = self._alpha_safe(fn, args)
            try:
                out = fn(*args)
            except TypeError as exc:
                raise SemanticsError(f"wrong number of arguments: {exc}") from None
            if out is sympy.nan and isinstance(fn.expr, sympy.Piecewise):
                raise SemanticsError(f"no row of the cases applies at {', '.join(map(interp.pyform, args))}",
                                     hint=r"add an `\text{otherwise}` row, or check the conditions.")
            return out
        kind = str(mod.args[0])
        u = fn.variables[0]
        if kind in ("prime", "nderiv"):
            n = mod.args[1]
            if len(fn.variables) != 1:
                raise UnsupportedError("prime notation needs a one-variable function",
                                       hint=r"use a partial derivative: `\frac{\partial f}{\partial x}`.")
            if not getattr(n, "is_Integer", False):
                d = sympy.Dummy(u.name)
                return NthDerivAt(fn.expr.xreplace({u: d}), d, n, args[0])
            d = sympy.diff(fn.expr, u, int(n))
            return d.subs(u, args[0])
        y = Symbol("_y")
        sols = sympy.solve(Eq(fn.expr, y), u)
        if not sols:
            raise SemanticsError("could not invert the function")
        if len(sols) > 1:
            best = self._principal_branch(sols, y, args[0])
            notes.append(f"the inverse has {len(sols)} branches; using the principal one, "
                         f"{interp.pyform(best.subs(y, Symbol('y')))}")
            return best.subs(y, args[0])
        return sols[0].subs(y, args[0])

    @staticmethod
    def _principal_branch(sols, y, at):
        """sin⁻¹ is asin, not π − asin; x² inverts to +√y: prefer the bare inverse function of y,
        then a branch that is real and non-negative at the argument, then the smallest in size."""
        bare = (sympy.asin, sympy.acos, sympy.atan, sympy.acot, sympy.log, sympy.asinh, sympy.acosh,
                sympy.atanh, sympy.LambertW)
        for s in sols:
            if isinstance(s, bare) and s.args[0].has(y) and not (s.args[0].is_Mul and s.args[0].could_extract_minus_sign()):
                return s
            if s.is_Pow and s.exp.is_Rational and s.exp > 0 and s.base.has(y) and not s.could_extract_minus_sign():
                if s.exp.q % 2 == 1 and s.exp.p == 1:         # x^3 inverts to the real cube root
                    return sympy.real_root(s.base, s.exp.q)
                return s

        def key(s):
            try:
                v = complex(s.subs(y, at).evalf())
            except (TypeError, ValueError):
                return (2, 0.0)
            return (0 if abs(v.imag) < 1e-12 and v.real >= 0 else 1, abs(v))
        return min(sols, key=key)

    @staticmethod
    def _apply_undefined_mod(f, args, mod):
        kind = str(mod.args[0])
        if kind == "inverse":
            return Function(f"{f.name}^{{-1}}")(*args)
        n = mod.args[1]
        a = args[0]
        if isinstance(a, Symbol):
            return Derivative(f(*args), (a, n))
        u = Symbol("xi")
        return Subs(Derivative(f(u), (u, n)), u, a)

    def _stats_apply(self, name, args, notes):
        if name not in _STATS_OPS:
            return None
        name = _OP_ALIAS.get(name, name)
        import sympy.stats as st
        arg = args[0] if args else None
        if arg is None:
            return None
        if len(args) > 1 and name != "Cov" and all(getattr(a, "is_number", False) for a in args):
            arg = sympy.Matrix(args)                             # mean(a, b, c) of listed numbers
        arg = _vec(arg)
        if isinstance(arg, MatrixBase):                          # a data vector
            n = len(arg)
            if name in ("E", "mean"):
                notes.append(f"mean of the {n} values")
                return sum(arg) / n
            if name in ("Var", "var"):
                notes.append(f"sample variance of the {n} values (divides by n − 1)")
                return _var(arg)
            if name in ("SD", "std", "Std"):
                notes.append(f"sample standard deviation of the {n} values (n − 1)")
                return sympy.sqrt(_var(arg))
            if name == "Cov" and len(args) == 2 and isinstance(_vec(args[1]), MatrixBase):
                b = _vec(args[1])
                ma, mb = sum(arg) / n, sum(b) / n
                notes.append("sample covariance (n − 1)")
                return sum((x - ma) * (y - mb) for x, y in zip(arg, b)) / (n - 1)
            raise SemanticsError(f"{name}(…) of data needs a random variable",
                                 hint=r"for data use \bar{x}, \operatorname{Var}(x); for probabilities bind `X \sim …` first.")
        cond = None
        if isinstance(arg, Mid):
            arg, cond = arg.args
        if name in ("mean", "var"):
            return None
        arg = self._resolve_markers(arg, notes)
        if cond is not None:
            cond = self._resolve_markers(cond, notes)
        if not _is_rv(arg) and not (cond is not None and _is_rv(cond)):
            notes.append(f"{name}(…) left symbolic (bind X \\sim … to compute)")
            return Function(name)(*args)
        arg = arg.replace(lambda n: isinstance(n, Relational) and n.lhs.has(sympy.Abs) and n.rhs.is_number,
                          lambda n: self._unabs(n))
        if name in ("P", "Pr", "Prob") and cond is None:
            fast = self._cdf_prob(arg)
            if fast is not None:
                notes.append(f"P computed from the CDF of {fast[1]}")
                return fast[0]
        notes.append(f"{name} computed with sympy.stats")
        try:
            if name in ("P", "Pr", "Prob"):
                return st.P(arg, cond) if cond is not None else st.P(arg)
            if name in ("E", "Var") and cond is None:
                out = st.E(arg) if name == "E" else st.variance(arg)
                if isinstance(out, Basic) and out.has(Sum, Integral, sympy.Piecewise):
                    formula = self._moment_formula(arg, name)
                    if formula is not None:
                        notes.append(f"the standard formula for the {'mean' if name == 'E' else 'variance'} "
                                     f"of this distribution")
                        return formula
                return out
            if name == "E":
                return st.E(arg, cond)
            if name == "Var":
                return st.variance(arg)
            if name == "Cov":
                return st.covariance(arg, args[1])
            return st.std(arg)
        except Exception as exc:
            raise SemanticsError(f"{name} failed: {exc}") from None

    @staticmethod
    def _cdf_prob(arg):
        """``P(X > c)``, ``P(a < X < b)`` for a continuous X, from its CDF: fast even where
        sympy.stats would integrate the density (Student t, χ²). -> (value, X) or None."""
        import sympy.stats as st
        from sympy.stats.crv import SingleContinuousPSpace
        from sympy.stats.rv import RandomSymbol
        rels = list(arg.args) if isinstance(arg, sympy.And) else [arg]
        if not rels or not all(isinstance(r, (sympy.StrictLessThan, sympy.LessThan, sympy.StrictGreaterThan,
                                              sympy.GreaterThan)) for r in rels):
            return None
        X, lo, hi = None, None, None
        for r in rels:
            a, b = r.lhs, r.rhs
            less = isinstance(r, (sympy.StrictLessThan, sympy.LessThan))
            if isinstance(a, RandomSymbol) and not _is_rv(b):
                rv, bound, upper = a, b, less            # X < b: an upper bound
            elif isinstance(b, RandomSymbol) and not _is_rv(a):
                rv, bound, upper = b, a, not less        # a < X: a lower bound
            else:
                return None
            if X is not None and rv != X:
                return None
            X = rv
            if upper:
                if hi is not None:
                    return None
                hi = bound
            else:
                if lo is not None:
                    return None
                lo = bound
        if not isinstance(X.pspace, SingleContinuousPSpace):
            return None
        own = {s.name: s for s in X.pspace.distribution.free_symbols if isinstance(s, Symbol)}
        lo, hi = [b.xreplace({s: own[s.name] for s in b.free_symbols if getattr(s, "name", None) in own})
                  if b is not None else None for b in (lo, hi)]        # σ in the bounds is the distribution's σ
        F = st.cdf(X)
        val = (F(hi) if hi is not None else sympy.S.One) - (F(lo) if lo is not None else sympy.S.Zero)
        if isinstance(val, Basic) and val.has(sympy.Piecewise):
            val = sympy.piecewise_fold(val)
        # a scale written as σ^2 is a variance: σ > 0 while simplifying (√(σ²) = σ)
        pos = {q: Symbol(q.name, positive=True) for q in val.free_symbols
               if isinstance(q, Symbol) and not q.is_positive
               and any(w.base == q ** 2 and not w.exp.is_integer for w in val.atoms(Pow))}
        try:
            if sympy.count_ops(val) < 150:
                val = sympy.simplify(val.xreplace(pos)).xreplace({v: k for k, v in pos.items()})
        except Exception:
            pass
        return val, X

    @staticmethod
    def _moment_formula(X, which):
        """E[X] / Var(X) of a named distribution, for parameters sympy.stats cannot sum over."""
        from sympy.stats.rv import RandomSymbol
        if not isinstance(X, RandomSymbol):
            return None
        d = getattr(X.pspace, "distribution", None)
        table = {
            "BinomialDistribution": (lambda n, p, *_: n * p, lambda n, p, *_: n * p * (1 - p)),
            "PoissonDistribution": (lambda lam: lam, lambda lam: lam),
            "NormalDistribution": (lambda mu, s: mu, lambda mu, s: s ** 2),
            "ExponentialDistribution": (lambda r: 1 / r, lambda r: 1 / r ** 2),
            "UniformDistribution": (lambda a, b: (a + b) / 2, lambda a, b: (b - a) ** 2 / 12),
            "GeometricDistribution": (lambda p: 1 / p, lambda p: (1 - p) / p ** 2),
            "BernoulliDistribution": (lambda p, *_: p, lambda p, *_: p * (1 - p)),
        }
        entry = table.get(type(d).__name__)
        if entry is None:
            return None
        try:
            return sympy.sympify((entry[0] if which == "E" else entry[1])(*d.args))
        except TypeError:
            return None

    @staticmethod
    def _unabs(rel):
        """P(|X| < c) -> P(-c < X < c): sympy.stats cannot reduce |X| itself."""
        inner = rel.lhs.args[0] if isinstance(rel.lhs, sympy.Abs) else None
        if inner is None:
            return rel
        c = rel.rhs
        if isinstance(rel, (sympy.StrictLessThan, sympy.LessThan)):
            op = type(rel)
            return sympy.And(op(-c, inner), op(inner, c))
        op = sympy.StrictGreaterThan if isinstance(rel, sympy.StrictGreaterThan) else sympy.GreaterThan
        return sympy.Or(op(inner, c), op(-c, inner))

    @staticmethod
    def _call_python(name, fn, args, notes: list):
        try:
            out = fn(*args)
        except Exception:
            if all(getattr(a, "is_number", False) for a in args):
                try:
                    out = fn(*[complex(a) if not a.is_real else float(a) for a in args])
                except Exception as exc:
                    raise SemanticsError(f"Python function {name} failed: {exc}") from None
            else:
                raise SemanticsError(
                    f"Python function {name} could not be called with symbolic arguments",
                    hint=f"bind the arguments to numbers first, or define {name} as a "
                         f"statement (`{name}(x) = ...`) so it stays symbolic.") from None
        notes.append(f"called Python function {name}")
        v = coerce(out)
        if v is None:
            raise SemanticsError(f"Python function {name} returned {type(out).__name__}, not a number")
        return v

    # --- reading policies (e, i, I) and declared assumptions -----------------------------------
    def _policies(self, expr, dependent, notes, shown, exclude=frozenset()):
        bound = self.bound_variables(expr)
        I_ = Symbol("I")
        if I_ in expr.free_symbols and self._lookup("I") is None and I_ not in bound:
            mats = [m for m in expr.atoms(MatrixBase) if m.rows == m.cols]
            if mats:
                expr = self._subst(expr, {I_: sympy.eye(mats[0].rows)}, notes)
                notes.append(f"I read as the {mats[0].rows}×{mats[0].rows} identity")
        e = Symbol("e")
        if e in expr.free_symbols and self._lookup("e") is None and e not in bound and e not in dependent \
                and "e" not in exclude:
            expr = self._subst(expr, {e: sympy.E}, notes)
            notes.append("e read as Euler's number (bind e to override)")
        for nm in ("gamma_E", "EulerGamma"):             # \gamma_E: the Euler–Mascheroni constant
            g = Symbol(nm)
            if g in expr.free_symbols and self._lookup(nm) is None and g not in bound and nm not in exclude:
                expr = self._subst(expr, {g: sympy.EulerGamma}, notes)
                notes.append(f"{'γ_E' if nm == 'gamma_E' else nm} read as the Euler–Mascheroni constant 0.5772…")
        g = Symbol("gamma")
        if g in expr.free_symbols and self._lookup("gamma") is None and g not in bound and "gamma" not in exclude \
                and not (expr.free_symbols - {g}) and g not in dependent:
            notes.append(r"γ is a plain symbol here; for the Euler–Mascheroni constant write \gamma_E")
        i = Symbol("i")
        if i in expr.free_symbols and i not in bound and i not in dependent and self._lookup("i") is None \
                and "i" not in exclude:
            if self.complex_i:
                expr = self._subst(expr, {i: sympy.I}, notes)
                notes.append("i read as the imaginary unit (bind i, or pass --plain-i, to override)")
            else:
                notes.append("i is a plain symbol here (read it as the imaginary unit by removing --plain-i)")
        j = Symbol("j")
        if j in expr.free_symbols and j not in bound and self._lookup("j") is None and "j" not in exclude \
                and not expr.has(Sum, Product):
            notes.append("j is a plain symbol here; for the imaginary unit write i (or bind j = 1j)")
        for special, meaning in (("i", "the imaginary unit"), ("e", "Euler's number")):
            if Symbol(special) in shown:
                notes.append(f"{special} is bound to {interp.short(shown[Symbol(special)])} here, not {meaning}")
        return expr

    def _assume(self, expr, exclude):
        reg = self._reg("_openlab_assume")
        if not reg or not isinstance(expr, Basic):
            return expr
        rep = {s: Symbol(s.name, **reg[s.name]) for s in expr.free_symbols
               if isinstance(s, Symbol) and s.name in reg and s.name not in exclude
               and not s.assumptions0.get("real") and self._lookup(s.name) is None}
        return expr.xreplace(rep) if rep else expr

    # --- marker resolution (after names) ----------------------------------------------
    def _pick_var(self, expr: Basic, what: str) -> Symbol:
        free = sorted((s for s in expr.free_symbols if isinstance(s, Symbol)), key=lambda s: s.name)
        for cand in ("x", "t"):
            if Symbol(cand) in free:
                return Symbol(cand)
        if len(free) == 1:
            return free[0]
        if not free:
            return Symbol("x")
        raise SemanticsError(f"{what}: which variable? ({', '.join(s.name for s in free)})",
                             hint=r"name it: `\left. F \right|_{x=1}` or `\frac{d}{dx}`.")

    _LATE = (EvalSubs, EvalBetween, Nabla, Transform, Optim, Prime, Congruence)

    def _resolve_markers(self, expr: Basic, notes: list) -> Basic:
        for k in range(5):                   # a product waits for a bar or ∇ inside it: another pass
            if not isinstance(expr, Basic) or not expr.has(Marker):
                return expr
            new = self._resolve_markers_once(expr, notes, wait=k < 4)
            if new == expr:
                if k < 4 and new.has(Star, Plus, Dot, Times):
                    return self._resolve_markers_once(new, notes, wait=False)
                return new
            expr = new
        return expr

    def _resolve_markers_once(self, expr: Basic, notes: list, wait: bool = True) -> Basic:
        late = self._LATE + (Dot, Times, Star, Plus, Kron, MPow, Transposed, Adjoint, Norm)
        if isinstance(expr, Basic) and expr.has(IntRange):
            expr = expr.replace(IntRange, self._int_range)
        if isinstance(expr, Basic) and expr.has(TupleSet):
            expr = expr.replace(TupleSet, lambda head, conds: self._tuple_set(head, conds, notes))
        if isinstance(expr, Basic) and expr.has(SetImage):
            expr = expr.replace(SetImage, lambda lam, S_: self._set_image(lam, S_, notes))
        if isinstance(expr, Basic) and expr.has(SetSum):
            expr = expr.replace(SetSum, lambda op, lam, S_: self._set_sum(op, lam, S_, notes))

        def waits(*xs):
            return wait and any(isinstance(x, Basic) and x.has(*late) for x in xs)

        def dot(a, b):
            if waits(a, b):
                return Dot(a, b)
            a, b = _vec(a), _vec(b)
            if _is_vector(a) and _is_vector(b):
                if len(a) != len(b):
                    raise SemanticsError(f"a dot product needs vectors of the same length ({len(a)} and {len(b)})")
                return _as_column(a).dot(_as_column(b))
            return self._mat_mul(a, b) if isinstance(a, MatrixBase) and isinstance(b, MatrixBase) else a * b

        def times(a, b):
            if waits(a, b):
                return Times(a, b)
            a, b = _vec(a), _vec(b)
            if _is_vector(a) and _is_vector(b):
                if len(a) == len(b) == 3:
                    notes.append("× read as the cross product")
                    return _as_column(a).cross(_as_column(b))
                raise SemanticsError("a cross product needs two 3-vectors")
            return self._mat_mul(a, b) if isinstance(a, MatrixBase) and isinstance(b, MatrixBase) else a * b

        def star(a, b):
            if waits(a, b):
                return Star(a, b)
            if isinstance(a, MatrixBase) and isinstance(b, MatrixBase):
                return self._mat_mul(a, b)
            return a * b

        def plus(a, b):
            if waits(a, b):
                return Plus(a, b)
            if isinstance(a, MatrixBase) or isinstance(b, MatrixBase):
                return self._mat_add([a, b], notes)
            return a + b

        def kron(a, b):
            if isinstance(a, MatrixBase) and isinstance(b, MatrixBase):
                return sympy.kronecker_product(a, b)
            return Kron(a, b)

        def inner(*args):
            args = [_vec(a) for a in args]
            if len(args) == 2 and _is_vector(args[0]) and _is_vector(args[1]):
                return _as_column(args[0]).dot(_as_column(args[1]), conjugate_convention="left")
            if len(args) == 1:
                return self._stats_apply("E", [args[0]], notes)
            if not any(isinstance(a, MatrixBase) for a in args):
                notes.append("⟨…⟩ of scalars read as a vector")
                return ImmutableMatrix(args)
            return Inner(*args)

        def norm(a, kind):
            a = _vec(a)
            k = str(kind)
            if isinstance(a, MatrixBase):
                if k in ("F", "fro"):
                    return a.norm("fro")
                if k in ("infty", "inf", "oo"):
                    return a.norm(sympy.oo)
                if k.isdigit():
                    return a.norm(int(k))
                return a.norm()
            return sympy.Abs(a)

        def bars(a):
            if isinstance(a, sympy.ConditionSet):
                if _free_vars(a.condition) - {a.sym}:
                    return Bars(a)                         # S(k) = \#\{r : r ≤ 2^k − 1, …\}: per k
                a = self._condsets(a, notes)
            if isinstance(a, (TupleSet, SetImage)):
                return Bars(a)
            if isinstance(a, sympy.FiniteSet) and any(_free_vars(x) for x in a):
                return Bars(a)                             # {0, 2 mod p}: how many distinct is not known yet
            a = _vec(a)
            if isinstance(a, Tuple):
                a = ImmutableMatrix(list(a))
            if isinstance(a, sympy.FiniteSet):
                notes.append("|A| of a set read as its number of elements")
                return Integer(len(a))
            if isinstance(a, sympy.Set) and a.is_finite_set:
                notes.append("|A| of a set read as its number of elements")
                return Integer(len(list(a)))
            if isinstance(a, sympy.Set):
                raise SemanticsError("|…| of an infinite set is not a number")
            if isinstance(a, MatrixBase):
                if a.rows == a.cols and a.rows > 1:
                    notes.append("|A| read as the determinant")
                    return a.det()
                notes.append("|v| read as the Euclidean norm (the length of the vector)")
                return a.norm()
            if isinstance(a, Expr) and a.has(sympy.Float) and not _free_vars(a):
                try:                                 # |Φ(0.618…)|: hundreds of complex terms, numerically
                    return sympy.Abs(_fast.evaluate(a, hook=self._fast_hook))
                except (_fast.Unsupported, KeyError, ZeroDivisionError, OverflowError, ValueError, TypeError):
                    pass
            return sympy.Abs(a)

        def abseach(a):
            a = _vec(a)
            if isinstance(a, MatrixBase):
                return a.applyfunc(sympy.Abs)
            return sympy.Abs(a)

        def adjoint(a):
            if isinstance(a, MatrixBase):
                return a.H
            return sympy.conjugate(a)

        def transposed(a, conj):
            if isinstance(a, MatrixBase):
                return a.H if conj else a.T
            return a

        def evalsubs(f, var, val):
            f = f.doit() if isinstance(f, Basic) else f
            if isinstance(var, Tuple):
                if len(var) != len(val):
                    raise SemanticsError("the evaluation bar lists a different number of variables and values")
                return f.subs(dict(zip(var, val)))
            if var == sympy.S.Zero:
                if isinstance(val, Tuple):
                    vars_ = [Symbol(c) for c in "xyz"[:len(val)]]
                    notes.append(f"the point is read as ({', '.join(map(str, vars_))})")
                    return f.subs(dict(zip(vars_, val)))
                var = self._pick_var(f, "evaluation bar")
            return f.subs(var, val)

        def evalbetween(f, lo, hi, var):
            f = f.doit() if isinstance(f, Basic) else f
            if var == sympy.S.Zero:
                var = self._pick_var(f, "evaluation bar")
            return f.subs(var, hi) - f.subs(var, lo)

        def subset(a, b):
            if isinstance(a, sympy.Set) and isinstance(b, sympy.Set):
                return sympy.sympify(a.is_subset(b))
            return Subset(a, b)

        def mid(a, b):
            if getattr(a, "is_Integer", False) and getattr(b, "is_Integer", False):
                return sympy.sympify(b % a == 0)
            return Mid(a, b)

        def rel(op, a, b):
            o = str(op)
            if o in _SET_OPS:
                if _letter_sets(a, b):                 # {a, b, c} ∩ {b, c, d}: letters are distinct elements
                    A, B = list(a.args), set(b.args)
                    out = {"cup": A + [x for x in b.args if x not in A], "sqcup": A + [x for x in b.args if x not in A],
                           "cap": [x for x in A if x in B], "setminus": [x for x in A if x not in B],
                           "triangle": [x for x in A if x not in B] + [x for x in b.args if x not in set(A)]}[o]
                    notes.append("the letters in the sets are taken as distinct elements")
                    return sympy.FiniteSet(*out)
                if isinstance(a, sympy.Set) and isinstance(b, sympy.Set):
                    return _SET_OPS[o](a, b)
                return Rel(op, a, b)
            if o == "in":
                if isinstance(b, sympy.Set):
                    return sympy.Contains(a, b)
                if isinstance(b, MatrixBase):
                    return sympy.sympify(any(sympy.simplify(a - x) == 0 for x in b))
            return Rel(op, a, b)

        def mpow(m, e):
            return self._mat_pow(m, e, notes)

        for cls, fn in ((Star, star), (Plus, plus), (Dot, dot), (Times, times), (Kron, kron),
                        (Inner, inner), (Norm, norm), (Adjoint, adjoint), (Rel, rel), (AbsEach, abseach), (Bars, bars),
                        (Transposed, transposed), (MPow, mpow), (Subset, subset), (Mid, mid), (Rel, rel),
                        (CallPow, lambda a, e: Pow(a, e))):
            if expr.has(cls):
                expr = expr.replace(cls, fn)
        if expr.has(Iverson):
            def iv(c, want):
                if isinstance(c, sympy.logic.boolalg.BooleanAtom):
                    return Integer(1) if (c is sympy.true) == (want is sympy.true) else Integer(0)
                return Iverson(c, want)
            expr = expr.replace(Iverson, iv)
        if expr.has(Prime):
            expr = expr.replace(Prime, lambda a, n: self._prime(a, int(n), notes))
        if expr.has(Optim):                      # outermost first: an inner max over a waits for its index
            for n in set(_outer_optims(expr)):
                v_ = _optim_var(n.args[2])
                if v_ is not None and self._lookup(v_.name) is None and \
                        v_ in expr.xreplace({n: sympy.Dummy()}).free_symbols:
                    op_ = str(n.args[0])
                    raise SemanticsError(
                        f"{v_} is the index of \\{op_} but also appears outside it: \\{op_}_{{{v_}}} A({v_}) - B({v_}) "
                        f"takes the {op_} of the first term only",
                        hint=rf"put the whole expression in parentheses: `\{op_}_{{{v_} \in S}} (A({v_}) - B({v_}))`.")
            outer = {n: self._optim(str(n.args[0]), n.args[1], n.args[2], notes) for n in set(_outer_optims(expr))}
            expr = expr.xreplace(outer)
        if expr.has(Nabla):
            expr = expr.replace(Nabla, lambda kind, body: self._nabla(str(kind), _vec(body), notes))
        if expr.has(Congruence):
            expr = expr.replace(Congruence, lambda a, b, n: self._congruence(a, b, n, notes))
        if expr.has(Transform):
            expr = expr.replace(Transform, lambda *a: self._transform(*a, notes=notes))
        for cls, fn in ((EvalSubs, evalsubs), (EvalBetween, evalbetween)):
            if expr.has(cls):
                expr = expr.replace(cls, fn)
        return expr

    def _indep_var(self, stmt_exprs) -> Symbol:
        """The independent variable of y' in this statement: t if the statement mentions t
        (and y is not t), else x (or t when the unknown function is itself x)."""
        free = set()
        for e in stmt_exprs:
            if isinstance(e, Basic):
                free |= {s for s in e.free_symbols if isinstance(s, Symbol)}
                for d in e.atoms(Derivative):
                    return d.variables[0]
        dep = set()
        for e in stmt_exprs:
            if isinstance(e, Basic):
                dep |= self._differentiated(e)
        t, x = Symbol("t"), Symbol("x")
        if t in free and t not in dep:
            return t
        if x in dep:
            return t
        return x

    def _prime(self, a, n, notes):
        var = self._indep or Symbol("x")
        if isinstance(a, AppliedUndef):
            u = a.args[0]
            if isinstance(u, Symbol):
                return Derivative(a, (u, n))
        if isinstance(a, Symbol):
            notes.append(f"{a}{chr(39) * n} read as d{'^' + str(n) if n > 1 else ''}{a}/d{var}{'^' + str(n) if n > 1 else ''}")
            return Derivative(a, (var, n))
        var = self._indep if (self._indep is not None and self._indep in a.free_symbols) else self._pick_var(a, "prime")
        return sympy.diff(a, var, n)

    def _optim(self, op, body, spec, notes):
        from sympy.calculus.util import maximum, minimum
        data = self._optim_data(op, body, spec, notes)
        if data is not None:
            return data
        if spec == sympy.S.Zero:
            if isinstance(body, (SetImage, TupleSet)) or (isinstance(body, sympy.FiniteSet) and _free_vars(body)):
                return Optim(Symbol(op), body, spec)          # p₀(n) = \min\{p ∈ primes(3, n) : …\}: per n
            if isinstance(body, sympy.FiniteSet) and op in ("max", "min"):
                if not body:
                    raise SemanticsError(f"\\{op} of an empty set")
                return (max if op == "max" else min)(body)
            if isinstance(body, sympy.ConditionSet):
                if _free_vars(body.condition) - {body.sym}:
                    return Optim(Symbol(op), body, spec)          # σ(n) = \min\{k : …n…\}: per n
                found = self._min_search(op, body, notes)
                if found is not None:
                    return found
            if isinstance(body, sympy.Set):
                if op in ("max", "sup"):
                    return body.sup
                if op in ("min", "inf"):
                    return body.inf
            if isinstance(body, MatrixBase):
                return sympy.Max(*body) if op in ("max", "sup") else sympy.Min(*body)
            if op in ("limsup", "liminf"):
                notes.append(f"{op} is represented, not computed")
                return Optim(Symbol(op), body, spec)
            free = [s for s in body.free_symbols if isinstance(s, Symbol)]
            if len(free) == 1:
                spec = free[0]
            else:
                notes.append(f"{op} left symbolic (say the variable: \\{op}_{{x}})")
                return Optim(Symbol(op), body, spec)
        domain = sympy.S.Reals
        if isinstance(spec, Tuple):
            notes.append(f"{op} is represented, not computed")
            return Optim(Symbol(op), body, spec)
        if isinstance(spec, Rel):                         # a \in \{1, …, k\}: waits until k is a number
            return Optim(Symbol(op), body, spec)
        if isinstance(spec, sympy.Contains):
            var, domain = spec.args
            if _free_vars(domain):
                return Optim(Symbol(op), body, spec)
        elif isinstance(spec, (Relational, sympy.And)):
            parts = sympy.And.make_args(spec) if isinstance(spec, sympy.And) else (spec,)
            bare = None
            for c_ in parts:                               # the index: a bare side of every comparison
                sides = {x for x in (getattr(c_, "lhs", None), getattr(c_, "rhs", None)) if isinstance(x, Symbol)}
                bare = sides if bare is None else bare & sides
            cands = sorted(bare or (s for s in spec.free_symbols if isinstance(s, Symbol)), key=str)
            var = cands[0] if cands else None
            if var is not None and (_free_vars(spec) - {var}):
                return Optim(Symbol(op), body, spec)       # 2 ≤ j ≤ n − 1: waits until n is a number
            try:
                domain = spec.as_set()
            except Exception:
                domain = sympy.solveset(spec, var, sympy.S.Reals)
        else:
            var = spec
        if not isinstance(var, Symbol):
            raise UnsupportedError(f"cannot optimise over {var}")
        if isinstance(body, Basic) and body.has(Optim):    # inner aggregates that do not involve var: now
            inner = {n: self._optim(str(n.args[0]), n.args[1], n.args[2], notes)
                     for n in set(_outer_optims(body)) if var not in n.free_symbols}
            body = body.xreplace(inner) if inner else body
        others = {s_ for s_ in _free_vars(body) - {var} if self._subscript_of(s_, var) is None}   # a_k is k's
        if others and isinstance(domain, (sympy.FiniteSet, sympy.Range, sympy.Interval)) \
                and (body.has(Optim) or self._whole_number_only(body, var) or isinstance(domain, (sympy.FiniteSet, sympy.Range))):
            return Optim(Symbol(op), body, spec)           # B(k, a) with k an outer index: per k
        if op in ("max", "min", "argmax", "argmin", "sup", "inf") and (
                isinstance(domain, (sympy.FiniteSet, sympy.Range)) or domain is sympy.S.EmptySet
                or self._whole_number_only(body, var)):
            return self._optim_discrete(op, body, var, domain, notes)
        try:
            if op in ("max", "sup"):
                return maximum(body, var, domain)
            if op in ("min", "inf"):
                return minimum(body, var, domain)
            if op in ("argmax", "argmin"):
                crit = [c for c in sympy.solve(sympy.diff(body, var), var)
                        if c.is_real is not False and domain.contains(c) is not sympy.false]
                cands = crit + [b for b in (domain.inf, domain.sup) if b.is_finite and domain.contains(b) is sympy.true]
                vals = []
                for c in cands:
                    v = body.subs(var, c)
                    if v.is_finite is not False and v.is_real is not False:
                        try:
                            vals.append((float(v), c))
                        except TypeError:
                            vals.append((v, c))
                if not vals:
                    raise ValueError("no admissible candidates")
                best = max(vals, key=lambda t: t[0]) if op == "argmax" else min(vals, key=lambda t: t[0])
                return best[1]
        except Exception:
            lo_, hi_ = getattr(domain, "inf", None), getattr(domain, "sup", None)
            if op in ("max", "min", "argmax", "argmin") and lo_ is not None and hi_ is not None \
                    and lo_.is_Integer and hi_.is_Integer and hi_ - lo_ <= 200_000 \
                    and not {s_ for s_ in _free_vars(body) - {var} if self._subscript_of(s_, var) is None}:
                notes.append(f"the continuous {op} could not be computed; {var} taken over the whole numbers "
                             f"{lo_}…{hi_} instead")
                return self._optim_discrete(op, body, var, domain, notes)
            notes.append(f"{op} could not be computed in closed form; represented")
            return Optim(Symbol(op), body, spec)
        notes.append(f"{op} is represented, not computed")
        return Optim(Symbol(op), body, spec)

    _WHOLE_FUNCS = (sympy.Mod, sympy.floor, sympy.ceiling, sympy.factorial, sympy.factorial2, sympy.binomial)

    def _whole_number_only(self, body, var) -> bool:
        """Is ``body`` meaningful only at whole-number values of ``var`` -- a sequence term a_k, a
        remainder, a number-theory function, a factorial? Then a max/min over a range of ``var``
        runs over the whole numbers in it (a real-variable optimum would be meaningless)."""
        if not isinstance(body, Basic):
            return False
        names = {"isprime", "primepi", "prime", "nextprime", "prevprime", "totient", "sigma", "omega",
                 "Omega", "gcd", "lcm"} | set(self._reg("_openlab_recur"))
        for node in sympy.preorder_traversal(body):
            if isinstance(node, self._WHOLE_FUNCS) and var in node.free_symbols:
                return True
            if isinstance(node, sympy.Function) and type(node).__name__ in names and var in node.free_symbols:
                return True
            if isinstance(node, sympy.Indexed) and var in node.free_symbols:
                return True
            if isinstance(node, Apply) and any(isinstance(x, CallMod) and str(x.args[0]) == "iter"
                                               for x in node.args) and var in node.free_symbols:
                return True
            if isinstance(node, sympy.ConditionSet) and var in node.free_symbols:
                return True
            if isinstance(node, (Optim, Bars)) and var in node.free_symbols:
                return True
            if isinstance(node, (Sum, Product)) and any(var in l[1].free_symbols | l[2].free_symbols
                                                        for l in node.limits):
                return True                            # Σ_{i=0}^{j−1}: j counts terms
            if isinstance(node, Symbol) and self._subscript_of(node, var) is not None:
                return True
        return False

    @staticmethod
    def _subscript_of(sym, var):
        """``a_k`` / ``a_{k+1}`` with the index mentioning var -> (base, index expr), else None."""
        m = re.fullmatch(r"([A-Za-z]+)_(?:\{(.+)\}|(.+))", getattr(sym, "name", ""))
        if not m:
            return None
        idx = m.group(2) or m.group(3)
        if not re.search(rf"(?<![A-Za-z]){re.escape(var.name)}(?![A-Za-z])", idx):
            return None
        try:
            e = parse_expr(idx)[0]
        except OpenlabError:
            return None
        e = e.xreplace({s_: var for s_ in e.free_symbols if getattr(s_, "name", None) == var.name})
        return (m.group(1), e) if var in e.free_symbols else None

    def _term_at(self, body, var, k, notes):
        """The value of ``body`` at the whole number var = k: sequence terms a_k computed."""
        try:
            return _fast.evaluate(body, {var.name: int(k)}, self._fast_hook)
        except (_fast.Unsupported, KeyError, ZeroDivisionError, OverflowError, ValueError, TypeError):
            pass
        rep = {}
        for s_ in body.free_symbols:
            sub = self._subscript_of(s_, var) if isinstance(s_, Symbol) else None
            if sub:
                i = sympy.sympify(sub[1]).xreplace({var: k}).doit()
                if not i.is_Integer:
                    raise SemanticsError(f"{s_.name} at {var} = {k}: the index {i} is not a whole number")
                val = self._element(f"{sub[0]}_{int(i)}", notes)
                rep[s_] = val if val is not None else Symbol(f"{sub[0]}_{int(i)}")
        for ix in body.atoms(sympy.Indexed):
            if var in ix.free_symbols:
                i = ix.indices[0].xreplace({var: k}).doit()
                val = self._element(f"{ix.base.label}_{int(i)}", notes) if i.is_Integer else None
                if val is not None:
                    rep[ix] = val
        with sympy.evaluate(False):
            e = body.xreplace(rep).xreplace({var: k})
        v = self._settle(e, notes) if isinstance(e, Basic) else e
        if isinstance(v, Basic) and v.is_number and not v.is_Integer and _max_digits(v) <= 20 \
                and sympy.count_ops(v) < 60 and not v.has(sympy.log, sympy.exp):     # logcombine can hang
            return sympy.simplify(v)                     # never on 100-digit rationals inside logs (hangs)
        return v

    def _optim_discrete(self, op, body, var, domain, notes):
        """max / min / argmax / argmin over the whole numbers of a finite range, term by term."""
        if domain is sympy.S.EmptySet or (isinstance(domain, (sympy.FiniteSet, sympy.Range)) and len(domain) == 0) \
                or (isinstance(domain, sympy.Interval) and domain.inf.is_finite and domain.sup.is_finite
                    and sympy.floor(domain.sup) < sympy.ceiling(domain.inf)):
            if op in ("max", "sup"):
                notes.append(f"the range of {var} is empty: the max of an empty set is −∞")
                return sympy.S.NegativeInfinity
            if op in ("min", "inf"):
                notes.append(f"the range of {var} is empty: the min of an empty set is +∞")
                return sympy.oo
            raise SemanticsError(f"\\{op}: the range of {var} is empty")
        if isinstance(domain, (sympy.FiniteSet, sympy.Range)):
            if isinstance(domain, sympy.Range) and not domain.is_finite_set:
                raise SemanticsError(f"\\{op} over an infinite range cannot be computed term by term")
            points = list(domain)
            where = f"{var} in {interp.pyform(domain)}"
        else:
            lo, hi = getattr(domain, "inf", None), getattr(domain, "sup", None)
            if lo is None or not (lo.is_finite and hi.is_finite):
                raise SemanticsError(f"\\{op} over whole numbers {var} needs finite bounds",
                                     hint=rf"write `\\{op}_{{1 \\le {var} \\le N}} …`.")
            a, b = int(sympy.ceiling(lo)), int(sympy.floor(hi))
            if isinstance(domain, sympy.Interval):             # only the ends can be open: no per-point test
                a += 1 if (domain.left_open and a == lo) else 0
                b -= 1 if (domain.right_open and b == hi) else 0
                points = list(range(a, b + 1))
            else:
                points = [k for k in range(a, b + 1) if domain.contains(Integer(k)) is sympy.true]
            where = f"the whole numbers {var} = {a}…{b}"
        if not points:
            raise SemanticsError(f"\\{op}: no whole number {var} in {interp.pyform(domain)}")
        if len(points) > 20_000_000:
            raise SemanticsError(f"\\{op} over {len(points)} values is too many to check one by one")
        vals = []
        scratch: list = []                       # per-term notes ("a_5 computed by iterating") stay out
        try:
            fast = _fast.compile_expr(self._index_bound_lists(body, var), {var.name}, self._fast_hook)
        except _fast.Unsupported:
            fast = None
        raw_ok = fast is not None
        raws = []
        if fast is not None:                             # certified values (exact, or 60+-digit): compare those,
            try:                                          # then compute only the winner exactly
                for k in points:
                    rv = _fast.certified(fast, {var.name: int(k)})
                    if isinstance(rv, bool) or (hasattr(rv, "imag") and not isinstance(rv, (int, Fraction)) and rv.imag):
                        raise TypeError
                    raws.append((rv, k))
            except (_fast.Unsupported, KeyError, ZeroDivisionError, OverflowError, ValueError, TypeError,
                    AttributeError):
                raw_ok = False
        if raw_ok and raws:
            pick = max if op in ("max", "sup", "argmax") else min
            best_raw = pick(r_ for r_, _ in raws)
            at = [k for r_, k in raws if r_ == best_raw]
            notes.append(f"{op} over {where} ({len(points)} values, each computed)")
            if op.startswith("arg"):
                if len(at) > 1:
                    notes.append(f"the {op[3:]} is attained at {', '.join(map(str, at[:6]))}"
                                 f"{' …' if len(at) > 6 else ''}; the smallest is given")
                return Integer(at[0]) if isinstance(at[0], int) else at[0]
            try:
                return _fast.to_sympy(best_raw)
            except _fast.Unsupported:
                return self._term_at(body, var, Integer(at[0]) if isinstance(at[0], int) else at[0], scratch)
        for k in points:
            v = None
            if v is None:
                v = self._term_at(body, var, Integer(k) if not isinstance(k, Basic) else k, scratch)
                if isinstance(v, Expr) and v.has(Integral) and not _free_vars(v):
                    num = self._reliable_number(v)          # 27.1/∫…: compare as a checked decimal
                    if num is None:
                        raise SemanticsError(f"\\{op}: the value at {var} = {k} has an integral with no reliable "
                                             f"numeric value")
                    v = num
            if not (isinstance(v, Basic) and v.is_number and v.is_real is not False):
                raise SemanticsError(f"\\{op}: the value at {var} = {k} is not a number ({interp.pyform(v)})")
            vals.append((v, k))
        pick = max if op in ("max", "sup", "argmax") else min
        best = pick(v for v, _ in vals)
        at = [k for v, k in vals if v == best]
        notes.append(f"{op} over {where} ({len(points)} values, each computed)")
        if op.startswith("arg"):
            if len(at) > 1:
                notes.append(f"the {op[3:]} is attained at {', '.join(map(str, at[:6]))}"
                             f"{' …' if len(at) > 6 else ''}; the smallest is given")
            return Integer(at[0]) if isinstance(at[0], int) else at[0]
        return best

    def _optim_data(self, op, body, spec, notes):
        """``\\max_i x_i`` / ``\\max_{1 \\le i \\le n} x_i`` over a bound data vector."""
        if spec == sympy.S.Zero or not isinstance(body, Basic):
            return None
        var = spec if isinstance(spec, Symbol) else next(
            (s for s in getattr(spec, "free_symbols", ()) if isinstance(s, Symbol) and len(s.name) == 1), None)
        if var is None:
            return None
        suffix = "_" + var.name
        bases = [s for s in body.free_symbols if isinstance(s, Symbol) and s.name.endswith(suffix)]
        vecs = {b: _vec(coerce(self._lookup(b.name[:-len(suffix)]))) for b in bases}
        if not bases or not all(isinstance(v, MatrixBase) for v in vecs.values()):
            return None
        n = min(len(v) for v in vecs.values())
        lo, hi = 1, n
        if isinstance(spec, sympy.Contains) and isinstance(spec.args[1], (sympy.FiniteSet, sympy.Range)):
            ks = [int(k) for k in spec.args[1]]
            if any(not 1 <= k <= n for k in ks):
                raise SemanticsError(f"{op}: an index in {interp.pyform(spec.args[1])} is outside 1..{n}")
            vals = [(body.xreplace({b: vecs[b][k - 1] for b in bases}), k) for k in ks]
            pick = max if op in ("max", "sup", "argmax") else min
            best = pick(vals, key=lambda t: float(t[0]))
            notes.append(f"{op} over {var} ∈ {interp.pyform(spec.args[1])} of the data")
            return best[1] if op.startswith("arg") else best[0]
        if isinstance(spec, (Relational, sympy.And)):
            rng = sympy.solveset(spec, var, sympy.S.Reals)
            if isinstance(rng, sympy.Interval):
                lo = max(lo, int(sympy.ceiling(rng.inf))) if rng.inf.is_finite else lo
                hi = min(hi, int(sympy.floor(rng.sup))) if rng.sup.is_finite else hi
        vals = [(body.xreplace({b: vecs[b][k - 1] for b in bases}), k) for k in range(lo, hi + 1)]
        pick = max if op in ("max", "sup", "argmax") else min
        best = pick(vals, key=lambda t: float(t[0]))
        notes.append(f"{op} over {var} = {lo}..{hi} of the data")
        return best[1] if op.startswith("arg") else best[0]

    def _nabla(self, kind, body, notes):
        xyz = [Symbol(c) for c in "xyz"]
        if kind in ("grad", "laplacian") and not isinstance(body, MatrixBase):
            free = body.free_symbols
            vars_ = [v for v in xyz if v in free] or sorted((s for s in free if isinstance(s, Symbol)), key=lambda s: s.name)
            if kind == "grad":
                return ImmutableMatrix([sympy.diff(body, v) for v in vars_])
            return sum(sympy.diff(body, v, 2) for v in vars_)
        if not _is_vector(body):
            notes.append(f"{kind} needs a vector field; represented")
            return Nabla(Symbol(kind), body)
        comps = list(_as_column(body))
        vars_ = xyz[:len(comps)]
        if kind in ("div", "grad"):
            return sum(sympy.diff(c, v) for c, v in zip(comps, vars_))
        if kind == "laplacian":
            return ImmutableMatrix([sum(sympy.diff(c, v, 2) for v in vars_) for c in comps])
        if len(comps) != 3:
            raise UnsupportedError("curl needs a 3-component field")
        Fx, Fy, Fz = comps
        x, y, z = xyz
        return ImmutableMatrix([sympy.diff(Fz, y) - sympy.diff(Fy, z),
                                sympy.diff(Fx, z) - sympy.diff(Fz, x),
                                sympy.diff(Fy, x) - sympy.diff(Fx, y)])

    def _congruence(self, a, b, n, notes):
        from .proof import divides
        if a.is_number and b.is_number and n.is_number:
            d = sympy.sympify(a - b).doit()          # \binom{6}{3} - 20 ≡ 0 (mod 3): evaluate first
            n = sympy.sympify(n).doit()
            if d.is_Integer and n.is_Integer:
                return sympy.sympify(d % n == 0)
        if isinstance(n, Expr) and not n.is_number:     # mod q, mod 1 + x^2: integer divisibility
            verdict, why = divides(n, a - b)
            if verdict is not None:
                notes.append(why)
                return sympy.sympify(verdict)
        unknowns = [s for s in (a.free_symbols | b.free_symbols) if isinstance(s, Symbol)]
        if len(unknowns) == 1 and n.is_Integer and int(n) <= 10 ** 5:
            x = unknowns[0]
            res = [r for r in range(int(n)) if ((a - b).subs(x, r) % n) == 0]
            if not res:
                return sympy.S.EmptySet
            k = Symbol("k", integer=True)
            notes.append(f"solved the congruence for {x}")
            return sympy.Union(*[sympy.ImageSet(Lambda(k, r + n * k), sympy.S.Integers) for r in res])
        return Congruence(a, b, n)

    def _transform(self, kind, f, inv, var=None, *, notes):
        from sympy import (fourier_transform, inverse_fourier_transform,
                           inverse_laplace_transform, laplace_transform)
        kind = str(kind)
        free = {s for s in f.free_symbols if isinstance(s, Symbol)}
        t, s, x, w = Symbol("t"), Symbol("s"), Symbol("x"), Symbol("omega")
        if kind == "L":
            if inv is sympy.true:
                src = s if s in free else (next(iter(free)) if len(free) == 1 else s)
                out = inverse_laplace_transform(f, src, var or t)
                notes.append("inverse Laplace transform (for t > 0)")
                tv = var or t
                return out.replace(lambda n: isinstance(n, sympy.Heaviside) and n.args[0] == tv,
                                   lambda n: sympy.S.One)
            src = t if t in free else (x if x in free else (next(iter(free)) if len(free) == 1 else t))
            notes.append(f"Laplace transform from {src} to {var or s}")
            return laplace_transform(f, src, var or s, noconds=True)
        k = Symbol("k")
        if inv is sympy.true:
            src = w if w in free else (next(iter(free)) if len(free) == 1 else w)
            notes.append("inverse Fourier transform, f(x) = (1/2π)∫F(ω)e^{iωx}dω")
            return inverse_fourier_transform(f.xreplace({src: 2 * sympy.pi * k}), k, var or x)
        src = x if x in free else (t if t in free else (next(iter(free)) if len(free) == 1 else x))
        notes.append("Fourier transform, F(ω) = ∫f(x)e^{-iωx}dx")
        return fourier_transform(f, src, (var or w) / (2 * sympy.pi))

    # --- evaluation ----------------------------------------------------------------
    def _functionalize(self, expr: Basic, notes: list) -> Basic:
        """A differentiated unbound name is a function of the variables it is differentiated
        by: ``dy/dx`` makes ``y`` into ``y(x)`` everywhere in the statement."""
        vars_by: dict = {}
        for d in expr.atoms(Derivative):
            base = d.expr
            if isinstance(base, Symbol) and (self._lookup(base.name) is None or base in self._dep):
                lst = vars_by.setdefault(base, [])
                for v in d.variables:
                    if v not in lst:
                        lst.append(v)
        if not vars_by:
            return expr
        conv = {b: Function(b.name)(*vs) for b, vs in vars_by.items()}
        expr = expr.xreplace(conv)
        notes.append(", ".join(f"{k} read as {v}" for k, v in conv.items()))
        return expr

    def _det(self, m, notes):
        if isinstance(m, MatrixBase):
            if m.rows != m.cols:
                raise SemanticsError(f"a determinant needs a square matrix (this one is {m.rows}×{m.cols})")
            return m.det()
        if isinstance(m, Basic) and not m.free_symbols:
            notes.append("the determinant of a number is the number")
            return m
        return Det(m)

    @staticmethod
    def _check_bounds(val):
        for node in val.atoms(Sum, Product):
            for _, lo, hi in node.limits:
                for b in (lo, hi):
                    if b.is_number and b.is_finite and not b.is_integer:
                        raise SemanticsError(f"the bounds of a sum must be whole numbers (got {b})")

    @staticmethod
    def _int_range(a, c, step):
        """{a, …, c} with whole-number ends -> Range; symbolic ends wait."""
        a, c, step = [sympy.sympify(x).doit() for x in (a, c, step)]
        if not all(x.is_number for x in (a, c, step)):
            return IntRange(a, c, step)
        if not all(x.is_Integer for x in (a, c, step)) or step == 0:
            raise SemanticsError(f"{{{interp.pyform(a)}, …, {interp.pyform(c)}}} needs whole-number ends and step")
        return sympy.Range(a, c + (1 if step > 0 else -1), step)

    def _tuple_set(self, head, conds, notes):
        """{(k, a) : 1 ≤ k ≤ 40, 1 ≤ a ≤ 40, 3^a < 2^k}: each variable bounded by the conditions, the box listed."""
        from itertools import product as _product
        cond = sympy.And(*conds)
        if _free_vars(cond) - set(head):
            return TupleSet(head, conds)                  # waits for the other letters
        ranges = []
        for v in head:
            lo, hi = self._int_bounds(v, sympy.S.Integers, cond)
            if isinstance(lo, sympy.Set) or lo is None or hi is None:
                raise SemanticsError(f"a set of tuples needs whole-number bounds for {v} (`1 \\le {v} \\le N`)")
            ranges.append(range(lo, hi + 1))
        total = 1
        for r_ in ranges:
            total *= len(r_)
        if total > 2_000_000:
            raise SemanticsError(f"a set of {total} tuples is too many to list")
        names = [v.name for v in head]
        try:
            f = _fast.compile_expr(cond, set(names), self._fast_hook)
        except _fast.Unsupported:
            f = None
        out, scratch = [], []
        for pt in _product(*ranges):
            ok = None
            if f is not None:
                try:
                    ok = bool(f(dict(zip(names, pt))))
                except (_fast.Unsupported, ZeroDivisionError, OverflowError, ValueError, TypeError, KeyError):
                    ok = None
            if ok is None:
                with sympy.evaluate(False):
                    c_ = cond.xreplace({v: Integer(x) for v, x in zip(head, pt)})
                v_ = self._settle(c_, scratch)
                if v_ not in (sympy.true, sympy.false):
                    raise SemanticsError(f"the condition at {tuple(pt)} did not decide")
                ok = v_ is sympy.true
            if ok:
                out.append(Tuple(*[Integer(x) for x in pt]))
        notes.append(f"the set listed by checking the {total} tuples in the box")
        return sympy.FiniteSet(*out)

    def _set_image(self, lam, S_, notes):
        """{f(p) : p ∈ S, conditions}: the values f(p) for the elements of a finite S that pass."""
        if isinstance(S_, sympy.Function) and not isinstance(S_, Marker):
            S_ = S_.func(*S_.args)                   # primes(3, 6) built with evaluation off: now a set
        if isinstance(S_, IntRange):
            S_ = self._int_range(*S_.args)
        if isinstance(S_, sympy.ConditionSet):
            S_ = self._condsets(S_, notes)
        if not (isinstance(S_, sympy.FiniteSet) or (isinstance(S_, sympy.Range) and S_.is_finite_set)):
            return SetImage(lam, S_)
        if _free_vars(lam) or _free_vars(S_):            # {h mod p : h ∈ {0, 2}} with p not yet a number: wait
            return SetImage(lam, S_)
        scratch: list = []
        out = []
        v0 = lam.variables[0]
        for x in S_:
            with sympy.evaluate(False):                  # r(n) = 0 must not be summed numerically first
                head, *conds = lam.expr.xreplace({v0: x})
            head = self._settle(head, scratch)
            vals = [self._settle(c, scratch) for c in conds]
            if any(v not in (sympy.true, sympy.false) for v in vals):
                raise SemanticsError(f"the condition at {lam.variables[0]} = {x} did not decide")
            if all(v is sympy.true for v in vals):
                out.append(head)
        notes.append(f"the set listed by checking the {len(S_)} elements of the domain")
        return sympy.FiniteSet(*out)

    def _set_sum(self, op, lam, S_, notes):
        """Σ / ∏ over the elements of a finite set, term by term."""
        if isinstance(S_, sympy.Function) and not isinstance(S_, Marker):
            S_ = S_.func(*S_.args)
        if isinstance(S_, IntRange):
            S_ = self._int_range(*S_.args)
        if isinstance(S_, sympy.ConditionSet):
            S_ = self._condsets(S_, notes)
        if not (isinstance(S_, sympy.FiniteSet) or (isinstance(S_, sympy.Range) and S_.is_finite_set)):
            return SetSum(op, lam, S_)
        if len(S_) > 1_000_000:
            raise SemanticsError(f"a {op} over {len(S_)} elements is too long to compute term by term")
        scratch: list = []
        v0 = lam.variables[0]
        terms = []
        for x in S_:
            with sympy.evaluate(False):
                t_ = lam.expr.xreplace({v0: x})
            terms.append(self._settle(t_, scratch))
        notes.append(f"{'summed' if str(op) == 'sum' else 'multiplied'} over the {len(terms)} elements of the set")
        return sympy.Add(*terms) if str(op) == "sum" else sympy.Mul(*terms)

    def _settle(self, e, notes):
        """Finish a substituted term: calls, markers, then evaluation (a sum's term, a set's condition)."""
        if not isinstance(e, Basic):
            return e
        if e.has(Apply, AppliedUndef):
            e = self._apply_all(e, notes)
        if e.has(Marker):
            e = self._resolve_markers(e, notes)
        if e.has(sympy.Indexed):
            e = self._indexed_terms(e, notes)
        if e.has(Sum, Product):
            e = _finite_sums(e, settle=lambda x: self._settle(x, []), hook=self._fast_hook)
        try:
            e = e.doit()
        except Exception:
            pass
        if isinstance(e, Basic) and e.has(sympy.ConditionSet):
            e = self._condsets(e, notes)
        return e

    def _is_whole(self, var, base, cond) -> bool:
        if isinstance(base, (sympy.Range, sympy.FiniteSet)) or base in (sympy.S.Integers, sympy.S.Naturals,
                                                                          sympy.S.Naturals0):
            return True
        return self._whole_number_only(cond, var)

    def _int_bounds(self, var, base, cond):
        """(lo, hi) for the whole numbers of the set (None = unbounded), from the base set and the
        comparisons of the condition that involve only the variable."""
        dom = sympy.S.Reals
        if base == sympy.S.Naturals:
            dom = sympy.Interval(1, sympy.oo)
        elif base == sympy.S.Naturals0:
            dom = sympy.Interval(0, sympy.oo)
        x = Symbol(var.name, real=True)
        parts = sympy.And.make_args(cond) if isinstance(cond, sympy.And) else (cond,)
        for c in parts:
            if isinstance(c, sympy.core.relational.Relational) and not isinstance(c, (sympy.Eq, sympy.Ne)) \
                    and c.free_symbols == {var} and (c.lhs - c.rhs).is_polynomial(var) \
                    and not (c.lhs - c.rhs).has(sympy.Function):       # bounds only; the rest is checked per element
                try:
                    dom = dom.intersect(sympy.solveset(c.xreplace({var: x}), x, sympy.S.Reals))
                except Exception:
                    pass
        if isinstance(base, (sympy.Range, sympy.FiniteSet)):
            return base, None
        lo = dom.inf if dom is not sympy.S.EmptySet else None
        hi = dom.sup if dom is not sympy.S.EmptySet else None
        lo = int(sympy.ceiling(lo)) if lo is not None and lo.is_finite else None
        hi = int(sympy.floor(hi)) if hi is not None and hi.is_finite else None
        return lo, hi

    def _holds_at(self, cond, var, k, scratch):
        key = (cond, var)
        cache = self.__dict__.setdefault("_fast_conds", {})
        if key not in cache:
            try:
                cache[key] = _fast.compile_expr(cond, {var.name}, self._fast_hook)
            except _fast.Unsupported:
                cache[key] = None
        if cache[key] is not None:
            try:
                return bool(cache[key]({var.name: int(k)}))
            except (_fast.Unsupported, ZeroDivisionError, OverflowError, ValueError, TypeError):
                pass
        with sympy.evaluate(False):              # Eq(Σ…, 0) would try a numeric sum while rebuilding
            c = cond.xreplace({var: Integer(k)})
        v = self._settle(c, scratch)
        if v in (sympy.true, sympy.false):
            return v is sympy.true
        raise SemanticsError(f"the condition at {var} = {k} did not decide ({interp.pyform(v)})")

    def _condsets(self, expr, notes):
        """{n ∈ ℤ : 4 ≤ n ≤ 1000, r(n) = 0}: a finite set of whole numbers, listed by checking each."""
        def listed(cs):
            var, cond, base = cs.sym, cs.condition, cs.base_set
            if (_free_vars(cond) - {var}) or not self._is_whole(var, base, cond):
                return cs
            lo, hi = self._int_bounds(var, base, cond)
            if hi is None and not isinstance(lo, sympy.Set):
                return cs                                  # infinite: only a \min can search it
            pts = list(lo) if isinstance(lo, sympy.Set) else (range(lo, hi + 1) if lo is not None else None)
            if pts is None or len(pts) > 1_000_000:
                return cs
            scratch: list = []
            out = [k for k in pts if self._holds_at(cond, var, k, scratch)]
            notes.append(f"the set listed by checking {var} = each of {len(pts)} whole numbers")
            return sympy.FiniteSet(*out)
        return expr.replace(lambda n: isinstance(n, sympy.ConditionSet), listed)

    def _min_search(self, op, cs, notes, cap=100_000):
        r"""\min\{k ≥ k0 : P(k)\}: the first whole number from k0 up that satisfies P (up to a cap)."""
        var, cond, base = cs.sym, cs.condition, cs.base_set
        lo, hi = self._int_bounds(var, base, cond)
        if isinstance(lo, sympy.Set) or hi is not None:
            listed = self._condsets(cs, notes)
            if isinstance(listed, sympy.FiniteSet):
                if not listed:
                    raise SemanticsError(f"\\{op} of an empty set: no {var} in the range satisfies the condition")
                return (min if op in ("min", "inf") else max)(listed)
            return None
        if op not in ("min", "inf") or lo is None:
            return None
        scratch: list = []
        from time import monotonic
        t0 = monotonic()
        for k in range(lo, lo + cap):
            if self._holds_at(cond, var, k, scratch):
                return Integer(k)
            if monotonic() - t0 > self.budget:
                raise SemanticsError(f"\\min: no {var} from {lo} to {k} satisfies the condition "
                                     f"(stopped after {self.budget:g} s)",
                                     hint=rf"give an upper bound, `\\min\\{{{lo} \\le {var} \\le N : …\\}}`.")
        raise SemanticsError(f"\\min: no {var} from {lo} to {lo + cap - 1} satisfies the condition",
                             hint=rf"give an explicit upper bound, `\\min\\{{{lo} \\le {var} \\le N : …\\}}`.")

    def _doit_failed(self, val, exc, notes):
        """SymPy raised inside doit (e.g. '>' between Poly and int in the integrator): a numeric value
        when the expression is a number, otherwise the expression left as written."""
        free = {s_ for s_ in val.free_symbols if isinstance(s_, Symbol)} - self.bound_variables(val)
        if not free:
            num = self._reliable_number(val)
            if num is not None:
                notes.append(f"no exact result (SymPy failed: {type(exc).__name__}); computed numerically "
                             f"(two quadratures agree)")
                return num
        notes.append(f"no closed form (SymPy failed: {type(exc).__name__}); left unevaluated")
        return val

    def _reliable_number(self, val):
        """A decimal for a closed expression only when it is trustworthy: every definite integral in it
        is computed by two quadratures that must agree (the second after t = e^u, which tames the
        slowly decaying logarithmic integrands of number theory); other parts by evalf at full precision."""
        digits = max(self.precision, 15)
        rep = {}
        for I in val.atoms(Integral):
            if len(I.limits) != 1 or len(I.limits[0]) != 3:
                return None
            x, a, b = I.limits[0]
            try:
                f = sympy.lambdify(x, I.function, "mpmath")
                a_, b_ = [mpmath.mpf(sympy.N(v, digits + 5)) if v.is_finite else (mpmath.inf if v > 0 else mpmath.ninf)
                          for v in (a, b)]
            except Exception:
                return None

            def pair(g, lo, hi):
                """g integrated whole and split into pieces: the value when the two agree."""
                pts = [lo] + ([lo + 10, lo + 100, lo + 1000] if hi == mpmath.inf else
                              [lo + (hi - lo) * k / 4 for k in (1, 2, 3)]) + [hi]
                try:
                    with mpmath.workdps(digits + 10):
                        w1 = mpmath.quad(g, [lo, hi], maxdegree=10)
                        w2 = mpmath.quad(g, pts, maxdegree=10)
                except Exception:
                    return None
                if mpmath.isfinite(w1) and mpmath.isfinite(w2) and abs(w1 - w2) <= max(1, abs(w2)) * mpmath.mpf(10) ** -10:
                    return w2
                return None
            best = None
            if a_ > 0 and a_ != mpmath.inf:                    # t = e^u first: log-type integrands
                best = pair(lambda u: f(mpmath.exp(u)) * mpmath.exp(u), mpmath.log(a_),
                            mpmath.log(b_) if b_ != mpmath.inf else mpmath.inf)
            if best is None and a_ != mpmath.ninf:
                best = pair(f, a_, b_)
            if best is None:
                return None
            rep[I] = sympy.Float(mpmath.nstr(best, 15), 15)
        out = val.xreplace(rep) if rep else val
        try:
            num = out.evalf(digits)
        except Exception:
            return None
        if not (isinstance(num, Expr) and num.is_number) or num.has(Integral, Sum, Product, Limit):
            return None
        if isinstance(num, sympy.Float) and num._prec < 30:          # fewer than ~9 trusted digits
            return None
        return num

    def _indexed_terms(self, expr, notes):
        """a[10] left by a summed-out a_k: the term of the sequence (or data vector) a."""
        rep, scratch = {}, []
        for ix in expr.atoms(sympy.Indexed):
            if len(ix.indices) == 1 and ix.indices[0].is_Integer:
                try:
                    v = self._element(f"{ix.base.label}_{int(ix.indices[0])}", scratch)
                except OpenlabError:
                    v = None
                if v is not None:
                    rep[ix] = v
        if not rep:
            return expr
        base = sorted({str(ix.base.label) for ix in rep})
        notes.append(f"the terms of {', '.join(base)} computed")
        out = expr.xreplace(rep)
        return out.doit() if isinstance(out, Basic) else out

    def _doit(self, val, notes):
        """``val.doit()`` within the symbolic budget; on timeout, a numeric value if possible."""
        if isinstance(val, Basic) and val.has(Sum) and val.has(sympy.binomial):
            val = _binomial_sums(val)
        if self.keep_sums and isinstance(val, Basic) and val.has(Sum, Product):
            return val.doit(deep=False) if not isinstance(val, (Sum, Product)) else val
        if isinstance(val, Expr) and val.has(sympy.factorial, sympy.gamma) and not val.free_symbols:
            val = self._factorial_ratio(val)
        if isinstance(val, Expr) and val.has(sympy.factorial, sympy.gamma) and not val.free_symbols:
            tok0 = None
            try:
                with time_limit(min(3.0, self.budget)) as tok0:
                    simp = sympy.gammasimp(sympy.combsimp(val))   # (10^100)!/(10^100 - 1)! = 10^100
                if not simp.has(sympy.factorial, sympy.gamma):
                    val = simp
            except _Timeout as t:
                if t.token is not tok0:
                    raise
            except Exception:
                pass
        if isinstance(val, Expr) and val.has(sympy.Float) and not _free_vars(val) and sympy.count_ops(val) > 40:
            try:                                     # a long decimal expression (|Φ(0.618)| over 303 primes)
                return _fast.evaluate(val, hook=self._fast_hook)
            except (_fast.Unsupported, KeyError, ZeroDivisionError, OverflowError, ValueError, TypeError):
                pass
        if isinstance(val, Basic) and val.has(sympy.ConditionSet):
            val = self._condsets(val, notes)         # {n ∈ ℤ : 4 ≤ n ≤ N, …}: listed, not simplified
        if isinstance(val, Basic) and val.has(Sum, Product):
            val = val.replace(                       # Σ (ij + 3) mod 7 over a million terms: vectorised, exact
                lambda n: isinstance(n, (Sum, Product)) and not _free_vars(n)
                and _integer_body(n.function, {k for k, _, _ in n.limits}),
                lambda n: _vector_or_keep(n))
        if isinstance(val, Basic) and val.has(Sum, Product):
            val = val.replace(
                lambda n: isinstance(n, (Sum, Product)) and not _free_vars(n)
                and n.function.has(Marker, sympy.ConditionSet, AppliedUndef, Pick),
                lambda n: _finite_sums(n, settle=lambda e: self._settle(e, []), hook=self._fast_hook))
            val = val.replace(                       # Σ r(2m)², Σ isprime(k)…: exact, term by term
                lambda n: isinstance(n, (Sum, Product)) and not _free_vars(n) and self._opaque_terms(n.function),
                self._fast_or_keep)
            val = val.replace(                       # Σ_{p ≤ 20, p prime} e^{2πipx}: written out, so ∫ dx sees terms
                lambda n: isinstance(n, (Sum, Product)) and _free_vars(n) and self._opaque_terms(n.function)
                and self._small_range(n),
                self._written_out)
        frozen = {}
        if isinstance(val, Basic) and val.has(Sum, Product):     # Σ_{p ≤ 10⁴, p prime} e^{2πipx}: kept as written
            for n in val.atoms(Sum, Product):
                if _free_vars(n) and self._opaque_terms(n.function) and not self._small_range(n):
                    frozen[n] = sympy.Dummy("frozen")
            if frozen:
                val = val.xreplace(frozen)
        tok = None
        try:
            with time_limit(self.budget) as tok:
                try:
                    out = val.doit()
                    if frozen:
                        out = out.xreplace({d: n for n, d in frozen.items()})
                except (TypeError, ValueError, AttributeError, NotImplementedError, IndexError,
                        RecursionError) as exc:       # a SymPy integrator bug: not the user's error
                    if frozen:
                        val = val.xreplace({d: n for n, d in frozen.items()})
                        frozen = {}
                    out = self._doit_failed(val, exc, notes)
                if isinstance(out, Basic) and out.has(sympy.Indexed):
                    out = self._indexed_terms(out, notes)
                if isinstance(out, Basic) and out.has(Sum, Product):
                    out = _finite_sums(out, settle=lambda e: self._settle(e, []), hook=self._fast_hook)
                    if isinstance(out, Basic) and out.has(sympy.Indexed):
                        out = self._indexed_terms(out, notes)
                if isinstance(out, Basic) and out.has(sympy.ConditionSet):
                    out = self._condsets(out, notes)
                if isinstance(out, Basic) and out.has(Integral):       # risch's NonElementaryIntegral
                    out = out.replace(lambda n: isinstance(n, Integral) and type(n) is not Integral,
                                      lambda n: Integral(*n.args))      # ignores evalf
                if isinstance(out, Basic) and out.has(Apply) and any(   # T^{∘2}(0) left by a written-out sum
                        not _free_vars(a) for a in out.atoms(Apply)):
                    out2 = self._apply_all(out, notes)
                    if out2 != out:
                        out = out2.doit() if isinstance(out2, Basic) else out2
                return out
        except _Timeout as t:
            if t.token is not tok:
                raise
        if frozen:
            val = val.xreplace({d: n for n, d in frozen.items()})
        free = {s for s in val.free_symbols if isinstance(s, Symbol)} - self.bound_variables(val)
        if not free:
            tok2 = None
            try:
                with time_limit(self.budget) as tok2:
                    try:
                        num = val.evalf(self.precision)
                    except (AttributeError, TypeError, ValueError):
                        num = val
                if isinstance(num, Expr) and num.is_number and not num.has(Integral, Sum, Product, Limit):
                    notes.append(f"no exact result within {self.budget:g} s; computed numerically")
                    return num
            except _Timeout as t:
                if t.token is not tok2:
                    raise
        raise SemanticsError(f"no result within the time limit ({self.budget:g} s for this step, "
                             f"-t {self.timeout:g} per statement)",
                             hint="give numeric values (the result is then computed numerically), "
                                  "simplify the input, or raise the limit with -t 60.")

    @staticmethod
    def _factorial_ratio(val):
        """a!/b! (or Γ(a)/Γ(b)) with a − b a small whole number is a short product: exact, whatever
        the size of a -- (10^100)!/(10^100 − 1)! is 10^100."""
        kinds = (sympy.factorial, sympy.gamma)

        def parts(n):
            if not n.is_Mul:
                return None
            for k in kinds:
                nums = [f for f in n.args if isinstance(f, k)]
                dens = [f.base for f in n.args if f.is_Pow and f.exp == -1 and isinstance(f.base, k)]
                if len(nums) == 1 and len(dens) == 1:
                    a, b = nums[0].args[0].doit(), dens[0].args[0].doit()
                    d = a - b
                    if d.is_Integer and 0 <= d <= 1000:
                        rest = [f for f in n.args if f is not nums[0] and not (f.is_Pow and f.base == dens[0])]
                        return k, a, b, int(d), rest
            return None

        def ratio(n):
            k, a, b, d, rest = parts(n)
            first = b + 1 if k is sympy.factorial else b          # a!/b! = (b+1)…a; Γ(a)/Γ(b) = b…(a−1)
            return sympy.Mul(*rest) * sympy.Mul(*[first + j for j in range(d)])
        try:
            return val.replace(lambda n: parts(n) is not None, ratio)
        except Exception:
            return val

    def _simplify(self, val, notes):
        if not (self.simplify and isinstance(val, Expr)) or val.has(Det, Marker):
            return val
        if sympy.count_ops(val) >= 200 or (val.is_number and isinstance(val, (Integer, Float, sympy.Rational))):
            return val
        tok = None
        try:
            with time_limit(3) as tok:
                s = sympy.simplify(val)
        except _Timeout as t:
            if t.token is not tok:
                raise
            return val
        if s != val:
            if sympy.count_ops(s) > sympy.count_ops(val):
                return val
            if _max_digits(s) > max(60, 2 * _max_digits(val)):     # log(2^24727/3^15601): not simpler
                return val
            if val.free_symbols and sympy.count_ops(s) < sympy.count_ops(val):
                notes.append("simplified")
            return s
        return val

    def evaluate(self, expr: Basic, notes: list, *, simplify: bool, subs: Optional[dict] = None) -> Basic:
        subs = {} if subs is None else subs
        if not isinstance(expr, Basic):
            return expr
        if expr.has(PlusMinus):
            plus = expr.replace(PlusMinus, lambda a, b: a + b)
            minus = expr.replace(PlusMinus, lambda a, b: a - b)
            return Tuple(self.evaluate(plus, notes, simplify=simplify, subs=subs),
                         self.evaluate(minus, notes, simplify=simplify, subs=subs))
        val = self._resolve_markers(expr, notes)
        if val.has(Det):
            val = val.replace(Det, lambda m: self._det(m, notes))
        val = self._functionalize(val, notes)
        val = self._zero_based(val, notes)
        self._check_bounds(val)
        calculus = val.has(*_CALCULUS)
        pre = val
        try:
            val = self._doit(val, notes)
            if isinstance(val, Basic) and (val.has(Subs) or (val.has(Derivative) and not val.has(AppliedUndef))):
                val = self._doit(val, notes)
        except (ValueError, NotImplementedError, TypeError) as exc:
            if isinstance(exc, ValueError) and "does not exist" in str(exc):
                notes.append("the limit does not exist: " + str(exc).replace("The limit does not exist since ", ""))
                val = DNE
            else:
                raise SemanticsError(f"could not evaluate: {exc}") from None
        if val is sympy.zoo and pre.has(Limit):
            notes.append("the one-sided limits are infinite with opposite signs: the limit does not exist")
            val = DNE
        if isinstance(val, Basic) and val.has(RealRoot):
            val = val.replace(RealRoot, lambda x, n: Pow(x, 1 / n))
        for pw in (pre.atoms(Pow) if isinstance(pre, Basic) else ()):
            ex = pw.exp.doit() if isinstance(pw.exp, Basic) else pw.exp
            if pw.base.is_number and pw.base.is_negative and getattr(ex, "is_Rational", False) \
                    and not ex.is_Integer and ex.q % 2 == 1:
                notes.append(f"({interp.pyform(pw.base)})^({interp.pyform(ex)}) is the principal (complex) "
                             f"root; \\sqrt[{ex.q}]{{{interp.pyform(pw.base)}}} gives the real one")
                break
        val = self._realify(pre, val, notes)
        if isinstance(val, Expr) and val.has(sympy.conjugate):      # z·conj(z) is |z|²
            for s_ in [q for q in val.free_symbols if isinstance(q, Symbol)]:
                val = val.subs(s_ * sympy.conjugate(s_), sympy.Abs(s_) ** 2)
        if isinstance(val, Basic) and val.has(Marker):
            val = self._resolve_markers(val, notes)
            if val.has(Det):
                val = val.replace(Det, lambda m: self._det(m, notes))
        if isinstance(val, sympy.AccumBounds) and pre.has(Limit):
            notes.append(f"the limit does not exist: the values keep oscillating between "
                         f"{interp.pyform(val.min)} and {interp.pyform(val.max)}")
            val = DNE
        if calculus and isinstance(val, Basic) and val.has(Sum):
            val = self._series_verdict(val, notes)
        if isinstance(val, sympy.Piecewise) and val.args and not val.args[0].expr.has(Sum, Integral) \
                and val.args[-1].expr.has(Sum, Integral):
            notes.append(f"the closed form holds when {interp.pyform(val.args[0].cond)}")
        elif calculus and isinstance(val, Basic) and val.has(Integral, Sum, Product, Limit) \
                and not val.has(AppliedUndef):
            notes.append("no closed form found; left unevaluated")
        if isinstance(val, Basic) and val.has(sympy.Indexed):
            rep = {ix: v for ix in val.atoms(sympy.Indexed) if (v := self._indexed_value(ix)) is not None}
            if rep:
                subs.update(rep)
                val = self._doit(val.xreplace(rep), notes)
        if _units.has_units(val):
            return _units.tidy(val, notes)
        if isinstance(val, Expr) and val.is_number and val.has(Float) and not isinstance(val, Float):
            val = val.evalf(self.precision)
        if self._keep_form:
            pass                          # \operatorname{expand}(…): the user chose the form
        elif simplify:
            val = self._simplify(val, notes)
        elif isinstance(val, Expr) and val.is_number and not val.free_symbols and self.simplify:
            val = self._simplify(val, notes)
        if isinstance(val, Expr) and val.is_number and not val.has(Float) and val.is_real is False \
                and val.has(Pow, sympy.exp):                      # 2e^{iπ/3} is 1 + √3 i, not 2∛(-1)
            try:
                c = sympy.expand_complex(val)
                if sympy.count_ops(c) <= sympy.count_ops(val) + 6:
                    val = c
            except Exception:
                pass
        return val

    @staticmethod
    def _infinite_sums(val) -> list:
        return [s for s in val.atoms(Sum)
                if len(s.limits) == 1 and len(s.limits[0]) == 3
                and (s.limits[0][2] is sympy.oo or s.limits[0][1] is sympy.S.NegativeInfinity)]

    def _series_verdict(self, val, notes):
        """An infinite series left unevaluated: decide whether it converges. A divergent series
        has no value (∞ when its terms are eventually positive); a convergent one is summed
        numerically by :meth:`approx`."""
        for s in self._infinite_sums(val):
            if s.free_symbols:
                continue
            n = s.limits[0][0]
            term = s.function.xreplace({n: Symbol(n.name, integer=True, positive=True)})
            tok = None
            conv = None
            try:
                with time_limit(min(5.0, self.budget)) as tok:
                    conv = Sum(term, (Symbol(n.name, integer=True, positive=True), *s.limits[0][1:])).is_convergent()
            except _Timeout as t:
                if t.token is not tok:
                    raise
            except Exception:
                conv = None
            if conv is sympy.false or conv is False:
                lo = s.limits[0][1]
                m = Symbol("m", integer=True, nonnegative=True)
                pos = s.function.xreplace({n: lo + m}).is_positive if lo.is_finite else term.is_positive
                tail, tok2 = None, None
                try:
                    with time_limit(min(3.0, self.budget)) as tok2:
                        tail = sympy.limit(term, Symbol(n.name, integer=True, positive=True), sympy.oo)
                except _Timeout as t:
                    if t.token is not tok2:
                        raise
                except Exception:
                    tail = None
                proven = pos or (tail is not None and tail != 0 and not isinstance(tail, sympy.AccumBounds)) \
                    or isinstance(tail, sympy.AccumBounds) and not (tail.min <= 0 <= tail.max)
                if not proven:                     # Σ sin n / n: SymPy says "diverges", but it converges
                    notes.append("convergence not proven; a value is shown only if several summation "
                                 "methods agree to 8 digits")
                    self._maybe.add(s)
                    continue
                notes.append("the series diverges" + (" (its terms are positive, so the sum is ∞)" if pos
                                                       else " (its terms do not tend to 0)"))
                return (sympy.oo if pos else DIVERGES) if val == s else DIVERGES
            if conv is None:
                conv = self._integral_test(s, notes)
                if conv is False:
                    notes.append("the series diverges (integral test: its terms are positive and decreasing, and "
                                 "the integral is infinite), so the sum is ∞")
                    return sympy.oo if val == s else DIVERGES
            if conv is sympy.true or conv is True:
                self._convergent.add(s)
                if not any("converges" in x for x in notes):
                    notes.append("the series converges; its value is summed numerically")
        return val

    def _integral_test(self, s, notes):
        """Σ_{n≥a} f(n) with f positive and decreasing on [a, ∞): converges iff ∫_a^∞ f(t) dt is finite."""
        n, lo, hi = s.limits[0]
        if hi is not sympy.oo or not lo.is_finite:
            return None
        t = Symbol("t", positive=True)
        f = s.function.xreplace({n: t})
        tok = None
        try:
            with time_limit(min(6.0, self.budget)) as tok:
                dom = sympy.Interval(lo, sympy.oo)
                if not (sympy.is_decreasing(f, dom) and f.xreplace({t: lo}).is_positive
                        and sympy.limit(f, t, sympy.oo) == 0):
                    return None
                I = sympy.integrate(f, (t, lo, sympy.oo))
        except _Timeout as e:
            if e.token is not tok:
                raise
            return None
        except Exception:
            return None
        if I is sympy.oo:
            return False
        if isinstance(I, Expr) and I.is_finite and not I.has(Integral):
            notes.append(f"converges by the integral test (terms positive and decreasing; "
                         f"∫ from {interp.pyform(lo)} to ∞ = {interp.pyform(I)})")
            return True
        return None

    def _nsum(self, s, min_digits: int = 3):
        """A convergent infinite series summed numerically (mpmath: Richardson/Shanks/Euler–Maclaurin)."""
        import mpmath
        n, lo, hi = s.limits[0]
        em = self._euler_maclaurin(s)            # a computable tail integral: most reliable, try first
        if em is not None and em[1] >= 10:
            return sympy.Float(mpmath.nstr(em[0], em[1]), em[1])
        f = sympy.lambdify(n, s.function, "mpmath")
        lo_m = -mpmath.inf if lo is sympy.S.NegativeInfinity else mpmath.mpf(int(lo))
        hi_m = mpmath.inf if hi is sympy.oo else mpmath.mpf(int(hi))
        tok = None
        vals = []
        try:
            with time_limit(min(8.0, self.budget)) as tok, mpmath.workdps(self.precision + 10):
                for method in ("r+s", "a", "e", "l"):             # several extrapolations: keep what agrees
                    try:
                        v = mpmath.nsum(f, [lo_m, hi_m], method=method)
                    except Exception:
                        if method == "r+s":
                            break
                        continue
                    if mpmath.isfinite(v):
                        vals.append(v)
                    elif method == "r+s":
                        break
        except _Timeout as t:
            if t.token is not tok:
                raise
        if len(vals) >= 2:
            v = vals[0]                         # Richardson+Shanks, confirmed digit by digit by another method
            scale = max(abs(v), mpmath.mpf(10) ** (-self.precision - 5))
            rel = min(abs(v - w) for w in vals[1:]) / scale
            digits = int(min(self.precision, (-mpmath.log10(rel) - 1) if rel > 0 else self.precision))
        else:
            digits = 0
        if em is not None and em[1] > digits:     # fewer digits than 10, but more than the extrapolations
            return sympy.Float(mpmath.nstr(em[0], em[1]), em[1])
        if len(vals) < 2 or digits < min_digits:
            return None                         # the extrapolations disagree: no trustworthy decimal
        brute = self._partial_sum_check(s)
        if brute is not None:                   # Σ 1/(n² + sin n): extrapolations can agree and be wrong
            L, tol = brute
            if abs(float(v) - L) > max(tol, 1e-12 * max(1.0, abs(L))):
                d = int(-math.log10(max(tol, 1e-15) / max(abs(L), 1e-300))) - 1 if tol > 0 else 0
                if d < min_digits:
                    return None
                return sympy.Float(mpmath.nstr(mpmath.mpf(L), min(d, 12)), min(d, 12))
        return sympy.Float(v, digits)

    def _euler_maclaurin(self, s):
        """Σ_{n≥a} f(n) = Σ_{a≤n<N} f(n) + ∫_N^∞ f + f(N)/2 − f′(N)/12 + f‴(N)/720 − f⁽⁵⁾(N)/30240 + …,
        the tail integral exact (or by agreeing quadratures). Two cut-offs N must agree: (value, digits)."""
        n, lo, hi = s.limits[0]
        if hi is not sympy.oo or not lo.is_Integer:
            return None
        t = Symbol("t", positive=True)
        f = s.function.xreplace({n: t})
        tok = None
        try:
            with time_limit(min(8.0, self.budget)) as tok:
                derivs = [sympy.lambdify(t, sympy.diff(f, t, k), "mpmath") for k in (1, 3, 5)]
                fl = sympy.lambdify(t, f, "mpmath")
                N_ = Symbol("N_", positive=True)
                tail = sympy.integrate(f, (t, N_, sympy.oo))
                if not isinstance(tail, Expr) or tail.has(Integral) or tail.has(sympy.oo):
                    return None
                tl = sympy.lambdify(N_, tail, "mpmath")
                out = []
                with mpmath.workdps(self.precision + 15):
                    for N in (int(lo) + 400, int(lo) + 1600):
                        head = mpmath.fsum(fl(mpmath.mpf(k)) for k in range(int(lo), N))
                        x = mpmath.mpf(N)
                        out.append(head + tl(x) + fl(x) / 2 - derivs[0](x) / 12 + derivs[1](x) / 720
                                   - derivs[2](x) / 30240)
        except _Timeout as e:
            if e.token is not tok:
                raise
            return None
        except Exception:
            return None
        a_, b_ = out
        if not (mpmath.isfinite(a_) and mpmath.isfinite(b_)):
            return None
        rel = abs(a_ - b_) / max(abs(b_), mpmath.mpf(10) ** -30)
        digits = int(min(self.precision, (-mpmath.log10(rel) - 1) if rel > 0 else self.precision))
        return (b_, digits) if digits >= 8 else None

    @staticmethod
    def _partial_sum_check(s):
        """The limit of the partial sums S_N, S_2N, S_4N (N = 10⁴), extrapolated geometrically, and a
        tolerance; ``None`` when the terms cannot be computed quickly as floats."""
        import numpy as np
        n, lo, hi = s.limits[0]
        if hi is not sympy.oo or not lo.is_Integer:
            return None
        try:
            f = sympy.lambdify(n, s.function, "numpy")
            N = 10000
            idx = np.arange(int(lo), int(lo) + 4 * N, dtype=float)
            with np.errstate(all="ignore"):
                terms = np.asarray(f(idx), dtype=float)
            if terms.shape != idx.shape or not np.all(np.isfinite(terms)):
                return None
            c = np.cumsum(terms)
            s1, s2, s4 = c[N - 1], c[2 * N - 1], c[4 * N - 1]
        except Exception:
            return None
        d1, d2 = s2 - s1, s4 - s2
        if d1 == 0:
            return float(s4), abs(d2)
        r = d2 / d1
        if not (0 < r < 0.95):
            return float(s4), abs(d2) + abs(d1)          # not geometric: the last sum, loosely
        L = s4 + d2 * r / (1 - r)
        return float(L), abs(d2) * 0.05

    def approx(self, val):
        p = self.precision
        if isinstance(val, Basic) and val is DIVERGES:
            return None
        if isinstance(val, Expr) and val.has(Sum) and not val.free_symbols:
            sums = self._infinite_sums(val)
            for s_ in sums:
                if s_ not in self._convergent:     # simplification may have rewritten the series
                    self._series_verdict(s_, [])
            if sums and not all(s in self._convergent or s in self._maybe for s in sums):
                return None              # an undecided series: a decimal would be guesswork
            if sums:
                rep = {}
                for s in sums:
                    v = self._nsum(s, 3 if s in self._convergent else 8)
                    if v is None:
                        return None
                    rep[s] = v
                try:
                    digits = min([p] + [int(v._prec * 0.30103) for v in rep.values()])
                    out = val.xreplace(rep)
                    return sympy.Float(mpmath.nstr(out.evalf(digits + 2), digits)) if out.is_number else None
                except Exception:
                    return None
        if isinstance(val, Tuple):
            parts = [self.approx(v) for v in val]
            return Tuple(*[a if a is not None else v for a, v in zip(parts, val)]) if any(
                a is not None for a in parts) else None
        if isinstance(val, MatrixBase):
            if all(getattr(x, "is_number", False) for x in val) and not all(
                    isinstance(x, (Integer, Float)) for x in val):
                return val.evalf(p)
            return None
        if isinstance(val, Integer) and val.p.bit_length() > 200:
            return val.evalf(p)
        if isinstance(val, Expr) and val.is_number and val.has(Integral):
            return self._reliable_number(val)           # never a 2-digit quadrature guess
        if isinstance(val, Expr) and val.is_number and val.has(sympy.log, sympy.exp) and not val.has(sympy.I):
            try:                                        # log(p/q − log₂3) with a 10⁻¹²⁴ difference: certified
                v_ = _fast.certified(_fast.compile_expr(val, set(), self._fast_hook), {})
                if isinstance(v_, mpmath.mpf):
                    return sympy.Float(mpmath.nstr(v_, p + 2), p)
            except (_fast.Unsupported, ZeroDivisionError, OverflowError, ValueError, TypeError, KeyError):
                pass
        if isinstance(val, Expr) and val.is_number and not isinstance(val, (Integer, Float)):
            try:
                n = val.evalf(p)
                if n.has(sympy.I):
                    re_, im_ = n.as_real_imag()
                    if re_ != 0 and abs(im_) < abs(re_) * sympy.Float(10) ** (3 - p):
                        n = re_
                    elif im_ != 0 and abs(re_) < abs(im_) * sympy.Float(10) ** (3 - p):
                        n = im_ * sympy.I
            except Exception:
                return None
            return n if n != val and not n.has(sympy.zoo, sympy.nan) else None
        if isinstance(val, Expr) and _units.has_units(val) and not val.has(Float):
            try:
                n = val.evalf(p)
                return n if n != val else None
            except Exception:
                return None
        return None

    # --- classification ---------------------------------------------------------------
    def _decl(self, rhs):
        """``x \\in \\mathbb{R}`` / ``\\omega, t \\in \\mathbb{R}`` with unbound names -> [(name, set)]."""
        items = []
        if isinstance(rhs, Tuple) and rhs.args and isinstance(rhs.args[-1], sympy.Contains) \
                and all(isinstance(a, Symbol) for a in rhs.args[:-1]):
            s = rhs.args[-1].args[1]
            items = [(a, s) for a in rhs.args[:-1]] + [(rhs.args[-1].args[0], s)]
        elif isinstance(rhs, sympy.Contains):
            items = [(rhs.args[0], rhs.args[1])]
        if not items or not all(isinstance(a, Symbol) and self._lookup(a.name) is None
                                and a not in self.solved for a, _ in items):
            return None
        S = sympy.S
        if not all(s in (S.Reals, S.Integers, S.Naturals0, S.Naturals, S.Rationals, S.Complexes)
                   or (isinstance(s, sympy.Interval) and s.inf == 0 and s.sup is S.Infinity) for _, s in items):
            return None
        return items

    def _rates(self) -> dict:
        return self._reg("_openlab_rates")

    def _is_rate(self, lhs, rhs) -> bool:
        """``r = 2`` then ``\\frac{dr}{dt} = 1``: a rate at this instant, not a differential equation.
        Also ``\\frac{dA}{dt} = 2\\pi r \\frac{dr}{dt}`` once a rate of r is known."""
        if not (isinstance(lhs, Derivative) and isinstance(lhs.expr, Symbol)
                and len(lhs.variable_count) == 1 and lhs.variable_count[0][1] == 1):
            return False
        name, var = lhs.expr.name, lhs.variables[0]
        if rhs.has(lhs.expr) or self._lookup(var.name) is not None:
            return False
        if self._number_bound(name):
            return True
        return any(isinstance(d.expr, Symbol) and (f"{d.expr.name}|{var.name}" in self._rates()
                                                   or self._number_bound(d.expr.name)
                                                   or self._lookup(d.expr.name + "dot") is not None)
                   for d in rhs.atoms(Derivative))

    def _rewrite_dot_names(self, expr, notes):
        """``\\dot m`` when ``mdot`` is bound: the flow-rate name, not dm/dt of an unknown m(t)."""
        def is_dn(n):
            return (isinstance(n, Derivative) and isinstance(n.expr, Symbol) and n.variable_count == ((Symbol("t"), 1),)
                    and self._lookup(n.expr.name + "dot") is not None)

        def dn(n):
            notes.append(f"\\dot {n.expr} read as the bound name {n.expr}dot")
            return Symbol(n.expr.name + "dot")
        return expr.replace(is_dn, dn) if expr.has(Derivative) else expr

    def _rewrite_rates(self, expr, notes):
        """Given rates (dr/dt = 1 at r = 2): d(name)/dt is the stated rate, and a formula's rate
        follows by the chain rule (V = 4/3 π r³ -> dV/dt = 4πr² dr/dt)."""
        rates = self._rates()
        if not rates or not expr.has(Derivative):
            return expr

        def rate_of(q, var):
            key = f"{q.name}|{var.name}"
            if key in rates:
                return rates[key]
            if q == var:
                return sympy.S.One
            if self._number_bound(q.name):
                return sympy.S.Zero
            return None

        def applies(n):
            if not (isinstance(n, Derivative) and isinstance(n.expr, Symbol) and len(n.variable_count) == 1
                    and n.variable_count[0][1] == 1):
                return False
            var = n.variables[0]
            if not any(k.endswith(f"|{var.name}") for k in rates):
                return False
            if f"{n.expr.name}|{var.name}" in rates:
                return True
            return self._formula_bound(n.expr.name)

        def rw(n):
            q, var = n.expr, n.variables[0]
            key = f"{q.name}|{var.name}"
            if key in rates:
                notes.append(f"d{q}/d{var} = {interp.short(rates[key])} (the stated rate)")
                return rates[key]
            f = coerce(self._lookup(q.name))
            total = sympy.S.Zero
            for s_ in sorted((x for x in f.free_symbols if isinstance(x, Symbol)), key=lambda x: x.name):
                r = rate_of(s_, var)
                total += sympy.diff(f, s_) * (r if r is not None else Derivative(s_, var))
            notes.append(f"d{q}/d{var} by the chain rule with the stated rates: {interp.pyform(total)}")
            return total
        return expr.replace(applies, rw)

    def _is_event_lhs(self, lhs) -> bool:
        return isinstance(lhs, Apply) and self._is_event_call(lhs)

    def _classify(self, stmt: Statement):
        """-> (kind, target, params); target may be a marker string for special paths."""
        if stmt.chain:
            return Kind.RELATION, "chain", ()
        lhs, rhs = stmt.lhs, stmt.rhs
        if stmt.role == "declare" or (lhs is None and self._decl(rhs)):
            return Kind.ASSUME, None, ()
        if lhs is None:
            if self._sign_decl(rhs) is not None:
                return Kind.ASSUME, "sign", ()
            if isinstance(rhs, Quant):
                return Kind.REPR, None, ()
            if isinstance(rhs, Rel):
                op = str(rhs.args[0])
                if op == "sim":
                    return Kind.ASSIGN, "~", ()
                if op in _SET_OPS:
                    return Kind.EXPR, None, ()
                if op == "in":
                    return Kind.RELATION, None, ()
                return Kind.REPR, None, ()
            if isinstance(rhs, System):
                return Kind.SOLVE, None, ()
            if isinstance(rhs, sympy.Contains) and isinstance(rhs.args[0], Symbol) \
                    and rhs.args[0] in self.solved:
                return Kind.RELATION, "solution", ()
            if isinstance(rhs, (Relational, BooleanFunction, BooleanAtom, sympy.Contains, Mid,
                                Subset, Congruence)):
                return Kind.RELATION, None, ()
            return Kind.EXPR, None, ()
        if stmt.approx:
            return Kind.RELATION, "approx", ()
        if self._is_rate(lhs, rhs):
            return Kind.ASSIGN, "rate", ()
        if stmt.role == "derived" and isinstance(lhs, Symbol) and self._lookup(lhs.name) is None:
            return Kind.RELATION, "solution", ()
        if isinstance(lhs, Symbol) and isinstance(rhs, Relational) and not isinstance(rhs, (Eq, Rel)):
            return Kind.RELATION, "bindcheck", ()        # x = 3 < 4
        if self._data_mean(lhs):
            if stmt.force:
                raise OpenlabError(rf"\bar{{{lhs.args[0]}}} is the mean of the data bound to {lhs.args[0]}: openlab "
                                 rf"computes it, it cannot be assigned. State `\bar{{{lhs.args[0]}}} = c` to check "
                                 "it, or bind the number to another name.")
            return Kind.RELATION, "datamean", ()         # \bar{p} = 0.9 with p data: checked against the mean, not bound
        if isinstance(lhs, (Symbol, Deco)) and not stmt.force and \
                self._fit_ref(lhs.name if isinstance(lhs, Symbol) else self._deco_name(lhs)) is not None:
            return Kind.RELATION, "fitref", ()           # \hat\beta_{m,x} = 0.4: a check of the fit, never a binding
        if isinstance(lhs, (Symbol, Deco)):
            name = lhs.name if isinstance(lhs, Symbol) else self._deco_name(lhs)
            if stmt.force:
                return Kind.ASSIGN, name, ()
            if self.solve_for and {n.strip() for n in self.solve_for.split(",")} & {
                    q.name for q in rhs.free_symbols if isinstance(q, Symbol)}:
                return Kind.SOLVE, "for", ()             # V = I R --for I
            if self._recurrence_base(name, rhs) is not None:
                return Kind.SOLVE, "recurrence", ()
            seq = self._sequence_target(name, rhs)
            if seq is not None:
                return Kind.DEFINE, seq[0], (seq[1],)
            if Symbol(name) in self.solved and self._lookup(name) is None:
                return Kind.RELATION, "solution", ()
            if self._lookup(name) is None and isinstance(lhs, Symbol) and lhs in rhs.free_symbols \
                    and lhs not in self.bound_variables(rhs):
                return Kind.SOLVE, "selfref", ()          # x = \cos x
            if self._formula_bound(name) and not self._same_origin(name, stmt):
                if not rhs.free_symbols or name in self.cell_assigned:
                    return Kind.SOLVE, "formula", ()
            if self._number_bound(name) and not self._same_origin(name, stmt):
                unknown = {s for s in rhs.free_symbols if isinstance(s, Symbol)
                           and self._lookup(s.name) is None and s.name not in ("e", "i")}
                if len(unknown) == 1:
                    return Kind.SOLVE, "number", ()
            return Kind.ASSIGN, name, ()
        if isinstance(lhs, Tuple) and all(isinstance(a, (Symbol, Deco)) for a in lhs):
            return Kind.ASSIGN, tuple(a.name if isinstance(a, Symbol) else self._deco_name(a) for a in lhs), ()
        if self._is_event_lhs(lhs) and not (isinstance(lhs, Apply) and rhs.free_symbols & set(lhs.args[1:])):
            return Kind.ASSIGN, "event", ()
        if (isinstance(lhs, Apply) and isinstance(lhs.args[0], (Symbol, Deco)) and lhs.args[1:]
                and not isinstance(lhs.args[-1], CallMod)
                and all(isinstance(a, Symbol) for a in lhs.args[1:])
                and len(set(lhs.args[1:])) == len(lhs.args) - 1):
            head = lhs.args[0]
            name = head.name if isinstance(head, Symbol) else self._deco_name(head)
            params = tuple(lhs.args[1:])
            if any(isinstance(a, Apply) and a.args[0] == head for a in rhs.atoms(Apply)) \
                    and self._lookup(name) is None:
                raise SemanticsError(f"{name} is defined in terms of itself: a recurrence",
                                     hint=f"write it with subscripts, `{name}_n = …{name}_{{n-1}}…`, and give a "
                                          f"starting value on a line above (`{name}_0 = 1`).")
            if stmt.force or self._same_origin(name, stmt):
                return Kind.DEFINE, name, params
            own = bool(rhs.free_symbols & set(params)) and not any(
                isinstance(a, Apply) and getattr(a.args[0], "name", None) in _STATS_OPS | {name}
                for a in rhs.atoms(Apply))
            if self._lookup(name) is None and (name in KNOWN_FUNCTIONS or name in _STATS_OPS) and own:
                self._shadow_note = name
                return Kind.DEFINE, name, params            # \delta(x) = … deflection, P(v) = … power
            if self._lookup(name) is None and name in KNOWN_FUNCTIONS:
                return Kind.SOLVE, "known", ()
            if (self._lookup(name) is None and name in _STATS_OPS and name not in _PROB_HEADS
                    and len(params) == 1):
                return Kind.RELATION, "statsid", ()
            if self._is_function_value(Symbol(name)):
                if not (rhs.free_symbols & set(params)) or self._uses_functions(rhs):
                    return Kind.SOLVE, "defined", ()
            if name in _PROB_HEADS and len(params) == 1 and self._lookup(name) is None:
                return Kind.ASSIGN, "event", ()
            if isinstance(head, Symbol) and self._call_kind(head, list(params), None)[0] == "product":
                if self._lookup(head.name) is not None and (rhs.free_symbols & set(params)):
                    return Kind.DEFINE, name, params      # v = 3, then v(t) = 9.8t: a new function
                return Kind.SOLVE, None, ()
            return Kind.DEFINE, name, params
        return Kind.SOLVE, None, ()

    def _ctx(self, lineno):
        """Where a line comes from: the notebook cell when the magic says so, else the rest of the
        cell around the line (equal when a cell is re-run with only that line edited). A one-line
        cell has no context: two different one-line cells are different origins."""
        if self.cell_id is not None:
            return ("cell", self.cell_id)
        if len(self.cell_lines) < 2:
            return None
        return hash(tuple(l for i, l in enumerate(self.cell_lines) if i != lineno - 1))

    def _record_origin(self, name, stmt):
        self._reg("_openlab_defs")[name] = {"src": stmt.src, "ctx": self._ctx(stmt.lineno), "line": stmt.lineno}

    def _same_origin(self, name, stmt) -> bool:
        """The statement (re)defines ``name`` from where it was defined: the same text, or the same
        line of a re-run cell whose other lines are unchanged."""
        rec = self._reg("_openlab_defs").get(name)
        if not isinstance(rec, dict):
            return rec == stmt.src
        if rec["src"] == stmt.src:
            return True
        return rec["ctx"] is not None and rec["ctx"] == self._ctx(stmt.lineno) and rec["line"] == stmt.lineno

    def _vec_value(self, name: str):
        return _vec(coerce(self._lookup(name)))

    def _number_bound(self, name: str) -> bool:
        v = coerce(self._lookup(name))
        if isinstance(v, Expr) and not isinstance(v, MatrixBase) and _units.has_units(v):   # 12 V
            return not {q for q in v.free_symbols if not isinstance(q, _units.Quantity)}
        return isinstance(v, Expr) and v.is_number and not isinstance(v, MatrixBase)

    def _formula_bound(self, name: str) -> bool:
        v = coerce(self._lookup(name))
        return (isinstance(v, Expr) and not isinstance(v, MatrixBase) and not _is_rv(v)
                and bool({s for s in v.free_symbols if isinstance(s, Symbol) and not isinstance(s, _units.Quantity)}))

    def _uses_functions(self, expr) -> bool:
        heads = [a.args[0] for a in expr.atoms(Apply) if isinstance(a.args[0], Symbol)]
        return any(self._is_function_value(h) for h in heads)

    def _sequence_target(self, name: str, rhs):
        """``a_n = 2n + 1``: (base, index) when the subscript is a letter used on the right."""
        m = re.fullmatch(r"([A-Za-z]+)_([a-zA-Z])", name)
        if not m or self._lookup(name) is not None or self._lookup(m.group(2)) is not None:
            return None                  # X_L = 2\pi f L with L bound: L is a value, not an index
        base, idx = m.group(1), Symbol(m.group(2))
        if idx.name not in "ijklmnt" and not (base.islower() and len(base) == 1):
            return None                  # V_R, F_g, W_C: engineering names, not sequences
        if idx not in rhs.free_symbols and not any(
                isinstance(self._vec_value(q.name[:-2]), MatrixBase) for q in rhs.free_symbols
                if isinstance(q, Symbol) and q.name.endswith("_" + idx.name)):
            return None
        cur = self._lookup(base)
        if cur is not None and not (isinstance(cur, Lambda) and len(cur.variables) == 1):
            return None
        return base, idx

    # --- execution ---------------------------------------------------------------------
    def _split_names(self, src: str) -> list:
        """Letter runs (``Re``, ``Nu``) that are bound Python names, though written as products."""
        clean = re.sub(r"\\(?:mathrm|text|operatorname|mathit|textrm|mathbf|openlab\w+)\s*\{[^{}]*\}", " ", src)
        clean = re.sub(r"_\{[^{}]*\}|\\[A-Za-z]+", " ", clean)
        out = []
        for m in re.finditer(r"(?<![A-Za-z_])([A-Za-z]{2,})(?![A-Za-z])", clean):
            w = m.group(1)
            if w not in out and not w.startswith("_") and w not in IGNORED and coerce(self._lookup(w)) is not None:
                out.append(w)
        return out

    def execute(self, stmt: Statement) -> Result:
        notes = list(stmt.notes)
        for w in self._split_names(stmt.src):
            notes.append(f"`{w}` is bound in Python, but letters written together multiply "
                         f"({'·'.join(w)}); write \\mathrm{{{w}}} to use it")
        raw = [stmt.lhs, stmt.rhs] + list(stmt.chain) + [e for ic in stmt.ics for e in ic]
        self._qualifier = ""
        self._ode_done = False
        self._keep_form = False
        self._indep = self._indep_var(raw)
        self._dep = set()
        for e in raw:
            if isinstance(e, Basic):
                self._dep |= self._differentiated(e)
        sols = self._reg("_openlab_odesol")                                         # a defined f is not an unknown
        self._dep = {d for d in self._dep if not self._is_function_value(d) or d.name in sols}
        try:
            return self._execute(stmt, notes)
        finally:
            self._indep, self._dep = None, set()

    def _execute(self, stmt: Statement, notes: list) -> Result:
        kind, target, params = self._classify(stmt)
        if stmt.domain is not None and kind in (Kind.ASSIGN, Kind.DEFINE) and stmt.lhs is not None:
            import dataclasses                            # f(x) = x^2 \text{ for } x > 0
            stmt = dataclasses.replace(stmt, rhs=sympy.Piecewise((stmt.rhs, stmt.domain)), domain=None)
        exclude = {p.name for p in params}
        bound: dict = {}
        unknown = None
        numeric = None
        subs: dict = {}
        value: Any = None
        body = interp.pyform(stmt.rhs)
        prefix = kind.value

        if kind is Kind.ASSUME:
            self._assume_body = None
            value, prefix = self._declare(stmt, notes)
            body = self._assume_body or body
        elif kind is Kind.ASSIGN and target == "~":
            try:
                value, target, subs = self._bind_distribution(stmt.rhs, notes)
                bound = {target: value}
                prefix = f"define {target} ~"
                kind = Kind.DEFINE
            except UnsupportedError:
                kind = Kind.REPR
                value, subs = self.resolve(stmt.rhs, set(), notes)
                notes.append(r"\sim with no distribution on the right: represented, not executed")
                prefix = Kind.REPR.value
        elif kind is Kind.ASSIGN and target == "rate":
            name, var = stmt.lhs.expr.name, stmt.lhs.variables[0]
            rhs, subs = self.resolve(stmt.rhs, set(), notes)
            value = self.evaluate(rhs, notes, simplify=True, subs=subs)
            self._rates()[f"{name}|{var.name}"] = value
            if var == Symbol("t") and self._lookup(name) is None:
                self.ns[name + "dot"] = value              # \dot Q = \dot m c \Delta T: Qdot is bound too
            now = coerce(self._lookup(name))
            notes.append("a rate at this instant" + (f" ({name} = {interp.short(now)} now)" if now is not None else "")
                         + f"; later d/d{var} use it (chain rule)")
            prefix = f"rate d{name}/d{var} :="
            body = interp.pyform(stmt.rhs)
            kind = Kind.RELATION
            target = "rate"
        elif kind is Kind.ASSIGN and target == "event":
            value, kind, prefix, subs = self._assign_event(stmt, notes)
            body = f"{interp.pyform(stmt.lhs)} = {interp.pyform(stmt.rhs)}"
        elif kind is Kind.REPR:
            value, subs = self.resolve(stmt.rhs, set(), notes)
            value = self._resolve_markers(value, notes)
            notes.append("represented, not executed")
        elif kind is Kind.RELATION and target == "chain":
            value, subs, bound = self._exec_chain(stmt, notes)
            body = " = ".join(interp.pyform(e) for e in stmt.chain)
            prefix = "check chain:"
            if bound:
                kind, target = Kind.ASSIGN, tuple(bound)
                prefix = f"assign {', '.join(bound)} :="
        elif kind is Kind.RELATION and target == "bindcheck":
            v = self._value_of(stmt.rhs.lhs, notes, subs)
            ok = self._value_of(stmt.rhs, notes, subs)
            body = f"{stmt.lhs} = {interp.pyform(stmt.rhs)}"
            if ok is sympy.true and self._lookup(stmt.lhs.name) is None:
                self.ns[stmt.lhs.name] = v
                bound = {stmt.lhs.name: v}
                notes.append(f"{stmt.lhs} := {interp.short(v)}, and {interp.pyform(stmt.rhs)} holds")
                kind, target, value, prefix = Kind.ASSIGN, stmt.lhs.name, v, f"assign {stmt.lhs} :="
            else:
                notes.append(f"{interp.pyform(stmt.rhs)} is {ok}; {stmt.lhs} was not bound")
                value, prefix = ok, "check chain:"
        elif kind is Kind.RELATION and target == "approx":
            value, subs = self._exec_approx(stmt, notes)
            body = f"{interp.pyform(stmt.lhs)} ≈ {interp.pyform(stmt.rhs)}"
            prefix = "check ≈:"
        elif kind is Kind.RELATION and target in ("datamean", "fitref"):
            lhs_r, subs_a = self.resolve(stmt.lhs, exclude, notes)
            rhs_r, subs_b = self.resolve(stmt.rhs, exclude, notes)
            mean = self.evaluate(lhs_r, notes, simplify=True, subs=subs_a)
            subs = {**subs_a, **subs_b}
            value = self._check(mean, self.evaluate(rhs_r, notes, simplify=True, subs=subs_b), False, stmt.src, notes)
            what = "the mean of bound data" if target == "datamean" else "a fitted model's estimate"
            if value is sympy.false:
                notes.append(f"the {'mean' if target == 'datamean' else 'estimate'} is {sympy.N(mean, 10)}")
            notes.append(f"a check, not an assignment: {what} is computed, never bound")
            body = f"{interp.pyform(stmt.lhs)} = {interp.pyform(stmt.rhs)}"
            prefix = "check:"
        elif kind is Kind.RELATION and target == "solution":
            value, subs, prefix = self._check_solution(stmt, notes)
            if stmt.lhs is not None:
                body = f"{interp.pyform(stmt.lhs)} = {interp.pyform(stmt.rhs)}"
        elif kind is Kind.RELATION and target == "statsid":
            a = self._value_of(stmt.lhs, notes, subs)
            b = self._value_of(stmt.rhs, notes, subs)
            if any(isinstance(v, Basic) and v.atoms(AppliedUndef) for v in (a, b)):
                notes.append(r"an identity about random variables: represented (bind X \sim … to check it)")
                value, kind, prefix = Eq(a, b, evaluate=False), Kind.REPR, Kind.REPR.value
            else:
                value = self._check(a, b, False, stmt.src, notes)
            body = f"{interp.pyform(stmt.lhs)} = {interp.pyform(stmt.rhs)}"
        elif kind is Kind.ASSIGN:
            value, bound, subs, prefix = self._exec_assign(stmt, target, exclude, notes)
            numeric = None if isinstance(target, tuple) or not bound else self.approx(value)
        elif kind is Kind.DEFINE:
            value, subs = self._exec_define(stmt, target, params, exclude, notes)
            bound = {target: value}
            prefix = f"define {target}({', '.join(p.name for p in params)}) :="
            if isinstance(stmt.lhs, Symbol):
                prefix = f"define the sequence {stmt.lhs.name} :="
        elif kind is Kind.SOLVE:
            value, unknown, subs, body, prefix, kind = self._solve_statement(stmt, target, notes)
        elif kind is Kind.RELATION and self._stat_sides(stmt.rhs):
            # `Var(y) < 1.02`: a statistic of data on a side. SymPy refuses to build an inequality around
            # the unevaluated statistic, so each side is evaluated on its own and compared afterwards.
            a_r, subs_a = self.resolve(stmt.rhs.lhs, exclude, notes)
            b_r, subs_b = self.resolve(stmt.rhs.rhs, exclude, notes)
            a = self.evaluate(a_r, notes, simplify=True, subs=subs_a)
            b = self.evaluate(b_r, notes, simplify=True, subs=subs_b)
            subs = {**subs_a, **subs_b}
            rel = value = stmt.rhs.func(a, b)
            if isinstance(value, Basic) and value.has(Rel):
                notes.append("could not be decided (the names are not sets or numbers); represented")
        elif kind is Kind.RELATION:
            rhs, subs = self.resolve(stmt.rhs, exclude, notes)
            rel = self.evaluate(rhs, notes, simplify=False, subs=subs)
            unknowns = [s for s in getattr(rel, "free_symbols", ()) if isinstance(s, Symbol)]
            if isinstance(rel, BooleanFunction) and unknowns and not rel.atoms(Relational, sympy.Contains) \
                    and not rel.has(Marker):
                value = self._decide_logic(rel, unknowns, notes)
            elif isinstance(rel, (sympy.Implies, sympy.Equivalent)) and len(unknowns) == 1 \
                    and all(isinstance(a, Relational) for a in rel.args):
                value = self._decide_sets(rel, unknowns[0], notes)
            elif isinstance(rel, Relational) and len(unknowns) == 1:
                unknown = unknowns[0]
                value = self._solve(rel, unknown, notes)
            elif isinstance(rel, sympy.And) and len(unknowns) == 1 and \
                    all(isinstance(a, Relational) for a in rel.args):
                unknown = unknowns[0]
                try:
                    value = sympy.reduce_inequalities(list(rel.args), unknown)
                except Exception:
                    value = rel
            else:
                value = rel
            if isinstance(value, Basic) and value.has(Rel):
                notes.append("could not be decided (the names are not sets or numbers); represented")
        else:  # EXPR
            rhs, subs = self.resolve(stmt.rhs, exclude, notes)
            value = self.evaluate(rhs, notes, simplify=True, subs=subs)
            if isinstance(value, MatrixBase) and value.shape == (1, 1):
                notes.append("a 1×1 result is shown as a number")
                value = value[0, 0]
            if isinstance(value, Basic) and value.has(Rel):
                notes.append("could not be decided (the names are not sets); represented")
            self._note_constant_derivative(stmt, subs, notes)
            if getattr(stmt, "convert", None) == "deg":
                if not (isinstance(value, Expr) and value.is_number and value.is_real):
                    raise SemanticsError("only a real number of radians can be expressed in degrees")
                value = sympy.nsimplify(value * 180 / sympy.pi) if not value.has(Float) else value * 180 / sympy.pi
                notes.append("the radians expressed in degrees")
            elif getattr(stmt, "convert", None):
                value = _units.convert(value, stmt.convert, notes, src=stmt.src)

        if isinstance(value, BooleanAtom):
            notes[:] = [n for n in notes if not n.startswith("no closed form")]
        if self.numeric:
            if isinstance(value, Expr) and self._infinite_sums(value):     # summed only when it converges
                summed = self.approx(value)
                value = summed if summed is not None else value
            elif isinstance(value, Expr) and value.has(Integral) and not _free_vars(value):
                rel = self._reliable_number(value)
                if rel is not None:
                    value = rel
                else:
                    notes.append("no reliable numeric value (the quadratures disagree); left as written")
            else:
                value = _evalf(value, self.precision)
            if isinstance(value, Expr) and value.is_number and value.has(sympy.I):
                re_, im_ = value.as_real_imag()
                if re_ != 0 and abs(im_) < abs(re_) * sympy.Float(10) ** (3 - self.precision):
                    value = re_                              # |F(1/3)|: rounding left 10⁻¹⁹ i
            numeric = None
        elif kind not in (Kind.ASSIGN, Kind.DEFINE, Kind.REPR, Kind.ASSUME) and isinstance(value, Basic):
            numeric = self.approx(value)
        latex_out = self._latex_out(kind, target, params, value, numeric, unknown, stmt)
        interpretation = interp.build(prefix, body, subs, notes)
        shown = display_src(stmt.display or stmt.src)
        return Result(src=shown, kind=kind, expr=stmt.rhs, value=value, numeric=numeric,
                      interpretation=interpretation, bound=bound, latex_in=shown,
                      latex_out=latex_out, notes=list(dict.fromkeys(notes)), lineno=stmt.lineno)

    @staticmethod
    def _decide_sets(rel, var, notes):
        """``x > 2 \\implies x > 1``: true when every real solution of the premise solves the conclusion."""
        real = Symbol(var.name, real=True)
        a, b = [sympy.solveset(r.xreplace({var: real}), real, sympy.S.Reals) for r in rel.args]
        ok = a.is_subset(b) if isinstance(rel, sympy.Implies) else (a.is_subset(b) and b.is_subset(a))
        if ok is None:
            return rel
        p_, q_ = interp.pyform(rel.args[0]), interp.pyform(rel.args[1])
        if isinstance(rel, sympy.Implies):
            notes.append(f"every real {var} with {p_} {'also has' if ok else 'does not always have'} {q_}")
        else:
            notes.append(f"{p_} and {q_} {'hold for the same' if ok else 'do not hold for the same'} real {var}")
        return sympy.sympify(bool(ok))

    @staticmethod
    def _decide_logic(rel, props, notes):
        """A statement about propositions p, q, …: a tautology is True, a contradiction False;
        otherwise the simplest equivalent form."""
        names = ", ".join(sorted(s.name for s in props))
        from sympy.logic.inference import satisfiable
        if not satisfiable(sympy.Not(rel)):
            notes.append(f"true for every truth value of {names}: a tautology")
            return sympy.true
        if not satisfiable(rel):
            notes.append(f"false for every truth value of {names}: a contradiction")
            return sympy.false
        s = sympy.simplify_logic(rel) if len(props) <= 8 else rel
        if s is sympy.true:
            notes.append(f"true for every truth value of {names}: a tautology")
            return sympy.true
        if s is sympy.false:
            notes.append(f"false for every truth value of {names}: a contradiction")
            return sympy.false
        if s != rel:
            notes.append(f"depends on {names}; simplest equivalent form shown")
        else:
            notes.append(f"depends on the truth values of {names}")
        return s

    def _claim_reading(self, claim) -> str:
        """How openlab reads a claim it was not asked to prove: each variable with its condition, and the
        body in SymPy form -- so the reading can be checked before a proof is attempted."""
        from .interp import pyform
        from .proof import Prover
        tr = {r"\ge": "≥", r"\geq": "≥", r"\le": "≤", r"\leq": "≤", r"\in": "∈", r"\mathbb{N}": "ℕ",
              r"\mathbb{Z}": "ℤ", r"\mathbb{R}": "ℝ", r"\mathbb{C}": "ℂ", r"\ne": "≠", r"\neq": "≠"}

        def cond(v):
            s = claim.conds.get(v.name, "").strip()
            for k in sorted(tr, key=len, reverse=True):
                s = s.replace(k, tr[k])
            return f"{v} {s}".strip() if s else f"{v} (no condition)"
        doms = ", ".join(cond(v) for v in claim.vars)
        try:
            kind, A, B, m = Prover(self)._body(claim)
            rel = {"eq": "=", "cong": "≡"}.get(kind) or {sympy.GreaterThan: "≥", sympy.StrictGreaterThan: ">",
                                                        sympy.LessThan: "≤", sympy.StrictLessThan: "<"}.get(m, "?")
            body = f"{pyform(A)} {rel} {pyform(B)}" + (f" (mod {pyform(m)})" if kind == "cong" else "")
        except Exception:
            return f"; read as: {doms}"
        return f"; read as: {'∀' if claim.quant == 'forall' else '∃'} {doms}: {body}"

    def note(self, c) -> Result:
        """Prose, a markdown line, or a claim without a technique: shown as written, never run."""
        text = c.display or c.text
        if c.role == "claim":
            from .proof import parse_claim
            read = ""
            try:
                claim = parse_claim(c.text)
                what = rf"{'∀' if claim.quant == 'forall' else '∃'} statement"
                read = self._claim_reading(claim)
            except OpenlabError:
                what = "statement"
            interp_ = (f"{what}: a note, not executed{read} -- add \\text{{by induction}}, \\text{{by algebra}} "
                       f"or \\text{{witness }} … to prove it")
            return Result(src=display_src(text), kind=Kind.NOTE, expr=sympy.S.Zero, value=None, numeric=None,
                          interpretation=interp_, latex_in=display_src(text), latex_out="", lineno=c.lineno)
        if c.role == "markdown":
            return Result(src=text, kind=Kind.NOTE, expr=sympy.S.Zero, value=None, numeric=None,
                          interpretation="markdown note", latex_in="", latex_out="", lineno=c.lineno,
                          notes=["markdown"])
        shown = display_src(text)
        return Result(src=shown, kind=Kind.NOTE, expr=sympy.S.Zero, value=None, numeric=None,
                      interpretation="prose: a note, not executed", latex_in=shown, latex_out="",
                      lineno=c.lineno)

    def prove(self, c) -> Result:
        """A claim over a set with a named technique (induction, algebra, a witness)."""
        from .proof import Prover, parse_claim
        claim = parse_claim(c.text)
        if claim.technique is None:
            return self.note(c)
        notes: list = []
        value, steps = Prover(self).run(claim, notes)
        if value is sympy.true and claim.quant == "forall":
            from .proof import record_fact
            record_fact(self, claim, c.lineno)
        verdict = {sympy.true: "proved", sympy.false: "disproved"}.get(value, "not proved")
        on = claim.on or (claim.vars[0].name if claim.vars else "")
        method = {"induction": f"by induction on {on}", "strong": f"by strong induction on {on}",
                  "algebra": "by algebra", "witness": "by a witness",
                  "exhaustion": "by exhaustion", "cases": "by cases"}[claim.technique]
        shown = display_src(c.display or c.text)
        if value is sympy.true:
            out = rf"\text{{proved {method}}}"
        elif value is sympy.false:
            out = r"\text{disproved}"
        else:
            out = rf"\text{{not proved ({method} did not go through)}}"
        interpretation = interp.build(f"{verdict} {method}:", claim.body, {}, steps + notes)
        return Result(src=shown, kind=Kind.PROOF, expr=sympy.S.Zero, value=value, numeric=None,
                      interpretation=interpretation, latex_in=shown, latex_out=out,
                      notes=list(dict.fromkeys(steps + notes)), lineno=c.lineno)

    def _note_constant_derivative(self, stmt, subs, notes):
        for d in stmt.rhs.atoms(Derivative):
            if isinstance(d.expr, Symbol) and d.expr in subs:
                val = subs[d.expr]
                if isinstance(val, Basic) and not (set(d.variables) & val.free_symbols):
                    notes.append(f"{d.expr} = {interp.short(val)} does not contain "
                                 f"{', '.join(map(str, d.variables))}, so its derivative is 0")

    # --- the individual statement kinds ------------------------------------------------------
    def _declare(self, stmt, notes):
        sign = self._sign_decl(stmt.rhs) if stmt.lhs is None else None
        if sign is not None:
            sym, kw = sign
            self._reg("_openlab_assume")[sym.name] = kw
            if stmt.role == "local":
                notes.append(f"assumed for the rest of this line only: {sym} is {next(iter(kw))}")
            else:
                notes.append(f"taken as an assumption: later statements treat {sym} as {next(iter(kw))}")
            return stmt.rhs, "assume"
        items = self._decl(stmt.rhs)
        if not items:
            raise SemanticsError("an assumption needs unbound names and a standard set",
                                 hint=r"e.g. `x \in \mathbb{R}`, `n \in \mathbb{N}`, `\omega, t \in \mathbb{R}`.")
        S = sympy.S
        reg = self._reg("_openlab_assume")
        for sym, s in items:
            kw = ({"real": True} if s is S.Reals else {"integer": True} if s is S.Integers
                  else {"integer": True, "nonnegative": True} if s is S.Naturals0
                  else {"integer": True, "positive": True} if s is S.Naturals
                  else {"rational": True} if s is S.Rationals
                  else {"complex": True} if s is S.Complexes
                  else {"positive": True} if isinstance(s, sympy.Interval) else {})
            if kw:
                reg[sym.name] = kw
        names = ", ".join(s.name for s, _ in items)
        setname = {S.Reals: "ℝ", S.Integers: "ℤ", S.Naturals0: "ℕ", S.Naturals: "ℕ", S.Rationals: "ℚ",
                   S.Complexes: "ℂ"}.get(items[0][1], interp.pyform(items[0][1]))
        if stmt.role == "local":
            notes.append(f"for the rest of this line only, {names} ∈ {setname}")
        else:
            notes.append(f"later statements treat {names} as members of {setname}")
        self._assume_body = f"{names} ∈ {setname}"
        return sympy.And(*[sympy.Contains(a, s, evaluate=False) for a, s in items]), "assume"

    def _exec_assign(self, stmt, target, exclude, notes):
        rhs_raw = stmt.rhs
        if isinstance(rhs_raw, Tuple) and not isinstance(target, tuple):
            rhs_raw = ImmutableMatrix(list(rhs_raw))
            notes.append("a tuple is stored as a column vector")
        rhs, subs = self.resolve(rhs_raw, exclude, notes)
        names = target if isinstance(target, tuple) else (target,)
        for nm in names:
            if Symbol(nm) in getattr(rhs, "free_symbols", ()) and self._lookup(nm) is None:
                raise SemanticsError(f"{nm} is used on both sides but is not bound yet",
                                     hint=f"bind {nm} on a line above, or use a different name.")
        value = self.evaluate(rhs, notes, simplify=False, subs=subs)
        if isinstance(value, Tuple) and not isinstance(target, tuple) and stmt.rhs.has(PlusMinus):
            notes.append(f"{target} = ± … has two values; nothing was bound (write {target} := … to pick one)")
            return value, {}, subs, f"values of {target}:"
        shown = None
        if self.numeric:                     # -n: the decimal is shown; the exact value is what is bound
            shown = _evalf(value, self.precision)
            exact_nums = (isinstance(value, Expr) and value.is_number) or (
                isinstance(value, (Tuple, MatrixBase)) and all(getattr(x, "is_number", False) for x in value))
            if exact_nums and _max_digits(value) > 60:      # C₂ as a 30 000-digit fraction: the decimal is bound
                notes.append(f"{target if not isinstance(target, tuple) else ', '.join(target)} is bound to its "
                             f"decimal under -n (the exact value has {_max_digits(value)}-digit integers)")
                value, shown = shown, None
            elif exact_nums and not value.has(sympy.Float) and shown != value:
                notes.append(f"{target if not isinstance(target, tuple) else ', '.join(target)} keeps its exact "
                             f"value; -n shows it as a decimal")
            else:
                value, shown = shown, None
        bound = {}
        if isinstance(target, tuple):
            if not isinstance(value, (Tuple, MatrixBase)) or len(value) != len(target):
                raise SemanticsError(f"{len(target)} names on the left but the right is not a {len(target)}-tuple")
            for nm, v in zip(target, value):
                self.ns[nm] = v
                bound[nm] = v
        else:
            old = self._lookup(target)
            if old is not None:
                shown_old = coerce(old)
                notes.append(f"rebinding {target} (was {interp.short(shown_old if shown_old is not None else old)})")
            if "." in target and target not in self.ns and self._frame_column(target) is not None:
                fr, _, col = target.partition(".")
                notes.append(f"`{target}` was column {col} of frame {fr}; this binds a value over it in statements "
                             rf"(the frame is unchanged). To check a value instead, write \bar{{\mathrm{{{target}}}}} "
                             rf"= {interp.pyform(stmt.rhs)} or \max(\mathrm{{{target}}}) = …")
            self.ns[target] = value
            bound = {target: value}
            self._reg("_openlab_odesol").pop(target, None)
            self._record_origin(target, stmt)
            if isinstance(value, Expr) and value.free_symbols:
                self.cell_assigned.add(target)
        if stmt.role == "local":
            notes.append(f"{', '.join(names)} = … holds for this line only (a `where` binding)")
        return (shown if shown is not None else value), bound, subs, f"assign {', '.join(names)} :="

    _shadow_note = None

    def _index_bound_lists(self, expr, var):
        """c_n, c_{2j} with c a bound list (or sequence) and n the variable -> c[n]: an entry, not a name."""
        from .transform import ToSympy
        if not isinstance(expr, Basic):
            return expr
        cand = ToSympy._index_body(expr, var)
        keep = set()
        for ix in cand.atoms(sympy.Indexed):
            base = str(ix.base.label)
            if isinstance(coerce(self._lookup(base)), (Tuple, MatrixBase)) or base in self._reg("_openlab_recur") \
                    or base in self._reg("_openlab_seqs"):
                keep.add(base)
        if not keep:
            return expr
        back = {ix: Symbol(f"{ix.base.label}_{{{sympy.sstr(ix.indices[0])}}}")
                for ix in cand.atoms(sympy.Indexed) if str(ix.base.label) not in keep}
        return cand.xreplace(back) if back else cand

    def _exec_define(self, stmt, target, params, exclude, notes):
        if self._shadow_note == target:
            meaning = {"delta": "the Dirac delta", "Phi": "the normal CDF", "P": "a probability",
                       "E": "an expectation"}.get(target, "a standard function")
            notes.append(f"{target} is also {meaning}; this defines your own {target}")
            self._shadow_note = None
        rhs_raw = stmt.rhs
        from .transform import ToSympy
        if isinstance(stmt.lhs, Symbol) and len(params) == 1:        # y_n = (x_n + x_{n-1})/2 with x data
            rhs_raw = ToSympy._index_body(rhs_raw, params[0])
        elif isinstance(rhs_raw, Basic):                             # G(n) = … c_n … with c a bound list:
            for p_ in params:                                        # the parameter indexes c
                rhs_raw = self._index_bound_lists(rhs_raw, p_)
        rhs, subs = self.resolve(rhs_raw, exclude, notes)
        body_expr = self.evaluate(rhs, notes, simplify=False, subs=subs)
        value = Lambda(params, body_expr)
        if self._lookup(target) is not None:
            notes.append(f"rebinding {target}")
        self.ns[target] = value
        self._record_origin(target, stmt)
        if isinstance(stmt.lhs, Symbol):                  # a_n = 2n + 1
            self._reg("_openlab_seqs")[target] = True
        if isinstance(stmt.lhs, Apply) and isinstance(stmt.lhs.args[0], Deco):
            deco = stmt.lhs.args[0]
            DISPLAY_NAMES[Symbol(target)] = rf"\{deco.args[1]}{{{sympy.latex(deco.args[0])}}}"
        return value, subs

    def _assign_event(self, stmt, notes):
        ev = stmt.lhs.args[1]
        key = self._ev_key(ev)
        rhs, subs = self.resolve(stmt.rhs, set(), notes)
        val = self.evaluate(rhs, notes, simplify=True, subs=subs)
        reg = self._reg("_openlab_events")
        known = reg[key] if key in reg else self._prob(ev)
        restate = key in reg and self._same_origin(f"event:{key}", stmt)     # the stating line, edited and re-run
        if known is not None and not stmt.force and not restate:
            ok = self._check(known, val, False, stmt.src, notes)
            notes.append(f"P({interp.pyform(ev)}) = {interp.pyform(known)} already follows from the stated probabilities")
            if ok is sympy.false:
                notes.append(f"to restate it, write P({interp.pyform(ev)}) := {interp.pyform(val)}")
            return ok, Kind.RELATION, Kind.RELATION.value, subs
        if isinstance(val, Expr) and val.is_number and val.is_real and not (0 <= val <= 1):
            raise SemanticsError(f"a probability must be between 0 and 1 (got {interp.pyform(val)})",
                                 hint="probabilities are written as numbers in [0, 1], e.g. 0.25 or 1/4.")
        self._frechet(ev, val)
        if key in reg and reg[key] != val:
            notes.append(f"restating P({interp.pyform(ev)}) (was {interp.pyform(reg[key])})")
        reg[key] = val
        self._record_origin(f"event:{key}", stmt)
        notes.append("recorded; P of unions, intersections, complements and conditionals now follow")
        return val, Kind.ASSIGN, "state", subs

    def _frechet(self, ev, val):
        """P(A∩B) ≤ min(P(A), P(B)) and ≥ P(A) + P(B) − 1; P(A∪B) between max and min(1, sum)."""
        if not (isinstance(ev, Rel) and str(ev.args[0]) in ("cap", "cup")) or not getattr(val, "is_number", False):
            return
        pa, pb = self._prob(ev.args[1]), self._prob(ev.args[2])
        known = [p for p in (pa, pb) if p is not None and getattr(p, "is_number", False)]
        if not known:
            return
        if len(known) == 2:
            lo, hi = ((max(0, pa + pb - 1), min(pa, pb)) if str(ev.args[0]) == "cap"
                      else (max(pa, pb), min(1, pa + pb)))
        else:                                     # only one side known: P(A∩B) ≤ P(A) ≤ P(A∪B)
            lo, hi = (0, known[0]) if str(ev.args[0]) == "cap" else (known[0], 1)
        if not (lo - 1e-12 <= val <= hi + 1e-12):
            given = " and ".join(f"P({interp.pyform(e)}) = {interp.pyform(p)}"
                                 for e, p in ((ev.args[1], pa), (ev.args[2], pb)) if p is not None)
            raise SemanticsError(f"inconsistent: with {given}, P({interp.pyform(ev)}) must lie in "
                                 f"[{interp.pyform(lo)}, {interp.pyform(hi)}]",
                                 hint="check the stated probabilities.")

    def _value_of(self, e, notes, subs):
        r, s = self.resolve(e, set(), notes)
        subs.update(s)
        return self.evaluate(r, notes, simplify=False, subs=subs)

    def _exec_chain(self, stmt, notes):
        exprs, ops = stmt.chain, stmt.chain_ops
        self._chain_stmt = stmt
        subs: dict = {}
        bound: dict = {}
        if all(isinstance(e, Symbol) and self._lookup(e.name) is None for e in exprs[:-1]) \
                and all(op in ("=", "def") for op in ops):          # a = b = 2
            v = self._value_of(exprs[-1], notes, subs)
            for e in exprs[:-1]:
                self.ns[e.name] = v
                bound[e.name] = v
            notes.append(f"{', '.join(e.name for e in exprs[:-1])} := {interp.short(v)}")
            return Tuple(*[v] * (len(exprs) - 1)), subs, bound
        start = 0
        first = exprs[0]
        pending = None
        if isinstance(first, Symbol) and self._lookup(first.name) is None and ops[0] in ("=", "def"):
            pending = self._value_of(exprs[1], notes, subs)    # s = 2 + 3 = 5: bind s if the rest holds
            start = 1
        results = []
        for i in range(start, len(exprs) - 1):
            a = self._value_of(exprs[i], notes, subs)
            b = self._value_of(exprs[i + 1], notes, subs)
            if ops[i] == "approx":
                ok = self._approx_check(a, b, stmt.src, notes)
            else:
                ok = self._check(a, b, False, stmt.src, notes)
            results.append(ok)
            if ok is sympy.false:
                notes.append(f"link {i + 1} fails: {interp.pyform(a)} ≠ {interp.pyform(b)}")
        if all(r is sympy.true for r in results):
            value = sympy.true
        elif any(r is sympy.false for r in results):
            value = sympy.false
        else:
            value = sympy.And(*results)
        if pending is not None:
            if value is sympy.false:
                notes.append(f"the chain does not hold, so {first.name} was not bound")
                return value, subs, bound
            self.ns[first.name] = pending
            self._record_origin(first.name, self._chain_stmt)
            bound[first.name] = pending
            notes.append(f"{first.name} := {interp.short(pending)}; "
                         + ("the rest of the chain checks out" if value is sympy.true
                            else "the rest of the chain could not be decided"))
            return Tuple(pending), subs, bound
        return value, subs, bound

    def _exec_approx(self, stmt, notes):
        subs: dict = {}
        a = self._value_of(stmt.lhs, notes, subs)
        b = self._value_of(stmt.rhs, notes, subs)
        if isinstance(a, Expr) and isinstance(b, Expr) and a.is_number and b.is_number:
            return self._approx_check(a, b, stmt.src, notes), subs
        notes.append("an approximation with unknowns is stated, not executed")
        return (sympy.Eq(a, b, evaluate=False) if isinstance(a, Expr) and isinstance(b, Expr) else a), subs

    def _approx_check(self, a, b, src, notes):
        places = []
        for m in re.finditer(r"(?<![\w.])\d+\.(\d+)(?:\s*\\times\s*10\s*\^\s*\{?\s*(-?\d+)\s*\}?|[eE]([+-]?\d+))?", src):
            places.append(-len(m.group(1)) + int(m.group(2) or m.group(3) or 0))
        place = max(places) if places else 0          # the coarsest digit written
        places = -place
        tol = sympy.Rational(5, 10) * sympy.Integer(10) ** place
        diff = abs((a - b).evalf(self.precision + 5))
        where = f"{places} decimal{'s' if places != 1 else ''}" if places >= 0 else f"±{sympy.N(tol, 2)}"
        notes.append(f"approximate check to {where} (|difference| = {sympy.N(diff, 3)})")
        return sympy.sympify(bool(diff <= tol))

    def _check_solution(self, stmt, notes):
        """``x = 4`` / ``x = \\pm 2`` / ``x \\in \\{…\\}`` after solving for x in this cell."""
        if stmt.lhs is not None:
            name = stmt.lhs.name
            rhs, subs = self.resolve(stmt.rhs, set(), notes)
        else:
            name = stmt.rhs.args[0].name
            rhs, subs = self.resolve(stmt.rhs.args[1], set(), notes)
        vals = self.evaluate(rhs, notes, simplify=True, subs=subs)
        cand = set(vals) if isinstance(vals, (Tuple, sympy.FiniteSet)) else {vals}
        entry = self.solved.get(Symbol(name))
        if entry is None:            # an \implies chain without an earlier solve for the name
            notes.append(f"{name} has no earlier solution to compare with; nothing was bound")
            return sympy.FiniteSet(*cand), subs, f"values of {name}:"
        sols, line = entry
        solset = sympy.FiniteSet(*[sympy.simplify(s) for s in sols]) if isinstance(sols, list) else sols
        cand = {sympy.simplify(c) for c in cand}
        where = f"line {line}" if line else "the earlier equation"
        if stmt.role == "derived" and isinstance(solset, sympy.FiniteSet):     # x^2 = 9 \implies x = 3
            lost = [q for q in solset if q not in cand]
            if lost:
                notes.append(f"does not follow: {name} = {interp.pyform(lost[0])} also satisfies {where}")
            else:
                notes.append(f"follows from {where}")
            return sympy.sympify(not lost), subs, "check the step:"
        ok = all(bool(solset.contains(c)) for c in cand)
        exact = isinstance(solset, sympy.FiniteSet) and set(solset) == cand
        if ok and exact:
            notes.append(f"matches the solutions of {where}")
        elif ok:
            notes.append(f"{'is one' if len(cand) == 1 else 'are some'} of the solutions of {where}: {interp.pyform(solset)}")
        else:
            notes.append(f"not what {where} gives: {name} ∈ {interp.pyform(solset)}")
        notes.append(f"{name} stays unbound (write {name} := … to bind it)")
        return sympy.sympify(ok), subs, f"check against {where}:"

    # --- solving ---------------------------------------------------------------------
    def _solve_statement(self, stmt: Statement, target, notes: list):
        if stmt.lhs is None:     # a bare \begin{cases} system
            exclude = {s.name for s in self._dep} | ({self._indep.name} if self._dep and self._indep else set())
            rhs, subs = self.resolve(stmt.rhs, exclude, notes)
            sys_ = self._functionalize(self._resolve_markers(rhs, notes), notes)
            eqs = [Eq(self.evaluate(e.lhs, notes, simplify=False), self.evaluate(e.rhs, notes, simplify=False))
                   for e in sys_.args]
            return self._solve_system(eqs, subs, notes, interp.pyform(stmt.rhs))
        if target == "recurrence":
            return self._solve_recurrence(stmt, notes)
        if target is None:
            imp = self._implicit(stmt, notes)
            if imp is not None:
                return imp
        body = f"Eq({interp.pyform(stmt.lhs)}, {interp.pyform(stmt.rhs)})"
        if target == "formula":
            notes.append(f"{stmt.lhs} holds a formula, so `{stmt.lhs} = …` is read as an equation "
                         f"(write {stmt.lhs} := … to rebind it)")
        elif target == "defined":
            notes.append(f"{stmt.lhs.args[0]} is already defined, so this is read as an equation "
                         f"(write {interp.pyform(stmt.lhs)} := … to redefine it)")
        elif target == "number":
            notes.append(f"{stmt.lhs} is bound to {interp.short(coerce(self._lookup(stmt.lhs.name)))}, so this is an "
                         f"equation (write {stmt.lhs} := … to rebind it)")
        elif target == "known":
            notes.append(f"{stmt.lhs.args[0]} is a standard function, so this is read as an equation "
                         f"(write {interp.pyform(stmt.lhs)} := … to define your own)")
        elif target == "for":
            notes.append(f"--for {self.solve_for}: `{stmt.lhs} = …` is read as an equation")
        vec = self._vector_unknown(stmt, notes)
        if vec is not None:
            return vec
        exclude = {s.name for s in self._dep}
        if self._dep and self._indep is not None:
            exclude.add(self._indep.name)
        if self.solve_for:
            exclude |= {n.strip() for n in self.solve_for.split(",")}
        lhs, lsubs = self.resolve(stmt.lhs, exclude, notes)
        rhs, rsubs = self.resolve(stmt.rhs, exclude, notes)
        subs = {**lsubs, **rsubs}
        if isinstance(rhs, System):
            eqs = [Eq(lhs, e) for e in self._resolve_markers(rhs, notes).args]
            return self._solve_system(eqs, subs, notes, body)
        both = self._resolve_markers(Tuple(lhs, rhs), notes)
        both = self._functionalize(both, notes)
        lhs, rhs = both.args
        lhs_v = self.evaluate(lhs, notes, simplify=False, subs=subs)
        rhs_v = self.evaluate(rhs, notes, simplify=False, subs=subs)
        if isinstance(lhs_v, MatrixBase) or isinstance(rhs_v, MatrixBase):
            return self._matrix_equation(lhs_v, rhs_v, subs, notes, body)
        if isinstance(lhs_v, sympy.Set) or isinstance(rhs_v, sympy.Set):
            return self._check(lhs_v, rhs_v, False, stmt.src, notes), None, subs, body, Kind.RELATION.value, Kind.RELATION
        if any(all(len(lim) == 1 for lim in i.limits) for i in stmt.lhs.atoms(Integral) | stmt.rhs.atoms(Integral)):
            ident = self._identity(lhs_v, rhs_v, stmt, notes)      # an antiderivative claim is checked
            if ident is None and notes[-1:] == [OUT_OF_TIME]:      # undecided, never a disproof
                return Eq(lhs_v, rhs_v, evaluate=False), None, subs, body, Kind.RELATION.value, Kind.RELATION
            if ident is None:
                notes.append("not an antiderivative: differentiating the right side does not give the integrand")
            return (sympy.true if ident is not None else sympy.false), None, subs, body, Kind.RELATION.value, Kind.RELATION
        eq = Eq(lhs_v, rhs_v, evaluate=False)
        funcs = sorted({f for f in eq.atoms(AppliedUndef)
                        if any(d.expr == f for d in eq.atoms(Derivative))}, key=str)
        if funcs:
            return self._solve_ode(eq, funcs, stmt, subs, notes, body)
        unknowns = sorted((s for s in eq.free_symbols if isinstance(s, Symbol)), key=lambda s: s.name)
        wanted = [n.strip() for n in self.solve_for.split(",")] if self.solve_for else []
        if len(wanted) > 1:                            # --for x,y: a system, solved with the next lines
            raise MultipleUnknowns(f"--for {self.solve_for} names {len(wanted)} unknowns",
                                   hint="put the equations of the system on consecutive lines.")
        if self.solve_for:
            unknown = Symbol(self.solve_for)
            if unknown not in unknowns:
                raise SemanticsError(f"{self.solve_for} does not appear in the equation",
                                     hint=f"the unknowns here are {', '.join(u.name for u in unknowns) or 'none'}.")
        elif len(unknowns) == 1:
            unknown = unknowns[0]
        elif target in ("defined", "known") and isinstance(stmt.lhs, Apply) and \
                sum(1 for p in stmt.lhs.args[1:] if p in unknowns) == 1:
            unknown = next(p for p in stmt.lhs.args[1:] if p in unknowns)
            notes.append(f"solved for {unknown}, the argument of {stmt.lhs.args[0]}")
        elif len(unknowns) > 1 and self._back_substitute(eq, unknowns, notes) is not None:
            out, pair = self._back_substitute(eq, unknowns, notes)
            return out, pair, subs, body, f"solve for {', '.join(p.name for p in pair)}:", Kind.SOLVE
        elif len(unknowns) > 1:
            ident = self._identity(lhs_v, rhs_v, stmt, notes)
            if ident is not None:
                self._qualifier = ident
                return sympy.true, None, subs, body, Kind.RELATION.value, Kind.RELATION
            names = ", ".join(s.name for s in unknowns)
            hint = (f"bind the others on a line above (`{unknowns[-1].name} = 2`), pass "
                    f"`--for {unknowns[0].name}`, or put the equations of a system on consecutive lines.")
            if target in ("formula", "number"):
                hint = f"{stmt.lhs} is already bound: write {stmt.lhs} := … to rebind it. Or " + hint
            split = self._split_names(stmt.src)
            if split:
                hint = f"`{split[0]}` is bound, but written as letters it is a product: write \\mathrm{{{split[0]}}}. " + hint
            raise MultipleUnknowns(f"{len(unknowns)} unknowns ({names}): which one should be solved for?", hint=hint)
        else:
            unknown = None
        if unknown is None:
            value = self._check(lhs_v, rhs_v, False, stmt.src, notes)
            names = [str(k) for k in subs if isinstance(k, Symbol)]
            if names:
                notes.append(f"{', '.join(names)} {'is' if len(names) == 1 else 'are'} bound, so this was checked, "
                             f"not solved (pass --for {names[0]} to solve for it anyway)")
            return value, None, subs, body, Kind.RELATION.value, Kind.RELATION
        ident = self._identity(lhs_v, rhs_v, stmt, notes, positive=False)
        if ident is not None:
            self._qualifier = ident
            return sympy.true, None, subs, body, Kind.RELATION.value, Kind.RELATION
        if target == "selfref":
            notes.append(f"{unknown} appears on both sides, so this is an equation (write {unknown} := … "
                         f"to bind it once {unknown} has a value)")
        if stmt.domain is not None:
            value = self._solve_in(eq, unknown, stmt.domain, notes)
        else:
            value = self._solve(eq, unknown, notes)
        if target == "selfref" and isinstance(value, list) and not value:     # x = x + 1
            raise SemanticsError(f"{unknown} is used on both sides but is not bound yet, and as an equation "
                                 f"it has no solution",
                                 hint=f"bind {unknown} on a line above (then `{unknown} := {interp.pyform(stmt.rhs)}` "
                                      f"updates it), or use a different name.")
        if isinstance(value, list) and not value:
            notes.append("no value satisfies the equation")
        if stmt.role == "derived" and self.last_solve is not None and self.last_solve[0] == unknown:
            return self._implication(unknown, value, subs, body, notes)
        if isinstance(value, list) and any(isinstance(v, Expr) and _units.has_units(v) for v in value):
            value = [_units.tidy(v, notes) if isinstance(v, Expr) and _units.has_units(v) else v for v in value]
        if "circ" in (stmt.display or stmt.src) and isinstance(value, list) and value and all(
                isinstance(v, Expr) and v.is_number and v.is_real for v in value):      # asked in degrees
            degs = [sympy.nsimplify(v * 180 / sympy.pi) if not v.has(Float) else (v * 180 / sympy.pi).evalf(6)
                    for v in value]
            notes.append("in degrees: " + ", ".join(f"{interp.pyform(d)}°" for d in degs))
        self.solved[unknown] = (value, stmt.lineno)
        self.last_solve = (unknown, value, stmt.lineno)
        return value, unknown, subs, body, f"solve for {unknown.name}:", Kind.SOLVE

    def _back_substitute(self, eq, unknowns, notes):
        """``\\omega_n^2 = 10`` then ``2 \\zeta \\omega_n = 2``: with every unknown but one solved on
        an earlier line of this cell, solve for the last one on each branch -> (solutions, names)."""
        known = [u for u in unknowns if u in self.solved and isinstance(self.solved[u][0], list)
                 and 0 < len(self.solved[u][0]) <= 4 and not isinstance(self.solved[u][0][0], dict)]
        rest = [u for u in unknowns if u not in known]
        if len(known) != 1 or len(rest) != 1:
            return None
        k, r = known[0], rest[0]
        out = []
        for v in self.solved[k][0]:
            sub_notes: list = []
            sols = self._solve(eq.xreplace({k: v}), r, sub_notes)
            for q in (sols if isinstance(sols, list) else []):
                out.append({k: v, r: q})
        if not out:
            return None
        line = self.solved[k][1]
        notes.append(f"{k} from {'line ' + str(line) if line else 'above'}; solved for {r} on each of its values")
        return out, Tuple(k, r)

    def _implication(self, unknown, value, subs, body, notes):
        prev, line = self.last_solve[1], self.last_solve[2]
        a = sympy.FiniteSet(*prev) if isinstance(prev, list) else prev
        b = sympy.FiniteSet(*value) if isinstance(value, list) else value
        ok = bool(a.is_subset(b))
        notes.append(f"{'follows from' if ok else 'does not follow from'} line {line}: "
                     f"{unknown} ∈ {interp.pyform(a)} there, ∈ {interp.pyform(b)} here")
        self.solved[unknown] = (value, line)
        return sympy.sympify(ok), None, subs, body, "check the step:", Kind.RELATION

    @staticmethod
    def _implicit_derivative(F, x, y, n=1):
        """d^n y/dx^n on the curve F(x, y) = 0 (y a function of x)."""
        Y = Function(y.name)(x)
        yp = Derivative(Y, x)
        first = sympy.solve(sympy.diff(F.xreplace({y: Y}), x), yp)
        out = []
        for p in first:
            cur = p
            for _ in range(n - 1):
                cur = sympy.simplify(sympy.diff(cur, x).subs(yp, p))
            out.append(sympy.simplify(cur.subs(Y, y)))
        return out

    def _implicit(self, stmt, notes):
        """``\\frac{d}{dx}(x^2 + y^2) = \\frac{d}{dx}(25)`` with y unbound: y is a function of x,
        and the equation is solved for dy/dx."""
        l, r = stmt.lhs, stmt.rhs
        if not (isinstance(l, Derivative) and len(l.variable_count) == 1 and l.variable_count[0][1] == 1) \
                or "partial" in stmt.src or isinstance(l.expr, Symbol):
            return None                     # ∂f/∂x = 0 is an equation in x, not an implicit curve
        x = l.variables[0]
        written = {s for s in l.expr.free_symbols | getattr(r, "free_symbols", set())
                   if isinstance(s, Symbol) and s != x and self._lookup(s.name) is None}
        if len(written) != 1:
            return None
        if isinstance(r, Derivative) and r.variables == l.variables:
            r_body = r.expr
        elif getattr(r, "is_number", False) and r == 0:
            r_body = sympy.S.Zero
        else:
            return None
        both, subs = self.resolve(Tuple(l.expr, r_body), {x.name}, notes)
        both = self._resolve_markers(both, notes)
        F = both[0] - both[1]
        others = [s for s in F.free_symbols if isinstance(s, Symbol) and s != x]
        if len(others) != 1 or not isinstance(F, Expr):
            return None
        y = others[0]
        vals = self._implicit_derivative(F, x, y)
        if not vals:
            return None
        notes.append(f"implicit differentiation: {y} is read as a function of {x}")
        body = f"Eq({interp.pyform(l)}, {interp.pyform(r)})"
        return vals, Derivative(y, x), subs, body, f"solve for d{y}/d{x}:", Kind.SOLVE

    def implicit_followup(self, stmt, nxt_clauses, parse) -> Optional[Result]:
        """``x^2 + y^2 = 25`` followed by ``\\frac{dy}{dx}``: implicit differentiation of the curve."""
        if not nxt_clauses or stmt.lhs is None or stmt.chain:
            return None
        try:
            nxt = parse(nxt_clauses[0])
        except OpenlabError:
            return None
        d = nxt.rhs
        if nxt.lhs is not None or not isinstance(d, Derivative) or not isinstance(d.expr, Symbol) \
                or len(d.variable_count) != 1:
            return None
        y, (x, n) = d.expr, d.variable_count[0]
        notes: list = []
        both, subs = self.resolve(Tuple(stmt.lhs, stmt.rhs), {x.name, y.name}, notes)
        both = self._resolve_markers(both, notes)
        F = both[0] - both[1]
        if not isinstance(F, Expr) or {s for s in F.free_symbols if isinstance(s, Symbol)} != {x, y}:
            return None
        vals = self._implicit_derivative(F, x, y, int(n))
        if not vals:
            return None
        notes.append(f"implicit differentiation of line {stmt.lineno}: {y} is read as a function of {x}")
        unknown = Derivative(y, (x, n))
        rows = r" \\ ".join(display_src(s.display or s.src) for s in (stmt, nxt))
        body = f"{interp.pyform(unknown)} on Eq({interp.pyform(stmt.lhs)}, {interp.pyform(stmt.rhs)})"
        return Result(src=rows, kind=Kind.SOLVE, expr=d, value=vals, numeric=None,
                      interpretation=interp.build(f"solve for d{y}/d{x}:", body, subs, notes),
                      latex_in=rows, latex_out=self._latex_out(Kind.SOLVE, None, (), vals, None, unknown, stmt),
                      notes=notes, lineno=stmt.lineno)

    def _identity(self, lhs, rhs, stmt, notes, positive: bool = True):
        """``None`` unless lhs = rhs for every value; otherwise the qualifier ('' for a plain
        identity or for all reals, 'for integer n, k, m', 'for positive values' -- the last only
        with several unknowns). An equation with an indefinite integral holds up to a constant."""
        try:
            d = lhs - rhs
        except Exception:
            return None
        if not isinstance(d, Expr) or sympy.count_ops(d) > 300:
            return None
        tok = None
        try:
            with time_limit(4) as tok:
                ints = [i for i in stmt.lhs.atoms(Integral) | stmt.rhs.atoms(Integral)
                        if all(len(lim) == 1 for lim in i.limits)]
                if ints:
                    var = ints[0].variables[0]
                    rv = {q: Symbol(q.name, real=True) for q in d.free_symbols if isinstance(q, Symbol)}
                    real = (d.xreplace(rv), rv.get(var, var))       # ln|x| needs x real; complex |.| is slow
                    for dd, vv in ((real,) if d.has(sympy.Abs) else ((d, var), real)):
                        if sympy.simplify(sympy.diff(dd, vv)) == 0:
                            notes.append("equal up to a constant of integration")
                            return ""
                    der = sympy.diff(d.xreplace(rv), rv.get(var, var))
                    if der.free_symbols <= {rv.get(var, var)} and self._zero_at_points(der, rv.get(var, var)):
                        notes.append("equal up to a constant of integration (on each interval of the domain)")
                        return ""
                    return None
                if sympy.simplify(d) == 0:
                    notes.append("both sides agree for every value: an identity")
                    return ""
                free = [s for s in d.free_symbols if isinstance(s, Symbol)]
                if not free:
                    return None
                tiers = [("real", "for all real values")]
                if any(s.name in ("n", "k", "m", "j") for s in free):
                    tiers.append(("integer", "for integer n, k, m"))
                if positive or d.has(sympy.log):
                    tiers.append(("positive", "for positive values"))
                for kw, words in tiers:
                    if kw == "integer":
                        rep = {s: Symbol(s.name, integer=True) for s in free if s.name in ("n", "k", "m", "j")}
                    else:
                        rep = {s: Symbol(s.name, **{kw: True}) for s in free}
                    if sympy.simplify(d.xreplace(rep)) == 0:
                        notes.append(f"an identity {words}")
                        return "" if kw == "real" else words
        except _Timeout as t:
            if t.token is not tok:
                raise
            notes.append(OUT_OF_TIME)
        return None

    @staticmethod
    def _zero_at_points(expr, var) -> bool:
        """``expr`` vanishes at a spread of sample points where it is defined (at least four)."""
        ok = 0
        for p in (0.37, 1.3, 2.9, 7.1, -0.43, -1.7, -3.3, -6.2):
            try:
                v = complex(expr.subs(var, p).evalf(30))
            except (TypeError, ValueError, ZeroDivisionError):
                continue
            if v != v or abs(v) == float("inf"):
                continue
            if abs(v) > 1e-20:
                return False
            ok += 1
        return ok >= 4

    def _vector_unknown(self, stmt, notes):
        """``A x = b`` / ``\\begin{pmatrix}…\\end{pmatrix} x = b`` with x unbound: x is a vector."""
        lhs = stmt.lhs
        if not ((getattr(lhs, "is_Mul", False) and len(lhs.args) >= 2) or (isinstance(lhs, Star) and len(lhs.args) == 2)):
            return None
        m_raw, xs = (Mul(*lhs.args[:-1], evaluate=False) if len(lhs.args) > 2 else lhs.args[0]), lhs.args[-1]
        if isinstance(xs, Deco):
            xs = xs.args[0] if str(xs.args[1]) == "vec" else Symbol(self._deco_name(xs))
        if not isinstance(xs, Symbol) or self._lookup(xs.name) is not None:
            return None
        sub_notes: list = []
        M_r, msubs = self.resolve(m_raw, set(), sub_notes)
        M = self.evaluate(M_r, sub_notes, simplify=False, subs=msubs)
        if not isinstance(M, MatrixBase):
            return None
        b_r, bsubs = self.resolve(stmt.rhs, set(), sub_notes)
        b = self.evaluate(b_r, sub_notes, simplify=False, subs=bsubs)
        if not isinstance(b, MatrixBase):
            return None
        notes.extend(sub_notes)
        if b.cols != 1 or b.rows != M.rows:
            raise SemanticsError(f"the shapes do not fit: a {M.rows}×{M.cols} matrix times a vector "
                                 f"cannot equal a {b.rows}×{b.cols} matrix")
        body = f"Eq({interp.pyform(lhs)}, {interp.pyform(stmt.rhs)})"
        unknowns = sympy.symbols(f"{xs.name}_1:{M.cols + 1}")
        sol = sympy.linsolve((sympy.Matrix(M), sympy.Matrix(b)), *unknowns)
        subs = {**msubs, **bsubs}
        if sol == sympy.S.EmptySet:
            if M.rows > M.cols:
                ls = M.solve_least_squares(b)
                notes.append("no exact solution; least-squares solution")
                return [ls], xs, subs, body, f"solve for {xs} (least squares):", Kind.SOLVE
            notes.append("the system is inconsistent: no solution")
            return [], xs, subs, body, f"solve for {xs}:", Kind.SOLVE
        vec = ImmutableMatrix(list(next(iter(sol))))
        if vec.free_symbols & set(unknowns):
            notes.append("infinitely many solutions; the free entries stay symbolic")
        notes.append(f"solved the linear system for the vector {xs}")
        return [vec], xs, subs, body, f"solve for {xs}:", Kind.SOLVE

    def _matrix_equation(self, lhs, rhs, subs, notes, body):
        lhs, rhs = self._zero_matrix(lhs, rhs)
        if not (isinstance(lhs, MatrixBase) and isinstance(rhs, MatrixBase)):
            raise SemanticsError("one side is a matrix and the other is not",
                                 hint=r"both sides of a matrix equation must be matrices of the same shape; "
                                      r"for eigenvalues write `\operatorname{eig}(A)`.")
        if lhs.shape != rhs.shape:
            raise _shape_error(None, lhs, rhs)
        diff = lhs - rhs
        if not diff.free_symbols:
            ok = sympy.sympify(bool(diff.is_zero_matrix))
            return ok, None, subs, body, Kind.RELATION.value, Kind.RELATION
        eqs = [Eq(e, 0) for e in diff if e != 0]
        unknowns = {s for e in eqs for s in e.free_symbols if isinstance(s, Symbol)}
        if len(unknowns) > len(eqs) and not self.solve_for and not self.underdetermined_ok:
            names = ", ".join(sorted(s.name for s in unknowns))
            err = MultipleUnknowns(f"{len(eqs)} equations but {len(unknowns)} unknowns ({names})",
                                   hint="add the remaining equations on the next lines (they are solved "
                                        "together), or pass --for to pick the unknowns.")
            err.retry = True
            raise err
        notes.append(f"a matrix equation: {len(eqs)} equations, one per entry")
        return self._solve_system(eqs, subs, notes, body)

    @staticmethod
    def _zero_matrix(lhs, rhs):
        """``\\nabla f = 0``: a scalar 0 beside a matrix is the zero matrix of its shape."""
        if isinstance(lhs, MatrixBase) and not isinstance(rhs, MatrixBase) and rhs == 0:
            rhs = ImmutableMatrix(sympy.zeros(*lhs.shape))
        elif isinstance(rhs, MatrixBase) and not isinstance(lhs, MatrixBase) and lhs == 0:
            lhs = ImmutableMatrix(sympy.zeros(*rhs.shape))
        return lhs, rhs

    def _solve_ode(self, eq, funcs, stmt, subs, notes, body):
        fn = funcs[0]
        if len(fn.args) > 1:
            try:
                sol = sympy.pdsolve(eq, fn)
                notes.append(f"solved as a partial differential equation for {fn}")
                return [sol.rhs], fn, subs, body, f"solve PDE for {fn}:", Kind.SOLVE
            except Exception:
                notes.append("a partial differential equation: represented (openlab solves ODEs "
                             "and first-order linear PDEs)")
                return eq, None, subs, body, Kind.REPR.value, Kind.REPR
        var = fn.args[0]
        ics = {}
        for l_raw, r_raw in stmt.ics:
            key = self._ic_key(l_raw, fn, var)
            if key is None:
                raise SemanticsError(f"could not read the initial condition {interp.pyform(l_raw)} = …",
                                     hint=f"write it as {fn.func}(0) = 1 or {fn.func}'(0) = 0.")
            r, s = self.resolve(r_raw, set(), notes)
            subs.update(s)
            v = self.evaluate(r, notes, simplify=False, subs=subs)
            if key in ics and sympy.simplify(ics[key] - v) != 0:
                raise SemanticsError(f"two different initial conditions for {interp.pyform(key)}: "
                                     f"{interp.pyform(ics[key])} and {interp.pyform(v)}",
                                     hint="keep one of them.")
            ics[key] = v
        floats = eq.has(Float) or any(isinstance(v, Basic) and v.has(Float) for v in ics.values())
        if floats:                           # SymPy's solvers recurse on float coefficients: solve exactly
            eq = _exact(eq)
            ics = {k: _exact(v) for k, v in ics.items()}
        tok = None
        try:
            with time_limit(self.budget * 1.5) as tok:
                sol = sympy.dsolve(eq, fn, ics=ics or None)
        except _Timeout as t:
            if t.token is not tok:
                raise
            raise SemanticsError(f"the differential equation was not solved within {self.budget * 1.5:g} s",
                                 hint="it may have no closed form (a nonlinear ODE such as a pendulum): "
                                      "linearise it, or raise the limit with -t 60.") from None
        except RecursionError:
            raise SemanticsError("no closed-form solution found for this differential equation",
                                 hint="linearise it (sin θ ≈ θ), or solve it numerically in Python "
                                      "(scipy.integrate.solve_ivp).") from None
        except Exception as exc:
            why = str(exc)
            internal = any(k in why for k in ("Subs(", "lie group", "operand", "object", "'", "list", "_"))
            raise SemanticsError("could not solve the differential equation" + ("" if internal else f": {why}"),
                                 hint="openlab solves linear ODEs and the standard nonlinear forms; "
                                      "check the initial conditions, or linearise it.") from None
        if any(q.lhs != fn for q in (sol if isinstance(sol, list) else [sol])):
            sol = self._explicit_ode(eq, fn, ics, sol)
        if floats:
            sol = [q.func(q.lhs, q.rhs.evalf(self.precision)) for q in (sol if isinstance(sol, list) else [sol])]
            sol = sol[0] if len(sol) == 1 else sol
        self._ode_done = True
        notes.append(f"solved as a differential equation for {fn}"
                     + (f" with {len(ics)} initial condition{'s' if len(ics) > 1 else ''}" if ics else ""))
        value = [sol.rhs] if isinstance(sol, Eq) else [s.rhs for s in sol]
        name = fn.func.__name__
        if len(value) == 1 and not any(str(c).startswith("C") and c.name[1:].isdigit()
                                       for c in value[0].free_symbols if isinstance(c, Symbol)):
            self.ns[name] = Lambda(var, value[0])
            self._reg("_openlab_odesol")[name] = True
            notes.append(f"{name} is now this solution: {name}({interp.pyform(var)}) can be evaluated")
        return value, fn, subs, body, f"solve ODE for {fn}:", Kind.SOLVE

    def _explicit_ode(self, eq, fn, ics, implicit):
        """dsolve gave F(y, t) = C (the logistic equation): try the methods that give y = …"""
        try:
            hints = [h for h in sympy.classify_ode(eq, fn) if not h.endswith("_Integral")]
        except Exception:
            return implicit
        for h in hints:
            if h in ("factorable", "separable", "1st_exact", "lie_group", "1st_power_series"):
                continue
            tok = None
            try:
                with time_limit(min(5.0, self.budget)) as tok:
                    out = sympy.dsolve(eq, fn, hint=h, ics=ics or None)
            except _Timeout as t:
                if t.token is not tok:
                    raise
                continue
            except Exception:
                continue
            if all(q.lhs == fn for q in (out if isinstance(out, list) else [out])):
                return out
        raise SemanticsError("the solution is only implicit: " + interp.pyform(
            implicit if not isinstance(implicit, list) else implicit[0]),
            hint="openlab could not solve it for the function; the implicit relation is shown.")

    def _ic_key(self, l_raw, fn, var):
        """`y(0)` -> y(0); `y'(0)` -> the derivative at 0; `\\dot{x}(0)` likewise."""
        name = fn.func.__name__
        if isinstance(l_raw, Apply):
            head, args, mod = self._split_apply(l_raw)
            if len(args) != 1:
                return None
            n = 0
            if isinstance(head, Derivative) and isinstance(head.expr, Symbol):
                n, head = head.variable_count[0][1], head.expr
            if mod is not None and str(mod.args[0]) in ("prime", "nderiv"):
                n = int(mod.args[1])
            if not isinstance(head, Symbol) or head.name != name:
                return None
            a = args[0]
            if isinstance(a, Basic) and a.free_symbols:          # y(L) = 0 with L bound: the point's value
                a = self.evaluate(self.resolve(a, set(), [])[0], [], simplify=False)
            if n == 0:
                return fn.func(a)
            return fn.diff(var, n).subs(var, a)
        return None

    def _solve_system(self, eqs, subs, notes, body, ic_list=None):
        funcs = sorted({f for e in eqs for f in e.atoms(AppliedUndef)
                        if any(d.expr == f for d in e.atoms(Derivative))}, key=str)
        if funcs:
            ics = {}
            for l_raw, r_raw in (ic_list or []):
                key = None
                for fn in funcs:
                    key = self._ic_key(l_raw, fn, fn.args[0])
                    if key is not None:
                        break
                if key is None:
                    raise SemanticsError(f"could not read the initial condition {interp.pyform(l_raw)} = …")
                r, s_ = self.resolve(r_raw, set(), notes)
                ics[key] = self.evaluate(r, notes, simplify=False, subs=s_)
            tok = None
            try:
                with time_limit(self.budget * 1.5) as tok:
                    sol = sympy.dsolve(eqs, funcs, ics=ics or None)
            except _Timeout as t:
                if t.token is not tok:
                    raise
                raise SemanticsError("the system of differential equations was not solved in time") from None
            except Exception as exc:
                raise SemanticsError(f"could not solve the system of differential equations: {exc}") from None
            sols = [{q.lhs: q.rhs for q in (sol if isinstance(sol, list) else [sol])}]
            notes.append(f"solved as a system of {len(eqs)} differential equations"
                         + (f" with {len(ics)} initial conditions" if ics else ""))
            return sols, Tuple(*funcs), subs, body, "solve ODE system:", Kind.SOLVE
        unknowns = sorted({s for e in eqs for s in e.free_symbols if isinstance(s, Symbol)}, key=lambda s: s.name)
        want = [Symbol(n.strip()) for n in self.solve_for.split(",")] if self.solve_for else None
        if want is None and len(unknowns) > len(eqs):
            pref = [u for u in unknowns if u.name in ("x", "y", "z", "w")]
            if len(pref) == len(eqs):
                params = [u for u in unknowns if u not in pref]
                notes.append(f"solved for {', '.join(p.name for p in pref)}; {', '.join(p.name for p in params)} "
                             f"{'is a parameter' if len(params) == 1 else 'are parameters'} (--for picks others)")
                want = pref
        tok = None
        try:
            with time_limit(self.budget) as tok:
                sols = sympy.solve(eqs, want if want and not self.solve_for else unknowns, dict=True)
        except _Timeout as t:
            if t.token is not tok:
                raise
            raise SemanticsError(f"the system was not solved within {self.budget:g} s") from None
        except Exception as exc:
            raise SemanticsError(f"could not solve the system: {exc}") from None
        if want:
            sols = [{k: v for k, v in s.items() if k in want} for s in sols]
            unknowns = want
        notes.append(f"solved {len(eqs)} equation{'s' if len(eqs) != 1 else ''} for "
                     f"{', '.join(s.name for s in unknowns)}")
        if not sols:
            notes.append("the system is inconsistent: no solution")
        for u_ in unknowns:
            vals = [s_[u_] for s_ in sols if u_ in s_]
            if vals:
                self.solved[u_] = (vals, None)
        return sols, Tuple(*unknowns), subs, body, "solve system:", Kind.SOLVE

    def execute_system(self, stmts: list) -> Result:
        """Consecutive equations (or coupled differential equations) solved together."""
        notes: list = []
        raw = [e for st in stmts for e in (st.lhs, st.rhs) if isinstance(e, Basic)]
        self._indep = self._indep_var(raw)
        sols = self._reg("_openlab_odesol")
        self._dep = {d for d in set().union(*[self._differentiated(e) for e in raw])
                     if not self._is_function_value(d) or d.name in sols}
        try:
            exclude = {s.name for s in self._dep}
            if self._dep:
                exclude.add(self._indep.name)
            if self.solve_for:
                exclude |= {n.strip() for n in self.solve_for.split(",")}
            sides, subs = [], {}
            for st in stmts:
                for e in (st.lhs, st.rhs):
                    r, s_ = self.resolve(e, exclude, notes)
                    subs.update(s_)
                    sides.append(r)
            sides = list(self._functionalize(self._resolve_markers(Tuple(*sides), notes), notes).args)
            sides = [self.evaluate(x, notes, simplify=False) for x in sides]
            eqs = []
            for k in range(0, len(sides), 2):
                a, b = self._zero_matrix(sides[k], sides[k + 1])
                if isinstance(a, MatrixBase) and isinstance(b, MatrixBase):      # ∇f = λ∇g: one per entry
                    if a.shape != b.shape:
                        raise _shape_error(None, a, b)
                    eqs.extend(Eq(p, q) for p, q in zip(a, b) if p != q)
                elif isinstance(a, MatrixBase) or isinstance(b, MatrixBase):
                    raise SemanticsError("one side is a matrix and the other is not")
                else:
                    eqs.append(Eq(a, b))
            where = (f"lines {stmts[0].lineno}–{stmts[-1].lineno}" if stmts[0].lineno != stmts[-1].lineno
                     else "these equations")
            notes.append(f"{where} solved together as a system")
            value, unk, subs, body, prefix, kind = self._solve_system(
                eqs, subs, notes, "; ".join(interp.pyform(e) for e in eqs),
                ic_list=[ic for st in stmts for ic in st.ics])
        finally:
            self._indep, self._dep = None, set()
        rows = r" \\ ".join(display_src(s.display or s.src) for s in stmts)
        return Result(src=rows, kind=kind, expr=Tuple(*eqs), value=value, numeric=None,
                      interpretation=interp.build(prefix, body, subs, notes),
                      latex_in=rf"\begin{{cases}} {rows} \end{{cases}}",
                      latex_out=self._latex_out(kind, None, (), value, None, unk, stmts[0]),
                      notes=list(dict.fromkeys(notes)), lineno=stmts[0].lineno)

    def joins_system(self, stmt: Statement, group: list) -> bool:
        """``y = x`` after ``x^2 + y^2 = 1``: an unbound name set equal to something is one more
        equation of the system, not an assignment."""
        if not isinstance(stmt.lhs, Symbol) or stmt.chain or stmt.force or self._lookup(stmt.lhs.name) is not None:
            return False
        names = set()
        for g in group:
            for e in (g.lhs, g.rhs):
                if isinstance(e, Basic):
                    names |= {s for s in e.free_symbols if isinstance(s, Symbol)}
        return stmt.lhs in names

    def _unbound_names(self, st) -> set:
        names = set()
        for e in (st.lhs, st.rhs):
            if isinstance(e, Basic):
                names |= {q for q in e.free_symbols if isinstance(q, Symbol) and self._lookup(q.name) is None}
        return names

    def shares_unknowns(self, stmt: Statement, group: list) -> bool:
        """``\\sin^2 a + \\cos^2 a = 1`` after a system in x, y is its own statement. A line whose
        unknowns only appear once names resolve (``\\nabla f = \\lambda \\nabla g``) cannot tell."""
        owned = self._unbound_names(stmt)
        theirs = set().union(*[self._unbound_names(g) for g in group])
        if any(e.has(Apply, Nabla) for g in [stmt, *group] for e in (g.lhs, g.rhs) if isinstance(e, Basic)):
            return True
        return not owned or not theirs or bool(owned & theirs)

    def ode_coupled(self, group: list, nxt: Statement) -> bool:
        """``\\dot x = -y`` then ``\\dot y = x``: differential equations that mention each other's
        unknowns are one system."""
        def deps(st):
            return set().union(*[self._differentiated(e) for e in (st.lhs, st.rhs) if isinstance(e, Basic)])

        def free(st):
            return set().union(*[e.free_symbols for e in (st.lhs, st.rhs) if isinstance(e, Basic)])
        if nxt.lhs is None or nxt.chain or not deps(nxt) or any(g.lhs is None or not deps(g) for g in group):
            return False
        gd = set().union(*[deps(g) for g in group])
        gf = set().union(*[free(g) for g in group])
        nd, nf = deps(nxt), free(nxt)
        return bool((nd & gf) or (gd & nf)) and not (nd & gd)

    def is_plain_equation(self, stmt: Statement) -> bool:
        if stmt.lhs is None or stmt.chain or stmt.approx or stmt.force:
            return False
        try:
            kind, target, _ = self._classify(stmt)
        except Exception:
            return False
        return kind is Kind.SOLVE and target in (None, "defined", "selfref")

    def _check(self, lhs, rhs, approx: bool, src: str, notes: list):
        if approx:
            return self._approx_check(lhs, rhs, src, notes)
        if isinstance(lhs, sympy.Set) or isinstance(rhs, sympy.Set):
            if not (isinstance(lhs, sympy.Set) and isinstance(rhs, sympy.Set)):
                return sympy.false
            if _letter_sets(lhs, rhs):
                return sympy.sympify(set(lhs.args) == set(rhs.args))
            return sympy.sympify(bool(lhs.is_subset(rhs)) and bool(rhs.is_subset(lhs)))
        if isinstance(lhs, MatrixBase) or isinstance(rhs, MatrixBase):
            try:
                return sympy.sympify(bool((lhs - rhs).is_zero_matrix))
            except Exception:
                return sympy.sympify(lhs == rhs)
        try:
            if isinstance(lhs, Expr) and isinstance(rhs, Expr) and (lhs.has(Float) or rhs.has(Float)) \
                    and lhs.is_number and rhs.is_number:
                a, b = complex(lhs.evalf(30)), complex(rhs.evalf(30))
                if abs(a - b) <= 1e-12 * max(abs(a), abs(b), 1):
                    notes.append("compared as decimals (floating-point tolerance)")
                    return sympy.true
                if re.search(r"\d\.\d", src) and self._approx_check(lhs, rhs, src, []) is sympy.true:
                    notes.append(r"the two sides agree to the digits shown, not exactly: \approx checks to that precision")
                return sympy.false
            return sympy.sympify(sympy.simplify(lhs - rhs) == 0)
        except TypeError:
            return sympy.sympify(lhs == rhs)

    def _solve(self, eq, unknown, notes):
        """Real solutions first; complex ones only when there are no real ones. Decimal
        coefficients are solved as the exact fractions they denote, then shown as decimals."""
        if isinstance(eq, Eq) and eq.has(Float) and not eq.has(sympy.Piecewise):
            try:
                exact = _exact(eq)
            except Exception:
                exact = None
            if isinstance(exact, Eq):
                out = self._solve_exact(exact, unknown, notes)
                if isinstance(out, list):
                    return [_evalf(v, self.precision) if isinstance(v, Basic) and v.is_number else v for v in out]
                return out
        return self._solve_exact(eq, unknown, notes)

    def _solve_exact(self, eq, unknown, notes):
        real = unknown if unknown.is_real else Symbol(unknown.name, real=True)
        eq_r = eq.xreplace({unknown: real})
        errors = []

        def attempt(e, v):
            tok = None
            try:
                with time_limit(self.budget) as tok:
                    return sympy.solve(e, v)
            except _Timeout as t:
                if t.token is not tok:
                    raise
                errors.append(TimeoutError(f"no answer within {self.budget:g} s"))
            except Exception as exc:
                errors.append(exc)
            return None

        def flat(sols, v):
            if isinstance(sols, dict):
                return [sols.get(v)]
            if isinstance(sols, list) and sols and isinstance(sols[0], dict):
                return [d[v] for d in sols if v in d]
            return sols

        if isinstance(eq, Eq) and not unknown.is_real:
            poly = None
            excluded = []
            try:
                f = eq.lhs - eq.rhs
                if f.is_polynomial(unknown):
                    poly = sympy.Poly(f, unknown)
                elif f.is_rational_function(unknown):          # 1 + 6/(s(s+1)(s+2)) = 0
                    num, den = sympy.together(f).as_numer_denom()
                    if num.is_polynomial(unknown) and den.has(unknown):
                        poly = sympy.Poly(num, unknown)
                        excluded = [r for r in sympy.roots(sympy.Poly(den, unknown))]
                        eq = Eq(num, 0)
            except Exception:
                poly = None
            if poly is not None and poly.degree() >= 2:
                if poly.degree() > 8 and not (poly.free_symbols - {unknown}):
                    reals = sorted(set(sympy.real_roots(poly)), key=float)
                    if not reals:
                        notes.append(f"no real solution; the {poly.degree()} complex roots, numerically")
                        return list(poly.nroots(n=self.precision))
                    notes.append(f"{len(reals)} real root{'s' if len(reals) != 1 else ''} shown; the other "
                                 f"{poly.degree() - len(sympy.real_roots(poly))} roots are complex")
                    return reals
                allr = flat(attempt(eq, unknown), unknown)
                if isinstance(allr, list) and allr:
                    allr = [r for r in allr if not any(sympy.simplify(r - e) == 0 for e in excluded)]
                    reals = [r for r in allr if getattr(r, "is_real", None) is True]
                    cplx = [r for r in allr if getattr(r, "is_real", None) is False]
                    other = [r for r in allr if r not in reals and r not in cplx]
                    if reals and cplx:
                        notes.append(f"{len(reals)} real and {len(cplx)} complex solution{'s' if len(cplx) != 1 else ''}")
                    elif cplx and not reals:
                        notes.append("no real solution; complex solutions shown")
                    return reals + other + cplx
        if isinstance(eq_r, Eq):             # e^{-1000t} = …: solve expands it as a degree-1000 polynomial
            try:
                nonpoly = not (eq_r.lhs - eq_r.rhs).is_polynomial(real)
            except Exception:
                nonpoly = False
            if nonpoly:
                tok = None
                quick = None
                try:
                    with time_limit(min(3.0, self.budget)) as tok:
                        quick = sympy.solveset(eq_r, real, sympy.S.Reals)
                except _Timeout as t:
                    if t.token is not tok:
                        raise
                except Exception:
                    quick = None
                if isinstance(quick, sympy.FiniteSet) and quick:
                    vals = self._verify_inverse_trig(eq_r, real, list(quick), notes)
                    if not vals:
                        notes.append("no value satisfies the equation")
                        return []
                    try:
                        vals.sort(key=float)
                    except TypeError:
                        pass
                    return [v.xreplace({real: unknown}) for v in vals]
        sols = flat(attempt(eq_r, real), real)
        if isinstance(sols, Basic):          # an inequality's solution set, or a relation
            return sols.xreplace({real: unknown})
        if sols and isinstance(sols, list):
            sols = self._verify_inverse_trig(eq_r, real, sols, notes)
            if not sols:
                return []
        if sols:
            sols = [s.xreplace({real: unknown}) if isinstance(s, Basic) else s for s in sols]
            self._periodic_note(eq_r, real, notes)
            return sols
        declared = self._reg("_openlab_assume").get(unknown.name)
        if sols == [] and declared:
            notes.append(f"no solution with {unknown} {', '.join(declared)} (as declared)")
            return []
        if any(isinstance(e, TimeoutError) for e in errors):     # no closed form in sight: go numeric
            roots = self._nsolve(eq, unknown)
            if roots:
                notes.append(f"no closed form within {self.budget:g} s; numeric roots from starting points "
                             f"in [-10, 10] and beyond (there may be others)")
                return roots
        comp = flat(attempt(eq, unknown), unknown)
        if isinstance(comp, Basic):
            return comp
        if comp:
            notes.append("no real solution; complex solutions shown")
            return comp
        tok = None
        try:
            with time_limit(self.budget) as tok:
                setsol = sympy.solveset(eq_r, real, sympy.S.Reals)
            if setsol.has(sympy.ConditionSet):
                roots = self._nsolve(eq, unknown)
                if roots:
                    notes.append("no closed form; numeric roots from several starting points (there may be others)")
                    return roots
                if isinstance(setsol, sympy.Complement) and isinstance(setsol.args[0], sympy.ConditionSet) \
                        and isinstance(setsol.args[1], sympy.FiniteSet):     # sin x / x = 1
                    cs, gone = setsol.args
                    inner = cs.condition.xreplace({cs.sym: unknown})
                    if isinstance(inner, Eq):
                        cands = self._nsolve(inner, unknown)
                        if cands and all(any(abs(c - g) < 1e-9 for g in gone.xreplace({real: unknown})) for c in cands):
                            notes.append(f"no solution: the only candidate{'s' if len(cands) > 1 else ''} found "
                                         f"({', '.join(map(interp.pyform, cands))}) {'are' if len(cands) > 1 else 'is'} "
                                         f"excluded (the equation is undefined there)")
                            return []
                notes.append("no closed-form solution found")
                return setsol.xreplace({real: unknown})
            if isinstance(setsol, sympy.FiniteSet):
                return [v.xreplace({real: unknown}) for v in setsol]
            if setsol != sympy.S.EmptySet:
                return setsol.xreplace({real: unknown})
        except _Timeout as t:
            if t.token is not tok:
                raise
        except Exception as exc:
            errors.append(exc)
        if sols is None and comp is None:
            roots = self._nsolve(eq, unknown)
            if roots:
                notes.append("no closed form; numeric roots from several starting points (there may be others)")
                return roots
            raise SemanticsError("could not solve: " + "; ".join(f"{type(e).__name__}: {e}" for e in errors)[:300])
        return []

    @staticmethod
    def _verify_inverse_trig(eq, var, sols, notes):
        """arcsin x = 2 has no solution: a candidate is kept only if it satisfies the equation."""
        if not eq.has(sympy.asin, sympy.acos, sympy.atan, sympy.acot):
            return sols
        kept = []
        for s_ in sols:
            try:
                r_ = complex((eq.lhs - eq.rhs).subs(var, s_).evalf(30))
            except (TypeError, ValueError):
                kept.append(s_)
                continue
            if abs(r_) < 1e-12:
                kept.append(s_)
        if len(kept) < len(sols):
            notes.append("candidates outside the range of the inverse trig function were dropped")
        return kept

    def _periodic_note(self, eq, var, notes):
        if not eq.has(sympy.sin, sympy.cos, sympy.tan, sympy.cot, sympy.sec, sympy.csc):
            return
        tok = None
        try:
            with time_limit(3) as tok:
                general = sympy.solveset(eq, var, sympy.S.Reals)
            if general.has(sympy.ImageSet):
                notes.append(f"principal solutions shown; general solution: {self._general_text(general, var)}")
        except _Timeout as t:
            if t.token is not tok:
                raise
        except Exception:
            pass

    def _bind_distribution(self, rel: Rel, notes):
        import sympy.stats as st
        _, X, dist = rel.args
        if not isinstance(X, Symbol):
            raise UnsupportedError(r"`\sim` needs a name on the left: `X \sim N(0, 1)`")
        if not isinstance(dist, Apply) or not isinstance(dist.args[0], Symbol):
            raise UnsupportedError(r"`\sim` is only executable with a distribution on the right",
                                   hint=r"e.g. `X \sim \mathcal{N}(\mu, \sigma^2)`, `\operatorname{Bin}(n, p)`, `\mathrm{Poisson}(\lambda)`, `U(a, b)`, `\mathrm{Exp}(\lambda)`.")
        head = dist.args[0]
        dargs = list(dist.args[1:])
        if dargs and isinstance(dargs[-1], CallMod):             # \chi^2(3)
            mod = dargs.pop()
            if not (head.name in ("chi", "chi2") and str(mod.args[0]) == "power" and mod.args[1] == 2):
                raise UnsupportedError(f"`{interp.pyform(dist)}` is not a distribution openlab knows",
                                       hint=r"e.g. `X \sim \chi^2(3)`, `\mathcal{N}(0, 1)`, `\operatorname{Bin}(10, 0.5)`.")
            head = Symbol("chi2")
        params, subs = [], {}
        for p_ in dargs:
            r, s = self.resolve(p_, set(), notes)
            params.append(self.evaluate(r, notes, simplify=False, subs=s))
            subs.update(s)
        name = head.name
        table = {
            ("N", "Normal", "Gauss", "Gaussian", "mathcalN"): lambda n, mu, var: st.Normal(n, mu, sympy.sqrt(var)),
            ("U", "Uniform", "Unif"): lambda n, a, b: st.Uniform(n, a, b),
            ("Bin", "Binomial", "B", "Binom"): lambda n, k, p: st.Binomial(n, k, p),
            ("Poisson", "Pois", "Po"): lambda n, lam: st.Poisson(n, lam),
            ("Exp", "Exponential"): lambda n, lam: st.Exponential(n, lam),
            ("Bernoulli", "Ber", "Bern"): lambda n, p: st.Bernoulli(n, p),
            ("Geom", "Geometric"): lambda n, p: st.Geometric(n, p),
            ("Beta",): lambda n, a, b: st.Beta(n, a, b),
            ("Gamma",): lambda n, k, th: st.Gamma(n, k, th),
            ("Cauchy",): lambda n, a, b: st.Cauchy(n, a, b),
            ("Laplace",): lambda n, a, b: st.Laplace(n, a, b),
            ("LogNormal", "Lognormal"): lambda n, mu, s: st.LogNormal(n, mu, s),
            ("t", "StudentT", "Student"): lambda n, nu: st.StudentT(n, nu),
            ("chi2", "ChiSquared", "chi"): lambda n, k: st.ChiSquared(n, k),
        }
        ctor = next((f for names, f in table.items() if name in names), None)
        if ctor is None:
            raise UnsupportedError(f"unknown distribution `{name}`",
                                   hint="known: N/Normal, U/Uniform, Bin/Binomial, Poisson, Exp, Bernoulli, Geom, Beta, Gamma, Cauchy, Laplace, LogNormal, t, chi2.")
        try:
            rv = ctor(X.name, *params)
        except (TypeError, ValueError) as exc:
            raise SemanticsError(f"{name} takes different parameters: {exc}") from None
        if name in ("N", "Normal", "Gauss", "Gaussian", "mathcalN"):
            var = params[1] if len(params) > 1 else None
            raw_var = dargs[1] if len(dargs) > 1 else None
            squared = (isinstance(raw_var, (Pow, CallPow)) and raw_var.args[-1] == 2) or \
                (isinstance(raw_var, Apply) and raw_var.args and isinstance(raw_var.args[-1], CallMod)
                 and str(raw_var.args[-1].args[0]) == "power" and raw_var.args[-1].args[1] == 2)
            if squared or var == 1:
                pass                                   # N(100, 15^2), N(0, 1): nothing to misread
            elif var is not None and getattr(var, "is_number", False):
                sd = interp.pyform(sympy.sqrt(var))
                notes.append(f"N(μ, σ²): {interp.pyform(var)} is read as the variance, so σ = {sd}; write "
                             f"N({interp.pyform(params[0])}, {interp.pyform(var)}^2) if you meant σ = {interp.pyform(var)}")
            else:
                notes.append("N(μ, σ²): the second argument is read as the variance")
        elif name in ("Exp", "Exponential") and params:
            notes.append(f"Exp(λ): {interp.pyform(params[0])} is read as the rate λ (mean 1/λ)")
        self.ns[X.name] = rv
        self._reg("_openlab_dists")[X.name] = dist
        notes.append(f"{X.name} is now a random variable (sympy.stats)")
        return rv, X.name, subs

    @staticmethod
    def _indefinite(stmt, value) -> bool:
        r = stmt.rhs
        return (isinstance(r, Integral) and all(len(lim) == 1 for lim in r.limits)
                and isinstance(value, Basic) and not value.has(Integral))

    def _zero_based(self, val, notes):
        """``\\sum_{k=0}^{N-1} x_k``: a sum starting at 0 counts data entries from 0."""
        def fix(node):
            var, lo, hi = node.limits[0]
            rep = {}
            if not any(ix.indices == (var,) for ix in node.function.atoms(sympy.Indexed)):
                return node          # \sum_{k=0}^{2} x_{k+1} counts from 1 already
            for ix in node.function.atoms(sympy.Indexed):
                base = coerce(self._lookup(str(ix.base.label)))
                if isinstance(base, MatrixBase) and len(ix.indices) == 1 and ix.indices[0].has(var):
                    rep[ix] = ix.base[ix.indices[0] + 1]
            if not rep:
                return node
            notes.append(f"the sum starts at {var} = 0, so its first term uses the first entry")
            return node.func(node.function.xreplace(rep), *node.limits)
        if not isinstance(val, Basic) or not val.has(Sum, Product):
            return val
        return val.replace(lambda n: isinstance(n, (Sum, Product)) and len(n.limits) == 1
                           and len(n.limits[0]) == 3 and n.limits[0][1] == 0, fix)

    def _realify(self, pre, val, notes):
        """re()/im() noise from treating a variable as complex (``\\frac{d}{dx} \\ln|x|``): redo
        the step with the variables real, unless the input itself is about complex numbers."""
        if not (isinstance(val, Expr) and val.has(sympy.re, sympy.im)) or \
                pre.has(sympy.re, sympy.im, sympy.I, sympy.arg, sympy.conjugate):
            return val
        free = [s for s in pre.free_symbols if isinstance(s, Symbol) and not s.is_real]
        if not free:
            return val
        rep = {s: Symbol(s.name, real=True) for s in free}
        try:
            alt = self._doit(pre.xreplace(rep), notes)
        except OpenlabError:
            return val
        if isinstance(alt, Basic) and not alt.has(sympy.re, sympy.im):
            notes.append(f"{', '.join(map(str, free))} treated as real")
            return alt.xreplace({v: k for k, v in rep.items()})
        return val

    @staticmethod
    def _general_text(general, var) -> str:
        """``Union(ImageSet(Lambda(_n, 2*_n*pi), Integers), …)`` -> ``x = 2πn or x = 2πn + π (n ∈ ℤ)``."""
        n = Symbol("n", integer=True)
        parts = []
        for piece in (general.args if isinstance(general, sympy.Union) else [general]):
            if isinstance(piece, sympy.ImageSet) and piece.base_sets == (sympy.S.Integers,):
                lam = piece.lamda
                parts.append(f"{var} = {interp.pyform(lam.expr.xreplace({lam.variables[0]: n}))}")
            elif isinstance(piece, sympy.FiniteSet):
                parts.extend(f"{var} = {interp.pyform(e)}" for e in piece)
            else:
                return interp.pyform(general)
        return " or ".join(parts) + " (n an integer)"

    def _nsolve(self, eq, unknown, span=None) -> list:
        """Numeric roots of a one-unknown equation from several starting points (spread over
        ``span`` when given). A candidate is kept only if the equation's residual is tiny at a
        higher precision; candidates closer than 1e-3 are one root (the best-fitting one is
        kept), so a slowly converging multiple root does not come back as a cloud of values."""
        if not isinstance(eq, Eq) or eq.free_symbols - {unknown}:
            return []
        f = eq.lhs - eq.rhs
        if span is not None:
            lo, hi = span
            starts = [lo + (hi - lo) * k / 24 for k in range(25)]
        else:
            starts = [0, 1, -1, 0.5, -0.5, 0.01, 0.02, 0.05, 0.1, 0.2] \
                + [k / 2 for k in range(-20, 21) if k not in (0, 1, -1, 2, -2)] + [20, -20, 50, -50, 100]
        import mpmath
        try:
            fl = sympy.lambdify(unknown, f, "mpmath")
        except Exception:
            return []
        dps = max(self.precision, 15) + 10
        cands: list = []
        tok = None
        try:
            with time_limit(6) as tok, mpmath.workdps(dps):
                for x0 in starts:
                    try:
                        r = mpmath.findroot(fl, mpmath.mpf(x0))
                        if isinstance(r, mpmath.mpc):
                            if abs(r.imag) > 1e-10 * max(1, abs(r)):
                                continue
                            r = r.real
                        res = abs(fl(r))
                    except Exception:        # no convergence from here, a pole, a domain error
                        continue
                    if abs(r) < 1e-12:
                        r, res = mpmath.mpf(0), abs(fl(mpmath.mpf(0)))
                    if res <= mpmath.mpf(10) ** (-(dps - 5)) * max(1, abs(r)):
                        cands.append((r, res))
        except _Timeout as t:
            if t.token is not tok:
                raise
        if span is None and not cands:             # bracket sign changes on a wide, log-spaced grid
            grid = sorted({sgn * 10 ** (k / 4) for k in range(-20, 13) for sgn in (1, -1)})
            tok3 = None
            try:
                with time_limit(4) as tok3, mpmath.workdps(dps):
                    vals = []
                    for g in grid:
                        try:
                            v = fl(mpmath.mpf(g))
                            vals.append((g, v if not isinstance(v, mpmath.mpc) or abs(v.imag) < 1e-20 else None))
                        except Exception:
                            vals.append((g, None))
                    for (a, fa), (b, fb) in zip(vals, vals[1:]):
                        if fa is None or fb is None:
                            continue
                        fa, fb = mpmath.re(fa), mpmath.re(fb)
                        if fa == 0 or fa * fb < 0:
                            try:
                                r = mpmath.findroot(fl, (mpmath.mpf(a), mpmath.mpf(b)), solver="anderson")
                                if isinstance(r, mpmath.mpc):
                                    r = r.real
                                if min(a, b) <= r <= max(a, b):
                                    cands.append((r, abs(fl(r))))
                            except Exception:
                                pass
            except _Timeout as t:
                if t.token is not tok3:
                    raise
            cands = [(r, res) for r, res in cands if res <= mpmath.mpf(10) ** (-(dps - 8)) * max(1, abs(r))]
        roots: list = []
        for r, res in sorted(cands, key=lambda c: c[1]):          # best fit first
            if all(abs(r - q) > 1e-3 * max(1, abs(q)) for q in roots):
                roots.append(r)
        out = [sympy.S.Zero if r == 0 else sympy.Float(r, self.precision) for r in roots]
        return sorted(out, key=float)

    def _domain_set(self, domain, unknown, notes):
        """The set of values a restriction (``0 \\le x < 2\\pi``, ``x \\in [0, \\pi]``, ``n > 0``)
        allows for ``unknown``; ``None`` when it is about another name."""
        dom, _ = self.resolve(domain, set(), notes)
        dom = self._resolve_markers(dom, notes)
        real = Symbol(unknown.name, real=True)
        dom = dom.xreplace({s: real for s in dom.free_symbols if isinstance(s, Symbol) and s.name == unknown.name})
        if isinstance(dom, sympy.Contains):
            return dom.args[1] if dom.args[0] == real and isinstance(dom.args[1], sympy.Set) else None
        if isinstance(dom, sympy.Set):             # \text{on} [0, 2\pi]
            return dom
        if isinstance(dom, (Relational, sympy.And, sympy.Or)) and dom.free_symbols == {real}:
            try:
                return dom.as_set()
            except Exception:
                return None
        return None

    def _solve_in(self, eq, unknown, domain, notes):
        """Every solution in a stated domain: ``\\sin x = \\frac12, 0 \\le x < 2\\pi`` -> π/6, 5π/6."""
        S = self._domain_set(domain, unknown, notes)
        if S is None:
            notes.append(f"the condition {interp.pyform(domain)} is not a range for {unknown}; ignored")
            return self._solve(eq, unknown, notes)
        where = interp.pyform(S)
        real = Symbol(unknown.name, real=True)
        eq_r = eq.xreplace({unknown: real})
        sol = None
        tok = None
        try:
            with time_limit(self.budget) as tok:
                gen = sympy.solveset(eq_r, real, sympy.S.Reals)
                if gen.has(sympy.ImageSet):      # every family, then the range (solveset over the
                    sol = gen.intersect(S)       # interval can drop families: 2sin²x − sin x − 1 = 0)
                if not isinstance(sol, sympy.FiniteSet):
                    sol = sympy.solveset(eq_r, real, S)
        except _Timeout as t:
            if t.token is not tok:
                raise
        except Exception:
            sol = None
        if isinstance(sol, sympy.FiniteSet):
            vals = list(sol)
            try:
                vals.sort(key=lambda v: float(v))
            except TypeError:
                pass
            notes.append(f"solutions with {unknown} in {where}")
            return [v.xreplace({real: unknown}) for v in vals]
        if sol is sympy.S.EmptySet:
            notes.append(f"no solution with {unknown} in {where}")
            return []
        if isinstance(sol, sympy.Set) and not isinstance(sol, sympy.ConditionSet) and not sol.has(sympy.ImageSet):
            return sol.xreplace({real: unknown})
        if isinstance(S, sympy.Interval) and S.inf.is_finite and S.sup.is_finite \
                and not (eq.free_symbols - {unknown}):
            roots = [r for r in self._nsolve(eq, unknown, span=(float(S.inf), float(S.sup)))
                     if S.contains(r) is sympy.true]
            notes.append(f"numeric roots with {unknown} in {where} (from 25 starting points; there may be others)")
            return roots
        sols = self._solve(eq, unknown, notes)
        if isinstance(sols, list):
            kept = [s for s in sols if S.contains(s) is not sympy.false]
            notes.append(f"kept the solutions with {unknown} in {where}")
            return kept
        if isinstance(sols, sympy.Set):
            return sols.intersect(S)
        return sols

    def _pk(self, key: str):
        reg = self._reg("_openlab_events")
        if key in reg:
            return reg[key]
        if f"not({key})" in reg:
            return 1 - reg[f"not({key})"]
        if key.startswith("not(") and key[4:-1] in reg:
            return 1 - reg[key[4:-1]]
        return None

    def _sign_decl(self, rhs):
        """``\\lambda > 0`` on its own (an unbound name compared with 0) is an assumption."""
        kinds = {sympy.StrictGreaterThan: {"positive": True}, sympy.GreaterThan: {"nonnegative": True},
                 sympy.StrictLessThan: {"negative": True}, sympy.LessThan: {"nonpositive": True}}
        if type(rhs) in kinds and isinstance(rhs.lhs, Symbol) and rhs.rhs == 0 \
                and self._lookup(rhs.lhs.name) is None and rhs.lhs not in self.solved:
            return rhs.lhs, kinds[type(rhs)]
        return None

    def _recurrence_base(self, name, rhs):
        """``a_n = a_{n-1} + 2`` / ``a_{n+1} = 2 a_n``: the base name if this is a recurrence."""
        m = re.fullmatch(r"([A-Za-z]+)_(?:\{(.+)\}|([A-Za-z0-9]))", name)
        if not m or not re.search(r"[A-Za-z]", m.group(2) or m.group(3)):
            return None
        base = m.group(1)
        if isinstance(coerce(self._lookup(base)), MatrixBase):
            return None
        same = [s for s in rhs.free_symbols if isinstance(s, Symbol) and s.name != name
                and re.fullmatch(rf"{base}_(?:\{{.+\}}|[A-Za-z0-9])", s.name)]
        if same and all(self._lookup(q.name) is not None for q in same):
            return None          # \omega_d = \omega_n … with \omega_n bound: a formula
        letters = set(re.findall(r"[A-Za-z]", m.group(2) or m.group(3)))
        if not any(letters & set(re.findall(r"[A-Za-z]", q.name[len(base) + 1:])) for q in same):
            return None          # v_f = v_0 + a t: final and initial values, not a recurrence
        return base if same else None

    def _solve_recurrence(self, stmt, notes):
        base = self._recurrence_base(stmt.lhs.name, stmt.rhs)
        a = Function(base)

        def idx_of(nm):
            m = re.fullmatch(rf"{base}_(?:\{{(.+)\}}|(.+))", nm)
            return parse_expr(m.group(1) or m.group(2))[0]
        lhs_idx = idx_of(stmt.lhs.name)
        letters = sorted((s for s in lhs_idx.free_symbols if isinstance(s, Symbol)), key=str)
        n = letters[0]
        terms = {s: a(idx_of(s.name)) for s in stmt.rhs.free_symbols
                 if isinstance(s, Symbol) and re.fullmatch(rf"{base}_(?:\{{.+\}}|.+)", s.name)}
        rhs, subs = self.resolve(stmt.rhs, {s.name for s in terms} | {n.name}, notes)
        rhs = self.evaluate(rhs.xreplace(terms), notes, simplify=False, subs=subs)
        ics = {}
        for k in range(0, 6):
            v = coerce(self._lookup(f"{base}_{k}"))
            if v is not None and not isinstance(v, (MatrixBase, Lambda)):
                ics[a(k)] = v
        try:
            sol = sympy.rsolve(a(lhs_idx) - rhs, a(n), ics or None)
        except Exception:
            sol = None
        shift = sympy.simplify(lhs_idx - n)
        if sol is not None and ics and shift.is_Integer and not any(
                str(c).startswith("C") for c in sol.free_symbols if c != n):
            # rsolve can be wrong (it gives 0 for a_n = 2(n-1)(a_{n-1} + a_{n-2})): check it by iteration
            rec = (n, int(shift), rhs, {int(k.args[0]): v for k, v in ics.items()})
            try:
                start = max(rec[3]) + 1
                ok = all(sympy.simplify(self._iterate(base, rec, k, []) - sol.subs(n, k)) == 0
                         for k in range(min(rec[3]), start + 8))
            except OpenlabError:
                ok = False
            if ok:                       # numbers agree: now the recurrence itself, for a symbolic index
                sol_fn = Lambda(n, sol)
                lhs_c = sol.xreplace({n: lhs_idx - 0}).xreplace({n: n}) if False else sol_fn(lhs_idx)
                rhs_c = rhs.replace(lambda f: isinstance(f, AppliedUndef) and f.func == a,
                                    lambda f: sol_fn(f.args[0]))
                tok_c = None
                try:
                    with time_limit(min(6.0, self.budget)) as tok_c:
                        d = sympy.simplify(sympy.gammasimp(sympy.combsimp(sympy.expand_func(lhs_c - rhs_c))))
                    ok = d == 0
                except _Timeout as t:
                    if t.token is not tok_c:
                        raise
                    ok = False
                except Exception:
                    ok = False
                if ok:
                    notes.append("closed form verified against the recurrence for every index")
            if not ok:
                notes.append("the closed form a solver proposed could not be verified against the recurrence "
                             "for every index; terms are computed by iterating instead")
                sol = None
        if sol is None:
            if ics and shift.is_Integer:      # x_{n+1} = x_n - f(x_n)/f'(x_n): iterate for concrete terms
                self._reg("_openlab_recur")[base] = (n, int(shift), rhs, {int(k.args[0]): v for k, v in ics.items()})
                notes.append(f"no closed form; {base}_k for a whole number k is computed by iterating "
                             f"from the initial values")
                body = f"{stmt.lhs.name} = {interp.pyform(stmt.rhs)}"
                self._record_origin(base, stmt)
                return rhs, a(lhs_idx), subs, body, f"define {base} by its recurrence:", Kind.DEFINE
            raise SemanticsError("could not solve the recurrence (no closed form)",
                                 hint=f"give an initial value on a line above ({base}_0 = 1) to iterate it, "
                                      f"or a closed form `{base}_n = …`.")
        consts = sorted((c for c in sol.free_symbols if c != n and c.name.startswith("C")), key=str)
        if consts:
            notes.append(f"general solution; bind initial values such as {base}_0 = 1 on a line above to fix "
                         f"{', '.join(map(str, consts))}")
        else:
            self.ns[base] = Lambda(n, sol)
            self._reg("_openlab_seqs")[base] = True
            notes.append(f"solved with the initial values in the namespace; {base}_n is now a sequence")
        body = f"{stmt.lhs.name} = {interp.pyform(stmt.rhs)}"
        return [sol], a(n), subs, body, f"solve the recurrence for {base}_{n}:", Kind.SOLVE

    def _latex_out(self, kind, target, params, value, numeric, unknown, stmt) -> str:
        if kind is Kind.ASSUME:
            return r"\text{assume } " + L(value)
        if kind is Kind.ASSIGN:
            if target == "event":
                return rf"{L(stmt.lhs.args[0])}\left({L(stmt.lhs.args[1])}\right) = {L(value)}"
            if isinstance(target, tuple):
                return ", ".join(f"{L(Symbol(n))} = {L(v)}" for n, v in zip(target, value))
            if isinstance(value, Tuple) and stmt.rhs.has(PlusMinus):      # x = ± 2: two values, nothing bound
                return rf"{L(Symbol(target))} \in \left\{{{', '.join(L(v) for v in value)}\right\}}"
            if isinstance(value, Tuple):
                return rf"{L(Symbol(target))} = \left({', '.join(L(v) for v in value)}\right)"
            out = f"{L(Symbol(target))} = {L(value)}"
            if numeric is not None:
                out += rf" \approx {L(numeric)}"
            return out
        if kind is Kind.DEFINE and target == "recurrence":
            return rf"{L(stmt.lhs)} = {L(stmt.rhs)} \quad (\text{{terms by iteration}})"
        if kind is Kind.DEFINE:
            if params and isinstance(value, Lambda):
                args = ", ".join(L(p) for p in params)
                if isinstance(stmt.lhs, Symbol):
                    return f"{L(stmt.lhs)} = {L(value.expr)}"
                return f"{L(Symbol(target))}\\left({args}\\right) = {L(value.expr)}"
            dist = self._reg("_openlab_dists").get(target)
            return rf"{L(Symbol(target))} \sim {L(dist) if dist is not None else L(value)}"
        if kind is Kind.SOLVE:
            if isinstance(value, list):
                if not value:
                    return r"\text{no solution}"
                if isinstance(value[0], dict):
                    return r" \quad\text{or}\quad ".join(
                        r",\; ".join(f"{L(k)} = {L(v)}" for k, v in sol.items()) for sol in value) or r"\text{no solution}"
                parts = []
                for s in value:
                    a = self.approx(s)
                    if a is not None and isinstance(s, Basic) and sympy.count_ops(s) > 40:
                        parts.append(rf"{L(unknown)} \approx {L(a)}")       # the exact form is unreadable
                        continue
                    parts.append(f"{L(unknown)} = {L(s)}" + (rf" \approx {L(a)}" if a is not None else ""))
                return r" \quad\text{or}\quad ".join(parts)
            return f"{L(unknown)} \\in {L(value)}"
        if kind is Kind.RELATION and target == "rate":
            return rf"{L(stmt.lhs)} = {L(value)}" + (rf" \approx {L(numeric)}" if numeric is not None else "")
        if kind is Kind.RELATION:
            if value is sympy.true and self._qualifier:
                return rf"\text{{true {self._qualifier}}}"
            if isinstance(value, list):
                return r" \quad\text{or}\quad ".join(f"{L(unknown)} = {L(s)}" for s in value) or r"\text{no solution}"
            if unknown is not None and isinstance(value, sympy.Set):
                return f"{L(unknown)} \\in {L(value)}"
            return L(value)
        if kind is Kind.REPR:
            return L(value)
        if isinstance(value, Integer) and value.p.bit_length() > 200:
            digits = len(str(value.p)) if value.p.bit_length() < 14000 else int(value.p.bit_length() * 0.30103) + 1
            return rf"\approx {L(value)} \quad (\text{{exact integer, {digits:,} digits}})"
        out = f"= {L(value)}" + (r"^\circ" if getattr(stmt, "convert", None) == "deg" else "")
        if self._indefinite(stmt, value):
            out += " + C"
        if numeric is not None:
            out += rf" \approx {L(numeric)}"
        return out


_FIT_REF = re.compile(r"(beta|se|p)_\{(\w+),+(.+)\}(hat)?")      # \hat\beta_{m,\,x} reads as beta_{m,,x}hat


def _term_key(term: str) -> str:
    """A model term as a statement can spell it: `has\\_children[b]`, `has_children_b` and `haschildrenb` agree."""
    return re.sub(r"[^0-9a-z]", "", term.lower())


def fit_refer(name: str, terms: list) -> str:
    """How a statement names model ``name``'s estimates, shown on one of its terms."""
    t = next((t for t in terms if t != "(intercept)"), terms[0] if terms else "x")
    x = re.sub(r"[^0-9A-Za-z\[\]]+", "_", t).strip("_").replace("_", r"\_")
    return (rf"in a statement, \hat\beta_{{{name},\,{x}}} is the estimate of `{t}`, "
            rf"\mathrm{{se}}_{{{name},\,{x}}} and p_{{{name},\,{x}}} its se and p; any term works the same way")


def _cite_fit(err: OpenlabError, text: str, ns: dict) -> OpenlabError:
    """A statement that did not read but names a fitted model's term: say how to cite the estimate."""
    fits = ns.get("_openlab_fits") if isinstance(ns.get("_openlab_fits"), dict) else {}
    words = {_term_key(w) for w in re.findall(r"[A-Za-z][A-Za-z0-9_]*", (text or "").replace("\\_", "_"))}
    for m, rec in reversed(list(fits.items())):
        t = next((t["term"] for t in rec.get("terms", ()) if t["term"] != "(intercept)"
                  and _term_key(t["term"].split("[")[0]) in words), None)
        if t:
            x = re.sub(r"[^0-9A-Za-z\[\]]+", "_", t).strip("_").replace("_", r"\_")
            tip = rf"to cite a fitted estimate write \hat\beta_{{{m},\,{x}}} (model {m}, term {t})"
            err.hint = f"{err.hint}; {tip}" if err.hint else tip
            break
    return err


def _formula_error(text: str, ns: dict) -> SemanticsError:
    """`y \\sim x + z` handed in as a statement: say what to write instead."""
    rhs = re.split(r"\\sim(?![A-Za-z])", text, 1)[-1]
    term = re.sub(r"\\(?:text|mathrm|operatorname)\{([^{}]*)\}", r"\1", rhs.split("+")[0]).replace("\\_", "_").strip()
    fits = ns.get("_openlab_fits") if isinstance(ns.get("_openlab_fits"), dict) else {}
    name = next((n for n, r in reversed(list(fits.items()))
                 if any(_term_key(t["term"]) == _term_key(term) for t in r.get("terms", ()))), "NAME")
    return SemanticsError("a model formula is not a statement; " + fit_refer(name, [term or "x"])
                          + r" (fit the model first, in a study run; `\sim` here names no distribution)", src=text,
                          hint=r"to hand in an estimate: answer with it and check it, e.g. "
                               rf"\hat\beta_{{{name},\,x}} \approx -0.41")


def iter_run(src: str, ns: Optional[dict] = None, **opts) -> Iterator[Result]:
    """Execute a cell statement by statement, yielding each :class:`Result` as it lands."""
    ns = {} if ns is None else ns
    from . import keep as _keep
    reg_name = opts.pop("registry", None)
    src, directives = _keep.split(src)
    if reg_name:
        directives = {0: ("registry", reg_name), **directives}
    ex = Executor(ns, **opts)
    ex.cell_lines = src.splitlines()
    clauses = statements(src)
    keeps = {}
    for j, c in enumerate(clauses):
        c.text, k_ = _keep.strip_keep(c.text)
        if k_:
            keeps[j] = k_
    i = 0
    local_restore = None       # (lineno, assumptions, namespace, names bound) before a `where` clause

    def restore(lr):
        ns["_openlab_assume"] = lr[1]
        for name in lr[3]:                                  # `where r = 3` held for its line only
            if name in lr[2]:
                ns[name] = lr[2][name]
            else:
                ns.pop(name, None)

    def parse(c: Clause) -> Statement:
        return parse_statement(c.text, c.lineno, role=c.role, ics=c.ics, display=c.display, domain=c.domain)

    while i < len(clauses):
        c = clauses[i]
        yield from _keep.before(directives, c.lineno, ns)
        if local_restore is not None and c.lineno != local_restore[0]:
            restore(local_restore)                           # a `where` clause ends with its line
            local_restore = None
        if c.role == "local" and local_restore is None:
            local_restore = (c.lineno, dict(ex._reg("_openlab_assume")), dict(ns), set())
        step = 1
        extra: list = []
        if c.role in ("note", "markdown") or (c.role == "claim" and not _has_technique(c.text)):
            r = ex.note(c)
            if i in keeps:                              # a conjecture is kept as what it is: unproved
                r.notes.append(_keep.log(ns, keeps[i], c.text))
            ns["_openlab"] = r
            yield r
            i += 1
            continue
        old_digits = _set_digits(0)
        try:
            with time_limit(ex.timeout):
                if c.role == "claim":
                    r = ex.prove(c)
                    raise _Done
                stmt = parse(c)
                ode_group, j = [stmt], i + 1
                is_ode = any(ex._differentiated(e) for e in (stmt.lhs, stmt.rhs) if isinstance(e, Basic))
                while is_ode and j < len(clauses):
                    try:                           # a later line that does not parse ends the group
                        nxt = parse(clauses[j])
                    except OpenlabError:
                        break
                    if not ex.ode_coupled(ode_group, nxt):
                        break
                    ode_group.append(nxt)
                    j += 1
                if len(ode_group) > 1:
                    r = ex.execute_system(ode_group)
                    step = j - i
                else:
                    try:
                        r = ex.execute(stmt)
                        if c.ics and not ex._ode_done:     # `g(1) = 1` after a line that was not an ODE
                            extra = [ex.execute(parse_statement(ic, c.lineno)) for ic in c.ics]
                    except MultipleUnknowns as err:
                        group, j = [stmt], i + 1
                        while j < len(clauses):
                            try:
                                nxt = parse(clauses[j])
                            except OpenlabError:
                                break
                            if not ((ex.is_plain_equation(nxt) and ex.shares_unknowns(nxt, group))
                                    or ex.joins_system(nxt, group)):
                                break
                            group.append(nxt)
                            j += 1
                        if len(group) >= 2:
                            r = ex.execute_system(group)
                            step = j - i
                        elif getattr(err, "retry", False):      # nothing to join: solve what there is
                            ex.underdetermined_ok = True
                            try:
                                r = ex.execute(stmt)
                            finally:
                                ex.underdetermined_ok = False
                        else:
                            implicit = ex.implicit_followup(stmt, clauses[i + 1:i + 2], parse)
                            if implicit is None:
                                raise
                            r = implicit
                            step = 2
        except _Done:
            pass
        except OpenlabError as err:
            if isinstance(err, ParseError) and re.search(r"\\sim(?![A-Za-z])", c.text):   # `y \sim x`: a formula
                err = _formula_error(c.text, ns)
            elif isinstance(err, ParseError):
                err = _cite_fit(err, c.text, ns)
            err.lineno = err.lineno or c.lineno
            err.src = err.src or c.text
            _record_failed(ns, c.text)
            raise err
        except _Timeout:
            _record_failed(ns, c.text)
            raise SemanticsError(f"stopped after {ex.timeout:g} s", src=c.text, lineno=c.lineno,
                                 hint="simplify the input or give numeric values; raise the limit with -t 60.") from None
        except RecursionError:
            raise SemanticsError("the expression is too deeply nested to evaluate", src=c.text,
                                 lineno=c.lineno) from None
        except Exception as exc:  # anything SymPy throws becomes an openlab error, never a traceback
            raise SemanticsError(f"could not execute `{c.text}`: {type(exc).__name__}: {exc}",
                                 src=c.text, lineno=c.lineno) from exc
        finally:
            _set_digits(old_digits)
            if local_restore is not None and c.role == "local" and sys.exc_info()[0] is None:
                local_restore[3].update(k for k in (r.bound or {}) if isinstance(k, str))
            if local_restore is not None and (i + step >= len(clauses) or sys.exc_info()[0] is not None):
                restore(local_restore)
        if ex.strict:
            _strict_check(r, ns)
        if i in keeps:
            r.notes.append(_keep.log(ns, keeps[i], c.text))
        ns["_openlab"] = r
        yield r
        for x in extra:
            ns["_openlab"] = x
            yield x
        i += step
    if local_restore is not None:
        restore(local_restore)
    yield from _keep.before(directives, None, ns)


class _Done(Exception):
    """A claim was executed by the prover: skip the statement path."""


_DEF_HEAD = re.compile(r"^\s*(\\?[A-Za-z]+|\\(?:mathrm|operatorname|text)\{[A-Za-z]+\})(?:_\{?\w+\}?)?\s*(?:\([^()=]*\))?\s*:?=")


def _record_failed(ns, text: str):
    """A definition that failed: remember its name, so a later use is not silently a built-in."""
    m = _DEF_HEAD.match(text or "")
    if not m:
        return
    name = re.sub(r"^\\(?:mathrm|operatorname|text)\{|\}$", "", m.group(1)).lstrip("\\")
    reg = ns.get("_openlab_failed")
    if not isinstance(reg, dict):
        reg = ns["_openlab_failed"] = {}
    reg[name] = text.strip()[:80]


def _has_technique(text: str) -> bool:
    return bool(re.search(r"\\openlabnote\{[^{}]*(?:by\s+(?:strong\s+)?induction|by\s+algebra|identically|"
                          r"by\s+direct\s+computation|witness|by\s+exhaustion|by\s+exhaustive|by\s+cases\s+on)", text, re.I))


def _strict_check(r: Result, ns: Optional[dict] = None):
    """--strict: a statement that was only represented, or a check left undecided, is an error."""
    undecided = (r.kind in (Kind.RELATION, Kind.PROOF) and not isinstance(r.value, (BooleanAtom, list, sympy.Set))
                 and r.value is not None) or (r.kind is Kind.PROOF and r.value is None)
    if r.kind is Kind.REPR and re.search(r"\\sim(?![A-Za-z])", r.src or ""):
        err = _formula_error(r.src, ns or {})
        err.lineno = r.lineno
        raise err
    if r.kind is Kind.REPR or undecided:
        raise _cite_fit(SemanticsError(f"strict mode: `{r.src}` was not verified ({r.kind.name.lower()}"
                                       f"{', ' + interp.short(r.value) if r.value is not None else ''})",
                                       src=r.src, lineno=r.lineno,
                                       hint="rewrite it so it executes, prove it with a technique, or make it a "
                                            "note (a `>` line or \\text{…})."), r.src, ns or {})


def run_all(src: str, ns: Optional[dict] = None, **opts) -> list:
    return list(iter_run(src, ns, **opts))


def run(src: str, ns: Optional[dict] = None, **opts):
    """Execute LaTeX. One statement -> a :class:`Result`; several -> a list of them.

    ``opts``: ``numeric`` (force decimals), ``precision`` (digits, default 15),
    ``complex_i`` (read ``i`` as the imaginary unit, default True), ``solve_for`` (name of
    the unknown when an equation has several), ``simplify`` (default True), ``timeout``
    (seconds per statement, default 30)."""
    results = run_all(src, ns, **opts)
    if not results:
        raise ParseError("nothing to evaluate", src=src)
    return results[0] if len(results) == 1 else results
