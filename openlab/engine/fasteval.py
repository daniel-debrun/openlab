"""Exact evaluation of whole-number arithmetic in plain Python, for finite sums, set listings and
term-by-term searches. Only exact operations are compiled (integers, fractions, remainders,
comparisons, the number-theory indicators); anything else raises :class:`Unsupported` and the
caller falls back to SymPy, so a result never depends on which path computed it."""
from __future__ import annotations

import cmath
import math
from fractions import Fraction

import mpmath
import sympy
from sympy.core.relational import Relational

_CERT_DPS = 60
_CERT_EPS = mpmath.mpf(10) ** -45           # a decision closer than this is left to SymPy
_STATE = {"dps": 60}                        # the working precision of a certified evaluation
_LEVELS = (60, 150, 400, 1000, 2500)        # a comparison too close to call is recomputed at the next level


def _dps():
    return _STATE["dps"]


def _eps():
    return mpmath.mpf(10) ** -(_STATE["dps"] - 15)


class _Cert:
    """Certified mode: exact integers and fractions stay exact; transcendental values are mpmath numbers
    at 60 digits, and a comparison, floor or ceiling is decided only when the margin is far above the
    rounding error."""
    funcs = {sympy.log: mpmath.log, sympy.exp: mpmath.exp, sympy.sin: mpmath.sin, sympy.cos: mpmath.cos,
             sympy.tan: mpmath.tan, sympy.atan: mpmath.atan}
    consts = {sympy.pi: lambda: +mpmath.pi, sympy.E: lambda: +mpmath.e, sympy.EulerGamma: lambda: +mpmath.euler}


def _is_mp(v):
    return isinstance(v, (mpmath.mpf, mpmath.mpc))


def _mp(v):
    return mpmath.mpf(v.numerator) / v.denominator if isinstance(v, Fraction) else mpmath.mpf(v)


class Unsupported(Exception):
    pass


def _num(v):
    if isinstance(v, Fraction) and v.denominator == 1:
        return v.numerator
    return v


def _as_int(v, what):
    v = _num(v)
    if isinstance(v, bool) or not isinstance(v, int):
        raise Unsupported(f"{what} needs a whole number")
    return v


_REL = {sympy.Eq: lambda a, b: a == b, sympy.Ne: lambda a, b: a != b,
        sympy.StrictLessThan: lambda a, b: a < b, sympy.LessThan: lambda a, b: a <= b,
        sympy.StrictGreaterThan: lambda a, b: a > b, sympy.GreaterThan: lambda a, b: a >= b}


_FLOAT_FUNCS = {sympy.exp: cmath.exp, sympy.log: cmath.log, sympy.sin: cmath.sin, sympy.cos: cmath.cos,
                sympy.tan: cmath.tan, sympy.sinh: cmath.sinh, sympy.cosh: cmath.cosh, sympy.atan: cmath.atan}
_FLOAT_CONSTS = {sympy.pi: math.pi, sympy.E: math.e, sympy.EulerGamma: 0.5772156649015329, sympy.I: 1j}


def _real(v):
    if isinstance(v, complex) and abs(v.imag) <= 1e-300:
        return v.real
    return v


def nonneg_static(e) -> bool:
    """Is every value of e ≥ 0: 0/1 indicators, non-negative constants, products and sums of such."""
    if isinstance(e, (sympy.Integer, sympy.Rational)):
        return e >= 0
    if getattr(type(e), "_is_pick", False):
        return nonneg_static(e.args[1]) and nonneg_static(e.args[2])
    if type(e).__name__ in ("isprime",):
        return True
    if isinstance(e, (sympy.Mul, sympy.Add)):
        return all(nonneg_static(a) for a in e.args)
    if isinstance(e, sympy.Sum):
        return nonneg_static(e.function)
    return bool(getattr(e, "_openlab_nonneg", False))


def _piecewise_linear(e, names):
    """A map like the Collatz T -- pieces (a·x + b)/d chosen by x mod c = r -- as plain integer arithmetic.
    Returns a closure (with .direct(v) taking the value itself), or None when the shape does not fit."""
    xs = {s_ for s_ in e.free_symbols}
    if len(xs) != 1:
        return None
    x = next(iter(xs))
    if x.name not in names:
        return None
    pieces = []
    for val, cond in e.args:
        if cond is sympy.true:
            test = None
        elif isinstance(cond, sympy.Eq) and isinstance(cond.lhs, sympy.Mod) and cond.lhs.args[0] == x \
                and cond.lhs.args[1].is_Integer and cond.rhs.is_Integer:
            test = (int(cond.lhs.args[1]), int(cond.rhs))
        else:
            return None
        try:
            poly = sympy.Poly(sympy.together(val).as_numer_denom()[0], x)
            den = sympy.together(val).as_numer_denom()[1]
        except Exception:
            return None
        if not den.is_Integer or poly.degree() > 1 or not all(c_.is_Integer for c_ in poly.all_coeffs()):
            return None
        co = [int(c_) for c_ in poly.all_coeffs()]
        a_, b_ = (co[0], co[1]) if len(co) == 2 else (0, co[0])
        pieces.append((test, a_, b_, int(den)))
    nm = x.name

    def direct(v):
        if type(v) is not int:
            raise Unsupported("not a whole number")
        for test, a_, b_, d_ in pieces:
            if test is None or v % test[0] == test[1]:
                n_ = a_ * v + b_
                if d_ == 1:
                    return n_
                q_, r_ = divmod(n_, d_)
                return q_ if r_ == 0 else Fraction(n_, d_)
        raise Unsupported("no piece applies")

    def f(env):
        return direct(env[nm])
    f.direct = direct
    return f


def compile_expr(expr, names=(), hook=None, numeric=None):
    """-> f(env) computing ``expr`` with the letters in ``names`` taken from ``env`` (a dict).
    ``hook(e, names)`` may compile nodes this module does not know (openlab's calls and searches).
    ``numeric``: allow decimals, logarithms, exponentials (default: only when ``expr`` holds a decimal),
    so an exact input keeps an exact answer."""
    names = set(names)
    if numeric is None:
        numeric = isinstance(expr, sympy.Basic) and expr.has(sympy.Float)
    cert = (not numeric and isinstance(expr, sympy.Basic)
            and (expr.has(*_Cert.funcs) or any(expr.has(k) for k in _Cert.consts)))

    def c(e):
        if isinstance(e, bool):
            return lambda env: e
        if e is sympy.true:
            return lambda env: True
        if e is sympy.false:
            return lambda env: False
        if isinstance(e, sympy.Integer):
            v = int(e)
            return lambda env: v
        if isinstance(e, sympy.Rational):
            v = Fraction(int(e.p), int(e.q))
            return lambda env: v
        if numeric and isinstance(e, sympy.Float):
            v = float(e)
            return lambda env: v
        if cert and e in _Cert.consts:
            mk = _Cert.consts[e]

            def cst(env):
                with mpmath.workdps(_dps()):
                    return mk()
            return cst
        if cert and type(e) in _Cert.funcs and len(e.args) >= 1:
            fs_, op = [c(a) for a in e.args], _Cert.funcs[type(e)]

            def tf(env):
                with mpmath.workdps(_dps()):
                    vals = [_mp(x) if not _is_mp(x) else x for x in (f(env) for f in fs_)]
                    if op is mpmath.log and len(vals) == 2:
                        return mpmath.log(vals[0]) / mpmath.log(vals[1])
                    return op(*vals)
            return tf
        if numeric and e in _FLOAT_CONSTS:
            v = _FLOAT_CONSTS[e]
            return lambda env: v
        if numeric and type(e) in _FLOAT_FUNCS and len(e.args) == 1:
            fa, op = c(e.args[0]), _FLOAT_FUNCS[type(e)]
            return lambda env: _real(op(complex(fa(env))))
        if isinstance(e, sympy.Symbol):
            if e.name not in names:
                raise Unsupported(f"free letter {e}")
            nm = e.name
            return lambda env: env[nm]
        if isinstance(e, sympy.Add):
            fs = [c(a) for a in e.args]
            if numeric:
                return lambda env: _num(sum(f(env) for f in fs))

            def add(env):
                vals = [f(env) for f in fs]
                if all(type(v) is int for v in vals):     # the common case: whole numbers, no Fraction
                    return sum(vals)
                if any(_is_mp(v) for v in vals):
                    with mpmath.workdps(_dps()):
                        return mpmath.fsum(_mp(v) if not _is_mp(v) else v for v in vals)
                return _num(sum(Fraction(v) for v in vals))
            if all(nonneg_static(a) for a in e.args):   # a sum of terms ≥ 0: may stop early (Σ ≥ 1)
                add.stream = lambda env: (f(env) for f in fs)
                add.nonneg = True
            return add
        if isinstance(e, sympy.Mul):                  # Iverson factors first: [P][Q] = 0 without Q when P fails
            args_ = sorted(e.args, key=lambda a: 0 if getattr(type(a), "_is_pick", False) else 1)
            fs = [c(a) for a in args_]

            def mul(env):
                out = 1
                for f in fs:
                    v = f(env)
                    if _is_mp(v) or _is_mp(out):
                        with mpmath.workdps(_dps()):
                            out = (_mp(out) if not _is_mp(out) else out) * (_mp(v) if not _is_mp(v) else v)
                        continue
                    out = out * v if (type(v) is int and type(out) is int) or isinstance(v, (float, complex)) \
                        or isinstance(out, (float, complex)) else Fraction(out) * v
                    if out == 0:
                        return 0
                return _num(out) if isinstance(out, Fraction) else out
            return mul
        if isinstance(e, sympy.Pow):
            fb, fe = c(e.base), c(e.exp)

            def pw(env):
                b, x = fb(env), _num(fe(env))
                if _is_mp(b) or _is_mp(x) or (cert and not isinstance(x, int)):
                    with mpmath.workdps(_dps()):
                        bb, xx = (b if _is_mp(b) else _mp(b)), (x if _is_mp(x) else _mp(x))
                        if bb <= 0 and not isinstance(x, int):
                            raise Unsupported("a power of a non-positive number")
                        return bb ** xx
                if not isinstance(x, int):
                    if numeric:
                        return _real(complex(b) ** complex(x))
                    raise Unsupported("a fractional power")
                if x < 0:
                    if b == 0:
                        raise Unsupported("division by zero")
                    return _num(Fraction(1) / Fraction(b) ** (-x))
                if abs(x) > 10_000:
                    raise Unsupported("a huge power")
                if isinstance(b, (float, complex)):
                    return b ** x
                return _num(Fraction(b) ** x)
            return pw
        if isinstance(e, sympy.Mod):
            fa, fb = c(e.args[0]), c(e.args[1])

            def md(env):
                a, b = fa(env), fb(env)
                if b == 0 or isinstance(a, complex) or isinstance(b, complex):
                    raise Unsupported("remainder by zero")
                if isinstance(a, float) or isinstance(b, float):
                    return a % b
                return _num(Fraction(a) % Fraction(b))
            return md
        if isinstance(e, (sympy.floor, sympy.ceiling)):
            fa = c(e.args[0])
            op = math.floor if isinstance(e, sympy.floor) else math.ceil

            def fl(env):
                v = fa(env)
                if _is_mp(v):
                    if not isinstance(v, mpmath.mpf):
                        raise Unsupported("floor of a complex number")
                    near = mpmath.nint(v)
                    if abs(v - near) <= _eps() * max(1, abs(v)):
                        raise Unsupported("too close to a whole number to decide")
                    return int(mpmath.floor(v) if op is math.floor else mpmath.ceil(v))
                return op(Fraction(v))
            return fl
        if isinstance(e, sympy.Abs):
            fa = c(e.args[0])
            return lambda env: abs(fa(env))
        if isinstance(e, sympy.factorial):
            fa = c(e.args[0])

            def fac(env):
                v = _as_int(fa(env), "a factorial")
                if v < 0 or v > 5000:
                    raise Unsupported("factorial out of range")
                return math.factorial(v)
            return fac
        if isinstance(e, sympy.binomial):
            fa, fb = c(e.args[0]), c(e.args[1])

            def bn(env):
                a, b = _as_int(fa(env), "a binomial"), _as_int(fb(env), "a binomial")
                if a < 0:
                    raise Unsupported("binomial with a negative top")
                return math.comb(a, b) if 0 <= b else 0
            return bn
        if isinstance(e, Relational) and type(e) in _REL:
            fa, fb, op = c(e.lhs), c(e.rhs), _REL[type(e)]

            flip = {sympy.GreaterThan: sympy.LessThan, sympy.StrictGreaterThan: sympy.StrictLessThan,
                    sympy.LessThan: sympy.GreaterThan, sympy.StrictLessThan: sympy.StrictGreaterThan}
            kind_ = type(e)
            s_side, c_side = (fa, fb) if getattr(fa, "nonneg", False) else (fb, fa)
            if getattr(s_side, "nonneg", False) and getattr(s_side, "stream", None) is not None:
                k2 = kind_ if s_side is fa else flip.get(kind_)
                if k2 in (sympy.GreaterThan, sympy.StrictGreaterThan):      # Σ ≥ c: stop once reached

                    def rel_sc(env):
                        c_ = c_side(env)
                        acc = 0
                        for t_ in s_side.stream(env):
                            acc = acc + t_
                            if (acc >= c_) if k2 is sympy.GreaterThan else (acc > c_):
                                return True
                        return (acc >= c_) if k2 is sympy.GreaterThan else (acc > c_)
                    return rel_sc

            def rel(env):
                a_, b_ = fa(env), fb(env)
                if not (_is_mp(a_) or _is_mp(b_)):
                    return op(a_, b_)
                outer = _STATE["dps"]
                try:
                    for level in [lv for lv in _LEVELS if lv >= outer]:
                        _STATE["dps"] = level
                        if level != outer:
                            a_, b_ = fa(env), fb(env)            # recomputed at the higher precision
                        with mpmath.workdps(level):
                            a2, b2 = (a_ if _is_mp(a_) else _mp(a_)), (b_ if _is_mp(b_) else _mp(b_))
                            if isinstance(a2, mpmath.mpc) or isinstance(b2, mpmath.mpc):
                                continue                    # a log of a rounding-negative difference: more digits
                            if abs(a2 - b2) > _eps() * max(1, abs(a2), abs(b2)):
                                return op(a2, b2)
                    raise Unsupported(f"the two sides agree to {_LEVELS[-1] - 15} digits: undecided")
                finally:
                    _STATE["dps"] = outer
            return rel
        if isinstance(e, sympy.And):
            fs = [c(a) for a in e.args]
            return lambda env: all(f(env) for f in fs)
        if isinstance(e, sympy.Or):
            fs = [c(a) for a in e.args]
            return lambda env: any(f(env) for f in fs)
        if isinstance(e, sympy.Not):
            fa = c(e.args[0])
            return lambda env: not fa(env)
        if isinstance(e, sympy.Piecewise):
            fastpw = _piecewise_linear(e, names)
            if fastpw is not None:
                return fastpw
            pairs = [(c(ex), c(cd)) for ex, cd in e.args]

            def pwise(env):
                for fx, fc in pairs:
                    if fc(env):
                        return fx(env)
                raise Unsupported("no piece applies")
            return pwise
        if isinstance(e, (sympy.Sum, sympy.Product)) and len(e.limits) >= 1:
            is_sum = isinstance(e, sympy.Sum)
            limits = list(e.limits)
            inner_names = names | {lim[0].name for lim in limits}
            body = compile_expr(e.function, inner_names, hook, numeric)
            lims = [(lim[0].name, compile_expr(lim[1], inner_names, hook, numeric),
                     compile_expr(lim[2], inner_names, hook, numeric)) for lim in limits]

            def big(env):
                def run(i, env_):
                    if i < 0:
                        return body(env_)
                    nm, flo, fhi = lims[i]
                    lo, hi = _as_int(flo(env_), "a bound"), _as_int(fhi(env_), "a bound")
                    if hi - lo > 200_000_000:
                        raise Unsupported("too many terms")
                    acc = 0 if is_sum else 1
                    e2 = dict(env_)
                    for v in range(lo, hi + 1):
                        e2[nm] = v
                        t = run(i - 1, e2)
                        acc = acc + t if is_sum else acc * t
                    return acc
                out = run(len(lims) - 1, env)              # sympy: the first limit is the innermost
                return _num(out) if isinstance(out, Fraction) else out
            if is_sum and len(lims) == 1:
                nm1, flo1, fhi1 = lims[0]

                def stream(env):
                    lo, hi = _as_int(flo1(env), "a bound"), _as_int(fhi1(env), "a bound")
                    e2 = dict(env)
                    for v in range(lo, hi + 1):
                        e2[nm1] = v
                        yield body(e2)
                big.stream, big.nonneg = stream, nonneg_static(e.function)
            return big
        if getattr(type(e), "_is_pick", False):       # Pick(cond, value, neutral): value only where cond
            fc, fv, fn_ = c(e.args[0]), c(e.args[1]), c(e.args[2])
            return lambda env: fv(env) if fc(env) else fn_(env)
        part = getattr(type(e), "_frac_part", None)   # numer / denom of an exact fraction
        if part is not None:
            fa = c(e.args[0])

            def fr(env):
                v = fa(env)
                if not isinstance(v, (int, Fraction)):
                    raise Unsupported("numer/denom of a non-rational")
                v = Fraction(v)
                return v.denominator if part == 1 else v.numerator
            return fr
        fn = getattr(type(e), "_pyfn", None)          # isprime, totient, gcd, … (openlab's own)
        if fn is not None:
            fs = [c(a) for a in e.args]
            low = getattr(type(e), "_low", None)

            def call(env):
                vals = [_as_int(f(env), type(e).__name__) for f in fs]
                if low is not None and vals[0] < low:
                    raise Unsupported("argument below the function's range")
                return fn(*vals)
            return call
        if hook is not None:
            f = hook(e, names)
            if f is not None:
                return f
        if numeric and isinstance(e, sympy.Abs):
            fa = c(e.args[0])
            return lambda env: abs(fa(env))
        raise Unsupported(type(e).__name__)
    return c(expr)


def certified(f, env):
    """f(env) at increasing precision until the value is real (a log of a difference that rounded negative
    at 60 digits becomes real at 400) -- the last level's value otherwise."""
    outer = _STATE["dps"]
    try:
        prev = None
        for level in _LEVELS:
            _STATE["dps"] = level
            v = f(env)
            if not _is_mp(v):
                return v                                   # exact: no precision question
            ok = isinstance(v, mpmath.mpf) and mpmath.isfinite(v)
            if ok and prev is not None:                    # accept only when two precisions agree
                with mpmath.workdps(level):
                    if abs(v - prev) <= mpmath.mpf(10) ** -20 * max(mpmath.mpf(10) ** -30, abs(v)):
                        return v                       # the lower level already had 20 correct digits
            prev, prev_level = (v, level) if ok else (None, level)
        raise Unsupported("the value did not settle at 2500 digits")
    finally:
        _STATE["dps"] = outer


def to_sympy(v):
    """A compiled result as a SymPy number (booleans as true/false). A 60-digit transcendental value is
    not an exact answer: Unsupported, so the caller keeps the exact form."""
    if _is_mp(v):
        raise Unsupported("a transcendental value (exact form kept)")
    if isinstance(v, bool):
        return sympy.true if v else sympy.false
    if isinstance(v, complex):
        return sympy.Float(v.real, 15) + sympy.I * sympy.Float(v.imag, 15)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise Unsupported("not finite")
        return sympy.Float(v, 15)
    v = _num(v)
    return sympy.Integer(v) if isinstance(v, int) else sympy.Rational(v.numerator, v.denominator)


def evaluate(expr, env=None, hook=None):
    """``expr`` with ``env`` substituted, exactly, as a SymPy number or boolean; raises Unsupported."""
    env = env or {}
    return to_sympy(compile_expr(expr, env.keys(), hook)(env))
