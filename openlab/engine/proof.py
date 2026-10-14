r"""Claims about a whole set, and the techniques that prove them.

A statement over a set is notation until a technique is named::

    \forall n \ge 1: \sum_{k=1}^{n} k = \frac{n(n+1)}{2}                     a note (shown, not run)
    \forall n \ge 1: \sum_{k=1}^{n} k = \frac{n(n+1)}{2} \text{ by induction}  proved
    \forall x \in \mathbb{R}: (x+1)^2 = x^2 + 2x + 1 \text{ by algebra}       proved
    \exists n \in \mathbb{N}: n^2 = 49 \text{ witness } n = 7                 proved
    \forall n \ge 0, q \text{ odd}: t_n \equiv 0 \pmod{q}
        \text{ by strong induction with } x_{n+1} = 2n(x_n + x_{n-1})       proved

openlab never checks a set element by element. Each variable may carry its own condition
(``n \ge 1, q \ge 1``, ``q \text{ odd}``, ``x \in [0, 1]``).

**Induction** computes the base case, then checks the step symbolically -- for an equation by
differences or ratios, for a congruence the same modulo m, for an inequality by increments or
ratios. **Strong induction with a step relation** ``x_{n+1} = F(n, x_n, x_{n-1}, …)`` checks the
first r cases and that *both sides* satisfy the relation (modulo m for a congruence; F must then
have integer coefficients).

In a step, a sequence defined by a recurrence is **unfolded** by that recurrence whenever the index
is provably in its range, and earlier **proved claims** are used as facts. A **congruence** holds when
(A − B)/m is a polynomial with integer coefficients in integer-valued pieces (integer variables,
sequence values, binomials, factorials) -- never merely a polynomial remainder over ℚ.
A step the checker cannot reduce is reported as *not proved*, never as true.
"""

from __future__ import annotations

import dataclasses
import math
import os
import re
from dataclasses import dataclass
from typing import Optional

import sympy
from sympy import Integer, Product, Sum, Symbol
from sympy.core.function import AppliedUndef

from . import interp
from .errors import OpenlabError, ParseError, SemanticsError
from .parse import _NOTE_RE, _depth0_split, parse_expr, parse_statement
from .transform import Congruence

_TECHNIQUES = {
    "strong": re.compile(r"(?i)\bby\s+strong\s+induction(?:\s+on\s+\\?([A-Za-z]+))?(?:\s*,?\s*(?:with|step)\b)?"),
    "induction": re.compile(r"(?i)\bby\s+induction(?:\s+on\s+\\?([A-Za-z]+))?"),
    "algebra": re.compile(r"(?i)\b(?:by\s+algebra|identically|by\s+direct\s+computation)\b"),
    "witness": re.compile(r"(?i)\bwitness(?:es)?\b"),
    "cases": re.compile(r"(?i)\bby\s+cases\s+on\b"),
    "exhaustion": re.compile(r"(?i)\bby\s+(?:exhaustion|exhaustive\s+(?:check|checking|computation|search))\b"),
}
_COLON = re.compile(r":|\\colon(?![A-Za-z])|\\mid(?![A-Za-z])")
_COMMA_ONLY = re.compile(r",|\\openlabquad|\\quad(?![A-Za-z])|\\;")
_VARCOND = re.compile(
    r"^\s*(?P<var>\\?[A-Za-z]+(?:_\{?\w+\}?)?)\s*"
    r"(?P<cond>(?:\\in|\\ge|\\geq|\\le|\\leq|>|<|\\ne|\\neq|\\gt|\\lt).*|\\openlabnote\{\s*(?:odd|even|prime)\s*\}|"
    r"(?:odd|even|prime))?\s*$", re.S)
_INTEGER_LETTERS = set("nmkjiqplrst")


@dataclass
class Claim:
    quant: str                           # "forall" | "exists"
    vars: list                           # [Symbol]
    conds: dict                          # var name -> condition text (`\ge 1`, `odd`, `\in [0, 1]`)
    body: str                            # the statement text
    technique: Optional[str]             # "induction" | "strong" | "algebra" | "witness" | None (a note)
    detail: str                          # `on q`, the step relation, the witness values
    src: str
    on: Optional[str] = None             # the induction variable when not the first
    subst: Optional[dict] = None         # by cases: the case's substitution n -> qk + r

    @property
    def cond(self) -> str:               # the first variable's condition (compatibility)
        return self.conds.get(self.vars[0].name, "") if self.vars else ""


def parse_claim(text: str) -> Claim:
    m = re.match(r"^\s*\\(forall|exists)(?![A-Za-z])\s*(.*)$", text, re.S)
    if not m:
        raise ParseError("not a claim", src=text)
    quant, rest = m.group(1), m.group(2)
    technique, detail, on = None, "", None
    for note in _NOTE_RE.finditer(rest):          # \text{by induction} -> \openlabnote{by induction}
        for name, pat in _TECHNIQUES.items():
            t = pat.search(note.group(1))
            if t:
                technique = name
                on = t.group(1) if t.groups() and t.group(1) else None
                after = note.group(1)[t.end():].strip()
                detail = (after + " " + rest[note.end():]).strip()    # the math after the technique
                rest = rest[:note.start()]
                break
        if technique:
            break
    if technique in ("induction", "strong") and on is None:      # \text{by induction on } k: the letter follows
        mt = re.match(r"\s*on\s+\\?([A-Za-z]+)\b\s*(.*)$", _NOTE_RE.sub(" ", detail), re.S)
        if mt:
            on, detail = mt.group(1), mt.group(2)
    rest = rest.strip().rstrip(".").strip()
    parts = _depth0_split(rest, _COLON)                 # \forall x \in [0, 1]: … -- the colon, then a comma
    if len(parts) < 2:
        parts = _depth0_split(rest, _COMMA_ONLY)
        spec_parts, body_parts = parts[:1], parts[1:]
    else:
        spec_parts, body_parts = [parts[0]], parts[1:]
    if not body_parts:
        raise ParseError(r"could not read the claim: write `\forall n \ge 1: …` (variables with optional "
                         r"conditions, a colon, then the statement)", src=text)
    inner = re.match(r"^\s*\\forall(?![A-Za-z])\s*(.*)$", ": ".join(p.strip() for p in body_parts), re.S)
    if inner and quant == "forall":                     # \forall r …: \forall m …: body -> \forall r …, m …: body
        merged = f"\\forall {spec_parts[0].strip()}, {inner.group(1)}"
        c2 = parse_claim(merged)
        return dataclasses.replace(c2, technique=technique, detail=_NOTE_RE.sub(" ", detail).strip(), src=text, on=on)
    vars_, conds, pending = [], {}, []
    coprime = []
    for piece in _depth0_split(spec_parts[0], re.compile(",")):
        g = re.fullmatch(r"\s*\\gcd\s*\(\s*(\\?[A-Za-z]+)\s*,\s*(\\?[A-Za-z]+)\s*\)\s*=\s*1\s*", piece)
        if g:                                     # \gcd(q, k) = 1: k is a unit modulo q (and q modulo k)
            coprime.append((g.group(1).lstrip("\\"), g.group(2).lstrip("\\")))
            continue
        two = re.fullmatch(r"\s*(.+?)\s*(\\leq?|<)\s*(\\?[A-Za-z]+)\s*(\\leq?|<)\s*(.+?)\s*", piece, re.S)
        if two and not re.search(r"\\(?:le|leq|<)", two.group(1)):   # 1 \le n \le q - 1: two-sided
            lo_t, op1, name2, op2, hi_t = two.groups()
            name2 = name2.lstrip("\\")
            vars_.append(Symbol(name2))
            lo_t = f"{lo_t} + 1" if op1 == "<" else lo_t
            conds[name2] = rf"\ge {lo_t}" if re.fullmatch(r"-?\d+", lo_t.strip()) else ""
            conds[f"upper:{name2}"] = f"{hi_t} - 1" if op2 == "<" else hi_t
            if not re.fullmatch(r"-?\d+", lo_t.strip()):
                conds[f"lower:{name2}"] = lo_t
            continue
        vm = _VARCOND.match(_NOTE_RE.sub(lambda n: f" {n.group(1).strip()} ", piece))
        if not vm:
            raise ParseError(f"could not read the variable condition `{piece.strip()}`",
                             hint=r"write `n \ge 1`, `q \text{ odd}`, `x \in [0, 1]`, or a bare name.", src=text)
        name = vm.group("var").strip().lstrip("\\")
        cond = (vm.group("cond") or "").strip()
        rng = re.fullmatch(r"\\in\s*\\\{\s*(-?\d+)\s*,\s*\\(?:ldots|dots|cdots)\s*,\s*(.+?)\s*\\\}", cond, re.S)
        if rng and re.search(r"[A-Za-z]", rng.group(2)):       # k ∈ {1, …, p − 1}: bounds, the upper one symbolic
            vars_.append(Symbol(name))
            conds[name] = rf"\ge {rng.group(1)}"
            conds[f"upper:{name}"] = rng.group(2)
            if pending:
                for p_ in pending:
                    conds.setdefault(p_, cond)
                pending = []
            continue
        if name in conds and cond:               # `n \in \mathbb{Z}, n \ge 1`: one integer range
            conds[name] = _merge_conds(conds[name], cond)
            continue
        vars_.append(Symbol(name))
        pending.append(name)
        if cond:
            for p in pending:                    # `x, y \in \mathbb{R}`: the condition covers both
                conds[p] = cond
            pending = []
    body = _NOTE_RE.sub(" ", " ".join(p.strip() for p in body_parts)).strip()
    for a_, b_ in coprime:
        conds[f"gcd:{a_}"] = conds.get(f"gcd:{a_}", "") + f" {b_}"
        conds[f"gcd:{b_}"] = conds.get(f"gcd:{b_}", "") + f" {a_}"
    return Claim(quant=quant, vars=vars_, conds=conds, body=body, technique=technique,
                 detail=_NOTE_RE.sub(" ", detail).strip(), src=text, on=on)


def _merge_conds(a: str, b: str) -> str:
    """Two conditions on one variable: a set of integers and a lower bound -> ℤ_{≥k}."""
    ca, cb = a.replace(" ", ""), b.replace(" ", "")
    if re.fullmatch(r"(?:\\ge|\\geq|>|\\gt)-?\d+", ca):
        ca, cb, a, b = cb, ca, b, a
    if re.fullmatch(r"\\in\\mathbb\{[ZN]\}", ca) and re.fullmatch(r"(?:\\ge|\\geq|>|\\gt)-?\d+", cb):
        k = _base_of(b, Symbol("x"))
        if "N" in ca:
            k = max(k, 0)
        return rf"\in \mathbb{{Z}}_{{\ge {k}}}"
    mt = re.fullmatch(r"\\neq?(-?\d+)", cb)
    if re.fullmatch(r"\\in\\mathbb\{[ZRQ]\}", ca) and mt:        # k \in \mathbb{Z}, k \ne 0
        return rf"{a.strip()} \setminus \{{{mt.group(1)}\}}"
    return b


def _real_set(cond: str):
    """The real set a simple condition names (`> 0`, `\\ne 1`, `\\in [0, 1]`, `\\in \\mathbb{R}`), else None."""
    c = cond.replace(" ", "")
    if c in ("", "\\in\\mathbb{R}"):
        return sympy.S.Reals
    num = r"(-?\d+(?:\.\d+)?)"
    m = re.fullmatch(r"(>|\\gt|\\ge|\\geq|<|\\lt|\\le|\\leq|\\ne|\\neq)" + num, c)
    if m:
        op, a = m.group(1).lstrip("\\"), sympy.Rational(m.group(2))
        return {">": sympy.Interval.open(a, sympy.oo), "gt": sympy.Interval.open(a, sympy.oo),
                "ge": sympy.Interval(a, sympy.oo), "geq": sympy.Interval(a, sympy.oo),
                "<": sympy.Interval.open(-sympy.oo, a), "lt": sympy.Interval.open(-sympy.oo, a),
                "le": sympy.Interval(-sympy.oo, a), "leq": sympy.Interval(-sympy.oo, a)}.get(
            op, sympy.S.Reals - sympy.FiniteSet(a))
    m = re.fullmatch(r"\\in([\[(])" + num + "," + num + r"([\])])", c)
    if m:
        return sympy.Interval(sympy.Rational(m.group(2)), sympy.Rational(m.group(3)), m.group(1) == "(", m.group(4) == ")")
    return None


def _base_of(cond: str, var: Symbol) -> Optional[int]:
    """The first integer of the set: `\\ge 1` -> 1, `> 0` -> 1, `\\in \\mathbb{N}` -> 0, odd -> 1,
    `\\in \\mathbb{Z}_{\\ge 2}` -> 2, `\\in \\mathbb{Z}^+` -> 1."""
    c = cond.replace(" ", "")
    m = re.fullmatch(r"\\in\\mathbb\{[ZN]\}_\{?(\\ge|\\geq|>|\\gt)(-?\d+)\}?", c)
    if m:
        return int(m.group(2)) + (1 if m.group(1) in (">", "\\gt") else 0)
    if re.fullmatch(r"\\in\\mathbb\{[ZN]\}(?:\^\{?\+\}?|_\{?\+\}?|_\{?>0\}?|\^\{?\*\}?)", c):
        return 1
    if re.fullmatch(r"\\in\\mathbb\{N\}_\{?0\}?", c):
        return 0
    m = re.fullmatch(r"(?:\\ge|\\geq)(-?\d+)", c)
    if m:
        return int(m.group(1))
    m = re.fullmatch(r"(?:>|\\gt)(-?\d+)", c)
    if m:
        return int(m.group(1)) + 1
    if re.fullmatch(r"\\in\\mathbb\{N\}", c) or c == "":
        return 0
    if c == "odd":
        return 1
    if c == "prime":
        return 2
    if c == "even":
        return 0
    return None


# --- integer divisibility -----------------------------------------------------------------------

_INTEGER_FUNCS = (sympy.binomial, sympy.factorial, sympy.factorial2)


_W1, _W2 = sympy.Wild("w1"), sympy.Wild("w2")


def _double_factorials(expr):
    """(2k)!/(2^k k!) is (2k−1)!!: an integer, which the quotient form hides."""
    def fix(n):
        if not n.is_Mul:
            return n
        args = list(n.args)
        for f in args:
            if isinstance(f, sympy.factorial):
                x = f.args[0] / 2
                inv_f = sympy.Pow(sympy.factorial(x), -1)
                inv_p = sympy.Pow(2, -x)
                if inv_f in args and inv_p in args:
                    rest = [g for g in args if g not in (f, inv_f, inv_p)]
                    return sympy.Mul(*rest, sympy.factorial2(2 * x - 1))
        return n
    return expr.replace(lambda n: n.is_Mul, fix) if expr.has(sympy.factorial) else expr


def _lower_factorials(expr):
    """(n+2)! is (n+2)(n+1)·n!, (2j+1)!! is (2j+1)(2j−1)!!, C(n, n+1) and C(n, −1) are 0: shifted
    arguments become polynomial factors, so ratios and differences cancel structurally."""
    def shift_of(e):
        if isinstance(e, sympy.Add):
            c, rest = e.as_coeff_Add()
            if c.is_Integer and rest.free_symbols:
                return c, rest
        return None

    def fact(n):
        sh = shift_of(n.args[0])
        if sh and sh[0] > 0 and sh[0] <= 12:
            c, base = sh
            return sympy.Mul(*[base + i for i in range(1, int(c) + 1)]) * sympy.factorial(base)
        return n

    def fact2(n):
        sh = shift_of(n.args[0])
        if sh and sh[0] > 0:
            c, base = sh
            k = int((c + 1) // 2) if c % 2 else int(c // 2)
            if c % 2:
                return sympy.Mul(*[base + c - 2 * i for i in range(k)]) * sympy.factorial2(base - 1)
            return sympy.Mul(*[base + c - 2 * i for i in range(k)]) * sympy.factorial2(base)
        return n

    def binom(n):
        a, b = n.args
        d = sympy.expand(b - a)
        if (d.is_Integer and d > 0) or (b.is_Integer and b < 0):
            return sympy.S.Zero
        return n
    if expr.has(sympy.factorial):
        expr = expr.replace(lambda n: isinstance(n, sympy.factorial), fact)
    if expr.has(sympy.factorial2):
        expr = expr.replace(lambda n: isinstance(n, sympy.factorial2), fact2)
    if expr.has(sympy.binomial):
        expr = expr.replace(lambda n: isinstance(n, sympy.binomial), binom)
    return expr


def _nonneg_exp(e, nonneg) -> bool:
    if e.is_number:
        return bool(e.is_integer and e >= 0)
    if nonneg is None:
        return True
    c, rest = e.as_coeff_Add()
    return bool({x.name for x in e.free_symbols} <= nonneg and c >= 0
                and all(t.as_coeff_Mul()[0] > 0 for t in sympy.Add.make_args(rest)))


def _reeval(expr):
    """Rebuild unevaluated Add/Mul/Pow nodes (openlab parses with evaluate=False) in canonical form."""
    if not isinstance(expr, sympy.Basic) or not expr.args:
        return expr
    args = [_reeval(a) for a in expr.args]           # one bottom-up pass (linear in the tree size)
    if isinstance(expr, (sympy.Add, sympy.Mul, sympy.Pow)):
        return expr.func(*args)
    if all(a is b for a, b in zip(args, expr.args)):
        return expr
    try:
        return expr.func(*args)
    except Exception:
        return expr


def _integer_valued_poly(expr, gens) -> Optional[bool]:
    """Is a polynomial with rational coefficients integer-valued on the integers? Yes iff its
    coordinates in the basis ∏ C(x_i, a_i) are integers -- read off by finite differences at 0."""
    try:
        poly = sympy.Poly(expr, *gens)
    except Exception:
        return None
    if not all(c.is_Rational for c in poly.coeffs()):
        return None
    if all(c.is_Integer for c in poly.coeffs()):
        return True
    degs = poly.degree_list()
    from itertools import product as _product
    for alpha in _product(*[range(d + 1) for d in degs]):
        e = expr
        for g, a_ in zip(gens, alpha):
            for _ in range(a_):
                e = e.xreplace({g: g + 1}) - e
        v = sympy.expand(e.xreplace({g: 0 for g in gens}))
        if not v.is_Integer:
            return False
    return True


def _integer_atoms(expr, int_funcs=frozenset(), nonneg=None):
    """Integer-valued pieces to treat as integer unknowns: values a(e) of integer sequences (named in
    ``int_funcs``), binomials, factorials, double factorials, rising factorials with an integer start,
    k^j ∏(c/k + i) = ∏(c + ik), powers b^e with b integer-valued and e ≥ 0 (its letters in ``nonneg``;
    ``None`` = not checked, outside proofs), and finite sums/products of integer pieces."""
    atoms = set()
    nonneg = None if nonneg is None else (set(nonneg) | {"t_range"})
    int_letters = set(nonneg or ()) | _INTEGER_LETTERS

    def integer_valued(e) -> bool:
        if e.is_Integer:
            return True
        if isinstance(e, Symbol):
            return e.name in int_letters or bool(e.is_integer)
        inner = _integer_atoms(e, int_funcs, nonneg)
        d = {a: sympy.Dummy() for a in inner}
        es = e.xreplace(d)
        gens = sorted(es.free_symbols, key=str)
        if not all(isinstance(g, sympy.Dummy) or g.name in int_letters or g.is_integer for g in gens):
            return False
        return bool(_integer_valued_poly(es, gens)) if gens else bool(es.is_Integer)

    for node in sympy.preorder_traversal(expr):
        if isinstance(node, AppliedUndef) and node.func.__name__ in int_funcs:
            atoms.add(node)
        elif isinstance(node, _INTEGER_FUNCS):
            if all(integer_valued(a) for a in node.args):      # (n/2)! is not an integer at n = 1
                atoms.add(node)
        elif isinstance(node, sympy.RisingFactorial) and _nonneg_exp(node.args[1], nonneg) \
                and integer_valued(node.args[1]) and integer_valued(node.args[0]):
            atoms.add(node)
        elif isinstance(node, sympy.Mul):               # k^j · RisingFactorial(c/k, j) = ∏_{i<j} (c + ik)
            pows = [f for f in node.args if isinstance(f, sympy.Pow) and not f.exp.is_number]
            rfs = [f for f in node.args if isinstance(f, sympy.RisingFactorial)]
            for pw in pows:
                for rf in rfs:
                    c = sympy.simplify(rf.args[0] * pw.base)
                    if rf.args[1] == pw.exp and _nonneg_exp(pw.exp, nonneg) and integer_valued(pw.exp) \
                            and integer_valued(pw.base) \
                            and integer_valued(c):
                        atoms.add(pw * rf)
        elif isinstance(node, sympy.Pow) and not node.exp.is_number and node.exp.free_symbols \
                and _nonneg_exp(node.exp, nonneg) and integer_valued(node.exp) and integer_valued(node.base):
            atoms.add(node)
        elif isinstance(node, (Sum, Product)) and len(node.limits) == 1:
            k, lo, hi = node.limits[0]
            t = Symbol("t_range", integer=True, nonnegative=True)
            f = sympy.powsimp(node.function, combine="exp")
            forms = [f.xreplace({k: Symbol(k.name, integer=True)}), f.xreplace({k: lo + t}), f.xreplace({k: hi - t})]
            if integer_valued(lo) and integer_valued(hi) and any(integer_valued(sympy.expand(g)) for g in forms):
                atoms.add(node)
    return atoms


def _abstract(expr, int_funcs, nonneg):
    """Replace integer-valued pieces by integer dummies, outermost first, until none remain."""
    dummies = {}
    for _ in range(6):
        atoms = sorted(_integer_atoms(expr, int_funcs, nonneg), key=lambda a: -sympy.count_ops(a))
        atoms = [a for a in atoms if a not in dummies]
        if not atoms:
            break
        top = [a for a in atoms if not any(o != a and o.has(a) for o in atoms)]    # outermost pieces
        for a in top:
            dummies[a] = sympy.Dummy(f"u{len(dummies)}", integer=True)
        expr = expr.xreplace({a: dummies[a] for a in top})
    return expr, dummies


def _safe_simplify(e):
    """simplify for a message: never lets a SymPy failure (Gosper recursion, …) break the claim."""
    from .semantics import _Timeout, time_limit
    tok = None
    try:
        with time_limit(2.0) as tok:
            return sympy.simplify(e)
    except _Timeout as t:
        if t.token is not tok:
            raise
        return e
    except (RecursionError, Exception):
        return e


def _par(e) -> str:
    """An expression for a quotient in a note: parenthesised unless it is one atom."""
    t = interp.pyform(e)
    return t if re.fullmatch(r"[\w.]+", t) else f"({t})"


def divides(m, expr, int_funcs=frozenset(), nonneg=None, units=(), integral=None):
    """(verdict, reason): does m divide expr, with every letter an integer (or a polynomial
    indeterminate)? True only when expr/m is an integer-valued polynomial in the integer-valued
    pieces -- or, when its denominator is made of the ``units`` of the modulus (2 for an odd m, k
    when gcd(m, k) = 1) and expr itself is integer-valued, by Euclid's lemma. False only when that
    is definitely not so for numbers or polynomials."""
    m, expr = sympy.sympify(m), sympy.sympify(expr)
    try:
        if not expr.free_symbols and not m.free_symbols:
            e, mm = expr.doit(), m.doit()
            if e.is_Integer and mm.is_Integer:
                return bool(e % mm == 0), f"{interp.pyform(e)} mod {interp.pyform(mm)} = {e % mm}"
            return None, "not an integer"
        expr = _lower_factorials(_double_factorials(expr))
        e_s, dummies = _abstract(expr, int_funcs, nonneg)
        m_s, md = _abstract(m, int_funcs, nonneg)
        dummies.update(md)
        q = sympy.cancel(sympy.together(e_s / m_s))
        gens = sorted(q.free_symbols | m_s.free_symbols | e_s.free_symbols, key=str)
        back = {v: k for k, v in dummies.items()}
        num0, den0 = sympy.fraction(sympy.together(q))
        bad_pow = lambda e: e.atoms(AppliedUndef) or any(p.exp.free_symbols for p in e.atoms(sympy.Pow))
        if bad_pow(num0) or (bad_pow(den0) and not units):
            return None, f"{_par(expr)}/{_par(m)} has pieces not known to be integers"
        if not gens:
            return (bool(q.is_Integer), f"the quotient is {interp.pyform(q)}")
        if q.is_polynomial(*gens):
            iv = _integer_valued_poly(q, gens)
            if iv:
                poly = sympy.Poly(q, *gens)
                how = "integer coefficients" if all(c.is_Integer for c in poly.coeffs()) else "integer values"
                return True, f"the quotient {interp.pyform(q.xreplace(back))} has {how}"
        num, den = sympy.fraction(sympy.together(q))
        unit_syms = {u for u in units if isinstance(u, Symbol)}
        unit_ints = {int(u) for u in units if getattr(u, "is_Integer", False)}
        if den != 1 and num.is_polynomial(*gens) and _integer_valued_poly(num, gens):
            ok_den = True
            for f, e in sympy.factorint(den).items() if den.is_Integer else den.as_powers_dict().items():
                f_ = sympy.sympify(f)
                if f_.is_Integer and f_ == 1:
                    continue
                if f_.is_Integer:
                    ok_den &= all(any(p_ == u or (u % p_ == 0 and p_ in sympy.factorint(u)) for u in unit_ints)
                                  for p_ in sympy.factorint(int(f_)))
                elif isinstance(f_, Symbol):
                    ok_den &= f_.name in {u.name for u in unit_syms}
                else:
                    ok_den = False
            whole = integral or (e_s.is_polynomial(*gens) and _integer_valued_poly(e_s, gens))
            if ok_den and whole:
                return True, (f"the quotient is {interp.pyform(num.xreplace(back))}/{interp.pyform(den)} with an integer "
                              f"numerator, and {interp.pyform(den)} is a unit modulo {interp.pyform(m)} (Euclid's lemma)")
            if ok_den and not whole:
                return None, (f"the quotient has the unit {interp.pyform(den)} below, but "
                              f"{interp.pyform(expr)} is not known to be an integer")
        # a polynomial modulus with leading coefficient ±1 in one indeterminate: the remainder decides
        if len(m_s.free_symbols) == 1:
            x = next(iter(m_s.free_symbols))
            num, den = sympy.fraction(sympy.together(e_s))
            pm = sympy.Poly(m_s, x) if m_s.is_polynomial(x) else None
            if pm is not None and abs(pm.LC()) == 1 and not den.has(x) and num.is_polynomial(x):
                pn = sympy.Poly(num, x)
                if all(c.is_Integer for c in pn.coeffs()) and den == 1:
                    r = sympy.rem(pn, pm)
                    if r.is_zero:
                        return True, f"the remainder on division by {interp.pyform(m)} is 0"
                    if not (r.free_symbols - {x}):
                        return False, f"the remainder on division by {interp.pyform(m)} is {interp.pyform(r.as_expr())}"
    except Exception:
        return None, "could not decide the divisibility"
    return None, (f"{_par(expr)}/{_par(m)} is not a polynomial with integer coefficients "
                  f"in integer-valued pieces")


class Prover:
    """Runs a claim's technique with an Executor for the computations."""

    def __init__(self, ex):
        self.ex = ex
        self.used: list = []                          # unfoldings and facts used, for the report
        self.int_funcs = set()                        # sequences whose every value is an integer
        for name, (n, sh, rhs, ics) in ex._reg("_openlab_recur").items():
            if all(getattr(v, "is_Integer", False) for v in ics.values()) and self._integer_relation(
                    rhs, sympy.Function(name)):
                self.int_funcs.add(name)
        self.nonneg: set = set()

    # --- evaluation helpers --------------------------------------------------------------
    def value(self, expr, notes, exclude=frozenset(), budget=6.0, structural=True):
        """Resolve and evaluate; a slow evaluation (a sum whose summand depends on n) is left as
        written, so the checks work structurally instead of timing out."""
        from .semantics import _Timeout, time_limit
        r, s = self.ex.resolve(expr, set(exclude), notes)
        tok = None
        old = self.ex.keep_sums
        self.ex.keep_sums = structural                 # sums and products stay structural in proofs
        try:
            with time_limit(budget) as tok:
                out = self.ex.evaluate(r, notes, simplify=False, subs=s)
                from .transform import Apply
                from .semantics import _free_vars, _finite_sums
                if isinstance(out, sympy.Basic) and out.has(Sum, Product):     # Σ_{i=0}^{3} (T^{∘i}(0) mod 2): a number
                    out = out.replace(lambda n_: isinstance(n_, (Sum, Product)) and not _free_vars(n_)
                                      and n_.has(Apply),
                                      lambda n_: _finite_sums(n_, settle=lambda e_: self.ex._settle(e_, []),
                                                              hook=self.ex._fast_hook))
                if isinstance(out, sympy.Basic) and out.has(Apply) and any(
                        not _free_vars(a) for a in out.atoms(Apply)):
                    out2 = self.ex._apply_all(out, notes)            # T^{∘2}(0) left by a written-out sum
                    if out2 != out:
                        out = self.ex.evaluate(out2, notes, simplify=False, subs={})
                return out
        except _Timeout as t:
            if t.token is not tok:
                raise
            notes.append("a sum was left unevaluated (too slow); checked by its structure")
            return self.ex._resolve_markers(r, notes)
        finally:
            self.ex.keep_sums = old

    @staticmethod
    def shift(expr, var):
        """expr with var -> var + 1; a sum or product running to var gains its next term."""
        def step(node):
            if isinstance(node, (Sum, Product)) and len(node.limits) == 1 and node.limits[0][2] == var:
                k, lo, _ = node.limits[0]
                nxt = node.function.xreplace({k: var + 1})
                return node + nxt if isinstance(node, Sum) else node * nxt
            return None
        from .transform import Apply, CallMod
        if expr.has(Apply):                      # n(n+2): a product, not a call of n (n + 1 is no function)
            expr = expr.replace(lambda a: isinstance(a, Apply) and a.args[0] == var
                                and not any(isinstance(x, CallMod) for x in a.args),
                                lambda a: sympy.Mul(a.args[0], *a.args[1:]))
        rep = {}
        for node in sympy.preorder_traversal(expr):
            out = step(node) if isinstance(node, (Sum, Product)) else None
            if out is not None:
                rep[node] = out
        placeholder = {node: sympy.Dummy() for node in rep}
        shifted = expr.xreplace(placeholder).xreplace({var: var + 1})
        return shifted.xreplace({placeholder[node]: rep[node] for node in rep})

    @staticmethod
    def _bare_index(node):
        """Shift the index so it appears without a numeric offset: ∑_{i=0}^{q-1} F(q, i+1) is
        ∑_{k=1}^{q} F(q, k)."""
        k, lo, hi = node.limits[0]
        offsets = set()
        for a in node.function.atoms(sympy.Add):
            if k in a.free_symbols:
                co = a.coeff(k)
                c = sympy.expand(a - co * k)
                c = c.as_coeff_Add()[0] if isinstance(c, sympy.Add) else c
                if co.is_Integer and co != 0 and c.is_Integer and c % co == 0:
                    offsets.add(c / co)
                else:
                    return node
        if len(offsets) != 1:
            return node
        c = offsets.pop()
        if c == 0:
            return node
        return node.func(node.function.xreplace({k: k - c}), (k, lo + c, hi + c))

    @staticmethod
    def _products(expr):
        """Rising factorials back into products: k^j·(c/k)^{(j)} is ∏_{i<j}(c + ik), x^{(j)} is ∏_{i<j}(x + i)."""
        if not expr.has(sympy.RisingFactorial):
            return expr
        i = Symbol("_p", integer=True)

        def mul(n):
            args = list(n.args)
            for rf in [f for f in args if isinstance(f, sympy.RisingFactorial)]:
                x, j = rf.args
                for pw in [f for f in args if isinstance(f, sympy.Pow) and f.exp == j]:
                    c = sympy.simplify(x * pw.base)
                    if not c.has(pw.base) or c.is_number:
                        rest = [g for g in args if g not in (rf, pw)]
                        return sympy.Mul(*rest, Product(c + i * pw.base, (i, 0, j - 1)))
            return n
        expr = expr.replace(lambda n: n.is_Mul and n.has(sympy.RisingFactorial), mul)
        return expr.replace(lambda n: isinstance(n, sympy.RisingFactorial) and not n.args[1].is_number,
                            lambda n: Product(n.args[0] + i, (i, 0, n.args[1] - 1)))

    @staticmethod
    def _canonical_index(expr):
        """Every sum/product index is named by nesting depth, so ∑_k f(k) and ∑_i f(i) coincide."""
        def rename(node, depth):
            if isinstance(node, (Sum, Product)) and len(node.limits) == 1:
                k, lo, hi = node.limits[0]
                new = Symbol(f"_i{depth}", integer=True)
                f = rename(node.function, depth + 1).xreplace({k: new})
                return node.func(f, (new, rename(lo, depth + 1), rename(hi, depth + 1)))
            if not node.args or isinstance(node, sympy.Atom):
                return node
            args = [rename(a, depth) for a in node.args]
            return node.func(*args) if any(a is not b for a, b in zip(args, node.args)) else node
        return rename(expr, 0)

    @staticmethod
    def split_sums(expr):
        """Canonical, structural sums and products: the index appears bare; ∑_{k=a}^{b+c} f is
        ∑_{k=a}^{b} f plus the c last terms; two sums of the same summand with numeric lower limits
        a < a' split the first a' − a terms off the lower one. Same for ∏ with factors."""
        expr = Prover._products(expr)
        if not expr.has(Sum, Product):
            return expr
        expr = _reeval(expr)
        expr = expr.replace(lambda n: isinstance(n, (Sum, Product)) and len(n.limits) == 1
                            and n.limits[0][1].is_Integer and n.limits[0][2].is_Integer
                            and n.limits[0][2] - n.limits[0][1] <= 30, lambda n: n.doit())
        expr = expr.replace(lambda n: isinstance(n, (Sum, Product)) and len(n.limits) == 1, Prover._bare_index)
        expr = Prover._canonical_index(expr)

        d = sympy.Dummy("k")

        def parts(e):
            c = e.as_coeff_Add()[0] if isinstance(e, sympy.Add) else (e if e.is_Integer else sympy.S.Zero)
            return c, sympy.expand(e - c)
        # last terms: within a group of the same summand and lower limit, every sum runs to the
        # smallest upper limit, the extra terms split off (never a division)
        groups: dict = {}
        for n in expr.atoms(Sum, Product):
            if len(n.limits) == 1:
                k, lo, hi = n.limits[0]
                c, base = parts(hi)
                groups.setdefault((type(n), n.function.xreplace({k: d}), lo, base), []).append((n, c))
        rep = {}
        for (typ, _, lo, base), items in groups.items():
            cmin = min(c for _, c in items)
            for n, c in items:
                if c > cmin and c - cmin <= 256:
                    k = n.limits[0][0]
                    core = typ(n.function, (k, lo, base + cmin))
                    extra = [n.function.xreplace({k: base + cmin + i}) for i in range(1, int(c - cmin) + 1)]
                    rep[n] = sympy.Add(core, *extra) if typ is Sum else sympy.Mul(core, *extra)
        expr = expr.xreplace(rep) if rep else expr
        # first terms: within a group of the same summand and upper limit, every sum starts at
        # the largest numeric lower limit
        groups = {}
        for n in expr.atoms(Sum, Product):
            if len(n.limits) != 1:
                continue
            k, lo, hi = n.limits[0]
            if lo.is_Integer:
                groups.setdefault((type(n), n.function.xreplace({k: d}), hi.xreplace({k: d})), []).append(n)
        rep = {}
        for (typ, _, _), nodes in groups.items():
            top = max(n.limits[0][1] for n in nodes)
            for n in nodes:
                k, lo, hi = n.limits[0]
                if lo < top and top - lo <= 50:
                    core = typ(n.function, (k, top, hi))
                    firsts = [n.function.xreplace({k: lo + i}) for i in range(int(top - lo))]
                    rep[n] = sympy.Add(core, *firsts) if typ is Sum else sympy.Mul(core, *firsts)
        return expr.xreplace(rep) if rep else expr

    @staticmethod
    def structural(expr):
        """Cancel what cancels by structure: canonical sums, lowered factorials, vanishing binomials."""
        e = _lower_factorials(_double_factorials(Prover.split_sums(_reeval(expr))))
        return sympy.expand(e)

    @staticmethod
    def _opaque_nested(e) -> bool:
        """A sum nested in a sum whose terms call a non-SymPy function (isprime, an undefined f, …)."""
        for s_ in e.atoms(Sum):
            if s_.function.has(Sum) and any(not type(f_).__module__.startswith("sympy")
                                            or isinstance(f_, AppliedUndef)
                                            for f_ in s_.function.atoms(sympy.Function)):
                return True
        return False

    @staticmethod
    def zero(expr, budget: float = 5.0, structural_only: bool = False) -> Optional[bool]:
        from .semantics import _Timeout, time_limit
        tok = None
        try:
            with time_limit(budget) as tok:
                e = Prover.structural(expr)
                if e == 0:
                    return True
                if structural_only:
                    return None
                if Prover._opaque_nested(e):
                    return None                # doit cannot close Σ over Σ of a number-theory summand: don't burn the budget
                if e.has(Sum, Product):
                    e = e.doit()
                e = sympy.simplify(sympy.powsimp(sympy.expand_power_base(e, force=True), force=True))
                if e != 0 and e.has(sympy.factorial, sympy.factorial2, sympy.binomial, sympy.RisingFactorial, sympy.gamma):
                    e = sympy.simplify(sympy.gammasimp(sympy.combsimp(e.rewrite(sympy.gamma))))
        except _Timeout as t:
            if t.token is not tok:
                raise
            return None                        # too slow to decide: not proved (never a guess)
        except Exception:
            return None
        if e == 0:
            return True
        if e.is_number and e.is_zero is False:
            return False                       # a concrete nonzero difference: definitely not equal
        return None

    # --- variables, conditions, facts -------------------------------------------------------
    def _integer_symbol(self, v: Symbol, cond: str) -> Symbol:
        c = cond.replace(" ", "")
        if "mathbb{R}" in c or "mathbb{C}" in c or c.startswith("\\in["):
            return Symbol(v.name, real=True)
        integer = v.name in _INTEGER_LETTERS or "mathbb{N}" in c or "mathbb{Z}" in c or c in ("odd", "even", "prime")
        lo = _base_of(cond, v)
        kw = {"integer": True} if integer else {"real": True}
        if integer and lo is not None and lo >= 0:
            kw["nonnegative"] = True
        if integer and lo is not None and lo >= 1:
            kw["positive"] = True
        return Symbol(v.name, **kw)

    def _lower(self, v: Symbol, conds: dict):
        return _base_of(conds.get(v.name, ""), v)

    def _nonneg(self, e, conds: dict, shift=0) -> bool:
        """e ≥ 0 for every variable at or above its lower bound (variable = bound + j, j ≥ 0)."""
        rep = {}
        known = set(conds) | set(getattr(self, "_claim_vars", ()))
        for s_ in e.free_symbols:
            if s_.name in known:
                lo = self._lower(Symbol(s_.name), conds)
            elif s_.is_integer and s_.is_nonnegative:        # m′ = m − 2 ≥ 0, a shifted variable
                lo = 0
            else:                                            # a summation index: its range is not known here
                return False
            if lo is None:
                return False
            if s_.name == getattr(self, "_step_var", None):
                lo = lo + shift
            j = Symbol(f"j_{s_.name}", integer=True, nonnegative=True)
            rep[s_] = lo + (2 * j if conds.get(s_.name, "").strip() == "odd" else j)
        val = sympy.simplify(e.xreplace(rep))
        return bool(val.is_nonnegative)

    def unfold(self, expr, conds, notes, shift=0, rounds=3):
        """Replace the highest-index terms a(e) of recurrence-defined sequences by their recurrence,
        when e is provably at least the first index the recurrence covers."""
        recs = self.ex._reg("_openlab_recur")
        for _ in range(rounds):
            terms = [f for f in expr.atoms(AppliedUndef) if f.func.__name__ in recs and len(f.args) == 1
                     and f.args[0].free_symbols]
            if not terms:
                return expr
            fam: dict = {}
            for f in terms:                      # the top term of each family a(n + q + c)
                key = (f.func.__name__, f.args[0].as_coeff_Add()[1] if isinstance(f.args[0], sympy.Add) else f.args[0])
                c = f.args[0].as_coeff_Add()[0] if isinstance(f.args[0], sympy.Add) else 0
                if key not in fam or c > fam[key][0]:
                    fam[key] = (c, f)
            rep = {}
            for c, f in fam.values():
                n, sh, rhs, ics = recs[f.func.__name__]
                first = max(ics) + 1
                e = f.args[0]
                if not self._nonneg(e - first, conds, shift):
                    continue
                rep[f] = rhs.xreplace({n: e - sh})
                self.used.append(f"{f.func.__name__}({interp.pyform(e)}) unfolded by its recurrence (index ≥ {first})")
            if not rep:
                return expr
            expr = expr.xreplace(rep)
        return expr

    def facts(self):
        return self.ex._reg("_openlab_facts").get("list", [])

    @staticmethod
    def by_name(expr, target):
        """Rename the symbols of ``expr`` to those of ``target`` with the same name (assumptions differ)."""
        names = {s.name: s for s in target.free_symbols if isinstance(s, Symbol)}
        rep = {s: names[s.name] for s in expr.free_symbols if isinstance(s, Symbol) and s.name in names and s != names[s.name]}
        return expr.xreplace(rep) if rep else expr

    @staticmethod
    def _implied(current: str, required: str) -> bool:
        """Does the current condition on a variable imply the fact's condition?"""
        cur, req = current.replace(" ", ""), required.replace(" ", "")
        if not req or cur == req:
            return True
        if req in ("odd", "even", "prime"):             # a property, not a bound: implied only by itself
            return cur == req
        lo_c, lo_r = _base_of(cur, Symbol("x")), _base_of(req, Symbol("x"))
        if lo_c is not None and lo_r is not None and cur not in ("prime",):
            return lo_c >= lo_r
        return False

    def _instances(self, fact, expr, conds):
        """Ways to instantiate a fact inside ``expr``: (A_inst, B_inst, how). By name first, then
        by matching the fact's variables as wildcards; each instance must satisfy the fact's conditions."""
        fvars = [Symbol(v) for v in fact.get("vars", [s.name for s in (fact["A"].free_symbols | fact["B"].free_symbols)
                                                             if isinstance(s, Symbol)])]
        out = []
        A, B = self.by_name(fact["A"], expr), self.by_name(fact["B"], expr)
        if fact.get("exact"):                    # the induction hypothesis: at n only (never at n+1)
            return [(A, B, "")]
        if all(self._implied(conds.get(v.name, ""), fact["conds"].get(v.name, "")) for v in fvars) \
                and self._upper_ok(fact, {}, conds):
            out.append((A, B, ""))
        seen = set()
        subsets = [[v] for v in fvars] + ([fvars] if len(fvars) > 1 else [])
        for subset in subsets:
            wilds = {v: sympy.Wild(f"W_{v.name}") for v in subset}
            A_w = A.xreplace({self.by_name(v, expr): w for v, w in wilds.items()})
            for node in sympy.preorder_traversal(expr):
                mt = node.match(A_w) if isinstance(node, sympy.Basic) else None
                if not mt or node in seen:
                    continue
                seen.add(node)
                inst = {self.by_name(v, expr): mt.get(w, v) for v, w in wilds.items()}
                self._instance_ok(inst, fact, conds, out, A, B)
        return out

    def _instance_ok(self, inst, fact, conds, out, A, B):
        if True:
            ok = True
            for vn in fact.get("vars", []):              # a variable kept as it is must satisfy the lemma too
                if all(k_.name != vn for k_ in inst) and not self._implied(conds.get(vn, ""),
                                                                            fact["conds"].get(vn, "")):
                    return
            for v, e in inst.items():
                req = fact["conds"].get(v.name, "")
                if e != v and not self._integer_point(e, v, req, conds):     # n/2, n − 1/2: not an instance
                    ok = False
                    break
                if not req:
                    continue
                if e == v and self._implied(conds.get(v.name, ""), req):
                    continue
                lo = _base_of(req, v)
                if req.strip() in ("odd", "even", "prime") or lo is None or not self._nonneg(e - lo, conds):
                    ok = False
                    break
            if ok and not self._upper_ok(fact, inst, conds):     # 1 ≤ i ≤ m − 3: the upper end too
                ok = False
            if ok and inst and any(e != v for v, e in inst.items()) and not A.xreplace(inst).is_Atom:
                out.append((A.xreplace(inst), B.xreplace(inst), " at " + ", ".join(
                    f"{v} = {interp.pyform(e)}" for v, e in inst.items() if e != v)))

    def _integer_point(self, e, v, req, conds) -> bool:
        """A fact over an integer variable v applies at e only if e is an integer for every value of the
        current variables: an integer-valued polynomial in integer letters."""
        c = req.replace(" ", "")
        if "mathbb{R}" in c or "mathbb{C}" in c or c.startswith("\\in["):
            return True
        if not (v.name in _INTEGER_LETTERS or _base_of(req, v) is not None or "mathbb" in c):
            return True
        if e.is_Integer:
            return True
        gens = sorted(e.free_symbols, key=str)
        if not gens or not all(g.name in _INTEGER_LETTERS or g.is_integer or conds.get(g.name, "").replace(" ", "")
                               in ("odd", "even") or "mathbb{Z}" in conds.get(g.name, "")
                               or "mathbb{N}" in conds.get(g.name, "") for g in gens):
            return False
        try:
            return bool(e.is_polynomial(*gens) and _integer_valued_poly(sympy.expand(e), gens))
        except Exception:
            return False

    def _apply_function_lemmas(self, expr, conds, notes):
        """An equation proved for an arbitrary (undefined) function f -- e.g. the parity split of
        Σ_{k=2}^{2m−4} f(k) -- applied to a specific summand: f := the summand (nothing was assumed about f)."""
        for fact in self.facts():
            if fact["kind"] != "eq" or not isinstance(fact["A"], Sum) or len(fact["A"].limits) != 1:
                continue
            FA = fact["A"]
            kf, lof, hif = FA.limits[0]
            if not (isinstance(FA.function, AppliedUndef) and FA.function.args == (kf,)):
                continue
            fname = FA.function.func
            if self.ex._lookup(str(fname)) is not None:
                continue                                    # f is a defined function: not arbitrary
            fvars = [Symbol(v) for v in fact.get("vars", [])]
            wilds = {v: sympy.Wild(f"W_{v.name}") for v in fvars}
            pat = sympy.Tuple(lof, hif).xreplace({s_: wilds[Symbol(s_.name)] for s_ in (lof.free_symbols | hif.free_symbols)
                                                  if Symbol(s_.name) in wilds})

            def fix(node):
                if len(node.limits) != 1:
                    return node
                j, lo, hi = node.limits[0]
                mt = sympy.Tuple(lo, hi).match(pat)
                if mt is None and pat == sympy.Tuple(lo, hi):
                    mt = {}
                if mt is None:
                    return node
                inst = {v: mt.get(w, v) for v, w in wilds.items()}
                chk: list = []
                self._instance_ok({k_: e_ for k_, e_ in inst.items()}, fact, conds, chk, FA, fact["B"])
                if inst and any(e_ != v for v, e_ in inst.items()) and not chk:
                    return node
                if not self._upper_ok(fact, inst, conds):
                    return node
                F = node.function
                lam = sympy.Lambda(j, F)
                Bi = fact["B"].xreplace({s_: inst.get(Symbol(s_.name), s_) for s_ in fact["B"].free_symbols})
                Bi = Bi.replace(lambda x: isinstance(x, AppliedUndef) and x.func == fname, lambda x: lam(*x.args))
                self.used.append(self._cite(fact, "") + f" with {fname} := the summand {interp.pyform(F)}")
                return Bi
            new = expr.replace(lambda n_: isinstance(n_, Sum), fix)
            if new != expr:
                expr = new
        return expr

    def _upper_ok(self, fact, inst, conds) -> bool:
        """A lemma stated for v ≤ hi (a two-sided binder) applies at e only when e ≤ hi is known here:
        from the current upper bound of e's letter, or from its lower bounds."""
        for key, hi_t in fact["conds"].items():
            if not (isinstance(key, str) and key.startswith("upper:")):
                continue
            vname = key[len("upper:"):]
            try:
                hi = parse_expr(hi_t)[0]
            except OpenlabError:
                return False
            sub = {Symbol(k_.name): e_ for k_, e_ in inst.items()}
            hi = hi.xreplace(sub)
            e = next((e_ for k_, e_ in inst.items() if k_.name == vname), Symbol(vname))
            names = {s_.name: s_ for s_ in e.free_symbols} if isinstance(e, sympy.Basic) else {}
            hi = hi.xreplace({Symbol(n_): s_ for n_, s_ in names.items()})
            if self._nonneg(hi - e, conds):
                continue
            cur = conds.get(f"upper:{e.name}") if isinstance(e, Symbol) else None
            if cur is not None:
                try:
                    cur_e = parse_expr(cur)[0] if isinstance(cur, str) else cur
                except OpenlabError:
                    return False
                if self._nonneg(sympy.expand(hi - cur_e), conds) or sympy.expand(hi - cur_e) == 0:
                    continue
            return False
        return True

    def apply_facts(self, expr, m, conds, notes, extra=()):
        """Rewrite with earlier proved claims (and the induction hypothesis in ``extra``): an
        equation A = B as a substitution; a congruence A ≡ B (mod m) as a substitution when the
        expression is an integer-coefficient polynomial in its pieces (then P(A) − P(B) is a
        multiple of A − B)."""
        if expr.has(Sum) and not getattr(self, "_in_sum_facts", False):     # lemmas about each term, inside Σ
            def inside(node):
                if len(node.limits) != 1:
                    return node
                j, a_, b_ = node.limits[0]
                c2 = dict(conds)
                if a_.is_Integer:
                    c2[j.name] = rf"\ge {a_}"
                c2[f"upper:{j.name}"] = b_
                self._in_sum_facts = True
                try:
                    f2 = self.apply_facts(node.function, m, c2, notes, extra)
                finally:
                    self._in_sum_facts = False
                return node if f2 == node.function else Sum(f2, node.limits[0])
            expr = expr.replace(lambda n_: isinstance(n_, Sum), inside)
        if expr.has(AppliedUndef):              # a(n + q(j+1)) matches a(W + q) only once expanded
            expr = expr.replace(lambda n: isinstance(n, AppliedUndef),
                                lambda n: n.func(*[sympy.expand(a) for a in n.args]))
        if expr.has(Sum):
            expr = self._apply_function_lemmas(expr, conds, notes)
        for f in list(extra) + self.facts():
            if f["kind"] not in ("eq", "cong"):
                continue
            if f["kind"] == "cong":
                if m is None:
                    continue
                fm = self.by_name(f["m"], m)
                if sympy.simplify(fm - m) != 0:
                    continue
                e_s, _ = _abstract(expr, self.int_funcs, self._nonneg_names(conds))
                gens = sorted(e_s.free_symbols, key=str)
                if gens and not (e_s.is_polynomial(*gens) and all(
                        c.is_Integer for c in sympy.Poly(e_s, *gens).coeffs())):
                    continue
            for A, B, how in self._instances(f, expr, conds):
                new = expr.xreplace({A: B}) if expr.has(A) else expr
                if new == expr and f["kind"] == "eq" and A.is_Mul:
                    fa = set(A.args)
                    new = expr.replace(lambda n: n.is_Mul and fa <= set(n.args),
                                       lambda n: sympy.Mul(*[g for g in n.args if g not in fa]) * B)
                if new != expr:
                    expr = new
                    src = ("used the induction hypothesis" if f.get("exact") else self._cite(f, how))
                    self.used.append(f"{src}: {interp.pyform(A)} "
                                     f"{'≡' if f['kind'] == 'cong' else '='} {interp.pyform(B)}"
                                     f"{' (mod ' + interp.pyform(f['m']) + ')' if f['kind'] == 'cong' else ''}")
        return expr

    @staticmethod
    def _cite(f, how):
        if f.get("src"):
            return f"used the earlier result “{f['src']}”{how}"
        return f"used the result of line {f['line']}{how}"

    def matches_fact(self, diff, conds):
        """diff is ± an instance of a proved equation's A − B: it vanishes by that fact."""
        for f in self.facts():
            if f["kind"] != "eq":
                continue
            for A, B, how in self._instances(f, diff, conds):
                for sign in (1, -1):
                    if self.zero(diff - sign * (A - B), budget=3.0, structural_only=True):
                        self.used.append(self._cite(f, how))
                        return True
        return False

    @staticmethod
    def _is_indicator(f):
        from .transform import Pick
        return (type(f).__name__ == "isprime" and len(f.args) == 1) or \
            (isinstance(f, Pick) and f.args[1] == 1 and f.args[2] == 0)

    def _indicator_rules(self, expr, conds):
        """𝟙² = 𝟙 for a 0/1 indicator; 𝟙_ℙ(a·b) = 0 when a, b are provably ≥ 2 on the claim's range."""
        if not isinstance(expr, sympy.Basic):
            return expr
        expr = expr.replace(lambda p: isinstance(p, sympy.Pow) and self._is_indicator(p.base)
                            and p.exp.is_Integer and p.exp >= 1, lambda p: p.base)
        primes_ = {k for k, c in conds.items() if isinstance(c, str) and c.strip() == "prime"}
        if primes_:                                      # p \text{ prime}: isprime(p) = 1
            expr = expr.replace(lambda f: type(f).__name__ == "isprime" and isinstance(f.args[0], Symbol)
                                and f.args[0].name in primes_, lambda f: sympy.Integer(1))

        def composite(f):
            if type(f).__name__ != "isprime" or f.args[0].is_number:
                return f
            arg = sympy.factor(sympy.expand(f.args[0]))
            parts = []
            for g in sympy.Mul.make_args(arg):
                if isinstance(g, sympy.Pow) and g.exp.is_Integer and g.exp > 1:
                    parts += [g.base] * int(g.exp)
                else:
                    parts.append(g)
            nums = [g for g in parts if g.is_number]
            others = [g for g in parts if not g.is_number]
            c_ = sympy.Mul(*nums) if nums else sympy.Integer(1)
            if (c_.is_Integer and c_ >= 4 and not sympy.isprime(int(c_)) and others
                    and all(self._nonneg(g - 1, conds) for g in others)):     # 6k, k ≥ 1: 2 · (3k)
                d_ = sympy.Integer(min(sympy.primefactors(int(c_))))
                parts = [d_, (c_ / d_) * sympy.Mul(*others)]
            if len(parts) >= 2 and all((g.is_number and g >= 2) or (not g.is_number and self._nonneg(g - 2, conds))
                                       for g in parts):
                self.used.append(f"isprime({interp.pyform(f.args[0])}) = 0: it is "
                                 f"{' · '.join(_par(g) for g in parts)} with every factor ≥ 2")
                return sympy.Integer(0)
            return f
        expr = expr.replace(lambda f: type(f).__name__ == "isprime", composite)
        if expr.has(Sum):
            expr = expr.replace(lambda n: isinstance(n, Sum) and len(n.limits) == 1
                                and n.function.has(sympy.Mod, sympy.core.relational.Relational),
                                lambda n: self._range_simplify(n, conds))
            expr = expr.replace(lambda n: isinstance(n, Sum) and len(n.limits) == 1 and any(
                                    self._is_indicator(f_) for f_ in n.function.atoms(sympy.Function)),
                                lambda n: self._vanishing_sum(n, conds))     # indicator summands only
        return expr

    def _range_nonneg(self, e, j, a, b, conds):
        """e ≥ 0 for every j in [a, b], e linear in j: check both ends (with the claim's bounds)."""
        e = sympy.expand(e)
        if not e.is_polynomial(j) or sympy.degree(e, j) > 1:
            return False
        return self._nonneg(e.xreplace({j: a}), conds) and self._nonneg(e.xreplace({j: b}), conds)

    def _range_simplify(self, node, conds):
        """Inside Σ_{j=a}^{b}: (q − j) mod q = q − j when 0 ≤ q − j ≤ q − 1 on the range; a comparison
        that holds (or fails) at both ends of the range is true (false) -- both linear in j."""
        j, a, b = node.limits[0]
        f = node.function
        conds0 = conds
        conds = dict(conds)                               # terms matter only when a ≤ b: use that
        gap = sympy.expand(b - a)
        for s_ in gap.free_symbols:
            known_ = s_.name in conds0 or (s_.is_integer and s_.is_nonnegative)     # q′ ≥ 0 too
            if isinstance(s_, Symbol) and known_ and sympy.expand(gap - s_).is_Integer:
                lo_s = _base_of(conds0.get(s_.name, ""), s_) if s_.name in conds0 else 0
                need = -int(sympy.expand(gap - s_))
                if lo_s is not None and need > lo_s:
                    conds[s_.name] = rf"\ge {need}"
        from .transform import Pick
        if isinstance(f, Pick) and isinstance(f.args[0], (sympy.Eq, sympy.Ne)) and f.args[2] == 0:
            c_, v_ = f.args[0], f.args[1]
            e_ = c_.rhs if c_.lhs == j else (c_.lhs if c_.rhs == j else None)
            if e_ is not None and j not in e_.free_symbols and self._nonneg(e_ - a, conds0) and self._nonneg(b - e_, conds0):
                one = v_.xreplace({j: e_})                # Σ_j [j = e] v(j) = v(e) for e in range
                self.used.append(f"Σ_{{{j}={interp.pyform(a)}}}^{{{interp.pyform(b)}}} [{j} = {interp.pyform(e_)}] "
                                 f"picks the single term {j} = {interp.pyform(e_)}")
                if isinstance(c_, sympy.Eq):
                    return one
                if v_ == 1:
                    return (b - a + 1) - one

        def mod(x):
            e_, m_ = x.args
            if self._range_nonneg(e_, j, a, b, conds) and self._range_nonneg(m_ - 1 - e_, j, a, b, conds):
                return e_
            if self._range_nonneg(e_ + m_, j, a, b, conds) and self._range_nonneg(-1 - e_, j, a, b, conds):
                return e_ + m_
            return x
        f2 = f.replace(lambda x: isinstance(x, sympy.Mod), mod)
        if f2 == f and f.has(sympy.Mod) and not getattr(self, "_splitting", False):
            for x in f.atoms(sympy.Mod):                   # (j − a) mod q: split where j − a changes sign
                e_ = sympy.expand(x.args[0])
                if e_.is_polynomial(j) and sympy.degree(e_, j) == 1 and abs(e_.coeff(j)) == 1:
                    j0 = sympy.solve(e_, j)[0]
                    if self._nonneg(j0 - a, conds0) and self._nonneg(b - j0, conds0):
                        self._splitting = True
                        try:
                            left = self._range_simplify(Sum(f, (j, a, j0 - 1)), conds0)
                            right = self._range_simplify(Sum(f, (j, j0 + 1, b)), conds0)
                        finally:
                            self._splitting = False
                        mid = f.xreplace({j: j0})
                        self.used.append(f"Σ split at {j} = {interp.pyform(j0)}, where {interp.pyform(e_)} changes sign")
                        return left + mid + right

        def rel(x):
            d_ = x.lhs - x.rhs
            if isinstance(x, (sympy.Ne, sympy.Gt, sympy.Ge)) and self._range_nonneg(d_ - (1 if not isinstance(x, sympy.Ge) else 0), j, a, b, conds):
                return sympy.true                         # d ≥ 1 on the range: ≠ 0 and > 0
            if isinstance(x, (sympy.Ne, sympy.Lt, sympy.Le)) and self._range_nonneg(-d_ - (1 if not isinstance(x, sympy.Le) else 0), j, a, b, conds):
                return sympy.true
            if isinstance(x, sympy.Eq) and (self._range_nonneg(d_ - 1, j, a, b, conds) or self._range_nonneg(-d_ - 1, j, a, b, conds)):
                return sympy.false
            return x
        f2 = f2.replace(lambda x: isinstance(x, sympy.core.relational.Relational) and x.has(j), rel)
        if not getattr(self, "_splitting", False):     # [j − a ≠ 0]: true except at one index -- split there
            for x in f2.atoms(sympy.Eq, sympy.Ne):
                e_ = sympy.expand(x.lhs - x.rhs)
                if x.has(j) and e_.is_polynomial(j) and sympy.degree(e_, j) == 1 and abs(e_.coeff(j)) == 1:
                    j0 = sympy.solve(e_, j)[0]
                    if self._nonneg(j0 - a, conds0) and self._nonneg(b - j0, conds0):
                        self._splitting = True
                        try:
                            left = self._range_simplify(Sum(f2, (j, a, j0 - 1)), conds0)
                            right = self._range_simplify(Sum(f2, (j, j0 + 1, b)), conds0)
                        finally:
                            self._splitting = False
                        self.used.append(f"Σ split at {j} = {interp.pyform(j0)}, where {interp.pyform(x)} changes")
                        return left + f2.xreplace({j: j0}) + right
        if f2 == f:
            return node
        self.used.append(f"inside Σ_{{{j}={interp.pyform(a)}}}^{{{interp.pyform(b)}}} the index range decides "
                         f"{interp.pyform(f)} = {interp.pyform(f2)}")
        return Sum(f2, (j, a, b))

    def _term_vanishes(self, f, j, start, conds, from_top):
        """Does the summand vanish for every index at or beyond `start` (j = start + t, or j = start − t)?"""
        t_ = Symbol(f"_t{j.name}", integer=True, nonnegative=True)
        c2 = dict(conds)
        c2[t_.name] = r"\ge 0"
        g = f.xreplace({j: (start - t_) if from_top else (start + t_)})
        saved = list(self.used)
        out = self._indicator_rules(sympy.expand(g), c2)
        self.used[:] = saved
        return self.zero(out, budget=2.0) is True

    def _vanishing_sum(self, node, conds):
        """Σ_{j=a}^{b} f(j) where f(j) = 0 on the range (a ≤ j ≤ b), possibly after peeling one or two
        terms off either end (then the range must have that many terms)."""
        j, a, b = node.limits[0]
        f = node.function
        for peel in range(0, 3):
            if peel and not self._nonneg(b - a - peel + 1, conds):     # the range has that many terms
                break
            for from_top in (False, True):
                if self._term_vanishes(f, j, (b - peel) if from_top else (a + peel), conds, from_top):
                    kept = [f.xreplace({j: (b - i) if from_top else (a + i)}) for i in range(peel)]
                    self.used.append(f"every term of Σ_{{{j}={interp.pyform(a)}}}^{{{interp.pyform(b)}}} "
                                     f"{interp.pyform(f)} vanishes"
                                     + (f" except the {'last' if from_top else 'first'} {peel}" if peel else "")
                                     + " (the index is in its range)")
                    return self._indicator_rules(sympy.Add(*kept), conds) if kept else sympy.Integer(0)
        return node

    def _nonneg_summand(self, f):
        """A product of 0/1 indicators, non-negative constants and sums of such: every value is ≥ 0."""
        for g in sympy.Mul.make_args(f):
            if self._is_indicator(g) or (g.is_number and g.is_nonnegative):
                continue
            if isinstance(g, Sum) and len(g.limits) == 1 and self._nonneg_summand(g.function):
                continue
            if isinstance(g, sympy.Pow) and g.exp.is_Integer and g.exp % 2 == 0:
                continue
            return False
        return True

    def _indicator_bound(self, e, upper, conds):
        """A sum of products of 0/1 indicators (times c ≥ 0) lies between 0 and c·(number of terms)."""
        def bound(node):
            if not (isinstance(node, Sum) and len(node.limits) == 1):
                return node
            k, lo, hi = node.limits[0]
            coeff, inds = sympy.Integer(1), []
            for f in sympy.Mul.make_args(node.function):
                if self._is_indicator(f):
                    inds.append(f)
                elif f.is_number and f >= 0:
                    coeff *= f
                else:
                    return node
            if not inds:
                return node
            if not upper:
                return sympy.Integer(0)
            count = hi - lo + 1
            if not self._nonneg(count, conds):
                return node
            self.used.append(f"each {', '.join(sorted({type(f).__name__ for f in inds}))} term is at most 1, "
                             f"so the sum is at most its {interp.pyform(count)} terms")
            return coeff * count
        if not isinstance(e, sympy.Basic):
            return e
        e = e.replace(lambda n: isinstance(n, Sum), bound)
        coeff, inds = sympy.Integer(1), []
        for f in sympy.Mul.make_args(e):
            if self._is_indicator(f):
                inds.append(f)
            elif f.is_number and f >= 0:
                coeff *= f
            else:
                return e
        if inds:                                            # a bare indicator: 0 ≤ 𝟙 ≤ 1
            self.used.append(f"{interp.pyform(e)} is between 0 and {interp.pyform(coeff)}")
            return coeff if upper else sympy.Integer(0)
        return e

    def holds_symbolic(self, kind, diff, m, conds, notes, shift=0, hyp=()):
        """diff = 0 (eq) or diff ≡ 0 (mod m) (cong), after unfolding recurrences and using facts
        (and the induction hypothesis ``hyp``)."""
        if kind == "eq" and diff.has(Sum):             # a proved schema over an arbitrary f, used on the summands first
            n_used = len(self.used)
            d_f = self._apply_function_lemmas(diff, conds, notes)
            if d_f != diff and self.zero(d_f, structural_only=True):
                return True, "by an earlier result"
            del self.used[n_used:]
        diff = self._indicator_rules(diff, conds)
        units = self.units_of(m, conds) if m is not None else set()
        nn = self._nonneg_names(conds)
        integral = None
        if kind == "cong":                       # the original difference: an integer? (for Euclid)
            base_ = sympy.powsimp(sympy.expand(diff))   # k^p/k -> k^{p−1}
            pw_rep = {}                          # k^{p−1} with p − 1 ≥ 0 on the range: a whole number
            for pw in base_.atoms(sympy.Pow):
                if pw.exp.free_symbols and self._nonneg(pw.exp, conds) and all(
                        s_.is_integer or s_.name in _INTEGER_LETTERS for s_ in pw.base.free_symbols) \
                        and (pw.base.free_symbols or pw.base.is_Integer):
                    pw_rep[pw] = sympy.Dummy("w", integer=True)
            e0, _ = _abstract(_lower_factorials(_double_factorials(base_.xreplace(pw_rep))), self.int_funcs, nn)
            g0 = sorted(e0.free_symbols, key=str)
            integral = bool(e0.is_polynomial(*g0) and _integer_valued_poly(e0, g0)) if g0 else bool(e0.is_Integer)
        import os
        dbg = os.environ.get("OPENLAB_PROOF_DEBUG")
        diff0 = diff
        for attempt in range(4):
            if dbg:
                print(f"[proof] attempt {attempt} {kind} diff = {sympy.sstr(diff)[:300]}", flush=True)
            if kind == "eq":
                if self.zero(diff, structural_only=True) or self.matches_fact(diff, conds):
                    return True, "by structure" if attempt == 0 else "by an earlier result"
                if self.zero(diff) or self.zero(self.parity(diff, conds)):
                    return True, ""
                if self.ratio_one(diff):
                    return True, "the two sides have ratio 1"
            else:
                ok, why = divides(m, self.structural(diff), self.int_funcs, nn, units, integral)
                if not ok:
                    peeled = self._peel_powers(self.structural(diff), conds)
                    if peeled is not None:
                        ok, why = divides(m, peeled, self.int_funcs, nn, units, integral)
                        for d_, pw_ in getattr(self, "_peeled", {}).items():
                            why = why.replace(interp.pyform(d_), f"({interp.pyform(pw_)})")
                if not ok and attempt == 0:
                    ok2, why2 = self._cong_variants(self.structural(diff), m, conds, notes, nn, units, integral)
                    if ok2:
                        ok, why = ok2, why2
                if ok:
                    return True, why
            new = self.apply_facts(self.unfold(diff, conds, notes, shift), m, conds, notes, extra=hyp)
            if new == diff:
                break
            diff = new
        if kind == "eq":
            variants = self._sum_variants(diff0) + (self._sum_variants(diff) if diff != diff0 else [])
            for d2, how in variants:
                d2 = self._indicator_rules(self.structural(d2), conds)
                if self.zero(d2, structural_only=True) or self.zero(d2):
                    self.used.append(how)
                    return True, how
            return None, ""
        return divides(m, self.structural(diff), self.int_funcs, nn, units, integral)

    def _peel_powers(self, expr, conds):
        """b^e with e ≥ e0 ≥ 1 on the claim's range -> b^e0 · P, P = b^(e−e0) an integer unknown
        (so 2·2^(n−1) = 2^n is seen to be even for n ≥ 1)."""
        rep = {}
        self._peeled = {}
        for pw in expr.atoms(sympy.Pow):
            b, e = pw.base, pw.exp
            if not (b.is_Integer and abs(b) > 1 and e.free_symbols):
                continue
            names = {s_.name for s_ in e.free_symbols}
            if not names <= _INTEGER_LETTERS | {v for v in conds if not v.startswith("gcd:")}:
                continue
            lows = {s_: self._lower(Symbol(s_.name), conds) for s_ in e.free_symbols}
            if any(lo is None for lo in lows.values()):
                continue
            gens = sorted(e.free_symbols, key=str)
            if not (e.is_polynomial(*gens) and _integer_valued_poly(e, gens)):
                continue
            e0 = sympy.expand(e.xreplace(lows))
            if not (e0.is_Integer and e0 >= 1 and self._nonneg(e - e0, conds)):
                continue
            d_ = sympy.Dummy(f"P{len(rep)}", integer=True, nonnegative=True)
            rep[pw] = b ** e0 * d_
            self._peeled[d_] = sympy.Pow(b, sympy.expand(e - e0), evaluate=False)
        return expr.xreplace(rep) if rep else None

    @staticmethod
    def _reflect(node):
        """Σ_{k=a}^{b} f(k) = Σ_{k=a}^{b} f(a + b − k): the orientation with the smaller sort key."""
        if not (isinstance(node, Sum) and len(node.limits) == 1):
            return node
        k, a, b = node.limits[0]
        f, fr = sympy.expand(node.function), sympy.expand(node.function.xreplace({k: a + b - k}))
        return node if sympy.default_sort_key(f) <= sympy.default_sort_key(fr) else Sum(fr, (k, a, b))

    def _sum_variants(self, diff):
        """Rewritings of Σ that simplification does not try: each sum reflected (k → a + b − k), and a
        sum split at the upper limit c of another sum with the same lower limit, its upper part
        reflected onto [a, a + b − c − 1] (ordered vs unordered partitions: Σ_{2}^{2m−2} = 2Σ_{2}^{m−1} + mid)."""
        out = []
        refl = diff.replace(lambda n: isinstance(n, Sum), self._reflect)
        if refl != diff:
            out.append((refl, "a sum reflected: Σ_{k=a}^{b} f(k) = Σ_{k=a}^{b} f(a + b − k)"))
        sums = [n for n in diff.atoms(Sum) if len(n.limits) == 1]
        for S1 in sums:                                     # even and odd indices
            k, a, b = S1.limits[0]
            if not k.is_integer and not isinstance(a, sympy.Integer):
                pass
            jj = Symbol(k.name, integer=True)
            ev = Sum(S1.function.xreplace({k: 2 * jj}), (jj, sympy.ceiling(a / 2), sympy.floor(b / 2)))
            od = Sum(S1.function.xreplace({k: 2 * jj + 1}), (jj, sympy.ceiling((a - 1) / 2), sympy.floor((b - 1) / 2)))
            if not (ev.limits[0][1].has(sympy.floor, sympy.ceiling) or ev.limits[0][2].has(sympy.floor, sympy.ceiling)
                    or od.limits[0][1].has(sympy.floor, sympy.ceiling) or od.limits[0][2].has(sympy.floor, sympy.ceiling)):
                out.append((diff.xreplace({S1: ev + od}), f"Σ_{{{k}={interp.pyform(a)}}}^{{{interp.pyform(b)}}} split "
                                                          f"into even and odd {k}"))
        for S1 in sums:
            k, a, b = S1.limits[0]
            for S2 in sums:
                if S2 is S1:
                    continue
                j, a2, c = S2.limits[0]
                if sympy.simplify(a - a2) != 0 or (sympy.simplify(b - c)).is_Integer:
                    continue
                lower = Sum(S1.function, (k, a, c))
                upper = Sum(S1.function.xreplace({k: a + b - k}), (k, a, sympy.expand(a + b - c - 1)))
                d2 = diff.xreplace({S1: lower + upper})
                out.append((d2, f"the sum to {interp.pyform(b)} split at {interp.pyform(c)}, its upper part "
                                f"reflected (k → {interp.pyform(a + b)} − k)"))
                out.append((d2.replace(lambda n: isinstance(n, Sum), self._reflect), out[-1][1]))
        return out

    def _cong_variants(self, diff, m, conds, notes, nn, units, integral, depth=0):
        if integral and units:                             # k·X ≡ 0 with k a unit ⟹ X ≡ 0 (Euclid)
            for u in units:
                if isinstance(u, Symbol):
                    uu = next((s_ for s_ in diff.free_symbols if s_.name == u.name), u)
                    d2 = self.apply_facts(sympy.expand(uu * diff), m, conds, notes)
                    ok2, why2 = divides(m, sympy.expand(d2), self.int_funcs, nn, (), None)
                    if ok2:
                        self.used.append(f"multiplied by the unit {u} (then cancelled by Euclid's lemma)")
                        return True, why2
        return self._cong_variants_core(diff, m, conds, notes, nn, units, integral, depth)

    def _cong_variants_core(self, diff, m, conds, notes, nn, units, integral, depth=0):
        """Rewrites a congruence proof may need: (u + v)^e with a symbolic exponent by the binomial theorem
        (end terms split off); C(n, k) = (n/k)·C(n−1, k−1) (absorption); and a sum ≡ 0 term by term,
        each term with its index in range (so 1 ≤ j ≤ p − 1 is a unit modulo a prime p)."""
        if depth > 2:
            return None, ""
        cands = [diff]
        for pw in diff.atoms(sympy.Pow):                   # (k + 1)^p = k^p + 1 + Σ_{j=1}^{p−1} C(p, j) k^j
            b_, e_ = pw.base, pw.exp
            if isinstance(b_, sympy.Add) and len(b_.args) == 2 and e_.free_symbols and e_.is_integer \
                    and self._nonneg(e_ - 1, conds):
                j = Symbol("j_b", integer=True)
                for u_, v_ in (b_.args, b_.args[::-1]):      # both orientations: k^j·1^{p−j} is the one that helps
                    mid = Sum(sympy.binomial(e_, j) * u_ ** j * v_ ** (e_ - j), (j, 1, e_ - 1))
                    cands.append(sympy.expand(diff.xreplace({pw: u_ ** e_ + v_ ** e_ + mid})))
                self.used.append(f"({interp.pyform(b_)})^{interp.pyform(e_)} expanded by the binomial theorem")
        for d_ in cands:
            rest, sums = [], []
            for t in sympy.Add.make_args(sympy.expand(d_)):
                (sums if isinstance(t, Sum) and len(t.limits) == 1 else rest).append(t)
            all_zero = True
            for S_ in sums:                                # each sum ≡ 0 term by term
                j, a_, b2 = S_.limits[0]
                jj = Symbol(j.name, integer=True)
                f_ = S_.function.xreplace({j: jj})
                c2 = dict(conds)
                c2[jj.name] = rf"\ge {a_}" if a_.is_Integer else ""
                c2[f"range:{jj.name}"] = (a_, b2)
                u2 = self.units_of(m, c2)
                ok_t, _ = divides(m, f_, self.int_funcs, nn | {jj.name}, u2, True)
                if not ok_t:
                    absorbed = f_.replace(lambda x: isinstance(x, sympy.binomial) and x.args[1].free_symbols,
                                          lambda x: x.args[0] / x.args[1] * sympy.binomial(x.args[0] - 1, x.args[1] - 1))
                    ok_t, _ = divides(m, absorbed, self.int_funcs, nn | {jj.name}, u2, True)
                if not ok_t:
                    all_zero = False
                    break
                self.used.append(f"every term of Σ_{{{j}={interp.pyform(a_)}}}^{{{interp.pyform(b2)}}} "
                                 f"{interp.pyform(S_.function)} is ≡ 0 (mod {interp.pyform(m)})")
            if not all_zero:
                continue
            r_ = sympy.Add(*rest)
            if r_ == 0:
                return True, "every remaining term is ≡ 0"
            ok_r, why_r = divides(m, r_, self.int_funcs, nn, units, integral)
            if not ok_r and r_.has(sympy.binomial):            # C(p, j) = (p/j)·C(p−1, j−1), j a unit
                absorbed = r_.replace(lambda x: isinstance(x, sympy.binomial) and x.args[1].free_symbols,
                                      lambda x: x.args[0] / x.args[1] * sympy.binomial(x.args[0] - 1, x.args[1] - 1))
                ok_r, why_r = divides(m, absorbed, self.int_funcs, nn, units, True if integral is None else integral)
                if ok_r:
                    self.used.append("C(n, k) = (n/k)·C(n−1, k−1)")
            if ok_r:
                return True, why_r
        return None, ""

    def ratio_one(self, diff) -> bool:
        """X − Y with X/Y simplifying to 1 (products, rising factorials)."""
        if not (diff.is_Add and len(diff.args) == 2):
            return False
        x, y = diff.args
        try:
            r = sympy.simplify(sympy.gammasimp(sympy.combsimp(self.structural(-x / y).doit().rewrite(sympy.gamma))))
        except Exception:
            return False
        return r == 1

    def _nonneg_names(self, conds):
        out = set(self.nonneg)
        for v, c in conds.items():
            lo = None if ":" in v or not isinstance(c, str) else _base_of(c, Symbol(v))
            if lo is not None and lo >= 0:
                out.add(v)
        return out

    @staticmethod
    def parity(expr, conds):
        """q odd -> q = 2r + 1, q even -> 2r (r ≥ 0): makes (−1)^q decidable."""
        rep = {}
        for s_ in expr.free_symbols:
            c = conds.get(s_.name, "").strip()
            if c in ("odd", "even"):
                r = Symbol(f"r_{s_.name}", integer=True, nonnegative=True)
                rep[s_] = 2 * r + 1 if c == "odd" else 2 * r
        return expr.xreplace(rep) if rep else expr

    def units_of(self, m, conds):
        """Numbers and letters that are units modulo m: 2 when m is an odd letter; k when gcd(m, k) = 1."""
        out = set()
        if isinstance(m, Symbol):
            if conds.get(m.name, "").strip() == "odd":
                out.add(Integer(2))
            for nm in conds.get(f"gcd:{m.name}", "").split():
                out.add(Symbol(nm))
            if conds.get(m.name, "").strip() == "prime":          # 1 ≤ j ≤ p − 1 is a unit modulo a prime p
                for key, (lo, hi) in [(k, v) for k, v in conds.items() if k.startswith("range:")]:
                    j = key[len("range:"):]
                    if (lo - 1).is_nonnegative is not False and self._nonneg(lo - 1, conds) and \
                            self._nonneg(sympy.Symbol(m.name, integer=True) - 1 - hi, conds):
                        out.add(Symbol(j))
        return out

    # --- techniques ------------------------------------------------------------------------
    def run(self, claim: Claim, notes: list):
        from .semantics import _Timeout, time_limit
        tok = None
        old_shadow = getattr(self.ex, "_shadow", frozenset())
        self.ex._shadow = frozenset(old_shadow) | {v.name for v in claim.vars}   # ∀ M: M is not the M(p) above
        limit = (max(5.0, self.ex.timeout - 2) if claim.technique == "exhaustion"     # plain integer checks: all of -t
                 else min(20.0, max(5.0, self.ex.timeout - 8)))
        self._progress = None
        try:
            with time_limit(limit) as tok:
                return self._run(claim, notes)
        except _Timeout as t:
            if t.token is not tok:
                raise
            if claim.technique == "exhaustion" and self._progress:
                k, n_ = self._progress
                return None, [f"not proved: ran out of time after checking {k} of {n_} cases (all held so far); "
                              f"raise the limit with -t, or split the range"]
            return None, [f"not proved: the {claim.technique} checks did not finish in time (a step is too hard "
                          f"to decide symbolically; split it into lemmas)"] + list(dict.fromkeys(self.used))
        finally:
            self.ex._shadow = old_shadow

    def _run(self, claim: Claim, notes: list):
        self._claim_vars = {v.name for v in claim.vars}
        out = self._run_technique(claim, notes)
        if out[0] is sympy.true and claim.quant == "forall" and claim.technique in ("algebra", "cases"):
            bad = self._first_cases(claim)                    # a symbolic proof is also checked on its first cases
            if bad is not None:
                return sympy.false, [f"counterexample at {bad} (the symbolic argument was wrong there; "
                                     f"please report this)"]
        return out

    def _first_cases(self, claim: Claim, width: int = 3):
        """The first few whole-number cases of a claim whose variables all have lower bounds, computed."""
        from itertools import product as _product
        los = []
        for v in claim.vars:
            if not self._integer_symbol(v, claim.conds.get(v.name, "")).is_integer:
                return None
            lo = _base_of(claim.conds.get(v.name, ""), v)
            if lo is None or len(claim.vars) > 3:
                return None
            if claim.conds.get(v.name, "").strip() == "prime":
                los.append([2, 3, 5][:width])
                continue
            step = 2 if claim.conds.get(v.name, "").strip() == "odd" else 1
            los.append([lo + step * i for i in range(width)])
        try:
            kind, A, B, m = self._body(claim)
        except OpenlabError:
            return None
        from .semantics import _free_vars
        names = {v.name for v in claim.vars}
        if any(s_.name not in names for s_ in _free_vars(A) | _free_vars(B)
               | (_free_vars(m) if isinstance(m, sympy.Basic) else set())
               if self.ex._lookup(s_.name) is None):
            return None                              # free parameters (n, k arbitrary): cases would not be numbers
        coprime = [(k[len("gcd:"):], nm) for k, c in claim.conds.items() if k.startswith("gcd:") for nm in c.split()]
        for pt in _product(*los):
            vals = {v.name: x for v, x in zip(claim.vars, pt)}
            if any(a in vals and b in vals and math.gcd(vals[a], vals[b]) != 1 for a, b in coprime):
                continue                              # excluded by gcd(·, ·) = 1
            if any(claim.conds.get(v.name, "").strip() == "prime" and not sympy.isprime(vals[v.name])
                   for v in claim.vars):
                continue
            out_of_range = False                          # 1 ≤ n ≤ q − 1: only points inside the binder
            for key, bound_t in claim.conds.items():
                if key.startswith(("upper:", "lower:")):
                    vname = key.split(":", 1)[1]
                    try:
                        bnd = parse_expr(bound_t)[0].xreplace({Symbol(k_): Integer(x_) for k_, x_ in vals.items()})
                        bnd = sympy.sympify(bnd).doit()
                    except Exception:
                        out_of_range = True
                        break
                    if not bnd.is_Integer or (vals[vname] > bnd if key.startswith("upper:") else vals[vname] < bnd):
                        out_of_range = True
                        break
            if out_of_range:
                continue
            sub = {v: Integer(x) for v, x in zip(claim.vars, pt)}
            try:
                ok, _a, _b = self._holds_at(kind, A.xreplace(sub), B.xreplace(sub),
                                            m.xreplace(sub) if kind == "cong" else None,
                                            m if kind == "ineq" else None, Symbol("_none"), 0, [])
            except Exception:                        # a case that cannot be computed is simply not checked
                continue
            if ok is False:
                return ", ".join(f"{v} = {x}" for v, x in zip(claim.vars, pt))
        return None

    def _run_technique(self, claim: Claim, notes: list):
        if claim.technique != "witness":
            bad = self._non_integer_vars(claim)
            if bad:
                return None, [f"a congruence is a statement about integers, but {', '.join(bad)} "
                              f"{'ranges' if len(bad) == 1 else 'range'} over non-integers; not proved "
                              r"(declare it in \mathbb{Z} or \mathbb{N}, or give an integer range like n \ge 1)"]
        if claim.technique == "induction":
            out = self.induction(claim, notes)
        elif claim.technique == "strong":
            out = self.strong(claim, notes)
        elif claim.technique == "algebra":
            out = self.algebra(claim, notes)
        elif claim.technique == "witness":
            out = self.witness(claim, notes)
        elif claim.technique == "exhaustion":
            out = self.exhaustion(claim, notes)
        elif claim.technique == "cases":
            out = self.cases(claim, notes)
        else:
            raise SemanticsError(f"unknown technique {claim.technique}")
        value, steps = out
        return value, steps + [u for u in dict.fromkeys(self.used)]

    def _non_integer_vars(self, claim: Claim):
        """Variables of a congruence claim that are not integers (a polynomial indeterminate that
        appears in the modulus is fine: that congruence is between polynomials)."""
        try:
            kind, A, B, m = self._body(claim)
        except OpenlabError:
            return []
        if kind != "cong":
            return []
        in_mod = {s.name for s in m.free_symbols}
        ind = None
        if claim.technique in ("induction", "strong"):
            ind = claim.on or (claim.vars[0].name if claim.vars else None)
        return [str(v) for v in claim.vars if v.name not in in_mod and v.name != ind
                and not self._integer_symbol(v, claim.conds.get(v.name, "")).is_integer]

    @staticmethod
    def functional(expr, var):
        """``a_{n+3}`` is one opaque name to SymPy, so shifting n would not touch it: rewrite every
        subscripted name whose subscript mentions a variable as a function of it, a(n + 3)."""
        if not isinstance(expr, sympy.Basic):
            return expr
        names = {v.name for v in (var if isinstance(var, (list, tuple, set)) else [var])}
        rep = {}
        for s_ in expr.free_symbols:
            m = re.fullmatch(r"([A-Za-z]+)_(?:\{(.+)\}|(.+))", getattr(s_, "name", ""))
            if not m:
                continue
            sub = m.group(2) or m.group(3)
            if not any(re.search(rf"(?<![A-Za-z]){re.escape(nm)}(?![A-Za-z])", sub) for nm in names):
                continue
            try:
                idx = parse_expr(sub)[0]
            except Exception:
                continue
            if {s.name for s in idx.free_symbols} & names:
                rep[s_] = sympy.Function(m.group(1))(idx)
        return expr.xreplace(rep) if rep else expr

    def _body(self, claim: Claim):
        """-> (kind, A, B, m or relation class), with integer variables carrying their assumptions."""
        st = parse_statement(claim.body, 1)
        if claim.vars:
            st = dataclasses.replace(st, lhs=self.functional(st.lhs, claim.vars),
                                     rhs=self.functional(st.rhs, claim.vars))
        if st.lhs is not None and not st.approx and not st.chain:
            return "eq", st.lhs, st.rhs, None
        r = st.rhs
        if isinstance(r, Congruence):
            return "cong", r.args[0], r.args[1], r.args[2]
        if isinstance(r, sympy.core.relational.Relational) and not isinstance(r, sympy.Eq):
            return "ineq", r.lhs, r.rhs, type(r)
        raise SemanticsError(f"the claim's statement `{claim.body}` is not an equation, congruence or inequality",
                             hint=r"induction and algebra prove `A = B`, `A \equiv B \pmod m` and `A \le B`.")

    def _holds_at(self, kind, A, B, m, rel, var, k, notes, conds=None):
        sub = {var: Integer(k)}
        concrete = not ((A.free_symbols | B.free_symbols) - {var})       # a case with numbers: compute it
        a = self.value(A.xreplace(sub), notes, structural=not concrete)
        b = self.value(B.xreplace(sub), notes, structural=not concrete)
        if kind == "eq":
            if a.free_symbols or b.free_symbols:
                ok, _ = self.holds_symbolic("eq", a - b, None, conds or {}, notes)
                return ok, a, b
            return self.zero(a - b), a, b
        if kind == "cong":
            mv = self.value(m.xreplace(sub), notes)
            if self.zero(a - b):
                return True, a, b
            ok, _ = self.holds_symbolic("cong", a - b, mv, conds or {}, notes)
            return ok, a, b
        try:
            return bool(rel(a, b)), a, b
        except TypeError:
            return None, a, b

    def _var(self, claim: Claim):
        name = claim.on or claim.vars[0].name
        v = next((x for x in claim.vars if x.name == name), None)
        if v is None:
            raise SemanticsError(f"{name} is not a variable of the claim")
        return v

    def induction(self, claim: Claim, notes: list):
        var = self._var(claim)
        n0 = _base_of(claim.conds.get(var.name, ""), var)
        if n0 is None:
            raise SemanticsError(f"induction needs a starting whole number: `{var} \\ge 1` or `{var} \\in \\mathbb{{N}}`",
                                 hint=r"write `\forall n \ge 1: … \text{ by induction}`.")
        kind, A, B, m = self._body(claim)
        others = {v: self._integer_symbol(v, claim.conds.get(v.name, "")) for v in claim.vars if v != var}
        others = {v: s_ for v, s_ in others.items() if s_.is_integer}     # p prime, k ≥ 0: 0^p = 0
        if others:
            A, B = A.xreplace(others), B.xreplace(others)
            if isinstance(m, sympy.Basic):
                m = m.xreplace(others)
        rel = m if kind == "ineq" else None
        mod = m if kind == "cong" else None
        conds = claim.conds
        ok, a, b = self._holds_at(kind, A, B, mod, rel, var, n0, notes, conds)
        if ok is None and kind == "ineq":                 # a real parameter is left: decide the base like `by algebra`
            ok, a, b = self._base_ineq(a, b, rel, claim)
        steps = [f"base case {var} = {n0}: {interp.pyform(a)} {'≡' if kind == 'cong' else '=' if kind == 'eq' else 'vs'} "
                 f"{interp.pyform(b)} {'✓' if ok else '✗' if ok is False else '?'}"]
        if ok is False:
            return sympy.false, steps + [f"the claim fails at {var} = {n0}"]
        if ok is None:
            return None, steps + ["the base case could not be decided"]
        n = Symbol(var.name, integer=True, nonnegative=True) if n0 >= 0 else Symbol(var.name, integer=True)
        An, Bn = A.xreplace({var: n}), B.xreplace({var: n})
        ex = {var.name}
        a_n, b_n = self.value(An, notes, ex), self.value(Bn, notes, ex)
        a_1, b_1 = self.value(self.shift(An, n), notes, ex), self.value(self.shift(Bn, n), notes, ex)
        self._step_var = var.name
        step = None
        hyp = [{"kind": kind, "A": a_n, "B": b_n, "m": None if kind != "cong" else self.value(mod.xreplace({var: n}), notes, ex),
                "conds": {var.name: conds.get(var.name, "")}, "vars": [var.name], "line": "the hypothesis",
                "exact": True}]
        if kind == "eq":
            if self.zero(a_n - b_n):
                step = f"A({var}) − B({var}) simplifies to 0 for a symbolic {var}: it holds directly, for every {var}"
            elif self.holds_symbolic("eq", (a_1 - a_n) - (b_1 - b_n), None, conds, notes)[0]:
                step = (f"step: A({var}+1) − A({var}) = B({var}+1) − B({var}), "
                        f"so A({var}) = B({var}) ⟹ A({var}+1) = B({var}+1)")
            elif self.zero(a_1 / a_n - b_1 / b_n):
                step = (f"step: A({var}+1)/A({var}) = {interp.pyform(sympy.simplify(a_1 / a_n))} = B({var}+1)/B({var}), "
                        f"so A({var}) = B({var}) ⟹ A({var}+1) = B({var}+1) (A({var}) ≠ 0)")
            elif self.holds_symbolic("eq", a_1 - b_1, None, conds, notes, hyp=hyp)[0]:
                step = f"step: A({var}+1) = B({var}+1) once A({var}) = B({var}) is used (the induction hypothesis)"
        elif kind == "cong" and self._moves(hyp[0]["m"], n, notes, ex):
            # A(n) ≡ B(n) (mod m(n)) says nothing modulo m(n+1): no hypothesis, prove case n+1 outright
            m1 = self.value(self.shift(mod.xreplace({var: n}), n), notes, ex)
            if self.holds_symbolic("cong", a_1 - b_1, m1, conds, notes)[0]:
                step = (f"step: the modulus changes with {var}, so the hypothesis (mod {interp.pyform(hyp[0]['m'])}) "
                        f"is not used; A({var}+1) ≡ B({var}+1) (mod {interp.pyform(m1)}) holds outright for every {var} ≥ {n0}")
            else:
                notes_ = [f"the modulus {interp.pyform(hyp[0]['m'])} changes with {var}: the hypothesis modulo "
                          f"m({var}) says nothing modulo m({var}+1), and case {var}+1 does not hold outright; "
                          f"not proved (induct on a variable the modulus does not depend on)"]
                return None, steps + notes_
        elif kind == "cong":
            mv = hyp[0]["m"]
            nn = self._nonneg_names(conds)
            units = self.units_of(mv, conds)
            try:
                ca = sympy.cancel(self.structural(a_1) / self.structural(a_n))
                cb = sympy.cancel(self.structural(b_1) / self.structural(b_n))
                ca, cb = [sympy.simplify(sympy.powsimp(c, force=True)) if any(
                    pw.exp.free_symbols for pw in c.atoms(sympy.Pow)) else c for c in (ca, cb)]
                if ca.has(Sum, Product, sympy.factorial, sympy.factorial2, sympy.binomial):
                    ca = sympy.simplify(sympy.gammasimp(sympy.combsimp(ca.doit().rewrite(sympy.gamma))))
                if cb.has(Sum, Product, sympy.factorial, sympy.factorial2, sympy.binomial):
                    cb = sympy.simplify(sympy.gammasimp(sympy.combsimp(cb.doit().rewrite(sympy.gamma))))
            except Exception as exc:
                if os.environ.get("OPENLAB_PROOF_DEBUG"):
                    print(f"[proof] ratio failed: {type(exc).__name__}: {exc}", flush=True)
                ca = cb = None
            ratio_ok = False
            if os.environ.get("OPENLAB_PROOF_DEBUG"):
                print(f"[proof] ratios {ca} | {cb}", flush=True)
            if ca is not None and not ca.has(AppliedUndef, Sum, Product) and not cb.has(AppliedUndef, Sum, Product):
                ea, _ = _abstract(ca, self.int_funcs, nn)
                eb, _ = _abstract(cb, self.int_funcs, nn)
                bq, _ = _abstract(b_n, self.int_funcs, nn)
                ga, gb, gq = [sorted(e.free_symbols, key=str) for e in (ea, eb, bq)]
                ints = (all(e.is_polynomial(*g) and _integer_valued_poly(e, g) for e, g in ((ea, ga), (eb, gb), (bq, gq)) if g)
                        and all(e.is_Integer for e, g in ((ea, ga), (eb, gb), (bq, gq)) if not g))
                if os.environ.get("OPENLAB_PROOF_DEBUG"):
                    print(f"[proof] ints={ints} ea={ea} eb={eb} bq={bq} div={divides(mv, ca - cb, self.int_funcs, nn, units)}", flush=True)
                if ints and divides(mv, ca - cb, self.int_funcs, nn, units)[0]:
                    ratio_ok = True
            if ratio_ok:
                step = (f"step: A({var}+1) = A({var})·({interp.pyform(ca)}) and B({var}+1) = B({var})·({interp.pyform(cb)}), "
                        f"integer factors with {interp.pyform(ca)} ≡ {interp.pyform(cb)} (mod {interp.pyform(mv)}); so "
                        f"A({var}) ≡ B({var}) ⟹ A({var}+1) ≡ B({var}+1)")
            elif self.holds_symbolic("cong", (a_1 - a_n) - (b_1 - b_n), mv, conds, notes, shift=0)[0]:
                step = (f"step: A({var}+1) − A({var}) ≡ B({var}+1) − B({var}) (mod {interp.pyform(mv)}), so "
                        f"A({var}) ≡ B({var}) ⟹ A({var}+1) ≡ B({var}+1)")
            elif self.holds_symbolic("cong", a_1 - b_1, mv, conds, notes, hyp=hyp)[0]:
                step = (f"step: A({var}+1) ≡ B({var}+1) (mod {interp.pyform(mv)}) once A({var}) ≡ B({var}) is used "
                        f"(the induction hypothesis)")
        else:
            j = Symbol("j", integer=True, nonnegative=True)                 # var = n0 + j
            at = {n: n0 + j}
            sign = 1 if rel in (sympy.LessThan, sympy.StrictLessThan) else -1
            da = self.structural(a_1 - a_n).xreplace(at)          # Σ_{k≤n+1} − Σ_{k≤n}: the new term
            db = self.structural(b_1 - b_n).xreplace(at)
            if self._ineq_nonneg(sign * (db - da), j):
                side, P, Q = ("right", "B", "A") if sign == 1 else ("left", "A", "B")
                step = (f"step: the {side} side grows at least as fast ({P}({var}+1) − {P}({var}) ≥ {Q}({var}+1) − {Q}({var}) "
                        f"for {var} ≥ {n0}), so the inequality carries from {var} to {var}+1")
            elif self._positive(a_n.xreplace(at)) and self._positive(b_n.xreplace(at)) and \
                    self._ineq_nonneg(sign * ((b_1 / b_n) - (a_1 / a_n)).xreplace(at), j):
                P, Q = ("B", "A") if sign == 1 else ("A", "B")
                step = (f"step: both sides are positive and {P}({var}+1)/{P}({var}) ≥ {Q}({var}+1)/{Q}({var}) for {var} ≥ {n0}, "
                        f"so the inequality carries from {var} to {var}+1")
            if step is None:
                step = self._mult_step(a_n, b_n, a_1, b_1, rel, claim, var)
        if step is None:
            return self._no_step(kind, A, B, mod, rel, var, n0, steps, notes, conds)
        steps.append(step)
        return sympy.true, steps

    def _moves(self, m, n, notes, ex):
        """Does the modulus m(n) differ from m(n+1)?"""
        if m is None or n.name not in {s.name for s in m.free_symbols}:
            return False
        return not self.zero(self.value(self.shift(m, n), notes, ex) - m, budget=2.0)

    def _no_step(self, kind, A, B, mod, rel, var, n0, steps, notes, conds):
        if not any(s.free_symbols - {var} for s in (A, B) if isinstance(s, sympy.Basic)) or kind != "cong":
            for k in range(n0 + 1, n0 + 6):          # not a proof: a search for a counterexample
                try:
                    bad, a, b = self._holds_at(kind, A, B, mod, rel, var, k, notes, conds)
                except OpenlabError:
                    break
                if bad is False:
                    rel_ = (f"≢ {interp.pyform(b)} (mod {interp.pyform(mod)})" if kind == "cong"
                            else f"≠ {interp.pyform(b)}" if kind == "eq" else f"fails against {interp.pyform(b)}")
                    return sympy.false, steps + [f"counterexample at {var} = {k}: {interp.pyform(a)} {rel_}"]
        return None, steps + ["inductive step not verified: neither the differences nor the ratios reduce; "
                              "the claim is not proved (split it into lemmas, or use strong induction with a step relation)"]

    def strong(self, claim: Claim, notes: list):
        """Strong induction with a step relation x_{n+1} = F(n, x_n, x_{n-1}, …): the first r cases
        hold, and both sides satisfy the relation, so the sides agree for every n."""
        var = self._var(claim)
        n0 = _base_of(claim.conds.get(var.name, ""), var)
        if n0 is None:
            raise SemanticsError(f"strong induction needs a starting whole number for {var}")
        kind, A, B, m = self._body(claim)
        if kind == "ineq":
            raise SemanticsError("strong induction with a step relation proves equations and congruences")
        rel_src = re.sub(r"^\s*(?:with|step)\b\s*:?", "", claim.detail).strip()
        if not rel_src:
            raise SemanticsError("give the step relation: `\\text{by strong induction with } x_{n+1} = 2n(x_n + x_{n-1})`")
        st = parse_statement(rel_src, 1)
        if st.lhs is None:
            raise SemanticsError("the step relation must be an equation x_{n+1} = F(n, x_n, …)")
        lhs, rhs = self.functional(st.lhs, [var]), self.functional(st.rhs, [var])
        rhs = self.ex._pre_rewrite(rhs, {var.name}, notes)        # 2n(x_n + …): a product, not a call
        if not (isinstance(lhs, AppliedUndef) and len(lhs.args) == 1):
            raise SemanticsError("the left side of the step relation must be one term, like x_{n+1}")
        X = lhs.func
        top = lhs.args[0]
        lags = sorted({sympy.simplify(top - f.args[0]) for f in rhs.atoms(AppliedUndef) if f.func == X})
        if not lags or any(not (l.is_Integer and l > 0) for l in lags):
            raise SemanticsError("the step relation must give x at the top index from earlier indices")
        r = int(max(lags))
        steps = []
        for k in range(n0, n0 + r):                   # the first r cases: computed (or from facts)
            ok, a, b = self._holds_at(kind, A, B, m, None, var, k, notes, claim.conds)
            steps.append(f"base case {var} = {k}: {interp.pyform(a)} {'≡' if kind == 'cong' else '='} "
                         f"{interp.pyform(b)} {'✓' if ok else '✗' if ok is False else '?'}")
            if ok is False:
                return sympy.false, steps + [f"the claim fails at {var} = {k}"]
            if ok is None:
                return None, steps + ["a base case could not be decided"]
        if kind == "cong" and self._moves(m.xreplace({var: Symbol(var.name, integer=True)}),
                                          Symbol(var.name, integer=True), notes, {var.name}):
            return None, steps + [f"the modulus {interp.pyform(m)} changes with {var}: strong induction on {var} "
                                  f"needs a modulus that does not depend on {var}; not proved"]
        if kind == "cong" and not self._integer_relation(rhs, X):
            return None, steps + ["for a congruence the step relation must have integer coefficients"]
        n = Symbol(var.name, integer=True)
        top_shift = sympy.simplify(top - var)
        self._step_var = var.name
        for side, S in (("left", A), ("right", B)):
            Sn = S.xreplace({var: n})

            def at(e):
                return Sn.xreplace({n: e})
            relation = rhs.xreplace({var: n}).replace(lambda f: isinstance(f, AppliedUndef) and f.func == X,
                                                      lambda f: at(f.args[0].xreplace({var: n})))
            diff = at(top.xreplace({var: n})) - relation
            diff = self.value(diff, notes, {var.name})
            conds = dict(claim.conds)
            shift = int(r - 1 - (top_shift - 1)) if top_shift.is_Integer else r - 1
            ok, why = self.holds_symbolic(kind, diff, m.xreplace({var: n}) if m is not None else None, conds, notes,
                                          shift=max(shift, r - 1))
            if not ok:
                return None, steps + [f"the {side} side does not satisfy the step relation"
                                      f"{' (mod ' + interp.pyform(m) + ')' if kind == 'cong' else ''}: "
                                      f"{interp.pyform(sympy.simplify(diff))} {'' if kind == 'eq' else '— ' + why}"]
            steps.append(f"the {side} side satisfies {interp.pyform(lhs)} = {interp.pyform(rhs)}"
                         f"{' (mod ' + interp.pyform(m) + ')' if kind == 'cong' else ''}")
        steps.append(f"both sides satisfy the same order-{r} relation and agree at {var} = {n0}…{n0 + r - 1}, "
                     f"so they agree for every {var} ≥ {n0}")
        return sympy.true, steps

    @staticmethod
    def _integer_relation(rhs, X) -> bool:
        vals = {f: sympy.Dummy() for f in rhs.atoms(AppliedUndef) if f.func == X}
        e = sympy.expand(rhs.xreplace(vals))
        gens = sorted(e.free_symbols, key=str)
        try:
            return all(c.is_Integer for c in sympy.Poly(e, *gens).coeffs()) if gens else e.is_Integer
        except Exception:
            return False

    def _ineq_nonneg(self, e, j) -> bool:
        raw = e
        e = sympy.simplify(e)
        if e.is_nonnegative:
            return True
        for form in (raw, e):                             # decreasing to a limit ≥ 0, or increasing from e(0) ≥ 0
            try:
                x_ = Symbol("x_mono", nonnegative=True)
                ex_ = form.xreplace({j: x_})
                dom = sympy.Interval(0, sympy.oo)
                lim = sympy.limit(ex_, x_, sympy.oo)
                if lim.is_nonnegative and sympy.is_decreasing(ex_, dom):
                    return True
                e0 = ex_.xreplace({x_: 0})
                if e0.is_nonnegative and sympy.is_increasing(ex_, dom):
                    return True
            except Exception:
                pass
        try:
            bad = sympy.solveset(sympy.StrictLessThan(e, 0), j, sympy.Interval(0, sympy.oo))
            return bad is sympy.S.EmptySet
        except Exception:
            return False

    @staticmethod
    def _positive(e) -> bool:
        return bool(sympy.simplify(e).is_positive)

    def algebra(self, claim: Claim, notes: list):
        kind, A, B, m = self._body(claim)
        if claim.subst:                                  # a case of `by cases on n mod q`
            A, B = A.xreplace(claim.subst), B.xreplace(claim.subst)
            if isinstance(m, sympy.Basic):
                m = m.xreplace(claim.subst)
        ups = [(k[len("upper:"):], c_) for k, c_ in claim.conds.items() if k.startswith("upper:")]
        if ups and not claim.subst and kind == "cong" and not getattr(self, "_in_range_try", False):
            c3 = dict(claim.conds)                         # k ∈ {1, …, p − 1} modulo a prime p: k is a unit
            for vname, hi_t in ups:
                try:
                    c3[f"range:{vname}"] = (Integer(_base_of(claim.conds.get(vname, ""), Symbol(vname)) or 0),
                                            parse_expr(hi_t)[0])
                except (OpenlabError, TypeError):
                    pass
            self._in_range_try = True
            try:
                val_, st_ = self.algebra(dataclasses.replace(claim, conds={k: v_ for k, v_ in c3.items()
                                                                           if not k.startswith("upper:")}), notes)
            finally:
                self._in_range_try = False
            if val_ is sympy.true:
                return val_, st_
        if ups and not claim.subst:                        # n ≤ q − 1: write q = n + 1 + q′ (q′ ≥ 0)
            sub_ = {}
            for vname, hi_t in ups:
                try:
                    hi_e = parse_expr(hi_t)[0]
                except OpenlabError:
                    continue
                for w in claim.vars:
                    if w.name != vname and w in hi_e.free_symbols and sympy.expand(hi_e - w).is_Integer:
                        c_ = sympy.expand(hi_e - w)            # n ≤ w + c  ⟺  w = n − c + w′
                        sub_[w] = Symbol(vname) - c_ + Symbol(w.name + "′", integer=True, nonnegative=True)
                        break
            if sub_:
                c2 = dataclasses.replace(claim, subst=sub_, conds={k: v_ for k, v_ in claim.conds.items()
                                                                 if not k.startswith("upper:")
                                                                 and k not in {w.name for w in sub_}})
                val_, st_ = self.algebra(c2, notes)
                how = ", ".join(f"{w} = {interp.pyform(e_)}" for w, e_ in sub_.items())
                return val_, [f"{st_[0] if st_ else ''} (with {how}, the primed letter ≥ 0)"] + st_[1:]
        from .transform import Pick

        def free_picks(e):                               # brackets outside every sum, integral or binder:
            if isinstance(e, Pick):                      # [cond(a)] inside Σ_a is not one case
                return [e]
            if isinstance(e, (Sum, Product, sympy.Integral, sympy.Lambda, sympy.ConditionSet)):
                return []
            out_ = []
            for x in getattr(e, "args", ()):
                if isinstance(x, sympy.Basic):
                    out_ += free_picks(x)
            return out_
        picks = free_picks(A) + free_picks(B)
        if picks and kind in ("eq", "ineq"):             # [P] ≤ f: the case P and the case not P
            p_ = sorted(picks, key=sympy.default_sort_key)[0]
            cond, val, neutral = p_.args
            true_sub = {p_: val}
            if isinstance(cond, sympy.Eq):
                for side, other in ((cond.lhs, cond.rhs), (cond.rhs, cond.lhs)):
                    if isinstance(side, Symbol) and side.name in {v.name for v in claim.vars} and not other.free_symbols:
                        true_sub[side] = other                 # [x = 0]: in its case, x is 0
                        break
            results = []
            for sub_, label in ((true_sub, f"{interp.pyform(cond)}"), ({p_: neutral}, f"not ({interp.pyform(cond)})")):
                c_ = dataclasses.replace(claim, subst={**(claim.subst or {}), **sub_})
                self.used.clear()
                val_, st = self.algebra(c_, [])
                if val_ is not sympy.true:
                    return None, [f"in the case {label}: " + "; ".join(st)[:200] + "; not proved"]
                results.append(f"{label}: {st[0] if st else 'holds'}")
            return sympy.true, ["split on the bracket — " + " | ".join(results)]
        dom = sympy.S.Reals
        c0 = claim.cond.replace(" ", "")
        if c0.startswith("\\in") and not re.search(r"mathbb", c0):
            try:
                dom = parse_expr(claim.cond[claim.cond.index("in") + 2:].strip())[0]
            except Exception:
                dom = sympy.S.Reals
        rep = {v: self._integer_symbol(v, claim.conds.get(v.name, "")) for v in claim.vars}
        if re.fullmatch(r"\\in\\mathbb\{C\}", c0):
            rep = {v: Symbol(v.name) for v in claim.vars}
        ex = {v.name for v in claim.vars}
        shifted = self._shift_to_zero(claim, rep) if kind in ("eq", "cong") else {}
        if shifted and (A.has(AppliedUndef) or B.has(AppliedUndef) or self._has_calls(A) or self._has_calls(B)):
            # k ≥ 1 written k = 1 + k′ with k′ ≥ 0: 2^k = 2·2^{k′} is visibly even, so a parity case decides
            a_s = self.value(A.xreplace(rep).xreplace(shifted), notes, ex)
            b_s = self.value(B.xreplace(rep).xreplace(shifted), notes, ex)
            how = ", ".join(f"{v} = {interp.pyform(e)}" for v, e in shifted.items())
            if kind == "eq" and self.holds_symbolic("eq", a_s - b_s, None, claim.conds, notes)[0]:
                return sympy.true, [f"A − B simplifies to 0 for every {', '.join(map(str, claim.vars))} "
                                    f"(written as {how}, each primed letter a whole number ≥ 0)"]
            if kind == "cong":
                mv = self.value(m.xreplace(rep).xreplace(shifted), notes, ex)
                ok, why = self.holds_symbolic("cong", a_s - b_s, mv, claim.conds, notes)
                if ok:
                    return sympy.true, [f"A − B ≡ 0 (mod {interp.pyform(mv)}) with {how}: {why}"]
        a, b = self.value(A.xreplace(rep), notes, ex), self.value(B.xreplace(rep), notes, ex)
        self._step_var = None
        if kind == "eq":
            ok, _ = self.holds_symbolic("eq", a - b, None, claim.conds, notes)
            if not ok:
                split = self._excluded_split(claim, rep, a - b, notes)
                if split:
                    return sympy.true, [f"A − B simplifies to 0 in each case: {split}"]
            if ok and (gaps := self._undefined(A.xreplace(rep), B.xreplace(rep), list(rep.values()), dom, claim.conds)):
                return None, [f"A − B simplifies to 0 where both sides are defined, but {gaps}: restrict the "
                              "domain to where they are; not proved"]
            if ok:
                return sympy.true, [f"A − B simplifies to 0 for every {', '.join(map(str, claim.vars))}"]
            d_ = _safe_simplify(a - b)
            if d_.is_number and d_.is_zero is False and not d_.has(sympy.nan, sympy.zoo):
                return sympy.false, [f"A − B = {interp.pyform(d_)}, a nonzero constant: it fails for every "
                                     f"{', '.join(map(str, claim.vars))}"]
            return None, [f"A − B = {interp.pyform(d_)} does not simplify to 0; not proved"]
        if kind == "cong":
            mv = self.value(m.xreplace(rep), notes, ex)
            ok, why = self.holds_symbolic("cong", a - b, mv, claim.conds, notes)
            return (sympy.true, [f"A − B ≡ 0 (mod {interp.pyform(mv)}): {why}"]) if ok \
                else ((sympy.false if ok is False else None), [f"not proved: {why}"])
        if len(claim.vars) != 1:
            return self._poly_inequality(claim, a, b, m, notes)
        v = rep[claim.vars[0]]
        cond = claim.conds.get(v.name, "")
        integer = bool(v.is_integer)
        a, b = self._indicator_rules(a, claim.conds), self._indicator_rules(b, claim.conds)
        a0, b0 = a, b
        if self._ineq_chain(claim, a0, b0, m):                  # X ≤ Y proved, Y = 3q proved: X ≤ 3q
            return sympy.true, ["by earlier results (an upper bound, then an equation)"]
        if self._termwise_from_fact(a0, b0, m, claim.conds):   # Σ f ≤ Σ g from a proved f ≤ g
            return sympy.true, ["each term satisfies the earlier inequality, so the sums do"]
        if m in (sympy.LessThan, sympy.StrictLessThan):         # A ≤ B: A at its largest, B at its smallest
            a, b = self._indicator_bound(a, True, claim.conds), self._indicator_bound(b, False, claim.conds)
        elif m in (sympy.GreaterThan, sympy.StrictGreaterThan):
            a, b = self._indicator_bound(a, False, claim.conds), self._indicator_bound(b, True, claim.conds)
        if m in (sympy.GreaterThan, sympy.StrictGreaterThan) or m in (sympy.LessThan, sympy.StrictLessThan):
            big = a0 if m in (sympy.GreaterThan, sympy.StrictGreaterThan) else b0
            for S_ in [n_ for n_ in big.atoms(Sum) if len(n_.limits) == 1 and self._nonneg_summand(n_.function)]:
                k_, lo_, hi_ = S_.limits[0]
                for off in range(3):                           # Σ of terms ≥ 0 is ≥ any one of them
                    if not self._nonneg(hi_ - lo_ - off, claim.conds):
                        break
                    one = S_.function.xreplace({k_: lo_ + off})
                    if m in (sympy.GreaterThan, sympy.StrictGreaterThan):
                        val_, st_ = self._ineq_one_var(claim, v, cond, integer, dom, a0.xreplace({S_: one}), b0, m)
                    else:
                        val_, st_ = self._ineq_one_var(claim, v, cond, integer, dom, a0, b0.xreplace({S_: one}), m)
                    if val_ is sympy.true:
                        return sympy.true, [f"every term of the sum is ≥ 0, so it is at least its term "
                                            f"{k_} = {interp.pyform(lo_ + off)}; then {st_[0] if st_ else 'it holds'}"]
        if (a, b) != (a0, b0):                                 # a bound proves, it never disproves
            val, steps = self._ineq_one_var(claim, v, cond, integer, dom, a, b, m)
            if val is sympy.true:
                return val, steps
            return None, ["the bound for the indicator terms is not enough to decide it; not proved"]
        return self._ineq_one_var(claim, v, cond, integer, dom, a, b, m)

    def _termwise_from_fact(self, a, b, m, conds):
        """Σ_{n=a}^{b} f ≤ Σ_{n=a}^{b} g from an earlier proved f(n) ≤ g(n) for all n ≥ a₀ (a ≥ a₀)."""
        if not (isinstance(a, Sum) and isinstance(b, Sum) and len(a.limits) == 1 and len(b.limits) == 1):
            return None
        (ja, la, ha), (jb, lb, hb) = a.limits[0], b.limits[0]
        if sympy.expand(la - lb) != 0 or sympy.expand(ha - hb) != 0:
            return None
        fb = b.function.xreplace({jb: ja})
        want = m if m in (sympy.LessThan, sympy.StrictLessThan) else {
            sympy.GreaterThan: sympy.LessThan, sympy.StrictGreaterThan: sympy.StrictLessThan}.get(m)
        fa = a.function if m in (sympy.LessThan, sympy.StrictLessThan) else fb
        fg = fb if m in (sympy.LessThan, sympy.StrictLessThan) else a.function
        for f in self.facts():
            if f["kind"] != "ineq" or len(f.get("vars", [])) != 1:
                continue
            rel = {"LessThan": sympy.LessThan, "StrictLessThan": sympy.StrictLessThan,
                   "GreaterThan": sympy.GreaterThan, "StrictGreaterThan": sympy.StrictGreaterThan}.get(f["rel"])
            FA, FB = f["A"], f["B"]
            if rel in (sympy.GreaterThan, sympy.StrictGreaterThan):
                FA, FB, rel = FB, FA, {sympy.GreaterThan: sympy.LessThan, sympy.StrictGreaterThan: sympy.StrictLessThan}[rel]
            if want is sympy.LessThan and rel not in (sympy.LessThan, sympy.StrictLessThan):
                continue
            v = f["vars"][0]
            vs = [s_ for s_ in (FA.free_symbols | FB.free_symbols) if s_.name == v]
            if not vs:
                continue
            FAj, FBj = FA.xreplace({vs[0]: ja}), FB.xreplace({vs[0]: ja})
            lo_f = _base_of(f["conds"].get(v, ""), Symbol(v))
            if lo_f is None or not self._nonneg(la - lo_f, conds):
                continue
            if self.zero(self.by_name(FAj, fa) - fa, budget=2.0) and self.zero(self.by_name(FBj, fg) - fg, budget=2.0):
                self.used.append(self._cite(f, "") + " term by term")
                return True
        return None

    def _ineq_chain(self, claim, a, b, m, depth=0):
        """A ≤ B from earlier results: an earlier A ≤ Y (so it suffices that Y ≤ B), and earlier equations
        used to rewrite either side (Y = 3q) -- transitivity."""
        if depth > 2 or m not in (sympy.LessThan, sympy.StrictLessThan, sympy.GreaterThan, sympy.StrictGreaterThan):
            return None
        if m in (sympy.GreaterThan, sympy.StrictGreaterThan):          # A ≥ B is B ≤ A
            a, b = b, a
            m = sympy.LessThan if m is sympy.GreaterThan else sympy.StrictLessThan
        conds = claim.conds
        b2 = self.apply_facts(b, None, conds, [])
        for f in self.facts():
            if f["kind"] != "ineq":
                continue
            rel = {"LessThan": sympy.LessThan, "StrictLessThan": sympy.StrictLessThan,
                   "GreaterThan": sympy.GreaterThan, "StrictGreaterThan": sympy.StrictGreaterThan}.get(f["rel"])
            FA, FB = (f["A"], f["B"]) if rel in (sympy.LessThan, sympy.StrictLessThan) else (f["B"], f["A"])
            for A_i, B_i, how in self._instances(dict(f, A=FA, B=FB), a, conds):
                if not self.zero(self.structural(a - A_i), budget=2.0):
                    continue
                y = self.apply_facts(B_i, None, conds, [])        # Y, rewritten by earlier equations
                if self.zero(self.structural(y - b2), budget=2.0) and (m is sympy.LessThan or
                                                                     rel is sympy.StrictLessThan):
                    self.used.append(self._cite(f, how) + ", then equal to the right side")
                    return True
                d_ = sympy.expand(self.structural(b2 - y))
                if d_.is_number and (d_ >= 0 if m is sympy.LessThan else d_ > 0):
                    self.used.append(self._cite(f, how) + f", and the right side exceeds it by {interp.pyform(d_)}")
                    return True
        return None

    def _ineq_one_var(self, claim, v, cond, integer, dom, a, b, m):
        if self._ineq_chain(claim, a, b, m):
            return sympy.true, ["by earlier results (an upper bound, then an equation)"]
        if self._termwise_from_fact(a, b, m, claim.conds):
            return sympy.true, ["each term satisfies the earlier inequality, so the sums do"]
        if a.has(Sum) or b.has(Sum):                      # the same sum on both sides (index names differ)
            d_ = self.structural(a - b)
            if d_ == 0 and m in (sympy.GreaterThan, sympy.LessThan):
                return sympy.true, ["the two sides are the same sum"]
        if integer:                                      # (4k+1)² mod 4 ≤ 1: decided for a whole-number k
            def settle(e):
                e = e.replace(lambda x: isinstance(x, sympy.Mod), lambda x: sympy.Mod(sympy.expand(x.args[0]), x.args[1]))
                return sympy.simplify(e)
            try:
                direct = m(settle(a), settle(b))
            except TypeError:
                direct = None
            if direct is sympy.true:
                return sympy.true, [f"it holds for every whole number {v} (the comparison simplifies to true)"]
        if not isinstance(dom, sympy.Set) or dom == sympy.S.Reals:
            dom = self._real_domain(v, cond, integer)
        x = Symbol(v.name, real=True)
        ax, bx = a.xreplace({v: x}), b.xreplace({v: x})
        try:
            bad = sympy.solveset(sympy.Not(m(ax, bx)), x, dom)
        except Exception:
            bad = None
        if integer:
            lo_ = dom.inf if isinstance(dom, sympy.Interval) else None
            where = (f"whole number {v} ≥ {lo_}" if lo_ is not None and lo_.is_finite and dom.sup == sympy.oo
                     else f"whole number {v}" if dom == sympy.S.Reals else f"whole number {v} in {interp.pyform(dom)}")
        else:
            where = f"{v} ∈ {interp.pyform(dom)}"
        if bad is sympy.S.EmptySet:
            return sympy.true, [f"no {where} violates it"]
        if isinstance(bad, sympy.Set) and not isinstance(bad, sympy.ConditionSet):
            if not integer:
                return sympy.false, [self._fails_on(bad, v, ax, bx, m, x)]
            try:
                ints = bad.intersect(sympy.S.Integers)
            except Exception:
                ints = None
            if ints is sympy.S.EmptySet:
                return sympy.true, [f"no {where} violates it: it fails only for {v} ∈ {interp.pyform(bad)}, "
                                    f"which contains no whole number"]
            k = self._an_element(ints)
            if k is not None:
                try:
                    holds = bool(m(ax.xreplace({x: k}), bx.xreplace({x: k})))
                except TypeError:
                    holds = None
                if holds is False:
                    return sympy.false, [f"counterexample at {v} = {k} (it fails for {v} ∈ {interp.pyform(bad)})"]
            return None, [f"it fails for {v} ∈ {interp.pyform(bad)} over the reals; whether a whole number "
                          f"there violates it was not decided; not proved"]
        got = self._poly_sign_on(ax, bx, m, x, dom, integer, v)       # solveset gave up: isolate the roots exactly
        if got is not None:
            return got
        return None, ["could not decide the inequality; not proved"]

    @staticmethod
    def _undefined(A, B, vs, dom, conds) -> str:
        """Where A or B is undefined (or breaks) inside a real domain: '' when nowhere, or when that can't be told."""
        from sympy.calculus.util import continuous_domain
        if not isinstance(dom, sympy.Set) or not dom.is_subset(sympy.S.Reals):
            return ""
        out = []
        for v in vs:
            where = _real_set(conds.get(v.name, ""))
            if v.is_real is not True or where is None or any(k in conds for k in (f"upper:{v.name}", f"lower:{v.name}")):
                continue                         # a condition this can't read: say nothing rather than guess
            for e in (A, B):
                if v not in e.free_symbols:
                    continue
                try:
                    gap = sympy.Complement(dom.intersect(where), continuous_domain(e, v, dom))
                    if v.is_integer:
                        gap = gap.intersect(sympy.S.Integers)
                except Exception:
                    continue
                if gap.is_empty is False:
                    out.append(f"{interp.pyform(e)} is undefined at {v} ∈ {gap}")
        return "; ".join(out)

    def _excluded_split(self, claim, rep, diff, notes):
        """A variable with one value excluded (k ∈ ℤ, k ≠ 0): prove the two cases k ≥ 1 and k ≤ −1
        (for a real variable, k = c ± t with t > 0) -- both must hold."""
        for v in claim.vars:
            c_ = claim.conds.get(v.name, "").replace(" ", "")
            mt = re.search(r"(?:\\setminus\\\{|\\neq?)(-?\d+)", c_)
            if not mt:
                continue
            c0 = Integer(int(mt.group(1)))
            sym = rep.get(v, v)
            if sym.is_integer:
                j = Symbol(f"{v.name}′", integer=True, nonnegative=True)
                cases = [(c0 + 1 + j, f"{v} = {c0 + 1} + {v}′"), (c0 - 1 - j, f"{v} = {c0 - 1} − {v}′")]
            else:
                t_ = Symbol(f"{v.name}′", positive=True)
                cases = [(c0 + t_, f"{v} > {c0}"), (c0 - t_, f"{v} < {c0}")]
            done = []
            for e_, how in cases:
                d_ = sympy.piecewise_fold(diff.xreplace({sym: e_}))
                if not (self.zero(d_) or self.zero(sympy.simplify(d_))):
                    return None
                done.append(how)
            return " and ".join(done) + ", each primed letter ≥ 0"
        return None

    def _poly_inequality(self, claim, a, b, rel, notes):
        """A ≤ B in several variables: B − A is a positive constant times even powers (so ≥ 0), or it
        equals the sum of squares named after `with` (`\\text{by algebra with } (ay - bx)^2`)."""
        if rel in (sympy.LessThan, sympy.StrictLessThan):
            D = sympy.expand(b - a)
        elif rel in (sympy.GreaterThan, sympy.StrictGreaterThan):
            D = sympy.expand(a - b)
        else:
            return None, ["not proved"]
        strict = rel in (sympy.StrictLessThan, sympy.StrictGreaterThan)
        if D.is_number:
            ok = D > 0 if strict else D >= 0
            return (sympy.true if ok else sympy.false), [f"B − A = {interp.pyform(D)}"]
        hint = re.sub(r"^\s*with\b\s*:?", "", claim.detail).strip() if claim.detail.strip().startswith("with") else ""
        if hint and not strict:
            try:
                terms = [parse_expr(t)[0] for t in _depth0_split(hint, re.compile(r",|\+"))]
                terms = [self.by_name(self.value(t, notes), D) for t in terms]
            except OpenlabError:
                terms = []
            if terms and all(self._square_like(t) or self._nonneg_term(t, claim) for t in terms) \
                    and sympy.expand(D - sympy.Add(*terms)) == 0:
                return sympy.true, [f"B − A = {' + '.join(interp.pyform(t) for t in terms)}, a sum of terms each ≥ 0"]
        f = sympy.factor(D)
        if not strict and self._square_like(f):
            return sympy.true, [f"B − A = {interp.pyform(f)} ≥ 0 (a positive constant times even powers)"]
        if not strict and self._nonneg_by_factors(D, claim):
            return sympy.true, [f"B − A = {interp.pyform(f)} ≥ 0: each term is a non-negative constant times factors "
                                f"that are ≥ 0 (even powers, or ≥ 0 by the variables' conditions)"]
        pos = self._positive_coefficients(claim, D, strict)
        if pos:
            return sympy.true, [pos]
        return None, [f"B − A = {interp.pyform(f)}: not recognised as ≥ 0 (name its terms, each a square or a product "
                      r"of factors ≥ 0 by the conditions: `\text{by algebra with } (ay - bx)^2 + n x^2`); not proved"]

    def _positive_coefficients(self, claim, D, strict):
        """Each variable written from an end of its range (v = lo + v′, or v = hi − v′ on an interval),
        v′ ≥ 0: a polynomial with every coefficient ≥ 0 (and a positive constant if strict) is ≥ 0 (> 0)."""
        from itertools import product as _product
        syms = {s_.name: s_ for s_ in D.free_symbols}
        options = []
        if any(s_.name not in {v.name for v in claim.vars} and not (s_.is_integer and s_.is_nonnegative)
               for s_ in D.free_symbols):
            return None
        for v in claim.vars:
            if v.name not in syms:
                continue
            c_ = claim.conds.get(v.name, "")
            sym = syms[v.name]
            prime = Symbol(v.name + "′", nonnegative=True)
            lo = _base_of(c_, v) if (v.name in _INTEGER_LETTERS or "mathbb{Z}" in c_ or "mathbb{N}" in c_) else None
            ends = []
            mt = re.fullmatch(r"\s*(?:\\ge|\\geq)\s*(-?\d+(?:\.\d+)?)\s*", c_) if lo is None else None
            if lo is not None:
                ends.append((lo + prime, f"{v} = {lo} + {v}′"))
            elif mt:
                a_ = sympy.nsimplify(mt.group(1))
                ends.append((a_ + prime, f"{v} = {a_} + {v}′"))
            else:
                mi = re.fullmatch(r"\s*\\in\s*\[\s*([^,\]]+),\s*([^\]]+)\]\s*", c_)
                if not mi:
                    return None
                try:
                    a_, b_ = parse_expr(mi.group(1))[0], parse_expr(mi.group(2))[0]
                except OpenlabError:
                    return None
                ends += [(a_ + prime, f"{v} = {interp.pyform(a_)} + {v}′"), (b_ - prime, f"{v} = {interp.pyform(b_)} − {v}′")]
            options.append([(sym, e, how) for e, how in ends])
        if len(options) > 6:
            return None
        if not options:                                  # only already-shifted letters (q′ ≥ 0) are left
            options = [[(None, None, "")]]
        for combo in _product(*options):
            E = sympy.expand(D.xreplace({sym: e for sym, e, _ in combo if sym is not None}))
            gens = sorted(E.free_symbols, key=str)
            try:
                poly = sympy.Poly(E, *gens) if gens else None
            except Exception:
                continue
            coeffs = poly.coeffs() if poly is not None else [E]
            const = E.xreplace({g: 0 for g in gens}) if gens else E
            if all(c_.is_number and c_ >= 0 for c_ in coeffs) and (not strict or const > 0):
                return (f"with {', '.join(how for _, _, how in combo if how) or 'the letters as they are'} (each primed letter ≥ 0), B − A = "
                        f"{interp.pyform(E)} has only non-negative coefficients{' and a positive constant' if strict else ''}")
        return None

    # --- inequalities: factors known to be ≥ 0, a multiplicative step, exact one-variable polynomials ------
    @staticmethod
    def _claim_lower(v, cond: str):
        """A variable's lower bound from its condition: a whole number's first value, `\\ge a` or `> a` for
        a real one, or the left end of `\\in [a, b]`; None when it has none."""
        c = cond.replace(" ", "")
        if v.name in _INTEGER_LETTERS or "mathbb{Z}" in c or "mathbb{N}" in c:
            lo = _base_of(cond, v)
            return None if lo is None else Integer(lo)
        mt = re.fullmatch(r"(?:\\ge|\\geq|>|\\gt)(-?\d+(?:\.\d+)?)", c)
        if mt:
            return sympy.nsimplify(mt.group(1))
        mi = re.fullmatch(r"\\in\[([^,\]]+),[^\]]+\]", c)
        if mi:
            try:
                return parse_expr(mi.group(1))[0]
            except OpenlabError:
                return None
        return None

    def _shifted_nonneg(self, e, claim) -> bool:
        """e ≥ 0 wherever its variables may be: each written from its lower bound (v = lo + v′, v′ ≥ 0),
        e expands to a polynomial whose coefficients are all ≥ 0."""
        sub = {}
        for s_ in e.free_symbols:
            v = next((w for w in claim.vars if w.name == s_.name), None)
            if v is None:
                if s_.is_nonnegative:
                    continue
                return False
            lo = self._claim_lower(v, claim.conds.get(v.name, ""))
            if lo is None:
                return False
            sub[s_] = lo + Symbol(s_.name + "′", nonnegative=True)
        E = sympy.expand(e.xreplace(sub))
        if E.is_number:
            return bool(E >= 0)
        try:
            poly = sympy.Poly(E, *sorted(E.free_symbols, key=str))
        except Exception:
            return bool(sympy.simplify(E).is_nonnegative)
        return all(c_.is_number and c_ >= 0 for c_ in poly.coeffs())

    def _nonneg_term(self, t, claim) -> bool:
        """A non-negative constant times factors that are each ≥ 0: even powers, or expressions ≥ 0 by
        the variables' conditions (n for n ≥ 1; 1 + x for x ≥ −1), raised to a positive power."""
        for fct in sympy.Mul.make_args(t):
            if fct.is_number:
                if not (fct.is_real and fct >= 0):
                    return False
                continue
            base, ex_ = (fct.base, fct.exp) if isinstance(fct, sympy.Pow) else (fct, sympy.Integer(1))
            if ex_.is_Integer and ex_ % 2 == 0 and base.is_real is not False:
                continue
            if ex_.is_positive and self._shifted_nonneg(base, claim):
                continue
            return False
        return True

    def _nonneg_by_factors(self, D, claim) -> bool:
        """D ≥ 0 as a sum of terms each of the form `_nonneg_term` accepts (D factored, or expanded)."""
        for form in (sympy.factor(D), sympy.expand(D)):
            if all(self._nonneg_term(t, claim) for t in sympy.Add.make_args(form)):
                return True
        return False

    def _base_ineq(self, a, b, rel, claim):
        """An induction's base case that still has a real parameter: decided the way `by algebra` would --
        sums with number bounds evaluated, then B − A = 0, a number, factors ≥ 0, or an exact polynomial check."""
        try:
            a2, b2 = sympy.expand(sympy.sympify(a).doit()), sympy.expand(sympy.sympify(b).doit())
        except Exception:
            return None, a, b
        le = rel in (sympy.LessThan, sympy.StrictLessThan)
        strict = rel in (sympy.StrictLessThan, sympy.StrictGreaterThan)
        D = sympy.simplify(b2 - a2 if le else a2 - b2)
        if D == 0:
            return (not strict), a2, b2
        if D.is_number:
            return bool(D > 0 if strict else D >= 0), a2, b2
        if not strict and self._nonneg_by_factors(D, claim):
            return True, a2, b2
        if len(D.free_symbols) == 1:
            s_ = next(iter(D.free_symbols))
            v = next((w for w in claim.vars if w.name == s_.name), None)
            if v is not None:
                cond = claim.conds.get(v.name, "")
                integer = bool(self._integer_symbol(v, cond).is_integer)
                x = Symbol(v.name, real=True)
                got = self._poly_sign_on(a2.xreplace({s_: x}), b2.xreplace({s_: x}), rel, x,
                                         self._real_domain(v, cond, integer), integer, v)
                if got is not None and got[0] in (sympy.true, sympy.false):
                    return got[0] is sympy.true, a2, b2
        return None, a2, b2

    def _mult_step(self, a_n, b_n, a_1, b_1, rel, claim, var):
        """The textbook step for A(n) ≥ B(n): A(n+1) = c·A(n) with c ≥ 0, so A(n+1) ≥ c·B(n), and it
        suffices that c·B(n) ≥ B(n+1) (for ≤: A(n+1) ≤ c·B(n) ≤ B(n+1)). Non-strict relations only."""
        if rel not in (sympy.GreaterThan, sympy.LessThan):
            return None
        try:
            c = sympy.simplify(sympy.powsimp(sympy.cancel(a_1 / a_n), force=True))
        except Exception:
            return None
        if c.has(Sum, Product, AppliedUndef) or not self._shifted_nonneg(c, claim):
            return None
        D = sympy.expand(c * b_n - b_1) if rel is sympy.GreaterThan else sympy.expand(b_1 - c * b_n)
        if not (D == 0 or self._nonneg_by_factors(D, claim)):
            return None
        sym = "≥" if rel is sympy.GreaterThan else "≤"
        cs = interp.pyform(c)
        return (f"step: A({var}+1) = A({var})·({cs}) with {cs} ≥ 0, so A({var}) {sym} B({var}) gives "
                f"A({var}+1) {sym} ({cs})·B({var}); and ({cs})·B({var}) {sym} B({var}+1) because the difference "
                f"{interp.pyform(sympy.factor(D))} is ≥ 0")

    def _poly_sign_on(self, ax, bx, m, x, dom, integer, v):
        """A one-variable polynomial inequality with rational coefficients, decided exactly: the real roots
        of B − A are isolated (sympy's real-root isolation), and its sign is fixed on every piece of the
        domain between them by evaluating it at a rational point there (and at the domain's closed ends)."""
        if m not in (sympy.LessThan, sympy.StrictLessThan, sympy.GreaterThan, sympy.StrictGreaterThan) or integer:
            return None
        if dom == sympy.S.Reals:
            dom = sympy.Interval(-sympy.oo, sympy.oo)
        if not isinstance(dom, sympy.Interval):
            return None
        le = m in (sympy.LessThan, sympy.StrictLessThan)
        strict = m in (sympy.StrictLessThan, sympy.StrictGreaterThan)
        D = sympy.expand((bx - ax) if le else (ax - bx))
        try:
            P = sympy.Poly(D, x)
            if P.degree() < 1 or P.degree() > 60 or not all(c_.is_rational for c_ in P.all_coeffs()):
                return None
            roots = [r for r in P.real_roots() if dom.contains(r) is sympy.true]
        except Exception:
            return None
        lo, hi = dom.inf, dom.sup
        marks = sorted(set(roots) | {p for p in (lo, hi) if p.is_finite}, key=lambda r: sympy.N(r, 60))
        pts = [self._simple_between(p, q) for p, q in zip(marks, marks[1:])]
        if lo == -sympy.oo:
            pts.append(sympy.floor(sympy.N(marks[0], 30)) - 1 if marks else Integer(0))
        if hi == sympy.oo:
            pts.append(sympy.ceiling(sympy.N(marks[-1], 30)) + 1 if marks else Integer(0))
        ends = [p for p, closed in ((lo, not dom.left_open), (hi, not dom.right_open)) if p.is_finite and closed]
        for t in pts + ends:
            val = P.eval(t)
            if val < 0 or (strict and val == 0):
                return sympy.false, [f"counterexample at {v} = {interp.pyform(t)}: B − A = {interp.pyform(val)}"]
        if strict:
            inside = [r for r in roots if r not in (lo, hi) or r in ends]
            if inside:
                r0 = inside[0]
                near = "" if r0.is_Rational else f" ≈ {sympy.N(r0, 8)}"
                return sympy.false, [f"it fails at {v} = {interp.pyform(r0)}{near}, where B − A = 0"]
        return sympy.true, [f"B − A = {interp.pyform(sympy.factor(D))}: a polynomial whose {len(roots)} real root(s) in "
                            f"{interp.pyform(dom)} were isolated exactly; it is {'>' if strict else '≥'} 0 on every piece "
                            f"between them"]

    def _fails_on(self, bad, v, ax, bx, m, x) -> str:
        """Where a one-variable inequality fails: a short point there, checked exactly, and the set --
        its ends shown as decimals when their exact form is long (a quartic's roots are)."""
        shown = interp.pyform(bad)
        if len(shown) > 80:
            parts = bad.args if isinstance(bad, sympy.Union) else (bad,)
            if all(isinstance(p, sympy.Interval) for p in parts):
                shown = " ∪ ".join(
                    f"{'(' if p.left_open else '['}{sympy.N(p.inf, 6) if not p.inf.is_Rational else p.inf}, "
                    f"{sympy.N(p.sup, 6) if not p.sup.is_Rational else p.sup}{')' if p.right_open else ']'}" for p in parts)
        point = None
        try:
            first = bad.args[0] if isinstance(bad, sympy.Union) else bad
            if isinstance(first, sympy.Interval):
                lo_, hi_ = first.inf, first.sup
                if lo_ == -sympy.oo:
                    point = sympy.floor(sympy.N(hi_, 30)) - 1
                elif hi_ == sympy.oo:
                    point = sympy.ceiling(sympy.N(lo_, 30)) + 1
                else:
                    point = self._simple_between(lo_, hi_)
                if point is not None and bool(m(ax.xreplace({x: point}), bx.xreplace({x: point}))) is not False:
                    point = None
        except Exception:
            point = None
        where = f"it fails for {v} ∈ {shown}"
        return f"counterexample at {v} = {interp.pyform(point)} ({where})" if point is not None else where

    @staticmethod
    def _simple_between(p, q):
        """The shortest-looking rational strictly between p < q: an integer if there is one, else a fraction
        with a small denominator -- so a counterexample reads x = -1/2, not a 60-digit fraction."""
        pf, qf = sympy.N(p, 60), sympy.N(q, 60)
        for den in (1, 2, 4, 5, 10, 20, 100, 1000, 10**4, 10**6, 10**9):
            t = sympy.Rational(sympy.floor(pf * den) + 1, den)
            if pf < t < qf:
                return t
        return sympy.Rational(str(sympy.N((p + q) / 2, 60)))

    @staticmethod
    def _square_like(t):
        """c · ∏ f_i^{2k_i} with c > 0 (or 0)."""
        for fct in sympy.Mul.make_args(t):
            if fct.is_number:
                if not (fct.is_real and fct >= 0):
                    return False
            elif isinstance(fct, sympy.Pow) and fct.exp.is_Integer and fct.exp % 2 == 0 and fct.base.is_real is not False:
                continue
            else:
                return False
        return True

    @staticmethod
    def _has_calls(e):
        from .transform import Apply
        return isinstance(e, sympy.Basic) and e.has(Apply)

    def _shift_to_zero(self, claim, rep):
        """Whole-number variables with a lower bound lo ≠ 0 -> lo + v′ (v′ ≥ 0), as a substitution."""
        out = {}
        for v in claim.vars:
            cond = claim.conds.get(v.name, "")
            sym = rep.get(v, v)
            lo = _base_of(cond, v)
            if not sym.is_integer or lo is None or lo == 0 or cond.replace(" ", "") in ("odd", "even"):
                continue
            out[sym] = lo + Symbol(v.name + "′", integer=True, nonnegative=True)
        return out

    @staticmethod
    def _an_element(S):
        """A concrete integer in S (a Range, a finite set, or a union of them), else None."""
        try:
            if isinstance(S, sympy.FiniteSet):
                return min(S, key=abs) if S else None
            if isinstance(S, sympy.Range):
                for end in (S.inf, S.sup):
                    if end.is_Integer:
                        return end
            if isinstance(S, sympy.Union):
                for part in S.args:
                    e = Prover._an_element(part)
                    if e is not None:
                        return e
        except Exception:
            return None
        return None

    def _real_domain(self, v, cond, integer):
        """The claim's range as a real set: `n \\ge 1` -> [1, ∞) (for a whole-number n, `> 0` -> [1, ∞));
        ℕ -> [0, ∞); ℤ, ℝ -> ℝ; `x > 0` for a real x -> (0, ∞)."""
        c = cond.replace(" ", "")
        if integer:
            lo = _base_of(cond, v)
            return sympy.Interval(lo, sympy.oo) if lo is not None else sympy.S.Reals
        mt = re.fullmatch(r"(\\ge|\\geq|>|\\gt|\\le|\\leq|<|\\lt)(-?\d+(?:\.\d+)?)", c)
        if mt:
            op, val = mt.group(1), sympy.nsimplify(mt.group(2))
            return {"\\ge": sympy.Interval(val, sympy.oo), "\\geq": sympy.Interval(val, sympy.oo),
                    ">": sympy.Interval.Lopen(val, sympy.oo), "\\gt": sympy.Interval.Lopen(val, sympy.oo),
                    "\\le": sympy.Interval(-sympy.oo, val), "\\leq": sympy.Interval(-sympy.oo, val),
                    "<": sympy.Interval.Ropen(-sympy.oo, val), "\\lt": sympy.Interval.Ropen(-sympy.oo, val)}[op]
        return sympy.S.Reals

    def cases(self, claim: Claim, notes: list):
        """`by cases on n mod q`: n = qk + r for r = 0, …, q − 1, each case by algebra (k from where
        n reaches its lower bound; if a case needs it, its first few values are computed)."""
        mt = re.match(r"\s*\\?([A-Za-z]+)\s*(?:\\bmod|\\mod|mod|\\pmod)\s*\{?\s*(\d+)\s*\}?", claim.detail)
        if not mt:
            raise SemanticsError(r"say which residues: `\text{by cases on } n \bmod 6`")
        name, q = mt.group(1), int(mt.group(2))
        v = next((x for x in claim.vars if x.name == name), None)
        if v is None or q < 1 or q > 1000:
            raise SemanticsError(f"`by cases on {name} mod {q}`: {name} must be a variable of the claim, q ≤ 1000")
        lo = _base_of(claim.conds.get(name, ""), v)
        if lo is None:
            raise SemanticsError(f"by cases needs a whole-number range for {name} (`{name} \\ge 0`, `{name} \\in \\mathbb{{N}}`)")
        used_names = {x.name for x in claim.vars}
        kname = next(c for c in ("k", "j", "t", "u", "w", "s") if c not in used_names)
        k = Symbol(kname, integer=True)                  # (6k + 5) mod 6 decides at once
        steps = []
        for r in range(q):
            k0 = -((r - lo) // q)                        # smallest k with qk + r ≥ lo
            proved = False
            for extra in range(0, 4):
                start = k0 + extra
                for kk in range(k0, start):              # the first few values: computed
                    sub = {v: Integer(q * kk + r)}
                    kind, A, B, m = self._body(claim)
                    ok, a, b = self._holds_at(kind, A.xreplace(sub), B.xreplace(sub),
                                              m.xreplace(sub) if kind == "cong" else None,
                                              m if kind == "ineq" else None, Symbol("_none"), 0, [])
                    if ok is False:
                        return sympy.false, steps + [f"counterexample at {name} = {q * kk + r}"]
                    if ok is not True:
                        break
                conds = {x: c for x, c in claim.conds.items() if x != name}
                conds[kname] = rf"\ge {start}"
                sub_claim = dataclasses.replace(
                    claim, vars=[k if x.name == name else x for x in claim.vars], conds=conds, technique="algebra",
                    detail="", subst={Symbol(name): q * k + r})
                self.used.clear()
                val, st = self.algebra(sub_claim, [])
                if val is sympy.true:
                    first = (f"{name} = {', '.join(str(q * kk + r) for kk in range(k0, start))} computed; "
                             if start > k0 else "")
                    steps.append(f"{name} ≡ {r} (mod {q}): {first}{name} = {q}{kname} + {r} with {kname} ≥ {start}: "
                                 + "; ".join(st + list(dict.fromkeys(self.used))))
                    proved = True
                    break
            if not proved:
                return None, steps + [f"the case {name} ≡ {r} (mod {q}) was not proved; not proved"]
        self.used.clear()
        return sympy.true, [f"every {name} ≥ {lo} is {q}{kname} + r for one r in 0…{q - 1}; all {q} cases hold"] + steps

    def _exhaust_on(self, claim, names, notes):
        """`by exhaustion on r` (or `on r, j`): those variables run over their finite sets; for each case
        the claim in the other variables is proved by algebra (so ∀m ∈ ℤ is covered symbolically)."""
        from itertools import product as _product
        if isinstance(names, str):
            names = [names]
        doms = []
        for name in names:
            v = next((x for x in claim.vars if x.name == name), None)
            if v is None:
                raise SemanticsError(f"`by exhaustion on {name}`: {name} is not a variable of the claim")
            S_ = self._finite_domain(v, claim.conds.get(name, ""), notes)
            if S_ is None:
                raise SemanticsError(f"`by exhaustion on {name}` needs a finite set for {name}",
                                     hint=rf"write `\forall {name} \in \{{0, \ldots, 15\}}, m \in \mathbb{{Z}}: …`.")
            doms.append((name, S_))
        total = 1
        for _, S_ in doms:
            total *= len(S_)
        if total > 20_000:
            raise SemanticsError(f"`by exhaustion on {', '.join(names)}`: {total} cases, each by algebra, is too many")
        rest = [x for x in claim.vars if x.name not in names]
        conds = {k: c for k, c in claim.conds.items() if k not in names}
        done = 0
        for pt in _product(*[list(S_) for _, S_ in doms]):
            sub = {Symbol(nm): Integer(int(x)) for (nm, _), x in zip(doms, pt)}
            sub_claim = dataclasses.replace(claim, vars=rest, conds=conds, technique="algebra", detail="", subst=sub)
            self.used.clear()
            val, st = self.algebra(sub_claim, [])
            if val is not sympy.true:
                why = "; ".join(st)[:200]
                verdict = sympy.false if val is sympy.false else None
                where = ", ".join(f"{nm} = {x}" for (nm, _), x in zip(doms, pt))
                return verdict, [f"{where}: {'fails' if verdict is sympy.false else 'not proved'} ({why})"]
            done += 1
        self.used.clear()

        def shown(S_):
            return f"{{{S_.inf}, …, {S_.sup}}}" if isinstance(S_, sympy.Range) else interp.pyform(S_)
        return sympy.true, [f"checked all {done} cases ({'; '.join(f'{nm} ∈ {shown(S_)}' for nm, S_ in doms)}), "
                            f"each for every {', '.join(map(str, rest))} by algebra"]

    def _finite_domain(self, v, cond, notes):
        """The finite set a variable ranges over (`\\in \\{2, …, 500\\}`, `\\in \\{1, 3, 5\\}`), else None."""
        c = cond.strip()
        mt = re.match(r"^\\in\s*(.+)$", c, re.S)
        if not mt:
            return None
        try:
            S_ = self.value(parse_expr(mt.group(1))[0], notes, structural=False)
        except OpenlabError:
            return None
        if isinstance(S_, sympy.FiniteSet) or (isinstance(S_, sympy.Range) and S_.is_finite_set):
            return S_
        return None

    def exhaustion(self, claim: Claim, notes: list):
        """Every case of a finite domain, computed: a proof for a finite set (and only for one)."""
        from itertools import product as _product
        from . import fasteval as _fast
        on = re.match(r"\s*on\s+((?:\\?[A-Za-z]+\s*,\s*)*\\?[A-Za-z]+)", claim.detail or "")
        if on and len(claim.vars) > 1:                   # by exhaustion on r (or r, j): each case, the rest by algebra
            names_ = [x.strip().lstrip("\\") for x in on.group(1).split(",")]
            if len(names_) < len(claim.vars):
                return self._exhaust_on(claim, names_, notes)
        doms = []
        for v in claim.vars:
            S_ = self._finite_domain(v, claim.conds.get(v.name, ""), notes)
            if S_ is None:
                raise SemanticsError(
                    f"proof by exhaustion needs a finite set for {v} (it has `{claim.conds.get(v.name, '') or 'no set'}`)",
                    hint=r"write `\forall m \in \{2, 3, \ldots, 500\}: … \text{ by exhaustion}`; an infinite "
                         r"set is never proved by checking cases -- use induction or algebra.")
            doms.append((v, S_))
        total = 1
        for _, S_ in doms:
            total *= len(S_)
        if total > 20_000_000:
            raise SemanticsError(f"proof by exhaustion over {total} cases is too many to check")
        kind, A, B, m = self._body(claim)
        ex_names = {v.name for v in claim.vars}
        scratch: list = []                        # "no closed form" for the symbolic side: not a finding
        a, b = self.value(A, scratch, ex_names, structural=False), self.value(B, scratch, ex_names, structural=False)
        mv = self.value(m, scratch, ex_names, structural=False) if kind == "cong" else None
        if kind == "eq":
            rel = sympy.Eq(a, b, evaluate=False)
        elif kind == "cong":
            rel = sympy.Eq(sympy.Mod(a - b, mv, evaluate=False), 0, evaluate=False)
        else:
            rel = m(a, b, evaluate=False)
        try:
            fast = _fast.compile_expr(rel, ex_names, self.ex._fast_hook)
        except _fast.Unsupported:
            fast = None
        names = [v.name for v, _ in doms]
        checked = 0
        for point in _product(*[list(S_) for _, S_ in doms]):
            env = {nm: int(x) for nm, x in zip(names, point)}
            ok = None
            if fast is not None:
                try:
                    ok = bool(fast(env))
                except (_fast.Unsupported, ZeroDivisionError, OverflowError, ValueError, TypeError, KeyError):
                    ok = None
            if ok is None:
                sub = {v: Integer(x) for (v, _), x in zip(doms, point)}
                got, _a, _b = self._holds_at(kind, A.xreplace(sub), B.xreplace(sub),
                                             m.xreplace(sub) if kind == "cong" else None,
                                             m if kind == "ineq" else None, Symbol("_none"), 0, scratch)
                ok = got
            where = ", ".join(f"{nm} = {x}" for nm, x in zip(names, point))
            if ok is None:
                return None, [f"checked {checked} cases; the case {where} could not be decided; not proved"]
            if claim.quant == "exists" and ok:
                return sympy.true, [f"found {where} after checking {checked + 1} cases"]
            if claim.quant == "forall" and not ok:
                return sympy.false, [f"counterexample at {where} (after {checked} cases held)"]
            checked += 1
            self._progress = (checked, total)
        if claim.quant == "exists":
            return sympy.false, [f"none of the {checked} cases satisfies it"]
        def shown(S_):
            if isinstance(S_, sympy.Range) and S_.step == 1:
                return f"{{{S_.inf}, …, {S_.sup}}}"
            if isinstance(S_, sympy.FiniteSet) and len(S_) > 8:
                el = sorted(S_, key=lambda x: (float(x) if x.is_real else 0))
                return f"{{{', '.join(map(str, el[:3]))}, …, {el[-1]}}} ({len(S_)} elements)"
            return interp.pyform(S_)
        span = "; ".join(f"{nm} ∈ {shown(S_)}" for nm, (_, S_) in zip(names, doms))
        return sympy.true, [f"checked all {checked} cases ({span}): each holds"]

    def witness(self, claim: Claim, notes: list):
        if claim.quant != "exists":
            raise SemanticsError(r"a witness proves an `\exists` claim", hint=r"`\exists n: n^2 = 49 \text{ witness } n = 7`.")
        m = re.findall(r"(\\?[A-Za-z]+(?:_\{?\w+\}?)?)\s*=\s*([^,;]+)", claim.detail)
        if not m:
            raise SemanticsError("name the witness: `\\text{witness } n = 7`")
        sub = {Symbol(k.lstrip("\\")): parse_expr(v)[0] for k, v in m}
        kind, A, B, mod = self._body(claim)
        a, b = self.value(A.xreplace(sub), notes), self.value(B.xreplace(sub), notes)
        if kind == "eq":
            ok = self.zero(a - b)
        elif kind == "cong":
            ok, _ = divides(self.value(mod.xreplace(sub), notes), a - b, self.int_funcs)
        else:
            try:
                ok = bool(mod(a, b))
            except TypeError:
                ok = None
        where = ", ".join(f"{k} = {interp.pyform(v)}" for k, v in sub.items())
        if ok:
            return sympy.true, [f"the witness {where} satisfies it"]
        return None, [f"the witness {where} does not satisfy it (another value might)"]


def _claim_text(claim) -> str:
    """The claim as written, without its technique (kept: `\\text{ prime}` and other conditions)."""
    t = re.sub(r"\\openlabnote\{[^{}]*\b(?:by|witness)\b[^{}]*\}.*$", "", claim.src, flags=re.S)
    t = re.sub(r"\\openlabnote\{([^{}]*)\}", lambda m: rf"\text{{ {m.group(1).strip()}}}", t)
    return re.sub(r"\s+", " ", t).strip().rstrip(".")


def record_fact(ex, claim: Claim, line: int):
    """A proved ∀-claim becomes a fact later proofs may use (stored resolved, so that a defined
    S(n) is its sum)."""
    old_shadow = getattr(ex, "_shadow", frozenset())
    ex._shadow = frozenset(old_shadow) | {v.name for v in claim.vars}     # the same names the proof saw
    try:
        pr = Prover(ex)
        kind, A, B, m = pr._body(claim)
        ex_ = {v.name for v in claim.vars}
        ints = {v: pr._integer_symbol(v, claim.conds.get(v.name, "")) for v in claim.vars}
        ints = {v: s_ for v, s_ in ints.items() if s_.is_integer}      # T^{∘10}(1024m − 1): parity decided
        A, B = pr.value(A.xreplace(ints), [], ex_), pr.value(B.xreplace(ints), [], ex_)
        A, B = A.xreplace({s_: v for v, s_ in ints.items()}), B.xreplace({s_: v for v, s_ in ints.items()})
        if isinstance(m, sympy.Basic):
            m = pr.value(m.xreplace(ints), [], ex_).xreplace({s_: v for v, s_ in ints.items()})
    except OpenlabError:
        return
    finally:
        ex._shadow = old_shadow
    if kind not in ("eq", "cong", "ineq"):
        return
    reg = ex._reg("_openlab_facts").setdefault("list", [])
    if kind == "ineq":                                     # A ≤ B for every n ≥ a: summable term by term later
        reg.append({"kind": "ineq", "A": A, "B": B, "rel": m.__name__ if isinstance(m, type) else str(m),
                    "conds": dict(claim.conds), "vars": [v.name for v in claim.vars], "line": line,
                    "src": _claim_text(claim)})
        return
    reg.append({"kind": kind, "A": A, "B": B, "m": m, "conds": dict(claim.conds),
                "vars": [v.name for v in claim.vars], "line": line,
                "src": _claim_text(claim)})
