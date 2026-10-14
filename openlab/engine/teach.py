r"""Near-miss teaching: the idiom that actually works, for the things people reasonably try.

Two tables:

* ``PRECHECK`` runs *before* parsing on shapes that would otherwise parse into nonsense
  (``sin(x)`` is the product s*i*n applied to nothing useful) or fail obscurely.
* ``POSTFAIL`` runs when the grammar rejects a statement, most specific first; the first
  match wins.
"""

from __future__ import annotations

import re
from typing import Optional

_MASK = re.compile(r"\\(?:text|mathrm|mathit|textit|mathbf|operatorname|begin|end|openlabnote|openlabunit)\{[^{}]*\}"
                   r"|_\{[^{}]*\}")

_BARE_CMDS = ("arcsin|arccos|arctan|sinh|cosh|tanh|sin|cos|tan|ln|log|exp|sqrt|lim|sum|prod|"
              "int|frac|infty|inf|det|pi")


def _mask(src: str) -> str:
    """Blank out ``\\text{...}`` bodies so words inside them are not flagged."""
    return _MASK.sub(lambda m: "X" + " " * (len(m.group()) - 1), src)     # still an operand


def precheck(src: str) -> Optional[tuple[str, str]]:
    """``(message, hint)`` for a shape that must not reach the parser, else ``None``."""
    s = _mask(src)
    if "**" in s:
        return ("`**` is Python, not LaTeX", "powers are written with `^`: `x^{2}`.")
    m = re.search(rf"(?<![\\A-Za-z])({_BARE_CMDS})(?![A-Za-z])", s, flags=re.IGNORECASE)
    if m:
        w = m.group(1)
        fix = {"inf": r"\infty"}.get(w.lower(), "\\" + w.lower())
        return (f"`{w}` needs a backslash",
                f"write `{fix}` -- without it, `{w}` is the product of the letters "
                f"{'·'.join(w)}.")
    if re.search(r"\^\s*(?:\{[^{}]*\}|[^{\s])\s*\^", s):
        return ("a double superscript is not LaTeX",
                "brace the outer exponent: `a^{b^c}`.")
    if re.search(r"(?<![\\\w])[A-Za-z]_\d{2,}", s):
        return ("a multi-digit subscript needs braces",
                "LaTeX subscripts take one character unless braced: `x_10` is `x_1 · 0`; write `x_{10}`.")
    if re.search(r"\\sqrt\s*\d{2,}", s):
        return ("a multi-digit radicand needs braces",
                "`\\sqrt 16` is `\\sqrt{1} · 6` in LaTeX; write `\\sqrt{16}`.")
    if re.search(r"\^\d{2,}", s):
        return ("a multi-digit exponent needs braces",
                "LaTeX exponents take one character unless braced: `x^10` is `x^1 · 0`; write `x^{10}`.")
    s_ = re.sub(r",\s*\\(?:cdots|ldots|dots)\s*,", ",", s)       # \{1, 2, \ldots, n\}: an integer range
    if re.search(r"\\(?:cdots|ldots|dots|vdots|ddots)(?![A-Za-z])", s_):
        return ("an ellipsis is not executable",
                r"write the pattern as a sum or product: `\sum_{k=1}^{n} k`, or list the terms explicitly.")
    if s.count("{") != s.count("}"):
        return ("unbalanced braces", "every `{` needs a matching `}`.")
    if len(re.findall(r"\\left\b", s)) != len(re.findall(r"\\right\b", s)):
        return (r"unbalanced \left / \right", r"every `\left(` needs a matching `\right)`.")
    return None


_HAS_DIFF = re.compile(r"(?<![A-Za-z\\])d\s*(?:\{[A-Za-z]\}|[A-Za-z](?![A-Za-z]))|\\(?:mathrm|text)\{d\}")

POSTFAIL: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\\int[^$]*\\frac\{[^}]*(?<![A-Za-z])d[A-Za-z]"),
     r"put the differential after the fraction: `\int \frac{1}{x} \, dx`, not `\int \frac{dx}{x}`."),
    (re.compile(r"\\frac\{d(?:\^\d)?\}\{d[A-Za-z](?:\^\d)?\}\s*$"),
     r"`\frac{d}{dx}` needs an expression to differentiate: `\frac{d}{dx} x^2`, or write `\frac{dy}{dx}` with y bound."),
    (re.compile(r"\\lim(?![a-zA-Z])(?!\s*_)"),
     r"`\lim` needs its variable and target: `\lim_{x \to 0} \frac{\sin x}{x}`."),
    (re.compile(r"\\(?:sum|prod)(?!\s*_\s*\{[^}]*=)"),
     r"`\sum` needs bounds to be computable: `\sum_{i=1}^{n} x_i`."),
    (re.compile(r"\\(?:iint|iiint)\s*_"),
     r"give explicit bounds for each variable: `\int_0^1 \int_0^1 f \, dx \, dy`."),
    (re.compile(r"\\forall|\\exists"),
     r"a quantifier needs the shape `\forall x \in S, P(x)` (comma or colon before the statement)."),
    (re.compile(r"\\mid(?![a-zA-Z])"),
     r"`\mid` belongs in set-builder notation `\{ x \mid x > 0 \}`, a conditional `P(A \mid B)`, or a divisibility statement `3 \mid 12`."),
    (re.compile(r"\\(?:pmod|bmod)"),
     r"congruences are `a \equiv b \pmod{n}`; a remainder is `a \bmod n`."),
    (re.compile(r"\\(?:sin|cos|tan|ln|log|exp|sinh|cosh|tanh)(?:\^\S+)?\s*[A-Za-z0-9\\]+\s*\("),
     r"a bare argument cannot be followed by parentheses: write `\sin(x) (y+1)` or `\sin(x (y+1))`."),
    (re.compile(r"\\(?:sin|cos|tan|ln|log|exp|sqrt|sinh|cosh|tanh|det)\s*$"),
     "the function at the end has no argument."),
    (re.compile(r"\\frac(?:\s*\{[^{}]*\})?\s*$"),
     r"`\frac` takes two arguments: `\frac{numerator}{denominator}`."),
]


def teaching_hint(src: str) -> str:
    """The lesson for a statement the grammar rejected, or ``''``."""
    if re.search(r"\\(?:i{0,2}nt|oint)(?![a-zA-Z])", src) and not _HAS_DIFF.search(src):
        return (r"an integral needs its differential to say which variable and where the "
                r"integrand ends: `\int x^2 \, dx`.")
    m = re.search(r"(?<![{\\A-Za-z])([A-Za-z][A-Za-z0-9]*(?:\\_[A-Za-z0-9]+)+)", src)
    if m and not re.search(r"\\(?:mathrm|text|operatorname|mathit)\s*\{[^{}]*" + re.escape(m.group(1)), src):
        return (rf"a name with an underscore is written upright: `\mathrm{{{m.group(1)}}}` "
                rf"(bare letters are a product, and `\_` outside a name is not math).")
    for pattern, lesson in POSTFAIL:
        if pattern.search(src):
            return lesson
    if _HAS_DIFF.search(src) and not re.search(r"\\(?:i{0,2}nt|oint)(?![a-zA-Z])|\\frac\{d", src):
        return (r"`dx` is a differential, only meaningful as `\int ... dx` or `\frac{d}{dx}`. "
                r"For the product d·x write `d \cdot x`.")
    return ""


_SYMPY_NAMES = {
    "sqrt": r"\sqrt{...}", "sin": r"\sin x", "cos": r"\cos x", "tan": r"\tan x",
    "log": r"\log x", "ln": r"\ln x", "exp": r"\exp(x) or e^{x}", "abs": r"\lvert x \rvert",
    "det": r"\det A", "floor": r"\lfloor x \rfloor", "ceil": r"\lceil x \rceil",
    "sinh": r"\sinh x", "cosh": r"\cosh x", "tanh": r"\tanh x",
}


def did_you_mean(name: str) -> str:
    """For an undefined function whose name is a known math function spelled without ``\\``."""
    fix = _SYMPY_NAMES.get(name.lower())
    if fix:
        return f"`{name}(...)` was created as an undefined function -- did you mean `{fix}`?"
    return ""
