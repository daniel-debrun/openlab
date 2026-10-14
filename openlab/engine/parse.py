r"""From cell text to :class:`Statement` objects: preprocessing, splitting, parsing.

Preprocessing is textual and conservative. It only rewrites what has one reading:
wrappers (``$$``, ``\[``, ``align``), sizing commands, unicode symbols, thousands
separators, physical units after a number (``3\,\text{m/s}``), definitional equals
(``:=``) and prose annotations -- the last are kept as ``\openlabnote{...}`` markers so the
interpretation line can say they were ignored instead of dropping them silently.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field, replace
from typing import Optional

import sympy
from lark.exceptions import UnexpectedCharacters, UnexpectedEOF, UnexpectedInput, VisitError

from . import units as _units
from .errors import OpenlabError, ParseError
from .grammar import DECO_NAMES, FUNC_NAMES, GRAMMAR, GREEK_NAMES, build_parser
from .grammar import dialect as get_dialect
from .teach import precheck, teaching_hint
from .transform import ToSympy


@dataclass
class Clause:
    """One statement of a cell, before parsing."""
    lineno: int
    text: str
    display: str = ""                     # what to render as the input (defaults to text)
    role: str = ""                        # "" | "derived" (after \implies) | "declare"
    ics: list = field(default_factory=list)   # initial conditions for a differential equation
    domain: str = ""                      # `\sin x = 1/2, 0 \le x < 2\pi`: where to look for solutions


@dataclass
class Statement:
    src: str                              # the cleaned statement
    lineno: int
    lhs_src: Optional[str]                # text left of a top-level `=` (None if no `=`)
    lhs: Optional[sympy.Basic]            # that text parsed with the same grammar
    rhs: sympy.Basic                      # the expression (right of `=`, or the whole thing)
    approx: bool = False                  # written with `\approx`
    force: bool = False                   # written with `:=` (always bind, never solve)
    notes: list = field(default_factory=list)
    chain: list = field(default_factory=list)       # a = b = c: every segment, parsed
    chain_ops: list = field(default_factory=list)   # "=" | "approx" | "def" between them
    ics: list = field(default_factory=list)         # [(lhs, rhs)] initial conditions
    role: str = ""
    display: str = ""
    convert: Optional[str] = None                   # "\text{ in } \text{m}": the unit to express the result in
    domain: Optional[sympy.Basic] = None            # a set or condition restricting the solutions


# --- preprocessing -----------------------------------------------------------------

_UNICODE = {
    "×": r" \times ", "·": r" \cdot ", "−": "-", "–": "-", "÷": r" \div ", "π": r"\pi ",
    "∞": r"\infty ", "≤": r" \le ", "≥": r" \ge ", "≠": r" \ne ", "≈": r" \approx ",
    "√": r"\sqrt ", "°": r"^\circ ", "²": "^2", "³": "^3", "∈": r" \in ", "∉": r" \notin ",
    "∪": r" \cup ", "∩": r" \cap ", "∑": r"\sum ", "∏": r"\prod ", "∫": r"\int ",
    "∂": r"\partial ", "∇": r"\nabla ", "→": r" \to ", "⇒": r" \implies ", "⇔": r" \iff ",
    "∀": r"\forall ", "∃": r"\exists ", "¬": r"\neg ", "∧": r" \land ", "∨": r" \lor ",
    "α": r"\alpha ", "β": r"\beta ", "γ": r"\gamma ", "δ": r"\delta ", "ε": r"\epsilon ",
    "ζ": r"\zeta ", "η": r"\eta ", "θ": r"\theta ", "κ": r"\kappa ", "λ": r"\lambda ",
    "μ": r"\mu ", "ν": r"\nu ", "ξ": r"\xi ", "ρ": r"\rho ", "σ": r"\sigma ", "τ": r"\tau ",
    "φ": r"\phi ", "χ": r"\chi ", "ψ": r"\psi ", "ω": r"\omega ", "Γ": r"\Gamma ",
    "Δ": r"\Delta ", "Θ": r"\Theta ", "Λ": r"\Lambda ", "Π": r"\Pi ", "Σ": r"\Sigma ",
    "Φ": r"\Phi ", "Ψ": r"\Psi ", "Ω": r"\Omega ",
}
_UNICODE_RE = re.compile("|".join(map(re.escape, _UNICODE)))
# √16 is the root of 16 and √(x+1) the root of x+1 -- not \sqrt 1 · 6
_ROOT_UNI = re.compile(r"√\s*(?:(\d+(?:\.\d+)?|[A-Za-z]|\\[A-Za-z]+)|\(([^()]*)\))")

_WRAPPERS = [
    (re.compile(r"\\(?:begin|end)\{(?:equation|align|alignat|aligned|split|gather|gathered|multline|eqnarray|displaymath|math|flalign)\*?\}"), " "),
    (re.compile(r"\\\[|\\\]|\\\(|\\\)"), " "),
    (re.compile(r"\\(?:textstyle|scriptstyle|intertext\{[^}]*\}|allowbreak)"), " "),
    # \text{...} inside a subscript is just the subscript text: x_{\text{max}} -> x_{max}
    (re.compile(r"_\{\s*\\(?:text|mathrm|mathit|textrm)\{([^{}]*)\}\s*\}"), r"_{\1}"),
    (re.compile(r"(?<!\\)\$+"), " "),
    (re.compile(r"\\openlabquad(?![A-Za-z])"), " "),
    (re.compile(r"\\(?:quad|qquad)(?![A-Za-z])"), r" \\openlabquad "),
    (re.compile(r"\\(?:displaystyle|limits|nolimits|nonumber|notag)(?![A-Za-z])"), " "),
    (re.compile(r"\\right\."), " "),
    (re.compile(r"\\(?:label|tag)\{[^}]*\}"), " "),
    # delimiter sizes carry no meaning: \bigl( \Big[ \bigg\{ ... -> the bare delimiter
    (re.compile(r"\\[bB]igg?[lrm]?(?=\s*(?:[\(\)\[\]\\]|\\\{|\\\}|\\l|\\r))"), ""),
    (re.compile(r"\\middle\|"), r" \\mid "),
    (re.compile(r"\\[lr]?Vert(?![A-Za-z])"), r"\\|"),
    (re.compile(r"\\[lr]?vert(?![A-Za-z])"), "|"),
    (re.compile(r"~"), " "),
    (re.compile(r"\{,\}"), ""),
    # definitional equals bind, never solve
    (re.compile(r":=|\\coloneqq|\\triangleq|\\doteq|\\overset\{[^}]*\}\{=\}|\\stackrel\{[^}]*\}\{=\}"), r" \\openlabdef "),
    # `X \sim Bin(10, 0.5)`: a distribution name typed without \mathrm
    (re.compile(r"(\\sim\s*)([A-Za-z]{2,})(?=\s*(?:\(|\\left\())"), r"\1\\mathrm{\2}"),
    # "such that" / "subject to" introduce a constraint (see _conditioned)
    (re.compile(r"\\text(?:rm|it|bf)?\{\s*(?:s\.t\.|such that|subject to)\s*\}"), r" \\text{s.t.} "),
    # `x = -3 \text{ so } |x| = 3`: a word that moves to the next statement ends this one
    (re.compile(r"\\text(?:rm|it|bf)?\{\s*(?:so|So|then|Then|thus|Thus|hence|Hence)\s*[,:.]?\s*\}\s*:?"), r" \\therefore "),
    # narrative words that carry no math
    (re.compile(r"\\text(?:rm|it|bf)?\{\s*(?:let|Let|given|Given|we have|i\.e\.|e\.g\.|Solve|solve|Find|find|Evaluate|evaluate|Compute|compute|Calculate|calculate|Simplify|simplify|Check|check|Verify|verify)\s*[,:.]?\s*\}\s*:?"), " "),
]
_SEPARATOR_WORDS = r"on|over|s\.t\.|and|And|where|Where|with|With|,\s*and|for|For|if|If|when|When|whenever|otherwise|else|or|Or|for all|for every|for each|for any|at|evaluated at"
# a \text{...} that is not a name (spaces, punctuation) and not a separator word is an annotation
_ANNOTATION = re.compile(                      # never inside a subscript: `P_{\text{XIC.TO},t}` names a series
    r"(?<!_\{)\\text(?:rm|it|bf)?\{(?![A-Za-z][A-Za-z0-9_]*(?:\\_[A-Za-z0-9_]*)*\})"
    rf"(?!\s*,?\s*(?:{_SEPARATOR_WORDS})\s*[,:.]?\s*\}})([^{{}}]*)\}}")
_NOTE_RE = re.compile(r"\\openlabnote\{([^{}]*)\}(?:\s*\^\s*(?:\{[^{}]*\}|\S))?")
_CONTINUATION = re.compile(r"^\s*(?:\+|=|<|>|\\(?:cdot|times|le|ge|leq|geq|approx|ne|neq)(?![A-Za-z]))")
_COMMENT = re.compile(r"(?<!\\)%.*")
_ENV_BEGIN = re.compile(r"\\begin\{[A-Za-z*]+\}")
_ENV_END = re.compile(r"\\end\{[A-Za-z*]+\}")

# physical units after a number: 3\,\text{m/s}, 9.81\,\mathrm{m}/\mathrm{s}^2
_USEP = r"(?:[ \t]|\\,|\\;|\\:|\\!|\\ )*"          # a unit is on its number's line, never the next
_UATOM = r"\\(?:text|mathrm|textrm|mathit)\s*\{[^{}]*\}(?:\s*\^\s*(?:\{\s*-?\s*\d+\s*\}|-?\d))?"
_UNIT_GROUP = re.compile(rf"{_USEP}({_UATOM}(?:{_USEP}(?:/|\\cdot|\*)?{_USEP}{_UATOM})*)")
# a unit follows a number literal (3, 9.81, 1e-3, 10^{-3}) -- never a name, a fraction or a power
_NUMBER_BEFORE = re.compile(r"(?:(?<![\w\\^_.])(?<!_\{)\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\}\s*\\[,;]"
                            r"|10\s*\^\s*(?:\{\s*-?\s*\d+\s*\}|\d)|\\pi|\))[ \t]*\Z")    # \Z: `$` also matches before a newline
_UATOM_PARTS = re.compile(r"\\(?:text|mathrm|textrm|mathit)\s*\{([^{}]*)\}(?:\s*\^\s*(\{\s*-?\s*\d+\s*\}|-?\d))?|(/)")


def _unit_text(group: str) -> str:
    out = []
    for m in _UATOM_PARTS.finditer(group):
        if m.group(3):
            out.append("/")
            continue
        body = m.group(1).strip()
        power = m.group(2)
        out.append(body + (f"^{power.strip('{} ')}" if power else ""))
    return " ".join(out).replace(" / ", "/").replace("/ ", "/").replace(" /", "/")


def _mark_units(text: str) -> str:
    def sub(m):
        before = m.string[:m.start()]
        spaced_after_group = before.rstrip(" \t").endswith(("}", ")")) and re.match(r"[ \t]*\\[,;]", m.group(0))
        if not (_NUMBER_BEFORE.search(before) or spaced_after_group):   # \frac{3}{4}\,\text{m}
            return m.group(0)
        group = m.group(1)
        ut = _unit_text(group)
        if _units.parse_unit(ut) is not None:
            return rf" \openlabunit{{{ut}}} "
        first = re.match(_UATOM, group)       # only the first group member is a unit
        if first:
            ut1 = _unit_text(first.group())
            if _units.parse_unit(ut1) is not None:
                return rf" \openlabunit{{{ut1}}} " + group[first.end():]
        return m.group(0)
    return _UNIT_GROUP.sub(sub, text)


_THOUSANDS = re.compile(r"(?<![\w.,\\])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d|,\d)")


def _thousands(text: str) -> str:
    """``1,000,000`` -> ``1000000`` outside brackets; inside brackets only when a group has a
    leading zero (``\\{1,000\\}``), since ``f(1,234)`` may well be two arguments."""
    out, last = [], 0
    for m in _THOUSANDS.finditer(text):
        stack = []
        for i, c in enumerate(text[:m.start()]):
            if c in "{([":
                stack.append("\\{" if c == "{" and i and text[i - 1] == "\\" else c)
            elif c in "})]" and stack:
                stack.pop()
        if not stack or stack[-1] == "{" or ",0" in m.group():
            out.append(text[last:m.start()])
            out.append(m.group().replace(",", ""))
            last = m.end()
    out.append(text[last:])
    return "".join(out)


_FUNC_THIN = re.compile(r"(\\(?:" + FUNC_NAMES + r")(?![A-Za-z])(?:\s*\^\s*(?:\{[^{}]*\}|\S))?\s*"
                        r"(?:\\[A-Za-z]+|[A-Za-z0-9]+|\{[^{}]*\}))\s*\\[,;]"
                        r"(?!\s*(?:d\s*[A-Za-z]|d\s*\\[A-Za-z]+|\\mathrm\{d\}|\\text\{d\}|[-+=<>),\]}&|]|\\\\|$"
                        r"|\\(?:le|ge|leq|geq|ne|neq|lt|gt|cdot|times|div|approx|pm|mp|right|end|text|mathrm"
                        r"|quad|qquad|implies|iff|land|lor|to|in|mid)(?![A-Za-z])))\s*")


_MARKER_BODY = re.compile(r"(\\openlab(?:unit|note|in)\{[^{}]*\})")
_DEGC = [      # °C / °F in every spelling -> the units degC / degF
    (re.compile(r"\\(?:text|mathrm)\s*\{\s*°\s*([CF])\s*\}"), r"\\text{deg\1}"),
    (re.compile(r"(?:\^\s*\{?\s*\\circ\s*\}?|°)\s*\\(?:text|mathrm)\s*\{\s*([CF])\s*\}"), r"\\text{deg\1}"),
    (re.compile(r"(?<=[\d}])((?:\s|\\[,;: ])*)(?:\^\s*\{?\s*\\circ\s*\}?|°)\s*([CF])(?![A-Za-z])"), r"\1\\text{deg\2}"),
]


def _outside_markers(text: str, fn) -> str:
    """Apply ``fn`` to the text outside ``\\openlabunit{…}``/``\\openlabnote{…}``/``\\openlabin{…}``."""
    parts = _MARKER_BODY.split(text)
    return "".join(p if i % 2 else fn(p) for i, p in enumerate(parts))


def preprocess(cell: str, units: bool = True) -> str:
    """``units=False`` (a panel cell) reads no physical units: there ``2 \\mathrm{m}_t`` is a column."""
    text = "\n".join(_COMMENT.sub("", line) for line in cell.splitlines())
    for pat, sub_ in _DEGC:
        text = pat.sub(sub_, text)
    text = _ROOT_UNI.sub(lambda m: "\\sqrt{%s}" % (m.group(1) or m.group(2)), text)
    text = _outside_markers(text, lambda t: _UNICODE_RE.sub(lambda m: _UNICODE[m.group()], t))
    for pat, rep in _WRAPPERS:
        text = pat.sub(rep, text)
    # `\sin\theta \, t`: a thin space ends a function's bare argument
    text = _FUNC_THIN.sub(r"\1 \\cdot ", text)
    if not units:
        text = _ANNOTATION.sub(lambda m: " \\openlabnote{%s} " % re.sub(r"[=\\{}$]", " ", m.group(1)).strip(), text)
        return _thousands(text)
    # unit spellings: \mu\text{F}, 10\,\Omega, \text{k}\Omega, 25\,^\circ\text{C}, "\text{ in } \text{m}"
    text = re.sub(r"\\mu\s*\\(text|mathrm)\s*\{", r"\\\1{µ", text)
    text = re.sub(r"\\(text|mathrm)\s*\{\s*\\mu\s*", r"\\\1{µ", text)          # \mathrm{\mu F}
    text = re.sub(r"\\(?:text|mathrm)\s*\{\s*([kKM]?)\s*\}\s*\\Omega(?![A-Za-z])", lambda m: r"\text{%sΩ}" % m.group(1), text)
    text = re.sub(r"(?<=\d)((?:\s|\\[,;: ])*)\\Omega(?![A-Za-z])", r"\1\\text{Ω}", text)
    text = re.sub(r"\\text(?:rm)?\{\s*([A-Za-zµΩ°]+)\s+in\s+([^{}]+?)\s*\}", r"\\text{\1} \\text{ in } \\text{\2}", text)
    text = re.sub(r"\\text(?:rm)?\{\s+in\s+([A-Za-zµΩ°][^{}]*?)\s*\}", r"\\text{ in } \\text{\1}", text)
    text = re.sub(r"(?<![\d,;\s])(?<!\\[,;])\s*\\text(?:rm)?\{\s*in\s*\}(?=\s*\\(?:text|mathrm)\s*\{)", r" \\text{ in }", text)
    text = re.sub(r"\\text(?:rm)?\{\s+in\s+\}\s*(?:\{\s*\})?\s*\(?\s*\\(?:text|mathrm)\s*\{([^{}]*)\}"
                  r"(?:\s*\^\s*\{?\s*(-?\d+)\s*\}?)?\s*\)?",
                  lambda m: r" \openlabin{%s%s}" % (m.group(1).strip(), "^" + m.group(2) if m.group(2) else ""), text)
    text = re.sub(r"\\text(?:rm)?\{\s*in\s*\}\s*(?:\{\s*\})?\s*\^\s*\{?\s*\\circ\s*\}?|\\text(?:rm)?\{\s*in\s+degrees\s*\}",
                  r" \\openlabin{deg}", text)
    text = _mark_units(text)
    # 100\,\text{km} / 2\,\text{h}: a number with its unit is one quantity
    text = re.sub(r"(?<![\w\\^_.{])(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(\\openlabunit\{[^{}]*\})", r"{\1 \2}", text)
    text = _ANNOTATION.sub(lambda m: " \\openlabnote{%s} " % re.sub(r"[=\\{}$]", " ", m.group(1)).strip(), text)
    return _thousands(text)


def display_src(text: str) -> str:
    """The statement as TeX for the rendered input: markers back to what was typed."""
    text = re.sub(r"\\openlabnote\{([^{}]*)\}", r"\\text{\1}", text)
    text = re.sub(r"\\openlabunit\{deg([CF])\}", r"\\,{}^\\circ\\mathrm{\1}", text)
    text = re.sub(r"\\openlabunit\{([^{}]*)\}", r"\\,\\mathrm{\1}", text)
    text = re.sub(r"\\openlabin\{deg([CF])\}", r"\\ \\text{in}\\ {}^\\circ\\mathrm{\1}", text)
    text = re.sub(r"\\openlabin\{([^{}]*)\}", r"\\ \\text{in}\\ \\mathrm{\1}", text)
    text = text.replace(r"\openlabquad", r"\quad")
    return text.replace(r"\openlabdef", ":=")


def _depth0_split(text: str, pattern: re.Pattern) -> list[str]:
    """Split ``text`` on ``pattern`` matches outside braces/brackets/parens/environments."""
    parts, cur, depth, env = [], [], 0, 0
    i = 0
    while i < len(text):
        m = _ENV_BEGIN.match(text, i)
        if m:
            env += 1; cur.append(m.group()); i = m.end(); continue
        m = _ENV_END.match(text, i)
        if m:
            env = max(env - 1, 0); cur.append(m.group()); i = m.end(); continue
        if depth == 0 and env == 0:
            m = pattern.match(text, i)
            if m and m.end() > i:
                parts.append("".join(cur)); cur = []; i = m.end(); continue
        ch = text[i]
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth -= 1
        cur.append(ch); i += 1
    parts.append("".join(cur))
    return parts


# --- clause splitting ------------------------------------------------------------------

_AND_SEP = re.compile(r"\\text(?:rm)?\{\s*(?:and|And|,\s*and)\s*\}|(?<!\\);")
_WHERE_SEP = re.compile(r"\\text(?:rm)?\{\s*,?\s*(?:where|Where|with|With)\s*\}")
_OR_SEP = re.compile(r"\\text(?:rm)?\{\s*(?:or|Or)\s*\}")
_IMPLIES_SEP = re.compile(r"\\(?:implies|Rightarrow|Longrightarrow|iff|Leftrightarrow)(?![A-Za-z])")
_THEREFORE_SEP = re.compile(r"\\(?:therefore|because)(?![A-Za-z])")
_FOR_SEP = re.compile(r"\\text(?:rm)?\{\s*(?:for|For|if|If|when|When|whenever|on|over)\s*\}")
_FORALL_SEP = re.compile(r"\\text(?:rm)?\{\s*(?:for all|for every|for each|for any)\s*\}")
_AT_SEP = re.compile(r"\\text(?:rm)?\{\s*(?:at|evaluated at)\s*\}")
_ST_SEP = re.compile(r"\\text\{\s*s\.t\.\s*\}")
_OPTIM_HEAD = re.compile(r"^\s*\\(max|min|sup|inf|argmax|argmin)(?![A-Za-z])\s*(?:_\s*(?:\{[^{}]*\}|[A-Za-z]))?")
_OTHERWISE_SEP = re.compile(r"\\text(?:rm)?\{\s*,?\s*(?:otherwise|else)\s*,?\s*\}")
_COMMA = re.compile(r",")
_ODE_HINT = re.compile(r"'|\\(?:dot|ddot|dddot)(?![A-Za-z])|\\frac\s*\{\s*(?:d|\\mathrm\{d\}|\\partial)")
_IC_RE = re.compile(
    r"^\s*(?:\\(?:dot|ddot|dddot)\s*\{?\s*\\?[A-Za-z]+\s*\}?|\\?[A-Za-z]+(?:_\{?\w+\}?)?)"
    r"\s*(?:'+|\^\s*\{\s*\(\s*\d+\s*\)\s*\})?\s*(?:\(|\\left\()\s*(?P<pt>[-+]?\s*[\d.]+|\\?[A-Za-z]+(?:_\{?\w+\}?)?"
    r"|\\pi\s*(?:/\s*\d+)?|\\frac\s*\{[^{}]*\}\s*\{[^{}]*\}|\d*\s*\\pi\s*/\s*\d+)\s*(?:\)|\\right\))\s*=")


def _is_ic(text: str) -> bool:
    """`y(0) = 1`, `y'(L) = 0`, `y(\\pi/4) = 1`; not `y(x) = x^2` (a point letter used on the right)."""
    m = _IC_RE.match(text)
    if not m:
        return False
    pt = m.group("pt").strip()
    if re.fullmatch(r"\\?[A-Za-z]+(?:_\{?\w+\}?)?", pt) and not pt.startswith("\\pi"):
        rhs = text[m.end():]
        if pt in ("x", "t") or re.search(rf"(?<![A-Za-z\\]){re.escape(pt)}(?![A-Za-z])", rhs):
            return False
    return True
_DEFINES = re.compile(r"^\s*\\?[A-Za-z]+(?:_\{?[^{}=]*\}?)?\s*(?:\([^()]*\))?\s*$")
_REC_IC = re.compile(r"^\s*([A-Za-z]+)_\{?\d+\}?\s*=")
_REC_DEF = re.compile(r"^\s*([A-Za-z]+)_(?:\{[^{}]*[A-Za-z][^{}]*\}|[A-Za-z])\s*=")
_DECL_RE = re.compile(r"^\s*(?:\\?[A-Za-z]+\s*,\s*)*\\?[A-Za-z]+\s*\\in\s*\\mathbb\s*\{?\s*[RNZQC]\s*\}?\s*$")


def _has_equals(s: str) -> bool:
    try:
        return split_assignment(s)[0] is not None
    except ParseError:
        return True        # a chain `a = b = c` has equals


_SPACING = re.compile(r"^(?:\s|\\[,;:! ])+|(?:\s|\\[,;:! ]|(?<!\\)[.,])+$")


def _trim(s: str) -> str:
    """Strip whitespace and spacing commands (`\\;`, `\\,`) from the ends of a clause."""
    return _SPACING.sub("", s)


_RESTRICT = re.compile(r"<|>|\\(?:le|ge|leq|geq|in)(?![A-Za-z])")


def _list_clauses(text: str) -> list[str]:
    """`a = 1 \\text{and} b = 2`, `a = 1, b = 2` -> one clause per equation; an equation
    followed by conditions (`\\sin x = \\frac12, 0 \\le x < 2\\pi`) keeps them as its domain."""
    out = []
    for p in _depth0_split(text, _AND_SEP):
        pieces = _depth0_split(p, _COMMA)
        if out and not _has_equals(p) and _RESTRICT.search(p) and (_ST_SEP.search(out[-1]) or _FOR_SEP.search(out[-1])):
            out[-1] = out[-1] + r" \land " + _trim(p)
            continue
        if len(pieces) > 1 and (_AT_SEP.search(pieces[0]) or _FOR_SEP.search(pieces[0])) and all(
                re.match(r"\s*\\?[A-Za-z]+(?:_\{?\w+\}?)?\s*=(?!=)", q) for q in pieces[1:]):
            out.append(", ".join(_trim(q) for q in pieces))
            continue
        if len(pieces) > 1 and all(_has_equals(q) for q in pieces):
            out.extend(pieces)
        elif (len(pieces) > 1 and _has_equals(pieces[0]) and _has_equals(pieces[1])
              and all(_has_equals(q) or _RESTRICT.search(q) for q in pieces)):
            out.extend(pieces)                             # `a = 2, b = 3, a < b`: three statements
        elif (len(pieces) > 1 and _has_equals(pieces[0])
              and all(not _has_equals(q) and _RESTRICT.search(q) for q in pieces[1:])):
            out.append(pieces[0] + r" \text{for} " + r" \land ".join(_trim(q) for q in pieces[1:]))
        else:
            out.append(p)
    return [t for t in (_trim(x) for x in out) if t]


def _or_clause(text: str) -> str:
    """`x = 2 \\text{ or } x = -2` -> `x \\in \\{2, -2\\}`; otherwise `\\lor`."""
    parts = _depth0_split(text, _OR_SEP)
    if len(parts) == 1:
        return text
    sides = []
    for p in parts:
        try:
            lhs, rhs, _, _ = split_assignment(p)
        except ParseError:
            lhs, rhs = None, p
        sides.append((lhs.strip() if lhs else None, rhs))
    names = {l for l, _ in sides}
    if len(names) == 1 and None not in names and re.fullmatch(r"\\?[A-Za-z]+(?:_\{?\w+\}?)?", next(iter(names))):
        return rf"{next(iter(names))} \in \{{{', '.join(r.strip() for _, r in sides)}\}}"
    return r" \lor ".join(f"({p.strip()})" for p in parts)


def _lhs_text(main: str) -> Optional[str]:
    try:
        return split_assignment(main)[0]
    except ParseError:
        return None


def lhs_is_equation(main: str) -> bool:
    lhs = _lhs_text(main)
    return lhs is not None and not _DEFINES.match(lhs)


def _evaluate_there(c: Clause, main: str, cond: str) -> Optional[list]:
    """`y = x^2 \text{ for } x = 3` -> y = x^2 at x = 3; `v(t) = 3t \text{ at } t = 2` -> define v,
    then v(2). ``None`` when the condition sets the name being defined (then it is not a point)."""
    try:
        lhs, rhs, _, force = split_assignment(main)
    except ParseError:
        return None
    if lhs is None:
        return None
    spec = ", ".join(_list_clauses(cond))
    names = {x.strip() for x in re.findall(r"(\\?[A-Za-z]+(?:_\{?\w+\}?)?)\s*=", spec)}
    call = re.fullmatch(r"\s*(\\?[A-Za-z]+(?:_\{?\w+\}?)?)\s*\(([^()]*)\)\s*", lhs)
    if lhs.strip() in names or (call and call.group(1) in names):
        return None
    shown = c.display or c.text
    if call:                                            # a definition, then its value at the point
        return [replace(c, text=main, display=shown),
                replace(c, text=rf"\left. {lhs.strip()} \right|_{{{spec}}}", display=shown)]
    op = r"\openlabdef" if force else "="
    return [replace(c, text=rf"{lhs} {op} \left. {rhs} \right|_{{{spec}}}", display=shown)]


def _conditioned(c: Clause) -> list[Clause]:
    """`f(x) = x^2 \\text{for } x > 0` -> a one-row cases; `... \\text{for } x \\in \\mathbb{R}`
    -> a declaration first; `... \\text{for all } y` -> a quantifier."""
    parts = _depth0_split(c.text, _ST_SEP)
    if len(parts) == 2:                                   # `… \text{s.t.} x \ge 0`: a constraint
        main, cond = parts[0].strip(), parts[1].strip()
        m = _OPTIM_HEAD.match(main)
        if m and not _has_equals(main):                   # \max_x f \text{ s.t. } x \ge 0 -> \max_{x \ge 0} f
            body = main[m.end():].strip()
            return [replace(c, text=rf"\{m.group(1)}_{{{cond}}} \left( {body} \right)", display=c.display or c.text)]
        lhs_text = _lhs_text(main)
        if lhs_text is not None and not _DEFINES.match(lhs_text):
            return [replace(c, text=main, domain=cond, display=c.display or c.text)]
        return [replace(c, text=rf"{main} \land {cond}", display=c.display or c.text)]
    parts = _depth0_split(c.text, _AT_SEP)
    if len(parts) == 2 and _has_equals(parts[1]) and not _has_equals(parts[0]):
        spec = ", ".join(_list_clauses(parts[1]))            # `f'(x) \text{ at } x = 2`
        return [replace(c, text=rf"\left. {parts[0].strip()} \right|_{{{spec}}}", display=c.display or c.text)]
    if len(parts) == 2 and _has_equals(parts[1]) and _has_equals(parts[0]):
        at = _evaluate_there(c, parts[0].strip(), parts[1])
        if at is not None:
            return at
    parts = _depth0_split(c.text, _FORALL_SEP)
    if len(parts) == 2:
        return [replace(c, text=rf"\forall {parts[1].strip()}, {parts[0].strip()}", display=c.display or c.text)]
    parts = _depth0_split(c.text, _FOR_SEP)
    if len(parts) != 2:
        return [c]
    main, cond = parts[0].strip(), parts[1].strip()
    m = _OPTIM_HEAD.match(main)
    if m and not _has_equals(main) and not _has_equals(cond):
        return [replace(c, text=rf"\{m.group(1)}_{{{cond}}} \left( {main[m.end():].strip()} \right)",
                        display=c.display or c.text)]
    if _DECL_RE.match(cond) and lhs_is_equation(main):
        return [replace(c, text=main, domain=cond, display=c.display or c.text)]
    if _DECL_RE.match(cond):
        return [Clause(c.lineno, cond, role="declare"), replace(c, text=main, display=c.display or c.text)]
    if _has_equals(cond) and not _has_equals(main):      # `v(t) \text{ for } t = 2`: evaluate there
        spec = ", ".join(_list_clauses(cond))
        return [replace(c, text=rf"\left. {main} \right|_{{{spec}}}", display=c.display or c.text)]
    if _has_equals(cond) and _has_equals(main):          # `y = x^2 \text{ for } x = 3`
        at = _evaluate_there(c, main, cond)
        if at is not None:
            return at
    try:
        lhs_text = split_assignment(main)[0]
    except ParseError:
        lhs_text = None
    if lhs_text is not None and not _DEFINES.match(lhs_text):   # an equation: restrict its solutions
        return [replace(c, text=main, domain=cond, display=c.display or c.text)]
    if lhs_text is not None and not _has_equals(cond) and _RESTRICT.search(cond) \
            and not _depth0_split(cond, _OTHERWISE_SEP)[1:]:
        # f(x) = x^2 \text{ for } x > 0: a piecewise definition -- or, if f is already defined,
        # an equation restricted to the range (decided once the namespace is known)
        return [replace(c, text=main, domain=cond, display=c.display or c.text)]
    other = _depth0_split(cond, _OTHERWISE_SEP)
    tail = ""
    if len(other) == 2:
        cond, tail = other[0].strip(), rf" \\ {other[1].strip()}"
    try:
        lhs, rhs, _, force = split_assignment(main)
    except ParseError:
        return [c]
    op = r"\openlabdef" if force else "="
    if lhs is not None:
        text = rf"{lhs} {op} \begin{{cases}} {rhs} & {cond}{tail} \end{{cases}}"
    else:
        text = rf"\begin{{cases}} {main} & {cond}{tail} \end{{cases}}"
    return [replace(c, text=text, display=c.display or c.text)]


def _segment(seg: str) -> list[Clause]:
    parts = _depth0_split(seg, _WHERE_SEP)
    main, where_parts = parts[0], parts[1:]
    bindings = [b for wp in where_parts for b in _list_clauses(wp)]
    mains = _list_clauses(main)
    # `c_n = c_{n-1} + c_{n-2}, c_0 = 0, c_1 = 1`: the initial values run before the recurrence
    rec_bases = {m.group(1) for x in mains if (m := _REC_DEF.match(x))}
    if rec_bases:
        first = [x for x in mains if (m := _REC_IC.match(x)) and m.group(1) in rec_bases]
        mains = first + [x for x in mains if x not in first]
    # initial conditions ride along with the differential equation they follow
    ode = next((m for m in mains if _ODE_HINT.search(m) and _has_equals(m)), None)
    ics: list[str] = []
    if ode is not None:
        ics = [x for x in mains + bindings if x is not ode and _is_ic(x)]
        mains = [m for m in mains if m not in ics]
        bindings = [b for b in bindings if b not in ics]
    # `where` bindings run first; a `where` condition (x > 0) holds for this statement only
    out = [Clause(0, _or_clause(b), role="local") for b in bindings]     # `where` holds for this line
    for m in mains:
        chain = _depth0_split(m, _IMPLIES_SEP)
        shown = seg.strip() if (m is ode and ics) else ""
        if len(chain) > 1 and all(_has_equals(x) or _OR_SEP.search(x) for x in chain):
            pieces = [Clause(0, _or_clause(chain[0].strip()), display=shown, ics=ics if m is ode else [])]
            pieces += [Clause(0, _or_clause(x.strip()), role="derived") for x in chain[1:]]
        else:
            pieces = [Clause(0, _or_clause(m), display=shown, ics=ics if m is ode else [])]
        for pc in pieces:
            out.extend(_conditioned(pc))
    return out


_SENTENCE_END = re.compile(r"(?i)(?:\.\s*\w|\b(?:then|so|thus|hence|therefore)\s*[.,:]?\s*$|^\s*be\b)")
_QUAD = re.compile(r"\\openlabquad(?![A-Za-z])")
_CLAUSE_WORD = re.compile(r"^\s*\\text(?:rm)?\{\s*,?\s*(?:where|Where|with|With|for|For|if|If|when|When|and|And|"
                          r"or|Or|on|over|at|s\.t\.|otherwise|else)\b")
_MID_PERIOD = re.compile(r"(?<!\\left)(?<!\\right)(?<=[\w)}\]])\.\s+(?=[\\A-Za-z(0-9])")
_NOTE_ONLY = re.compile(r"^(?:\s|[.,:;?!]|\\openlabnote\{[^{}]*\}|\\[,;:! ])*$")
_NAME_ONLY = re.compile(r"^\s*\\?[A-Za-z]+(?:_\{?\w+\}?)?\s*$")


def _sentences(line: str) -> list[str]:
    """`x + 1 = 5. y = 2`, `\\text{Let } x \\text{ be the number. Then } 3x - 7 = 29`: a sentence
    end (a period between statements, or a note that ends a sentence) separates statements."""
    line = re.sub(r"(\\openlabnote\{[^{}]*\})\s*:", r"\1", line)          # \text{Solve}: x^2 = 16
    line = re.sub(r"\?\s*$", "", line)                                      # 2x + 1 = 7?
    cuts = []
    for m in re.finditer(r"\\openlabnote\{([^{}]*)\}", line):
        if _SENTENCE_END.search(m.group(1)):
            cuts.append(m.end())
    pieces, last = [], 0
    for c in cuts:
        pieces.append(line[last:c])
        last = c
    pieces.append(line[last:])
    out = []
    for p in pieces:
        for q in _depth0_split(p, _MID_PERIOD):
            parts = _depth0_split(q, _QUAD)
            if len(parts) > 1 and all(_has_equals(x) for x in parts) and not any(
                    _CLAUSE_WORD.match(x) for x in parts[1:]):             # x = 5 \quad y = 6
                out.extend(parts)
            else:
                out.append(_QUAD.sub(" ", q))
    kept = []
    for p in out:
        bare = _NOTE_RE.sub(" ", p)
        if _NOTE_ONLY.match(p):
            if _NOTE_RE.search(p):
                kept.append("\\openlabprose " + p)    # a line of prose is a note: shown, never run
            continue
        if _NAME_ONLY.match(bare) and "openlabnote" in p and len(out) > 1:
            continue                          # "Let x be the number": introduces x, nothing to run
        kept.append(p)
    return kept


_CLAIM_START = re.compile(r"^\s*\\(?:forall|exists)(?![A-Za-z])")


def _split_clauses(line: str) -> list[Clause]:
    if _CLAIM_START.match(line):              # \forall n \ge 1: … [\text{by induction}]: one claim
        return [Clause(0, line.strip(), role="claim")]
    out = []
    for sentence in _sentences(line):
        if sentence.startswith("\\openlabprose "):
            out.append(Clause(0, sentence[len("\\openlabprose "):].strip(), role="note"))
            continue
        for seg in _depth0_split(sentence, _THEREFORE_SEP):
            if seg.strip():
                out.extend(_segment(seg))
    return [c for c in out if c.text.strip()]


def split_statements(text: str) -> list[tuple[int, str]]:
    """``[(lineno, statement)]``. Separators: newline (outside braces/environments) and
    ``\\\\`` (outside environments). ``&`` outside an environment is an align marker and
    is dropped; inside a matrix it separates entries and is kept."""
    out: list[tuple[int, str]] = []
    cur: list[str] = []
    brace = env = 0
    line = start = 1
    i, n = 0, len(text)

    def flush():
        nonlocal cur, start
        s = "".join(cur).strip()
        if s:
            out.append((start, s))
        cur = []

    while i < n:
        m = _ENV_BEGIN.match(text, i)
        if m:
            env += 1
            cur.append(m.group())
            i = m.end()
            continue
        m = _ENV_END.match(text, i)
        if m:
            env = max(env - 1, 0)
            cur.append(m.group())
            i = m.end()
            continue
        if text.startswith("\\\\", i):
            if env == 0 and brace == 0:
                flush()
                start = line
            else:
                cur.append("\\\\")
            i += 2
            continue
        ch = text[i]
        if ch == "\n":
            line += 1
            if env == 0 and brace == 0:
                flush()
                start = line
            else:
                cur.append(" ")
        elif ch == "&" and env == 0:
            pass
        else:
            if ch == "{" and not text.startswith("\\", i - 1):
                brace += 1
            elif ch == "}" and not text.startswith("\\", i - 1):
                brace -= 1
            cur.append(ch)
        i += 1
    flush()
    # a line that starts with an operator continues the previous one (derivations)
    merged: list[tuple[int, str]] = []
    for ln, st in out:
        if merged and _CONTINUATION.match(st):
            pl, ps = merged[-1]
            merged[-1] = (pl, ps + " " + st)
        else:
            merged.append((ln, st))
    return merged


_OPERAND_CMDS = set(GREEK_NAMES.split("|")) | {"pi", "infty", "rvert", "rfloor", "rceil"}


_COND_HEAD = re.compile(r"(?:(?<![A-Za-z\\])(?:P|E|Pr)|\\Pr|\\mathbb\s*\{\s*[PE]\s*\}|\\operatorname\s*\{\s*(?:P|E|Pr|Var)\s*\})"
                        r"\s*(?:\\left)?\s*([(\[])")


def _conditional_bars(s: str) -> str:
    """``P(A|B)``, ``E[X | Y]``: the one bar inside a probability or expectation is ``\\mid``."""
    out, pos = [], 0
    for m in _COND_HEAD.finditer(s):
        if m.start() < pos:
            continue
        close = {"(": ")", "[": "]"}[m.group(1)]
        depth, bars, j = 0, [], m.end()
        while j < len(s):
            ch = s[j]
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                if depth == 0:
                    break
                depth -= 1
            elif ch == "|" and depth == 0 and not re.search(r"\\(?:left|right|[bB]igg?|middle)$", s[:j]):
                bars.append(j)
            j += 1
        if j < len(s) and s[j] == close and len(bars) == 1:
            out.append(s[pos:bars[0]] + r" \mid ")
            pos = bars[0] + 1
    out.append(s[pos:])
    return "".join(out)


def normalize_bars(s: str) -> str:
    r"""Rewrite bare ``|...|`` as ``\lvert ... \rvert`` so the grammar never has to guess
    whether a bar opens or closes. A bar closes when it follows an operand and a bar
    is open; otherwise it opens."""
    if "|" not in s:
        return s
    s = _conditional_bars(s)
    if "|" not in s:
        return s
    out: list[str] = []
    depth = 0
    prev = "op"          # 'op' | 'operand'
    i, n = 0, len(s)
    while i < n:
        m = re.match(r"\\(?:left|right|[bB]igg?|middle)?\|", s[i:])
        if m:
            out.append(m.group())
            prev = "operand" if s.startswith("\\right|", i) else "op"
            i += m.end()
            continue
        ch = s[i]
        if ch == "|":
            if depth > 0 and prev == "operand":
                out.append(r" \rvert ")
                depth -= 1
                prev = "operand"
            else:
                out.append(r" \lvert ")
                depth += 1
                prev = "op"
            i += 1
            continue
        if ch == "\\":
            m = re.match(r"\\([A-Za-z]+)", s[i:])
            if m:
                out.append(m.group())
                word = m.group(1)
                prev = "operand" if word in _OPERAND_CMDS else "op"
                i += m.end()
                continue
        out.append(ch)
        if ch.isalnum() or ch in ")}]!":
            prev = "operand"
        elif not ch.isspace():
            prev = "op"
        i += 1
    return "".join(out)


# --- the top-level `=` -------------------------------------------------------------------

def split_chain(src: str) -> tuple[list[str], list[str]]:
    """Segments and the operators between them at depth 0: ``=``, ``approx`` (``\\approx``)
    and ``def`` (``:=``)."""
    depth = env = 0
    cuts: list[tuple[int, int, str]] = []
    i, n = 0, len(src)
    while i < n:
        m = _ENV_BEGIN.match(src, i)
        if m:
            env += 1; i = m.end(); continue
        m = _ENV_END.match(src, i)
        if m:
            env = max(env - 1, 0); i = m.end(); continue
        ch = src[i]
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth -= 1
        elif depth == 0 and env == 0:
            if ch == "=":
                prev, nxt = src[i - 1:i], src[i + 1:i + 2]
                if nxt != "=" and prev not in ("=", "<", ">", "!"):
                    cuts.append((i, i + 1, "="))
            elif src.startswith(r"\approx", i):
                cuts.append((i, i + len(r"\approx"), "approx"))
                i += len(r"\approx") - 1
            elif src.startswith(r"\openlabdef", i):
                cuts.append((i, i + len(r"\openlabdef"), "def"))
                i += len(r"\openlabdef") - 1
        i += 1
    segs, ops, last = [], [], 0
    for s, e, op in cuts:
        segs.append(src[last:s])
        ops.append(op)
        last = e
    segs.append(src[last:])
    return segs, ops


def split_assignment(src: str) -> tuple[Optional[str], str, bool, bool]:
    """``(lhs, rhs, approx, force)`` at the single top-level ``=`` / ``\\approx`` / ``:=``."""
    segs, ops = split_chain(src)
    if not ops:
        return None, src, False, False
    if len(ops) > 1:
        raise ParseError("several `=` in one statement", src=src,
                         hint="a chain `a = b = c` is checked link by link; to bind and check, "
                              "put each statement on its own line.")
    return segs[0], segs[1], ops[0] == "approx", ops[0] == "def"


# --- parsing -------------------------------------------------------------------------

KNOWN_COMMANDS = ({"land", "lor", "wedge", "vee", "implies", "iff", "neg", "lnot", "Rightarrow",
                   "Leftrightarrow", "oplus"}
                  | set(re.findall(r"\\\\([A-Za-z]+)", GRAMMAR))
                  | set(GREEK_NAMES.split("|")) | set(FUNC_NAMES.split("|"))
                  | set(DECO_NAMES.split("|")) | {"left", "right", "begin", "end", "frac",
                                                   "cdot", "times", "to", "choose", "mid"})


def _describe(exc: UnexpectedInput) -> str:
    if isinstance(exc, UnexpectedEOF):
        return "the expression ends too early (something is missing at the end)"
    if isinstance(exc, UnexpectedCharacters):
        return f"could not read `{exc.char}` at position {exc.column}"
    return "unexpected input"


_CASES_WORDS = re.compile(r"\\text(?:rm)?\{\s*(?:if|for|when|otherwise|else|elsewhere)\s*[,:]?\s*\}")


_CASES_PARITY = re.compile(r"&([^&]*?)\\(?:text(?:rm)?|openlabnote)\{\s*(?:is\s+)?(even|odd|prime|composite)\s*\}")


def _parity_cond(m):
    """`n \\text{ even}` in a cases condition -> `n \\bmod 2 = 0` (odd: `= 1`; prime: isprime(n) = 1)."""
    x, word = m.group(1).strip(), m.group(2)
    if not x:
        return m.group(0)
    if word in ("even", "odd"):
        return rf"& ({x}) \bmod 2 = {0 if word == 'even' else 1} "
    return rf"& \operatorname{{isprime}}({x}) = {1 if word == 'prime' else 0} "


def _preclean_cases(src: str) -> str:
    """Inside ``\\begin{cases}``: drop the prose (`\\text{if }`, `\\text{otherwise}`) and the
    alignment `&` of an `&=` so a row is `value & condition` or an equation."""
    if r"\begin{cases}" not in src:
        return src

    def fix(m):
        body = m.group(1)
        body = re.sub(r"&\s*\\text(?:rm)?\{\s*otherwise\s*\}", "", body)
        body = _CASES_WORDS.sub(" ", body)
        body = _CASES_PARITY.sub(_parity_cond, body)
        body = re.sub(r"&\s*(=|\\le|\\ge|<|>|\\ne)", r"\1", body)
        return r"\begin{cases}" + body + r"\end{cases}"
    return re.sub(r"\\begin\{cases\}(.*?)\\end\{cases\}", fix, src, flags=re.S)


def _balanced(src: str, i: int) -> int:
    """Index just past the brace group that opens at src[i] == "{" (−1 if unbalanced)."""
    depth = 0
    for j in range(i, len(src)):
        if src[j] == "{" and (j == 0 or src[j - 1] != "\\"):
            depth += 1
        elif src[j] == "}" and (j == 0 or src[j - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return j + 1
    return -1


_PRIME_WORDS = re.compile(r"\\(?:openlabnote|text(?:rm)?)\{\s*(?:is\s+|are\s+)?(?:a\s+)?primes?\s*\}")


def _sum_subscripts(src: str) -> str:
    """In `\\sum_{…}` / `\\prod_{…}` / `\\max_{…}` subscripts: `\\substack{a \\\\ b}` -> `a, b`, and
    `p \\text{ prime}` -> `p \\in \\mathbb{P}` (so the word is not dropped as prose)."""
    out, i = [], 0
    for m in re.finditer(r"\\(?:sum|prod)\s*_\s*(?=\{)", src):
        if m.start() < i:
            continue
        j = _balanced(src, m.end())
        if j < 0:
            break
        sub = src[m.end() + 1:j - 1]
        st = re.match(r"\s*\\substack\s*\{", sub)
        if st:
            k = _balanced(sub, st.end() - 1)
            if k > 0:
                sub = sub[st.end():k - 1] + sub[k:]
            sub = re.sub(r"\\\\", ",", sub)
        sub = _PRIME_WORDS.sub(r" \\in \\mathbb{P} ", sub)
        out.append(src[i:m.end()] + "{" + sub + "}")
        i = j
    return "".join(out) + src[i:]


_IN_PRIMES = re.compile(r"^\s*(.+?)\s*\\(not)?in\s*\\mathbb\s*\{?\s*P\s*\}?\s*$", re.S)


_GLUE_BEFORE = re.compile(r"(?:[\w)\]}|!']|\\rvert)\s*$")
_GLUE_AFTER = re.compile(r"\s*(?:[\w(|]|\\\{|\\(?!(?:cdot|times|div|pm|mp|le|ge|leq|geq|lt|gt|ne|neq|approx|in|notin|"
                         r"sim|to|mid|implies|iff|land|lor|wedge|vee|right|end|openlab[a-z]+)(?![A-Za-z])))")
_CHAIN_HEAD = re.compile(r"<|>|\\(?:le|ge|leq|geq|lt|gt|ne|neq|implies|iff|land|lor|wedge|vee|Rightarrow|"
                         r"Leftrightarrow|Longrightarrow)(?![A-Za-z])")


_DFRAC = re.compile(r"\\[dt]?frac\s*\{\s*((?:\\mathrm\{d\}|\\text\{d\}|d)\s*(?:[a-zA-Z]|\\[a-zA-Z]+))\s*\}\s*(?=\{)")


def _int_dfrac(src: str) -> str:
    """`\\int \\frac{dt}{\\ln^2 t}` -> `\\int \\frac{1}{\\ln^2 t} \\, dt`: the numerator is exactly the differential."""
    if "\\int" not in src:
        return src
    out, i = [], 0
    for m in _DFRAC.finditer(src):
        if m.start() < i or "\\int" not in src[:m.start()]:
            continue
        j = _balanced(src, m.end())
        if j < 0:
            break
        out.append(src[i:m.start()] + r"\frac{1}" + src[m.end():j] + r" \, " + m.group(1) + " ")
        i = j
    return "".join(out) + src[i:]


def _count_sets(src: str) -> str:
    """`\\#\\{n : …\\}` (the number of elements) -> `\\left| \\{n : …\\} \\right|`."""
    out, i = [], 0
    while True:
        m = re.compile(r"\\#\s*(?=\\\{)").search(src, i)
        if not m:
            break
        j, depth = m.end(), 0
        k = j
        while k < len(src):
            if src.startswith("\\{", k):
                depth += 1
                k += 2
                continue
            if src.startswith("\\}", k):
                depth -= 1
                k += 2
                if depth == 0:
                    break
                continue
            k += 1
        if depth != 0:
            break
        out.append(src[i:m.start()] + r"\left| " + src[j:k] + r" \right|")
        i = k
    return "".join(out) + src[i:]


#: Annotations that ask for a hypothesis test. The engine does not run one, and the equality it *does* run
#: is the dangerous part: `\bar{a} = \bar{b}` compares two floats exactly and is `false` for any two samples
#: that are not identical, which reads like a finding. Say where the test actually lives.
_NAMED_TEST = re.compile(r"(?i)\b(?:t[\s-]?test|welch|permutation|wilcoxon|mann[\s-]?whitney|z[\s-]?test|anova|"
                         r"chi[\s-]?squared?|significan(?:t|ce))\b")
_NAMED_TEST_NOTE = (
    "a test of means is not an annotation on an equality: the check above compares two numbers exactly and "
    "is false for any two samples that differ at all. Test it in a study: study_compare (a paired difference "
    "with its interval and p) or study_test (a hypothesis against a baseline, spending the study's alpha "
    "budget)")


def parse_expr(src: str, dialect: Optional[str] = None) -> tuple[sympy.Basic, list[str]]:
    """LaTeX -> ``(sympy expression, notes)``. Raises :class:`ParseError` with a hint.
    ``dialect`` names a registered reading (``"panel"``); ``None`` is the core."""
    src = _int_dfrac(_count_sets(_sum_subscripts(_preclean_cases(preprocess(src, units=dialect != "panel").strip()))))
    m_p = _IN_PRIMES.match(src)
    if m_p and not re.search(r"\\(?:forall|exists)", src):          # 7 \in \mathbb{P}: is 7 prime?
        src = rf"\operatorname{{isprime}}\left({m_p.group(1)}\right) {'< 1' if m_p.group(2) else '> 0'}"
    annotations = [m.group(1).strip() for m in _NOTE_RE.finditer(src) if m.group(1).strip()]
    notes = [f"ignored the text “{a}” (annotations are not executed)" for a in annotations]
    if any(_NAMED_TEST.search(a) for a in annotations):
        notes.append(_NAMED_TEST_NOTE)
    for m in _NOTE_RE.finditer(src):          # dropped prose between two terms would multiply them
        if _GLUE_BEFORE.search(src[:m.start()]) and _GLUE_AFTER.match(src[m.end():]):
            raise ParseError(f"the text “{m.group(1).strip()}” sits between two terms, which would multiply them",
                             src=src, hint="end the statement before the prose (a new line), or write the "
                                           "operator the words stand for")
    src = _NOTE_RE.sub(" ", src).strip()
    src = re.sub(r"(?<!\\)[.,]\s*$", "", src).strip()        # a sentence's period, after the prose went
    if re.search(r"\\pi\s*\(\s*\d+\s*\)", src):
        notes.append(r"\pi(…) read as π times …; the prime-counting function is \operatorname{primepi}(…)")
    if re.search(r"\d\s*\\%\s*[\d(]", src):
        notes.append(r"\% read as percent; for a remainder write a \bmod b")
    if re.search(r"(?<![\\\w{])(?:\d+|[A-Za-z])\s*/\s*(?:\d+|[A-Za-z])\s*(?=[A-Za-z(\\])", src):
        notes.append(r"a/b c is read as (a/b)·c; write \frac{a}{bc} for a/(bc)")
    if not src:
        raise ParseError("empty expression", src=src)
    pre = precheck(src)
    if pre:
        raise ParseError(pre[0], hint=pre[1], src=src)
    old = sys.getrecursionlimit()
    d = get_dialect(dialect)
    tr = (d.transformer if d is not None and d.transformer is not None else ToSympy)()
    try:
        sys.setrecursionlimit(max(old, 8000))
        try:
            tree = build_parser("resolve", dialect).parse(normalize_bars(src))
        except UnexpectedInput as exc:
            col = None if isinstance(exc, UnexpectedEOF) else getattr(exc, "column", None)
            hint = teaching_hint(src)
            if not hint and col and src[col - 1:col] == "\\":
                m = re.match(r"\\([A-Za-z]+)", src[col - 1:])
                if m and m.group(1) not in KNOWN_COMMANDS:
                    hint = (f"`\\{m.group(1)}` is not a command openlab knows; if it is decorative, "
                            "remove it, otherwise spell the operation out.")
                elif m:
                    hint = (f"openlab knows `\\{m.group(1)}` but could not read what surrounds it; "
                            "check the operands on either side.")
            raise ParseError(f"could not read `{src}`: {_describe(exc)}", hint=hint,
                             src=src, column=col) from None
        try:
            expr = tr.transform(tree)
        except VisitError as exc:
            orig = exc.orig_exc
            if isinstance(orig, OpenlabError):
                orig.src = orig.src or src
                raise orig from None
            if isinstance(orig, RecursionError):
                raise orig
            raise ParseError(f"read `{src}` but could not build it ({type(orig).__name__}: {orig})",
                             src=src) from None
    except RecursionError:
        raise ParseError("the expression is too long or too deeply nested to read", src=src,
                         hint=r"split it over several lines, or write a long sum as `\sum_{k=1}^{n} ...`.") from None
    finally:
        sys.setrecursionlimit(old)
    return expr, notes + tr.notes


_MATHY = re.compile(r"=|<|>|\\(?:le|ge|leq|geq|ne|neq|approx|in|openlabdef|sim)(?![A-Za-z])")
_BARE_WORD = re.compile(r"(?<![\\{A-Za-z])[A-Za-z]{3,}(?![A-Za-z}])")   # `train`, not `\sin`, not `{GELU}`


def parse_statement(src: str, lineno: int = 1, *, role: str = "", ics=(),
                    display: Optional[str] = None, domain: str = "",
                    dialect: Optional[str] = None) -> Statement:
    prose = " ".join(m.group(1) for m in _NOTE_RE.finditer(src))
    bare = _BARE_WORD.findall(_NOTE_RE.sub(" ", src))          # words outside \text and not commands
    if (len(prose.split()) >= 4 or len(bare) >= 3) and not _MATHY.search(_NOTE_RE.sub(" ", src)):
        raise ParseError("this reads as a sentence, not as math", src=src, lineno=lineno,
                         hint=r"write the equations themselves (`60 \text{ miles} / 2 \text{ h}`, "
                              r"`x + y = 15 \text{ and } x - y = 3`), or start a note with `>` to keep prose.")
    if re.search(r"(?:(?<![<>!\\&])=|\\openlabdef)\s*\??\s*$", src.strip()):
        raise ParseError("nothing after `=`", src=src, lineno=lineno,
                         hint="a trailing `=` asks for nothing: write `1+1` to evaluate it, or `x = 1+1` to bind x.")
    convert = None
    m = re.search(r"\\openlabin\{([^{}]*)\}", src)
    if m:
        convert = m.group(1).strip()
        src = (src[:m.start()] + src[m.end():]).strip()
    try:
        segs, ops = split_chain(src)
        if ops and len(_depth0_split(segs[0], _CHAIN_HEAD)) > 1:
            segs, ops = [src], []          # `2 < 3 = 3`, `p \implies q = r`: the grammar reads the whole relation
        notes: list = []
        if len(ops) >= 2:
            exprs = []
            for sg in segs:
                e, n = parse_expr(sg, dialect)
                if isinstance(e, sympy.Tuple) and len(_depth0_split(sg, _COMMA)) > 1:
                    raise ParseError("a comma inside a chain `a = b = c` makes a tuple of two statements",
                                     src=src, hint="put each statement on its own line")
                exprs.append(e)
                notes += n
            return Statement(src=src, lineno=lineno, lhs_src=segs[0], lhs=exprs[0], rhs=exprs[1],
                             notes=notes, chain=exprs, chain_ops=ops, role=role,
                             display=display or src, convert=convert)
        lhs = lhs_src = None
        approx = force = False
        if ops:
            lhs_src, rhs_src = segs
            approx, force = ops[0] == "approx", ops[0] == "def"
            lhs, notes = parse_expr(lhs_src, dialect)
        else:
            rhs_src = src
        rhs, rnotes = parse_expr(rhs_src, dialect)
        if domain:
            domain = re.sub(r"\\(?:text|mathrm)\s*\{\s*and\s*\}", r" \\land ", domain)
        domain_expr = parse_expr(domain, dialect)[0] if domain else None
        parsed_ics = []
        for ic in ics:
            l_src, r_src, _, _ = split_assignment(ic)
            parsed_ics.append((parse_expr(l_src, dialect)[0], parse_expr(r_src, dialect)[0]))
    except OpenlabError as err:
        err.lineno = lineno
        err.src = err.src or src
        raise
    return Statement(src=src, lineno=lineno, lhs_src=lhs_src, lhs=lhs, rhs=rhs, approx=approx,
                     force=force, notes=notes + rnotes, ics=parsed_ics, role=role,
                     display=display or src, convert=convert, domain=domain_expr)


_MARKDOWN = re.compile(r"^\s*>\s?(.*)$")


def statements(cell: str, units: bool = True) -> list[Clause]:
    """Cell text -> the statements to run, in order. A line starting with `>` is a markdown note."""
    out = []
    notes_by_line = {}
    kept_lines = []
    for i, raw in enumerate(cell.splitlines(), 1):
        m = _MARKDOWN.match(raw)
        if m:
            notes_by_line[i] = m.group(1)
            kept_lines.append("")
        else:
            kept_lines.append(raw)
    if notes_by_line:
        cell = "\n".join(kept_lines)
    pending_notes = sorted(notes_by_line.items())
    for lineno, line in split_statements(preprocess(cell, units=units)):
        while pending_notes and pending_notes[0][0] <= lineno:
            ln, text = pending_notes.pop(0)
            out.append(Clause(ln, text, role="markdown", display=text))
        try:
            clauses = _split_clauses(line)
        except OpenlabError as err:
            err.lineno = lineno
            raise
        for c in clauses:
            c.lineno = lineno
            c.display = c.display or c.text
            prev = out[-1] if out else None
            # an initial condition on the line after a differential equation belongs to it
            if (prev is not None and _is_ic(c.text) and _ODE_HINT.search(prev.text)
                    and _has_equals(prev.text) and not _is_ic(prev.text)):
                prev.ics.append(c.text)
                prev.display = f"{prev.display},\\; {c.text}"
                continue
            out.append(c)
    for ln, text in pending_notes:
        out.append(Clause(ln, text, role="markdown", display=text))
    return out
