r"""The paper's lints and gates: deterministic code over the agent's prose and the record, never a prompt.

Each finding has an id (stable while its sentence is unchanged), a rule, a location and a message. Findings of
bindings (unresolved, stale) never pass; any other passes only with a person's waiver: an approval with decision
waive, linked `on` a paper of the study or a claim in the finding's sentence, that names the finding's id.

| rule | fails when |
|---|---|
| number.typed | a digit in prose outside math, bindings, citations, \ref/\label and "Section 2"-style references, |
|              | not allowed by paper.toml (`allow`, and `years = true` for 1900-2100) |
| wording.significance | "significant(ly)" without a \labclaim in the sentence whose supported verdict rests on a |
|                      | test that rejected |
| wording.comparison | "outperform / beat / better than / improves on" without a supported claim or a bound |
|                    | comparison whose verdict is better |
| wording.causal | "cause(s) / causal / leads to / effect of" without a claim a person approved |
| wording.novelty | "novel / first to / no prior work / unprecedented": always |
| wording.proof | "proves / demonstrates conclusively": always |
| cite.quote | a \labcite quote not found in the source's abstract or full text (evidence's normalisation) |
| eq.gate | a \labeq whose LaTeX and Python function disagree on the probes, or cannot be checked |
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from dataclasses import dataclass

from .. import schema
from ..errors import StudyError

DEFAULT_ALLOW = ("1D", "2D", "3D")
_YEAR = re.compile(r"19\d\d|20\d\d|2100")
_REF_WORD = re.compile(r"(?:Sections?|Sec\.|§|Tables?|Figures?|Fig\.|Equations?|Eq\.|Appendix)[~\s]*$")
_TOKEN = re.compile(r"[\w.,\-]*\d[\w.,\-]*")
_SENTENCE_END = re.compile(r"(?<!\be\.g)(?<!\bi\.e)(?<!\bet al)(?<!\bvs)(?<!\bcf)(?<!\bFig)(?<!\bEq)(?<!\bSec)"
                           r"[.!?](?=\s|$)|\n[ \t]*\n")
_NOT_PROSE = [re.compile(p, re.S) for p in (
    r"\\\\(?:\[[^\]]*\])?",                                                         # line breaks, \\[2pt]
    r"\$\$.*?\$\$", r"(?<!\\)\$.*?(?<!\\)\$", r"\\\(.*?\\\)", r"\\\[.*?\\\]",       # math
    r"\\begin\{(equation|align|gather|multline|eqnarray|math|displaymath)(\*?)\}.*?\\end\{\1\2\}",
    r"\\(?:[cC]?ref|eqref|autoref|pageref|label|cite\w*|url|href|includegraphics|input|include|begin|end|"
    r"[vh]space|labsection|usepackage|RequirePackage|graphicspath|scalebox)\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?",
    # layout, not results: a command's optional argument, a length argument, a table's column spec, a set length
    r"(?:\\[A-Za-z]+\*?|\\begin\{[^{}]*\})\s*\[[^\]\n]*\]",
    r"\{\s*-?\d*\.?\d+\s*(?:pt|cm|mm|in|em|ex|bp|\\(?:line|text|column)width|\\textheight)\s*\}",
    r"\\begin\{(?:tabular\w*|array|longtable)\*?\}(?:\{[^{}]*\})?\{(?:[^{}]|\{[^{}]*\})*\}",
    r"\\(?:setlength|addtolength|setcounter|addtocounter|rule|resizebox)\{[^{}]*\}\{[^{}]*\}")]
# a name with digits in it (H0, pnl3, T2, x_1): a word that starts with a letter and has no decimal point
_NAME = re.compile(r"[A-Za-z][A-Za-z_]*\d+[A-Za-z_\d]*")


@dataclass(frozen=True)
class Finding:
    id: str
    rule: str
    location: str
    message: str
    waivable: bool = True
    claims: tuple = ()                  # the claims bound in the finding's sentence: a waiver may be on one of them

    def to_dict(self) -> dict:
        return {**dataclasses.asdict(self), "claims": list(self.claims)}

    @classmethod
    def of(cls, d: dict) -> "Finding":
        return cls(**{f.name: d[f.name] for f in dataclasses.fields(cls) if f.name in d} | {"claims": tuple(d.get("claims", ()))})


def finding(rule: str, file: str, line: int, key: str, message: str, waivable: bool = True, claims=()) -> Finding:
    """A finding whose id hashes the rule, the file and `key` (the sentence and the token, whitespace aside)."""
    h = hashlib.sha256(f"{rule}|{file}|{' '.join(key.split())}".encode()).hexdigest()[:8]
    return Finding(f"{rule}:{file}:{h}", rule, f"{file}:{line}", message, waivable, tuple(claims))


def _blank(m: re.Match) -> str:
    return re.sub(r"[^\n]", " ", m.group(0))


def prose(text: str, macros: list) -> str:
    """The text with everything that is not prose blanked (offsets kept): the binding macros, math, line breaks,
    and references, labels, citations, URLs and other arguments that are not words."""
    chars = list(text)
    for m in macros:
        chars[m.start:m.end] = re.sub(r"[^\n]", " ", text[m.start:m.end])
    out = "".join(chars)
    for p in _NOT_PROSE:
        out = p.sub(_blank, out)
    return out


def sentences(plain: str) -> list:
    """(start, end) of each sentence: ended by . ! or ? before a space, or by a blank line."""
    spans, start = [], 0
    for m in _SENTENCE_END.finditer(plain):
        spans.append((start, m.end()))
        start = m.end()
    return spans + [(start, len(plain))]


def _at(spans: list, pos: int) -> tuple:
    return next(s for s in spans if s[0] <= pos < s[1] or s == spans[-1])


def _line(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


_MATH = _NOT_PROSE[1:6]
# in math, a decimal, or an integer of two or more digits that is not a sub/superscript, is a typed result
_MATH_NUMBER = re.compile(r"(?<![_^\w.])(?<!_\{)(?<!\^\{)(?:\d*\.\d+|\d{2,})")


def math_numbers(file: str, text: str, macros: list, allow: tuple) -> list:
    """A finding for each result typed inside math (`$0.85$`): math may hold constants and indices, not numbers
    that should be bound. Equations bound with \labeq are gated instead and blanked with the other macros."""
    chars = list(text)
    for m in macros:
        chars[m.start:m.end] = re.sub(r"[^\n]", " ", text[m.start:m.end])
    masked, out = "".join(chars), []
    for p in _MATH:
        for span in p.finditer(masked):
            for n in _MATH_NUMBER.finditer(span.group(0)):
                if n.group(0) in allow:
                    continue
                pos = span.start() + n.start()
                out.append(finding("number.math", file, _line(text, pos), f"{span.group(0)}|{n.group(0)}",
                                   f"`{n.group(0)}` is typed inside math: bind it with \\lab{{...}} or allow it in "
                                   "paper.toml if it is a constant"))
    return out


def numbers(file: str, plain: str, allow: tuple, years: bool) -> list:
    """A finding for each number typed in prose."""
    spans, out = sentences(plain), []
    for m in _TOKEN.finditer(plain):
        tok = m.group(0).strip(".,-")
        if (tok in allow or tok in DEFAULT_ALLOW or (years and _YEAR.fullmatch(tok)) or _NAME.fullmatch(tok)
                or (re.fullmatch(r"\d+(\.\d+)*", tok) and _REF_WORD.search(plain[max(0, m.start() - 16):m.start()]))):
            continue
        s, e = _at(spans, m.start())
        hint = "; years pass when paper.toml sets years = true" if _YEAR.fullmatch(tok) else ""
        out.append(finding("number.typed", file, _line(plain, m.start()), f"{plain[s:e]}|{tok}",
                           f"`{tok}` is typed in the prose: bind it with \\lab{{<node id>#<path>}}, or add it to "
                           f"paper.toml allow if it is a name{hint}"))
    return out


# --- wording ---------------------------------------------------------------------------------------------------
def _claims(nodes: list) -> list:
    return [n for n in nodes if n.kind == "claim"]


def tested(rec, claim) -> bool:
    """The claim's supported verdict rests on a test that rejected: its tests, or those of the hypotheses it
    tests or answers, give it (schema.tested), as they decide its verdict."""
    hyps = [d for _, rel, d in rec.links(claim.id) if rel in ("tests", "answers") and d.startswith("hypothesis:")]
    return schema.tested(rec, [claim.id, *hyps]) == "supported" and schema.current_verdict(rec, claim) == "supported"


def approved(rec, claim) -> bool:
    """A person approved the claim."""
    return any(a.actor == "person" and a.payload["decision"] == "approve"
               for a in (rec.get(s) for s, _, _ in rec.links(claim.id, rel="on", incoming=True)) if a.kind == "approval")


WORDING = (
    ("significance", r"significant(?:ly)?",
     lambda rec, ns: any(tested(rec, c) for c in _claims(ns)),
     "a \\labclaim in the same sentence whose supported verdict rests on a test that rejected"),
    ("comparison", r"outperform(?:s|ed|ing)?|beat(?:s|ing)?|better than|improve[sd]? on",
     lambda rec, ns: any(schema.current_verdict(rec, n) == "supported" if n.kind == "claim"
                         else n.kind == "comparison" and n.payload["verdict"] == "better" for n in ns),
     "a \\labclaim with verdict supported, or a bound comparison whose verdict is better, in the same sentence"),
    ("causal", r"cause[sd]?|causal(?:ly)?|leads? to|effect of",
     lambda rec, ns: any(approved(rec, c) for c in _claims(ns)),
     "a \\labclaim in the same sentence that a person approved (a causal reading needs a causal design)"),
    ("novelty", r"novel(?:ty)?|first to|no prior (?:work|stud(?:y|ies))|unprecedented", None,
     'it is never allowed: write "not found in the searched sources"'),
    ("proof", r"proves?|demonstrates? conclusively", None,
     "it is never allowed: say what the result supports, and with what interval"),
)
_WORDING = [(name, re.compile(rf"\b(?:{pat})\b", re.I), ok, need) for name, pat, ok, need in WORDING]


def wording(rec, file: str, plain: str, bounds: list) -> list:
    """A finding for each trigger word whose sentence lacks what the word requires."""
    out, owned = [], [b for b in bounds if b.node and b.macro.file == file]
    for s, e in sentences(plain):
        sent, nodes = plain[s:e], None
        for name, pat, ok, need in _WORDING:
            for m in pat.finditer(sent):
                if nodes is None:
                    nodes = [rec.get(b.node) for b in owned if s <= b.macro.start < e]
                if ok is not None and ok(rec, nodes):
                    continue
                said = (f"“{m.group(0)}” needs {need}" if ok else f"“{m.group(0)}”: {need}")
                out.append(finding(f"wording.{name}", file, _line(plain, s + m.start()), f"{sent}|{m.group(0).lower()}",
                                   said, claims=[n.id for n in _claims(nodes)]))
    return out


# --- gates and bindings ------------------------------------------------------------------------------------------
def bound(b) -> list:
    """The findings of one binding: it does not resolve, or it is stale. Neither can be waived."""
    m = b.macro
    if b.error:
        return [finding("binding.unresolved", m.file, m.line, m.text, f"{m.text[:120]}: {b.error}", waivable=False)]
    if b.stale:
        return [finding("binding.stale", m.file, m.line, m.text, f"{b.stale}: run `openlab study run` and bind the "
                        "new result" + ("" if m.option == "ablation" else "; an earlier variant in an ablation table "
                                        "binds as \\lab[ablation]{...} (same spec and data, earlier code)"),
                        waivable=False)]
    return []


def cites(rec, bounds: list) -> list:
    """A finding for each \\labcite whose quote is not in its source (schema's quote check, as evidence uses)."""
    out = []
    for b in bounds:
        if b.macro.name == "labcite" and b.node:
            try:
                schema._quoted(rec, rec.get(b.node), b.macro.args[1])
            except StudyError as e:
                out.append(finding("cite.quote", b.macro.file, b.macro.line, b.macro.text,
                                   f"{b.node}: {e.error} ({e.meaning})"))
    return out


def traced(rec, bounds: list) -> list:
    """A finding for each \labclaim whose provenance chain is incomplete (a ground retracted since, a hypothesis
    with no test, a run with no recorded code or environment): the 100% traceability gate, applied at build."""
    from .. import graph
    out = []
    for b in bounds:
        if b.macro.name == "labclaim" and b.node:
            _, missing = graph.chain(rec, b.node)
            if missing:
                out.append(finding("claim.untraceable", b.macro.file, b.macro.line, b.macro.text,
                                   f"{b.node}: its chain is incomplete: {'; '.join(map(str, missing))[:300]}",
                                   claims=(b.node,)))
    return out


def equation(macro, why: str) -> Finding:
    return finding("eq.gate", macro.file, macro.line, macro.text, f"\\labeq{{{macro.args[0]}}}: {why}")


def waiver(rec, f: Finding):
    """The person's waiver that lets finding `f` pass, or None: an approval with decision waive, by a person, whose
    reason names the finding's id, on a paper of the study or on a claim in the finding's sentence."""
    if not f.waivable:
        return None
    for a in rec.find("approval", decision="waive"):
        if a.actor != "person" or f.id not in a.payload["reason"]:
            continue
        on = rec.links(a.id, rel="on")
        target = rec.get(on[0][2]) if on else None
        if target is not None and (target.kind == "paper" or (target.kind == "claim" and target.id in f.claims)):
            return a
    return None
