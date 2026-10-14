"""Rendering of results and errors: HTML for Jupyter, LaTeX, and plain text."""

from __future__ import annotations

import html
import re

import sympy

_BOX = ("border-left:3px solid #7c8cff;padding:.2em .8em;margin:.3em 0;"
        "line-height:1.6")
_MUTED = "opacity:.65;font-family:monospace;font-size:.85em;white-space:pre-wrap"
_ERR = ("border-left:3px solid #e0524f;padding:.3em .8em;margin:.3em 0;"
        "font-family:monospace;font-size:.9em;white-space:pre-wrap")


def _math(tex: str) -> str:
    return rf"<div>\(\displaystyle {html.escape(tex)}\)</div>"


_NOTE = "border-left:3px solid #b8b8b8;padding:.2em .8em;margin:.3em 0;font-style:italic;opacity:.85"
_PROOF = {"proved": "#3aa05a", "disproved": "#e0524f", "not proved": "#e0a030"}


def _markdown(text: str) -> str:
    """A small, safe markdown subset: **bold**, *italic*, `code`, and $math$ (left to MathJax)."""
    t = html.escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?![*\w])", r"<i>\1</i>", t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    return t


def _prose(tex: str) -> str:
    """Rendered input with its \\text{…} runs as plain prose."""
    return rf"\(\displaystyle {html.escape(tex)}\)"


def result_html(r) -> str:
    kind = getattr(r.kind, "name", "")
    if kind == "NOTE":
        body = _markdown(r.src) if "markdown" in r.notes else _prose(r.latex_in)
        hint = "" if "markdown" in r.notes else f'<div style="{_MUTED}">{html.escape(r.interpretation)}</div>'
        return f'<div class="openlab-note" style="{_NOTE}">{body}{hint}</div>'
    if kind == "PROOF":
        verdict = r.interpretation.split(" ", 2)[0] if not r.interpretation.startswith("not proved") else "not proved"
        colour = _PROOF.get(verdict, "#7c8cff")
        steps = "".join(f"<li>{html.escape(n)}</li>" for n in r.notes)
        return (f'<div class="openlab-proof" style="border-left:3px solid {colour};padding:.2em .8em;margin:.3em 0">'
                f'{_math(r.latex_in)}<ol style="{_MUTED};margin:.2em 0">{steps}</ol>{_math(r.latex_out)}</div>')
    parts = [f'<div class="openlab" style="{_BOX}">', _math(r.latex_in)]
    if not r.quiet:
        parts.append(f'<div style="{_MUTED}">interpreted as: {html.escape(r.interpretation)}</div>')
    parts.append(_math(r.latex_out))
    parts.append("</div>")
    return "".join(parts)


def result_latex(r) -> str:
    return rf"$$\begin{{aligned}} {r.latex_in} \\ {r.latex_out} \end{{aligned}}$$"


def result_text(r) -> str:
    kind = getattr(r.kind, "name", "")
    if kind == "NOTE":
        return f"note: {r.src}"
    if kind == "PROOF":
        lines = [f"claim: {r.src}"] + [f"  {i}. {n}" for i, n in enumerate(r.notes, 1)]
        verdict = {True: "proved", False: "disproved"}.get(bool(r.value) if r.value is not None else None, "not proved")
        lines.append(f"=> {verdict}")
        return "\n".join(lines)
    lines = [f"in:   {r.src}"]
    if not r.quiet:
        lines.append(f"interpreted as: {r.interpretation}")
    out = r.value
    if isinstance(out, sympy.Integer) and out.p.bit_length() > 200:
        out = out.evalf(15)                 # exact value stays in r.value
    try:
        pretty = sympy.pretty(out, use_unicode=True)
    except Exception:  # pretty-printing is best effort
        try:
            pretty = str(out)
        except ValueError:
            pretty = "(too large to print)"
    if "\n" in pretty:
        lines.append("out:")
        lines.append(pretty)
    else:
        lines.append(f"out:  {pretty}")
    if r.numeric is not None:
        lines.append(f"      ≈ {r.numeric}")
    return "\n".join(lines)


def _shown(src):
    from .parse import display_src      # lazy: parse imports nothing from render
    return display_src(src or "")


def error_html(err) -> str:
    where = f" (line {err.lineno})" if getattr(err, "lineno", None) else ""
    parts = [f'<div class="openlab-error" style="{_ERR}">',
             f"<b>openlab{where}:</b> {html.escape(err.message)}"]
    if err.src:
        parts.append(f"\n  {html.escape(_shown(err.src))}")
        if err.column:
            parts.append("\n  " + " " * (err.column - 1) + "^")
    if err.hint:
        parts.append(f"\n<b>Hint:</b> {html.escape(err.hint)}")
    parts.append("</div>")
    return "".join(parts)


def error_text(err) -> str:
    where = f" (line {err.lineno})" if getattr(err, "lineno", None) else ""
    out = f"openlab{where}: {err.message}"
    if err.src:
        out += f"\n  {_shown(err.src)}"
        if err.column:
            out += "\n  " + " " * (err.column - 1) + "^"
    if err.hint:
        out += f"\nHint: {err.hint}"
    return out
