"""Error taxonomy. Every error carries a ``hint`` -- the idiom that would have worked."""

from __future__ import annotations

from typing import Optional


class OpenlabError(Exception):
    """Base class. ``message`` says what went wrong, ``hint`` says what to type instead."""

    def __init__(self, message: str, *, hint: str = "", src: str = "",
                 column: Optional[int] = None, lineno: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.src = src
        self.column = column
        self.lineno = lineno

    def __str__(self) -> str:
        out = self.message
        if self.hint:
            out += f"\nHint: {self.hint}"
        return out


class ParseError(OpenlabError):
    """The LaTeX could not be read as an expression."""


class StrayDifferentialError(ParseError):
    """A ``dx`` appeared somewhere other than the end of an integrand."""


class UnsupportedError(OpenlabError):
    """Valid LaTeX that openlab deliberately does not execute (yet)."""


class SemanticsError(OpenlabError):
    """Parsed fine, but could not be executed against the namespace."""


class MultipleUnknowns(SemanticsError):
    """An equation with several unknowns; the caller may group it into a system."""
