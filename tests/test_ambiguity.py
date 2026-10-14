"""Resolve-mode Earley never silently picks between different readings: the grammar has
NO ambiguity at all on the corpus. Every new grammar rule must keep this green."""

import pytest
from lark import Tree

from corpus import CORPUS
from openlab.engine.grammar import build_parser
from openlab.engine.parse import normalize_bars, preprocess

EXTRA = [r"\sin x \cos y \sin z", r"a b c d", r"2 x \sin x", r"x \sin x \int y dy",
         r"\frac{1}{2} \sum_{k=1}^n k \cdot 2", r"\int_0^1 \sin x \cos x dx",
         r"\sin(x)(y)", r"\sin x^2 y^3", r"x^2 y^2 z^2", r"-a - b - c", r"a - -b"]


def _ambig(tree, acc):
    if isinstance(tree, Tree):
        if tree.data == "_ambig":
            acc.append(tree)
        for ch in tree.children:
            _ambig(ch, acc)
    return acc


@pytest.mark.parametrize("src", [c.src for c in CORPUS] + EXTRA)
def test_no_ambiguity(src):
    tree = build_parser("explicit").parse(normalize_bars(preprocess(src).strip()))
    nodes = _ambig(tree, [])
    assert not nodes, f"{src!r} is ambiguous:\n" + "\n---\n".join(
        alt.pretty() for alt in nodes[0].children)
