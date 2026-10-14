"""Write docs/index.md from README.md (the README stays the single source of the front page), docs/pypi.md (the
package's description on PyPI: the README without its images and charts, every link absolute) and docs/bench.md
from bench/README.md; every other page is written by hand.

    python docs/build_docs.py && mkdocs build
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")

PAGES = {"index.md": ("^# Openlab", "## More")}

WHERE = """
## Where to go

- [Install](install.md): the extras, the sandbox, Claude Code and Codex
- [The first five minutes](first-five-minutes.md): a workspace, your person key, one study from scaffold to a holdout look
- [Supervising agents](supervising-agents.md): the tree, `openlab watch`, the inbox, budgets, prompting leads, hooks, the person key
- [The live view: openlab show](show.md): agents, plans, the inbox, a lead's Prompt chat, pause and play, the Workspace
- [Research from a prompt](research-from-prompt.md): proposal, approval, execution by a tree of agents
- [A literature review](literature-review.md): search, screening by several agents, checked quotes, PRISMA
- [The plan: study.toml](plan.md): the seal, amendments, the embargo and budgets
- [Guarantees and refusals](guards.md): each guarantee, what enforces it, and the refusal you meet
- [Claims, tests and the paper](claims.md)
- [ML models in a study](ml.md): seeds, the search tree, the registry
- [Module checks, ensembles, critiques](checks.md): a broken model fails in seconds; ensembles as counted runs; advice with a track record
- [Search, selection and lessons](search.md): what to try next, the final candidate by a sealed rule, lessons that rest on comparisons
- [On a server, a container or a GPU box](box.md): container limits, the queue, GPUs, person acts over SSH
- [A W&B mirror](mirror.md): a view of a study's runs in Weights & Biases, never the record
- [Coming from R](from-r.md)
- [What is in the box](features.md): paper to code, the paper's bindings, the frontier, lessons, the record's files
- [Command reference](cli.md)
- [For agents](AGENTS.md): the contract `openlab-mcp` serves
- [Glossary](GLOSSARY.md)
- [The benchmark](bench.md)
- [Benchmark mode: MLE-bench](benchmark.md): the same agent with and without openlab, graded where the answers are
"""


def section(start: str, end: str) -> str:
    a = re.search(start, README, re.M)
    b = re.search(re.escape(end), README[a.end():]) if a else None
    if a is None:
        return ""
    return README[a.start(): a.end() + b.start()] if b else README[a.start():]


BLOB = "https://github.com/daniel-debrun/openlab/blob/main"


def pypi() -> str:
    """The README for PyPI, which draws no Mermaid and cannot load this repository's images: the logo, the
    screenshots (and their gallery) and the charts left out; every relative link made absolute."""
    text = re.sub(r'^<p align="center"><img[^\n]*\n+', "", README, flags=re.M)
    text = re.sub(r"^### openlab show\n.*?(?=^## )", "", text, flags=re.M | re.S)
    text = re.sub(r"^!\[[^\]]*\]\((?!https?://)[^)]*\)\n+", "", text, flags=re.M)
    text = re.sub(r"```mermaid\n.*?```\n+", "", text, flags=re.S)
    text = re.sub(r"^\[!\[tests\][^\n]*\n", "", text, flags=re.M)          # the CI badge needs the repository's access
    return re.sub(r"\]\((?!https?://|#)([^)]+)\)", lambda m: f"]({BLOB}/{m.group(1)})", text)


def main() -> None:
    out = ROOT / "docs"
    for page, (start, end) in PAGES.items():
        text = section(start, end).strip().replace("](docs/", "](").replace("](openlab/AGENTS.md)", "](AGENTS.md)") + "\n" + WHERE
        (out / page).write_text(text, encoding="utf-8")
    (out / "pypi.md").write_text(pypi(), encoding="utf-8")
    (out / "bench.md").write_text((ROOT / "bench" / "README.md").read_text(encoding="utf-8").replace("# bench: what the structure catches", "# The benchmark: what the structure catches"), encoding="utf-8")
    print("docs written:", ", ".join(sorted(p.name for p in out.glob("*.md"))))


if __name__ == "__main__":
    main()
