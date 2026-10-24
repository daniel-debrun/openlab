"""Openlab: a research framework you can trust. Researchers run deep research, paper-to-code replication and
prompt-to-proposal-to-execution through their own coding agents (Claude Code, Codex); openlab owns what a result's
trust rests on: the data, the splits, the sandboxed runner, the counted trials, the person-only holdout looks and
the hash-chained record of each study.

>>> import openlab
>>> s = openlab.study.new("demo", template="temporal")      # study.toml, model.py, recipes/
>>> s.materialise(); s.run()                              # seals study.toml, runs, audits, scores
>>> openlab.study.open("demo").status()

From a shell: ``openlab study new|materialise|run|status|claim|paper|replicate|... FOLDER``; for an agent, the MCP
server ``openlab-mcp``. The LaTeX checker behind derived claims and the paper's equation gate is internal
(:mod:`openlab.engine`).
"""

__version__ = "1.4.0"


def __getattr__(name):
    """`openlab.study` loads on first use, so a sandboxed child imports fast."""
    import importlib
    if name == "study":
        return importlib.import_module(".study", __name__)
    raise AttributeError(f"module 'openlab' has no attribute {name!r}")


__all__ = ["study", "__version__"]
