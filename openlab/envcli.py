"""`openlab env`: the Python environment a study's code runs in (openlab's own: the sandbox runs this interpreter).

    openlab env                      the installed distributions, name and version
    openlab env add PKG [PKG ...]    install packages a study needs (e.g. utide, statsmodels, torch), never changing
                                    the version of one already installed: each is pinned as a pip constraint, so an
                                    install that would upgrade numpy or polars under openlab and the other studies is
                                    refused, and nothing changes

A run records the versions of the packages its code imports (its env), so a study whose packages change gets a new
run key, and the studies that do not import them keep theirs.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from importlib import metadata


def installed() -> dict:
    return {d.metadata["Name"]: d.version for d in metadata.distributions() if d.metadata["Name"]}


def add(packages: list) -> dict:
    """pip install `packages` under constraints pinning every installed distribution."""
    before = installed()
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("".join(f"{n}=={v}\n" for n, v in sorted(before.items())))
        pins = f.name
    p = subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-c", pins, *packages],
                       capture_output=True, text=True)
    after = installed()
    return {"ok": p.returncode == 0, "added": {n: v for n, v in after.items() if n not in before},
            "changed": {n: f"{before[n]} -> {v}" for n, v in after.items() if n in before and before[n] != v},
            "log": (p.stdout + p.stderr)[-3000:]}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv in (["-h"], ["--help"]):
        print(__doc__.strip())
        return 0
    if not argv:
        for n, v in sorted(installed().items(), key=lambda x: x[0].lower()):
            print(f"{n}=={v}")
        return 0
    if argv[0] != "add" or len(argv) < 2:
        print("usage: openlab env [add PKG ...]", file=sys.stderr)
        return 2
    r = add(argv[1:])
    if not r["ok"]:
        print(f"error: pip could not install {' '.join(argv[1:])} without changing an installed package\n"
              f"meaning: every installed distribution is pinned, so openlab and the other studies keep their versions\n"
              f"fix: pick a version of it that fits the installed ones (PKG==X.Y), or do without it\n\n{r['log']}",
              file=sys.stderr)
        return 1
    print("added: " + (", ".join(f"{n} {v}" for n, v in r["added"].items()) or "nothing new (already installed)"))
    return 0
