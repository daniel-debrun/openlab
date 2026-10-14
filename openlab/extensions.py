"""Methods a package adds to the core, so the core itself holds none of a domain's vocabulary.

The core holds the metrics a study scores with and the functions the LaTeX engine knows; a domain's own (a
field's metric, its special functions) is registered here by the package that owns it. Any installed package
names a module under the ``openlab.extensions`` entry-point group
(``[project.entry-points."openlab.extensions"] example = "example_pkg.openlab_ext"``): importing that module registers.

A name the core already has wins over a registered one: adding never changes what an existing name means.
"""

from __future__ import annotations

from collections.abc import MutableMapping


class Registry(MutableMapping):
    """name -> method. The first read imports the packages that register, so a lookup never misses one."""

    def __init__(self):
        self._d: dict = {}

    def __getitem__(self, name):
        load()
        return self._d[name]

    def __setitem__(self, name, fn):
        self._d[name] = fn

    def __delitem__(self, name):
        del self._d[name]

    def __iter__(self):
        load()
        return iter(self._d)

    def __len__(self):
        load()
        return len(self._d)


#: study metric -> (fn(y: list, yhat: list) -> float, higher_is_better: bool), read by openlab.study.metrics
METRICS = Registry()
#: function -> fn(*args) -> a SymPy expression, for ``\operatorname{name}(…)`` in a statement
FUNCS = Registry()

_loaded = False


def load() -> None:
    """Import the registering packages once: every ``openlab.extensions`` entry point."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    import warnings
    from importlib.metadata import entry_points
    for ep in entry_points(group="openlab.extensions"):
        try:
            ep.load()
        except Exception as exc:                 # noqa: BLE001 -- one broken package must not take openlab down
            warnings.warn(f"openlab extension {ep.name!r} did not load: {type(exc).__name__}: {exc}", stacklevel=2)
