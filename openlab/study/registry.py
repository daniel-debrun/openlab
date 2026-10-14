r"""The study's model store, in MLflow: each run's final model is logged here, packed in the sandbox and never
unpickled outside it.

``Registry(name, arm="study", root=<study>/.openlab/mlflow)`` opens the store; ``log_packed`` keeps a model that
:func:`pack` packed (a new version each time), ``packed`` reads its pickle back as bytes and ``get`` its entry.
Each entry is an MLflow logged model registered as ``<arm>-<name>``. ``check_deterministic`` refuses a model whose
``predict`` gives different outputs on the same input. Browse a store with
``mlflow ui --backend-store-uri sqlite:///<study>/.openlab/mlflow/mlflow.db``.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

#: What an entry can be.
KINDS = ("model",)
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}")        # an entry: MLflow model names take no dots
_ARM = re.compile(r"[a-z][a-z0-9_]{0,30}")                   # the arm prefixes the name: `<arm>-<name>`
_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}")        # a study's run name


@dataclass
class Entry:
    """One registered statement: its LaTeX, what openlab said about it, and where it came from."""
    name: str
    version: int
    kind: str
    verdict: str
    methods: list
    sha: str
    tex: str
    notebook: str                   # the run name (the study); the tag keeps its older name
    note: str = ""
    refutes: str = ""
    results: list = field(default_factory=list, repr=False)

    def __str__(self):
        how = f" by {', '.join(self.methods)}" if self.methods else ""
        ref = f", refutes {self.refutes}" if self.refutes else ""
        if not self.version:
            return f"not logged: {self.verdict}{how} [{self.sha}]"
        return f"{self.kind} {self.name} v{self.version}: {self.verdict}{how}{ref} [{self.sha}, from {self.notebook}]"


def _default_root() -> Path:
    """``<project>/.data/mlflow``, the project being the nearest folder above the working directory
    that has a ``.git`` or a ``.data`` -- so a run from a subfolder shares the project's store."""
    here = Path.cwd().resolve()
    for d in (here, *here.parents):
        if (d / ".git").exists() or (d / ".data").is_dir():
            return d / ".data" / "mlflow"
    return here / ".data" / "mlflow"


def _quiet():
    """MLflow prints progress bars and "Successfully registered" lines: keep the output clean."""
    stack = contextlib.ExitStack()
    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
    stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
    return stack


class Registry:
    """The model store as seen from one study: every entry it logs is attached to the study's run."""

    def __init__(self, notebook: str, *, arm: str = "study", root: Optional[str] = None):
        import mlflow
        from mlflow import MlflowClient
        if not _RUN.fullmatch(notebook):
            raise ValueError("a run name is letters, digits, `_`, `.`, `-` (up to 100)")
        if not _ARM.fullmatch(arm):
            raise ValueError("an arm is lower-case letters, digits and `_`")
        self.notebook, self.arm = notebook, arm
        self.root = Path(root or os.environ.get("OPENLAB_REGISTRY") or _default_root()).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.uri = f"sqlite:///{self.root / 'mlflow.db'}"
        mlflow.set_tracking_uri(self.uri)
        self.client = MlflowClient(self.uri)
        with _quiet():
            name = f"openlab/{arm}"
            exp = self.client.get_experiment_by_name(name)
            here = (self.root / "artifacts" / arm).as_uri()
            self.experiment_id = exp.experiment_id if exp else self.client.create_experiment(name, artifact_location=here)
            if exp and exp.artifact_location != here:      # a copied registry: MLflow's own writes land here too
                import sqlite3
                with contextlib.closing(sqlite3.connect(self.root / "mlflow.db")) as db, db:
                    db.execute("update experiments set artifact_location = ? where experiment_id = ?",
                               (here, int(exp.experiment_id)))
            runs = self.client.search_runs([self.experiment_id], filter_string=f"tags.mlflow.runName = '{notebook}'",
                                           max_results=1, order_by=["attributes.start_time DESC"])
            self._run_id = runs[0].info.run_id if runs else None     # made by the first log: opening leaves no trace

    @property
    def run_id(self) -> str:
        """The study's run, created the first time something is logged."""
        if self._run_id is None:
            with _quiet():
                self._run_id = self.client.create_run(self.experiment_id, run_name=self.notebook,
                                                      tags={"openlab.notebook": self.notebook, "openlab.arm": self.arm}).info.run_id
        return self._run_id

    def __repr__(self):
        run = self._run_id[:8] if self._run_id else "none yet"
        return f"Registry({self.arm}/{self.notebook}: {self.uri}, run {run})"

    def _registered(self, name: str) -> str:
        if not _NAME.fullmatch(name):
            raise ValueError(f"`{name}`: a name is letters, digits, `_` and `-` (up to 100; no dots)")
        return f"{self.arm}-{name}"

    def log(self, name: str, tex: str, *, kind: str = "model", note: str = "", force: bool = False) -> Entry:
        """Keep ``tex`` (a model's formula or repr) under ``name``: a new version, unless the same text was the last
        version and ``force`` is off. What is recorded is the text and its sha, never a verdict."""
        import mlflow
        from .. import __version__
        if kind not in KINDS:
            raise ValueError(f"kind is one of {', '.join(KINDS)}")
        reg_name = self._registered(name)
        results, verdict, methods, refutes = [], "unverified", [], ""
        sha = hashlib.sha256(tex.encode()).hexdigest()[:12]
        if not force:
            vs = self._versions(reg_name)
            last = vs[-1].tags if vs else {}
            if vs and (last.get("sha"), last.get("verdict"), last.get("kind")) == (sha, verdict, kind):
                return Entry(name, int(vs[-1].version), kind, verdict, methods, sha, tex, last.get("notebook", ""),
                             last.get("note", ""), refutes=last.get("refutes", ""), results=results)
        path = f"{kind}/{name}@{sha}.openlab"
        meta = {"kind": kind, "verdict": verdict, "method": ", ".join(methods)[:500], "sha": sha,
                "notebook": self.notebook, "statements": str(len(results)), "note": note[:500], "refutes": refutes}
        with _quiet():
            self._write(path, tex.encode())
            mlflow.set_experiment(experiment_id=self.experiment_id)
            model = mlflow.create_external_model(name=reg_name, source_run_id=self.run_id, model_type=f"openlab-{kind}",
                                                 params=meta, tags={"artifact": path, "openlab": __version__})
            mv = mlflow.register_model(f"models:/{model.model_id}", reg_name,
                                       tags={**meta, "artifact": path, "run_id": self.run_id, "model_id": model.model_id})
        return Entry(name, int(mv.version), kind, verdict, methods, sha, tex, self.notebook, note, refutes=refutes,
                     results=results)

    def log_model(self, name: str, model, note: str = "") -> Entry:
        """Keep a fitted model (anything with ``predict`` that pickles): a new version each time. What is
        recorded as its statement is its formula (a ``%%fit`` model) or its repr; the object itself is an
        artifact beside it, read back with :meth:`model`. Beside the pickle: its class source (when
        ``inspect`` finds it) and the Python and package versions it was pickled under."""
        return self.log_packed(name, *pack(model), note=note)

    def log_packed(self, name: str, blob: bytes, info: dict, note: str = "") -> Entry:
        """Keep a model that :func:`pack` packed, perhaps in another process: it is stored, never unpickled here
        (a study's model is packed in its sandbox, and unpickling would run its code outside it)."""
        import json
        e = self.log(name, info["text"], kind="model", force=True, note=note or info["note"])
        for ext, data in (("pkl", blob), ("json", json.dumps(info["meta"], indent=1).encode()),
                          ("py", info["source"].encode())):
            self._write(f"model/{name}@v{e.version}.{ext}", data)
        return e

    def _write(self, rel: str, data: bytes) -> None:
        """An artifact of this study's run, written under this registry's root (where `_local` reads it), not
        the experiment's recorded location: so a copied registry keeps its new entries."""
        f = self.root / "artifacts" / self.arm / self.run_id / "artifacts" / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data)

    def model(self, name: str, version: Optional[int] = None):
        """A registered model, unpickled. Only read a registry you wrote: unpickling runs code. Pickled under
        another Python minor version, a failed load says so and points at the stored class source."""
        import json
        import pickle
        import platform
        import warnings
        e, d = self._model_dir(name, version)
        f, meta_f = d / f"{name}@v{e.version}.pkl", d / f"{name}@v{e.version}.json"
        meta = json.loads(meta_f.read_text()) if meta_f.is_file() else {}
        was, now = meta.get("python", ""), platform.python_version()
        if not was or was.split(".")[:2] == now.split(".")[:2]:
            return pickle.loads(f.read_bytes())
        hint = (f"{name} v{e.version} was pickled under Python {was}, this is {now}; its class source is "
                f"{d / f'{name}@v{e.version}.py'} and its package versions are in {meta_f}: rebuild it from those")
        try:
            obj = pickle.loads(f.read_bytes())
        except Exception as ex:                     # noqa: BLE001 -- any unpickling failure gets the same pointer
            raise RuntimeError(f"{hint} ({type(ex).__name__}: {ex})") from ex
        warnings.warn(hint, stacklevel=2)
        return obj

    def packed(self, name: str, version: Optional[int] = None) -> bytes:
        """A registered model's pickle as bytes, never unpickled here (a study unpickles it in its sandbox)."""
        e, d = self._model_dir(name, version)
        return (d / f"{name}@v{e.version}.pkl").read_bytes()

    def _model_dir(self, name: str, version: Optional[int]) -> tuple:
        """(the entry, the folder of its pickle, metadata and class source)."""
        e = self.get(name, version)
        if e.kind != "model":
            raise KeyError(f"{name} is a {e.kind}, not a model")
        mv = next(v for v in self._versions(self._registered(name)) if int(v.version) == e.version)
        return e, Path(self._local(mv.tags["run_id"])) / "model"

    def _local(self, run_id: str) -> str:
        """A run's artifact folder under this registry's root (so a copied registry reads on another machine);
        the location MLflow recorded when that folder is not here."""
        here = self.root / "artifacts" / self.arm / run_id / "artifacts"
        if here.is_dir():
            return str(here)
        from urllib.parse import unquote, urlparse
        with _quiet():
            uri = self.client.get_run(run_id).info.artifact_uri
        return unquote(urlparse(uri).path) if uri.startswith("file:") else uri

    # --- reading ------------------------------------------------------------------------------------
    def _versions(self, reg_name: str) -> list:
        with _quiet():
            return sorted(self.client.search_model_versions(f"name='{reg_name}'"), key=lambda v: int(v.version))

    def get(self, name: str, version: Optional[int] = None) -> Entry:
        """A registered entry (the latest version unless one is named), with its LaTeX."""
        vs = self._versions(self._registered(name))
        if not vs:
            raise KeyError(f"nothing registered as {self._registered(name)}")
        mv = vs[-1] if version is None else next((v for v in vs if int(v.version) == int(version)), None)
        if mv is None:
            raise KeyError(f"{self._registered(name)} has no version {version}")
        t = mv.tags
        tex = (Path(self._local(t["run_id"])) / t["artifact"]).read_text()
        return Entry(name, int(mv.version), t.get("kind", ""), t.get("verdict", ""),
                     [m for m in t.get("method", "").split(", ") if m], t.get("sha", ""), tex,
                     t.get("notebook", ""), t.get("note", ""), refutes=t.get("refutes", ""))


def pack(model) -> tuple[bytes, dict]:
    """A fitted model as :meth:`Registry.log_packed` stores it: its cloudpickle, and {"text", "note", "source",
    "meta"} (its formula or repr, a default note, its class source, the Python and packages it was pickled under)."""
    import cloudpickle
    import importlib.metadata as md
    import inspect
    import platform
    import sys
    if not hasattr(model, "predict"):
        raise ValueError("a model is an object with `predict`")
    # cloudpickle stores a class defined in a notebook or script by value, so it loads in another process;
    # a module that is not an installed distribution (a local .py file) is stored by value too
    mod = sys.modules.get(type(model).__module__)
    dists = md.packages_distributions()
    # a class defined in a notebook cell reports `builtins` (the cell runs without a module): cloudpickle stores it
    # by value on its own, and registering `builtins` itself by value would try to pickle the interpreter
    byval = mod is not None and mod.__name__ not in ("__main__", "builtins") and mod.__name__.split(".")[0] not in dists
    if byval:
        cloudpickle.register_pickle_by_value(mod)
    try:
        blob = cloudpickle.dumps(model)
    finally:
        if byval:
            cloudpickle.unregister_pickle_by_value(mod)
    try:
        src = inspect.getsource(type(model))
    except (OSError, TypeError):
        src = ""
    pkgs = {}
    for top in sorted({m.split(".")[0] for m in list(sys.modules)} & set(dists)):
        for d in dists[top]:
            with contextlib.suppress(md.PackageNotFoundError):
                pkgs[d] = md.version(d)
    meta = {"python": platform.python_version(), "class": f"{type(model).__module__}.{type(model).__qualname__}",
            "by_value": [mod.__name__] if byval else [], "packages": pkgs}
    return blob, {"text": getattr(model, "formula", "") or repr(model), "source": src, "meta": meta,
                  "note": f"{type(model).__name__} {getattr(model, 'family', '')}".strip()}


def check_deterministic(model, X) -> None:
    """Raise ValueError unless ``model.predict(X)`` gives the same output twice (NaN equal to NaN). A model
    whose scores change between calls (unseeded sampling, set iteration order) cannot be graded or replicated."""
    import numpy as np
    a, b = model.predict(X), model.predict(X)
    a, b = (a[0], b[0]) if isinstance(a, tuple) else (a, b)
    if hasattr(a, "columns"):                       # a frame (mu/sigma, quantiles, members): column by column
        if list(a.columns) != list(getattr(b, "columns", [])) or len(a) != len(b):
            raise ValueError("predict is not deterministic: two calls on the same input gave different columns or "
                             "rows. Fix the seed or remove the randomness, then submit again")
        for c in a.columns:
            try:
                x, y = np.asarray(a[c].to_numpy(), dtype=float), np.asarray(b[c].to_numpy(), dtype=float)
                same = bool(np.all((x == y) | (np.isnan(x) & np.isnan(y))))
            except (TypeError, ValueError):
                same = a[c].to_list() == b[c].to_list()
            if not same:
                raise ValueError(f"predict is not deterministic: two calls on the same input gave a different {c}. "
                                 "Fix the seed (random_state, np.random.default_rng(seed)) or remove the randomness, "
                                 "then submit again")
        return
    try:
        x = np.asarray([np.nan if v is None else v for v in a], dtype=float)
        y = np.asarray([np.nan if v is None else v for v in b], dtype=float)
        diff = int((~((x == y) | (np.isnan(x) & np.isnan(y)))).sum()) if x.shape == y.shape else -1
    except (TypeError, ValueError):                 # labels or other non-numeric output
        x, y = list(a), list(b)
        diff = sum(u != v for u, v in zip(x, y)) if len(x) == len(y) else -1
    if diff:
        what = "a different number of outputs" if diff < 0 else f"different values on {diff} of {len(x)} rows"
        raise ValueError(f"predict is not deterministic: two calls on the same input gave {what}. Fix the seed "
                         "(random_state, np.random.default_rng(seed)) or remove the randomness, then submit again")
