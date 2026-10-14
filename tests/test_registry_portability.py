"""Registry portability (Workstream A): check_deterministic, the stored source/metadata beside a pickled model,
a local module's class loading after the module is gone, a copied registry, and the Python-mismatch message."""

import json
import shutil
import sys
import uuid

import numpy as np
import pytest

from openlab.study import registry
from openlab.study.registry import check_deterministic


class _Out:
    def __init__(self, f):
        self.predict = f


def test_check_deterministic_nan_equal_passes():
    check_deterministic(_Out(lambda X: [1.0, float("nan"), None]), None)


def test_check_deterministic_refuses_random_predict():
    rng = np.random.default_rng()
    with pytest.raises(ValueError, match="not deterministic"):
        check_deterministic(_Out(lambda X: rng.normal(size=5)), None)


def test_check_deterministic_labels_and_tuples():
    check_deterministic(_Out(lambda X: ["buy", "sell", "hold"]), None)
    check_deterministic(_Out(lambda X: ([0.1, 0.2], {"extra": 1})), None)
    calls = iter([["a", "b"], ["a", "c"]])
    with pytest.raises(ValueError, match="1 of 2"):
        check_deterministic(_Out(lambda X: next(calls)), None)


# --- the MLflow registry ----------------------------------------------------------------------------------
pytest.importorskip("mlflow")

SRC = '''
class Doubler:
    """A model from a local file that is not an installed package."""
    def __init__(self, k):
        self.k = k
    def predict(self, X):
        return [self.k * x for x in X]
'''


@pytest.fixture
def local_model(tmp_path, monkeypatch):
    mod = f"openlab_test_local_{uuid.uuid4().hex[:8]}"
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / f"{mod}.py").write_text(SRC)
    monkeypatch.syspath_prepend(str(tmp_path / "src"))
    m = __import__(mod).Doubler(3)
    yield m, mod
    sys.modules.pop(mod, None)


def _artifacts(root):
    return {p.suffix: p for p in root.rglob("m@v1.*")}


def test_log_model_stores_meta_and_source_and_loads_without_the_module(tmp_path, monkeypatch, local_model):
    monkeypatch.chdir(tmp_path)
    m, mod = local_model
    reg = registry.Registry("nb", arm="runs", root=str(tmp_path / "ml"))
    reg.log_model("m", m)
    files = _artifacts(tmp_path / "ml")
    assert set(files) == {".pkl", ".json", ".py"}
    meta = json.loads(files[".json"].read_text())
    assert meta["python"] and isinstance(meta["packages"], dict) and meta["by_value"] == [mod]
    assert "class Doubler" in files[".py"].read_text()
    sys.modules.pop(mod)                                   # the module is gone: by-value pickle still loads
    shutil.rmtree(tmp_path / "src")
    assert reg.model("m").predict([1, 2]) == [3, 6]


def test_a_copied_registry_loads(tmp_path, monkeypatch, local_model):
    monkeypatch.chdir(tmp_path)
    m, _ = local_model
    reg = registry.Registry("nb", arm="runs", root=str(tmp_path / "ml"))
    reg.log("eq", "x = 1")
    reg.log_model("m", m)
    shutil.copytree(tmp_path / "ml", tmp_path / "moved")
    shutil.rmtree(tmp_path / "ml")
    other = registry.Registry("nb", arm="runs", root=str(tmp_path / "moved"))
    assert other.get("eq").tex == "x = 1"
    assert other.model("m").predict([2]) == [6]


def test_other_python_minor_warns_and_points_at_the_source(tmp_path, monkeypatch, local_model):
    import platform
    monkeypatch.chdir(tmp_path)
    reg = registry.Registry("nb", arm="runs", root=str(tmp_path / "ml"))
    reg.log_model("m", local_model[0])
    monkeypatch.setattr(platform, "python_version", lambda: "2.0.0")
    with pytest.warns(UserWarning, match="pickled under Python"):
        assert reg.model("m").predict([1]) == [3]
    _artifacts(tmp_path / "ml")[".pkl"].write_bytes(b"not a pickle")
    with pytest.raises(RuntimeError, match=r"m@v1\.py"):
        reg.model("m")


def test_new_entries_in_a_copied_registry_stay_in_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    registry.Registry("nb", arm="runs", root=str(tmp_path / "ml")).log("eq", "x = 1")
    shutil.copytree(tmp_path / "ml", tmp_path / "moved")
    shutil.rmtree(tmp_path / "ml")
    other = registry.Registry("nb", arm="runs", root=str(tmp_path / "moved"))
    other.log("eq2", "y = 2")
    assert not (tmp_path / "ml").exists() and other.get("eq2").tex == "y = 2"
