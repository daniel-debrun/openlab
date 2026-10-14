"""Guarantees about what a trial costs: the run key hashes only the packages a run can import (a tool installed into
a shared environment spends nothing), every run says whether it was cached or which trial it spent and why its key
is new, and a rerun that differs only in its note is free."""

import shutil
from importlib import metadata

import pytest

import openlab.study as vs
from openlab.study import fingerprint


class _Dist:
    def __init__(self, name, version, requires=()):
        self.metadata, self.version, self.requires = {"Name": name}, version, list(requires)

    def read_text(self, f):
        return ""


def test_the_key_hashes_the_packages_the_code_imports_not_every_installed_one(tmp_path, monkeypatch):
    (tmp_path / "model.py").write_text("import numpy\nfrom sklearn.linear_model import LinearRegression\n"
                                       "from helpers import f\nimport os\n")
    (tmp_path / "helpers.py").write_text("def f(): pass\n")
    assert fingerprint.imports(tmp_path) == ["numpy", "os", "sklearn.linear_model"]       # helpers is the study's
    installed = [_Dist("numpy", "2.0"), _Dist("polars", "1.0"), _Dist("scikit-learn", "1.5", ["scipy>=1", "joblib",
                 "pytest; extra == 'tests'"]), _Dist("scipy", "1.1", ["numpy"]), _Dist("joblib", "1.4")]
    monkeypatch.setattr(metadata, "distributions", lambda: list(installed))
    monkeypatch.setattr(fingerprint, "_owners", lambda m, dists: {"sklearn": ["scikit-learn"]}.get(
        m, [m] if m in dists else []))
    before = fingerprint.env(tmp_path)
    assert before["packages"] == {"joblib": "1.4", "numpy": "2.0", "polars": "1.0", "scikit-learn": "1.5",
                                  "scipy": "1.1"}
    installed.append(_Dist("pytest", "9.1"))                # a test tool installed for something else
    assert fingerprint.env(tmp_path)["packages_sha"] == before["packages_sha"]
    installed[2] = _Dist("scikit-learn", "1.6", ["scipy>=1", "joblib"])
    after = fingerprint.env(tmp_path)
    assert after["packages_sha"] != before["packages_sha"]
    assert fingerprint.changes({"env": before}, {"env": after}) == [("env packages", "scikit-learn 1.5→1.6")]


@pytest.mark.skipif(not shutil.which("bwrap"), reason="runs are tested under bwrap")
def test_every_run_says_cached_or_which_trial_and_why(tmp_path, capsys):          # ~15 s
    s = vs.new(tmp_path / "t", "temporal")
    (s.folder / "study.toml").write_text((s.folder / "study.toml").read_text() + "\n[audit]\nn = 2\n")
    first = s.run(note="first")
    assert list(first)[0] == "trial" and first["trial"]["summary"] == "trial 1 of 20: the study's first run"
    again = s.run(note="only the note differs")
    assert again["id"] == first["id"] and again["trial"]["cached"]
    assert again["trial"]["summary"].startswith("cached: no trial spent (key ")
    assert s.status()["trials"]["used"] == 1
    model = s.folder / "model.py"
    model.write_text(model.read_text().replace('COLUMNS = ["x", "y_now"]', 'COLUMNS = ["x"]'))
    variant = s.run(note="x only")["trial"]
    assert variant["k"] == 2 and variant["changed"]["components"] == ["code"]
    assert variant["summary"].startswith("trial 2 of 20: new key because code (model.py ")
    assert variant["summary"].endswith(f"changed since {s.record.find('run')[0].id}")
    from openlab.study.cli import TEXT
    line = TEXT["run"](s.run(note="x only, again"))[0]
    assert line.startswith("cached: no trial spent") and s.status()["trials"]["used"] == 2
    from openlab.study import cli
    assert cli.main(["runs", str(s.folder)]) == 0                  # every trial, its code version, note, cached reruns
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "2 of 20 trials: a result chosen among them is selected from 2 trials"
    assert "trial   1" in out[1] and "code v1" in out[1] and "cached x1" in out[1] and '"first"' in out[1]
    assert "trial   2" in out[2] and "code v2" in out[2] and "changed model.py" in out[2] and "mae " in out[2]
