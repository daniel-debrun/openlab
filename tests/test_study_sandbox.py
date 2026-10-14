"""Guarantee: a study's code runs with no network and cannot read the record, the raw data files or anything
under $HOME outside the study; it can import model.py and write its stage dir."""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from openlab.study import child, sandbox


@pytest.mark.skipif(not shutil.which("bwrap"), reason="bwrap is not installed")
def test_the_sandbox_blocks_network_record_data_and_home(tmp_path):
    study = tmp_path / "study"
    (study / ".openlab").mkdir(parents=True)
    (study / ".openlab" / "record.db").write_text("record")
    (study / "data").mkdir()
    (study / "data" / "secret.csv").write_text("holdout rows")
    (study / "model.py").write_text("X = 1\n")
    cache = Path.home() / ".cache" / "openlab-build"
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache) as home:
        (Path(home) / "secret.txt").write_text("key")
        hidden = [str(study / ".openlab" / "record.db"), str(study / "data" / "secret.csv"), f"{home}/secret.txt"]
        res = sandbox.run({"entry": "openlab.study.child:probe",
                           "args": {"read": [*hidden, str(study / "model.py")], "module": "model"}},
                          study_dir=study, stage_dir=study / ".openlab" / "stage" / "r1", timeout=60)
    assert res.ok and res.sandbox == "bwrap", res.stderr
    out = json.loads(res.output.read_text())
    assert out["connect"] != "ok"
    assert all(out["read"][p] != "ok" for p in hidden), out["read"]
    assert out["read"][str(study / "model.py")] == "ok"
    assert out["write"] == "ok" and out["import"] == "ok"


@pytest.mark.skipif(not shutil.which("bwrap"), reason="bwrap is not installed")
def test_a_server_job_leaves_nothing_for_the_next(tmp_path):
    """The audit's replay server forks a fresh process per job: what one job wrote to /tmp, to the stage dir or into
    a module's globals is gone for the next, and the next job sees only its own staged files."""
    study = tmp_path / "study"
    study.mkdir()
    (study / "model.py").write_text("import os\nSEEN = os.path.exists('data/owned.txt')\n"
                                    "open('/tmp/leak.txt', 'w').write('x')\nopen('../leak.txt', 'w').write('x')\n")
    srv = sandbox.Server(study_dir=study, stage_dir=tmp_path / "stage", timeout=60, preload=["json"])
    try:
        first = srv.clear()
        (first / "data").mkdir()
        (first / "data" / "owned.txt").write_text("step 1's data")
        res = srv.run({"entry": "openlab.study.child:probe", "args": {"module": "model"}}, first)
        assert res.ok and res.sandbox == "bwrap", res.stderr
        assert (tmp_path / "stage" / "leak.txt").is_file()
        second = srv.clear()
        assert not first.exists() and not (tmp_path / "stage" / "leak.txt").exists()
        res = srv.run({"entry": "openlab.study.child:probe",
                       "args": {"read": ["/tmp/leak.txt", str(first / "data" / "owned.txt")]}}, second)
        assert res.ok, res.stderr
        out = json.loads(res.output.read_text())
        assert all(v != "ok" for v in out["read"].values()), out
    finally:
        srv.close()


def test_only_openlab_study_entries_run():
    with pytest.raises(ValueError):
        child._entry("os:system")


def _nested(tmp_path):
    """A study holding a nested study (its own study.toml, data/ and .openlab/), a plain data folder of its own
    code, a nested .git, and a venv that is never walked."""
    study = tmp_path / "study"
    inner = study / "wxmodels" / "temp"
    for d in (study / ".openlab", study / "data", inner / "data", inner / ".openlab", study / "lib" / "data",
              study / "vendor" / ".git", study / ".venv" / "x" / ".git"):
        d.mkdir(parents=True)
    (study / "study.toml").write_text("")
    (inner / "study.toml").write_text("")
    (inner / "data" / "holdout.csv").write_text("holdout rows")
    (inner / ".openlab" / "record.db").write_text("record")
    (study / "lib" / "data" / "table.csv").write_text("the model's own table")
    (study / "model.py").write_text("X = 1\n")
    (study / "linked").symlink_to(inner)
    return study, inner


def test_nested_studies_records_and_git_are_masked_and_a_plain_data_folder_is_not(tmp_path):
    study, inner = _nested(tmp_path)
    got = {p.relative_to(study).as_posix() for p in sandbox.masks(study)}
    assert got == {".openlab", "data", "wxmodels/temp/data", "wxmodels/temp/.openlab", "vendor/.git"}, got


@pytest.mark.skipif(not shutil.which("bwrap"), reason="bwrap is not installed")
def test_a_run_cannot_read_a_nested_studys_data_or_record(tmp_path):
    study, inner = _nested(tmp_path)
    hidden = [str(inner / "data" / "holdout.csv"), str(inner / ".openlab" / "record.db"),
              str(study / "linked" / "data" / "holdout.csv")]
    seen = str(study / "lib" / "data" / "table.csv")
    res = sandbox.run({"entry": "openlab.study.child:probe", "args": {"read": [*hidden, seen]}},
                      study_dir=study, stage_dir=study / ".openlab" / "stage" / "r1", timeout=60)
    assert res.ok and res.sandbox == "bwrap", res.stderr
    out = json.loads(res.output.read_text())
    assert all(out["read"][p] != "ok" for p in hidden), out["read"]
    assert out["read"][seen] == "ok"


def test_a_timeout_is_said_first(tmp_path):
    study = tmp_path / "study"
    study.mkdir()
    res = sandbox.run({"entry": "openlab.study.child:probe", "args": {}}, study_dir=study,
                      stage_dir=tmp_path / "stage", timeout=0.01)
    assert not res.ok and res.timed_out
    assert res.stderr.startswith("timed out after 0.01 s") and "no result.json" not in res.stderr
    assert "timed out after" in res.stderr[-3000:]                    # a reply that shows the tail still says it
