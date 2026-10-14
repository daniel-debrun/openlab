"""Guarantees of the W&B mirror (openlab.study.mirror): it is a view, never the record. It mirrors only when wandb is
installed and the workspace allows it (by default when W&B has a key here), and never a study that opts out. A
recorded run is pushed by a detached process, so the mirror never waits or fails a run. A push sends the run's
metrics and its key, lineage and shas, never data, as an offline W&B run whose id is the run's key."""

import shutil
import sys
import types

import pytest

from openlab import study as vs
from openlab.study import mirror


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENLAB_MIRROR", raising=False)              # the suite turns it off everywhere else
    (tmp_path / "openlab.toml").write_text("[agents]\n")
    monkeypatch.setattr(mirror, "_installed", lambda: True)
    monkeypatch.setattr(mirror, "_has_key", lambda: False)
    return tmp_path


def test_when_it_mirrors(ws, monkeypatch):
    folder = ws / "s"
    folder.mkdir()
    (folder / "study.toml").write_text('[study]\nname = "s"\n')
    assert not mirror.settings(folder)["on"] and "no W&B key" in mirror.settings(folder)["why"]
    monkeypatch.setattr(mirror, "_has_key", lambda: True)
    st = mirror.settings(folder)
    assert st["on"] and st["project"] == "openlab" and st["sync"]
    (ws / "openlab.toml").write_text('[agents]\n\n[mirror]\nwandb = false\n')
    assert not mirror.settings(folder)["on"]
    (ws / "openlab.toml").write_text('[mirror]\nproject = "lab"\nentity = "team"\n')
    assert mirror.settings(folder)["project"] == "lab" and mirror.settings(folder)["entity"] == "team"
    (folder / "study.toml").write_text('[study]\nname = "s"\n\n[mirror]\nwandb = false\n')
    assert mirror.settings(folder)["why"].startswith("the study opts out")
    monkeypatch.setattr(mirror, "_installed", lambda: False)
    (folder / "study.toml").write_text('[study]\nname = "s"\n')
    assert "not installed" in mirror.settings(folder)["why"]
    monkeypatch.setenv("OPENLAB_MIRROR", "off")
    assert mirror.settings(folder)["why"] == "OPENLAB_MIRROR=off"


def test_a_mirror_that_breaks_never_breaks_a_run(ws, monkeypatch):
    monkeypatch.setattr(mirror, "settings", lambda folder: 1 / 0)
    mirror.after_run(types.SimpleNamespace(folder=ws), types.SimpleNamespace(id="run:x"))


class FakeRun:
    def __init__(self, kw):
        self.kw, self.summary, self.dir, self.finished = kw, {}, str(kw["dir"]) + "/offline-run/files", False

    def finish(self):
        self.finished = True


@pytest.mark.skipif(not shutil.which("bwrap"), reason="runs are tested under bwrap")
def test_a_recorded_run_is_mirrored_with_its_metrics_and_lineage_never_its_data(ws, monkeypatch):     # ~10 s
    monkeypatch.setattr(mirror, "_has_key", lambda: True)
    started = []
    monkeypatch.setattr(mirror, "after_run", lambda study, run: started.append(run.id))    # what the runner calls
    s = vs.new(ws / "demo", "temporal")
    (s.folder / "study.toml").write_text((s.folder / "study.toml").read_text() + "\n[audit]\nn = 1\n")
    out = s.run(note="first", op="draft")
    run_id = s.record.links(out["id"], rel="of")[0][2]
    assert started == [run_id]
    runs = []
    fake = types.ModuleType("wandb")
    fake.init = lambda **kw: runs.append(FakeRun(kw)) or runs[-1]
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setattr(mirror, "_has_key", lambda: False)                  # offline only: nothing to sync to
    monkeypatch.setattr(mirror, "settings", lambda folder: {"on": True, "why": "", "project": "openlab", "entity": "",
                                                            "sync": True})
    res = mirror.push(str(s.folder), run_id)
    kw, run = runs[0].kw, s.record.get(run_id)
    assert kw["mode"] == "offline" and kw["id"] == "r" + run.payload["key"][:40] and kw["group"] == s.spec.name
    assert kw["job_type"] == "draft" and kw["config"]["key"] == run.payload["key"] and kw["config"]["parent"] is None
    assert runs[0].finished and "mae" in runs[0].summary and "mae.lo" in runs[0].summary
    text = repr(kw) + repr(runs[0].summary)
    assert "predictions" not in text and "y_now" not in text                # shas and metrics, no rows
    assert res["synced"] is False and [o["run"] for o in mirror.pending(s.folder)] == [run_id]
    fake.init = lambda **kw: 1 / 0                                          # W&B breaks: logged, still pending
    bad = mirror.push(str(s.folder), run_id)
    assert "ZeroDivisionError" in bad["error"] and mirror.pending(s.folder)[0]["error"] == bad["error"]
