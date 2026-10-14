"""Per-branch working copies: each experiment branch of a proposal runs its own copy of model.py
(branches/<id>/), so parallel experimenters never overwrite each other's code; the run records its branch, its key
holds its code's hash, `study runs` shows the branch, a paper binding is stale against the run's own branch's
code, and promoting a branch's code to the study's model.py is a recorded act."""

import shutil
import threading

import pytest

import openlab.study as vs
from openlab.study import branches, fingerprint
from openlab.study.errors import StudyError
from openlab.study.paper import bindings

HELPER = "SCALE = 1.0\n"


def _study(folder):
    s = vs.new(folder, "static")
    model = s.folder / "model.py"
    model.write_text(model.read_text().replace("from sklearn.linear_model import LogisticRegression",
                                               "from sklearn.linear_model import LogisticRegression\n"
                                               "from helper import SCALE"))
    (s.folder / "helper.py").write_text(HELPER)
    (s.folder / "unused.py").write_text("X = 1\n")
    return s


def test_a_branch_copy_holds_model_py_and_its_local_modules(tmp_path):
    s = _study(tmp_path / "s")
    main = fingerprint.code(s.folder)
    made = branches.make(s, "b1")
    assert made["made"] and made["files"] == ["helper.py", "model.py"]
    assert (s.folder / "branches" / "b1" / "helper.py").read_text() == HELPER
    assert not (s.folder / "branches" / "b1" / "unused.py").exists()
    (s.folder / "branches" / "b1" / "model.py").write_text("# b1's edit\n")
    assert fingerprint.code(s.folder) == main                       # a branch's edit is not the study's code
    assert not branches.make(s, "b1")["made"]                        # an existing copy is the branch's work: kept
    assert (s.folder / "branches" / "b1" / "model.py").read_text() == "# b1's edit\n"
    assert branches.resolve(s, "branches/b1") == (s.folder / "branches" / "b1", "b1")
    assert branches.resolve(s, "b1")[1] == "b1" and branches.resolve(s, "") == (None, "")
    with pytest.raises(StudyError, match="no branch code at branches/b9") as e:
        branches.resolve(s, "branches/b9")
    assert "branches: b1" in e.value.meaning
    with pytest.raises(StudyError, match="outside the study"):
        branches.resolve(s, str(tmp_path))


def test_promote_copies_a_branch_over_the_study_code_and_is_recorded(tmp_path):
    s = _study(tmp_path / "s")
    branches.make(s, "b2")
    (s.folder / "branches" / "b2" / "helper.py").write_text("SCALE = 2.0\n")
    out = s.promote("b2")
    assert out["changed"] == ["helper.py"] and (s.folder / "helper.py").read_text() == "SCALE = 2.0\n"
    act = s.record.last("action", verb="promote")
    assert act.payload["args"]["branch"] == "b2" and "helper.py" in act.payload["args"]["files"]
    with pytest.raises(StudyError, match="no working copy"):
        s.promote("b7")


@pytest.mark.skipif(not shutil.which("bwrap"), reason="runs are tested under bwrap")
def test_concurrent_branch_runs_keep_their_own_code(tmp_path):                       # ~15 s
    s = _study(tmp_path / "s")
    s.materialise()
    s.seal()
    main_model = (s.folder / "model.py").read_text()
    for b, c in (("b1", "max_iter=1000"), ("b2", "max_iter=1000, C=0.1")):
        branches.make(s, b)
        m = s.folder / "branches" / b / "model.py"
        m.write_text(m.read_text().replace("max_iter=1000", c))
    got, errors = {}, []

    def run(b):
        try:
            got[b] = vs.open(s.folder).run(note=f"{b}'s variant", code=f"branches/{b}")
        except Exception as e:                                       # noqa: BLE001
            errors.append(e)
    threads = [threading.Thread(target=run, args=(b,)) for b in ("b1", "b2")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert got["b1"]["branch"] == "b1" and got["b2"]["branch"] == "b2"
    rec = s.record
    runs = {r.payload["branch"]: r for r in rec.find("run")}
    assert set(runs) == {"b1", "b2"} and runs["b1"].payload["code"] != runs["b2"].payload["code"]
    assert runs["b1"].payload["key"] != runs["b2"].payload["key"]
    assert (s.folder / "model.py").read_text() == main_model                 # nobody touched the study's model.py
    rows = s.runs()["runs"]
    assert {r["branch"] for r in rows} == {"b1", "b2"}
    from openlab.study.cli import TEXT
    text = "\n".join(TEXT["runs"](s.runs()))
    assert "  b1      code v" in text and "  b2      code v" in text
    now = bindings.current(rec, s.folder)
    res = {b: rec.get(rec.links(r.id, rel="of", incoming=True)[0][0]) for b, r in runs.items()}
    assert bindings.stale(rec, res["b1"], now) == "" and bindings.stale(rec, res["b2"], now) == ""
    (s.folder / "branches" / "b1" / "model.py").write_text(main_model + "\n# b1 moved on\n")
    now = bindings.current(rec, s.folder)
    assert "code (model.py)" in bindings.stale(rec, res["b1"], now) and bindings.stale(rec, res["b2"], now) == ""
    again = vs.open(s.folder).run(code="branches/b2")                 # same code: cached, no trial
    assert again["trial"]["cached"]
