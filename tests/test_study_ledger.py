"""Guarantees that budgets belong to the question and the data, not the folder: a copied record is refused at run,
naming the original; a second study on the same data and target is refused until it continues the first (one
shared trial and holdout ledger) or a person declares a fresh question."""

import shutil

import pytest

import openlab.study as vs
from openlab.study import ledger
from openlab.study.errors import StudyError

pytestmark = pytest.mark.skipif(not shutil.which("bwrap"), reason="runs are tested under bwrap")


def _study(folder):
    s = vs.new(folder, "temporal")
    (s.folder / "study.toml").write_text((s.folder / "study.toml").read_text() + "\n[audit]\nn = 2\n")
    return s


def test_a_copied_record_is_refused_and_the_original_runs(tmp_path):            # ~8 s
    demo = _study(tmp_path / "demo")
    first = demo.run()
    shutil.copytree(demo.folder, tmp_path / "copy")
    copy = vs.open(tmp_path / "copy")
    with pytest.raises(StudyError, match="holds a copy of the record of") as e:
        copy.run()
    assert str(demo.folder) in e.value.error and not copy.record.find("run")[1:]
    assert demo.run()["id"] == first["id"]                     # the original is not affected (cached)
    moved = tmp_path / "moved"                                  # a moved study (the original gone) runs where it is
    shutil.move(str(demo.folder), moved)
    assert vs.open(moved).run()["id"] == first["id"]
    assert (moved / ".openlab" / "origin").read_text() == str(moved.resolve())


def test_a_second_study_of_the_same_data_shares_the_ledger_or_is_refused(tmp_path, person_key):      # ~15 s
    demo = _study(tmp_path / "demo")
    demo.run()
    demo2 = _study(tmp_path / "demo2")
    with pytest.raises(StudyError, match="asks the question of demo") as e:
        demo2.run()
    assert 'continues = "../demo"' in e.value.fix and not demo2.record.find("run")
    assert not demo2.record.find("plan")                        # a refused first run does not seal study.toml
    toml = demo2.folder / "study.toml"
    text = toml.read_text()
    toml.write_text(text.replace('mode = "temporal"', 'mode = "temporal"\nfresh = "another question, stated"'))
    with pytest.raises(StudyError, match="asks the question of demo"):      # an agent's fresh is not a person's
        demo2.run()
    with pytest.raises(StudyError, match="not sealed"):
        demo2.amend("an agent cannot declare a fresh budget")
    shutil.copytree(demo.folder, tmp_path / "democopy")         # a refused copy is not a study of its own
    with pytest.raises(StudyError, match="holds a copy"):
        vs.open(tmp_path / "democopy").run()
    toml.write_text(text.replace('mode = "temporal"', 'mode = "temporal"\ncontinues = "../demo"'))
    model = demo2.folder / "model.py"
    model.write_text(model.read_text().replace('COLUMNS = ["x", "y_now"]', 'COLUMNS = ["x"]'))
    r = demo2.run()
    assert r["trial"]["summary"].startswith("trial 2 of 20 (1 in demo)")
    st = demo2.status()
    assert st["trials"]["used"] == 2 and st["trials"]["shared"] == {"demo": 1} and st["looks"]["shared"] == {"demo": 0}
    assert demo.status()["trials"]["shared"] == {"demo2": 1}                     # and demo counts demo2's
    from openlab.study.cli import TEXT
    assert any(x.startswith("trials 2 of 20 (1 in demo), holdout looks 0 of 1 (0 in demo)") for x in TEXT["status"](st))
    assert ledger.shared(demo.folder, demo.record)["trials"] == 2

    demo3 = _study(tmp_path / "demo3")                          # a person declares a fresh question
    with pytest.raises(StudyError, match="asks the question of demo2? ") as e:
        demo3.run()
    assert "democopy" not in e.value.error
    toml3 = demo3.folder / "study.toml"
    toml3.write_text(toml3.read_text().replace('mode = "temporal"', 'mode = "temporal"\nfresh = "x at a new site"'))
    with person_key.act():
        demo3.amend("a new question on the same panel", person=True)
    assert demo3.run()["trial"]["summary"] == "trial 1 of 20: the study's first run"
