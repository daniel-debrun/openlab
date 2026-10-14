"""Guarantees of holdout labels kept by the person (openlab.study.custody): the holdout is the rows whose target is
empty in the data, research never sees them, and the box records a holdout run's predictions without a result; the
person's machine scores them against its own labels with openlab's metric code and writes the result as a signed
person act; the box refuses a score that skips rows or uses other labels than the earlier looks, and a result for
such a run written by anyone but the person."""

import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from conftest import PASSPHRASE
from openlab import study as vs
from openlab.agents import person, remote
from openlab.study import custody, splits
from openlab.study.errors import StudyError

ROOT = str(Path(__file__).resolve().parents[1])

RECIPE = '''from pathlib import Path
import numpy as np
import polars as pl
rng = np.random.default_rng(7)
n = 1000
X = rng.normal(size=(n, 5))
p = 1 / (1 + np.exp(-(1.2 * X[:, 0] - 0.8 * X[:, 1])))
y = (rng.random(n) < p).astype(float)
df = pl.DataFrame({"id": [f"r{i}" for i in range(n)], **{f"f{i}": X[:, i] for i in range(5)}, "y": y})
held = df["id"].is_in([f"r{i}" for i in range(800, n)])
out = Path(__file__).resolve().parent.parent / "data" / "table.parquet"
out.parent.mkdir(exist_ok=True)
df.with_columns(y=pl.when(held).then(None).otherwise(pl.col("y"))).write_parquet(out)
'''


def test_the_holdout_is_the_rows_without_a_target():
    df = pl.DataFrame({"id": ["a", "b", "c", "d"], "y": [1.0, None, float("nan"), 0.0]})
    assert custody.unlabelled(df, "y").tolist() == [1, 2]
    assert custody.labels_sha(["b", "a"], [1, 0]) == custody.labels_sha(["a", "b"], [0, 1])


def person_study(tmp_path) -> "vs.Study":
    s = vs.new(tmp_path / "w" / "held", "static")
    (s.folder / "recipes" / "make_data.py").write_text(RECIPE)
    model = s.folder / "model.py"                                   # the id names a row; it is not a feature
    model.write_text(model.read_text().replace('if not c.startswith("_")', 'if not c.startswith("_") and c != "id"'))
    toml = s.folder / "study.toml"
    text = toml.read_text().replace("fraction = 0.2                  # the holdout, kept outside the study until a "
                                    "person looks\n", 'labels = "person"\nlabels_key = "id"\n')
    assert 'labels = "person"' in text
    toml.write_text(text)
    s.materialise()
    return s


def test_research_never_sees_the_unlabelled_rows(tmp_path):
    s = person_study(tmp_path)
    df = pl.read_parquet(s.folder / "data" / "table.parquet")
    research, hold = splits.carve(df, s.spec)
    assert len(hold) == 200 and set(hold.tolist()) == set(range(800, 1000)) and len(research) == 800


@pytest.mark.parametrize("edit, error", [
    (lambda t: t.replace('labels_key = "id"\n', ""), "needs split.labels_key"),
    (lambda t: t.replace('labels = "person"\n', 'labels = "person"\nfraction = 0.2\n'), "takes no split.fraction"),
])
def test_a_person_labelled_study_must_say_how(tmp_path, edit, error):
    s = person_study(tmp_path)
    toml = s.folder / "study.toml"
    toml.write_text(edit(toml.read_text()))
    with pytest.raises(StudyError, match=error):
        s.run(note="first")


def box_transport(home: Path) -> list:
    return ["env", f"HOME={home}", f"PYTHONPATH={ROOT}", "OPENLAB_MODULE_CACHE=off", "sh", "-c"]


@pytest.mark.skipif(not shutil.which("bwrap"), reason="runs are tested under bwrap")
def test_the_box_records_predictions_and_the_persons_machine_scores_and_signs_them(tmp_path, person_key, capsys):
    s = person_study(tmp_path)
    root, home = s.folder.parent, tmp_path / "boxhome"
    home.mkdir()
    (root / "openlab.toml").write_text("[agents]\n")
    unlock = lambda summary, details: person.unlock(PASSPHRASE, summary)                     # noqa: E731
    act = lambda *argv: remote.act_on("box", ["-m", "openlab", *argv], openlab=sys.executable, cd=str(root),  # noqa: E731
                                      transport=box_transport(home), unlock=unlock)
    assert act("person", "pin") == 0
    s.run(note="research")
    assert act("study", "holdout", "held", "--reason", "final") == 0
    s = vs.open(s.folder)
    run = s.record.find("run", segment="holdout")[0]
    assert not s.record.links(run.id, rel="of", incoming=True)                # predictions, no result on the box
    exported = custody.export(s)
    assert exported["run"] == run.id and exported["rows"] == 200 and "y" not in exported["predictions"][:0]
    rng = np.random.default_rng(7)                                           # the person's labels, as the recipe drew
    X = rng.normal(size=(1000, 5))
    p = 1 / (1 + np.exp(-(1.2 * X[:, 0] - 0.8 * X[:, 1])))
    y = (rng.random(1000) < p).astype(float)
    labels = pl.DataFrame({"_key": [f"r{i}" for i in range(800, 1000)], "y": y[800:]})
    body = custody.score(exported, labels)
    assert body["n"] == 200 and body["scored"]["rows"] == 200 and 0.5 < body["metrics"]["auc"]["value"] <= 1
    with pytest.raises(StudyError, match="covers 199 rows"):
        custody.record_score(s, run.id, {**body, "n": 199})
    # `openlab study score HOST FOLDER --labels FILE`, on the person's machine: ssh replaced by the stand-in box
    import json
    import subprocess
    from openlab.study import cli
    labels_file = tmp_path / "labels.csv"
    labels.rename({"_key": "id"}).write_csv(labels_file)
    real_run, real_act = subprocess.run, remote.act_on
    monkey = pytest.MonkeyPatch()
    monkey.setattr(subprocess, "run", lambda cmd, **kw: real_run(
        [*box_transport(home), f"cd {root} && {sys.executable} -m openlab " + cmd[-1].split(" ", 1)[1]], **kw)
        if cmd[:2] == ["ssh", "-T"] else real_run(cmd, **kw))
    monkey.setattr(remote, "act_on", lambda host, args, openlab, cd: real_act(
        "box", ["-m", "openlab", *args], openlab=sys.executable, cd=str(root), transport=box_transport(home),
        unlock=unlock))
    try:
        assert cli.main(["score", "box", "held", "--labels", str(labels_file), "--json"]) == 0
    finally:
        monkey.undo()
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["run"] == run.id and out["n"] == 200 and out["metrics"]["auc"] == body["metrics"]["auc"]
    s = vs.open(s.folder)
    result = s.record.links(run.id, rel="of", incoming=True)
    res = s.record.get(result[0][0])
    assert res.actor == "person" and res.payload["sig"]["key"] == person.info()["fingerprint"]
    assert res.payload["scored"]["labels_sha"] == body["scored"]["labels_sha"] and s.verify()["ok"]
    assert s.record.find("action", verb="holdout_score")
