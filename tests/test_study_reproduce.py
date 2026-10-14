"""Guarantees of reproduction: a recorded run executed again from the record alone (its code blobs, its data, the
sealed spec at the run) matches it exactly, whatever the study folder says now; a dataset changed since the run
refuses; a match that disagrees with the predictions blobs cannot be typed; a claim is at level reproduced only once
the results it rests on have a matched reproduction; a bundle reproduces in a temp dir from itself alone, and a bundle
holding more than its manifest is refused. One template run for the file (about 15 s), then reproductions (about
4 s each)."""

import io
import json
import shutil
import tarfile

import polars as pl
import pytest

import openlab.study as vs
from openlab.study import cli, fingerprint, reproduce, schema
from openlab.study.errors import StudyError

pytestmark = pytest.mark.skipif(not shutil.which("bwrap"), reason="reproductions run sandboxed, as runs do")


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    s = vs.new(tmp_path_factory.mktemp("reproduce") / "study", "temporal")
    toml = s.folder / "study.toml"
    toml.write_text(toml.read_text() + "\n[audit]\nn = 2\n")
    s.run()
    return s.folder


@pytest.fixture
def study(base, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENLAB_SIGNING_KEY", str(tmp_path / "signing.key"))
    shutil.copytree(base, tmp_path / "study")
    return vs.open(tmp_path / "study")


def _shifted(rec, run, by: float) -> str:
    """A predictions blob: the run's, with the model's predictions moved by `by`."""
    buf = io.BytesIO()
    P = pl.read_parquet(io.BytesIO(rec.blob(run.payload["predictions"])))
    P.with_columns(pl.col("pred") + by).write_parquet(buf)
    return rec.put_blob(buf.getvalue())


def _write(rec, run, predictions: str, **typed):
    """A reproduction written straight to the record: the comparison as computed, unless `typed` overrides it."""
    env = typed.pop("env", fingerprint.env())
    fields = {**reproduce.compare(rec, run, predictions), "env_diff": reproduce.env_diff(run.payload["env"], env),
              **typed}
    return rec.add(schema.Reproduction(run=run.id, predictions=predictions, env=env, sandbox="bwrap", seconds=1.0,
                                       **fields), links=[("of", run.id)])


def test_a_temporal_run_reproduces_exactly_from_the_record_alone(study):         # ~4 s
    rec, run = study.record, study.record.last("run")
    (study.folder / "model.py").write_text("raise RuntimeError('not the code the run used')\n")
    toml = study.folder / "study.toml"
    toml.write_text(toml.read_text().replace("trials = 20", "trials = 3"))        # an unamended edit
    (r,) = study.reproduce()["reproductions"]
    assert r["run"] == run.id and r["matched"] and r["differs"] == "" and r["max_abs_diff"] == 0.0, r
    assert set(r["metrics"]) == {"mae", "rmse"} and all(m["diff"] == 0.0 for m in r["metrics"].values())
    assert r["env_diff"] == {} and r["sandbox"] == "bwrap"
    node = rec.get(r["id"])
    assert rec.links(node.id, rel="of") == [(node.id, "of", run.id)] and node.payload["predictions"] != ""
    act = rec.last("action")
    assert act.payload["verb"] == "reproduce" and rec.links(act.id, rel="did") == [(act.id, "did", node.id)]
    assert study.reproductions()["reproduced"] == 1


def test_a_dataset_changed_since_the_run_refuses(study):
    path = study.record.dir / "data" / "panel.parquet"
    pl.read_parquet(path).with_columns(pl.col("y") * 1.001).write_parquet(path)
    with pytest.raises(StudyError, match="not the data the run used"):
        study.reproduce()
    assert not study.record.find("reproduction") and not study.record.last("action").payload["outcome"]["ok"]


def test_a_typed_match_that_disagrees_with_the_blobs_is_refused(study):
    rec, run = study.record, study.record.last("run")
    other = _shifted(rec, run, 0.5)
    honest = reproduce.compare(rec, run, other)
    assert not honest["matched"] and honest["max_abs_diff"] == pytest.approx(0.5)
    assert "pred differs" in honest["differs"]
    assert honest["metrics"]["mae"]["diff"] != 0
    with pytest.raises(StudyError, match="never typed"):
        _write(rec, run, other, matched=True, differs="", max_abs_diff=0.0)
    with pytest.raises(StudyError, match="never typed"):
        _write(rec, run, run.payload["predictions"], matched=False)
    with pytest.raises(StudyError, match="env_diff"):
        _write(rec, run, run.payload["predictions"], env={**fingerprint.env(), "python": "2.7.18"}, env_diff={})
    assert not _write(rec, run, other).payload["matched"]                  # written as computed: it did not match


def test_a_claims_level_is_reproduced_only_after_a_matched_reproduction(study):   # ~4 s
    rec = study.record
    run, res = rec.last("run"), rec.last("result")
    c = study.claim("The model's mean absolute error is on record.", "observed", [res.id])
    assert c["payload"]["level"] == "recorded"
    with pytest.raises(StudyError, match="never typed"):
        rec.add("claim", {**c["payload"], "level": "recorded+reproduced"}, links=[("rests_on", res.id)])
    _write(rec, run, _shifted(rec, run, 0.5))                                 # a reproduction that did not match
    assert [x["level"] for x in study.claims()] == ["recorded"]
    study.reproduce()
    (row,) = study.claims()
    assert row["level"] == "recorded+reproduced" and row["level_written"] == "recorded"
    later = study.claim("The model's error is reproduced.", "inferred", [c["id"]])
    assert later["payload"]["level"] == "recorded+reproduced" and \
        schema.current_level(rec, rec.get(later["id"])) == "recorded+reproduced"


def test_a_bundle_reproduces_in_a_temp_dir_from_itself_alone(study, tmp_path, capsys):   # ~8 s
    tar = tmp_path / "out" / "study.tar"
    b = study.bundle(tar)
    with tarfile.open(tar) as t:
        names = set(t.getnames())
    assert {".openlab/record.db", ".openlab/data/panel.parquet", "study.toml", "model.py", "recipes/make_data.py",
            reproduce.MANIFEST, reproduce.SIG} <= names and "data/panel.parquet" not in names
    run = study.record.last("run").id
    shutil.rmtree(study.folder)                                                # the bundle is all there is
    out = reproduce.from_bundle(tar)
    assert out["head"] == b["head"] and out["key_id"] == b["key_id"]
    assert [(r["run"], r["matched"]) for r in out["reproductions"]] == [(run, True)]
    assert cli.main(["reproduce", "--bundle", str(tar), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["reproductions"][0]["matched"]
    extra = tmp_path / "extra.py"
    extra.write_text("x = 1\n")
    with tarfile.open(tar, "a") as t:
        t.add(extra, arcname="extra.py")
    with pytest.raises(StudyError, match="does not hold what its manifest lists"):
        reproduce.from_bundle(tar)
