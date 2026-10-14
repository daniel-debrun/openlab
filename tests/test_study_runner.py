"""Guarantees of the study runner: research never stages a row past the research end, the truncation audit flags
look-ahead, static folds and the holdout never stage held-out labels, and a run is a counted, cached, branching
trial. Slow by nature: every run starts sandboxed children (about 45 s for the file)."""

import shutil
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from openlab.study.errors import StudyError

import openlab.study as vs
from openlab.study import vault
from openlab.study import fingerprint, runner, sandbox, splits

pytestmark = pytest.mark.skipif(not shutil.which("bwrap"), reason="the runner's guarantees are tested under bwrap")

LEAKY = '''
from datetime import timedelta

import polars as pl
from sklearn.linear_model import LinearRegression

_STAGE = {}


def features(view, t):
    """Tomorrow's y, read from the staged file behind the view's back and kept in a module-level cache."""
    if "panel" not in _STAGE:
        _STAGE["panel"] = pl.read_parquet("data/panel.parquet")
    nxt = (_STAGE["panel"].filter(pl.col("date") == (t + timedelta(days=1)).date())
           .select("site", pl.col("y").alias("y_next")))
    now = (view.frame("panel").sort("date").group_by("site")
           .agg(pl.col("x").last(), pl.col("y").last().alias("y_now")).sort("site"))
    return now.join(nxt, on="site", how="left").with_columns(pl.col("y_next").fill_null(0.0))


def label(view, t):
    return view.frame("panel").sort("date").group_by("site").agg(pl.col("y").last()).sort("site")


class Model:
    def fit(self, X, y):
        self.m = LinearRegression().fit(X.select("x", "y_now", "y_next").to_numpy(), y)
        return self

    def predict(self, X):
        return self.m.predict(X.select("x", "y_now", "y_next").to_numpy())
'''


def _study(tmp_path: Path, template: str, edit=lambda text: text) -> vs.Study:
    s = vs.new(tmp_path / template, template)
    toml = s.folder / "study.toml"
    toml.write_text(edit(toml.read_text()) + "\n[audit]\nn = 2\n")
    return s


@pytest.fixture
def staged(monkeypatch):
    """(entry, args, {file: frame}) for every child, read from its stage just before it runs."""
    seen, real = [], sandbox.run

    def look(job, stage_dir):
        files = {p.name: pl.read_parquet(p) for p in sorted((Path(stage_dir) / "data").glob("*.parquet"))}
        seen.append((job["entry"].split(":")[1], job["args"], files))

    def spy(job, *, study_dir, stage_dir, **kw):
        look(job, stage_dir)
        return real(job, study_dir=study_dir, stage_dir=stage_dir, **kw)

    def served(self, job, sub):                         # the audit's replays run in a server, one sub dir each
        look(job, sub)
        return real_served(self, job, sub)
    real_served = sandbox.Server.run
    monkeypatch.setattr(sandbox, "run", spy)
    monkeypatch.setattr(sandbox.Server, "run", served)
    return seen


def test_a_temporal_run_is_scored_staged_to_research_cached_and_branched(tmp_path, staged, capfd):   # ~10 s
    s = _study(tmp_path, "temporal")
    first = s.run()
    res, run = first["payload"], s.record.last("run")
    assert all(m["lo"] <= m["value"] <= m["hi"] for m in res["metrics"].values()) and set(res["metrics"]) == {"mae", "rmse"}
    assert set(res["baselines"]) == set(res["paired"]) == {"persistence", "climatology"}
    assert res["paired"]["persistence"]["metric"] == "mae" and res["paired"]["persistence"]["better"] == "model"
    audit = run.payload["audit"]
    assert {k: audit[k] for k in ("n", "checked", "mismatches", "verdict")} == {"n": 2, "checked": 2, "mismatches": [],
                                                                               "verdict": "clean"} and "seed" in audit
    assert s.status()["last_result"]["audit"] == "clean" and s.record.last("model").payload["version"] == 1
    assert {e for e, _, _ in staged} == {"temporal_job", "replay_job"}
    for entry, args, files in staged:                     # research end, or the replayed step's time
        until = date(2024, 12, 31) if entry == "temporal_job" else date.fromisoformat(args["step"][0][:10])
        assert files["panel.parquet"]["date"].max() <= until
    runs = len(s.record.find("run"))
    assert s.run()["id"] == first["id"] and len(s.record.find("run")) == runs          # cached, not a new trial
    model = s.folder / "model.py"
    model.write_text(model.read_text().replace('COLUMNS = ["x", "y_now"]', 'COLUMNS = ["x"]'))
    s.run()
    branch = s.record.last("run")
    assert branch.id != run.id and s.record.links(branch.id, rel="branches_from") == [(branch.id, "branches_from", run.id)]
    assert s.status()["trials"]["used"] == 2
    out, err = capfd.readouterr()
    assert "mlflow" not in (out + err).lower(), err             # no MLflow hint or migration log in run output


def test_the_truncation_audit_flags_features_that_read_past_t(tmp_path):            # ~8 s
    s = _study(tmp_path, "temporal")
    (s.folder / "model.py").write_text(LEAKY)
    s.run()
    audit = s.record.last("run").payload["audit"]
    assert audit["verdict"] == "look-ahead" and len(audit["mismatches"]) == 2, audit
    assert s.status()["last_result"]["audit"] == "look-ahead"
    run = s.record.last("run")                      # a temporal run cannot declare itself separated to dodge the audit
    forged = {**run.payload, "key": "forged", "audit": {"n": 0, "checked": 0, "mismatches": [], "verdict": "separated"}}
    links = [(rel, dst) for _, rel, dst in s.record.links(run.id) if rel != "branches_from"]
    with pytest.raises(StudyError, match="only a static run"):
        s.record.add("run", forged, links=links)


def test_static_folds_and_the_holdout_never_stage_held_out_labels(tmp_path, staged, monkeypatch, person_key):   # ~20 s
    monkeypatch.setenv("OPENLAB_VAULT", str(tmp_path / "vault"))
    s = _study(tmp_path, "static", lambda text: text.replace("k = 5", "k = 3"))
    s.run()
    assert s.record.last("run").payload["env"]["threads"] == 1                # [sandbox] threads, recorded
    df = pl.read_parquet(s.folder / "data" / "table.parquet")
    research, hold = splits.carve(df, s.spec)
    held = set(df["f0"].gather(hold).to_list())
    assert [e for e, _, _ in staged] == ["static_job"] * 4                      # 3 folds, then the final fit
    for _, _, files in staged:
        assert held.isdisjoint(pl.concat([f.select("f0") for f in files.values()])["f0"].to_list())
        assert "y" not in files.get("valid.parquet", pl.DataFrame()).columns
    kept = vault.load(s.folder, s.record.find("plan")[0].id, "holdout")         # the labels live in the vault
    assert kept["rows"] == hold.tolist() and not vault.vault_dir(s.folder).is_relative_to(s.folder)

    staged.clear()
    fp = fingerprint.fingerprint(s)
    from openlab.agents import person
    from openlab.study.personkey import signing
    from conftest import PASSPHRASE
    with signing(person.unlock(PASSPHRASE)):
        look = s.record.add("look", {"n": 1, "reason": "test", "spec_sha": fp["spec_sha"], "code": fp["code"],
                                     "data": fp["data"]}, actor="person", links=[("under", s.record.last("plan").id)])
        result = runner.run(s, segment="holdout", look=look.id, actor="person")
    ((_, _, files),) = staged
    assert "y" not in files["valid.parquet"].columns and set(files["valid.parquet"]["f0"].to_list()) == held
    assert held.isdisjoint(files["train.parquet"]["f0"].to_list())
    assert result.payload["n"] == len(hold) and result.payload["metrics"]["auc"]["value"] > 0.6


def test_an_edit_while_a_run_is_in_flight_reaches_neither_its_children_nor_its_audit(tmp_path, monkeypatch):  # ~10 s
    """Agent A runs; agent B edits model.py while A's children run. A's replays import A's snapshot, so A's audit is
    clean and A records A's code; B's run is its own trial with its own code hash, also clean."""
    s = _study(tmp_path, "temporal")
    model = s.folder / "model.py"
    a_code = fingerprint.code(s.folder)
    real, edited = sandbox.run, []

    def spy(job, *, study_dir, stage_dir, **kw):
        assert Path(study_dir).resolve() != s.folder                    # children import a snapshot, never the folder
        out = real(job, study_dir=study_dir, stage_dir=stage_dir, **kw)
        if not edited:                                                   # B edits after A's main child
            model.write_text(model.read_text().replace('COLUMNS = ["x", "y_now"]', 'COLUMNS = ["x"]'))
            edited.append(1)
        return out
    monkeypatch.setattr(sandbox, "run", spy)
    s.run(note="A")
    a = s.record.last("run")
    assert a.payload["audit"]["verdict"] == "clean", a.payload["audit"]
    assert a.payload["code"] == a_code
    monkeypatch.setattr(sandbox, "run", real)
    s.run(note="B")
    b = s.record.last("run")
    assert b.payload["audit"]["verdict"] == "clean" and b.payload["code"] != a.payload["code"]
    assert b.payload["code"] == fingerprint.code(s.folder) and not list((s.record.dir / "stage").iterdir())


def test_runs_in_flight_hold_their_trials(tmp_path):
    """A research run holds .openlab/runs/<key>.lock while in flight: the budget counts it, so two agents cannot both
    take the last trial, and a run of the same key waits for it instead of running twice."""
    import fcntl
    s = _study(tmp_path, "temporal", lambda text: text.replace("trials = 20", "trials = 1"))
    s.seal()
    lock = runner._runs_dir(s.record) / f"{'a' * 64}.lock"
    with open(lock, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        assert runner.in_flight(s.record) == ["a" * 64]
        with pytest.raises(StudyError, match="taken by runs in flight"):
            runner._budget(s, s.spec, "b" * 64)
    assert runner.in_flight(s.record) == []
    runner._budget(s, s.spec, "b" * 64)
