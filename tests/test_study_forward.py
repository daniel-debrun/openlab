"""Guarantees of forward runs: a forecast is a signed file whose signature verifies plus a recorded node, one per
decision and model; an outcome waits for its label and is written once; a model whose run read past t never
forecasts; a schedule touches only its own crontab line (a fake crontab, never the real one). Slow: research runs
and sandboxed children (about 40 s for the file)."""

import shutil
from datetime import datetime

import pytest

import openlab.study as vs
from openlab.study.signing import verify_file
from openlab.study import schema
from openlab.study.errors import StudyError

pytestmark = pytest.mark.skipif(not shutil.which("bwrap"), reason="the runner's guarantees are tested under bwrap")

EXTRA = """
[forecast]
gate = ["count(*) = 4", "not_null(pred)", "max(_t) < TIMESTAMP '2026-01-01 00:00:00'"]
expires = "6h"

[schedule]
cron = "7 6 * * *"

[audit]
n = 2
"""

LEAKY = '''
from datetime import timedelta

import polars as pl
from sklearn.linear_model import LinearRegression


def features(view, t):
    """Tomorrow's y, read from the staged file behind the view's back."""
    nxt = pl.read_parquet("data/panel.parquet").filter(pl.col("date") == (t + timedelta(days=1)).date())
    now = view.frame("panel").group_by("site").agg(pl.col("x").last()).sort("site")
    return (now.join(nxt.select("site", pl.col("y").alias("y_next")), on="site", how="left")
            .with_columns(pl.col("y_next").fill_null(0.0)))


def label(view, t):
    return view.frame("panel").group_by("site").agg(pl.col("y").last()).sort("site")


class Model:
    def fit(self, X, y):
        self.m = LinearRegression().fit(X.select("x", "y_next").to_numpy(), y)
        return self

    def predict(self, X):
        return self.m.predict(X.select("x", "y_next").to_numpy())
'''


def _study(tmp_path) -> vs.Study:
    """The temporal template, its holdout ending 2025-05-31 so June 2025 holds forward decisions with known labels."""
    s = vs.new(tmp_path / "temporal", "temporal")
    toml = s.folder / "study.toml"
    toml.write_text(toml.read_text().replace("holdout = [2025-01-01, 2025-06-30]", "holdout = [2025-01-01, 2025-05-31]")
                    + EXTRA)
    return s


def _outcome(s: vs.Study, f: dict):
    """An outcome written straight to the record for forecast `f`."""
    ds = s.record.last("dataset", name="panel")
    return s.record.add(schema.Outcome(forecast=f["id"], label_at=f["payload"]["label_at"], metrics={}, n=0,
                                       data={"panel": ds.payload["sha256"]}, labels=""),
                        actor="openlab", links=[("of", f["id"]), ("uses", ds.id)])


def test_forecasts_are_signed_once_per_decision_and_outcomes_wait_for_their_label(tmp_path, monkeypatch):  # ~25 s
    monkeypatch.setenv("OPENLAB_SIGNING_KEY", str(tmp_path / "signing.key"))
    monkeypatch.setenv("OPENLAB_ALERT_DRY_RUN", "1")
    s = _study(tmp_path)
    s.run()
    f = s.forecast(now=datetime(2025, 6, 10, 12))
    p = f["payload"]
    assert p["t"] == "2025-06-10T00:00:00" and p["rows"] == 4 and p["gate"]["passed"] and not p["prospective"]
    assert s.record.links(f["id"], rel="by")[0][2] == s.record.last("model").id
    v = verify_file(s.folder / p["signed"])
    assert v["ok"] and v["study"] == s.spec.name and v["t"] == p["t"] and v["gate"]["passed"], v
    assert s.record.blob(p["sha256"]) == (s.folder / p["file"]).read_bytes()
    with pytest.raises(StudyError, match="already forecast decision"):
        s.forecast(now=datetime(2025, 6, 10, 18))

    today = s.forecast()                               # label tomorrow; the gate's date check fails it
    assert today["payload"]["prospective"] and not today["payload"]["gate"]["passed"]
    assert verify_file(s.folder / today["payload"]["signed"])["ok"]          # signed, and a consumer sees FAILED
    assert (s.folder / ".openlab" / "outbox.jsonl").is_file()                 # and the failure was delivered
    with pytest.raises(StudyError, match="is not known until"):
        _outcome(s, today)

    filled = s.fill()["outcomes"]
    assert [o["payload"]["forecast"] for o in filled] == [f["id"]] and filled[0]["payload"]["n"] == 4
    assert set(filled[0]["payload"]["metrics"]) == {"mae", "rmse"}
    assert all(m["value"] >= 0 for m in filled[0]["payload"]["metrics"].values())
    assert s.fill()["outcomes"] == []
    with pytest.raises(StudyError, match="already has an outcome"):
        _outcome(s, f)


def test_a_model_whose_run_read_past_t_never_forecasts(tmp_path):                           # ~10 s
    s = _study(tmp_path)
    (s.folder / "model.py").write_text(LEAKY)
    s.run()
    assert s.record.last("run").payload["audit"]["verdict"] == "look-ahead"
    with pytest.raises(StudyError, match="truncation audit is look-ahead"):
        s.forecast(now=datetime(2025, 6, 10))
    assert not s.record.find("forecast") and not (s.folder / "forecasts").exists()


def test_a_schedule_touches_only_its_own_crontab_line(tmp_path, monkeypatch):
    fake, tab = tmp_path / "crontab", tmp_path / "crontab.txt"
    fake.write_text(f'#!/bin/sh\nif [ "$1" = "-l" ]; then cat "{tab}" 2>/dev/null || exit 1; else cat > "{tab}"; fi\n')
    fake.chmod(0o755)
    monkeypatch.setenv("OPENLAB_CRONTAB", str(fake))
    tab.write_text("0 1 * * * echo owned\n")
    s = _study(tmp_path)
    assert s.schedule()["state"] == "off"
    on = s.schedule("on")
    owned, line = tab.read_text().splitlines()
    assert owned == "0 1 * * * echo owned" and on["state"] == "on" and on["next_run"]
    assert line.startswith("7 6 * * * ") and line.endswith(f"# openlab:{s.folder.parent}::{s.folder.name}")
    assert "study forecast" in line and "study fill" in line and "flock -n" in line
    s.schedule("off")
    assert tab.read_text().splitlines() == ["0 1 * * * echo owned"] and s.schedule()["state"] == "off"
