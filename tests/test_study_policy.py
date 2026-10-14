"""act(): the value of acting on each decision, scored by evaluate.policy_metrics for the model and each policy
baseline. Each policy acts on its own forecast, and a declared policy with no act() is refused, not skipped."""

import shutil

import pytest

import openlab.study as vs
from openlab.study.errors import StudyError

ACT = '''

def act(pred_row, view, t):
    """The mean forecast of the decision's rows, and how many there were."""
    p = pred_row["pred"] if isinstance(pred_row["pred"], list) else [pred_row["pred"]]
    return {"value": float(np.mean(p)), "rows": len(p)}


class Hundred:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(X.height, 100.0)

    def act(self, pred_row, view, t):
        return act(pred_row, view, t)
'''


def _study(tmp_path, with_act=True):
    s = vs.new(tmp_path / "pol", "temporal")
    (s.folder / "model.py").write_text((s.folder / "model.py").read_text().replace("import polars as pl",
                                                                                     "import numpy as np\nimport polars as pl")
                                       + (ACT if with_act else ACT.split("class Hundred")[0].split("def act")[0]
                                          + "\n\nclass Hundred:\n    def fit(self, X, y):\n        return self\n\n"
                                            "    def predict(self, X):\n        return np.full(X.height, 100.0)\n"))
    toml = s.folder / "study.toml"
    toml.write_text(toml.read_text().replace('baselines = ["persistence", "climatology"]',
                                             'baselines = ["persistence", "model:Hundred"]\npolicy_metrics = ["mean"]\n'
                                             'policy_baselines = ["model:Hundred"]'))
    return s


@pytest.mark.skipif(not shutil.which("bwrap"), reason="the study's code runs under bwrap")
def test_each_policy_acts_on_its_own_forecast(tmp_path):
    out = _study(tmp_path).run()
    pol = out["payload"]["policy"]
    assert pol["decisions"] > 0
    assert pol["baselines"]["model:Hundred"]["mean"]["value"] == pytest.approx(100.0)
    assert pol["metrics"]["mean"]["value"] < 50 and pol["series"]["rows"]["value"] == 4
    # the per-decision series an agent reads: one value per decision, oldest first, no labels or predictions
    from openlab.app import ToolContext, dispatch
    ctx = ToolContext(str(tmp_path))
    env = dispatch(ctx, "study_series", {"folder": "pol", "result": out["id"]})
    assert env["ok"], env
    r = env["result"]
    assert r["decisions"] == pol["decisions"]
    assert set(r["columns"]) == {"act", "act.rows", "act:model:Hundred", "act:model:Hundred.rows"}
    assert r["t"] == sorted(r["t"]) and len(r["t"]) == len(r["act"]) == min(200, r["decisions"])
    assert set(r["act.rows"]) == {4} and set(r["act:model:Hundred"]) == {100.0} and "y" not in r and "pred" not in r
    page = dispatch(ctx, "study_series", {"folder": "pol", "result": out["id"], "columns": ["rows"], "n": 3,
                                          "offset": 1})["result"]
    assert page["columns"] == ["act.rows"] and page["t"] == r["t"][1:4] and page["more"] == r["decisions"] - 4
    assert not dispatch(ctx, "study_series", {"folder": "pol", "result": out["id"], "columns": ["nope"]})["ok"]
    plan = vs.open(tmp_path / "pol").record.find("plan")[0].id
    assert "is a plan" in dispatch(ctx, "study_series", {"folder": "pol", "result": plan})["error"]


@pytest.mark.skipif(not shutil.which("bwrap"), reason="the study's code runs under bwrap")
def test_a_declared_policy_with_no_act_is_refused(tmp_path):
    s = _study(tmp_path, with_act=False)
    (s.folder / "model.py").write_text((s.folder / "model.py").read_text().replace(
        "class Model:", "class Model:\n    def act(self, pred_row, view, t):\n        return 0.0\n"))
    with pytest.raises(StudyError, match="defines no act.*Model has an act method"):
        s.run()
