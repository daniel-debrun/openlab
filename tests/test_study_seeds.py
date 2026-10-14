"""One run per model seed (a paper that averages over seeds): the seed is the model's, part of the run's key and its
record, while split.seed keeps the folds and the holdout; a reproduction runs it with the same seed."""

import shutil

import pytest

import openlab.study as vs

RANDOM = '''import numpy as np
from sklearn.ensemble import RandomForestRegressor


class Model:
    def fit(self, X, y):
        self.m = RandomForestRegressor(n_estimators=5, random_state=np.random.randint(1 << 30)).fit(
            X.select("f0", "f1").to_numpy(), y)
        return self

    def predict(self, X):
        return self.m.predict(X.select("f0", "f1").to_numpy())
'''


@pytest.mark.skipif(not shutil.which("bwrap"), reason="the study's code runs under bwrap")
def test_a_run_per_seed_counts_a_trial_each_and_reproduces(tmp_path):
    s = vs.new(tmp_path / "seeds", "static")
    (s.folder / "model.py").write_text(RANDOM)
    a = s.run(seed=1)
    b = s.run(seed=2)
    assert a["trial"]["key"] != b["trial"]["key"] and not b["trial"]["cached"]
    assert s.run(seed=1)["trial"]["cached"]
    ra, rb = (s.record.get(s.record.links(x["id"], rel="of")[0][2]) for x in (a, b))
    assert (ra.payload["seed"], rb.payload["seed"]) == (1, 2)
    va, vb = (x["payload"]["metrics"] for x in (a, b))
    assert va != vb                                     # the model's randomness follows the seed
    (r,) = [x for x in s.reproduce(ra.id)["reproductions"]]
    assert r["matched"], r


@pytest.mark.skipif(not shutil.which("bwrap"), reason="the study's code runs under bwrap")
def test_an_in_sample_run_says_so(tmp_path):
    s = vs.new(tmp_path / "ins", "static")
    (s.folder / "model.py").write_text(RANDOM)
    toml = s.folder / "study.toml"
    import re
    toml.write_text(re.sub(r'scheme = "\w+"', 'scheme = "insample"', toml.read_text()))
    res = s.run(seed=1)
    run = s.record.get(s.record.links(res["id"], rel="of")[0][2])
    assert run.payload["audit"]["verdict"] == "in-sample" and res["payload"]["n"] > 0
    from openlab.study import schema
    assert schema._in_sample(s.record, s.record.get(res["id"]))
