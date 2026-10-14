"""evaluate.by = "key": a result also scores each entity on its own (a station, a dataset), with its intervals,
so a paper's per-station numbers are checked against that station's scores."""

import shutil

import pytest

import openlab.study as vs
from openlab.study import schema
from openlab.study.errors import StudyError


@pytest.mark.skipif(not shutil.which("bwrap"), reason="the study's code runs under bwrap")
def test_each_entity_is_scored_on_its_own(tmp_path):
    s = vs.new(tmp_path / "g", "temporal")
    toml = s.folder / "study.toml"
    toml.write_text(toml.read_text().replace("[evaluate]", '[evaluate]\nby = "key"'))
    res = s.run()["payload"]
    g = res["groups"]
    assert sorted(g) == ["a", "b", "c", "d"] and sum(x["n"] for x in g.values()) == res["n"]
    a = g["a"]["metrics"]["mae"]
    assert a["lo"] <= a["value"] <= a["hi"] and g["a"]["baselines"]["persistence"]["mae"]["value"] > 0
    r = s.record.get(s.runs()["runs"][-1]["result"]) if "result" in s.runs()["runs"][-1] else s.record.last("result")
    assert schema._ours(s.record, r, "mae", "model", "b")["value"] == g["b"]["metrics"]["mae"]["value"]
    with pytest.raises(StudyError, match="scored no group `z`"):
        schema._ours(s.record, r, "mae", "model", "z")


@pytest.mark.skipif(not shutil.which("bwrap"), reason="the study's code runs under bwrap")
def test_show_prints_a_result_with_its_groups(tmp_path, capsys):
    from openlab.study import cli
    s = vs.new(tmp_path / "g", "temporal")
    toml = s.folder / "study.toml"
    toml.write_text(toml.read_text().replace("[evaluate]", '[evaluate]\nby = "key"'))
    rid = s.run()["id"]
    assert cli.main(["show", str(s.folder), rid]) == 0
    out = capsys.readouterr().out
    assert f"{rid}  result" in out and "group a (" in out and "baseline persistence" in out


@pytest.mark.skipif(not shutil.which("bwrap"), reason="the study's code runs under bwrap")
@pytest.mark.parametrize("scheme", ["stratified", "insample"])
def test_a_static_study_scores_each_key_too(tmp_path, scheme):
    """Two datasets stacked in one table, keyed by `ds`: each is scored on its own, held out or in-sample."""
    import re
    s = vs.new(tmp_path / "st", "static")
    recipe = s.folder / "recipes" / "make_data.py"
    recipe.write_text(recipe.read_text().replace(".with_columns(y=", '.with_columns(ds=pl.Series(["a", "b"] * (n // 2)), y='))
    model = s.folder / "model.py"
    model.write_text(model.read_text() + '\n\ndef features(frame):\n    return frame.drop("ds")\n')
    toml = s.folder / "study.toml"
    toml.write_text(re.sub(r'scheme = "\w+"', f'scheme = "{scheme}"', toml.read_text())
                    .replace('recipe = "recipes/make_data.py"', 'recipe = "recipes/make_data.py"\nkey = ["ds"]')
                    .replace("[evaluate]", '[evaluate]\nby = "key"'))
    res = s.run()["payload"]
    g = res["groups"]
    assert sorted(g) == ["a", "b"] and sum(x["n"] for x in g.values()) == res["n"]
    assert 0.5 < g["a"]["metrics"]["auc"]["value"] <= 1
