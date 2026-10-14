"""Guarantee: study.toml is sealed on first use; later edits need a recorded amendment, and the fields that
protect the result (split, budget, ...) can only be amended by a person."""

import pytest

from openlab import study
from openlab.study import spec
from openlab.study.errors import StudyError
from openlab.study.record import Record

TOML = """
[study]
name = "t"
question = "q?"
mode = "static"

[data.table]
file = "data/table.parquet"
recipe = "recipes/make.py"

[split]
scheme = "kfold"
k = 5
fraction = 0.2

[evaluate]
metrics = ["auc", "brier"]

[budget]
trials = 3
"""


def write(folder, text):
    (folder / "study.toml").write_text(text, encoding="utf-8")


def edit(folder, old, new):
    p = folder / "study.toml"
    p.write_text(p.read_text().replace(old, new, 1), encoding="utf-8")


@pytest.mark.parametrize("old, new, key", [
    ('mode = "static"', 'mode = "both"', "study.mode"),
    ('question = "q?"\n', "", "study.question"),
    ('scheme = "kfold"', 'shceme = "kfold"', "split.shceme"),
    ("[budget]", "[budjet]", "budjet"),
    ('scheme = "kfold"', 'scheme = "walk_forward"', "split.scheme"),
    ('metrics = ["auc", "brier"]', "metrics = []", "evaluate.metrics"),
    ("trials = 3", "trials = 0", "budget.trials"),
    ("trials = 3", "trials = 3\nholdout_looks = -1", "budget.holdout_looks"),
    ('recipe = "recipes/make.py"', "", "data.table.recipe"),
    ('file = "data/table.parquet"', 'file = "x"\nprovider = "fred"', "data.table"),
    ("trials = 3", "trials = 3\n\n[sandbox]\nthreads = 0", "sandbox.threads"),
])
def test_load_names_the_bad_key(tmp_path, old, new, key):
    write(tmp_path, TOML)
    edit(tmp_path, old, new)
    with pytest.raises(StudyError) as e:
        spec.load(tmp_path)
    assert f"`{key}`" in e.value.error


def test_load_fills_defaults_and_dataset_specs(tmp_path):
    write(tmp_path, TOML)
    s = spec.load(tmp_path)
    assert s.budget == {"holdout_looks": 1, "trials": 3}
    assert s.data["table"].recipe == "recipes/make.py" and s.evaluate["metrics"][0] == "auc"


def test_sandbox_threads_default_to_one_and_are_a_persons_field(tmp_path):
    from openlab.study import runner, sandbox
    write(tmp_path, TOML)
    assert runner.threads(spec.load(tmp_path)) == 1
    assert {sandbox._env()[k] for k in sandbox.THREADS} == {"1"} and sandbox._env(4)["OMP_NUM_THREADS"] == "4"
    write(tmp_path, TOML + "\n[sandbox]\nthreads = 4\n")
    assert runner.threads(spec.load(tmp_path)) == 4
    assert "sandbox" in spec.PERSON_FIELDS


def test_seal_is_idempotent(tmp_path):
    write(tmp_path, TOML)
    rec = Record(tmp_path)
    a, b = spec.seal(tmp_path, rec, "agent"), spec.seal(tmp_path, rec, "agent")
    assert a == b and len(rec.find("plan")) == 1
    assert spec.check(tmp_path, rec).sha() == spec.load(tmp_path).sha()


def test_edit_without_amendment_fails_check_and_names_the_key(tmp_path):
    write(tmp_path, TOML)
    rec = Record(tmp_path)
    spec.seal(tmp_path, rec, "agent")
    edit(tmp_path, 'metrics = ["auc", "brier"]', 'metrics = ["auc", "logloss"]')
    with pytest.raises(StudyError) as e:
        spec.check(tmp_path, rec)
    assert "evaluate.metrics" in e.value.meaning


def test_agent_may_amend_agent_fields(tmp_path):
    write(tmp_path, TOML)
    rec = Record(tmp_path)
    spec.seal(tmp_path, rec, "agent")
    edit(tmp_path, 'metrics = ["auc", "brier"]', 'metrics = ["auc", "brier", "logloss"]')
    node = spec.amend(tmp_path, rec, "add logloss", person=False, actor="agent")
    assert list(node.payload["changes"]) == ["evaluate.metrics"] and node.payload["person_fields"] == []
    assert spec.sealed(rec).evaluate["metrics"] == ["auc", "brier", "logloss"]
    spec.check(tmp_path, rec)


@pytest.mark.parametrize("old, new", [
    ("fraction = 0.2", "fraction = 0.3"),                                          # split.*
    ('metrics = ["auc", "brier"]', 'metrics = ["brier", "auc"]'),                  # evaluate.metrics[0]
    ('recipe = "recipes/make.py"', 'recipe = "recipes/make.py"\nknown_at = "t"'),  # data.*.known_at
])
def test_agent_amendment_of_person_field_is_refused_and_person_one_accepted(tmp_path, old, new, person_key):
    write(tmp_path, TOML)
    rec = Record(tmp_path)
    spec.seal(tmp_path, rec, "agent")
    edit(tmp_path, old, new)
    with pytest.raises(StudyError, match="only be amended by a person"):
        spec.amend(tmp_path, rec, "tweak", person=False, actor="agent")
    assert rec.find("amendment") == []
    with person_key.act():
        spec.amend(tmp_path, rec, "a person agreed", person=True, actor="person")
    assert spec.sealed(rec).sha() == spec.load(tmp_path).sha()


def test_person_amends_split_holdout_and_sealed_reflects_it(tmp_path, person_key):
    study.new(tmp_path / "s", "temporal")
    s = study.open(tmp_path / "s")
    s.seal()
    edit(s.folder, "holdout = [2025-01-01, 2025-06-30]", "holdout = [2025-02-01, 2025-06-30]")
    with pytest.raises(StudyError):
        s.amend("move the holdout")
    with person_key.act():
        out = s.amend("move the holdout", person=True)
    assert out["actor"] == "person" and "split.holdout" in out["payload"]["changes"]
    assert s.spec.split["holdout"] == ["2025-02-01", "2025-06-30"]
    assert [n.payload["outcome"]["ok"] for n in s.record.find("action", verb="amend")] == [False, True]
    assert s.status()["amendments"][0]["changed"] == ["split.holdout"] and s.verify()["ok"]


@pytest.mark.parametrize("template", study.TEMPLATES)
def test_new_scaffolds_a_study_that_loads(tmp_path, template):
    s = study.new(tmp_path / template, template)
    assert s.spec.mode == template and s.spec.name == template
    files = ("README.md", ".gitignore", ".git") if template == "review" else \
        ("model.py", "recipes/make_data.py", "data", ".gitignore", ".git")
    assert all((s.folder / f).exists() for f in files)
    with pytest.raises(StudyError):
        study.new(tmp_path / template, template)
