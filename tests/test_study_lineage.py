"""Guarantees of a run's place in the search tree: `op` and `parent` are checked before anything runs (a draft has no
parent; improve, debug and fuse name one; only a fuse or an ensemble refs other runs; the search moves over research
runs only); a run records the parent it was given, else the branch's last run as before; and every run records the
machine it ran on."""

import shutil

import pytest

from openlab import study as vs
from openlab.study import schema
from openlab.study.errors import StudyError


class N:
    def __init__(self, id, segment="research"):
        self.id, self.payload = id, {"segment": segment}


@pytest.mark.parametrize("op, parent, refs, error", [
    ("draft", [N("run:a")], [], "a draft has no parent"),
    ("improve", [], [], "an improve run needs its parent"),
    ("debug", [], [], "a debug run needs its parent"),
    ("improve", [N("run:a")], [N("run:b")], "only a fuse or an ensemble refs other runs"),
    ("fuse", [N("run:a")], [N("run:a")], "not also one of its refs"),
    ("improve", [N("run:h", "holdout")], [], "is a holdout run"),
    ("anneal", [], [], "no run op"),
])
def test_an_op_that_does_not_fit_its_links_is_refused(op, parent, refs, error):
    with pytest.raises(StudyError, match=error):
        schema._run_op(op, "research", {"branches_from": parent, "refs": refs})


def test_the_ops_that_fit_pass():
    schema._run_op("draft", "research", {})
    schema._run_op("improve", "research", {"branches_from": [N("run:a")]})
    schema._run_op("fuse", "research", {"branches_from": [N("run:a")], "refs": [N("run:b"), N("run:c")]})
    schema._run_op("", "research", {"branches_from": [N("run:a")]})          # a run from before operators
    with pytest.raises(StudyError, match="only a research run has an op"):
        schema._run_op("improve", "holdout", {"branches_from": [N("run:a")]})


@pytest.mark.skipif(not shutil.which("bwrap"), reason="runs are tested under bwrap")
def test_runs_record_the_parent_and_op_they_were_given_and_their_machine(tmp_path):          # ~30 s
    s = vs.new(tmp_path / "t", "temporal")
    (s.folder / "study.toml").write_text((s.folder / "study.toml").read_text() + "\n[audit]\nn = 2\n")
    first = s.run(note="first", op="draft")
    run1 = s.record.links(first["id"], rel="of")[0][2]
    with pytest.raises(StudyError, match="needs its parent"):           # refused before anything runs
        s.run(op="improve")
    with pytest.raises(StudyError, match="no run `run:nope`"):
        s.run(op="improve", parent="run:nope")
    assert len(s.record.find("run")) == 1
    model = s.folder / "model.py"
    model.write_text(model.read_text().replace('COLUMNS = ["x", "y_now"]', 'COLUMNS = ["x"]'))
    second = s.run(note="x only", op="improve", parent=first["id"])     # a result id names its run
    run2 = s.record.get(s.record.links(second["id"], rel="of")[0][2])
    assert run2.payload["op"] == "improve" and s.record.links(run2.id, rel="branches_from")[0][2] == run1
    assert run2.payload["box"]["cores"] >= 1 and run2.payload["box"]["host"]
    rows = {r["id"]: r for r in s.runs()["runs"]}
    assert rows[run2.id]["op"] == "improve" and rows[run2.id]["parent"] == run1 and rows[run1]["parent"] is None
    assert s.verify()["ok"]
