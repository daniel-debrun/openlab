"""Guarantees about comparing a study's variants: two of its own results compare without a [review], paired on the
rows both scored, a test may take another result as its baseline, and results of different protocols (an amended
split) are refused, naming what differs."""

import shutil

import pytest

import openlab.study as vs
from openlab.study import schema
from openlab.study.errors import StudyError

pytestmark = pytest.mark.skipif(not shutil.which("bwrap"), reason="runs are tested under bwrap")


def test_variants_compare_and_test_paired_on_their_rows_and_refuse_another_protocol(tmp_path, person_key):     # ~20 s
    s = vs.new(tmp_path / "v", "temporal")
    toml = s.folder / "study.toml"
    toml.write_text(toml.read_text() + "\n[audit]\nn = 2\n")
    a = s.run(note="x and y_now")
    model = s.folder / "model.py"
    model.write_text(model.read_text().replace('COLUMNS = ["x", "y_now"]', 'COLUMNS = ["x"]'))
    b = s.run(note="x only")
    assert "review" not in s.spec.raw

    cmp = s.compare(b["id"], a["id"], "mae")["payload"]
    p = cmp["paired"]
    assert cmp["values"] == [pytest.approx(b["payload"]["metrics"]["mae"]["value"]),
                             pytest.approx(a["payload"]["metrics"]["mae"]["value"])]   # every row is shared here
    assert p["n"] == a["payload"]["n"] and p["only_a"] == p["only_b"] == 0 and p["interval"] == "block_bootstrap"
    assert p["lo"] <= p["diff"] <= p["hi"] and 0 < p["p"] <= 1
    assert p["diff"] == pytest.approx(cmp["values"][0] - cmp["values"][1])
    assert cmp["verdict"] == ("better" if cmp["values"][0] < cmp["values"][1] else "worse")
    with pytest.raises(StudyError, match="does not follow"):            # the paired numbers cannot be typed
        s.record.add("comparison", {**cmp, "paired": {**p, "p": 0.0001}},
                     links=[("compares", b["id"]), ("compares", a["id"]), ("under", s.record.last("plan").id)])

    q = s.question("does y_now help?")
    h = s.hypothesis("x alone does as well", "mae within 0.01", "x only is worse", q["id"], ["features"])
    t = s.test(h["id"], b["id"], a["id"], 0.02)
    assert t["payload"]["p"] == pytest.approx(p["p"]) and t["payload"]["baseline"] == a["id"]
    assert s.record.links(t["id"], rel="against") == [(t["id"], "against", a["id"])]
    with pytest.raises(StudyError, match="has no paired p"):          # a name that is neither a baseline nor a result
        s.test(h["id"], b["id"], "nonesuch", 0.01)

    toml.write_text(toml.read_text().replace("research = [2024-01-01, 2024-12-31]",
                                             "research = [2024-03-01, 2024-12-31]"))
    with person_key.act():
        s.amend("a shorter research window", person=True)
    c = s.run(note="shorter window")
    with pytest.raises(StudyError, match="different protocols") as e:
        s.compare(c["id"], b["id"], "mae")
    assert "[split]" in e.value.meaning
    with pytest.raises(StudyError, match="different protocols"):
        schema.paired_p(s.record.get(c["id"]), b["id"], s.record)
