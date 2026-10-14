"""Guarantee: the study record is append-only and hash-chained; editing any row shows in verify()."""

import sqlite3

import pytest

from openlab.study.errors import StudyError
from openlab.study.record import Record


@pytest.fixture
def rec(tmp_path):
    return Record(tmp_path)


SHA = "a" * 64


def plan(rec, v=1):
    return rec.add("plan", {"spec": {"v": v}, "sha": SHA})


def dataset(rec, name="x", sha=SHA):
    return rec.add("dataset", {"name": name, "source": {"kind": "provider"}, "sha256": sha, "rows": 1,
                               "columns": ["a"], "card": {}})


def run(rec, p, d, **kw):
    body = {"segment": "research", "key": "k", "cached": False, "spec_sha": SHA, "code": {}, "data": {"x": SHA},
            "env": {}, "sandbox": "bwrap", "audit": {"verdict": "unchecked"}, "seconds": 1.0, **kw}
    return rec.add("run", body, links=[("under", p.id), ("uses", d.id)])


def act(rec, i):
    return rec.add("action", {"verb": "run", "args": {"i": i}, "outcome": {"ok": True}})


def test_add_get_find_last(rec):
    a = plan(rec)
    d = dataset(rec)
    b = run(rec, a, d)
    c = run(rec, a, d, cached=True)
    assert (a.seq, d.seq, b.seq, c.seq) == (1, 2, 3, 4)
    assert a.id == f"plan:{a.hash[:12]}" and b.prev == d.hash and a.prev == ""
    assert rec.get(b.id) == b
    assert [n.id for n in rec.find("run")] == [b.id, c.id]
    assert [n.id for n in rec.find("run", cached=False)] == [b.id]
    assert rec.last("run").id == c.id and rec.last("look") is None
    assert rec.head() == c.hash
    with pytest.raises(StudyError):
        rec.add("nonsense", {})
    with pytest.raises(StudyError):
        rec.add("action", {"verb": "x", "args": {}, "outcome": {}}, links=[("did", "plan:000000000000")])


def test_why_and_impact_are_transitive_deduplicated_and_ordered(rec):
    p, d = plan(rec), dataset(rec)
    r = run(rec, p, d)
    res = rec.add("result", {"metrics": {"mae": {"value": 1.0}}}, links=[("of", r.id)])
    note = rec.add("action", {"verb": "status", "args": {}, "outcome": {}}, links=[("did", res.id)])
    assert [n.id for n in rec.why(res.id)] == [p.id, d.id, r.id]
    assert [n.id for n in rec.impact(p.id)] == [r.id, res.id, note.id]
    assert rec.links(r.id) == [(r.id, "under", p.id), (r.id, "uses", d.id)]


def test_schema_refuses_bad_bodies_links_and_actors(rec):
    p, d = plan(rec), dataset(rec)
    with pytest.raises(StudyError, match="no field"):
        rec.add("dataset", {"name": "y", "nonsense": 1})
    with pytest.raises(StudyError, match="missing"):
        rec.add("dataset", {"name": "y"})
    with pytest.raises(StudyError, match="wrong type"):
        rec.add("dataset", {"name": "y", "source": {}, "sha256": SHA, "rows": "1", "columns": [], "card": {}})
    with pytest.raises(StudyError, match="no recipe"):
        rec.add("dataset", {"name": "y", "source": {"kind": "file"}, "sha256": SHA, "rows": 1, "columns": [],
                            "card": {}})
    with pytest.raises(StudyError, match="needs 1..1 `under`"):
        rec.add("run", {"segment": "research", "key": "k", "cached": False, "spec_sha": SHA, "code": {},
                        "data": {"x": SHA}, "env": {}, "sandbox": "bwrap", "audit": {"verdict": "unchecked"}, "seconds": 1.0},
                links=[("uses", d.id)])
    with pytest.raises(StudyError, match="already sealed"):
        plan(rec, v=2)
    with pytest.raises(StudyError, match="person"):
        rec.add("look", {"n": 1, "reason": "r", "spec_sha": SHA, "code": {}, "data": {}}, links=[("under", p.id)])


def test_a_holdout_run_needs_a_matching_look_and_one_look_one_run(rec, person_key):
    p, d = plan(rec), dataset(rec)
    with pytest.raises(StudyError, match="needs a look"):
        run(rec, p, d, segment="holdout")
    with person_key.act():
        look = rec.add("look", {"n": 1, "reason": "final", "spec_sha": SHA, "code": {"model.py": SHA},
                                "data": {"x": SHA}}, actor="person", links=[("under", p.id)])
    body = {"segment": "holdout", "key": "k", "cached": False, "spec_sha": SHA, "code": {"model.py": "b" * 64},
            "data": {"x": SHA}, "env": {}, "sandbox": "bwrap", "audit": {"verdict": "unchecked"}, "seconds": 1.0}
    with pytest.raises(StudyError, match="differs from its look"):
        rec.add("run", body, links=[("under", p.id), ("uses", d.id), ("after", look.id)])
    body["code"] = {"model.py": SHA}
    rec.add("run", body, links=[("under", p.id), ("uses", d.id), ("after", look.id)])
    with pytest.raises(StudyError, match="already used"):
        rec.add("run", body, links=[("under", p.id), ("uses", d.id), ("after", look.id)])


def test_a_result_is_written_once_per_run(rec):
    r = run(rec, plan(rec), dataset(rec))
    rec.add("result", {"metrics": {"mae": {"value": 1.0}}}, links=[("of", r.id)])
    with pytest.raises(StudyError, match="already has a result"):
        rec.add("result", {"metrics": {"mae": {"value": 2.0}}}, links=[("of", r.id)])


def test_tampering_one_payload_is_reported_at_that_seq(rec):
    for i in range(5):
        act(rec, i)
    assert {k: v for k, v in rec.verify().items() if k != "signatures"} == {"ok": True, "nodes": 5, "broken_at": None,
                                                                          "head": rec.head()}
    with sqlite3.connect(rec.path) as con:
        con.execute("UPDATE nodes SET payload=? WHERE seq=3", ('{"args":{"i":99},"outcome":{"ok":true},"verb":"run"}',))
    v = rec.verify()
    assert v["ok"] is False and v["broken_at"] == 3


def test_tampering_a_link_or_session_is_reported(rec):
    p, d = plan(rec), dataset(rec)
    rec.add("run", {"segment": "research", "key": "k", "cached": False, "spec_sha": SHA, "code": {},
                    "data": {"x": SHA}, "env": {}, "sandbox": "bwrap", "audit": {"verdict": "unchecked"}, "seconds": 1.0},
            links=[("under", p.id), ("uses", d.id)], session="s1")
    act(rec, 2)
    with sqlite3.connect(rec.path) as con:
        con.execute("UPDATE edges SET rel='made' WHERE src LIKE 'run:%' AND rel='uses'")
    assert rec.verify()["broken_at"] == 3
    with sqlite3.connect(rec.path) as con:
        con.execute("UPDATE edges SET rel='uses' WHERE src LIKE 'run:%' AND rel='made'")
        con.execute("UPDATE nodes SET session='s2' WHERE seq=3")
    assert rec.verify()["broken_at"] == 3


def test_once_returns_the_first_node_of_its_kind(rec):
    first = rec.add("plan", {"spec": {"v": 1}, "sha": SHA}, once=True)
    assert rec.add("plan", {"spec": {"v": 2}, "sha": SHA}, once=True) == first and len(rec.find("plan")) == 1


def test_deleting_a_row_is_reported(rec):
    for i in range(4):
        act(rec, i)
    with sqlite3.connect(rec.path) as con:
        con.execute("DELETE FROM nodes WHERE seq=2")
    assert rec.verify()["broken_at"] == 3


def test_chain_continues_across_instances(tmp_path):
    a = plan(Record(tmp_path))
    b = dataset(Record(tmp_path))
    assert b.seq == 2 and b.prev == a.hash
    assert Record(tmp_path).verify()["ok"]


def test_blobs_are_content_addressed(rec):
    sha = rec.put_blob(b"predictions")
    assert rec.put_blob(b"predictions") == sha and rec.blob(sha) == b"predictions"
    (rec.dir / "blobs" / sha).write_bytes(b"edited")
    with pytest.raises(StudyError, match="changed"):
        rec.blob(sha)


def test_the_trial_budget_is_enforced_by_the_record_itself(rec):
    p = rec.add("plan", {"spec": {"budget": {"trials": 1}}, "sha": SHA})
    d = dataset(rec)
    run(rec, p, d)                                         # trial 1 of 1
    run(rec, p, d, cached=True)                            # the same key again is not a new trial
    with pytest.raises(StudyError, match="trial budget is spent"):
        run(rec, p, d, key="k2")
