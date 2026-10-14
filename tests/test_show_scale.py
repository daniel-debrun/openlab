"""openlab show at scale: the snapshot the page asks for on every control store change stays fast on a big workspace
(60 agents, 20,000 events, 200 inbox items, 30 plans of 60 steps, 3 studies of thousands of nodes), and it never fails
on a broken openlab.toml, a missing transcript or a study folder that is gone.

`python tests/test_show_scale.py [nodes per study]` builds the full fixture with Record.add and times the snapshot."""

import json
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path

from openlab.agents import plan as P
from openlab.agents.store import Store, _hash, now
from openlab.app import ToolContext, dispatch
from openlab.study.record import Record

KINDS = ("tool", "say", "note", "tool", "tool", "refusal", "run_progress", "plan_node")


def _events(st: Store, agents: list, n: int) -> None:
    """`n` hash-chained events spread over `agents`, written in one transaction (Store.event is one per call)."""
    rows, prev, at = [], "", now()
    for seq in range(1, n + 1):
        a, kind = agents[seq % len(agents)], KINDS[seq % len(KINDS)]
        summary, payload = f"{kind} number {seq}", "{}"
        h = _hash(seq, a, at, kind, "study_status" if kind == "tool" else "", summary, payload, prev)
        rows.append((seq, a, at, kind, "study_status" if kind == "tool" else "", summary, payload, prev, h))
        prev = h
    with closing(sqlite3.connect(st.path)) as con:
        con.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?)", rows)
        con.execute("UPDATE agents SET last_event_at=?", (at,))
        con.commit()


def build(root: Path, agents: int = 60, events: int = 20000, inbox: int = 200, plans: int = 30, steps: int = 60,
          studies: int = 3, nodes: int = 3000) -> Store:
    root = Path(root)
    (root / "openlab.toml").write_text('[team.sam]\nrole = "meteorologist"\nhours = 10\n')
    names = []
    for s in range(studies):
        (root / f"study{s}").mkdir()
        (root / f"study{s}" / "study.toml").write_text(f'name = "study{s}"\n')
        rec = Record(root / f"study{s}")
        for i in range(nodes):
            rec.add("question", {"text": f"question {i} of study {s}"}, session=f"agent-{i % agents:02d}")
        names.append(f"study{s}")
    st = Store(root)
    ids = [f"agent-{i:02d}" for i in range(agents)]
    for i, a in enumerate(ids):
        st.register(a, "spawned", role="lead" if i < 3 else "branch", parent="" if i < 3 else ids[i % 3],
                    status=("running", "done", "idle", "failed")[i % 4], study=names[i % studies],
                    budget={"usd": 5}, question=f"does variant {i} beat the baseline")
    _events(st, ids, events)
    for i in range(inbox):
        item = st.request(ids[i % agents], ("question", "budget", "holdout")[i % 3], f"request {i}",
                          study=names[i % studies], args={"choices": ["a", "b"]} if i % 3 == 0 else {"usd": 1})
        if i % 2:
            st.decide(item, "answered" if i % 3 == 0 else "approved", "you", "ok")
    for a in ids[:plans]:
        P._append(st, a, {"v": 1, "at": now(), "note": "", "nodes": [
            {"id": f"s{k}", "title": f"Step {k}", "kind": "run", "state": ("done", "active", "planned")[k % 3],
             "needs": [f"s{k - 1}"] if k else [], "links": [f"question:{k:04x}", f"inbox:{k + 1}"] if k % 5 == 0 else []}
            for k in range(steps)]})
    return st


def snapshot(root: Path) -> dict:
    env = dispatch(ToolContext(str(root)), "show_snapshot", {})
    assert env["ok"], env
    return env["result"]


def test_the_snapshot_is_fast_on_a_big_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENLAB_AGENT", "")
    build(tmp_path, nodes=150)
    t = time.perf_counter()
    first = snapshot(tmp_path)
    cold = time.perf_counter() - t
    t = time.perf_counter()
    second = snapshot(tmp_path)
    warm = time.perf_counter() - t
    assert cold < 3.0 and warm < 3.0, (cold, warm)
    assert len(first["sessions"]) == 60 and len(first["plans"]) == 30 and len(first["inbox"]) == 100
    assert first["ledger"]["total"] >= 450 and len(first["ledger"]["entries"]) == 400
    assert first["ledger"]["entries"][-1]["at"] >= first["ledger"]["entries"][0]["at"]
    assert first["control_head"] == Store(tmp_path).verify()["head"]
    assert second["plans"] == first["plans"] and second["ledger"] == first["ledger"]


def test_a_cached_read_sees_new_record_nodes_and_plan_steps(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENLAB_AGENT", "")
    st = build(tmp_path, agents=4, events=50, inbox=4, plans=2, steps=6, studies=1, nodes=5)
    s = snapshot(tmp_path)
    Record(tmp_path / "study0").add("question", {"text": "a new one"}, session="agent-00")
    P.step(st, "agent-00", "s1", "done")
    s2 = snapshot(tmp_path)
    assert s2["ledger"]["total"] == s["ledger"]["total"] + 1
    assert s2["ledger"]["entries"][-1]["text"] and "new one" in json.dumps(s2["ledger"]["entries"][-1])
    assert next(n for n in s2["plans"]["agent-00"][-1]["nodes"] if n["id"] == "s1")["state"] == "done"


def test_a_broken_openlab_toml_never_fails_the_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENLAB_AGENT", "")
    build(tmp_path, agents=2, events=10, inbox=2, plans=1, steps=3, studies=1, nodes=2)
    for text in ['[team]\nsam = "not a table"\nbob = 3\n',
                 '[team.sam]\nhours = "ten"\nhours_used = [1]\naway_until = 5\nkey = "yes"\n',
                 'team = ["a", "b"]\n[agents]\ndeny = "rm"\n',
                 '[agents]\nbudget_usd = "lots"\nstuck_minutes = "x"\n[mirror]\nproject = 1\n',
                 'mirror = "on"\n[show]\ncompile_on_done = "no"\n',
                 '[team.sam\nrole = "broken toml',
                 '\x00\xff']:
        (tmp_path / "openlab.toml").write_text(text, encoding="utf-8", errors="surrogateescape")
        s = snapshot(tmp_path)
        assert s["people"][0]["id"] == "you" and len(s["sessions"]) == 2, text
        assert all(isinstance(p["hours"], float) and isinstance(p["cap"], float) for p in s["people"])
        assert all(isinstance(r["label"], str) and "value" in r for r in s["settings"])


def test_a_missing_transcript_and_a_study_that_is_gone_never_fail_the_snapshot(tmp_path, monkeypatch):
    import shutil
    monkeypatch.setenv("OPENLAB_AGENT", "")
    st = build(tmp_path, agents=4, events=40, inbox=4, plans=4, steps=6, studies=2, nodes=5)
    (st.agent_dir("agent-01") / "transcript.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (st.agent_dir("agent-01") / "transcript.jsonl").write_bytes(b'\xff{"type":"result", broken\n')
    st.register("agent-gone", "spawned", status="done", study="study-deleted")
    P._append(st, "agent-gone", {"v": 1, "at": now(), "note": "", "nodes": [
        {"id": "a", "title": "A", "kind": "run", "state": "done", "links": ["question:abcd"]}]})
    shutil.rmtree(tmp_path / "study1")
    st.update("agent-02", study="../outside/nowhere")
    s = snapshot(tmp_path)
    gone = next(x for x in s["sessions"] if x["id"] == "agent-gone")
    assert gone["now"]["kind"] == "done" and gone["study"] == "study-deleted"
    assert s["plans"]["agent-gone"][-1]["nodes"][0]["ev"] == "bad"
    assert {h["study"] for h in s["ledger"]["heads"]} == {"study0"}
    assert next(x for x in s["sessions"] if x["id"] == "agent-01")["now"]["kind"] == "done"


if __name__ == "__main__":
    import tempfile
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    with tempfile.TemporaryDirectory() as d:
        t = time.perf_counter()
        build(Path(d), nodes=n)
        print(f"built in {time.perf_counter() - t:.1f}s")
        for i in range(3):
            t = time.perf_counter()
            snapshot(Path(d))
            print(f"show_snapshot #{i + 1}: {time.perf_counter() - t:.3f}s")
        import cProfile
        import pstats
        pr = cProfile.Profile()
        pr.enable()
        snapshot(Path(d))
        pr.disable()
        pstats.Stats(pr).sort_stats("cumulative").print_stats(25)


def test_full_scale_timing(tmp_path, monkeypatch, capsys):
    """The full fixture (3 studies of OPENLAB_SHOW_SCALE nodes each, Record.add); off unless that variable is set."""
    import os
    n = int(os.environ.get("OPENLAB_SHOW_SCALE") or 0)
    if not n:
        import pytest
        pytest.skip("set OPENLAB_SHOW_SCALE=<nodes per study> to time the full fixture")
    monkeypatch.setenv("OPENLAB_AGENT", "")
    t = time.perf_counter()
    build(tmp_path, nodes=n)
    out = [f"built in {time.perf_counter() - t:.1f}s"]
    for i in range(3):
        t = time.perf_counter()
        snapshot(tmp_path)
        out.append(f"show_snapshot #{i + 1}: {time.perf_counter() - t:.3f}s")
    import cProfile
    import io
    import pstats
    pr = cProfile.Profile()
    pr.enable()
    snapshot(tmp_path)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(30)
    with capsys.disabled():
        print("\n".join(out))
        print(s.getvalue() if os.environ.get("OPENLAB_SHOW_PROFILE") else "")
