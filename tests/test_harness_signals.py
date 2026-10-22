"""Probe signals (openlab.harness.signals): rates read from a run's events, keyed by the harness that briefed it; a
contrast that calls a difference only beyond the runs' noise; the transcript grammar, never cut short."""

import json

import pytest

from harness_fake import candidate, run
from openlab.harness import signals as S
from openlab.harness import versions as V


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "openlab.toml").write_text("")
    return tmp_path


def test_signals_are_rates_keyed_by_the_harness(ws):
    a = run(ws, calls=20, refusals=4, errors=2, tools=("WebFetch", "Read"))
    s = S.of(ws, a)
    assert s["signals"]["refusal_rate"] == 0.2 and s["signals"]["tool_error_rate"] == 0.1
    assert s["signals"]["loop_rate"] == 5.0                          # one refusal repeated 4 times, per 100 calls
    assert s["signals"]["deliverable"] == 0.0                        # it wrote nothing to a record or a plan
    assert s["harness_sha"] and s["harness"] and s["untrusted"] is True
    assert set(s["signals"]) == set(S.FIELDS)


def test_a_contrast_calls_a_difference_only_beyond_the_noise(ws):
    sha = V.install(ws, candidate(ws, ws / "cand"))["sha"]
    ref = [run(ws, calls=10, refusals=r) for r in (3, 4, 3)]
    cand = [run(ws, calls=10, refusals=r, harness=sha) for r in (0, 1, 0)]
    rows = S.runs(ws, role="screener")
    by = S.groups(rows, "harness")
    assert len(by[sha]) == 3 and sum(len(v) for v in by.values()) == 6
    c = S.contrast(by[V.default_sha()], by[sha])
    r = next(x for x in c["rows"] if x["field"] == "refusal_rate")
    assert r["moved"] and r["favours"] == "cand" and r["delta"] < 0
    one = S.contrast([S.of(ws, ref[0])], [S.of(ws, cand[0])])
    r1 = next(x for x in one["rows"] if x["field"] == "refusal_rate")
    assert not r1["moved"] and r1["favours"] == "none (noise)"        # one run a side shows nothing
    cid = S.save(ws, c, note="t")
    assert cid.startswith("hc-") and S.load_contrast(ws, cid)["n"] == [3, 3] and S.save(ws, c, note="t") == cid
    with pytest.raises(KeyError):
        S.load_contrast(ws, "hc-000000000000")


def test_the_transcript_grammar_keeps_long_observations_whole(ws):
    from openlab.agents.store import Store
    a = run(ws, calls=0)
    long = "x" * 12_000
    lines = [{"type": "assistant", "message": {"content": [
                 {"type": "text", "text": "Reading."},
                 {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "a.py"}},
                 {"type": "tool_use", "id": "t2", "name": "mcp__openlab__openlab_study_run", "input": {}}]}},
             {"type": "user", "message": {"content": [
                 {"type": "tool_result", "tool_use_id": "t1", "content": long},
                 {"type": "tool_result", "tool_use_id": "t2", "content": json.dumps(
                     {"ok": False, "error": "refused", "meaning": "m", "try": "t"})}]}},
             {"type": "result", "subtype": "success", "result": "done"}]
    (Store(ws).agent_dir(a) / "transcript.jsonl").write_text("\n".join(json.dumps(x) for x in lines))
    t = S.transcript(ws, a, observation_max=100)                      # asked for less: 10,000 at least
    assert "[SAY] Reading." in t and '[ACTION] Read {"file_path": "a.py"}' in t
    assert "[OBSERVATION] Read: " + "x" * 10_000 in t and "2000 more characters cut" in t
    assert "[ERROR] mcp__openlab__openlab_study_run" in t and "[RESULT] success: done" in t
    b = run(ws, calls=2, refusals=1)                                  # no CLI transcript: the events
    assert "[ERROR] study_run: refused the same way" in S.transcript(ws, b)
