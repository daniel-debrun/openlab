"""A workspace's harness versions (openlab.harness.versions): stored by fingerprint, made active only by the person
(signed) and only with a supporting lesson, inherited by a probe's children, refused when changed on disk; the
workspace's LAWS.openlab binds every version; the person verbs are refused to an agent's shell."""

import pytest

from harness_fake import candidate, launch
from openlab import harness as H
from openlab.harness import versions as V


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "openlab.toml").write_text("")
    return tmp_path


def test_a_version_is_stored_once_by_its_fingerprint(ws):
    c = candidate(ws, ws / "cand")
    one = V.install(ws, c, note="a sentence more")
    assert one["new"] and (V.path(ws, one["sha"]) / "common.md").read_text().endswith("once more.\n")
    assert not V.install(ws, c)["new"] and V.resolve(ws, one["sha"][:8]) == one["sha"]
    assert V.install(ws, H.DEFAULT)["default"] and V.resolve(ws, "default") == ""
    assert [v["sha"] for v in V.stored(ws)] == [H.load().sha, one["sha"]]
    assert "+- Before you finish, read your own plan once more." in V.diff(ws, "default", one["sha"])
    (c / "hooks.json").write_text("{}")
    with pytest.raises(H.HarnessError, match="hooks.json is not part of a harness"):
        V.install(ws, c)


def test_making_a_version_active_is_a_signed_person_act_resting_on_a_lesson(ws, person_key):
    sha = V.install(ws, candidate(ws, ws / "cand"))["sha"]
    with pytest.raises(PermissionError, match="person's act"):
        V.use(ws, sha, "try it")
    with person_key.act(), pytest.raises(H.HarnessError, match="rests on no approved harness lesson"):
        V.use(ws, sha, "try it")
    with person_key.act():                                            # the packaged one is always allowed
        assert not V.use(ws, "default", "nothing to change")["changed"]
    # once active (here: recorded as active before), it may come back without a lesson
    from openlab.agents.store import Store
    Store(ws).event(V.AGENT, "harness_use", payload={"sha": sha, "prev": ""})
    with person_key.act():
        out = V.use(ws, sha, "the probe held")
    assert out["changed"] and V.active_sha(ws) == sha and V.active(ws).sha == sha
    ev = Store(ws).events(V.AGENT, n=5, kinds=("harness_use",))[-1]
    assert ev["payload"]["sig"]["key"] and ev["payload"]["by"].startswith("person:")
    with person_key.act():
        back = V.rollback(ws, "go back")
    assert back["changed"] and V.active_sha(ws) == ""


def test_the_active_version_briefs_spawns_and_a_childs_harness_is_its_parents(ws, person_key):
    from openlab.agents.spawn import spawn
    sha = V.install(ws, candidate(ws, ws / "cand", extra="- PROBE SENTENCE.", rel="brief.md"))["sha"]
    probe = spawn(ws, "x", role="lead", start=False, harness=sha, budget={"usd": 1.0})
    child = spawn(ws, "y", role="screener", start=False, parent=probe["id"], budget={"usd": 0.2, "minutes": 5})
    plain = spawn(ws, "z", role="screener", start=False)
    assert launch(ws, probe["id"])["harness"] == sha == launch(ws, child["id"])["harness"]
    assert launch(ws, plain["id"])["harness"] == H.load().sha
    from openlab.agents.store import Store
    brief = lambda a: (Store(ws).agent_dir(a) / "brief.txt").read_text()                   # noqa: E731
    assert "PROBE SENTENCE" in brief(child["id"]) and "PROBE SENTENCE" not in brief(plain["id"])
    (V.path(ws, sha) / "self.md").write_text("tampered\n")
    from openlab.agents.spawn import AgentError
    with pytest.raises(AgentError, match="changed on disk"):
        spawn(ws, "x", role="lead", start=False, harness=sha)


def test_the_workspace_laws_bind_every_harness(ws):
    (ws / "LAWS.openlab").write_text("# the person's\nforbid: (?i)skip the audit\nrequire: A refusal is the structure "
                                     "working\nnot a rule: ignored\n")
    assert H.laws(ws) == {"forbid": ["(?i)skip the audit"], "require": ["A refusal is the structure working"]}
    assert H.validate(H.load(), H.laws(ws)) == []
    bad = candidate(ws, ws / "bad", extra="- When short of time, Skip the audit.")
    assert "matches the law `forbid: (?i)skip the audit`" in "\n".join(H.validate(H.read(bad), H.laws(ws)))
    (ws / "LAWS.openlab").write_text("require: Ask the person before any trial.\n")
    from openlab.agents.spawn import AgentError, spawn
    with pytest.raises(AgentError, match="lacks the law"):
        spawn(ws, "x", start=False)


def test_a_harness_request_is_decided_in_the_inbox_and_the_shell_verbs_are_the_persons(ws, person_key):
    from openlab.agents import inbox
    from openlab.agents.hooks import guard
    from openlab.agents.store import Store
    sha = V.install(ws, candidate(ws, ws / "cand"))["sha"]
    Store(ws).event(V.AGENT, "harness_use", payload={"sha": sha, "prev": ""})            # active once before
    st = Store(ws)
    args = inbox.check_request(st, "lead-1", "harness", {"action": "use", "version": sha[:10]})
    item = st.request("lead-1", "harness", "the probe's harness held", args=args)
    head, details = inbox.describe(st, st.item(item))
    assert f"make harness {sha[:12]} active" in head and any("once more" in d for d in details)
    with person_key.act():
        inbox.decide(st, item, True, "ok")
    assert V.active_sha(ws) == sha
    with pytest.raises(inbox.RequestError, match="names its act"):
        inbox.check_request(st, "lead-1", "harness", {"action": "delete"})
    for cmd in ("openlab harness use abc123 --reason x", "openlab harness rollback --reason x",
                "openlab harness refine task . --k 1", "openlab harness reflect lead-1",
                "python -c 'from openlab.harness import versions; versions.use(r, s, x)'"):
        assert (guard("Bash", {"command": cmd}) or [None])[0] == "bypass", cmd
    for cmd in ("openlab harness show", "openlab harness signals", "openlab harness lessons"):
        assert guard("Bash", {"command": cmd}) is None, cmd


def test_per_role_tool_allowlists_come_from_the_persons_openlab_toml(ws):
    (ws / "openlab.toml").write_text('[agents.roles.screener]\nallowed_tools = ["mcp__openlab", "Read", "Grep"]\n')
    from openlab.agents.spawn import spawn
    s = spawn(ws, "x", role="screener", start=False)
    e = spawn(ws, "x", role="experimenter", start=False)
    assert launch(ws, s["id"])["allowed_tools"] == ["mcp__openlab", "Read", "Grep"]
    assert "Bash" in launch(ws, e["id"])["allowed_tools"]
    other = spawn(ws, "x", role="screener", start=False, allowed_tools=["mcp__openlab", "Read"])
    assert launch(ws, other["id"])["harness_sha"] != launch(ws, s["id"])["harness_sha"]       # tools are told too


def test_mcp_instructions_and_role_files_come_from_the_active_version(ws, person_key, monkeypatch):
    from openlab import agent_guide
    from openlab.agents.claude_setup import init_claude
    from openlab.agents.store import Store
    sha = V.install(ws, candidate(ws, ws / "cand", extra="- GUIDE SENTENCE.", rel="guide.md"))["sha"]
    c2 = candidate(ws, ws / "cand2", extra="- ROLE SENTENCE.", rel="roles/reviewer.md")
    sha2 = V.install(ws, c2)["sha"]
    for s in (sha, sha2):
        Store(ws).event(V.AGENT, "harness_use", payload={"sha": s, "prev": ""})
    with person_key.act():
        V.use(ws, sha2, "try")
    init_claude(ws)
    assert "ROLE SENTENCE" in (ws / ".claude" / "agents" / "openlab-reviewer.md").read_text()
    with person_key.act():
        V.use(ws, sha, "try")
    monkeypatch.setenv("OPENLAB_WORKSPACE", str(ws))
    assert "GUIDE SENTENCE" in agent_guide.instructions("")
