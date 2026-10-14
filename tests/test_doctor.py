"""`openlab doctor`: the rows a research install needs, and never a credential."""

import json
import sys

import pytest

from openlab import doctor as D


@pytest.mark.parametrize("gpus", [None, "0, NVIDIA GeForce RTX 3090, 24576 MiB"])
def test_every_row_is_a_verdict_or_a_path_and_carries_a_fix_when_it_fails(gpus, monkeypatch):
    monkeypatch.setattr(D, "_gpus", lambda: gpus)            # a machine without, and with, nvidia-smi
    r = D.check()
    names = [row["check"] for row in r.rows]
    for want in ("python", "openlab", "core", "sandbox", "p-values", "agent CLI", "MCP", "web UI",
                 "workspace", "hooks", "person key", "vault", "signing key", "snapshots"):
        assert want in names, want
    assert [n for n in names if n.startswith("extra: ")] == [f"extra: {k}" for k in D.EXTRAS]
    for row in r.rows:
        assert set(row) == {"check", "ok", "detail", "fix", "note"}
        assert row["ok"] in (True, False, None) and row["detail"]
        if row["ok"] is False:
            assert row["fix"], row                       # a failure without a fix is a complaint
        if row["ok"] is None:
            assert not row["fix"]                        # a path is information, not a problem
            assert row["note"]                           # it explains itself in `note` instead


def test_the_python_and_core_rows_pass_in_the_environment_running_this():
    by = {row["check"]: row for row in D.check().rows}
    assert by["python"]["ok"] and sys.version.split()[0] in by["python"]["detail"]
    assert by["core"]["ok"] and by["openlab"]["ok"]
    assert by["MCP"]["ok"] and "registers" in by["MCP"]["detail"]


def test_the_workspace_row_finds_the_studies_here(tmp_path, monkeypatch):
    """Paths are reported *for the current directory*, because that is what decides them."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "study.toml").write_text("[study]\n")
    by = {row["check"]: row for row in D.check().rows}
    assert by["workspace"]["detail"].startswith(f"{tmp_path} (") and "1 study: a" in by["workspace"]["note"]
    assert str(tmp_path) not in by["vault"]["detail"]           # outside the workspace, on purpose


def test_the_sandbox_row_says_what_a_missing_bwrap_means(monkeypatch):
    import openlab.study.sandbox as S
    monkeypatch.setattr(S, "_bwrap", lambda study, stage, network: ([], "bwrap is not installed"))
    monkeypatch.setattr(D, "_userns_blocked", lambda: False)
    row = {row["check"]: row for row in D.check().rows}["sandbox"]
    assert row["ok"] is False and "unsandboxed" in row["detail"] and "bubblewrap" in row["fix"]
    monkeypatch.setattr(D, "_userns_blocked", lambda: True)          # a container that forbids namespaces
    row = {row["check"]: row for row in D.check().rows}["sandbox"]
    assert "forbids user namespaces" in row["fix"] and "another machine" in row["fix"]


def test_the_agent_cli_row_finds_claude_or_codex(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENLAB_AGENT_CMD", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    row = {row["check"]: row for row in D.check().rows}["agent CLI"]
    assert row["ok"] is False and "claude" in row["detail"] and "codex" in row["detail"]
    (tmp_path / "codex").write_text("#!/bin/sh\n")
    (tmp_path / "codex").chmod(0o755)
    row = {row["check"]: row for row in D.check().rows}["agent CLI"]
    assert row["ok"] is True and "codex" in row["detail"]


def test_a_missing_extra_is_a_failed_row_with_the_pip_line(monkeypatch):
    monkeypatch.setattr(D, "_have", lambda m: m not in ("scipy", "mlflow"))
    by = {row["check"]: row for row in D.check().rows}
    assert by["extra: study"]["ok"] is False and 'pip install "openlab-research[study]"' in by["extra: study"]["fix"]
    assert "missing mlflow" in by["extra: study"]["detail"]
    assert by["p-values"]["ok"] is False and "normal approximation" in by["p-values"]["detail"]
    assert by["core"]["ok"] is True
    assert "extra: agent" not in by and "extra: app" not in by       # the MCP and the web workspace are core


def test_a_missing_core_dependency_says_reinstall_openlab(monkeypatch):
    monkeypatch.setattr(D, "_have", lambda m: m != "fastapi")
    row = {row["check"]: row for row in D.check().rows}["core"]
    assert row["ok"] is False and "missing fastapi" in row["detail"] and "pip install" in row["fix"]
    assert "openlab-research[" not in row["fix"]


def test_a_missing_ui_build_is_reported_because_the_page_would_simply_be_blank(monkeypatch, tmp_path):
    monkeypatch.setattr(D, "__file__", str(tmp_path / "doctor.py"))
    by = {row["check"]: row for row in D.check().rows}
    assert by["web UI"]["ok"] is False and "blank page" in by["web UI"]["detail"]
    assert "npm run build" in by["web UI"]["fix"]


def test_no_credential_is_ever_read(monkeypatch):
    """The agent CLI row reports a binary on PATH and nothing else: openlab holds no model key anywhere."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-never-be-read")
    monkeypatch.setenv("OPENLAB_APP_TOKEN", "tok-should-never-be-read")
    rendered = D.render(D.check()) + json.dumps(D.check().to_record())
    assert "should-never-be-read" not in rendered
    src = open(D.__file__, encoding="utf-8").read()
    for forbidden in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENLAB_API_KEY", "OPENLAB_APP_TOKEN"):
        assert forbidden not in src, forbidden


def test_provider_keys_are_reported_set_or_not_never_by_value(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "fred-secret-value")
    monkeypatch.delenv("EIA_API_KEY", raising=False)
    by = {row["check"]: row for row in D.check().rows}
    assert "FRED_API_KEY" in by["provider keys"]["detail"] and "EIA_API_KEY" not in str(by["provider keys"])
    assert "fred-secret-value" not in D.render(D.check())
    from openlab.sources import PROVIDERS
    for p in PROVIDERS.values():                                    # none set: no row about data-provider keys
        for var in p.needs:
            monkeypatch.delenv(var, raising=False)
    assert "provider keys" not in {row["check"] for row in D.check().rows}
    assert "equation gate (sympy/lark" in {row["check"]: row for row in D.check().rows}["core"]["detail"]


def test_render_marks_failures_and_prints_the_fix_under_them(monkeypatch):
    monkeypatch.setattr(D, "_have", lambda m: m != "mcp")
    text = D.render(D.check())
    assert "FAIL  MCP" in text and "-> pip install --force-reinstall openlab" in text
    assert "to fix" in text.splitlines()[-1] and "MCP" in text.splitlines()[-1]
    assert "  ok    " in text and "  --    " in text                # verdicts and paths read differently


def test_the_cli_prints_a_table_or_json_and_exits_on_the_verdict(capsys, monkeypatch, tmp_path):
    from openlab.__main__ import main
    monkeypatch.chdir(tmp_path)
    broken = bool(D.check().broken)
    assert main(["doctor"]) == int(broken)
    assert ("to fix" if broken else "all good") in capsys.readouterr().out
    assert main(["doctor", "--json"]) == int(broken)
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is not broken and out["rows"] and out["python"] == sys.version.split()[0]
    assert set(out) == {"ok", "python", "openlab", "cwd", "rows"}
    monkeypatch.setattr(D, "_have", lambda m: False)
    assert main(["doctor"]) == 1                                    # non-zero, so it can gate a setup script
    assert "to fix" in capsys.readouterr().out


@pytest.mark.parametrize("bad", [["doctor", "--nope"]])
def test_an_unknown_flag_is_a_usage_error(bad):
    from openlab.__main__ import main
    with pytest.raises(SystemExit):
        main(bad)


def test_the_workspace_rows_say_openlab_toml_hooks_person_key_and_agents(tmp_path, monkeypatch, person_home):
    monkeypatch.chdir(tmp_path)
    by = {row["check"]: row for row in D.check().rows}
    assert "no openlab.toml" in by["workspace"]["detail"] and by["hooks"]["detail"] == "no"
    assert by["person key"]["detail"] == "no" and "openlab person init" in by["person key"]["note"]
    assert "agents" not in by                                       # no control store: nothing started
    from openlab.agents.claude_setup import init_claude
    from openlab.agents.store import Store
    init_claude(tmp_path)
    (person_home / ".config" / "openlab").mkdir(parents=True)
    (person_home / ".config" / "openlab" / "person.pub").write_text("{}")
    Store(tmp_path).register("lead", "claude-session")
    by = {row["check"]: row for row in D.check().rows}
    assert "(openlab.toml)" in by["workspace"]["detail"] and by["hooks"]["detail"].startswith("yes")
    assert by["person key"]["detail"].startswith("yes") and by["agents"]["detail"].endswith("0 waiting in the inbox")
    assert by["hooks"]["ok"] is True and by["person key"]["ok"] is True       # a part that is set up passes


def test_the_machine_row_shows_the_containers_limits_beside_the_hosts(monkeypatch):
    from openlab.study import machine
    monkeypatch.setattr(machine, "describe", lambda: {"cores": 23, "memory_gb": 120.8, "host_cores": 96,
                                                      "host_memory_gb": 251.0, "limited": True})
    row = next(x for x in D.check().rows if x["check"] == "machine")
    assert row["detail"].startswith("23 cores, 120.8 GB (the container's limits; the host reports 96 cores")
    assert "at most 23 sandboxed children" in row["note"]
