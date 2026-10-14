"""The front door: `openlab --help` lists the research commands, a study command loads no LaTeX engine, and where
the data roots resolve."""

import subprocess
import sys

import pytest

from openlab.__main__ import COMMANDS, main


def test_help_lists_the_research_commands_and_nothing_old(capsys):
    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    for name in ("study", "init", "app", "tool", "tools", "guide", "doctor", "verify"):
        assert f"  {name} " in out, name
    for gone in ("submit", "remote", "sanity", "modules"):
        assert gone not in COMMANDS
    assert main(["submit"]) == 2 and "no command 'submit'" in capsys.readouterr().err


def test_an_unknown_command_says_so_in_two_lines_with_the_nearest_name(capsys):
    assert main(["studdy"]) == 2
    err = capsys.readouterr().err
    assert err.splitlines() == ["openlab: no command 'studdy'; did you mean 'study'?",
                                "`openlab --help` lists the commands"]
    assert main(["frobnicate"]) == 2
    assert capsys.readouterr().err.splitlines()[0] == "openlab: no command 'frobnicate'"


def test_init_writes_the_conventions_the_workspace_and_the_mcp_entry(tmp_path):
    pytest.importorskip("fastapi")
    from openlab.init import init
    r = init(tmp_path)
    assert {"OPENLAB.md", "openlab.toml", ".mcp.json"} <= set(r["wrote"])
    assert "[agents]" in (tmp_path / "openlab.toml").read_text()
    assert not (tmp_path / "LAWS.openlab").exists() and not (tmp_path / "HOOKS.openlab").exists()
    assert set(init(tmp_path)["kept"]) >= {"OPENLAB.md", "openlab.toml"}


def test_init_tells_a_person_what_it_did_and_a_script_gets_json(tmp_path, capsys, monkeypatch):
    import json
    monkeypatch.delenv("OPENLAB_HOOK_COMMAND", raising=False)
    assert main(["init", str(tmp_path), "--claude"]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"openlab workspace at {tmp_path.resolve()}") and "{" not in out.splitlines()[0]
    assert "wrote: OPENLAB.md, openlab.toml, .mcp.json, .claude/settings.json" in out
    assert ".claude/agents/ (openlab-lead" in out and out.splitlines()[-1].startswith("next: ")
    assert main(["init", str(tmp_path), "--claude", "--json"]) == 0
    r = json.loads(capsys.readouterr().out)
    assert {"OPENLAB.md", "openlab.toml", ".claude/settings.json"} <= set(r["kept"]) | set(r["updated"])
    assert len(r["kept"]) == len(set(r["kept"])) and r["next"]


def test_every_command_resolves_to_a_function():
    import importlib
    for name, (target, what) in COMMANDS.items():
        mod, _, fn = target.partition(":")
        assert callable(getattr(importlib.import_module(mod), fn.rstrip("+"))), name
        assert what, name


def test_a_study_command_loads_neither_the_engine_nor_the_server():
    code = ("import sys\nfrom openlab.__main__ import main\n"
            "try:\n    main(['study', '--help'])\nexcept SystemExit:\n    pass\n"
            "print(sorted(m for m in ('sympy', 'lark', 'fastapi', 'openlab.engine') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120).stdout
    assert out.strip().splitlines()[-1] == "[]"


def test_verify_refuses_what_is_not_a_signed_file(tmp_path, capsys, monkeypatch):
    pytest.importorskip("nacl")
    monkeypatch.setenv("OPENLAB_SIGNING_KEY", str(tmp_path / "key"))
    assert main(["verify", str(tmp_path / "report.html")]) == 2
    assert ".sig.json" in capsys.readouterr().err
    data = tmp_path / "f.csv"
    data.write_text("a\n1\n")
    from openlab.study.signing import sign_file
    sig = sign_file(data, {"file": "f.csv", "study": "s"})
    assert main(["verify", str(sig)]) == 0 and "valid" in capsys.readouterr().out
    data.write_text("a\n2\n")
    assert main(["verify", str(sig)]) == 1 and "changed after it was signed" in capsys.readouterr().out


def test_data_roots_and_the_sec_contact(monkeypatch, tmp_path):
    from openlab.sources import http, snapshot
    monkeypatch.delenv("OPENLAB_SNAPSHOTS", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert snapshot.root() == tmp_path / "openlab-data" / "snapshots"
    monkeypatch.setenv("OPENLAB_SNAPSHOTS", str(tmp_path / "s"))
    assert snapshot.root() == tmp_path / "s"
    monkeypatch.delenv("OPENLAB_SEC_CONTACT", raising=False)
    with pytest.raises(RuntimeError, match="OPENLAB_SEC_CONTACT"):
        http._sec_agent()
