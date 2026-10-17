"""An instrument's usage and its run output, for the agent that runs it (B8).

Registering probes the entry's `--help` (in the run's own minimal environment and snapshot) and keeps the redacted
text in the content store, its sha256 in the manifest, so it is part of the version. `instrument show` prints it.
`instrument run` returns its exit code, the redacted stdout and stderr tails and each declared output's content (text
and small enough), and `instrument output RUN [FILE]` reads a recorded run back, each sha checked first."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from openlab.study import instruments as I
from openlab.study.errors import StudyError

CLI = '''import argparse, json, os, sys
p = argparse.ArgumentParser(prog="tcli", description="a test broker CLI")
p.add_argument("--account", default="DU0000000", help="the account (default DU0000000)")
sub = p.add_subparsers(dest="cmd", required=True)
s = sub.add_parser("positions", help="list the positions")
s.add_argument("--min-qty", type=int, default=0, help="only positions of at least this many shares")
o = sub.add_parser("orders", help="list the open orders")
o.add_argument("--days", type=int, default=1)
a = p.parse_args()
if a.cmd == "positions":
    print("2 positions")
    print("token", os.environ.get("API_TOKEN", ""), file=sys.stderr)
    json.dump([{"sym": "F", "qty": 99, "acct": "DU0000000", "tok": os.environ.get("API_TOKEN", "")}],
              open("positions.json", "w"))
    open("big.txt", "w").write("x" * (300 * 1024))
    open("blob.bin", "wb").write(bytes(range(256)) * 4)
    for i in range(500):
        print("line", i)
'''


@pytest.fixture(scope="session")
def tiny_python(tmp_path_factory):
    d = tmp_path_factory.mktemp("venv")
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(d / "v")], check=True)
    return str(d / "v" / "bin" / "python")


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "openlab.toml").write_text("# a test workspace\n")
    return root


def _pkg(root: Path, body: str = CLI) -> Path:
    src = root / "tcli_src"
    (src / "tcli").mkdir(parents=True)
    (src / "tcli" / "__init__.py").write_text("")
    (src / "tcli" / "__main__.py").write_text(body)
    return src


def _script(root: Path, name: str, body: str) -> Path:
    src = root / f"{name}_src"
    src.mkdir()
    (src / "run.sh").write_text("#!/bin/sh\n" + body)
    (src / "run.sh").chmod(0o755)
    return src


def _register(ws, person, name, entry, **kw):
    with person.act():
        return I.register(ws, name, entry, reason=kw.pop("reason", "a test instrument"), **kw)


def _tcli(ws, person, tiny_python, **kw):
    src = _pkg(ws)
    return _register(ws, person, "tcli", f"{tiny_python} -m tcli", source=str(src),
                     outputs="positions.json,big.txt,blob.bin", secrets="API_TOKEN",
                     redact=[r"\bDU\d{5,9}\b"], **kw)["payload"]


# --- usage ---------------------------------------------------------------------------------------------------------
def test_usage_is_probed_at_register_kept_in_the_store_and_part_of_the_version(ws, tiny_python, person_key,
                                                                              monkeypatch):
    monkeypatch.setenv("API_TOKEN", "tok-s3cr3t")
    src = _pkg(ws)
    p = I.prepare(ws, "tcli", f"{tiny_python} -m tcli", source=str(src), secrets="API_TOKEN",
                  redact=[r"\bDU\d{5,9}\b"])
    u = p.manifest["usage"]
    assert u["status"] == "ok" and u["args"] == ["--help"] and u["sha256"] == I.sha_bytes(p.usage_text.encode())
    assert "usage: tcli" in p.usage_text and "{positions,orders}" in p.usage_text
    assert "DU0000000" not in p.usage_text and "[redacted]" in p.usage_text          # the redactions apply
    preview = "\n".join(I.preview(p))
    assert "usage (--help" in preview and "{positions,orders}" in preview            # shown before the passphrase
    other = {**p.manifest, "usage": {**u, "sha256": "0" * 64}}
    assert I.manifest_sha(other) != p.sha                                             # the usage is in the version
    with person_key.act():
        node = I.commit(ws, p, "a test")
    reg = I.Registry(ws)
    assert reg.blob(node.payload["usage"]["sha256"]).decode() == p.usage_text
    assert I.verify(reg, node)["ok"]
    view = I.usage_view(reg, node.payload)
    assert view["status"] == "ok" and "{positions,orders}" in view["text"]


def test_the_subcommand_probe_adds_each_subcommands_help(ws, tiny_python, person_key):
    m = _register(ws, person_key, "tcli", f"{tiny_python} -m tcli", source=str(_pkg(ws)), usage_sub=True)["payload"]
    u = m["usage"]
    assert u["subcommands"] == ["positions", "orders"]
    text = I.Registry(ws).blob(u["sha256"]).decode()
    assert "--min-qty" in text and "--days" in text and "tcli positions --help" in text
    assert I.parse_subcommands("usage: x [-h] [--runtime {a,b}] {run,list} ...") == ["run", "list"]


def test_a_usage_probe_failure_is_recorded_not_refused(ws, person_key, monkeypatch):
    m = _register(ws, person_key, "fails", "./run.sh", source=str(_script(ws, "fails", 'echo "no help here"\nexit 3\n')))
    u = m["payload"]["usage"]
    assert u["status"] == "unavailable" and u["reason"] == "exit 3"
    monkeypatch.setattr(I, "USAGE_TIMEOUT", 0.5)
    m = _register(ws, person_key, "hangs", "./run.sh", source=str(_script(ws, "hangs", "sleep 30\n")))
    u = m["payload"]["usage"]
    assert u["status"] == "unavailable" and "timed out" in u["reason"]
    from openlab.study import instrument_cli
    assert instrument_cli.main(["show", "fails", "--root", str(ws)]) == 0


def test_the_usage_probe_runs_in_the_run_environment_with_secrets_masked(ws, person_key, monkeypatch):
    monkeypatch.setenv("API_TOKEN", "tok-s3cr3t")
    monkeypatch.setenv("UNRELATED_VAR", "leaky")
    src = _script(ws, "envp", 'echo "token=$API_TOKEN other=$UNRELATED_VAR snap=${OPENLAB_INSTRUMENT_SNAPSHOT:+set}"\n'
                  'touch "$OPENLAB_INSTRUMENT_SNAPSHOT/x" 2>/dev/null && echo writable || echo read-only\n')
    m = _register(ws, person_key, "envp", "./run.sh", source=str(src), secrets="API_TOKEN")["payload"]
    text = I.Registry(ws).blob(m["usage"]["sha256"]).decode()
    assert text == "token=[secret API_TOKEN] other= snap=set\nread-only\n"


def test_usage_options_skip_supply_and_choose_the_argument(ws, person_key, tmp_path, capsys):
    src = _script(ws, "helpcmd", 'if [ "$1" = help ]; then echo "commands: a b"; else echo "ran $1"; fi\n')
    m = _register(ws, person_key, "helpcmd", "./run.sh", source=str(src), usage_args=["help"])["payload"]
    assert m["usage"]["args"] == ["help"]
    assert I.Registry(ws).blob(m["usage"]["sha256"]).decode() == "commands: a b\n"
    (src / "run.sh").write_text("#!/bin/sh\necho changed\n")
    m2 = _register(ws, person_key, "helpcmd", "./run.sh", source=str(src), usage=False)["payload"]
    assert m2["usage"]["status"] == "skipped"
    doc = tmp_path / "usage.txt"
    doc.write_text("helpcmd: prints what it is told\n")
    (src / "run.sh").write_text("#!/bin/sh\necho changed again\n")
    m3 = _register(ws, person_key, "helpcmd", "./run.sh", source=str(src), usage_file=str(doc))["payload"]
    assert m3["usage"]["status"] == "supplied"
    from openlab.study import instrument_cli
    assert instrument_cli.main(["show", I.ident(m3), "--usage", "--root", str(ws)]) == 0
    assert capsys.readouterr().out == "helpcmd: prints what it is told\n"
    assert instrument_cli.main(["show", I.ident(m2), "--root", str(ws)]) == 0
    assert "usage: not captured (--no-usage)" in capsys.readouterr().out


def test_a_registration_from_before_usage_capture_still_loads_verifies_and_shows(ws, tiny_python, person_key,
                                                                                 capsys):
    from openlab.study.record import canonical
    p = I.prepare(ws, "tcli", f"{tiny_python} -m tcli", source=str(_pkg(ws)), usage=False)
    old = {k: p.manifest[k] for k in I.MANIFEST_KEYS}                       # the manifest as 1.3.0 wrote it
    sha = I.sha_bytes(canonical(old).encode())
    assert I.manifest_sha(old) == sha == I.manifest_sha({**old, "usage": {}})
    p.manifest.pop("usage")
    p.sha = sha
    with person_key.act():
        node = I.commit(ws, p, "an old registration")
    reg = I.Registry(ws)
    assert node.payload["version"] == sha[:12] and I.verify(reg, node)["ok"]
    assert I.run(ws, I.ident(node.payload), ["orders"])["exit"] == 0
    from openlab.study import instrument_cli
    assert instrument_cli.main(["show", I.ident(node.payload), "--root", str(ws)]) == 0
    assert "usage: not recorded (registered before usage capture)" in capsys.readouterr().out
    assert instrument_cli.main(["show", I.ident(node.payload), "--usage", "--root", str(ws)]) == 0
    assert "not recorded" in capsys.readouterr().out


# --- run output -----------------------------------------------------------------------------------------------------
def test_a_run_returns_its_exit_its_log_tails_and_its_outputs_content(ws, tiny_python, person_key, monkeypatch):
    monkeypatch.setenv("API_TOKEN", "tok-s3cr3t")
    m = _tcli(ws, person_key, tiny_python)
    r = I.run(ws, I.ident(m), ["positions"])
    assert r["exit"] == 0
    out = r["stdout"]
    assert out["text"].startswith("line 300\n") and out["text"].endswith("line 499\n")       # the last 200 lines
    assert out["truncated"] and out["lines"] == 501
    assert "[secret API_TOKEN]" in r["stderr"]["text"]
    pos = r["outputs"]["positions.json"]
    assert json.loads(pos["content"])[0]["qty"] == 99 and pos["sha256"] == r["payload"]["outputs"][
        "positions.json"]["sha256"]
    big = r["outputs"]["big.txt"]
    assert "content" not in big and big["bytes"] == 300 * 1024 and "too large; materialise it in a study" in big[
        "withheld"]
    assert "content" not in r["outputs"]["blob.bin"] and "not text" in r["outputs"]["blob.bin"]["withheld"]
    dumped = json.dumps(r)
    assert "tok-s3cr3t" not in dumped and "DU0000000" not in dumped               # masked in all returned text
    reg = I.Registry(ws)
    assert b"tok-s3cr3t" in reg.blob(pos["sha256"])                              # the output itself is kept as written


def test_a_long_line_log_is_capped_by_bytes(ws, person_key):
    src = _script(ws, "wide", 'i=0; while [ $i -lt 100 ]; do printf "%01000d\\n" $i; i=$((i+1)); done\n')
    m = _register(ws, person_key, "wide", "./run.sh", source=str(src), usage=False)["payload"]
    r = I.run(ws, I.ident(m))
    assert len(r["stdout"]["text"].encode()) <= I.TAIL_BYTES and r["stdout"]["truncated"]
    assert r["stdout"]["text"].endswith("99\n")


def test_instrument_output_reads_a_recorded_run_back_and_refuses_a_tampered_blob(ws, tiny_python, person_key,
                                                                                monkeypatch, capsys):
    monkeypatch.setenv("API_TOKEN", "tok-s3cr3t")
    m = _tcli(ws, person_key, tiny_python)
    r = I.run(ws, I.ident(m), ["positions"])
    monkeypatch.delenv("API_TOKEN")                                              # a reader without the secret
    o = I.output(ws, r["id"])
    assert o["exit"] == 0 and o["stdout"]["text"].endswith("line 499\n")
    assert json.loads(o["outputs"]["positions.json"]["content"])[0]["tok"] == "[secret API_TOKEN]"
    f = I.output(ws, r["id"], "positions.json")
    assert "tok-s3cr3t" not in f["content"] and "DU0000000" not in f["content"]
    assert I.output(ws, r["id"], "stdout")["content"].startswith("2 positions\nline 0\n")
    with pytest.raises(StudyError, match="not text"):
        I.output(ws, r["id"], "blob.bin")
    with pytest.raises(StudyError, match="no output"):
        I.output(ws, r["id"], "nope.json")
    with pytest.raises(StudyError, match="no instrument run"):
        I.output(ws, m["name"] + "-not-a-run")
    from openlab.study import instrument_cli
    root = ["--root", str(ws)]
    assert instrument_cli.main(["output", r["id"], "positions.json", *root]) == 0
    assert json.loads(capsys.readouterr().out)[0]["qty"] == 99
    assert instrument_cli.main(["output", r["id"], *root]) == 0
    text = capsys.readouterr().out
    assert "2 positions" not in text and "line 499" in text and "too large; materialise it in a study" in text
    reg = I.Registry(ws)
    (reg.dir / "blobs" / r["payload"]["stdout"]).write_text("forged\n")
    with pytest.raises(StudyError, match="was changed"):
        I.output(ws, r["id"])
    assert instrument_cli.main(["output", r["id"], *root]) == 1
    assert "was changed" in capsys.readouterr().err


def test_the_cli_and_the_tools_print_usage_and_run_output(ws, tiny_python, person_key, monkeypatch, capsys):
    from openlab.app import ToolContext, dispatch
    from openlab.study import instrument_cli
    monkeypatch.setenv("API_TOKEN", "tok-s3cr3t")
    m = _tcli(ws, person_key, tiny_python)
    root = ["--root", str(ws)]
    assert instrument_cli.main(["show", I.ident(m), *root]) == 0
    shown = capsys.readouterr().out
    assert "usage (--help" in shown and "{positions,orders}" in shown
    assert instrument_cli.main(["run", I.ident(m), *root, "--", "positions"]) == 0
    ran = capsys.readouterr().out
    assert "exit 0" in ran and "line 499" in ran and '"qty": 99' in ran and "tok-s3cr3t" not in ran
    ctx = ToolContext(str(ws))
    env = dispatch(ctx, "instrument_show", {"ref": I.ident(m)})
    assert env["ok"] and "{positions,orders}" in env["result"]["usage"]["text"]
    env = dispatch(ctx, "instrument_show", {"ref": I.ident(m), "usage_only": True})
    assert env["ok"] and set(env["result"]) == {"instrument", "usage"}
    env = dispatch(ctx, "instrument_run", {"ref": I.ident(m), "args": ["positions"]})
    assert env["ok"] and env["result"]["exit"] == 0 and "line 499" in env["result"]["stdout"]["text"]
    assert "tok-s3cr3t" not in json.dumps(env)
    env = dispatch(ctx, "instrument_output", {"run": env["result"]["id"], "file": "positions.json"})
    assert env["ok"] and json.loads(env["result"]["content"])[0]["qty"] == 99


def test_the_hooks_let_an_agent_show_read_and_run_but_not_register():
    from openlab.agents import hooks
    for cmd in ("openlab instrument show tcli@0123456789ab", "openlab instrument show tcli@0123456789ab --usage",
                "openlab instrument output 0123abcd", "openlab instrument output 0123abcd positions.json",
                "openlab instrument runs", "openlab instrument run tcli@0123456789ab -- positions"):
        assert hooks.guard("Bash", {"command": cmd}) is None, cmd
    g = hooks.guard("Bash", {"command": "openlab instrument register x --entry 'mycli' --reason r"})
    assert g and g[0] == "bypass"
