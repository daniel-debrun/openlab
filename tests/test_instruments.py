"""Instruments (openlab.study.instruments): registered, immutable programs that measure or act outside the sandbox.

A registration is the person's signed act; a version is the sha of everything it runs with (its snapshot, entry,
settings and environment fingerprint); a run is refused when any of it drifted; every run is recorded with its
hashes; a study runs and reads only the versions it pins; a live-order instrument runs only as the person."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from openlab.study import instruments as I
from openlab.study.errors import StudyError

PY_MAIN = '''import json, os, sys
print("hello from", os.environ["OPENLAB_INSTRUMENT"], "v1", " ".join(sys.argv[1:]))
json.dump([{"sym": "F", "qty": 99}, {"sym": "T", "qty": 3}], open("out.json", "w"))
'''


@pytest.fixture(scope="session")
def tiny_python(tmp_path_factory):
    """A venv with no distributions, so its fingerprint is quick."""
    d = tmp_path_factory.mktemp("venv")
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(d / "v")], check=True)
    return str(d / "v" / "bin" / "python")


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "openlab.toml").write_text("# a test workspace\n")
    return root


def _pkg(root: Path, body: str = PY_MAIN) -> Path:
    src = root / "tinst_src"
    (src / "tinst").mkdir(parents=True)
    (src / "tinst" / "__init__.py").write_text("")
    (src / "tinst" / "__main__.py").write_text(body)
    (src / ".venv").mkdir()                                     # never snapshotted
    (src / ".venv" / "junk.txt").write_text("x")
    (src / "state").mkdir()
    (src / "state" / "cache.json").write_text("{}")
    return src


def _register(ws, person, name, entry, **kw):
    with person.act():
        return I.register(ws, name, entry, reason=kw.pop("reason", "a test instrument"), **kw)


def _writable(p: Path) -> None:
    p.parent.chmod(p.parent.stat().st_mode | stat.S_IWUSR)
    p.chmod(p.stat().st_mode | stat.S_IWUSR)


# --- registration is the person's --------------------------------------------------------------------------------
def test_registering_without_the_person_key_is_refused(ws, tiny_python, person_key):
    src = _pkg(ws)
    p = I.prepare(ws, "tinst", f"{tiny_python} -m tinst", source=str(src), outputs="out.json")
    with pytest.raises(StudyError, match="person's act"):
        I.commit(ws, p, "no key")
    reg = I.Registry(ws)
    payload = {**{k: p.manifest[k] for k in I.MANIFEST_KEYS}, "version": p.version, "sha": p.sha,
               "source": p.source, "reason": "typed by an agent"}
    with pytest.raises(StudyError, match="can only be written by person"):
        reg.add("instrument", payload, actor="agent")
    assert not reg.find("instrument")


def test_hooks_refuse_an_agent_registering_and_the_inbox_takes_the_exact_registration(ws):
    from openlab.agents import hooks
    from openlab.agents.inbox import RequestError, check_request
    from openlab.agents.store import Store
    g = hooks.guard("Bash", {"command": "openlab instrument register x --entry 'mycli' --reason r"})
    assert g and g[0] == "bypass" and "registering an instrument" in g[1]
    assert hooks.guard("Bash", {"command": "openlab instrument run x@0123456789ab -- positions"}) is None
    store = Store(ws)
    with pytest.raises(RequestError):
        check_request(store, "a1", "instrument", {"name": "x"})
    with pytest.raises(RequestError):
        check_request(store, "a1", "instrument", {"name": "x", "entry": "y", "sudo": True})
    assert check_request(store, "a1", "instrument", {"name": "x", "entry": "y"})["entry"] == "y"
    with pytest.raises(RequestError):
        check_request(store, "a1", "instrument_run", {"args": ["a"]})


def test_the_inbox_approval_registers_it_as_the_person(ws, tiny_python, person_key):
    from openlab.agents.inbox import decide
    from openlab.agents.store import Store
    src = _pkg(ws)
    store = Store(ws)
    item = store.request("a1", "instrument", "register the test instrument",
                         args={"name": "tinst", "entry": f"{tiny_python} -m tinst", "source": str(src),
                               "outputs": "out.json"})
    with person_key.act():
        done = decide(store, item, True, "fine")
    assert done["status"] == "approved" and done["result"]["summary"].startswith("registered tinst@")
    node = I.Registry(ws).find("instrument")[0]
    assert node.actor == "person" and node.payload["sig"]["key"] == person_key.signer.fingerprint


# --- a python -m instrument: snapshot, run, hashes ------------------------------------------------------------------
def test_a_python_module_instrument_runs_from_its_snapshot_and_records_its_hashes(ws, tiny_python, person_key):
    src = _pkg(ws)
    out = _register(ws, person_key, "tinst", f"{tiny_python} -m tinst", source=str(src), outputs="out.json")
    m = out["payload"]
    assert m["runtime"] == "python" and m["entry"][:2] == [tiny_python, "-m"]
    assert set(m["files"]) == {"tinst/__init__.py", "tinst/__main__.py"}        # no .venv, no state
    assert out["payload"]["version"] == I.manifest_sha(m)[:12]
    (src / "tinst" / "__main__.py").write_text(PY_MAIN.replace("v1", "v2"))      # the live folder moves on
    r = I.run(ws, "tinst", ["--x", "1"])
    p = r["payload"]
    assert p["exit"] == 0 and p["args"] == ["--x", "1"]
    reg = I.Registry(ws)
    stdout = reg.blob(p["stdout"]).decode()
    assert f"hello from {I.ident(m)} v1 --x 1" in stdout                          # the snapshot ran, not the folder
    data = reg.blob(p["outputs"]["out.json"]["sha256"])
    assert json.loads(data)[0] == {"sym": "F", "qty": 99}
    assert p["environment"] == m["environment"]["sha"] and r["kind"] == "instrument_run"
    assert reg.verify()["ok"]
    assert I.runs(reg, "tinst")[0]["id"] == r["id"]


def test_an_interpreter_inside_the_source_is_the_environment_by_its_absolute_path(ws, tiny_python, person_key,
                                                                                 monkeypatch, tmp_path):
    src = _pkg(ws)
    (src / ".venv" / "bin").mkdir()
    os.symlink(tiny_python, src / ".venv" / "bin" / "python")                  # a venv's python links out of it
    m = _register(ws, person_key, "tinst", ".venv/bin/python -m tinst", source=str(src))["payload"]
    assert m["entry"][0] == str(src / ".venv" / "bin" / "python")
    assert m["environment"]["parts"][0]["spec"]["path"] == str(src / ".venv" / "bin" / "python")
    monkeypatch.chdir(tmp_path)                                                 # checked from anywhere
    assert I.run(ws, "tinst")["payload"]["exit"] == 0


def test_the_same_registration_is_one_version_and_a_change_is_a_new_one(ws, tiny_python, person_key):
    src = _pkg(ws)
    a = _register(ws, person_key, "tinst", f"{tiny_python} -m tinst", source=str(src))
    with pytest.raises(StudyError, match="already registered"):
        _register(ws, person_key, "tinst", f"{tiny_python} -m tinst", source=str(src), reason="again")
    (src / "tinst" / "__main__.py").write_text(PY_MAIN.replace("v1", "v2"))
    b = _register(ws, person_key, "tinst", f"{tiny_python} -m tinst", source=str(src))
    assert a["payload"]["version"] != b["payload"]["version"]
    with pytest.raises(StudyError, match="2 versions"):
        I.run(ws, "tinst")
    assert I.run(ws, I.ident(a["payload"]))["payload"]["exit"] == 0


# --- immutability ---------------------------------------------------------------------------------------------------
def test_a_changed_snapshot_refuses_the_run_and_the_refusal_is_recorded(ws, tiny_python, person_key):
    src = _pkg(ws)
    m = _register(ws, person_key, "tinst", f"{tiny_python} -m tinst", source=str(src))["payload"]
    reg = I.Registry(ws)
    f = reg.snapshot_dir(I.ident(m)) / "tinst" / "__main__.py"
    _writable(f)
    f.write_text("print('tampered')\n")
    with pytest.raises(StudyError, match="tinst/__main__.py: changed"):
        I.run(ws, I.ident(m))
    assert reg.find("action")[-1].payload["outcome"]["ok"] is False
    assert not reg.find("instrument_run")
    I._writable(reg.snapshot_dir(I.ident(m)))
    shutil.rmtree(reg.snapshot_dir(I.ident(m)))                                  # gone: rebuilt from the store
    v = I.verify(reg, I.resolve(reg, I.ident(m)))
    assert v["ok"] and v["rebuilt"]


def test_an_environment_change_refuses_the_run_and_names_it(ws, tiny_python, person_key, tmp_path):
    src = _pkg(ws)
    lock = tmp_path / "requirements.lock"
    lock.write_text("polars==1.0\n")
    m = _register(ws, person_key, "tinst", f"{tiny_python} -m tinst", source=str(src),
                  env_files=[str(lock)])["payload"]
    site = next(Path(tiny_python).parent.parent.glob("lib/python*/site-packages"))
    info = site / "fakepkg-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: fakepkg\nVersion: 1.0\n")
    (info / "RECORD").write_text("")
    try:
        with pytest.raises(StudyError, match="fakepkg 1.0 was installed"):
            I.run(ws, I.ident(m))
    finally:
        shutil.rmtree(info)
    assert I.run(ws, I.ident(m))["payload"]["exit"] == 0                         # back as registered
    lock.write_text("polars==1.1\n")
    with pytest.raises(StudyError, match="requirements.lock"):
        I.run(ws, I.ident(m))


# --- any command: a shell script, node, a bare binary -----------------------------------------------------------------
def test_a_shell_script_instrument_keeps_state_and_fingerprints_its_interpreter(ws, person_key):
    src = ws / "sh_src"
    src.mkdir()
    script = src / "collect.sh"
    script.write_text('#!/bin/sh\necho "run $1"\nn=$(cat "$OPENLAB_INSTRUMENT_STATE/n" 2>/dev/null || echo 0)\n'
                      'echo $((n + 1)) > "$OPENLAB_INSTRUMENT_STATE/n"\necho "rows=$((n + 1))" > rows.txt\n')
    script.chmod(0o755)
    m = _register(ws, person_key, "collector", "./collect.sh", source=str(src), state=True,
                  outputs="rows.txt")["payload"]
    assert m["entry"] == ["{snapshot}/collect.sh"] and m["runtime"] == "binary"
    assert [p["spec"]["path"] for p in m["environment"]["parts"]] == ["/bin/sh"]
    I.run(ws, "collector", ["a"])
    r = I.run(ws, "collector", ["b"])
    reg = I.Registry(ws)
    assert reg.blob(r["payload"]["outputs"]["rows.txt"]["sha256"]) == b"rows=2\n"
    assert "run b" in reg.blob(r["payload"]["stdout"]).decode()


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed here")
def test_a_node_instrument(ws, person_key):
    src = ws / "node_src"
    src.mkdir()
    (src / "run.js").write_text('require("fs").writeFileSync("o.json", JSON.stringify([{a: 1}]));'
                                'console.log("node ok", process.argv.slice(2).join(" "));\n')
    m = _register(ws, person_key, "nodeinst", "node run.js", source=str(src), outputs="o.json")["payload"]
    assert m["runtime"] == "node" and m["entry"] == ["node", "{snapshot}/run.js"]
    r = I.run(ws, "nodeinst", ["x"])
    assert "node ok x" in I.Registry(ws).blob(r["payload"]["stdout"]).decode()


def test_a_bare_binary_instrument_is_verified_never_copied(ws, person_key, tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "myecho"
    shutil.copy("/bin/echo", exe)
    m = _register(ws, person_key, "echoer", str(exe), version_probe=False)["payload"]
    assert m["files"] == {} and m["runtime"] == "binary"
    reg = I.Registry(ws)
    assert not reg.snapshot_dir(I.ident(m)).exists()
    r = I.run(ws, "echoer", ["hi", "there"])
    assert reg.blob(r["payload"]["stdout"]) == b"hi there\n"
    assert I.verify(reg, I.resolve(reg, "echoer"))["external"]
    shutil.copy("/bin/true", exe)                                                # another program at the same path
    with pytest.raises(StudyError, match="sha256"):
        I.run(ws, "echoer")


def test_secrets_pass_by_name_and_never_reach_the_record(ws, person_key, tmp_path, monkeypatch):
    src = ws / "sec_src"
    src.mkdir()
    (src / "s.sh").write_text('#!/bin/sh\necho "token=$API_TOKEN acct DU1234567"\n')
    (src / "s.sh").chmod(0o755)
    m = _register(ws, person_key, "sec", "./s.sh", source=str(src), secrets="API_TOKEN",
                  redact=[r"\bDU\d{5,9}\b"])["payload"]
    monkeypatch.delenv("API_TOKEN", raising=False)
    with pytest.raises(StudyError, match="secret API_TOKEN is not set"):
        I.run(ws, I.ident(m))
    monkeypatch.setenv("API_TOKEN", "s3cr3t-value")
    r = I.run(ws, I.ident(m))
    reg = I.Registry(ws)
    text = reg.blob(r["payload"]["stdout"]).decode()
    assert "s3cr3t" not in text and "[secret API_TOKEN]" in text and "DU1234567" not in text
    assert "s3cr3t" not in (reg.dir / r["payload"]["run_dir"].split("instruments/", 1)[1] / "stdout.log").read_text()
    assert "s3cr3t" not in json.dumps(r)


# --- live orders are the person's, one authority per run ---------------------------------------------------------------
def test_a_live_order_instrument_runs_only_as_the_person(ws, person_key, tmp_path):
    from openlab.app import ToolContext, dispatch
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "broker"
    shutil.copy("/bin/echo", exe)
    _register(ws, person_key, "broker", str(exe), capabilities="read,live-order", version_probe=False)
    with pytest.raises(StudyError, match="each run is a person's act"):
        I.run(ws, "broker", ["buy", "1"])
    env = dispatch(ToolContext(str(ws)), "instrument_run", {"ref": "broker", "args": ["buy", "1"]})
    assert not env["ok"] and "person's act" in env["error"]
    reg = I.Registry(ws)
    assert not reg.find("instrument_run")
    with person_key.act():
        with pytest.raises(StudyError, match="needs a reason"):
            I.run(ws, "broker", ["buy", "1"])
        r = I.run(ws, "broker", ["buy", "1"], reason="one share, by hand")
    assert r["actor"] == "person" and r["authority"]
    auth = reg.get(r["authority"])
    assert auth.kind == "instrument_authority" and auth.payload["args"] == ["buy", "1"]
    assert reg.links(r["id"], rel="under") == [(r["id"], "under", auth.id)]
    payload = {**r["payload"]}
    payload.pop("sig", None)
    with pytest.raises(StudyError):                                              # an agent cannot record one
        reg.add("instrument_run", payload, actor="agent", links=[("under", auth.id)])
    with person_key.act():                                                       # nor reuse the authority
        with pytest.raises(StudyError, match="already allowed a run"):
            reg.add("instrument_run", payload, actor="person", links=[("under", auth.id)])
    assert reg.verify()["ok"]


# --- studies pin versions; datasets read their recorded outputs ---------------------------------------------------------
def _study(ws, pin: str, alias: str = "paper"):
    from openlab import study
    s = study.new(ws / "tender", "temporal")
    with open(s.folder / "study.toml", "a") as f:
        f.write(f'\n[instruments]\n{alias} = "{pin}"\n\n[data.pos]\ninstrument = "{alias}"\noutput = "out.json"\n'
                f'key = ["sym"]\n')
    return study.open(s.folder)


def test_a_study_pins_its_versions_and_only_a_person_changes_the_pin(ws, tiny_python, person_key):
    src = _pkg(ws)
    a = I.ident(_register(ws, person_key, "tinst", f"{tiny_python} -m tinst", source=str(src),
                          outputs="out.json")["payload"])
    st = _study(ws, a)
    st.seal()
    (src / "tinst" / "__main__.py").write_text(PY_MAIN.replace("v1", "v2"))
    b = I.ident(_register(ws, person_key, "tinst", f"{tiny_python} -m tinst", source=str(src),
                          outputs="out.json")["payload"])
    with pytest.raises(StudyError, match="does not pin"):
        I.run(ws, b, study="tender")
    r = I.run(ws, "paper", study="tender")                                       # by alias, the pinned version
    assert r["payload"]["instrument"] == a and r["study_run"]
    assert st.record.get(r["study_run"]).payload["registry_run"] == r["id"]
    toml = st.folder / "study.toml"
    toml.write_text(toml.read_text().replace(a, b))
    with pytest.raises(StudyError, match="instruments.*can only be amended by a person"):
        st.amend("move to the new version")
    with person_key.act():
        st.amend("move to the new version", person=True)
    assert I.run(ws, "paper", study="tender")["payload"]["instrument"] == b


def test_an_instrument_dataset_materialises_from_the_recorded_run_with_its_hash(ws, tiny_python, person_key):
    src = _pkg(ws)
    pin = I.ident(_register(ws, person_key, "tinst", f"{tiny_python} -m tinst", source=str(src),
                            outputs="out.json")["payload"])
    st = _study(ws, pin)
    with pytest.raises(StudyError, match="no successful run"):
        st.materialise("pos")
    r = I.run(ws, "paper", study="tender")
    node = st.materialise("pos")
    src_ = node["payload"]["source"]
    assert src_["kind"] == "instrument" and src_["pin"] == pin and src_["run"] == r["id"]
    assert src_["output_sha256"] == r["payload"]["outputs"]["out.json"]["sha256"]
    assert node["payload"]["rows"] == 2
    reg = I.Registry(ws)
    blob = reg.dir / "blobs" / src_["output_sha256"]
    blob.write_bytes(b'[{"sym": "X", "qty": 1}]')
    with pytest.raises(StudyError, match="was changed"):
        I.dataset_output(st.folder, pin, "out.json")


def test_the_study_toml_checks_pins_and_instrument_datasets(ws):
    from openlab.study import spec
    raw = {"study": {"name": "t", "question": "q", "mode": "temporal"},
           "split": {"scheme": "walk_forward"}, "evaluate": {"metrics": ["mae"]}, "budget": {"trials": 1},
           "instruments": {"paper": "tinst@0123456789ab"},
           "data": {"pos": {"instrument": "paper", "output": "out.json"}}}
    s = spec.build(raw, ws)
    assert s.instruments == {"paper": "tinst@0123456789ab"} and s.data["pos"].pin == "tinst@0123456789ab"
    for bad, why in ((dict(raw, instruments={"paper": "tinst"}), "NAME@VERSION"),
                     (dict(raw, data={"pos": {"instrument": "nope", "output": "o.json"}}), "does not pin"),
                     (dict(raw, data={"pos": {"instrument": "paper"}}), "output"),
                     (dict(raw, data={"pos": {"file": "a.csv", "recipe": "r.py", "output": "o"}}), "instrument")):
        with pytest.raises(StudyError, match=why):
            spec.build(bad, ws)
    assert "instruments" in spec.PERSON_FIELDS and "data.*.instrument" in spec.PERSON_FIELDS


def test_the_cli_lists_shows_verifies_and_runs(ws, tiny_python, person_key, capsys):
    from openlab.study import instrument_cli
    src = _pkg(ws)
    m = _register(ws, person_key, "tinst", f"{tiny_python} -m tinst", source=str(src), outputs="out.json")["payload"]
    root = ["--root", str(ws)]
    assert instrument_cli.main(["list", *root]) == 0
    assert I.ident(m) in capsys.readouterr().out
    assert instrument_cli.main(["verify", I.ident(m), *root]) == 0
    assert instrument_cli.main(["run", "tinst", *root, "--", "a", "b"]) == 0
    assert "exit 0" in capsys.readouterr().out
    assert instrument_cli.main(["runs", *root, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["runs"][0]["payload"]["args"] == ["a", "b"]
    assert instrument_cli.main(["register", "other", "--entry", "mycli-that-is-not-here", "--reason", "r",
                                *root]) == 1
    assert "not found" in capsys.readouterr().err


def test_doctor_reports_the_instruments_and_a_changed_snapshot(ws, tiny_python, person_key):
    from openlab import doctor
    assert not [r for r in doctor.check(ws).rows if r["check"] == "instruments"]       # none registered: no row
    src = _pkg(ws)
    m = _register(ws, person_key, "tinst", f"{tiny_python} -m tinst", source=str(src))["payload"]
    row = next(r for r in doctor.check(ws).rows if r["check"] == "instruments")
    assert row["ok"] is True and "1 versions of 1 instruments" in row["detail"]
    f = I.Registry(ws).snapshot_dir(I.ident(m)) / "tinst" / "__init__.py"
    _writable(f)
    f.write_text("x = 1\n")
    row = next(r for r in doctor.check(ws).rows if r["check"] == "instruments")
    assert row["ok"] is False and row["fix"] == f"openlab instrument verify {I.ident(m)}"
