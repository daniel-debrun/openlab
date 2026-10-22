"""PaperBench's adapter (bench/paperbench): the lead's instruction and the instrument it hands in through. A
submission is a repository's committed HEAD, refused without reproduce.sh and README.md, over 1 GB, outside the jobs
folder or for an unknown paper; warned for uncommitted files, absolute paths and blacklisted addresses; copied as a
clean clone to <paper>/submission/, the replaced one kept in the instrument's state; registered by the person and
run on the record, with no grade anywhere in what it returns."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench" / "paperbench"))
import submit as S  # noqa: E402

from openlab.study import instruments as I  # noqa: E402


def sh(cwd, *args):
    subprocess.run(args, cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def pb(tmp_path, monkeypatch):
    jobs, subs, papers = tmp_path / "jobs", tmp_path / "submissions", tmp_path / "papers"
    repo = jobs / "fre" / "submission"
    repo.mkdir(parents=True)
    subs.mkdir()
    (papers / "fre").mkdir(parents=True)
    (papers / "fre" / "blacklist.txt").write_text("https://github.com/authors/fre\n")
    sh(repo, "git", "init", "-q")
    sh(repo, "git", "config", "user.email", "a@b.c")
    sh(repo, "git", "config", "user.name", "a")
    for k, v in (("PAPERBENCH_WORK_ROOT", jobs), ("PAPERBENCH_SUBMISSIONS", subs), ("PAPERBENCH_PAPERS", papers),
                 ("OPENLAB_INSTRUMENT_STATE", tmp_path / "state"), ("OPENLAB_INSTRUMENT_OUT", tmp_path)):
        monkeypatch.setenv(k, str(v))
    return repo


def commit(repo, files):
    for name, text in files.items():
        (repo / name).write_text(text)
    sh(repo, "git", "add", *files)
    sh(repo, "git", "commit", "-qm", "work")


def test_a_repository_is_refused_until_its_committed_head_can_be_judged(pb, tmp_path):
    for paper, repo, why in (("Bad Id", str(pb), "not a PaperBench paper id"),
                             ("nope", str(pb), "no paper 'nope'"),
                             ("fre", "submission", "not an absolute path"),
                             ("fre", str(tmp_path), "outside the jobs folder"),
                             ("fre", str(pb), "has no commit")):
        with pytest.raises(S.Refused, match=why):
            S.inspect(paper, repo)
    commit(pb, {"main.py": "print(1)\n"})
    with pytest.raises(S.Refused, match="no committed reproduce.sh or README.md"):
        S.inspect("fre", str(pb))
    (pb / "README.md").write_text("what I did\n")                    # written, not committed
    commit(pb, {"reproduce.sh": "python3 main.py\n"})
    with pytest.raises(S.Refused, match="README.md"):
        S.inspect("fre", str(pb))
    assert S.main(["check", "--paper", "fre", str(pb)]) == 2
    assert "README.md" in json.loads((tmp_path / "submission.json").read_text())["refused"]


def test_over_a_gigabyte_is_refused(pb, monkeypatch):
    monkeypatch.setattr(S, "LIMIT", 10)
    commit(pb, {"reproduce.sh": "python3 main.py\n", "README.md": "r\n", "weights.txt": "x" * 100})
    with pytest.raises(S.Refused, match="over PaperBench's 1 GB.*weights.txt"):
        S.inspect("fre", str(pb))


def test_submit_copies_head_as_a_clean_clone_and_keeps_the_one_it_replaces(pb, tmp_path, capsys):
    commit(pb, {"reproduce.sh": f"cd {pb} && python3 main.py\n", "README.md": "see https://github.com/authors/fre\n",
                "main.py": "print(1)\n"})
    (pb / "scratch.txt").write_text("not committed")
    info = S.inspect("fre", str(pb))
    assert len(info["warnings"]) == 3
    assert "not submitted" in info["warnings"][0] and "scratch.txt" in info["warnings"][0]
    assert any("absolute path" in w and "reproduce.sh" in w for w in info["warnings"])
    assert any("blacklisted address" in w and "README.md" in w for w in info["warnings"])
    assert S.main(["check", "--paper", "fre", str(pb)]) == 0 and not (tmp_path / "submissions" / "fre").exists()
    assert S.main(["submit", "--paper", "fre", str(pb)]) == 0
    dest = tmp_path / "submissions" / "fre" / "submission"
    first = json.loads((tmp_path / "submission.json").read_text())
    assert first["submitted"] == str(dest) and first["commit"] == info["commit"]
    assert (dest / "reproduce.sh").exists() and not (dest / "scratch.txt").exists()
    assert subprocess.run(["git", "-C", str(dest), "remote"], capture_output=True, text=True).stdout == ""
    assert sorted(p.name for p in (tmp_path / "submissions").iterdir()) == ["fre"]       # nothing else to grade
    assert [p.name for p in (tmp_path / "submissions" / "fre").iterdir()] == ["submission"]
    commit(pb, {"main.py": "print(2)\n"})
    assert S.main(["submit", "--paper", "fre", str(pb)]) == 0
    second = json.loads((tmp_path / "submission.json").read_text())
    assert second["replaced"] == first["commit"] and (dest / "main.py").read_text() == "print(2)\n"
    kept = list((tmp_path / "state" / "fre").iterdir())
    assert len(kept) == 1 and kept[0].name.endswith(first["commit"][:12])
    assert "replacing" in capsys.readouterr().out


def test_the_lead_hands_in_through_the_registered_instrument_on_the_record(pb, tmp_path, person_key):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "openlab.toml").write_text("")
    commit(pb, {"reproduce.sh": "python3 main.py\n", "README.md": "r\n", "main.py": "print(1)\n"})
    env = {k: str(tmp_path / v) for k, v in (("PAPERBENCH_WORK_ROOT", "jobs"),
                                              ("PAPERBENCH_SUBMISSIONS", "submissions"),
                                              ("PAPERBENCH_PAPERS", "papers"))}
    with person_key.act():
        reg = I.register(ws, "paperbench_submit", "python3 submit.py", source=str(ROOT / "bench" / "paperbench"),
                         include=["submit.py"], exclude=["*.md", "judge.py"], runtime="binary", env=env,
                         capabilities=["read", "write"], outputs=["submission.json"], state=True, usage_sub=True,
                         reason="PaperBench hand-ins")
    assert list(reg["payload"]["files"]) == ["submit.py"], reg["payload"]["files"]
    run = I.run(ws, "paperbench_submit", ["submit", "--paper", "fre", str(pb)])
    assert run["exit"] == 0, run
    out = json.loads(run["outputs"]["submission.json"]["content"])
    assert out["ok"] and (tmp_path / "submissions" / "fre" / "submission" / "main.py").exists()
    assert not any(k in json.dumps(run).lower() for k in ("score", "grade\"", "medal"))
    bad = I.run(ws, "paperbench_submit", ["submit", "--paper", "fre", str(tmp_path)])
    assert bad["exit"] == 2 and "outside the jobs folder" in bad["stderr"]["text"]
    instruction = (ROOT / "bench" / "paperbench" / "instruction.md").read_text()
    assert '"ref": "paperbench_submit"' in instruction and "rubric" not in instruction
