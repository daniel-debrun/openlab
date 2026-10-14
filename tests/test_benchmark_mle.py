"""Guarantees of benchmark mode (openlab.benchmark.mle): the answers come to the person's machine and leave the box,
whose public split is read-only; a job is a fresh directory with the data, the task and an agent under a budget, the
arms differing only in openlab (its init, its deny patterns, its way of working); grading happens here with the
competition's grader; the agent's log is scanned for leak attempts in both arms; the A/B resumes and reports medal
rates with their standard errors. The box is a local stand-in: ssh runs the script here, scp copies."""

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from openlab.benchmark import mle


def local_run(argv, **kw):
    """ssh HOST SCRIPT runs SCRIPT here; scp HOST:REMOTE LOCAL copies."""
    if argv[0] == "ssh":
        return subprocess.run(["bash", "-c", argv[-1]], capture_output=True, text=True, timeout=kw.get("timeout"))
    if argv[0] == "scp":
        src, dst = argv[-2].split(":", 1)[1], argv[-1]
        if not Path(src).exists():
            return subprocess.CompletedProcess(argv, 1, "", f"no {src}")
        (shutil.copytree if Path(src).is_dir() else shutil.copy)(src, dst)
        return subprocess.CompletedProcess(argv, 0, "", "")
    raise AssertionError(argv)


def script(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/bash\n" + text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


@pytest.fixture
def box(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENLAB_BENCH_DIR", str(tmp_path / "here"))
    root = tmp_path / "box"
    repo, data, jobs = root / "mle-bench", root / "mle-data", root / "jobs"
    comp = repo / "mlebench" / "competitions" / "toy"
    comp.mkdir(parents=True)
    (comp / "description.md").write_text("Predict y from x.")
    (comp / "leaderboard.csv").write_text("teamName,score\na,0.9\nb,0.8\n")
    script(repo / ".venv" / "bin" / "mlebench", f'''comp=$3; d={data}/$comp
mkdir -p $d/raw $d/prepared/public $d/prepared/private
echo "id,x" > $d/prepared/public/test.csv; echo "id,y" > $d/prepared/public/sample_submission.csv
echo "id,y" > $d/prepared/private/test.csv; echo raw > $d/raw/train.csv''')
    openlab = script(root / "openlab", '''case "$1 $2" in
"init --claude") echo '[agents]' > openlab.toml ;;
"agent spawn") echo "spawned bench-a1b2c3 (claude, role bench) in $PWD"; echo 'id,y' > submission/submission.csv
  echo '{"type":"tool","input":"kaggle competitions download -c toy"}' > transcript.jsonl
  mkdir -p .openlab/agents/bench-a1b2c3 && mv transcript.jsonl .openlab/agents/bench-a1b2c3/ ;;
"agent status") echo '{"id": "bench-a1b2c3", "status": "done", "spent": {"usd": 1.5}}' ;;
esac''')
    grader = script(tmp_path / "grader", 'echo \'{"score": 0.85, "gold_medal": false, "silver_medal": false, '
                                         '"bronze_medal": true, "above_median": true, "valid": true}\'')
    monkeypatch.setenv("OPENLAB_MLEBENCH_PYTHON", grader)
    return mle.Box("box", openlab, run=local_run, data=str(data), jobs=str(jobs), repo=str(repo))


def test_prepare_brings_the_answers_here_and_leaves_the_box_the_public_split_read_only(box):
    out = mle.prepare(box, "toy")
    here = mle.bench_dir() / "answers" / "toy"
    assert (here / "test.csv").is_file() and out["files"] == ["test.csv"]
    assert (mle.bench_dir() / "leaderboards" / "toy.csv").read_text().startswith("teamName")
    d = Path(box.data) / "toy"
    assert not (d / "prepared" / "private").exists() and not (d / "raw").exists()
    assert not os.access(d / "prepared" / "public" / "test.csv", os.W_OK)


def test_a_job_grade_and_report(box):
    mle.prepare(box, "toy")
    j = mle.job(box, "toy", "openlab", 0, hours=1, usd=5)
    d = Path(j["dir"])
    assert j["agent"] == "bench-a1b2c3" and (d / "data" / "test.csv").is_file()
    task = (d / "TASK.md").read_text()
    assert "Predict y from x." in task and "Working through openlab" in task and "$5" in task
    toml = (d / "openlab.toml").read_text()
    assert "cap_usd = 5" in toml and "kaggle" in toml                     # the openlab arm denies the downloads
    bare = mle.job(box, "toy", "bare", 0, hours=1, usd=5)
    assert "Working through openlab" not in (Path(bare["dir"]) / "TASK.md").read_text()
    assert "deny" not in (Path(bare["dir"]) / "openlab.toml").read_text()
    r = mle.grade(box, j)
    assert r["score"] == 0.85 and r["any_medal"] and r["spent"] == {"usd": 1.5}
    assert r["leaks"] and r["leaks"][0]["why"] == "a Kaggle download"       # scanned in both arms
    assert mle.jobs()[j["id"]]["state"] == "graded"


def test_the_ab_runs_what_is_not_graded_and_resumes(box):
    mle.prepare(box, "toy")
    seen = []
    out = mle.ab(box, ["toy"], seeds=2, hours=1, usd=5, parallel=2, poll_every=0, log=seen.append)
    assert len(out) == 4 and sum(s.startswith("graded") for s in seen) == 4
    assert mle.ab(box, ["toy"], seeds=2, hours=1, usd=5, poll_every=0, log=seen.append) == []    # all graded
    rep = mle.report()
    assert set(rep) == {"bare", "openlab"} and rep["openlab"]["any_medal"] == 100.0 and rep["openlab"]["seeds"] == 2
    assert rep["bare"]["runs_with_leak_attempts"] == 2


def test_the_report_is_the_mean_over_seeds_with_its_standard_error():
    rows = [{"arm": "openlab", "seed": s, "competition": c, "any_medal": m, "valid": True, "above_median": m}
            for s, ms in enumerate([(1, 0), (1, 1), (0, 0)]) for c, m in zip("ab", ms)]
    rep = mle.report(rows)["openlab"]
    assert rep["any_medal"] == 50.0 and rep["sem"] == pytest.approx(28.9, abs=0.1) and rep["valid"] == 100.0


def test_the_leak_scan_names_what_it_found():
    hits = mle.leak_scan('curl https://www.kaggle.com/competitions/toy/data; cat ~/mle-data/toy/prepared/private/test.csv')
    assert {h["why"] for h in hits} == {"the competition's data page", "MLE-bench's private answers"}
    assert mle.leak_scan("pip install lightgbm && python train.py") == []
