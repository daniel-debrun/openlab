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
        src, dst = argv[-2], argv[-1]
        src, dst = (src, dst.split(":", 1)[1]) if dst.startswith("box:") else (src.split(":", 1)[1], dst)
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


def test_an_arm_names_a_harness_version_and_samples(box, tmp_path):
    import json
    from openlab.harness import versions
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "openlab.toml").write_text("")
    src = tmp_path / "cand"
    shutil.copytree(versions.DEFAULT, src, ignore=shutil.ignore_patterns("__pycache__"))
    (src / "common.md").write_text((src / "common.md").read_text() + "- one more line\n")
    sha = versions.install(ws, src)["sha"]
    assert mle.parse_arm(f"openlab@{sha[:8]}*3", ws) == {"spec": f"openlab@{sha[:12]}*3", "kind": "openlab",
                                                         "harness": sha, "n": 3, "label": f"openlab-{sha[:12]}x3"}
    for bad in ("bare@abcdef12", "openlab*0", "nope"):
        with pytest.raises(ValueError):
            mle.parse_arm(bad, ws)
    mle.prepare(box, "toy")
    j = mle.job(box, "toy", f"openlab@{sha[:8]}", 0, hours=1, usd=6, root=ws)
    d = Path(j["dir"])
    assert (d / ".openlab" / "harness" / "active").read_text() == sha and j["harness"] == sha
    assert (d / ".openlab" / "harness" / "versions" / sha / "common.md").read_text().endswith("one more line\n")
    out = mle.ab(box, ["toy"], arms=("openlab*2",), seeds=1, hours=1, usd=6, poll_every=0, log=lambda s: None,
                 root=ws)
    assert len(out) == 2 and {r["sample"] for r in out} == {0, 1} and all(r["usd"] == 3.0 for r in out)
    assert "validation.json" in (Path(mle.jobs()[out[0]["id"]]["dir"]) / "TASK.md").read_text()
    rep = mle.report()
    assert rep["openlab*2"]["runs"] == 1 and rep["openlab*2"]["usd"] == 3.0 and rep["openlab*2"]["oracle_medal"] == 100


def test_best_of_n_picks_by_the_samples_own_validation_never_the_grade():
    rows = [{"competition": "c", "arm": "openlab*2", "seed": 0, "n": 2, "sample": k, "any_medal": m,
             "validation": {"score": v, "higher_is_better": False}, "spent": {"usd": 1}}
            for k, m, v in ((0, True, 0.9), (1, False, 0.2))]
    (one,) = mle.select(rows)
    assert one["selected"] == 1 and not one["any_medal"] and one["oracle_medal"] and one["spent"]["usd"] == 2


def test_accept_reads_the_graded_rows_and_records_its_verdict(box, tmp_path):
    from openlab.agents.store import Store
    from openlab.harness import evaluate, versions
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "openlab.toml").write_text("")
    src = tmp_path / "cand"
    shutil.copytree(versions.DEFAULT, src, ignore=shutil.ignore_patterns("__pycache__"))
    (src / "common.md").write_text((src / "common.md").read_text() + "- one more line\n")
    sha = versions.install(ws, src)["sha"]
    evaluate.fixed_pools(mle.bench_dir() / "pools.json", ["a", "b"], private=1.0)
    for arm, medals in (("openlab", [(1, 0), (0, 0), (1, 0)]), (f"openlab@{sha[:12]}", [(1, 1), (1, 1), (1, 1)])):
        for s, ms in enumerate(medals):
            for c, m in zip("ab", ms):
                mle._append("results.jsonl", {"competition": c, "arm": arm, "seed": s, "any_medal": bool(m),
                                              "valid": True, "spent": {"usd": 1}})
    out = mle.accept_cli("openlab", f"openlab@{sha[:8]}", traps={"traps": {"traps": [{"trap": "x", "caught": True}]}},
                         root=ws)
    assert out["delta"] > out["band"] > 0 and out["accepted"], out["reasons"]
    ev = Store(ws).events("harness", n=5, kinds=("harness_eval",))[-1]
    assert ev["payload"]["accepted"] and ev["payload"]["usd"] == 6
