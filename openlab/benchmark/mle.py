"""Benchmark mode for MLE-bench: the same agent with and without openlab, graded where the answers are.

The person's machine keeps every competition's private answers and grades every submission; the box (where the
agents work) holds the public data only. That makes test-set feedback impossible by construction, rather than a
matter of trust, which is how the MLE-bench leaderboard has to treat it.

    openlab bench mle doctor --host gpu-box                 # what is missing on either side
    openlab bench mle prepare spaceship-titanic --host gpu-box
    openlab bench mle job spaceship-titanic --arm openlab --seed 0 --hours 6 --usd 20 --host gpu-box
    openlab bench mle status --host gpu-box
    openlab bench mle grade JOB --host gpu-box
    openlab bench mle ab --competitions lite --seeds 3 --hours 6 --usd 20 --parallel 2 --host gpu-box
    openlab bench mle report

prepare: on the box, `mlebench prepare` downloads the competition from Kaggle (the box's ~/.kaggle/kaggle.json) and
splits it. The private answers then move here, and the raw download and the private split are deleted on the box;
the public split is made read-only. The leaderboard (public) comes here too, for medals.

job: a fresh directory on the box with a read-only copy of the public data, TASK.md (the competition's description
and the rules), and the agent spawned under openlab's supervisor (budget in $ and hours). The arms differ in one thing:
`openlab` runs `openlab init --claude` there (the MCP tools, the hooks, the record) and its TASK.md asks the agent to work
through openlab; `bare` has neither. Both write submission/submission.csv.

grade: the submission comes here and is graded against the answers with MLE-bench's own grader, in its own venv
(OPENLAB_MLEBENCH_PYTHON, default <bench dir>/venv/bin/python), then ranked on the leaderboard. The agent's log is
scanned for leak attempts (Kaggle downloads, the competition's raw data, MLE-bench's private paths), in both arms.

State lives in <bench dir> (OPENLAB_BENCH_DIR, default ~/.local/share/openlab/bench): answers/, leaderboards/,
jobs.jsonl, results.jsonl.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path

DATA = "/root/mle-data"                     # the public splits, on the box
JOBS = "/root/bench/jobs"
REPO = "/root/mle-bench"                    # MLE-bench's repository on the box (registry, prepare, leaderboards)
LITE = ("aerial-cactus-identification", "aptos2019-blindness-detection", "denoising-dirty-documents",
        "detecting-insults-in-social-commentary", "dog-breed-identification", "dogs-vs-cats-redux-kernels-edition",
        "histopathologic-cancer-detection", "jigsaw-toxic-comment-classification-challenge", "leaf-classification",
        "mlsp-2013-birds", "new-york-city-taxi-fare-prediction", "nomad2018-predict-transparent-conductors",
        "plant-pathology-2020-fgvc7", "random-acts-of-pizza", "ranzcr-clip-catheter-line-classification",
        "siim-isic-melanoma-classification", "spooky-author-identification", "tabular-playground-series-dec-2021",
        "tabular-playground-series-may-2022", "text-normalization-challenge-english-language",
        "text-normalization-challenge-russian-language", "the-icml-2013-whale-challenge-right-whale-redux")
ARMS = ("bare", "openlab")
LEAKS = [(re.compile(r"kaggle\s+(competitions|datasets)\s+download"), "a Kaggle download"),
         (re.compile(r"kaggle\.com/(c|competitions)/[\w-]+/(data|download)"), "the competition's data page"),
         (re.compile(r"prepared/private|/private/test\.csv|answers\.csv"), "MLE-bench's private answers"),
         (re.compile(r"mle-data/[\w-]+/raw\b|/raw/train"), "the raw download")]


def bench_dir() -> Path:
    return Path(os.environ.get("OPENLAB_BENCH_DIR") or Path.home() / ".local" / "share" / "openlab" / "bench")


def grader_python() -> str:
    return os.environ.get("OPENLAB_MLEBENCH_PYTHON") or str(bench_dir() / "venv" / "bin" / "python")


class Box:
    """The box, reached over ssh (tests replace `run` with a local stand-in)."""

    def __init__(self, host: str, openlab: str = "/root/openlab/.venv/bin/openlab", run=None, data: str = DATA,
                 jobs: str = JOBS, repo: str = REPO):
        self.host, self.openlab, self.data, self.jobs, self.repo = host, openlab, data, jobs, repo
        self._run = run or (lambda argv, **kw: subprocess.run(argv, capture_output=True, text=True, **kw))

    def sh(self, script: str, check: bool = True, timeout: float | None = None) -> str:
        p = self._run(["ssh", "-T", "-o", "ClearAllForwardings=yes", "-o", "LogLevel=ERROR", self.host, script],
                      timeout=timeout)
        if check and p.returncode:
            raise RuntimeError(f"on {self.host}: {script[:120]}...\n{(p.stderr or p.stdout)[-800:]}")
        return p.stdout

    def fetch(self, remote: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        p = self._run(["scp", "-q", "-r", "-o", "ClearAllForwardings=yes", "-o", "LogLevel=ERROR",
                       f"{self.host}:{remote}", str(local)])
        if p.returncode:
            raise RuntimeError(f"scp {self.host}:{remote}: {(p.stderr or p.stdout)[-400:]}")


# --- the person's side: state --------------------------------------------------------------------------------------
def _append(name: str, row: dict) -> None:
    bench_dir().mkdir(parents=True, exist_ok=True)
    with open(bench_dir() / name, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _rows(name: str) -> list:
    p = bench_dir() / name
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.is_file() else []


def jobs() -> dict:
    """{job id: the latest state of the job}."""
    out = {}
    for r in _rows("jobs.jsonl"):
        out[r["id"]] = {**out.get(r["id"], {}), **r}
    return out


# --- doctor --------------------------------------------------------------------------------------------------------
def doctor(box: Box) -> list:
    """[(check, ok, detail)] for both sides."""
    rows = []
    py = grader_python()
    ok = Path(py).is_file() and subprocess.run([py, "-c", "import mlebench"], capture_output=True).returncode == 0
    rows.append(("grader here", ok, py if ok else f"no MLE-bench at {py}: python3 -m venv {bench_dir() / 'venv'} && "
                 f"{bench_dir() / 'venv/bin/pip'} install -e <a checkout of github.com/openai/mle-bench>"))
    out = box.sh(f"""cd /root; echo mlebench=$(test -x {box.repo}/.venv/bin/mlebench && echo yes || echo no);
                     echo kaggle=$(test -f /root/.kaggle/kaggle.json && echo yes || echo no);
                     echo claude=$(command -v claude >/dev/null && echo yes || echo no);
                     echo openlab=$(test -x {shlex.quote(box.openlab)} && echo yes || echo no);
                     echo gpus=$(nvidia-smi -L 2>/dev/null | wc -l); echo disk=$(df -BG --output=avail / | tail -1 | tr -d " ")""",
                 check=False)
    got = dict(line.split("=", 1) for line in out.split() if "=" in line)
    fixes = {"mlebench": f"git clone https://github.com/openai/mle-bench {box.repo} && cd {box.repo} && git lfs pull && "
                         "uv venv && uv pip install -e .",
             "kaggle": "put your Kaggle API token at /root/.kaggle/kaggle.json (chmod 600), and accept each "
                       "competition's rules on kaggle.com",
             "claude": "npm install -g @anthropic-ai/claude-code, then `claude` once to log in (interactive)",
             "openlab": "scripts/box_check.sh step 1 installs it"}
    for k, fix in fixes.items():
        rows.append((f"{k} on {box.host}", got.get(k) == "yes", "ok" if got.get(k) == "yes" else fix))
    rows.append((f"gpus on {box.host}", int(got.get("gpus", "0") or 0) > 0, got.get("gpus", "?")))
    rows.append((f"disk on {box.host}", True, got.get("disk", "?")))
    return rows


# --- prepare ----------------------------------------------------------------------------------------------------
def prepare(box: Box, comp: str) -> dict:
    """Download and split on the box; the answers and the leaderboard come here; the box keeps the public split."""
    box.sh(f"cd {box.repo} && .venv/bin/mlebench prepare -c {shlex.quote(comp)} --data-dir {box.data}", timeout=6 * 3600)
    private, public = f"{box.data}/{comp}/prepared/private", f"{box.data}/{comp}/prepared/public"
    here = bench_dir() / "answers" / comp
    if here.exists():
        import shutil
        shutil.rmtree(here)
    box.fetch(private, here)
    box.fetch(f"{box.repo}/mlebench/competitions/{comp}/leaderboard.csv", bench_dir() / "leaderboards" / f"{comp}.csv")
    lb = (bench_dir() / "leaderboards" / f"{comp}.csv").read_text(errors="replace")
    if lb.startswith("version https://git-lfs"):
        raise RuntimeError(f"the leaderboard of {comp} is a git-lfs pointer: run `git lfs pull` in {box.repo}")
    box.sh(f"rm -rf {box.data}/{comp}/prepared/private {box.data}/{comp}/raw && chmod -R a-w {public}")
    out = {"competition": comp, "answers": str(here), "files": sorted(p.name for p in here.rglob("*") if p.is_file()),
           "at": time.time()}
    _append("prepared.jsonl", out)
    return out


# --- a job --------------------------------------------------------------------------------------------------------
RULES = """## Rules of this run

- The data is in `data/` (read-only). Write your final predictions to `submission/submission.csv`, in the format of
  `data/sample_submission.csv`. Only that file is graded, once, after you end.
- You have about {hours} hours and ${usd} of budget. Use the GPUs (`nvidia-smi`); install packages with pip as you need.
- Do not download this competition's data (or anything derived from it) from the internet, and do not look for its
  answers: the grade is computed elsewhere, and such attempts are recorded.
"""

OPENLAB_WAY = """## Working through openlab

This directory is an openlab workspace (its tools are in your MCP list; `openlab guide` from a shell). Use it to keep
your search honest and to learn as you go:

- If the task is one table with one target, make it a study (`openlab_study_new` with the static template, its data
  `data/train.csv`): `openlab_study_check` the code, then `openlab_study_run` each variant with `op` and `parent`
  (`openlab_tool("study_search")` ranks what to grow next), compare variants with `openlab_study_compare`, and record
  clear differences as lessons (`openlab_study_lesson`). Search earlier lessons with `openlab_tool("lessons_find", ...)`.
- For any other task, keep a note of every variant you try and its validation score (`openlab_agent_note`).
- Score variants on your own validation split, never on the test data.
- Decide how you will pick your final model before you start, and pick it that way at the end.
"""


def task_text(comp: str, arm: str, description: str, hours: float, usd: float) -> str:
    head = f"# Task: {comp}\n\nYou are competing in a Kaggle competition, offline. Here is its description.\n\n"
    return head + description.strip() + "\n\n" + RULES.format(hours=hours, usd=usd) + (
        "\n" + OPENLAB_WAY if arm == "openlab" else "")


def job(box: Box, comp: str, arm: str, seed: int, hours: float, usd: float, model: str = "") -> dict:
    """Make the job's directory on the box and spawn its agent; returns the job's state."""
    if arm not in ARMS:
        raise ValueError(f"arm is one of {', '.join(ARMS)}")
    jid = f"{comp}--{arm}--s{seed}--{uuid.uuid4().hex[:6]}"
    d = f"{box.jobs}/{jid}"
    desc = box.sh(f"cat {box.repo}/mlebench/competitions/{shlex.quote(comp)}/description.md")
    task = task_text(comp, arm, desc, hours, usd)
    deny = ", ".join(json.dumps(rx.pattern) for rx, _ in LEAKS)
    init = (f"{shlex.quote(box.openlab)} init --claude >/dev/null && "
            f"printf '%s\\n' '' '[agents]' 'cap_usd = {usd}' 'cap_minutes = {int(hours * 60) + 30}' "
            f"'deny = [{deny}]' >> openlab.toml && " if arm == "openlab" else
            f"printf '%s\\n' '[agents]' 'cap_usd = {usd}' 'cap_minutes = {int(hours * 60) + 30}' > openlab.toml && ")
    spawn = (f"{shlex.quote(box.openlab)} agent spawn \"$(cat TASK.md)\" --role bench --cli claude "
             f"--budget-usd {usd} --minutes {int(hours * 60)} --turns 100000"
             + (f" --model {shlex.quote(model)}" if model else ""))
    out = box.sh(f"""set -e; mkdir -p {d}/submission && cd {d} && cp -r {box.data}/{shlex.quote(comp)}/prepared/public data &&
                     chmod -R a-w data && cat > TASK.md <<'OPENLAB_TASK'
{task}
OPENLAB_TASK
                     {init}{spawn}""")
    agent = (re.search(r"spawned (\S+)", out) or [None, ""])[1]
    state = {"id": jid, "competition": comp, "arm": arm, "seed": seed, "hours": hours, "usd": usd, "dir": d,
             "agent": agent, "started": time.time(), "state": "running", "spawn": out.strip()[-300:]}
    _append("jobs.jsonl", state)
    return state


def poll(box: Box, j: dict) -> dict:
    """The job's agent status on the box (running, or an end state), and whether its submission exists."""
    out = box.sh(f"cd {j['dir']} && {shlex.quote(box.openlab)} agent status {shlex.quote(j['agent'])}; "
                 f"test -f submission/submission.csv && echo SUBMISSION=yes || echo SUBMISSION=no", check=False)
    status = "unknown"
    try:
        status = json.loads(out[:out.rindex("}") + 1]).get("status", "unknown")
    except ValueError:
        pass
    return {"status": status, "submission": "SUBMISSION=yes" in out}


def leak_scan(text: str) -> list:
    hits = []
    for rx, why in LEAKS:
        for m in rx.finditer(text):
            hits.append({"why": why, "text": text[max(0, m.start() - 60):m.end() + 60].replace("\n", " ")})
    return hits


GRADE = r"""
import json, sys
from pathlib import Path
import pandas as pd
from mlebench.registry import registry
from mlebench.grade_helpers import InvalidSubmissionError
comp_id, sub, answers_dir, lb = sys.argv[1:5]
c = registry.get_competition(comp_id)
answers = Path(answers_dir) / Path(str(c.answers)).name
try:
    sub_df = pd.read_csv(sub)
    score = c.grader(sub_df, pd.read_csv(answers) if answers.suffix == ".csv" else answers)
except (InvalidSubmissionError, FileNotFoundError, ValueError) as e:
    print(json.dumps({"score": None, "valid": False, "error": str(e)[:300]})); sys.exit(0)
rank = c.grader.rank_score(score, pd.read_csv(lb))
print(json.dumps({"score": None if score is None else float(score), "valid": score is not None,
                  **{k: (bool(v) if isinstance(v, (bool,)) or str(type(v)).endswith("bool_'>") else v)
                     for k, v in rank.items()}}, default=float))
"""


def grade_local(comp: str, submission: Path) -> dict:
    """MLE-bench's grader, in its own venv, against the answers kept here."""
    lb = bench_dir() / "leaderboards" / f"{comp}.csv"
    p = subprocess.run([grader_python(), "-c", GRADE, comp, str(submission), str(bench_dir() / "answers" / comp),
                        str(lb)], capture_output=True, text=True, timeout=1800)
    if p.returncode:
        raise RuntimeError(f"the grader failed: {(p.stderr or p.stdout)[-800:]}")
    out = json.loads(p.stdout.strip().splitlines()[-1])
    out["any_medal"] = bool(out.get("gold_medal") or out.get("silver_medal") or out.get("bronze_medal"))
    return out


def grade(box: Box, j: dict) -> dict:
    here = bench_dir() / "submissions" / j["id"]
    here.mkdir(parents=True, exist_ok=True)
    sub = here / "submission.csv"
    try:
        box.fetch(f"{j['dir']}/submission/submission.csv", sub)
        g = grade_local(j["competition"], sub)
    except RuntimeError as e:
        g = {"score": None, "valid": False, "any_medal": False, "error": str(e)[-300:]}
    log = box.sh(f"cd {j['dir']} && cat .openlab/agents/*/transcript.jsonl 2>/dev/null | tail -c 50000000", check=False)
    spend = box.sh(f"cd {j['dir']} && {shlex.quote(box.openlab)} agent status {shlex.quote(j['agent'])}", check=False)
    try:
        spent = json.loads(spend[:spend.rindex("}") + 1]).get("spent", {})
    except ValueError:
        spent = {}
    row = {**{k: j[k] for k in ("id", "competition", "arm", "seed", "hours", "usd")}, **g,
           "leaks": leak_scan(log), "spent": spent, "graded": time.time()}
    _append("results.jsonl", row)
    _append("jobs.jsonl", {"id": j["id"], "state": "graded"})
    return row


# --- the A/B ------------------------------------------------------------------------------------------------------
def ab(box: Box, competitions: list, arms=ARMS, seeds: int = 3, hours: float = 6, usd: float = 20, parallel: int = 2,
       poll_every: float = 60, model: str = "", log=print) -> list:
    """Run every (competition, arm, seed) that has no graded result yet, `parallel` at a time; grade each as it
    ends. Resumable: graded and running jobs are picked up from jobs.jsonl."""
    done = {(r["competition"], r["arm"], r["seed"]) for r in _rows("results.jsonl")}
    todo = [(c, a, s) for s in range(seeds) for c in competitions for a in arms if (c, a, s) not in done]
    running = [j for j in jobs().values() if j.get("state") == "running"
               and (j["competition"], j["arm"], j["seed"]) in set(todo)]
    taken = {(j["competition"], j["arm"], j["seed"]) for j in running}
    queue = [t for t in todo if t not in taken]
    graded = []
    while queue or running:
        while queue and len(running) < parallel:
            c, a, s = queue.pop(0)
            log(f"start {c} {a} seed {s}")
            running.append(job(box, c, a, s, hours, usd, model))
        time.sleep(poll_every)
        for j in list(running):
            st = poll(box, j)
            over = time.time() - j["started"] > (j["hours"] + 0.5) * 3600
            if st["status"] in ("done", "failed", "stopped", "lost", "ended") or over:
                if over and st["status"] == "running":
                    box.sh(f"cd {j['dir']} && {shlex.quote(box.openlab)} agent stop {shlex.quote(j['agent'])} "
                           "--reason 'time budget'", check=False)
                r = grade(box, j)
                log(f"graded {j['id']}: score {r.get('score')} medal {r.get('any_medal')} leaks {len(r['leaks'])}")
                graded.append(r)
                running.remove(j)
    return graded


def report(rows: list | None = None) -> dict:
    """Per arm: medal rate (mean over seeds of the per-seed medal rate, ± its standard error), valid submissions,
    above-median rate, leak attempts, spend."""
    import statistics
    rows = _rows("results.jsonl") if rows is None else rows
    out = {}
    for arm in sorted({r["arm"] for r in rows}):
        rs = [r for r in rows if r["arm"] == arm]
        per_seed = {}
        for r in rs:
            per_seed.setdefault(r["seed"], []).append(bool(r.get("any_medal")))
        rates = [sum(v) / len(v) for v in per_seed.values()]
        mean = statistics.mean(rates) if rates else 0.0
        sem = statistics.stdev(rates) / len(rates) ** 0.5 if len(rates) > 1 else 0.0
        out[arm] = {"runs": len(rs), "competitions": len({r["competition"] for r in rs}), "seeds": len(per_seed),
                    "any_medal": round(100 * mean, 1), "sem": round(100 * sem, 1),
                    "valid": round(100 * sum(bool(r.get("valid")) for r in rs) / len(rs), 1),
                    "above_median": round(100 * sum(bool(r.get("above_median")) for r in rs) / len(rs), 1),
                    "leak_attempts": sum(len(r.get("leaks", [])) for r in rs),
                    "runs_with_leak_attempts": sum(1 for r in rs if r.get("leaks")),
                    "usd": round(sum(float((r.get("spent") or {}).get("usd") or 0) for r in rs), 2)}
    return out


# --- the command -------------------------------------------------------------------------------------------------
def main(argv) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="openlab bench mle", description="MLE-bench: the same agent with and without "
                                "openlab, graded on this machine, which keeps the answers.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def host(x):
        x.add_argument("--host", required=True, help="the ssh host where the agents work")
        x.add_argument("--openlab", default="/root/openlab/.venv/bin/openlab", help="openlab there")
    host(sub.add_parser("doctor", help="what is missing, here and on the box"))
    pr = sub.add_parser("prepare", help="download and split a competition on the box; keep its answers here")
    pr.add_argument("competition")
    host(pr)
    jb = sub.add_parser("job", help="one run: a fresh directory on the box and an agent in it")
    jb.add_argument("competition")
    jb.add_argument("--arm", required=True, choices=ARMS)
    jb.add_argument("--seed", type=int, default=0)
    jb.add_argument("--hours", type=float, default=6)
    jb.add_argument("--usd", type=float, default=20)
    jb.add_argument("--model", default="")
    host(jb)
    st = sub.add_parser("status", help="the jobs and their agents")
    host(st)
    gr = sub.add_parser("grade", help="grade a job's submission here")
    gr.add_argument("job")
    host(gr)
    a = sub.add_parser("ab", help="every competition × arm × seed not graded yet, a few at a time")
    a.add_argument("--competitions", default="lite", help="lite, or a comma-separated list")
    a.add_argument("--arms", default="bare,openlab")
    a.add_argument("--seeds", type=int, default=3)
    a.add_argument("--hours", type=float, default=6)
    a.add_argument("--usd", type=float, default=20)
    a.add_argument("--parallel", type=int, default=2)
    a.add_argument("--model", default="")
    host(a)
    sub.add_parser("report", help="medal rates, validity and leak attempts by arm")
    args = p.parse_args(argv)
    if args.cmd == "report":
        print(json.dumps(report(), indent=1))
        return 0
    box = Box(args.host, args.openlab)
    if args.cmd == "doctor":
        rows = doctor(box)
        for name, ok, detail in rows:
            print(f"  {'ok  ' if ok else 'FAIL'} {name:<22} {detail}")
        return 0 if all(ok for _, ok, _ in rows) else 1
    if args.cmd == "prepare":
        print(json.dumps(prepare(box, args.competition), indent=1))
    elif args.cmd == "job":
        print(json.dumps(job(box, args.competition, args.arm, args.seed, args.hours, args.usd, args.model), indent=1))
    elif args.cmd == "status":
        for j in jobs().values():
            st = poll(box, j) if j.get("state") == "running" else {}
            print(f"{j['id']:<60} {j.get('state'):<8} {st.get('status', ''):<8} "
                  f"{'submitted' if st.get('submission') else ''}")
    elif args.cmd == "grade":
        print(json.dumps(grade(box, jobs()[args.job]), indent=1, default=str))
    elif args.cmd == "ab":
        comps = list(LITE) if args.competitions == "lite" else args.competitions.split(",")
        ab(box, comps, tuple(args.arms.split(",")), args.seeds, args.hours, args.usd, args.parallel, model=args.model)
        print(json.dumps(report(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
