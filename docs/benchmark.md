# Benchmark mode: MLE-bench, the same agent with and without openlab

MLE-bench runs on trust. Its leaderboard keeps a separate table for submissions that used test-set feedback, and
its maintainers say they cannot verify any individual submission. Openlab's benchmark mode makes that feedback
impossible by construction: your machine keeps every competition's answers and grades every submission, and the box
where the agents work holds the public data only.

## What you need

| where | what |
|---|---|
| your machine | MLE-bench's grader in its own venv: `python3 -m venv ~/.local/share/openlab/bench/venv && ~/.local/share/openlab/bench/venv/bin/pip install -e <a checkout of github.com/openai/mle-bench>` |
| the box | MLE-bench at `/root/mle-bench` (`git lfs pull` for the leaderboards, `uv venv && uv pip install -e .`) |
| the box | your Kaggle API token at `/root/.kaggle/kaggle.json`, and each competition's rules accepted on kaggle.com |
| the box | Claude Code, signed in on the box: an API key (recommended here, see below), or `claude` run once there to log in |
| the box | openlab (`scripts/box_check.sh` installs it) |

`openlab bench mle doctor --host HOST` checks each, and gives the fix for what is missing.

Benchmark runs are long and unattended, so use an API key for them (`ANTHROPIC_API_KEY` in the environment the
agent starts in on the box) rather than a subscription login. Which way of signing in fits automated use is set by
Anthropic's terms: see [Claude Code's legal and compliance page](https://code.claude.com/docs/en/legal-and-compliance)
(the jobs run Claude Code; for Codex elsewhere, see [Codex authentication](https://developers.openai.com/codex/auth)).
Never copy your own `~/.claude` credentials onto the box. See [On a server or GPU box](box.md#how-agents-sign-in-on-a-box).

## A run

```
openlab bench mle prepare spaceship-titanic --host gpu-box
openlab bench mle job spaceship-titanic --arm openlab --seed 0 --hours 2 --usd 10 --host gpu-box
openlab bench mle status --host gpu-box
openlab bench mle grade spaceship-titanic--openlab--s0--1a2b3c --host gpu-box
```

- **prepare** runs `mlebench prepare` on the box. It then moves the private answers and the leaderboard to your
  machine, deletes the raw download and the private split from the box, and makes the public split read-only.
- **job** makes a fresh directory on the box with a read-only copy of the public data, a `TASK.md` (the
  competition's description and the rules), and an agent (`claude -p`) under openlab's supervisor with a budget in
  dollars and hours. The two arms differ only in openlab:

  | | bare | openlab |
  |---|---|---|
  | supervision, budget, log | yes | yes |
  | openlab's tools (MCP), hooks and record | no | `openlab init --claude` |
  | the competition's data and answers denied | no | `[agents] deny`: Kaggle downloads, the data page, the private split, the raw download |
  | how to work | the task alone | the task, plus: a study for a single-target table (checks, the run tree, comparisons, lessons), a note per variant otherwise, the final pick decided first |

- **grade** brings the submission to your machine and grades it with the competition's own grader, then ranks it on
  the leaderboard for medals. It also scans the agent's whole transcript for leak attempts, in both arms.

## The A/B

```
openlab bench mle ab --competitions lite --seeds 3 --hours 6 --usd 20 --parallel 2 --host gpu-box
openlab bench mle report
```

`ab` runs every competition × arm × seed that has no graded result yet, `--parallel` at a time, and grades each as
it ends. A job past its hours (plus half an hour) is stopped and graded as it stands. It resumes from
`~/.local/share/openlab/bench/jobs.jsonl` and `results.jsonl`, so stop and start it as you like.

`report` gives, per arm:

- the any-medal rate, as the mean over seeds of each seed's rate, with its standard error (MLE-bench's protocol);
- valid submissions and the above-median rate;
- leak attempts, and the runs that made them;
- spend.

Lite is 22 competitions; 3 seeds × 2 arms is 132 runs. Run a pilot first: one small competition, both arms, one
seed.

## Harness arms

An arm may also be `openlab@<sha>` (a stored [harness version](harness.md), shipped to the job and made active there)
and any arm may end in `*N`: best-of-N at matched dollars, N samples with 1/N of the budget each, each writing its own
validation score to `submission/validation.json`; the arm's answer for a seed is the sample with the best one, never
the best grade (`oracle_medal` reports that, as a bound). `pools` splits the competitions once into a development
pool and a private pool, and `accept` gates a candidate harness on the private pool ([harness.md](harness.md)).
`report --out bench/results/mle/<date>.json` keeps a report beside the code.

## What it does not claim

Most Lite competitions are images or text, which an openlab study (a table) does not run. On those, the openlab arm
contributes supervision, the deny patterns and its way of working, not the run tree or lessons. Report results by
competition type as well as overall. The two arms share the box and the model, so their exposure to the network is
the same, and the scan counts attempts in both.

## PaperBench: an instruction and an instrument

[PaperBench](https://github.com/openai/preparedness/tree/main/project/paperbench) (MIT) asks an agent to replicate an
ICML 2024 paper from scratch: a git repository whose `reproduce.sh` regenerates its results, judged against the paper's
rubric. It is the closest public benchmark to a replication study. Openlab's adapter is two files, no runner:

- `bench/paperbench/instruction.md`, the lead's task: the paper in `./paper`, the repository in `./submission`, what
  counts, and how to hand it in;
- `bench/paperbench/submit.py`, an [instrument](instruments.md) the lead hands in through. `check` says whether a
  repository would be accepted; `submit` copies its committed HEAD to `<submissions>/<paper>/submission/`, the layout
  PaperBench's direct-submission grader reads, and keeps a replaced one in the instrument's state. It refuses a
  repository outside the jobs folder, with no commit, without a committed `reproduce.sh` and `README.md`, or over
  1 GB, and warns of uncommitted files, absolute paths and blacklisted addresses. It never grades.

Once, in a workspace for the benchmark (`openlab init --claude` in `~/pb`), register the instrument (your act):

```
openlab instrument register paperbench_submit --source ~/openlab/bench/paperbench --include submit.py \
    --exclude "*.md" --exclude judge.py --entry "python3 submit.py" --runtime binary --capabilities read,write \
    --env PAPERBENCH_WORK_ROOT=$HOME/pb/jobs --env PAPERBENCH_SUBMISSIONS=$HOME/pb/submissions \
    --env PAPERBENCH_PAPERS=$PB/data/papers --outputs submission.json --state --usage-sub \
    --reason "PaperBench hand-ins"
```

For each paper, give the lead the paper and an empty repository, and keep the rest out of its reach:

```
mkdir -p ~/pb/jobs/fre/paper && cd ~/pb/jobs/fre
cp -r $PB/data/papers/fre/{paper.pdf,paper.md,addendum.md,blacklist.txt,config.yaml,assets} paper/
git init -q submission
openlab agent spawn "$(cat ~/openlab/bench/paperbench/instruction.md)" --role lead --cwd ~/pb/jobs/fre \
    --budget-usd 50 --minutes 720
```

Never copy `rubric.json`, `judge.addendum.md` or `judge/`. Add `[agents] deny` patterns to `~/pb/openlab.toml`, such
as `"preparedness|paperbench/data"` and each blacklisted address, so the agents cannot read the judge's notes or the
authors' code. A pattern is matched against a call's input as written, so match a name any spelling of the path
holds, not the full path.

When the leads have finished, grade each submission yourself. `bench/paperbench/judge.py` runs PaperBench's own judge
(its prompts, rubric and file selection) with Claude for each of its three calls, through the Anthropic SDK, on
PaperBench Code-Dev: only the Code Development criteria, so no reproduction, Docker or GPU. Run it in PaperBench's
environment with the key loaded for that command only, and keep its output from the agents:

```
cd $PB && set -a && . ~/.config/pb-judge.env && set +a && uv run python ~/openlab/bench/paperbench/judge.py \
    --paper fre --submission ~/pb/submissions/fre/submission --out ~/pb-grades/fre      # --model, --effort, --context
```

`grade.json` holds the score, the invalid criteria, every criterion's verdict and the judge's tokens and dollars. The
paper and addendum are cached across criteria. A Claude judge's scores are not comparable with PaperBench's published
ones (an o3-mini judge): compare arms judged alike. For the full benchmark, with reproduction on a GPU, use
PaperBench's direct-submission solver and its judge (the commands are in its README). The `lite` split is five
papers, `dev` three, `all` twenty.

PaperBench's papers are in Git LFS, and its repository moved to `openai/frontier-evals`: set
`git config lfs.url https://github.com/openai/frontier-evals.git/info/lfs` in the clone before `git lfs pull`.

For an A/B, give a bare `claude -p` the same instruction with the last section replaced by "leave it committed in
`./submission`", and copy its HEAD into the submissions folder yourself. The judge reads only the repositories, so both
arms are judged alike.
