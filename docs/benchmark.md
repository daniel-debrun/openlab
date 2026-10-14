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

## What it does not claim

Most Lite competitions are images or text, which an openlab study (a table) does not run. On those, the openlab arm
contributes supervision, the deny patterns and its way of working, not the run tree or lessons. Report results by
competition type as well as overall. The two arms share the box and the model, so their exposure to the network is
the same, and the scan counts attempts in both.
