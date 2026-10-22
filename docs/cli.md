# Command reference

Every command prints its own help: `openlab --help`, `openlab study --help`, `openlab agent tell --help`, and so on.
`openlab study --help` lists every study verb with one line each, and ends with the words it uses (seal, look,
trial, alpha budget, frontier, lattice, level).

## The workspace and the person key

| Command | What it does |
|---|---|
| `openlab init [DIR] [--claude] [--codex]` | set up a workspace: `openlab.toml`, `OPENLAB.md`, `.mcp.json`; `--claude` adds the Claude Code hooks and five role agents; `--codex` adds an `AGENTS.md` section and the Codex MCP snippet |
| `openlab doctor [--json]` | what works in this install and workspace, and the one line that fixes what does not |
| `openlab person init \| show \| pin [STUDY] \| rotate` | your passphrase-protected signing key, its fingerprint, the workspace's (or a study's) pin, and a rotation endorsed by the old key |
| `openlab person act HOST -- ARGS` | a person act on a machine that keeps the record (over ssh), signed on yours: the key never leaves it |
| `openlab verify FILE.sig.json` | check a signed file (a forecast, a bundle, a paper) offline |

## Studies

`openlab study VERB FOLDER ...`. The verbs, by kind of work:

| Kind | Verbs |
|---|---|
| core | `new`, `materialise`, `seal` (`--note`), `run` (`--seed`, `--op`, `--parent`, `--ref`), `promote`, `status`, `runs`, `why`, `show`, `amend`, `verify` |
| holdout | `holdout` (a person's look), `holdout-export`, `holdout-score`, `score` (labels kept by the person) |
| search and selection | `search`, `select`, `check`, `ensemble`, `critiques`, `lesson`, `lessons`, `compare` |
| literature | `discover`, `fetch`, `find`, `screen`, `extract`, `evidence`, `sources`, `prisma`, `lattice`, `recall`, `monitor` |
| claims and organisation | `claim`, `claims`, `test`, `question`, `hypothesis`, `tree`, `assume`, `break`, `dead-end`, `contra`, `trace`, `traceability`, `approve` |
| work | `frontier`, `take`, `release` |
| replication | `replicate`, `fidelity`, `replication`, `replications`, `reproduce`, `bundle` |
| paper | `paper`, `paper-lint`, `verify-paper` |
| forward | `forecast`, `fill`, `schedule` |

`openlab study seal STUDY [--note N]` seals `study.toml` on purpose. It is the study's first research run, counted
as trial 1. Before it runs, it prints what the seal freezes (the split and holdout, budget, instrument pins, primary
metric and the other protected fields) and what it costs. It is refused, with nothing run, when the study is already
sealed (it names when and the `plan` node), is a review study (its first `discover` seals it), or its data is not
materialised. `openlab study run` still seals implicitly. Over MCP it is `study_seal`.

`openlab lessons find QUERY` ranks the lessons of every study in the workspace.

## Instruments

| Command | What it does |
|---|---|
| `openlab instrument register NAME --entry CMD [--source DIR] ... [--usage-arg A] [--usage-sub] [--usage-file F \| --no-usage] --reason R` | register a version (the person's). It probes the entry's `--help` (10 s, in a throwaway copy of the run's snapshot and environment, redacted), shows what the version holds, usage included, and asks for your passphrase. The usage's sha is part of the version |
| `openlab instrument list \| show REF [--verify] [--usage] \| verify REF\|--all` | the registered versions; one manifest with its usage (`--usage`: only the usage); every hash checked again |
| `openlab instrument run REF [--study S] [--reason R] -- ARGS` | run a version (verified first) and record it. It prints the exit code, the redacted stdout and stderr tails (200 lines / 32 KB each), and each declared output's content if it is text of at most 256 KB. A `live-order` instrument is the person's to run, signed per run |
| `openlab instrument output RUN_ID [FILE]` | a recorded run's redacted logs and outputs, read back from the content store with each sha checked; `FILE` (an output, `stdout` or `stderr`) prints that one whole (text, at most 4 MB) |
| `openlab instrument runs [REF] [--study S]` | the recorded runs |

See [instruments.md](instruments.md).

## Experiments

| Command | What it does |
|---|---|
| `openlab experiment new STUDY NAME [--file F]` | write `STUDY/experiments/NAME.toml` from a template with the study's pins (or from `F`), and check it |
| `openlab experiment list` | every experiment of the workspace's studies, with its state and passes |
| `openlab experiment status STUDY NAME` | its state, purpose, Roadmap step, passes and time against the stop rule, each job's next time and last runs, its end and the reaction agent |
| `openlab experiment request STUDY NAME --reason R` | ask the person to start it (inbox kind `experiment`) |
| `openlab experiment start STUDY NAME --reason R [--sha S]` | start it (the person's): shows what will run unattended, asks for your passphrase, installs its crontab lines |
| `openlab experiment kill STUDY NAME --reason R` | end it now: its lines go, the end names who killed it, the reaction agent is fired |
| `openlab experiment job STUDY NAME JOB [--now]` | what a crontab line runs: the job, only at a minute it is due, once; `--now` runs it off schedule (the person's) |
| `openlab experiment react STUDY NAME --verdict V --conclusion C --rests-on ID... [--records ID...]` | record the conclusion of an ended experiment; `--spawn` fires the reaction agent again when its spawn failed |

See [experiments.md](experiments.md).

## Replication and research from a prompt

| Command | What it does |
|---|---|
| `openlab replicate PAPER [FOLDER]` | a paper (arXiv id, DOI or PDF) turned into checked code: an agent drafts the plan, you approve it, agents build it |
| `openlab replicate draft \| lint \| show \| approve \| build \| status \| report FOLDER` | the steps one at a time |
| `openlab propose new \| lint \| show \| approve \| amend \| execute \| status \| stop` | research from a prompt: a proposal, your approval, its execution by a tree of agents |

## Agents

| Command | What it does |
|---|---|
| `openlab agents [--tree] [--all] [--json]` | the agents as a table or a tree |
| `openlab agent spawn PROMPT [--role R] [--cli claude\|codex] [--budget-usd N] [--minutes N] [--turns N] [--parent ID]` | start an agent (`claude -p` or `codex exec`) under a supervisor that enforces its budget |
| `openlab agent log \| status ID` | one agent's timeline, or its state |
| `openlab agent tell ID "TEXT" [--person]` | direct a lead (or, from its supervising agent, a child) at its next tool call; `--person` signs it. Telling an agent that works under a lead is refused: tell its lead |
| `openlab agent pause ID --reason ...` | refuse its next tool call (and stop a spawned agent's process) |
| `openlab agent resume ID --reason ...` | resume it: a person's act, signed with your passphrase |
| `openlab agent stop ID --reason ...` | end it |
| `openlab watch [ROOT] [--once]` | the live supervision screen for a tmux pane |
| `openlab inbox [--all] [--json]` | what needs you |
| `openlab inbox approve \| deny ITEM --reason ...` | your decision on a request: a person's act |
| `openlab inbox answer ITEM --pick N [--other TEXT] [-m MESSAGE]` | answer an agent's question: advice, not signed |
| `openlab inbox wait` | block until an agent asks you something or a spawned agent ends |

## The web views

| Command | What it does |
|---|---|
| `openlab show --root DIR` | the live view: every agent and person, their plans, the inbox, burn, RAID, the audit ledger and the paper ([show.md](show.md)) |
| `openlab app --root DIR` | the local web UI for reading the record and doing person acts |

Both bind 127.0.0.1:8321 by default (`--host`, `--port`, `--token`, or `OPENLAB_APP_HOST`, `OPENLAB_APP_PORT`,
`OPENLAB_APP_TOKEN`) and ask for the token they print.

## Machines, environments and mirrors

| Command | What it does |
|---|---|
| `openlab env [add PKG ...]` | the Python a study's code runs in; `add` installs a package without changing any installed one |
| `openlab box [show \| queue]` | this machine as openlab uses it (cores, memory, GPUs, the schedule in `box.toml`) and its queue of runs ([box.md](box.md)) |
| `openlab node list \| probe \| allocations \| release ALLOC` | the nodes workers are placed on (this machine and box.toml's `[[node]]`s: SSH machines, Slurm clusters), what each has free, and the allocations ([nodes.md](nodes.md)) |
| `openlab node exec -- CMD \| runs \| logs RUN \| wait RUN \| stop RUN \| pull` | a worker's command inside its allocation, on its node; results copied back ([nodes.md](nodes.md)) |
| `openlab mirror status \| sync FOLDER` | the Weights & Biases mirror of a study's runs, a view and never the record ([mirror.md](mirror.md)) |
| `openlab bench mle doctor \| prepare \| job \| status \| grade \| ab \| report \| pools \| accept` | MLE-bench, the same agent with and without openlab (or with a harness version, or best-of-n), graded on your machine, which keeps the answers ([benchmark.md](benchmark.md)) |
| `openlab harness show \| check \| install \| use \| rollback \| signals \| contrast \| lessons \| refine \| ...` | the texts agents are briefed with, as versions you make active; probe signals, harness lessons, refinement ([harness.md](harness.md)) |

## For agents

| Command | What it does |
|---|---|
| `openlab-mcp` | the MCP server an agent connects to (`openlab init` writes `.mcp.json`; or `claude mcp add openlab -- openlab-mcp`) |
| `openlab tools`, `openlab tool NAME key=value ...` | the agent's tools, and one tool called from a shell |
| `openlab guide` | the agent's guide, from a shell |
| `openlab hook EVENT` | the Claude Code hook entry point (written by `openlab init --claude`) |
