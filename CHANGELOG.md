# Changelog

## Unreleased (Nightly-latest)

- **Instruments.** Registered, immutable programs that measure or act outside the sandbox: a broker's paper account,
  a collector, a scraper, a CLI. An instrument wraps any command (an argv entry: `python -m pkg`, `node x.js`, a shell
  script, a binary, `npx`, `uv run`).
  - **Versions.** A version `NAME@VERSION` is the sha256 of its manifest: the snapshot of its code (copied into the
    content store and a read-only snapshot that runs execute), its settings and a pluggable environment fingerprint
    (python distributions file by file, node and node_modules, binaries by sha256 and `--version`, lockfiles, the
    stdout of any command). An instrument with no source folder is verified, never copied.
  - **Registration** is the person's, signed, on the workspace's instrument record (`.openlab/instruments/`). Agents
    file `agent_request` kind `instrument`, and the hooks refuse the verb to them.
  - **Runs.** Every run re-checks the version and refuses on drift, naming the part that changed. It is recorded as an
    `instrument_run` with the hashes of its redacted logs and declared outputs. Secrets are passed through by name and
    masked in the logs.
  - **Studies** pin versions (`[instruments]`, a protected field) and read their outputs as datasets
    (`[data.x] instrument = "alias"`, `output = "file"`).
  - **Live orders.** A `live-order` instrument runs only as the person, under a signed authority per run (inbox kind
    `instrument_run`).
  - **Interfaces.** CLI `openlab instrument register|list|show|verify|run|runs`; tools `instrument_list`,
    `instrument_show`, `instrument_verify`, `instrument_run`, `instrument_runs`; a `doctor` row. See
    docs/instruments.md.
- **Experiments.** A study's pinned instruments run on a schedule, unattended, for a set number of passes; then the
  experiment ends and an agent is fired to conclude it.
  - **The spec** is `<study>/experiments/<name>.toml`: purpose, the Roadmap step it implements, jobs (a pinned alias,
    args, cron and time zone), the pass rule (a job's successful run, or a cycle of every job), the stop rule
    (passes, until, max failures in a row) and the reaction agent (role, prompt, budget). It is checked strictly; a
    `live-order` instrument is refused.
  - **The record** holds `experiment` (the spec and who asked), `experiment_start` (a signed person node: it runs
    outside the sandbox), `experiment_pass`, `experiment_end` (concluded, killed or failed) and `experiment_reaction`
    (the verdict and conclusion). A started experiment runs the spec it was started with.
  - **Running.** The start installs one tagged crontab line per job (only openlab's own lines are touched, under a
    lock). A job runs only at a minute its schedule fires in its zone, once; each run is recorded as with `instrument
    run --study`. At the end the lines go and the reaction agent is spawned. Agents and the person kill one with a
    reason.
  - **Interfaces.** CLI `openlab experiment new|list|status|request|start|kill|job|react`; tools `experiment_list`,
    `experiment_status`, `experiment_new`, `experiment_request`, `experiment_kill`, `experiment_react`; inbox kind
    `experiment`; the hooks keep `experiment start` and `job --now` the person's. `openlab show` has an
    **Experiments** tab: each experiment's progress, jobs, runs (with their output), end and conclusion, its Roadmap
    step, and the registered instruments. See docs/experiments.md.

## 2026-10-14 -- 1.3.0: openlab show, the workspace and the release

`openlab show` becomes the place a person runs research from: every agent and person, their plans and the record on
one live page, the research's files edited and commented on, and each study compiled into a release when its lead
ends.

- **openlab show, live.** One server for every research under its root, each in its own top tab; a new session is a
  new research. Agents report plans (`agent_plan`, `agent_plan_node`) whose evidence openlab computes from the
  record ids they link. The tabs: **Activity** (every agent and person as measured: state and why, the tool call it
  is in, the last three actions, spend and flags), **Inbox**, **Roadmap** (each plan as a Gantt chart, by version),
  **Board**, **Burn**, **RAID**, the **Audit ledger** (the hash-chained record, with verify), the **Machine queue**
  and **Settings**. People declared in `openlab.toml` `[team]` stand beside the agents, with hours instead of $.
- **The Workspace.** The research's files as a tree that hides; the file you pick on the right. LaTeX and Markdown
  open in an editor beside their rendering (a study's paper through the real LaTeX build, with its findings on their
  lines), a PDF or an image as itself, other text read-only. **Comment** works on every file: on the selected text,
  or on the whole file; the live lead gets the path, the quote and the comment, may hand it to an agent under it,
  and answers in the comment's thread. Saves are refused over a file an agent changed since you opened it.
- **The release.** When a study's lead ends done (or on **Compile now**) the study compiles into `release/`: the
  white paper (only when every binding resolves), the code the last result ran, the signed bundle, a MANIFEST and a
  README. The README is the study's own `README.md` with the record's part (data, the result beside its baselines
  and paired difference, the policy, dead ends, claims, how to reproduce) rewritten between two marks at each
  compile. One compile per study at a time, swapped in atomically; `verify` rechecks it.
- **The paper is one LaTeX document.** `paper/paper.tex` starts from a template (the abstract and the usual
  headings); load any package and include images from the study's folder (copied into the build, hashed into the
  paper). `\labsection{name}` puts a section openlab writes from the record. `study_paper(draft=true)` (`openlab study
  paper --draft`) renders it as it stands with its findings, recording nothing; `--init` starts it. A typed number
  is still a finding the signed build refuses; layout arguments (lengths, column specs) no longer are.
- **Agents ask instead of ending.** The brief asks for a plan first and a question in the inbox when blocked;
  `agent_wait` waits for inbox answers. `openlab inbox answer` answers in the terminal. With `decide="self"` (`openlab
  agent spawn --decide self`, or "Let the agents decide" in show) the agents make their own choices and say why in
  their notes and plans; a question to the person is refused.
- **An adversary role.** `openlab-adversary` attacks a result (look-ahead, the wrong outcome, selection over trials,
  flattering costs or fills, a thin sample, a weak baseline) with counted runs on a branch copy (`study_branch`),
  records breaks and critiques for what fails and a dead end for each attack that found nothing. The lead's role
  lists it.
- **A step's conversation.** A Roadmap step opens its agent's transcript (prompt, messages, each tool call with its
  input and result) up to the call that marked it done.
- **Long runs outlive their agent.** An agent's `study_run` runs in a process of its own: after 14 minutes
  (`OPENLAB_RUN_WAIT`) it returns a job id that `agent_wait` waits on, `study runs` and `status` list it in flight,
  and it records its result even when the agent ended.
- **Smaller tools and fixes.** `study_materialise(refresh=true)` rebuilds a stale dataset and warns when its recipe
  changed; `study_series` reads a result's policy series; a dead end can rest on a claim and close a question; a
  study inside another study is refused; the sandbox hides a nested study's `.openlab`, `.git` and data; the hooks
  refuse a read of `.openlab` on its own (not as a bypass), flag a redirect only into a protected file, and say what
  to do instead; flags in show are one chip per kind with a count; the activity spinner turns only while something
  runs.
- **Prompt a lead.** In `openlab show`, a lead's row has a Prompt button: a short conversation with the lead. **Ask now
  (btw)** answers in seconds from a fork of the lead's Claude session (no tools, not saved), like Claude Code's
  `/btw`; the lead keeps working and never sees it, and an ended lead can be asked too. **Tell the lead** reaches it
  at its next tool call (a waiting lead wakes for it); it answers with `openlab_agent_reply` and carries on, changing
  course if asked. A Codex lead gets it in its next openlab tool's reply.
  Only leads take the person's messages: the agents under a lead hear from it (`openlab agent tell` to one of them
  is refused, and draft comments go to the leads).
- **Pause and play the leads.** A ⏸/▶ beside each lead and Pause all leads / Play all leads in the header; the
  agents under a lead are held with it. A resume is signed with the passphrase.
- **A limit pauses, it does not end.** At an agent's $, minutes or turns limit openlab pauses it and files a budget
  request for it ("asks for more time: 240 -> 480 minutes"); approving raises the limit (or your own, in the New
  limit field or `--pick`) and resumes it, and a denial resumes it briefly to wrap up. At 80% of a limit the agent is
  told to ask for more rather than end with work in progress. `openlab show` marks a held agent "at its limit" and an
  ended one with what it used ("exited: done at 226 of 240 min"); a step an ended agent left open is drawn as
  dropped.
- **A person's request is never held.** The web server no longer refuses the person's own acts when its
  environment names a paused agent.
- **Public release.** Third-party notices for the web bundle, PyPI metadata (classifiers, URLs), the `agent` and
  `app` extras removed (in the core since 1.0), the `bench` harness no longer installed as a top-level package, CI
  on Python 3.10 to 3.13, and a publish workflow (PyPI trusted publishing).

## 2026-10-12 -- 1.2.0: search, lessons, checks and benchmark mode

The machinery of the agents leading MLE-bench, run inside openlab's guardrails, and a way to measure what openlab adds.

- **The search.** `openlab study search` scores the run tree (UCT with a cost-shaped reward, after MARS, on a study
  clock that explores early and exploits late, after MLEvolve). The frontier offers `expand`, `fuse` (a stalled
  line), `draft` (a stalled study) and `select` (late in the budget). `[search]` tunes it. See docs/search.md.
- **The final candidate.** `[selection] rule` (`best_val` or `seed_robust`) is a person's field, sealed with the
  plan. `openlab study select` records the pick among N with its value adjusted for being the best of N; every
  field is derived on write. The holdout look is on the selected run only.
- **Lessons.** `openlab study lesson` records a rule whose direction (improves, hurts, no_gain, fixes) must follow
  from the comparison or run it rests on. Runs cite lessons (`--cite`), each lesson has a computed track record,
  and `openlab lessons find` searches every study in the workspace.
- **Module checks.** `openlab study check`: lint, import, a smoke run of the runner's own job on a little data, and
  determinism, in seconds. A run of code whose latest check is red is refused. See docs/checks.md.
- **Ensembles.** `openlab study ensemble`: mean, rank, or weights cross-fitted on halves of the rows. Each is a
  counted, cached run with its members as refs.
- **Critiques.** A reviewer's findings are advice on the record, answered by the checks that settle them, with a
  computed track record. Nothing reads them into a verdict.
- **Benchmark mode.** `openlab bench mle doctor|prepare|job|status|grade|ab|report`: MLE-bench with the answers and
  the grading on the person's machine and the public data on the box; the same agent with and without openlab, leak
  attempts scanned in both arms. See docs/benchmark.md.
- **Deny patterns.** `[agents] deny` in openlab.toml: tool calls that match a pattern are refused by the hooks and
  recorded.

## 2026-10-12 -- 1.1.0: the shared box

A box (a server or rented GPU container) can now keep the records, run the queue and host the agents, while the
person's signing key and holdout labels stay on their own machine. Checked end to end on a 2× RTX 3090 container.

- **GPUs.** `[sandbox] gpus = N` leases N specific GPUs for a run's whole length. A run sees only its own GPUs
  (`CUDA_VISIBLE_DEVICES`), and one that leased none sees none. bwrap binds the NVIDIA devices. The run records the
  GPUs it held.
- **The queue.** Every run is admitted to the machine by its `[sandbox] cores`, `memory_gb` and `gpus`, first come
  first served with backfill. The person's `box.toml` keeps reservations and an `available_until`. A run says why
  it waits and records the wait. `openlab box [queue]` shows the machine and the queue, `openlab watch` shows QUEUED,
  and agents read it with the `box` tool.
- **Person acts from the person's machine.** `openlab person act HOST -- ARGS` runs the act where the record is and
  signs each node here, shown as it is signed. Never a key rotation, and never by an agent.
- **Holdout labels kept by the person.** `split.labels = "person"`: the holdout is the rows with no target, the box
  records predictions only, and `openlab study score HOST STUDY --labels FILE` scores them here and records the
  signed result, with the labels' sha.
- **Compute requests.** An agent asks the person for another machine. Openlab attaches what this machine has, its
  queue and its availability; it never rents, pays for or stops one.
- **A W&B mirror.** `pip install "openlab[mirror]"`: each recorded run is pushed as an offline W&B run by a detached
  process (the run key as id, the study as group, the op as job type, metrics and shas, never data), then synced.
  `openlab mirror status|sync`.
- New pages: docs/mirror.md; docs/box.md covers the queue, GPUs, person acts and holdout labels.

## 2026-10-11 -- 1.0.0: the first release

Openlab 1.0 is the research-framework control plane, checked on four paper replications (kref, sealevel,
scalingscore, energyad) and on a rented GPU container. Older records still verify; nothing here costs an existing
study a trial.

- **Runs form a tree.** `study run --op draft|improve|debug|fuse --parent RUN [--ref RUN]` records how a run was
  made, checked against its links before anything runs; `study runs` shows each run's op and parent. Every run also
  records the machine it ran on.
- **Servers and containers.** Cores and memory come from the container's cgroup limits (v1 or v2), not the host's
  counts: a container that showed 96 cores and 251 GB is now sized at its 23 CPUs and 120.8 GB. Each sandboxed child
  holds one of the machine's per-core slots, shared by every openlab process, so concurrent studies never
  oversubscribe it (`OPENLAB_BOX_SLOTS`, `OPENLAB_BOX_DIR`). `openlab doctor` shows the machine, its GPUs, and says when
  a container forbids the sandbox outright. New page: docs/box.md.
- **The person key.** A study records the key it was sealed under. Its first signature must be that key or a
  rotation it signed, even after the workspace's pin stores are deleted, and `verify` flags anything else.
- **Seeds and groups.** `study run --seed N` (one run per seed, each a trial; `split.reseed` lets the seed deal the
  folds); `evaluate.by = "key"` scores each entity (a station, a dataset) for temporal and static studies; policy
  baselines act on their own forecasts; wide predictive distributions are reduced on disk as a run writes them;
  audit replays wait for memory.
- **Replication.** A mean over several runs with the seed spread in its interval; per-entity fidelities and
  replications (`group`); tables whose methods are columns (`study extract --header`); several proposed methods per
  plan; re-approval keeps the primary claim; a plan states its compute (runs, run minutes, parallel), lint checks it
  against the minutes, and `replicate status` shows the measured pace.
- **Supervision.** `openlab agent tell` directs a working agent at its next tool call (signed directions are marked
  as yours); `[agents] notify` runs a command when an agent asks you something or ends; `openlab inbox wait`;
  approving a request resumes a requester that has ended; an amend request carries the fields it sets; `openlab
  watch` shows a run in progress.
- **Environment.** `openlab env` lists the study Python's packages; `openlab env add PKG` installs one without changing
  what is installed. MCP results say when openlab's code changed after the server started.
- **Fixes.** The frontier counts trials as the runner does (the shared ledger and declared trials), so it never
  calls a spent budget open. The unused `inference.holm` is gone.

## 2026-10-08 -- research-framework: a control plane for your research agents

Openlab now supervises the coding agents researchers already use (Claude Code, Codex) and keeps the study record they
work on. It holds no model key.

- **Agents.** Every record write names its agent, and agents form a tree (`openlab agents --tree`). `openlab watch` is
  a live screen for a tmux pane with STUCK, LOOP, OVER, WAITING, BYPASS and DEAD-END flags. `openlab agent
  spawn|pause|resume|stop|log`, budgets in $, minutes and turns (billed cost when the CLI reports it, `~` when
  estimated), and `openlab inbox` for what needs a person. `openlab init --claude` writes Claude Code hooks (identity,
  activity, pause, denial of person acts and bypass attempts) and five role agents; `--codex` writes the AGENTS.md
  section and MCP snippet.
- **Person key.** Every person act (holdout look, protected amendment, approval, inbox decision, resume) is signed
  with a passphrase-protected Ed25519 key (`openlab person init|show|pin|rotate`); the record refuses an unsigned one
  and `verify` checks them all. The old tty check, `LAWS.openlab` and `HOOKS.openlab` are gone.
- **Research from a prompt.** `openlab propose new|lint|approve|execute|status|stop|amend`: a proposal with
  falsifiers, branches with budgets, stop rules and kill criteria; approval seals the plan; a lead agent runs the
  branches.
- **Trust fixes.** Every run says whether it spent a trial and why; runs execute a snapshot of the code; budgets
  belong to the question and data, not the folder; `study compare` and `study test --baseline result:` compare
  variants; "selected from N trials" on claims; `study runs`, `study paper-lint`, `study fetch --page`;
  `--template review`, batch screening by several agents, `study evidence --table`. A template run takes about 6 s.
- **MCP.** The default tool list is 31 short tools (under 5k tokens with the instructions); the rest are one
  `openlab_tool` call away.
- **Docs.** A new README and docs around the three flagships and supervising agents; `openlab/AGENTS.md` rewritten as
  the agent's contract; `scripts/fresh_install_check.sh` walks the front door from an empty venv.

## 2026-10-08 -- openlab is a research framework (the cleanup)

Researchers run deep research, paper-to-code replication and prompt-to-proposal-to-execution through their own
coding agents; openlab manages and monitors them and keeps the study record. Removed: the finance code (`quant`, the
finance providers, options, pricing, portfolio, risk, runs, custody), the notebook and session world (sessions, the
frame tools, fits, holdouts outside a study, dashboards, exports, signed reports), the agent harness, `cloud/`,
`demo/` and the ablation bench. The LaTeX engine stays as `openlab.engine`, internal to derived claims and the paper's
equation gate: no `openlab FILE`, `submit`, `modules` or `openlab_run`. The helpers a study uses moved into
`openlab.study` (clock, schedule, notify, signing, vault, checks, the model store).

## 2026-09-27 -- 200 features toward "analysis you can sign"

The full list, one line each, is `plans/FEATURES_200.md`. In short:

- **Plan**: `%%plan` lines for fits, holdouts, alpha, primary, n_min, stop, reason; amendments with diffs;
  exploratory marking for direct `stat_test`/`model_fit`; `nb_plan`; study templates; plan events in the
  journal and the ledger.
- **Guards on fits**: VIF, thin fits, Jarque-Bera, Breusch-Pagan (`se=hc1`), Cook's distance, rare classes,
  outliers, leakage, ID and period-code predictors, Durbin-Watson (`hac=`); `weights=`, `standardize`,
  interactions `x:z`, AIC/BIC, moving-block stability for HAC fits.
- **Tests**: Cohen's d, minimum detectable effect, one-sided alternatives, Mann-Whitney, KS, two-proportion,
  Newey-West mean, deflated Sharpe, `tests_summary` with Holm and BH, independence and small-sample notes.
- **Data**: `%%expect` cells, `not_null`/`unique`/`max_age`, summary flags, `data_check_unique`, fan-out and
  empty-result notes, look-ahead in `%%py` and more SQL shapes, `zscore`/`winsor` shims, time and group
  holdout splits, lineage, sample, crosstab, corr, export, `.csv.gz`, parquet globs, three trap samples.
- **Custody and signing**: brier, mape, precision, recall, f1; baselines; `holdout_close`; provenance in the
  signed report; `verify --json/--source`; `key_info`; reviewer and reason rules; review journal events.
- **Workflow**: `from_uid`, `stop_on_error`, run summaries, `{{name}}` in prose, outline, search, duplicate,
  move, clear, status, lint, check; markdown export; `openlab new/export/check/sanity`; `make check`.
- **Charts**: histograms, trend lines, residuals, coefficients, deciles, cumulative returns, QQ, facets,
  sort, log scale, even thinning; `points=` now actually reaches the chart (it was being taken as a timeout).
- **Interface**: plan chip, Review first in the inspector, claim links, SE estimator, rewritten SQL, effect
  sizes, warning-toned guard notes, `%%expect` cells.
