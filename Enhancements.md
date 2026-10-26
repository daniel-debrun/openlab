# Enhancements

Friction met while using openlab on real work, and features it would have helped to have. Newest study first.
Each entry: what happened, and what would help.

## ACP: Humanoid Mobility via Attention-based Contacts (arXiv 2610.31608), 2026-10-25

A from-scratch replication of a humanoid RL paper (Unitree G1, three training stages, multi-GPU PPO in simulation)
whose code is not released. Lead spawned in `~/pb/ws` on the claude CLI login, with GPU work on the `vulcan` Slurm
node. The lead's own notes are in `~/pb/ws/jobs/acp-humanoid-mobility/openlab-friction.md`; they are merged here
when the run ends.

### Setting up (the person's side)

- **No hand-in path for a paper outside PaperBench.** The `paperbench_submit` instrument refuses a paper id that is
  not in PaperBench's data, and `bench/paperbench/instruction.md` assumes its layout (addendum, blacklist, config.yaml,
  A10 container). For a new paper I hand-wrote the instruction, the addendum and the scope notes. Would help: a
  generic `openlab replicate`-style job for any paper that a lead works on (paper folder, scope addendum, empty
  repository, hand-in), with PaperBench as one configuration of it.
- **Unclear which flow fits.** `openlab replicate PAPER` exists, but its plan is a tabular study (a metric, a
  tolerance, bootstrap intervals, data recipes). An RL/robotics paper whose results are success rates from simulator
  rollouts trained for hours on a cluster does not obviously fit, so I used the lead flow. Would help: a line in the
  docs on when to use `replicate` and when a lead, or `replicate` accepting "simulation rollouts" as the protocol.
- **Fetching the paper by hand.** `curl` plus `pdftotext` for the lead flow; `replicate` fetches papers but nothing
  else exposes that. Would help: `openlab paper fetch ARXIV_ID DIR` (PDF, text, figures, project-page links).
- **Budget caps edited by hand.** A $50 / 12 h lead needed `cap_usd` and `cap_minutes` raised in `openlab.toml`
  first (the workspace had 20 / 480 from earlier runs). Would help: `openlab agent spawn` saying, before it spawns,
  that a budget exceeds the cap and the one line that raises it.
- **No grading without an API key.** `bench/paperbench/judge.py` calls the Anthropic SDK, so a person who runs only
  the claude CLI login cannot grade; and a paper outside PaperBench has no rubric at all. Would help: a judge that
  runs through `claude -p`, and a way to draft a rubric from the paper (the person approves it before the run).
- **Cluster facts the planner cannot see.** `openlab node list` shows vulcan's partitions and GPUs, but not whether
  its compute nodes reach the internet, which modules or containers (apptainer) exist, or the scratch quota. Those
  decide how a lead installs a simulator. Would help: `openlab node probe` reporting network, container runtime and
  disk quota per node, and the node list telling the lead.

### The lead's first minutes (seen in its log)

- **`study_replicate` refused three times in a row before it took.** (1) "methods must be class names: Model": the
  lead named the paper's methods as the paper does. (2) "no metric `success_rate`": the metric list is tabular-ML
  only (accuracy, rmse, crps, ...), with no success rate, rate-with-Wilson-interval or other rollout metric, which is
  what RL and robotics papers report. (3) "no PDF at /home/barracuda/pb/ws/paper/paper.pdf": a relative study path
  resolved against the workspace root, not the agent's cwd (`jobs/acp-humanoid-mobility`). Would help: resolve paths
  against the caller's cwd; a `success_rate` (binomial, Wilson CI) metric and a lower-is-better `error` metric;
  methods given as free names mapped to classes; and one refusal listing every problem at once instead of one per call.
- **A GPU allocation is held for the worker's whole life, even while it only writes code.** The replicator got 2 L40S
  on vulcan at 16:24 and spent its first ten minutes reading mjlab, downloading data and writing files on this
  machine, with the Slurm job idle. Would help: an allocation the worker takes when its first `openlab node exec`
  needs it (or `openlab_agent_spawn(..., resources=..., lazy=true)`), released after idle minutes and taken again,
  so the cluster time covers runs, not typing.
- **A budget request blocks the lead while it waits.** The lead's request to raise $50 to $150 (inbox #1) went in at
  16:34 and the lead then waited on `inbox:1`, with one `sleep 1200` in Bash. It did say how it would cut scope if
  denied. Would help: the request carrying a default ("if unanswered by T, proceed with the cut"), and a push to the
  person (`[agents] notify`) for budget requests on by default.
- **FIXED: `study status` told a replication's agents the plan was frozen while they could still amend it.**
  `study_replicate` seals study.toml at once (to pin the target paper), and until the first run openlab guards only
  `review.coverage_floor` and `review.comparable_on` (`spec._planning`). But `study status` and the agents'
  `study_status` said "study.toml is frozen; an edit needs `openlab study amend`", so the planner experimenter
  filed inbox #2 asking the person to sign an amendment the agent could have made itself (success_rate as the
  primary metric, seven baselines). Fix: `Study.status()` now returns `planning` and `person_fields`, and the status
  line says the planning window and which fields need a person (openlab/study/__init__.py, openlab/study/cli.py,
  test in tests/test_study_replication.py). Still open: an agent's `agent_request(kind="amend")` could be checked
  against `person_fields` when filed, and answered "you may amend this yourself" instead of reaching the inbox.
- **A refusal listing an empty option reads as a typo.** `study_amend` refused `metrics.success_rate.unit = "share"`
  with "must be one of , fraction": the first allowed value is the empty string. Would help: quote the values
  (`"" (none), "fraction"`), and accept the obvious synonyms (`share`, `proportion`, `rate`) or say "a share is a
  fraction".
- **A lead stays alive between checks only by starting background `sleep`s.** While its replicator trains, the lead
  ends its Claude turn (a `result` event) with a background `sleep 1200` / `sleep 1500` pending; Claude Code `-p`
  wakes it when the sleep ends. If it ever ended a turn with nothing pending, it would exit as done while its
  children (detached, so they live on) still work, with nobody to collect them. `openlab watch` meanwhile flags the
  lead STUCK (no event for 5 min) though it is waiting by design. Would help: the supervisor relaunching a lead
  (`claude --resume`) when it ends while children it spawned are still running, or when one of them ends; and
  `agent_wait` accepting long timeouts with the wait shown as WAITING, not STUCK. The same false STUCK comes from a
  foreground `openlab node exec --follow 560` (a 7-minute smoke run): openlab knows that run is live and could show
  the agent as RUNNING on vulcan instead.
- **FIXED: `openlab node exec` killed running trainings by re-syncing files they held open.** The replicator's two
  detached Stage-1 runs died after 7 min with `OSError: [Errno 116] Stale file handle`. The copy-back of an earlier
  exec had brought their log files here; the next short exec pushed the folder again, and rsync renamed a fresh copy
  over each file the runs held open on vulcan's shared scratch. Fix: the push uses `rsync --update`, so a file newer
  on the node is left alone (openlab/nodes/run.py; test in tests/test_nodes.py checks the open file keeps its inode
  and its writes). Still open: files deleted here come back with the next copy-back (no delete propagation).
- **FIXED: the Bash guard refused `bash "$(dirname "$0")/py.sh"` as decoding a payload,** and then refused the
  replicator's friction note quoting it (a heredoc into a markdown file, not blanked as data because the line held
  other commands). The `sh $(...)` alternative of `DECODE_EXEC` now counts only `sh -c "$(...)"` and `sh <(...)`;
  running a file at a computed path passes (openlab/agents/hooks.py; tests in tests/test_agents_person.py).
- **`openlab agent pause LEAD` pauses only the lead.** The show's ⏸ is described as pausing "the lead and the agents
  under it", but the CLI paused the lead while its replicator kept working; I paused the replicator by hand. Would
  help: `openlab agent pause --tree` (or the CLI matching the show), and resume likewise.
- **FIXED (test only): `test_until_ends_it_at_the_first_job_after` failed once the calendar passed 2026-10-20;** its
  clock is now pinned to the Monday it is written around.
- **`openlab show` opens the Roadmap on an old finished lead.** With the live lead paused, the Roadmap's breadcrumb
  read `lead-1049db` (a lead from the first PaperBench run, done four days ago) and the person's ▶ did not resume
  the live lead or its replicator; both stayed SIGSTOPped with no resume on the record and no error shown. Resumed
  from the CLI instead. Would help: the Roadmap defaulting to the live (or paused) leads, finished ones folded
  below; a paused lead pinned at the top with its ▶; and an error toast when a resume does not go through.
- **FIXED: a lead ended as `done` mid-run because Claude Code `-p` stops background tasks 10 min after the last
  turn.** lead-3aacd7 ended its turn at 16:55 with a background `sleep 1500`; `claude -p` stops background tasks
  still running 10 min after the last turn and exits ("Set CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0 to wait
  indefinitely", in its stderr.log). On resume after the pause the deadline had long passed and it exited 0 as done,
  leaving its replicator orphaned. It would have ended at about 17:05 without the pause. Fixes: the supervisor
  launches every Claude agent with `CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0` (the wall-minute budget bounds the
  wait); and **`openlab agent relaunch ID --reason ...`** (a
  signed person act; agents are refused it) starts an ended Claude agent again in its own session
  (`claude -p MSG --resume SESSION`), same id, budget and children (openlab/agents/supervisor.py, spawn.py, cli.py,
  hooks.py; tests in tests/test_agents_spawn.py, tests/test_agents_person.py).
- **FIXED: an orphaned worker could not be told anything.** With its lead ended, `openlab agent tell` to the
  replicator was refused ("only its lead directs it"). The person may now tell a worker directly when every agent
  above it has ended (openlab/agents/cli.py).
- **Still open:** an agent whose lead ends should hear it, and openlab could relaunch or alert on its own when a
  lead ends with live children; and a worker started before the ceiling fix still has the 10-minute limit (told to
  wait in the foreground until it ends).
- **Spend after a relaunch:** a `--resume`d Claude session reports `total_cost_usd` for the whole session (it
  re-emits the previous total on resume), so the supervisor keeps recording it as is. My first version added the
  earlier spend on top and double-counted: lead-3aacd7 shows about $2.80 too much until its current process ends
  (its supervisor runs that version). Fixed in the code; the fake CLI now models a resumed session's cumulative cost.
- **Works well: the agents outlived the supervising session's crash.** My Claude session restarted at about 17:45;
  the lead, the replicator, their supervisors and the vulcan allocation carried on untouched (each supervisor runs
  in its own session). Only `openlab show`, started in a terminal tab of that session, went down with it, and came
  back with a new token, so the old link died. Would help: `openlab show --daemon` (or a user systemd unit) that
  keeps one long-lived server per workspace, with a stable token in `~/.openlab`.
- **Extracting a robotics paper's numbers fights the review schema.** `study_extract` refused the lead twice: the
  `method` value must appear verbatim in the quote (the paper's figure caption says "Stage-1 trackers using contacts
  alone", not the lead's name for the arm), and `success_rate` is not a field because `study_replicate` seals
  `[review.extract]` with the template's `mean, method, setting`. The schema can still be amended in the planning
  window, but nothing says so. Would help: the replicate scaffold taking its extract fields from the metric it was
  given (`success_rate` here); the method check accepting a quote span plus a declared alias; and the refusal
  naming the amend route.
- **My own regression, caught by the suite and fixed:** adding `agent relaunch` to `PERSON_VERBS` shifted the list,
  and `SCRIPT_VERBS = PERSON_VERBS[:4] + PERSON_VERBS[6:8]` silently lost the `.holdout(` check, so a written Python
  file calling `.holdout()` was no longer refused (about 17:30 to 18:20; no look was taken, and a look still needs
  the person's key). Now `[6:9]`, with a test that names the script verbs by what they are. Would help: build
  `SCRIPT_VERBS` by name rather than by slice.
- **A load-sensitive test.** `tests/test_conjecture_round.py::test_min_over_a_deferred_set_and_max_over_a_bound_set`
  hit the engine's 30 s limit in a full run while the agents loaded this 8-core laptop (load average about 5); the
  file passes alone (51 tests). Would help: a test-only timeout scaled to the machine, or marking it slow.
- **A worker's budget request skips its lead.** The replicator asked the person (inbox #3: $30 to $40, 200 to 280
  turns) though its lead holds $150 of which it has spent under $6, and the lead had planned to "top up the
  replicator" itself. Its first try asked for `usd 10` as an increment and was refused ("not above the current 30").
  Waiting in the foreground on long GPU runs (the workaround for the 10-minute ceiling) costs a turn per check, so
  the 200-turn default runs short for a GPU worker. Would help: a lead granting a child more from its own envelope
  without the person (only a request beyond the lead's envelope reaches the inbox); requests taking `+10` as an
  increment; and turns not counted for a foreground `openlab node wait`.
- **Pausing for the night means choosing between losing GPU work and holding GPUs idle.** Pausing the agents left
  the vulcan allocation and its two Stage-2 runs going; releasing it (`openlab node release`) stopped both runs at
  48 min and cancelled the Slurm job, keeping the files on scratch (Stage-1 checkpoints, partial Stage-2
  checkpoints). A paused worker whose allocation has ended has no way to get a new one: there is no "re-allocate"
  for a live worker, so on resume it must be told, or a new worker spawned with `resources`. Would help: `openlab
  agent pause --tree --release` that also ends the tree's allocations, and a resume that requests them again (same
  shape and node, so the files on scratch are found again).
- **A reboot stops every agent, and nothing brings them back.** The laptop rebooted at about 20:06 with the tree
  paused; both supervisors were terminated (`stopped`, exit 143), and the lead's `agent_wait` had hung 35 min under
  SIGSTOP ("sent no response or progress for 2099s"). Sessions and the record survived, so `openlab agent relaunch`
  brings each back in its own session, and a relaunched worker's new supervisor requests its allocation again (a new
  id, so a new work dir on the node; the old one's files stay on scratch). Would help: `openlab agents relaunch
  --stopped-by-shutdown` (one signature for the tree), a note in the relaunch message naming the old allocation's
  work dir, and an MCP wait that returns early when the agent is paused rather than timing out.
- **A node behind MFA works only while an SSH master connection lives.** vulcan (Alliance) needs keyboard-interactive
  MFA; openlab's batch-mode SSH rode on the person's `ControlMaster` (`ControlPersist 120m`). The reboot ended it,
  and `openlab node` then fails with "Permission denied (keyboard-interactive)" until the person logs in again. The
  same would happen after 120 idle minutes. Would help: `openlab node list` telling a node that needs a fresh login
  apart from one that is down ("vulcan: log in again: ssh -fN vulcan"), an inbox item when a running worker's node
  needs it, and `openlab doctor` naming the ControlPersist window.
- **FIXED: a relaunched worker with resources got no allocation.** At spawn, `spawn()` files the allocation request
  and the supervisor waits up to 60 s for it; my first `relaunch()` started the supervisor without filing one, so
  the replicator's relaunch would have ended as unplaced. I filed the request by hand inside the 60 s (al-1e73f5,
  vulcan, running at 21:04), then made `relaunch()` request it like `spawn()` does (openlab/agents/spawn.py; test
  in tests/test_nodes.py: a relaunched worker gets a second, new allocation and lets it go at its end).
- **No way to compare two baselines head to head.** A run pairs each baseline against `Model` only, and
  `study_compare` takes two results, so the experimenter's `result:...#model:AttnS2PlannerS3Tracker` was refused.
  The paper's Fig. 7B and 7D compare attention against MLP (two systems, neither of them "the method" in every
  panel), and a paper with eight systems wants several pairs. Would help: `study_compare` accepting
  `RESULT#model:NAME` arms (paired on the rows both scored, as variants are), or `evaluate.pairs = [[a, b], ...]`
  in study.toml so a run reports the declared pairs.
- **FIXED: `study_replication` crashed with `KeyError: 'gated_by'` on an ungated claim.** The experimenter checked
  the paper's C.1 claim ("at least 98.5% in every category") with `gated_by: []`, because the paper prints no
  baseline number a fidelity could check. `replication_fields` indexed `links["gated_by"]`, absent when there are no
  gates (and would then have failed on `max([])`). It now refuses with a reason: a verdict is against the best
  gating baseline, and a passed fidelity is what shows the rebuild is faithful; when the paper prints no baseline,
  record an observed claim instead (openlab/study/schema.py; test in tests/test_study_replication.py). I paused the
  experimenter for the fix. **Still open (a design question for the person):** papers whose claims are levels with no
  printed baseline (robotics success rates) cannot get a replication verdict at all; an ungated "level" replication
  (replicated when ours is within tolerance of the paper's, verdict marked "rebuild not certified") would fit them.
- **FIXED: the observed-claim refusal said "at least one" when the problem was "only".** Given a result and a claim
  as grounds, it said "observed claims rest on at least one `rests_on` link to evidence or result: got result:...,
  claim:...", which reads as satisfied. It now says "and on nothing else", names the link not allowed, and points to
  `derived` for claims built on claims (openlab/study/schema.py; test in tests/test_study_claims.py).
- **An adversary cannot turn its critique into a dead end.** The adversary tried `study_dead_end` `of` a claim
  (refused: `of` takes hypothesis, question, run, target, evidence, source, fidelity or replication) and `shown_by`
  its own critiques (refused: critique is not among the kinds). Its natural move, "this claim is a dead end, as my
  critique shows", has no path; it must find the result or comparison under the critique. Would help: the refusal
  saying what an adversary should do instead (a critique already marks the claim; a dead end needs the run or
  comparison that shows it), or `critique` accepted in `shown_by` when the critique itself cites a result.
- **FIXED: "paired" differences were not paired when a method skipped rows.** The adversary found that the study's
  run reported the Stage-1 trackers against the planner systems as paired, though they were scored on different
  episodes (NaN where a system did not run a row): on the shared slots, attention Stage 3 is about 1 point *below*
  the contact-only tracker, not 15 above. `inference.paired` scored each side on all rows, so a study metric that
  skips NaN averaged each method over its own rows. Point predictions are now paired on the rows both scored, and
  the result says how many were `dropped` (openlab/study/inference.py, runner.py; test in
  tests/test_study_own_metrics.py). The run already on the record keeps its old numbers; a rerun records new ones.
- **A study cannot declare a clustering column.** Each reference category's 250 episodes are about 7 replays of
  30 to 98 clips, and replays share outcomes, so episode-level bootstrap intervals were 3 to 4 times too narrow. The
  adversary computed clip-clustered intervals off the record. Would help: `evaluate.cluster = "clip"` (a column
  the bootstrap resamples whole), as `blocks` already does for time.
- **FIXED: the pty guard read prose inside a heredoc as a command.** The lead's README text, written through
  `python3 - <<EOF`, had a line beginning "empty here.", read as the `empty` pty tool. Heredoc bodies an interpreter
  reads (python, perl...) are no longer read as shell lines; code in them that names a pty tool at the start of a
  quoted string (`os.system("empty ...")`, `["script", ...]`) is still refused, and a heredoc a shell reads is read
  as before (openlab/agents/hooks.py; tests in tests/test_agents_person.py). Still refused: prose that quotes the
  tool's name itself ("empty").
- **A fix to how results are computed leaves the record's old results standing, and nobody is told.** After the
  paired-rows fix, the experimenter went on writing claims on the three trials run before it; only the lead had been
  told, and the lead was waiting on an inbox item. Would help: a result records the openlab version (or the
  inference code's sha) that computed it, and `study_status` flags results computed by a version with a known fix
  ("paired differences before 1.4.1 may pair different rows: re-run"); and an inbox item a person makes moot by
  acting elsewhere (resuming the agent directly) closing itself.
- **A hypothesis with a margin is tested without it.** The C.1 hypothesis was "contacts alone are within 5 points
  of contacts plus kinematics"; `study_test` asks only which side is better, so a 1.8-point gap with p 0.006 was
  recorded as refuting it, while the README (clip-clustered, margin-aware) calls the claim reproduced. Would help:
  equivalence and non-inferiority tests (TOST with a declared margin) for hypotheses that state a margin, and the
  test's verdict naming which question it answered.
- **The record disagrees with the write-up, and the reader is not told.** The record's C.1 is refuted (direction
  only, episodes independent); the README's C.1 is reproduced (margin, clip-clustered). Both are honest about their
  own method. Would help: `study_paper` lint flagging a README or paper sentence whose verdict contradicts the
  record's for the same question, with both methods named.
- **Method checks done off the record.** The planner parameter counts that match Table S3 exactly live in
  `results/planner_params.json`; no record node holds them, so no claim can rest on them. Would help: a small
  "check" node (a script's output hashed, a value with its source) that claims may rest on, for facts that are not
  model results.

### From the agents' own friction log (jobs/acp-humanoid-mobility/openlab-friction.md), not covered above

- **A replication study has no holdout, yet says it has one.** The static replication template
  (`openlab/study/templates/replicate/experiment_static.toml`) has `[split]` without `fraction`, unlike the
  static template (`fraction = 0.2`), so every run scores all rows (2,600 of 2,600 here) while `study status`
  reports "holdout looks 0 of 1". Nothing was held out, so nothing leaked, but the record implies a holdout. Would
  help (a decision for the person): add the holdout fraction to the replication template, or have status say
  "no holdout: this split cuts none" and the budget drop the look.
- **No lead-to-child message.** A lead can answer the person (`agent_reply`) but cannot tell its own child
  anything; fixes and corrections reached the replicator and experimenter only through the person. Would help:
  `agent_tell(child, text)` for an agent's own descendants.
- **No top-up of a child's budget by its lead.** After the lead got $150, its replicator stayed at the $30 given
  at spawn, and the replicator's raise had to come from the person. Would help: `agent_budget(id, usd=...)` for a
  parent, within its own envelope.
- **No dry run for model.py.** Whether static mode's `X` holds every column, or what built-in `mean` averages,
  could only be learned by spending a trial. Would help: `study_check` running model.py on a synthetic frame,
  spending nothing and reading no data.
- **No per-cell paired differences.** `evaluate.by` gives per-cell scores, but paired differences and tests are
  pooled only, so the paper's per-category comparisons (Fig. 7 panels) cannot be bound cell by cell. Would help:
  `groups.<cell>.paired`.
- **Dead ends and claims cannot rest on work done off the record.** Training logs and CSVs made by `openlab node
  exec` on vulcan are not record nodes, so a dead end shown by them, or a claim on the parameter-count check,
  has nothing to point to. Would help: a node for an instrument-like output or a hashed file that claims and dead
  ends may cite.
- **Refusals with an empty `try`.** The adversary's two `study_dead_end` refusals came back with no "what to do
  instead".
- **"The MCP server runs older openlab code than the CLI".** After my fixes, `study_status` warned agents of the
  version skew between their long-running MCP server and the CLI; an agent told to use MCP tools cannot act on it.
  Would help: say which fixes the running server lacks, or let the supervisor restart an agent's MCP server.
- **A spawn reply with a burn estimate.** The GPU replicator spent about $3 in its first 10 minutes reading the
  paper and the simulator's source; a per-role estimate at spawn would help set budgets.
- **Worked well (from the agents):** the question and hypothesis tree mapped cleanly onto the paper's C.1 to C.3
  ablations; a CPU child given Table S3 matched both planner parameter counts exactly for $2.6; derived claims with
  LaTeX over `groups.<cell>` paths computed their values; `node exec --detach` with `node logs` is a good loop;
  vulcan's internet made setup take minutes; shared scratch made the lost allocation recoverable; the adversary
  ($3.3) caught the unpaired comparisons and the too-narrow intervals.
