# Glossary

## Studies

**Study.** A folder with `study.toml`, `model.py`, `recipes/`, `data/`, an optional `paper/`, and its record in
`.openlab/`. Its mode is `temporal` (decisions walk forward in time), `static` (folds over a table) or `review` (a
literature review with no experiment).

**Workspace.** A folder with a `openlab.toml` that holds studies. Its agents, budgets and inbox are managed together.

**Record.** The study's history: an append-only SQLite file (`.openlab/record.db`) plus a store of files kept by
their hash. Each entry (a node) has a kind, a body, links to other nodes, who wrote it, and a hash that chains it to
the entry before. `openlab study verify` recomputes the chain.

**Seal.** The first run freezes `study.toml` as the plan. After that, an edit is an **amendment** with its diff and
reason, and edits to the **protected fields** (split, budget, sandbox, audit, primary metric, `known_at`, the alpha
budget, the review's coverage floor) need you.

**Dataset.** A registered input: a provider call, a snapshot, or a file with the recipe that makes it, identified by
its sha256.

**View, `known_at`.** A view is the data as known at time `t`. `known_at` says when each row became known: an offset
from a time column, a column, or a pair of vintage columns.

**Horizon, embargo.** The horizon is the time from a decision to its label. The embargo keeps a row out of training
until its label is known.

**Research and holdout.** The two segments of the data. Research runs never load holdout rows.

**Trial.** A research run with a new key (its code, data, spec, packages and seed), counted against
`budget.trials`. A rerun with a known key is cached and costs no trial. The first line of every run says which.

**Truncation audit.** After a run, openlab replays sampled steps in fresh processes that hold only the data known at
each step; the outputs must match. A run whose audit is not clean cannot support a claim.

**Look.** One scoring of the held-out rows. Only you can spend one; it is signed, counted against
`budget.holdout_looks`, and recorded with the code, data and spec hashes before anything is scored.

**Result.** A run's metrics with intervals, its baselines, and the paired differences.

**Claim.** A sentence the paper may print. Its **state** (`observed`, `derived`, `inferred`, `hypothesized`,
`unknown`) says what it must rest on. Its **verdict** (`supported`, `refuted`, `inconclusive`, `untested`) and
**level** are computed, never typed. Levels: `recorded` (from research runs), `measured` (from the holdout),
`+reproduced` (its runs re-ran and matched). "Selected from N trials" says how many trials the result was chosen
from.

**Alpha budget.** The study's error budget for tests (alpha-investing). Each test spends some; a rejection earns some
back; when it is spent, tests are refused.

**Evidence, source, coverage.** A source is a paper found by a recorded search. Evidence is a quote and the values
read from it, checked on the page they came from. Coverage records how much a search covered against the review's
floor.

**Fidelity, replication.** In a replication study, a fidelity says whether one of the paper's baselines was rebuilt
within tolerance. A replication is the computed verdict on one of the paper's claims; it stays inconclusive while any
fidelity it depends on fails.

**Target-valued metric, unit.** A metric is ordered one of three ways: higher is better, lower is better, or
target-valued, best at a nominal value (coverage90 at 0.9), where better means closer to it. The paper's numbers are
compared in the metric's own unit: a share printed in percent (92.62%) is read as 0.9262, and the tolerance is in the
metric's unit too.

**Reproduction, bundle.** A reproduction reruns a recorded run from the record alone. A bundle is a signed tar from
which someone else can reproduce the study.

**Question tree, dead end.** Questions and hypotheses form a tree whose statuses openlab derives. A dead end is a
dropped hypothesis or approach with its reason; agents read them before starting, and a new hypothesis that shares a
tag must say how it differs.

**Frontier, lease.** The frontier is the list of next steps openlab derives from the record (untested hypotheses,
claims at risk, gaps), ranked by what each should tell you for the time it costs. Taking an item leases it to one
agent so no other agent does the same step.

**Instrument.** A registered, immutable program that measures or acts outside the sandbox (a broker's paper account,
a collector, a CLI), wrapping any command. Its version `NAME@VERSION` is the sha of its snapshot, entry, settings and
environment fingerprint; a person registers it, every run is verified and recorded with its hashes, and a study pins
the versions it runs and reads. See [instruments.md](instruments.md).

**Experiment, pass.** A study's pinned instruments run on a schedule, unattended, until a stop rule (a number of
passes, a date, failed passes in a row) ends it; then an agent is fired to write its conclusion. A pass is one
successful run of a named job, or a cycle of every job. An agent sets one up and asks; the person's signed start
installs its crontab lines. See [experiments.md](experiments.md).

## Agents and supervision

**Agent.** A Claude Code session or sub-agent, a Codex session, or an agent spawned by openlab. Each has an id, a
role, a parent, a study and question, and a budget. Every record write names the agent that made it.

**Tree.** Agents and the children they delegated to. `openlab agents --tree` and `openlab watch` show it.

**Budget.** For a study, the trials and holdout looks. For an agent, $, wall minutes and turns; a child's budget
comes out of its parent's. `~$` is an estimate from token counts; a billed figure replaces it when the CLI reports
one.

**Inbox.** Requests from agents that need a person (a look, more budget, an amendment, an approval, a screening
conflict). You approve or deny each with a reason. It also holds agents' questions, which you answer instead.

**Question.** An agent's inbox item of kind `question`: 2 to 8 choices, and room for your own words when `other` is
set. You answer it in `openlab show`. An answer is advice, not a person act: it is not signed, and it cannot be
approved or denied.

**Plan.** The steps an agent says it will take (`agent_plan`), each with a kind, the steps it needs, a state and the
record ids it rests on. A change is a new version; a step moves with `agent_plan_node`. `openlab show` draws it.

**Evidence outline.** How `openlab show` outlines a plan step, computed from its links and never typed by the agent:
solid when every link is on the record, dashed when nothing is linked (the agent's word only), red when the record
disagrees (a missing link, or done on a run whose audit found look-ahead). The fill is the state the agent reports.

**Research tab.** One tab in `openlab show` per research: the root folder (when it is a workspace) and each folder
directly under it with a `openlab.toml`. Each has its own agents, inbox, records and paper.

**Release.** `<study>/release/`: the README, the white paper (only when its checks pass), the code folder, the
signed bundle and a manifest. The server compiles it when a study's lead ends done, or on **Compile now**.

**Pause, stop.** Pausing refuses an agent's next tool call (and stops a spawned agent's process); resuming is yours.
Stopping ends it.

**Flags.** What `openlab watch` marks: STUCK (no activity), LOOP (the same refusal again and again), OVER (near a
budget), WAITING (needs you), BYPASS (tried a person's act), DEAD-END (repeating a sibling's dead end).

**Bypass attempt.** A tool call the Claude Code hooks denied because it tried a person's act or a way around one (a
faked terminal, a write into `.openlab/`). Each is logged on the agent's timeline.

**Person act.** An act only you can do: a holdout look, a protected amendment, a screening decision over an agent's,
an approval or waiver, approving a proposal, deciding an inbox item, resuming a paused agent. Each is signed with
your person key.

**Person key.** A passphrase-protected Ed25519 key (`openlab person init`). Its **fingerprint** is pinned in the
workspace (`openlab person pin`); `verify` checks every person act against it.

## Research from a prompt

**Proposal.** `proposal.toml` (rendered as `proposal.md`): the question, hypotheses with falsifiers, branches, data,
metrics and a total budget. Approving it seals the plan.

**Falsifier.** The result that would refute a hypothesis.

**Branch, stop rule, kill criterion.** A branch is one sub-question given to one agent, with its own budget. Its stop
rule says when to stop exploring; its kill criterion is the result that ends it as a dead end.
