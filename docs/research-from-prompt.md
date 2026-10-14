# Research from a prompt

You write a paragraph. An agent turns it into a proposal. You read it, edit it and approve it; approving seals the
plan. A lead agent then runs it as a tree of agents, one per branch, each with its own budget, stop rule and kill
criterion. The lead delegates only with `openlab_agent_spawn`: Claude Code's own Agent/Task tool is disallowed for
it and refused by the hooks, so no branch runs outside its budget or its role prompt. At any time `openlab propose
status` tells you which hypotheses were tested, killed or left untouched, by which agent, and what each branch has
spent. A branch whose stop rule fired reads `STOPPED (stop rule)`; `KILLED (kill criterion)` only when its kill
criterion was met (`propose stop --kill`, refused while one of its hypotheses is supported, or every hypothesis it
serves dropped as a dead end). A tested hypothesis reads its verdict (`supported`, `refuted`, `inconclusive`); an
untested one on a killed branch reads `killed`.

## Set up once

```
openlab init --claude          # hooks, the openlab-lead/-screener/-replicator/-experimenter/-reviewer roles, .mcp.json
openlab person init            # your person key: approving a proposal is signed with it
```

`[agents] cap_usd` and `cap_minutes` in `openlab.toml` bound one proposal's total budget (default $20 and 480
minutes).

## 1. Draft

```
openlab propose new lagx --from prompt.txt --mode temporal --budget-usd 1 --minutes 30
openlab propose new churn "Does tenure predict churn better than the majority baseline? ..." --mode static
```

Give the prompt after the folder (before or after the options), with `--prompt`, or in a file with `--from`. This scaffolds the study folder (study.toml from
the mode's template), `proposal.toml` with your paragraph as the question and background, and `proposal.md`.
With `claude` installed, a planning agent (role planner, at most $0.50 and 15 minutes) fills it in and runs `openlab
propose lint` until it is clean. `--planner-budget-usd 0.2` and `--model haiku` set its budget and model. `--draft-with none` skips it, so you can fill it in yourself or ask the Claude
Code session you are already in: it has `propose_lint`, `propose_new` and `propose_show` through `openlab_tool`.

## 2. Read and edit

`proposal.toml` is the proposal; `proposal.md` is rendered from it, so edit the TOML. It holds:

- `title`, `question`, `background`, `mode`, `metrics` (the primary first), `baselines`, `risks`
- `[budget]`: the total `usd`, `minutes`, `trials` (and `holdout_looks`)
- `[[hypotheses]]`: `id`, `statement`, `prediction`, `falsifier` (the result that would refute it), `tags`
- `[[branches]]`: `id`, `question`, `serves` (hypothesis ids), `role` (screener, replicator, experimenter,
  reviewer), `method`, `data_needs`, `budget = {usd, minutes, trials}`, `stop` (when to stop exploring), `kill` (the
  result that kills it)
- `[[data]]`: `name`, `source`, `recipe`; and `[review]` (queries, include, exclude, coverage_floor) for a
  literature branch

```
$ openlab propose lint lagx
status: draft: not approved
error: proposal.toml: `hypotheses.h1.falsifier` is required: every hypothesis has a falsifier
  meaning: the test or result that would refute it; a hypothesis nothing could refute is not tested
  fix: e.g. falsifier = "a paired test against majority with p >= 0.05 on the research split"
error: proposal.toml: `branches.b1.stop` is required: every branch has a stop rule
  meaning: when to stop exploring, so no agent spends hours on a dead end
  fix: e.g. stop = "3 variants in a row without a gain of 0.005 auc"
error: proposal.toml: `branches.b1.budget.minutes` is more than the total (60 > 30)
  meaning: branches run in parallel: each one's wall time is at most the total's
  fix: edit proposal.toml
...
```

`lint` refuses a hypothesis without a falsifier; a branch without a budget, stop rule or kill criterion; branches
whose $ or trials add up to more than the total; a branch longer than the total's minutes; a total over the
workspace cap; and an experiment with no primary metric. A clean lint renders `proposal.md`.

## 3. Approve (yours)

```
$ openlab propose approve lagx --reason "the plan to run"
openlab person act: approve the proposal of lagx
  Lagged features for next-day y: Does a linear model on yesterday's x beat persistence and climatology at each site?
  total budget: $1, 30 min, 5 trials
    b1 [experimenter] $0.4, 20 min, 2 trials: Does x alone beat climatology and persistence?
        stop: one clean run answers it; a second trial only if the first run's audit is not clean | kill: the x-only model is not better than climatology at p < 0.05
    b2 [experimenter] $0.5, 25 min, 3 trials: Does adding the lag of y help on top of x?
        stop: 3 variants without a gain of 0.005 mae | kill: no variant beats the x-only result at p < 0.05
  hypotheses: h1, h2; primary metric: mae
  Approving writes study.toml from it, makes the question tree and seals the plan.
passphrase to sign (empty cancels):
approved: proposal:49b0ffd72f91 (sha 8d0ab3617709)
study.toml written; sealed as plan:10e58fd2a552; trials budget 5
question tree: root question:90e998d42f1c, branches b1 question:bb4e8f4b4806, b2 question:ca4bf0740501
```

Approving writes study.toml from the proposal (mode, metrics, baselines, `budget.trials` as the branches' sum,
`[review]` for a literature branch), makes the question tree (the root question, one question per branch, each
hypothesis under its branch with its falsifier), records the proposal with its sha, and seals the study. From then
on an edit of proposal.toml is refused by `lint` and `execute`, with the diff, until it is recorded with `openlab
propose amend lagx --reason ...`. Budgets, stop rules, kill criteria, falsifiers and the primary metric need
`--person`. When the amendment changes what study.toml carries (the trial budget, `holdout_looks`, the metrics, the
question), the same act rewrites study.toml and records the study's amendment too: one passphrase, and the prompt
lists the study.toml changes before you sign.

### What lint and approve check before anything runs

Once the proposal itself is clean, `lint` and `approve` also check what would stop the agents later:

- the study.toml that approval would write, and that it is valid;
- model.py, for experimenter and replicator branches;
- each dataset's recipe, file or snapshot, and each `[[data]]` name, which must be a `[data.NAME]` in study.toml;
- a review protocol with queries, for a screener branch;
- another study in the workspace asking the same question (same target) of the same data (same sha256).

`approve` refuses while any of these fails. The same-data check makes the data with its recipe first, as materialise
would, when a sibling has the same target. The conflict comes with its two fixes:

```
error: proposal.toml: churn asks the question of demo (/lab/demo) of the same data (5 trials and 0 holdout looks spent there)
  fix: add `continues = "../demo"` to proposal.toml (one shared trial ledger), or `fresh = "<why this is a new question>"` (a person declares it by approving)
same data as demo (/lab/demo): either
  continue it: [study] continues = "../demo": its 5 trials and 0 holdout looks count against this study's budget too, and this study's against it
  or fresh: [study] fresh = "<why this is a new question>", declared by a person: a budget and a holdout of its own
```

On a terminal, `approve` asks `continue or fresh? type c or f` and writes your answer into proposal.toml before you
sign. Approval copies `continues` or `fresh` into study.toml's `[study]`. A `fresh` you approve is your declaration,
and changing it later needs `propose amend --person`. A refused run never seals study.toml, so you can still fix it
by hand before the first run.

## 4. Execute

```
openlab propose execute lagx --dry-run      # the lead's brief and the planned tree; nothing starts
openlab propose execute lagx                # spawns the lead with the proposal's total budget
```

The dry run prints the brief the lead receives: the hypotheses with node ids and falsifiers, and each branch with
its frontier item, budget, stop rule and kill criterion. The lead delegates each branch with `agent_spawn`, never
above the branch's budget. Each branch is a frontier item; the child leases it with `study_take`, and another
agent's take is refused until the lease is released, expires, or the holder ends. A child that meets its kill
criterion records a dead end with its evidence, then `propose_stop` with `killed`; when a stop rule fires, it calls
`propose_stop`. You can record either yourself with `openlab propose stop lagx b2 --reason ... [--kill]`. Person
acts (holdout looks, more budget, a fresh question) go to `openlab inbox`.

`--model haiku` runs the lead on that model (claude `--model`, codex `-m`), and the children it spawns inherit it.
Without it, the model is `[agents] model` in openlab.toml, else the CLI's default. `openlab propose new` takes
`--model` for the planner too, and `--planner-budget-usd 0.2` sets the planner's budget (default $0.50).

### Each experiment branch has its own model.py

Execute gives every experimenter and replicator branch a working copy in `branches/<id>/`: model.py and the study's
own modules it imports. An existing copy is kept, because it is that branch's work. The brief tells each child to
edit only its copy and run it with `study_run(folder, code="branches/b1")`. From a shell:

```
openlab study run lagx --code branches/b1 --note "x only"
openlab study runs lagx                  # a branch column once a branch's copy has run
openlab study promote lagx b1            # copy b1's code over the study's model.py (recorded)
```

The run records its branch, and its key holds its code's hash as any run's does, so two branches with the same code
share one cached result. The study's own code hash never includes `branches/`. A paper binding is stale when the
code of its run's branch has moved on (the study's model.py for a run without a branch). To bind the main model
to a branch's result, promote that branch first.

### Resume after a person decides

A child that needs you (the same-data choice, more budget, a holdout look) files an inbox item and ends. The lead
ends too when nothing else can go on. `propose status` then says why and what to run:

```
b1 [experimenter] BLOCKED  $0.01/0.4  1/20 min  trials 0/2  ...
   BLOCKED: experimenter-c0d92d ended on inbox #1 (fresh, pending): demo uses the same data; a person must choose
   fix: a person decides it: `openlab inbox approve 1 --reason ...` (or deny), then `openlab propose execute lagx --resume`
```

`openlab inbox` marks such an item with "the requester has ended — run `openlab propose execute lagx --resume`".
Once you decide (approving a `fresh` or `continues` item writes it into study.toml and signs the amendment):

```
openlab propose execute lagx --resume --dry-run    # what is left, the budget left, the brief
openlab propose execute lagx --resume
```

The resumed lead gets only the unfinished branches and the budget left: the proposal's total less what every
earlier agent of the study spent. Its brief names the finished branches (done, killed, stopped) and the person's
decisions since. If the budget is spent, raise it in proposal.toml with `openlab propose amend lagx --person`
first.

## 5. Watch

```
$ openlab propose status lagx
Lagged features for next-day y: Does a linear model on yesterday's x beat persistence and climatology at each site?
approved 2026-10-08T19:57 by person:sam (proposal:49b0ffd72f91)
spent $0.00 of $1, trials 0 of 5, tokens 0
hypotheses: 0 tested, 0 killed, 2 untouched

b1 [experimenter] UNASSIGNED  $0.00/0.4  0/20 min  trials 0/2  tokens 0  Does x alone beat climatology and persistence?
   no agent yet
   h1 untouched: A linear model on x alone has lower MAE than climatology.
   stop rule: one clean run answers it; a second trial only if the first run's audit is not clean -> not fired
   kill: the x-only model is not better than climatology at p < 0.05
...
```

`openlab agents --tree` shows the tree (a branch's agents carry `b1: ...` as their question), and `openlab watch` is
the live screen for a tmux pane. See [Supervising agents](supervising-agents.md).
