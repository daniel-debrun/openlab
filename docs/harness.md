# The harness: what agents are told, as versions

openlab runs no model of its own. Its agents are your Claude Code and Codex, and what openlab adds around them is a
*harness*: the brief a spawned agent gets, its role's prompt, the briefs of a proposal's and a replication's agents,
the MCP instructions and the one-line description of each default tool. Those texts are files, not code
(`openlab/harness/default/`), so they can be versioned, compared and improved on evidence.

What enforces stays out of the harness: the hooks, the person verbs, `[agents] deny`, the sandbox, budgets, allowed
tools and the holdout live in code and `openlab.toml`. A harness folder holding anything else (a hook, a setting) is
refused, and its one kind of code, a skill's script, runs only sandboxed.

## Versions

```
openlab harness show                       # the active version, the stored ones, [harness] settings
openlab harness export default ./h         # a copy to edit
openlab harness check ./h                  # what makes it unfit, if anything
openlab harness install ./h --note "..."   # store it (by its fingerprint) under .openlab/harness/versions/
openlab harness diff default <sha>
openlab harness use <sha> --reason "..."   # make it active: your act, signed with your person key
openlab harness rollback --reason "..."
```

`check` refuses a text whose `{fields}` differ from what openlab fills, a role file without its description line, a
tool list other than openlab's, text over its cap (what a spawned agent is always told: 6,000 characters), a skill no
always-on text names, a script that does not compile, and words that tie a harness to how a benchmark grades. Your
`LAWS.openlab` adds rules of your own: `forbid: <regular expression>` lines no harness text may match and
`require: <text>` lines the always-on text must hold.

A version becomes active only if it is the packaged one, was active before, or rests on a harness lesson you approved.

Every spawn records what briefed it: `harness` (the version) and `harness_sha` (the version, the role, the decide note,
the lessons put in its brief, its CLI, tools, permission mode, deny list and `OPENLAB.md`) in its `launch.json` and its
`spawn` event. Spawns that share a `harness_sha` were told the same things, whatever their task. A child is briefed by
its parent's version. `[agents.roles.<role>] allowed_tools` gives a role its own tool list.

## Probe signals

```
openlab harness signals AGENT                       # one run, as rates
openlab harness signals --role screener --by harness
openlab harness contrast --ref <sha> --cand <sha> --role screener --save
openlab harness transcript AGENT                    # [ACTION] / [OBSERVATION] / [ERROR]
```

A run's signals are rates read from its events and the records it wrote, with no model: tool errors and refusals per
call, bypass attempts, loops and stalls, budget holds, budget used, refused record writes, red module checks,
look-ahead audits, plan steps the record shows or contradicts, a repeated dead end, $ and minutes per step shown, and
whether it delivered. A contrast calls a difference only when both sides have 2+ runs and it exceeds twice their
pooled standard error. Signals are evidence, never an acceptance score.

## Harness lessons and the reflector

A harness lesson is advice drawn from runs: its cue (the shape of the situation), the advice, the stored contrast it
rests on (2+ reference runs, a field that moved), the signals and tasks it predicts will move, the runs it came from,
and optionally an edit of the harness (stored as a version). A cue or advice naming a file path, a copied constant or a
benchmark is refused. A lesson drawn from a run that read the open web or a fetched source is quarantined: its advice
must state a pattern, not give an order, and you approve it with `--pick untrusted`. An edit that adds a script needs
`--pick script`.

Lessons arrive in your inbox (`openlab inbox approve ID`). Approved lessons are put in the briefs of later agents of
their role and template (`[harness] advice` of them, under the always-on cap), and their track record is computed from
the spawns that had them against matched spawns that did not (`openlab harness lessons ID`, `confirm ID`). A lesson
not confirmed for `lesson_ttl_days` stops being given; `openlab harness retire ID` ends one.

With `[harness] reflect = true`, a reflector (`reflector_budget_usd`, $0.50 by default) reads each spawned agent's run
when it ends: verdicts on each part of the harness (helped, hurt, no observed effect, confounded) and at most one
lesson, or none. It proposes; only you approve.

## The evaluator

A candidate harness is accepted on MLE-bench competitions it never saw ([benchmark mode](benchmark.md)):

```
openlab bench mle pools --competitions lite --private 0.5 --canary <comp>
openlab bench mle ab --arms "openlab,openlab@<candidate>,openlab*3" --seeds 3 --host gpu-box
openlab bench mle accept --candidate openlab@<candidate> --bestofn "openlab*3" --traps bench/results/harness/<t>.json
```

`openlab*3` is the baseline at matched dollars: three samples, each with a third of the budget, the one with the best
validation score of its own kept. `accept` rejects a candidate with a leak attempt, a medal on a canary, a failed trap
suite, no measured noise band (2+ baseline seeds), a private rate below the best so far less the band, always-on text
that grew without a gain beyond the band, too many edits at once, predictions that failed, or a fall out of
distribution. It says whether the candidate also beats best-of-n by more than the band. Every verdict is recorded, so
`openlab harness cost` reports cost per accepted edit with the rejected ones counted.

## Refinement

```
openlab harness refine propose-plan studies/churn --k 1
openlab harness refine task ./folder --prompt "..." --k 2 --selector-cli codex
```

One step runs the unit with the active harness (the reference), has two editors edit copies of it (one for procedure
and tooling, one for practitioner knowledge), runs each valid candidate, has a composer keep the base, adopt a
candidate or merge, citing evidence for each edit, and runs the result; at k > 1 a blind selector chooses among the
groups' outputs and may keep the base's. That is 4k runs and 3k editor and composer sessions, on copies under
`.probes/`, never the study. The step selects a harness, stored as a version; nothing becomes active until you say so.
It is your act, and spends `[harness] refine_budget_usd` at most (and at most `refine_max_fraction` of `cap_usd`).
`openlab harness replay` screens refinement policies on the recorded steps, a pre-screen, never an acceptance.

Treat refinement as an experiment: matched-compute studies have found harness evolution losing to plain parallel
sampling. Run the evaluator first, and refine only once a candidate beats the best-of-n arm.
