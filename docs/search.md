# Search, selection and lessons

A study's research runs form a tree. Openlab scores it, suggests what to grow next, picks the final candidate by a
rule sealed before the first run, and keeps lessons that rest on what the record shows. The machinery comes from
the agents leading MLE-bench (MARS, MLEvolve); the guardrails are openlab's: every node is a counted trial, the
search sees research data only, and the holdout stays out of it.

## The tree

```
openlab study run demo --op draft                       # a new line
openlab study run demo --op improve --parent run:ab12   # a change meant to do better
openlab study run demo --op debug --parent run:cd34     # a fix of a run that misbehaved
openlab study run demo --op fuse --parent run:ab12 --ref run:ef56    # borrowing from another line
```

## What to try next

`openlab study search demo` scores every research run now:

| | |
|---|---|
| G | the run's primary metric, min-max normalised over the study's runs, higher is better (0 for a run with no result or an audit that found look-ahead) |
| Q, n | the mean G of the run's subtree, and its size |
| U | sqrt(ln(N + 1) / n): how little the subtree has been tried |
| c(t) | how much U counts: 1.41 early, falling to 0.5 between 30% and 70% of the budget |
| cost | (the run's seconds / the budget's) ^ -0.07: a little in favour of quick runs |
| uct | (Q + c(t)·U) · cost |
| elite | the top three runs by G, picked with p ∝ 1/rank |
| P | w(t) for the highest uct plus (1 − w(t)) times elite p: the chance this policy picks the run |

The clock `t` is the larger of the trials used over `budget.trials` and, when `[search] minutes` is set, the
research minutes used over it. `w(t)` is 1 until `explore_until` (0.5), then falls to 0.2.

`openlab study frontier demo` turns the scores into items:

| item | when | what to run |
|---|---|---|
| `expand` | always, for the runs with the highest P | `--op improve` (or `debug`) `--parent RUN` |
| `fuse` | a line has had 3 runs without a new best | `--op fuse --parent ITS_BEST --ref OTHER_BEST` |
| `draft` | the study has had 6 runs without a new best | `--op draft` |
| `select` | the clock passed `consolidate` (0.85); drafts stop | `openlab study select` |

These are suggestions. The agent chooses, and `study take` records what it took. Every knob is in `[search]`:
`minutes`, `explore_until`, `consolidate`, `c_hi`, `c_lo`, `w_lo`, `cost_weight`, `stagnation_branch`,
`stagnation_global`.

## The final candidate

Choosing the final model by its holdout score, rather than validation, inflated medal rates by 9 to 17 points in
AIRA-dojo's measurements, and long searches overfit validation. So the rule is part of the plan:

```toml
[selection]
rule = "seed_robust"     # or "best_val"
k = 3                    # seed_robust: seeds per configuration
```

`selection` is a person's field: it is sealed with the plan, and changing it needs a signed amendment.

- `best_val` picks the best primary metric among the research runs whose audit is clean.
- `seed_robust` groups runs that differ only in their seed (`--seed`) and picks the configuration with the best
  mean over at least `k` seeds, so a lucky seed does not win.

`openlab study select demo` applies the rule and records a `selection`: the run it picked, the number of candidates
N, and the value adjusted for being the best of N. The adjustment moves the value toward worse by its standard
error times the expected maximum of N standard normals. Every field is derived again when it is written, so none can
be typed. With a rule declared, a holdout look is refused until something is selected, and is on the selected run's
code only.

## Lessons

A lesson is a rule learned from two runs. It rests on what the record shows about them:

```
openlab study compare demo result:a1 result:b2 --metric mae        # a paired difference on the rows both scored
openlab study lesson demo --rule "keep y_now as a feature" --direction improves \
    --basis comparison:c3 --tags features,lagged-target
```

| direction | the basis must show |
|---|---|
| `improves` | the first side better, with an interval that excludes 0 |
| `hurts` | the second side better, with an interval that excludes 0 |
| `no_gain` | an interval inside ±`--margin` |
| `fixes` | basis `run:<id>`: a run that is valid, whose parent had no result or an audit that found look-ahead |

A lesson whose basis does not show its direction is refused; keep it as a note. A run that applies a lesson cites
it (`--cite lesson:<id>`, or `--cite other-study/lesson:<id>`), and `openlab study lessons demo` shows each lesson's
track record: how many citing runs did better than their parent.

`openlab lessons find "lagged features for station panels"` ranks the lessons of every study in the workspace by
their rule and tags, so the next study starts from what earlier ones showed.
