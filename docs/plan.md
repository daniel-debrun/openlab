# The plan: study.toml

`study.toml` says what the study will test before any data is scored. The first run seals it into a `plan` node with
its sha256. After that, the file must equal the sealed spec plus its amendments, or the next run is refused.

```toml
[study]
name = "demo"
question = "Does today's signal x predict tomorrow's y at each site better than persistence and climatology?"
mode = "temporal"                   # or static, or review (a literature review, no experiment)

[data.panel]                        # one of provider / snapshot / file; a file needs the recipe that makes it
file = "data/panel.parquet"
recipe = "recipes/make_data.py"
time = "date"
key = ["site"]
known_at = "date"                   # when a row became known: a column, "date - 1d + 18h", or vintages

[target]
horizon = "1d"

[split]
scheme = "walk_forward"
decisions = "daily"
refit = "30d"
window = "expanding"
research = [2024-01-01, 2024-12-31]
holdout = [2025-01-01, 2025-06-30]  # a research run never loads these rows

[evaluate]
prediction = "point"
metrics = ["mae", "rmse"]           # the first is primary
baselines = ["persistence", "climatology"]
interval = "block_bootstrap"

[budget]
trials = 20
holdout_looks = 1
```

The static template uses `scheme = "stratified"`, `k = 5` and `fraction = 0.2`, so the holdout is a fifth of the rows,
carved out of the study by a seeded draw. Every key is listed in `openlab/AGENTS_REFERENCE.md` (served to agents as
`openlab://reference`).

## Embargo

A decision at `t` belongs to a segment only if its label, known at `t + horizon`, falls inside that segment. A row
whose decision is `t_i` trains a refit at `s` only if `t_i + horizon <= s`. A label therefore never trains a model that
predicts before it was known, and the research segment never borrows a label from the holdout.

## Amendments

After the seal, edit `study.toml` and record the edit:

```
openlab study amend demo --reason "add rmse as a secondary metric"
```

The amendment stores the diff and the reason, and the effective spec becomes the plan plus every amendment, in order.
Some fields protect the result, so a change to them is a person's act: `split`, `budget`, `sandbox`, `audit`, the
primary metric, `known_at`, `forecast.gate`, `inference` and the review's coverage floor and comparison
coordinates. A person runs `openlab study amend demo --person --reason ...`, which shows the change and asks for
the person's passphrase; an agent files an `amend` request in `openlab inbox` instead. Methods and the paper show
every amendment.

## Budgets

`budget.trials` counts distinct research runs. A run whose key (spec, code, data, the packages it ran, seed) matches
an earlier run returns the cached result, and the first line of every run says whether it spent a trial and why.
A budget belongs to the question, not the folder: a copied record is refused, and a new study whose datasets and
target equal a sealed sibling's must declare `[study] continues = "../demo"` (one shared count) or be declared a
fresh question by a person (`[study] fresh = "<why>"`, recorded with `openlab study amend --person`). `budget.holdout_looks` counts a person's looks, and a holdout run after a change to the
code or spec needs a new look. `[inference]` sets the alpha budget that tests spend (see [Claims](claims.md)).
Agents get their own budgets in $, minutes and turns (see [Supervising agents](supervising-agents.md)).
