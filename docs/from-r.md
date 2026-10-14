# Coming from R

R gives a researcher solid artifacts: a script, a seed, `sessionInfo()`, an R Markdown report. Openlab keeps that
habit and moves the parts that decide whether a result can be trusted out of the script and into a record the script
cannot edit.

| In R you would | In a study |
|---|---|
| `source("analysis.R")` top to bottom | `model.py` defines `features`, `label` and `Model`; `openlab study run` decides what each call may see |
| `set.seed(1)` | `split.seed` in `study.toml`, part of every run's key |
| `sessionInfo()`, `renv.lock` | each run records the hashes of the packages it ran, the code, the data and the spec |
| `read.csv("data.csv")` | `[data.NAME]` with the `recipe` that makes the file; Openlab registers it by sha256 and checks it on every read |
| `caret::trainControl(method = "timeslice")`, `rsample::rolling_origin()` | `[split] scheme = "walk_forward"`, with `refit`, `window` and an embargo taken from the horizon |
| `createDataPartition()` for a test set | `[split] fraction` (static) or `holdout = [start, end]` (temporal): kept out of every research run, and opened only by a person's look |
| `lm()`, `glm()`, `ranger()`, `xgboost()` | anything in Python inside `Model.fit` (`statsmodels` for formulas) |
| `summary(fit)`, `t.test()` | the result: metrics with block-bootstrap intervals, and paired differences against declared baselines |
| `p.adjust()` after the fact | the alpha-investing budget: each `study test` spends it, and testing stops when it is spent |
| `knitr`, R Markdown | `openlab study paper`: your prose with bindings, generated sections, and a build that fails on an unbound number |
| `targets`, `drake` | recipes plus `openlab study materialise`; an unchanged sha reuses the registered dataset |
