# Openlab for agents: the reference

The detail behind `AGENTS.md` (served as `openlab://guide`). The guide is the contract; this page is the lookup. Every
verb below is a `openlab_study_*` tool, `openlab tool study_VERB key=value`, and (mostly) `openlab study VERB FOLDER`.

## study.toml

| Section | Keys |
|---|---|
| `[study]` | `name`, `question`, `mode` (`temporal`, `static` or `review`) |
| `[data.NAME]` | exactly one of `provider` (+ `endpoint`, `args`), `snapshot`, `file` (+ `recipe`), `instrument` (+ `output`: a file a run of the pinned instrument wrote); then `time`, `key`, `known_at`, `vintages = [start_col, end_col]`, `licence` |
| `[target]` | `horizon` (temporal, e.g. `"1d"`, `"30h"`, `"1M"`) or `column` (static) |
| `[split]` | temporal: `scheme = "walk_forward"`, `decisions`, `refit`, `window`, `research = [start, end]`, `holdout = [start, end]`, `embargo`, `burn_in`, `train_from`. Static: `scheme` (`kfold`, `stratified`, `group`, `insample`: the model is fit on every research row and scores those same rows, as some papers do; the run's audit says `in-sample` and every fidelity and replication on it is marked so), `k`, `fraction` (the holdout; 1/k), `by`, `seed`, `reseed` (true: each run's `--seed` deals the research folds anew, for a paper's mean over random splits; the holdout keeps `seed`) |
| `[evaluate]` | `metrics` (the first is primary), `prediction` (`point`, `prob`, `gaussian`, `quantiles`, `ensemble`, `interval`, `frame`: columns of the model's own, read only by the study's own metric with `reads = ["frame"]`), `level` (the central interval coverage and interval_width read, e.g. 0.9), `by = "key"` (each entity's own scores besides the pooled ones: a station, a dataset; `openlab study show FOLDER RESULT` prints them), `baselines`, `interval`, `alpha`, `block`, `policy_metrics`, `policy_baselines` |
| `[budget]` | `trials`, `holdout_looks` (default 1) |
| `[metrics.NAME]` | the study's own metric: `better` (`lower` or `higher`), `reads` (default `["point"]`), `unit`, `description`; its function `NAME(y, pred) -> float` is in `metrics.py`. openlab writes `code_sha` (metrics.py's sha256) into the spec |
| `[sandbox]`, `[audit]` | `network` (default false), `timeout`, `threads` (default 1; recorded in the run's env); `n`, the audited steps (default 20) |
| `[inference]` | the error budget: `alpha`, `wealth`, `payout` |
| `[instruments]` | `alias = "NAME@VERSION"`: the registered instrument versions the study runs and reads (see Instruments below) |
| `[forecast]`, `[schedule]`, `[monitor]` | `model`, `gate` (SQL predicates over the predictions, table `t`), `expires`; `cron`, `refresh`; `cron`, `notify` |
| `[review]` | `question`, `scope`, `sources` (`openalex`, `arxiv`, `crossref`), `queries`, `since`, `include`, `exclude` (`[{id, text}]`), `citation_hops`, `n`, `coverage_floor`, `extract` (field types), `coordinates`, `comparable_on` |

`known_at` is an expression over a time column (`"date - 1d + 18h"`), a column name, or the `vintages` pair. Metrics:
`mae`, `rmse`, `r2`, `mape`, `crps`, `crps_empirical`, `pinball`, `coverage90`, `coverage`, `width90`, `interval_width`, `auc`, `brier`, `logloss`, `accuracy`,
`balanced_accuracy`, `f1`, `macro_f1`, `precision`, `recall`, `hit_rate`; series metrics `mean`, `total`, any a
package registers (`openlab.study.metrics.register`), and the study's own (`[metrics.NAME]` and `metrics.py`: the seal
holds the file's sha, and openlab imports it only in the sandbox, for runs, reproductions and comparisons; a name of
openlab's own is refused). Baselines: `persistence`, `climatology`, `majority`, `mean`, or `model:ClassName` from
model.py.

Protected fields, which only a person amends: `split`, `budget`, `sandbox`, `audit`, `forecast.gate`, the primary
metric, `data.*.known_at`, `review.coverage_floor`, `review.comparable_on`, `inference`, `metrics` (so an edit of
metrics.py after the seal is a person's amendment), `instruments` and `data.*.instrument`.

## model.py

- `features(view, t)` returns a scalar dict (one entity) or a polars frame (the key columns plus features).
  `label(view, t)` has the same shape, with `y`. Openlab joins the two on the key.
- `X` is what `features` returned, plus `_t` (the decision time) and `_key` (the entity key as a string). Static mode
  has no `_t`. `y` is a 1-D numpy array, or 2-D (rows x outputs) when the label has several outputs: `label` returns
  `y` as a list per row (one width; pad with NaN), and `predict` returns a 2-D array. Metrics score every output;
  the bootstrap resamples rows.
- `predict(X)` returns a 1-D array (`point`, `prob`) or a frame: `mu`, `sigma` (`gaussian`); `q0.1`, `q0.5`, ...
  (`quantiles`); `m0`, `m1`, ... (`ensemble`), with optional weights `w0`, `w1`, ... (a weighted empirical
  distribution); `lower`, `upper` (`interval`, at `evaluate.level`). `crps` scores equal members by the fair CRPS;
  `crps_empirical`, and `crps` on weighted members, score the members' empirical distribution (properscoring's
  estimator, the one most papers use).
- With several outputs per row, each frame column holds one array per row (`pl.Series("m0", rows_x_outputs)`). A
  temporal walk whose step predicts such a distribution writes each step to disk and keeps only the predictive mean
  and the per-row statistics of the metrics in `evaluate.metrics`: declare every metric you will need before the
  run (a metric without per-row statistics, or the study's own, is refused for it).
- Optionally, `update(self, X, y)` beside `fit`: a temporal refit whose training rows only grow (an expanding
  window) then passes the new rows alone to `update` instead of all rows to `fit` (a retrieval archive, an online
  model). The truncation audit fits from scratch at the steps it replays, so an update that a fresh fit would not
  reproduce shows as a look-ahead mismatch.
- Optionally, `act(pred_row, view, t) -> float` gives the value of acting on one decision, scored by
  `policy_metrics` against `policy_baselines` (`model:Name` classes with an `act`).
- model.py may import sibling modules, and every one is hashed into the run.
- A package the study needs (utide, statsmodels, torch, ...): `openlab env add PKG` installs it into the environment
  the sandbox runs, never changing an installed package's version (a conflict is refused). Never `pip install`
  otherwise. The run records the versions of what its code imports.

## The temporal walk

- Decisions: `split.decisions` is a cadence (`daily`, `weekly`, `monthly`) or `"<dataset>.<column>"`, every distinct
  value of a column (e.g. each hourly forecast origin). Only decisions inside `research` (or `holdout`) are scored,
  and a decision t counts only if t + horizon <= the segment end.
- Fits: the walk refits at the segment's first decision and again once `refit` has passed (`refit = "1h"` with
  hourly decisions: every step). A fit at s trains on the decisions t_i with t_i + horizon <= s (every label already
  known), within `window` (`"expanding"`, or a duration). `train_from = <date>` (before the research start) adds
  decisions that only train: the paper's train and validation windows ahead of its test set, never predicted or
  scored. `burn_in` instead predicts the first stretch of the segment but leaves it unscored.
- At each decision t, `features(view, t)` sees the rows known at t (`known_at <= t`), and `label(view, t)` sees
  the rows known at t + horizon. The truncation audit replays sampled steps on data cut at t, fitting from scratch,
  and compares the model's and each `model:` baseline's predictions there to a relative 1e-9: an `update` must give
  what a fresh fit gives to that precision (sum in the same order; avoid running sums that drift), and `predict`
  must give the same output twice on the same input (seed any randomness).
- `horizon` is a duration (`"96h"`); a label of the next H steps is H rows of the view after t.

## What a run does

1. It checks the seal and refuses an unamended edit.
2. It checks the budget: an identical key returns the cached result, and a new key past `budget.trials` is refused.
3. It stages the data cut to the segment. Research rows only, and labels only up to the segment's end.
4. It runs the child (bwrap: no network, `.openlab/`, `data/` and `.git` masked, and a nested study's too; `unsandboxed` where bwrap cannot
   start) over the temporal steps or static folds.
5. The truncation audit replays `audit.n` sampled steps in fresh children that hold only the data known then. A
   mismatch makes the audit not clean.
6. It scores the run: metrics, baselines, block-bootstrap intervals, and paired differences with p-values.
7. It writes `run` (key, code, data and env hashes, sandbox, audit, stdout, predictions, `choices`, `branches_from`
   the previous run), `result`, and `model` (registered in `.openlab/mlflow` with its lineage).

## Refusals you will meet

| You did | Openlab says | Do |
|---|---|---|
| edited study.toml after the seal | `study.toml was edited after the seal` | undo it, or `study_amend` with the reason |
| amended a protected field | `<field> can only be amended by a person` | undo that part, or ask the person |
| a file dataset with no recipe | refused at materialise | write `recipes/<name>.py` |
| ran past `budget.trials` | refused | stop and report, or `agent_request(kind="budget", args={...})` |
| asked for a holdout look | refused: a person's act | `agent_request(kind="holdout", ...)`; carry on meanwhile |
| choosing what to run next | `study_search` scores the tree; `study_frontier` offers expand / fuse / draft / select | run with `op` and `parent`; cite the lessons you apply (`lessons_find`, `cites=[...]`) |
| a comparison shows a clear difference | record it as a lesson | `study_lesson(rule, direction, basis=comparison:<id>, tags)`; refused if the interval does not show it |
| a run says it is queued | it waits for cores, memory or GPUs other runs hold (or a person reserved) | carry on with other work; `box()` shows the machine's queue and why |
| a run needs more than this machine has, or would end after its availability | refused before it runs | `agent_request(kind="compute", args={"gpus": 2, "hours": 6})`; the person adds a machine |
| typed a claim's verdict, level or number | refused, with the computed value | let Openlab compute it; bind the input |
| rested a claim on a run whose audit is not clean | refused | fix the look-ahead, run again |
| tested past the error budget | refused | stop; the budget is the point |
| extracted a quote not on that page | refused | quote the page as it is (`layout=true` keeps table rows) |
| compared results under different protocols | refused | compare like with like, or record why they differ |
| a hypothesis that matches a dead end | refused | link `differs_from` and say what is different |
| built a paper with findings | refused, with every finding | fix the prose or the binding; only a person waives |

## Claims, tests and the question tree

| State | Rests on | Verdict |
|---|---|---|
| `observed` | `rests_on` evidence or results (at least one) | supported by its checked ground, unless tests say otherwise |
| `derived` | `rests_on` claims, results or evidence, with `expression` over `inputs` (`{name}` → `"<node-id>#<path>"`) | the weakest premise's verdict |
| `inferred` | `rests_on` claims | from the tests of the hypotheses it `answers`; else untested |
| `hypothesized` | `tests` one hypothesis | from that hypothesis's tests |
| `unknown` | `searched` one coverage | untested |

Tests of the claim itself decide first. Optional links on any claim are `needs` (assumptions) and `answers`
(questions, hypotheses). `study_test(of, result,
baseline, alpha_j)` tests a claim or hypothesis on a result's paired difference: supported when it rejects with the
model better, refuted when it rejects with the baseline better, inconclusive otherwise. Each test spends the alpha budget (alpha-investing wealth).
`alpha_j` is at most `wealth / (1 + wealth)`, and a rejection earns the payout back. `study_break(assumption,
shown_by)` puts every claim that needs the assumption at risk. `study_contra(claim, queries)` records a contradiction
search, which ends `found`, `none_found` or `insufficient` (below the review's coverage floor). `study_tree` derives
each question's status from its children. `study_frontier` ranks what to do next by expected information per hour,
and `study_take` records the choice.

Level: `measured` when every result a claim rests on comes from the holdout, else `recorded`; `+reproduced` once each
of their runs has a matched reproduction.

## Literature

`study_discover` runs the sealed `[review]` queries on each index (and the citation hops), deduplicates by DOI,
arXiv and OpenAlex ids, links preprints to their published versions, flags retractions and writes a coverage node.
`study_fetch_text(source, page, layout)` keeps the open-access full text by page. `study_screen(source, decision,
criterion, rationale, quote)` takes `include`, `exclude` or `maybe`, and a person settles a `maybe`.
`study_extract(source, page, quote, fields, section)` records evidence whose quote and numbers are checked on that
page. `study_lattice` is protocols × methods with the gaps, `study_compare` compares two results on one metric
(same protocol only), and `study_recall(gold)` checks recall against a published review's references.
`study_monitor` reruns discovery, checks retractions and fills forecasts (`[monitor]` schedules it).

## The paper

The paper is one LaTeX document, `paper/paper.tex`: `study_paper_init(folder)` writes it from the template (the
abstract and the usual headings). It is yours: load any package with `\usepackage`, and include images from the
study's folder with `\includegraphics` (a path from `paper/`, e.g. `figures/chart.png`, or from the study; png, jpg
or pdf, never from `.openlab/`). Each included file is copied into the build, its sha256 recorded, and the paper
lists it as a figure from a file: openlab did not draw it and checks no number in it. Only the document body is
linted. `\labsection{<name>}` puts in the text of a section openlab writes from the record (`data`, `methods`,
`literature`, `results`, `limitations`, `reproducibility`, `checklist`). A `paper/paper.toml` beside it sets only
`digits`, `years`, `allow` and `checklist`. (The older layout, still built: `paper.toml` with `title`, `authors`,
`abstract`, `sections`, and the prose in `paper/sections/<name>.tex`; openlab writes each section's heading.) The
bindings:

| Macro | Inserts |
|---|---|
| `\lab{<id>#<path>}` | a number or value |
| `\labclaim{<claim>}` | the statement, with its state and verdict |
| `\labtab{<result>}`, `\labfig{<result>}` | a table, a figure |
| `\labcite{<source>}{<quote>}` | a citation whose quote must be in the source |
| `\labeq{<name>}{<module.fn>}{<latex>}` | an equation, checked against the Python function on probe inputs |

A `<path>` is the dotted path into the node's body as `study_node` shows it. For a result: `metrics.brier` (the
estimate with its interval), `metrics.brier.value`, `baselines.model:Market.brier`, `paired.model:Market` (the
difference with its interval), `policy.metrics.mean`, `policy.series.pnl_3c`. A path that ends at a table is
refused, naming its entries.

The lints: `number.typed` (a digit in the prose outside math and bindings), `wording.significance`,
`wording.comparison`, `wording.causal` (each needs the claim that licenses the word), `wording.novelty` and
`wording.proof` (always), `cite.quote`, `eq.gate`, and `claim.untraceable` (a bound claim whose chain is
incomplete). Unresolved and stale bindings never pass. Any other finding
passes only with a person's waiver, which the paper prints. A name with digits (`H0`, `pnl3`) and a layout
argument (`[11pt]`, `{0.5\linewidth}`, a table's column spec) are not numbers. `study_paper(draft=true)` renders the
paper as it stands into `paper/build/draft.pdf` and lists the findings, recording nothing: write against it.
`study_paper_lint` checks without building; `study_paper` builds `paper/build/main.pdf` with latexmk, writes the
`paper` node and signs the bundle, refused while a finding is open.

## Replication

`study_replicate(paper, folder, mode, methods, metric, tolerance)` pins the paper (an arXiv id, DOI or PDF), fixes the
tolerance before any run, and scaffolds `model.py` (one class per method; the last is `Model`, the others baselines),
`paper/equations.tex`, a recipe stub and a README with the steps. Extract the paper's numbers with their pages, then
call `study_fidelity(paper=<evidence>, ours=<result>, arm=model:<Name>)` for each baseline and
`study_replication(..., gated_by=[<fidelity>])` for each claim. A claim is inconclusive while any fidelity gating it
fails. A number the paper averages over runs (horizons, datasets, seeds) takes every run's result: `ours` as
comma-separated result ids (CLI: repeat `--ours`); openlab compares the paper's number with their uniform mean (its
interval the mean of theirs, widened by the seeds' spread: runs on the same data and split that differ only in
`--seed` add t * sd/sqrt(n), so a handful of seeds gives an honest, wide interval), and gates must cover the same set of protocols. A number
the paper prints per station or dataset: set `evaluate.by = "key"` before the run, then pass `group` (CLI
`--group NAME`) to fidelity and replication; a claim is gated by fidelities of its own group. `kind = "reproduction"` with `code` runs the authors' code instead of rebuilding from the text.
Coverage is target-valued (best at its nominal level): a coverage claim replicates within tolerance of the paper's
value, and "better than the baseline" means closer to the level. A share the paper prints in percent is percent when
the quote shows `%`; otherwise pass `units={"coverage90": "percent"}` to `study_extract` (or `paper_unit="percent"`
to `study_replicate`). Values are compared in the metric's unit, and the tolerance is in that unit.

## Forward runs and reproduction

`study_forecast` predicts at the latest decision time with the pinned model, checks `[forecast] gate`, and writes
`forecasts/<t>.csv` with a signature (`openlab verify <file>.sig.json`). A failed gate is recorded and signed as
failed. `study_fill` records outcomes once labels are known. `study_reproduce(run | all)` runs recorded runs again
from the record alone and compares the predictions. `study_bundle(out)` writes a signed tar, and `openlab study
reproduce --bundle <tar>` reproduces every run from it in a temporary directory.

## Plans, questions and the release

`agent_plan(nodes, note)` reports your whole plan as a new version. A step is `{id, title, kind, needs, state, links,
owner}`: `id` 1-40 letters, digits, `_ . -`; `kind` one of review, data, run, branch, test, select, person, paper,
critique, check, lesson, other; `needs` the plan's own step ids, with no cycle; `state` one of planned, active,
waiting, done, failed, dropped; `links` up to 12 record ids (`run:1a07c3`, `test:9b0e`, `dataset:c84f`, `inbox:7`,
`ticket:43`); at most 60 steps. `agent_plan_node(id, state, links, note)` moves one step of your latest version.
Both append to `.openlab/agents/<id>/plan.jsonl` and log an event on the control store.

Openlab checks the links, not you: a step with no links is your word only; a link that is not on the record, or a
step you call done on a run whose audit found look-ahead, shows that the record disagrees.

`agent_request(kind="question", reason, args={"choices": [...], "multi": false, "other": true})` asks the person
(2 to 8 choices; `other` lets them type their own answer and is on unless set false). The answer is in
`agent_inbox_item(id)`'s `result` (`picks`, `other`); `agent_wait(ids=["inbox:7"])` waits for it, up to 300 s (with agents and `job:` runs in the same call). An
answer is advice: it is not signed, and a question cannot be approved or denied.

When a study's lead ends done, the running `openlab show` compiles `<study>/release/`: `README.md`, `paper.pdf` (only
when every binding resolves and every lint finding is fixed or waived), `code/`, the signed bundle and
`MANIFEST.json`. `[show] compile_on_done = false` in `openlab.toml` turns it off.

## Instruments

An instrument is a registered, immutable program that acts outside the sandbox (a broker's paper account, a collector,
a CLI). Its version `NAME@VERSION` is the sha of its snapshot, entry, settings and environment fingerprint.
- `instrument_list`, `instrument_show(ref, verify)`, `instrument_verify(ref)`: what is registered, and whether a version
  still hashes to its registration.
- `instrument_run(ref, args, study, note)` runs a version after verifying it, and records the run with the hashes of
  its logs and outputs. With `study`, the study must pin the version (`ref` may be its alias), and the run is also on
  the study's record. Drift is refused, naming what changed.
- `instrument_runs(ref, study, n)`: the recorded runs.
- A dataset `[data.x] instrument = "alias"`, `output = "file"` materialises from the latest successful run of the
  pinned version that wrote the file.

Registering a version is the person's act: file `agent_request(kind="instrument", args={name, entry, source,
capabilities, secrets, outputs, state, ...})`. So is every run of a `live-order` instrument: file
`agent_request(kind="instrument_run", args={instrument, args, study})`. Pinning a version is a protected amendment.

## Workspace tools

`ws_list`, `ws_read`, `ws_write`, `ws_move`, `ws_delete` work on the workspace's files (the vault is never listed or
read); `notes_add` and `notes_list` keep the scratchpad `NOTES.md`; `agent_activity` and `usage` say what the agents
did and what it cost. Over MCP call them by name: `openlab_tool(name="ws_read", args={"path": "study.toml"})`;
`openlab_tools(query="notes")` lists the tools that match, each with its first sentence and arguments, at most 50 per
reply. They pass the same refusals as a listed tool.

**Bounds.** Replies carry 200k characters of a file (`ws_read`, with `truncated` and `chars_withheld`). Never write
back a file you read truncated.
