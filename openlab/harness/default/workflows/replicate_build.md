Build the replication of {title} ({key}) in the openlab study {folder}.
A person approved the plan ({folder}/{md}; action {aid}, sha {sha}). The target and its tolerance ({tol:g} on {metric},
up to {max:g} with a justification) are fixed in the record, and each paper number below is recorded as evidence
quoted from its page. Do not edit {file}; the methods, the metric and the tolerance are the plan's.

Methods:
{methods}

Numbers to check:
{claims}

Data: {data_name}, from {data_source}. Protocol: {protocol}

Steps (the study's README.md has the commands in full):
1. Data: write recipes/make_data.py for the paper's data, and set study.toml's data, target, split and evaluate to
   the paper's protocol; record it: `openlab study amend {folder} --reason "the paper's protocol"`.
2. Methods: one file per method under methods/ (methods/<name in lower case>.py, a class with fit and predict, its
   docstring citing the page and the equations), methods/__init__.py, and model.py importing every class, with
   `Model = {proposed}`. Bind each equation a method implements in paper/equations.tex as
   \labeq{{<name>}}{{model.<function>}}{{<the equation as printed>}} (model.py imports the function too).
3. Baselines first: `openlab study materialise {folder}`, then `openlab study run {folder}`. Record a fidelity per
   baseline number: `openlab study fidelity {folder} --paper <evidence> --ours <result> --arm model:<Name>`. A failed
   fidelity means the rebuild differs: fix the method and run again (every trial counts); never the tolerance.
4. The paper's numbers: `openlab study replication {folder} --paper <evidence> --ours <result> --arm <the claim's
   arm> --gated-by <fidelity id>` (repeat --gated-by for each baseline fidelity). openlab computes the verdict.
5. `openlab study paper-lint {folder}` checks each \labeq against its function; fix every eq.gate finding.
6. Write every deviation from the paper under "## Protocol and deviations" in README.md (kept by the report), then
   `openlab replicate report {folder}`, which writes README.md and results/ from the record.
7. The evaluation period (the scored rows) is scored only by `openlab study run`, each run a counted trial. A
   check of your own (a script, a notebook) may use the training and validation data, never the evaluation
   period's labels: an off-record score there is an uncounted trial, and a choice made from it is selected on the
   test set. If you ran one, list it under the deviations and say what it showed.
8. Budget: ${usd:g}, {minutes:g} min, {trials} trials. A person's act (more budget, a holdout look) goes to
   openlab_agent_request; if nothing can go on without the person, say so and end. After the first runs,
   `openlab replicate status {folder}` shows the pace against the plan's estimate: when it says the runs left do not
   fit the minutes left, ask the person (more minutes, or which runs to drop) before cutting seeds or settings.

What openlab handles for you (openlab://reference has the details):
- A number the paper averages over runs (horizons, datasets, seeds): one run each (amend study.toml between them,
  e.g. target.horizon), then give every result: `--ours <r1> --ours <r2> ...`; openlab takes their mean.
- A number the paper prints per station, dataset or class: set evaluate.by = "key" before the first run (the key
  column names the entity); the result then scores each entity too, and `--group NAME` checks that entity's score.
- A model refitted at every decision on a growing archive: define `update(self, X, y)` beside fit; openlab then
  passes only the new rows (the truncation audit refits from scratch to check it).
- A predictive distribution over several outputs (a frame whose mu/sigma, q0.05..., m0... and weights w0...
  columns hold one array per row): the walk scores it step by step, keeping the mean and each declared metric's
  per-row statistics, so declare every metric you need in evaluate.metrics before the run. `crps` is the fair
  CRPS of equal members; `crps_empirical` (and `crps` with weights) the empirical distribution's.
- Keep long runs under the minutes left; `openlab study run` prints each trial as it is spent.
Finish with each number's verdict, the fidelities, and the deviations.
