# Module checks, ensembles and critiques

## Check the code before a run

A full run can take an hour. `openlab study check FOLDER` takes seconds, and catches what would fail or mislead:

| check | what it does |
|---|---|
| lint | the code names no path a run must not read (the record, the vault, `..`, a holdout file), and imports no network library when `[sandbox] network = false` |
| import | `model.py` and its modules import in the sandbox, and define `Model.fit` and `Model.predict` |
| smoke | the runner's own child job on a little data: a sample of research rows (static), or three steps from the middle of the research segment (temporal) |
| determinism | the smoke job again gives the same predictions |

```
$ openlab study check demo
module_check:5e1…  red: smoke: RuntimeError: shapes (512,37) and (38,) not aligned
$ openlab study run demo
error: this code's module check is red (module_check:5e1…): smoke: ...
```

The outcome is a `module_check` node keyed by the code's shas. A run of exactly that code is refused while its
latest check is red; any edit makes new code, which runs as usual (check it again). A check scores no research row
and spends no trial. `--code branches/<id>` checks a branch's copy.

## Ensembles

`openlab study ensemble FOLDER --members RUN,RUN --method mean|rank|weighted` combines recorded research runs on the
rows all of them scored:

- `mean` averages their predictions.
- `rank` averages rank-percentiles. It is for probabilities only, since it changes the scale.
- `weighted` picks non-negative weights summing to one (a 0.1 grid, at most four members) that best the primary
  metric. They are cross-fitted: weights fit on half the rows predict the other half, so no row is predicted by
  weights fit on it. Both halves' weights are recorded.

Members must share the protocol and the labels, and none may have an audit that found look-ahead. The ensemble is
a run of its own (op `ensemble`, its first member as parent, the others as refs) and a counted trial; the same
members and method again are cached. It has no fitted model, so it is a research candidate: compare it, learn from
it, and cite it. The selection rule considers single runs, whose code a holdout look can run.

## Critiques

A reviewer agent's findings are advice. A `critique` records them on a run, result, claim or paper, with the model,
a sha of its prompt and its cost:

```
openlab_tool study_critique folder=demo on=run:ab12 model=claude-sonnet-5-5 prompt_sha=9b2e… \
  findings='[{"where": "features.py:88", "text": "the rolling mean includes the current row"}]'
```

Each finding is then answered with `study_finding`:

- `checked`, naming the check that settled it (`by`): a module check, a run (its truncation audit), a comparison or
  a test;
- `dismissed`, with the reason.

`openlab study critiques demo` shows each critique's track record: findings confirmed (a red module check, an audit
that found look-ahead), not confirmed, dismissed, and still open. Nothing reads a critique into a verdict: a claim's
verdict and level are computed from runs, tests and evidence, as before.
