# Replication of $title

Target: $key, a $kind, pinned in the record as $source (full text: $pages pages, $document).
Tolerance: $tolerance on $metric, up to $max_tolerance with a justification, in $metric's own unit; fixed before
any run.

## Next steps

1. Read the paper: `study_fetch_text` with `source=$source` and `page=N` (`layout=true` keeps table rows intact).
2. Extract each number you will check, with its page:
   `openlab study extract $folder $source --page N --section "Table 2" --quote "<the row as the page has it>" --fields '{"method": "...", "$metric": 0.0, "setting": "..."}'`
3. Write `recipes/make_data.py` for the paper's data, and the paper's protocol into study.toml's experiment (data,
   target, split, evaluate, budget, audit), then `openlab study amend $folder --reason "the paper's protocol"`. Until
   the first run an agent may amend all of it. The split must keep the paper's evaluation set out of training; if
   that set is public (every reader has seen it), it can be the research window: say so below.
4. Implement each method in `model.py` (in a temporal study also `features` and `label`), and bind its equations in
   `paper/equations.tex`.
5. `openlab study materialise $folder`, then `openlab study run $folder`. The first run ends the planning: from then on
   the protected fields need a person.
6. Fidelity, the paper's baselines rebuilt within tolerance:
   `openlab study fidelity $folder --paper <evidence id> --ours <result id> --arm model:<Name>`
7. Each claim: `openlab study replication $folder --paper <evidence id> --ours <result id> --arm <arm> --gated-by <fidelity id>`.
   openlab computes the verdict; with any gating fidelity failed it is inconclusive (rebuild not faithful).
8. `openlab study replications $folder` is the report.

## Intervals, coverage and units

A method that predicts intervals returns a `lower`/`upper` frame (declare its level: `evaluate.level = 0.9`) or
quantiles (`q0.05`, `q0.95`). `width90` and `interval_width` score the mean length (shorter is better);
`coverage90`, or `coverage` at `evaluate.level`, the share covered. Coverage is target-valued: better is closer to
its nominal level, so a coverage claim replicates when ours is within tolerance of the paper's value, and the report
says whether ours is closer to the level than the paper's baseline. A paper that prints coverage as 92.62 means
0.9262: a number its quote writes with `%` is recorded as percent; otherwise add `--unit $metric=percent` to
`study extract`. Fidelity and replication compare in the metric's unit and print the value as printed; a share above
1 with no unit is refused. The tolerance is in the metric's unit (0.02 is two coverage points).

## Protocol and deviations

The split chosen and why, and every deviation from the paper.
