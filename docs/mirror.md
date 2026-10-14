# A Weights & Biases mirror

The record stays the record: hash-chained, signed, and the only thing openlab checks. A W&B mirror is a view of a
study's runs for dashboards, comparisons and teammates who live in W&B.

```
pip install -e ".[mirror]"        # installs wandb
wandb login                       # or WANDB_API_KEY, or ~/.netrc
```

With `wandb` installed and a key on the machine, every recorded run is mirrored. To set where it goes, or to turn it
off for the workspace, use `openlab.toml`:

```toml
[mirror]
wandb = true            # default: on when W&B has a key here; true mirrors offline even without one; false: off
project = "openlab"      # default "openlab"
entity = ""             # default: the key's own
sync = true             # false: write offline runs only, and upload them with `openlab mirror sync`
```

A study opts out in its own `study.toml` with `[mirror] wandb = false`. `OPENLAB_MIRROR=off` turns the mirror off
for a process.

## What is sent, and what never is

| W&B | From the record |
|---|---|
| run id | the run's key: a cached rerun is the same W&B run |
| name | the run node's id |
| group | the study |
| job type | the run's op (`draft`, `improve`, ...), else its segment |
| tags | segment, branch, op, the audit's verdict |
| config | key, parent and refs, branch, seed, segment, audit, sandbox, the spec, code and data shas, the machine, the packages |
| summary | each metric's value and interval, and each paired difference with its interval and p |

Never data: not the rows, not the predictions, not the holdout. A W&B sweep is not a way to run trials: every trial
is an openlab run, counted against the study's budget.

## When W&B is down

After a run is on the record, openlab starts a separate process that writes an offline W&B run under the study's
`.openlab/wandb` and then tries to upload it. The run itself never waits for it and never fails because of it.
What did not upload stays on disk:

```
openlab mirror status demo        # on or off and why, and how many runs have not synced
openlab mirror sync demo          # upload them now
```

Each attempt is logged in `.openlab/mirror.jsonl`.
