# The first five minutes

This page sets up a workspace, makes your person key, and takes one study from scaffold to a holdout look on
generated data, with nothing to download. The outputs are real; your ids and times will differ. Install first
(`pip install -e ".[study]"`; see [Install](install.md)).

## 1. A workspace

```
mkdir research && cd research
openlab init --claude          # .mcp.json, openlab.toml, OPENLAB.md, Claude Code hooks, five role agents
openlab person init            # once per machine: a handle and a passphrase
openlab person pin             # pin the key's fingerprint in this workspace (asks for the passphrase)
openlab doctor                 # hooks: yes, person key: yes, sandbox ok, ...
```

A workspace is a folder with a `openlab.toml`. Its studies share agent supervision, budgets and the inbox. Your person
key signs your person acts: a holdout look, an approval, a protected amendment, an inbox decision. An agent on your
account can read the encrypted key file but cannot sign without your passphrase.

If you use Claude Code, open two tmux panes here: `claude` in one, `openlab watch` in the other. Everything below
works the same whether you type it or your agent does; the record names who did what.

## 2. Scaffold and register the data

```
$ openlab study new demo --template temporal
new study demo in ~/research/demo
next: `openlab study materialise ~/research/demo`, then `openlab study run ~/research/demo` (the first run seals study.toml)

$ openlab study materialise demo
dataset:10312d449e63  panel: 2192 rows, sha d74309e84d36
```

The template asks whether today's signal `x` predicts tomorrow's `y` at four sites better than persistence and
climatology. Research runs cover 2024, and the holdout is the first half of 2025. `study new` writes `study.toml`,
`model.py` and `recipes/make_data.py`, and runs `git init` unless the folder is already in a repository. The recipe
writes `data/panel.parquet`; openlab hashes it and checks the hash again every time a run reads the file. A data file
with no recipe is refused. `--template static` is the cross-validated variant, and `--template review` a literature
review.

## 3. Run

```
$ openlab study run demo
trial 1 of 20: the study's first run
result:7d0a9a3deb0a  audit: clean  scored rows: 1340
  mae            0.4231 [0.4058, 0.4404]
  rmse           0.526 [0.506, 0.5444]
  baseline climatology: mae 0.7497 [0.7031, 0.7934], rmse 0.9485 [0.892, 1.009]
  baseline persistence: mae 0.9021 [0.8501, 0.9576], rmse 1.145 [1.079, 1.212]
  model - climatology (mae): -0.3267 [-0.3648, -0.2855], p 0.002, better: model
  model - persistence (mae): -0.479 [-0.5308, -0.433], p 0.002, better: model
```

The first run **seals** `study.toml`: the plan is frozen before any result is seen. The run then walks forward one day
at a time, refitting every 30 days on rows whose labels were already known. `features` gets a view of what was known
at each decision time. The truncation audit replays 20 sampled steps in fresh processes that hold only the past, and
the outputs must match. The baselines are scored on the same rows, with block-bootstrap intervals and paired
differences. It takes about 6 seconds.

## 4. Try a variant

Edit `model.py` (here, `COLUMNS = ["x"]`) and run again with a note:

```
$ openlab study run demo --note "x only"
trial 2 of 20: new key because code (model.py f51c3a04→f3b099f4) changed since run:2db2394f41c5
result:a6041e0e2db0  audit: clean  scored rows: 1340
  mae            0.4222 [0.4045, 0.4394]
  ...

$ openlab study runs demo
2 of 20 trials: a result chosen among them is selected from 2 trials
trial   1  run:2db2394f41c5  code v1  audit clean  mae 0.4231 [0.4058, 0.4404]  vs climatology -0.327 (p 0.002)
trial   2  run:40a3b4f05915  code v2  audit clean  mae 0.4222 [0.4045, 0.4394]  vs climatology -0.328 (p 0.002)  changed model.py  "x only"
```

Each distinct run is a **trial**, counted against `budget.trials`, and the first line says why a trial was spent.
Running the same code, data and spec again is cached and free (`cached: no trial spent`). `study runs` shows every
variant you tried, not just the best, and `study compare demo A B --metric mae` pairs two of them on their shared
rows.

## 5. Read the record

```
$ openlab study why demo result:a6041e0e2db0
result:a6041e0e2db0  by person:sam (person)  mae 0.4222 [0.4045, 0.4394]; rmse 0.5251 [0.5049, 0.5436]
  dataset:10312d449e63  person:sam (person)  panel: 2192 rows, sha d74309e84d36
  plan:2535f4a37c7f  person:sam (person)  sealed spec effbd6859870
  run:2db2394f41c5  person:sam (person)  research run, key a51817437fc1, bwrap
  run:40a3b4f05915  person:sam (person)  research run, key a7e1fea3ef89, bwrap: x only
```

Every node names who wrote it. Commands typed at your terminal are labelled `person:<handle>`; an agent's are
labelled with its id (`cc-7c0ffee1.agent1 (openlab-experimenter)`). The label is attribution; a person's act is the
signed one.

## 6. Hit a rule

Change `trials = 20` to `50` in `study.toml` and run:

```
$ openlab study run demo
error: study.toml was edited after the seal
meaning: changed: budget.trials
fix: undo the edit, or record it with `openlab study amend <folder> --reason ...`

$ openlab study amend demo --reason "more trials"
error: budget can only be amended by a person
meaning: changed: budget.trials; split, budget, sandbox, the primary metric and known_at protect the result
fix: undo that part of the edit, or ask a person to amend
```

Every refusal says what was refused, what it means and how to fix it. The budget protects the result, so only a
person changes it: `openlab study amend demo --person --reason ...` asks for your passphrase. An agent that needs it
files an inbox request instead. To carry on, put `trials = 20` back in `study.toml` by hand: `study new`
makes no commit, so git has no copy to restore it from.

A second study of the same question on the same data cannot start a fresh budget either:

```
$ openlab study run demo2
error: demo2 asks the question of demo (~/research/demo) of the same data
meaning: the same datasets (sha d74309e84d36) and target; a second study would start a fresh trial budget and a fresh holdout (2 trials and 1 holdout looks are spent there)
fix: declare [study] continues = "../demo" (its trials and looks then count here too) and record it with `openlab study amend`; or a person declares a new question: [study] fresh = "<why>", recorded with `openlab study amend --person`
```

## 7. Claim and test

```
$ openlab study claim demo "The x-only model has lower MAE than climatology." --state observed --rests-on result:a6041e0e2db0
claim:2ee3feda334b  observed, supported, recorded, selected from 2 trials: The x-only model has lower MAE than climatology.

$ openlab study test demo claim:2ee3feda334b --result result:a6041e0e2db0 --baseline climatology --alpha 0.02
test:40324b1c99e6  primary: p 0.001998 at alpha 0.02 vs climatology, rejected; alpha earned 0.05, alpha budget left: 0.0975
```

Openlab computed the claim's verdict and level; you cannot type them. "Selected from 2 trials" travels with the claim.
The test spent part of the alpha budget (the error budget tests draw from), and the rejection earned some back. The
level is `recorded` because the result comes from research data.

## 8. The holdout look (yours)

```
$ openlab study holdout demo --reason "the model is fixed"
openlab person act: holdout look 1 of 1 on demo
  spec sha effbd6859870
  code sha model.py f3b099f457b3
  data sha panel d74309e84d36
  reason: the model is fixed
passphrase to sign (empty cancels):
running the holdout look...
result:15745b86905e  mae 0.3811 [0.3646, 0.3984]; rmse 0.4795 [0.4593, 0.5]

$ openlab study verify demo
ok: 25 nodes, head 4710e4326d54
person signatures: 6 valid, key ed25519:9083:5b62:4a0d:81a7:53d0:04af:1ac4:b873
```

A **look** is one scoring of the held-out rows. It shows what it will sign (the code, data and spec hashes), asks for
your passphrase, and is recorded before anything is scored. A change to the code or spec after a look needs a new
look, counted against `budget.holdout_looks`. Claims that rest on the holdout result are `measured`. An agent that
asks for a look is refused and files an inbox request; you approve it with `openlab inbox approve N`.

Next: [Supervising agents](supervising-agents.md) for running this with a tree of agents, [The plan](plan.md) for
every key of `study.toml`, and [Claims, tests and the paper](claims.md) for reporting.
