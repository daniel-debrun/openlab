# bench: what the structure catches, from the records

`bench/harness.py` measures Openlab's guarantees from study records, with no model call and no network: the studies
are the templates the harness builds and runs itself, and the literature is a canned corpus.

```
python -m bench.harness                                      # every suite on the harness's own studies
python -m bench.harness --suite traps,reproducibility --study studies/weather_emos
```

| suite | what it reports |
|---|---|
| `traceability` | the share of claims (and of approved claims) whose provenance chain is complete |
| `traps` | planted traps run through the real framework; each must be refused or flagged |
| `contradiction` | recall of planted contradicting papers by a recorded contradiction search |
| `reproducibility` | every run executed again from its record; the share that matched |
| `cost` | per study: wall time, actions, trials and holdout looks against their budgets, reproductions |

A study given with `--study` is copied to a temp dir first. Results go to `bench/results/harness/<UTC time>.json`.
Needs bwrap (the runner's sandbox).

`bench/paperbench/` is the PaperBench adapter: the lead's instruction and the instrument it hands in through
([docs/benchmark.md](../docs/benchmark.md#paperbench-an-instruction-and-an-instrument)).
