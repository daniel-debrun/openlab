# ML models in a study

`model.py` is ordinary Python, and any library works: scikit-learn, PyTorch, JAX, statsmodels. Openlab calls `fit` and
`predict` and nothing else, so it does not train models or wrap a framework. What it owns is everything around the
model: what the model may see, which rows it trains on, how often it ran, and how it is scored.

```python
import polars as pl
import torch

class Model:                               # Openlab calls fit at each refit and predict at each decision
    def fit(self, X, y):
        x = torch.tensor(X.drop("_t", "_key").to_numpy(), dtype=torch.float32)
        ...                                # train as you would anywhere
        return self

    def predict(self, X):                  # a gaussian prediction: a frame with mu and sigma
        ...
        return pl.DataFrame({"mu": mu, "sigma": sigma})
```

## Prediction kinds and metrics

`evaluate.prediction` declares what `predict` returns, and the run refuses a metric that cannot score it before any
trial is spent:

| Prediction | `predict` returns | Metrics |
|---|---|---|
| `point` | a 1-D array | `mae`, `rmse`, `r2`, `mape`, `crps`; `accuracy`, `balanced_accuracy`, `f1`, `macro_f1`, `precision`, `recall` for classes |
| `prob` | a 1-D array of probabilities | `auc`, `brier`, `logloss`, `hit_rate`, `f1`, `precision`, `recall` |
| `gaussian` | a frame with `mu`, `sigma` | `crps`, `coverage90`, `width90`, `mae`, `rmse` |
| `quantiles` | a frame with `q0.1`, `q0.5`, ... | `pinball`, `crps`, `coverage90`, `width90`, `mae`, `rmse` |
| `ensemble` | a frame with `m0`, `m1`, ... | `crps`, `coverage90`, `width90`, `mae`, `rmse` |
| `interval` | a frame with `lower`, `upper` (one central interval per row) | `coverage90`, `width90`, `coverage`, `interval_width` |

`coverage90` and `width90` read the central 90% interval (the 5% and 95% quantiles of a quantile, gaussian or
ensemble prediction). `coverage` and `interval_width` read it at the level the study declares, `evaluate.level`
(e.g. 0.8). A `lower`/`upper` frame does not say its level, so it needs `evaluate.level`, and a fixed-level metric
is refused when the two differ. Width is in the target's units (shorter is better).

Every metric has a direction: higher is better, lower is better, or target-valued. Coverage is target-valued: it is
best at its nominal level (0.9 for `coverage90`, `evaluate.level` for `coverage`), so the paired difference against
a baseline names as `better` the one closer to that level, and the result records it as `nominal`. A package
registers a target-valued metric with `Metric(..., higher_is_better=None, nominal=0.5)`, and a share that papers may
print in percent with `unit="fraction"`.

Baselines (`persistence`, `climatology`, `majority`, `mean`, or another class of model.py as `model:Name`) are scored
on the same rows, with block-bootstrap intervals and paired differences. Metrics are versioned, and each run records
the versions it used. A new metric is registered with `openlab.study.metrics.register(Metric(...))`.

## Seeds, caching and the registry

The split's `seed` is part of the run key, along with the hashes of model.py (and every sibling module it imports),
the data, the spec and the packages the run executed. The same key returns the cached result. `study run --seed N`
gives the model another seed for one run (a paper that averages over seeds: one run per seed, each a counted trial);
the folds and the holdout keep `split.seed`, unless `split.reseed = true` lets the run's seed deal the folds too. Every run registers its final model
in the study's MLflow registry (`.openlab/mlflow`), tagged with the run that made it. `study forecast` predicts with a
registered model, unpickled only inside the sandbox.

## The search tree

Runs form a tree. `study run --op OP --parent RUN` says how a run was made:

| `--op` | Parent | Meaning |
|---|---|---|
| `draft` | none | a new line of runs |
| `improve` | one | a change meant to do better than its parent |
| `debug` | one | a fix of a parent that failed or misbehaved |
| `fuse` | one, plus `--ref RUN` (repeatable) | the parent, borrowing from runs of other branches |

The parent may be named by its run or its result id. Openlab checks the op against its links before anything runs:
a draft with a parent, an improve without one, refs on an improve and a holdout run as a parent are each refused.
Without `--op`, a run follows the branch's last run, as before. `study runs` shows each run's op and parent. Every
node of the tree is a counted trial; the holdout is never part of it.

## Equations that must match the code

A paper's methods equation is bound to the function that implements it:

```latex
\labeq{mu}{model.mu}{\mu(x) = 0.5 + 2x}
```

Openlab reads the equation, runs your function in the sandbox on probe inputs, and the paper build fails if the two
disagree. In a replication the scaffold puts the paper's equations in `paper/equations.tex`, so a rebuilt method is
checked against the paper's own definition.

The pilots used this path. The Rasp and Lerch (2018) replication rebuilt EMOS with scipy and the embedding network with
CPU PyTorch inside `model.py`; the weather EMOS study fits per-station Gaussian EMOS on a rolling 60-day window.
