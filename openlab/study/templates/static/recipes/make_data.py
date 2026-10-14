"""Make data/table.parquet: 1,000 rows, five numeric features (two carry signal) and a binary class y."""

from pathlib import Path

import numpy as np
import polars as pl

rng = np.random.default_rng(7)
n = 1000
X = rng.normal(size=(n, 5))
p = 1 / (1 + np.exp(-(1.2 * X[:, 0] - 0.8 * X[:, 1])))
df = pl.DataFrame({f"f{i}": X[:, i] for i in range(5)}).with_columns(y=pl.Series((rng.random(n) < p).astype(int)))

out = Path(__file__).resolve().parent.parent / "data" / "table.parquet"
out.parent.mkdir(exist_ok=True)
df.write_parquet(out)
print(f"wrote {out}")
