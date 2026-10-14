"""Make data/panel.parquet: four sites, a daily signal x, and an outcome y that follows x one day later."""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

rng = np.random.default_rng(7)
days = [date(2024, 1, 1) + timedelta(days=i) for i in range(548)]           # 2024-01-01 .. 2025-07-01
rows = []
for site in ("a", "b", "c", "d"):
    x = rng.normal(size=len(days)).cumsum() * 0.1 + rng.normal(size=len(days))
    y = np.r_[0.0, 0.6 * x[:-1]] + rng.normal(scale=0.5, size=len(days))       # y today follows x yesterday
    rows.append(pl.DataFrame({"date": days, "site": site, "x": x, "y": y}))

out = Path(__file__).resolve().parent.parent / "data" / "panel.parquet"
out.parent.mkdir(exist_ok=True)
pl.concat(rows).write_parquet(out)
print(f"wrote {out}")
