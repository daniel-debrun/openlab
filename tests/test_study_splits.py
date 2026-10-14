"""Guarantee: no label crosses a split edge. A decision is in a segment only if its label is known by the
segment end; a fit at s trains only on decisions whose labels were known by s; a group carve keeps groups whole."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from openlab.study import spec, splits
from openlab.study.errors import StudyError


def make(folder, mode, target, split):
    (folder / "study.toml").write_text(f"""
[study]
name = "t"
question = "q?"
mode = "{mode}"

[target]
{target}

[split]
{split}

[evaluate]
metrics = ["mae"]

[budget]
trials = 3
""", encoding="utf-8")
    return spec.load(folder)


def test_no_research_decision_earns_a_holdout_month(tmp_path):
    # going_concern: monthly decisions with a one-month label, research ending on a month-end
    s = make(tmp_path, "temporal", 'horizon = "1M"', 'scheme = "walk_forward"\ndecisions = "monthly"\nrefit = "1M"\n'
             "research = [2015-01-01, 2021-12-31]\nholdout = [2022-01-01, 2023-12-31]")
    h, times = splits.horizon(s), splits.decision_times(s, {})
    research, holdout = (splits.temporal_steps(s, times, seg) for seg in ("research", "holdout"))
    assert research[-1].t == datetime(2021, 11, 30)              # its label is December's, known on 2021-12-31
    assert all(st.t + h <= datetime(2021, 12, 31) for st in research)
    assert datetime(2021, 12, 31) in times and holdout[0].t == datetime(2022, 1, 31)
    assert all(st.refit for st in research)


def test_walk_forward_trains_only_on_labels_known_by_the_fit(tmp_path):
    s = make(tmp_path, "temporal", 'horizon = "3d"', 'scheme = "walk_forward"\ndecisions = "daily"\nrefit = "7d"\n'
             'window = "10d"\nresearch = [2024-01-01, 2024-03-31]')
    h, times = splits.horizon(s), splits.decision_times(s, {})
    steps = splits.temporal_steps(s, times, "research")
    fits = [st.t for st in steps if st.refit]
    assert all(b - a == timedelta(days=7) for a, b in zip(fits, fits[1:]))
    for st in steps:
        assert st.train_until == max(f for f in fits if f <= st.t)
        train = [t for t in times if st.trains_on(t, h)]
        assert all(t + h <= st.train_until for t in train)
        if train:                                                 # the window trims, the embargo is not overdone
            assert max(train) == st.train_until - timedelta(days=3)
            assert min(train) == max(times[0], st.train_until - timedelta(days=9))


def test_a_group_carve_keeps_groups_whole(tmp_path):
    s = make(tmp_path, "static", 'column = "y"', 'scheme = "group"\nby = "g"\nk = 3\nfraction = 0.2\nseed = 1')
    df = pl.DataFrame({"g": [f"g{i // 5}" for i in range(100)], "y": [i % 2 for i in range(100)]})
    research, holdout = splits.carve(df, s)
    g = df.get_column("g")
    assert set(g[research]).isdisjoint(g[holdout]) and len(research) + len(holdout) == 100
    assert g[holdout].n_unique() == 4
    for train, val in splits.static_folds(df[research], s):
        assert set(df[research]["g"][train]).isdisjoint(df[research]["g"][val])


def test_a_fraction_that_is_not_one_fold_is_refused(tmp_path):
    s = make(tmp_path, "static", 'column = "y"', 'scheme = "kfold"\nfraction = 0.3')
    with pytest.raises(StudyError):
        splits.carve(pl.DataFrame({"y": range(10)}), s)
