"""The study's model. openlab calls features and label with point-in-time views, and Model.fit / Model.predict."""

import polars as pl
from sklearn.linear_model import LinearRegression

COLUMNS = ["x", "y_now"]


def features(view, t):
    """What is known at t, one row per site."""
    return (view.frame("panel").sort("date").group_by("site")
            .agg(pl.col("x").last(), pl.col("y").last().alias("y_now")).sort("site"))


def label(view, t):
    """The outcome at t + horizon, one row per site: this view runs to then, and features never receive it."""
    return view.frame("panel").sort("date").group_by("site").agg(pl.col("y").last()).sort("site")


class Model:
    def fit(self, X, y):
        self.m = LinearRegression().fit(X.select(COLUMNS).to_numpy(), y)
        return self

    def predict(self, X):
        return self.m.predict(X.select(COLUMNS).to_numpy())
