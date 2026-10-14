"""The study's model. openlab calls Model.fit on each fold's training rows and Model.predict on the rest."""

from sklearn.linear_model import LogisticRegression


def _numeric(X):
    return X.select([c for c in X.columns if not c.startswith("_")]).to_numpy()     # openlab adds _key


class Model:
    def fit(self, X, y):
        self.m = LogisticRegression(max_iter=1000).fit(_numeric(X), y)
        return self

    def predict(self, X):
        return self.m.predict_proba(_numeric(X))[:, 1]
