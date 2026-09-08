"""Model definitions (kept in their own module so pickled models can be
re-loaded from anywhere, e.g. the prediction CLI or a notebook)."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config

# The handful of features that carry most of the signal (used by the compact
# logistic model, which is also a strong, well-calibrated ensemble member).
RATING_FEATURES = [
    "elo_diff", "elo_exp_home", "pelo_diff", "pi_diff_venue", "pi_diff_overall",
    "poisv_exp_gd", "poisv_pH", "poisv_pA", "d_std_ewm", "d_gd_r38", "d_venue_pts_ewm",
    "d_prev_ppg", "h2h_home_gd", "league_home_winrate",
]


class PriorClassifier(BaseEstimator, ClassifierMixin):
    """Predicts the training class frequencies (H/D/A) for every match."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1, 2])
        self.prior_ = np.bincount(y, minlength=3) / len(y)
        return self

    def predict_proba(self, X):
        return np.tile(self.prior_, (len(X), 1))

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


class ProbAverager(BaseEstimator, ClassifierMixin):
    """Weighted average of the member models' predicted probabilities."""

    def __init__(self, members: dict[str, BaseEstimator], weights: dict[str, float] | None = None):
        self.members = members
        self.weights = weights

    def fit(self, X, y, **fit_params):
        self.classes_ = np.array([0, 1, 2])
        self.fitted_ = {name: clone(m).fit(X, y) for name, m in self.members.items()}
        return self

    def predict_proba(self, X):
        w = self.weights or {n: 1.0 for n in self.fitted_}
        total = sum(w.values())
        out = sum(w[n] * self.fitted_[n].predict_proba(X) for n in self.fitted_) / total
        return out

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def make_xgb(params: dict | None = None):
    from xgboost import XGBClassifier
    base = dict(
        objective="multi:softprob", num_class=3, tree_method="hist",
        n_estimators=400, learning_rate=0.03, max_depth=3, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.5, reg_lambda=5.0, reg_alpha=0.5, gamma=0.1,
        n_jobs=2, random_state=config.RANDOM_STATE, eval_metric="mlogloss",
    )
    if params:
        base.update(params)
    return XGBClassifier(**base)


def make_lgbm(params: dict | None = None):
    from lightgbm import LGBMClassifier
    base = dict(
        objective="multiclass", n_estimators=500, learning_rate=0.02, num_leaves=8,
        min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
        reg_lambda=5.0, reg_alpha=0.5, n_jobs=2, random_state=config.RANDOM_STATE, verbose=-1,
    )
    if params:
        base.update(params)
    return LGBMClassifier(**base)


def make_models(xgb_params: dict | None = None) -> dict[str, BaseEstimator]:
    imputer = lambda: SimpleImputer(strategy="median")  # noqa: E731
    return {
        "prior": PriorClassifier(),
        "elo_logit": Pipeline([
            ("select", ColumnSelector(["elo_diff", "elo_exp_home"])),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=2000)),
        ]),
        # compact model on the strongest engineered signals only
        "ratings_logit": Pipeline([
            ("select", ColumnSelector(RATING_FEATURES)),
            ("impute", imputer()),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=0.3, max_iter=5000)),
        ]),
        "logreg": make_pipeline(imputer(), StandardScaler(),
                                LogisticRegression(C=0.02, max_iter=5000)),
        "random_forest": make_pipeline(imputer(), RandomForestClassifier(
            n_estimators=600, min_samples_leaf=25, max_features=0.25, n_jobs=2,
            random_state=config.RANDOM_STATE, class_weight=None)),
        "xgboost": make_xgb(xgb_params),
        "lightgbm": make_lgbm(),
    }


class ColumnSelector(BaseEstimator):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns].to_numpy()


