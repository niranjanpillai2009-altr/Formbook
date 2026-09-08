"""Model training, tuning and selection with season-based time-series CV.

Protocol
--------
* Rows: seasons TRAIN_FROM_SEASON .. last season, one per match.
* Final hold-out: TEST_SEASONS (never used for tuning or model selection).
* Model selection / tuning: expanding-window CV over the last N_CV_FOLDS
  seasons before the hold-out (train on every earlier season, validate on
  that season). This mimics real use: predict a season from its past.
* Models: naive baselines, Elo-only logistic, logistic regression, random
  forest, XGBoost (Optuna-tuned), LightGBM, and a probability-averaging
  ensemble. All produce H/D/A probabilities; the predicted class is argmax.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from . import config
from .features import feature_columns, get_features
from .metrics import evaluate_probs
from .models import ProbAverager, make_models

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

ENSEMBLE_MEMBERS = ["ratings_logit", "logreg", "random_forest", "xgboost", "lightgbm"]


# ---------------------------------------------------------------------------
# Data split helpers
# ---------------------------------------------------------------------------
@dataclass
class Dataset:
    df: pd.DataFrame
    features: list[str]
    dev_idx: np.ndarray
    test_idx: np.ndarray
    folds: list[tuple[int, np.ndarray, np.ndarray]] = field(default_factory=list)

    @property
    def X(self) -> pd.DataFrame:
        return self.df[self.features]

    @property
    def y(self) -> np.ndarray:
        return self.df["target"].to_numpy()


def load_dataset(refresh: bool = False, refresh_data: bool = False,
                 features: list[str] | None = None) -> Dataset:
    df = get_features(refresh=refresh, refresh_data=refresh_data)
    df = df[df["Season"] >= config.TRAIN_FROM_SEASON].reset_index(drop=True)
    feats = features or feature_columns(df)
    seasons = np.sort(df["Season"].unique())
    test_mask = df["Season"].isin(config.TEST_SEASONS)
    dev_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]

    dev_seasons = [s for s in seasons if s not in config.TEST_SEASONS]
    cv_seasons = dev_seasons[-config.N_CV_FOLDS:]
    folds = []
    for s in cv_seasons:
        tr = np.where((df["Season"] < s) & ~test_mask)[0]
        va = np.where(df["Season"] == s)[0]
        folds.append((int(s), tr, va))
    return Dataset(df=df, features=feats, dev_idx=dev_idx, test_idx=test_idx, folds=folds)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
def cross_validate(model, ds: Dataset, name: str = "") -> tuple[pd.DataFrame, np.ndarray]:
    """Expanding-window CV. Returns per-fold metrics and out-of-fold probs
    (aligned with ds.df rows; NaN for rows outside the CV seasons)."""
    oof = np.full((len(ds.df), 3), np.nan)
    rows = []
    X, y = ds.X, ds.y
    for season, tr, va in ds.folds:
        t0 = time.time()
        m = clone(model).fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[va])
        oof[va] = p
        r = evaluate_probs(p, y[va])
        r.update(model=name, season=season, n_train=len(tr), n_val=len(va), secs=time.time() - t0)
        rows.append(r)
        log.info("  %-14s %s  acc=%.3f  logloss=%.4f  rps=%.4f  (%.1fs)",
                 name, season, r["accuracy"], r["log_loss"], r["rps"], r["secs"])
    return pd.DataFrame(rows), oof


def tune_xgb(ds: Dataset, n_trials: int = 40, timeout: int | None = None) -> dict:
    """Optuna search over XGBoost hyper-parameters, scored by mean CV log loss."""
    import optuna
    from xgboost import XGBClassifier
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X, y = ds.X, ds.y

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 6),
            min_child_weight=trial.suggest_float("min_child_weight", 1, 40, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 0.9),
            reg_lambda=trial.suggest_float("reg_lambda", 0.5, 30, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5, log=True),
            gamma=trial.suggest_float("gamma", 0.0, 1.0),
        )
        losses, iters = [], []
        for season, tr, va in ds.folds:
            m = XGBClassifier(objective="multi:softprob", num_class=3, tree_method="hist",
                              n_estimators=1500, early_stopping_rounds=60, n_jobs=2,
                              random_state=config.RANDOM_STATE, eval_metric="mlogloss", **params)
            m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
            p = m.predict_proba(X.iloc[va])
            losses.append(evaluate_probs(p, y[va])["log_loss"])
            iters.append(m.best_iteration + 1)
        trial.set_user_attr("n_estimators", int(np.median(iters)))
        return float(np.mean(losses))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    best = dict(study.best_params)
    best["n_estimators"] = study.best_trial.user_attrs["n_estimators"]
    log.info("best XGB CV log loss %.4f with %s", study.best_value, best)
    return best


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def run(n_trials: int = 40, refresh: bool = False, refresh_data: bool = False,
        skip_tuning: bool = False) -> dict:
    ds = load_dataset(refresh=refresh, refresh_data=refresh_data)
    log.info("dataset: %d matches, %d features, dev=%d test=%d, CV seasons=%s",
             len(ds.df), len(ds.features), len(ds.dev_idx), len(ds.test_idx),
             [f[0] for f in ds.folds])

    # 1. tune XGBoost on the CV folds
    xgb_params_path = config.MODELS_DIR / "xgb_params.json"
    if skip_tuning and xgb_params_path.exists():
        xgb_params = json.loads(xgb_params_path.read_text())
    elif skip_tuning:
        xgb_params = None
    else:
        log.info("tuning XGBoost with Optuna (%d trials)...", n_trials)
        xgb_params = tune_xgb(ds, n_trials=n_trials)
        xgb_params_path.write_text(json.dumps(xgb_params, indent=2))

    # 2. CV every candidate model
    models = make_models(xgb_params)
    cv_rows, oof = [], {}
    for name, model in models.items():
        log.info("cross-validating %s", name)
        r, p = cross_validate(model, ds, name)
        cv_rows.append(r)
        oof[name] = p

    # 3. ensemble = weighted average of the learners' probabilities. The simple
    #    Elo-only model is included on purpose: it is well calibrated and its
    #    errors are less correlated with the tree models', which lowers log loss.
    strong = ENSEMBLE_MEMBERS
    cv_df = pd.concat(cv_rows, ignore_index=True)
    mean_ll = cv_df.groupby("model")["log_loss"].mean()
    weights = {n: float(1.0 / (mean_ll[n] - 0.9)) for n in strong}   # sharper than plain 1/ll
    wsum = sum(weights.values())
    weights = {n: w / wsum for n, w in weights.items()}
    log.info("ensemble weights: %s", {k: round(v, 3) for k, v in weights.items()})

    ens_oof = sum(weights[n] * oof[n] for n in strong)
    ens_rows = []
    for season, tr, va in ds.folds:
        r = evaluate_probs(ens_oof[va], ds.y[va])
        r.update(model="ensemble", season=season, n_train=len(tr), n_val=len(va), secs=0.0)
        ens_rows.append(r)
    cv_df = pd.concat([cv_df, pd.DataFrame(ens_rows)], ignore_index=True)
    oof["ensemble"] = ens_oof
    cv_df.to_csv(config.REPORTS_DIR / "cv_results.csv", index=False)

    summary = (cv_df.groupby("model")[["accuracy", "log_loss", "brier", "rps"]]
               .mean().sort_values("log_loss"))
    log.info("\nCV summary (mean over %d seasons):\n%s", len(ds.folds), summary.round(4).to_string())

    # 4. fit on all dev seasons, evaluate once on the untouched hold-out
    log.info("fitting final models on %d dev matches, scoring %d hold-out matches",
             len(ds.dev_idx), len(ds.test_idx))
    X, y = ds.X, ds.y
    test_rows, test_probs, fitted = [], {}, {}
    for name, model in models.items():
        m = clone(model).fit(X.iloc[ds.dev_idx], y[ds.dev_idx])
        fitted[name] = m
        p = m.predict_proba(X.iloc[ds.test_idx])
        test_probs[name] = p
        r = evaluate_probs(p, y[ds.test_idx]); r["model"] = name
        test_rows.append(r)
    ens = ProbAverager({n: models[n] for n in strong}, weights)
    ens.fitted_ = {n: fitted[n] for n in strong}
    ens.classes_ = np.array([0, 1, 2])
    p = ens.predict_proba(X.iloc[ds.test_idx])
    test_probs["ensemble"] = p
    r = evaluate_probs(p, y[ds.test_idx]); r["model"] = "ensemble"
    test_rows.append(r)
    test_df = pd.DataFrame(test_rows).set_index("model").sort_values("log_loss")
    test_df.to_csv(config.REPORTS_DIR / "holdout_results.csv")
    log.info("\nHold-out (%s):\n%s", [f"{s}/{(s+1)%100:02d}" for s in config.TEST_SEASONS],
             test_df.round(4).to_string())

    # 5. per-season hold-out breakdown for the ensemble & xgboost
    per_season = []
    for name in ("xgboost", "ensemble"):
        for s in config.TEST_SEASONS:
            m = (ds.df.iloc[ds.test_idx]["Season"] == s).to_numpy()
            r = evaluate_probs(test_probs[name][m], y[ds.test_idx][m]); r.update(model=name, season=s)
            per_season.append(r)
    per_season = pd.DataFrame(per_season)
    log.info("\nHold-out by season:\n%s", per_season.round(4).to_string())

    # 6. choose the deployed model by *CV* log loss (the hold-out is only ever
    #    reported, never used to pick), then refit it on all data (dev + test)
    #    so the deployed model is as current as possible.
    candidates = strong + ["ensemble"]
    best_name = summary.loc[candidates, "log_loss"].idxmin()
    log.info("deploying '%s' (best CV log loss; refit on all %d matches)", best_name, len(ds.df))
    if best_name == "ensemble":
        final = ProbAverager({n: models[n] for n in strong}, weights).fit(X, y)
    else:
        final = clone(models[best_name]).fit(X, y)

    bundle = {
        "model": final, "model_name": best_name, "features": ds.features,
        "classes": config.CLASSES, "weights": weights, "xgb_params": xgb_params,
        "trained_through": str(ds.df["Date"].max().date()),
        "cv_summary": summary.to_dict(), "holdout": test_df.to_dict(),
    }
    joblib.dump(bundle, config.MODELS_DIR / "final_model.joblib")
    # keep what the evaluation plots need (probabilities + the dev-fitted XGBoost)
    joblib.dump({"fitted": {"xgboost": fitted["xgboost"]}, "test_probs": test_probs, "oof": oof,
                 "test_idx": ds.test_idx, "features": ds.features, "weights": weights},
                config.MODELS_DIR / "eval_artifacts.joblib")
    (config.MODELS_DIR / "metrics.json").write_text(json.dumps({
        "cv_summary": summary.round(4).to_dict(), "holdout": test_df.round(4).to_dict(),
        "holdout_by_season": per_season.round(4).to_dict(orient="records"),
        "deployed": best_name, "n_features": len(ds.features),
    }, indent=2))
    return bundle


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Train the EPL match-outcome predictor")
    ap.add_argument("--trials", type=int, default=40, help="Optuna trials for XGBoost")
    ap.add_argument("--refresh", action="store_true", help="rebuild features from the stored matches")
    ap.add_argument("--refresh-data", action="store_true", help="re-download the latest seasons, then rebuild")
    ap.add_argument("--skip-tuning", action="store_true", help="reuse models/xgb_params.json")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    run(n_trials=args.trials, refresh=args.refresh, refresh_data=args.refresh_data,
        skip_tuning=args.skip_tuning)


if __name__ == "__main__":
    main()
