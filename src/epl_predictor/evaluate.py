"""Evaluation figures (matplotlib) for the trained models.

Run after `train.py`; reads models/eval_artifacts.joblib and writes PNGs to
reports/figures/. Every plot uses the same restrained style: thin marks, a
fixed categorical palette, hairline solid gridlines, one axis per chart.
"""
from __future__ import annotations

import json
import logging
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from . import config  # noqa: E402
from .features import get_features  # noqa: E402
from .metrics import evaluate_probs  # noqa: E402
from .train import load_dataset  # noqa: E402

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUES = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.labelsize": 10, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 9,
    "lines.linewidth": 2, "lines.markersize": 6, "text.color": INK, "figure.dpi": 130,
})

LABELS = {"prior": "Class prior", "elo_logit": "Elo-only logistic", "ratings_logit": "Ratings logistic",
          "logreg": "Logistic regression", "random_forest": "Random forest", "xgboost": "XGBoost",
          "lightgbm": "LightGBM", "ensemble": "Ensemble"}
OUTCOMES = ["Home win", "Draw", "Away win"]


def _save(fig, name: str) -> None:
    path = config.FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path.relative_to(config.PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_model_comparison(cv: pd.DataFrame, holdout: pd.DataFrame) -> None:
    order = ["prior", "elo_logit", "ratings_logit", "logreg", "random_forest", "xgboost", "lightgbm", "ensemble"]
    order = [m for m in order if m in cv.index]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, metric, title in zip(axes, ["accuracy", "log_loss"],
                                 ["Accuracy (higher is better)", "Log loss (lower is better)"]):
        y = np.arange(len(order))
        a, b = cv.loc[order, metric].to_numpy(), holdout.loc[order, metric].to_numpy()
        ax.hlines(y, a, b, color=AXIS, lw=1.2, zorder=1)                       # dumbbell connector
        ax.scatter(a, y, s=55, color=SERIES[0], zorder=3, label="Cross-validation (2018/19–2023/24)")
        ax.scatter(b, y, s=55, color=SERIES[1], zorder=3, label="Hold-out (2024/25–2025/26)")
        ax.set_yticks(y, [LABELS[m] for m in order])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        pad = (hi - lo) * 0.22
        ax.set_xlim(lo - pad, hi + pad)
        if metric == "accuracy":
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        for yi, va, vb in zip(y, a, b):
            left, right = (va, vb) if va < vb else (vb, va)
            ax.text(right + pad * 0.12, yi, f"{vb:.3f}" if metric == "log_loss" else f"{vb:.1%}",
                    va="center", fontsize=8, color=INK2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=2, bbox_to_anchor=(0.01, 0.955))
    fig.suptitle("Model comparison: 3-way outcome (home / draw / away)", x=0.01, y=0.995, ha="left",
                 fontsize=13, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, "model_comparison.png")


def plot_accuracy_by_season(cv_long: pd.DataFrame, holdout_by_season: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    series = [("ensemble", "Ensemble", SERIES[0]), ("xgboost", "XGBoost", SERIES[1]),
              ("elo_logit", "Elo-only", SERIES[2]), ("prior", "Always pick home win", SERIES[3])]
    for name, label, color in series:
        d = cv_long[cv_long["model"] == name].sort_values("season")
        x = [f"{s}/{(s + 1) % 100:02d}" for s in d["season"]]
        ax.plot(x, d["accuracy"], marker="o", color=color, label=label)
        hs = holdout_by_season[holdout_by_season["model"] == name].sort_values("season") if name in set(holdout_by_season["model"]) else None
        if hs is not None and len(hs):
            hx = [f"{s}/{(s + 1) % 100:02d}" for s in hs["season"]]
            ax.plot([x[-1]] + hx, [d["accuracy"].iloc[-1]] + hs["accuracy"].tolist(), marker="o", color=color)
    n_cv = cv_long["season"].nunique()
    ax.axvspan(n_cv - 0.5, n_cv + len(config.TEST_SEASONS) - 0.5, color=GRID, alpha=0.5, lw=0)
    ax.text(n_cv + len(config.TEST_SEASONS) - 0.6, ax.get_ylim()[1], "hold-out", ha="right", va="top", fontsize=9, color=INK2)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy season by season (each season predicted from all earlier ones)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    _save(fig, "accuracy_by_season.png")


def plot_confusion(probs: np.ndarray, y: np.ndarray, name: str) -> None:
    pred = probs.argmax(axis=1)
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(cm, cmap=BLUES, vmin=0, vmax=cm.max() * 1.15)
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            share = cm[i, j] / cm[i].sum()
            ax.text(j, i, f"{cm[i, j]}\n{share:.0%}", ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > cm.max() * 0.55 else INK)
    ax.set_xticks(range(3), OUTCOMES); ax.set_yticks(range(3), OUTCOMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for s in ax.spines.values():
        s.set_visible(False)
    acc = (pred == y).mean()
    ax.set_title(f"{LABELS.get(name, name)}, hold-out seasons\naccuracy {acc:.1%}  (rows = actual, cells = count and row share)", fontsize=10)
    fig.tight_layout()
    _save(fig, f"confusion_matrix_{name}.png")


def plot_calibration(probs: np.ndarray, y: np.ndarray, name: str, n_bins: int = 8) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1)
    edges = np.linspace(0, 1, n_bins + 1)
    for k, (label, color) in enumerate(zip(OUTCOMES, SERIES[:3])):
        p = probs[:, k]
        obs = (y == k).astype(float)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
            if m.sum() >= 15:
                xs.append(p[m].mean()); ys.append(obs[m].mean())
        ax.plot(xs, ys, marker="o", color=color, label=label)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration on the hold-out seasons ({LABELS.get(name, name)})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, f"calibration_{name}.png")


def plot_feature_importance(xgb_model, features: list[str], top: int = 25) -> None:
    booster = xgb_model.get_booster()
    gain = booster.get_score(importance_type="gain")
    imp = pd.Series({features[int(k[1:])] if k.startswith("f") and k[1:].isdigit() else k: v
                     for k, v in gain.items()}).sort_values(ascending=False).head(top)
    imp = imp / imp.sum()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(range(len(imp)), imp.values[::-1], color=SERIES[0], height=0.7)
    ax.set_yticks(range(len(imp)), imp.index[::-1], fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Share of total gain")
    ax.set_title(f"XGBoost: top {top} features by gain")
    fig.tight_layout()
    _save(fig, "feature_importance_xgboost.png")


def plot_elo_history(feats: pd.DataFrame, teams: list[str] | None = None) -> None:
    long = pd.concat([
        feats[["Date", "HomeTeam", "elo_home"]].rename(columns={"HomeTeam": "team", "elo_home": "elo"}),
        feats[["Date", "AwayTeam", "elo_away"]].rename(columns={"AwayTeam": "team", "elo_away": "elo"}),
    ]).sort_values("Date")
    long = long[long["Date"] >= "2010-07-01"]
    if teams is None:
        teams = ["Man City", "Liverpool", "Arsenal", "Chelsea", "Man United", "Tottenham"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ends = []
    for t, color in zip(teams, SERIES):
        d = long[long["team"] == t]
        d = d.set_index("Date")["elo"].resample("30D").mean().dropna()
        ax.plot(d.index, d.values, color=color, label=t, lw=1.6)
        ends.append([d.index[-1], d.values[-1], t, color])
    # direct labels at the line ends, nudged apart so they never overlap
    ends.sort(key=lambda e: e[1])
    for i in range(1, len(ends)):
        if ends[i][1] - ends[i - 1][1] < 9:
            ends[i][1] = ends[i - 1][1] + 9
    for x, y, t, color in ends:
        ax.text(x, y, f"  {t}", color=color, fontsize=8.5, va="center")
    ax.axhline(config.ELO_START, color=AXIS, lw=0.8)
    ax.set_ylabel("Elo rating (pre-match, 30-day mean)")
    ax.set_title("Elo ratings of the 'big six' since 2010")
    ax.legend(ncol=6, loc="lower left")
    ax.set_xlim(long["Date"].min(), long["Date"].max() + pd.Timedelta(days=420))
    fig.tight_layout()
    _save(fig, "elo_history.png")


def plot_home_advantage(feats: pd.DataFrame) -> None:
    d = feats[["Date", "league_home_winrate", "league_drawrate"]].dropna()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(d["Date"], d["league_home_winrate"], color=SERIES[0], label="Home wins", lw=1.6)
    ax.plot(d["Date"], 1 - d["league_home_winrate"] - d["league_drawrate"], color=SERIES[1], label="Away wins", lw=1.6)
    ax.plot(d["Date"], d["league_drawrate"], color=SERIES[2], label="Draws", lw=1.6)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Home advantage over time (share of results, rolling 380 matches)")
    ax.legend(ncol=3, loc="upper right")
    ax.annotate("2020/21: empty stadiums", xy=(pd.Timestamp("2021-01-15"), 0.385), xytext=(pd.Timestamp("2014-06-01"), 0.34),
                fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    fig.tight_layout()
    _save(fig, "home_advantage.png")


def plot_probability_histogram(probs: np.ndarray, y: np.ndarray, name: str) -> None:
    """How confident is the model, and does confidence translate to accuracy?"""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    edges = np.array([0.33, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1.0])
    xs, acc, n = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() >= 10:
            xs.append(f"{lo:.2f}–{hi:.2f}"); acc.append((pred[m] == y[m]).mean()); n.append(m.sum())
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(xs, acc, color=SERIES[0], width=0.6)
    for b, k in zip(bars, n):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"n={k}", ha="center", fontsize=8, color=INK2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Model confidence (probability of the predicted outcome)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy rises with confidence ({LABELS.get(name, name)}, hold-out)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, f"accuracy_by_confidence_{name}.png")


# ---------------------------------------------------------------------------
def run() -> None:
    art = joblib.load(config.MODELS_DIR / "eval_artifacts.joblib")
    metrics = json.loads((config.MODELS_DIR / "metrics.json").read_text())
    cv = pd.DataFrame(metrics["cv_summary"])
    holdout = pd.DataFrame(metrics["holdout"])
    cv_long = pd.read_csv(config.REPORTS_DIR / "cv_results.csv")

    ds = load_dataset()
    y_test = ds.y[art["test_idx"]]
    feats = get_features()

    # per-season hold-out metrics for every model
    test_seasons = ds.df.iloc[art["test_idx"]]["Season"].to_numpy()
    rows = []
    for name, p in art["test_probs"].items():
        for s in config.TEST_SEASONS:
            m = test_seasons == s
            r = evaluate_probs(p[m], y_test[m]); r.update(model=name, season=int(s))
            rows.append(r)
    holdout_by_season = pd.DataFrame(rows)
    holdout_by_season.to_csv(config.REPORTS_DIR / "holdout_by_season.csv", index=False)

    plot_model_comparison(cv, holdout)
    plot_accuracy_by_season(cv_long, holdout_by_season)
    for name in ("ensemble", "xgboost"):
        plot_confusion(art["test_probs"][name], y_test, name)
        plot_calibration(art["test_probs"][name], y_test, name)
    for name in ("ensemble", "xgboost"):
        plot_probability_histogram(art["test_probs"][name], y_test, name)
    plot_feature_importance(art["fitted"]["xgboost"], art["features"])
    plot_elo_history(feats)
    plot_home_advantage(feats)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    run()
