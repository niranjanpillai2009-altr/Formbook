"""Generate notebooks/walkthrough.ipynb (run once; the notebook is then executed)."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))   # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))     # noqa: E731

md("""
# Premier League match-outcome predictor — walkthrough

This notebook walks through the whole project end to end: **data → features → ratings → models → predictions**.
It uses the package in `src/epl_predictor`, so every step here is the same code the CLI runs.

The task: before kick-off, predict whether the **home team wins, it's a draw, or the away team wins**,
using nothing but what happened in earlier matches.
""")

code("""
import sys, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

from epl_predictor import config
from epl_predictor.data import get_matches
from epl_predictor.features import get_features, feature_columns
from epl_predictor.evaluate import SERIES, INK2   # shared chart style (applied on import)

pd.set_option("display.width", 140, "display.max_columns", 30)
""")

md("""
## 1. Data

33 seasons (1993/94 → 2025/26) of Premier League results from football-data.co.uk, via a GitHub mirror.
Match statistics (shots, shots on target, corners, fouls, cards) are available from **2000/01** onward.
""")

code("""
matches = get_matches()
print(f"{len(matches):,} matches, {matches.Season.nunique()} seasons, {matches.Date.min().date()} → {matches.Date.max().date()}")
matches.tail(3)
""")

code("""
share = matches.groupby("Season")["FTR"].value_counts(normalize=True).unstack()[["H", "D", "A"]]
fig, ax = plt.subplots(figsize=(10, 3.8))
for col, label, c in zip(["H", "D", "A"], ["Home win", "Draw", "Away win"], SERIES[:3]):
    ax.plot(share.index, share[col], color=c, label=label)
ax.set_title("Result shares by season — the home-advantage baseline the model has to beat")
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.legend(ncol=3); plt.show()
print(f"Overall: home {share['H'].mean():.1%}  draw {share['D'].mean():.1%}  away {share['A'].mean():.1%}")
""")

md("""
Home teams win ~45% of matches, so "always predict a home win" is already 45% accurate.
That is the floor. Bookmakers, who use far more information than we have, land around 55%.

## 2. Features (all computed *before* kick-off)

`features.py` builds ~280 pre-match features per fixture. Everything is derived from earlier matches only:

| family | examples |
|---|---|
| **Ratings** | Elo (margin-of-victory, season regression), *performance* Elo driven by shots-on-target share, pi-ratings (home/away specific) |
| **Form** | rolling 5/10-match and exponentially-weighted goals, shots, shots on target, corners, points; shot-share ratios (TSR/STR) as possession proxies; momentum vs. 38-match baseline |
| **Venue form** | home team's last 5 *home* games, away team's last 5 *away* games, team-specific home advantage |
| **Season context** | points per game, goal difference, league position and gap to top, previous-season finish, promoted flag |
| **Match-up** | Poisson expected goals (attack × opponent defence × league rate) → P(H/D/A); attack-vs-defence differentials |
| **Head-to-head** | record over the last 6 meetings, and at this venue |
| **Schedule** | rest days, fixtures in the last 14 days, unbeaten/winless streaks |
| **League environment** | rolling home-win rate, draw rate, goals per game (captures e.g. the empty-stadium 2020/21 season) |
""")

code("""
feats = get_features()
cols = feature_columns(feats)
print(f"{len(cols)} features for {len(feats):,} matches")
example = feats[feats.Season == 2025].iloc[-1]
print(f"Example: {example.HomeTeam} v {example.AwayTeam} on {example.Date.date()}  (result {example.FTHG:.0f}-{example.FTAG:.0f})")
example[["elo_home", "elo_away", "elo_exp_home", "pelo_diff", "pi_exp_gd", "poisv_exp_hg", "poisv_exp_ag",
         "h_pts_r5", "a_pts_r5", "h_league_pos", "a_league_pos", "h2h_home_gd", "h_rest_days", "a_rest_days"]].round(3)
""")

code("""
train_rows = feats[feats.Season >= config.TRAIN_FROM_SEASON]
gd = train_rows.FTHG - train_rows.FTAG
corr = train_rows[cols].corrwith(gd).sort_values(key=abs, ascending=False).head(20)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(corr.index[::-1], corr.values[::-1], color=[SERIES[0] if v > 0 else SERIES[7] for v in corr.values[::-1]], height=0.7)
ax.set_title("Features most correlated with the final goal difference (home − away)")
ax.grid(axis="y", visible=False); plt.show()
""")

md("""
No single feature gets above |r| ≈ 0.45 — football is noisy, and that is what caps the achievable accuracy.
Notice that the *shot-based* performance Elo (`pelo_diff`) is as informative as result-based Elo.

### Elo ratings over time
""")

code("display(Image('../reports/figures/elo_history.png'))")
code("display(Image('../reports/figures/home_advantage.png'))")

md("""
## 3. Models and evaluation protocol

* **Training rows:** seasons 2002/03 onward (two seasons of shot data warm the rolling stats up).
* **Cross-validation:** expanding window over the six seasons 2018/19 – 2023/24: train on every earlier season, predict that one.
  This is exactly how the model would be used in practice, and it is the only honest way to validate time-series data.
* **Hold-out:** 2024/25 and 2025/26 were never touched during tuning or model selection.
* **Models:** class-prior and Elo-only baselines, a compact "ratings" logistic regression, full logistic regression,
  random forest, **XGBoost (Optuna-tuned)**, LightGBM, and a weighted probability-averaging **ensemble**.
* **Metrics:** accuracy (what you asked for), plus log loss, Brier score and the ranked probability score (RPS) —
  the standard scoring rule for football forecasts, which rewards probabilities that are *close* to the truth.
""")

code("""
metrics = json.loads((Path.cwd().parent / "models" / "metrics.json").read_text())
cv = pd.DataFrame(metrics["cv_summary"]).sort_values("log_loss")
holdout = pd.DataFrame(metrics["holdout"]).sort_values("log_loss")
print("Cross-validation (mean over six seasons)"); display(cv.round(4))
print("Hold-out 2024/25 + 2025/26 (760 matches)"); display(holdout.round(4))
print("Deployed model:", metrics["deployed"])
""")

code("display(Image('../reports/figures/model_comparison.png'))")
code("display(Image('../reports/figures/accuracy_by_season.png'))")

md("""
### What the model gets right and wrong

The confusion matrix shows the classic football-prediction pattern: draws are almost never the *most likely* outcome
for any single match (they hover around 25–30%), so an accuracy-maximising classifier rarely picks them —
even though it assigns them sensible probabilities (see the calibration plot).
""")

code("display(Image('../reports/figures/confusion_matrix_xgboost.png'))")
code("display(Image('../reports/figures/calibration_xgboost.png'))")
code("display(Image('../reports/figures/accuracy_by_confidence_xgboost.png'))")

md("""
When the model is confident it is right much more often — the useful signal is in the probabilities, not just the label.

### Which features matter?
""")

code("display(Image('../reports/figures/feature_importance_xgboost.png'))")

md("""
## 4. A single cross-validation fold by hand

To see the machinery without the wrapper: train XGBoost on everything before 2023/24 and score that season.
""")

code("""
from epl_predictor.train import load_dataset
from epl_predictor.models import make_xgb
from epl_predictor.metrics import evaluate_probs

ds = load_dataset()
season, tr, va = ds.folds[-1]
xgb_params = json.loads((Path.cwd().parent / "models" / "xgb_params.json").read_text())
model = make_xgb(xgb_params).fit(ds.X.iloc[tr], ds.y[tr])
probs = model.predict_proba(ds.X.iloc[va])
print(f"Season {season}/{(season+1)%100:02d}: trained on {len(tr):,} matches ->", {k: round(v, 4) for k, v in evaluate_probs(probs, ds.y[va]).items()})
""")

md("""
## 5. Predict a fixture

`predict.py` appends the fixture to the match table with an unknown result, re-runs the feature pipeline
(so every rating and rolling stat is exactly what it would be on the eve of the match), and scores it with the deployed model.
""")

code("""
from epl_predictor.predict import predict_fixtures, load_bundle, format_report

bundle = load_bundle()
fixtures = pd.DataFrame({
    "HomeTeam": ["Arsenal", "Man City", "Spurs", "Burnley"],
    "AwayTeam": ["Chelsea", "Liverpool", "Manchester United", "Everton"],
    "Date": ["2026-09-13"] * 4,
})
pred = predict_fixtures(fixtures, bundle, matches)
print(format_report(pred, bundle))
pred.round(3)
""")

md("""
## 6. Where to take it next

* **Better inputs beat better models here.** Possession, xG and lineup data (FBref / Understat / FPL) would add real signal;
  bookmaker odds even more so. The feature pipeline takes any extra per-match column.
* **Dixon–Coles** with time decay as a feature, or a Bayesian rating model.
* **Betting-style evaluation:** compare the probabilities to the closing odds and simulate a value-betting strategy.
* **Live season tracking:** once the 2026/27 file appears in the data mirror, `python run_pipeline.py --quick --refresh-data` retrains on it.
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
Path(__file__).with_name("walkthrough.ipynb").write_text(nbf.writes(nb))
print("wrote walkthrough.ipynb")
