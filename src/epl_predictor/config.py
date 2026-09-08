"""Central configuration for the EPL match-outcome predictor."""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
# A GitHub mirror of football-data.co.uk's Premier League files. Each season is
# one CSV named season-YYYY.csv with the two-digit start/end years (e.g. 2324).
DATA_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/datasets/football-datasets/master/"
    "datasets/premier-league/season-{code}.csv"
)
FIRST_SEASON = 1993   # 1993/94 is the earliest season in the mirror
# Detailed match stats (shots, corners, cards...) exist from 2000/01 onward.
FIRST_STATS_SEASON = 2000

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
TARGET = "FTR"                       # H / D / A
CLASSES = ["H", "D", "A"]            # fixed label order used everywhere
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}
INT_TO_CLASS = {i: c for c, i in CLASS_TO_INT.items()}

# The first season used for model training: we need a couple of seasons of
# history for ratings/rolling stats to be meaningful *and* detailed stats.
TRAIN_FROM_SEASON = 2002
# Seasons held out entirely as a final test set (never touched during CV/tuning).
TEST_SEASONS = [2024, 2025]          # 2024/25 and 2025/26
# Number of most recent pre-test seasons used as expanding-window CV folds.
N_CV_FOLDS = 6

RANDOM_STATE = 42

# Elo settings (K / regression / promoted rating grid-searched on 2002-2017,
# i.e. before the CV seasons, by the log loss of an Elo-only logistic model)
ELO_START = 1500.0
ELO_K = 10.0                         # effective K is larger: scaled by margin of victory
ELO_HOME_ADV = 60.0                  # Elo points added to the home side
ELO_SEASON_REGRESSION = 0.15         # regress 15% toward the mean each summer
ELO_PROMOTED_RATING = 1450.0         # newcomers start below average

# "Performance Elo": same machinery but the match 'score' is the share of
# shots on target rather than the result -> a rating of underlying dominance
# (an xG-style signal that is less noisy than goals).
PERF_ELO_K = 12.0

# Pi-rating settings (Constantinou & Fenton, 2013)
PI_LAMBDA = 0.035
PI_GAMMA = 0.70

# Rolling windows
FORM_WINDOWS = (5, 10)
EWMA_HALFLIFE = 6                    # matches
H2H_WINDOW = 6                       # last N meetings

# Canonical team names (football-data.co.uk spelling) and common aliases
TEAM_ALIASES = {
    "manchester united": "Man United", "man utd": "Man United", "man u": "Man United",
    "manchester city": "Man City", "mcfc": "Man City",
    "tottenham hotspur": "Tottenham", "spurs": "Tottenham",
    "nottingham forest": "Nott'm Forest", "notts forest": "Nott'm Forest",
    "forest": "Nott'm Forest", "nottm forest": "Nott'm Forest",
    "wolverhampton wanderers": "Wolves", "wolverhampton": "Wolves",
    "newcastle united": "Newcastle", "west ham united": "West Ham",
    "brighton & hove albion": "Brighton", "brighton and hove albion": "Brighton",
    "leicester city": "Leicester", "leeds united": "Leeds",
    "sheffield united": "Sheffield United", "sheffield utd": "Sheffield United",
    "sheff utd": "Sheffield United", "sheffield wednesday": "Sheffield Weds",
    "afc bournemouth": "Bournemouth", "crystal palace": "Crystal Palace",
    "aston villa": "Aston Villa", "villa": "Aston Villa",
    "west bromwich albion": "West Brom", "west bromwich": "West Brom",
    "wba": "West Brom", "qpr": "QPR", "queens park rangers": "QPR",
    "hull city": "Hull", "stoke city": "Stoke", "swansea city": "Swansea",
    "cardiff city": "Cardiff", "norwich city": "Norwich", "ipswich town": "Ipswich",
    "luton town": "Luton", "burnley fc": "Burnley", "everton fc": "Everton",
    "liverpool fc": "Liverpool", "chelsea fc": "Chelsea", "arsenal fc": "Arsenal",
    "fulham fc": "Fulham", "brentford fc": "Brentford", "southampton fc": "Southampton",
    "sunderland afc": "Sunderland", "middlesbrough fc": "Middlesbrough", "boro": "Middlesbrough",
}
