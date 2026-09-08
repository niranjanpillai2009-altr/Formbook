"""Predict the outcome of upcoming Premier League fixtures.

Usage
-----
    python -m epl_predictor.predict "Arsenal" "Chelsea"
    python -m epl_predictor.predict "Man City" "Liverpool" --date 2026-09-20
    python -m epl_predictor.predict --fixtures fixtures.csv      # HomeTeam,AwayTeam[,Date]
    python -m epl_predictor.predict --teams                        # list known team names

How it works: the fixture is appended to the historical match table with an
unknown result, the full feature pipeline is re-run (so every rolling stat,
rating, head-to-head record and table position is exactly what it would be on
the eve of the match), and the trained model scores the resulting row.
"""
from __future__ import annotations

import argparse
import datetime as dt
import difflib
import logging
import sys

import joblib
import numpy as np
import pandas as pd

from . import config
from .data import get_matches
from .features import next_match_features
from . import models as _models  # noqa: F401  (registers classes for unpickling)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
def known_teams(matches: pd.DataFrame) -> list[str]:
    return sorted(set(matches["HomeTeam"]) | set(matches["AwayTeam"]))


def resolve_team(name: str, teams: list[str], allow_new: bool = False) -> str:
    """Map user input ('spurs', 'Manchester United') to the canonical name."""
    key = name.strip().lower()
    if key in config.TEAM_ALIASES:
        return config.TEAM_ALIASES[key]
    for t in teams:
        if t.lower() == key:
            return t
    close = difflib.get_close_matches(name.strip(), teams, n=3, cutoff=0.6)
    if close and close[0].lower().startswith(key[:4]):
        return close[0]
    if allow_new:
        log.warning("'%s' is not in the data; treating it as a newly promoted team", name)
        return name.strip()
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise ValueError(f"Unknown team '{name}'.{hint} Use --teams to list valid names.")


def _season_of(date: pd.Timestamp) -> int:
    return date.year if date.month >= 7 else date.year - 1


def load_bundle() -> dict:
    """The deployed model plus its metadata.

    Prefers models/final_model.joblib (written by train.py). Falls back to
    models/final_model.json + model_meta.json — XGBoost's portable format,
    which is what the self-refreshing web page carries.
    """
    path = config.MODELS_DIR / "final_model.joblib"
    if path.exists():
        return joblib.load(path)
    js, meta = config.MODELS_DIR / "final_model.json", config.MODELS_DIR / "model_meta.json"
    if js.exists() and meta.exists():
        import json
        from xgboost import XGBClassifier
        model = XGBClassifier()
        model.load_model(js)
        bundle = json.loads(meta.read_text())
        bundle["model"] = model
        return bundle
    raise FileNotFoundError("No trained model found. Run `python -m epl_predictor.train` first.")


def predict_fixtures(fixtures: pd.DataFrame, bundle: dict | None = None,
                     matches: pd.DataFrame | None = None, allow_new: bool = False,
                     season_teams: list[str] | None = None) -> pd.DataFrame:
    """`fixtures` needs HomeTeam and AwayTeam columns; Date is optional
    (defaults to today). Returns probabilities and the predicted outcome.
    `season_teams` (the 20 clubs of the season being predicted) makes the
    start-of-season Elo reset exact; it defaults to the teams in `fixtures`."""
    bundle = bundle or load_bundle()
    matches = matches if matches is not None else get_matches()
    teams = known_teams(matches)

    fx = fixtures.copy()
    fx["HomeTeam"] = [resolve_team(t, teams, allow_new) for t in fx["HomeTeam"]]
    fx["AwayTeam"] = [resolve_team(t, teams, allow_new) for t in fx["AwayTeam"]]
    if "Date" not in fx or fx["Date"].isna().all():
        fx["Date"] = pd.Timestamp(dt.date.today())
    fx["Date"] = pd.to_datetime(fx["Date"]).fillna(pd.Timestamp(dt.date.today()))
    last_played = matches["Date"].max()
    fx.loc[fx["Date"] <= last_played, "Date"] = last_played + pd.Timedelta(days=1)

    # Every fixture is scored as the *next* match of both teams given all
    # played matches (see features.next_match_features).
    rows = next_match_features(matches, fx[["HomeTeam", "AwayTeam", "Date"]], season_teams=season_teams)

    X = rows[bundle["features"]]
    probs = bundle["model"].predict_proba(X)
    out = pd.DataFrame({
        "Date": rows["Date"].dt.date.to_numpy(),
        "HomeTeam": rows["HomeTeam"].to_numpy(), "AwayTeam": rows["AwayTeam"].to_numpy(),
        "P(home win)": probs[:, 0], "P(draw)": probs[:, 1], "P(away win)": probs[:, 2],
    })
    out["Prediction"] = [ {"H": "Home win", "D": "Draw", "A": "Away win"}[config.CLASSES[i]] for i in probs.argmax(axis=1)]
    out["Confidence"] = probs.max(axis=1)
    out["Elo home"] = rows["elo_home"].round(0).to_numpy()
    out["Elo away"] = rows["elo_away"].round(0).to_numpy()
    out["xG home"] = rows["poisv_exp_hg"].round(2).to_numpy()   # Poisson expected goals
    out["xG away"] = rows["poisv_exp_ag"].round(2).to_numpy()
    return out.reset_index(drop=True)


def format_report(pred: pd.DataFrame, bundle: dict) -> str:
    lines = [f"Model: {bundle['model_name']}  |  data through {bundle['trained_through']}", ""]
    for _, r in pred.iterrows():
        lines.append(f"{r['Date']}  {r['HomeTeam']} vs {r['AwayTeam']}")
        lines.append(f"   home {r['P(home win)']:5.1%}   draw {r['P(draw)']:5.1%}   away {r['P(away win)']:5.1%}"
                     f"   ->  {r['Prediction']} ({r['Confidence']:.0%})")
        lines.append(f"   Elo {r['Elo home']:.0f} v {r['Elo away']:.0f}   expected goals {r['xG home']:.2f} - {r['xG away']:.2f}")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Predict Premier League match outcomes")
    ap.add_argument("home", nargs="?", help="home team")
    ap.add_argument("away", nargs="?", help="away team")
    ap.add_argument("--date", help="match date YYYY-MM-DD (default: today)")
    ap.add_argument("--fixtures", help="CSV with HomeTeam,AwayTeam[,Date] columns")
    ap.add_argument("--teams", action="store_true", help="list known team names and exit")
    ap.add_argument("--allow-new-team", action="store_true", help="accept teams not seen in the data (treated as promoted)")
    ap.add_argument("--csv", help="also write the predictions to this CSV path")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    from .live import get_live_matches
    try:
        matches, schedule = get_live_matches(refresh_data=False)
        season_teams = sorted(set(schedule["HomeTeam"]))
    except Exception as e:                       # offline: fall back to the stored table
        log.warning("could not fetch live results (%s); using stored data", e)
        matches, season_teams = get_matches(), None
    if args.teams:
        recent = set(matches[matches["Season"] == matches["Season"].max()]["HomeTeam"])
        for t in known_teams(matches):
            print(f"  {t}{'   (in the latest season)' if t in recent else ''}")
        return
    if args.fixtures:
        fixtures = pd.read_csv(args.fixtures)
    elif args.home and args.away:
        fixtures = pd.DataFrame({"HomeTeam": [args.home], "AwayTeam": [args.away],
                                 "Date": [args.date] if args.date else [None]})
    else:
        ap.error("give HOME and AWAY team names, or --fixtures file.csv")

    bundle = load_bundle()
    pred = predict_fixtures(fixtures, bundle, matches, allow_new=args.allow_new_team, season_teams=season_teams)
    print(format_report(pred, bundle))
    if args.csv:
        pred.to_csv(args.csv, index=False)
        print(f"saved {args.csv}")


if __name__ == "__main__":
    main()
