"""Live data for the season in progress: results so far and the remaining fixtures.

Source: the openfootball project (https://github.com/openfootball/football.json),
which publishes every Premier League season as JSON with kick-off times, the
matchday, and full/half-time scores once a match has been played. It carries no
shot statistics, so those columns stay empty for current-season matches until
the football-data.co.uk mirror publishes its file for the season (data.py picks
that up automatically and it then takes precedence).
"""
from __future__ import annotations

import difflib
import json
import logging
import urllib.error
import urllib.request

import pandas as pd

from . import config
from .data import STAT_COLS, get_matches, season_code

log = logging.getLogger(__name__)

OPENFOOTBALL_URL = "https://raw.githubusercontent.com/openfootball/football.json/master/{start}-{end:02d}/en.1.json"

# openfootball spelling -> football-data.co.uk spelling used throughout the project
OPENFOOTBALL_NAMES = {
    "AFC Bournemouth": "Bournemouth", "Arsenal FC": "Arsenal", "Aston Villa FC": "Aston Villa",
    "Brentford FC": "Brentford", "Brighton & Hove Albion FC": "Brighton", "Burnley FC": "Burnley",
    "Chelsea FC": "Chelsea", "Coventry City FC": "Coventry", "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton", "Fulham FC": "Fulham", "Hull City AFC": "Hull", "Ipswich Town FC": "Ipswich",
    "Leeds United FC": "Leeds", "Leicester City FC": "Leicester", "Liverpool FC": "Liverpool",
    "Luton Town FC": "Luton", "Manchester City FC": "Man City", "Manchester United FC": "Man United",
    "Middlesbrough FC": "Middlesbrough", "Newcastle United FC": "Newcastle", "Norwich City FC": "Norwich",
    "Nottingham Forest FC": "Nott'm Forest", "Sheffield United FC": "Sheffield United",
    "Southampton FC": "Southampton", "Stoke City FC": "Stoke", "Sunderland AFC": "Sunderland",
    "Swansea City FC": "Swansea", "Tottenham Hotspur FC": "Tottenham", "Watford FC": "Watford",
    "West Bromwich Albion FC": "West Brom", "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves", "Blackburn Rovers FC": "Blackburn", "Bolton Wanderers FC": "Bolton",
    "Cardiff City FC": "Cardiff", "Huddersfield Town AFC": "Huddersfield", "Queens Park Rangers FC": "QPR",
    "Reading FC": "Reading", "Wigan Athletic FC": "Wigan", "Derby County FC": "Derby",
    "Birmingham City FC": "Birmingham", "Blackpool FC": "Blackpool", "Portsmouth FC": "Portsmouth",
    "Charlton Athletic FC": "Charlton", "Wimbledon FC": "Wimbledon", "Sheffield Wednesday FC": "Sheffield Weds",
    "Oldham Athletic AFC": "Oldham", "Swindon Town FC": "Swindon", "Barnsley FC": "Barnsley",
    "Bradford City AFC": "Bradford", "Plymouth Argyle FC": "Plymouth", "Preston North End FC": "Preston",
    "Millwall FC": "Millwall", "Wrexham AFC": "Wrexham",
}


def canonical_name(name: str, known: list[str] | None = None) -> str:
    if name in OPENFOOTBALL_NAMES:
        return OPENFOOTBALL_NAMES[name]
    stripped = name.replace(" FC", "").replace(" AFC", "").replace("AFC ", "").strip()
    if known:
        for t in known:
            if t.lower() == stripped.lower():
                return t
        close = difflib.get_close_matches(stripped, known, n=1, cutoff=0.75)
        if close:
            return close[0]
    log.warning("no canonical name for '%s'; using '%s'", name, stripped)
    return stripped


def fetch_openfootball_season(start_year: int) -> list[dict] | None:
    url = OPENFOOTBALL_URL.format(start=start_year, end=(start_year + 1) % 100)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))["matches"]
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def season_schedule(start_year: int, known_teams: list[str] | None = None) -> pd.DataFrame:
    """All 380 fixtures of a season: played ones carry scores, the rest do not."""
    raw = fetch_openfootball_season(start_year)
    if raw is None:
        return pd.DataFrame(columns=["Season", "Round", "Date", "Time", "HomeTeam", "AwayTeam",
                                     "FTHG", "FTAG", "HTHG", "HTAG", "played"])
    rows = []
    for m in raw:
        ft = (m.get("score") or {}).get("ft")
        ht = (m.get("score") or {}).get("ht")
        rows.append({
            "Season": start_year,
            "Round": int(str(m.get("round", "0")).replace("Matchday", "").strip() or 0),
            "Date": pd.Timestamp(m["date"]),
            "Time": m.get("time"),
            "HomeTeam": canonical_name(m["team1"], known_teams),
            "AwayTeam": canonical_name(m["team2"], known_teams),
            "FTHG": ft[0] if ft else None, "FTAG": ft[1] if ft else None,
            "HTHG": ht[0] if ht else None, "HTAG": ht[1] if ht else None,
            "played": bool(ft),
        })
    df = pd.DataFrame(rows).sort_values(["Date", "Time", "HomeTeam"]).reset_index(drop=True)
    return df


def current_season_start(matches: pd.DataFrame | None = None) -> int:
    import datetime as dt
    today = dt.date.today()
    return today.year if today.month >= 7 else today.year - 1


def get_live_matches(refresh_data: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(matches, schedule): the historical match table extended with the
    current season's played matches, and the current season's full schedule."""
    matches = get_matches(refresh=refresh_data)
    known = sorted(set(matches["HomeTeam"]) | set(matches["AwayTeam"]))
    season = current_season_start()
    schedule = season_schedule(season, known)
    if schedule.empty:
        # the season may not have started; fall back to last season's schedule for context
        log.info("no openfootball data for %s; using previous season's schedule", season)
        season -= 1
        schedule = season_schedule(season, known)

    played = schedule[schedule["played"]].copy()
    have = matches[matches["Season"] == season]
    if len(have):
        # the mirror (with shot stats) already covers some/all of this season: keep it,
        # add only the matches it does not have yet
        key = set(zip(have["HomeTeam"], have["AwayTeam"]))
        played = played[[(h, a) not in key for h, a in zip(played["HomeTeam"], played["AwayTeam"])]]
    if len(played):
        add = pd.DataFrame({
            "match_id": range(matches["match_id"].max() + 1, matches["match_id"].max() + 1 + len(played)),
            "Date": played["Date"].to_numpy(), "HomeTeam": played["HomeTeam"].to_numpy(),
            "AwayTeam": played["AwayTeam"].to_numpy(),
            "FTHG": played["FTHG"].astype(int).to_numpy(), "FTAG": played["FTAG"].astype(int).to_numpy(),
            "HTHG": played["HTHG"].to_numpy(), "HTAG": played["HTAG"].to_numpy(), "Referee": None,
            "Season": season,
        })
        add["FTR"] = ["H" if h > a else "A" if a > h else "D" for h, a in zip(add["FTHG"], add["FTAG"])]
        add["HTR"] = None
        for c in STAT_COLS:
            add[c] = float("nan")
        matches = pd.concat([matches, add[matches.columns]], ignore_index=True)
        matches = matches.sort_values(["Date", "match_id"]).reset_index(drop=True)
        log.info("added %d played matches of %s/%02d from openfootball", len(add), season, (season + 1) % 100)
    return matches, schedule


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    m, s = get_live_matches(refresh_data=False)
    print(m.tail(3).to_string())
    print(s[s["played"]].tail(3).to_string())
    print("next round:", s[~s["played"]]["Round"].min(), "| played", s["played"].sum(), "of", len(s))
