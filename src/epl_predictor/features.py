"""Leak-free feature engineering for match-outcome prediction.

Every feature for a match is computed from information available *before*
kick-off: earlier matches only. The building blocks are

* Elo and pi-ratings (sequential, pre-match values)          -> ratings.py
* rolling / exponentially-weighted form over the last N games (all venues)
* venue-specific form (home team's home form, away team's away form)
* season-to-date table position, points per game, goal difference
* head-to-head record over the last meetings (overall and at this venue)
* rest days, streaks, previous-season strength, promoted flag
* league-wide home-advantage level and scoring environment

Stats used: goals, shots, shots on target, corners, fouls, cards. Possession is
not in the free data, so shot-share ratios (TSR / STR) act as proxies.
"""
from __future__ import annotations

import logging
import warnings
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from . import config

# Columns are added one at a time for readability; pandas nags about
# fragmentation, which we fix by defragmenting (.copy()) at stage boundaries.
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
from .ratings import EloTracker, PerformanceEloTracker, PiRatingTracker

log = logging.getLogger(__name__)

ID_COLS = ["match_id", "Date", "Season", "HomeTeam", "AwayTeam"]
RAW_OUTCOME_COLS = ["FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR", "Referee",
                    "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]

# per-team-per-match stats in the long table:  <stat>f = for, <stat>a = against
LONG_STATS = ["gf", "ga", "sf", "sa", "stf", "sta", "cf", "ca", "ff", "fa", "yf", "ya", "rf", "ra"]


# ---------------------------------------------------------------------------
# 1. Long (team-match) table
# ---------------------------------------------------------------------------
def to_long(matches: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match: one from each team's perspective."""
    def side(is_home: bool) -> pd.DataFrame:
        t, o = ("Home", "Away") if is_home else ("Away", "Home")
        p, q = ("H", "A") if is_home else ("A", "H")
        d = pd.DataFrame({
            "match_id": matches["match_id"], "Date": matches["Date"], "Season": matches["Season"],
            "team": matches[f"{t}Team"], "opponent": matches[f"{o}Team"], "is_home": int(is_home),
            "gf": matches[f"FT{p}G"], "ga": matches[f"FT{q}G"],
            "sf": matches[f"{p}S"], "sa": matches[f"{q}S"],
            "stf": matches[f"{p}ST"], "sta": matches[f"{q}ST"],
            "cf": matches[f"{p}C"], "ca": matches[f"{q}C"],
            "ff": matches[f"{p}F"], "fa": matches[f"{q}F"],
            "yf": matches[f"{p}Y"], "ya": matches[f"{q}Y"],
            "rf": matches[f"{p}R"], "ra": matches[f"{q}R"],
        })
        return d

    long = pd.concat([side(True), side(False)], ignore_index=True)
    long["gd"] = long["gf"] - long["ga"]
    played = long["gd"].notna()                    # future fixtures have no result yet
    long["win"] = (long["gd"] > 0).astype(float).where(played)
    long["draw"] = (long["gd"] == 0).astype(float).where(played)
    long["loss"] = (long["gd"] < 0).astype(float).where(played)
    long["pts"] = long["win"] * 3 + long["draw"]
    long["sd"] = long["sf"] - long["sa"]          # shot difference
    long["std"] = long["stf"] - long["sta"]       # shots-on-target difference
    # goals per shot-on-target (finishing) and SoT per shot (shot quality) proxies
    long = long.sort_values(["Date", "match_id", "is_home"], ascending=[True, True, False]).reset_index(drop=True)
    return long


# ---------------------------------------------------------------------------
# 2. Rolling / EWMA form (all venues)
# ---------------------------------------------------------------------------
def _shifted(g, col):
    """Series of `col` shifted by one *within each team* (previous matches only)."""
    return g[col].shift(1)


def add_rolling_form(long: pd.DataFrame) -> pd.DataFrame:
    g = long.groupby("team", sort=False)
    mean_cols = ["pts", "gf", "ga", "gd", "sf", "sa", "stf", "sta", "cf", "ca", "sd", "std", "ff", "yf"]

    for w in config.FORM_WINDOWS:
        for c in mean_cols:
            long[f"{c}_r{w}"] = _shifted(g, c).groupby(long["team"]).transform(
                lambda s: s.rolling(w, min_periods=1).mean())
        # ratio features from rolling sums (more stable than mean-of-ratios)
        sums = {}
        for c in ["gf", "sf", "sa", "stf", "sta", "win", "draw", "loss"]:
            sums[c] = _shifted(g, c).groupby(long["team"]).transform(
                lambda s: s.rolling(w, min_periods=1).sum())
        long[f"conv_r{w}"] = sums["gf"] / sums["sf"].replace(0, np.nan)          # goals per shot
        long[f"sotpct_r{w}"] = sums["stf"] / sums["sf"].replace(0, np.nan)      # SoT per shot
        long[f"tsr_r{w}"] = sums["sf"] / (sums["sf"] + sums["sa"]).replace(0, np.nan)   # total shots ratio
        long[f"str_r{w}"] = sums["stf"] / (sums["stf"] + sums["sta"]).replace(0, np.nan)  # SoT ratio
        long[f"winrate_r{w}"] = sums["win"] / w
        long[f"drawrate_r{w}"] = sums["draw"] / w
        long[f"lossrate_r{w}"] = sums["loss"] / w

    # Exponentially weighted (recent matches count more)
    for c in ["pts", "gf", "ga", "gd", "sf", "sa", "stf", "sta", "sd", "std", "cf", "ca"]:
        long[f"{c}_ewm"] = _shifted(g, c).groupby(long["team"]).transform(
            lambda s: s.ewm(halflife=config.EWMA_HALFLIFE, min_periods=1).mean())

    # Long-run (last 38 = ~ one season) baseline quality
    for c in ["pts", "gd", "std"]:
        long[f"{c}_r38"] = _shifted(g, c).groupby(long["team"]).transform(
            lambda s: s.rolling(38, min_periods=5).mean())

    # Momentum: short-term form relative to long-run level
    long["momentum_pts"] = long["pts_r5"] - long["pts_r38"]
    long["momentum_gd"] = long["gd_r5"] - long["gd_r38"]
    return long


# ---------------------------------------------------------------------------
# 3. Venue-specific form and team-specific home advantage
# ---------------------------------------------------------------------------
def add_venue_form(long: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """For each team-match row: the team's form in its last `window` matches at
    the *same venue type* (home rows -> home form, away rows -> away form),
    plus a home-advantage measure = home ppg - away ppg (EWMA, pre-match).

    Implementation trick: compute inclusive rolling values on the venue subset,
    then shift by one row within the team and forward-fill, which yields the
    latest *completed* venue value before every match.
    """
    g_team = long.groupby("team", sort=False)
    for venue, flag in (("home", 1), ("away", 0)):
        mask = long["is_home"] == flag
        sub = long.loc[mask]
        gs = sub.groupby("team", sort=False)
        incl = {
            f"{venue}_pts_v{window}": gs["pts"].transform(lambda s: s.rolling(window, min_periods=1).mean()),
            f"{venue}_gf_v{window}": gs["gf"].transform(lambda s: s.rolling(window, min_periods=1).mean()),
            f"{venue}_ga_v{window}": gs["ga"].transform(lambda s: s.rolling(window, min_periods=1).mean()),
            f"{venue}_std_v{window}": gs["std"].transform(lambda s: s.rolling(window, min_periods=1).mean()),
            f"{venue}_pts_ewm": gs["pts"].transform(lambda s: s.ewm(halflife=10, min_periods=1).mean()),
            f"{venue}_gd_ewm": gs["gd"].transform(lambda s: s.ewm(halflife=10, min_periods=1).mean()),
            f"{venue}_gf_ewm": gs["gf"].transform(lambda s: s.ewm(halflife=10, min_periods=1).mean()),
            f"{venue}_ga_ewm": gs["ga"].transform(lambda s: s.ewm(halflife=10, min_periods=1).mean()),
            f"{venue}_stf_ewm": gs["stf"].transform(lambda s: s.ewm(halflife=10, min_periods=1).mean()),
            f"{venue}_sta_ewm": gs["sta"].transform(lambda s: s.ewm(halflife=10, min_periods=1).mean()),
        }
        for col, vals in incl.items():
            tmp = pd.Series(np.nan, index=long.index)
            tmp.loc[mask] = vals
            # value after the previous row of this team, carried forward -> pre-match
            long[col] = tmp.groupby(long["team"]).shift(1).groupby(long["team"]).ffill()

    is_home = long["is_home"] == 1
    for stat in ("pts", "gf", "ga", "std"):
        long[f"venue_{stat}_v{window}"] = np.where(
            is_home, long[f"home_{stat}_v{window}"], long[f"away_{stat}_v{window}"])
    for stat in ("pts", "gd", "gf", "ga", "stf", "sta"):
        long[f"venue_{stat}_ewm"] = np.where(is_home, long[f"home_{stat}_ewm"], long[f"away_{stat}_ewm"])
    # team-specific home advantage (how much better the team is at home than away)
    long["team_home_adv_pts"] = long["home_pts_ewm"] - long["away_pts_ewm"]
    long["team_home_adv_gd"] = long["home_gd_ewm"] - long["away_gd_ewm"]
    return long


# ---------------------------------------------------------------------------
# 4. Season-to-date, league table position, rest, streaks
# ---------------------------------------------------------------------------
def add_season_to_date(long: pd.DataFrame) -> pd.DataFrame:
    g = long.groupby(["team", "Season"], sort=False)
    long["games_played"] = g.cumcount()
    for c in ["pts", "gf", "ga", "gd", "win", "draw", "loss"]:
        filled = long[c].fillna(0)
        long[f"season_{c}"] = filled.groupby([long["team"], long["Season"]]).cumsum() - filled  # before this match
    gp = long["games_played"].replace(0, np.nan)
    long["season_ppg"] = long["season_pts"] / gp
    long["season_gd_pg"] = long["season_gd"] / gp
    long["season_gf_pg"] = long["season_gf"] / gp
    long["season_ga_pg"] = long["season_ga"] / gp
    long["season_winrate"] = long["season_win"] / gp
    long["season_drawrate"] = long["season_draw"] / gp
    return long


def add_league_position(long: pd.DataFrame) -> pd.DataFrame:
    """League position (and points gap to leader) before each match date."""
    pos_frames = []
    for season, d in long.groupby("Season", sort=False):
        pts = d.pivot_table(index="Date", columns="team", values="pts", aggfunc="sum").fillna(0)
        gd = d.pivot_table(index="Date", columns="team", values="gd", aggfunc="sum").fillna(0)
        gf = d.pivot_table(index="Date", columns="team", values="gf", aggfunc="sum").fillna(0)
        cpts = pts.cumsum().shift(1).fillna(0)
        cgd = gd.cumsum().shift(1).fillna(0)
        cgf = gf.cumsum().shift(1).fillna(0)
        key = cpts * 1e6 + cgd * 1e3 + cgf                      # PL tie-break order
        pos = key.rank(axis=1, ascending=False, method="min")
        gap = cpts.max(axis=1).to_numpy()[:, None] - cpts.to_numpy()
        n_teams = pts.shape[1]
        st = pos.stack().rename("league_pos").reset_index()
        st.columns = ["Date", "team", "league_pos"]
        st["pos_pct"] = (st["league_pos"] - 1) / (n_teams - 1)
        st["pts_gap_to_top"] = pd.DataFrame(gap, index=cpts.index, columns=cpts.columns).stack().to_numpy()
        st["Season"] = season
        pos_frames.append(st)
    pos_all = pd.concat(pos_frames, ignore_index=True)
    return long.merge(pos_all, on=["Season", "Date", "team"], how="left")


def add_rest_and_streaks(long: pd.DataFrame) -> pd.DataFrame:
    g = long.groupby("team", sort=False)
    rest = g["Date"].diff().dt.days
    long["rest_days"] = rest.clip(upper=21).fillna(21)
    # matches played in the last 14 days (fixture congestion)
    long["congestion_14d"] = 0.0
    for team, idx in g.indices.items():
        dates = long.loc[idx, "Date"].to_numpy()
        d_num = dates.astype("datetime64[D]").astype(np.int64)
        lo = np.searchsorted(d_num, d_num - 14, side="left")
        long.loc[idx, "congestion_14d"] = np.arange(len(d_num)) - lo

    def run_before(flags: pd.Series) -> pd.Series:
        grp = (flags != flags.shift()).cumsum()
        run = flags.groupby(grp).cumsum()
        return run.shift(1).fillna(0)

    win, loss = long["win"].fillna(0), long["loss"].fillna(0)
    for name, flag in (("unbeaten", loss == 0), ("winless", win == 0),
                       ("win", win == 1), ("loss", loss == 1)):
        long[f"streak_{name}"] = flag.astype(int).groupby(long["team"]).transform(run_before)
    return long


def add_previous_season(long: pd.DataFrame) -> pd.DataFrame:
    season_tbl = (long.groupby(["Season", "team"])
                  .agg(prev_ppg=("pts", "mean"), prev_gd_pg=("gd", "mean"),
                       prev_std=("std", "mean"), n=("pts", "size"))
                  .reset_index())
    season_tbl["prev_pos"] = season_tbl.groupby("Season")["prev_ppg"].rank(ascending=False, method="min")
    season_tbl["Season"] = season_tbl["Season"] + 1          # attach to the *next* season
    season_tbl = season_tbl.drop(columns="n")
    long = long.merge(season_tbl, on=["Season", "team"], how="left")
    long["promoted"] = long["prev_ppg"].isna().astype(int)
    # Promoted teams: fill with a typical newly-promoted profile (bottom of table)
    long["prev_ppg"] = long["prev_ppg"].fillna(0.95)
    long["prev_gd_pg"] = long["prev_gd_pg"].fillna(-0.7)
    long["prev_std"] = long["prev_std"].fillna(-1.5)
    long["prev_pos"] = long["prev_pos"].fillna(18)
    return long


# ---------------------------------------------------------------------------
# 5. Head-to-head
# ---------------------------------------------------------------------------
def head_to_head(matches: pd.DataFrame, window: int = config.H2H_WINDOW) -> pd.DataFrame:
    """Record of previous meetings, from the home team's perspective."""
    hist: dict[frozenset, deque] = defaultdict(lambda: deque(maxlen=50))
    venue_hist: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=50))
    rows = np.full((len(matches), 7), np.nan)
    for i, (home, away, hg, ag) in enumerate(
        zip(matches["HomeTeam"], matches["AwayTeam"], matches["FTHG"], matches["FTAG"])
    ):
        key = frozenset((home, away))
        past = list(hist[key])[-window:]
        if past:
            # each entry: (team that was at home, goal diff for that home side)
            gds = [gd if h == home else -gd for h, gd in past]
            wins = np.mean([x > 0 for x in gds])
            draws = np.mean([x == 0 for x in gds])
            losses = np.mean([x < 0 for x in gds])
            rows[i, :5] = (len(past), wins, draws, losses, np.mean(gds))
        vpast = list(venue_hist[(home, away)])[-window:]
        if vpast:
            rows[i, 5:] = (len(vpast), np.mean(vpast))
        if pd.isna(hg) or pd.isna(ag):
            continue                                   # fixture not played yet
        gd = int(hg) - int(ag)
        hist[key].append((home, gd))
        venue_hist[(home, away)].append(gd)
    h2h = pd.DataFrame(rows, columns=["h2h_n", "h2h_home_winrate", "h2h_drawrate", "h2h_home_lossrate",
                                      "h2h_home_gd", "h2h_venue_n", "h2h_venue_gd"], index=matches.index)
    h2h["h2h_n"] = h2h["h2h_n"].fillna(0)
    h2h["h2h_venue_n"] = h2h["h2h_venue_n"].fillna(0)
    return h2h


# ---------------------------------------------------------------------------
# 6. League environment
# ---------------------------------------------------------------------------
def league_context(matches: pd.DataFrame, window: int = 380) -> pd.DataFrame:
    """Rolling league-wide home-win rate, draw rate and goals per game."""
    out = pd.DataFrame(index=matches.index)
    is_home_win = matches["FTR"].map({"H": 1.0, "D": 0.0, "A": 0.0})   # NaN for unplayed
    is_draw = matches["FTR"].map({"H": 0.0, "D": 1.0, "A": 0.0})
    goals = (matches["FTHG"] + matches["FTAG"]).astype(float)
    out["league_home_winrate"] = is_home_win.shift(1).rolling(window, min_periods=50).mean()
    out["league_drawrate"] = is_draw.shift(1).rolling(window, min_periods=50).mean()
    out["league_goals_pg"] = goals.shift(1).rolling(window, min_periods=50).mean()
    out["league_home_goals_pg"] = matches["FTHG"].astype(float).shift(1).rolling(window, min_periods=50).mean()
    out["league_away_goals_pg"] = matches["FTAG"].astype(float).shift(1).rolling(window, min_periods=50).mean()
    out["month"] = matches["Date"].dt.month
    out["day_of_week"] = matches["Date"].dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    return out


# ---------------------------------------------------------------------------
# 7. Poisson attack/defence match-up model (a lightweight Dixon-Coles cousin)
# ---------------------------------------------------------------------------
def poisson_outcome_probs(exp_home: np.ndarray, exp_away: np.ndarray, max_goals: int = 10) -> np.ndarray:
    """P(H), P(D), P(A) from independent Poisson goal expectations."""
    from scipy.stats import poisson
    k = np.arange(max_goals + 1)
    ph = poisson.pmf(k[None, :], np.asarray(exp_home)[:, None])   # (n, k)
    pa = poisson.pmf(k[None, :], np.asarray(exp_away)[:, None])
    joint = ph[:, :, None] * pa[:, None, :]                        # (n, kh, ka)
    kh, ka = np.meshgrid(k, k, indexing="ij")
    p_home = joint[:, kh > ka].sum(axis=1)
    p_draw = joint[:, kh == ka].sum(axis=1)
    p_away = joint[:, kh < ka].sum(axis=1)
    return np.c_[p_home, p_draw, p_away]


def add_poisson_features(wide: pd.DataFrame) -> pd.DataFrame:
    """Expected goals for each side = attack strength x opponent defence
    weakness x league scoring rate, using exponentially-weighted goal rates.
    Two flavours: all-venue strengths, and venue-specific (home team's home
    attack vs away team's away defence)."""
    # league scoring rates (long-run PL averages as a fallback for the first matches)
    L = (wide["league_goals_pg"].fillna(2.65) / 2.0).replace(0, np.nan)   # goals per team per match
    Lh = wide["league_home_goals_pg"].fillna(1.50)
    La = wide["league_away_goals_pg"].fillna(1.15)

    exp_hg = (wide["h_gf_ewm"] / L) * (wide["a_ga_ewm"] / L) * Lh
    exp_ag = (wide["a_gf_ewm"] / L) * (wide["h_ga_ewm"] / L) * La
    exp_hg_v = wide["h_venue_gf_ewm"] * wide["a_venue_ga_ewm"] / Lh
    exp_ag_v = wide["a_venue_gf_ewm"] * wide["h_venue_ga_ewm"] / La

    for name, eh, ea in (("pois", exp_hg, exp_ag), ("poisv", exp_hg_v, exp_ag_v)):
        eh = eh.clip(0.05, 6.0).fillna(Lh)
        ea = ea.clip(0.05, 6.0).fillna(La)
        p = poisson_outcome_probs(eh.to_numpy(), ea.to_numpy())
        wide[f"{name}_exp_hg"] = eh
        wide[f"{name}_exp_ag"] = ea
        wide[f"{name}_exp_gd"] = eh - ea
        wide[f"{name}_pH"], wide[f"{name}_pD"], wide[f"{name}_pA"] = p[:, 0], p[:, 1], p[:, 2]
    # shots-on-target version of the match-up (dominance rather than goals)
    wide["sot_matchup_home"] = wide["h_venue_stf_ewm"] * wide["a_venue_sta_ewm"]
    wide["sot_matchup_away"] = wide["a_venue_stf_ewm"] * wide["h_venue_sta_ewm"]
    wide["sot_matchup_diff"] = wide["sot_matchup_home"] - wide["sot_matchup_away"]
    return wide


# ---------------------------------------------------------------------------
# 8. Assemble the wide feature matrix
# ---------------------------------------------------------------------------
TEAM_FEATURE_EXCLUDE = set(["match_id", "Date", "Season", "team", "opponent", "is_home"] + LONG_STATS +
                           ["gd", "win", "draw", "loss", "pts", "sd", "std"] +
                           [f"{v}_{s}" for v in ("home", "away") for s in
                            ("pts_v5", "gf_v5", "ga_v5", "std_v5", "pts_ewm", "gd_ewm",
                             "gf_ewm", "ga_ewm", "stf_ewm", "sta_ewm")])

DIFF_FEATURES = [
    "pts_r5", "pts_r10", "gd_r5", "gd_r10", "std_r5", "std_r10", "sd_r10", "tsr_r10", "str_r10",
    "conv_r10", "pts_ewm", "gd_ewm", "std_ewm", "gf_ewm", "ga_ewm", "pts_r38", "gd_r38", "std_r38",
    "venue_pts_v5", "venue_gd_ewm", "venue_pts_ewm", "venue_std_v5", "season_ppg", "season_gd_pg",
    "league_pos", "pts_gap_to_top", "prev_ppg", "prev_pos", "rest_days", "congestion_14d",
    "momentum_pts", "streak_unbeaten", "streak_winless", "team_home_adv_pts",
]


def team_features_from_long(long: pd.DataFrame) -> pd.DataFrame:
    long = long.sort_values(["Date", "match_id", "is_home"], ascending=[True, True, False]).reset_index(drop=True)
    long = add_rolling_form(long).copy()
    long = add_venue_form(long).copy()
    long = add_season_to_date(long).copy()
    long = add_league_position(long)
    long = add_rest_and_streaks(long).copy()
    long = add_previous_season(long)
    return long.copy()


def build_team_features(matches: pd.DataFrame) -> pd.DataFrame:
    return team_features_from_long(to_long(matches))


def _pairwise_derived(wide: pd.DataFrame) -> pd.DataFrame:
    """Features that combine the two sides' team features (diffs, match-ups)."""
    for f in DIFF_FEATURES:
        if f"h_{f}" in wide and f"a_{f}" in wide:
            wide[f"d_{f}"] = wide[f"h_{f}"] - wide[f"a_{f}"]
    # attack-vs-defence match-ups (home attack vs away defence, and vice versa)
    wide["h_att_vs_a_def"] = wide["h_gf_ewm"] - wide["a_ga_ewm"]
    wide["a_att_vs_h_def"] = wide["a_gf_ewm"] - wide["h_ga_ewm"]
    wide["h_sot_att_vs_a_def"] = wide["h_stf_ewm"] - wide["a_sta_ewm"]
    wide["a_sot_att_vs_h_def"] = wide["a_stf_ewm"] - wide["h_sta_ewm"]
    return wide


def _rating_derived(wide: pd.DataFrame) -> pd.DataFrame:
    wide["elo_diff"] = wide["elo_home"] - wide["elo_away"]
    wide["pelo_diff"] = wide["pelo_home"] - wide["pelo_away"]
    wide["pi_diff_overall"] = wide["pi_home_overall"] - wide["pi_away_overall"]
    wide["pi_diff_venue"] = wide["pi_home_venue"] - wide["pi_away_venue"]
    return add_poisson_features(wide)


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Return one row per match with all pre-match features + the target."""
    matches = matches.sort_values(["Date", "match_id"]).reset_index(drop=True)
    long = build_team_features(matches)
    feat_cols = [c for c in long.columns if c not in TEAM_FEATURE_EXCLUDE]

    home = long[long["is_home"] == 1].set_index("match_id")[feat_cols].add_prefix("h_")
    away = long[long["is_home"] == 0].set_index("match_id")[feat_cols].add_prefix("a_")

    wide = matches.set_index("match_id")
    wide = wide.join(home).join(away)
    wide = _pairwise_derived(wide).reset_index()

    # Ratings, head-to-head and league context (all pre-match)
    elo = EloTracker().run(wide)
    pelo = PerformanceEloTracker().run(wide)
    pi = PiRatingTracker().run(wide)
    wide = pd.concat([wide, elo, pelo, pi, head_to_head(wide), league_context(wide)], axis=1)
    wide = _rating_derived(wide)

    wide["target"] = wide["FTR"].map(config.CLASS_TO_INT)
    return wide.copy()


# ---------------------------------------------------------------------------
# 9. Features for hypothetical fixtures (any number, teams may repeat)
# ---------------------------------------------------------------------------
def next_match_features(matches: pd.DataFrame, fixtures: pd.DataFrame,
                        season_teams: list[str] | None = None) -> pd.DataFrame:
    """Pre-match features for hypothetical fixtures, each treated as the *next*
    match of both teams given every played match in `matches`.

    `fixtures` needs HomeTeam, AwayTeam and Date. Unlike appending fixtures to
    the match table one round at a time, this evaluates hundreds of pairings
    (a full matchup matrix, the rest of the season) from one pass over the
    history, because a team's own form/venue/season features do not depend on
    the opponent: they are read off once per team, and only the pairwise parts
    (ratings expectation, head-to-head, Poisson match-up, diffs, rest) are
    assembled per fixture.
    """
    from .ratings import _pi_expected_gd, elo_expected

    matches = matches.sort_values(["Date", "match_id"]).reset_index(drop=True)
    fixtures = fixtures.copy().reset_index(drop=True)
    fixtures["Date"] = pd.to_datetime(fixtures["Date"])
    last_date = matches["Date"].max()
    anchor = max(fixtures["Date"].min(), last_date + pd.Timedelta(days=1))
    season = anchor.year if anchor.month >= 7 else anchor.year - 1
    teams = sorted(set(fixtures["HomeTeam"]) | set(fixtures["AwayTeam"]))
    season_teams = set(season_teams or teams) | set(teams)

    # --- team-side features: one dummy row per team and venue, in two runs so
    #     a team's dummy rows never see each other
    long_real = to_long(matches)
    next_id = matches["match_id"].max() + 1
    side_feats = {}
    for venue, is_home in (("h", 1), ("a", 0)):
        dummy = pd.DataFrame({
            "match_id": range(next_id, next_id + len(teams)), "Date": anchor, "Season": season,
            "team": teams, "opponent": "", "is_home": is_home,
        })
        for c in LONG_STATS + ["gd", "win", "draw", "loss", "pts", "sd", "std"]:
            dummy[c] = np.nan
        long = team_features_from_long(pd.concat([long_real, dummy[long_real.columns]], ignore_index=True))
        rows = long[long["match_id"] >= next_id].set_index("team")
        feat_cols = [c for c in long.columns if c not in TEAM_FEATURE_EXCLUDE]
        side_feats[venue] = rows[feat_cols].add_prefix(f"{venue}_")

    wide = fixtures[["HomeTeam", "AwayTeam", "Date"]].copy()
    wide["Season"] = season
    wide = wide.join(side_feats["h"], on="HomeTeam").join(side_feats["a"], on="AwayTeam")

    # rest days / congestion depend on the fixture date, not the anchor
    last_played = long_real.groupby("team")["Date"].max()
    dates_by_team = {t: np.sort(d["Date"].to_numpy().astype("datetime64[D]").astype(np.int64))
                     for t, d in long_real.groupby("team")}
    for venue, col in (("h", "HomeTeam"), ("a", "AwayTeam")):
        rest, cong = [], []
        for team, date in zip(wide[col], wide["Date"]):
            gap = (date - last_played.get(team, date - pd.Timedelta(days=21))).days
            if (date - last_date).days > 14:          # far ahead: assume a normal week
                rest.append(7.0); cong.append(1.0)
            else:
                rest.append(float(min(gap, 21)))
                d_num = np.int64(date.to_datetime64().astype("datetime64[D]").astype(np.int64))
                arr = dates_by_team.get(team, np.array([], dtype=np.int64))
                cong.append(float(len(arr) - np.searchsorted(arr, d_num - 14, side="left")))
        wide[f"{venue}_rest_days"] = rest
        wide[f"{venue}_congestion_14d"] = cong
    wide = _pairwise_derived(wide)

    # --- ratings: tracker state after all played matches (+ new-season reset)
    elo, pelo, pi = EloTracker(), PerformanceEloTracker(), PiRatingTracker()
    elo.run(matches); pelo.run(matches); pi.run(matches)
    if season != matches["Season"].max():
        elo._new_season(season, season_teams)
        pelo._new_season(season, season_teams)
    wide["elo_home"] = [elo.get(t) for t in wide["HomeTeam"]]
    wide["elo_away"] = [elo.get(t) for t in wide["AwayTeam"]]
    wide["elo_exp_home"] = [elo_expected(h, a, elo.home_adv) for h, a in zip(wide["elo_home"], wide["elo_away"])]
    wide["pelo_home"] = [pelo.get(t) for t in wide["HomeTeam"]]
    wide["pelo_away"] = [pelo.get(t) for t in wide["AwayTeam"]]
    wide["pelo_exp_home"] = [elo_expected(h, a, pelo.home_adv) for h, a in zip(wide["pelo_home"], wide["pelo_away"])]
    wide["pi_home_venue"] = [pi.home_r.get(t, 0.0) for t in wide["HomeTeam"]]
    wide["pi_home_overall"] = [(pi.home_r.get(t, 0.0) + pi.away_r.get(t, 0.0)) / 2 for t in wide["HomeTeam"]]
    wide["pi_away_venue"] = [pi.away_r.get(t, 0.0) for t in wide["AwayTeam"]]
    wide["pi_away_overall"] = [(pi.home_r.get(t, 0.0) + pi.away_r.get(t, 0.0)) / 2 for t in wide["AwayTeam"]]
    wide["pi_exp_gd"] = [pi.expected_gd(h, a) for h, a in zip(wide["HomeTeam"], wide["AwayTeam"])]

    # --- head-to-head (fixtures are unplayed, so they never enter the history)
    fx = wide[["HomeTeam", "AwayTeam", "Date"]].copy()
    fx["FTHG"] = np.nan; fx["FTAG"] = np.nan
    h2h = head_to_head(pd.concat([matches[["HomeTeam", "AwayTeam", "Date", "FTHG", "FTAG"]], fx], ignore_index=True))
    h2h = h2h.iloc[len(matches):].reset_index(drop=True)
    wide = pd.concat([wide.reset_index(drop=True), h2h], axis=1)

    # --- league environment as of now
    recent = matches.tail(380)
    wide["league_home_winrate"] = (recent["FTR"] == "H").mean()
    wide["league_drawrate"] = (recent["FTR"] == "D").mean()
    wide["league_goals_pg"] = (recent["FTHG"] + recent["FTAG"]).mean()
    wide["league_home_goals_pg"] = recent["FTHG"].mean()
    wide["league_away_goals_pg"] = recent["FTAG"].mean()
    wide["month"] = wide["Date"].dt.month
    wide["day_of_week"] = wide["Date"].dt.dayofweek
    wide["is_weekend"] = (wide["day_of_week"] >= 5).astype(int)
    wide = _rating_derived(wide)
    return wide.copy()


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All model input columns (everything that is not an id / raw outcome)."""
    drop = set(ID_COLS + RAW_OUTCOME_COLS + ["target"])
    return [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


def get_features(refresh: bool = False, refresh_data: bool = False) -> pd.DataFrame:
    """Cached feature matrix. `refresh` rebuilds features from the stored
    matches; `refresh_data` additionally re-downloads the latest seasons."""
    from .data import get_matches
    out = config.PROCESSED_DIR / "features.parquet"
    if out.exists() and not (refresh or refresh_data):
        return pd.read_parquet(out)
    matches = get_matches(refresh=refresh_data)
    feats = build_features(matches)
    feats.to_parquet(out, index=False)
    log.info("built %d features for %d matches", len(feature_columns(feats)), len(feats))
    return feats


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    f = get_features(refresh=True)
    cols = feature_columns(f)
    print(f"{len(cols)} features, {len(f)} matches")
    print(f[cols].describe().T.to_string())
