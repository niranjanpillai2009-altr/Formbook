"""Sanity tests: features must be computed from *earlier* matches only."""
import numpy as np
import pandas as pd
import pytest

from epl_predictor.features import build_features, feature_columns, head_to_head, to_long
from epl_predictor.ratings import EloTracker, PiRatingTracker, elo_expected


def _toy_matches() -> pd.DataFrame:
    rows = [
        ("2023-08-12", "A", "B", 2, 0), ("2023-08-12", "C", "D", 1, 1),
        ("2023-08-19", "B", "C", 0, 3), ("2023-08-19", "D", "A", 1, 2),
        ("2023-08-26", "A", "C", 0, 0), ("2023-08-26", "B", "D", 2, 1),
        ("2023-09-02", "C", "A", 1, 0), ("2023-09-02", "D", "B", 0, 0),
    ]
    df = pd.DataFrame(rows, columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df["Date"] = pd.to_datetime(df["Date"])
    df["FTR"] = np.where(df.FTHG > df.FTAG, "H", np.where(df.FTHG < df.FTAG, "A", "D"))
    for c in ["HTHG", "HTAG", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]:
        df[c] = np.random.RandomState(0).randint(0, 10, len(df)).astype(float)
    df["HTR"] = "D"; df["Referee"] = "X"; df["Season"] = 2023
    df.insert(0, "match_id", range(len(df)))
    return df


def test_elo_is_pre_match_and_zero_sum():
    m = _toy_matches()
    e = EloTracker(k=20).run(m)
    assert np.isclose(e.loc[0, "elo_home"], e.loc[0, "elo_away"])   # nobody has played yet
    # A beat B in match 0, so A must be above B before their next games
    assert e.loc[3, "elo_away"] > e.loc[0, "elo_home"] > e.loc[2, "elo_home"]
    assert 0 < elo_expected(1500, 1500) < 1


def test_pi_rating_moves_toward_result():
    m = _toy_matches()
    pi = PiRatingTracker().run(m)
    assert pi.loc[0, "pi_home_venue"] == 0.0
    assert pi.loc[4, "pi_home_venue"] > 0            # A won at home earlier


def test_head_to_head_uses_only_previous_meetings():
    m = _toy_matches()
    h = head_to_head(m)
    assert h.loc[0, "h2h_n"] == 0                    # first ever meeting
    assert h.loc[6, "h2h_n"] == 1 and h.loc[6, "h2h_home_gd"] == 0   # C v A: only the 0-0 before


def test_rolling_form_excludes_current_match():
    m = _toy_matches()
    f = build_features(m)
    a_rows = f[f["HomeTeam"] == "A"].sort_values("Date")
    # A's first match: no history -> rolling stats NaN, season points 0
    first = a_rows.iloc[0]
    assert np.isnan(first["h_pts_r5"]) and first["h_season_pts"] == 0
    # A's home game on 26 Aug: A had won twice before -> 3.0 ppg, 6 points
    third = f[(f["HomeTeam"] == "A") & (f["Date"] == "2023-08-26")].iloc[0]
    assert third["h_pts_r5"] == 3.0 and third["h_season_pts"] == 6 and third["h_games_played"] == 2


def test_no_feature_is_a_function_of_the_result():
    m = _toy_matches()
    f = build_features(m)
    cols = feature_columns(f)
    assert "FTHG" not in cols and "FTR" not in cols and "target" not in cols
    # Flip one result: features of *that* match must not change
    m2 = m.copy(); m2.loc[4, ["FTHG", "FTAG"]] = [5, 0]
    m2["FTR"] = np.where(m2.FTHG > m2.FTAG, "H", np.where(m2.FTHG < m2.FTAG, "A", "D"))
    f2 = build_features(m2)
    pd.testing.assert_series_equal(f.loc[4, cols].astype(float), f2.loc[4, cols].astype(float), check_names=False)


def test_unplayed_fixture_gets_features_without_result():
    m = _toy_matches()
    future = pd.DataFrame({"match_id": [99], "Date": [pd.Timestamp("2023-09-09")], "HomeTeam": ["A"],
                           "AwayTeam": ["D"], "Season": [2023]})
    combined = pd.concat([m, future], ignore_index=True)
    f = build_features(combined)
    row = f[f["match_id"] == 99].iloc[0]
    assert np.isnan(row["target"])
    assert row["h_games_played"] == 4 and row["elo_home"] > 0 and not np.isnan(row["poisv_pH"])
