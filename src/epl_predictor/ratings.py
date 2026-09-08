"""Sequential team-rating systems: Elo (with margin of victory) and pi-ratings.

Both are computed by walking through matches in chronological order and
recording the *pre-match* ratings, so they can never see the result of the
match they describe (no leakage).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------
def elo_expected(home_elo: float, away_elo: float, home_adv: float = config.ELO_HOME_ADV) -> float:
    """Probability-like expectation that the home side 'wins' (draw = half)."""
    return 1.0 / (1.0 + 10 ** ((away_elo - home_elo - home_adv) / 400.0))


def _mov_multiplier(goal_diff: int, elo_diff_winner: float) -> float:
    """FiveThirtyEight-style margin-of-victory multiplier.

    Bigger wins move ratings more, but with diminishing returns, and blowouts
    by already-stronger teams are discounted (autocorrelation correction).
    """
    return math.log(abs(goal_diff) + 1.0) * (2.2 / (elo_diff_winner * 0.001 + 2.2))


@dataclass
class EloTracker:
    k: float = config.ELO_K
    home_adv: float = config.ELO_HOME_ADV
    start: float = config.ELO_START
    promoted: float = config.ELO_PROMOTED_RATING
    season_regression: float = config.ELO_SEASON_REGRESSION
    ratings: dict[str, float] = field(default_factory=dict)
    last_season: dict[str, int] = field(default_factory=dict)
    _current_season: int | None = None

    def _new_season(self, season: int, teams: set[str]) -> None:
        """Regress ratings toward the mean and seed newly promoted teams."""
        for t in teams:
            if t in self.ratings and self.last_season.get(t) == season - 1:
                self.ratings[t] = self.ratings[t] + self.season_regression * (self.start - self.ratings[t])
            else:
                # Newly promoted (or returning after a gap): below-average start.
                self.ratings[t] = self.promoted
        # Keep the league mean anchored at `start` so ratings are comparable
        # across eras (only differences matter for prediction anyway). Skipped
        # when only a few teams are known (e.g. predicting the first fixtures
        # of a season that is not in the data yet).
        if len(teams) >= 10:
            mean = np.mean([self.ratings[t] for t in teams])
            for t in teams:
                self.ratings[t] += self.start - mean
        self._current_season = season

    def get(self, team: str) -> float:
        return self.ratings.get(team, self.promoted)

    def update(self, home: str, away: str, hg: int, ag: int) -> None:
        rh, ra = self.get(home), self.get(away)
        exp_h = elo_expected(rh, ra, self.home_adv)
        s_h = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        gd = hg - ag
        if gd == 0:
            mult = 1.0
        else:
            winner_diff = (rh + self.home_adv - ra) if gd > 0 else (ra - rh - self.home_adv)
            mult = _mov_multiplier(gd, winner_diff)
        delta = self.k * mult * (s_h - exp_h)
        self.ratings[home] = rh + delta
        self.ratings[away] = ra - delta

    def run(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Return pre-match Elo columns aligned with `matches` (chronological)."""
        out = np.zeros((len(matches), 3))
        for i, (season, home, away, hg, ag) in enumerate(
            zip(matches["Season"], matches["HomeTeam"], matches["AwayTeam"], matches["FTHG"], matches["FTAG"])
        ):
            if season != self._current_season:
                teams = set(matches.loc[matches["Season"] == season, "HomeTeam"])
                self._new_season(season, teams)
            rh, ra = self.get(home), self.get(away)
            out[i] = (rh, ra, elo_expected(rh, ra, self.home_adv))
            if not (pd.isna(hg) or pd.isna(ag)):        # skip fixtures not played yet
                self.update(home, away, int(hg), int(ag))
            self.last_season[home] = season
            self.last_season[away] = season
        return pd.DataFrame(out, columns=["elo_home", "elo_away", "elo_exp_home"], index=matches.index)


@dataclass
class PerformanceEloTracker:
    """Elo driven by *shot dominance* instead of the scoreline.

    The 'score' of a match for the home side is its share of shots on target,
    S = (SoT_home + 1) / (SoT_home + SoT_away + 2) (Laplace-smoothed), and the
    rating moves by K * (S - E). Goals are rare and noisy; shots on target are
    a far more repeatable measure of how good a team actually is, so this
    rating reacts to a team playing well before the results catch up.
    Matches without shot data (pre-2000) leave the ratings untouched.
    """
    k: float = config.PERF_ELO_K
    home_adv: float = config.ELO_HOME_ADV
    start: float = config.ELO_START
    promoted: float = config.ELO_PROMOTED_RATING
    season_regression: float = config.ELO_SEASON_REGRESSION
    ratings: dict[str, float] = field(default_factory=dict)
    last_season: dict[str, int] = field(default_factory=dict)
    _current_season: int | None = None

    _new_season = EloTracker._new_season
    get = EloTracker.get

    def run(self, matches: pd.DataFrame) -> pd.DataFrame:
        out = np.zeros((len(matches), 3))
        for i, (season, home, away, hst, ast) in enumerate(
            zip(matches["Season"], matches["HomeTeam"], matches["AwayTeam"], matches["HST"], matches["AST"])
        ):
            if season != self._current_season:
                teams = set(matches.loc[matches["Season"] == season, "HomeTeam"])
                self._new_season(season, teams)
            rh, ra = self.get(home), self.get(away)
            exp_h = elo_expected(rh, ra, self.home_adv)
            out[i] = (rh, ra, exp_h)
            if not (pd.isna(hst) or pd.isna(ast)):
                s_h = (float(hst) + 1.0) / (float(hst) + float(ast) + 2.0)
                delta = self.k * (s_h - exp_h) * 2.0     # x2: shares live in ~[0.3, 0.7]
                self.ratings[home] = rh + delta
                self.ratings[away] = ra - delta
            self.last_season[home] = season
            self.last_season[away] = season
        return pd.DataFrame(out, columns=["pelo_home", "pelo_away", "pelo_exp_home"], index=matches.index)


# ---------------------------------------------------------------------------
# Pi-ratings (Constantinou & Fenton, 2013)
# ---------------------------------------------------------------------------
def _pi_expected_gd(rating: float, b: float = 10.0, c: float = 3.0) -> float:
    """Map a pi-rating to an expected goal difference (sign-preserving)."""
    return math.copysign(b ** (abs(rating) / c) - 1.0, rating)


@dataclass
class PiRatingTracker:
    """Each team keeps a *home* rating and an *away* rating.

    The expected goal difference of a match is g(R_home_H) - g(R_away_A). After
    the match both teams move their venue-specific rating by lambda * psi(error)
    and their other-venue rating by gamma times that change.
    """
    lam: float = config.PI_LAMBDA
    gamma: float = config.PI_GAMMA
    c: float = 3.0
    home_r: dict[str, float] = field(default_factory=dict)
    away_r: dict[str, float] = field(default_factory=dict)

    def expected_gd(self, home: str, away: str) -> float:
        return _pi_expected_gd(self.home_r.get(home, 0.0), c=self.c) - _pi_expected_gd(
            self.away_r.get(away, 0.0), c=self.c
        )

    def update(self, home: str, away: str, hg: int, ag: int) -> None:
        exp = self.expected_gd(home, away)
        obs = hg - ag
        err = abs(obs - exp)
        psi = self.c * math.log10(1.0 + err)
        psi_h, psi_a = (psi, -psi) if obs > exp else (-psi, psi)

        old = self.home_r.get(home, 0.0)
        self.home_r[home] = old + psi_h * self.lam
        self.away_r[home] = self.away_r.get(home, 0.0) + (self.home_r[home] - old) * self.gamma

        old = self.away_r.get(away, 0.0)
        self.away_r[away] = old + psi_a * self.lam
        self.home_r[away] = self.home_r.get(away, 0.0) + (self.away_r[away] - old) * self.gamma

    def run(self, matches: pd.DataFrame) -> pd.DataFrame:
        out = np.zeros((len(matches), 5))
        for i, (home, away, hg, ag) in enumerate(
            zip(matches["HomeTeam"], matches["AwayTeam"], matches["FTHG"], matches["FTAG"])
        ):
            hh, ha = self.home_r.get(home, 0.0), self.away_r.get(home, 0.0)
            ah, aa = self.home_r.get(away, 0.0), self.away_r.get(away, 0.0)
            out[i] = (hh, (hh + ha) / 2, aa, (ah + aa) / 2, self.expected_gd(home, away))
            if not (pd.isna(hg) or pd.isna(ag)):
                self.update(home, away, int(hg), int(ag))
        return pd.DataFrame(
            out,
            columns=["pi_home_venue", "pi_home_overall", "pi_away_venue", "pi_away_overall", "pi_exp_gd"],
            index=matches.index,
        )
