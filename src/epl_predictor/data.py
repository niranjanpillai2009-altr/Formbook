"""Download, load and clean Premier League match data.

Source: a GitHub mirror of football-data.co.uk (one CSV per season). The
mirror strips the bookmaker-odds columns, leaving the match facts:

    Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HTHG, HTAG, HTR, Referee,
    HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR

(FT = full time, HT = half time, H/A = home/away, G = goals, S = shots,
ST = shots on target, F = fouls, C = corners, Y/R = yellow/red cards).
"""
from __future__ import annotations

import datetime as dt
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from . import config

log = logging.getLogger(__name__)

STAT_COLS = ["HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]
KEEP_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
             "HTHG", "HTAG", "HTR", "Referee"] + STAT_COLS


def season_code(start_year: int) -> str:
    """1993 -> '9394', 2005 -> '0506', 2023 -> '2324'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    return f"{start_year}/{(start_year + 1) % 100:02d}"


def _current_season_start() -> int:
    """The season that is (or was most recently) in progress, by calendar."""
    today = dt.date.today()
    return today.year if today.month >= 7 else today.year - 1


def download_season(start_year: int, force: bool = False) -> Path | None:
    """Download one season's CSV into data/raw. Returns the path or None on 404."""
    dest = config.RAW_DIR / f"season-{season_code(start_year)}.csv"
    if dest.exists() and not force:
        return dest
    url = config.DATA_URL_TEMPLATE.format(code=season_code(start_year))
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
            if len(data) < 100:                      # empty / placeholder file
                return dest if dest.exists() else None
            dest.write_bytes(data)
            log.info("downloaded %s", dest.name)
            return dest
        except urllib.error.HTTPError as e:
            if e.code == 404:
                log.info("season %s not available (404)", season_label(start_year))
                return dest if dest.exists() else None
            if e.code == 429 and attempt < 2:        # rate limited: back off and retry
                time.sleep(5 * (attempt + 1))
                continue
            if dest.exists():
                log.warning("could not refresh %s (%s); using cached copy", dest.name, e)
                return dest
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            if dest.exists():
                log.warning("could not refresh %s (%s); using cached copy", dest.name, e)
                return dest
            if attempt == 2:
                raise
            time.sleep(3)
    return dest if dest.exists() else None


def download_all(force_latest: bool = True) -> list[Path]:
    """Download every season from FIRST_SEASON to the current one.

    The most recent season(s) are re-downloaded each run so that new
    fixtures are picked up as the mirror is updated.
    """
    paths = []
    latest = _current_season_start()
    for year in range(config.FIRST_SEASON, latest + 1):
        force = force_latest and year >= latest - 1
        p = download_season(year, force=force)
        if p is not None:
            paths.append(p)
    return paths


def _parse_dates(s: pd.Series) -> pd.Series:
    """football-data files mix dd/mm/yy, dd/mm/yyyy and ISO formats."""
    out = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        mask = out.isna()
        if mask.any():
            out[mask] = pd.to_datetime(s[mask], format=fmt, errors="coerce")
    return out


def load_raw_season(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    df.columns = [c.strip() for c in df.columns]
    for c in KEEP_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[KEEP_COLS].copy()
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df["Date"] = _parse_dates(df["Date"].astype(str).str.strip())
    df = df.dropna(subset=["Date"])
    code = path.stem.replace("season-", "")
    start = int(code[:2])
    df["Season"] = 1900 + start if start >= 90 else 2000 + start
    return df


def build_matches(paths: list[Path] | None = None) -> pd.DataFrame:
    """Combine all seasons into one clean, chronologically ordered table."""
    if paths is None:
        paths = sorted(config.RAW_DIR.glob("season-*.csv"))
    return build_matches_from_frames([load_raw_season(p) for p in paths])


def refresh_latest(df: pd.DataFrame) -> pd.DataFrame:
    """Re-download the two most recent season files and splice them into `df`.

    Two requests instead of thirty-three: everything older is final.
    """
    latest = _current_season_start()
    frames = []
    for year in (latest - 1, latest):
        p = download_season(year, force=True)
        if p is not None:
            frames.append(load_raw_season(p))
    if not frames:
        return df
    new = build_matches_from_frames(frames)
    keep = df[~df["Season"].isin(new["Season"].unique())]
    out = pd.concat([keep, new], ignore_index=True)
    out = out.sort_values(["Date", "Season"], kind="stable").reset_index(drop=True)
    out["match_id"] = range(len(out))
    return out


def build_matches_from_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(frames, ignore_index=True)
    for c in ["FTHG", "FTAG", "HTHG", "HTAG"] + STAT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    df["FTR"] = pd.Series(
        ["H" if h > a else "A" if a > h else "D" for h, a in zip(df["FTHG"], df["FTAG"])], index=df.index)
    df["HomeTeam"] = df["HomeTeam"].str.strip()
    df["AwayTeam"] = df["AwayTeam"].str.strip()
    df = df.sort_values(["Date", "Season"], kind="stable").reset_index(drop=True)
    df.insert(0, "match_id", range(len(df)))
    return df.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)


def get_matches(refresh: bool = False) -> pd.DataFrame:
    """Return the clean matches table.

    First call: download every season and build it. Later calls read the
    stored table; with `refresh=True` only the two newest season files are
    re-downloaded and spliced in.
    """
    out = config.PROCESSED_DIR / "matches.parquet"
    csv = config.PROCESSED_DIR / "matches.csv"
    if not out.exists() and csv.exists():            # restored from a bundle: CSV only
        df = pd.read_csv(csv, parse_dates=["Date"])
        df.to_parquet(out, index=False)
    if out.exists():
        df = pd.read_parquet(out)
        if not refresh:
            return df
        try:
            df = refresh_latest(df)
        except Exception as e:                       # offline: keep what we have
            log.warning("could not refresh the latest seasons (%s); using stored data", e)
            return df
    else:
        paths = download_all()
        df = build_matches(paths)
    df.to_parquet(out, index=False)
    df.to_csv(config.PROCESSED_DIR / "matches.csv", index=False)
    log.info("saved %d matches across %d seasons", len(df), df["Season"].nunique())
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    m = get_matches(refresh=True)
    print(m.groupby("Season").size().to_string())
    print(m.tail())
