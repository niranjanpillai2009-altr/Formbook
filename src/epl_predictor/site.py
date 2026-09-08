"""Build the data behind the web app and render it into a single HTML page.

    python -m epl_predictor.site            # -> site/index.html (+ site/data.json)

Everything the page shows is computed here from the match history, the
current season's results and fixtures (live.py), and the trained model:

* forecasts for every remaining fixture (each scored as the next match of
  both sides given all played matches) and the model's pre-match call on
  every match already played this season;
* the live table plus Monte-Carlo season projections (expected points,
  title / top-four / relegation probabilities);
* a running scorecard of the model against simple baselines;
* per-team ratings, form and Elo history; a full 20x20 matchup matrix.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import config
from .features import build_features, next_match_features
from .live import get_live_matches
from .metrics import evaluate_probs
from .predict import load_bundle

log = logging.getLogger(__name__)

SITE_DIR = config.PROJECT_ROOT / "site"
TEMPLATE = Path(__file__).with_name("templates") / "site.html"

SHORT = {"Man United": "Man Utd", "Nott'm Forest": "Forest", "Crystal Palace": "Palace",
         "Sheffield United": "Sheff Utd", "Sheffield Weds": "Sheff Wed", "West Brom": "West Brom"}
CODES = {"Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU", "Brentford": "BRE", "Brighton": "BHA",
         "Burnley": "BUR", "Chelsea": "CHE", "Coventry": "COV", "Crystal Palace": "CRY", "Everton": "EVE",
         "Fulham": "FUL", "Hull": "HUL", "Ipswich": "IPS", "Leeds": "LEE", "Leicester": "LEI", "Liverpool": "LIV",
         "Luton": "LUT", "Man City": "MCI", "Man United": "MUN", "Newcastle": "NEW", "Norwich": "NOR",
         "Nott'm Forest": "NFO", "Sheffield United": "SHU", "Southampton": "SOU", "Sunderland": "SUN",
         "Tottenham": "TOT", "Watford": "WAT", "West Brom": "WBA", "West Ham": "WHU", "Wolves": "WOL",
         "Middlesbrough": "MID", "Stoke": "STK", "Swansea": "SWA", "Cardiff": "CAR", "Huddersfield": "HUD",
         "QPR": "QPR", "Reading": "RDG", "Wigan": "WIG", "Blackburn": "BLB", "Bolton": "BOL", "Derby": "DER"}

N_SIMS = 20_000
WIN_MARGIN_P = np.array([0.52, 0.28, 0.13, 0.07])       # goal margin of a win: 1,2,3,4


def _code(team: str) -> str:
    return CODES.get(team, team[:3].upper())


def _season_label(start: int) -> str:
    return f"{start}/{(start + 1) % 100:02d}"


# ---------------------------------------------------------------------------
# baselines for the scorecard
# ---------------------------------------------------------------------------
def fit_elo_baseline(hist: pd.DataFrame) -> LogisticRegression:
    d = hist[(hist["Season"] >= config.TRAIN_FROM_SEASON) & hist["target"].notna()]
    return LogisticRegression(max_iter=2000).fit(d[["elo_diff"]], d["target"].astype(int))


def baseline_probs(rows: pd.DataFrame, elo_model: LogisticRegression, prior: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "home": np.tile(prior, (len(rows), 1)),
        "elo": elo_model.predict_proba(rows[["elo_diff"]]),
        "poisson": rows[["poisv_pH", "poisv_pD", "poisv_pA"]].to_numpy(),
    }


# ---------------------------------------------------------------------------
# season simulation
# ---------------------------------------------------------------------------
def simulate_season(teams: list[str], table: pd.DataFrame, remaining: pd.DataFrame,
                    n_sims: int = N_SIMS, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    pts = np.tile(table.set_index("team").loc[teams, "pts"].to_numpy(dtype=float), (n_sims, 1))
    gd = np.tile(table.set_index("team").loc[teams, "gd"].to_numpy(dtype=float), (n_sims, 1))
    if len(remaining):
        p = remaining[["pH", "pD", "pA"]].to_numpy()
        hi = remaining["HomeTeam"].map(idx).to_numpy()
        ai = remaining["AwayTeam"].map(idx).to_numpy()
        u = rng.random((n_sims, len(remaining)))
        home_win = u < p[:, 0]
        draw = (~home_win) & (u < p[:, 0] + p[:, 1])
        away_win = ~(home_win | draw)
        margin = rng.choice(np.arange(1, 5), size=(n_sims, len(remaining)), p=WIN_MARGIN_P)
        for j in range(len(remaining)):
            h, a = hi[j], ai[j]
            pts[:, h] += 3 * home_win[:, j] + draw[:, j]
            pts[:, a] += 3 * away_win[:, j] + draw[:, j]
            gd[:, h] += margin[:, j] * (home_win[:, j].astype(int) - away_win[:, j].astype(int))
            gd[:, a] -= margin[:, j] * (home_win[:, j].astype(int) - away_win[:, j].astype(int))
    key = pts * 1000 + gd + rng.random((n_sims, n)) * 0.01          # tiny noise breaks exact ties
    order = np.argsort(-key, axis=1)
    pos = np.empty_like(order)
    rows = np.arange(n_sims)[:, None]
    pos[rows, order] = np.arange(1, n + 1)
    out = pd.DataFrame({
        "team": teams,
        "proj_pts": pts.mean(axis=0), "proj_pos": pos.mean(axis=0),
        "p_title": (pos == 1).mean(axis=0), "p_top4": (pos <= 4).mean(axis=0),
        "p_top6": (pos <= 6).mean(axis=0), "p_rel": (pos >= n - 2).mean(axis=0),
        "pts_p10": np.percentile(pts, 10, axis=0), "pts_p90": np.percentile(pts, 90, axis=0),
    })
    return out


# ---------------------------------------------------------------------------
def standings(played: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    rows = {t: dict(team=t, p=0, w=0, d=0, l=0, gf=0, ga=0, form=[]) for t in teams}
    for _, m in played.sort_values("Date").iterrows():
        h, a, hg, ag = m["HomeTeam"], m["AwayTeam"], int(m["FTHG"]), int(m["FTAG"])
        for t, gf, ga in ((h, hg, ag), (a, ag, hg)):
            r = rows[t]; r["p"] += 1; r["gf"] += gf; r["ga"] += ga
            res = "W" if gf > ga else "D" if gf == ga else "L"
            r["w" if res == "W" else "d" if res == "D" else "l"] += 1
            r["form"].append(res)
    df = pd.DataFrame(rows.values())
    df["gd"] = df["gf"] - df["ga"]
    df["pts"] = df["w"] * 3 + df["d"]
    df["form"] = df["form"].map(lambda f: "".join(f[-5:]))
    df = df.sort_values(["pts", "gd", "gf", "team"], ascending=[False, False, False, True]).reset_index(drop=True)
    df["pos"] = np.arange(1, len(df) + 1)
    return df


def elo_histories(hist: pd.DataFrame, teams: list[str], since: str = "2018-07-01") -> dict[str, list]:
    long = pd.concat([
        hist[["Date", "HomeTeam", "elo_home"]].rename(columns={"HomeTeam": "team", "elo_home": "elo"}),
        hist[["Date", "AwayTeam", "elo_away"]].rename(columns={"AwayTeam": "team", "elo_away": "elo"}),
    ])
    long = long[long["Date"] >= since]
    out = {}
    for t in teams:
        d = long[long["team"] == t].set_index("Date")["elo"].resample("MS").mean().dropna()
        out[t] = [[i.strftime("%Y-%m"), round(float(v), 1)] for i, v in d.items()]
    return out


# ---------------------------------------------------------------------------
def build_site_data(refresh_data: bool = True) -> dict:
    bundle = load_bundle()
    feats = bundle["features"]
    model = bundle["model"]

    matches, schedule = get_live_matches(refresh_data=refresh_data)
    season = int(schedule["Season"].iloc[0])
    teams = sorted(set(schedule["HomeTeam"]) | set(schedule["AwayTeam"]))
    played_sched = schedule[schedule["played"]]
    unplayed = schedule[~schedule["played"]].copy()
    last_date = matches["Date"].max()
    log.info("season %s: %d played, %d to play; history through %s",
             _season_label(season), len(played_sched), len(unplayed), last_date.date())

    # ---- pre-match features for every played match (history + this season)
    hist = build_features(matches)
    hist_probs = model.predict_proba(hist[feats])
    elo_model = fit_elo_baseline(hist)
    prior = np.bincount(hist.loc[hist["Season"] < season, "target"].dropna().astype(int), minlength=3) / \
        hist.loc[hist["Season"] < season, "target"].notna().sum()

    # ---- forecasts for every remaining fixture + the full matchup matrix
    next_round = int(unplayed["Round"].min()) if len(unplayed) else None
    anchor = unplayed["Date"].min() if len(unplayed) else last_date + pd.Timedelta(days=7)
    matrix_fx = pd.DataFrame([(h, a, anchor) for h in teams for a in teams if h != a],
                             columns=["HomeTeam", "AwayTeam", "Date"])
    fx_all = pd.concat([unplayed[["HomeTeam", "AwayTeam", "Date"]], matrix_fx], ignore_index=True)
    fx_feats = next_match_features(matches, fx_all, season_teams=teams)
    fx_probs = model.predict_proba(fx_feats[feats])
    fx_feats[["pH", "pD", "pA"]] = fx_probs
    n_up = len(unplayed)
    up = fx_feats.iloc[:n_up].reset_index(drop=True)
    mx = fx_feats.iloc[n_up:].reset_index(drop=True)

    # ---- rounds: played (with the model's pre-match call) and upcoming
    this = hist[hist["Season"] == season].reset_index()
    this_probs = hist_probs[hist["Season"].to_numpy() == season]
    base = baseline_probs(this, elo_model, prior)
    played_lookup = {}
    for i, r in this.iterrows():
        played_lookup[(r["HomeTeam"], r["AwayTeam"])] = (i, this_probs[i])

    rounds = {}
    for _, f in schedule.iterrows():
        rd = int(f["Round"])
        entry = {"home": f["HomeTeam"], "away": f["AwayTeam"], "date": f["Date"].strftime("%Y-%m-%d"),
                 "time": f["Time"], "played": bool(f["played"])}
        if f["played"] and (f["HomeTeam"], f["AwayTeam"]) in played_lookup:
            i, p = played_lookup[(f["HomeTeam"], f["AwayTeam"])]
            row = this.iloc[i]
            entry.update(hg=int(row["FTHG"]), ag=int(row["FTAG"]), result=row["FTR"],
                         p=[round(float(x), 4) for x in p], pick=config.CLASSES[int(np.argmax(p))],
                         correct=bool(config.CLASSES[int(np.argmax(p))] == row["FTR"]),
                         p_result=round(float(p[config.CLASS_TO_INT[row["FTR"]]]), 4),
                         xg=[round(float(row["poisv_exp_hg"]), 2), round(float(row["poisv_exp_ag"]), 2)],
                         elo=[round(float(row["elo_home"])), round(float(row["elo_away"]))])
        elif f["played"]:
            entry.update(hg=int(f["FTHG"]), ag=int(f["FTAG"]))
        else:
            j = unplayed.index.get_loc(f.name)
            row = up.iloc[j]
            p = row[["pH", "pD", "pA"]].to_numpy(dtype=float)
            entry.update(p=[round(float(x), 4) for x in p], pick=config.CLASSES[int(np.argmax(p))],
                         xg=[round(float(row["poisv_exp_hg"]), 2), round(float(row["poisv_exp_ag"]), 2)],
                         elo=[round(float(row["elo_home"])), round(float(row["elo_away"]))],
                         pending=bool(f["Date"] < pd.Timestamp(dt.date.today())))
        rounds.setdefault(rd, []).append(entry)
    rounds_list = [{"round": rd, "fixtures": fx} for rd, fx in sorted(rounds.items())]

    # ---- table and projections
    table = standings(played_sched, teams)
    remaining = up[["HomeTeam", "AwayTeam", "pH", "pD", "pA"]]
    proj = simulate_season(teams, table, remaining)
    table = table.merge(proj, on="team")

    # ---- scorecard (this season, matches the model has been scored on)
    scored = this[this["target"].notna()]
    y = scored["target"].astype(int).to_numpy()
    sc = {"n": int(len(scored)), "season": _season_label(season)}
    if len(scored):
        r4 = lambda d: {k: round(float(v), 4) for k, v in d.items()}  # noqa: E731
        sc["model"] = r4(evaluate_probs(this_probs, y))
        sc["baselines"] = {k: r4(evaluate_probs(v, y)) for k, v in base.items()}
        # cumulative by round
        by_round = []
        rounds_played = sorted(set(int(r) for r in played_sched["Round"]))
        rd_of = {(h, a): int(r) for h, a, r in zip(played_sched["HomeTeam"], played_sched["AwayTeam"], played_sched["Round"])}
        r_idx = np.array([rd_of.get((h, a), 0) for h, a in zip(scored["HomeTeam"], scored["AwayTeam"])])
        for rd in rounds_played:
            m = r_idx <= rd
            if m.sum() == 0:
                continue
            e = {"round": rd, "n": int(m.sum()),
                 "model": round(evaluate_probs(this_probs[m], y[m])["accuracy"], 4),
                 "model_ll": round(evaluate_probs(this_probs[m], y[m])["log_loss"], 4)}
            for k, v in base.items():
                e[k] = round(evaluate_probs(v[m], y[m])["accuracy"], 4)
            by_round.append(e)
        sc["by_round"] = by_round
        # calibration bins for the predicted-outcome probability
        conf = this_probs.max(axis=1); pred = this_probs.argmax(axis=1)
        bins = []
        for lo, hi in [(0.3, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 1.01)]:
            m = (conf >= lo) & (conf < hi)
            if m.sum():
                bins.append({"lo": lo, "hi": min(hi, 1.0), "n": int(m.sum()), "acc": round(float((pred[m] == y[m]).mean()), 4),
                             "conf": round(float(conf[m].mean()), 4)})
        sc["confidence_bins"] = bins
        # standout calls: most confident correct, and biggest surprises
        p_res = this_probs[np.arange(len(y)), y]
        order = np.argsort(p_res)
        def _call(i):
            r = scored.iloc[i]
            return {"home": r["HomeTeam"], "away": r["AwayTeam"], "hg": int(r["FTHG"]), "ag": int(r["FTAG"]),
                    "date": r["Date"].strftime("%Y-%m-%d"), "p": [round(float(x), 3) for x in this_probs[i]],
                    "p_result": round(float(p_res[i]), 3)}
        sc["surprises"] = [_call(i) for i in order[:3]]
        sc["best_calls"] = [_call(i) for i in order[::-1][:3]]

    # ---- teams
    elo_hist = elo_histories(hist, teams)
    team_state = {}
    for t in teams:
        row_h = mx[mx["HomeTeam"] == t].iloc[0]         # team-side features are opponent-independent
        row_a = mx[mx["AwayTeam"] == t].iloc[0]
        tb = table[table["team"] == t].iloc[0]
        nxt = [f for r in rounds_list for f in r["fixtures"] if not f["played"] and t in (f["home"], f["away"])][:5]
        team_state[t] = {
            "code": _code(t), "short": SHORT.get(t, t),
            "elo": round(float(row_h["elo_home"])), "pelo": round(float(row_h["pelo_home"])),
            "pi_home": round(float(row_h["pi_home_venue"]), 3), "pi_away": round(float(row_a["pi_away_venue"]), 3),
            "att": round(float(row_h["h_gf_ewm"]), 2), "def": round(float(row_h["h_ga_ewm"]), 2),
            "sot_for": None if pd.isna(row_h["h_stf_ewm"]) else round(float(row_h["h_stf_ewm"]), 1),
            "sot_against": None if pd.isna(row_h["h_sta_ewm"]) else round(float(row_h["h_sta_ewm"]), 1),
            "ppg_r10": round(float(row_h["h_pts_r10"]), 2), "gd_r10": round(float(row_h["h_gd_r10"]), 2),
            "home_ppg": round(float(row_h["h_venue_pts_ewm"]), 2), "away_ppg": round(float(row_a["a_venue_pts_ewm"]), 2),
            "promoted": int(row_h["h_promoted"]), "prev_pos": int(row_h["h_prev_pos"]),
            "streak_unbeaten": int(row_h["h_streak_unbeaten"]), "streak_winless": int(row_h["h_streak_winless"]),
            "pos": int(tb["pos"]), "pts": int(tb["pts"]), "played": int(tb["p"]), "form": tb["form"],
            "proj_pts": round(float(tb["proj_pts"]), 1), "p_title": round(float(tb["p_title"]), 4),
            "p_top4": round(float(tb["p_top4"]), 4), "p_rel": round(float(tb["p_rel"]), 4),
            "elo_history": elo_hist[t], "next": nxt,
        }
    elo_rank = sorted(teams, key=lambda t: -team_state[t]["elo"])
    for i, t in enumerate(elo_rank):
        team_state[t]["elo_rank"] = i + 1

    # ---- matchup matrix (home index x away index)
    ti = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    P = [[None] * n for _ in range(n)]
    XG = [[None] * n for _ in range(n)]
    for _, r in mx.iterrows():
        i, j = ti[r["HomeTeam"]], ti[r["AwayTeam"]]
        P[i][j] = [round(float(r["pH"]), 4), round(float(r["pD"]), 4), round(float(r["pA"]), 4)]
        XG[i][j] = [round(float(r["poisv_exp_hg"]), 2), round(float(r["poisv_exp_ag"]), 2)]

    data = {
        "meta": {
            "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "season": season, "season_label": _season_label(season),
            "data_through": last_date.strftime("%Y-%m-%d"),
            "played": int(len(played_sched)), "total": int(len(schedule)),
            "next_round": next_round,
            "model": {"name": bundle["model_name"], "trained_through": bundle["trained_through"],
                      "n_features": len(feats),
                      "cv_accuracy": round(float(bundle["cv_summary"]["accuracy"][bundle["model_name"]]), 4),
                      "cv_log_loss": round(float(bundle["cv_summary"]["log_loss"][bundle["model_name"]]), 4),
                      "holdout_accuracy": round(float(bundle["holdout"]["accuracy"][bundle["model_name"]]), 4),
                      "cv_prior_accuracy": round(float(bundle["cv_summary"]["accuracy"]["prior"]), 4),
                      "cv_elo_accuracy": round(float(bundle["cv_summary"]["accuracy"]["elo_logit"]), 4)},
            "history_matches": int(len(matches)), "history_from": int(matches["Season"].min()),
        },
        "teams": team_state,
        "team_order": teams,
        "rounds": rounds_list,
        "standings": [
            {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (np.integer,)) else v))
             for k, v in r.items()} for r in table.to_dict(orient="records")
        ],
        "scorecard": sc,
        "matrix": {"teams": teams, "p": P, "xg": XG},
    }
    return data


def render(data: dict, template: Path = TEMPLATE, standalone: bool = False) -> str:
    """Fill the template with `data`.

    By default the output is a page *fragment* (title/style/body, no doctype)
    as the Claude artifact host expects. `standalone=True` wraps it in a
    complete HTML document for GitHub Pages or opening from disk.
    """
    html = template.read_text()
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    html = html.replace("__SITE_DATA__", payload)
    if not standalone:
        return html
    head, body = html.split("<!-- body -->", 1)
    return ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            f"{head}\n</head>\n<body>\n{body}\n</body>\n</html>\n")


def build_site(refresh_data: bool = True, out_dir: Path = SITE_DIR) -> Path:
    data = build_site_data(refresh_data=refresh_data)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.json").write_text(json.dumps(data, indent=1, ensure_ascii=False))
    out = out_dir / "index.html"
    out.write_text(render(data))
    log.info("wrote %s (%.1f KB)", out, out.stat().st_size / 1024)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    build_site(refresh_data="--offline" not in sys.argv)
