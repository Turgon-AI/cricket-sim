"""How much of the match to match spread in ODI and T20I results can pre-match inputs explain?

A measurement, not a simulator. For each test year from --first-test-year to the last year in gold, fit on the ten
years before it and predict that year (walk forward), then pool. Two targets on the same clean matches the
simulator uses: the first-innings total, and whether the side batting first won.

Inputs are added in layers, so the gain of each layer shows:
  level    the format's mean first-innings total over the 365 days before the match, and the year
  elo      + the pre-match Elo gap (bat first minus bat second), the repo's own Elo
  venue    + how far above or below the format level first innings at this ground ran in the 8 years before,
             shrunk towards zero, the number of those innings, and whether either side is at home
  xi       + the playing XI of both sides from fact_player_match_form (point in time): batting runs and outs per
             ball of the top 7 batters, runs and wickets per ball of the top 6 bowlers, career and last 365 days,
             shrunk towards the format average, and total caps
  toss     + whether the toss winner batted first, and the toss decision
  weather  + the weather in the playing window on day 1 (observed, so this layer is a best case: a forecast would
             know less)
  conditions + added 2026-11-18: local start hour of day 1 (first ball, else Wikipedia's scheduled start), the weather over each
             innings' expected hours and how long after sunset the second innings starts (int_match_weather_daily,
             timed or scheduled windows only), how recently and how often the ground was used (dim_match), and
             the ground's altitude

Each layer is fitted with a linear model and a gradient-boosted model. Share of spread explained = 1 minus the
squared error over the squared distance from that test year's own mean, so missing a year's scoring level counts
against the model. Wins are scored with the Brier score against the training years' bat-first win rate.

Prototype outside the build; it reads gold and writes only to --out (a new folder).
"""

import argparse
import json
import subprocess
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cricket_sim.models.elo import pre_match_ratings
from cricket_sim.models.train import SEED
from cricket_sim.paths import GOLD

FAMILY = {"ODI": ("one_day", ["ODI"]), "T20I": ("t20", ["T20I", "IPL"])}  # venue history uses the whole family
VENUE_YEARS, VENUE_K = 8, 4  # venue history window and shrinkage (pseudo-innings at zero)
PLAYER_K = 300  # pseudo-balls at the format average when shrinking a side's rates
LAYERS = {
    "level": ["recent_level", "year"],
    "elo": ["elo_diff"],
    "venue": ["venue_dev", "venue_n", "bat_first_home", "bat_second_home"],
    "xi": [f"{s}_{f}" for s in ("bf", "bs") for f in
           ("bat_rpb", "bat_opb", "bat_rpb_365", "bowl_rpb", "bowl_wpb", "bowl_rpb_365", "caps")],
    "toss": ["toss_winner_bats_first", "toss_bat"],
    "weather": ["temp_c", "humidity", "dew_point_depression", "cloud", "wind", "rain_hours"],
    "conditions": ["start_hour", "inn1_temp", "inn1_humidity", "inn1_dew_dep", "inn2_temp", "inn2_humidity",
                   "inn2_dew_dep", "inn2_after_sunset_min", "venue_days_since", "venue_prior_14d",
                   "venue_prior_365d", "elevation_m"],
}


def gold(con, t):
    return f"read_parquet('{GOLD / t}.parquet')"


def first_innings(con) -> pd.DataFrame:
    """Every full-length first innings with no rain rule, all four segments, for level and venue history."""
    return con.execute(f"""
        SELECT m.match_id, m.segment, m.start_date, m.venue_id, i.runs AS total
        FROM {gold(con, 'dim_match')} m JOIN {gold(con, 'fact_innings')} i
          ON i.match_id = m.match_id AND i.innings = 1 AND NOT i.is_super_over
        WHERE m.method IS NULL AND m.passes_quality AND i.innings_length_status = 'full_known'
        ORDER BY m.start_date""").df()


def matches(con, segment) -> pd.DataFrame:
    """Clean matches (same rule as simulate_matches.py) with Elo, toss, home team and day-1 weather."""
    fixtures = con.execute(f"""SELECT match_id, start_date, team1, team2, winner, result
                               FROM {gold(con, 'dim_match')} WHERE segment = ?""", [segment]).fetchall()
    con.execute("CREATE OR REPLACE TEMP TABLE elo_df (match_id VARCHAR, team VARCHAR, elo_before DOUBLE, "
                "rated_before INTEGER)")
    con.executemany("INSERT INTO elo_df VALUES (?, ?, ?, ?)", pre_match_ratings(fixtures))
    df = con.execute(f"""
        WITH clean AS (
            SELECT m.match_id
            FROM {gold(con, 'dim_match')} m JOIN {gold(con, 'fact_innings')} i
              ON i.match_id = m.match_id AND NOT i.is_super_over AND i.innings <= 2
            WHERE m.segment = $seg AND m.method IS NULL AND m.passes_quality
              AND (m.winner IS NOT NULL OR m.result = 'tie')
            GROUP BY m.match_id
            HAVING bool_and(i.innings_length_status = 'full_known') AND bool_and(i.innings = 1 OR i.target_status = 'set'))
        SELECT m.match_id, m.start_date, year(m.start_date) AS year, m.venue_id, m.team1, m.team2,
            m.winner, m.result, m.toss_winner, m.toss_decision, v.home_team,
            i1.batting_team AS bat_first, i2.batting_team AS bat_second, i1.runs AS first_total,
            e1.elo_before - e2.elo_before AS elo_diff,
            w.window_temperature_c AS temp_c, w.window_humidity_pct AS humidity,
            w.window_dew_point_depression_c AS dew_point_depression, w.window_cloud_cover_pct AS cloud,
            w.window_wind_speed_kmh AS wind, w.window_hours_with_rain AS rain_hours,
            hour(pw.window_start_local) + minute(pw.window_start_local) / 60.0 AS start_hour,
            w.innings1_temperature_c AS inn1_temp, w.innings1_humidity_pct AS inn1_humidity,
            w.innings1_dew_point_depression_c AS inn1_dew_dep, w.innings2_temperature_c AS inn2_temp,
            w.innings2_humidity_pct AS inn2_humidity, w.innings2_dew_point_depression_c AS inn2_dew_dep,
            w.innings2_minutes_after_sunset AS inn2_after_sunset_min,
            least(m.venue_days_since_last_match, 365) AS venue_days_since, m.venue_matches_prior_14d AS venue_prior_14d,
            m.venue_matches_prior_365d AS venue_prior_365d, dv.elevation_m
        FROM {gold(con, 'dim_match')} m JOIN clean USING (match_id)
        LEFT JOIN {gold(con, 'int_match_play_windows')} pw ON pw.match_id = m.match_id AND pw.match_day_number = 1
            AND pw.window_source IN ('scheduled_start', 'ball_timestamps')
        LEFT JOIN {gold(con, 'dim_venue_version')} dv ON dv.venue_id = m.venue_id AND dv.venue_version = m.venue_version
        JOIN {gold(con, 'fact_innings')} i1 ON i1.match_id = m.match_id AND i1.innings = 1 AND NOT i1.is_super_over
        JOIN {gold(con, 'fact_innings')} i2 ON i2.match_id = m.match_id AND i2.innings = 2 AND NOT i2.is_super_over
        LEFT JOIN {gold(con, 'dim_venue_version')} v ON v.venue_id = m.venue_id AND v.venue_version = m.venue_version
        LEFT JOIN elo_df e1 ON e1.match_id = m.match_id AND e1.team = i1.batting_team
        LEFT JOIN elo_df e2 ON e2.match_id = m.match_id AND e2.team = i2.batting_team
        LEFT JOIN {gold(con, 'int_match_weather_daily')} w ON w.match_id = m.match_id AND w.match_day_number = 1
            AND w.weather_available
        ORDER BY m.start_date, m.match_id""", {"seg": segment}).df()
    df["bat_first_won"] = np.select([df.result == "tie", df.winner == df.bat_first], [0.5, 1.0], 0.0)
    df["toss_winner_bats_first"] = (df.toss_winner == df.bat_first).astype(float)
    df["toss_bat"] = (df.toss_decision == "bat").astype(float)
    df["bat_first_home"] = (df.home_team == df.bat_first).astype(float)
    df["bat_second_home"] = (df.home_team == df.bat_second).astype(float)
    return df


def add_level_and_venue(df, hist, segment):
    """recent_level: mean first-innings total of this segment in the 365 days before the match.
    venue_dev: shrunk mean of (total minus its own segment's recent level) at this ground, same format family,
    over the VENUE_YEARS before the match. Only earlier dates are used."""
    family = FAMILY[segment][1]
    hist = hist[hist.segment.isin(family)].copy()
    hist["start_date"] = pd.to_datetime(hist.start_date)
    hd = hist.start_date.to_numpy()
    levels = {}
    for seg in family:
        h = hist[hist.segment == seg]
        d, t = h.start_date.to_numpy(), h.total.to_numpy()
        levels[seg] = (d, t)

    def level_at(seg, date):
        d, t = levels[seg]
        sel = (d < date) & (d >= date - np.timedelta64(365, "D"))
        return t[sel].mean() if sel.any() else np.nan

    hist["level_then"] = [level_at(s, dd) for s, dd in zip(hist.segment, hist.start_date.to_numpy())]
    hist["dev"] = hist.total - hist.level_then
    dates = pd.to_datetime(df.start_date).to_numpy()
    df["recent_level"] = [level_at(segment, d) for d in dates]
    vd, vn = [], []
    for venue, d in zip(df.venue_id, dates):
        sel = ((hist.venue_id == venue).to_numpy() & (hd < d)
               & (hd >= d - np.timedelta64(int(365.25 * VENUE_YEARS), "D")) & hist.dev.notna().to_numpy())
        devs = hist.dev.to_numpy()[sel]
        vd.append(devs.sum() / (len(devs) + VENUE_K))
        vn.append(len(devs))
    df["venue_dev"], df["venue_n"] = vd, vn
    return df


def add_xi(con, df, segment):
    """Strength of each side's playing XI from fact_player_match_form (records before the match only)."""
    fam = FAMILY[segment][0]
    form = con.execute(f"""
        SELECT match_id, team, prior_matches AS caps,
            prior_bat_balls AS bb, prior_bat_runs AS br, prior_bat_outs AS bo,
            prior_365d_bat_balls AS bb365, prior_365d_bat_runs AS br365,
            prior_bowl_balls AS wb, prior_bowl_runs AS wr, prior_bowl_wickets AS ww,
            prior_365d_bowl_balls AS wb365, prior_365d_bowl_runs AS wr365
        FROM {gold(con, 'fact_player_match_form')} WHERE format_family = ?""", [fam]).df().fillna(0)
    # format averages from the players' own records, as the shrink target
    stats = con.execute(f"""SELECT sum(bat_runs)/sum(bat_balls), sum(bat_outs)/sum(bat_balls),
                               sum(bowl_wickets)/sum(bowl_balls)
                            FROM {gold(con, 'int_player_match_stats')} WHERE format_family = ?""", [fam]).fetchone()
    rpb, opb, wpb = stats
    rows = []
    for (mid, team), g in form.groupby(["match_id", "team"]):
        bat = g.nlargest(7, "bb")
        bowl = g.nlargest(6, "wb")
        k7, k6 = PLAYER_K * 7, PLAYER_K * 6
        rows.append({
            "match_id": mid, "team": team, "caps": g.caps.sum(),
            "bat_rpb": (bat.br.sum() + k7 * rpb) / (bat.bb.sum() + k7),
            "bat_opb": (bat.bo.sum() + k7 * opb) / (bat.bb.sum() + k7),
            "bat_rpb_365": (bat.br365.sum() + k7 * rpb) / (bat.bb365.sum() + k7),
            "bowl_rpb": (bowl.wr.sum() + k6 * rpb) / (bowl.wb.sum() + k6),
            "bowl_wpb": (bowl.ww.sum() + k6 * wpb) / (bowl.wb.sum() + k6),
            "bowl_rpb_365": (bowl.wr365.sum() + k6 * rpb) / (bowl.wb365.sum() + k6),
        })
    side = pd.DataFrame(rows)
    for pre, col in (("bf", "bat_first"), ("bs", "bat_second")):
        s = side.rename(columns={c: f"{pre}_{c}" for c in side.columns if c not in ("match_id", "team")})
        df = df.merge(s, left_on=["match_id", col], right_on=["match_id", "team"], how="left").drop(columns="team")
    return df


def regressors():
    return {
        "linear": lambda: make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=1.0)),
        "boosted": lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.04, max_leaf_nodes=8,
                                                         min_samples_leaf=25, l2_regularization=1.0,
                                                         random_state=SEED),
    }


def classifiers():
    return {
        "linear": lambda: make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
        "boosted": lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.04, max_leaf_nodes=8,
                                                          min_samples_leaf=25, l2_regularization=1.0,
                                                          random_state=SEED),
    }


def walk_forward(df, first_test_year):
    """One prediction per match in each test year, from models fitted on the ten years before it."""
    preds = []
    sets, cols = [], []
    for layer, feats in LAYERS.items():
        cols = cols + feats
        sets.append((layer, list(cols)))
    for year in range(first_test_year, int(df.year.max()) + 1):
        train = df[(df.year >= year - 10) & (df.year < year)]
        test = df[df.year == year]
        if test.empty:
            continue
        base_rate = train.bat_first_won.mean()
        out = test[["match_id", "year", "full_member", "first_total", "bat_first_won"]].copy()
        out["base_rate"] = base_rate
        wins = train[train.bat_first_won != 0.5]
        for layer, feats in sets:
            for name, make in regressors().items():
                out[f"total_{layer}_{name}"] = make().fit(train[feats], train.first_total).predict(test[feats])
            for name, make in classifiers().items():
                m = make().fit(wins[feats], wins.bat_first_won.astype(int))
                out[f"win_{layer}_{name}"] = m.predict_proba(test[feats])[:, 1]
        preds.append(out)
    return pd.concat(preds, ignore_index=True)


def share_explained(p, col):
    """1 - squared error / squared distance from each test year's own mean."""
    year_mean = p.groupby("year").first_total.transform("mean")
    return 1 - ((p.first_total - p[col]) ** 2).sum() / ((p.first_total - year_mean) ** 2).sum()


def brier(p, col):
    return ((p.bat_first_won - p[col]) ** 2).mean()


def score(p, rng, boot=300):
    rows = []
    idx = np.arange(len(p))
    draws = [rng.choice(idx, len(idx)) for _ in range(boot)]
    for layer in LAYERS:
        for name in ("linear", "boosted"):
            t, w = f"total_{layer}_{name}", f"win_{layer}_{name}"
            r2 = share_explained(p, t)
            r2_boot = np.array([share_explained(p.iloc[d], t) for d in draws])
            b = brier(p, w)
            b0 = brier(p, "base_rate")
            skill_boot = np.array([1 - brier(p.iloc[d], w) / brier(p.iloc[d], "base_rate") for d in draws])
            rows.append({"layer": layer, "model": name, "matches": len(p),
                         "score_share_explained": r2, "score_share_lo": np.quantile(r2_boot, 0.05),
                         "score_share_hi": np.quantile(r2_boot, 0.95),
                         "mean_abs_error_runs": (p.first_total - p[t]).abs().mean(),
                         "win_brier": b, "win_brier_base_rate": b0, "win_skill": 1 - b / b0,
                         "win_skill_lo": np.quantile(skill_boot, 0.05), "win_skill_hi": np.quantile(skill_boot, 0.95)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-test-year", type=int, default=2019)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    import yaml
    full = set(yaml.safe_load(open("config/scope.yml"))["full_members"])
    con = duckdb.connect()
    hist = first_innings(con)
    rng = np.random.default_rng(SEED)
    results, coverage = [], {}
    for segment in ("ODI", "T20I"):
        df = matches(con, segment)
        df = add_level_and_venue(df, hist, segment)
        df = add_xi(con, df, segment)
        df["full_member"] = df.team1.isin(full) & df.team2.isin(full)
        test = df[df.year >= args.first_test_year]
        coverage[segment] = {
            "clean_matches": len(df), "test_matches": len(test),
            "share_with_xi": float(test.bf_bat_rpb.notna().mean()),
            "share_with_weather": float(test.humidity.notna().mean()),
            "share_with_venue_history": float((test.venue_n > 0).mean()),
            "share_with_home_team": float(test.home_team.notna().mean()),
        }
        p = walk_forward(df, args.first_test_year)
        p.to_csv(args.out / f"predictions_{segment}.csv", index=False)
        subsets = {"all": p, "2025": p[p.year == 2025], "2026": p[p.year == 2026]}
        if segment == "ODI":
            subsets["full_members"] = p[p.full_member]
            subsets["full_members_2026"] = p[p.full_member & (p.year == 2026)]
        for name, s in subsets.items():
            for r in score(s, rng):
                results.append({"segment": segment, "test_set": name, **r})
    res = pd.DataFrame(results)
    res.to_csv(args.out / "explained_share.csv", index=False)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    build = json.load(open(GOLD / "run.json")).get("build_fingerprint")
    json.dump({"commit": commit, "build_fingerprint": build, "first_test_year": args.first_test_year,
               "coverage": coverage}, open(args.out / "summary.json", "w"), indent=2)
    pd.set_option("display.width", 250)
    print(json.dumps(coverage, indent=2))
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
