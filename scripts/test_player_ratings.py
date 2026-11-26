"""Pass/fail test: does knowing the batter on strike, the bowler and the ground make the ball model better on
matches it has not seen? Step 2 of the player-level plan (2026-11-19). A prototype that reads gold; it is not part
of the build, its contracts or the export.

Ratings (as of each match date, from earlier dates only):
- Every ball of the format family counts (T20: T20Is, the IPL and the nine leagues; ODI: ODIs), clean or not,
  first two innings, no super overs.
- Each ball has an expected chance of each outcome class (dot, 1, 2, 3, 4, 6, wicket, extra): the share of that
  class in the same innings, over band and wickets band, counted on the family's balls of earlier calendar years
  only. Balls of the family's first year have no expectation and are not counted.
- For a batter, a bowler and a ground, observed and expected counts per class are summed over earlier dates, each
  ball weighted 0.5 ** (age / HALF_LIFE_YEARS). Matches on the same date do not see each other.
- Rating per class = (observed + k p) / (expected + k p), p the family's class share before 2019, k = PSEUDO balls
  (PSEUDO_WICKET for wickets): a newcomer sits at 1, the average. The model gets log(rating) per class and the
  log of the weighted expected balls (how much evidence there is). All settings were fixed before any scoring.
- No league strength term in this version: a ball in a weak league counts the same as one in the IPL.

Test: walk forward, each year 2019-2026 scored by models fitted on the ten years before 1 January of that year
(T20: T20I and league balls together, as the simulator trains; ODI: ODI balls). Three ball models, the same
gradient-boosted classifier and settings as the simulator's BallModel:
  base          the simulator's inputs today (match state, Elo gap, venue and XI averages, in-play inputs)
  ratings       base + batter, bowler and ground ratings
  ratings_no_xi ratings without the six XI-average inputs (would the ratings replace them?)
Score: mean log loss per ball on the test year's clean balls, pooled over years, with a bootstrap band from
resampling whole matches (paired: the same matches for every model). Pass: the 90% band of ratings minus base
is below zero.

Checks: grounds' six rates (does a ground's six rating as of 1 January predict its six rate that year?) and MS
Dhoni's ODI death-overs scoring by year.

Writes (in --out, which must not exist): per_match.csv (log-loss sums per match and model), summary.json,
checks.json, and the rating tables ratings_<family>_<entity>.parquet.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import measure_explained_share as mes
import simulate_matches as sm
from cricket_sim.models.dataset import CLASSES
from cricket_sim.models.train import OUTCOME_FEATURES, OUTCOME_SEASONS, SEED
from cricket_sim.paths import GOLD
from player_ratings import (ENTITIES, HALF_LIFE_YEARS, NC, PSEUDO, PSEUDO_WICKET, XI, all_balls, expected_probs,
                            rating_cols, ratings)

CLASS_RUNS = np.array([0, 1, 2, 3, 4, 6, 0, 1], float)  # rough runs per class, only for the Dhoni check


def model_balls(con, family: str) -> pd.DataFrame:
    """The simulator's training balls (clean matches, its inputs), with a segment column."""
    if family == "ODI":
        b, _ = sm.load(con, "ODI")
        return b.assign(segment="ODI", is_international=1.0)
    parts = [sm.load(con, "T20I")[0].assign(segment="T20I", is_international=1.0)]
    for seg in sm.T20_LEAGUES:
        mes.FAMILY[seg] = ("t20", ["T20I", "IPL", seg])
        parts.append(sm.load(con, seg)[0].assign(segment=seg, is_international=0.0))
    return pd.concat(parts, ignore_index=True)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    y = train.outcome_class.map(CLASSES.index).to_numpy()
    gbm = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=2000,
                                         l2_regularization=1.0, early_stopping=False, random_state=SEED)
    gbm.fit(train[features], y)
    p = np.zeros((len(test), NC))
    p[:, gbm.classes_] = gbm.predict_proba(test[features])
    yt = test.outcome_class.map(CLASSES.index).to_numpy()
    return -np.log(np.clip(p[np.arange(len(test)), yt], 1e-12, None))


def band(per_match: pd.DataFrame, a: str, b: str, rng, reps: int = 2000) -> dict:
    """Mean per-ball log loss of b minus a, with a 90% band from resampling matches."""
    da, db, n = per_match[a].to_numpy(), per_match[b].to_numpy(), per_match.balls.to_numpy()
    point = (db.sum() - da.sum()) / n.sum()
    idx = rng.integers(0, len(n), size=(reps, len(n)))
    boot = ((db[idx] - da[idx]).sum(1)) / n[idx].sum(1)
    lo, hi = np.percentile(boot, [5, 95])
    return {"diff": round(float(point), 5), "band_90": [round(float(lo), 5), round(float(hi), 5)],
            "base_log_loss": round(float(da.sum() / n.sum()), 5), "matches": int(len(n)), "balls": int(n.sum())}


def checks(t20: pd.DataFrame, t20_venue: pd.DataFrame, odi: pd.DataFrame, odi_exp: np.ndarray) -> dict:
    out = {}
    # grounds' six rates: rating as of each ground's first match of the year vs that year's observed six ratio
    rows = []
    for y in range(2016, 2027):
        yr = t20[t20.y == y]
        obs = yr.groupby("venue_id").agg(balls=("cls", "size"), o6=("cls", lambda s: (s == CLASSES.index("6")).sum()),
                                         e6=("e6", "sum"))
        first = t20_venue[t20_venue.start_date.dt.year == y].sort_values("start_date").groupby("venue_id").head(1)
        j = obs.join(first.set_index("venue_id")[["venue_lr_r6", "venue_balls"]], how="inner")
        j = j[(j.balls >= 600) & (j.venue_balls > np.log1p(600))]
        j["observed_lr"] = np.log((j.o6 + 1) / (j.e6 + 1))
        rows.append(j.assign(year=y))
    v = pd.concat(rows)
    out["venue_six"] = {
        "ground_years": int(len(v)), "years": "2016-2026, T20 family, grounds with 600+ balls that year and 600+ "
                                              "weighted balls of history",
        "correlation_rating_vs_that_year": round(float(np.corrcoef(v.venue_lr_r6, v.observed_lr)[0, 1]), 3),
        "slope": round(float(np.polyfit(v.venue_lr_r6, v.observed_lr, 1)[0]), 3),
        "rating_spread_sd": round(float(v.venue_lr_r6.std()), 3)}
    # Dhoni's ODI death overs (overs 40-49) by year: observed runs per ball over expected
    dhoni = odi.batter_id.isin(DHONI_IDS) & (odi.over >= 40) & ~np.isnan(odi_exp).any(1)
    d = odi[dhoni].assign(exp_runs=odi_exp[dhoni.to_numpy()] @ CLASS_RUNS,
                          exp_six=odi_exp[dhoni.to_numpy(), CLASSES.index("6")],
                          is_six=lambda x: x.cls == CLASSES.index("6"),
                          is_dot=lambda x: x.cls == CLASSES.index("dot"),
                          exp_dot=odi_exp[dhoni.to_numpy(), CLASSES.index("dot")])
    g = d.groupby("y").agg(balls=("cls", "size"), runs=("runs_batter", "sum"), exp_runs=("exp_runs", "sum"),
                           sixes=("is_six", "sum"), exp_six=("exp_six", "sum"), dots=("is_dot", "sum"),
                           exp_dot=("exp_dot", "sum"))
    g["runs_vs_expected"] = (g.runs / g.exp_runs).round(2)
    g["sixes_vs_expected"] = (g.sixes / g.exp_six).round(2)
    g["dots_vs_expected"] = (g.dots / g.exp_dot).round(2)
    out["dhoni_odi_death_overs"] = g[["balls", "runs", "runs_vs_expected", "sixes_vs_expected",
                                      "dots_vs_expected"]].reset_index().to_dict("records")
    return out


DHONI_IDS: list[str] = []


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--families", default="T20,ODI")
    ap.add_argument("--first-year", type=int, default=2019)
    ap.add_argument("--last-year", type=int, default=2026)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    con = duckdb.connect()
    con.execute("SET memory_limit = '4GB'")
    started = time.time()
    DHONI_IDS.extend(r[0] for r in con.execute(
        f"SELECT cricsheet_id FROM read_parquet('{GOLD / 'dim_player'}.parquet') WHERE name = 'MS Dhoni'").fetchall())
    per_match, summary, keep = [], {}, {}
    for family in args.families.split(","):
        allb = all_balls(con, family)
        exp, prior = expected_probs(allb)
        tables = {e: ratings(allb, exp, prior, col, e) for e, col in ENTITIES.items()}
        for e, t in tables.items():
            t.to_parquet(args.out / f"ratings_{family}_{e}.parquet", index=False)
        keep[family] = (allb.assign(e6=exp[:, CLASSES.index("6")]), exp, tables["venue"])
        print(f"{family}: {len(allb):,} balls, ratings built at {time.time() - started:.0f}s", flush=True)

        balls = model_balls(con, family)
        balls["start_date"] = pd.to_datetime(balls.start_date)
        balls = balls.merge(allb[["match_id", "innings", "delivery_number", "bowler_id", "venue_id"]],
                            on=["match_id", "innings", "delivery_number"], how="left")
        for e, col in ENTITIES.items():
            balls = balls.merge(tables[e], on=[col, "start_date"], how="left")
        base = OUTCOME_FEATURES + sm.CONTEXT + sm.RECENT
        rate = [c for e in ENTITIES for c in rating_cols(e)]
        models = {"base": base, "ratings": base + rate, "ratings_no_xi": [c for c in base if c not in XI] + rate}
        cover = {e: float(balls[f"{e}_balls"].notna().mean()) for e in ENTITIES}
        print(f"{family}: {len(balls):,} model balls, rating coverage {cover}", flush=True)
        for year in range(args.first_year, args.last_year + 1):
            cutoff = pd.Timestamp(f"{year}-01-01")
            train = balls[(balls.start_date < cutoff) & (balls.start_date >= cutoff - pd.DateOffset(years=OUTCOME_SEASONS))]
            test = balls[balls.start_date.dt.year == year]
            if test.empty:
                continue
            ll = pd.DataFrame({"match_id": test.match_id.to_numpy(), "segment": test.segment.to_numpy()})
            for name, feats in models.items():
                ll[name] = fit_predict(train, test, feats)
            pm = ll.groupby(["match_id", "segment"]).agg(balls=("base", "size"), **{m: (m, "sum") for m in models})
            per_match.append(pm.reset_index().assign(family=family, year=year))
            msg = {m: round(float(ll[m].mean()), 4) for m in models}
            print(f"  {family} {year}: {len(test):,} test balls, train {len(train):,}, log loss {msg}, "
                  f"{time.time() - started:.0f}s", flush=True)

    pm = pd.concat(per_match, ignore_index=True)
    pm["group"] = np.where(pm.family == "ODI", "ODI", np.where(pm.segment == "T20I", "T20I", "league"))
    pm.to_csv(args.out / "per_match.csv", index=False)
    rng = np.random.default_rng(SEED)
    for grp, g in pm.groupby("group"):
        summary[grp] = {"ratings_minus_base": band(g, "base", "ratings", rng),
                        "ratings_no_xi_minus_base": band(g, "base", "ratings_no_xi", rng),
                        "by_year_ratings_minus_base": {int(y): round(float((h.ratings.sum() - h.base.sum()) / h.balls.sum()), 5)
                                                       for y, h in g.groupby("year")}}
        summary[grp]["pass"] = summary[grp]["ratings_minus_base"]["band_90"][1] < 0
    ck = {}
    if "T20" in keep and "ODI" in keep:
        ck = checks(keep["T20"][0], keep["T20"][2], keep["ODI"][0], keep["ODI"][1])
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = {"commit": commit, "build_fingerprint": json.loads((GOLD / "run.json").read_text())["build_fingerprint"],
            "half_life_years": HALF_LIFE_YEARS, "pseudo_balls": PSEUDO, "pseudo_balls_wicket": PSEUDO_WICKET,
            "years": [args.first_year, args.last_year], "minutes": round((time.time() - started) / 60, 1)}
    (args.out / "summary.json").write_text(json.dumps({"meta": meta, **summary}, indent=2))
    (args.out / "checks.json").write_text(json.dumps(ck, indent=2, default=int))
    print(json.dumps(summary, indent=2))
    print(json.dumps(ck, indent=2, default=int)[:3000])


if __name__ == "__main__":
    main()
