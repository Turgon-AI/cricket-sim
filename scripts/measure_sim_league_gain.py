"""Does the ball-by-ball simulator get better at T20Is when its ball model also learns from league balls?

measure_league_gain.py found that a match-level model of T20I first-innings totals explains 9.9% of the spread
trained on T20Is alone and 15.9% with the IPL and nine leagues as extra training rows. This checks the same idea
inside the simulator (simulate_matches.py), which predicts every ball and plays the match out.

For each test year (walk forward, --first-test-year on), two ball models are fitted on the ten years before it:
  t20i_only    T20I balls only (what simulate_matches.py does today)
  with_leagues + every ball of the IPL and the nine leagues in those years, with a flag is_international
               (1 on T20I balls, 0 on league balls), so the model can learn where the two differ
The league loading and the is_international input live in simulate_matches.py (league_balls). Ball inputs are
simulate_matches.py's: the match state, Elo gap, year, and the venue and playing-XI context. Each
league's Elo, level and ground history are its own; T20I ground history is unchanged (league innings in T20I
ground history did not help in measure_league_gain.py). Every clean full-length T20I of the test year is then
played --sims times with each model, pitch factor off, as in measure_luck_share.py.

Scored on the same matches for both models:
  explained share   1 minus squared error of the mean simulated first-innings total over the squared distance
                    from the test year's own mean (same denominator as the other measures)
  luck share        mean variance of the simulated totals over the same denominator
  CRPS              how close the whole simulated distribution of the first-innings total is to the actual total
                    (runs, lower is better)
  win Brier         the simulated chance that the side batting first wins, against what happened
Gains are bootstrapped over test matches (paired), 90% band.

Prototype outside the build; reads gold, writes only to --out (a new folder).
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import simulate_matches as sm
from cricket_sim.models.train import SEED
from cricket_sim.paths import GOLD

def scores(t: pd.DataFrame, col: str) -> dict:
    dev2 = (t.actual - t.groupby("year").actual.transform("mean")) ** 2
    return {"explained_share": 1 - ((t.actual - t[f"{col}_mean"]) ** 2).sum() / dev2.sum(),
            "luck_share": t[f"{col}_var"].sum() / dev2.sum(),
            "crps_runs": t[f"{col}_crps"].mean(),
            "win_brier": ((t[f"{col}_win"] - t.bat_first_won) ** 2).mean(),
            "mean_bias_runs": (t[f"{col}_mean"] - t.actual).mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-test-year", type=int, default=2019)
    ap.add_argument("--sims", type=int, default=300)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(SEED)
    con = duckdb.connect()
    con.execute("SET memory_limit = '4GB'")

    t20i_balls, t20i_matches = sm.load(con, "T20I")
    league_balls = sm.league_balls(con)
    print(f"T20I balls {len(t20i_balls):,}, league balls {len(league_balls):,}", flush=True)
    all_balls = pd.concat([t20i_balls, league_balls], ignore_index=True)

    t20i_matches["year"] = pd.to_datetime(t20i_matches.start_date).dt.year
    rows = []
    for year in range(args.first_test_year, int(t20i_matches.year.max()) + 1):
        test = sm.full_length(t20i_matches[t20i_matches.year == year]).reset_index(drop=True)
        if test.empty:
            continue
        cutoff = pd.Timestamp(f"{year}-01-01")
        t = pd.DataFrame({"match_id": test.match_id, "year": year, "actual": test.first_total.astype(float),
                          "bat_first_won": test.bat_first_won})
        for name, pool in (("t20i_only", t20i_balls), ("with_leagues", all_balls)):
            start = time.time()
            model = sm.fit(pool, cutoff, None)
            s = sm.simulate(model, test, args.sims, rng, 1.0, 0.0)
            t[f"{name}_mean"] = s["first"].mean(1)
            t[f"{name}_var"] = s["first"].var(1)
            t[f"{name}_crps"] = sm.crps(s["first"], t.actual.to_numpy())
            t[f"{name}_win"] = s["result"].mean(1)
            print(f"  {year} {name}: {len(test)} matches, trained on {len(model.gbm.classes_)} classes "
                  f"in {time.time() - start:.0f}s", flush=True)
        rows.append(t)

    t = pd.concat(rows, ignore_index=True)
    t.to_csv(args.out / "per_match.csv", index=False)
    n = len(t)
    draws = [rng.integers(0, n, n) for _ in range(args.boot)]
    result = {name: scores(t, name) for name in ("t20i_only", "with_leagues")}
    gain = {}
    for k in result["t20i_only"]:
        d = [scores(t.iloc[i], "with_leagues")[k] - scores(t.iloc[i], "t20i_only")[k] for i in draws]
        gain[k] = {"gain": result["with_leagues"][k] - result["t20i_only"][k],
                   "band": [float(x) for x in np.percentile(d, [5, 95])]}
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    summary = {"test_matches": n, "years": sorted(t.year.unique().tolist()), "results": result, "gain": gain,
               "run": {"commit": commit, "build_fingerprint": json.load(open(GOLD / "run.json"))["build_fingerprint"],
                       "sims": args.sims, "seed": SEED, "train_balls_t20i": len(t20i_balls),
                       "train_balls_leagues": len(league_balls)}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({"results": result, "gain": gain}, indent=2, default=float))


if __name__ == "__main__":
    main()
