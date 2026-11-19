"""Walk-forward backtest of the ball-by-ball simulator (simulate_matches.py) that keeps every play of every match,
so any score can be computed later without simulating again (score_sim_backtest.py does the scoring).

For each year from --first-year on, a ball model is fitted on the ten years before 1 January of that year, then
every clean full-length match of that year is played --sims-* times. Formats:
  T20I    T20I test matches; the ball model trains on T20I balls plus, unless the variant says no, every ball of
          the IPL and the nine leagues (is_international 0 on those)
  league  held-out IPL and league matches of the same year, played by the same T20 model (is_international 0)
  ODI     ODI test matches, ODI balls only
Years before the scoring years (2016 to 2018 by default) are simulated too: the range widening in
score_sim_backtest.py is fitted on earlier years only, so 2019 needs earlier years to learn from.

Variants: base is the simulator as it is (league balls in the T20 model, pitch factor off, as in every measurement
run). recent adds the in-play inputs (simulate_matches.RECENT: last-6-ball runs and boundaries, the striker's
balls faced). combined is recent with fixed spreads (no tuning) and a note asking the scorer to stretch ODI totals.
recent_spread adds to recent a pitch factor and a per-innings batting-side factor, tuned like the
pitch factor in simulate_matches.py: fitted on the ten years before --tune-year, played on that year (T20I and
league matches together for T20, ODIs for ODI) with --tune-plays plays; first the pitch spread, then the innings
spread at that pitch spread, each the lowest CRPS of first-innings totals. --tune-year is before every scoring year.

Seeds: each (format, year) gets its own generator seeded from (SEED, year, format), the same for every variant,
so variants are compared on the same random streams as far as their draws line up. Plays are independent, so the
first and second half of each match's plays are two independent runs; the scorer uses them to measure Monte
Carlo noise.

Writes (in --out/<variant>/, which must not exist): one plays_<format>_<year>.npz per format and year (first:
matches x plays first-innings totals, result: 2 x the batting-first side's result, so 2 win, 1 tie, 0 loss),
matches.csv (one row per test match, in the same order) and run.json.

Prototype outside the build; reads gold, writes only to --out.
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
from cricket_sim.models.elo import expected
from cricket_sim.models.train import SEED
from cricket_sim.paths import GOLD

FORMAT_CODE = {"T20I": 1, "league": 2, "ODI": 3}
# leagues: the T20 ball model also trains on league balls; score_leagues: league matches are also played
VARIANTS = {
    "base": {"leagues": True, "score_leagues": True, "recent": False, "spread": False},
    "t20i_only": {"leagues": False, "score_leagues": False, "recent": False, "spread": False},
    "recent": {"leagues": True, "score_leagues": True, "recent": True, "spread": False},
    "recent_spread": {"leagues": True, "score_leagues": True, "recent": True, "spread": True},
    # the pick from the 2026-11-19 report: in-play inputs, T20 pitch spread 0.025 as tuned on 2018, ODI pitch factor
    # off, and ODI totals stretched after the sim (done by score_sim_backtest.py, fitted on earlier years)
    "combined": {"leagues": True, "score_leagues": True, "recent": True, "spread": False,
                 "fixed_spread": {"T20I": [0.025, 0.0], "ODI": [0.0, 0.0]}, "post_stretch": ["ODI"]},
    # combined plus player ratings (2026-11-19): the striker, the bowler of the over and the ground, as of the match
    # date, in place of the six XI averages; real batting order and real bowler per over (simulate_matches.add_ratings)
    "ratings": {"leagues": True, "score_leagues": True, "recent": True, "spread": False, "ratings": True,
                "fixed_spread": {"T20I": [0.025, 0.0], "ODI": [0.0, 0.0]}, "post_stretch": ["ODI"]},
}


def tune(balls, tests, year, plays, pitch_sds, innings_sds, recent, rng) -> tuple[float, float, list]:
    """Pitch spread first, then the innings spread at the chosen pitch spread, each by the lowest mean CRPS of
    first-innings totals on `year`, with a model fitted on the ten years before it."""
    model = sm.fit(balls, pd.Timestamp(f"{year}-01-01"), None, recent=recent)
    val = pd.concat([sm.full_length(ms[pd.to_datetime(ms.start_date).dt.year == year]) for _, ms in tests],
                    ignore_index=True)
    rows = []

    def run(p, i):
        s = sm.simulate(model, val, plays, rng, 1.0, p, i)
        y = val.bat_first_won.to_numpy(float)
        c = np.clip(s["result"].mean(1), 0.01, 0.99)
        rows.append({"pitch_sd": p, "innings_sd": i, "matches": len(val),
                     "crps_runs": float(sm.crps(s["first"], val.first_total.to_numpy(float)).mean()),
                     "log_loss": float(-(y * np.log(c) + (1 - y) * np.log(1 - c)).mean())})
        print(f"    tune {year}: pitch {p}, innings {i}: {rows[-1]}", flush=True)
        return rows[-1]["crps_runs"]

    best_p = min(pitch_sds, key=lambda p: run(p, 0.0))
    best_i = min(innings_sds, key=lambda i: run(best_p, i) if i else
                 next(r["crps_runs"] for r in rows if r["pitch_sd"] == best_p and r["innings_sd"] == 0.0))
    return best_p, best_i, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--formats", default="T20I,ODI")
    ap.add_argument("--first-year", type=int, default=2016)
    ap.add_argument("--last-year", type=int, default=2026)
    ap.add_argument("--sims-t20i", type=int, default=400)
    ap.add_argument("--sims-league", type=int, default=200)
    ap.add_argument("--sims-odi", type=int, default=300)
    ap.add_argument("--tune-year", type=int, default=2018)
    ap.add_argument("--tune-plays", type=int, default=100)
    ap.add_argument("--pitch-sds", default="0,0.025,0.05,0.075,0.1")
    ap.add_argument("--innings-sds", default="0,0.025,0.05,0.075,0.1,0.15")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    v = VARIANTS[args.variant]
    out = args.out / args.variant
    out.mkdir(parents=True, exist_ok=False)
    con = duckdb.connect()
    con.execute("SET memory_limit = '4GB'")
    sims = {"T20I": args.sims_t20i, "league": args.sims_league, "ODI": args.sims_odi}
    rows, started, spread = [], time.time(), {}

    for fmt in args.formats.split(","):
        balls, matches = sm.load(con, fmt)
        matches = matches.assign(is_international=1.0, segment=fmt)
        tests = [(fmt, matches)]
        if fmt == "T20I":
            lb, lm = sm.league_data(con)
            if v["leagues"]:
                balls = pd.concat([balls, lb], ignore_index=True)
            if v["score_leagues"]:
                tests.append(("league", lm))
        lineups = None
        if v.get("ratings"):
            balls, lineups = sm.add_ratings(con, balls, pd.concat([ms for _, ms in tests], ignore_index=True),
                                            "T20" if fmt == "T20I" else "ODI")
        print(f"{fmt}: {len(balls):,} training balls in the pool, loaded at {time.time() - started:.0f}s", flush=True)
        pitch_sd, innings_sd = v.get("fixed_spread", {}).get(fmt, [0.0, 0.0])
        if v.get("fixed_spread"):
            spread[fmt] = {"pitch_sd": pitch_sd, "innings_sd": innings_sd, "fixed": True}
        if v["spread"]:
            pitch_sd, innings_sd, grid = tune(balls, tests, args.tune_year, args.tune_plays,
                                              [float(x) for x in args.pitch_sds.split(",")],
                                              [float(x) for x in args.innings_sds.split(",")], v["recent"],
                                              np.random.default_rng([SEED, args.tune_year, 9]))
            spread[fmt] = {"pitch_sd": pitch_sd, "innings_sd": innings_sd, "tuned_on": args.tune_year, "grid": grid}
            print(f"  {fmt}: chosen pitch sd {pitch_sd}, innings sd {innings_sd}", flush=True)
        for year in range(args.first_year, args.last_year + 1):
            cutoff = pd.Timestamp(f"{year}-01-01")
            model = sm.fit(balls, cutoff, None, recent=v["recent"], ratings=v.get("ratings", False))
            for group, ms in tests:
                test = sm.full_length(ms[pd.to_datetime(ms.start_date).dt.year == year]).reset_index(drop=True)
                if test.empty:
                    continue
                rng = np.random.default_rng([SEED, year, FORMAT_CODE[group]])
                s = sm.simulate(model, test, sims[group], rng, 1.0, pitch_sd, innings_sd, lineups)
                np.savez_compressed(out / f"plays_{group}_{year}.npz", first=s["first"].astype(np.int16),
                                    result=(2 * s["result"]).astype(np.int8))
                elo = [expected(a, b) for a, b in zip(test.elo_bat_first, test.elo_bat_second)]
                rows.append(pd.DataFrame({
                    "format": group, "segment": test.segment, "year": year, "match_id": test.match_id,
                    "start_date": test.start_date, "bat_first": test.bat_first, "bat_second": test.bat_second,
                    "actual_first_total": test.first_total, "bat_first_won": test.bat_first_won,
                    "elo_bat_first_win": elo, "plays": sims[group]}))
                print(f"  {group} {year}: {len(test)} matches x {sims[group]} plays, "
                      f"{time.time() - started:.0f}s so far", flush=True)

    pd.concat(rows, ignore_index=True).to_csv(out / "matches.csv", index=False)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    (out / "run.json").write_text(json.dumps({
        "variant": args.variant, **v, "commit": commit, "tree_dirty": dirty,
        "build_fingerprint": json.loads((GOLD / "run.json").read_text())["build_fingerprint"],
        "seed": SEED, "first_year": args.first_year, "last_year": args.last_year, "plays": sims,
        "spread": spread or "pitch and innings factors off", "minutes": round((time.time() - started) / 60, 1)}, indent=2))


if __name__ == "__main__":
    main()
