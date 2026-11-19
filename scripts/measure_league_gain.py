"""Do the franchise leagues help predict T20Is? Same measure as measure_explained_share.py, on the same T20I test
matches (clean full-member T20Is from --first-test-year, walk forward, ten training years before each test year).

Three variants, each scored on exactly the same T20I matches:
  base         as measure_explained_share.py: trained on T20Is only; ground history from T20Is and the IPL
  venue        + ground history also from the nine leagues added 2026-11-18 (each innings measured against its own
                 league's recent level, so a high-scoring league does not inflate a ground)
  rows_only    the IPL and the nine leagues as extra training rows (below), ground history as in base
  rows         + the IPL and the nine leagues as extra training rows, with a flag for "international", so the model
                 can learn how a T20 total follows from level, Elo, ground, XI and toss from 3 to 4 times as many
                 matches. Test rows are still T20Is only.

Inputs are the pre-match set (level, Elo, venue, XI, toss); weather and conditions are left out (they explained
nothing on 2026-11-18). The player ratings (XI) already included league balls before the leagues were added, so
that channel is not tested here.

Share explained = 1 minus squared error over the squared distance from each test year's own mean. The gain of a
variant over base is bootstrapped over test matches (paired: same draws for both), 90% band.

Prototype outside the build; reads gold, writes only to --out (a new folder).
"""

import argparse
import json
import subprocess
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import measure_explained_share as mes
from cricket_sim.models.train import SEED
from cricket_sim.paths import GOLD

LEAGUES = ["T20 Blast", "BBL", "BPL", "CPL", "PSL", "LPL", "ILT20", "SA20", "MLC"]
BASE_FAMILY = ["T20I", "IPL"]
WIDE_FAMILY = BASE_FAMILY + LEAGUES
FEATS = [f for layer in ("level", "elo", "venue", "xi", "toss") for f in mes.LAYERS[layer]] + ["is_international"]


def frame(con, hist, segment, venue_family):
    mes.FAMILY[segment] = ("t20", venue_family)
    df = mes.matches(con, segment)
    df = mes.add_level_and_venue(df, hist, segment)
    df = mes.add_xi(con, df, segment)
    df["segment"] = segment
    df["is_international"] = float(segment == "T20I")
    return df


def walk_forward(train_pool, test_pool, first_test_year):
    out = []
    for year in range(first_test_year, int(test_pool.year.max()) + 1):
        train = train_pool[(train_pool.year >= year - 10) & (train_pool.year < year)]
        test = test_pool[test_pool.year == year]
        if test.empty:
            continue
        o = test[["match_id", "year", "first_total", "bat_first_won"]].copy()
        o["base_rate"] = train[train.segment == "T20I"].bat_first_won.mean()
        wins = train[train.bat_first_won != 0.5]
        for name, make in mes.regressors().items():
            o[f"total_{name}"] = make().fit(train[FEATS], train.first_total).predict(test[FEATS])
        for name, make in mes.classifiers().items():
            o[f"win_{name}"] = make().fit(wins[FEATS], wins.bat_first_won.astype(int)).predict_proba(test[FEATS])[:, 1]
        o["train_rows"] = len(train)
        out.append(o)
    return pd.concat(out, ignore_index=True)


def share(p, col):
    year_mean = p.groupby("year").first_total.transform("mean")
    return 1 - ((p.first_total - p[col]) ** 2).sum() / ((p.first_total - year_mean) ** 2).sum()


def skill(p, col):
    return 1 - ((p.bat_first_won - p[col]) ** 2).mean() / ((p.bat_first_won - p.base_rate) ** 2).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-test-year", type=int, default=2019)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    con = duckdb.connect()
    hist = mes.first_innings(con)
    rng = np.random.default_rng(SEED)

    t20i_base = frame(con, hist, "T20I", BASE_FAMILY)
    t20i_wide = frame(con, hist, "T20I", WIDE_FAMILY)
    others = pd.concat([frame(con, hist, s, WIDE_FAMILY) for s in ["IPL"] + LEAGUES], ignore_index=True)
    # a league's own innings always count for its own level and ground history; only T20I history stays narrow
    others_base = pd.concat([frame(con, hist, s, BASE_FAMILY + [s]) for s in ["IPL"] + LEAGUES], ignore_index=True)
    print("training rows by segment:", others.groupby("segment").size().to_dict(), "T20I", len(t20i_wide), flush=True)

    variants = {
        "base": walk_forward(t20i_base, t20i_base, args.first_test_year),
        "venue": walk_forward(t20i_wide, t20i_wide, args.first_test_year),
        "rows_only": walk_forward(pd.concat([t20i_base, others_base], ignore_index=True), t20i_base,
                                  args.first_test_year),
        "rows": walk_forward(pd.concat([t20i_wide, others], ignore_index=True), t20i_wide, args.first_test_year),
    }
    ids = variants["base"].match_id
    assert all((v.match_id.to_numpy() == ids.to_numpy()).all() for v in variants.values()), "test sets differ"

    n = len(ids)
    draws = [rng.integers(0, n, n) for _ in range(args.boot)]
    rows = []
    for name, p in variants.items():
        p.to_csv(args.out / f"predictions_{name}.csv", index=False)
        for model in ("linear", "boosted"):
            t, w = f"total_{model}", f"win_{model}"
            row = {"variant": name, "model": model, "test_matches": n,
                   "train_rows_last_year": int(p.train_rows.iloc[-1]),
                   "share_explained": share(p, t), "mean_abs_error_runs": (p.first_total - p[t]).abs().mean(),
                   "win_skill": skill(p, w)}
            if name != "base":
                b = variants["base"]
                d_share = [share(p.iloc[d], t) - share(b.iloc[d], t) for d in draws]
                d_skill = [skill(p.iloc[d], w) - skill(b.iloc[d], w) for d in draws]
                row.update({"gain_share": row["share_explained"] - share(b, t),
                            "gain_share_band": [float(x) for x in np.percentile(d_share, [5, 95])],
                            "gain_win_skill": row["win_skill"] - skill(b, w),
                            "gain_win_skill_band": [float(x) for x in np.percentile(d_skill, [5, 95])]})
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(args.out / "league_gain.csv", index=False)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    json.dump({"commit": commit, "build_fingerprint": json.load(open(GOLD / "run.json")).get("build_fingerprint"),
               "first_test_year": args.first_test_year, "features": FEATS, "seed": SEED,
               "training_rows_by_segment": {**others.groupby("segment").size().to_dict(), "T20I": len(t20i_wide)},
               "results": rows}, open(args.out / "summary.json", "w"), indent=2, default=float)
    pd.set_option("display.width", 250)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
