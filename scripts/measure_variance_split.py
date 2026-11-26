"""Split the spread in first-innings totals into what was knowable before the match and what only showed up on
the day. Follows measure_luck_share.py, which found that about half the spread is ball-to-ball luck.

Every share below is of the same denominator as measure_explained_share.py: the squared distance of each actual
first-innings total from its test year's own mean, pooled over the test years (walk forward, 2019 on).

  explained    the pre-match inputs we already use (level, Elo, venue history, home, playing XI, toss), linear,
               fitted on the ten years before each test year
  persistent   what the misses of earlier matches would add: the mean residual at this ground, of this batting side
               when batting, of this bowling side when bowling, and of the whole format over the last 120 days,
               all from matches before this one. A missed pattern that carries over from match to match was
               knowable before the match, so this is pre-match headroom we can show today with no new data.
  shared day   what both innings of the same match have in common after the two steps above: the covariance of
               the first-innings residual with the second innings' residual. The second innings is put on the
               same scale as a total by adding, to the runs it made, the runs a first innings typically still
               scores from the balls and wickets it had left (so a chase that won with 10 overs to spare counts
               as a bigger total). This is that day's pitch and conditions as both sides met them.
  luck         from measure_luck_share.py (per_match.csv), hindsight and model estimates
  side on the day   what is left: specific to the batting side on that day (form, plans, selection on the day,
               a pitch that changed between innings, dew), plus pre-match signal we have not found.

Which way each part leans:
  persistent   a floor. Better ways to use the same history (or player-level misses) could find more.
  shared day   counts only conditions that hit both innings alike. Dew or a pitch that wears between innings shows
               up as different for the two innings and lands in "side on the day". A chase also plays to its
               target, which blurs the second innings as a measure.

Prototype outside the build; reads gold and the luck-share output, writes only to --out (a new folder).
"""

import argparse
import json
import subprocess
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cricket_sim.models.train import SEED
from cricket_sim.paths import GOLD
from measure_explained_share import LAYERS, add_level_and_venue, add_xi, first_innings, gold, matches

PRE_MATCH = [f for layer in ("level", "elo", "venue", "xi", "toss") for f in LAYERS[layer]]
# --with-conditions adds the day's observed weather and the 2026-11-18 conditions inputs (start time, innings
# weather, ground use, altitude) to the first step, to see how much of the shared-day part they take
CONDITIONS = [f for layer in ("weather", "conditions") for f in LAYERS[layer]]
LAG = ["lag_venue", "lag_venue_n", "lag_bat", "lag_bat_n", "lag_bowl", "lag_bowl_n", "lag_format"]
WINDOW_DAYS = {"venue": 730, "bat": 365, "bowl": 365, "format": 120}
SHRINK = {"venue": 3, "bat": 5, "bowl": 5}  # pseudo-matches at zero residual


def ridge():
    return make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=1.0))


def runs_to_come(con, segment: str):
    """A smooth table: mean runs a first innings still scores from (balls left, wickets down), fitted on every
    clean full-length first innings of the format. Returns a function of (balls_left, wickets) arrays."""
    d = con.execute(f"""
        SELECT i.innings_balls_limit - f.legal_balls_before AS balls_left, f.innings_wickets_before AS wkts,
               i.runs - f.innings_runs_before AS to_come
        FROM {gold(con, 'fact_delivery')} f JOIN {gold(con, 'dim_match')} m USING (match_id)
        JOIN {gold(con, 'fact_innings')} i ON i.match_id = f.match_id AND i.innings = 1 AND NOT i.is_super_over
        WHERE m.segment = ? AND m.method IS NULL AND m.passes_quality AND f.innings = 1 AND NOT f.is_super_over
          AND i.innings_length_status = 'full_known' AND i.innings_balls_limit = m.scheduled_overs * 6
          AND hash(f.match_id, f.delivery_number) % 4 = 0""", [segment]).df()  # a quarter of balls is plenty
    gbm = HistGradientBoostingRegressor(max_iter=200, monotonic_cst=[1, -1], random_state=SEED)
    gbm.fit(d[["balls_left", "wkts"]], d.to_come)

    def f(balls_left, wkts):
        x = pd.DataFrame({"balls_left": balls_left, "wkts": np.minimum(wkts, 9)})
        return np.where((balls_left <= 0) | (wkts >= 10), 0.0, gbm.predict(x))
    return f


def second_innings(con, segment: str) -> pd.DataFrame:
    return con.execute(f"""
        SELECT match_id, runs AS second_runs, legal_balls AS second_balls, dismissals AS second_wkts,
               innings_balls_limit AS second_limit
        FROM {gold(con, 'fact_innings')} WHERE innings = 2 AND NOT is_super_over""").df()


def walk_forward(df, target: str, feats: list, first_year: int) -> pd.Series:
    """Out-of-sample prediction of target for every match from first_year on (ten training years before)."""
    out = pd.Series(np.nan, index=df.index)
    for year in range(first_year, int(df.year.max()) + 1):
        train = df[(df.year >= year - 10) & (df.year < year) & df[target].notna()]
        test = df.year == year
        if test.any() and len(train) > 50:
            out[test] = ridge().fit(train[feats], train[target]).predict(df.loc[test, feats])
    return out


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Shrunk mean residual of earlier matches (strictly earlier dates) by ground, batting side, bowling side, and
    the whole format. Built for both innings: lag features for innings 1 (suffix _1) and innings 2 (_2)."""
    # one row per innings with its residual, the side batting and the side bowling
    inn = pd.concat([
        pd.DataFrame({"date": df.start_date, "venue": df.venue_id, "bat": df.bat_first, "bowl": df.bat_second,
                      "r": df.r1}),
        pd.DataFrame({"date": df.start_date, "venue": df.venue_id, "bat": df.bat_second, "bowl": df.bat_first,
                      "r": df.r2})]).dropna(subset=["r"])
    d_all = inn.date.to_numpy()
    r_all = inn.r.to_numpy()

    def lag(date, key=None, value=None, kind="format"):
        w = np.timedelta64(WINDOW_DAYS[kind], "D")
        sel = (d_all < date) & (d_all >= date - w)
        if key is not None:
            sel &= inn[key].to_numpy() == value
        r = r_all[sel]
        if kind == "format":
            return (r.mean() if len(r) else 0.0), len(r)
        return r.sum() / (len(r) + SHRINK[kind]), len(r)

    for suffix, bat, bowl in (("_1", "bat_first", "bat_second"), ("_2", "bat_second", "bat_first")):
        rows = []
        for date, venue, b, w in zip(df.start_date.to_numpy(), df.venue_id, df[bat], df[bowl]):
            v, vn = lag(date, "venue", venue, "venue")
            bt, btn = lag(date, "bat", b, "bat")
            bw, bwn = lag(date, "bowl", w, "bowl")
            f, _ = lag(date)
            rows.append([v, vn, bt, btn, bw, bwn, f])
        df[[c + suffix for c in LAG]] = np.array(rows)
    return df


def shares(t: pd.DataFrame, luck_col: str) -> dict:
    dev2 = (t.first_total - t.groupby("year").first_total.transform("mean")) ** 2
    denom = dev2.sum()
    sse = lambda e: (e ** 2).sum()
    # the shared part: covariance of the two innings' residuals after the persistent step, within each year
    a = t.r1p - t.groupby("year").r1p.transform("mean")
    b = t.r2p - t.groupby("year").r2p.transform("mean")
    out = {
        "explained": 1 - sse(t.first_total - t.pred1) / denom,
        "persistent": (sse(t.r1) - sse(t.r1p)) / denom,
        "shared_day": (a * b).sum() / denom,
        "luck": t[luck_col].sum() / denom,
    }
    out["side_on_the_day"] = 1 - sum(out.values())
    return out


def report(t: pd.DataFrame, luck_col: str, rng, boot: int = 300) -> dict:
    point = shares(t, luck_col)
    draws = [shares(t.iloc[rng.integers(0, len(t), len(t))], luck_col) for _ in range(boot)]
    res = {"matches": len(t), "spread_sd_runs": float(np.sqrt(
        ((t.first_total - t.groupby("year").first_total.transform("mean")) ** 2).mean()))}
    for k, v in point.items():
        res[k] = float(v)
        res[f"{k}_band"] = [float(x) for x in np.percentile([d[k] for d in draws], [5, 95])]
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--first-test-year", type=int, default=2019)
    ap.add_argument("--first-residual-year", type=int, default=2013,
                    help="first year with out-of-sample residuals, the history the lag features read")
    ap.add_argument("--luck", type=Path, default=Path("sims/2026-11-17-luck-share/per_match.csv"))
    ap.add_argument("--with-conditions", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    inputs = PRE_MATCH + (CONDITIONS if args.with_conditions else [])
    rng = np.random.default_rng(SEED)
    full = set(yaml.safe_load(open("config/scope.yml"))["full_members"])
    luck = pd.read_csv(args.luck, dtype={"match_id": str})
    con = duckdb.connect()
    con.execute("SET memory_limit = '3GB'")
    hist = first_innings(con)
    summary, per_match = {}, []

    for segment in ("ODI", "T20I"):
        df = matches(con, segment)
        df = add_level_and_venue(df, hist, segment)
        df = add_xi(con, df, segment)
        df["start_date"] = pd.to_datetime(df.start_date)
        df["full_member"] = df.team1.isin(full) & df.team2.isin(full)
        # the second innings on the scale of a total: runs made + what a first innings scores from what was left
        df = df.merge(second_innings(con, segment), on="match_id", how="left")
        left = runs_to_come(con, segment)
        df["second_proj"] = df.second_runs + left((df.second_limit - df.second_balls).to_numpy(),
                                                  df.second_wkts.to_numpy())
        df = df.sort_values(["start_date", "match_id"]).reset_index(drop=True)
        # step 1: pre-match prediction of both innings (walk forward), residuals
        df["pred1"] = walk_forward(df, "first_total", inputs, args.first_residual_year)
        df["pred2"] = walk_forward(df, "second_proj", inputs, args.first_residual_year)
        df["r1"], df["r2"] = df.first_total - df.pred1, df.second_proj - df.pred2
        # step 2: what earlier matches' misses add (lag features from earlier residuals only)
        df = add_lags(df)
        df["p1"] = walk_forward(df[df.year >= args.first_residual_year + 1].assign(), "r1",
                                [c + "_1" for c in LAG], args.first_test_year).reindex(df.index)
        df["p2"] = walk_forward(df[df.year >= args.first_residual_year + 1].assign(), "r2",
                                [c + "_2" for c in LAG], args.first_test_year).reindex(df.index)
        df["r1p"], df["r2p"] = df.r1 - df.p1, df.r2 - df.p2
        # the test set: matches the luck measurement also covered (clean, full length, 2019 on)
        t = df[df.year >= args.first_test_year].merge(
            luck[luck.segment == segment][["match_id", "hind_var", "model_var"]], on="match_id", how="inner")
        t = t.dropna(subset=["pred1", "pred2", "p1", "p2", "second_proj"])
        per_match.append(t[["match_id", "start_date", "year", "full_member", "bat_first", "bat_second",
                            "first_total", "second_proj", "pred1", "pred2", "p1", "p2", "r1", "r2",
                            "hind_var", "model_var"]].assign(segment=segment))
        summary[segment] = {}
        for name, d in (("all", t), ("full_members", t[t.full_member])):
            if segment == "T20I" and name == "full_members":
                continue  # T20Is in gold are full-member matches already
            summary[segment][f"{name}_hindsight_luck"] = report(d, "hind_var", rng)
            m = d[d.model_var.notna()]
            summary[segment][f"{name}_model_luck_2023_on"] = report(m, "model_var", rng)
            for k in (f"{name}_hindsight_luck", f"{name}_model_luck_2023_on"):
                r = summary[segment][k]
                print(segment, k, r["matches"], {x: round(r[x], 3) for x in
                      ("explained", "persistent", "shared_day", "luck", "side_on_the_day")}, flush=True)
        summary[segment]["second_innings_projection_check"] = {
            "corr_first_total_vs_second_proj": float(t.first_total.corr(t.second_proj)),
            "mean_second_proj_minus_first_total": float((t.second_proj - t.first_total).mean())}

    pd.concat(per_match).to_csv(args.out / "per_match.csv", index=False)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    summary["run"] = {"commit": commit, "build_fingerprint": json.load(open(GOLD / "run.json")).get(
        "build_fingerprint"), "first_test_year": args.first_test_year, "luck_file": str(args.luck),
        "pre_match_inputs": inputs, "lag_inputs": LAG, "windows_days": WINDOW_DAYS, "seed": SEED}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
