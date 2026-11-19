"""How much of the spread in first-innings totals is luck that no pre-match input could ever explain?

The luck part is the spread a total would still have if the chance of every ball outcome were known exactly.
Two independent estimates of it, on the same clean full-length matches as the simulator:

  hindsight  Each real first innings is replayed N times. A ball's chances are what all first innings did in
             that phase (ODI overs 1-10, 11-40, 41-50; T20I overs 1-6, 7-15, 16-20) with that many wickets down,
             times this innings' own lift: how much more or less often each outcome came in this innings than
             expected for the balls it faced, per phase, shrunk a little towards 1. The odds come from what
             actually happened, so nothing about the match is unknown; what still varies between replays is luck.
             No model involved.
  model      The simulator (simulate_matches.py: venue and playing-XI inputs, pitch factor OFF) fitted on the ten
             years before each test year plays each test match N times. The spread of its totals is the luck
             under the model's ball odds.

Luck share = mean variance of the replays / mean squared distance of the actual totals from their test year's own
mean (the same denominator as measure_explained_share.py). 1 minus the luck share is the most that any input
known before the match could explain: the ceiling.

How the two differ: most of the luck comes from wickets, because a side that loses early wickets then scores
slower, so one bad over spreads over the whole innings. The answer moves with how strongly the odds react to the
state of the innings: a first hindsight version with no wicket effect gave a luck share of only 28% (ODI) and 37%
(T20I) on 2019-2026 (run 2026-11-17). The hindsight replay reacts to wickets and phase only; the model also reacts
to run rate and the exact ball, so it spreads wider. Read the two as a range, hindsight the lower end.

Prototype outside the build; it reads gold and writes only to --out (a new folder).
"""

import argparse
import json
import subprocess
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

from cricket_sim.models.dataset import CLASSES
from cricket_sim.models.train import SEED
from cricket_sim.paths import GOLD
from simulate_matches import WICKET, fit, full_length, league_balls, load, simulate  # scripts/ is on sys.path

PHASE_CUTS = {"ODI": np.array([60, 240]), "T20I": np.array([36, 90])}  # legal balls bowled before the ball
PRIOR_BALLS = 30  # pseudo-balls at the format's rates when shrinking an innings' own lift
WICKET_CUTS = np.array([3, 5, 7])  # wickets down before the ball: 0-2, 3-4, 5-6, 7+


def run_draws(balls: pd.DataFrame) -> list:
    """Per outcome class, the (runs, legal) pairs seen in these balls and their shares."""
    draws = []
    for cls in CLASSES:
        pairs = balls[balls.outcome_class == cls].groupby(["runs_total", "is_legal"]).size()
        draws.append((pairs.index.get_level_values(0).to_numpy(), pairs.index.get_level_values(1).to_numpy().astype(int),
                      (pairs / pairs.sum()).to_numpy()))
    return draws


def wicket_bucket(wkts: np.ndarray) -> np.ndarray:
    return np.searchsorted(WICKET_CUTS, wkts, side="right")


def hindsight_odds(balls: pd.DataFrame, ids: np.ndarray, cuts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (base, lift). base: (phases x wicket buckets x classes) outcome chances over all these first
    innings, so a side that has lost wickets scores the way sides in that state do. lift: (matches x phases x
    classes) how much more or less often each class came in this innings than base expected for the balls it
    actually faced, shrunk towards 1 by PRIOR_BALLS pseudo-balls. A replay ball's chances are base x lift."""
    b = balls[(balls.innings == 1) & balls.match_id.isin(ids)]
    ph = np.searchsorted(cuts, b.legal_balls_before.to_numpy(), side="right")
    wb = wicket_bucket(b.innings_wickets_before.to_numpy())
    cls = b.outcome_class.map(CLASSES.index).to_numpy()
    n_ph, n_wb, n_cls = len(cuts) + 1, len(WICKET_CUTS) + 1, len(CLASSES)
    pooled = np.zeros((n_ph, n_wb, n_cls))
    np.add.at(pooled, (ph, wb, cls), 1)
    by_phase = pooled.sum(1, keepdims=True)
    base = pooled + 50 * by_phase / by_phase.sum(2, keepdims=True)  # thin cells lean on the phase's rates
    base /= base.sum(2, keepdims=True)
    row = pd.Series(np.arange(len(ids)), index=ids)[b.match_id].to_numpy()
    counts = np.zeros((len(ids), n_ph, n_cls))
    expect = np.zeros((len(ids), n_ph, n_cls))
    np.add.at(counts, (row, ph, cls), 1)
    np.add.at(expect, (row, ph), base[ph, wb])
    share = base.mean(1)[None]  # pseudo-balls spread like an average ball of the phase
    lift = (counts + PRIOR_BALLS * share) / (expect + PRIOR_BALLS * share)
    return base, lift


def replay(base: np.ndarray, lift: np.ndarray, limit: np.ndarray, draws: list, cuts: np.ndarray, n: int,
           rng) -> np.ndarray:
    """n replays of every innings at once; each ball's chances are base[phase, wickets] x lift[innings, phase].
    Returns (matches x n) totals."""
    m = len(limit)
    who = np.repeat(np.arange(m), n)
    lim = np.repeat(limit, n)
    runs, wkts, legal = (np.zeros(m * n, int) for _ in range(3))
    live = np.ones(m * n, bool)
    while live.any():
        i = np.flatnonzero(live)
        ph = np.searchsorted(cuts, legal[i], side="right")
        p = base[ph, wicket_bucket(wkts[i])] * (lift[who[i], ph] if lift is not None else 1.0)
        p /= p.sum(1, keepdims=True)
        cls = np.minimum((p.cumsum(1) < rng.random(len(i))[:, None]).sum(1), len(CLASSES) - 1)
        for k in np.unique(cls):
            rows = i[cls == k]
            r, lg, w = draws[k]
            pick = rng.choice(len(w), size=len(rows), p=w)
            runs[rows] += r[pick]
            legal[rows] += lg[pick]
            if k == WICKET:
                wkts[rows] += 1
        live &= ~((wkts >= 10) | (legal >= lim))
    return runs.reshape(m, n)


def shares(t: pd.DataFrame, rng, boot: int = 300) -> dict:
    """Luck shares (and the model's explained share) with 90% bootstrap bands over matches."""
    t = t.assign(dev2=(t.actual - t.groupby("year").actual.transform("mean")) ** 2)

    def calc(d: pd.DataFrame) -> dict:
        out = {"luck_share_hindsight": d.hind_var.sum() / d.dev2.sum()}
        if d.model_var.notna().all():
            out["luck_share_model"] = d.model_var.sum() / d.dev2.sum()
            out["model_explained_share"] = 1 - ((d.actual - d.model_mean) ** 2).sum() / d.dev2.sum()
            out["model_error_that_is_luck"] = d.model_var.sum() / ((d.actual - d.model_mean) ** 2).sum()
        return out

    point = calc(t)
    draws = [calc(t.iloc[rng.integers(0, len(t), len(t))]) for _ in range(boot)]
    res = {"matches": len(t), "spread_sd_runs": float(np.sqrt(t.dev2.mean())),
           "luck_sd_runs_hindsight": float(np.sqrt(t.hind_var.mean()))}
    if "luck_share_model" in point:
        res["luck_sd_runs_model"] = float(np.sqrt(t.model_var.mean()))
    for k, v in point.items():
        lo, hi = np.percentile([d[k] for d in draws], [5, 95])
        res[k] = float(v)
        res[f"{k}_band"] = [float(lo), float(hi)]
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--first-test-year", type=int, default=2019)
    ap.add_argument("--replays", type=int, default=1000, help="hindsight replays per innings")
    ap.add_argument("--model-years", default="2023,2024,2025,2026", help="test years for the model estimate")
    ap.add_argument("--model-sims", type=int, default=300, help="simulator plays per match")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(SEED)
    members = yaml.safe_load(Path("config/scope.yml").read_text())["full_members"]
    con = duckdb.connect()
    con.execute("SET memory_limit = '3GB'")
    model_years = [int(y) for y in args.model_years.split(",") if y]
    rows, summary = [], {}

    for segment in ("ODI", "T20I"):
        balls, matches = load(con, segment)
        matches["year"] = pd.to_datetime(matches.start_date).dt.year
        test = full_length(matches[matches.year >= args.first_test_year]).reset_index(drop=True)
        cuts = PHASE_CUTS[segment]
        print(f"{segment}: {len(test)} test matches from {args.first_test_year}", flush=True)

        base, lift = hindsight_odds(balls, test.match_id.to_numpy(), cuts)
        hind = replay(base, lift, test.balls_limit.to_numpy(int), run_draws(balls), cuts, args.replays, rng)
        t = pd.DataFrame({"segment": segment, "match_id": test.match_id, "start_date": test.start_date,
                          "year": test.year, "full_members": test.team1.isin(members) & test.team2.isin(members),
                          "actual": test.first_total.astype(float), "hind_mean": hind.mean(1),
                          "hind_var": hind.var(1), "model_mean": np.nan, "model_var": np.nan})
        print(f"  hindsight done: replay mean minus actual {np.mean(t.hind_mean - t.actual):+.1f} runs", flush=True)

        # the T20I ball model also learns from league balls (simulate_matches.py, from 2026-11-18); the hindsight
        # replay above uses the T20I balls only
        pool = pd.concat([balls, league_balls(con)], ignore_index=True) if segment == "T20I" else balls
        for y in model_years:
            sel = (t.year == y).to_numpy()
            if not sel.any():
                continue
            model = fit(pool, pd.Timestamp(f"{y}-01-01"), None)
            s = simulate(model, test[sel], args.model_sims, rng, 1.0, 0.0)
            t.loc[sel, "model_mean"] = s["first"].mean(1)
            t.loc[sel, "model_var"] = s["first"].var(1)
            print(f"  model {y}: {sel.sum()} matches", flush=True)

        rows.append(t)
        summary[segment] = {}
        for name, d in (("all_years", t), ("model_years", t[t.year.isin(model_years)])):
            for subset in ("all", "full_members"):
                dd = d if subset == "all" else d[d.full_members]
                if len(dd) < 20 or (subset == "full_members" and segment == "T20I"):
                    continue  # T20Is in gold are full-member matches already
                summary[segment][f"{name}_{subset}"] = shares(dd if name == "model_years" else
                                                              dd.assign(model_var=np.nan), rng)
                print(f"  {name} {subset}: {json.dumps(summary[segment][f'{name}_{subset}'])}", flush=True)

    pd.concat(rows).to_csv(args.out / "per_match.csv", index=False)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    summary["run"] = {"commit": commit, "build_fingerprint": json.loads((GOLD / "run.json").read_text()).get(
        "build_fingerprint"), "first_test_year": args.first_test_year, "replays": args.replays,
        "model_years": model_years, "model_sims": args.model_sims, "prior_balls": PRIOR_BALLS, "seed": SEED}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary["run"]))


if __name__ == "__main__":
    main()
