"""Score the plays sim_backtest.py kept, and test two ways to widen the simulator's score ranges after the sim.

Scores, per test match, then averaged over all scoring years together (2019 on by default):
  log loss      the main win score: minus the log of the chance the sim gave to what happened (a tie counts half
                each way). The sim's chance is clipped to 1% .. 99%, because a sim with a few hundred plays can give
                0 or 1, and one such miss would dominate the average. Lower is better.
  Brier         squared gap between the sim's win chance and the result (1, 0.5, 0)
  CRPS          how close the whole simulated spread of first-innings totals is to the actual total (runs)
  explained     1 minus the squared miss of the mean simulated total over the squared distance from that format's
                year mean (the measure used since 2026-11-17)
  PIT           where the actual total falls in the simulated spread: 0.3 means 30% of plays scored less. Ties are
                split at random (randomised PIT), so a well-calibrated sim gives an even histogram; a U shape means
                the ranges are too narrow. inside_50 and inside_80 are the shares of actual totals inside the
                central 50% and 80% of plays.
Gains between two runs are paired: bootstrap over test matches, all scoring years pooled, 90% band.
Monte Carlo noise: the plays of each match are split at random into two halves many times; the spread of the
half-against-half difference, scaled to the full number of plays, is the noise of a score (one standard deviation).

Range widening, fitted per format on earlier years only (every year simulated before the test year, from the run's
first year, so 2019 learns from 2016 to 2018):
  raw              the sim as it is
  scaled           each match's plays are spread out around their mean by one factor s, the s with the lowest CRPS
                   on earlier years (EMOS style, Gneiting 2005)
  scaled_shifted   as scaled, plus one shift in runs fitted with it
  conformal        PIT recalibration: the plays' quantiles are re-read at the levels earlier years' PIT values say
                   they really sit at (split conformal on PIT values). It cannot reach outside the lowest and highest
                   play.
None of these changes the win chance; only totals are rescored.
A run whose run.json lists post_stretch formats also gets method "final" (scaled for those formats, raw for the
rest), compared with the baseline's raw run.

Prototype outside the build; reads the run folders, writes only to --out (a new folder).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import simulate_matches as sm
from cricket_sim.models.train import SEED

CLIP = 0.01
METHODS = ["raw", "scaled", "scaled_shifted", "conformal"]
LOWER_IS_BETTER = {"log_loss": True, "brier": True, "crps_runs": True, "explained_share": False,
                   "inside_50": None, "inside_80": None, "outer_pit_share": None}


def load_run(folder: Path) -> dict:
    """{format: (matches, first, result)} with first and result as (matches x plays) arrays."""
    meta = pd.read_csv(folder / "matches.csv")
    out = {}
    for fmt, m in meta.groupby("format", sort=False):
        firsts, results = [], []
        for year in m.year.unique():
            z = np.load(folder / f"plays_{fmt}_{year}.npz")
            firsts.append(z["first"].astype(float))
            results.append(z["result"] / 2.0)
        out[fmt] = (m.reset_index(drop=True), np.vstack(firsts), np.vstack(results))
    return out


def pit(first: np.ndarray, actual: np.ndarray, rng) -> np.ndarray:
    below = (first < actual[:, None]).mean(1)
    at = (first == actual[:, None]).mean(1)
    return below + rng.random(len(actual)) * at


def per_match(m: pd.DataFrame, first: np.ndarray, result: np.ndarray, rng) -> pd.DataFrame:
    """One row per match: the per-match pieces every score is built from."""
    y, actual = m.bat_first_won.to_numpy(float), m.actual_first_total.to_numpy(float)
    p = result.mean(1)
    c = np.clip(p, CLIP, 1 - CLIP)
    elo = np.clip(m.elo_bat_first_win.to_numpy(float), CLIP, 1 - CLIP)
    u = pit(first, actual, rng)
    return pd.DataFrame({
        "match_id": m.match_id, "format": m.format, "year": m.year, "actual": actual, "win": p,
        "log_loss": -(y * np.log(c) + (1 - y) * np.log(1 - c)), "brier": (p - y) ** 2,
        "elo_log_loss": -(y * np.log(elo) + (1 - y) * np.log(1 - elo)), "elo_brier": (elo - y) ** 2,
        "crps_runs": sm.crps(first, actual), "sq_miss": (first.mean(1) - actual) ** 2,
        "dev2": (actual - m.groupby(["format", "year"]).actual_first_total.transform("mean").to_numpy()) ** 2,
        "pit": u, "inside_50": (np.abs(u - 0.5) <= 0.25).astype(float),
        "inside_80": (np.abs(u - 0.5) <= 0.40).astype(float),
        "outer_pit": ((u < 0.1) | (u > 0.9)).astype(float), "clipped": (p != c).astype(float)})


def summarise(t: pd.DataFrame) -> dict:
    return {"matches": len(t), "log_loss": t.log_loss.mean(), "brier": t.brier.mean(), "crps_runs": t.crps_runs.mean(),
            "explained_share": 1 - t.sq_miss.sum() / t.dev2.sum(), "inside_50": t.inside_50.mean(),
            "inside_80": t.inside_80.mean(), "outer_pit_share": t.outer_pit.mean(),
            "elo_log_loss": t.elo_log_loss.mean(), "elo_brier": t.elo_brier.mean(), "clipped_share": t.clipped.mean()}


# ---- range widening, fitted on earlier years ----

stretch, fit_scale = sm.stretch, sm.fit_scale  # moved to simulate_matches.py 2026-11-19, unchanged


def recalibrate(first: np.ndarray, cal_pit: np.ndarray) -> np.ndarray:
    """Re-read each match's sorted plays at the levels where earlier years' PIT values put them."""
    k = first.shape[1]
    levels = np.quantile(cal_pit, (np.arange(k) + 0.5) / k)
    idx = np.minimum((levels * k).astype(int), k - 1)
    return np.sort(first, axis=1)[:, idx]


def widen(m: pd.DataFrame, first: np.ndarray, first_score_year: int, rng) -> tuple[dict, list]:
    """{method: widened plays for the scoring years} and the fitted settings per year."""
    years = m.year.to_numpy()
    actual = m.actual_first_total.to_numpy(float)
    raw_pit = pit(first, actual, rng)
    out = {k: [] for k in METHODS}
    fits = []
    for year in sorted(set(years[years >= first_score_year])):
        cal, test = years < year, years == year
        s1, _ = fit_scale(first[cal], actual[cal], False)
        s2, a2 = fit_scale(first[cal], actual[cal], True)
        out["raw"].append(first[test])
        out["scaled"].append(stretch(first[test], s1))
        out["scaled_shifted"].append(stretch(first[test], s2, a2))
        out["conformal"].append(recalibrate(first[test], raw_pit[cal]))
        fits.append({"year": int(year), "calibration_matches": int(cal.sum()), "scale": s1, "scale_with_shift": s2,
                     "shift_runs": a2, "earlier_inside_80": float((np.abs(raw_pit[cal] - 0.5) <= 0.4).mean())})
    return {k: np.vstack(v) for k, v in out.items()}, fits


# ---- comparisons ----

def score_of(t: pd.DataFrame, key: str) -> float:
    return summarise(t)[key]


def paired_gain(a: pd.DataFrame, b: pd.DataFrame, keys, draws) -> dict:
    """b minus a on the same matches (same order), with the 90% band over bootstrap draws of matches."""
    out = {}
    for k in keys:
        d = [score_of(b.iloc[i], k) - score_of(a.iloc[i], k) for i in draws]
        out[k] = {"a": score_of(a, k), "b": score_of(b, k), "gain": score_of(b, k) - score_of(a, k),
                  "band": [float(x) for x in np.percentile(d, [5, 95])]}
    return out


def mc_noise(pieces: list, keys, splits: int, rng) -> dict:
    """pieces: [(sign, m, first, result), ...] whose signed scores are summed (one run, or b minus a). The plays are
    split at random into halves; the half-against-half difference, over 2, estimates the noise of the full run."""
    diffs = {k: [] for k in keys}
    for _ in range(splits):
        half = {}
        for j, (sign, m, first, result) in enumerate(pieces):
            k_plays = first.shape[1]
            perm = rng.permutation(k_plays)
            h1, h2 = perm[: k_plays // 2], perm[k_plays // 2:]
            for name, h in (("1", h1), ("2", h2)):
                half.setdefault(name, []).append((sign, summarise(per_match(m, first[:, h], result[:, h],
                                                                            np.random.default_rng(j)))))
        for k in keys:
            v = {name: sum(sign * s[k] for sign, s in lst) for name, lst in half.items()}
            diffs[k].append(v["1"] - v["2"])
    return {k: float(np.sqrt(np.mean(np.square(d))) / 2) for k, d in diffs.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, required=True, help="folder holding one sub-folder per variant")
    ap.add_argument("--variants", default="base,t20i_only")
    ap.add_argument("--baseline", default="base")
    ap.add_argument("--first-score-year", type=int, default=2019)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(SEED)
    runs = {v: load_run(args.runs / v) for v in args.variants.split(",") if (args.runs / v / "matches.csv").exists()}
    keys = ["log_loss", "brier", "crps_runs", "explained_share", "inside_50", "inside_80", "outer_pit_share"]
    total_keys = ["crps_runs", "explained_share", "inside_50", "inside_80", "outer_pit_share"]

    scores, gains, pit_rows, fits_rows, tables = [], [], [], [], {}
    for variant, data in runs.items():
        for fmt, (m, first, result) in data.items():
            keep = (m.year >= args.first_score_year).to_numpy()
            ms, fs, rs = m[keep].reset_index(drop=True), first[keep], result[keep]
            widened, fits = widen(m, first, args.first_score_year, np.random.default_rng(SEED))
            fits_rows += [{"variant": variant, "format": fmt, **f} for f in fits]
            for method in METHODS:
                t = per_match(ms, widened[method], rs, np.random.default_rng(SEED))
                tables[(variant, fmt, method)] = (t, ms, widened[method], rs)
                noise = mc_noise([(1, ms, widened[method], rs)], keys, args.splits, rng) if method == "raw" else {}
                scores.append({"variant": variant, "format": fmt, "method": method, "plays": first.shape[1],
                               **summarise(t), **{f"noise_{k}": v for k, v in noise.items()}})
                hist = np.histogram(t.pit, bins=10, range=(0, 1))[0] / len(t)
                pit_rows.append({"variant": variant, "format": fmt, "method": method,
                                 **{f"{10 * i}-{10 * i + 10}%": h for i, h in enumerate(hist)}})
                print(f"{variant:10s} {fmt:6s} {method:15s} " + " ".join(
                    f"{k} {v:.4f}" for k, v in summarise(t).items() if k != "matches"), flush=True)

    # T20 pooled (T20I and league matches together), same per-match rows
    for variant in runs:
        for method in METHODS:
            parts = [tables[(variant, f, method)][0] for f in ("T20I", "league") if (variant, f, method) in tables]
            if len(parts) == 2:
                t = pd.concat(parts, ignore_index=True)
                scores.append({"variant": variant, "format": "T20 pooled", "method": method, **summarise(t)})

    def compare(label, a_key, b_key, ks):
        ta, ma, fa, ra = tables[a_key]
        tb, mb, fb, rb = tables[b_key]
        common = ta.match_id.isin(tb.match_id).to_numpy()
        ia = np.flatnonzero(common)
        ib = tb.reset_index().set_index("match_id").loc[ta.match_id[common], "index"].to_numpy()
        A, B = ta.iloc[ia].reset_index(drop=True), tb.iloc[ib].reset_index(drop=True)
        n = len(A)
        draws = [rng.integers(0, n, n) for _ in range(args.boot)]
        g = paired_gain(A, B, ks, draws)
        noise = mc_noise([(-1, ma.iloc[ia].reset_index(drop=True), fa[ia], ra[ia]),
                          (1, mb.iloc[ib].reset_index(drop=True), fb[ib], rb[ib])], ks, args.splits, rng)
        for k, v in g.items():
            gains.append({"comparison": label, "format": a_key[1], "matches": n, "score": k, **v,
                          "mc_noise": noise[k], "lower_is_better": LOWER_IS_BETTER[k]})
        print(f"{label} {a_key[1]}: " + "; ".join(f"{k} {v['gain']:+.4f} [{v['band'][0]:+.4f}, {v['band'][1]:+.4f}]"
                                                   for k, v in g.items()), flush=True)

    # a run that asks for some formats' totals to be stretched after the sim (run.json post_stretch) is also scored
    # as it would be used: method "final" is scaled for those formats and raw for the rest
    for variant in runs:
        post = json.loads((args.runs / variant / "run.json").read_text()).get("post_stretch")
        if not post:
            continue
        for fmt in runs[variant]:
            tables[(variant, fmt, "final")] = tables[(variant, fmt, "scaled" if fmt in post else "raw")]
            scores.append({"variant": variant, "format": fmt, "method": "final",
                           **summarise(tables[(variant, fmt, "final")][0])})
        parts = [tables[(variant, f, "final")][0] for f in ("T20I", "league") if (variant, f, "final") in tables]
        if len(parts) == 2:
            scores.append({"variant": variant, "format": "T20 pooled", "method": "final",
                           **summarise(pd.concat(parts, ignore_index=True))})
        for fmt in runs[variant]:
            if (args.baseline, fmt, "raw") in tables:
                compare(f"{variant} final vs {args.baseline}", (args.baseline, fmt, "raw"), (variant, fmt, "final"),
                        keys)
    # when the baseline has its own "final" too, compare final against final: both as they would be used
    for variant in runs:
        if variant == args.baseline:
            continue
        for fmt in runs[variant]:
            if (variant, fmt, "final") in tables and (args.baseline, fmt, "final") in tables:
                compare(f"{variant} final vs {args.baseline} final", (args.baseline, fmt, "final"),
                        (variant, fmt, "final"), keys)

    base = args.baseline
    for variant in runs:
        if variant == base:
            continue
        for fmt in runs[variant]:
            if (base, fmt, "raw") in tables:
                compare(f"{variant} vs {base}", (base, fmt, "raw"), (variant, fmt, "raw"), keys)
    for variant in runs:  # the sim's win chance against Elo alone, same matches
        for fmt in runs[variant]:
            t = tables[(variant, fmt, "raw")][0]
            n = len(t)
            for k in ("log_loss", "brier"):
                d = (t[k] - t[f"elo_{k}"]).to_numpy()
                boot = [d[rng.integers(0, n, n)].mean() for _ in range(args.boot)]
                gains.append({"comparison": f"{variant} sim vs Elo alone", "format": fmt, "matches": n, "score": k,
                              "a": t[f"elo_{k}"].mean(), "b": t[k].mean(), "gain": d.mean(),
                              "band": [float(x) for x in np.percentile(boot, [5, 95])], "lower_is_better": True})
                print(f"{variant} sim vs Elo {fmt} {k}: {d.mean():+.4f} "
                      f"[{np.percentile(boot, 5):+.4f}, {np.percentile(boot, 95):+.4f}]", flush=True)
    if base in runs:
        for fmt in runs[base]:
            for method in METHODS[1:]:
                compare(f"{method} vs raw ({base})", (base, fmt, "raw"), (base, fmt, method), total_keys)

    pd.DataFrame(scores).to_csv(args.out / "scores.csv", index=False)
    pd.DataFrame(gains).to_csv(args.out / "gains.csv", index=False)
    pd.DataFrame(pit_rows).to_csv(args.out / "pit_histogram.csv", index=False)
    pd.DataFrame(fits_rows).to_csv(args.out / "widening_fits.csv", index=False)
    runs_meta = {v: json.loads((args.runs / v / "run.json").read_text()) for v in runs}
    (args.out / "summary.json").write_text(json.dumps({
        "runs": runs_meta, "first_score_year": args.first_score_year, "boot": args.boot, "splits": args.splits,
        "clip": CLIP, "scores": scores, "gains": gains}, indent=2, default=float))


if __name__ == "__main__":
    main()
