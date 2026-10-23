"""Proper scoring rules and calibration, on clean held-out balls only (docs/VALIDATION.md).

Every prediction scored here came from a model fitted on earlier seasons. Ties score as an outcome of 0.5.
Every score carries a 95% interval from a bootstrap over matches (the balls of one match share its result, so
balls are not the independent unit), and `paired` scores the model against each baseline on the same balls:
the mean difference with its interval, and a two-sided sign test over seasons.
"""

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .dataset import CLASSES

EPS = 1e-12
WP_MODELS = {"model": "wp_model", "state_logistic": "wp_state_logistic", "base_rate": "wp_base_rate"}
OUTCOME_MODELS = {"model": "p_", "phase_average": "phase_p_", "player_venue_average": "pv_p_", "base_rate": "base_p_"}
PAIRED_BASELINES = {"win_probability": ["state_logistic", "base_rate"],
                    "ball_outcome": ["phase_average", "player_venue_average", "base_rate"]}
ERAS = [(0, 2014, "2008-2014"), (2015, 2020, "2015-2020"), (2021, 9999, "2021-")]
BOOT_RESAMPLES, BOOT_SEED = 1000, 20261023


def _binary(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    p = np.clip(p, EPS, 1 - EPS)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))), float(np.mean((p - y) ** 2))


def _era(season: int) -> str:
    return next(name for lo, hi, name in ERAS if lo <= season <= hi)


def _wp_loss(df: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-ball log loss and Brier of a win-probability column (NaN where the column is NaN)."""
    p = np.clip(df[col].to_numpy(dtype=float), EPS, 1 - EPS)
    y = df["batting_won"].to_numpy(dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)), (p - y) ** 2


def _outcome_loss(df: pd.DataFrame, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-ball multiclass log loss and Brier (summed over classes) of an outcome-probability block."""
    yi = df["outcome_class"].map(CLASSES.index).to_numpy()
    P = np.clip(df[[f"{prefix}{c}" for c in CLASSES]].to_numpy(dtype=float), EPS, 1)
    onehot = np.eye(len(CLASSES))[yi]
    return -np.log(P[np.arange(len(df)), yi]), ((P - onehot) ** 2).sum(axis=1)


def _by_match(match_id: np.ndarray, ll: np.ndarray, br: np.ndarray) -> pd.DataFrame:
    g = pd.DataFrame({"m": match_id, "ll": ll, "br": br}).groupby("m")
    out = g[["ll", "br"]].sum()
    out["n"] = g.size()
    return out


def _interval(sums: np.ndarray, n: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """2.5th and 97.5th percentile of the ball-weighted mean under BOOT_RESAMPLES resamples of whole matches."""
    m = len(n)
    if m < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, m, size=(BOOT_RESAMPLES, m))
    means = sums[idx].sum(axis=1) / n[idx].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _scopes(df: pd.DataFrame, seasons: bool = True) -> list[tuple[str, pd.Index]]:
    groups = [("all", df.index)]
    if seasons:
        groups += [(str(s), g.index) for s, g in df.groupby("season")]
    return groups + [(e, df.index[df["season"].map(_era) == e]) for e in [name for *_, name in ERAS]]


def scores(df: pd.DataFrame) -> pd.DataFrame:
    """df: clean held-out balls with predictions, `season`, `batting_won`, `outcome_class`. One row per target x
    slice x model x scope with the ball-weighted log loss and Brier and their match-bootstrap 95% intervals."""
    out, rng = [], np.random.default_rng(BOOT_SEED)
    losses = {("win_probability", name): _wp_loss(df, col) for name, col in WP_MODELS.items()}
    losses.update({("ball_outcome", name): _outcome_loss(df, prefix) for name, prefix in OUTCOME_MODELS.items()})
    for scope, idx in _scopes(df):
        pos = df.index.get_indexer(idx)
        sub = df.iloc[pos]
        if sub.empty:
            continue
        for innings in (1, 2):
            inn = (sub["innings"] == innings).to_numpy()
            for name in WP_MODELS:
                ll, br = (a[pos][inn] for a in losses[("win_probability", name)])
                ok = ~np.isnan(ll)
                if ok.any():
                    by = _by_match(sub["match_id"].to_numpy()[inn][ok], ll[ok], br[ok])
                    out.append(("win_probability", f"innings_{innings}", name, scope, int(ok.sum()), len(by),
                                float(ll[ok].mean()), float(br[ok].mean()),
                                *_interval(by["ll"].to_numpy(), by["n"].to_numpy(), rng),
                                *_interval(by["br"].to_numpy(), by["n"].to_numpy(), rng)))
        for name in OUTCOME_MODELS:
            ll, br = (a[pos] for a in losses[("ball_outcome", name)])
            by = _by_match(sub["match_id"].to_numpy(), ll, br)
            out.append(("ball_outcome", "all_classes", name, scope, len(sub), len(by), float(ll.mean()), float(br.mean()),
                        *_interval(by["ll"].to_numpy(), by["n"].to_numpy(), rng),
                        *_interval(by["br"].to_numpy(), by["n"].to_numpy(), rng)))
    cols = ["target", "slice", "model", "scope", "balls", "matches", "log_loss", "brier",
            "log_loss_lo", "log_loss_hi", "brier_lo", "brier_hi"]
    return pd.DataFrame(out, columns=cols).round({c: 5 for c in cols[6:]})


def paired(df: pd.DataFrame) -> pd.DataFrame:
    """The model against each baseline on the same balls: mean per-ball difference (model minus baseline, so
    negative means the model is better) with its match-bootstrap interval, and a two-sided sign test on the
    number of seasons in which the model's Brier is lower. Scopes: all and the eras."""
    out, rng = [], np.random.default_rng(BOOT_SEED)
    for scope, idx in _scopes(df, seasons=False):
        sub = df.loc[idx]
        if sub.empty:
            continue
        for innings in (1, 2):
            s = sub[sub["innings"] == innings]
            ll_m, br_m = _wp_loss(s, WP_MODELS["model"])
            for baseline in PAIRED_BASELINES["win_probability"]:
                ll_b, br_b = _wp_loss(s, WP_MODELS[baseline])
                ok = ~(np.isnan(ll_m) | np.isnan(ll_b))
                if ok.any():
                    out.append(_paired_row("win_probability", f"innings_{innings}", baseline, scope, s[ok],
                                           ll_m[ok] - ll_b[ok], br_m[ok] - br_b[ok], rng))
        ll_m, br_m = _outcome_loss(sub, OUTCOME_MODELS["model"])
        for baseline in PAIRED_BASELINES["ball_outcome"]:
            ll_b, br_b = _outcome_loss(sub, OUTCOME_MODELS[baseline])
            out.append(_paired_row("ball_outcome", "all_classes", baseline, scope, sub, ll_m - ll_b, br_m - br_b, rng))
    cols = ["target", "slice", "baseline", "scope", "balls", "matches", "seasons", "seasons_model_better",
            "sign_test_p", "brier_diff", "brier_diff_lo", "brier_diff_hi", "log_loss_diff", "log_loss_diff_lo",
            "log_loss_diff_hi"]
    return pd.DataFrame(out, columns=cols).round({c: 5 for c in cols[8:]})


def _paired_row(target, slice_, baseline, scope, s, dll, dbr, rng) -> tuple:
    by = _by_match(s["match_id"].to_numpy(), dll, dbr)
    by_season = pd.DataFrame({"season": s["season"].to_numpy(), "d": dbr}).groupby("season")["d"].mean()
    seasons, better = len(by_season), int((by_season < 0).sum())
    decided = int((by_season != 0).sum())  # a season where both score the same is a tie: dropped from the test
    p = float(binomtest(better, decided, 0.5).pvalue) if decided else 1.0
    return (target, slice_, baseline, scope, len(s), len(by), seasons, better, p,
            float(dbr.mean()), *_interval(by["br"].to_numpy(), by["n"].to_numpy(), rng),
            float(dll.mean()), *_interval(by["ll"].to_numpy(), by["n"].to_numpy(), rng))


def calibration(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Reliability table: predicted vs observed, in equal-width probability bins, per innings and era."""
    out = []
    df = df.assign(era=df["season"].map(_era))
    for (innings, era), g in list(df.groupby(["innings", "era"])) + [((i, "all"), df[df["innings"] == i]) for i in (1, 2)]:
        for name, col in WP_MODELS.items():
            s = g[g[col].notna()]
            b = np.minimum((s[col] * bins).astype(int), bins - 1)
            for k, h in s.groupby(b):
                out.append(("win_probability", f"innings_{innings}", name, era, int(k), len(h),
                            float(h[col].mean()), float(h["batting_won"].mean())))
    for cls in ("wicket", "4", "6", "dot"):
        col = f"p_{cls}"
        b = np.minimum((df[col] * bins * 2.5).astype(int), bins - 1)  # finer bins: these rarely exceed 40%
        for k, h in df.groupby(b):
            out.append(("ball_outcome", cls, "model", "all", int(k), len(h), float(h[col].mean()),
                        float((h["outcome_class"] == cls).mean())))
    cols = ["target", "slice", "model", "era", "bin", "balls", "mean_predicted", "observed_rate"]
    return pd.DataFrame(out, columns=cols).round({"mean_predicted": 5, "observed_rate": 5})


def checkpoints(df: pd.DataFrame) -> pd.DataFrame:
    """Match-level Brier at fixed moments: first ball, first ball of the chase, and with 30 balls of the chase
    left (the last 5 overs of an ODI). One prediction per match per checkpoint, from the batting side's view
    converted to the side batting first, so every match counts once."""
    df = df.sort_values(["match_id", "innings", "delivery_number"])
    first = df[df["innings"] == 1].groupby("match_id").head(1).assign(checkpoint="start_of_match")
    chase = df[df["innings"] == 2]
    half = chase.groupby("match_id").head(1).assign(checkpoint="start_of_chase")
    late = chase[chase["scheduled_balls_remaining_before"] <= 30].groupby("match_id").head(1).assign(checkpoint="last_5_overs")
    out = []
    for g in (first, half, late):
        for name, col in WP_MODELS.items():
            s = g[g[col].notna()]
            # express everything as the probability that the side batting first wins
            p = np.where(s["innings"] == 1, s[col], 1 - s[col])
            y = np.where(s["innings"] == 1, s["batting_won"], 1 - s["batting_won"])
            ll, br = _binary(p, y)
            out.append((g["checkpoint"].iloc[0] if len(g) else "", name, len(s), ll, br))
    return pd.DataFrame(out, columns=["checkpoint", "model", "matches", "log_loss", "brier"]).round({"log_loss": 5, "brier": 5})
