"""Proper scoring rules and calibration, on clean held-out balls only (docs/VALIDATION.md).

Every prediction scored here came from a model fitted on earlier seasons. Ties score as an outcome of 0.5.
"""

import numpy as np
import pandas as pd

from .dataset import CLASSES

EPS = 1e-12
WP_MODELS = {"model": "wp_model", "state_logistic": "wp_state_logistic", "base_rate": "wp_base_rate"}
OUTCOME_MODELS = {"model": "p_", "phase_average": "phase_p_", "base_rate": "base_p_"}
ERAS = [(0, 2014, "2008-2014"), (2015, 2020, "2015-2020"), (2021, 9999, "2021-")]


def _binary(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    p = np.clip(p, EPS, 1 - EPS)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))), float(np.mean((p - y) ** 2))


def _era(season: int) -> str:
    return next(name for lo, hi, name in ERAS if lo <= season <= hi)


def scores(df: pd.DataFrame) -> pd.DataFrame:
    """df: clean held-out balls with predictions, `season`, `batting_won`, `outcome_class`."""
    out = []
    y_cls = df["outcome_class"].map(CLASSES.index).to_numpy()
    groups = [("all", df.index)] + [(str(s), g.index) for s, g in df.groupby("season")] + \
             [(e, df.index[df["season"].map(_era) == e]) for e in [name for *_, name in ERAS]]
    for scope, idx in groups:
        sub = df.loc[idx]
        if sub.empty:
            continue
        for innings in (1, 2):
            s = sub[sub["innings"] == innings]
            for name, col in WP_MODELS.items():
                ok = s[col].notna()
                if ok.any():
                    ll, br = _binary(s.loc[ok, col].to_numpy(), s.loc[ok, "batting_won"].to_numpy())
                    out.append(("win_probability", f"innings_{innings}", name, scope, int(ok.sum()),
                                int(s.loc[ok, "match_id"].nunique()), ll, br))
        yi = y_cls[df.index.get_indexer(idx)]
        for name, prefix in OUTCOME_MODELS.items():
            P = np.clip(sub[[f"{prefix}{c}" for c in CLASSES]].to_numpy(), EPS, 1)
            onehot = np.eye(len(CLASSES))[yi]
            out.append(("ball_outcome", "all_classes", name, scope, len(sub), int(sub["match_id"].nunique()),
                        float(-np.mean(np.log(P[np.arange(len(sub)), yi]))), float(np.mean(((P - onehot) ** 2).sum(axis=1)))))
    cols = ["target", "slice", "model", "scope", "balls", "matches", "log_loss", "brier"]
    return pd.DataFrame(out, columns=cols).round({"log_loss": 5, "brier": 5})


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
