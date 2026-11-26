"""Batter, bowler and ground ratings as of any date, from earlier dates only (moved here from
test_player_ratings.py on 2026-11-19 so the simulator can use them; see that script's docstring for the method
and its pass/fail result). Prototype outside the build; reads gold.
"""

import numpy as np
import pandas as pd

from cricket_sim.models.dataset import CLASSES, OUTCOME_CLASS
from cricket_sim.paths import GOLD

T20_LEAGUES = ["IPL", "T20 Blast", "BBL", "BPL", "CPL", "PSL", "LPL", "ILT20", "SA20", "MLC"]  # as simulate_matches

HALF_LIFE_YEARS = 3.0
PSEUDO, PSEUDO_WICKET = 200, 500
PRIOR_BEFORE = 2019  # class shares for the pull toward average come from years before this
FAMILIES = {"T20": ["T20I"] + T20_LEAGUES, "ODI": ["ODI"]}
OVER_BANDS = {"T20": [-1, 5, 14, 1000], "ODI": [-1, 9, 39, 1000]}
WICKET_BANDS = [-1, 2, 5, 7, 10]
ENTITIES = {"bat": "batter_id", "bowl": "bowler_id", "venue": "venue_id"}
CLASS_NAME = {c: ("r" + c if c.isdigit() else c) for c in CLASSES}
XI = ["bat_rpb", "bat_opb", "bat_rpb_365", "bowl_rpb", "bowl_wpb", "bowl_rpb_365"]
LAMBDA = np.log(2) / (HALF_LIFE_YEARS * 365.25)
K = np.array([PSEUDO_WICKET if c == "wicket" else PSEUDO for c in CLASSES], float)
NC = len(CLASSES)


def rating_cols(e: str) -> list[str]:
    return [f"{e}_lr_{CLASS_NAME[c]}" for c in CLASSES] + [f"{e}_balls"]


def all_balls(con, family: str) -> pd.DataFrame:
    segs = FAMILIES[family]
    df = con.execute(f"""
        SELECT f.match_id, f.innings, f.delivery_number, m.start_date, year(m.start_date) AS y, m.segment,
            m.venue_id, f.batter_id, f.bowler_id, f.over, f.innings_wickets_before, f.runs_batter,
            {OUTCOME_CLASS} AS outcome_class
        FROM read_parquet('{GOLD / 'fact_delivery'}.parquet') f
        JOIN read_parquet('{GOLD / 'dim_match'}.parquet') m USING (match_id)
        WHERE m.segment IN ({', '.join('?' * len(segs))}) AND NOT f.is_super_over AND f.innings <= 2""",
                     segs).df()
    df["start_date"] = pd.to_datetime(df.start_date)
    band = np.searchsorted(OVER_BANDS[family][1:], df.over.to_numpy(), side="left")
    wk = np.searchsorted(WICKET_BANDS[1:], df.innings_wickets_before.to_numpy(), side="left")
    df["cell"] = df.innings * 100 + band * 10 + wk
    df["cls"] = df.outcome_class.map(CLASSES.index).astype(int)
    return df


def expected_probs(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(expected class probabilities per ball from earlier years' cells, NaN rows in the first year; prior p)."""
    counts = df.groupby(["y", "cell", "cls"]).size().unstack("cls", fill_value=0).reindex(columns=range(NC),
                                                                                           fill_value=0)
    prior = np.bincount(df.cls[df.y < PRIOR_BEFORE], minlength=NC).astype(float)
    prior /= prior.sum()
    exp = np.full((len(df), NC), np.nan)
    years = sorted(df.y.unique())
    for y in years[1:]:
        before = counts[counts.index.get_level_values("y") < y].groupby(level="cell").sum()
        table = (before + 0.5).div((before + 0.5).sum(axis=1), axis=0)
        rows = np.flatnonzero(df.y.to_numpy() == y)
        t = table.reindex(df.cell.to_numpy()[rows])
        vals = t.to_numpy().copy()
        missing = np.isnan(vals).any(1)
        vals[missing] = prior  # a cell never seen before: the overall shares
        exp[rows] = vals
    return exp, prior


def ratings(df: pd.DataFrame, exp: np.ndarray, prior: np.ndarray, entity: str, e: str,
            extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (entity id, date) the entity played on: its ratings from strictly earlier dates.
    extra: more (entity, start_date) rows to rate, e.g. a playing XI member who did not bat that day."""
    ok = ~np.isnan(exp).any(1) & df[entity].notna().to_numpy()
    d = df.loc[ok, [entity, "start_date"]].reset_index(drop=True)
    t = (d.start_date - pd.Timestamp("2000-01-01")).dt.days.to_numpy(float)
    w = np.exp(LAMBDA * t)
    obs = np.eye(NC)[df.cls.to_numpy()[ok]] * w[:, None]
    ex = exp[ok] * w[:, None]
    cols = [f"o{i}" for i in range(NC)] + [f"e{i}" for i in range(NC)]
    g = pd.concat([d, pd.DataFrame(np.hstack([obs, ex]), columns=cols)], axis=1) \
        .groupby([entity, "start_date"], sort=True)[cols].sum().reset_index()
    if extra is not None:
        q = extra[[entity, "start_date"]].dropna().drop_duplicates().assign(**{c: 0.0 for c in cols})
        g = pd.concat([g, q]).groupby([entity, "start_date"], sort=True)[cols].sum().reset_index()
    prev = g.groupby(entity, sort=False)[cols].cumsum() - g[cols]  # earlier dates only
    decay = np.exp(-LAMBDA * (g.start_date - pd.Timestamp("2000-01-01")).dt.days.to_numpy(float))[:, None]
    o, x = prev[cols[:NC]].to_numpy() * decay, prev[cols[NC:]].to_numpy() * decay
    kp = K * prior
    lr = np.log((o + kp) / (x + kp))
    out = pd.DataFrame(lr, columns=rating_cols(e)[:-1])
    out[f"{e}_balls"] = np.log1p(x.sum(1))
    out.insert(0, "start_date", g.start_date)
    out.insert(0, entity, g[entity])
    return out
