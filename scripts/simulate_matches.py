"""Simulate past ODIs and T20Is ball by ball, then compare each event's simulated chance with how often it
happened in history. A prototype that reads gold; it is not part of the build, its contracts or the export.

How it works:
- One ball-outcome model per format (dot, 1, 2, 3, 4, 6, wicket, extra), the same gradient-boosted classifier
  and inputs as `cricket model` (train.OUTCOME_FEATURES, which include the pre-match Elo gap). It is fitted only on
  clean matches from the ten years before --train-before, so every simulated match is later than its training.
- Elo is the repo's own pre-match Elo (models/elo.py), computed per format from earlier dates only.
- A match is played out ball by ball N times: innings 1 to the end, then the chase of that total + 1. The side
  that actually batted first bats first (the toss is taken as known). Runs on a ball and whether it was legal are
  drawn from what that outcome class gave in the training balls (so a wide can carry 5 runs, a wicket a run out).
- Two settings are tuned on the year before --train-before (fit on the ten years before that year, play that
  year), never on the test period: a half-life that weights recent seasons more (judged on first-innings
  totals), then a multiplier on the Elo gap the ball model sees (judged on who won). On 2025 (run 2026-11-17)
  neither helped: equal weights and the full Elo gap scored best in both formats, so both stay at their defaults.
- A pitch factor per play (shared by both innings) tilts every ball towards or away from runs, so some plays are
  flat pitches and some are slow ones. Its spread is tuned on the same year, judged on first-innings totals.
- Venue and playing XI (added 2026-11-17): every ball also sees, from the batting side's point of view, how far
  first innings at this ground ran above or below the format level in the 8 years before, whether the batting or
  bowling side is at home, the batting XI's career and last-365-day runs and outs per ball (top 7), and the
  bowling XI's runs and wickets per ball (top 6). All from earlier matches only (measure_explained_share.py has
  the code). The backtest's "before" is the earlier simulator without these inputs.
- League balls (added 2026-11-18): the T20I ball model also learns from every ball of the IPL and the nine
  leagues in its ten training years (about 4.5 times the T20I balls), with an input is_international (1 on T20I
  balls, 0 on league balls); each league's Elo, level and ground history are its own. On 2019-2026 T20Is this
  raised the simulator's explained share of first-innings totals from 2.6% to 11.5% and brought its 80% range
  from 67% to 77% of actual totals (measure_sim_league_gain.py). --no-t20-leagues turns it off.
- Settings from the 2026-11-19 scoring work (sims/2026-11-18-sim-scoring/REPORT-combined-2026-11-19.md), now the
  defaults: the ball model also sees in-play inputs (RECENT: runs and boundaries off the last 6 balls, balls the
  striker has faced); the pitch spread is fixed per format (T20I 0.025, ODI 0) instead of tuned on one year, which
  proved noisy; and ODI first-innings totals are stretched around each match's mean by one factor fitted on the
  --stretch-years before --train-before (walk forward: each year played by a model fitted on the ten years before
  it, the factor with the lowest CRPS). The stretch changes only the totals and their ranges, never the win chance.
  Walk-forward 2019-2026 against the simulator before: T20I win log loss 0.659 to 0.648, ODI CRPS 37.68 to 37.20 runs,
  ODI 80% range coverage 67% to 77%. --no-recent, --pitch-sd, --pitch-sds and --stretch-formats '' give the old
  behaviour back.
- Not modelled: single players' events, batting order, the pitch itself, weather, rain rules. The 10-wicket limit
  ignores absent batters.

Outputs (in --out): events.csv (one row per match and event), backtest.csv (every clean match from
--train-before on, before and after the tuned settings), tuning.csv and summary.json.
"""

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

import duckdb
import yaml
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier

from cricket_sim.models.dataset import CLASSES, OUTCOME_CLASS
from cricket_sim.models.elo import expected, pre_match_ratings
from cricket_sim.models.train import OUTCOME_FEATURES, OUTCOME_SEASONS, SEED
from cricket_sim.paths import GOLD
import measure_explained_share as mes  # scripts/ is on sys.path
import player_ratings as pr
from measure_explained_share import add_level_and_venue, add_xi, first_innings

BIG_TOTAL = {"ODI": 300, "T20I": 180}
FORMAT_CODE = {"T20I": 1, "league": 2, "ODI": 3}  # generator seeds, shared with sim_backtest.py
WICKET = CLASSES.index("wicket")
# Pre-match inputs the context model adds to every ball, from the batting side's point of view.
CONTEXT = ["venue_dev", "bat_home", "bowl_home", "bat_rpb", "bat_opb", "bat_rpb_365",
           "bowl_rpb", "bowl_wpb", "bowl_rpb_365", "is_international"]
# In-play inputs (on by default since 2026-11-19, --no-recent turns them off): runs and boundaries off the last 6 deliveries of the innings,
# and legal balls the striker has faced so far (a set batter). The simulator tracks all three as it plays.
RECENT = ["last6_runs", "last6_boundaries", "striker_balls"]
BOUNDARY = [CLASSES.index("4"), CLASSES.index("6")]
# Leagues whose balls also train the T20I ball model (one segment each in config/scope.yml)
T20_LEAGUES = ["IPL", "T20 Blast", "BBL", "BPL", "CPL", "PSL", "LPL", "ILT20", "SA20", "MLC"]
assert T20_LEAGUES == pr.T20_LEAGUES
# Player ratings (added 2026-11-19, sims/2026-11-19-player-ratings/REPORT.md): the batter on strike, the bowler of
# the over and the ground, as of the match date. With ratings on, they replace the six XI-average inputs.
RATING_BAT, RATING_BOWL, RATING_VENUE = (pr.rating_cols(e) for e in ("bat", "bowl", "venue"))
CONTEXT_NO_XI = [c for c in CONTEXT if c not in pr.XI]
SLOTS = 11  # batting order length the simulator carries

def load(con: duckdb.DuckDBPyConnection, segment: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(balls, matches) for one format: every delivery of the first two innings, and one row per match."""
    for t in ("fact_delivery", "fact_innings", "dim_match"):
        con.execute(f"CREATE OR REPLACE VIEW {t} AS SELECT * FROM read_parquet('{GOLD / t}.parquet')")
    fixtures = con.execute("""SELECT match_id, start_date, team1, team2, winner, result FROM dim_match
                              WHERE segment = ?""", [segment]).fetchall()
    con.execute("CREATE OR REPLACE TEMP TABLE elo_df (match_id VARCHAR, team VARCHAR, elo_before DOUBLE, "
                "rated_before INTEGER)")
    con.executemany("INSERT INTO elo_df VALUES (?, ?, ?, ?)", pre_match_ratings(fixtures))
    clean = """
        SELECT m.match_id,
            m.method IS NULL AND m.passes_quality AND (m.winner IS NOT NULL OR m.result = 'tie')
            AND bool_and(i.innings_length_status = 'full_known')
            AND bool_and(i.innings = 1 OR i.target_status = 'set') AS is_clean
        FROM dim_match m JOIN fact_innings i ON i.match_id = m.match_id AND NOT i.is_super_over AND i.innings <= 2
        WHERE m.segment = $seg GROUP BY m.match_id, m.method, m.passes_quality, m.winner, m.result"""
    balls = con.execute(f"""
        WITH clean AS ({clean})
        SELECT f.match_id, f.innings, f.delivery_number, m.start_date, year(m.start_date) AS year,
            f.innings_runs_before, f.innings_wickets_before, f.legal_balls_before,
            f.scheduled_balls_remaining_before, f.nominal_runs_required_before,
            CAST(6 * f.innings_runs_before AS DOUBLE) / nullif(f.legal_balls_before, 0) AS run_rate,
            CAST(6 * f.nominal_runs_required_before AS DOUBLE) / nullif(f.scheduled_balls_remaining_before, 0)
                AS required_rate,
            eb.elo_before - ew.elo_before AS elo_diff,
            {OUTCOME_CLASS} AS outcome_class, f.runs_total, f.is_legal, f.batter_id
        FROM fact_delivery f JOIN dim_match m USING (match_id) JOIN clean c USING (match_id)
        LEFT JOIN elo_df eb ON eb.match_id = f.match_id AND eb.team = f.batting_team
        LEFT JOIN elo_df ew ON ew.match_id = f.match_id
            AND ew.team = (CASE WHEN f.batting_team = m.team1 THEN m.team2 ELSE m.team1 END)
        WHERE m.segment = $seg AND c.is_clean AND NOT f.is_super_over AND f.innings <= 2
        ORDER BY f.match_id, f.innings, f.delivery_number""", {"seg": segment}).df()
    matches = con.execute(f"""
        WITH clean AS ({clean})
        SELECT m.match_id, m.start_date, m.team1, m.team2, m.venue_id, m.venue_name, v.home_team, m.winner,
            m.result, m.scheduled_overs, m.balls_per_over, i1.batting_team AS bat_first,
            i2.batting_team AS bat_second, i1.runs AS first_total, i2.runs AS second_total,
            i1.innings_balls_limit AS balls_limit, e1.elo_before AS elo_bat_first, e2.elo_before AS elo_bat_second
        FROM dim_match m JOIN clean c USING (match_id)
        JOIN fact_innings i1 ON i1.match_id = m.match_id AND i1.innings = 1 AND NOT i1.is_super_over
        JOIN fact_innings i2 ON i2.match_id = m.match_id AND i2.innings = 2 AND NOT i2.is_super_over
        LEFT JOIN read_parquet('{GOLD / 'dim_venue_version'}.parquet') v
            ON v.venue_id = m.venue_id AND v.venue_version = m.venue_version
        LEFT JOIN elo_df e1 ON e1.match_id = m.match_id AND e1.team = i1.batting_team
        LEFT JOIN elo_df e2 ON e2.match_id = m.match_id AND e2.team = i2.batting_team
        WHERE m.segment = $seg AND c.is_clean ORDER BY m.start_date, m.match_id""", {"seg": segment}).df()
    matches["bat_first_won"] = np.select([matches.result == "tie", matches.winner == matches.bat_first],
                                         [0.5, 1.0], 0.0)
    matches = add_context(con, matches, segment)
    ctx = pd.concat([side_context(matches, inn).assign(innings=inn) for inn in (1, 2)])
    balls = add_recent(balls).merge(ctx, on=["match_id", "innings"], how="left")
    return balls, matches


def add_recent(balls: pd.DataFrame) -> pd.DataFrame:
    """RECENT columns from earlier deliveries of the same innings (balls sorted by match, innings, delivery).
    A boundary is a ball in the '4' or '6' class, as the simulator draws them."""
    g = balls.groupby(["match_id", "innings"], sort=False).ngroup().to_numpy()
    runs = balls.runs_total.to_numpy(float)
    bnd = balls.outcome_class.isin(["4", "6"]).to_numpy(float)
    last_runs, last_bnd = np.zeros(len(balls)), np.zeros(len(balls))
    for k in range(1, 7):
        same = g[k:] == g[:-k]
        last_runs[k:] += np.where(same, runs[:-k], 0.0)
        last_bnd[k:] += np.where(same, bnd[:-k], 0.0)
    legal = balls.is_legal.astype(int)
    faced = legal.groupby([balls.match_id, balls.innings, balls.batter_id], sort=False).cumsum() - legal
    return balls.assign(last6_runs=last_runs, last6_boundaries=last_bnd, striker_balls=faced.astype(float))


def add_context(con, matches: pd.DataFrame, segment: str) -> pd.DataFrame:
    """Pre-match venue and playing-XI columns, built from earlier matches only (the same code as
    measure_explained_share.py): venue_dev, the two home flags, and each side's XI rates (bf_* bats first)."""
    m = add_level_and_venue(matches.copy(), first_innings(con), segment)
    m = add_xi(con, m, segment)
    m["bat_first_home"] = (m.home_team == m.bat_first).astype(float)
    m["bat_second_home"] = (m.home_team == m.bat_second).astype(float)
    return m


def side_context(ms: pd.DataFrame, innings: int) -> pd.DataFrame:
    """CONTEXT columns from the point of view of the side batting in this innings."""
    bat, bowl = ("bf", "bs") if innings == 1 else ("bs", "bf")
    home = ("bat_first_home", "bat_second_home") if innings == 1 else ("bat_second_home", "bat_first_home")
    return pd.DataFrame({
        "match_id": ms.match_id.to_numpy(), "venue_dev": ms.venue_dev.to_numpy(float),
        "bat_home": ms[home[0]].to_numpy(float), "bowl_home": ms[home[1]].to_numpy(float),
        **{f"bat_{f}": ms[f"{bat}_bat_{f}"].to_numpy(float) for f in ("rpb", "opb", "rpb_365")},
        **{f"bowl_{f}": ms[f"{bowl}_bowl_{f}"].to_numpy(float) for f in ("rpb", "wpb", "rpb_365")},
        # 1 for internationals; league balls and league matches carry 0 (league_data() sets it)
        "is_international": ms["is_international"].to_numpy(float) if "is_international" in ms else 1.0,
    })


def league_data(con) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(balls, matches) of every clean T20 league match, both with is_international = 0 and a segment column.
    Each league's Elo, level and ground history are its own; its ground history also counts T20I and IPL innings."""
    balls, matches = [], []
    for seg in T20_LEAGUES:
        mes.FAMILY[seg] = ("t20", ["T20I", "IPL", seg])
        b, m = load(con, seg)
        balls.append(b.assign(is_international=0.0))
        matches.append(m.assign(is_international=0.0, segment=seg))
    return pd.concat(balls, ignore_index=True), pd.concat(matches, ignore_index=True)


def add_ratings(con, balls: pd.DataFrame, matches: pd.DataFrame, family: str) -> tuple[pd.DataFrame, dict]:
    """Rating columns on every training ball (striker, bowler, ground, as of the match date), and each match's
    line-ups for the simulator: {match_id: {innings: (order, plan, venue)}}.

    order (SLOTS x 9): the batting side's ratings in the real batting order (who came in, in that order), then the
    rest of the playing XI by balls faced per innings in earlier matches (most first), padded with average.
    plan (overs x 9): the bowler of each over of the real innings (most balls in that over). Overs the real innings
    never reached go to the real bowlers with overs left, most-used first, never the bowler of the over before.
    venue (9): the ground's ratings. family: "T20" (T20I, IPL and the nine leagues) or "ODI"."""
    allb = pr.all_balls(con, family)
    exp, prior = pr.expected_probs(allb)
    ids = list(matches.match_id.astype(str).unique())
    con.execute("CREATE OR REPLACE TEMP TABLE want (match_id VARCHAR)")
    con.executemany("INSERT INTO want VALUES (?)", [(m,) for m in ids])
    xi = con.execute(f"""
        SELECT x.match_id, x.player_id, x.team, x.prior_bat_balls / nullif(x.prior_batting_matches, 0) AS bpi
        FROM read_parquet('{GOLD / 'fact_player_match_form'}.parquet') x JOIN want USING (match_id)""").df()
    dl = con.execute(f"""
        SELECT f.match_id, f.innings, f.delivery_number, f.batting_team, f.batter_id, f.non_striker_id,
            f.bowler_id, f.over
        FROM read_parquet('{GOLD / 'fact_delivery'}.parquet') f JOIN want USING (match_id)
        WHERE NOT f.is_super_over AND f.innings <= 2 ORDER BY f.match_id, f.innings, f.delivery_number""").df()
    date = dict(zip(matches.match_id.astype(str), pd.to_datetime(matches.start_date)))
    xi["start_date"] = xi.match_id.map(date)
    tables = {"bat": pr.ratings(allb, exp, prior, "batter_id", "bat", xi.rename(columns={"player_id": "batter_id"})),
              "bowl": pr.ratings(allb, exp, prior, "bowler_id", "bowl", xi.rename(columns={"player_id": "bowler_id"})),
              "venue": pr.ratings(allb, exp, prior, "venue_id", "venue")}
    balls = balls.copy()
    balls["start_date"] = pd.to_datetime(balls.start_date)
    balls = balls.merge(allb[["match_id", "innings", "delivery_number", "bowler_id", "venue_id"]],
                        on=["match_id", "innings", "delivery_number"], how="left")
    for e, col in pr.ENTITIES.items():
        balls = balls.merge(tables[e], on=[col, "start_date"], how="left")
    look = {e: tables[e].set_index([col, "start_date"]) for e, col in pr.ENTITIES.items()}
    cols = {"bat": RATING_BAT, "bowl": RATING_BOWL, "venue": RATING_VENUE}

    def vec(e, key, d):
        try:
            return look[e].loc[(key, d), cols[e]].to_numpy(float)
        except KeyError:
            return np.r_[np.zeros(8), 0.0]  # never seen: average, no evidence

    lineups = {}
    xi_by = {k: g.sort_values("bpi", ascending=False, na_position="last").player_id.tolist()
             for k, g in xi.groupby(["match_id", "team"])}
    for (mid, inn), g in dl.groupby(["match_id", "innings"], sort=False):
        d = date[mid]
        seen = list(dict.fromkeys(x for pair in zip(g.batter_id, g.non_striker_id) for x in pair if x))
        team = g.batting_team.iloc[0]
        order = (seen + [p for p in xi_by.get((mid, team), []) if p not in seen])[:SLOTS]
        order_m = np.array([vec("bat", p, d) for p in order] + [vec("bat", None, d)] * (SLOTS - len(order)))
        lineups.setdefault(mid, {})[inn] = (order_m, g, d)
    sched = dict(zip(matches.match_id.astype(str), matches.scheduled_overs))
    ground = dict(zip(matches.match_id.astype(str), matches.venue_id))
    for mid, ms in lineups.items():
        overs, venue_id = int(sched[mid]), ground[mid]
        for inn, (order_m, g, d) in ms.items():
            real = g.groupby("over").bowler_id.agg(lambda s: s.value_counts().index[0])
            plan = [real.get(o) for o in range(overs)]
            quota = max(1, overs // 5)
            used = pd.Series([b for b in plan if b is not None]).value_counts().to_dict()
            ranked = list(used)  # most overs first
            for o in range(overs):
                if plan[o] is None:
                    prev = plan[o - 1] if o else None
                    pick = next((b for b in ranked if used[b] < quota and b != prev), None) \
                        or next((b for b in ranked if b != prev), ranked[0] if ranked else None)
                    plan[o] = pick
                    if pick is not None:
                        used[pick] = used.get(pick, 0) + 1
            plan_m = np.array([vec("bowl", b, d) for b in plan])
            ms[inn] = (order_m, plan_m, vec("venue", venue_id, d))
    return balls, lineups


def lineup_arrays(lineups: dict, ms: pd.DataFrame, inn: int, n: int) -> dict:
    """Per-row (match x n plays) arrays for play_innings: order (rows x SLOTS x 9), plan (rows x overs x 9) and
    the ground's rating columns. A match without a line-up gets average players."""
    overs = int(ms.scheduled_overs.max())
    avg = np.r_[np.zeros(8), 0.0]
    order = np.tile(avg, (len(ms), SLOTS, 1))
    plan = np.tile(avg, (len(ms), overs, 1))
    venue = np.tile(avg, (len(ms), 1))
    for k, mid in enumerate(ms.match_id.astype(str)):
        if mid in lineups and inn in lineups[mid]:
            o, p, v = lineups[mid][inn]
            order[k], plan[k, :len(p)], venue[k] = o, p[:overs], v
    rep = lambda a: np.repeat(a, n, axis=0)
    return {"order": rep(order), "plan": rep(plan), "venue": rep(venue)}


def league_balls(con) -> pd.DataFrame:
    """Every clean ball of the T20 leagues, with is_international = 0, to train the T20I ball model."""
    return league_data(con)[0]


class BallModel:
    """The outcome classifier plus, per class, the (runs, legal) pairs seen in training and their shares.

    half_life (years): weight each training ball by 0.5 ** (age / half_life), age measured back from `as_of`,
    so recent seasons count more. None: every ball counts the same (what `cricket model` does)."""

    def __init__(self, train: pd.DataFrame, half_life: float | None = None, as_of: pd.Timestamp | None = None,
                 context: bool = False, recent: bool = False, ratings: bool = False):
        self.recent, self.ratings = recent, ratings
        ctx = (CONTEXT_NO_XI if ratings else CONTEXT) if context else []
        self.features = (OUTCOME_FEATURES + ctx + (RECENT if recent else [])
                         + (RATING_BAT + RATING_BOWL + RATING_VENUE if ratings else []))
        y = train["outcome_class"].map(CLASSES.index).to_numpy()
        weight = None
        if half_life:
            age = (as_of - pd.to_datetime(train["start_date"])).dt.days.to_numpy() / 365.25
            weight = 0.5 ** (age / half_life)
        self.gbm = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=31,
                                                  min_samples_leaf=2000, l2_regularization=1.0,
                                                  early_stopping=False, random_state=SEED)
        self.gbm.fit(train[self.features], y, sample_weight=weight)
        self.draws = []
        w = pd.Series(1.0 if weight is None else weight, index=train.index)
        for cls in CLASSES:
            sel = train.outcome_class == cls
            pairs = w[sel].groupby([train.loc[sel, "runs_total"], train.loc[sel, "is_legal"]]).sum()
            self.draws.append((pairs.index.get_level_values(0).to_numpy(),
                               pairs.index.get_level_values(1).to_numpy().astype(int),
                               (pairs / pairs.sum()).to_numpy()))
        self.class_runs = np.array([(r * w).sum() for r, _, w in self.draws])  # mean runs per class

    def proba(self, x: pd.DataFrame) -> np.ndarray:
        p = np.zeros((len(x), len(CLASSES)))
        p[:, self.gbm.classes_] = self.gbm.predict_proba(x[self.features])
        return p


def play_innings(model: BallModel, rng, innings: int, limit: np.ndarray, static: dict,
                 target: np.ndarray | None = None, tilt: np.ndarray | None = None,
                 lineup: dict | None = None) -> np.ndarray:
    """Play one innings to the end for every row at once (a row is one play of one match); returns the totals.
    limit, target and tilt are per row; static maps each fixed input (elo_diff as batting minus bowling, year,
    and the CONTEXT columns) to a per-row array. tilt is the play's pitch factor:
    each class's chance is multiplied by exp(tilt * that class's mean runs), then renormalised, so a positive
    tilt means more boundaries and fewer dots on every ball of the match. Wickets are not tilted.
    lineup (lineup_arrays, needed when the model takes ratings): each ball sees the striker's ratings (batting order
    slot of whoever is on strike; a wicket is always the striker's, and the next batter in the order takes his
    place) and the ratings of the bowler planned for that over."""
    n = len(limit)
    runs = np.zeros(n, int)
    wkts = np.zeros(n, int)
    legal = np.zeros(n, int)
    live = np.ones(n, bool)
    recent = getattr(model, "recent", False) or lineup is not None  # the lineup needs the strike tracking too
    if lineup is not None:
        slot = np.tile([0, 1], (n, 1))  # batting-order slot of the two batters at the crease
        next_in = np.full(n, 2)
        last_over = lineup["plan"].shape[1] - 1
    if recent:  # last 6 deliveries (ring buffer), balls faced by the two batters, which of them is on strike
        ring_runs, ring_bnd, pos = np.zeros((n, 6)), np.zeros((n, 6)), np.zeros(n, int)
        faced, striker = np.zeros((n, 2)), np.zeros(n, int)
        drawn_runs, drawn_legal = np.zeros(n, int), np.zeros(n, int)
    while live.any():
        i = np.flatnonzero(live)
        remaining = limit[i] - legal[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            need = (target[i] - runs[i]) if target is not None else np.full(len(i), np.nan)
            x = pd.DataFrame({
                "innings": innings, "innings_runs_before": runs[i], "innings_wickets_before": wkts[i],
                "legal_balls_before": legal[i], "scheduled_balls_remaining_before": remaining,
                "nominal_runs_required_before": need,
                "run_rate": np.where(legal[i] > 0, 6 * runs[i] / legal[i], np.nan),
                "required_rate": np.where(remaining > 0, 6 * need / remaining, np.nan),
                **{k: v[i] for k, v in static.items()}})
        if recent:
            x["last6_runs"] = ring_runs[i].sum(1)
            x["last6_boundaries"] = ring_bnd[i].sum(1)
            x["striker_balls"] = faced[i, striker[i]]
        if lineup is not None:
            bat = lineup["order"][i, slot[i, striker[i]]]
            bowl = lineup["plan"][i, np.minimum(legal[i] // 6, last_over)]
            x = pd.concat([x, pd.DataFrame(np.hstack([bat, bowl, lineup["venue"][i]]),
                                           columns=RATING_BAT + RATING_BOWL + RATING_VENUE)], axis=1)
        p = model.proba(x)
        if tilt is not None:
            p *= np.exp(tilt[i][:, None] * model.class_runs[None, :])
            p /= p.sum(1, keepdims=True)
        cls = np.minimum((p.cumsum(1) < rng.random(len(i))[:, None]).sum(1), len(CLASSES) - 1)
        for k in np.unique(cls):
            rows = i[cls == k]
            r, lg, w = model.draws[k]
            pick = rng.choice(len(w), size=len(rows), p=w)
            runs[rows] += r[pick]
            legal[rows] += lg[pick]
            if recent:
                drawn_runs[rows], drawn_legal[rows] = r[pick], lg[pick]
            if k == WICKET:
                wkts[rows] += 1
        if recent:
            ring_runs[i, pos[i]] = drawn_runs[i]
            ring_bnd[i, pos[i]] = np.isin(cls, BOUNDARY)
            pos[i] = (pos[i] + 1) % 6
            s = striker[i]
            faced[i, s] += drawn_legal[i]
            out = cls == WICKET
            faced[i[out], s[out]] = 0  # the new batter takes the striker's place
            if lineup is not None:
                gone = i[out]
                slot[gone, s[out]] = np.minimum(next_in[gone], SLOTS - 1)
                next_in[gone] += 1
            ran = drawn_runs[i] - (1 - drawn_legal[i])  # a wide or no-ball's own run is not run
            swap = (ran % 2 == 1) ^ ((drawn_legal[i] == 1) & (legal[i] % 6 == 0))  # odd runs, end of over
            striker[i] = np.where(swap, 1 - s, s)
        done = (wkts >= 10) | (legal >= limit)
        if target is not None:
            done |= runs >= target
        live &= ~done
    return runs


def simulate(model: BallModel, ms: pd.DataFrame, n: int, rng, elo_scale: float = 1.0, pitch_sd: float = 0.0,
             innings_sd: float = 0.0, lineups: dict | None = None) -> dict:
    """n plays of every match in ms at once. Returns (matches x n) arrays: first-innings totals and the
    batting-first side's result (1 win, 0.5 tie, 0 loss). elo_scale multiplies the Elo gap the ball model sees.
    pitch_sd: each play draws one pitch factor from Normal(0, pitch_sd), shared by both innings.
    innings_sd: each innings of each play also draws its own batting-side factor from Normal(0, innings_sd), added
    to the pitch factor (a side having a good or bad day; tested 2026-11-18, off by default)."""
    diff = np.repeat(elo_scale * (ms.elo_bat_first - ms.elo_bat_second).to_numpy(float), n)
    limit = np.repeat(ms.balls_limit.to_numpy(int), n)
    year = np.repeat(pd.to_datetime(ms.start_date).dt.year.to_numpy(), n)
    static = {inn: {"elo_diff": diff if inn == 1 else -diff, "year": year,
                    **{c: np.repeat(side_context(ms, inn)[c].to_numpy(float), n) for c in CONTEXT}}
              for inn in (1, 2)}
    tilt = rng.normal(0.0, pitch_sd, len(limit)) if pitch_sd else None
    tilts = [tilt, tilt]
    if innings_sd:
        base = np.zeros(len(limit)) if tilt is None else tilt
        tilts = [base + rng.normal(0.0, innings_sd, len(limit)) for _ in (1, 2)]
    lu = {inn: lineup_arrays(lineups, ms, inn, n) if getattr(model, "ratings", False) else None for inn in (1, 2)}
    first = play_innings(model, rng, 1, limit, static[1], tilt=tilts[0], lineup=lu[1])
    target = first + 1
    second = play_innings(model, rng, 2, limit, static[2], target, tilts[1], lineup=lu[2])
    result = np.where(second >= target, 0.0, np.where(second == first, 0.5, 1.0))
    return {"first": first.reshape(len(ms), n), "result": result.reshape(len(ms), n)}


def crps(samples: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Continuous ranked probability score of each row's samples against its actual value (lower is better;
    in runs). E|X - y| - E|X - X'| / 2, the second term from the sorted samples."""
    x = np.sort(samples, axis=1)
    n = x.shape[1]
    spread = (2 * np.arange(1, n + 1) - n - 1) @ x.T * 2 / n ** 2
    return np.abs(x - actual[:, None]).mean(1) - spread / 2


def stretch(first: np.ndarray, s: float, a: float = 0.0) -> np.ndarray:
    """Spread each row's plays around their mean by factor s (and move them by a runs)."""
    mean = first.mean(1, keepdims=True)
    return mean + a + s * (first - mean)


def fit_scale(first: np.ndarray, actual: np.ndarray, shift: bool = False) -> tuple[float, float]:
    """The stretch (and, with shift, the move in runs) with the lowest mean CRPS of first against actual."""
    if not shift:
        r = minimize_scalar(lambda s: crps(stretch(first, s), actual).mean(), bounds=(0.5, 3.0), method="bounded")
        return float(r.x), 0.0
    r = minimize(lambda v: crps(stretch(first, v[0], v[1]), actual).mean(), x0=[1.2, 0.0], method="Nelder-Mead",
                 options={"xatol": 1e-3, "fatol": 1e-4})
    return float(r.x[0]), float(r.x[1])


def fit_stretch(balls: pd.DataFrame, matches: pd.DataFrame, cutoff: pd.Timestamp, years: int, plays: int,
                pitch_sd: float, recent: bool, segment: str) -> tuple[float, list]:
    """Walk forward over the `years` years before cutoff: fit on the ten years before each, play that year's
    full-length matches, and return the stretch with the lowest CRPS over all of them, plus one row per year.
    Each year has its own generator, seeded like sim_backtest.py, so the plays match the backtest's."""
    mdate = pd.to_datetime(matches.start_date)
    firsts, actuals, rows = [], [], []
    for k in range(years, 0, -1):
        start, end = cutoff - pd.DateOffset(years=k), cutoff - pd.DateOffset(years=k - 1)
        test = full_length(matches[(mdate >= start) & (mdate < end)]).reset_index(drop=True)
        if test.empty:
            continue
        rng = np.random.default_rng([SEED, start.year, FORMAT_CODE[segment]])
        s = simulate(fit(balls, start, None, recent=recent), test, plays, rng, 1.0, pitch_sd)
        firsts.append(s["first"].astype(float))
        actuals.append(test.first_total.to_numpy(float))
        rows.append({"segment": segment, "from": str(start.date()), "to": str(end.date()), "matches": len(test),
                     "scale_this_year_alone": fit_scale(firsts[-1], actuals[-1])[0]})
    scale = fit_scale(np.vstack(firsts), np.concatenate(actuals))[0] if firsts else 1.0
    return scale, rows


def score(ms: pd.DataFrame, s: dict) -> dict:
    lo, hi = np.percentile(s["first"], [10, 90], axis=1)
    win = s["result"].mean(1)
    actual = ms.first_total.to_numpy(float)
    return {"matches": len(ms), "brier": float(((win - ms.bat_first_won) ** 2).mean()),
            "crps_first_total": float(crps(s["first"], actual).mean()),
            "inside_80pct_range": float(((actual >= lo) & (actual <= hi)).mean()),
            "mean_bias_runs": float((np.median(s["first"], axis=1) - actual).mean())}


def fit(balls: pd.DataFrame, cutoff: pd.Timestamp, half_life: float | None, context: bool = True,
        recent: bool = False, ratings: bool = False) -> BallModel:
    when = pd.to_datetime(balls.start_date)
    train = balls[(when < cutoff) & (when >= cutoff - pd.DateOffset(years=OUTCOME_SEASONS))]
    return BallModel(train, half_life, cutoff, context, recent, ratings)


def full_length(matches: pd.DataFrame) -> pd.DataFrame:
    return matches[matches.balls_limit == matches.scheduled_overs * 6]


def pct(x: float) -> float:
    return round(100 * float(x), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train-before", default="2026-01-01", help="fit on clean matches before this date")
    ap.add_argument("--last", type=int, default=5, help="how many of the latest clean matches per format")
    ap.add_argument("--sims", type=int, default=5000, help="plays per chosen match")
    ap.add_argument("--backtest-sims", type=int, default=1000, help="plays per match in the backtest (0: skip)")
    ap.add_argument("--tune-sims", type=int, default=500, help="plays per match while tuning (0: no tuning)")
    # 2026-11-17: half-lives 0,4,2 and Elo scales 1,0.75,0.5,0.25,0 were tried on 2025; the defaults won both.
    ap.add_argument("--half-lives", default="0", help="half-lives in years to try (0: equal weights)")
    ap.add_argument("--elo-scales", default="1", help="Elo-gap multipliers to try")
    ap.add_argument("--pitch-sd", default="ODI=0,T20I=0.025",
                    help="fixed pitch-factor spread per format (2026-11-19: tuned on 2018, checked on 2019-2026)")
    ap.add_argument("--pitch-sds", default="",
                    help="pitch-factor spreads to tune over instead, e.g. 0,0.025,0.05,0.075,0.1,0.15 (the default "
                         "before 2026-11-19); empty: use --pitch-sd")
    ap.add_argument("--no-recent", action="store_true",
                    help="no in-play inputs (the simulator before 2026-11-19)")
    ap.add_argument("--stretch-formats", default="ODI",
                    help="formats whose first-innings totals are stretched after the sim (empty: none)")
    ap.add_argument("--stretch-years", type=int, default=3, help="years before --train-before the stretch is fitted on")
    ap.add_argument("--stretch-sims", type=int, default=300, help="plays per match while fitting the stretch")
    ap.add_argument("--before-pitch-sd", type=float, default=0.025,
                    help="pitch spread of the 'before' run, the earlier simulator without venue and XI inputs")
    ap.add_argument("--full-members", action="store_true",
                    help="pick the chosen matches only from games between two full members (config/scope.yml); "
                         "training uses every match, the backtest reports both")
    ap.add_argument("--no-t20-leagues", action="store_true",
                    help="train the T20I ball model on T20I balls only (the simulator before 2026-11-18)")
    ap.add_argument("--out", type=Path, default=Path("sims") / dt.date.today().isoformat())
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cutoff = pd.Timestamp(args.train_before)
    tune_cutoff = cutoff - pd.DateOffset(years=1)
    members = yaml.safe_load(Path("config/scope.yml").read_text())["full_members"]
    rng = np.random.default_rng(SEED)
    con = duckdb.connect()
    con.execute("SET memory_limit = '3GB'")
    events, backtest, tuning, summary = [], [], [], {}
    half_lives = [float(h) or None for h in args.half_lives.split(",")]
    scales = [float(a) for a in args.elo_scales.split(",")]
    pitch_sds = [float(v) for v in args.pitch_sds.split(",") if v.strip()]
    fixed_pitch = {k: float(v) for k, v in (kv.split("=") for kv in args.pitch_sd.split(",") if kv.strip())}
    stretch_formats = {f for f in args.stretch_formats.split(",") if f.strip()}
    recent = not args.no_recent

    for segment in ("ODI", "T20I"):
        balls, matches = load(con, segment)
        if segment == "T20I" and not args.no_t20_leagues:
            balls = pd.concat([balls, league_balls(con)], ignore_index=True)
        mdate = pd.to_datetime(matches.start_date)
        is_fm = matches.team1.isin(members) & matches.team2.isin(members)
        start = cutoff - pd.DateOffset(years=OUTCOME_SEASONS)
        hist = matches[(mdate < cutoff) & (mdate >= start)]
        big = BIG_TOTAL[segment]

        # Tuning: fit on the ten years before tune_cutoff, play the year after it (never the test period).
        # Half-life first (judged on first-innings totals, Elo gap as is), then the Elo scale at that half-life
        # (judged on who won).
        half_life, scale, pitch_sd = None, 1.0, fixed_pitch.get(segment, 0.0)
        if args.tune_sims:
            val = full_length(matches[(mdate >= tune_cutoff) & (mdate < cutoff)])
            print(f"{segment}: tuning on {len(val)} matches from {tune_cutoff.date()}", flush=True)
            models = {}
            for h in half_lives:
                models[h] = fit(balls, tune_cutoff, h, recent=recent)
                r = score(val, simulate(models[h], val, args.tune_sims, rng, 1.0))
                tuning.append({"segment": segment, "half_life": h, "elo_scale": 1.0, "pitch_sd": 0.0, **r})
                print(f"  half-life {h}: {r}", flush=True)
            seg_t = pd.DataFrame([t for t in tuning if t["segment"] == segment])
            half_life = seg_t.loc[seg_t.crps_first_total.idxmin(), "half_life"]
            half_life = None if pd.isna(half_life) else float(half_life)
            for a in scales:
                if a == 1.0:
                    continue
                r = score(val, simulate(models[half_life], val, args.tune_sims, rng, a))
                tuning.append({"segment": segment, "half_life": half_life, "elo_scale": a, "pitch_sd": 0.0, **r})
                print(f"  half-life {half_life}, Elo scale {a}: {r}", flush=True)
            seg_t = pd.DataFrame([t for t in tuning if t["segment"] == segment])
            same_h = seg_t[seg_t.half_life.isna()] if half_life is None else seg_t[seg_t.half_life == half_life]
            scale = float(same_h.loc[same_h.brier.idxmin(), "elo_scale"])
            # Pitch factor last, at the chosen half-life and Elo scale, judged on first-innings totals; only when
            # --pitch-sds asks for it (the half-life and Elo-scale runs above are at pitch 0).
            best = {0.0: same_h[same_h.elo_scale == scale].crps_first_total.min()}
            for v in pitch_sds:
                if v == 0.0:
                    continue
                r = score(val, simulate(models[half_life], val, args.tune_sims, rng, scale, v))
                tuning.append({"segment": segment, "half_life": half_life, "elo_scale": scale, "pitch_sd": v, **r})
                best[v] = r["crps_first_total"]
                print(f"  half-life {half_life}, Elo scale {scale}, pitch sd {v}: {r}", flush=True)
            if pitch_sds:
                pitch_sd = min(best, key=best.get)
            print(f"  chosen: half-life {half_life}, Elo scale {scale}, pitch sd {pitch_sd}", flush=True)

        before = fit(balls, cutoff, None, context=False)
        after = fit(balls, cutoff, half_life, recent=recent)
        scale_totals, stretch_rows = 1.0, []
        if segment in stretch_formats:
            scale_totals, stretch_rows = fit_stretch(balls, matches, cutoff, args.stretch_years, args.stretch_sims,
                                                     pitch_sd, recent, segment)
            print(f"  {segment} totals stretch {scale_totals:.3f}, fitted on {stretch_rows}", flush=True)
        later = full_length(matches[mdate >= cutoff]).copy()
        later["full_members"] = is_fm[later.index]
        summary[segment] = {"half_life_years": half_life, "elo_scale": scale, "pitch_sd": pitch_sd,
                            "context_inputs": CONTEXT, "in_play_inputs": RECENT if recent else [],
                            "totals_stretch": scale_totals, "stretch_fit": stretch_rows, "before": f"no context inputs, pitch sd {args.before_pitch_sd}"}

        if args.backtest_sims:
            runs = {"before": simulate(before, later, args.backtest_sims, rng, 1.0, args.before_pitch_sd),
                    "after": simulate(after, later, args.backtest_sims, rng, scale, pitch_sd)}
            runs["after"]["first"] = stretch(runs["after"]["first"], scale_totals)
            elo = np.array([expected(a, b) for a, b in zip(later.elo_bat_first, later.elo_bat_second)])
            for subset, mask in (("all", np.ones(len(later), bool)), ("full_members", later.full_members.to_numpy())):
                if not mask.any():
                    continue
                ms = later[mask]
                out = {k: score(ms, {"first": v["first"][mask], "result": v["result"][mask]}) for k, v in runs.items()}
                out["brier_elo_only"] = float(((elo[mask] - ms.bat_first_won) ** 2).mean())
                out["brier_history_rate"] = float(((hist.bat_first_won.mean() - ms.bat_first_won) ** 2).mean())
                summary[segment][subset] = out
                print(f"  backtest {subset}: {json.dumps(out)}", flush=True)
            for j, (_, m) in enumerate(later.iterrows()):
                backtest.append({"segment": segment, "match_id": m.match_id, "start_date": m.start_date,
                                 "full_members": m.full_members, "bat_first": m.bat_first,
                                 "actual_bat_first_won": m.bat_first_won, "actual_first_total": m.first_total,
                                 "elo_bat_first_win": elo[j],
                                 **{f"{k}_sim_win": v["result"][j].mean() for k, v in runs.items()},
                                 **{f"{k}_sim_first_median": np.median(v["first"][j]) for k, v in runs.items()}})

        chosen = later[later.full_members] if args.full_members else later
        chosen = chosen.tail(args.last)
        s = simulate(after, chosen, args.sims, rng, scale, pitch_sd)
        s["first"] = stretch(s["first"], scale_totals)  # totals only; the win chance stays from the plays
        for j, (_, m) in enumerate(chosen.iterrows()):
            first, result = s["first"][j], s["result"][j]
            other = m.team2 if m.bat_first == m.team1 else m.team1
            base = {"segment": segment, "match_id": m.match_id, "start_date": m.start_date,
                    "match": f"{m.bat_first} (batted first) v {other}", "venue": m.venue_name,
                    "elo_gap_bat_first": round(m.elo_bat_first - m.elo_bat_second)}
            lo, med, hi = np.percentile(first, [10, 50, 90])
            rows = [
                (f"{m.bat_first} wins", result.mean(), hist.bat_first_won.mean(),
                 {1.0: "yes", 0.5: "tie", 0.0: "no"}[m.bat_first_won],
                 f"Elo alone says {pct(expected(m.elo_bat_first, m.elo_bat_second))}%"),
                (f"First innings {big}+", (first >= big).mean(), (hist.first_total >= big).mean(),
                 "yes" if m.first_total >= big else "no",
                 f"sim median {med:.0f}, 80% range {lo:.0f} to {hi:.0f}, actual {m.first_total}"),
                (f"First innings {m.first_total}+ (the actual score)", (first >= m.first_total).mean(),
                 (hist.first_total >= m.first_total).mean(), "yes", "how surprising the real score was"),
            ]
            for event, sim, hist_rate, happened, note in rows:
                events.append({**base, "event": event, "sim_pct": pct(sim), "history_pct": pct(hist_rate),
                               "happened": happened, "note": note})
        print(f"  simulated {len(chosen)} chosen matches", flush=True)

    pd.DataFrame(events).to_csv(args.out / "events.csv", index=False)
    if backtest:
        pd.DataFrame(backtest).to_csv(args.out / "backtest.csv", index=False)
    if tuning:
        pd.DataFrame(tuning).to_csv(args.out / "tuning.csv", index=False)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    build = json.loads((GOLD / "run.json").read_text()).get("build_fingerprint")
    summary["run"] = {"commit": commit, "build_fingerprint": build, "train_before": args.train_before,
                      "tuned_on": f"{tune_cutoff.date()} to {cutoff.date()}", "train_years": OUTCOME_SEASONS,
                      "sims_per_match": args.sims, "backtest_sims_per_match": args.backtest_sims,
                      "tune_sims_per_match": args.tune_sims, "seed": SEED, "full_members_only": args.full_members,
                      "t20i_trained_on_leagues": not args.no_t20_leagues, "in_play_inputs": recent,
                      "stretch_formats": sorted(stretch_formats), "stretch_years": args.stretch_years}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary["run"]))


if __name__ == "__main__":
    main()
