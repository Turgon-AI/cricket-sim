"""The report tables and headline.json, checked against counts taken from the facts, and the model
scoring reports checked against a few rows scored by hand.

Before this file the report numbers (coverage, IPL seasons, headline, Brier, checkpoints, eras) were only
checked for bounds and for being consistent with each other, so a column that counted balls where it said
matches, a Brier score swapped for mean absolute error or a checkpoint one ball off still passed."""

import json
import math

import duckdb
import numpy as np
import pandas as pd
import pytest

from cricket_sim.models import dataset, report, train


def q(data, sql):
    return duckdb.connect().execute(sql.replace("{gold}", str(data / "gold"))).fetchall()


# ---- the gold reports against the facts -------------------------------------------------------------------

def test_headline_counts_are_the_counts_of_the_tables(tiny_build):
    headline = json.loads((tiny_build / "gold" / "headline.json").read_text())
    matches = q(tiny_build, "SELECT count(*) FROM '{gold}/dim_match.parquet'")[0][0]
    balls = q(tiny_build, "SELECT count(*) FROM '{gold}/fact_delivery.parquet'")[0][0]
    passing = q(tiny_build, "SELECT count(*) FROM '{gold}/dim_match.parquet' WHERE passes_quality")[0][0]
    assert (matches, balls) == (9, 7793)                                  # the nine in-scope fixture matches
    assert headline["matches_in_scope"] == matches
    assert headline["balls_in_scope"] == balls
    assert headline["matches_passing_quality_pct"] == round(100.0 * passing / matches, 2) == 100.0


def test_coverage_by_segment_year_counts_matches_and_balls(tiny_build):
    rows = q(tiny_build, """SELECT segment, year, matches, matches_passing_quality, balls
                            FROM '{gold}/report_coverage_by_segment_year.parquet' ORDER BY 1, 2""")
    assert rows == [("IPL", 2019, 1, 1, 247), ("ODI", 2011, 1, 1, 604), ("ODI", 2017, 1, 1, 500),
                    ("ODI", 2019, 2, 2, 1254), ("ODI", 2023, 1, 1, 569), ("T20I", 2019, 1, 1, 152),
                    ("Test", 2021, 1, 1, 2417), ("Test", 2024, 1, 1, 2050)]
    # and they are the facts' own counts: matches is a count of matches (not balls), balls a count of deliveries
    by_facts = q(tiny_build, """SELECT d.segment, year(d.start_date), count(DISTINCT d.match_id), count(*)
                                FROM '{gold}/dim_match.parquet' d JOIN '{gold}/fact_delivery.parquet' f USING (match_id)
                                GROUP BY 1, 2 ORDER BY 1, 2""")
    assert [(s, y, m, b) for s, y, m, _, b in rows] == by_facts
    assert sum(r[2] for r in rows) == q(tiny_build, "SELECT count(*) FROM '{gold}/dim_match.parquet'")[0][0]


def test_ipl_seasons_cover_ipl_matches_only(tiny_build):
    # the fixture has one IPL match (the 2019 final); the other eight matches are not IPL
    assert q(tiny_build, "SELECT season_year, matches, matches_passing_quality "
                         "FROM '{gold}/report_ipl_seasons.parquet'") == [(2019, 1, 1)]
    assert q(tiny_build, "SELECT year(start_date), count(*) FROM '{gold}/dim_match.parquet' "
                         "WHERE segment = 'IPL' GROUP BY 1") == [(2019, 1)]


# ---- the model scoring reports, scored by hand ------------------------------------------------------------

P_DOT = [0.5, 0.1, 0.1, 0.1, 0.1, 0.05, 0.025, 0.025]       # one probability vector for every ball, in CLASSES order
CLASS_P = {c: p for c, p in zip(dataset.CLASSES, P_DOT)}


def _scored_frame() -> pd.DataFrame:
    #  match innings season won  wp   outcome
    rows = [("m1", 1, 2014, 1.0, 0.8, "dot"), ("m1", 2, 2014, 0.0, 0.3, "wicket"),
            ("m2", 1, 2015, 0.0, 0.6, "dot"), ("m3", 1, 2021, 1.0, 0.9, "dot")]
    df = pd.DataFrame(rows, columns=["match_id", "innings", "season", "batting_won", "wp_model", "outcome_class"])
    df["delivery_number"] = [1, 1, 1, 1]
    df["wp_state_logistic"] = 0.5
    df["wp_base_rate"] = 0.5
    for prefix in ("p_", "phase_p_", "base_p_"):
        for c in dataset.CLASSES:
            df[f"{prefix}{c}"] = CLASS_P[c]
    return df


def _score(df, **where):
    got = report.scores(df)
    mask = pd.Series(True, index=got.index)
    for k, v in where.items():
        mask &= got[k] == v
    assert mask.sum() == 1, where
    return got[mask].iloc[0]


def test_win_probability_scores_are_brier_and_log_loss_by_hand():
    df = _scored_frame()
    # innings 1, all seasons: p = .8, .6, .9 against won = 1, 0, 1
    row = _score(df, target="win_probability", slice="innings_1", model="model", scope="all")
    assert (row["balls"], row["matches"]) == (3, 3)
    assert row["brier"] == pytest.approx((0.2 ** 2 + 0.6 ** 2 + 0.1 ** 2) / 3, abs=1e-5)        # 0.13667, not a mean error
    assert row["log_loss"] == pytest.approx(-(math.log(0.8) + math.log(0.4) + math.log(0.9)) / 3, abs=1e-5)  # 0.41493
    # a constant 0.5 has Brier 0.25 and log loss ln 2 whatever happens
    base = _score(df, target="win_probability", slice="innings_1", model="base_rate", scope="all")
    assert (base["brier"], base["log_loss"]) == (0.25, round(math.log(2), 5))
    # innings 2, season 2014: p = .3, won = 0: Brier .09, log loss -ln(.7)
    row = _score(df, target="win_probability", slice="innings_2", model="model", scope="2014")
    assert row["brier"] == pytest.approx(0.09, abs=1e-5) and row["log_loss"] == pytest.approx(-math.log(0.7), abs=1e-5)


def test_eras_split_at_2014_2015_and_2020_2021():
    df = _scored_frame()
    first = _score(df, target="win_probability", slice="innings_1", model="model", scope="2008-2014")
    assert first["balls"] == 1 and first["brier"] == pytest.approx(0.04, abs=1e-5)             # only the 2014 ball
    second = _score(df, target="win_probability", slice="innings_1", model="model", scope="2015-2020")
    assert second["balls"] == 1 and second["brier"] == pytest.approx(0.36, abs=1e-5)           # 2015 is in this era
    third = _score(df, target="win_probability", slice="innings_1", model="model", scope="2021-")
    assert third["balls"] == 1 and third["brier"] == pytest.approx(0.01, abs=1e-5)
    assert [report._era(y) for y in (2008, 2014, 2015, 2020, 2021, 2026)] == \
        ["2008-2014", "2008-2014", "2015-2020", "2015-2020", "2021-", "2021-"]


def test_ball_outcome_scores_are_multiclass_brier_and_log_loss_by_hand():
    df = _scored_frame()
    # season 2014 has a dot and a wicket ball. Brier = sum over the 8 classes of (p - 1 if it happened else p) ** 2
    dot_brier = 0.5 ** 2 + 4 * 0.1 ** 2 + 0.05 ** 2 + 2 * 0.025 ** 2                           # 0.29375
    wicket_brier = 0.5 ** 2 + 4 * 0.1 ** 2 + 0.05 ** 2 + 0.025 ** 2 + 0.975 ** 2               # 1.24375
    row = _score(df, target="ball_outcome", model="model", scope="2014")
    assert (row["balls"], row["matches"]) == (2, 1)
    assert row["brier"] == pytest.approx((dot_brier + wicket_brier) / 2, abs=1e-5)             # 0.76875
    assert row["log_loss"] == pytest.approx((-math.log(0.5) - math.log(0.025)) / 2, abs=1e-5)  # 2.19101


def test_calibration_bins_predicted_against_observed():
    cal = report.calibration(_scored_frame())
    pick = lambda **kw: cal[np.logical_and.reduce([cal[k] == v for k, v in kw.items()])]
    row = pick(target="win_probability", slice="innings_1", model="model", era="all", bin=6).iloc[0]
    assert (row["balls"], row["mean_predicted"], row["observed_rate"]) == (1, 0.6, 0.0)        # p = .6 lands in bin 6
    row = pick(target="win_probability", slice="innings_1", model="model", era="all", bin=8).iloc[0]
    assert (row["balls"], row["mean_predicted"], row["observed_rate"]) == (1, 0.8, 1.0)
    # the ball-outcome bins are finer: p(dot) = .5 -> int(.5 * 25) = 12, capped at the last bin (9); 3 of 4 balls were dots
    row = pick(target="ball_outcome", slice="dot", bin=9).iloc[0]
    assert (row["balls"], row["mean_predicted"], row["observed_rate"]) == (4, 0.5, 0.75)
    row = pick(target="ball_outcome", slice="wicket", bin=0).iloc[0]
    assert (row["balls"], row["mean_predicted"], row["observed_rate"]) == (4, 0.025, 0.25)


def _chase_frame() -> pd.DataFrame:
    #  innings delivery balls_left  wp_model (batting side's view)
    rows = [(1, 1, 300, 0.70), (2, 1, 120, 0.40), (2, 90, 31, 0.20), (2, 91, 30, 0.15), (2, 92, 29, 0.10)]
    df = pd.DataFrame(rows, columns=["innings", "delivery_number", "scheduled_balls_remaining_before", "wp_model"])
    df["match_id"] = "m"
    df["batting_won"] = np.where(df["innings"] == 1, 1.0, 0.0)           # the side batting first won
    df["wp_state_logistic"] = 0.5
    df["wp_base_rate"] = 0.5
    return df


def test_checkpoints_are_scored_from_the_side_batting_first():
    got = report.checkpoints(_chase_frame())
    brier = {cp: b for cp, m, n, ll, b in got[got["model"] == "model"].itertuples(index=False)}
    # first ball: .7 against a win = .09. Start of chase: the chaser's .4 is .6 for the side batting first: .16.
    # Last five overs = the first chase ball with 30 or fewer balls left, which is the one with exactly 30:
    # the chaser's .15 is .85 for the side batting first: .0225 (the ball with 31 left is not it, nor is 29)
    assert brier == {"start_of_match": pytest.approx(0.09, abs=1e-5), "start_of_chase": pytest.approx(0.16, abs=1e-5),
                     "last_5_overs": pytest.approx(0.0225, abs=1e-5)}
    assert set(got["matches"]) == {1}


def test_phase_baseline_boundaries():
    # powerplay = overs 0 to 9 (the first 10), middle = 10 to 39, death = 40 on; wickets 0-2, 3-5, 6-7, 8+
    df = pd.DataFrame({"innings": [1] * 12, "over": [0, 9, 10, 39, 40, 49, 0, 0, 0, 0, 0, 0],
                       "innings_wickets_before": [0, 0, 0, 0, 0, 0, 2, 3, 5, 6, 7, 8]})
    got = train._phase(df).tolist()
    assert got == ["1|powerplay|0-2", "1|powerplay|0-2", "1|middle|0-2", "1|middle|0-2", "1|death|0-2", "1|death|0-2",
                   "1|powerplay|0-2", "1|powerplay|3-5", "1|powerplay|3-5", "1|powerplay|6-7", "1|powerplay|6-7",
                   "1|powerplay|8+"]
