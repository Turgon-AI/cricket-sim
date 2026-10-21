"""models/dataset.py: the modelling table's features, checked value by value on the fixture's real matches.

Hand-computed Elo (K 24, start 1500): India beat Sri Lanka in the 2011 final (both at 1500, expected 0.5), so
India goes into the 2017 Champions Trophy match at 1512 against Pakistan's 1500. New Zealand beat India
(1523.586) in the 2019 semi-final and take 24 * (1 - expected) = 12.813 points into the final against England.
"""

import duckdb
import numpy as np
import pandas as pd
import pytest

from cricket_sim.models import dataset, train


def _by_match(rows, match_id):
    return rows[rows["match_id"] == match_id]


def test_bowling_team_is_the_other_side(tiny_model_rows, tiny_build):
    m = duckdb.connect().execute(f"SELECT match_id, team1, team2 FROM read_parquet('{tiny_build}/gold/dim_match.parquet')").df()
    r = tiny_model_rows.merge(m, on="match_id")
    assert (r["bowling_team"] != r["batting_team"]).all()
    assert (((r["batting_team"] == r["team1"]) & (r["bowling_team"] == r["team2"]))
            | ((r["batting_team"] == r["team2"]) & (r["bowling_team"] == r["team1"]))).all()


def test_elo_diff_is_batting_minus_bowling_side_before_the_match(tiny_model_rows):
    r = tiny_model_rows
    first = _by_match(r, "433606")
    assert (first["elo_diff"] == 0).all()  # the fixture's first ODI: both sides start at 1500
    ct = _by_match(r, "1022353")
    assert np.allclose(ct.loc[ct["batting_team"] == "India", "elo_diff"], 12.0)
    assert np.allclose(ct.loc[ct["batting_team"] == "Pakistan", "elo_diff"], -12.0)
    final = _by_match(r, "1144530")
    assert np.allclose(final.loc[final["batting_team"] == "New Zealand", "elo_diff"], 12.813, atol=1e-3)
    assert np.allclose(final.loc[final["batting_team"] == "England", "elo_diff"], -12.813, atol=1e-3)


def test_labels_and_clean_flags_on_the_fixture(tiny_model_rows):
    r = tiny_model_rows
    assert set(r["match_id"]) == {"433606", "1022353", "1144528", "1144530", "1384439"}  # the five men's ODIs
    assert r["innings"].max() == 2  # the 2019 final's super over is not modelled
    semi = _by_match(r, "1144528")
    assert (semi.loc[semi["batting_team"] == "New Zealand", "batting_won"] == 1.0).all()
    assert (semi.loc[semi["batting_team"] == "India", "batting_won"] == 0.0).all()
    assert (_by_match(r, "1144530")["batting_won"] == 0.5).all()  # a tie is half a win for both sides
    rain = _by_match(r, "1022353")
    assert not rain["is_clean"].any() and (rain["exclusion_reason"] == "rain_rule_applied").all()
    others = r[r["match_id"] != "1022353"]
    assert others["is_clean"].all() and others["exclusion_reason"].isna().all()
    assert (r["season"] == pd.to_datetime(r["start_date"]).dt.year).all() and (r["year"] == r["season"]).all()


def test_pre_ball_rates_follow_the_state(tiny_model_rows):
    r = tiny_model_rows
    with_balls = r[r["legal_balls_before"] > 0]
    assert np.allclose(with_balls["run_rate"], 6 * with_balls["innings_runs_before"] / with_balls["legal_balls_before"])
    assert r.loc[r["legal_balls_before"] == 0, "run_rate"].isna().all()
    chase = r[(r["innings"] == 2) & (r["scheduled_balls_remaining_before"] > 0)]
    assert len(chase) > 500 and np.allclose(
        chase["required_rate"], 6 * chase["nominal_runs_required_before"] / chase["scheduled_balls_remaining_before"])
    assert r.loc[r["innings"] == 1, "required_rate"].isna().all()


def test_outcome_class_of_each_ball(tiny_model_rows):
    r = tiny_model_rows
    wicket = r["dismissal_count"] > 0
    assert (r.loc[wicket, "outcome_class"] == "wicket").all()
    assert (r.loc[~wicket & ~r["is_legal"], "outcome_class"] == "extra").all()
    legal = r[~wicket & r["is_legal"]]
    expect = legal["runs_total"].map({0: "dot", 1: "1", 2: "2", 3: "3", 4: "4", 5: "4"}).fillna("6")
    assert (legal["outcome_class"] == expect).all()
    assert set(r["outcome_class"]) <= set(dataset.CLASSES)


def test_batting_at_home_sign(tiny_build, monkeypatch):
    # No fixture venue has three earlier internationals, so every match is neutral ...
    con = duckdb.connect()
    dataset.load(con, gold=tiny_build / "gold")
    assert con.execute("SELECT count(*) FILTER (batting_at_home <> 0) FROM model_rows").fetchone()[0] == 0
    # ... so plant a home side for one match (the SQL that turns it into +1 / -1 is what is checked)
    real = dataset.home_teams_as_of
    monkeypatch.setattr(dataset, "home_teams_as_of", lambda rows: {**real(rows), "1144528": "India"})
    con = duckdb.connect()
    dataset.load(con, gold=tiny_build / "gold")
    got = dict(con.execute("""SELECT batting_team, min(batting_at_home) * 10 + max(batting_at_home) FROM model_rows
                              WHERE match_id = '1144528' GROUP BY 1""").fetchall())
    assert got == {"India": 11, "New Zealand": -11}
    assert con.execute("SELECT count(*) FILTER (batting_at_home <> 0) FROM model_rows "
                       "WHERE match_id <> '1144528'").fetchone()[0] == 0


def test_batting_won_toss_follows_the_recorded_toss_winner(tiny_model_rows, tiny_build):
    m = duckdb.connect().execute(f"SELECT match_id, toss_winner FROM read_parquet('{tiny_build}/gold/dim_match.parquet')").df()
    r = tiny_model_rows.merge(m, on="match_id")
    assert (r["batting_won_toss"] == (r["toss_winner"] == r["batting_team"]).astype(int)).all()
    assert set(r["batting_won_toss"].unique()) == {0, 1}  # both sides bat in every match, so both values occur
    # pinned: Sri Lanka won the toss in the 2011 final
    final = _by_match(r, "433606")
    assert (final.loc[final["batting_team"] == "Sri Lanka", "batting_won_toss"] == 1).all()
    assert (final.loc[final["batting_team"] == "India", "batting_won_toss"] == 0).all()


def test_a_match_failing_quality_is_not_clean(tiny_build, tmp_path):
    gold = tmp_path / "gold"
    gold.mkdir()
    con = duckdb.connect()
    for t in ("fact_delivery", "fact_innings", "dim_venue"):
        (gold / f"{t}.parquet").symlink_to(tiny_build / "gold" / f"{t}.parquet")
    con.execute(f"""COPY (SELECT * REPLACE (CASE WHEN match_id = '1144528' THEN false ELSE passes_quality END
                                           AS passes_quality)
                    FROM read_parquet('{tiny_build}/gold/dim_match.parquet')) TO '{gold}/dim_match.parquet'""")
    dataset.load(con, gold=gold)
    got = con.execute("""SELECT match_id, bool_or(is_clean), min(exclusion_reason) FROM model_rows
                         GROUP BY 1 ORDER BY 1""").fetchall()
    assert ("1144528", False, "fails_quality_checks") in got
    assert ("433606", True, None) in got


# ---------------------------------------------------------------- home team as of each date

def _m(mid, day, venue="v", team_type="international", t1="A", t2=None):
    return (mid, day, venue, team_type, t1, t2 or "X" + mid)


def test_home_team_needs_three_earlier_internationals():
    out = dataset.home_teams_as_of([_m("m1", "2020-01-01"), _m("m2", "2020-01-02"), _m("m3", "2020-01-03"),
                                    _m("m4", "2020-01-04"), _m("other", "2020-01-04", venue="w")])
    assert out == {"m1": None, "m2": None, "m3": None, "m4": "A", "other": None}


def test_home_team_ignores_same_day_matches_and_club_matches():
    out = dataset.home_teams_as_of([
        _m("m1", "2020-01-01"), _m("m2", "2020-01-02"),
        _m("m3a", "2020-01-03"), _m("m3b", "2020-01-03"),       # a double-header: neither sees the other
        _m("club", "2020-01-04", team_type="club"),               # club cricket does not make a home side
        _m("m5", "2020-01-05")])
    assert out["m3a"] is None and out["m3b"] is None
    assert out["club"] == "A" and out["m5"] == "A"


def test_home_team_is_none_when_two_sides_share_the_ground():
    out = dataset.home_teams_as_of([_m("m1", "2020-01-01", t2="B"), _m("m2", "2020-01-02", t2="B"),
                                    _m("m3", "2020-01-03", t2="B"), _m("m4", "2020-01-04", t2="B")])
    assert out["m4"] is None


# ---------------------------------------------------------------- fitting helpers fed by the table

def test_a_tie_counts_as_half_a_win():
    rows = pd.DataFrame({"batting_won": [1.0, 0.0, 0.5], "k": ["win", "loss", "tie"]})
    out, y, w = train._with_ties_split(rows)
    got = sorted(zip(out["k"], y.tolist(), w.tolist()))
    assert got == [("loss", 0, 1.0), ("tie", 0, 0.5), ("tie", 1, 0.5), ("win", 1, 1.0)]


def test_resource_table_falls_with_wickets_and_rises_with_balls():
    rng = np.random.default_rng(3)
    n = 20000
    first = pd.DataFrame({"scheduled_balls_remaining_before": rng.integers(1, 301, n),
                          "innings_wickets_before": rng.integers(0, 10, n), "innings_runs_before": 0})
    first["innings_total"] = (0.9 * first["scheduled_balls_remaining_before"] * (1 - first["innings_wickets_before"] / 11)
                              + rng.normal(0, 5, n))
    table = train.Resources(first).table
    assert (np.diff(table, axis=0) <= 1e-9).all()       # one more wicket down: never more runs to come
    assert (np.diff(table, axis=1) >= -1e-9).all()      # one more over left: never fewer
    assert table[0, -1] - table[9, -1] > 100            # and wickets matter a lot with the whole innings left
