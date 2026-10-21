"""Player match stats and the form columns, checked against numbers worked out by hand.

The contracts only check signs and relations between these columns (death balls <= balls, 365-day <= career),
and the leakage tests check which matches count. None of them said what a stat IS, so a wide counted as a ball
faced, a bowler charged with byes, a death over starting one over late or a form window one day too long
went through. Three kinds of check here:
  1. published scorecards of two fixture matches (the 2011 World Cup final and the 2019 semi-final);
  2. one hand-made match that has every awkward ball in it: wide, no-ball, bye, leg bye, a four that was run,
     caught and bowled, run out, a retirement and a super over, plus a T20 and a Test for the death overs;
  3. one hand-made career for the form columns, with matches exactly 365 and 366 days back."""

from datetime import date

import duckdb
import pandas as pd
import pytest

from cricket_sim import gold
from cricket_sim.context_players import MAX_LAST_MATCH_AGE

STAT_COLUMNS = ["bat_balls", "bat_runs", "bat_outs", "bat_fours", "bat_sixes", "bat_dots", "bat_death_balls",
                "bat_death_runs", "bowl_balls", "bowl_runs", "bowl_wickets", "bowl_death_balls", "bowl_death_runs"]


# ---- 1. published scorecards ------------------------------------------------------------------------------

def _scorecard(tiny_build, match_id):
    con = duckdb.connect()
    rows = con.execute(f"""SELECT p.name, s.team, {', '.join('s.' + c for c in STAT_COLUMNS)}
        FROM read_parquet('{tiny_build}/gold/int_player_match_stats.parquet') s
        JOIN read_parquet('{tiny_build}/silver/people.parquet') p ON p.identifier = s.player_id
        WHERE s.match_id = '{match_id}'""").fetchall()
    return {r[0]: dict(zip(["team"] + STAT_COLUMNS, r[1:])) for r in rows}


def test_2011_world_cup_final_scorecard(tiny_build):
    """Wankhede, 2 April 2011. India 277/4 chasing Sri Lanka's 274/6. From the published scorecard."""
    card = _scorecard(tiny_build, "433606")
    batting = {  # name: (balls faced, runs, fours, sixes, out?)
        "G Gambhir": (122, 97, 9, 0, 1), "MS Dhoni": (79, 91, 8, 2, 0), "V Kohli": (49, 35, 4, 0, 1),
        "V Sehwag": (2, 0, 0, 0, 1), "SR Tendulkar": (14, 18, 2, 0, 1), "Yuvraj Singh": (24, 21, 2, 0, 0),
        "DPMD Jayawardene": (88, 103, 13, 0, 0), "KC Sangakkara": (67, 48, 5, 0, 1), "NLTC Perera": (9, 22, 3, 1, 0)}
    for name, (balls, runs, fours, sixes, out) in batting.items():
        got = card[name]
        assert (got["bat_balls"], got["bat_runs"], got["bat_fours"], got["bat_sixes"], got["bat_outs"]) == \
            (balls, runs, fours, sixes, out), name
    bowling = {  # name: (legal balls, runs conceded, wickets)
        "SL Malinga": (54, 42, 2), "Z Khan": (60, 60, 2), "Harbhajan Singh": (60, 50, 1), "Yuvraj Singh": (60, 49, 2),
        "M Muralitharan": (48, 39, 0), "KMDN Kulasekara": (50, 64, 0)}
    for name, (balls, runs, wickets) in bowling.items():
        got = card[name]
        assert (got["bowl_balls"], got["bowl_runs"], got["bowl_wickets"]) == (balls, runs, wickets), name


def test_2019_world_cup_semi_final_scorecard_including_death_overs(tiny_build):
    """Old Trafford, 9 and 10 July 2019. New Zealand 239/8, India all out 221. Death = overs 41 to 50."""
    card = _scorecard(tiny_build, "1144528")
    assert (card["MS Dhoni"]["bat_balls"], card["MS Dhoni"]["bat_runs"], card["MS Dhoni"]["bat_outs"]) == (72, 50, 1)
    assert (card["RA Jadeja"]["bat_balls"], card["RA Jadeja"]["bat_runs"], card["RA Jadeja"]["bat_fours"],
            card["RA Jadeja"]["bat_sixes"]) == (59, 77, 4, 4)
    assert (card["LRPL Taylor"]["bat_balls"], card["LRPL Taylor"]["bat_runs"]) == (90, 74)
    assert (card["KS Williamson"]["bat_balls"], card["KS Williamson"]["bat_runs"]) == (95, 67)
    for name, (balls, runs, wickets) in {"MJ Henry": (60, 37, 3), "TA Boult": (60, 42, 2), "B Kumar": (60, 43, 3),
                                         "JJ Bumrah": (60, 39, 1), "YS Chahal": (60, 63, 1), "HH Pandya": (60, 55, 1)}.items():
        got = card[name]
        assert (got["bowl_balls"], got["bowl_runs"], got["bowl_wickets"]) == (balls, runs, wickets), name
    # death overs: Bumrah bowled 3 of his 10 overs from the 41st (3 overs = 18 balls) and went for 21
    assert (card["JJ Bumrah"]["bowl_death_balls"], card["JJ Bumrah"]["bowl_death_runs"]) == (18, 21)
    # Jadeja and Dhoni batted in the death overs: 25 and 26 balls, 38 and 26 runs
    assert (card["RA Jadeja"]["bat_death_balls"], card["RA Jadeja"]["bat_death_runs"]) == (25, 38)
    assert (card["MS Dhoni"]["bat_death_balls"], card["MS Dhoni"]["bat_death_runs"]) == (26, 26)


# ---- 2. one hand-made match with every awkward ball ------------------------------------------------------

def _ball(match, innings, n, over, batter, bowler, runs_batter=0, *, legal=True, wides=0, noballs=0, byes=0,
          legbyes=0, non_boundary=False, super_over=False):
    extras = wides + noballs + byes + legbyes
    return (match, innings, n, over, batter, bowler, legal, super_over, wides, noballs, byes, legbyes,
            runs_batter, non_boundary, runs_batter + extras)


def _stats_connection():
    con = duckdb.connect()
    con.execute("""CREATE TABLE matches (match_id VARCHAR, start_date DATE, gender VARCHAR, match_type VARCHAR)""")
    con.executemany("INSERT INTO matches VALUES (?, ?, 'male', ?)", [
        ("odi", date(2020, 1, 1), "ODI"), ("t20", date(2020, 1, 2), "T20"), ("test", date(2020, 1, 3), "Test")])
    con.execute("""CREATE TABLE balls (match_id VARCHAR, innings SMALLINT, delivery_number INTEGER, over SMALLINT,
                   batter_id VARCHAR, bowler_id VARCHAR, is_legal BOOLEAN, is_super_over BOOLEAN, wides SMALLINT,
                   noballs SMALLINT, byes SMALLINT, legbyes SMALLINT, runs_batter SMALLINT,
                   runs_non_boundary BOOLEAN, runs_total SMALLINT)""")
    odi = [
        _ball("odi", 1, 1, 0, "bat1", "bwl1", 4),                       # a four
        _ball("odi", 1, 2, 0, "bat1", "bwl1", legal=False, wides=1),    # a wide: no ball faced, 1 run to the bowler
        _ball("odi", 1, 3, 0, "bat1", "bwl1", 6, legal=False, noballs=1),  # no-ball hit for six: faced, 7 to the bowler
        _ball("odi", 1, 4, 0, "bat1", "bwl1", byes=2),                  # 2 byes: faced (a dot), nothing to the bowler
        _ball("odi", 1, 5, 0, "bat1", "bwl1", legbyes=1),               # a leg bye: the same
        _ball("odi", 1, 6, 0, "bat1", "bwl1", 4, non_boundary=True),    # four runs run out: 4 runs, not a four
        _ball("odi", 1, 7, 0, "bat1", "bwl1", 0),                       # a dot
        _ball("odi", 1, 8, 0, "bat1", "bwl1", 0),                       # caught and bowled
        _ball("odi", 1, 9, 1, "bat2", "bwl2", 1),
        _ball("odi", 1, 10, 1, "bat2", "bwl2", 0),                      # run out: not the bowler's wicket
        _ball("odi", 1, 11, 39, "bat3", "bwl2", 2),                     # the 40th over: not yet a death over
        _ball("odi", 1, 12, 40, "bat3", "bwl2", 6),                     # the 41st over: death
        _ball("odi", 1, 13, 40, "bat3", "bwl2", legal=False, wides=1),  # a death-over wide
        _ball("odi", 1, 14, 40, "bat3", "bwl2", 0),                     # bat3 retires not out on this ball
        _ball("odi", 3, 1, 0, "bat1", "bwl1", 6, super_over=True),      # super over: not in anyone's stats
    ]
    t20 = [_ball("t20", 1, 1, 15, "bat1", "bwl1", 1), _ball("t20", 1, 2, 16, "bat1", "bwl1", 4)]
    test = [_ball("test", 1, 1, 90, "bat1", "bwl1", 1)]
    con.executemany("INSERT INTO balls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", odi + t20 + test)
    con.execute("""CREATE TABLE wickets (match_id VARCHAR, innings SMALLINT, delivery_number INTEGER,
                   player_out_id VARCHAR, kind VARCHAR, is_retirement BOOLEAN)""")
    con.executemany("INSERT INTO wickets VALUES (?, ?, ?, ?, ?, ?)", [
        ("odi", 1, 8, "bat1", "caught and bowled", False), ("odi", 1, 10, "bat2", "run out", False),
        ("odi", 1, 14, "bat3", "retired not out", True)])
    con.execute("CREATE TABLE match_players (match_id VARCHAR, team VARCHAR, player VARCHAR, player_id VARCHAR)")
    people = [("X", "bat1"), ("X", "bat2"), ("X", "bat3"), ("X", "bench"), ("Y", "bwl1"), ("Y", "bwl2")]
    con.executemany("INSERT INTO match_players VALUES (?, ?, ?, ?)",
                    [(m, team, pid, pid) for m in ("odi", "t20", "test") for team, pid in people])
    return con


def test_player_stats_of_a_match_with_every_awkward_ball():
    con = _stats_connection()
    df = con.execute(f"SELECT * FROM ({gold.PLAYER_STATS_SQL})").fetchdf()
    got = {(r["match_id"], r["player_id"]): tuple(int(r[c]) for c in STAT_COLUMNS) for _, r in df.iterrows()}
    families = {r["match_id"]: r["format_family"] for _, r in df.iterrows()}
    assert families == {"odi": "one_day", "t20": "t20", "test": "multi_day"}
    #                         bat: balls runs outs 4s 6s dots | death balls runs | bowl: balls runs wkts | death balls runs
    zero = (0,) * 13
    assert got == {
        # bat1: faced 7 (not the wide): 4 + 6 + 0 + 0 + 4 + 0 + 0 = 14 runs; one four (the run four is not), one six,
        # dots 4 (bye, leg bye, dot, c&b ball); out once (caught and bowled); the super-over six is not counted
        ("odi", "bat1"): (7, 14, 1, 1, 1, 4, 0, 0, 0, 0, 0, 0, 0),
        # bat2: faced 2, 1 run, run out, one dot
        ("odi", "bat2"): (2, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
        # bat3: faced the 2, the 6 and the retirement ball (not the wide) = 3, 8 runs, NOT out (retired), one six,
        # one dot; death overs: the six and the retirement ball (2 balls), 6 runs
        ("odi", "bat3"): (3, 8, 0, 0, 1, 1, 2, 6, 0, 0, 0, 0, 0),
        ("odi", "bench"): zero,
        # bwl1: legal balls 1, 4, 5, 6, 7, 8 = 6 (not the wide, not the no-ball); runs = bat runs + wides + no-balls:
        # 4 + 1 + 7 + 0 + 0 + 4 + 0 + 0 = 16 (byes and leg byes are not his); one wicket (caught and bowled)
        ("odi", "bwl1"): (0, 0, 0, 0, 0, 0, 0, 0, 6, 16, 1, 0, 0),
        # bwl2: legal 1, 0, 2, 6, 0 = 5 balls; runs 1 + 0 + 2 + 6 + 1 (wide) + 0 = 10; no wickets (a run out and a
        # retirement are not the bowler's); death: the six and the retirement ball = 2 legal balls, 6 + 1 + 0 = 7 runs
        ("odi", "bwl2"): (0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 0, 2, 7),
        # T20: death overs are the last four (over index 16 on); over 15 is not death, over 16 is
        ("t20", "bat1"): (2, 5, 0, 1, 0, 0, 1, 4, 0, 0, 0, 0, 0),
        ("t20", "bwl1"): (0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 0, 1, 4),
        # a Test has no death overs at all
        ("test", "bat1"): (1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        ("test", "bwl1"): (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0),
        **{(m, p): zero for m in ("t20", "test") for p in ("bat2", "bat3", "bench", "bwl2")},
    }


# ---- 3. the form columns: what a player had done BEFORE a match ---------------------------------------------

def _form_row(mid, day, family, bat_balls, bat_runs, bat_outs, bat_death_balls, bat_death_runs,
              bowl_balls, bowl_runs, bowl_wickets, bowl_death_balls, bowl_death_runs, player="p1"):
    return (mid, player, "X", day, family, bat_balls, bat_runs, bat_outs, 0, 0, 0, bat_death_balls, bat_death_runs,
            bowl_balls, bowl_runs, bowl_wickets, bowl_death_balls, bowl_death_runs)


def _form(rows, dobs):
    con = duckdb.connect()
    con.execute("""CREATE TABLE int_player_match_stats (match_id VARCHAR, player_id VARCHAR, team VARCHAR, start_date DATE,
                   format_family VARCHAR, bat_balls BIGINT, bat_runs BIGINT, bat_outs BIGINT, bat_fours BIGINT,
                   bat_sixes BIGINT, bat_dots BIGINT, bat_death_balls BIGINT, bat_death_runs BIGINT, bowl_balls BIGINT,
                   bowl_runs BIGINT, bowl_wickets BIGINT, bowl_death_balls BIGINT, bowl_death_runs BIGINT)""")
    con.executemany("INSERT INTO int_player_match_stats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    con.execute("CREATE TABLE scope AS SELECT DISTINCT match_id FROM int_player_match_stats")
    con.execute("CREATE TABLE players (cricsheet_id VARCHAR, dob DATE)")
    con.executemany("INSERT INTO players VALUES (?, ?)", dobs)
    df = con.execute(gold.PLAYER_FORM_SQL).fetchdf()
    return {(r["match_id"], r["player_id"]): r for _, r in df.iterrows()}


D = date(2021, 1, 1)


def _career():
    from datetime import timedelta
    return [
        # 366 days before D: in the career, one day OUTSIDE the 365-day window
        _form_row("a", D - timedelta(days=366), "one_day", 10, 10, 1, 5, 8, 12, 18, 1, 6, 10),
        # exactly 365 days before D: INSIDE the 365-day window
        _form_row("b", D - timedelta(days=365), "one_day", 20, 20, 1, 0, 0, 24, 24, 0, 0, 0),
        # the day before D
        _form_row("c", D - timedelta(days=1), "one_day", 20, 40, 0, 10, 20, 6, 3, 2, 6, 12),
        # the same day as the match we look at, and its own match: never counted
        _form_row("t", D, "one_day", 99, 99, 1, 9, 9, 99, 99, 9, 9, 9),
        # another format on an earlier day: a different form table
        _form_row("t20", D - timedelta(days=2), "t20", 50, 100, 1, 0, 0, 0, 0, 0, 0, 0),
        # another player on an earlier day
        _form_row("q", D - timedelta(days=3), "one_day", 77, 77, 7, 7, 7, 77, 77, 7, 7, 7, player="p2"),
    ]


def test_player_form_before_a_match_from_a_hand_made_career():
    got = _form(_career(), [("p1", date(1990, 6, 15)), ("p2", None)])[("t", "p1")]
    # career before D, one-day cricket only: matches a, b, c
    assert got["prior_matches"] == 3 and got["prior_batting_matches"] == 3 and got["prior_bowling_matches"] == 3
    assert pd.Timestamp(got["prior_first_match_date"]).date() == date(2020, 1, 1) and got["days_since_last_match"] == 1
    assert (got["prior_bat_balls"], got["prior_bat_runs"], got["prior_bat_outs"]) == (50, 70, 2)
    assert (got["prior_bowl_balls"], got["prior_bowl_runs"], got["prior_bowl_wickets"]) == (42, 45, 3)
    assert (got["prior_bat_death_balls"], got["prior_bat_death_runs"]) == (15, 28)
    assert (got["prior_bowl_death_balls"], got["prior_bowl_death_runs"]) == (12, 22)
    assert got["prior_strike_rate"] == 140.0                          # 100 * 70 / 50
    assert got["prior_batting_average"] == 35.0                       # 70 / 2 outs
    assert got["prior_death_strike_rate"] == pytest.approx(186.67)    # 100 * 28 / 15
    assert got["prior_economy"] == pytest.approx(6.429)               # 6 * 45 / 42
    assert got["prior_bowling_average"] == 15.0                       # 45 / 3 wickets
    assert got["prior_death_economy"] == 11.0                         # 6 * 22 / 12
    # the 365-day window: b and c, NOT a (366 days back) and NOT the match's own day
    assert got["prior_365d_matches"] == 2
    assert (got["prior_365d_bat_balls"], got["prior_365d_bat_runs"], got["prior_365d_bat_outs"]) == (40, 60, 1)
    assert (got["prior_365d_bowl_balls"], got["prior_365d_bowl_runs"], got["prior_365d_bowl_wickets"]) == (30, 27, 2)
    assert got["prior_365d_strike_rate"] == 150.0                     # 100 * 60 / 40
    assert got["prior_365d_economy"] == pytest.approx(5.4)            # 6 * 27 / 30
    # age as of the match: born 15 June 1990, so 30 on 1 January 2021 (30.55 years)
    assert got["age_years"] == 30


def test_player_form_with_no_history_is_zero_counts_and_null_rates():
    rows = _career()
    first = _form(rows, [("p1", date(1990, 6, 15))])[("a", "p1")]
    assert first["prior_matches"] == 0 and first["prior_bat_runs"] == 0 and first["prior_365d_matches"] == 0
    for rate in ("prior_strike_rate", "prior_batting_average", "prior_economy", "prior_bowling_average",
                 "prior_death_strike_rate", "prior_death_economy", "prior_365d_strike_rate", "prior_365d_economy"):
        assert pd.isna(first[rate]), rate                       # NULL (NaN), never 0 or an error
    assert pd.isna(first["days_since_last_match"])


def test_player_form_age_is_as_of_the_match_and_absent_when_implausible():
    # p1 born 1 January 1960: 61 on D, over MAX_LAST_MATCH_AGE (50), so no age. p2 born 1 January 1980 played
    # match q on 29 December 2020: 40 then (40.99 years), not 41 (the age at the end of the snapshot)
    dobs = [("p1", date(1960, 1, 1)), ("p2", date(1980, 1, 1))]
    rows = _career()
    assert MAX_LAST_MATCH_AGE == 50
    form = _form(rows, dobs)
    assert pd.isna(form[("t", "p1")]["age_years"])
    assert form[("q", "p2")]["age_years"] == 40
    # a player with no birth date has no age
    assert pd.isna(_form(rows, [("p1", None)])[("t", "p1")]["age_years"])
