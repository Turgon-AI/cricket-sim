"""Post-innings penalty runs reach the next innings only once the awarded innings is complete.

Cricsheet stores a post-innings award against the innings receiving the runs. Once that innings is over
the award is part of its total, so the nominal target and the Test lead of every LATER innings include it.
The award must still never show in a pre-ball value of its own innings or an earlier one.
"""

import duckdb
import pytest

from cricket_sim import ball_layers, contracts


def _connection(kind="Test"):
    con = duckdb.connect()
    con.execute("""CREATE TABLE matches (match_id VARCHAR, match_type VARCHAR, method VARCHAR,
                   balls_per_over SMALLINT, scheduled_overs SMALLINT)""")
    con.execute("INSERT INTO matches VALUES ('m', ?, NULL, 6, ?)",
                ["Test" if kind == "Test" else "T20", None if kind == "Test" else 20])
    con.execute("""CREATE TABLE innings (match_id VARCHAR, innings SMALLINT, team VARCHAR,
                   is_super_over BOOLEAN, target_runs INTEGER, target_balls INTEGER,
                   forfeited BOOLEAN, absent_hurt VARCHAR, penalty_runs_pre SMALLINT,
                   penalty_runs_post SMALLINT, miscounted_overs VARCHAR)""")
    con.execute("""CREATE TABLE balls (match_id VARCHAR, innings SMALLINT, delivery_number INTEGER,
                   runs_total SMALLINT, is_legal BOOLEAN, dismissal_count SMALLINT, over SMALLINT,
                   legal_balls_in_over_before SMALLINT)""")
    # First innings: 3 awarded before play, plus 1, 5, 2 on the three deliveries = 11 timed runs.
    # The five runs on delivery 2 stand in for a penalty explicitly recorded on that delivery.
    # Second innings: 2 pre-innings runs plus 4 and 0 = 6. Third innings: 6. Fourth starts with 4.
    innings = [(1, "A", 3, [1, 5, 2]), (2, "B", 2, [4, 0])]
    if kind == "Test":
        innings += [(3, "A", 0, [6]), (4, "B", 4, [1, 0])]
    for number, team, pen_pre, runs in innings:
        con.execute("INSERT INTO innings VALUES ('m', ?, ?, ?, NULL, NULL, false, NULL, ?, NULL, NULL)",
                    [number, team, kind == "super_over", pen_pre])
        con.executemany("INSERT INTO balls VALUES ('m', ?, ?, ?, true, 0, ?, ?)",
                        [(number, i, r, (i - 1) // 6, (i - 1) % 6) for i, r in enumerate(runs, 1)])
    con.execute(f"CREATE VIEW int_innings_state AS {ball_layers.INNINGS_SQL}")
    con.execute(f"CREATE VIEW state AS {ball_layers.STATE_PRE_SQL}")
    return con


def _pre_ball_rows(con):
    contract = contracts.get("int_ball_state_pre")
    columns = [c["name"] for c in contract.columns
               if c.get("primaryKey") or contract.column_props(c).get("timingClass") == "pre_ball"]
    return con.execute(f"SELECT {', '.join(columns)} FROM state ORDER BY innings, delivery_number").fetchall()


@pytest.mark.parametrize("kind", ["Test", "limited", "super_over"])
def test_pre_ball_state_ignores_awards_to_its_own_and_later_innings(kind):
    with _connection(kind) as con:
        expected = _pre_ball_rows(con)
        # An award is part of the innings that received it, so nothing in that innings or before it may move
        # (the award could have been credited during that innings, or during a later one).
        # Removing the eventual award must also preserve every pre_ball value, including NULL membership.
        for innings in range(1, 5 if kind == "Test" else 3):
            for award in (5, 50, 0, None):
                con.execute("UPDATE innings SET penalty_runs_post = ? WHERE innings = ?", [award, innings])
                same = [row for row in _pre_ball_rows(con) if row[1] <= innings]
                assert same == [row for row in expected if row[1] <= innings], (kind, innings, award)


def test_test_lead_and_nominal_target_include_the_awards_of_completed_innings():
    with _connection() as con:
        con.execute("UPDATE innings SET penalty_runs_post = 5 WHERE innings = 1")
        con.execute("UPDATE innings SET penalty_runs_post = 7 WHERE innings = 2")
        # A has (11+5)+6 = 22, B has 6+7 = 13. B starts innings 4 on 4/0: lead -9, target 10, and 6 still needed.
        # The nominal target is the target (it once was 12 against a target of 10).
        assert con.execute("""SELECT test_lead_at_start, nominal_target_runs, target_runs
                            FROM int_innings_state WHERE innings = 4""").fetchone() == (-9, 10, 10)
        assert con.execute("""SELECT match_lead_before, nominal_runs_required_before
                            FROM state WHERE innings = 4 AND delivery_number = 1""").fetchone() == (-5, 6)


@pytest.mark.parametrize("kind", ["limited", "super_over"])
def test_chase_nominal_target_includes_the_post_award_of_the_first_innings(kind):
    with _connection(kind) as con:
        con.execute("UPDATE innings SET penalty_runs_post = 5 WHERE innings = 1")
        assert con.execute("""SELECT nominal_target_runs, target_runs FROM int_innings_state
                            WHERE innings = 2""").fetchone() == (17, 17)


def test_pre_innings_and_delivery_penalties_still_enter_at_their_recorded_time():
    with _connection() as con:
        # Delivery 2's five runs only appear before delivery 3; pre-innings runs appear from ball 1.
        assert con.execute("""SELECT innings_runs_before, match_lead_before FROM state
                            WHERE innings = 1 ORDER BY delivery_number""").fetchall() == [(3, 3), (4, 4), (9, 9)]
        # Completed timed innings are still fully counted when the opposing side begins.
        assert con.execute("""SELECT match_lead_before FROM state
                            WHERE innings = 2 AND delivery_number = 1""").fetchone() == (-9,)
