"""Each test pins a specific, previously fixed bug so it cannot silently return."""

import duckdb
import pytest

from cricket_sim import ball_layers


def _innings_con(innings_rows, match_rows, ball_rows):
    """innings_rows: (match_id, innings, team, is_super_over, target_runs, target_balls, absent_hurt, pen_pre)
    match_rows: (match_id, match_type, method, balls_per_over, scheduled_overs)
    ball_rows: (match_id, innings, delivery_number, runs_total, is_legal, dismissal_count)"""
    con = duckdb.connect()
    con.execute("""CREATE TABLE innings (match_id VARCHAR, innings INT, team VARCHAR, is_super_over BOOL,
        target_runs INT, target_balls INT, absent_hurt VARCHAR, penalty_runs_pre INT, penalty_runs_post INT,
        forfeited BOOL, miscounted_overs VARCHAR)""")
    for r in innings_rows:
        con.execute("INSERT INTO innings VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, false, NULL)", list(r))
    con.execute("""CREATE TABLE matches (match_id VARCHAR, match_type VARCHAR, method VARCHAR,
        balls_per_over INT, scheduled_overs INT)""")
    for r in match_rows:
        con.execute("INSERT INTO matches VALUES (?, ?, ?, ?, ?)", list(r))
    con.execute("""CREATE TABLE balls (match_id VARCHAR, innings INT, delivery_number INT, runs_total INT,
        is_legal BOOL, dismissal_count INT, over INT, legal_balls_in_over_before INT)""")
    for r in ball_rows:  # every ball is legal and every over has 6 balls
        con.execute("INSERT INTO balls VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [*r, (r[2] - 1) // 6, (r[2] - 1) % 6])
    con.execute(f"CREATE VIEW int_innings_state AS {ball_layers.INNINGS_SQL}")
    con.execute(f"CREATE VIEW ball_state_pre AS {ball_layers.STATE_PRE_SQL}")
    return con


def _balls(match_id, innings, n_legal, runs=1):
    return [(match_id, innings, k, runs, True, 0) for k in range(1, n_legal + 1)]


def test_rain_revised_chase_has_no_known_target_or_horizon():
    # DLS target 89 in 8 overs was shown from ball 1
    con = _innings_con(
        [("m", 1, "A", False, None, None, None, None), ("m", 2, "B", False, 89, 48, None, None)],
        [("m", "T20", "D/L", 6, 20)],
        _balls("m", 1, 66) + _balls("m", 2, 48, 2))
    row = con.execute("SELECT target_runs, balls_remaining_before, target_status FROM ball_state_pre "
                      "WHERE innings = 2 AND delivery_number = 1").fetchone()
    assert row == (None, None, "revised_timing_unknown")


def test_unrevised_chase_target_and_horizon():
    con = _innings_con(
        [("m", 1, "A", False, None, None, None, None), ("m", 2, "B", False, 121, 120, None, None)],
        [("m", "T20", None, 6, 20)],
        _balls("m", 1, 120) + _balls("m", 2, 30))
    row = con.execute("SELECT target_runs, runs_required_before, balls_remaining_before, target_status "
                      "FROM ball_state_pre WHERE innings = 2 AND delivery_number = 1").fetchone()
    assert row == (121, 121, 120, "set")


def test_super_over_has_six_balls_two_wickets_and_derived_target():
    # super overs showed 120/300 balls remaining and no target
    con = _innings_con(
        [("m", 1, "A", False, None, None, None, None), ("m", 2, "B", False, 121, 120, None, None),
         ("m", 3, "B", True, None, None, None, None), ("m", 4, "A", True, None, None, None, None)],
        [("m", "T20", None, 6, 20)],
        _balls("m", 1, 120) + _balls("m", 2, 120) + _balls("m", 3, 6, 2) + _balls("m", 4, 6, 2))
    rows = con.execute("SELECT innings, balls_remaining_before, wickets_limit, target_runs FROM ball_state_pre "
                       "WHERE innings IN (3, 4) AND delivery_number = 1 ORDER BY innings").fetchall()
    assert rows == [(3, 6, 2, None), (4, 6, 2, 13)]


def test_first_innings_cut_short_has_unknown_length():
    # a first innings reduced by rain still showed the full allocation
    con = _innings_con(
        [("m", 1, "A", False, None, None, None, None), ("m", 2, "B", False, 150, 180, None, None)],
        [("m", "ODI", "D/L", 6, 50)],
        _balls("m", 1, 180) + _balls("m", 2, 10))
    row = con.execute("SELECT balls_remaining_before, innings_length_status FROM ball_state_pre "
                      "WHERE innings = 1 AND delivery_number = 1").fetchone()
    assert row == (None, "reduced_timing_unknown")


def test_penalty_runs_before_the_innings_count_in_the_score():
    # England started on 5/0 at Rajkot 2024 but state started at 0
    con = _innings_con([("m", 1, "A", False, None, None, None, 5)], [("m", "Test", None, 6, None)], _balls("m", 1, 3))
    assert con.execute("SELECT innings_runs_before FROM ball_state_pre WHERE delivery_number = 1").fetchone() == (5,)


def test_test_fourth_innings_target_and_lead():
    # Tests had no target or lead
    con = _innings_con(
        [("m", 1, "A", False, None, None, None, None), ("m", 2, "B", False, None, None, None, None),
         ("m", 3, "A", False, None, None, None, None), ("m", 4, "B", False, None, None, None, None)],
        [("m", "Test", None, 6, None)],
        _balls("m", 1, 300) + _balls("m", 2, 250) + _balls("m", 3, 100) + _balls("m", 4, 10))
    # A: 300 + 100, B: 250 -> B needs 151
    row = con.execute("SELECT target_runs, match_lead_before, balls_remaining_before FROM ball_state_pre "
                      "WHERE innings = 4 AND delivery_number = 1").fetchone()
    assert row == (151, -150, None)


def test_registry_keeps_predicted_and_observer_labels_out_of_pre_ball():
    pre = set(ball_layers.LAYERS["int_ball_state_pre"][1]) | set(ball_layers.LAYERS["cricinfo_ball_pre"][1])
    for table, (cls, cols) in ball_layers.LAYERS.items():
        if cls in ("observer_label", "result", "post_ball", "predicted_delivery", "predicted_outcome"):
            assert not pre & set(cols), table
    assert "shot_control_cricinfo" in ball_layers.LAYERS["cricinfo_ball_labels"][1]
    assert "hawkeye_link_quality" in ball_layers.LAYERS["hawkeye_ball_delivery"][1]


def _leak_con(pre_leaks: bool):
    con = duckdb.connect()
    b, pre, post = [], [], []
    for n in range(1, 21):
        b.append(f"('m', 1, {n}, {1 if n == 10 else 0})")
        pre.append(f"('m', 1, {n}, {40 if (n > 10 or (pre_leaks and n == 10)) else 50})")
        post.append(f"('m', 1, {n}, {50 if n < 10 else 40})")
    con.execute(f"CREATE VIEW balls AS SELECT * FROM (VALUES {','.join(b)}) t(match_id, innings, delivery_number, dismissal_count)")
    con.execute(f"CREATE VIEW cricinfo_ball_pre AS SELECT * FROM (VALUES {','.join(pre)}) t(match_id, innings, delivery_number, win_probability_before)")
    con.execute(f"CREATE VIEW cricinfo_ball_post AS SELECT * FROM (VALUES {','.join(post)}) t(match_id, innings, delivery_number, win_probability_after)")
    return con


def test_win_probability_leak_is_caught():
    with pytest.raises(SystemExit, match="leak check failed"):
        ball_layers.check_no_leak(_leak_con(pre_leaks=True))


def test_chase_that_passes_its_stated_target_is_flagged_not_trusted():
    # upstream error seen in Cricsheet: innings continued after reaching the stated target
    con = _innings_con(
        [("m", 1, "A", False, None, None, None, None), ("m", 2, "B", False, 10, 120, None, None)],
        [("m", "T20", None, 6, 20)],
        _balls("m", 1, 120) + _balls("m", 2, 15))
    row = con.execute("SELECT target_runs, target_status FROM ball_state_pre WHERE innings = 2 AND delivery_number = 1").fetchone()
    assert row == (None, "inconsistent_with_balls")


def test_chase_passing_a_derived_target_is_flagged():
    # Cricsheet gives no target; first innings 120 -> derived target 121, but the chase passes it early
    con = _innings_con(
        [("m", 1, "A", False, None, None, None, None), ("m", 2, "B", False, None, None, None, None)],
        [("m", "T20", None, 6, 20)],
        _balls("m", 1, 120) + _balls("m", 2, 100, 2))
    row = con.execute("SELECT target_runs, target_status FROM ball_state_pre WHERE innings = 2 AND delivery_number = 1").fetchone()
    assert row == (None, "inconsistent_with_balls")


def test_a_wide_does_not_make_a_short_innings_look_like_a_full_one():
    # Innings length counts LEGAL balls. 119 legal balls plus a wide are 120 deliveries but
    # one ball short of 20 overs, so the first innings (not all out, rain method) did not run its full length.
    matches = [("m", "T20", "D/L", 6, 20)]
    innings = [("m", 1, "A", False, None, None, None, None), ("m", 2, "B", False, 150, 90, None, None)]
    short = _innings_con(innings, matches, _balls("m", 1, 119) + [("m", 1, 120, 1, False, 0)] + _balls("m", 2, 10))
    full = _innings_con(innings, matches, _balls("m", 1, 120) + _balls("m", 2, 10))
    first_ball = "SELECT balls_remaining_before, innings_length_status FROM ball_state_pre WHERE innings = 1 AND delivery_number = 1"
    assert short.execute(first_ball).fetchone() == (None, "reduced_timing_unknown")
    assert full.execute(first_ball).fetchone() == (120, "full_known")
