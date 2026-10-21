import duckdb

from cricket_sim.hawkeye_link import build_hawkeye_link

PEOPLE = [("id_a", "A Smith", "11"), ("id_b", "B Jones", "12"), ("id_c", "C Brown", "13"),
          ("id_x", "X Khan", "21"), ("id_y", "Y Ali", None)]  # Y Ali has no Pulse id -> surname fallback

# (innings, over, ball_in_over, is_legal, batter, bowler, runs_total)
SEQ_M1 = [(1, 0, 1, False, "a", "x", 1), (1, 0, 2, True, "a", "x", 0), (1, 0, 3, True, "a", "x", 4),
          (1, 0, 4, True, "a", "x", 1), (1, 0, 5, True, "b", "x", 0), (1, 0, 6, True, "b", "x", 0),
          (1, 0, 7, True, "b", "x", 6), (1, 1, 1, True, "a", "y", 0), (1, 1, 2, True, "a", "y", 2)]
# Same players, different ball-by-ball story: pair overlap alone cannot tell M1 and M2 apart.
SEQ_M2 = [(1, 0, 1, True, "a", "x", 0), (1, 0, 2, True, "a", "x", 1), (1, 0, 3, True, "b", "x", 1),
          (1, 0, 4, True, "a", "x", 0), (1, 0, 5, True, "a", "x", 0), (1, 0, 6, True, "b", "x", 2),
          (1, 1, 1, True, "a", "y", 1), (1, 1, 2, True, "b", "y", 0)]

NAMES = {"a": ("A Smith", "id_a", "Adam Smith", 11), "b": ("B Jones", "id_b", "Ben Jones", 12),
         "x": ("X Khan", "id_x", "Xavier Khan", 21), "y": ("Y Ali", "id_y", "Yusuf Ali", 99),
         "c": ("C Brown", "id_c", "Carl Brown", 13)}


def _setup(con):
    con.execute("CREATE TABLE people (identifier VARCHAR, name VARCHAR, key_pulse VARCHAR, key_pulse_2 VARCHAR)")
    con.executemany("INSERT INTO people VALUES (?, ?, ?, NULL)", PEOPLE)
    con.execute("CREATE TABLE matches (match_id VARCHAR, match_type VARCHAR, gender VARCHAR, event_name VARCHAR)")
    con.executemany("INSERT INTO matches VALUES (?, ?, 'male', NULL)", [("m1", "ODI"), ("m2", "ODI"), ("t1", "Test")])
    con.execute("CREATE TABLE match_players (match_id VARCHAR, team VARCHAR, player VARCHAR, player_id VARCHAR)")
    con.execute("""CREATE TABLE balls (match_id VARCHAR, innings SMALLINT, delivery_number INT, over SMALLINT,
        ball_in_over SMALLINT, is_legal BOOLEAN, is_super_over BOOLEAN, batter VARCHAR, batter_id VARCHAR,
        bowler VARCHAR, bowler_id VARCHAR, runs_total SMALLINT)""")
    for mid, seq in [("m1", SEQ_M1), ("m2", SEQ_M2), ("t1", SEQ_M1)]:
        for n, (inn, ov, bio, legal, bat, bowl, runs) in enumerate(seq, start=1):
            con.execute("INSERT INTO balls VALUES (?, ?, ?, ?, ?, ?, false, ?, ?, ?, ?, ?)",
                        [mid, inn, n, ov, bio, legal, NAMES[bat][0], NAMES[bat][1],
                         NAMES[bowl][0], NAMES[bowl][1], runs])
    # the staged shape (staging.py has already renamed, typed and turned 'NA' into NULL)
    con.execute("""CREATE TABLE stg_hawkeye_balls (competition VARCHAR, hawkeye_match_id VARCHAR, delivery_code VARCHAR,
        legal_ball_in_over SMALLINT, batter_name VARCHAR, batter_pulse_id VARCHAR, bowler_name VARCHAR,
        bowler_pulse_id VARCHAR, ball_speed_mps DOUBLE, runs_total INT, pitch_x DOUBLE, pitch_y DOUBLE,
        stumps_x DOUBLE, stumps_y DOUBLE, field_x DOUBLE, field_y DOUBLE, right_handed_bat BOOLEAN DEFAULT false)""")
    cols = ("competition, hawkeye_match_id, delivery_code, legal_ball_in_over, batter_name, batter_pulse_id, "
            "bowler_name, bowler_pulse_id, ball_speed_mps, runs_total, pitch_x, pitch_y, stumps_x, stumps_y, field_x, field_y")
    legal_count = 0
    for i, (inn, ov, bio, legal, bat, bowl, runs) in enumerate(SEQ_M1):
        if (ov, bio) == (0, 5):
            continue  # a delivery Hawk-Eye never recorded
        legal_count = legal_count + 1 if legal else legal_count
        speed = -1.0 if i == 2 else 35.0 + i
        pitch = None if i == 2 else 0.1
        field = (5, 5) if i == 1 else (50, 90)
        con.execute(f"INSERT INTO stg_hawkeye_balls ({cols}) VALUES ('mensODI', '500', ?, ?, ?, ?, ?, ?, ?, ?, ?, 6.2, ?, 0.7, ?, ?)",
                    [f"{inn}.{ov + 1}.{bio}", legal_count, NAMES[bat][2], str(NAMES[bat][3]), NAMES[bowl][2],
                     str(NAMES[bowl][3]), speed, runs, pitch, pitch, *field])
    # A Hawk-Eye match with no Cricsheet counterpart
    con.execute(f"INSERT INTO stg_hawkeye_balls ({cols}) VALUES ('mensODI', '777', '1.1.1', 1, 'Carl Brown', '13', "
                "'Xavier Khan', '21', 30.0, 3, 0, 5, 0, 0.5, 40, 40)")


def _run(tmp_path):
    con = duckdb.connect()
    _setup(con)
    build_hawkeye_link(con, tmp_path)
    return con


def test_match_link(tmp_path):
    con = _run(tmp_path)
    rows = con.execute("SELECT hawkeye_match_id, match_id, method, hawkeye_balls, aligned_balls, agreement_rate, "
                       "confidence FROM hawkeye_match_link ORDER BY hawkeye_match_id").fetchall()
    assert rows[0][:6] == ("500", "m1", "pulse_id_pairs+ball_alignment", 8, 8, 1.0)
    assert rows[0][6] > 0.5  # m2 shares every player pair but not the ball sequence
    assert rows[1][:3] == ("777", None, "unmatched")


def test_ball_tracking(tmp_path):
    con = _run(tmp_path)
    rows = con.execute("SELECT delivery_number, speed_kmh, pitch_x, field_x, field_y, link_quality "
                       "FROM hawkeye_ball_tracking ORDER BY delivery_number").fetchall()
    assert [r[0] for r in rows] == [1, 2, 3, 4, 6, 7, 8, 9]  # delivery 5 has no Hawk-Eye row; wide is delivery 1
    assert rows[0][1] == 126.0
    assert rows[1][3:5] == (None, None)      # (5, 5) sentinel -> unknown
    assert rows[2][1:3] == (None, None)      # -1 speed and 'NA' -> unknown
    assert {r[5] for r in rows[:6]} == {"verified"}
    assert {r[5] for r in rows[6:]} == {"verified_surname"}  # bowler Y Ali has no Pulse id


def test_ball_tracking_is_not_persisted(tmp_path):
    _run(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["hawkeye_match_link.parquet"]


def test_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    ca, cb = _run(a), _run(b)
    assert (a / "hawkeye_match_link.parquet").read_bytes() == (b / "hawkeye_match_link.parquet").read_bytes()
    sql = "SELECT * FROM hawkeye_ball_tracking ORDER BY ALL"
    assert ca.execute(sql).fetchall() == cb.execute(sql).fetchall()


# ---- The acceptance threshold (MIN_SCORE = 0.5) is tested on both sides of the edge -----------------

def _link_with_corrupted_balls(tmp_path, n_wrong):
    """Hawk-Eye match 500 has 8 recorded balls that all agree with Cricsheet match m1. Change the runs of the
    first `n_wrong` of them (in delivery order) so only 8 - n_wrong agree, then link."""
    con = duckdb.connect()
    _setup(con)
    codes = ["1.1.1", "1.1.2", "1.1.3", "1.1.4", "1.1.6", "1.1.7", "1.2.1", "1.2.2"]
    if n_wrong:
        con.execute("UPDATE stg_hawkeye_balls SET runs_total = runs_total + 100 WHERE hawkeye_match_id = '500' "
                    f"AND delivery_code IN ({', '.join(repr(c) for c in codes[:n_wrong])})")
    build_hawkeye_link(con, tmp_path)
    return con.execute("SELECT match_id, hawkeye_balls, aligned_balls, agreement_rate FROM hawkeye_match_link "
                       "WHERE hawkeye_match_id = '500'").fetchone()


def test_a_link_needs_at_least_half_of_the_hawkeye_balls_to_agree(tmp_path):
    (tmp_path / "good").mkdir()
    (tmp_path / "edge").mkdir()
    (tmp_path / "poor").mkdir()
    assert _link_with_corrupted_balls(tmp_path / "good", 0)[0] == "m1"          # 8 of 8 agree
    # exactly half (4 of 8) is accepted: the rule is "at least", not "more than"
    match, hawkeye_balls, aligned, rate = _link_with_corrupted_balls(tmp_path / "edge", 4)
    assert (match, hawkeye_balls, aligned) == ("m1", 8, 8)
    assert rate == 0.5
    # 3 of 8 (0.375) is below the threshold: no link, and the match is reported as unmatched, not as a poor link
    match, hawkeye_balls, aligned, rate = _link_with_corrupted_balls(tmp_path / "poor", 5)
    assert match is None and aligned == 0 and rate is None
