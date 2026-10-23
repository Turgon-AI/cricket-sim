"""Match rules: format bounds on scheduled overs, 5-ball and 7-ball overs, post-innings penalties in the
nominal target and lead, the known-chase contract check, results that need events outside the ball stream and
the scope.yml schema. Each test pins one rule."""

import json
import zipfile

import duckdb
import pyarrow.parquet as pq
import pytest

from cricket_sim import ball_layers, contracts, gold
from cricket_sim.cricsheet_parse import parse_zip


# ---- helpers -----------------------------------------------------------------------------------------

def _rule(table: str, name: str) -> str:
    """SQL of the named quality rule of a contract (table level or column level)."""
    c = contracts.get(table)
    rules = list(c.obj.get("quality") or []) + [r for col in c.columns for r in col.get("quality") or []]
    found = [r for r in rules if r.get("name") == name]
    assert found, f"{table} has no quality rule named {name}"
    return found[0]["query"]


def _rule_hits(table: str, name: str, columns: str, rows: str) -> int:
    """Run a contract SQL rule on a few literal rows; the result is the number of violations."""
    con = duckdb.connect()
    con.execute(f"CREATE TABLE t AS SELECT * FROM (VALUES {rows}) v({columns})")
    return con.execute(_rule(table, name).replace("${table}", "t")).fetchone()[0]


def _match_json(match_type="T20", overs=20, target_overs="20"):
    ball = {"batter": "a1", "bowler": "b1", "non_striker": "a2", "runs": {"batter": 1, "extras": 0, "total": 1}}
    return {
        "meta": {"data_version": "1.1.0", "created": "2026-01-01", "revision": 1},
        "info": {"dates": ["2026-01-02"], "match_type": match_type, "gender": "male", "team_type": "international",
                 "teams": ["A", "B"], "toss": {"winner": "A", "decision": "bat"},
                 "outcome": {"winner": "A", "by": {"runs": 5}}, "overs": overs, "balls_per_over": 6,
                 "season": "2026", "venue": "Somewhere", "players": {"A": ["a1", "a2"], "B": ["b1"]},
                 "registry": {"people": {"a1": "id_a1", "a2": "id_a2", "b1": "id_b1"}}},
        "innings": [{"team": "A", "overs": [{"over": 0, "deliveries": [ball]}]},
                    {"team": "B", "target": {"runs": 2, "overs": target_overs},
                     "overs": [{"over": 0, "deliveries": [ball]}]}],
    }


def _parse(tmp_path, doc):
    zpath = tmp_path / "all.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("1.json", json.dumps(doc))
    parts = parse_zip(zpath, tmp_path / "out")
    return pq.read_table(parts["matches"][0]).to_pylist()[0]


# ---- Scheduled overs by format -------------------------------------------------------

@pytest.mark.parametrize("match_type", ["T20", "IT20"])
def test_t20_with_wrong_cricsheet_overs_gets_the_format_overs(tmp_path, match_type):
    """Five T20Is carry Cricsheet info.overs = 50 (a source error: the chase target is 'in 20 overs')."""
    match = _parse(tmp_path, _match_json(match_type, overs=50))
    assert match["scheduled_overs"] == 20
    # the correction is visible: the value Cricsheet wrote stays in its own column
    assert match["scheduled_overs_source"] == 50


def test_correct_overs_are_left_alone(tmp_path):
    assert _parse(tmp_path, _match_json("T20", overs=20))["scheduled_overs"] == 20
    (tmp_path / "odi").mkdir()
    (tmp_path / "test").mkdir()
    odi = _parse(tmp_path / "odi", _match_json("ODI", overs=50, target_overs="50"))
    assert (odi["scheduled_overs"], odi["scheduled_overs_source"]) == (50, 50)
    test = _match_json("Test", overs=None)
    test["info"].pop("overs")
    assert _parse(tmp_path / "test", test)["scheduled_overs"] is None


@pytest.mark.parametrize("table", ["matches", "dim_match"])
def test_contract_bounds_scheduled_overs_by_format(table):
    rule = "scheduled_overs_match_format"
    cols = "match_type, scheduled_overs"
    assert _rule_hits(table, rule, cols, "('T20', 50)") == 1          # the 5 bad T20Is
    assert _rule_hits(table, rule, cols, "('ODI', 20)") == 1
    assert _rule_hits(table, rule, cols, "('Test', 50)") == 1         # Tests have no over limit
    assert _rule_hits(table, rule, cols, "('T20', 20), ('ODI', 50), ('Test', NULL)") == 0


# ---- balls and innings builders for ball_layers ------------------------------------------------------

def _con(matches, innings, balls):
    """matches: (match_id, match_type, method, bpo, scheduled_overs)
    innings: (match_id, innings, team, is_super_over, target_runs, target_balls, pen_pre, pen_post, miscounted_overs)
    balls: (match_id, innings, delivery_number, over, legal_balls_in_over_before, runs_total, is_legal)"""
    con = duckdb.connect()
    con.execute("""CREATE TABLE matches (match_id VARCHAR, match_type VARCHAR, method VARCHAR,
                   balls_per_over SMALLINT, scheduled_overs SMALLINT)""")
    con.executemany("INSERT INTO matches VALUES (?, ?, ?, ?, ?)", matches)
    con.execute("""CREATE TABLE innings (match_id VARCHAR, innings SMALLINT, team VARCHAR, is_super_over BOOLEAN,
                   target_runs INTEGER, target_balls INTEGER, forfeited BOOLEAN, absent_hurt VARCHAR,
                   penalty_runs_pre SMALLINT, penalty_runs_post SMALLINT, miscounted_overs VARCHAR)""")
    con.executemany("INSERT INTO innings VALUES (?, ?, ?, ?, ?, ?, false, NULL, ?, ?, ?)", innings)
    con.execute("""CREATE TABLE balls (match_id VARCHAR, innings SMALLINT, delivery_number INTEGER, over SMALLINT,
                   legal_balls_in_over_before SMALLINT, runs_total SMALLINT, is_legal BOOLEAN,
                   dismissal_count SMALLINT)""")
    con.executemany("INSERT INTO balls VALUES (?, ?, ?, ?, ?, ?, ?, 0)", balls)
    con.execute(f"CREATE VIEW int_innings_state AS {ball_layers.INNINGS_SQL}")
    con.execute(f"CREATE VIEW state AS {ball_layers.STATE_PRE_SQL}")
    return con


def _overs(match_id, innings, sizes, runs=1, first_over=0):
    """One legal ball per listed ball; `sizes` is the number of legal balls in each over."""
    rows, n = [], 0
    for over, size in enumerate(sizes, start=first_over):
        for k in range(size):
            n += 1
            rows.append((match_id, innings, n, over, k, runs, True))
    return rows


# ---- 5-ball and 7-ball overs ------------------------------------------------------------------

def test_chase_with_a_listed_7_ball_over_keeps_its_target():
    """A 50-over chase whose umpire listed one 7-ball over has 301 legal balls: that is not an upstream error
    (257470, 289110 and 1486772 were wrongly blanked as inconsistent_with_balls)."""
    sizes = [6] * 10 + [7] + [6] * 39                     # 50 overs, 301 legal balls
    con = _con([("m", "ODI", None, 6, 50)],
               [("m", 1, "A", False, None, None, None, None, None),
                ("m", 2, "B", False, 302, 300, None, None, json.dumps({"10": {"balls": "7"}}))],
               _overs("m", 1, [6] * 50) + _overs("m", 2, sizes))
    status, target, limit = con.execute(
        "SELECT target_status, target_runs, innings_balls_limit FROM int_innings_state WHERE innings = 2").fetchone()
    assert (status, target, limit) == ("set", 302, 301)
    last = con.execute("SELECT balls_remaining_before, scheduled_balls_remaining_before FROM state "
                       "WHERE innings = 2 ORDER BY delivery_number DESC LIMIT 1").fetchone()
    assert last == (1, 1)  # the last legal ball always has one ball left, never 0


def test_balls_left_follow_the_overs_after_a_5_ball_over():
    sizes = [6, 6, 6, 5] + [6] * 46                       # 299 legal balls in 50 overs
    con = _con([("m", "ODI", None, 6, 50)],
               [("m", 1, "A", False, None, None, None, None, json.dumps({"3": {"balls": 5}}))],
               _overs("m", 1, sizes))
    # first ball of over 4: 46 overs are left (276 balls), not 300 - 23 = 277
    row = con.execute("""SELECT s.scheduled_balls_remaining_before, s.balls_remaining_before, s.legal_balls_before
                         FROM state s WHERE s.delivery_number = 24""").fetchone()
    assert row == (276, 276, 23)
    assert con.execute("SELECT innings_balls_limit, innings_length_status FROM int_innings_state").fetchone() == (
        299, "full_known")


def test_balls_left_in_a_listed_over_never_reach_zero_on_a_legal_ball():
    sizes = [6] * 49 + [7]                                # 7-ball last over
    con = _con([("m", "ODI", None, 6, 50)],
               [("m", 1, "A", False, None, None, None, None, json.dumps({"49": {"balls": 7}}))],
               _overs("m", 1, sizes))
    assert con.execute("SELECT min(scheduled_balls_remaining_before), min(balls_remaining_before) FROM state"
                       ).fetchone() == (1, 1)


# ---- Post-innings penalties flow into the nominal target and the lead -------------------------

def _test_match(post):
    """Test: A 10, B 20, A 5 + `post` penalty runs in innings 1, B chases."""
    innings = [("m", 1, "A", False, None, None, None, post, None), ("m", 2, "B", False, None, None, None, None, None),
               ("m", 3, "A", False, None, None, None, None, None), ("m", 4, "B", False, None, None, None, None, None)]
    balls = (_overs("m", 1, [1], runs=10) + _overs("m", 2, [1], runs=20) + _overs("m", 3, [1], runs=5)
             + _overs("m", 4, [1], runs=1))
    return _con([("m", "Test", None, 6, None)], innings, balls)


def test_test_nominal_target_and_lead_include_post_innings_penalties():
    con = _test_match(post=5)
    # A has 10 + 5 = 15 then 5, B has 20. A trails by 5 only if the award is left out.
    row = con.execute("SELECT test_lead_at_start, nominal_target_runs, target_runs FROM int_innings_state "
                      "WHERE innings = 4").fetchone()
    assert row == (0, 1, 1)
    assert con.execute("SELECT match_lead_before FROM state WHERE innings = 2 AND delivery_number = 1").fetchone() == (-15,)


def test_nominal_target_equals_target_runs_whenever_the_target_is_unrevised():
    for post in (None, 5):
        con = _test_match(post)
        assert con.execute("SELECT count(*) FROM int_innings_state WHERE target_status = 'test_derived' "
                           "AND nominal_target_runs <> target_runs").fetchone() == (0,)
    # limited overs and super overs
    for kind, bpo, overs in (("T20", 6, 20),):
        con = _con([("m", kind, None, bpo, overs)],
                   [("m", 1, "A", False, None, None, None, 5, None), ("m", 2, "B", False, 12, 120, None, None, None)],
                   _overs("m", 1, [6], runs=1) + _overs("m", 2, [6], runs=1))
        assert con.execute("SELECT nominal_target_runs, target_runs FROM int_innings_state WHERE innings = 2"
                           ).fetchone() == (12, 12)


def test_nominal_target_rule_is_a_contract_check():
    cols = "target_status, nominal_target_runs, target_runs"
    for table in ("fact_innings", "fact_delivery"):
        assert _rule_hits(table, "nominal_target_equals_target", cols, "('test_derived', 296, 301)") == 1
        assert _rule_hits(table, "nominal_target_equals_target",
                          cols, "('test_derived', 301, 301), ('revised_timing_unknown', NULL, NULL), "
                                "('none', NULL, NULL)") == 0


# ---- known_chase_not_finished -----------------------------------------------------------------

@pytest.mark.parametrize("table", ["fact_delivery", "int_ball_state_pre"])
def test_known_chase_check_catches_a_ball_with_zero_balls_left(table):
    cols = "target_status, runs_required_before, balls_remaining_before"
    assert _rule_hits(table, "known_chase_not_finished", cols, "('set', 5, 0)") == 1
    assert _rule_hits(table, "known_chase_not_finished", cols, "('set', 5, 1), ('set', 1, 1), ('none', 0, 0)") == 0


# ---- A result that needs events outside the ball stream ---------------------------------------

def _quality(second_innings_runs, post):
    """A scored 10 off the balls. B won by 9 wickets: `second_innings_runs` off the balls, plus `post` penalty runs."""
    con = duckdb.connect()
    con.execute("""CREATE TABLE matches (match_id VARCHAR, winner VARCHAR, win_by_runs SMALLINT, win_by_wickets SMALLINT,
                   win_by_innings SMALLINT, method VARCHAR, result VARCHAR, balls_per_over SMALLINT,
                   scheduled_overs SMALLINT, n_innings SMALLINT, cricsheet_missing VARCHAR)""")
    con.execute("INSERT INTO matches VALUES ('m', 'B', NULL, 9, NULL, NULL, NULL, 6, 20, 2, NULL)")
    con.execute("""CREATE TABLE innings (match_id VARCHAR, innings SMALLINT, team VARCHAR, is_super_over BOOLEAN,
                   target_runs INTEGER, target_balls INTEGER, penalty_runs_pre SMALLINT, penalty_runs_post SMALLINT,
                   miscounted_overs VARCHAR)""")
    con.execute("INSERT INTO innings VALUES ('m', 1, 'A', false, NULL, NULL, NULL, NULL, NULL), "
                f"('m', 2, 'B', false, 11, 120, NULL, {post}, NULL)")
    con.execute("""CREATE TABLE balls (match_id VARCHAR, innings SMALLINT, over SMALLINT, is_legal BOOLEAN,
                   runs_total SMALLINT, runs_batter SMALLINT, runs_extras SMALLINT, wides SMALLINT, noballs SMALLINT,
                   byes SMALLINT, legbyes SMALLINT, penalty SMALLINT, dismissal_count SMALLINT,
                   is_super_over BOOLEAN, bowler_id VARCHAR, batter_id VARCHAR)""")
    a = second_innings_runs
    con.execute("INSERT INTO balls VALUES ('m', 1, 0, true, 10, 10, 0, 0, 0, 0, 0, 0, 0, false, 'bwl', 'bat'), "
                f"('m', 2, 0, true, {a}, {a}, 0, 0, 0, 0, 0, 0, 0, false, 'bwl', 'bat')")
    return con.execute(f"SELECT * FROM ({gold.QUALITY_SQL})").fetchdf().iloc[0]


def test_result_decided_by_post_innings_penalty_runs_is_flagged_and_not_clean():
    """Match 1420222: Nepal won by 9 wickets but its balls ended 4 short; 5 untimed penalty runs won it."""
    row = _quality(second_innings_runs=7, post=5)          # 7 + 5 = 12 >= 11, balls alone 7 < 11
    assert row["result_in_ball_stream"] == False  # noqa: E712
    assert row["passes_quality"] == False  # noqa: E712
    assert row["result_consistent"] == True  # noqa: E712  (it uses the total, so it cannot see this)


def test_result_reached_on_the_balls_is_unflagged():
    row = _quality(second_innings_runs=12, post=0)
    assert row["result_in_ball_stream"] == True  # noqa: E712
    assert row["passes_quality"] == True  # noqa: E712
    # a penalty that does not change the winner is not a problem either
    row = _quality(second_innings_runs=12, post=5)
    assert row["result_in_ball_stream"] == True and row["passes_quality"] == True  # noqa: E712


# ---- The scope.yml schema ---------------------------------------------------------------------

def _scope():
    from cricket_sim.paths import load_yaml
    return load_yaml("scope.yml")


def test_repo_scope_config_passes_its_schema():
    gold.validate_scope(_scope())


def test_segment_without_match_type_is_refused_by_name():
    scope = _scope()
    del scope["segments"][1]["match_type"]            # the ODI segment
    with pytest.raises(SystemExit) as err:
        gold.validate_scope(scope)
    assert "ODI" in str(err.value) and "match_type" in str(err.value)


@pytest.mark.parametrize("break_it, message", [
    (lambda s: s["segments"][0].pop("name"), "segment #1: missing required key 'name'"),
    (lambda s: s["segments"][0].update(match_types="Test"), "segment 'Test': unknown key 'match_types'"),   # misspelt key
    (lambda s: s["segments"][2].update(both_teams_in="no_such_list"),
     "segment 'T20I': both_teams_in names 'no_such_list'"),
    (lambda s: s.pop("from_date"), "missing top-level key 'from_date'"),
    (lambda s: s.update(segments=[]), "'segments' must be a non-empty list"),
    (lambda s: s["segments"].append(dict(s["segments"][0])), "duplicate segment name 'Test'"),
])
def test_other_scope_mistakes_are_refused(break_it, message):
    # Each mistake is refused for ITS OWN reason (a bare SystemExit would also pass on an unrelated one)
    scope = _scope()
    break_it(scope)
    with pytest.raises(SystemExit, match="config/scope.yml is not valid") as err:
        gold.validate_scope(scope)
    assert message in str(err.value)
