"""Gaurav Pandey's 2026-10-27 review, items 5-7: nationality as of the match (dim_player_team), the hindsight state
kept out of fact_delivery, cross-table reconciliation rules and a results check against Wikipedia.

Each reconciliation rule runs on the fixture build and must find nothing, then runs again with one value tampered
and must fire, so a rule that quietly stops comparing would be noticed. The spell and result-check SQL run on a
few literal rows that pin each case."""

import re

import duckdb
import pytest

from cricket_sim import contracts, gold

LAYERS = ("staging", "silver", "gold")


def _rule(table: str, name: str) -> str:
    rules = contracts.get(table).obj.get("quality") or []
    return next(r["query"] for r in rules if r.get("name") == name)


@pytest.fixture(scope="module")
def con(tiny_build):
    c = duckdb.connect()
    for layer in LAYERS:
        for path in sorted((tiny_build / layer).glob("*.parquet")):
            c.execute(f"CREATE VIEW {path.stem} AS SELECT * FROM read_parquet('{path}')")
    return c


def hits(con, table: str, name: str, tampered: dict | None = None) -> int:
    """Violations of rule `name` of `table`'s contract, reading `tampered[t]` (a SELECT) in place of table t."""
    tampered = tampered or {}

    def source(t: str) -> str:
        return f"({tampered[t]})" if t in tampered else t

    sql = re.sub(r"\$\{table:(\w+)\}", lambda m: source(m.group(1)), _rule(table, name))
    return con.execute(sql.replace("${table}", source(table))).fetchone()[0]


# one value changed on (or one row dropped from) a single row of the named table
FIRST = "row_number() OVER (ORDER BY match_id) = 1"
TAMPER = [
    ("fact_innings", "reconcile_with_fact_delivery",
     {"fact_delivery": f"SELECT * REPLACE (runs_total + CASE WHEN {FIRST} THEN 1 ELSE 0 END AS runs_total) FROM fact_delivery"}),
    ("fact_innings", "reconcile_rows_with_silver",
     {"innings": "SELECT * FROM innings WHERE NOT (innings = 1 AND match_id = (SELECT min(match_id) FROM dim_match))"}),
    ("fact_delivery", "reconcile_rows_with_silver",
     {"fact_delivery": f"SELECT * FROM fact_delivery QUALIFY NOT ({FIRST})"}),
    ("fact_delivery", "reconcile_wickets_with_fact_dismissal",
     {"fact_dismissal": f"SELECT * FROM fact_dismissal QUALIFY NOT ({FIRST})"}),
    ("fact_dismissal", "reconcile_rows_with_silver",
     {"wickets": "SELECT * FROM wickets UNION ALL (SELECT * FROM wickets WHERE match_id IN (SELECT match_id FROM dim_match) LIMIT 1)"}),
    ("dim_match", "reconcile_innings_with_fact_innings",
     {"dim_match": f"SELECT * REPLACE (n_innings + CASE WHEN {FIRST} THEN 1 ELSE 0 END AS n_innings) FROM dim_match"}),
    ("fact_player_match_form", "reconcile_rows_with_team_sheets",
     {"fact_player_match_form": f"SELECT * FROM fact_player_match_form QUALIFY NOT ({FIRST})"}),
    ("int_player_match_stats", "reconcile_with_balls",
     {"int_player_match_stats": f"SELECT * REPLACE (bat_runs + CASE WHEN {FIRST} THEN 1 ELSE 0 END AS bat_runs) "
                                "FROM int_player_match_stats"}),
    ("int_player_match_stats", "reconcile_with_wickets",
     {"int_player_match_stats": f"SELECT * REPLACE (bowl_wickets + CASE WHEN {FIRST} THEN 1 ELSE 0 END AS bowl_wickets) "
                                "FROM int_player_match_stats"}),
    ("fact_delivery_hindsight", "one_row_per_delivery",
     {"fact_delivery_hindsight": f"SELECT * FROM fact_delivery_hindsight QUALIFY NOT ({FIRST})"}),
]


@pytest.mark.parametrize("table, name, tampered", TAMPER, ids=[f"{t}.{n}" for t, n, _ in TAMPER])
def test_reconciliation_rule_holds_and_has_teeth(con, table, name, tampered):
    assert hits(con, table, name) == 0
    assert hits(con, table, name, tampered) > 0


def test_the_fixture_build_has_rows_to_reconcile(con):
    """A rule over empty tables would pass for nothing."""
    for t in ("fact_innings", "fact_delivery", "fact_dismissal", "dim_match", "fact_player_match_form",
              "int_player_match_stats", "fact_delivery_hindsight"):
        assert con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] > 0, t


# ---- dim_player_team ------------------------------------------------------------------------------------------

def spells(rows: list[tuple]) -> list[tuple]:
    """dim_player_team for team-sheet rows (player_id, team, match_id, start_date, team_type)."""
    c = duckdb.connect()
    c.execute("CREATE TABLE r (player_id VARCHAR, team VARCHAR, match_id VARCHAR, start_date DATE, team_type VARCHAR)")
    c.executemany("INSERT INTO r VALUES (?, ?, ?, ?, ?)", rows)
    c.execute("CREATE VIEW match_players AS SELECT match_id, team, player_id FROM r")
    c.execute("CREATE VIEW matches AS SELECT DISTINCT match_id, start_date, team_type FROM r")
    c.execute("CREATE VIEW dim_player AS SELECT DISTINCT player_id AS cricsheet_id FROM r")
    return c.execute(f"SELECT player_id, spell, team, CAST(valid_from AS VARCHAR), CAST(valid_to AS VARCHAR), matches, "
                     f"two_teams_same_day FROM ({gold.PLAYER_TEAM_SQL}) ORDER BY ALL").fetchall()


def test_a_player_who_changed_country_has_two_spells():
    # van der Merwe: South Africa, then the Netherlands; a World XI match and a domestic match change nothing
    assert spells([
        ("p", "South Africa", "1", "2009-01-10", "international"),
        ("p", "World XI", "2", "2009-06-01", "international"),
        ("p", "Somerset", "3", "2010-05-01", "club"),
        ("p", "South Africa", "4", "2010-02-01", "international"),
        ("p", "Netherlands", "5", "2015-07-01", "international"),
    ]) == [("p", 1, "South Africa", "2009-01-10", "2015-06-30", 2, False),
           ("p", 2, "Netherlands", "2015-07-01", None, 1, False)]


def test_a_player_who_went_back_gets_a_new_spell():
    # Rankin: Ireland, England, Ireland again
    assert [r[:3] for r in spells([
        ("p", "Ireland", "1", "2007-01-01", "international"),
        ("p", "England", "2", "2013-09-01", "international"),
        ("p", "Ireland", "3", "2016-06-01", "international"),
    ])] == [("p", 1, "Ireland"), ("p", 2, "England"), ("p", 3, "Ireland")]


def test_two_teams_on_one_day_is_flagged():
    rows = spells([("p", "Italy", "1", "2022-07-15", "international"),
                   ("p", "Singapore", "2", "2022-07-15", "international")])
    assert {r[6] for r in rows} == {True}


# ---- report_result_check --------------------------------------------------------------------------------------

def check(wiki: list[tuple], cs: list[tuple]) -> dict:
    """status and the three checks per match, for reference rows (number, result_text, result_class, winner) and
    Cricsheet rows (number, winner, result, win_by_runs, win_by_wickets, win_by_innings, method, result_consistent);
    the match_id is the number."""
    c = duckdb.connect()
    c.execute("""CREATE TABLE wikipedia_fixtures (match_format VARCHAR, match_number INTEGER, result_text VARCHAR,
                 result_class VARCHAR, winner VARCHAR)""")
    c.executemany("INSERT INTO wikipedia_fixtures VALUES ('ODI', ?, ?, ?, ?)", wiki)
    c.execute("""CREATE TABLE cs (n INTEGER, winner VARCHAR, result VARCHAR, win_by_runs SMALLINT, win_by_wickets SMALLINT,
                 win_by_innings SMALLINT, method VARCHAR, result_consistent BOOLEAN)""")
    c.executemany("INSERT INTO cs VALUES (?, ?, ?, ?, ?, ?, ?, ?)", cs)
    c.execute("""CREATE VIEW map_match_ids AS SELECT CAST(n AS VARCHAR) AS match_id, 'ODI' AS segment,
                 n AS reference_number, 'page' AS page_title, 1 AS page_revid FROM cs""")
    c.execute("CREATE VIEW dim_match AS SELECT CAST(n AS VARCHAR) AS match_id, * EXCLUDE (n, result_consistent) FROM cs")
    c.execute("CREATE VIEW int_match_quality AS SELECT CAST(n AS VARCHAR) AS match_id, result_consistent FROM cs")
    return {r[0]: r[1:] for r in c.execute(
        f"SELECT match_id, status, class_agrees, winner_agrees, margin_agrees FROM ({gold.RESULT_CHECK_SQL})").fetchall()}


def test_result_check_cases():
    got = check(
        [(1, "New Zealand by 7 wickets", "result", "New Zealand"),
         (2, "Australia by 227 runs", "result", "Australia"),
         (3, "India by 6 runs", "result", "India"),
         (4, "Match abandoned", "abandoned", None),
         (5, "by an innings and 5 runs", "result", None),
         (6, "", "unknown", None),
         (7, "Match tied", "tie", None),
         (8, "England by 3 wickets", "result", "England")],
        [(1, "New Zealand", None, None, 7, None, None, True),
         (2, "Australia", None, 277, None, None, None, True),
         (3, "Sri Lanka", None, 6, None, None, None, True),
         (4, None, "no result", None, None, None, None, None),
         (5, "England", None, 5, None, 1, None, None),
         (6, "India", None, 10, None, None, None, True),
         (7, None, "draw", None, None, None, None, None),
         (8, "England", None, 40, None, None, None, True)])
    assert got["1"] == ("agrees", True, True, True)
    assert got["2"] == ("margin_differs", True, True, False)       # a Wikipedia typo: Cricsheet's totals give 277
    assert got["3"] == ("winner_differs", True, False, True)
    assert got["4"] == ("agrees", True, None, None)                # abandoned after play started = no result
    assert got["5"] == ("agrees", True, None, True)                # innings win, the cell names no team
    assert got["6"] == ("not_checkable", None, None, None)
    assert got["7"] == ("class_differs", False, None, None)
    assert got["8"] == ("margin_differs", True, True, False)       # won by wickets on one side, runs on the other
