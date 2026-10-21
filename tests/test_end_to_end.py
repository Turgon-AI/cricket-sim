"""The whole pipeline on the committed fixture: verify -> build -> export, then the facts it must get right.

Every pinned number below is the real scorecard (checked against the published result), not whatever the
pipeline happened to produce. If one fails, the pipeline is wrong, not the test.
"""

import json

import duckdb
import pytest

from cricket_sim import contracts

from conftest import build_fixture


def q(data, sql):
    return duckdb.connect().execute(sql.replace("{gold}", str(data / "gold"))).fetchall()


def innings(data, match_id):
    return {r[0]: r[1:] for r in q(data, f"""
        SELECT innings, batting_team, runs, dismissals, legal_balls FROM '{{gold}}/fact_innings.parquet'
        WHERE match_id = '{match_id}' ORDER BY innings""")}


def test_scope_keeps_men_and_drops_the_womens_final(tiny_build):
    rows = dict(q(tiny_build, "SELECT match_id, segment FROM '{gold}/dim_match.parquet'"))
    assert len(rows) == 9 and "1085975" not in rows
    assert rows["1187669"] == "T20I" and rows["1181768"] == "IPL" and rows["1389401"] == "Test"


def test_multi_day_matches_start_on_their_first_day(tiny_build):
    # The 2019 semi-final ran into its reserve day (9 and 10 July); the Gabba Test is 15 to 19 January 2021
    # and the Rajkot Test 15 to 18 February 2024 (four days listed). start_date is the first of them.
    rows = {r[0]: r[1:] for r in q(tiny_build, """SELECT match_id, CAST(start_date AS VARCHAR), CAST(end_date AS VARCHAR),
                                                          n_dates FROM '{gold}/dim_match.parquet'""")}
    assert rows["1144528"] == ("2019-07-09", "2019-07-10", 2)
    assert rows["1223872"] == ("2021-01-15", "2021-01-19", 5)
    assert rows["1389401"] == ("2024-02-15", "2024-02-18", 4)
    assert rows["433606"] == ("2011-04-02", "2011-04-02", 1)


def test_quality_verdicts_of_the_fixture_matches(tiny_build):
    # All nine are real, complete matches, so all pass. What each CAN be checked for follows from the
    # published result: a stated margin is checked against the scorecard only when there was no rain revision
    # (1022353, a D/L result) and a winner exists (the two ties 1144530 and 1187669 have none).
    rows = {r[0]: r[1:] for r in q(tiny_build, """SELECT match_id, target_matches, result_consistent,
        result_in_ball_stream, passes_quality FROM '{gold}/int_match_quality.parquet'""")}
    none = None
    womens_final = rows.pop("1085975")          # out of scope for dim_match, but int_match_quality scores every match
    assert womens_final[-1] is True
    assert rows == {"433606": (True, True, True, True),     # India by 6 wickets, chasing 275
                    "1144528": (True, True, True, True),    # New Zealand by 18 runs: 239 v 221
                    "1144530": (True, none, none, True),    # tied, no winner to check
                    "1022353": (none, none, none, True),    # D/L revised target: not checkable, not failed
                    "1181768": (True, True, True, True),    # Mumbai by 1 run: 149 v 148
                    "1187669": (none, none, none, True),    # tied after a reduced-overs chase
                    "1384439": (True, True, True, True),    # Australia by 6 wickets, 241 v 240
                    "1223872": (none, True, True, True),    # Gabba Test: India by 3 wickets
                    "1389401": (none, True, True, True)}    # Rajkot Test: India by 434 runs
    dim = dict(q(tiny_build, "SELECT match_id, passes_quality FROM '{gold}/dim_match.parquet'"))
    assert dim == {mid: True for mid in rows}


def test_2011_world_cup_final(tiny_build):
    assert innings(tiny_build, "433606") == {1: ("Sri Lanka", 274, 6, 300), 2: ("India", 277, 4, 290)}
    assert q(tiny_build, "SELECT winner, win_by_wickets FROM '{gold}/dim_match.parquet' WHERE match_id = '433606'") == [("India", 6)]


def test_2019_world_cup_semi_dhoni_run_out(tiny_build):
    assert innings(tiny_build, "1144528") == {1: ("New Zealand", 239, 8, 300), 2: ("India", 221, 10, 297)}
    row = q(tiny_build, """
        SELECT f.innings_runs_before, f.innings_wickets_before, f.nominal_runs_required_before,
               f.scheduled_balls_remaining_before, f.runs_total
        FROM '{gold}/fact_delivery.parquet' f JOIN '{gold}/fact_dismissal.parquet' d USING (match_id, innings, delivery_number)
        WHERE f.match_id = '1144528' AND d.player_out = 'MS Dhoni'""")
    assert row == [(215, 7, 25, 10, 1)]  # 215/7, needing 25 off 10; he completed a run before the throw hit


def test_2019_world_cup_final_tie_super_over_and_boundary_count(tiny_build):
    assert innings(tiny_build, "1144530") == {1: ("New Zealand", 241, 8, 300), 2: ("England", 241, 10, 300),
                                              3: ("England", 15, 0, 6), 4: ("New Zealand", 15, 1, 6)}
    assert q(tiny_build, """SELECT result, winner, winner_after_tiebreak FROM '{gold}/dim_match.parquet'
                            WHERE match_id = '1144530'""") == [("tie", None, "England")]
    # super overs: 6 balls, 2 wickets, target = previous super over + 1
    assert q(tiny_build, """SELECT innings_balls_limit, wickets_limit, target_runs, target_status FROM '{gold}/fact_innings.parquet'
                            WHERE match_id = '1144530' AND innings = 4""") == [(6, 2, 16, "super_over_derived")]


def test_2023_world_cup_final(tiny_build):
    assert innings(tiny_build, "1384439") == {1: ("India", 240, 10, 300), 2: ("Australia", 241, 4, 258)}


def test_rain_revised_chase_keeps_nominal_target_and_hides_the_revision(tiny_build):
    # 2017 Champions Trophy: India 319/3 in 48 overs; Pakistan's DLS target 289 in 41 overs; India won by 124 (D/L)
    assert innings(tiny_build, "1022353") == {1: ("India", 319, 3, 288), 2: ("Pakistan", 164, 9, 202)}
    assert q(tiny_build, """SELECT nominal_target_runs, target_runs, final_target_runs, final_target_balls, target_status
                            FROM '{gold}/fact_innings.parquet' WHERE match_id = '1022353' AND innings = 2""") == \
        [(320, None, 289, 246, "revised_timing_unknown")]


def test_test_match_penalty_runs_and_fourth_innings_target(tiny_build):
    # Rajkot 2024: England begin their innings on 5 penalty runs; India win by 434
    assert innings(tiny_build, "1389401")[2] == ("England", 319, 10, 427)
    assert q(tiny_build, """SELECT innings_runs_before FROM '{gold}/fact_delivery.parquet'
                            WHERE match_id = '1389401' AND innings = 2 AND delivery_number = 1""") == [(5,)]
    # Gabba 2021: 369 & 294 v 336, so India chase 328 and get there 329/7
    assert innings(tiny_build, "1223872")[4] == ("India", 329, 7, 582)
    assert q(tiny_build, """SELECT nominal_target_runs FROM '{gold}/fact_innings.parquet'
                            WHERE match_id = '1223872' AND innings = 4""") == [(328,)]


def test_t20i_tie_decided_by_super_over(tiny_build):
    # Eden Park 2019, 11 overs a side (reduced before the start): 146/5 v 146/7, super over 17/0 v 8/1
    assert innings(tiny_build, "1187669") == {1: ("New Zealand", 146, 5, 66), 2: ("England", 146, 7, 66),
                                              3: ("England", 17, 0, 6), 4: ("New Zealand", 8, 1, 6)}


def test_ipl_final_won_by_one_run(tiny_build):
    assert innings(tiny_build, "1181768") == {1: ("Mumbai Indians", 149, 8, 120), 2: ("Chennai Super Kings", 148, 7, 120)}


def test_every_contract_check_ran_and_passed(tiny_build):
    headline = json.loads((tiny_build / "gold" / "headline.json").read_text())
    built_layers = sum(c.layer in contracts.BUILD_LAYERS for c in contracts.load_all().values())
    assert headline["contract_checks"]["tables"] == built_layers and headline["contract_checks"]["checks_passed"] > 600
    assert headline["leak_check"]["state_continuity_breaks"] == 0


def test_export_is_gold_only_licensed_and_described(tiny_build):
    run = json.loads((tiny_build / "gold" / "run.json").read_text())
    out = tiny_build / "export" / run["build_fingerprint"][:12]
    manifest = json.loads((out / "manifest.json").read_text())
    tables = {m["file"].rsplit(".", 1)[0] for m in manifest}
    assert tables == {"fact_delivery", "fact_innings", "fact_dismissal", "fact_match_weather_hourly", "dim_match",
                      "dim_player", "dim_venue", "map_venue_alias", "map_match_ids", "report_match_quality",
                      "report_international_coverage", "model_ball_win_probability", "model_ball_outcome",
                      "model_team_elo", "report_model_scores", "report_model_calibration", "report_model_checkpoints",
                      "report_model_folds", "fact_player_match_form", "report_player_careers"}
    assert {"README.md", "LICENSES.md", "croissant.json", "DATASET_CARD.md", "NOTICE"} <= {p.name for p in out.iterdir()}
    croissant = json.loads((out / "croissant.json").read_text())
    assert croissant["conformsTo"] == "http://mlcommons.org/croissant/1.0"
    assert {r["name"] for r in croissant["recordSet"]} == tables
    assert all(f["description"] for r in croissant["recordSet"] for f in r["field"])
    # nothing restricted leaks into an exported file
    for m in manifest:
        if m["file"].endswith(".parquet"):
            cols = {c[0] for c in duckdb.connect().execute(f"DESCRIBE SELECT * FROM '{out / m['file']}'").fetchall()}
            assert not cols & {"line_cricinfo", "speed_kmh", "win_probability_after", "phase", "ari_batting_style",
                               "cricinfo_brier"}


@pytest.mark.slow
def test_two_builds_are_identical(tiny_build, tmp_path):
    again = build_fixture(tmp_path / "data")
    first = json.loads((tiny_build / "gold" / "run.json").read_text())
    assert again["build_fingerprint"] == first["build_fingerprint"]
    assert again["tables"] == first["tables"]
    models = [json.loads((d / "models" / "run.json").read_text())["model_fingerprint"] for d in (tiny_build, tmp_path / "data")]
    assert models[0] == models[1], "the walk-forward models must be deterministic"
