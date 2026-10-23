"""passes_quality (gold.QUALITY_SQL) checked against matches whose right answer is known by hand.

Before this file no test looked at the verdict itself, so a flipped sign in the win-by-runs check, an awarded
match that could pass, or an all-out innings that failed all went through. Each case below is a tiny hand-made
match that is clean except for ONE thing, and the verdict (and the check that gave it) is stated in the test."""

import duckdb
import pandas as pd
import pytest

from cricket_sim import gold


def _quality(*matches: dict) -> dict:
    """Run the real QUALITY_SQL on hand-made matches; returns {match_id: row}.

    A match is a dict: id, sched (scheduled overs, None for a Test), bpo, winner, by_runs, by_wkts, by_inns,
    method, result, and `inns`: a list of dicts with team, runs, dismissals, super, target, tballs, pen_pre,
    pen_post, miscounted and `balls` (raw ball tuples, to override the default single ball)."""
    con = duckdb.connect()
    con.execute("""CREATE TABLE matches (match_id VARCHAR, winner VARCHAR, win_by_runs SMALLINT, win_by_wickets SMALLINT,
                   win_by_innings SMALLINT, method VARCHAR, result VARCHAR, balls_per_over SMALLINT,
                   scheduled_overs SMALLINT, n_innings SMALLINT, cricsheet_missing VARCHAR)""")
    con.execute("""CREATE TABLE innings (match_id VARCHAR, innings SMALLINT, team VARCHAR, is_super_over BOOLEAN,
                   target_runs INTEGER, target_balls INTEGER, penalty_runs_pre SMALLINT, penalty_runs_post SMALLINT,
                   miscounted_overs VARCHAR)""")
    con.execute("""CREATE TABLE balls (match_id VARCHAR, innings SMALLINT, over SMALLINT, is_legal BOOLEAN,
                   runs_total SMALLINT, runs_batter SMALLINT, runs_extras SMALLINT, wides SMALLINT, noballs SMALLINT,
                   byes SMALLINT, legbyes SMALLINT, penalty SMALLINT, dismissal_count SMALLINT,
                   is_super_over BOOLEAN, bowler_id VARCHAR, batter_id VARCHAR)""")
    for m in matches:
        mid, inns = m["id"], m["inns"]
        con.execute("INSERT INTO matches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [mid, m.get("winner"), m.get("by_runs"), m.get("by_wkts"), m.get("by_inns"), m.get("method"),
                     m.get("result"), m.get("bpo", 6), m.get("sched", 20), len(inns), m.get("missing")])
        for n, i in enumerate(inns, start=1):
            con.execute("INSERT INTO innings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [mid, n, i["team"], i.get("super", False), i.get("target"), i.get("tballs"),
                         i.get("pen_pre"), i.get("pen_post"), i.get("miscounted")])
            default = [(0, True, i["runs"], i["runs"], 0, 0, 0, 0, 0, 0, i.get("dismissals", 0))]
            # a ball tuple may carry a 12th and 13th value: the bowler id and the batter id (default one of each)
            for over, legal, total, bat, extras, wides, noballs, byes, legbyes, penalty, dismissals, *who in i.get("balls", default):
                bowler, batter = (who + ["bwl", "bat"])[:2] if len(who) < 2 else who[:2]
                con.execute("INSERT INTO balls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            [mid, n, over, legal, total, bat, extras, wides, noballs, byes, legbyes, penalty, dismissals,
                             i.get("super", False), bowler, batter])
    df = con.execute(f"SELECT * FROM ({gold.QUALITY_SQL})").fetchdf()
    return {r["match_id"]: r for _, r in df.iterrows()}


def _t20(mid="m", *, a=150, b=140, b_dismissals=0, **kw) -> dict:
    """A bat first and make `a`; B chase 151 in 120 balls and make `b`. Default: A won by 10 runs."""
    base = {"id": mid, "winner": "A", "by_runs": 10,
            "inns": [{"team": "A", "runs": a}, {"team": "B", "runs": b, "dismissals": b_dismissals,
                                                "target": a + 1, "tballs": 120}]}
    base.update(kw)
    return base


def _is(value) -> bool:
    """True only for a real True (not False, not NULL)."""
    return value is not None and not pd.isna(value) and bool(value)


def _false(value) -> bool:
    """True only for a real False: a NULL ('not checkable') is neither passed nor failed."""
    return value is not None and not pd.isna(value) and not bool(value)


def _null(value) -> bool:
    return value is None or bool(pd.isna(value))


def test_a_clean_limited_overs_match_passes_every_check():
    row = _quality(_t20())["m"]
    for check in ("runs_add_up", "extras_add_up", "overs_consistent", "wickets_within_limit", "bowler_quota_respected",
                  "batters_within_limit", "target_matches",
                  "result_consistent", "result_in_ball_stream"):
        assert _is(row[check]), check
    assert _is(row["passes_quality"])


# ---- the stated margin must follow from the innings totals (a flipped sign) ------------------------

def test_win_by_runs_margin_is_first_innings_total_minus_second():
    rows = _quality(_t20("ok", a=150, b=140, by_runs=10),
                    _t20("wrong_margin", a=150, b=140, by_runs=11),
                    _t20("flipped", a=140, b=150, by_runs=10, winner="A"),     # the chaser has more runs
                    _t20("wrong_winner", a=150, b=140, by_runs=10, winner="B"))
    assert _is(rows["ok"]["result_consistent"]) and _is(rows["ok"]["passes_quality"])
    for mid in ("wrong_margin", "flipped", "wrong_winner"):
        assert _false(rows[mid]["result_consistent"]) and not _is(rows[mid]["passes_quality"]), mid


def test_win_by_wickets_needs_the_chaser_ahead_and_the_chaser_to_win():
    def chase(mid, b, winner="B"):
        return {"id": mid, "winner": winner, "by_wkts": 7,
                "inns": [{"team": "A", "runs": 150}, {"team": "B", "runs": b, "dismissals": 3, "target": 151, "tballs": 120}]}
    rows = _quality(chase("ok", 151), chase("level", 150), chase("behind", 120), chase("wrong_side", 151, winner="A"))
    assert _is(rows["ok"]["passes_quality"])
    for mid in ("level", "behind", "wrong_side"):
        assert _false(rows[mid]["result_consistent"]) and not _is(rows[mid]["passes_quality"]), mid


def test_a_margin_after_rain_or_a_tie_or_no_result_is_not_checkable_and_not_a_failure():
    rows = _quality(_t20("rain", method="D/L", by_runs=124, a=319, b=164, winner="A"),
                    {"id": "tie", "result": "tie", "inns": [{"team": "A", "runs": 146}, {"team": "B", "runs": 146}]},
                    {"id": "nr", "result": "no result", "inns": [{"team": "A", "runs": 90}, {"team": "B", "runs": 3}]})
    for mid in ("rain", "tie", "nr"):
        assert _null(rows[mid]["result_consistent"]), mid
        assert _is(rows[mid]["passes_quality"]), mid


# ---- awarded matches never pass ---------------------------------------------------------------------

def test_an_awarded_match_does_not_pass_even_when_every_other_check_is_clean():
    row = _quality(_t20("m", method="Awarded"))["m"]
    assert _is(row["is_awarded"])
    assert not _is(row["passes_quality"])


# ---- wickets within the limit: all out passes, one more fails ----------------------------------------

def test_ten_wickets_is_all_out_and_passes_but_eleven_fails():
    rows = _quality(_t20("all_out", b_dismissals=10), _t20("eleven", b_dismissals=11))
    assert _is(rows["all_out"]["wickets_within_limit"]) and _is(rows["all_out"]["passes_quality"])
    assert _false(rows["eleven"]["wickets_within_limit"]) and not _is(rows["eleven"]["passes_quality"])


def test_a_super_over_may_lose_two_wickets_not_three():
    def tie(mid, so_wkts):
        return {"id": mid, "result": "tie", "inns": [{"team": "A", "runs": 146}, {"team": "B", "runs": 146},
                                                     {"team": "A", "runs": 17, "super": True},
                                                     {"team": "B", "runs": 8, "super": True, "dismissals": so_wkts}]}
    rows = _quality(tie("two", 2), tie("three", 3))
    assert _is(rows["two"]["passes_quality"]) and not _is(rows["three"]["passes_quality"])


# ---- the chase target is the first-innings total plus one ----------------------------------------------------

def test_an_unrevised_target_must_be_total_plus_one_and_a_revised_one_is_not_checked():
    ok = _t20("ok")
    bad = _t20("bad")
    bad["inns"][1]["target"] = 152                       # total + 2
    revised = _t20("revised")
    revised["inns"][1].update(target=120, tballs=90)     # a shortened chase: a different target is expected
    rows = _quality(ok, bad, revised)
    assert _is(rows["ok"]["target_matches"]) and _is(rows["ok"]["passes_quality"])
    assert _false(rows["bad"]["target_matches"]) and not _is(rows["bad"]["passes_quality"])
    assert _null(rows["revised"]["target_matches"])
    assert _is(rows["revised"]["passes_quality"])


# ---- ball arithmetic ----------------------------------------------------------------------------------------

def test_runs_and_extras_must_add_up_ball_by_ball():
    def with_ball(mid, ball):
        m = _t20(mid)
        m["inns"][0]["balls"] = [ball, (0, True, 149, 149, 0, 0, 0, 0, 0, 0, 0)]   # 1 + 149 = 150 either way
        return m
    #                 over legal total bat extras wides noballs byes legbyes penalty dismissals
    rows = _quality(with_ball("ok", (0, False, 1, 0, 1, 1, 0, 0, 0, 0, 0)),
                    with_ball("runs", (0, False, 2, 0, 1, 1, 0, 0, 0, 0, 0)),       # total 2 <> batter 0 + extras 1
                    with_ball("extras", (0, False, 1, 0, 1, 2, 0, 0, 0, 0, 0)))     # extras 1 <> wides 2
    assert _is(rows["ok"]["runs_add_up"]) and _is(rows["ok"]["extras_add_up"]) and _is(rows["ok"]["passes_quality"])
    assert _false(rows["runs"]["runs_add_up"]) and not _is(rows["runs"]["passes_quality"])
    assert _false(rows["extras"]["extras_add_up"]) and not _is(rows["extras"]["passes_quality"])


def test_an_over_must_hold_six_legal_balls_unless_it_is_listed_or_the_last():
    def over_of(n_legal, over, last_over, miscounted=None):
        m = _t20()
        legal = [(over, True, 0, 0, 0, 0, 0, 0, 0, 0, 0)] * n_legal
        later = [(last_over, True, 150, 150, 0, 0, 0, 0, 0, 0, 0)]
        m["inns"][0]["balls"] = legal + later
        m["inns"][0]["miscounted"] = miscounted
        return m
    full, short, short_listed, short_last = (over_of(6, 3, 4), over_of(5, 3, 4), over_of(5, 3, 4, '{"3": {"balls": 5}}'),
                                             over_of(5, 3, 3))
    for name, m in (("full", full), ("short", short), ("listed", short_listed), ("last", short_last)):
        m["id"] = name
    rows = _quality(full, short, short_listed, short_last)
    assert _is(rows["full"]["overs_consistent"]) and _is(rows["listed"]["overs_consistent"])
    assert _false(rows["short"]["overs_consistent"]) and not _is(rows["short"]["passes_quality"])
    # over 3 is both the over under test and (in "last") the final over of the innings: a short last over is fine
    assert _is(rows["last"]["overs_consistent"])


# ---- Tests: no scheduled overs ----------------------------------------------------------------------------

def _test_match(mid, **kw) -> dict:
    """A 300 and 100, B 250 and 120: A is 30 ahead in total (400 - 370)."""
    base = {"id": mid, "sched": None, "winner": "A", "by_runs": 30,
            "inns": [{"team": "A", "runs": 300}, {"team": "B", "runs": 250}, {"team": "A", "runs": 100},
                     {"team": "B", "runs": 120, "dismissals": 10}]}
    base.update(kw)
    return base


def test_test_match_margin_is_the_winners_total_minus_the_losers():
    rows = _quality(_test_match("ok"), _test_match("wrong", by_runs=31), _test_match("wrong_side", winner="B"))
    assert _is(rows["ok"]["result_consistent"]) and _is(rows["ok"]["passes_quality"])
    assert _false(rows["wrong"]["result_consistent"]) and not _is(rows["wrong"]["passes_quality"])
    assert _false(rows["wrong_side"]["result_consistent"]) and not _is(rows["wrong_side"]["passes_quality"])


def test_test_match_won_by_wickets_needs_the_winner_ahead_and_an_innings_win_is_not_checked():
    by_wkts = _test_match("ok", winner="B", by_runs=None, by_wkts=4)
    by_wkts["inns"][3]["runs"] = 200          # B 450, A 400: B ahead
    behind = _test_match("behind", winner="B", by_runs=None, by_wkts=4)   # B 370, A 400: B is behind
    innings_win = _test_match("inns", by_runs=None, by_inns=1)
    rows = _quality(by_wkts, behind, innings_win)
    assert _is(rows["ok"]["passes_quality"])
    assert _false(rows["behind"]["result_consistent"]) and not _is(rows["behind"]["passes_quality"])
    assert _is(rows["inns"]["passes_quality"])


def test_a_match_with_no_balls_has_nothing_to_fail():
    rows = _quality({"id": "washout", "result": "no result", "inns": []})
    assert not _is(rows["washout"]["has_balls"]) and _is(rows["washout"]["passes_quality"])


# ---------------------------------------------------------------- bowler quota, batter count, Cricsheet's own note

def _overs(spells: dict, runs_per_over: int = 1, extra_ball_in_over: int | None = None) -> list:
    """Overs 0, 1, 2, ... bowled in turn by the bowlers in `spells` ({bowler: overs}), six legal balls each (seven
    in `extra_ball_in_over`, like a miscounted over), one run per ball."""
    balls, over = [], 0
    for bowler, n in spells.items():
        for _ in range(n):
            for _ in range(7 if over == extra_ball_in_over else 6):
                balls.append((over, True, 1, 1, 0, 0, 0, 0, 0, 0, 0, bowler, "bat"))
            over += 1
    return balls


def _innings_of(balls: list, team: str = "A") -> dict:
    return {"team": team, "runs": sum(b[2] for b in balls), "balls": balls}


def _chase_of(first: dict, runs: int = 5, **kw) -> dict:
    return {"team": "B", "runs": runs, "target": first["runs"] + 1, "tballs": 120, **kw}


def test_a_bowler_may_start_a_fifth_of_the_scheduled_overs_and_not_one_more():
    # a 20-over match allows 4 overs a bowler; ten overs of six singles make 60, the chaser makes 5
    ok_first, bad_first = _innings_of(_overs({"x": 4, "y": 4, "z": 2})), _innings_of(_overs({"x": 5, "y": 4, "z": 1}))
    rows = _quality({"id": "ok", "winner": "A", "by_runs": 55, "inns": [ok_first, _chase_of(ok_first)]},
                    {"id": "bad", "winner": "A", "by_runs": 55, "inns": [bad_first, _chase_of(bad_first)]})
    assert _is(rows["ok"]["bowler_quota_respected"]) and _is(rows["ok"]["passes_quality"])
    assert _false(rows["bad"]["bowler_quota_respected"]) and not _is(rows["bad"]["passes_quality"])


def test_a_seven_ball_over_is_still_one_over_for_the_quota():
    first = _innings_of(_overs({"x": 4, "y": 4, "z": 2}, extra_ball_in_over=0))
    first["miscounted"] = '{"0": 7}'
    rows = _quality({"id": "seven", "winner": "A", "by_runs": 56, "inns": [first, _chase_of(first)]})
    assert _is(rows["seven"]["bowler_quota_respected"]) and _is(rows["seven"]["overs_consistent"])
    assert _is(rows["seven"]["passes_quality"])


def test_the_quota_is_not_checked_in_a_test_or_after_a_rain_rule():
    long = _innings_of(_overs({"x": 30}))                                   # 180 runs by one bowler
    test = {"id": "t", "sched": None, "winner": "A", "by_runs": 70,
            "inns": [long, {"team": "B", "runs": 80}, {"team": "A", "runs": 50}, {"team": "B", "runs": 80}]}
    rain_first = _innings_of(_overs({"x": 5, "y": 5}))
    rain = {"id": "rain", "winner": "A", "method": "D/L", "by_runs": 55, "inns": [rain_first, _chase_of(rain_first)]}
    rows = _quality(test, rain)
    assert _null(rows["t"]["bowler_quota_respected"]) and _null(rows["rain"]["bowler_quota_respected"])
    assert _is(rows["t"]["passes_quality"]) and _is(rows["rain"]["passes_quality"])


def test_twelve_batters_is_a_concussion_replacement_but_thirteen_is_not():
    def batting(n):  # one single each, six balls an over, a new batter every ball
        return [(k // 6, True, 1, 1, 0, 0, 0, 0, 0, 0, 0, "bwl", f"bat{k}") for k in range(n)]
    twelve, thirteen = _innings_of(batting(12)), _innings_of(batting(13))
    rows = _quality({"id": "twelve", "winner": "A", "by_runs": 7, "inns": [twelve, _chase_of(twelve)]},
                    {"id": "thirteen", "winner": "A", "by_runs": 8, "inns": [thirteen, _chase_of(thirteen)]})
    assert _is(rows["twelve"]["batters_within_limit"]) and _is(rows["twelve"]["passes_quality"])
    assert _false(rows["thirteen"]["batters_within_limit"]) and not _is(rows["thirteen"]["passes_quality"])


def test_cricsheets_own_missing_fields_note_is_recorded_and_never_fails_a_match():
    rows = _quality(_t20("noted", missing='["player_of_match"]'), _t20("clean"))
    assert _is(rows["noted"]["cricsheet_missing_fields_noted"]) and _is(rows["noted"]["passes_quality"])
    assert _false(rows["clean"]["cricsheet_missing_fields_noted"])
