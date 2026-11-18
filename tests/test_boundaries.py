"""The exact edges of the cut-offs that decide time-safe history, scope and which rows link.

Each check below sits on the boundary value and on both sides of it, because a '>=' that became '>' or a
'< 3' that became '<= 3' moves nothing on data that happens to have no row ON the edge, so nothing else
notices. (The Hawk-Eye link threshold has its edge test in test_hawkeye_link.py.)"""

import os
import shutil
import subprocess
import sys
from datetime import date

import duckdb
import pytest

from cricket_sim import gold
from cricket_sim.venues import checks
from cricket_sim.venues.canonical import era_for

from conftest import FIXTURE, ROOT

REF = {"composite_teams": ["ICC World XI"]}


# ---- venue eras: a ground rebuilt on a date belongs to the new era FROM that date ------------------------------

def test_venue_era_starts_on_its_from_date_not_the_day_after():
    ov = {"eras": {"v": [{"from": "2020-02-24"}]}}
    assert era_for("v", date(2020, 2, 23), ov) == "v"                    # the day before: still the old ground
    assert era_for("v", date(2020, 2, 24), ov) == "v@2020-02-24"         # ON the date: the new ground
    assert era_for("v", date(2020, 2, 25), ov) == "v@2020-02-24"
    assert era_for("v", "2020-02-24", ov) == "v@2020-02-24"              # the same with a text date


def test_two_eras_each_start_on_their_own_date():
    ov = {"eras": {"v": [{"from": "2020-02-24"}, {"from": "2023-01-01"}]}}
    assert [era_for("v", d, ov) for d in (date(2020, 2, 24), date(2022, 12, 31), date(2023, 1, 1))] == \
        ["v@2020-02-24", "v@2020-02-24", "v@2023-01-01"]


# ---- home team: three earlier internationals, and half of them -------------------------------------------------

def test_home_team_needs_three_internationals_not_two_and_not_four():
    assert checks.HOME_TEAM_MIN_MATCHES == 3
    assert checks.home_team({"A": 2}, 2, REF) is None                    # two matches: too few
    assert checks.home_team({"A": 3}, 3, REF) == "A"                     # exactly three: enough
    assert checks.home_team({"A": 4}, 4, REF) == "A"


def test_home_team_needs_at_least_half_of_the_matches_and_a_clear_lead():
    assert checks.HOME_TEAM_MIN_SHARE == 0.5
    assert checks.home_team({"A": 2, "B": 1, "C": 1}, 4, REF) == "A"     # exactly half of 4: at least half is enough
    assert checks.home_team({"A": 1, "B": 1, "C": 1, "D": 1}, 4, REF) is None   # a quarter
    assert checks.home_team({"A": 2, "B": 2}, 4, REF) is None            # half each: a tie, no home side
    assert checks.home_team({"A": 3, "B": 2}, 5, REF) == "A"             # 60%


# ---- scope: a match ON from_date is in scope (>=), the day before is not ---------------------------------------

# `cricket build` generates in a worker process, so patch the scope in a process that calls the generator itself
BUILD_WITH_SCOPE_FROM = """
from cricket_sim import build, contracts, gold
# the contracts' row-count floors assume the real from_date (2001-01-01), so skip them: only the scope is under test
contracts.enforce = lambda con: {"tables": 0, "checks_passed": 0}
real = gold.load_yaml
gold.load_yaml = lambda name: {**real(name), "from_date": FROM_DATE} if name == "scope.yml" else real(name)
build._build(allow_dirty=True)
"""


def _matches_in_scope_from(tmp_path, from_date):
    data = tmp_path / "data"
    shutil.copytree(FIXTURE / "bronze", data / "bronze")
    env = {**os.environ, "CRICKET_SIM_DATA": str(data), "CRICKET_SIM_SNAPSHOTS": str(FIXTURE)}
    for args in ([sys.executable, "-m", "cricket_sim.cli", "verify"],
                 [sys.executable, "-c", BUILD_WITH_SCOPE_FROM.replace('FROM_DATE', repr(from_date))]):
        done = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
        assert done.returncode == 0, done.stdout + done.stderr
    return {r[0] for r in duckdb.connect().execute(
        f"SELECT match_id FROM read_parquet('{data}/gold/dim_match.parquet')").fetchall()}


@pytest.mark.parametrize("from_date, kept_433606", [("2011-04-02", True), ("2011-04-03", False)],
                         ids=["match_on_from_date_is_kept", "match_the_day_before_is_dropped"])
def test_scope_from_date_is_inclusive(tmp_path, from_date, kept_433606):
    """The fixture's earliest in-scope match, the 2011 World Cup final, was played on 2011-04-02."""
    got = _matches_in_scope_from(tmp_path, from_date)
    assert ("433606" in got) is kept_433606
    assert {"1022353", "1144528", "1384439"} <= got                       # later matches are in either way


# ---- segments: a T20I needs BOTH teams in the full-member list ---------------------------------

def test_a_t20i_needs_two_full_member_teams_not_one():
    from cricket_sim.paths import load_yaml
    scope = load_yaml("scope.yml")
    con = duckdb.connect()
    con.execute("CREATE TABLE m (id VARCHAR, match_type VARCHAR, team_type VARCHAR, event_name VARCHAR, team1 VARCHAR, "
                "team2 VARCHAR, start_date DATE)")
    con.executemany("INSERT INTO m VALUES (?, ?, ?, ?, ?, ?, DATE '2019-06-01')", [
        ("both", "T20", "international", None, "India", "England"),
        ("one_a", "T20", "international", None, "India", "Nepal"),            # an associate opponent
        ("one_b", "T20", "international", None, "Nepal", "India"),
        ("none", "T20", "international", None, "Nepal", "Oman"),
        ("ipl", "T20", "club", "Indian Premier League", "Mumbai Indians", "Chennai Super Kings"),
        ("odi", "ODI", "international", None, "India", "Nepal"),               # ODIs have no such rule
        ("club", "T20", "club", "Some Other League", "India", "England")])
    got = dict(con.execute(f"SELECT id, {gold._segment_case(scope)} FROM m").fetchall())
    assert got == {"both": "T20I", "one_a": None, "one_b": None, "none": None, "ipl": "IPL", "odi": "ODI", "club": None}


# ---- league segments: every sponsor name, and each league's own start date --------------------

def test_league_segments_take_every_sponsor_name_from_their_own_start_date():
    from cricket_sim.paths import load_yaml
    scope = load_yaml("scope.yml")
    con = duckdb.connect()
    con.execute("CREATE TABLE m (id VARCHAR, match_type VARCHAR, team_type VARCHAR, event_name VARCHAR, team1 VARCHAR, "
                "team2 VARCHAR, start_date DATE)")
    con.executemany("INSERT INTO m VALUES (?, 'T20', 'club', ?, 'A', 'B', ?)", [
        ("natwest", "NatWest T20 Blast", "2015-05-15"),
        ("vitality", "Vitality Blast", "2019-07-01"),
        ("vitality_men", "Vitality Blast Men", "2025-06-01"),
        ("blast_2014", "NatWest T20 Blast", "2014-06-01"),        # before the leagues' 2015-04-01 start
        ("bbl_on_start", "Big Bash League", "2015-04-01"),        # the start date itself is kept
        ("bbl_day_before", "Big Bash League", "2015-03-31"),
        ("ipl_2008", "Indian Premier League", "2008-04-18"),      # the IPL keeps the global 2001 start
        ("hundred", "The Hundred Men's Competition", "2023-08-01"),
        ("smat", "Syed Mushtaq Ali Trophy", "2023-10-20")])
    got = dict(con.execute(f"SELECT id, {gold._segment_case(scope)} FROM m").fetchall())
    assert got == {"natwest": "T20 Blast", "vitality": "T20 Blast", "vitality_men": "T20 Blast", "blast_2014": None,
                   "bbl_on_start": "BBL", "bbl_day_before": None, "ipl_2008": "IPL", "hundred": None, "smat": None}


def test_missing_list_entries_follow_each_leagues_start_date():
    from cricket_sim.paths import load_yaml
    con = duckdb.connect()
    con.execute("CREATE TABLE x (list_name VARCHAR, start_date DATE)")
    con.executemany("INSERT INTO x VALUES (?, ?)", [("T20 blast", "2016-06-01"), ("T20 blast", "2014-06-01"),
                                                   ("Big bash league", "2016-01-10"), ("Super smash", "2018-01-01")])
    got = con.execute(f"SELECT list_name, start_date FROM x WHERE {gold._missing_list_sql(load_yaml('scope.yml'))}").fetchall()
    assert [(n, str(d)) for n, d in got] == [("T20 blast", "2016-06-01"), ("Big bash league", "2016-01-10")]
