"""Wikipedia series pages: page list, template parsing, start time and day/night reading, de-duplication."""

from datetime import date, datetime, time

from cricket_sim.venues.weather import day_windows, innings_windows, scheduled_start_utc, sunset_utc
from cricket_sim.wiki_series import (dedupe, parse_day_night, parse_page, parse_time, season_series_titles,
                                     stated_zone, sub_page_titles, template_fields)

SEASON = """
==May==
===England in Ireland===
{{main|English cricket team in Ireland in 2019}}
{| class="wikitable"
| [http://www.espncricinfo.com/ci/engine/match/1161014.html ODI 4127] || 3 May || x || y
|}
===Women's series===
{{main|Some women's series}}
| [http://www.espncricinfo.com/ci/engine/match/1.html WODI 1100] || 4 May || {{crw|IRE}} || {{crw|ENG}}
==June==
{{main|2019 Cricket World Cup|2019 Cricket World Cup group stage}}
===Group stage===
| [http://www.espncricinfo.com/ci/engine/match/1144483.html ODI 4140] || 30 May
===Test series===
{{main|Some Test series}}
| [http://www.espncricinfo.com/ci/engine/match/2.html Test 2350] || 1 June
"""

PAGE = """
Intro.
{{Single-innings cricket match
| date = 30 May 2019
| time = 10:30 {{small|(D/N)}}
| team1 = {{cr-rt|ENG}}
| team2 = {{cr|RSA}}
| report = [https://www.espncricinfo.com/series/8039/scorecard/1144483/x Scorecard]
| venue = [[The Oval]], London
| daynight =
}}
{{Limited overs international
| date = 2 June
| time = 2:30 pm
| report = [http://www.espncricinfo.com/ci/engine/match/1144484.html Scorecard]
| daynight = no
}}
{{Limited overs matches
| date = 3 June
| report = [https://www.cricketarchive.com/x Scorecard]
}}
{{main|2019 Cricket World Cup Pool A|2019 Cricket World Cup squads}}
"""


def test_season_page_list_follows_main_links_of_odi_and_t20i_sections_only():
    titles = season_series_titles(SEASON)
    assert titles == ["English cricket team in Ireland in 2019", "2019 Cricket World Cup",
                      "2019 Cricket World Cup group stage"]  # the women's and Test sections add nothing


def test_sub_pages_must_extend_the_title_and_hold_matches():
    assert sub_page_titles("2019 Cricket World Cup", PAGE) == ["2019 Cricket World Cup Pool A"]


def test_template_parsing_keys_on_the_cricinfo_id():
    rows = parse_page("2019 Cricket World Cup", PAGE)
    assert [r["cricinfo_match_id"] for r in rows] == [1144483, 1144484]  # the CricketArchive row has no id
    first, second = rows
    assert first["scheduled_start_stated"] == time(10, 30) and first["day_night_flag"] is True
    assert second["scheduled_start_stated"] == time(14, 30) and second["day_night_flag"] is False


def test_fields_split_only_at_the_top_level():
    f = template_fields("| a = [[x|y]] | b = {{t|1|2}} | c = 3")
    assert f == {"a": "[[x|y]]", "b": "{{t|1|2}}", "c": "3"}


def test_time_parsing():
    assert parse_time("14:00 (ACDT)") == time(14, 0)
    assert parse_time("9.30 am") == time(9, 30)
    assert parse_time("12:15 pm") == time(12, 15)
    assert parse_time("12:05 am") == time(0, 5)
    assert parse_time("25:00") is None
    assert parse_time("TBC") is None


def test_blank_day_night_field_says_nothing():
    assert parse_day_night({"daynight": ""}, "10:30") is None
    assert parse_day_night({"night": "yes"}, "") is True
    assert parse_day_night({}, "14:00 D/N") is True


def test_dedupe_prefers_a_row_with_a_time_and_flags_conflicts():
    rows = [{"cricinfo_match_id": 1, "scheduled_start_stated": None, "day_night_flag": None, "_order": (0, 0)},
            {"cricinfo_match_id": 1, "scheduled_start_stated": time(14), "day_night_flag": True, "_order": (1, 0)},
            {"cricinfo_match_id": 1, "scheduled_start_stated": time(10), "day_night_flag": None, "_order": (2, 0)}]
    (row,) = dedupe(rows)
    assert row["scheduled_start_stated"] == time(14) and row["duplicates"] == 2 and row["duplicate_conflict"]


def test_scheduled_start_places_day_one_only_when_no_ball_times():
    d = [date(2019, 5, 30)]
    start = scheduled_start_utc(d[0], time(10, 30), None, "ODI", "Europe/London")
    (w,) = day_windows(d, "ODI", "Europe/London", [], start)
    assert w["window_source"] == "scheduled_start"
    assert w["window_start_utc"] == datetime(2019, 5, 30, 9, 30)  # BST
    assert w["window_end_utc"] == datetime(2019, 5, 30, 17, 30)
    (default,) = day_windows(d, "ODI", "Europe/London", [], None)
    assert default["window_source"] == "default_estimate"
    test_days = day_windows([date(2019, 8, 1), date(2019, 8, 2)], "Test", "Europe/London", [],
                            datetime(2019, 8, 1, 10))
    assert {x["window_source"] for x in test_days} == {"default_estimate"}  # Tests keep the default


def test_innings_windows_split_a_limited_overs_day():
    start = datetime(2019, 5, 30, 9, 30)
    first, second = innings_windows("ODI", start, datetime(2019, 5, 30, 17, 30))
    assert first == (start, datetime(2019, 5, 30, 13, 0))
    assert second == (datetime(2019, 5, 30, 13, 45), datetime(2019, 5, 30, 17, 30))
    assert innings_windows("Test", start, datetime(2019, 5, 30, 17, 30)) is None


def test_sunset_is_about_right():
    # London, 14 July 2019: sunset 21:14 BST = 20:14 UTC
    s = sunset_utc(51.5, -0.12, date(2019, 7, 14))
    assert abs((s - datetime(2019, 7, 14, 20, 14)).total_seconds()) < 180


def test_stated_zone_and_the_ipl_rule():
    assert stated_zone("14:00 GMT") == "UTC"
    assert stated_zone("19:30 IST") == "Asia/Kolkata"
    assert stated_zone("14:00 (ACDT)") is None and stated_zone("10:00 UTC+5:30") is None
    # IPL 2020 in Abu Dhabi: the page's 19:30 is Indian time, 14:00 UTC, 18:00 in the UAE
    assert scheduled_start_utc(date(2020, 9, 19), time(19, 30), None, "IPL", "Asia/Dubai") == datetime(2020, 9, 19, 14)
    assert scheduled_start_utc(date(2020, 9, 19), time(19, 30), None, "T20", "Asia/Dubai") == datetime(2020, 9, 19, 15, 30)
    assert scheduled_start_utc(date(2019, 6, 1), time(10, 30), "UTC", "ODI", "Europe/London") == datetime(2019, 6, 1, 10, 30)
