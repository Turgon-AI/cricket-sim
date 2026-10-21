from datetime import date

import pyarrow.parquet as pq

from cricket_sim import wiki_fixtures
from cricket_sim.wiki_fixtures import classify_result, cricinfo_id, dedupe, parse_wikitext, resolve_start_date

PAGE = """
==May==
===England in Ireland===
{| class="wikitable"
|-
! colspan="9"|ODI series
|-
! No.
! Date
! Home captain
! Away captain
! Venue
! Result
|-
| [http://www.espncricinfo.com/ci/engine/match/1161014.html ODI 4127] || 3 May || [[William Porterfield]] || [[Eoin Morgan]] || [[Malahide Cricket Ground|The Village]], [[Dublin]] || {{cr|ENG}} by 4 wickets
|-
| [http://www.espncricinfo.com/ci/engine/match/1161015.html ODI 4128] || 5 May || [[William Porterfield]] || [[Eoin Morgan]] || [[Malahide Cricket Ground|The Village]], [[Dublin]] || Match abandoned without a ball bowled
|}

===2019 Ireland Tri-Nation Series===
{| class="wikitable"
! No. !! Date !! Team 1 !! Captain 1 !! Team 2 !! Captain 2 !! Venue !! Result
|-
| [https://www.espncricinfo.com/series/ireland-tri-series-2019-1167315/ireland-vs-west-indies-1st-match-1167321/full-scorecard ODI 4130] || 5 May || {{cr|IRE}} || [[William Porterfield]] || {{cr|WIN}} || [[Jason Holder]] || [[Clontarf Cricket Club Ground|Clontarf]], [[Dublin]] || {{cr|WIN}} by 196 runs
|-
| [http://www.espncricinfo.com/ci/engine/match/1167322.html ODI 4131] || 7 May || {{cr|IRE}} || [[William Porterfield]] || {{cr|BAN}} || [[Mashrafe Mortaza]] || [[Malahide Cricket Ground|The Village]], [[Dublin]] || No result
|}

===Afghanistan in Scotland===
{| class="wikitable"
! No. !! Date !! Home captain !! Away captain !! Venue !! Result
|-
| [http://www.espncricinfo.com/ci/engine/match/1180063.html ODI 4140] || 8–9 May || [[Kyle Coetzer]] || [[Gulbadin Naib]] || [[The Grange Club]], [[Edinburgh]] || {{cr|AFG}} by 7 wickets ([[Duckworth–Lewis–Stern method|DLS]])
|}

===West Indies women in Ireland===
{| class="wikitable"
! No. !! Date !! Home captain !! Away captain !! Venue !! Result
|-
| [http://www.espncricinfo.com/ci/engine/match/1190000.html WODI 1150] || 10 May || [[Laura Delany]] || [[Stafanie Taylor]] || [[Malahide Cricket Ground|The Village]], [[Dublin]] || {{crw|WIN}} by 5 wickets
|-
| [http://www.espncricinfo.com/ci/engine/match/1190001.html T20I 800] || 12 May || [[Laura Delany]] || [[Stafanie Taylor]] || [[Malahide Cricket Ground|The Village]], [[Dublin]] || {{crw|IRE}} by 3 runs
|}
"""


def rows_by_number():
    return {r["match_number"]: r for r in parse_wikitext("International cricket in 2019", PAGE)}


def test_home_away_row_uses_heading_for_teams():
    r = rows_by_number()[4127]
    assert (r["match_format"], r["cricinfo_match_id"], r["start_date"]) == ("ODI", 1161014, date(2019, 5, 3))
    assert (r["team1"], r["team2"], r["teams_source"]) == ("Ireland", "England", "heading")
    assert r["result_text"] == "England by 4 wickets"
    assert r["result_class"] == "result" and not r["is_abandoned_no_ball"]
    assert r["section_heading"] == "England in Ireland" and r["page_title"] == "International cricket in 2019"


def test_tri_series_row_reads_team_cells_and_new_style_url():
    r = rows_by_number()[4130]
    assert (r["team1"], r["team2"], r["teams_source"]) == ("Ireland", "West Indies", "row")
    assert r["cricinfo_match_id"] == 1167321
    assert rows_by_number()[4131]["result_class"] == "no_result"


def test_abandoned_row():
    r = rows_by_number()[4128]
    assert r["result_class"] == "abandoned" and r["is_abandoned_no_ball"] is True


def test_afghanistan_row():
    r = rows_by_number()[4140]
    assert r["involves_afghanistan"] is True
    assert (r["team1"], r["team2"]) == ("Scotland", "Afghanistan")
    assert r["start_date"] == date(2019, 5, 8)
    assert not rows_by_number()[4127]["involves_afghanistan"]


def test_womens_rows_excluded():
    numbers = rows_by_number()
    assert 1150 not in numbers  # WODI label
    assert 800 not in numbers   # plain "T20I" label, but a women's section with {{crw}} templates
    assert set(numbers) == {4127, 4128, 4130, 4131, 4140}


def test_split_season_year_logic():
    assert resolve_start_date("12–16 September", "International cricket in 2000–01") == date(2000, 9, 12)
    assert resolve_start_date("31 December–4 January", "International cricket in 2000–01") == date(2000, 12, 31)
    assert resolve_start_date("3 January", "International cricket in 2000–01") == date(2001, 1, 3)
    assert resolve_start_date("A", "International cricket in 2000–01") is None


def test_helpers():
    assert cricinfo_id("[http://www.espncricinfo.com/series/15628/scorecard/63896/Zim-v-NZ/ Test 1510]") == 63896
    assert cricinfo_id("[http://www.espncricinfo.com/ci/engine/series/61093.html ODI 1756]") is None
    assert classify_result("Match drawn") == ("draw", False)
    assert classify_result("Match tied (England won the Super Over)") == ("tie", False)


def test_dedupe_keeps_most_complete_and_flags_conflicts():
    base = {"match_format": "ODI", "match_number": 1, "start_date": date(2001, 1, 1), "team1": "A", "team2": "B",
            "result_class": "result"}
    a = base | {"cricinfo_match_id": None, "_order": (0, 0)}
    b = base | {"cricinfo_match_id": 9, "_order": (1, 0)}
    (kept,) = dedupe([a, b])
    assert kept["cricinfo_match_id"] == 9 and kept["duplicates"] == 1 and not kept["duplicate_conflict"]
    c = base | {"cricinfo_match_id": 10, "_order": (2, 0)}
    (kept,) = dedupe([b, c])
    assert kept["duplicate_conflict"] and kept["conflict_fields"] == "cricinfo_match_id"


def test_build_is_deterministic(tmp_path, monkeypatch):
    import json
    from cricket_sim import bronze

    raw = tmp_path / "2019.wiki"
    raw.write_text(PAGE, encoding="utf-8")
    bronze.compress_to_zst(raw, tmp_path / "2019.wiki.zst")
    (tmp_path / "index.json").write_text(json.dumps(
        [{"title": "International cricket in 2019", "revid": 1, "stored_as": "2019.wiki.zst"}]))
    monkeypatch.setattr(wiki_fixtures, "SILVER", tmp_path)
    wiki_fixtures.build_wiki_fixtures(tmp_path)
    first = (tmp_path / "wikipedia_fixtures.parquet").read_bytes()
    wiki_fixtures.build_wiki_fixtures(tmp_path)
    assert (tmp_path / "wikipedia_fixtures.parquet").read_bytes() == first
    table = pq.read_table(tmp_path / "wikipedia_fixtures.parquet")
    assert table.column("match_number").to_pylist() == [4127, 4128, 4130, 4131, 4140]
