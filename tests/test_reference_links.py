"""Linking the Wikipedia reference list to Cricsheet. The linker applies only the strict
rule (teams agree, dates within a day); reviewed exceptions live in config/link_overrides.yml. Cases are real
rows that once linked wrongly, cut down to the fields the linker reads."""

from datetime import date

import duckdb
import pytest

from cricket_sim import gold, paths
from cricket_sim.wiki_fixtures import LinkOverrideError, link_reference, load_link_overrides


def _ref(segment, number, day, t1, t2, cricinfo_id=None):
    return {"segment": segment, "match_number": number, "start_date": day, "team1": t1, "team2": t2,
            "cricinfo_match_id": cricinfo_id}


def _cs(segment, match_id, number, day, t1, t2):
    return {"segment": segment, "match_id": match_id, "match_type_number": number, "start_date": day,
            "team1": t1, "team2": t2}


def _links(refs, cs, overrides=()):
    out = link_reference(refs, cs, overrides)
    assert len({(l["segment"], l["match_number"]) for l in out}) == len(out)  # each reference row at most once
    assert len({l["cricsheet_match_id"] for l in out}) == len(out)            # each Cricsheet match at most once
    return {(l["segment"], l["match_number"]): (l["cricsheet_match_id"], l["link_method"]) for l in out}


def test_t3_01_a_wrong_cricinfo_id_does_not_beat_teams_and_number():
    """Wikipedia's 2014-15 page links ODI 3617 to 656441 and 3618 to 656437: wrong scorecards."""
    refs = [_ref("ODI", 3617, date(2015, 2, 28), "New Zealand", "Australia", 656441),
            _ref("ODI", 3618, date(2015, 2, 28), "India", "United Arab Emirates", 656437),
            _ref("ODI", 3619, date(2015, 3, 1), "England", "Sri Lanka", 656449)]
    cs = [_cs("ODI", "656437", 3617, date(2015, 2, 28), "Australia", "New Zealand"),
          _cs("ODI", "656439", 3618, date(2015, 2, 28), "India", "United Arab Emirates"),
          _cs("ODI", "656441", 3619, date(2015, 3, 1), "Sri Lanka", "England")]
    assert _links(refs, cs) == {("ODI", 3617): ("656437", "number_and_teams"),
                                ("ODI", 3618): ("656439", "number_and_teams"),
                                ("ODI", 3619): ("656441", "number_and_teams")}


def test_t3_03_runner_up_is_tried_and_a_two_day_date_error_needs_a_reviewed_override():
    refs = [_ref("ODI", 1996, date(2003, 4, 5), "Kenya", "Zimbabwe", 65805),
            _ref("ODI", 1997, date(2003, 4, 6), "Sri Lanka", "Kenya", 65805),     # repeats 1996's id
            _ref("ODI", 2327, date(2006, 2, 9), "Pakistan", "India", None)]       # date cell 2 days early
    cs = [_cs("ODI", "65805", 1996, date(2003, 4, 5), "Kenya", "Zimbabwe"),
          _cs("ODI", "65806", 1997, date(2003, 4, 6), "Kenya", "Sri Lanka"),
          _cs("ODI", "236520", 2327, date(2006, 2, 11), "India", "Pakistan")]
    strict = {("ODI", 1996): ("65805", "cricinfo_id"), ("ODI", 1997): ("65806", "number_and_teams")}
    assert _links(refs, cs) == strict                      # strict rule: two days apart never links
    fix = [_override("ODI", 2327, None, "236520", 2327, "Wikipedia date cell wrong")]
    assert _links(refs, cs, fix) == {**strict, ("ODI", 2327): ("236520", "reviewed_override")}


def _override(segment, number, ref_id, match_id, cs_number, reason):
    return {"segment": segment, "reference_number": number, "reference_cricinfo_id": ref_id,
            "cricsheet_match_id": match_id, "cricsheet_number": cs_number, "reason": reason}


def test_strict_rule_needs_teams_and_date():
    """An id or a number alone is never enough: the two teams must agree and the dates be within a day."""
    day = date(2010, 1, 1)
    cs = [_cs("ODI", "1", 100, day, "Bangladesh", "New Zealand"), _cs("ODI", "2", 200, day, "India", "Sri Lanka")]
    refs = [_ref("ODI", 100, day, "Ireland", "New Zealand", 1),   # team typo on Wikipedia
            _ref("ODI", 200, day, None, None, 2)]                  # no teams on Wikipedia
    assert _links(refs, cs) == {}
    assert _links([_ref("ODI", 555, date(2010, 1, 3), "India", "Sri Lanka", 2)], cs) == {}  # 2 days apart
    assert _links([_ref("ODI", 555, date(2010, 1, 2), "Sri Lanka", "India")], cs) == {("ODI", 555): ("2", "date_and_teams")}
    fix = [_override("ODI", 100, 1, "1", 100, "Wikipedia team typo"), _override("ODI", 200, 2, "2", 200, "reference lists no teams")]
    assert _links(refs, cs, fix) == {("ODI", 100): ("1", "reviewed_override"), ("ODI", 200): ("2", "reviewed_override")}


def test_an_override_cannot_rewire_or_duplicate_a_strict_link():
    day = date(2015, 2, 28)
    refs = [_ref("ODI", 3617, day, "New Zealand", "Australia", 656441), _ref("ODI", 3618, day, None, None, 656437)]
    cs = [_cs("ODI", "656437", 3617, day, "Australia", "New Zealand"), _cs("ODI", "656439", 3618, day, "India", "UAE")]
    with pytest.raises(LinkOverrideError, match="contradicts the strict link"):   # ref already linked elsewhere
        _links(refs, cs, [_override("ODI", 3617, 656441, "656439", 3618, "x")])
    with pytest.raises(LinkOverrideError, match="already linked"):                # Cricsheet side already taken
        _links(refs, cs, [_override("ODI", 3618, 656437, "656437", 3617, "x")])
    with pytest.raises(LinkOverrideError, match="not needed"):                    # the strict rule already makes it
        _links(refs, cs, [_override("ODI", 3617, 656441, "656437", 3617, "x")])
    with pytest.raises(LinkOverrideError, match="do not match the data"):         # identifiers must agree with the data
        _links(refs, cs, [_override("ODI", 3618, 999, "656439", 3618, "x")])
    with pytest.raises(LinkOverrideError, match="reason"):
        _links(refs, cs, [_override("ODI", 3618, 656437, "656439", 3618, "")])


# ---------------------------------------------------------------- on the real data (skipped without a full build)

def _real_inputs():
    wf, dm = paths.SILVER / "wikipedia_fixtures.parquet", paths.GOLD / "dim_match.parquet"
    if not (wf.exists() and dm.exists()):
        pytest.skip("needs a full build in data/")
    con = duckdb.connect()
    con.execute(f"CREATE VIEW wikipedia_fixtures AS SELECT * FROM read_parquet('{wf}')")
    con.execute(f"CREATE VIEW dim_match AS SELECT * FROM read_parquet('{dm}')")
    if con.execute("SELECT count(*) FROM dim_match").fetchone()[0] < 5000:
        pytest.skip("data/ holds a small build, not the full one")
    return gold.reference_link_inputs(con, paths.load_yaml("scope.yml"))


def test_real_data_strict_rule_alone_and_with_the_reviewed_overrides(tmp_path):
    refs, cs = _real_inputs()
    empty = tmp_path / "link_overrides.yml"
    empty.write_text("links: []\n")
    reviewed = load_link_overrides()
    keys = {(o["segment"], o["reference_number"]): o["cricsheet_match_id"] for o in reviewed}
    assert len(reviewed) == 14 and all(o["reason"] for o in reviewed)

    strict = {(l["segment"], l["match_number"]): l for l in link_reference(refs, cs, load_link_overrides(empty))}
    per_segment = lambda links: {s: sum(1 for k in links if k[0] == s) for s in ("ODI", "Test", "T20I")}
    assert per_segment(strict) == {"ODI": 2554, "Test": 894, "T20I": 1135}
    # without the file, the 14 reviewed rows are unlinked (or fall back to another match, which would show here)
    assert {k: strict[k]["cricsheet_match_id"] for k in keys if k in strict} == {}

    full = {(l["segment"], l["match_number"]): l for l in link_reference(refs, cs, reviewed)}
    assert per_segment(full) == {"ODI": 2567, "Test": 895, "T20I": 1135}
    assert {k: (full[k]["cricsheet_match_id"], full[k]["link_method"]) for k in keys} == {
        k: (v, "reviewed_override") for k, v in keys.items()}
    assert {k: v for k, v in full.items() if k not in keys} == strict
