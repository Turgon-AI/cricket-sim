import json
import zipfile

import pyarrow.parquet as pq
import pytest

from cricket_sim.cricsheet_parse import check_shape, load_json_pins, parse_zip

from conftest import FIXTURE

MATCH = {
    "meta": {"data_version": "1.2.0", "created": "2026-01-01", "revision": 1},
    "info": {
        "dates": ["2026-01-02"], "match_type": "T20", "gender": "male", "team_type": "international",
        "teams": ["A", "B"], "toss": {"winner": "A", "decision": "bat"}, "outcome": {"winner": "A", "by": {"runs": 5}},
        "overs": 20, "balls_per_over": 6, "season": "2026", "venue": "Somewhere",
        "players": {"A": ["a1", "a2"], "B": ["b1"]},
        "registry": {"people": {"a1": "id_a1", "a2": "id_a2", "b1": "id_b1", "f1": "id_f1"}},
    },
    "innings": [{
        "team": "A",
        "overs": [{"over": 0, "deliveries": [
            {"batter": "a1", "bowler": "b1", "non_striker": "a2", "runs": {"batter": 0, "extras": 1, "total": 1},
             "extras": {"wides": 1}},
            {"batter": "a1", "bowler": "b1", "non_striker": "a2", "runs": {"batter": 4, "extras": 0, "total": 4}},
            {"batter": "a1", "bowler": "b1", "non_striker": "a2", "runs": {"batter": 0, "extras": 0, "total": 0},
             "wickets": [{"player_out": "a1", "kind": "caught", "fielders": [{"name": "f1"}]}]},
        ]}],
    }],
}


def test_parse_keys_and_legality(tmp_path):
    zpath = tmp_path / "all.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("123.json", json.dumps(MATCH))
        zf.writestr("README.txt", "ignored")
    parts = parse_zip(zpath, tmp_path / "out")

    balls = pq.read_table(parts["balls"][0]).to_pylist()
    assert [b["delivery_number"] for b in balls] == [1, 2, 3]
    assert [b["ball_in_over"] for b in balls] == [1, 2, 3]
    assert [b["is_legal"] for b in balls] == [False, True, True]
    assert balls[0]["batter_id"] == "id_a1" and balls[0]["wides"] == 1
    assert balls[2]["wicket_count"] == 1

    wicket = pq.read_table(parts["wickets"][0]).to_pylist()[0]
    assert (wicket["delivery_number"], wicket["fielder_ids"]) == (3, "id_f1")

    match = pq.read_table(parts["matches"][0]).to_pylist()[0]
    assert match["match_id"] == "123" and match["win_by_runs"] == 5


def test_overs_to_balls_uses_cricket_notation():
    from cricket_sim.cricsheet_parse import overs_to_balls
    assert overs_to_balls(37.4) == 226
    assert overs_to_balls("40.2") == 242
    assert overs_to_balls(19.5) == 119
    assert overs_to_balls(20) == 120
    assert overs_to_balls(None) is None


def test_tiebreak_review_replacement_substitute_retirement(tmp_path):
    import copy
    m = copy.deepcopy(MATCH)
    m["info"]["outcome"] = {"result": "tie", "eliminator": "A"}
    inn = m["innings"][0]
    inn["powerplays"] = [{"from": 0.1, "to": 5.6, "type": "mandatory"}]
    inn["absent_hurt"] = ["a2"]
    d = inn["overs"][0]["deliveries"]
    d[1]["review"] = {"by": "B", "umpire": "u1", "batter": "a1", "decision": "struck down"}
    d[1]["replacements"] = {"role": [{"in": "b1", "reason": "injury", "role": "bowler"}],
                            "match": [{"in": "f1", "out": "a2", "reason": "concussion_substitute", "team": "A"}]}
    d[2]["wickets"] = [{"player_out": "a1", "kind": "caught", "fielders": [{"name": "f1", "substitute": True}]},
                       {"player_out": "a2", "kind": "retired hurt"}]
    zpath = tmp_path / "all.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("9.json", json.dumps(m))
    parts = parse_zip(zpath, tmp_path / "out")
    read = lambda t: pq.read_table(parts[t][0]).to_pylist()

    match = read("matches")[0]
    assert match["winner"] is None and match["winner_after_tiebreak"] == "A" and match["tiebreak_method"] == "super_over"
    balls = read("balls")
    assert [b["actual_delivery"] for b in balls] == ["0.1", "0.1", "0.2"]  # the wide repeats ball 1's label
    assert balls[2]["wicket_count"] == 2 and balls[2]["dismissal_count"] == 1
    assert balls[1]["has_review"]
    wk = read("wickets")
    assert wk[0]["fielder_is_substitute"] and wk[1]["is_retirement"]
    reps = read("replacements")
    assert [(r["kind"], r["player_in"], r["player_out"]) for r in reps] == [("match", "f1", "a2"), ("role", "b1", None)]
    assert read("reviews")[0]["decision"] == "struck down"
    assert read("powerplays")[0]["to_ball"] == "5.6"
    assert read("innings")[0]["absent_hurt"] == "a2"


def test_multi_day_match_start_date_is_the_earliest_day_whatever_the_order(tmp_path):
    """start_date is the clock for every 'earlier matches only' rule, so it must be the FIRST day of a
    multi-day match (a Test, or a reserve-day finish), not the last, and not whatever Cricsheet listed first.
    The days are listed out of order and cross a new year to catch both a last-day and a list-order mistake."""
    import copy
    from datetime import date

    m = copy.deepcopy(MATCH)
    m["info"]["dates"] = ["2026-01-02", "2025-12-31", "2026-01-01", "2026-01-03"]
    zpath = tmp_path / "all.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("77.json", json.dumps(m))
    match = pq.read_table(parse_zip(zpath, tmp_path / "out")["matches"][0]).to_pylist()[0]
    assert match["start_date"] == date(2025, 12, 31)
    assert match["end_date"] == date(2026, 1, 3)
    assert match["n_dates"] == 4
    assert match["season_year"] == 2025          # the year of the first day, so a match is never moved across a year


def test_single_day_match_has_equal_start_and_end(tmp_path):
    from datetime import date

    zpath = tmp_path / "all.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("5.json", json.dumps(MATCH))
    match = pq.read_table(parse_zip(zpath, tmp_path / "out")["matches"][0]).to_pylist()[0]
    assert (match["start_date"], match["end_date"], match["n_dates"]) == (date(2026, 1, 2), date(2026, 1, 2), 1)


def test_every_kind_of_retirement_is_flagged_and_is_not_a_dismissal(tmp_path):
    """'Retired not out' never occurs in the fixture, but 45 real wickets are of this kind. A
    retirement ends a batter's innings without a dismissal; 'retired out' is a dismissal."""
    import copy
    m = copy.deepcopy(MATCH)
    deliveries = m["innings"][0]["overs"][0]["deliveries"]
    deliveries[2]["wickets"] = [{"player_out": "a1", "kind": "retired not out"}]
    deliveries[1]["wickets"] = [{"player_out": "a2", "kind": "retired hurt"}]
    deliveries[0]["wickets"] = [{"player_out": "a1", "kind": "retired out"}]
    zpath = tmp_path / "all.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("11.json", json.dumps(m))
    parts = parse_zip(zpath, tmp_path / "out")
    kinds = {w["kind"]: w["is_retirement"] for w in pq.read_table(parts["wickets"][0]).to_pylist()}
    assert kinds == {"retired not out": True, "retired hurt": True, "retired out": False}
    balls = pq.read_table(parts["balls"][0]).to_pylist()
    assert [(b["wicket_count"], b["dismissal_count"]) for b in balls] == [(1, 1), (1, 0), (1, 0)]


# ---------------------------------------------------------------- the key-set pin per data_version

def _zip_with(tmp_path, doc: dict):
    zpath = tmp_path / "all.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("42.json", json.dumps(doc))
    return zpath


def test_pins_cover_the_fixture_snapshot_and_every_pinned_version_has_the_four_objects():
    pins = load_json_pins()
    assert {"1.2.0", "1.3.0"} <= set(pins)
    for version, objects in pins.items():
        assert set(objects) == {"meta", "info", "innings", "delivery"}, version
        for required, allowed in objects.values():
            assert required <= allowed
        assert "data_version" in objects["meta"][0] and {"batter", "bowler", "runs"} <= objects["delivery"][0]
    with zipfile.ZipFile(FIXTURE / "bronze" / "cricsheet" / "2026-10-14" / "all_json.zip") as zf:
        docs = [json.loads(zf.read(n)) for n in zf.namelist() if n.endswith(".json")]
    assert len(docs) == 10
    for doc in docs:
        check_shape("fixture", doc, pins)


def test_unpinned_data_version_refuses_the_build(tmp_path):
    import copy
    m = copy.deepcopy(MATCH)
    m["meta"]["data_version"] = "9.9.9"
    with pytest.raises(SystemExit, match=r"42.json: Cricsheet data_version 9.9.9 has no key-set pin"):
        parse_zip(_zip_with(tmp_path, m), tmp_path / "out")
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())


@pytest.mark.parametrize("mutate, message", [
    (lambda m: m["info"].__setitem__("pitch_report", "dry"), r"info object .* unexpected \['pitch_report'\]"),
    (lambda m: m["innings"][0].__setitem__("powerplay", []), r"innings object .* unexpected \['powerplay'\]"),
    (lambda m: m["innings"][0]["overs"][0]["deliveries"][1].__setitem__("ball_speed_kph", 140),
     r"delivery object .* unexpected \['ball_speed_kph'\]"),
    (lambda m: m["meta"].__setitem__("schema", 2), r"meta object .* unexpected \['schema'\]"),
    (lambda m: m["info"].pop("balls_per_over"), r"info object .*missing \['balls_per_over'\]"),
    (lambda m: m["info"].pop("registry"), r"info object .*missing \['registry'\]"),
    (lambda m: m["innings"][0]["overs"][0]["deliveries"][2].pop("non_striker"), r"delivery object .*missing \['non_striker'\]"),
    # a rename shows up as one key missing and one unexpected on the same object
    (lambda m: m["info"].__setitem__("match_format", m["info"].pop("match_type")),
     r"info object .*missing \['match_type'\], unexpected \['match_format'\]"),
], ids=["info_extra", "innings_extra", "delivery_extra", "meta_extra", "info_missing", "registry_missing",
        "delivery_missing", "renamed"])
def test_a_mutated_file_refuses_the_build_with_the_object_and_keys_named(tmp_path, mutate, message):
    import copy
    m = copy.deepcopy(MATCH)
    mutate(m)
    with pytest.raises(SystemExit, match="42.json: a Cricsheet " + message) as err:
        parse_zip(_zip_with(tmp_path, m), tmp_path / "out")
    assert "config/input_schemas.yml" in str(err.value)


def test_the_check_runs_on_every_file_not_only_the_first(tmp_path):
    import copy
    bad = copy.deepcopy(MATCH)
    bad["innings"][0]["overs"][0]["deliveries"][0]["hawkeye"] = {}
    zpath = tmp_path / "all.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("1.json", json.dumps(MATCH))
        zf.writestr("2.json", json.dumps(bad))
    with pytest.raises(SystemExit, match=r"2.json: a Cricsheet delivery object"):
        parse_zip(zpath, tmp_path / "out")
