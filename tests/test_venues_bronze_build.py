"""Bronze hygiene, scope-free offline build and ingest planning for the venue context, on synthetic snapshots in a
temporary bronze root. No network: any HTTP attempt fails the test."""

import json
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
import pytest

from cricket_sim import bronze, paths
from cricket_sim.venues import build as vb, geocode as gc, ingest as ing, store
from cricket_sim.venues.checks import load_reference

CFG = {"licence": "test", "attribution": "test", "redistributable": True}
OVERRIDES = {"merge": {}, "assign": {}, "split_by_city": [], "city_aliases": {}, "venues": {}, "eras": {}}
EDEN = (22.5645, 88.3433)
WD = {"results": {"bindings": [
    {"item": {"value": "http://www.wikidata.org/entity/Q2035004"}, "itemLabel": {"value": "Eden Gardens"},
     "c": {"value": f"Point({EDEN[1]} {EDEN[0]})"}, "locLabel": {"value": "Kolkata"}, "countryLabel": {"value": "India"},
     "elev": {"value": "9"}}]}}


@pytest.fixture(autouse=True)
def tmp_bronze(tmp_path, monkeypatch):
    root = tmp_path / "bronze"
    root.mkdir()
    monkeypatch.setattr(paths, "BRONZE", root)
    monkeypatch.setattr(bronze, "BRONZE", root)
    monkeypatch.setattr(bronze, "LOCK_FILE", tmp_path / "lock.yml")
    monkeypatch.setattr(bronze, "PINS_FILE", tmp_path / "pins.yml")

    def no_network(*a, **k):
        raise AssertionError("network used in a test")
    monkeypatch.setattr(ing.Session, "get_json", no_network)
    return root


def _weather_resp(local_start: date, days: int, offset_s=19800, tz="Asia/Kolkata", precip=0.0):
    utc0 = datetime.combine(local_start, datetime.min.time()) - timedelta(seconds=offset_s)
    utc = [utc0 + timedelta(hours=h) for h in range(days * 24)]
    hourly = {"time": [(t + timedelta(seconds=offset_s)).strftime("%Y-%m-%dT%H:%M") for t in utc]}
    for v in ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "cloud_cover", "wind_speed_10m",
              "wind_direction_10m", "surface_pressure"]:
        hourly[v] = [20.0 + t.hour for t in utc]
    hourly["precipitation"] = [precip] * len(utc)
    return {"timezone": tz, "utc_offset_seconds": offset_s, "elevation": 30.0, "hourly": hourly}


def make_geo(sid="g1", *, failed_mystery=True, extra=None):
    w = store.Writer("venues_geo", sid).open()
    w.write_plan({"venues": [
        {"venue_id": "eden-gardens", "aliases": [["Eden Gardens", "Kolkata"], ["Eden Gardens, Kolkata", "Kolkata"]]},
        {"venue_id": "mystery-oval", "aliases": [["Mystery Oval", None]]}]})
    w.store(gc.WIKIDATA_FILE, {"url": "wd", "params": {}}, WD)
    # legacy snapshots hold Overpass answers from the retired geometry feature; the build must ignore them
    w.store("overpass/legacy.json.zst", {"url": "op", "params": {"data": "legacy"}}, {"elements": []})
    for name, (req, resp) in (extra or {}).items():
        w.store(name, req, resp)
    if failed_mystery:
        w.fail(gc.nominatim_file("Mystery Oval"), {"url": "n", "params": {"q": "Mystery Oval"}}, "HTTP 500")
    w.finalize({}, CFG, allow_gaps=failed_mystery)
    return store.Snapshot("venues_geo", sid)


def make_wx(sid="w1", lat=EDEN[0], lon=EDEN[1], precip=0.0):
    w = store.Writer("weather", sid).open()
    name = "eden-gardens__2020-01-01__2020-01-02.json.zst"
    w.write_plan({"geo_snapshot_id": "g1", "last_available": "2026-10-01",
                  "files": [{"file": name, "venue_id": "eden-gardens", "aliases": [["Eden Gardens", "Kolkata"]],
                             "lat": lat, "lon": lon, "start": "2020-01-01", "end": "2020-01-02"}], "deferred": []})
    w.store(name, {"url": "om", "params": {"latitude": lat, "longitude": lon, "start_date": "2020-01-01", "end_date": "2020-01-02"}},
            _weather_resp(date(2020, 1, 1), 2, precip=precip))
    w.finalize({}, CFG)
    return store.Snapshot("weather", sid)


def matches_con():
    con = duckdb.connect()
    con.execute("""CREATE TABLE matches AS SELECT * FROM (VALUES
        ('m1', 'Eden Gardens', 'Kolkata', DATE '2020-01-01', DATE '2020-01-01', 1, 'ODI', NULL, 'international', 'India', 'Australia'),
        ('m2', 'Eden Gardens, Kolkata', 'Kolkata', DATE '2020-01-02', DATE '2020-01-02', 1, 'T20', 'Club Cup', 'club', 'A', 'B'),
        ('m3', 'Mystery Oval', NULL, DATE '2020-01-01', DATE '2020-01-01', 1, 'ODI', NULL, 'international', 'X', 'Y'),
        ('m4', 'Unplanned Ground', 'Somewhere', DATE '2015-01-01', DATE '2015-01-01', 1, 'T20', 'Women League', 'club', 'C', 'D'),
        ('m5', 'Eden Gardens', 'Kolkata', DATE '2020-03-01', DATE '2020-03-01', 1, 'ODI', NULL, 'international', 'India', 'England'),
        ('m6', 'Eden Gardens', 'Kolkata', DATE '2019-11-22', DATE '2019-11-24', 3, 'Test', NULL, 'international', 'India', 'Bangladesh')
    ) t(match_id, venue, city, start_date, end_date, n_dates, match_type, event_name, team_type, team1, team2)""")
    return con


def run_build(tmp_path, geo, wx, overrides=OVERRIDES, out="out", con=None):
    con = con or matches_con()
    return con, vb._build(con, geo=geo, wx=wx, out_dir=tmp_path / out, overrides=overrides, reference=load_reference())


def q(con, sql):
    return con.execute(sql).fetchall()


# ---------------------------------------------------------------- bronze hygiene


def test_store_is_atomic(tmp_path, monkeypatch):
    def boom(src, dest):
        Path(dest).write_bytes(b"partial")
        raise OSError("disk full")
    monkeypatch.setattr(bronze, "compress_to_zst", boom)
    with pytest.raises(OSError, match="disk full"):   # The simulated failure, not another error
        store.store_zst_atomic(tmp_path / "s" / "x.json.zst", {"a": 1})
    assert list((tmp_path / "s").iterdir()) == []  # neither the target nor a temp file is left behind


def test_finalize_refuses_failures_unless_allow_gaps():
    w = store.Writer("weather", "s1").open()
    w.store("a.json.zst", {"url": "u", "params": {}}, {"x": 1})
    w.fail("b.json.zst", {"url": "u", "params": {}}, "HTTP 500")
    with pytest.raises(SystemExit, match="failed"):
        w.finalize({}, CFG)
    assert "s1" not in bronze.list_snapshots("weather")
    w.finalize({}, CFG, allow_gaps=True)
    snap = store.Snapshot("weather", "s1")
    assert set(snap.failures) == {"b.json.zst"} and snap.has("failures.json") and snap.manifest["upstream"]["allow_gaps"]


def test_snapshot_refuses_unlisted_files_and_reads_only_listed_ones():
    make_geo()
    folder = paths.snapshot_dir("venues_geo", "g1")
    (folder / "._wikidata_cricket_grounds.json.zst").write_bytes(b"junk")
    with pytest.raises(SystemExit, match="unlisted file"):
        store.Snapshot("venues_geo", "g1")


def test_snapshot_ids_never_collide_across_plans():
    a, b = store.new_snapshot_id({"x": 1}), store.new_snapshot_id({"x": 2})
    assert a != b and a[:10] == b[:10]  # same day, different plan -> new snapshot


# ---------------------------------------------------------------- offline build


def test_build_is_scope_free_complete_deterministic_and_explains_missing_weather(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "load_yaml", lambda name: (_ for _ in ()).throw(AssertionError(f"read {name}")))
    geo, wx = make_geo(), make_wx()
    con, stats = run_build(tmp_path, geo, wx)
    # every match (club, women's, unplanned venue) has a venue; scope.yml was never read
    assert q(con, "SELECT count(*) FROM match_venues") == [(6,)]
    assert q(con, "SELECT count(DISTINCT match_id) FROM int_match_weather_daily") == [(6,)]
    reasons = dict(q(con, "SELECT match_id, weather_missing_reason FROM int_match_weather_daily WHERE match_day_number = 1"))
    assert reasons["m1"] is None and reasons["m2"] is None            # club match on a fetched day: weather is there
    assert reasons["m3"] == "venue_not_geocoded"                     # planned venue whose search failed (recorded gap)
    assert reasons["m4"] == "venue_not_geocoded"
    assert reasons["m5"] == "not_in_weather_plan"
    eden = q(con, "SELECT country, needs_review, timezone, elevation_m, weather_location_quality, country_check, elevation_check "
                  "FROM venues WHERE venue_id = 'eden-gardens'")[0]
    assert eden == ("India", False, "Asia/Kolkata", 9.0, "verified_ground", "pass", "pass")
    assert q(con, "SELECT geocode_gap IS NULL, in_geo_plan FROM venues WHERE venue_id = 'mystery-oval'") == [(True, True)]
    row = q(con, "SELECT weather_location_quality, geocode_confidence, window_source, window_hours_present > 0 "
                 "FROM int_match_weather_daily WHERE match_id = 'm1'")[0]
    assert row == ("verified_ground", "high", "default_estimate", True)
    # a 3-day Test gets three match days; ODI default window 10:00-22:00 IST
    assert q(con, "SELECT count(*) FROM int_match_weather_daily WHERE match_id = 'm6'") == [(3,)]
    assert q(con, "SELECT min(window_start_local) FROM int_match_play_windows WHERE match_id = 'm1'") == [(datetime(2020, 1, 1, 10),)]
    first = {p.name: p.read_bytes() for p in (tmp_path / "out").glob("*.parquet")}
    assert {"venues.parquet", "venue_aliases.parquet", "match_venues.parquet", "int_match_weather_daily.parquet",
            "match_weather_hourly.parquet", "int_match_play_windows.parquet"} <= set(first)
    assert "venue_geometry_internal.parquet" not in first
    cols = {c for (c,) in q(con, "SELECT column_name FROM information_schema.columns WHERE table_name = 'venues'")}
    assert not {"has_geometry", "geometry_confidence"} & cols and "venue_id" in cols
    run_build(tmp_path, geo, wx)
    assert {p.name: p.read_bytes() for p in (tmp_path / "out").glob("*.parquet")} == first


def test_weather_aggregates_only_playing_hours(tmp_path):
    con, _ = run_build(tmp_path, make_geo(), make_wx(precip=0.5))
    hours, rain, pre = q(con, "SELECT window_hours_present, window_precipitation_mm, pre_match_12h_precipitation_mm "
                              "FROM int_match_weather_daily WHERE match_id = 'm1'")[0]
    assert hours == 13 and rain == pytest.approx(0.5 * 12)  # 12 hourly totals inside 10:00-22:00, nothing after
    assert q(con, "SELECT DISTINCT phase FROM match_weather_hourly WHERE match_id = 'm1' ORDER BY 1") == [("playing",), ("pre_match",)]


def test_redistributable_hourly_weather_does_not_depend_on_ball_timestamps(tmp_path):
    geo, wx = make_geo(), make_wx()
    con_default = matches_con()
    run_build(tmp_path, geo, wx, out="default", con=con_default)
    default_rows = (tmp_path / "default" / "match_weather_hourly.parquet").read_bytes()

    con_timed = matches_con()
    con_timed.execute("""CREATE TABLE stg_ari_tracking AS
        SELECT 'm1' AS match_id,
               TIMESTAMP '2020-01-01 04:00:00' + i * INTERVAL '1 minute' AS ball_timestamp
        FROM range(30) r(i)""")
    run_build(tmp_path, geo, wx, out="timed", con=con_timed)
    timed_rows = (tmp_path / "timed" / "match_weather_hourly.parquet").read_bytes()
    assert timed_rows == default_rows


def test_build_fails_when_a_weather_file_is_more_than_2km_from_the_venue(tmp_path):
    with pytest.raises(SystemExit, match="km from eden-gardens"):
        run_build(tmp_path, make_geo(), make_wx(lat=EDEN[0] + 0.05))


def test_build_fails_when_an_override_needs_a_request_the_snapshot_lacks(tmp_path):
    ov = {**OVERRIDES, "venues": {"eden-gardens": {"osm_ref": "way/123"}}}
    with pytest.raises(store.MissingRequest, match="osm_elements/way_123"):
        run_build(tmp_path, make_geo(), make_wx(), overrides=ov)


def test_build_fails_on_coordinates_that_fail_the_sanity_checks(tmp_path):
    # a manual point in the Gulf of Guinea for a ground declared to be in Kenya
    ov = {**OVERRIDES, "venues": {"unplanned-ground": {"lat": 0.0, "lon": 0.0, "country": "Kenya"}}}
    # seal each snapshot once: re-storing into a sealed snapshot in a later second changes fetched_at_utc
    geo, wx = make_geo(), make_wx()
    with pytest.raises(SystemExit, match="unplanned-ground: country"):
        run_build(tmp_path, geo, wx, overrides=ov)
    ov["venues"]["unplanned-ground"]["acknowledge"] = ["country"]
    con, _ = run_build(tmp_path, geo, wx, overrides=ov)
    assert q(con, "SELECT needs_review FROM venues WHERE venue_id = 'unplanned-ground'") == [(True,)]


def test_context_code_does_not_depend_on_gold_or_scope():
    src = Path(vb.__file__).parent
    for f in [*src.glob("*.py"), src.parent / "context_players.py", src.parent / "context_venues.py"]:
        text = f.read_text()
        assert "from .gold" not in text and "from ..gold" not in text and "scope.yml\")" not in text, f


# ---------------------------------------------------------------- ingest planning


def _zip(tmp_path, matches):
    z = tmp_path / "all_json.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for mid, info in matches.items():
            zf.writestr(f"{mid}.json", json.dumps({"meta": {}, "info": info, "innings": []}))
    return z


INFO = {"dates": ["2020-01-01"], "gender": "male", "match_type": "ODI", "team_type": "international",
        "teams": ["India", "Australia"], "venue": "Eden Gardens", "city": "Kolkata"}


def test_plan_comes_from_the_bronze_zip_not_silver(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "SILVER", tmp_path / "no-silver-here")
    z = _zip(tmp_path, {"1": INFO, "2": {**INFO, "team_type": "club", "venue": "Club Ground", "city": "X"}})
    matches = ing.bronze_matches(z)
    sel = {"gender": "male", "from_date": "2001-01-01", "any_of": [{"match_type": "ODI", "team_type": "international"}]}
    plan, venues, _ = ing.geo_plan(matches, sel, OVERRIDES, "cs1")
    assert [v["venue_id"] for v in plan["venues"]] == ["eden-gardens"]   # planned: selected matches only
    assert {v["venue_id"] for v in venues} == {"eden-gardens", "club-ground"}  # but every venue is canonical
    assert plan["config_sha256"]["venue_overrides.yml"]


def test_weather_ingest_refuses_without_a_sealed_locked_geo_snapshot(tmp_path, monkeypatch):
    z = _zip(tmp_path, {"1": INFO})
    (paths.snapshot_dir("cricsheet", "c1")).mkdir(parents=True)
    (paths.snapshot_dir("cricsheet", "c1") / "all_json.zip").write_bytes(z.read_bytes())
    bronze.write_lock({"cricsheet": "c1"})
    monkeypatch.setattr(ing, "load_overrides", lambda: OVERRIDES)
    with pytest.raises(SystemExit, match="sealed, locked venues_geo"):
        ing.ingest_venues_and_weather(stage="weather")


def test_weather_plan_reuses_nearby_files_and_refetches_moved_venues():
    make_wx()
    bronze.lock_source("weather", "w1")
    matches = [{"venue": "Eden Gardens", "city": "Kolkata", "dates": [date(2020, 1, 1), date(2020, 1, 2)], "gender": "male",
                "match_type": "ODI", "team_type": "international", "event_name": None, "teams": []}]
    aliases = [{"venue": "Eden Gardens", "city": "Kolkata", "venue_id": "eden-gardens"}]
    sel = {"any_of": [{"match_type": "ODI"}]}
    cfg = {"archive_lag_days": 6, "max_window_days": 14, "hourly": []}
    near = (EDEN[0] + 0.005, EDEN[1])  # ~550 m: same point for weather purposes
    plan, reused = ing.weather_plan(cfg, matches, sel, aliases, {"eden-gardens": near}, "g1", "sha", "c1",
                                    ing._Reuse("weather"), date(2026, 10, 1))
    assert len(reused) == 1 and all("reused_from" in f for f in plan["files"])
    moved = (EDEN[0] + 0.05, EDEN[1])  # ~5.5 km: coordinates changed, so the weather must be fetched again
    plan, reused = ing.weather_plan(cfg, matches, sel, aliases, {"eden-gardens": moved}, "g1", "sha", "c1",
                                    ing._Reuse("weather"), date(2026, 10, 1))
    assert reused == [] and [(f["start"], f["end"]) for f in plan["files"]] == [("2019-12-31", "2020-01-03")]


def test_geo_dry_run_reuses_the_locked_snapshot_without_network():
    make_geo(failed_mystery=False)
    bronze.lock_source("venues_geo", "g1")
    venues = [{"venue_id": "eden-gardens", "name": "Eden Gardens", "city": "Kolkata", "search_names": ["Eden Gardens"],
               "places": ["Kolkata"], "n_matches": 1}]
    plan = {"venues": [{"venue_id": "eden-gardens", "aliases": [["Eden Gardens", "Kolkata"]]}]}
    cfg = {"user_agent": "test", "endpoints": {"wikidata_sparql": "x", "nominatim": "x", "nominatim_lookup": "x"}, "nominatim_min_interval_seconds": 0, **CFG}
    assert ing.ingest_geo(cfg, plan, venues, OVERRIDES, allow_gaps=False, dry_run=True) is None  # no HTTP happened
