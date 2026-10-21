"""Venue context as of each match date (venue_context: asof). Synthetic snapshots in a temporary bronze root,
plus a cutoff rebuild on the tiny fixture's real snapshots. No network."""

from datetime import date

import duckdb
import pytest
import yaml

from cricket_sim import bronze, paths
from cricket_sim.venues import asof, build as vb, geocode as gc, store
from cricket_sim.venues.canonical import canonicalize, load_overrides
from cricket_sim.venues.checks import load_reference

CFG = {"licence": "test", "attribution": "test", "redistributable": True}
BASE = {"merge": {"Eden Gardens": ["Calcutta Oval"]}, "assign": {}, "split_by_city": [], "city_aliases": {},
        "venues": {}, "eras": {}}
SNAP, ASOF = {**BASE, "venue_context": "snapshot"}, {**BASE, "venue_context": "asof"}
EDEN = (22.5645, 88.3433)
WD = {"results": {"bindings": [
    {"item": {"value": "http://www.wikidata.org/entity/Q2035004"}, "itemLabel": {"value": "Eden Gardens"},
     "c": {"value": f"Point({EDEN[1]} {EDEN[0]})"}, "locLabel": {"value": "Kolkata"}, "countryLabel": {"value": "India"},
     "elev": {"value": "9"}}]}}
# 2010: the ground under an old spelling no geocoder knows; 2020: its current name, which Wikidata knows
MATCHES = [("m1", "Calcutta Oval", "Kolkata", date(2010, 1, 5), "India", "Australia"),
           ("m2", "Calcutta Oval, Kolkata", "Kolkata", date(2010, 1, 5), "India", "England"),
           ("m3", "Eden Gardens", "Kolkata", date(2020, 2, 1), "India", "Sri Lanka")]


def _dicts(rows):
    return [{"match_id": r[0], "venue": r[1], "city": r[2], "start_date": r[3], "team_type": "international",
             "team1": r[4], "team2": r[5]} for r in rows]


def _versions(rows, overrides):
    ms = _dicts(rows)
    pairs = sorted({(m["venue"], m["city"]) for m in ms})
    venues, aliases = canonicalize([(v, c, 1) for v, c in pairs], overrides)
    alias_id = {(a["venue"], a["city"]): a["venue_id"] for a in aliases}
    return asof.venue_versions(ms, venues, alias_id, overrides, load_reference())


def test_t1_04_later_spelling_does_not_change_the_version_of_earlier_matches():
    full, early = _versions(MATCHES, ASOF)["eden-gardens"], _versions(MATCHES[:2], ASOF)["eden-gardens"]
    assert [(v["venue_version"], v["valid_from"], v["valid_to"]) for v in full] == [
        (1, date(2010, 1, 5), date(2020, 1, 31)), (2, date(2020, 2, 1), None)]
    assert full[0]["venue"]["search_names"] == ["Calcutta Oval"]  # same-day spellings share one version
    assert full[1]["venue"]["search_names"] == ["Calcutta Oval", "Eden Gardens"]
    assert asof.version_for(full, date(2010, 1, 5)) == {**early[0], "valid_to": date(2020, 1, 31)}
    # snapshot mode (the default): one version from every match, so the 2010 matches see the 2020 spelling
    snap = _versions(MATCHES, SNAP)["eden-gardens"]
    assert len(snap) == 1 and snap[0]["venue"]["search_names"] == ["Calcutta Oval", "Eden Gardens"]


# ---------------------------------------------------------------- full venue build on synthetic snapshots


@pytest.fixture
def tmp_bronze(tmp_path, monkeypatch):
    root = tmp_path / "bronze"
    root.mkdir()
    monkeypatch.setattr(paths, "BRONZE", root)
    monkeypatch.setattr(bronze, "BRONZE", root)
    monkeypatch.setattr(bronze, "LOCK_FILE", tmp_path / "lock.yml")
    monkeypatch.setattr(bronze, "PINS_FILE", tmp_path / "pins.yml")
    return root


def _geo(planned=(), extra=None, sid="g1", wd=None):
    w = store.Writer("venues_geo", sid).open()
    w.write_plan({"venues": [{"venue_id": vid, "aliases": aliases} for vid, aliases in planned]})
    w.store(gc.WIKIDATA_FILE, {"url": "wd", "params": {}}, wd or WD)
    for name, (req, resp) in (extra or {}).items():
        w.store(name, req, resp)
    w.finalize({}, CFG)
    return store.Snapshot("venues_geo", sid)


def _wx(files=(), sid="w1"):
    w = store.Writer("weather", sid).open()
    w.write_plan({"geo_snapshot_id": "g1", "last_available": "2026-10-01", "files": list(files), "deferred": []})
    w.finalize({}, CFG, allow_gaps=True)
    return store.Snapshot("weather", sid)


def _build(tmp_path, rows, overrides, out, geo, wx):
    con = duckdb.connect()
    con.execute("CREATE TABLE matches (match_id VARCHAR, venue VARCHAR, city VARCHAR, start_date DATE, end_date DATE, "
                "n_dates INTEGER, match_type VARCHAR, event_name VARCHAR, team_type VARCHAR, team1 VARCHAR, team2 VARCHAR)")
    for mid, venue, city, day, t1, t2 in rows:
        con.execute("INSERT INTO matches VALUES (?, ?, ?, ?, ?, 1, 'ODI', NULL, 'international', ?, ?)",
                    [mid, venue, city, day, day, t1, t2])
    vb._build(con, geo=geo, wx=wx, out_dir=tmp_path / out, overrides=overrides, reference=load_reference())
    return con


def _early_rows(con, ids=("m1", "m2")):
    """Everything the venue context says about the early matches: version, name, point, confidence, weather row."""
    q = f"""SELECT mv.match_id, mv.venue_id, mv.venue_version, v.name, v.city, v.lat, v.lon, v.geocode_confidence,
                   v.search_names, d.geocode_confidence AS day_confidence, d.weather_missing_reason
            FROM match_venues mv JOIN venue_versions v USING (venue_id, venue_version)
            JOIN int_match_weather_daily d USING (match_id)
            WHERE mv.match_id IN {ids} ORDER BY 1"""
    return con.execute(q).fetchall()


def test_t1_04_cutoff_rebuild_keeps_every_early_row_with_asof(tmp_bronze, tmp_path):
    geo, wx = _geo(), _wx()
    full = _early_rows(_build(tmp_path, MATCHES, ASOF, "full", geo, wx))
    cut = _early_rows(_build(tmp_path, MATCHES[:2], ASOF, "cut", geo, wx))
    assert full == cut and full[0][5] is None  # 2010: the old spelling has no point yet; adding 2020 changes nothing
    # the snapshot-wide default (the old behaviour): the 2020 spelling moves the 2010 matches onto Eden Gardens
    snap_full = _early_rows(_build(tmp_path, MATCHES, SNAP, "sfull", geo, wx))
    snap_cut = _early_rows(_build(tmp_path, MATCHES[:2], SNAP, "scut", geo, wx))
    assert snap_full != snap_cut and snap_full[0][5] == pytest.approx(EDEN[0])


def test_t1_04_asof_fails_loudly_on_a_missing_geocode_request(tmp_bronze, tmp_path):
    # planned venue: version 1 (2010, old spelling only) needs Nominatim searches the snapshot never made
    geo = _geo(planned=[("eden-gardens", [["Calcutta Oval", "Kolkata"], ["Eden Gardens", "Kolkata"]])])
    with pytest.raises(store.MissingRequest, match="lacks nominatim/"):
        _build(tmp_path, MATCHES, ASOF, "out", geo, _wx())
    _build(tmp_path, MATCHES, SNAP, "snap", geo, _wx())  # snapshot mode: Wikidata answers, no request needed


# Calcutta Oval: a second Wikidata item for the same ground, 3 km from the Eden Gardens point (an older point)
OLD = (EDEN[0] + 0.027, EDEN[1])
WD2 = {"results": {"bindings": WD["results"]["bindings"] + [
    {"item": {"value": "http://www.wikidata.org/entity/Q9999999"}, "itemLabel": {"value": "Calcutta Oval"},
     "c": {"value": f"Point({OLD[1]} {OLD[0]})"}, "locLabel": {"value": "Kolkata"}, "countryLabel": {"value": "India"}}]}}


def _resp(start: date, days: int):
    from datetime import datetime, timedelta
    utc0 = datetime.combine(start, datetime.min.time()) - timedelta(seconds=19800)
    utc = [utc0 + timedelta(hours=h) for h in range(days * 24)]
    hourly = {"time": [(t + timedelta(seconds=19800)).strftime("%Y-%m-%dT%H:%M") for t in utc]}
    for v in ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "cloud_cover", "wind_speed_10m",
              "wind_direction_10m", "surface_pressure"]:
        hourly[v] = [25.0] * len(utc)
    hourly["precipitation"] = [0.0] * len(utc)
    return {"timezone": "Asia/Kolkata", "utc_offset_seconds": 19800, "elevation": 9.0, "hourly": hourly}


def _wx_files(files, sid):
    """files: (name, (lat, lon), first day, last day, stored?)"""
    w = store.Writer("weather", sid).open()
    plan = []
    for name, (lat, lon), start, end, stored in files:
        plan.append({"file": name, "venue_id": "eden-gardens", "aliases": [["Eden Gardens", "Kolkata"], ["Calcutta Oval", "Kolkata"]],
                     "lat": lat, "lon": lon, "start": start.isoformat(), "end": end.isoformat()})
        req = {"url": "om", "params": {"latitude": lat, "longitude": lon, "start_date": start.isoformat(), "end_date": end.isoformat()}}
        if stored:
            w.store(name, req, _resp(start, (end - start).days + 1))
        else:
            w.fail(name, req, "HTTP 500")
    w.write_plan({"geo_snapshot_id": "g1", "last_available": "2026-10-01", "files": plan, "deferred": []})
    w.finalize({}, CFG, allow_gaps=True)
    return store.Snapshot("weather", sid)


def _weather(con):
    return dict(con.execute("SELECT match_id, coalesce(weather_missing_reason, 'ok') || '/' || CASE WHEN weather_available THEN 'yes' ELSE 'no' END "
                            "FROM int_match_weather_daily WHERE match_day_number = 1").fetchall())


# The ingest plans each version's match days with one padding day on each side (12 h before a morning start, the
# hour after a late finish). Version 2's first match is 2020-02-01, so its file starts 2020-01-31: the last day
# version 1 is valid. That padding day is not a match day of version 1 and must not tie the file to version 1
# (the 2026-10-21 rebuild failed exactly so at zohur-ahmed-chowdhury-stadium).
V2_FILE = ("eden-gardens--v2__2020-01-31__2020-02-02.json.zst", EDEN, date(2020, 1, 31), date(2020, 2, 2), True)


def test_t1_04_padding_day_does_not_tie_a_file_to_a_previous_version_without_a_point(tmp_bronze, tmp_path):
    con = _build(tmp_path, MATCHES, ASOF, "out", _geo(), _wx_files([V2_FILE], "w3"))
    w = _weather(con)
    assert w["m3"] == "ok/yes"                                  # version 2 gets its weather
    assert w["m1"] == w["m2"] == "venue_not_geocoded/no"      # version 1 has no point: its days lose weather


def test_t1_04_padding_day_does_not_tie_a_file_to_a_previous_version_at_another_point(tmp_bronze, tmp_path):
    v1_file = ("eden-gardens__2010-01-04__2010-01-06.json.zst", OLD, date(2010, 1, 4), date(2010, 1, 6), True)
    con = _build(tmp_path, MATCHES, ASOF, "out", _geo(wd=WD2), _wx_files([v1_file, V2_FILE], "w4"))
    pts = con.execute("SELECT venue_version, lat FROM venue_versions ORDER BY 1").fetchall()
    assert pts == [(1, pytest.approx(OLD[0])), (2, pytest.approx(EDEN[0]))]   # the two versions are 3 km apart
    assert _weather(con) == {"m1": "ok/yes", "m2": "ok/yes", "m3": "ok/yes"}


def test_t1_04_asof_still_fails_loudly_when_a_match_day_has_weather_only_at_another_point(tmp_bronze, tmp_path):
    """Version 1 (Calcutta Oval point) plays on 2010-01-05, but the only file for that day was fetched at Eden
    Gardens, 3 km away (where the snapshot-wide build put it). As-of that weather needs a new request: stop."""
    wrong = ("eden-gardens__2010-01-04__2010-01-06.json.zst", EDEN, date(2010, 1, 4), date(2010, 1, 6), True)
    geo, wx = _geo(wd=WD2), _wx_files([wrong, V2_FILE], "w5")
    with pytest.raises(SystemExit, match=r"km from eden-gardens \(version 1\)"):
        _build(tmp_path, MATCHES, ASOF, "asof", geo, wx)


def test_t1_04_a_version_without_a_point_loses_weather_planned_at_a_later_point(tmp_bronze, tmp_path):
    """The weather for the 2010 days was planned at the snapshot-wide point (Eden Gardens). As-of, version 1 (the
    old spelling) has no point: its days lose weather (the design allows it); the file is not moved onto them."""
    wx = _wx_files([("eden-gardens__2010-01-04__2010-01-06.json.zst", EDEN, date(2010, 1, 4), date(2010, 1, 6), False)], "w2")
    geo = _geo()
    _build(tmp_path, MATCHES, SNAP, "snap", geo, wx)  # one version, at Eden Gardens: the file fits
    w = _weather(_build(tmp_path, MATCHES, ASOF, "asof", geo, wx))
    assert w["m1"] == w["m2"] == "venue_not_geocoded/no"


# ---------------------------------------------------------------- the tiny fixture (real snapshots)


@pytest.mark.parametrize("cutoff", [date(2015, 12, 31), date(2019, 7, 10), date(2020, 12, 31)])
def test_t1_04_tiny_fixture_cutoff_rebuild_with_asof(tiny_build, tmp_path, monkeypatch, cutoff):
    """The design's proof: rebuild the venue context with every match after the cutoff removed; every row of every
    match on or before the cutoff is identical (venue version, point, confidence, weather values)."""
    from conftest import FIXTURE
    monkeypatch.setattr(paths, "BRONZE", FIXTURE / "bronze")
    monkeypatch.setattr(bronze, "BRONZE", FIXTURE / "bronze")
    monkeypatch.setattr(vb, "SUBSET", True)
    lock = yaml.safe_load((FIXTURE / "snapshots.lock.yml").read_text())
    geo, wx = store.Snapshot("venues_geo", lock["venues_geo"]), store.Snapshot("weather", lock["weather"])
    loader = vb.zip_dates_loader(paths.snapshot_dir("cricsheet", lock["cricsheet"]) / "all_json.zip")
    ov = {**load_overrides(), "venue_context": "asof"}
    silver = tiny_build / "silver" / "matches.parquet"

    def rows(where, out):
        con = duckdb.connect()
        con.execute(f"CREATE VIEW matches AS SELECT * FROM read_parquet('{silver}') WHERE {where}")
        vb._build(con, geo=geo, wx=wx, out_dir=tmp_path / out, overrides=ov, reference=load_reference(), dates_loader=loader)
        return con.execute(f"""SELECT d.* EXCLUDE (venue_version), mv.venue_version, v.lat, v.lon, v.name, v.search_names
            FROM int_match_weather_daily d JOIN match_venues mv USING (match_id, venue_id)
            JOIN venue_versions v ON v.venue_id = mv.venue_id AND v.venue_version = mv.venue_version
            JOIN matches m USING (match_id)
            WHERE m.start_date <= DATE '{cutoff}' ORDER BY d.match_id, d.match_day_number""").fetchall()

    full, cut = rows("true", "full"), rows(f"start_date <= DATE '{cutoff}'", "cut")
    assert full == cut and len(full) > 0
    assert any(r[6] for r in full)  # weather_available on some early day: the check covers real weather values


def test_t1_04_weather_plan_requests_each_version_at_its_own_point(tmp_bronze):
    """For the one re-download: every version's match days are planned at that version's point."""
    from cricket_sim.venues import ingest as ing
    versions = _versions(MATCHES, ASOF)
    aliases = [{"venue": v, "city": c, "venue_id": "eden-gardens"} for v, c in
               [("Calcutta Oval", "Kolkata"), ("Calcutta Oval, Kolkata", "Kolkata"), ("Eden Gardens", "Kolkata")]]
    matches = [{"venue": r[1], "city": r[2], "dates": [r[3]], "gender": "male", "match_type": "ODI",
                "team_type": "international", "event_name": None, "teams": [r[4], r[5]]} for r in MATCHES]
    old_point = (EDEN[0] + 0.05, EDEN[1])
    points = {("eden-gardens", 1): old_point, ("eden-gardens", 2): EDEN}
    cfg = {"archive_lag_days": 6, "max_window_days": 14, "hourly": []}
    plan, reused = ing.weather_plan(cfg, matches, {"any_of": [{"match_type": "ODI"}]}, aliases, points, "g1", "sha",
                                    "c1", ing._Reuse("weather"), date(2026, 10, 1), versions=versions)
    got = [(f["file"], f.get("venue_version"), f["lat"], f["start"]) for f in plan["files"]]
    assert got == [("eden-gardens__2010-01-04__2010-01-06.json.zst", None, round(old_point[0], 5), "2010-01-04"),
                   ("eden-gardens--v2__2020-01-31__2020-02-02.json.zst", 2, EDEN[0], "2020-01-31")]
    assert reused == []
