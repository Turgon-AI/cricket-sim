"""Pure-function tests for the venue context (no network, no real data). Each past bug has a test that
fails on the old behaviour."""

from datetime import date, datetime, timedelta

import pytest

from cricket_sim import context_venues as cv
from cricket_sim.venues import checks, geocode as gc, weather as wx
from cricket_sim.venues.canonical import OverrideError, canonicalize, era_for, load_overrides

OVERRIDES = {"merge": {"Arun Jaitley Stadium": ["Feroz Shah Kotla"]}, "assign": {}, "split_by_city": ["County Ground"],
             "city_aliases": {}, "venues": {}, "eras": {}}
REF = checks.load_reference()


# ---------------------------------------------------------------- identity

def test_shim_keeps_public_entry_points():
    assert callable(cv.build_venue_context) and callable(cv.ingest_venues_and_weather)
    assert cv.norm("M.Chinnaswamy Stadium") == "m chinnaswamy stadium"
    assert cv.slugify("Lord's") == "lords"


def test_canonicalize_folds_spellings_and_splits_shared_names():
    pairs = [("Eden Gardens", "Kolkata", 100), ("Eden Gardens, Kolkata", "Kolkata", 50), ("Eden Gardens", None, 3),
             ("Feroz Shah Kotla", "Delhi", 80), ("Arun Jaitley Stadium, Delhi", "Delhi", 40),
             ("County Ground", "Bristol", 10), ("County Ground, Taunton", None, 2), ("County Ground", "Taunton", 2),
             ("County Ground", None, 1)]
    venues, aliases = canonicalize(pairs, OVERRIDES)
    ids = {v["venue_id"]: v for v in venues}
    assert set(ids) == {"eden-gardens", "arun-jaitley-stadium", "county-ground-bristol", "county-ground-taunton"}
    by_key = {(a["venue"], a["city"]): a["venue_id"] for a in aliases}
    assert by_key[("County Ground, Taunton", None)] == "county-ground-taunton"  # city read from the venue string
    assert by_key[("County Ground", None)] == "county-ground-bristol"            # else the most frequent city
    assert canonicalize(list(reversed(pairs)), OVERRIDES) == (venues, aliases)


def test_venue_id_does_not_depend_on_scope_or_counts():
    """Ids were computed from in-scope counts, so a scope change could rename venues."""
    everything = [("Sardar Patel Stadium, Motera", "Ahmedabad", 28), ("Narendra Modi Stadium, Ahmedabad", "Ahmedabad", 62),
                  ("Eden Gardens", "Kolkata", 5), ("Eden Gardens, Kolkata", "Kolkata", 900)]
    ov = {**OVERRIDES, "merge": {"Narendra Modi Stadium": ["Sardar Patel Stadium"]}}
    in_scope_only = [("Sardar Patel Stadium, Motera", "Ahmedabad", 3), ("Eden Gardens", "Kolkata", 5)]
    a1 = {(a["venue"], a["city"]): a["venue_id"] for a in canonicalize(everything, ov)[1]}
    a2 = {(a["venue"], a["city"]): a["venue_id"] for a in canonicalize(in_scope_only, ov, check_stale=False)[1]}
    assert all(a1[k] == v for k, v in a2.items())
    assert a2[("Sardar Patel Stadium, Motera", "Ahmedabad")] == "narendra-modi-stadium"


def test_merges_and_splits_in_committed_overrides():
    """Merges, splits and collisions found once silver covered every match."""
    pairs = [("Goodyear Park", "Bloemfontein", 1), ("Chevrolet Park", "Bloemfontein", 2), ("OUTsurance Oval", "Bloemfontein", 6),
             ("Mangaung Oval", "Bloemfontein", 9), ("North West Cricket Stadium, Potchefstroom", "Potchefstroom", 2),
             ("Senwes Park", "Potchefstroom", 34), ("North-West University No1 Ground", "Potchefstroom", 6),
             ("Castle Avenue", "Dublin", 4), ("Clontarf Cricket Club Ground", "Dublin", 15),
             ("Gucherre Cricket Ground", "Albergaria", 3), ("Santarem Cricket Ground", "Albergaria", 18),
             ("Bermuda National Stadium", "Hamilton", 1), ("National Stadium", "Hamilton", 11), ("National Stadium", "Karachi", 63),
             ("Vidarbha C.A. Ground", "Nagpur", 2), ("Vidarbha Cricket Association Ground", "Nagpur", 1),
             ("Vidarbha Cricket Association Stadium, Jamtha", "Nagpur", 30),
             ("Lord's", "London", 103), ("Lords, St David's Cricket Club Ground", "Bermuda", 3),
             ("University Oval", "Dunedin", 34), ("University Oval, Hobart", "Hobart", 2),
             ("Queen's Park", "Chesterfield", 7), ("Queen's Park", None, 1), ("Queen's Park, Invercargill", "Invercargill", 3),
             ("Marrara Cricket Ground", "Darwin", 3), ("Marrara Stadium, Darwin", "Darwin", 3)]
    _, aliases = canonicalize(pairs, load_overrides(), check_stale=False)
    vid = {(a["venue"], a["city"]): a["venue_id"] for a in aliases}
    assert len({vid[p[:2]] for p in pairs[:4]}) == 1 == len({"mangaung-oval"} | {vid[p[:2]] for p in pairs[:4]})
    assert vid[("North West Cricket Stadium, Potchefstroom", "Potchefstroom")] == vid[("Senwes Park", "Potchefstroom")]
    assert vid[("North-West University No1 Ground", "Potchefstroom")] != vid[("Senwes Park", "Potchefstroom")]
    assert vid[("Castle Avenue", "Dublin")] == vid[("Clontarf Cricket Club Ground", "Dublin")]
    assert vid[("Gucherre Cricket Ground", "Albergaria")] == vid[("Santarem Cricket Ground", "Albergaria")]
    assert vid[("National Stadium", "Hamilton")] == vid[("Bermuda National Stadium", "Hamilton")] == "bermuda-national-stadium"
    assert vid[("National Stadium", "Karachi")] == "national-stadium-karachi"
    # Nagpur: the old Civil Lines ground is one venue and is NOT Jamtha
    assert vid[("Vidarbha C.A. Ground", "Nagpur")] == vid[("Vidarbha Cricket Association Ground", "Nagpur")]
    assert vid[("Vidarbha Cricket Association Ground", "Nagpur")] != vid[("Vidarbha Cricket Association Stadium, Jamtha", "Nagpur")]
    assert vid[("Lords, St David's Cricket Club Ground", "Bermuda")] != vid[("Lord's", "London")] == "lords"
    assert vid[("University Oval, Hobart", "Hobart")] != vid[("University Oval", "Dunedin")] == "university-oval"
    assert vid[("Queen's Park", None)] == vid[("Queen's Park, Invercargill", "Invercargill")] == "queens-park-invercargill"
    assert vid[("Marrara Cricket Ground", "Darwin")] == vid[("Marrara Stadium, Darwin", "Darwin")]


def test_stale_overrides_fail_loudly():
    with pytest.raises(OverrideError, match="match no Cricsheet"):
        canonicalize([("Eden Gardens", "Kolkata", 1)], {**OVERRIDES, "merge": {"Eden Gardens": ["Gone Ground"]}})
    with pytest.raises(OverrideError, match="do not exist"):
        canonicalize([("Eden Gardens", "Kolkata", 1)], {**OVERRIDES, "merge": {}, "venues": {"no-such-venue": {"lat": 1}}})


def test_venue_eras():
    ov = {"eras": {"narendra-modi-stadium": [{"from": "2020-02-24"}]}}
    assert era_for("narendra-modi-stadium", date(2011, 3, 1), ov) == "narendra-modi-stadium"
    assert era_for("narendra-modi-stadium", date(2021, 2, 24), ov) == "narendra-modi-stadium@2020-02-24"


# ---------------------------------------------------------------- geocoding

WD = {"results": {"bindings": [
    {"item": {"value": "http://www.wikidata.org/entity/Q2035004"}, "itemLabel": {"value": "Eden Gardens"},
     "c": {"value": "Point(88.3433 22.5645)"}, "locLabel": {"value": "Kolkata"}, "countryLabel": {"value": "India"},
     "elev": {"value": "9"}},
    {"item": {"value": "http://www.wikidata.org/entity/Q1"}, "itemLabel": {"value": "County Ground"},
     "c": {"value": "Point(-2.5 51.4)"}, "locLabel": {"value": "Bristol"}, "countryLabel": {"value": "United Kingdom"}},
    {"item": {"value": "http://www.wikidata.org/entity/Q2"}, "itemLabel": {"value": "County Ground"},
     "c": {"value": "Point(-1.8 52.6)"}, "locLabel": {"value": "Derby"}, "countryLabel": {"value": "United Kingdom"}},
]}}


class Provider:
    def __init__(self, nominatim=None, entity=None, osm=None):
        self._n, self._e, self._o = nominatim or {}, entity or {}, osm or {}

    def nominatim(self, q):
        return self._n.get(q)

    def peek_nominatim(self, q):
        return self._n.get(q)

    def entity(self, qid):
        return self._e.get(qid)

    def osm(self, ref):
        return self._o.get(ref)


def _venue(name, city):
    return {"venue_id": cv.slugify(name), "name": name, "city": city, "search_names": [name], "places": [city]}


def test_wikidata_match_and_nominatim_fallback():
    cands = gc.parse_wikidata(WD)
    g = gc.resolve_geocode(_venue("Eden Gardens", "Kolkata"), cands, Provider(), {})
    assert (g["geocode_source"], g["wikidata_qid"], g["declared_elevation_m"]) == ("wikidata", "Q2035004", 9.0)
    assert gc.match_wikidata(_venue("County Ground", "Taunton"), cands) is None  # ambiguous -> refuse
    city = [{"lat": "1", "lon": "2", "display_name": "Kathmandu", "osm_type": "node", "osm_id": 9}]
    g = gc.resolve_geocode(_venue("Nowhere Oval", "Kathmandu"), cands, Provider({"Kathmandu": city}), {})
    assert (g["geocode_source"], g["geocode_confidence"]) == ("nominatim_city", "low")
    # outside the geo plan: no network-type request is made at all
    g = gc.resolve_geocode(_venue("Nowhere Oval", "Kathmandu"), cands, None, {}, searchable=False)
    assert g["lat"] is None


def test_pinned_qid_without_coordinates_raises_instead_of_falling_back():
    """Udayana's pinned QID had no P625 and silently fell through to the Bali centroid."""
    entity = {"entities": {"Q123494664": {"labels": {"en": {"value": "Udayana Cricket Ground"}}, "claims": {}}}}
    with pytest.raises(gc.GeocodeError, match="no coordinates"):
        gc.resolve_geocode(_venue("Udayana Cricket Ground", "Bali"), [], Provider(entity={"Q123494664": entity}),
                           {"wikidata_qid": "Q123494664"})


def test_osm_pin_resolves_through_nominatim_lookup():
    lookup = [{"osm_type": "way", "osm_id": 524758013, "lat": "-1.25819", "lon": "36.85754", "name": "Ruaraka Sports Club",
               "address": {"country": "Kenya"}}]
    g = gc.resolve_geocode(_venue("Ruaraka Sports Club Ground", "Nairobi"), [], Provider(osm={"way/524758013": lookup}),
                           {"osm_ref": "way/524758013"})
    assert g["lat"] < 0 and (g["geocode_source"], g["geocoder_country"]) == ("osm_pinned", "Kenya")
    assert gc.osm_lookup_id("way/524758013") == "W524758013"


# ---------------------------------------------------------------- sanity checks

def _statuses(lat, lon, elev, city, tz, *, override=None, home=None, label=None, coastal=False):
    c, country, _ = checks.country_check(lat, lon, override=override, home=home, label=label, ref=REF)
    e, _ = checks.elevation_check(elev, city, REF)
    return {"country": c, "elevation": e, "timezone": checks.timezone_check(tz), "sea": checks.sea_check(elev, coastal),
            "anchor": "unknown"}


@pytest.mark.parametrize("case", [
    # (name, old bad point, old elevation, city, tz, home team, failing check)
    ("Ruaraka (+1.259 sign error)", 1.259, 36.858, 1266, "Nairobi", "Africa/Nairobi", "Kenya", "elevation"),
    ("Happy Valley in the sea", 34.403, 32.530, 0.0, "Episkopi", "Etc/GMT-2", "Cyprus", "country"),
    ("Happy Valley in the sea (tz)", 34.403, 32.530, 0.0, "Episkopi", "Etc/GMT-2", "Cyprus", "timezone"),
    ("Happy Valley in the sea (dem)", 34.403, 32.530, 0.0, "Episkopi", "Etc/GMT-2", "Cyprus", "sea"),
    ("Happy Valley 2 in Crete", 35.257, 25.238, 360, "Episkopi", "Europe/Athens", "Cyprus", "country"),
    ("Udayana Bali centroid", -8.227, 115.192, 1268, "Bali", "Asia/Makassar", "Indonesia", "elevation"),
    ("Arnos Vale island centroid", 13.253, -61.195, 650, "Kingstown", "America/St_Vincent", "West Indies", "elevation"),
])
def test_known_bad_points_fail_a_check(case):
    _, lat, lon, elev, city, tz, home, failing = case
    assert _statuses(lat, lon, elev, city, tz, home=home)[failing] == "fail"


@pytest.mark.parametrize("case", [
    (-1.25819, 36.85754, 1637, "Nairobi", "Africa/Nairobi", "Kenya"),
    (34.67229, 32.82147, 18, "Episkopi", "Asia/Nicosia", "Cyprus"),
    (-8.79473, 115.17390, 59, "Bali", "Asia/Makassar", "Indonesia"),
    (13.14290, -61.21069, 5, "Kingstown", "America/St_Vincent", "West Indies"),
])
def test_fixed_points_pass(case):
    lat, lon, elev, city, tz, home = case
    st = _statuses(lat, lon, elev, city, tz, home=home)
    assert st["country"] == st["elevation"] == st["timezone"] == st["sea"] == "pass"


def test_needs_review_false_only_for_trusted_and_passing_rows():
    ok = {"country": "pass", "elevation": "pass", "timezone": "pass", "sea": "pass", "anchor": "unknown"}
    row = {"venue_id": "x", "geocode_confidence": "high"}
    assert checks.assess(row, {}, ok) == (False, [], [])
    assert checks.assess({**row, "geocode_confidence": "low"}, {}, ok)[0] is True
    assert checks.assess(row, {}, {**ok, "elevation": "unknown"})[0] is True
    review, _, failed = checks.assess(row, {}, {**ok, "country": "fail"})
    assert review and failed == ["country"]                       # an unacknowledged failure fails the build
    review, _, failed = checks.assess(row, {"acknowledge": ["country"]}, {**ok, "country": "fail"})
    assert review and failed == []                                # acknowledged: build passes, row stays in review
    review, _, failed = checks.assess(row, {"acknowledge": ["anchor"]}, {**ok, "anchor": "fail"})
    assert not review and failed == []                            # metro-name anchors may be explained away


def test_home_team_and_label_evidence():
    assert checks.home_team({"Cyprus": 30, "Greece": 10, "Estonia": 5}, 35, REF) == "Cyprus"
    assert checks.home_team({"India": 3, "Pakistan": 3}, 6, REF) is None  # neutral venue
    assert checks.country_check(51.53, -0.17, override=None, home=None, label="United Kingdom", ref=REF)[1] == "England"
    assert checks.country_check(19.38, -81.40, override=None, home="Cayman Islands", label="United Kingdom", ref=REF)[1] == "Cayman Islands"


# ---------------------------------------------------------------- weather timing

def _resp(local_start: date, days: int, offset_s: int, tz="Europe/London", precip_at_utc=None):
    """An Open-Meteo-like answer: hourly local labels at a fixed offset; values keyed by the UTC instant."""
    utc0 = datetime.combine(local_start, datetime.min.time()) - timedelta(seconds=offset_s)
    utc = [utc0 + timedelta(hours=h) for h in range(days * 24)]
    loc = [t + timedelta(seconds=offset_s) for t in utc]
    hourly = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in loc]}
    for v in wx.WX_VARS:
        hourly[v] = [float(t.hour) for t in utc]
    hourly["wind_direction_10m"] = [350.0 if t.hour % 2 else 10.0 for t in utc]
    hourly["precipitation"] = [(2.0 if precip_at_utc and t in precip_at_utc else 0.0) for t in utc]
    return {"timezone": tz, "utc_offset_seconds": offset_s, "elevation": 30.0, "hourly": hourly}


def test_parsed_series_does_not_depend_on_the_fetch_date_offset():
    """A London request fetched in summer (BST +3600) and in winter (GMT +0) must give the same UTC rows."""
    summer = dict(wx.parse_response(_resp(date(2019, 7, 9), 2, 3600)))
    winter = dict(wx.parse_response(_resp(date(2019, 7, 9), 2, 0)))
    overlap = set(summer) & set(winter)
    assert len(overlap) >= 47 and all(summer[t] == winter[t] for t in overlap)


def test_non_hourly_series_is_refused():
    r = _resp(date(2019, 3, 30), 2, 0)
    r["hourly"]["time"][5] = r["hourly"]["time"][4]
    with pytest.raises(ValueError, match="Open-Meteo series is not hourly at 2019-03-30 04:00"):
        wx.parse_response(r)


def test_default_window_uses_zoneinfo_at_the_match_date_across_dst():
    before = wx.day_windows([date(2019, 3, 30)], "Test", "Europe/London", [])[0]
    after = wx.day_windows([date(2019, 4, 1)], "Test", "Europe/London", [])[0]
    assert before["window_start_utc"] == datetime(2019, 3, 30, 10) and after["window_start_utc"] == datetime(2019, 4, 1, 9)
    syd_summer = wx.day_windows([date(2020, 1, 3)], "Test", "Australia/Sydney", [])[0]
    syd_winter = wx.day_windows([date(2020, 6, 3)], "Test", "Australia/Sydney", [])[0]
    assert syd_summer["window_start_utc"] == datetime(2020, 1, 2, 23) and syd_winter["window_start_utc"] == datetime(2020, 6, 3, 0)
    # historical offsets come from the tz database, not the present: Colombo was UTC+6 in 2006
    assert wx.local_to_utc(datetime(2006, 1, 1, 10), "Asia/Colombo") == datetime(2006, 1, 1, 4)
    assert wx.local_to_utc(datetime(2020, 1, 1, 10), "Asia/Colombo") == datetime(2020, 1, 1, 4, 30)


def test_ball_timestamps_set_the_window_and_rest_days_get_no_window():
    t0 = datetime(2021, 6, 2, 10, 0)
    balls = [t0 + timedelta(minutes=2 * k) for k in range(200)] + [datetime(2021, 6, 4, 10) + timedelta(minutes=2 * k) for k in range(100)]
    days = wx.day_windows([date(2021, 6, 2), date(2021, 6, 4)], "Test", "Europe/London", wx.ball_clusters(balls))
    assert [d["window_source"] for d in days] == ["ball_timestamps", "ball_timestamps"]
    assert days[0]["window_start_utc"] == t0 and days[0]["window_end_utc"] == balls[199] + timedelta(minutes=5)
    assert [d["match_date"] for d in days] == [date(2021, 6, 2), date(2021, 6, 4)]  # 3 June (rest day) absent
    assert wx.day_windows([date(2021, 6, 3)], "Test", "Europe/London", wx.ball_clusters(balls))[0]["window_source"] == "default_estimate"


def test_aggregates_ignore_rain_after_play_and_report_pre_match_rain_separately():
    rain = {datetime(2019, 7, 9, 21), datetime(2019, 7, 9, 3)}  # 22:00 BST (after a 17:00 BST finish) and 04:00 BST
    series = dict(wx.parse_response(_resp(date(2019, 7, 8), 3, 3600, precip_at_utc=rain)))
    start, end = datetime(2019, 7, 9, 9), datetime(2019, 7, 9, 16)  # 10:00-17:00 BST
    agg = wx.aggregate(series, start, end)
    assert agg["window_precipitation_mm"] == 0.0           # the evening shower is after play ended
    assert agg["pre_match_12h_precipitation_mm"] == 2.0    # the overnight shower is pre-match, labelled as such
    assert agg["window_temperature_c"] == sum(range(9, 17)) / 8
    assert agg["window_hours_present"] == agg["window_hours_expected"] == 8
    assert abs(agg["window_wind_direction_deg"] % 360) < 1 or abs(agg["window_wind_direction_deg"] - 360) < 1


def test_half_hour_zone_series_sits_on_its_own_grid():
    series = dict(wx.parse_response(_resp(date(2008, 4, 20), 1, 19800, tz="Asia/Kolkata")))
    assert wx.series_grid(series) == 30
    day = wx.day_windows([date(2008, 4, 20)], "IPL", "Asia/Kolkata", [])[0]  # 15:30-23:30 IST
    agg = wx.aggregate(series, day["window_start_utc"], day["window_end_utc"])
    assert agg["window_hours_present"] > 0


def test_plan_windows_merges_days_and_defers_the_archive_lag():
    d = date
    wins, deferred = wx.plan_windows({"a": {d(2020, 1, 1), d(2020, 1, 8), d(2020, 3, 1)}, "b": {d(2026, 10, 12)}}, 14, d(2026, 10, 9))
    assert wins == [("a", d(2020, 1, 1), d(2020, 1, 8)), ("a", d(2020, 3, 1), d(2020, 3, 1))]
    assert deferred == [("b", d(2026, 10, 12))]


# ---------------------------------------------------------------- Split grounds, geocoder country bound

def test_t3_09_split_grounds_are_one_venue():
    """WACA, Centurion, Launceston and the Ekana typo were each two venue ids for one ground."""
    pairs = [("WACA, Perth", "Perth", 4), ("W.A.C.A. Ground", "Perth", 32),
             ("Western Australia Cricket Association Ground", "Perth", 64),
             ("Western Australia Cricket Association Ground, Perth", "Perth", 79),
             ("Centurion Park", "Centurion", 1), ("SuperSport Park", "Centurion", 74),
             ("SuperSport Park, Centurion", "Centurion", 86),
             ("Aurora Stadium", "Launceston", 11), ("Aurora Stadium, Launceston", "Launceston", 9),
             ("University of Tasmania Stadium, Launceston", "Launceston", 8),
             ("Bharat Ratna Shri Atal Bihari Vajpai Ekana Cricket Stadium", "Lucknow", 2),
             ("Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium", "Lucknow", 7),
             ("Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow", "Lucknow", 41),
             ("Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium B", None, 5)]
    _, aliases = canonicalize(pairs, load_overrides(), check_stale=False)
    vid = {(a["venue"], a["city"]): a["venue_id"] for a in aliases}
    groups = [pairs[0:4], pairs[4:7], pairs[7:10], pairs[10:13]]
    assert [sorted({vid[p[:2]] for p in g}) for g in groups] == [
        ["western-australia-cricket-association-ground"], ["supersport-park"], ["university-of-tasmania-stadium"],
        ["bharat-ratna-shri-atal-bihari-vajpayee-ekana-cricket-stadium"]]
    # the second Ekana ground (B) stays its own venue
    assert vid[pairs[13][:2]] == "bharat-ratna-shri-atal-bihari-vajpayee-ekana-cricket-stadium-b"


YORK_PARK = {"qid": "Q5360275", "label": "York Park", "lat": -41.425833, "lon": 147.138889, "n_coords": 1, "elev": None,
             "locs": ["Launceston"], "country": "Australia", "names": ["Aurora Stadium", "York Park"]}
CLIFTON_PARK = {"qid": "Q5133289", "label": "Clifton Park Ground", "lat": 53.9725, "lon": -1.10584, "n_coords": 1,
                "elev": None, "locs": [], "country": "United Kingdom", "names": ["Clifton Park Ground"]}


def test_t3_09_geocoder_rejects_a_point_outside_the_expected_country():
    """York Cricket Club (England) was matched to York Park, Launceston by the shared word 'York', with confidence
    high, and passed the country check on the geocoder's own label."""
    york = _venue("York Cricket Club", "York")
    # outside the geo plan: the automatic Wikidata match lands in Tasmania and must be rejected, not kept
    g = gc.resolve_geocode(york, [YORK_PARK], None, {"country": "England"}, searchable=False)
    assert g["lat"] is None and g["geocode_confidence"] is None
    assert "England" in (g["geocode_gap"] or "") and "Q5360275" in g["geocode_gap"]
    # in the plan: a Nominatim hit outside England is rejected the same way
    tas = [{"lat": "-41.4", "lon": "147.1", "display_name": "York Cricket Club, Launceston, Tasmania, Australia",
            "osm_type": "node", "osm_id": 1, "address": {"country": "Australia"}}]
    g = gc.resolve_geocode(york, [YORK_PARK], Provider({"York Cricket Club, York": tas}), {"country": "England"})
    assert g["lat"] is None and "England" in (g["geocode_gap"] or "")
    # a point inside the expected country is kept; no expected country: behaviour unchanged
    eng = [{**tas[0], "lat": "53.97", "lon": "-1.10", "display_name": "York Cricket Club, Shipton Road, York, England"}]
    g = gc.resolve_geocode(york, [YORK_PARK], Provider({"York Cricket Club, York": eng}), {"country": "England"})
    assert (g["geocode_source"], round(g["lat"], 2), g["geocode_gap"]) == ("nominatim", 53.97, None)
    assert gc.resolve_geocode(york, [YORK_PARK], None, {}, searchable=False)["wikidata_qid"] == "Q5360275"


def test_t3_09_york_cricket_club_override_lands_in_england():
    fix = load_overrides()["venues"]["york-cricket-club"]
    assert fix["country"] == "England"
    g = gc.resolve_geocode(_venue("York Cricket Club", "York"), [YORK_PARK, CLIFTON_PARK], None, fix, searchable=False)
    assert checks.in_territory(g["lat"], g["lon"], "England", REF)
    status, country, _ = checks.country_check(g["lat"], g["lon"], override=fix["country"], home=None,
                                              label=g["geocoder_country"], ref=REF)
    assert (status, country) == ("pass", "England")
