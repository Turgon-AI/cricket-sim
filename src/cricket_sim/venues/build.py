"""Offline, deterministic build of the venue context. Scope-free: covers every match and venue in silver.

Inputs: silver view `matches` (all matches) and, when present, the staging view `stg_ari_tracking` (per-ball UTC timestamps); the
locked, strictly verified bronze snapshots venues_geo and weather (only manifest-listed files are read);
config/venue_overrides.yml and config/venue_reference.yml. It never reads config/scope.yml.

Silver outputs
  venues                       one row per canonical venue (its latest version), with sanity-check results and review flags
  venue_versions               one row per venue version (one per venue unless venue_context: asof)
  venue_aliases                every Cricsheet (venue, city) string -> venue_id
  match_venues                 every match -> venue_id, venue_version, venue_era
  int_match_weather_daily      (gold) every match day of every match; weather columns null + reason when not available
  match_weather_hourly         hours around a format-default window, independent of Cricinfo timestamps
  int_match_play_windows       (gold) exact first/last-ball windows (Cricinfo-derived timing: never exported)
"""

from __future__ import annotations

import json
import zipfile
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import pyarrow as pa

from .. import bronze
from ..paths import GOLD, SILVER, SUBSET, snapshot_dir
from . import asof, checks
from . import geocode as gc
from . import store
from .canonical import canonicalize, era_for, load_overrides
from .weather import (DEFAULT_WINDOWS, WX_COLS, aggregate, ball_clusters, day_windows, local_to_utc, match_class,
                      parse_response, series_grid, to_local, window_hours)

GEO_SOURCE = "venues_geo"
WX_SOURCE = "weather"
MAX_WEATHER_OFFSET_KM = 2.0
DEM_ELEVATION_MAX_OFFSET_KM = 0.1


class BuildProvider:
    """Answers geocode requests from a sealed snapshot only."""

    def __init__(self, snap: store.Snapshot):
        self.snap = snap

    def _get(self, name: str):
        if self.snap.has(name):
            return self.snap.response(name)
        if name in self.snap.failures:
            return None  # recorded gap: the snapshot was sealed with allow_gaps
        raise store.MissingRequest(f"{self.snap.source}/{self.snap.snapshot_id} lacks {name}, which the current "
                                   "venue_overrides.yml / plan needs; run ingest_venues_and_weather() again")

    def nominatim(self, query):
        return self._get(gc.nominatim_file(query))

    def peek_nominatim(self, query):
        name = gc.nominatim_file(query)
        return self.snap.response(name) if self.snap.has(name) else None

    def entity(self, qid):
        return self._get(gc.entity_file(qid))

    def osm(self, ref):
        return self._get(gc.osm_file(ref))


def geocode_venues(venues: list[dict], planned: set, overrides: dict, provider, cands: list[dict]) -> dict[str, dict]:
    out = {}
    for v in venues:
        fix = (overrides.get("venues") or {}).get(v["venue_id"]) or {}
        out[v["venue_id"]] = gc.resolve_geocode(v, cands, provider, fix, searchable=v["venue_id"] in planned)
    return out


def geocode_versions(flat: list[dict], planned: set, overrides: dict, provider, cands: list[dict]) -> list[dict]:
    """resolve_geocode for every venue version (asof.venue_versions), in order. Same rules as geocode_venues."""
    fixes = overrides.get("venues") or {}
    return [gc.resolve_geocode(x["venue"], cands, provider, fixes.get(x["venue_id"]) or {},
                               searchable=x["venue_id"] in planned) for x in flat]


# ---------------------------------------------------------------- inputs


def zip_dates_loader(zip_path: Path):
    """match_id -> exact Cricsheet dates, read from the bronze zip (only for matches with gaps in their dates)."""
    def load(match_ids: list[str]) -> dict[str, list[date]]:
        out = {}
        with zipfile.ZipFile(zip_path) as zf:
            for mid in sorted(match_ids):
                info = json.loads(zf.read(f"{mid}.json"))["info"]
                out[mid] = sorted(date.fromisoformat(str(d)) for d in info["dates"])
        return out
    return load


def _matches(con) -> list[dict]:
    cur = con.execute("""SELECT match_id, venue, city, start_date, coalesce(end_date, start_date) AS end_date,
                                coalesce(n_dates, 1) AS n_dates, match_type, event_name, team_type, team1, team2
                         FROM matches ORDER BY match_id""")
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _match_dates(matches: list[dict], dates_loader) -> dict[str, list[date]]:
    out, gaps = {}, []
    for m in matches:
        span = (m["end_date"] - m["start_date"]).days + 1
        if m["n_dates"] == span:
            out[m["match_id"]] = [m["start_date"] + timedelta(days=i) for i in range(span)]
        else:
            gaps.append(m["match_id"])
    if gaps:
        if dates_loader is None:
            raise SystemExit(f"{len(gaps)} matches have non-consecutive dates and no dates loader was given")
        out.update(dates_loader(gaps))
    return out


def _ball_times(con) -> dict[str, list[datetime]]:
    have = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    if "stg_ari_tracking" not in have:
        return {}
    rows = con.execute("""SELECT match_id, list(ball_timestamp ORDER BY ball_timestamp) FROM stg_ari_tracking
                          WHERE ball_timestamp IS NOT NULL GROUP BY match_id""").fetchall()
    return {mid: ts for mid, ts in rows}


# ---------------------------------------------------------------- build


def _weather_files(wx: store.Snapshot, alias_id: dict, geo: dict, versions: dict, match_days: dict) -> tuple[dict, dict, dict]:
    """Assign every planned weather file to the versions of its venue whose point is within 2 km of the file's
    request point. Then check that every match day of a located version that a file covers is covered by a file at
    that version's point: else the weather was fetched at another point and must be fetched again (fail loudly).
    Only real match days count: the ingest pads each version's days by one day on each side, and a padding day may
    be the last day another version is valid. A version with no point loses weather on its days.
    -> (series per (venue, version): utc hour -> values, site info per (venue, version): timezone + DEM elevation,
    counts of files consumed and skipped by reason). One version per venue (snapshot mode) is the old check."""
    if wx.plan is None:
        raise SystemExit(f"{WX_SOURCE}/{wx.snapshot_id} has no plan.json (made by an older ingest); "
                         "run ingest_venues_and_weather() to make a planned snapshot")
    listed = {f["file"] for f in wx.plan["files"]}
    unplanned = sorted(n for n in wx.names() if n not in listed)
    if unplanned:
        raise SystemExit(f"weather files in the manifest but not in plan.json: {unplanned[:5]}")
    candidates = defaultdict(list)  # (venue, version) -> [(distance, file, rows, response)]
    errors, counts = [], Counter()
    by_venue = defaultdict(list)  # venue -> [(file, start, end, {version: km})]
    for f in sorted(wx.plan["files"], key=lambda f: f["file"]):
        ids = {alias_id.get((v, c)) for v, c in f["aliases"]} - {None}
        if not ids:
            counts["skipped: no match uses its spellings"] += 1  # e.g. a date-cut subset
            continue
        if len(ids) != 1:
            errors.append(f"{f['file']}: its venue strings now map to {sorted(ids)}")
            continue
        vid = ids.pop()
        vs = versions.get(vid) or []
        located = [v for v in vs if geo[(vid, v["venue_version"])]["lat"] is not None]
        if not located:
            errors.append(f"{f['file']}: venue {vid} has no coordinates any more")
            continue
        dist = {v["venue_version"]: checks.km(geo[(vid, v["venue_version"])]["lat"], geo[(vid, v["venue_version"])]["lon"],
                                              float(f["lat"]), float(f["lon"])) for v in located}
        f_start, f_end = date.fromisoformat(f["start"]), date.fromisoformat(f["end"])
        by_venue[vid].append((f, f_start, f_end, dist))
        fits = [(ver, d) for ver, d in dist.items() if d <= MAX_WEATHER_OFFSET_KM]
        serves = any(f_start <= day <= f_end for ver, _ in fits for day in match_days.get((vid, ver), ()))
        if not fits:
            counts["skipped: request point fits no version of the venue"] += 1
            continue
        if f["file"] not in wx.entries:
            if f["file"] in wx.failures:
                counts["recorded as failed in the snapshot"] += 1
                continue
            errors.append(f"{f['file']}: in plan.json but neither stored nor recorded as failed")
            continue
        counts["consumed" if serves else "stored, fits a version but serves none of its match days"] += 1
        _, resp = wx.record(f["file"])
        rows = parse_response(resp)
        for ver, d in fits:
            candidates[(vid, ver)].append((round(d, 6), f["file"], rows, resp))
    for vid, files in sorted(by_venue.items()):
        vs = versions[vid]
        for v in vs:
            ver, g = v["venue_version"], geo[(vid, v["venue_version"])]
            if g["lat"] is None:
                continue  # no point: its match days lose weather
            for day in sorted(match_days.get((vid, ver), ())):
                covering = [x for x in files if x[1] <= day <= x[2]]
                if covering and not any(x[3][ver] <= MAX_WEATHER_OFFSET_KM for x in covering):
                    f, d = covering[0][0], covering[0][3][ver]
                    label = vid if len(vs) == 1 else f"{vid} (version {ver})"
                    errors.append(f"{f['file']}: requested at {f['lat']},{f['lon']}, {d:.1f} km from {label}'s current point "
                                  f"{g['lat']:.5f},{g['lon']:.5f} (coordinates changed since the weather snapshot; "
                                  f"match day {day} needs weather at that point)")
                    break
    if errors:
        raise SystemExit("weather snapshot does not fit the current venues:\n  " + "\n  ".join(errors[:30]))

    series, site = {}, {}
    for vid, files in candidates.items():
        files.sort(key=lambda x: (x[0], x[1]))  # nearest request point first, then file name: deterministic
        merged = {}
        for _, _, rows, _ in files:
            for t, values in rows:
                merged.setdefault(t, values)
        series[vid] = merged
        d0, _, _, resp = files[0]
        site[vid] = {"timezone": resp.get("timezone"),
                     "dem_elevation_m": float(resp["elevation"]) if d0 <= DEM_ELEVATION_MAX_OFFSET_KM and resp.get("elevation") is not None else None}
    return series, site, dict(counts)


VENUE_SCHEMA = pa.schema([
    ("venue_id", pa.string()), ("name", pa.string()), ("city", pa.string()), ("country", pa.string()),
    ("home_team", pa.string()), ("lat", pa.float64()), ("lon", pa.float64()),
    ("elevation_m", pa.float64()), ("elevation_source", pa.string()), ("elevation_dem_m", pa.float64()),
    ("timezone", pa.string()), ("wikidata_qid", pa.string()), ("osm_ref", pa.string()),
    ("geocode_source", pa.string()), ("geocode_confidence", pa.string()), ("geocode_matched_name", pa.string()),
    ("in_geo_plan", pa.bool_()), ("geocode_gap", pa.string()),
    ("country_check", pa.string()), ("country_evidence", pa.string()), ("elevation_check", pa.string()),
    ("city_elevation_m", pa.float64()), ("timezone_check", pa.string()), ("sea_check", pa.string()),
    ("anchor_check", pa.string()), ("acknowledged_checks", pa.string()),
    ("needs_review", pa.bool_()), ("review_reasons", pa.string()), ("weather_location_quality", pa.string()),
    ("note", pa.string()),
    ("n_matches", pa.int64()), ("first_match_date", pa.date32()), ("last_match_date", pa.date32()),
])
_VERSION_EXTRA = [("venue_version", pa.int16()), ("valid_from", pa.date32()), ("valid_to", pa.date32()),
                  ("inputs_as_of", pa.date32()),
                  ("search_names", pa.list_(pa.string())), ("places", pa.list_(pa.string()))]
VERSION_SCHEMA = pa.schema([VENUE_SCHEMA.field("venue_id"), *_VERSION_EXTRA,
                            *[f for f in VENUE_SCHEMA if f.name not in ("venue_id", "first_match_date", "last_match_date")]])
DAILY_SCHEMA = pa.schema([
    ("match_id", pa.string()), ("match_day_number", pa.int16()), ("match_date", pa.date32()), ("venue_id", pa.string()),
    ("venue_version", pa.int16()),
    ("timezone", pa.string()), ("window_source", pa.string()), ("weather_available", pa.bool_()),
    ("weather_missing_reason", pa.string()), ("geocode_confidence", pa.string()), ("weather_location_quality", pa.string()),
    ("window_hours_expected", pa.int16()), ("window_hours_present", pa.int16()),
    ("window_temperature_c", pa.float64()), ("window_max_temperature_c", pa.float64()), ("window_humidity_pct", pa.float64()),
    ("window_dew_point_c", pa.float64()), ("window_dew_point_depression_c", pa.float64()),
    ("window_cloud_cover_pct", pa.float64()), ("window_wind_speed_kmh", pa.float64()),
    ("window_wind_direction_deg", pa.float64()), ("window_station_pressure_hpa", pa.float64()),
    ("window_precipitation_mm", pa.float64()), ("window_precip_hours_present", pa.int16()),
    ("window_max_hourly_precipitation_mm", pa.float64()), ("window_hours_with_rain", pa.int16()),
    ("pre_match_12h_precipitation_mm", pa.float64()), ("pre_match_hours_present", pa.int16()),
])
HOURLY_SCHEMA = pa.schema([
    ("match_id", pa.string()), ("match_day_number", pa.int16()), ("venue_id", pa.string()), ("venue_version", pa.int16()),
    ("utc_datetime", pa.timestamp("us")), ("local_datetime", pa.timestamp("us")), ("phase", pa.string()),
    *[(c, pa.float64()) for c in WX_COLS],
])
WINDOW_SCHEMA = pa.schema([
    ("match_id", pa.string()), ("match_day_number", pa.int16()), ("match_date", pa.date32()), ("venue_id", pa.string()),
    ("window_source", pa.string()), ("window_start_utc", pa.timestamp("us")), ("window_end_utc", pa.timestamp("us")),
    ("window_start_local", pa.timestamp("us")), ("window_end_local", pa.timestamp("us")), ("n_balls_timed", pa.int32()),
    ("internal_note", pa.string()),
])

INTERNAL_TIMING = "internal-only: windows from Cricinfo-derived ball timestamps (Ari) are not redistributable"


def _location_quality(row: dict) -> str | None:
    if row["lat"] is None:
        return None
    if row["geocode_source"] == "nominatim_city":
        return "city_centroid"
    return "verified_ground" if not row["needs_review"] else "unverified_point"


def _write(con, table: pa.Table, out_dir: Path, name: str, order_by: str) -> None:
    con.register(f"_ctx_{name}", table)
    dest = out_dir / f"{name}.parquet"
    con.execute(f"COPY (SELECT * FROM _ctx_{name} ORDER BY {order_by}) TO '{dest}' (FORMAT parquet, COMPRESSION zstd)")
    con.unregister(f"_ctx_{name}")
    con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{dest}')")


def _build(con, *, geo: store.Snapshot, wx: store.Snapshot, out_dir: Path, overrides: dict, reference: dict,
           dates_loader=None, derived_dir: Path | None = None) -> dict:
    """Facts (venues, aliases, match venues, hourly weather) go to out_dir (silver); the two tables we DERIVE
    (daily play-window weather and the play windows themselves) go to derived_dir (gold, as int_ tables)."""
    derived_dir = derived_dir or out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    matches = _matches(con)
    pair_n = Counter((m["venue"], m["city"]) for m in matches)
    pairs = sorted(((v, c, n) for (v, c), n in pair_n.items()), key=lambda r: (r[0], r[1] or ""))
    venues, aliases = canonicalize(pairs, overrides, check_stale=not SUBSET)
    alias_id = {(a["venue"], a["city"]): a["venue_id"] for a in aliases}
    fixes = overrides.get("venues") or {}
    versions = asof.venue_versions(matches, venues, alias_id, overrides, reference)

    # geocode: planned venues (in the geo plan, by their Cricsheet strings) may use every cached request
    if geo.plan is None:
        raise SystemExit(f"{GEO_SOURCE}/{geo.snapshot_id} has no plan.json (made by an older ingest); "
                         "run ingest_venues_and_weather() to make a planned snapshot")
    plan_aliases = {(v, c) for pv in geo.plan["venues"] for v, c in pv["aliases"]}
    planned = {alias_id[p] for p in plan_aliases if p in alias_id}
    cands = gc.parse_wikidata(geo.response(gc.WIKIDATA_FILE))
    # every version is geocoded with the same rules; a request the snapshot lacks raises store.MissingRequest
    flat = [x for vid in sorted(versions) for x in versions[vid]]
    geo_rows = {(x["venue_id"], x["venue_version"]): g for x, g in
                zip(flat, geocode_versions(flat, planned, overrides, BuildProvider(geo), cands))}

    dates = _match_dates(matches, dates_loader)
    match_days = defaultdict(set)  # (venue, version) -> every day of its matches
    for m in matches:
        vid = alias_id[(m["venue"], m["city"])]
        match_days[(vid, asof.version_for(versions[vid], m["start_date"])["venue_version"])].update(dates[m["match_id"]])
    series, site, file_counts = _weather_files(wx, alias_id, geo_rows, versions, match_days)
    span = defaultdict(list)
    for m in matches:
        span[alias_id[(m["venue"], m["city"])]].append(m["start_date"])

    rows, version_rows, failures = [], [], []
    for x in flat:
        vid, ver, v = x["venue_id"], x["venue_version"], x["venue"]
        key = (vid, ver)
        g, fix, s = geo_rows[key], fixes.get(vid) or {}, site.get(key, {})
        tz = fix.get("timezone") or s.get("timezone")
        dem = s.get("dem_elevation_m")
        if fix.get("elevation_m") is not None:
            elev, elev_src = float(fix["elevation_m"]), "manual"
        elif g["declared_elevation_m"] is not None:
            elev, elev_src = float(g["declared_elevation_m"]), "wikidata"
        elif dem is not None:
            elev, elev_src = dem, "open-meteo 90 m DEM at the point"
        else:
            elev, elev_src = None, None
        c_status, country, c_evidence = checks.country_check(
            g["lat"], g["lon"], override=fix.get("country"), home=x["home_team"], label=g["geocoder_country"], ref=reference)
        e_status, city_elev = checks.elevation_check(elev, v["city"], reference, fix.get("city_elevation_m"))
        statuses = {"country": c_status, "elevation": e_status, "timezone": checks.timezone_check(tz),
                    "sea": checks.sea_check(dem, bool(fix.get("coastal"))),
                    # a city-level search can land in the wrong country (Hamilton, Ontario for Bermuda): such an
                    # anchor is not evidence about the venue
                    "anchor": checks.anchor_check(g["lat"], g["lon"], g["anchor"])
                    if g["anchor"] and country and checks.in_territory(*g["anchor"], country, reference) else "unknown"}
        if g["lat"] is None:
            statuses = {k: "unknown" for k in statuses}
        row = {
            "venue_id": vid, "name": v["name"], "city": v["city"], "country": country, "home_team": x["home_team"],
            "lat": g["lat"], "lon": g["lon"], "elevation_m": elev, "elevation_source": elev_src, "elevation_dem_m": dem,
            "timezone": tz, "wikidata_qid": g["wikidata_qid"], "osm_ref": g["osm_ref"],
            "geocode_source": g["geocode_source"], "geocode_confidence": g["geocode_confidence"],
            "geocode_matched_name": g["geocode_matched_name"], "in_geo_plan": vid in planned, "geocode_gap": g["geocode_gap"],
            "country_check": statuses["country"], "country_evidence": c_evidence, "elevation_check": statuses["elevation"],
            "city_elevation_m": city_elev, "timezone_check": statuses["timezone"], "sea_check": statuses["sea"],
            "anchor_check": statuses["anchor"], "acknowledged_checks": ",".join(sorted(fix.get("acknowledge") or [])) or None,
            "note": fix.get("note"),
            "n_matches": len(span[vid]), "first_match_date": min(span[vid]), "last_match_date": max(span[vid]),
        }
        needs_review, reasons, failed = checks.assess(row, fix, statuses)
        row["needs_review"], row["review_reasons"] = needs_review, "; ".join(reasons) or None
        row["weather_location_quality"] = _location_quality(row)
        if failed and g["lat"] is not None:
            label = vid if len(versions[vid]) == 1 else f"{vid} (version {ver})"
            failures.append(f"{label}: {', '.join(failed)} check failed ({row['lat']:.4f}, {row['lon']:.4f}; "
                            f"country evidence: {c_evidence}; elevation {elev} vs city {city_elev}; tz {tz})")
        version_rows.append({**row, "venue_version": ver, "valid_from": x["valid_from"], "valid_to": x["valid_to"],
                             "inputs_as_of": x["inputs_as_of"],
                             "search_names": v["search_names"], "places": v["places"], "n_matches": x["n_matches"]})
        if x is versions[vid][-1]:
            rows.append(row)  # silver venues: the latest version, with counts over all the venue's matches
    if failures:
        raise SystemExit("venue sanity checks failed (fix the coordinates in config/venue_overrides.yml, or acknowledge "
                         "the check there with a note):\n  " + "\n  ".join(failures))

    by_key = {(r["venue_id"], r["venue_version"]): r for r in version_rows}
    daily, hourly, windows = _weather_rows(matches, alias_id, versions, by_key, series, wx, dates, _ball_times(con))

    match_rows = [{"match_id": m["match_id"], "venue_id": alias_id[(m["venue"], m["city"])],
                   "venue_version": asof.version_for(versions[alias_id[(m["venue"], m["city"])]], m["start_date"])["venue_version"],
                   "venue_era": era_for(alias_id[(m["venue"], m["city"])], m["start_date"], overrides)} for m in matches]
    _write(con, pa.Table.from_pylist(rows, schema=VENUE_SCHEMA), out_dir, "venues", "venue_id")
    _write(con, pa.Table.from_pylist(version_rows, schema=VERSION_SCHEMA), out_dir, "venue_versions",
           "venue_id, venue_version")
    _write(con, pa.Table.from_pylist(aliases, schema=pa.schema([("venue", pa.string()), ("city", pa.string()),
                                                                ("venue_id", pa.string()), ("n_matches", pa.int64())])),
           out_dir, "venue_aliases", "venue, city NULLS FIRST")
    _write(con, pa.Table.from_pylist(match_rows, schema=pa.schema([("match_id", pa.string()), ("venue_id", pa.string()),
                                                                   ("venue_version", pa.int16()), ("venue_era", pa.string())])),
           out_dir, "match_venues", "match_id")
    _write(con, pa.Table.from_pylist(daily, schema=DAILY_SCHEMA), derived_dir, "int_match_weather_daily",
           "match_id, match_day_number")
    _write(con, pa.Table.from_pylist(hourly, schema=HOURLY_SCHEMA), out_dir, "match_weather_hourly",
           "match_id, match_day_number, utc_datetime")
    _write(con, pa.Table.from_pylist(windows, schema=WINDOW_SCHEMA), derived_dir, "int_match_play_windows",
           "match_id, match_day_number")
    stats = {"venues": len(rows), "match_days": len(daily), "hourly_rows": len(hourly)}
    if asof.mode(overrides) == "asof":
        stats["venue_versions"] = len(version_rows)
    stats["weather_files"] = file_counts
    return stats


def _weather_rows(matches, alias_id, versions, venues, series, wx: store.Snapshot, dates, ball_times):
    last_available = date.fromisoformat(wx.plan["last_available"])
    failed_days = defaultdict(set)
    for f in wx.plan["files"]:
        if f["file"] in wx.failures:
            ids = {alias_id.get((v, c)) for v, c in f["aliases"]} - {None}
            for vid in ids:
                d, end = date.fromisoformat(f["start"]), date.fromisoformat(f["end"])
                while d <= end:
                    failed_days[vid].add(d)
                    d += timedelta(days=1)
    daily, hourly, windows = [], [], []
    for m in matches:
        vid = alias_id[(m["venue"], m["city"])]
        ver = asof.version_for(versions[vid], m["start_date"])["venue_version"]
        v = venues[(vid, ver)]
        tz = v["timezone"]
        klass = match_class(m["match_type"], m["event_name"])
        clusters = ball_clusters(ball_times.get(m["match_id"], []))
        s = series.get((vid, ver), {})
        grid = series_grid(s)
        for day in day_windows(dates[m["match_id"]], klass, tz, clusters):
            base = {"match_id": m["match_id"], "match_day_number": day["match_day_number"], "match_date": day["match_date"],
                    "venue_id": vid, "venue_version": ver, "timezone": tz, "window_source": day["window_source"],
                    "geocode_confidence": v["geocode_confidence"], "weather_location_quality": None}
            reason, agg = None, {}
            if v["lat"] is None:
                reason = "venue_not_geocoded"
            elif not tz:
                reason = "no_timezone (venue outside the weather plan)"
            else:
                agg = aggregate(s, day["window_start_utc"], day["window_end_utc"], grid)
                if agg["window_hours_present"] == 0:
                    if day["match_date"] in failed_days[vid]:
                        reason = "fetch_failed"
                    elif day["match_date"] > last_available:
                        reason = "after_archive_lag"
                    else:
                        reason = "not_in_weather_plan"
                    agg = {}
                elif agg["window_hours_present"] < agg["window_hours_expected"] or agg["window_precipitation_mm"] is None:
                    reason = "partial_hours"
            available = bool(agg)
            daily.append({**base, **agg, "weather_available": available, "weather_missing_reason": reason,
                          "weather_location_quality": v["weather_location_quality"] if available else None})
            if day["window_start_utc"] is not None:
                windows.append({"match_id": m["match_id"], "match_day_number": day["match_day_number"],
                                "match_date": day["match_date"], "venue_id": vid, "window_source": day["window_source"],
                                "window_start_utc": day["window_start_utc"], "window_end_utc": day["window_end_utc"],
                                "window_start_local": to_local(day["window_start_utc"], tz),
                                "window_end_local": to_local(day["window_end_utc"], tz),
                                "n_balls_timed": day["n_balls_timed"], "internal_note": INTERNAL_TIMING
                                if day["window_source"] == "ball_timestamps" else None})
            # This is the redistributable hourly product. Use a format-default window for every day, regardless
            # of whether internal aggregates use Ari/Cricinfo timestamps, so row times reveal no restricted timing.
            if tz and s:
                start_local, end_local = DEFAULT_WINDOWS[klass]
                public_start = local_to_utc(datetime.combine(day["match_date"], start_local), tz)
                public_end = local_to_utc(datetime.combine(day["match_date"], end_local), tz)
                inst, precip_h, pre_h = window_hours(public_start, public_end, grid)
                playing = set(inst) | set(precip_h)
                for t in sorted(set(pre_h) | playing):
                    if t in s:
                        hourly.append({"match_id": m["match_id"], "match_day_number": day["match_day_number"],
                                       "venue_id": vid, "venue_version": ver, "utc_datetime": t, "local_datetime": to_local(t, tz),
                                       "phase": "playing" if t in playing else "pre_match",
                                       **dict(zip(WX_COLS, s[t]))})
    return daily, hourly, windows


def build_venue_context(con) -> None:
    """Offline, deterministic. Needs the silver view `matches` (and uses `stg_ari_tracking` when present) plus locked,
    sealed snapshots of venues_geo and weather."""
    lock = bronze.read_lock()
    missing = [s for s in (GEO_SOURCE, WX_SOURCE, "cricsheet") if s not in lock]
    if missing:
        raise SystemExit(f"no locked snapshot for {missing}; run ingest_venues_and_weather() first")
    geo = store.Snapshot(GEO_SOURCE, lock[GEO_SOURCE])
    wx = store.Snapshot(WX_SOURCE, lock[WX_SOURCE])
    if wx.plan and wx.plan.get("geo_snapshot_id") != lock[GEO_SOURCE]:
        print(f"note: weather snapshot was planned from venues_geo/{wx.plan.get('geo_snapshot_id')}, "
              f"locked geo is {lock[GEO_SOURCE]}; every file's point is re-checked against current coordinates")
    from .checks import load_reference
    stats = _build(con, geo=geo, wx=wx, out_dir=SILVER, derived_dir=GOLD, overrides=load_overrides(), reference=load_reference(),
                   dates_loader=zip_dates_loader(snapshot_dir("cricsheet", lock["cricsheet"]) / "all_json.zip"))
    print(f"  venue context: {stats}")
