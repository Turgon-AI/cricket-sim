"""Network half of the venue context: two bronze sources, ``venues_geo`` and ``weather``.

Planning reads only the locked bronze Cricsheet zip and committed config (never silver), so a fresh clone can
ingest before its first build. Each snapshot stores its request plan as plan.json (hashed into the manifest).

  geo      plan  = canonical venues of the selected matches (+ the override/reference file hashes)
           fetch = Wikidata bulk query, pinned Wikidata entities / OSM elements, Nominatim searches.
                   Cached answers are copied from the locked snapshot, so only requests the previous snapshot lacks go to the network.
  weather  refuses unless the geo snapshot it plans from is sealed and locked; records its id + manifest hash.
           plan  = for each venue: its coordinates and the match days needed; files of the locked weather snapshot
                   whose request point is within 2 km of the venue's current point are reused for the days they
                   cover, the rest is fetched.

`plan_selection` in config/context_sources.yml says which matches to fetch for. It is a network budget, not a
scope: silver is built over every match, and venues/days outside the plan just have null weather with a reason.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

from .. import bronze
from ..cricsheet_parse import iter_match_info
from ..paths import CONFIG, load_yaml, snapshot_dir
from . import asof
from . import geocode as gc
from . import store
from .canonical import canonicalize, load_overrides
from .checks import km
from .weather import WX_VARS, plan_windows, weather_file

GEO_SOURCE = "venues_geo"
WX_SOURCE = "weather"
WEATHER_REUSE_KM = 2.0  # same tolerance the build asserts


# ---------------------------------------------------------------- planning (offline)


def bronze_matches(zip_path: Path) -> list[dict]:
    out = []
    for match_id, info in iter_match_info(zip_path):
        teams = info.get("teams") or []
        out.append({"match_id": match_id, "venue": info.get("venue"), "city": info.get("city"),
                    "dates": sorted(date.fromisoformat(str(d)) for d in info["dates"]),
                    "match_type": info.get("match_type"), "gender": info.get("gender"),
                    "team_type": info.get("team_type"), "event_name": (info.get("event") or {}).get("name"),
                    "teams": list(teams)})
    return out


def selected(match: dict, selection: dict) -> bool:
    if selection.get("all"):
        return True
    if selection.get("gender") and match["gender"] != selection["gender"]:
        return False
    if selection.get("from_date") and match["dates"][0] < date.fromisoformat(str(selection["from_date"])):
        return False
    return any(all(match.get(k) == v for k, v in rule.items()) for rule in selection["any_of"])


def pairs_of(matches: list[dict]) -> list[tuple]:
    c = Counter((m["venue"], m["city"]) for m in matches)
    return sorted(((v, ct, n) for (v, ct), n in c.items()), key=lambda r: (r[0], r[1] or ""))


def file_sha(name: str) -> str:
    return bronze.sha256(CONFIG / name)


def geo_plan(matches: list[dict], selection: dict, overrides: dict, cricsheet_sid: str) -> tuple[dict, list, list]:
    venues, aliases = canonicalize(pairs_of(matches), overrides)
    alias_id = {(a["venue"], a["city"]): a["venue_id"] for a in aliases}
    n_sel = Counter(alias_id[(m["venue"], m["city"])] for m in matches if selected(m, selection))
    alias_lists = defaultdict(list)
    for a in aliases:
        alias_lists[a["venue_id"]].append([a["venue"], a["city"]])
    planned = [{"venue_id": v["venue_id"], "name": v["name"], "city": v["city"], "search_names": v["search_names"],
                "places": v["places"], "aliases": alias_lists[v["venue_id"]], "n_selected_matches": n_sel[v["venue_id"]]}
               for v in venues if n_sel[v["venue_id"]]]
    plan = {"kind": "venues_geo request plan", "cricsheet_snapshot": cricsheet_sid, "selection": selection,
            "config_sha256": {n: file_sha(n) for n in ("venue_overrides.yml", "venue_reference.yml")},
            "venues": planned}
    return plan, venues, aliases


# ---------------------------------------------------------------- HTTP


class Session:
    def __init__(self, user_agent: str):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = user_agent  # exactly the configured value
        self.last: dict[str, float] = {}

    def throttle(self, key: str, interval: float) -> None:
        wait = self.last.get(key, 0) + interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self.last[key] = time.monotonic()

    def get_json(self, url, params, key, interval, retries=4, mirrors=None, method="get"):
        last = None
        for attempt in range(retries):
            target = mirrors[attempt % len(mirrors)] if mirrors else url
            self.throttle(key, interval)
            try:
                if method == "post":
                    r = self.s.post(target, data=params, timeout=120)
                else:
                    r = self.s.get(target, params=params, timeout=120)
            except requests.RequestException as exc:
                last = exc
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                last = RuntimeError(f"HTTP {r.status_code} from {target}")
                time.sleep(15 * (attempt + 1))
                continue
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code} from {target}: {r.text[:200]}")
            try:
                return r.json()
            except ValueError:
                last = RuntimeError(f"non-JSON answer from {target}: {r.text[:100]!r}")
                time.sleep(15 * (attempt + 1))
        raise RuntimeError(f"giving up on {url or mirrors}: {last}")


# ---------------------------------------------------------------- geo


class _Reuse:
    """Responses of the currently locked snapshot of a source, addressable by request."""

    def __init__(self, source: str):
        self.snap = None
        sid = bronze.read_lock().get(source)
        if sid:
            self.snap = store.Snapshot(source, sid)

    def find(self, predicate) -> list[str]:
        if not self.snap:
            return []
        return [n for n, e in sorted(self.snap.entries.items()) if n.endswith(".zst") and predicate(n, e)]


class IngestProvider:
    """Answers geocode requests from the open snapshot, else the previous snapshot (copied in), else the network."""

    def __init__(self, writer: store.Writer, reuse: _Reuse, cfg: dict, dry_run: bool):
        self.w, self.reuse, self.cfg, self.dry_run = writer, reuse, cfg, dry_run
        self.sess = Session(cfg["user_agent"])
        self.counts = Counter()
        self.would_fetch: list[str] = []
        self._memo: dict[str, object] = {}
        self._prev_by_name = {}
        if reuse.snap:
            for name, entry in reuse.snap.entries.items():
                params = entry.get("params") or {}
                if name.startswith("nominatim/") and "q" in params:
                    self._prev_by_name[gc.nominatim_file(params["q"])] = name
                elif name.startswith(("wikidata_entities/", "osm_elements/")) or name == gc.WIKIDATA_FILE:
                    self._prev_by_name[name] = name

    def _cached(self, name: str):
        if name in self._memo:
            return True, self._memo[name]
        if self.w.has(name):
            self._memo[name] = self.w.response(name)
            return True, self._memo[name]
        prev = self._prev_by_name.get(name)
        if prev:
            request, response = self.reuse.snap.record(prev)
            if not self.dry_run:
                self.w.store(name, request, response, reused_from=f"{self.reuse.snap.snapshot_id}/{prev}")
            self.counts["reused"] += 1
            self._memo[name] = response
            return True, response
        return False, None

    def _get(self, name: str, url: str, params: dict, key: str, interval: float, **kw):
        hit, value = self._cached(name)
        if hit:
            return value
        request = {"url": url, "params": params}
        if self.dry_run:
            self.would_fetch.append(f"{key}: {params.get('q') or params.get('data', '')[:120] or url}")
            self._memo[name] = None
            return None
        try:
            res = self.sess.get_json(url, params, key, interval, **kw)
        except RuntimeError as exc:
            self.w.fail(name, request, str(exc))
            self._memo[name] = None
            return None
        self.w.store(name, request, res)
        self.counts[key] += 1
        self._memo[name] = res
        return res

    def nominatim(self, query: str):
        params = {"q": query, "format": "jsonv2", "limit": 3, "addressdetails": 1, "accept-language": "en"}
        return self._get(gc.nominatim_file(query), self.cfg["endpoints"]["nominatim"], params, "nominatim",
                         self.cfg["nominatim_min_interval_seconds"])

    def peek_nominatim(self, query: str):
        return self._cached(gc.nominatim_file(query))[1]

    def entity(self, qid: str):
        return self._get(gc.entity_file(qid), f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json", {},
                         "wikidata", 1.0)

    def osm(self, ref: str):
        params = {"osm_ids": gc.osm_lookup_id(ref), "format": "jsonv2", "addressdetails": 1, "accept-language": "en"}
        return self._get(gc.osm_file(ref), self.cfg["endpoints"]["nominatim_lookup"], params, "nominatim",
                         self.cfg["nominatim_min_interval_seconds"])

    def wikidata_bulk(self):
        return self._get(gc.WIKIDATA_FILE, self.cfg["endpoints"]["wikidata_sparql"],
                         {"query": gc.WIKIDATA_QUERY, "format": "json"}, "wikidata", 2.0)

def _snapshot_id(cfg: dict, plan: dict) -> str:
    sid = str(cfg.get("snapshot_id", "auto"))
    return store.new_snapshot_id(plan) if sid == "auto" else sid


def version_matches(matches: list[dict]) -> list[dict]:
    """Bronze matches in the shape asof.venue_versions reads (start date, the two teams)."""
    return [{"match_id": m.get("match_id"), "venue": m["venue"], "city": m["city"], "start_date": m["dates"][0],
             "gender": m["gender"], "team_type": m["team_type"], "team1": (m["teams"] + [None, None])[0], "team2": (m["teams"] + [None, None])[1]}
            for m in matches]


def ingest_geo(cfg: dict, plan: dict, venues: list[dict], overrides: dict, *, allow_gaps: bool, dry_run: bool,
               versions: list[dict] | None = None) -> str | None:
    """versions: every venue version (asof.venue_versions, flattened). With venue_context: asof the geo snapshot must
    answer every version's requests, so all of them are geocoded here; None geocodes the venues as before."""
    from .build import geocode_venues, geocode_versions  # shared with the build so both make the same requests

    sid = _snapshot_id(cfg, plan)
    writer = store.Writer(GEO_SOURCE, sid)
    if writer.sealed:
        print(f"{GEO_SOURCE}/{sid} already sealed for this plan")
        return sid
    if not dry_run:
        writer.open().write_plan(plan)
    provider = IngestProvider(writer, _Reuse(GEO_SOURCE), cfg, dry_run)
    bulk = provider.wikidata_bulk()
    cands = gc.parse_wikidata(bulk) if bulk else []
    planned = {v["venue_id"] for v in plan["venues"]}
    if versions is None:
        geo = geocode_venues(venues, planned, overrides, provider, cands)
    else:
        geo = {(x["venue_id"], x["venue_version"]): g
               for x, g in zip(versions, geocode_versions(versions, planned, overrides, provider, cands))}
    print("geocoded (planned venues):", dict(Counter((g["geocode_source"], g["geocode_confidence"])
                                                     for key, g in geo.items()
                                                     if (key[0] if isinstance(key, tuple) else key) in planned)))

    print("api calls this run:", dict(provider.counts))
    if dry_run:
        print(f"[dry run] {GEO_SOURCE}/{sid}: would fetch {len(provider.would_fetch)} request(s):")
        for line in provider.would_fetch:
            print("   ", line)
        return None
    writer.finalize({"api_calls_this_run": dict(provider.counts), "plan_sha256": store.sha256_bytes(store.canonical_json(plan))},
                    cfg, allow_gaps=allow_gaps)
    bronze.lock_source(GEO_SOURCE, sid)
    print(f"sealed and locked {GEO_SOURCE} -> {sid}")
    return sid


# ---------------------------------------------------------------- weather


def weather_plan(cfg: dict, matches: list[dict], selection: dict, aliases: list[dict], geo_points: dict,
                 geo_sid: str, geo_manifest_sha: str, cricsheet_sid: str, reuse: _Reuse, today: date,
                 versions: dict | None = None) -> tuple[dict, list]:
    """-> (plan, [(file, prev_name)] reused). geo_points: venue_id -> (lat, lon) for venues with coordinates; or, with
    `versions` (asof.venue_versions), (venue_id, venue_version) -> (lat, lon): each version's match days are planned
    at that version's point."""
    alias_id = {(a["venue"], a["city"]): a["venue_id"] for a in aliases}
    alias_lists = defaultdict(list)
    for a in aliases:
        alias_lists[a["venue_id"]].append([a["venue"], a["city"]])
    points = geo_points if versions is not None else {(vid, 1): pt for vid, pt in geo_points.items()}
    days = defaultdict(set)
    for m in matches:
        vid = alias_id[(m["venue"], m["city"])]
        ver = asof.version_for(versions[vid], m["dates"][0])["venue_version"] if versions is not None else 1
        if selected(m, selection) and (vid, ver) in points:
            days[(vid, ver)].update(m["dates"])
    last_available = today - timedelta(days=cfg["archive_lag_days"])

    prev_files = []  # (name, lat, lon, start, end)
    if reuse.snap:
        for name, e in sorted(reuse.snap.entries.items()):
            p = e.get("params") or {}
            if name.endswith(".json.zst") and "latitude" in p:
                prev_files.append((name, float(p["latitude"]), float(p["longitude"]),
                                   date.fromisoformat(p["start_date"]), date.fromisoformat(p["end_date"])))
    files, reused = [], []
    for vid, ver in sorted(days):
        lat, lon = points[(vid, ver)]
        tag = {} if ver == 1 else {"venue_version": ver}
        stem = vid if ver == 1 else f"{vid}--v{ver}"
        near = [f for f in prev_files if km(lat, lon, f[1], f[2]) <= WEATHER_REUSE_KM]
        covered = set()
        for name, flat, flon, s, e in near:
            hit = {d for d in days[(vid, ver)] if s <= d <= e}
            if hit - covered:
                covered |= hit
                new_name = weather_file(stem, s, e)
                files.append({"file": new_name, "venue_id": vid, **tag, "aliases": alias_lists[vid],
                              "lat": flat, "lon": flon, "start": s.isoformat(), "end": e.isoformat(),
                              "reused_from": f"{reuse.snap.snapshot_id}/{name}"})
                reused.append((new_name, name))
        missing = {vid: days[(vid, ver)] - covered}
        windows, deferred = plan_windows(missing, cfg["max_window_days"], last_available)
        for _, s, e in windows:
            # one extra day each side: the 12 h before a morning start and the hour after a late finish
            s2, e2 = s - timedelta(days=1), min(e + timedelta(days=1), last_available)
            files.append({"file": weather_file(stem, s2, e2), "venue_id": vid, **tag, "aliases": alias_lists[vid],
                          "lat": round(lat, 5), "lon": round(lon, 5), "start": s2.isoformat(), "end": e2.isoformat()})
        files.extend({"deferred_day": d.isoformat(), "venue_id": vid, **tag, "aliases": alias_lists[vid]} for _, d in deferred)
    plan = {"kind": "weather request plan", "geo_snapshot_id": geo_sid, "geo_manifest_sha256": geo_manifest_sha,
            "cricsheet_snapshot": cricsheet_sid, "selection": selection, "last_available": last_available.isoformat(),
            "window_days": cfg["max_window_days"], "hourly": cfg["hourly"],
            "files": [f for f in files if "file" in f], "deferred": [f for f in files if "deferred_day" in f]}
    return plan, reused


def ingest_weather(cfg: dict, plan: dict, reused: list, *, allow_gaps: bool, dry_run: bool) -> str | None:
    sid = _snapshot_id(cfg, plan)
    writer = store.Writer(WX_SOURCE, sid)
    if writer.sealed:
        print(f"{WX_SOURCE}/{sid} already sealed for this plan")
        return sid
    reuse = _Reuse(WX_SOURCE)
    to_fetch = [f for f in plan["files"] if "reused_from" not in f]
    print(f"weather plan: {len(plan['files'])} files ({len(reused)} reused, {len(to_fetch)} to fetch), "
          f"{len(plan['deferred'])} day(s) deferred by the archive lag")
    if dry_run:
        for f in to_fetch:
            print(f"    open-meteo: {f['venue_id']} {f['start']}..{f['end']} at {f['lat']},{f['lon']}")
        return None
    writer.open().write_plan(plan)
    for new_name, prev in reused:
        if not writer.has(new_name):
            request, response = reuse.snap.record(prev)
            writer.store(new_name, request, response, reused_from=f"{reuse.snap.snapshot_id}/{prev}")
    sess = Session(cfg["user_agent"])
    calls = 0.0
    for f in to_fetch:
        if writer.has(f["file"]):
            continue
        weight = max(1.0, ((date.fromisoformat(f["end"]) - date.fromisoformat(f["start"])).days + 1) / cfg["max_window_days"])
        if calls + weight > cfg["max_weighted_calls_per_run"]:
            raise SystemExit("call budget for this run reached; re-run later to resume (cache is kept)")
        params = {"latitude": f["lat"], "longitude": f["lon"], "start_date": f["start"], "end_date": f["end"],
                  "hourly": ",".join(WX_VARS), "timezone": "auto"}
        request = {"url": cfg["endpoint"], "params": params}
        try:
            res = sess.get_json(cfg["endpoint"], params, "openmeteo", cfg["min_interval_seconds"], retries=5)
        except RuntimeError as exc:
            writer.fail(f["file"], request, str(exc))
            continue
        if res.get("error"):
            writer.fail(f["file"], request, str(res.get("reason")))
            continue
        writer.store(f["file"], request, res)
        calls += weight
    print(f"open-meteo weighted calls this run: {calls:.1f}")
    writer.finalize({"geo_snapshot_id": plan["geo_snapshot_id"], "geo_manifest_sha256": plan["geo_manifest_sha256"],
                     "plan_sha256": store.sha256_bytes(store.canonical_json(plan)), "weighted_calls_this_run": round(calls, 2)},
                    cfg, allow_gaps=allow_gaps)
    bronze.lock_source(WX_SOURCE, sid)
    print(f"sealed and locked {WX_SOURCE} -> {sid}")
    return sid


# ---------------------------------------------------------------- entry point


def ingest_venues_and_weather(allow_gaps: bool = False, dry_run: bool = False, stage: str = "all",
                              today: date | None = None) -> None:
    """Fetch what the current plan needs that the locked snapshots do not already hold, seal, and lock.
    stage: 'geo' stops after the geo snapshot; 'weather' plans weather from the already-locked geo snapshot."""
    from .build import BuildProvider, geocode_versions
    from .checks import load_reference

    cfg = load_yaml("context_sources.yml")
    lock = bronze.read_lock()
    if "cricsheet" not in lock:
        raise SystemExit("no locked cricsheet snapshot; run `cricket ingest cricsheet` first")
    zip_path = snapshot_dir("cricsheet", lock["cricsheet"]) / "all_json.zip"
    matches = bronze_matches(zip_path)
    overrides = load_overrides()
    selection = cfg["plan_selection"]
    plan, venues, aliases = geo_plan(matches, selection, overrides, lock["cricsheet"])
    print(f"{len(matches)} bronze matches, {len(venues)} canonical venues, {len(plan['venues'])} in the geo plan")
    alias_id = {(a["venue"], a["city"]): a["venue_id"] for a in aliases}
    versions = asof.venue_versions(version_matches(matches), venues, alias_id, overrides, load_reference())
    flat = [x for vid in sorted(versions) for x in versions[vid]]
    print(f"venue_context: {asof.mode(overrides)}, {len(flat)} venue versions")

    geo_cfg = {**cfg["venues_geo"], "user_agent": cfg["user_agent"]}
    if stage in ("all", "geo"):
        geo_sid = ingest_geo(geo_cfg, plan, venues, overrides, allow_gaps=allow_gaps, dry_run=dry_run,
                             versions=flat if asof.mode(overrides) == "asof" else None)
        if stage == "geo" or geo_sid is None:
            return
    lock = bronze.read_lock()
    geo_sid = lock.get(GEO_SOURCE)
    if not geo_sid or geo_sid not in bronze.list_snapshots(GEO_SOURCE):
        raise SystemExit("weather ingest needs a sealed, locked venues_geo snapshot first")
    geo = store.Snapshot(GEO_SOURCE, geo_sid)
    if geo.plan is None:
        raise SystemExit(f"{GEO_SOURCE}/{geo_sid} has no plan.json (made by an older ingest); re-run the geo stage")
    planned = {v["venue_id"] for v in geo.plan["venues"]}
    cands = gc.parse_wikidata(geo.response(gc.WIKIDATA_FILE))
    coords = geocode_versions(flat, planned, overrides, BuildProvider(geo), cands)
    points = {(x["venue_id"], x["venue_version"]): (g["lat"], g["lon"]) for x, g in zip(flat, coords)
              if g["lat"] is not None and x["venue_id"] in planned}
    wx_cfg = {**cfg["weather"], "user_agent": cfg["user_agent"]}
    wplan, reused = weather_plan(wx_cfg, matches, selection, aliases, points, geo_sid, geo.manifest_sha256,
                                 lock["cricsheet"], _Reuse(WX_SOURCE), today or datetime.now(timezone.utc).date(),
                                 versions=versions)
    ingest_weather(wx_cfg, wplan, reused, allow_gaps=allow_gaps, dry_run=dry_run)
