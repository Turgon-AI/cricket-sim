"""Geocoding: manual coordinates, pinned OSM elements, pinned Wikidata items, then the automatic Wikidata match and
OSM Nominatim fallback. All decisions are pure functions over cached responses, so ingest (which fetches whatever
these functions ask for) and build (which only reads the snapshot) always agree.

Requests go through a provider with four methods; it returns the parsed response, or None when the request is
recorded as failed in the snapshot (or the venue is not in the geo plan, so no network source is consulted).
At build time a request that is neither stored nor recorded as failed raises store.MissingRequest.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from .canonical import norm, sig_tokens
from .checks import in_territory, km, load_reference

WIKIDATA_FILE = "wikidata_cricket_grounds.json.zst"
WIKIDATA_QUERY = """
SELECT ?item ?itemLabel ?c ?elev ?locLabel ?countryLabel (GROUP_CONCAT(DISTINCT ?alt; separator="|") AS ?alts) WHERE {
  { ?item wdt:P31/wdt:P279* wd:Q682943 } UNION { ?item wdt:P641 wd:Q5375 }
  ?item wdt:P625 ?c .
  OPTIONAL { ?item wdt:P2044 ?elev }
  OPTIONAL { ?item wdt:P131 ?loc }
  OPTIONAL { ?item wdt:P17 ?country }
  OPTIONAL { ?item skos:altLabel ?alt FILTER(LANG(?alt)="en") }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
} GROUP BY ?item ?itemLabel ?c ?elev ?locLabel ?countryLabel
"""
NOMINATIM_ANCHOR_RADIUS_KM = 80.0


class GeocodeError(ValueError):
    """An override that cannot be honoured (e.g. a pinned QID without coordinates). Never silently skipped."""


def query_key(query: str) -> str:
    return hashlib.sha1(query.encode()).hexdigest()[:16]


def nominatim_file(query: str) -> str:
    return f"nominatim/{query_key(query)}.json.zst"


def entity_file(qid: str) -> str:
    return f"wikidata_entities/{qid}.json.zst"


def osm_file(ref: str) -> str:
    return f"osm_elements/{ref.replace('/', '_')}.json.zst"


OSM_TYPE_LETTER = {"node": "N", "way": "W", "relation": "R"}


def osm_lookup_id(ref: str) -> str:
    """'way/123' -> 'W123' (Nominatim /lookup syntax)."""
    kind, _, osm_id = ref.partition("/")
    if kind not in OSM_TYPE_LETTER or not osm_id.isdigit():
        raise GeocodeError(f"bad OSM reference {ref!r}; expected way/123, node/123 or relation/123")
    return OSM_TYPE_LETTER[kind] + osm_id


def parse_osm_lookup(resp, ref: str) -> dict | None:
    """One pinned OSM element from a Nominatim /lookup answer: its point (Nominatim's centroid), name, country."""
    kind, _, osm_id = ref.partition("/")
    for hit in resp if isinstance(resp, list) else []:
        if hit.get("osm_type") == kind and str(hit.get("osm_id")) == osm_id and hit.get("lat") is not None:
            return {"lat": float(hit["lat"]), "lon": float(hit["lon"]), "name": hit.get("name") or hit.get("display_name"),
                    "country": (hit.get("address") or {}).get("country")}
    return None


def _is_num(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_wikidata(obj: dict) -> list[dict]:
    """SPARQL JSON -> one candidate per QID (several coordinate/location rows are merged deterministically)."""
    by_qid: dict[str, list] = defaultdict(list)
    for b in obj["results"]["bindings"]:
        m = re.match(r"Point\((-?[\d.eE+-]+) (-?[\d.eE+-]+)\)", b["c"]["value"])
        if not m:
            continue
        qid = b["item"]["value"].rsplit("/", 1)[-1]
        by_qid[qid].append({
            "lon": float(m.group(1)), "lat": float(m.group(2)),
            "label": b.get("itemLabel", {}).get("value", ""),
            "elev": float(b["elev"]["value"]) if "elev" in b and _is_num(b["elev"]["value"]) else None,
            "loc": b.get("locLabel", {}).get("value", ""), "country": b.get("countryLabel", {}).get("value", ""),
            "alts": [a for a in b.get("alts", {}).get("value", "").split("|") if a],
        })
    cands = []
    for qid in sorted(by_qid):
        rows = sorted(by_qid[qid], key=lambda r: (r["lat"], r["lon"], r["loc"], r["country"]))
        first = rows[0]
        elevs = sorted({r["elev"] for r in rows if r["elev"] is not None})
        countries = sorted({r["country"] for r in rows if r["country"]})
        cands.append({
            "qid": qid, "label": first["label"], "lat": first["lat"], "lon": first["lon"],
            "n_coords": len({(r["lat"], r["lon"]) for r in rows}),
            # several elevations or countries on one item: keep none rather than pick one
            "elev": elevs[0] if len(elevs) == 1 else None,
            "locs": sorted({r["loc"] for r in rows if r["loc"]}), "country": countries[0] if len(countries) == 1 else "",
            "names": sorted({first["label"], *[a for r in rows for a in r["alts"]]}),
        })
    return cands


def parse_entity(obj: dict, qid: str) -> dict | None:
    """A single Wikidata entity (Special:EntityData JSON) as a candidate, or None if it has no coordinates."""
    ents = obj.get("entities") or {}
    ent = ents.get(qid) or (next(iter(ents.values())) if len(ents) == 1 else {})  # a redirect answers with the target id
    claims = ent.get("claims") or {}
    try:
        val = claims["P625"][0]["mainsnak"]["datavalue"]["value"]
    except (KeyError, IndexError):
        return None
    elev = None
    try:
        elev = float(claims["P2044"][0]["mainsnak"]["datavalue"]["value"]["amount"])
    except (KeyError, IndexError, ValueError):
        pass
    label = (ent.get("labels") or {}).get("en", {}).get("value", qid)
    return {"qid": qid, "label": label, "lat": float(val["latitude"]), "lon": float(val["longitude"]), "elev": elev,
            "locs": [], "country": "", "names": [label], "n_coords": 1}


_TOKEN_FIXES = {"st": "saint", "mt": "mount", "chattogram": "chittagong", "bengaluru": "bangalore", "mumbai": "bombay"}


def _tokens(text: str) -> set:
    return {_TOKEN_FIXES.get(t, t) for t in norm(text).split()}


def _place_ok(places: list[str], cand: dict) -> bool:
    text = _tokens(" ".join([*cand["locs"], cand["country"], *cand["names"]]))
    return any(p_t and p_t <= text for p_t in (_tokens(p) for p in places))


def match_wikidata(venue: dict, cands: list[dict]) -> dict | None:
    """Best Wikidata item for a canonical venue, or None. Returns {cand, confidence, rule}."""
    v_norm = {norm(n) for n in venue["search_names"]}
    v_sig = [sig_tokens(n) for n in venue["search_names"]]
    scored = []
    for cand in cands:
        names = cand["names"]
        c_norm = {norm(n) for n in names}
        c_sig = [sig_tokens(n) for n in names]
        place = _place_ok(venue["places"], cand)
        if v_norm & c_norm:
            rule, score = "exact_name", 3
        elif any(a and a == b for a in v_sig for b in c_sig):
            rule, score = "same_key_words", 2
        elif place and any(a and b and (a <= b or b <= a) for a in v_sig for b in c_sig):
            rule, score = "partial_name", 1
        else:
            continue
        scored.append((score, place, cand, rule))
    if not scored:
        return None
    best = max(s[0] * 2 + (1 if s[1] else 0) for s in scored)
    top = [s for s in scored if s[0] * 2 + (1 if s[1] else 0) == best]
    score, place, cand, rule = sorted(top, key=lambda s: s[2]["qid"])[0]
    if any(km(cand["lat"], cand["lon"], s[2]["lat"], s[2]["lon"]) > 5 for s in top):
        return None  # ambiguous: same quality of match but different places
    if place:
        confidence = "high" if score >= 2 else "medium"
        if score == 1 and max(len(t) for t in v_sig) < 2:
            return None
    elif score >= 2:
        confidence = "unverified"  # name agrees, place unknown: corroborate with Nominatim before trusting
    else:
        return None
    return {"cand": cand, "confidence": confidence, "rule": rule, "place_ok": place}


def nominatim_queries(venue: dict, fix: dict) -> list[str]:
    """Search strings, best first; the last one is city-level (the anchor, and a rough last resort)."""
    city = venue.get("city")
    hint = f", {fix['country_hint']}" if fix.get("country_hint") else ""
    if fix.get("nominatim_query"):
        queries = [fix["nominatim_query"]]
    else:
        names, seen = [], set()
        for n in [venue["name"], *venue["search_names"]]:
            n = re.sub(r"^The ", "", n)
            if norm(n) not in seen:
                seen.add(norm(n))
                names.append(n)
        names = names[:3]
        queries = [f"{n}, {city}{hint}" for n in names] if city else []
        queries += [f"{n}{hint}" for n in names]
    if city:
        queries.append(f"{city}{hint}")
    return queries


def anchor_query(venue: dict, fix: dict) -> str | None:
    return nominatim_queries(venue, fix)[-1] if venue.get("city") else None


def _nom(top: dict, confidence: str, query: str, city_level: bool = False) -> dict:
    return {
        "lat": float(top["lat"]), "lon": float(top["lon"]), "confidence": confidence, "query": query,
        "display_name": top.get("display_name"), "country": (top.get("address") or {}).get("country"),
        "osm_ref": f"{top.get('osm_type')}/{top.get('osm_id')}", "city_level": city_level,
    }


def pick_nominatim(venue: dict, results: list, queries: list[str]) -> dict | None:
    want = sig_tokens(venue["name"])
    city_idx = len(queries) - 1 if venue.get("city") else None
    first_low = None
    for i, res in enumerate(results):
        if not res or i == city_idx:
            continue
        top = res[0]
        have = _tokens(top.get("display_name", ""))
        if (want and want & have) or (not want and _tokens(venue["name"]) & have):
            return _nom(top, "medium", queries[i])
        first_low = first_low or _nom(top, "low", queries[i])
    if city_idx is not None and results[city_idx]:
        return _nom(results[city_idx][0], "low", queries[city_idx], city_level=True)
    return first_low


def run_nominatim(venue: dict, fix: dict, provider) -> tuple[dict | None, tuple | None]:
    """-> (pick, city anchor). Name hits are only trusted near the city (guards against 'Hamilton, Ontario')."""
    queries = nominatim_queries(venue, fix)
    city_idx = len(queries) - 1 if venue.get("city") else None
    want = sig_tokens(venue["name"]) or _tokens(venue["name"])
    anchor = None
    results = [None] * len(queries)
    if city_idx is not None:
        results[city_idx] = provider.nominatim(queries[city_idx])
        if results[city_idx]:
            anchor = (float(results[city_idx][0]["lat"]), float(results[city_idx][0]["lon"]))
    for i, q in enumerate(queries):
        if i == city_idx:
            continue
        res = provider.nominatim(q)
        if res and anchor:
            res = [r for r in res if km(anchor[0], anchor[1], float(r["lat"]), float(r["lon"])) <= NOMINATIM_ANCHOR_RADIUS_KM]
        results[i] = res
        if res and want & _tokens(res[0].get("display_name", "")):
            break  # a name-matching hit; skip the remaining (weaker) queries
    return pick_nominatim(venue, results, queries), anchor


_REFERENCE: dict = {}


def _outside_country(lat: float, lon: float, country: str | None, ref: dict | None) -> bool:
    """True when the override names an expected country and the point lies outside its box."""
    if not country:
        return False
    if ref is None:
        ref = _REFERENCE.setdefault("ref", load_reference())
    return not in_territory(lat, lon, country, ref)


def resolve_geocode(venue: dict, cands: list[dict], provider, fix: dict, searchable: bool = True,
                    ref: dict | None = None) -> dict:
    """Coordinates for one venue. `searchable` is False for venues outside the geo plan: then only the bulk Wikidata
    file (already in the snapshot) is used, never Nominatim or entity requests.
    When the override names the venue's `country`, an automatic (Wikidata or Nominatim) point outside that country
    is rejected, never kept (York CC matched York Park, Tasmania); geocode_gap says what was rejected.
    Manual and pinned points are not filtered: they go to the country check, which fails the build."""
    base = {"lat": None, "lon": None, "declared_elevation_m": None, "wikidata_qid": None, "osm_ref": None,
            "geocode_source": None, "geocode_confidence": None, "geocode_matched_name": None, "geocoder_country": None,
            "anchor": None, "geocode_gap": None}
    anchor = None
    if searchable and venue.get("city"):
        anchor_res = provider.peek_nominatim(anchor_query(venue, fix))
        anchor = (float(anchor_res[0]["lat"]), float(anchor_res[0]["lon"])) if anchor_res else None
    base["anchor"] = anchor

    if fix.get("lat") is not None and fix.get("lon") is not None:
        return {**base, "lat": float(fix["lat"]), "lon": float(fix["lon"]), "wikidata_qid": fix.get("wikidata_qid"),
                "osm_ref": fix.get("osm_ref"), "geocode_source": "manual", "geocode_confidence": "manual",
                "geocode_matched_name": fix.get("source") or fix.get("note")}
    if fix.get("osm_ref"):
        obj = provider.osm(fix["osm_ref"])
        if obj is None:
            return {**base, "geocode_gap": f"pinned OSM element {fix['osm_ref']} failed to fetch"}
        pt = parse_osm_lookup(obj, fix["osm_ref"])
        if pt is None:
            raise GeocodeError(f"{venue['venue_id']}: pinned OSM element {fix['osm_ref']} was not found by Nominatim")
        return {**base, "lat": pt["lat"], "lon": pt["lon"], "osm_ref": fix["osm_ref"], "wikidata_qid": fix.get("wikidata_qid"),
                "geocode_source": "osm_pinned", "geocode_confidence": "high", "geocoder_country": pt["country"],
                "geocode_matched_name": f"{pt['name'] or ''} ({fix['osm_ref']})".strip()}
    if fix.get("wikidata_qid"):
        qid = fix["wikidata_qid"]
        cand = next((c for c in cands if c["qid"] == qid), None)
        if cand is None:
            obj = provider.entity(qid)
            if obj is None:
                return {**base, "geocode_gap": f"pinned Wikidata item {qid} failed to fetch"}
            cand = parse_entity(obj, qid)
        if cand is None:
            raise GeocodeError(f"{venue['venue_id']}: pinned Wikidata item {qid} has no coordinates (P625); "
                               "give lat/lon or osm_ref in venue_overrides.yml instead")
        return {**base, "lat": cand["lat"], "lon": cand["lon"], "declared_elevation_m": cand["elev"],
                "wikidata_qid": qid, "geocode_source": "wikidata_pinned", "geocode_confidence": "high",
                "geocode_matched_name": cand["label"], "geocoder_country": cand["country"] or None}

    rejected = []
    hit = None if fix.get("skip_wikidata") else match_wikidata(venue, cands)
    if hit and _outside_country(hit["cand"]["lat"], hit["cand"]["lon"], fix.get("country"), ref):
        rejected.append(f"Wikidata {hit['cand']['label']} ({hit['cand']['qid']})")
        hit = None
    if hit and hit["confidence"] != "unverified":
        return _from_wikidata(base, hit, hit["confidence"], hit["rule"])
    if not searchable:
        return _rejected(base, rejected, fix)
    pick, anchor2 = run_nominatim(venue, fix, provider)
    base["anchor"] = base["anchor"] or anchor2
    if pick and _outside_country(pick["lat"], pick["lon"], fix.get("country"), ref):
        rejected.append(f"Nominatim {pick['display_name']}")
        pick = None
    if hit:
        c = hit["cand"]
        if pick and km(c["lat"], c["lon"], pick["lat"], pick["lon"]) <= 30:
            return _from_wikidata(base, hit, "high", hit["rule"] + ", corroborated by nominatim")
        if not pick and len(sig_tokens(venue["name"])) >= 2:
            return _from_wikidata(base, hit, "medium", hit["rule"] + ", place unverified")
    if pick:
        return {**base, "lat": pick["lat"], "lon": pick["lon"],
                "geocode_source": "nominatim_city" if pick["city_level"] else "nominatim",
                "geocode_confidence": pick["confidence"], "geocode_matched_name": pick["display_name"],
                "geocoder_country": pick["country"]}
    return _rejected(base, rejected, fix)


def _rejected(base: dict, rejected: list[str], fix: dict) -> dict:
    if not rejected:
        return base
    return {**base, "geocode_gap": f"rejected outside {fix['country']}: " + "; ".join(rejected)}


def _from_wikidata(base: dict, hit: dict, confidence: str, rule: str) -> dict:
    c = hit["cand"]
    return {**base, "lat": c["lat"], "lon": c["lon"], "declared_elevation_m": c["elev"], "wikidata_qid": c["qid"],
            "geocode_source": "wikidata", "geocode_confidence": confidence,
            "geocode_matched_name": f"{c['label']} ({rule})", "geocoder_country": c["country"] or None}
