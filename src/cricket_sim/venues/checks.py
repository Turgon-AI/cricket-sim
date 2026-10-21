"""Post-geocode sanity checks. Every coordinate, whatever its source (manual, pinned OSM/Wikidata, automatic),
goes through the same checks; a failed check fails the build unless the venue override acknowledges it.

Checks (status pass / fail / unknown):
  country    the point lies in the bounding box of the venue's country. The expected country comes from, in order:
             the override, the home team (a team in >= 50% of the venue's international matches), the geocoder's
             own country label. Boxes: config/venue_reference.yml (deliberately a little padded).
  elevation  |venue elevation - city elevation| <= max(150 m, 15% of the city elevation). City elevations are a
             committed hand-made table in config/venue_reference.yml.
  timezone   the IANA zone is a real place zone (not Etc/GMT+-N, which Open-Meteo returns for open sea).
  sea        a 90 m DEM elevation of exactly 0 m means the point is probably in the sea (override `coastal: true`).
  anchor     distance to the city-level geocode, when that request is in the snapshot and the anchor lies in the
             venue's country, is <= 30 km.
An acknowledged failure (override `acknowledge: [check]` with a note) does not fail the build; acknowledged anchor
or sea failures no longer force review, acknowledged country/elevation failures always do.
"""

from __future__ import annotations

import math
from pathlib import Path

import yaml

from ..paths import CONFIG
from .canonical import norm

CHECKS = ("country", "elevation", "timezone", "sea", "anchor")
TRUSTED_CONFIDENCE = ("manual", "high", "medium")
ANCHOR_MAX_KM = 30.0
HOME_TEAM_MIN_SHARE = 0.5
HOME_TEAM_MIN_MATCHES = 3


def km(lat1, lon1, lat2, lon2) -> float:
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * math.asin(math.sqrt(min(1.0, a)))


def load_reference(path: Path | None = None) -> dict:
    ref = yaml.safe_load((path or CONFIG / "venue_reference.yml").read_text())
    boxes = {}
    for name, value in ref["territory_bounds"].items():
        boxes[name] = [value] if isinstance(value[0], (int, float)) else value
    ref["territory_bounds"] = boxes
    ref["city_elevation_m"] = {norm(k): float(v) for k, v in (ref.get("city_elevation_m") or {}).items()}
    ref["composite_teams"] = set(ref.get("composite_teams") or [])
    return ref


def territories_for_team(team: str, ref: dict) -> list[str]:
    if team in ref["composite_teams"]:
        return []
    mapped = (ref.get("team_territories") or {}).get(team)
    return list(mapped) if mapped else ([team] if team in ref["territory_bounds"] else [])


def territories_for_label(label: str | None, ref: dict) -> list[str]:
    if not label:
        return []
    mapped = (ref.get("label_aliases") or {}).get(label)
    if mapped:
        return list(mapped)
    return [label] if label in ref["territory_bounds"] else []


def in_territory(lat: float, lon: float, territory: str, ref: dict) -> bool:
    return any(s <= lat <= n and w <= lon <= e for s, n, w, e in ref["territory_bounds"].get(territory, []))


def home_team(team_counts: dict[str, int], n_international: int, ref: dict) -> str | None:
    """The team that plays in at least half of a venue's international matches (None at neutral venues)."""
    if n_international < HOME_TEAM_MIN_MATCHES:
        return None
    ranked = sorted(((n, t) for t, n in team_counts.items() if t not in ref["composite_teams"]), key=lambda x: (-x[0], x[1]))
    if ranked and ranked[0][0] >= HOME_TEAM_MIN_SHARE * n_international:
        if len(ranked) > 1 and ranked[1][0] == ranked[0][0]:
            return None
        return ranked[0][1]
    return None


def country_check(lat, lon, *, override: str | None, home: str | None, label: str | None, ref: dict) -> tuple[str, str | None, str]:
    """-> (status, country, evidence)."""
    if lat is None:
        return "unknown", override, "no coordinates"
    if override:
        if override not in ref["territory_bounds"]:
            return "fail", override, f"override country {override!r} has no bounding box in venue_reference.yml"
        ok = in_territory(lat, lon, override, ref)
        return ("pass" if ok else "fail"), override, f"override {override}" + ("" if ok else " (point outside its box)")
    evidence = [("home team", t) for t in territories_for_team(home, ref)] if home else []
    evidence += [("geocoder label", t) for t in territories_for_label(label, ref)]
    if not evidence:
        return "unknown", None, "no country evidence"
    for kind, territory in evidence:
        if in_territory(lat, lon, territory, ref):
            return "pass", territory, f"{kind} {territory}"
    return "fail", None, "point outside " + ", ".join(sorted({t for _, t in evidence}))


def elevation_check(elevation, city: str | None, ref: dict, city_elevation_override=None) -> tuple[str, float | None]:
    city_elev = city_elevation_override if city_elevation_override is not None else ref["city_elevation_m"].get(norm(city))
    if elevation is None or city_elev is None:
        return "unknown", city_elev
    tolerance = max(150.0, 0.15 * city_elev)
    return ("pass" if abs(elevation - city_elev) <= tolerance else "fail"), city_elev


def timezone_check(tz: str | None) -> str:
    if not tz:
        return "unknown"
    return "fail" if tz.startswith("Etc/") or tz in ("UTC", "GMT") else "pass"


def sea_check(dem_elevation, coastal: bool) -> str:
    if dem_elevation is None:
        return "unknown"
    return "pass" if dem_elevation != 0 or coastal else "fail"


def anchor_check(lat, lon, anchor) -> str:
    if lat is None or anchor is None:
        return "unknown"
    return "pass" if km(lat, lon, anchor[0], anchor[1]) <= ANCHOR_MAX_KM else "fail"


def assess(row: dict, fix: dict, statuses: dict) -> tuple[bool, list[str], list[str]]:
    """-> (needs_review, review_reasons, unacknowledged_failures)."""
    acknowledged = set(fix.get("acknowledge") or [])
    reasons, failures = [], []
    if row["geocode_confidence"] not in TRUSTED_CONFIDENCE:
        reasons.append(f"geocode_confidence={row['geocode_confidence']}")
    for name in CHECKS:
        status = statuses[name]
        if status == "fail":
            if name not in acknowledged:
                failures.append(name)
                reasons.append(f"{name}_check=fail")
            elif name in ("country", "elevation"):  # acknowledged, but a trusted row must still pass these
                reasons.append(f"{name}_check=fail (acknowledged)")
        elif status == "unknown" and name in ("country", "elevation", "timezone"):
            reasons.append(f"{name}_check=unknown")
    if fix.get("needs_review"):
        reasons.append("override: needs_review")
    needs_review = bool(reasons)
    # invariant: trusted rows are medium+ and pass country + elevation
    if not needs_review:
        assert row["geocode_confidence"] in TRUSTED_CONFIDENCE
        assert statuses["country"] == "pass" and statuses["elevation"] == "pass", row["venue_id"]
    return needs_review, reasons, failures
