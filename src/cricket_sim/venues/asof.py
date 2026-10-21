"""Venue versions: what the venue context knows about a ground on a match date.

The geocoder inputs of a venue (display name, city, search names, places) and its home team are built from its
matches. Built from EVERY match in the snapshot (venue_context: snapshot, the default) a later spelling can move the
point, and so the weather, of a 2005 match. With venue_context: asof, a venue gets a new version on the first date its
inputs change, built only from the matches up to and including that date; each match uses the version valid on its
start date, and same-day matches share one version. venue_id never changes (its rule is name-only).

Both modes return the same shape, so the build and the ingest plan treat them alike: in snapshot mode every venue has
exactly one version (version 1, open-ended), equal to the old snapshot-wide venue.
"""

from __future__ import annotations

import bisect
from collections import Counter, defaultdict
from datetime import date, timedelta

from . import checks
from .canonical import apply_display_fix, present, primary_names


def mode(overrides: dict) -> str:
    return overrides.get("venue_context") or "snapshot"


def _inputs(venue_id: str, core: str, pair_n: Counter, teams: Counter, n_int: int, primary: dict, overrides: dict,
            ref: dict) -> dict:
    members = sorted(((v, c, n) for (v, c), n in pair_n.items() if n), key=lambda r: (r[0], r[1] or ""))
    venue = apply_display_fix({"venue_id": venue_id, **present(core, members, primary), "core": core}, overrides)
    return {"venue": venue, "home_team": checks.home_team(teams, n_int, ref) if n_int else None}


def _key(inputs: dict) -> tuple:
    v = inputs["venue"]
    return (v["name"], v["city"], tuple(v["search_names"]), tuple(v["places"]), inputs["home_team"])


def venue_versions(matches: list[dict], venues: list[dict], alias_id: dict, overrides: dict, ref: dict) -> dict[str, list[dict]]:
    """venue_id -> versions sorted by date. Each version: venue_id, venue_version (1, 2, ...), valid_from, valid_to
    (inclusive; None for the last), inputs_as_of, venue (the geocoder's input dict), home_team, n_matches (in the
    version).
    matches need venue, city, start_date, team_type, team1, team2."""
    by_venue = defaultdict(list)
    for m in matches:
        by_venue[alias_id[(m["venue"], m["city"])]].append(m)
    primary = primary_names(overrides)
    asof = mode(overrides) == "asof"
    out = {}
    for v in venues:
        vid, ms = v["venue_id"], sorted(by_venue.get(v["venue_id"], []), key=lambda m: (m["start_date"], m["match_id"]))
        if not ms:
            continue
        pair_n, teams, n_int = Counter(), Counter(), 0
        days = defaultdict(list)
        for m in ms:
            days[m["start_date"]].append(m)
        versions = []
        for day in (sorted(days) if asof else [None]):
            for m in (days[day] if asof else ms):
                pair_n[(m["venue"], m["city"])] += 1
                if m["team_type"] == "international":
                    n_int += 1
                    for t in {m["team1"], m["team2"]} - {None}:
                        teams[t] += 1
            inputs = _inputs(vid, v["core"], pair_n, teams, n_int, primary, overrides, ref)
            if versions and _key(versions[-1]) == _key(inputs):
                continue
            versions.append({"venue_id": vid, "venue_version": len(versions) + 1,
                             "valid_from": day if asof else ms[0]["start_date"], "valid_to": None,
                             # the last match date whose matches built these inputs: = valid_from when as-of
                             "inputs_as_of": day if asof else ms[-1]["start_date"], **inputs})
        for prev, nxt in zip(versions, versions[1:]):
            prev["valid_to"] = nxt["valid_from"] - timedelta(days=1)
        starts = [x["valid_from"] for x in versions]
        n = Counter(bisect.bisect_right(starts, m["start_date"]) for m in ms)
        for i, x in enumerate(versions):
            x["n_matches"] = n[i + 1] + (n[0] if i == 0 else 0)
        out[vid] = versions
    return out


def version_for(versions: list[dict], day: date) -> dict:
    """The version valid on `day` (the first version also covers any earlier day)."""
    i = bisect.bisect_right([v["valid_from"] for v in versions], day)
    return versions[max(i, 1) - 1]


def overlaps(versions: list[dict], v: dict, start: date, end: date) -> bool:
    """Does version v cover any day of [start, end]? The first version is open at the start, the last at the end."""
    lo = None if v is versions[0] else v["valid_from"]
    hi = v["valid_to"]
    return (lo is None or end >= lo) and (hi is None or start <= hi)
