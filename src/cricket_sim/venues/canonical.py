"""Canonical venues: which Cricsheet (venue, city) strings are the same ground.

The rule is scope-free and count-free for identity: a venue_id is the slug of the ground's normalised core name
(the text before the first comma), after manual merges/assignments, plus a city suffix for core names listed in
``split_by_city``. It never depends on config/scope.yml or on how many matches a spelling has, so widening the
scope or adding matches cannot rename a venue. Match counts only choose the *display* name/city.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from ..paths import CONFIG


def norm(text: str | None) -> str:
    """Lower-case ASCII with single spaces: 'M.Chinnaswamy Stadium' -> 'm chinnaswamy stadium'."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    t = t.lower().replace("&", " and ").replace("'", "")
    t = re.sub(r"[^a-z0-9]+", " ", t).strip()
    t = re.sub(r"^the ", "", t)
    return t


def slugify(text: str) -> str:
    return norm(text).replace(" ", "-")


def core_name(venue: str) -> str:
    """The ground's own name: Cricsheet appends the city after the first comma."""
    return venue.split(",")[0].strip()


GENERIC_TOKENS = {
    "the", "stadium", "ground", "grounds", "cricket", "oval", "park", "club", "sports", "sport", "international",
    "national", "academy", "field", "complex", "association", "centre", "center", "of", "and", "de", "new", "no",
    "county", "gardens", "turf", "country", "st", "university", "school",
}


def sig_tokens(text: str) -> frozenset:
    return frozenset(t for t in norm(text).split() if t not in GENERIC_TOKENS)


# venue_context: 'snapshot' = one version per venue from every match in the snapshot (default);
# 'asof' = a new version whenever the geocoder inputs or home team change, each match uses the version valid on its date.
VENUE_CONTEXT_MODES = ("snapshot", "asof")
OVERRIDE_SECTIONS = {"merge": dict, "assign": dict, "split_by_city": list, "city_aliases": dict, "venues": dict, "eras": dict}


def load_overrides(path: Path | None = None) -> dict:
    path = path or CONFIG / "venue_overrides.yml"
    data = (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}
    for key, kind in OVERRIDE_SECTIONS.items():
        data[key] = data.get(key) or kind()
    data["venue_context"] = data.get("venue_context") or "snapshot"
    if data["venue_context"] not in VENUE_CONTEXT_MODES:
        raise OverrideError(f"venue_context must be one of {VENUE_CONTEXT_MODES}, not {data['venue_context']!r}")
    return data


def _split_target(text: str) -> tuple[str, str | None]:
    """'National Stadium|Hamilton' -> ('national stadium', 'hamilton'); 'Lord's' -> ('lords', None)."""
    core, _, city = text.partition("|")
    return norm(core_name(core)), (norm(city) or None)


class OverrideError(ValueError):
    pass


def canonicalize(pairs: list[tuple], overrides: dict, check_stale: bool = True) -> tuple[list[dict], list[dict]]:
    """pairs: (venue, city, n_matches) for every distinct string pair. Returns (venues, aliases).

    Override hooks (config/venue_overrides.yml), applied in this order:
      assign        "<venue> | <city or ~>": "<Core>" or "<Core>|<City>"   exact string pair -> ground
      merge         primary core -> [other core, or "Core|City" for a city-qualified merge]
      split_by_city core names shared by different grounds in different cities
    Every override must take effect; a stale one raises OverrideError instead of being ignored.
    """
    city_alias = {norm(k): norm(v) for k, v in (overrides.get("city_aliases") or {}).items()}

    def city_key(city):
        c = norm(city)
        return city_alias.get(c, c) or None

    assign = {}
    for key, target in (overrides.get("assign") or {}).items():
        venue, _, city = key.partition(" | ")
        city = city.strip()
        assign[(venue.strip(), None if city in ("", "~") else city)] = _split_target(target)
    merge_to, merge_city_to, primary_name = {}, {}, primary_names(overrides)
    for primary, others in (overrides.get("merge") or {}).items():
        p = norm(core_name(primary))
        for other in others or []:
            o_core, o_city = _split_target(other)
            if o_city:
                merge_city_to[(o_core, o_city)] = p
            else:
                merge_to[o_core] = p
    split = {norm(c) for c in overrides.get("split_by_city") or []}

    used_assign, used_merge, used_merge_city = set(), set(), set()
    rows = []
    for venue, city, n in pairs:
        forced_city = None
        if (venue, city) in assign:
            c, forced_city = assign[(venue, city)]
            used_assign.add((venue, city))
        else:
            c = norm(core_name(venue))
        ck = city_key(city)
        if (c, ck) in merge_city_to:
            used_merge_city.add((c, ck))
            c = merge_city_to[(c, ck)]
        elif c in merge_to:
            used_merge.add(c)
            c = merge_to[c]
        rows.append((venue, city, int(n), c, forced_city))

    stale = ([f"assign {k}" for k in assign if k not in used_assign]
             + [f"merge {k!r}" for k in merge_to if k not in used_merge]
             + [f"merge {k!r}" for k in merge_city_to if k not in used_merge_city])
    if stale and check_stale:
        raise OverrideError("venue overrides that match no Cricsheet venue string: " + "; ".join(sorted(stale)))

    # cities known for each split core, so a null city can be read from the venue string's suffix
    split_cities = defaultdict(Counter)
    for venue, city, n, c, forced in rows:
        if c in split and (forced or city_key(city)):
            split_cities[c][forced or city_key(city)] += n

    def split_city(venue, city, c, forced):
        if forced:
            return forced
        if city_key(city):
            return city_key(city)
        for part in venue.split(",")[1:]:
            if city_key(part.strip()) in split_cities[c]:
                return city_key(part.strip())
        ranked = sorted(split_cities[c].items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[0][0] if ranked else "unknown"

    groups = defaultdict(list)
    for venue, city, n, c, forced in rows:
        if c in split:
            key = f"{c}|{split_city(venue, city, c, forced)}"
        elif forced:
            key = f"{c}|{forced}"  # an assign with a city always makes its own ground
        else:
            key = c
        groups[key].append((venue, city, n, c))

    venues, aliases, used = [], [], {}
    for key in sorted(groups):
        members = groups[key]
        c = members[0][3]
        slug = c.replace(" ", "-")  # name-derived only: never depends on scope or counts
        if "|" in key:
            slug += "-" + key.split("|", 1)[1].replace(" ", "-")
        if slug in used and used[slug] != key:
            raise OverrideError(f"venue_id collision on {slug!r}: {used[slug]!r} vs {key!r}")
        used[slug] = key
        venues.append({"venue_id": slug, **present(c, [m[:3] for m in members], primary_name), "core": c})
        for venue, city, n, _ in members:
            aliases.append({"venue": venue, "city": city, "venue_id": slug, "n_matches": n})

    aliases.sort(key=lambda a: (a["venue"], a["city"] or ""))
    ids = {v["venue_id"] for v in venues}
    unknown = sorted(set(overrides.get("venues") or {}) - ids)
    if unknown and check_stale:
        raise OverrideError(f"venue overrides for venue_ids that do not exist: {unknown}")
    for v in venues:
        apply_display_fix(v, overrides)
    return venues, aliases


def primary_names(overrides: dict) -> dict:
    """normalised core of each merge primary -> its display name."""
    return {norm(core_name(p)): p for p in (overrides.get("merge") or {})}


def present(c: str, members: list[tuple], primary_name: dict) -> dict:
    """What the geocoder sees of one ground, from its spellings and their match counts (venue, city, n), n > 0:
    display name and city (most matches), search names and places. Shared by the snapshot-wide and the as-of
    (only matches up to a date) builds, so the two agree when they see the same matches."""
    cores, cities = Counter(), Counter()
    for venue, city, n in members:
        cores[core_name(venue)] += n
        if city:
            cities[city] += n
    display = primary_name.get(c) or sorted(cores.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    city_display = sorted(cities.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if cities else None
    places = set(cities) | {s.strip() for v, _, _ in members for s in v.split(",")[1:] if s.strip()}
    return {"name": display, "city": city_display, "search_names": sorted(cores), "places": sorted(places),
            "n_matches": sum(m[2] for m in members)}


def apply_display_fix(v: dict, overrides: dict) -> dict:
    fix = (overrides.get("venues") or {}).get(v["venue_id"]) or {}
    for k in ("name", "city"):
        if fix.get(k):
            v[k] = fix[k]
    return v


def era_for(venue_id: str, match_date, overrides: dict) -> str:
    """Venue era label: '<venue_id>' before the first configured break, '<venue_id>@<from>' after a rebuild."""
    label = venue_id
    for era in sorted((overrides.get("eras") or {}).get(venue_id) or [], key=lambda e: str(e["from"])):
        if str(match_date) >= str(era["from"]):
            label = f"{venue_id}@{era['from']}"
    return label
