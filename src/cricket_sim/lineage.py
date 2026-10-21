"""Which upstream sources every published table is built from, and what that means for its licence.

The export refuses any table that is not listed here, any table with a non-redistributable source,
and any file whose declared licence is weaker than its sources require.
"""

from . import contracts
from .paths import load_yaml

SOURCE_CONFIGS = ["sources.yml", "context_sources.yml", "player_sources.yml", "wiki_sources.yml"]

# Licence family each source imposes on anything derived from it. Share-alike families cannot be mixed.
SOURCE_LICENCE = {
    "cricsheet": "ODC-BY-1.0",
    "cricsheet_missing": "ODC-BY-1.0",
    "ari": None,             # Cricinfo-derived, non-commercial: never exported
    "hawkeye": None,         # no licence: never exported
    "wikidata_players": "CC0-1.0",
    "venues_geo": "ODbL-1.0",  # Wikidata (CC0) mixed with OpenStreetMap (ODbL): treat as ODbL
    "weather": "CC-BY-4.0",
    "wiki_fixtures": "CC-BY-SA-4.0",
    "derived": "CC0-1.0",    # our own computations from the other sources
}

SHARE_ALIKE = {"ODbL-1.0", "CC-BY-SA-4.0"}
STRENGTH = ["CC0-1.0", "ODC-BY-1.0", "CC-BY-4.0", "CC-BY-SA-4.0", "ODbL-1.0"]

# Which sources each table is built from comes from its contract (customProperties.sources), and a column
# that adds a source beyond its table's carries customProperties.extraSources (exporting it adds that source).
TABLE_SOURCES = contracts.table_sources()
COLUMN_SOURCES = contracts.column_sources()
TABLE_UPSTREAMS = {t: c.props.get("upstream", []) for t, c in contracts.load_all().items()}


def redistributable_flags() -> dict:
    flags = {"derived": True}
    for name in SOURCE_CONFIGS:
        for source, cfg in (load_yaml(name) or {}).items():
            if isinstance(cfg, dict) and "redistributable" in cfg:
                flags[source] = bool(cfg["redistributable"])
    return flags


def source_closure(table: str, columns: list[str] | None = None) -> list[str]:
    """Conservative source closure including effects on row membership and ancestor columns.

    An upstream table edge has no column projection, so every ancestor column contributes.
    Unknown ancestors and cycles are invalid provenance, never permission to omit a source.
    Only the final table's explicitly unselected column extras may be excluded.
    """
    def visit(name: str, selected: list[str] | None, trail: tuple[str, ...]) -> set[str]:
        if name not in TABLE_SOURCES:
            raise SystemExit(f"{name}: no lineage recorded (no contract), so it cannot be exported")
        if name in trail:
            raise SystemExit("lineage cycle: " + " -> ".join((*trail, name)))
        sources = set(TABLE_SOURCES[name])
        for (t, column), extra in COLUMN_SOURCES.items():
            if t == name and (selected is None or column in selected):
                sources.update(extra)
        for ancestor in TABLE_UPSTREAMS.get(name, []):
            sources.update(visit(ancestor, None, (*trail, name)))
        return sources

    sources = visit(table, columns, ())
    if not sources:
        raise SystemExit(f"{table}: no source lineage recorded")
    return sorted(sources)


def required_licence(table: str, columns: list[str] | None = None) -> str:
    """Require a licence compatible with the complete declared source closure."""
    flags = redistributable_flags()
    families = set()
    for source in source_closure(table, columns):
        if not flags.get(source, False) or SOURCE_LICENCE.get(source) is None:
            raise SystemExit(f"{table}: built from '{source}', which is not redistributable")
        families.add(SOURCE_LICENCE[source])
    if len(families & SHARE_ALIKE) > 1:
        raise SystemExit(f"{table}: mixes incompatible share-alike licences {sorted(families & SHARE_ALIKE)}")
    return max(families, key=STRENGTH.index)
