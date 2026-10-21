import pytest

from cricket_sim import export, lineage


def test_restricted_sources_can_never_be_exported():
    for table in ["cricinfo_ball_labels", "hawkeye_ball_delivery", "cricinfo_ball_wagon", "cricinfo_ball_pre",
                  "cricinfo_ball_outcome_predicted", "cricinfo_players"]:
        with pytest.raises(SystemExit, match="not redistributable"):
            lineage.required_licence(table)


def test_unknown_table_cannot_be_exported():
    with pytest.raises(SystemExit, match="no lineage"):
        lineage.required_licence("venue_geometry")


def test_licence_follows_the_strictest_source():
    assert lineage.required_licence("fact_delivery") == "ODC-BY-1.0"
    assert lineage.required_licence("map_match_ids") == "CC-BY-SA-4.0"
    assert lineage.required_licence("dim_venue") == "ODbL-1.0"


def test_export_reads_gold_marts_only():
    for table in ["balls", "int_ball_state_pre", "stg_ari_tracking", "venues"]:
        with pytest.raises(SystemExit, match="only gold"):
            export._table_path(table)


def test_share_alike_licences_cannot_mix(monkeypatch):
    monkeypatch.setitem(lineage.TABLE_SOURCES, "bad", ["wiki_fixtures", "venues_geo"])
    with pytest.raises(SystemExit, match="incompatible share-alike"):
        lineage.required_licence("bad")


def test_export_config_is_valid_and_excludes_geometry():
    files = export.plan()
    assert "venue_geometry" not in {f["table"] for f in files}


def test_weather_window_is_internal_but_published_hourly_uses_open_window():
    with pytest.raises(SystemExit, match="not redistributable"):
        lineage.required_licence("int_match_weather_daily")
    assert lineage.required_licence("match_weather_hourly") == "ODbL-1.0"
    assert lineage.required_licence("fact_match_weather_hourly") == "ODbL-1.0"


def test_export_licence_follows_transitive_venue_lineage():
    assert lineage.required_licence("model_ball_win_probability") == "ODbL-1.0"


def test_ancestor_column_restriction_cannot_be_hidden_by_projection(monkeypatch):
    monkeypatch.setitem(lineage.COLUMN_SOURCES, ("balls", "runs_total"), ["ari"])
    with pytest.raises(SystemExit, match="not redistributable"):
        lineage.required_licence("fact_delivery", ["match_id"])


def test_ancestor_table_restriction_is_transitive(monkeypatch):
    monkeypatch.setitem(lineage.TABLE_SOURCES, "balls", ["cricsheet", "hawkeye"])
    with pytest.raises(SystemExit, match="not redistributable"):
        lineage.required_licence("fact_delivery")


def test_unknown_ancestor_and_cycles_fail_closed(monkeypatch):
    monkeypatch.setitem(lineage.TABLE_UPSTREAMS, "balls", ["missing_source_table"])
    with pytest.raises(SystemExit, match="no lineage"):
        lineage.required_licence("fact_delivery")
    monkeypatch.setitem(lineage.TABLE_UPSTREAMS, "balls", ["fact_delivery"])
    with pytest.raises(SystemExit, match="lineage cycle"):
        lineage.required_licence("fact_delivery")


def test_final_projection_does_not_drop_ancestor_extras(monkeypatch):
    monkeypatch.setitem(lineage.COLUMN_SOURCES, ("fact_delivery", "private_note"), ["ari"])
    assert lineage.required_licence("fact_delivery", ["match_id"]) == "ODC-BY-1.0"
    with pytest.raises(SystemExit, match="not redistributable"):
        lineage.required_licence("fact_delivery")
