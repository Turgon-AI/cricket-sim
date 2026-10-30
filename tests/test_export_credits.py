"""Export credits, card checks and replace protection."""

import json
import re

import duckdb
import pytest

from cricket_sim import export, lineage
from cricket_sim.paths import load_yaml

# A word that must show in a file's credit row for each source in its lineage.
CREDIT_WORD = {"cricsheet": "Cricsheet", "cricsheet_missing": "Cricsheet", "weather": "Open-Meteo",
               "venues_geo": "OpenStreetMap", "wiki_fixtures": "Wikipedia", "wikidata_players": "Wikidata"}


def _export_dir(data):
    run = json.loads((data / "gold" / "run.json").read_text())
    return data / "export" / run["build_fingerprint"][:12]


def _credit_rows(out) -> dict:
    rows = {}
    for line in (out / "LICENSES.md").read_text().splitlines():
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) == 3 and parts[0].endswith((".parquet", ".csv")):
            rows[parts[0]] = parts[2]
    return rows


# credits from lineage ---------------------------------------------------------------------------------------------------------

def test_every_file_credits_every_source_in_its_lineage(tiny_build):
    out = _export_dir(tiny_build)
    rows = _credit_rows(out)
    config = {f["file"]: f for f in load_yaml("export.yml")["files"]}
    assert all(not f.get("drop_columns") for f in config.values())
    manifest = json.loads((out / "manifest.json").read_text())
    missing = []
    for m in manifest:
        f = config[m["file"].rsplit(".", 1)[0]]
        for source in lineage.source_closure(f["table"]):
            word = CREDIT_WORD.get(source)
            if word and word not in rows[m["file"]]:
                missing.append(f"{m['file']}: no {word} credit (source {source})")
        assert export.ATTRIBUTION[m["licence"]] in rows[m["file"]], "the existing credit line must stay"
    assert not missing, missing


def test_weather_file_credits_open_meteo_copernicus_and_cricsheet(tiny_build):
    row = _credit_rows(_export_dir(tiny_build))["fact_match_weather_hourly.parquet"]
    assert "Open-Meteo" in row and "Copernicus" in row and "Cricsheet" in row


def test_export_refuses_a_source_with_no_attribution(monkeypatch):
    monkeypatch.setattr(export, "_source_attributions", lambda: {})
    f = {"file": "x", "licence": "ODC-BY-1.0", "sources": ["cricsheet", "weather"]}
    with pytest.raises(SystemExit, match="no attribution"):
        export.credit_text(f)


# share-alike files ---------------------------------------------------------------------------------------------------------

def test_cc_by_sa_files_carry_source_revisions_licence_link_and_change_note(tiny_build):
    out = _export_dir(tiny_build)
    text = (out / "LICENSES.md").read_text()
    revisions = duckdb.sql(f"SELECT DISTINCT page_revid FROM read_parquet('{tiny_build / 'silver' / 'wikipedia_fixtures.parquet'}')"
                           ).fetchall()
    assert revisions
    for (revid,) in revisions:
        assert f"oldid={revid}" in text
    for name in ("map_match_ids.parquet", "report_international_coverage.parquet"):
        row = _credit_rows(out)[name]
        assert "https://creativecommons.org/licenses/by-sa/4.0/" in row
        assert re.search(r"modified|changes made", row, re.I)
    assert "creativecommons.org/licenses/by-sa" not in _credit_rows(out)["fact_delivery.parquet"]


def test_cc_by_sa_files_ship_the_page_and_revision_each_row_came_from(tiny_build):
    out = _export_dir(tiny_build)
    silver = f"read_parquet('{tiny_build / 'silver' / 'wikipedia_fixtures.parquet'}')"
    pairs = set(duckdb.sql(f"SELECT DISTINCT page_title, page_revid FROM {silver}").fetchall())
    for name in ("map_match_ids", "report_international_coverage"):
        for ext in ("parquet", "csv"):
            rel = f"'{out / (name + '.' + ext)}'"
            cols = [r[0] for r in duckdb.sql(f"DESCRIBE SELECT * FROM {rel}").fetchall()]
            assert cols[-2:] == ["page_title", "page_revid"], (name, ext, cols)
        rows = duckdb.sql(f"SELECT page_title, page_revid FROM '{out / (name + '.parquet')}'").fetchall()
        assert rows
        if name == "map_match_ids":  # every linked match came from a reference row, so it names its page
            assert all(r[0] is not None and r[1] is not None for r in rows)
        assert {r for r in rows if r[0] is not None} <= pairs


# cards match the build --------------------------------------------------------------------------------------------------------

FACTS = {"DATASET_CARD.md": {"build_fingerprint": ("aaaa", "aaaa"), "matches_in_scope": ("12", "12")},
         "MODEL_CARD.md": {"build_fingerprint": ("aaaa", "aaaa"), "model_fingerprint": ("bbbb", "bbbb"),
                           "innings_1_brier": ("0.1891", "0.1891")}}
STAMP = {"DATASET_CARD.md": "build_fingerprint=aaaa matches_in_scope=12",
         "MODEL_CARD.md": "build_fingerprint=aaaa model_fingerprint=bbbb innings_1_brier=0.1891"}
PROSE = {"DATASET_CARD.md": "Build aaaa has 12 matches.", "MODEL_CARD.md": "Model bbbb on build aaaa: Brier 0.1891."}


def _docs(root, stamps=STAMP, prose=PROSE):
    (root / "docs").mkdir(parents=True, exist_ok=True)
    for card in export.CARDS:
        (root / "docs" / card).write_text(f"# {card}\n{prose[card]}\n<!-- cricket-sim card stamp: {stamps[card]} -->\n")


def _ship(monkeypatch, root, allow_dirty=False):
    monkeypatch.setattr(export, "ROOT", root)
    monkeypatch.setattr(export, "card_facts", lambda run, model_run: FACTS)
    return export._checked_cards({}, {}, allow_dirty)


def test_matching_cards_ship_unchanged(tmp_path, monkeypatch):
    _docs(tmp_path)
    cards = _ship(monkeypatch, tmp_path)
    assert cards == {c: (tmp_path / "docs" / c).read_text() for c in export.CARDS}


def test_export_refuses_cards_from_another_build(tmp_path, monkeypatch):
    _docs(tmp_path, stamps={**STAMP, "MODEL_CARD.md": STAMP["MODEL_CARD.md"].replace("bbbb", "cccc")})
    with pytest.raises(SystemExit, match="MODEL_CARD.md: stamp model_fingerprint is cccc, this build has bbbb"):
        _ship(monkeypatch, tmp_path)


def test_export_refuses_a_card_whose_text_shows_other_numbers(tmp_path, monkeypatch):
    _docs(tmp_path, prose={**PROSE, "DATASET_CARD.md": "Build aaaa has 99 matches."})
    with pytest.raises(SystemExit, match="does not show '12'"):
        _ship(monkeypatch, tmp_path)


def test_export_refuses_a_card_with_no_stamp(tmp_path, monkeypatch):
    _docs(tmp_path)
    (tmp_path / "docs" / "DATASET_CARD.md").write_text("# card without a stamp\n")
    with pytest.raises(SystemExit, match="no 'cricket-sim card stamp'"):
        _ship(monkeypatch, tmp_path)


def test_allow_dirty_ships_a_mismatched_card_with_a_warning(tmp_path, monkeypatch):
    _docs(tmp_path, stamps={**STAMP, "DATASET_CARD.md": "build_fingerprint=zzzz matches_in_scope=12"})
    cards = _ship(monkeypatch, tmp_path, allow_dirty=True)
    assert cards["DATASET_CARD.md"].startswith("> **WARNING: this card does not describe this export.**")
    assert cards["MODEL_CARD.md"] == (tmp_path / "docs" / "MODEL_CARD.md").read_text()


def test_card_facts_come_from_the_run_headline_scores_and_coverage(tmp_path, monkeypatch):
    gold, models = tmp_path / "gold", tmp_path / "models"
    gold.mkdir()
    models.mkdir()
    (gold / "headline.json").write_text(json.dumps({"matches_in_scope": 5871, "balls_in_scope": 3657877,
                                                    "pct_matches_with_weather": 100.0,
                                                    "pct_in_scope_players_with_dob_wikidata": 90.5}))
    con = duckdb.connect()
    con.execute(f"COPY (SELECT * FROM (VALUES ('ODI', 'afghanistan_withheld_by_cricsheet'), ('ODI', 'in_cricsheet'), "
                f"('ODI', 'afghanistan_withheld_by_cricsheet')) t(segment, status)) TO '{gold / 'report_international_coverage.parquet'}'")
    con.execute(f"COPY (SELECT * FROM (VALUES ('win_probability', 'innings_1', 'model', 'all', 0.18912), "
                f"('win_probability', 'innings_2', 'model', 'all', 0.11927), "
                f"('win_probability', 'innings_1', 'base_rate', 'all', 0.2474)) t(target, slice, model, scope, brier)) "
                f"TO '{models / 'report_model_scores.parquet'}'")
    monkeypatch.setattr(export, "GOLD", gold)
    monkeypatch.setattr(export, "MODELS", models)
    facts = export.card_facts({"build_fingerprint": "f0c27be17d2ded3acb9d"}, {"model_fingerprint": "487a4285d89a8dac1234"})
    assert facts["DATASET_CARD.md"] == {
        "build_fingerprint": ("f0c27be17d2ded3a", "f0c27be17d2ded3a"), "matches_in_scope": ("5871", "5,871"),
        "balls_in_scope": ("3657877", "3,657,877"), "pct_matches_with_weather": ("100", "100%"),
        "pct_in_scope_players_with_dob_wikidata": ("90.5", "90.5%"),
        "afghanistan_withheld_odis": ("2", "2 involve Afghanistan")}
    assert facts["MODEL_CARD.md"] == {
        "build_fingerprint": ("f0c27be17d2ded3a", "f0c27be17d2ded3a"),
        "model_fingerprint": ("487a4285d89a8dac", "487a4285d89a8dac"),
        "innings_1_brier": ("0.1891", "0.1891"), "innings_2_brier": ("0.1193", "0.1193")}


def test_docs_refresh_rewrites_the_stamp_and_only_the_stamp():
    old = "# card\nBuild aaaa has 12 matches.\n\n<!-- cricket-sim card stamp: build_fingerprint=zzzz matches_in_scope=1 -->\n"
    facts = {"build_fingerprint": ("aaaa", "aaaa"), "matches_in_scope": ("12", "12")}
    new = export.refresh_stamp(old, facts)
    assert new == old.replace("build_fingerprint=zzzz matches_in_scope=1", "build_fingerprint=aaaa matches_in_scope=12")
    assert export.card_problems("DATASET_CARD.md", new, facts) == []
    assert export.refresh_stamp("# no stamp\n", facts).endswith("matches_in_scope=12 -->\n")
    stale_prose = old.replace("12 matches", "1 matches")  # a refreshed stamp cannot hide stale prose
    assert export.card_problems("DATASET_CARD.md", export.refresh_stamp(stale_prose, facts), facts)


def test_tiny_export_does_not_ship_the_real_data_cards_as_its_own(tiny_build):
    """The fixture build is not the build the repo's cards describe, so the shipped cards must say so."""
    out = _export_dir(tiny_build)
    for card in export.CARDS:
        assert (out / card).read_text().startswith("> **WARNING: this card does not describe this export.**")


# no silent replace ---------------------------------------------------------------------------------------------------------

def _stub_export(monkeypatch, tmp_path, readme):
    data, gold = tmp_path / "data", tmp_path / "gold"
    gold.mkdir(exist_ok=True)
    (gold / "run.json").write_text(json.dumps({"build_fingerprint": "abcdef1234567890"}))
    monkeypatch.setattr(export, "DATA", data)
    monkeypatch.setattr(export, "GOLD", gold)

    def write(_allow_dirty, out, _skip_full_check=False):
        out.mkdir(parents=True)
        (out / "README.md").write_text(readme)
        (out / "fact_x.parquet").write_bytes(b"data")

    monkeypatch.setattr(export, "_write_export", write)
    return data / "export" / "abcdef123456"


def test_re_export_that_differs_does_not_replace_the_existing_folder(tmp_path, monkeypatch):
    live = _stub_export(monkeypatch, tmp_path, "built at commit AAA")
    export.export()
    assert (live / "README.md").read_text() == "built at commit AAA"
    _stub_export(monkeypatch, tmp_path, "built at commit BBB")
    with pytest.raises(SystemExit, match="differs in 1 file.*README.md.*Not replacing it"):
        export.export()
    assert (live / "README.md").read_text() == "built at commit AAA"


def test_identical_re_export_is_allowed(tmp_path, monkeypatch):
    live = _stub_export(monkeypatch, tmp_path, "same")
    export.export()
    _stub_export(monkeypatch, tmp_path, "same")
    export.export()
    assert (live / "README.md").read_text() == "same"
