"""Failure handling: readers must be loud on a half-published tree, a cut-off zstd file,
an empty source and a missing or partial lock."""

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
import zstandard

from cricket_sim import bronze, contracts, missing, paths, transaction

from conftest import FIXTURE, ROOT


def cli(data: Path, snapshots: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "CRICKET_SIM_DATA": str(data), "CRICKET_SIM_SNAPSHOTS": str(snapshots)}
    return subprocess.run([sys.executable, "-m", "cricket_sim.cli", *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=300)


# ---------------------------------------------------------------- Verify needs a usable lock

@pytest.fixture
def snaps(tmp_path):
    folder = tmp_path / "snaps"
    folder.mkdir()
    for name in ("snapshots.lock.yml", "snapshots.pins.yml", "SUBSET"):
        shutil.copy(FIXTURE / name, folder / name)
    return folder


@pytest.fixture(scope="module")
def bronze_copy(tmp_path_factory):
    data = tmp_path_factory.mktemp("verify") / "data"
    shutil.copytree(FIXTURE / "bronze", data / "bronze")
    return data


def test_verify_passes_with_intact_lock_and_pins(bronze_copy, snaps):
    assert cli(bronze_copy, snaps, "verify").returncode == 0


def test_verify_fails_when_lock_is_cut_at_a_line_boundary(bronze_copy, snaps):
    lines = (snaps / "snapshots.lock.yml").read_text().splitlines(keepends=True)
    (snaps / "snapshots.lock.yml").write_text("".join(lines[:3]))
    result = cli(bronze_copy, snaps, "verify")
    assert result.returncode != 0 and "required" in result.stdout + result.stderr


def test_verify_fails_when_lock_is_empty(bronze_copy, snaps):
    (snaps / "snapshots.lock.yml").write_text("")
    result = cli(bronze_copy, snaps, "verify")
    assert result.returncode != 0 and "lock file is empty" in result.stdout + result.stderr


def test_verify_fails_when_lock_is_missing(bronze_copy, snaps):
    (snaps / "snapshots.lock.yml").unlink()
    result = cli(bronze_copy, snaps, "verify")
    assert result.returncode != 0 and "snapshots.lock.yml" in result.stdout + result.stderr


def test_verify_fails_when_snapshot_folder_has_no_lock(bronze_copy, tmp_path):
    empty = tmp_path / "typo"
    empty.mkdir()
    result = cli(bronze_copy, empty, "verify")
    assert result.returncode != 0 and "lock file missing" in result.stdout + result.stderr


@pytest.mark.parametrize("damage", ["missing", "empty", "cut_at_line", "cut_mid_line"])
def test_verify_fails_on_missing_empty_or_cut_pins(bronze_copy, snaps, damage):
    pins = snaps / "snapshots.pins.yml"
    text = pins.read_text()
    if damage == "missing":
        pins.unlink()
        message = "pins file missing"
    elif damage == "empty":
        pins.write_text("")
        message = "no manifest hash pin for locked snapshot"
    elif damage == "cut_at_line":
        pins.write_text("".join(text.splitlines(keepends=True)[:3]))
        message = "no manifest hash pin for locked snapshot"
    else:
        pins.write_text(text[:len(text) - 20])
        message = "_snapshot.json differs from the one that was locked"
    # It must fail for the pin damage, not for any other reason
    result = cli(bronze_copy, snaps, "verify")
    assert result.returncode != 0 and message in result.stdout + result.stderr


# ---------------------------------------------------------------- A cut-off zstd file must fail

def _snapshot_with(tmp_path, raw: bytes, cut: float | None = None, **entry_extra) -> Path:
    """A one-file snapshot folder whose manifest and stored hashes match the (possibly cut) stored bytes."""
    folder = tmp_path / "snap"
    folder.mkdir()
    stored = zstandard.ZstdCompressor(level=bronze.ZSTD_LEVEL).compress(raw)
    if cut:
        stored = stored[:int(len(stored) * cut)]
    (folder / "missing.html.zst").write_bytes(stored)
    entry = {"name": "missing.html", "stored_as": "missing.html.zst", "compressed": True,
             "raw_bytes": len(raw), "raw_sha256": hashlib.sha256(raw).hexdigest(),
             "stored_bytes": len(stored), "stored_sha256": hashlib.sha256(stored).hexdigest(), **entry_extra}
    (folder / "_snapshot.json").write_text(json.dumps({"files": [entry]}))
    return folder


def test_read_zst_rejects_truncated_stream_even_with_matching_stored_hash(tmp_path):
    folder = _snapshot_with(tmp_path, b"<dt>2020-01-01</dt><dd>A vs B</dd>\n" * 5000, cut=0.6)
    with pytest.raises(SystemExit, match="truncated|incomplete|cut off"):
        bronze.read_zst(folder / "missing.html.zst")


def test_read_zst_rejects_length_that_disagrees_with_manifest(tmp_path):
    raw = b"hello world\n" * 1000
    folder = _snapshot_with(tmp_path, raw)
    manifest = json.loads((folder / "_snapshot.json").read_text())
    manifest["files"][0]["raw_bytes"] += 1
    (folder / "_snapshot.json").write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="raw_bytes"):
        bronze.read_zst(folder / "missing.html.zst")


def test_read_zst_rejects_content_that_disagrees_with_manifest_hash(tmp_path):
    folder = _snapshot_with(tmp_path, b"hello world\n" * 1000)
    manifest = json.loads((folder / "_snapshot.json").read_text())
    manifest["files"][0]["raw_sha256"] = "0" * 64
    (folder / "_snapshot.json").write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="raw_sha256"):
        bronze.read_zst(folder / "missing.html.zst")


def test_read_zst_still_reads_a_good_file(tmp_path):
    raw = b"hello world\n" * 1000
    folder = _snapshot_with(tmp_path, raw)
    assert bronze.read_zst(folder / "missing.html.zst") == raw
    lone = tmp_path / "lone.zst"  # no manifest next to it (a resumed ingest page): completeness is still checked
    lone.write_bytes(zstandard.ZstdCompressor().compress(raw))
    assert bronze.read_zst(lone) == raw


def test_build_cricsheet_missing_refuses_a_cut_file(tmp_path, monkeypatch):
    folder = _snapshot_with(tmp_path, b"<h4>By match type</h4><h5>Test Matches</h5><h6>Male matches</h6>\n"
                            + b"<dt>2020-01-01</dt><dd>A vs B</dd>\n" * 5000, cut=0.6)
    monkeypatch.setattr(missing, "SILVER", tmp_path)
    with pytest.raises(SystemExit, match="missing.html.zst: zstd stream is cut off"):   # The cut, not any exit
        missing.build_cricsheet_missing(folder)
    assert not (tmp_path / "cricsheet_missing.parquet").exists()


def test_verify_decompresses_every_zst_and_catches_a_cut_file(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "BRONZE", tmp_path / "bronze")
    monkeypatch.setattr(bronze, "BRONZE", tmp_path / "bronze")
    monkeypatch.setattr(bronze, "LOCK_FILE", tmp_path / "lock.yml")
    monkeypatch.setattr(bronze, "PINS_FILE", tmp_path / "pins.yml")
    folder = paths.snapshot_dir("src", "s1")
    folder.mkdir(parents=True)
    raw = b"a,b\n1,2\n" * 5000
    stored = zstandard.ZstdCompressor().compress(raw)
    (folder / "a.csv.zst").write_bytes(stored[:len(stored) // 2])
    bronze.write_manifest("src", "s1", {}, [{"name": "a.csv", "stored_as": "a.csv.zst", "compressed": True,
                                             "raw_bytes": len(raw), "raw_sha256": hashlib.sha256(raw).hexdigest()}])
    bronze.write_lock({"src": "s1"})
    bronze.PINS_FILE.write_text(yaml.safe_dump({"src": bronze.manifest_sha("src", "s1")}))
    assert any("a.csv.zst" in p and ("truncated" in p or "incomplete" in p or "cut off" in p)
               for p in bronze.verify("src", "s1"))


# ---------------------------------------------------------------- Empty or shrunken sources

FLOORED = {
    "staging": ["stg_ari_players", "stg_ari_predictions", "stg_ari_tracking", "stg_hawkeye_balls"],
    "silver": ["balls", "wickets", "innings", "matches", "match_players", "people", "players", "cricsheet_missing",
               "wikipedia_fixtures", "cricinfo_ball_pre", "cricinfo_ball_post", "cricinfo_ball_labels",
               "hawkeye_ball_delivery", "match_venues", "match_weather_hourly", "venues"],
    "gold": ["dim_match", "dim_player", "dim_venue", "fact_delivery", "fact_innings", "fact_dismissal",
             "report_cricsheet_known_missing", "report_international_coverage", "report_hawkeye_summary"],
}


def _row_floor(c) -> int | None:
    for rule in c.obj.get("quality") or []:
        if rule.get("metric") == "rowCount" and "mustBeGreaterOrEqualTo" in rule:
            return rule["mustBeGreaterOrEqualTo"]
    return None


@pytest.mark.parametrize("layer", sorted(FLOORED))
def test_source_fed_tables_have_a_row_count_floor(layer):
    for table in FLOORED[layer]:
        floor = _row_floor(contracts.get(table))
        assert floor is not None and floor >= 1, f"{table}: no rowCount floor in its contract"


def test_row_count_floors_hold_on_the_sample_build(tiny_build):
    rows = {k.split("/")[1]: v["rows"] for k, v in json.loads((tiny_build / "gold" / "run.json").read_text())["tables"].items()}
    for c in contracts.load_all().values():
        if c.layer != "models" and _row_floor(c) is not None:
            assert rows[c.table] >= _row_floor(c), c.table


def test_row_count_floors_hold_on_the_full_data_when_present():
    run = ROOT / "data" / "gold" / "run.json"
    if not run.exists():
        pytest.skip("no full build under data/")
    rows = {k.split("/")[1]: v["rows"] for k, v in json.loads(run.read_text())["tables"].items()}
    for c in contracts.load_all().values():
        if c.layer != "models" and _row_floor(c) is not None:
            assert rows[c.table] >= _row_floor(c), c.table


def test_contracts_fail_when_a_source_fed_silver_table_is_empty(tiny_build, tmp_path):
    import duckdb

    data = tmp_path / "data"
    shutil.copytree(tiny_build, data)
    target = data / "silver" / "cricsheet_missing.parquet"
    duckdb.sql(f"COPY (SELECT * FROM read_parquet('{target}') LIMIT 0) TO '{target}.new' (FORMAT parquet)")
    os.replace(f"{target}.new", target)
    result = cli(data, FIXTURE, "contracts")
    assert result.returncode != 0 and "cricsheet_missing" in result.stdout + result.stderr


def test_wikidata_page_with_fewer_rows_than_its_manifest_is_refused(tmp_path):
    from cricket_sim import context_players

    raw = json.dumps({"results": {"bindings": []}}).encode()
    stored = zstandard.ZstdCompressor().compress(raw)
    (tmp_path / "base.p000.json.zst").write_bytes(stored)
    entry = {"name": "base.p000.json", "stored_as": "base.p000.json.zst", "rows": 500,
             "raw_bytes": len(raw), "raw_sha256": hashlib.sha256(raw).hexdigest()}
    (tmp_path / "_snapshot.json").write_text(json.dumps({"files": [entry]}))
    with pytest.raises(SystemExit, match="500"):
        context_players._read_bronze(tmp_path)


# ---------------------------------------------------------------- Readers refuse a half-published tree

def _prepared_journal(data: Path, destinations=("staging", "silver", "gold", "warehouse.duckdb")) -> None:
    record = {"id": "deadbeef", "state": "prepared", "stage": ".pending/build-x", "backup": ".history/deadbeef",
              "destinations": list(destinations), "existed": {n: True for n in destinations}}
    transaction._save(data / ".transactions" / "deadbeef.json", record)


def test_contracts_refuse_a_tree_with_a_prepared_journal(tiny_build, tmp_path):
    data = tmp_path / "data"
    shutil.copytree(tiny_build, data)
    assert cli(data, FIXTURE, "contracts").returncode == 0
    _prepared_journal(data)
    result = cli(data, FIXTURE, "contracts")
    assert result.returncode != 0
    assert "half-published" in result.stdout + result.stderr


def test_contracts_say_so_when_the_models_layer_is_absent(tiny_build, tmp_path):
    data = tmp_path / "data"
    shutil.copytree(tiny_build, data)
    shutil.rmtree(data / "models")
    result = cli(data, FIXTURE, "contracts")
    assert "models" in result.stdout and "not checked" in result.stdout


def test_contracts_check_an_empty_models_directory_instead_of_skipping_it(tiny_build, tmp_path):
    data = tmp_path / "data"
    shutil.copytree(tiny_build, data)
    shutil.rmtree(data / "models")
    (data / "models").mkdir()
    result = cli(data, FIXTURE, "contracts")
    assert result.returncode != 0 and "was not built" in result.stdout + result.stderr


def test_pending_publications_lists_only_prepared_journals(tmp_path):
    assert transaction.pending_publications(tmp_path) == []
    _prepared_journal(tmp_path)
    assert [r["id"] for r in transaction.pending_publications(tmp_path)] == ["deadbeef"]
    transaction.recover(tmp_path)
    assert transaction.pending_publications(tmp_path) == []
