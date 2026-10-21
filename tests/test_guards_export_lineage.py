"""The export and lineage guard paths that no output exercised.

None of these guards changes a number today (no config file uses drop_columns, every run.json carries its dirty
flag, every source is flagged), so a weakened default or a skipped branch would pass every other test. Each guard
is exercised here on the input it exists for."""

import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import duckdb
import pytest
import yaml

from cricket_sim import contracts, export, integrity, lineage

from conftest import FIXTURE, ROOT


# ---- drop_columns: the licence is judged on the columns that stay, and the writer drops the same ones ---------

def _plan_with(monkeypatch, files, columns=("match_id", "umpires", "venue_name")):
    monkeypatch.setattr(export, "load_yaml", lambda name: {"title": "t", "files": files})
    monkeypatch.setattr(export, "_columns", lambda table: list(columns))
    return export.plan()


def test_plan_judges_the_licence_on_the_columns_that_stay(monkeypatch):
    # pretend one column of dim_match brings in a stricter source (ODbL, from venues_geo)
    monkeypatch.setitem(lineage.COLUMN_SOURCES, ("dim_match", "umpires"), ["venues_geo"])
    keep = {"file": "dim_match", "table": "dim_match", "licence": "ODC-BY-1.0"}
    with pytest.raises(SystemExit, match="declared ODC-BY-1.0 but its sources require ODbL-1.0"):
        _plan_with(monkeypatch, [dict(keep)])                                            # all columns: ODbL needed
    files = _plan_with(monkeypatch, [{**keep, "drop_columns": ["umpires"]}])             # dropped: ODC-BY is enough
    assert "venues_geo" not in files[0]["sources"]
    # dropping some OTHER column does not remove the stricter source
    with pytest.raises(SystemExit, match="require ODbL-1.0"):
        _plan_with(monkeypatch, [{**keep, "drop_columns": ["venue_name"]}])


@pytest.fixture(scope="module")
def export_dropping_columns(tmp_path_factory, tiny_build):
    """The real export, run from a copy of the repo whose config drops two columns of dim_match."""
    tmp_path = tmp_path_factory.mktemp("drop")
    repo = tmp_path / "repo"
    shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns(".git", ".venv", "data", "__pycache__",
                                                              ".pytest_cache"))
    config = yaml.safe_load((repo / "config" / "export.yml").read_text())
    for f in config["files"]:
        if f["file"] == "dim_match":
            f["drop_columns"] = list(DROPPED)
    (repo / "config" / "export.yml").write_text(yaml.safe_dump(config, sort_keys=False))
    data = tmp_path / "data"
    shutil.copytree(tiny_build, data, ignore=shutil.ignore_patterns("export"))
    git = lambda *a: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *a], cwd=repo,
                                    capture_output=True, text=True, check=True)
    git("init", "-q"), git("add", "-A"), git("commit", "-q", "-m", "scratch", "--no-gpg-sign")
    env = {**os.environ, "CRICKET_SIM_DATA": str(data), "CRICKET_SIM_SNAPSHOTS": str(FIXTURE),
           "PYTHONPATH": str(repo / "src")}
    done = subprocess.run([sys.executable, "-m", "cricket_sim.cli", "export", "--allow-dirty"], cwd=repo, env=env,
                          capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stdout + done.stderr
    return next((data / "export").iterdir())


DROPPED = ("umpires", "match_referees")


def test_export_writes_exactly_the_columns_plan_checked_the_licence_for(export_dropping_columns):
    out = export_dropping_columns
    parquet = [r[0] for r in duckdb.connect().execute(f"DESCRIBE SELECT * FROM '{out / 'dim_match.parquet'}'").fetchall()]
    full = [c["name"] for c in contracts.get("dim_match").columns]
    assert parquet == [c for c in full if c not in DROPPED]
    assert (out / "dim_match.csv").read_text().splitlines()[0].split(",") == parquet     # CSV and Parquet agree
    assert duckdb.connect().execute(f"SELECT count(*) FROM '{out / 'dim_match.parquet'}'").fetchone()[0] == 9


@pytest.mark.xfail(strict=True, reason="product bug (latent): the export README and croissant.json list a "
                   "dim_match column even when drop_columns removed it from the written file")
def test_export_documents_describe_the_columns_that_were_written(export_dropping_columns):
    out = export_dropping_columns
    section = (out / "README.md").read_text().split("## dim_match")[1].split("\n## ")[0]
    assert "| `umpires` |" not in section and "| `match_id` |" in section
    croissant = json.loads((out / "croissant.json").read_text())
    fields = [f["name"] for r in croissant["recordSet"] if r["name"] == "dim_match" for f in r["field"]]
    assert not set(DROPPED) & set(fields)


# ---- lineage: unknown and unflagged sources are refused ---------------------------------------------------------

def test_a_source_with_no_redistributable_flag_is_not_redistributable(monkeypatch):
    monkeypatch.setitem(lineage.TABLE_SOURCES, "mystery_table", ["mystery_source"])
    monkeypatch.setitem(lineage.SOURCE_LICENCE, "mystery_source", "CC0-1.0")      # a licence, but no flag in any config
    assert "mystery_source" not in lineage.redistributable_flags()
    with pytest.raises(SystemExit, match="mystery_table: built from 'mystery_source', which is not redistributable"):
        lineage.required_licence("mystery_table")


def test_a_flagged_source_with_no_licence_family_is_not_redistributable(monkeypatch):
    monkeypatch.setitem(lineage.TABLE_SOURCES, "unlicensed_table", ["unlicensed_source"])
    monkeypatch.setattr(lineage, "redistributable_flags", lambda: {"derived": True, "unlicensed_source": True})
    with pytest.raises(SystemExit, match="built from 'unlicensed_source', which is not redistributable"):
        lineage.required_licence("unlicensed_table")


def test_an_explicit_false_flag_is_not_redistributable(monkeypatch):
    monkeypatch.setitem(lineage.TABLE_SOURCES, "t", ["cricsheet"])
    monkeypatch.setattr(lineage, "redistributable_flags", lambda: {"derived": True, "cricsheet": False})
    with pytest.raises(SystemExit, match="'cricsheet', which is not redistributable"):
        lineage.required_licence("t")


# ---- export: only gold and model tables, by prefix AND by layer ---------------------------------------------------

def test_a_silver_table_with_a_publishable_name_is_still_refused(monkeypatch):
    # defence in depth: the prefix check passes for 'fact_', so the LAYER check is what has to stop it
    monkeypatch.setattr(contracts, "get", lambda table: SimpleNamespace(layer="silver", data_path=ROOT / "no" / "such.parquet"))
    with pytest.raises(SystemExit, match="only gold dim_/fact_/map_/report_ tables and model_ outputs can be exported"):
        export._table_path("fact_made_up")


def test_the_publishable_layers_are_gold_and_models_only():
    assert export.PUBLISHABLE_LAYERS == ("gold", "models")
    assert export.PUBLISHABLE_PREFIXES == ("dim_", "fact_", "map_", "report_", "model_")


# ---- release inputs: the dirty flag and the recorded artifacts ---------------------------------------------------

def _release_copy(tiny_build, tmp_path):
    data = tmp_path / "data"
    shutil.copytree(tiny_build, data, ignore=shutil.ignore_patterns("export"))
    return data, json.loads((data / "gold" / "run.json").read_text())


def _set_dirty(data, gold_run, value):
    """Record `value` as the code.dirty flag in both runs; None removes the flag (and the whole code record)."""
    model_path = data / "models" / "run.json"
    model_run = json.loads(model_path.read_text())
    for run in (gold_run, model_run):
        if value is None:
            run.pop("code", None)
        else:
            run["code"] = {**run.get("code", {}), "dirty": value}
    model_path.write_text(json.dumps(model_run))
    return gold_run


def test_a_run_with_no_dirty_flag_is_treated_as_dirty(tiny_build, tmp_path):
    data, gold_run = _release_copy(tiny_build, tmp_path)
    gold_run = _set_dirty(data, gold_run, None)
    with pytest.raises(SystemExit, match="refusing to export data or models made from uncommitted code"):
        integrity.verify_release_inputs(data, gold_run, allow_dirty=False)


def test_a_dirty_run_is_refused_unless_dirty_is_allowed_and_a_clean_run_passes(tiny_build, tmp_path):
    data, gold_run = _release_copy(tiny_build, tmp_path)
    gold_run = _set_dirty(data, gold_run, True)
    with pytest.raises(SystemExit, match="refusing to export data or models made from uncommitted code"):
        integrity.verify_release_inputs(data, gold_run, allow_dirty=False)
    assert integrity.verify_release_inputs(data, gold_run, allow_dirty=True)["model_fingerprint"]
    gold_run = _set_dirty(data, gold_run, False)
    assert integrity.verify_release_inputs(data, gold_run, allow_dirty=False)["model_fingerprint"]


def test_one_dirty_run_of_the_two_is_enough_to_refuse(tiny_build, tmp_path):
    data, gold_run = _release_copy(tiny_build, tmp_path)
    gold_run = _set_dirty(data, gold_run, False)
    model_path = data / "models" / "run.json"
    model_run = json.loads(model_path.read_text())
    model_run["code"]["dirty"] = True
    model_path.write_text(json.dumps(model_run))
    with pytest.raises(SystemExit, match="uncommitted code"):
        integrity.verify_release_inputs(data, gold_run, allow_dirty=False)


# ---- recorded artifacts: every kind of change is caught, not only a change of row count --------------------------

def _rewrite(path, sql, compression="zstd"):
    con = duckdb.connect()
    con.execute(f"CREATE TABLE changed AS {sql.format(src=path)}")
    path.unlink()
    con.execute(f"COPY changed TO '{path}' (FORMAT parquet, COMPRESSION {compression})")
    con.close()


CHANGES = {
    "content_same_rows": "SELECT * REPLACE (runs_total + 1 AS runs_total) FROM read_parquet('{src}')",
    "column_type_same_rows": "SELECT * REPLACE (CAST(runs_total AS INTEGER) AS runs_total) FROM read_parquet('{src}')",
    "column_renamed_same_rows": "SELECT * EXCLUDE (runs_total), runs_total AS runs_total_x FROM read_parquet('{src}')",
    "fewer_rows": "SELECT * FROM read_parquet('{src}') LIMIT 5",
    "same_data_other_bytes": "SELECT * FROM read_parquet('{src}')",     # same rows and values, written with snappy
}


@pytest.mark.parametrize("change", list(CHANGES))
def test_verify_artifacts_catches_every_kind_of_change(tiny_build, tmp_path, change):
    data, gold_run = _release_copy(tiny_build, tmp_path)
    integrity.verify_artifacts(data, gold_run, ("gold",))                               # untouched: fine
    _rewrite(data / "gold" / "fact_delivery.parquet", CHANGES[change],
             "snappy" if change == "same_data_other_bytes" else "zstd")
    with pytest.raises(SystemExit, match="gold/fact_delivery: artifact fingerprint mismatch"):
        integrity.verify_artifacts(data, gold_run, ("gold",))


def test_verify_artifacts_catches_a_missing_extra_or_unrecorded_table_and_a_legacy_record(tiny_build, tmp_path):
    data, gold_run = _release_copy(tiny_build, tmp_path)
    shutil.copy(data / "gold" / "dim_match.parquet", data / "gold" / "dim_match_copy.parquet")
    with pytest.raises(SystemExit, match="artifact inventory mismatch"):
        integrity.verify_artifacts(data, gold_run, ("gold",))                           # an extra file
    (data / "gold" / "dim_match_copy.parquet").unlink()
    (data / "gold" / "dim_match.parquet").rename(tmp_path / "moved.parquet")
    with pytest.raises(SystemExit, match="artifact inventory mismatch"):
        integrity.verify_artifacts(data, gold_run, ("gold",))                           # a missing file
    (tmp_path / "moved.parquet").rename(data / "gold" / "dim_match.parquet")
    del gold_run["tables"]["gold/dim_match"]["sha256"]
    with pytest.raises(SystemExit, match="gold/dim_match: legacy artifact fingerprint lacks SHA256"):
        integrity.verify_artifacts(data, gold_run, ("gold",))
    with pytest.raises(SystemExit, match="artifact inventory mismatch"):
        integrity.verify_artifacts(data, {"tables": {}}, ("gold",))                     # nothing recorded
