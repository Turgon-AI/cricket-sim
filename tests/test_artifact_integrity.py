"""A valid new export must never silently carry stale or changed input artifacts."""

import hashlib
import json
import shutil

import duckdb
import pytest

from conftest import run_cli


@pytest.mark.parametrize("mutation", ["stale_model", "changed_gold", "changed_model", "changed_headline"])
def test_export_refuses_mutated_inputs_and_preserves_existing_export(tiny_build, tmp_path, mutation):
    data = tmp_path / "data"
    shutil.copytree(tiny_build, data)
    before = {p.relative_to(data/"export"): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (data/"export").rglob("*") if p.is_file()}
    if mutation == "stale_model":
        path = data / "models/run.json"
        run = json.loads(path.read_text())
        run["build_fingerprint"] = "different-valid-data-generation"
        path.write_text(json.dumps(run))
        message = "model/gold build fingerprint mismatch"
    elif mutation == "changed_headline":
        path = data / "gold/headline.json"
        path.write_text(path.read_text() + " ")
        message = "headline fingerprint mismatch"
    else:
        path = data / ("gold/fact_delivery.parquet" if mutation == "changed_gold" else "models/model_ball_win_probability.parquet")
        con = duckdb.connect()
        con.execute(f"CREATE TABLE changed AS SELECT * FROM read_parquet('{path}')")
        if mutation == "changed_gold":
            con.execute("UPDATE changed SET runs_total = runs_total + 1 WHERE delivery_number = 1")
        else:
            con.execute("UPDATE changed SET wp_before = 0.5 WHERE evaluated AND delivery_number = 1")
        path.unlink()
        con.execute(f"COPY changed TO '{path}' (FORMAT parquet, COMPRESSION zstd)")
        con.close()
        message = "artifact fingerprint mismatch"
    result = run_cli(data, "export", "--allow-dirty", "--skip-full-check")
    assert result.returncode != 0
    assert message in result.stdout + result.stderr
    after = {p.relative_to(data/"export"): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (data/"export").rglob("*") if p.is_file()}
    assert before == after
