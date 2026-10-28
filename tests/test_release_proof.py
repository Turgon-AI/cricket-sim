"""What a release must prove about itself (data-engineering review, 2026-10-27).

- a build cannot record itself as clean when git cannot report the tree's state
- export refuses a release unless scripts/full_check.sh passed on exactly this build and model
- a table that loses rows fails the build: real-data row floors in the contracts, and a build-to-build
  comparison that refuses a drop of more than 1% unless --allow-shrink is passed"""

import json
import subprocess

import pytest

from cricket_sim import build, contracts, integrity


# ---- git state ---------------------------------------------------------------------------------------------

def test_missing_git_is_recorded_as_dirty(monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")
    monkeypatch.setattr(subprocess, "run", no_git)
    state = build._git_state()
    assert state["dirty"] is True and state["commit"] is None


def test_failing_git_is_recorded_as_dirty(monkeypatch):
    # e.g. a copied tree that is not a repository: git runs but exits non-zero with empty stdout
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 128, "", "not a git repo"))
    state = build._git_state()
    assert state["dirty"] is True and state["commit"] is None


# ---- full-check record -------------------------------------------------------------------------------------

def _runs(build_fp="b" * 64, model_fp="m" * 64):
    gold = {"build_fingerprint": build_fp, "code": {"commit": "c" * 40, "dirty": False, "source_sha256": "s" * 64}}
    return gold, {"model_fingerprint": model_fp}


def _write_runs(data, gold, model):
    (data / "gold").mkdir(parents=True, exist_ok=True)
    (data / "models").mkdir(parents=True, exist_ok=True)
    (data / "gold/run.json").write_text(json.dumps(gold))
    (data / "models/run.json").write_text(json.dumps(model))


def test_release_export_needs_a_passed_full_check(tmp_path):
    gold, model = _runs()
    _write_runs(tmp_path, gold, model)
    with pytest.raises(SystemExit, match="no passed full-data check"):
        integrity.verify_full_check(tmp_path, gold, model, allow_dirty=False)
    integrity.verify_full_check(tmp_path, gold, model, allow_dirty=True)        # tests and dirty exports skip it

    integrity.record_full_check(tmp_path)
    integrity.verify_full_check(tmp_path, gold, model, allow_dirty=False)
    record = json.loads((tmp_path / integrity.FULL_CHECK).read_text())
    assert record["passed"] and record["build_fingerprint"] == gold["build_fingerprint"]


@pytest.mark.parametrize("stale", ["build", "model", "source"])
def test_a_full_check_of_another_build_does_not_count(tmp_path, stale):
    gold, model = _runs()
    _write_runs(tmp_path, gold, model)
    integrity.record_full_check(tmp_path)
    if stale == "build":
        gold = {**gold, "build_fingerprint": "x" * 64}
    elif stale == "model":
        model = {"model_fingerprint": "x" * 64}
    else:
        gold = {**gold, "code": {**gold["code"], "source_sha256": "x" * 64}}
    with pytest.raises(SystemExit, match="no passed full-data check"):
        integrity.verify_full_check(tmp_path, gold, model, allow_dirty=False)


def test_the_tiny_build_has_no_full_check_record(tiny_build):
    # CI builds the 10-match sample only; nothing may claim the full check ran on it
    assert not (tiny_build / integrity.FULL_CHECK).exists()


# ---- row floors and build-to-build shrink ------------------------------------------------------------------

def _tables(**rows):
    return {f"gold/{name}": {"rows": n} for name, n in rows.items()}


def test_shrink_of_more_than_one_percent_fails_the_build():
    previous = {"build_fingerprint": "p" * 64, "tables": _tables(fact_delivery=1000, dim_match=100)}
    with pytest.raises(SystemExit, match=r"gold/fact_delivery: 1,000 -> 980 rows"):
        build.shrink_check(_tables(fact_delivery=980, dim_match=100), previous, allow_shrink=False)
    ok = build.shrink_check(_tables(fact_delivery=990, dim_match=120), previous, allow_shrink=False)
    assert ok == {"previous_build": "p" * 64, "allowed": False, "shrunk": {}}


def test_allow_shrink_records_what_shrank():
    previous = {"build_fingerprint": "p" * 64, "tables": _tables(fact_delivery=1000)}
    result = build.shrink_check(_tables(fact_delivery=10, new_table=5), previous, allow_shrink=True)
    assert result["shrunk"] == {"gold/fact_delivery": {"previous": 1000, "now": 10}}
    assert build.shrink_check(_tables(fact_delivery=10), None, allow_shrink=False)["previous_build"] is None


def test_every_contract_has_a_real_data_row_floor():
    missing = [t for t, c in contracts.load_all().items()
               if not any(r.get("metric") == "rowCount" and contracts._custom(r.get("customProperties")).get("fullDataOnly")
                          for r in c.obj.get("quality") or [])]
    assert not missing


def test_real_data_floors_apply_to_full_data_and_skip_the_sample(monkeypatch, tiny_build):
    import duckdb

    c = contracts.get("fact_delivery")
    rules = c.obj["quality"]
    con = duckdb.connect()
    monkeypatch.setattr(contracts, "DATA", tiny_build)
    monkeypatch.setattr(contracts.Contract, "data_path", property(lambda self: tiny_build / self.layer / f"{self.table}.parquet"))
    monkeypatch.setattr(contracts, "SUBSET", True)
    assert contracts._check_rules(con, c, rules, None)[1] == []
    monkeypatch.setattr(contracts, "SUBSET", False)
    failures = contracts._check_rules(con, c, rules, None)[1]
    assert any("row_count_floor_full_data" in f for f in failures)


def test_lint_rejects_unknown_rule_properties(monkeypatch):
    c = contracts.get("dim_match")
    bad = {"name": "x", "metric": "rowCount", "mustBeGreaterOrEqualTo": 1,
           "customProperties": [{"property": "fullDataOnyl", "value": True}]}
    monkeypatch.setitem(c.obj, "quality", [*(c.obj.get("quality") or []), bad])
    assert any("unknown customProperties ['fullDataOnyl']" in p for p in contracts.lint())
