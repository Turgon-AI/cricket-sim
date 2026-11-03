"""The contracts are the single source of truth, so they must be valid, complete and actually enforced."""

import duckdb
import pytest
import yaml

from cricket_sim import contracts, docs


def test_every_contract_is_valid_odcs_and_complete():
    assert contracts.lint() == []


def test_generated_docs_are_current():
    assert (docs.DOCS / "DATA_DICTIONARY.md").read_text() == docs.dictionary_md(), "run `cricket docs`"
    assert (docs.DOCS / "LINEAGE.md").read_text() == docs.lineage_md(), "run `cricket docs`"


def test_per_ball_tables_declare_one_timing_class_and_the_ball_key():
    registry = contracts.timing_registry()
    assert {"int_ball_state_pre", "hawkeye_ball_delivery", "cricinfo_ball_labels", "cricinfo_ball_post"} <= set(registry)
    for table in registry:
        assert contracts.get(table).primary_key == ["match_id", "innings", "delivery_number"], table


def _timings(table: str) -> dict:
    c = contracts.get(table)
    return {col["name"]: c.column_props(col).get("timingClass") for col in c.columns if not col.get("primaryKey")}


def test_fact_delivery_marks_every_column_pre_ball_or_outcome():
    timing = _timings("fact_delivery")
    assert set(timing.values()) == {"pre_ball", "outcome"}, timing
    assert timing["innings_runs_before"] == "pre_ball" and timing["runs_total"] == "outcome"
    assert timing["nominal_runs_required_before"] == "pre_ball"
    # columns are grouped: everything knowable before the ball, then the outcome
    order = list(timing.values())
    assert order == sorted(order, key=["pre_ball", "outcome"].index)


def test_hindsight_state_lives_outside_fact_delivery():
    """SELECT * on fact_delivery must not hand a model a column whose NULLs reveal how the innings ended."""
    hindsight = _timings("fact_delivery_hindsight")
    assert set(hindsight.values()) == {"hindsight"}
    assert {"target_status", "target_runs", "runs_required_before", "balls_remaining_before"} <= set(hindsight)
    assert not set(hindsight) & set(_timings("fact_delivery"))
    assert contracts.get("fact_delivery_hindsight").primary_key == contracts.get("fact_delivery").primary_key


def _fixture(tmp_path, monkeypatch, parent_rows, child_rows):
    """A two-table world (parent, child) with its own contracts and data folders."""
    data, specs = tmp_path / "data", tmp_path / "contracts"
    (data / "silver").mkdir(parents=True)
    (specs / "silver").mkdir(parents=True)
    con = duckdb.connect()
    con.execute(f"COPY (SELECT * FROM (VALUES {parent_rows}) t(id, kind)) TO '{data / 'silver' / 'parent.parquet'}'")
    con.execute(f"COPY (SELECT * FROM (VALUES {child_rows}) t(id, parent_id, runs, extras, total)) "
                f"TO '{data / 'silver' / 'child.parquet'}'")

    def contract(name, props, quality=None):
        obj = {"name": name, "physicalType": "table", "logicalType": "object", "description": "x",
               "dataGranularityDescription": "x", "properties": props}
        if quality:
            obj["quality"] = quality
        doc = {"apiVersion": "v3.1.0", "kind": "DataContract", "id": f"cricket-sim.silver.{name}", "name": name,
               "version": "1.0.0", "status": "active", "schema": [obj],
               "customProperties": [{"property": "layer", "value": "silver"}, {"property": "kind", "value": "entity"},
                                    {"property": "sources", "value": ["cricsheet"]}]}
        (specs / "silver" / f"{name}.yaml").write_text(yaml.safe_dump(doc))

    col = lambda name, ptype, **kw: {"name": name, "logicalType": "string" if ptype == "VARCHAR" else "integer",
                                     "physicalType": ptype, "description": "x", **kw}
    contract("parent", [col("id", "VARCHAR", primaryKey=True, primaryKeyPosition=1, required=True),
                        col("kind", "VARCHAR", quality=[{"metric": "invalidValues",
                                                         "arguments": {"validValues": ["a", "b"]}, "mustBe": 0}])])
    contract("child", [col("id", "VARCHAR", primaryKey=True, primaryKeyPosition=1, required=True),
                       col("parent_id", "VARCHAR", required=True, relationships=[{"to": "parent.id"}]),
                       col("runs", "INTEGER"), col("extras", "INTEGER"), col("total", "INTEGER")],
             quality=[{"type": "sql", "name": "total_adds_up",
                       "query": "SELECT count(*) FROM ${table} WHERE total <> runs + extras", "mustBe": 0}])
    monkeypatch.setattr(contracts, "CONTRACTS", specs)
    monkeypatch.setattr(contracts, "LAYERS", {"staging": data / "staging", "silver": data / "silver", "gold": data / "gold"})
    contracts.load_all.cache_clear()
    return con


@pytest.fixture(autouse=True)
def _reset_cache():
    yield
    contracts.load_all.cache_clear()


def test_enforce_passes_clean_data(tmp_path, monkeypatch):
    con = _fixture(tmp_path, monkeypatch, "('p1', 'a'), ('p2', 'b')", "('c1', 'p1', 1, 0, 1), ('c2', 'p2', 4, 1, 5)")
    assert contracts.lint() == []
    assert contracts.enforce(con)["checks_passed"] >= 6


def test_enforce_reports_every_kind_of_violation(tmp_path, monkeypatch):
    con = _fixture(tmp_path, monkeypatch, "('p1', 'a'), ('p1', 'z')",
                   "('c1', 'p1', 1, 0, 1), ('c2', 'p9', 4, 1, 9), ('c3', NULL, 0, 0, 0)")
    with pytest.raises(SystemExit) as err:
        contracts.enforce(con)
    msg = str(err.value)
    assert "parent: primary key" in msg and "1 duplicates" in msg       # duplicate key
    assert "parent.kind: invalidValues = 1" in msg                       # enum
    assert "child.parent_id: required but 1 NULLs" in msg                # not-null
    assert "child.parent_id: 1 values missing from parent.id" in msg     # foreign key
    assert "child: total_adds_up = 1" in msg                             # table-level SQL rule


def test_enforce_rejects_schema_drift_and_uncontracted_tables(tmp_path, monkeypatch):
    con = _fixture(tmp_path, monkeypatch, "('p1', 'a')", "('c1', 'p1', 1, 0, 1)")
    con.execute(f"COPY (SELECT 1 AS x) TO '{tmp_path / 'data' / 'silver' / 'stray.parquet'}'")
    with pytest.raises(SystemExit, match="stray: built but has no contract"):
        contracts.enforce(con)
    (tmp_path / "data" / "silver" / "stray.parquet").unlink()
    con.execute(f"COPY (SELECT 'p1' AS id, 1 AS kind) TO '{tmp_path / 'data' / 'silver' / 'parent.parquet'}'")
    with pytest.raises(SystemExit, match="parent: columns differ from contract"):
        contracts.enforce(con)
