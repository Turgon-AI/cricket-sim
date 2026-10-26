"""The checking code is itself tested, on its edges, with the sabotage that proves the test would notice.

A check that quietly stops checking looks exactly like a check that passes. For each edge below there are two
tests: one runs the real check and expects it to fire on the bad row, and one runs the SAME assertions against a
copy of the module with the check weakened by one operator (`<` to `<=`, a NULL test dropped, ...) and expects
the assertions to fail. The weakened copy lives only inside the test (it is built from the module's own source
text); nothing is written to src/."""

import sys
import types
from pathlib import Path

import duckdb
import pytest
import yaml

from cricket_sim import ball_layers, contracts


def weakened(real, old: str, new: str):
    """A copy of module `real` whose source has `old` replaced by `new` (exactly one occurrence)."""
    src = Path(real.__file__).read_text()
    assert src.count(old) == 1, f"{old!r} is not in {real.__name__} exactly once"
    mod = types.ModuleType(real.__name__ + "_weakened")
    mod.__package__, mod.__file__ = real.__package__, real.__file__
    sys.modules[mod.__name__] = mod
    try:
        exec(compile(src.replace(old, new), real.__file__, "exec"), mod.__dict__)
    except BaseException:
        del sys.modules[mod.__name__]
        raise
    return mod


@pytest.fixture(autouse=True)
def _clean():
    """_world() points the contracts module at a one-table world; put the real folders back afterwards."""
    saved = contracts.CONTRACTS, contracts.LAYERS
    yield
    contracts.CONTRACTS, contracts.LAYERS = saved
    contracts.load_all.cache_clear()
    for name in [n for n in sys.modules if n.endswith("_weakened")]:
        del sys.modules[name]


# ---- contract operators: the boundary value itself ----------------------------------------------------------

#   operator, value, bound, does the check pass?
OPERATOR_EDGES = [
    ("mustBe", 5, 5, True), ("mustBe", 6, 5, False),
    ("mustNotBe", 5, 5, False), ("mustNotBe", 6, 5, True),
    ("mustBeGreaterThan", 5, 5, False), ("mustBeGreaterThan", 6, 5, True),
    ("mustBeGreaterOrEqualTo", 5, 5, True), ("mustBeGreaterOrEqualTo", 4, 5, False),
    ("mustBeLessThan", 5, 5, False), ("mustBeLessThan", 4, 5, True),
    ("mustBeLessOrEqualTo", 5, 5, True), ("mustBeLessOrEqualTo", 6, 5, False),
    ("mustBeBetween", 0, [0, 10], True), ("mustBeBetween", 10, [0, 10], True), ("mustBeBetween", 11, [0, 10], False),
    ("mustBeBetween", -1, [0, 10], False),
    ("mustNotBeBetween", 0, [0, 10], False), ("mustNotBeBetween", 10, [0, 10], False), ("mustNotBeBetween", 11, [0, 10], True),
]


def assert_operator_edges(mod):
    for name, value, bound, passes in OPERATOR_EDGES:
        assert mod.OPERATORS[name](value, bound) is passes, f"{name}({value}, {bound}) should be {passes}"


def test_every_contract_operator_is_right_on_its_boundary_value():
    assert_operator_edges(contracts)
    assert {n for n, *_ in OPERATOR_EDGES} == set(contracts.OPERATORS)       # none left untested


@pytest.mark.parametrize("old, new", [
    ('"mustBeGreaterThan": lambda v, x: v > x', '"mustBeGreaterThan": lambda v, x: v >= x'),
    ('"mustBeGreaterOrEqualTo": lambda v, x: v >= x', '"mustBeGreaterOrEqualTo": lambda v, x: v > x'),
    ('"mustBeLessThan": lambda v, x: v < x', '"mustBeLessThan": lambda v, x: v <= x'),
    ('"mustBeLessOrEqualTo": lambda v, x: v <= x', '"mustBeLessOrEqualTo": lambda v, x: v < x'),
    ('"mustBeBetween": lambda v, x: x[0] <= v <= x[1]', '"mustBeBetween": lambda v, x: x[0] < v < x[1]'),
    ('"mustNotBeBetween": lambda v, x: not x[0] <= v <= x[1]', '"mustNotBeBetween": lambda v, x: not x[0] < v < x[1]'),
    ('"mustBe": lambda v, x: v == x', '"mustBe": lambda v, x: v >= x'),
], ids=["gt_to_ge", "ge_to_gt", "lt_to_le", "le_to_lt", "between_exclusive", "not_between_exclusive", "eq_to_ge"])
def test_the_operator_edge_test_fails_when_an_operator_is_weakened(old, new):
    assert_operator_edges(contracts)                                         # real code: green
    with pytest.raises(AssertionError, match=r"should be (True|False)"):
        assert_operator_edges(weakened(contracts, old, new))                 # sabotaged copy: red


# ---- a contract world of one table, used for the end-to-end checks ------------------------------------------

def _world(tmp_path, mod, rows, primary_key=("a",), quality=None):
    """One silver table `t` (columns a, b, v) holding `rows`, with a contract; `mod` is pointed at it."""
    data, specs = tmp_path / "data", tmp_path / "contracts"
    (data / "silver").mkdir(parents=True, exist_ok=True)
    (specs / "silver").mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"COPY (SELECT * FROM (VALUES {rows}) t(a, b, v)) TO '{data / 'silver' / 't.parquet'}'")
    col = lambda name, ptype, **kw: {"name": name, "logicalType": "string" if ptype == "VARCHAR" else "integer",
                                     "physicalType": ptype, "description": "x", **kw}
    keys = list(primary_key)
    props = [col("a", "VARCHAR", **({"primaryKey": True, "primaryKeyPosition": keys.index("a") + 1} if "a" in keys else {})),
             col("b", "VARCHAR", **({"primaryKey": True, "primaryKeyPosition": keys.index("b") + 1} if "b" in keys else {})),
             col("v", "INTEGER")]
    obj = {"name": "t", "physicalType": "table", "logicalType": "object", "description": "x",
           "dataGranularityDescription": "x", "properties": props}
    if quality:
        obj["quality"] = quality
    doc = {"apiVersion": "v3.1.0", "kind": "DataContract", "id": "cricket-sim.silver.t", "name": "t", "version": "1.0.0",
           "status": "active", "schema": [obj],
           "customProperties": [{"property": "layer", "value": "silver"}, {"property": "kind", "value": "entity"},
                                {"property": "sources", "value": ["cricsheet"]}]}
    (specs / "silver" / "t.yaml").write_text(yaml.safe_dump(doc))
    mod.CONTRACTS = specs
    mod.LAYERS = {"staging": data / "staging", "silver": data / "silver", "gold": data / "gold"}
    mod.load_all.cache_clear()
    return con


def assert_rule_at_its_bound_fails(mod, tmp_path):
    """max(v) = 5 against `mustBeLessThan 5` is a violation; against `mustBeLessThan 6` it is fine."""
    rule = lambda bound: [{"type": "sql", "name": "v_small", "query": "SELECT max(v) FROM ${table}", "mustBeLessThan": bound}]
    con = _world(tmp_path / "at", mod, "('x', 'y', 5)", quality=rule(5))
    with pytest.raises(SystemExit, match="t: v_small = 5, expected mustBeLessThan 5"):
        mod.enforce(con)
    con = _world(tmp_path / "under", mod, "('x', 'y', 5)", quality=rule(6))
    assert mod.enforce(con)["tables"] == 1


def test_a_rule_whose_value_sits_exactly_on_a_strict_bound_fails_the_contract(tmp_path):
    assert_rule_at_its_bound_fails(contracts, tmp_path)


def test_the_strict_bound_test_fails_when_less_than_accepts_equality(tmp_path):
    weak = weakened(contracts, '"mustBeLessThan": lambda v, x: v < x', '"mustBeLessThan": lambda v, x: v <= x')
    with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
        assert_rule_at_its_bound_fails(weak, tmp_path)         # the weakened check lets the bad value through


# ---- primary keys: a NULL in a composite key fails the contract ----------------------------------------------

def assert_null_in_a_composite_key_fails(mod, tmp_path):
    """('x', NULL) has no duplicate (a tuple holding a NULL is distinct), so only the NULL test can catch it."""
    con = _world(tmp_path / "bad", mod, "('x', 'y', 1), ('x', NULL, 2)", primary_key=("a", "b"))
    with pytest.raises(SystemExit, match=r"t: primary key \(\"a\", \"b\"\) has 0 duplicates and 1 NULLs"):
        mod.enforce(con)
    con = _world(tmp_path / "ok", mod, "('x', 'y', 1), ('x', 'z', 2)", primary_key=("a", "b"))
    assert mod.enforce(con)["tables"] == 1


def test_a_null_in_a_composite_primary_key_fails_the_contract(tmp_path):
    assert_null_in_a_composite_key_fails(contracts, tmp_path)


def test_a_null_in_a_single_column_primary_key_fails_the_contract(tmp_path):
    con = _world(tmp_path, contracts, "('x', 'y', 1), (NULL, 'y', 2)", primary_key=("a",))
    with pytest.raises(SystemExit, match=r"t: primary key \(\"a\"\) has \d+ duplicates and 1 NULLs"):
        contracts.enforce(con)


def test_the_null_key_test_fails_when_the_null_check_is_dropped(tmp_path):
    weak = weakened(contracts, "            if dupes or nulls:", "            if dupes:")
    with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
        assert_null_in_a_composite_key_fails(weak, tmp_path)


# ---- the layer registry: a layer row with no Cricsheet ball ---------------------------------------------------

def _layer_folders(tiny_build, tmp_path, planted=None):
    """Every registry table as a parquet under tmp_path (symlinks to the fixture's), plus `planted` = a SQL
    expression list to append to one sparse layer: {table: 'SELECT ...'} replaces that table's file."""
    folders = {}
    for table in ball_layers.LAYERS:
        folder = tmp_path / ("gold" if table.startswith("int_") else "silver")
        folder.mkdir(parents=True, exist_ok=True)
        source = tiny_build / folder.name / f"{table}.parquet"
        target = folder / f"{table}.parquet"
        if planted and table in planted:
            duckdb.connect().execute(f"COPY ({planted[table].format(src=source)}) TO '{target}' (FORMAT parquet)")
        else:
            target.symlink_to(source)
        folders[table] = folder
    con = duckdb.connect()
    con.execute(f"CREATE VIEW balls AS SELECT * FROM read_parquet('{tiny_build}/silver/balls.parquet')")
    return con, folders


ORPHAN_ROW = ("SELECT * FROM read_parquet('{src}') UNION ALL "
              "(SELECT * REPLACE ('no-such-match' AS match_id) FROM read_parquet('{src}') LIMIT 1)")
DUPLICATE_ROW = "SELECT * FROM read_parquet('{src}') UNION ALL (SELECT * FROM read_parquet('{src}') LIMIT 1)"


def assert_registry_flags_bad_layers(mod, tiny_build, tmp_path):
    con, folders = _layer_folders(tiny_build, tmp_path / "clean")
    mod.check_registry(con, folders)                                                   # the real layers are fine
    con, folders = _layer_folders(tiny_build, tmp_path / "orphan", {"cricinfo_ball_labels": ORPHAN_ROW})
    with pytest.raises(SystemExit, match="cricinfo_ball_labels: 0 duplicate keys, 1 rows with no Cricsheet ball"):
        mod.check_registry(con, folders)
    con, folders = _layer_folders(tiny_build, tmp_path / "dupe", {"cricinfo_ball_labels": DUPLICATE_ROW})
    with pytest.raises(SystemExit, match="cricinfo_ball_labels: 1 duplicate keys, 0 rows with no Cricsheet ball"):
        mod.check_registry(con, folders)


def test_registry_check_flags_a_layer_row_with_no_cricsheet_ball_and_duplicate_keys(tiny_build, tmp_path):
    assert_registry_flags_bad_layers(ball_layers, tiny_build, tmp_path)


def test_the_registry_test_fails_when_orphan_rows_are_ignored(tiny_build, tmp_path):
    weak = weakened(ball_layers, "if distinct != rows or orphans:", "if distinct != rows:")
    with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
        assert_registry_flags_bad_layers(weak, tiny_build, tmp_path)


def test_the_registry_check_wants_exactly_one_row_per_ball_in_a_dense_layer(tiny_build, tmp_path):
    drop_one = "SELECT * FROM read_parquet('{src}') WHERE NOT (match_id, innings, delivery_number) = " \
               "(SELECT (match_id, innings, delivery_number) FROM read_parquet('{src}') LIMIT 1)"
    con, folders = _layer_folders(tiny_build, tmp_path, {"int_ball_state_pre": drop_one})
    with pytest.raises(SystemExit, match=r"int_ball_state_pre: \d+ rows, expected exactly one per ball"):
        ball_layers.check_registry(con, folders)


# ---- the leak check: a chase that carries on after exactly reaching its target ----------------------------------

def _leak_connection(tiny_build, replace: str | None, replace_innings: str | None = None):
    """The fixture's real layers; `replace` is a SELECT * REPLACE (...) body applied to int_ball_state_pre, and
    `replace_innings` one applied to the raw silver innings table (the stated target before any relabelling)."""
    con = duckdb.connect()
    replacements = {"int_ball_state_pre": replace, "innings": replace_innings}
    for name, folder in (("balls", "silver"), ("innings", "silver"), ("matches", "silver"), ("cricinfo_ball_pre", "silver"),
                         ("cricinfo_ball_post", "silver"), ("int_ball_state_pre", "gold")):
        path = f"{tiny_build}/{folder}/{name}.parquet"
        body = f"SELECT * REPLACE ({replacements[name]}) FROM read_parquet('{path}')" if replacements.get(name) \
            else f"SELECT * FROM read_parquet('{path}')"
        con.execute(f"CREATE VIEW {name} AS {body}")
    return con


PLANTED = {
    # the first ball of a chase that has a known target now needs exactly 0 more runs: the target was reached
    "target_reached": "CASE WHEN (match_id, innings, delivery_number) = ('433606', 2, 5) THEN 0 ELSE runs_required_before END "
                      "AS runs_required_before",
    # ... or has exactly 0 balls left
    "horizon_reached": "CASE WHEN (match_id, innings, delivery_number) = ('433606', 2, 5) THEN 0 ELSE balls_remaining_before END "
                       "AS balls_remaining_before",
}


# The raw stated target of that chase is cut to 1 run: every later ball then shows the batting side past the
# target Cricsheet states, while the published state still says 'set' with the real target. Only a check that reads
# the stated target before the relabelling can see this; the published columns are consistent with themselves.
STATED_TARGET_CUT = "CASE WHEN (match_id, innings) = ('433606', 2) THEN 1 ELSE target_runs END AS target_runs"


def assert_leak_check_flags_a_finished_chase(mod, tiny_build):
    clean = mod.check_no_leak(_leak_connection(tiny_build, None))
    assert clean["chases_exceeding_known_target_or_horizon"] == 0 and clean["state_continuity_breaks"] == 0
    for replace in PLANTED.values():
        with pytest.raises(SystemExit, match=r"leak check failed: 1 chases continue after reaching a 'known' target"):
            mod.check_no_leak(_leak_connection(tiny_build, replace))
    with pytest.raises(SystemExit, match=r"leak check failed: 1 chases continue after reaching a 'known' target"):
        mod.check_no_leak(_leak_connection(tiny_build, None, replace_innings=STATED_TARGET_CUT))


def test_leak_check_flags_a_chase_that_goes_on_after_reaching_its_target_or_its_balls(tiny_build):
    con = _leak_connection(tiny_build, None)
    # the planted ball must really be a ball of a chase with a known target, or the case proves nothing
    assert con.execute("SELECT target_status, runs_required_before > 0, balls_remaining_before > 0 FROM int_ball_state_pre "
                       "WHERE (match_id, innings, delivery_number) = ('433606', 2, 5)").fetchone() == ("set", True, True)
    assert_leak_check_flags_a_finished_chase(ball_layers, tiny_build)


def test_the_stated_target_case_is_invisible_to_the_published_state_alone(tiny_build):
    """The cut stated target leaves every published column consistent: the old check on int_ball_state_pre alone
    (target_status = 'set' and the published target reached) sees nothing, so the raw clause is not redundant."""
    con = _leak_connection(tiny_build, None, replace_innings=STATED_TARGET_CUT)
    assert con.execute("SELECT count(*) FROM int_ball_state_pre WHERE target_status = 'set' "
                       "AND (runs_required_before <= 0 OR balls_remaining_before <= 0)").fetchone() == (0,)
    assert con.execute("SELECT target_runs FROM innings WHERE (match_id, innings) = ('433606', 2)").fetchone() == (1,)


@pytest.mark.parametrize("old, new", [
    ("(runs_required_before <= 0 OR balls_remaining_before <= 0)", "(runs_required_before < 0 OR balls_remaining_before <= 0)"),
    ("(runs_required_before <= 0 OR balls_remaining_before <= 0)", "(runs_required_before <= 0 OR balls_remaining_before < 0)"),
    # the raw clause dropped: a chase past its stated target may keep its published target
    ("(s.target_runs IS NOT NULL OR s.innings_balls_limit IS NOT NULL)", "false"),
    # the raw clause judged on the published target instead of the stated one
    ("WHERE s.innings_runs_before >= t.target OR", "WHERE s.innings_runs_before >= s.target_runs OR"),
], ids=["target_reached_allowed", "last_ball_allowed", "stated_target_ignored", "stated_target_replaced_by_published"])
def test_the_leak_test_fails_when_the_chase_check_is_weakened(tiny_build, old, new):
    weak = weakened(ball_layers, old, new)
    with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
        assert_leak_check_flags_a_finished_chase(weak, tiny_build)
