"""Every gold and model write reads only tables its contract declares as upstream.

lineage.py takes each table's sources and upstream tables from its contract, and lint only checks that a named
upstream has a contract. Nothing compared the declaration with the SQL, so a new mart that joined a restricted
table under a copied contract header would have shipped under the wrong licence. This test runs the real gold
build on the fixture's silver layer through a connection that records every statement, tokenises the table names
each write reads (after FROM and JOIN, plus read_parquet paths), subtracts CTE and temp-table names, and requires
what is left to be a subset of the contract's declared upstream. The model layer's SQL (models/dataset.py and the
internal Cricinfo comparison) is checked the same way against the model tables built from it."""

import re
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from cricket_sim import contracts, gold
from cricket_sim.models import dataset
from cricket_sim.models import run as model_run

NAME = r"[A-Za-z_][A-Za-z0-9_]*"
REFERENCE = re.compile(rf"\b(?:FROM|JOIN)\s+({NAME})", re.I)
CTE = re.compile(rf"(?:\bWITH|,)\s*({NAME})\s+AS\s*\(", re.I)
PARQUET = re.compile(r"read_parquet\('([^']+)'\)")
COPY = re.compile(r"\s*COPY\s*\((.*)\)\s*TO\s*'([^']+)'", re.S | re.I)

# gold tables written by venues/build.py from Python objects, not SQL: nothing to tokenise
WRITTEN_WITHOUT_SQL = {"int_match_weather_daily", "int_match_play_windows"}


def referenced_tables(sql: str) -> set[str]:
    """The contracted tables a statement reads: names after FROM and JOIN and read_parquet file stems, minus the
    statement's own CTE names. Temp tables and aliases are not contracted tables, so they drop out."""
    ctes = {m.group(1) for m in CTE.finditer(sql)}
    names = {m.group(1) for m in REFERENCE.finditer(sql)} | {Path(p).stem for p in PARQUET.findall(sql)}
    return (names - ctes) & set(contracts.load_all())


def declared_upstream(table: str) -> set[str]:
    return set(contracts.get(table).props.get("upstream") or [])


class Recording:
    """A DuckDB connection that also keeps every statement it ran, and the SELECT body of every COPY ... TO file."""

    def __init__(self, con):
        self.con, self.statements, self.writes = con, [], {}

    def execute(self, sql, *args, **kwargs):
        self.statements.append(sql)
        copy = COPY.match(sql)
        if copy:
            self.writes[Path(copy.group(2)).stem] = copy.group(1)
        return self.con.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.con, name)


@pytest.fixture(scope="module")
def gold_writes(tiny_build, tmp_path_factory) -> dict:
    """{gold table: the SQL that built it} from a real gold build on the fixture's staging and silver layers."""
    con = Recording(duckdb.connect())
    for layer in ("staging", "silver", "gold"):  # the gold ones venues/build.py wrote are read by the headline
        for path in sorted((tiny_build / layer).glob("*.parquet")):
            con.execute(f"CREATE VIEW {path.stem} AS SELECT * FROM read_parquet('{path}')")
    out = tmp_path_factory.mktemp("gold")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(gold, "GOLD", out)
        patch.setattr(gold, "SUBSET", True)
        gold.build_gold(con)
    return con.writes


def test_every_gold_table_built_by_sql_was_captured(gold_writes):
    expected = {t for t, c in contracts.load_all().items() if c.layer == "gold"} - WRITTEN_WITHOUT_SQL
    assert set(gold_writes) == expected


def test_tokeniser_sees_through_ctes_subqueries_and_parquet_paths():
    sql = """WITH totals AS (SELECT * FROM balls b JOIN matches USING (match_id)), x AS (SELECT 1)
             SELECT * FROM totals t LEFT JOIN (SELECT * FROM innings) i USING (match_id)
             JOIN scope USING (match_id) JOIN read_parquet('/tmp/gold/dim_venue.parquet') v USING (venue_id)"""
    assert referenced_tables(sql) == {"balls", "matches", "innings", "dim_venue"}


@pytest.mark.parametrize("table", sorted({t for t, c in contracts.load_all().items() if c.layer == "gold"} - WRITTEN_WITHOUT_SQL))
def test_gold_write_reads_only_its_declared_upstream(gold_writes, table):
    read = referenced_tables(gold_writes[table])
    assert read, f"{table}: no table reference found in its SQL"
    assert read <= declared_upstream(table), \
        f"{table} reads {sorted(read - declared_upstream(table))} that its contract does not declare as upstream"


@pytest.fixture(scope="module")
def model_statements(tiny_build) -> list[str]:
    con = Recording(duckdb.connect())
    dataset.load(con, gold=tiny_build / "gold")
    clean = pd.DataFrame({"match_id": ["x"], "innings": [1], "delivery_number": [1], "wp_model": [0.5], "batting_won": [1.0]})
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(model_run, "SILVER", tiny_build / "silver")
        model_run._cricinfo_comparison(con, clean)
    return [s for s in con.statements if not re.match(r"\s*CREATE (OR REPLACE )?(TEMP )?(VIEW|TABLE)\s+\w+\s*\(", s)
            and not re.match(r"\s*CREATE (OR REPLACE )?VIEW", s)]


def _model_reads(statements, marker: str) -> set[str]:
    found = [s for s in statements if marker in s]
    assert found, f"no model statement contains {marker!r}"
    return set().union(*(referenced_tables(s) for s in found))


def test_model_rows_and_the_ball_models_built_from_them_declare_every_gold_read(model_statements):
    read = _model_reads(model_statements, "model_rows")
    assert read >= {"fact_delivery", "dim_match"}
    for table in ("model_ball_win_probability", "model_ball_outcome"):
        assert read <= declared_upstream(table), f"{table}: model_rows reads {sorted(read - declared_upstream(table))}"


def test_elo_reads_only_what_model_team_elo_declares(model_statements):
    read = _model_reads(model_statements, "winner, result FROM dim_match")
    assert read == {"dim_match"} and read <= declared_upstream("model_team_elo")


def test_the_internal_cricinfo_comparison_declares_its_silver_read(model_statements):
    read = _model_reads(model_statements, "win_probability_before")
    assert read == {"cricinfo_ball_pre"} and read <= declared_upstream("report_model_vs_cricinfo_internal")
