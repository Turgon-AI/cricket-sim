"""Rebuild staging, silver and gold from the locked bronze snapshots, then fingerprint every table.

Same locked snapshots + same uv.lock => identical fingerprints. `cricket build` twice proves it.
"""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import duckdb

from . import ball_layers, bronze, contracts
from .cricsheet_parse import parse_zip
from .gold import build_gold, validate_scope
from .hawkeye_link import build_hawkeye_link
from .context_players import build_player_context
from .context_venues import build_venue_context
from .missing import build_cricsheet_missing
from .wiki_fixtures import build_wiki_fixtures
from .paths import DATA, GOLD, ROOT, SILVER, STAGING, load_yaml, snapshot_dir
from .staging import build_staging

DUCKDB_SETTINGS = {"memory_limit": "3GB", "threads": 4, "preserve_insertion_order": True}


def connect(path=":memory:") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(path)
    for key, value in DUCKDB_SETTINGS.items():
        con.execute(f"SET {key} = '{value}'")
    return con


def write_table(con, sql: str, dest: Path, order_by: str) -> None:
    con.execute(f"COPY (SELECT * FROM ({sql}) ORDER BY {order_by}) TO '{dest}' (FORMAT parquet, COMPRESSION zstd)")


def build_silver(con, lock: dict) -> None:
    """Staging (typed third-party CSVs), then silver: Cricsheet parsed into conformed tables, reference
    lists, the Hawk-Eye link, third-party per-ball layers and the venue/weather/player context."""
    build_staging(con, lock)
    cs = snapshot_dir("cricsheet", lock["cricsheet"])
    with tempfile.TemporaryDirectory(dir=DATA) as tmp:
        parts = parse_zip(cs / "all_json.zip", Path(tmp))
        keys = {"balls": "match_id, innings, delivery_number",
                "wickets": "match_id, innings, delivery_number, wicket_number",
                "innings": "match_id, innings", "matches": "match_id",
                "match_players": "match_id, team, player",
                "powerplays": "match_id, innings, powerplay_number",
                "reviews": "match_id, innings, delivery_number",
                "replacements": "match_id, innings, delivery_number, replacement_number"}
        for table, files in parts.items():
            file_list = ", ".join(f"'{p}'" for p in files)
            write_table(con, f"SELECT * FROM read_parquet([{file_list}])", SILVER / f"{table}.parquet", keys[table])

    # the Cricsheet register is already tidy (all identifiers are text), so it goes straight to silver
    write_table(con, f"SELECT * FROM read_csv('{cs / 'people.csv'}', all_varchar=true)", SILVER / "people.parquet", "identifier")
    people_cols = [f"{n}:{t}" for n, t, *_ in con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{SILVER / 'people.parquet'}')").fetchall()]
    if people_cols != load_yaml("input_schemas.yml")["people"]:
        raise SystemExit("people: Cricsheet register columns changed; review, then update config/input_schemas.yml")

    build_cricsheet_missing(snapshot_dir("cricsheet_missing", lock["cricsheet_missing"]))
    build_wiki_fixtures(snapshot_dir("wiki_fixtures", lock["wiki_fixtures"]))
    for table in ["balls", "wickets", "innings", "matches", "match_players", "powerplays", "reviews", "replacements",
                  "people", "cricsheet_missing", "wikipedia_fixtures"]:
        con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{SILVER / table}.parquet')")

    build_hawkeye_link(con)
    for table, sql in ball_layers.SQL.items():
        write_table(con, sql, SILVER / f"{table}.parquet", "match_id, innings, delivery_number")
        con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{SILVER / table}.parquet')")
    con.execute("DROP TABLE hawkeye_ball_tracking")

    build_player_context(con)
    build_venue_context(con)


def fingerprint(con, path: Path) -> dict:
    """Row count, column names and types, and an order-independent content hash.

    hash() alone ignores column names and integer widths, so the schema is fingerprinted too.
    """
    schema = [f"{name}:{dtype}" for name, dtype, *_ in con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()]
    rows, digest = con.execute(
        f"SELECT count(*), coalesce(sum(hash(t)::HUGEINT), 0)::VARCHAR FROM read_parquet('{path}') t").fetchone()
    return {"rows": rows, "schema_sha256": hashlib.sha256("|".join(schema).encode()).hexdigest()[:16],
            "content_hash": digest, "sha256": bronze.sha256(path)}


def _git_state() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    from .integrity import source_provenance

    return {"commit": git("rev-parse", "HEAD") or None, "dirty": bool(git("status", "--porcelain")),
            **source_provenance(ROOT)}


REQUIRED_SOURCES = bronze.REQUIRED_SOURCES


def _config_hashes() -> dict:
    # config changes facts (scope, venue overrides), so it is part of what a build depends on
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted((ROOT / "config").glob("*.yml"))}


def _outputs() -> list[Path]:
    return sorted(p for folder in (STAGING, SILVER, GOLD) for p in folder.glob("*.parquet"))


def build(allow_dirty: bool = False) -> None:
    from .transaction import new_stage, publish, run_worker, writer_lock

    with writer_lock(DATA):
        stage = new_stage(DATA, "build")
        (stage / "bronze").symlink_to(DATA / "bronze", target_is_directory=True)
        run_worker(stage, "build", {"allow_dirty": allow_dirty})

        def warehouse(name):
            if name != "warehouse.duckdb":
                return
            con = duckdb.connect(str(stage / name))
            try:
                for path in _outputs():
                    con.execute(f"CREATE VIEW {path.stem} AS SELECT * FROM read_parquet('{path}')")
            finally:
                con.close()

        publish(DATA, stage, ["staging", "silver", "gold", "warehouse.duckdb"], prepare=warehouse)


def _build(allow_dirty: bool = False) -> None:
    for folder in (STAGING, SILVER, GOLD):
        folder.mkdir(parents=True, exist_ok=True)
    code = _git_state()
    if code["dirty"] and not allow_dirty:
        raise SystemExit("working tree has uncommitted changes, so this build could not be reproduced from any "
                         "commit. Commit first, or pass --allow-dirty (recorded in run.json).")
    validate_scope(load_yaml("scope.yml"))  # refuse a bad scope before any slow step
    lock = bronze.read_lock()
    missing = [s for s in REQUIRED_SOURCES if s not in lock]
    if missing:
        raise SystemExit(f"no locked snapshot for {missing}; run `cricket ingest` first")
    problems = [p for s, sid in lock.items() for p in bronze.verify(s, sid)]
    if problems:
        raise SystemExit("bronze does not verify (try `cricket restore`):\n" + "\n".join(problems))

    con = connect()
    print("building staging and silver")
    build_silver(con, lock)
    print("building gold")
    headline = build_gold(con)
    ball_layers.check_registry(con, {t: GOLD if t.startswith("int_") else SILVER for t in ball_layers.LAYERS})
    headline["leak_check"] = ball_layers.check_no_leak(con)
    problems = contracts.lint()
    if problems:
        raise SystemExit("contracts do not lint:\n" + "\n".join(problems))
    headline["contract_checks"] = contracts.enforce(con)
    (GOLD / "headline.json").write_text(json.dumps(headline, indent=2) + "\n")

    tables = {f"{p.parent.name}/{p.stem}": fingerprint(con, p) for p in _outputs()}
    inputs = {
        "snapshots": lock,
        "snapshot_pins": bronze.read_pins(),
        "config": _config_hashes(),
        "uv_lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        "duckdb_version": duckdb.__version__,
    }
    outputs = {"tables": tables,
               "headline_sha256": hashlib.sha256((GOLD / "headline.json").read_bytes()).hexdigest()}
    run = {**inputs, "code": code, **outputs,
           "build_fingerprint": hashlib.sha256(json.dumps(outputs, sort_keys=True).encode()).hexdigest()}
    (GOLD / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    con.close()
    print(f"build fingerprint {run['build_fingerprint'][:16]}  ({len(tables)} tables)")
