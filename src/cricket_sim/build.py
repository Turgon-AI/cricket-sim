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
    """The commit and whether the tree is dirty. If git is missing or fails (a container, a copied tree), the
    state is unknown, so it is recorded as dirty: a build must never claim to be clean when it cannot prove it."""
    def git(*args):
        try:
            result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    from .integrity import source_provenance

    commit, status = git("rev-parse", "HEAD"), git("status", "--porcelain")
    return {"commit": commit or None, "dirty": commit is None or status is None or bool(status),
            **source_provenance(ROOT)}


REQUIRED_SOURCES = bronze.REQUIRED_SOURCES
DIRTY_MESSAGE = ("working tree has uncommitted changes (or git could not report its state), so this build could not be "
                 "reproduced from any commit. Commit first, or pass --allow-dirty (recorded in run.json).")


def _config_hashes() -> dict:
    # config changes facts (scope, venue overrides), so it is part of what a build depends on
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted((ROOT / "config").glob("*.yml"))}


def _outputs() -> list[Path]:
    return sorted(p for folder in (STAGING, SILVER, GOLD) for p in folder.glob("*.parquet"))


SHRINK_LIMIT = 0.01   # a table may lose at most 1% of its rows from one build to the next ...
SHRINK_MIN_ROWS = 5   # ... and a loss of fewer rows than this is never flagged (row floors cover tiny tables)


def shrink_expected() -> set[str]:
    """Tables whose contract says they are meant to shrink (gap lists that empty as sources fill the gaps)."""
    return {f"{c.layer}/{t}" for t, c in contracts.load_all().items() if c.props.get("shrinkExpected")}


def shrink_check(tables: dict, previous: dict | None, allow_shrink: list[str], expected: set[str]) -> dict:
    """Compare row counts with the build being replaced. Contract floors catch a gross loss; this catches a
    snapshot or step that quietly dropped rows. A table marked shrinkExpected in its contract may shrink; any other
    deliberate drop (narrower scope, a dedup fix) needs --allow-shrink naming that table (or `all`)."""
    allowed = {a if "/" in a else next((n for n in tables if n.split("/", 1)[1] == a), a) for a in allow_shrink}
    unknown = sorted(a for a in allowed if a != "all" and a not in tables)
    if unknown:
        raise SystemExit(f"--allow-shrink names no built table: {', '.join(unknown)}")
    if previous is None:
        return {"previous_build": None, "allowed": sorted(allowed), "shrunk": {}}
    shrunk = {}
    for name, t in tables.items():
        before = previous["tables"].get(name, {}).get("rows")
        if before is not None and before - t["rows"] >= SHRINK_MIN_ROWS and t["rows"] < before * (1 - SHRINK_LIMIT):
            why = "expected" if name in expected else "allowed" if {"all", name} & allowed else None
            shrunk[name] = {"previous": before, "now": t["rows"], "accepted_as": why}
    refused = {n: v for n, v in shrunk.items() if v["accepted_as"] is None}
    if refused:
        lines = [f"{name}: {v['previous']:,} -> {v['now']:,} rows" for name, v in sorted(refused.items())]
        raise SystemExit(f"{len(refused)} table(s) lost more than {SHRINK_LIMIT:.0%} of their rows since build "
                         f"{previous['build_fingerprint'][:16]}:\n  " + "\n  ".join(lines) +
                         "\nIf this is intended (scope or a fix), rebuild with --allow-shrink TABLE for each "
                         "(recorded in run.json).")
    return {"previous_build": previous["build_fingerprint"], "allowed": sorted(allowed), "shrunk": shrunk}


def build(allow_dirty: bool = False, allow_shrink: list[str] = ()) -> None:
    from .transaction import new_stage, publish, run_worker, writer_lock

    # refuse before the slow build, and before a stage exists, so a refusal leaves nothing behind
    known = {"all", *contracts.load_all(), *(f"{c.layer}/{t}" for t, c in contracts.load_all().items())}
    if set(allow_shrink) - known:
        raise SystemExit(f"--allow-shrink names no table with a contract: {', '.join(sorted(set(allow_shrink) - known))}")
    if _git_state()["dirty"] and not allow_dirty:
        raise SystemExit(DIRTY_MESSAGE)
    with writer_lock(DATA):
        stage = new_stage(DATA, "build")
        (stage / "bronze").symlink_to(DATA / "bronze", target_is_directory=True)
        previous = DATA / "gold" / "run.json"
        run_worker(stage, "build", {"allow_dirty": allow_dirty, "allow_shrink": list(allow_shrink),
                                    "previous_run": str(previous) if previous.exists() else None})

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


def _build(allow_dirty: bool = False, allow_shrink: list[str] = (), previous_run: str | None = None) -> None:
    for folder in (STAGING, SILVER, GOLD):
        folder.mkdir(parents=True, exist_ok=True)
    code = _git_state()
    if code["dirty"] and not allow_dirty:
        raise SystemExit(DIRTY_MESSAGE)
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
    previous = json.loads(Path(previous_run).read_text()) if previous_run else None
    run = {**inputs, "code": code, **outputs, "shrink_check": shrink_check(tables, previous, list(allow_shrink), shrink_expected()),
           "build_fingerprint": hashlib.sha256(json.dumps(outputs, sort_keys=True).encode()).hexdigest()}
    (GOLD / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    con.close()
    print(f"build fingerprint {run['build_fingerprint'][:16]}  ({len(tables)} tables)")
