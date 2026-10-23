# Architecture

Medallion on DuckDB + Parquet. Runs on one machine with no services. Every table has a data contract.

```
upstream ─ingest─▶ BRONZE ─build─▶ STAGING ─▶ SILVER ─▶ GOLD ─export─▶ data/export/<fingerprint>/
           (immutable     (typed       (conformed   (int_ building blocks;     (gold dim_/fact_/map_/report_
            snapshots)     3rd-party    facts, all   dim_/fact_/map_/report_    only, licence-checked,
                           CSVs)        matches)     products, in scope)        Croissant metadata)
              │
              └─backup (rclone)─▶ off-machine copy          contracts/<layer>/<table>.yaml checks every table
```

## Layers

| Layer | Where | What | Rules |
|---|---|---|---|
| Bronze | `data/bronze/<source>/<snapshot_id>/` + `_snapshot.json` | Upstream files exactly as fetched (CSVs zstd-compressed, losslessly) | Never edited. Each snapshot is checksummed. New upstream data means a new snapshot, never an overwrite. |
| Staging | `data/staging/stg_*.parquet` | Third-party CSVs (Ari's Cricinfo tracking, Hawk-Eye), one table per file, renamed to snake_case and typed | No joins, no logic, no row filtering. Raw columns are checked against `config/input_schemas.yml` before anything is written. All restricted, never exported. Cricsheet JSON skips staging: `cricsheet_parse.py` is its typed staging step. |
| Silver | `data/silver/*.parquet` | Conformed facts as recorded: Cricsheet tables (matches, innings, balls, wickets, ...), reference lists, the Hawk-Eye link, sparse third-party per-ball layers split by timing class, venue/weather/player context | Keeps **all** matches; scope is never read here. Unknown means NULL, never zero. Each per-ball table holds one timing class and one source family. Nothing we *derive* lives here (no totals, no summaries, no windows). |
| Gold | `data/gold/*.parquet`, `headline.json`, `run.json` | `int_*`: our derivations over all matches (pre-ball match state, innings state, quality checks, per-player match stats, daily play-window weather). `dim_*` / `fact_*` / `map_*` / `report_*`: in-scope products, including point-in-time player features | Scope (`config/scope.yml`) is applied here only. Only `dim_`/`fact_`/`map_`/`report_` tables can be exported. |
| Models | `data/models/*.parquet`, `run.json` | `cricket model`: walk-forward ODI win probability, ball outcomes, Elo, evaluation reports | Reads gold only. See docs/MODEL_CARD.md. |

### Naming

One rule per kind of table, so a name says where the data came from and whether it can leave the machine:

| pattern | meaning | examples |
|---|---|---|
| `stg_<source>_<file>` | staging: one typed third-party file | `stg_ari_tracking`, `stg_hawkeye_balls` |
| plural noun | silver fact we conform from Cricsheet or canonicalise ourselves (open) | `matches`, `balls`, `players`, `venues` |
| `<source>_<thing>` | silver data restated from a third party, named after it | `cricinfo_ball_labels`, `hawkeye_ball_delivery`, `wikipedia_fixtures`, `cricsheet_missing` |
| `<source>_ball_<timing>` | a third-party per-ball layer, one timing class each | `cricinfo_ball_pre`, `cricinfo_ball_post`, `hawkeye_ball_landing` |
| `int_` / `dim_` / `fact_` / `map_` / `report_` | gold: intermediate / dimension / fact / crosswalk / report | `int_ball_state_pre`, `fact_player_match_form` |
| `model_` | model outputs | `model_ball_win_probability` |

Restriction is never a suffix: `cricinfo_` and `hawkeye_` tables are restricted because their sources are, and
`lineage.py` enforces that from each contract's `sources`.

`data/warehouse.duckdb` holds a view per table (names are unique across layers), for ad-hoc queries.

## Data contracts

`contracts/<layer>/<table>.yaml` is one [Open Data Contract Standard](https://bitol-io.github.io/open-data-contract-standard/) v3.1.0 file per table: what one row is, the key, every column's meaning, type and unit, the upstream sources and tables, and the checks the data must pass. It is the single source of truth:

- **Build:** every table must match its contract exactly (column names, order, types), keys must be unique and non-null, required columns non-null, foreign keys must resolve, and every quality rule must hold. Any failure stops the build. A table without a contract, or a contract without a table, also stops it.
- **Generated from contracts:** licence lineage (`lineage.py`), the per-ball timing registry (`ball_layers.LAYERS`), [DATA_DICTIONARY.md](DATA_DICTIONARY.md), [LINEAGE.md](LINEAGE.md) and the export's README and `croissant.json`.
- **Validated:** `cricket contracts` checks every file against the official ODCS JSON schema (vendored in `contracts/_schema/`) and the project's own rules, then enforces them against the built data.

## Timing labels: what a model may see

Per-ball columns carry a timing class in their contract. In `fact_delivery`:
- **`pre_ball`:** knowable before the ball is bowled. Safe as a model input.
- **`hindsight`:** computed from earlier balls, but blanked or flagged using how the innings ended (rain
  revision, a shortened innings, absentees). Use it to choose training rows, never as an input.
- **`outcome`:** what happened on the ball.

`tests/test_leakage.py` checks the labels on the committed fixture. It cuts each fixture match off at hundreds
of points, as if it were live, and recomputes: no tested `pre_ball` value may change, and every `hindsight`
column must actually change somewhere. `scripts/preball_invariance.py` does the same on production rows (200
random in-scope matches, 20 cuts each) and runs inside `scripts/full_check.sh`. The walk-forward fit split is
separately checked to contain only seasons before its test season.

## Test fixture and CI

`tests/fixtures/tiny/` is a committed 10-match bronze, made by `scripts/make_fixture.py`:
- the 2011, 2019 and 2023 World Cup finals and the 2019 semi;
- a DLS chase, two Tests, a super-over T20I, the 2019 IPL final;
- one women's match, which scope must drop.

It holds real open data, plus synthetic stand-ins in the exact raw formats for the two restricted sources.

Setting `CRICKET_SIM_DATA` and `CRICKET_SIM_SNAPSHOTS` points the unchanged CLI at it. Its `SUBSET` marker skips
only the checks that need complete data, such as stale venue overrides. `cricket export` on the fixture needs
`--allow-dirty`: the dataset and model cards are stamped with the full build's numbers, so a fixture export
without it refuses, by design.

`.github/workflows/ci.yml` runs the whole suite on every push:
- the fixture built twice;
- contracts;
- the leakage test;
- the pinned real scorecards;
- the export.

## Reproducibility contract

The same locked snapshots and the same `uv.lock` give the same `build_fingerprint` in `data/gold/run.json`.

1. **Pinned snapshots:** builds read only `config/snapshots.lock.yml`, which is committed. Changing it is deliberate: `cricket lock <source> <id>`.
2. **Pinned software:** `uv.lock` pins every package, and Python is pinned to 3.12.
3. **Determinism:** inputs are sorted, outputs are written with `ORDER BY`, and nothing uses wall-clock time or randomness.
4. **Proof:** every table gets a fingerprint (row count plus an order-independent content hash). Run `cricket build` twice and compare.

## Built to absorb changing requirements

- **New source:** add an entry to `config/sources.yml`, an ingester, a staging table if it is a CSV, silver SQL, and a contract per new table. Nothing else changes.
- **Wider scope** (women, BBL, ...): edit `config/scope.yml` and rebuild. Silver already holds everything.
- **Different backup target:** point `config/remote.yml` at any rclone remote, such as S3, Backblaze or Dropbox.
- **Exporting:** `redistributable` in `sources.yml` marks what may leave the machine. Only redistributable sources and our own derived context go into the export.

## Safety rails

- **Bronze:**
  - `verify` fails on missing, altered or unlisted files.
  - `config/snapshots.pins.yml` pins each locked manifest's hash.
  - Ingest writes atomically and records failures. A snapshot with gaps is never sealed silently.
- **Build:**
  - It refuses dirty code unless you pass `--allow-dirty`, which is recorded.
  - If it fails, the previous staging, silver and gold are restored, and the failed outputs are kept in
    `data/<layer>.failed/` for inspection.
  - Upstream schemas are pinned in `config/input_schemas.yml`.
  - It fails if any table breaks its contract, a per-ball layer gains an unclassified column or a ball Cricsheet does not have, the pre-ball state loses its one-row-per-ball shape, the leak check fails, or the crosswalk is not one-to-one.
  - Venue sanity checks (country box, elevation, sea, timezone) fail it too.
- **Export:** licences come from lineage (`lineage.py`), never from intent. A table built from Cricinfo or Hawk-Eye data cannot be exported, and a file's declared licence must match its strictest source.
- **Validation:** see `docs/VALIDATION.md`.

## Commands

```
uv run cricket ingest all|cricsheet|ari|hawkeye   # new bronze snapshot(s)
uv run cricket status                             # locked snapshots, local health
uv run cricket verify                             # checksum locked bronze
uv run cricket build                              # staging + silver + gold, contracts enforced, fingerprints
uv run cricket contracts                          # lint contracts, check the built tables against them
uv run cricket docs                               # regenerate DATA_DICTIONARY.md and LINEAGE.md
uv run cricket export                             # licence-checked dataset export -> data/export/
uv run cricket backup | restore                   # rclone to/from off-machine copy
```
