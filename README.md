# cricket-sim

A reproducible, contract-checked pipeline that turns open ball-by-ball cricket data into an analysis-ready
dataset: every delivery of men's Tests, ODIs, full-member T20Is and the IPL since 2001 (5,871 matches,
3.66M deliveries), with match state computed from earlier balls and checked for pre-ball invariance on the
committed fixture, plus venue, weather and player context, and per-match quality checks. On top of it sits an open ODI ball-by-ball
win-probability and expected-runs model in which **every published probability comes from a model trained
only on earlier seasons** (1,740 held-out matches, scored
against simple baselines).

The structure of this README follows the
[Social Science Data Editors' template](https://social-science-data-editors.github.io/template_README/) for
replication packages.

## Overview

`cricket ingest` downloads each source into an immutable, checksummed snapshot (**bronze**). `cricket build`
turns the pinned snapshots into typed third-party tables (**staging**), conformed facts for every match
(**silver**) and in-scope analysis tables (**gold**: `dim_*`, `fact_*`, `map_*`, `report_*`). Every table is
checked against its data contract on every build. `cricket export` writes the licence-checked dataset export.
A build takes about 4.5 minutes on a laptop. A 10-match fixture of the same pipeline runs in CI on every
push.

Venue context is as-of (`venue_context: asof` in `config/venue_overrides.yml`): a match's venue name, search strings,
geocode and weather placement use only matches up to its date. A build stops if the locked `venues_geo` or `weather`
snapshots lack a request that this needs.

Start with: [docs/DATASET_CARD.md](docs/DATASET_CARD.md) (what the data is and how to use it safely),
[docs/MODEL_CARD.md](docs/MODEL_CARD.md) (the win-probability and expected-runs models, and how well they score),
[docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) (every table and column, generated) and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (layers, contracts, reproducibility).

## Data availability and provenance

| source | used for | licence | in the export? | how to obtain |
|---|---|---|---|---|
| [Cricsheet](https://cricsheet.org) | ball-by-ball, people register, known-missing list | ODC-BY 1.0 | yes | `cricket ingest cricsheet`, `cricsheet_missing` |
| [Wikipedia](https://en.wikipedia.org) season pages (pinned revisions) | reference list of internationals | CC BY-SA 4.0 | yes (2 share-alike files) | `cricket ingest wiki_fixtures` |
| [Wikidata](https://www.wikidata.org) | player birth date and nationality; venue coordinates | CC0 | yes | `cricket ingest wikidata_players`, `venues_geo` |
| [OpenStreetMap](https://www.openstreetmap.org) via Nominatim | venue geocodes | ODbL 1.0 | yes (`dim_venue`) | `cricket ingest venues_geo` |
| [Open-Meteo](https://open-meteo.com) (ECMWF ERA5) | hourly weather | CC BY 4.0 | yes | `cricket ingest weather` |
| [Ari Adaikalam's dataset](https://www.kaggle.com/datasets/ariadaikalam/the-ultimate-ball-by-ball-cricket-dataset) (Cricinfo tracking) | internal analysis only | non-commercial; Cricinfo's terms | **no** | `cricket ingest ari` (Kaggle account) |
| [HawkeyeStats](https://github.com/alittlefitness/HawkeyeStats) | internal analysis only | none stated | **no** | `cricket ingest hawkeye` |

**Rights.** The code is MIT-licensed ([LICENSE](LICENSE)). Each data file carries the licence of its sources,
enforced by the export from lineage, never from intent: see [NOTICE](NOTICE) and
[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md). Raw data is never committed. The only data in git is the small
test fixture in `tests/fixtures/tiny/`: real open data for 10 matches, plus synthetic stand-ins for the two
restricted sources.

## Computational requirements

- **Software:** Python 3.12 and [uv](https://docs.astral.sh/uv/). Every package is pinned in `uv.lock`
  (DuckDB 1.4.1, PyArrow 21).
- **Hardware:** timed on a laptop with 16 GB of RAM. DuckDB is capped at 3 GB, so 8 GB of RAM
  should be enough. About 1.5 GB of disk is used: bronze 0.3 GB, the export 0.33 GB, silver, gold, staging and
  models 0.3 GB, and one previous generation of each (kept under `data/.history/` so a build, model run or export
  can be undone). Each successful run drops the generation before that. `cricket prune` frees the rest: failed
  stages and rollback quarantines, or every previous generation with `--keep 0`.
- **Run time:**
  - `model`: about 9 minutes (19 walk-forward seasons, a locked holdout fit and 1,000 bootstrap resamples per score).
  - `ingest all`: about 1.5 hours for a first full run. Venues and weather take most of that: one geocoding
    request per second (the locked snapshot made 35 new calls and reused 379 cached answers) and 3,897 weather
    windows, one per venue version and match window, because venue context is as-of. Later runs reuse cached
    responses.
  - `build`: about 4.5 minutes.
  - `contracts` and `export`: under 1 minute.
  - Test suite: about 3 minutes on the full data (about 400 tests).
- **Network:** needed only for `ingest`. Builds are offline.

## Instructions to replicators

```bash
uv sync --locked
uv run cricket ingest all       # new bronze snapshots; or `cricket restore` from a backup
uv run cricket verify           # checksums of every pinned snapshot
uv run cricket build            # staging -> silver -> gold, contracts enforced, fingerprints in data/gold/run.json
uv run cricket model            # walk-forward ODI win probability and expected runs -> data/models/ (~6 min)
uv run cricket export           # licence-checked dataset in data/export/<fingerprint>/
uv run pytest                   # fixture build twice, leakage test, famous-match pins, contracts
```

Two builds from the same pinned snapshots and the same `uv.lock` give the same `build_fingerprint`. New
upstream data gives a new snapshot. `cricket lock <source> <snapshot_id>` pins it deliberately, and the lock is
committed.

## Programs and outputs

| output | produced by | code |
|---|---|---|
| `data/bronze/<source>/<snapshot>/` | `cricket ingest` | `ingest.py`, `bronze.py`, `venues/ingest.py`, `context_players.py`, `wiki_fixtures.py` |
| `data/staging/stg_*` | `cricket build` | `staging.py` |
| `data/silver/*` | `cricket build` | `cricsheet_parse.py`, `build.py`, `hawkeye_link.py`, `ball_layers.py`, `context_*.py`, `venues/` |
| `data/gold/*`, `headline.json`, `run.json` | `cricket build` | `gold.py`, `ball_layers.py`, `contracts.py` |
| `data/models/*`, `run.json` | `cricket model` | `models/` (elo, dataset, train, report, run) |
| `data/export/<fingerprint>/` incl. `croissant.json` | `cricket export` | `export.py`, `lineage.py` |
| `docs/DATA_DICTIONARY.md`, `docs/LINEAGE.md` | `cricket docs` | `docs.py` |
| `tests/fixtures/tiny/` | `python scripts/make_fixture.py` | `scripts/make_fixture.py` |

Every headline number in the dataset card is in `data/gold/headline.json`, computed by `gold.py`.

## How correctness is checked

- **Data contracts:** every contract in `contracts/` is validated against the official JSON schema. Every build
  must match the contracts exactly: types, keys, foreign keys and quality rules.
- **Pre-ball state:** every pre-ball value is checked twice:
  - **During the build:** each ball's state must equal the previous state plus the previous ball.
  - **In the tests:** `tests/test_leakage.py` cuts each committed fixture match off at hundreds of points and
    confirms that no tested `pre_ball` column changes.
  - **On production rows:** `scripts/full_check.sh` runs `scripts/preball_invariance.py`, which re-parses 200
    random in-scope matches from the locked Cricsheet zip, cuts each at 20 random points and recomputes the state:
    no `pre_ball` column may change (4,000 cuts, about a minute). The walk-forward model split also has a direct
    test that fit rows come only from prior seasons.
- **Real scorecards:** the 2011, 2019 and 2023 World Cup finals, Dhoni's 2019 run-out, a DLS chase, a super
  over and Test penalty runs are pinned to their published numbers (`tests/test_end_to_end.py`).
- **Reproducibility:** the fixture is built twice and must give identical fingerprints.
- **Licences:** the export refuses any table whose lineage includes a non-redistributable source.

## Full-data check

CI builds only the 10-match fixture, so faults that need the real data (retired-not-out wickets, awarded
matches, associate-team T20Is, a model input that leaks the result) pass CI. Run this on a machine that has the
full bronze data, before a release and after any change to `gold.py`, `cricsheet_parse.py`, `models/` or the scope:

```bash
scripts/full_check.sh [DATA_DIR]    # default: $CRICKET_SIM_FULL_DATA, else ./data (about 20 minutes)
```

It runs `verify`, `build`, the production pre-ball invariance check, `model` and `contracts` on that data, then
checks the full-data-only cases:
retirements, awarded matches never passing quality, the scope recomputed from silver (T20Is need two full-member
teams), the headline and coverage numbers against the tables, and model scores that are too good to be true.
When all of that passes it writes `gold/full_check.json` (the build and model fingerprints, commit and source hash
it vouches for). It pins the build and model it checked: if another `cricket build` or `cricket model` replaced
either while the checks ran, nothing is recorded. `cricket export` refuses a release without a record that matches
the current build and model, so every released export has passed this script; the next build drops the record.
The export carries a copy of the record (`full_check.json`), and its README says when the check passed.
It stops at the first failing step and exits non-zero. It adds `--allow-dirty` when the git tree has changes, or
when git cannot report the tree's state.
Its build step fails if a table lost more than 1% of its rows since the previous build (see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)). If that drop is intended, rebuild once with
`uv run cricket build --allow-shrink TABLE` for each such table, then run the script again.

The script does not export. A new build changes the numbers in the dataset and model cards, so it ends by saying
whether `docs/DATASET_CARD.md` and `docs/MODEL_CARD.md` match the build. If not: run `uv run cricket docs` (it
rewrites their stamp lines), edit the card text to show the new numbers, commit, then `uv run cricket export`.

`cricket export --skip-full-check` exports without the record, for tests. Such an export, and one made with
`--allow-dirty` from uncommitted code, opens its README with a "not a release" warning saying why.

## Credits

Please credit Cricsheet and the other sources as listed in [NOTICE](NOTICE).

## References

- Cricsheet. https://cricsheet.org
- Gebru, T. et al. (2021). Datasheets for Datasets. *Communications of the ACM* 64(12).
- Akhtar, M. et al. (2024). Croissant: A Metadata Format for ML-Ready Datasets. arXiv:2403.19546.
- Bitol / LF AI & Data. Open Data Contract Standard v3.1.0. https://bitol-io.github.io/open-data-contract-standard/
- Wilson, G. et al. (2017). Good enough practices in scientific computing. *PLOS Computational Biology* 13(6).
