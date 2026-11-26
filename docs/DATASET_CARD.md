# Dataset card: Cricket ball-by-ball, with match state, context and quality checks

Structured after *Datasheets for Datasets* (Gebru et al., 2021). Describes the dataset export
(`cricket export` → `data/export/<build fingerprint>/`). Every number here comes from `data/gold/headline.json`
of build `abf009871cfe15fd` unless stated; the column-level documentation is the export's README.md (generated from
the data contracts) and `croissant.json` (MLCommons Croissant 1.0).

## Motivation

**Why was it created?** To be a trustworthy foundation for ball-level cricket modelling, starting with an open,
calibrated ODI win-probability and expected-runs model. Existing open
ball-by-ball data (Cricsheet) records what happened; this dataset adds state features computed before each ball.
Truncation checks confirm that pre-ball values do not change when later balls are removed, on the committed
fixture in every CI run and on 200 random production matches cut at 20 points each in every full-data check. The
dataset also adds venue, weather, player context and per-match quality checks.

**Affiliation:** not affiliated with Cricsheet, the ICC, ESPNcricinfo or any team.

## Composition

**What does an instance represent?** The central table, `fact_delivery`, has one row per delivery bowled
(wides and no-balls included). Around it: one row per match (`dim_match`), innings (`fact_innings`), dismissal
(`fact_dismissal`), player (`dim_player`), venue (`dim_venue`, and `dim_venue_version` for each dated version) and venue-hour of weather
(`fact_match_weather_hourly`), plus identifier crosswalks (`map_*`) and quality/coverage reports (`report_*`).
`fact_player_match_form` gives each player's record before every match: career to date and the last 365
days, batting and bowling, death overs, days since the last match and age. It is built from strictly earlier
matches only, so it is safe as a model input. `report_player_careers` holds whole-snapshot totals, for browsing
only. For ODIs from 2008, the model tables add, per ball, the batting side's win probability before and after it
(`model_ball_win_probability`) and the outcome probabilities and expected runs for an average batter
(`model_ball_outcome`), each from a model trained only on earlier seasons, plus pre-match Elo
(`model_team_elo`) and the evaluation reports (`report_model_*`). See docs/MODEL_CARD.md.

**How many instances?** 9,540 matches and 4,507,621 deliveries:

| segment | matches | first | last |
|---|---|---|---|
| Test | 896 | 2001-12-19 | 2026-10-09 |
| ODI | 2,596 | 2002-06-27 | 2026-10-13 |
| T20I (both teams ICC full members) | 1,136 | 2005-02-17 | 2026-10-13 |
| IPL | 1,243 | 2008-04-18 | 2026-05-31 |
| T20 Blast (England) | 1,448 | 2015-05-15 | 2026-07-18 |
| BBL (Big Bash League) | 539 | 2015-12-18 | 2026-01-25 |
| BPL (Bangladesh Premier League) | 414 | 2015-11-22 | 2026-01-23 |
| CPL (Caribbean Premier League) | 395 | 2015-06-20 | 2026-09-20 |
| PSL (Pakistan Super League) | 357 | 2016-02-04 | 2026-05-03 |
| LPL (Lanka Premier League) | 143 | 2020-11-26 | 2026-08-08 |
| ILT20 (International League T20) | 134 | 2023-01-13 | 2026-01-04 |
| SA20 | 130 | 2023-01-10 | 2026-01-25 |
| MLC (Major League Cricket) | 109 | 2023-07-13 | 2026-07-18 |

**Is it a sample?** It is every match Cricsheet holds in that scope (men's, from 2001-01-01; the nine leagues
after the IPL from 2015-04-01), which is not every match played. For the leagues there is no outside list: the
only check is Cricsheet's own missing list, which names 21 league matches in scope (T20 Blast 5, BPL 6, CPL 5,
BBL 3, PSL 2), so each league is at least 98.6% held by that list. A league with no entry on the list (the IPL,
LPL, ILT20, SA20, MLC) may simply be one Cricsheet does not track gaps for. `report_league_seasons` counts
matches per league and year. `report_international_coverage` reconciles Cricsheet against every numbered international listed
on Wikipedia's season pages and gives a named reason for each gap. The 3,321 listed ODIs split like this:

| ODI status | count |
|---|---|
| `in_cricsheet` | 2,567 |
| `cricsheet_known_missing` (on Cricsheet's own missing list) | 313 |
| `afghanistan_withheld_by_cricsheet` (188 involve Afghanistan, which Cricsheet withholds) | 188 |
| `before_cricsheet_coverage` | 176 |
| `missing_unlisted_2002_sparse_year` | 59 |
| `no_result_possibly_no_play` | 10 |
| `after_snapshot` (newer than the Cricsheet snapshot, so not a gap) | 6 |
| `abandoned_without_a_ball` | 1 |
| `missing_unlisted` (no known reason) | 1 |

The same report has 1,135 of the listed T20Is and 895 of the listed Tests in Cricsheet, with 2 T20Is and no Tests
after the snapshot. The match between a Cricsheet match and a reference row is strict: the same two teams and a
start date within one day, one-to-one, with 14 reviewed overrides in `config/link_overrides.yml` for cases the
rule cannot make. `map_match_ids` holds 4,597 such links. The gaps are not random: early years, Afghanistan and
smaller nations are under-represented.

**What does each instance consist of?** For every delivery:
- who batted, bowled and was at the other end;
- the over and ball position;
- the match state *before* the ball;
- the outcome: runs by type, extras, boundaries, wickets and reviews.

Each `fact_delivery` column is labelled, in the contract and the README, with one of two timings:
- `pre_ball`: knowable before the ball was bowled.
- `outcome`: what happened on the ball.

A third timing, `hindsight` (built from earlier balls, but blanked or flagged using how the innings ended), is
kept in its own table, `fact_delivery_hindsight`, on the same ball key, so `SELECT *` on `fact_delivery` never
returns it.

**Is there a label or target?** Not one. Typical targets are runs on the ball, wicket on the ball, the innings
total, or the match winner (`dim_match`).

**Is anything missing?**
- **Weather:** 99.98% of matches have it (two matches at a venue that could not be geocoded have none). Two
  league grounds (Pomona, 7 MLC matches; Geelong, 1 BBL match) are placed at the city centre, not the ground.
- **Player birth dates:** 87.8% of in-scope players have one (Wikidata). League-only players have fewer.
- **Player country:** `dim_player.country` is kept as built. `country_conflict` flags the 206 in-scope players
  whose Wikidata citizenship names no national side they played for in Cricsheet; the country itself is not
  changed.
- **Player country over time:** `dim_player.country` is the latest national side. `dim_player_team` gives each
  player's spells with one side (2,422 spells for 2,380 capped in-scope players; 40 played for more than one
  country, e.g. Roelof van der Merwe for South Africa, then the Netherlands), so nationality can be taken as of any
  match date.
- **Player physicals, batting hand and bowling style:** absent. No open source records them with usable
  coverage.
- **Field placements, pitch reports and ball-tracking:** not included. They are not openly available.
- **Rain-revised chases:** 419 innings carry `target_status = 'revised_timing_unknown'`. Cricsheet records only
  the final revised target, not when it was set.

**Are there errors or noise?** The nine arithmetic and consistency checks in `report_match_quality` pass for
9,536 of 9,540 scoped matches (99.96%). This is an internal-consistency rate, not a scorecard reconciliation
accuracy estimate. The four flagged matches are: a 7-ball over in Zimbabwe v South Africa (T20I, 2022), the 2006 Oval Test, which was awarded
to England when Pakistan did not take the field, and two matches decided by post-innings penalty runs that are on no ball
(`result_in_ball_stream = false`): Canada v Nepal (ODI, 2024), which Nepal won through 5 such runs, and Kent v
Somerset (T20 Blast, 2018), where Kent's 6 such runs turned a Somerset win into a Kent win by 5 runs. All four stay
in the data with `passes_quality = false`.

**Checked against an outside source:** `report_result_check` compares the result of every international linked
to the Wikipedia reference list (4,597 matches) with the pinned season page. 4,582 agree on the kind of result,
the winner and the margin. None differ on the kind of result or the winner. 15 differ only on the margin; in 14 of
them Cricsheet's own innings totals give Cricsheet's margin (e.g. Australia by 277 runs over West Indies in June 2015, which
Wikipedia lists as 227), so the page is the likely error; the 15th is a D/L result, where the margin cannot be
recomputed. The build fails if a result or winner ever disagrees. Cricsheet itself is volunteer-built and occasionally
corrected; `dim_match.cricsheet_revision` records the revision used. No in-scope chase is flagged `inconsistent_with_balls` any more: the three that were had a 7-ball
over that Cricsheet lists as miscounted, which the ball limit now allows for.

A pass rate means little unless the checks can fail, so `headline.json` publishes how often each one passed, failed
or was not checkable on the matches in scope (a check is not checkable where its precondition is absent: no
scheduled overs, a rain rule, no winner). Four of the nine cannot fail in a completed build: a data contract rule
upstream tests the same predicate and stops the build first, so their zero failures below are a consequence of
the contracts, not evidence from this table. Five checks are live here and can fail; two of them have.

| check | passed | failed | not checkable | can it fail in a completed build? |
|---|---|---|---|---|
| `runs_add_up` (runs = batter + extras on every ball) | 9,540 | 0 | 0 | no: enforced upstream by the silver `balls` contract rule `total_is_bat_plus_extras` |
| `extras_add_up` (extras = wides + no-balls + byes + leg byes + penalty) | 9,540 | 0 | 0 | no: enforced upstream by the silver `balls` contract rule `extras_add_up` |
| `overs_consistent` (six legal balls an over unless listed as miscounted) | 9,539 | 1 | 0 | yes |
| `wickets_within_limit` (at most 10 dismissals, 2 in a super over) | 9,540 | 0 | 0 | no: enforced upstream by the gold `fact_innings` contract rule `dismissals_within_limit` |
| `bowler_quota_respected` (no bowler starts more than a fifth of the scheduled overs) | 8,208 | 0 | 1,332 | yes |
| `batters_within_limit` (at most 12 batters an innings: eleven plus a concussion replacement) | 9,540 | 0 | 0 | yes |
| `target_matches` (unrevised target = first-innings total + 1) | 7,719 | 0 | 1,821 | no: enforced upstream by the `fact_innings` contract rules `set_target_equals_stated` and `nominal_target_equals_target` |
| `result_consistent` (the stated margin follows from the totals) | 8,463 | 0 | 1,077 | yes |
| `result_in_ball_stream` (the winner follows from the balls alone) | 8,461 | 2 | 1,077 | yes |

The build's own leak check on chases (no chase may show a target or ball limit the batting side then exceeded) is
judged from the target Cricsheet states, before the pipeline relabels a contradicted chase `inconsistent_with_balls`;
across all matches three chases passed their stated target before their last ball, and all three are published
without a target.

One match was awarded (always a failure), and 178 carry Cricsheet's own note that a field is missing (a fielding
powerplay, the player of the match, reviews); that note is recorded in `cricsheet_missing_fields_noted` and never
fails a match, since the ball stream is complete.

**Is it self-contained?** Yes, as files. It is a derivative of the sources below, credited per file in
LICENSES.md.

**Is any of it confidential or about identifiable people?** It names professional cricketers and match
officials in their public roles. `dim_player` holds public birth dates and nationalities from Wikidata.
Nothing else personal is included.

## Collection process

**How was the data acquired?** Every source is downloaded into an immutable, checksummed bronze snapshot, and
builds read only the snapshots pinned in `config/snapshots.lock.yml`:

| source | what | licence |
|---|---|---|
| Cricsheet (cricsheet.org) | ball-by-ball JSON, people register, known-missing list | ODC-BY 1.0 |
| Wikipedia season pages (pinned revisions) | reference list of numbered internationals | CC BY-SA 4.0 |
| Wikidata | player birth date, birthplace, citizenship; venue coordinates | CC0 |
| OpenStreetMap via Nominatim | venue geocodes | ODbL 1.0 |
| Open-Meteo (ECMWF ERA5 reanalysis) | hourly weather at each venue | CC BY 4.0 |

**Timeframe:** the Cricsheet snapshot is dated 2026-10-14. Weather is ERA5 reanalysis: a model estimate on a
roughly 25 km grid, not a reading at the ground. It can miss local showers; it does not show the rain that
stopped the 2019 World Cup semi-final.

**Ethical review:** not applicable. The data is public sports records.

## Preprocessing, cleaning and labelling

The pipeline runs bronze → staging → silver → gold (see docs/ARCHITECTURE.md):
1. Cricsheet JSON is parsed into typed tables, keeping powerplays, reviews, replacements, substitutes and
   tie-break winners.
2. Venue spellings are canonicalised with a reviewed override file. Venue context is built as of each match's
   date (`venue_context: asof`): see below.
3. Weather is converted to UTC and local time.
4. Wikidata conflicts are quarantined rather than guessed.

**As-of venue context:** a match's venue name, city, search strings, home team, geocode and weather placement use
only matches up to that match's date. A venue gets a new version whenever one of those inputs changes: the silver
layer holds 590 venues in 1,184 as-of versions (263 venues have more than one), and each match takes the version
valid on its start date. Weather is placed at that version's point. The contracts on `venue_versions` and
`match_venues` check this on every build, and a test rebuilds the context with every later match removed and
requires every earlier row to stay the same. Gold publishes those versions for the 182 in-scope grounds as
`dim_venue_version`, and `dim_match.venue_version` names the one each match was played under, so Pakistan's home
matches in the UAE keep Pakistan as the home side. A version valid on a date also counts that date's matches, so a
pre-match feature takes the latest version with `inputs_as_of` before the match date. `dim_venue` is different: it
holds the 182 grounds at their latest version, with `n_matches` and dates over all matches, so it is for browsing
and is not a point-in-time feature table. The re-downloaded `venues_geo` and `weather` snapshots (locked 2026-10-21) hold every
version's request; a build with a missing request stops.

**Pre-ball match state:** each value is computed from earlier balls only. Two independent checks enforce this:
- **During the build:** each ball's state must equal the previous state plus the previous ball.
- **In the test suite** (`tests/test_leakage.py`): every match is cut off at hundreds of points and recomputed.
  Every `pre_ball` column is unchanged; only `hindsight` columns may move.

**Data contracts:** every table has an Open Data Contract Standard v3.1.0 file under `contracts/`, enforced on
every build for types, keys, foreign keys and quality rules.

**Raw data:** the bronze snapshots are kept, but are not redistributed with this dataset. The code
(`cricket ingest`) re-fetches them.

## Uses

**Intended:**
- ball-level modelling: win probability, expected runs, wicket probability, death-overs analysis;
- era, venue and weather effects;
- teaching leakage-safe feature design with sports data.

**Use with care:**
- **Model inputs:** use only `pre_ball` columns. The `hindsight` columns of `fact_delivery_hindsight`
  (`target_status`, `innings_length_status`, `innings_balls_limit`, `balls_remaining_before`, `target_runs`,
  `runs_required_before`, `batters_absent`) are for choosing which rows to train on.
- **Venue features:** use `dim_venue_version`, taking the latest version with `inputs_as_of` before the match
  date (the weather in `fact_match_weather_hourly` is placed by venue version too). `dim_venue` is a
  latest-version view and is not a feature source.
- **Player features:** use `fact_player_match_form`, which is point-in-time. `report_player_careers` spans the
  whole snapshot, so it is not a feature source.
- **Generalising:** results from men's elite cricket do not transfer to women's or associate cricket.

**Not intended:**
- betting or wagering products;
- judging individual players' character or worth beyond on-field performance;
- any commercial use that would breach the share-alike files' terms.

## Distribution

**How:** the export folder written by `cricket export`, with this repository as the reproducible source.

**Formats:** Parquet for every table, and CSV for tables of up to 2 million rows. `fact_delivery` (4.51M rows) is
Parquet only.

**Licences:** each file carries its own licence (LICENSES.md, croissant.json):
- **ODC-BY 1.0:** the tables built from Cricsheet alone (matches, players, deliveries, innings, dismissals,
  form, the ball-outcome and Elo model outputs, and the quality and career reports).
- **ODbL 1.0:** everything that uses the OpenStreetMap geocodes, directly or through the venue: `dim_venue`,
  `dim_venue_version`, `fact_match_weather_hourly` (weather is taken at the geocoded point, so the Open-Meteo
  CC BY 4.0 data is carried under the stricter licence), `model_ball_win_probability` (it uses the home team) and
  the `report_model_*` files, which are built partly from it.
- **CC BY-SA 4.0:** `map_match_ids`, `report_international_coverage` and `report_result_check`, because of
  Wikipedia. These three files must stay share-alike.

**What is excluded:** nothing derived from ESPNcricinfo or Hawk-Eye is in the export. The export step refuses
any such table.

## Maintenance

**Maintainer:** through the repository's issues.

**Updates:** each update is a new Cricsheet snapshot rebuilt through the same pipeline. The build fingerprint in each file's metadata identifies exactly which data it holds.

**Errata:** errata are fixed in the next update. Errors in the source should be
reported to Cricsheet (cricsheet.org), so everyone downstream benefits.

**Can others contribute?** Yes, through pull requests. Every change must pass the contracts, the fixture build
and the leakage test in CI.

<!-- cricket-sim card stamp: build_fingerprint=abf009871cfe15fd matches_in_scope=9540 balls_in_scope=4507621 pct_matches_with_weather=99.98 pct_in_scope_players_with_dob_wikidata=87.8 afghanistan_withheld_odis=188 -->
