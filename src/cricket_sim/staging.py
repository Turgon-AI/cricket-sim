"""Staging: third-party CSVs, one table per source file, renamed to snake_case and typed. Nothing else.

Rules (the dbt "staging" convention):
  - one staging table per upstream file; no joins, no business logic, no filtering of rows
  - the raw columns must match config/input_schemas.yml exactly, checked BEFORE anything is written,
    so an upstream release that renames or retypes a column fails the build loudly
  - columns are renamed to snake_case and cast to their real types; unreadable values become NULL
    (sentinels such as Hawk-Eye's -1 speed and 'NA' coordinates are left for silver to interpret)

Cricsheet is not staged here: its JSON is parsed straight into silver by cricsheet_parse.py, which is
its typed, tested staging step. Every staged source is restricted (Cricinfo-derived or unlicensed),
so nothing in staging is ever exported.
"""

from .paths import STAGING, load_yaml, snapshot_dir


def _ari(name: str) -> str:
    return "read_csv('{folder}/" + name + ".zst', compression='zstd', header=true, sample_size=-1)"


def _hawkeye_union(files: list[str]) -> str:
    parts = [f"SELECT '{f.replace('HawkeyeStats.csv', '')}' AS competition, * FROM read_csv('{{folder}}/{f}.zst', "
             f"compression='zstd', header=true, sample_size=-1)" for f in files]
    return "(" + " UNION ALL BY NAME ".join(parts) + ")"


def _num(col: str) -> str:
    return f'try_cast("{col}" AS DOUBLE)'


def _bool(col: str) -> str:
    return f"CASE lower(CAST(\"{col}\" AS VARCHAR)) WHEN 'true' THEN true WHEN 'false' THEN false END"


# table -> (bronze source, raw reader, input_schemas.yml key, SELECT list over the raw columns, key)
TABLES = {
    "stg_ari_tracking": ("ari", _ari("cricinfo_tracking.csv"), "ari_tracking", """
        CAST(match_id AS VARCHAR) AS match_id, CAST(innings AS SMALLINT) AS innings,
        CAST(delivery_number AS INTEGER) AS delivery_number,
        ball_line, ball_length, shot_played, shot_control,
        CAST(wagon_x AS INTEGER) AS wagon_x, CAST(wagon_y AS INTEGER) AS wagon_y,
        CAST(wagon_zone AS SMALLINT) AS wagon_zone, wagon_angle, wagon_distance, wagon_in_ring,
        shot_direction, ball_timestamp, win_probability, CAST(predicted_score AS INTEGER) AS predicted_score,
        ball_events""", "match_id, innings, delivery_number"),
    "stg_ari_predictions": ("ari", _ari("cricinfo_text_predictions.csv"), "ari_predictions", """
        CAST(match_id AS VARCHAR) AS match_id, CAST(innings AS SMALLINT) AS innings,
        CAST(delivery_number AS INTEGER) AS delivery_number,
        line, line_confidence, line_explicit = 1 AS line_explicit,
        length, length_confidence, length_explicit = 1 AS length_explicit,
        shot, shot_confidence, shot_explicit = 1 AS shot_explicit,
        direction, direction_confidence, direction_explicit = 1 AS direction_explicit""",
        "match_id, innings, delivery_number"),
    "stg_ari_players": ("ari", _ari("players.csv"), "ari_players", """
        player_id, name, CAST(cricinfo_id AS VARCHAR) AS cricinfo_id, "full name" AS full_name, gender, country,
        "batting style" AS batting_style, "bowling style" AS bowling_style, "playing role" AS playing_role,
        "fielding style" AS fielding_style, CAST("first international year" AS SMALLINT) AS first_international_year,
        CAST("last international year" AS SMALLINT) AS last_international_year, "major teams" AS major_teams,
        "image url" AS image_url""", "player_id"),
    "stg_hawkeye_balls": ("hawkeye", None, "hawkeye_balls", f"""
        competition, CAST(matchId AS VARCHAR) AS hawkeye_match_id, delivery AS delivery_code,
        CAST(ball AS SMALLINT) AS legal_ball_in_over,
        batter AS batter_name, CAST(batterId AS VARCHAR) AS batter_pulse_id, {_bool('rightHandedBat')} AS right_handed_bat,
        nonStriker AS non_striker_name, CAST(nonStrikerId AS VARCHAR) AS non_striker_pulse_id,
        bowler AS bowler_name, CAST(bowlerId AS VARCHAR) AS bowler_pulse_id, {_bool('rightArmedBowl')} AS right_arm_bowler,
        bowlingStyle AS bowling_style, ballSpeed AS ball_speed_mps, dismissalDetails AS dismissal_details,
        try_cast(runs AS INTEGER) AS runs_total, CAST(batterRuns AS INTEGER) AS runs_batter,
        CAST(bowlerRuns AS INTEGER) AS runs_bowler, extras AS extras_code,
        {_num('pitchX')} AS pitch_x, {_num('pitchY')} AS pitch_y, {_num('stumpsX')} AS stumps_x,
        {_num('stumpsY')} AS stumps_y, {_num('fieldX')} AS field_x, {_num('fieldY')} AS field_y""",
        "competition, hawkeye_match_id, delivery_code"),
}


def _reader(source: str, raw: str | None, folder) -> str:
    if raw is None:  # Hawk-Eye ships one file per competition
        raw = _hawkeye_union(load_yaml("sources.yml")["hawkeye"]["files"])
    return raw.format(folder=folder)


def build_staging(con, lock: dict) -> None:
    """Check every raw file against its declared schema, then write the staged tables and register views."""
    declared = load_yaml("input_schemas.yml")
    readers = {}
    for table, (source, raw, schema_key, _, _) in TABLES.items():
        readers[table] = _reader(source, raw, snapshot_dir(source, lock[source]))
        actual = [f"{n}:{t}" for n, t, *_ in con.execute(f"DESCRIBE SELECT * FROM {readers[table]}").fetchall()]
        if actual != declared[schema_key]:
            drift = sorted(set(actual) ^ set(declared[schema_key]))
            raise SystemExit(f"{table}: upstream schema changed ({drift[:6]}...). "
                             "Review the release, then update config/input_schemas.yml deliberately")
    for table, (_, _, _, select, key) in TABLES.items():
        dest = STAGING / f"{table}.parquet"
        con.execute(f"COPY (SELECT {select} FROM {readers[table]} ORDER BY {key}) TO '{dest}' "
                    "(FORMAT parquet, COMPRESSION zstd)")
        con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{dest}')")
