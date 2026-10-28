"""The modelling table: one row per delivery of the first two innings of every ODI in scope.

Inputs are `pre_ball` columns of gold.fact_delivery (contract timing class) plus three match facts known
before the first ball: pre-match Elo of both sides (elo.py, earlier matches only), whether the batting side is
at home, and whether it won the toss. `hindsight` columns are used for ONE thing only: deciding which matches
are "clean" enough to train and score on (docs/VALIDATION.md). They are never inputs.

A clean match: no rain rule applied, a result (win or tie), both innings of known length, an unrevised chase,
and every quality check passed. Other ODI rows retain an exclusion reason; published numeric model outputs
are withheld on those rows. Internal predictions are never evidence of event-time validity.
"""

import re

import duckdb

from .. import contracts
from ..paths import GOLD
from .elo import pre_match_ratings

# Every input must be a pre_ball column of fact_delivery, or a derived/pre-match feature listed here
# (check_inputs_are_pre_ball enforces it on the column lists train.py actually fits).
STATE_COLUMNS = ["innings_runs_before", "innings_wickets_before", "legal_balls_before",
                 "scheduled_balls_remaining_before", "nominal_target_runs", "nominal_runs_required_before"]
# What DERIVED may read besides pre_ball fact_delivery columns (alias f): facts fixed before the first ball.
# m = dim_match, eb / ew = pre-match Elo (elo.py, earlier dates only), hm = home_teams_as_of (dim_venue_version
# with inputs before the match date),
# unqualified bowling_team = the other side of dim_match (model_rows' own column).
PRE_MATCH_SOURCES = {"m": {"start_date", "toss_winner", "team1", "team2", "venue_id"}, "eb": {"elo_before"},
                     "ew": {"elo_before"}, "hm": {"home_team"}, "": {"bowling_team"}}
PRE_BALL_KEYS = {"innings"}  # a key column without a timing class that is known before the ball
DERIVED = {
    "run_rate": "CAST(6 * f.innings_runs_before AS DOUBLE) / nullif(f.legal_balls_before, 0)",
    "required_rate": "CAST(6 * f.nominal_runs_required_before AS DOUBLE) / nullif(f.scheduled_balls_remaining_before, 0)",
    "elo_diff": "eb.elo_before - ew.elo_before",
    "batting_at_home": "CASE WHEN hm.home_team = f.batting_team THEN 1 WHEN hm.home_team = bowling_team THEN -1 ELSE 0 END",
    "batting_won_toss": "CASE WHEN m.toss_winner = f.batting_team THEN 1 ELSE 0 END",
    "year": "year(m.start_date)",
    "venue_id": "m.venue_id",  # fixed before the first ball; only the player/venue baseline reads it
}
FEATURES = {
    1: ["innings_runs_before", "innings_wickets_before", "legal_balls_before", "scheduled_balls_remaining_before",
        "run_rate", "elo_diff", "batting_at_home", "batting_won_toss", "year"],
    2: ["innings_runs_before", "innings_wickets_before", "legal_balls_before", "scheduled_balls_remaining_before",
        "nominal_target_runs", "nominal_runs_required_before", "run_rate", "required_rate", "elo_diff",
        "batting_at_home", "batting_won_toss", "year"],
}
CLASSES = ["dot", "1", "2", "3", "4", "6", "wicket", "extra"]
OUTCOME_CLASS = """CASE WHEN f.dismissal_count > 0 THEN 'wicket'
    WHEN NOT f.is_legal THEN 'extra'
    WHEN f.runs_total = 0 THEN 'dot' WHEN f.runs_total = 1 THEN '1' WHEN f.runs_total = 2 THEN '2'
    WHEN f.runs_total = 3 THEN '3' WHEN f.runs_total IN (4, 5) THEN '4' ELSE '6' END"""


def check_inputs_are_pre_ball(fitted: dict[str, list[str]] | None = None,
                              derived: dict[str, list[str]] | None = None) -> None:
    """Refuse to model with any input that is not known before the ball.

    Checked: STATE_COLUMNS; every column each DERIVED expression reads (f.<col> must be a pre_ball column of
    fact_delivery, other tables only the PRE_MATCH_SOURCES columns); and, from train.walk_forward, every column
    list a model is fitted on (`fitted`). Each fitted column must be a pre_ball column of fact_delivery, the
    innings number, a DERIVED feature, or a feature train.py computes (`derived`: name -> its inputs) whose
    inputs pass the same check. Labels (the result, the ball's outcome) and hindsight columns never do."""
    c = contracts.get("fact_delivery")
    timing = {col["name"]: c.column_props(col).get("timingClass") for col in c.columns}
    pre_ball = {name for name, t in timing.items() if t == "pre_ball"}
    bad = [col for col in STATE_COLUMNS if col not in pre_ball]
    for name, sql in DERIVED.items():
        # every lower-case name that is not a function call is a column (SQL words are written in upper case)
        for alias, col in re.findall(r"(?<![\w.])(?:([a-z]\w*)\.)?([a-z_]\w*)\b(?!\s*\()", re.sub(r"'[^']*'", "", sql)):
            if not (alias == "f" and col in pre_ball or col in PRE_MATCH_SOURCES.get(alias, set())):
                bad.append(f"{name} reads {alias + '.' if alias else ''}{col}")
    derived = derived or {}
    allowed = pre_ball | PRE_BALL_KEYS | set(DERIVED)

    def known_before_ball(col: str, seen: tuple = ()) -> bool:
        if col in allowed:
            return True
        return col in derived and col not in seen and all(known_before_ball(s, seen + (col,)) for s in derived[col])

    for model, cols in (fitted or {}).items():
        bad += [f"{model} fits {col} ({timing.get(col) or 'not a pre-ball input'})"
                for col in cols if not known_before_ball(col)]
    if bad:
        raise SystemExit(f"model inputs must be known before the ball (pre_ball); these are not: {bad}")


# The venue's home team as known before the match: the latest dim_venue_version whose inputs end before the match
# date. A version valid ON the date also counts that date's matches (a double-header would see itself), and under
# venue_context: snapshot the one version's inputs end at the venue's last match, so it is never used before then.
HOME_ASOF_SQL = """
    SELECT m.match_id, v.home_team FROM dim_match m
    LEFT JOIN dim_venue_version v ON v.venue_id = m.venue_id AND v.inputs_as_of < m.start_date
    QUALIFY row_number() OVER (PARTITION BY m.match_id ORDER BY v.venue_version DESC NULLS LAST) = 1"""


def home_teams_as_of(con: duckdb.DuckDBPyConnection) -> dict[str, str | None]:
    """match_id -> the venue's home team known before the match (None: neutral, or not known yet)."""
    return dict(con.execute(HOME_ASOF_SQL).fetchall())


def load(con: duckdb.DuckDBPyConnection, gold=GOLD) -> None:
    """Create the TEMP table `model_rows` (and `elo`) in con."""
    check_inputs_are_pre_ball()
    for t in ("fact_delivery", "fact_innings", "dim_match", "dim_venue_version"):
        con.execute(f"CREATE OR REPLACE VIEW {t} AS SELECT * FROM read_parquet('{gold / t}.parquet')")
    matches = con.execute("""SELECT match_id, start_date, team1, team2, winner, result FROM dim_match
                             WHERE segment = 'ODI'""").fetchall()
    con.execute("CREATE OR REPLACE TEMP TABLE elo (match_id VARCHAR, team VARCHAR, elo_before DOUBLE, rated_before INTEGER)")
    rows = pre_match_ratings(matches)
    if rows:
        con.executemany("INSERT INTO elo VALUES (?, ?, ?, ?)", rows)
    home = home_teams_as_of(con)
    con.execute("CREATE OR REPLACE TEMP TABLE home_asof (match_id VARCHAR, home_team VARCHAR)")
    con.executemany("INSERT INTO home_asof VALUES (?, ?)", sorted(home.items()))
    derived = ",\n            ".join(f"{sql} AS {name}" for name, sql in DERIVED.items())
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE model_rows AS
        WITH clean AS (
            SELECT m.match_id,
                m.method IS NULL AND m.passes_quality AND (m.winner IS NOT NULL OR m.result = 'tie')
                AND bool_and(i.innings_length_status = 'full_known')
                AND bool_and(i.innings = 1 OR i.target_status = 'set') AS is_clean,
                CASE WHEN m.result = 'no result' THEN 'no_result'
                     WHEN m.method IS NOT NULL THEN 'rain_rule_applied'
                     WHEN NOT m.passes_quality THEN 'fails_quality_checks'
                     WHEN NOT bool_and(i.innings_length_status = 'full_known') THEN 'innings_length_unknown'
                     WHEN NOT bool_and(i.innings = 1 OR i.target_status = 'set') THEN 'chase_target_not_set' END
                    AS exclusion_reason
            FROM dim_match m JOIN fact_innings i ON i.match_id = m.match_id AND NOT i.is_super_over AND i.innings <= 2
            WHERE m.segment = 'ODI' GROUP BY m.match_id, m.method, m.passes_quality, m.winner, m.result
        )
        SELECT f.match_id, f.innings, f.delivery_number, m.start_date, year(m.start_date) AS season,
            f.batting_team, CASE WHEN f.batting_team = m.team1 THEN m.team2 ELSE m.team1 END AS bowling_team,
            f.batter_id, f.bowler_id, f.over,
            {", ".join("f." + c for c in STATE_COLUMNS)},
            {derived},
            c.is_clean, c.exclusion_reason,
            CAST(CASE WHEN m.result = 'tie' THEN 0.5 WHEN m.winner = f.batting_team THEN 1.0
                 WHEN m.winner IS NOT NULL THEN 0.0 END AS DOUBLE) AS batting_won,
            {OUTCOME_CLASS} AS outcome_class, f.runs_total,
            f.is_legal, f.dismissal_count  -- this ball's own outcome: only to state the innings-1 end (train.py)
        FROM fact_delivery f
        JOIN dim_match m USING (match_id)
        JOIN clean c USING (match_id)
        LEFT JOIN home_asof hm USING (match_id)
        LEFT JOIN elo eb ON eb.match_id = f.match_id AND eb.team = f.batting_team
        LEFT JOIN elo ew ON ew.match_id = f.match_id AND ew.team = (CASE WHEN f.batting_team = m.team1 THEN m.team2 ELSE m.team1 END)
        WHERE m.segment = 'ODI' AND NOT f.is_super_over AND f.innings <= 2
        ORDER BY f.match_id, f.innings, f.delivery_number""")
