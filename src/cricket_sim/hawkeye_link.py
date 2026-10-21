"""Link Hawk-Eye ball-tracking rows (HawkeyeStats) to Cricsheet deliveries.

What the Hawk-Eye columns encode (checked against the data):
  delivery  "innings.over.seq": innings 1-4 (3/4 are super overs in limited-overs), over is 1-based,
            seq counts every delivery in the over, wides and no-balls included (= Cricsheet ball_in_over).
  ball      legal-ball counter within the over (a wide/no-ball repeats the number).
  runs      total runs off the delivery (text; one corrupt row), extras 'Wd'/'Nb'/'B'/'Lb'/'NbB'/'NbLb'.
  ballSpeed m/s; -1 (and a couple of 0s) = not tracked -> null.
  pitchX/Y  bounce point in metres: X lateral from middle stump, Y distance from the batter's stumps.
  stumpsX/Y ball position at the stumps plane in metres: X lateral, Y height. 'NA' = not tracked.
  fieldX/Y  unitless 0-100 grid of the field graphic, striker near (50, 50), boundary at radius ~45-50.
            (0, 0) and (5, 5) are "no location" sentinels (both lie outside the boundary) -> null here.

Method:
  1. Players: Hawk-Eye ids are ICC Pulselive ids -> Cricsheet people.key_pulse / key_pulse_2.
     Players without a Pulse key (or whose register Pulse key points at someone else) fall back to
     surname comparison, and that is flagged ('verified_surname').
  2. Candidates: per Hawk-Eye match, Cricsheet matches of the right format (male IPL / ODI / Test)
     ranked by overlap of (batter, bowler) pair sets, once on Pulse ids and once on surnames.
  3. Alignment: Hawk-Eye (innings, over, seq) -> Cricsheet (innings, over + 1, ball_in_over), the same
     every-delivery counting on both sides, so wides/no-balls line up and missing Hawk-Eye rows
     (tracking gaps, whole untracked overs) simply leave Cricsheet balls without tracking.
     A ball "agrees" when total runs, batter and bowler all match (players by Pulse id, else surname).
  4. Pick: candidate with the highest share of agreeing balls (ties -> lowest match_id), then
     one-to-one (a Cricsheet match goes to the Hawk-Eye match that agrees best with it).
     confidence = best score - runner-up score, so it is high only when the pick is unambiguous.

Everything is deterministic: no randomness, every ranking has a full tie-break.
"""

from pathlib import Path

from .paths import SILVER

MIN_SCORE = 0.5          # share of Hawk-Eye balls that must agree to accept a link
CANDIDATES_PER_KEY = 3   # candidates kept per Hawk-Eye match from each pair-overlap ranking

COMPETITION_FILTER = """
    CASE WHEN m.match_type = 'Test' THEN 'mensTest'
         WHEN m.match_type = 'ODI' THEN 'mensODI'
         WHEN m.match_type = 'T20' AND m.event_name LIKE 'Indian Premier League%' THEN 'mensIPL' END"""


def _surname(expr: str) -> str:
    return f"lower(regexp_replace(list_last(string_split(trim({expr}), ' ')), '[^A-Za-z]', '', 'g'))"


STEPS = [
    # Pulse id -> Cricsheet identifier (key_pulse values are unique across both columns)
    ("hk_pulse", """
        SELECT key_pulse AS pulse_id, identifier FROM people WHERE key_pulse IS NOT NULL
        UNION ALL
        SELECT key_pulse_2, identifier FROM people WHERE key_pulse_2 IS NOT NULL"""),

    ("hk_he", f"""
        SELECT h.competition, h.hawkeye_match_id,
            split_part(h.delivery_code, '.', 1)::INT AS innings,
            split_part(h.delivery_code, '.', 2)::INT - 1 AS over,
            split_part(h.delivery_code, '.', 3)::INT AS ball_in_over,
            pb.identifier AS batter_id, pw.identifier AS bowler_id,
            {_surname('h.batter_name')} AS batter_sn, {_surname('h.bowler_name')} AS bowler_sn,
            h.runs_total,
            CASE WHEN h.ball_speed_mps > 0 THEN h.ball_speed_mps END AS speed_mps,
            h.pitch_x, h.pitch_y, h.stumps_x, h.stumps_y, h.field_x AS fx, h.field_y AS fy,
            h.right_handed_bat
        FROM stg_hawkeye_balls h
        LEFT JOIN hk_pulse pb ON pb.pulse_id = h.batter_pulse_id
        LEFT JOIN hk_pulse pw ON pw.pulse_id = h.bowler_pulse_id"""),

    ("hk_he_n", """
        SELECT competition, hawkeye_match_id, count(*) AS hawkeye_balls FROM hk_he GROUP BY ALL"""),

    ("hk_cand_matches", f"""
        SELECT m.match_id, {COMPETITION_FILTER} AS competition
        FROM matches m WHERE m.gender = 'male' AND {COMPETITION_FILTER} IS NOT NULL"""),

    ("hk_cs", f"""
        SELECT c.competition, b.match_id, b.innings, b.over, b.ball_in_over, b.delivery_number,
            b.batter_id, b.bowler_id, {_surname('b.batter')} AS batter_sn, {_surname('b.bowler')} AS bowler_sn,
            b.runs_total
        FROM balls b JOIN hk_cand_matches c USING (match_id)"""),

    # Pair-set overlap, on Pulse ids and (fallback) on surnames
    ("hk_pairs", f"""
        WITH hp AS (
            SELECT DISTINCT competition, hawkeye_match_id, 'pulse_id' AS key, batter_id AS a, bowler_id AS b
            FROM hk_he WHERE batter_id IS NOT NULL AND bowler_id IS NOT NULL
            UNION ALL
            SELECT DISTINCT competition, hawkeye_match_id, 'surname', batter_sn, bowler_sn
            FROM hk_he WHERE batter_sn <> '' AND bowler_sn <> ''
        ), cp AS (
            SELECT DISTINCT competition, match_id, 'pulse_id' AS key, batter_id AS a, bowler_id AS b
            FROM hk_cs WHERE batter_id IS NOT NULL AND bowler_id IS NOT NULL
            UNION ALL
            SELECT DISTINCT competition, match_id, 'surname', batter_sn, bowler_sn
            FROM hk_cs WHERE batter_sn <> '' AND bowler_sn <> ''
        ), hn AS (SELECT competition, hawkeye_match_id, key, count(*) AS n FROM hp GROUP BY ALL),
        cn AS (SELECT match_id, key, count(*) AS n FROM cp GROUP BY ALL),
        shared AS (
            SELECT hp.competition, hp.hawkeye_match_id, hp.key, cp.match_id, count(*) AS s
            FROM hp JOIN cp USING (competition, key, a, b) GROUP BY ALL
        )
        SELECT s.competition, s.hawkeye_match_id, s.key, s.match_id,
            s.s / hn.n AS containment, s.s / (hn.n + cn.n - s.s) AS jaccard
        FROM shared s JOIN hn USING (competition, hawkeye_match_id, key) JOIN cn USING (match_id, key)
        QUALIFY row_number() OVER (PARTITION BY s.competition, s.hawkeye_match_id, s.key
                                   ORDER BY containment DESC, jaccard DESC, s.match_id) <= {CANDIDATES_PER_KEY}"""),

    ("hk_cands", """
        SELECT competition, hawkeye_match_id, match_id,
            arg_min(key, key) AS key_found,   -- 'pulse_id' sorts before 'surname'
            max(jaccard) FILTER (key = 'pulse_id') AS pulse_jaccard
        FROM hk_pairs GROUP BY ALL"""),

    # Positional alignment of every Hawk-Eye row against every candidate
    ("hk_aligned", """
        SELECT c.competition, c.hawkeye_match_id, c.match_id, h.innings, h.over, h.ball_in_over,
            cs.delivery_number,
            h.runs_total IS NOT DISTINCT FROM cs.runs_total AND cs.runs_total IS NOT NULL AS runs_ok,
            coalesce(h.batter_id = cs.batter_id, false) OR h.batter_sn = cs.batter_sn AS batter_ok,
            coalesce(h.bowler_id = cs.bowler_id, false) OR h.bowler_sn = cs.bowler_sn AS bowler_ok,
            NOT (coalesce(h.batter_id = cs.batter_id, false) AND coalesce(h.bowler_id = cs.bowler_id, false))
                AS used_surname
        FROM hk_cands c
        JOIN hk_he h USING (competition, hawkeye_match_id)
        JOIN hk_cs cs ON cs.match_id = c.match_id AND cs.innings = h.innings
                     AND cs.over = h.over AND cs.ball_in_over = h.ball_in_over"""),

    ("hk_scores", """
        SELECT a.competition, a.hawkeye_match_id, a.match_id, any_value(c.key_found) AS key_found,
            count(*) AS aligned_balls,
            count(*) FILTER (runs_ok AND batter_ok AND bowler_ok) AS agreeing_balls,
            count(*) FILTER (runs_ok AND batter_ok AND bowler_ok) / any_value(n.hawkeye_balls) AS score
        FROM hk_aligned a
        JOIN hk_cands c USING (competition, hawkeye_match_id, match_id)
        JOIN hk_he_n n USING (competition, hawkeye_match_id)
        GROUP BY ALL"""),

    # Best candidate per Hawk-Eye match, runner-up for the margin, then one-to-one on Cricsheet match
    ("hk_best", f"""
        WITH ranked AS (
            SELECT *, row_number() OVER (PARTITION BY competition, hawkeye_match_id
                                         ORDER BY score DESC, match_id) AS rk
            FROM hk_scores
        ), top AS (
            SELECT r1.*, coalesce(r2.score, 0) AS runner_up FROM ranked r1
            LEFT JOIN ranked r2 ON r2.competition = r1.competition AND r2.hawkeye_match_id = r1.hawkeye_match_id
                               AND r2.rk = 2
            WHERE r1.rk = 1 AND r1.score >= {MIN_SCORE}
        )
        SELECT * FROM top
        QUALIFY row_number() OVER (PARTITION BY match_id ORDER BY score DESC, competition, hawkeye_match_id) = 1"""),
]

MATCH_LINK_SQL = """
SELECT n.hawkeye_match_id, n.competition, b.match_id,
    coalesce(round(b.score - b.runner_up, 4), 0.0) AS confidence,
    CASE WHEN b.match_id IS NULL THEN 'unmatched'
         ELSE b.key_found || '_pairs+ball_alignment' END AS method,
    n.hawkeye_balls,
    coalesce(b.aligned_balls, 0) AS aligned_balls,
    round(b.agreeing_balls / b.aligned_balls, 4) AS agreement_rate
FROM hk_he_n n
LEFT JOIN hk_best b USING (competition, hawkeye_match_id)
"""

BALL_TRACKING_SQL = """
SELECT a.match_id, CAST(a.innings AS SMALLINT) AS innings, a.delivery_number,
    -- physically impossible readings become NULL (fastest recorded delivery is ~161 km/h)
    CASE WHEN h.speed_mps * 3.6 BETWEEN 50 AND 165 THEN round(h.speed_mps * 3.6, 2) END AS speed_kmh,
    h.right_handed_bat,
    CASE WHEN abs(h.pitch_x) <= 3 THEN h.pitch_x END AS pitch_x,
    CASE WHEN h.pitch_y BETWEEN -2 AND 25 THEN h.pitch_y END AS pitch_y,
    CASE WHEN abs(h.stumps_x) <= 3 THEN h.stumps_x END AS stumps_x,
    CASE WHEN h.stumps_y BETWEEN 0 AND 4 THEN h.stumps_y END AS stumps_y,
    -- Hawk-Eye x is not mirrored for left-handers; positive = towards the batter's off side
    CASE WHEN abs(h.pitch_x) <= 3 THEN CASE WHEN h.right_handed_bat THEN -h.pitch_x WHEN NOT h.right_handed_bat THEN h.pitch_x END END
        AS pitch_x_off_side,
    CASE WHEN abs(h.stumps_x) <= 3 THEN CASE WHEN h.right_handed_bat THEN -h.stumps_x WHEN NOT h.right_handed_bat THEN h.stumps_x END END
        AS stumps_x_off_side,
    CASE WHEN (h.fx, h.fy) NOT IN ((0, 0), (5, 5)) THEN h.fx END AS field_x,
    CASE WHEN (h.fx, h.fy) NOT IN ((0, 0), (5, 5)) THEN h.fy END AS field_y,
    CASE WHEN a.runs_ok AND a.batter_ok AND a.bowler_ok AND NOT a.used_surname THEN 'verified'
         WHEN a.runs_ok AND a.batter_ok AND a.bowler_ok THEN 'verified_surname'
         WHEN a.runs_ok THEN 'player_mismatch'
         ELSE 'runs_mismatch' END AS link_quality
FROM hk_aligned a
JOIN hk_best b USING (competition, hawkeye_match_id, match_id)
JOIN hk_he h USING (competition, hawkeye_match_id, innings, over, ball_in_over)
"""


def build_hawkeye_link(con, out_dir: Path = SILVER) -> None:
    """Write hawkeye_match_link.parquet to out_dir (with a view), and leave the per-ball alignment in the
    TEMP table hawkeye_ball_tracking for ball_layers to split by timing class (so it is stored once).

    Needs views stg_hawkeye_balls, balls, matches, people (match_players is not used: the pair sets come
    from balls, which already carry Cricsheet person ids).
    """
    for name, sql in STEPS:
        con.execute(f"CREATE OR REPLACE TEMP TABLE {name} AS {sql}")

    dest = Path(out_dir) / "hawkeye_match_link.parquet"
    con.execute(f"COPY (SELECT * FROM ({MATCH_LINK_SQL}) ORDER BY competition, hawkeye_match_id) "
                f"TO '{dest}' (FORMAT parquet, COMPRESSION zstd)")
    con.execute(f"CREATE OR REPLACE VIEW hawkeye_match_link AS SELECT * FROM read_parquet('{dest}')")
    con.execute(f"CREATE OR REPLACE TEMP TABLE hawkeye_ball_tracking AS {BALL_TRACKING_SQL}")

    for name, _ in reversed(STEPS):
        con.execute(f"DROP TABLE IF EXISTS {name}")
