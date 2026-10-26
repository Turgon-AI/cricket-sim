"""Per-ball columns split by WHEN they are known and WHO recorded them, so a model can never read the answer.

Timing classes (each per-ball table holds exactly one, and one licence family):
  pre_ball        known before the ball is bowled, from Cricsheet only (match state)
  benchmark_model Cricinfo's model outputs from the PREVIOUS ball (non-commercial licence). Pre-ball in
                  timing, but recomputed after the fact by a model fitted on later seasons, so as a feature
                  in a time-sliced backtest it leaks later knowledge. A benchmark to score against, never
                  an input.
  delivery        physically measured delivery (Hawk-Eye speed, bounce point, stumps)
  observer_label  Cricinfo's human labels (line, length, shot, control). Written after seeing the
                  outcome: measured against Hawk-Eye they lean towards the result, and control nearly
                  encodes the wicket. Never pre-ball; use only as noisy, bias-corrected outcome inputs.
  result          where the ball went, plus the Hawk-Eye link check (which compares runs)
  post_ball       Cricinfo model outputs computed after this ball
  predicted_*     labels inferred from commentary text, never merged with recorded labels; the
                  outcome-shaped ones (shot, direction) are kept apart from line/length
  hindsight       a column-level marker inside a pre_ball table: computed only from earlier balls, but
                  cross-checked against how the innings went (NULL when it was later cut short or
                  rain-revised), so its NULLs and statuses reveal the future. Use it to choose training
                  rows, never as a feature. tests/test_leakage.py proves every pre_ball column is
                  unchanged when a match is cut off at any ball, and that only hindsight columns move.

A decision model may use pre_ball only; benchmark_model is for comparison, never a feature. An outcome model
may add delivery.
`LAYERS` is the registry, generated from the contracts (customProperties.timingClass); the build fails if a
table's columns drift from it.

Where they live: the pre-ball match state is OUR derivation from Cricsheet, so it is built in gold
(int_innings_state, int_ball_state_pre, one row per ball). The other layers restate third-party
per-ball data, so they live in silver, named after their source (cricinfo_ball_*, hawkeye_ball_*), and
are sparse: a row only where the source has a value.
"""

from . import contracts

KEYS = ["match_id", "innings", "delivery_number"]

# {table: (timing class, non-key columns)}, read from the contracts of the per-ball layer tables
LAYERS = contracts.timing_registry()

# Innings-level facts that are knowable before the innings' first ball, with honest gaps:
# - Cricsheet stores only the FINAL revised target and the overs where rain stopped play, not when
#   revisions happened, so a revised chase gets no target or horizon (target_status says why).
# - A limited-overs first innings that ended short of its overs, not all out, may have been cut
#   before or during the innings; its length is unknown, not assumed.
# - The legal balls an innings may hold are the scheduled overs x balls per over, plus the extra balls of
#   every over Cricsheet lists as miscounted (a 7-ball over adds one, a 5-ball over takes one off). Only
#   the nominal pre-ball limit (scheduled_balls_limit) stays overs x balls per over.
INNINGS_SQL = """
WITH inn AS (
    SELECT i.match_id, i.innings, i.team, i.is_super_over, i.target_runs, i.target_balls, i.forfeited,
        coalesce(i.penalty_runs_pre, 0) AS pen_pre, coalesce(i.penalty_runs_post, 0) AS pen_post,
        coalesce(len(string_split(i.absent_hurt, '|')), 0) AS batters_absent,
        m.match_type, m.method, m.balls_per_over AS bpo, m.scheduled_overs * m.balls_per_over AS sched_balls,
        CAST(m.scheduled_overs * m.balls_per_over + coalesce(x.extra_balls, 0) AS SMALLINT) AS allowed_balls,
        coalesce(t.runs, 0) AS runs, coalesce(t.legal, 0) AS legal, coalesce(t.dismissals, 0) AS dismissals,
        coalesce(t.runs_before_last, 0) AS runs_before_last, coalesce(t.legal_before_last, 0) AS legal_before_last
    FROM innings i JOIN matches m USING (match_id)
    LEFT JOIN (SELECT match_id, innings, sum(runs_total) AS runs, count(*) FILTER (is_legal) AS legal,
                      sum(dismissal_count) AS dismissals,
                      sum(runs_total) - arg_max(runs_total, delivery_number) AS runs_before_last,
                      count(*) FILTER (is_legal) - CASE WHEN arg_max(is_legal, delivery_number) THEN 1 ELSE 0 END AS legal_before_last
               FROM balls GROUP BY ALL) t USING (match_id, innings)
    LEFT JOIN (SELECT mi.match_id, mi.innings,
                      sum(CAST(json_extract_string(mi.miscounted_overs, '$."' || k || '"."balls"') AS INTEGER)
                          - mm.balls_per_over) AS extra_balls
               FROM innings mi JOIN matches mm USING (match_id), unnest(json_keys(mi.miscounted_overs)) AS u(k)
               WHERE mi.miscounted_overs IS NOT NULL AND CAST(k AS INTEGER) < mm.scheduled_overs
               GROUP BY ALL) x USING (match_id, innings)
), inn2 AS (
    -- Every award counts in the innings total, including post-innings penalties: they are known once the
    -- innings is complete, so they belong in the nominal target and the Test lead of the next innings.
    SELECT *, runs + pen_pre + pen_post AS total,
        CASE WHEN is_super_over THEN 'super_over' WHEN sched_balls IS NULL THEN 'unlimited' ELSE 'limited' END AS kind,
        row_number() OVER (PARTITION BY match_id, is_super_over ORDER BY innings) AS seq
    FROM inn
), chase0 AS (
    SELECT c.match_id, c.target_runs, c.target_balls,
        c.method IS NULL AND (c.target_balls IS NULL OR c.target_balls = c.sched_balls) AS unrevised_stated,
        -- stated target, or first-innings total + 1 when Cricsheet gives none
        coalesce(c.target_runs, f.total + 1) AS effective_target,
        c.runs_before_last, c.legal_before_last, c.pen_pre, c.sched_balls, c.allowed_balls
    FROM inn2 c LEFT JOIN inn2 f ON f.match_id = c.match_id AND f.kind = 'limited' AND f.seq = 1
    WHERE c.kind = 'limited' AND c.seq = 2
), chase1 AS (
    -- upstream errors: the chase passed its target, or ran past its overs, before its last ball
    SELECT *, unrevised_stated AND (runs_before_last + pen_pre >= effective_target OR legal_before_last >= allowed_balls)
        AS inconsistent
    FROM chase0
), chase AS (
    SELECT *, unrevised_stated AND NOT inconsistent AS unrevised FROM chase1
)
SELECT a.match_id, a.innings, a.is_super_over, a.pen_pre, a.batters_absent,
    CASE a.kind WHEN 'super_over' THEN 2 ELSE 10 END AS wickets_limit,
    -- NOMINAL: the target as it stood when the innings began, from completed innings (every award they
    -- received included), without later target revisions. It is not the official target after a rain revision.
    CASE a.kind WHEN 'super_over' THEN a.bpo WHEN 'limited' THEN a.sched_balls END AS scheduled_balls_limit,
    CAST(CASE
        WHEN a.kind = 'limited' AND a.seq = 2
            THEN (SELECT p.total + 1 FROM inn2 p WHERE p.match_id = a.match_id AND p.kind = 'limited' AND p.seq = 1)
        WHEN a.kind = 'super_over' AND a.seq % 2 = 0
            THEN (SELECT p.total + 1 FROM inn2 p WHERE p.match_id = a.match_id AND p.kind = 'super_over' AND p.seq = a.seq - 1)
        WHEN a.kind = 'unlimited' AND a.match_type = 'Test' AND a.innings = 4
            THEN (SELECT sum(CASE WHEN p.team = a.team THEN -p.total ELSE p.total END) + 1
                  FROM inn2 p WHERE p.match_id = a.match_id AND p.innings < 4)
    END AS INTEGER) AS nominal_target_runs,
    -- HINDSIGHT (below, to test_lead_at_start excluded): limits and targets cross-checked against how the
    -- innings actually went, NULL when it was cut short or rain-revised. Their NULLs and statuses reveal
    -- the future, so use them to choose training rows, never as features.
    CASE
        WHEN a.kind = 'super_over' THEN a.bpo
        WHEN a.kind = 'unlimited' THEN NULL
        WHEN a.seq = 1 AND (a.legal >= a.allowed_balls OR a.dismissals + a.batters_absent >= 10
                            OR c.unrevised) THEN a.allowed_balls
        WHEN a.seq = 2 AND c.unrevised THEN a.allowed_balls
    END AS innings_balls_limit,
    CASE
        WHEN a.kind = 'super_over' THEN 'super_over'
        WHEN a.kind = 'unlimited' THEN 'unlimited'
        WHEN a.seq = 1 AND (a.legal >= a.allowed_balls OR a.dismissals + a.batters_absent >= 10 OR c.unrevised) THEN 'full_known'
        WHEN a.seq = 2 AND c.unrevised THEN 'full_known'
        ELSE 'reduced_timing_unknown'
    END AS innings_length_status,
    CAST(CASE
        WHEN a.kind = 'limited' AND a.seq = 2 AND c.unrevised
            THEN coalesce(c.target_runs, (SELECT p.total + 1 FROM inn2 p WHERE p.match_id = a.match_id
                                          AND p.kind = 'limited' AND p.seq = 1))
        WHEN a.kind = 'super_over' AND a.seq % 2 = 0
            THEN (SELECT p.total + 1 FROM inn2 p WHERE p.match_id = a.match_id AND p.kind = 'super_over' AND p.seq = a.seq - 1)
        WHEN a.kind = 'unlimited' AND a.match_type = 'Test' AND a.innings = 4
            THEN (SELECT sum(CASE WHEN p.team = a.team THEN -p.total ELSE p.total END) + 1
                  FROM inn2 p WHERE p.match_id = a.match_id AND p.innings < 4)
    END AS INTEGER) AS target_runs,
    CASE
        WHEN a.kind = 'limited' AND a.seq = 2 AND c.unrevised THEN 'set'
        WHEN a.kind = 'limited' AND a.seq = 2 AND c.inconsistent THEN 'inconsistent_with_balls'
        WHEN a.kind = 'limited' AND a.seq = 2 THEN 'revised_timing_unknown'
        WHEN a.kind = 'super_over' AND a.seq % 2 = 0 THEN 'super_over_derived'
        WHEN a.kind = 'unlimited' AND a.match_type = 'Test' AND a.innings = 4 THEN 'test_derived'
        ELSE 'none'
    END AS target_status,
    -- Tests: the balance of runs before this innings, post-innings awards of completed innings included.
    CAST(CASE WHEN a.kind = 'unlimited' THEN
        (SELECT coalesce(sum(CASE WHEN p.team = a.team THEN p.total ELSE -p.total END), 0)
         FROM inn2 p WHERE p.match_id = a.match_id AND p.innings < a.innings)
    END AS INTEGER) AS test_lead_at_start
FROM inn2 a LEFT JOIN chase c ON c.match_id = a.match_id AND a.kind = 'limited'
"""

STATE_PRE_SQL = """
WITH running AS (
    SELECT b.match_id, b.innings, b.delivery_number,
        coalesce(sum(b.runs_total) OVER w, 0) AS runs_before,
        coalesce(sum(b.dismissal_count) OVER w, 0) AS wickets_before,
        coalesce(sum(CASE WHEN b.is_legal THEN 1 ELSE 0 END) OVER w, 0) AS legal_balls_before,
        b.over, b.legal_balls_in_over_before
    FROM balls b
    WINDOW w AS (PARTITION BY b.match_id, b.innings ORDER BY b.delivery_number
                 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
)
SELECT r.match_id, r.innings, r.delivery_number,
    s.is_super_over,
    CAST(s.pen_pre + r.runs_before AS INTEGER) AS innings_runs_before,
    CAST(r.wickets_before AS SMALLINT) AS innings_wickets_before,
    CAST(r.legal_balls_before AS SMALLINT) AS legal_balls_before,
    s.innings_balls_limit,
    CAST(s.innings_balls_limit - r.legal_balls_before AS SMALLINT) AS balls_remaining_before,
    s.innings_length_status,
    s.target_runs,
    CAST(s.target_runs - (s.pen_pre + r.runs_before) AS INTEGER) AS runs_required_before,
    s.target_status,
    s.scheduled_balls_limit,
    -- Balls left counted by overs, from this ball's own over only (a later miscounted over cannot be known yet):
    -- the balls of the overs still to come, plus what is left of this over. A ball inside an over that has
    -- already run past its usual length still has itself left, so a legal ball never sees 0.
    CAST((s.scheduled_balls_limit // m.balls_per_over - r.over - 1) * m.balls_per_over
         + greatest(m.balls_per_over - r.legal_balls_in_over_before, 1) AS SMALLINT) AS scheduled_balls_remaining_before,
    s.nominal_target_runs,
    CAST(s.nominal_target_runs - (s.pen_pre + r.runs_before) AS INTEGER) AS nominal_runs_required_before,
    CAST(CASE WHEN s.test_lead_at_start IS NOT NULL THEN s.test_lead_at_start + s.pen_pre + r.runs_before END AS INTEGER)
        AS match_lead_before,
    s.wickets_limit,
    s.batters_absent
FROM running r JOIN int_innings_state s USING (match_id, innings) JOIN matches m USING (match_id)
"""

MODEL_PRE_SQL = """
SELECT * FROM (
    SELECT b.match_id, b.innings, b.delivery_number,
        lag(t.win_probability) OVER w AS win_probability_before,
        lag(t.predicted_score) OVER w AS predicted_score_before
    FROM balls b LEFT JOIN stg_ari_tracking t USING (match_id, innings, delivery_number)
    WINDOW w AS (PARTITION BY b.match_id, b.innings ORDER BY b.delivery_number)
) WHERE win_probability_before IS NOT NULL OR predicted_score_before IS NOT NULL
"""

# One row per Cricsheet ball that has a linked Hawk-Eye row (see hawkeye_link.py)
DELIVERY_SQL = """
SELECT h.match_id, h.innings, h.delivery_number,
    h.speed_kmh, h.right_handed_bat, h.pitch_x, h.pitch_y, h.pitch_x_off_side,
    h.stumps_x, h.stumps_y, h.stumps_x_off_side, h.link_quality AS hawkeye_link_quality
FROM hawkeye_ball_tracking h
"""

OBSERVER_SQL = """
SELECT t.match_id, t.innings, t.delivery_number,
    t.ball_line AS line_cricinfo, t.ball_length AS length_cricinfo,
    t.shot_played AS shot_cricinfo, t.shot_control AS shot_control_cricinfo
FROM stg_ari_tracking t
WHERE coalesce(t.ball_line, t.ball_length, t.shot_played, t.shot_control) IS NOT NULL
"""

# Where the ball went, one table per source (each per-ball table holds one licence family)
WAGON_SQL = """
SELECT t.match_id, t.innings, t.delivery_number,
    t.shot_direction AS direction_cricinfo, t.wagon_x, t.wagon_y, t.wagon_zone, t.wagon_angle,
    t.wagon_distance, t.wagon_in_ring
FROM stg_ari_tracking t
WHERE coalesce(t.shot_direction, CAST(t.wagon_x AS VARCHAR), CAST(t.wagon_zone AS VARCHAR), CAST(t.wagon_angle AS VARCHAR),
               CAST(t.wagon_distance AS VARCHAR), CAST(t.wagon_in_ring AS VARCHAR)) IS NOT NULL
"""
LANDING_SQL = """
SELECT h.match_id, h.innings, h.delivery_number, h.field_x, h.field_y
FROM hawkeye_ball_tracking h WHERE h.field_x IS NOT NULL OR h.field_y IS NOT NULL
"""

POST_SQL = """
SELECT t.match_id, t.innings, t.delivery_number,
    t.win_probability AS win_probability_after, t.predicted_score AS predicted_score_after
FROM stg_ari_tracking t WHERE t.win_probability IS NOT NULL OR t.predicted_score IS NOT NULL
"""


def _pred(field: str) -> str:
    return (f"p.{field} AS {field}_predicted, p.{field}_confidence AS {field}_predicted_conf, "
            f"p.{field}_explicit AS {field}_predicted_explicit")


# Only balls that have a prediction get a row.
PREDICTED_DELIVERY_SQL = f"""
SELECT p.match_id, p.innings, p.delivery_number, {_pred('line')}, {_pred('length')}
FROM stg_ari_predictions p WHERE p.line IS NOT NULL OR p.length IS NOT NULL
"""
PREDICTED_OUTCOME_SQL = f"""
SELECT p.match_id, p.innings, p.delivery_number, {_pred('shot')}, {_pred('direction')}
FROM stg_ari_predictions p WHERE p.shot IS NOT NULL OR p.direction IS NOT NULL
"""

# Silver: third-party per-ball layers, sparse
SQL = {"cricinfo_ball_pre": MODEL_PRE_SQL, "hawkeye_ball_delivery": DELIVERY_SQL,
       "cricinfo_ball_labels": OBSERVER_SQL, "cricinfo_ball_wagon": WAGON_SQL, "hawkeye_ball_landing": LANDING_SQL,
       "cricinfo_ball_post": POST_SQL, "cricinfo_ball_labels_predicted": PREDICTED_DELIVERY_SQL,
       "cricinfo_ball_outcome_predicted": PREDICTED_OUTCOME_SQL}

# Tables with exactly one row per Cricsheet ball; every other layer is sparse.
DENSE = ["int_ball_state_pre"]


def check_registry(con, folders: dict) -> None:
    """Fail the build if a layer table's columns drift from the registry, a dense layer does not have
    exactly one row per ball, or a sparse layer has duplicate keys or a ball Cricsheet does not have.
    `folders` maps each registry table to the folder it was written to."""
    n_balls = con.execute("SELECT count(*) FROM balls").fetchone()[0]
    for table, (_, columns) in LAYERS.items():
        path = folders[table] / f"{table}.parquet"
        actual = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()]
        if actual != KEYS + columns:
            raise SystemExit(f"{table}: columns {actual} do not match the timing registry {KEYS + columns}")
        rows, distinct, orphans = con.execute(f"""
            SELECT count(*), count(DISTINCT (t.match_id, t.innings, t.delivery_number)),
                   count(*) FILTER (b.match_id IS NULL)
            FROM read_parquet('{path}') t LEFT JOIN balls b USING (match_id, innings, delivery_number)""").fetchone()
        if distinct != rows or orphans:
            raise SystemExit(f"{table}: {rows - distinct} duplicate keys, {orphans} rows with no Cricsheet ball")
        if table in DENSE and rows != n_balls:
            raise SystemExit(f"{table}: {rows} rows, expected exactly one per ball ({n_balls})")


def check_no_leak(con) -> dict:
    """Pre-ball values must not move on the ball they describe.

    1. Win probability: the wicket shows in post - pre on its own ball, not in how pre moved onto it.
    2. Match state: on every ball, next.before = this.before + this ball (so `before` never includes it).
    3. Horizon: no chase may show a target or ball limit that the batting side then exceeds
       (that would mean a target fixed after the fact). Judged from the raw data, before the relabelling in
       INNINGS_SQL: a chase whose stated target (or first-innings total + 1 when Cricsheet gives none) or ball
       allocation the batting side passed before its last ball must be published with no target and no limit,
       and no chase published as 'set' may run past its published target or limit. The published state alone
       cannot show the first kind, because INNINGS_SQL relabels those chases 'inconsistent_with_balls'.
    """
    pre_move, wicket_effect, normal_move = con.execute("""
        WITH x AS (
            SELECT b.dismissal_count > 0 AS wicket, s.win_probability_before AS pre, p.win_probability_after AS post,
                lag(s.win_probability_before) OVER (PARTITION BY b.match_id, b.innings ORDER BY b.delivery_number) AS prev_pre
            FROM balls b JOIN cricinfo_ball_pre s USING (match_id, innings, delivery_number)
            JOIN cricinfo_ball_post p USING (match_id, innings, delivery_number)
        )
        SELECT round(avg(pre - prev_pre) FILTER (wicket), 2),
               round(avg(post - pre) FILTER (wicket), 2),
               round(avg(pre - prev_pre) FILTER (NOT wicket), 2)
        FROM x WHERE pre IS NOT NULL AND post IS NOT NULL AND prev_pre IS NOT NULL""").fetchone()
    if wicket_effect is None or wicket_effect > -2 or abs(pre_move - normal_move) > 0.25:
        raise SystemExit(f"leak check failed: pre-ball win prob moved {pre_move} onto wicket balls "
                         f"(normal {normal_move}); wicket effect {wicket_effect}")

    breaks = con.execute("""
        WITH x AS (
            SELECT s.innings_runs_before + b.runs_total AS runs_after, s.innings_wickets_before + b.dismissal_count AS wk_after,
                s.legal_balls_before + CASE WHEN b.is_legal THEN 1 ELSE 0 END AS legal_after,
                lead(s.innings_runs_before) OVER w AS nr, lead(s.innings_wickets_before) OVER w AS nw,
                lead(s.legal_balls_before) OVER w AS nl
            FROM balls b JOIN int_ball_state_pre s USING (match_id, innings, delivery_number)
            WINDOW w AS (PARTITION BY b.match_id, b.innings ORDER BY b.delivery_number)
        )
        SELECT count(*) FROM x WHERE nr IS NOT NULL AND (nr <> runs_after OR nw <> wk_after OR nl <> legal_after)""").fetchone()[0]
    if breaks:
        raise SystemExit(f"leak check failed: {breaks} balls where the next pre-ball state is not this state + this ball")

    exceeded, examples, passed_stated = con.execute("""
        WITH first_innings AS (
            SELECT i.match_id, coalesce(sum(b.runs_total), 0) + coalesce(any_value(i.penalty_runs_pre), 0)
                + coalesce(any_value(i.penalty_runs_post), 0) AS total
            FROM innings i LEFT JOIN balls b USING (match_id, innings)
            WHERE i.innings = 1 GROUP BY i.match_id
        ), extra_balls AS (
            SELECT mi.match_id, mi.innings,
                sum(CAST(json_extract_string(mi.miscounted_overs, '$."' || k || '"."balls"') AS INTEGER) - mm.balls_per_over) AS balls
            FROM innings mi JOIN matches mm USING (match_id), unnest(json_keys(mi.miscounted_overs)) AS u(k)
            WHERE mi.miscounted_overs IS NOT NULL AND CAST(k AS INTEGER) < mm.scheduled_overs GROUP BY ALL
        ), stated AS (
            -- the chase as Cricsheet states it, before any relabelling: an unrevised limited-overs second innings,
            -- its target (first-innings total + 1 when none is given) and its ball allocation (miscounted overs included)
            SELECT i.match_id, i.innings, coalesce(i.target_runs, f.total + 1) AS target,
                m.scheduled_overs * m.balls_per_over + coalesce(x.balls, 0) AS allowed_balls
            FROM innings i JOIN matches m USING (match_id)
            LEFT JOIN first_innings f USING (match_id) LEFT JOIN extra_balls x USING (match_id, innings)
            WHERE i.innings = 2 AND NOT i.is_super_over AND m.scheduled_overs IS NOT NULL AND m.method IS NULL
              AND (i.target_balls IS NULL OR i.target_balls = m.scheduled_overs * m.balls_per_over)
        ), passed AS (
            -- chases that reached that target, or used up that allocation, before a ball they then bowled
            SELECT DISTINCT s.match_id, s.innings
            FROM stated t JOIN int_ball_state_pre s USING (match_id, innings)
            WHERE s.innings_runs_before >= t.target OR s.legal_balls_before >= t.allowed_balls
        )
        SELECT count(DISTINCT (s.match_id, s.innings)), list(DISTINCT s.match_id)[:5],
            (SELECT count(*) FROM passed)
        FROM int_ball_state_pre s LEFT JOIN passed p USING (match_id, innings)
        WHERE (s.target_status = 'set' AND (runs_required_before <= 0 OR balls_remaining_before <= 0))
           OR (p.match_id IS NOT NULL AND (s.target_runs IS NOT NULL OR s.innings_balls_limit IS NOT NULL))""").fetchone()
    if exceeded:
        raise SystemExit(f"leak check failed: {exceeded} chases continue after reaching a 'known' target or "
                         f"ball limit, e.g. {examples}")
    print(f"  leak check: {passed_stated} chases passed their stated target or ball allocation before their last ball; "
          "all published without a target or limit")

    return {"pre_ball_win_prob_move_onto_wicket_ball": pre_move, "pre_ball_move_onto_other_balls": normal_move,
            "win_prob_change_caused_by_wicket": wicket_effect, "state_continuity_breaks": breaks,
            "chases_exceeding_known_target_or_horizon": exceeded}
