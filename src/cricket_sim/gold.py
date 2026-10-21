"""Gold: everything we derive, in four kinds of table (prefix = kind).

  int_*     intermediate derivations over ALL matches (scope-free), e.g. the pre-ball match state.
            Building blocks for the marts; never exported.
  dim_*     one row per thing (match, player, venue), in scope.
  fact_*    one row per event (innings, delivery, dismissal, weather hour), in scope.
  map_*     identifier crosswalks.
  report_*  coverage, quality and audit tables that back the headline numbers.

Scope (config/scope.yml) is applied here and only here: silver keeps every match, so widening the
scope never needs a re-ingest. The export (export.py) reads gold only.
"""

import pyarrow as pa

from . import ball_layers
from .context_players import MAX_LAST_MATCH_AGE
from .paths import GOLD, SUBSET, load_yaml
from .wiki_fixtures import link_reference, load_link_overrides


SCOPE_KEYS = {"gender", "from_date", "full_members", "segments"}
SEGMENT_REQUIRED = ("name", "match_type")
SEGMENT_KEYS = set(SEGMENT_REQUIRED) | {"team_type", "event_name", "both_teams_in"}


def validate_scope(scope) -> None:
    """Refuse a config/scope.yml that would silently widen or relabel a segment.

    _segment_case builds a CASE from whatever keys a segment has, so a dropped condition is not an error
    there: removing `match_type` from the ODI segment made a T20 match an ODI. Every segment must have a
    name and a match_type, may only use the known condition keys, and may only name a list that exists."""
    problems = []
    if not isinstance(scope, dict):
        raise SystemExit("config/scope.yml: expected a mapping at the top level")
    for key in sorted(SCOPE_KEYS - scope.keys()):
        problems.append(f"missing top-level key '{key}'")
    for key in sorted(scope.keys() - SCOPE_KEYS):
        problems.append(f"unknown top-level key '{key}'")
    if "full_members" in scope and not (isinstance(scope["full_members"], list) and scope["full_members"]
                                        and all(isinstance(t, str) for t in scope["full_members"])):
        problems.append("'full_members' must be a non-empty list of team names")
    segments = scope.get("segments")
    if "segments" in scope and not (isinstance(segments, list) and segments):
        problems.append("'segments' must be a non-empty list")
        segments = []
    names = []
    for n, seg in enumerate(segments or [], start=1):
        label = f"segment {seg.get('name')!r}" if isinstance(seg, dict) and seg.get("name") else f"segment #{n}"
        if not isinstance(seg, dict):
            problems.append(f"{label}: expected a mapping")
            continue
        names.append(seg.get("name"))
        problems += [f"{label}: missing required key '{key}'" for key in SEGMENT_REQUIRED if not seg.get(key)]
        problems += [f"{label}: unknown key '{key}' (allowed: {', '.join(sorted(SEGMENT_KEYS))})"
                     for key in sorted(seg.keys() - SEGMENT_KEYS)]
        problems += [f"{label}: '{key}' must be text" for key, value in seg.items()
                     if key in SEGMENT_KEYS and value is not None and not isinstance(value, str)]
        if isinstance(seg.get("both_teams_in"), str) and not isinstance(scope.get(seg["both_teams_in"]), list):
            problems.append(f"{label}: both_teams_in names '{seg['both_teams_in']}', which is not a list in scope.yml")
    problems += [f"duplicate segment name '{name}'" for name in sorted({n for n in names if names.count(n) > 1})]
    if problems:
        raise SystemExit("config/scope.yml is not valid:\n  " + "\n  ".join(problems))


def _segment_case(scope: dict, alias: str = "m") -> str:
    whens = []
    for seg in scope["segments"]:
        conds = []
        for key, value in seg.items():
            if key == "name":
                continue
            if key == "both_teams_in":
                teams = ", ".join("'" + t.replace("'", "''") + "'" for t in scope[value])
                conds.append(f"{alias}.team1 IN ({teams}) AND {alias}.team2 IN ({teams})")
            else:
                conds.append(f"{alias}.{key} = '{value}'")
        whens.append(f"WHEN {' AND '.join(conds)} THEN '{seg['name']}'")
    return "CASE " + " ".join(whens) + " END"


def _full_members(scope: dict) -> str:
    return ", ".join("'" + t.replace("'", "''") + "'" for t in scope["full_members"])


def _pct(expr: str) -> str:
    return f"round(100.0 * avg(CASE WHEN {expr} THEN 1 ELSE 0 END), 1)"


# Per-match checks are tri-state: true = passed, false = failed, NULL = not checkable for this match
# (e.g. a rain-revised result). A match passes only if no check is false and it was not awarded.
QUALITY_SQL = """
WITH totals AS (
    SELECT i.match_id, i.innings, i.team, i.is_super_over,
        coalesce(sum(b.runs_total), 0) + coalesce(any_value(i.penalty_runs_pre), 0) + coalesce(any_value(i.penalty_runs_post), 0) AS runs,
        -- what the ball stream alone shows: delivery runs and pre-innings penalties, no post-innings award
        coalesce(sum(b.runs_total), 0) + coalesce(any_value(i.penalty_runs_pre), 0) AS ball_runs,
        coalesce(sum(b.dismissal_count), 0) AS dismissals
    FROM innings i LEFT JOIN balls b USING (match_id, innings) GROUP BY ALL
), ball_checks AS (
    SELECT match_id,
        bool_and(runs_total = runs_batter + runs_extras) AS runs_add_up,
        bool_and(runs_extras = wides + noballs + byes + legbyes + penalty) AS extras_add_up
    FROM balls GROUP BY match_id
), overs AS (
    -- an over may differ from balls_per_over only if Cricsheet lists THAT over as miscounted
    SELECT b.match_id, b.innings, b.over, count(*) FILTER (b.is_legal) AS legal, any_value(m.balls_per_over) AS bpo,
        json_extract(any_value(i.miscounted_overs), '$."' || b.over || '"') IS NOT NULL AS listed,
        b.over = max(b.over) OVER (PARTITION BY b.match_id, b.innings) AS last_over
    FROM balls b JOIN matches m USING (match_id) JOIN innings i USING (match_id, innings)
    GROUP BY b.match_id, b.innings, b.over
), over_checks AS (
    SELECT match_id, bool_and(listed OR legal = bpo OR (last_over AND legal < bpo)) AS overs_consistent
    FROM overs WHERE bpo IS NOT NULL GROUP BY match_id
), wicket_checks AS (
    SELECT match_id, bool_and(dismissals <= CASE WHEN is_super_over THEN 2 ELSE 10 END) AS wickets_within_limit
    FROM totals GROUP BY match_id
), limited AS (
    SELECT m.match_id, m.winner, m.win_by_runs, m.win_by_wickets, m.method, m.result,
        t1.team AS team1_bat, t1.runs AS r1, t2.team AS team2_bat, t2.runs AS r2, t2.dismissals AS d2,
        t1.ball_runs AS b1, t2.ball_runs AS b2,
        i2.target_runs, i2.target_balls, m.scheduled_overs * m.balls_per_over AS sched_balls
    FROM matches m
    JOIN totals t1 ON t1.match_id = m.match_id AND t1.innings = 1
    JOIN totals t2 ON t2.match_id = m.match_id AND t2.innings = 2
    JOIN innings i2 ON i2.match_id = m.match_id AND i2.innings = 2
    WHERE m.scheduled_overs IS NOT NULL
), limited_checks AS (
    SELECT match_id,
        -- unrevised chase target is first-innings total + 1
        CASE WHEN method IS NULL AND target_runs IS NOT NULL AND target_balls = sched_balls
             THEN target_runs = r1 + 1 END AS target_matches,
        -- the stated margin must follow from the innings totals (not checkable after rain or an award)
        CASE WHEN method IS NOT NULL OR winner IS NULL THEN NULL
             WHEN win_by_runs IS NOT NULL THEN winner = team1_bat AND r1 - r2 = win_by_runs
             WHEN win_by_wickets IS NOT NULL THEN winner = team2_bat AND r2 > r1 END AS result_consistent,
        -- the winner must also follow from the balls alone; false when only a post-innings penalty award
        -- (not timed, not on any ball) gets the recorded winner there (match 1420222)
        CASE WHEN method IS NOT NULL OR winner IS NULL THEN NULL
             WHEN win_by_runs IS NOT NULL THEN winner = team1_bat AND b1 > b2
             WHEN win_by_wickets IS NOT NULL THEN winner = team2_bat AND b2 > b1 END AS result_in_ball_stream
    FROM limited
), test_checks AS (
    SELECT m.match_id,
        CASE WHEN m.method IS NOT NULL OR m.winner IS NULL OR m.win_by_innings IS NOT NULL THEN NULL
             WHEN m.win_by_runs IS NOT NULL THEN
                 (SELECT sum(CASE WHEN t.team = m.winner THEN t.runs ELSE -t.runs END) FROM totals t WHERE t.match_id = m.match_id) = m.win_by_runs
             WHEN m.win_by_wickets IS NOT NULL THEN
                 (SELECT sum(CASE WHEN t.team = m.winner THEN t.runs ELSE -t.runs END) FROM totals t WHERE t.match_id = m.match_id) > 0
        END AS result_consistent,
        CASE WHEN m.method IS NOT NULL OR m.winner IS NULL OR m.win_by_innings IS NOT NULL THEN NULL
             WHEN m.win_by_runs IS NOT NULL OR m.win_by_wickets IS NOT NULL THEN
                 (SELECT sum(CASE WHEN t.team = m.winner THEN t.ball_runs ELSE -t.ball_runs END) FROM totals t WHERE t.match_id = m.match_id) > 0
        END AS result_in_ball_stream
    FROM matches m WHERE m.scheduled_overs IS NULL
), checks AS (
    SELECT m.match_id,
        bc.runs_add_up, bc.extras_add_up, oc.overs_consistent, wc.wickets_within_limit,
        lc.target_matches,
        coalesce(lc.result_consistent, tc.result_consistent) AS result_consistent,
        coalesce(lc.result_in_ball_stream, tc.result_in_ball_stream) AS result_in_ball_stream,
        coalesce(m.method = 'Awarded', false) AS is_awarded,
        m.n_innings > 0 AS has_balls
    FROM matches m
    LEFT JOIN ball_checks bc USING (match_id)
    LEFT JOIN over_checks oc USING (match_id)
    LEFT JOIN wicket_checks wc USING (match_id)
    LEFT JOIN limited_checks lc USING (match_id)
    LEFT JOIN test_checks tc USING (match_id)
)
SELECT *,
    -- only an explicit false fails; coalesce must wrap the whole OR, or one NULL check leaks NULL
    coalesce(NOT (runs_add_up IS FALSE OR extras_add_up IS FALSE OR overs_consistent IS FALSE
        OR wickets_within_limit IS FALSE OR target_matches IS FALSE OR result_consistent IS FALSE
        OR result_in_ball_stream IS FALSE), true)
        AND NOT is_awarded AS passes_quality
FROM checks
"""


def _reference_where(scope: dict) -> str:
    """The reference rows the coverage report covers (scope start, known result, full-member T20Is)."""
    return f"""start_date >= DATE '{scope['from_date']}' AND result_class <> 'unknown'
              AND (match_format <> 'T20I' OR (team1 IN ({_full_members(scope)}) AND team2 IN ({_full_members(scope)})))"""


def reference_link_inputs(con, scope: dict) -> tuple[list[dict], list[dict]]:
    """(reference rows, Cricsheet internationals) as wiki_fixtures.link_reference reads them; needs the views
    wikipedia_fixtures and dim_match."""
    refs = con.execute(f"""SELECT match_format AS segment, match_number, start_date, team1, team2, cricinfo_match_id
        FROM wikipedia_fixtures WHERE {_reference_where(scope)}""").fetchall()
    cs_rows = con.execute("""SELECT segment, match_id, match_type_number, start_date, team1, team2 FROM dim_match
        WHERE segment IN ('Test', 'ODI', 'T20I')""").fetchall()
    ref_keys = ("segment", "match_number", "start_date", "team1", "team2", "cricinfo_match_id")
    cs_keys = ("segment", "match_id", "match_type_number", "start_date", "team1", "team2")
    return [dict(zip(ref_keys, r)) for r in refs], [dict(zip(cs_keys, r)) for r in cs_rows]


def build_gold(con) -> dict:
    scope = load_yaml("scope.yml")
    validate_scope(scope)

    def write(name: str, sql: str, order_by: str) -> None:
        path = GOLD / f"{name}.parquet"
        con.execute(f"COPY (SELECT * FROM ({sql}) ORDER BY {order_by}) TO '{path}' (FORMAT parquet, COMPRESSION zstd)")
        con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{path}')")

    # ---- intermediate: all matches ---------------------------------------------------------------
    write("int_innings_state", ball_layers.INNINGS_SQL, "match_id, innings")
    write("int_ball_state_pre", ball_layers.STATE_PRE_SQL, "match_id, innings, delivery_number")
    write("int_match_quality", QUALITY_SQL, "match_id")
    nulls = con.execute("SELECT count(*) FROM int_match_quality WHERE passes_quality IS NULL").fetchone()[0]
    if nulls:
        raise SystemExit(f"int_match_quality: passes_quality is NULL for {nulls} matches (a check leaked NULL)")

    # ---- scope -----------------------------------------------------------------------------------
    con.execute(f"""CREATE OR REPLACE TEMP TABLE scope AS
        SELECT m.match_id, {_segment_case(scope)} AS segment FROM matches m
        WHERE m.gender = '{scope['gender']}' AND m.start_date >= DATE '{scope['from_date']}'
          AND ({_segment_case(scope)}) IS NOT NULL""")

    # ---- dimensions ------------------------------------------------------------------------------
    write("dim_match", """
        SELECT m.match_id, s.segment, m.match_type, m.match_type_number, m.gender, m.team_type,
            m.event_name, m.event_match_number, m.event_stage, m.season, m.season_year,
            m.start_date, m.end_date, m.n_dates, v.venue_id, m.venue AS venue_name, m.city,
            m.team1, m.team2, m.toss_winner, m.toss_decision,
            m.result, m.winner, m.win_by_runs, m.win_by_wickets, m.win_by_innings, m.method,
            m.winner_after_tiebreak, m.tiebreak_method,
            m.scheduled_overs, m.balls_per_over, m.n_innings, m.player_of_match,
            m.umpires, m.tv_umpires, m.match_referees, m.supersubs,
            q.passes_quality, m.cricsheet_missing AS cricsheet_missing_fields,
            m.data_version AS cricsheet_data_version, m.revision AS cricsheet_revision
        FROM matches m JOIN scope s USING (match_id)
        JOIN int_match_quality q USING (match_id)
        LEFT JOIN match_venues v USING (match_id)""", "segment, start_date, match_id")

    # a whole-snapshot dimension: a birth date implying an age over 50 at the player's last match is left out here
    write("dim_player", """
        SELECT p.* REPLACE (CASE WHEN p.dob_flag = 'implausible_age_at_last_match' THEN NULL ELSE p.dob END AS dob)
        FROM players p
        WHERE p.cricsheet_id IN (SELECT mp.player_id FROM match_players mp JOIN scope USING (match_id))""",
          "cricsheet_id")

    write("dim_venue", """
        SELECT v.* FROM venues v WHERE v.venue_id IN (SELECT venue_id FROM dim_match)""", "venue_id")

    # ---- facts -----------------------------------------------------------------------------------
    write("fact_innings", """
        WITH totals AS (
            SELECT match_id, innings, count(*) AS deliveries, count(*) FILTER (is_legal) AS legal_balls,
                sum(runs_total) AS runs_off_balls, sum(dismissal_count) AS dismissals
            FROM balls GROUP BY ALL
        )
        SELECT i.match_id, i.innings, i.team AS batting_team, i.is_super_over, i.declared, i.forfeited,
            s.scheduled_balls_limit, s.nominal_target_runs,
            s.innings_balls_limit, s.innings_length_status, s.wickets_limit,
            s.target_runs, s.target_status, s.test_lead_at_start,
            i.target_runs AS final_target_runs, i.target_overs AS final_target_overs,
            i.target_balls AS final_target_balls,
            coalesce(i.penalty_runs_pre, 0) AS penalty_runs_pre, coalesce(i.penalty_runs_post, 0) AS penalty_runs_post,
            s.batters_absent, i.absent_hurt, i.miscounted_overs,
            coalesce(t.deliveries, 0) AS deliveries, coalesce(t.legal_balls, 0) AS legal_balls,
            CAST(coalesce(t.dismissals, 0) AS SMALLINT) AS dismissals,
            CAST(coalesce(t.runs_off_balls, 0) + coalesce(i.penalty_runs_pre, 0) + coalesce(i.penalty_runs_post, 0) AS INTEGER) AS runs
        FROM innings i JOIN scope USING (match_id)
        JOIN int_innings_state s USING (match_id, innings)
        LEFT JOIN totals t USING (match_id, innings)""", "match_id, innings")

    # The product: one row per ball, the state BEFORE it (from earlier balls only), then what happened.
    write("fact_delivery", """
        SELECT b.match_id, b.innings, b.delivery_number, b.batting_team, b.is_super_over,
            b.over, b.ball_in_over, b.actual_delivery, b.legal_balls_in_over_before,
            b.batter_id, b.batter, b.non_striker_id, b.non_striker, b.bowler_id, b.bowler,
            s.innings_runs_before, s.innings_wickets_before, s.legal_balls_before,
            s.scheduled_balls_limit, s.scheduled_balls_remaining_before,
            s.nominal_target_runs, s.nominal_runs_required_before,
            s.match_lead_before, s.wickets_limit,
            -- hindsight: cross-checked against how the innings went (choose rows with these, never predict with them)
            s.batters_absent, s.innings_balls_limit, s.balls_remaining_before, s.innings_length_status,
            s.target_runs, s.runs_required_before, s.target_status,
            b.is_legal, b.runs_batter, b.runs_extras, b.runs_total, b.runs_non_boundary,
            b.runs_batter = 4 AND NOT b.runs_non_boundary AS is_four,
            b.runs_batter = 6 AND NOT b.runs_non_boundary AS is_six,
            b.wides, b.noballs, b.byes, b.legbyes, b.penalty, b.noball_includes_byes,
            b.wicket_count, b.dismissal_count, b.has_review
        FROM balls b JOIN scope USING (match_id)
        JOIN int_ball_state_pre s USING (match_id, innings, delivery_number)""",
          "match_id, innings, delivery_number")

    write("fact_dismissal", "SELECT w.* FROM wickets w JOIN scope USING (match_id)",
          "match_id, innings, delivery_number, wicket_number")

    # `phase` is excluded: it splits pre-match / in-play hours with Cricinfo ball timestamps.
    # `venue_version` is a silver join key only; the published hourly table keeps its columns.
    write("fact_match_weather_hourly", """
        SELECT w.* EXCLUDE (phase, venue_version) FROM match_weather_hourly w JOIN scope USING (match_id)""",
          "match_id, match_day_number, utc_datetime")

    # ---- maps ------------------------------------------------------------------------------------
    write("map_venue_alias", "SELECT * FROM venue_aliases", "venue, city")

    # ---- reports ---------------------------------------------------------------------------------
    write("report_match_quality", """
        SELECT q.* FROM int_match_quality q JOIN scope USING (match_id)""", "match_id")

    write("report_coverage_by_segment_year", f"""
        SELECT m.segment, year(m.start_date) AS year,
            count(DISTINCT m.match_id) AS matches,
            count(DISTINCT m.match_id) FILTER (m.passes_quality) AS matches_passing_quality,
            count(*) AS balls,
            {_pct("r.line_cricinfo IS NOT NULL")} AS pct_line_cricinfo,
            {_pct("r.length_cricinfo IS NOT NULL")} AS pct_length_cricinfo,
            {_pct("r.shot_cricinfo IS NOT NULL")} AS pct_shot_cricinfo,
            {_pct("r.shot_control_cricinfo IS NOT NULL")} AS pct_shot_control_cricinfo,
            {_pct("x.wagon_x IS NOT NULL")} AS pct_wagon,
            {_pct("d.speed_kmh IS NOT NULL")} AS pct_speed_hawkeye,
            {_pct("d.pitch_x IS NOT NULL")} AS pct_pitch_xy_hawkeye,
            {_pct("p.win_probability_after IS NOT NULL")} AS pct_win_probability,
            {_pct("r.line_cricinfo IS NULL AND lp.line_predicted IS NOT NULL")} AS pct_line_predicted_only,
            round(avg(lp.line_predicted_conf) FILTER (r.line_cricinfo IS NULL), 3) AS mean_conf_line_predicted_only
        FROM dim_match m
        JOIN balls b USING (match_id)
        LEFT JOIN hawkeye_ball_delivery d USING (match_id, innings, delivery_number)
        LEFT JOIN cricinfo_ball_labels r USING (match_id, innings, delivery_number)
        LEFT JOIN cricinfo_ball_wagon x USING (match_id, innings, delivery_number)
        LEFT JOIN cricinfo_ball_post p USING (match_id, innings, delivery_number)
        LEFT JOIN cricinfo_ball_labels_predicted lp USING (match_id, innings, delivery_number)
        GROUP BY ALL""", "segment, year")

    # Every numbered international in the reference list (Wikipedia season pages), linked one-to-one to Cricsheet
    # by wiki_fixtures.link_reference: the strict rule (the two teams agree, dates within a day; id, then number,
    # rank the candidates), best pairs first, then the reviewed exceptions in config/link_overrides.yml. Unlinked
    # matches get a named reason.
    ref_where = _reference_where(scope)
    refs, cs_rows = reference_link_inputs(con, scope)
    links = link_reference(refs, cs_rows, load_link_overrides(), check_stale=not SUBSET)
    con.register("ref_links", pa.Table.from_pylist(links, schema=pa.schema([
        ("segment", pa.string()), ("match_number", pa.int32()), ("cricsheet_match_id", pa.string()),
        ("link_method", pa.string())])))
    write("report_international_coverage", f"""
        WITH ref AS (
            SELECT match_format AS segment, *, list_sort([team1, team2]) AS team_set
            FROM wikipedia_fixtures WHERE {ref_where}
        ), cs AS (
            SELECT d.segment, d.start_date FROM dim_match d WHERE d.segment IN ('Test', 'ODI', 'T20I')
        ), links AS (SELECT * FROM ref_links),
        snapshot_end AS (SELECT max(start_date) AS last_date FROM dim_match),  -- last in-scope match held
        coverage_start AS (SELECT segment, min(start_date) AS first_date FROM cs GROUP BY segment),
        linked AS (
            SELECT r.*, l.cricsheet_match_id, l.link_method,
                'Afghanistan' IN (r.team1, r.team2) OR (r.team1 IS NULL AND r.involves_afghanistan) AS afghanistan,
                EXISTS (SELECT 1 FROM cricsheet_missing m WHERE m.gender = 'male' AND m.list_type = 'match_type'
                        AND m.list_name = CASE r.segment WHEN 'ODI' THEN 'Odi' ELSE r.segment END
                        AND abs(m.start_date - r.start_date) <= 1
                        -- no reference teams: the date alone
                        AND (r.team1 IS NULL OR list_sort(string_split(m.teams, ' vs ')) = r.team_set)) AS on_cricsheet_missing_list,
                r.start_date < cs0.first_date AS before_cricsheet_coverage,
                r.start_date > (SELECT last_date FROM snapshot_end) AS after_snapshot
            FROM ref r
            LEFT JOIN links l USING (segment, match_number)
            LEFT JOIN coverage_start cs0 USING (segment)
        )
        SELECT segment, match_number, start_date, team1, team2, result_class, cricinfo_match_id,
            cricsheet_match_id, link_method, teams_source, duplicate_conflict,
            CASE WHEN cricsheet_match_id IS NOT NULL THEN 'in_cricsheet'
                 -- newer than the last match in the pinned Cricsheet snapshot: not a gap
                 WHEN after_snapshot THEN 'after_snapshot'
                 WHEN is_abandoned_no_ball THEN 'abandoned_without_a_ball'
                 WHEN afghanistan THEN 'afghanistan_withheld_by_cricsheet'
                 WHEN before_cricsheet_coverage THEN 'before_cricsheet_coverage'
                 WHEN on_cricsheet_missing_list THEN 'cricsheet_known_missing'
                 -- Cricsheet holds no match without a ball bowled; Wikipedia often just says "No result"
                 WHEN result_class = 'no_result' THEN 'no_result_possibly_no_play'
                 WHEN year(start_date) = 2002 THEN 'missing_unlisted_2002_sparse_year'
                 ELSE 'missing_unlisted' END AS status,
            page_title, page_revid
        FROM linked""", "segment, match_number")

    # Cricsheet internationals the reference does not list (T20I reference is incomplete after 2019)
    write("report_cricsheet_not_in_reference", """
        SELECT d.segment, d.match_id, d.match_type_number, d.start_date FROM dim_match d
        WHERE d.segment IN ('Test', 'ODI', 'T20I')
          AND d.match_id NOT IN (SELECT cricsheet_match_id FROM report_international_coverage
                                 WHERE cricsheet_match_id IS NOT NULL)""", "segment, start_date, match_id")

    # Openly licensed crosswalk: Cricsheet match id <-> Cricinfo match id <-> official match number
    write("map_match_ids", """
        SELECT c.cricsheet_match_id AS match_id, c.segment, d.match_type_number AS cricsheet_number,
            c.match_number AS reference_number,
            -- only a confirmed id: the reference's id when it is this match's own, so it is one-to-one
            CASE WHEN CAST(c.cricinfo_match_id AS VARCHAR) = c.cricsheet_match_id THEN c.cricinfo_match_id END
                AS cricinfo_match_id, c.link_method,
            d.match_type_number = c.match_number AS numbers_agree, abs(d.start_date - c.start_date) AS date_diff_days,
            c.page_title, c.page_revid
        FROM report_international_coverage c JOIN dim_match d ON d.match_id = c.cricsheet_match_id""",
          "segment, match_id")
    dupes = con.execute("SELECT count(*) - count(DISTINCT match_id) FROM map_match_ids").fetchone()[0]
    if dupes:
        raise SystemExit(f"map_match_ids: {dupes} Cricsheet matches linked to more than one reference match")

    write("report_cricsheet_known_missing", f"""
        SELECT list_name, start_date, teams FROM cricsheet_missing
        WHERE gender = '{scope['gender']}' AND start_date >= DATE '{scope['from_date']}'
          AND ((list_type = 'match_type' AND list_name IN ('Test', 'Odi'))
               OR (list_type = 'competition' AND lower(list_name) = 'indian premier league'))""",
          "list_name, start_date, teams")

    write("report_ipl_seasons", """
        SELECT year(start_date) AS season_year, count(*) AS matches,
            count(*) FILTER (passes_quality) AS matches_passing_quality
        FROM dim_match WHERE segment = 'IPL' GROUP BY ALL""", "season_year")

    # Ari's Cricinfo tracking is joined to Cricsheet on (match_id, innings, delivery_number). Proof that those keys
    # mean the same ball: every row must land on a Cricsheet delivery, and Cricinfo's DRS-review events must sit on
    # the balls Cricsheet records reviews for. A key shifted by one ball would move them to the next ball instead.
    write("report_cricinfo_alignment", """
        WITH reviewed AS (SELECT DISTINCT match_id, innings, delivery_number FROM reviews),
        t AS (SELECT match_id, innings, delivery_number, coalesce(ball_events, '') LIKE '%drs review%' AS drs
              FROM stg_ari_tracking)
        SELECT t.match_id, count(*) AS tracking_rows,
            count(*) FILTER (b.match_id IS NULL) AS unmatched_rows,
            count(*) FILTER (t.drs) AS drs_events,
            count(*) FILTER (t.drs AND r0.match_id IS NOT NULL) AS drs_same_ball,
            count(*) FILTER (t.drs AND r1.match_id IS NOT NULL) AS drs_next_ball,
            count(*) FILTER (t.drs) >= 3
                AND count(*) FILTER (t.drs AND r1.match_id IS NOT NULL) > count(*) FILTER (t.drs AND r0.match_id IS NOT NULL)
                AS alignment_suspect
        FROM t
        LEFT JOIN balls b USING (match_id, innings, delivery_number)
        LEFT JOIN reviewed r0 USING (match_id, innings, delivery_number)
        LEFT JOIN reviewed r1 ON r1.match_id = t.match_id AND r1.innings = t.innings
                             AND r1.delivery_number = t.delivery_number + 1
        GROUP BY t.match_id""", "match_id")

    _build_player_form(write)

    write("report_hawkeye_summary", """
        SELECT competition, count(DISTINCT hawkeye_match_id) AS matches, count(*) AS balls,
            -- -1 and 0 mean "not tracked"
            round(100.0 * avg(CASE WHEN ball_speed_mps > 0 THEN 1 ELSE 0 END), 1) AS pct_speed,
            round(100.0 * avg(CASE WHEN abs(pitch_x) < 5 THEN 1 ELSE 0 END), 1) AS pct_pitch_xy
        FROM stg_hawkeye_balls GROUP BY competition""", "competition")

    return _headline(con, scope)


FORMAT_FAMILY = """CASE WHEN m.match_type IN ('Test', 'MDM') THEN 'multi_day'
    WHEN m.match_type IN ('ODI', 'ODM') THEN 'one_day' ELSE 't20' END"""
# the death overs: the last 10 overs of a one-day innings, the last 4 of a T20 (none in multi-day cricket)
DEATH = "CASE fam WHEN 'one_day' THEN b.over >= 40 WHEN 't20' THEN b.over >= 16 ELSE false END"
BOWLER_KINDS = "('caught', 'bowled', 'lbw', 'caught and bowled', 'stumped', 'hit wicket')"


PLAYER_STATS_SQL = f"""
        WITH m AS (SELECT m.match_id, m.start_date, m.gender, {FORMAT_FAMILY} AS fam FROM matches m),
        b AS (SELECT b.*, m.fam, {DEATH} AS death FROM balls b JOIN m USING (match_id) WHERE NOT b.is_super_over),
        bat AS (
            SELECT match_id, batter_id AS player_id,
                count(*) FILTER (wides = 0) AS bat_balls, sum(runs_batter) AS bat_runs,
                count(*) FILTER (wides = 0 AND runs_batter = 4 AND NOT runs_non_boundary) AS bat_fours,
                count(*) FILTER (wides = 0 AND runs_batter = 6 AND NOT runs_non_boundary) AS bat_sixes,
                count(*) FILTER (wides = 0 AND runs_batter = 0) AS bat_dots,
                count(*) FILTER (wides = 0 AND death) AS bat_death_balls,
                coalesce(sum(runs_batter) FILTER (death), 0) AS bat_death_runs
            FROM b GROUP BY ALL),
        outs AS (
            SELECT w.match_id, w.player_out_id AS player_id, count(*) AS bat_outs
            FROM wickets w JOIN b USING (match_id, innings, delivery_number)
            WHERE NOT w.is_retirement GROUP BY ALL),
        bowl AS (
            SELECT match_id, bowler_id AS player_id, count(*) FILTER (is_legal) AS bowl_balls,
                sum(runs_batter + wides + noballs) AS bowl_runs,
                count(*) FILTER (is_legal AND death) AS bowl_death_balls,
                coalesce(sum(runs_batter + wides + noballs) FILTER (death), 0) AS bowl_death_runs
            FROM b GROUP BY ALL),
        bwk AS (
            SELECT w.match_id, b.bowler_id AS player_id, count(*) AS bowl_wickets
            FROM wickets w JOIN b USING (match_id, innings, delivery_number)
            WHERE w.kind IN {BOWLER_KINDS} GROUP BY ALL),
        mp AS (SELECT match_id, player_id, min(team) AS team FROM match_players GROUP BY ALL)
        SELECT mp.match_id, mp.player_id, mp.team, m.start_date, m.gender, m.fam AS format_family,
            CAST(coalesce(bat.bat_balls, 0) AS BIGINT) AS bat_balls, CAST(coalesce(bat.bat_runs, 0) AS BIGINT) AS bat_runs,
            CAST(coalesce(outs.bat_outs, 0) AS BIGINT) AS bat_outs, CAST(coalesce(bat.bat_fours, 0) AS BIGINT) AS bat_fours,
            CAST(coalesce(bat.bat_sixes, 0) AS BIGINT) AS bat_sixes, CAST(coalesce(bat.bat_dots, 0) AS BIGINT) AS bat_dots,
            CAST(coalesce(bat.bat_death_balls, 0) AS BIGINT) AS bat_death_balls, CAST(coalesce(bat.bat_death_runs, 0) AS BIGINT) AS bat_death_runs,
            CAST(coalesce(bowl.bowl_balls, 0) AS BIGINT) AS bowl_balls, CAST(coalesce(bowl.bowl_runs, 0) AS BIGINT) AS bowl_runs,
            CAST(coalesce(bwk.bowl_wickets, 0) AS BIGINT) AS bowl_wickets,
            CAST(coalesce(bowl.bowl_death_balls, 0) AS BIGINT) AS bowl_death_balls, CAST(coalesce(bowl.bowl_death_runs, 0) AS BIGINT) AS bowl_death_runs
        FROM mp JOIN m USING (match_id)
        LEFT JOIN bat USING (match_id, player_id) LEFT JOIN outs USING (match_id, player_id)
        LEFT JOIN bowl USING (match_id, player_id) LEFT JOIN bwk USING (match_id, player_id)"""

_SUMS = ["bat_balls", "bat_runs", "bat_outs", "bat_fours", "bat_sixes", "bat_dots", "bat_death_balls",
        "bat_death_runs", "bowl_balls", "bowl_runs", "bowl_wickets", "bowl_death_balls", "bowl_death_runs"]
_CAREER = ", ".join(f"CAST(coalesce(sum({c}) OVER before, 0) AS BIGINT) AS prior_{c}" for c in _SUMS)
_RECENT = ", ".join(f"CAST(coalesce(sum({c}) OVER last_year, 0) AS BIGINT) AS prior_365d_{c}"
                   for c in ("bat_balls", "bat_runs", "bat_outs", "bowl_balls", "bowl_runs", "bowl_wickets"))
# reads int_player_match_stats, the `scope` table and `players`
PLAYER_FORM_SQL = f"""
        WITH w AS (
            SELECT match_id, player_id, team, start_date, format_family,
                CAST(count(*) OVER before AS BIGINT) AS prior_matches,
                CAST(count(*) FILTER (bat_balls > 0) OVER before AS BIGINT) AS prior_batting_matches,
                CAST(count(*) FILTER (bowl_balls > 0) OVER before AS BIGINT) AS prior_bowling_matches,
                CAST(count(*) OVER last_year AS BIGINT) AS prior_365d_matches,
                max(start_date) OVER before AS last_match_date, min(start_date) OVER before AS first_match_date,
                {_CAREER}, {_RECENT}
            FROM int_player_match_stats
            WINDOW before AS (PARTITION BY player_id, format_family ORDER BY start_date
                              RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL 1 DAY PRECEDING),
                   last_year AS (PARTITION BY player_id, format_family ORDER BY start_date
                                 RANGE BETWEEN INTERVAL 365 DAY PRECEDING AND INTERVAL 1 DAY PRECEDING))
        SELECT w.match_id, w.player_id, w.team, w.format_family, w.prior_matches, w.prior_batting_matches,
            w.prior_bowling_matches, w.first_match_date AS prior_first_match_date,
            datediff('day', w.last_match_date, w.start_date) AS days_since_last_match,
            -- as of THIS match: no age where it would be over {MAX_LAST_MATCH_AGE} (an implausible birth date)
            CASE WHEN datediff('day', p.dob, w.start_date) / 365.25 <= {MAX_LAST_MATCH_AGE}
                 THEN CAST(floor(datediff('day', p.dob, w.start_date) / 365.25) AS INTEGER) END AS age_years,
            w.* EXCLUDE (match_id, player_id, team, start_date, format_family, prior_matches, prior_batting_matches,
                         prior_bowling_matches, last_match_date, first_match_date),
            round(100.0 * w.prior_bat_runs / nullif(w.prior_bat_balls, 0), 2) AS prior_strike_rate,
            round(w.prior_bat_runs / nullif(w.prior_bat_outs, 0), 2) AS prior_batting_average,
            round(100.0 * w.prior_bat_death_runs / nullif(w.prior_bat_death_balls, 0), 2) AS prior_death_strike_rate,
            round(6.0 * w.prior_bowl_runs / nullif(w.prior_bowl_balls, 0), 3) AS prior_economy,
            round(w.prior_bowl_runs / nullif(w.prior_bowl_wickets, 0), 2) AS prior_bowling_average,
            round(6.0 * w.prior_bowl_death_runs / nullif(w.prior_bowl_death_balls, 0), 3) AS prior_death_economy,
            round(100.0 * w.prior_365d_bat_runs / nullif(w.prior_365d_bat_balls, 0), 2) AS prior_365d_strike_rate,
            round(6.0 * w.prior_365d_bowl_runs / nullif(w.prior_365d_bowl_balls, 0), 3) AS prior_365d_economy
        FROM w JOIN scope USING (match_id)
        LEFT JOIN players p ON p.cricsheet_id = w.player_id"""


def _build_player_form(write) -> None:
    """Per player per match: what they did (int_player_match_stats, all matches), and what was known about them
    BEFORE the match (fact_player_match_form, in scope): only matches on strictly earlier dates count."""
    write("int_player_match_stats", PLAYER_STATS_SQL, "player_id, start_date, match_id")
    write("fact_player_match_form", PLAYER_FORM_SQL, "match_id, player_id")

    # Whole-snapshot career summary: handy for browsing, NOT a model input (it includes matches after any given date)
    write("report_player_careers", """
        SELECT s.player_id, s.format_family, s.gender, count(*) AS matches, min(s.start_date) AS first_match_date,
            max(s.start_date) AS last_match_date,
            CAST(sum(s.bat_balls) AS BIGINT) AS bat_balls, CAST(sum(s.bat_runs) AS BIGINT) AS bat_runs,
            CAST(sum(s.bat_outs) AS BIGINT) AS bat_outs, CAST(sum(s.bowl_balls) AS BIGINT) AS bowl_balls,
            CAST(sum(s.bowl_runs) AS BIGINT) AS bowl_runs, CAST(sum(s.bowl_wickets) AS BIGINT) AS bowl_wickets
        FROM int_player_match_stats s WHERE s.player_id IN (SELECT cricsheet_id FROM dim_player)
        GROUP BY ALL""", "player_id, format_family, gender")


def _headline(con, scope: dict) -> dict:
    q = lambda sql: con.execute(sql).fetchone()
    total_matches, total_balls, passing = q(
        "SELECT count(DISTINCT match_id), count(*), count(DISTINCT match_id) FILTER (passes_quality) "
        "FROM dim_match JOIN balls USING (match_id)")
    line_c, length_c, shot_c, wagon, speed, pitch, line_pred, line_pred_conf = q(f"""
        SELECT {_pct("r.line_cricinfo IS NOT NULL")}, {_pct("r.length_cricinfo IS NOT NULL")},
               {_pct("r.shot_cricinfo IS NOT NULL")}, {_pct("x.wagon_x IS NOT NULL")},
               {_pct("d.speed_kmh IS NOT NULL")}, {_pct("d.pitch_x IS NOT NULL")},
               {_pct("r.line_cricinfo IS NULL AND lp.line_predicted IS NOT NULL")},
               round(avg(lp.line_predicted_conf) FILTER (r.line_cricinfo IS NULL), 2)
        FROM fact_delivery f
        LEFT JOIN hawkeye_ball_delivery d USING (match_id, innings, delivery_number)
        LEFT JOIN cricinfo_ball_labels r USING (match_id, innings, delivery_number)
        LEFT JOIN cricinfo_ball_wagon x USING (match_id, innings, delivery_number)
        LEFT JOIN cricinfo_ball_labels_predicted lp USING (match_id, innings, delivery_number)""")
    return {
        "scope": scope,
        "matches_in_scope": total_matches,
        "balls_in_scope": total_balls,
        "matches_passing_quality_pct": round(100.0 * passing / total_matches, 2),
        "pct_balls_line_cricinfo": line_c,
        "pct_balls_length_cricinfo": length_c,
        "pct_balls_shot_cricinfo": shot_c,
        "pct_balls_wagon": wagon,
        "pct_balls_speed_hawkeye": speed,
        "pct_balls_pitch_xy_hawkeye": pitch,
        "predicted_labels_not_counted_above": {
            "pct_balls_line_predicted_only": line_pred, "mean_confidence": line_pred_conf,
            "caveat": "inferred from commentary text; shot/direction predictions are shaped by the outcome"},
        "hawkeye_matches_linked": dict(con.execute(
            "SELECT competition, count(match_id) || '/' || count(*) FROM hawkeye_match_link GROUP BY ALL ORDER BY ALL").fetchall()),
        "international_coverage": {f"{seg} {st}": n for seg, st, n in con.execute(
            "SELECT segment, status, count(*) FROM report_international_coverage GROUP BY ALL ORDER BY ALL").fetchall()},
        "cricsheet_internationals_not_in_reference": dict(con.execute(
            "SELECT segment, count(*) FROM report_cricsheet_not_in_reference GROUP BY ALL ORDER BY ALL").fetchall()),
        "cricsheet_known_missing_in_scope": dict(con.execute(
            "SELECT list_name, count(*) FROM report_cricsheet_known_missing GROUP BY ALL ORDER BY ALL").fetchall()),
        "pct_matches_with_weather": q("SELECT round(100.0 * avg(CASE WHEN EXISTS (SELECT 1 FROM int_match_weather_daily w "
                                      "WHERE w.match_id = d.match_id AND w.weather_available) THEN 1 ELSE 0 END), 1) FROM dim_match d")[0],
        "venues": q("SELECT count(*) || ' canonical venues with weather plans, ' || count(*) FILTER (NOT needs_review) "
                    "|| ' trusted at ground level' FROM venues WHERE in_geo_plan")[0],
        "pct_in_scope_players_with_dob_wikidata": q(
            "SELECT round(100.0 * avg(CASE WHEN dob IS NOT NULL THEN 1 ELSE 0 END), 1) FROM dim_player")[0],
        "cricinfo_alignment": dict(zip(["tracking_rows_unmatched", "drs_events", "drs_on_same_ball_pct",
                                        "drs_on_next_ball_pct", "matches_suspect"], q("""
            SELECT sum(unmatched_rows), sum(drs_events), round(100.0 * sum(drs_same_ball) / nullif(sum(drs_events), 0), 2),
                   round(100.0 * sum(drs_next_ball) / nullif(sum(drs_events), 0), 2), count(*) FILTER (alignment_suspect)
            FROM report_cricinfo_alignment"""))),
    }
