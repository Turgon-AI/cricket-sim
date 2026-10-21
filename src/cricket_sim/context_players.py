"""Openly-licensed player attributes, built over EVERY player in the Cricsheet register that appears in a match
(scope-free: config/scope.yml is never read; gold applies the scope).

Public table  : Wikidata (CC0) + Cricsheet-derived facts (ODC-BY)            -> silver/players.parquet
Restricted    : Ari/Cricinfo styles and roles only (NOT redistributable)     -> silver/cricinfo_players.parquet

Rules (unknown is NULL, never an empty string, never a guess):
  dob       Wikidata day-precision dates, deprecated statements ignored when the snapshot records ranks.
            Several different dates -> NULL + dob_flag='conflict'. 1970-01-01 -> NULL ('placeholder').
            Age at first Cricsheet match outside [14, 50] -> NULL + flag (the rejected value is kept in
            wikidata_dob_quarantine). Age at the LAST match over 50 is only flagged here: that check depends on
            matches played later, so the date stays for gold, which gives no age at a match where the player is
            over 50 (as of that match) and leaves the date out of dim_player.
            1 January dates are kept but flagged 'jan1_low_precision'.
  link      Cricsheet -> Wikidata by ESPNcricinfo id (P2697). Several candidate items whose dates disagree
            -> no link (wikidata_flag='ambiguous').
  country   the national team the player played for in Cricsheet internationals (composite XIs ignored);
            several -> the most recent, flagged. No internationals -> the single Wikidata citizenship if it names a
            cricket nation, else NULL + flag. Citizenships stay available as a list. country_conflict flags a
            disagreement: no citizenship matches the Cricsheet national side(s), or (no Cricsheet internationals) the
            player was born in a nation Cricsheet withholds and no citizenship is that nation.

ingest_wikidata_players() is the only function that touches the network.
"""

import json
import tempfile
import time
from pathlib import Path

import pyarrow as pa
import requests

from . import bronze
from .paths import SILVER, STAGING, load_yaml, snapshot_dir
from .venues import store

SOURCE = "wikidata_players"
ENDPOINT = "https://query.wikidata.org/sparql"
COMPOSITE_TEAMS = ("Africa XI", "Asia XI", "ICC World XI", "World XI")
MIN_DEBUT_AGE, MAX_DEBUT_AGE, MAX_LAST_MATCH_AGE = 14, 50, 50
# Nations whose internationals Cricsheet withholds, so a player can be capped without a Cricsheet international.
WITHHELD_NATIONS = ("Afghanistan",)

_PREFIX = "?s wdt:P31 wd:Q5 ; wdt:P2697 ?cricinfo_id ."


def _label(var: str, out: str) -> str:
    return f'OPTIONAL {{ {var} rdfs:label ?{out} FILTER(lang(?{out}) = "en") }}'


# One narrow query per property avoids the row explosion of one giant OPTIONAL join.
QUERIES = {
    "base": f"""SELECT ?s ?cricinfo_id ?label WHERE {{ {_PREFIX} {_label('?s', 'label')} }} ORDER BY ?s ?cricinfo_id""",
    "dob": f"""SELECT ?s ?cricinfo_id ?dob ?precision ?rank WHERE {{ {_PREFIX}
        ?s p:P569 ?st . ?st psv:P569 ?n ; wikibase:rank ?rank . ?n wikibase:timeValue ?dob ; wikibase:timePrecision ?precision . }}
        ORDER BY ?s ?cricinfo_id ?dob ?rank""",
    "sex": f"""SELECT ?s ?cricinfo_id ?sex ?sex_label WHERE {{ {_PREFIX} ?s wdt:P21 ?sex . {_label('?sex', 'sex_label')} }}
        ORDER BY ?s ?cricinfo_id ?sex""",
    "height": f"""SELECT ?s ?cricinfo_id ?amount ?unit WHERE {{ {_PREFIX}
        ?s p:P2048/psv:P2048 ?n . ?n wikibase:quantityAmount ?amount ; wikibase:quantityUnit ?unit . }}
        ORDER BY ?s ?cricinfo_id ?amount ?unit""",
    "birthplace": f"""SELECT ?s ?cricinfo_id ?place ?place_label ?place_country_label WHERE {{ {_PREFIX} ?s wdt:P19 ?place .
        {_label('?place', 'place_label')}
        OPTIONAL {{ ?place wdt:P17 ?pc . ?pc rdfs:label ?place_country_label FILTER(lang(?place_country_label) = "en") }} }}
        ORDER BY ?s ?cricinfo_id ?place ?place_country_label""",
    "citizenship": f"""SELECT ?s ?cricinfo_id ?country ?country_label WHERE {{ {_PREFIX} ?s wdt:P27 ?country .
        {_label('?country', 'country_label')} }} ORDER BY ?s ?cricinfo_id ?country""",
    "handedness": f"""SELECT ?s ?cricinfo_id ?prop ?value ?value_label WHERE {{ {_PREFIX}
        {{ ?s wdt:P552 ?value . BIND("P552" AS ?prop) }} UNION {{ ?s wdt:P741 ?value . BIND("P741" AS ?prop) }}
        {_label('?value', 'value_label')} }} ORDER BY ?s ?cricinfo_id ?prop ?value""",
    "position": f"""SELECT ?s ?cricinfo_id ?value ?value_label WHERE {{ {_PREFIX} ?s wdt:P413 ?value .
        {_label('?value', 'value_label')} }} ORDER BY ?s ?cricinfo_id ?value""",
}


def _run_query(cfg: dict, query: str, limit: int, offset: int) -> bytes:
    paged = f"{query} LIMIT {limit} OFFSET {offset}"
    headers = {"User-Agent": cfg["user_agent"], "Accept": "application/sparql-results+json"}
    for attempt in range(6):
        try:
            r = requests.post(cfg.get("endpoint", ENDPOINT), data={"query": paged}, headers=headers,
                              timeout=cfg["timeout_seconds"])
            if r.status_code in (429, 502, 503, 504):
                wait = int(r.headers.get("Retry-After", 0)) or 15 * (attempt + 1)
                print(f"  HTTP {r.status_code}; waiting {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.content
        except (requests.Timeout, requests.ConnectionError):
            time.sleep(15 * (attempt + 1))
    raise SystemExit("wikidata: query kept failing; try again later (completed pages are kept)")


def ingest_wikidata_players() -> None:
    """Network. Writes bronze/wikidata_players/<date>-<sha8 of the queries>/, atomically per page, and locks it."""
    cfg = load_yaml("player_sources.yml")[SOURCE]
    plan = {"endpoint": cfg["endpoint"], "page_size": cfg["page_size"], "queries": QUERIES}
    snapshot_id = store.new_snapshot_id(plan) if str(cfg.get("snapshot_id", "auto")) == "auto" else str(cfg["snapshot_id"])
    if snapshot_id in bronze.list_snapshots(SOURCE):
        print(f"{SOURCE}/{snapshot_id} already sealed; nothing to do")
        return
    folder = snapshot_dir(SOURCE, snapshot_id)
    folder.mkdir(parents=True, exist_ok=True)
    for tmp in folder.glob("*.tmp"):
        tmp.unlink()
    store.write_json_atomic(folder / store.PLAN, plan)
    files = [{"name": store.PLAN, "stored_as": store.PLAN, "compressed": False}]
    page = cfg["page_size"]
    with tempfile.TemporaryDirectory() as tmp:
        for name, query in QUERIES.items():
            offset, n_page = 0, 0
            while True:
                stored = f"{name}.p{n_page:03d}.json.zst"
                raw = Path(tmp) / f"{name}.p{n_page:03d}.json"
                if (folder / stored).exists():  # resumed run: the page was written atomically, so it is complete
                    raw.write_bytes(bronze.read_zst(folder / stored))
                else:
                    raw.write_bytes(_run_query(cfg, query, page, offset))
                    part = folder / (stored + ".tmp")
                    bronze.compress_to_zst(raw, part)
                    part.replace(folder / stored)
                    time.sleep(cfg["pause_seconds"])
                rows = len(json.loads(raw.read_bytes())["results"]["bindings"])
                files.append({"name": raw.name, "query": f"{query} LIMIT {page} OFFSET {offset}", "stored_as": stored,
                              "compressed": True, "rows": rows, "raw_bytes": raw.stat().st_size,
                              "raw_sha256": bronze.sha256(raw)})
                print(f"  {name} page {n_page}: {rows} rows")
                if rows < page:
                    break
                offset += page
                n_page += 1
    bronze.write_manifest(SOURCE, snapshot_id, {"endpoint": cfg["endpoint"], "licence": cfg["licence"],
                                                "attribution": cfg["attribution"]}, files)
    problems = store.strict_problems(SOURCE, snapshot_id)
    if problems:
        raise SystemExit("\n".join(problems))
    bronze.lock_source(SOURCE, snapshot_id)
    print(f"sealed and locked {SOURCE} -> {snapshot_id}")


# ---------------------------------------------------------------- build (offline)

def _read_bronze(folder: Path) -> dict[str, list[dict]]:
    """Decode every manifest-listed page into plain dict rows, in stable file order."""
    manifest = json.loads((folder / "_snapshot.json").read_text())
    out: dict[str, list[dict]] = {name: [] for name in QUERIES}
    for entry in sorted(manifest["files"], key=lambda e: e["stored_as"]):
        if not entry["stored_as"].endswith(".json.zst"):
            continue
        kind = entry["stored_as"].split(".")[0]
        payload = json.loads(bronze.read_zst(folder / entry["stored_as"], entry))
        if "rows" in entry and len(payload["results"]["bindings"]) != entry["rows"]:
            raise SystemExit(f"{entry['stored_as']}: {len(payload['results']['bindings'])} rows parsed, "
                             f"manifest says {entry['rows']}")
        for b in payload["results"]["bindings"]:
            out[kind].append({k: v["value"] for k, v in b.items()})
    return out


def _register(con, name: str, rows: list[dict], columns: list[str]) -> None:
    def clean(v):
        return v.strip() or None if isinstance(v, str) else v
    table = pa.table({c: pa.array([clean(r.get(c)) for r in rows], type=pa.string()) for c in columns})
    con.register(name, table)


def _qid(expr: str) -> str:
    return f"regexp_extract({expr}, '(Q[0-9]+)$', 1)"


def _ensure_views(con, silver: Path) -> None:
    have = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    for t in ("people", "match_players", "matches"):
        if t not in have:
            con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{silver / (t + '.parquet')}')")
    if "stg_ari_players" not in have:
        con.execute(f"CREATE VIEW stg_ari_players AS SELECT * FROM read_parquet('{STAGING / 'stg_ari_players.parquet'}')")


# Wikidata unit QID -> centimetres. Anything else is dropped, never guessed.
UNIT_TO_CM = {"Q174728": 1.0, "Q11573": 100.0, "Q3710": 30.48, "Q218593": 2.54, "Q200323": 10.0}
# Wikidata citizenship labels that are not spelled like the Cricsheet team of that nation.
CITIZENSHIP_TO_TEAMS = {
    "United Kingdom": ["England", "Scotland"], "Kingdom of the Netherlands": ["Netherlands"],
    "United States of America": ["United States of America"], "People's Republic of China": ["China"],
    "Czech Republic": ["Czech Republic"], "Eswatini": ["Eswatini", "Swaziland"], "Republic of Ireland": ["Ireland"],
    "Jamaica": ["West Indies"], "Barbados": ["West Indies", "Barbados"], "Trinidad and Tobago": ["West Indies"],
    "Guyana": ["West Indies"], "Antigua and Barbuda": ["West Indies"], "Saint Lucia": ["West Indies"],
    "Saint Vincent and the Grenadines": ["West Indies"], "Grenada": ["West Indies"], "Dominica": ["West Indies"],
    "Saint Kitts and Nevis": ["West Indies"],
}


def build_player_context(con, silver: Path = SILVER, bronze_folder: Path | None = None) -> None:
    """Offline, deterministic. Writes players.parquet and cricinfo_players.parquet and creates views of both."""
    if bronze_folder is None:
        sid = bronze.read_lock().get(SOURCE)
        if sid is None:
            raise SystemExit(f"no locked {SOURCE} snapshot; run ingest_wikidata_players() first")
        problems = store.strict_problems(SOURCE, sid)
        if problems:
            raise SystemExit("bronze does not verify:\n" + "\n".join(problems))
        bronze_folder = snapshot_dir(SOURCE, sid)
    _ensure_views(con, silver)

    data = _read_bronze(bronze_folder)
    _register(con, "wd_base", data["base"], ["s", "cricinfo_id", "label"])
    _register(con, "wd_dob", data["dob"], ["s", "cricinfo_id", "dob", "precision", "rank"])
    _register(con, "wd_sex", data["sex"], ["s", "sex_label"])
    _register(con, "wd_height", data["height"], ["s", "amount", "unit"])
    _register(con, "wd_birthplace", data["birthplace"], ["s", "place_label", "place_country_label"])
    _register(con, "wd_citizenship", data["citizenship"], ["s", "country", "country_label"])
    _register(con, "wd_hand", data["handedness"], ["s", "prop", "value", "value_label"])
    _register(con, "wd_position", data["position"], ["s", "value", "value_label"])
    unit_cases = " ".join(f"WHEN '{k}' THEN {v}" for k, v in UNIT_TO_CM.items())
    composite = ", ".join(f"'{t}'" for t in COMPOSITE_TEAMS)
    def sql_str(text: str) -> str:
        return "'" + text.replace("'", "''") + "'"
    cit_map = " ".join(f"WHEN {sql_str(k)} THEN [{', '.join(sql_str(t) for t in v)}]" for k, v in CITIZENSHIP_TO_TEAMS.items())

    # --- Wikidata, one row per QID. Several values for a scalar fact -> NULL (and a count), never "the first".
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE wd_person AS
        WITH b AS (
            SELECT {_qid('s')} AS qid, CASE WHEN count(DISTINCT label) = 1 THEN min(label) END AS wd_label,
                   list_sort(list(DISTINCT CAST(cricinfo_id AS VARCHAR))) AS cricinfo_ids
            FROM wd_base GROUP BY 1),
        d0 AS (  -- day-precision, AD, not deprecated (rank is only in newer snapshots)
            SELECT {_qid('s')} AS qid, CAST(substr(dob, 1, 10) AS DATE) AS dob, rank
            FROM wd_dob WHERE CAST(precision AS INT) >= 11 AND dob NOT LIKE '-%'
              AND coalesce(rank, '') NOT LIKE '%DeprecatedRank'),
        d AS (
            SELECT qid, count(DISTINCT dob) AS n_dob, list_sort(list(DISTINCT dob)) AS dob_candidates,
                   CASE WHEN count(DISTINCT dob) = 1 THEN min(dob)
                        WHEN count(DISTINCT dob) FILTER (rank LIKE '%PreferredRank') = 1
                        THEN min(dob) FILTER (rank LIKE '%PreferredRank') END AS dob
            FROM d0 GROUP BY 1),
        dl AS (SELECT {_qid('s')} AS qid, count(*) AS n_low FROM wd_dob WHERE CAST(precision AS INT) < 11 GROUP BY 1),
        x AS (SELECT {_qid('s')} AS qid, CASE WHEN count(DISTINCT sex_label) = 1 THEN min(sex_label) END AS sex
              FROM wd_sex GROUP BY 1),
        h AS (  -- normalise to cm, plausible adult human range only (outside = unit/entry error, dropped)
            SELECT qid, round(median(cm), 1) AS height_cm FROM (
                SELECT {_qid('s')} AS qid, CAST(amount AS DOUBLE) * (CASE {_qid('unit')} {unit_cases} END) AS cm FROM wd_height)
            WHERE cm BETWEEN 120 AND 240 GROUP BY 1),
        p AS (SELECT {_qid('s')} AS qid,
                     CASE WHEN count(DISTINCT place_label) = 1 THEN min(place_label) END AS birthplace,
                     CASE WHEN count(DISTINCT place_country_label) = 1 THEN min(place_country_label) END AS birthplace_country
              FROM wd_birthplace GROUP BY 1),
        c AS (SELECT {_qid('s')} AS qid, list_sort(list(DISTINCT country_label) FILTER (country_label IS NOT NULL)) AS citizenships
              FROM wd_citizenship GROUP BY 1),
        hd AS (  -- P552 "handedness" and P741 "playing hand" both used; reduce to right/left/ambidextrous
            SELECT {_qid('s')} AS qid, CASE WHEN count(DISTINCT hand) = 1 THEN min(hand) END AS handedness FROM (
                SELECT s, CASE WHEN lower(value_label) LIKE 'right%' THEN 'right'
                               WHEN lower(value_label) LIKE 'left%' THEN 'left'
                               WHEN lower(value_label) LIKE 'ambi%' THEN 'ambidextrous' END AS hand
                FROM wd_hand WHERE prop IN ('P552', 'P741')) WHERE hand IS NOT NULL GROUP BY 1),
        pos AS (SELECT {_qid('s')} AS qid, string_agg(DISTINCT value_label, '|' ORDER BY value_label) AS wd_positions
                FROM wd_position GROUP BY 1)
        SELECT b.qid, b.wd_label, b.cricinfo_ids, d.dob, coalesce(d.n_dob, 0) AS n_dob, d.dob_candidates, coalesce(dl.n_low, 0) AS n_low_precision,
               x.sex, h.height_cm, p.birthplace, p.birthplace_country, c.citizenships, hd.handedness, pos.wd_positions
        FROM b LEFT JOIN d USING (qid) LEFT JOIN dl USING (qid) LEFT JOIN x USING (qid) LEFT JOIN h USING (qid)
        LEFT JOIN p USING (qid) LEFT JOIN c USING (qid) LEFT JOIN hd USING (qid) LEFT JOIN pos USING (qid)""")

    # --- Cricsheet people <-> Wikidata via the ESPNcricinfo id (any of key_cricinfo, _2, _3).
    con.execute("""
        CREATE OR REPLACE TEMP TABLE cs_cricinfo AS
        SELECT DISTINCT identifier, id AS cricinfo_id, rank FROM (
            SELECT identifier, nullif(trim(CAST(key_cricinfo AS VARCHAR)), '') AS id, 1 AS rank FROM people UNION ALL
            SELECT identifier, nullif(trim(CAST(key_cricinfo_2 AS VARCHAR)), ''), 2 FROM people UNION ALL
            SELECT identifier, nullif(trim(CAST(key_cricinfo_3 AS VARCHAR)), ''), 3 FROM people) WHERE id IS NOT NULL""")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE wd_link AS
        WITH cand AS (
            SELECT DISTINCT l.identifier, w.qid, w.dob, w.n_dob
            FROM cs_cricinfo l JOIN (SELECT qid, dob, n_dob, unnest(cricinfo_ids) AS cricinfo_id FROM wd_person) w USING (cricinfo_id))
        SELECT identifier, count(*) AS n_candidates,
               CASE WHEN count(*) = 1 THEN min(qid)
                    -- several items for one person: usable only if they agree on the date of birth
                    WHEN count(DISTINCT dob) = 1 AND count(dob) = count(*) THEN min(qid) END AS qid
        FROM cand GROUP BY identifier""")

    # --- Cricsheet facts over all matches (scope-free).
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE pm AS
        SELECT DISTINCT mp.player_id AS identifier, mp.match_id, mp.team, m.start_date, m.match_type, m.team_type,
               m.gender, m.event_name
        FROM match_players mp JOIN matches m USING (match_id) WHERE mp.player_id IS NOT NULL""")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE career AS
        WITH k AS (SELECT *, CASE WHEN match_type = 'Test' THEN 'test' WHEN match_type = 'ODI' THEN 'odi'
                                  WHEN match_type = 'T20' AND team_type = 'international' THEN 't20i'
                                  WHEN event_name = 'Indian Premier League' THEN 'ipl' ELSE 'other' END AS cat FROM pm)
        SELECT identifier,
            count(DISTINCT match_id) AS matches_total,
            min(start_date) AS first_match_date, max(start_date) AS last_match_date,
            CASE WHEN count(DISTINCT gender) = 1 THEN min(gender) END AS cricsheet_gender,
            count(DISTINCT match_id) FILTER (cat = 'test') AS matches_test,
            min(start_date) FILTER (cat = 'test') AS first_test_date, max(start_date) FILTER (cat = 'test') AS last_test_date,
            count(DISTINCT match_id) FILTER (cat = 'odi') AS matches_odi,
            min(start_date) FILTER (cat = 'odi') AS first_odi_date, max(start_date) FILTER (cat = 'odi') AS last_odi_date,
            count(DISTINCT match_id) FILTER (cat = 't20i') AS matches_t20i,
            min(start_date) FILTER (cat = 't20i') AS first_t20i_date, max(start_date) FILTER (cat = 't20i') AS last_t20i_date,
            count(DISTINCT match_id) FILTER (cat = 'ipl') AS matches_ipl,
            min(start_date) FILTER (cat = 'ipl') AS first_ipl_date, max(start_date) FILTER (cat = 'ipl') AS last_ipl_date
        FROM k GROUP BY identifier""")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE nation AS
        WITH t AS (
            SELECT identifier, team, count(DISTINCT match_id) AS n, max(start_date) AS last_date
            FROM pm WHERE team_type = 'international' AND team NOT IN ({composite}) GROUP BY ALL)
        SELECT identifier, list_sort(list(team)) AS national_teams,
               arg_max(team, (last_date, n, team)) AS latest_team, count(*) AS n_teams
        FROM t GROUP BY identifier""")

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE player_base AS
        WITH j AS (
            SELECT c.*, nullif(trim(CAST(p.name AS VARCHAR)), '') AS name, l.n_candidates, l.qid, w.*  EXCLUDE (qid),
                   n.national_teams, n.latest_team, n.n_teams,
                   CAST(datediff('day', w.dob, c.first_match_date) / 365.25 AS DOUBLE) AS age_first,
                   CAST(datediff('day', w.dob, c.last_match_date) / 365.25 AS DOUBLE) AS age_last
            FROM career c
            JOIN people p ON p.identifier = c.identifier
            LEFT JOIN wd_link l ON l.identifier = c.identifier
            LEFT JOIN wd_person w ON w.qid = l.qid
            LEFT JOIN nation n ON n.identifier = c.identifier),
        cit AS (  -- citizenships mapped to the Cricsheet team names they could play for
            SELECT identifier, list_distinct(flatten(list(CASE x {cit_map} ELSE [x] END))) AS citizen_teams
            FROM (SELECT identifier, unnest(citizenships) AS x FROM j) GROUP BY identifier)
        SELECT j.*, cit.citizen_teams,
            CASE
                WHEN j.n_candidates > 1 AND j.qid IS NULL THEN 'ambiguous'
                WHEN j.n_candidates > 1 THEN 'several_items_same_dob' END AS wikidata_flag,
            CASE
                WHEN j.qid IS NULL THEN NULL
                WHEN j.n_dob > 1 AND j.dob IS NULL THEN 'conflict'
                WHEN j.dob = DATE '1970-01-01' THEN 'placeholder'
                WHEN j.age_first < {MIN_DEBUT_AGE} THEN 'implausible_age_at_first_match'
                WHEN j.age_first > {MAX_DEBUT_AGE} THEN 'implausible_age_at_first_match'
                WHEN j.age_last > {MAX_LAST_MATCH_AGE} THEN 'implausible_age_at_last_match'
                WHEN j.dob IS NULL AND j.n_low_precision > 0 THEN 'low_precision_only'
                WHEN month(j.dob) = 1 AND day(j.dob) = 1 THEN 'jan1_low_precision'
                WHEN j.n_dob > 1 THEN 'conflict_resolved_by_preferred_rank' END AS dob_flag
        FROM j LEFT JOIN cit USING (identifier)""")

    withheld = "[" + ", ".join(f"'{n}'" for n in WITHHELD_NATIONS) + "]"
    public_sql = f"""
        SELECT b.identifier AS cricsheet_id, b.name,
               (SELECT min(x.cricinfo_id) FROM cs_cricinfo x WHERE x.identifier = b.identifier
                AND x.rank = (SELECT min(y.rank) FROM cs_cricinfo y WHERE y.identifier = b.identifier)) AS cricinfo_id,
               b.qid AS wikidata_qid, b.n_candidates AS wikidata_candidates, b.wikidata_flag,
               -- not 'implausible_age_at_last_match': that gate reads the player's later matches (gold applies it
               -- as of each match instead), so it must not decide the age at an earlier one
               CASE WHEN b.dob_flag IN ('conflict', 'placeholder', 'implausible_age_at_first_match') THEN NULL
                    ELSE b.dob END AS dob,
               b.dob_flag, b.n_dob AS wikidata_dob_values,
               b.birthplace, b.birthplace_country, b.sex, b.cricsheet_gender, b.handedness,
               CASE WHEN b.n_teams = 1 THEN b.latest_team
                    WHEN b.n_teams > 1 THEN b.latest_team
                    WHEN len(b.citizen_teams) = 1 AND b.citizen_teams[1] IN (SELECT DISTINCT team FROM pm
                         WHERE team_type = 'international' UNION ALL SELECT unnest({withheld})) THEN b.citizen_teams[1] END AS country,
               CASE WHEN b.n_teams = 1 THEN 'cricsheet_national_team'
                    WHEN b.n_teams > 1 THEN 'cricsheet_latest_of_several_national_teams'
                    WHEN len(b.citizen_teams) = 1 AND b.citizen_teams[1] IN (SELECT DISTINCT team FROM pm
                         WHERE team_type = 'international' UNION ALL SELECT unnest({withheld})) THEN 'wikidata_citizenship_no_internationals'
                    WHEN b.citizenships IS NOT NULL THEN 'unresolved_citizenship_no_internationals'
                    ELSE 'unknown' END AS country_source,
               b.national_teams,
               CASE WHEN b.n_teams >= 1 AND b.citizen_teams IS NOT NULL
                    THEN len(list_intersect(b.citizen_teams, b.national_teams)) > 0 END AS citizenship_consistent,
               -- The country stays as above; this flags where Wikidata and Cricsheet facts disagree.
               -- Only Wikidata (CC0) and Cricsheet facts are compared, never Ari or Cricinfo values.
               CASE WHEN b.n_teams >= 1 AND b.citizen_teams IS NOT NULL
                         AND len(list_intersect(b.citizen_teams, b.national_teams)) = 0 THEN 'citizenship_not_national_team'
                    -- no Cricsheet international can show the side of a player born in a withheld nation
                    WHEN b.n_teams IS NULL AND len(b.citizen_teams) >= 1 AND b.birthplace_country IN (SELECT unnest({withheld}))
                         AND NOT list_contains(b.citizen_teams, b.birthplace_country) THEN 'born_in_withheld_nation'
                    END AS country_conflict,
               b.citizenships, b.height_cm AS wikidata_height_cm, b.dob_candidates AS wikidata_dob_quarantine
        FROM player_base b"""
    # Career spans and counts are NOT stored here: they summarise the whole snapshot, so they would leak the
    # future into any model. Gold has report_player_careers (snapshot summary) and fact_player_match_form (as-of).
    cricinfo_sql = """
        SELECT b.identifier AS cricsheet_id,
               a.country AS ari_country, a.batting_style AS ari_batting_style, a.bowling_style AS ari_bowling_style,
               a.playing_role AS ari_playing_role, a.fielding_style AS ari_fielding_style,
               a.full_name AS ari_full_name, a.cricinfo_id AS ari_cricinfo_id
        FROM player_base b JOIN stg_ari_players a ON a.player_id = b.identifier"""

    for table, sql, dest in (("players", public_sql, "players.parquet"),
                             ("cricinfo_players", cricinfo_sql, "cricinfo_players.parquet")):
        path = silver / dest
        con.execute(f"COPY (SELECT * FROM ({sql}) ORDER BY cricsheet_id) TO '{path}' (FORMAT parquet, COMPRESSION zstd)")
        con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{path}')")
    n, d = con.execute("SELECT count(*), count(DISTINCT cricsheet_id) FROM players").fetchone()
    if n != d:
        raise SystemExit(f"players has {n - d} duplicate cricsheet_id rows")
    for t in ("wd_base", "wd_dob", "wd_sex", "wd_height", "wd_birthplace", "wd_citizenship", "wd_hand", "wd_position"):
        con.unregister(t)
