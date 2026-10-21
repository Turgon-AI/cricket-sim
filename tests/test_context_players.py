import json
from pathlib import Path

import duckdb
import pytest
import zstandard

from cricket_sim import context_players
from cricket_sim.context_players import QUERIES, build_player_context

W = "http://www.wikidata.org/entity/"


def _bindings(rows):
    return {"head": {}, "results": {"bindings": [{k: {"type": "literal", "value": v} for k, v in r.items()} for r in rows]}}


def write_bronze(folder, pages):
    folder.mkdir(parents=True)
    files = []
    for kind in QUERIES:
        name = f"{kind}.p000.json.zst"
        (folder / name).write_bytes(zstandard.ZstdCompressor().compress(json.dumps(_bindings(pages.get(kind, []))).encode()))
        files.append({"stored_as": name})
    (folder / "_snapshot.json").write_text(json.dumps({"files": files}))


def E(q, cid):
    return {"s": W + q, "cricinfo_id": cid}


def DOB(q, cid, d, precision="11", rank=None):
    row = {**E(q, cid), "dob": f"{d}T00:00:00Z", "precision": precision}
    if rank:
        row["rank"] = "http://wikiba.se/ontology#" + rank
    return row


@pytest.fixture
def built(tmp_path):
    pages = {
        "base": [{**E("Q1", "100"), "label": "Alpha One"}, {**E("Q2", "200"), "label": "Beta"},
                 {**E("Q3", "300"), "label": "Dup A"}, {**E("Q4", "300"), "label": "Dup B"},
                 {**E("Q10", "1000"), "label": "Gilly"}, {**E("Q11", "1100"), "label": "Old Latham"},
                 {**E("Q12", "1200"), "label": "Stokesy"}, {**E("Q13", "1300"), "label": "Gul"},
                 {**E("Q14", "1400"), "label": "Epoch"}, {**E("Q15", "1500"), "label": "Pref"},
                 {**E("Q16", "1600"), "label": "Afghan"}, {**E("Q17", "1700"), "label": "Gulbadin"}],
        "dob": [DOB("Q1", "100", "1990-05-17"), DOB("Q2", "200", "1985-01-01", precision="9"),
                DOB("Q3", "300", "1980-01-01"), DOB("Q4", "300", "1981-02-02"),
                # Gilchrist: Wikidata holds both 1969-11-14 and 1971-11-14 -> no date, flagged (never "the earliest")
                DOB("Q10", "1000", "1969-11-14"), DOB("Q10", "1000", "1971-11-14"),
                DOB("Q11", "1100", "1923-04-30"),            # Latham: 88 at a 2012 debut -> rejected
                DOB("Q12", "1200", "1991-06-04"), DOB("Q13", "1300", "1984-11-14"),
                DOB("Q14", "1400", "1970-01-01"),            # placeholder
                DOB("Q15", "1500", "1990-02-02", rank="PreferredRank"), DOB("Q15", "1500", "1909-02-02", rank="DeprecatedRank"),
                DOB("Q16", "1600", "1998-09-20")],
        "sex": [{"s": W + "Q1", "sex_label": "male"}, {"s": W + "Q2", "sex_label": " "}],
        "height": [{"s": W + "Q1", "amount": "6", "unit": W + "Q3710"}, {"s": W + "Q2", "amount": "1.2", "unit": W + "Q174728"}],
        "birthplace": [{"s": W + "Q1", "place_label": "Leeds", "place_country_label": "United Kingdom"},
                       {"s": W + "Q17", "place_label": "Pol-e Alam", "place_country_label": "Afghanistan"}],
        "citizenship": [{"s": W + "Q1", "country": W + "Q145", "country_label": "United Kingdom"},
                        {"s": W + "Q12", "country": W + "Q664", "country_label": "New Zealand"},
                        {"s": W + "Q12", "country": W + "Q145", "country_label": "United Kingdom"},
                        {"s": W + "Q13", "country": W + "Q408", "country_label": "Australia"},
                        {"s": W + "Q16", "country": W + "Q889", "country_label": "Afghanistan"},
                        {"s": W + "Q17", "country": W + "Q843", "country_label": "Pakistan"}],
        "handedness": [{"s": W + "Q1", "prop": "P552", "value": W + "Q789447", "value_label": "left-handedness"},
                       {"s": W + "Q2", "prop": "P552", "value": W + "Q789447", "value_label": "left-handedness"},
                       {"s": W + "Q2", "prop": "P741", "value": W + "Q3039938", "value_label": "right-handedness"}],
        "position": [],
    }
    bronze_dir = tmp_path / "bronze"
    write_bronze(bronze_dir, pages)
    silver = tmp_path / "silver"
    silver.mkdir()
    con = duckdb.connect()
    con.execute("""CREATE TABLE people AS SELECT * FROM (VALUES
        ('p1','Alpha One','100',NULL,NULL), ('p2','Beta','200',NULL,NULL), ('p3','Gamma','300',NULL,NULL),
        ('p4','Delta','',NULL,NULL), ('p5','Eps','999','100',NULL), ('p6','Old Timer',NULL,NULL,NULL),
        ('p7','Woman','777',NULL,NULL), ('p10','AC Gilchrist','1000',NULL,NULL), ('p11','TWM Latham','1100',NULL,NULL),
        ('p12','BA Stokes','1200',NULL,NULL), ('p13','Umar Gul','1300',NULL,NULL), ('p14','MF Manjate','1400',NULL,NULL),
        ('p15','Pref','1500',NULL,NULL), ('p16','Rashid Khan','1600',NULL,NULL), ('p17','Gulbadin Naib','1700',NULL,NULL)
        ) t(identifier, name, key_cricinfo, key_cricinfo_2, key_cricinfo_3)""")
    con.execute("""CREATE TABLE matches AS SELECT * FROM (VALUES
        ('m1','Test','international','male',NULL,DATE '2010-01-01'), ('m2','Test','international','male',NULL,DATE '2012-02-02'),
        ('m3','T20','club','male','Indian Premier League',DATE '2015-04-01'),
        ('m4','Test','international','male',NULL,DATE '1999-01-01'),
        ('m5','Test','international','female',NULL,DATE '2010-01-01'),
        ('m6','T20','international','male',NULL,DATE '2007-09-13'))
        t(match_id, match_type, team_type, gender, event_name, start_date)""")
    con.execute("""CREATE TABLE match_players AS SELECT * FROM (VALUES
        ('m1','England','x','p1'),('m2','England','x','p1'),('m3','Mumbai','x','p1'),('m1','England','x','p2'),
        ('m1','Australia','x','p3'),('m1','Australia','x','p4'),('m2','Australia','x','p5'),('m4','England','x','p6'),
        ('m5','England','x','p7'),('m1','Australia','x','p10'),('m2','New Zealand','x','p11'),('m2','England','x','p12'),
        ('m6','Pakistan','x','p13'),('m2','Asia XI','x','p13'),('m2','Mozambique','x','p14'),('m2','India','x','p15'),
        ('m3','Gujarat','x','p16'), ('m3','Delhi Capitals','x','p17'))
        t(match_id, team, player, player_id)""")
    con.execute("""CREATE TABLE stg_ari_players AS SELECT 'p1' AS player_id, '100' AS cricinfo_id, 'England' AS country,
        'left-hand bat' AS batting_style, 'right-arm offbreak' AS bowling_style, 'Batter' AS playing_role,
        NULL AS fielding_style, 'Alpha One Full' AS full_name""")
    build_player_context(con, silver=silver, bronze_folder=bronze_dir)
    return con, silver


def row(con, table, pid):
    cur = con.execute(f"SELECT * FROM {table} WHERE cricsheet_id = ?", [pid])
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, cur.fetchone()))


def test_scope_free_every_player_in_any_match(built):
    con, _ = built
    ids = {r[0] for r in con.execute("SELECT cricsheet_id FROM players").fetchall()}
    assert {"p6", "p7"} <= ids  # pre-2001 and women's players are in silver; gold applies the scope
    assert row(con, "players", "p7")["cricsheet_gender"] == "female"


def test_conflicting_dobs_are_null_and_flagged_not_earliest(built):
    con, _ = built
    g = row(con, "players", "p10")
    assert g["dob"] is None and g["dob_flag"] == "conflict" and g["wikidata_dob_values"] == 2
    assert [str(d) for d in row(con, "players", "p10")["wikidata_dob_quarantine"]] == ["1969-11-14", "1971-11-14"]


def test_age_gates_and_placeholders(built):
    con, _ = built
    lat = row(con, "players", "p11")
    assert lat["dob"] is None and lat["dob_flag"] == "implausible_age_at_first_match"
    assert row(con, "players", "p14")["dob_flag"] == "placeholder"
    assert row(con, "players", "p14")["dob"] is None
    assert str(row(con, "players", "p12")["dob"]) == "1991-06-04"


def test_preferred_rank_resolves_and_deprecated_is_ignored(built):
    con, _ = built
    p = row(con, "players", "p15")
    assert str(p["dob"]) == "1990-02-02" and p["dob_flag"] is None


def test_nationality_follows_cricsheet_teams(built):
    con, _ = built
    stokes = row(con, "players", "p12")  # citizenships New Zealand + United Kingdom, plays for England
    assert stokes["country"] == "England" and stokes["citizenship_consistent"] is True
    gul = row(con, "players", "p13")     # Wikidata citizenship Australia; Asia XI ignored
    assert gul["country"] == "Pakistan" and gul["national_teams"] == ["Pakistan"] and gul["citizenship_consistent"] is False
    rk = row(con, "players", "p16")      # no Cricsheet internationals (Afghanistan withheld)
    assert rk["country"] == "Afghanistan" and rk["country_source"] == "wikidata_citizenship_no_internationals"
    assert row(con, "players", "p5")["country"] == "Australia"


def test_t3_10_country_conflict_flag(built):
    """Gulbadin Naib: born in Afghanistan, Wikidata citizenship Pakistan, no Cricsheet internationals (Afghanistan
    is withheld). The country stays the Wikidata one (decision); the conflict is flagged. Only Wikidata and
    Cricsheet facts are compared, never Ari or Cricinfo."""
    con, _ = built
    gb = row(con, "players", "p17")
    assert gb["country"] == "Pakistan" and gb["country_source"] == "wikidata_citizenship_no_internationals"
    assert gb["country_conflict"] == "born_in_withheld_nation"
    assert row(con, "players", "p13")["country_conflict"] == "citizenship_not_national_team"  # Australia v Pakistan
    for pid in ("p12", "p16", "p1", "p6"):  # consistent, or nothing to compare
        assert row(con, "players", pid)["country_conflict"] is None, pid


def test_unknowns_are_null_never_empty_strings(built):
    con, _ = built
    b = row(con, "players", "p2")
    assert b["dob"] is None and b["dob_flag"] == "low_precision_only"
    assert b["sex"] is None and b["handedness"] is None
    assert row(con, "players", "p4")["cricinfo_id"] is None
    text_cols = [r[0] for r in con.execute("DESCRIBE players").fetchall() if r[1] == "VARCHAR"]
    for c in text_cols:
        assert con.execute(f"SELECT count(*) FROM players WHERE trim({c}) = ''").fetchone()[0] == 0, c


def test_ambiguous_wikidata_link_is_not_guessed(built):
    con, _ = built
    c = row(con, "players", "p3")  # two items for one Cricinfo id with different dates of birth
    assert c["wikidata_candidates"] == 2 and c["wikidata_qid"] is None and c["wikidata_flag"] == "ambiguous"


def test_public_table_has_no_cricinfo_columns_and_no_career_totals(built):
    con, _ = built
    pub = {r[0] for r in con.execute("DESCRIBE players").fetchall()}
    assert not any(c.startswith("ari_") or "style" in c for c in pub)
    assert not any(c.startswith(("matches_", "first_", "last_")) for c in pub)  # whole-snapshot totals leak the future
    assert row(con, "players", "p1")["wikidata_height_cm"] == pytest.approx(182.9, abs=0.1)
    restricted = {r[0] for r in con.execute("DESCRIBE cricinfo_players").fetchall()}
    assert restricted == {"cricsheet_id", "ari_country", "ari_batting_style", "ari_bowling_style", "ari_playing_role",
                          "ari_fielding_style", "ari_full_name", "ari_cricinfo_id"}


def test_one_row_per_player_deterministic_and_no_gold_import(built, tmp_path):
    con, silver = built
    first = (silver / "players.parquet").read_bytes()
    build_player_context(con, silver=silver, bronze_folder=tmp_path / "bronze")
    assert (silver / "players.parquet").read_bytes() == first
    src = Path(context_players.__file__).read_text()
    assert "gold" not in src.split('"""', 2)[2].split("def ")[0] and "_segment_case" not in src and "scope.yml\")" not in src
