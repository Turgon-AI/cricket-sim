"""Future-blindness: every column marked `pre_ball` must be computable at that moment of a live match.

Method (a truncation test). For each fixture match, cut the match off after ball k, as if it were being
played live: drop every later ball and innings, blank the result fields of the match (winner, margin,
method, ...) and the fields of the current innings that are only written once it ends (the final target,
declaration, end-of-innings penalties, absentees, miscounted overs). The cut is made in the raw Cricsheet
JSON, so the real parser runs on it too; then the pre-ball state is recomputed, and every column of the
full match is compared with the cut one for every ball up to k.

  pre_ball   columns must be identical at every cut point: every pre_ball column of fact_delivery (the
             parser's own columns, such as legal_balls_in_over_before, and the state columns) and of
             int_ball_state_pre, plus the model features that dataset.DERIVED computes from them.
  hindsight  columns may differ (that is what the label means), and the test checks that the label is
             earned: on the fixture, each hindsight column really does change somewhere.

The build's own check (ball_layers.check_no_leak) is complementary: it proves each ball's `_before` state
equals the previous state plus the previous ball, on the full data.

The model's own guards are tested here too: every column a model is fitted on must be known before the ball
(dataset.check_inputs_are_pre_ball), and walk_forward refuses any fit on rows of the season it scores.
"""

import copy
import inspect
import json
import re
import textwrap
import zipfile
from pathlib import Path

import duckdb
import pyarrow as pa
import pytest

from cricket_sim import ball_layers, context_players, contracts
from cricket_sim.cricsheet_parse import SCHEMAS, _parse_match
from cricket_sim.models import dataset, train

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tiny"
STEP = 15  # cut after every 15th delivery, plus the first and last ball of every innings


def _timing(table: str) -> dict:
    c = contracts.get(table)
    return {col["name"]: c.column_props(col).get("timingClass") for col in c.columns if not col.get("primaryKey")}


def timing() -> dict:
    return _timing("int_ball_state_pre")


def _fixture_zip() -> Path:
    lock = (FIXTURE / "snapshots.lock.yml").read_text()
    snapshot = re.search(r"^cricsheet: '?([^'\n]+)'?$", lock, re.M).group(1)
    return FIXTURE / "bronze" / "cricsheet" / snapshot / "all_json.zip"


def cut_doc(doc: dict, ci: int, cd: int) -> dict:
    """The match as it stood live after delivery cd of innings ci: later balls and innings gone, the result and
    the end-of-innings fields of innings ci blank."""
    d = copy.deepcopy(doc)
    d["info"]["outcome"] = {}
    d["info"].pop("player_of_match", None)
    innings = d.get("innings", [])[:ci]
    current = innings[ci - 1]
    for k in ("target", "declared", "absent_hurt", "miscounted_overs"):
        current.pop(k, None)
    current.get("penalty_runs", {}).pop("post", None)
    n, overs = 0, []
    for over in current.get("overs", []):
        keep = [dl for k, dl in enumerate(over["deliveries"], start=n + 1) if k <= cd]
        n += len(over["deliveries"])
        if keep:
            overs.append({**over, "deliveries": keep})
        if n >= cd:
            break
    current["overs"] = overs
    d["innings"] = innings
    return d


def _derived_from_fact_delivery() -> dict:
    """The DERIVED model features that read fact_delivery only (alias f), so they can be recomputed here."""
    return {name: sql for name, sql in dataset.DERIVED.items()
            if set(re.findall(r"\b([a-z]\w*)\.", sql)) == {"f"}}


def truncation_changes(parse=_parse_match, match_ids=None) -> dict:
    """Parse each fixture match whole and cut short at many balls; compare every timed column of fact_delivery
    and int_ball_state_pre (and the DERIVED features) on every ball up to each cut."""
    state_t, fd_t = timing(), _timing("fact_delivery")
    columns = sorted((set(state_t) | set(fd_t)) - {c for c, t in fd_t.items() if t not in ("pre_ball", "hindsight")})
    derived = _derived_from_fact_delivery()
    con = duckdb.connect()
    changed, compared, cuts = {}, 0, 0
    with zipfile.ZipFile(_fixture_zip()) as zf:
        ids = match_ids or sorted(n[:-5] for n in zf.namelist() if n.endswith(".json"))
        for mid in ids:
            doc = json.loads(zf.read(f"{mid}.json"))
            out = {t: [] for t in SCHEMAS}
            parse(mid, doc, out)
            flat = [(i, k) for i, inn in enumerate(doc.get("innings", []), start=1)
                    for k in range(1, sum(len(o["deliveries"]) for o in inn.get("overs", [])) + 1)]
            points = set(range(0, len(flat), STEP))
            for p in range(len(flat)):
                if p + 1 == len(flat) or flat[p + 1][0] != flat[p][0]:
                    points |= {p, p - flat[p][1] + 1}  # last and first ball of the innings
            for v, p in enumerate(sorted(points)):
                parse(f"{mid}#{v}", cut_doc(doc, *flat[p]), out)
                cuts += 1
            for t, rows in out.items():
                con.register(f"_{t}", pa.Table.from_pylist(rows, schema=SCHEMAS[t]))
                con.execute(f"CREATE OR REPLACE TABLE {t} AS SELECT * FROM _{t}")
                con.unregister(f"_{t}")
            con.execute(f"CREATE OR REPLACE TABLE int_innings_state AS {ball_layers.INNINGS_SQL}")
            con.execute(f"CREATE OR REPLACE TABLE st AS {ball_layers.STATE_PRE_SQL}")
            have = {r[0] for r in con.execute("DESCRIBE st").fetchall()}
            balls = {r[0] for r in con.execute("DESCRIBE balls").fetchall()}
            missing = [c for c in columns if c not in have and c not in balls]
            assert not missing, f"timed columns this test cannot recompute: {missing}"
            pick = ", ".join(f"s.{c}" if c in have else f"b.{c}" for c in columns)
            # the DERIVED SQL reads fact_delivery as alias f; here f is the recomputed state joined to the balls
            con.execute(f"""CREATE OR REPLACE TABLE x AS
                WITH f AS (SELECT s.match_id, s.innings, s.delivery_number, {pick}
                           FROM st s JOIN balls b USING (match_id, innings, delivery_number))
                SELECT f.*{"".join(f", {sql} AS {name}" for name, sql in derived.items())} FROM f""")
            cols = columns + list(derived)
            diffs = ", ".join(f"count(*) FILTER (a.{c} IS DISTINCT FROM v.{c})" for c in cols)
            examples = ", ".join(f"min(v.match_id || ' i' || v.innings || ' d' || v.delivery_number || ' full='"
                                 f" || coalesce(CAST(a.{c} AS VARCHAR), 'NULL') || ' cut='"
                                 f" || coalesce(CAST(v.{c} AS VARCHAR), 'NULL')) FILTER (a.{c} IS DISTINCT FROM v.{c})"
                                 for c in cols)
            row = con.execute(f"""
                WITH v AS (SELECT *, split_part(match_id, '#', 1) AS base FROM x WHERE match_id LIKE '%#%'),
                     a AS (SELECT * FROM x WHERE match_id NOT LIKE '%#%')
                SELECT count(*), {diffs}, {examples}
                FROM v JOIN a ON a.match_id = v.base AND a.innings = v.innings
                             AND a.delivery_number = v.delivery_number""").fetchone()
            compared += row[0]
            for c, n, ex in zip(cols, row[1:1 + len(cols)], row[1 + len(cols):]):
                if n:
                    changed.setdefault(c, ex)
            if mid == ids[0]:
                con.execute("CREATE OR REPLACE TABLE full_first AS SELECT * FROM x WHERE match_id NOT LIKE '%#%'")
    pre_ball = {c for c in columns if state_t.get(c, fd_t.get(c)) == "pre_ball"} | set(derived)
    return {"changed": changed, "compared": compared, "cuts": cuts, "pre_ball": pre_ball,
            "hindsight": {c for c in columns if state_t.get(c, fd_t.get(c)) == "hindsight"}, "columns": cols}


@pytest.fixture(scope="module")
def comparisons():
    return truncation_changes()


def test_pre_ball_columns_never_change_when_the_match_is_cut_short(comparisons):
    assert comparisons["cuts"] > 300 and comparisons["compared"] > 100_000
    # every pre_ball column of fact_delivery is covered, the parser's own columns included
    fd_pre = {c for c, t in _timing("fact_delivery").items() if t == "pre_ball"}
    assert fd_pre <= comparisons["pre_ball"] and {"legal_balls_in_over_before", "actual_delivery", "batter_id",
                                                   "run_rate", "required_rate"} <= comparisons["pre_ball"]
    leaked = {c: ex for c, ex in comparisons["changed"].items() if c in comparisons["pre_ball"]}
    assert not leaked, f"pre_ball columns that depend on later balls (match, ball, full, truncated): {leaked}"


def test_hindsight_labels_are_earned(comparisons):
    hindsight = comparisons["hindsight"]
    assert hindsight, "the contract should mark the hindsight columns"
    unearned = hindsight - set(comparisons["changed"])
    assert not unearned, f"marked hindsight but never changed on the fixture: {unearned}"


def test_the_truncation_parse_is_the_build(tiny_build):
    """The columns this test recomputes from the raw JSON are the ones the build published."""
    con = duckdb.connect()
    with zipfile.ZipFile(_fixture_zip()) as zf:
        out = {t: [] for t in SCHEMAS}
        for name in sorted(n for n in zf.namelist() if n.endswith(".json")):
            _parse_match(name[:-5], json.loads(zf.read(name)), out)
    for t, rows in out.items():
        con.register(f"_{t}", pa.Table.from_pylist(rows, schema=SCHEMAS[t]))
        con.execute(f"CREATE TABLE {t} AS SELECT * FROM _{t}")
    con.execute(f"CREATE TABLE int_innings_state AS {ball_layers.INNINGS_SQL}")
    con.execute(f"CREATE TABLE st AS {ball_layers.STATE_PRE_SQL}")
    gold, keys = tiny_build / "gold", ["match_id", "innings", "delivery_number"]
    state_cols = ", ".join(keys + list(timing()))
    fd = [c for c, t in _timing("fact_delivery").items() if t in ("pre_ball", "hindsight")]
    fd_cols = ", ".join(keys + fd)
    pick = ", ".join([f"s.{k}" for k in keys] + [f"s.{c}" if c in timing() else f"b.{c}" for c in fd])
    for mine, published in (
            (f"SELECT {state_cols} FROM st", f"read_parquet('{gold}/int_ball_state_pre.parquet')"),
            (f"SELECT {pick} FROM st s JOIN balls b USING (match_id, innings, delivery_number)",
             f"read_parquet('{gold}/fact_delivery.parquet')")):
        cols = state_cols if "int_ball" in published else fd_cols
        a = f"SELECT {cols} FROM ({mine}) WHERE match_id IN (SELECT match_id FROM {published})"
        b = f"SELECT {cols} FROM {published}"
        assert con.execute(f"SELECT count(*) FROM {published}").fetchone()[0] > 5000
        assert con.execute(f"SELECT count(*) FROM (({a}) EXCEPT ({b}))").fetchone()[0] == 0, published
        assert con.execute(f"SELECT count(*) FROM (({b}) EXCEPT ({a}))").fetchone()[0] == 0, published


def _parse_reading_the_rest_of_the_over(match_id: str, doc: dict, out: dict) -> None:
    """A deliberately leaky parser: legal_balls_in_over_before := legal balls still to come."""
    start = len(out["balls"])
    _parse_match(match_id, doc, out)
    mine = out["balls"][start:]
    legal_in_over = {}
    for b in mine:
        legal_in_over[(b["innings"], b["over"])] = legal_in_over.get((b["innings"], b["over"]), 0) + b["is_legal"]
    for b in mine:
        b["legal_balls_in_over_before"] = legal_in_over[(b["innings"], b["over"])] - b["legal_balls_in_over_before"]


def test_truncation_check_catches_a_parser_column_that_reads_ahead():
    found = truncation_changes(parse=_parse_reading_the_rest_of_the_over, match_ids=["433606"])
    assert "legal_balls_in_over_before" in found["changed"], found["changed"]


def test_fact_delivery_uses_the_same_timing_as_the_state_it_copies():
    state = timing()
    c = contracts.get("fact_delivery")
    for col in c.columns:
        if col["name"] in state:
            assert c.column_props(col)["timingClass"] == state[col["name"]], col["name"]


# ---------------------------------------------------------------- model inputs

LABELS = ["batting_won", "outcome_class", "runs_total", "dismissal_count", "is_legal"]
HINDSIGHT = ["runs_required_before", "balls_remaining_before", "target_runs"]
# every column list of train.py that a model is fitted on: (module attribute, key or None)
FITTED_LISTS = {"wp_splines_1": ("WP_SPLINES", 1), "wp_splines_2": ("WP_SPLINES", 2), "wp_linear": ("WP_LINEAR", None),
                "state_logit_1": ("STATE_LOGIT", 1), "state_logit_2": ("STATE_LOGIT", 2),
                "ball_outcome": ("OUTCOME_FEATURES", None), "resources": ("RESOURCE_INPUTS", None)}


def _add_input(monkeypatch, where: str, col: str) -> None:
    attr, key = FITTED_LISTS[where]
    if key is None:
        monkeypatch.setattr(train, attr, getattr(train, attr) + [col])
        return
    table = getattr(train, attr)
    monkeypatch.setitem(table, key, table[key] + [col])
    if attr == "WP_SPLINES":
        monkeypatch.setitem(train.WP_DIRECTION, key, {**train.WP_DIRECTION[key], col: 0})


@pytest.mark.parametrize("col", LABELS + HINDSIGHT)
@pytest.mark.parametrize("where", sorted(FITTED_LISTS))
def test_model_step_refuses_a_label_or_hindsight_input(tiny_model_rows, monkeypatch, where, col):
    rows = tiny_model_rows.copy()
    for c in HINDSIGHT:  # model_rows does not carry hindsight state; give it some so only the guard can refuse
        rows[c] = 0.0
    _add_input(monkeypatch, where, col)
    with pytest.raises(SystemExit, match="known before the ball"):
        train.walk_forward(rows, first_season=2000, min_train_matches=1)


def test_model_step_refuses_a_derived_feature_built_from_hindsight(monkeypatch):
    # required_rate from the hindsight target and ball limit instead of the nominal ones
    monkeypatch.setitem(dataset.DERIVED, "required_rate",
                        "CAST(6 * f.runs_required_before AS DOUBLE) / nullif(f.balls_remaining_before, 0)")
    with pytest.raises(SystemExit, match="runs_required_before"):
        dataset.check_inputs_are_pre_ball()


def test_model_step_refuses_a_derived_feature_built_from_the_result(monkeypatch):
    monkeypatch.setitem(dataset.DERIVED, "elo_diff", "CASE WHEN m.winner = f.batting_team THEN 1 ELSE 0 END")
    with pytest.raises(SystemExit, match="winner"):
        dataset.check_inputs_are_pre_ball()


def test_the_real_model_inputs_pass_the_guard():
    dataset.check_inputs_are_pre_ball(train.fitted_inputs(), train.RESOURCE_FEATURES)


# ---------------------------------------------------------------- walk-forward fits

# A fit that reads `rows` (every season) instead of the fold's training rows.
WALK_FORWARD_LEAKS = {
    "resources_and_par_include_the_scored_season": (
        'first = train[(train["innings"] == 1) & (train["season"] >= season - RESOURCE_SEASONS)]',
        'first = rows[(rows["innings"] == 1) & (rows["season"] >= season - RESOURCE_SEASONS) & (rows["season"] <= season)]'),
    "ball_outcome_model_includes_the_scored_season": (
        'recent_train = train[train["season"] >= season - OUTCOME_SEASONS]',
        'recent_train = rows[(rows["season"] >= season - OUTCOME_SEASONS) & (rows["season"] <= season)]'),
}


def _mutated_walk_forward(old: str, new: str, guard: bool = True):
    source = textwrap.dedent(inspect.getsource(train.walk_forward))
    assert source.count(old) == 1, "the walk_forward line this mutation targets has changed"
    namespace = dict(vars(train))
    if not guard and "_prior_only" in namespace:
        namespace["_prior_only"] = lambda frame, *args: frame
    exec(source.replace(old, new), namespace)
    return namespace["walk_forward"]


@pytest.mark.parametrize("leak", sorted(WALK_FORWARD_LEAKS))
def test_walk_forward_refuses_a_fit_that_includes_the_scored_season(tiny_model_rows, leak):
    leaky = _mutated_walk_forward(*WALK_FORWARD_LEAKS[leak])
    with pytest.raises(ValueError, match="held-out or excluded"):
        leaky(tiny_model_rows, 2000, 1)


SCORED = 2023  # the fixture's last ODI season: its fold really fits (three earlier matches, a win, a loss, a tie)


def _scramble_labels(rows, season):
    s = rows["season"] == season
    out = rows.copy()
    out.loc[s, "batting_won"] = 1.0 - out.loc[s, "batting_won"]
    out.loc[s, "outcome_class"] = out.loc[s, "outcome_class"].map(
        lambda c: dataset.CLASSES[(dataset.CLASSES.index(c) + 3) % len(dataset.CLASSES)])
    out.loc[s, "runs_total"] = (out.loc[s, "runs_total"] + 3) % 7
    return out


def held_out_label_leaks(walk_forward, rows, season=SCORED) -> list[str]:
    """Columns of the season's predictions that move when only that season's labels (result, ball outcome, runs)
    are scrambled. A model trained on earlier seasons cannot see them, so nothing may move; wp_model_end alone
    is the state AFTER the last ball of innings 1, which rightly reads that ball's runs."""
    keys = ["match_id", "innings", "delivery_number"]
    a, _ = walk_forward(rows, 2000, 1)
    b, _ = walk_forward(_scramble_labels(rows, season), 2000, 1)
    m = a.merge(b, on=keys, suffixes=("", "_s"))
    m = m[m["match_id"].isin(rows.loc[rows["season"] == season, "match_id"])]
    assert len(m) == (rows["season"] == season).sum() > 0
    cols = [c for c in a.columns if c not in keys and c != "wp_model_end"]
    return [c for c in cols if not ((m[c] == m[c + "_s"]) | (m[c].isna() & m[c + "_s"].isna())).all()]


def test_walk_forward_predictions_never_see_the_scored_season_labels(tiny_model_rows):
    assert held_out_label_leaks(train.walk_forward, tiny_model_rows) == []


@pytest.mark.parametrize("leak", sorted(WALK_FORWARD_LEAKS))
def test_label_scramble_catches_a_fit_on_the_scored_season(tiny_model_rows, leak):
    # the scramble check sees these two leaks by itself, with the runtime guard switched off
    leaky = _mutated_walk_forward(*WALK_FORWARD_LEAKS[leak], guard=False)
    assert held_out_label_leaks(leaky, tiny_model_rows)


def test_label_scramble_catches_the_result_used_as_an_input(tiny_model_rows, monkeypatch):
    monkeypatch.setattr(train, "WP_LINEAR", train.WP_LINEAR + ["batting_won"])
    monkeypatch.setattr(dataset, "check_inputs_are_pre_ball", lambda *args, **kwargs: None)
    monkeypatch.setattr(train, "monotone_violations", lambda *args: 0)  # its probe grid has no result column
    assert "wp_model" in held_out_label_leaks(train.walk_forward, tiny_model_rows)


# ---------------------------------------------------------------- player features as of each match

def _wikidata_folder() -> Path:
    lock = (FIXTURE / "snapshots.lock.yml").read_text()
    snapshot = re.search(r"^wikidata_players: '?([^'\n]+)'?$", lock, re.M).group(1)
    return FIXTURE / "bronze" / "wikidata_players" / snapshot


def _player_form(con, data: Path, out: Path, before_or_on=None) -> dict:
    """fact_player_match_form recomputed from the matches up to a date only. The players table (birth dates and
    their checks) is rebuilt from those same matches too, so no whole-snapshot player attribute is held fixed."""
    from cricket_sim import gold

    silver = data / "silver"
    date_filter = f"WHERE start_date <= DATE '{before_or_on}'" if before_or_on else ""
    con.execute(f"CREATE OR REPLACE VIEW matches AS SELECT * FROM read_parquet('{silver / 'matches.parquet'}') {date_filter}")
    for t in ("balls", "wickets", "match_players"):
        con.execute(f"CREATE OR REPLACE VIEW {t} AS SELECT * FROM read_parquet('{silver / t}.parquet') "
                    "WHERE match_id IN (SELECT match_id FROM matches)")
    # the Cricsheet register (ids and names, no match facts) and the Ari stand-in are inputs, not match data
    con.execute(f"CREATE OR REPLACE VIEW people AS SELECT * FROM read_parquet('{silver / 'people.parquet'}')")
    con.execute(f"CREATE OR REPLACE VIEW stg_ari_players AS SELECT * FROM "
                f"read_parquet('{data / 'staging' / 'stg_ari_players.parquet'}')")
    folder = out / str(before_or_on or "full")
    folder.mkdir(parents=True)
    context_players.build_player_context(con, silver=folder, bronze_folder=_wikidata_folder())
    con.execute("CREATE OR REPLACE TEMP TABLE scope AS SELECT match_id FROM matches")
    con.execute(f"CREATE OR REPLACE TEMP TABLE int_player_match_stats AS {gold.PLAYER_STATS_SQL}")
    cur = con.execute(gold.PLAYER_FORM_SQL)
    names = [d[0] for d in cur.description]
    return {(r[0], r[1]): dict(zip(names, r)) for r in cur.fetchall()}


def _assert_form_ignores_every_later_match(data: Path, out: Path) -> dict:
    con = duckdb.connect()
    full = _player_form(con, data, out)
    dates = [r[0] for r in con.execute(f"SELECT DISTINCT start_date FROM read_parquet('{data / 'silver' / 'matches.parquet'}') "
                                       "ORDER BY 1").fetchall()]
    assert len(dates) >= 8 and len(full) > 150
    for d in dates[:-1]:
        cut = _player_form(con, data, out, before_or_on=d)
        assert cut, d
        for key, row in cut.items():
            assert row == full[key], (d, key, {k: (v, full[key][k]) for k, v in row.items() if v != full[key][k]})
    return full


DHONI, DHONI_QID = "4a8a2e3b", "Q470774"
PLANTED_DOB = "1968-06-01"  # NOT his real date: 42 at the 2011 final, 49 in June 2017, over 50 from May 2019


def _plant_dob(monkeypatch, qid: str, dob: str) -> None:
    """Replace one player's Wikidata birth date in the rows the players build reads (the bronze stays as it is)."""
    real = context_players._read_bronze

    def planted(folder):
        data = real(folder)
        data["dob"] = [r for r in data["dob"] if not r["s"].endswith("/" + qid)] + [
            {"s": "http://www.wikidata.org/entity/" + qid, "dob": dob + "T00:00:00Z", "precision": "11"}]
        return data

    monkeypatch.setattr(context_players, "_read_bronze", planted)


@pytest.mark.parametrize("planted", [False, True], ids=["fixture_dates", "planted_dob_over_50_later"])
def test_player_form_ignores_every_later_match(tiny_build, tmp_path, monkeypatch, planted):
    """Recompute the features with every match after date D deleted: features for matches up to D must not move.

    The birth-date checks depend on a player's matches too (a date implying an age over 50 at the
    LAST match was dropped for every match). The fixture has no player whose age crosses 50 inside it, so the
    second case plants such a date: it must give the age at the earlier matches, the same with or without the
    later ones, and no age once the player would be over 50."""
    if planted:
        _plant_dob(monkeypatch, DHONI_QID, PLANTED_DOB)
    full = _assert_form_ignores_every_later_match(tiny_build, tmp_path)
    if planted:
        ages = {mid: row["age_years"] for (mid, pid), row in full.items() if pid == DHONI}
        assert ages == {"433606": 42, "1022353": 49, "1181768": None, "1144528": None}, ages


def test_player_form_counts_only_strictly_earlier_matches(tiny_build, tmp_path):
    # In the fixture Dhoni plays the 2011 final, the 2017 Champions Trophy match and the 2019 semi-final
    con = duckdb.connect()
    form = _player_form(con, tiny_build, tmp_path)
    dhoni = form[("1144528", DHONI)]
    assert dhoni["prior_matches"] == 2 and dhoni["prior_first_match_date"].isoformat() == "2011-04-02"
    assert form[("1022353", DHONI)]["prior_matches"] == 1
    assert form[("433606", DHONI)]["prior_matches"] == 0  # nothing before the 2011 final in the fixture
