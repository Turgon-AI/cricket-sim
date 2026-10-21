"""The ODI models: no look-ahead anywhere, and outputs that obey cricket's arithmetic."""

import duckdb
import numpy as np
import pandas as pd
import pytest

from cricket_sim.models import elo, train
from cricket_sim.models import run as model_run
from cricket_sim.models.run import _wp_after


def test_elo_uses_only_earlier_dates_and_ignores_same_day_results():
    matches = [("m1", "2020-01-01", "A", "B", "A", None),
               ("m2", "2020-01-02", "A", "C", "C", None),
               ("m3", "2020-01-02", "A", "B", "A", None)]  # same day as m2: must not see m2's result
    r = {(m, t): (e, n) for m, t, e, n in elo.pre_match_ratings(matches)}
    assert r[("m1", "A")] == (1500.0, 0)
    assert r[("m2", "A")] == r[("m3", "A")]          # both rated from the ratings before 2020-01-02
    # Exact values, not just a direction. m1: equal ratings, expected 0.5, K * (1 - 0.5) = 12 points.
    assert r[("m2", "A")] == (1512.0, 1) and r[("m3", "A")] == (1512.0, 1)
    assert r[("m3", "B")] == (1488.0, 1) and r[("m2", "C")] == (1500.0, 0)


def test_elo_tie_and_no_result():
    out = elo.pre_match_ratings([("m1", "2020-01-01", "A", "B", None, "tie"),
                                 ("m2", "2020-01-02", "A", "B", None, "no result"),
                                 ("m3", "2020-01-03", "A", "B", "A", None)])
    r = {(m, t): (e, n) for m, t, e, n in out}
    assert r[("m2", "A")] == (1500.0, 1)             # an even tie moves nothing but counts as played
    assert r[("m3", "A")] == (1500.0, 1)             # a no-result is not counted


# Hand-computed Elo, K = 24, scale 400, start 1500. expected(a, b) = 1 / (1 + 10 ** ((b - a) / 400)).
#   d1  m1: A beats B, both 1500, expected 0.5          -> A 1512, B 1488 (A and B have played 1)
#   d2  m2: A (1512) v C (1500), C wins. Expected for A is 1 / (1 + 10 ** (-12 / 400)) = 0.517263.
#           A takes 24 * (0 - 0.517263) = -12.4143       -> A 1499.5857, C 1512.4143
#           (computed from the wrong side the expectation is 0.482737 and A would end on 1500.4143)
#   d3  m3: B (1488) v A (1499.5857) tie. Expected for B is 1 / (1 + 10 ** (11.5857 / 400)) = 0.483333.
#           B takes 24 * (0.5 - 0.483333) = +0.4000      -> B 1488.4000, A 1499.1857
#   d4  m4: B v A again, no result: nothing moves, nobody's count changes
#   d5  m5: just reads the ratings
ELO_MATCHES = [("m1", "2020-01-01", "A", "B", "A", None),
               ("m2", "2020-01-02", "A", "C", "C", None),
               ("m3", "2020-01-03", "B", "A", None, "tie"),
               ("m4", "2020-01-04", "B", "A", None, "no result"),
               ("m5", "2020-01-05", "A", "B", "B", None)]


def test_elo_expected_score_is_for_the_first_team_and_ratings_match_hand_computation():
    assert elo.expected(1512.0, 1500.0) == pytest.approx(0.517263, abs=1e-6)
    assert elo.expected(1500.0, 1512.0) == pytest.approx(0.482737, abs=1e-6)      # the other side is not the same
    assert elo.expected(1500.0, 1500.0) == 0.5
    r = {(m, t): (e, n) for m, t, e, n in elo.pre_match_ratings(ELO_MATCHES)}
    want = {("m1", "A"): (1500.0, 0), ("m1", "B"): (1500.0, 0),
            ("m2", "A"): (1512.0, 1), ("m2", "C"): (1500.0, 0),
            ("m3", "B"): (1488.0, 1), ("m3", "A"): (1499.585699, 2),
            ("m4", "B"): (1488.400008, 2), ("m4", "A"): (1499.185692, 3),
            ("m5", "B"): (1488.400008, 2), ("m5", "A"): (1499.185692, 3)}      # m4 was a no-result: nothing moved
    assert set(r) == set(want)
    for key, (rating, played) in want.items():
        assert r[key][0] == pytest.approx(rating, abs=1e-5), key
        assert r[key][1] == played, key


def test_elo_the_underdog_team_listed_first_is_rated_on_its_own_side():
    # B (1488) is listed FIRST and wins from behind: the gain is K * (1 - expected(B, A)), not K * (1 - expected(A, B)).
    out = elo.pre_match_ratings([("m1", "2020-01-01", "A", "B", "A", None), ("m2", "2020-01-02", "B", "A", "B", None),
                                 ("m3", "2020-01-03", "A", "B", None, "no result")])
    r = {(m, t): e for m, t, e, n in out}
    assert r[("m2", "B")] == pytest.approx(1488.0) and r[("m2", "A")] == pytest.approx(1512.0)
    # expected(B, A) = 1 / (1 + 10 ** (24 / 400)) = 0.46551; B takes 24 * 0.53449 = +12.8276, A loses the same
    assert r[("m3", "B")] == pytest.approx(1488.0 + 24 * (1 - 1 / (1 + 10 ** (24 / 400))), abs=1e-6)
    assert r[("m3", "B")] == pytest.approx(1500.8276, abs=1e-4)
    assert r[("m3", "A")] == pytest.approx(1499.1724, abs=1e-4)


def test_elo_counts_matches_played_for_both_teams():
    out = elo.pre_match_ratings([("m1", "2020-01-01", "A", "B", "A", None), ("m2", "2020-01-02", "A", "B", "B", None),
                                 ("m3", "2020-01-03", "B", "C", None, "tie"), ("m4", "2020-01-04", "C", "B", None, None)])
    played = {(m, t): n for m, t, e, n in out}
    assert (played[("m3", "B")], played[("m3", "C")]) == (2, 0)
    assert (played[("m4", "B")], played[("m4", "C")]) == (3, 1)     # the tie counts for BOTH teams; m4 is a no-result


def test_wp_after_does_not_credit_the_innings_break_to_a_ball():
    # The last ball of innings 1 ends on the innings-1 model's WP for the state after that ball
    # (wp_model_end), not on the chase model's first WP flipped (0.7 here). The gap between the two models
    # belongs to the innings break, not to the striker of the last ball. The match's last ball ends on the result.
    df = pd.DataFrame({"match_id": ["m"] * 4, "innings": [1, 1, 2, 2], "delivery_number": [1, 2, 1, 2],
                       "wp_model": [0.5, 0.6, 0.3, 0.2], "wp_model_end": [np.nan, 0.65, np.nan, np.nan],
                       "batting_won": [0.0, 0.0, 1.0, 1.0]})
    assert _wp_after(df).tolist() == [0.6, 0.65, 0.2, 1.0]


def _synthetic_wp_rows(innings: int, n: int = 20000) -> pd.DataFrame:
    """Training rows whose win rate turns the WRONG way in places (as thin, noisy history can): it falls at very
    high projected totals, and rises at 9 down. A free spline follows that; the published model must not."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"scheduled_balls_remaining_before": rng.integers(1, 301, n),
                       "innings_wickets_before": rng.integers(0, 10, n),
                       "elo_diff": rng.normal(0, 100, n), "batting_at_home": rng.integers(-1, 2, n)})
    wk = df["innings_wickets_before"].to_numpy()
    if innings == 1:
        df["projected_vs_par"] = rng.uniform(-150, 150, n)
        x = df["projected_vs_par"].to_numpy()
        logit = np.where(x < 80, 0.03 * x, 0.03 * (160 - x)) - 0.2 * wk + 1.5 * (wk == 9)
    else:
        df["chase_difficulty"] = rng.uniform(-2, 2, n)
        df["log_required_rate"] = np.log1p(rng.uniform(0, 20, n))
        cd, lrr = df["chase_difficulty"].to_numpy(), df["log_required_rate"].to_numpy()
        logit = -1.5 * cd + 2.5 * np.clip(cd - 1, 0, None) - 0.5 * lrr + 1.0 * (lrr > 2.8) - 0.2 * wk + 1.5 * (wk == 9)
    df["batting_won"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    return df


@pytest.mark.parametrize("innings", [1, 2])
def test_win_probability_model_is_monotone_even_when_history_is_not(innings):
    # Innings 1 WP never falls as the projected total rises; WP never rises with a wicket down;
    # a chase's WP never rises as the chase gets harder (runs needed against resources, or required rate).
    train_rows = _synthetic_wp_rows(innings)
    rising = {1: ["projected_vs_par"], 2: []}[innings]
    falling = {1: ["innings_wickets_before"],
               2: ["chase_difficulty", "log_required_rate", "innings_wickets_before"]}[innings]
    axes = {"projected_vs_par": np.linspace(-250, 300, 56), "chase_difficulty": np.linspace(-3, 3, 61),
            "log_required_rate": np.log1p(np.linspace(0, 36, 37)), "innings_wickets_before": np.arange(10)}
    base = {"scheduled_balls_remaining_before": 60, "innings_wickets_before": 3, "elo_diff": 0.0,
            "batting_at_home": 0, "projected_vs_par": 0.0, "chase_difficulty": 0.0, "log_required_rate": np.log1p(6)}
    bad = {}
    for feature in rising + falling:
        for balls in (1, 12, 60, 240):
            for wk in (0, 5, 8):
                grid = pd.DataFrame({**base, "scheduled_balls_remaining_before": balls, "innings_wickets_before": wk,
                                     feature: axes[feature]})
                step = np.diff(train._gam(train_rows, grid, innings)[0])
                wrong = (step < -1e-12) if feature in rising else (step > 1e-12)
                bad[feature] = bad.get(feature, 0) + int(wrong.sum())
    assert bad == {f: 0 for f in rising + falling}


def test_every_fold_passes_the_monotone_grid_check(tiny_build):
    # Each fold's fitted models are probed on a grid of raw match states (runs, wickets, balls,
    # target, Elo, home); the count of wrong-way steps is published and the contract requires 0.
    rows = q(tiny_build, """SELECT season, innings_1_monotone_violations, innings_2_monotone_violations
                            FROM '{models}/report_model_folds.parquet'""")
    assert rows and all(v1 == 0 and v2 == 0 for _, v1, v2 in rows), rows


# Count definition, the same as the model contract: evaluated rows only; a boundary ball is
# fact_delivery.is_four OR is_six with no dismissal on it; a wicket ball is dismissal_count > 0 (retirements are
# not dismissals) with runs_total = 0 (runs on the same ball can rightly outweigh the wicket); every ball of the
# innings counts, the last one included. A wrong-way move of up to WPA_TOLERANCE (0.1 WP points) is allowed: a ball
# also uses up one ball of time (a free spline term), which can outweigh a runs or wickets term that is flat where its
# monotone constraint binds. The contract uses the same number.
WPA_TOLERANCE = 0.001
DELIVERY_JOIN = """FROM '{models}/model_ball_win_probability.parquet' w JOIN read_parquet('{gold}/fact_delivery.parquet') f
                   USING (match_id, innings, delivery_number) WHERE w.evaluated"""


def test_no_first_innings_boundary_lowers_the_batting_side_win_probability(tiny_build):
    (n, wrong), = q(tiny_build, (f"SELECT count(*), count(*) FILTER (w.wpa < -{WPA_TOLERANCE}) " + DELIVERY_JOIN +
                                 " AND w.innings = 1 AND (f.is_four OR f.is_six) AND f.dismissal_count = 0")
                    .replace("{gold}", str(tiny_build / "gold")))
    assert n > 0 and wrong == 0, f"{wrong} of {n} first-innings boundary balls lower WP"


def q(data, sql):
    return duckdb.connect().execute(sql.replace("{models}", str(data / "models"))).fetchall()


def test_every_prediction_comes_from_a_model_trained_on_earlier_seasons(tiny_build):
    bad = q(tiny_build, """SELECT count(*) FROM '{models}/model_ball_win_probability.parquet' w
                           JOIN read_parquet('""" + str(tiny_build / "gold" / "dim_match.parquet") + """') m USING (match_id)
                           WHERE w.trained_through_season >= year(m.start_date)""")
    assert bad == [(0,)]


def test_walk_forward_split_uses_only_prior_seasons_for_fit_rows():
    rows = pd.DataFrame({"season": [2018, 2019, 2020, 2020], "is_clean": [True, True, True, False],
                         "match_id": ["old", "prior", "heldout", "dirty"]})
    fit, heldout = train._season_split(rows, 2020)
    assert fit["match_id"].tolist() == ["old", "prior"]
    assert fit["season"].max() < 2020
    assert heldout["match_id"].tolist() == ["heldout", "dirty"]


def test_failed_model_run_restores_previous_outputs(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    (models / "known-good.txt").write_text("prior run")
    monkeypatch.setattr(model_run, "MODELS", models)
    monkeypatch.setattr(model_run, "_git_state", lambda: {"dirty": False, "commit": "test"})

    def fail(stage, *args):
        assert (models / "known-good.txt").read_text() == "prior run"
        (stage / "models").mkdir()
        (stage / "models" / "partial.txt").write_text("incomplete")
        raise RuntimeError("simulated model failure")

    monkeypatch.setattr(model_run, "run_worker", fail)
    with pytest.raises(RuntimeError, match="simulated"):
        model_run.run(allow_dirty=True)
    assert (models / "known-good.txt").read_text() == "prior run"
    assert not (models / "partial.txt").exists()


def test_probabilities_are_valid_and_swings_add_up(tiny_build):
    (n, bad_wp, bad_sum, max_gap), = q(tiny_build, """
        SELECT count(*), count(*) FILTER (wp_before NOT BETWEEN 0 AND 1),
               count(*) FILTER (abs(p_dot + p_1 + p_2 + p_3 + p_4 + p_6 + p_wicket + p_extra - 1) > 1e-5),
               max(abs(w.wpa - (w.wp_after - w.wp_before)))
        FROM '{models}/model_ball_win_probability.parquet' w JOIN '{models}/model_ball_outcome.parquet' USING (match_id, innings, delivery_number)""")
    assert n > 1000 and bad_wp == 0 and bad_sum == 0 and max_gap < 1e-5


@pytest.mark.parametrize("innings", [1, 2])
def test_win_probability_falls_when_a_wicket_falls(tiny_build, innings):
    # Every evaluated wicket ball with no run, not the average, must not raise the batting side's WP
    (n, wrong), = q(tiny_build, (f"SELECT count(*), count(*) FILTER (w.wpa > {WPA_TOLERANCE}) " + DELIVERY_JOIN +
                                 f" AND w.innings = {innings} AND f.dismissal_count > 0 AND f.runs_total = 0")
                    .replace("{gold}", str(tiny_build / "gold")))
    assert n > 0 and wrong == 0, f"innings {innings}: {wrong} of {n} wicket balls with no run raise WP"


@pytest.mark.parametrize("rule, corrupt", [
    ("first_innings_boundary_never_lowers_win_probability",
     "innings = 1 AND (f.is_four OR f.is_six) AND f.dismissal_count = 0"),
    ("wicket_without_runs_never_raises_win_probability", "innings = 2 AND f.dismissal_count > 0 AND f.runs_total = 0"),
])
def test_contract_rejects_wrong_signed_wpa(tiny_build, tmp_path, monkeypatch, rule, corrupt):
    from cricket_sim import contracts
    import shutil

    models = tmp_path / "models"
    shutil.copytree(tiny_build / "models", models)
    monkeypatch.setitem(contracts.LAYERS, "models", models)
    monkeypatch.setitem(contracts.LAYERS, "gold", tiny_build / "gold")
    path = models / "model_ball_win_probability.parquet"
    con = duckdb.connect()
    con.execute(f"CREATE TABLE corrupted AS SELECT * FROM read_parquet('{path}')")
    sign = "-" if "boundary" in rule else ""
    con.execute(f"""UPDATE corrupted w SET wpa = {sign}0.01, wp_after = wp_before + {sign}0.01
                    FROM read_parquet('{tiny_build / "gold" / "fact_delivery.parquet"}') f
                    WHERE w.match_id = f.match_id AND w.innings = f.innings AND w.delivery_number = f.delivery_number
                      AND w.evaluated AND w.wp_before BETWEEN 0.02 AND 0.98 AND w.{corrupt}""")
    path.unlink()
    con.execute(f"COPY corrupted TO '{path}' (FORMAT parquet)")
    with pytest.raises(SystemExit, match=rule):
        contracts.enforce(con, layers=("models",))


def test_excluded_matches_have_no_published_win_probability(tiny_build):
    rows = q(tiny_build, """
        SELECT count(*), count(*) FILTER (wp_before IS NOT NULL OR wp_after IS NOT NULL
                    OR wpa IS NOT NULL OR expected_remaining_runs IS NOT NULL)
        FROM '{models}/model_ball_win_probability.parquet' WHERE NOT evaluated
    """)
    assert rows[0][0] > 0, "fixture must include an excluded rain-affected ODI"
    assert rows[0][1] == 0


def test_contract_rejects_numeric_probability_on_excluded_row(tiny_build, tmp_path, monkeypatch):
    from cricket_sim import contracts
    import shutil

    models = tmp_path / "models"
    shutil.copytree(tiny_build / "models", models)
    monkeypatch.setitem(contracts.LAYERS, "models", models)
    # model rules join gold tables (${table:dim_match}), so gold must point at the same build, not <repo>/data
    monkeypatch.setitem(contracts.LAYERS, "gold", tiny_build / "gold")
    path = models / "model_ball_win_probability.parquet"
    con = duckdb.connect()
    con.execute(f"CREATE TABLE corrupted AS SELECT * FROM read_parquet('{path}')")
    con.execute("UPDATE corrupted SET wp_before = 0.5 WHERE NOT evaluated")
    path.unlink()
    con.execute(f"COPY corrupted TO '{path}' (FORMAT parquet)")
    with pytest.raises(SystemExit, match="excluded_rows_have_no_numeric_outputs"):
        contracts.enforce(con, layers=("models",))


def test_excluded_matches_have_no_ball_outcome_numbers(tiny_build):
    count, numeric = q(tiny_build, """
        SELECT count(*), count(*) FILTER (p_dot IS NOT NULL OR p_1 IS NOT NULL OR p_2 IS NOT NULL
          OR p_3 IS NOT NULL OR p_4 IS NOT NULL OR p_6 IS NOT NULL OR p_wicket IS NOT NULL
          OR p_extra IS NOT NULL OR expected_runs IS NOT NULL)
        FROM '{models}/model_ball_outcome.parquet' WHERE NOT evaluated
    """)[0]
    assert count > 0 and numeric == 0


def test_actual_training_guard_kills_inclusive_cutoff_mutation(monkeypatch):
    import inspect
    source = inspect.getsource(train._season_split)
    assert 'rows["season"] < season' in source
    namespace = {"pd": pd}
    exec(source.replace('rows["season"] < season', 'rows["season"] <= season'), namespace)
    monkeypatch.setattr(train, "_season_split", namespace["_season_split"])
    rows = pd.DataFrame({"match_id": ["past", "heldout"], "season": [2019, 2020],
                         "is_clean": [True, True], "innings": [1, 1], "delivery_number": [1, 1],
                         "innings_runs_before": [0, 0], "runs_total": [1, 1]})
    with pytest.raises(ValueError, match="training rows contain held-out"):
        train.walk_forward(rows, first_season=2020, min_train_matches=1)


@pytest.mark.parametrize("min_train_matches, folds", [
    (1, [(2017, 1), (2019, 1), (2023, 3)]),   # each season has at least 1 clean earlier match
    (2, [(2023, 3)]),
    (3, [(2023, 3)]),                         # 2023 has EXACTLY 3 clean earlier matches: enough ("at least")
    (4, []),                                  # one short: no season is scored
])
def test_a_season_is_scored_when_it_has_exactly_min_train_matches_of_clean_history(tiny_model_rows, min_train_matches, folds):
    # This edge needs its own test: a size assertion elsewhere would only catch it by accident.
    # The fixture's clean ODIs are 2011 (1 match), 2019 (2 matches) and 2023; the 2017 rain match is not clean.
    pred, records = train.walk_forward(tiny_model_rows, first_season=2000, min_train_matches=min_train_matches)
    assert [(r["season"], r["train_matches"]) for r in records] == folds
    assert len(pred) == sum(r["test_balls"] for r in records)
