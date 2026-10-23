"""The Cricsheet-only player and venue baseline (docs/VALIDATION.md): counting on training balls only."""

import numpy as np
import pandas as pd

from cricket_sim.models import dataset, train
from cricket_sim.models.dataset import CLASSES

C = len(CLASSES)


def _phase_rows(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"innings": rng.integers(1, 3, n), "over": rng.integers(0, 50, n),
                         "innings_wickets_before": rng.integers(0, 10, n)})


def test_unknown_players_and_venues_get_exactly_the_phase_average():
    train_df = _phase_rows(500, 1).assign(outcome_class=np.random.default_rng(2).choice(CLASSES, 500),
                                        batter_id="a", bowler_id="b", venue_id="v")
    test_df = _phase_rows(20, 3).assign(batter_id="new", bowler_id="new", venue_id="new")
    overall = np.full(C, 1 / C)
    phase_train, phase_test = np.full((500, C), 1 / C), np.random.default_rng(4).dirichlet(np.ones(C), 20)
    out = train.player_venue_average(train_df, test_df, phase_train, phase_test, overall)
    assert np.allclose(out, phase_test)


def test_a_boundary_hitter_raises_fours_and_the_result_still_sums_to_one():
    rng = np.random.default_rng(5)
    train_df = _phase_rows(4000, 6).assign(outcome_class=rng.choice(CLASSES, 4000), bowler_id="b", venue_id="v",
                                         batter_id=np.where(np.arange(4000) < 2000, "hitter", "blocker"))
    train_df.loc[train_df["batter_id"] == "hitter", "outcome_class"] = "4"       # every ball a four
    train_df.loc[train_df["batter_id"] == "blocker", "outcome_class"] = "dot"    # every ball a dot
    overall = np.bincount(train_df["outcome_class"].map(CLASSES.index), minlength=C) / 4000
    phase_train = np.tile(overall, (4000, 1))
    test_df = _phase_rows(2, 7).assign(batter_id=["hitter", "blocker"], bowler_id="b", venue_id="v")
    phase_test = np.tile(overall, (2, 1))
    out = train.player_venue_average(train_df, test_df, phase_train, phase_test, overall)
    assert np.allclose(out.sum(axis=1), 1)
    four, dot = CLASSES.index("4"), CLASSES.index("dot")
    assert out[0, four] > overall[four] > out[1, four]
    assert out[1, dot] > overall[dot] > out[0, dot]
    # shrinkage: 200 pseudo-balls of "as expected" keep a 2,000-ball perfect record short of certainty
    assert out[0, four] < 1 and out[1, four] > 0


def test_the_baseline_inputs_are_known_before_the_ball():
    fitted = train.fitted_inputs()
    assert set(fitted["player_venue_baseline"]) == {"innings", "over", "innings_wickets_before",
                                                    "batter_id", "bowler_id", "venue_id"}
    dataset.check_inputs_are_pre_ball(fitted, train.RESOURCE_FEATURES)   # raises if venue_id were not pre-match
