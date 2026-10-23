"""Walk-forward training. For each season Y, every model is fitted on clean matches from seasons before Y and
then predicts every ball of season Y. So every published probability comes from a model that had never seen
that match, or anything after it.

Win probability (one model per innings): a logistic regression with smooth spline terms for the match state
and linear terms for the two match facts (a generalised additive model):
  resources   a DLS/WASP-style table of expected remaining runs given balls left and wickets lost, learned
              each fold from earlier first innings (monotone: more balls never fewer runs, more wickets
              down never more), read at every over boundary and joined linearly in between, so one ball is
              never worth a jump of several runs. Innings 1 uses projected total versus recent par; innings 2
              uses log((runs needed + 1) / (expected remaining + 1)) and log(1 + required rate).
  state       balls left and wickets down, as splines (the end of a chase is strongly non-linear).
  elo_diff    pre-match Elo difference (elo.py), batting minus bowling side: linear.
  home        +1 batting side at home, -1 bowling side at home, 0 neutral: linear.
  Shape (MONOTONE): the spline terms that cricket fixes a direction for are monotone by construction (a
  monotone spline basis with sign-bounded coefficients): innings 1 WP rises with the projected total and falls
  with wickets down; a chase's WP falls with chase difficulty, required rate and wickets down. Balls left stays
  a free spline. A chase with scores level cannot be lost, so its WP is at least 0.5. Each fold is probed on a
  grid of raw match states and the count of wrong-way steps is published (report_model_folds, must be 0).
  Why this and not boosting: on held-out seasons, gradient boosting memorised training matches (every ball of a
  match shares its ratings and result) and scored worse; a purely linear logistic was overconfident late in
  chases. This model scored best on log loss, Brier and late-chase Brier (docs/MODEL_CARD.md).

Ball outcome (dot, 1, 2, 3, 4, 6, wicket, extra): gradient boosting on the pre-ball state, trained on the 10
seasons before Y (scoring drifts by era). Outcomes vary ball to ball, so there is nothing to memorise, and it
beats the phase-average baseline.

Baselines fitted on the same rows: the base rate; a logistic regression on the raw match state (in the spirit of
Asif & McHale 2016); for the ball outcome, class frequencies by innings x phase x wickets, and the Cricsheet-only
player and venue averages of docs/VALIDATION.md: the phase frequencies scaled by how often the striker, the
bowler and the venue produced each outcome in the training seasons relative to what the phase frequencies
expected of them, each ratio shrunk towards 1 by PLAYER_VENUE_PSEUDO_BALLS balls of "as expected"
(player_venue_average).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize  # installed with scikit-learn, which requires it
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from . import dataset
from .dataset import CLASSES

SEED = 20261016
FIRST_TEST_SEASON = 2008
HOLDOUT_FROM_SEASON = 2023  # locked holdout: one fit on every clean season before this, scored once from it on
RESOURCE_SEASONS, PAR_SEASONS, OUTCOME_SEASONS = 8, 4, 10
WP_SPLINES = {1: ["projected_vs_par", "scheduled_balls_remaining_before", "innings_wickets_before"],
              2: ["chase_difficulty", "scheduled_balls_remaining_before", "innings_wickets_before", "log_required_rate"]}
WP_LINEAR = ["elo_diff", "batting_at_home"]
# Direction cricket fixes for each spline term: +1 WP never falls as it rises, -1 never rises, 0 free.
WP_DIRECTION = {1: {"projected_vs_par": 1, "scheduled_balls_remaining_before": 0, "innings_wickets_before": -1},
                2: {"chase_difficulty": -1, "scheduled_balls_remaining_before": 0, "innings_wickets_before": -1,
                    "log_required_rate": -1}}
STATE_LOGIT = {1: ["innings_runs_before", "innings_wickets_before", "scheduled_balls_remaining_before", "elo_diff"],
               2: ["nominal_runs_required_before", "innings_wickets_before", "scheduled_balls_remaining_before",
                   "required_rate", "elo_diff"]}
OUTCOME_FEATURES = ["innings", "innings_runs_before", "innings_wickets_before", "legal_balls_before",
                    "scheduled_balls_remaining_before", "nominal_runs_required_before", "run_rate", "required_rate",
                    "elo_diff", "year"]
RESOURCE_INPUTS = ["scheduled_balls_remaining_before", "innings_wickets_before"]
PHASE_INPUTS = ["innings", "over", "innings_wickets_before"]
PLAYER_VENUE_ENTITIES = ["batter_id", "bowler_id", "venue_id"]
PLAYER_VENUE_PSEUDO_BALLS = 200  # fixed before the baseline was scored, never tuned on the folds
# The features add_resource_features computes, and what each is computed from (the resource table and par are
# fitted per fold on earlier seasons). dataset.check_inputs_are_pre_ball follows these back to model_rows.
RESOURCE_FEATURES = {"expected_remaining_runs": RESOURCE_INPUTS,
                     "projected_total": ["innings_runs_before", "expected_remaining_runs"],
                     "projected_vs_par": ["projected_total"],
                     "chase_difficulty": ["nominal_runs_required_before", "expected_remaining_runs"],
                     "log_required_rate": ["required_rate"]}


def fitted_inputs() -> dict[str, list[str]]:
    """Every column list a model here is fitted on. walk_forward refuses to start unless each column is known
    before the ball (dataset.check_inputs_are_pre_ball)."""
    return {"resource_table": RESOURCE_INPUTS,
            **{f"win_probability_innings_{i}": WP_SPLINES[i] + WP_LINEAR for i in (1, 2)},
            **{f"state_logistic_innings_{i}": STATE_LOGIT[i] for i in (1, 2)},
            "ball_outcome": OUTCOME_FEATURES, "phase_baseline": PHASE_INPUTS,
            "player_venue_baseline": PHASE_INPUTS + PLAYER_VENUE_ENTITIES}


class Resources:
    """Expected runs still to come in an innings, given legal balls left and wickets down."""

    def __init__(self, first_innings: pd.DataFrame):
        x = first_innings[RESOURCE_INPUTS]
        self.model = HistGradientBoostingRegressor(monotonic_cst=[1, -1], min_samples_leaf=500, max_iter=200,
                                                   learning_rate=0.1, early_stopping=False, random_state=SEED)
        self.model.fit(x, first_innings["innings_total"] - first_innings["innings_runs_before"])
        # The boosted table is a staircase: between two neighbouring balls it can jump by 5+ runs, so a four
        # could LOWER the projected total. Read it at every over boundary (0, 6, 12, ... balls left) and join
        # those points linearly. Monotone in balls and in wickets still holds: the joined points keep the order.
        top = int(np.ceil(max(float(x["scheduled_balls_remaining_before"].max()), 6) / 6) * 6)
        self.knots = np.arange(0, top + 1, 6, dtype=float)
        self.table = np.stack([np.clip(self.model.predict(pd.DataFrame({
            "scheduled_balls_remaining_before": self.knots, "innings_wickets_before": w})), 0, None) for w in range(11)])

    def expected_remaining(self, df: pd.DataFrame) -> np.ndarray:
        balls = df["scheduled_balls_remaining_before"].to_numpy(dtype=float)
        wkts = np.clip(np.rint(df["innings_wickets_before"].to_numpy(dtype=float)), 0, 10).astype(int)
        out = np.zeros(len(df))
        for w in np.unique(wkts):
            k = wkts == w
            out[k] = np.interp(balls[k], self.knots, self.table[w])
        return out


def add_resource_features(df: pd.DataFrame, resources: Resources, par: float) -> pd.DataFrame:
    df = df.copy()
    df["expected_remaining_runs"] = resources.expected_remaining(df)
    df["projected_total"] = df["innings_runs_before"] + df["expected_remaining_runs"]
    df["projected_vs_par"] = df["projected_total"] - par
    df["chase_difficulty"] = np.log((df["nominal_runs_required_before"].clip(lower=0) + 1)
                                    / (df["expected_remaining_runs"] + 1))
    # a required rate is undefined with no balls left; 36 an over (six sixes) caps it
    df["log_required_rate"] = np.log1p(df["required_rate"].clip(lower=0).fillna(36).clip(upper=36))
    return df


def _with_ties_split(train: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """A tie counts as half a win: its rows appear once as a win and once as a loss, each at weight 0.5."""
    decided = train[train["batting_won"] != 0.5]
    ties = train[train["batting_won"] == 0.5]
    rows = pd.concat([decided, ties, ties], ignore_index=True)
    y = np.concatenate([decided["batting_won"].to_numpy(), np.ones(len(ties)), np.zeros(len(ties))]).astype(int)
    w = np.concatenate([np.ones(len(decided)), np.full(2 * len(ties), 0.5)])
    return rows, y, w


def _logit(train: pd.DataFrame, test: pd.DataFrame, feats: list[str]) -> tuple[np.ndarray, dict]:
    rows, y, w = _with_ties_split(train)
    ok = rows[feats].notna().all(axis=1).to_numpy()
    if len(np.unique(y[ok])) < 2:  # too little history to fit (only happens on tiny test fixtures)
        return np.full(len(test), float(np.average(y, weights=w))), {}
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(rows.loc[ok, feats], y[ok], logisticregression__sample_weight=w[ok])
    p = np.full(len(test), np.nan)
    test_ok = test[feats].notna().all(axis=1).to_numpy()
    if test_ok.any():
        p[test_ok] = model.predict_proba(test.loc[test_ok, feats])[:, 1]
    scaler, lr = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    # coefficients per unit of each input (undoing the scaling), for the model card
    coefs = {f: float(c / s) for f, c, s in zip(feats, lr.coef_[0], scaler.scale_)}
    coefs["intercept"] = float(lr.intercept_[0] - sum(c * m / s for c, m, s in zip(lr.coef_[0], scaler.mean_, scaler.scale_)))
    return p, coefs


class WinProbabilityGAM:
    """The published win-probability model: a penalised logistic regression (same objective as scikit-learn's
    LogisticRegression with C=1) on cubic B-spline terms (7 knots, linear extrapolation) and standardised linear
    terms. A term with a fixed direction uses the cumulative ("monotone") form of its B-spline basis,
    I_j = B_j + ... + B_K, each of which never falls as the input rises; its coefficients are bounded to one sign,
    so the term can only move WP in that direction, inside and outside the training range."""

    def __init__(self, innings: int):
        self.innings, self.spl, self.lin = innings, WP_SPLINES[innings], WP_LINEAR
        self.constant, self.coefficients = None, {}

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for f, st in zip(self.spl, self.splines):
            b = st.transform(df[[f]].to_numpy(dtype=float))
            direction = WP_DIRECTION[self.innings][f]
            # basis functions sum to 1 (the intercept), so one column of each term is dropped
            cols.append(b[:, 1:] if direction == 0 else direction * np.cumsum(b[:, ::-1], axis=1)[:, ::-1][:, 1:])
        cols.append(self.scaler.transform(df[self.lin].to_numpy(dtype=float)))
        return np.hstack(cols)

    def fit(self, train: pd.DataFrame) -> "WinProbabilityGAM":
        rows, y, w = _with_ties_split(train)
        if len(np.unique(y)) < 2:  # too little history to fit (only happens on tiny test fixtures)
            self.constant = float(np.average(y, weights=w))
            return self
        self.splines = [SplineTransformer(n_knots=7, degree=3, extrapolation="linear").fit(rows[[f]].to_numpy(dtype=float))
                        for f in self.spl]
        self.scaler = StandardScaler().fit(rows[self.lin].to_numpy(dtype=float))
        x = self._design(rows)
        lower = [0.0 if WP_DIRECTION[self.innings][f] else None for f, st in zip(self.spl, self.splines)
                 for _ in range(st.n_features_out_ - 1)] + [None] * len(self.lin)

        def loss(theta):
            z = x @ theta[1:] + theta[0]
            g = w * (1 / (1 + np.exp(-z)) - y)
            return float(w @ (np.logaddexp(0, z) - y * z) + 0.5 * theta[1:] @ theta[1:]), np.r_[g.sum(), x.T @ g + theta[1:]]

        fit = minimize(loss, np.zeros(x.shape[1] + 1), jac=True, method="L-BFGS-B",
                       bounds=[(None, None)] + [(lo, None) for lo in lower], options={"maxiter": 5000})
        self.theta = fit.x
        lin_coef = self.theta[-len(self.lin):]
        self.coefficients = {f: float(c / sc) for f, c, sc in zip(self.lin, lin_coef, self.scaler.scale_)}
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.constant is not None:
            p = np.full(len(df), self.constant)
        else:
            p = 1 / (1 + np.exp(-(self._design(df) @ self.theta[1:] + self.theta[0])))
        if self.innings == 2 and "nominal_runs_required_before" in df:
            # scores level (1 run needed): the chasing side can no longer lose, a tie counts as half a win
            p = np.where(df["nominal_runs_required_before"].to_numpy(dtype=float) <= 1, np.maximum(p, 0.5), p)
        return p


def _gam(train: pd.DataFrame, test: pd.DataFrame, innings: int) -> tuple[np.ndarray, dict]:
    """Fit the published win-probability model and predict test. Returns predictions and the linear coefficients
    (per unit)."""
    model = WinProbabilityGAM(innings).fit(train)
    return model.predict(test), model.coefficients


MONOTONE_GRID_BALLS = [1, 2, 6, 12, 30, 60, 120, 180, 240, 299]


def monotone_violations(model: WinProbabilityGAM, resources: Resources, par: float) -> int:
    """Probe a fitted model on a grid of raw match states (Elo -300/0/+300, home -1/0/1, the balls in
    MONOTONE_GRID_BALLS, wickets 0-9; innings 1: runs 0-500 in 5s; chase: runs needed 1-400 in 3s) and count the
    steps that go the wrong way: innings-1 WP falling as runs rise, any WP rising with one more wicket down, a
    chase's WP rising with more runs needed. A step counts when it moves by more than 1e-12 (float noise)."""
    wrong = 0
    for elo_diff in (-300.0, 0.0, 300.0):
        for home in (-1, 0, 1):
            if model.innings == 1:
                runs, wk, balls = np.meshgrid(np.arange(0, 501, 5), np.arange(10), MONOTONE_GRID_BALLS, indexing="ij")
                g = pd.DataFrame({"innings_runs_before": runs.ravel(), "nominal_runs_required_before": np.nan,
                                  "required_rate": np.nan})
            else:
                need, wk, balls = np.meshgrid(np.arange(1, 401, 3), np.arange(10), MONOTONE_GRID_BALLS, indexing="ij")
                g = pd.DataFrame({"innings_runs_before": 0, "nominal_runs_required_before": need.ravel(),
                                  "required_rate": 6 * need.ravel() / balls.ravel()})
            g["innings_wickets_before"], g["scheduled_balls_remaining_before"] = wk.ravel(), balls.ravel()
            g["elo_diff"], g["batting_at_home"] = elo_diff, home
            p = model.predict(add_resource_features(g, resources, par)).reshape(wk.shape)
            sign = 1 if model.innings == 1 else -1  # along axis 0: innings 1 runs (rise), chase runs needed (fall)
            wrong += int((sign * np.diff(p, axis=0) < -1e-12).sum()) + int((np.diff(p, axis=1) > 1e-12).sum())
    return wrong


def innings_end_state(last_balls: pd.DataFrame, resources: Resources, par: float) -> pd.DataFrame:
    """The innings-1 model's input for the state AFTER the last ball of innings 1: runs, wickets and balls left
    after it, read exactly like any other state. Balls left is held at 1 or more, the fewest the model was fitted
    on (every training row is a state before a ball), so the ball is scored on its runs and wicket and not on an
    extrapolation to 0 balls left."""
    end = last_balls.copy()
    end["innings_runs_before"] = end["innings_runs_before"] + end["runs_total"]
    end["innings_wickets_before"] = (end["innings_wickets_before"] + end["dismissal_count"]).clip(upper=10)
    end["scheduled_balls_remaining_before"] = (end["scheduled_balls_remaining_before"]
                                               - end["is_legal"].astype(int)).clip(lower=1)
    return add_resource_features(end, resources, par)


def fit_outcome(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    y = train["outcome_class"].map(CLASSES.index).to_numpy()
    gbm = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=2000,
                                         l2_regularization=1.0, early_stopping=False, random_state=SEED)
    gbm.fit(train[OUTCOME_FEATURES], y)
    proba = np.zeros((len(test), len(CLASSES)))
    proba[:, gbm.classes_] = gbm.predict_proba(test[OUTCOME_FEATURES])
    overall = np.bincount(y, minlength=len(CLASSES)) / len(y)
    freq = (pd.crosstab(_phase(train), train["outcome_class"], normalize="index")
            .reindex(columns=CLASSES, fill_value=0.0))
    phase = freq.reindex(_phase(test)).to_numpy(copy=True)
    phase[np.isnan(phase).any(axis=1)] = overall
    mean_runs = train.groupby("outcome_class")["runs_total"].mean().reindex(CLASSES).fillna(0).to_numpy()
    phase_train = freq.reindex(_phase(train)).to_numpy(copy=True)
    phase_train[np.isnan(phase_train).any(axis=1)] = overall
    return {"model": proba, "phase_average": phase, "base_rate": np.tile(overall, (len(test), 1)),
            "player_venue_average": player_venue_average(train, test, phase_train, phase, overall),
            "class_mean_runs": mean_runs}


def player_venue_average(train: pd.DataFrame, test: pd.DataFrame, phase_train: np.ndarray, phase_test: np.ndarray,
                         overall: np.ndarray, pseudo_balls: int = PLAYER_VENUE_PSEUDO_BALLS) -> np.ndarray:
    """The Cricsheet-only player and venue baseline (docs/VALIDATION.md): the phase average, scaled per class by
    the striker's, the bowler's and the venue's observed-over-expected ratio in the training seasons.

    For an entity e (a batter, a bowler or a venue) and a class c, observed is the count of c on e's training
    balls and expected is the sum of the phase-average probability of c on those same balls, so an opener who
    faces mostly powerplay balls is compared with powerplay expectations, not with the overall rate. The ratio is
    shrunk towards 1 with `pseudo_balls` balls of "as expected" (ratio = (observed + k p_c) / (expected + k p_c),
    p_c the overall class rate), so a debutant or a new ground gets exactly the phase average. The three ratios
    multiply the phase average and the result is renormalised to sum to 1. Nothing here is fitted: it is counting
    on training balls only, so it uses no information from the scored season."""
    y = np.eye(len(CLASSES))[train["outcome_class"].map(CLASSES.index).to_numpy()]
    out = phase_test.copy()
    prior = pseudo_balls * overall
    for entity in PLAYER_VENUE_ENTITIES:
        key = train[entity].to_numpy()
        observed = pd.DataFrame(y, index=key).groupby(level=0).sum()
        expected = pd.DataFrame(phase_train, index=key).groupby(level=0).sum()
        ratio = (observed + prior) / (expected + prior)
        rows = ratio.reindex(test[entity].to_numpy())  # unknown entity (NaN): ratio 1, the phase average
        out *= np.nan_to_num(rows.to_numpy(), nan=1.0)
    return out / out.sum(axis=1, keepdims=True)


def _phase(df: pd.DataFrame) -> pd.Series:
    innings, overs, wickets = PHASE_INPUTS
    over = pd.cut(df[overs], [-1, 9, 39, 1000], labels=["powerplay", "middle", "death"]).astype(str)
    wk = pd.cut(df[wickets], [-1, 2, 5, 7, 10], labels=["0-2", "3-5", "6-7", "8+"]).astype(str)
    return df[innings].astype(str) + "|" + over + "|" + wk


def innings_totals(rows: pd.DataFrame) -> pd.Series:
    """Final first-innings total per match (runs before the last ball + runs off it)."""
    first = rows[rows["innings"] == 1].sort_values(["match_id", "delivery_number"]).groupby("match_id").tail(1)
    return (first["innings_runs_before"] + first["runs_total"]).set_axis(first["match_id"])


def _prior_only(frame: pd.DataFrame, season: int, what: str) -> pd.DataFrame:
    """Every frame a model is fitted on passes through here, at the fit itself: only clean rows of seasons before
    the scored one. A fit that sees the scored season (or an excluded match) stops the run."""
    if not frame.empty and (frame["season"].max() >= season or not frame["is_clean"].all()):
        raise ValueError(f"season {season}: {what} training rows contain held-out or excluded matches")
    return frame


def _season_split(rows: pd.DataFrame, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return clean prior-season training rows and the held-out season rows."""
    return rows[(rows["season"] < season) & rows["is_clean"]], rows[rows["season"] == season]


def _fit_fold(train: pd.DataFrame, test: pd.DataFrame, season: int, n_train: int) -> tuple[pd.DataFrame, dict]:
    """Fit every model and baseline on `train` (clean rows of seasons before `season`) and predict every ball of
    `test`. Returns one prediction row per test ball and a record of the fold."""
    first = train[(train["innings"] == 1) & (train["season"] >= season - RESOURCE_SEASONS)]
    resources = Resources(_prior_only(first, season, "resource table"))
    recent = first[first["season"] >= season - PAR_SEASONS]
    par = float(_prior_only(recent, season, "par score").groupby("match_id")["innings_total"].first().mean())
    train_f, test_f = add_resource_features(train, resources, par), add_resource_features(test, resources, par)
    pred = test[["match_id", "innings", "delivery_number"]].copy()
    pred["trained_through_season"] = int(train["season"].max())
    pred["train_matches"] = n_train
    pred["expected_remaining_runs"] = test_f["expected_remaining_runs"].to_numpy()
    fold = {"season": int(season), "train_matches": int(n_train), "test_matches": int(test["match_id"].nunique()),
            "test_balls": len(test), "par_first_innings_total": round(par, 1)}
    pred["wp_model_end"] = np.nan
    for innings in (1, 2):
        fold[f"innings_{innings}_monotone_violations"] = 0
        tr, mask = train_f[train_f["innings"] == innings], (test_f["innings"] == innings).to_numpy()
        if tr.empty or not mask.any():
            continue
        te = test_f[mask]
        gam = WinProbabilityGAM(innings).fit(_prior_only(tr, season, f"innings {innings} win probability"))
        pred.loc[mask, "wp_model"], fold[f"coefficients_innings_{innings}"] = gam.predict(te), gam.coefficients
        fold[f"innings_{innings}_monotone_violations"] = monotone_violations(gam, resources, par)
        if innings == 1:  # WP after the last ball of innings 1, from the same model (see run._wp_after)
            last = test[test["innings"] == 1].sort_values(["match_id", "delivery_number"]).groupby("match_id").tail(1)
            pred.loc[last.index, "wp_model_end"] = gam.predict(innings_end_state(last, resources, par))
        pred.loc[mask, "wp_state_logistic"], _ = _logit(_prior_only(tr, season, "state logistic"), te,
                                                        STATE_LOGIT[innings])
        rows_t, y, w = _with_ties_split(_prior_only(tr, season, "base rate"))
        pred.loc[mask, "wp_base_rate"] = float(np.average(y, weights=w))
    recent_train = train[train["season"] >= season - OUTCOME_SEASONS]
    oc = fit_outcome(_prior_only(recent_train, season, "ball outcome"), test)
    for k, cls in enumerate(CLASSES):
        pred[f"p_{cls}"] = oc["model"][:, k]
        pred[f"phase_p_{cls}"] = oc["phase_average"][:, k]
        pred[f"base_p_{cls}"] = oc["base_rate"][:, k]
        pred[f"pv_p_{cls}"] = oc["player_venue_average"][:, k]
    pred["expected_runs"] = oc["model"] @ oc["class_mean_runs"]
    return pred, fold


def _prepare(rows: pd.DataFrame) -> pd.DataFrame:
    dataset.check_inputs_are_pre_ball(fitted_inputs(), RESOURCE_FEATURES)
    rows = rows.copy()
    rows["innings_total"] = rows["match_id"].map(innings_totals(rows))
    return rows


def _guard(train: pd.DataFrame, test: pd.DataFrame, season: int) -> None:
    """Check the actual rows passed downstream to every fitter, not a hard-coded provenance label."""
    if (not train.empty and (train["season"].max() >= season or not train["is_clean"].all())) or \
            set(train["match_id"]) & set(test["match_id"]):
        raise ValueError(f"season {season}: training rows contain held-out or excluded matches")


def walk_forward(rows: pd.DataFrame, first_season: int = FIRST_TEST_SEASON,
                 min_train_matches: int = 300) -> tuple[pd.DataFrame, list[dict]]:
    """One prediction row per ball of every season with enough earlier clean matches, plus a record per fold."""
    rows = _prepare(rows)
    out, folds = [], []
    for season in sorted(s for s in rows["season"].unique() if s >= first_season):
        train, test = _season_split(rows, season)
        _guard(train, test, season)
        n_train = train["match_id"].nunique()
        if n_train < min_train_matches or test.empty:
            continue
        pred, fold = _fit_fold(train, test, season, n_train)
        out.append(pred)
        folds.append(fold)
        print(f"  season {season}: trained on {n_train} matches, predicted {len(test):,} balls")
    return (pd.concat(out, ignore_index=True) if out else pd.DataFrame()), folds


def locked_holdout(rows: pd.DataFrame, from_season: int = HOLDOUT_FROM_SEASON,
                   min_train_matches: int = 300) -> tuple[pd.DataFrame, dict]:
    """The locked temporal holdout of docs/VALIDATION.md: ONE fit on every clean season before `from_season`,
    scored once on every ball from `from_season` on. The walk-forward folds refit each season on more history,
    so this is the stricter number; the two should agree if the fold choice did not flatter the model."""
    rows = _prepare(rows)
    train = rows[(rows["season"] < from_season) & rows["is_clean"]]
    test = rows[rows["season"] >= from_season]
    _guard(train, test, from_season)
    n_train = train["match_id"].nunique()
    if n_train < min_train_matches or test.empty:
        return pd.DataFrame(), {}
    pred, fold = _fit_fold(train, test, from_season, n_train)
    print(f"  locked holdout from {from_season}: trained on {n_train} matches, predicted {len(test):,} balls")
    return pred, fold
