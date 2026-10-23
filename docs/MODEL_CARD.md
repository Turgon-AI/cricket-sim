# Model card: ODI ball-by-ball win probability and expected runs

> **Status (2026-10-21):** Every number below was regenerated from the current code (build `38eb3e0f0d008218`,
> model run `11d528f2bf4769a4`).
> **What the code guarantees:**
> - Venue home status uses only matches on earlier dates (`dataset.home_teams_as_of`).
> - Win probabilities, expected runs and outcome probabilities are withheld on unevaluated rows: none of the
>   127,515 such rows in this run has a number.
> - Each fold checks at run time that its training rows are clean, come from earlier seasons and share no match
>   with the scored season.
> - Win probability is monotone in the state terms whose direction cricket fixes, with balls left held fixed: a
>   higher projected total (innings 1) or fewer runs needed (chase) never lowers the batting side's chance, and more
>   wickets down never raises it. 0 wrong-way steps on the state grid in all 19 folds. On real balls, which also use
>   up a ball, the contract checks that no first-innings boundary lowers it and no wicket without runs raises it,
>   within 0.001.
> - WPA no longer bridges the innings break: the last ball of innings 1 ends at the first-innings model's own end
>   state, so the jump between the two models goes to no ball.
> **Still open:** the model family was chosen on the same walk-forward folds that are scored here, and no locked
> temporal holdout has been run. The first-innings gain over the state logistic is small (0.1892 against 0.1937).
> The model has the lower first-innings Brier in 15 of 19 seasons (two-sided sign test p = 0.02), but those are the
> folds the family was chosen on, the reports hold no confidence intervals, and there is no baseline with the same
> inputs (the state logistic has no home term). The boosted expected-runs table is far off in rare late states
> with few wickets down. Read the scores as walk-forward estimates, not holdout results.

Structured after *Model Cards for Model Reporting* (Mitchell et al., 2019). Numbers are from model run
`11d528f2bf4769a4` (model fingerprint), built on data build `38eb3e0f0d008218`. They are in
`data/models/report_model_*.parquet`, which also ships with the dataset.

## Model details

Two models, trained and published by `cricket model` (`src/cricket_sim/models/`):

- **Win probability.** One model per innings: a logistic regression with smooth spline terms for the match
  state and linear terms for two match facts (a generalised additive model). Its inputs:
  - **Resources:** expected remaining runs for the balls and wickets left, a DLS/WASP-style table re-learned
    each season from earlier first innings, constrained so that more balls never mean fewer runs and more
    wickets down never mean more. The first innings uses projected total minus recent par. The chase uses
    `log((runs needed + 1) / (expected remaining + 1))` and the required rate.
  - **State:** balls left and wickets down.
  - **Match facts:** pre-match Elo difference and home advantage.
  - **Monotone constraint:** each spline term whose direction cricket fixes uses a monotone basis with
    sign-bounded coefficients. In innings 1, a higher projected total never lowers the batting side's chance and
    more wickets down never raise it. In the chase, a harder target, a higher required rate and more wickets down
    never raise it. Balls left stays free. Once the scores are level in a chase the win probability is at least
    0.5. The expected-runs table is read per over, so one ball cannot move the projected total by a table jump.
    Every fold checks a grid of states (`innings_1_monotone_violations`, `innings_2_monotone_violations` in
    `report_model_folds`: 0 in all 19 folds), and the contract checks each evaluated ball: no first-innings
    boundary lowers win probability and no wicket without runs raises it, within 0.001.
  - **Innings break:** `wp_after` on the last ball of innings 1 is the first-innings model's own probability for
    the state after that ball. It is not the chase model's opening probability, so WPA never bridges the innings
    break.
- **Ball outcome.** Eight classes (dot, 1, 2, 3, 4, 6, wicket, extra), from a gradient-boosted classifier on
  the pre-ball state and season, trained on the ten seasons before. The expected runs per ball follow from
  it.

**Walk-forward:** for each season from 2008, both models are fitted only on earlier seasons. The season field
records the training cutoff. The contract checks only that label, but each fold also checks at run time that its
actual training rows are clean, come from earlier seasons and share no match with the scored season, and a test
covers that guard. The model-family choice was made on the same folds used for the reported scores, so a locked
holdout is still needed.

**Licence:** ODC-BY 1.0 for the outputs (derived from Cricsheet). The code is MIT.

## Intended use

- **Primary:** describing ODI matches ball by ball, with a "chance of winning" line, the swing each ball caused
  (win probability added) and runs against an average batter in the same situation. Also as a transparent
  baseline for research.
- **Users:** analysts, journalists, students, and researchers comparing models.
- **Out of scope:**
  - betting;
  - live in-match use where team news, pitch or weather matter;
  - formats other than men's ODIs;
  - judging a player on a few matches.

## Factors

The model sees only:
- the score, wickets, balls left and target;
- the season;
- pre-match Elo;
- home or away.

It does not know:
- the players, the bowler or the batting order;
- the pitch, weather, dew or toss decision;
- team news.

Performance is reported by innings, by era (2008-14, 2015-20, 2021-) and at fixed moments of the match.

## Metrics

Proper scoring rules, as fixed in advance in docs/VALIDATION.md:
- log loss and Brier score per ball;
- Brier score per match at three checkpoints;
- reliability (calibration) tables.

Ties score as 0.5. Every model is compared with baselines fitted on the same rows:
- the base rate;
- a logistic regression on the raw match state (in the spirit of Asif & McHale, 2016);
- for ball outcomes, class frequencies by innings, phase and wickets.

## Evaluation data

1,740 clean held-out ODIs, 2008-2026: no rain rule, a result, both innings of known length, an unrevised chase,
and every quality check passed. That is 962,915 balls. Rain-affected matches, no-results, innings of unknown length and
the one match that fails quality are not predicted: their rows are in the tables with `evaluated = false`, an
`exclusion_reason` and no numbers (127,515 rows), and they are not scored.

## Results

**Walk-forward metrics, regenerated on 2026-10-21 from the current source; not a locked holdout** (lower is better):

| | innings 1 Brier | innings 1 log loss | innings 2 Brier | innings 2 log loss |
|---|---|---|---|---|
| **this model** | **0.1892** | **0.5659** | **0.1197** | **0.3797** |
| state logistic | 0.1937 | 0.5764 | 0.1213 | 0.3842 |
| base rate | 0.2474 | 0.6936 | 0.2469 | 0.6934 |

**By era:**
- **Innings 1:** the model has the lowest Brier in every era (0.189, 0.182 and 0.195), but only against the
  baselines above, none of which gets the home input. Against a baseline given the same inputs the
  first-innings gain is not shown (see the status note).
- **Innings 2:** it is best in 2015-20 (0.107) and 2021- (0.131). In 2008-14 it is slightly worse than the
  state logistic (0.1203 against 0.1195).

**Match-level Brier, side batting first wins:**

| checkpoint | matches | this model | state logistic | base rate |
|---|---|---|---|---|
| first ball of the match | 1,740 | **0.214** | 0.219 | 0.247 |
| first ball of the chase | 1,740 | **0.167** | 0.176 | 0.247 |
| chase, 30 balls left | 856 | 0.076 | 0.080 | 0.244 |

No interval is computed for these checkpoints. The 30-balls-left gap (0.004 on 856 matches) is too small to
read as a real difference.

**Ball outcome, eight classes:**

| model | log loss | Brier |
|---|---|---|
| **this model** | **1.2996** | **0.6297** |
| phase average | 1.3125 | 0.6374 |
| base rate | 1.3438 | 0.6548 |

The model is best in every era.

**Cricinfo's published win probability (internal, same balls):** on the clean held-out balls where Cricinfo's
pre-ball number exists, Brier 0.189 against Cricinfo's 0.210 in the first innings (41,619 balls) and 0.119 against
0.118 in the chase (35,664 balls). Cricinfo's model sees team news and conditions that this one does not, its
perspective is inferred from the data in its favour, and the table (`report_model_vs_cricinfo_internal`) is
Cricinfo-derived, so it is never exported. docs/VALIDATION.md records how the comparison is made.

**Other models (the comparison sets differ, so read these as context, not a ranking):**
- **The open chase model of akshatsharma-02/odi-cricket-analytics** reports Brier 0.132 on chases after August
  2022. This model scores 0.131 on chases from 2021.

**Model choices previously compared on the reported folds (not an independent holdout):**
- **Gradient boosting for win probability:** 0.01-0.03 worse Brier. Every ball of a match shares its ratings and
  its result, so the trees memorise training matches.
- **A purely linear logistic:** overconfident late in chases. The spline terms fixed that.

## Calibration

The full tables are in `report_model_calibration`.

- **First innings:** well calibrated in the middle (25-55%). Overconfident for the favourite: when it says 85%,
  the batting side wins 80% of the time, and when it says 94%, 90%. Underconfident at the low end: when it says
  15%, the side wins 19%. Because win probability can no longer fall as runs rise, huge totals (above about 300
  in the 2015 fold) sit flat near 0.87-0.90 and stay under-confident.
- **Chases:** the chasing side wins 3-5 points more often than predicted in the 15-65% range. The gap shows
  in 2008-14 as well as in later eras, so it is not only that chasing became easier faster than the model,
  trained on earlier seasons, adjusts. From 2021 the model is also overconfident for the chasing side above
  65%: when it says 75%, the side wins 70% of the time. This is the main known weakness. A recalibration step
  fitted on the most recent seasons is a candidate fix.
- **Ball outcomes:** wicket and six probabilities are well calibrated up to about 12%. Above that, the rare
  high-probability balls (end-of-innings slogging) are somewhat overconfident for sixes.

## Ethical considerations

- The data is public match records. Using expected-runs or win-probability-added numbers to judge a player
  needs care:
  - the yardstick is an *average* batter;
  - it ignores role, bowler and pitch;
  - single seasons are noisy, so compare periods with match-level bootstrap intervals.
- The model is not built or intended for betting.

## Caveats and recommendations

- **Only evaluated rows carry numbers.** Rain-affected matches, no-results and innings of unknown length have
  NULL win probability, WPA and expected runs, with the reason in `exclusion_reason`.
- **Early seasons (2008-2010) are trained on less history** (460-630 matches), and Elo ratings are less settled
  then (see `model_team_elo.rated_before`).
- **Win probability added** credits the whole swing of a ball to the striker; the bowler and fielders are not
  separated out. It does not include the innings break.
- **Known gaps:**
  - recalibration;
  - as-of player features;
  - T20 versions.

<!-- cricket-sim card stamp: build_fingerprint=38eb3e0f0d008218 model_fingerprint=11d528f2bf4769a4 innings_1_brier=0.1892 innings_2_brier=0.1197 -->
