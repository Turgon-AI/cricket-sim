# How the models are validated

Decided 2026-10-15, before any model is built, so the test can't be bent to fit the results.

## What we will use

1. **Walk-forward forecasting.** Train only on matches before date D, then forecast every ball of the matches after D. Roll D forward in steps through the years. Nothing from the future is ever visible, and player and venue features are computed as of each match date (see the as-of feature rule below).
2. **Proper scoring rules** on every held-out ball and match:
   - **Ball level:** log loss and Brier score over the outcome categories (dot, 1, 2, 3, 4, 6, wicket, extras).
   - **Match level:** the Brier score of the win probability at fixed checkpoints (start, halfway, final 5 overs).
3. **Calibration.** Of all the times the model said 30%, roughly 30% should have happened. This is checked with reliability curves, by format and by era.
4. **Baselines the model must beat on the same held-out data:**
   - format and phase averages;
   - Cricsheet-only player and venue averages (no tracking);
   - Cricinfo's own published win probability, wherever it exists.

## What was actually run (recorded 2026-10-23, after model run `0f4e92549d533de9`)

The plan above is left as written on 2026-10-15. This section records where the build kept to it and where it
did not, so nobody has to read the code to find out.

| planned | status | where |
|---|---|---|
| walk-forward, train only on earlier seasons | done, with a run-time guard per fold; since 2026-10-23 a locked holdout runs beside it (one fit on clean seasons before 2023, scored once on 2023-2026, scope `holdout_2023-2026`), and it agrees with the folds on those seasons within 0.001 | `models/train.py`, `report_model_folds`, `report_model_scores` |
| ball-level log loss and Brier over eight outcome classes | done; since 2026-10-23 every score carries a 95% interval from 1,000 bootstrap resamples of whole matches, and `report_model_paired` gives the model-minus-baseline difference with its interval and a sign test over seasons (the plan asked for neither; a reviewer would) | `report_model_scores`, `report_model_paired` |
| match-level Brier at fixed checkpoints | done, at the first ball, the first ball of the chase and 30 balls left in the chase (the plan said start, halfway, final 5 overs) | `report_model_checkpoints` |
| calibration by era | done; by format is moot, the models are ODI-only | `report_model_calibration` |
| baseline: format and phase averages | done, as the base rate and the by-innings-phase-wickets class frequencies for ball outcomes, and a logistic regression on the raw match state for win probability | `report_model_scores`, models `base_rate`, `phase_average`, `state_logistic` |
| baseline: Cricsheet-only player and venue averages | done for the ball outcome (added 2026-10-23): the phase frequencies scaled by the striker's, the bowler's and the venue's observed-over-expected ratio for each class on the training seasons' balls, each ratio shrunk towards 1 with 200 pseudo-balls fixed before scoring. Counting only, on prior clean seasons, Cricsheet ids only. It scores between the phase average and the model: log loss 1.3078 and Brier 0.6356 against 1.3125 / 0.6374 for the phase average and 1.2996 / 0.6297 for the model. Not built for win probability: the published model has no player or venue input, so a player baseline there would compare different inputs rather than a weaker model on the same ones. | `report_model_scores`, model `player_venue_average`; `models/train.py player_venue_average` |
| baseline: Cricinfo's published win probability | done, internally only. Cricinfo's number is the previous ball's published value, so it is pre-ball. Its perspective is undocumented and is inferred per innings by picking the orientation with the lower Brier against the result, a one-bit choice made on the scored balls and in Cricinfo's favour. On the 41,619 first-innings and 35,664 chase balls where both exist, the Brier scores are 0.189 vs 0.210 (innings 1, model better) and 0.119 vs 0.118 (innings 2, Cricinfo better). The table is Cricinfo-derived and never exported. | `report_model_vs_cricinfo_internal` |

Two things the plan did not foresee are also in force: rain-affected matches, no-results and innings of unknown
length get no prediction at all (the plan only excluded them from training and scoring), and the model family was
chosen on the same walk-forward folds that are reported. The locked holdout checks those folds; it is not a fresh
model choice (see the status note at the top of docs/MODEL_CARD.md).

## Data rules that make this test honest

- **Pre-ball inputs only:** columns with timing class `pre_ball` (in `fact_delivery`, and every column of `fact_player_match_form`), plus Cricinfo's `cricinfo_ball_pre` for internal work. Cricinfo's line, length, shot and control labels are recorded after the outcome. They are never decision-model inputs. They are outcome-model inputs only after correction against Hawk-Eye.
- **Excluded from decision-model training and validation:** rain-revised chases (`target_status = 'revised_timing_unknown'`) and first innings of unknown length (`innings_length_status = 'reduced_timing_unknown'`).
- **Excluded from outcome training:** awarded matches (`is_awarded`).
- **As-of feature rule:** any player or venue feature for a match on date D uses only data from before D (player features: windows end the day before; venue context: matches up to D, see the next rule). Whole-career and current-day attributes are forbidden as features. Player features are `gold.fact_player_match_form`, whose windows end the day before the match; `tests/test_leakage.py` proves it by deleting every later match and checking that no feature moves. Whole-snapshot career totals live only in `report_player_careers`.
- **As-of venue context (`venue_context: asof`, in force for this build):** a match's venue name, city, search strings, home team, geocodes and weather placement use only matches up to that match's date. A venue gets a new version when one of those inputs changes (590 venues, 1,296 versions), and each match uses the version valid on its start date. The contracts on `venue_versions`, `match_venues` and the weather tables check this on every build, and `tests/test_venue_asof.py` rebuilds the context with every later match removed and requires every earlier venue and weather row to stay identical. The venue part of the as-of promise above is therefore true for this build. Gold publishes the in-scope versions as `dim_venue_version` and names each match's version in `dim_match.venue_version`; the model's home status takes the latest version whose inputs end before the match date. `dim_venue` is a latest-version view for browsing and is not a feature source.
