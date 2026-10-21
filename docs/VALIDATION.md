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

## Data rules that make this test honest

- **Pre-ball inputs only:** columns with timing class `pre_ball` (in `fact_delivery`, and every column of `fact_player_match_form`), plus Cricinfo's `cricinfo_ball_pre` for internal work. Cricinfo's line, length, shot and control labels are recorded after the outcome. They are never decision-model inputs. They are outcome-model inputs only after correction against Hawk-Eye.
- **Excluded from decision-model training and validation:** rain-revised chases (`target_status = 'revised_timing_unknown'`) and first innings of unknown length (`innings_length_status = 'reduced_timing_unknown'`).
- **Excluded from outcome training:** awarded matches (`is_awarded`).
- **As-of feature rule:** any player or venue feature for a match on date D uses only data from before D (player features: windows end the day before; venue context: matches up to D, see the next rule). Whole-career and current-day attributes are forbidden as features. Player features are `gold.fact_player_match_form`, whose windows end the day before the match; `tests/test_leakage.py` proves it by deleting every later match and checking that no feature moves. Whole-snapshot career totals live only in `report_player_careers`.
- **As-of venue context (`venue_context: asof`, in force for this build):** a match's venue name, city, search strings, home team, geocodes and weather placement use only matches up to that match's date. A venue gets a new version when one of those inputs changes (590 venues, 1,296 versions), and each match uses the version valid on its start date. The contracts on `venue_versions`, `match_venues` and the weather tables check this on every build, and `tests/test_venue_asof.py` rebuilds the context with every later match removed and requires every earlier venue and weather row to stay identical. The venue part of the as-of promise above is therefore true for this build. `dim_venue` is a latest-version view for browsing and is not a feature source.
