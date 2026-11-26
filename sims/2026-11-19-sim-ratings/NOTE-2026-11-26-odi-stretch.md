# ODI range widening on the ratings version: checked 2026-11-26

REPORT.md (2026-11-19) said the ODI widening "was tuned on the old version" and "may now be slightly too much".
Both guesses were wrong. Checked from the existing scores, no new run.

- **The widening is fitted per version.** `score_sim_backtest.py` fits each year's factor on that version's own
  plays from the years before (`score-final/widening_fits.csv`). The ratings version got smaller factors than
  combined: 1.06 to 1.20 against 1.14 to 1.24.
- **The ratings ranges are still a little narrow, not too wide.** With the widening, 76.2% of actual totals fall
  inside the 80% range and 46.0% inside the 50% range (combined: 77.2% and 49.3%). Both sit below target, so the
  inside-50% drop means narrower, not wider.
- **Conformal does a bit better on ratings.** Re-reading the plays at the levels earlier years' PIT values give:
  CRPS 37.00 runs (widening 37.35, combined's final 37.20), 48.0% inside 50%, 77.2% inside 80%. The gap against
  combined is inside the 90% band of the earlier comparison (about +/-0.85 runs), so it is not a proven gain.

Numbers: `score-final/scores.csv`, ODI rows, methods raw, scaled, conformal and final.

Verdict: nothing to fix. If ODI ranges matter later, try conformal as the ODI final method and score it on a
new run.
