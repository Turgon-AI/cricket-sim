# Combined version against base, 2026-11-19

Follows REPORT.md in this folder, which suggested this combination. It does not replace it.

## Verdict

The combined version beats base everywhere it was expected to and is worse nowhere.

- T20I win log loss: 0.659 to 0.648, better by 0.012 (90% band 0.003 to 0.021, noise 0.003). Real.
- ODI totals: CRPS 37.68 to 37.20 runs, better by 0.48 (band 0.05 to 0.93, noise 0.10). Real. The 80% range now
  holds 77% of actual totals instead of 67%.
- T20I and league ranges: the 80% range holds 79% and 78% instead of 77% and 76%. Real, small.
- League and ODI win log loss: no change (bands cross zero). Total CRPS for T20I and league: no change.

## What the combined version is

- Ball model with the in-play inputs: runs and boundaries in the last six balls, balls the striker has faced.
- T20 pitch factor 0.025, as tuned on 2018 in the earlier run. ODI pitch factor off. No per-innings factor.
- ODI first-innings totals stretched around their mean after the sim. The stretch is fitted each year on earlier
  years only (about 1.16 for 2019, 1.21 for 2026). T20 totals are not stretched.
- Same walk-forward, matches, plays and seeds as REPORT.md: 576 T20Is, 3,066 league matches and 732 ODIs, all
  2019 to 2026.

Repeat check: the run produced plays byte for byte the same as the earlier runs (T20 the same as recent_spread, ODI
the same as recent; 33 of 33 files). The simulator repeats exactly from its seeds.

## Results, combined against base

| Format | Score | Base | Combined | Change (90% band) | Noise | Call |
|---|---|---|---|---|---|---|
| T20I | win log loss | 0.659 | 0.648 | -0.012 (-0.021 to -0.003) | 0.003 | better |
| T20I | Brier | 0.230 | 0.224 | -0.006 (-0.010 to -0.002) | 0.001 | better |
| T20I | CRPS runs | 18.83 | 18.71 | -0.12 (-0.32 to +0.07) | 0.07 | no change |
| T20I | inside 80% | 77.4% | 79.3% | +1.9 pts (+0.5 to +3.3) | 0.5 | better |
| T20I | inside 50% | 45.3% | 48.4% | +3.1 pts (+1.4 to +5.0) | 0.9 | better |
| League | win log loss | 0.698 | 0.699 | +0.000 (-0.004 to +0.004) | 0.001 | no change |
| League | CRPS runs | 18.43 | 18.38 | -0.05 (-0.13 to +0.03) | 0.04 | no change |
| League | inside 80% | 75.7% | 78.3% | +2.6 pts (+1.9 to +3.4) | 0.3 | better |
| League | inside 50% | 46.3% | 49.8% | +3.5 pts (+2.6 to +4.4) | 0.4 | better |
| ODI | win log loss | 0.638 | 0.638 | -0.000 (-0.011 to +0.010) | 0.003 | no change |
| ODI | CRPS runs | 37.68 | 37.20 | -0.48 (-0.93 to -0.05) | 0.10 | better |
| ODI | inside 80% | 67.5% | 77.2% | +9.7 pts (+7.9 to +11.8) | 0.8 | better |
| ODI | inside 50% | 42.9% | 49.3% | +6.4 pts (+4.6 to +8.3) | 0.5 | better |

Pooled T20 (T20I and league together, 3,642 matches): win log loss 0.692 to 0.691, CRPS 18.49 to 18.43, 80% range
76.0% to 78.5%.

## PIT shape (share of actual totals in each tenth; 10% each when right)

| Format | Lowest tenth | Middle tenths | Highest tenth | Read |
|---|---|---|---|---|
| T20I | 10.2% | 8.0 to 11.8% | 10.4% | flat |
| League | 10.4% | 8.5 to 10.8% | 11.4% | close to flat |
| ODI, before the stretch | 13.5% | 6.7 to 12.0% | 16.1% | too narrow, misses high totals |
| ODI, after the stretch | 10.4% | 8.2 to 11.9% | 12.6% | much flatter, still a little light on high totals |

## Against Elo alone (win log loss)

- T20I: the sim is 0.003 better than Elo (band -0.019 to +0.013). It ties Elo; base trailed by 0.008.
- League: the sim is still 0.007 worse than Elo (band +0.002 to +0.013). This is real and is the open problem.
- ODI: the sim is 0.012 better than Elo (band -0.030 to +0.004). Likely better, but the band crosses zero.

## What is noise

- Monte Carlo noise (from the number of plays) is small: about 0.003 on log loss, 0.04 to 0.10 runs on CRPS, under
  1 point on coverage. The bigger uncertainty is which matches were played, which the bands cover.
- The T20 pitch value 0.025 came from a noisy one-year tune. It helps coverage here, but a better tune might choose
  something a little different.

## Still open

- League win chances trail Elo alone.
- ODI still has slightly too many actual totals in the top tenth (12.6%), so very high ODI totals are a bit under
  covered.
- Nothing is in the build or committed. To make this the default, the changes in scripts/simulate_matches.py
  (in-play inputs, pitch 0.025 for T20) and an ODI stretch fitted on earlier years would need to go into the
  simulator itself.

## Files

All under /Users/srikantheranki/Downloads/cricket-sim-clean/sims/2026-11-18-sim-scoring/

- combined/: plays, matches.csv, run.json
- score-combined/: scores.csv (method "final" is the combined version as used), gains.csv, pit_histogram.csv,
  widening_fits.csv, summary.json
- logs/sims-scoring-combined.log, logs/score-combined.log
