# Simulator scoring and ranges, 2026-11-19

## Verdict

- The new in-play inputs (runs and boundaries in the last six balls, balls the striker has faced) make the T20I win
  chance better: log loss 0.659 to 0.649, a gain of 0.011 (90% band 0.002 to 0.019). Monte Carlo noise is 0.003, so
  this is real. On league matches and ODIs win log loss does not move. It got worse nowhere.
- Totals barely move with the new inputs. Every CRPS change is inside its band.
- Ranges: the best fix for ODI is to stretch the spread after the sim, fitted on earlier years. The 80% range then
  holds 77% of actual totals instead of 67%, and CRPS drops 0.42 runs (band 0.25 to 0.59). For T20, the small pitch
  factor (0.025) gets the 80% range to 79% without hurting anything. Stretching T20I after the sim overshoots.
- The per-innings batting-side factor did not help. Tuning on 2018 set it to zero for both formats.

## What was run

- Walk forward: for each year the ball model is fitted on the ten years before 1 January. Years 2016 to 2018 are
  played only so the range fixes have earlier years to learn from. Everything below is scored on 2019 to 2026.
- Matches: 576 T20Is, 3,066 held-out IPL and league matches, 732 ODIs. Plays per match: T20I 400, league 200, ODI 300.
- Four versions:
  - base: the simulator as it is (league balls in the T20 model, pitch factor off)
  - t20i_only: T20 model trained on T20I balls only
  - recent: base plus the in-play inputs
  - recent_spread: recent plus a pitch factor and a per-innings factor, tuned on 2018 by lowest first-innings CRPS.
    Chosen: T20 pitch 0.025, ODI pitch 0.075, per-innings 0 for both.
- Scores: win log loss (main, chances clipped to 1% and 99%), Brier, CRPS of the first-innings total (runs, lower is
  better), share of actual totals inside the 50% and 80% ranges, and the PIT histogram (where the actual total falls
  among the plays; flat at 10% per bin when the range is right).
- Bands are 90% paired bootstrap bands over matches, all years pooled, 1,000 draws. Noise is Monte Carlo noise: the
  plays of each match split into two halves, 20 times.
- Prototype scripts outside the build: scripts/simulate_matches.py, scripts/sim_backtest.py,
  scripts/score_sim_backtest.py. Build ca15e28ebb4af18d, commit 3bb1947 with an uncommitted tree.

## Win log loss (lower is better)

| | T20I | League | ODI |
|---|---|---|---|
| Elo alone | 0.651 | 0.692 | 0.650 |
| base | 0.659 | 0.698 | 0.638 |
| t20i_only | 0.684 | | |
| recent | 0.649 | 0.697 | 0.638 |
| recent_spread | 0.648 | 0.699 | 0.639 |

Against base, with bands:

| Change | T20I | League | ODI |
|---|---|---|---|
| recent | -0.011 (-0.019 to -0.002), real | -0.001 (-0.005 to +0.002), noise | -0.000 (-0.010 to +0.010), noise |
| recent_spread | -0.012 (-0.020 to -0.003), real | +0.000 (-0.004 to +0.004), noise | +0.001 (-0.010 to +0.011), noise |
| t20i_only | +0.025 (-0.001 to +0.050), just short of real | | |

Against Elo alone:
- T20I: base was 0.008 worse than Elo; recent is 0.002 better. Neither gap is outside its band, so the sim now ties
  Elo on T20I instead of trailing it.
- League: the sim is still worse than Elo, by 0.005 to 0.007. For base and recent_spread the band is above zero.
- ODI: the sim is about 0.012 better than Elo, but the band crosses zero.

Brier tells the same story. recent against base on T20I is -0.005 (band -0.009 to -0.001).

## Totals: CRPS and ranges, raw sim

| Version | Format | CRPS runs | Inside 50% | Inside 80% |
|---|---|---|---|---|
| base | T20I | 18.83 | 45% | 77.4% |
| recent | T20I | 18.74 | 46% | 77.3% |
| recent_spread | T20I | 18.71 | 48% | 79.3% |
| base | league | 18.43 | 46% | 75.7% |
| recent | league | 18.39 | 47% | 75.7% |
| recent_spread | league | 18.38 | 50% | 78.3% |
| base | ODI | 37.68 | 43% | 67.5% |
| recent | ODI | 37.50 | 42% | 70.6% |
| recent_spread | ODI | 37.79 | 58% | 84.6% |

What is real against base:
- recent: no CRPS change is outside its band. ODI 80% coverage rises 3 points (band +1.5 to +4.9).
- recent_spread: T20I 80% coverage +1.9 points (band +0.5 to +3.5) and league +2.6 (band +1.9 to +3.4). CRPS gains
  are inside their bands.
- recent_spread on ODI overshoots. The 80% range holds 85% and the 50% range holds 58%, so the ranges are too wide.
  CRPS is 0.11 worse (band -0.37 to +0.60, so noise). The 0.075 pitch pick came from 91 ODIs in 2018 with 100 plays,
  and the tuning grid jumped around, so that pick is mostly noise.
- t20i_only is clearly worse on totals: CRPS +1.40 runs (band +0.94 to +1.93) and 80% coverage 66% against 77%.

## PIT shape (share of actual totals in each tenth; 10% each when right)

| Version | Format | Lowest tenth | Highest tenth | Shape |
|---|---|---|---|---|
| base | T20I | 11.6% | 11.1% | close to flat, ends a little heavy |
| base | ODI | 13.9% | 18.6% | too narrow, misses high totals most |
| t20i_only | T20I | 15.8% | 18.1% | too narrow at both ends |
| recent | ODI | 13.5% | 16.1% | a little better than base, same problem |
| recent_spread | T20I | 10.2% | 10.4% | flat |
| recent_spread | league | 10.4% | 11.4% | close to flat |
| recent_spread | ODI | 10.2% | 5.3% | too wide on the high side, middle bins 13% |
| base ODI, stretched after the sim | ODI | 10.0% | 12.8% | much flatter |

## Range widening after the sim (fitted on earlier years only), base version

| Format | Method | CRPS change, runs | Inside 80% |
|---|---|---|---|
| ODI | stretch | -0.42 (-0.59 to -0.25), real | 67% to 77% |
| ODI | stretch plus shift | -0.42 (-0.57 to -0.24), real | 67% to 77% |
| ODI | conformal | -0.42 (-0.63 to -0.21), real | 67% to 77% |
| T20I | stretch | +0.03 (-0.05 to +0.12), noise | 77% to 85%, overshoots |
| T20I | stretch plus shift | +0.18 (+0.07 to +0.30), worse | 77% to 84% |
| T20I | conformal | +0.25 (+0.12 to +0.40), worse | 77% to 83% |
| League | stretch | -0.03 (-0.05 to -0.02), real but tiny | 76% to 79% |
| League | conformal | -0.04 (-0.07 to +0.00), noise | 76% to 80% |

- For ODI the three methods do the same thing. The fitted stretch is 1.19 in 2019 and 1.25 in 2026, and the fitted
  shift is close to zero, so ODI plays are too narrow but centred right. Use the plain stretch, the simplest one.
- On the recent version the ODI stretch gives the best ODI result of all: CRPS 37.20, 80% coverage 77%.
- For T20I, earlier years were too narrow (2019 fit 1.32, from 145 matches), but recent years are not, so the stretch
  overshoots. Shift and conformal add noise.
- Widening never changes win log loss; it only touches first-innings totals.

## What is noise

- Monte Carlo noise: win log loss 0.002 to 0.004, Brier about 0.001, CRPS 0.03 to 0.15 runs, 80% coverage 0.2 to 0.8
  points. Every gain called real above is several times its noise and its band does not cross zero.
- Bigger than the Monte Carlo noise is the sampling of matches: the bootstrap bands. Most CRPS changes from the new
  inputs sit inside them.
- The 2018 tuning used one year and 100 plays. Its grid was not smooth (T20 CRPS 17.8 to 19.0 with no clear trend
  below 0.1), so the chosen pitch values are rough. The per-innings factor came out at zero in both formats.

## Suggested next step, not done

- Keep the in-play inputs in the T20 model, keep T20 pitch 0.025, turn the ODI pitch factor off, and stretch ODI
  totals after the sim with a factor fitted on earlier years. This combination was not run as one version, so it
  needs one more run to confirm.
- League win chances are still worse than Elo alone. That is the next thing to look at for leagues.

## Files

All under /Users/srikantheranki/Downloads/cricket-sim-clean/sims/2026-11-18-sim-scoring/

- base/, t20i_only/, recent/, recent_spread/: plays per match, matches.csv, run.json (tuning grid inside)
- score-all/: scores.csv, gains.csv, pit_histogram.csv, widening_fits.csv, summary.json (this report's numbers)
- score-base-only/, score-base-vs-t20i-only/: earlier scoring passes, kept
- logs/: run and scoring logs
