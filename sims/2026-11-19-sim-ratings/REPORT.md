# Player ratings in the simulator: full backtest, 2026-11-19

Step 4 of the plan. Follows `sims/2026-11-19-player-ratings/REPORT.md` (the ball-model pass/fail test).

## Verdict

**Keep it for T20. ODI is a draw.**

- **Leagues: better on both win chances and totals.** Win log loss 0.699 to 0.687. Before this, the simulator was
  worse than plain Elo on league matches; now it is level with Elo.
- **T20I: totals clearly better, win chances unchanged.** The first-innings total is off by less (CRPS 18.71 to
  18.13 runs). The share of the spread in totals the simulator explains goes from 12% to 18%.
- **ODI: no change on win chances or totals.** One range check is slightly worse (below).

## What was run

- Version `ratings`: today's best version (`combined`) plus the step 3 changes:
  - the real batting order;
  - the real bowler each over;
  - each ball sees the striker's, the bowler's and the ground's ratings in place of the six team-average inputs.
- Same matches, years, plays per match and seeds as `combined`. Each year 2016 to 2026 uses models trained on the
  years before it. Scored on 2019 to 2026: 576 T20Is, 3,066 league matches and 732 ODIs.
- Both versions are scored as they would be used: ODI first-innings totals are stretched after the sim, and T20
  totals are left as they come out.
- Run time: 101 minutes.

## Results, ratings against combined

Lower is better for log loss and CRPS. "Inside 80%" should sit near 80%, "inside 50%" near 50%. The 90% band
comes from resampling whole matches. Noise is how much the score moves from the random plays alone.

| Format | Score | Combined | Ratings | Change (90% band) | Noise | Call |
|---|---|---|---|---|---|---|
| T20I | win log loss | 0.648 | 0.651 | +0.004 (-0.013 to +0.021) | 0.003 | no change |
| T20I | CRPS runs | 18.71 | 18.13 | -0.58 (-0.95 to -0.17) | 0.06 | better |
| T20I | explained share | 12.3% | 18.2% | +5.8 pts (+2.0 to +9.5) | 0.5 | better |
| T20I | inside 80% | 79.3% | 80.0% | +0.7 pts (-1.2 to +2.6) | 0.7 | no change |
| T20I | inside 50% | 48.4% | 52.6% | +4.2 pts (+1.6 to +6.8) | 0.9 | a bit wide now |
| League | win log loss | 0.699 | 0.687 | -0.012 (-0.018 to -0.006) | 0.002 | better |
| League | CRPS runs | 18.38 | 17.82 | -0.55 (-0.71 to -0.40) | 0.04 | better |
| League | explained share | 1.5% | 7.3% | +5.8 pts (+4.4 to +7.3) | 0.3 | better |
| League | inside 80% | 78.3% | 80.1% | +1.7 pts (+0.9 to +2.6) | 0.4 | better |
| ODI | win log loss | 0.638 | 0.643 | +0.005 (-0.012 to +0.023) | 0.004 | no change |
| ODI | CRPS runs | 37.20 | 37.35 | +0.14 (-0.71 to +1.00) | 0.10 | no change |
| ODI | inside 80% | 77.2% | 76.2% | -1.0 pts (-3.0 to +1.2) | 0.7 | no change |
| ODI | inside 50% | 49.3% | 46.0% | -3.3 pts (-5.7 to -0.8) | 0.6 | a bit worse |

Against plain Elo, win log loss (negative means the simulator is better):

| Format | Combined vs Elo | Ratings vs Elo |
|---|---|---|
| T20I | -0.003 (-0.018 to +0.012) | +0.000 (-0.018 to +0.018) |
| League | +0.007 (+0.001 to +0.013), worse than Elo | -0.005 (-0.011 to +0.002), level |
| ODI | -0.012 (-0.031 to +0.005) | -0.007 (-0.021 to +0.010) |

## What this means

- Player ratings mostly help the **totals**, which is what the ball model predicts directly. They help **win
  chances** only in the leagues. There, line-ups change a lot from match to match, so team-level history knows
  least about who is playing.
- For T20I and ODI win chances, the bands are about 0.035 wide. A gain the size the ball-level test suggests would
  be too small to see in 576 or 732 matches. "No change" here means no difference we can detect. It does not
  prove the ratings do nothing.
- ODI: the stretch applied to ODI totals after the sim was tuned on the old version. With ratings the raw ODI
  ranges are already a little wider (raw inside 80% goes from 70.6% to 72.3%). So the stretch may now be slightly
  too much, which would explain the inside-50% drop. Not checked.

## Changes made for this run (not committed)

- `scripts/simulate_matches.py`, `scripts/sim_backtest.py`, `scripts/player_ratings.py`: step 3.
- `scripts/score_sim_backtest.py`: now also compares a version's "final" against the baseline's "final", so that
  ODI is scored stretched against stretched. The older comparison, final against the baseline's raw run, is
  still printed.

## Files

- Plays: `ratings/`. `combined` is a link to `sims/2026-11-18-sim-scoring/combined`.
- Scores: `score-final/` (use this one). `score/` is the same run without the final-against-final rows.
- Log: `run.log`.
