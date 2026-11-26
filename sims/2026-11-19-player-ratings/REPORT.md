# Player ratings in the ball model: pass/fail test, 2026-11-19

**Verdict: pass, in every format and every year.** Knowing the batter on strike, the bowler and the ground makes
the ball model better on matches it has not seen. The gain is about as big as everything the match state (score,
wickets, overs, target) adds over a plain phase average. The ratings also make the simulator's team-average
inputs unneeded.

Script: `scripts/test_player_ratings.py` (prototype, outside the build). Commit 5f9274a, build ca15e28ebb4af18d,
33 minutes. Numbers: `summary.json`, `per_match.csv`, `checks.json`; rating tables `ratings_*.parquet`.

## What was tested

- **Ratings.** For each batter, bowler and ground: how often each outcome (dot, 1, 2, 3, 4, 6, wicket, extra)
  happened, compared with what a typical player gets in the same innings, overs and wickets. Built only from
  earlier dates, recent seasons count more (half-life 3 years), and pulled toward average (200 balls, 500 for
  wickets) so a newcomer starts at average. T20 ratings span T20Is, the IPL and the nine leagues. All settings
  were fixed before scoring.
- **Test.** Each year 2019 to 2026 is scored by models trained on the ten years before it. Three versions of the
  simulator's ball model: today's inputs, today's plus ratings, and ratings without the team averages.
- **Score.** Log loss per ball (lower is better), with a 90% band from resampling whole matches.

## Results

| test balls | matches | today's model | with ratings | change | 90% band |
|---|---|---|---|---|---|
| T20I | 577 | 1.5674 | 1.5551 | -0.0123 | -0.0134 to -0.0113 |
| leagues | 3,070 | 1.5568 | 1.5428 | -0.0140 | -0.0145 to -0.0136 |
| ODI | 744 | 1.2948 | 1.2855 | -0.0093 | -0.0099 to -0.0087 |

- Better in all 8 years in every format (smallest yearly gain: T20I 2021 -0.0096, ODI 2024 -0.0075).
- For scale: in the model card, going from a phase average to the full match-state model gains about 0.013 on
  ODI balls. The ratings add nearly as much again.
- Ratings without the six team-average inputs score the same (T20I -0.0122, leagues -0.0143, ODI -0.0094). So in
  the simulator the ratings can replace the team averages.

## Leak check

Ratings for 40 random batters and dates, recomputed by hand from the raw balls using strictly earlier dates,
match the script to 1e-13. The 1,926 debut rows sit exactly at average.

## The two checks from the October spec

- **Grounds' six rates show up.** A ground's six rating on 1 January predicts its six rate that year:
  correlation 0.58 over 553 ground-years (T20, 2016 to 2026, grounds with 600+ balls that year), slope 0.8.
- **Dhoni's death-overs decline: weak at best.** Runs against expected in ODI overs 40 to 49 were 1.06 to 1.34
  in 2011 to 2013, then about 1.0 from 2014 to 2019, with one bad year (2018: 0.73 on 117 balls) and a normal
  2019 (1.03 on 273 balls). With 100 to 300 balls a year, single years are noisy. The data shows him going from
  above average to average, not a collapse.

## Not covered

- No league strength term: a ball in a weak league counts the same as an IPL ball. Fixing it could add more.
- No bowler type (pace or spin): not in the data.
- This is the ball model only. Whether it improves match win chances and score ranges is step 3 (put ratings in
  the simulator, with real batting order and bowlers) and step 4 (score it).
