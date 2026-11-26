# Cricket match forecasting: state of the art against cricket-sim (2026-11-18)

Verdict: the simulator's shape (a model of each ball, played out many times) is the same shape Opta, CricViz and
the main academic simulator use, and nobody has shown that a neural or transformer ball model does better. The
gaps are elsewhere, in four places:
1. The simulator does not know which batter and bowler are on each ball.
2. It treats balls as independent of each other, so its score ranges come out too narrow.
3. It has no betting-market benchmark to compare against.
4. It tests on too few matches to tell versions apart.
The pitch is the one thing the pros have and open data lacks. They learn it during play, not before the match.

How this was made: four research agents searched the web on 2026-11-18 (simulators, player and league ratings,
benchmarks and calibration, pitch data). I then checked these key sources myself: arXiv 2608.14696, arXiv
2608.18020 and the ICC pitch ratings file. Each claim is marked "read" (someone read the source) or "snippet"
(seen only in search results). The comparisons to our numbers come from `scripts/simulate_matches.py` and the
memory notes for 2026-11-16 to 11-18.

## 1. Where cricket-sim already matches the state of the art

- **The shape.** It has a per-ball outcome model, Monte Carlo play-out, Elo, venue and home terms.
  - Opta: a neural net gives next-ball chances, then Monte Carlo plays out the match (read,
    https://theanalyst.com/2022/06/introducing-cricket-simulation-models/).
  - Davis, Perera, Swartz (2015) built a T20 simulator with the same 8-outcome design (read,
    https://www.sfu.ca/~tswartz/papers/t20sim.pdf).
  - CricViz WinViz is also a simulation (read, swarajyamag.com interview).
- **Ball model type.** No paper or company shows a transformer or "foundation model" on ball sequences
  beating boosted trees. The newest neural cricket work found was LSTM win-probability models from 2020-21.
  Our HistGradientBoosting choice is fine.
- **Testing.** Walk-forward tests, CRPS, Brier, the luck share and the variance split are more careful than
  anything published. No company publishes accuracy numbers. The only market comparison in a paper covers
  15 games (Swartz).

## 2. Gaps, ranked by evidence and likely gain

### Gap 1: the simulator does not know who is batting and bowling

Today each ball sees the top-7 batting and top-6 bowling averages. It does not see the batter on strike, the
bowler, or the batting order. `src/cricket_sim/models/train.py` already has batter, bowler and venue ratios
pulled toward the average by 200 balls (`player_venue_baseline`), but the simulator does not use them.

What the state of the art does:
- **Swartz.** Each batter's outcome chances are pulled toward the average batter with a prior worth 60 balls,
  and results barely move between 50 and 80. Matches are weighted by recency at 0.88 per match. A batter
  against a bowler is batter plus bowler minus average, so sparse pairs are never a problem. (read)
- **Opta.** Player and venue strength come from hierarchical Bayesian ratings (batter, bowler and venue, for
  runs per ball and balls per out), adjusted for opponent and situation. (read)
- **One model across all leagues.** Fit batter and bowler ratings on every competition at once, with a term
  for each competition. League strength then comes out of the fit, because many players play in more than
  one league. Hockey and football league equivalents are built this way. A cricket mixed model on 511 batters
  puts the IPL about 2.5% above the top T20I group, and the BPL and LPL below the weaker T20I sides. (read,
  https://arnavj.substack.com/p/the-real-t20-batting-performance; https://thibaudchatel.substack.com/p/rethinking-nhle)
- **Form.** It adds little once career numbers are properly pulled toward the average. For Test batters a
  slowly changing ability model beat recent form, and recent form alone was the worst predictor (Stevenson
  and Brewer, read).
- **How many balls before a number means something.** A T20 batter's run rate needs about 175 to 200 balls
  before it is mostly signal (cricketsims, read). Dismissal rate needs several times more, but no published
  number was found.
- **Why this should help.** In football, Elo plus player ratings beat either one alone (Arntzen and Hvattum,
  read). This also explains why our raw XI averages added nothing: they had no pull toward the average, no
  adjustment for opponent or league, and overlapped with Elo.

### Gap 2: score ranges too narrow (72-78% inside an 80% range)

- **arXiv 2608.14696** (Mishra, 9 Aug 2026, abstract checked by me). A T20 chase model can get each ball's
  chances right and still give wrong win chances, because it treats balls as independent. Runs cluster over
  3 to 5 balls, and conditions for the whole innings explain only about 18% of that clustering. Replaying
  real runs of balls in blocks closed 26% of the gap.
- **Our pitch factor is a match-level effect, so it can fix at most the smaller part.** The 2025 tuning
  picked a small spread (sd 0.025) because bigger spreads overshot.
- **What to try:**
  - Inputs for the last 6 balls' runs and boundaries.
  - How many balls the striker has faced (a set batter). Neither is in OUTCOME_FEATURES today.
  - A second random effect, per innings for the batting side, next to the pitch factor.
- **The standard fix after the simulation is to widen the range.** Two ways:
  - Scale the simulated variance by a factor fitted to minimise CRPS on a rolling window (EMOS, Gneiting
    2005, snippet).
  - Or conformal intervals on the simulated quantiles, which guarantee coverage on average when years are
    comparable (Romano et al., read).
- **Check with a PIT histogram per format.** Too-narrow ranges show as a U shape.

### Gap 3: no market benchmark

- **No published Brier or log loss for cricket betting markets, pre-match, in any format.** Every cricket
  market study found is in-play. Norton, Gray, Faff 2015 found Betfair ODI prices overreact to first-innings
  wickets (snippet).
- **Elsewhere the market is hard to beat.** In football a strong rating model came out 2.8% worse than market
  odds on Brier (read, arXiv 2607.01722). In tennis the market beat all 11 models tested (snippet).
- **One agent's estimate, not a sourced number:** a well-calibrated market with favourites at about 63% gives
  a T20I Brier around 0.21 to 0.22. Our 0.216 may already be close to what is reachable, but we cannot know
  without odds.
- **The pros' answer:** take closing odds, remove the bookmaker margin, score them on the same matches, and
  blend a walk-forward logistic fit of the market and our model. Before building this, check the source and
  its licence: Betfair historical data, or another feed.

### Gap 4: too few test matches to rank versions

- **Arithmetic from the benchmarks agent.** To tell apart two forecasters whose Brier differs by 0.01 needs
  about 780 matches. That is about 10 years of T20Is.
- **One year's Brier has a standard error of about 0.023.** So most year-on-year Brier changes in the memory
  notes are noise. The 0.01 run-to-run noise we saw sits inside that.
- **Log loss should lead, with Brier beside it.** In a football study it told forecasters apart better than
  Brier or RPS (Wheatcroft, read).
- **My suggestion, not from a source:** the T20I model now trains on the leagues, so score it on held-out
  league matches too. That gives thousands of test matches instead of about 75 a year.

## 3. The pitch: what the pros have, and the open substitutes

- **Not open.**
  - Ball-tracking pitch measures: seam, swing, bounce, turn (Hawk-Eye, CricViz PitchViz). All snippet.
  - Opta feeds: sold to media and betting firms, no public price (snippet).
  - Boundary length per match: no open dataset found.
- **What the pros do instead: learn the pitch during play.**
  - Opta starts from venue ratings and updates as the match shows how it is playing (read).
  - WASP sets a pre-match par score by expert judgement (read).
  - This fits our measure: the shared-day part (15-21% in T20Is) is close to unknowable before the toss.
- **ICC pitch and outfield ratings exist per match. I checked the 2022-2025 file.**
  - It has 2,020 rows: date, series, match, venue, pitch rating, outfield rating. It covers Tests, ODIs and
    T20Is.
  - Other files cover May 2019 to Dec 2021 and 2026 up to September.
  - The match referee rates the pitch after the match, so it is hindsight.
  - The page says "All rights reserved", so it is internal use only unless the ICC agrees.
  - Most T20I pitches are rated "Very Good" (536 of 920 by the agent's count), so it may not separate much.
- **Open substitutes worth a test:**
  1. Join ICC ratings to gold by date and venue and check whether they explain the shared-day residual. This
     is a test of the idea, not a pre-match input.
  2. Use the venue's last rating in the same series as a pre-match input.
  3. Inputs known after the toss: the toss call against the venue's usual call, and the number of spinners
     picked against the team's usual. Bowling style is held but is internal only. This only counts if the
     forecast is issued after the toss.
  4. An in-play update of the day effect from the first 6 overs. This turns the sim into a live forecaster,
     which is a different product question.
- **Toss effect sizes** (Sood and Willis, 41,060 matches, read): 5.9 points of win rate in day-night ODIs, 2.9
  in day ODIs, 1.3 in T20s. On dew there are only anecdotes, no controlled study.

## 4. Not worth it now

- Transformers or embeddings for the ball model: no evidence. One blog claimed a jump from 0.74 to 0.94, but
  it most likely leaked balls from the same match into training and test.
- More weather variables: already tested, no gain.
- Individual batter-against-bowler pair terms: too sparse. Use bowler type against batter hand (pace or spin,
  arm, batter's hand), pulled hard toward the average (about 1,250 balls in arXiv 2608.18020, a figure the
  agent read in the HTML).

## 5. Suggested order

| Step | What | Why first | Effort |
|---|---|---|---|
| 1 | Score on log loss plus Brier, paired bootstrap pooled over years, league matches as extra tests, fixed seeds | Without it, steps 2 to 5 cannot be told apart from noise | Small |
| 2 | Widen score ranges after the sim (variance scale by CRPS, or conformal) plus a PIT check | Fixes 72-78% coverage directly and cheaply | Small |
| 3 | Last-6-ball and balls-faced inputs, plus a per-innings batting-side effect | Goes at the cause of narrow ranges and overconfident win chances | Medium |
| 4 | Player ratings across all leagues: batter and bowler pulled toward average, league terms, real batting order in the sim | The one thing every serious simulator has and ours lacks; also gives league strength | Large |
| 5 | Market benchmark from closing odds, then a blend | The only way to know if 0.216 is good; needs a licensed source | Medium, source TBD |
| 6 | ICC rating check and inputs known after the toss | Cheapest test of the pitch idea | Small |

Rough ceiling, from the benchmarks agent's judgement and not a sourced figure: about 20-25% of first-innings
spread explained before the match for ODIs, and 25-30% for T20Is. T20I Brier maybe 0.01 better than today.

## Not read or not verified

- Asif and McHale 2016 full text.
- ESPNcricinfo Forecaster method (403).
- CricViz's own Impact method, and how Hawk-Eye data is licensed.
- The WASP ground-conditions PDF.
- The Brier numbers inside arXiv 2608.14696 (the agent saw 0.164 Markov against 0.140 logistic in the HTML).
  I could confirm only the abstract.
- How the ICC file splits between men's and women's matches.
