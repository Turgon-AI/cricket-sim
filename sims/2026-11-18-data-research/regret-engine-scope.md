# Is the original "regret engine" scope worth it? (2026-11-18)

Verdict: yes as an idea, no as written. The full scope has four models, tree search, and a regret number per
ball with an interval. That headline output cannot be supported by the data: one ball's choice moves win
probability far less than one ball's luck does, the delivery and shot labels are written after the outcome,
and searching over a learned model mostly finds the model's own errors. A smaller version is worth building.
It looks at decisions made once per over or once per innings, such as who bowls the 19th over or who bats at 5.
Those decisions are logged in Cricsheet with no label problem, and they are where other sports found real
mistakes. As a testbed for the economic simulation, cricket is weak.

How this was made: three research agents (prior work, feasibility, value and the testbed idea) on 2026-11-18.
I checked the key papers myself (Brill et al. abstract, arXiv 2604.13861) and ran my own measurement on the
repo's data (section 2). "read" means someone read the source; "snippet" means it was seen only in search results.

## 1. Has anyone built this?

Not as one piece, but every piece exists somewhere:
- **Ball-by-ball cricket simulators** with player ratings pulled toward the average: Swartz (SFU), Opta since
  2022, and Ganesh, arXiv 2604.13861 (2026, submitted to JQAS). Ganesh searches IPL batting orders and bowling
  plans and claims +4 to +5 points of win probability, but has no backtest. One of its bowling plans gives three
  death overs to a bowler with no IPL data, so his profile is just the league average. (read)
- **Valuing a decision by the win probability it lost** is a known idea under other names: NFL fourth-down
  bots, chess engines' centipawn loss, "regret" in game theory. The framing is not new. (read / general knowledge)
- **Changing one player's own policy and simulating**: Sandholtz and Bornn in basketball changed a player's shot
  choices and simulated 300 seasons. They checked the simulator against real counts first, and called it
  "exploratory" because the defence does not react. (read)
- **Not found anywhere public:** a cricket model of which delivery a bowler picks and which shot a batter plays,
  a captain-response model, or any counterfactual claim in cricket that was checked against real outcomes.

The new part is the combination and the level of detail, not the method.

## 2. The core problem: the signal is small, the noise is large

My measurement on the repo's data, all IPL death overs (16 to 20) with Hawk-Eye, 31,174 legal balls, chasing
side's win probability from Cricinfo:

| Where the ball pitched | Balls | Runs per ball | Wicket % | Average win prob change | Spread of change per ball |
|---|---|---|---|---|---|
| under 1.5 m (yorker or full toss) | 3,617 | 1.66 | 6.5 | -0.02 | 7.5 |
| 1.5 to 4 m (full) | 6,165 | 1.39 | 7.3 | -0.30 | 6.9 |
| 4 to 7 m (good length) | 10,228 | 1.70 | 10.3 | +0.29 | 6.9 |
| 7 m and more (short) | 11,164 | 1.54 | 8.4 | +0.09 | 6.5 |

(Win probability change in points, from the bowling side's view a negative number is good. Second innings only.)

- **The gap between the best and worst length is about 0.6 points of win probability. One ball's luck
  moves it by about 7 points.**
- These raw numbers are also confounded. A ball in the first row may have been a yorker that worked or one
  that became a full toss, and bowlers pick lengths for reasons the data does not show.
- **My arithmetic:** telling a 0.3-point difference apart needs roughly 8,000 balls per option. That is
  possible pooled over a whole league. It is not possible for one player in one match.
- **Other sports agree.** In NFL fourth downs, with 600,000 plays, the best call is clear on only 48% of
  decisions, and 44% of decisions are worth under 1 point (Brill, Yurko, Wyner, read; their abstract says
  uncertainty is "far greater" than analysts admit). In baseball, swapping the earlier setup pitches inside a
  Transformer moved predicted outcomes much less than swapping the final pitch (Takamido and Nakamoto 2026,
  read abstract). In tennis, top servers already mix their choices close to what game theory predicts, so the
  regret on each serve is near zero (UTS paper, read).

What this means: the Dhoni and Bumrah flagship cases are single moments. Their regret intervals will almost
always cross zero, so the honest answer will usually be "can't tell".

## 3. Feasibility of each part

| Part | Feasible? | Main risk | What makes it work |
|---|---|---|---|
| Ball outcome model with players, venue, state | Yes | Luck is 50 to 60% | Pull players toward the average; intervals from refits that resample whole matches |
| Outcome given line and length | Partly | Cricinfo labels lean toward the result | Use Hawk-Eye as the gold set (1,689 men's ODI, Test and IPL matches, 2011 to 2021) to correct the labels; drop "shot control" |
| Delivery and shot decision model | Partly | Labels written after the ball; hidden intent and field | Coarse choices only: pace or spin, who bowls, length band from Hawk-Eye |
| Captain bowling-change model | Yes | Hidden fitness and pitch information | Who bowls each over is in Cricsheet; check for selection bias, as Romer did for fourth downs |
| Sequence memory (setup balls) | Partly | Needs speed and tracking data | Hawk-Eye has speed for the 1,689 matches; measure whether earlier balls add anything before building it |
| Tree search over the models | Partly, risky | The search finds spots where the model is wrong and too hopeful | Shallow search, penalise uncertainty (MOPO, MOReL), allow only actions seen in similar states |
| Per-ball regret with an interval | No, as a headline | Effects sit inside the noise | Report only decisions whose interval excludes zero, and report the share that stay uncertain (Brill style) |
| Bowler and batter learning against each other | Partly | Elite players may already be near equilibrium | First test whether outcomes are equal across choices, as in the tennis paper |

The feasibility agent assumed there is no ball tracking. That is wrong for this repo: Hawk-Eye speed and pitch
position exist for 1,689 matches, including Dhoni's 2019 semi-final (619 balls). They do not exist for the
2023 final, because the feed stopped in February 2022. Hawk-Eye carries no licence, so it is internal use only.

## 4. Value

- **Teams.** Analysts work on matchups and bowling plans. No source shows teams using simulated "what ifs"
  (RCB analyst interview, read). There is public pushback on matchup-led selection (Badrinath on Gambhir,
  Feb 2026, read).
- **The basic simulator is a commodity.** Opta has sold ball-by-ball cricket simulation since 2022 (read).
- **Fantasy money in India is gone.** The August 2025 ban on real-money gaming shut Dream11's paid contests
  (snippet, several outlets).
- **Content has a precedent.** The NYT 4th Down Bot drew a following in one season. Its makers said its biggest
  flaw was not showing uncertainty (read).

## 5. The economic simulation testbed premise

Weak. A cricket match gives you fixed rules, one clear goal and many repeated trials, which are exactly what
an economy lacks. So passing cricket shows only that ball-level imitation works. It tells you nothing about
rules that change, agents reacting to policy, or behaviour that emerges from many agents together. Failing
cricket would not rule out an economy model either.

Agent-based economics names validation as its own weak point (Fagiolo et al., snippet). LLM economic agents
reproduce known patterns only inside their own simulations (EconAgent, read). Against real survey answers they
drift toward one average persona (ACL Findings 2026, read abstract).

Cheaper and closer testbeds for that goal:
- Replicate classic lab economics experiments, where the human results are published (Horton 2023, read
  abstract).
- Check agents against survey panels such as the Survey of Consumer Expectations.
- Replay real order-book data and score against known market patterns (Vyetrenko et al., read abstract).

## 6. The smaller version worth building

1. **Player-level ball model first** (the one-week falsification test from the original prompt). It is needed by
   every version, and it is also the biggest gap in the current forecaster.
2. **Decisions made once per over or once per innings,** not once per ball:
   - who bowls each over, especially overs 16 to 20 and 41 to 50
   - batting order
   - when to accelerate

   These are logged in Cricsheet, have no label problem, happen thousands of times, and are where other
   sports found real mistakes (NFL fourth downs, MLB pitch mix). The Dhoni 2019 semi-final fits here: he
   batted at 7, coming in at ball 144 of the chase (checked in gold). The question "should he have batted at
   5?" is a batting-order decision.
3. **Intervals by resampling whole matches.** Call a choice a mistake only when most refits agree, and publish
   the share that stay uncertain.
4. **Delivery-level questions only inside the Hawk-Eye matches,** as a separate test, with Hawk-Eye length
   and speed instead of Cricinfo labels.
5. **Drop for now:**
   - tree search
   - the learned captain model (use the real bowling order as fixed)
   - the shot model built from Cricinfo labels
   - per-ball regret as the headline
   - Bumrah 2023, unless it is framed as "what you would have to believe"
6. **Video:** pick one famous over-level decision, fix the method before running it, and lead with the
   uncertainty band.
7. **Economic simulation:** a separate track, tested on lab-experiment replication, not on cricket.

## Not read or not verified

- The 48% and 44% NFL figures come from the paper's body, read by an agent. I confirmed only the abstract.
- MOPO, MOReL, MuZero Unplugged and Gottesman: abstracts only.
- The team-analyst picture rests on one 2022 interview. No solid 2024 to 2026 inside-team reporting was found.
- No data was found on how cricket "what if" videos perform.
