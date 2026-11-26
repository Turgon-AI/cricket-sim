# Adding men's T20 leagues to gold: scoping

Written 2026-11-18 (IST). Research only: no repo file, config or data under `data/` was changed.
Snapshot used: `data/bronze/cricsheet/2026-10-14-13af720e`, silver and gold of build `0240fb4ce447d1eb`.

## Recommendation (five lines)

1. The main reason given for this work is already met: `fact_player_match_form` already counts every Cricsheet T20 league ball for T20I players. Adding league segments will not put more balls behind T20I player ratings. It adds league matches as rows (for training T20 ball models, venue history and league-strength adjustment).
2. Use one segment per league (as the IPL is today), not one "t20_league" segment. Coverage, home grounds and team names differ by league, and the reports already break out by segment.
3. First wave, nine leagues: T20 Blast, BBL, BPL, CPL, PSL, LPL, ILT20, SA20, MLC, from 2015-04-01. That is 3,669 matches and 849,744 deliveries, on top of 2,379 T20 matches and 555,971 deliveries in gold today.
4. Coverage claim for that wave: 98.6% or better per league against Cricsheet's own missing list (21 listed gaps in total). No outside reference list exists for leagues, so the claim is the same kind as the IPL's, not the kind made for internationals.
5. Leave out for now: Syed Mushtaq Ali Trophy (58 to 65% held, 603 matches with no umpires), CSA T20 Challenge (83%), Super Smash before 2019/20, The Hundred (100-ball format, needs its own decision), and the small domestic events. "Edit scope.yml and rebuild" does not work: about 20 places change, and 30 new grounds need geocode and weather fetches from the network first.

## 0. The finding that changes the plan

Measured, not inferred.

- `int_player_match_stats` is built over all matches with no scope (`src/cricket_sim/gold.py:598-600`, reads `matches m`), and its format family puts every non-Test, non-one-day match in `t20` (`src/cricket_sim/gold.py:591-592`).
- `fact_player_match_form` sums that table per player and format family over earlier dates, then keeps the rows of in-scope matches (`src/cricket_sim/gold.py:643-656`, `:673`).
- The contract says the same: "t20 (T20I, IPL and every T20 league Cricsheet covers)" (`contracts/gold/fact_player_match_form.yaml:50`).
- Check on the data: for the 6,008 player rows of full-member T20Is from 2024, `prior_bat_balls` equals the sum over all earlier T20 matches (leagues included) in 5,985 rows, and equals the internationals-only sum in 350 rows. I did not chase the 23 rows that differ from both.
- The simulator reads this table for its XI strength (`scripts/measure_explained_share.py:146-155`).

So the 10 to 17% explained share in `sims/2026-11-18-variance-split/summary.json` was measured with league balls already inside the player ratings. What is not there today: league balls are mixed in raw, with no adjustment for how strong the league is, and league matches are not rows in `dim_match` or `fact_delivery`, so nothing can be trained or checked on them.

One side effect to know about: The Hundred (5-ball overs, 100 balls) is also counted as `t20` in player form today (37,292 legal balls).

## 1. Inventory

Measured from `data/silver/matches.parquet` and `balls.parquet`. Silver holds all 23,132 Cricsheet matches (the zip has 23,134 files), so no re-ingest of Cricsheet is needed. Men's club T20: 7,131 matches, 1,583,625 legal balls. Every one of them has ball-by-ball data and at least one innings. "Two innings" below means both sides batted. "Quality" is `passes_quality` from gold `int_match_quality`, which already covers every match.

### Men's leagues, all years held

| League | Cricsheet event_name values | Matches | First | Last | Legal balls | Two innings | No result | Pass quality |
|---|---|---|---|---|---|---|---|---|
| T20 Blast | NatWest T20 Blast (2014 to 2017), Vitality Blast (2018 to 2024), Vitality Blast Men (2025 on) | 1,570 | 2014-05-16 | 2026-07-18 | 349,165 | 1,539 | 39 | 1,569 |
| IPL (in gold today) | Indian Premier League | 1,243 | 2008-04-18 | 2026-05-31 | 284,630 | 1,237 | 9 | 1,243 |
| Syed Mushtaq Ali Trophy | Syed Mushtaq Ali Trophy | 695 | 2016-01-02 | 2024-12-15 | 154,382 | 691 | 4 | 694 |
| BBL | Big Bash League | 662 | 2011-12-16 | 2026-01-25 | 148,259 | 656 | 11 | 662 |
| BPL | Bangladesh Premier League | 469 | 2012-02-11 | 2026-01-23 | 107,278 | 469 | 0 | 469 |
| CPL | Caribbean Premier League | 445 | 2013-07-30 | 2026-09-20 | 98,847 | 439 | 7 | 445 |
| PSL | Pakistan Super League | 357 | 2016-02-04 | 2026-05-03 | 80,816 | 354 | 3 | 357 |
| CSA T20 Challenge | MiWAY T20 Challenge (2011/12), Ram Slam T20 Challenge (2012/13 to 2017/18), CSA T20 Challenge (2016/17 to 2024/25), CSA Pro20 (2026/27) | 320 | 2012-02-15 | 2026-09-29 | 69,960 | 310 | 14 | 320 |
| Super Smash | HRV Cup (2012/13), HRV Twenty20 (2013/14), Super Smash (2017/18 on) | 270 | 2013-01-20 | 2026-01-31 | 59,113 | 258 | 13 | 270 |
| The Hundred | The Hundred Men's Competition | 201 | 2021-07-22 | 2026-08-16 | 37,292 | 197 | 7 | 201 |
| LPL | Lanka Premier League | 143 | 2020-11-26 | 2026-08-08 | 31,663 | 141 | 3 | 143 |
| ILT20 | International League T20 | 134 | 2023-01-13 | 2026-01-04 | 30,171 | 133 | 1 | 134 |
| SA20 | SA20 | 130 | 2023-01-10 | 2026-01-25 | 27,939 | 123 | 7 | 130 |
| Sri Lanka Major Clubs T20 | Major Clubs T20 Tournament | 121 | 2024-05-12 | 2026-09-16 | 23,404 | 114 | 8 | 121 |
| Ireland Inter-Provincial T20 | Cricket Ireland Inter-Provincial Twenty20 Trophy | 111 | 2016-05-27 | 2026-07-29 | 23,019 | 108 | 3 | 111 |
| MLC | Major League Cricket | 109 | 2023-07-13 | 2026-07-18 | 24,416 | 109 | 1 | 109 |
| Nepal Premier League | Nepal Premier League | 64 | 2024-11-30 | 2025-12-13 | 14,464 | 64 | 0 | 64 |
| Mzansi Super League | Mzansi Super League | 56 | 2018-11-16 | 2019-12-16 | 12,183 | 54 | 3 | 56 |
| European T20 Premier League | European T20 Premier League | 31 | 2026-08-26 | 2026-09-20 | 6,624 | 31 | 1 | 31 |

Notes on naming:

- Three leagues change `event_name` across seasons (T20 Blast, CSA T20 Challenge, Super Smash). Cricsheet's download page groups each under one code (ntb, ctc, ssm; checked on cricsheet.org/matches on 2026-11-18), but the match files carry the season's sponsor name. `scope.yml` today can only match one exact `event_name` per segment.
- The Hundred is stored as `match_type` T20 with `scheduled_overs` 20 and `balls_per_over` 5.
- Not on Cricsheet at all: T20 Blast before 2014, Champions League T20, Global T20 Canada, Abu Dhabi T10, and the Afghanistan Premier League (listed with 0 matches).
- Team names are never shared between two leagues (measured). Inside a league, a renamed franchise is a new name: the IPL has 19 team names, the BPL 34, the LPL 19.

### Men's leagues, matches by calendar year of start date

| Year | T20 Blast | IPL | SMAT | BBL | BPL | CPL | PSL | CSA T20 | Super Smash | Hundred | LPL | ILT20 | SA20 | SL Clubs | Ireland | MLC | NPL | MSL | ETPL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2008 | 0 | 58 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2009 | 0 | 57 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2010 | 0 | 60 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2011 | 0 | 73 | 0 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2012 | 0 | 74 | 0 | 34 | 17 | 0 | 0 | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2013 | 0 | 76 | 0 | 18 | 38 | 23 | 0 | 25 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2014 | 122 | 60 | 0 | 37 | 0 | 27 | 0 | 56 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2015 | 124 | 59 | 0 | 34 | 31 | 32 | 0 | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2016 | 121 | 60 | 83 | 31 | 45 | 29 | 23 | 27 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| 2017 | 121 | 59 | 0 | 36 | 44 | 34 | 24 | 25 | 13 | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 0 |
| 2018 | 126 | 60 | 79 | 45 | 0 | 33 | 33 | 0 | 19 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 30 | 0 |
| 2019 | 112 | 60 | 261 | 62 | 73 | 34 | 34 | 22 | 38 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 26 | 0 |
| 2020 | 89 | 60 | 0 | 65 | 18 | 33 | 32 | 0 | 22 | 0 | 23 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 |
| 2021 | 120 | 60 | 181 | 67 | 0 | 33 | 34 | 16 | 44 | 32 | 24 | 0 | 0 | 0 | 15 | 0 | 0 | 0 | 0 |
| 2022 | 130 | 74 | 44 | 56 | 32 | 31 | 34 | 59 | 22 | 34 | 24 | 0 | 0 | 0 | 14 | 0 | 0 | 0 | 0 |
| 2023 | 132 | 74 | 39 | 57 | 46 | 32 | 34 | 0 | 27 | 33 | 24 | 32 | 33 | 0 | 11 | 19 | 0 | 0 | 0 |
| 2024 | 126 | 71 | 8 | 39 | 49 | 34 | 32 | 30 | 26 | 34 | 24 | 34 | 33 | 32 | 17 | 23 | 32 | 0 | 0 |
| 2025 | 132 | 74 | 0 | 42 | 48 | 32 | 34 | 0 | 32 | 34 | 0 | 65 | 41 | 40 | 22 | 33 | 32 | 0 | 0 |
| 2026 | 115 | 74 | 0 | 27 | 28 | 38 | 43 | 6 | 23 | 34 | 24 | 3 | 23 | 49 | 7 | 34 | 0 | 0 | 31 |

2026 is a part year (snapshot of 2026-10-14).

### Women's leagues (counts only)

| event_name | Matches | First | Last | Deliveries |
|---|---|---|---|---|
| Women's Big Bash League | 519 | 2015-12-18 | 2025-12-13 | 118,656 |
| The Hundred Women's Competition | 188 | 2021-07-21 | 2026-08-16 | 36,850 |
| Women's Super Smash | 148 | 2021-12-12 | 2026-01-31 | 33,578 |
| Charlotte Edwards Cup | 124 | 2021-06-26 | 2024-06-22 | 29,065 |
| Vitality Blast Women | 112 | 2025-05-30 | 2026-07-17 | 25,909 |
| Women's Cricket Super League | 95 | 2016-07-30 | 2019-09-01 | 21,484 |
| Women's Premier League | 88 | 2023-03-04 | 2026-02-05 | 20,656 |
| New Zealand Cricket Women's Twenty20 | 49 | 2020-01-02 | 2021-12-10 | 10,933 |
| FairBreak Invitational Tournament | 39 | 2022-05-04 | 2023-04-16 | 8,985 |
| Women's Caribbean Premier League | 33 | 2022-08-31 | 2026-09-17 | 7,575 |
| T20 Blaze, Women's T20 Blaze | 17 | 2019-03-29 | 2022-06-14 | 3,798 |
| Women's T20 Challenge (and Challenge Match) | 13 | 2018-05-22 | 2022-05-28 | 3,088 |

Total 1,425 women's club T20 matches. The balls column here counts every delivery, wides and no-balls included.

## 2. Player overlap with T20Is from 2019

Measured from gold `int_player_match_stats` (all matches, keyed on the Cricsheet people id; all 13,741 player ids in it are in the `people.csv` register). "Before" means matches on earlier dates only. "League" means the 18 non-IPL events above.

### Full-member T20Is (the gold T20I segment), 2019-01-01 on: 638 matches, 139,742 balls faced, 139,154 legal balls bowled

| Share of T20I balls by players who, before that match, had | Batting | Bowling |
|---|---|---|
| any league balls, IPL included | 94.5% | 96.0% |
| non-IPL league balls | 92.0% | 95.3% |
| balls in BBL, PSL, CPL, SA20, T20 Blast, BPL, LPL, ILT20, MLC, Super Smash, The Hundred or MSL | 79.5% | 80.8% |
| IPL balls | 49.5% | 53.2% |

| Average earlier T20 balls behind the player on strike or bowling (weighted by T20I balls) | Batting | Bowling |
|---|---|---|
| in T20Is (all men's T20Is) | 675 | 673 |
| in the IPL | 484 | 396 |
| in the other leagues | 1,046 | 989 |
| total | 2,205 | 2,058 |

So counting leagues gives about 3.3 times the balls of T20Is alone for batters and 3.1 times for bowlers. In the last 365 days before the match the batter has 265 league balls (IPL included) against 169 T20I balls.

| Share of T20I balls by a player with under 200 earlier balls | Batting | Bowling |
|---|---|---|
| T20Is only | 27.6% | 25.0% |
| T20Is and IPL | 23.2% | 19.9% |
| all T20 on Cricsheet | 7.7% | 4.8% |

Median per player appearance: 185 earlier T20I balls faced against 554 in leagues (IPL included); bowlers 479 against 1,060.

Where the league balls come from (average earlier balls faced, weighted by T20I balls): IPL 484, T20 Blast 172, CPL 151, PSL 131, BPL 124, BBL 115, CSA T20 Challenge 64, Super Smash 62, Syed Mushtaq Ali 49, LPL 41, The Hundred 33, SA20 30, MSL 24, ILT20 23, MLC 12, Ireland 12, others 2 or less.

By team (batting, average earlier balls behind the batter):

| Team | T20Is | T20I | IPL | Other leagues | Share of balls by batters with non-IPL league balls |
|---|---|---|---|---|---|
| India | 163 | 761 | 2,032 | 375 | 81.0% |
| Pakistan | 141 | 844 | 2 | 1,215 | 99.3% |
| New Zealand | 136 | 686 | 173 | 1,266 | 99.9% |
| West Indies | 134 | 565 | 350 | 1,514 | 99.8% |
| England | 120 | 708 | 521 | 1,979 | 100.0% |
| South Africa | 117 | 583 | 451 | 1,322 | 99.7% |
| Sri Lanka | 116 | 626 | 23 | 424 | 89.6% |
| Australia | 106 | 719 | 880 | 1,456 | 100.0% |
| Bangladesh | 106 | 576 | 32 | 886 | 98.9% |
| Zimbabwe | 79 | 572 | 12 | 210 | 32.8% |
| Ireland | 58 | 714 | 0 | 610 | 100.0% |

Zimbabwe is the one side that leagues barely help. India's non-IPL league balls are almost all Syed Mushtaq Ali.

### All men's T20Is (associates included), 2019 on: 3,015 matches

Only 28.4% of balls are faced by a batter with any league balls (29.0% for bowlers). Leagues do nothing for associate sides.

### What this means

These balls are already in `fact_player_match_form` (section 0). The gain from this project is therefore not "more balls behind the rating". It is being able to weight a league ball differently from a T20I ball, and to train and test on league matches.

## 3. Coverage: what can be claimed per league

### How the IPL is handled today

- There is no outside reference list for the IPL. The Wikipedia reference is numbered men's internationals only: the page list is "International cricket in YYYY" (`config/wiki_sources.yml:1-14`, `src/cricket_sim/wiki_fixtures.py:30-36`), the parser looks for "Test / ODI / T20I No." (`src/cricket_sim/wiki_fixtures.py:154`), and the coverage report is limited to those three segments (`src/cricket_sim/gold.py:302-305`, `:487`).
- The IPL has two light checks. `report_cricsheet_known_missing` keeps the IPL entries of Cricsheet's missing list (`src/cricket_sim/gold.py:545-549`); there are none. `report_ipl_seasons` counts matches per season (`src/cricket_sim/gold.py:552-555`), with a floor of 18 seasons (`contracts/gold/report_ipl_seasons.yaml:46-50`).
- The card's wording is "every match Cricsheet holds in that scope, which is not every match played" (`docs/DATASET_CARD.md:42-43`).

### Does it generalise

Yes, the IPL check carries over as it is, with two catches.

- The missing list names a competition differently from the match files: "T20 blast", "Csa t20 challenge", "Super smash" (title-cased by Cricsheet, one name per competition across sponsor renames). A small map from list name to league is needed.
- A league with no gaps has no heading on the missing list at all (IPL, SA20, ILT20, LPL, MLC, The Hundred, MSL, Nepal). "Not on the list" then reads as zero gaps. That cannot be told apart from "Cricsheet does not track gaps here".

A second, independent check exists inside the data: Cricsheet's `event_match_number`. A number missing inside a season points at a match Cricsheet lacks or one abandoned without a ball. I measured both.

### Measured, seasons Cricsheet holds

"Listed missing" = entries on Cricsheet's missing list inside a held season. "Number gaps" = missing `event_match_number` values (not available where Cricsheet gives no numbers).

| League | Seasons | Held | Listed missing | Coverage | Held from 2015 | Missing from 2015 | Coverage from 2015 | Number gaps, seasons from 2019 |
|---|---|---|---|---|---|---|---|---|
| T20 Blast | 13 | 1,570 | 8 | 99.5% | 1,448 | 5 | 99.7% | no numbers |
| IPL | 19 | 1,243 | 0 | 100% | 785 | 0 | 100% | 2 |
| BBL | 15 | 662 | 16 | 97.6% | 539 | 3 | 99.4% | 8 |
| BPL | 12 | 469 | 30 | 94.0% | 414 | 6 | 98.6% | 4 |
| CPL | 14 | 445 | 9 | 98.0% | 395 | 5 | 98.8% | 8 |
| PSL | 12 | 357 | 2 | 99.4% | 357 | 2 | 99.4% | 5 (after removing a season-label split) |
| LPL | 6 | 143 | 0 | 100% | 143 | 0 | 100% | 0 |
| ILT20 | 4 | 134 | 0 | 100% | 134 | 0 | 100% | 2 |
| SA20 | 4 | 130 | 0 | 100% | 130 | 0 | 100% | 5 |
| MLC | 4 | 109 | 0 | 100% | 109 | 0 | 100% | 2 |
| The Hundred | 6 | 201 | 0 | 100% | 201 | 0 | 100% | 3 |
| MSL | 2 | 56 | 0 | 100% | 56 | 0 | 100% | 4 |
| Super Smash | 11 | 270 | 35 | 88.5% | 266 | 4 | 98.5% | 13 |
| CSA T20 Challenge | 14 | 320 | 65 | 83.1% | 209 | 44 | 82.6% | 20 |
| Syed Mushtaq Ali | 9 | 695 | 369 | 65.3% | 695 | 369 | 65.3% | no numbers |
| Ireland Inter-Pro | 11 | 111 | 4 | 96.5% | 111 | 4 | 96.5% | 25 |

Things the table hides:

- Whole seasons Cricsheet does not hold do not show as a low percentage. Super Smash has 263 listed-missing matches from 2006 to January 2017 in seasons with no held match. Syed Mushtaq Ali has 141 for 2025/26 and nothing held; its held seasons fall from 91% (2015/16) to 34% (2022/23), 30% (2023/24) and 6% (2024/25). CSA T20 Challenge holds 4 of 26 for 2023/24.
- The number gaps in recent seasons are not on the missing list. The likely cause is washouts with no ball bowled, which Cricsheet never holds (inferred, not checked match by match). The PSL figure of 19 in the raw output is mostly the 2021 season, which Cricsheet splits across two season labels.
- No listed-missing match is also held (0 of the 1,222 men's competition entries match a held T20 match on teams and date).

### The claim I would make

For the nine first-wave leagues from 2015-04-01: "every match Cricsheet holds; Cricsheet's own missing list names 21 more (T20 Blast 5, BBL 3, BPL 6, CPL 5, PSL 2), so at least 98.6% of each league by that list". This is the IPL-grade claim. It is weaker than the internationals claim, which rests on an outside list with a named reason for every gap.

A claim of the internationals kind needs a new reference list per league (Wikipedia season pages such as "2023 Pakistan Super League", or the league's own fixture list). `wiki_fixtures.py` cannot read those pages: it keys on international match numbers. That is a new parser and about 75 season pages for nine leagues. I would not block the first wave on it.

## 4. What must change to add a league segment

"Edit scope.yml and rebuild" fails today. I dry-ran the gold contract rules in memory on all 5,888 non-IPL league matches (views only, nothing written): `dim_match` fails `valid_segment` on every row and `umpires: required` on 605 rows; `dim_venue` fails its required geocode columns on 81 grounds. `fact_innings`, `fact_delivery`, `fact_dismissal` and `fact_player_match_form` pass every rule.

### Must change (build fails otherwise)

1. `config/scope.yml:5-21` and `src/cricket_sim/gold.py:22-24`, `:54-58`, `:66-79`. A segment can match one exact `event_name` and there is one global `from_date`. Add a list form (for example `event_name_in`) for renamed leagues, and an optional per-segment `from_date`. `validate_scope` must accept a list for that key (`:57-58` refuses non-text). Risk: medium. `_segment_case` builds SQL from whatever keys are present; a mistake silently relabels matches. The guard tests are `tests/test_match_rules.py:258-290`.
2. `scripts/full_check.sh:104-113` recomputes scope with its own copy of the same rule. It must learn the same new keys or the release check fails. Risk: low, but easy to forget.
3. `contracts/gold/dim_match.yaml:47-55` (`valid_segment` lists Test, ODI, T20I, IPL) and `contracts/gold/report_coverage_by_segment_year.yaml:36-44` (same list). Add the new segment names. Risk: low.
4. `contracts/gold/dim_match.yaml:297-301` `umpires` is required. 603 Syed Mushtaq Ali matches (2016 to 2021) and 2 Super Smash matches (December 2021) have none. No first-wave league is affected (0 missing). Risk: none for wave one; relax the rule before adding those two.
5. `contracts/gold/dim_venue.yaml:68-72` (`lat` required; `lon`, `country`, `timezone`, `geocode_source`, `geocode_confidence`, `geocode_matched_name`, `weather_location_quality` fail the same way) and `:175-178` ("in_geo_plan always true in scope"). League grounds outside the geocode plan have no coordinates, so the build fails until they are fetched. See item 6.
6. `config/context_sources.yml:13-20` `plan_selection` is the second scope the review named: it decides which matches get geocodes and weather. Add the league event names (it matches exact values, `src/cricket_sim/venues/ingest.py:57-64`, so it needs the list form too). Then run the geo and weather ingest, which needs the network (Wikidata, Nominatim, Open-Meteo), and lock the new snapshots. With `venue_context: asof`, a request the locked snapshots lack fails the build (`config/venue_overrides.yml:8-12`). Size for wave one, measured: 87 grounds, 30 of them new to the plan (T20 Blast 18, BBL 9, MLC 3); 3,238 ground-days of weather. All 18 leagues would be 97 new grounds. Weather calls: the wider 12-league set over all years came to 1,773 fourteen-day windows by my count, under the 8,000 cap (`config/context_sources.yml:56-58`). Risk: medium. New grounds are small out-grounds (Cheltenham, Arundel, Moe, Alice Springs) that may geocode badly and land in `needs_review`. "County Ground, Brighton" and "County Ground, Hove" are two venue ids for what I believe is one ground, so a merge entry is likely needed (inferred, not checked).

### Should change (build passes, but the output would be wrong or misleading)

7. `src/cricket_sim/gold.py:545-549` and `contracts/gold/report_cricsheet_known_missing.yaml:14`, `:22`, `:29`, `:47-51`. The known-missing report hard-codes Test, Odi and "indian premier league". Add the league list names (through the map in section 3). Row floor 390 only grows. Risk: low.
8. `src/cricket_sim/gold.py:552-555`, `contracts/gold/report_ipl_seasons.yaml`, `scripts/full_check.sh:139-140`, `tests/test_report_values.py:53-57`. The per-season count exists for the IPL only. Make it a per-league seasons report (or add a segment column). Risk: low; it is a contract and test rename.
9. `src/cricket_sim/gold.py:591-592` `FORMAT_FAMILY`. Everything that is not Test or one-day is `t20`, so The Hundred is already in T20 player form. Decide this before The Hundred becomes a segment; the death-overs rule `over >= 16` (`:594`) means the last 20 balls there, not the last 24. Risk: changes existing form numbers if The Hundred is split out.
10. `src/cricket_sim/venues/weather.py:38-47` and `contracts/gold/int_match_play_windows.yaml:15`, `:60`. Only the IPL has its own default playing window; other leagues get the T20 default of 14:00 to 23:00 local. That fits evening leagues. It would be wrong for morning starts, which Syed Mushtaq Ali has (from memory, not measured: ball timestamps exist only for internationals). Risk: low for wave one.
11. Home team. The venue home team counts internationals only (`src/cricket_sim/venues/asof.py:65`, `src/cricket_sim/venues/checks.py:73-78`, `config/venue_overrides.yml:16-19`), so it names a national side and says nothing about a franchise. Leave that rule alone. For a franchise, "home" is a property of team and season, not of the ground. Measured with a simple rule (a side's most-used ground that season, at least 35% of its matches and 3 matches): exactly one home side in 94% of Hundred matches, 90% SA20, 86% T20 Blast, 75% BBL, 62% IPL, 50% CPL, 43% PSL, 16% MLC, 11% LPL, 5% BPL. Home sides win 53 to 57% in the leagues where the label is clear. So a home flag is worth having for T20 Blast, BBL, SA20 and the IPL and is close to meaningless for BPL, LPL and MLC. My rule looks at the whole season, so it is hindsight; a model input needs a version built from earlier matches or a small config list of home grounds. Risk: new design, not a config edit. Fine to ship wave one without it.
12. Elo. `src/cricket_sim/models/elo.py:17-47` rates whatever fixture list it is given; today that is ODIs (`src/cricket_sim/models/dataset.py:108`) and, in the uncommitted simulator, one segment at a time (`scripts/simulate_matches.py:59-60`). With one segment per league this works per league with no code change. Two cautions: a renamed franchise starts again at 1500 (Delhi Daredevils and Delhi Capitals are two teams to Elo; the BPL has 34 names), and squads are rebuilt each season, so a league Elo should pull back toward the mean between seasons. A team alias map is needed for it to be useful. Never rate leagues in one pool: teams from different leagues never meet.
13. Docs and export text that state the scope by hand: `README.md:4`, `docs/DATASET_CARD.md:36-43` (the stamped numbers are checked against `headline.json` by `src/cricket_sim/export.py:155-170`, so the card must be restamped), `src/cricket_sim/export.py:353` (Croissant description says "international and IPL"), and the contract descriptions `contracts/gold/dim_match.yaml:14-16`, `:43-44`, `:59`, `:74`, `:91`, `:282` ("6 balls per over throughout the current scope"). The exported tables themselves need no entry in `config/export.yml`: league rows ride in the existing files. Risk: low; the export refuses a stale card.
14. Row floors. Every floor is "at least" (for example `contracts/gold/dim_match.yaml:346-350` 5,500; `fact_delivery.yaml:462-466` 3.4 million; `fact_player_match_form.yaml:385-389` 120,000), and the build-to-build shrink check only refuses a loss (`src/cricket_sim/build.py:124-156`). So adding leagues passes both. Raise the floors after the first league build, or they stop protecting the new rows. Dropping a league later needs `--allow-shrink`. Wave one adds about 3,669 rows to `dim_match`, 849,744 to `fact_delivery`, 81,024 to `fact_player_match_form` and 1,041 players to `dim_player`.
15. Tests and fixture. The 10-match fixture has no non-IPL league match (`scripts/make_fixture.py:46-50`); add one from a renamed league so the list form is exercised. `tests/test_end_to_end.py:28-30` and `tests/test_report_values.py:38-57` pin segment values and counts. `tests/test_venue_asof.py:74` ties `home_team_gender` to the scope gender (no change for men's leagues).

### No change needed (checked)

16. `src/cricket_sim/gold.py:314`, `:487`, `:523` and the contracts `map_match_ids.yaml:45-50`, `report_international_coverage.yaml:38-43`, `report_cricsheet_not_in_reference.yaml:29-34` pin Test, ODI, T20I. Correct as they are: leagues have no reference list.
17. `contracts/gold/dim_match.yaml:62-68` (match_type Test, ODI, T20), `:83-86` (male), `:94-99` (international, club), `:133` (2001 floor), `:373-377` (T20 is 20 overs). League matches satisfy all of them, The Hundred included.
18. Models: `src/cricket_sim/models/dataset.py:108`, `:131`, `:149` and `contracts/models/model_ball_win_probability.yaml:37` are ODI only. Untouched.
19. `src/cricket_sim/hawkeye_link.py:41` (IPL only, the Hawk-Eye feed has no other league) and `src/cricket_sim/context_players.py:293-306` (per-player IPL match counters; optional to extend).
20. Player ids: all 4,900 players in league matches are in the people register, no null ids. Birth dates are thinner for league-only players (72.8% against 94.3% for players already in `dim_player`), so the headline share of players with a birth date will drop.

## 5. Recommendation in full

### One segment per league

- It is what the IPL already is, so every report, test and filter keeps working the same way.
- `report_coverage_by_segment_year` gives per-league counts with no new code.
- Coverage (98.6% to 100% in wave one, 58 to 88% for the ones left out), date floor, home-ground meaning and Elo pool are all per league. One "t20_league" segment would need a new league column to carry the same facts, and the rename map would still have to live somewhere.
- Cost: nine new names in two contract lists, and a list form for `event_name`.

### First wave and floor

Nine leagues from 2015-04-01 (after the 2014/15 BBL, before the 2015 T20 Blast, so no season is cut in half):

| League | Matches | Deliveries | First match kept | Listed missing | Coverage | New grounds |
|---|---|---|---|---|---|---|
| T20 Blast | 1,448 | 332,198 | 2015-05-15 | 5 | 99.7% | 18 |
| BBL | 539 | 124,721 | 2015-12-18 | 3 | 99.4% | 9 |
| BPL | 414 | 98,135 | 2015-11-22 | 6 | 98.6% | 0 |
| CPL | 395 | 92,018 | 2015-06-20 | 5 | 98.8% | 0 |
| PSL | 357 | 83,799 | 2016-02-04 | 2 | 99.4% | 0 |
| LPL | 143 | 32,917 | 2020-11-26 | 0 | 100% | 0 |
| ILT20 | 134 | 31,407 | 2023-01-13 | 0 | 100% | 0 |
| SA20 | 130 | 29,020 | 2023-01-10 | 0 | 100% | 0 |
| MLC | 109 | 25,529 | 2023-07-13 | 0 | 100% | 3 |
| Total | 3,669 | 849,744 | | 21 | | 30 |

Why these: they supply most of the league balls behind full-member T20I players (section 2), every match passes the quality checks, none lacks umpires, and each is at least 98.6% held by Cricsheet's own list. Why the floor: before 2015 the BPL is 51% and 83% held in its first two seasons and the BBL 80 to 87% in its first two; those seasons matter little for T20Is from 2019.

### Second wave, each with a decision attached

- Super Smash from 2019/20 (212 matches in those seasons, no listed gaps since; 6 new grounds; 2 matches without umpires).
- Mzansi Super League (56 matches, complete, ended 2019).
- The Hundred (201 matches, complete, no new grounds), only after the format-family question in item 9 is settled.

### Leave out

- Syed Mushtaq Ali Trophy: 65% held and falling to 6%, 603 matches without umpires, 41 new grounds. It is the only league source for uncapped Indian players, so it is worth a second look if Cricsheet fills it.
- CSA T20 Challenge: 83% held, one season at 4 of 26.
- Sri Lanka Major Clubs, Ireland Inter-Provincial, Nepal, European T20: small, little overlap with full-member T20I players, 18 new grounds between them.
- Women's leagues: out of scope while `gender: male`.

## What I measured and what I inferred

- Measured: every count, share and coverage figure above; the form-table check in section 0; the contract dry run; the ground counts.
- Inferred: that recent event-number gaps are washouts; that the T20 default weather window suits the first-wave leagues; that Brighton and Hove are one ground; the weather call estimate for wave one (I counted the wider 12-league set, not wave one alone).
- Not done: no build was run with a changed scope, so the change list comes from reading the code and from the in-memory dry run of contract rules. I did not run the test suite. I did not check league results against any outside source.

Scratch scripts and CSVs: `/private/tmp/claude-501/-Users-srikantheranki--claude-spares-spare-uxdT01/33fb1b78-e0ef-4abc-9ebc-ed9114884c9d/scratchpad/leagues-research/` (may be cleared on reboot).
