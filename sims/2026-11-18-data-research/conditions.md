# Match day conditions: open data research (2026-11-18)

Verdict: the best next step is the scheduled start time and day/night flag from Wikipedia series pages. It is open (CC BY-SA 4.0), it keys on our match id, and it fixes a real fault in what we hold: for every ODI before late 2019 our weather window is a fixed 10:00 to 22:00, so day matches and day/night matches get the same weather numbers. Evening dew can then be built from weather we already hold. No open source exists for pitch reports or for which strip was used.

Nothing in the repo was changed except this file. All numbers below are marked measured or inferred.

## Ranked table

| Rank | Candidate | Source | Licence | Exportable | Measured coverage | Effort | Expected value for T20I and ODI scoring (inferred) |
|---|---|---|---|---|---|---|---|
| 1 | Scheduled start time and day/night flag | English Wikipedia series and season pages, match templates | CC BY-SA 4.0 (share alike) | Yes, with attribution and share alike | Sample of 50 gold matches: ODI 15 of 25 have a time (0 of 7 before 2009, 15 of 18 from 2009 on); T20I 21 of 22 pages read. IPL 2014: 60 of 60. | Small ingest (about 670 series pages plus 19 IPL season pages) | High. Splits day from day/night, and makes every weather feature point at the right hours. |
| 2 | Evening dew proxy for the second innings | Derive from Open-Meteo hourly data we hold (dew point gap, wind, cloud) plus the start time from row 1 | CC BY 4.0 | Yes | Weather held for 5,869 of 5,871 gold matches. True start time today: T20I 585 of 1,136, ODI 575 of 2,596, IPL 482 of 1,243 (internal only source) | Derive (after row 1) | Medium to high for day/night games in South Asia and the UAE; low elsewhere. |
| 3 | Pitch read on the day: spinners picked in the two XIs, and the toss choice | Derive from Cricsheet XIs and toss (held). Bowling style is from the Cricinfo-derived file (held, internal only) | Cricsheet ODC-BY 1.0; bowling style not exportable | Toss yes. Spinner count no, unless style comes from an open source | Bowling style known for 99.9 to 100% of balls in ODI, T20I, IPL | Derive | Medium. It is the closest thing to a pitch report that we can get; both teams choose their XI after seeing the pitch. |
| 4 | Days since the last match on the ground, and matches there in the last 14 days | Derive from all Cricsheet matches in silver (not only gold) | ODC-BY 1.0 | Yes | A previous match found for ODI 2,486 of 2,596, T20I 1,126 of 1,136, IPL 1,237 of 1,243 | Derive | Low to medium. It is ground use, not strip use, and Cricsheet misses most domestic cricket. |
| 5 | Rain in the 2 to 7 days before, and top soil moisture | Open-Meteo archive (same source as now, wider window and two more variables) | CC BY 4.0; free tier non-commercial only | Yes | Sample call returned real values. Today we hold 1 day before the match in 3,290 of 3,336 weather files that have a gold match | Small ingest (about 3,700 calls, inside the free daily limit) | Low to medium. Pitches are covered and watered, so outside rain is a weak signal for the pitch; better for a slow outfield. |
| 6 | Ground size (playing field length and width) | OpenStreetMap cricket field polygons, through Overpass | ODbL 1.0 (share alike) | Yes, with attribution and share alike | 20 of 30 sampled venues have a field-sized polygon (18 with our venue point inside it) | Small ingest | Low for the on-the-day part (it does not change by day; the rope does). Useful for grounds with few matches. |
| 7 | Altitude | Already held: dim_venue.elevation_m | CC BY 4.0 (Open-Meteo 90 m model), 1 Wikidata, 1 manual | Yes | 182 of 182 venues, 100% of gold matches | Derive (nothing to fetch) | Low for the on-the-day part, same reason as ground size. 436 ODIs and 132 T20Is are above 1,000 m. |
| 8 | Sunset time against the innings times | Compute from latitude, longitude and date (or Open-Meteo daily sunrise and sunset) | No licence needed for the formula | Yes | 100% of venues have coordinates | Derive (after row 1) | Part of rows 1 and 2; share of the chase played under lights. |
| 9 | Rain and delay notes per match | Wikipedia match templates, `rain` and `notes` fields | CC BY-SA 4.0 | Yes | Seen on some sampled matches; not counted | Comes free with row 1 | Low. Free text. |
| 10 | Pitch type or pitch report | None open | - | - | 0% | - | Would be high, but there is no legal open source. See ruled out. |

## 1. Scheduled start time and day/night flag

What we hold today (measured):
- Cricsheet has no start time and no day/night field. `src/cricket_sim/venues/weather.py` line 37 says so.
- `data/bronze/ari/v11/matches.csv` has no day/night, start time, pitch or ground field. I read every column in `data/bronze/ari/v11/schema.json`.
- The 54 pinned Wikipedia season pages in `data/bronze/wiki_fixtures/2026-10-20/` have 0 "D/N" marks, 0 clock times, and 11 uses of "day/night" in prose. They cannot give this.
- A real start time exists only where the Cricinfo-derived ball timestamps exist (internal only). `data/gold/int_match_play_windows.parquet`, day 1: T20I 585 timed and 550 default; ODI 575 timed and 2,020 default; IPL 482 timed and 761 default. No ODI or T20I before 2019 is timed. From 2020 on, T20I is about 97% timed.
- The default window is fixed by format (docs/DATA_DICTIONARY.md line 1427): ODI 10:00 to 22:00, T20 14:00 to 23:00, IPL 15:30 to 23:30. So the weather features for 2,020 ODIs average over 12 hours whether the game was a day game or a day/night game. This is a likely reason weather added only 2 to 3 points.
- Where timed, ODIs split almost evenly: 271 start before noon, 302 start 12:00 to 16:59. T20Is: 33 before noon, 199 afternoon, 353 from 17:00.

The open source:
- English Wikipedia series pages (for example "Australian cricket team in India in 2023-24") and league season pages hold one template per match: `{{Single-innings cricket match}}`, `{{Limited overs matches}}` or `{{Limited overs international}}`. Fields: `date`, `time`, `daynight` or `night`, `venue`, `toss`, `rain`, `notes`, `report`.
- Template documentation (https://en.wikipedia.org/wiki/Template:Single-innings_cricket_match): time is "The scheduled time for the start of the match." It does not say local time. The `night` field is used on pages but is not in the documentation table.
- Licence: CC BY-SA 4.0. Same terms as the `wiki_fixtures` source already in `config/wiki_sources.yml`. May be republished with attribution; derived data must be shared alike.
- Attribution text (same pattern as the one in use): Wikipedia contributors, "<series page title>", English Wikipedia (https://en.wikipedia.org), CC BY-SA 4.0; derived data must be shared alike. Revision ids in the snapshot index.
- Access: the same MediaWiki API call the repo already uses, no account. Up to 50 titles per call. Follow redirects (4 of the pages I read were redirects).

Key to our matches (measured):
- The `report` link holds the ESPNcricinfo match id, which is our `match_id`. I asked the reader to find each sampled match by that id: it found 43 of the 47 rows read that way, 2 rows from 2006 and 2007 link to CricketArchive instead and need date plus teams, and 2 were unclear. On the league pages every template had the id (60 of 60, 61 of 61, 34 of 34).
- The pinned season pages already tell us which series page each match is on. Walking back from the match row to the nearest `{{main|...}}` link gives a series page for 2,433 of 2,596 gold ODIs (93.7%) and 1,129 of 1,136 gold T20Is (99.4%). 666 unique pages. Script: `/private/tmp/claude-501/-Users-srikantheranki--claude-spares-spare-uxdT01/33fb1b78-e0ef-4abc-9ebc-ed9114884c9d/scratchpad/scripts/wiki_map.py`.

Coverage on a seeded random sample of 25 gold ODIs and 25 gold T20Is (measured; values in `/private/tmp/claude-501/-Users-srikantheranki--claude-spares-spare-uxdT01/33fb1b78-e0ef-4abc-9ebc-ed9114884c9d/scratchpad/conditions-research/wikipedia-series-pages/template_values_transcribed.csv`):
- ODI: time present for 15 of 25. By period: 0 of 7 for 2003 to 2008, 15 of 18 from 2009 on, 8 of 8 from 2022 on. One more match has the day/night flag but no time.
- T20I: time present for 21 of the 22 I could read. 3 World Cup matches were not read because the reader stops at 100,000 characters and those pages are longer.
- World Cups: the main page sometimes has no time and the group sub page does (2015 World Cup: main page no, "Pool A" page yes).
- A blank day/night field with a morning time means a day match, so the time is the field that matters.
- IPL: "2014 Indian Premier League" has 60 of 60 templates with time and match id. "2025 Indian Premier League" 46 of 46 read. "2008 Indian Premier League" has 59 templates; the first 8 all have time and match id.
- Other T20 leagues: "2019-20 Big Bash League season" 61 of 61, "2023 Pakistan Super League" 34 of 34. One season each, so this is a spot check, not a coverage number.

Does the Wikipedia time agree with our data (measured): for the 19 sampled matches that have both a Wikipedia time and a first-ball time, 15 agree within 3 minutes. The other 4 start 50 to 152 minutes late in our data (rain delays, for example India v Pakistan in New York on 2024-06-09: scheduled 10:30, first ball 11:20). One of them (match 1433369, exactly 60 minutes) should be checked for a time zone slip.

Limits: early matches (before 2009) mostly have no time on Wikipedia. For those, a day/night flag could still be set from the pages that have only the flag, or left null.

## 2. Evening dew

- Derive first. We hold hourly dew point, temperature, humidity, cloud and wind for the match days (`data/bronze/weather/`, 3,727 files in the newest snapshot). Gold already has `window_dew_point_depression_c`, but over the whole play window.
- What is missing is the right hours. With the start time from section 1, compute for the second innings only: the gap between air temperature and dew point, wind speed, cloud cover, and whether it is after sunset.
- Open-Meteo also returns `is_day`, `vapour_pressure_deficit`, `shortwave_radiation` and daily `sunrise` and `sunset` (measured, one sample call; notes in `/private/tmp/claude-501/-Users-srikantheranki--claude-spares-spare-uxdT01/33fb1b78-e0ef-4abc-9ebc-ed9114884c9d/scratchpad/conditions-research/open-meteo-sample/README.txt`).
- Inferred: ERA5 is a model on a coarse grid, so it gives the chance of dew, not whether the ball was wet. No open source records actual dew.

## 3. Pitch read on the day: team selection and toss

- No one publishes the pitch, but both teams look at it and then pick the XI. The number of spinners picked by each side is known at the toss.
- XIs and toss choice are in Cricsheet (held, exportable). Bowling style is in `data/silver/cricinfo_players.parquet` (`ari_bowling_style`), which is Cricinfo-derived and internal only. Measured: style is known for 100% of IPL and T20I balls and 99.9% of ODI balls in gold.
- So this can be used inside the model now, but the spinner count cannot go in the export. An open source for bowling style (Wikipedia player infoboxes, CC BY-SA 4.0) exists but I did not measure it.

## 4. Pitch reuse and days since the last match on the ground

- Derive from `data/silver/matches.parquet` joined to `data/silver/match_venues.parquet`, which cover every Cricsheet match (19,000 or so, including women's games and other leagues), not only the 5,871 in gold.
- Measured, days from the end of the previous Cricsheet match on the same ground:

| Segment | Matches | Has a previous match | 0 to 1 days | 2 to 3 | 4 to 7 | 8 to 30 | Over 30 | Mean matches there in the prior 14 days |
|---|---|---|---|---|---|---|---|---|
| ODI | 2,596 | 2,486 | 117 | 837 | 231 | 407 | 894 | 1.12 |
| T20I | 1,136 | 1,126 | 149 | 368 | 103 | 176 | 330 | 1.73 |
| IPL | 1,243 | 1,237 | 67 | 371 | 350 | 286 | 163 | 1.87 |

- Limits (inferred): this is ground use, not which strip. Cricsheet does not hold most domestic first-class and one-day cricket, so "over 30 days" is often wrong. No open source says which strip was used or whether it was a used pitch.

## 5. Rain before the match and soil moisture

- Held today: `pre_match_12h_precipitation_mm` for all but 3 gold matches, and rain hours inside the play window. The weather files cover 1 day before the first match day in 3,290 of 3,336 files that contain a gold match (measured from file names against `dim_match`).
- To get 2 to 7 days before, widen the request window. The source, licence and attribution stay as in `config/context_sources.yml`.
- New variables that the same endpoint returns (measured on one sample call): `soil_moisture_0_to_7cm`, `soil_temperature_0_to_7cm`, `et0_fao_evapotranspiration`, `wind_gusts_10m`, `cloud_cover_low`, `wet_bulb_temperature_2m`.
- Terms (https://open-meteo.com/en/terms, read today): "You may only use the free API services for non-commercial purposes." Limits: under 10,000 calls a day, 5,000 an hour, 600 a minute. Data licence CC BY 4.0. No account needed.

## 6. Ground size

- No open table of boundary lengths exists. A news article describes a community project that measured 124 grounds from satellite images, but gives no link or licence (https://www.arabnews.com/cricket/one-size-does-not-fit-all-cricket-grounds-3007709), so I could not check it.
- OpenStreetMap maps the playing field of many grounds as a polygon tagged `leisure=pitch` and `sport=cricket`. Licence ODbL 1.0, share alike, attribution "(c) OpenStreetMap contributors". The repo already uses OSM for geocoding under the same terms.
- Access: Overpass API, no account. It is a shared free service; one of my 4 calls got HTTP 429 and worked on retry, so go slowly. Key: our venue coordinates, then store the OSM way id on the venue.
- Measured on 30 random venues from `dim_venue` (250 m search): 20 have a field-sized polygon, 18 of those with our venue point inside its box, 14 with exactly one candidate. The 10 without include Gaddafi Stadium, Stadium Australia and Bangabandhu National Stadium (mapped as a stadium only, or not a cricket field now). Box sizes range from about 132 by 123 m to 191 by 192 m. Files: `/private/tmp/claude-501/-Users-srikantheranki--claude-spares-spare-uxdT01/33fb1b78-e0ef-4abc-9ebc-ed9114884c9d/scratchpad/conditions-research/osm-overpass-sample/`.
- Limits (inferred): the polygon is the grass, not the rope on the day. Ropes move match to match. Grounds with several ovals (Colombo, King City, Potchefstroom) need a pick by distance. I measured boxes, not true straight and square lengths.
- Internal only option: Hawk-Eye `field_x` and `field_y` in `data/silver/hawkeye_ball_landing.parquet` could give where boundaries were hit, for IPL and ODIs up to February 2022. Not exportable. Not measured.

## 7. Altitude

- Already in gold. `dim_venue.elevation_m` is filled for 182 of 182 venues: 180 from the Open-Meteo 90 m elevation model (the weather responses carry an `elevation` field; checked in one bronze file), 1 from Wikidata, 1 manual.
- Gold matches above 1,000 m: ODI 436, T20I 132, IPL 44, Test 75.
- Nothing to fetch. It does not change by day, so it can only help the venue part of the model, not the on-the-day part.

## Other things found

- Wikipedia `rain` and `notes` fields: free text on delays and reduced overs. Comes with section 1.
- Toss choice (held): a weak sign of how captains read the conditions.
- Rule changes by date (for example two new balls in ODIs from October 2011): a small hand-made calendar, no source needed. Not researched here.

## Ruled out

- ESPNcricinfo pages (pitch reports, match notes, ground pages, start times): terms forbid automated access and reuse. Not fetched.
- Cricbuzz, ICC site (including the pitch ratings page at https://www.icc-cricket.com/news/pitch-ratings), news sites with pitch reports: copyrighted, no open licence. Not fetched.
- Kaggle or GitHub sets with "pitch type" columns: a web search found none with an open licence; the general ones are scraped from Cricinfo.
- UK government "Cricket Pitches" set: community pitches in England only, and it states no licence.
- Blog lists of boundary lengths (itsonlycricket.com and similar): no licence, some garbled numbers.
- Wikipedia season pages for day/night: measured, they do not carry it.

## Could not check

- Raw sample downloads. A scripted download of the Wikipedia pages was blocked by the permission system, so every web read here went through the page reader tool. The scratch folder holds transcribed values, not raw files, and the reader is a small model, so a real parser should confirm the counts before anyone relies on them.
- 3 of the 25 sampled T20Is (2024 and 2026 T20 World Cup pages are longer than the reader's 100,000 character limit).
- Whether Wikipedia times are always local time. The 15 of 19 matches that agree within 3 minutes with our local first-ball times say yes for recent matches.
- Open bowling style coverage from Wikipedia infoboxes.
- The community boundary project (no link found).
- Wikipedia time coverage for IPL seasons other than 2008, 2014 and 2025, and for leagues other than one BBL and one PSL season.
- The scratch folder is temporary and can be cleared on reboot.

## Sources read today

- https://en.wikipedia.org/wiki/Template:Single-innings_cricket_match
- English Wikipedia series and season pages listed in the transcribed CSV (wikitext, `action=raw`)
- https://archive-api.open-meteo.com/v1/archive (one sample call) and https://open-meteo.com/en/terms
- https://overpass-api.de/api/interpreter (4 calls, OpenStreetMap data, ODbL)
- https://www.arabnews.com/cricket/one-size-does-not-fit-all-cricket-grounds-3007709 (through web search only)
