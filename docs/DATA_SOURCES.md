# Data sources and licences

| Source | What we use | Licence | Can we republish it? |
|---|---|---|---|
| [Cricsheet](https://cricsheet.org) | Ball-by-ball JSON for all matches, plus the people register | Open Data Commons Attribution 1.0 | Yes, with credit |
| [Ari Adaikalam, The Ultimate Ball-by-Ball Cricket Dataset](https://www.kaggle.com/datasets/ariadaikalam/the-ultimate-ball-by-ball-cricket-dataset) | Cricinfo line, length, shot and wagon-wheel tracking; text-predicted labels; player styles | Cricsheet parts are ODC-BY. Cricinfo parts are non-commercial and must credit ESPNcricinfo | **No.** Analysis and charts only |
| [Cricsheet missing list](https://cricsheet.org/missing/) | Tests and ODIs Cricsheet knows it lacks | ODC-BY 1.0 | Yes, with credit |
| [Wikipedia "International cricket in YYYY" season pages](https://en.wikipedia.org/wiki/International_cricket_in_2019) | Reference list of every numbered men's Test, ODI and T20I (date, teams, result, Cricinfo id); pinned by page revision | CC BY-SA 4.0 | Yes, with credit; tables derived from it (report_international_coverage, map_match_ids) must stay CC BY-SA |
| [Wikidata](https://www.wikidata.org) | Player birth date, birthplace, citizenship (height and handedness are almost never recorded); venue coordinates | CC0 | Yes |
| [OpenStreetMap](https://www.openstreetmap.org) via Nominatim | Venue geocoding fallback | ODbL | Yes, with "© OpenStreetMap contributors" |
| [Open-Meteo historical archive](https://open-meteo.com) (ERA5) | Hourly weather at each match | CC BY 4.0; the free API is for non-commercial use | Yes, with credit, non-commercial |
| [alittlefitness/HawkeyeStats](https://github.com/alittlefitness/HawkeyeStats) | Hawk-Eye ball speed and pitch/stumps/field x-y (ICC feed, frozen Feb 2022) | None stated, so all rights reserved | **No.** Analysis and charts only |

Credit line for any published output:

> Data: Cricsheet (ODC-BY); Wikipedia (CC BY-SA 4.0); Wikidata (CC0); © OpenStreetMap contributors (ODbL); weather by Open-Meteo.com (CC BY 4.0); The Ultimate Ball-by-Ball Cricket Dataset by Ari Adaikalam, with tracking from ESPNcricinfo; Hawk-Eye data via alittlefitness/HawkeyeStats. Non-commercial analysis.

We never quote or publish Cricinfo commentary text. We never scrape Cricinfo or Cricbuzz ourselves.
