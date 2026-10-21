"""Venue context layer: canonical venues, geocodes, match-day weather.

  canonical  venue identity (scope-free, name-derived venue_id)      checks    post-geocode sanity checks
  geocode    Wikidata / OSM / Nominatim resolution
  weather    UTC normalisation, playing windows, aggregates            store     bronze hygiene for fetched sources
  ingest     network: plan from the bronze Cricsheet zip, fetch, seal  build     offline silver build
"""
