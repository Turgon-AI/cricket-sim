"""Gold keeps the venue as it was on each match date (data-engineering review, 2026-10-27).

dim_venue holds only the latest version of a ground. Before dim_venue_version, gold lost 331 in-scope home teams that
applied at the time (Pakistan's home matches in the UAE, 2009 to 2021, among them) and filled in today's home team for
581 matches played before it was known."""

import duckdb
import pytest

from conftest import ROOT


def _q(data, sql):
    return duckdb.connect().execute(sql.replace("{d}", str(data))).fetchall()


def test_dim_match_names_the_silver_venue_version(tiny_build):
    assert _q(tiny_build, """SELECT count(*) FROM '{d}/gold/dim_match.parquet' d
                              JOIN '{d}/silver/match_venues.parquet' s USING (match_id)
                              WHERE d.venue_version IS DISTINCT FROM s.venue_version""") == [(0,)]


def test_dim_venue_version_is_silver_scoped_to_in_scope_grounds(tiny_build):
    gold = _q(tiny_build, "SELECT * FROM '{d}/gold/dim_venue_version.parquet' ORDER BY ALL")
    silver = _q(tiny_build, """SELECT * FROM '{d}/silver/venue_versions.parquet' WHERE venue_id IN
                               (SELECT venue_id FROM '{d}/gold/dim_match.parquet') ORDER BY ALL""")
    assert gold == silver and gold


FULL = ROOT / "data"


@pytest.mark.skipif(not (FULL / "gold" / "dim_venue_version.parquet").exists(), reason="no full build under data/")
def test_pakistan_home_matches_in_the_uae_keep_pakistan_as_home():
    rows = _q(FULL, """SELECT v.home_team, count(*) FROM '{d}/gold/dim_match.parquet' m
                       JOIN '{d}/gold/dim_venue_version.parquet' v USING (venue_id, venue_version)
                       WHERE m.segment IN ('Test', 'ODI', 'T20I') AND 'Pakistan' IN (m.team1, m.team2)
                         AND m.venue_id IN ('dubai-international-cricket-stadium', 'sheikh-zayed-stadium')
                         AND year(m.start_date) BETWEEN 2012 AND 2018 GROUP BY 1""")
    assert dict(rows).get("Pakistan", 0) > 0.9 * sum(n for _, n in rows), rows
