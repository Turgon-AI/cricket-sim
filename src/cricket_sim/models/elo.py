"""Pre-match Elo ratings for ODI teams, each computed only from matches on earlier dates.

Plain Elo: every team starts at 1500; after each decided match the winner takes K * (1 - expected) points from
the loser, a tie counts as half a win, and a no-result changes nothing. Matches on the same date are rated
from the ratings before that date, so a double-header cannot see its twin's result.
"""

from collections import defaultdict

START, K, SCALE = 1500.0, 24.0, 400.0


def expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / SCALE))


def pre_match_ratings(matches: list[tuple]) -> list[tuple]:
    """matches: (match_id, start_date, team1, team2, winner, result), any order.
    Returns (match_id, team, elo_before, matches_rated_before) for both teams of every match."""
    rating = defaultdict(lambda: START)
    played = defaultdict(int)
    out = []
    by_date = defaultdict(list)
    for m in matches:
        by_date[m[1]].append(m)
    for day in sorted(by_date):
        todays = sorted(by_date[day])
        for mid, _, t1, t2, _, _ in todays:
            out += [(mid, t1, rating[t1], played[t1]), (mid, t2, rating[t2], played[t2])]
        updates = defaultdict(float)
        for _, _, t1, t2, winner, result in todays:
            if result == "tie":
                score = 0.5
            elif winner == t1:
                score = 1.0
            elif winner == t2:
                score = 0.0
            else:
                continue  # no result
            delta = K * (score - expected(rating[t1], rating[t2]))
            updates[t1] += delta
            updates[t2] -= delta
            played[t1] += 1
            played[t2] += 1
        for team, delta in updates.items():
            rating[team] += delta
    return sorted(out)
