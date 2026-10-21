"""Flatten Cricsheet match JSON (inside the bronze zip) into Parquet parts.

Keys follow Ari Adaikalam's convention so tracking joins on (match_id, innings, delivery_number):
delivery_number counts every delivery in the innings (wides and no-balls included) from 1.
"""

import json
import zipfile
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

BATCH_MATCHES = 1500

BALL_SCHEMA = pa.schema([
    ("match_id", pa.string()), ("innings", pa.int16()), ("batting_team", pa.string()),
    ("is_super_over", pa.bool_()), ("delivery_number", pa.int32()), ("over", pa.int16()),
    ("ball_in_over", pa.int16()), ("is_legal", pa.bool_()),
    ("batter", pa.string()), ("batter_id", pa.string()),
    ("non_striker", pa.string()), ("non_striker_id", pa.string()),
    ("bowler", pa.string()), ("bowler_id", pa.string()),
    ("runs_batter", pa.int16()), ("runs_extras", pa.int16()), ("runs_total", pa.int16()),
    ("runs_non_boundary", pa.bool_()),
    ("wides", pa.int16()), ("noballs", pa.int16()), ("byes", pa.int16()),
    ("legbyes", pa.int16()), ("penalty", pa.int16()), ("wicket_count", pa.int16()),
    ("dismissal_count", pa.int16()), ("actual_delivery", pa.string()), ("legal_balls_in_over_before", pa.int16()),
    ("noball_includes_byes", pa.bool_()), ("has_review", pa.bool_()),
])

WICKET_SCHEMA = pa.schema([
    ("match_id", pa.string()), ("innings", pa.int16()), ("delivery_number", pa.int32()),
    ("wicket_number", pa.int16()), ("player_out", pa.string()), ("player_out_id", pa.string()),
    ("kind", pa.string()), ("fielders", pa.string()), ("fielder_ids", pa.string()),
    ("fielder_is_substitute", pa.bool_()), ("is_retirement", pa.bool_()),
])

INNINGS_SCHEMA = pa.schema([
    ("match_id", pa.string()), ("innings", pa.int16()), ("team", pa.string()),
    ("is_super_over", pa.bool_()), ("declared", pa.bool_()), ("forfeited", pa.bool_()),
    ("target_runs", pa.int32()), ("target_overs", pa.string()), ("target_balls", pa.int32()),
    ("penalty_runs_pre", pa.int16()), ("penalty_runs_post", pa.int16()),
    ("miscounted_overs", pa.string()), ("absent_hurt", pa.string()),
])

MATCH_SCHEMA = pa.schema([
    ("match_id", pa.string()), ("data_version", pa.string()), ("created", pa.string()),
    ("revision", pa.int16()), ("match_type", pa.string()), ("match_type_number", pa.int32()),
    ("gender", pa.string()), ("team_type", pa.string()), ("event_name", pa.string()),
    ("event_match_number", pa.int32()), ("event_stage", pa.string()), ("season", pa.string()),
    ("start_date", pa.date32()), ("end_date", pa.date32()), ("n_dates", pa.int16()),
    ("venue", pa.string()), ("city", pa.string()), ("team1", pa.string()), ("team2", pa.string()),
    ("toss_winner", pa.string()), ("toss_decision", pa.string()), ("winner", pa.string()),
    ("win_by_runs", pa.int16()), ("win_by_wickets", pa.int16()), ("win_by_innings", pa.int16()),
    ("result", pa.string()), ("method", pa.string()), ("scheduled_overs", pa.int16()),
    ("balls_per_over", pa.int16()), ("player_of_match", pa.string()), ("umpires", pa.string()),
    ("tv_umpires", pa.string()), ("match_referees", pa.string()), ("n_innings", pa.int16()),
    ("cricsheet_missing", pa.string()), ("winner_after_tiebreak", pa.string()), ("tiebreak_method", pa.string()),
    ("season_year", pa.int16()), ("supersubs", pa.string()), ("scheduled_overs_source", pa.int16()),
])

PLAYER_SCHEMA = pa.schema([
    ("match_id", pa.string()), ("team", pa.string()), ("player", pa.string()), ("player_id", pa.string()),
])

POWERPLAY_SCHEMA = pa.schema([
    ("match_id", pa.string()), ("innings", pa.int16()), ("powerplay_number", pa.int16()),
    ("type", pa.string()), ("from_ball", pa.string()), ("to_ball", pa.string()),
])

REVIEW_SCHEMA = pa.schema([
    ("match_id", pa.string()), ("innings", pa.int16()), ("delivery_number", pa.int32()),
    ("review_by", pa.string()), ("umpire", pa.string()), ("umpire_id", pa.string()),
    ("batter", pa.string()), ("batter_id", pa.string()), ("decision", pa.string()),
    ("umpires_call", pa.bool_()), ("review_type", pa.string()),
])

REPLACEMENT_SCHEMA = pa.schema([
    ("match_id", pa.string()), ("innings", pa.int16()), ("delivery_number", pa.int32()),
    ("replacement_number", pa.int16()), ("kind", pa.string()), ("role", pa.string()),
    ("team", pa.string()), ("player_in", pa.string()), ("player_in_id", pa.string()),
    ("player_out", pa.string()), ("player_out_id", pa.string()), ("reason", pa.string()),
])

SCHEMAS = {"balls": BALL_SCHEMA, "wickets": WICKET_SCHEMA, "innings": INNINGS_SCHEMA,
           "matches": MATCH_SCHEMA, "match_players": PLAYER_SCHEMA, "powerplays": POWERPLAY_SCHEMA,
           "reviews": REVIEW_SCHEMA, "replacements": REPLACEMENT_SCHEMA}

RETIREMENTS = {"retired hurt", "retired not out"}

# Formats whose scheduled length is fixed by the format itself. Cricsheet's `info.overs` is wrong for some of
# them (19 T20 matches say 50, five of them in scope: the chase target of those matches reads "in 20 overs"),
# so the format wins and the value Cricsheet wrote is kept in `scheduled_overs_source`.
FORMAT_OVERS = {"T20": 20, "IT20": 20}


def scheduled_overs_for(match_type, overs):
    """Scheduled overs per side: Cricsheet's `overs`, except in a format with a fixed length."""
    fixed = FORMAT_OVERS.get(match_type)
    return fixed if fixed is not None and overs is not None else overs


def overs_to_balls(overs, balls_per_over: int = 6) -> int | None:
    """Cricket notation: 37.4 overs = 37 overs and 4 balls = 226 balls (not 37.4 * 6)."""
    if overs is None:
        return None
    whole, _, part = str(overs).partition(".")
    return int(whole) * balls_per_over + (int(part) if part else 0)


def _join(values) -> str | None:
    return "|".join(values) if values else None


def _parse_match(match_id: str, doc: dict, out: dict) -> None:
    info = doc["info"]
    registry = info.get("registry", {}).get("people", {})
    rid = registry.get
    event = info.get("event", {})
    outcome = info.get("outcome", {})
    by = outcome.get("by", {})
    officials = info.get("officials", {})
    dates = sorted(date.fromisoformat(str(d)) for d in info["dates"])
    teams = info.get("teams", [None, None])
    out["matches"].append({
        "match_id": match_id, "data_version": doc["meta"].get("data_version"),
        "created": doc["meta"].get("created"), "revision": doc["meta"].get("revision"),
        "match_type": info.get("match_type"), "match_type_number": info.get("match_type_number"),
        "gender": info.get("gender"), "team_type": info.get("team_type"),
        "event_name": event.get("name"), "event_match_number": event.get("match_number")
        if isinstance(event.get("match_number"), int) else None,
        "event_stage": event.get("stage"), "season": str(info.get("season")),
        "start_date": dates[0], "end_date": dates[-1], "n_dates": len(dates),
        "venue": info.get("venue"), "city": info.get("city"),
        "team1": teams[0], "team2": teams[1] if len(teams) > 1 else None,
        "toss_winner": info.get("toss", {}).get("winner"), "toss_decision": info.get("toss", {}).get("decision"),
        "winner": outcome.get("winner"), "win_by_runs": by.get("runs"), "win_by_wickets": by.get("wickets"),
        "win_by_innings": by.get("innings"), "result": outcome.get("result"), "method": outcome.get("method"),
        "scheduled_overs": scheduled_overs_for(info.get("match_type"), info.get("overs")),
        "balls_per_over": info.get("balls_per_over"),
        "player_of_match": _join(info.get("player_of_match")), "umpires": _join(officials.get("umpires")),
        "tv_umpires": _join(officials.get("tv_umpires")), "match_referees": _join(officials.get("match_referees")),
        "n_innings": len(doc.get("innings", [])),
        "cricsheet_missing": json.dumps(info["missing"], sort_keys=True) if info.get("missing") else None,
        # a regulation tie keeps winner NULL; the super over / bowl-out winner goes here
        "winner_after_tiebreak": outcome.get("winner") or outcome.get("eliminator") or outcome.get("bowl_out"),
        "tiebreak_method": "super_over" if "eliminator" in outcome else "bowl_out" if "bowl_out" in outcome else None,
        "season_year": dates[0].year,
        "supersubs": json.dumps(info["supersubs"], sort_keys=True) if info.get("supersubs") else None,
        "scheduled_overs_source": info.get("overs"),
    })
    for team, players in sorted(info.get("players", {}).items()):
        for p in players:
            out["match_players"].append({"match_id": match_id, "team": team, "player": p, "player_id": rid(p)})

    for i, inn in enumerate(doc.get("innings", []), start=1):
        target = inn.get("target", {})
        pen = inn.get("penalty_runs", {})
        super_over = bool(inn.get("super_over", False))
        out["innings"].append({
            "match_id": match_id, "innings": i, "team": inn.get("team"), "is_super_over": super_over,
            "declared": bool(inn.get("declared", False)), "forfeited": bool(inn.get("forfeited", False)),
            "target_runs": target.get("runs"),
            "target_overs": str(target["overs"]) if target.get("overs") is not None else None,
            "target_balls": overs_to_balls(target.get("overs"), info.get("balls_per_over", 6)),
            "penalty_runs_pre": pen.get("pre"), "penalty_runs_post": pen.get("post"),
            "miscounted_overs": json.dumps(inn["miscounted_overs"], sort_keys=True) if inn.get("miscounted_overs") else None,
            "absent_hurt": _join(inn.get("absent_hurt")),
        })
        for n, pp in enumerate(inn.get("powerplays", []), start=1):
            out["powerplays"].append({"match_id": match_id, "innings": i, "powerplay_number": n, "type": pp.get("type"),
                                      "from_ball": str(pp.get("from")), "to_ball": str(pp.get("to"))})
        delivery_number = 0
        for over in inn.get("overs", []):
            legal_before = 0
            for ball_in_over, d in enumerate(over["deliveries"], start=1):
                delivery_number += 1
                runs, extras = d["runs"], d.get("extras", {})
                wickets = d.get("wickets", [])
                is_legal = "wides" not in extras and "noballs" not in extras
                out["balls"].append({
                    "match_id": match_id, "innings": i, "batting_team": inn.get("team"),
                    "is_super_over": super_over, "delivery_number": delivery_number,
                    "over": over["over"], "ball_in_over": ball_in_over,
                    "is_legal": is_legal,
                    "batter": d["batter"], "batter_id": rid(d["batter"]),
                    "non_striker": d.get("non_striker"), "non_striker_id": rid(d.get("non_striker")),
                    "bowler": d["bowler"], "bowler_id": rid(d["bowler"]),
                    "runs_batter": runs["batter"], "runs_extras": runs["extras"], "runs_total": runs["total"],
                    "runs_non_boundary": bool(runs.get("non_boundary", False)),
                    "wides": extras.get("wides", 0), "noballs": extras.get("noballs", 0),
                    "byes": extras.get("byes", 0), "legbyes": extras.get("legbyes", 0),
                    "penalty": extras.get("penalty", 0), "wicket_count": len(wickets),
                    "dismissal_count": sum(1 for w in wickets if w["kind"] not in RETIREMENTS),
                    # official over.ball label (wides/no-balls repeat the number of the ball that replaces them)
                    "actual_delivery": f"{over['over']}.{legal_before + 1}",
                    "legal_balls_in_over_before": legal_before,
                    # before late 2017 Cricsheet folded byes off a no-ball into the no-ball count
                    "noball_includes_byes": extras.get("noballs", 0) > 1 and runs["batter"] == 0,
                    "has_review": "review" in d,
                })
                if is_legal:
                    legal_before += 1
                if "review" in d:
                    rv = d["review"]
                    out["reviews"].append({
                        "match_id": match_id, "innings": i, "delivery_number": delivery_number,
                        "review_by": rv.get("by"), "umpire": rv.get("umpire"), "umpire_id": rid(rv.get("umpire")),
                        "batter": rv.get("batter"), "batter_id": rid(rv.get("batter")), "decision": rv.get("decision"),
                        "umpires_call": rv.get("umpires_call"), "review_type": rv.get("type"),
                    })
                rep_n = 0
                for kind in ("match", "role"):
                    for rep in d.get("replacements", {}).get(kind, []):
                        rep_n += 1
                        out["replacements"].append({
                            "match_id": match_id, "innings": i, "delivery_number": delivery_number,
                            "replacement_number": rep_n, "kind": kind, "role": rep.get("role"), "team": rep.get("team"),
                            "player_in": rep.get("in"), "player_in_id": rid(rep.get("in")),
                            "player_out": rep.get("out"), "player_out_id": rid(rep.get("out")), "reason": rep.get("reason"),
                        })
                for w_num, w in enumerate(wickets, start=1):
                    fielders = w.get("fielders", [])
                    out["wickets"].append({
                        "match_id": match_id, "innings": i, "delivery_number": delivery_number,
                        "wicket_number": w_num, "player_out": w["player_out"], "player_out_id": rid(w["player_out"]),
                        "kind": w["kind"], "fielders": _join([f.get("name", "") for f in fielders]),
                        "fielder_ids": _join([rid(f.get("name"), "") or "" for f in fielders]),
                        "fielder_is_substitute": any(f.get("substitute", False) for f in fielders),
                        "is_retirement": w["kind"] in RETIREMENTS,
                    })


def iter_match_info(zip_path: Path):
    """Yield (match_id, info dict) straight from the bronze zip, sorted by file name.

    For ingest steps that must not depend on a silver build (e.g. planning weather requests).
    """
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(n for n in zf.namelist() if n.endswith(".json")):
            yield Path(name).stem, json.loads(zf.read(name))["info"]


def parse_zip(zip_path: Path, staging: Path) -> dict[str, list[Path]]:
    """Write Parquet parts per table into staging; returns {table: [part paths]}."""
    staging.mkdir(parents=True, exist_ok=True)
    parts = {t: [] for t in SCHEMAS}

    def flush(out, n):
        for table, rows in out.items():
            if rows:
                path = staging / f"{table}-{n:04d}.parquet"
                pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMAS[table]), path)
                parts[table].append(path)

    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".json"))
        out = {t: [] for t in SCHEMAS}
        for k, name in enumerate(names, start=1):
            _parse_match(Path(name).stem, json.loads(zf.read(name)), out)
            if k % BATCH_MATCHES == 0:
                flush(out, k // BATCH_MATCHES)
                out = {t: [] for t in SCHEMAS}
        flush(out, len(names) // BATCH_MATCHES + 1)
    print(f"  parsed {len(names)} Cricsheet matches")
    return parts
