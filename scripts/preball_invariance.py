"""Pre-ball invariance on PRODUCTION rows (not the committed fixture): a random sample of in-scope matches is
re-parsed from the locked Cricsheet zip, cut off at random points as if the match were live, and recomputed with
the pipeline's own parser and state SQL. No `pre_ball` column of fact_delivery or int_ball_state_pre, and no
derived model feature, may change; `hindsight` columns are expected to. tests/test_leakage.py does this on the
10-match fixture; scripts/full_check.sh runs this on the real data.

    uv run python scripts/preball_invariance.py [--matches 200] [--cuts 20] [--seed 20261023]

Reads the data root in CRICKET_SIM_DATA (default ./data). Exit code 1 if any pre-ball value changed.
"""

import argparse
import copy
import json
import random
import re
import sys
import time
import zipfile

import duckdb
import pyarrow as pa

from cricket_sim import ball_layers, bronze, contracts
from cricket_sim.cricsheet_parse import SCHEMAS, _parse_match
from cricket_sim.models import dataset
from cricket_sim.paths import GOLD, snapshot_dir


def _timing(table: str) -> dict:
    c = contracts.get(table)
    return {col["name"]: c.column_props(col).get("timingClass") for col in c.columns if not col.get("primaryKey")}


def _cut(doc: dict, innings_no: int, deliveries: int) -> dict:
    """The match as it looked after `deliveries` balls of innings `innings_no`: later innings dropped, the result
    and everything only known at the end of the innings removed."""
    d = copy.deepcopy(doc)
    d["info"]["outcome"] = {}
    d["info"].pop("player_of_match", None)
    innings = d.get("innings", [])[:innings_no]
    current = innings[innings_no - 1]
    for key in ("target", "declared", "absent_hurt", "miscounted_overs"):
        current.pop(key, None)
    current.get("penalty_runs", {}).pop("post", None)
    seen, overs = 0, []
    for over in current.get("overs", []):
        keep = [dl for k, dl in enumerate(over["deliveries"], start=seen + 1) if k <= deliveries]
        seen += len(over["deliveries"])
        if keep:
            overs.append({**over, "deliveries": keep})
        if seen >= deliveries:
            break
    current["overs"], d["innings"] = overs, innings
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=200)
    ap.add_argument("--cuts", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261023)
    args = ap.parse_args()
    zip_path = snapshot_dir("cricsheet", bronze.read_lock()["cricsheet"]) / "all_json.zip"

    state_t, fd_t = _timing("int_ball_state_pre"), _timing("fact_delivery")
    columns = sorted((set(state_t) | set(fd_t)) - {c for c, t in fd_t.items() if t not in ("pre_ball", "hindsight")})
    derived = {n: s for n, s in dataset.DERIVED.items() if set(re.findall(r"\b([a-z]\w*)\.", s)) == {"f"}}
    pre_ball = {c for c in columns if state_t.get(c, fd_t.get(c)) == "pre_ball"} | set(derived)

    con = duckdb.connect()
    ids = [r[0] for r in con.execute(f"SELECT match_id FROM read_parquet('{GOLD / 'dim_match.parquet'}') ORDER BY 1").fetchall()]
    random.seed(args.seed)
    sample = sorted(random.sample(ids, min(args.matches, len(ids))))
    changed, compared, cuts, t0 = {}, 0, 0, time.time()
    with zipfile.ZipFile(zip_path) as zf:
        for mid in sample:
            doc = json.loads(zf.read(f"{mid}.json"))
            out = {t: [] for t in SCHEMAS}
            _parse_match(mid, doc, out)
            flat = [(i, k) for i, inn in enumerate(doc.get("innings", []), start=1)
                    for k in range(1, sum(len(o["deliveries"]) for o in inn.get("overs", [])) + 1)]
            for v, p in enumerate(sorted(random.sample(range(len(flat)), min(args.cuts, len(flat))))):
                _parse_match(f"{mid}#{v}", _cut(doc, *flat[p]), out)
                cuts += 1
            for t, rows in out.items():
                con.register(f"_{t}", pa.Table.from_pylist(rows, schema=SCHEMAS[t]))
                con.execute(f"CREATE OR REPLACE TABLE {t} AS SELECT * FROM _{t}")
                con.unregister(f"_{t}")
            con.execute(f"CREATE OR REPLACE TABLE int_innings_state AS {ball_layers.INNINGS_SQL}")
            con.execute(f"CREATE OR REPLACE TABLE st AS {ball_layers.STATE_PRE_SQL}")
            have = {r[0] for r in con.execute("DESCRIBE st").fetchall()}
            pick = ", ".join(f"s.{c}" if c in have else f"b.{c}" for c in columns)
            extra = "".join(f", {sql} AS {name}" for name, sql in derived.items())
            con.execute(f"""CREATE OR REPLACE TABLE x AS WITH f AS (SELECT s.match_id, s.innings, s.delivery_number, {pick}
                FROM st s JOIN balls b USING (match_id, innings, delivery_number)) SELECT f.*{extra} FROM f""")
            cols = columns + list(derived)
            diffs = ", ".join(f"count(*) FILTER (a.{c} IS DISTINCT FROM v.{c})" for c in cols)
            row = con.execute(f"""WITH v AS (SELECT *, split_part(match_id, '#', 1) AS base FROM x WHERE match_id LIKE '%#%'),
                a AS (SELECT * FROM x WHERE match_id NOT LIKE '%#%')
                SELECT count(*), {diffs} FROM v JOIN a ON a.match_id = v.base AND a.innings = v.innings
                    AND a.delivery_number = v.delivery_number""").fetchone()
            compared += row[0]
            for c, n in zip(cols, row[1:]):
                if n:
                    changed[c] = changed.get(c, 0) + n
    leaked = {c: n for c, n in changed.items() if c in pre_ball}
    print(f"pre-ball invariance: {len(sample)} matches, {cuts} cuts, {compared:,} ball-rows compared, "
          f"{len(pre_ball)} pre-ball columns, {time.time() - t0:.0f}s")
    print("hindsight columns that changed (expected):", {c: n for c, n in changed.items() if c not in pre_ball} or "none")
    if leaked:
        print("PRE-BALL COLUMNS CHANGED (leak):", leaked, file=sys.stderr)
        return 1
    print("pre-ball columns that changed: NONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
